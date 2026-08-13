# 3. Scheduler 核心实现

> 源码：vllm/vllm/v1/core/sched/scheduler.py（约2800行）

---

## 目录

1. [__init__ 初始化](#1-__init__-初始化)
2. [schedule() 总览](#2-schedule-总览)
3. [第一阶段：调度 running 队列](#3-第一阶段调度-running-队列)
4. [第二阶段：调度 waiting 队列](#4-第二阶段调度-waiting-队列)
5. [Prefix Cache 查找](#5-prefix-cache-查找)
6. [Preemption 机制](#6-preemption-机制)
7. [KV Connector 集成（P/D分离）](#7-kv-connector-集成pd分离)
8. [Multimodal Encoder 调度](#8-multimodal-encoder-调度)
9. [SchedulerOutput 构建与后处理](#9-scheduleroutput-构建与后处理)
10. [update_from_output()](#10-update_from_output)
11. [请求生命周期管理](#11-请求生命周期管理)
12. [Mamba SSM 块对齐拆分](#12-mamba-ssm-块对齐拆分)
13. [reset_prefix_cache](#13-reset_prefix_cache)
14. [延迟块释放与deferred_fences](#14-延迟块释放与deferred_fences)
15. [其他辅助方法](#15-其他辅助方法)

---

## 1. __init__ 初始化

`scheduler.py:70`，约280行。按功能模块说明：

### 1.1 配置引用
```python
self.vllm_config = vllm_config
self.scheduler_config = vllm_config.scheduler_config
self.cache_config = vllm_config.cache_config
self.lora_config = vllm_config.lora_config
self.kv_cache_config = kv_cache_config
self.parallel_config = vllm_config.parallel_config
self.observability_config = vllm_config.observability_config
```

### 1.2 调度约束（budget参数）
```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
self.max_num_scheduled_tokens = (
    self.scheduler_config.max_num_scheduled_tokens
    if self.scheduler_config.max_num_scheduled_tokens is not None
    else self.scheduler_config.max_num_batched_tokens
)
self.max_model_len = vllm_config.model_config.max_model_len
```

- `max_num_seqs`：最大并发running请求数
- `max_num_scheduled_tokens`：单轮总token budget，优先使用显式配置，fallback到`max_num_batched_tokens`
- `max_model_len`：单请求最大上下文长度（prompt+output），由模型位置编码维度决定

### 1.3 Diffusion模型适配
```python
self.num_sampled_tokens_per_step = (
    1 if not vllm_config.model_config.is_diffusion else 0
)
```
自回归AR模型每步采样1个新token；Diffusion模型denoising步不采样新token，该值为0。

### 1.4 KV Connector初始化
```python
self.connector = None
kv_transfer_config = self.vllm_config.kv_transfer_config
if kv_transfer_config is not None:
    self.connector = KVConnectorFactory.create_connector(
        config=self.vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=self.kv_cache_config,
    )
    self.recompute_kv_load_failures = (kv_load_failure_policy == "recompute")
    multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
    self.defer_block_free = multiple_inflight_batches and kv_transfer_config.is_kv_consumer
```
- `recompute_kv_load_failures`：远程KV加载失败时是否回退到本地重算
- `defer_block_free`：多在飞批次 + KV consumer场景下延迟block释放，避免异步读写竞态

EC Connector（encoder cache transfer）：
```python
self.ec_connector = None
if self.vllm_config.ec_transfer_config is not None:
    self.ec_connector = ECConnectorFactory.create_connector(
        config=self.vllm_config, role=ECConnectorRole.SCHEDULER)
```

### 1.5 请求容器
```python
self.requests: dict[str, Request] = {}
self.policy = SchedulingPolicy(self.scheduler_config.policy)
self.waiting = create_request_queue(self.policy)
self.skipped_waiting = create_request_queue(self.policy)
self.running: list[Request] = []
self.finished_req_ids: set[str] = set()
self.reset_preempted_req_ids: set[str] = set()
self.num_waiting_for_streaming_input: int = 0
```

### 1.6 KV Connector异步状态
```python
self.finished_recving_kv_req_ids: set[str] = set()
self.failed_recving_kv_req_ids: set[str] = set()
self.grammar_compile_error_reqs: set[str] = set()
```

### 1.7 Encoder/多模态
```python
supports_mm_inputs = mm_registry.supports_multimodal_inputs(vllm_config.model_config)
mm_budget = MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None

self.max_num_encoder_input_tokens = mm_budget.encoder_compute_budget if mm_budget else 0
encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
# EncoderCacheManager或EncoderDecoderCacheManager
self.encoder_cache_manager = ...
```

### 1.8 Speculative decoding配置
```python
self.num_spec_tokens = vllm_config.num_speculative_tokens
self.use_eagle = False
self.num_lookahead_tokens = 0
self.dynamic_sd_lookup: list[int] | None = None
if speculative_config is not None:
    if speculative_config.num_speculative_tokens_per_batch_size:
        self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(...)
    if speculative_config.use_eagle():
        self.use_eagle = True
        self.num_lookahead_tokens = self.num_spec_tokens
    if speculative_config.uses_draft_model():
        self.num_lookahead_tokens = self.num_spec_tokens
    if speculative_config.use_dflash():
        self.num_lookahead_tokens = self.num_spec_tokens + 1  # in-fill style需要额外query
    if speculative_config.use_dspark():
        self.num_lookahead_tokens = self.num_spec_tokens      # anchor即首预测位
```
`num_lookahead_tokens`决定allocate_slots时为将来draft tokens预留的KV block数量。

### 1.9 KVCacheManager
```python
self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_in_flight_tokens=vllm_config.max_in_flight_tokens,
    enable_caching=self.cache_config.enable_prefix_caching,
    use_eagle=self.use_eagle,
    log_stats=self.log_stats,
    enable_kv_cache_events=self.enable_kv_cache_events,
    dcp_world_size=self.dcp_world_size,
    pcp_world_size=1,
    scheduler_block_size=self.block_size,
    hash_block_size=hash_block_size,
    metrics_collector=self.kv_metrics_collector,
    watermark=self.scheduler_config.watermark,
)
if self.connector is not None:
    self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
```

### 1.10 PP/async/step控制
```python
self.use_pp = self.parallel_config.pipeline_parallel_size > 1
self.use_v2_model_runner = vllm_config.use_v2_model_runner
self.current_step = 0
self.prefill_capacity_bound = False
self.scheduler_reserve_full_isl = self.scheduler_config.scheduler_reserve_full_isl
```

### 1.11 Mamba/SSM
```python
self.has_mamba_layers = kv_cache_config.has_mamba_layers
self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
self._skip_zero_block_ids: set[int] = set()
self.need_mamba_block_aligned_split = (
    self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align")
self.mamba_partial_cache_hit = (
    self.need_mamba_block_aligned_split and self.hash_block_size < self.block_size)
```

### 1.12 延迟释放fence
```python
self.sched_step_seq = 0
self.processed_step_seq = 0
self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()
```

### 1.13 Routed experts（MoE专家路由返回）
```python
self.enable_return_routed_experts = vllm_config.model_config.enable_return_routed_experts
if self.enable_return_routed_experts:
    self.routed_experts_mgr = RoutedExpertsManager(...)
    self._re_block_ids: dict[str, list[int]] = {}
```

### 1.14 其他
```python
self._pause_state: PauseState = PauseState.UNPAUSED
self._inflight_prefills: set[Request] = set()
```

---

## 2. schedule() 总览

`scheduler.py:427`。函数开头的NOTE(woosuk)是理解V1调度器的核心：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and num_tokens_with_spec.
# num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its num_tokens_with_spec.
# This is general enough to cover chunked prefills, prefix caching,
# speculative decoding, and the "jump decoding" optimization in the future.
```

### 2.1 局部变量初始化
```python
self.current_step += 1

scheduled_new_reqs: list[Request] = []
scheduled_resumed_reqs: list[Request] = []
scheduled_running_reqs: list[Request] = []
preempted_reqs: list[Request] = []

req_to_new_blocks: dict[str, KVCacheBlocks] = {}
num_scheduled_tokens: dict[str, int] = {}
token_budget = self.max_num_scheduled_tokens
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0

scheduled_encoder_inputs: dict[str, list[int]] = {}
encoder_compute_budget = self.max_num_encoder_input_tokens
scheduled_spec_decode_tokens: dict[str, list[int]] = {}
prefill_scheduled = False
scheduled_timestamp = time.monotonic()
self.kv_cache_manager.new_step_starts()
```

### 2.2 DP prefill throttle
```python
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```
- DP rank间prefill步对齐：非cadence对齐步defer新prefill，除非rank已饱和（`prefill_capacity_bound`）
- 已在decode的running请求存在时，prefill chunk被跳过，保证decode进度

### 2.3 两阶段调度
```
第一阶段：遍历self.running，为每个running请求尝试分配新token和KV blocks
第二阶段：仅当第一阶段无preempt且UNPAUSED时，遍历waiting/skipped_waiting接入新请求
```
调度顺序是"running优先"——优先服务已持有KV资源的请求，避免decode被prefill阻塞导致TTFT/TTFT抖动。

---

## 3. 第一阶段：调度 running 队列

### 3.1 主循环
```python
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
```

### 3.2 跳过条件
```python
# async调度：当已确定到达max_tokens时跳过多余调度步
if (request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
        >= request.num_prompt_tokens + request.max_tokens):
    req_index += 1
    continue

# PP+v2+async：请求距离上次decode未满pp_size步，跳过
if self.current_step < request.next_decode_eligible_step:
    req_index += 1
    continue

# DP平衡：prefill chunk在defer_prefills步跳过
if defer_prefills and request.is_prefill_chunk:
    req_index += 1
    continue
```

### 3.3 计算num_new_tokens
```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)

# 长prefill分块阈值
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold

# token budget约束
num_new_tokens = min(num_new_tokens, token_budget)

# max_model_len约束（spec decode时可能越界）
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step,
)
```

### 3.4 Encoder输入调度
```python
encoder_inputs_to_schedule = None
external_load_encoder_input: list[int] = []
if request.has_encoder_inputs:
    (encoder_inputs_to_schedule, num_new_tokens,
     new_encoder_compute_budget, external_load_encoder_input) = \
        self._try_schedule_encoder_inputs(
            request, request.num_computed_tokens, num_new_tokens,
            encoder_compute_budget,
            shift_computed_tokens=1 if self.use_eagle else 0,
        )
```

### 3.5 Mamba块对齐裁剪
```python
if self.need_mamba_block_aligned_split:
    num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
```

### 3.6 num_new_tokens==0处理
```python
if num_new_tokens == 0:
    # 原因：PP在飞、async到长度上限、encoder预算耗尽、Mamba无法凑齐一个对齐块
    # continue而非break，允许后续请求继续调度
    req_index += 1
    continue
```

### 3.7 KV block分配与preemption循环
```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request,
        num_new_tokens,
        num_lookahead_tokens=self.num_lookahead_tokens,
    )
    if new_blocks is not None:
        break

    # 分配失败，需要preempt
    if self.policy == SchedulingPolicy.PRIORITY:
        # Priority模式：选priority值最大（最低优先级）、arrival_time最晚的请求
        preempted_req = max(self.running,
                            key=lambda r: (r.priority, r.arrival_time))
        self.running.remove(preempted_req)
        if preempted_req in scheduled_running_reqs:
            # 已在本轮scheduled列表中的，退还预算
            pid = preempted_req.request_id
            scheduled_running_reqs.remove(preempted_req)
            token_budget += num_scheduled_tokens.pop(pid)
            req_to_new_blocks.pop(pid)
            scheduled_spec_decode_tokens.pop(pid, None)
            pei = scheduled_encoder_inputs.pop(pid, None)
            if pei:
                num_embeds_to_restore = sum(
                    preempted_req.get_num_encoder_embeds(i) for i in pei)
                encoder_compute_budget += num_embeds_to_restore
            req_index -= 1
    else:
        # FCFS模式：pop末尾（最近入队的）
        preempted_req = self.running.pop()

    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
    if preempted_req == request:
        break  # 自己被抢占，无法继续
```

### 3.8 分配成功后记录
```python
scheduled_running_reqs.append(request)
prefill_scheduled |= request.is_prefill_chunk
request_id = request.request_id
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
req_index += 1
```

### 3.9 Spec tokens切片
```python
if request.spec_token_ids:
    num_scheduled_spec_tokens = (
        num_new_tokens + request.num_computed_tokens
        - request.num_tokens - request.num_output_placeholders
    )
    if num_scheduled_spec_tokens > 0:
        spec_token_ids = request.spec_token_ids
        if len(spec_token_ids) > num_scheduled_spec_tokens:
            spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
        scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
    request.spec_token_ids = []  # 清空，等待update_draft_token_ids填充下一轮
```

### 3.10 Encoder alloc
```python
if encoder_inputs_to_schedule:
    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
    for i in encoder_inputs_to_schedule:
        self.encoder_cache_manager.allocate(request, i)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_alloc(request, i)
    encoder_compute_budget = new_encoder_compute_budget
if external_load_encoder_input:
    for i in external_load_encoder_input:
        self.encoder_cache_manager.allocate(request, i)
        if self.ec_connector is not None:
            self.ec_connector.update_state_after_alloc(request, i)
```

---

## 4. 第二阶段：调度 waiting 队列

仅当`not preempted_reqs and self._pause_state == PauseState.UNPAUSED`时进入。

### 4.1 外层循环
```python
step_skipped_waiting = create_request_queue(self.policy)
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    # 并发数上限检查（含streaming slot占用）
    num_running = len(self.running) + self.num_waiting_for_streaming_input
    if num_running >= self.max_num_running_reqs:
        break

    request_queue = self._select_waiting_queue_for_scheduling()
    request = request_queue.peek_request()
    request_id = request.request_id
```

### 4.2 阻塞状态处理
```python
if self._is_blocked_waiting_status(request.status):
    if not self._try_promote_blocked_waiting_request(request):
        request_queue.pop_request()
        step_skipped_waiting.prepend_request(request)
        continue
```
`_is_blocked_waiting_status`识别`WAITING_FOR_REMOTE_KVS`、`WAITING_FOR_STREAMING_REQ`等阻塞状态；`_try_promote_blocked_waiting_request`检查KV是否已接收完成、streaming input是否就绪。

### 4.3 LoRA slot约束
```python
if self.lora_config and request.lora_request and (
    len(scheduled_loras) == self.lora_config.max_loras
    and request.lora_request.lora_int_id not in scheduled_loras
):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

### 4.4 Prefix cache查找（首次调度）
```python
num_computed_tokens = 0
new_computed_blocks = ...
num_new_local_computed_tokens = 0
load_kv_async = False
num_external_computed_tokens = 0

if request.num_computed_tokens == 0:
    did_prefix_cache_lookup = True
    # 本地prefix cache
    if self.connector is not None:
        (new_computed_blocks, num_new_local_computed_tokens,
         request.shared_prefix_boundary, hit_diverged) = \
            self.kv_cache_manager.get_computed_blocks_for_connector(request)
    else:
        (new_computed_blocks, num_new_local_computed_tokens,
         request.shared_prefix_boundary) = \
            self.kv_cache_manager.get_computed_blocks(request)

    # 远程prefix cache（KV Connector）
    if self.connector is not None:
        ext_tokens, load_kv_async = \
            self.connector.get_num_new_matched_tokens(request, block_aligned_local)
        # 处理本地partial tail与远程命中的取舍（见第5节）
        ...

    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
else:
    # 恢复的请求（如async KV接收完成），num_computed_tokens已被设置
    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
    num_new_local_computed_tokens = 0
    num_computed_tokens = request.num_computed_tokens
```

### 4.5 Encoder可用性检查（EC Connector）
```python
if self.ec_connector is not None and request.mm_features and not \
        self.ec_connector.ensure_cache_available(request, num_computed_tokens):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
```

### 4.6 Prefill stats
```python
if request.prefill_stats and request.num_preemptions <= 0:
    request.prefill_stats.set(
        num_prompt_tokens=request.num_prompt_tokens,
        num_local_cached_tokens=num_new_local_computed_tokens,
        num_external_cached_tokens=num_external_computed_tokens,
    )
```

### 4.7 计算num_new_tokens
```python
encoder_inputs_to_schedule = None
external_load_encoder_input = []
new_encoder_compute_budget = encoder_compute_budget
pad_spec_decode = False

if load_kv_async:
    num_new_tokens = 0  # 异步拉KV阶段不做本地compute
elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
    break  # DP throttle，且仍在prefill阶段，break
else:
    num_new_tokens = request.num_tokens - num_computed_tokens

    # Spec decode padding：新decode请求pad到统一spec长度，保持cudagraph capture
    if ((self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
        and self.num_sampled_tokens_per_step > 0
        and num_new_tokens == 1
        and (scheduled_running_reqs and not prefill_scheduled)):
        num_new_tokens = 1 + self.num_spec_tokens
        if num_new_tokens > token_budget or \
           num_computed_tokens + num_new_tokens > self.max_model_len:
            break
        pad_spec_decode = True

    if 0 < threshold < num_new_tokens:
        num_new_tokens = threshold

    if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
        break  # chunked prefill未启用且放不下，整体跳过

    num_new_tokens = min(num_new_tokens, token_budget)
    assert num_new_tokens > 0

    if request.has_encoder_inputs:
        (encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget,
         external_load_encoder_input) = self._try_schedule_encoder_inputs(...)
        if num_new_tokens == 0:
            break
```

### 4.8 Mamba块对齐
```python
if self.need_mamba_block_aligned_split and not load_kv_async:
    num_new_tokens = self._mamba_block_aligned_split(
        request, num_new_tokens, num_new_local_computed_tokens,
        num_external_computed_tokens)
    if num_new_tokens == 0:
        break
```

### 4.9 Cross-attention blocks（encoder-decoder）
```python
num_encoder_tokens = 0
if self.is_encoder_decoder and request.has_encoder_inputs and encoder_inputs_to_schedule:
    num_encoder_tokens = sum(request.get_num_encoder_embeds(i) for i in encoder_inputs_to_schedule)
```

### 4.10 KV blocks分配
```python
reserved_blocks = 0
if load_kv_async:
    reserved_blocks = self._inflight_prefill_reserved_blocks()

new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
    num_encoder_tokens=num_encoder_tokens,
    full_sequence_must_fit=self.scheduler_reserve_full_isl,
    reserved_blocks=reserved_blocks,
    has_scheduled_reqs=bool(self.running),
)
if new_blocks is None:
    if request.has_encoder_inputs:
        self.encoder_cache_manager.free(request)
    break  # waiting队列分配失败直接break，不preempt running
```
注意：waiting阶段不触发preempt——running已在第一阶段优先调度，preempt running接新请求会导致running请求饥饿。

### 4.11 Connector状态更新
```python
if self.connector is not None:
    self.connector.update_state_after_alloc(
        request,
        self.kv_cache_manager.get_blocks(request_id),
        num_external_computed_tokens,
    )
    if self.connector_prefix_cache_stats is not None and connector_prefix_cache_queries != 0:
        self.connector_prefix_cache_stats.record(...)
if did_prefix_cache_lookup:
    self.kv_cache_manager.record_prefix_cache_stats(request, num_new_local_computed_tokens)
```

### 4.12 加入running或跳过队列
```python
request = request_queue.pop_request()
if load_kv_async:
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    step_skipped_waiting.prepend_request(request)
    self._inflight_prefills.add(request)
else:
    self.running.append(request)
    scheduled_new_reqs.append(request)
    prefill_scheduled |= (num_computed_tokens + num_new_tokens < request.num_tokens) or num_new_tokens > 1
    req_to_new_blocks[request_id] = new_blocks
    num_scheduled_tokens[request_id] = num_new_tokens
    token_budget -= num_new_tokens
    request.num_computed_tokens = num_computed_tokens

if self.lora_config and request.lora_request and request.lora_request.lora_int_id > 0:
    scheduled_loras.add(request.lora_request.lora_int_id)
```

### 4.13 合并step_skipped
```python
while step_skipped_waiting:
    self.skipped_waiting.prepend_request(step_skipped_waiting.pop_request())
```

---

## 5. Prefix Cache 查找

对于首次调度（`num_computed_tokens == 0`）的请求，`KVCacheManager.get_computed_blocks(_for_connector)`执行基于block hash的前缀匹配：

1. 将prompt按`hash_block_size`分块，计算每块的hash
2. 从第一块开始在全局block hash表中查找，返回最长匹配前缀
3. 返回值：
   - `new_computed_blocks`：匹配到的物理block引用
   - `num_new_local_computed_tokens`：匹配到的token数
   - `shared_prefix_boundary`：Marconi模式下多请求共享前缀的boundary位置
   - `hit_diverged`（仅connector变体）：不同attention group之间命中不一致

### KV Connector存在时的三层命中组合

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 本地块对齐命中 block_aligned_local                            │
│    = num_new_local_computed_tokens - (num_new_local % block_size)│
├─────────────────────────────────────────────────────────────────┤
│ 2. 本地partial tail（不对齐部分）                                 │
│    partial_tail = num_new_local_computed_tokens % block_size    │
├─────────────────────────────────────────────────────────────────┤
│ 3. 远程命中 ext_tokens（Connector返回）                          │
└─────────────────────────────────────────────────────────────────┘
```

处理逻辑：
- 若`partial_tail > 0`且`ext_tokens > partial_tail`：远程严格长于本地完整命中（覆盖partial tail）。truncate本地partial tail（避免CoW），`num_external_computed_tokens = ext_tokens`，`load_kv_async = True`
- 若`partial_tail > 0`且`ext_tokens <= partial_tail`：远程不超过本地，保留本地partial，不加载远程，`load_kv_async = False`
- 若`partial_tail == 0`：直接`num_external_computed_tokens = ext_tokens`；若ext_tokens>0则`load_kv_async = True`
- 若`hit_diverged`且`num_external_computed_tokens == 0`：重新调用`get_computed_blocks`（非connector变体）以收敛到所有group一致的boundary

---

## 6. Preemption 机制

### 6.1 _preempt_request

`scheduler.py:1247`：
```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    # 1. 释放KV blocks
    self._free_request_blocks(request)
    # 2. 释放encoder cache
    if request.has_encoder_inputs:
        self._free_encoder_inputs(request, free_all=True)
    # 3. 重置计算状态
    request.num_computed_tokens = 0
    request.spec_token_ids = []
    request.status = RequestStatus.WAITING
    request.num_preemptions += 1
    # 4. 标记finished（通知worker清缓存）与preempted
    self.reset_preempted_req_ids.add(request.request_id)
    self.finished_req_ids.add(request.request_id)
    # 5. 放回waiting队首
    self.waiting.prepend_request(request)
```

### 6.2 Preempt目标选择策略

| 策略 | 目标选择 | 理由 |
|------|----------|------|
| FCFS | `self.running.pop()`（末尾元素） | LIFO式：最近入队的请求prefill进度最短，抢占损失最小；避免老请求饥饿 |
| Priority | `max(self.running, key=lambda r: (r.priority, r.arrival_time))` | 选priority值最大（最低优先级）、arrival_time最晚的请求 |

### 6.3 被抢占请求为何回队首（FCFS）
- 请求已经过部分prefill计算，回队首可优先重新调度，减少重复计算（KV blocks虽释放，但block hash表中其他请求可能仍持有这些块，重入后prefix cache可能再次命中）
- 防止请求反复被抢占导致饥饿

Priority模式下prepend语义退化为add，位置由(priority, arrival_time)决定。

---

## 7. KV Connector 集成（P/D分离）

KV Connector用于Prefill-Decode disaggregation：Prefill实例与Decode实例通过网络传输KV cache。

### 7.1 异步KV加载流程
```
schedule()检测到远程ext_tokens > block_aligned_local
    │
    ▼
load_kv_async = True
allocate_slots(delay_cache_blocks=True)  # 预留blocks但不注册到cache
request.status = WAITING_FOR_REMOTE_KVS
移入skipped_waiting
_inflight_prefills.add(request)
    │
    ▼ （后台异步传输）
Connector完成后通过_update_from_kv_xfer_finished回调
    │
    ├─→ 成功：finished_recving_kv_req_ids.add(req_id)
    │       request.num_computed_tokens = 外部加载的token数
    │       request.status = WAITING（下次schedule可被接入）
    │
    └─→ 失败：failed_recving_kv_req_ids.add(req_id)
            recompute模式：request.num_computed_tokens重置，回退本地prefill
            否则：abort请求
```

### 7.2 _inflight_prefill_reserved_blocks
```python
def _inflight_prefill_reserved_blocks(self) -> int:
    return sum(self._request_remaining_blocks(r) for r in self._inflight_prefills)
```
统计正在异步KV加载的in-flight请求预留的block总数。新的async load准入时需扣除这些预留，避免预留超出空闲block导致死锁。

### 7.3 deferred block free
当`max_concurrent_batches > 1`且为KV consumer时启用：
- 一个仍在执行的batch可能正在写已释放的blocks
- Consumer Connector后续可能重新分配并写入这些block，产生WAW/WAR竞态
- 释放操作入队`deferred_frees`，fence seq为`sched_step_seq`
- `_drain_deferred_frees()`在`processed_step_seq >= fence_seq`时才真正调用`kv_cache_manager.free(blocks)`

---

## 8. Multimodal Encoder 调度

### 8.1 _try_schedule_encoder_inputs

`scheduler.py:1423`，约160行。主要逻辑：
1. 遍历request的encoder输入（mm_features列表），计算哪些index已被encoder_cache_manager缓存（相同mm_hash）
2. 对未缓存的index，估算需要的encoder embedding数（通过`request.get_num_encoder_embeds(i)`）
3. 在`encoder_compute_budget`内选择本批处理的index集合
4. 若encoder输入对应位置超出当前token window（通过`get_mm_features_in_window`判断），可能裁剪num_new_tokens
5. 返回：
   - `encoder_inputs_to_schedule`：本批要计算encoder的index列表
   - `new_num_new_tokens`：可能被裁剪后的num_new_tokens
   - `new_encoder_compute_budget`：扣除后的encoder预算
   - `external_load_encoder_input`：需要从EC Connector加载的encoder输入index

### 8.2 Eagle shift
`shift_computed_tokens=1`（Eagle模式）：因为Eagle在每个position额外需要一个head token，encoder计算位置相对num_computed_tokens偏移1。

---

## 9. SchedulerOutput 构建与后处理

### 9.1 构造NewRequestData
```python
scheduled_new_req_data: list[NewRequestData] = []
for request in scheduled_new_reqs:
    new_blocks = req_to_new_blocks[request.request_id]
    num_scheduled_tokens_req = num_scheduled_tokens[request.request_id]
    prefill_token_ids = None
    if request.num_computed_tokens + num_scheduled_tokens_req < request.num_tokens:
        # chunked prefill：只传当前chunk的token ids
        prefill_start = request.num_computed_tokens
        prefill_end = prefill_start + num_scheduled_tokens_req
        prefill_token_ids = request.all_token_ids[prefill_start:prefill_end]
    scheduled_new_req_data.append(NewRequestData.from_request(
        request, new_blocks.local_block_ids, prefill_token_ids))
```

### 9.2 构造CachedRequestData
`_make_cached_request_data(scheduled_running_reqs, scheduled_resumed_reqs, num_scheduled_tokens, req_to_new_blocks, scheduled_spec_decode_tokens)`：
- 对running和resumed请求按列表顺序构造并行数组（req_ids, new_block_ids, num_computed_tokens, num_output_tokens等）
- `resumed_req_ids`集合标记preempt后恢复的请求，这些请求的block_ids是替换语义而非追加语义
- PP>1时填充`new_token_ids`（上轮新生成的token）
- MRV1下填充`all_token_ids`（给Connector传未调度请求的token ids）

### 9.3 零填充blocks与CoW copies
```python
new_block_ids_to_zero = self._get_new_block_ids_to_zero()
kv_cache_block_copies = self.kv_cache_manager.get_pending_copies()
```
- `_get_new_block_ids_to_zero`：本轮freshly allocated的物理block id列表（排除`_skip_zero_block_ids`即被async KV写入覆盖的块）
- Worker在forward前对这些block做KV内存清零，防止NaN/脏数据污染attention
- CoW copies处理prefix cache共享块的写前复制（partial tail hit、Marconi共享前缀等场景）

### 9.4 KV Connector metadata
```python
kv_connector_metadata = self._build_kv_connector_meta(...)
```
构建Connector所需的元信息（哪些请求在加载/发送、block映射等）。

### 9.5 组装SchedulerOutput
```python
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=scheduled_new_req_data,
    scheduled_cached_reqs=scheduled_cached_req_data,
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total_num_scheduled_tokens,
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    num_common_prefix_blocks=num_common_prefix_blocks,
    finished_req_ids=finished_req_ids_to_send,
    free_encoder_mm_hashes=free_encoder_mm_hashes,
    scheduled_encoder_input_stats=...,
    preempted_req_ids=self.reset_preempted_req_ids if self.use_v2_model_runner else None,
    new_block_ids_to_zero=new_block_ids_to_zero,
    kv_cache_block_copies=kv_cache_block_copies,
    partial_tail_offloads=partial_tail_offloads,
    num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
    kv_connector_metadata=kv_connector_metadata,
    ec_connector_metadata=ec_connector_metadata,
    ec_manager_metadata=ec_manager_metadata,
    ...
)
```

### 9.6 _update_after_schedule
`scheduler.py:1271`：
```python
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    self.finished_req_ids.clear()
    self.reset_preempted_req_ids.clear()

    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        request = self.requests[req_id]
        request.num_computed_tokens += num_tokens
        if request.num_computed_tokens < request.num_tokens:
            request.status = RequestStatus.RUNNING
            request.is_prefill_chunk = True
        else:
            request.is_prefill_chunk = False

    self.sched_step_seq += 1
    self.current_step += 1
    self._drain_deferred_frees()
    # KV event publishing、stats记录等
```

---

## 10. update_from_output()

`scheduler.py:1624`，约370行。

### 10.1 处理KV Connector输出
```python
if model_runner_output.kv_connector_output is not None:
    self._update_from_kv_xfer_finished(model_runner_output.kv_connector_output)
```

### 10.2 处理invalid blocks
```python
if model_runner_output.invalid_req_id_to_block_ids is not None:
    self._handle_invalid_blocks(model_runner_output.invalid_req_id_to_block_ids)
```
异步KV加载失败/校验失败的blocks，触发重算或abort。

### 10.3 处理draft tokens（async场景）
```python
if model_runner_output.next_draft_token_ids is not None:
    self.update_draft_token_ids_in_output(
        model_runner_output.next_draft_token_ids, scheduler_output)
```

### 10.4 遍历请求更新
```python
req_ids = []
new_token_ids_list = []
for req_id, req_new_token_ids in req_id_to_new_token_ids.items():
    if req_id not in self.requests:
        continue
    request = self.requests[req_id]
    del self.requests[req_id]

    new_token_ids, stopped = self._update_request_with_output(request, req_new_token_ids)

    if stopped:
        self._handle_stopped_request(request)
        # 生成EngineCoreOutput
        ...
    else:
        self.requests[req_id] = request
        self.running.append(request)
        req_ids.append(req_id)
        new_token_ids_list.append(new_token_ids)
        # 生成EngineCoreOutput（stream delta）
        ...
```

### 10.5 _update_request_with_output
`scheduler.py:2043`：
```python
def _update_request_with_output(self, request, new_token_ids):
    request.output_token_ids.extend(new_token_ids)
    request.num_output_tokens += len(new_token_ids)
    stopped = check_stop(request, self.max_model_len)
    return new_token_ids, stopped
```
AsyncScheduler重写此方法添加placeholder管理和异步cache_blocks逻辑（见[4_async_scheduler.md](./4_async_scheduler.md)）。

### 10.6 _handle_stopped_request
`scheduler.py:2025`：
- 设置request.status为对应finished状态
- 调用`_free_request(request)`释放KV blocks和encoder cache
- 将req_id加入`finished_req_ids`供下轮schedule通知worker清理

### 10.7 组装返回值
```python
engine_core_outputs: dict[int, EngineCoreOutputs] = defaultdict(EngineCoreOutputs)
for client_id, outputs in req_outputs.items():
    engine_core_outputs[client_id].outputs.extend(outputs)
    engine_core_outputs[client_id].engine_core_events.extend(events)
if self.finished_req_ids_dict is not None:
    for client_id, fset in self.finished_req_ids_dict.items():
        engine_core_outputs[client_id].finished_req_ids.update(fset)
        fset.clear()
self.processed_step_seq += 1
self._drain_deferred_frees()
return engine_core_outputs
```

---

## 11. 请求生命周期管理

### 11.1 add_request
`scheduler.py:2157`：
```python
def add_request(self, request: Request) -> None:
    assert request.request_id not in self.requests
    self.requests[request.request_id] = request
    request.status = RequestStatus.WAITING
    self._enqueue_waiting_request(request)
```

### 11.2 finish_requests
`scheduler.py:2181`，约60行：
1. 归一化request_ids为集合；None表示finish所有waiting+running
2. 在running中查找：找到则从running移除，`_free_request`，加入finished_req_ids
3. 在waiting/skipped_waiting中查找：找到则`remove_request`，设置FINISHED_ABORTED
4. 处理streaming input、grammar error、_inflight_prefills中引用
5. 返回实际被finish的Request列表

### 11.3 _free_request / _free_blocks / _free_request_blocks
- `_free_request(request)`：释放blocks（考虑defer）、释放encoder cache、从requests dict删除、从routed_experts_mgr清理
- `_free_blocks(request)`：调用kv_cache_manager.free_blocks(request)
- `_free_request_blocks(request)`：立即或延迟释放blocks。defer模式下入队deferred_frees，否则调用_free_blocks
- `_free_cow_retained_blocks(request)`：释放CoW过程中临时retain的blocks

---

## 12. Mamba SSM 块对齐拆分

对于含Mamba层且`mamba_cache_mode == "align"`的模型，SSM state只在block boundary materialize，chunk必须在block边界结束。`_mamba_block_aligned_split`（scheduler.py:357）对num_new_tokens做裁剪。

```python
def _mamba_block_aligned_split(self, request, num_new_tokens,
                               num_new_local_computed_tokens=0,
                               num_external_computed_tokens=0):
    start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
        return num_new_tokens  # decode阶段无需对齐

    block_size = self.cache_config.block_size
    last_cache_position = request.num_tokens - request.num_tokens % block_size
    if self.use_eagle:
        last_cache_position = max(last_cache_position - block_size, 0)

    end = start + num_new_tokens
    if end < last_cache_position:
        end = end // block_size * block_size  # 对齐到block boundary

    next_block_boundary = (start // block_size + 1) * block_size
    tail_boundary = (request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
                     if self.mamba_partial_cache_hit else 0)

    stops = (
        # 起始位置不对齐时，先对齐到下一个block
        next_block_boundary if start % block_size != 0 and next_block_boundary <= last_cache_position else 0,
        # 不超过最后一个可cache位置
        last_cache_position,
        # partial tail hash boundary（用于细粒度partial hit注册）
        tail_boundary if last_cache_position < tail_boundary < request.num_prompt_tokens else 0,
        # Marconi共享前缀boundary（block对齐后的位置）
        (start + (request.shared_prefix_boundary - start) // block_size * block_size
         if start < request.shared_prefix_boundary < end else 0),
    )
    end = min((s for s in stops if start < s < end), default=end)
    return max(end - start, 0)
```

强制停止点优先级：所有落在chunk内的stops取最小者。

---

## 13. reset_prefix_cache

`scheduler.py:2363`：
```python
def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False) -> bool:
    if not reset_running_requests and self.running:
        return False
    if reset_running_requests:
        for request in self.running.copy():
            self._preempt_request(request, time.monotonic())
            # async_scheduler场景设置async_tokens_to_discard
    self.kv_cache_manager.reset_prefix_cache()
    if reset_connector and self.connector:
        self.connector.reset_cache()
    self.encoder_cache_manager.reset_cache()
    return True
```

AsyncScheduler中，强制preempt running时为每个请求递增`async_tokens_to_discard`，以丢弃后续返回的在飞output帧。

---

## 14. 延迟块释放与deferred_fences

```python
def _drain_deferred_frees(self):
    while self.deferred_frees and self.deferred_frees[0][0] <= self.processed_step_seq:
        _, blocks = self.deferred_frees.popleft()
        self.kv_cache_manager.free(blocks)
```

每个deferred entry记录`(fence_seq, blocks)`，表示该step之前的forward都已完成时才可以安全释放blocks。`sched_step_seq`在schedule()末尾递增，`processed_step_seq`在update_from_output()末尾递增，形成生产-消费fence。

---

## 15. 其他辅助方法

- `_build_kv_connector_meta`：构造KV Connector metadata
- `_get_new_block_ids_to_zero`：收集本轮新分配且不在`_skip_zero_block_ids`中的block ids
- `_make_scheduled_encoder_input_stats`：统计本轮encoder调度数量
- `get_grammar_bitmask`：通过structured_output_manager为有pending结构化输出的请求生成grammar bitmask
- `get_request_counts`：返回`(len(running), len(waiting)+len(skipped_waiting)+num_waiting_for_streaming_input)`
- `get_num_unfinished_requests/has_finished_requests/has_requests`：状态查询
- `pause_state property / set_pause_state`：暂停控制
- `reset_encoder_cache`：调用encoder_cache_manager.reset()
- `make_stats`：构造SchedulerStats（running/waiting数、prefix cache命中率、preempt数、KV使用率、spec decode stats等）
- `shutdown`：关闭connector、kv_event_publisher等后台资源
- `get_kv_connector / get_ec_connector`：返回connector实例
- `_connector_finished`：KV Connector完成回调（注册到Connector的finish callback）
- `_request_remaining_blocks`：估算请求剩余需要的blocks数（用于in-flight预留计算）
- `_update_waiting_for_remote_kv`：处理单个请求的KV接收完成事件
- `_try_promote_blocked_waiting_request`：检查阻塞请求是否可被提升为WAITING
- `_update_from_kv_xfer_finished`：批量处理KV Connector完成事件
- `_update_requests_with_invalid_blocks / _handle_invalid_blocks`：处理KV加载校验失败的blocks
