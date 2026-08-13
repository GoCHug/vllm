# 4. AsyncScheduler 异步调度器

> 源码：vllm/vllm/v1/core/sched/async_scheduler.py（约100行）

---

## 目录

1. [AsyncScheduler定位](#1-asyncscheduler定位)
2. [__init__](#2-__init__)
3. [_update_after_schedule](#3-_update_after_schedule)
4. [_update_request_with_output](#4-_update_request_with_output)
5. [num_output_placeholders 机制](#5-num_output_placeholders-机制)
6. [PP微批次调度：next_decode_eligible_step](#6-pp微批次调度next_decode_eligible_step)

---

## 1. AsyncScheduler定位

`AsyncScheduler`继承自`Scheduler`，仅重写3个方法以支持：
- 异步调度（CPU上schedule与GPU上forward重叠执行）
- 流水线并行（Pipeline Parallelism, PP）v2 model runner下的微批次调度
- 异步场景下的KV cache时序正确性

```python
class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None: ...
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None: ...
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]: ...
```

采用模板方法模式：`Scheduler.schedule()`定义完整调度流程骨架，在特定钩子点调用上述方法，AsyncScheduler通过重写钩子方法注入异步行为。

---

## 2. __init__

```python
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
    self.pp_size = self.parallel_config.pipeline_parallel_size
```

字段说明：

| 字段 | 含义 |
|------|------|
| `_spec_token_placeholders` | 长度为`num_spec_tokens`的列表，元素为-1。作为可复用的只读占位符数组，避免每轮重新分配 |
| `pp_size` | 流水线并行stage数，驱动`next_decode_eligible_step`间隔计算 |

**为什么用-1占位**：异步调度下`schedule()`可能在worker尚未返回上一轮draft tokens时就被调用，此时下一轮spec token ids还未生成。先用-1占位构造SchedulerOutput传递给ModelRunner，worker端在真正执行前会通过`update_draft_token_ids_in_output`回填真实token ids。

---

## 3. _update_after_schedule

在`Scheduler._update_after_schedule`基础上追加异步/PP相关状态更新。

```python
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    super()._update_after_schedule(scheduler_output)
    spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

    # 更新占位符长度：用本轮num_spec_tokens_to_schedule作为下一轮占位符大小
    self._spec_token_placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule

    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]

        # prefill chunk请求不做decode阶段处理
        if request.is_prefill_chunk:
            continue

        # 结构化输出pending标记
        scheduler_output.pending_structured_output_tokens |= (
            request.use_structured_output and request.num_output_placeholders > 0
        )

        # 更新num_output_placeholders
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        request.num_output_placeholders += (
            self.num_sampled_tokens_per_step + cur_num_spec_tokens
        )

        # 设置spec token占位符
        request.spec_token_ids = self._spec_token_placeholders

        # PP+v2：设置下次decode允许步
        if self.use_v2_model_runner:
            request.next_decode_eligible_step = self.current_step + self.pp_size
```

### 关键逻辑

1. **placeholder长度动态调整**：动态spec decoding（`dynamic_sd_lookup`）下每轮K值可能变化，用`scheduler_output.num_spec_tokens_to_schedule`更新占位符长度。

2. **placeholder计数递增**：本轮调度后，该请求将产生`num_sampled_tokens_per_step`（AR=1，diffusion=0）个采样token加`cur_num_spec_tokens`个draft token，这些都是"已调度但尚未收到输出"的token，计入`num_output_placeholders`。

3. **PP decode节流**：`next_decode_eligible_step = current_step + pp_size`，使得同一请求必须间隔pp_size步才能再次被decode调度，匹配PP下微批次填充节奏。

---

## 4. _update_request_with_output

在`Scheduler._update_request_with_output`基础上处理placeholder兑现和异步KV缓存。

```python
def _update_request_with_output(
    self, request: Request, new_token_ids: list[int]
) -> tuple[list[int], bool]:
    # 强制preempt后丢弃在飞陈旧输出帧
    if request.async_tokens_to_discard > 0:
        request.async_tokens_to_discard -= 1
        return [], False

    status_before_update = request.status
    new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)

    # 兑现placeholder：收到M个有效输出，相应扣减
    request.num_output_placeholders -= len(new_token_ids)
    assert request.num_output_placeholders >= 0

    # 异步KV缓存：仅对RUNNING请求缓存新块
    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(
            request,
            request.num_computed_tokens - request.num_output_placeholders
        )
    return new_token_ids, stopped
```

### async_tokens_to_discard
`reset_prefix_cache(reset_running_requests=True)`强制preempt running请求时，AsyncScheduler会为每个请求递增`async_tokens_to_discard`。由于异步调度下可能仍有旧批次在GPU上执行，其输出会在后续`update_from_output`中返回，这些陈旧输出必须被丢弃直至计数器归零，避免对重置后重新开始的请求造成污染。

### cache_blocks的位置参数
缓存KV块时使用`request.num_computed_tokens - request.num_output_placeholders`作为已确认token位置，而非直接用`num_computed_tokens`。原因：

```
时间线示例（spec decode K=2）：
schedule(N):   分配3个token位置（1 sample + 2 spec）, placeholders=3
               GPU开始forward
schedule(N+1): 异步提前调度N+1（GPU还在跑N）
...
GPU返回N的输出：假设2个spec全部被reject，仅1个sample有效
→ new_token_ids长度=1，placeholders=3-1=2
→ num_computed_tokens包含了3个调度token，但只有1个是真实确认的
→ cache位置必须用 num_computed_tokens - placeholders = (旧+3) - 2 = 旧+1
```

被拒绝的spec token对应的KV是无效的，不应被prefix cache表注册为可复用块。减去未兑现的placeholder位置，确保只缓存已确认存在的token对应的KV。

---

## 5. num_output_placeholders 机制

### 定义
`num_output_placeholders`记录请求中已被调度出去但尚未收到模型输出确认的token数量，是异步正确性的核心计数器。

### 更新轨迹
```
初始值：0

schedule()中_update_after_schedule:
  placeholders += num_sampled_tokens_per_step + cur_num_spec_tokens

update_from_output()中_update_request_with_output:
  placeholders -= len(new_token_ids)  # new_token_ids是本轮确认的有效token数
```

### 在Scheduler调度循环中的使用
scheduler.py running调度阶段开头有跳过检查：
```python
if (request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
        >= request.num_prompt_tokens + request.max_tokens):
    req_index += 1
    continue
```
当`num_computed_tokens + 2 - placeholders >= prompt_len + max_tokens`时，即使表面上`num_computed_tokens`还没到上限，实际上已确认token数（扣除placeholder）已到达max_tokens，继续调度会产生多余的空step，故跳过。`+2`是为边界情况留出的安全余量。

### 结构化输出关联
`pending_structured_output_tokens`标志位由placeholder是否>0驱动：有未兑现placeholder时，grammar bitmask无法基于完整的最新上下文计算，标记为pending。

---

## 6. PP微批次调度：next_decode_eligible_step

### 机制
```python
# _update_after_schedule中设置
if self.use_v2_model_runner:
    request.next_decode_eligible_step = self.current_step + self.pp_size
```

```python
# schedule running阶段检查
if self.current_step < request.next_decode_eligible_step:
    req_index += 1
    continue
```

### PP=4流水线时序示例
```
step:  1    2    3    4    5    6    7    8
S0:   [A0][B0][C0][D0][A1][B1][C1][D1]...
S1:       [A0][B0][C0][D0][A1][B1][C1]...
S2:           [A0][B0][C0][D0][A1][B1]...
S3:               [A0][B0][C0][D0][A1]...
                                  ↑
                          A在step4完成S3输出
```
请求A在step1被S0处理，`next_decode_eligible_step = 1 + 4 = 5`，即step5才允许A再次decode（A1）。这保证每轮微批次的填充节奏，避免S0因A的数据未就绪而产生气泡。

### 与同步Scheduler的区别
同步Scheduler（非Async）不需要`next_decode_eligible_step`，因为每步严格串行：上一步所有stage完成后才调度下一步，不存在异步提前调度导致的同请求step间隔不足问题。
