# vLLM V1 Scheduler 调度机制总览（Full Attention 主线）

> 以**纯 Full Attention 模型 Llama-3-8B（block_size=16）**为主线，系统梳理 vLLM V1 调度器（`vllm/v1/core/sched/`）从启动装配、每步决策到输出回写的完整链路。本文与 KV Cache 文档配套：[`../kvcache_docs_v2/0_kv_cache_management_arch.md`](../../kvcache_docs_v2/0_kv_cache_management_arch.md) 讲"显存与块怎么管"，本文讲"每个调度步让谁、算多少 token"。

**章节顺序**：§1 为什么需要 → §2 模块全景与文件职责 → §3 启动时如何静态装配 → §4 统一调度模型（请求账本 + 预算） → §5 运行期一个调度步的完整节拍 → §6 请求状态机与三条队列 → §7 抢占机制 → §8 调度输出契约 → §9 扩展特性概览 → §10 设计要点小结。

**后续分篇规划（预告）**：本文只讲主线骨架，下列主题后续各自单独成文。

| 分篇 | 主题 | 对应源码 |
|---|---|---|
| 队列篇 | FCFS/Priority 双策略、waiting/skipped_waiting 选择与 blocked 提升 | `request_queue.py`、`scheduler.py:2007-2023,2623-2657` |
| 主流程篇 | `schedule()` 全分支逐行（encoder/LoRA/DP 节流/mamba 对齐） | `scheduler.py:427-1225` |
| 回写篇 | `update_from_output()`、stop 判定、投机接受/拒绝回滚 | `scheduler.py:1623-1997`、`utils.py` |
| 高级篇 | KV Connector（P/D）、异步调度与延迟释放、streaming、PP 节拍 | `async_scheduler.py`、`scheduler.py:2285-2324,2512-2860` |

---

## 1. 为什么需要 Scheduler

LLM 自回归推理时，GPU 每一步（一次 forward）可以同时处理一批请求的若干 token，但有三类资源是有限的：

1. **单步 token 容量**：一次 forward 的 token 总数有上限（`max_num_batched_tokens`，默认 2048，`config/scheduler.py:42`），超了显存和算子都吃不消。
2. **KV Cache 块**：物理块总数固定（如 KV Cache 文档中算出的 4096 块），每个在算的请求都要占块。
3. **并发请求槽位**：同时处于 running 的请求数有上限（`max_num_seqs`，默认 128，`config/scheduler.py:44`），受 batching 元数据结构约束。

同时，系统里不断有新请求到来、老请求结束，还要支持分块 prefill、前缀缓存、投机解码等优化。**Scheduler（调度器）就是每个调度步做一次"填预算"决策的组件：在三类约束内，决定这一步让哪些请求、各算多少个 token。**

> **生活化类比：Scheduler = 餐厅大堂经理**
>
> - **waiting 队列** = 门口等位的客人；**running 队列** = 已经入座正在吃的客人
> - **token** = 一道菜；**一次 forward（一个调度步）** = 厨房出一轮餐
> - **token 预算（2048）** = 厨房一轮最多能出的菜数
> - **KV Cache 块** = 餐桌（数量固定）；**max_num_seqs** = 餐厅最多同时开多少桌
> - **分块 prefill** = 大桌客人太多，厨房先上一部分菜，下一轮继续
> - **抢占** = 餐桌不够时，请吃到一半（decode 中）的客人先离桌把桌子让出来，等下重新入座、之前吃的（KV）作废重来
>
> 大堂经理每轮按固定规矩办事：**先照顾在座客人（RUNNING），再安排等位客人（WAITING）**，每轮把 2048 道菜的出餐额度尽量填满。

vLLM V1 调度器有三条核心设计（源码注释，`scheduler.py:429-438`）：

1. **没有"prefill 阶段"和"decode 阶段"之分**：每个请求只有两个进度数字——已算多少、总共要算多少，调度就是让前者追赶后者。这一套统一模型天然覆盖分块 prefill、前缀缓存、投机解码。
2. **只做决策，不碰显存**：Scheduler 不分配/读写任何 KV 张量，它通过第 5 层门面 `KVCacheManager` 拿到整数 `block_id` 列表，连同 token 数一起下发给 worker（见 KV Cache 文档 §2）。
3. **每步一个不可分割的决策包**：`schedule()` 产出 `SchedulerOutput`，EngineCore 拿去执行，执行完再用 `update_from_output()` 回写结果——调度状态只在这两个时点变更。

---

## 2. 模块全景：Scheduler 在引擎中的位置

### 2.1 一个调度步中的上下游

```
┌────────────────────────────────────────────────────────────────────┐
│ EngineCore（engine/core.py，引擎主循环）                             │
│                                                                     │
│   step()（core.py:581）每步调一次：                                  │
│                                                                     │
│   ① scheduler.schedule() ──────────────── 产出 SchedulerOutput       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Scheduler（sched/scheduler.py）                               │  │
│  │   ├─ waiting / skipped_waiting / running 三条请求队列          │  │
│  │   ├─ requests: dict[req_id → Request]（全部请求的总账）        │  │
│  │   ├─ EncoderCacheManager（多模态编码器输出缓存）               │  │
│  │   ├─ KVConnector / ECConnector（可选，P/D 远端 KV）            │  │
│  │   └─ KVCacheManager（KV Cache 第5层门面，唯一入口）            │  │
│  │         └─ get_computed_blocks / allocate_slots / free ...    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│   ② model_executor.execute_model(output)  ── RPC/直调 ──▶ Worker    │
│      Worker 按 output 里的 block_table 与 token 数执行 forward       │
│   ③ model_executor.sample_tokens()          采样新 token            │
│   ④ scheduler.update_from_output(output, model_output)              │
│      回写新 token、判定结束、释放块 ──▶ dict[client → 输出]          │
└────────────────────────────────────────────────────────────────────┘
```

注意 Scheduler 与物理显存之间始终隔着 `KVCacheManager`：调度器说"请求 R 这一步算 38 个 token"，`KVCacheManager.allocate_slots()` 负责把整数块号准备好；真正的张量读写发生在 worker 的 attention 算子里。

### 2.2 关键文件职责（`vllm/v1/core/sched/` 目录）

| 文件 | 职责 | 关键内容 |
|------|------|---------|
| `interface.py` | 调度器抽象接口 | `SchedulerInterface`（ABC）、`PauseState` 枚举（`interface.py:23-34`） |
| `scheduler.py` | 调度主类（约 2860 行） | `Scheduler(SchedulerInterface)`：`schedule()`、`update_from_output()`、请求生命周期、抢占、connector 交互 |
| `async_scheduler.py` | 异步调度子类（75 行） | `AsyncScheduler(Scheduler)`：投机占位符、PP 微批节拍、异步 `cache_blocks` |
| `output.py` | 调度输出数据结构 | `SchedulerOutput`、`NewRequestData`、`CachedRequestData`、`GrammarOutput` |
| `request_queue.py` | 请求队列 | `RequestQueue` 抽象类、`FCFSRequestQueue`（deque）、`PriorityRequestQueue`（最小堆）、工厂 |
| `utils.py` | 调度工具 | `check_stop()`（停止条件判定）、`check_sequence_repetition()`、`remove_all()` |
| `scheduler_design.md` | 已有设计笔记 | 早期章节式笔记，可与本文对照阅读 |

### 2.3 类继承关系

```
SchedulerInterface（interface.py · ABC，定义 schedule/update_from_output 等契约）
└── Scheduler（scheduler.py · 唯一完整实现）
    └── AsyncScheduler（async_scheduler.py · 仅覆写 2 个方法）
```

`SchedulerConfig.get_scheduler_cls()`（`config/scheduler.py:170-175`）按是否开启异步调度选择二者之一，EngineCore 只依赖抽象接口，不感知具体子类。

```
RequestQueue（request_queue.py · ABC）
├── FCFSRequestQueue(deque[Request], RequestQueue)   # 策略 fcfs（默认）
└── PriorityRequestQueue(RequestQueue)               # 策略 priority，内部最小堆
```

---

## 3. 系统启动期：Scheduler 的静态装配

Scheduler 在 **KV Cache 五层全部建好之后**才创建。回顾 KV Cache 文档 §3 的装配流水线：① 算规格 → ② 测预算 → ③ 做编排（`KVCacheConfig`）→ ④ 落物理张量，RPC 返回后 ⑤ 才创建 Scheduler。

入口：`EngineCore.__init__`（`engine/core.py:141-166`）：

```python
kv_cache_config = self._initialize_kv_caches(vllm_config)   # KV 五层先就绪
scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(...)
self.scheduler = Scheduler(                                 # engine/core.py:158
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    block_size=scheduler_block_size,
    hash_block_size=hash_block_size,
    ...
)
```

`Scheduler.__init__`（`scheduler.py:70-355`）内部装配顺序：

| 顺序 | 装配内容 | 源码位置 | 说明 |
|------|---------|---------|------|
| 1 | 三类约束字段 | `scheduler.py:109-123` | `max_num_running_reqs=max_num_seqs`、`max_num_scheduled_tokens`、`max_model_len` |
| 2 | KV / EC Connector | `scheduler.py:128-163` | 仅配置了 `kv_transfer_config` / `ec_transfer_config` 时创建（P/D 分离等场景） |
| 3 | 请求总账与三条队列 | `scheduler.py:173-185` | `requests` 字典 + `waiting` + `skipped_waiting` + `running` |
| 4 | EncoderCacheManager | `scheduler.py:210-238` | 多模态/编码器-解码器模型才有实际预算，纯文本为 0 |
| 5 | 投机解码参数 | `scheduler.py:239-265` | `num_spec_tokens`、`num_lookahead_tokens`（EAGLE/draft model/dflash 各不同） |
| 6 | **KVCacheManager（第5层门面）** | `scheduler.py:271-285` | 传入 `kv_cache_config`、block_size、watermark 等；随后 connector 绑定其 block_pool（`scheduler.py:288-289`） |
| 7 | 杂项状态 | `scheduler.py:291-355` | `current_step` 计数器、`_inflight_prefills` 集合、`deferred_frees` 延迟释放 FIFO 等 |

> 与 KV Cache 文档的衔接点：Scheduler 构造函数的第 6 步就是 KV Cache 逻辑三层（BlockPool → Manager → Coordinator → KVCacheManager）的诞生处。**Scheduler 持有 KVCacheManager，且是它唯一的调用者**；两者生命周期完全相同，都在 EngineCore 进程内，没有 RPC。

### 3.1 主线示例配置（全文共用）

沿用 KV Cache 文档的 Llama-3-8B 示例，本文涉及的调度参数固定为：

| 参数 | 值 | 来源 | 含义 |
|---|---|---|---|
| `block_size` | 16 | cache 配置 | 每块 16 token |
| `max_num_batched_tokens` | 2048（默认） | `config/scheduler.py:42` | 一个调度步的 token 总预算 |
| `max_num_seqs` | 128（默认） | `config/scheduler.py:44` | running 请求数上限（并发槽位） |
| `enable_chunked_prefill` | True（默认） | `config/scheduler.py:74` | 长 prompt 是否允许切块跨步 |
| `policy` | `"fcfs"`（默认） | `config/scheduler.py:99` | waiting 出队策略 |
| `long_prefill_token_threshold` | 0（关闭） | `config/scheduler.py:70` | 单请求单步 prefill token 上限，0 不限 |
| `watermark` | 0.0（关闭） | `config/scheduler.py:136` | 预留空闲块比例，透传给 KVCacheManager |

---

## 4. 核心心智模型：请求账本与 token 预算

### 4.1 请求头上的四个进度数字

理解调度器只需盯住 `Request`（`vllm/v1/request.py`）上的几个字段：

| 字段 | 定义 | 源码 |
|------|------|------|
| `num_tokens` | prompt token + 已生成输出 token 的总数，即 `len(all_token_ids)` | `request.py:266-267` |
| `num_tokens_with_spec` | `num_tokens + len(spec_token_ids)`，投机草稿 token 也算"待算" | `request.py:270-271` |
| `num_computed_tokens` | **已经完成 forward 的 token 数**（调度器视角的进度条） | `request.py:167` |
| `num_in_flight_tokens` | 已调度但结果尚未回写的 token 数（异步调度/PP 下 >0） | `request.py:156` |

另外两个异步调度才用的字段：`num_output_placeholders`（已提前许诺、还没拿到真实 id 的输出坑位数，`request.py:150`）、`spec_token_ids`（下一步的草稿 token，`request.py:166`）。

**调度的全部数学就是：每步给每个请求发 `num_new_tokens` 个新额度，让它的 `num_computed_tokens` 去追 `num_tokens_with_spec`。**

- 纯同步、无投机时，一个 decode 请求的差距恒为 1（上一步采样出 1 个新 token 追加在 `all_token_ids` 尾部），所以每步调 1 个 token；
- prefill 请求第一次差距等于 prompt 长度（如 70），可能一步算完，也可能被预算切块；
- 开了投机（如 4 个草稿），差距 = 1 + 4 = 5，一步调 5 个 token，回写时再按接受/拒绝修正进度（见 §5.4）。

### 4.2 调度器手上的三本账

```
self.requests: dict[str, Request]     # 总账：所有未彻底消失的请求（含异步收尾中的）
self.waiting:    RequestQueue         # 等位队列：尚未完成首次 prefill 的请求
self.skipped_waiting: RequestQueue    # 挂起等位：因外部依赖本轮跳过的请求（等远端KV/语法编译/流式输入）
self.running:    list[Request]        # 在座队列：已入过学、正在持续 decode（含未算完的 prefill 切块）
```

加上两个"捎带通知"集合：

- `self.finished_req_ids: set[str]`（`scheduler.py:191`）：上一步结束、需要通知 worker 清理缓存的请求 ID，**每步末尾清空**（`scheduler.py:1317`）；
- `self.reset_preempted_req_ids: set[str]`（`scheduler.py:194`）：本步被抢占的请求 ID，同样每步末尾清空（`scheduler.py:1318`）。

### 4.3 每步递减的 token 预算

`schedule()` 开头建立预算（`scheduler.py:447`）：

```python
token_budget = self.max_num_scheduled_tokens   # 主线 = 2048；PAUSED_ALL 时置 0
```

每成功调度一个请求就 `token_budget -= num_new_tokens`（running：`scheduler.py:621`；waiting：`scheduler.py:1045`），所有调度决策本质上都是在回答："**剩下的预算还够不够这个请求？**"多模态场景另有一本 `encoder_compute_budget = max_num_encoder_input_tokens`（`scheduler.py:454`，默认同 2048），约束视觉编码器的 token 数。

### 4.4 数字最小例子（请求进度追赶）

```
请求 R：prompt = 70 token，max_tokens = 32，block_size = 16

时刻 t0（刚入队）： num_tokens=70  num_computed_tokens=0  → 差距 70
t1 prefill 一步：  调度 num_new_tokens=38（前缀命中 32，见 §5.3）
                  schedule 末尾乐观推进：num_computed_tokens = 32+38 = 70
t2 decode 第1步：  all_token_ids 尾部已有 t1 采样的 1 个 token
                  num_tokens=71, num_computed_tokens=70 → num_new_tokens=1
t3 decode 第2步：  num_tokens=72, num_computed_tokens=71 → num_new_tokens=1
…每步差距恒为 1，直到生成 32 个输出或遇到 EOS
```

---

## 5. 运行期：一个调度步的完整节拍

### 5.1 三步闭环（EngineCore 驱动）

`EngineCore.step()`（`engine/core.py:581-611`）是一个 busy loop，每步严格按下列顺序执行：

```
                 ┌─────────────────────────── schedule() ───────────────────────────┐
 步开始          │ 1. new_step_starts()：通知 KVCacheManager 新步开始                   │
 ──────▶         │ 2. 阶段一：遍历 running，逐个"续算"（含必要时抢占）                   │
                 │ 3. 阶段二：预算有余且本步无抢占时，从 waiting 准入新请求              │
                 │ 4. 组装 SchedulerOutput（新请求全量 / 老请求增量 / 清零块 / ...）     │
                 │ 5. _update_after_schedule()：乐观推进 num_computed_tokens            │
                 └──────────────────────────────────────────────────────────────────┘
                                          │ SchedulerOutput
                                          ▼
                 execute_model() ──▶ Worker forward + sample（采样新 token）
                                          │ ModelRunnerOutput
                                          ▼
                 ┌──────────────────── update_from_output() ────────────────────────┐
                 │ 1. 投机接受/拒绝 → 回滚 num_computed_tokens                         │
                 │ 2. 追加新 token、check_stop() 判定是否结束                          │
                 │ 3. 结束的请求：出队 + 释放 KV 块 + 登记 finished_req_ids            │
                 │ 4. 组装每个 client 的 EngineCoreOutputs（流式返回 token）           │
                 └───────────────────────────────────────────────────────────────────┘
```

两个方向的数据契约：**下行** `SchedulerOutput`（`sched/output.py`），**上行** `ModelRunnerOutput`（`vllm/v1/outputs.py`）。

### 5.2 阶段一：RUNNING 优先（`scheduler.py:473-656`）

```
while running 中还有请求 and token_budget > 0:
    1. 跳过判定：
       - 占位符已到 max_tokens（async，避免多调一步）           scheduler.py:476-490
       - PP+V2 节拍未到（next_decode_eligible_step）            scheduler.py:492-496
       - DP 节流步且它是 prefill 切块                           scheduler.py:498-502
    2. 算差距 num_new_tokens = num_tokens_with_spec
                                + num_output_placeholders
                                - num_computed_tokens           scheduler.py:504-508
       再依次受 long_prefill_threshold / token_budget / max_model_len 三道裁剪
    3. 多模态：_try_schedule_encoder_inputs() 同步占编码器预算    scheduler.py:526-538
    4. allocate_slots(request, num_new_tokens, lookahead)       scheduler.py:566
       ├─ 返回新块 → 调度成功：记账、扣预算                      scheduler.py:616-640
       └─ 返回 None（块不够）→ 抢占受害者后重试                  scheduler.py:576-609（见 §7）
```

关键顺序结论：**RUNNING 永远先于 WAITING 被服务**；正在 decode 的请求延迟最稳定，新 prefill 只能吃 running 用完后剩下的预算——这就是 vLLM 调度反直觉但重要的一点。如果某请求这一步因预算为 0、编码器不足等原因算不了，用的是 `continue` 而非 `break`（`scheduler.py:560-561`），允许跳过它服务后面的 running 请求，即**队首阻塞被刻意放松**。

### 5.3 阶段二：WAITING 准入（`scheduler.py:668-1078`）

进入条件本身就是一条重要规则：**本步发生过抢占，或处于 pause 状态，就完全不准入新请求**（`scheduler.py:668`）。

```
while (waiting 或 skipped_waiting 非空) and token_budget > 0:
    0. 槽位检查：len(running) + 等流式输入数 ≥ max_num_seqs(128) → 直接 break
                                                                scheduler.py:674-676
    1. 选队列、peek 队头；blocked 状态先尝试提升（等远端KV/语法/流式）
       提升不了 → 移入本步 skipped，continue                scheduler.py:685-695
    2. LoRA 约束：新 adapter 会超 max_loras → skip           scheduler.py:699-710
    3. 【仅首次】前缀查找：
       kv_cache_manager.get_computed_blocks(request)
         → (new_computed_blocks, num_new_local_computed_tokens, shared_prefix_boundary)
       配了 connector 时还要问远端命中                        scheduler.py:718-804
    4. num_new_tokens = request.num_tokens - num_computed_tokens
       - 不启用 chunked_prefill 且超预算 → break（后续请求也没戏） scheduler.py:877-883
       - 启用（主线）→ min(剩余prompt, token_budget)，切块       scheduler.py:885
       - 投机对齐：新请求只 decode 1 token 时垫料到 1+num_spec   scheduler.py:856-869
    5. allocate_slots(...)（传入本地/外部已算块、lookahead 等） scheduler.py:945-957
       ├─ None → 释放编码器触点后 break（★ waiting 请求不触发抢占） scheduler.py:959-966
       └─ 成功 → 出队、append 进 running、状态 WAITING/PREEMPTED → RUNNING
                   登记 num_scheduled_tokens、扣预算            scheduler.py:994-1047
    6. 特例：远端 KV 异步加载中 → 状态置 WAITING_FOR_REMOTE_KVS，
       本步不跑 forward，continue                              scheduler.py:995-1025
```

注意两个与阶段一的差异：

- **waiting 请求分配不到块时直接 `break`，不抢占任何人**（`scheduler.py:959-966`）。抢占只发生在阶段一（running 续算时）。
- waiting 的 num_new_tokens 用 `request.num_tokens`（而非 `_with_spec`）计算，因为被抢占后重回 waiting 的请求（resumed）已经带着输出 token（`scheduler.py:848-851` 注释）。

### 5.4 阶段三：回写闭环 `update_from_output()`（`scheduler.py:1623-1997`）

GPU 执行完后，按上一步 `SchedulerOutput.num_scheduled_tokens` 逐请求回写：

```
for req_id, num_tokens_scheduled in 上一步调度表.items():
    1. num_in_flight_tokens -= num_tokens_scheduled          scheduler.py:1691
    2. 【投机】generated = 采样 token（含被接受草稿）：
         num_accepted = len(generated) - 1（减去 1 个新采样）
         num_rejected = num_draft_tokens - num_accepted
         num_computed_tokens -= num_rejected  ← 多算的进度退回去  scheduler.py:1720-1734
    3. _update_request_with_output：逐个追加输出 token，
       每追加一个调 check_stop()（utils.py:94）                 scheduler.py:2043-2059
       EOS / stop_token_ids / max_model_len / max_tokens / 重复检测
         → 置 FINISHED_* 状态
    4. stopped：
       ├─ 可恢复流式会话 → 回 waiting 继续（不结束）            scheduler.py:2025-2041
       └─ 真结束 → _free_request()：通知 connector、释放编码器、
                   登记 finished_req_ids、释放 KV 块           scheduler.py:2244-2271
    5. 装 EngineCoreOutput（new_token_ids/finish_reason/...）  scheduler.py:1874-1891
出循环后：从 running/waiting 移除已结束请求，汇总错误、统计、KV 事件
```

**为什么 schedule 末尾要"乐观推进" `num_computed_tokens`**（`_update_after_schedule`，`scheduler.py:1280-1290`）：这样下一步可以立刻继续调度这个请求（尤其 prefill 切块场景），不必等 GPU 回写；等回写时如果投机 token 被拒绝，再把多算的部分减回来。`is_prefill_chunk` 也在同一处刷新（`scheduler.py:1288-1290`）：`num_computed_tokens < num_tokens(+占位符)` 即仍在 prefill。

### 5.5 一个完整调度步的数字推演（全文主线配置）

设某一调度步开始时：

```
running = [A, B, C]          # 三个 decode 请求，每步各差 1 token
waiting = [R, S, T]          # R: prompt 70（前缀命中 32）；S: prompt 4096；T: prompt 100
token_budget = 2048；num_gpu_blocks 充足；max_num_seqs=128
```

**阶段一（RUNNING）**：

| 子步 | 请求 | num_new_tokens | 累计已用 | token_budget 剩余 | allocate_slots |
|---|---|---|---|---|---|
| 1 | A | 1 | 1 | 2047 | 续 0 新块（当前尾块没满） |
| 2 | B | 1 | 2 | 2046 | 弹 1 新块（尾块刚写满） |
| 3 | C | 1 | 3 | 2045 | 续 0 新块 |

**阶段二（WAITING）**：

| 子步 | 请求 | 前缀命中 | num_new_tokens | budget 剩余 | 分配结果 | 去向 |
|---|---|---|---|---|---|---|
| 4 | R | 32（touch 块1/2） | 70−32 = **38** | 2007 | 新弹 3 块（装 38 token：16+16+6） | running，status→RUNNING |
| 5 | S | 0 | min(4096, 2007) = **2007**（切块！） | 0 | 按 2007 token 预占 ceil(2007/16) 块 | running，但仍在 `_inflight_prefills` |
| — | T | — | 预算为 0，while 条件不满足，留队 waiting | — | — | 下一轮再说 |

步末 `_update_after_schedule` 乐观推进：

```
A/B/C：num_computed_tokens 各 +1
R    ：num_computed_tokens 32 → 70；is_prefill_chunk=False（70 ≥ 70）→ 下一轮以 decode 身份进阶段一
S    ：num_computed_tokens 0 → 2007；is_prefill_chunk=True（2007 < 4096）→ 下一轮继续在 running 里续 prefill
```

`SchedulerOutput` 关键字段（`scheduler.py:1180-1201`）：`num_scheduled_tokens={A:1,B:1,C:1,R:38,S:2007}`、`total_num_scheduled_tokens=2048`、R 走 `scheduled_new_reqs`（全量）、A/B/C/S（若 S 是新请求则同样全量）其余走 `scheduled_cached_reqs`（增量）。

**回写时的投机子例子**（假设 B 开了 4 个草稿 token）：

```
调度时：num_new_tokens = 1 + 4 = 5，num_computed_tokens 乐观 +5
Worker 验证：草稿前 2 个被接受
  generated_token_ids 长度 = 1（新采样）+ 2（接受）= 3
  num_accepted = 3 − 1 = 2；num_rejected = 4 − 2 = 2
回写：num_computed_tokens −= 2（把后 2 个拒绝草稿对应的虚高进度退回）
      下一轮从"已接受位置 + 1 个新 token"继续
```

---

## 6. 请求状态机与三条队列

### 6.1 状态枚举（`vllm/v1/request.py:343-366`）

```
                         add_request()
                              │
                              ▼
                    ┌───────────────────┐
        ┌──────────▶│      WAITING       │ 普通等位（waiting 队列）
        │           └───────────────────┘
        │             │  ▲       │  ▲
        │  阻塞子状态   │  │提升    │  │重回
        │  (skipped_   │  │       │  │
        │   waiting)  ▼  │       │  │
        │   ┌─────────────────────────────────────┐
        │   │ WAITING_FOR_REMOTE_KVS              │ 远端 KV 异步拉取中
        │   │ WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR│ 语法编译中
        │   │ WAITING_FOR_STREAMING_REQ           │ 流式请求等下一段输入
        │   └─────────────────────────────────────┘
        │                     │ 首次/恢复调度成功（分配到块）
        │                     ▼
        │           ┌───────────────────┐  生成结束
        │           │      RUNNING       │──────────────┐
        │           └───────────────────┘              │
        │                     ▲ 恢复调度               ▼
        │                     │              ┌─────────────────────┐
        └─────────────────────┘              │ FINISHED_STOPPED     │
          抢占：释放块、num_computed_tokens=0 │ FINISHED_LENGTH_CAPPED
          状态 PREEMPTED、prepend 回 waiting  │ FINISHED_ABORTED     │
                                             │ FINISHED_IGNORED     │
                                             │ FINISHED_ERROR       │
                                             │ FINISHED_REPETITION  │
                                             └─────────────────────┘
```

判定规则：枚举值排在 `PREEMPTED` 之后的都算已结束（`RequestStatus.is_finished`，`request.py:364-366`）。`PREEMPTED` 只是一个瞬态标签：请求被放回 waiting 队列**队首**（`scheduler.py:1267`），下一轮在阶段二以"resumed"身份重新准入（`scheduler.py:1034-1035`）。

### 6.2 三条队列的协作

| 队列 | 数据结构（FCFS 主线） | 装谁 | 谁来消费 |
|------|----------------------|------|---------|
| `waiting` | `FCFSRequestQueue`（deque） | 普通 WAITING/PREEMPTED 请求 | 阶段二 |
| `skipped_waiting` | 同上 | 三种 blocked 子状态、以及因 LoRA/编码器/connector 暂不能调度而跳过的请求 | 阶段二，**FCFS 下比 waiting 优先**（`scheduler.py:2013-2015`） |
| `running` | `list[Request]` | 已完成准入、持续 decode/切块 prefill 的请求 | 阶段一 |

blocked 请求的"提升"发生在阶段二遍历到它时（`_try_promote_blocked_waiting_request`，`scheduler.py:2623-2657`）：例如 `WAITING_FOR_REMOTE_KVS` 的请求只有等 `update_from_output()` 从 worker 收到 `finished_recving` 信号、记入 `finished_recving_kv_req_ids` 后，才被允许回 WAITING/PREEMPTED 重新参与调度。

### 6.3 请求的进出 API

| 时机 | 方法 | 行为 |
|------|------|------|
| 新请求到达 | `add_request()`（`scheduler.py:2157-2179`） | 登记 `requests` 总账，按状态分流进 waiting/skipped_waiting；重复 req_id 走流式会话分支 |
| 客户端中止 / 前端停串 | `finish_requests()`（`scheduler.py:2181-2242`） | 从队列摘除、置 FINISHED_*、释放资源 |
| 自然生成结束 | `update_from_output()` 内 `check_stop()` → `_free_request()` | 见 §5.4 |

---

## 7. 抢占机制（Recomputation Preemption）

**V1 只支持"重算型抢占"，没有 V0 的 swap 到 CPU 的选项。** 被抢占请求的 KV 块全部归还，`num_computed_tokens` 清零，下次准入时从头重算 prompt（前缀缓存若还在，可通过 `get_computed_blocks` 找回一部分）。

触发点只在阶段一（`scheduler.py:564-613`）：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break                          # 块够，调度成功
    # 块不够：选受害者
    if policy == PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
        # priority 值越大越先被牺牲；同级先到先保护
        ...
    else:  # FCFS
        preempted_req = self.running.pop()   # 队尾 = 最新入座的请求，天然牺牲
    self._preempt_request(preempted_req, ...)
```

`_preempt_request()`（`scheduler.py:1246-1268`）做四件事：

1. `_free_request_blocks()` 归还全部 KV 块、`encoder_cache_manager.free()` 释放编码器触点；
2. `status = PREEMPTED`、`num_computed_tokens = 0`、清空 spec token、`num_preemptions += 1`；
3. `waiting.prepend_request(request)` 放回等位队首（FCFS 下恢复时最先被服务）；
4. 登记 `reset_preempted_req_ids`，随本步 `SchedulerOutput` 通知 worker 清缓存。

如果受害者就是当前发起请求自己（running 循环里），说明能牺牲的都牺牲了仍不够，本请求放弃，阶段一 `break`（`scheduler.py:607-613`）。PRIORITY 模式下若受害者本步已被调度，还要把它已扣的预算、块、编码器额度**回滚**（`scheduler.py:584-601`）。

> **设计取舍**：牺牲 decode 进度保护吞吐。decode 请求每步只用 1 token，重算 prompt 的代价主要落在被抢者身上；而新 prefill 往往携带一大批 token，让它尽快开始对整体吞吐更有利。配前缀缓存后，被抢者重算时能直接命中自己刚写过的哈希块（若未被驱逐），代价进一步降低。

---

## 8. 调度输出契约：SchedulerOutput（`sched/output.py`）

一个调度步对 worker 的全部指示装在一个 dataclass 里（`output.py:193-283`）。核心字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `scheduled_new_reqs` | `list[NewRequestData]` | **首次**调度的请求，全量数据（prompt、mm features、采样参数、block_ids…），worker 缓存后不再重发（`output.py:35-69`） |
| `scheduled_cached_reqs` | `CachedRequestData` | 已缓存请求的**增量**：新块、新 token、最新进度等（`output.py:116-181`），由 `_make_cached_request_data()`（`scheduler.py:1363-1420`）拼装 |
| `num_scheduled_tokens` | `dict[str, int]` | **核心决策表**：req_id → 本步 token 数；worker 据此切片输入 |
| `total_num_scheduled_tokens` | `int` | 上表之和，步末有 `≤ 2048` 断言（`scheduler.py:1082`） |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | 每请求的草稿 token（`-1` 为占位，等 worker 填真实 id） |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | 本步要跑编码器的多模态输入序号 |
| `num_common_prefix_blocks` | `list[int]` | running 请求的最长公共前缀块数，供 cascade attention 用（`scheduler.py:1095-1101`） |
| `finished_req_ids` | `set[str]` | 捎带通知 worker 删除已结束请求的缓存状态 |
| `new_block_ids_to_zero` | `list[int] \| None` | 本步新开的块，worker forward 前先清零防脏数据（`scheduler.py:1232-1244`） |
| `kv_cache_block_copies` | `list \| None` | 前缀分叉时的 CoW（copy-on-write）拷贝任务 |
| `kv_connector_metadata` / `ec_connector_metadata` | connector 对象 | 远端 KV / 编码器缓存的 load/save 指令（`scheduler.py:1207-1216`） |
| `num_spec_tokens_to_schedule` | `int` | 动态投机解码按当前 batch size 选出的下一步 K（`scheduler.py:1164-1169`） |

**新请求全量、老请求增量**是这份契约最重要的设计：请求数据可能很大（长 prompt、图片特征），但 worker 端有缓存，正常 decode 步只需下发几个整数，把引擎进程与 worker 间的通信压到最小。

---

## 9. 扩展特性概览（都长在同一副骨架上）

| 特性 | 对主线的改动点 | 源码位置 |
|------|---------------|---------|
| **Chunked Prefill** | waiting 的 `num_new_tokens` 受 `min(prompt剩余, token_budget)` 裁剪，请求带 `is_prefill_chunk` 留在 running 跨步续算 | `scheduler.py:877-885,1052-1054,1288` |
| **Prefix Caching** | waiting 首次准入时 `get_computed_blocks()` 查本地/远端命中，命中 token 直接计入 `num_computed_tokens`；细节见 KV Cache 文档 | `scheduler.py:718-804` |
| **投机解码（EAGLE/MTP）** | 差距公式加 `spec_token_ids`；lookahead 预留块；准入时垫料到统一长度保住 cudagraph；回写时接受/拒绝回滚 | `scheduler.py:504-508,624-640,856-869,1710-1741` |
| **KV Connector（P/D 分离/离线）** | waiting 多一次远端命中查询；异步加载进 `WAITING_FOR_REMOTE_KVS`；加载失败块可重算或报错；结束时 push/save；块释放可延迟 | `scheduler.py:742-799,995-1025,2512-2860` |
| **多模态（VLM）** | 第二本预算 `encoder_compute_budget`、EncoderCacheManager、跨不过图片位置就把切块回退 | `scheduler.py:1422-1580,2061-2092` |
| **流式输入（streaming）** | 请求可 resumable：生成一段后回 WAITING 等下一段输入，会话不销毁 | `scheduler.py:1320-1361,2025-2041` |
| **异步调度 / PP** | `AsyncScheduler` 子类管理占位符与 `pp_size` 解码节拍；`defer_block_free` 用 `sched_step_seq/processed_step_seq` 栅栏保证块不在 GPU 在途写入时被复用 | `async_scheduler.py`、`scheduler.py:322-326,2285-2324` |
| **Priority 调度** | 队列换最小堆（键 `(priority, arrival_time)`）；抢占选 `max` 即最低优先级者；两队列按队头比较选择 | `request_queue.py:131-198`、`scheduler.py:578-582,2017-2023` |
| **LoRA** | 每步收集在用 adapter 集合，受 `max_loras` 约束，新 adapter 塞不进就跳过该请求 | `scheduler.py:658-665,699-710,1039-1040` |
| **Pause 流控** | `PauseState`：PAUSED_ALL 预算清零；PAUSED_NEW 只服务 running | `interface.py:23-34`、`scheduler.py:448-450,2278-2283` |
| **DP prefill 对齐** | `throttle_prefills` 非对齐步推迟 prefill、只跑 decode，保证多 DP rank 节拍一致 | `scheduler.py:467-469,498-502,842-845` |
| **Mamba/混合模型** | `"align"` 模式下切块尾部必须对齐到块边界，保证 SSM state 可缓存 | `scheduler.py:357-425,540-543,907-915` |
| **权重热更新** | `reset_prefix_cache()`：可选强制抢占全部 running 请求后清缓存；另有 encoder/connector 缓存重置 | `scheduler.py:2363-2441` |

阅读建议：先用 §4 的账本模型 + §5 的三步闭环吃透纯文本、同步、无投机主线，再回到本表按需求分支对照源码。

---

## 10. 设计要点小结

1. **统一进度模型，消除阶段概念。** 没有 prefill 引擎/decode 引擎，只有"`num_computed_tokens` 追 `num_tokens_with_spec`"一件事；分块 prefill、前缀命中、投机解码全部体现为差距公式与回写修正的不同参数（`scheduler.py:429-438`）。
2. **每步一个决策包，状态只在两头变更。** `schedule()` 只读状态+产出 `SchedulerOutput`，GPU 执行期间调度状态冻结，`update_from_output()` 统一回写；异步调度下用"乐观推进 + 拒绝回滚 + 占位符/栅栏"在不破坏这条不变量的前提下提前跑下一步。
3. **RUNNING 优先，WAITING 吃剩余预算；抢占只发生在 RUNNING 续算时，且 V1 只重算不 swap。** 这保证了 decode 延迟稳定、失败代价局部化；waiting 分配失败只 `break`，绝不新抢块。
4. **调度器只做整数决策，物理世界全部经 KVCacheManager 桥接。** 决策产物里没有张量，只有 `{req_id: token 数}` 与整数 `block_id` 列表；零显存拷贝、可序列化、可跨进程下发。
5. **下行增量契约压通信。** 新请求全量一次、老请求每步只发增量（`NewRequestData` vs `CachedRequestData`），这是高并发下引擎- worker 通信的关键优化。
6. **三队列 + 阻塞子状态把"外部依赖"显式化。** 等远端 KV、等语法编译、等流式输入的请求不占调度通道，在 `skipped_waiting` 里挂起，条件满足（信号在 `update_from_output()` 中到达）后提升回正常队列。
