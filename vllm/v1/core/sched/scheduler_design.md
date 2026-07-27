# vLLM V1 调度器架构详解

> 基于源码 `vllm/vllm/v1/core/sched/` 目录，深入解析 vLLM V1 调度器架构及其与 KV Cache 的深度交互。
>
> **生活化类比**：把调度器想象成**餐厅的大堂经理**
> - 等待队列 = 等位的客人
> - 运行队列 = 正在用餐的客人
> - KV Cache = 餐桌（有限资源）
> - Token = 菜品
> - 调度 = 安排客人入座、点餐、上菜
> - 抢占 = 客人太多时，请吃得慢的客人先起来等一下

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [核心文件与数据结构](#2-核心文件与数据结构)
3. [调度主流程 — schedule()](#3-调度主流程--schedule)
4. [调度器与 KV Cache 的深度交互](#4-调度器与-kv-cache-的深度交互)
5. [请求状态机与生命周期](#5-请求状态机与生命周期)
6. [抢占机制](#6-抢占机制)
7. [前缀缓存查找流程](#7-前缀缓存查找流程)
8. [KV Connector 集成（P/D 分离）](#8-kv-connector-集成分离)
9. [高级特性](#9-高级特性)
10. [设计要点总结](#10-设计要点总结)

---

## 1. 整体架构概览

vLLM V1 调度器采用**统一调度模型**：没有单独的"prefill 阶段"和"decode 阶段"，每个请求只是在追赶自己的 `num_tokens_with_spec`。

### 1.1 四层调度架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     EngineCore (引擎核心)                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Scheduler (调度器)                      │  │
│  │  ┌────────────────────────────────────────────────────┐   │  │
│  │  │         KVCacheManager (KV Cache 管理器)            │   │  │
│  │  │  ┌──────────────────────────────────────────────┐  │   │  │
│  │  │  │       KVCacheCoordinator (协调器)             │  │   │  │
│  │  │  │   ┌──────────┴──────────┐                    │  │   │  │
│  │  │  │  SingleTypeManager    SingleTypeManager ...  │  │   │  │
│  │  │  │                      BlockPool                │  │   │  │
│  │  │  └──────────────────────────────────────────────┘  │   │  │
│  │  └────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 调度器核心思想

> **"There's no decoding phase nor prefill phase."**
>
> 每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`。
> 调度器的目标就是让每个请求的 `num_computed_tokens` 追上 `num_tokens_with_spec`。

这种统一模型天然支持：
- ✅ Chunked Prefill（分块预填充）
- ✅ Prefix Caching（前缀缓存）
- ✅ Speculative Decoding（投机解码）
- ✅ Jump Decoding（跳跃解码）

---

## 2. 核心文件与数据结构

### 2.1 文件职责一览

| 文件                     | 职责                                                              |
| ------------------------ | ----------------------------------------------------------------- |
| `interface.py`         | `SchedulerInterface` 抽象接口、`PauseState` 枚举                |
| `scheduler.py`         | `Scheduler` 主类，核心调度逻辑                                   |
| `output.py`            | 调度输出数据结构：`SchedulerOutput`、`NewRequestData`、`CachedRequestData` |
| `request_queue.py`     | 请求队列：`FCFSRequestQueue`、`PriorityRequestQueue`             |
| `utils.py`             | 工具函数                                                          |
| `async_scheduler.py`   | 异步调度器（v2 模型运行器用）                                     |

### 2.2 请求队列（RequestQueue）

**定义位置**：`request_queue.py:13`

#### 两种调度策略

```
SchedulingPolicy
    ├── FCFS (First-Come-First-Served) → FCFSRequestQueue (deque 实现)
    └── PRIORITY                      → PriorityRequestQueue (heap 实现)
```

#### FCFSRequestQueue

```python
class FCFSRequestQueue(deque[Request], RequestQueue):
    # 队头 = 最早到达的请求（最高优先级）
    # 队尾 = 最晚到达的请求（最低优先级）
```

| 方法                  | 操作       | 时间复杂度 |
| --------------------- | ---------- | ---------- |
| `add_request()`     | append     | O(1)       |
| `pop_request()`     | popleft    | O(1)       |
| `prepend_request()` | appendleft | O(1)       |
| `remove_request()`  | remove     | O(n)       |

#### PriorityRequestQueue

```python
class PriorityRequestQueue(RequestQueue):
    self._heap: list[Request]  # 最小堆
```

**排序键**：`(priority, arrival_time)` — priority 越小优先级越高，相同优先级按到达时间早的优先。

### 2.3 SchedulerOutput — 调度输出

**定义位置**：`output.py:193`

调度器每一步的输出，包含所有模型运行器需要的信息。

```
SchedulerOutput
├── scheduled_new_reqs: list[NewRequestData]     # 新调度的请求（全量数据）
├── scheduled_cached_reqs: CachedRequestData     # 已缓存的请求（增量数据）
├── num_scheduled_tokens: dict[str, int]         # 每个请求调度的 token 数
├── total_num_scheduled_tokens: int              # 总调度 token 数
├── scheduled_spec_decode_tokens: dict[str, list[int]]  # 投机解码 token
├── scheduled_encoder_inputs: dict[str, list[int]]     # 编码器输入
├── num_common_prefix_blocks: list[int]          # 公共前缀 block 数（级联注意力）
├── finished_req_ids: set[str]                   # 上一步完成的请求 ID
├── free_encoder_mm_hashes: list[str]            # 待释放的编码器缓存
├── new_block_ids_to_zero: list[int] | None      # 新分配需要清零的 block
├── kv_cache_block_copies: list[KVCacheBlockCopy] | None  # CoW 拷贝任务
├── partial_tail_offloads: dict[...] | None      # Partial tail 卸载
└── num_spec_tokens_to_schedule: int             # 下一步投机解码 token 数
```

#### NewRequestData vs CachedRequestData

| 类型               | 用途                     | 数据量 | 包含内容                                           |
| ------------------ | ------------------------ | ------ | -------------------------------------------------- |
| `NewRequestData`   | 首次调度的请求           | 大     | 完整 prompt、mm features、sampling params 等       |
| `CachedRequestData`| 已调度过的请求（增量）   | 小     | 新 token IDs、新 block IDs、num_computed_tokens 等 |

**设计目的**：最小化调度器→Worker 的通信开销。Worker 缓存了请求数据，后续只需要发增量。

### 2.4 调度器内部状态

```python
class Scheduler:
    # 请求存储
    self.requests: dict[str, Request]              # 所有请求的字典
    
    # 三个队列
    self.waiting: RequestQueue                     # 等待队列
    self.skipped_waiting: RequestQueue             # 被跳过的等待请求
    self.running: list[Request]                    # 运行队列
    
    # 已完成的请求（等待通知 Worker 释放）
    self.finished_req_ids: set[str]
    
    # KV Cache 管理
    self.kv_cache_manager: KVCacheManager
    
    # KV Connector（P/D 分离）
    self.connector: KVConnectorBase_V1 | None
    
    # 调度约束
    self.max_num_running_reqs: int                 # 最大并发请求数
    self.max_num_scheduled_tokens: int             # 最大批 token 数
    self.max_model_len: int                        # 最大序列长度
```

---

## 3. 调度主流程 — schedule()

**定义位置**：`scheduler.py:427`

这是调度器的核心方法，每一步调用一次，对应模型的一次前向传播。

### 3.1 整体流程图

```
schedule()
  │
  ├── 初始化：token_budget, 各类计数器
  │
  ├── kv_cache_manager.new_step_starts()  ← KV Cache 新步开始
  │
  ├── 第一阶段：调度 RUNNING 请求
  │   │
  │   └── for each request in running:
  │       ├── 跳过检查（暂停/解码节奏/DP节流）
  │       ├── 计算 num_new_tokens
  │       ├── 编码器输入调度
  │       ├── Mamba block 对齐裁剪
  │       ├── allocate_slots()  ← 【KV Cache 核心交互】
  │       │   ├── 成功 → 加入调度
  │       │   └── 失败 → 抢占最低优先级请求，重试
  │       └── 更新 token_budget
  │
  ├── 第二阶段：调度 WAITING 请求
  │   │
  │   └── while waiting and token_budget > 0:
  │       ├── 最大并发数检查
  │       ├── LoRA 数量检查
  │       ├── 前缀缓存查找  ← 【KV Cache 核心交互】
  │       │   ├── 本地缓存查找 (get_computed_blocks)
  │       │   └── 外部缓存查找 (KVConnector)
  │       ├── 计算 num_new_tokens
  │       ├── allocate_slots()  ← 【KV Cache 核心交互】
  │       │   ├── 成功 → 加入 running
  │       │   └── 失败 → break
  │       └── 更新 token_budget
  │
  ├── 计算公共前缀 block 数（级联注意力）
  │
  ├── 构造 SchedulerOutput
  │   ├── NewRequestData（新请求）
  │   ├── CachedRequestData（运行请求）
  │   ├── new_block_ids_to_zero
  │   ├── kv_cache_block_copies (CoW)
  │   ├── partial_tail_offloads
  │   └── ...
  │
  ├── KV Connector 元数据构建
  │
  └── _update_after_schedule()  ← 更新请求状态
      └── num_computed_tokens += num_scheduled_tokens
```

### 3.2 第一阶段：调度 RUNNING 请求

**优先级最高**：已经在运行的请求先调度，保证吞吐。

```
┌──────────────────────────────────────────────────────────────┐
│               调度 RUNNING 请求循环                           │
│                                                              │
│  req_index = 0                                               │
│  while req_index < len(running) and token_budget > 0:        │
│      │                                                       │
│      ├── 跳过检查                                            │
│      │   ├── 已到达 max_tokens？→ skip                       │
│      │   ├── 未到解码 eligible step？→ skip                  │
│      │   └── DP 预填充节流？→ skip                            │
│      │                                                       │
│      ├── 计算 num_new_tokens                                 │
│      │   = num_tokens_with_spec + placeholders               │
│      │     - num_computed_tokens                             │
│      │   受限于：token_budget, max_model_len, long_prefill_threshold │
│      │                                                       │
│      ├── Mamba block 对齐裁剪（如果需要）                     │
│      │                                                       │
│      ├── allocate_slots()  ★ KV Cache 交互                   │
│      │   ├── 成功 → break out of while                       │
│      │   └── 失败 → 抢占最低优先级请求，重试                  │
│      │                                                       │
│      └── 加入调度集合，更新 token_budget                     │
└──────────────────────────────────────────────────────────────┘
```

**关键细节**：
- 不是 FCFS 严格顺序：某个请求因预算不足跳过，后面的请求仍可能被调度
- 抢占策略：PRIORITY 策略抢优先级最低的；FCFS 策略抢队尾的

### 3.3 第二阶段：调度 WAITING 请求

**前提条件**：没有发生抢占 + 未暂停新请求 + 还有 token 预算

```
┌──────────────────────────────────────────────────────────────┐
│               调度 WAITING 请求循环                           │
│                                                              │
│  while (waiting or skipped_waiting) and token_budget > 0:    │
│      │                                                       │
│      ├── 并发数检查：num_running >= max_num_running_reqs → break │
│      │                                                       │
│      ├── 取队首请求 peek_request()                           │
│      │                                                       │
│      ├── 阻塞状态提升检查（如 WAITING_FOR_REMOTE_KVS）        │
│      │                                                       │
│      ├── LoRA 并发数检查                                     │
│      │                                                       │
│      ├── ★ 前缀缓存查找（见第7章详解）                        │
│      │   ├── 本地前缀缓存：get_computed_blocks()             │
│      │   └── 外部前缀缓存：KVConnector.get_num_new_matched_tokens() │
│      │                                                       │
│      ├── 计算 num_new_tokens                                 │
│      │                                                       │
│      ├── Mamba block 对齐裁剪                                │
│      │                                                       │
│      └── ★ allocate_slots() 分配 KV 块                      │
│          ├── 成功 → pop 请求，加入 running                   │
│          └── 失败 → break（容量不足）                        │
└──────────────────────────────────────────────────────────────┘
```

### 3.4 _update_after_schedule

**定义位置**：`scheduler.py:1271`

调度决策完成后，更新请求的内部状态。

```python
def _update_after_schedule(self, scheduler_output):
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]
        request.num_computed_tokens += num_scheduled_token    # 已计算 token 增加
        request.num_in_flight_tokens += num_scheduled_token   # 在途 token 增加
        request.is_prefill_chunk = (num_computed_tokens 
                                    < num_tokens + placeholders)
        if not request.is_prefill_chunk:
            self._inflight_prefills.discard(request)
    
    # 清空 finished_req_ids 和 reset_preempted_req_ids
    self.finished_req_ids = set()
    self.reset_preempted_req_ids = set()
```

**为什么要在调度后立即更新？**

1. `scheduler_output` 里需要原始的 `num_computed_tokens` 来确定输入位置
2. 提前更新后，下一步调度可以立即继续调度 prefill 请求（chunked prefill）
3. 如果后续有 token 被拒绝（投机解码），会在 `update_from_output` 中回退

---

## 4. 调度器与 KV Cache 的深度交互

调度器与 KV Cache Manager 是最紧密的合作伙伴。调度器决定"谁上"，KV Cache Manager 决定"能不能上"。

### 4.1 交互总览

```
┌─────────────────┐        allocate_slots         ┌────────────────────┐
│                 │ ─────────────────────────────▶ │                    │
│                 ◀───────────────────────────── │                    │
│   Scheduler     │     KVCacheBlocks / None      │  KVCacheManager    │
│                 │                               │                    │
│  - 决定调度谁   │      get_computed_blocks      │  - 管理 block 池   │
│  - 决定调度多少 │ ─────────────────────────────▶ │  - 前缀缓存查找    │
│                 ◀───────────────────────────── │  - 分配/释放        │
│                 │  (blocks, num_tokens, boundary)│  - 驱逐            │
└─────────────────┘                               └────────────────────┘
         │                                                    │
         │                                                    │
         └───────────────────────┬────────────────────────────┘
                                 │
                    调用点汇总（共 15+ 处）：
                    - schedule(): get_computed_blocks
                    - schedule(): allocate_slots (×2)
                    - _preempt_request: free blocks
                    - update_from_output: cache_blocks
                    - update_from_output: free blocks
                    - ...
```

### 4.2 核心交互点详解

#### 交互点 1：前缀缓存查找

```
位置：schedule() 中 WAITING 请求调度阶段
调用：kv_cache_manager.get_computed_blocks(request)
返回：(KVCacheBlocks, num_computed_tokens, shared_prefix_boundary)

作用：
  - 新请求进来时，先看看有多少前缀已经被缓存了
  - 命中的话，这些 token 不需要重新计算，直接复用
  - 直接影响 num_new_tokens 的计算
```

#### 交互点 2：Slot 分配（最核心）

```
位置：schedule() 中 RUNNING 和 WAITING 调度阶段
调用：kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
返回：KVCacheBlocks | None

作用：
  - 尝试为请求分配 num_new_tokens 对应的 KV block
  - 成功 → 返回 KVCacheBlocks，请求可以被调度
  - 失败 → 返回 None，调度器需要抢占或跳过
```

**alllocate_slots 是调度准入的"守门员"**：
- 有空间 → 让请求进来
- 没空间 → 对不起，请等待或请别人出去

#### 交互点 3：抢占时释放

```
位置：_preempt_request()
调用：kv_cache_manager.free(request)

作用：
  - 被抢占的请求释放其所有 KV block
  - 释放的 block 回到空闲队列
  - 为其他请求腾出空间
```

#### 交互点 4：请求完成时释放

```
位置：update_from_output() → _free_request()
调用：kv_cache_manager.free(request)

作用：
  - 请求完成（正常结束/错误/中止）时释放所有 KV block
  - 将 request_id 加入 finished_req_ids 通知 Worker
```

#### 交互点 5：前缀缓存统计

```
位置：schedule() 中 WAITING 请求准入后
调用：kv_cache_manager.record_prefix_cache_stats(request, num_new_local_computed_tokens)

作用：
  - 记录前缀缓存命中率统计
  - 用于监控和调优
```

#### 交互点 6：公共前缀计算

```
位置：schedule() 末尾
调用：kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
返回：list[int] （每个 KV Cache group 的公共前缀 block 数）

作用：
  - 计算所有运行请求的最长公共前缀
  - 用于 cascade attention 优化（公共前缀只算一次）
```

#### 交互点 7：新 Block 清零

```
位置：schedule() 末尾构造 SchedulerOutput
调用：kv_cache_manager.take_new_block_ids()
返回：list[int]

作用：
  - 获取本步新分配的 block ID 列表
  - Worker 在使用前清零这些 block 的 GPU 内存
  - 防止旧数据污染注意力计算
```

#### 交互点 8：CoW 拷贝任务

```
位置：schedule() 末尾构造 SchedulerOutput
调用：kv_cache_manager.take_kv_cache_block_copies()
返回：(copies, retained_blocks)

作用：
  - 取出待执行的 Copy-on-Write 拷贝任务
  - Worker 在 forward 前执行这些拷贝
  - 用于部分前缀命中的场景
```

#### 交互点 9：Partial Tail Offload

```
位置：schedule() 末尾（仅 producer 模式）
调用：kv_cache_manager.take_partial_tail_offloads()
返回：dict[str, list[tuple[group_id, block_id, boundary_tokens]]]

作用：
  - 取出 producer 端的 partial tail 卸载任务
  - 交给 KV Connector 传输给 consumer
```

#### 交互点 10：缓存 Block（update_from_output）

```
位置：update_from_output() 中请求完成计算后
调用：kv_cache_manager.cache_blocks(request, num_computed_tokens)

作用：
  - 将新填满的 block 计算哈希并存入前缀缓存
  - 这样后续请求可以命中这些前缀
```

### 4.3 调度决策与 KV Cache 的关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                      调度决策流程                                │
│                                                                 │
│  有新请求要调度？                                                 │
│      │                                                          │
│      ▼                                                          │
│  get_computed_blocks()  ← 查前缀缓存                            │
│      │                                                          │
│      ├─ 命中 N 个 token → 只需计算剩余部分                       │
│      └─ 全不命中 → 需要计算全部                                  │
│                                                                 │
│      ▼                                                          │
│  计算需要分配多少新 block                                        │
│      │                                                          │
│      ▼                                                          │
│  allocate_slots()  ← 尝试分配                                   │
│      │                                                          │
│      ├─ 成功 → ✅ 调度该请求                                    │
│      │                                                          │
│      └─ 失败 → ❌ 空间不足                                      │
│               │                                                 │
│               ├─ 有可抢占的请求？→ 抢占，重试                    │
│               │                                                 │
│               └─ 无可抢占的请求 → 跳过/等待                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 请求状态机与生命周期

### 5.1 RequestStatus 状态枚举

```
WAITING
    │
    │  add_request()
    │
    ▼
WAITING_FOR_STREAMING_REQ ──(streaming input complete)──┐
    │                                                    │
    │  等待远程 KV 加载                                   │
    ▼                                                    │
WAITING_FOR_REMOTE_KVS ──(KV loaded/failed)──┐          │
    │                                         │          │
    │  schedule() 准入                        │          │
    ▼                                         │          │
RUNNING  ◄───────────────────┐               │          │
    │  正常计算              │               │          │
    │                        │ 抢占           │          │
    │  完成/出错             │               │          │
    ▼                        │               │          │
FINISHED_* ────────────────┐ │               │          │
                            │ │               │          │
                    PREEMPTED ◄───────────────┴──────────┘
                            │
                            └── 重新调度 → RUNNING
```

### 5.2 各状态说明

| 状态                      | 含义                                                | 所在队列  |
| ------------------------- | --------------------------------------------------- | --------- |
| `WAITING`               | 等待调度，还没分配任何资源                          | waiting   |
| `WAITING_FOR_STREAMING_REQ` | 等待流式输入到达                                  | waiting   |
| `WAITING_FOR_REMOTE_KVS` | 等待远程 KV Cache 加载完成（P/D 分离）          | waiting   |
| `RUNNING`               | 正在运行，已分配 KV block，参与每步调度            | running   |
| `PREEMPTED`             | 被抢占，KV block 已释放，等待重新调度              | waiting   |
| `FINISHED_STOPPED`      | 正常结束（遇到 stop token / 达到 max_tokens）     | -         |
| `FINISHED_LENGTH`       | 因达到 max_model_len 而结束                       | -         |
| `FINISHED_ABORTED`      | 被客户端中止                                       | -         |
| `FINISHED_ERROR`        | 因错误结束                                         | -         |
| `FINISHED_IGNORED`      | 被忽略（异步调度中重复的）                         | -         |

### 5.3 完整生命周期示例

以一个标准 text generation 请求为例：

```
1. add_request()
   └── 请求进入 waiting 队列，状态 WAITING

2. schedule() — 第一次调度
   ├── get_computed_blocks() → 前缀缓存命中
   ├── allocate_slots() → 分配成功
   ├── 请求移入 running 队列，状态 RUNNING
   └── 本次调度 N 个 token（可能是 chunked prefill）

3. update_from_output() — 模型计算完成
   ├── num_computed_tokens 已在调度后增加
   ├── cache_blocks() → 将新填满的 block 存入前缀缓存
   ├── 生成 M 个输出 token
   └── 未完成 → 继续留在 running

4. schedule() — 后续 decode 步（每步 1 个 token + spec tokens）
   ├── allocate_slots() → 分配新的 block
   └── update_from_output() → 重复...

5. 请求完成（遇到 stop token）
   ├── update_from_output() 检测到 stop
   ├── _free_request() → 释放所有 KV block
   ├── finished_req_ids.add(req_id)
   └── 请求状态 → FINISHED_STOPPED

6. 下一次 schedule()
   └── SchedulerOutput 中带上 finished_req_ids
       → Worker 释放缓存的请求状态
```

---

## 6. 抢占机制（Preemption）

### 6.1 什么时候发生抢占？

当 `allocate_slots()` 返回 `None`（KV 空间不足）时：

```
allocate_slots() 失败？
    │
    ├── 有 running 请求可以抢占？
    │   ├── 是 → 抢占最低优先级的，释放其 block，重试
    │   └── 否 → 无法调度当前请求
    │
    └── WAITING 请求调度阶段 → 直接 break 退出
```

### 6.2 抢占流程图

```
┌──────────────────────────────────────────────────────────────┐
│                    抢占发生时                                 │
│                                                              │
│  allocate_slots() 返回 None                                  │
│      │                                                       │
│      ▼                                                       │
│  选择被抢占的请求                                             │
│      │                                                       │
│      ├── FCFS 策略：                                          │
│      │   preempted_req = running.pop()  (队尾)               │
│      │                                                       │
│      └── PRIORITY 策略：                                      │
│          preempted_req = max(running,                        │
│              key=lambda r: (r.priority, r.arrival_time))     │
│          running.remove(preempted_req)                       │
│                                                              │
│      ▼                                                       │
│  _preempt_request(preempted_req)                             │
│      │                                                       │
│      ├── kv_cache_manager.free(request)  ← 释放 KV block     │
│      ├── encoder_cache_manager.free(request)                 │
│      ├── request.status = PREEMPTED                          │
│      ├── request.num_computed_tokens = 0  ← 重置！           │
│      ├── request.num_preemptions += 1                        │
│      └── waiting.prepend_request(request)  ← 放回等待队首     │
│                                                              │
│      ▼                                                       │
│  如果抢占的就是当前请求？→ break（没法继续了）                │
│  否则 → 重试 allocate_slots()                                │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 抢占的代价

抢占是一个"昂贵"的操作：
- ❌ 被抢占请求的所有 KV Cache 计算全部作废
- ❌ 重新调度时需要从头计算（前缀缓存能救回一部分）
- ❌ 增加延迟

**Watermark 机制**就是为了减少抢占的发生——预留一部分 block 作为缓冲。

---

## 7. 前缀缓存查找流程

前缀缓存是 vLLM 的核心优化之一，能大幅提升相同前缀请求的吞吐。

### 7.1 何时触发查找？

**WAITING 请求第一次被调度时**（`num_computed_tokens == 0`）。

> 注意：RUNNING 状态的请求不会再查前缀缓存，因为它们已经有自己的 KV block 了。

### 7.2 完整查找流程图

```
┌──────────────────────────────────────────────────────────────────┐
│              WAITING 请求前缀缓存查找流程                         │
│                                                                  │
│  request.num_computed_tokens == 0 ?                              │
│      │                                                           │
│      ├── 否 → 跳过查找（继续调度）                                │
│      │                                                           │
│      └── 是 → 开始查找                                           │
│              │                                                   │
│              ├── 有 KVConnector 吗？                              │
│              │   │                                               │
│              │   ├── 是 → get_computed_blocks_for_connector()   │
│              │   │     (混合模型更准确的查找，可能 diverge)       │
│              │   │                                               │
│              │   └── 否 → get_computed_blocks()                  │
│              │         (标准查找)                                 │
│              │                                                   │
│              ├── 得到本地命中：new_computed_blocks, num_local    │
│              │                                                   │
│              ├── 有 KVConnector 吗？                              │
│              │   │                                               │
│              │   └── 是 → 查远程缓存                             │
│              │         │                                         │
│              │         ├── 计算 block_aligned_local              │
│              │         │   (去掉 partial tail，避免 CoW 冲突)    │
│              │         │                                         │
│              │         ├── connector.get_num_new_matched_tokens()│
│              │         │   返回 (ext_tokens, load_kv_async)      │
│              │         │                                         │
│              │         └── 比较本地 partial 和远程               │
│              │             │                                     │
│              │             ├── 远程 > 本地 partial               │
│              │             │   → 丢弃本地 partial，用远程        │
│              │             │                                     │
│              │             ├── 远程 ≤ 本地 partial               │
│              │             │   → 保留本地 partial，不用远程      │
│              │             │                                     │
│              │             └── 无 partial → 直接用远程           │
│              │                                                   │
│              └── 总命中 = 本地 + 外部                             │
│                  num_computed_tokens = num_local + num_external  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.3 为什么有两种查找方式？

| 查找方式                       | 调用方法                                | 适用场景 |
| ------------------------------ | --------------------------------------- | -------- |
| 标准查找                       | `get_computed_blocks()`               | 无 KV Connector |
| Connector 专用查找             | `get_computed_blocks_for_connector()` | 有 KV Connector |

**Connector 专用查找的特殊之处**：
- 混合模型中，不同 group 的命中长度可能 diverge（不一致）
- 返回额外的 `hit_diverged` 标志
- 如果 diverge 且没有外部 token 支撑，需要回退到标准查找的一致边界

### 7.4 Local Hit vs External Hit 的协调

这是 P/D 分离架构中的一个微妙问题：

```
场景：
  本地缓存命中了 100.5 个 block（100 个整 block + 半个 block 的 partial hit）
  远程缓存命中了 150 个 block

问题：
  本地 partial hit 需要 CoW
  如果远程更长，直接用远程的话，不需要 CoW，更高效

解决：
  把本地 hit 裁剪到整 block 边界（100 个）
  然后和远程比：
    - 远程 > 本地 partial 部分？→ 用远程，丢本地 partial
    - 远程 ≤ 本地 partial 部分？→ 保本地 partial，不用远程
```

---

## 8. KV Connector 集成（P/D 分离）

KV Connector 是 vLLM V1 支持 P/D 分离（Prefill/Decode 分离架构）的关键组件。

### 8.1 两种角色

```
KVConnectorRole
    ├── SCHEDULER  ← 调度器端
    └── WORKER     ← Worker 端
```

### 8.2 P/D 分离架构

```
┌──────────────┐         KV Transfer         ┌──────────────┐
│  Producer    │ ──────────────────────────▶ │  Consumer    │
│  (Prefill)   │                             │  (Decode)    │
│              │ ◀────────────────────────── │              │
└──────────────┘         KV Events           └──────────────┘
       │                                            │
       │                                            │
       └─────────────────┬──────────────────────────┘
                         │
                    Scheduler 中的 KVConnector
                         │
          ┌──────────────┴──────────────┐
          │                             │
    Producer 模式                  Consumer 模式
    - 发送 KV 给 consumer       - 从 producer 接收 KV
    - partial tail offload      - 异步加载 KV
    - 发送 BlockStored 事件     - 监听事件更新缓存
```

### 8.3 Consumer 模式流程

```
WAITING 请求调度时：
  ├── 查本地前缀缓存
  ├── 查远程前缀缓存（KVConnector）
  ├── 如果有远程命中且需要异步加载：
  │   ├── allocate_slots() 分配空间
  │   ├── request.status = WAITING_FOR_REMOTE_KVS
  │   ├── 加入 skipped_waiting
  │   └── 不实际调度（num_new_tokens = 0）
  │
  └── 加载完成后：
      ├── 状态提升回 WAITING
      └── 下次调度时正常计算
```

### 8.4 Producer 模式流程

```
请求计算完成（update_from_output）时：
  ├── cache_blocks() 把 block 存入前缀缓存
  ├── BlockStored 事件通过 KV Event Publisher 发出
  └── [Mamba align 模式] partial tail offload:
      ├── schedule() 末尾 take_partial_tail_offloads()
      ├── 交给 KVConnector 传输给 consumer
      └── consumer 可以通过细粒度前缀命中加载
```

### 8.5 延迟释放（Deferred Block Free）

**问题**：异步调度 + consumer 模式下，一个请求的 block 刚被释放，可能立刻被 KV Connector 的异步加载重新分配，而此时 GPU 可能还在写入旧数据。

**解决**：延迟释放

```python
if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
    self.defer_block_free = True
```

**机制**：
- 用 `sched_step_seq` 和 `processed_step_seq` 计数
- 释放的 block 先放入 `deferred_frees` 队列
- 等 `processed_step_seq >= fence_seq` 时才真正释放
- 确保 GPU 写入全部完成后，block 才能被重用

---

## 9. 高级特性

### 9.1 Mamba Block 对齐裁剪

**定义位置**：`scheduler.py:357` (`_mamba_block_aligned_split`)

**问题**：Mamba "align" 模式下，SSM 状态只在 chunk 末尾才被物化和缓存。如果 chunk 不在缓存边界结束，那这段状态就白算了，下次没法复用。

**解决**：把 chunk 的结束位置裁剪到缓存对齐边界。

```
原始 chunk（长度不限）：
┌───────────────────────────────────────┐
│                                       │
└───────────────────────────────────────┘
                         ↑ 非对齐结束，状态无法缓存

裁剪后的 chunk：
┌──────────────────┐
│                  │
└──────────────────┘
          ↑ 在 block 边界结束，状态可以被缓存
```

**对齐停止点（stops）**：
1. 下一个 block 边界（如果当前在 block 中间）
2. 最后一个可缓存的 block 边界
3. Prompt 的最后一个 hash 边界（partial tail 注册用）
4. 共享前缀交界处（Marconi 优化）

取这些停止点中最靠前的那个。

### 9.2 动态投机解码

根据批大小动态选择最优的投机解码 token 数：

```python
if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
    num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
        len(num_scheduled_tokens)
    ]
```

**查表法**：`num_speculative_tokens_per_batch_size` 配置 → 构建查找表 → 根据当前批大小查最优 K。

### 9.3 DP 预填充平衡

Data Parallel 模式下，通过 `throttle_prefills` 参数控制预填充节奏，让各 DP rank 的 prefill 步调一致。

```
throttle_prefills = True 且 有非 prefill 的运行请求？
    → 推迟所有 prefill 计算到 cadence-aligned step
    → decode 继续正常调度
```

### 9.4 级联注意力（Cascade Attention）

**`num_common_prefix_blocks`**：所有运行请求共享的前缀 block 数。

```
请求A: [P0][P1][P2][A3][A4][A5]...
请求B: [P0][P1][P2][B3][B4][B5]...
请求C: [P0][P1][P2][C3][C4][C5]...
         ↑  ↑  ↑
         公共前缀（3 个 block）
```

**用途**：公共前缀只计算一次，所有请求共享结果，大幅减少计算量。

### 9.5 暂停状态（PauseState）

| 状态           | 含义                                         |
| -------------- | -------------------------------------------- |
| `UNPAUSED`   | 正常运行                                     |
| `PAUSED_NEW` | 不调度新请求，已有运行请求继续调度           |
| `PAUSED_ALL` | 什么都不调度                                 |

用于服务滚动升级、模型权重热更新等场景。

### 9.6 流式输入支持

`WAITING_FOR_STREAMING_REQ` 状态：
- 请求已创建但 prompt 还没到齐
- 到达一块输入就追加一块
- 全部到达后转为 WAITING 状态

---

## 10. 设计要点总结

### 10.1 核心设计思想

1. **统一调度模型**：没有 prefill/decode 阶段之分，都是追赶 `num_tokens_with_spec`，天然支持 chunked prefill、spec decoding 等。

2. **KV Cache 是核心约束**：调度决策围绕 KV Cache 容量展开，`allocate_slots` 是准入的守门员。

3. **两阶段调度**：先调度 RUNNING（保证吞吐），再调度 WAITING（增加并发），优先级清晰。

4. **增量通信**：`NewRequestData` vs `CachedRequestData`，最小化调度器→Worker 的通信量。

### 10.2 与 KV Cache 的协同设计

5. **前缀缓存优先**：新请求进来先查前缀缓存，命中就不用重新算，大幅提升效率。

6. **抢占式调度**：KV 空间不足时抢占低优先级请求，保证高优先级请求的服务质量。

7. **Watermark 水位线**：预留一部分 block 减少抢占，在利用率和稳定性间取得平衡。

8. **CoW 写时复制**：部分前缀命中时用 CoW 隔离，最大化共享的同时保证安全。

### 10.3 可扩展性设计

9. **KV Connector 插件化**：P/D 分离架构通过 KVConnector 集成，不侵入核心调度逻辑。

10. **多策略队列**：FCFS 和 Priority 两种策略，抽象为统一的 RequestQueue 接口。

11. **事件驱动**：KV Cache 事件通过 EventPublisher 发出，支持外部系统监听和集成。

12. **延迟释放机制**：异步 + P/D 分离场景下的安全保证，避免数据竞争。
