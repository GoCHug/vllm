# 一条请求的 KV Cache 端到端时序（Llama-3-8B 视角）

> 主线：**纯 Full Attention 模型 Llama-3-8B**，单 KV cache group。用 Mermaid 时序图串起一条请求从进入 Scheduler 到最终释放的全过程。时序图用语义化表述；真实的源码调用点与行号保留在正文各节的**要点**中。
>
> **本套三篇总览的分工**：本文讲"流"（请求怎么一步步走）；[`0_kv_cache_management_arch.md`](./0_kv_cache_management_arch.md) 讲"层"（五层静态架构）；[`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) 讲"形状"（KV cache 字节布局）。基础数据结构见 [`0_kvcache_management_of_type.md`](./0_kvcache_management_of_type.md)。

---

## 0. 一条请求走过五个类（先认脸）

Llama-3-8B 是纯 Full Attention，KV 管理只经过 **五个类**，正好对应五层架构。全文只讲这五个类，不再用"第X层"。

| 类（文件） | 在流程中扮演什么 | 一句话职责 |
|---|---|---|
| `KVCacheManager`（`kv_cache_manager.py`） | 顶层编排入口，Scheduler 唯一对话方 | 对外暴露 `get_computed_blocks` / `allocate_slots` / `take_*` / `free` |
| `UnitaryKVCacheCoordinator`（`kv_cache_coordinator.py`） | 单组协调器，原样透传到下层 | 把 KVCacheManager 的每次调用下放给唯一管理器 |
| `FullAttentionManager`（`single_type_kv_cache_manager.py`） | 单组实现，真正的前缀/分配逻辑 | 最长前缀查找、算块数、touch、分配、缓存、释放 |
| `BlockPool`（`block_pool.py`） | 逻辑块池，KV 数据的"房主" | 哈希表查/写、`get_new_blocks`、`touch`、`free_blocks` |
| `GPUModelRunner`（`gpu_model_runner.py`） | 物理层，持有真实 K/V 张量 | 申请 `kv_caches[layer]`、`_zero_block_ids`、`forward`、`sample_tokens` |

> **调用链**：`KVCacheManager → UnitaryKVCacheCoordinator → FullAttentionManager → BlockPool`（KV 编排）；`GPUModelRunner` 独立持有物理张量，与逻辑块通过 `block_id` 相接。

---

## 1. 总览

`EngineCore.step()`（core.py:580）每步驱动 `schedule → execute_model → sample_tokens`。

**一条请求的端到端过程（编号速览）**（以示例 R：70/32 token 为准）：

1. **入队**：收到请求 → 构造 `Request` 并预计算 token 链式哈希 → `Scheduler.add_request()` 放入 **WAITING** 队列待调度。
2. **首次调度（prefill）**：`schedule()` 取出请求，准备一次性算完整段 prompt（70 token）。
3. **前缀缓存查找**：`get_computed_blocks()` 沿 KV 管理链下钻，用已算哈希查缓存表，找到可复用的已算前缀（命中前 2 块）。
4. **分配物理块**：`allocate_slots()` 内部 5 步——touch 命中块、为剩余 token 申请新块，拼出该请求的 `block_table`；新块登记为零清。
5. **组装调度输出**：Scheduler 组 `SchedulerOutput`，附上清零块 id 与 CoW 拷贝清单，交给 Worker 执行。
6. **GPU 写 KV**：`execute_model()` 清零新块 → attention 依 `block_table` 把整段 prompt 的 K/V 写入显存 → 采样出第 1 个输出 token，请求进入 **RUNNING**。
7. **decode 续写**：此后每步只算 1 个新输出 token，不做前缀缓存查找，只走 `allocate_slots`（当前块未满则 0 分配，填满才续申请 1 块）＋ forward 写 KV，重复到输出全部生成。
8. **释放**：请求完成，按 `block_table` **逆序**归还块——`ref_cnt` 归零才回收，有哈希的进队尾保护、无哈希的进队首优先复用。

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant Client
    participant LLMEngine
    participant EngineCore
    participant Scheduler
    participant KVCacheManager
    participant GPUModelRunner

    Note over Client,GPUModelRunner: ① 入队
    Client->>LLMEngine: generate(prompt)
    LLMEngine->>EngineCore: add_request()
    EngineCore->>Scheduler: add_request()
    Note over EngineCore,Scheduler: 构造 Request（预计算链式哈希）→ 入 WAITING 队列
    Scheduler->>EngineCore: 已入队

    Note over Scheduler,GPUModelRunner: ② 首次调度（prefill）
    EngineCore->>Scheduler: schedule()（调度 WAITING 请求）
    Scheduler->>KVCacheManager: ③ get_computed_blocks（前缀查找）
    Scheduler->>KVCacheManager: ④ allocate_slots（分配块）
    Note over KVCacheManager: 下钻链 KVCacheManager→UnitaryKVCacheCoordinator→<br/>FullAttentionManager→BlockPool；touch 命中块+申请新块 → block_table
    KVCacheManager->>Scheduler: KVCacheBlocks（新块 id、block_table、命中信息）
    Note over Scheduler: ⑤ 组装 SchedulerOutput（清零块 id + CoW 清单）
    Scheduler->>EngineCore: SchedulerOutput

    Note over EngineCore,GPUModelRunner: ⑥ GPU 写 KV
    EngineCore->>GPUModelRunner: execute_model
    Note over GPUModelRunner: 清零新块 → forward 中 attention 层<br/>依 block_table 写新 K/V、读已有 KV
    GPUModelRunner->>EngineCore: 返回
    EngineCore->>GPUModelRunner: sample_tokens
    GPUModelRunner->>EngineCore: 第 1 个输出 token（进 RUNNING）

    Note over Scheduler,GPUModelRunner: ⑦ decode 续写（循环）
    loop 每步 1 个新 token，直到输出全部生成
        EngineCore->>Scheduler: schedule()（调度 RUNNING 请求）
        Scheduler->>KVCacheManager: allocate_slots（不前缀查找；块未满0/填满则1）
        KVCacheManager->>Scheduler: KVCacheBlocks
        Scheduler->>EngineCore: SchedulerOutput
        EngineCore->>GPUModelRunner: execute_model
        Note over GPUModelRunner: forward 依 block_table 读+写 KV（填满块入缓存）
        GPUModelRunner->>EngineCore: 返回
        EngineCore->>GPUModelRunner: sample_tokens
        GPUModelRunner->>EngineCore: 1 个输出 token
    end

    Note over Scheduler,GPUModelRunner: ⑧ 结束释放
    Scheduler->>KVCacheManager: free(request)
    Note over KVCacheManager: 逆序归还；ref_cnt 归零才回收，<br/>有哈希进队尾、无哈希进队首
    KVCacheManager->>Scheduler: 完成
```

---

## 2. Llama-3-8B 的 KV 尺度与示例请求

下文所有阶段共用同一个请求示例，模型参数固定为 Llama-3-8B。

**模型 KV 尺度（Llama-3-8B config / vLLM `FullAttentionSpec`）**：

| 参数（Llama-3-8B config） | 值 | 含义 / 对应 vLLM `FullAttentionSpec` 字段 |
|---|---|---|
| `num_hidden_layers` | 32 | 层数；每层各有一份 KV 张量，但**共享同一套 block_id** |
| `num_attention_heads` | 32 | 查询头数（GQA） |
| `num_key_value_heads`（=`num_kv_heads`） | 8 | KV 头数；`32/8 = 4` 个查询头共享 1 个 KV 头 |
| `head_dim`（=`head_size`） | 128 | 每个 head 的维度 |
| `hidden_size` | 4096 | 隐层宽度 `= num_attention_heads × head_dim` |

下表为 **Llama-3-8B 模型 config 字段**；KV 缓存侧另有配套：`block_size=16`（一般块容纳 token 数）、`dtype=fp16`（2 字节/元素）。

每个 page 的物理大小：

```text
page_size_bytes = 2 × block_size × num_key_value_heads × head_dim × 2B
                = 2 × 16 × 8 × 128 × 2 = 65,536 B = 64 KiB
```

对应每层张量 `kv_caches[layer]` 的 shape：**`(num_blocks, 8, 16, 256)`**（最后 256 = 2×head_dim，K、V 拼接）。假设给 KV cache 划分 2 GiB，则 `num_blocks = 2 GiB ÷ 64 KiB ÷ 32 层 = 1024`——`GPUModelRunner` 每层申请 `(1024, 8, 16, 256)` 的 int8→view 张量，`BlockPool` 同时建 `KVCacheBlock(0..1023)`。

**示例请求 R**：

```
prompt     = "请用中文解释一下数据库索引，并举例说明 B+ 树索引与哈希索引的区别……"（70 个 token）
max_tokens = 32
block_size = 16
```

宏观路径：**入队（WAITING）→ 首次调度 prefill（算完 70 token，→ RUNNING）→ 每步 decode 续写 1 token（至 32 个输出）→ 结束释放**。

```
入队(WAITING) → prefill 首次调度
  ├─ get_computed_blocks: 70//16=4 个满块 hash 查表，命中前 2 块 → hit_length=32
  ├─ allocate_slots:  touch 2 命中块 + get_new_blocks(3) → block_table = [命中,命中,新,新,新]
  ├─ cache_blocks:   新满块 2、3 入哈希表；未满块 4 不入
  ├─ execute_model: 一次 forward 写 70 token KV 到 5 块
  └─ sample → 第 1 个输出 token → 状态变 RUNNING

decode 续写 32 步（每步 1 输出 token）
  ├─ 第 1~10 步：填第 5 块（0 块分配）；第 10 步填满 → 入哈希表
  ├─ 第 11~26 步：申请第 6 块并填满 → 入哈希表
  ├─ 第 27~32 步：申请第 7 块，填 6 个 slot → 输出完成（未满，不入表）

释放
  └─ free 逆序归还：第 7→6→5→4→3 块；命中块 0/1 仅减计数
     有哈希块 append 队尾（保护缓存），无哈希块 prepend 队首（优先复用）
```

> **核心结论：新块同样会被缓存**（与命中无关）。prefill 新分配的满块、decode 逐步填满的新块，行为一致——`allocate_slots` 内部总是调 `cache_full_blocks`，只要某块写满就哈希入前缀缓存表；未满的尾块等写满的当步入表。

---

## 3. 分阶段详解

### 3.0 阶段 0（启动期前传）：物理显存初始化

> 启动期**一次性**执行（`EngineCore._initialize_kv_caches`，core.py:254），产出两样供运行时消费：
> 1. `num_blocks`（1024）→ `BlockPool.__init__` 建 `KVCacheBlock(0..1023)`
> 2. `kv_caches[layer]` 物理张量 → `GPUModelRunner` 申请，§3.3 按 `block_id` 读写

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant EngineCore
    participant ModelExecutor
    participant GPUWorker
    participant GPUModelRunner
    participant kv_cache_utils

    Note over EngineCore,kv_cache_utils: 步骤1 各层产出 FullAttentionSpec
    EngineCore->>ModelExecutor: get_kv_cache_specs()（收集每层 spec）
    ModelExecutor->>GPUWorker: 遍历 attention 层 → get_kv_cache_spec()<br/>→ FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128)<br/>（即 num_key_value_heads=8、head_dim=128）
    GPUWorker-->>ModelExecutor: dict[layer, FullAttentionSpec]
    ModelExecutor-->>EngineCore: kv_cache_specs

    Note over EngineCore,kv_cache_utils: 步骤2 profile_run 测可用显存 + 算 num_blocks
    EngineCore->>ModelExecutor: determine_available_memory()（profile_run 测显存）
    Note over ModelExecutor,GPUWorker: collective_rpc → GPUWorker.profile_run()<br/>available = total×util − non_kv − cudagraph
    ModelExecutor-->>EngineCore: available_gpu_memory
    EngineCore->>kv_cache_utils: get_kv_cache_configs(...)
    Note over kv_cache_utils: num_blocks = available // page_size // num_hidden_layers<br/>（2GiB → 1024）；多 worker 取 min 对齐
    kv_cache_utils-->>EngineCore: KVCacheConfig(num_blocks=1024, ...)

    Note over EngineCore,GPUModelRunner: 步骤3 Worker 申请 + 绑定张量
    EngineCore->>ModelExecutor: initialize_from_config(kv_cache_configs)
    ModelExecutor->>GPUWorker: collective_rpc("initialize_from_config")
    GPUWorker->>GPUModelRunner: initialize_kv_cache(config)
    Note over GPUModelRunner: 4a 申请 int8 字节池 (torch.zeros)<br/>4b reshape: (1024, 8, 16, 256) 每层<br/>4c 绑定到 forward_context

    GPUModelRunner-->>GPUWorker: 完成
    GPUWorker-->>ModelExecutor: 完成
    ModelExecutor-->>EngineCore: 完成
    Note over EngineCore: Scheduler 读 num_blocks → BlockPool.__init__(1024)<br/>建 KVCacheBlock(0..1023)；kv_caches[layer] 就绪，等运行时消费
```

**要点**：物理申请只在启动期做一次；运行时 `KVCacheManager` 的"分配/释放"只操作 `block_id` 整数 + `ref_cnt`，**零显存搬运**。`block_id` 即物理张量第 0 维行号，两步由这位桥接。

### 3.1 ① 入队

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant Client
    participant LLMEngine
    participant EngineCore
    participant Scheduler
    Client->>LLMEngine: generate(prompt)
    LLMEngine->>EngineCore: add_request()
    EngineCore->>Scheduler: add_request()
    Note over Scheduler: 构造 Request，update_block_hashes()<br/>预计算链式哈希 → block_hashes
    Scheduler-->>EngineCore: 入 WAITING 队列
```

**要点**：
- 入队即预计算：70 token → `70//16=4` 个满块有 hash；未满的第 5 块无 hash
- `request.block_hashes` 存的是**纯 `BlockHash`**，group id 在 ③ 前缀查找 / ④ 分配落库时才临时拼上

### 3.2 ② 首次调度（WAITING → prefill）

`schedule()`（scheduler.py:427）每步**先遍历 RUNNING（:473）再遍历 WAITING（:671）**。KV 编排链固定为 `KVCacheManager → UnitaryKVCacheCoordinator → FullAttentionManager → BlockPool`，下面三个子步骤都走这条链。

#### 3.2.1 ③ 前缀缓存查找（get_computed_blocks）

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant Scheduler
    participant KVCacheManager
    participant UnitaryKVCacheCoordinator
    participant FullAttentionManager
    participant BlockPool
    Scheduler->>KVCacheManager: get_computed_blocks(request)
    KVCacheManager->>UnitaryKVCacheCoordinator: find_longest_cache_hit(block_hashes, max_len)
    UnitaryKVCacheCoordinator->>FullAttentionManager: find_longest_cache_hit()
    Note over FullAttentionManager: 逐块查表，遇 miss 即 break<br/>hit_length = 命中块数 × block_size
    FullAttentionManager->>BlockPool: get_cached_block(hash, [0])
    Note over BlockPool: 临时构造 BlockHashWithGroupId 作查询 key<br/>查 cached_block_hash_to_block
    BlockPool-->>FullAttentionManager: [block | None]
    FullAttentionManager-->>UnitaryKVCacheCoordinator: (hit_blocks, hit_length)
    UnitaryKVCacheCoordinator-->>KVCacheManager: (hit_blocks, hit_length, 0)
    KVCacheManager-->>Scheduler: (KVCacheBlocks, num_tokens, boundary)
```

**要点**：本次**只读不写**——临时构造查询 key，不改 `ref_cnt`。链式哈希从左到右遇 miss 即 break，命中的是已满块。示例 R 命中前 2 块 → `hit_length=32`，剩 `70−32=38` token 需重算；真正的 `ref_cnt++` 在 ④ 的 touch。

#### 3.2.2 ④ 分配物理块（allocate_slots · 内部 5 步）

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant Scheduler
    participant KVCacheManager
    participant UnitaryKVCacheCoordinator
    participant FullAttentionManager
    participant BlockPool
    Note over Scheduler,BlockPool: allocate_slots 内部 5 步，各步含义见下方要点 ①②③④⑤
    Scheduler->>KVCacheManager: allocate_slots(request)
    KVCacheManager->>UnitaryKVCacheCoordinator: remove_skipped_blocks
    KVCacheManager->>UnitaryKVCacheCoordinator: get_num_blocks_to_allocate
    KVCacheManager->>UnitaryKVCacheCoordinator: allocate_new_computed_blocks
    UnitaryKVCacheCoordinator->>FullAttentionManager: add_local_computed_blocks
    KVCacheManager->>UnitaryKVCacheCoordinator: allocate_new_blocks
    FullAttentionManager->>BlockPool: get_new_blocks
    KVCacheManager->>UnitaryKVCacheCoordinator: cache_blocks
    FullAttentionManager->>BlockPool: cache_full_blocks
    UnitaryKVCacheCoordinator-->>KVCacheManager: 完成
    KVCacheManager-->>Scheduler: KVCacheBlocks
```

**要点（5 步全走）**：
- ①②：full attention 下"释放滑窗块"恒为空操作；容量检查用 `free − reserved`，不足则 `None` → 抢占
- ③：`touch` 命中块 `ref_cnt` 1→2，并摘出 free 队列
- ④：剩余 38 token 切 `16+16+6` → `get_new_blocks(3)`；block_table 变 `[命中0,命中1,新2,新3,新4]`
- ⑤：命中块 0/1 哈希已存在（幂等早退）；真正入表的是新满块 2、3；未满块 4 不入

#### 3.2.3 ⑤ 组装 SchedulerOutput

分配完物理块后，Scheduler 还需打包两件"后处理指令"给 Worker 在 forward 前执行：**清零新块**（`new_block_ids_to_zero`，清残留数据）与 **CoW 拷贝**（部分命中请求需把共享块 KV 拷到私有块）。R 是首次 prefill（无 partial hit），`kv_cache_block_copies` 为空，3 个新块 id（2/3/4）进 `new_block_ids_to_zero`。

### 3.3 ⑥ GPU 写 KV（GPUModelRunner forward）

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant EngineCore
    participant GPUModelRunner
    participant AttentionBackend
    EngineCore->>GPUModelRunner: execute_model(sched_out)
    GPUModelRunner->>AttentionBackend: _zero_block_ids(ids)（清零本轮新块）
    GPUModelRunner->>AttentionBackend: forward(hidden_states)
    Note over AttentionBackend: prefill 写新 token KV / decode 读+写<br/>kv_caches[layer][block_id] fancy index 第0维
    AttentionBackend-->>GPUModelRunner: model_output
    GPUModelRunner-->>EngineCore: 返回
```

**要点**：attention 后端拿着每个请求的 `block_table`（一串 `block_id`）作索引，从 `kv_caches[layer][block_id]` 第 0 维 gather 对应行；同一 `block_id` 在 32 层对应同一逻辑块，全套层共用一份 block_table。示例 R：3 个新块先清零，一次 forward 写 70 token 的 K/V 到 5 块，命中块 0/1 复用不重算。

### 3.4 ⑦ decode 续写（RUNNING）

`schedule()` 每步**先遍历所有 RUNNING 请求**（:473），每请求 append 1 token，全部处理完后一次性 `execute_model + sample_tokens`。与 prefill 走**同一套** `allocate_slots`（内部 5 步），差异仅在量级：通常无前缀命中（③跳过），当前块未满则 0 块、写满则 1 块。

示例 R：prefill 后第 5 块只装 6 token。decode 第 1~9 步填第 5 块（0 分配），**第 10 步填满入表**；第 11 步申请第 6 块、第 26 步入表；第 27 步申请第 7 块、填 6 个 slot 后完成（未满不入表）。**新满块同样入缓存**——这是前缀缓存持续增长的方式。

### 3.5 ⑧ 请求结束 → 释放

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant Scheduler
    participant KVCacheManager
    participant UnitaryKVCacheCoordinator
    participant FullAttentionManager
    participant BlockPool
    Scheduler->>KVCacheManager: free(request)
    KVCacheManager->>UnitaryKVCacheCoordinator: free(req_id)
    UnitaryKVCacheCoordinator->>FullAttentionManager: free(req_id)
    FullAttentionManager->>BlockPool: free_blocks(reversed(blocks))
    Note over BlockPool: ref_cnt-- 逆序释放，归 0 才回收：<br/>有 hash → append 队尾(LRU 保护)<br/>无 hash → prepend 队首(优先复用)
    UnitaryKVCacheCoordinator-->>KVCacheManager: 完成
    KVCacheManager-->>Scheduler: 完成
```

**要点**：**逆序释放**（`reversed`）让最近用的块最先被复用；`ref_cnt>0` 的共享块仅减计数不回收。示例 R：第 7 块先归还；命中块 0/1 因仍被其他请求共享只减计数；有哈希的块 append 队尾，方便后续前缀复用。

### 3.6 扩展：抢占（容量不足时）

`allocate_slots` 返回 `None` 时反复抢占直到成功或无可抢占（scheduler.py:565 的 `while True`）：

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "boxTextMargin": 4, "messageAlign": "center", "mirrorActors": true, "height": 45}}}%%
sequenceDiagram
    autonumber
    participant Scheduler
    participant KVCacheManager
    loop while True   (反复抢占直至成功或无对象)
        Scheduler->>KVCacheManager: allocate_slots(request, ...)
        alt 返回 None（② 容量检查失败）
            KVCacheManager-->>Scheduler: None
            Note over Scheduler: 抢占 running 中最低优先级请求<br/>PRIORITY → priority+arrival_time 最小
            Scheduler->>KVCacheManager: free(preempted_req)（内部下钻见 §3.5）
            Note over Scheduler: 被抢占者回到 waiting，num_computed_tokens=0
        else 成功
            KVCacheManager-->>Scheduler: KVCacheBlocks
        end
    end
```

**要点**：抢占者被放回 waiting 且 `num_computed_tokens=0`（:1260），块已释放，重调度需重新 prefill。示例 R：设其 3 个新块放不下 → 抢占优先级最低的 X 并释放其块 → 腾出空间后重试成功，R 进入 RUNNING，X 暂停待重调度。

---

## 4. 小结：prefill 与 decode 的统一

> ② 首次 prefill 与 ⑦ decode 续写共用同一套 **`allocate_slots` 分配块 → forward 写 KV → 满块 `cache_blocks` 入哈希** 骨架，只是规模不同。**唯一的阶段差异**在前置：前缀查找 `get_computed_blocks` 是 prefill 独有的（首次带着整段 prompt 查可复用前缀），decode 跳过它（续写的是全新 token，无前缀可查）。

| 维度 | prefill（WAITING 首次） | decode（RUNNING 续写） |
|---|---|---|
| 处理 token 数 | 一次整个 prompt（70 个） | 每步 1 个 |
| 前缀查找 | 是（`get_computed_blocks`） | 否（续写无新命中） |
| 分配块数 | 一次多块（3 新块） | 0 或 1 块 |
| 内部 5 步 | ①~⑤ 全走（③ touch 命中块） | ①③ 空操作，②④⑤ 照走 |
| 状态机 | `WAITING → RUNNING` | 保持 `RUNNING` 直到完成 |

状态机全路径：`WAITING →(首次调度) RUNNING →(持续 decode) → 完成 → 释放`，与 §3.1 / §3.5 无缝衔接。