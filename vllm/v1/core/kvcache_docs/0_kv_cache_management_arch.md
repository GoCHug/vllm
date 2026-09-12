# vLLM V1 KV Cache 管理机制（Full Attention 主线）

> 以**纯 Full Attention 模型 Llama-3-8B（pp2tp2，4 卡环境）**为主线，系统梳理 vLLM V1 架构中 KV Cache 从显存申请、逻辑建池到调度使用的完整链路。

**章节顺序**：§1 为什么需要 → §2 五层架构静态总览 → §3 启动时五层如何静态装配 → §4 一条请求的动态生命周期。

**五层分篇详解（自底向上，与 §2 的五层一一对应）：**

| 篇章 | 层 | 主题（Full Attention 主线） |
|---|---|---|
| 物理显存篇 | 第 1 层 · 物理显存层（最底） | KV 物理张量的申请、reshape，`block_id == 张量行号` 的桥接关系 |
| 逻辑块池篇 | 第 2 层 · 逻辑块池层 | `KVCacheBlock`、空闲队列、链式哈希表、`BlockPool` 分配/释放/缓存/驱逐 |
| 单类型管理篇 | 第 3 层 · 单类型管理层 | `SingleTypeKVCacheManager` 基类 + `FullAttentionManager` 核心逻辑（前缀查找/分配/释放） |
| 协调器篇 | 第 4 层 · 协调器层 | `UnitaryKVCacheCoordinator`（单 Full Attention 组直通），混合模型协调器作为扩展 |
| 顶层门面篇 | 第 5 层 · 顶层接口层（最顶） | `KVCacheManager` + `KVCacheBlocks`，Scheduler 唯一入口，完整请求生命周期 |

---

## 1. 为什么需要 KV Cache 管理

大型语言模型自回归推理时，每个 token 的生成依赖之前所有 token 的 Key/Value 张量。如果每次生成都重新计算前面所有 token 的 K/V，复杂度是 O(n²)。KV Cache 把之前算好的 K/V 缓存起来，每次只算新 token，复杂度降为 O(n)，但代价是需要占用大量 GPU 显存。

vLLM V1 的 KV Cache 管理围绕三条核心设计：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（如每个 block 存 16 个 token），按块分配、回收和共享，彻底解决内存碎片问题。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 `GPUModelRunner` 一次性申请并 reshape。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。

---

## 2. vLLM V1 KV Cache 管理五层架构（Full Attention 视角）

vLLM V1 的 KV Cache 管理按职责自下而上分为五层。纯 Full Attention 模型只有一个 KV cache group。**五层自下而上编号**：第 1 层=物理显存 → 第 2 层=块池 → 第 3 层=单类型管理 → 第 4 层=协调器 → 第 5 层=顶层门面（`KVCacheManager`）。下图按"上层在下层之上"自上而下排列；`Scheduler` 是调用者，不算层。这些对象在启动时按什么顺序、在哪个进程里被创建，见 §3。

```
┌───────────────────────────────────────────────────────────────┐
│  ▲ 五层架构（Full Attention 视角 · 唯一 KV cache group）        │
│  层号自下而上：第1层物理显存 … 第5层门面；下图自上而下排列           │
├───────────────────────────────────────────────────────────────┤
│  Scheduler（调度器 · 调用者）                                 │
│  只通过第5层门面统一调用，不直接触碰 KV cache 内部结构             │
├───────────────────────────────────────────────────────────────┤
│  第5层 · 顶层门面   KVCacheManager                             │
│     持有 1 个第4层；对 Scheduler 暴露统一 API                   │
├───────────────────────────────────────────────────────────────┤
│  第4层 · 协调器    KVCacheCoordinator                          │
│             主线 UnitaryKVCacheCoordinator                     │
│     · 持有 N 个第3层（每 spec group 1 个，主线 N=1）             │
│     · 持有 1 个第2层 BlockPool（所有第3层共享）                   │
├───────────────────────────────────────────────────────────────┤
│  第3层 ×N · 单类型管理  SingleTypeKVCacheManager               │
│             主线 N=1 FullAttentionManager                      │
│     前缀查找(链式哈希)/分配释放/block_table 维护                  │
├───────────────────────────────────────────────────────────────┤
│  第2层 ×1 · 逻辑块池  BlockPool（唯一，所有第3层共享）             │
│     逻辑块分配/释放/缓存哈希/LRU 驱逐（仅 block_id，无显存）       │
│        ┌──────────────────┴───────────────┐                   │
│    FreeKVCacheBlockQueue              BlockHashToBlockMap     │
│     (LRU 空闲块队列)                 (链式哈希→block映射)      │
├───────────────────────────────────────────────────────────────┤
│  第1层 · 物理显存  GPUModelRunner.kv_caches[layer]             │
│     torch [num_blocks, num_kv_heads, block_size, 2*head_dim]  │
│     block_id 直接索引第0维，即物理张量行号                        │
└───────────────────────────────────────────────────────────────┘
```

**文本关系（tree 状展开；实线箭头 ─ 上的数字 = 持有数量，虚线 ╎ = 非持有的桥接）**：第 5 层持有 **1** 个第 4 层 → 第 4 层持有 **N** 个第 3 层 ＋ **1** 个第 2 层 → 每个第 3 层引用**同一个**第 2 层。注意持有关系到第 2 层为止：第 2 层在引擎进程、第 1 层在 worker 进程，两者之间没有任何对象引用，只有 `block_id` 桥接。

```
Scheduler（调度器 · 调用者）
└─1→ KVCacheManager（第5层 · 顶层门面）── 对 Scheduler 暴露统一 API
    └─1→ UnitaryKVCacheCoordinator（第4层 · 协调器）
        ├─1→ 第3层 FullAttentionManager（前缀查找/分配释放/block_table 维护）
        └─1→ 第2层 BlockPool（唯一，所有第3层共享）── 仅索引 block_id
             │    FreeKVCacheBlockQueue(LRU) + BlockHashToBlockMap(链式哈希)
             │
             ╎ 桥接（不是持有）：逻辑侧在引擎进程、物理侧在 worker 进程
             ╎ 同一份 KVCacheConfig 下发两侧——
             ╎   物理侧：遍历 KVCacheConfig.kv_cache_tensors，按每个
             ╎           KVCacheTensor.size 字节数 torch.zeros 出 int8 张量
             ╎           （shared_by 里的多个层名（每个group中的同一层）共享同一张，主线每层一张）
             ╎   逻辑侧：new 出 num_blocks 个逻辑块 KVCacheBlock，block_id 取序号 0…n−1
             ╎ 两侧约定 block_id 从 0 编号、即物理行号（见 3.2）
             ▼
             第1层 物理显存 GPUModelRunner.kv_caches[layer]
                （真正读写 K/V 的是注意力算子，调度器从不直接碰张量）
```

### 2.1 关键文件职责

| 文件（`vllm/v1/` 下） | 职责 | 所属层 |
|------|------|------|
| `core/kv_cache_manager.py` | 顶层门面，对 Scheduler 暴露统一接口（`get_computed_blocks`/`allocate_slots`/`free` 等） | 第 5 层 · 顶层门面 |
| `core/kv_cache_coordinator.py` | 协调器：单组直通（Full Attention）或多组对齐（混合模型）；含工厂 `get_kv_cache_coordinator` | 第 4 层 · 协调器 |
| `core/single_type_kv_cache_manager.py` | `FullAttentionManager`：前缀查找、block 分配/释放；含工厂 `get_manager_for_kv_cache_spec` | 第 3 层 · 单类型管理 |
| `core/block_pool.py` | 逻辑 block 池：分配/释放/缓存哈希/LRU 驱逐 | 第 2 层 · 逻辑块池 |
| `core/kv_cache_utils.py` | `KVCacheBlock`、`BlockHash`、空闲队列、block hash 计算工具；容量规划 `get_num_blocks`/`get_kv_cache_configs` | 第 2/1 层 |
| `worker/gpu_model_runner.py` | 物理显存申请（int8 字节缓冲 → reshape 零拷贝视图）并绑定到 attention 层 | 第 1 层 · 物理层 |
| `kv_cache_interface.py` | `KVCacheSpec` / `FullAttentionSpec` / `KVCacheConfig` 定义（含 `real_page_size_bytes`） | 第 1 层 · 物理层 |

---

## 3. 系统启动期：五层初始化

> 本节只讲**装配主线**：五层对象按什么顺序、在哪个进程里被创建，彼此靠什么衔接。物理张量申请的逐步调用链、代码片段、reshape 细节与 PP/TP 物理分布属于"物理显存篇"，块池内部结构属于"逻辑块池篇"，本节均不展开。

### 3.1 装配全景：规格先行，物理先于逻辑

启动装配是一条单向流水线，横跨**引擎进程**与 **worker 进程**：①～③ 在引擎进程完成"规格推导"（全程不碰显存），④ RPC 到每卡完成"物理落地"，RPC 返回后 ⑤ 才在引擎进程创建 Scheduler 及其逻辑三层。入口为 `EngineCore._initialize_kv_caches()`（`engine/core.py:248`）。

```
引擎进程                                                        worker 进程（每卡）
① 算规格  get_kv_cache_specs()          ── RPC ──▶  每层 attention 返回一个 FullAttentionSpec
② 测预算  determine_available_memory()  ── RPC ──▶  profile_run 实测本卡 KV 可用显存
③ 做编排  get_kv_cache_configs()：同规格层合并为 1 个 group
          → 算 num_blocks → 多卡取 min 对齐，产出 KVCacheConfig
④ 落张量  initialize_from_config()      ── RPC ──▶  按字节申请物理张量 → view 成后端 shape
          （engine/core.py:329）                    → bind 到各 attention 层
                                                    （gpu_worker.py:649 → gpu_model_runner.py:7606）
⑤ 建逻辑层 Scheduler(...)（engine/core.py:158）
          └ KVCacheManager（sched/scheduler.py:271）→ Coordinator → BlockPool + FullAttentionManager
```

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ① 算规格 | 每层一个 `FullAttentionSpec` | 纯 Full Attention 所有层规格一致，合并为全模型**唯一 KV cache group** |
| ② 测预算 | 每卡一个可用显存字节数 | dummy forward 实测：总显存 × 利用率 − 权重 − 激活 − CUDAGraph 预留 |
| ③ 做编排 | `KVCacheConfig`（含 `num_blocks`） | 单层单块字节 `page_size = 2 × block_size × num_kv_heads × head_dim × dtype_size`（系数 2 对应 K+V）；本卡 `num_blocks = 可用显存 // page_size // 本 worker 层数`；多卡取最小值，共享同一套 `block_id` 空间 |
| ④ 落张量 | 每 worker 每层 1 张物理张量（PP 下只含本 worker 负责的层） | 先按字节数申请、再零拷贝 view 成后端要求的 shape；同一个行号在所有层语义相同 |
| ⑤ 建逻辑层 | 第 2～5 层全部对象 | 构造顺序见 3.2 |

### 3.2 两个衔接点（理解后续章节的关键）

**衔接点 1：物理 ↔ 逻辑——同一份配置定容量，`block_id` 即行号的约定做桥接**

引擎算出 `num_blocks` 后写入 `KVCacheConfig` 并下发两侧，两侧各自做自己的事：

- **物理侧（worker，只按配置执行，不自行算块数）**
  - 遍历 `KVCacheConfig.kv_cache_tensors`；
  - 对每个 `KVCacheTensor` 按其 `size`（普通 layout 下恰为 `num_blocks × page_size` 字节）调 `torch.zeros(..., dtype=torch.int8)`，得到一张扁平字节张量；
  - 该条目 `shared_by` 列出的层名都指向这同一张张量——纯 Full Attention 主线无层共享，恰好每层一张。
  - 源码：`kv_cache_interface.py:926-932`（配置定义）、`gpu_model_runner.py:7303-7320`（申请执行）。
- **逻辑侧（引擎进程的 BlockPool，只建元数据对象）**
  - new 出 `num_blocks` 个 `KVCacheBlock`，`block_id` 取序号 `0 … n−1`；
  - 每个对象只带 `block_id`、引用计数等元数据，**不含任何 K/V 数据**。
  - 源码：`block_pool.py:175-177`。
- **桥接结果**：同一份配置保证两侧容量相等，"从 0 顺序编号"的约定让 `block_id` 直接等于物理张量行号——无需查表或拷贝。此后物理张量不再变动，分配/共享/驱逐只改引用计数和哈希表。
  - 特例：`block_id=0` 开池即留作 `null_block` 占位，不维护引用计数、不可分配，实际可分配块为 `n−1`。

**衔接点 2：逻辑层内部——代码上自上而下构造，持有关系自下而上**

Scheduler 只 new 第 5 层，下层对象在各自构造函数中级联创建：

1. `Scheduler` 构造第 5 层 `KVCacheManager`（`sched/scheduler.py:271`）；
2. 其 `__init__` 调工厂 `get_kv_cache_coordinator()` 按 group 数选型——单 group 即 `UnitaryKVCacheCoordinator`（第 4 层）；
3. 协调器构造函数内**先**建唯一的 `BlockPool`（第 2 层），**再**为每个 group 建一个 manager（主线 1 个 `FullAttentionManager`，第 3 层），各 manager 引用同一个 `BlockPool`。

最终的持有树形如 §2 所示。

---

## 4. 系统运行期：一条请求的 KV Cache 生命周期

### 4.1 请求状态机与固定编排链

一条请求的状态迁移很简单：`WAITING`（等待首次调度）→ `RUNNING`（prefill 完成后持续 decode）→ 生成结束 → 释放。
整个生命周期中，逻辑层的调用路径始终是同一条固定链路：
```
Scheduler ─▶ 第5层 KVCacheManager ─▶ 第4层 Coordinator ─▶ 第3层 FullAttentionManager ─▶ 第2层 BlockPool
                                                                       （第1层物理张量不在此链上）
```

第 1 层物理张量**从不出现在编排链里**：逻辑链只产出整数 `block_id` 列表（`block_table`），真正读写 K/V 的是 GPU forward 时的 attention 算子。

### 4.2 五阶段 × 五层职责

`EngineCore` 每个调度步执行 `schedule → execute_model → sample`。请求在这条节拍下依次经历五个阶段，每个阶段只动用 §2 中的特定几层：

```
 WAITING ── 入队：token 切块并预计算链式哈希（此时不触碰任何管理对象）
    │
    │ 首次调度（prefill）
    ▼
 前缀查找 ── 只读：沿哈希链问 BlockPool，命中仅"标记可复用"，不改引用计数
    ▼
 分配 slot ── touch 命中块(ref_cnt++) ＋ 弹空闲块，拼出 block_table；已满块顺带记入哈希
    ▼
 GPU 执行 ── 新块清零 → 算子按 block_table 索引物理张量，读旧 K/V、写本步新 K/V
    │
    ▼
 RUNNING ── decode 循环：每步重走「分配 slot → GPU 执行」（不再查前缀，每步只新增 0 或 1 块）
    │
    │ EOS / max_tokens
    ▼
 释放 ── 逆序归还、ref_cnt--；归零才回收（带哈希的块入队尾受保护，无哈希的块入队首优先复用）
```

| 阶段 | 发生时机 | 第 2～5 层（逻辑） | 第 1 层（物理） |
|------|----------|-------------------|----------------|
| 入队 | WAITING | 不创建管理对象；`Request` 只持有按满块预计算的链式哈希 | 无 |
| 前缀查找 | 仅 prefill 一次 | 沿编排链只读哈希表，命中块**只标记**，`ref_cnt` 不变 | 无 |
| 分配 slot | prefill 与每步 decode | touch 命中块（`ref_cnt++`、摘出空闲队列）→ 弹空闲块得新 `block_id` → 组成 `block_table`；写满的块在这一步顺带写入哈希表 | 无；仅登记"待清零新块"清单 |
| GPU 执行 | 每调度步 | 不参与；`SchedulerOutput` 把 `block_table` 下发给 worker | 先清零本轮新块，再由 attention 算子以 `block_table` 为 fancy index gather 物理行、写入新 K/V |
| 释放 | 生成结束 | 逆序 `ref_cnt--`；仍被共享的块只减计数，归零才回收 | 无 |

### 4.3 三个架构级结论

1. **逻辑与物理只有一个接口——`block_table`**。它就是 `[5, 12, 8, 33]` 这样一组整数 `block_id`：逻辑链用它完成全部分配/共享/驱逐，物理算子用它 gather 张量行。调度全程零显存拷贝，物理张量自 §3 申请后不再变动。
2. **prefill 与 decode 共用同一套骨架**：二者都是"分配 slot → GPU 执行"，decode 只是规模变小（每步 1 个新 token、新增 0/1 块）并**跳过前缀查找**。因此系统里没有独立的 prefill/decode 两套管理逻辑，也没有全局阶段切换；调度器在一个共享 token 预算下按"先 RUNNING、后 WAITING"逐请求填充。
3. **块的命运由引用计数与空闲队列位置共同决定**：命中前缀 → `touch`（计数加一、移出空闲队列）实现零拷贝共享；请求结束 → 计数减一，仍被别的请求引用则不回收；归零回收时，**带哈希的块排到队尾受 LRU 保护、无哈希的块排到队首被优先复用**——这就是前缀缓存能跨请求持续积累并被后续请求复用的机制。

---

## 扩展：其他注意力类型概览

本文主线是最基础的 Full Attention 模型。vLLM V1 同样支持以下场景，它们都在上述五层骨架上做扩展：

| 类型 | 代表模型 | 主要差异 | 扩展落点（代码层） |
|------|---------|---------|---------|
| **Sliding Window Attention (SWA)** | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐；前缀查找从右往左找窗口内命中 | 第 3 层滑窗管理器、第 4 层协调器 |
| **Mamba/SSM** | Bamba、Jamba | 无 KV 只有 state，block 存 recurrent state 而非 K/V；缓存逻辑不同 | 第 3 层状态管理器，第 1 层张量布局随之改变 |
| **混合模型 (Full + SWA/Mamba)** | Gemma3、Jamba、Llama4 | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool 但 page size 必须统一 | 第 4 层 `HybridKVCacheCoordinator` |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | KV 低秩压缩，物理张量形状不同 | 第 1 层 MLA 规格与张量布局 |
| **Cross-Attention** | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | 第 3 层 `CrossAttentionManager` |
| **投机解码 (EAGLE/MTP)** | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑 | 第 4 层协调器与第 5 层门面 |

阅读建议：按 §2 的层号自底向上（第 1 层物理显存 → 第 2 层块池 → 第 3 层单类型管理 → 第 4 层协调器 → 第 5 层门面）吃透 Full Attention 主线，再对照本表按需理解复杂场景。
