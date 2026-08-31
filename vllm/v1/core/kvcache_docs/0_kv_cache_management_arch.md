# vLLM V1 KV Cache 管理机制（Full Attention 主线）

> 本文档以**纯 Full Attention 模型 Llama-3-8B（pp2tp2，4卡环境）**为主线，系统梳理 vLLM V1 架构中 KV Cache 从显存申请、逻辑建池到调度使用的完整链路。
>
> 本文属**三篇总览**之一：本架构文档讲"**层**"（静态结构）；[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) 讲"**流**"（一条请求的时序）；[`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) 讲"**形状**"（各注意力类型的 KV 字节布局）。

**五篇分层详解（自底向上，与本文档 §3 的五层一一对应）：**

| 分层文档 | 层 | 主题（Full Attention 主线） |
|---|---|---|
| [`1_physical_memory.md`](./1_physical_memory.md) | 第1层 · 物理显存层（最底） | KV 物理张量的申请、reshape，`block_id == 张量行号`的桥接关系 |
| [`2_block_pool.md`](./2_block_pool.md) | 第2层 · 逻辑块池层 | `KVCacheBlock`、空闲队列、链式哈希表、`BlockPool` 分配/释放/缓存/驱逐 |
| [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) | 第3层 · 单类型管理层 | `SingleTypeKVCacheManager` 基类 + `FullAttentionManager` 核心逻辑（前缀查找/分配/释放/CoW） |
| [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md) | 第4层 · 协调器层 | `UnitaryKVCacheCoordinator`（单 Full Attention 组直通），混合模型协调器作为扩展 |
| [`5_kv_cache_manager.md`](./5_kv_cache_manager.md) | 第5层 · 顶层接口层（最顶） | `KVCacheManager` + `KVCacheBlocks`，Scheduler 唯一入口，完整请求生命周期 |

---

## 1. 为什么需要 KV Cache 管理

大型语言模型自回归推理时，每个 token 的生成依赖之前所有 token 的 Key/Value 张量。如果每次生成都重新计算前面所有 token 的 K/V，复杂度是 O(n²)。KV Cache 把之前算好的 K/V 缓存起来，每次只算新 token，复杂度降为 O(n)，但代价是需要占用大量 GPU 显存。

vLLM V1 的 KV Cache 管理围绕六条核心设计（①②③ 是三个支柱，④⑤⑥ 是其派生保障）：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（如每个 block 存 16 个 token），按块分配、回收和共享，彻底解决内存碎片问题。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 `GPUModelRunner` 一次性申请并 reshape。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。
4. **Copy-on-Write**：部分命中（命中前缀的结尾落在 block 中间）时，为请求复制旧块内容再续写，避免覆盖其它请求仍共享的旧数据。
5. **两阶段 touch + allocate**：先对所有命中块 `ref_cnt++`（touch，防驱逐），再分配新块，避免分配过程中命中块被驱逐。
6. **Watermark 准入控制**：调度时预留一定数量的空闲块（`watermark_blocks`），防止出现频繁抢占。

> 上面 ②③ 的更多设计细节（链式哈希前缀、LRU append/prepend、逆序释放）见 [`2_block_pool.md`](./2_block_pool.md) 与 [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)。

---

## 2. 一条请求的 KV Cache 生命周期（Full Attention 模型）

### 2.1 五阶段生命周期（概览）

一条请求从入队到释放，跨过五个阶段；每阶段只会与 §3 的某几层交互。**以下是宏观流程**，逐阶段的层间调用链与源码行号见时序文档 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) §1~§3。

```
等待调度 (WAITING)  ── 请求带预计算的链式哈希入队
   │
   ▼
前缀缓存查找  ── 沿 KV 链问 BlockPool：这段前缀有没有缓存块？命中即复用
   │
   ▼
分配 slot  ── touch 命中块(ref_cnt++) + 申请新 block + 部分命中做 CoW，拼出 block_table
   │
   ▼
GPU forward  ── 注意力算子用 block_table 索引物理张量，写入本步新 token 的 K/V
   │
   ▼
缓存 / 释放 / 抢占  ── 写满的块入哈希缓存；释放 ref_cnt-- 归零块入空闲队列
```

### 2.2 数据流：从 token 到物理显存

当一条请求进入调度器，它的 token 列表按 `block_size` 分块。调度器不直接操作 GPU 显存，而是：
1. 用 token 内容算**链式哈希**，问 `BlockPool`：这些 block 有没有已缓存的？
2. 命中则直接复用 `block_id`（`ref_cnt++`，零拷贝），未命中从空闲队列申请新 `block_id`。
3. 每个请求只维护一个 `block_table = [5, 12, 8, 33]`——一组整数 `block_id`。
4. GPU forward 时，attention 算子用 `block_table` 作 fancy index，从物理 KV 张量中 gather 对应的 K/V 行。

**核心**：调度器全程只操作 `block_id`（整数），不搬移任何显存；物理张量一次性申请好后不再变动，所有分配/共享/驱逐只改引用计数和哈希表。

---

## 3. 五层架构（Full Attention 视角）

纯 Full Attention 模型只有一个 KV cache group。**五层自下而上编号**：第1层=物理显存 → 第2层=块池 → 第3层=单类型管理 → 第4层=协调器 → 第5层=顶层门面（`KVCacheManager`）。下图按"上层在下层之上"自上而下排列；`Scheduler` 是调用者、不算层。

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
│  第4层 · 协调器    KVCacheCoordinator                        │
│             本文 UnitaryKVCacheCoordinator                  │
│     · 持有 N 个第3层（每 spec group 1 个，主线 N=1）             │
│     · 持有 1 个第2层 BlockPool（所有第3层共享）                   │
├───────────────────────────────────────────────────────────────┤
│  第3层 ×N · 单类型管理  SingleTypeKVCacheManager               │
│             本文 N=1 FullAttentionManager                     │
│     前缀查找(链式哈希)/分配释放/CoW/block_table 维护              │
├───────────────────────────────────────────────────────────────┤
│  第2层 ×1 · 逻辑块池  BlockPool（唯一，所有第3层共享）             │
│     逻辑块分配/释放/缓存哈希/LRU驱逐（仅 block_id，无显存）        │
│        ┌──────────────────┴───────────────┐                   │
│    FreeKVCacheBlockQueue              BlockHashToBlockMap     │
│     (LRU 空闲块队列)                 (链式哈希→block映射)      │
├───────────────────────────────────────────────────────────────┤
│  第1层 · 物理显存  GPUModelRunner.kv_caches[layer]             │
│     torch [num_blocks, num_kv_heads, block_size, 2*head_dim]  │
│     block_id 直接索引第0维，即物理张量行号                        │
└───────────────────────────────────────────────────────────────┘
```

**文本持有关系（tree 状展开，数字 = 持有数量）**：第5层持有 **1** 个第4层 → 第4层持有 **N** 个第3层 ＋ **1** 个第2层 → 每个第3层引用**同一个**第2层。

```
Scheduler（调度器 · 调用者）
└─1→ KVCacheManager（第5层 · 顶层门面）── 对 Scheduler 暴露统一 API
    └─1→ UnitaryKVCacheCoordinator（第4层 · 协调器）
        ├─1→ 第3层 FullAttentionManager （前缀查找/分配释放/CoW/block_table 维护）
        └─1→ 第2层 BlockPool（唯一，所有第3层共享）── 仅索引 block_id
             │    FreeKVCacheBlockQueue(LRU) + BlockHashToBlockMap(链式哈希)
             └─1→ 第1层 物理显存 GPUModelRunner.kv_caches[layer]
                  （block_id == 第0维行号；BlockPool 只管行号使用权，真正读写 K/V 由注意力算子完成）
```

### 3.1 关键文件职责

| 文件 | 职责 | 层 |
|------|------|------|
| `kv_cache_manager.py` | 顶层门面，对 Scheduler 暴露统一接口（`get_computed_blocks`/`allocate_slots`/`free` 等） | 第5层 · 顶层门面 · `5_kv_cache_manager.md` |
| `kv_cache_coordinator.py` | 协调器：单组直通（Full Attention）或多组对齐（混合模型） | 第4层 · 协调器 · `4_kv_cache_coordinator.md` |
| `single_type_kv_cache_manager.py` | `FullAttentionManager`：前缀查找、block分配/释放、CoW | 第3层 · 单类型管理 · `3_single_type_kv_cache_manager.md` |
| `block_pool.py` | 逻辑 block 池：分配/释放/缓存哈希/LRU驱逐 | 第2层 · 逻辑块池 · `2_block_pool.md` |
| `kv_cache_utils.py` | `KVCacheBlock`、`BlockHash`、空闲队列、block hash计算工具 | 块池+物理层（第2/1层） |
| `gpu_model_runner.py` | 物理显存申请（`torch.zeros` → reshape）并绑定到 attention 层 | 第1层 · 物理层 · `1_physical_memory.md` |
| `kv_cache_interface.py` | `KVCacheSpec` / `KVCacheConfig` 定义 | 第1层 · 物理层 · `1_physical_memory.md` |

---

## 4. 系统初始化：五层如何装配（启动时一次性）

> 本节是**静态装配**——引擎启动时如何创建出 §3 的五层对象，不针对某条具体请求。一条请求的**动态流转**（前缀查找→分配→forward→释放的逐层调用链与源码行号）本套文档交给时序文档 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md)，本节不再重复。

### 4.1 从引擎初始化到管理层创建

调用链从引擎初始化开始：`EngineCore._initialize_kv_caches()` → `GPUWorker.initialize_from_config()` → 创建各管理层。

1. **计算 `KVCacheSpec`**：
   - 每个 attention 层调用 `get_kv_cache_spec(vllm_config)` 返回 `FullAttentionSpec`
   - 纯FullAttention模型所有层spec相同，`is_kv_cache_spec_uniform=True`，合并为1个KV cache group
   - 单token单layer的KV字节数：`kv_dim_bytes = 2 × num_kv_heads × head_size × dtype_size`（`2` for K+V）
2. **计算 `page_size`**：每个逻辑block的物理字节数 = `block_size × kv_dim_bytes`，即 `FullAttentionSpec.real_page_size_bytes`（`kv_cache_interface.py:327-342`）。这是**单层**一个 block 的字节数；模型所有层共享同一套 `block_id`，所以一个 block 跨所有层总占用 `num_layers × page_size_bytes`，但 `num_blocks` 配容量按**单层** page_size 计算
3. **计算 `num_blocks`**：`num_gpu_blocks = available_gpu_memory // page_size_bytes`（`num_blocks = raw_tensor.numel() // spec.page_size_bytes`，见 `gpu_model_runner.py:7389`），分布式下所有worker取最小值对齐
4. **申请物理KV张量**：`GPUModelRunner._allocate_kv_cache_tensors()` → `_reshape_kv_cache_tensors()` → `bind_kv_cache()`：
   - 创建Python列表 `kv_caches = []`，为每一层单独调用 `torch.zeros(...)` 申请独立张量，共 `num_layers` 张
   - 每张张量经 `_reshape_attention_kv_cache()` 按backend要求 permute 后形状为主流 `[num_blocks, num_kv_heads, block_size, 2*head_dim]`（FlashInfer/FlashAttn 默认，维度顺序由 `get_kv_cache_shape()` 决定；ROCm 等用 `[2, num_blocks, block_size, num_kv_heads, head_dim]`，各后端 logical shape 对比见 [`1_physical_memory.md`](./1_physical_memory.md) 与 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md)）
   - `bind_kv_cache()` 把所有层张量绑定到 `ModelRunner.kv_caches` 列表，`kv_caches[i]` 就是第i层的KV cache张量
   - 设计核心：同一个 `block_id=5` 在所有层都对应第5行，全局共用一份 `block_table`，不需要每层单独一份
5. **创建 `BlockPool`**：
   - 构造 `KVCacheBlock(i) for i in range(num_blocks)`，每个block的 `block_id == i`，对应物理张量第i行
   - 所有block初始放入 `free_block_queue`（`FreeKVCacheBlockQueue`双向链表），`ref_cnt=0`
   - 初始化空的 `cached_block_hash_to_block`（`BlockHashToBlockMap`，前缀缓存哈希映射）
6. **创建管理层**：
   - ① `FullAttentionManager(kv_cache_spec, block_pool, ...)`：第3层单类型管理器，持有BlockPool引用
   - ② `UnitaryKVCacheCoordinator(managers=[full_attention_manager], ...)`：第4层单组协调器，直接透传给manager
   - ③ `KVCacheManager(coordinator, block_pool, watermark_blocks, ...)`：第5层门面，Scheduler唯一交互入口

---

## 扩展：其他注意力类型概览

本文主线是最基础的 Full Attention 模型。vLLM V1 同样支持以下场景，它们在 Full Attention 基础上做扩展：

| 类型 | 代表模型 | 主要差异 | 扩展位置 |
|------|---------|---------|---------|
| **Sliding Window Attention (SWA)** | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐；前缀查找从右往左找窗口内命中 | [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)、[`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md) |
| **Mamba/SSM** | Bamba、Jamba | 无 KV 只有 state，block 存 recurrent state 而非 K/V；缓存逻辑不同 | [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) 与 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) 家族 C |
| **混合模型 (Full + SWA/Mamba)** | Gemma3、Jamba、Llama4 | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool 但 page size 必须统一 | [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)（`HybridKVCacheCoordinator`） |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | KV 低秩压缩，物理张量形状不同 | [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) 家族 B、[`1_physical_memory.md`](./1_physical_memory.md) |
| **Cross-Attention** | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)（`CrossAttentionManager`） |
| **投机解码 (EAGLE/MTP)** | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑 | [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)、[`5_kv_cache_manager.md`](./5_kv_cache_manager.md) |

阅读建议：先按本文档顺序自底向上（1→2→3→4→5）吃透 Full Attention 主线，再按需查阅对应扩展章节理解复杂场景。
