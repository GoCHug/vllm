# vLLM V1 KV Cache 管理机制（Full Attention 主线）

> 本文档以**纯 Full Attention 模型**（如 Llama、Qwen、Mistral 等经典 Decoder-only 模型）为主线，系统梳理 vLLM V1 架构中 KV Cache 从显存申请、逻辑建池到调度使用的完整链路。
>
> Sliding Window Attention、Mamba、混合模型等更复杂的场景在各文档末尾以"扩展"章节简要提及，核心逻辑仍然基于 Full Attention 框架。

## 本套文档怎么读（阅读地图）

KV Cache 文档分两层组织：**三篇 `0_` 总览**（从三个视角看同一套机制）+ **五篇分层详解**（`1`~`5`，自底向上）。

**三篇总览，分工互补，别混着看：**

| 文档 | 视角 | 解决什么问题 | 什么时候看 |
|---|---|---|---|
| **本文档** `0_kv_cache_management_arch.md` | **静态架构** | vLLM 的 KV Cache 管理长什么样：为什么需要、五层如何组织、核心概念（block / block_table / 链式哈希） | 第一本，建立全局框架 |
| [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) | **动态时序** | 一条请求从入队到释放，在 Scheduler / 各管理层 / Worker 之间如何逐步调用（含 Mermaid 时序图与源码行号） | 想把"一次运行"完整走一遍时 |
| [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) | **存储格式** | 不同 attention/SSM（Full / MLA / Mamba-GDN）的 KV cache 存什么、物理 shape 什么样、block_size / page_size_bytes 怎么算 | 研究某种模型的 KV 缓存字节布局时 |

> 一句话分工：**本文档讲"层"，时序文档讲"流"，attention 文档讲"形状"。** 三者不重复——本文档只在 §5 概述请求的五阶段流向，把逐层调用细节交给时序文档。

**五篇分层详解（自底向上，与本文档 §3 的五层一一对应）：**

| 分层文档 | 层 | 主题（Full Attention 主线） |
|---|---|---|
| [`1_physical_memory.md`](./1_physical_memory.md) | 物理显存层（最底） | KV 物理张量的申请、reshape，`block_id == 张量行号`的桥接关系 |
| [`2_block_pool.md`](./2_block_pool.md) | 逻辑块池层 | `KVCacheBlock`、空闲队列、链式哈希表、`BlockPool` 分配/释放/缓存/驱逐 |
| [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) | 单类型管理层 | `SingleTypeKVCacheManager` 基类 + `FullAttentionManager` 核心逻辑（前缀查找/分配/释放/CoW） |
| [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md) | 协调器层 | `UnitaryKVCacheCoordinator`（单 Full Attention 组直通），混合模型协调器作为扩展 |
| [`5_kv_cache_manager.md`](./5_kv_cache_manager.md) | 顶层接口层（最顶） | `KVCacheManager` + `KVCacheBlocks`，Scheduler 唯一入口，完整请求生命周期 |

---

## 1. 为什么需要 KV Cache 管理

大型语言模型自回归推理时，每个 token 的生成依赖之前所有 token 的 Key/Value 张量。如果每次生成都重新计算前面所有 token 的 K/V，复杂度是 O(n²)。KV Cache 把之前算好的 K/V 缓存起来，每次只算新 token，复杂度降为 O(n)，但代价是需要占用大量 GPU 显存。

vLLM V1 的 KV Cache 管理围绕三条核心设计：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（如每个 block 存 16 个 token），按块分配、回收和共享，彻底解决内存碎片问题。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 `GPUModelRunner` 一次性申请并 reshape。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。

---

## 2. 一条 Full Attention 请求的 KV Cache 生命周期

### 2.1 请求生命周期五阶段

```
等待调度 (WAITING)
      │
      ▼
前缀缓存查找 (get_computed_blocks)  →  链式哈希比对，返回命中 block 列表
      │
      ▼
分配 slot (allocate_slots)          →  触摸命中块(ref_cnt++) + 申请新 block + 处理部分命中CoW
      │
      ▼
GPU forward 计算                    →  attn backend 用 block_table 索引物理张量读写 KV
      │
      ▼
缓存新填满的 block (cache_blocks)   →  计算链式哈希，写入哈希表
      │
      ▼
释放/抢占 (free / preempt)          →  逆序释放，ref_cnt--，归零块入空闲队列
```

> 这五步是**请求生命周期的宏观概括**，对应时序文档的规范化阶段 **`A入队 → B调度(B1前缀查找/B2分配/B3组装) → C GPU forward → D decode续写 → E 释放 → F 抢占`**。这里的"五阶段"是粗粒度周览，逐层源码调用链与行号见 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) §3；后文 `allocate_slots` 内部子阶段用 `①②③` 表示，勿与"阶段X"混淆。

### 2.2 数据流：从 token 到物理显存

当一条请求进入调度器，它的 token 列表按 `block_size` 分块。调度器不直接操作 GPU 显存，而是：
1. 用 token 内容算**链式哈希**，问 `BlockPool`：这些 block 有没有已缓存的？
2. 命中则直接复用 `block_id`（`ref_cnt++`，零拷贝），未命中从空闲队列申请新 `block_id`。
3. 每个请求只维护一个 `block_table = [5, 12, 8, 33]`——一组整数 `block_id`。
4. GPU forward 时，attention 算子用 `block_table` 作 fancy index，从物理 KV 张量中 gather 对应的 K/V 行。

**核心直觉**：调度器全程只操作 `block_id`（整数），不搬移任何显存；物理张量一次性申请好后不再变动，所有分配/共享/驱逐只改引用计数和哈希表。

> 上面五阶段"哪一层调用哪一层"的**逐层调用链、源码行号与 Mermaid 时序图**，见 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md)；`block_hash` / `block_table` / `ref_cnt` 等实体细节见 [`2_block_pool.md`](./2_block_pool.md) 与本文 §4。

---

## 3. 五层架构（Full Attention 视角）

纯 Full Attention 模型只有一个 KV cache group，五层关系如下：

```
┌──────────────────────────────────────────────────────────────────┐
│                        Scheduler (调度器)                         │
├──────────────────────────────────────────────────────────────────┤
│            KVCacheManager + KVCacheBlocks (顶层接口)              │  见 5_kv_cache_manager.md
│              对 Scheduler 暴露统一 API，隐藏内部结构               │
├──────────────────────────────────────────────────────────────────┤
│              UnitaryKVCacheCoordinator (协调器-单组直通)          │  见 4_kv_cache_coordinator.md
│              单 Full Attention 组：直接转发给下层manager           │
├──────────────────────────────────────────────────────────────────┤
│                  FullAttentionManager (单类型管理)                │  见 3_single_type_kv_cache_manager.md
│      前缀查找(链式哈希)、分配/释放、Copy-on-Write、block_table维护  │
├──────────────────────────────────────────────────────────────────┤
│                    BlockPool (逻辑块池)                            │  见 2_block_pool.md
│     逻辑块分配/释放/缓存/驱逐（仅持 block_id，不持显存指针）        │
│       ┌─────────────────┴──────────────────┐                     │
│    FreeKVCacheBlockQueue             BlockHashToBlockMap          │
│     (LRU 空闲块队列)                 (链式哈希→block映射)          │
├──────────────────────────────────────────────────────────────────┤
│           GPUModelRunner.kv_caches[layer] (物理显存层)            │  见 1_physical_memory.md
│      torch.Tensor [num_blocks, num_kv_heads, block_size, 2*head_dim]
│        ↑ block_id 直接索引第0维：block_table[b]即张量行号          │
│        （维度顺序由 attention backend 决定，此处为主流 blocks-first）│
└──────────────────────────────────────────────────────────────────┘
                          底层物理显存
```

**自上而下的持有关系**：
- `KVCacheManager` 持有一个 `KVCacheCoordinator`（Full Attention 下是 `UnitaryKVCacheCoordinator`）
- `UnitaryKVCacheCoordinator` 持有一个 `BlockPool` 和一个 `FullAttentionManager`
- `FullAttentionManager` 持有请求到 block 的映射 `req_to_blocks`（即 block_table）
- `BlockPool` 持有全部 `KVCacheBlock`（仅 `block_id` + 元数据），与物理张量通过 `block_id` 一一对应

### 3.1 关键文件职责

| 文件 | 职责 | 层 |
|------|------|------|
| `kv_cache_manager.py` | 顶层管理器，对 Scheduler 暴露统一接口（`get_computed_blocks`/`allocate_slots`/`free` 等） | 顶层 · `5_kv_cache_manager.md` |
| `kv_cache_coordinator.py` | 协调器：单组直通（Full Attention）或多组对齐（混合模型） | 协调层 · `4_kv_cache_coordinator.md` |
| `single_type_kv_cache_manager.py` | `FullAttentionManager`：前缀查找、block分配/释放、CoW | 单类型层 · `3_single_type_kv_cache_manager.md` |
| `block_pool.py` | 逻辑 block 池：分配/释放/缓存哈希/LRU驱逐 | 块池层 · `2_block_pool.md` |
| `kv_cache_utils.py` | `KVCacheBlock`、`BlockHash`、空闲队列、block hash计算工具 | 块池 + 物理层 |
| `gpu_model_runner.py` | 物理显存申请（`torch.zeros` → reshape）并绑定到 attention 层 | 物理层 · `1_physical_memory.md` |
| `kv_cache_interface.py` | `KVCacheSpec` / `KVCacheConfig` 定义 | 物理层 · `1_physical_memory.md` |

---

## 4. 核心概念速览

### 4.1 块实体（Block）

block 是 KV cache 的最小调度单位，固定存 `block_size` 个 token。

| 术语 | 含义 |
|------|------|
| `KVCacheBlock` | 逻辑块对象，只含 `block_id` 和元数据（ref_cnt、block_hash等），**不含显存指针** |
| `block_id` | 逻辑块全局编号 `[0, num_blocks-1]`，同时也是物理 KV 张量 reshape 后第 0 维的行号——这是逻辑层与物理层桥接的关键 |
| `block_size` | 一个 block 容纳的 token 数（如16），决定 block_table 的粒度 |
| `num_blocks` | GPU 上总 block 数，由可用显存除以单个 block 物理大小算出 |
| `null_block` | `block_id=0` 的占位块，不可分配/释放，用于对齐 block_table 长度 |
| `ref_cnt` | 引用计数：多少请求正在使用此 block。新分配=1，命中前缀时自增（共享），释放时自减，归零才能回收到空闲队列 |

### 4.2 请求→块映射（block_table）

每个请求在 `FullAttentionManager.req_to_blocks[request_id]` 中维护一个 block 列表：

```python
req_to_blocks["req_abc"] = [KVCacheBlock(5), KVCacheBlock(12), KVCacheBlock(8)]
```

这就是 `block_table`——forward 时 attention 算子直接用这些 `block_id` 作为索引，从物理张量 `kv_caches[layer][block_id]` 中 gather 对应行的 K/V 数据。

### 4.3 链式哈希（Chained Hash）

前缀缓存的核心机制。每个 block 的哈希不仅依赖自身 token 内容，还依赖前一个 block 的哈希：

```
H(b0) = hash(tokens[0:block_size])
H(b1) = hash(H(b0), tokens[block_size:2*block_size])
H(b2) = hash(H(b1), tokens[2*block_size:3*block_size])
...
```

这保证了**相同前缀 → 相同哈希链**。查找时从左到右依次比对哈希，遇到 miss 即 break，返回已命中的前缀 block 列表。

### 4.4 关键直觉

- 分配一个 `block_id` 的物理意义 = 在每一层的 KV 张量上占用一行（16个token的K/V），所有层共享同一套 `block_id`
- 前缀缓存命中 = 两个请求的 block_table 里有相同的 `block_id`，指向同一物理行，`ref_cnt++`，**零显存拷贝**
- 驱逐 = 把 `ref_cnt=0` 且无哈希（或LRU最旧）的 block 从缓存中移除，放回空闲队列头部（优先重新分配）

---

## 5. 系统初始化：五层如何装配（启动时一次性）

> 本节是**静态装配**——引擎启动时如何创建出 §3 的五层对象，不针对某条具体请求。一条请求的**动态流转**（前缀查找→分配→forward→释放的逐层调用链与源码行号）本套文档交给时序文档 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md)，本节不再重复。

### 5.1 从引擎初始化到管理层创建

调用链从引擎初始化开始：`EngineCore._initialize_kv_caches()` → `GPUWorker.initialize_from_config()` → 创建各管理层。

1. **计算 `KVCacheSpec`**：
   - 每个 attention 层调用 `get_kv_cache_spec(vllm_config)` 返回 `FullAttentionSpec`
   - 纯FullAttention模型所有层spec相同，`is_kv_cache_spec_uniform=True`，合并为1个KV cache group
   - 单token单layer的KV字节数：`kv_dim_bytes = 2 × num_kv_heads × head_size × dtype_size`（`2` for K+V）
2. **计算 `page_size`**：每个逻辑block的物理字节数 = `block_size × kv_dim_bytes`，即 `FullAttentionSpec.real_page_size_bytes`（`kv_cache_interface.py:327-342`）。这是**单层**一个 block 的字节数；模型所有层共享同一套 `block_id`，所以一个 block 跨所有层总占用 `num_layers × page_size_bytes`，但 `num_blocks` 配容量按**单层** page_size 计算
3. **计算 `num_blocks`**：`num_gpu_blocks = available_gpu_memory // page_size_bytes`（`num_blocks = raw_tensor.numel() // spec.page_size_bytes`，见 `gpu_model_runner.py:7389`），分布式下所有worker取最小值对齐
4. **申请物理KV张量**：`GPUModelRunner._allocate_kv_cache_tensors()` → `_reshape_kv_cache_tensors()` → `bind_kv_cache()`：
   - 创建Python列表 `kv_caches = []`，为每一层单独调用 `torch.zeros(...)` 申请独立张量，共 `num_layers` 张
   - 每张张量经 `_reshape_attention_kv_cache()` 按backend要求 permute 后形状为主流 `[num_blocks, num_kv_heads, block_size, 2*head_dim]`（FlashInfer/FlashAttn 默认，维度顺序由 `get_kv_cache_shape()` 决定；ROCm 等用 `[2, num_blocks, block_size, num_kv_heads, head_dim]`，见总览文档 §1.3/§八 block_dim）
   - `bind_kv_cache()` 把所有层张量绑定到 `ModelRunner.kv_caches` 列表，`kv_caches[i]` 就是第i层的KV cache张量
   - 设计核心：同一个 `block_id=5` 在所有层都对应第5行，全局共用一份 `block_table`，不需要每层单独一份
5. **创建 `BlockPool`**：
   - 构造 `KVCacheBlock(i) for i in range(num_blocks)`，每个block的 `block_id == i`，对应物理张量第i行
   - 所有block初始放入 `free_block_queue`（`FreeKVCacheBlockQueue`双向链表），`ref_cnt=0`
   - 初始化空的 `cached_block_hash_to_block`（`BlockHashToBlockMap`，前缀缓存哈希映射）
6. **创建管理层**：
   - ① `FullAttentionManager(kv_cache_spec, block_pool, ...)`：单类型管理器，持有BlockPool引用
   - ② `UnitaryKVCacheCoordinator(managers=[full_attention_manager], ...)`：单组协调器，直接透传给manager
   - ③ `KVCacheManager(coordinator, block_pool, watermark_blocks, ...)`：顶层门面，Scheduler唯一交互入口

---

### 5.2 请求流转总览（逐层调用细节见时序文档）

一条请求自入队到释放，宏观上走五个阶段，各阶段在层间的调用关系（左侧箭头）如下：

```
入队 (WAITING)                                              Scheduler → (构造 Request, 预计算链式哈希)
   │
   ▼
前缀缓存查找  get_computed_blocks        Scheduler → KVCacheManager → Coordinator → Manager → BlockPool(查哈希表)
   │
   ▼
分配 slots    allocate_slots             KM → CO：remove_skipped → get_num_blocks → touch命中块 → 分配新块 → cache满块
   │
   ▼
GPU forward   execute_model              Worker：清零新块 → attention kernel 用 block_table 索引 kv_caches[layer][block_id]
   │
   ▼
释放 / 抢占   free                       KM → CO → Manager → BlockPool：逆序释放，ref_cnt-- 归0块入空闲队列
```

> 这张图的**每一层调用链、每个箭头的源码行号、以及包含 70-token 全流程的 Mermaid 时序详解**，见 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) 的 §1~§3。本文档到宏观五阶段为止，深入逐步骤请转至该文档。

## 6. 设计要点总结

1. **PagedAttention 分页**：固定大小 block 分配，彻底解决内存碎片
2. **逻辑-物理分离**：`BlockPool` 管逻辑 block_id，`GPUModelRunner` 管物理 tensor，通过 `block_id` 桥接，调度零拷贝
3. **引用计数共享**：多请求命中相同前缀时 `ref_cnt++` 共享物理 block，`ref_cnt==0` 才回收
4. **链式哈希前缀缓存**：每个 block 哈希包含父哈希，保证前缀一致性，左到右扫描遇 miss 即停
5. **LRU 驱逐策略**：所有可分配块都在 `free_block_queue` 中；有哈希的缓存块 `append` 到尾部（尽量晚分配，保护缓存命中率），无哈希的空白块 `prepend` 到头部（优先弹走分配），逆序释放保证尾块位置
6. **Copy-on-Write**：部分命中（结尾落在 block 内部）时复制旧块内容，避免覆盖共享数据
7. **两阶段 touch+allocate**：先触摸所有命中块 `ref_cnt++` 防驱逐，再分配新块，避免分配过程中命中块被驱逐
8. **Watermark 准入控制**：调度时预留一定空闲块，防止频繁抢占

---

## 扩展：其他注意力类型概览

本文主线是最基础的 Full Attention 模型。vLLM V1 同样支持以下场景，它们在 Full Attention 基础上做扩展：

| 类型 | 代表模型 | 主要差异 | 扩展位置 |
|------|---------|---------|---------|
| **Sliding Window Attention (SWA)** | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐；前缀查找从右往左找窗口内命中 | §3 扩展、§4 扩展 |
| **Mamba/SSM** | Bamba、Jamba | 无 KV 只有 state，block 存 recurrent state 而非 K/V；缓存逻辑不同 | §3 扩展 |
| **混合模型 (Full + SWA/Mamba)** | Gemma3、Jamba、Llama4 | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool 但 page size 必须统一 | §4 扩展：HybridKVCacheCoordinator |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | KV 低秩压缩，物理张量形状不同 | §1 扩展 |
| **Cross-Attention** | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | §3 扩展 |
| **投机解码 (EAGLE/MTP)** | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑 | §4 扩展 |

阅读建议：先按本文档顺序自底向上（1→2→3→4→5）吃透 Full Attention 主线，再按需查阅对应扩展章节理解复杂场景。
