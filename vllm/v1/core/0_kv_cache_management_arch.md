# vLLM V1 KV Cache 管理机制（总览）

> 基于 `vllm/vllm/v1/core/` 源码，系统梳理 vLLM V1 架构中 KV Cache 从显存申请、逻辑建池到调度使用的完整链路。
>
> 本文是**架构总览**，按「物理 → 逻辑 → 单类型 → 协调 → 顶层」五层组织，每层细节下沉到对应子文档：

| 子文档 | 层 | 主题 |
|---|---|---|
| [`1_physical_memory.md`](./1_physical_memory.md) | 物理显存层（最底） | `KVCacheSpec`/`KVCacheConfig`/`KVCacheTensor` → `GPUModelRunner` 申请、reshape、bind |
| [`2_block_pool.md`](./2_block_pool.md) | 逻辑块池层 | `KVCacheBlock`、`FreeKVCacheBlockQueue`、哈希表、`BlockPool` |
| [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) | 单类型管理层 | `SingleTypeKVCacheManager` ABC + Full/SWA/Mamba 等 7 个子类 |
| [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md) | 协调器层 | `KVCacheCoordinator`：NoPrefix / Unitary / Hybrid，跨组命中对齐 |
| [`5_kv_cache_manager.md`](./5_kv_cache_manager.md) | 顶层接口层（最顶） | `KVCacheManager` + `KVCacheBlocks`，Scheduler 唯一入口 |

---

## 1. 绪论：为什么这样设计 KV Cache

大型语言模型的自回归推理中，KV Cache 通常占据 GPU 显存的最大头。如何高效管理这块显存，直接影响吞吐、延迟和并发能力。vLLM V1 在这块设计上遵循三条主线：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**，按块分配、回收和共享，避免内存碎片。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id`）；物理显存（`torch.Tensor`）由 `GPUModelRunner` 申请并 reshape。两者通过 `block_id` 关联，调度决策零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，把有哈希的缓存尽量保留在队尾。

---

## 2. 全景：一条请求的 KV Cache 之旅

### 2.1 请求生命周期的四个阶段

```
等待调度 (WAITING)
      │
      ▼
前缀缓存查找 (get_computed_blocks)
      │
      ▼
分配 block (allocate_slots)
      │
      ▼
计算与缓存 (forward + cache_blocks)
      │
      ▼
释放或抢占 (free / preempt)
```

### 2.2 从 token 到物理显存的数据流

当一条请求 L 进入调度器，它携带的 token 列表被划分成一个个 `block_size` 大小的 chunk。调度器不直接碰 GPU 显存，而是通过这些 chunk 的**哈希**去问 `BlockPool`：这些 block 有没有已经算好的？有则直接复用 `block_id`；没有就从空闲队列申请新的 `block_id`。

最终，每个请求在调度器侧只保存一个 `block_table`：

```python
block_table = [5, 12, 8, 33]   # 只是一组 int block_id
```

请求上 GPU 计算时，worker 根据这些 `block_id` 索引物理张量 `kv_caches[layer_name]`，把 KV 值写入/读出对应位置。不同层各自持有独立的张量，但共享同一套 `block_id`。

设计的妙处：
- **调度器做决策时只操作 `block_id` 和元数据**，不搬移显存；
- **前缀缓存命中、block 共享、驱逐都只改引用计数和空闲队列**；
- **物理张量一次申请好、reshape 好，后续直接按 `block_id` 使用**。

---

## 3. 分层架构：五层图

```
┌──────────────────────────────────────────────────────────────────┐
│                        Scheduler (调度器)                         │  ▲
├──────────────────────────────────────────────────────────────────┤  │
│            KVCacheManager + KVCacheBlocks (顶层接口)              │  │  详见 5_kv_cache_manager.md
│              对 Scheduler 暴露统一 API，隐藏内部数据结构            │  │
├──────────────────────────────────────────────────────────────────┤  │
│                  KVCacheCoordinator (协调器)                       │  │  详见 4_kv_cache_coordinator.md
│            协调多个 KV Cache Group 的缓存命中一致性                │  │
│              ┌───────────────┴──────────────┐                    │  │
│       SingleTypeKVCacheManager    SingleTypeKVCacheManager       │  │  详见 3_single_type_kv_cache_manager.md
│      (FullAttentionManager)      (SlidingWindowManager)   ...    │  │
├──────────────────────────────────────────────────────────────────┤  │
│                    BlockPool (逻辑块池)                            │  │  详见 2_block_pool.md
│     逻辑块分配、释放、缓存、驱逐（仅持 block_id，不持显存指针）     │  │
│       ┌─────────────────┴──────────────────┐                    │  │
│    FreeKVCacheBlockQueue             BlockHashToBlockMap         │  │
│     (LRU 空闲块队列)                 (前缀缓存哈希表)             │  │
├──────────────────────────────────────────────── ── ── ── ── ── ──┤  │
│           GPUModelRunner.kv_caches[layer] (物理显存层)            │  │  详见 1_physical_memory.md
│      torch.Tensor — int8 裸字节池申请，reshape 为后端形状         │  │
│   [2, num_blocks, block_size, num_kv_heads, head_size] (Full)    │  │
│        ↑ block_id 关联：attn backend 用 block_table 索引张量     │  │
└──────────────────────────────────────────────────────────────────┘  ▼
                          底层物理显存
```

**自上而下的组合关系**：

- `KVCacheManager` 持有一个 `KVCacheCoordinator`，把调用全部转发；
- `Coordinator` 持有一个共享 `BlockPool` 和一组 `SingleTypeKVCacheManager`，每个 manager 管一种注意力类型、**共用同一个 BlockPool**；
- `BlockPool` 持有全部 `KVCacheBlock`（仅 `block_id`），与 `GPUModelRunner` 的物理张量通过 `block_id` 桥接。

### 3.1 关键文件职责

| 文件 | 职责 | 层 |
|------|------|------|
| `kv_cache_manager.py` | 顶层管理器，对 Scheduler 暴露统一接口 | 顶层 → [`5`](./5_kv_cache_manager.md) |
| `kv_cache_coordinator.py` | 协调器，管理多类型 KV Cache Group 的协作 | 协调层 → [`4`](./4_kv_cache_coordinator.md) |
| `single_type_kv_cache_manager.py` | 按注意力类型（Full/SWA/Mamba 等）管理具体分配逻辑 | 单类型层 → [`3`](./3_single_type_kv_cache_manager.md) |
| `block_pool.py` | 逻辑 block 的分配、释放、缓存、驱逐 | 块池层 → [`2`](./2_block_pool.md) |
| `kv_cache_utils.py` | `KVCacheBlock`、`FreeKVCacheBlockQueue`、block hash 计算、group 划分等 | 块池 + 物理层 |
| `gpu_model_runner.py` | 物理显存申请（`torch.zeros` → reshape）并绑定到注意力层 | 物理层 → [`1`](./1_physical_memory.md) |
| `engine/core.py` | `_initialize_kv_caches` 编排整个初始化流程 | 初始化编排 |
| `kv_cache_interface.py` | `KVCacheSpec` / `KVCacheConfig` / `KVCacheGroupSpec` 定义 | 物理层 → [`1`](./1_physical_memory.md) |

---

## 4. 关键概念速览

为方便阅读后续子文档，先把术语按**职责层级**分成四组。每组都标注了"主要在哪一层管"，读者可据此建立"概念→层"的映射。

### 4.1 块实体（Block 体系）｜块池层 §2 管

块是 KV cache 的最小调度单位。这一组概念描述"块本身是什么、有多少、生命周期怎么管"。

| 术语 | 含义 |
|------|------|
| `KVCacheBlock` | 逻辑块对象，只含 `block_id` 和元数据（无显存指针） |
| `block_id` | 逻辑块全局编号 `[0, num_blocks-1]`，也是物理张量 reshape 后第 0 维的行号 |
| `block_size` | 一个 block 容纳的 token 数（决定 block_table 的粒度） |
| `num_blocks` | 每个 worker 的逻辑块总数（物理层 §1 算出，块池层 §2 据此建池） |
| `null_block` | `block_id=0` 的占位块，不可释放、不维护 ref_cnt，用于稀疏注意 / Mamba 对齐 block_table |
| `ref_cnt` | 引用计数，多少请求正在使用此 block；归零才能进空闲队列，命中前缀时自增实现零拷贝共享 |

### 4.2 请求 ↔ 块映射（Request Mapping 体系）｜单类型层 §3 管

这一组把"一条请求"和"它用到的一组 block"绑在一起。

| 术语 | 含义 |
|------|------|
| `block_table` | 请求 → `[block_id, ...]` 映射；代码上属于 `SingleTypeKVCacheManager.req_to_blocks[request_id]`，不是 `Request` 字段。forward 时 attention 算子用它作 fancy index 从物理张量 gather KV |

### 4.3 模型层分组（Group 体系）｜协调层 §4 管

异构 attention（Full / SWA / Mamba / MLA…）不能混用同一套 block_table，必须按"形状兼容"分组。

| 术语 | 含义 |
|------|------|
| `group` | 形状兼容（`block_size` 相同）、共用 block_table 与分配决策的层集合。每个 group 对应一个 `SingleTypeKVCacheManager`；所有 group 共用同一个 `BlockPool`。多 group 场景下同一条请求在每个 group 都有一份独立的 `req_to_blocks[request_id]`，由 `KVCacheCoordinator` 跨组对齐 |

### 4.4 块内容哈希（Hash 体系）｜块池层 §2 管

哈希是前缀缓存的"指纹"，用来判断两个 block 的内容是否相同、能否复用。

| 术语 | 含义 |
|------|------|
| `BlockHash` | 单个 block 的链式哈希（**组无关**，只基于 token 内容 + 父块哈希） |
| `BlockHashWithGroupId` | 带 `group_id` 的哈希，是 `BlockPool` 缓存表的真实 key。**为什么带 group_id？** 不同 group 的 block 即使 token 相同，物理布局也可能不同（如 Full vs Mamba），不能跨组复用，所以 key 必须区分 group |

### 4.5 四组概念的关系图

```
┌─────────────────────────────────────────────────────────────────┐
│  模型层分组（§4.3 Group 体系，协调层 §4 管）                      │
│  group = 形状兼容的层集合，每个 group 一个 SingleTypeKVCacheManager │
│         group_0 (Full attn)    group_1 (Mamba)    ...            │
└─────────────────────────────────────────────────────────────────┘
                              │ 共用同一个
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  块实体（§4.1 Block 体系，块池层 §2 管）                          │
│  BlockPool.blocks = [KVCacheBlock(0), KVCacheBlock(1), ...]      │
│  ↑ block_id == 物理张量第 0 维行号（§4.6 关键直觉）               │
│  生命周期：ref_cnt 跟踪 → null_block 占位 → num_blocks 定容量     │
└─────────────────────────────────────────────────────────────────┘
              ▲                              ▲
              │ 每请求一份                    │ 内容指纹
              │                              │
┌─────────────┴────────────┐    ┌────────────┴────────────────────┐
│ §4.2 请求↔块映射           │    │ §4.4 块内容哈希                  │
│ block_table =             │    │ BlockHash → BlockHashWithGroupId │
│   req_to_blocks[req_id]   │    │ （组无关内容）→（带 group 的 key）│
│ 单类型层 §3 管             │    │ 块池层 §2 管                     │
└──────────────────────────┘    └─────────────────────────────────┘
```

阅读后续子文档时，可按此图定位"这一层主要管哪组概念"：
- [`1_physical_memory.md`](./1_physical_memory.md)：物理显存如何申请、reshape，使 `block_id == 张量行号`成立
- [`2_block_pool.md`](./2_block_pool.md)：`KVCacheBlock` / `null_block` / `ref_cnt` / `BlockHash` / `BlockHashWithGroupId` 的实现
- [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)：`block_table`（`req_to_blocks`）的单组维护
- [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)：`group` 之间的跨组协调
- [`5_kv_cache_manager.md`](./5_kv_cache_manager.md)：把上述全部封装成 Scheduler 唯一入口

### 4.6 关键直觉

- 分配一个 `block_id` 的真实显存开销 = 所有层 `page_size_bytes` 之和（uniform spec 场景下所有层相同，简化为 `num_layers × page_size_bytes`；异构 attention 严格形式见 [`1_physical_memory.md` §5.3](./1_physical_memory.md)）；
- `block_table` 相当于「房间号表」，不同层各自有独立房间空间，但房间号一致；
- 前缀缓存命中 = 命中 block 的 `block_id` 被多个请求复用，`ref_cnt` 增加，**零显存拷贝**。

---

## 5. 端到端旅程：贯穿五层

把 §2 的四阶段铺到五层架构上，看清一次请求如何穿过所有子文档：

1. **系统启动（一次性）**：`EngineCore._initialize_kv_caches()` 先让各 attention 层产出 `KVCacheSpec`（物理层 §1），测可用显存 → 算 `num_blocks` → `GPUModelRunner` 申请/reshape 物理 tensor → `BlockPool` 创建 `KVCacheBlock(0..N-1)` 入空闲队列（块池层 §2）。从此物理张量就绪、逻辑块池就绪。

2. **前缀缓存查找** `get_computed_blocks`（顶层 §5）：请求进 RUNNING 前，`KVCacheManager` 委托 `Coordinator.find_longest_cache_hit`（协调层 §4），后者让各 `SingleTypeKVCacheManager` 用链式哈希查 `BlockPool`（单类型层 §3 + 块池层 §2）。命中则返回一组 `block_id`。

3. **分配** `allocate_slots`（顶层 §5）：三阶段——admission gate（`full_sequence_must_fit` + watermark）→ 触摸命中块（`touch` 使 `ref_cnt++`，零拷贝）→ 从空闲队列申请新 `block_id`。空间不足返回 `None` 表示无法调度。命中结尾落在 block 内部时触发 CoW（单类型层 §3）。

4. **计算与缓存**：上 GPU forward 后，底层 attn backend 用 `block_table` 索引物理张量（物理层 §1）。新填满的 block 由 `cache_blocks` 算链式哈希写入 `BlockPool` 的两张哈希表（块池层 §2）。

5. **释放/驱逐** `free`（顶层 §5 → 块池层 §2）：逆序释放，`ref_cnt-- `；归零块按有无哈希分流到空闲队列头/尾——无哈希优先驱逐、有哈希尽量保留（LRU 保护前缀缓存）。

---

## 6. 设计要点总结

1. **PagedAttention**：KV Cache 按 block 分页管理，避免内存碎片。
2. **逻辑-物理分离**：`BlockPool` 管逻辑块（只含 `block_id`），`GPUModelRunner` 管物理显存（`torch.Tensor`），通过 `block_id` 桥接。所有分配/释放/缓存/驱逐操作零显存拷贝。
3. **引用计数共享**：多请求命中相同前缀共享物理 block，通过 `ref_cnt` 管理；`touch()` 命中时 `ref_cnt++` 并从空闲队列移除。
4. **链式哈希前缀缓存**：前缀匹配靠哈希链实现，相同前缀 → 相同哈希；`FullAttentionManager` 从左到右扫描，遇到 miss 即 break。
5. **LRU 三层驱逐**：无哈希 `prepend_n` 队头、有哈希 `append_n` 队尾、`free()` 逆序传入使尾部 block 先释放。
6. **迭代不动点算法**：`HybridKVCacheCoordinator` 用单调递减循环取各注意力类型的缓存命中交集（协调层 §4）。
7. **两阶段分配**（修复 issue #33775）：先 touch 所有组的缓存命中块（`ref_cnt++`），再分配外部 block，防止跨组驱逐。
8. **Copy-on-Write**：部分命中结尾落在 block 内部时，检测登记 → 预留 block → `_apply_cow` 重定向（单类型层 §3）。
9. **三种 block_size 协同**：`scheduler_block_size = LCM`、`hash_block_size = GCD`、`BlockHashListWithBlockSize` 懒加载转换（协调层 §4 + 块池层 §2）。
10. **稀疏缓存保留**：SWA/Mamba 通过 `reachable_block_mask` 只缓存边界检查点。
11. **Watermark 准入控制**：仅对 WAITING/PREEMPTED + 已有 running 时生效，预留水位线防频繁抢占。
12. **事件驱动架构**：`BlockStored` / `BlockRemoved` / `AllBlocksCleared` 事件，松耦合支持 P/D 分离、KV offload。

> 阅读建议：先看 §5 端到端旅程建立全局直觉，再按 §3 五层图自底向上（1→2→3→4→5）深入每层源码细节。