# SingleTypeKVCacheManager 源码详解

源码位置：`vllm/v1/core/single_type_kv_cache_manager.py`

---

## 一、是什么

`SingleTypeKVCacheManager` 是 vLLM v1 调度器中负责**单一注意力类型 KV 缓存管理**的抽象基类（ABC）。

一个大语言模型可能包含多种不同类型的层：
- 标准全注意力层（Full Attention）
- 滑动窗口注意力层（Sliding Window Attention）
- 分块局部注意力层（Chunked Local Attention）
- Mamba/RNN 状态空间层
- 交叉注意力层（Cross Attention）

不同类型的层，KV 缓存的保留策略、前缀缓存命中逻辑、block 回收时机完全不同。`SingleTypeKVCacheManager` 为每种类型的层提供一个独立的管理器实例，各自维护自己的 block 分配、缓存命中、回收逻辑，但所有管理器共享同一个 `BlockPool`（GPU KV 缓存内存池）。

---

## 二、干什么用

### 2.1 在整体架构中的位置

KV Cache 管理分五层：物理显存层 → `BlockPool`（逻辑块池）→ **`SingleTypeKVCacheManager`（单类型管理层，本文）** → `KVCacheCoordinator`（跨组协调）→ `KVCacheManager`（顶层接口）→ Scheduler。

```
┌─────────────────────────────────────────────────────────┐
│  Scheduler (调度器)                                       │
├─────────────────────────────────────────────────────────┤
│  KVCacheManager          ← Scheduler唯一入口              │
├─────────────────────────────────────────────────────────┤
│  KVCacheCoordinator      ← 跨组对齐命中结果                │
│  ┌──────────────────┬──────────────────┬──────────────┐ │
│  │ FullAttention    │ SlidingWindow    │ Mamba        │ │  ← 本文讲这一层
│  │ Manager          │ Manager          │ Manager      │ │
│  │ (group 0)        │ (group 1)        │ (group 2)    │ │
│  └──────────────────┴──────────────────┴──────────────┘ │
│             所有manager共用同一个 BlockPool               │
├─────────────────────────────────────────────────────────┤
│  BlockPool               ← 逻辑块分配/释放/哈希表/LRU     │
├─────────────────────────────────────────────────────────┤
│  GPUModelRunner.kv_caches ← torch.Tensor物理显存         │
└─────────────────────────────────────────────────────────┘
```

`SingleTypeKVCacheManager` 位于第三层，是连接协调器和 BlockPool 的关键纽带。它的上一层 `KVCacheCoordinator` 负责跨多个 KV 组的命中对齐（因为一个模型可能同时包含 Full Attention 层、SWA 层、Mamba 层，它们各自有独立的 manager 和 block_table），下一层 `BlockPool` 只管逻辑 block 的分配释放和全局哈希表，不关心"这个 block 属于哪个请求、为什么要保留/回收"。

**为什么需要按类型拆分？** 不同注意力机制的 KV 保留策略完全不同：
- **Full Attention**：所有历史 token 的 KV 都要保留，支持细粒度前缀缓存
- **Sliding Window Attention**：只保留最近 window_size 个 token，滑出窗口的 block 要立即回收，不能缓存滑出部分的哈希
- **Mamba/RNN**：只需要最后一个循环状态，每生成一个新 token 旧状态就可以丢弃
- **Chunked Local Attention**：每个 chunk 内部做 attention，跨 chunk 的 KV 不可见

如果不拆分，用一个通用管理器处理所有类型，代码会充满 `if attention_type == ...` 分支，既难维护也容易出错。抽象基类定义统一接口，子类各自实现类型特定逻辑。

### 2.2 核心职责（结合调度流程）

一个请求从进入调度器到完成生成，会按以下顺序调用 manager 的方法：

| 调度阶段 | 调用方法 | 作用 |
|---------|---------|------|
| **1. 前缀缓存查找** | `find_longest_cache_hit()` | 用 token 哈希在 BlockPool 中查找最长已缓存前缀，返回可复用的 block 列表和命中 token 数。这是零拷贝共享的核心——命中的 block 不需要重新计算 KV |
| **2. 准入控制** | `get_num_blocks_to_allocate()` | 告诉调度器当前请求还需要多少新 block（考虑命中、回收、CoW 开销）。调度器汇总所有 manager 的需求后判断空闲 block 是否足够，不够则拒绝/抢占 |
| **3. 注册命中 block** | `add_local_computed_blocks()` | 将命中的 block 绑定到请求，增加 ref_cnt，检测并登记部分命中需要 CoW 的情况 |
| **4. 分配新 block** | `allocate_new_blocks()` | 从 BlockPool 空闲队列获取新 block，处理 CoW（把共享的部分尾 block 复制到私有块） |
| **5. 获取 Worker 执行信息** | `take_new_block_ids()` / `take_pending_cow_copies()` | Worker 前向计算前，取出需要清零的新 block ID 和需要 GPU 拷贝的 CoW 对 |
| **6. 缓存写满的 block** | `cache_blocks()` | 前向计算完成后，将写满的 block 计算链式哈希并注册到 BlockPool 哈希表，供后续请求前缀命中 |
| **7. 回收过期 block** | `remove_skipped_blocks()` | 每步结束后，对于 SWA/Mamba 等类型回收滑出窗口/过期的 block（替换为 null_block 占位） |
| **8. 级联注意力优化** | `get_num_common_prefix_blocks()` | 计算当前 batch 中 RUNNING 请求的公共前缀长度，用于 cascaded attention 共享计算 |
| **9. 请求结束** | `free()` | 请求完成/被抢占时，释放其持有的所有 block，ref_cnt 减 1，归零后放回空闲队列 |

### 2.3 实际场景举例

**场景：RAG 问答系统，多个用户同时提问**

假设系统 prompt 有 1000 token，知识库上下文有 2000 token，用户问题 50 token。10 个用户并发提问，他们共享相同的 system prompt + 知识库前缀。

1. 第一个请求到达时，`find_longest_cache_hit` 返回空（缓存为空），`get_num_blocks_to_allocate` 计算需要 `cdiv(3050, 16)=191` 个新 block
2. 分配 block → Worker 计算 prefill → `cache_blocks` 将写满的 190 个整块 + 1 个部分尾块（10 token）的哈希注册到 BlockPool
3. 第二个请求到达时，`find_longest_cache_hit` Phase 1 命中前 190 个整块（3040 token），Phase 2 探测到部分尾块命中 8 token（hash_block_size=4，10 token 内最后一个对齐点是 8 token），hit_length=3048
4. `get_num_blocks_to_allocate` 计算：需要 `cdiv(3050,16)=191` 块，已有 191 个命中块（含部分尾块），但部分命中需要 1 个 CoW 块，返回 1
5. 只新分配 1 个 CoW 块，Worker 执行 CoW 拷贝后直接从第 3049 个 token 开始计算——**3048 个 token 的 KV 完全复用，节省了 99% 的 prefill 计算量**
6. 进入 decode 阶段，所有 RUNNING 请求的前 190 个 block 的 `ref_cnt == 10`，`get_num_common_prefix_blocks` 返回 190，cascaded attention 可以在前 190 个 block 上只算一次 attention

**场景：SWA 模型长文档生成**

使用滑动窗口注意力（window_size=4096, block_size=16），用户让模型总结一本 10 万 token 的书。

1. Prefill 阶段分配了 `cdiv(100000,16)=6250` 个 block
2. Decode 生成时，每生成 16 个 token，`remove_skipped_blocks` 就回收一个滑出窗口的 block（ref_cnt 减 1，放回空闲队列）
3. 最终稳定状态：请求始终只持有 256 个 block（4096 token），不会因为生成长度增长而 OOM
4. `find_longest_cache_hit` 从右往左匹配窗口内的 block（不从左），因为窗口外的 KV 已经不存在了

这就是为什么不同类型需要不同的 manager：FullAttention 的前缀匹配从左到右、从不回收 block；SWA 从右到左匹配、持续回收；Mamba 只关心最后一个状态——策略差异巨大。


---

## 三、类继承结构

```
SingleTypeKVCacheManager (ABC)                           # 抽象基类：本文件详解
│
├── FullAttentionManager                                 # 全注意力管理器：本文件详解
│   ├── RSWAManager                                      # 参考滑动窗口注意力（继承全注意力细粒度能力）
│   └── SinkFullAttentionManager                         # 带Sink Token的全注意力（简略）
│
├── SlidingWindowManager                                 # 标准滑动窗口注意力
├── ChunkedLocalAttentionManager                         # 分块局部注意力
├── MambaManager                                         # Mamba/RNN状态空间层
└── CrossAttentionManager                                # 交叉注意力（编码器-解码器模型用）
```

**类变量开关**：
- `supports_fine_grained_hash_lookup: ClassVar[bool] = False`：是否支持块内细粒度部分命中。
  - `True`：FullAttentionManager、MambaManager
  - `False`：SlidingWindowManager、ChunkedLocalAttentionManager、CrossAttentionManager

**抽象方法**（子类必须实现）：
- `find_longest_cache_hit()`：前缀缓存命中查找逻辑
- `get_num_common_prefix_blocks()`：计算所有 RUNNING 请求的公共前缀 block 数（用于级联注意力优化）

**可重写方法**（子类按需重写，基类提供默认实现）：
- `get_num_skipped_tokens()`：返回需要跳过（回收）的前缀 token 数，默认返回 0（全注意力不回收）
- `reachable_block_mask()`：返回稀疏缓存掩码，默认返回 `None`（dense 缓存，所有块都缓存）
- `cache_blocks()`：缓存写满的 block，FullAttentionManager 重写增加了部分尾块缓存
- `remove_skipped_blocks()`：回收滑出窗口的 block，RSWAManager/MambaManager 重写
- `get_num_blocks_to_allocate()`：计算所需 block 数，MambaManager 重写 align 模式逻辑

---

## 四、SingleTypeKVCacheManager 基类详解

源码位置：`single_type_kv_cache_manager.py:36-676`

### 4.1 构造函数 `__init__`（36-127行）

#### 4.1.1 配置参数

```python
self.scheduler_block_size = scheduler_block_size  # 调度器统一对齐粒度，所有KV组block_size的LCM
self.block_size = kv_cache_spec.block_size        # 本类型物理block大小，一个block存储多少token的KV
self.dcp_world_size = dcp_world_size              # 解码上下文并行度，多GPU分片存储同一block时使用
self.pcp_world_size = pcp_world_size
if dcp_world_size > 1:
    self.block_size *= dcp_world_size             # DCP下逻辑block_size = 物理block_size × dcp_world_size
```

```python
self.kv_cache_spec = kv_cache_spec
self.block_pool = block_pool                                    # 所有管理器共享的BlockPool实例
self.enable_caching = enable_caching                            # 是否启用前缀缓存
# SWA/ChunkedLocal等回收型管理器的单请求最大准入block数，防止长prompt预留过多block导致死锁(issue#39734)
self._max_admission_blocks_per_request = max_admission_blocks_per_request
```

```python
# 只有长期保留block的类型(FullAttention/MLA/HiddenState)需要记录新blockID做memset(0)清零
# SWA/Mamba等block很快被覆盖或append-only写入，不需要清零，性能优化
self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (
    FullAttentionSpec, TQFullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec,
)
self.new_block_ids: list[int] = []  # 本调度周期新分配的block ID，Worker前向计算前统一清零
```

#### 4.1.2 核心状态映射

```python
# 【最重要的数据结构】请求ID → 该请求持有的KVCacheBlock列表
# - 值顺序即block_table顺序；请求结束时从这里查询释放哪些block
# - SWA等回收型管理器中，滑出窗口的block替换为_null_block占位
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
```

```python
# 记录每个RUNNING请求已缓存(设置了哈希)的block数量
# 作用1: 快速路径判断，此dict存在则走RUNNING快速路径
# 作用2: cache_blocks时知道从哪个block开始缓存，避免重复
self.num_cached_block: dict[str, int] = {}
```

```python
self.kv_cache_group_id = kv_cache_group_id  # 本管理器属于哪个KV缓存组(注意力层/Mamba层等)
self._null_block = block_pool.null_block    # 空block哨兵对象: is_null=True, ref_cnt=∞, 单例共用
                                            # 填充block_table中"逻辑位置存在但物理已释放"的空位
                                            # Worker看到null_block就知道这部分KV是零
```

```python
self.use_eagle = False  # EAGLE/MTP推测解码标志，协调器构造后设置；EAGLE需丢弃一个哈希粒度block重算
```

#### 4.1.3 Copy-on-Write 相关状态

```python
# _partial_hit_reqs: 记录发生部分命中的请求 {request_id: (block_idx, source_block)}
# - block_idx: 部分命中block在req_to_blocks中的索引
# - source_block: 共享的源block；add_local_computed_blocks时登记，allocate_new_blocks时处理CoW
self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}
# _pending_cow_copies: 等待GPU执行复制的(源block, 目标block)对，Worker前向计算前取出执行src→dst拷贝
# 仅FullAttention、Mamba align模式使用(SWA不支持部分命中)
self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
```

```python
# 外部KV缓存连接器(如Mooncake)的卸载队列，记录(请求ID, 组ID, block, token边界)
# 仅Mamba "align"模式使用
self._pending_partial_tail_offloads: list[tuple[str, int, KVCacheBlock, int]] = []
```

---

### 4.2 工具方法

#### 4.2.1 `_get_num_evictable_blocks`（128-130行）

```python
@classmethod
def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
    # 统计给定block列表中ref_cnt==0且不是null_block的数量
    # 【关键理解】为什么ref_cnt=0的命中block也要计入配额？
    # 请求结束block ref_cnt减到0放回空闲队列；另一请求前缀命中touch()将ref_cnt加回1，block"复活"
    # 虽不新分配cudaMalloc内存，但空闲块数量确实减少；不算会超额接纳请求导致OOM
    return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)
```

#### 4.2.2 `_has_partial_local_hit`（132-142行）

```python
def _has_partial_local_hit(
    self,
    new_computed_blocks: Sequence[KVCacheBlock],
    num_local_computed_tokens: int,
) -> bool:
    # 判断是否发生块内部分命中，两条件同时满足:
    # 1. len(new_computed_blocks)>0 至少命中一个block；2. 命中token数不是block_size整数倍(在block中间)
    # 【为什么能命中块中间？】hash_block_size比物理block_size小，链式哈希使每个哈希边界独立匹配
    # 部分命中最后一个block是共享的，必须CoW复制到新block再写入，需额外分配1个block
    return (
        len(new_computed_blocks) > 0
        and num_local_computed_tokens % self.block_size != 0
    )
```

---

### 4.3 核心方法：`get_num_blocks_to_allocate`（144-231行）

这是整个基类最核心、最复杂的方法，它决定了调度器能否接纳请求、会不会 OOM。

#### 4.3.1 方法签名

```python
def get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,                          # 总共需要多少token的位置（包括已有的）
    new_computed_blocks: Sequence[KVCacheBlock],  # 刚命中前缀缓存的block列表
    total_computed_tokens: int,               # 总共已计算的token（本地+外部）
    num_local_computed_tokens: int,           # 本地前缀缓存命中的token数
    num_tokens_main_model: int,               # 主模型token数（推测解码时减去lookahead）
    apply_admission_cap: bool = False,        # 是否应用SWA准入上限
) -> int:                                     # 返回: 需要从空闲池中新获取的block数(含ref_cnt=0块+CoW块)
```

#### 4.3.2 执行流程

```
┌─ 1. 计算理论总需要block数: num_required_blocks = cdiv(num_tokens, block_size)
│     └─ SWA类型应用准入上限裁剪
│
├─ 2. 查询请求已有block数: num_req_blocks = len(req_to_blocks.get(request_id, ()))
│
├─ 3. 快速路径: request_id in num_cached_block（RUNNING请求）
│     └─ assert new_computed_blocks为空
│     └─ 返回 max(num_required_blocks - num_req_blocks, 0)
│
└─ 4. 慢路径: 新请求/被抢占后重新调度的请求
      ├─ num_skipped_tokens = get_num_skipped_tokens(total_computed_tokens)
      ├─ num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
      ├─ num_skipped_blocks = num_skipped_tokens // block_size
      ├─ num_new_blocks = max(num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0)
      ├─ 计算窗口内命中block中ref_cnt=0的数量num_evictable_blocks
      ├─ 如果部分命中，num_new_blocks += 1（CoW需要1个额外block）
      └─ 返回 num_new_blocks + num_evictable_blocks
```

**关键行解释**：
```python
num_new_blocks = max(
    # num_skipped_blocks: 窗口外需跳过(填null)的block，不需要分配
    # num_local_computed_blocks: 已持有(前缀命中+之前分配)的block，不需要新分配
    # 两者取max: 可能重叠，哪个覆盖范围大，"不需要新block"的范围就越大
    num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0,
)
```

#### 4.3.3 场景实例（block_size = 16）

**场景1：全注意力新请求，命中3整块**
- 输入：80 token，命中3块（48 token），全注意力不跳过，无部分命中，命中块 ref_cnt>0
- 计算：`num_required_blocks=5`，`num_req_blocks=0`，`num_skipped_blocks=0`，`num_local_computed_blocks=3`
- `num_new_blocks = max(5-3,0) = 2`，`num_evictable_blocks=0`，无部分命中
- 返回：2 → 新分配2块（3命中+2新=5块，正好装80token）✓

**场景2：RUNNING请求decode生成新token**
- 输入：已有4块（50 token），生成到第65个token
- 快速路径：`num_required_blocks=cdiv(65,16)=5`，`num_req_blocks=4`
- 返回：`max(5-4,0)=1` → 新分配1块 ✓

**场景3：部分命中需要CoW**
- 输入：52 token，命中2块但第2块只命中前4个token（num_local_computed_tokens=20，20%16=4≠0）
- 计算：`num_required_blocks=4`，`num_local_computed_blocks=2`，`num_new_blocks=max(4-2,0)=2`
- `_has_partial_local_hit`返回True → `num_new_blocks += 1` → 变成3
- 返回：3（1个CoW块 + 2个新块）✓

物理布局：
```
命中: [0-15]整块(共享块0)  [16-19]前4个(共享块1!)
新写:                                        [20-51]
物理:  └──块0──┘  └──CoW新块──┘  └──新块1──┘  └──新块2──┘
```

**场景4：命中空闲队列中的block（ref_cnt=0）**
- 输入：48 token，命中3块，其中2块ref_cnt=0（上个请求刚用完，在空闲队列）
- 计算：`num_new_blocks = max(3-3,0)=0`，但`num_evictable_blocks=2`
- 返回：0+2=2 → 虽然不新分配GPU内存，但这2块从空闲队列取出，消耗配额 ✓

---

### 4.4 `add_local_computed_blocks`（232-289行）

将前缀缓存命中的 block 注册到请求的 `req_to_blocks` 中。

**执行流程**：
1. 获取请求的 block 列表 `blocks = req_to_blocks[request_id]`
2. 遍历 `new_computed_blocks`，对每个 block 调用 `block.touch()` 增加引用计数
3. 将 block append 到 `blocks` 列表
4. 如果启用了缓存，更新 `num_cached_block[request_id]` 计数
5. 如果发生部分命中（`_has_partial_local_hit`），将 `(block_idx, source_block)` 记录到 `_partial_hit_reqs` 中，等待后续 CoW 处理
6. 如果是外部缓存 block（`is_external`），标记为 external

**注意**：部分命中的最后一个 block 虽然被 append 到列表中，但它是共享源块，后续 `allocate_new_blocks` 会把它替换成 CoW 后的私有块。

---

### 4.5 `allocate_external_computed_blocks`（291-328行）

分配外部 KV 缓存（如 Mooncake 分布式缓存）的 block。给外部 KV 连接器用，本地前缀缓存不经过这个方法。逻辑与 `add_local_computed_blocks` 类似，但 block 来源是外部连接器而不是本地 BlockPool 哈希表。

---

### 4.6 `allocate_new_blocks`（330-369行）

从 BlockPool 分配新 block，并处理 CoW。

**执行流程**：
1. 如果请求在 `_partial_hit_reqs` 中有记录，先调用 `_apply_cow` 处理 CoW
2. 计算还需要多少个全新 block：`num_blocks_to_allocate - len(_pending_cow_copies中本次的)`
3. 调用 `block_pool.allocate_blocks(num_blocks_to_allocate)` 获取新 block
4. 将新 block append 到 `req_to_blocks[request_id]`
5. 如果需要清零（`_record_new_block_ids`），将 block ID 加入 `new_block_ids`

---

### 4.7 `_apply_cow`（405-425行）

执行 Copy-on-Write 替换。

1. 从 `_partial_hit_reqs` 取出 `(block_idx, source_block)`
2. 从 BlockPool 分配一个新 block 作为目标块
3. 将 `req_to_blocks[request_id][block_idx]` 替换为新分配的目标块
4. 将 `(source_block, dst_block)` 加入 `_pending_cow_copies`，等待 Worker 在 GPU 上执行内存复制
5. 源 block 保留在哈希表中供其他请求命中，目标 block 是当前请求私有的，可以安全写入

---

### 4.8 `take_*` 排水槽方法（371-403行）

这些方法在 Worker 前向计算前被调用，取出本周期需要执行的操作列表，同时清空内部队列：

- `records_new_block_ids`（property）：是否需要记录新 block ID
- `take_new_block_ids()`：取出并清空 `new_block_ids`，返回给 Worker 做 memset(0)
- `take_pending_cow_copies()`：取出并清空 `_pending_cow_copies`，返回 `(src_block_id, dst_block_id)` 列表给 Worker 做 GPU 复制
- `take_pending_partial_tail_offloads()`：取出并清空 `_pending_partial_tail_offloads`，给外部缓存连接器用

---

### 4.9 `cache_blocks`（427-477行）

将请求已经写满的 block 注册到 BlockPool 的前缀缓存哈希表中。

**执行流程**：
1. 获取请求已缓存 block 的起始索引 `start_idx = num_cached_block.get(request_id, 0)`
2. 遍历 `req_to_blocks[request_id][start_idx:]` 中从 `start_idx` 开始的 block
3. 对每个写满的 block（且不是 null、不是 external），调用 `block_pool.cache_block()` 设置哈希并加入哈希表
4. 更新 `num_cached_block[request_id]`

基类版本只缓存完整写满的 block。FullAttentionManager 重写了这个方法，在调用 `super().cache_blocks()` 之后额外缓存部分尾块。

---

### 4.10 `reachable_block_mask`（479-497行）

```python
@classmethod
def reachable_block_mask(
    cls,
    num_blocks: int,
    num_computed_tokens: int,
    has_partial_hit: bool,
    retention_interval: int | None = None,
) -> list[bool] | None:
    # 返回list[bool]掩码，表示哪些block"可达"(值得设置哈希缓存)
    # 基类默认返回None，表示dense缓存——所有block都缓存
    # SWA/Mamba等稀疏缓存类型重写，只在特定边界保留block哈希，节省哈希表内存
```

---

### 4.11 `pop_blocks_for_free` / `free`（500-527行）

- `pop_blocks_for_free(request_id)`：从 `req_to_blocks` 弹出该请求的 block 列表，清理 `num_cached_block` 和 `_partial_hit_reqs` 中的记录，返回 block 列表
- `free(request_id)`：调用 `pop_blocks_for_free` 取出 block 列表，然后调用 `block_pool.free_blocks(blocks)` 释放 block（ref_cnt 减 1，若 ref_cnt==0 则放回空闲队列）

---

### 4.12 `_remove_blocks_in_range` / `remove_skipped_blocks`（595-672行）

- `_remove_blocks_in_range(blocks, start, end)`：将 `blocks[start:end]` 范围内的 block 从请求中移除并释放，替换为 `_null_block` 占位
- `remove_skipped_blocks(num_computed_tokens)`：基类版本回收前缀跳过 block
  1. 计算 `num_skipped_tokens = get_num_skipped_tokens(num_computed_tokens)`
  2. 如果 num_skipped_tokens == 0，直接返回
  3. 计算需要回收的 block 范围 `[0, num_skipped_tokens // block_size]`
  4. 调用 `_remove_blocks_in_range` 回收这些 block
  5. 更新 `num_cached_block` 计数

SWA/RSWAManager/MambaManager 会重写这个方法实现不同的回收策略。

---

### 4.13 抽象方法（529-593行）

```python
@abstractmethod
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    # 计算指定RUNNING请求与其他所有RUNNING请求共享的公共前缀block数，用于级联注意力优化
```

```python
@classmethod
@abstractmethod
def find_longest_cache_hit(
    cls,
    block_hashes: BlockHashList,
    block_pool: BlockPool,
    kv_cache_group_id: int,
    block_size: int,
    num_lookahead_tokens: int = 0,
    drop_eagle_block: bool = False,
) -> tuple[list[KVCacheBlock], int]:
    # 前缀缓存命中查找核心方法，子类必须实现，各注意力类型命中逻辑不同:
    # - 全注意力: 从头连续匹配，支持块内细粒度部分命中
    # - 滑动窗口: 从右往左匹配窗口内连续block
    # - Mamba: 从右往左找任意一个哈希匹配的状态块
    # - Chunked Local: 只在当前chunk内从头匹配
    # 返回(computed_blocks, hit_length):
    # - computed_blocks: 命中block列表(可能含null_block保持索引对齐)
    # - hit_length: 精确命中token数
```

---

### 4.14 默认实现方法

```python
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    return 0  # 基类默认不跳过任何token(全注意力保留所有历史)，SWA/Mamba/ChunkedLocal重写
```

```python
def new_step_starts(self) -> None:
    pass  # 每步调度开始时的钩子，MambaManager用它重置align模式每步状态
```

---

## 五、FullAttentionManager 详解

源码位置：`single_type_kv_cache_manager.py:678-829`

FullAttentionManager 是最常用的管理器，标准解码器-only 模型（如 LLaMA、Qwen、GPT 等）的注意力层都使用它。它在基类基础上开启了细粒度哈希查找能力，支持**块内部分 token 命中**。

### 5.1 关键开关

```python
class FullAttentionManager(SingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup: ClassVar[bool] = True  # 开启细粒度哈希查找
    # 影响两处: 1.resolve_block_hashes保留hash_block_size粒度链式哈希(BlockHashListWithBlockSize)
    #          2.find_longest_cache_hit执行Phase2细粒度探测块内部分命中
```

### 5.2 三种块大小概念

在理解 FullAttentionManager 之前，必须分清三个"块大小"：

- `hash_block_size`（典型值 4，GCD）：哈希计算粒度，每个哈希对应 hash_block_size 个 token
- `alignment_tokens`（= hash_block_size）：调度对齐粒度，命中长度必须是它的倍数
- `block_size`（典型值 16）：GPU 内存分配粒度，一个物理 block 存储 block_size 个 token 的 KV

其中 `hash_block_size` 是模型所有层 `block_size` 的 GCD（最大公约数），`scheduler_block_size` 是所有层 `block_size` 的 LCM（最小公倍数）。例如：
- 注意力层 block_size=16，Mamba 层 block_size=8
- 则 hash_block_size=GCD(16,8)=8（实际代码中可能更小，如4），scheduler_block_size=LCM(16,8)=16

### 5.3 链式哈希机制

```
tokens:           0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19 ...
hash blocks:  [======= h0 =======][======= h1 =======][======= h2 =======][======= h3 =======][======= h4 =======] ...
phys block 0: [============================ block 0 (16 tokens) =============================]
phys block 1:                                                                                 [==== block 1 ==== ...
```

- `h_i` 的计算包含前面所有 `h_0 ~ h_{i-1}` 的内容（哈希链），因此 `h_i` 唯一标识前 `(i+1) × hash_block_size` 个 token
- 物理块 0 的整块哈希是 `h3`（因为 h3 包含 h0-h3，覆盖 0-15 token）
- 物理块 1 内，h4 覆盖 0-19 token，h5 覆盖 0-23 token，依此类推
- 这使得即使一个物理块没有写满，也可以在 hash_block_size 边界上缓存和命中

### 5.4 `find_longest_cache_hit`（681-778行）

FullAttentionManager 的前缀缓存查找分两阶段执行：**Phase 1 整块连续命中** → **Phase 2 细粒度部分命中扩展**，最后做 EAGLE 丢块和对齐裁剪。

#### 5.4.1 方法签名与参数

```python
@classmethod
def find_longest_cache_hit(
    cls,
    block_hashes: BlockHashList,                              # 输入token序列的哈希列表(已按hash_block_size粒度计算好)
    max_length: int,                                          # 最大查找token数限制(不超过prompt长度)
    kv_cache_group_ids: list[int],                            # 要查询的KV缓存组ID列表(多个层共享同一哈希时同时查找)
    block_pool: BlockPool,                                    # 全局BlockPool实例，所有管理器共享
    kv_cache_spec: KVCacheSpec,                               # 本类型KV缓存的规格(block_size等)
    drop_eagle_block: bool,                                   # EAGLE/MTP推测解码是否需要丢一个块
    alignment_tokens: int,                                    # 调度对齐粒度(=hash_block_size，细粒度模式)
    dcp_world_size: int = 1,                                  # 解码上下文并行度，多GPU分片时使用
    pcp_world_size: int = 1,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:              # 返回(各组命中block列表的元组, 命中token数)
```

返回值是元组的元组：外层元组每个元素对应一个 KV 缓存组的命中 block 列表（`kv_cache_group_ids` 中有几个组，就有几个列表），`hit_length` 是所有组共享的命中 token 数。同一个哈希前缀在不同层（如不同注意力层）可能对应不同物理 block，但哈希命中长度是一致的。

#### 5.4.2 准备阶段：block_size 与哈希聚合

```python
    assert isinstance(
        kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
    ), (
        "FullAttentionManager can only be used for full attention "
        "and chunked local attention groups"
    )                                                         # 类型守卫：FullAttentionManager只服务全注意力和分块局部注意力
    block_size = kv_cache_spec.block_size                     # 获取物理block大小(一个block存多少token的KV)
    if dcp_world_size > 1:
        block_size *= dcp_world_size                          # DCP将一个block的KV分片到多GPU，逻辑block_size=物理×dcp_world_size
    block_hashes = resolve_block_hashes(
        block_hashes,
        block_pool.hash_block_size,                           # 全局哈希粒度(所有层block_size的GCD，典型值4)
        block_size,                                           # 本类型物理block大小
        supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,  # 是否保留细粒度哈希
        alignment_tokens=alignment_tokens,                    # 调度对齐粒度
    )                                                         # 将原始hash粒度聚合为物理block粒度的哈希列表
```

`resolve_block_hashes` 负责把外部传入的 `hash_block_size` 粒度哈希列表，根据本层的 `block_size` 聚合为物理 block 粒度的哈希。如果开启了细粒度查找，原始细粒度哈希列表会保留在内部，后续 Phase 2 使用。

#### 5.4.3 细粒度模式判断

```python
    # 细粒度模式: alignment_tokens < block_size且整除，即物理block比哈希粒度大，可以探测块内部
    # 此时resolve_block_hashes保留了hash_block_size粒度的原始哈希列表，供Phase2探测块内边界
    fine_grained = (
        alignment_tokens < block_size and block_size % alignment_tokens == 0
    )
    if fine_grained:
        assert isinstance(block_hashes, Sequence)
        # 包装为BlockHashListWithBlockSize：同时持有细粒度哈希列表和物理block哈希视图
        # 迭代full_block_hashes时每次给出一个物理block对应的哈希，底层仍可通过索引访问细粒度哈希
        full_block_hashes: BlockHashList = BlockHashListWithBlockSize(
            block_hashes, alignment_tokens, block_size
        )
    else:
        full_block_hashes = block_hashes                     # 非细粒度模式(alignment==block_size)，直接用原始哈希
```

注意 `fine_grained` 是运行时判断，不是简单看类变量：
- 当 `alignment_tokens == block_size` 时（哈希粒度等于物理块大小），退化为非细粒度模式，不执行 Phase 2
- 当 `alignment_tokens < block_size` 且整除时（如 alignment=4, block_size=16），才是细粒度模式

#### 5.4.4 Phase 1：整块连续命中

```python
    # 外层tuple不可变(防止替换列表引用)，内层list可变(可append)
    # tuple中每个list对应一个KV组的命中block列表
    computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
        [] for _ in range(len(kv_cache_group_ids))
    )                                                        
    # Phase 1: 从头连续匹配完整物理block
    # 链式哈希保证：某块miss则其后所有块必定miss(子哈希包含父哈希作为前缀)
    # islice限制最多遍历max_length//block_size个块，不超过prompt长度
    for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
        # 在BlockPool哈希表中查找该哈希对应的block(支持同时查多个KV组)
        # 返回cached_block是各组block的元组，或None(任一group miss则整体miss)
        cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
        if not cached_block:
            break                                             # 遇到第一个miss就break，radix tree式前缀匹配
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)                           # computed是内层list的引用(不是副本)，append原地修改computed_blocks
    hit_length = len(computed_blocks[0]) * block_size         # 已命中的整块token数(可能为0表示0命中)
```

**核心逻辑**：
- 从第一个物理块的哈希开始，依次在 BlockPool 哈希表中查找
- 找到就 touch 增加引用计数，并将 block 加入每个 KV 组的命中列表
- 遇到第一个 miss 立即 break，因为链式哈希中子哈希包含父哈希，父块 miss 则所有后续块必然也 miss
- 此时 `computed_blocks` 可能为空（0命中），也可能包含 N 个连续命中的整块

#### 5.4.5 Phase 2：细粒度部分命中扩展（仅细粒度模式）

```python
    if fine_grained:
        assert isinstance(block_hashes, Sequence)
        scale_factor = block_size // alignment_tokens         # 每个物理block包含多少个哈希粒度(如16/4=4)
        first_partial_idx = len(computed_blocks[0]) * scale_factor  # Phase1结束位置对应的细粒度哈希索引
        # 探测上限取三者最小值:
        # 1. first_partial_idx + scale_factor - 1: 同一物理块内的最后一个细粒度哈希索引
        # 2. max_length // alignment_tokens: prompt长度限制，不超过max_length
        # 3. len(block_hashes): 哈希列表本身的长度
        max_partial_idx = min(
            first_partial_idx + scale_factor - 1,
            max_length // alignment_tokens,
            len(block_hashes),
        )
        # 从该物理块内最长边界开始，从右往左(从长到短)探测
        for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
            cached_tail = block_pool.get_cached_block(
                block_hashes[fine_idx], kv_cache_group_ids
            )
            if not cached_tail:
                continue                                      # 这个边界没命中，试更短的边界(continue而非break!)
            for computed, cached in zip(computed_blocks, cached_tail):
                computed.append(cached)                       # computed是内层list引用，append原地修改computed_blocks对应组的列表
            hit_length = (fine_idx + 1) * alignment_tokens    # 更新命中token数(精确到hash边界)
            break                                             # 找到最长命中就停止探测
```

**关键细节**：
- **探测范围限制在同一物理块内**：链式哈希保证跨块不会有部分命中（Phase 1 miss 的块其所有子哈希也必然 miss），所以只在 Phase 1 结束位置所在的那一个物理块内探测
- **从长到短探测**：`range` 步长为 -1，从最长的边界开始找，找到第一个命中的就是最长命中
- **miss 时用 `continue` 而不是 `break`**：某粒度 miss 继续试更短粒度（如 24token miss 试 20token、16token...）
- 举例（block_size=16, alignment=4, prompt=20token）：Phase1命中1整块(16token)，Phase2探测fine_idx=4对应20token，命中，hit_length=20

#### 5.4.6 EAGLE 处理与返回

```python
    if drop_eagle_block and hit_length > 0:
        # EAGLE/MTP推测解码需要重算生成点前的一个哈希粒度token以获得draft head隐状态
        # 细粒度模式丢alignment_tokens(一个哈希粒度)，非细粒度模式丢block_size(一个整块)
        hit_length -= min(alignment_tokens, block_size)
    hit_length -= hit_length % alignment_tokens               # 向下对齐到alignment_tokens边界(防御性代码)
    num_blocks = cdiv(hit_length, block_size)                 # 裁剪后实际需要的物理block数(向上取整)
    for computed in computed_blocks:
        del computed[num_blocks:]                             # EAGLE丢块后可能多了一个block，删除尾部多余block
    return computed_blocks, hit_length                        # 返回(各组命中block列表, 命中token长度)
```

**收尾逻辑**：
1. **EAGLE 丢块**：即使命中了前缀，EAGLE/MTP 也需要重算最后一个对齐粒度的 token，因为 draft head 隐状态必须在当前请求上下文中计算
2. **对齐裁剪**：确保 hit_length 是 alignment_tokens 的整数倍（防御性代码，Phase 2 命中本身就在边界上）
3. **裁剪多余 block**：EAGLE 丢块后 computed_blocks 可能多了一个 block，删除尾部保证列表长度与实际需要的 block 数一致

### 5.5 `cache_blocks` 与 `_cache_partial_tail_block`（779-819行）

FullAttentionManager 重写了 `cache_blocks`，在父类缓存完整 block 之后，额外缓存 prompt 结尾的**部分尾块**——这是细粒度命中能工作的写入侧保证。

```python
def cache_blocks(self, request, num_tokens, retention_interval=None):
    super().cache_blocks(request, num_tokens, retention_interval=retention_interval)  # 先缓存完整写满的block
    hash_block_size = self.block_pool.hash_block_size
    if self.block_size == hash_block_size:
        return  # 物理块和哈希粒度一样大，不存在部分尾块
    self._cache_partial_tail_block(request, num_tokens)
```

如果 block_size == hash_block_size，物理块和哈希粒度一样大，不存在部分尾块，直接返回。

```python
def _cache_partial_tail_block(self, request, num_tokens):
    hash_block_size = self.block_pool.hash_block_size
    boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size  # prompt内最后一个hash边界
    # 例: prompt=19→16; prompt=20→20(正好对齐); prompt=18→16
```

找到 prompt 长度内最后一个 `hash_block_size` 边界：
- prompt=19, hash_block_size=4 → boundary=16
- prompt=20, hash_block_size=4 → boundary=20（正好对齐）
- prompt=18, hash_block_size=4 → boundary=16

```python
    if boundary_tokens == 0 or boundary_tokens > num_tokens:
        return  # prompt比一个hash_block_size还短/边界在已计算token之外，跳过
    if boundary_tokens % self.block_size == 0:
        return  # 正好在物理块边界，父类已缓存整块，不重复
```

边界检查：
- boundary_tokens == 0：prompt 比一个 hash_block_size 还短，没法缓存
- boundary_tokens > num_tokens：边界在已计算 token 之外，跳过
- boundary_tokens % block_size == 0：正好在物理块边界，父类已经缓存过整块，不重复

```python
    blocks = self.req_to_blocks[request.request_id]         # 获取该请求持有的block列表(按block_table顺序)
    block_idx = boundary_tokens // self.block_size           # 计算boundary所在的物理块索引
                                                             # 例: boundary=20, block_size=16 → block_idx=1(第二个物理块)
    if block_idx >= len(blocks):
        return                                               # 越界保护: 该block还未分配/不存在，跳过
    self.block_pool.cache_partial_block(
        request=request,                                     # 请求对象，用于计算链式哈希(包含前面所有token的哈希)
        block=blocks[block_idx],                             # 要缓存的物理block(即prompt结尾的那个未满块)
        num_tokens=boundary_tokens,                          # 该block内有效token数(到hash边界为止)
        kv_cache_group_id=self.kv_cache_group_id,            # 本管理器所属KV组ID
        block_size=self.block_size,                          # 物理block大小
    )
    # cache_partial_block内部做三件事:
    # 1. 根据request的token序列和num_tokens计算boundary位置的链式哈希
    # 2. 设置block._block_hash = 该哈希, block._block_hash_num_tokens = boundary_tokens
    # 3. 将block以该哈希为key加入BlockPool的哈希表，供后续find_longest_cache_hit命中
```

调用 `block_pool.cache_partial_block` 缓存部分块：
- 设置 block 的哈希为 boundary_tokens 位置对应的链式哈希
- 标记 `block._block_hash_num_tokens = boundary_tokens`，记录有效 token 数
- 加入哈希表，这样 Phase 2 细粒度探测时就能找到它

**"块内部分 token 命中"完整闭环**：
1. **写入侧**：prompt 结束时 `_cache_partial_tail_block` 将物理块内最后一个 hash 边界的内容缓存，设置好哈希和有效长度
2. **查找侧**：`find_longest_cache_hit` Phase 2 在物理块内从长到短探测 hash 边界，找到就命中
3. **使用侧**：命中的部分 block 是共享的，`add_local_computed_blocks` 检测到部分命中 → `allocate_new_blocks` 做 CoW 复制到私有块 → Worker 执行 GPU 复制 → 后续在私有块上 append 写入新 token

### 5.6 `get_num_common_prefix_blocks`（821-829行）

**作用**：计算指定 RUNNING 请求与其他所有 RUNNING 请求共享的**公共前缀 block 数量**，用于级联注意力（Cascaded Attention）优化。当一批请求共享相同 prompt 前缀时，可以在前缀部分用一次 attention kernel 计算代替多次重复计算，显著提升 decode 吞吐量。

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    blocks = self.req_to_blocks[running_request_id]       # 获取指定RUNNING请求持有的block列表
    num_common_blocks = 0                                 # 公共前缀block计数器
    for block in blocks:                                  # 从头开始连续遍历block(前缀必须从位置0开始)
        if block.ref_cnt == len(self.req_to_blocks):      # ref_cnt == 当前RUNNING请求总数 → 该block被所有请求共享
            num_common_blocks += 1                        # 是公共前缀，计数+1
        else:
            break                                         # 遇到第一个不被所有请求共享的block就停止
                                                          # 前缀必须连续，中间断开后后面不可能再是公共前缀
    return num_common_blocks                              # 返回从头开始连续共享的block数量
```

**判断原理**：当多个请求通过前缀缓存命中同一批 block 时，这些共享 block 的 `ref_cnt` 等于持有它们的 RUNNING 请求数量（每个请求 touch 时 ref_cnt+1）。如果某个 block 的 `ref_cnt == len(self.req_to_blocks)`，说明当前所有 RUNNING 请求都引用了这个 block，即它们在该位置之前拥有完全相同的 KV 缓存。从头连续计数直到第一个不满足条件的 block，得到的就是公共前缀长度。

**使用场景**：调度器在组织 decode batch 时，如果发现一批请求有 N 个公共前缀 block，可以让这些请求在公共前缀部分共享一次 attention 计算，而不是每个请求都独立计算一次。这在多轮对话、RAG 等场景（多个请求共享 system prompt + 长 context）下效果显著。

---

## 六、其他 Manager 概览

### 6.1 RSWAManager（832-875行）

`RSWAManager(FullAttentionManager)` 继承自 FullAttentionManager。

**核心特点**：Reference Sliding Window Attention（参考滑动窗口注意力）。
- 解决普通 SWA 的问题：普通 SWA 只保留最近 window_size 个 token，prompt 前缀部分在 prefill 后就被回收，但共享前缀其实可以保留给其他请求命中
- R-SWA 不回收 prompt 前缀 block，只回收 prefill 结束位置到当前窗口起点之间的 "gap block"
- 内存占用 O(prefix_len + rswa_window)，而非 O(total_decode_length)
- 继承 FullAttentionManager，因此保留细粒度部分命中能力
- 重写 `remove_skipped_blocks` 实现 gap block 回收，配合 `rswa_mask_mod` 在 FA4 kernel 中将 gap 位置标记为不可见

### 6.2 SlidingWindowManager（878-1092行）

`SlidingWindowManager(SingleTypeKVCacheManager)` 标准滑动窗口注意力管理器。

**核心特点**：
- `supports_fine_grained_hash_lookup = False`，不支持块内细粒度命中
- `find_longest_cache_hit` 从**右往左**找（不是从左往右）：SWA 只看最近 window_size 个 token，前缀缓存只需要命中窗口内的连续 N 块。窗口外位置填 null_block 保持索引对齐
- `reachable_block_mask` 返回稀疏缓存掩码：只缓存窗口对齐边界上"可达"的 block，其他 block 不设置哈希，节省哈希表内存。支持 `retention_interval` 稀疏检查点
- `get_num_skipped_tokens` 返回 `max(0, num_computed_tokens - sliding_window + 1)`，即窗口外的 token 数
- `get_num_common_prefix_blocks` 直接返回 0（SWA 前缀位置是 null_block，无法通过 ref_cnt 判断公共前缀）

### 6.3 ChunkedLocalAttentionManager（1095-1250行）

`ChunkedLocalAttentionManager(SingleTypeKVCacheManager)` 分块局部注意力管理器。

**核心特点**：
- 每 chunk_size 个 token 做一次局部注意力，看不到更早的 chunk，类似 SWA 但窗口按 chunk 边界对齐而非滑动
- `supports_fine_grained_hash_lookup = False`
- 不支持 EAGLE、不支持 DCP/PCP
- `find_longest_cache_hit` 只在当前 chunk 内从头匹配，chunk 外全部填 null_block
- `get_num_skipped_tokens` 返回当前 chunk 之前的所有 token 数（整个 chunk 滑过就回收）
- `get_num_common_prefix_blocks` 返回 0，不支持级联注意力

### 6.4 MambaManager（1253-1745行）

`MambaManager(SingleTypeKVCacheManager)` Mamba/RNN 状态空间层管理器。

**核心特点**：
- Mamba 是 RNN 类模型，不是 Transformer 注意力，只需要最后一个循环状态就能继续生成
- `supports_fine_grained_hash_lookup = True`，支持细粒度状态对齐
- 两种模式：
  - 默认模式：类似全注意力，但每步回收旧 block，只保留最后一个状态
  - **align 模式**：特殊状态对齐模式，支持 Mooncake 等分布式 KV 传输，有部分尾块卸载逻辑
- Mamba 层使用 TP（张量并行）不使用 DCP，构造函数中 undo 了 DCP 的 block_size 缩放
- `find_longest_cache_hit` 逻辑特殊：从右往左找任意一个哈希匹配的状态块，前面填 null_block（不需要前缀连续匹配，只需最后一个状态）
- `get_num_skipped_tokens` 返回 `num_computed_tokens - 1`，只保留最后 1 个 token 对应的状态，前面全部回收
- `remove_skipped_blocks` 在 align 模式下额外回收上一步分配的状态块（RNN 每步生成新状态块，旧状态块立即回收）
- `get_num_blocks_to_allocate` 重写 align 模式分配逻辑，处理同 step 生成 block 不能互相命中的约束

### 6.5 CrossAttentionManager（1747-1808行）

交叉注意力管理器，用于编码器-解码器模型（如 T5、Whisper）。管理编码器输出的 KV 缓存，逻辑相对简单。

### 6.6 SinkFullAttentionManager（1810行起）

带 Sink Token 的全注意力管理器，继承 FullAttentionManager，在注意力开头保留若干 sink token 的 KV 不回收。

---

## 附录：核心调用时序

```
新请求到达调度器
    │
    ├─ 1. 协调器计算所有token的block hash（resolve_block_hashes）
    │
    ├─ 2. find_longest_cache_hit()  【子类实现】前缀缓存命中查找
    │       返回：命中block列表 + hit_length
    │
    ├─ 3. get_num_blocks_to_allocate()  计算所需新block数
    │       返回：需要从空闲池获取的block数
    │
    ├─ 4. 调度器检查空闲block是否足够 → 不够则驱逐/等待
    │
    ├─ 5. add_local_computed_blocks()  注册命中block，增加ref_cnt，登记部分命中
    │
    ├─ 6. (可选) allocate_external_computed_blocks()  分配外部缓存block
    │
    ├─ 7. allocate_new_blocks()  分配新block，处理CoW
    │
    ├─ 8. Worker执行前向计算
    │       ├─ take_new_block_ids() → 新block memset(0)
    │       └─ take_pending_cow_copies() → GPU侧CoW内存复制
    │
    ├─ 9. cache_blocks()  将写满的block注册到缓存哈希表
    │
    └─ 10. 每步decode/prefill结束后
            ├─ remove_skipped_blocks()  回收滑出窗口的block
            └─ 请求结束时free()  释放所有block回BlockPool
```
