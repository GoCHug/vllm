# vLLM V1 KV Cache 管理机制详解

> 基于源码 `vllm/vllm/v1/core/` 目录，深入解析 vLLM V1 架构中 KV Cache 的管理机制。

---

## 1. 整体架构概览

vLLM V1 的 KV Cache 管理采用**分层设计**，核心由以下几个组件构成：

```
┌──────────────────────────────────────────────────────────┐
│                     Scheduler                             │
│                        │                                  │
│                KVCacheManager                             │
│         (对外统一接口, 隐藏内部结构)                        │
│                        │                                  │
│              KVCacheCoordinator                           │
│      (协调多个 KV Cache Group 的协作)                       │
│           ┌────────┴────────┐                             │
│    SingleTypeKVCacheManager  SingleTypeKVCacheManager     │
│   (FullAttentionManager)    (SlidingWindowManager)  ...  │
│              │                   │                        │
│           BlockPool (共享的底层块池)                        │
│        ┌─────┴──────┐                                   │
│   FreeKVCacheBlock   BlockHashToBlockMap                  │
│    Queue (LRU)         (前缀缓存哈希表)                    │
└──────────────────────────────────────────────────────────┘
```

### 核心文件职责

| 文件                                | 职责                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| `kv_cache_manager.py`             | **顶层管理器**，对 Scheduler 暴露统一接口                                      |
| `kv_cache_coordinator.py`         | **协调器**，管理多类型 KV Cache Group 的协作                                   |
| `block_pool.py`                   | **块池**，底层物理 block 的分配、释放、缓存、驱逐                              |
| `kv_cache_utils.py`               | **核心数据结构**：`KVCacheBlock`、`FreeKVCacheBlockQueue`、block hash 计算 |
| `single_type_kv_cache_manager.py` | **单类型管理器**，按注意力类型(Full/SWA/Chunked等)管理分配逻辑                 |
| `kv_cache_metrics.py`             | **指标收集**，采样跟踪 block 生命周期指标                                      |

---

## 2. 核心数据结构

### 2.1 KVCacheBlock — 最小管理单元

定义于 `kv_cache_utils.py:118`，是 KV Cache 的最小分配单位：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 物理块 ID [0, num_gpu_blocks-1]
    ref_cnt: int = 0       # 引用计数，跟踪有多少请求共享此 block
    _block_hash: BlockHashWithGroupId | None = None  # 前缀缓存哈希（仅满块有）
    _block_hash_num_tokens: int | None = None         # 哈希覆盖的 token 数
    prev_free_block: KVCacheBlock | None = None  # 双向链表前驱
    next_free_block: KVCacheBlock | None = None  # 双向链表后继
    is_null: bool = False  # null block 占位符（滑窗/分块中跳过的块）
```

**关键设计：**

- `ref_cnt`：引用计数法实现 block 共享。多个请求命中同一前缀时共享同一物理 block，仅当 `ref_cnt` 归零才回收到空闲队列。
- `_block_hash`：block 的内容哈希，仅当 block **填满** 时才设置，用于前缀缓存查找。
- 双向链表指针：用于 `FreeKVCacheBlockQueue` 中 O(1) 的插入/删除操作。

### 2.2 FreeKVCacheBlockQueue — LRU 空闲块队列

定义于 `kv_cache_utils.py:179`，使用双向链表实现 LRU 空闲块队列：

```
[fake_head] -> [block_3] -> [block_7] -> [block_1] -> [fake_tail]
                 ↑ 最旧(最先被驱逐)              ↑ 最新(最后被驱逐)
```

**驱逐顺序规则：**

1. **LRU 优先**：最近最少使用的 block 在队头，优先被驱逐。
2. **同批逆序释放**：释放请求占用的 blocks 时按逆序释放，使同一请求中**尾部 block（更多 hash tokens）排在队头**，优先被驱逐。这通过 `free()` 中的 `reversed()` 实现（`single_type_kv_cache_manager.py:409`）。
3. **无哈希优先驱逐**：没有 block_hash 的 block 通过 `prepend_n` 插到队头之前，比有哈希的 block 更先被驱逐（`block_pool.py:633-635`）。

**关键方法：**

- `popleft()` / `popleft_n(n)`：从队头弹出 block（分配时使用）
- `append(block)` / `append_n(blocks)`：回收到队尾
- `remove(block)`：O(1) 从中间移除（被 touch 命中时使用）
- `prepend_n(blocks)`：插到队头（无哈希 block 优先驱逐）

### 2.3 BlockHashToBlockMap — 前缀缓存哈希表

定义于 `block_pool.py:34`，维护 `BlockHashWithGroupId -> KVCacheBlock` 的映射：

```python
class BlockHashToBlockMap:
    self._cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

**支持两种查找模式：**

- **单映射**：一个 hash 对应一个 block（常见情况）
- **多映射**：一个 hash 对应多个 block（partial caching 场景，通过 `dict[int, KVCacheBlock]` 存储）

### 2.4 BlockHash 与哈希链

```python
BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)
```

**哈希链机制**（`kv_cache_utils.py:577`）：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    return BlockHash(hash_function((parent_block_hash, curr_block_token_ids, extra_keys)))
```

每个 block 的哈希 = `hash(父block哈希 + 当前block的token_ids + 额外键)`，形成**链式哈希**。这意味着：

- 相同的前缀必然产生相同的哈希链
- 任何 token 的改变都会导致后续所有 block 哈希不同
- 天然支持前缀缓存匹配

**额外键**（`extra_keys`）来源：

- **多模态输入**：`(mm_identifier, offset)` 确保不同位置的相同 MM item 不会冲突
- **LoRA**：LoRA name 隔离不同 adapter
- **cache_salt**：用户自定义盐值
- **prompt_embeds**：prompt 嵌入的哈希

---

## 3. 分层管理架构

### 3.1 KVCacheManager — 顶层接口

定义于 `kv_cache_manager.py:110`，是 Scheduler 唯一直接交互的对象：

```python
class KVCacheManager:
    def __init__(self, kv_cache_config, max_model_len, scheduler_block_size,
                 hash_block_size, ...):
        self.coordinator = get_kv_cache_coordinator(...)
        self.block_pool = self.coordinator.block_pool
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
```

**核心方法：**

| 方法                                             | 功能                                         |
| ------------------------------------------------ | -------------------------------------------- |
| `get_computed_blocks(request)`                 | 查找请求的前缀缓存命中                       |
| `allocate_slots(request, num_new_tokens, ...)` | 为请求分配新 block 槽位                      |
| `free(request)`                                | 释放请求占用的所有 block                     |
| `cache_blocks(request, num_tokens)`            | 将已计算的 block 存入前缀缓存                |
| `take_new_block_ids()`                         | 获取新分配的 block ID（用于 GPU 端 zeroing） |
| `get_num_common_prefix_blocks()`               | 计算共享前缀长度（用于 cascade attention）   |

### 3.2 KVCacheCoordinator — 多类型协调器

定义于 `kv_cache_coordinator.py:61`，根据模型注意力类型选择不同的协调器：

```
get_kv_cache_coordinator()
    ├── enable_caching=False → KVCacheCoordinatorNoPrefixCache
    ├── 单 KV Cache Group   → UnitaryKVCacheCoordinator
    └── 多 KV Cache Group   → HybridKVCacheCoordinator
```

**三种实现：**

#### (1) KVCacheCoordinatorNoPrefixCache

禁用前缀缓存时使用。`find_longest_cache_hit()` 直接返回空，不执行任何缓存查找。支持任意数量的 KV Cache Group（包括 0 个）。

#### (2) UnitaryKVCacheCoordinator

单一注意力类型模型使用（如全部 Full Attention）。直接委托给唯一的 `SingleTypeKVCacheManager`。

#### (3) HybridKVCacheCoordinator

混合注意力模型使用（如 Full Attention + Sliding Window）。核心是 `find_longest_cache_hit()` 使用**迭代不动点算法**：

```python
while True:
    curr_hit_length = hit_length
    for spec, group_ids, manager_cls, use_eagle in attention_groups:
        # 每种注意力类型要么接受当前长度，要么缩减它
        hit_blocks = manager_cls.find_longest_cache_hit(...)
        curr_hit_length = len(hit_blocks[0]) * spec.block_size
    if curr_hit_length >= hit_length:
        break  # 收敛
    hit_length = curr_hit_length
```

**算法原理：** 不同注意力类型的缓存命中长度可能不同（如 Full Attention 可以命中更多 block，但 Sliding Window 由于窗口限制只能命中尾部 block）。取所有类型的**交集长度**作为最终命中长度，迭代直到收敛。

### 3.3 SingleTypeKVCacheManager — 单类型管理器

定义于 `single_type_kv_cache_manager.py:32`，按注意力类型管理 block 分配逻辑。主要子类：

| 管理器                           | 对应 Spec                     | 特点                                  |
| -------------------------------- | ----------------------------- | ------------------------------------- |
| `FullAttentionManager`         | `FullAttentionSpec`         | 全注意力，所有 token 都需要 KV Cache  |
| `SlidingWindowManager`         | `SlidingWindowSpec`         | 滑动窗口，窗口外的 block 用 null 填充 |
| `ChunkedLocalAttentionManager` | `ChunkedLocalAttentionSpec` | 分块局部注意力                        |
| `MambaManager`                 | `MambaSpec`                 | Mamba 状态空间模型                    |
| `CrossAttentionManager`        | `CrossAttentionSpec`        | 交叉注意力（编码器-解码器）           |

**每个请求的 block 跟踪：**

```python
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
self.num_cached_block: dict[str, int] = {}  # 已缓存的 block 数
```

---

## 4. 核心工作流程

### 4.1 前缀缓存查找 — `get_computed_blocks`

当请求进入调度器时，首先查找已有的前缀缓存命中：

```
KVCacheManager.get_computed_blocks(request)
    │
    ├── max_cache_hit_length = request.num_tokens - 1
    │   (所有 token 都命中时需重算最后一个 token 的 logits)
    │
    └── coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
        │
        ├── [Unitary] 委托给唯一的 manager.find_longest_cache_hit()
        └── [Hybrid]  迭代不动点算法，取所有注意力类型的交集
            │
            └── FullAttentionManager.find_longest_cache_hit()
                ├── 从左到右遍历 block_hashes
                ├── 在 BlockPool.get_cached_block() 中查找
                └── 遇到 miss 即 break（链式哈希保证后续必 miss）
```

**返回值：** `(KVCacheBlocks, num_computed_tokens)` — 命中的 block 列表和对应的 token 数。

### 4.2 Block 分配 — `allocate_slots`

这是最核心的方法，为请求分配新的 block 槽位。完整流程：

```
allocate_slots(request, num_new_tokens, new_computed_blocks, ...)
    │
    ├── 1. 计算总 computed tokens
    │      num_local_computed = request.num_computed + num_new_computed
    │      total_computed = min(local + external, max_model_len)
    │
    ├── 2. 计算 watermark（仅对 WAITING/PREEMPTED 请求）
    │
    ├── 3. [可选] full_sequence_must_fit 检查
    │      预检查完整序列能否容纳，防止 chunked prefill 过度准入
    │
    ├── 4. remove_skipped_blocks()
    │      释放滑窗/分块注意力中不再需要的 block（窗口外的）
    │      → 释放到 free queue，减少后续需要驱逐的 block
    │
    ├── 5. get_num_blocks_to_allocate()
    │      精确计算需要分配的新 block 数（考虑缓存命中、跳过、可驱逐块）
    │
    ├── 6. 容量检查
    │      required > available_blocks → return None (无法调度)
    │
    ├── 7. allocate_new_computed_blocks()
    │      ├── add_local_computed_blocks(): touch 命中的缓存块（ref_cnt++）
    │      └── allocate_external_computed_blocks(): 外部 KV 的 block 分配
    │
    ├── 8. allocate_new_blocks()
    │      从 BlockPool 获取新 block，追加到 req_to_blocks
    │
    └── 9. cache_blocks()
           将填满的 block 计算哈希并存入前缀缓存
```

**分配的 block 布局（`allocate_slots` 文档中的图）：**

```
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
                                                      | < to be computed >    |
                                  |          < to be allocated >          |
| < cached by vLLM >    | not cached by |
| ref_cnt  | ref_cnt not  | vLLM, but     |
| increased| increased yet| cached by     |
                         | connector     |
```

其中：

- `comp` = 已计算 tokens（之前 step 的）
- `new_comp` = 本次新命中前缀缓存的 tokens
- `ext_comp` = 外部连接器缓存的 tokens
- `new` = 本次需要新计算的 tokens
- `lookahead` = 推测解码的 lookahead tokens

### 4.3 前缀缓存存储 — `cache_blocks` / `cache_full_blocks`

当一个 block 填满后，计算其哈希并存入缓存：

```
SingleTypeKVCacheManager.cache_blocks(request, num_tokens)
    │
    ├── num_full_blocks = num_tokens // block_size
    ├── reachable_block_mask() → 确定哪些 block 值得缓存
    │      (SWA 中只有可被命中的 block 才缓存)
    │
    └── BlockPool.cache_full_blocks(request, blocks, num_cached, num_full, ...)
        │
        └── 对每个 new_full_block:
            ├── block_hash = request.block_hashes[i]  (hash_block_size 粒度)
            ├── block_hash_with_group_id = pack(block_hash, group_id)
            ├── 如果 block 已有部分哈希 → 移除旧哈希（partial→full 提升）
            └── _insert_block_hash(hash_with_group_id, block)
                → 存入 cached_block_hash_to_block
                → 记录到 cached_block_hashes_by_block
```

**partial block caching**（`cache_partial_block`）：

- 当不同 KV Cache Group 有不同 block_size 时，较细粒度（`hash_block_size`）的边界需要部分缓存
- 允许从 block 内部的细粒度前缀边界命中

### 4.4 Block 释放与驱逐

**释放流程：**

```
KVCacheManager.free(request)
    └── coordinator.free(request_id)
        └── for each manager:
            manager.free(request_id)
                └── block_pool.free_blocks(reversed(req_blocks))
                    │                    ↑ 逆序：尾部先释放（先被驱逐）
                    │
                    ├── ref_cnt -= 1
                    ├── if ref_cnt == 0 and not is_null:
                    │     if block_hash is None:
                    │         blocks_without_hash.append(block)
                    │     else:
                    │         blocks_with_hash.append(block)
                    │
                    ├── free_block_queue.prepend_n(blocks_without_hash)
                    │     ↑ 无哈希的插到队头前（优先驱逐）
                    └── free_block_queue.append_n(blocks_with_hash)
                          ↑ 有哈希的插到队尾（保留更久，可能被命中）
```

**驱逐机制**（`get_new_blocks` 时触发）：

```
BlockPool.get_new_blocks(num_blocks)
    └── for each block from free_block_queue:
        ├── _maybe_evict_cached_block(block)
        │     ├── 如果 block 有 hash → 移除哈希映射
        │     └── 发出 BlockRemoved 事件
        └── block.ref_cnt = 1
```

当空闲 block 不足时，从队头弹出 block 进行分配。如果该 block 恰好有缓存的哈希，则先驱逐其缓存映射。

### 4.5 Touch — 前缀缓存命中

当新请求命中已有前缀缓存时，通过 `touch()` 增加引用计数：

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # 从空闲队列移除
        block.ref_cnt += 1
```

这使得被命中的 block 不再是驱逐候选，同时多个请求可以共享同一 block。

---

## 5. 多类型注意力支持

### 5.1 Sliding Window Attention

**核心挑战：** 滑动窗口只关注最近 `sliding_window` 个 token，窗口外的 block 不再需要。

**`get_num_skipped_tokens`**（`single_type_kv_cache_manager.py:770`）：

```
sliding_window=4, num_computed=7

Tokens: [0  1  2  3  4  5  6  7]
         |----computed---|
                                ^ next token
                      |---------| sliding window
         |--skipped---|

get_num_skipped_tokens(7) = max(0, 7 - 4 + 1) = 4
```

**`find_longest_cache_hit`** 特殊逻辑（`single_type_kv_cache_manager.py:620`）：

- 从右到左搜索，找到**连续命中**的 block 组
- 需要至少 `sliding_window_contiguous_blocks` 个连续命中才算有效
- 窗口外的 block 用 `null_block` 填充

**`remove_skipped_blocks`**：

- 随着请求进度推进，窗口外的 block 被释放（替换为 null_block）
- 这使得滑动窗口模型可以处理远超 `sliding_window` 长度的序列

### 5.2 Chunked Local Attention

类似滑动窗口，但按固定的 `attention_chunk_size` 分块。每个 chunk 内做局部注意力，chunk 外的 token 不参与计算。

### 5.3 Hybrid Models（混合注意力）

如 Gemma2 等，同时使用 Full Attention 和 Sliding Window：

- `HybridKVCacheCoordinator` 按注意力 spec 分组
- `find_longest_cache_hit` 使用迭代不动点算法取所有组的**交集命中长度**
- `scheduler_block_size` = 所有组 block_size 的 LCM
- `hash_block_size` = 所有组 block_size 的 GCD（或用户指定）

---

## 6. 高级特性

### 6.1 EAGLE/MTP 支持

EAGLE 投机解码需要最后一个 block 的隐藏状态用于 draft head，因此需要**丢弃最后一个命中的 block** 强制重算：

```python
if drop_eagle_block and computed_blocks[0]:
    for computed in computed_blocks:
        computed.pop()
```

在 `HybridKVCacheCoordinator` 中，EAGLE 的处理更复杂——需要匹配多一个 block 然后丢弃，且每次候选长度变化时需重新验证。

### 6.2 Context Parallelism

支持 DCP（Decode Context Parallel）和 PCP（Prefill Context Parallel）：

- `block_size *= dcp_world_size * pcp_world_size`
- 每个 block 横跨多个 GPU shard

### 6.3 External KV Cache（P/D Disaggregation）

支持从外部 connector 加载 KV Cache（如 P/D 分离架构）：

- `num_external_computed_tokens`：外部缓存的 token 数
- `allocate_external_computed_blocks()`：为外部 token 分配新 block
- `delay_cache_blocks=True`：暂不缓存，等待 KV 传输完成

**两阶段分配**（issue #33775）：

1. 先 touch 所有 group 的本地缓存命中块
2. 再分配外部 block
   避免前组的外部分配驱逐后组还没 touch 的缓存命中块。

### 6.4 Watermark 机制

```python
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
```

- 仅对 WAITING/PREEMPTED 请求且已有 running 请求时生效
- 在可用 block 数中保留 `watermark_blocks` 个不被新请求占用
- 防止频繁的 preemption（抢占）

### 6.5 KV Cache Events

当 `enable_kv_cache_events=True` 时，BlockPool 会发出事件：

- `BlockStored`：block 被存入缓存
- `BlockRemoved`：block 被驱逐
- `AllBlocksCleared`：所有缓存被清除

这些事件用于 P/D 分离、KV offload 等外部系统追踪缓存状态。

### 6.6 Prefix Cache Retention

```python
retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
```

控制 SWA 的稀疏检查点粒度：

- `None`：密集缓存（每个 alignment 边界都缓存）
- `0`：仅保留最近的重放边界
- 正整数：每 `retention_interval` 个 token 保留一次检查点

### 6.7 Metrics 采样

`KVCacheMetricsCollector` 以采样率（默认 1%）跟踪 block 生命周期：

- `birth_time_ns`：block 分配时间
- `access_history`：访问历史（bounded deque）
- `idle_time_seconds`：空闲时间
- `reuse_gaps_seconds`：重用间隔

驱逐时生成 `KVCacheEvictionEvent`，用于分析缓存效率。

---

## 7. 完整生命周期

以一个请求的完整生命周期为例：

```
1. 请求到达 Scheduler
   └── KVCacheManager.get_computed_blocks(request)
       ├── 计算请求的 block_hashes（哈希链）
       └── find_longest_cache_hit() → 命中 N 个 cached blocks

2. 调度准入检查
   └── allocate_slots(request, num_new_tokens, new_computed_blocks, ...)
       ├── remove_skipped_blocks() → 释放滑窗外 block
       ├── get_num_blocks_to_allocate() → 需要分配 M 个
       └── 容量检查 → 通过

3. 分配命中块 + 新块
   ├── allocate_new_computed_blocks()
   │   ├── add_local_computed_blocks() → touch 命中块 (ref_cnt++)
   │   └── allocate_external_computed_blocks() (如有外部 KV)
   └── allocate_new_blocks() → 从 free_block_queue 分配 M 个新块

4. 模型计算
   └── GPU worker 使用分配的 block_ids 进行 attention 计算

5. 缓存新填满的 block
   └── cache_blocks(request, num_computed_tokens)
       └── cache_full_blocks() → 计算哈希，存入 cached_block_hash_to_block

6. 后续 decode step
   └── 重复步骤 2-5（allocate_slots 对 RUNNING 请求走快速路径）

7. 请求完成
   └── KVCacheManager.free(request)
       └── block_pool.free_blocks(reversed(blocks))
           ├── ref_cnt -= 1
           └── ref_cnt == 0 → 回收到 free_block_queue
               ├── 无 hash → prepend（优先驱逐）
               └── 有 hash → append（保留更久，可能被命中）

8. 空间不足时
   └── get_new_blocks() 从队头弹出
       └── _maybe_evict_cached_block() → 驱逐其哈希映射
```

---

## 8. 设计要点总结

1. **PagedAttention**：KV Cache 按 block 分页管理，而非连续分配，避免内存碎片。
2. **引用计数共享**：多个请求命中相同前缀时共享物理 block，通过 `ref_cnt` 管理。
3. **链式哈希前缀缓存**：前缀匹配通过哈希链实现，确保前缀相同则 block 哈希相同。
4. **LRU 驱逐策略**：空闲块按 LRU 排序，无哈希块优先驱逐，有哈希块保留更久。
5. **分层管理**：Manager → Coordinator → SingleTypeManager → BlockPool，职责清晰。
6. **多注意力类型支持**：通过不同的 Coordinator 和 Manager 支持混合注意力模型。
7. **两阶段分配**：避免跨 group 的缓存命中块被提前驱逐。
8. **Watermark 准入控制**：防止过度准入导致频繁抢占。
9. **事件驱动**：KV Cache 事件支持 P/D 分离等外部系统。
10. **延迟缓存**：P/D 场景下可延迟缓存，等待 KV 传输完成后再提交。
