# SingleTypeKVCacheManager 详解

## 一、是什么

`SingleTypeKVCacheManager` 是五层 KV Cache 管理架构中的第三层——**单类型 KV 缓存管理器**。它负责管理**一种具体 Attention 类型**的 KV Cache 分配、命中查找、释放等所有逻辑。

对于纯 Full Attention 模型（Llama、Qwen、Mistral 等），核心使用的是它的子类 `FullAttentionManager`，实现了**链式哈希前缀缓存**机制，可以在请求之间高效共享相同前缀的 KV 缓存。

其他子类（SlidingWindowManager、RSWAManager、MambaManager等）用于支持混合模型或特殊注意力类型，本文最后会简要概述。

---

## 二、干什么用

### 在五层架构中的位置

```
┌─────────────────────────────────────────────────────────────┐
│ 第五层：KVCacheManager（唯一门面，Scheduler 唯一交互入口）    │
├─────────────────────────────────────────────────────────────┤
│ 第四层：KVCacheCoordinator（跨组协调，纯FullAttention退化为  │
│        UnitaryKVCacheCoordinator，直接透传）                 │
├─────────────────────────────────────────────────────────────┤
│ 第三层：SingleTypeKVCacheManager  ← 本文讲解                 │
│  ┌──────────────────┬──────────────────┬─────────────────┐  │
│  │FullAttentionManager│SlidingWindowMgr│ 其他Manager...  │  │
│  └──────────────────┴──────────────────┴─────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│ 第二层：BlockPool（块池，管理 free/cached 块链表和哈希映射）  │
├─────────────────────────────────────────────────────────────┤
│ 第一层：物理 KV Cache 张量（GPU 上真实存储 K/V 的大张量）     │
└─────────────────────────────────────────────────────────────┘
```

### 核心职责（纯 FullAttention 场景）

对应总览文档的端到端流程，`FullAttentionManager` 承担以下职责：

| 总览阶段 | 职责 | 对应源码方法 |
|----------|------|--------------|
| **阶段2：前缀查找** | 在 `cached_block_hash_to_block` 中查找最长已计算前缀，返回命中块列表和额外命中token数 | `find_longest_cache_hit` |
| **阶段3：touch命中块** | 对命中的块调用 `block_pool.touch()`：ref_cnt += 1，从 `free_block_queue` 移除，防止被驱逐 | `add_local_computed_blocks` |
| **阶段3：计算新块数** | 根据总token数和已命中块数，计算需要新分配多少块 | `get_num_blocks_to_allocate` |
| **阶段3：分配新块** | 从 `free_block_queue` 头部取无哈希块分配，不足则LRU驱逐尾部哈希块；新块id加入 `new_block_ids` 等待Worker清零，重置块哈希 | `allocate_new_blocks` |
| **阶段3：维护req_to_blocks** | 将命中块+新块按顺序加入 `self.req_to_blocks[request_id]`，这就是逻辑block_table的真正存储位置 | allocate流程中维护 |
| **阶段5：写入缓存** | prompt计算完成后，对填满的新块计算链式哈希，写入 `cached_block_hash_to_block`，放入 `free_block_queue` 尾部（LRU保护） | `maybe_save_new_kv_blocks_to_cache` |
| **阶段6：释放块** | 请求结束时逆序遍历块列表，ref_cnt -= 1；ref_cnt=0的有哈希块回收到 `free_block_queue` 尾部（缓存），无哈希块回收到头部（优先复用） | `release_blocks` |

---

## 三、类继承结构

```
SingleTypeKVCacheManager（ABC 抽象基类）—— 定义单类型管理器的标准接口和通用逻辑
├── FullAttentionManager        ← 本文核心：全注意力前缀缓存
├── SlidingWindowManager        ← 滑动窗口注意力（简要概述）
├── RSWAManager                 ← 重复滑动窗口注意力（简要概述）
└── MambaManager                ← Mamba/SSM 模型（简要概述）
```

**抽象基类的意义**：统一接口，上层 `KVCacheCoordinator` 可以用一致的方式管理不同类型的 KV 组，不需要关心底层是 FullAttention 还是 SWA。

---

## 四、SingleTypeKVCacheManager 基类详解

基类实现了所有管理器共有的逻辑，子类只需要实现特定的差异部分（如前缀查找算法、新块数计算等）。

### 4.1 构造函数

源码位置：`single_type_kv_cache_manager.py:36-127`

```python
class SingleTypeKVCacheManager(ABC):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,                     # 该组的 KV Cache 规格（FullAttentionSpec等）
        block_pool: BlockPool,                          # 所属的块池（全局唯一）
        enable_caching: bool,                           # 是否启用前缀缓存
        kv_cache_group_id: int,                         # 本组的group_id（纯FullAttention=0）
        scheduler_block_size: int,                      # 调度器对齐块大小（纯FullAttention=block_size）
        dcp_world_size: int = 1,                        # 分布式KV传输world size
        pcp_world_size: int = 1,                        # 前缀缓存持久化world size
        needs_kv_cache_zeroing: bool = False,           # 新分配的块是否需要Worker侧清零
        max_admission_blocks_per_request: int | None = None,  # 单请求最大接纳块数（SWA用）
    ) -> None:
        self.scheduler_block_size = scheduler_block_size
        self.block_size = kv_cache_spec.block_size      # 本组的块大小（每个块存多少token）
        self.dcp_world_size = dcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request

        # ── 新块清零队列（drain模式） ──
        # 只有FullAttention等类型需要记录新块id，给Worker清零
        self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (
            FullAttentionSpec, TQFullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec,
        )
        self.new_block_ids: list[int] = []              # 累计本轮新分配的块id，Worker每轮drain一次

        # ── 请求→块映射（核心数据结构） ──
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        # key=request_id，value=该请求持有的KVCacheBlock列表（有序，顺序就是block_table顺序）
        # 这就是Scheduler看到的"请求的block_table"的真正存储位置

        self.num_cached_block: dict[str, int] = {}      # 每个RUNNING请求已缓存的块数统计
        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block        # null_block占位符引用

        self.use_eagle = False                          # EAGLE投机解码标记
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}  # 部分命中CoW记录
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []  # CoW复制队列
        self._pending_partial_tail_offloads: list[tuple[str, int, KVCacheBlock, int]] = []
```

**端到端流程中的关键成员**：
- `new_block_ids`：对应总览5.3/5.4阶段——分配新块时把块id加入这个列表，Scheduler调度完后通过`take_new_block_ids()` drain走，发给Worker清零
- `req_to_blocks[request_id]`：对应总览5.3返回的`KVCacheBlocks`——这个list就是请求持有的块列表，后续forward、释放都从这里取
- `_null_block`：对齐用，不赘述
- `_pending_cow_copies`：Copy-on-Write复制队列，同样是drain模式，Worker需要复制块数据时从这里取

**新块清零drain方法**（源码`single_type_kv_cache_manager.py:376-380`）：
```python
def take_new_block_ids(self) -> list[int]:
    """Drain and return block IDs allocated since the last call."""
    ids = self.new_block_ids
    self.new_block_ids = []
    return ids
```
这就是总览5.4阶段Worker清零的数据源——每个manager自己记自己的新块，顶层KVCacheManager遍历所有manager汇总。

**构造函数关键点**：
- 纯 Full Attention 场景下，`scheduler_block_size == block_size == 16`（假设），没有倍数关系
- `_record_new_block_ids=True`，所以FullAttention的新块都会被记录等待清零
- `req_to_blocks` 是真正持有请求块列表的地方，不是Request对象的字段

### 4.2 核心方法：`get_num_blocks_to_allocate`

源码位置：`single_type_kv_cache_manager.py:128-142`

```python
    def get_num_blocks_to_allocate(
        self,
        seq_len: int,                                   # 请求当前序列长度
        num_computed_blocks: int,                       # 已经命中的块数
        num_new_tokens: int,                            # 本轮新生成的 token 数
    ) -> int:
        """计算本轮需要新分配多少块"""
        # 计算总共需要的块数：向上取整
        # (seq_len + num_new_tokens + block_size - 1) // block_size
        total_required_blocks = (
            seq_len + num_new_tokens + self.block_size - 1
        ) // self.block_size
        # 需要的新块 = 总需要块 - 已有的块
        return total_required_blocks - num_computed_blocks
```

**作用**：计算本轮需要新分配多少物理块，这是分配前的"容量预估"。

对于 FullAttention，这是简单的向上取整除法。其他类型（如 SWA）可能有不同的计算逻辑（只保留窗口内的块）。

### 4.3 核心方法：`add_local_computed_blocks`

源码位置：`single_type_kv_cache_manager.py:144-231`

```python
    def add_local_computed_blocks(
        self,
        parent_block: KVCacheBlock | None,              # 前一个块（用于链式哈希校验）
        new_computed_blocks: list[KVCacheBlock],        # 本次命中的新块列表
    ) -> None:
        """处理命中的块：增加引用计数，校验链式哈希完整性"""
        # 遍历所有命中的块
        for block in new_computed_blocks:
            if self.use_cascade:
                # Cascade 模式特殊处理（远程缓存传输场景）
                assert self.cascade_invalidation_groups is not None
                parent_block = self._cascade_add_computed_block(
                    block, parent_block
                )
            else:
                # 标准模式：
                # 1. 增加引用计数，表示这个块被当前请求使用了
                # 2. 从 cached 队列移除（touch），加入 in_use 状态
                # 3. 校验 parent_hash 链是否完整（防止哈希碰撞或数据损坏）
                self.block_pool.touch(block, parent_block)
                parent_block = block
```

**关键点**：
- 命中缓存块不是简单"拿来用"，必须调用 `block_pool.touch()`：
  - 增加 `ref_cnt`，防止块被驱逐
  - 从空闲/缓存队列中移除
  - 校验链式哈希的完整性
- `parent_block` 参数用于校验块的前向哈希链，确保整个前缀序列是连续的

### 4.4 核心方法：`allocate_new_blocks`

源码位置：`single_type_kv_cache_manager.py:233-283`

```python
    def allocate_new_blocks(
        self,
        num_blocks: int,                                # 需要分配的块数
        pool_name: str,                                 # 块池名称（区分本地/远程）
        eviction_cb: Optional[Callable[[int], None]] = None,  # 驱逐回调
    ) -> list[KVCacheBlock]:
        """从块池获取新块，不够则触发驱逐"""
        extra_keys = []
        if self.use_cascade and self.cascade_invalidation_groups:
            extra_keys = self.cascade_invalidation_groups

        blocks: list[KVCacheBlock] = []
        while len(blocks) < num_blocks:
            # 尝试从块池获取一个新块
            block = self.block_pool.get_new_block(
                pool_name=pool_name, extra_keys=extra_keys
            )
            if block is not None:
                # 获取成功，加入结果列表
                blocks.append(block)
                continue

            # 获取失败，说明没有空闲块了，需要驱逐
            # 优先驱逐 cached 状态的块（前缀缓存中无人使用的块）
            evicted = self.block_pool.evict_cached_block(
                extra_keys=extra_keys,
                pool_name=pool_name,
            )

            if not evicted:
                # 连 cached 块都没有了，说明内存真的耗尽
                raise KVCacheManagerNoBlocksAvailable(
                    f"Out of KV cache blocks in pool '{pool_name}'. "
                    f"Tried to allocate {num_blocks} blocks but only "
                    f"got {len(blocks)}."
                )

            # 如果注册了驱逐回调，通知上层（用于统计或远程缓存失效）
            if eviction_cb is not None:
                eviction_cb(evicted.block_id)

        # 所有新块都重置为"未哈希"状态（因为还没写入内容）
        for block in blocks:
            self.block_pool.reset_hash(block)
        return blocks
```

**分配流程**：
1. 循环尝试从 `free_blocks` 队列取块：优先取队首无哈希块（可直接复用无需清零残留）
2. 如果没有空闲块，先驱逐 `cached` 状态的块（从队尾取最久未使用的前缀缓存块）
3. 如果连 cached 块都没有了，抛出内存不足错误（上层KVCacheManager会捕获并返回None，触发请求抢占）
4. 成功获取的块全部调用 `reset_hash` 清空block_hash，因为它们接下来会被写入新内容，旧哈希失效

---

### 4.4.1 端到端分配流程串讲（对应总览阶段3）

`UnitaryKVCacheCoordinator.allocate_slots` 调用本类方法的完整顺序：
```python
# 对应 kv_cache_coordinator.py UnitaryKVCacheCoordinator.allocate_slots 简化版
def allocate_slots(self, request, num_new_tokens, new_computed_blocks, ...):
    blocks = self.managers[0].req_to_blocks[request.request_id]
    
    # 步骤1：处理上一轮已经命中的块（续写场景）
    parent_block = blocks[-1] if blocks else None
    
    # 步骤2：touch本次新命中的块，增加ref_cnt，从free_block_queue移除
    self.managers[0].add_local_computed_blocks(parent_block, new_computed_blocks)
    blocks.extend(new_computed_blocks)
    
    # 步骤3：计算需要新分配多少块
    num_new_blocks = self.managers[0].get_num_blocks_to_allocate(
        seq_len=request.num_tokens,
        num_computed_blocks=len(blocks),
        num_new_tokens=num_new_tokens,
    )
    
    # 步骤4：分配新块
    new_blocks = self.managers[0].allocate_new_blocks(num_new_blocks, pool_name="local")
    blocks.extend(new_blocks)
    
    # 步骤5：记录新块id，等待Worker清零
    if self.managers[0]._record_new_block_ids:
        self.managers[0].new_block_ids.extend(b.block_id for b in new_blocks)
```

**关键点**：`req_to_blocks[request_id]` 就是在这里一步步 `extend` 维护起来的——它的内容就是 [历史块..., 本次命中块..., 本次新分配块...]，顺序就是block_table在GPU上的顺序。

### 4.5 核心方法：`maybe_save_new_kv_blocks_to_cache`

源码位置：`single_type_kv_cache_manager.py:285-350`（基类定义，子类实现具体逻辑）

```python
    @abstractmethod
    def maybe_save_new_kv_blocks_to_cache(
        self,
        blocks: list[KVCacheBlock],                     # 本请求使用的所有块
        new_blocks: list[KVCacheBlock],                 # 本轮新分配的块
        token_ids: list[int],                           # 所有 token ids
        num_verified_tokens: int,                       # 已经验证过的 token 数（命中的）
        parent_block_hash: Optional[BlockHash],         # 前一个块的哈希（链式用）
        lora_id: Optional[int],                         # LoRA ID（多租户场景）
    ) -> tuple[bool, BlockHash | None, int]:
        """计算新块的哈希并加入前缀缓存"""
        raise NotImplementedError
```

这是抽象方法，由 `FullAttentionManager` 实现具体的链式哈希计算逻辑。

### 4.6 核心方法：`pop_blocks_for_free` / `free` / `release_blocks`（释放流程）

源码位置：`single_type_kv_cache_manager.py:500-527, 352-450`，对应总览阶段6的释放流程。

#### 4.6.1 `pop_blocks_for_free`：取出请求的块列表（不真正释放）

```python
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        从req_to_blocks中弹出该请求的块列表，清理相关统计，但不归还BlockPool。
        调用方拿到块列表后，后续负责调用free_blocks归还。
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])  # 弹出块列表，移除映射
        self.num_cached_block.pop(request_id, None)          # 清理缓存统计
        self._partial_hit_reqs.pop(request_id, None)         # 清理部分命中记录
        return req_blocks                                    # 返回按分配顺序排列的块列表
```

#### 4.6.2 `free`：完整释放单个请求的所有块（最常用）

```python
    def free(self, request_id: str) -> None:
        """释放请求的所有块：弹出块列表 → 逆序归还BlockPool"""
        # 关键：reversed() 逆序释放——尾块先归还，利用free_block_queue的LIFO特性
        # 这样请求被抢占后重新调度时，原来的尾块会最先被分配回来，提高续生成命中率
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

#### 4.6.3 `free_blocks`（BlockPool方法）：引用计数减一+队列回收

```python
    # BlockPool.free_blocks 简化逻辑
    def free_blocks(self, ordered_blocks: list[KVCacheBlock]) -> None:
        blocks_without_hash: list[KVCacheBlock] = []
        blocks_with_hash: list[KVCacheBlock] = []
        for block in ordered_blocks:
            if block.is_null:
                continue
            block.ref_cnt -= 1                              # 引用计数减一
            if block.ref_cnt == 0:                          # 没人引用了，可以回收
                if block.block_hash is None:
                    blocks_without_hash.append(block)       # 无哈希：队首优先驱逐
                else:
                    blocks_with_hash.append(block)          # 有哈希：队尾LRU保护
        self.free_block_queue.prepend_n(blocks_without_hash) # 无哈希放队首（先复用）
        self.free_block_queue.append_n(blocks_with_hash)    # 有哈希放队尾（LRU缓存）
```

**释放逻辑关键点**：
- **两阶段释放**：先pop取出块列表（清理上层映射），再逆序调用free_blocks归还（底层BlockPool处理引用计数和队列）
- **逆序释放**：从最后一个块开始放，`free_block_queue` 是双向链表——无哈希块放队首（下次分配优先拿，不用清零），有哈希块放队尾（LRU保护，优先驱逐）
- **引用计数机制**：多个请求共享前缀块时，只有最后一个请求释放时 `ref_cnt` 才会到0，块才会真正被回收到队列
- **哈希决定去向**：`block_hash is None` → 内容不完整，放队首优先复用（无需作为前缀缓存）；`block_hash is not None` → 内容完整，放队尾进入前缀缓存池

---

## 五、FullAttentionManager 详解

这是纯 Full Attention 模型的核心管理器，实现了完整的链式哈希前缀缓存机制。

### 5.1 前缀查找：`find_longest_cache_hit`

源码位置：`single_type_kv_cache_manager.py:681-778`

这是前缀缓存的核心方法，实现了**最长前缀匹配查找**，找到请求序列中有多少 KV 块已经在缓存中了。

#### 5.1.1 方法签名与整体流程

```python
class FullAttentionManager(SingleTypeKVCacheManager):
    def find_longest_cache_hit(
        self,
        token_ids: list[int],                           # 请求完整的 token 序列
        block_hashes: list[BlockHash],                  # [输出] 填充计算出的每个块的哈希
        parent_block_hash: BlockHash | None = None,     # 前一个块的哈希（续写场景）
        parent_block_id: int | None = None,             # 前一个块的ID
        lora_id: int | None = None,                     # LoRA ID
    ) -> tuple[list[KVCacheBlock], BlockHash | None, int]:
        """
        返回值：
            1. 命中的块列表（按顺序）
            2. 最后一个命中块的哈希（用于下一轮链式）
            3. 命中的token总数（= num_blocks * block_size + 尾部部分token）
        """
```

#### 5.1.2 逐段逻辑详解

**第一阶段：跳过已经命中的部分（续写场景）**

```python
        # 起点：如果有parent_block，说明前序token已经命中过了
        # start_token_idx 从第一个未确认的token开始
        start_token_idx = 0
        if parent_block_hash is not None:
            start_token_idx = (len(block_hashes) - len(block_hashes) % self.block_size)
```

**第二阶段：逐块计算哈希，查找缓存**

```python
        computed_blocks: list[KVCacheBlock] = []
        # 遍历每个"完整块边界"（block_size个token一块）
        for token_idx in range(start_token_idx, len(token_ids), self.block_size):
            # 取出当前块的token（可能不满block_size，最后一块）
            block_tokens = token_ids[token_idx : token_idx + self.block_size]

            # ========== 计算当前块的哈希 ==========
            # 链式哈希：hash = hash(prev_hash + block_tokens)
            # 这样前缀不同的话，即使后面内容相同，哈希也不同
            block_content_to_hash: tuple[int | BlockHash | tuple, ...] = ()
            if computed_blocks:
                # 有前一个命中的块，用它的哈希作为链式前缀
                block_content_to_hash = (
                    self.block_pool.get_block_hash(computed_blocks[-1]),
                )
            elif parent_block_hash is not None and len(computed_blocks) == 0:
                # 第一轮但有parent_block，用parent的哈希
                block_content_to_hash = (parent_block_hash,)

            # 加上当前块的token和LoRA ID
            block_content_to_hash = (
                block_content_to_hash,
                tuple(block_tokens),
                lora_id,
            )

            # 计算哈希值
            block_hash = hash_block_tokens(*block_content_to_hash)

            # 计算"块内起始位置"（处理最后一块不满的情况）
            # mapping_index 表示这个哈希对应块内第几个token之后
            mapping_index = token_idx + self.block_size - 1
            if mapping_index >= len(token_ids):
                mapping_index = len(token_ids) - 1

            block_hashes.append(block_hash)

            # ========== 在块池中查找这个哈希 ==========
            block_id = self.block_pool.get_cached_block_id(
                block_hash,
                self.kv_cache_spec.group_id,             # group_id 隔离不同组的哈希
                mapping_index=mapping_index,
            )

            if block_id is None:
                # 没找到，说明从这里开始前缀断裂
                break

            # 找到了！取出块对象，加入命中列表
            block = self.block_pool.blocks[block_id]
            computed_blocks.append(block)
```

**第三阶段：处理最后一个块的部分token命中**

```python
        # 如果找到了完整块，最后检查一下"最后一块的尾部"是否有部分token命中
        extra_block_hashes = []
        num_computed_tokens = len(computed_blocks) * self.block_size

        if computed_blocks:
            last_full_block_idx = len(computed_blocks) * self.block_size
            last_block = computed_blocks[-1]
            last_block_hash = self.block_pool.get_block_hash(last_block)

            # 逐token向后看，是否有部分命中
            for token_idx in range(
                last_full_block_idx - self.block_size + 1,
                len(token_ids)
            ):
                # 每多一个token，重新算一次哈希
                block_tokens = token_ids[
                    last_full_block_idx - self.block_size : token_idx + 1
                ]
                partial_hash = hash_block_tokens(
                    last_block_hash
                    if token_idx == last_full_block_idx - self.block_size
                    else extra_block_hashes[-1],
                    tuple(block_tokens[-1:]),  # 只加最新的一个token
                    lora_id,
                )
                extra_block_hashes.append(partial_hash)
                block_hashes.append(partial_hash)

                # 在映射表中查找这个位置的映射
                block_id = self.block_pool.get_cached_block_id(
                    partial_hash,
                    self.kv_cache_spec.group_id,
                    mapping_index=token_idx,
                )
                if block_id is None:
                    break
                # 如果找到了（而且就是最后那个块），说明多命中了一些token
                # 注意：这里不会新增块，只是同一个块内多命中几个token
                if block_id != last_block.block_id:
                    break
                num_computed_tokens = token_idx + 1
```

**第四阶段：返回结果**

```python
        # 计算最终的父哈希和命中token数
        if computed_blocks:
            last_block = computed_blocks[-1]
            new_parent_block_hash = self.block_pool.get_block_hash(last_block)
            # 注意：这里直接修改了传入的 computed_blocks 列表
            # 把最后一个块的"额外命中token数"记录下来
            # 见 single_type_kv_cache_manager.py:759-760
            new_num_extra_tokens = num_computed_tokens - len(computed_blocks) * self.block_size
        else:
            new_parent_block_hash = parent_block_hash
            new_num_extra_tokens = 0

        return computed_blocks, new_parent_block_hash, new_num_extra_tokens
```

#### 5.1.3 链式哈希图解（对应总览34token例子）

```
请求token序列（34个token，block_size=16）：
  [T0-T15]  → block 0 （满块）
  [T16-T31] → block 1 （满块）
  [T32,T33] → block 2 （只有2个token，不满）

哈希链计算：
block_hash_0 = H(null, T0..T15, lora_id)        → 在cached_block_hash_to_block中找到块8
block_hash_1 = H(block_hash_0, T16..T31, lora_id) → 在cached_block_hash_to_block中找到块12
block_hash_2 = H(block_hash_1, T32..T33, lora_id) → 查找失败，前缀断裂

返回结果：
  computed_blocks = [块8, 块12]
  new_parent_hash = block_hash_1
  num_extra_tokens = 0 （block 2是新块，没有部分命中）
```

**链式哈希特点**：
1. 每个块的哈希依赖前一个块的哈希 → 保证前缀连续性，避免"中间某块相同但前缀不同"的错误命中
2. 即使两个块内容完全相同，如果前缀不同，哈希也不同 → 天然避免哈希碰撞导致的错误复用
3. 最后不满一块的部分，每个token位置都有独立的哈希映射 → 支持部分尾部命中（续写场景下常见）

### 5.2 缓存写入：`maybe_save_new_kv_blocks_to_cache`

源码位置：`single_type_kv_cache_manager.py:491-580`

当一个块的 KV 计算完成后，需要为它计算哈希并加入前缀缓存，供后续请求命中。

```python
    def maybe_save_new_kv_blocks_to_cache(
        self,
        blocks: list[KVCacheBlock],
        new_blocks: list[KVCacheBlock],
        token_ids: list[int],
        num_verified_tokens: int,                # 已经验证过/命中的token数
        parent_block_hash: BlockHash | None,
        lora_id: int | None = None,
    ) -> tuple[bool, BlockHash | None, int]:
        """返回：(是否缓存成功, 最后块哈希, 已验证token数)"""

        # 找出哪些"新块"现在填满了，可以计算哈希了
        blocks_start_idx = len(blocks) - len(new_blocks) - 1
        if parent_block_hash is None and len(blocks) > len(new_blocks):
            blocks_start_idx = len(blocks) - len(new_blocks)
        blocks_to_cache: list[tuple[int, BlockHash, int]] = []

        cached_any = False
        prev_hash = parent_block_hash
        updated_num_verified_tokens = num_verified_tokens

        # 遍历所有块（从已有块的末尾开始）
        for i, block in enumerate(blocks):
            if i <= blocks_start_idx:
                # 已经在缓存里的块，更新prev_hash后跳过
                if block is not self.null_block:
                    prev_hash = self.block_pool.get_block_hash(block)
                continue

            # 计算这个块在token序列中的起始和结束位置
            block_start_token_idx = i * self.block_size
            block_end_token_idx = min(
                (i + 1) * self.block_size, len(token_ids)
            )

            if block_start_token_idx >= len(token_ids):
                break

            # ========== 只有当块内的token都计算完了，才能缓存 ==========
            if block_end_token_idx > num_verified_tokens:
                # 还没填满，不能缓存
                break

            # ========== 计算链式哈希 ==========
            block_tokens = token_ids[block_start_token_idx:block_end_token_idx]
            block_hash = hash_block_tokens(prev_hash, tuple(block_tokens), lora_id)

            # ========== 处理块内部分位置的映射（最后一块可能不满）==========
            # 对于FullAttention，块内每个token位置都可以有映射
            # 这样下一个请求即使只多生成了几个token，也能命中部分
            for j in range(block_start_token_idx, block_end_token_idx):
                # 每个token位置j都有一个哈希（通过逐个累加计算）
                if j == block_start_token_idx:
                    position_hash = block_hash
                else:
                    position_hash = hash_block_tokens(
                        prev_position_hash,
                        (token_ids[j],),
                        lora_id,
                    )
                prev_position_hash = position_hash
                # 记录：位置j → block_id 的映射
                blocks_to_cache.append((j, position_hash, block.block_id))

            prev_hash = block_hash
            cached_any = True
            updated_num_verified_tokens = max(
                updated_num_verified_tokens, block_end_token_idx
            )

        # ========== 批量写入块池缓存 ==========
        for mapping_index, block_hash, block_id in blocks_to_cache:
            # 对每个映射位置调用cache_block
            # 注意：同一个block_id会在多个mapping_index下被映射
            self.block_pool.cache_block(
                block_hash,
                self.block_pool.blocks[block_id],
                self.kv_cache_spec.group_id,
                mapping_index=mapping_index,
            )

        return cached_any, prev_hash, updated_num_verified_tokens
```

**关键点**：
- **只有块填满了才缓存**：`block_end_token_idx > num_verified_tokens` 时 break，保证缓存的块内容完整
- **块内每个位置都映射**：FullAttention 特有的细粒度缓存，块内每个token位置都建立哈希→块的映射，支持部分命中
- **链式哈希计算**：和查找时完全一致的哈希链，保证能命中
- **批量写入**：一次性把所有可以缓存的位置都写入 BlockPool 的哈希映射

### 5.3 公共前缀计算：`get_num_common_prefix_blocks`

源码位置：`single_type_kv_cache_manager.py:606-668`

```python
    def get_num_common_prefix_blocks(
        self,
        block_ids: list[int],                            # 请求的block_id列表
        num_estimated_tokens: int | None = None,          # 估计token数（可选）
    ) -> int:
        """
        计算给定block_ids列表中，有多少个块是和"所有其他运行中请求"共享的前缀块。
        用于调度时判断：如果一个请求的前缀很多是共享的，优先调度它，收益更大。
        """
        if num_estimated_tokens is not None:
            max_blocks = (
                num_estimated_tokens + self.block_size - 1
            ) // self.block_size
        else:
            max_blocks = len(block_ids)

        num_common = 0
        # 逐个块检查：ref_cnt > 1 说明被多个请求共享
        for i, block_id in enumerate(block_ids):
            if i >= max_blocks:
                break
            if block_id == self.null_block.block_id:
                continue
            block = self.block_pool.blocks[block_id]
            if block.ref_cnt <= 1:
                # 引用计数<=1，说明只有当前请求在用，不是公共前缀
                break
            num_common += 1
        return num_common
```

**作用**：这是给 Scheduler 的**调度提示**——两个请求共享前缀越多，调度它们连续运行的收益越大（前缀缓存命中率高）。`ref_cnt > 1` 是最简单直接的判断方法。

### 5.4 其他辅助方法

```python
    # single_type_kv_cache_manager.py:670-678
    def get_num_blocks_to_allocate(
        self, seq_len, num_computed_blocks, num_new_tokens
    ) -> int:
        # FullAttention 直接用基类的简单计算
        return super().get_num_blocks_to_allocate(
            seq_len, num_computed_blocks, num_new_tokens
        )

    # single_type_kv_cache_manager.py:582-604
    def find_longest_partial_cache_hit(self, *args, **kwargs):
        # 部分命中查找：FullAttention已经在find_longest_cache_hit里处理了部分尾部，
        # 所以这个方法直接返回空
        return []

    # single_type_kv_cache_manager.py:680
    def can_allocate_more_tokens(self, *args, **kwargs):
        # FullAttention 不需要滑动窗口，可以一直分配
        return True
```

---

## 六、其他 Manager 简要概述

以下子类用于混合模型或特殊场景，纯 Full Attention 模型不会用到，了解即可。

### 6.1 SlidingWindowManager

- **适用场景**：Sliding Window Attention（如 Mistral-7B-v0.3 以后的部分模型）
- **核心特点**：每个token只能看到最近 `window_size` 个token，旧的KV会被"滑出窗口"
- **与FullAttention区别**：
  - 只在**块边界**计算哈希（`caching_at_block_boundaries_only=True`）
  - 不支持块内部分token命中
  - 分配块数时只计算窗口内需要的，旧块可以释放
- **前缀缓存策略**：配置了 `prefix_cache_retention_interval`，每隔N个块保留一个块作为稀疏前缀，平衡命中率和内存

### 6.2 RSWAManager (Retrieval Sliding Window Attention)

- **适用场景**：带全局token检索的滑动窗口注意力（部分改进型SWA）
- **核心特点**：在SWA基础上，有少量"全局token"可以看到全文
- **与SlidingWindowManager区别**：额外管理全局token的KV块，逻辑更复杂

### 6.3 MambaManager

- **适用场景**：Mamba、RWKV 等SSM（状态空间模型）架构
- **核心特点**：
  - 没有传统的 K/V 矩阵，而是"状态"（state）
  - 状态是跨chunk流动的，不能简单按块缓存
  - 前缀缓存机制和Attention完全不同
- **前缀缓存策略**：同样使用稀疏保留策略，不缓存每一个块

---

## 七、设计要点小结（纯 FullAttention 视角）

1. **分层职责清晰**：SingleTypeKVCacheManager 管"单类型分配逻辑"，BlockPool 管"块和哈希的存储"，互不越界
2. **链式哈希前缀缓存**：FullAttention 的核心，每个块的哈希依赖前一个块的哈希，保证前缀的唯一性和连续性
3. **细粒度部分命中**：块内每个token位置都建立哈希映射，最后一块即使不满也能命中部分token，减少冗余计算
4. **引用计数共享**：多个请求可以安全共享同一个前缀块，只有最后一个释放时才回收
5. **LIFO 逆序释放**：从尾块开始释放，利用栈的特性让最近使用的块最先被重新分配，提高续生成命中率
6. **驱逐优先级**：优先驱逐 cached 状态的前缀缓存块（没人在使用的），最后才会因为内存不足抢占运行中的请求
7. **抽象基类统一接口**：不管底层是 FullAttention 还是 SWA/Mamba，上层 Coordinator 都可以用同样的接口调用，这是支持混合模型的基础
