# SingleTypeKVCacheManager 详解

## 一、是什么

`SingleTypeKVCacheManager` 是五层 KV Cache 管理架构中的第三层——**单类型 KV 缓存管理器**。它负责管理**一种具体 Attention/SSM 类型**的 KV Cache 分配、命中查找、释放等所有逻辑。

对于纯 Full Attention 模型（Llama、Qwen、Mistral 等），核心使用的是它的子类 `FullAttentionManager`，实现了**链式哈希前缀缓存**机制，可以在请求之间高效共享相同前缀的 KV 缓存。

其他子类（SlidingWindowManager、RSWAManager、ChunkedLocalAttentionManager、MambaManager、CrossAttentionManager、SinkFullAttentionManager 等）用于支持混合模型或特殊注意力类型，本文最后会简要概述。

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
| **阶段2：前缀查找** | 在 `cached_block_hash_to_block` 中查找最长已计算前缀，返回命中块列表和命中 token 数 | `find_longest_cache_hit`（classmethod） |
| **阶段3：touch命中块** | 对命中的块调用 `block_pool.touch()`：ref_cnt += 1，从 `free_block_queue` 移除，防止被驱逐 | `add_local_computed_blocks` |
| **阶段3：计算新块数** | 根据总 token 数和已命中块数，计算需要新分配多少块 | `get_num_blocks_to_allocate` |
| **阶段3：分配新块** | 从 `free_block_queue` 头部取无哈希块分配，ref_cnt=1；处理部分命中 CoW；新块 id 加入 `new_block_ids` | `allocate_new_blocks` |
| **阶段3：维护req_to_blocks** | 将命中块+新块按顺序加入 `self.req_to_blocks[request_id]`，这就是逻辑 block_table 的真正存储位置 | `add_local_computed_blocks` / `allocate_new_blocks` |
| **阶段5：写入缓存** | prompt 计算完成后，对填满的新块把哈希写入 `cached_block_hash_to_block`，供后续命中 | `cache_blocks` → `block_pool.cache_full_blocks` |
| **阶段6：释放块** | 请求结束时逆序遍历块列表，ref_cnt -= 1；ref_cnt=0 的块回收到 `free_block_queue` | `free` / `pop_blocks_for_free` |

---

## 三、类继承结构

```
SingleTypeKVCacheManager（ABC 抽象基类）—— 定义单类型管理器的标准接口和通用逻辑
├── FullAttentionManager        ← 本文核心：全注意力前缀缓存
│   ├── RSWAManager             ← 参考滑动窗口注意力（R-SWA）
│   └── SinkFullAttentionManager ← Sink 注意力
├── SlidingWindowManager        ← 滑动窗口注意力（简要概述）
├── ChunkedLocalAttentionManager ← 分块局部注意力
├── MambaManager                ← Mamba/SSM 模型（简要概述）
└── CrossAttentionManager       ← 交叉注意力（encoder-decoder）
```

**抽象基类的意义**：统一接口，上层 `KVCacheCoordinator` 可以用一致的方式管理不同类型的 KV 组，不需要关心底层是 FullAttention 还是 SWA。

---

## 四、SingleTypeKVCacheManager 基类详解

基类实现了所有管理器共有的逻辑，子类只需要实现特定的差异部分（如前缀查找算法、跳过 token 计算等）。

### 4.1 构造函数

源码位置：`single_type_kv_cache_manager.py:44-126`

```python
class SingleTypeKVCacheManager(ABC):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,                     # 该组的 KV Cache 规格（FullAttentionSpec等）
        block_pool: BlockPool,                          # 所属的块池（全局唯一）
        enable_caching: bool,                           # 是否启用前缀缓存
        kv_cache_group_id: int,                         # 本组的group_id（纯FullAttention=0）
        scheduler_block_size: int,                      # 调度器对齐块大小（LCM，纯FullAttention=block_size）
        dcp_world_size: int = 1,                        # 分布式KV传输world size
        pcp_world_size: int = 1,                        # 前缀缓存持久化world size
        needs_kv_cache_zeroing: bool = False,           # 新分配的块是否需要Worker侧清零
        max_admission_blocks_per_request: int | None = None,  # 单请求最大接纳块数（SWA用）
    ) -> None:
        self.scheduler_block_size = scheduler_block_size
        self.block_size = kv_cache_spec.block_size      # 本组的块大小（每个块存多少token）
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self._max_admission_blocks_per_request = max_admission_blocks_per_request

        # ── 新块清零开关（record new block ids） ──
        # 只有需要清零且 spec 类型在下面集合里的 manager 才记录新块 id
        self._record_new_block_ids = needs_kv_cache_zeroing and type(kv_cache_spec) in (
            FullAttentionSpec, TQFullAttentionSpec, MLAAttentionSpec, HiddenStateCacheSpec,
        )
        self.new_block_ids: list[int] = []

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
- `new_block_ids`：对应总览 5.3/5.4 阶段——分配新块时把块 id 加入这个列表，Scheduler 调度完后通过`take_new_block_ids()` drain 走，发给 Worker 清零
- `req_to_blocks[request_id]`：对应总览 5.3 返回的 `KVCacheBlocks`——这个 list 就是请求持有的块列表，后续 forward、释放都从这里取
- `_null_block`：对齐用，不赘述
- `_pending_cow_copies`：Copy-on-Write 复制队列，同样是 drain 模式，Worker 需要复制块数据时从这里取

**新块清零 drain 方法**（源码 `single_type_kv_cache_manager.py:376-380`）：
```python
def take_new_block_ids(self) -> list[int]:
    """Drain and return block IDs allocated since the last call."""
    ids = self.new_block_ids
    self.new_block_ids = []
    return ids
```
这就是总览 5.4 阶段 Worker 清零的数据源——每个 manager 自己记自己的新块，顶层 KVCacheManager 遍历所有 manager 汇总。

**构造函数关键点**：
- 纯 Full Attention 场景下，`scheduler_block_size == block_size == 16`（假设），没有倍数关系
- `_record_new_block_ids=True`，所以 FullAttention 的新块都会被记录等待清零
- `req_to_blocks` 是真正持有请求块列表的地方，不是 Request 对象的字段

### 4.2 核心方法：`get_num_blocks_to_allocate`

源码位置：`single_type_kv_cache_manager.py:144-230`

```python
def get_num_blocks_to_allocate(
    self,
    request_id: str,
    num_tokens: int,                      # 需要槽位的总 token 数（含已分配）
    new_computed_blocks: Sequence[KVCacheBlock],  # 刚命中的前缀块
    total_computed_tokens: int,           # 本地+外部总共已计算 token 数
    num_local_computed_tokens: int,       # 本地前缀缓存命中 token 数
    num_tokens_main_model: int,           # 主模型 token 数（投机解码时不含 lookahead）
    apply_admission_cap: bool = False,    # 是否应用准入上限（SWA/ChunkedLocal用）
) -> int:
    num_required_blocks = cdiv(num_tokens, self.block_size)
    if apply_admission_cap and self._max_admission_blocks_per_request is not None:
        num_required_blocks = min(
            num_required_blocks, self._max_admission_blocks_per_request
        )
    num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

    if request_id in self.num_cached_block:
        # 快路径：running 请求不会再有新的前缀命中
        assert len(new_computed_blocks) == 0
        return max(num_required_blocks - num_req_blocks, 0)

    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
    num_skipped_blocks = num_skipped_tokens // self.block_size
    num_new_blocks = max(
        num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
        0,
    )
    # ... 部分命中 CoW 预留 +1、驱逐候选块计数等
    return num_new_blocks + num_evictable_blocks
```

**作用**：计算本轮需要新分配多少物理块，这是分配前的"容量预估"。

关键点：
- `cdiv(num_tokens, block_size)` 向上取整得到总块数，减去已持有的块数
- **running 请求快路径**：已在 `num_cached_block` 中的请求不会再有新前缀命中，直接用 `num_required_blocks - num_req_blocks`
- **滑动窗口跳过**：`get_num_skipped_tokens` 算出窗口外 token 数，跳过块不占新分配
- **部分命中 CoW 预留**：若命中落在块内边界（`_has_partial_local_hit`），额外 +1 块用于 CoW 重定向
- 内部还统计驱逐候选块数量（`_get_num_evictable_blocks`），因为它们在 touch 后会被移出空闲队列，必须计入容量检查

### 4.3 核心方法：`add_local_computed_blocks`

源码位置：`single_type_kv_cache_manager.py:232-289`

```python
def add_local_computed_blocks(
    self,
    request_id: str,
    new_computed_blocks: Sequence[KVCacheBlock],   # 本次命中的新块列表
    num_local_computed_tokens: int,               # 本地命中 token 数
    num_external_computed_tokens: int,            # 外部命中 token 数
) -> None:
    """处理命中的块：增加引用计数，加入 req_to_blocks"""
    req_blocks = self.req_to_blocks[request_id]
    assert len(req_blocks) == 0       # coordinator 只在首次分配时调用
    num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
    num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
    num_skipped_blocks = num_skipped_tokens // self.block_size
    if num_skipped_blocks > 0:
        new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

    # touch 命中块，防止被驱逐
    if self.enable_caching:
        self.block_pool.touch(new_computed_blocks)
    else:
        assert not any(new_computed_blocks), "..."

    # 跳过的块用 null_block 填充
    req_blocks.extend([self._null_block] * num_skipped_blocks)
    req_blocks.extend(new_computed_blocks)
    # 标记已缓存块数，cache_blocks() 不会重复缓存
    self.num_cached_block[request_id] = len(req_blocks)
    if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
        # 部分命中：记录尾块用于 CoW 重定向
        block_idx = num_local_computed_tokens // self.block_size
        self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
        self.num_cached_block[request_id] = block_idx
```

**关键点**：
- 命中缓存块不是简单"拿来用"，必须调用 `block_pool.touch()`：增加 `ref_cnt` 并从空闲队列摘出，防止被驱逐
- **零断言保障**：coordinator 只在请求首次分配时调用本方法，此时 `req_to_blocks[request_id]` 必为空
- 滑动窗口场景下，跳过的块用 `null_block` 填充，保持 block_table 长度与位置对齐
- 部分命中（命中落在块内）时，把尾块记录到 `_partial_hit_reqs`，供后续 `allocate_new_blocks` 做 CoW 重定向

### 4.4 核心方法：`allocate_new_blocks`

源码位置：`single_type_kv_cache_manager.py:330-369`

```python
def allocate_new_blocks(
    self,
    request_id: str,
    num_tokens: int,                    # 需要槽位的总 token 数
    num_tokens_main_model: int,         # 主模型 token 数
) -> list[KVCacheBlock]:
    """为请求分配新块，使其至少有 num_tokens 个 token 槽位"""
    cow_blocks: list[KVCacheBlock] = []
    if request_id in self._partial_hit_reqs:
        # 部分命中：把共享尾块重定向到私有 CoW 副本
        block_idx, source_block = self._partial_hit_reqs.pop(request_id)
        cow_block = self.block_pool.get_new_blocks(1)[0]
        self._apply_cow(request_id, block_idx, source_block, cow_block)
        self.new_block_ids.append(cow_block.block_id)
        cow_blocks.append(cow_block)

    req_blocks = self.req_to_blocks[request_id]
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_new_blocks = num_required_blocks - len(req_blocks)
    if num_new_blocks <= 0:
        return cow_blocks
    else:
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks)
        return cow_blocks + new_blocks
```

**分配流程**：
1. **部分命中 CoW 重定向**：若该请求有部分命中记录，先从块池取一个新块作为 CoW 副本，把共享尾块替换为私有副本（`_apply_cow`），并把 CoW 请求加入 `_pending_cow_copies` 等待 Worker 复制
2. 计算还需多少新块：`cdiv(num_tokens, block_size) - len(req_blocks)`
3. 从块池 `get_new_blocks` 取块，追加到 `req_to_blocks`，新块 id 记录进 `new_block_ids` 等待清零
4. 返回新块（cow_block 在前，普通新块在后）

> 注意：这里的分配**不做驱逐**——空闲块不足时 `get_new_blocks` 会抛异常，由上层 KVCacheManager 在 `allocate_slots` 的容量检查中提前拦截，触发抢占而非直接崩溃。

### 4.4.1 端到端分配流程串讲（对应总览阶段3）

`UnitaryKVCacheCoordinator.allocate_*` 调用本类方法的完整顺序（以纯 FullAttention 单组为例）：

```python
# 对应 kv_cache_coordinator.py 中 Unitary 的 allocate 流程（简化）
def allocate_slots(self, request, num_new_tokens, new_computed_blocks, ...):
    blocks = self.req_to_blocks[request.request_id]

    # ① 若上一轮有已命中块（续写场景），本次命中块需与前块链式衔接（由 find 处理）

    # ② touch 本次新命中的块，增加 ref_cnt，从 free_block_queue 移除
    self.managers[0].add_local_computed_blocks(
        request.request_id, new_computed_blocks,
        num_local_computed_tokens, num_external_computed_tokens,
    )

    # ③ 计算需要新分配多少块
    num_new_blocks = self.managers[0].get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_blocks,
        total_computed_tokens=...,
        num_local_computed_tokens=...,
        num_tokens_main_model=...,
    )

    # ④ 分配新块（含部分命中 CoW 重定向）
    new_blocks = self.managers[0].allocate_new_blocks(
        request.request_id, num_tokens_need_slot, num_tokens_main_model
    )
```

**关键点**：`req_to_blocks[request_id]` 就是在这里一步步维护起来的——它的内容就是 [历史块..., 本次命中块..., 本次新分配块...]，顺序就是 block_table 在 GPU 上的顺序。

### 4.5 核心方法：`cache_blocks`（缓存写入）

源码位置：`single_type_kv_cache_manager.py:427-477`（基类统一实现）

```python
def cache_blocks(
    self,
    request: Request,
    num_tokens: int,                    # 需要缓存的总 token 数（含已缓存的）
    retention_interval: int | None = None,  # 稀疏保留间隔（SWA用，FullAttention忽略）
) -> None:
    """把满块写入前缀缓存"""
    num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
    num_full_blocks = num_tokens // self.block_size

    if num_cached_blocks >= num_full_blocks:
        return    # 幂等：已缓存完，跳过

    # 计算可缓存掩码（默认为 None，表示全缓存；SWA/Mamba 会覆盖）
    block_mask = self.reachable_block_mask(...)
    self.block_pool.cache_full_blocks(
        request=request,
        blocks=self.req_to_blocks[request.request_id],
        num_cached_blocks=num_cached_blocks,
        num_full_blocks=num_full_blocks,
        block_size=self.block_size,
        kv_cache_group_id=self.kv_cache_group_id,
        block_mask=block_mask,
    )
    self.num_cached_block[request.request_id] = num_full_blocks
```

**关键点**：
- **只缓存满块**：`num_full_blocks = num_tokens // block_size`，尾块不缓存
- **幂等**：`num_cached_blocks >= num_full_blocks` 时直接返回，多次调用安全
- **哈希不在这里算**：真正计算哈希并插入映射的是 `block_pool.cache_full_blocks`，它从 `request.block_hashes` 取预计算哈希
- **FullAttention 额外处理**：`FullAttentionManager.cache_blocks`（779-789）在基类基础上，若 `block_size != hash_block_size` 会额外调用 `_cache_partial_tail_block` 缓存 prompt 尾块（块内部分边界）

> 注意：本文最初草稿中提到的 `maybe_save_new_kv_blocks_to_cache` **在该版本源码中不存在**，统一由 `cache_blocks` 承担。

### 4.6 核心方法：`pop_blocks_for_free` / `free`（释放流程）

源码位置：`single_type_kv_cache_manager.py:500-527`，对应总览阶段 6 的释放流程。

#### 4.6.1 `pop_blocks_for_free`：取出请求的块列表（不真正归还）

```python
def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
    """从 req_to_blocks 中弹出该请求的块列表，清理相关统计，但不归还 BlockPool。
    调用方拿到块列表后，后续负责调用 free_blocks 归还。"""
    req_blocks = self.req_to_blocks.pop(request_id, [])  # 弹出块列表，移除映射
    self.num_cached_block.pop(request_id, None)          # 清理缓存统计
    self._partial_hit_reqs.pop(request_id, None)         # 清理部分命中记录
    return req_blocks                                    # 返回按分配顺序排列的块列表
```

#### 4.6.2 `free`：完整释放单个请求的所有块（最常用）

```python
def free(self, request_id: str) -> None:
    """释放请求的所有块：弹出块列表 → 逆序归还 BlockPool"""
    # 关键：reversed() 逆序释放——尾块先归还，利用 free_block_queue 的 LIFO 特性
    # 这样请求被抢占后重新调度时，原来的尾块会最先被分配回来，提高续生成命中率
    self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

#### 4.6.3 `free_blocks`（BlockPool 方法）：引用计数减一+队列回收

```python
# BlockPool.free_blocks（block_pool.py:719-742）
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_with_hash = []
    blocks_without_hash = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_hash is None and self.enable_caching:
                blocks_without_hash.append(block)   # 无哈希：队首优先驱逐
            else:
                blocks_with_hash.append(block)      # 有哈希：队尾LRU保护
    self.free_block_queue.prepend_n(blocks_without_hash) # 无哈希放队首（先复用）
    self.free_block_queue.append_n(blocks_with_hash)     # 有哈希放队尾（LRU缓存）
```

**释放逻辑关键点**：
- **两阶段释放**：先 pop 取出块列表（清理上层映射），再逆序调用 free_blocks 归还（底层 BlockPool 处理引用计数和队列）
- **逆序释放**：从最后一个块开始放，`free_block_queue` 是双向链表——无哈希块放队首（下次分配优先拿，不用清零），有哈希块放队尾（LRU 保护，优先驱逐）
- **引用计数机制**：多个请求共享前缀块时，只有最后一个请求释放时 `ref_cnt` 才会到 0，块才会真正被回收到队列
- **哈希决定去向**：`block_hash is None` → 内容不完整，放队首优先复用；`block_hash is not None` → 内容完整，放队尾进入前缀缓存池

---

## 五、FullAttentionManager 详解

这是纯 Full Attention 模型的核心管理器，实现了完整的链式哈希前缀缓存机制。

### 5.1 前缀查找：`find_longest_cache_hit`

源码位置：`single_type_kv_cache_manager.py:681-777`

这是前缀缓存的核心方法，是一个 **classmethod**（不依赖实例状态），实现了**最长前缀匹配查找**，找到请求序列中有多少 KV 块已经在缓存中了。

#### 5.1.1 方法签名

```python
@classmethod
def find_longest_cache_hit(
    cls,
    block_hashes: BlockHashList,          # 请求的哈希列表（Request 预计算）
    max_length: int,                      # 最大查找长度（token 数）
    kv_cache_group_ids: list[int],        # 需要同时命中的所有 group id
    block_pool: BlockPool,                # 块池
    kv_cache_spec: KVCacheSpec,           # 该组 spec
    drop_eagle_block: bool,               # EAGLE/MTP 是否丢最后一块
    alignment_tokens: int,                # 返回的命中长度需对齐的 token 数
    dcp_world_size: int = 1,              # 分布式 KV 传输 world size
    pcp_world_size: int = 1,              # 前缀缓存持久化 world size
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    """返回：(按组的命中块列表, 命中 token 精确长度)"""
```

#### 5.1.2 逐段逻辑详解

**第一阶段：对齐哈希粒度**

```python
block_size = kv_cache_spec.block_size
if dcp_world_size > 1:
    block_size *= dcp_world_size
# 把 Request 的哈希从 hash_block_size 粒度对齐到本组 block_size 粒度
block_hashes = resolve_block_hashes(
    block_hashes, block_pool.hash_block_size, block_size,
    supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
    alignment_tokens=alignment_tokens,
)
# 细粒度模式：alignment_tokens < block_size 时，可探块内边界
fine_grained = (alignment_tokens < block_size and block_size % alignment_tokens == 0)
```

**第二阶段：逐块查找满块命中**

```python
computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
    [] for _ in range(len(kv_cache_group_ids))
)
# Phase 1: 从开头找最长的一段已缓存满块 run
for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
    cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
    if not cached_block:
        break                       # 链式哈希：miss 后面必然全 miss
    for computed, cached in zip(computed_blocks, cached_block):
        computed.append(cached)
hit_length = len(computed_blocks[0]) * block_size
```

**第三阶段（细粒度）：探第一块内部的边界命中**

```python
if fine_grained:
    # 从高到低探测第一块内部的 hash 边界（最长命中优先）
    for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
        cached_tail = block_pool.get_cached_block(block_hashes[fine_idx], kv_cache_group_ids)
        if not cached_tail:
            continue
        for computed, cached in zip(computed_blocks, cached_tail):
            computed.append(cached)
        hit_length = (fine_idx + 1) * alignment_tokens
        break
```

**第四阶段：EAGLE 丢块 + 对齐收尾**

```python
if drop_eagle_block and hit_length > 0:
    hit_length -= min(alignment_tokens, block_size)   # EAGLE 重算生成点前一块
hit_length -= hit_length % alignment_tokens            # 对齐到 alignment_tokens
num_blocks = cdiv(hit_length, block_size)
for computed in computed_blocks:
    del computed[num_blocks:]                          # 裁剪超出命中长度的块
return computed_blocks, hit_length
```

#### 5.1.3 链式哈希图解（block_size=16，34token 例子）

```
请求 token 序列（34 个 token，block_size=16）：
  [T0-T15]  → block 0 （满块）
  [T16-T31] → block 1 （满块）
  [T32,T33] → block 2 （只有 2 个 token，不满）

哈希链计算（由 Request 预计算）：
block_hash_0 = H(null, T0..T15, lora_id)          → 在缓存中找到命中
block_hash_1 = H(block_hash_0, T16..T31, lora_id) → 在缓存中找到命中
block_hash_2 = H(block_hash_1, T32..T33, lora_id) → 查找失败，前缀断裂

返回结果：
  computed_blocks = ([块A, 块B],)     ← 命中前 2 个满块
  hit_length = 32
```

**链式哈希特点**：
1. 每个块的哈希依赖前一个块的哈希 → 保证前缀连续性，避免"中间某块相同但前缀不同"的错误命中
2. 即使两个块内容完全相同，如果前缀不同，哈希也不同 → 天然避免哈希碰撞导致的错误复用
3. 细粒度模式下，块内每个 hash 边界都有独立映射 → 支持命中落在块内（续写场景常见）

### 5.2 缓存写入：`cache_blocks`

源码位置：`single_type_kv_cache_manager.py:779-819`

```python
def cache_blocks(self, request, num_tokens, retention_interval=None):
    # 1. 先走基类：把满块写入前缀缓存
    super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
    # 2. 若 hash_block_size != block_size，额外缓存 prompt 尾块（块内部分边界）
    hash_block_size = self.block_pool.hash_block_size
    if self.block_size == hash_block_size:
        return
    self._cache_partial_tail_block(request, num_tokens)
```

**关键点**：
- **核心逻辑在基类 `cache_blocks`**（见 4.5），它调用 `block_pool.cache_full_blocks` 计算哈希并写入映射
- **FullAttention 特有增强**：当 `hash_block_size < block_size`（混合模型多粒度），额外调用 `_cache_partial_tail_block`，只缓存 prompt 尾块**最后一个 hash 边界**，中间边界故意跳过（减少缓存条目）
- 这部分是"块内部分命中"的**写入侧**，与 5.1 第三阶段（细粒度查找）对应

### 5.3 公共前缀计算：`get_num_common_prefix_blocks`

源码位置：`single_type_kv_cache_manager.py:821-829`

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
    blocks = self.req_to_blocks[running_request_id]
    num_common_blocks = 0
    for block in blocks:
        # 该块被所有已分配 KV cache 的请求共享 → 是公共前缀
        if block.ref_cnt == len(self.req_to_blocks):
            num_common_blocks += 1
        else:
            break
    return num_common_blocks
```

**作用**：这是给 Scheduler 的**调度提示**——两个请求共享前缀越多，调度它们连续运行的收益越大（前缀缓存命中率高）。判断标准是 `ref_cnt == len(self.req_to_blocks)`（即所有请求都在用这块）。

### 5.4 其他辅助方法

```python
# remove_skipped_blocks：FullAttention 不跳过任何 token，基类默认 no-op
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    return 0    # 基类默认：FullAttention 从不跳过 token

# can allocate more：FullAttention 没有滑动窗口，每步都分配
```

---

## 六、其他 Manager 简要概述

以下子类用于混合模型或特殊场景，纯 Full Attention 模型不会用到，了解即可。

### 6.1 SlidingWindowManager

源码位置：`single_type_kv_cache_manager.py:878-1093`

- **适用场景**：Sliding Window Attention（如 Mistral、Gemma 的部分模型）
- **核心特点**：每个 token 只能看到最近 `window_size` 个 token，旧的 KV 会被"滑出窗口"
- **与 FullAttention 区别**：
  - 只在**块边界**查找哈希（`caching_at_block_boundaries_only`），不支持块内部分 token 命中
  - `get_num_skipped_tokens` 返回窗口外 token 数，`remove_skipped_blocks` 会真正释放窗口外块并替换为 `null_block`
  - `reachable_block_mask` 返回掩码，只在可命中的块上建立缓存
- **前缀缓存策略**：配置了 `prefix_cache_retention_interval`，每隔 N 个块保留一个块作为稀疏前缀，平衡命中率和内存

### 6.2 RSWAManager（Reference Sliding Window Attention）

源码位置：`single_type_kv_cache_manager.py:832-876`

- **适用场景**：带全局 token 检索的滑动窗口注意力（部分改进型 SWA）
- **核心特点**：在 SWA 基础上，有少量"全局 token"可以看到全文，且会驱逐中间 gap 块（而非头部前缀）
- **与 SlidingWindowManager 区别**：`remove_skipped_blocks` 需要 `num_prompt_tokens` 参数，驱逐的是 prefill 尾与当前窗口之间的 gap 块

### 6.3 MambaManager

源码位置：`single_type_kv_cache_manager.py:1253-1745`

- **适用场景**：Mamba、RWKV 等 SSM（状态空间模型）架构
- **核心特点**：
  - 没有传统的 K/V 矩阵，而是"状态"（state）
  - 状态是跨 chunk 流动的，不能简单按块缓存
  - 前缀缓存机制和 Attention 完全不同，`cache_blocks` 逻辑更复杂
- **前缀缓存策略**：同样使用稀疏保留策略，不缓存每一个块

### 6.4 CrossAttentionManager / SinkFullAttentionManager / ChunkedLocalAttentionManager

- `CrossAttentionManager`（1747）：encoder-decoder 模型（如 Whisper）的交叉注意力，处理静态 encoder KV
- `SinkFullAttentionManager`（1810）：Sink 注意力，sink block 常驻
- `ChunkedLocalAttentionManager`（1095）：块内局部注意力（如 GLM-4v）

---

## 七、设计要点小结（纯 FullAttention 视角）

1. **分层职责清晰**：SingleTypeKVCacheManager 管"单类型分配逻辑"，BlockPool 管"块和哈希的存储"，互不越界
2. **链式哈希前缀缓存**：FullAttention 的核心，每个块的哈希依赖前一个块的哈希，保证前缀的唯一性和连续性
3. **细粒度部分命中**：`find_longest_cache_hit` 在细粒度模式下可探块内 hash 边界，最后一块即使不满也能命中部分 token，减少冗余计算
4. **引用计数共享**：多个请求可以安全共享同一个前缀块，只有最后一个释放时才回收
5. **LIFO 逆序释放**：从尾块开始释放，利用栈的特性让最近使用的块最先被重新分配，提高续生成命中率
6. **部分命中 CoW**：`_partial_hit_reqs` + `_apply_cow` 处理命中落在块内的情况，把共享尾块重定向到私有副本，保证不污染共享缓存
7. **抽象基类统一接口**：不管底层是 FullAttention 还是 SWA/Mamba，上层 Coordinator 都可以用同样的接口调用，这是支持混合模型的基础
8. **为 classmethod 的查找**：`find_longest_cache_hit` 不依赖实例状态，便于 fine-grained / 多 group 场景复用