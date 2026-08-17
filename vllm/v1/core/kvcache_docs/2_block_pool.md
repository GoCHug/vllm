# vLLM V1 BlockPool 逻辑块池层（Full Attention 主线）

> 五层架构第 2 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`1_physical_memory.md`](./1_physical_memory.md) ｜上层 ➔ [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)
>
> 源文件：`vllm/vllm/v1/core/block_pool.py`、`vllm/vllm/v1/core/kv_cache_utils.py`
>
> 本文以纯 Full Attention 单 group 模型（如 Llama、Qwen、Mistral）为主线讲解 BlockPool。多 group 混合模型场景在文末"扩展"章节简要提及。

---

## 一、是什么

`BlockPool` 是 KV Cache 管理五层架构的**逻辑块池层**，负责管理所有 `KVCacheBlock` 逻辑块的分配、释放、缓存和驱逐。

它**不持有任何 GPU 显存指针**，只管理逻辑元数据：
- `block_id`：整数编号 `[0, num_blocks-1]`，与物理张量第 0 维行号一一对应
- `ref_cnt`：引用计数，跟踪多少请求正在使用某块
- `block_hash`：链式哈希指纹，用于前缀缓存命中查找
- 空闲链表指针：实现 LRU 驱逐策略

BlockPool 的核心价值是把"显存管理"简化为"整数 ID 管理"——调度器全程只操作 `block_id` 整数，不搬移任何 GPU 显存数据。

---

## 二、干什么用

在纯 Full Attention 模型中，BlockPool 承担以下职责：

| 职责 | 说明 |
|------|------|
| **空闲块管理** | 维护 LRU 空闲块队列，决定新块从哪里分配、哪些块优先驱逐 |
| **前缀缓存命中** | 维护 `hash → block` 映射表，通过链式哈希快速定位可复用的前缀 block |
| **引用计数共享** | `ref_cnt` 跟踪共享状态，多个请求命中相同前缀时共享物理块（零拷贝） |
| **生命周期管理** | `touch`/`free_blocks`/`get_new_blocks` 等方法管理 block 从分配到释放的全生命周期 |
| **事件广播** | 向 KV connector 等旁路组件广播 `BlockStored`/`BlockRemoved` 事件 |

**纯 Full Attention 单 group 模型下的关键简化**：
- 全模型只有一个 group，`group_id = 0`，不存在跨组命中对齐问题
- `scheduler_block_size == hash_block_size == group.block_size`，三种 block_size 完全相等
- 哈希表 key 为 `(block_hash, group_id=0)`，查就是了，不需要"所有组同时命中"的判断

---

## 三、核心数据结构

BlockPool 初始化在 `__init__`（`block_pool.py:162-196`）中完成。核心数据结构分四组：

### 3.1 全局块数组 `blocks` 与 `KVCacheBlock`

`blocks: list[KVCacheBlock]` 是按 `block_id` 索引的全部逻辑块数组，启动时一次性创建 `KVCacheBlock(i) for i in range(num_gpu_blocks)`。

`KVCacheBlock`（`kv_cache_utils.py:117-176`）是 BlockPool 管理的基本单元——一个"轻量元数据壳"，只承载四类元数据，**不持有任何 torch.Tensor / 显存指针**：

```python
@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""
    block_id: int                                    # ① 编号：全局唯一，=物理张量行号
    ref_cnt: int = 0                                 # ② 引用计数：多少请求正在使用此块
    _block_hash: BlockHashWithGroupId | None = None  # ③ 内容哈希指纹（带group_id）
    _block_hash_num_tokens: int | None = None        # ③ 哈希覆盖的token数
    prev_free_block: "KVCacheBlock | None" = None    # ④ 空闲链表前驱指针
    next_free_block: "KVCacheBlock | None" = None    # ④ 空闲链表后继指针
    is_null: bool = False                            # ② 是否为null_block占位块
```

#### 字段语义（按职责分四组）

| 职责 | 字段 | 说明 |
|---|---|---|
| **① 编号** | `block_id: int` | 全局编号 `[0, num_gpu_blocks-1]`，创建后不变；同时是物理张量第 0 维行号、`blocks` 列表下标 |
| **② 生命周期** | `ref_cnt: int` | 引用计数。新分配=`1`，命中前缀时自增（共享），释放时自减；归零才能进入空闲队列被驱逐/重用 |
| **② 生命周期** | `is_null: bool` | 是否为 `null_block`（`block_id=0`）。null_block **不维护 ref_cnt、不进空闲队列、不可释放**，用于对齐 block_table 长度 |
| **③ 哈希指纹** | `_block_hash` | 该 block 内容的哈希 key（带 group_id）。仅当 block **写满且入缓存**时才设置；`None` 表示未缓存/已驱逐 |
| **③ 哈希指纹** | `_block_hash_num_tokens` | 哈希覆盖的前缀 token 数。满块时等于 `block_size`；部分块场景下小于 `block_size`（Full Attention 单 group 通常是满块） |
| **④ 链表指针** | `prev_free_block` / `next_free_block` | 空闲块双向链表指针，**仅由 `FreeKVCacheBlockQueue` 操作** |

#### 关键认知：block 代表"一组 token 在所有层的 KV"

同一个 `block_id=5` 在每一层的物理张量中都占用第 5 行，但它们存储的是**同一组 token** 的 K/V 数据（layer0 的 K/V、layer1 的 K/V、...、layer31 的 K/V）。因此 block 上只挂**一个**基于 token 内容的 hash（而不是每层一个）——hash 的语义是"这组 token 的内容指纹"，与层无关。

#### 哈希状态机

```python
# 写入：block 被写满并入缓存时调用
def set_block_hash(self, block_hash, num_tokens=None):
    assert self.block_hash is None and self._block_hash_num_tokens is None, (
        "The block already has a hash. This should not happen."
    )
    self._block_hash = block_hash
    self._block_hash_num_tokens = num_tokens

# 清空：block 被驱逐时调用
def reset_hash(self):
    """Reset the block hash when the block is evicted."""
    self._block_hash = None
    self._block_hash_num_tokens = None
```

断言保证"一块一哈希"：一块在生命期内只允许设置一次主哈希，要换内容必须先 `reset_hash` 走驱逐流程。这是前缀缓存正确性的基础保障。

#### `null_block` 的特殊性

`BlockPool.__init__` 启动时立即把 `block_id=0` 从空闲队列摘出，置 `is_null=True`，作为全局占位块。用于填充 block table 中不需要实际 KV 数据的位置。所有释放/计数路径都对其特判，跳过 `ref_cnt` 维护。

> 实际可分配块数 = `num_gpu_blocks - 1`（减去 null_block）。`get_usage()`（`block_pool.py:807-818`）计算使用率时也显式减 1。

### 3.2 空闲块队列 `free_block_queue`（空间调度）

`FreeKVCacheBlockQueue` 是带假头/假尾的双向链表，按驱逐优先级排序：

| 操作 | 语义 |
|------|------|
| `popleft_n(n)` | 从假头侧弹出 n 个块分配出去 → **靠近头部 = 最先被驱逐/重用** |
| `prepend_n(blocks)` | 插到假头之后 → **优先驱逐侧**（无哈希块） |
| `append_n(blocks)` | 插到假尾之前 → **尽量保留侧**（有哈希块，MRU） |
| `remove(block)` | 把某块从链表摘除（`touch` 命中时用，防止被驱逐） |

### 3.3 前缀缓存双向映射（内容复用）

block↔hash 的关系由三处共同维护：

| 字段 | 方向 | 说明 |
|---|---|---|
| `block._block_hash` / `block._block_hash_num_tokens` | block → 主哈希 | 每块**唯一**主哈希及其覆盖的前缀长度 |
| `cached_block_hash_to_block: BlockHashToBlockMap` | hash → block(s) | **正向查询入口**，前缀命中查找用 |
| `cached_block_hashes_by_block: dict[int, set]` | block_id → 别名哈希集合 | **反向索引**，登记除主哈希外的别名键，供清理时枚举 |

> `BlockHashToBlockMap`（`block_pool.py:33-139`）的 key 是 `BlockHashWithGroupId`（`block_hash + group_id`），value 通常是单个 `KVCacheBlock`；当同一 hash 对应多个物理块时，value 是 `{block_id: KVCacheBlock}` 字典。它实现了 `get_one_block` / `insert` / `pop` / `contain` 等接口。

纯 Full Attention 单 group 场景下，主哈希就是块的唯一哈希身份，别名表只在 partial block 等细粒度场景使用。

### 3.4 旁路字段

| 字段 | 说明 |
|---|---|
| `kv_event_queue` | `BlockStored`/`BlockRemoved`/`AllBlocksCleared` 事件队列，供 KV connector 消费 |
| `metrics_collector` | 块驻留时长、分配/驱逐计数 |
| `enable_caching` | 是否启用前缀缓存（影响 free 时的队列策略） |
| `enable_kv_cache_events` | 是否产生 KV 事件 |

---

## 四、关键不变量

BlockPool 的正确性依赖以下不变量：

1. **每块恰好一个主哈希**：`block.block_hash` 不为 `None` 时即唯一主哈希；任何额外同义键只进 `cached_block_hashes_by_block`，不覆盖主哈希
2. **正向表与反向表对齐**：所有别名键都在反向表有记录；清理时主哈希+别名一并从正向表删除
3. **`ref_cnt == 0` ⇔ 处于 free_block_queue**（非 null 块）：ref_cnt 归零的块必须进入空闲链表；ref_cnt > 0 的块被运行中请求持有，不可驱逐
4. **`null_block` 不参与计数/释放**：`get_usage`、`get_num_free_blocks` 等都特判跳过
5. **同一 hash 可挂多个物理块**：`BlockHashToBlockMap` 不去重，保持 block table append-only 性质

---

## 五、核心方法详解（结合源码）

> 以下所有方法签名与行号均与 2026 年代码库 `block_pool.py` 一致。注意：**BlockPool 不直接计算哈希**——哈希由 `Request` 对象在创建/追加 token 时提前算好，BlockPool 只负责"哈希→块的插入/查询/删除"。

### 5.1 前缀缓存查找 `get_cached_block`

```python
# block_pool.py:198-223
def get_cached_block(
    self,
    block_hash: BlockHash,               # 组无关的内容哈希
    kv_cache_group_ids: list[int],       # 需要同时命中的所有 group id
) -> list[KVCacheBlock] | None:
```

**作用**：给定一个内容哈希和 group 列表，返回每个 group 对应的一块；**任一 group miss 即返回 `None`**。

Full Attention 单 group 下，`kv_cache_group_ids=[0]`，实际就是查 `cached_block_hash_to_block[(block_hash, 0)]`。命中返回 `[block]`，未命中返回 `None`。

> 注意返回的是**列表**（每个 group 一块），不是单个 block。这是多 group 接口统一的结果，单 group 时列表长度恒为 1。

### 5.2 分配新块 `get_new_blocks`

```python
# block_pool.py:647-677
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

    ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

    if self.enable_caching:
        for block in ret:
            self._maybe_evict_cached_block(block)  # 清掉旧缓存条目
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            ...
    else:
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            ...
    return ret
```

**流程**：
1. 先检查空闲块数是否足够（不足抛异常，由上层 manager 处理）
2. 从空闲链表**队首**弹出 n 个块（最该被驱逐的优先分配）
3. 若启用前缀缓存且该块仍挂着旧哈希，先 `_maybe_evict_cached_block` 清理（避免旧 hash 错指向新内容）
4. `ref_cnt = 1`，返回给上层

**为什么分配前要清缓存？** 空闲队列里的块可能是"ref_cnt=0 但仍挂在哈希表上"的缓存块（有效缓存条目，在队尾等着被命中）。如果它们被弹出来分配新内容，必须先把旧哈希从映射表中删除——否则新内容会被旧 hash 错误命中。

### 5.3 命中复用 `touch`

```python
# block_pool.py:702-717
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        # ref_cnt=0 means this block is in the free list (i.e. eviction
        # candidate), so remove it.
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # 从空闲队列摘出，不再被驱逐
        block.ref_cnt += 1
```

**作用**：前缀缓存命中后调用——把块的 `ref_cnt++`，如果之前 `ref_cnt=0`（在空闲队列里等着被驱逐），就从空闲队列摘除，防止它被后续分配抢走。

零拷贝共享的核心：命中的块不需要复制数据，只需 `ref_cnt++`。

### 5.4 释放块 `free_blocks`（双队列分流）

```python
# block_pool.py:719-742
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_with_hash = []
    blocks_without_hash = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            # When caching is disabled we always append for better
            # GPU cache locality from reusing recently used blocks
            if block.block_hash is None and self.enable_caching:
                blocks_without_hash.append(block)  # 死块：永不命中
            else:
                blocks_with_hash.append(block)      # 有效缓存条目

    # Blocks without hash get evicted first - prepend them last to the tail
    self.free_block_queue.prepend_n(blocks_without_hash)  # → 队首：优先驱逐
    self.free_block_queue.append_n(blocks_with_hash)       # → 队尾：尽量保留
```

**设计意图**：
- **无哈希块**（`block_hash is None`）：永远无法被前缀缓存命中 → 放队首，下次分配最先弹走（驱逐零成本）
- **有哈希块**：是有效前缀缓存条目 → 放队尾，尽量保留以延长复用机会
- `enable_caching=False` 时统一走 `with_hash` 侧按 recency 追加（仅关缓存时，为了 GPU cache locality 也按最近使用顺序追加）

**为什么要求 `ordered_blocks` 是逆序？** 调用方（manager 的 `free` 方法）传入的 block 列表是按从尾到头的顺序释放的（先释放最新尾部块，再释放老块），使得尾部 block 先入空闲队列、老的前缀 block 后入队列。这保证了 LRU 顺序的正确性。

### 5.5 缓存满块 `cache_full_blocks`

源码位置：`block_pool.py:225-342`

**作用**：把请求里**新填满的块**写入前缀缓存映射表（`cached_block_hash_to_block`），使其成为可被后续请求命中的条目。调用链：`SingleTypeKVCacheManager.cache_blocks`（§4.6）→ 本方法。

**两个方向的映射**：
- `cached_block_hash_to_block: BlockHashWithGroupId → KVCacheBlock`——正向表，查找命中用
- `cached_block_hashes_by_block: block_id → set[BlockHashWithGroupId]`——反向表，记录块的"别名哈希"（一块多哈希时清理用）

#### 5.5.1 完整源码 + 逐行注释

```python
def cache_full_blocks(
    self,
    request: Request,                    # 请求对象（提供预计算的 block_hashes + token_ids）
    blocks: list[KVCacheBlock],          # 请求的全部块（req_to_blocks，顺序即 block_table 顺序）
    num_cached_blocks: int,              # 已缓存的块数（此前轮次已写过哈希的块数）
    num_full_blocks: int,                # 本次应缓存到的满块数（num_tokens // block_size）
    block_size: int,                     # 本组块大小（可能 != hash_block_size）
    kv_cache_group_id: int,              # 本 KV cache 组 ID（拼进哈希 key，组间隔离）
    block_mask: list[bool] | None = None,  # 掩码，False 的块跳过不缓存（SWA 尾窗口 / Mamba 对齐）
) -> None:
    # ===== 第 1 步：幂等检查 =====
    # 没有新增的满块（已缓存数 >= 满块数）直接返回，重复调用安全
    if num_cached_blocks >= num_full_blocks:
        return

    # ===== 第 2 步：切出待缓存的新满块 =====
    # 例：blocks 共 5 块，此前缓存到第 2 块，现共 4 块满 → 缓存 blocks[2:4]
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    # 掩码长度必须与新满块数一致（由 manager 的 reachable_block_mask 保证）
    assert block_mask is None or len(block_mask) == len(new_full_blocks)

    # ===== 第 3 步：取哈希（不算哈希，只做粒度对齐） =====
    # request.block_hashes 在 Request 创建/追加 token 时就算好了（hash_block_size 粒度）。
    # 本组 block_size 可能是 hash_block_size 的倍数（混合组场景），需要按 block_size
    # 粒度"合并视图"；相等则直接复用（kv_cache_utils.py:2300 resolve_block_hashes）
    block_hashes = resolve_block_hashes(
        request.block_hashes, self.hash_block_size, block_size
    )
    # 从"已缓存位置"切到末尾，下标 i 与 new_full_blocks[i] 对齐
    new_block_hashes = block_hashes[num_cached_blocks:]

    # 事件收集器：仅 enable_kv_cache_events 时启用（默认 None 不收集）
    new_hashes: list[ExternalBlockHash] | None = (
        [] if self.enable_kv_cache_events else None
    )

    # ===== 第 4 步：逐块写入哈希表 =====
    for i, blk in enumerate(new_full_blocks):
        # 跳过 null 块（滑窗/Mamba 对齐的占位）和被掩码排除的块
        # （这些块永远不会被 find_longest_cache_hit 查到，不该进哈希表）
        if blk.is_null or (block_mask is not None and not block_mask[i]):
            continue
        block_hash = new_block_hashes[i]
        # 该哈希覆盖的累积 token 数 = (块位置+1) * block_size
        # 记到 blk.block_hash_num_tokens，供部分尾块哈希（partial tail）判断用
        num_hash_tokens = (num_cached_blocks + i + 1) * block_size

        # 哈希 + group_id 拼成复合 key——不同组即使 token 相同哈希也不同，组间隔离
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        if blk.block_hash is not None:
            # 块已有主哈希：唯一合法场景是"部分尾块 → 满块晋升"
            # （此前以 partial hash 缓存过，如 fine-grained lookup 的尾块）
            # 断言旧哈希覆盖的 token 数确实更少（是部分哈希不是同哈希重写）
            assert (
                blk.block_hash_num_tokens is not None
                and blk.block_hash_num_tokens < num_hash_tokens
            )
            # 清掉旧哈希条目（主哈希+别名都清，见 :571），避免过期别名残留
            removed_hashes = self._remove_cached_block_hashes(blk)
            # 开事件时为每个被移除的哈希发 BlockRemoved 事件
            self._emit_block_removed_events(removed_hashes)
        # 写入新哈希（:607）：
        #   - blk.block_hash 为 None → 设为主哈希（set_block_hash，记录 num_tokens）
        #   - blk.block_hash 已有   → 记入别名集 cached_block_hashes_by_block
        #   - 已存在相同映射则跳过（幂等）
        #   - 最后插入正向表 cached_block_hash_to_block
        self._insert_block_hash(
            block_hash_with_group_id,
            blk,
            num_tokens=num_hash_tokens,
        )
        if new_hashes is not None:
            new_hashes.append(maybe_convert_block_hash(block_hash))

    # ===== 第 5 步：发 BlockStored 事件（仅 enable_kv_cache_events 时） =====
    if self.enable_kv_cache_events:
        # 父哈希 = 第一个新块之前那块的哈希（链式哈希的前驱）；
        # 从头缓存（num_cached_blocks==0）则无父
        if num_cached_blocks == 0:
            parent_block_hash: ExternalBlockHash | None = None
        else:
            parent_block_hash = maybe_convert_block_hash(
                block_hashes[num_cached_blocks - 1]
            )

        # 本次缓存覆盖的 token 区间
        start_token_idx = num_cached_blocks * block_size
        end_token_idx = num_full_blocks * block_size

        # 逐块生成 extra_keys：多模态特征 / cache_salt（仅首块）等附加键。
        # 与上面循环一样跳过 null/掩码块，保证与 new_hashes 长度对齐
        extra_keys_list: list[tuple[Any, ...] | None] = []
        curr_mm_idx = 0        # 多模态 item 游标（跨块推进）
        for i in range(num_cached_blocks, num_full_blocks):
            if blocks[i].is_null:
                continue
            if block_mask is not None and not block_mask[i - num_cached_blocks]:
                continue
            block_start = i * block_size
            block_end = block_start + block_size
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            extra_keys_list.append(extra_keys)

        # 组装 BlockStored 事件入队，后续 drain 给外部消费者
        # （gateway / KV connector 等通过事件感知缓存写入）
        self.kv_event_queue.append(
            self._build_block_stored_event(
                request,
                block_hashes=new_hashes,
                parent_block_hash=parent_block_hash,
                start_token_idx=start_token_idx,
                end_token_idx=end_token_idx,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
                extra_keys_list=extra_keys_list,
            )
        )
```

#### 5.5.2 关键设计点

- **哈希不在这里算**：`request.block_hashes` 由 `Request` 在 token 创建/追加时预计算，本方法只通过 `resolve_block_hashes()` 做粒度对齐后取用（`block_size == hash_block_size` 时零开销复用；倍数时合并视图）
- **幂等**：`num_cached_blocks >= num_full_blocks` 直接返回；`_insert_block_hash` 内部对相同映射也是 no-op。manager 侧用 `num_cached_block[request_id]` 记住进度，多轮调用只处理增量
- **部分尾块 → 满块晋升**：唯一允许"新满块已带哈希"的场景。先用 `_remove_cached_block_hashes` 清掉旧的部分哈希（主哈希+别名一起清），再写入覆盖更多 token 的满块哈希，避免过期别名残留导致错误命中
- **一块可挂多个哈希（别名）**：`_insert_block_hash` 里主哈希之外的哈希都进 `cached_block_hashes_by_block` 反向表。清理时（remove / evict）两个表一起删，防止泄漏
- **group_id 拼进哈希 key**：不同 KV cache 组（如 full attention 组 vs SWA 组）即使 token 相同，物理 KV 布局也不同，必须组间隔离，否则跨组命中会读到错误布局的数据
- **掩码块/null 块不进缓存**：这些块永远不会被本组的 `find_longest_cache_hit` 查到（滑窗外/状态对齐占位），写进哈希表只会污染索引
- **事件是旁路**：`BlockRemoved`/`BlockStored` 只在 `enable_kv_cache_events` 时产生，服务于外部消费者（gateway、connector），不影响缓存本身的正确性

#### 5.5.3 底层辅助：`_insert_block_hash`（主哈希 vs 别名分流）

源码位置：`block_pool.py:607-627`

**作用**：把一条 `(哈希 → 块)` 映射写入正向表，同时决定该哈希成为块的**主哈希**还是**别名**。三个调用方：
- `cache_full_blocks`（:293）——满块入缓存
- `cache_partial_block`（:508）——细粒度部分尾块条目（`block_size > hash_block_size` 场景）
- `move_block_hashes`（:645）——CoW 哈希转移

```python
def _insert_block_hash(
    self,
    block_hash_with_group_id: BlockHashWithGroupId,  # 复合 key：块哈希 + group_id
    block: KVCacheBlock,                             # 目标物理块
    num_tokens: int | None,                          # 该哈希覆盖的累积前缀 token 数（仅主哈希记录）
) -> None:
    # ===== 防重 1：块的主哈希就是它 =====
    # 完全相同的 (块, 哈希) 组合，无事可做（幂等）
    if block.block_hash == block_hash_with_group_id:
        return

    # ===== 防重 2：这条映射已存在 =====
    # 正向表里该哈希已经指向本块（可能以主哈希或别名形式登记过），跳过。
    # 注意 contain 检查的是"该哈希 → 该 block_id"这一条映射，不是哈希存不存在——
    # 同一哈希指向**别的块**不算重复（见下文 insert 的多块 dict）
    if self.cached_block_hash_to_block.contain(
        block_hash_with_group_id, block.block_id
    ):
        return

    # ===== 分流：本块还没有主哈希 → 该哈希升为主哈希 =====
    if block.block_hash is None:
        # set_block_hash（kv_cache_utils.py:148）断言当前无哈希，写入
        # _block_hash 和 _block_hash_num_tokens（累积前缀长度，供晋升判断）
        block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
    # ===== 分流：本块已有别的主哈希 → 该哈希只登记为别名 =====
    else:
        # 反向表 cached_block_hashes_by_block[block_id] 追加这个哈希，
        # 供驱逐/清空/晋升时反查"本块身上挂了哪些哈希"一次性全删
        self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
            block_hash_with_group_id
        )
    # ===== 写正向表 =====
    # cached_block_hash_to_block.insert（:88）支持一哈希多块：
    #   - key 不存在 → 直接挂单块
    #   - key 已挂单块 → 升级成 dict{block_id: block}（同内容多份物理拷贝共存）
    #   - key 已是 dict → 追加
    self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)
```

**执行路径一览**（4 种情况）：

| 进入状态 | 走的分支 | 效果 |
|---|---|---|
| 主哈希 == 新哈希 | 防重 1 返回 | 无变化（幂等） |
| 该映射已在正向表 | 防重 2 返回 | 无变化（幂等） |
| 块无主哈希 | 主哈希分支 | 设主哈希 + 记 `num_tokens` + 写正向表 |
| 块已有别的**不同**主哈希 | 别名分支 | 哈希进反向别名集 + 写正向表 |

**关键认知**：

- **主哈希与别名的区别只在"块身上"**：正向表里两者地位相同（都能被 `get_one_block` 查到）；区别是主哈希存在 `KVCacheBlock._block_hash`（随块走，还带 `num_tokens`），别名只存在 pool 的反向表里（按 block_id 聚合）。
- **什么时候出现别名**：典型是"一块内容同时匹配多个哈希边界"。如 `block_size > hash_block_size` 时，一个物理块内的多个细粒度前缀边界都能命中它——第一个边界成为主哈希，其余是别名。`cache_partial_block` 的 docstring 说的就是这个："If the block already has a primary hash, the partial entry is tracked in `cached_block_hashes_by_block`"。
- **为什么必须先 remove 再晋升**：部分→满块晋升若不走 `_remove_cached_block_hashes`，旧的部分哈希会变成残留别名继续指向该块——但块内容对应的正确 key 已经变了，残留别名会让后续查找命中到错误边界。这就是 §5.5.1 第 4 步先 remove 的原因。
- **一哈希多块（dict）的语义**：`insert` 允许同一哈希指向多个物理块（如 CoW 后新旧两块内容相同）。查找时 `get_one_block` 取任意一个；`pop(key, block_id)` 只摘指定块的那条，不影响其它。

### 5.6 驱逐缓存条目 `evict_blocks`

```python
# block_pool.py:744-761
def evict_blocks(self, block_ids: set[int]) -> None:
```

**作用**：仅按 `block_id` 从哈希表移除缓存条目，**不改变块的占用状态**。`ref_cnt > 0` 的块仍被请求持有，只是失去"可被前缀命中"的身份。常用于显式缓存失效（如 RLHF 权重更新后）。

### 5.7 哈希转移 `move_block_hashes`

```python
# block_pool.py:629-645
def move_block_hashes(
    self,
    src_block: KVCacheBlock,
    dst_block: KVCacheBlock,
) -> None:
```

**作用**：Copy-on-Write 场景使用——当一个被共享的块需要被某请求修改时，先把数据复制到新块（dst），再把 src 的所有哈希条目转移给 dst。这样请求继续在 src 上写私有数据，缓存保留在 dst 上供后续命中。

---

## 六、链式哈希机制

### 6.1 两种哈希类型

```python
# kv_cache_utils.py
BlockHash            = NewType("BlockHash", bytes)             # 组无关，只算一次
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)  # 拼上 group_id
```

- **`BlockHash`**：基于 token 内容+父哈希的链式哈希值，组无关。内容只依赖 token 序列，不依赖层的 KV 形态。
- **`BlockHashWithGroupId`**：`BlockHash`（32字节）+ `group_id`（4字节 big-endian），作为哈希映射表的 key。纯 Full Attention 下 group_id=0。

### 6.2 链式哈希生成

`get_block_hash()` / `hash_block_tokens()`（`kv_cache_utils.py`）实现链式哈希：

```
block_0 哈希 = H(NONE_HASH,    token_0~bs-1,   extra_keys)
block_1 哈希 = H(block_0_hash, token_bs~2bs-1, extra_keys)
block_2 哈希 = H(block_1_hash, token_2bs~3bs-1, extra_keys)
...
```

每个 block 的哈希都包含前面所有 block 的信息（像区块链一样），三大特性：
1. **相同前缀 → 相同哈希链**
2. **改一处 → 全链变化**
3. **天然支持前缀匹配**：从第一个 block 顺着查，第一个 miss 后面必然全 miss，可直接 break

`NONE_HASH` 是链头种子，默认 `os.urandom(32)` 防碰撞。`extra_keys` 可附加额外指纹（如 LoRA id、多模态特征），保证不同上下文下相同 token 序列不误命中。

---

## 七、端到端生命周期示例

以 Llama 模型（`block_size=16`，11个block，id 0~10）为例，演示两个请求的完整生命周期。请求 A 有 32 个 token（2个满块），请求 B 与请求 A 前 16 个 token 相同。

```
【初始状态】BlockPool(num_gpu_blocks=11) 初始化后
  null_block = block_0（is_null=True，从空闲队列摘出）
  FreeQueue: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]（队首→队尾）
  HashMap:  {}

【步骤1：请求A到达，分配2个块】
  get_new_blocks(2) → popleft_n(2) → [block_1, block_2]
  → 每个块 ref_cnt=1（初始无哈希，无清理操作）
  A.req_to_blocks = [block_1, block_2]
  FreeQueue: [3, 4, 5, 6, 7, 8, 9, 10]

【步骤2：请求A完成前向，前32个token写满block_1和block_2】
  cache_blocks → block_pool.cache_full_blocks(...)
  → block_1.set_block_hash((H0,0)), block_2.set_block_hash((H1,0))
  → HashMap: {(H0,0)→block_1, (H1,0)→block_2}
  FreeQueue: [3, 4, 5, 6, 7, 8, 9, 10]（不变，ref_cnt>0）

【步骤3：请求B到达，前缀查找】
  B的block_hashes = [H0, H1', H2']（H1'与H1不同，从第2块开始分叉）
  get_cached_block(H0, [0]) → 命中 [block_1]
  get_cached_block(H1', [0]) → 未命中 → break
  → computed_blocks = ([block_1],)，即前16个token命中

  分配：touch([block_1]) 把block_1从空闲队列摘出（本来ref_cnt=1，不在队列中），ref_cnt→2
  get_new_blocks(2) → popleft_n(2) → [block_3, block_4]
  B.req_to_blocks = [block_1, block_3, block_4]
  → block_1 被A和B共享（零拷贝）！

【步骤4：请求A完成，逆序释放】
  free → free_blocks(reversed([block_1, block_2]))
  → block_2: ref_cnt 1→0, 有hash(H1) → append（队尾）
  → block_1: ref_cnt 2→1（B还在用）→ 不进队列
  FreeQueue: [3, 4, 5, 6, 7, 8, 9, 10, 2]
  HashMap:  {(H0,0)→block_1, (H1,0)→block_2}（不变）

【步骤5：请求B完成，逆序释放】
  free → free_blocks(reversed([block_1, block_3, block_4]))
  → block_4: ref_cnt 1→0, 有hash(H2') → append
  → block_3: ref_cnt 1→0, 有hash(H1') → append
  → block_1: ref_cnt 1→0, 有hash(H0) → append
  FreeQueue: [5, 6, 7, 8, 9, 10, 2, 4, 3, 1]
              ↑无哈希优先驱逐（本例无）      ↑有哈希尽量保留
  HashMap:  {(H0,0)→block_1, (H1,0)→block_2, (H1',0)→block_3, (H2',0)→block_4}
```

**要点串联**：
- 首次分配：`get_new_blocks` 从队首取块，填满后 `cache_full_blocks` 写入哈希
- 前缀命中：`get_cached_block` 查哈希表找到 block_1，`touch` 将 `ref_cnt++` 实现零拷贝共享
- 双队列分流释放：无哈希块插队首，有哈希块插队尾（本例全是有哈希块，故全在队尾）
- 共享保护：block_1 在 A 释放时 ref_cnt 从 2→1，B 仍在用，不回收；B 释放后 ref_cnt→0 才进队列

---

## 八、设计要点小结

1. **逻辑-物理分离**：BlockPool 只持 `block_id` 整数和元数据，不持显存指针；物理张量在 GPU 侧通过 `block_id == 张量行号` 自然桥接，调度全程零显存拷贝
2. **二维耦合**：空间维度（free_block_queue LRU）与内容维度（hash 映射表）通过 `ref_cnt` 联动——`ref_cnt` 归零才进驱逐候选，但驱逐候选仍可挂在 hash 表上等命中
3. **引用计数共享**：多请求命中相同前缀时 `ref_cnt++` 共享物理块，零拷贝；`ref_cnt==0` 才回收到空闲队列
4. **双队列分流策略**：无哈希块放队首（优先驱逐，零成本），有哈希块放队尾（LRU 保护前缀缓存）
5. **链式哈希前缀匹配**：每个 block 哈希包含父哈希，从左到右扫描遇 miss 即停，保证前缀一致性
6. **一块一哈希**：`set_block_hash` 断言保护，生命期内只设一次主哈希，换内容必须先 reset
7. **Copy-on-Write 支持**：`move_block_hashes` 在共享块需要写入时转移哈希身份，保证缓存不被污染
8. **事件旁路不参与决策**：`kv_event_queue` 只广播给 connector，不影响调度逻辑
9. **哈希与分配解耦**：BlockPool 不计算哈希，哈希由 `Request` 预计算，BlockPool 只做插入/查询/删除，职责单一

---

## 扩展：多 group 混合模型场景

纯 Full Attention 单 group 是最简单的场景。当模型包含多种注意力类型（Full + SWA、Full + Mamba 等）时，需要划分多个 KV cache group，此时 BlockPool 的核心机制不变，但有以下扩展：

### E1. 跨组共享编号空间

所有 group **共用同一个 BlockPool**——共享 `[0, num_blocks-1]` 编号空间、同一个空闲链表、同一张哈希表。但共享的只是"编号空间"而非物理存储：同一个 `block_id=5` 在不同 group 中指向各自物理张量的第 5 行，互不干扰。

为什么共享编号？
- 请求只需维护一份 `block_table`，所有 group 都按同一组编号找各自张量的行
- 一次 `get_new_blocks(k)` 拿到 `[n, n+1, ..., n+k-1]`，所有 group 用同一组编号，原子分配无需跨组同步
- 跨组命中对齐简单：命中第 k 个 block 时所有 group 都用 `block_id=k`

### E2. 跨组同时命中语义

`get_cached_block(block_hash, kv_cache_group_ids)` 输入一个组无关的 `BlockHash` + 一组 `group_ids`，对每个 group 用 `(hash, group_id)` 查哈希表；**任一 group miss 就整体返回 `None`**。

原因：一条请求需要所有 group 同时持有该 block 才能复用。如果 Full 组命中但 Mamba 组没命中，Mamba 仍需重新计算状态，整体上等于没命中。

### E3. 三种 block_size 关系

混合模型里不同 group 可能有不同物理 `block_size`。`resolve_kv_cache_block_sizes()`（`kv_cache_utils.py`）通过 LCM/GCD 统一：

| 尺寸 | 含义 | 多 group 计算 | 单 group（本文主线） |
|---|---|---|---|
| `scheduler_block_size` | 调度对齐粒度 | 各 group block size 的 **LCM** | = `block_size` |
| `hash_block_size` | 计算 `Request.block_hashes` 的粒度 | 各 group block size 的 **GCD** | = `block_size` |
| `group.block_size` | 各组实际物理 block 大小 | LCM 的因子 | = `block_size` |

`BlockHashListWithBlockSize`（`kv_cache_utils.py`）利用链式哈希"子哈希覆盖整个前缀"的特性，把 GCD 粒度的细哈希懒加载转换为各组目标粒度的哈希。

### E4. Partial block 缓存

当 `hash_block_size < block_size` 时，一个物理大块内部可能有多个 hash 边界。`cache_partial_block`（`block_pool.py:445`）把大块在内部前缀边界注册为可命中的别名，此时 `_block_hash_num_tokens < block_size`。纯 Full Attention 单 group 场景下三种 block_size 相等，partial block 仅在 CoW 等特殊场景出现。