# KVCache 端到端函数详解（纯 Full Attention · Llama-3-8B · pp2tp2·4卡）

> 场景主线：**一次 prefill + 前缀缓存命中**的两条调用链，按**调用栈深度优先**依次详解每一层函数。
> 与按文件分章节的文档不同，这里以**一次调用的路径**为序，把散落在 4 个文件里的函数**就地**串起来讲。
>
> 环境：Llama-3-8B · Pipeline/Tensor Parallel 均为 2（4 卡）· 单 KV group（纯 Full Attention）· 单 BlockPool（4096 块，`block_size=16`）。
> 请求 **R**：prompt = 70 token（含 32 token 共享前缀 SP），前置请求 P 已把 SP 写入前缀缓存（块 0/1 为带哈希缓存块）。
>
> 涉及源文件（调用栈自顶向下）：
>
> - `vllm/vllm/v1/core/kv_cache_manager.py`　— `KVCacheManager`
>
> - `vllm/vllm/v1/core/kv_cache_coordinator.py`　— `UnitaryKVCacheCoordinator`（基类 `KVCacheCoordinator`）
>
> - `vllm/vllm/v1/core/single_type_kv_cache_manager.py`　— `FullAttentionManager`（基类 `SingleTypeKVCacheManager`）
>
> - `vllm/vllm/v1/core/block_pool.py`　— `BlockPool`

***

## 0. 本次详解走的调用栈总览

```
阶段③ 前缀查找  get_computed_blocks（调度 ①②-step）
┌─────────────────────────────────────────────────────────────┐
│ KVCacheManager.get_computed_blocks        kv_cache_manager.py   │
│   └─ UnitaryKVCacheCoordinator.find_longest_cache_hit        │
│        └─ FullAttentionManager.find_longest_cache_hit        │
│             └─ BlockPool.get_cached_block  → 命中块 0/1      │
└─────────────────────────────────────────────────────────────┘

阶段④ 分配与缓存  allocate_slots（调度 ④-step）
┌─────────────────────────────────────────────────────────────┐
│ KVCacheManager.allocate_slots          kv_cache_manager.py   │
│   ├─ ④a UnitaryKVCacheCoordinator.get_num_blocks_to_allocate │
│   │      └─ FullAttentionManager.get_num_blocks_to_allocate  → 需 3 新块
│   ├─ ④b UnitaryKVCacheCoordinator.allocate_new_computed_blocks
│   │      └─ FullAttentionManager.add_local_computed_blocks
│   │            └─ BlockPool.touch        → 命中块 ref_cnt++   │
│   ├─ ④c UnitaryKVCacheCoordinator.allocate_new_blocks        │
│   │      └─ FullAttentionManager.allocate_new_blocks         │
│   │            └─ BlockPool.get_new_blocks → 弹 2/3/4         │
│   │        block_table = [命中0, 命中1, 新2, 新3, 新4]         │
│   └─ ④d UnitaryKVCacheCoordinator.cache_blocks               │
│          └─ FullAttentionManager.cache_blocks                │
│                └─ BlockPool.cache_full_blocks → 满块2、3入哈希│
│                    （块4 未满不入）                            │
└─────────────────────────────────────────────────────────────┘
```

> **阅读口诀**：第③阶段只**查**（读已有缓存），不做分配；第④阶段才**分**（touch 命中 + 申请新块），并在分配后**顺手缓存新增的满块**。两阶段共用一个 BlockPool、一套 `ref_cnt`。

***

## 阶段③ 前缀查找：`get_computed_blocks` 调用链

### ③-1. `KVCacheManager.get_computed_blocks`

源码位置：`kv_cache_manager.py:229-295`（门面层，Scheduler 调用的第一个方法）

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
    # 返回值：(命中块KVCacheBlocks, 命中token数, shared_prefix_boundary)
    #           ③管理器只负责拼装参数 + 包装结果，真正的"找"在下面三层

    # ① 禁用前缀缓存 or 请求标了跳过KV读  → 直接返回空
    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0

    # ② 关键：max_cache_hit_length = num_tokens - 1
    #     即使全部token都命中缓存，最后一个token也必须重算logits
    #     且allocate_slots要求命中数 block 对齐，减1可能触发整块重算
    max_cache_hit_length = request.num_tokens - 1          # 70-1 = 69

    # ③ 委托 Coordinator 查找（这就是本次要深挖的调用链）
    computed_blocks, num_new_computed_tokens, num_uncached = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )

    # ④ （可选）full report 模式下为命中的缓存块发 BlockStored 事件
    #    给外部消费者（如 gateway）同步前缀命中信息
    if (num_new_computed_tokens > 0 and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"):
        for group_idx, group_blocks in enumerate(computed_blocks):
            ...
            self.block_pool.emit_cached_block_events(...)

    # ⑤ shared_prefix_boundary：Hybrid多组才非0，单组恒为0
    shared_prefix_boundary = num_new_computed_tokens + num_uncached if num_uncached else 0

    # ⑥ 复用空对象避免GC，包装成KVCacheBlocks返回
    blocks = self.create_kv_cache_blocks(computed_blocks)
    return blocks, num_new_computed_tokens, shared_prefix_boundary
```

**R 的落点**：`block_hashes` 是 4 个**满块**哈希（`70 // 16 = 4`，尾块 t64-69 未满无哈希）。本层把 `max_cache_hit_length=69` 传下去，最终拿到 `computed_blocks=([hit0块, hit1块],)`、`num_new_computed_tokens=32`。

***

### ③-2. `UnitaryKVCacheCoordinator.find_longest_cache_hit`

源码位置：`kv_cache_coordinator.py:486-503`（第4层协调器，单组场景的**转发壳**）

```python
def find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],     # 请求的链式满块哈希
    max_cache_hit_length: int,         # = num_tokens-1
) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
    # 单组：直接把请求转发给它持有的唯一 SingleTypeManager（FullAttentionManager）
    hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
        block_hashes=block_hashes,
        max_length=max_cache_hit_length,
        kv_cache_group_ids=[0],        # 只有 group 0
        block_pool=self.block_pool,    # 把 BlockPool 引用传下去
        kv_cache_spec=self.kv_cache_spec,
        drop_eagle_block=0 in self.eagle_group_ids,  # 纯FullAttention无eagle → False
        alignment_tokens=self.block_size,  # =16（hash块与宿块同尺寸，非fine-grained）
        dcp_world_size=self.dcp_world_size,
        pcp_world_size=self.pcp_world_size,
    )
    # 单组没有"另一组滞后"的概念，num_uncached 恒为 0
    return hit_blocks, hit_length, 0
```

> **为什么叫做"协调器"**：对多 group（Hybrid）它要协调各组的命中进度、取公共交集；对单 group 它退化为一个「参数补齐 + 转发」的壳。本环境走的就是这条最简路径。

***

### ③-3. `FullAttentionManager.find_longest_cache_hit`

源码位置：`single_type_kv_cache_manager.py:682-777`（真正做**逐块查表**的一层）

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                           block_pool, kv_cache_spec, drop_eagle_block,
                           alignment_tokens, dcp_world_size=1, pcp_world_size=1):
    # 类型守卫：本类只能管 FullAttention / ChunkedLocalAttention
    assert isinstance(kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec)

    block_size = kv_cache_spec.block_size                 # = 16
    if dcp_world_size > 1:
        block_size *= dcp_world_size                      # dcp=1 不变

    # 把"请求的hash粒度"解析成"本manager的块粒度"
    # 若 hash_block_size 与 block_size 不同，相邻hash合并成整块hash
    block_hashes = resolve_block_hashes(
        block_hashes, block_pool.hash_block_size, block_size,
        supports_fine_grained_hash_lookup=cls.supports_fine_grained_hash_lookup,
        alignment_tokens=alignment_tokens,
    )

    # fine-grained：alignment_tokens(=16) < block_size(=16) 且整除 → False
    # 本环境 alignment_tokens == block_size，走非细粒度分支（整块查）
    fine_grained = alignment_tokens < block_size and block_size % alignment_tokens == 0
    if fine_grained:
        ...
    else:
        full_block_hashes = block_hashes   # 直接用整块哈希列表

    computed_blocks = tuple([] for _ in range(len(kv_cache_group_ids)))  # [([],)]

    # Phase 1（核心）：从前往后，逐个"满块哈希"查 BlockPool 前缀缓存
    #   islice(full_block_hashes, max_length // block_size)
    #   = islice(4个hash, 69//16=4)  → 最多查 4 块
    #   关键：链式哈希 —— 第1块没命中，后面的必然也全miss，直接break
    for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
        cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
        if not cached_block:          # 查不到 → 后面不可能命中
            break
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)   # 命中 → 记入 computed_blocks[0]

    hit_length = len(computed_blocks[0]) * block_size     # 命中块数 × 16

    # Phase 2（仅fine-grained，此处跳过）...

    # drop_eagle_block=False 不走；hit_length 已按 alignment 对齐
    hit_length -= hit_length % alignment_tokens
    num_blocks = cdiv(hit_length, block_size)
    for computed in computed_blocks:
        del computed[num_blocks:]     # 裁掉超出命中长度的块
    return computed_blocks, hit_length   # = ([块0, 块1],), 32
```

**R 的命中过程**：

- 4 个满块哈希依次查：`hash(t0-15)`→**命中块0**；`hash(t16-31)`→**命中块1**；`hash(t32-47)`→**miss**，`break`。

- `hit_length = 2 * 16 = 32`，返回 `([块0, 块1],), 32`。

> **生活化类比**：这像对着一本"读书笔记目录"逐页对答案——第一页对上了、第二页对上了，到第三页第一次没对上，说明后面的页也不再是同一个"章节"了，直接停。这就是链式哈希带来的**提前短路**。

***

### ③-4. `BlockPool.get_cached_block`

源码位置：`block_pool.py:198-223`（第1层物理块池，真正查哈希表）

```python
def get_cached_block(self, block_hash, kv_cache_group_ids) -> list[KVCacheBlock] | None:
    cached_blocks = []
    for group_id in kv_cache_group_ids:                 # 单组 → 只查 group 0
        # 哈希要"组感知"：同一token哈希在不同group里是不同的块
        block_hash_with_group_id = make_block_hash_with_group_id(block_hash, group_id)
        # 查前缀缓存哈希表（BlockHashToBlockMap 内部是字典+双向链表支持LRU驱逐）
        block = self.cached_block_hash_to_block.get_one_block(block_hash_with_group_id)
        if not block:
            return None                 # 任一group miss 就整体 miss
        cached_blocks.append(block)
    return cached_blocks                # 返回该哈希在每组的缓存块
```

**R 的落点**：`hash(t0-15)+group0` 命中块 0，`hash(t16-31)+group0` 命中块 1；`hash(t32-47)` 在哈希表中查不到 → 返回 `None` → 触发 ③-3 的 `break`。

> `cached_block_hash_to_block` 就是五层架构里的**前缀缓存哈希表**。键 = `(token哈希, group_id)`，值 = `KVCacheBlock`；同时块上也存着 `block_hash`（`KVCacheBlock` 的 `_block_hash` 字段）形成双向索引，驱逐/更新时能互删（见 ④d）。

***

## 阶段④ 分配与缓存：`allocate_slots` 调用链

### ④-0. `KVCacheManager.allocate_slots` —— 外层框架

源码位置：`kv_cache_manager.py:344-565`（门面层最复杂的方法，约220行）

#### ④-0.1 前置：参数校验 + token 统计 + watermark

```python
# 全空校验：既无新token也无外部computed token，无法分配
if num_new_tokens == 0 and num_external_computed_tokens == 0:
    raise ValueError("num_new_tokens must be greater than 0 ...")

# new_computed_blocks 为 None 就换成空占位，避免到处判空
if new_computed_blocks is not None:
    new_computed_block_list = new_computed_blocks.blocks   # 有命中：命中块
else:
    new_computed_block_list = self.empty_kv_cache_blocks.blocks

# 本地已缓存 token = comp(历史) + new_comp(③命中)
num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
# 全部已缓存 = 本地 + ext_comp，受 max_model_len 封顶
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens, self.max_model_len,
)

# watermark 只对 WAITING/PREEMPTED 且已有请求在调度时生效留 headroom
watermark_blocks = 0
if has_scheduled_reqs and request.status in (RequestStatus.WAITING,
                                             RequestStatus.PREEMPTED):
    watermark_blocks = self.watermark_blocks
```

**R 的数值**：`num_local_computed_tokens = 0 + 32 = 32`；`total_computed_tokens = 32`（无 ext）。

#### ④-0.2 三个子阶段的骨架（源码 docstring 里的三段式）

```python
# 阶段1: 释放 comp 中不需要的块，检查空闲块是否足够（不足返回 None）
# 阶段2: 处理前缀 token（comp + new_comp + ext_comp）
#        - 释放不需要的块（如滑动窗口外）
#        - 为 ext_comp 在滑动窗口内的 token 分配新块
# 阶段3: 为待计算的 token（new + lookahead）分配新块
```

逐段拆解（每段调用点都在后文 ④b/④c/④d 深挖）:

```python
# ---------- 子阶段①：空间检查（见 ④a） ----------
num_tokens_main_model = total_computed_tokens + num_new_tokens          # 32+38=70
num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens,
                           self.max_model_len)                          # = 70
# 释放SWA窗口外快（纯FullAttention是no-op）
self.coordinator.remove_skipped_blocks(...)
# 计算本轮要分配多少新块 + 空间检查（不足 return None）
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)  # → 4a
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
if required_blocks > available_blocks:
    return None

# ---------- 子阶段②：touch命中块 + 为ext_comp分配（见 ④b） ----------
if (new_computed_block_list is not self.empty_kv_cache_blocks.blocks
        or num_external_computed_tokens > 0):
    self.coordinator.allocate_new_computed_blocks(...)                   # → 4b

# ---------- 子阶段③：为 new + lookahead 分配新块（见 ④c / ④d） ----------
new_blocks = self.coordinator.allocate_new_blocks(...)                   # → 4c
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)                       # P/D留缓存
num_tokens_to_cache = min(total_computed_tokens + num_new_tokens,
                          request.num_tokens)                            # = min(70,70)=70
self.coordinator.cache_blocks(request, num_tokens_to_cache)              # → 4d
return self.create_kv_cache_blocks(new_blocks)
```

> **两阶段分配修复竞态**：必须先 touch 全部命中块（子阶段②，`ref_cnt++`），再申请新块（子阶段③）。否则申请新块触发驱逐时，可能把还没 touch 的命中块驱逐掉（`issue #33775`）。

***

### ④a. `get_num_blocks_to_allocate`：算出本轮需要几个新块

#### ④a-1. `UnitaryKVCacheCoordinator.get_num_blocks_to_allocate`

源码位置：`kv_cache_coordinator.py:130-190`（基类 `KVCacheCoordinator`，分组累加）

```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                               num_encoder_tokens, total_computed_tokens,
                               num_local_computed_tokens, num_tokens_main_model,
                               apply_admission_cap=False) -> int:
    num_blocks_to_allocate = 0
    for i, manager in enumerate(self.single_type_managers):      # 只循环1次
        if isinstance(manager, CrossAttentionManager):           # 本环境不走
            ...
        else:
            # 自注意力group：把"需求token数"转成"块数"交给manager算
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id, num_tokens, new_computed_blocks[i],
                total_computed_tokens, num_local_computed_tokens,
                num_tokens_main_model, apply_admission_cap=apply_admission_cap,
            )
    return num_blocks_to_allocate
```

#### ④a-2. `FullAttentionManager.get_num_blocks_to_allocate`

源码位置：`single_type_kv_cache_manager.py:144-230`（`SingleTypeKVCacheManager` 基类实现，FullAttention 不重写）

```python
num_required_blocks = cdiv(num_tokens, self.block_size)        # cdiv(70,16)=5 块

num_req_blocks = len(self.req_to_blocks.get(request_id, ()))   # 本R还没有块 → 0

if request_id in self.num_cached_block:
    # 快路径：RUNNING请求不再有新的前缀命中，直接算差额
    return max(num_required_blocks - num_req_blocks, 0)

# —— 首次prefill才走到这 ——
num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)  # FullAttention=0
num_skipped_blocks = num_skipped_tokens // self.block_size     # 0
num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks  # 2+0=2

# 需要的新块 = 总需求块 - max(被窗口跳过的块, 已缓存的命中块)
num_new_blocks = max(
    num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0,
)                                                              # 5 - max(0,2) = 3

num_evictable_blocks = self._get_num_evictable_blocks(
    new_computed_blocks[num_skipped_new_computed_blocks:])     # 命中块已在队内不可再算 → 0
if self._has_partial_local_hit(...):                           # 无部分命中 → False
    num_new_blocks += 1
return num_new_blocks + num_evictable_blocks                   # = 3
```

**结论**：需要 **3 个新块**（块 2、3、4）。加上命中的 2 块，R 最终占 5 块。

> **为什么是"总块 - 命中块"**：`5` 是装着全部 70 token 所需的总块数，其中 `2` 块已被前缀缓存命中（复用不新分），所以真正要从空闲池 pop 的只有 `5 - 2 = 3` 块。

***

### ④b. `allocate_new_computed_blocks`：touch 命中块（`issue #33775`）

#### ④b-1. `UnitaryKVCacheCoordinator.allocate_new_computed_blocks`

源码位置：`kv_cache_coordinator.py:192-236`（基类）

```python
def allocate_new_computed_blocks(self, request_id, new_computed_blocks,
                                 num_local_computed_tokens,
                                 num_external_computed_tokens) -> None:
    # RUNNING请求已无新命中，直接跳过
    if any(request_id in manager.num_cached_block
           for manager in self.single_type_managers):
        return

    # 两阶段分配（issue #33775）：
    #   Phase A: 先遍历所有group，add_local_computed_blocks → touch（防驱逐）
    #   Phase B: 全部touch完后，再为ext_comp分配外部块
    for i, manager in enumerate(self.single_type_managers):
        manager.add_local_computed_blocks(
            request_id, new_computed_blocks[i],
            num_local_computed_tokens, num_external_computed_tokens,
        )
    if num_external_computed_tokens > 0:              # 本环境 =0 跳过
        for manager in self.single_type_managers:
            manager.allocate_external_computed_blocks(...)
```

#### ④b-2. `FullAttentionManager.add_local_computed_blocks`

源码位置：`single_type_kv_cache_manager.py:232-289`（基类）

```python
req_blocks = self.req_to_blocks[request_id]     # 首次：空列表
assert len(req_blocks) == 0

num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens  # 32

# FullAttention：get_num_skipped_tokens=0 → 不裁剪，直接全部touch
if self.enable_caching:
    self.block_pool.touch(new_computed_blocks)  # 命中块 0/1 → touch（见 ④b-3）
else:
    assert not any(new_computed_blocks)

# 把命中的块正式挂到本请求名下
req_blocks.extend(new_computed_blocks)          # req_to_blocks[R] = [块0, 块1]

# 标记这些块"已缓存"，cache_blocks 不用重复缓存
self.num_cached_block[request_id] = len(req_blocks)          # = 2
# 无部分命中 → 不写 _partial_hit_reqs
```

#### ④b-3. `BlockPool.touch`

源码位置：`block_pool.py:702-717`

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        # ref_cnt==0 说明它此刻在空闲队列里（驱逐候选）→ 先从free队列拿走
        # 命中块P还没释放时ref_cnt=1，通常在调用前已在结构里
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1           # 引用计数+1，表示多了一个请求在用
        if self.metrics_collector:
            self.metrics_collector.on_block_accessed(block)
```

**R 的落点**：块 0、块 1 的 `ref_cnt` 各 +1（此前 P 在用时已是 1，touch 后变 2，表示 P、R 共享）。这两个命中块**零拷贝复用**，不重新分配物理块。

> **为什么必须 touch 后再分新块**：只有 `ref_cnt > 0` 的块才能免疫驱逐。若先申请新块（可能触发 LRU 驱逐）再 touch，命中块可能在驱逐名单上被误杀。

***

### ④c. `allocate_new_blocks`：为新 token 申请新块 `2/3/4`

#### ④c-1. `UnitaryKVCacheCoordinator.allocate_new_blocks`

源码位置：`kv_cache_coordinator.py:238-271`（基类，逐组转发）

```python
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model,
                        num_encoder_tokens=0):
    return tuple(
        manager.allocate_new_blocks(
            request_id,
            num_encoder_tokens if isinstance(manager, CrossAttentionManager)
            else num_tokens,             # 自注意力 → num_tokens=70
            num_tokens_main_model,
        )
        for manager in self.single_type_managers      # 只循环1次
    )
```

#### ④c-2. `FullAttentionManager.allocate_new_blocks`

源码位置：`single_type_kv_cache_manager.py:330-369`（基类）

```python
cow_blocks: list[KVCacheBlock] = []
if request_id in self._partial_hit_reqs:
    # 部分命中才走CoW：把共享尾部重定向到私有副本块（本环境无部分命中）
    ...

req_blocks = self.req_to_blocks[request_id]            # [块0, 块1]
num_required_blocks = cdiv(num_tokens, self.block_size) # cdiv(70,16)=5
num_new_blocks = num_required_blocks - len(req_blocks)  # 5 - 2 = 3
if num_new_blocks <= 0:
    return cow_blocks
else:
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)  # pop 2/3/4
    req_blocks.extend(new_blocks)        # block_table=[0,1,2,3,4]
    if self._record_new_block_ids:
        self.new_block_ids.extend(b.block_id for b in new_blocks)  # 记下待清零
    return cow_blocks + new_blocks
```

#### ④c-3. `BlockPool.get_new_blocks`

源码位置：`block_pool.py:647-677`

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(...)              # 防超额申请

    ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)  # 队首 pop 2/3/4

    if self.enable_caching:
        for block in ret:
            self._maybe_evict_cached_block(block)   # 复用块前若还挂着旧hash先驱逐
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            ...
    else:
        ...   # 不缓存场景：只 ref_cnt++

    return ret     # 返回 2/3/4 三块
```

**R 的落点**：从空闲队列队首弹出块 2、3、4，`ref_cnt` 各记为 1，挂到 `req_to_blocks[R]`，使 `block_table=[0,1,2,3,4]`。同时 `new_block_ids` 攒下 `[2,3,4]`，供调度收尾时 `take_new_block_ids()` 交给 4 个 worker 在 forward 前清零。

> **生活化类比**：命中块 = "练习题册里的旧笔记页"（直接复用前面那本书的页）；新块 = "从练习册拆下 3 张空白页"填到当前这本里。`new_block_ids` 像一张"待批注页"清单，交给工人(worker)先把页清空再写字。

***

### ④d. `cache_blocks`：把新增满块写进前缀缓存 `2/3`

#### ④d-1. `UnitaryKVCacheCoordinator.cache_blocks`

源码位置：`kv_cache_coordinator.py:273-290`（基类，逐组转发）

```python
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    for manager in self.single_type_managers:
        manager.cache_blocks(request, num_computed_tokens,
                             retention_interval=self.retention_interval)
```

#### ④d-2. `FullAttentionManager.cache_blocks`（先基类再子类）

基类 `SingleTypeKVCacheManager.cache_blocks`：`single_type_kv_cache_manager.py:427-482`
子类 `FullAttentionManager.cache_blocks`：`single_type_kv_cache_manager.py:779-789`

```python
# —— 基类（SingleTypeKVCacheManager）——
def cache_blocks(self, request, num_tokens, retention_interval=None):
    num_cached_blocks = self.num_cached_block.get(request.request_id, 0)   # = 2
    num_full_blocks = num_tokens // self.block_size                        # 70//16=4

    if num_cached_blocks >= num_full_blocks:   # 已缓存够就不重复缓存（幂等）
        return

    # （稀疏保留掩码，仅SWA用，FullAttention走默认全1）
    block_mask = self.reachable_block_mask(...)

    self.block_pool.cache_full_blocks(
        request=request,
        blocks=self.req_to_blocks[request.request_id],   # [0,1,2,3,4]
        num_cached_blocks=num_cached_blocks,             # 2（块0/1已缓存）
        num_full_blocks=num_full_blocks,                 # 4（要缓存到第4块）
        block_size=self.block_size,                      # 16
        kv_cache_group_id=self.kv_cache_group_id,
        block_mask=block_mask,
    )
    self.num_cached_block[request.request_id] = num_full_blocks   # 更新为 4

# —— 子类（FullAttentionManager）——
def cache_blocks(self, request, num_tokens, retention_interval=None):
    super().cache_blocks(request, num_tokens, retention_interval=retention_interval)
    # block_size == hash_block_size 时没有"部分尾块"需要额外缓存，直接返回
    if self.block_size == hash_block_size:
        return
    self._cache_partial_tail_block(request, num_tokens)
```

**关键解读**：`num_full_blocks=4` 表示要缓存到第 4 块满块；但前两块（块0/1）已在前缀缓存中（`num_cached_blocks=2`），所以实际**新缓存的区间是** **`blocks[2:4]`，即块 2、块 3**。块 4 只有 6 个 token 未满，**不入哈希表**。

#### ④d-3. `BlockPool.cache_full_blocks`

源码位置：`block_pool.py:225-342`（真正写哈希表）

```python
def cache_full_blocks(self, request, blocks, num_cached_blocks,
                      num_full_blocks, block_size, kv_cache_group_id,
                      block_mask=None):
    if num_cached_blocks >= num_full_blocks:
        return

    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]   # = [块2, 块3]

    # 把请求的全部块哈希里，取本轮要缓存的部分
    block_hashes = resolve_block_hashes(request.block_hashes, self.hash_block_size, block_size)
    new_block_hashes = block_hashes[num_cached_blocks:]           # 从头取

    for i, blk in enumerate(new_full_blocks):                     # 依次处理块2、块3
        if blk.is_null or (block_mask is not None and not block_mask[i]):
            continue
        block_hash = new_block_hashes[i]                          # hash(t32-47)/hash(t48-63)
        num_hash_tokens = (num_cached_blocks + i + 1) * block_size

        block_hash_with_group_id = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
        if blk.block_hash is not None:
            # 该块已有hash（partial→full升级）才需要先清旧键（本环境块2/3是全新块，不走）
            ...
        self._insert_block_hash(block_hash_with_group_id, blk, num_tokens=num_hash_tokens)
    ...
    self.num_cached_block[...] = ...   # （此处在manager层已更新）
```

`_insert_block_hash`（`block_pool.py:607-627`）：把 `(块哈希, group)` → `块` 写进 `cached_block_hash_to_block`，同/反向索引一起维护（块上 `set_block_hash`）。

**R 的落点**：块 2 记 `hash(t32-47)`、块 3 记 `hash(t48-63)`，双双入前缀缓存哈希表。此后任何请求碰到 `t32-63` 的哈希都能命中复用。块 4（只有 t64-69，6 token）**未满**，等 decode 填满后再由后续步的 `cache_blocks` 补录。

***

## 阶段④ 收尾：把待清零块交给 Worker

`KVCacheManager.take_new_block_ids`（`kv_cache_manager.py:796-801`）→ `FullAttentionManager.take_new_block_ids`（`single_type_kv_cache_manager.py:376-380`）：

```python
# KVCacheManager（门面，汇集所有子manager）
def take_new_block_ids(self) -> list[int]:
    ids = []
    for mgr in self.coordinator.single_type_managers:
        ids.extend(mgr.take_new_block_ids())   # 取出并清空
    return ids          # R场景返回 [2, 3, 4]

# SingleTypeManager（真正持 new_block_ids）
def take_new_block_ids(self):
    ids = self.new_block_ids
    self.new_block_ids = []      # drain：取走即清空
    return ids
```

4 个 worker 拿到 `[2,3,4]` 后在 GPU 上对这三块的 KV 内存执行 memset 清零，随后 forward 写入新 KV。

> **Drain 模式**：调度过程中先"记账"（`new_block_ids` 累加），调度结束后**一次性取走并清空**，取完内部归零，避免反复取、重复清零。

***

## 端到端数据演变汇总（R 一次 prefill）

| 步骤       | 执行函数                                           | 状态变化                                        |
| -------- | ---------------------------------------------- | ------------------------------------------- |
| ③ 前缀查找   | `get_computed_blocks` → … → `get_cached_block` | 命中块 0/1，`hit_length=32`                     |
| ④a 需求计算  | `get_num_blocks_to_allocate`                   | `5 − 2 = 3` 新块                              |
| ④b touch | `allocate_new_computed_blocks` → `touch`       | 块0/1 `ref_cnt 1→2`                          |
| ④c 分新块   | `allocate_new_blocks` → `get_new_blocks`       | pop 2/3/4，`block_table=[0,1,2,3,4]`         |
| 收尾       | `take_new_block_ids`                           | 返回 `[2,3,4]` 交 worker 清零                    |
| ④d 缓存    | `cache_blocks` → `cache_full_blocks`           | 块2(hash t32-47)、块3(hash t48-63) 入哈希表；块4未满不入 |

**当前** **`req_to_blocks[R]`**：`[块0, 块1, 块2, 块3, 块4]`
**当前** **`ref_cnt`**：块0/1 = 2（P+R 共享），块2/3/4 = 1
**前缀缓存里新增键值**：`(hash(t32-47), g0)→块2`、`(hash(t48-63), g0)→块3`

***

## 设计要点小结

1. **门面 → 协调器 → 单类型管理器 → 物理块池**的四层下钻：Manager 只拼参数与包装结果，真正逻辑逐层下放，Scheduler 只碰门面。
2. **链式哈希短路**：满块哈希从头逐个查，首个 miss 即停，命中查找是 `O(命中块数)`。
3. **两阶段分配防驱逐竞态**（`issue #33775`）：先 touch 全部命中块（`ref_cnt++`）再申请新块，杜绝驱逐误杀未 touch 的命中块。
4. **复用 vs 新分**：命中块零拷贝复用（touch 即可），新 token 才从空闲池 pop 物理块。
5. **只缓存满块**：额外的不完整尾块不进前缀缓存哈希表，待填满后再补录，保证前缀命中只对完整 KV 块生效、可靠。
6. **cache\_blocks 幂等**：`num_cached_block` 记录已缓存数，`num_cached_blocks >= num_full_blocks` 直接返回，前两步命中块不会重复缓存。
7. **hash 基于 token ID**：不依赖 KV 数据，故能在 forward 前算好并写入前缀缓存。

