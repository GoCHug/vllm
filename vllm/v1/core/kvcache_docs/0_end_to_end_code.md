# KVCache 端到端函数详解（纯 Full Attention · Llama-3-8B · pp2tp2·4卡）

> 场景主线：**请求 R 从入队到释放的完整生命周期**（WAITING 入队 → 首次 prefill → 31 步 decode → free），按**调度顺序 + 调用栈深度**依次详解 KVCache 各层函数。
> 与按文件分章节的文档不同，这里以**一次调用的路径**为序，把散落在 4 个文件里的函数**就地**串起来讲。
>
> 环境：Llama-3-8B · Pipeline/Tensor Parallel 均为 2（4 卡）· 单 KV group（纯 Full Attention）· 单 BlockPool（4096 块，`block_size=16`）。
> 请求 **R**：prompt = 70 token（含 32 token 共享前缀 SP），前置请求 P 已把 SP 写入前缀缓存（块 0/1 为带哈希缓存块）。R 续写 31 个 token 后完成释放，最终占用块 0..6 共 7 块。
> **本章编号**：① 前缀查找 · ② 分配与缓存 · ③ decode · ④ 释放。
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

## 0. 本次详解走的完整调用链总览

```
入队 → 请求进入 WAITING 队列
├─ 首次调度做 prefill
│  ├─ 阶段① 前缀查找  get_computed_blocks（调度 ①-step）
│  │  ┌───────────────────────────────────────────────────────────┐
│  │  │ KVCacheManager.get_computed_blocks        kv_cache_manager.py│
│  │  │   └─ UnitaryKVCacheCoordinator.find_longest_cache_hit     │
│  │  │        └─ FullAttentionManager.find_longest_cache_hit     │
│  │  │             └─ BlockPool.get_cached_block → 命中块 0/1     │
│  │  └───────────────────────────────────────────────────────────┘
│  ├─ 阶段② 分配与缓存  allocate_slots（调度 ②-step）
│  │  ┌───────────────────────────────────────────────────────────┐
│  │  │ KVCacheManager.allocate_slots          kv_cache_manager.py │
│  │  │   ├─ ②a Coordinator.get_num_blocks_to_allocate            │
│  │  │   │     └─ FullAttentionManager.get_num_blocks_to_allocate→ 需 3 新块
│  │  │   ├─ ②b Coordinator.allocate_new_computed_blocks           │
│  │  │   │     └─ FullAttentionManager.add_local_computed_blocks  │
│  │  │   │           └─ BlockPool.touch       → 命中块 ref_cnt++   │
│  │  │   ├─ ②c Coordinator.allocate_new_blocks                    │
│  │  │   │     └─ FullAttentionManager.allocate_new_blocks        │
│  │  │   │           └─ BlockPool.get_new_blocks → 弹 2/3/4        │
│  │  │   │       block_table = [命中0, 命中1, 新2, 新3, 新4]        │
│  │  │   └─ ②d Coordinator.cache_blocks                           │
│  │  │         └─ FullAttentionManager.cache_blocks               │
│  │  │               └─ BlockPool.cache_full_blocks → 满块2、3入哈希│
│  │  │                   （块4 未满不入）                           │
│  │  └───────────────────────────────────────────────────────────┘
│  ├─ SchedulerOutput   # 调度输出，附清零块 id 2/3/4
│  └─ GPUModelRunner.execute_model   # forward 写 70 token KV → sample → 第1token
→ 请求进入 RUNNING 队列
├─ 阶段③ decode（续写 31 步，每步都做 ② 的"减配版" + SchedulerOutput + execute_model）
│  块4占6/16 → 步1~10 填满块4(0分配)；步11 申请块5；步12~26 填满块5；步27 申请块6；步28~31 块6占5/16未满
│  情况A·0分配（当前块未满） / 情况B·申请1块（跨块边界）
│  SchedulerOutput 附清零块 id = 新申请的块（步11块5、步27块6）；0分配步无清零
→ 请求生成结束（FINISHED_*）→ 从 RUNNING 移除
└─ 阶段④ 释放  KVCacheManager.free（逆序归还 6→5→4→3→2→1→0）
   └─ UnitaryKVCacheCoordinator.free
      └─ FullAttentionManager.free
         └─ BlockPool.free_blocks   # ref_cnt--，归 0 才回收
             命中块 0/1（共享）→ 仅减 ref_cnt；有哈希 2/3/4/5 → 逆序 append 队尾；无哈希 6 → prepend 队首
```

> **阅读口诀**：全生命周期=「**一次 prefill（①查+②分）** → **③ 的 31 步 decode（②减配重复）** → **④ 释放**」。其中 ①只**查**（读已有缓存，不做分配）；②才**分**（touch 命中 + 申请新块），并在分配后**顺手缓存新增的满块**。两阶段共用一个 BlockPool、一套 `ref_cnt`。SchedulerOutput / `execute_model` 是每个调度步的连接件（无独立编号）。

***

## 生命周期背景 · 入队 → WAITING（起点）

R 被加入 `WAITING` 队列（尚未持有任何 KV 块），由 Scheduler 在调度循环里选中后进入首次调度（prefill）。本阶段无独立编号，是 ①-④ 的触发上下文。

***

## 阶段① 前缀查找：`get_computed_blocks` 调用链（prefill 首步）

### ①-1. `KVCacheManager.get_computed_blocks`

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

### ①-2. `UnitaryKVCacheCoordinator.find_longest_cache_hit`

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

### ①-3. `FullAttentionManager.find_longest_cache_hit`

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

### ①-4. `BlockPool.get_cached_block`

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

**R 的落点**：`hash(t0-15)+group0` 命中块 0，`hash(t16-31)+group0` 命中块 1；`hash(t32-47)` 在哈希表中查不到 → 返回 `None` → 触发 ①-3 的 `break`。

> `cached_block_hash_to_block` 就是五层架构里的**前缀缓存哈希表**。键 = `(token哈希, group_id)`，值 = `KVCacheBlock`；同时块上也存着 `block_hash`（`KVCacheBlock` 的 `_block_hash` 字段）形成双向索引，驱逐/更新时能互删（见 ②d）。

***

## 阶段② 分配与缓存：`allocate_slots` 调用链

### ②-0. `KVCacheManager.allocate_slots` —— 外层框架

源码位置：`kv_cache_manager.py:344-565`（门面层最复杂的方法，约220行）

#### ②-0.1 前置：参数校验 + token 统计 + watermark

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

#### ②-0.2 三个子阶段的骨架（源码 docstring 里的三段式）

```python
# 阶段1: 释放 comp 中不需要的块，检查空闲块是否足够（不足返回 None）
# 阶段2: 处理前缀 token（comp + new_comp + ext_comp）
#        - 释放不需要的块（如滑动窗口外）
#        - 为 ext_comp 在滑动窗口内的 token 分配新块
# 阶段3: 为待计算的 token（new + lookahead）分配新块
```

逐段拆解（每段调用点都在后文 ②b/②c/②d 深挖）:

```python
# ---------- 子阶段①：空间检查（见 ②a） ----------
num_tokens_main_model = total_computed_tokens + num_new_tokens          # 32+38=70
num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens,
                           self.max_model_len)                          # = 70
# 释放SWA窗口外快（纯FullAttention是no-op）
self.coordinator.remove_skipped_blocks(...)
# 计算本轮要分配多少新块 + 空间检查（不足 return None）
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)  # → 2a
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
if required_blocks > available_blocks:
    return None

# ---------- 子阶段②：touch命中块 + 为ext_comp分配（见 ②b） ----------
if (new_computed_block_list is not self.empty_kv_cache_blocks.blocks
        or num_external_computed_tokens > 0):
    self.coordinator.allocate_new_computed_blocks(...)                   # → 2b

# ---------- 子阶段③：为 new + lookahead 分配新块（见 ②c / ②d） ----------
new_blocks = self.coordinator.allocate_new_blocks(...)                   # → 2c
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)                       # P/D留缓存
num_tokens_to_cache = min(total_computed_tokens + num_new_tokens,
                          request.num_tokens)                            # = min(70,70)=70
self.coordinator.cache_blocks(request, num_tokens_to_cache)              # → 2d
return self.create_kv_cache_blocks(new_blocks)
```

> **两阶段分配修复竞态**：必须先 touch 全部命中块（子阶段②，`ref_cnt++`），再申请新块（子阶段③）。否则申请新块触发驱逐时，可能把还没 touch 的命中块驱逐掉（`issue #33775`）。

***

### ②a. `get_num_blocks_to_allocate`：算出本轮需要几个新块

#### ②a-1. `UnitaryKVCacheCoordinator.get_num_blocks_to_allocate`

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

#### ②a-2. `FullAttentionManager.get_num_blocks_to_allocate`

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

### ②b. `allocate_new_computed_blocks`：touch 命中块（`issue #33775`）

#### ②b-1. `UnitaryKVCacheCoordinator.allocate_new_computed_blocks`

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

#### ②b-2. `FullAttentionManager.add_local_computed_blocks`

源码位置：`single_type_kv_cache_manager.py:232-289`（基类）

```python
req_blocks = self.req_to_blocks[request_id]     # 首次：空列表
assert len(req_blocks) == 0

num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens  # 32

# FullAttention：get_num_skipped_tokens=0 → 不裁剪，直接全部touch
if self.enable_caching:
    self.block_pool.touch(new_computed_blocks)  # 命中块 0/1 → touch（见 ②b-3）
else:
    assert not any(new_computed_blocks)

# 把命中的块正式挂到本请求名下
req_blocks.extend(new_computed_blocks)          # req_to_blocks[R] = [块0, 块1]

# 标记这些块"已缓存"，cache_blocks 不用重复缓存
self.num_cached_block[request_id] = len(req_blocks)          # = 2
# 无部分命中 → 不写 _partial_hit_reqs
```

#### ②b-3. `BlockPool.touch`

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

### ②c. `allocate_new_blocks`：为新 token 申请新块 `2/3/4`

#### ②c-1. `UnitaryKVCacheCoordinator.allocate_new_blocks`

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

#### ②c-2. `FullAttentionManager.allocate_new_blocks`

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

#### ②c-3. `BlockPool.get_new_blocks`

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

### ②d. `cache_blocks`：把新增满块写进前缀缓存 `2/3`

#### ②d-1. `UnitaryKVCacheCoordinator.cache_blocks`

源码位置：`kv_cache_coordinator.py:273-290`（基类，逐组转发）

```python
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    for manager in self.single_type_managers:
        manager.cache_blocks(request, num_computed_tokens,
                             retention_interval=self.retention_interval)
```

#### ②d-2. `FullAttentionManager.cache_blocks`（先基类再子类）

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

#### ②d-3. `BlockPool.cache_full_blocks`

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

## 调度收尾（连接 ①② 与 ③）· 产出 `SchedulerOutput` / 把待清零块交给 Worker

每个调度步在「分配完成」之后、把结果交给 worker 之前，都会做这一步（无独立编号）。首次 prefill 与 31 步 decode 共用同一套流程：`take_new_block_ids` 把本步**新申请**的块 id 取走，打进 `SchedulerOutput.zero_out_block_ids` 一并下发。

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

## 阶段③ decode：续写 31 步（每步都走 ①/② 的"减配版"）

R 进入 RUNNING 后不再 prefill，而是单 token 逐次续写。**每个 decode 步依然会调一次** **`get_computed_blocks`** **+** **`allocate_slots`，但几乎都是空转的减配路径**：前缀通常不再命中，`allocate_slots` 只算"还差几块"。

### ③-1. 两种动作（以 `allocate_slots` 的需求为核心）

| 动作               | 触发条件                     | `get_num_blocks_to_allocate`（走 ②a 快路径） | `allocate_new_blocks`（②c）                            | `cache_blocks`（②d）          |
| ---------------- | ------------------------ | -------------------------------------- | ---------------------------------------------------- | --------------------------- |
| **情况A · 0 分配**   | 当前块还有空位，本轮 token 落进已有块   | `cdiv(需槽位数,16) − 已有块数 = 0`             | 需 0 块 → 不调 `BlockPool.get_new_blocks`                | 每步都调；当前块恰好写满时才入哈希表，否则 no-op |
| **情况B · 申请 1 块** | 前一步把当前块写满，本轮 token 跨进下一块 | 跨过块边界 → 需 1 块                          | 调 `BlockPool.get_new_blocks(1)` → block\_table 尾部 +1 | 新块刚建、未满，不入哈希表               |

> decode 的 `get_num_blocks_to_allocate` 走 **②a 快路径**：`request_id in num_cached_block` 已成立，直接返回 `max(cdiv(需槽位数,16) − 已有块数, 0)`，不再做前缀命中/窗口相关计算。注意 decode 没有 `allocate_new_computed_blocks`/touch 这一支（无新命中）。

### ③-2. R 的 31 步数值分布（对照你的 outline）

prefill 结束时块 4 已占 `6/16`（t64-69），还剩 10 个空位：

| 步数         | 落点             | 动作       | 状态变化                                                |
| ---------- | -------------- | -------- | --------------------------------------------------- |
| 步1 \~ 步10  | 写进块4剩余 10 个空位  | 情况A（0分配） | 块4 填满（步10 时满 → ②d cache 补录 `hash(t64-79)`）          |
| **步11**    | 跨入块5           | **情况B**  | 申请块5（`get_new_blocks(1)`），SchedulerOutput 附清零块 id=5 |
| 步12 \~ 步26 | 写进块5（15 token） | 情况A      | 块5 填满（步26 时满 → ②d 补录 `hash(t80-95)`）                |
| **步27**    | 跨入块6           | **情况B**  | 申请块6，SchedulerOutput 附清零块 id=6                      |
| 步28 \~ 步31 | 写进块6（4 token）  | 情况A      | 块6 占 `5/16` 未满 → 不入哈希表，也无清零块                        |

- **31 步里两次申请新块**（步11、步27），其余 29 步为 0 分配；0 分配步的 `SchedulerOutput` **不附带清零块 id**。

- 步11/步27 的"新申请块"即本步要清零的块（`take_new_block_ids` 取到后交给 worker memset）。

- 最终 `req_to_blocks[R] = [块0, 块1, 块2, 块3, 块4, 块5, 块6]`，共 7 块；`ref_cnt`：块0/1=2（共享），块2..6=1。

- 每步末尾的 `GPUModelRunner.execute_model` 与首次调度为**同一组件**：按 `block_table` 读写 KV → forward → sample → 产出 1 个 token。

> **decode 与 prefill 的差别**：②a 在 prefill 会扣掉命中块、可能触发部分命中 CoW；decode 走快路径只算"还差几块"。②d `cache_blocks` 则每步都执行（幂等由 `num_cached_block` 保证），确保某块一旦填满就立刻能复用于后续请求。

***

## 阶段④ 释放：`KVCacheManager.free`（请求结束）

R 续写完成（到达 `max_model_len` 或 EOS，状态 `FINISHED_*`），Scheduler 从 RUNNING 移除 R 并调用 `KVCacheManager.free(request)` 归还全部块。**释放严格逆序（尾块先放）**，四层下钻：

```
KVCacheManager.free(request)         # kv_cache_manager.py:567
└─ UnitaryKVCacheCoordinator.free     # kv_cache_coordinator.py:290（逐组转发）
   └─ FullAttentionManager.free       # single_type_kv_cache_manager.py:519
      └─ BlockPool.free_blocks        # block_pool.py:719「逆序、ref_cnt-1、归0才回收」
```

```python
# SingleTypeManager.free —— pop 本请求全部块，逆序交给 BlockPool
def free(self, request_id: str) -> None:
    # 逆序释放（rev → 6,5,4,3,2,1,0），让尾块先被回收/保留
    self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

# BlockPool.free_blocks —— 逐个 ref_cnt-1，归 0 的块才决定去队首/队尾
def free_blocks(self, ordered_blocks):
    blocks_with_hash = []
    blocks_without_hash = []
    for block in ordered_blocks:            # 逐个逆序到位：6→5→4→3→2→1→0
        block.ref_cnt -= 1                  # 引用计数-1
        if block.ref_cnt == 0 and not block.is_null:   # 归 0 才真正回收
            if block.block_hash is None and self.enable_caching:
                blocks_without_hash.append(block)      # 无哈希 → 立即复用区
            else:
                blocks_with_hash.append(block)         # 有哈希 → 保留区
    self.free_block_queue.prepend_n(blocks_without_hash)  # 无哈希 → prepend 队首
    self.free_block_queue.append_n(blocks_with_hash)      # 有哈希 → append 队尾
```

逆序释放 `6→5→4→3→2→1→0` 的三种命运：

| 释放的块     | 块属性                         | `free_blocks` 的行为                              |
| -------- | --------------------------- | ---------------------------------------------- |
| 块0/1     | **共享命中块**（ref\_cnt=2，P 仍在用） | 仅 `ref_cnt -= 1`（2→1），**不回收**，仍在结构里供 P 继续使用    |
| 块2/3/4/5 | **有哈希**（已入前缀缓存）             | `ref_cnt -= 1`（1→0）后 **append 队尾**（保留缓存，供再次命中） |
| 块6       | **无哈希**（从未填满，没入缓存）          | `ref_cnt -= 1`（1→0）后 **prepend 队首**（立即复用）      |

> **生活化类比**：块6 像一张废弃草稿纸——没记入"读书笔记"（无哈希），直接扔回抽屉最上面随时取用；块2/3/4/5 像记好的笔记页，放回书架留作参考（队尾）。命中块0/1 是 P、R 共用的共享页，R 不再用时只是把"正在读"的人数减一，页面仍留给 P。

**free 之后**：块0/1 `ref_cnt=1`（归 P 独占）；块2/3/4/5 回空闲队列队尾、块6 回队首；前缀缓存哈希表里 2/3/4/5 的键值**保留**（供新请求命中），其中块 2/3（hash t32-63）可反哺给下一个拥有相同前缀的请求。

***

## 端到端数据演变汇总（R 完整生命周期：prefill → decode → free）

| 步骤       | 执行函数                                                | 状态变化                                              |
| -------- | --------------------------------------------------- | ------------------------------------------------- |
| ① 前缀查找   | `get_computed_blocks` → … → `get_cached_block`      | 命中块 0/1，`hit_length=32`                           |
| ②a 需求计算  | `get_num_blocks_to_allocate`                        | `5 − 2 = 3` 新块                                    |
| ②b touch | `allocate_new_computed_blocks` → `touch`            | 块0/1 `ref_cnt 1→2`                                |
| ②c 分新块   | `allocate_new_blocks` → `get_new_blocks`            | pop 2/3/4，`block_table=[0,1,2,3,4]`               |
| 收尾       | `take_new_block_ids`                                | 返回 `[2,3,4]` 交 worker 清零                          |
| ②d 缓存    | `cache_blocks` → `cache_full_blocks`                | 块2(hash t32-47)、块3(hash t48-63) 入哈希表；块4未满不入       |
| ③ decode | 每步 `get_computed_blocks`+`allocate_slots`（减配）       | 步11/27 各 +1 块，附清零块 id；步10/26 块满补录哈希               |
| ④ free   | `KVCacheManager.free` → … → `BlockPool.free_blocks` | 块6 prepend 队首、块2/3/4/5 append 队尾、块0/1 仅减 ref\_cnt |

**prefill 结束时** **`req_to_blocks[R]`**：`[块0, 块1, 块2, 块3, 块4]`
**prefill 结束时** **`ref_cnt`**：块0/1 = 2（P+R 共享），块2/3/4 = 1
**前缀缓存里新增键值**：`(hash(t32-47), g0)→块2`、`(hash(t48-63), g0)→块3`
**decode 结束时**：`req_to_blocks[R] = [块0..块6]`（7块），步10/26 补录块4(hash t64-79)、块5(hash t80-95)
**free 之后**：块0/1 ref\_cnt=1（归 P）、块2/3/4/5 回队尾保留哈希、块6 回队首

***

## 设计要点小结

1. **门面 → 协调器 → 单类型管理器 → 物理块池**的四层下钻：Manager 只拼参数与包装结果，真正逻辑逐层下放，Scheduler 只碰门面。
2. **链式哈希短路**：满块哈希从头逐个查，首个 miss 即停，命中查找是 `O(命中块数)`。
3. **两阶段分配防驱逐竞态**（`issue #33775`）：先 touch 全部命中块（`ref_cnt++`）再申请新块，杜绝驱逐误杀未 touch 的命中块。
4. **复用 vs 新分**：命中块零拷贝复用（touch 即可），新 token 才从空闲池 pop 物理块。
5. **只缓存满块**：额外的不完整尾块不进前缀缓存哈希表，待填满后靠 decode 的 ②d 补录，保证前缀命中只对完整 KV 块生效、可靠。
6. **cache\_blocks 幂等**：`num_cached_block` 记录已缓存数，`num_cached_blocks >= num_full_blocks` 直接返回，prefill 命中块与 decode 满块都不会重复缓存。
7. **hash 基于 token ID**：不依赖 KV 数据，故能在 forward 前算好并写入前缀缓存。
8. **decode 走快路径**：RUNNING 请求无新前缀命中，②a 直接算 `max(cdiv(需槽位数,16)−已有块数,0)`；多数步为 0 分配，仅跨块边界（步11/27）才 pop 1 块，且该块随 SchedulerOutput 交 worker 清零。
9. **释放逆序 + ref\_cnt**：`free` 沿四层下钻到 `BlockPool.free_blocks`，逆序逐块 `ref_cnt-1`；有哈希块 append 队尾（保留缓存）、无哈希块 prepend 队首（立即复用）、共享命中块仅减引用不回收。

