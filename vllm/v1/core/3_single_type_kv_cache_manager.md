# SingleTypeKVCacheManager 设计文档

> 五层架构第 3 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`2_block_pool.md`](./2_block_pool.md) ｜上层 ➔ [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)
>
> 源文件：`vllm/vllm/v1/core/single_type_kv_cache_manager.py`

## 1. 一句话定位

`SingleTypeKVCacheManager` 是 vLLM V1 KV cache 分层管理架构中的 **单类型管理器**（抽象基类）：每一种注意力类型（Full / SWA / R-SWA / Chunked-Local / Mamba / Cross / Sink）派生一个子类，管理本组内 **request ↔ block 绑定、命中查询、partial/CoW、释放** 等单组逻辑，**所有子类共用同一个共享 `BlockPool`**。

它在五层架构里处在第 3 层：

```
KVCacheManager          ← 面向 Scheduler 的门面
   │
KVCacheCoordinator      ← 跨组协调（迭代不动点求交集、两阶段分配编排）
   │  └─ single_type_managers[0..N]   ← 每个 KV cache group 一个
   │       ├─ FullAttentionManager        ┐
   │       ├─ SlidingWindowManager        │
   │       ├─ RSWAManager                 │  都引用同一个 block_pool
   │       ├─ ChunkedLocalAttentionManager│  只管「本组」的 req↔block 绑定
   │       ├─ MambaManager                │
   │       ├─ CrossAttentionManager       │
   │       └─ SinkFullAttentionManager    ┘
   │
BlockPool               ← 全模型共享的逻辑块池
```

- **Coordinator 管「跨组」**：统一调度粒度、跨组预算、求各类型命中交集。
- **SingleTypeKVCacheManager 管「单组」**：本组 request↔block 绑定、命中查询、partial/CoW、释放。
- **BlockPool 是共享底层**：所有 manager 都通过 `self.block_pool.*` 调它，**不各自建池**。

---

## 2. ABC 核心字段表

抽象基类定义在 `single_type_kv_cache_manager.py:36`，核心字段在 `__init__`（`single_type_kv_cache_manager.py:44-127`）中初始化。

| 字段 | 类型 | 说明 |
|---|---|---|
| `req_to_blocks` | `defaultdict[str, list[KVCacheBlock]]` | **每请求 block 表**，按 token 顺序记录本组分配/命中的 block，释放时按此枚举 |
| `num_cached_block` | `dict[str, int]` | 每请求已缓存 block 数（仅追踪 RUNNING，被抢占的不计）；用于 `cache_blocks` 跳过已缓存段 |
| `block_pool` | `BlockPool` | 底层块池引用，所有子类共用同一个 |
| `block_size` | `int` | 本管理器的 block 大小；DCP > 1 时会乘以 `dcp_world_size` 做分片 |
| `scheduler_block_size` | `int` | 调度粒度（所有 group block size 的 LCM），是本组 `block_size` 的整数倍 |
| `kv_cache_group_id` | `int` | 本组在 coordinator 中的 ID，写缓存时隔离不同组的同内容块 |
| `kv_cache_spec` | `KVCacheSpec` | 本组规格，携带类型相关参数（sliding_window / attention_chunk_size 等） |
| `_null_block` | `KVCacheBlock` | `block_id=0` 的占位符，用于 SWA / Mamba / ChunkedLocal 跳过窗口/分块外的位置 |
| `_partial_hit_reqs` | `dict[str, tuple[int, KVCacheBlock]]` | 部分命中需 CoW 的登记：`{req_id: (block_idx, source_block)}`，仅细粒度 manager 写入 |
| `_pending_cow_copies` | `list[tuple[KVCacheBlock, KVCacheBlock]]` | 待 worker 执行的 CoW `(src, dst)` 对，由 `take_pending_cow_copies` 排空 |
| `_pending_partial_tail_offloads` | `list[tuple[str, int, KVCacheBlock, int]]` | Mamba "align" 专用的 partial-tail 异步卸载交接：`(req_id, group_id, block, boundary_tokens)` |
| `new_block_ids` | `list[int]` | 本步新分配的 block ID，供 worker 做显存 zeroing（仅 `_record_new_block_ids=True` 时记录） |
| `enable_caching` | `bool` | 是否启用前缀缓存 |
| `use_eagle` | `bool` | 本组的命中是否要为 EAGLE/MTP 丢弃最后一个命中块；由 coordinator 在初始化后注入 |
| `_record_new_block_ids` | `bool` | `needs_kv_cache_zeroing and spec ∈ {Full, TQFull, MLA, HiddenState}`，决定是否记录 `new_block_ids` |
| `_max_admission_blocks_per_request` | `int \| None` | recycling-aware 准入上限，只给 SWA / ChunkedLocal 设置；其它 spec 为 `None` 表示不设上限 |

> 关键直觉：除了 `req_to_blocks` 和 `num_cached_block` 是「每请求」状态，其余字段都是「每管理器」级别的本组全局簿记。CoW 与 partial-tail 的待办队列都在这里，因为它们都属于「本组单步内需要预约的额外动作」。

---

## 3. 可重写钩子

基类提供了一系列可被子类覆盖的钩子，定义了不同注意力类型的策略差异。所有方法名、签名、默认行为如下：

| 钩子 | 默认行为 | 子类覆盖情况 |
|---|---|---|
| `find_longest_cache_hit` (`single_type_kv_cache_manager.py:547`) | **抽象方法**，必须实现 | 每个子类各有一份策略实现（见 §4） |
| `reachable_block_mask` (`single_type_kv_cache_manager.py:480`) | 返回 `None`，表示「全部 block 都可缓存」（密集策略） | SWA / Mamba 覆盖为稀疏 mask，只保留边界检查点 |
| `get_num_skipped_tokens` (`single_type_kv_cache_manager.py:661`) | 返回 `0`，即不跳过任何 token（Full Attention 不释放旧 block） | SWA / ChunkedLocal / Mamba 覆盖为各自窗口/分块的左边界 |
| `get_num_blocks_to_allocate` (`single_type_kv_cache_manager.py:144`) | 通用公式（见下） | Mamba "align" 覆盖以处理 speculative blocks + 复用上一步块 |
| `allocate_new_blocks` (`single_type_kv_cache_manager.py:330`) | 默认：处理 CoW 重定向 + 长度差额分配 | Mamba "align" 覆盖为 running-state 复用模式 |
| `cache_blocks` (`single_type_kv_cache_manager.py:427`) | 默认：调 `reachable_block_mask` + `block_pool.cache_full_blocks` | FullAttention 追加 `_cache_partial_tail_block`；Mamba 覆盖以追踪 `cached_blocks_this_step`；CrossAttention 直接 raise |
| `add_local_computed_blocks` (`single_type_kv_cache_manager.py:232`) | 默认：touch 命中块 + 跳过块填 null + 登记 partial hit | CrossAttention 覆盖为空操作（不参与缓存共享） |
| `pop_blocks_for_free` (`single_type_kv_cache_manager.py:500`) | 默认：弹出三本字典 + 返回 block 列表 | Mamba "align" 覆盖以清理 `_allocated_block_reqs` 等额外簿记 |
| `get_num_common_prefix_blocks` (abstract) | 必须实现 | FullAttention 数 `ref_cnt == len(req_to_blocks)` 的连续前缀；SWA / ChunkedLocal / Mamba / Cross 直接返回 `0`（cascade attention 未支持） |
| `supports_fine_grained_hash_lookup` (ClassVar) | `False` | FullAttention / Mamba 设为 `True`，使 `resolve_block_hashes` 保留细粒度哈希以供 Phase 2 探测 |

### `supports_fine_grained_hash_lookup` 语义

ClassVar（`single_type_kv_cache_manager.py:42`），表示该管理器是否支持在 `block_size > alignment_tokens`（细粒度）模式下 probe 块内部边界的哈希命中。设为 `True` 时：

- `resolve_block_hashes` 在细粒度模式下保留原始 hash 粒度的列表，不做合并；
- `find_longest_cache_hit` 可以做 Phase 2 内部边界探测（FullAttention、Mamba）。

设为 `False`（SWA、ChunkedLocal）时，`find_longest_cache_hit` 会断言 `alignment_tokens % block_size == 0`，**不支持细粒度 partial 命中**。

### `get_num_blocks_to_allocate` 默认公式

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
num_skipped_tokens  = self.get_num_skipped_tokens(total_computed_tokens)
num_skipped_blocks  = num_skipped_tokens // self.block_size
num_new_blocks = max(
    num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
    0,
) + num_evictable_blocks
if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
    num_new_blocks += 1   # 为 CoW 预留一个额外 block
```

要点：
- `num_skipped_blocks`（窗口/分块外的旧 block）和 `num_local_computed_blocks`（命中块 + 已分配块）取**大值**扣减，因为它们互斥：落在窗口外就不用实际有 KV 的块，命中块多就不用 skip 的 null。
- `num_evictable_blocks` 是 `new_computed_blocks` 里 `ref_cnt == 0`（在 free queue 中）的命中块——`touch` 时会从 free queue 摘除，相当于消耗空闲预算，必须计入。
- partial hit 时 `+1` 为 CoW 重定向预留空间，由 `allocate_new_blocks` 用掉。

---

## 4. 子类速查表

七个子类对应七种注意力类型（注册见 `register_all_kvcache_specs`，`single_type_kv_cache_manager.py:1881`）：

| 管理器 | 源码位置 | 查找方向 | 细粒度 | 核心策略 |
|---|---|---|---|---|
| `FullAttentionManager` | `:678` | 左→右 | True (`:679`) | **密集缓存**：满块从左扫到第一个 miss 即 break（链式哈希保证后续全 miss）；fine-grained 模式下 Phase 2 探测首个非满块的内部边界。`cache_blocks` 在 `block_size > hash_block_size` 时额外缓存 prompt 尾部的 partial 哈希。支持 CoW |
| `SlidingWindowManager` | `:878` | 右→左 | False | **稀疏缓存**：窗口外的 block 用 `_null_block` 填充；命中需连续 `_contiguous_blocks_for_hit = cdiv(window-1, block_size)` 块（EAGLE 时 +1 用来 drop tail）；从右往左找最长连续命中，找到后 trim 尾部。`reachable_block_mask` 只缓存窗口尾块以节省空间 |
| `RSWAManager` | `:832` | 左→右（继承 Full） | True (继承) | **Reference SWA**：prefill 前缀全保留（继承 FullAttention 查找），decode 时窗口外的 gap block 通过 `remove_skipped_blocks` 释放，约束每请求内存为 `O(prefix_len + rswa_window)`。不设 admission cap（峰值仍 ≤ max_model_len） |
| `ChunkedLocalAttentionManager` | `:1095` | 左→右 | False | **按 `attention_chunk_size` 分块**：当前 chunk 之前的整块全部用 `_null_block` 标记为 computed（注意力扫不到），只在当前 chunk 内做满块扫描命中。`get_num_skipped_tokens = (computed // chunk_size) * chunk_size` |
| `MambaManager` | `:1253` | 右→左 | True (`:1254`) | **状态快照**：只命中最后一个状态块；`get_num_skipped_tokens = computed - 1`（只保留最后一个状态）。支持 `mamba_cache_mode="align"`：每步复用上一步的 running state 块，partial hit 时把缓存条目 `move_block_hashes` 到 cow_block（运行中请求的 block table 是 append-only，不能原地替换）。`reachable_block_mask` 做稀疏状态保留 |
| `CrossAttentionManager` | `:1747` | 不支持 | False | **编码器状态每请求唯一**：encoder KV 不可共享，`add_local_computed_blocks` / `allocate_external_computed_blocks` / `cache_blocks` / `find_longest_cache_hit` 全部空操作或 raise。`get_num_common_prefix_blocks` 永远返回 `0` |
| `SinkFullAttentionManager` | `:1810` | 左→右（继承 Full） | True (继承) | **Sink block 常驻**：构造时从 `free_block_queue.popleft_n(sink_len // block_size)` 取出 sink 块长期持有，其余行为继承 `FullAttentionManager`。用于 attention sink 场景，sink 区域不可被驱逐 |

### `get_manager_for_kv_cache_spec` 工厂

`single_type_kv_cache_manager.py:1836` 是创建子类的入口：

```python
def get_manager_for_kv_cache_spec(kv_cache_spec, max_in_flight_tokens, max_model_len, **kwargs):
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    if isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens, max_model_len=max_model_len)
        )
    return manager_class(kv_cache_spec, **kwargs)
```

- 通过 `KVCacheSpecRegistry` 查表，**支持 `@register_kv_cache_spec` 注册的自定义 spec**；
- 只对 SWA / ChunkedLocal 计算 `max_admission_blocks_per_request` 做 recycling-aware 准入上限，与启动期 pool sizer 用同一份公式（单一真相源），保证 `sum(reservations) <= pool`、`sum(peak_real_held) <= pool`，避免 issue #39734 的死锁。

### 注册表覆盖关系

`register_all_kvcache_specs`（`:1881`）注册了 9 个 spec → 7 个 manager 的映射，关键点：

- `FullAttentionSpec` / `TQFullAttentionSpec` / `MLAAttentionSpec` / `HiddenStateCacheSpec` → `FullAttentionManager`（`uniform_type_base_spec=FullAttentionSpec`，归为一组）；
- `RSWASpec` → `RSWAManager`（base spec 仍是 `FullAttentionSpec`，与 Full 同组兼容）；
- `SinkFullAttentionSpec` → `SinkFullAttentionManager`（同 Full 组）；
- `SlidingWindowMLASpec` → `SlidingWindowManager`；
- `MambaSpec` / `ChunkedLocalAttentionSpec` / `CrossAttentionSpec` 各自独立。

---

## 5. FullAttention 两阶段查找

`FullAttentionManager.find_longest_cache_hit`（`single_type_kv_cache_manager.py:682-777`）是所有 manager 中最复杂、也是混合模型协调时最常走到的查找路径。它包含两个阶段，外加 EAGLE 裁剪和对齐 trim。

### Phase 1：从左到右满块扫描

```
for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
    cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
    if not cached_block:
        break       # 链式哈希：一个 miss 后面全 miss
    for computed, cached in zip(computed_blocks, cached_block):
        computed.append(cached)
hit_length = len(computed_blocks[0]) * block_size
```

- **`full_block_hashes`** 是 `resolve_block_hashes` 处理后的 hash 序列；细粒度模式下被包成 `BlockHashListWithBlockSize`（`single_type_kv_cache_manager.py:722`），按 `block_size` 视角迭代；
- **任一 group miss 即整体 miss**：`get_cached_block(hash, group_ids)` 要求所有 group 同时命中，否则返回 `None`；
- **遇到第一个 miss 立即 break**：链式哈希的特性保证了后续 block 必然也 miss。

### Phase 2：细粒度内部边界探测（仅 fine-grained 模式）

触发条件（`single_type_kv_cache_manager.py:716-719`）：

```python
fine_grained = alignment_tokens < block_size and block_size % alignment_tokens == 0
```

即 `hash_block_size < block_size` 的混合模型场景（如 Full=16 + Mamba=32 时，Full 这侧的 `alignment_tokens=32` 反而 > `block_size=16`，则不走 Phase 2；反过来如果 scheduler 对齐粒度更细就走）。逻辑：

```
scale_factor = block_size // alignment_tokens
first_partial_idx = len(computed_blocks[0]) * scale_factor    # 满块之后第一个 sub-hash 索引
max_partial_idx = min(
    first_partial_idx + scale_factor - 1,   # 本块内最后一个 sub-hash
    max_length // alignment_tokens,          # 上界
    len(block_hashes),                       # 实际 hash 数
)

# 从最长到最短逆向探测，命中即收尾（longest hit first）
for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
    cached_tail = block_pool.get_cached_block(block_hashes[fine_idx], kv_cache_group_ids)
    if not cached_tail:
        continue
    for computed, cached in zip(computed_blocks, cached_tail):
        computed.append(cached)
    hit_length = (fine_idx + 1) * alignment_tokens
    break
```

要点：
- 只有**第一个非满块**会被探测（之后的块内容还没算出来不可能命中）；
- 从最长 sub-hash 往最短扫，保证拿到的是「最长有效命中」；
- `hit_length` 严格落在 `alignment_tokens` 的整数倍上。

### EAGLE 丢弃 + 对齐 trim

```
# EAGLE 需要重算生成点前一单位的 KV（拿 draft head 的 hidden state）
if drop_eagle_block and hit_length > 0:
    hit_length -= min(alignment_tokens, block_size)

# 向下对齐到 alignment_tokens（细粒度模式天然对齐，普通模式 == block_size 时也无操作）
hit_length -= hit_length % alignment_tokens

num_blocks = cdiv(hit_length, block_size)
for computed in computed_blocks:
    del computed[num_blocks:]   # 截掉超长命中的尾部
```

### 完整代码块

下方是源文件中的实际实现（`single_type_kv_cache_manager.py:728-777`）：

```python
computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
    [] for _ in range(len(kv_cache_group_ids))
)
# Phase 1: longest run of cached full blocks from the start. A missing
# block implies every later block misses too (chained hashes).
for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
    cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
    if not cached_block:
        break
    for computed, cached in zip(computed_blocks, cached_block):
        computed.append(cached)
hit_length = len(computed_blocks[0]) * block_size

# Phase 2 (fine-grained only): extend into the first non-full block by
# probing its interior hash boundaries high-to-low (longest hit first).
if fine_grained:
    assert isinstance(block_hashes, Sequence)
    scale_factor = block_size // alignment_tokens
    first_partial_idx = len(computed_blocks[0]) * scale_factor
    max_partial_idx = min(
        first_partial_idx + scale_factor - 1,
        max_length // alignment_tokens,
        len(block_hashes),
    )
    for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
        cached_tail = block_pool.get_cached_block(
            block_hashes[fine_idx], kv_cache_group_ids
        )
        if not cached_tail:
            continue
        for computed, cached in zip(computed_blocks, cached_tail):
            computed.append(cached)
        hit_length = (fine_idx + 1) * alignment_tokens
        break

# Eagle needs the tokens right before the generation point recomputed:
# drop one hash unit when fine-grained (the tail block's KV is
# append-only, so it still covers the reduced length), else one cache block.
if drop_eagle_block and hit_length > 0:
    hit_length -= min(alignment_tokens, block_size)
# Round down to the alignment; a no-op when fine-grained (hits land on
# hash boundaries by construction) and when alignment_tokens == block_size.
# Then trim blocks past the new tail.
hit_length -= hit_length % alignment_tokens
num_blocks = cdiv(hit_length, block_size)
for computed in computed_blocks:
    del computed[num_blocks:]
return computed_blocks, hit_length
```

---

## 6. cache_blocks — 存入前缀缓存

`cache_blocks`（`single_type_kv_cache_manager.py:427`）在每步 GPU 计算后由 coordinator 调用，把新填满的 block 注册进前缀缓存。基类默认实现：

```python
def cache_blocks(self, request, num_tokens, retention_interval=None):
    num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
    num_full_blocks = num_tokens // self.block_size

    if num_cached_blocks >= num_full_blocks:
        return   # 没有新满块，跳过

    reachable_boundaries = [request.num_prompt_tokens - 1]
    if request.shared_prefix_boundary:
        reachable_boundaries.append(request.shared_prefix_boundary)

    block_mask = self.reachable_block_mask(
        start_block=num_cached_blocks,
        end_block=num_full_blocks,
        alignment_tokens=self.scheduler_block_size,
        kv_cache_spec=self.kv_cache_spec,
        use_eagle=self.use_eagle,
        retention_interval=retention_interval,
        reachable_boundaries=reachable_boundaries,
    )
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

### 关键路径

1. **`num_full_blocks` 检查**：`num_tokens // block_size` 计算新满块数；`num_cached_blocks >= num_full_blocks` 时直接返回，避免重复缓存已注册段。
2. **`reachable_block_mask`**：基类返回 `None`（密集缓存），表示「整段都缓存」；SWA / Mamba 重写为稀疏 mask，只保留命中需要的边界检查点 block。
3. **`block_pool.cache_full_blocks`**：实际写哈希索引；若块之前有 partial 哈希则先删除旧的再晋升为 full。
4. **`num_cached_block` 更新**：把已缓存块数推进到 `num_full_blocks`，下次跳过这段。

### partial → full 晋升

`BlockPool.cache_full_blocks` 在写新主哈希前会校验旧 `num_tokens < 新 num_tokens`，并通过 `_remove_cached_block_hashes` 清理旧的 partial 别名，再用 `_insert_block_hash` 注册新的主哈希。这一步是 **partial → full 晋升**：原本以 partial 哈希登记在反向别名表的条目被移除，新 full 哈希成为块的主哈希。

### 稀疏缓存（SWA / Mamba）

`reachable_block_mask` 的稀疏保留由环境变量 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 控制（通过 `retention_interval` 参数传入）：

| `retention_interval` | 行为 |
|---|---|
| `None` | 密集：每个 `alignment_tokens` 边界都留一个 tail（默认） |
| `0` | 只在 `reachable_boundaries`（replay 边界 + 共享前缀交汇点）留 tail |
| `>0` | 每 `retention_interval` 大小段留一个 tail，外加 `reachable_boundaries` |

- **SWA**（`single_type_kv_cache_manager.py:995-1055`）：每个边界留 `_contiguous_blocks_for_hit` 个块作为窗口尾，使该对齐位置的前缀能被窗口命中复用；`reachable_boundaries` 是 replay 边界和共享前缀 junction，防止稀疏保留把已知的复用点丢弃。
- **Mamba**（`single_type_kv_cache_manager.py:1358-1414`）：每个边界留**单个状态块**（mamba 命中只需最后一个状态，没有窗口）。

`reachable_boundaries` 由 coordinator 钉住（见 `KVCacheManager` 的 `_partial_tail_pins`），其中 `shared_prefix_boundary` 来自 `get_computed_blocks` 返回的 `num_uncached_common_prefix_tokens`，是 Marconi-style APC 的「跨请求共享前缀交汇点」。

---

## 7. Copy-on-Write（部分命中的写时复制）

### 触发场景

1. **partial hit ending inside block**：前缀缓存命中结束在 block 内部（`num_local_computed_tokens % block_size != 0`）。
2. **多请求共享同一部分填充的 tail block**：其中任一请求继续写新 token 会污染其他请求的 KV，因此需要 CoW——把共享块的内容拷到新私有块，新写入打到私有块上。

### 检测：`_has_partial_local_hit`

`single_type_kv_cache_manager.py:132-142`：

```python
def _has_partial_local_hit(
    self,
    new_computed_blocks: Sequence[KVCacheBlock],
    num_local_computed_tokens: int,
) -> bool:
    # The local prefix-cache hit ends inside one of this manager's
    # blocks: the shared tail block needs CoW.
    return (
        len(new_computed_blocks) > 0
        and num_local_computed_tokens % self.block_size != 0
    )
```

仅当有命中块且命中长度不是 `block_size` 整数倍时返回 `True`，表示 tail 块只填了一部分，后续会被本请求继续写入。

### 三步执行流程

```
Step 1: 检测与登记（add_local_computed_blocks, :232）
  if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
      block_idx = num_local_computed_tokens // self.block_size
      self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
      self.num_cached_block[request_id] = block_idx    # 限制到满块数，cache_blocks 后续会重存

Step 2: 计算需求时预留 +1 block（get_num_blocks_to_allocate, :226-229）
  if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
      num_new_blocks += 1   # 为 CoW 重定向预留一个额外 block

Step 3: 执行 CoW（allocate_new_blocks, :348-357）
  if request_id in self._partial_hit_reqs:
      block_idx, source_block = self._partial_hit_reqs.pop(request_id)
      cow_block = self.block_pool.get_new_blocks(1)[0]
      self._apply_cow(request_id, block_idx, source_block, cow_block)
      self.new_block_ids.append(cow_block.block_id)
      cow_blocks.append(cow_block)
```

第三步里 `_partial_hit_reqs` 被 `pop` 出来，CoW 现场执行：从 BlockPool 取一个新块 `cow_block`，调 `_apply_cow` 做「原地替换 + 入队待拷贝」。后续普通分配按 `len(req_blocks)` 计算差额，因为 `cow_block` 已经替换了原索引位置，长度保持不变。

### `_apply_cow` 源码

`single_type_kv_cache_manager.py:405-425`：

```python
def _apply_cow(
    self,
    request_id: str,
    block_idx: int,
    source_block: KVCacheBlock,
    cow_block: KVCacheBlock,
) -> None:
    """Redirect a partial prefix-cache hit to a private CoW block.

    Both copy endpoints stay retained until the copy has run on the worker,
    so a same-step free cannot recycle them: ``source_block`` keeps its
    hit-ref, ``cow_block`` takes an extra ref beyond the one handed to
    the request.
    """
    req_blocks = self.req_to_blocks[request_id]
    assert block_idx < len(req_blocks)
    assert req_blocks[block_idx] is source_block
    assert not source_block.is_null and source_block.ref_cnt > 0
    req_blocks[block_idx] = cow_block            # 原地替换为私有块
    self._pending_cow_copies.append((source_block, cow_block))   # 待 worker 拷贝
    cow_block.ref_cnt += 1                       # 额外引用，防止 GC 在拷贝完成前回收
```

不变量：
- **`source_block` 仍是 hit-ref**（在 `add_local_computed_blocks` 里 `touch` 过，`ref_cnt >= 1`），所以同一步的 `free` 不会回收它；
- **`cow_block` 拿到一个额外 `ref_cnt`**，加上从 BlockPool `get_new_blocks` 出来时自带的一个，共 `ref_cnt >= 2`，保证 worker GPU 拷贝完成前不被回收；
- **三个断言**保证调用顺序合法：`block_idx` 在范围内、原块确实是 `source_block`、source 块非 null 且有引用。

### Mamba "align" 的特殊路径

Mamba align 模式（`single_type_kv_cache_manager.py:1545-1651`）下，运行中请求的 worker block table 是 **append-only** 的，不能像默认路径那样原地替换。它走一条特殊路径：

```python
if blocks_allocated:   # running 请求
    # 请求必须留在 source_block 上，把缓存条目 move 到 cow_block
    self.block_pool.move_block_hashes(source_block, cow_block)
    self._pending_cow_copies.append((source_block, cow_block))
    source_block.ref_cnt += 1
    # 如果这是 producer 自己的 boundary tail，登记给 connector 异步卸载
    if boundary_tokens is not None:
        self._pending_partial_tail_offloads.append(
            (request_id, self.kv_cache_group_id, cow_block, boundary_tokens))
else:                   # 首次 prefill，与默认路径相同
    self._apply_cow(request_id, block_idx, source_block, cow_block)
```

### Worker 执行 GPU 拷贝

`take_pending_cow_copies`（`:382`）排空 `_pending_cow_copies`，drain 出 `(src, dst)` 对列表。`KVCacheManager.take_kv_cache_block_copies` 把它打包成 `KVCacheBlockCopy(src_block_id, dst_block_id)` 发给 worker，worker 在 GPU 上做实际的 `torch.Tensor.copy_`，把 source 块的现有 KV 字节拷到 cow 块，之后请求就可以在 cow 块上 append 新 token 了。

---

## 8. 关键不变量

1. **`free()` 逆序释放，尾部先驱逐利于缓存重用**：`single_type_kv_cache_manager.py:527`，`block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))`。尾部 block 的前缀链最长（覆盖更多前缀），先入队意味着在 LRU 队列里更靠前驱逐，但其 `block_hash` 仍在哈希索引里保留。配合 BlockPool 的「有 hash → append_n 队尾、无 hash → prepend_n 队头」分流，整体把无 hash 的死块排到驱逐最前线，保护了有 reuse 价值的块。
2. **`_partial_hit_reqs` 在 `free` 时随同排空**：`pop_blocks_for_free`（`:516`）显式 `self._partial_hit_reqs.pop(request_id, None)`，避免悬挂请求留下 stale 登记。
3. **`num_cached_block` 只追踪 RUNNING**：被抢占的请求的状态在 coordinator 层处理，单类型管理器不保留抢占请求的缓存进度。
4. **每块恰好一个主哈希**：partial hit 的 CoW 之后，`cow_block` 在 `cache_blocks` 被重新注册时晋升出新的主哈希，`source_block` 保留原主哈希继续作为共享前缀被其他请求命中。
5. **CoW 两端在 worker 拷贝完成前都不可回收**：`source_block` 保持 hit-ref，`cow_block` 拿额外 ref——同一步的 `free` 都动不了它们，只有 `take_kv_cache_block_copies` 把它们返回之后 manager 才能释放（见 `KVCacheManager` 的 retained_blocks 处理）。
6. **`_record_new_block_ids` 只对会被 zeroed 的 spec 开启**：`FullAttentionSpec` / `TQFullAttentionSpec` / `MLAAttentionSpec` / `HiddenStateCacheSpec`；其它 spec（SWA、Mamba）不需要 worker 做 zeroing。
7. **`null_block` 不参与计数 / 释放**：CoW / free / 计数都对 `is_null` 做特判。`_null_block` 由 BlockPool 持有，manager 通过 `self.block_pool.null_block` 引用，生命周期跨整个进程。

---

## 9. 与上下游协作

### 向下：BlockPool

所有 manager 共享同一个 `BlockPool` 实例，通过以下接口协作：

| Manager 调用 | BlockPool 接口 | 用途 |
|---|---|---|
| `find_longest_cache_hit` | `get_cached_block(hash, group_ids)` | 哈希正查命中块 |
| `add_local_computed_blocks` | `touch(blocks)` | 命中块 `ref_cnt++`，从 free queue 摘出 |
| `allocate_new_blocks` / `allocate_external_computed_blocks` | `get_new_blocks(n)` | 从队头取 n 个新块，自动驱逐旧缓存 |
| `_apply_cow` | — | 直接拿 `get_new_blocks(1)` 的结果做 cow_block |
| Mamba align | `move_block_hashes(src, dst)` | 把缓存条目从 src 转嫁到 dst（append-only 场景） |
| `cache_blocks` | `cache_full_blocks(...)` / `cache_partial_block(...)` | 注册 hash 到 block 的映射 |
| `free` / `remove_skipped_blocks` | `free_blocks(...)` | 减引用计数，归零分流到 free queue |
| `_get_num_evictable_blocks` | — | 枚举命中块里 `ref_cnt == 0` 的（仍在 free queue 中）|

> 单一 BlockPool 让「跨组驱逐安全」成为可能：group A 拿新块时如果驱逐了某块，hash 索引会被同步清理，group B 后续查 hash 自然 miss，不会拿到被覆盖的旧内容。

### 向上：KVCacheCoordinator

`HybridKVCacheCoordinator`（混合模型）/ `UnitaryKVCacheCoordinator`（单类型）以 `single_type_managers: list[SingleTypeKVCacheManager]` 持有所有子类，做跨组协调：

- **迭代不动点求命中交集**：每轮把 `curr_hit_length` 喂给各 manager 的 `find_longest_cache_hit`，单调递减收敛；FullAttention 向下封闭，后续只 trim 不重查。
- **两阶段分配编排**（修复 issue #33775）：先让所有 manager 的 `add_local_computed_blocks` 完成 touch（`ref_cnt++`），再统一 `allocate_external_computed_blocks` 分配外部 block。这样 group A 申请新块时不会驱逐 group B 还没 touch 的命中块。
- **per-group CoW / partial tail 收集**：`take_pending_cow_copies`、`take_partial_tail_offloads`、`take_new_block_ids` 都在 coordinator 层把所有 manager 的待办队列汇成总表，供 `KVCacheManager` 暴露给 scheduler。
- **`use_eagle` 注入**：coordinator 在确定 attention group 后逐个 manager 设置 `use_eagle` flag，决定其 `find_longest_cache_hit` 是否 drop 最后一个命中块。
- **`shared_prefix_boundary` 钉住**：coordinator 把 `num_uncached_common_prefix_tokens` 作为 `shared_prefix_boundary` 写回 request（`get_computed_blocks` 返回值），进而在 `cache_blocks` 时传入 `reachable_boundaries` 防止稀疏保留把跨组共享前缀交汇点丢掉（Marconi-style APC）。

---

## 10. 设计要点小结

1. **一类型一子类，共用一个 BlockPool**：把跨组协调留给 coordinator，把单组策略留给子类，BlockPool 是唯一的物理块所有者。各 manager 只管「本组」的 req↔block 绑定和策略选择，绝不自己建池。
2. **钩子驱动的策略差异**：通过 `find_longest_cache_hit` / `reachable_block_mask` / `get_num_skipped_tokens` 等钩子，相同的外壳（`allocate_slots`-类编排）适配截然不同的注意力语义——密集 / 滑窗 / 状态机 / 分块。
3. **CoW 三步走，跨调度步骤联动**：检测登记（`add_local_computed_blocks`）→ 预留 +1（`get_num_blocks_to_allocate`）→ 现场执行（`allocate_new_blocks` → `_apply_cow`）。两端块在 worker GPU 拷贝完成前都保持引用，保证一致性。
4. **稀疏缓存保护 + 边界对齐**：SWA / Mamba 通过 `reachable_block_mask` 只缓存命中所需的边界检查点，由 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 控制粒度；`reachable_boundaries` 显式保留已知的复用点，避免稀疏保留把跨请求共享前缀的交汇点误删。
5. **`supports_fine_grained_hash_lookup` 决定链式哈希保留粒度**：Full / Mamba 开启，使 Phase 2 能 probe 块内部边界；SWA / ChunkedLocal 关闭，断言对齐到 `block_size` 整数倍。
6. **`free()` 逆序 + 三层 LRU 驱逐**：尾部先驱逐的约定配合 BlockPool 的「无 hash 入队头 / 有 hash 入队尾」分流，把死块排到驱逐最前线，保护有复用价值的块——这是 manager 与 block_pool 配合实现「前缀缓存友好」的关键。
7. **recycling-aware 准入上限单一真相源**：SWA / ChunkedLocal 的 `max_admission_blocks_per_request` 由 spec 自身计算，启动期 pool sizer 和运行期 admission 都引用同一个公式，避免 `sum(reservations) <= pool` 与 `sum(peak_real_held) <= pool` 之间出现 drift 导致死锁或 OOM（issue #39734）。
8. **可注册 spec 体系**：`KVCacheSpecRegistry` + `get_manager_for_kv_cache_spec` 支持插件式扩展自定义注意力类型，平台层可通过 `current_platform.register_custom_kv_cache_specs` 注入硬件相关 spec。