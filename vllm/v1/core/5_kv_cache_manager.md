# KVCacheManager 设计文档（第 5 层：顶层接口层）

> 五层架构第 5 层（最顶）｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)
>
> 源文件：`vllm/vllm/v1/core/kv_cache_manager.py`

## 1. 一句话定位

`KVCacheManager` 是 Scheduler 与 KV cache 管理体系交互的**唯一入口**。它把内部 `KVCacheCoordinator`（多类型协调器）和 `BlockPool`（块池）封装成一个门面（facade），向 Scheduler 屏蔽多 group（Full/SWA/Mamba/Cross 等）的复杂性；而 `KVCacheBlocks` 则是 Scheduler ↔ Manager 之间**唯一的数据交换协议**，把内部 `KVCacheBlock` 对象的细节统统隐藏掉。

```
┌───────────────────────────────────────────────┐
│                  Scheduler                     │
├───────────────────────────────────────────────┤
│              KVCacheManager（门面）              │
│    唯一直接交互对象，转调 coordinator / block_pool │
│         ↓ 唯一交换协议：KVCacheBlocks            │
├───────────────────────────────────────────────┤
│         KVCacheCoordinator + BlockPool         │
└───────────────────────────────────────────────┘
```

本文聚焦这两个顶层组件。Coordinator 的内部迭代算法、BlockPool 的哈希表实现见同目录其他文档。

---

## 2. KVCacheBlocks — 调度接口数据协议

**源码位置**：`kv_cache_manager.py:33`（`@dataclass` 装饰器在 `kv_cache_manager.py:32`）

`KVCacheBlocks` 是 `KVCacheManager.allocate_slots` 等方法返回结果的外壳。Scheduler 只通过它的几个方法获取 `block_id`，不接触底层 `KVCacheBlock` 对象，保证后续内部重构不会影响上层代码。

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
```

### 2.1 字段语义：`blocks[i][j]`

- **第一维 `i`** — 第 `i` 个 KV cache group。一个模型可能包含多个 group（encoder-decoder、多模态、speculative decoding 等场景）。
- **第二维 `j`** — 该 group 内第 `j` 个**逻辑 block**。一个逻辑 block 固定容纳 `block_size` 个 token 的 KV；`j` 按 token 序列中出现的先后顺序递增。

具体地，假设 `block_size = 16`，某 group 负责长度为 50 的序列：

```
token index:  [0..15]  [16..31]  [32..47]  [48..49]
              ▼         ▼         ▼         ▼
blocks[i][j]:  j=0       j=1       j=2       j=3
```

- `blocks[i][0]` 储存 token 0~15 的 KV；
- `blocks[i][3]` 只储存 token 48~49 的 KV，是**未满 block**（partial）。

所以"第 `j` 个 block"应理解为：**把连续 token 序列按固定 `block_size` 切分，第 `j` 个 chunk 对应的逻辑 block**，而非"第 `j` 个 token"。

**为什么 group 做外维**：源码注释（`kv_cache_manager.py:43-48`）明说——如果把 block index 做外维，就隐含假设"所有 group 的 block 数相同"。现在虽然成立，未来若支持不同 `block_size` 就会打破此假设，group 做外维更具扩展性。

**为什么是 `tuple` 而非 `list`**：分配结果一旦产生就是只读的；`tuple` 既保证不可变，也便于对象复用、安全共享（见 §2.5 `new_empty`）。

### 2.2 `__add__`：拼接两段分配结果

```python
def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
    return KVCacheBlocks(
        tuple(
            list(itertools.chain(blk1, blk2))
            for blk1, blk2 in zip(self.blocks, other.blocks)
        )
    )
```
（`kv_cache_manager.py:55-62`）

典型用法是把**前缀缓存命中的 blocks** 和**新分配的 blocks** 合并成一个完整请求的 block 序列（`Scheduler` 在 waiting 队列处理时把 `get_computed_blocks` 的命中块和 `allocate_slots` 分配的新块拼接）。`itertools.chain` 避免手动循环，`zip` 保证两个对象的 group 数量一致。

### 2.3 `get_block_ids`：转为整数 block_id

```python
def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
    if allow_none and all(len(group) == 0 for group in self.blocks):
        return None
    return tuple([blk.block_id for blk in group] for group in self.blocks)
```
（`kv_cache_manager.py:76-91`）

- 返回结构：`tuple[list[int], ...]`，外层 tuple 对应 group，内层 list 是该 group 的 `block_id` 序列。
- `allow_none=True` 时，若所有 group 都为空则返回 `None`，方便上层快速判断无需向 Worker 发送 zeroing 任务。
- 源码用 `@overload` 在类型层面区分 `allow_none=True/False` 的返回签名（`kv_cache_manager.py:64-74`）。

### 2.4 `get_unhashed_block_ids` / `get_unhashed_block_ids_all_groups`

```python
def get_unhashed_block_ids(self) -> list[int]:
    assert len(self.blocks) == 1, "Only one group is supported"
    return [block.block_id for block in self.blocks[0] if block.block_hash is None]

def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
    # Skip padding blocks.
    return [
        [block.block_id for block in group
         if block.block_hash is None and not block.is_null]
        for group in self.blocks
    ]
```
（`kv_cache_manager.py:93-108`）

- **用途**：找出尚未被前缀缓存的 block（`block_hash is None`），这些 block 在 GPU 计算前通常需要执行 **zeroing**，防止旧 KV 值干扰。
- **单 group 版本** `get_unhashed_block_ids`：内部 `assert` 保护，只支持单 group，返回扁平 `list[int]`。
- **多 group 版本** `get_unhashed_block_ids_all_groups`：按 group 返回 `list[list[int]]`，并额外 **跳过 `is_null` 的占位/padding block**（null block 不需 zeroing）。

### 2.5 `new_empty`：复用空对象

```python
def new_empty(self) -> "KVCacheBlocks":
    return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))
```
（`kv_cache_manager.py:110-114`）

构造一个 group 结构相同但每个 group 均为空的 `KVCacheBlocks`。`KVCacheManager` 会预计算并复用这个空对象（`empty_kv_cache_blocks`，`kv_cache_manager.py:185`），避免无 KV block 的请求反复触发对象分配和 GC（见源码注释 `kv_cache_manager.py:180-184`）。

### 2.6 方法速查表

| 方法 | 功能 | 关键说明 |
|------|------|----------|
| `__add__(other)` | 按 group 拼接两段 blocks | 前缀命中 + 新分配 blocks 合并 |
| `get_block_ids(allow_none)` | 转为整数 ID 列表 | `allow_none=True` 时空 blocks 返回 `None`；有 `@overload` 区分签名 |
| `get_unhashed_block_ids()` | 单 group 下未缓存 block ID | 用于 Worker zeroing；有 `assert` 单 group 守卫 |
| `get_unhashed_block_ids_all_groups()` | 多 group 下未缓存 block ID | 跳过 `is_null` padding |
| `new_empty()` | 构造同结构空对象 | 预计算复用，避免 GC |

---

## 3. KVCacheManager 构造与核心字段

**源码位置**：`kv_cache_manager.py:117`

### 3.1 `__init__` 签名

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
    ) -> None:
```
（`kv_cache_manager.py:118-133`）

### 3.2 核心字段表

字段在 `kv_cache_manager.py:134-191` 间设置：

| 字段 | 类型 | 用途 | 源码行 |
|------|------|------|--------|
| `coordinator` | `HybridKVCacheCoordinator` 等 | 工厂函数 `get_kv_cache_coordinator()` 根据配置自动选择，持有所有 group 的管理器 | `kv_cache_manager.py:151-163` |
| `block_pool` | `BlockPool` | 块池，引用自 `coordinator.block_pool`（不自建，保证全局一份） | `kv_cache_manager.py:165` |
| `enable_caching` | `bool` | 是否启用前缀缓存 | `kv_cache_manager.py:141` |
| `empty_kv_cache_blocks` | `KVCacheBlocks` | 预构造的空对象，复用以避免 GC 开销 | `kv_cache_manager.py:185-187` |
| `watermark_blocks` | `int` | 水位线 block 数 = `int(watermark * num_blocks)`，admission 时保留的最小空闲块 | `kv_cache_manager.py:171` |
| `kv_cache_event_metadata` | `tuple` | 每 group 的 `(spec_kind, sliding_window)`，用于标注 KV cache 事件 | `kv_cache_manager.py:172-178` |
| `_partial_tail_pins` | `dict[str, list[KVCacheBlock]]` | KV connector 的 partial-tail offload 钉住的 block，请求释放时解钉 | `kv_cache_manager.py:191` |
| `max_in_flight_tokens`（局部回退） | `int` | 未设时回退为 `max_model_len`，使 recycling-aware 上限退化为不设限行为 | `kv_cache_manager.py:138-139` |
| `num_kv_cache_groups` | `int` | group 数量，用于构造空 `KVCacheBlocks` | `kv_cache_manager.py:164` |

### 3.3 委托关系说明

Manager 自己**不实现**调度逻辑，所有调用都转发给 `coordinator` 或 `block_pool`：

```
KVCacheManager
  ├─ self.coordinator  ───────► KVCacheCoordinator (Hybrid/Unitary/NoPrefixCache)
  │                             ├─ self.block_pool  ───────► BlockPool
  │                             └─ self.single_type_managers[0..N]
  │                                  ├─ FullAttentionManager
  │                                  ├─ SlidingWindowManager
  │                                  └─ ...
  └─ self.block_pool = coordinator.block_pool  (同一份引用)
```

Manager 持有 Coordinator；**Coordinator 持有 BlockPool + 一组 SingleTypeKVCacheManager**；每个 single_type_manager 共用同一个 BlockPool。这种"挂引用不重建"的设计保证全模型只有一份 block 池，所有分配/释放/命中操作最终落到同一个 `BlockPool` 实例上。

---

## 4. 方法分类速查表

下表涵盖 `KVCacheManager` 暴露给 Scheduler 的全部公开方法。每个方法名都已对照源码确认存在。

| 分类 | 方法 | 功能 | 关键说明 |
|------|------|------|----------|
| **查缓存** | `get_computed_blocks(request)` | 查找前缀缓存命中 | 返回 `(blocks, num_new_computed_tokens, shared_prefix_boundary)` |
| | `get_computed_blocks_for_connector(request)` | 带 KV connector 的前缀查找 | 额外返回 `hit_diverged` 标志，处理 Mamba/Full 命中分歧 |
| | `prefix_cache_lookup_enabled(request)` | 判断是否允许前缀查找 | `enable_caching and not request.skip_reading_prefix_cache` |
| | `estimate_cached_tokens(request)` | 估算请求已缓存 token 数 | 取所有 group 的最小值，跳过 cross-attention/encoder-only |
| **分配** | `allocate_slots(request, ...)` | 分配新 block 槽位 | 核心方法，三阶段分配，空间不足返回 `None` |
| **存储** | `cache_blocks(request, num_computed_tokens)` | 存入前缀缓存 | 委托 `coordinator.cache_blocks`，仅 `enable_caching` 时执行 |
| **释放** | `free(request)` | 释放请求所有 block | 先释放 partial-tail pins，再交 coordinator 逆序释放 |
| | `pop_blocks_for_free(request)` | 弹出但不归还块池 | 供调用方延迟释放，pins 随同弹出 |
| | `remove_skipped_blocks(...)` | 移除不在注意力窗口内的 block | 如 sliding-window 外的旧 block |
| **准备 GPU** | `take_new_block_ids()` | 获取新 block ID 用于 zeroing | 遍历所有 `single_type_managers` drain `new_block_ids` |
| | `take_kv_cache_block_copies()` | 获取 CoW 拷贝任务 | 返回 `(copies, retained_blocks)` |
| | `take_partial_tail_offloads()` | 获取 partial-tail offload 任务 | 仅 Mamba "align" group 贡献 |
| | `get_zeroing_block_ids_in_range(...)` | 获取 `[start, end)` 范围内需 zeroing 的 block ID | |
| | `record_blocks_for_zeroing(...)` | 重新记录从 `start_token` 起需 zeroing 的 block | 用于异步 KV 加载失败后重置 |
| **查询** | `get_blocks(request_id)` | 获取请求的 blocks | |
| | `get_block_ids(request_id)` | 获取请求的 block ID 列表 | |
| | `get_block_ids_for_computed_tokens(...)` | 截取已计算 token 覆盖的 block ID | 按 `spec.block_size` 对齐裁剪 |
| | `get_num_common_prefix_blocks(running_id)` | 公共前缀块数 | 所有 allocated 请求共享的块，非仅当前步调度请求 |
| | `usage` (property) | KV cache 使用率 (0.0–1.0) | 委托 `block_pool.get_usage()` |
| **生命周期** | `new_step_starts()` | 通知协调器新 step 开始 | 委托 `coordinator.new_step_starts()` |
| | `take_events()` | 取出 KV cache 事件 | 标注 spec_kind / sliding_window 元数据 |
| | `reset_prefix_cache()` | 重置前缀缓存 | RLHF 权重更新后或 benchmark 时使用 |
| | `evict_blocks(block_ids)` | 按 block ID 驱逐缓存块 | |

---

## 5. `get_computed_blocks` — 前缀缓存查找

**源码位置**：`kv_cache_manager.py:229-295`

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
```

**返回三元组**：`(blocks, num_new_computed_tokens, shared_prefix_boundary)`

- `blocks`：命中的前缀缓存块（必须已满 block），封装为 `KVCacheBlocks`。
- `num_new_computed_tokens`：本轮新命中的 token 数。
- `shared_prefix_boundary`：稀疏保留组（Mamba / sliding window）尚未缓存但与 FullAttention 共享的前缀边界位置，用于 Marconi-style APC；无则返回 0。

**关键逻辑**：

1. **跳过条件**：`prefix_cache_lookup_enabled()` 为 `False`（`kv_cache_manager.py:250`）——缓存禁用或请求标记 `skip_reading_prefix_cache`（如 prompt logprobs / pooling 模型）——直接返回 `(empty_kv_cache_blocks, 0, 0)`。
2. **留一重算**（`kv_cache_manager.py:259`）：
   ```python
   max_cache_hit_length = request.num_tokens - 1
   ```
   全部 token 命中时，最后一个 token 仍须重新计算以获取 logits。这也可能触发整 block 重算（因 `allocate_slots` 要求 `num_computed_tokens` 按 `block_size` 对齐）。
3. **委托查找**（`kv_cache_manager.py:260-264`）：
   ```python
   computed_blocks, num_new_computed_tokens, num_uncached = (
       self.coordinator.find_longest_cache_hit(
           request.block_hashes, max_cache_hit_length
       )
   )
   ```
4. **事件广播**（`kv_cache_manager.py:269-284`）：若 `kv_cache_report_mode == "full"` 且启用事件，为命中的每个 group 发射 `BlockStored` 事件（供 gateway 等外部消费者感知复用）。
5. **shared_prefix_boundary**（`kv_cache_manager.py:290-292`）：
   ```python
   shared_prefix_boundary = (
       num_new_computed_tokens + num_uncached if num_uncached else 0
   )
   ```
   该边界会被钉住，防止 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 驱逐这个交汇点。

---

## 6. `get_computed_blocks_for_connector` — 带 Connector 的前缀查找

**源码位置**：`kv_cache_manager.py:297-342`

```python
def get_computed_blocks_for_connector(
    self, request: Request
) -> tuple[KVCacheBlocks, int, int, bool]:
```

**额外返回 `hit_diverged`**：当 full-attention 命中比某些滞后 group 更深时为 `True`，表示该边界处 Mamba 状态缺失，调用方需回退 `get_computed_blocks` 做对账。

**触发场景**：混合模型（Mamba + FullAttention）中，不同 group 的前缀命中在 block 压力下可能产生分歧——FullAttention 尾部可能被驱逐而 Mamba 状态存活更深（或反之）。

**实现路径**：

1. **非混合模型快速路径**（`kv_cache_manager.py:319-324`）：若不是 Mamba 混合或 coordinator 非 Hybrid，直接返回 `(*get_computed_blocks(request), False)`。
2. **混合模型精确 per-group 查找**（`kv_cache_manager.py:329-342`）：
   ```python
   computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
       request.block_hashes, request.num_tokens - 1
   )
   ```
   - 若某 group 比 FullAttention 命中更深 → FullAttention 尾块被驱逐，回退 `get_computed_blocks` + `hit_diverged=False`。
   - 否则取 FullAttention 命中作为本地前缀，`hit_diverged = min(per_group_hits) < num_local`。connector 只需传输剩余后缀，Mamba 状态由 nixl 的 `_apply_prefix_caching` 无条件传输。

---

## 7. `allocate_slots` — 核心分配方法

**源码位置**：`kv_cache_manager.py:344-565`

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
    has_scheduled_reqs: bool = True,
) -> KVCacheBlocks | None:
```

Scheduler 调度请求时最核心的方法。空间不足时返回 `None` 表示无法调度。

### 7.1 Block 布局图

源码注释（`kv_cache_manager.py:390-411`）：

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                  |            < to be allocated >                   |
----------------------------------------------------------------------
                  | < to be cached (roughly, |
                  | details below)>          |
----------------------------------------------------------------------
| Prefix-cached tokens from either vLLM   |
| or connector. Can be safely removed if  |
| they are outside sliding window.        |
----------------------------------------------------------------------
|   < cached by vLLM >    | not cached by |
                          | vLLM, but     |
| ref_cnt  | ref_cnt not  | cached by     |
| increased| increased yet| connector     |
----------------------------------------------------------------------
```

缩写（`kv_cache_manager.py:413-422`）：

| 缩写 | 含义 |
|------|------|
| `comp` | `request.num_computed_tokens`（已计算） |
| `new_comp` | `num_new_computed_tokens`（本轮新命中缓存）= `len(new_computed_blocks) * block_size` |
| `ext_comp` | `num_external_computed_tokens`（connector 缓存） |
| `new` | `num_new_tokens`（新计算，含未验证 draft） |
| `lookahead` | `num_lookahead_tokens`（投机解码 lookahead） |

底部三行说明 ref_cnt 状态：`comp`/`new_comp` 已 `ref_cnt++`；`ext_comp` 由 connector 持有，**vLLM 尚未增引用**。

> NOTE（`kv_cache_manager.py:424-426`）：`new` 包含 verified + unverified draft token，但只 cache verified（以 `request.num_tokens` 为上限）。

### 7.2 三阶段分配流程

**阶段 1：admission gate + 移除窗口外 block**（`kv_cache_manager.py:440-508`）

- **入参校验**（`kv_cache_manager.py:442-446`）：`num_new_tokens == 0 and num_external_computed_tokens == 0` 时直接 `ValueError`。
- **计算本地命中**（`kv_cache_manager.py:455-461`）：
  ```python
  num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
  total_computed_tokens = min(
      num_local_computed_tokens + num_external_computed_tokens,
      self.max_model_len,
  )
  ```
- **Watermark 决定**（`kv_cache_manager.py:463-470`）：仅 `has_scheduled_reqs and request.status in (WAITING, PREEMPTED)` 时取 `self.watermark_blocks`，否则 0。
- **admission gate**（`kv_cache_manager.py:472-488`）：若 `full_sequence_must_fit=True`，先计算整条 sequence 需要的 block 数（`get_num_blocks_to_allocate(apply_admission_cap=True)`）+ watermark，超出 `block_pool.get_num_free_blocks()` 则返回 `None`。
- **remove_skipped_blocks**（`kv_cache_manager.py:504-508`）：调用 `coordinator.remove_skipped_blocks(request_id, max(0, total_computed_tokens - request.num_in_flight_tokens), num_prompt_tokens=...)`，释放超出注意力窗口的 block（如 sliding-window 外），**即使最终不调度也执行此清理**。参数保证 in-flight 步骤的注意力窗口仍可读。

**阶段 2：处理前缀 token（comp + new_comp + ext_comp）**（`kv_cache_manager.py:510-540`）

- 计算 `num_blocks_to_allocate`（`coordinator.get_num_blocks_to_allocate()`，`kv_cache_manager.py:510-519`）。
- 容量检查（`kv_cache_manager.py:523-527`）：
  ```python
  available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
  required_blocks = num_blocks_to_allocate + watermark_blocks
  if required_blocks > available_blocks:
      return None
  ```
- 若有前缀命中或外部 token，调用 `coordinator.allocate_new_computed_blocks(...)`（`kv_cache_manager.py:529-540`）将命中块追加到请求的 block 列表（增加 `ref_cnt`）。这是**两阶段分配的第一阶段——先 touch 所有本地命中块**，防止跨组驱逐（修复 issue #33775）。

**阶段 3：分配新 block + 缓存**（`kv_cache_manager.py:542-565`）

- `coordinator.allocate_new_blocks(request_id, num_tokens_need_slot, num_tokens_main_model, num_encoder_tokens)`（`kv_cache_manager.py:542-547`）—— 从 free queue 取新 block；如有 partial_hit 则执行 CoW 重定向。
- **delay 路径**（`kv_cache_manager.py:551-552`）：若 `delay_cache_blocks=True`（P/D 异步 KV 传输）或 `enable_caching=False`，跳过缓存，直接返回新分配的 blocks。
- **正常路径**（`kv_cache_manager.py:559-563`）：
  ```python
  num_tokens_to_cache = min(
      total_computed_tokens + num_new_tokens,
      request.num_tokens,
  )
  self.coordinator.cache_blocks(request, num_tokens_to_cache)
  ```
  以 `request.num_tokens` 为上限排除未验证 draft token，调用 `coordinator.cache_blocks()` 存入前缀缓存。

### 7.3 参数详解

| 参数 | 用途 | 源码 |
|------|------|------|
| `num_new_tokens` | 需分配并计算的 token 数（含未验证 draft） | `kv_cache_manager.py:347` |
| `num_new_computed_tokens` | 本轮新前缀命中 token（不含外部） | `kv_cache_manager.py:348` |
| `new_computed_blocks` | 上述命中对应的 cached blocks，按 group 分组 | `kv_cache_manager.py:349` |
| `num_lookahead_tokens` | 投机解码 lookahead token 数（eagle 等） | `kv_cache_manager.py:350` |
| `num_external_computed_tokens` | connector 缓存但非 vLLM 缓存的 token 数 | `kv_cache_manager.py:351` |
| `delay_cache_blocks` | P/D 时跳过缓存，等远端 KV 传输在后续 step 完成 | `kv_cache_manager.py:352` |
| `num_encoder_tokens` | encoder-decoder 模型（如 Whisper）的 cross-attention token 数 | `kv_cache_manager.py:353` |
| `full_sequence_must_fit` | admission gate：整条 sequence 必须能放下才分配，防 chunked prefill 过度准入 | `kv_cache_manager.py:354` |
| `reserved_blocks` | 为其他 in-flight 序列保留的空闲块，gate 异步 connector 加载 | `kv_cache_manager.py:355` |
| `has_scheduled_reqs` | 本步是否有已调度请求，控制是否应用 watermark | `kv_cache_manager.py:356` |

### 7.4 Watermark 策略

`watermark_blocks`（`kv_cache_manager.py:171`）= `int(watermark * kv_cache_config.num_blocks)`，是 admission 时保留的最小空闲块数，避免频繁抢占。

应用条件（`kv_cache_manager.py:463-470`）：

```python
if has_scheduled_reqs and request.status in (
    RequestStatus.WAITING,
    RequestStatus.PREEMPTED,
):
    watermark_blocks = self.watermark_blocks
```

即**仅对 WAITING/PREEMPTED 状态的请求，且 `has_scheduled_reqs=True`（本步已有其他请求调度）时应用**。对 running 队列的 decode 不应用——decode 不该因 watermark 被挡住；当没有其他 running 请求时也不应用——避免死锁。

### 7.5 与 §7.2 的 9 步流程对照

可与架构文档 §7.2 的 9 步流程对应理解：

```
步骤 1: 计算 total_computed_tokens / num_tokens_need_slot
         └─ kv_cache_manager.py:455-493
步骤 2: Watermark 水位线检查（仅 WAITING/PREEMPTED + has_scheduled_reqs）
         └─ kv_cache_manager.py:463-470
步骤 3: [可选] full_sequence_must_fit admission gate 预检查
         └─ kv_cache_manager.py:472-488（不通过 return None）
步骤 4: remove_skipped_blocks() 释放窗口外 block
         └─ kv_cache_manager.py:504-508
步骤 5: get_num_blocks_to_allocate() 精确计算需求
         └─ kv_cache_manager.py:510-519
步骤 6: 容量检查 required <= available - reserved
         └─ kv_cache_manager.py:523-527（不通过 return None）
步骤 7: 两阶段分配 —— allocate_new_computed_blocks (touch 命中块, ref_cnt++)
         └─ kv_cache_manager.py:529-540
步骤 8: allocate_new_blocks() 取新 block + CoW 重定向
         └─ kv_cache_manager.py:542-547
步骤 9: cache_blocks() + 返回
         └─ kv_cache_manager.py:551-565
```

---

## 8. 其他关键方法

### 8.1 `free(request)`

**源码位置**：`kv_cache_manager.py:567-578`

先弹出 `_partial_tail_pins` 中钉住的 block 释放回块池，再调用 `coordinator.free(request_id)` 逆序释放所有 block（尾部先驱逐，利于缓存重用）。

```python
pins = self._partial_tail_pins.pop(request.request_id, None)
if pins:
    self.block_pool.free_blocks(pins)
self.coordinator.free(request.request_id)
```

### 8.2 `take_new_block_ids()`

**源码位置**：`kv_cache_manager.py:796-801`

遍历 `coordinator.single_type_managers`，drain 各 manager 的 `new_block_ids` 列表，返回全部新 block ID。Worker 拿到这些 ID 后对 GPU KV cache 执行 zeroing。

### 8.3 `take_kv_cache_block_copies()`

**源码位置**：`kv_cache_manager.py:831-846`

drain 各 manager 的 pending CoW 拷贝，构造 `KVCacheBlockCopy(src_block_id, dst_block_id)` 列表。同时返回需保留的 source 和 cow block 列表（防止 GC 回收正在拷贝的 block）。Worker 端拿到 `(src_id, dst_id)` 对后执行 GPU tensor copy。

### 8.4 `take_partial_tail_offloads()`

**源码位置**：`kv_cache_manager.py:848-874`

drain Mamba "align" group 的 partial-tail offload，返回 `{request_id: [(group_id, block_id, boundary_tokens), ...]}`。被取出的 block 不在请求 block 表中，因此在此钉住（`block_pool.touch` + `_partial_tail_pins`），直到请求释放时由 `free()` 解钉。

### 8.5 `take_events()`

**源码位置**：`kv_cache_manager.py:677-701`

从 `block_pool` 取出 KV cache 事件，为每个 `BlockStored` 事件标注 `kv_cache_spec_kind` 和 `kv_cache_spec_sliding_window`（从 `kv_cache_event_metadata` 中按 `group_idx` 查找）。这样 BlockPool 发射结构化事件时不必持有语义 spec 元数据。

### 8.6 `estimate_cached_tokens(request)`

**源码位置**：`kv_cache_manager.py:731-758`

遍历各 group 的 block，取 `block.block_hash_num_tokens` 的最大值作为该 group 的缓存 token 数，最终取所有 group 的最小值。跳过 cross-attention 和 encoder-only group（这些不参与前缀缓存）。

### 8.7 `get_num_common_prefix_blocks(running_request_id)`

**源码位置**：`kv_cache_manager.py:643-675`

计算每个 KV cache group 的公共前缀块数。一个 block 被视为 common prefix block 当且仅当**所有已分配 KV cache 的请求都共享它**（`ref_cnt == 已分配请求数`）。返回 `list[int]`，每 group 一项。

> 注（`kv_cache_manager.py:651-665`）："已分配 KV cache"的请求数 ≥ 当前步调度请求数，因为还包含未调度但未释放 block 的请求。这导致存在边界情况：即使所有调度请求共享某前缀，计数可能为 0（被未调度的请求拉低）。用于 Cascade Attention。

### 8.8 `usage` (property)

**源码位置**：`kv_cache_manager.py:193-200`

委托 `block_pool.get_usage()`，返回 0.0–1.0 之间的 KV cache 占用率。

---

## 9. Scheduler 交互节奏

Scheduler 一步调度中与 KVCacheManager 的交互顺序（对应架构文档 §6.2.8）：

```
① new_step_starts()
   └─ 清空前一步临时数据（new_block_ids、CoW copies、partial-tail offloads 等）
      源码：kv_cache_manager.py:876-878

② 处理 running 队列：
   allocate_slots(request, num_new_tokens, has_scheduled_reqs=True)
   └─ 追加 decode token 所需的新 block
      源码：kv_cache_manager.py:344

③ 处理 waiting 队列：
   a) get_computed_blocks(request)
      └─ → (cached_blocks, num_hit, boundary)
         源码：kv_cache_manager.py:229
   b) allocate_slots(request, num_new_tokens,
                     new_computed_blocks=cached_blocks,
                     num_new_computed_tokens=num_hit,
                     full_sequence_must_fit=True,        # admission gate
                     has_scheduled_reqs=True)
   └─ 命中缓存 + 分配新 block；空间不足返回 None 则等待
      源码：kv_cache_manager.py:344

④ GPU 计算完成后：
   cache_blocks(request, num_computed_tokens)
   └─ 把新填满的 block 存进前缀缓存（ref_cnt 管理 + hash 索引）
      源码：kv_cache_manager.py:760-769

⑤ 准备发给 Worker：
   - take_new_block_ids()          → worker 对这些 block 执行 zeroing
     源码：kv_cache_manager.py:796
   - take_kv_cache_block_copies()  → worker 执行 CoW 拷贝
     源码：kv_cache_manager.py:831
   - take_partial_tail_offloads()  → KV connector offload
     源码：kv_cache_manager.py:848
   - take_events()                 → KV cache 事件（BlockStored 等）
     源码：kv_cache_manager.py:677
```

---

## 10. 与下游协作

### 10.1 与 KVCacheCoordinator

Manager 持有 `self.coordinator`（`kv_cache_manager.py:151-163`），由工厂函数 `get_kv_cache_coordinator()` 根据配置自动选择具体子类：

| 协调器类型 | 适用场景 |
|-----------|----------|
| `KVCacheCoordinatorNoPrefixCache` | 禁用前缀缓存 |
| `UnitaryKVCacheCoordinator` | 单一注意力类型 |
| `HybridKVCacheCoordinator` | 混合注意力（Full + SWA + Mamba） |

Manager 调用的 coordinator 方法清单：`find_longest_cache_hit`、`find_longest_cache_hit_per_group`、`remove_skipped_blocks`、`get_num_blocks_to_allocate`、`allocate_new_computed_blocks`、`allocate_new_blocks`、`cache_blocks`、`free`、`get_blocks`、`pop_blocks_for_free`、`get_num_common_prefix_blocks`、`new_step_starts`。这些方法内部再协调各 `single_type_manager`，最终落到 `block_pool`。

### 10.2 与 BlockPool

Manager 通过 `self.block_pool`（引用自 `coordinator.block_pool`，`kv_cache_manager.py:165`）直接调用的方法：

| BlockPool 方法 | Manager 内调用点 |
|----------------|-----------------|
| `get_num_free_blocks()` | `allocate_slots` 容量检查 |
| `get_usage()` | `usage` property |
| `free_blocks(pins)` | `free` 释放 partial-tail pins |
| `touch((block,))` | `take_partial_tail_offloads` 钉住 offload block |
| `emit_cached_block_events(...)` | `get_computed_blocks` 在 `kv_cache_report_mode=="full"` 时补发事件 |
| `evict_blocks(block_ids)` | `evict_blocks` 方法 |
| `reset_prefix_cache()` | `reset_prefix_cache` 方法 |
| `take_events()` | `take_events` 方法 |

---

## 11. 设计要点小结

1. **唯一入口 + 唯一协议**：`KVCacheManager` 是 Scheduler 唯一直接交互对象，`KVCacheBlocks` 是唯一数据交换协议。Scheduler 不接触 `KVCacheBlock`、`Coordinator`、`BlockPool` 的内部细节，便于内部重构。
2. **门面模式**：Manager 不实现调度逻辑，全部转调 `coordinator.*` 或 `block_pool.*`。持有 `block_pool` 仅是引用（不自建），保证全模型只有一份块池。
3. **`KVCacheBlocks` 不可变**：`tuple` 外层 + `Sequence` 内层，分配结果只读，便于对象复用与安全共享。`new_empty()` 和预计算的 `empty_kv_cache_blocks` 避免无 block 请求反复触发 GC。
4. **留一重算**：`get_computed_blocks` 中 `max_cache_hit_length = num_tokens - 1`，强制最后一个 token 重算以获取 logits，即使全部命中缓存。
5. **三阶段分配**：admission gate + remove_skipped → 前缀 token 容量检查 + allocate_new_computed_blocks（两阶段 touch 防跨组驱逐）→ allocate_new_blocks + cache_blocks。
6. **Watermark 准入控制**：仅对 WAITING/PREEMPTED + `has_scheduled_reqs=True` 时应用，预留水位线 block 防止过度准入导致频繁抢占；对 running decode 不应用避免被卡住，无 running 时也不应用避免死锁。
7. **事件旁路**：`take_events()` 标注 spec_kind / sliding_window 元数据，BlockPool 只发射结构化事件不持有语义 spec，松耦合支持 KV connector 等外部消费者。
8. **partial-tail pin**：被 `take_partial_tail_offloads` 取出的 off-table block 由 Manager 钉住（`block_pool.touch` + `_partial_tail_pins`），直到请求 `free()` 时解钉，保证 connector 读取期间不被驱逐。
9. **`full_sequence_must_fit` admission gate**：防止 chunked prefill 只检查首 chunk 导致过度准入；waiting 队列处理时强制整条 sequence 能放下才分配。
10. **`reserved_blocks` 异步保护**：为其他 in-flight 序列保留空闲块，gate 异步 KV-connector 加载，防止新请求吃掉正在运行的 prefill 序列依赖的 block。