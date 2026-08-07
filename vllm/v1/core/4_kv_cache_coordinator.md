# KVCacheCoordinator 设计文档

> 五层架构第 4 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) ｜上层 ➔ [`5_kv_cache_manager.md`](./5_kv_cache_manager.md)
>
> 源文件：`vllm/vllm/v1/core/kv_cache_coordinator.py`

## 1. 一句话定位

`KVCacheCoordinator` 是 vLLM v1 多类型 KV-cache 体系下的 **总调度协调器层**：

- 位于 `BlockPool` / `SingleTypeKVCacheManager`（单类型管理器）之上，顶层 `KVCacheManager` 之下；
- 统一调度粒度（`scheduler_block_size`）、跨组预算与各注意力类型前缀命中交集；
- **持有** 一个共享 `BlockPool` + 一组 `single_type_managers`（每个注意力类型一个），所有 manager 共用同一个 block pool；
- 向上层暴露 `find_longest_cache_hit` / `allocate_new_computed_blocks` 等 API，向下游转发到各 single-type manager。

它不直接管理单条 request↔block 的绑定（那是 `SingleTypeKVCacheManager` 的职责），只在「跨组一致性」这一层做协调。

---

## 2. 类体系与工厂选择

### 2.1 抽象基类与三个具体子类

`KVCacheCoordinator(ABC)`（`kv_cache_coordinator.py:60`）定义了构造期与公开 API；具体策略由三个子类承担：

| 协调器类型 | 源码位置 | 适用场景 | 关键行为 |
|---|---|---|---|
| `KVCacheCoordinatorNoPrefixCache` | `kv_cache_coordinator.py:385` | `enable_caching=False` 或不支持前缀缓存 | 支持任意 group 数（含 0 组）；`find_longest_cache_hit` 恒返回空；`get_num_common_prefix_blocks` 恒返回 0 |
| `UnitaryKVCacheCoordinator` | `kv_cache_coordinator.py:435` | 仅 1 个 KV cache group（同构模型） | 断言 `len(kv_cache_groups) == 1` 与 `hash_block_size == block_size`；命中查找顺直委派给唯一 manager，无迭代收敛 |
| `HybridKVCacheCoordinator` | `kv_cache_coordinator.py:521` | ≥2 个 KV cache group（混合模型：Full + SWA + Mamba 等） | `SpecGroup` 分组 + 迭代不动点取交集 + `num_uncached_common_prefix_tokens` 输出 |

### 2.2 工厂函数 `get_kv_cache_coordinator`

`get_kv_cache_coordinator()`（`kv_cache_coordinator.py:851`）依据配置自动选择，决策顺序固定：

```
1. enable_caching=False           ──► KVCacheCoordinatorNoPrefixCache
2. len(kv_cache_groups) == 1      ──► UnitaryKVCacheCoordinator
3. 否则 (≥2 groups)               ──► HybridKVCacheCoordinator
```

注意 `KVCacheCoordinatorNoPrefixCache` 不接收 `enable_caching` 参数，构造时硬编码为 `False`（`kv_cache_coordinator.py:406-418`）；`Unitary` 与 `Hybrid` 则透传该参数。这意味着「禁用缓存」优先级最高，会绕过 group 数判别。

### 2.3 何时各被选中

- **NoPrefixCache**：用户设 `enable_prefix_caching=False`，或后端不支持。允许 0 group（部分自定义 backend）。
- **Unitary**：Llama / Qwen / Mistral 等同构模型——所有 attention 层 spec 相同，仅 1 group。
- **Hybrid**：Gemma3 / LLaMA4（Full + SWA）、Samba（Full + Mamba）、Whisper（含 cross-attention）等 ≥2 group 模型。`verify_and_split_kv_cache_groups` 会断言 `len(attention_groups) > 1`（`kv_cache_coordinator.py:627`）。

---

## 3. 核心字段与 SpecGroup 分组

### 3.1 基类构造期字段（`kv_cache_coordinator.py:65-128`）

| 字段 | 用途 |
|---|---|
| `block_pool` | 唯一共享 `BlockPool`，所有 single-type manager 共用 |
| `single_type_managers` | `tuple[SingleTypeKVCacheManager, ...]`，按 `kv_cache_group_id` 顺序 |
| `scheduler_block_size` | 跨组调度对齐粒度（多 group 时为各 group block size 的 LCM） |
| `eagle_group_ids` | EAGLE/MTP 投机解码需丢弃最后一个命中块的 group 集合；无显式 flag 时保守地全部启用（`:99-104`） |
| `retention_interval` | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`，对 SWA / Mamba 稀疏化检查点；构造时经 `_validate_prefix_cache_retention_interval`（`:30`）校验 |

### 3.2 SpecGroup：按 spec 类型合并 group

`verify_and_split_kv_cache_groups()`（`kv_cache_coordinator.py:601-650`）将「spec 完全相等」的所有 KV cache group 合并成一个 `SpecGroup`，联合查找缓存命中。`SpecGroup`（`kv_cache_coordinator.py:506`）是 `NamedTuple`：

```python
class SpecGroup(NamedTuple):
    spec: KVCacheSpec                       # 合并代表的 spec
    group_ids: list[int]                    # 该 spec 覆盖的所有 group_id
    manager_cls: type[SingleTypeKVCacheManager]  # 同 spec 必须同 manager 类
    use_eagle: bool                         # 任一成员是 EAGLE group 即为 True
```

合并与排序关键步骤：

1. **遍历合并**：相同 spec 的 group 追加进同一 `SpecGroup.group_ids`，并取 `use_eagle` 的析取（`:613-625`）。
2. **FullAttention 排首位**：`attention_groups.sort(key=lambda g: not isinstance(g.spec, FullAttentionSpec))`（`:633-635`）。Full attention 的左→右扫描能给出一个**紧的下界**，后续 group 在更短的上界内搜索，减少工作量。
3. **记录 `full_attention_group_id`**：首个 attention group 是 FullAttention 则记其 `group_ids[0]`，否则 `None`（`:641-644`）。它是 per-group lookup 的稠密参照——full attention 向下封闭，任一其他 group 报告更长 per-group 命中即说明并集不在同一边界（issue #46453）。
4. **传播 `use_eagle` 到各 manager**（`:646-650`）。

---

## 4. 迭代不动点算法 `find_longest_cache_hit`

`HybridKVCacheCoordinator.find_longest_cache_hit`（`kv_cache_coordinator.py:685-817`）求各注意力类型缓存命中的**交集长度**。不同 manager 的命中长度可能不同（Full 密集、SWA 稀疏、Mamba 状态快照），需迭代收敛。

### 4.1 完整伪代码

```python
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length              # 当前轮上界
    longest_hit_length = 0                          # 历史最深命中（用于 uncached 检测）
    hit_blocks_by_group = [None] * num_groups
    hit_length_by_group = [0] * num_groups

    # 简单混合（1 Full + 1 Other）：1 次迭代足够
    is_simple_hybrid = (len(self.attention_groups) == 2
                       and isinstance(self.attention_groups[0].spec, FullAttentionSpec))

    eagle_verified = set()                         # EAGLE drop 验证集合，按 candidate length 记忆

    while True:
        curr_hit_length = hit_length
        for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(self.attention_groups):
            first_group_id = group_ids[0]
            cached_blocks = hit_blocks_by_group[first_group_id]

            # FullAttention 向下封闭优化：首次查找后只 trim
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                curr_hit_length = min(curr_hit_length, hit_length_by_group[first_group_id])
                continue

            drop_eagle_block = use_eagle and idx not in eagle_verified
            _max_length = curr_hit_length
            # EAGLE: 多匹配一个 drop unit 后丢掉，回到 candidate 长度；Mamba 不加 margin
            if drop_eagle_block and not isinstance(spec, MambaSpec):
                eagle_margin = (hash_block_size if enable_partial_hash_hits
                                and manager_cls.supports_fine_grained_hash_lookup
                                and group_block_size > hash_block_size
                                else group_block_size)
                _max_length = min(curr_hit_length + eagle_margin, max_cache_hit_length)

            hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes, max_length=_max_length,
                kv_cache_group_ids=group_ids, block_pool=self.block_pool,
                kv_cache_spec=spec, drop_eagle_block=drop_eagle_block,
                alignment_tokens=self._cache_hit_alignment_tokens,
                dcp_world_size=(self.dcp_world_size if isinstance(spec, FullAttentionSpec) else 1),
            )

            if drop_eagle_block:
                eagle_verified.add(idx)             # 当前 length 下该 EAGLE drop 已应用
            elif _new_hit_length < curr_hit_length:
                eagle_verified.clear()              # length 收缩 → 之前的 EAGLE 验证作废

            curr_hit_length = _new_hit_length
            for gid, blks in zip(group_ids, hit_blocks):
                hit_blocks_by_group[gid] = blks
                hit_length_by_group[gid] = _new_hit_length
            longest_hit_length = max(longest_hit_length, curr_hit_length)

        if curr_hit_length >= hit_length:           # 不动点：未收缩 → 收敛
            break
        hit_length = curr_hit_length
        if is_simple_hybrid:                        # 简单混合不必再迭代
            break

    # 末轮 trim FullAttention 命中块到最终 hit_length
    first_group = self.attention_groups[0]
    if isinstance(first_group.spec, FullAttentionSpec):
        group_block_size = self.single_type_managers[first_group.group_ids[0]].block_size
        num_blocks = cdiv(hit_length, group_block_size)
        for group_id in first_group.group_ids:
            if (blks := hit_blocks_by_group[group_id]) is not None:
                del blks[num_blocks:]
                hit_length_by_group[group_id] = hit_length

    num_uncached_common_prefix_tokens = longest_hit_length - hit_length
    cache_hit_blocks = tuple(
        blocks if blocks is not None else [] for blocks in hit_blocks_by_group)
    return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens
```

### 4.2 收敛性

`hit_length` 在每轮迭代中**单调不增**，且下界为 0，必然收敛。终止条件是 `curr_hit_length >= hit_length`（本轮无收缩）。常见情形（特别是 simple hybrid）一轮即可结束。

### 4.3 关键优化

- **FullAttention 向下封闭**（`:738-745`）：链式哈希保证 miss 后全 miss，FullAttention 只要在首轮上界内查找一次；后续轮直接用 `hit_length_by_group[first_group_id]` trim `curr_hit_length`，省去重复扫描。
- **simple-hybrid 快速路径**（`:718-720, 795-796`）：「1 Full + 1 Other」结构下，第一轮 Full 给出上界，Other 收缩到这个上界内的命中——再迭代也不会改变，于是直接 `break`。
- **EAGLE 验证集合**（`:725, 747, 780-784`）：每个 `attention_group` 的 EAGLE drop 在**同一 candidate length** 下最多应用一次（issue #32802）；一旦 hit_length 收缩（`_new_hit_length < curr_hit_length`），已记录的 `eagle_verified` 全部清空，需在新长度下重新验证。
- **EAGLE margin**：非 Mamba 的 EAGLE group 允许多探一个 drop unit（细粒度时是 `hash_block_size`，否则是 `group_block_size`）再丢；Mamba 因 finder 永不 drop（draft model 无 mamba 层），不加 margin，避免命中越过 candidate。

### 4.4 返回值

返回三元组 `(cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens)`：

- `cache_hit_blocks`：`tuple[list[KVCacheBlock], ...]`，按 group_id 顺序的命中块。
- `hit_length`：调和后的合并命中长度。
- `num_uncached_common_prefix_tokens = longest_hit_length - hit_length`（`:813`）：稀疏保留组（SWA / Mamba）尚未缓存但其他组已命中的共享前缀长度，非混合场景恒为 0。供上层 `KVCacheManager.get_computed_blocks` 计算 `shared_prefix_boundary`，pin junction，避免 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 把这个交汇点驱逐掉（参考 §8.2）。

---

## 5. 两阶段分配 `allocate_new_computed_blocks`

`KVCacheCoordinator.allocate_new_computed_blocks`（`kv_cache_coordinator.py:192-237`）把前缀命中块追加到请求 block 表，并分配外部 connector 缓存对应的 block。

### 5.1 两阶段流程

```
Phase 1: for each manager:
            manager.add_local_computed_blocks(
                request_id, new_computed_blocks[i],
                num_local_computed_tokens, num_external_computed_tokens)
          └─► block_pool.touch(blocks)  → ref_cnt++（命中块从空闲队列救出）

Phase 2: if num_external_computed_tokens > 0:
            for each manager:
                manager.allocate_external_computed_blocks(
                    request_id, num_local_computed_tokens, num_external_computed_tokens)
          └─► block_pool.get_new_blocks(n)  → 可能从队头弹出驱逐候选
```

**顺序至关重要**：必须先让所有 group `touch` 完本地命中块（`ref_cnt` 抬升），再统一 `allocate_external_computed_blocks`，这样每个 group 在做外部 `get_new_blocks` 时拿到的驱逐候选**不可能**是另一个 group 尚未 touch 的命中块。

### 5.2 修复的 issue #33775

旧实现按 group 次序逐个分配，一个 group 的 `allocate_external_computed_blocks` 可能在下一个 group 还没 `touch` 本地命中块之前，从空闲队列队头弹出并驱逐后者的命中块（`block.ref_cnt == 0` 时它是合法驱逐候选）。两阶段分配把「全部 touch」与「全部外部 allocate」拆开，保证跨组 touch-ahead 一致性。

### 5.3 运行中请求短路

`if any(request_id in manager.num_cached_block for manager in self.single_type_managers)`（`:212-217`）：running 请求已记入 `num_cached_block`，不会有新的前缀命中，直接断言 `new_computed_blocks` 全空并返回。

详细上下文见顶层架构 §7.2 step 7「两阶段分配」。

---

## 6. 三种 block_size 的协同

混合模型中不同注意力类型可能有不同物理 `block_size`。为统一哈希计算与调度对齐，系统引入三种尺寸（`resolve_kv_cache_block_sizes`，`kv_cache_utils.py:626-688`）：

| 尺寸 | 含义 | 单 group | 多 group |
|---|---|---|---|
| `scheduler_block_size` | 调度器对齐粒度（`num_computed_tokens` 取整、retention interval 校验基准） | `cache_config.block_size * dcp` | 各 group effective block size 的 **LCM**（Attention group 乘 DCP，Mamba 不乘） |
| `hash_block_size` | 计算 `Request.block_hashes` 的粒度 | = `scheduler_block_size` | `cache_config.prefix_match_unit` override 或各 group block size 的 **GCD** |
| `group.kv_cache_spec.block_size` | 各组实际物理 block 大小 | = `scheduler_block_size` | LCM 的因子（必须能被 `hash_block_size` 整除） |

**示例**：Full Attention `block_size=16`，Mamba `block_size=32`：

```
scheduler_block_size = LCM(16, 32) = 32
hash_block_size      = GCD(16, 32) = 16
```

- 调度以 32 token 粒度对齐；
- 哈希以 16 token 粒度计算，更细，使各 group 都能在自己的 block 边界复用前缀。

### 6.1 退化为 scheduler_block_size 的两种情形

`resolve_kv_cache_block_sizes` 在以下两种情况下回退 `hash_block_size = scheduler_block_size`（关闭细粒度哈希）：

1. **无消费者**：`enable_prefix_caching=False` 且 `kv_transfer_config` 未设（`:664-666`）——block hashes 只被前缀缓存与 KV connector 消费。
2. **Mamba block_size 与 cache block_size 不一致**（`mamba_cache_mode != "align"`，`:671-676`）：Mamba 状态尺寸打破整除性。

### 6.2 `BlockHashListWithBlockSize` 懒加载转换

`BlockHashListWithBlockSize`（`kv_cache_utils.py:2224-2294`）把 `hash_block_size` 粒度的哈希**懒加载**转换为各 group `target_block_size` 粒度的哈希。

关键洞察：链式哈希中**每个 `hash_block_size` 哈希已覆盖其完整前缀**，因此一个 `target_block_size` block 内**最后一个**子哈希就唯一指纹了该 block 的前缀——直接取用它即可：

```python
def _get_value_at(self, idx: int) -> BlockHash:
    # 最后一个 hash_block_size 哈希已链覆盖整个前缀，直接当 target block 哈希
    return self.block_hashes[(idx + 1) * self.scale_factor - 1]
```

**示例**（`hash_block_size=16`，`target_block_size=32`，scale_factor=2）：

```
hash_block_size=16 哈希链：
  Token 0-15: A   (覆盖 prefix 0-15)
  Token 16-31: B  (覆盖 prefix 0-31)
  Token 32-47: C  (覆盖 prefix 0-47)
  Token 48-63: D  (覆盖 prefix 0-63)

target_block_size=32 取值：
  Block 0 (Token 0-31):  block_hashes[(0+1)*2-1] = block_hashes[1] = B
  Block 1 (Token 32-63): block_hashes[(1+1)*2-1] = block_hashes[3] = D
```

`scale_factor = target_block_size // hash_block_size`，必须整除。`__getitem__` 与 `__iter__` 按转换后长度逐个返回，使各 group manager 可像访问普通 `list[BlockHash]` 一样使用。

---

## 7. 关键不变量

1. **共享同一 `BlockPool`**：所有 `single_type_managers` 持有的 `block_pool` 是同一个对象（`__init__` 中先建池再传给每个 manager，`:90-120`）；跨组分配从同一池子取块。
2. **`scheduler_block_size` 是所有 group block size 的公倍数**：基类 `__init__` 断言 `scheduler_block_size % hash_block_size == 0` 且对每个 group `scheduler_block_size % g.kv_cache_spec.block_size == 0`（`:84-87`），保证调度对齐边界覆盖每组物理边界。
3. **`hash_block_size` 是所有 group block size 的公约数**：`HybridKVCacheCoordinator` 额外断言 `block_size % hash_block_size == 0`（`:563-569`）；基类校验保持各组 block size 整除关系。
4. **FullAttention 排在 `attention_groups` 首位**（若存在），为不动点算法提供紧下界，并作为 per-group lookup 的稠密参照（`:633-644`）。
5. **EAGLE drop 在每个 candidate length 下至多应用一次**：`eagle_verified` 集合记忆；length 收缩时清空重建（`:725, 780-784`）。
6. **两阶段分配顺序不可颠倒**：先全组 `add_local_computed_blocks`（touch）再全组 `allocate_external_computed_blocks`（issue #33775，`:219-236`）。
7. **`Unitary` 单组断言**：`hash_block_size == block_size` 与 `len(kv_cache_groups) == 1`（`:477-482`），无细粒度哈希需求。
8. **Hybrid 至少两组**：`verify_and_split_kv_cache_groups` 断言 `len(attention_groups) > 1`（`:627-629`）；构造期 `pcp_world_size == 1`（`:570`）。

---

## 8. 公开 API 速查

下列方法均已在源码中核对。基类 `KVCacheCoordinator` 提供统一实现，子类只覆写 `find_longest_cache_hit`（及 `cache_blocks` 在 Hybrid 中）。

| 方法 | 源码位置 | 一句话说明 |
|---|---|---|
| `find_longest_cache_hit(block_hashes, max_cache_hit_length)` | `:368` (abstract), `:424`/`:486`/`:685` (impl) | 求各类型命中交集，返回 `(blocks_per_group, hit_length, num_uncached_common_prefix_tokens)` |
| `find_longest_cache_hit_per_group(block_hashes, max_cache_hit_length)` | `:819` (Hybrid only) | 各 group **独立**查找，返回 `(blocks_per_group, hit_lengths_per_group)`，供 connector 对账 |
| `allocate_new_computed_blocks(request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens)` | `:192` | 两阶段把命中块追加到请求 block 表，touch 本地 + 分配外部 |
| `allocate_new_blocks(request_id, num_tokens, num_tokens_main_model, num_encoder_tokens)` | `:238` | 给请求分配至少 `num_tokens` 个 token slot 的新 block；`CrossAttentionManager` 走 `num_encoder_tokens` 分支 |
| `cache_blocks(request, num_computed_tokens)` | `:273` (base), `:652` (Hybrid) | 把已计算 token 存入前缀缓存；Hybrid 版本对齐到 `scheduler_block_size` 边界、EAGLE group 多缓存一个 lookahead block |
| `free(request_id)` | `:290` | 逆序释放请求在所有 manager 中的 block |
| `remove_skipped_blocks(request_id, processed_computed_tokens, num_prompt_tokens)` | `:336` | 移除注意力窗口外的 block（如 sliding window 外的旧 block），用 `null_block` 占位 |
| `new_step_starts()` | `:379` | 通知每个 manager 新 step 开始，清空本步临时数据（`new_block_ids`、CoW copies 等） |
| `get_num_blocks_to_allocate(...)` | `:130` | 累加各 manager 需分配 block 数；`CrossAttentionManager` 走静态分配分支 |
| `get_num_common_prefix_blocks(running_request_id)` | `:319` | 各 group 公共前缀块数；NoPrefixCache 恒返回 0 |
| `pop_blocks_for_free(request_id)` | `:300` | 弹出请求 bookkeeping 但**不归还**块池；调用方需逆序 `block_pool.free_blocks` |
| `get_blocks(request_id)` | `:359` | 取请求在各 manager 中的 block 元组 |

---

## 9. 与上下游协作

### 9.1 下游：`single_type_managers` + `block_pool`

```
KVCacheCoordinator
  ├─ block_pool ────────► BlockPool (全模型共享一份)
  │                         └─► KVCacheBlock / FreeKVCacheBlockQueue / BlockHashToBlockMap
  └─ single_type_managers[0..N]
        ├─ FullAttentionManager        ┐
        ├─ SlidingWindowManager        │  都用同一个 block_pool
        ├─ MambaManager                │
        └─ ...                         ┘
```

Coordinator **不直接操作 `KVCacheBlock` 元数据**，所有「ref_cnt 增减」「哈希插入」「空闲队列移动」都通过 manager 转发到 `BlockPool`。Coordinator 只负责「跨组一致性」与「调度粒度对齐」。

### 9.2 上游：`KVCacheManager` 委派

`KVCacheManager` 是面向 Scheduler 的门面，自己几乎不实现调度逻辑，全部转调：

- `coordinator.find_longest_cache_hit` ← `manager.get_computed_blocks`
- `coordinator.allocate_new_computed_blocks` / `allocate_new_blocks` ← `manager.allocate_slots`
- `coordinator.cache_blocks` ← `manager.cache_blocks`
- `coordinator.free` ← `manager.free`
- `coordinator.new_step_starts` ← `manager.new_step_starts`
- `coordinator.remove_skipped_blocks` ← `manager.allocate_slots` 阶段 1
- `coordinator.get_num_blocks_to_allocate` ← `manager.allocate_slots` 容量检查
- `coordinator.get_num_common_prefix_blocks` ← `manager.get_num_common_prefix_blocks`

同时 `KVCacheManager` 也直接访问 `coordinator.block_pool`（如 `take_events`、`get_num_free_blocks`），绕过 coordinator 取底层池状态。

---

## 10. 设计要点小结

1. **协调器只管跨组**：单组 request↔block 绑定交给 single-type manager，协调器只做「跨组命中取交集」「调度粒度对齐」「跨组 touch-ahead」。
2. **工厂先按缓存开关、再按 group 数决策**：`NoPrefixCache` → `Unitary` → `Hybrid` 顺序固定，避免 `Hybrid` 介入单组场景。
3. **SpecGroup 按 spec 合并 + FullAttention 首位**：同 spec 的 group 联合查找，避免重复扫描；Full attention 的左→右扫描给出紧下界，减少后续 group 工作量。
4. **迭代不动点收敛**：`hit_length` 单调不减地有下界，必然收敛；FullAttention 向下封闭、simple-hybrid 快速路径、EAGLE 验证集合共同把常见路径压缩到一轮。
5. **两阶段分配修复 #33775**：先全组 `touch` 本地命中块（`ref_cnt++`），再全组分配外部 block，杜绝跨组驱逐竞态。
6. **三种 block_size 协同**：`scheduler_block_size = LCM`、`hash_block_size = GCD`，`BlockHashListWithBlockSize` 利用链式哈希「最后一个子哈希覆盖整个前缀」的特性懒加载转换。
7. **`num_uncached_common_prefix_tokens` 守护稀疏保留组的共享前缀**：供上层 pin junction，避免 retention interval 驱逐交汇点。
8. **共享单一 `BlockPool`**：所有 manager 共用，跨组分配从同一池取块，空间维度全局可见，便于 watermark 与抢占决策。