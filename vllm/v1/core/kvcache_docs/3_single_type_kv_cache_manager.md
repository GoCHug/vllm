# SingleTypeKVCacheManager 详解

> 五层架构第 3 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`2_block_pool.md`](./2_block_pool.md) ｜上层 ➔ [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)
> 时序位置：[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) B1～E 阶段
>
> 源文件：`vllm/vllm/v1/core/single_type_kv_cache_manager.py`
>
> 主线：纯 Full Attention 单 group，核心是子类 `FullAttentionManager`。**本文重点：时序路径上被 Coordinator 直接下放的方法逐行看源码（短注释）；其余辅助方法一张表带过。**

## 1. 概览

`SingleTypeKVCacheManager` 是五层架构的**第三层——单类型 KV 缓存管理器**，负责管理**一种具体 Attention/SSM 类型**的 KV Cache 分配、命中查找、释放等逻辑。纯 FullAttention 模型（Llama/Qwen/Mistral）核心用其子类 `FullAttentionManager`，实现**链式哈希前缀缓存**。

**关键数据**：请求↔块映射就存在本层的 `req_to_blocks[request_id]`（`defaultdict[str, list[KVCacheBlock]]`）里，**它不是 `Request` 的字段**——这正是 Scheduler 看到的"请求 block_table"的真正存储位置。

---

## 2. 类继承结构

```
SingleTypeKVCacheManager（ABC 抽象基类）—— 统一接口，子类实现差异部分
├── FullAttentionManager          ← 本文核心：全注意力前缀缓存
│   ├── RSWAManager（Jamba R-SWA）
│   └── SinkFullAttentionManager
├── SlidingWindowManager          ← 滑动窗口注意力
├── ChunkedLocalAttentionManager  ← 块内局部注意力
├── MambaManager                  ← Mamba/SSM
└── CrossAttentionManager         ← encoder-decoder 交叉注意力
```

抽象基类的意义：上层 `KVCacheCoordinator` 用一致接口管理不同类型的组，不关心底层具体类型。

---

## 3. 时序映射（纯 FullAttention 场景）

对应 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) 的端到端流程，`FullAttentionManager` 承担：

| 时序阶段 | 职责 | 对应方法 |
|---|---|---|
| **B1 前缀查找** | 在 `cached_block_hash_to_block` 查最长已计算前缀 | `find_longest_cache_hit`（classmethod） |
| **B2 touch命中块** | 命中块 `ref_cnt+=1`、移出 free 队列，防驱逐 | `add_local_computed_blocks` |
| **B2 算新块数** | 总 token 数 − 已命中块数 → 需新分配块数 | `get_num_blocks_to_allocate` |
| **B2 外部命中** | 为远端/CPU 命中的 token 分配新物理块 | `allocate_external_computed_blocks` |
| **B2 分配新块** | 从 free 队列取块 | `allocate_new_blocks` |
| **B2 写缓存** | 满块哈希写入前缀表 | `cache_blocks` → `block_pool.cache_full_blocks` |
| **E 释放** | 逆序遍历块 `ref_cnt-=1`，0 则回收 | `free` / `pop_blocks_for_free` |

---

## 4. 时序路径核心方法（逐行注释）

### 4.1 `get_num_blocks_to_allocate`：算新块数（`base` 基类）

> 源码 `:144-230`。容量预估，返回值经上层 `kv_cache_manager.py:521` `required>available` 比较决定是否准入。

```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                               total_computed_tokens, num_local_computed_tokens,
                               num_tokens_main_model, apply_admission_cap=False) -> int:
    # ① 按"全序列长度"算总块数（含已命中），是容量预估分母
    num_required_blocks = cdiv(num_tokens, self.block_size)
    if apply_admission_cap and self._max_admission_blocks_per_request is not None:
        num_required_blocks = min(num_required_blocks, self._max_admission_blocks_per_request)
    num_req_blocks = len(self.req_to_blocks.get(request_id, ()))  # 请求已持有的块数

    # ② running 请求快路径：前缀命中只发生在首次 prefill，running 后不再有新增命中
    if request_id in self.num_cached_block:
        assert len(new_computed_blocks) == 0
        return max(num_required_blocks - num_req_blocks, 0)   # draft 被拒时可能为负，兜底 0

    # ③ FullAttention 不跳过 token → get_num_skipped_tokens 恒 0
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
    num_skipped_blocks = num_skipped_tokens // self.block_size

    # ④ 核心：新块 = 总块 − max(跳过的块, 已有归属块)。取较大者是保守估计
    num_new_blocks = max(
        num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0)
    num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

    # ⑤ 命中块里"仍在 free 队列、touch 后会消失"的块，必须计入容量
    num_evictable_blocks = self._get_num_evictable_blocks(
        new_computed_blocks[num_skipped_new_computed_blocks:])

    return num_new_blocks + num_evictable_blocks
```

**辅助函数**：`_get_num_evictable_blocks`（`:128`）= 统计 `ref_cnt==0 且非 null` 的块数；`get_num_skipped_tokens`（`:661`）= 基类恒 0，SWA 子类覆写。

### 4.2 `add_local_computed_blocks`：touch 命中块（`base`）

> 源码 `:232-289`。Coordinator 两阶段协议**第一阶段**，处理**本地前缀命中**的块。

```python
def add_local_computed_blocks(self, request_id, new_computed_blocks,
                              num_local_computed_tokens, num_external_computed_tokens) -> None:
    req_blocks = self.req_to_blocks[request_id]
    assert len(req_blocks) == 0       # 零断言：Coordinator 仅在首次 prefill 调本方法

    # ① 滑窗跳过（FullAttention 下 get_num_skipped_tokens=0 → 不进分支）
    num_total = num_local_computed_tokens + num_external_computed_tokens
    num_skipped_blocks = self.get_num_skipped_tokens(num_total) // self.block_size
    if num_skipped_blocks > 0:
        new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

    # ② touch：命中块就不分配新物理块，只是 ref_cnt+=1 并移出 free 队列（防驱逐）
    if self.enable_caching:
        self.block_pool.touch(new_computed_blocks)
    else:
        assert not any(new_computed_blocks)   # 关闭前缀缓存时不应有命中块

    # ③ 占位 null + 追加命中块，维护 req_to_blocks
    req_blocks.extend([self._null_block] * num_skipped_blocks)
    req_blocks.extend(new_computed_blocks)
    self.num_cached_block[request_id] = len(req_blocks)
```

**要点**：引用计数共享（非复制）是前缀缓存省显存核心；必须所有组都完成本方法后 Coordinator 才逐组调 `allocate_external_computed_blocks`（issue #33775，避免 `get_new_blocks` 驱逐未 touch 的命中块）。

### 4.3 `allocate_external_computed_blocks`：外部命中分配（`base`）

> 源码 `:291-328`。两阶段协议**第二阶段**：外部 connector（CPU offload / remote）KV 在 GPU 无现成物理块，须 `get_new_blocks` 现编新块，后续由 Worker 加载填充。

```python
def allocate_external_computed_blocks(self, request_id,
                                      num_local_computed_tokens,
                                      num_external_computed_tokens) -> None:
    num_total = num_local_computed_tokens + num_external_computed_tokens
    num_skipped = self.get_num_skipped_tokens(num_total)
    if num_skipped > 0:   # SWA 扣掉滑窗跳过（FullAttention 恒 0）
        num_external_computed_tokens = min(num_total - num_skipped,
                                           num_external_computed_tokens)
    if num_external_computed_tokens <= 0:
        return

    # ① 需补的块 = 覆盖全部命中所需块数 − req_blocks 已有的
    req_blocks = self.req_to_blocks[request_id]
    allocated_blocks = self.block_pool.get_new_blocks(
        cdiv(num_total, self.block_size) - len(req_blocks))
    req_blocks.extend(allocated_blocks)

    # ② 新块 ref_cnt=1、在 free 队列外；记入 new_block_ids 供 Worker 清零
    if self._record_new_block_ids:
        self.new_block_ids.extend(b.block_id for b in allocated_blocks)
```

**区别**：本地命中只 `touch` 增引用（GPU 已存在）；外部命中必须 `get_new_blocks` 分配新块并记 ID 清零。

### 4.4 `allocate_new_blocks`：分配新块（`base`）

> 源码 `:329-368`。`allocate_slots` 第三阶段，为未命中 token 补足块。

```python
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model) -> list:
    # ① 需补块数 = 总块数 − 已有；draft 被拒时可能 ≤0
    req_blocks = self.req_to_blocks[request_id]
    num_new_blocks = cdiv(num_tokens, self.block_size) - len(req_blocks)
    if num_new_blocks <= 0:
        return []

    # ② 取新块、追加、条件记录 ID（不需要清零的 backend 可省 kernel）
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)
    if self._record_new_block_ids:
        self.new_block_ids.extend(b.block_id for b in new_blocks)
    return new_blocks
```

### 4.5 `cache_blocks`：缓存写入

基类 `:427-477`；`FullAttentionManager` 覆写 `:779-789`。

```python
# 基类 cache_blocks：只缓存满块，幂等（num_cached_block 起写）
num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
num_full_blocks = num_tokens // self.block_size
if num_cached_blocks >= num_full_blocks:
    return                                            # 幂等：已缓存完
block_mask = self.reachable_block_mask(...)           # SWA/Mamba 覆写，FullAttention 恒 None
self.block_pool.cache_full_blocks(request, self.req_to_blocks[request.request_id],
                                  num_cached_blocks, num_full_blocks,
                                  self.block_size, self.kv_cache_group_id, block_mask)
self.num_cached_block[request.request_id] = num_full_blocks
```

**FullAttention 覆写**：先走基类，再当 `hash_block_size != block_size`（混合模型多粒度）时额外调 `_cache_partial_tail_block` 缓存 prompt 尾块最后一个 hash 边界。**注意**：本文早期草稿的 `maybe_save_new_kv_blocks_to_cache` **该版本源码不存在**，统一由 `cache_blocks` 承担。

### 4.6 `pop_blocks_for_free` / `free`：释放

> 源码 `:500-527`，对应时序 **E** 阶段。

```python
def pop_blocks_for_free(self, request_id) -> list:   # 弹出块列表，不真正归还
    req_blocks = self.req_to_blocks.pop(request_id, [])
    self.num_cached_block.pop(request_id, None)
    return req_blocks

def free(self, request_id) -> None:                  # 完整释放：弹出 → 逆序归还
    self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

**逆序释放**：尾块（多是不完整块）先回 free 队列，下次分配时优先被复用，提高续生成命中率。`free_blocks`（`block_pool.py:719-742`）逻辑见 [`2_block_pool.md`](./2_block_pool.md)：无哈希块放队首（优先复用）、有哈希块放队尾（LRU 保护）。

### 4.7 `find_longest_cache_hit`：最长前缀查找（`FullAttentionManager` classmethod）

> 源码 `:681-777`。**classmethod**，不依赖实例状态，便于 fine-grained / 多 group 复用。返回 `(按组命中块列表, 命中 token 精确长度)`。

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                           block_pool, kv_cache_spec, drop_eagle_block,
                           alignment_tokens, ...) -> tuple[..., int]:
    block_size = kv_cache_spec.block_size * (dcp_world_size if dcp_world_size > 1 else 1)
    # ① 把 Request 哈希从 hash_block_size 粒度对齐到本组 block_size 粒度
    block_hashes = resolve_block_hashes(block_hashes, block_pool.hash_block_size,
                                        block_size, alignment_tokens=alignment_tokens, ...)
    fine_grained = (alignment_tokens < block_size and block_size % alignment_tokens == 0)

    computed_blocks = tuple([] for _ in range(len(kv_cache_group_ids)))
    # ② Phase 1：从开头找最长"满块"连续命中 run
    for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
        cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
        if not cached_block:
            break                      # 链式哈希：miss 之后必然全 miss
        [c.append(cb) for c, cb in zip(computed_blocks, cached_block)]
    hit_length = len(computed_blocks[0]) * block_size

    # ③ 细粒度：从高到低探测第一块内部 hash 边界（续写场景命中落在块内）
    if fine_grained:
        for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
            cached_tail = block_pool.get_cached_block(block_hashes[fine_idx], kv_cache_group_ids)
            if not cached_tail:
                continue
            [c.append(cb) for c, cb in zip(computed_blocks, cached_tail)]
            hit_length = (fine_idx + 1) * alignment_tokens
            break
    # ④ EAGLE 丢最后一块 + 对齐收尾
    if drop_eagle_block and hit_length > 0:
        hit_length -= min(alignment_tokens, block_size)
    hit_length -= hit_length % alignment_tokens
    for computed in computed_blocks:
        del computed[cdiv(hit_length, block_size):]   # 裁剪超长块
    return computed_blocks, hit_length
```

**链式哈希特性**：每个块哈希依赖前一块哈希 → 保证前缀连续性、避免"内容同但前缀不同"的错误命中。细粒度模式下块内每个 hash 边界有独立映射，支持块内部分命中。

---

## 5. 其余方法速览（不在时序主路径，大概讲作用）

| 方法 | 源码 | 作用 | 被谁调 |
|---|---|---|---|
| `get_num_skipped_tokens` | `:661` | 滑窗外 token 数；FullAttention 恒 0 | 各分配方法内部 |
| `remove_skipped_blocks` | `:622` | 释放滑窗外的块并 `null` 占位；FullAttention no-op | `allocate_slots` ① |
| `get_num_common_prefix_blocks` | `:821` | 公共前缀块数（`ref_cnt==len(req_to_blocks)`），调度优先级 | Coordinator → Scheduler |
| `reachable_block_mask` | — | 可命中块掩码（SWA/Mamba 意义） | `cache_blocks` 内部 |
| `take_new_block_ids` | `:376` | drain 需清零的新块 id | KM `take_new_block_ids` |
| `new_step_starts` | — | 新调度步开始，重置状态 | Scheduler 步开始 |
| `supports_fine_grained_hash_lookup` | 类属性 | 是否支持块内细粒度命中 | `find_longest_cache_hit` |

---

## 6. 其他 Manager 简要概述（了解即可）

| 子类 | 源码 | 特点 |
|---|---|---|
| `SlidingWindowManager` | `:878-1093` | 每 token 只见最近 `window_size`；只在块边界查哈希、`get_num_skipped_tokens` 返回窗口外 token 数、`remove_skipped_blocks` 真释放；稀疏前缀保留 |
| `RSWAManager` | `:832-876` | SWA + 全局 token 检索，驱逐中间 gap 块；需 `num_prompt_tokens` 参数 |
| `MambaManager` | `:1253-1745` | SSM 无 K/V，缓存"状态"，`cache_blocks` 复杂；稀疏保留 |
| `ChunkedLocalAttentionManager` | `:1095` | 块内局部注意力（GLM-4v） |
| `CrossAttentionManager` | `:1747` | encoder-decoder 交叉注意力，处理静态 encoder KV |
| `SinkFullAttentionManager` | `:1810` | Sink 注意力，sink block 常驻 |

---

## 7. 设计要点小结（纯 FullAttention 视角）

1. `req_to_blocks` 是请求 block_table 的真正存储位置（非 `Request` 字段）。
2. **链式哈希** + **引用计数共享**：命中块只 `touch` 不复制，最后释放才回收。
3. **LIFO 逆序释放**：尾块先回自由队列，提高续生成命中率。
4. 抽象基类统一接口 → 上层 Coordinator 可一致管理 FullAttention/SWA/Mamba。
5. `find_longest_cache_hit` 是 classmethod，便于 fine-grained / 多 group 复用。