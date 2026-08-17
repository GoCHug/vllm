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

        # ── 部分命中 CoW 相关簿记（仅 FullAttention / Mamba align 填充，其它 manager 永远空）──
        # 部分命中"预约表"：记录需要 CoW 的请求信息。
        # key=request_id，value=(block_idx, source_block)，
        # 其中 block_idx 是尾块在 block_table 中的索引，source_block 是被多请求共享的源块。
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}
        # CoW 复制任务队列：drain 给 Worker，元素 (source_block, cow_block) 表示"把 source 的 KV 拷到 cow"
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        # 外部 KV connector 的部分尾部 offload 交接队列（仅 Mamba align）：元素 (req_id, group_id, block, boundary_tokens)
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

**作用**：计算本轮需要新分配多少物理块，这是分配前的"容量预估"。返回值会传给上层 [`kv_cache_manager.py:521`] 的 `required_blocks > available_blocks` 比较，决定是否触发抢占。

#### 4.2.1 完整源码 + 逐行注释

```python
def get_num_blocks_to_allocate(
    self,
    request_id: str,                     # 请求 ID，用于在 req_to_blocks / num_cached_block 里查状态
    num_tokens: int,                     # "需要槽位的总 token 数"，= total_computed_tokens + num_new_tokens + num_lookahead_tokens
                                         #   注意：含已经计算过的 token（前缀命中的也算），是"全序列长度"，不是"本轮新算的长度"
    new_computed_blocks: Sequence[KVCacheBlock],  # B1 刚查到的"前缀命中块"列表（满块），这些块已有归属，无需重新分配
    total_computed_tokens: int,          # 本地命中 + 外部 connector 命中的总已计算 token 数；用于算滑窗跳过
    num_local_computed_tokens: int,      # 仅本地前缀缓存命中的 token 数（不含 connector），= len(new_computed_blocks) * block_size
    num_tokens_main_model: int,          # 主模型 token 数；非投机解码时 = num_tokens；投机解码时 = num_tokens - num_lookahead_tokens
    apply_admission_cap: bool = False,   # 是否应用"每请求准入上限"（SWA / chunked-local 才用，full-attn 一般 False）
) -> int:
    """
    Get the number of blocks needed to be allocated for the request.
    ...
    """

    # ===== 第 1 步：按"全序列长度"算总块数 =====
    # cdiv 是向上取整除法。这里把"整个序列需要多少块"先算出来，
    # 是容量预估的"分母"。注意此时还没扣掉前缀命中、滑窗跳过等，
    # 是"假设全要新分配"的最大可能值。
    num_required_blocks = cdiv(num_tokens, self.block_size)

    # 准入上限：只对 SWA / chunked-local 这类"回收型"spec 生效。
    # 目的：让"准入阶段承诺的块数"和"启动时 pool sizer 给的额度"对齐，
    # 防止 sum(每个请求的预留) > pool 总量，避免 issue #39734 那种死锁或 mid-prefill OOM。
    # full-attn 模型 _max_admission_blocks_per_request 一般是 None，这一段不触发。
    if apply_admission_cap and self._max_admission_blocks_per_request is not None:
        num_required_blocks = min(
            num_required_blocks, self._max_admission_blocks_per_request
        )

    # req_to_blocks: dict[req_id, list[KVCacheBlock]]，记录该请求"已经持有"的块。
    # defaultdict，请求首次出现时返回空 list，num_req_blocks = 0。
    # 这一步查的是"过去步骤已分给它的块数"，后面要从中扣减。
    num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

    # ===== 第 2 步：running 请求快路径 =====
    # num_cached_block: dict[req_id, int]，只跟踪 RUNNING 请求的已缓存块数。
    # 一个请求一旦进入 running（已经 prefill 过至少一次），后续不会再有新的前缀命中
    # （前缀命中只发生在第一次 prefill 的 B1 阶段）。
    if request_id in self.num_cached_block:
        # 断言：running 请求不应再传新的命中块进来
        assert len(new_computed_blocks) == 0
        # 直接用 "总块数 - 已持有块数"。
        # 投机解码时已持有块可能含被拒的 draft token 对应的块，
        # 此时 num_required_blocks 可能 < num_req_blocks，所以 max(..., 0) 兜底。
        return max(num_required_blocks - num_req_blocks, 0)

    # ===== 第 3 步：滑窗跳过 =====
    # get_num_skipped_tokens：基类默认返回 0（full-attn 不跳过任何 token）。
    # SlidingWindowManager / ChunkedLocalAttentionManager 等子类会覆盖它，
    # 返回"窗口外、不再参与 attention"的 token 数。
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)

    # ===== 第 4 步：算"已有归属"的块数 =====
    # 命中块（new_computed_blocks）+ 已持有块（req_to_blocks）合在一起
    # 都是"不需要新分配"的块。这两者并集而非相加更精确，但这里用相加做保守上界
    # （因为 new_computed_blocks 是本轮新查到的，和 req_to_blocks 一般不重叠）。
    num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks

    # 滑窗跳过的"整块"数（窗口外不完整的尾块不算，整除截断）
    num_skipped_blocks = num_skipped_tokens // self.block_size

    # 核心公式：要新分配的块 = 总块数 - 已有归属块数
    # 这里取 max(num_skipped_blocks, num_local_computed_blocks) 而不是相加，
    # 因为两者可能有重叠（命中的块可能恰好也在窗口外）。
    # 取较大者是保守估计："假设它们不重叠时能省的最大块数"。
    # 注释里说"local-computed blocks inside the window contribute to required capacity;
    # otherwise, skipped blocks dominates"——意思是：
    #   - 滑窗内：用 local_computed_blocks（命中的块在窗口里仍占容量）
    #   - 滑窗外：用 skipped_blocks（窗口外的块直接不算）
    # 取较大者覆盖两种情况。
    num_new_blocks = max(
        num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
        0,
    )

    # ===== 第 5 步：算"前缀命中块里有多少落在滑窗外" =====
    # 前缀命中块 new_computed_blocks 是从头开始的连续块，
    # 前 num_skipped_blocks 块可能在窗口外（窗口往前推时）。
    # 这部分"窗口外的命中块"在后面 _get_num_evictable_blocks 时要排除掉，
    # 因为它们 touch 后是直接被释放的（不在 free 队列里挪位）。
    # num_req_blocks 已在 req_to_blocks 里，先扣掉，剩下的才是
    # "new_computed_blocks 中位于窗口外的部分"。
    num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

    # ===== 第 6 步：驱逐候选块计数 =====
    # _get_num_evictable_blocks：统计 blocks 里 ref_cnt==0 且非 null 的块数。
    # 含义：这些命中块现在还挂在 free 队列里（被人 touch 过又释放过），
    # 所以它们当前仍被算在 block_pool.get_num_free_blocks() 里。
    # 一旦本请求 touch 它们（allocate_new_computed_blocks 时），
    # 它们会从 free 队列里被摘出、ref_cnt++ → free_blocks 减少。
    # 因此这部分块必须计入 required_blocks，否则上游
    # (kv_cache_manager.py:521) required > available 的判断会虚高估可用容量。
    # 切片 [num_skipped_new_computed_blocks:] 是为了排除"在窗口外的命中块"，
    # 因为那些块 touch 后会立刻被 remove_skipped_blocks 释放，不会真正占住 free 队列位。
    num_evictable_blocks = self._get_num_evictable_blocks(
        new_computed_blocks[num_skipped_new_computed_blocks:]
    )

    # ===== 第 7 步：部分命中 CoW 预留 +1 块 =====
    # _has_partial_local_hit：判断前缀命中是否"落在块边界中间"。
    #   条件：有命中块 且 num_local_computed_tokens % block_size != 0
    #   含义：最后一块命中是"半块"（命中了部分 token，但不到块尾）。
    # 这种半块在 allocate_new_blocks 时会触发 Copy-on-Write：
    #   - 原命中块可能正被别的请求引用，不能直接覆盖
    #   - 所以新拿一个空块，把原块内容拷贝过来，再追加新 token
    # 这个 "+1" 就是给 CoW 重定向预留的那个新块。
    if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
        num_new_blocks += 1

    # 最终返回：本轮要新分配的块 + 命中块中驱逐候选块数。
    # 注意 num_new_blocks 是"真要从 free 池拿的新块"，
    # num_evictable_blocks 是"已被命中但要 touch 摘出 free 队列的旧块"，
    # 两者都让 free_blocks 减少，所以一起返回给上游做容量检查。
    return num_new_blocks + num_evictable_blocks
```

#### 4.2.2 三个辅助函数实现

**`_get_num_evictable_blocks`**（`single_type_kv_cache_manager.py:128-130`）

```python
@classmethod
def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
    return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)
```

- `ref_cnt == 0`：当前没人引用，块还在 free 队列里
- `not blk.is_null`：排除 null_block（占位符块，不占真实容量）
- 返回的是"touch 后会从 free 队列消失"的块数

**`_has_partial_local_hit`**（`single_type_kv_cache_manager.py:132-142`）

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

- 命中 token 数对 block_size 取余非 0 → 最后一块没填满 → 是半块命中
- 半块要 CoW 重定向，所以前面 `num_new_blocks += 1`

**`get_num_skipped_tokens`（基类）**（`single_type_kv_cache_manager.py:661-672`）

```python
def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
    """
    Get the number of tokens that will be skipped for attention computation.
    ...
    """
    # The default behavior is to not skip any tokens.
    return 0
```

- FullAttention 默认不跳过任何 token（窗口覆盖全部历史）
- `SlidingWindowManager` 等子类会覆盖此方法，返回窗口外的 token 数

#### 4.2.3 整体逻辑串起来

1. 先按"全序列需要多少块"算 `num_required_blocks`（带前缀命中）
2. running 请求直接走快路径：总块数 - 已持有块数
3. 新请求要从总块数里减掉"已有归属的块"（命中块 + 已持有块）和"滑窗跳过的块"，取较大者保守估计
4. 加上"半块命中"需要的 CoW 预留块
5. 加上"还在 free 队列里的命中块"数（touch 后会从 free 队列消失，必须计入容量检查）

最终返回值就是 [`allocate_slots` L521]里 `required_blocks` 的来源，和 `available_blocks = free - reserved` 比较，决定是否触发抢占。

### 4.3 核心方法：`add_local_computed_blocks`

源码位置：`single_type_kv_cache_manager.py:232-289`

**作用**：处理"本地前缀缓存命中"的块——把 B1 阶段查到的命中块 add 到请求的 block 列表里，并通过 `touch` 锁定引用计数，防止它们被后续分配驱逐。这是 [`allocate_new_computed_blocks` 两阶段 protocol]（coordinator.py:192）的**第一阶段**，必须在 `allocate_external_computed_blocks`（§4.4）之前对所有组执行完毕，否则 connector 的新块 `get_new_blocks` 可能驱逐本组尚未 touch 的命中块（issue #33775）。

#### 4.3.1 完整源码 + 逐行注释

```python
def add_local_computed_blocks(
    self,
    request_id: str,                                          # 请求 ID
    new_computed_blocks: Sequence[KVCacheBlock],              # B1 阶段刚查到的前缀命中块列表（均为满块）
    num_local_computed_tokens: int,                           # 本地前缀命中 token 数 = len(new_computed_blocks) * block_size（满块对齐时）
    num_external_computed_tokens: int,                        # 外部 connector 命中 token 数（本方法不用，只参与滑窗跳过计算）
) -> None:
    """
    Add the locally cached (prefix-hit) blocks to the request:
    1. Touch the computed blocks (paired with adding them to `req_blocks`)
       so their ref_cnt exactly tracks the referencing requests.
    1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
    2. Add the remaining computed blocks.
    """

    # ===== 第 1 步：取出请求当前的 block 列表，并断言为空 =====
    # req_to_blocks: defaultdict[str, list[KVCacheBlock]]，记录请求已持有的块。
    # coordinator（kv_cache_coordinator.py:192）只在"首次分配"（即请求第一次进 scheduler 做 prefill）时
    # 才调用本方法——running 请求已在 coordinator 层被短路（running 不会再有新前缀命中）。
    # 因此此处 req_blocks 必为空：request 还没被分配过任何命中块。
    req_blocks = self.req_to_blocks[request_id]
    assert len(req_blocks) == 0   # 零断言：保证后面的"追加"语义安全

    # ===== 第 2 步：处理滑窗跳过（full-attn 下 noop）=====
    # 滑动窗口/RSWA 会在序列头部丢弃 token（滑出窗口），这些 token 对应的块
    # 不需要参与 attention，但需要在 block_table 里"占位"以保持位置对齐。
    # 对 full attention (基类 FullAttentionManager, :661) get_num_skipped_tokens 恒返回 0，
    # 所以 num_skipped_blocks = 0，下面的 if 分支不进入。
    num_total_computed_tokens = (
        num_local_computed_tokens + num_external_computed_tokens
    )
    num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
    num_skipped_blocks = num_skipped_tokens // self.block_size
    if num_skipped_blocks > 0:
        # SWA 场景：丢弃前 num_skipped_blocks 个命中块——它们虽然在前缀缓存活，
        # 但已经滑出窗口，本请求不再引用。直接切片跳过，后面用 null 占位。
        # 注意：被跳过的块没有 touch，ref_cnt 不增，仍留在 free 队列可被驱逐。
        new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

    # ===== 第 3 步：touch 命中块——锁定引用计数，防止被驱逐 =====
    # block_pool.touch(blocks)（block_pool.py:702）逐块执行：
    #   - 若 block.ref_cnt == 0 且非 null：说明它在 free_block_queue 里（驱逐候选），
    #     先 free_block_queue.remove(block) 把它从空闲队列摘出；
    #   - block.ref_cnt += 1（0→1 或 1→2，多请求共享同一命中块时累加）；
    #   - 这一步是"命中块不分配新物理块，只是增加引用"的核心。
    if self.enable_caching:
        self.block_pool.touch(new_computed_blocks)
    else:
        # 缓存未开启时不应该有命中块——find_longest_cache_hit 在 caching 关闭时返回空，
        # 所以 new_computed_blocks 应该是空列表/空元组。
        assert not any(new_computed_blocks), (
            "Computed blocks should be empty when prefix caching is disabled"
        )

    # ===== 第 4 步：把命中块追加到请求的 block 列表 =====
    # 先用 null_block 填充滑窗跳过的位置（full-attn 下 num_skipped_blocks=0，等价于不追加）。
    # null_block 是一个特殊的哨兵块（is_null=True），不占物理资源，touch/evict 时都会跳过它。
    # 占位的目的：让后续的 "token index → block index" 映射保持连续（block_table[i] 对应第 i 个 block）。
    req_blocks.extend([self._null_block] * num_skipped_blocks)
    # 追加真正命中的块。此时 req_blocks = [null...] * skip + 命中块...，
    # 这些块的 ref_cnt 已在 touch 里 +1，物理内存将由 GPU 复用（不需要清零/重算）。
    req_blocks.extend(new_computed_blocks)

    # ===== 第 5 步：标记已缓存块数，避免 cache_blocks() 重复缓存 =====
    # num_cached_block: dict[req_id, int]，记录"这个请求有多少块已经是缓存命中/已写哈希的"。
    # 后续 cache_blocks()（§4.6）会从 num_cached_block[req_id] 开始往后写哈希，
    # 已标记的块直接跳过（命中块的 block_hash 在最初缓存它们的请求时已写入映射表）。
    self.num_cached_block[request_id] = len(req_blocks)

    # ===== 第 6 步：部分命中检测——尾块落在块内部，需要 CoW =====
    # _has_partial_local_hit（:132）判断命中是否"不整除"：
    #   len(new_computed_blocks) > 0 and num_local_computed_tokens % self.block_size != 0
    # 即本地命中的 token 数不是 block_size 的整数倍 → 最后一个命中块只覆盖了部分 token，
    # 但该块同时还被其他请求引用（ref_cnt >= 2），不能直接往里写新 token（会污染共享数据）。
    if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
        # 记录这对 (block_idx, source_block) 到 _partial_hit_reqs（:116），
        # 供后续 allocate_new_blocks（§4.5）做 CoW（Copy-On-Write）重定向：
        #   - 分配一块全新的 cow_block 替换 req_blocks[block_idx]；
        #   - 把 source_block 的数据拷贝到 cow_block；
        #   - 之后新 token 往 cow_block 写，不影响其他请求读 source_block。
        block_idx = num_local_computed_tokens // self.block_size
        self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])
        # 把 num_cached_block 回退到"满块数"（不含部分命中的尾块）。
        # 原因：尾块即将被 CoW 替换，新 cow_block 是私有块（block_hash=None），
        # 需要等它被写满后由 cache_blocks() 重新写入哈希表，所以不能把它算作"已缓存"。
        # block_idx = num_local_computed_tokens // block_size 正好是"完整命中的满块数"。
        self.num_cached_block[request_id] = block_idx
```

#### 4.3.2 关键设计点

- **引用计数而非复制**：命中块是已有请求写入的物理块，多请求共享。用 `touch`（`ref_cnt++` 并从 free 队列摘出）来"占座"，不分配新物理块，是前缀缓存省显存的核心机制。
- **两阶段 protocol 的第一阶段**：本方法只是 coordinator 两阶段（`add_local_computed_blocks` → `allocate_external_computed_blocks`）的第一阶段，只处理"本地命中"。必须所有组都完成 `add_local_computed` 后，coordinator 才会逐组调 `allocate_external_computed_blocks`，否则跨组的 `get_new_blocks` 可能驱逐尚未 touch 的命中块（issue #33775）。
- **零断言保障**：`assert len(req_blocks) == 0`——coordinator 在 running 请求路径已短路，首次分配时请求不可能已持有块。
- **滑窗占位**：跳过的块用 `null_block` 填充保持 `block_table` 位置对齐，`null_block` 不占物理资源、touch/evict 均跳过。
- **部分命中 → CoW 预约**：当 `num_local_computed_tokens % block_size != 0`（命中落在块内），尾块被多请求共享不能直接写。把 `(block_idx, source_block)` 存入 `_partial_hit_reqs` 预约，由后续 `allocate_new_blocks` 真正执行 CoW 替换，同时把 `num_cached_block` 回退到满块数，由 `cache_blocks` 后续写哈希。

### 4.4 核心方法：`allocate_external_computed_blocks`

源码位置：`single_type_kv_cache_manager.py:291-328`

**作用**：为"外部 connector 命中"（如 CPU offload、remote KV cache）的 token 分配**新的物理块**。这是 [`allocate_new_computed_blocks` 两阶段 protocol]（coordinator.py:192）的**第二阶段**。

与 `add_local_computed_blocks`（§4.3，复用已有命中块）不同：外部 connector 的 KV 数据在远端 / CPU 上，GPU 端**没有现成的物理块可复用**，所以必须 `get_new_blocks` 从空闲池分配新块，后续由 Worker 从远端加载填充。必须在本组及所有组 `add_local_computed_blocks` 完成（即所有命中块已 touch 锁定）之后才调用，避免新块分配驱逐尚未锁定的命中块。

#### 4.4.1 完整源码 + 逐行注释

```python
def allocate_external_computed_blocks(
    self,
    request_id: str,                                # 请求 ID
    num_local_computed_tokens: int,                 # 本地前缀命中 token 数（已完成 touch，本方法不再处理）
    num_external_computed_tokens: int,              # 外部 connector 命中 token 数（本方法要为其分配新块）
) -> None:
    """
    Allocate new blocks for external (KV-connector) computed tokens.

    Must run only after every group's local blocks have been touched via
    `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
    evict another group's cache-hit blocks (issue #33775).
    """
    # 注意：本方法不接收 new_computed_blocks 参数——因为外部命中的块
    # 不存在于 GPU 端，没有现成的 KVCacheBlock 可 touch，需要现编新物理块。

    # ===== 第 1 步：计算滑窗跳过对外部命中 token 数的影响 =====
    # 同 add_local_computed_blocks 的逻辑：滑窗头部跳过的 token 同样
    # 应该从外部命中里扣除（滑出窗口的 token 不需要 KV slot）。
    # 对 full attention (基类 :661) get_num_skipped_tokens 恒返回 0，本段 noop。
    num_total_computed_tokens = (
        num_local_computed_tokens + num_external_computed_tokens
    )
    num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
    if num_skipped_tokens > 0:
        # SWA 场景：扣掉滑窗跳过后，外部命中真正需要分配块的 token 数。
        # 取 min(总数 - 跳过, 原外部数)：如果跳过部分吃掉了本地命中的 token，
        # 这里保守不超出原始 num_external_computed_tokens。
        num_external_computed_tokens = min(
            num_total_computed_tokens - num_skipped_tokens,
            num_external_computed_tokens,
        )
    # 外部命中扣减跳过后若 ≤ 0（极端情况：滑窗跳过 ≥ 总命中），直接返回，不分块。
    if num_external_computed_tokens <= 0:
        return

    # ===== 第 2 步：计算还需要分配多少新块 =====
    # req_blocks 已在 §4.3 add_local_computed_blocks 里被追加过命中块（含 null 占位），
    # 它现在覆盖了 num_local_computed_tokens 这么多 token（满块部分）。
    # 要让 block_table 覆盖"本地命中 + 外部命中"全部 token，需要的总块数：
    #   cdiv(num_total_computed_tokens, block_size)
    # 扣掉已有的 len(req_blocks) 个块，剩下的就是要新分配的块数。
    # 举例：block_size=16，local=32(2块) + external=38(2.375→3块去掉2块已占≈需补3块)，
    #   total=70, cdiv(70,16)=5, req_blocks 已有 2 块 → 新分配 3 块。
    req_blocks = self.req_to_blocks[request_id]
    allocated_blocks = self.block_pool.get_new_blocks(
        cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
    )

    # ===== 第 3 步：把新块追加进请求的 block 列表 =====
    # 这些块是新鲜块（block_hash=None）, ref_cnt=1（get_new_blocks 内置赋值，
    # 见 block_pool.py:668），不在 free_block_queue 里（popleft 取出时已摘除）。
    # 它们的物理内容目前是"脏的/未填充"，后续由 Worker 从外部 connector 加载真实 KV 填入。
    # 注意：这里没有 touch（因为是新块，没有 ref_cnt=0 → +1 的共享语义），也没有写哈希表
    # （哈希在 cache_blocks 阶段才写——但外部命中块是否写哈希取决于 connector 策略，
    # 多数 connector 实现会跳过缓存，由 connector 自己管理，避免污染本地 hash 索引）。
    req_blocks.extend(allocated_blocks)

    # ===== 第 4 步：记录新块 ID，供 Worker 清零 =====
    # _record_new_block_ids（:86）：构造时根据 needs_kv_cache_zeroing 和 kv_cache_spec 类型决定，
    # 多数 attention 类型需要记。记下来的 block_id 后续由 take_new_block_ids()（:411）
    # drain 给 scheduler，在 SchedulerOutput 里下发给 Worker，Worker 对这些块做清零。
    # 外部命中的块虽然将被外部 connector 填充，但 Worker 不知道哪些块会被填——保守起见
    # 仍然清零，避免"未填充 + 残留旧数据"混合导致 attention 计算出错。
    if self._record_new_block_ids:
        self.new_block_ids.extend(b.block_id for b in allocated_blocks)
```

#### 4.4.2 关键设计点

- **与 `add_local_computed_blocks` 的根本区别**：本地命中块在 GPU 上已存在，只需 `touch` 增引用（ref_cnt 1→2 等共享）；外部命中块在远端 / CPU，GPU 端无现成物理块，必须 `get_new_blocks` 分配新块，后续由 Worker 从外部加载填充。
- **两阶段 ordering 的必要性**：本方法会调用 `get_new_blocks`，而 `get_new_blocks` 在开启缓存时会 `_maybe_evict_cached_block` 驱逐 free 队列尾部的缓存块（block_pool.py:666）。如果某组的命中块还没 touch（还在 free 队列里），就可能被这一步误驱逐。所以 coordinator 强制所有组先做完 `add_local_computed_blocks`（把命中块 touch 摘出 free 队列），再逐组 `allocate_external_computed_blocks`，这是 issue #33775 的修复。
- **新块哈希不写表**：本方法只分配新块和追加，不调用 `cache_full_blocks`/`_insert_block_hash`。外部命中块的哈希管理归 connector（多数跳过本地缓存，避免外部 hash 污染本地索引）；这些块的哈希写入时机由后续 `cache_blocks`（§4.6）统一处理——但 connector 命中的块在 `cache_blocks` 里一般也被排除在哈希写入之外。
- **新块 ID 记录用于清零**：分配的新块物理内容是脏的（上一任请求残留），记入 `new_block_ids` 交给 Worker 清零。即使外部 connector 会填充，Worker 仍保守清零，保证 attention 计算安全。

### 4.5 核心方法：`allocate_new_blocks`

源码位置：`single_type_kv_cache_manager.py:329-368`

**作用**：本方法是 `allocate_slots` 流程的**第三阶段**（继 `allocate_new_computed_blocks` 处理完命中块之后），为请求中"未命中缓存"的 token 分配**新的物理块**，让请求最终持有覆盖 `num_tokens` 个 token 槽位的 block_table。同时把"部分命中"场景下需要 CoW（Copy-on-Write）的共享尾块替换为私有副本。

调用链：`KVCacheManager.allocate_slots`（kv_cache_manager.py:541）→ `KVCacheCoordinator.allocate_new_blocks`（kv_cache_coordinator.py:238-271，遍历所有 single_type_manager）→ 本方法。

#### 4.5.1 完整源码 + 逐行注释

```python
def allocate_new_blocks(
    self,
    request_id: str,                                # 请求 ID
    num_tokens: int,                                # 需要槽位的总 token 数（含已分配命中 token + spec decode lookahead）
    num_tokens_main_model: int,                     # 主模型 token 数（无 spec decode = num_tokens；
                                                    # 有 spec decode = num_tokens - num_lookahead_tokens）
                                                    # 注意：基类不使用此参数，仅 MambaManager "align" 覆写用
) -> list[KVCacheBlock]:
    """为请求分配新块，使其至少有 num_tokens 个 token 槽位"""
    # ===== 第 1 步：处理部分命中 CoW 重定向 =====
    # 若尾块被多请求共享（部分命中场景），替换为私有 CoW 副本，避免写穿破坏其它请求 KV
    cow_blocks: list[KVCacheBlock] = []
    if request_id in self._partial_hit_reqs:
        # 取出预约：(block_idx, source_block) = 尾块下标 + 被共享的原块
        block_idx, source_block = self._partial_hit_reqs.pop(request_id)
        # 取 1 个新块作为 CoW 副本（容量早在 get_num_blocks_to_allocate 里 +1 预约过）
        cow_block = self.block_pool.get_new_blocks(1)[0]
        # _apply_cow: 原地替换 req_blocks[block_idx]=cow_block + 记录待 Worker 拷贝任务 + cow_block.ref_cnt += 1
        self._apply_cow(request_id, block_idx, source_block, cow_block)
        # 无条件记入 new_block_ids：CoW 块拷贝前必须清零，避免部分拷贝残留脏数据
        self.new_block_ids.append(cow_block.block_id)
        cow_blocks.append(cow_block)

    # ===== 第 2 步：计算还需新分配多少块 =====
    # 总块数 = cdiv(num_tokens, block_size)，扣掉已有的（命中块 / 外部块 / CoW 尾块），剩下就是新块数
    req_blocks = self.req_to_blocks[request_id]
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_new_blocks = num_required_blocks - len(req_blocks)
    # spec decode draft token 被拒绝时 num_new_blocks 可能 < 0，直接返回
    if num_new_blocks <= 0:
        return cow_blocks
    else:
        # ===== 第 3 步：取新块、追加、记录 ID =====
        # 容量检查在 caller (kv_cache_manager.py:522-526) 已做，这里块不够会抛异常（precondition 违反）
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        # 条件记录：_record_new_block_ids 决定是否需要 Worker 清零（多数 attention 后端需要）
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks)
        # 返回 [cow_block?, ...new_blocks]：CoW 块在前（替换中间位置），新块在后（追加末尾）
        return cow_blocks + new_blocks
```

#### 4.5.2 关键设计点

- **流程定位**：本方法是 `allocate_slots` 的第三阶段，在 `add_local_computed_blocks`（touch 本地命中块）+ `allocate_external_computed_blocks`（为外部命中分配新块）之后调用，把 block_table 从"覆盖命中 token"扩展到"覆盖全部 num_tokens"。
- **部分命中 CoW 延迟执行**：`add_local_computed_blocks`（§4.3）只把 `(block_idx, source_block)` 预约到 `_partial_hit_reqs`；真正取新块做 CoW 延迟到本方法——因为必须等容量检查（`get_num_blocks_to_allocate`，:226-229 为 CoW 块 +1 预约）通过后才动手。
- **CoW 块 ID 无条件记录 vs 新块 ID 条件记录**：`cow_block` 的 ID 直接 `append`（拷贝前必须清零），新块的 ID 走 `_record_new_block_ids` 条件分支（不需要清零的 backend 可省 kernel）。
- **`num_tokens_main_model` 基类未使用**：基类只用 `num_tokens`（含 lookahead）算 `cdiv`。该参数给子类 `MambaManager` "align" 覆写用（mamba 不为 draft token 预留块，:1550 把 `num_tokens = num_tokens_main_model` 抹掉 lookahead）。基类保留是为了让 coordinator 用统一签名遍历调用所有 manager。
- **不做容量检查**：caller `KVCacheManager.allocate_slots`（kv_cache_manager.py:509-526）已用 `get_num_blocks_to_allocate` + 空闲块/watermark 提前检查；若不够返回 `None` 触发抢占，不到本方法。

### 4.5.3 端到端分配流程串讲（对应总览阶段3）

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

    # ③ 计算需要新分配多少块（含部分命中 CoW 块的 +1 预约）
    num_new_blocks = self.managers[0].get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_blocks,
        total_computed_tokens=...,
        num_local_computed_tokens=...,
        num_tokens_main_model=...,
    )

    # ④ 为外部 connector 命中分配新块（两阶段 protocol 的第二阶段，可选）
    self.managers[0].allocate_external_computed_blocks(
        request.request_id,
        num_local_computed_tokens, num_external_computed_tokens,
    )

    # ⑤ 分配新块（含部分命中 CoW 重定向）
    new_blocks = self.managers[0].allocate_new_blocks(
        request.request_id, num_tokens_need_slot, num_tokens_main_model
    )
```

**关键点**：`req_to_blocks[request_id]` 就是在这里一步步维护起来的——它的内容就是 [历史块..., 本次命中块..., (CoW 替换的尾块), 本次新分配块...]，顺序就是 block_table 在 GPU 上的顺序。

### 4.6 核心方法：`cache_blocks`（缓存写入）

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

### 4.7 核心方法：`pop_blocks_for_free` / `free`（释放流程）

源码位置：`single_type_kv_cache_manager.py:500-527`，对应总览阶段 6 的释放流程。

#### 4.7.1 `pop_blocks_for_free`：取出请求的块列表（不真正归还）

```python
def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
    """从 req_to_blocks 中弹出该请求的块列表，清理相关统计，但不归还 BlockPool。
    调用方拿到块列表后，后续负责调用 free_blocks 归还。"""
    req_blocks = self.req_to_blocks.pop(request_id, [])  # 弹出块列表，移除映射
    self.num_cached_block.pop(request_id, None)          # 清理缓存统计
    self._partial_hit_reqs.pop(request_id, None)         # 清理部分命中记录
    return req_blocks                                    # 返回按分配顺序排列的块列表
```

#### 4.7.2 `free`：完整释放单个请求的所有块（最常用）

```python
def free(self, request_id: str) -> None:
    """释放请求的所有块：弹出块列表 → 逆序归还 BlockPool"""
    # 关键：reversed() 逆序释放——尾块先归还，利用 free_block_queue 的 LIFO 特性
    # 这样请求被抢占后重新调度时，原来的尾块会最先被分配回来，提高续生成命中率
    self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))
```

#### 4.7.3 `free_blocks`（BlockPool 方法）：引用计数减一+队列回收

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