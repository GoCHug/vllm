# KVCacheManager 源码详解

源码位置：`vllm/v1/core/kv_cache_manager.py`

---

## 一、是什么

`KVCacheManager` 是 vLLM v1 调度器中 **Scheduler 与 KV Cache 管理子系统交互的唯一门面（Facade）**，位于 KV Cache 五层架构的第五层（最顶层）。

整个 KV Cache 管理体系包含多个层次：物理显存层 → `BlockPool`（逻辑块池）→ `SingleTypeKVCacheManager`（单类型管理层）→ `KVCacheCoordinator`（跨组协调）→ **`KVCacheManager`（顶层接口，本文）** → Scheduler。下层的复杂性（多 group、不同注意力类型的保留策略、CoW 拷贝、跨组驱逐竞态等）全部被 `KVCacheManager` 封装，Scheduler 只需要通过它暴露的十几个方法完成 KV cache 的完整生命周期管理。

文件中还定义了 `KVCacheBlocks`——这是 `KVCacheManager` 与 Scheduler 之间**唯一的数据交换协议**，一个轻量不可变 dataclass，封装了按 group 组织的 `KVCacheBlock` 序列。Scheduler 只通过它获取整数 `block_id`，不直接接触底层 `KVCacheBlock` 对象，保证内部重构不会影响上层调度代码。

---

## 二、干什么用

### 2.1 在整体架构中的位置

KV Cache 管理分五层：物理显存层 → `BlockPool`（逻辑块池）→ `SingleTypeKVCacheManager`（单类型管理层）→ `KVCacheCoordinator`（跨组协调）→ **`KVCacheManager`（顶层接口，本文）** → Scheduler。

```
┌─────────────────────────────────────────────────────────┐
│  Scheduler (调度器)                                       │
├─────────────────────────────────────────────────────────┤
│  KVCacheManager          ← 本文：Scheduler唯一入口        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  KVCacheBlocks  ← Scheduler与Manager唯一数据协议   │  │
│  └───────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  KVCacheCoordinator      ← 跨组对齐命中结果                │
│  ┌──────────────────┬──────────────────┬──────────────┐ │
│  │ FullAttention    │ SlidingWindow    │ Mamba        │ │
│  │ Manager          │ Manager          │ Manager      │ │
│  │ (group 0)        │ (group 1)        │ (group 2)    │ │
│  └──────────────────┴──────────────────┴──────────────┘ │
│             所有manager共用同一个 BlockPool               │
├─────────────────────────────────────────────────────────┤
│  BlockPool               ← 逻辑块分配/释放/哈希表/LRU     │
├─────────────────────────────────────────────────────────┤
│  GPUModelRunner.kv_caches ← torch.Tensor物理显存         │
└─────────────────────────────────────────────────────────┘
```

`KVCacheManager` 直接被 Scheduler 调用，它自己不实现具体的分配/命中/释放逻辑，而是把所有请求转发给内部持有的 `KVCacheCoordinator`（通过工厂函数 `get_kv_cache_coordinator()` 根据配置自动选择 `HybridKVCacheCoordinator` / `UnitaryKVCacheCoordinator` / `KVCacheCoordinatorNoPrefixCache`）和共享 `BlockPool`。`block_pool` 是从 `coordinator` 引用的，不是自己创建的，保证整个模型只有一份全局块池。

**Manager 做什么？** 它不做跨组协调（那是 Coordinator 的职责），也不做单类型管理（那是 SingleTypeManager 的职责），它专注于：
1. **门面封装**：把下层复杂 API 包装成 Scheduler 易用的接口，隐藏多 group 细节
2. **前缀缓存查找入口**：处理"留一重算"（最后一个 token 必须重算取 logits）、`skip_reading_prefix_cache` 跳过条件、KV Connector 命中分歧等边界情况
3. **分配流程编排**：`allocate_slots()` 实现完整的三阶段分配流程（admission gate → 两阶段 touch+allocate → 缓存写入），包括 watermark、reserved_blocks、full_sequence_must_fit 等准入控制
4. **GPU 任务聚合**：每 step drain 各 manager 的 new_block_ids、CoW copies、partial-tail offloads 聚合后发给 Worker
5. **事件标注与统计**：为 BlockPool 发射的事件标注 spec 元数据，收集前缀缓存命中率统计
6. **资源钉住管理**：维护 `_partial_tail_pins`，保证 KV Connector 读取的 off-table block 不被驱逐

### 2.2 核心职责（结合调度流程）

一个请求从进入调度器到完成生成，Scheduler 会按以下顺序调用 KVCacheManager 的方法：

| 调度阶段 | 调用方法 | Manager 的作用 |
|---------|---------|------|
| **0. 步初始化** | `new_step_starts()` | 通知 Coordinator 清空前一步的临时数据（new_block_ids、CoW copies、partial-tail offloads 等） |
| **1. 前缀缓存查找** | `get_computed_blocks()` | 跳过禁用/跳过缓存的请求；设置 `max_cache_hit_length = num_tokens - 1` 留一重算；委托 Coordinator 执行跨组命中查找；在 `kv_cache_report_mode=="full"` 时补发 BlockStored 事件；计算 `shared_prefix_boundary` |
| **2. 核心分配** | `allocate_slots()` | **最核心方法**：admission gate（full_sequence_must_fit + watermark + reserved_blocks）→ remove_skipped_blocks 清理窗口外块 → 两阶段分配（先 touch 命中块抬 ref_cnt 防跨组驱逐，再取新块+CoW）→ 以 `request.num_tokens` 为上限只缓存 verified token；空间不足返回 None 表示无法调度 |
| **3. 获取 Worker 任务** | `take_new_block_ids()` <br> `take_kv_cache_block_copies()` <br> `take_partial_tail_offloads()` <br> `take_events()` | 聚合所有 manager 的新块 ID（Worker zeroing）、CoW GPU 拷贝对、KV Connector partial-tail offload 任务、KV cache 事件（标注 spec 元数据） |
| **4. 缓存写入** | `cache_blocks()` | GPU 计算完成后，委托 Coordinator 把写满的 block 计算 hash 写入前缀缓存 |
| **5. 运行中查询** | `get_blocks()` / `get_block_ids()` <br> `get_num_common_prefix_blocks()` <br> `estimate_cached_tokens()` <br> `usage` (property) | 获取请求 block 列表；计算 Cascade Attention 公共前缀；估算已缓存 token 数；查询 KV cache 使用率 |
| **6. 请求结束/抢占** | `free()` <br> `pop_blocks_for_free()` <br> `remove_skipped_blocks()` | 正常结束：逆序释放所有 block + 解钉 partial-tail pins；抢占：弹出块列表延迟释放；每步清理滑动窗口外不需要的 block |
| **特殊控制** | `reset_prefix_cache()` <br> `evict_blocks()` <br> `get_computed_blocks_for_connector()` | RLHF 权重更新后重置缓存；按 ID 驱逐指定块；带 KV Connector 的混合模型前缀查找（处理 Mamba/FullAttention 命中分歧） |

### 2.3 实际场景举例

**场景：Llama 同构模型 decode（单 group）**

Llama 只有 Full Attention 一种层类型，工厂创建 `UnitaryKVCacheCoordinator`。
1. Scheduler 处理 running 队列，每个请求调用 `allocate_slots(request, num_new_tokens=1, has_scheduled_reqs=...)`
2. Manager 转调 UnitaryCoordinator，Coordinator 直接委派给唯一的 FullAttentionManager，不需要迭代收敛
3. 分配 1 个新 block slot（或复用未满的尾块）
4. Scheduler 调用 `take_new_block_ids()` 获取新块 ID，发给 Worker 执行 zeroing
5. Worker 完成 decode 计算后，`cache_blocks(request, num_computed_tokens)` 把填满的 block 写入前缀缓存
6. 全程无 CoW、无跨组协调开销，开销最小

**场景：RAG 多用户并发（Gemma3 混合模型）**

Gemma3 同时有 Full Attention 层和 Sliding Window 层，工厂创建 `HybridKVCacheCoordinator`。
1. 新请求进入 waiting 队列，`get_computed_blocks()` 调用 `HybridCoordinator.find_longest_cache_hit()` 执行迭代不动点算法，找到 FullAttention 与 SWA 共同命中的前缀长度
2. `allocate_slots()` 中：
   - `full_sequence_must_fit=True`，先做 admission gate 检查整条序列能否放下
   - watermark 生效（WAITING 状态 + has_scheduled_reqs=True），预留空闲块防频繁抢占
   - 两阶段分配：先 `allocate_new_computed_blocks()` touch 所有命中块抬升 ref_cnt，再 `allocate_new_blocks()` 从 free queue 取新块——不会出现"group 0 分配时驱逐了 group 1 还没 touch 的命中块"（issue #33775）
3. `num_tokens_to_cache = min(total_computed + new_tokens, request.num_tokens)`，排除 EAGLE 未验证的 draft token

**场景：EAGLE 投机解码**

1. `allocate_slots()` 接收 `num_lookahead_tokens=N`，为 draft token 预分配 N 个 lookahead block
2. Worker 执行 spec decode，产生 N 个 draft token 的 KV
3. 验证后 accepted M 个 token，rejected (N-M) 个
4. `cache_blocks()` 以 `request.num_tokens` 为上限（此时 `request.num_tokens` 只包含 verified token），rejected draft token 的 block 不写入前缀缓存
5. 被拒绝的 lookahead block 后续会被覆盖或释放，不污染前缀缓存

**场景：禁用前缀缓存（`--enable-prefix-caching=False`）**

1. 工厂创建 `KVCacheCoordinatorNoPrefixCache`
2. `get_computed_blocks()` 中 `prefix_cache_lookup_enabled()` 返回 False，直接返回空 blocks
3. `allocate_slots()` 中跳过前缀命中处理，所有 block 走新分配路径
4. `cache_blocks()` 中 `enable_caching=False`，不写入前缀缓存哈希表
5. 没有哈希查找、没有 CoW、没有 ref_cnt 共享，行为最简

---

## 三、文件结构

`kv_cache_manager.py` 包含两个顶层类，无子类继承：

```
kv_cache_manager.py
├── KVCacheBlocks (dataclass)          ← Scheduler接口数据协议
│   ├── blocks: tuple[Sequence[KVCacheBlock], ...]
│   ├── __add__()                      按group拼接两段分配结果
│   ├── get_block_ids()                转为整数block_id列表
│   ├── get_unhashed_block_ids()       单group未hash块ID（zeroing用）
│   ├── get_unhashed_block_ids_all_groups()  多group版本
│   └── new_empty()                    构造同结构空对象
│
└── KVCacheManager                     ← 核心门面类
    ├── __init__()                     构造：创建coordinator、预分配空KVCacheBlocks
    ├── usage (property)               KV cache使用率查询
    ├── 前缀查找
    │   ├── get_computed_blocks()
    │   ├── get_computed_blocks_for_connector()
    │   ├── prefix_cache_lookup_enabled()
    │   ├── record_prefix_cache_stats()
    │   └── make_prefix_cache_stats()
    ├── 核心分配
    │   └── allocate_slots()           三阶段分配（本文重点）
    ├── 释放/清理
    │   ├── free()
    │   ├── pop_blocks_for_free()
    │   └── remove_skipped_blocks()
    ├── GPU准备数据drain
    │   ├── take_new_block_ids()
    │   ├── take_kv_cache_block_copies()
    │   ├── take_partial_tail_offloads()
    │   ├── get_zeroing_block_ids_in_range()
    │   └── record_blocks_for_zeroing()
    ├── 查询/事件
    │   ├── get_blocks() / get_block_ids()
    │   ├── get_block_ids_for_computed_tokens()
    │   ├── get_num_common_prefix_blocks()
    │   ├── estimate_cached_tokens()
    │   └── take_events()
    └── 生命周期
        ├── cache_blocks()
        ├── create_kv_cache_blocks()
        ├── truncate_computed_blocks()
        ├── new_step_starts()
        ├── reset_prefix_cache()
        └── evict_blocks()
```

---

## 四、KVCacheBlocks 详解

**源码位置**：`kv_cache_manager.py:32-115`

### 4.1 类定义与字段

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j]: i = kv_cache_group 索引, j = 该group内第j个逻辑block
    # 不用block做外维是因为未来不同group可能有不同block_size，block数不一定相同
    # tuple外层保证不可变，便于安全共享和对象复用
```

字段语义：
- **第一维 `i`**：第 `i` 个 KV cache group。一个模型可能包含多个 group（encoder-decoder 有 cross-attention group、混合模型有 Full/SWA/Mamba 等多个 group）。
- **第二维 `j`**：该 group 内第 `j` 个**逻辑 block**，按 token 序列顺序排列，每个逻辑 block 对应 `block_size` 个连续 token 的 KV。末尾可能是未满的 partial block。

例如 `block_size=16`、序列长度 50 时：
```
token index:  [0..15]  [16..31]  [32..47]  [48..49]
blocks[i][j]:   j=0       j=1       j=2       j=3 (partial)
```

### 4.2 方法逐个详解

#### 4.2.1 `__add__`：拼接两段分配结果（55-62行）

用于把前缀缓存命中的 blocks 和新分配的 blocks 合并成完整请求序列。

**执行流程**：
1. 遍历两个 `KVCacheBlocks` 的对应 group
2. 用 `itertools.chain` 把同一 group 的两个 block 序列拼接
3. 包装为新的 `KVCacheBlocks` 返回

```python
def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
    return KVCacheBlocks(
        tuple(
            list(itertools.chain(blk1, blk2))   # 按group拼接两个block序列
            for blk1, blk2 in zip(self.blocks, other.blocks)
        )
    )
```

典型用法：Scheduler 处理 waiting 请求时，把 `get_computed_blocks()` 返回的命中块与 `allocate_slots()` 返回的新分配块拼接，得到完整的 block 序列。`itertools.chain` 避免手动循环，`zip` 保证两个对象的 group 数量一致。

#### 4.2.2 `get_block_ids`：转为整数 block_id（64-91行）

Scheduler 拿到 `KVCacheBlocks` 后，调用此方法获取整数 ID 列表发给 Worker。

**执行流程**：
1. 如果 `allow_none=True` 且所有 group 为空，返回 `None`（上层无需发 zeroing）
2. 否则把每个 `KVCacheBlock` 转为其 `block_id` 整数
3. 返回 `tuple[list[int], ...]`，外层对应 group，内层是该 group 的 block_id 序列

```python
def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
    if allow_none and all(len(group) == 0 for group in self.blocks):
        return None                                # 所有group都空→返回None
    return tuple([blk.block_id for blk in group] for group in self.blocks)
```

`@overload` 在类型层面区分 `allow_none=True/False` 的返回类型，避免类型检查器报 `Optional` 错误。

#### 4.2.3 `get_unhashed_block_ids` / `get_unhashed_block_ids_all_groups`：获取未缓存块 ID（93-108行）

找出尚未写入前缀缓存的 block（`block_hash is None`），这些 block 在 GPU 计算前需要 zeroing 防止旧值干扰。

```python
def get_unhashed_block_ids(self) -> list[int]:
    assert len(self.blocks) == 1, "Only one group is supported"  # 单group场景使用
    return [block.block_id for block in self.blocks[0] if block.block_hash is None]

def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
    return [
        [
            block.block_id
            for block in group
            if block.block_hash is None and not block.is_null   # 跳过null padding block
        ]
        for group in self.blocks
    ]
```

- 单 group 版本有 `assert` 守卫，返回扁平 `list[int]`
- 多 group 版本按 group 返回 `list[list[int]]`，额外跳过 `is_null` 的占位/padding block（null block 不需 zeroing）

#### 4.2.4 `new_empty`：构造同结构空对象（110-114行）

```python
def new_empty(self) -> "KVCacheBlocks":
    return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))
```

构造一个 group 数量相同、但每个 group 为空的 `KVCacheBlocks`。Manager 预计算了全局单例 `empty_kv_cache_blocks`（`kv_cache_manager.py:185-187`），所有无 block 请求复用它，避免反复创建对象触发 GC。

---

## 五、KVCacheManager 详解

**源码位置**：`kv_cache_manager.py:117-878`

### 5.1 构造函数 `__init__`（117-191行）

#### 5.1.1 基础配置参数

```python
self.max_model_len = max_model_len
# When unset, fall back to `max_model_len` so the recycling-aware cap
# collapses to the prior (uncapped) admission behavior.
if max_in_flight_tokens is None:
    max_in_flight_tokens = max_model_len             # 未设置时回退，保持向后兼容

self.enable_caching = enable_caching                # 前缀缓存全局开关
self.enable_kv_cache_events = enable_kv_cache_events  # 是否发射KV cache事件供外部消费
self.use_eagle = use_eagle                          # EAGLE投机解码标志
self.log_stats = log_stats                          # 是否记录前缀缓存命中率统计
self.metrics_collector = metrics_collector
self.prefix_cache_stats = PrefixCacheStats() if log_stats else None  # 命中率统计器
```

- `max_in_flight_tokens`：recycling-aware 准入控制上限，Scheduler 运行时传入真实值，防止超长 prompt 耗尽 KV cache
- `enable_caching=False` 时走最简路径：不查哈希、不做 CoW、不缓存块，适合 benchmark 或特殊场景

#### 5.1.2 Coordinator 与 BlockPool

```python
# ===== 通过工厂函数创建coordinator（自动选择Hybrid/Unitary/NoPrefixCache）=====
self.coordinator = get_kv_cache_coordinator(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    max_in_flight_tokens=max_in_flight_tokens,
    use_eagle=self.use_eagle,
    enable_caching=self.enable_caching,
    enable_kv_cache_events=enable_kv_cache_events,
    dcp_world_size=dcp_world_size,
    pcp_world_size=pcp_world_size,
    scheduler_block_size=scheduler_block_size,
    hash_block_size=hash_block_size,
    metrics_collector=self.metrics_collector,
)
self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
self.block_pool = self.coordinator.block_pool        # 引用共享BlockPool（不自建，保证全局唯一）
self.kv_cache_config = kv_cache_config
```

`get_kv_cache_coordinator` 工厂根据配置自动选择：
- 禁用前缀缓存 → `KVCacheCoordinatorNoPrefixCache`
- 单 group（同构模型）→ `UnitaryKVCacheCoordinator`
- 多 group（混合模型）→ `HybridKVCacheCoordinator`

`block_pool` 从 coordinator 引用而非自己 new，保证整个模型只有一份全局块池。

#### 5.1.3 Watermark 与事件元数据

```python
# Watermark: 准入时保留的最小空闲块数，防止频繁抢占
assert watermark >= 0.0, "watermark must be non-negative"
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
# 每个group的(spec_kind, sliding_window)元数据，用于标注KV cache事件
self.kv_cache_event_metadata = tuple(
    (
        get_kv_cache_spec_kind(group.kv_cache_spec).value,
        get_kv_cache_spec_sliding_window(group.kv_cache_spec),
    )
    for group in kv_cache_config.kv_cache_groups
)
```

- `watermark_blocks` = `watermark` 比例 × 总 block 数，仅对 WAITING/PREEMPTED 请求准入时生效
- `kv_cache_event_metadata`：BlockPool 发射 `BlockStored` 事件时不带语义元数据（松耦合），由 Manager 在 `take_events()` 时标注 spec_kind 和 sliding_window

#### 5.1.4 空对象单例与 partial-tail pins

```python
# 预构造空KVCacheBlocks单例，复用以避免GC开销
# 使用嵌套tuple保证不可变
self.empty_kv_cache_blocks = KVCacheBlocks(
    tuple(() for _ in range(self.num_kv_cache_groups))
)

# KV connector partial-tail offload钉住的off-table block，请求free时解钉
# key=request_id, value=被钉住的block列表
self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}
```

- `empty_kv_cache_blocks`：预构造的不可变空对象，`create_kv_cache_blocks()` 对空 blocks 直接返回它，避免反复创建对象
- `_partial_tail_pins`：KV Connector 场景专用。Mamba align group 把 off-table 的 partial tail block hand-off 给 connector 读取，这些 block 不在请求 block 表中，必须通过 `block_pool.touch()` 钉住（ref_cnt++），记录在此 dict 中，请求 `free()` 时才解钉归还块池

### 5.2 工具属性与统计方法

```python
@property
def usage(self) -> float:                              # kv_cache_manager.py:193-200
    return self.block_pool.get_usage()                # KV cache使用率(0.0-1.0)，委托block_pool

def make_prefix_cache_stats(self) -> PrefixCacheStats | None:  # kv_cache_manager.py:202-212
    if not self.log_stats:
        return None
    stats = self.prefix_cache_stats
    self.prefix_cache_stats = PrefixCacheStats()       # 取出并重置，供定期上报
    return stats

def prefix_cache_lookup_enabled(self, request: Request) -> bool:  # kv_cache_manager.py:214-216
    return self.enable_caching and not request.skip_reading_prefix_cache
    # enable_caching全局开关 AND 请求未标记跳过（prompt logprobs/pooling模型会跳过）

def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:  # kv_cache_manager.py:218-227
    if not self.log_stats or not self.prefix_cache_lookup_enabled(request):
        return                                         # 跳过缓存查找的请求不统计
    assert self.prefix_cache_stats is not None
    self.prefix_cache_stats.record(
        num_tokens=request.num_tokens,
        num_hits=num_hits,
        preempted=request.num_preemptions > 0,         # 被抢占过的请求不计入命中率
    )
```

### 5.3 前缀缓存查找方法

#### 5.3.1 `get_computed_blocks`：主前缀查找入口（229-295行）

waiting 队列新请求调度前调用，查找前缀缓存命中。返回 `(blocks, num_new_computed_tokens, shared_prefix_boundary)`。

**执行流程**：
1. 检查 `prefix_cache_lookup_enabled()`，禁用/跳过则返回空
2. 设置 `max_cache_hit_length = request.num_tokens - 1`（留一重算：所有 token 全命中时最后一个 token 也必须重算取 logits）
3. 委托 `coordinator.find_longest_cache_hit()` 执行跨 group 前缀查找（Hybrid 下是迭代不动点算法），得到 `(computed_blocks, num_new_computed_tokens, num_uncached)`
4. 如果 `kv_cache_report_mode == "full"`，为命中块补发 `BlockStored` 事件（供 gateway 等外部消费者感知）
5. 计算 `shared_prefix_boundary = num_new_computed_tokens + num_uncached`（稀疏保留组的交汇点，钉住防止 retention_interval 驱逐）
6. 包装为 `KVCacheBlocks` 返回

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0

    max_cache_hit_length = request.num_tokens - 1       # 留一重算取logits
    computed_blocks, num_new_computed_tokens, num_uncached = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    # computed_blocks: tuple[list[KVCacheBlock], ...] 按group组织的命中块
    # num_new_computed_tokens: 所有group共同命中的token数（交集）
    # num_uncached: 最长单group命中与交集的差值（未缓存的共享前缀token数）

    if (
        num_new_computed_tokens > 0
        and self.enable_kv_cache_events
        and getattr(request, "kv_cache_report_mode", "incremental") == "full"
    ):
        for group_idx, group_blocks in enumerate(computed_blocks):
            num_blocks = len(group_blocks)
            if num_blocks > 0:
                group = self.kv_cache_config.kv_cache_groups[group_idx]
                block_size = group.kv_cache_spec.block_size
                self.block_pool.emit_cached_block_events(
                    request, num_blocks, block_size, group_idx
                )

    shared_prefix_boundary = (
        num_new_computed_tokens + num_uncached if num_uncached else 0
    )
    blocks = self.create_kv_cache_blocks(computed_blocks)
    return blocks, num_new_computed_tokens, shared_prefix_boundary
```

**为什么留一重算？** `allocate_slots()` 要求 `num_computed_tokens` 按 block_size 对齐，所以即使只差 1 个 token，也可能导致整块重算。这是当前实现的一个性能限制，注释中也提到未来可以优化。

#### 5.3.2 `get_computed_blocks_for_connector`：带 KV Connector 的前缀查找（297-342行）

用于启用 KV connector（如 nixl 跨节点 KV 传输）的场景，处理混合模型中 Full Attention 与 Mamba 命中分歧问题。

**执行流程**：
1. 快速路径判断：非混合模型（无 Mamba 层 或 非 HybridCoordinator 或 无 full_attention_group_id）→ 直接调用普通 `get_computed_blocks()`
2. 禁用缓存则返回空
3. 执行 per-group 独立查找（`find_longest_cache_hit_per_group`，不做交集收敛）
4. 如果某 group 命中比 FullAttention 更深 → FullAttention 尾块已被驱逐，回退到普通 `get_computed_blocks()` 用收敛后的交集边界
5. 否则返回 FullAttention 命中结果 + `hit_diverged = min(per_group_hits) < num_local`（存在 group 命中更浅时为 True）

```python
def get_computed_blocks_for_connector(
    self, request: Request
) -> tuple[KVCacheBlocks, int, int, bool]:
    coordinator = self.coordinator
    if not (
        self.kv_cache_config.has_mamba_layers
        and isinstance(coordinator, HybridKVCacheCoordinator)
        and coordinator.full_attention_group_id is not None
    ):
        return *self.get_computed_blocks(request), False   # 非混合→用普通查找

    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0, False

    fa_group_id = coordinator.full_attention_group_id
    computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
        request.block_hashes, request.num_tokens - 1
    )
    if any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
        return *self.get_computed_blocks(request), False   # FA尾块被驱逐→回退收敛边界

    num_local = per_group_hits[fa_group_id]
    blocks = self.create_kv_cache_blocks(computed)
    return blocks, num_local, 0, min(per_group_hits) < num_local
```

`hit_diverged=True` 表示该边界处 Mamba 状态缺失，调用方（connector 调度逻辑）需要回退 `get_computed_blocks` 对账，确保本地命中与远端传输状态一致。

### 5.4 `allocate_slots`：核心分配方法（344-565行）

Scheduler 调度请求时最核心的方法。空间不足时返回 `None` 表示无法调度该请求。

#### 5.4.1 Block 布局图（源码注释原文）

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                  |            < to be allocated >                   |
----------------------------------------------------------------------
                  | < to be cached (roughly, details below)>         |
----------------------------------------------------------------------
| Prefix-cached tokens from either vLLM or connector.                |
| Can be safely removed if they are outside sliding window.          |
----------------------------------------------------------------------
|   < cached by vLLM >    | not cached by vLLM, but cached by connector |
| ref_cnt increased       | ref_cnt not increased yet                  |
----------------------------------------------------------------------
```

缩写定义：
- `comp` = `request.num_computed_tokens`（已在之前 step 计算完成的 token）
- `new_comp` = `num_new_computed_tokens`（本轮前缀缓存新命中的 token）
- `ext_comp` = `num_external_computed_tokens`（KV connector 缓存的 token，vLLM 未持引用）
- `new` = `num_new_tokens`（本步需新计算的 token，含未验证 draft）
- `lookahead` = `num_lookahead_tokens`（EAGLE 投机解码预分配的 lookahead token）

> NOTE：`new` 包含 verified + unverified draft token，但只 cache verified（以 `request.num_tokens` 为上限）。

#### 5.4.2 方法签名

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,                           # 本步需新计算的token数（含未验证draft）
    num_new_computed_tokens: int = 0,              # 本轮前缀缓存新命中token数（不含外部）
    new_computed_blocks: KVCacheBlocks | None = None,  # 上述命中对应的cached blocks
    num_lookahead_tokens: int = 0,                # 投机解码lookahead token数（如EAGLE）
    num_external_computed_tokens: int = 0,        # connector缓存但非vLLM缓存的token数
    delay_cache_blocks: bool = False,             # P/D时延迟缓存（等远端KV传输完成）
    num_encoder_tokens: int = 0,                  # encoder-decoder模型cross-attention token数
    full_sequence_must_fit: bool = False,         # admission gate：整条sequence必须能放下才分配
    reserved_blocks: int = 0,                     # 为其他in-flight序列保留的空闲块数
    has_scheduled_reqs: bool = True,              # 本步是否已有其他请求调度（控制watermark应用）
) -> KVCacheBlocks | None:
```

#### 5.4.3 执行流程九步

```
┌─ ① 入参校验：num_new_tokens 和 num_external_computed_tokens 不能同时为0
│
├─ ② 基础计算：new_computed_block_list（空则用empty）、num_local_computed_tokens、total_computed_tokens
│
├─ ③ Watermark决策：WAITING/PREEMPTED + has_scheduled_reqs 才应用水位线
│
├─ ④ [可选] full_sequence_must_fit 预检查：整条序列能放下才继续（admission gate）
│
├─ ⑤ remove_skipped_blocks()：释放滑动窗口外不需要的block（先释放后分配，减少eviction）
│     用 processed_computed_tokens 保证 in-flight 步骤注意力窗口仍可读
│
├─ ⑥ get_num_blocks_to_allocate()：精确计算本步需要的block数
│
├─ ⑦ 容量检查：required_blocks <= available_blocks - reserved_blocks
│     available_blocks = free_blocks - reserved_blocks（为其他in-flight序列保留）
│     required_blocks = num_blocks_to_allocate + watermark_blocks
│
├─ ⑧ 两阶段分配（修复issue #33775跨组驱逐竞态）：
│   ├─ allocate_new_computed_blocks()：先touch所有命中块（ref_cnt++），抬升引用防止被驱逐
│   └─ allocate_new_blocks()：再从free queue取新块，如有partial_hit则执行CoW重定向
│
└─ ⑨ cache_blocks()：以request.num_tokens为上限缓存verified token，返回新块
       - delay_cache_blocks或enable_caching=False时跳过缓存
       - num_tokens_to_cache = min(total_computed + new_tokens, request.num_tokens)
         排除EAGLE rejected draft token
```

#### 5.4.4 关键代码段注释

**① 入参校验与基础计算（440-461行）**：

```python
    if num_new_tokens == 0 and num_external_computed_tokens == 0:
        raise ValueError(...)                       # 异步KV加载时num_new_tokens=0但ext_comp>0是合法的

    if new_computed_blocks is not None:
        new_computed_block_list = new_computed_blocks.blocks
    else:
        new_computed_block_list = self.empty_kv_cache_blocks.blocks  # 无前缀命中用空tuple

    num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
    total_computed_tokens = min(
        num_local_computed_tokens + num_external_computed_tokens, self.max_model_len
    )
```

**③ Watermark 决定（463-470行）**：

```python
    watermark_blocks = 0
    # Watermark仅对WAITING/PREEMPTED状态的请求，且本步已有其他请求调度时应用
    # - 对running decode不应用（不能因为水位线卡住正在解码的请求）
    # - 无其他调度请求时不应用（避免死锁，如所有请求都被watermark挡住）
    if has_scheduled_reqs and request.status in (
        RequestStatus.WAITING, RequestStatus.PREEMPTED,
    ):
        watermark_blocks = self.watermark_blocks
```

**④ Admission Gate 预检查（472-488行）**：

```python
    if full_sequence_must_fit:
        # waiting队列准入时先检查整条request sequence能否全部放下
        # 防止chunked prefill只检查首chunk导致过度准入
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,                     # 整条sequence长度
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,                       # 启用recycling-aware上限
        )
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > self.block_pool.get_num_free_blocks():
            return None                                     # 放不下→拒绝调度
```

**⑤ 清理窗口外 block（490-508行）**：

```python
    num_tokens_main_model = total_computed_tokens + num_new_tokens
    num_tokens_need_slot = min(num_tokens_main_model + num_lookahead_tokens, self.max_model_len)

    # 在分配新block之前释放不需要的block，先释放后分配减少eviction
    # 即使最终空间不足不调度此请求，也执行此清理
    # processed_computed_tokens = total_computed_tokens - num_in_flight_tokens
    # 保证in-flight步骤的注意力窗口仍可读（rejected spec tokens可回滚）
    self.coordinator.remove_skipped_blocks(
        request.request_id,
        max(0, total_computed_tokens - request.num_in_flight_tokens),
        num_prompt_tokens=request.num_prompt_tokens,
    )
```

**⑦ 精确容量检查（510-527行）**：

```python
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=num_local_computed_tokens + num_external_computed_tokens,
        num_local_computed_tokens=num_local_computed_tokens,
        num_tokens_main_model=num_tokens_main_model,
    )

    available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > available_blocks:
        return None                                        # 空间不足→无法调度
```

**⑧ 两阶段分配（529-547行）**：

```python
    if (
        new_computed_block_list is not self.empty_kv_cache_blocks.blocks
        or num_external_computed_tokens > 0
    ):
        # 第一阶段：touch所有本地命中块（ref_cnt++），追加到req_to_blocks
        # 防止第二阶段allocate_new_blocks时，group 0分配新块驱逐了group 1尚未touch的命中块
        # 修复跨group驱逐竞态问题（issue #33775）
        self.coordinator.allocate_new_computed_blocks(
            request_id=request.request_id,
            new_computed_blocks=new_computed_block_list,
            num_local_computed_tokens=num_local_computed_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
        )

    # 第二阶段：从free queue取新block分配；如有partial_hit则执行CoW重定向
    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id, num_tokens_need_slot, num_tokens_main_model, num_encoder_tokens,
    )
```

**⑨ 缓存写入与返回（549-565行）**：

```python
    if not self.enable_caching or delay_cache_blocks:
        return self.create_kv_cache_blocks(new_blocks)    # 延迟缓存或禁用→直接返回

    # 需要缓存的token数 = 已计算 + 新计算，但不超过request.num_tokens
    # request.num_tokens是已验证finalized的token数，排除rejected draft token
    # 保证EAGLE spec decode中未验证的draft token不会被写入前缀缓存
    num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)
    self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return self.create_kv_cache_blocks(new_blocks)
```

#### 5.4.5 场景实例（block_size = 16）

**场景1：waiting 请求，无命中，full_sequence_must_fit=True**

- 输入：prompt=80 token，无命中，free_blocks=10，watermark=2，reserved=0
- admission gate：`num_blocks_to_allocate=cdiv(80,16)=5`，`required=5+2=7 <= 10`，通过
- `remove_skipped_blocks`：FullAttention 不跳过，无操作
- 精确检查：`required=5+2=7 <= 10-0=10`，通过
- 两阶段分配：无命中块→跳过touch；分配5个新块
- 缓存：`num_tokens_to_cache=min(0+80, 80)=80`，缓存填满的块
- 返回：5个新块 ✓

**场景2：running 请求 decode，watermark 不应用**

- 输入：已有5块（80 token），num_new_tokens=1，request.status=RUNNING
- Watermark：RUNNING 状态→watermark_blocks=0
- admission gate：full_sequence_must_fit=False→跳过
- `remove_skipped_blocks`：FullAttention 不跳过
- `get_num_blocks_to_allocate`：RUNNING 快速路径→`cdiv(81,16)-5=6-5=1`
- 容量检查：`required=1+0=1 <= available-reserved`
- 两阶段分配：无新命中→跳过touch；分配1个新块
- 返回：1个新块 ✓

**场景3：EAGLE 投机解码，rejected draft token 不缓存**

- 输入：已计算100 token，num_new_tokens=5（1 verified + 4 draft），num_lookahead_tokens=4，request.num_tokens=101（只含verified）
- `num_tokens_to_cache = min(100+5, 101) = 101`
- 前100 token已缓存，新填满的块是token 100-111所在块，但实际只验证到101
- `cache_blocks` 只缓存到block对齐边界下token 101能覆盖的部分，rejected draft token所在块不写入哈希表
- 后续draft被拒绝的block会被覆盖或释放，不污染前缀缓存 ✓

### 5.5 释放与清理方法

#### 5.5.1 `free`：请求完成释放所有块（567-578行）

**执行流程**：
1. 从 `_partial_tail_pins` 弹出该请求的 pinned blocks（如有）
2. 如果有 pins，先调用 `block_pool.free_blocks(pins)` 释放钉住的 off-table blocks
3. 调用 `coordinator.free(request_id)` 逆序释放请求 blocks（尾部先释放，利于尾部后续被前缀缓存命中）

```python
def free(self, request: Request) -> None:
    pins = self._partial_tail_pins.pop(request.request_id, None)
    if pins:
        self.block_pool.free_blocks(pins)              # 先释放partial-tail钉住的block
    self.coordinator.free(request.request_id)          # 逆序释放请求block
```

逆序释放原因：新填充的块在尾部，逆序释放让尾部块先进入 LRU 空闲队列，优先保留尾部块可以提高后续请求命中率。

#### 5.5.2 `pop_blocks_for_free`：弹出但不归还块池（599-617行）

**执行流程**：
1. 调用 `coordinator.pop_blocks_for_free()` 弹出请求 blocks（从 req_to_blocks 移除，清理 num_cached_block 和 partial_hit_reqs）
2. 从 `_partial_tail_pins` 弹出 pins，拼接到 blocks 前面（pins 与请求块一起延迟释放）
3. 返回 block 列表，调用方最终必须逆序 free（swap-out 等延迟释放场景使用）

```python
def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
    blocks = self.coordinator.pop_blocks_for_free(request.request_id)
    pins = self._partial_tail_pins.pop(request.request_id, None)
    if pins:
        blocks = pins + blocks                          # pins随同弹出，一起延迟释放
    return blocks
```

#### 5.5.3 `remove_skipped_blocks`：移除窗口外 block（580-597行）

```python
def remove_skipped_blocks(
    self,
    request_id: str,
    processed_computed_tokens: int,          # 已完全处理完成的token前缀长度
    num_prompt_tokens: int | None = None,    # prompt长度（R-SWA gap eviction用）
) -> None:
    """processed_computed_tokens必须是已完全committed的token边界，
    不能包含in-flight step仍需读取的block。"""
    self.coordinator.remove_skipped_blocks(
        request_id, processed_computed_tokens, num_prompt_tokens
    )
```

将滑动窗口外/过期的 block 替换为 `_null_block` 占位（null_block 的 ref_cnt=∞，不占用空闲池配额），SWA/Mamba/R-SWA 管理器各自实现具体回收策略。

### 5.6 GPU 准备数据 drain 方法

每 step 调度结束后，Scheduler 调用这些方法取出本步产生的 GPU 任务，发给 Worker 执行。这些方法都是"排水槽"模式——调用时取出内部队列内容并清空，返回给上层使用。

#### 5.6.1 `take_new_block_ids`：取需 zeroing 的新块（796-801行）

```python
def take_new_block_ids(self) -> list[int]:
    ids: list[int] = []
    for mgr in self.coordinator.single_type_managers:
        ids.extend(mgr.take_new_block_ids())     # drain每个manager的new_block_ids队列
    return ids
    # Worker拿到这些ID后，对GPU KV cache对应位置执行zeroing
    # 只有records_new_block_ids=True的管理器(FuncAttention/Mamba align等)会记录
```

#### 5.6.2 `take_kv_cache_block_copies`：取 CoW 拷贝任务（831-846行）

```python
def take_kv_cache_block_copies(
    self,
) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
    pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
    for mgr in self.coordinator.single_type_managers:
        pending_copies.extend(mgr.take_pending_cow_copies())
        # drain每个manager的CoW拷贝队列（source_block → cow_block）
    copies = [
        KVCacheBlockCopy(src_block_id=src.block_id, dst_block_id=cow.block_id)
        for src, cow in pending_copies
    ]
    retained_blocks = [block for pair in pending_copies for block in pair]
    # retained_blocks保留source和cow引用，防止GC回收正在拷贝的block
    return copies, retained_blocks
```

Worker 拿到 `(src_id, dst_id)` 对后执行 GPU tensor copy，完成后源 block 可继续被其他请求共享，目标块是当前请求私有的可写入块。

#### 5.6.3 `take_partial_tail_offloads`：取 partial-tail offload 任务（848-874行）

```python
def take_partial_tail_offloads(self) -> dict[str, list[tuple[int, int, int]]]:
    offloads: dict[str, list[tuple[int, int, int]]] = {}
    for mgr in self.coordinator.single_type_managers:
        for (req_id, group_id, block, boundary_tokens) in mgr.take_pending_partial_tail_offloads():
            self.block_pool.touch((block,))               # 钉住block：ref_cnt++
            self._partial_tail_pins.setdefault(req_id, []).append(block)
            offloads.setdefault(req_id, []).append(
                (group_id, block.block_id, boundary_tokens)
            )
    return offloads
```

仅 Mamba "align" group 贡献。被 hand-off 的 block 不在请求 block 表中（off-table），因此 `touch()` 钉住它防止被驱逐，记录在 `_partial_tail_pins`，请求 `free()` 时才解钉归还。KV connector 读取这些 block 并 offload，使后续请求可以命中 sub-block 前缀。

#### 5.6.4 其他 zeroing 辅助方法

```python
def get_zeroing_block_ids_in_range(                     # kv_cache_manager.py:803-815
    self, request_id: str, start_token: int, end_token: int
) -> list[int]:
    """获取[start_token, end_token)范围内需zeroing的block ID。
    仅从records_new_block_ids的group收集。"""
    ids: list[int] = []
    for mgr in self.coordinator.single_type_managers:
        if mgr.records_new_block_ids:
            start_idx = start_token // mgr.block_size
            end_idx = cdiv(end_token, mgr.block_size)
            blocks = mgr.req_to_blocks[request_id]
            ids.extend(blk.block_id for blk in blocks[start_idx:end_idx])
    return ids

def record_blocks_for_zeroing(                          # kv_cache_manager.py:817-829
    self, request_id: str, start_token: int
) -> None:
    """重新记录从start_token起需zeroing的block。
    用于异步KV加载失败时，把加载失败的block重新加入zeroing队列。
    start_token必须block-aligned（zeroing部分有效块会擦除其有效前缀）。"""
    for mgr in self.coordinator.single_type_managers:
        if mgr.records_new_block_ids:
            assert start_token % mgr.block_size == 0
            start_idx = start_token // mgr.block_size
            blocks = mgr.req_to_blocks[request_id]
            mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])
```

### 5.7 查询、事件与生命周期方法

#### 5.7.1 `get_num_common_prefix_blocks`：公共前缀块数（643-675行）

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
    """Calculate the number of common prefix blocks for each kv cache group.
    以一个running请求为参考，遍历其blocks。
    一个block是common prefix当且仅当所有已分配KV cache的请求都共享它
    （ref_cnt == req_to_blocks中的条目数）。
    返回list[int]，每group一项。

    注意：已分配KV cache的请求数 ≥ 当前步调度请求数，因为还包含未调度但未释放block的请求。
    这导致边界情况：即使所有调度请求共享前缀，返回值也可能为0（被未调度请求拉低）。
    用于Cascade Attention优化共享前缀的decode吞吐。
    """
    return self.coordinator.get_num_common_prefix_blocks(running_request_id)
```

#### 5.7.2 `estimate_cached_tokens`：估算请求已缓存 token 数（731-758行）

**执行流程**：
1. 遍历每个 group，跳过 CrossAttention 和 EncoderOnly group（不参与前缀缓存）
2. 对每个 group 的 blocks，取所有 block 中 `block_hash_num_tokens` 的最大值（该 group 已缓存到的最远 token）
3. 最终取所有 group 的最小值（前缀缓存命中必须是所有 group 的交集）

```python
def estimate_cached_tokens(self, request: Request) -> int:
    cached_tokens: int | None = None
    for group, blocks in zip(
        self.kv_cache_config.kv_cache_groups,
        self.get_blocks(request.request_id).blocks,
    ):
        if isinstance(group.kv_cache_spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)):
            continue                                        # cross/encoder不参与前缀缓存
        group_cached_tokens = 0
        for block in blocks:
            group_cached_tokens = max(group_cached_tokens, block.block_hash_num_tokens or 0)
        cached_tokens = (
            group_cached_tokens if cached_tokens is None
            else min(cached_tokens, group_cached_tokens)    # 取所有group的最小值（交集）
        )
    return cached_tokens or 0
```

#### 5.7.3 `take_events`：取 KV cache 事件并标注元数据（677-701行）

```python
def take_events(self) -> list[KVCacheEvent]:
    events = self.block_pool.take_events()
    for event in events:
        if not isinstance(event, BlockStored):
            continue
        if event.group_idx is None:
            continue
        if event.group_idx < 0 or event.group_idx >= len(self.kv_cache_event_metadata):
            logger.warning("Group index `%s` not in KV cache metadata", event.group_idx)
            continue
        kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
        event.kv_cache_spec_kind = kind                    # 标注spec类型（full_attention/sliding_window/mamba等）
        event.kv_cache_spec_sliding_window = sliding_window  # 标注滑动窗口大小
    return events
```

事件标注解耦设计：BlockPool 只发射结构化事件不持有语义 spec 元数据，Manager 在此为 `BlockStored` 事件标注元数据，松耦合支持 KV connector 等外部消费者。

#### 5.7.4 其他方法简要说明

```python
def get_blocks(self, request_id: str) -> KVCacheBlocks:          # kv_cache_manager.py:703-705
    """获取请求的blocks，封装为KVCacheBlocks。"""
    return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:  # kv_cache_manager.py:707-709
    """获取请求的block ID列表。"""
    return self.get_blocks(request_id).get_block_ids()

def get_block_ids_for_computed_tokens(                           # kv_cache_manager.py:711-729
    self, request_id: str, num_computed_tokens: int
) -> tuple[list[int], ...]:
    """截取已计算token覆盖的block ID（按各group block_size对齐裁剪）。
    CrossAttention和EncoderOnly group返回完整ids不裁剪。"""

def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:  # kv_cache_manager.py:760-769
    """委托coordinator.cache_blocks，enable_caching=False时跳过。"""

def create_kv_cache_blocks(                                      # kv_cache_manager.py:771-775
    self, blocks: tuple[list[KVCacheBlock], ...]
) -> KVCacheBlocks:
    """非空blocks创建KVCacheBlocks，空blocks复用empty_kv_cache_blocks单例。"""
    return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

def truncate_computed_blocks(                                    # kv_cache_manager.py:777-794
    self, blocks: KVCacheBlocks, num_computed_tokens: int
) -> KVCacheBlocks:
    """纯切片：返回截断到block对齐边界的KVCacheBlocks视图，refcount不变。"""

def new_step_starts(self) -> None:                               # kv_cache_manager.py:876-878
    """通知coordinator新step开始，清空前一步的临时状态。"""
    self.coordinator.new_step_starts()

def reset_prefix_cache(self) -> bool:                            # kv_cache_manager.py:627-641
    """重置前缀缓存（RLHF权重更新后、benchmark时使用），block_pool.reset失败返回False。"""

def evict_blocks(self, block_ids: set[int]) -> None:             # kv_cache_manager.py:619-625
    """按block ID驱逐前缀缓存中的块，委托block_pool。"""
```

---

## 六、Scheduler 交互节奏

Scheduler 一个 step 内与 KVCacheManager 的典型交互顺序：

```
① new_step_starts()
   清空前一步临时数据（new_block_ids、CoW copies、partial-tail offloads）
   源码：kv_cache_manager.py:876-878

② 处理running队列（decode阶段）：
   对每个running请求：
   allocate_slots(request, num_new_tokens=1, has_scheduled_reqs=...)
   → 追加1个decode token的block slot
   源码：kv_cache_manager.py:344

③ 处理waiting队列（prefill阶段）：
   对每个waiting请求：
   a) get_computed_blocks(request)
      → (cached_blocks, num_hit, shared_prefix_boundary)
      源码：kv_cache_manager.py:229
   b) allocate_slots(request, num_new_tokens=prefill_chunk_size,
                     new_computed_blocks=cached_blocks,
                     num_new_computed_tokens=num_hit,
                     full_sequence_must_fit=True,       # admission gate
                     has_scheduled_reqs=True)
      → 命中前缀+分配新块；空间不足返回None则该请求继续等待
      源码：kv_cache_manager.py:344

④ GPU计算完成后：
   对本步调度的请求：
   cache_blocks(request, num_computed_tokens)
   → 把新填满的block存入前缀缓存（hash计算+哈希表写入）
   源码：kv_cache_manager.py:760

⑤ 准备发给Worker的数据：
   - take_new_block_ids()          → Worker zeroing新块
     源码：kv_cache_manager.py:796
   - take_kv_cache_block_copies()  → Worker执行CoW GPU拷贝
     源码：kv_cache_manager.py:831
   - take_partial_tail_offloads()  → KV connector offload任务
     源码：kv_cache_manager.py:848
   - take_events()                 → KV cache事件（BlockStored等）
     源码：kv_cache_manager.py:677

⑥ 请求结束/抢占：
   - 正常结束：free(request) → 逆序释放所有block + 解钉pins
     源码：kv_cache_manager.py:567
   - 抢占：pop_blocks_for_free(request) → 弹出块列表，延迟释放
     源码：kv_cache_manager.py:599
```

---

## 七、设计要点小结

1. **门面模式 + 唯一协议**：`KVCacheManager` 是 Scheduler 唯一直接交互对象，`KVCacheBlocks` 是唯一数据交换协议。Scheduler 不接触 `KVCacheBlock`、`Coordinator`、`SingleTypeManager`、`BlockPool` 的内部细节，内部重构不影响上层。

2. **空对象复用**：`empty_kv_cache_blocks` 是预构造的不可变单例，所有无 block 请求复用它，`create_kv_cache_blocks()` 对空 blocks 直接返回单例，避免 Python GC 开销。

3. **留一重算**：`get_computed_blocks` 中 `max_cache_hit_length = num_tokens - 1`，强制最后一个 token 重算取 logits，即使全部命中也不例外。这可能导致整 block 重算（因为 `allocate_slots` 要求 `num_computed_tokens` 按 block_size 对齐）。

4. **三阶段分配 + 两阶段 touch**：admission gate（`full_sequence_must_fit` + watermark + reserved_blocks）→ `remove_skipped_blocks` 先释放后分配 → 精确容量检查 → 两阶段分配（先 `allocate_new_computed_blocks` 抬 `ref_cnt` 防跨组驱逐 issue #33775，再 `allocate_new_blocks` 取新块+CoW）→ `cache_blocks` 以 `request.num_tokens` 为上限只缓存 verified token。

5. **Watermark 策略**：仅对 `WAITING/PREEMPTED` + `has_scheduled_reqs=True` 时应用水位线预留空闲块，防止过度准入导致频繁抢占；running decode 不应用（不能卡住正在解码的请求），无 running 请求时也不应用（避免死锁）。

6. **reserved_blocks 异步保护**：为其他 in-flight 序列保留空闲块，gate 异步 KV-connector 加载，防止新请求吃掉正在运行的 prefill 序列依赖的 block。

7. **partial-tail pin 机制**：`take_partial_tail_offloads()` 取出的 off-table block 通过 `block_pool.touch()` 钉住，记录在 `_partial_tail_pins`，请求 `free()` 时才解钉归还，保证 connector 读取期间 block 不被驱逐。

8. **事件标注解耦**：BlockPool 只发射结构化事件不持有语义 spec 元数据，`take_events()` 在此为 `BlockStored` 事件标注 `kv_cache_spec_kind` 和 `sliding_window`，松耦合支持 KV connector 等外部消费者。

9. **KV Connector 分歧处理**：`get_computed_blocks_for_connector()` 在混合模型下执行 per-group 独立查找，检测 Full Attention 与 Mamba 命中分歧，必要时回退到收敛后的交集边界，确保本地命中与远端传输状态一致。

10. **统计条件记录**：`record_prefix_cache_stats()` 对跳过缓存查找的请求（prompt logprobs、pooling 模型）和抢占的请求不计入命中率统计，保证统计数据准确反映真实前缀缓存效果。
