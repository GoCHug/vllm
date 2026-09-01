# KVCacheManager 详解

> 五层架构第 5 层（最顶门面，Scheduler 唯一交互入口）｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)
> 时序位置：[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) **③ 前缀查找 → ④ 分配与缓存 → ⑤ 组装 SchedulerOutput → ⑧ 释放，每一步都从它入口**
>
> 源文件：`vllm/vllm/v1/core/kv_cache_manager.py`
>
> 主线：纯 Full Attention 单 group（内部持 `UnitaryKVCacheCoordinator`）。**本文重点：Scheduler 在时序 ③/④/⑤/⑧ 阶段真正调用的方法（`get_computed_blocks` / `allocate_slots` / `take_new_block_ids` / `free` / `pop_blocks_for_free`）逐行看源码；其余查询/统计/事件方法一张表带过。**

## 1. 概览

`KVCacheManager` 是五层 KV Cache 管理架构中的**第五层——最顶层门面**，也是 Scheduler 与 KV Cache 子系统交互的**唯一入口**。

对于纯 Full Attention 模型，它内部持有一个 `UnitaryKVCacheCoordinator`，把 Scheduler 的请求转发给下层 Coordinator，同时提供 Scheduler 需要的所有接口：前缀查找、槽位分配、块释放、新块清零数据收集、事件收集等。

**Scheduler 不需要知道下面有 Coordinator、Manager、BlockPool、物理张量这些层次**——它只和 KVCacheManager 打交道。

---

## 2. 职责与定位

### 调度流程中 KVCacheManager 的职责与调用时序

> 源码入口：`Scheduler.schedule()` 位于 [vllm\vllm\v1\core\sched\scheduler.py:427-1226]


#### 全景图

> **环境**：Llama-3-8B · pp2tp2 · 4 worker · 单 group (Full Attention) · 单 BlockPool (4096 块, block_size=16)
>
> **请求 R**：prompt = 70 token (含 32 token 共享前缀 SP)，max_tokens = 32
>
> **前置**：请求 P 先于 R 服务，已把 SP（32 token = 2 满块）写入前缀缓存，块 0/1 作为带哈希缓存块保留。R 与 P 共享 SP 前缀。

下面以 R 的一生为线索，展示与 KVCacheManager 的完整交互：

```
┌──────────────────────────────────────────────────────────────────────┐
│ A. 入队                                                              │
│                                                                      │
│   EngineCore.add_request(R) → Scheduler: R 入 WAITING 队列           │
│   预计算链式哈希: 70÷16=4 个满块有 hash (t0-63)                       │
│   ※ KVCacheManager 尚未参与                                          │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ B. 首次调度 prefill（WAITING → RUNNING）                              │
│                                                                      │
│   km.new_step_starts()                    新步开始，重置内部状态      │
│                                                                      │
│   ① km.get_computed_blocks(R)             前缀缓存查找               │
│      → 链式哈希逐块查表，命中 P 缓存的块 0/1                          │
│      → hit_length=32, 返回 KVCacheBlocks(([blk0, blk1],))            │
│                                                                      │
│   ② km.allocate_slots(R, num_new_tokens=70)  准入→分配→缓存          │
│      ├─ 容量检查: 需 3 新块, 4096 池足够 → 通过                      │
│      ├─ touch 命中块 0/1: ref_cnt 1→2, 移出 free 队列                │
│      ├─ get_new_blocks(3): pop [2, 3, 4] from BlockPool             │
│      ├─ block_table = [命中0, 命中1, 新2, 新3, 新4]                  │
│      └─ cache_blocks: 满块 2/3 入哈希表 (块4 未满不入)               │
│                                                                      │
│   ③ km.take_new_block_ids() → [2, 3, 4]    打包给 Worker 清零        │
│                                                                      │
│   → SchedulerOutput(block_ids=([0,1,2,3,4],)) → 4 worker 各收到      │
│      4 worker 共享同一 block_id 命名, 物理张量各自独立                │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ C. GPU forward（KVCacheManager 不参与）                               │
│                                                                      │
│   4 worker 各自:                                                     │
│   清零 block 2/3/4 → forward 写 70 token KV → sample                 │
│   kv_caches[layer][block_id] fancy index 第0维 (每 worker 16 张量)   │
│   → 第 1 个输出 token → R 状态变 RUNNING                             │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ D. decode 续写（32 步循环, 每步 1 token）                              │
│                                                                      │
│   ┌─ loop 每步 ──────────────────────────────────────────────────┐  │
│   │                                                              │  │
│   │  km.allocate_slots(R, num_new_tokens=1)                      │  │
│   │   ├─ 当前块未满 → 0 新块                                     │  │
│   │   └─ 当前块写满 → 1 新块 (pop from BlockPool)                │  │
│   │  满块入哈希表 (cache_blocks)                                  │  │
│   │                                                              │  │
│   │  GPU forward: 读已有 KV + 写新 1 token → sample → 1 输出     │  │
│   │                                                              │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   R 的 block_table 演变:                                             │
│   prefill 后  [0, 1, 2, 3, 4]     块4 有 6 token                     │
│   步 1~10    [0, 1, 2, 3, 4]     填块4 → 步10 满 → 入哈希表          │
│   步 11      [0, 1, 2, 3, 4, 5]  申块5, 0 分配中                     │
│   步 11~26   [0, 1, 2, 3, 4, 5]  填块5 → 步26 满 → 入哈希表          │
│   步 27      [0, 1, 2, 3, 4, 5, 6]  申块6                            │
│   步 27~32   [0, 1, 2, 3, 4, 5, 6]  填块6 (6 token, 未满不入表)      │
│                                                                      │
│   32 步分布: 块4=10 · 块5=16 · 块6=6  (合计 32 ✓)                    │
│   km.take_new_block_ids() 在步 11/27 返回 [5]/[6] 供 Worker 清零     │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ E. 释放                                                               │
│                                                                      │
│   km.free(R)                                                         │
│   ├─ 逆序释放 block_table: 块6→5→4→3→2                               │
│   │   ref_cnt-- 归 0 → 回 free_block_queue                           │
│   │   有哈希 → append 队尾 (LRU 保护, 可被后续请求前缀命中)            │
│   │   无哈希 → prepend 队首 (优先复用)                                │
│   └─ 命中块 0/1: 仅 ref_cnt-- (仍被 P 或其他请求共享, 不回收)          │
│                                                                      │
│   R 的 block_table 销毁, 7 个块 ID 归还 BlockPool (0/1 除外)         │
└──────────────────────────────────────────────────────────────────────┘
```

**R 的完整交互清单**（KVCacheManager 方法调用时序）：

| 阶段 | 方法 | R 的实际参数与结果 |
|------|------|-------------------|
| A 入队 | — | km 不参与（Scheduler 预计算哈希） |
| B① 前缀查找 | `get_computed_blocks` | hit_blocks=[blk0, blk1], hit_length=32 |
| B② 分配 | `allocate_slots` | 70 token → touch 2 命中 + pop 3 新[2,3,4] → block_table=[0,1,2,3,4] |
| B③ 打包 | `take_new_block_ids` | 返回 [2,3,4] → Worker 清零 |
| C forward | — | km 不参与（GPU 侧执行） |
| D decode×32 | `allocate_slots` | 每步 1 token → 步11 pop[5], 步27 pop[6] |
| D drain×2 | `take_new_block_ids` | 步11→[5], 步27→[6] → Worker 清零 |
| D 缓存 | `cache_blocks` | 块4 步10满→入表, 块5 步26满→入表 |
| E 释放 | `free` | 逆序 6→5→4→3→2 归还; 0/1 ref_cnt-- |

**核心概念**：
- **BlockPool 唯一**：全模型 1 个 BlockPool（4096 块），4 worker 共享同一 block_id 命名空间
- **block_table**：每个请求维护的 block_id 列表，R 从 [0,1,2,3,4] 增长到 [0,1,2,3,4,5,6]
- **Drain（排空/取清单）**：调度中 km 一边干活一边"记账"（新分配了哪些块），等调度完了**一次性取走**交给 Worker（`take_*` 方法），取完内部列表清空
- **touch vs allocate**：命中块只 touch（ref_cnt++ , 零拷贝复用），未命中才 allocate（从 free 队列 pop 新块）

---


## 3. 文件结构

```
kv_cache_manager.py
├── KVCacheBlocks（dataclass）—— Scheduler与KVCacheManager交换块数据的协议
│   ├── blocks: tuple[Sequence[KVCacheBlock], ...]  # 按组组织的块
│   ├── __add__()               # 合并两个KVCacheBlocks
│   ├── get_block_ids()         # 提取block_id
│   ├── get_unhashed_block_ids() # 获取未哈希的块ID（清零用）
│   └── new_empty()             # 创建空块
│
└── KVCacheManager —— 唯一门面类
    ├── __init__()              # 构造：创建Coordinator、BlockPool引用
    │
    ├── 【前缀查找】
    │   ├── get_computed_blocks()          # 标准前缀查找
    │   └── get_computed_blocks_for_connector()  # KV Connector专用查找
    │
    ├── 【核心分配】
    │   └── allocate_slots()               # 三阶段分配（最核心方法，约130行）
    │
    ├── 【释放与清理】
    │   ├── free()                         # 直接释放请求的所有块
    │   ├── pop_blocks_for_free()          # 弹出块（延迟逆序释放）
    │   ├── remove_skipped_blocks()        # 移除不需要的块（SWA窗口外）
    │   ├── evict_blocks()                 # 按ID驱逐缓存块
    │   └── reset_prefix_cache()           # 重置整个前缀缓存
    │
    ├── 【Drain方法（给Worker准备GPU数据）】
    │   ├── take_new_block_ids()           # 收集需要清零的新块ID
    │   └── take_partial_tail_offloads()   # 收集partial tail卸载任务
    │
    ├── 【查询与事件】
    │   ├── take_events()                  # 收集KV cache事件
    │   ├── get_blocks() / get_block_ids() # 查询请求的块
    │   ├── get_num_common_prefix_blocks() # 公共前缀块数（调度优先级）
    │   ├── estimate_cached_tokens()       # 估算缓存token数
    │   ├── usage                          # KV cache使用率属性
    │   └── get_block_ids_for_computed_tokens()  # 获取计算token对应的块ID
    │
    ├── 【生命周期】
    │   ├── new_step_starts()              # 新调度步开始通知
    │   ├── cache_blocks()                 # 触发缓存写入
    │   ├── truncate_computed_blocks()     # 截断到对齐的命中长度
    │   ├── record_blocks_for_zeroing()    # 重新记录需要清零的块
    │   └── get_zeroing_block_ids_in_range() # 获取范围内需要清零的块
    │
    └── 【辅助】
        ├── create_kv_cache_blocks()       # 工厂方法：非空才创建新对象，否则复用empty
        ├── prefix_cache_lookup_enabled()  # 是否启用前缀查找
        ├── record_prefix_cache_stats()    # 记录前缀缓存统计
        └── make_prefix_cache_stats()      # 获取并重置统计
```

---

## 4. KVCacheBlocks 详解

这是 Scheduler 和 KVCacheManager 之间交换块数据的**不可变数据类**，目的是隐藏内部数据结构，提供类型安全的接口。

源码位置：`kv_cache_manager.py:32-115`

### 4.1 类定义与核心字段

```python
@dataclass
class KVCacheBlocks:
    """Scheduler与KVCacheManager的接口协议，隐藏内部结构"""

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    blocks[i][j] = 第i个kv_cache_group的第j个块
    
    纯FullAttention场景：blocks = ([block0, block1, ...],)  ← 只有一个组
    """
```

### 4.2 核心方法

```python
    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """合并两个KVCacheBlocks（按组合并），用于追加新块"""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )
        # 例：old_blocks=([A,B],), new_blocks=([C],)
        # old + new = ([A,B,C],)

    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
        """提取block_id，用于传给Worker构造block_table"""
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)
        # 返回格式：([id0, id1, id2],)  ← 纯FullAttention

    def get_unhashed_block_ids(self) -> list[int]:
        """获取未哈希的块ID（这些块还没写入缓存，需要清零）"""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """所有组中获取未哈希的块ID（跳过null_block）"""
        return [
            [block.block_id for block in group if block.block_hash is None and not block.is_null]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """创建一个同样组数的空KVCacheBlocks，避免GC开销"""
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))
```

**设计要点**：
- 预创建`empty_kv_cache_blocks`复用，避免频繁创建空对象的GC开销（`create_kv_cache_blocks`方法会自动复用）
- 不可变设计：blocks是tuple of Sequence，不支持原地修改，防止意外篡改
- 按组组织，为未来多组不同block_size预留扩展能力

---

## 5. KVCacheManager 详解

### 5.1 构造函数

源码位置：`kv_cache_manager.py:118-191`

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,               # KV Cache全局配置
        max_model_len: int,                           # 模型最大上下文长度
        scheduler_block_size: int,                    # 调度块大小
        hash_block_size: int,                         # 哈希块大小
        max_in_flight_tokens: int | None = None,      # 最大同时处理token数
        enable_caching: bool = True,                  # 是否启用前缀缓存
        use_eagle: bool = False,                      # 是否启用EAGLE投机解码
        log_stats: bool = False,                      # 是否记录统计信息
        enable_kv_cache_events: bool = False,         # 是否启用KV事件
        dcp_world_size: int = 1,                      # 上下文并行world size
        pcp_world_size: int = 1,                      # 前缀缓存并行world size
        metrics_collector: KVCacheMetricsCollector | None = None,  # metrics收集器
        watermark: float = 0.0,                       # 空闲块水印比例（0-1），为运行中请求留 headroom
    ) -> None:
        # ========== 1. 基础配置保存 ==========
        self.max_model_len = max_model_len
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len  # 默认等于max_model_len
        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        # ========== 2. 创建Coordinator（工厂方法自动选择Unitary/Hybrid/NoPrefix）==========
        # 纯FullAttention单组场景下，这里创建UnitaryKVCacheCoordinator
        # Coordinator构造函数内部会创建BlockPool和SingleTypeManagers
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

        # ========== 3. 保存引用和配置 ==========
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool  # 直接引用Coordinator里的BlockPool
        self.kv_cache_config = kv_cache_config

        # ========== 4. Watermark计算 ==========
        # watermark是给等待请求预留的空闲块
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)

        # ========== 5. 预创建空KVCacheBlocks，避免GC ==========
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # ========== 6. 其他辅助字段 ==========
        self.kv_cache_event_metadata = tuple(...)  # 事件元数据（组类型、滑动窗口大小）
        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}  # partial tail pin记录
```

### 5.2 核心方法：前缀查找 `get_computed_blocks`

源码位置：`kv_cache_manager.py:229-295`

这是 Scheduler 在分配槽位之前调用的第一个方法——查找请求的prompt有多长的前缀已经在缓存里了。

```python
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
        """
        获取请求的已缓存（命中）块
        返回：(命中块KVCacheBlocks, 命中token数, shared_prefix_boundary)
        """
        # ========== 1. 如果禁用前缀缓存，或请求标记跳过，直接返回空 ==========
        if not self.prefix_cache_lookup_enabled(request):
            return self.empty_kv_cache_blocks, 0, 0

        # ========== 2. 关键：max_cache_hit_length = num_tokens - 1 ==========
        # 为什么要减1？
        # → 即使所有token都命中缓存，最后一个token也必须重新计算logits
        # → 而且allocate_slots要求num_computed_tokens是block对齐的，减1可能触发整个最后一块重算
        max_cache_hit_length = request.num_tokens - 1

        # ========== 3. 调用Coordinator查找 ==========
        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        # 纯FullAttention返回：
        # computed_blocks = ([hit0, hit1],)  （按组的命中块列表）
        # num_new_computed_tokens = 32       （命中token数）
        # num_uncached = 0                   （单组无此概念）

        # ========== 4. 发送KV事件（如果启用full report模式）==========
        if (num_new_computed_tokens > 0 and self.enable_kv_cache_events and ...):
            for group_idx, group_blocks in enumerate(computed_blocks):
                ...  # 发送BlockStored事件给外部消费者

        # ========== 5. 计算shared_prefix_boundary（Hybrid用，单组为0）==========
        shared_prefix_boundary = num_new_computed_tokens + num_uncached if num_uncached else 0

        # ========== 6. 包装成KVCacheBlocks返回 ==========
        blocks = self.create_kv_cache_blocks(computed_blocks)
        return blocks, num_new_computed_tokens, shared_prefix_boundary
```

**端到端例子**：示例 R（prompt = 70 token，前 32 token 为共享前缀 SP，由前置请求 P 缓存为块 0/1）
- `request.num_tokens = 70`
- `max_cache_hit_length = 69`（减1）
- `request.block_hashes = [hash(t0-15), hash(t16-31), hash(t32-47), hash(t48-63)]`——只有 4 个**满块**哈希（70 // 16 = 4）；尾块 t64-69 未满**没有哈希**，需等生成填满后由 `update_block_hashes()` 补上（`request.py:257`）
- 查找返回：命中前2个满块（P 缓存的 SP 块 0/1），共32token
- 返回：`(KVCacheBlocks([blockA, blockB]), 32, 0)`

### 5.3 核心方法：槽位分配 `allocate_slots`（最复杂，约130行）

源码位置：`kv_cache_manager.py:344-565`

这是整个KV Cache管理**最核心的方法**，Scheduler拿到前缀命中结果后调用它来分配需要的新块。源码注释里有详细的块布局图：

    Blocks layout:
    ----------------------------------------------------------------------
    | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
    ----------------------------------------------------------------------
                                                |   < to be computed >     |
    ----------------------------------------------------------------------
                                |            < to be allocated >           |
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
    comp      = request.num_computed_tokens
    new_comp  = num_new_computed_tokens
                = len(new_computed_blocks) * block_size
    ext_comp  = num_external_computed_tokens, cached by the connector
    new       = num_new_tokens, including unverified draft tokens
    lookahead = num_lookahead_tokens
    
源码docstring明确声明**分配分为三个阶段**：

```
阶段1: 释放 comp 中不需要的块，检查空闲块是否足够（不足则返回 None）
阶段2: 处理前缀 token（comp + new_comp + ext_comp）
        - 释放不需要的块（如滑动窗口外的）
        - 为 ext_comp 在滑动窗口内的 token 分配新块
阶段3: 为待计算的 token（new + lookahead）分配新块
```

> 注意：这里的 `阶段1/2/3` 是 `allocate_slots` **方法内部**的三个子阶段，与时序文档 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) 的应用级阶段 **A~E**（A入队/B调度/C forward/D decode/E释放）不是同一层级，勿混用。下文以"子阶段①/②/③"指代之。

下面按"前置准备 → 子阶段① → 子阶段② → 子阶段③"的顺序逐行注释源码。

#### 函数签名与参数

```python
def allocate_slots(
    self,
    request: Request,                              # 当前请求
    num_new_tokens: int,                           # 本轮要计算的新token数（含未验证draft）
    num_new_computed_tokens: int = 0,              # 刚命中的本地前缀缓存token数（不含ext）
    new_computed_blocks: KVCacheBlocks | None = None,  # 上面命中token对应的块（按group分组）
    num_lookahead_tokens: int = 0,                 # 投机解码lookahead token数（eagle等用）
    num_external_computed_tokens: int = 0,         # 外部Connector缓存的token数（vLLM不持有KV）
    delay_cache_blocks: bool = False,              # 是否延迟缓存（P/D远程传输未完成时跳过cache）
    num_encoder_tokens: int = 0,                   # encoder token数（encoder-decoder如Whisper的cross-attn）
    full_sequence_must_fit: bool = False,          # 准入门控：整个序列必须放得下才允许进入
    reserved_blocks: int = 0,                      # 为其他in-flight请求保留的空闲块数
    has_scheduled_reqs: bool = True,               # 本step是否已有请求被调度（控制watermark）
) -> KVCacheBlocks | None:
    """分配槽位，返回新分配的块；如果空间不足返回 None"""
```

#### 前置准备：参数校验 + token统计 + watermark设置

```python
# 异步加载KV数据时，可能num_new_tokens=0但仍需为external token分配slot
if num_new_tokens == 0 and num_external_computed_tokens == 0:
    raise ValueError(
        "num_new_tokens must be greater than 0 when there are no "
        "external computed tokens"
    )

# 统一取出命中块的内部列表，避免后续到处做None判断
if new_computed_blocks is not None:
    new_computed_block_list = new_computed_blocks.blocks   # 有命中：用命中块
else:
    new_computed_block_list = self.empty_kv_cache_blocks.blocks  # 无命中：用空列表占位

# 本地已计算token = 之前已计算的 + 刚命中前缀的
num_local_computed_tokens = (
    request.num_computed_tokens + num_new_computed_tokens
)
# 总已计算token = 本地 + 外部Connector，不能超过max_model_len
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    self.max_model_len,
)

# watermark只对WAITING请求生效，且本step已有其他请求在调度时才加
# 目的：给正在运行中的请求留出headroom
watermark_blocks = 0
if has_scheduled_reqs and request.status in (
    RequestStatus.WAITING,
):
    watermark_blocks = self.watermark_blocks
```

#### 子阶段①：释放 comp 中不需要的块，检查空闲块是否足够

源码docstring：*"Free unnecessary blocks in `comp` and check if we have sufficient free blocks (return None if not)."*

```python
# ---- 1a. full_sequence_must_fit 准入门控检查 ----
# chunked prefill场景下，如果只检查第一个chunk就可能放行一个永远放不下的请求
# 这里先按"完整序列"算一遍需求，放不下直接拒绝，避免无效占用
if full_sequence_must_fit:
    # 完整序列token数，同样受max_model_len约束
    full_num_tokens = min(request.num_tokens, self.max_model_len)

    # 向coordinator查询：放下整个序列还需要分配多少新块
    # apply_admission_cap=True 表示应用准入上限（防止过度估算）
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=full_num_tokens,              # 完整序列长度
        new_computed_blocks=new_computed_block_list,  # 可复用的命中块
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=total_computed_tokens,
        num_local_computed_tokens=num_local_computed_tokens,
        num_tokens_main_model=full_num_tokens,   # 主模型按完整序列算
        apply_admission_cap=True,
    )
    # 需求 = 实际要分配的块 + watermark预留
    required_blocks = num_blocks_to_allocate + watermark_blocks
    # 比较的是 get_num_free_blocks()（不含reserved，因为这是准入阶段）
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None   # 整个序列放不下，拒绝准入

# ---- 1b. 计算本轮实际需要slot的token数 ----
# 主模型token = 总已计算token + 本轮新token
num_tokens_main_model = total_computed_tokens + num_new_tokens
# 需要slot的token = 主模型token + lookahead，受max_model_len约束
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)

# ---- 1c. 释放滑动窗口（SWA）外不需要的块 ----
# 在分配新块之前先释放，能减少后续需要驱逐的块数
# 即使本请求最终因空间不足无法调度，这个释放也是安全的（SWA外的块确实不再需要）
# 基于"已处理token"来释放：in-flight的step还在读optimistic boundary以下的块，
# 被拒绝的spec token也可能回滚，所以用 (total_computed - num_in_flight_tokens) 作为下界
self.coordinator.remove_skipped_blocks(
    request.request_id,
    max(0, total_computed_tokens - request.num_in_flight_tokens),
    num_prompt_tokens=request.num_prompt_tokens,
)
# 纯FullAttention模型下这个函数基本是no-op（没有滑动窗口）

# ---- 1d. 计算本轮实际需要分配多少新块 ----
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
    request_id=request.request_id,
    num_tokens=num_tokens_need_slot,             # 本轮需要slot的token数
    new_computed_blocks=new_computed_block_list,  # 可复用的命中块（这些不算新分配）
    num_encoder_tokens=num_encoder_tokens,
    total_computed_tokens=num_local_computed_tokens
    + num_external_computed_tokens,
    num_local_computed_tokens=num_local_computed_tokens,
    num_tokens_main_model=num_tokens_main_model,
)

# ---- 1e. 空间检查：可用块 = 空闲块 - 预留块 ----
# reserved_blocks 是给其他in-flight请求留的（如async KV-connector加载时，
# 不能让新请求吃掉正在prefill的请求所依赖的块）
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None   # 空间不足，等待下轮调度
```

#### 子阶段②：处理前缀 token（comp + new_comp + ext_comp）

源码docstring：*"Handle prefix tokens (comp + new_comp + ext_comp): Free unnecessary blocks / Allocate new blocks for ext_comp tokens inside sliding window"*

```python
# 关键：必须先 touch 所有命中块（ref_cnt++），再分配新块！
# 否则分配新块时可能触发驱逐，把还没touch的命中块给驱逐掉（issue #33775）
# 触发条件：有本地命中块，或 有外部Connector token（ext_comp也需要分配slot）
if (
    new_computed_block_list is not self.empty_kv_cache_blocks.blocks
    or num_external_computed_tokens > 0
):
    # touch命中块：把命中块追加到请求的block列表中，ref_cnt++，标记为"正在使用"
    # 对于ext_comp的token：connector缓存了KV但vLLM没有，这里会为它们分配本地slot
    # （在滑动窗口范围内的ext_comp token需要本地块来接收传输的KV数据）
    self.coordinator.allocate_new_computed_blocks(
        request_id=request.request_id,
        new_computed_blocks=new_computed_block_list,
        num_local_computed_tokens=num_local_computed_tokens,
        num_external_computed_tokens=num_external_computed_tokens,
    )
```

#### 子阶段③：为待计算的 token（new + lookahead）分配新块

源码docstring：*"Allocate new blocks for tokens to be computed (new + lookahead)"*

```python
# ---- 3a. 真正分配新块 ----
# 从 free_block_queue 取出空闲块，加入 manager 的 req_to_blocks 映射
# 新分配的块ID会被追加到 manager.new_block_ids 列表，
# 后续 Worker 调用 take_new_block_ids() 取走这些ID，在forward前把对应块清零
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id,
    num_tokens_need_slot,      # 需要slot的总token数（含lookahead）
    num_tokens_main_model,     # 主模型token数（不含lookahead，决定主模型block边界）
    num_encoder_tokens,        # encoder token数（cross-attn用，decoder-only为0）
)

# ---- 3b. P/D 延迟缓存：远程传输未完成时先不缓存 ----
# P/D场景下，KV数据要从remote接收，本step还没收完，cache了会写入不完整数据
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)   # 直接返回，跳过cache

# ---- 3c. 缓存写入（调度阶段，forward之前）----
# 想缓存到 total_computed + num_new_tokens，但必须排除"不可提交"的token
# （如可能被拒绝的draft token），所以用 request.num_tokens 来cap，
# 确保只缓存"已finalized"的token
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
# cache_blocks 只缓存"满块"（num_tokens // block_size），尾块不缓存
# hash 基于 token ID（不依赖KV数据），所以 forward 之前就能算 hash 并写入
# cache_blocks 是幂等的：已缓存的块（num_cached_block >= num_full_blocks）直接跳过
#   - prompt阶段：前2块已在prefix cache中，num_cached_block=2 < num_full_blocks=4 → 缓存新满块2、3
#   - decode阶段：每满一个block_size的块，这里就会把它写入哈希表
# 外部调用方（async PP / KV Connector）也会在forward之后追加调用 cache_blocks
self.coordinator.cache_blocks(request, num_tokens_to_cache)

# ---- 3d. 返回新分配的块 ----
return self.create_kv_cache_blocks(new_blocks)
```

#### 端到端例子

示例 R（prompt = 70 token，前 32 token 为共享前缀 SP、已由前置请求 P 缓存为块 0/1），命中32token（2块），num_new_tokens=38：

- **前置准备**：`num_local_computed_tokens = 0 + 32 = 32`，`total_computed_tokens = 32`
- **子阶段①**：
  - `num_tokens_need_slot = min(32 + 38, max_model_len) = 70`
  - `remove_skipped_blocks`：FullAttention下no-op
  - `num_blocks_to_allocate = ceil(70/16) - 2 = 5 - 2 = 3`块
  - 空间检查：假设空闲块足够，通过
- **子阶段②**：`allocate_new_computed_blocks` → touch命中blockA、blockB，ref_cnt都+1
- **子阶段③**：
  - `allocate_new_blocks` → 从free_block_queue分配blockC、blockD、blockE，`new_block_ids=[blockC_id, blockD_id, blockE_id]`
  - `num_tokens_to_cache = min(32+38, 70) = 70`
    - `num_full_blocks = 70 // 16 = 4`，`num_cached_block = 2`（prefix hit已缓存前2块）
    - `num_cached_block(2) < num_full_blocks(4)` → 缓存新满块2、3（blockC、blockD），未满块4不入
  - 返回：`KVCacheBlocks(([blockC, blockD, blockE],))`

### 5.4 块释放方法

#### 5.4.1 直接释放 `free`

源码位置：`kv_cache_manager.py:567-578`

```python
    def free(self, request: Request) -> None:
        """请求结束时直接释放所有块"""
        # 先释放partial tail pin的块
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            self.block_pool.free_blocks(pins)
        # 再释放请求的块（manager内部逆序free）
        self.coordinator.free(request.request_id)
```

#### 5.4.2 弹出延迟释放 `pop_blocks_for_free`

源码位置：`kv_cache_manager.py:599-617`

用于延迟释放场景：需要先把块弹出来，但不立即归还——等 GPU in-flight 操作确认后再逆序释放。

```python
    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """弹出块（不归还free_block_queue），调用方必须最终逆序释放"""
        blocks = self.coordinator.pop_blocks_for_free(request.request_id)
        # partial tail pin的块也一起弹出
        pins = self._partial_tail_pins.pop(request.request_id, None)
        if pins:
            blocks = pins + blocks
        return blocks
        # 返回分配顺序：[blockA, blockB, blockC]
        # 上层释放时要反过来：block_pool.free_blocks([blockC, blockB, blockA])
        # 这样尾块（不完整块）先回free_block_queue，下次分配优先复用
```

### 5.5 Drain方法：给Worker准备GPU数据

这些方法是**drain模式**：调用一次就把累积的数据取走并清空，Worker拿到数据后在GPU上执行对应的内存操作（清零、卸载）。

#### 5.5.1 收集新块清零 `take_new_block_ids`

源码位置：`kv_cache_manager.py:796-801`

```python
    def take_new_block_ids(self) -> list[int]:
        """Drain：收集所有manager的new_block_ids，返回需要清零的块ID列表"""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())  # 每个manager取出自己的new_block_ids并清空
        return ids
        # 纯FullAttention返回：[blockC_id, ...]
        # Worker拿到后会在GPU上把这些块的KV内存清零
```

**工作流**：
1. `allocate_new_blocks`分配新块时，manager把block_id加入`self.new_block_ids`
2. 模型forward前，Worker调用`take_new_block_ids()`拿走所有ID
3. Worker在GPU上对这些块执行memset清零
4. 清零后新块才能安全写入KV数据

#### 5.5.2 收集Partial Tail卸载 `take_partial_tail_offloads`

源码位置：`kv_cache_manager.py:848-874`

这是Mamba/"align"模式的partial tail卸载，纯FullAttention基本不用，了解即可。

### 5.6 事件收集 `take_events`

源码位置：`kv_cache_manager.py:677-701`

```python
    def take_events(self) -> list[KVCacheEvent]:
        """Drain：收集BlockPool发出的KV事件，补充元数据后返回"""
        events = self.block_pool.take_events()
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            # 给事件补充组类型和滑动窗口大小等元数据
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events
        # 外部KV Connector（如Mooncake）会消费这些事件来做分布式缓存同步
```

### 5.7 其他辅助方法

```python
    @property
    def usage(self) -> float:
        """KV cache使用率（0.0-1.0）"""
        return self.block_pool.get_usage()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """获取请求的所有块"""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """获取请求的block_id列表，用于构造block_table传给模型"""
        return self.get_blocks(request_id).get_block_ids()

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """获取公共前缀块数，用于调度优先级计算"""
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def new_step_starts(self) -> None:
        """通知新调度步开始——每个manager重置new_step_starts（如清空new_block_ids）"""
        self.coordinator.new_step_starts()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """缓存写入：把满块按token ID计算hash写入前缀缓存。
        主要被 allocate_slots 内部调用，也会被外部调用方在 forward 之后追加调用。
        幂等：仅缓存满块，已缓存的块跳过。
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(self, blocks: tuple[list[KVCacheBlock], ...]) -> KVCacheBlocks:
        """工厂方法：非空才创建新对象，空的话复用empty_kv_cache_blocks，减少GC"""
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
```

---

## 6. 方法调用总览（对照时序阶段）

一个请求从分配到释放，Scheduler 按端到端阶段逐一调用 KVCacheManager 的方法：

| 端到端阶段 | 方法 | 说明 |
|---------|------|------|
| 步开始 | `new_step_starts()` | 重置内部状态（清空 `new_block_ids` 等） |
| **③ 前缀查找** | `get_computed_blocks(req)` | 返回命中块和命中 token 数 → 下放 `coordinator.find_longest_cache_hit` |
| **④ 分配与缓存** | `allocate_slots(req, ...)` | 内部含准入检查、两阶段分配、`cache_blocks`（见 §5.3） |
| **⑤ 组装 SchedulerOutput** | `take_new_block_ids()` | 给 Worker 准备清零清单 |
| （GPU forward） | — | KVCacheManager 不参与 |
| 补缓存（可选） | `cache_blocks(req, ...)` | async PP / KV Connector 场景 forward 后外部追加 |
| **⑧ 释放** | `free(req)` / `pop_blocks_for_free(req)` | 正常结束直接释放；延迟释放先弹出再逆序释放 |

时序图可见 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) §3.2-§3.5。

---

## 7. 设计要点小结

1. **门面模式**：KVCacheManager是典型的Facade门面，为Scheduler提供一个简化的单一入口，封装了下面4层的所有复杂度
2. **三阶段分配**：`allocate_slots`是核心，逻辑分为：准入检查 → 两阶段touch+allocate → 缓存写入，每一步都有明确的职责
3. **两阶段分配修复竞态**：先`allocate_new_computed_blocks`（touch所有命中块，ref_cnt++防驱逐），再`allocate_new_blocks`（真正分配），这是修复issue #33775的关键
4. **cache_blocks基于token ID计算hash**：不依赖KV数据，所以能在forward之前调用；幂等设计支持被外部调用方在forward之后追加调用（async PP / KV Connector场景）
5. **cache_blocks幂等性**：只缓存满块（`num_tokens // block_size`），已缓存的块（`num_cached_block >= num_full_blocks`）直接跳过，多次调用安全
6. **Drain模式数据准备**：`take_new_block_ids`、`take_partial_tail_offloads`、`take_events`都是"调用即取走并清空"的drain模式，Worker批量拿到后在GPU上执行，CPU/GPU解耦
7. **逆序释放优化**：`pop_blocks_for_free`返回分配顺序的块，上层必须逆序释放，让尾部分配的不完整块优先回到free_block_queue头部，提高下次分配的尾块复用率
8. **Watermark机制**：给WAITING请求预留watermark_blocks的空闲块，为运行中请求留出 headroom
9. **GC优化**：预创建`empty_kv_cache_blocks`复用，`create_kv_cache_blocks`工厂方法避免频繁创建空对象
10. **不可变数据协议**：`KVCacheBlocks`使用tuple不可变结构，作为Scheduler和KVCacheManager之间的安全接口，防止内部状态被意外篡改
11. **num_tokens-1细节**：前缀查找时`max_cache_hit_length = num_tokens - 1`，即使全命中也要重算最后一个token的logits，保证输出正确性
12. **投机解码安全**：`cache_blocks`用`request.num_tokens`做cap，只缓存已finalized的token，防止被拒绝的draft token污染前缀缓存
