# KVCacheManager 详解

## 一、是什么

`KVCacheManager` 是五层 KV Cache 管理架构中的**第五层——最顶层门面**，也是 Scheduler 与 KV Cache 子系统交互的**唯一入口**。

对于纯 Full Attention 模型，它内部持有一个 `UnitaryKVCacheCoordinator`，把 Scheduler 的请求转发给下层 Coordinator，同时提供 Scheduler 需要的所有接口：前缀查找、槽位分配、块释放、新块清零数据收集、事件收集等。

**Scheduler 不需要知道下面有 Coordinator、Manager、BlockPool、物理张量这些层次**——它只和 KVCacheManager 打交道。

---

## 二、干什么用

### 在五层架构中的位置

```
┌─────────────────────────────────────────────────────────────┐
│ Scheduler（调度器，只和KVCacheManager交互）                   │
│     ↓ 调用                                                    │
│ 第五层：KVCacheManager  ← 本文讲解（唯一门面）                │
├─────────────────────────────────────────────────────────────┤
│ 第四层：KVCacheCoordinator                                    │
│  └── UnitaryKVCacheCoordinator（单组透传）                   │
├─────────────────────────────────────────────────────────────┤
│ 第三层：SingleTypeKVCacheManager                              │
│  └── FullAttentionManager（链式哈希前缀缓存）                 │
├─────────────────────────────────────────────────────────────┤
│ 第二层：BlockPool（块池，管理 free_block_queue 和哈希映射）    │
├─────────────────────────────────────────────────────────────┤
│ 第一层：物理 KV Cache 张量（GPU 上真实存储 K/V 的大张量）       │
└─────────────────────────────────────────────────────────────┘
```

### 调度流程中 KVCacheManager 的职责与调用时序

> 源码入口：`Scheduler.schedule()` 位于 [vllm\vllm\v1\core\sched\scheduler.py:427-1226]


#### 全景图

KVCacheManager 是 Scheduler 操作 KV Cache 的**唯一入口**。vLLM 的推理是一个"调度→计算"不断循环的过程：Scheduler 决定算哪些 token，KVCacheManager 分配/管理 KV 块，Worker 在 GPU 上执行，结果返回后进入下一轮。先看全景图（每个方法的内部细节在后续章节详细展开）：

```
┌──────────────────────────────────────────────────────────────────────┐
│                     EngineCore 主循环（每步重复）                     │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ① 调度阶段：Scheduler.schedule()                                     │
│                                                                      │
│ ├── kv_cache_manager.new_step_starts()    新步开始，重置内部状态      │
│ │                                                                    │
│ ├── 调度 Running 请求（已有块，继续decode/prefill）                   │
│ │   └── 每个请求：kv_cache_manager.allocate_slots(...)               │
│ │         └→ 返回None(空间不够)? → 抢占：                            │
│ │              kv_cache_manager.free()/pop_blocks_for_free() 释放    │
│ │              被抢占请求的块 → 重试                                  │
│ │                                                                    │
│ ├── 调度 Waiting 请求（新来的/被抢占的）                              │
│ │   ├── kv_cache_manager.get_computed_blocks()    前缀缓存查找       │
│ │   └── kv_cache_manager.allocate_slots(...)      准入→分配→缓存     │
│ │         └→ 返回None(空间不够)? → 跳过（不抢占Running）             │
│ │                                                                    │
│ ├── kv_cache_manager.get_num_common_prefix_blocks()  公共前缀查询    │
│ │                                                                    │
│ └── Drain：取出GPU待办事项，打包给Worker                             │
│     ├── kv_cache_manager.take_new_block_ids()        → 新块需清零   │
│     ├── kv_cache_manager.take_kv_cache_block_copies()→ COW拷贝对    │
│     └── kv_cache_manager.take_partial_tail_offloads()→ 尾块传输信息 │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ② GPU计算：Worker.forward()（KVCacheManager不参与GPU计算）           │
│   清零新块 → 执行COW拷贝 → 模型前向计算写K/V到GPU张量                │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ③ 处理结果                                                           │
│ ├── 追加生成的token到请求，更新num_computed_tokens                   │
│ ├── 完成/被抢占的请求释放块：                                        │
│ │   ├── kv_cache_manager.free()           立即释放                  │
│ │   └── kv_cache_manager.pop_blocks_for_free()  延迟释放(GPU in-flight)│
│ └── kv_cache_manager.take_events()    取出KV事件(供metrics/connector)│
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                              ┌─────┴─────┐
                              │ 回到① ↺   │
                              └───────────┘

退出：正常结束(EOS/max_tokens)→free移除 | 被抢占→放回Waiting下次重走前缀查找
```

**核心概念先明确**：
- **Running 请求**：已经在跑的请求（之前步骤已经分配过 KV 块），每步 decode 1个或多个 token
- **Waiting 请求**：新来的请求或被抢占后等待重新调度的请求，需要先做前缀缓存查找
- **Drain（排空/取清单）**：调度过程中，KVCacheManager 会一边干活一边"记账"——比如新分配了哪些块、产生了哪些COW拷贝，都随手记在内部列表里。等所有请求调度完了，要给Worker准备GPU任务清单时，就**一次性把这些记的东西全部取出来交给Worker，取完内部列表就空了**（所以叫"排空"）。对应的三个 `take_*` 方法就是干这个的。
- **块（Block）**：KV Cache 的分配单位，固定大小（如16个token）。每个请求的 KV 按块组织，称为 block_table

---


## 三、文件结构

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
    │   ├── pop_blocks_for_free()          # 弹出块（延迟逆序释放，用于preempt）
    │   ├── remove_skipped_blocks()        # 移除不需要的块（SWA窗口外）
    │   ├── evict_blocks()                 # 按ID驱逐缓存块
    │   └── reset_prefix_cache()           # 重置整个前缀缓存
    │
    ├── 【Drain方法（给Worker准备GPU数据）】
    │   ├── take_new_block_ids()           # 收集需要清零的新块ID
    │   ├── take_kv_cache_block_copies()   # 收集COW拷贝任务
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

## 四、KVCacheBlocks 详解

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
    
    为什么外层是组维度，不是块维度？
    → 因为未来不同组可能有不同block_size，块数不一定相等
    → 按组组织更灵活
    
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

## 五、KVCacheManager 详解

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
        watermark: float = 0.0,                       # 空闲块水印比例（0-1），防止频繁抢占
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
        # watermark是给等待/抢占请求预留的空闲块，防止它们进来导致频繁抢占
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

**端到端例子**：34token prompt
- `request.num_tokens = 34`
- `max_cache_hit_length = 33`（减1）
- `request.block_hashes = [hash(t0-15), hash(t16-31), hash(t32-33)]`
- 查找返回：命中前2个满块，共32token
- 返回：`(KVCacheBlocks([blockA, blockB]), 32, 0)`

### 5.3 核心方法：槽位分配 `allocate_slots`（最复杂，约130行）

源码位置：`kv_cache_manager.py:344-565`

这是整个KV Cache管理**最核心的方法**，Scheduler拿到前缀命中结果后调用它来分配需要的新块。源码注释里有详细的块布局图：

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                              |   < to be computed >     |
----------------------------------------------------------------------
                              |            < to be allocated >           |
----------------------------------------------------------------------
comp      = request.num_computed_tokens  （已计算的token）
new_comp  = num_new_computed_tokens      （刚命中的本地前缀token）
ext_comp  = num_external_computed_tokens （外部Connector缓存的token）
new       = num_new_tokens               （本轮新token，含未验证的draft）
lookahead = num_lookahead_tokens         （投机解码的lookahead token）
```

分配分为三个主要阶段：

```python
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,                           # 本轮要计算的新token数
        num_new_computed_tokens: int = 0,              # 刚命中的本地前缀token数
        new_computed_blocks: KVCacheBlocks | None = None,  # 刚命中的块
        num_lookahead_tokens: int = 0,                 # 投机解码lookahead
        num_external_computed_tokens: int = 0,         # 外部Connector缓存的token
        delay_cache_blocks: bool = False,              # 是否延迟缓存（P/D传输用）
        num_encoder_tokens: int = 0,                   # encoder token数（cross-attn用）
        full_sequence_must_fit: bool = False,          # 全序列必须放得下才准入（准入门控）
        reserved_blocks: int = 0,                      # 为其他in-flight请求保留的块
        has_scheduled_reqs: bool = True,               # 是否已有请求在调度
    ) -> KVCacheBlocks | None:
        """分配槽位，返回新分配的块；如果空间不足返回None"""

        # ========== 参数校验 ==========
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(...)

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # ========== 1. 计算token统计 ==========
        num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # ========== 2. Watermark设置：只对WAITING/PREEMPTED请求生效 ==========
        watermark_blocks = 0
        if has_scheduled_reqs and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            watermark_blocks = self.watermark_blocks

        # ========== 3. 【阶段1】full_sequence_must_fit准入检查 ==========
        # 这是chunked prefill的准入门控：如果整个序列都放不下，直接拒绝，不要只放第一个chunk
        if full_sequence_must_fit:
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,  # 应用准入上限
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None  # 空间不足，拒绝准入

        # ========== 4. 【阶段2】先清理不需要的块（SWA滑动窗口外的）==========
        # 在分配之前先释放，减少需要驱逐的块数
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )
        # 纯FullAttention下这个函数基本什么都不做（没有滑动窗口）

        # ========== 5. 【阶段3】计算本次实际需要分配多少块 ==========
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # ========== 6. 【阶段4】空间检查：可用块 = 空闲块 - 预留块 ==========
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            return None  # 空间不足，需要抢占或等待

        # ========== 7. 【阶段5】两阶段分配（修复issue #33775）==========
        # 关键：必须先touch所有命中块（ref_cnt++），再分配新块！
        # 否则分配新块时可能驱逐还没touch的命中块
        if (new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0):
            # 阶段5a：touch命中块 → ref_cnt++，标记为"正在使用"，不会被驱逐
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        # 阶段5b：真正分配新块 → 从free_block_queue取块，加入manager的req_to_blocks
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )
        # 新分配的块会被加入manager.new_block_ids列表，等下Worker调用take_new_block_ids()来拿去清零

        # ========== 8. P/D延迟缓存：如果是远程传输，先不缓存 ==========
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # ========== 9. 【阶段6】缓存写入（调度阶段，forward之前）==========
        # cache_blocks只缓存"满块"(num_tokens // block_size)，尾块不缓存
        # hash基于token ID，不依赖KV数据，所以forward之前就能算hash
        # 用request.num_tokens来cap，排除可能被拒绝的draft token（只缓存finalized token）
        # 注意：cache_blocks是幂等的——已缓存的块(num_cached_block >= num_full_blocks)直接跳过
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)
        # prompt阶段：前2块已在prefix cache中，num_cached_block=2 >= num_full_blocks=2，是no-op
        # decode阶段：每满一个block_size的块，这里就会把它写入哈希表
        # cache_blocks也会被外部调用方在forward之后追加调用（async PP / KV Connector场景）

        # ========== 10. 返回新分配的块 ==========
        return self.create_kv_cache_blocks(new_blocks)
```

**端到端例子**：34token prompt，命中32token（2块），num_new_tokens=2
- `num_local_computed_tokens = 0 + 32 = 32`（假设是新请求，之前没计算过）
- `num_tokens_need_slot = 32 + 2 = 34`
- `num_blocks_to_allocate = ceil(34/16) - 2 = 3 - 2 = 1`块
- 检查空间：假设空闲块足够
- 阶段5a：touch命中的blockA、blockB → ref_cnt都+1
- 阶段5b：从free_block_queue分配blockC → new_block_ids=[blockC.block_id]
- 阶段6：`num_tokens_to_cache = min(32+2, 34) = 34`
  - `num_full_blocks = 34 // 16 = 2`，`num_cached_block = 2`（prefix hit已缓存前2块）
  - `num_cached_block(2) >= num_full_blocks(2)` → 提前返回，no-op，不写入任何新块
- 返回：`KVCacheBlocks(([blockC],))`

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

用于preempt（抢占）场景：需要先把块弹出来，但不立即归还——等抢占决策确定后再逆序释放。

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

这些方法是**drain模式**：调用一次就把累积的数据取走并清空，Worker拿到数据后在GPU上执行对应的内存操作（清零、拷贝、卸载）。

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

#### 5.5.2 收集COW拷贝 `take_kv_cache_block_copies`

源码位置：`kv_cache_manager.py:831-846`

Copy-on-Write拷贝任务：当多个请求共享同一块，其中一个请求要写这块时，需要先拷贝一份副本。

```python
    def take_kv_cache_block_copies(self) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain：收集待执行的COW拷贝任务"""
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(src_block_id=src.block_id, dst_block_id=dst.block_id)
            for src, dst in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks
        # Worker拿到后在GPU上执行src → dst的KV数据拷贝
```

#### 5.5.3 收集Partial Tail卸载 `take_partial_tail_offloads`

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

## 六、方法调用总览

一个请求从分配到释放，KVCacheManager 的方法按以下顺序被调用：

| 步骤 | 方法 | 说明 |
|------|------|------|
| 步开始 | `new_step_starts()` | 重置内部状态（清空 new_block_ids 等） |
| 前缀查找 | `get_computed_blocks(req)` | 返回命中块和命中 token 数 |
| 分配 | `allocate_slots(req, ...)` | 内部包含准入检查、两阶段分配、cache_blocks |
| Drain | `take_new_block_ids()` / `take_kv_cache_block_copies()` | 给 Worker 准备 GPU 数据 |
| （forward） | — | KVCacheManager 不参与 |
| 补缓存（可选） | `cache_blocks(req, ...)` | async PP / KV Connector 场景外部追加 |
| 释放 | `free(req)` 或 `pop_blocks_for_free(req)` | 正常结束直接释放；抢占先弹出再逆序释放 |

---

## 七、设计要点小结

1. **门面模式**：KVCacheManager是典型的Facade门面，为Scheduler提供一个简化的单一入口，封装了下面4层的所有复杂度
2. **三阶段分配**：`allocate_slots`是核心，逻辑分为：准入检查 → 两阶段touch+allocate → 缓存写入，每一步都有明确的职责
3. **两阶段分配修复竞态**：先`allocate_new_computed_blocks`（touch所有命中块，ref_cnt++防驱逐），再`allocate_new_blocks`（真正分配），这是修复issue #33775的关键
4. **cache_blocks基于token ID计算hash**：不依赖KV数据，所以能在forward之前调用；幂等设计支持被外部调用方在forward之后追加调用（async PP / KV Connector场景）
5. **cache_blocks幂等性**：只缓存满块（`num_tokens // block_size`），已缓存的块（`num_cached_block >= num_full_blocks`）直接跳过，多次调用安全
6. **Drain模式数据准备**：`take_new_block_ids`、`take_kv_cache_block_copies`、`take_partial_tail_offloads`、`take_events`都是"调用即取走并清空"的drain模式，Worker批量拿到后在GPU上执行，CPU/GPU解耦
7. **逆序释放优化**：`pop_blocks_for_free`返回分配顺序的块，上层必须逆序释放，让尾部分配的不完整块优先回到free_block_queue头部，提高下次分配的尾块复用率
8. **Watermark机制**：给WAITING/PREEMPTED请求预留watermark_blocks的空闲块，防止它们进来把空闲块吃光导致正在运行的请求频繁被抢占
9. **GC优化**：预创建`empty_kv_cache_blocks`复用，`create_kv_cache_blocks`工厂方法避免频繁创建空对象
10. **不可变数据协议**：`KVCacheBlocks`使用tuple不可变结构，作为Scheduler和KVCacheManager之间的安全接口，防止内部状态被意外篡改
11. **num_tokens-1细节**：前缀查找时`max_cache_hit_length = num_tokens - 1`，即使全命中也要重算最后一个token的logits，保证输出正确性
12. **投机解码安全**：`cache_blocks`用`request.num_tokens`做cap，只缓存已finalized的token，防止被拒绝的draft token污染前缀缓存
