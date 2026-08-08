# KVCacheCoordinator 源码详解

源码位置：`vllm/v1/core/kv_cache_coordinator.py`

---

## 一、是什么

`KVCacheCoordinator` 是 vLLM v1 调度器中负责**跨多个 KV Cache 组协调**的抽象基类（ABC），位于 KV Cache 五层架构的第四层。

一个模型的不同层可能使用不同的注意力机制（Full Attention、Sliding Window、Mamba 等），每种注意力机制有独立的 `SingleTypeKVCacheManager` 管理自己的 block_table 和缓存命中逻辑。但这些 manager 之间需要协调：
- 前缀缓存命中必须取所有组的**交集**（一个组命中 1000 token，另一个组只命中 800 token，最终只能用 800 token）
- 不同组的 `block_size` 可能不同，需要统一调度粒度（`scheduler_block_size`）
- 分配 block 时必须防止跨组驱逐竞态（一个组分配新块时不能驱逐另一个组还没 touch 的命中块）

`KVCacheCoordinator` 就是这个协调层，它持有一个共享的 `BlockPool` 和一组 `SingleTypeKVCacheManager`，对上层 `KVCacheManager` 暴露统一的 API，把具体工作转发给各 manager，但在跨组一致性这一层做协调。

---

## 二、干什么用

### 2.1 在整体架构中的位置

KV Cache 管理分五层：物理显存层 → `BlockPool`（逻辑块池）→ `SingleTypeKVCacheManager`（单类型管理层）→ **`KVCacheCoordinator`（协调器层，本文）** → `KVCacheManager`（顶层接口）→ Scheduler。

```
┌─────────────────────────────────────────────────────────┐
│  Scheduler (调度器)                                       │
├─────────────────────────────────────────────────────────┤
│  KVCacheManager          ← Scheduler唯一入口              │
├─────────────────────────────────────────────────────────┤
│  KVCacheCoordinator      ← 本文讲这一层                   │
│  协调多组命中交集、统一调度粒度、两阶段分配防竞态            │
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

`KVCacheCoordinator` 的上层是 `KVCacheManager`（Scheduler 的唯一入口，几乎所有逻辑都委派给 Coordinator），下层是一组 `SingleTypeKVCacheManager`（各管一种注意力类型的 block_table 和命中逻辑）。

**Coordinator 不直接管理单条 request→block 的绑定**（那是 SingleTypeKVCacheManager 的职责），只做三件事：
1. **跨组命中取交集**：不同组的命中长度可能不同，必须找到所有组都一致认可的最大公共前缀
2. **统一调度粒度**：`scheduler_block_size` 是所有组 block_size 的 LCM，`hash_block_size` 是 GCD，确保调度边界对齐
3. **跨组分配安全**：两阶段分配（先全组 touch 命中块，再全组分配新块）防止竞态

### 2.2 核心职责（结合调度流程）

一个请求从进入调度器到完成生成，Coordinator 在以下环节介入：

| 调度阶段 | 调用方法 | 协调器的作用 |
|---------|---------|------|
| **前缀缓存查找** | `find_longest_cache_hit()` | 迭代各 manager 查找命中，取交集（不动点算法），返回所有组一致的 hit_length |
| **准入控制** | `get_num_blocks_to_allocate()` | 累加各 manager 所需 block 数（CrossAttention 单独处理），供调度器做 OOM 判断 |
| **注册命中+分配** | `allocate_new_computed_blocks()` | **两阶段分配**：先全组 add_local_computed_blocks（touch 提升 ref_cnt），再全组 allocate_external_computed_blocks，防止跨组驱逐 |
| **分配新块** | `allocate_new_blocks()` | 转发给各 manager，CrossAttention 走 encoder_tokens 分支 |
| **缓存写满块** | `cache_blocks()` | Hybrid 版本对齐到 scheduler_block_size 边界，EAGLE 组多缓存一个 lookahead block |
| **释放/回收** | `free()` / `remove_skipped_blocks()` | 转发给各 manager |
| **级联注意力** | `get_num_common_prefix_blocks()` | 返回各组的公共前缀 block 数列表 |
| **步开始通知** | `new_step_starts()` | 通知各 manager 清空本步临时数据（new_block_ids、CoW copies 等） |

### 2.3 实际场景举例

**场景：混合模型 Gemma3（Full Attention + Sliding Window）**

Gemma3 模型的部分层是 Full Attention（block_size=16），部分层是 Sliding Window Attention（block_size=16, window_size=4096），形成 2 个 KV group。

1. 请求到达，Coordinator 先让 FullAttention manager 查哈希，命中 200 个整块（3200 token）
2. 再让 SWA manager 查，但 SWA 因为 retention_interval 稀疏化，每 N 个 block 才保留一个 checkpoint，在 2500 token 位置 miss
3. 不动点算法迭代：FullAttention 收到收缩后的长度 2500，确认 2500 以内也都命中，收敛
4. 最终 hit_length=2500，FullAttention 返回 157 个块，SWA 返回对应的稀疏块
5. 两阶段分配：先 touch 两组的命中块（ref_cnt++），再统一分配新块——不会出现"A 组分配新块时驱逐了 B 组还没 touch 的命中块"的竞态
6. 返回 `num_uncached_common_prefix_tokens=700`（FullAttention 命中了 3200 但最终只取 2500，差值 700 是 SWA 未缓存但 FullAttention 已命中的公共前缀），上层用这个信息 pin junction 防止 retention 驱逐交汇点

**场景：Llama 同构模型（只有 Full Attention）**

- 只有 1 个 KV group，`UnitaryKVCacheCoordinator` 直接把调用转发给唯一的 manager，没有迭代收敛的开销
- `hash_block_size == block_size`，不需要细粒度哈希转换

**场景：禁用前缀缓存**

- 用户设置 `enable_prefix_caching=False`，`KVCacheCoordinatorNoPrefixCache` 返回空命中，`get_num_common_prefix_blocks` 恒返回 0，跳过所有前缀缓存相关逻辑，支持任意数量 group（包括 0）

---

## 三、类继承结构

### 3.1 类层次

```
KVCacheCoordinator (ABC)                          ← 抽象基类，定义接口和公共逻辑
├── KVCacheCoordinatorNoPrefixCache               ← 禁用前缀缓存时使用
├── UnitaryKVCacheCoordinator                     ← 单KV group（同构模型）
└── HybridKVCacheCoordinator                      ← 多KV group（混合模型）
```

### 3.2 工厂函数 `get_kv_cache_coordinator`

根据配置自动选择合适的 Coordinator，决策顺序固定：

```python
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,                          # KV缓存配置
    max_model_len: int,                                      # 模型最大序列长度
    max_in_flight_tokens: int,                               # 最大在飞token数
    use_eagle: bool,                                         # 是否启用EAGLE/MTP投机解码
    enable_caching: bool,                                    # 是否启用前缀缓存
    enable_kv_cache_events: bool,                            # 是否启用KV缓存事件
    dcp_world_size: int,                                     # DCP world size
    pcp_world_size: int,                                     # PCP world size
    scheduler_block_size: int,                               # 调度对齐粒度
    hash_block_size: int,                                    # 哈希粒度
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(...)          # 优先级最高：禁用缓存
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(...)                # 单group
    return HybridKVCacheCoordinator(...)                     # 多group（≥2）
```

| Coordinator | 适用场景 | 关键特征 |
|------------|---------|---------|
| `NoPrefixCache` | `enable_prefix_caching=False` 或后端不支持 | 支持任意 group 数（含0）；`find_longest_cache_hit` 恒返回空；无缓存相关开销 |
| `Unitary` | 仅 1 个 KV group（Llama/Qwen/Mistral 等同构模型） | 断言 `hash_block_size == block_size`；命中查找直接委派，无迭代 |
| `Hybrid` | ≥2 个 KV group（Gemma3/LLaMA4/Samba 等混合模型） | SpecGroup 合并同 spec group；迭代不动点取交集；`num_uncached_common_prefix_tokens` 输出 |

### 3.3 辅助函数 `_validate_prefix_cache_retention_interval`

构造函数中调用，校验稀疏保留间隔的合法性：

```python
def _validate_prefix_cache_retention_interval(
    retention_interval: int | None,                          # 保留间隔：None=稠密，0=只保留边界，>0=间隔N token保留
    scheduler_block_size: int,                               # 调度对齐粒度
    kv_cache_config: KVCacheConfig,                          # KV缓存配置
) -> None:
    if retention_interval is None:
        return

    # retention只对SWA和Mamba类型生效；FullAttention/ChunkedLocal是稠密缓存，设置了会报错
    if not any(
        isinstance(g.kv_cache_spec, (SlidingWindowSpec, MambaSpec))
        for g in kv_cache_config.kv_cache_groups
    ):
        raise ValueError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL is set but this model has "
            "no sliding-window or Mamba KV cache group, so retention has no "
            "effect. Unset it (it only applies to sliding-window and Mamba "
            "attention)."
        )

    # retention_interval必须是非负且scheduler_block_size的倍数，确保落在真实缓存命中边界上
    if retention_interval < 0 or retention_interval % scheduler_block_size != 0:
        raise ValueError(
            f"VLLM_PREFIX_CACHE_RETENTION_INTERVAL ({retention_interval}) "
            "must be non-negative and a multiple of scheduler_block_size "
            f"({scheduler_block_size})."
        )
```

### 3.4 SpecGroup（Hybrid 专用）

`SpecGroup` 是 Hybrid 内部用来合并"spec 完全相同"的 group 的 NamedTuple：

```python
class SpecGroup(NamedTuple):
    spec: KVCacheSpec                                        # 合并代表的spec
    group_ids: list[int]                                     # 该spec覆盖的所有group_id
    manager_cls: type[SingleTypeKVCacheManager]              # 同spec必须同manager类
    use_eagle: bool                                          # 任一成员是EAGLE group即为True
```

相同 spec 的 group 合并后联合查找缓存命中（一次 find_longest_cache_hit 调用查出所有同 spec group 的结果），减少重复哈希表查找。

---

## 四、KVCacheCoordinator 基类详解

`KVCacheCoordinator`（`kv_cache_coordinator.py:60-382`）是抽象基类，实现了构造逻辑和大部分公共方法，只有 `find_longest_cache_hit` 是抽象方法交给子类实现。

### 4.1 `__init__` 构造函数（65-128行）

```python
def __init__(
    self,
    kv_cache_config: KVCacheConfig,                          # KV缓存配置（含各组spec、num_blocks等）
    max_model_len: int,                                      # 模型最大序列长度
    max_in_flight_tokens: int,                               # 最大在飞token数（控制batch）
    use_eagle: bool,                                         # 是否启用EAGLE/MTP投机解码
    enable_caching: bool,                                    # 是否启用前缀缓存
    enable_kv_cache_events: bool,                            # 是否启用KV缓存事件（用于P/D分离、offload）
    dcp_world_size: int,                                     # Decode Context Parallelism world size
    pcp_world_size: int,                                     # Prefill Context Parallelism world size
    scheduler_block_size: int,                               # 跨组调度对齐粒度（各group block_size的LCM）
    hash_block_size: int,                                    # 哈希计算粒度（各group block_size的GCD）
    metrics_collector: KVCacheMetricsCollector | None = None,
):
    self.kv_cache_config = kv_cache_config
    self.max_model_len = max_model_len
    self.enable_caching = enable_caching

    # 调度对齐约束：scheduler_block_size必须是hash_block_size和每个group block_size的公倍数
    assert scheduler_block_size % hash_block_size == 0 and all(
        scheduler_block_size % g.kv_cache_spec.block_size == 0
        for g in kv_cache_config.kv_cache_groups
    )
    self.scheduler_block_size = scheduler_block_size

    # 创建共享的BlockPool——所有single_type_managers共用同一个池子
    self.block_pool = BlockPool(
        num_gpu_blocks=kv_cache_config.num_blocks,            # GPU可用block总数
        enable_caching=enable_caching,
        hash_block_size=hash_block_size,
        enable_kv_cache_events=enable_kv_cache_events,
        metrics_collector=metrics_collector,
    )

    # EAGLE/MTP投机解码：需要丢弃最后一个命中块的group集合
    # draft head的hidden states在最后一个hash粒度token上，多匹配一个block再丢掉
    self.eagle_group_ids: set[int] = {
        i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
    }
    # 保守回退：use_eagle=True但没有group标记为eagle时，所有group都启用drop
    if use_eagle and not self.eagle_group_ids:
        self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

    # 为每个KV group创建对应的SingleTypeKVCacheManager实例
    self.single_type_managers = tuple(
        get_manager_for_kv_cache_spec(                        # 工厂函数：根据spec类型选manager子类
            kv_cache_spec=kv_cache_group.kv_cache_spec,
            max_in_flight_tokens=max_in_flight_tokens,
            max_model_len=max_model_len,
            block_pool=self.block_pool,                       # 所有manager共享同一个BlockPool
            enable_caching=enable_caching,
            kv_cache_group_id=i,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=self.scheduler_block_size,
            needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
        )
        for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
    )

    # 稀疏保留间隔：对SWA/Mamba类型每retention_interval个token保留一个checkpoint
    # 0=只保留最新边界；None=稠密保留（不稀疏化）；FullAttention/ChunkedLocal忽略此参数
    self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
    _validate_prefix_cache_retention_interval(               # 校验合法性
        self.retention_interval, self.scheduler_block_size, kv_cache_config
    )
```

### 4.2 `get_num_blocks_to_allocate`（130-190行）

计算当前请求需要新分配的 block 总数，遍历所有 manager 累加。

```python
def get_num_blocks_to_allocate(
    self,
    request_id: str,                                         # 请求ID
    num_tokens: int,                                         # 需要slot的总token数（含已分配的）
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...], # 刚命中前缀缓存的blocks（按group）
    num_encoder_tokens: int,                                 # encoder输入token数（cross-attention用）
    total_computed_tokens: int,                              # 本地+外部命中token总数
    num_local_computed_tokens: int,                          # 本地前缀缓存命中token数
    num_tokens_main_model: int,                              # 主模型token数（spec decode时减去lookahead）
    apply_admission_cap: bool = False,                       # 是否应用per-request准入上限（SWA/ChunkedLocal）
) -> int:
    num_blocks_to_allocate = 0
    for i, manager in enumerate(self.single_type_managers):
        if isinstance(manager, CrossAttentionManager):
            # Cross-attention单独处理：基于encoder token数一次性静态分配
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id,
                num_encoder_tokens,                          # cross-attn用num_encoder_tokens而非num_tokens
                [],                                          # cross-attn无前缀命中
                0,
                0,
                num_encoder_tokens,
                apply_admission_cap=apply_admission_cap,
            )
        else:
            num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks[i],                      # 传入该group刚命中的blocks
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
    return num_blocks_to_allocate
```

**关键点**：`CrossAttentionManager` 单独走分支，因为它的 block 数取决于 encoder 输入长度，不是 decoder 序列长度。

### 4.3 `allocate_new_computed_blocks`（192-236行）——两阶段分配（修复 issue #33775）

这是基类中最重要的方法之一，实现了两阶段分配来修复跨组驱逐竞态。

```python
def allocate_new_computed_blocks(
    self,
    request_id: str,
    new_computed_blocks: tuple[Sequence[KVCacheBlock], ...], # 各group刚命中的blocks
    num_local_computed_tokens: int,                          # 本地命中token数
    num_external_computed_tokens: int,                       # 外部connector命中token数
) -> None:
    # RUNNING请求已经在num_cached_block中跟踪，不会有新的前缀命中，直接返回
    if any(
        request_id in manager.num_cached_block
        for manager in self.single_type_managers
    ):
        assert all(len(blocks) == 0 for blocks in new_computed_blocks)
        return

    # ===== Phase 1: 所有group先touch本地命中块 =====
    # touch会让ref_cnt++，把block从空闲队列中救出，防止被驱逐
    # 必须先完成所有group的touch，再做任何可能触发驱逐的分配
    for i, manager in enumerate(self.single_type_managers):
        manager.add_local_computed_blocks(
            request_id,
            new_computed_blocks[i],
            num_local_computed_tokens,
            num_external_computed_tokens,
        )

    # ===== Phase 2: 所有group再分配外部blocks =====
    # 外部blocks需要从空闲队列get_new_blocks，可能触发LRU驱逐
    # 此时所有命中块的ref_cnt已经>0，不会被误驱逐
    if num_external_computed_tokens > 0:
        for manager in self.single_type_managers:
            manager.allocate_external_computed_blocks(
                request_id,
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
```

**为什么必须两阶段？**

旧实现按 group 逐个分配：group 0 touch+分配 → group 1 touch+分配。问题在于：group 0 分配外部块时，可能从空闲队列弹出并驱逐一个 block——这个 block 恰好是 group 1 还没 touch 的命中块（此时它的 ref_cnt==0，是合法驱逐候选）。

**竞态时序（旧实现，有bug）**：
1. group 0 命中块 A，group 1 命中块 B
2. 先处理 group 0：touch A（ref_cnt[A]=1）→ 分配外部块 C → 空闲队列不足，LRU 选中 B（ref_cnt[B]==0）驱逐 → B 的内容被覆盖
3. 再处理 group 1：touch B，但 B 已经被驱逐并覆盖了 → 命中失效，KV 损坏

**两阶段修复后**：
1. Phase 1：touch A（ref_cnt[A]=1）、touch B（ref_cnt[B]=1）→ 两个命中块都被"钉住"
2. Phase 2：group 0 分配 C → 即使触发驱逐，B 的 ref_cnt==1 不会被选中 → group 1 分配 D → 安全

两阶段把"全组 touch"和"全部分配"拆开，确保先把所有命中块的 ref_cnt 都抬升后再做可能触发驱逐的分配，杜绝了这个竞态。

### 4.4 其他公共方法（238-382行）

基类的其他方法都是循环转发给各 manager，逻辑直接：

```python
def allocate_new_blocks(
    self,
    request_id: str,
    num_tokens: int,                                         # decoder序列总token数
    num_tokens_main_model: int,                              # 主模型token数（spec decode用）
    num_encoder_tokens: int = 0,                             # encoder token数（cross-attn用）
) -> tuple[list[KVCacheBlock], ...]:
    # 给每个manager分配新blocks，CrossAttention用num_encoder_tokens
    return tuple(
        manager.allocate_new_blocks(
            request_id,
            num_encoder_tokens
            if isinstance(manager, CrossAttentionManager)
            else num_tokens,
            num_tokens_main_model,
        )
        for manager in self.single_type_managers
    )
```

```python
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    # 转发给各manager缓存写满的block（Hybrid子类会覆写此方法加对齐逻辑）
    for manager in self.single_type_managers:
        manager.cache_blocks(
            request,
            num_computed_tokens,
            retention_interval=self.retention_interval,      # 传入稀疏保留间隔
        )
```

```python
def free(self, request_id: str) -> None:
    # 请求结束/被抢占时，释放各manager中持有的blocks
    for manager in self.single_type_managers:
        manager.free(request_id)
```

```python
def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
    # 弹出请求在所有manager中的blocks但不归还池（调用方负责逆序free）
    # 逆序free保证tail blocks先被驱逐，符合LRU语义
    blocks: list[KVCacheBlock] = []
    for manager in self.single_type_managers:
        blocks.extend(manager.pop_blocks_for_free(request_id))
    return blocks
```

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
    # 返回各group的公共前缀block数列表，用于cascaded attention优化
    return [
        manager.get_num_common_prefix_blocks(running_request_id)
        for manager in self.single_type_managers
    ]
```

```python
def remove_skipped_blocks(
    self,
    request_id: str,
    processed_computed_tokens: int,                          # 已完全处理并提交的token前缀长度
    num_prompt_tokens: int | None = None,                    # prompt长度（R-SWA用）
) -> None:
    # 移除注意力窗口外的block（SWA/Mamba），替换为null_block
    for manager in self.single_type_managers:
        manager.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )
```

```python
def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
    # 获取请求在各manager中的block列表
    return tuple(
        manager.req_to_blocks.get(request_id) or []
        for manager in self.single_type_managers
    )
```

```python
@abstractmethod
def find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],                           # 请求的token哈希链
    max_cache_hit_length: int,                               # 最大查找长度
) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
    # 抽象方法：子类实现具体的命中查找策略
    # 返回: (各group命中blocks, 协调后的hit_length, num_uncached_common_prefix_tokens)
    pass
```

```python
def new_step_starts(self) -> None:
    # 通知各manager新step开始，清空本步临时数据（new_block_ids、CoW copies等）
    for manager in self.single_type_managers:
        manager.new_step_starts()
```

---

## 五、KVCacheCoordinatorNoPrefixCache 详解

`KVCacheCoordinatorNoPrefixCache`（`kv_cache_coordinator.py:385-432`）用于禁用前缀缓存的场景。

### 5.1 构造函数（393-419行）

```python
def __init__(
    self,
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
):
    super().__init__(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        False,                                                 # 硬编码enable_caching=False
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
    self.num_single_type_manager = len(self.single_type_managers)
```

唯一特殊点：调用父类构造时硬编码 `enable_caching=False`，且不接收 `enable_caching` 参数。这会让 BlockPool 不创建前缀哈希表，所有 manager 也不做缓存相关操作。

### 5.2 覆写的方法

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
    return [0] * self.num_single_type_manager    # 无缓存，公共前缀恒为0
```

```python
def find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
    blocks: tuple[list[KVCacheBlock], ...] = tuple(
        [] for _ in range(self.num_single_type_manager)
    )
    return blocks, 0, 0                           # 空命中，hit_length=0，uncached=0
```

**特点**：
- 支持任意数量 KV group（包括 0 个 group 的情况）
- 前缀缓存完全关闭，所有命中查找返回空，公共前缀恒为 0
- 分配、释放、回收等基类方法正常工作，只是不做缓存复用
- 性能最优：无哈希表查找、无迭代、无 CoW 开销

---

## 六、UnitaryKVCacheCoordinator 详解

`UnitaryKVCacheCoordinator`（`kv_cache_coordinator.py:435-503`）用于只有 1 个 KV group 的同构模型（Llama、Qwen、Mistral 等标准全注意力模型）。

### 6.1 构造函数（442-484行）

```python
def __init__(
    self,
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
):
    super().__init__(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
    self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec  # 唯一group的spec
    self.block_size = self.kv_cache_spec.block_size
    self.dcp_world_size = dcp_world_size
    self.pcp_world_size = pcp_world_size
    if dcp_world_size > 1:
        self.block_size *= dcp_world_size          # DCP多卡时block_size乘world_size（跨卡分片）

    # 单group断言：禁用缓存或hash_block_size必须等于block_size（无细粒度哈希需求）
    # Mamba禁用缓存时block_size被设为max_model_len，跳过此校验
    assert not enable_caching or (hash_block_size == self.block_size), (
        "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
    )
    assert len(self.kv_cache_config.kv_cache_groups) == 1, (
        "UnitaryKVCacheCoordinator assumes only one kv cache group"
    )
    # 单group；设置use_eagle标志保持一致性
    self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids
```

**关键约束**：单 group 模式下 `hash_block_size == block_size`，不需要 GCD/LCM 换算，也没有细粒度部分命中。`scheduler_block_size` 也自然等于 `block_size`。

### 6.2 `find_longest_cache_hit`（486-503行）

```python
def find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
    # 直接委派给唯一的manager，无迭代收敛开销
    hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
        block_hashes=block_hashes,
        max_length=max_cache_hit_length,
        kv_cache_group_ids=[0],                                    # 只有group 0
        block_pool=self.block_pool,
        kv_cache_spec=self.kv_cache_spec,
        drop_eagle_block=0 in self.eagle_group_ids,
        alignment_tokens=self.block_size,                          # 对齐到物理block_size
        dcp_world_size=self.dcp_world_size,
        pcp_world_size=self.pcp_world_size,
    )
    # 单group没有"uncached common prefix"——没有其他group来lag（滞后），恒为0
    return hit_blocks, hit_length, 0
```

单 group 场景下不存在跨组协调问题，直接一次调用拿到结果，`num_uncached_common_prefix_tokens` 恒为 0。这是最常见的情况（Llama 系模型都走这里）。

---

## 七、HybridKVCacheCoordinator 详解

`HybridKVCacheCoordinator`（`kv_cache_coordinator.py:521-848`）是最复杂的实现，用于 ≥2 个 KV group 的混合模型（Gemma3、LLaMA4、Samba 等）。核心难点是**不同组的命中长度可能不同，必须取交集**。

### 7.1 构造函数（527-589行）

```python
def __init__(
    self,
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
):
    super().__init__(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
    self.hash_block_size = hash_block_size                       # 哈希计算粒度（GCD）
    self.dcp_world_size = dcp_world_size

    # 校验：每个group的block_size必须能被hash_block_size整除
    group_block_sizes = [
        manager.block_size for manager in self.single_type_managers
    ]
    assert all(
        block_size % hash_block_size == 0 for block_size in group_block_sizes
    ), (
        "Each KV cache group's real block_size must be divisible by "
        f"hash_block_size. block_sizes={group_block_sizes}, "
        f"hash_block_size={hash_block_size}"
    )

    assert pcp_world_size == 1, "PCP not support hybrid attn now."  # PCP暂不支持混合注意力

    if dcp_world_size > 1:
        # DCP多卡目前只支持FullAttention + Mamba混合，其他类型（如SWA）拒绝
        for g in kv_cache_config.kv_cache_groups:
            assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (
                "DCP with hybrid KV cache layouts only supports "
                "full-attention and Mamba groups, got: "
                f"{type(g.kv_cache_spec).__name__}."
            )

    # 部分哈希命中（细粒度）启用条件：
    # 单卡 + 存在Mamba(align模式)且其block_size > hash_block_size
    # 此时Mamba可以在hash_block_size粒度命中，而非必须在物理block边界
    self.enable_partial_hash_hits = dcp_world_size == 1 and any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_cache_mode == "align"
        and g.kv_cache_spec.block_size > hash_block_size
        for g in kv_cache_config.kv_cache_groups
    )

    self.verify_and_split_kv_cache_groups()    # 分组：把相同spec的group合并成SpecGroup
```

### 7.2 `_cache_hit_alignment_tokens` 属性（591-599行）

```python
@property
def _cache_hit_alignment_tokens(self) -> int:
    # 缓存命中返回长度的对齐粒度：
    # - 细粒度部分命中时：hash_block_size（允许更小粒度的命中）
    # - 否则：scheduler_block_size（必须对齐到调度边界）
    return (
        self.hash_block_size
        if self.enable_partial_hash_hits
        else self.scheduler_block_size
    )
```

### 7.3 `verify_and_split_kv_cache_groups`（601-650行）——SpecGroup 分组

将相同 spec 的 group 合并为 SpecGroup，并排序。

```python
def verify_and_split_kv_cache_groups(self) -> None:
    self.attention_groups: list[SpecGroup] = []
    for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
        manager_cls = self.single_type_managers[i].__class__
        spec = g.kv_cache_spec
        use_eagle = i in self.eagle_group_ids

        # 查找是否已有相同spec的group，有则追加group_id
        for idx, group in enumerate(self.attention_groups):
            if group.spec == spec:
                assert manager_cls is group.manager_cls, (
                    "Expected same manager class for identical KV cache specs."
                )
                group.group_ids.append(i)
                # 任一成员是eagle group，则整个SpecGroup标记为use_eagle=True
                if use_eagle and not group.use_eagle:
                    self.attention_groups[idx] = group._replace(use_eagle=True)
                break
        else:
            # 没有相同spec，创建新的SpecGroup
            self.attention_groups.append(
                SpecGroup(spec, [i], manager_cls, use_eagle)
            )

    assert len(self.attention_groups) > 1, (
        "HybridKVCacheCoordinator requires at least two attention groups."
    )

    # FullAttention排首位：左→右扫描提供紧下界，减少后续group的工作量
    # 排序key：not isinstance(FullAttentionSpec) → False=0排前面，True=1排后面
    self.attention_groups.sort(
        key=lambda g: not isinstance(g.spec, FullAttentionSpec)
    )

    # 记录full_attention_group_id作为稠密参照（如果存在FullAttention）
    # FullAttention向下封闭（miss后全miss），其他group报告更长命中说明边界不一致（issue #46453）
    first = self.attention_groups[0]
    self.full_attention_group_id: int | None = (
        first.group_ids[0] if isinstance(first.spec, FullAttentionSpec) else None
    )

    # 传播eagle标记到各manager（默认use_eagle=False）
    for group in self.attention_groups:
        if group.use_eagle:
            for gid in group.group_ids:
                self.single_type_managers[gid].use_eagle = True
```

**为什么 FullAttention 排首位？**
- FullAttention 的链式哈希保证"miss 之后全 miss"，它给出的命中长度是**最紧的上界**
- 后续 group 在这个上界内查找，不需要扫描更长的范围
- 如果其他 group 报告了比 FullAttention 更长的命中，说明存在不一致（issue #46453），FullAttention 作为稠密参照可以检测这种情况

### 7.4 `cache_blocks`（652-683行）——对齐缓存边界

```python
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    if self.enable_partial_hash_hits:
        aligned_num_computed_tokens = num_computed_tokens    # 细粒度模式：不做对齐
    else:
        # 对齐到scheduler_block_size边界（只缓存完整scheduler block）
        # SWA组在每个scheduler_block内可能只访问部分block，未访问的block不加入哈希表
        aligned_num_computed_tokens = (
            num_computed_tokens
            // self.scheduler_block_size
            * self.scheduler_block_size
        )

    for manager in self.single_type_managers:
        num_tokens_to_cache = aligned_num_computed_tokens
        # EAGLE组：多缓存一个lookahead block（多匹配一个drop unit后丢弃）
        # 这样下次查找时可以多探一个block再drop回来，保证draft head的hidden states位置正确
        if manager.use_eagle and aligned_num_computed_tokens > 0:
            num_tokens_to_cache = min(
                num_computed_tokens,
                aligned_num_computed_tokens + manager.block_size,
            )
        # manager已知道细粒度命中粒度（scheduler_block_size）；retention单独传入
        manager.cache_blocks(
            request,
            num_tokens_to_cache,
            retention_interval=self.retention_interval,
        )
```

### 7.5 `find_longest_cache_hit`（685-817行）——迭代不动点算法（核心）

这是整个 Coordinator 最复杂的方法，解决多 group 命中长度不一致的问题。

#### 算法思想

每个 SpecGroup 要么接受当前候选长度，要么将其缩短。如果任何 group 缩短了长度，所有 group 必须在新长度下重新检查。因为长度单调递减且下界为 0，算法必然收敛。

#### 逐段详解

**初始化**（710-725行）：

```python
num_groups = len(self.kv_cache_config.kv_cache_groups)
hit_length = max_cache_hit_length                              # 初始上界 = 最大可能命中
longest_hit_length = 0                                         # 历史最深命中（用于uncached检测）
hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups   # 各group命中blocks缓存
hit_length_by_group: list[int] = [0] * num_groups              # 各group命中长度缓存

# 快速路径：1 Full + 1 Other的简单混合，一轮迭代即可收敛
# Full attn总是排在首位（如果存在）
is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
    self.attention_groups[0].spec, FullAttentionSpec
)

# EAGLE验证集合：同一candidate length下每个eagle group最多drop一次
# length收缩时清空重建（issue #32802：防止在收缩后的长度错误复用之前的drop结果）
eagle_verified: set[int] = set()
```

**迭代主循环**（727-796行）：

```python
while True:
    curr_hit_length = hit_length                              # 本轮起始候选长度

    for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
        self.attention_groups
    ):
        first_group_id = group_ids[0]
        # DCP/PCP跨卡分片时，manager的有效block_size可能超过spec定义的
        group_block_size = self.single_type_managers[first_group_id].block_size
        cached_blocks = hit_blocks_by_group[first_group_id]

        # ===== FullAttention向下封闭优化 =====
        # FullAttention的链式哈希保证：首轮查找后，后续迭代只需trim到新长度，无需重复扫描哈希表
        # 因为如果长度L处全命中，那么所有≤L的位置也必定命中（向下封闭性）
        if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
            curr_hit_length = min(
                curr_hit_length, hit_length_by_group[first_group_id]
            )
            continue                                           # 跳过重复查找

        # ===== EAGLE drop处理 =====
        drop_eagle_block = use_eagle and idx not in eagle_verified

        _max_length = curr_hit_length
        # EAGLE多探一个drop unit再丢掉（细粒度时是hash_block_size，否则是group_block_size）
        # 这样draft head的hidden states落在多探的那个block上，drop后刚好对齐到curr_hit_length
        # Mamba不加margin：finder从不drop（draft model无mamba层），否则命中会越过candidate
        if drop_eagle_block and not isinstance(spec, MambaSpec):
            eagle_margin = (
                self.hash_block_size
                if self.enable_partial_hash_hits
                and manager_cls.supports_fine_grained_hash_lookup
                and group_block_size > self.hash_block_size
                else group_block_size
            )
            _max_length = min(
                curr_hit_length + eagle_margin, max_cache_hit_length
            )

        # ===== 调用manager查找命中 =====
        hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=_max_length,                             # 查找上界（可能含eagle margin）
            kv_cache_group_ids=group_ids,                      # 同SpecGroup的所有group_id一起查
            block_pool=self.block_pool,
            kv_cache_spec=spec,
            drop_eagle_block=drop_eagle_block,
            alignment_tokens=self._cache_hit_alignment_tokens,
            dcp_world_size=(                                   # 只有FullAttention支持DCP分片
                self.dcp_world_size
                if isinstance(spec, FullAttentionSpec)
                else 1
            ),
        )

        # ===== 更新EAGLE验证状态 =====
        if drop_eagle_block:
            eagle_verified.add(idx)                            # 当前length下该group的drop已验证
        elif _new_hit_length < curr_hit_length:
            eagle_verified.clear()                             # length收缩→之前的验证作废，需重新验证

        curr_hit_length = _new_hit_length
        # 缓存本轮结果，供后续迭代/向下封闭优化使用
        for group_id, blocks in zip(group_ids, hit_blocks):
            hit_blocks_by_group[group_id] = blocks
            hit_length_by_group[group_id] = _new_hit_length

        longest_hit_length = max(longest_hit_length, curr_hit_length)

    # ===== 收敛判断 =====
    if curr_hit_length >= hit_length:
        break                                                  # 不动点：本轮无收缩→收敛
    hit_length = curr_hit_length
    if is_simple_hybrid:
        break                                                  # 简单混合一轮即可，无需再迭代
```

**收敛后处理**（798-817行）：

```python
# 截断FullAttention的blocks到最终hit_length
# 因为FullAttention在后续迭代中被continue跳过，它的hit_blocks还是首轮长度的结果
first_group = self.attention_groups[0]
if isinstance(first_group.spec, FullAttentionSpec):
    group_block_size = self.single_type_managers[
        first_group.group_ids[0]
    ].block_size
    num_blocks = cdiv(hit_length, group_block_size)
    for group_id in first_group.group_ids:
        if (blks := hit_blocks_by_group[group_id]) is not None:
            del blks[num_blocks:]                              # 删除超出最终长度的blocks
            hit_length_by_group[group_id] = hit_length

# 计算uncached common prefix：
# longest_hit_length是任何group曾经命中的最长长度
# hit_length是所有group协调后的最终长度
# 差值表示：某些稠密组（如FullAttention）缓存了更长的前缀，但稀疏保留组（SWA/Mamba）未缓存到那里
# 上层用这个值pin junction，防止retention interval驱逐交汇点导致的缓存不一致
num_uncached_common_prefix_tokens = longest_hit_length - hit_length

cache_hit_blocks = tuple(
    blocks if blocks is not None else [] for blocks in hit_blocks_by_group
)
return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens
```

#### 算法图示

假设模型有 FullAttention 和 SWA 两个 group，max_cache_hit_length=3200 token：

```
第1轮迭代:
  FullAttention: 扫描0~3200，命中3200 token（全命中）→ curr=3200, 缓存结果
  SWA:           扫描0~3200，因retention稀疏化在2500 miss → curr=2500
  curr(2500) < hit(3200) → 未收敛，hit=2500

第2轮迭代:
  FullAttention: 向下封闭优化，直接取min(2500, 3200)=2500 → curr=2500, continue（不重复扫描）
  SWA:           扫描0~2500（长度没变，上次结果就是2500）→ curr=2500
  curr(2500) >= hit(2500) → 收敛！

结果: hit_length=2500, longest_hit_length=3200
      num_uncached_common_prefix_tokens = 3200 - 2500 = 700
      （FullAttention缓存了3200，但SWA只缓存到2500，700是SWA未缓存的共享前缀）
```

**快速路径优化**：简单混合（1 Full + 1 Other）第一轮后直接 break，不需要第二轮。因为 Full 给出初始上界、Other 收缩后，再迭代 Full 只会取 min（不会继续收缩），Other 在收缩后的长度下结果也不变。

**FullAttention 向下封闭**：FullAttention 首轮查找后，后续迭代不需要重新扫描哈希表——链式哈希保证如果在长度 L 处命中，那么所有 ≤L 的位置也必定命中（"向下封闭"性质），所以直接取 min trim 即可。

**为什么需要 `longest_hit_length`？**

在多 group（≥3）场景下，可能出现：group A 命中 3000，group B 命中 2500，group C 命中 2800。协调后 hit_length=2500，但 group A 实际缓存到了 3000。`longest_hit_length - hit_length = 500` 表示有 500 token 的前缀是部分 group 已缓存但未被所有组共同命中的，上层用这个信息防止 retention 策略错误地驱逐交汇点。

### 7.6 `find_longest_cache_hit_per_group`（819-848行）——各组独立查找

```python
def find_longest_cache_hit_per_group(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
    """Like find_longest_cache_hit but evaluates each group independently.
    各组独立查找，不做交集收敛，用于connector对账/调试。

    Returns:
        (blocks_per_group, hit_lengths_per_group)
    """
    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
    hit_lengths: list[int] = [0] * num_groups

    for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
        blocks, group_hit = manager_cls.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=group_ids,
            block_pool=self.block_pool,
            kv_cache_spec=spec,
            drop_eagle_block=use_eagle,
            alignment_tokens=self._cache_hit_alignment_tokens,
        )
        for gid, blks in zip(group_ids, blocks):
            hit_blocks[gid] = blks
            hit_lengths[gid] = group_hit

    return tuple(hit_blocks), tuple(hit_lengths)
```

与 `find_longest_cache_hit` 的区别：这个方法不做迭代收敛，各组独立查找返回自己的结果，供 KV connector（如 MooncakeConnector）做对账使用，检查分布式缓存和本地缓存的一致性。

---

## 八、三种 block_size 的关系

Hybrid 场景下存在三种 block 粒度，理解它们的关系对读懂代码至关重要：

| 粒度 | 计算方式 | 用途 |
|-----|---------|------|
| `hash_block_size` | 所有 group block_size 的 **GCD**（最大公约数） | 链式哈希计算的最小单位；细粒度命中的对齐点 |
| `group.block_size` | 各 attention 类型自己的物理 block 大小 | 单个 manager 管理的物理 KV block 大小（如 FullAttention=16, Mamba=256） |
| `scheduler_block_size` | 所有 group block_size 的 **LCM**（最小公倍数） | 调度器统一对齐粒度；cache_blocks 缓存边界；非细粒度模式下的命中对齐 |

```
示例（Gemma3: FullAttention block_size=16, SWA block_size=16）:
  GCD(16,16) = 16, LCM(16,16) = 16
  → 三种粒度相同，无换算

示例（Gemma3 某变种: FullAttention block_size=16, Mamba align block_size=64）:
  GCD(16,64) = 16, LCM(16,64) = 64
  hash_block_size=16, group block_size分别为16/64, scheduler_block_size=64
  → Mamba细粒度模式下可以每16 token一个哈希命中，但缓存边界对齐到64
```

---

## 九、设计要点小结

1. **协调器只管跨组**：单组 request↔block 绑定交给 `SingleTypeKVCacheManager`，协调器只做「跨组命中取交集」「调度粒度对齐」「跨组 touch-ahead 防竞态」
2. **工厂决策顺序固定**：`NoPrefixCache` → `Unitary` → `Hybrid`，禁用缓存优先级最高，单 group 走快速路径
3. **SpecGroup 合并 + FullAttention 首位**：同 spec 的 group 联合查找避免重复扫描；FullAttention 的左→右扫描给出紧下界，减少后续 group 工作量
4. **迭代不动点收敛**：`hit_length` 单调不增且下界为 0，必然收敛；FullAttention 向下封闭、simple-hybrid 快速路径、EAGLE 验证集合共同把常见路径压缩到一轮
5. **两阶段分配修复 #33775**：先全组 touch（ref_cnt++），再全部分配外部 block，杜绝跨组驱逐竞态——这是基类最重要的安全设计
6. **三种 block_size 协同**：`scheduler_block_size = LCM`（调度对齐）、`hash_block_size = GCD`（哈希粒度），通过整除关系保证边界对齐
7. **`num_uncached_common_prefix_tokens`**：稀疏保留组（SWA/Mamba）尚未缓存但其他组已命中的共享前缀长度，供上层 pin junction 防止 retention interval 驱逐交汇点
8. **共享单一 BlockPool**：所有 manager 共用同一个池子，跨组分配从同一池取块，空间全局可见、全局 LRU
9. **EAGLE drop 机制**：eagle group 多探一个 margin 再丢掉，确保 draft head 的 hidden states 落在正确位置；同一候选长度下每个 eagle group 只验证一次，长度收缩后清空验证状态
