# KVCacheCoordinator 详解

> 五层架构第 4 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) ｜上层 ➔ [`5_kv_cache_manager.md`](./5_kv_cache_manager.md)
> 时序位置：[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) ③前缀查找（find_longest_cache_hit）、④分配与缓存（两阶段分配 + cache_blocks）、⑧释放（free / pop_blocks_for_free）
>
> 源文件：`vllm/vllm/v1/core/kv_cache_coordinator.py`
>
> 主线：纯 Full Attention 单 group → `UnitaryKVCacheCoordinator`（透传层）。**本文以 Llama-3-8B pp2tp2（4 卡）+ 示例请求 R 为统一锚点：§2 先立好全部示例数值，§5 基类每个方法、§6 子类每个方法都带"示例 R 中发生了什么"参考块。**

## 1. 是什么

`KVCacheCoordinator` 是五层 KV Cache 管理架构中的**第四层——跨组协调层**。

下钻链的位置：`KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager → BlockPool`。上层的 KVCacheManager 只跟 Coordinator 对话；Coordinator 负责把请求**按组拆分、分发给各组的 SingleTypeManager**，并把各组的 BlockPool 统一建好。

对于纯 Full Attention 模型（Llama、Qwen、Mistral 等），所有层同类型，只会分成**一个 KV 组**，此时使用最简单的子类 `UnitaryKVCacheCoordinator`——基本是个"透传层"，把请求直接转发给下层的 `FullAttentionManager`；基类中统一创建 `BlockPool`。其他子类（NoPrefix、Hybrid）服务多组混合场景，本文 §7 一句话带过。

## 2. 统一示例设定（全文锚点）

以下设定在本文所有代码讲解中反复使用，与 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) §2 完全一致。

### 2.1 模型与部署配置：Llama-3-8B pp2tp2

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | Llama-3-8B | 32 层 · 32 Q头 · 8 KV头 · head_dim=128 · fp16(2B) |
| 部署 | PP=2 × TP=2（4 卡） | 每卡 16 层 / 4 KV头 / 可用显存 2 GiB |
| block_size | 16 | scheduler_block_size = hash_block_size = block_size = 16 |
| page_size_bytes | 32 KiB | 16×4×128×2B×2（单卡单层一页） |
| num_blocks | **4096** | 2GiB÷32KiB÷16，min(4 卡) 对齐后（见 [`1_physical_memory.md`](./1_physical_memory.md) §4） |
| KV 组 | **1 个**（全局，32 层） | `group_id = 0`；这是 Unitary 子类生效的前提 |
| BlockPool | 全局唯一 | `coordinator.py:90` 基类构造中创建，所有组共享 |

### 2.2 请求 R 与前置请求 P

```
前置请求 P（先于 R 服务、已结束）:
  prompt = 共享前缀 SP（32 token）+ P 自己的追问
  → P 服务时把 SP 写成满块 0/1，写满即哈希入 cached_block_hash_to_block
  → P 结束释放后，块 0/1 成为"带哈希的缓存块"：ref_cnt=0、进 free 队列队尾（LRU 保护）

示例请求 R:
  prompt     = 共享前缀 SP（32 token）+ 追加问题（38 token） = 70 token
  max_tokens = 32
  → prefill: 命中 P 缓存的块 0/1（hit_length=32），新分配块 2/3/4
  → decode:  续写 32 token，写满块 4 后再分配块 5/6
  → 结束时:  block_table = [0,1,2,3,4,5,6]（102 token = 6 满块 + 1 残块）
```

**贯穿全文的一句话**：R 在 prefill 时"命中 2 块 → touch 块 0/1 → 新分块 2/3/4 → 满块 2/3 入哈希表"，这四个动作分别由 Coordinator 的四个入口方法透传完成。

## 3. 干什么用

### 核心职责（纯 FullAttention 场景）

| 调度阶段 | 职责 | 对应方法 | 示例 R 中的结果 |
|---------|------|---------|----------------|
| **初始化** | 创建全局唯一 `BlockPool`，创建各组的 `SingleTypeKVCacheManager` | `__init__` | BlockPool 容量 4096；1 个 FullAttentionManager（group_id=0） |
| **前缀查找③** | 调用 `FullAttentionManager.find_longest_cache_hit` | `find_longest_cache_hit` | 命中块 0/1，hit_length=32 |
| **命中块处理④-1** | 两阶段分配阶段①：touch 命中块（`ref_cnt`+1 防驱逐） | `allocate_new_computed_blocks` | 块 0/1 的 ref_cnt 0→1 |
| **新块分配④-2** | 调用 `FullAttentionManager.allocate_new_blocks` | `allocate_new_blocks` | 从空闲队列弹出块 2/3/4 |
| **缓存写入④-3** | 计算完后把满块写入哈希缓存 | `cache_blocks` | 满块 2/3 入哈希表；残块 4 不入 |
| **块释放⑧** | 请求结束时释放块 | `free` / `pop_blocks_for_free` | 逆序释放块 6→5→4→3→2；命中块 0/1 仅减计数 |
| **新块收集** | 收集各 manager 的 `new_block_ids` 供 Worker 清零新块 | （由上层 KVCacheManager 汇总） | 新块 id 2/3/4 附进 SchedulerOutput⑤ |

简单说：**在纯 FullAttention 场景下，Coordinator 基类负责创建 BlockPool，Unitary 子类几乎是透明透传**。它的存在主要是为了统一单组和多组的接口，让上层 KVCacheManager 不需要关心底层是单组还是多组。

### 端到端流程中的位置（示例 R prefill）

```text
③ 前缀查找
Scheduler.get_computed_blocks()
    ↓
KVCacheManager.get_computed_blocks()
    ↓
KVCacheCoordinator.find_longest_cache_hit()          ← 本层入口1
    ↓ （透传）
FullAttentionManager.find_longest_cache_hit()        → 命中 2 个满块（P 缓存的 SP 块 0/1）

④ 分配与缓存
Scheduler.allocate_slots()
    ↓
KVCacheManager.allocate_slots()
    ↓
KVCacheCoordinator.allocate_new_computed_blocks()    ← 本层入口2（两阶段协议·阶段①：touch 命中块）
    ↓ （透传）
FullAttentionManager.add_local_computed_blocks()     → touch 块 0/1，ref_cnt 0→1
    ↓
KVCacheCoordinator.allocate_new_blocks()             ← 本层入口3（两阶段协议·阶段②：分配新块）
    ↓ （透传）
FullAttentionManager.allocate_new_blocks()           → 弹出 3 个新块 2/3/4，new_block_ids 收集
    ↓
模型 forward 计算（⑥ GPU 写 KV）
    ↓
KVCacheManager.cache_blocks()
    ↓
KVCacheCoordinator.cache_blocks()                    ← 本层入口4
    ↓ （透传）
FullAttentionManager.cache_blocks()                  → 计算链式哈希，满块 2/3 写入 cached_block_hash_to_block
```

## 4. 结构是什么

```text
KVCacheCoordinator（ABC 抽象基类）—— 定义跨组协调的标准接口，统一创建 BlockPool
├── KVCacheCoordinatorNoPrefixCache  ← 关闭前缀缓存的协调器（§7.1 一句话概述）
├── UnitaryKVCacheCoordinator        ← 本文核心：单组 FullAttention 透传层（Llama-3-8B 走这里）
└── HybridKVCacheCoordinator         ← 多组混合模型协调器（§7.2 一句话概述）
```

**工厂函数**：上层通过 `get_kv_cache_coordinator()` 工厂函数根据配置自动创建合适的 Coordinator（源码 851-903 行）：
- `enable_caching=False` → `KVCacheCoordinatorNoPrefixCache`
- 只有 1 个 kv_cache_group → `UnitaryKVCacheCoordinator`（**Llama-3-8B 全局 1 组，走这里**）
- 多个 kv_cache_group → `HybridKVCacheCoordinator`

## 5. 基类详解：KVCacheCoordinator

基类负责创建 BlockPool、创建所有组的 SingleTypeKVCacheManager，并定义所有 Coordinator 共用的基础方法。

### 5.1 构造函数

源码位置：`kv_cache_coordinator.py:65-128`

```python
class KVCacheCoordinator(ABC):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,               # 全局KV Cache配置（含所有组的spec、num_blocks等）
        max_model_len: int,                           # 模型最大上下文长度
        max_in_flight_tokens: int,                    # 最大同时处理token数（用于manager预分配）
        use_eagle: bool,                              # 是否启用EAGLE投机解码
        enable_caching: bool,                         # 是否启用前缀缓存
        enable_kv_cache_events: bool,                 # 是否启用KV cache事件（用于metrics/debug）
        dcp_world_size: int,                          # 上下文并行（DCP）的world size
        pcp_world_size: int,                          # 前缀缓存并行（PCP）的world size
        scheduler_block_size: int,                    # 调度块大小（必须是各组block_size和hash_block_size的公倍数）
        hash_block_size: int,                         # 哈希块大小（计算block hash的粒度）
        metrics_collector: KVCacheMetricsCollector | None = None,  # metrics收集器
    ):
        # ========== 1. 保存基础配置 ==========
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        # ========== 2. 校验调度块大小合法性 ==========
        # scheduler_block_size 是Scheduler视角的统一调度粒度
        # 必须同时满足：整除hash_block_size，且整除每个组的block_size
        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size

        # ========== 3. 创建BlockPool（核心！所有组共享同一个BlockPool）==========
        # 这就是为什么所有组的block_id是全局唯一的——它们共用同一个块池
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,  # GPU总块数
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # ========== 4. 确定EAGLE组ID ==========
        # EAGLE投机解码需要"丢弃最后一个块"的特殊处理，标记哪些组需要
        self.eagle_group_ids: set[int] = {
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # 如果开启了use_eagle但没有显式标记eagle组，保守起见标记所有组
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        # ========== 5. 为每个组创建对应的SingleTypeKVCacheManager ==========
        # 纯FullAttention场景下，这里只创建1个FullAttentionManager
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,  # 该组的KV规格（FullAttentionSpec等）
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pool,                  # 共享同一个BlockPool
                enable_caching=enable_caching,
                kv_cache_group_id=i,                         # 组ID
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        # ========== 6. 稀疏前缀缓存保留间隔校验（仅SWA/Mamba组，FullAttention忽略）==========
        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )
```

**示例（Llama-3-8B pp2tp2）中各关键值的落点**：
- `kv_cache_config.kv_cache_groups` 只有 1 个元素 → `single_type_managers` 是长度为 1 的 tuple，`single_type_managers[0]` 就是 `FullAttentionManager`，`kv_cache_group_id=0`
- `BlockPool(num_gpu_blocks=4096)` —— 全局唯一，此后的块 0/1/2/3/4 全部从这里分配
- `scheduler_block_size = hash_block_size = block_size = 16`（三者相等，校验断言轻松通过，没有多粒度问题）
- `eagle_group_ids = set()`（不用 EAGLE；`use_eagle=False` 不触发保守标记）

### 5.2 核心方法：计算需要分配的块数 `get_num_blocks_to_allocate`

源码位置：`kv_cache_coordinator.py:130-190`

```python
    def get_num_blocks_to_allocate(
        self,
        request_id: str,                             # 请求ID
        num_tokens: int,                              # 需要槽位的总token数（含已分配的）
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],  # 刚命中的前缀块（按组）
        num_encoder_tokens: int,                      # encoder token数（cross-attention用，FullAttention为0）
        total_computed_tokens: int,                   # 总命中token数（本地+外部）
        num_local_computed_tokens: int,               # 本地前缀缓存命中token数
        num_tokens_main_model: int,                   # 主模型token数（投机解码时不含lookahead）
        apply_admission_cap: bool = False,            # 是否应用准入上限
    ) -> int:
        """计算总共需要分配多少新块（所有组累加）"""
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # CrossAttention单独处理（encoder-decoder模型用，纯FullAttention不走）
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(...)
            else:
                # 调用对应组的manager计算，纯FullAttention下i=0
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],           # 该组的命中块
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
        return num_blocks_to_allocate
```

> **示例 R 中发生了什么**：prefill 时 `num_tokens=70`、`new_computed_blocks=([块0, 块1],)`、`num_local_computed_tokens=32`。组 0 的 manager 算出：70 token 需 5 块槽位，已有 2 块 → **返回 3**。Scheduler 拿这个数做调度准入判断（显存够不够），真正分配在 §5.4。

### 5.3 核心方法：两阶段分配之阶段①——touch 命中块 `allocate_new_computed_blocks`

源码位置：`kv_cache_coordinator.py:192-236`

这是修复 issue #33775 的关键——**两阶段分配**：先 touch 所有组的命中块（增加 ref_cnt），再分配新块，防止跨组驱逐竞争。

```python
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],  # 刚命中的块（按组）
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """两阶段分配第一阶段：touch所有命中块，增加ref_cnt防止被驱逐"""

        # 运行中的请求已经被跟踪，不会再有新的前缀命中，直接跳过
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # ========== 阶段1a：先touch所有组的本地命中块 ==========
        # 关键：必须先touch完所有组，再去分配新块！
        # 否则如果先分配组0的块，可能驱逐组1还没touch的命中块
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
            # touch内部会调用block_pool.touch() → block.ref_cnt += 1
            # ref_cnt>0的块不会被放在free_block_queue里，也就不会被驱逐

        # ========== 阶段1b：如果有外部命中（分布式缓存），分配外部块 ==========
        if num_external_computed_tokens > 0:
            for manager in self.single_type_managers:
                manager.allocate_external_computed_blocks(
                    request_id,
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                )
```

> **示例 R 中发生了什么**：单组，循环只跑 i=0 一次 → `FullAttentionManager.add_local_computed_blocks(R, [块0, 块1], 32, 0)` → `block_pool.touch()` 把块 0/1 的 `ref_cnt` 从 0 加到 1（P 结束释放时已归 0，但仍在哈希表里，所以能被 touch）。touch 后块 0/1 脱离"可驱逐"状态，接下来分配块 2/3/4 时**绝不会**把刚命中的前缀块挤出去。

**纯 FullAttention 单组场景下**：只有一个组不存在跨组竞争，但仍然遵循相同的两阶段流程，保证接口统一。两阶段协议的"真正用武之地"在 Hybrid 多组场景（§7.2）。

### 5.4 核心方法：两阶段分配之阶段②——分配新块 `allocate_new_blocks`

源码位置：`kv_cache_coordinator.py:238-271`

```python
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,                              # 需要槽位的总token数
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """两阶段分配第二阶段：从free_block_queue分配新块，返回按组组织的新块"""
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
        # 纯FullAttention返回: ([new_block1, new_block2, ...],)
        # 外层tuple是组维度，内层list是该组的新块
```

> **示例 R 中发生了什么**：返回 `([块2, 块3, 块4],)`。三个块从 BlockPool 的 free_block_queue 弹出（`popleft_n(3)`），`req_to_blocks[R] = [块0, 块1, 块2, 块3, 块4]`，`new_block_ids` 收集 [2,3,4] 供 SchedulerOutput⑤ 附带清零。decode 期间本方法还会被调用：块 4 写满后再来一次分配 1 块（块 5、块 6 各一次）。

### 5.5 核心方法：缓存写入 `cache_blocks`

源码位置：`kv_cache_coordinator.py:273-288`

```python
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """模型计算完后，把满块写入前缀缓存（计算链式哈希，加入cached_block_hash_to_block）"""
        for manager in self.single_type_managers:
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.retention_interval,  # SWA/Mamba用，FullAttention忽略
            )
```

> **示例 R 中发生了什么**：prefill forward 完成后 `num_computed_tokens=70` → 满块 2/3（t32-47、t48-63）以链式哈希 `hash(P的hash前缀 + 本块token)` 写入 `cached_block_hash_to_block`；残块 4（6 token）未满**不入表**。P 缓存的块 0/1 在 R 的 prefill 前就在表里（这就是③能命中的原因）。decode 期间每写满一块（块 4→5→6）都会再触发一次入表。

### 5.6 核心方法：块释放 `free`

源码位置：`kv_cache_coordinator.py:290-298`

```python
    def free(self, request_id: str) -> None:
        """请求结束时，释放所有块（直接归还到BlockPool）"""
        for manager in self.single_type_managers:
            manager.free(request_id)
            # manager.free内部会：
            # 1. 取出req_to_blocks[request_id]
            # 2. 调用block_pool.free_blocks(blocks) → ref_cnt -= 1，ref_cnt=0放回free_block_queue
            # 3. 删除req_to_blocks中的记录
```

> **示例 R 中发生了什么**：R 生成完 32 token 结束（共 102 token，7 块）→ `manager.free(R)` 处理块 0~6：命中块 0/1 `ref_cnt` 1→0（重新可复用）；自有块 2~6 `ref_cnt` 1→0 放回空闲队列——有哈希的（2/3/4/5/6 中已入表的）进**队尾**、无哈希的进**队首**。

### 5.7 核心方法：弹出块用于逆序释放 `pop_blocks_for_free`

源码位置：`kv_cache_coordinator.py:300-317`

这个方法用于延迟释放场景：需要先把块从 manager 的记录中弹出，**但不立即归还到 free_block_queue**，等 GPU in-flight 操作确认后再逆序释放。

```python
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """
        从所有manager弹出请求的块记录，但不立即归还到BlockPool。
        调用方必须最终逆序传给block_pool.free_blocks()（尾块先释放，提高复用率）。
        """
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks
        # 返回的是分配顺序的块列表：[block0, block1, block2]
        # 上层释放时要反过来：free([block2, block1, block0])
        # 这样最后分配的不完整尾块先被释放，下次分配时更容易被复用
```

> **示例 R 中发生了什么**：返回 `[块0, 块1, ..., 块6]`（分配顺序），上层逆序调 `free_blocks([块6, 块5, 块4, 块3, 块2])`，命中块 0/1 仅减计数（见总览⑧"逆序 7→6→5→4→3"）。为什么逆序：块 6 是未满残块、无哈希，先归还它就能被下一个请求**立刻当新块复用**；如果按正序先归还满块 2，下一次分配可能拿不到最想要的残块。

### 5.8 其他辅助方法

```python
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """获取每个组的公共前缀块数，用于调度优先级"""
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self, request_id: str, processed_computed_tokens: int, ...
    ) -> None:
        """移除不再需要的块（如SWA滑动窗口外的块），替换为null_block"""
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, processed_computed_tokens, ...)
        # 纯FullAttention下是空操作：所有块永远在窗口内

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """获取请求当前的所有块（按组）"""
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )
        # 示例R prefill后: ([块0, 块1, 块2, 块3, 块4],)

    def new_step_starts(self) -> None:
        """通知每个manager新调度步开始（重置new_block_ids等）"""
        for manager in self.single_type_managers:
            manager.new_step_starts()
```

### 5.9 抽象方法：前缀查找 `find_longest_cache_hit`

```python
    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],              # 预计算好的所有块哈希
        max_cache_hit_length: int,                  # 最大查找长度（token数）
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """
        返回：(按组的命中块列表, 命中token长度, 未缓存公共前缀token数)
        第三个返回值只有Hybrid场景用，单组FullAttention永远返回0
        """
        pass
```

> **示例 R 中发生了什么**：`block_hashes` 只有 4 个**满块**哈希 `[hash(t0-15), hash(t16-31), hash(t32-47), hash(t48-63)]`（70 // 16 = 4，残块 t64-69 未满不参与哈希）；`max_cache_hit_length = 69`（70 − 1，永远给最后 1 个 token 留计算）。具体查找逻辑由子类实现（§6.2）。

## 6. 子类详解：UnitaryKVCacheCoordinator（纯 FullAttention 核心）

这是纯 Full Attention 模型使用的协调器，也是最简单的实现——Llama-3-8B 全局 1 组，工厂函数选中它。

源码位置：`kv_cache_coordinator.py:435-503`

### 6.1 构造函数

```python
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """单KV组协调器：所有层都是同一种类型（纯FullAttention或纯SWA）"""

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
        # 先调用基类构造（创建BlockPool、创建manager等）
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

        # ========== 单组特殊校验 ==========
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size

        if dcp_world_size > 1:
            # DCP上下文并行：实际block_size要乘world_size
            self.block_size *= dcp_world_size

        # 单组场景下，hash_block_size必须等于block_size
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        # 确认确实只有一个组
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

        # 设置EAGLE标志
        self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids
```

> **示例中各校验的落点**：`block_size=16`（无 DCP 不放大）；两条 assert 都通过（1 组、hash=block=16）；`single_type_managers[0].use_eagle = False`。

### 6.2 核心方法：前缀查找 `find_longest_cache_hit`

源码位置：`kv_cache_coordinator.py:486-503`

```python
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """直接透传给FullAttentionManager查找"""

        # 调用下层manager的链式哈希查找
        hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],                     # 只有组0
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids, # 是否做EAGLE丢块
            alignment_tokens=self.block_size,            # 对齐粒度就是block_size
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )

        # 单组场景没有"未缓存公共前缀"，第三个返回值恒为0
        return hit_blocks, hit_length, 0
        # hit_blocks格式: ([hit_block0, hit_block1],)  ← 外层tuple是组维度
```

> **示例 R 中发生了什么（端到端）**：
> - 入参：`block_hashes = [hash(t0-15), hash(t16-31), hash(t32-47), hash(t48-63)]`、`max_cache_hit_length=69`
> - 透传：`FullAttentionManager.find_longest_cache_hit(..., kv_cache_group_ids=[0], alignment_tokens=16)` 逐哈希查 `cached_block_hash_to_block`
> - 查表：前 2 个哈希命中（P 缓存的 SP 块 0/1）；hash(t32-47) 未命中（P 只缓存到 t31 就结束）→ 链式查找到此截断
> - 返回：`([块0, 块1], 32, 0)` —— 命中 2 个满块共 32 token，剩余 38 token 需重新计算

## 7. 其他 Coordinator 一句话概述

以下子类用于多组或特殊场景，纯 Full Attention 单组模型不会用到，了解即可。

### 7.1 KVCacheCoordinatorNoPrefixCache

源码位置：`kv_cache_coordinator.py:385-432`

- **适用场景**：配置中关闭了前缀缓存（`enable_caching=False`）
- **核心特点**：`find_longest_cache_hit` 永远返回空，所有请求每次从头分配新块
- **存在意义**：提供一个"关闭前缀缓存"的开关，不用改其他代码逻辑。若示例 R 走这里：块 0/1 不可复用，prefill 5 块全新分配

### 7.2 HybridKVCacheCoordinator

源码位置：`kv_cache_coordinator.py:521-848`

- **适用场景**：多组混合模型（Jamba、MiniCPM3 等混合 FullAttention+SWA+Mamba），或开启 EAGLE 投机解码
- **核心特点**：
  - **跨组命中对齐**：用**不动点迭代法**找"所有组都能接受的最长公共前缀"
  - **SpecGroup 优化**：spec 相同的组批量查找，减少哈希表查询
  - **两阶段分配的真正用武之地**（§5.3 修复的 issue #33775）：先 touch 所有组，再分配新块，避免跨组驱逐竞争
  - **FullAttention 优先**：FullAttention 组优先决策，其他组跟随
- **复杂度**：整个 KV Cache 管理中最复杂的类，纯 FullAttention 模型不会走这些逻辑

## 8. 设计要点小结（纯 FullAttention 视角）

1. **BlockPool 统一管理**：基类 `__init__` 创建唯一 BlockPool（示例中容量 4096），所有 SingleTypeKVCacheManager 共享，保证 block_id 全局唯一
2. **透传层设计**：UnitaryKVCacheCoordinator 是典型的"透明代理"，存在意义是**接口统一**——让上层 KVCacheManager 用完全相同的代码处理单组和多组
3. **两阶段分配**：`allocate_new_computed_blocks`（touch）→ `allocate_new_blocks`（分配）的顺序修复了多组竞态（issue #33775）；单组场景同样执行，示例 R 中 touch 块 0/1 后才分块 2/3/4
4. **逆序释放优化**：`pop_blocks_for_free` 返回分配顺序的块，上层必须逆序 `free_blocks`（示例 R：块 6→5→4→3→2），让残块优先被复用
5. **多组复杂度隔离**：不动点迭代、SpecGroup 等复杂逻辑全部封装在 HybridKVCacheCoordinator，纯 FullAttention 场景零开销
6. **抽象工厂创建**：`get_kv_cache_coordinator` 按组数自动选择实现，上层无感知
