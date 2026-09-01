# vLLM V1 KVCacheCoordinator 跨组协调层（Full Attention 主线）

> 五层架构第 4 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) ｜上层 ➔ [`5_kv_cache_manager.md`](./5_kv_cache_manager.md)
> 时序位置：[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) ③前缀查找（find_longest_cache_hit）、④分配与缓存（两阶段分配 + cache_blocks）、⑧释放（free / pop_blocks_for_free）
>
> 源文件：`vllm/vllm/v1/core/kv_cache_coordinator.py`
>
> 主线：纯 Full Attention 单 group → `UnitaryKVCacheCoordinator`（透传层）。**本文以 Llama-3-8B pp2tp2（4 卡）+ 示例请求 R 为统一锚点，每个方法都带"示例 R 中发生了什么"参考块。**

## 1. 概览

`KVCacheCoordinator` 是五层 KV Cache 管理架构中的**第四层——跨组协调层**。

下钻链的位置：`KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager → BlockPool`。上层的 KVCacheManager 只跟 Coordinator 对话；Coordinator 负责把请求**按组拆分、分发给各组的 SingleTypeManager**，并在基类中统一创建 `BlockPool`。

对于纯 Full Attention 模型，所有层同类型，只会分成**一个 KV 组**，此时使用子类 `UnitaryKVCacheCoordinator`——纯透传层，所有方法只转发给唯一的 `FullAttentionManager`。

### 核心职责（纯 FullAttention 场景）

| 调度阶段 | 职责 | 对应方法 | 示例 R 中的结果 |
|---------|------|---------|----------------|
| **初始化** | 创建全局唯一 `BlockPool`，创建各组的 `SingleTypeKVCacheManager` | `__init__` | BlockPool 容量 4096；1 个 FullAttentionManager（group_id=0） |
| **前缀查找③** | 透传 `FullAttentionManager.find_longest_cache_hit` | `find_longest_cache_hit` | 命中块 0/1，hit_length=32 |
| **命中块处理④-1** | touch 命中块（`ref_cnt`+1 防驱逐） | `allocate_new_computed_blocks` | 块 0/1 的 ref_cnt 0→1 |
| **新块分配④-2** | 透传 `FullAttentionManager.allocate_new_blocks` | `allocate_new_blocks` | 从空闲队列弹出块 2/3/4 |
| **缓存写入④-3** | 透传 `FullAttentionManager.cache_blocks`，满块写入哈希缓存 | `cache_blocks` | 满块 2/3 入哈希表；残块 4 不入 |
| **块释放⑧** | 透传 `FullAttentionManager.free` | `free` / `pop_blocks_for_free` | 逆序释放块 6→5→4→3→2；命中块 0/1 仅减计数 |

## 2. 示例设定（全文锚点）

与 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) §2 完全一致。

### 2.1 模型与部署配置：Llama-3-8B pp2tp2

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | Llama-3-8B | 32 层 · 32 Q头 · 8 KV头 · head_dim=128 · fp16(2B) |
| 部署 | PP=2 × TP=2（4 卡） | 每卡 16 层 / 4 KV头 / 可用显存 2 GiB |
| block_size | 16 | scheduler_block_size = hash_block_size = block_size = 16 |
| page_size_bytes | 32 KiB | 16×4×128×2B×2（单卡单层一页） |
| num_blocks | **4096** | 2GiB÷32KiB÷16，min(4 卡) 对齐后（见 [`1_physical_memory.md`](./1_physical_memory.md) §4） |
| KV 组 | **1 个**（全局，32 层） | `group_id = 0`；这是 Unitary 子类生效的前提 |
| BlockPool | 全局唯一 | `coordinator.py:128` 基类构造中创建，所有组共享 |

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

## 3. 端到端时序（请求 R 视角）

> UnitaryKVCacheCoordinator 在单组场景下是**纯透传层**：所有方法只转发给 `single_type_managers[0]`（FullAttentionManager）。下面以 R 的完整生命周期展示交互。

```
┌──────────────────────────────────────────────────────────────────────┐
│ A. 入队                                                              │
│   Coordinator 不参与（Scheduler 预计算链式哈希）                      │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ B. 首次调度 prefill（WAITING → RUNNING）                              │
│                                                                      │
│  ① coord.find_longest_cache_hit(R的hash, 69)    透传→ FM[0]         │
│     → 逐哈希查表: 命中 P 缓存的块 0/1 (hit_length=32)                │
│     → 返回 ([块0, 块1], 32, 0)                                      │
│                                                                      │
│  ② coord.allocate_new_computed_blocks(R, ([块0,块1],), 32, 0)        │
│     透传→ FM[0]                                                      │
│     → touch 块 0/1: ref_cnt 0→1 (移出 free 队列, 防驱逐)              │
│                                                                      │
│  ③ coord.allocate_new_blocks(R, 70, 70)         透传→ FM[0]         │
│     → pop [2, 3, 4] from BlockPool.free_block_queue                  │
│     → block_table = [0, 1, 2, 3, 4]                                 │
│     → new_block_ids = [2, 3, 4] (供 Worker 清零)                     │
│                                                                      │
│  ④ coord.cache_blocks(R, 70)                    透传→ FM[0]         │
│     → 满块 2(t32-47)/3(t48-63) 入 cached_block_hash_to_block         │
│     → 残块 4(t64-69) 未满 16, 不入表                                │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ C. GPU forward（Coordinator 不参与）                                  │
│   4 worker 各自清零 block 2/3/4 → forward 写 70 token KV → sample    │
│   → 第 1 个输出 token → R 变 RUNNING                                  │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ D. decode 续写（32 步, 每步 1 token）                                  │
│                                                                      │
│   ┌─ 每步开始 ───────────────────────────────────────────────────┐   │
│   │  coord.new_step_starts()      透传→ FM[0]                    │   │
│   │  → 重置 new_block_ids (本步重新收集新块)                       │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   步 1~9:   coord.allocate_new_blocks(R, N, N) → 0 新块 (块4 未满)    │
│             coord.cache_blocks(R, N)       → 无满块不入表              │
│                                                                      │
│   步 10:   coord.allocate_new_blocks(R, N, N) → 0 新块               │
│             coord.cache_blocks(R, N)       → 块4 满(6+10=16) → 入表    │
│                                                                      │
│   步 11:   coord.allocate_new_blocks(R, N, N) → pop [5] → table+1    │
│   步 12~25: ...                        → 0 新块, 填块5                │
│                                                                      │
│   步 26:   coord.cache_blocks(R, N)       → 块5 满(16) → 入表         │
│                                                                      │
│   步 27:   coord.allocate_new_blocks(R, N, N) → pop [6] → table+1    │
│   步 28~32: ...                        → 0 新块 (块6 仅 6 token)      │
│             coord.cache_blocks(R, N)       → 块6 未满, 不入表         │
│                                                                      │
│   block_table 演变: [0,1,2,3,4] → [0..5] → [0..6]                    │
└───────────────────────────────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│ E. 释放                                                               │
│                                                                      │
│  coord.free(R)                                  透传→ FM[0]         │
│     → 逆序释放 block_table: 块6→5→4→3→2                            │
│     ├─ ref_cnt-- 归 0 → 回 free_block_queue                          │
│     │  有哈希 → append 队尾 (LRU 保护, 后续可前缀命中)                 │
│     │  无哈希 → prepend 队首 (优先复用)                               │
│     └─ 命中块 0/1: ref_cnt 1→0 (仍被 P 或其他共享, 不回收)            │
│                                                                      │
│  ※ pop_blocks_for_free (延迟释放变体): 先弹出不归还, 等 GPU 确认      │
│    后再逆序 free_blocks. 示例 R 正常结束时直接 free, 不走此分支.       │
└──────────────────────────────────────────────────────────────────────┘
```

**R 与 Coordinator 的完整交互清单**：

| 阶段 | Coordinator 方法 | 透传目标 | R 的实际参数与结果 |
|------|-----------------|----------|-------------------|
| A 入队 | — | — | 不参与（Scheduler 预计算哈希） |
| B① 前缀查找 | `find_longest_cache_hit` | FM[0] | 命中 [块0,块1], hit_length=32 |
| B② touch | `allocate_new_computed_blocks` | FM[0] | touch 0/1, ref_cnt 0→1 |
| B③ 分配 | `allocate_new_blocks` | FM[0] | pop [2,3,4] → block_table=[0,1,2,3,4] |
| B④ 缓存 | `cache_blocks` | FM[0] | 满块2/3 入哈希表 |
| C forward | — | — | 不参与（GPU 侧执行） |
| D 每步 | `new_step_starts` | FM[0] | 重置 new_block_ids |
| D decode | `allocate_new_blocks` | FM[0] | 步11→pop[5], 步27→pop[6] |
| D 满块 | `cache_blocks` | FM[0] | 步10→块4入表, 步26→块5入表 |
| E 释放 | `free` | FM[0] | 逆序 6→5→4→3→2 归还; 0/1 ref_cnt-- |

> **一句话总结**：R 与 Coordinator 的全部交互只有 7 个方法，每个都**原样透传**给唯一的 `FullAttentionManager`。Coordinator 在单组场景下的存在价值纯粹是**接口统一**——让上层 KVCacheManager 的代码不必区分单组还是多组。

## 4. 类继承结构

```text
KVCacheCoordinator（ABC 抽象基类）—— 定义跨组协调的标准接口，统一创建 BlockPool
├── KVCacheCoordinatorNoPrefixCache  ← 关闭前缀缓存的协调器（§7.1）
├── UnitaryKVCacheCoordinator        ← 本文核心：单组 FullAttention 透传层（Llama-3-8B 走这里）
└── HybridKVCacheCoordinator         ← 多组混合模型协调器（§7.2）
```

**工厂函数** `get_kv_cache_coordinator()`（源码 851-903 行）按配置自动选择：
- `enable_caching=False` → `KVCacheCoordinatorNoPrefixCache`
- 只有 1 个 kv_cache_group → `UnitaryKVCacheCoordinator`（**Llama-3-8B 全局 1 组，走这里**）
- 多个 kv_cache_group → `HybridKVCacheCoordinator`

## 5. 基类 KVCacheCoordinator 详解

基类负责创建 BlockPool、创建所有组的 SingleTypeKVCacheManager，并定义所有 Coordinator 共用的基础方法。

### 5.1 构造函数

源码位置：`kv_cache_coordinator.py:65-128`

```python
class KVCacheCoordinator(ABC):
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,               # 全局 KV Cache 配置（含所有组的 spec、num_blocks 等）
        max_model_len: int,                           # 模型最大上下文长度
        max_in_flight_tokens: int,                    # 最大同时处理 token 数（用于 manager 预分配）
        use_eagle: bool,                              # 是否启用 EAGLE 投机解码
        enable_caching: bool,                         # 是否启用前缀缓存
        enable_kv_cache_events: bool,                 # 是否启用 KV cache 事件
        dcp_world_size: int,                          # 上下文并行（DCP）的 world size
        pcp_world_size: int,                          # 前缀缓存并行（PCP）的 world size
        scheduler_block_size: int,                    # 调度块大小（须为各组 block_size 和 hash_block_size 的公倍数）
        hash_block_size: int,                         # 哈希块大小（计算 block hash 的粒度）
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        # ========== 1. 保存基础配置 ==========
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        # ========== 2. 校验调度块大小合法性 ==========
        assert scheduler_block_size % hash_block_size == 0 and all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size

        # ========== 3. 创建 BlockPool（全局唯一，所有组共享）==========
        # 这就是为什么所有组的 block_id 是全局唯一的——它们共用同一个块池
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,  # GPU 总块数
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # ========== 4. 为每个组创建对应的 SingleTypeKVCacheManager ==========
        # 纯 FullAttention 场景下，这里只创建 1 个 FullAttentionManager
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pool,                  # 共享同一个 BlockPool
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )
```

> **Llama-3-8B pp2tp2 中各关键值的落点**：
> - `kv_cache_config.kv_cache_groups` 只有 1 个元素 → `single_type_managers` 是长度为 1 的 tuple，`single_type_managers[0]` 就是 `FullAttentionManager`，`kv_cache_group_id=0`
> - `BlockPool(num_gpu_blocks=4096)` —— 全局唯一，此后块 0/1/2/3/4/5/6 全部从这里分配
> - `scheduler_block_size = hash_block_size = block_size = 16`（三者相等，校验断言轻松通过）
> - `eagle_group_ids = set()`（Llama-3-8B 不用 EAGLE）

### 5.2 计算需要分配的块数 `get_num_blocks_to_allocate`

源码位置：`kv_cache_coordinator.py:130-190`

```python
    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],  # 刚命中的前缀块（按组）
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """计算总共需要分配多少新块（所有组累加）"""
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
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

### 5.3 touch 命中块 `allocate_new_computed_blocks`

源码位置：`kv_cache_coordinator.py:192-236`

修复 issue #33775 的关键——**两阶段分配**：先 touch 所有组的命中块（增加 ref_cnt），再分配num_external_computed_tokens需要的新块，防止跨组驱逐竞争。

```python
    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """两阶段分配阶段①：touch 所有命中块，增加 ref_cnt 防止被驱逐"""

        # 运行中的请求已被跟踪，不会再有新的前缀命中，直接跳过
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # 阶段1a：先 touch 所有组的本地命中块
        # 必须先 touch 完所有组，再去分配新块——否则分配可能驱逐未 touch 的命中块
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
            # touch 内部调 block_pool.touch() → block.ref_cnt += 1
            # ref_cnt > 0 的块不在 free_block_queue 里，不会被驱逐

        # 阶段1b：外部命中（分布式缓存），纯 FullAttention 主线 num_external=0 跳过
        if num_external_computed_tokens > 0:
            ...
```

> **示例 R 中发生了什么**：单组，循环只跑 i=0 一次 → `FullAttentionManager.add_local_computed_blocks(R, [块0, 块1], 32, 0)` → `block_pool.touch()` 把块 0/1 的 `ref_cnt` 从 0 加到 1。touch 后块 0/1 脱离"可驱逐"状态，接下来分配块 2/3/4 时**绝不会**把刚命中的前缀块挤出去。

### 5.4 分配新块 `allocate_new_blocks`

源码位置：`kv_cache_coordinator.py:238-271`

```python
    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """两阶段分配阶段②：从 free_block_queue 分配新块，返回按组组织的新块"""
        return tuple(
            manager.allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )
        # 纯 FullAttention 返回: ([块2, 块3, 块4],)
        # 外层 tuple 是组维度，内层 list 是该组的新块
```

> **示例 R 中发生了什么**：返回 `([块2, 块3, 块4],)`。三个块从 BlockPool 的 free_block_queue 弹出（`popleft_n(3)`），`req_to_blocks[R] = [块0, 块1, 块2, 块3, 块4]`，`new_block_ids` 收集 [2,3,4] 供 SchedulerOutput⑤ 附带清零。decode 期间本方法还会被调用：块 4 写满后再分配块 5、块 6 各一次。

### 5.5 缓存写入 `cache_blocks`

源码位置：`kv_cache_coordinator.py:273-288`

```python
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """模型计算完后，把满块写入前缀缓存（计算链式哈希，加入 cached_block_hash_to_block）"""
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)
```

> **示例 R 中发生了什么**：prefill forward 完成后 `num_computed_tokens=70` → 满块 2/3（t32-47、t48-63）以链式哈希 `hash(P的hash前缀 + 本块token)` 写入 `cached_block_hash_to_block`；残块 4（6 token）未满**不入表**。decode 期间每写满一块（块 4 步10、块 5 步26）都会再触发一次入表。

### 5.6 块释放 `free`

源码位置：`kv_cache_coordinator.py:290-298`

```python
    def free(self, request_id: str) -> None:
        """请求结束时，释放所有块（直接归还到 BlockPool）"""
        for manager in self.single_type_managers:
            manager.free(request_id)
            # manager.free 内部：
            # 1. 取出 req_to_blocks[request_id]
            # 2. 调 block_pool.free_blocks(blocks) → ref_cnt -= 1, 归 0 放回 free_block_queue
            # 3. 删除 req_to_blocks 中的记录
```

> **示例 R 中发生了什么**：R 生成完 32 token 结束（共 102 token，7 块）→ `manager.free(R)` 处理块 0~6：命中块 0/1 `ref_cnt` 1→0（重新可复用）；自有块 2~6 `ref_cnt` 1→0 放回空闲队列——有哈希的进**队尾**、无哈希的进**队首**。

### 5.7 延迟释放 `pop_blocks_for_free`

源码位置：`kv_cache_coordinator.py:300-317`

延迟释放场景：先把块从 manager 记录中弹出，**但不立即归还**，等 GPU in-flight 操作确认后再逆序释放。

```python
    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """从所有 manager 弹出请求的块记录，但不立即归还到 BlockPool。
        调用方必须最终逆序传给 block_pool.free_blocks()（尾块先释放，提高复用率）。"""
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks
        # 返回分配顺序: [块0, 块1, ..., 块6]
        # 上层逆序: free([块6, 块5, 块4, 块3, 块2]) — 残块先归还, 下次优先复用
```

> **示例 R 中发生了什么**：R 正常结束时直接 `free`，不走此分支。此方法用于 GPU in-flight 期间不能立即释放的场景（如异步 KV 加载）。

### 5.8 辅助方法

```python
    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """获取每个组的公共前缀块数，用于调度优先级"""
        return [m.get_num_common_prefix_blocks(running_request_id)
                for m in self.single_type_managers]

    def remove_skipped_blocks(self, request_id, processed_computed_tokens, ...):
        """移除不再需要的块（SWA 滑动窗口外），替换为 null_block"""
        for m in self.single_type_managers:
            m.remove_skipped_blocks(request_id, processed_computed_tokens, ...)
        # 纯 FullAttention 下是空操作：所有块永远在窗口内

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """获取请求当前的所有块（按组）"""
        return tuple(m.req_to_blocks.get(request_id) or []
                     for m in self.single_type_managers)
        # 示例 R prefill 后: ([块0, 块1, 块2, 块3, 块4],)

    def new_step_starts(self) -> None:
        """通知每个 manager 新调度步开始（重置 new_block_ids 等）"""
        for m in self.single_type_managers:
            m.new_step_starts()
```

### 5.9 抽象方法：前缀查找 `find_longest_cache_hit`

```python
    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],              # 预计算好的所有块哈希
        max_cache_hit_length: int,                  # 最大查找长度（token 数）
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """返回: (按组的命中块列表, 命中 token 长度, 未缓存公共前缀 token 数)
        第三个返回值只有 Hybrid 场景用，单组 FullAttention 永远返回 0"""
        pass
```

> **示例 R 中发生了什么**：`block_hashes` 只有 4 个满块哈希 `[hash(t0-15), hash(t16-31), hash(t32-47), hash(t48-63)]`（70 // 16 = 4，残块 t64-69 未满不参与哈希）；`max_cache_hit_length = 69`（70 − 1，永远给最后 1 个 token 留计算）。具体查找逻辑由子类实现（§6.2）。

## 6. 子类 UnitaryKVCacheCoordinator 详解

纯 Full Attention 模型使用的协调器，也是最简单的实现——Llama-3-8B 全局 1 组，工厂函数选中它。

源码位置：`kv_cache_coordinator.py:435-503`

### 6.1 构造函数

```python
class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """单 KV 组协调器：所有层都是同一种类型（纯 FullAttention 或纯 SWA）"""

    def __init__(self, kv_cache_config, max_model_len, max_in_flight_tokens,
                 use_eagle, enable_caching, enable_kv_cache_events,
                 dcp_world_size, pcp_world_size, scheduler_block_size,
                 hash_block_size, metrics_collector=None):
        # 先调基类构造（创建 BlockPool、创建 manager 等）
        super().__init__(...)

        # 单组特殊校验
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size

        # 单组场景下 hash_block_size 必须等于 block_size
        assert not enable_caching or (hash_block_size == self.block_size)
        # 确认确实只有一个组
        assert len(self.kv_cache_config.kv_cache_groups) == 1
```

> **Llama-3-8B 中各校验的落点**：`block_size=16`（无 DCP 不放大）；两条 assert 都通过（1 组、hash=block=16）；`single_type_managers[0].use_eagle = False`。

### 6.2 前缀查找 `find_longest_cache_hit`

源码位置：`kv_cache_coordinator.py:486-503`

```python
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """直接透传给 FullAttentionManager 查找"""

        hit_blocks, hit_length = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],                     # 只有组 0
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,  # False
            alignment_tokens=self.block_size,            # 对齐粒度 = block_size = 16
        )

        # 单组场景没有"未缓存公共前缀"，第三个返回值恒为 0
        return hit_blocks, hit_length, 0
        # hit_blocks 格式: ([blk0, blk1],)  ← 外层 tuple 是组维度
```

> **示例 R 中发生了什么（端到端）**：
> - 入参：`block_hashes = [hash(t0-15), hash(t16-31), hash(t32-47), hash(t48-63)]`、`max_cache_hit_length=69`
> - 透传：`FullAttentionManager.find_longest_cache_hit(..., kv_cache_group_ids=[0], alignment_tokens=16)` 逐哈希查 `cached_block_hash_to_block`
> - 查表：前 2 个哈希命中（P 缓存的 SP 块 0/1）；hash(t32-47) 未命中（P 只缓存到 t31 就结束）→ 链式查找到此截断
> - 返回：`([块0, 块1], 32, 0)` —— 命中 2 个满块共 32 token，剩余 38 token 需重新计算

## 7. 扩展子类概述

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
