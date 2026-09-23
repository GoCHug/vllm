# KVC 打印 patch 讲解（为什么这么加、每处加在哪、想验证什么）

> 本目录 9 个 patch 覆盖实操中加的全部 40 处 `[KVC]` 打印：8 个在 vllm 包（`/vllm-workspace/vllm`），1 个在 vllm-ascend 包（`/vllm-workspace/vllm-ascend`）。
> 对照阅读：理论文档 `../0_kv_cache_management_arch.md`（五层架构）、`../0_kvcache_management_of_type.md`（9 个类型）、`../0_runtime_sequence.md`（请求时序）；实操取证 `../docs/1_kvc_patch_apply_e2e_record.md`。

---

## 0. 快速使用

```bash
# 一键应用/回滚（推荐；含 dry-run 预检 + [KVC] 计数 + py_compile 验证）
./apply_patches.sh    # 容器内默认 /vllm-workspace/{vllm,vllm-ascend}; 本地: VLLM_DIR=... VLLM_ASCEND_DIR=... ./apply_patches.sh
./revert_patches.sh   # patch -R 反向应用，不依赖 .orig 备份（容器重启丢可写层后同样可用）

# 手动逐个（也可用 git apply）
cd /vllm-workspace/vllm        && patch -p1 < 01_vllm_v1_request.py.patch   # 01~08 逐个
cd /vllm-workspace/vllm-ascend && patch -p1 < 09_vllm_ascend_worker_model_runner_v1.py.patch

# 看打印（每行自动带: INFO 时间戳 [文件名:行号] [KVC][层] 消息）
grep '\[KVC\]' llama.log
```

| patch | 文件 | 层 | 行号（补丁后） |
|---|---|---|---|
| 01 | `vllm/v1/request.py` | ENQ | 184 |
| 02 | `vllm/v1/core/kv_cache_utils.py` | ENQ+L2 | 216/257/339/365/395/616 |
| 03 | `vllm/v1/core/block_pool.py` | L2 | 69/83/107/117/208/241/248/290/334/410/457/490/515 |
| 04 | `vllm/v1/core/kv_cache_manager.py` | L5 | 174/222/242/258/359/430/449/467/487/510 |
| 05 | `vllm/v1/core/kv_cache_coordinator.py` | L4 | 187/219/260/283/298/460/475/493 |
| 06 | `vllm/v1/core/single_type_kv_cache_manager.py` | L3 | 233/293/347/395/576/589/597/613 |
| 07 | `vllm/v1/engine/core.py` | CFG | 264/276/314 |
| 08 | `vllm/v1/worker/gpu_model_runner.py` | L1 | 7017/7147（**NPU 不触发**） |
| 09 | `vllm_ascend/worker/model_runner_v1.py` | L1 | 4256/4693（**NPU 实际路径**） |

---

## 1. 设计总纲：打印点怎么选的

### 1.1 两条主线决定选点

1. **静态装配线（启动一次）**：配置侧 4 类型（`KVCacheSpec → KVCacheGroupSpec → KVCacheTensor → KVCacheConfig`）→ 物理侧张量申请/reshape → 逻辑侧三大件装配（`BlockPool/FreeKVCacheBlockQueue/BlockHashToBlockMap` → managers → coordinator → KVCacheManager）。每个类型落地处打一次"装配快照"。
2. **动态生命周期线（每请求）**：理论时序文档把一条请求拆为 **入队 → 前缀查找（只读）→ 分配 slots（S1~S4）→ GPU 执行 → decode 循环 → 释放（逆序）**。每个阶段的**入口、关键决策点、出口**各打一条——入口带参数、决策点带判定依据、出口带结果。这样日志读起来就是时序文档的逐行验证。

### 1.2 标签体系

所有打印统一 `[KVC]` 前缀 + 层标签，`grep '\[KVC\]'` 一网打尽，按标签过滤即得某一层视角：

| 标签 | 位置 | 回答的问题 |
|---|---|---|
| `[CFG]` | engine/core.py | KV cache "应该长什么样"：多少块、每块多大、怎么分组、显存怎么来 |
| `[ENQ]` | request.py + kv_cache_utils.py | 入队时算出了哪些哈希（前缀缓存的 key 是什么） |
| `[L1]` | vllm-ascend model_runner_v1.py | 物理显存实际怎么申请、什么形状 |
| `[L2]` | block_pool.py + kv_cache_utils.py | 每个块此刻归谁：分配/释放/命中/驱逐/入表 |
| `[L3]` | single_type_kv_cache_manager.py | 前缀逐块查表的 HIT/MISS、每步分配计算 |
| `[L4]` | kv_cache_coordinator.py | 单组直通：L5 的指令如何下发到 L3/L2 |
| `[L5]` | kv_cache_manager.py | Scheduler 视角：S1~S4 决策链与最终 block_table |

### 1.3 打印风格约定（为什么这么写）

1. **40 处全部统一 `logger.info(...)`**：vllm logger 每行自动携带 `INFO 时间戳 [文件名:行号] [KVC][层] 消息`，日志免 grep 即可精确回溯源码行；worker 进程必须走 vllm logger（见 2），引擎进程亦统一走 logger 以获得稳定转发与行号定位。
2. **worker 进程必须走 vllm logger**（L1，model_runner_v1.py）：worker 的裸 `print` **不会**进主日志（软硬件栈对子进程 stdout 的捕获差异），必须走 vllm logger 才带 `(Worker_PP0_TP0 pid=...)` 前缀转发。这是实操踩坑后改的——第一版 L1 用 print，一条都没打出来；第二版改 logger 才可见。
3. **只打元数据，绝不碰显存/张量**：块列表打 `[block_id, ...]` 或 `[(block_id, ref_cnt), ...]`；哈希只打前 12 个 hex 字符（完整是 32 字节 bytes，比对足够且日志可读）；shape 打 `tuple(tensor.shape)` + dtype + device。打印本身零拷贝、零开销（相对推理）。

4. **不改控制流**：唯一"重写"的是 7 处值捕获——`return X` → `_x = X; logger.info(...); return _x`，语义完全等价（完整清单见 §5 第 5 条）。

---

## 2. 逐 patch 详解

### 01 request.py — 入队：链式哈希的诞生点（ENQ）

**位置**：`Request.__init__` 尾部，紧跟 `self.update_block_hashes()`。

**为什么在这**：入队是 KVCache 生命周期第一步。构造函数尾部时点，`block_hashes` 刚算完、`prompt_token_ids/max_tokens` 齐备——**这是能一次性看到"这条请求带来了哪些缓存 key"的最早也最全的时刻**。理论文档 4.1：满块才有哈希（`N // block_size` 个）。

**实测样例**（P 请求，block_size=128）：
```
[KVC][ENQ] hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=dcaeded48257
[KVC][ENQ] hash_block_tokens: parent=dcaeded48257, tokens=128 -> BlockHash=8c4d2a2ea88b
[KVC][ENQ] Request(...) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['dcaeded48257', '8c4d2a2ea88b']
```
→ 324 token 只产生 2 个满块哈希（尾 68 token 无哈希）；第 2 块父哈希=第 1 块结果，链式结构肉眼可见。

### 02 kv_cache_utils.py — 两大补位：哈希函数本体 + 空闲队列原语（ENQ+L2）

**加了两类点，因为它们分别在两个包里"无人打"的底层原语：**

1. `hash_block_tokens`（哈希计算的**实现点**，:616）：打印 `(parent, tokens数) -> 结果`。01 在调用方打结果列表，02 在实现处打每一块的推导——**验证 H(bn)=fn(H(bn-1), tokens(bn)) 链式定义**，也验证首块用 `NONE_HASH` 种子。P/R 两请求此函数输出完全相同（前 256 token 一致 → 链一致），这就是 R 能命中的全部前提。
2. `FreeKVCacheBlockQueue` 五个原语（init/popleft/append/prepend_n/append_n）：空闲队列是理论文档 3.5 节"驱逐优先级"的全部载体。每个原语的打印对应一条结论：
   - `init`：建队规模 + 伪头尾哨兵（block_id=-1）
   - `popleft`：**队头弹出**=分配或 null 块摘取（开池第一条就是 null_block 的摘取）
   - `append`（单块回队尾）/ `append_n`（批量回队尾，带哈希 LRU 保护）/ `prepend_n`（插队首，优先复用）
   - 理论文档说"无哈希块 prepend 队首"——**实操正是靠这几条打印发现 0.23.0 的 FullAttentionManager.free 走的全部是 append_n**（P 的无哈希块 3、R 的无哈希块 6 都进了队尾），推翻了文档说法。这是加这组打印最大的收获。

### 03 block_pool.py — 第 2 层全普查：块的一生所有状态迁移（L2）

BlockPool 是唯一能回答"**此刻每个块归谁**"的组件，13 处打印覆盖块的所有状态迁移边：

| 状态迁移 | 打印点 | 验证的理论论断 |
|---|---|---|
| 建池（KVCacheBlock×N + 队列 + 哈希表 + null 摘取） | `__init__` :208 | 配置侧 num_blocks 落地逻辑侧；块 0 开池即 null |
| 哈希表查询 | `get_one_block` :69/:83、`get_cached_block` :241/:248 | 前缀命中判定：任一 group miss 即整块 miss |
| 满块入表 | `cache_full_blocks` :290(幂等早退)/:334(入表汇总) | 只有**写满**的块才入缓存（P 尾块 68/128 不入） |
| 新块分配 | `get_new_blocks` :410 | popleft_n 从队头弹、ref_cnt 0→1 |
| 复用前驱逐 | `_maybe_evict_cached_block` :457 | 弹哈希表条目 + reset_hash 后块才能复用 |
| 命中共享 | `touch` :490 | ref_cnt++ 并 O(1) 摘出空闲队列（零拷贝共享的实现点） |
| 归还 | `free_blocks` :515 | 逆序归还、ref_cnt-- **归零才回收**、prepend/append 语义 |

另外 `BlockHashToBlockMap.insert/pop`（:107/:117）：验证**入表不去重**（保证 block_table append-only）和驱逐时条目弹出。

**为什么把 `get_cached_block` 的 HIT/MISS 打在返回前而不是循环里**：循环里每个 group 打一条太啰嗦；MISS 早退点打一条、全部命中后汇总打一条，一眼看出"断在第几块"。

### 04 kv_cache_manager.py — 第 5 层门面：Scheduler 的完整决策链（L5）

**为什么重点是 `allocate_slots` 的 S1~S4**：理论文档 4.2.2 把分配拆成 S1 容量检查、S2 touch 命中块、S3 分配新块、S4 缓存满块——**prefill 和 decode 走的是同一个入口同一套四步**，这是 V1 "统一骨架" 论断的核心。四步逐点打出来，对照时序文档即可逐步验证：

- 进入（:359）：`num_new_tokens/num_new_computed_tokens/request.num_computed_tokens`——区分这是 prefill（大 num_new_tokens）还是 decode（恒 1）
- S1（:430）：`需 X 块 vs 可用 Y 块`——容量判定与"等待下轮调度"的 return None 分支
- S2（:449）：touch 哪些命中块（无命中时这步压根不进，L4 的 print 也证实）
- S3（:467）：新块号列表
- S4+返回（:487）：本轮缓存到第几个 token + **当前完整 block_table**（逻辑侧对物理侧的唯一接口，全流程的"结果"）

`get_computed_blocks`（:222 跳过分支 / :242 命中结果 / :258 返回 KVCacheBlocks）：prefill 独有阶段。命中结果行 `hit_length=256` = 2 块 × 128，`max_cache_hit_length=485`（= num_tokens−1，"全命中也要重算最后一个 token"论断的实证）。

`free`（:510）：释放前的 block_table 与 03 的归还明细呼应，构成释放闭环。

### 05 kv_cache_coordinator.py — 第 4 层：验证"单组直通"（L4）

纯 Full Attention 只有一个 KV cache group，理论文档说协调器退化为**直通**（UnitaryKVCacheCoordinator → single_type_managers[0]）。加打印的目的不是看复杂逻辑，而是**实证这层确实是透传**：

- `find_longest_cache_hit` 进入（:475）："下钻 single_type_managers[0]" + 返回（:493）`hit_length = 命中块数 × block_size`（乘法发生在这一层，理论文档 4.2.1 的公式落点）
- S1~S4 各自的"逐组下放"（:187/:219/:260/:283）：N=1 时每个动作就一行，证明所有组的求和/分发对单组模型是平凡的
- `__init__`（:460）：把 spec 类型、page_size_bytes、coordinator_block_size 打出来——**block_size=128、page_size_bytes=262144 这两个关键数字第一次露面就在这**

### 06 single_type_kv_cache_manager.py — 第 3 层：前缀查表的算法本体（L3）

**`find_longest_cache_hit`（FullAttentionManager）是前缀缓存的算法核心**，:576/:589/:597/:613 四点打出算法的每一步：

```
入口: 最多查 (max_length // block_size) 个哈希
循环: 第 N 块 HIT: hash=xxx -> cached blocks=[..]   ← 每块命中一行
      第 N 块 MISS: hash=xxx -> break               ← 断链即止（链式哈希 miss 后必 miss 的论断）
返回: computed_blocks=[[1, 2]]
```

其余四点对应"分配四步"在本层的实现细节：
- `allocate_new_computed_blocks`（:233）：确认 touch 发生在这里（下钻 L2）
- `allocate_new_blocks`（:293）：**纯算式** `需 4 块 − 已有 2 = 新分配 2 块`——把 cdiv 计算打的明明白白，R prefill "总 4 块、命中 2、补 2（1 满 1 尾）" 的账目就是这条
- `cache_blocks`（:347）：`已缓存 0 块 → 满块数 2`（满块数 = num_tokens // block_size）
- `free`（:395）：持有块列表（**逆序归还**的取出点）

### 07 engine/core.py — 配置侧出口：4 类型一次打齐（CFG）

**为什么打在 `_initialize_kv_caches` 而不是配置生成函数内部**：`get_kv_cache_configs` 是条长流水线（specs 合并→分组→逐 worker 算块数→对齐），在内部打既琐碎又容易漏；**在流水线的三个里程碑打结果快照**，字段全、一次成型：

1. `determine_available_memory` 之后（:264）：各 worker profile 实测可用 KV 显存——"实际可用显存以实际跑为准"的数据源（51.98/51.99/51.94/51.95 GiB）
2. `get_kv_cache_configs` 之后（:276）：逐 worker 打 **KVCacheConfig → KVCacheGroupSpec → KVCacheSpec(repr 全字段) → page_size_bytes → KVCacheTensor(size/shared_by)**——配置侧 4 个类型一屏打完，PP2TP2 切分（每 worker 16 层、num_kv_heads=4）也在 group 的 layer_names 里现形
3. `generate_scheduler_kv_cache_config` 之后（:314）：**min 对齐后的最终 num_blocks**（13295）——这正是下发给逻辑侧 BlockPool 建池的数字，与 03 的 `BlockPool.__init__` 打印首尾呼应

> 踩坑注记：此处曾用 `_t.offset` 打印 KVCacheTensor 的 packed 偏移字段，0.23.0 该类只有 `size/shared_by`（理论文档基于更新版本），AttributeError 直接把 EngineCore 打崩——所以 patch 里 KVCacheTensor 只打 size/shared_by，这正是"以容器实际代码为准"的教训。

### 08 gpu_model_runner.py — vllm 原版物理侧（NPU 不触发，为何保留）

按理论文档，物理侧 = 申请 int8 字节池（`_allocate_kv_cache_tensors`）+ reshape 成后端 shape（`_reshape_kv_cache_tensors`），于是先在 vllm 原版 GPUModelRunner 里加了这两点（:7017/:7147）。

**实测一条也不出**——排查发现 NPU 上 worker 实际执行的是 vllm-ascend 重写版（见 09）。保留这个 patch 的理由：
1. 在 GPU/CUDA 环境跑同版本 vllm 时它就是生效路径；
2. 它是"理论文档描述的物理侧"的忠实实现样本，与 09 对照正好展示 **vllm-ascend 对物理层的重写差异**（单张合并布局 vs K/V 分离布局）。

### 09 model_runner_v1.py — vllm-ascend 重写物理侧（NPU 真正的 L1）

vllm-ascend 的 `NPUModelRunner` 重写了 `_allocate_kv_cache_tensors` / `_reshape_kv_cache_tensors`（**为支持 PD 分离，K、V 拆成两张独立的 int8 字节池，2M 对齐**）。1/2 两点都打在 dense attention 分支（llama3-8b 走的路径）：

1. K/V 分配后（:4256）：`KVCacheTensor(size=3323.75MiB) -> K int8 1661.88MiB + V int8 1661.88MiB (alignment=2M)`——每层两池各半
2. reshape 装配点（:4693，仅首层，16 层同形）：`K_cache shape=(13295,128,4,128) / V_cache shape=(13295,128,4,128) bf16`

**为什么值得单独成 patch**：这组打印揭示了与理论文档最大的一处布局差异——**K/V 分离、维序 (num_blocks, block_size, num_kv_heads, head_dim)**（理论上文档是单张 `(num_blocks, num_kv_heads, block_size, 2*head_dim)`）；同一 block_id 在 K/V 两张张量中索引同一行，"`block_id == 张量行号`"的桥接约定在 NPU 上依然成立。

> 顺带成为 vllm/vllm-ascend 双源码结构的实证：改 vllm 必须搞清哪个组件被平台插件覆盖，否则补丁会"静默失效"。

---

## 3. 打印点 ↔ 理论论断映射表（拿日志对答案）

| 理论论断（文档出处） | 验证打印层 | 实测结论 |
|---|---|---|
| 相同前缀 → 相同哈希链（类型篇 §1.3） | ENQ hash_block_tokens | P/R 前 2 块哈希逐字节一致 ✓ |
| 满块才有哈希（时序 4.1） | ENQ Request 入队 | 324 token → 2 哈希；486 → 3 ✓ |
| 前缀查找"遇 miss 即断"（时序 4.2.1） | L3 第 N 块 MISS -> break | P 首块 breaking ✓ |
| hit_length = 命中块数 × block_size（时序 4.2.1） | L4 返回 | 2 × 128 = 256 ✓ |
| max_cache_hit_length = N−1（时序 4.2.1） | L5 get_computed_blocks | 486−1=485 ✓ |
| touch：ref_cnt++ 摘出队列零拷贝共享（架构 4.2） | L2 touch | [(1,1),(2,1)] ✓ |
| 分配 = cdiv(num_tokens, block_size)−已有（时序 S3） | L3 allocate_new_blocks | 需 4−已有 2=新 2 ✓ |
| 新满块才入哈希表（架构 4.2） | L2 insert/cache_full_blocks | P 尾块 68/128 不入表 ✓ |
| decode 每步 0/1 块（时序 4.4） | L5 S1 | 步 1~26 "需 0 块"、步 27 "需 1 块" ✓ |
| 当步填满的块入缓存（时序 4.4） | L2 块 5 insert（decode 步 27） | map size 3→4 ✓ |
| 逆序释放、归零才回收（时序 4.5） | L2 free_blocks | [6,5,4,2,1] 全归零 ✓ |
| 无哈希块 prepend 队首优先复用（时序 4.5） | L2 append_n vs prepend_n | **实测全部 append_n，论断不成立于 0.23.0** ✗ |
| 物理侧单张 (num_blocks, kv_heads, block, 2*head)（类型篇 1.4） | L1 reshape | **NPU 为 K/V 分离两张 (13295,128,4,128)** ✗ |
| block_id == 张量行号（架构 3.2） | L1 shape 第 0 维 | 13295 行 = num_blocks ✓ |

## 4. 一次 P→R 用例的打印时序速查（对照找行）

以实操用例（P=324 token / R=486 in + 35 out，block_size=128，2026-09-23 实测）按出现顺序：

```
P:  [ENQ] hash_block_tokens ×2 (NONE_HASH→dcaeded48257→8c4d2a2ea88b)                       ← 链式哈希生成
    [ENQ] Request 入队: BlockHash × 2
    [L4][L3] find_longest_cache_hit 进入 → [L2] MISS → [L3] 第 1 块 MISS break
    [L4][L5] 返回 hit_length=0
    [L5] allocate_slots 进入 → [L4] 需 3 块 ×2 → [L5] S1 3 vs 可用 13294
    [L2] get_new_blocks(3) -> [1,2,3]                 ← ref_cnt 0→1
    [L3] 新分配 3 块 → [L5] S3
    [L3] cache_blocks 0→2 → [L2] insert 块1(dcaeded48257)、块2(8c4d2a2ea88b) → [L2] 满块入表
    [L5] S4 + 返回 block_table=([1,2,3],)
    [L5] free → [L3] 持有 [1,2,3] → [L2] free_blocks [(3,0),(2,0),(1,0)] → append_n [3,2,1]
R:  [ENQ] hash ×3（前 2 与 P 同链，第 3 = 2807a32a7a28）
    [L2] get_one_block HIT 块1 → [L3] 第 1 块 HIT
    [L2] get_one_block HIT 块2 → [L3] 第 2 块 HIT
    [L4] 返回 hit_length=256 → [L5] get_computed_blocks (486, [1,2], 256)
    [L5] allocate_slots 进入 (num_new_tokens=230)
    [L4] 需 4 块 → [L5] S1 4 vs 可用 13294 → [L5] S2 [[1,2]]
    [L3] touch 命中块 [1,2] → [L2] touch [(1,1),(2,1)]
    [L2] get_new_blocks(2) -> [4,5] → [L3] 需4−已有2=新2 [4,5] → req_blocks=[1,2,4,5]
    [L4][L5] S3 → S4 → block_table=([1,2,4,5],)
    [L2] insert 块4(2807a32a7a28, map 2→3)
    decode 步1~26: [L5] allocate_slots (num_new_tokens=1) → S1 需 0 → S3 新块 []（尾块 102→128）
    decode 步27:  [L4] 需 1 块 (num_tokens=513) → [L2] get_new_blocks(1) -> [6]
                  [L2] insert 块5 hash=4324a28794de (map size=4)   ← 步26填满 步27入表+跨界
    decode 步28~34: S1 需 0（块6 占 8/128 不满）
    [L5] free 释放前 block_table=([1,2,4,5,6],) → [L2] free_blocks [(6,0),(5,0),(4,0),(2,0),(1,0)] → append_n [6,5,4,2,1]
```

## 5. 附：与 patch 形态相关的踩坑记录

1. **`_t.offset` 崩溃**：理论文档的 KVCacheTensor 带 packed 布局字段，0.23.0 没有——见 07。教训：**锚点字段先 grep 实际代码再引**。
2. **worker 裸 print 静默丢失**：L1 第一版 print 一条不见 → vllm-ascend 路径 + logger 转发两个问题叠加，9 号 patch 由 print+错文件 演进为 logger+重写文件。
3. **`model_runner_v1.py.orig.bak` 备份时机**：初次备份晚于打补丁（备成了补丁版），09 patch 一度为空；后验证 vllm-ascend 仓库 HEAD 即原始码、git diff 仅含 27 行纯插入后，改用 `git diff` 生成标准 patch，并用 `git show HEAD:` 恢复真原件。
4. **等价重写的 return**：`return create_kv_cache_blocks(x)` → 先赋值、打印、再 return——本目录所有此类改动均语义等价，可放心用于生产观察。
5. **批量 print→logger 转换正则误伤**（2026-09-23 修复）：转换脚本对整个文件执行 `re.sub(r',(\s*\n\s*\))', r'\1')`（本意只清理 flush=True 删除后的尾逗号），把全文件所有函数调用/参数列表的**尾逗号**全部误删——数百行纯格式改动混进 patch（如 `-    UniformTypeKVCacheSpecs,` → `+    UniformTypeKVCacheSpecs`）。修复：以 `git show HEAD:` 干净源码为底做 diff 对齐重建，非 [KVC] 区域 100% 还原——发现并保住了两处正则额外咬伤的值捕获行（`hash_block_tokens` 的 `return BlockHash(` 与 coordinator 的 `return tuple(`）。现 9 个 patch = logger.info 打印插入 + logger 导入 + 全部 7 处值捕获重写（`02 _block_hash`、`03 _b`、`04 _computed_kv_blocks`/`_res_kv_blocks`、`05 _res_blocks`/`_hit_len`、`08 _kv_cache`），无任何其他源码改动。

## 6. 端到端应用实测记录（2026-09-23）

9 个 patch 在 gggtest 容器（vllm 0.23.0 干净基线，`git checkout` 还原）上完成完整"还原 → 应用 → 启动 → P/R 运行"验证，全部通过：

| 验证点 | 结果 |
|---|---|
| 应用 | `apply_patches.sh`：dry-run 9/9 预检 → 9/9 应用 → 逐文件计数合计 113 行 `[KVC]`（request 2 / utils 12 / block_pool 27 / manager 22 / coordinator 16 / single_type 16 / core 10 / gpu_model_runner 4 / model_runner_v1 4）+ py_compile OK |
| hunk 纯净性 | 9 patch 共 52 hunk 全部含打印行；删除行恰好 **7 处值捕获重写**（§5 第 5 条清单），零其他源码改动 |
| 本地往返 | git 还原 → `apply_patches.sh` 9/9 → 113 行 + py_compile OK → `revert_patches.sh` 反向应用后 `[KVC]` 全归零 → 再 apply |
| 启动运行 | 启动期 155 行 `[KVC]`（CFG82 / L1 68 / L2 3 / L4 1 / L5 1）；P 轨迹 33 行、R 轨迹 355 行，R 五块生命周期（复用 1,2 + prefill 4,5 + decode 填满块5 并跨界申请块6）完整走通，completion_tokens=35 / finish=length |

日志行号实证（本轮实测出现过的全部打印点）：`[request.py:185]`、`[kv_cache_utils.py:217/258/396/617]`、`[block_pool.py:70/84/108/209/242/249/340/411/491/516]`、`[single_type_kv_cache_manager.py:234/294/348/396/577/590/598/614]`、`[kv_cache_coordinator.py:188/220/261/284/299/461/476/494]`、`[kv_cache_manager.py:175/243/259/360/431/450/468/491/495/511]`。

完整日志与逐段解读见 `../docs/1_kvc_patch_apply_e2e_record.md`。本节可作为 patch 文件正确性的直接证据。
