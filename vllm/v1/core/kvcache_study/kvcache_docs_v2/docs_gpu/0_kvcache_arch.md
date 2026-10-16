# vLLM V1 KV Cache管理架构总览

**章节顺序**：§1 为什么需要 KV Cache → §2 为什么需要 KV Cache 管理 → §3 启动期三侧架构 / 运行期五层架构 → §4 启动时五层如何静态装配 → §5 一条请求的动态生命周期。

---

## 1. 为什么需要 KV Cache

当前主流大语言模型都是**自回归**（autoregressive）生成：模型由 N 层 Transformer block 堆叠而成，每层包含 **attention** 与 **FFN** 两个子层；生成过程逐 token 进行，每个新 token 都以之前全部 token 为上下文。

两个子层对历史信息的依赖不同：

- **attention 层**：新 token 的 Query 必须与**之前所有 token** 的 Key/Value 做注意力计算。如果不做缓存，每生成一个 token 都要重新计算前面全部 token 的 K/V，整个序列的总计算量为 O(n²)。
- **FFN 层**：只对当前 token 自己的隐藏向量做逐位置变换，不依赖任何历史 token 的信息，无需缓存。

```
无 KV Cache 时：每生成一个 token，前面所有 token 的 K/V 都要重新计算一遍
（prompt="你是谁"，逐 token 输出"我是小明"；† 表示本步重算）

 prefill → 生成"我"：  [K/V 你] [K/V 是] [K/V 谁]
 生成"是"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]
 生成"小"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]† [K/V 是]
 生成"明"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]† [K/V 是]† [K/V 小]

 prompt"你是谁"的 K/V 被重算 3 次，每步都从零算起；每步计算量 O(n)，n 步合计 O(n²)
```

KV Cache 把 attention 层已经算好的历史 K/V 保存在 GPU 显存中，每步只计算新 token 的 K/V 并追加进缓存，总计算量降为 O(n)。attention 的 Q 只属于当前 token、FFN 没有历史状态，因此缓存对象只有 K/V；代价是显存占用随序列长度线性增长。

```
  Transformer Block（每层 = attention + FFN，模型共 N 层堆叠）
              ┌──────────────────────────────┐
当前 token ──▶│             FFN              │  逐位置变换：只处理本 token
              │          （无需缓存）         │  的隐藏向量，不依赖历史 token
              ├──────────────────────────────┤
当前 token ──▶│           attention           │  本 token 的 Q 与之前全部
              │        （依赖历史 K/V）       │  token 的 K/V 做注意力计算
              └────────┬─────────────▲───────┘
                       │ 写入新 K/V   │ 读取历史 K/V
                       ▼             │
              ┌──────────────────────────────┐
              │      KV Cache（GPU 显存）      │
              │  已缓存第 1…n 个 token 的 K/V  │
              │    每步只增量追加 1 个 token    │
              └──────────────────────────────┘
```

---

## 2. KV Cache 如何管理

KV Cache 的显存占用随序列长度与并发请求数增长，相同前缀在不同请求间还会重复出现；不加管理的话，显存会碎片化、无法跨请求共享、块的分配与回收也没有统一规则。vLLM V1 的 KV Cache 管理围绕三条核心设计：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（如每个 block 存 16 个 token），按块分配、回收和共享，彻底解决内存碎片问题。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 `GPUModelRunner` 一次性申请并 reshape。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。

---

## 3. vLLM V1 KV Cache 管理架构
### 3.1 启动期三侧架构

vLLM V1 的 KV Cache 管理按运行位置与职责分为三侧。**配置侧**（引擎进程）在启动期依次 ① 算规格、② 测预算、③ 做编排，产出一份 `KVCacheConfig`；这份配置下发两侧：**物理侧**（worker 进程）执行 ④ 落张量，**逻辑侧**（引擎进程）执行 ⑤ 建逻辑层，创建 `KVCacheManager`。物理侧与逻辑侧之间没有对象引用，仅靠 `block_id == 物理张量行号` 桥接；三侧的完整装配链路见 §4。

```
           ┌────────────────────────────────────────────┐
           │ 配置侧（引擎进程 · 全程不碰显存）          │
           │ ① 算规格 → 每层一个 FullAttentionSpec   │
           │ ② 测预算 → 每卡 KV 可用显存字节数        │
           │ ③ 做编排 → KVCacheConfig(num_blocks)    │
           └─────────────────────┬──────────────────────┘
                                 │ 下发同一份 KVCacheConfig
                ┌────────────────┴─────────────────┐
                ▼                                  ▼
┌───────────────────────────────┐  ┌───────────────────────────────┐
│ 物理侧（worker 进程 · 每卡）  │  │ 逻辑侧（引擎进程）            │
│ ④ 落张量：按字节申请物理张量   │  │ ⑤ 建逻辑层 → KVCacheManager │
│   → view 成后端 shape        │  │   new num_blocks 个逻辑块      │
│   → bind 到各 attention 层   │  │   block_id 从 0 编号          │
│ block_id == 张量行号         │  │   对 Scheduler 暴露统一 API   │
└───────────────────────────────┘  └───────────────────────────────┘
两侧无对象引用，仅靠 block_id 桥接（详见 §4.5）
```

### 3.2 运行期五层架构

五层是运行期视角下同一套对象的纵向切分：第 1 层对应物理侧，第 2～5 层构成逻辑侧。运行期每个调度步，请求都沿"第 5 层门面 → 第 4 层协调器 → 第 3 层单类型管理 → 第 2 层块池"把分配决策做下去，最终由第 1 层物理张量承接 K/V 读写。纯 Full Attention 模型只有一个 KV cache group。**五层自下而上编号**：第 1 层=物理显存 → 第 2 层=块池 → 第 3 层=单类型管理 → 第 4 层=协调器 → 第 5 层=顶层门面（`KVCacheManager`）。下图按"上层在下层之上"自上而下排列；`Scheduler` 是调用者，不算层。

```
┌───────────────────────────────────────────────────────────────┐
│  ▲ 运行期五层架构（Full Attention 视角 · 唯一 KV cache group）  │
│  层号自下而上：第1层物理显存 … 第5层门面；下图自上而下排列           │
├───────────────────────────────────────────────────────────────┤
│  Scheduler（调度器 · 调用者）                                 │
│  只通过第5层门面统一调用，不直接触碰 KV cache 内部结构             │
├───────────────────────────────────────────────────────────────┤
│  第5层 · 顶层门面   KVCacheManager                             │
│     持有 1 个第4层；对 Scheduler 暴露统一 API                   │
├───────────────────────────────────────────────────────────────┤
│  第4层 · 协调器    KVCacheCoordinator                          │
│             主线 UnitaryKVCacheCoordinator                     │
│     · 持有 N 个第3层（每 spec group 1 个，主线 N=1）             │
│     · 持有 1 个第2层 BlockPool（所有第3层共享）                   │
├───────────────────────────────────────────────────────────────┤
│  第3层 ×N · 单类型管理  SingleTypeKVCacheManager               │
│             主线 N=1 FullAttentionManager                      │
│     前缀查找(链式哈希)/分配释放/block_table 维护                  │
├───────────────────────────────────────────────────────────────┤
│  第2层 ×1 · 逻辑块池  BlockPool（唯一，所有第3层共享）             │
│     逻辑块分配/释放/缓存哈希/LRU 驱逐（仅 block_id，无显存）       │
│        ┌──────────────────┴───────────────┐                   │
│    FreeKVCacheBlockQueue              BlockHashToBlockMap     │
│     (LRU 空闲块队列)                 (链式哈希→block映射)      │
├───────────────────────────────────────────────────────────────┤
│  第1层 · 物理显存  GPUModelRunner.kv_caches[layer]             │
│     torch [num_blocks, num_kv_heads, block_size, 2*head_dim]  │
│     block_id 直接索引第0维，即物理张量行号                        │
└───────────────────────────────────────────────────────────────┘
```

**五层分篇详解（自底向上）：**

| 层 | 主题（Full Attention 主线） |
|---|---|
| 第 1 层 · 物理显存层（最底） | KV 物理张量的申请、reshape，`block_id == 张量行号` 的桥接关系 |
| 第 2 层 · 逻辑块池层 | `KVCacheBlock`、空闲队列、链式哈希表、`BlockPool` 分配/释放/缓存/驱逐 |
| 第 3 层 · 单类型管理层 | `SingleTypeKVCacheManager` 基类 + `FullAttentionManager` 核心逻辑（前缀查找/分配/释放） |
| 第 4 层 · 协调器层 | `UnitaryKVCacheCoordinator`（单 Full Attention 组直通），混合模型协调器作为扩展 |
| 第 5 层 · 顶层接口层（最顶） | `KVCacheManager` + `KVCacheBlocks`，Scheduler 唯一入口，完整请求生命周期 |

**文本关系（tree 状展开；实线箭头 ─ 上的数字 = 持有数量，虚线 ╎ = 非持有的桥接）**：第 5 层持有 **1** 个第 4 层 → 第 4 层持有 **N** 个第 3 层 ＋ **1** 个第 2 层 → 每个第 3 层引用**同一个**第 2 层。注意持有关系到第 2 层为止：第 2 层在引擎进程、第 1 层在 worker 进程，两者之间没有任何对象引用，只有 `block_id` 桥接。

```
Scheduler（调度器 · 调用者）
└─1→ KVCacheManager（第5层 · 顶层门面）── 对 Scheduler 暴露统一 API
    └─1→ UnitaryKVCacheCoordinator（第4层 · 协调器）
        ├─1→ 第3层 FullAttentionManager（前缀查找/分配释放/block_table 维护）
        └─1→ 第2层 BlockPool（唯一，所有第3层共享）── 仅索引 block_id
             │    FreeKVCacheBlockQueue(LRU) + BlockHashToBlockMap(链式哈希)
             │
             ╎ 桥接（不是持有）：逻辑侧在引擎进程、物理侧在 worker 进程
             ╎ 同一份 KVCacheConfig 下发两侧——
             ╎   物理侧：遍历 KVCacheConfig.kv_cache_tensors，按每个
             ╎           KVCacheTensor.size 字节数 torch.zeros 出 int8 张量
             ╎           （shared_by 里的多个层名（每个group中的同一层）共享同一张，主线每层一张）
             ╎   逻辑侧：new 出 num_blocks 个逻辑块 KVCacheBlock，block_id 取序号 0…n−1
             ╎ 两侧约定 block_id 从 0 编号、即物理行号（见 4.5）
             ▼
             第1层 物理显存 GPUModelRunner.kv_caches[layer]
                （真正读写 K/V 的是注意力算子，调度器从不直接碰张量）
```

### 3.3 关键文件职责

| 文件（`vllm/v1/` 下） | 职责 | 所属层 |
|------|------|------|
| `core/kv_cache_manager.py` | 顶层门面，对 Scheduler 暴露统一接口（`get_computed_blocks`/`allocate_slots`/`free` 等） | 第 5 层 · 顶层门面 |
| `core/kv_cache_coordinator.py` | 协调器：单组直通（Full Attention）或多组对齐（混合模型）；含工厂 `get_kv_cache_coordinator` | 第 4 层 · 协调器 |
| `core/single_type_kv_cache_manager.py` | `FullAttentionManager`：前缀查找、block 分配/释放；含工厂 `get_manager_for_kv_cache_spec` | 第 3 层 · 单类型管理 |
| `core/block_pool.py` | 逻辑 block 池：分配/释放/缓存哈希/LRU 驱逐 | 第 2 层 · 逻辑块池 |
| `core/kv_cache_utils.py` | `KVCacheBlock`、`BlockHash`、空闲队列、block hash 计算工具；容量规划 `get_num_blocks`/`get_kv_cache_configs` | 第 2/1 层 |
| `worker/gpu_model_runner.py` | 物理显存申请（int8 字节缓冲 → reshape 零拷贝视图）并绑定到 attention 层 | 第 1 层 · 物理层 |
| `kv_cache_interface.py` | `KVCacheSpec` / `FullAttentionSpec` / `KVCacheConfig` 定义（含 `real_page_size_bytes`） | 第 1 层 · 物理层 |

---

## 4. 系统启动期：三侧装配

> 本节只讲**装配主线**：三侧对象按什么顺序、在哪个进程里被创建，彼此靠什么衔接。物理张量申请的逐步调用链、代码片段、reshape 细节与 PP/TP 物理分布属于"物理显存篇"，块池内部结构属于"逻辑块池篇"，本节均不展开。

### 4.1 装配全景：规格先行，物理先于逻辑

启动装配是一条单向流水线，横跨**引擎进程**与 **worker 进程**：配置侧 ①～③ 在引擎进程完成"规格推导"（全程不碰显存），物理侧 ④ RPC 到每卡完成"物理落地"，RPC 返回后逻辑侧 ⑤ 才在引擎进程创建 Scheduler 及其逻辑三层。入口为 `EngineCore._initialize_kv_caches()`（`engine/core.py:236`）。

```
引擎进程                                                        worker 进程（每卡）
① 算规格  get_kv_cache_specs()          ── RPC ──▶  每层 attention 返回一个 FullAttentionSpec
② 测预算  determine_available_memory()  ── RPC ──▶  profile_run 实测本卡 KV 可用显存
③ 做编排  get_kv_cache_configs()：同规格层合并为 1 个 group
          → 算 num_blocks → 多卡取 min 对齐，产出 KVCacheConfig
④ 落张量  initialize_from_config()      ── RPC ──▶  按字节申请物理张量 → view 成后端 shape
          （engine/core.py:290）                    → bind 到各 attention 层
                                                    （gpu_worker.py:563 → gpu_model_runner.py:7284）
⑤ 建逻辑层 Scheduler(...)（engine/core.py:149）
          └ KVCacheManager（sched/scheduler.py:231）→ Coordinator → BlockPool + FullAttentionManager
```

### 4.2 配置侧：① 算规格 ② 测预算 ③ 做编排（引擎进程，全程不碰显存）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ① 算规格 | 每层一个 `FullAttentionSpec` | 纯 Full Attention 所有层规格一致，合并为全模型**唯一 KV cache group** |
| ② 测预算 | 每卡一个可用显存字节数 | dummy forward 实测：总显存 × 利用率 − 权重 − 激活 − CUDAGraph 预留 |
| ③ 做编排 | `KVCacheConfig`（含 `num_blocks`） | 单层单块字节 `page_size = 2 × block_size × num_kv_heads × head_dim × dtype_size`（系数 2 对应 K+V）；本卡 `num_blocks = 可用显存 // page_size // 本 worker 层数`；多卡取最小值，共享同一套 `block_id` 空间 |

引擎算出 `num_blocks` 后写入 `KVCacheConfig`，同一份配置下发物理侧与逻辑侧，两侧各自做自己的事。

### 4.3 物理侧：④ 落张量（worker 进程 · 每卡）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ④ 落张量 | 每 worker 每层 1 张物理张量（PP 下只含本 worker 负责的层） | 先按字节数申请、再零拷贝 view 成后端要求的 shape；同一个行号在所有层语义相同 |

物理侧只按配置执行、不自行算块数，逐条执行细节见 4.5 衔接点。

### 4.4 逻辑侧：⑤ 建逻辑层（引擎进程）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ⑤ 建逻辑层 | 第 2～5 层全部对象 | 构造链路：`Scheduler → KVCacheManager → Coordinator → BlockPool + FullAttentionManager`（见 4.1 图 ⑤） |

逻辑侧只建元数据对象，逐条执行细节见 4.5 衔接点。

### 4.5 衔接点：一份配置定容量，`block_id` 即行号

引擎算出 `num_blocks` 后写入 `KVCacheConfig` 并下发两侧，两侧各自做自己的事：

- **物理侧（worker，只按配置执行，不自行算块数）**
  - 遍历 `KVCacheConfig.kv_cache_tensors`；
  - 对每个 `KVCacheTensor` 按其 `size`（普通 layout 下恰为 `num_blocks × page_size` 字节）调 `torch.zeros(..., dtype=torch.int8)`，得到一张扁平字节张量；
  - 该条目 `shared_by` 列出的层名都指向这同一张张量——纯 Full Attention 主线无层共享，恰好每层一张。
  - 源码：`kv_cache_interface.py:830-836`（配置定义）、`gpu_model_runner.py:7013-7030`（申请执行）。
- **逻辑侧（引擎进程的 BlockPool，只建元数据对象）**
  - new 出 `num_blocks` 个 `KVCacheBlock`，`block_id` 取序号 `0 … n−1`；
  - 每个对象只带 `block_id`、引用计数等元数据，**不含任何 K/V 数据**。
  - 源码：`block_pool.py:162-164`。
- **桥接结果**：同一份配置保证两侧容量相等，"从 0 顺序编号"的约定让 `block_id` 直接等于物理张量行号——无需查表或拷贝。此后物理张量不再变动，分配/共享/驱逐只改引用计数和哈希表。
  - 特例：`block_id=0` 开池即留作 `null_block` 占位，不维护引用计数、不可分配，实际可分配块为 `n−1`。

---

## 5. 系统运行期：一条请求的 KV Cache 生命周期

### 5.1 宏观：请求状态机与两侧调用链

一条请求由 Scheduler 在两个队列间推进：`WAITING`（等待首次 prefill）→ `RUNNING`（持续 decode）→ 生成结束后释放。prefill 只做一次、处理整段 prompt，decode 每步生成 1 个 token；二者共用同一套 step 骨架（`schedule → execute_model → sample`，内部都是"分配 slot → GPU 执行"），decode 只是规模更小并跳过前缀查找，系统里没有两套独立的管理逻辑。调度器在共享 token 预算下按"先 RUNNING、后 WAITING"逐请求填充。

```
WAITING ──首次调度──▶ RUNNING ──每步 decode──▶ 生成结束 ──▶ 释放
      （prefill，一次）         （EOS / max_tokens）
```

分配决策全部在逻辑侧沿五层编排链完成，产出整数 `block_id` 列表（`block_table`）；第 1 层物理张量不在这条链上——物理侧收到下发的 `block_table` 后，才由 attention 算子读写张量：

```
逻辑侧（引擎进程）：
Scheduler ─▶ 第5层 KVCacheManager ─▶ 第4层 Coordinator ─▶ 第3层 FullAttentionManager ─▶ 第2层 BlockPool
                       只改元数据（引用计数 / 哈希表 / 空闲队列），产出 block_table
                                             │ block_table 随 SchedulerOutput 下发
                                             ▼
物理侧（worker 进程）：
attention 算子按 block_table 读写物理张量（第 1 层）
```

### 5.2 时间线：prefill 一次，decode 循环

四个阶段中，前缀查找只在 prefill 出现一次；分配 slot 与 GPU 执行每个调度步都有；释放在生成结束时做一次。

**prefill（仅一次）：**

```
WAITING ── 入队：预计算链式哈希（不触碰任何管理对象）
   │
   ▼ 首次调度
前缀查找 ── 沿哈希链只读询问 BlockPool：命中仅标记可复用，ref_cnt 不变
   ▼
分配 slot ── touch 命中块（ref_cnt++、移出空闲队列）＋ 弹空闲块，拼出 block_table；写满的块记入哈希
   ▼
GPU 执行 ── 新块清零 → 算子按 block_table 索引物理张量，读命中前缀的 K/V、写整段 prompt 的 K/V
   ▼
进入 RUNNING
```

**decode（每步循环，直到 EOS / max_tokens）：**

```
RUNNING
   ▼ 每步
分配 slot ── 弹空闲块拼出本步 block_table（每步新增 0 或 1 块）；写满的块记入哈希
   ▼
GPU 执行 ── 新块清零 → 算子读历史 K/V、写本步 1 个 token 的 K/V
   │
   ├─ 未结束：回到分配 slot，进入下一步
   └─ EOS / max_tokens
   ▼
释放 ── 逆序归还、ref_cnt--；归零才回收，按 LRU 规则入空闲队列
```

### 5.3 空间分工：决策在逻辑侧，执行在物理侧

- **逻辑侧（第 2～5 层）承担全部分配决策**：前缀查找、分配 slot、释放只改引用计数、哈希表与空闲队列，全程不触碰物理张量。
- **物理侧（第 1 层）只在 GPU 执行时动张量**：这是两侧唯一的衔接时点——worker 清零本轮新块后，attention 算子按 `block_table` 索引物理行，读历史 K/V、写新 K/V。

块的命运由此只由引用计数与空闲队列位置决定：前缀命中经 `touch`（计数加一、移出空闲队列）零拷贝共享给新请求；请求结束只减计数，仍被其他请求引用的块不回收；归零回收时，带哈希的块排到队尾受 LRU 保护、无哈希的块排到队首被优先复用——前缀缓存因此能跨请求持续积累，并被后续请求复用。

---

## 6. 扩展：其他注意力类型概览

本文主线是最基础的 Full Attention 模型。vLLM V1 同样支持以下场景，它们都在上述五层骨架上做扩展：

| 类型 | 代表模型 | 主要差异 | 扩展落点（代码层） |
|------|---------|---------|---------|
| **Sliding Window Attention (SWA)** | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐；前缀查找从右往左找窗口内命中 | 第 3 层滑窗管理器、第 4 层协调器 |
| **Mamba/SSM** | Bamba、Jamba | 无 KV 只有 state，block 存 recurrent state 而非 K/V；缓存逻辑不同 | 第 3 层状态管理器，第 1 层张量布局随之改变 |
| **混合模型 (Full + SWA/Mamba)** | Gemma3、Jamba、Llama4 | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool 但 page size 必须统一 | 第 4 层 `HybridKVCacheCoordinator` |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | KV 低秩压缩，物理张量形状不同 | 第 1 层 MLA 规格与张量布局 |
| **Cross-Attention** | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | 第 3 层 `CrossAttentionManager` |
| **投机解码 (EAGLE/MTP)** | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑 | 第 4 层协调器与第 5 层门面 |

阅读建议：按 §3 的层号自底向上（第 1 层物理显存 → 第 2 层块池 → 第 3 层单类型管理 → 第 4 层协调器 → 第 5 层门面）吃透 Full Attention 主线，再对照本表按需理解复杂场景。
