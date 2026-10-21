# vLLM V1 KV Cache管理架构总览（NPU · vllm-ascend 视角）

**章节顺序**：§1 为什么需要 KV Cache → §2 为什么需要 KV Cache 管理 → §3 启动期三侧架构 / 运行期五层架构 → §4 启动时五层如何静态装配 → §5 一条请求的动态生命周期。

本文以 **NPU（昇腾 Ascend，经 vllm-ascend 插件接入 vLLM V1）** 为主线；上游 vLLM 的 GPU 实现作为对照出现。

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

KV Cache 把 attention 层已经算好的历史 K/V 保存在 **NPU HBM 显存**中，每步只计算新 token 的 K/V 并追加进缓存，总计算量降为 O(n)。attention 的 Q 只属于当前 token、FFN 没有历史状态，因此缓存对象只有 K/V；代价是显存占用随序列长度线性增长。

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
              │     KV Cache（NPU HBM 显存）  │
              │  已缓存第 1…n 个 token 的 K/V  │
              │    每步只增量追加 1 个 token    │
              └──────────────────────────────┘
```

NPU 侧真正读写这对张量的是 CANN 算子：`reshape_and_cache`（写 K/V）、`npu_fused_infer_attention_score`（FIA，prefill/混合读）、`_npu_paged_attention`（PA，decode 读）——调度器与 KV 管理层自始至终不直接触碰张量。

---

## 2. KV Cache 如何管理

KV Cache 的显存占用随序列长度与并发请求数增长，相同前缀在不同请求间还会重复出现；不加管理的话，显存会碎片化、无法跨请求共享、块的分配与回收也没有统一规则。vLLM V1 的 KV Cache 管理围绕三条核心设计（NPU 主线下完全同构）：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（Ascend 注意力算子的核内分页为 128 token），按块分配、回收和共享，彻底解决内存碎片问题。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 **`NPUModelRunner`** 一次性申请并 reshape——NPU 侧每层 K、V 各一张独立张量。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。

> NPU 与 GPU 的差异全部封装在物理侧内部：分页粒度（128 vs 16/128）、张量形态（K/V 双池 vs 单张量）不同，但"block_id == 物理张量行号"的编号约定两侧一致，逻辑侧三件套（`BlockPool`/`KVCacheManager`/coordinator）对平台完全无感。

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
│ 物理侧（worker 进程 · 每卡 NPU）│  │ 逻辑侧（引擎进程）            │
│ ④ 落张量：按字节申请物理张量   │  │ ⑤ 建逻辑层 → KVCacheManager │
│   （K、V 各一次独立申请，     │  │   new num_blocks 个逻辑块      │
│    开 KV 传输时 2MiB 对齐）   │  │   block_id 从 0 编号          │
│   → view 成后端 shape（无 permute）│ 对 Scheduler 暴露统一 API │
│   → bind 到各 attention 层   │  │                               │
│   → 双张量登记给 KV 传输层   │  │                               │
│ block_id == 张量行号（K/V 双    │  │                               │
│  池各一套同序号行空间）       │  │                               │
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
│  第1层 · 物理显存  NPUModelRunner.kv_caches[layer]              │
│     = (k_cache, v_cache) 二元组，各一张                        │
│     torch [num_blocks, block_size(128), num_kv_heads, head_dim]│
│     block_id 直接索引第0维：K 池、V 池各自一套同序号行号          │
└───────────────────────────────────────────────────────────────┘
```

**五层分篇详解（自底向上）：**

| 层 | 主题（Full Attention 主线 · NPU） |
|---|---|
| 第 1 层 · 物理显存层（最底） | K/V 双物理张量的申请（每层两次独立 `torch.zeros`，开传输时 2MiB 对齐）、`view` 成后端 shape，`block_id == 张量行号` 的桥接关系 |
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
             ╎           KVCacheTensor.size 字节数拆成 K/V 两个 int8 字节池
             ╎           （calc_split_factor 对半拆；开 KV 传输时每池 2MiB
             ╎             对齐；dense 层无条件双池，主线每层两张）
             ╎           shared_by 里的多个层名共享同一张（主线每层一对）
             ╎   逻辑侧：new 出 num_blocks 个逻辑块 KVCacheBlock，block_id 取序号 0…n−1
             ╎ 两侧约定 block_id 从 0 编号、即物理行号（K 池/V 池同序号）（见 4.5）
             ▼
             第1层 物理显存 NPUModelRunner.kv_caches[layer] = (k_cache, v_cache)
                （真正读写 K/V 的是 CANN 注意力算子，调度器从不直接碰张量）
```

### 3.3 关键文件职责

**逻辑侧（第 2～5 层）与配置编排全部复用上游 vLLM**（`vllm/v1/` 下），平台差异只落在物理侧与注意力后端（**`vllm_ascend/` 插件**）：

| 文件 | 职责 | 所属层 |
|------|------|------|
| `vllm/v1/core/kv_cache_manager.py` | 顶层门面，对 Scheduler 暴露统一接口（`get_computed_blocks`/`allocate_slots`/`free` 等）——平台无关，上游复用 | 第 5 层 · 顶层门面 |
| `vllm/v1/core/kv_cache_coordinator.py` | 协调器：单组直通（Full Attention）或多组对齐（混合模型）；含工厂 `get_kv_cache_coordinator`）——平台无关 | 第 4 层 · 协调器 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `FullAttentionManager`：前缀查找、block 分配/释放；含工厂 `get_manager_for_kv_cache_spec`）——平台无关 | 第 3 层 · 单类型管理 |
| `vllm/v1/core/block_pool.py` | 逻辑 block 池：分配/释放/缓存哈希/LRU 驱逐）——平台无关 | 第 2 层 · 逻辑块池 |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`、`BlockHash`、空闲队列、block hash 计算工具；容量规划 `get_num_blocks`/`get_kv_cache_configs`）——平台无关 | 第 2/1 层 |
| `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner`：物理显存申请（K/V 双 int8 字节池，开传输时 2MiB 对齐 → 零拷贝 `view` 后端 shape）并绑定到 attention 层 | 第 1 层 · 物理层 |
| `vllm_ascend/worker/worker.py` | `NPUWorker`：①②④ 阶段的 RPC 执行体（profile、初始化、NPUGraph 编译预热） | 第 1 层 · 物理层 |
| `vllm_ascend/attention/attention_v1.py` | `AscendAttentionBackend`：`get_kv_cache_shape`(2, N, B, H, D)/`swap_blocks`/`copy_blocks`；`AscendAttentionBackendImpl`：FIA/PA 算子与 `reshape_and_cache` 双池写入 | 第 1 层 · 算子接口 |
| `vllm/v1/kv_cache_interface.py` | `KVCacheSpec` / `FullAttentionSpec` / `KVCacheConfig` 定义（含 `real_page_size_bytes`）——平台无关 | 第 1 层 · 物理层 |

---

## 4. 系统启动期：三侧装配

> 本节只讲**装配主线**：三侧对象按什么顺序、在哪个进程里被创建，彼此靠什么衔接。物理张量申请的逐步调用链、代码片段、对齐细节与 PP/TP 物理分布属于"物理显存篇"，块池内部结构属于"逻辑块池篇"，本节均不展开。

### 4.1 装配全景：规格先行，物理先于逻辑

启动装配是一条单向流水线，横跨**引擎进程**与 **worker 进程**：配置侧 ①～③ 在引擎进程完成"规格推导"（全程不碰显存），物理侧 ④ RPC 到每张 NPU 卡完成"物理落地"，RPC 返回后逻辑侧 ⑤ 才在引擎进程创建 Scheduler 及其逻辑三层。入口为 `EngineCore._initialize_kv_caches()`（`engine/core.py:236`）。

```
引擎进程                                                        worker 进程（每卡 NPU）
① 算规格  get_kv_cache_specs()          ── RPC ──▶  每层 attention 返回一个 FullAttentionSpec
② 测预算  determine_available_memory()  ── RPC ──▶  profile_run 实测本卡 KV 可用显存
③ 做编排  get_kv_cache_configs()：同规格层合并为 1 个 group
          → 算 num_blocks → 多卡取 min 对齐，产出 KVCacheConfig
④ 落张量  initialize_from_config()      ── RPC ──▶  按字节申请 K/V 双物理张量 → view 成后端 shape
          （engine/core.py:290）                    → bind 到各 attention 层 → 登记 KV 传输层
                                                    （worker.py:907 → model_runner_v1.py:3843）
⑤ 建逻辑层 Scheduler(...)（engine/core.py:149）
          └ KVCacheManager（sched/scheduler.py:231）→ Coordinator → BlockPool + FullAttentionManager
```

### 4.2 配置侧：① 算规格 ② 测预算 ③ 做编排（引擎进程，全程不碰显存）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ① 算规格 | 每层一个 `FullAttentionSpec` | 纯 Full Attention 所有层规格一致，合并为全模型**唯一 KV cache group**；dense decoder 层在 NPU 上原样收上游 spec（模型层自产，与平台无关） |
| ② 测预算 | 每卡一个可用显存字节数 | dummy forward 实测：总显存 × 利用率 − 权重 − 激活（NPU 此步**无图内存项**，NPUGraph 推迟到 ④ capture 后事后记账） |
| ③ 做编排 | `KVCacheConfig`（含 `num_blocks`） | 单层单块字节 `page_size = 2 × block_size(128) × num_kv_heads × head_dim × dtype_size`；本卡 `num_blocks = 可用显存 // page_size // 本 worker 层数`；多卡取最小值，共享同一套 `block_id` 空间 |

引擎算出 `num_blocks` 后写入 `KVCacheConfig`，同一份配置下发物理侧与逻辑侧，两侧各自做自己的事。

### 4.3 物理侧：④ 落张量（worker 进程 · 每卡 NPU）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ④ 落张量 | 每 worker 每层 **K、V 各 1 张**物理张量（PP 下只含本 worker 负责的层） | `KVCacheTensor.size` 经 `calc_split_factor` 对半拆成两次独立 `torch.zeros(int8)`（开 KV 传输时 2MiB 对齐）；再零拷贝 view 成 `(num_blocks, block_size, heads, head_dim)`；K 池、V 池同一个行号语义相同 |

物理侧只按配置执行、不自行算块数，逐条执行细节见 4.5 衔接点。

### 4.4 逻辑侧：⑤ 建逻辑层（引擎进程）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ⑤ 建逻辑层 | 第 2～5 层全部对象 | 构造链路：`Scheduler → KVCacheManager → Coordinator → BlockPool + FullAttentionManager`（见 4.1 图 ⑤）；全部纯元数据，与平台完全无关 |

逻辑侧只建元数据对象，逐条执行细节见 4.5 衔接点。

### 4.5 衔接点：一份配置定容量，`block_id` 即行号

引擎算出 `num_blocks` 后写入 `KVCacheConfig` 并下发两侧，两侧各自做自己的事：

- **物理侧（NPU worker，只按配置执行，不自行算块数）**
  - 遍历 `KVCacheConfig.kv_cache_tensors`；
  - 对每个 `KVCacheTensor.size`（普通 layout 下恰为 `num_blocks × page_size` 字节）按 `calc_split_factor` 拆出 K、V 两份字节数，各调 `torch.zeros(..., dtype=torch.int8)`，得到两张扁平字节张量（两次独立分配、地址不连续；prefill/decode 分离传输开启时每张先多申请 2MiB、`data_ptr` 向上对齐后切片）；
  - 该条目 `shared_by` 列出的层名都指向这**同一对**张量——纯 Full Attention 主线无层共享，恰好每层一对。
  - 源码：`kv_cache_interface.py:830-836`（配置定义）、`model_runner_v1.py:4082-4309`（K/V 双池申请执行）。
- **逻辑侧（引擎进程的 BlockPool，只建元数据对象）**
  - new 出 `num_blocks` 个 `KVCacheBlock`，`block_id` 取序号 `0 … n−1`；
  - 每个对象只带 `block_id`、引用计数等元数据，**不含任何 K/V 数据**。
  - 源码：`block_pool.py:162-164`。
- **桥接结果**：同一份配置保证两侧容量相等，"从 0 顺序编号"的约定让 `block_id` 直接等于物理张量行号——K 池、V 池**各有一套同序号的行空间**，无需查表或拷贝。此后物理张量不再变动，分配/共享/驱逐只改引用计数和哈希表。
  - 特例：`block_id=0` 开池即留作 `null_block` 占位，不维护引用计数、不可分配，实际可分配块为 `n−1`。

---

## 5. 系统运行期：一条请求的 KV Cache 生命周期

### 5.1 宏观：请求状态机与两侧调用链

一条请求由 Scheduler 在两个队列间推进：`WAITING`（等待首次 prefill）→ `RUNNING`（持续 decode）→ 生成结束后释放。prefill 只做一次、处理整段 prompt，decode 每步生成 1 个 token；二者共用同一套 step 骨架（`schedule → execute_model → sample`，内部都是"分配 slot → NPU 执行"），decode 只是规模更小并跳过前缀查找，系统里没有两套独立的管理逻辑。调度器在共享 token 预算下按"先 RUNNING、后 WAITING"逐请求填充。

```
WAITING ──首次调度──▶ RUNNING ──每步 decode──▶ 生成结束 ──▶ 释放
      （prefill，一次）         （EOS / max_tokens）
```

分配决策全部在逻辑侧沿五层编排链完成，产出整数 `block_id` 列表（`block_table`）；第 1 层物理张量不在这条链上——物理侧收到下发的 `block_table` 后，才由 CANN 注意力算子读写张量：

```
逻辑侧（引擎进程）：
Scheduler ─▶ 第5层 KVCacheManager ─▶ 第4层 Coordinator ─▶ 第3层 FullAttentionManager ─▶ 第2层 BlockPool
                       只改元数据（引用计数 / 哈希表 / 空闲队列），产出 block_table
                                             │ block_table 随 SchedulerOutput 下发
                                             ▼
物理侧（worker 进程 · NPU）：
CANN 注意力算子按 block_table 读写 K/V 双物理张量（第 1 层）
  reshape_and_cache 按 slot_mapping 把新 K/V scatter 进 k_cache / v_cache
  npu_fused_infer_attention_score(FIA) / _npu_paged_attention(PA) 按 block_table 读历史 K/V
```

### 5.2 时间线：prefill 一次，decode 循环

四个阶段中，前缀查找只在 prefill 出现一次；分配 slot 与 NPU 执行每个调度步都有；释放在生成结束时做一次。

**prefill（仅一次）：**

```
WAITING ── 入队：预计算链式哈希（不触碰任何管理对象）
   │
   ▼ 首次调度
前缀查找 ── 沿哈希链只读询问 BlockPool：命中仅标记可复用，ref_cnt 不变
   ▼
分配 slot ── touch 命中块（ref_cnt++、移出空闲队列）＋ 弹空闲块，拼出 block_table；写满的块记入哈希
   ▼
NPU 执行 ── 新块清零 → FIA 算子按 block_table 索引 K/V 双池，读命中前缀的 K/V、
   reshape_and_cache 写整段 prompt 的 K/V 到 k_cache / v_cache
   ▼
进入 RUNNING
```

**decode（每步循环，直到 EOS / max_tokens）：**

```
RUNNING
   ▼ 每步
分配 slot ── 弹空闲块拼出本步 block_table（每步新增 0 或 1 块）；写满的块记入哈希
   ▼
NPU 执行 ── 新块清零 → PA/FIA 算子读历史 K/V、reshape_and_cache 写本步 1 个 token 的 K/V
   │
   ├─ 未结束：回到分配 slot，进入下一步
   └─ EOS / max_tokens
   ▼
释放 ── 逆序归还、ref_cnt--；归零才回收，按 LRU 规则入空闲队列
```

### 5.3 空间分工：决策在逻辑侧，执行在物理侧

- **逻辑侧（第 2～5 层）承担全部分配决策**：前缀查找、分配 slot、释放只改引用计数、哈希表与空闲队列，全程不触碰物理张量。
- **物理侧（第 1 层）只在 NPU 执行时动张量**：这是两侧唯一的衔接时点——worker 清零本轮新块后，`AscendAttentionBackendImpl` 先 `reshape_and_cache` 把新 K/V 按 `slot_mapping` scatter 进 K/V 双池，再由 FIA/PA 算子按 `block_table` 索引物理行做注意力计算。

块的命运由此只由引用计数与空闲队列位置决定：前缀命中经 `touch`（计数加一、移出空闲队列）零拷贝共享给新请求；请求结束只减计数，仍被其他请求引用的块不回收；归零回收时，带哈希的块排到队尾受 LRU 保护、无哈希的块排到队首被优先复用——前缀缓存因此能跨请求持续积累，并被后续请求复用。

---

## 6. 扩展：其他注意力类型概览

本文主线是最基础的 Full Attention 模型。vLLM V1 同样支持以下场景，它们都在上述五层骨架上做扩展；NPU 侧（vllm-ascend）为多数场景提供了对应的 Ascend 后端实现：

| 类型 | 代表模型 | 主要差异 | 扩展落点（代码层 · NPU） |
|------|---------|---------|---------|
| **Sliding Window Attention (SWA)** | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐；前缀查找从右往左找窗口内命中 | 第 3 层滑窗管理器、第 4 层协调器（平台无关）；FIA 算子 `sparse_mode=4`/`pre_tokens` 走滑窗分支（attention_v1.py:1317-1334） |
| **Mamba/SSM** | Bamba、Jamba | 无 KV 只有 state，block 存 recurrent state 而非 K/V；缓存逻辑不同 | 第 3 层状态管理器；NPU 单张量分支（不走 K/V 双池），状态更新走 `npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule` 算子 |
| **混合模型 (Full + SWA/Mamba)** | Gemma3、Jamba、Llama4 | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool 但 page size 必须统一 | 第 4 层 `HybridKVCacheCoordinator`（平台无关）；NPU 侧 spec 收集时把 attn 层 `page_size_padded` 对齐到 mamba 页（model_runner_v1.py:5034-5036） |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | KV 低秩压缩，物理张量形状不同 | 第 1 层 NPU 特有 `AscendMLAAttentionSpec` + `AscendMLABackend`（mla_v1.py:80）；nope/rope 拆成两张独立 4D 张量（支持 prefill 分离传输） |
| **稀疏注意力 (DSA/SFA/C8)** | DeepSeek-V3.2 系 | attention 后含 indexer/稀疏选块逻辑，额外张量 | NPU `dsa_v1.py` / `sfa_v1.py` / `AscendC8AttentionBackendImpl`（attention_v1.py:1548），sparse MLA 额外分配 indexer/scale 张量 |
| **Cross-Attention** | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | 第 3 层 `CrossAttentionManager`（平台无关）；NPU `_forward_encoder_attention` 走 `npu_fusion_attention`（attention_v1.py:1384-1406） |
| **投机解码 (EAGLE/MTP)** | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑 | 第 4 层协调器与第 5 层门面；NPU PA 算子路径在 `using_paged_attention` 里对 speculative 直接关闭、回落 FIA（attention/utils.py:104-106） |
| **上下文并行 (CP)** | 长序列训练/推理切分 | 序列维切到多卡，注意力需跨卡 KV 交互 | NPU 专属 CP metadata/impl：`AscendAttentionCPImpl` / `AscendMlaCPImpl`（attention/context_parallel/） |

阅读建议：按 §3 的层号自底向上（第 1 层物理显存 → 第 2 层块池 → 第 3 层单类型管理 → 第 4 层协调器 → 第 5 层门面）吃透 Full Attention 主线，再对照本表按需理解复杂场景。
