# vLLM V1 KV Cache管理架构总览（NPU / vllm-ascend 视角 · MLA 路线）

> 主线是 **vllm-ascend 插件在昇腾 NPU 上的实现**。配置侧编排与逻辑侧（第 2～5 层）全部跑在引擎进程、平台无关；NPU 的平台特性全部集中在物理侧（worker 进程的第 1 层）与三处 Worker/ModelRunner 重写（① 算规格 / ② 测预算 / ④ 落张量，均由 `NPUWorker` / `NPUModelRunner` 承担），这也是本篇的叙述重点。
>
> 实战主线模型为 **DeepSeek-V2-Lite**（MLA，2026-10-18 于 a3 4 卡 pod `gggtest2` TP4 全流程实测）：其 KV 低秩压缩、latent/rope 双池 8:1 拆分与容量数字（48.25 GiB → 13,013 块 → 1,665,664 tokens，手算与日志一字不差）详见启动篇 [`2_start_init.md`](./2_start_init.md) §2.0/§3；部署日志与冒烟实测见 [`4_deploy_deepseekv2lite_tp4.md`](./4_deploy_deepseekv2lite_tp4.md)。正文以纯 Full Attention 讲五层骨架，MLA 差异见 §6 扩展表——**MLA 对第 1 层与 spec 类型做替换，第 2～5 层与逻辑管线零改动**。

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
 生成"小"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]† [K/V 是]† [K/V 小]

 prompt"你是谁"的 K/V 被重算 3 次，每步都从零算起；每步计算量 O(n)，n 步合计 O(n²)
```

KV Cache 把 attention 层已经算好的历史 K/V 保存在 **NPU 设备显存（HBM）**中，每步只计算新 token 的 K/V 并追加进缓存，总计算量降为 O(n)。attention 的 Q 只属于当前 token、FFN 没有历史状态，因此缓存对象只有 K/V；代价是显存占用随序列长度线性增长。

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
              │    KV Cache（NPU 设备显存）    │
              │  已缓存第 1…n 个 token 的 K/V  │
              │    每步只增量追加 1 个 token   │
              └──────────────────────────────┘
```

MLA 模型（DeepSeek-V2/V3）在此基础上把"存什么"改写：attention 的 K、V 全部由一个**低秩 latent 向量**生成（外加一段无法吸收的 rope），缓存对象从"逐头 K/V"变为"**每层一份 latent+rope**"——逐 token 因果追加的**节奏不变**，只是每步追加的内容被压缩了约 8.9 倍（实测数字见 §6 注）。K/V 的物理载体仍是 **NPU HBM**；**NPU 上 K 与 V 各占一张独立张量**（MLA 下分别是 latent 池与 rope 池，详见 §3.2 第 1 层），"缓存对象垂直增量追加"的结论不变。

---

## 2. KV Cache 如何管理

KV Cache 的显存占用随序列长度与并发请求数增长，相同前缀在不同请求间还会重复出现；不加管理的话，显存会碎片化、无法跨请求共享、块的分配与回收也没有统一规则。vLLM V1（vllm-ascend 完整复用这套管理框架）的 KV Cache 管理围绕三条核心设计：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（MLA 主线实测每 block 存 128 个 token），按块分配、回收和共享，彻底解决内存碎片问题。分页管理跑在引擎进程、平台无关。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。MLA 的 latent+rope 仍是**逐 token 因果追加**的序列，链式哈希照常成立——**与 GDN（递归状态）路线自动关闭前缀缓存形成分野**；实测命中率 61.3%→68.9%，见 [`3_runtime_sequence.md`](./3_runtime_sequence.md) §4。

---

## 3. vLLM V1 KV Cache 管理架构
### 3.1 启动期三侧架构

vLLM V1 的 KV Cache 管理按运行位置与职责分为三侧。**配置侧**（引擎进程）在启动期依次 ① 算规格、② 测预算、③ 做编排，产出一份 `KVCacheConfig`；这份配置下发两侧：**物理侧**（worker 进程）执行 ④ 落张量，**逻辑侧**（引擎进程）执行 ⑤ 建逻辑层，创建 `KVCacheManager`。物理侧与逻辑侧之间没有对象引用，仅靠 `block_id == 物理张量行号` 桥接；三侧的完整装配链路见 §4。

NPU 部署下三侧的角色分配：①② 的实现主体经 collective RPC 落在 **NPUWorker / NPUModelRunner**（vllm-ascend 插件重写）；③⑤ 跑在引擎进程，用的是**平台无关的上游代码**；④ 同样是 NPUWorker/NPUModelRunner 的重写实现。**MLA 在这条链上只改两处格子**：① 产出 `AscendMLAAttentionSpec`（NPU 包装的 MLA spec），④ 把每层整池按 8:1 拆 latent/rope 双池。

```
           ┌────────────────────────────────────────────────┐
           │ 配置侧（引擎进程 · 全程不碰显存）                 │
           │ ① 算规格 → 每层一个 AscendMLAAttentionSpec        │
           │ ② 测预算 → 每卡 KV 可用显存字节数（实测 48.25GiB） │
           │ ③ 做编排 → KVCacheConfig(num_blocks=13,013)     │
           └─────────────────────┬──────────────────────────┘
                                 │ 下发同一份 KVCacheConfig
                ┌────────────────┴─────────────────┐
                ▼                                  ▼
┌────────────────────────────────┐  ┌───────────────────────────────┐
│ 物理侧（worker 进程 · 每卡）   │  │ 逻辑侧（引擎进程）            │
│ NPUWorker + NPUModelRunner     │  │ ⑤ 建逻辑层 → KVCacheManager │
│ ④ 落张量：整池按 8:1 拆        │  │   new num_blocks 个逻辑块     │
│   latent 池(512 维) + rope 池  │  │   block_id 从 0 编号          │
│   (64 维)；开 KV 传输时 2MiB   │  │   对 Scheduler 暴露统一 API   │
│   对齐 → view 4D → 绑定各层    │  │                               │
│ block_id == 各张量行号         │  │                               │
└────────────────────────────────┘  └───────────────────────────────┘
两侧无对象引用，仅靠 block_id 桥接（详见 §4.5）
```

### 3.2 运行期五层架构

五层是运行期视角下同一套对象的纵向切分：第 1 层对应物理侧，第 2～5 层构成逻辑侧。运行期每个调度步，请求都沿"第 5 层门面 → 第 4 层协调器 → 第 3 层单类型管理 → 第 2 层块池"把分配决策做下去，最终由第 1 层物理张量承接读写。**MLA 模型与纯 Full Attention 一样只有一个 KV cache group**（27 层 spec 同构可 merge）——这使五层图对 MLA 主线完全适用。**五层自下而上编号**：第 1 层=物理显存 → 第 5 层=顶层门面；下图自上而下排列；`Scheduler` 是调用者，不算层。

```
┌───────────────────────────────────────────────────────────────┐
│  ▲ 运行期五层架构（MLA 视角 · 唯一 KV cache group）            │
├───────────────────────────────────────────────────────────────┤
│  Scheduler（调度器 · 调用者）                                 │
├───────────────────────────────────────────────────────────────┤
│  第5层 · 顶层门面   KVCacheManager                             │
│     持有 1 个第4层；对 Scheduler 暴露统一 API                   │
├───────────────────────────────────────────────────────────────┤
│  第4层 · 协调器    UnitaryKVCacheCoordinator（单组直通）        │
│     · 持有 N 个第3层（主线 N=1） + 1 个第2层 BlockPool          │
├───────────────────────────────────────────────────────────────┤
│  第3层 ×N · 单类型管理  FullAttentionManager                    │
│     前缀查找(链式哈希)/分配释放/block_table 维护                │
├───────────────────────────────────────────────────────────────┤
│  第2层 ×1 · 逻辑块池  BlockPool（唯一，所有第3层共享）          │
│     FreeKVCacheBlockQueue(LRU) + BlockHashToBlockMap(链式哈希) │
├───────────────────────────────────────────────────────────────┤
│  第1层 · 物理显存  NPUModelRunner.kv_caches[layer]            │
│     MLA：每层 (latent_cache, rope_cache) 二元组：              │
│     (13013,128,1,512) / (13013,128,1,64)，block_id == 行号     │
└───────────────────────────────────────────────────────────────┘
```

**NPU 第 1 层物理显存要点**（MLA 主线速查，详见 [`1_kvcache_type.md`](./1_kvcache_type.md) 中篇）：

| 要点 | 值 |
|--------|--------------------|
| 每层类型 | **二元组**：K 位装 **latent 池（kv_lora_rank=512 宽）**、V 位装 **rope 池（qk_rope_head_dim=64 宽）**——源码注释 `k_cache: nope_cache    v_cache: rope_cache`（model_runner_v1.py:4606） |
| 每池 4D shape | 固定 `(num_blocks, block_size, 1, head_dim)`，MLA 实测 `(13013, 128, 1, 512)` 与 `(13013, 128, 1, 64)` |
| 字节池申请 | latent/rope **两次独立** `torch.zeros(int8)`：整层 1.79 GiB 按 `calc_split_factor([512,64])=(1.125,9.0)` **8:1 拆分**；开 KV 传输时各多申请 2MiB 后对齐切片 |
| reshape | 纯 **`view`，无 permute**（MLA 后端描述符直接 4D，无 dense 后端 dim0 的"2"） |
| 写 KV 算子 | 双池入参——MLA 层传入的 key/value 实为 `(c_KV, k_rope)`（编译算子 `vllm::unified_mla_kv_cache_update`） |
| value 去向 | **V 不落池**：decode 时由 latent 经 `W_UV` 即时上采样（mla_v1.py:898-906）——"V 池"是物理拆分名义位，装的是 k_rope |

**五层分篇详解（自底向上）：**

| 层 | 主题（MLA 主线） |
|---|---|
| 第 1 层 · 物理显存层（最底） | latent/rope 双 4D 池的申请（8:1 拆分）、reshape，`block_id == 各张量行号` 桥接 |
| 第 2 层 · 逻辑块池层 | `KVCacheBlock`、空闲队列、链式哈希表、`BlockPool` 分配/释放/缓存/驱逐（**与纯 FA 逐字相同**） |
| 第 3 层 · 单类型管理层 | `FullAttentionManager` 前缀查找/分配/释放（MLA spec 注册指向它，零定制） |
| 第 4 层 · 协调器层 | `UnitaryKVCacheCoordinator`（单组直通），混合模型协调器作为扩展 |
| 第 5 层 · 顶层接口层（最顶） | `KVCacheManager` + `KVCacheBlocks`，Scheduler 唯一入口，完整请求生命周期 |

**文本关系（实线 ─ = 持有，虚线 ╎ = 非持有的桥接）**：第 5 层持有 **1** 个第 4 层 → 第 4 层持有 **N** 个第 3 层 ＋ **1** 个第 2 层 → 每个第 3 层引用**同一个**第 2 层。持有关系到第 2 层为止：第 2 层在引擎进程、第 1 层在 worker 进程，两者之间没有任何对象引用，只有 `block_id` 桥接。

```
Scheduler（调度器 · 调用者）
└─1→ KVCacheManager（第5层 · 顶层门面）── 对 Scheduler 暴露统一 API
    └─1→ UnitaryKVCacheCoordinator（第4层 · 单组直通）
        ├─1→ 第3层 FullAttentionManager（前缀查找/分配释放/block_table 维护）
        └─1→ 第2层 BlockPool（唯一，所有第3层共享）── 仅索引 block_id
             │
             ╎ 桥接（不是持有）：同一份 KVCacheConfig 下发两侧——
             ╎   物理侧（_allocate_kv_cache_tensors，model_runner_v1.py:4082）：
             ╎     MLA 层 dims 取 (512,64)，calc_split_factor 8:1 拆两次 torch.zeros(int8)
             ╎     （开 KV 传输时 2MiB 对齐），view(dtype).view(4D) 后登记为
             ╎     kv_cache_raw_tensors[layer] = (latent_tensor, rope_tensor)
             ╎   逻辑侧：new num_blocks 个逻辑块，block_id 取序号 0…n−1
             ╎ 两者约定 block_id 从 0 编号、即物理行号
             ▼
             第1层 物理显存 NPUModelRunner.kv_caches[layer] = (latent_cache, rope_cache)
                （真正读写的是 Ascend MLA 算子：写 (c_KV,k_rope) 进双池、
                  decode 端 latent 经 W_UK/W_UV 展开各头 K/V——调度器从不直接碰张量）
```

### 3.3 关键文件职责

| 文件 | 职责 | 所属层 |
|------|------|------|
| `vllm/vllm/v1/core/kv_cache_manager.py` | 顶层门面，对 Scheduler 暴露统一接口（`get_computed_blocks`/`allocate_slots`/`free` 等） | 第 5 层 · 顶层门面 |
| `vllm/vllm/v1/core/kv_cache_coordinator.py` | 协调器：单组直通（**MLA 与纯 FA 同为单组**）或多组对齐（混合模型)；含工厂 `get_kv_cache_coordinator` | 第 4 层 · 协调器 |
| `vllm/vllm/v1/core/single_type_kv_cache_manager.py` | `FullAttentionManager`：前缀查找、block 分配/释放；含工厂 `get_manager_for_kv_cache_spec` 与注册表 | 第 3 层 · 单类型管理 |
| `vllm/vllm/v1/core/block_pool.py` | 逻辑 block 池：分配/释放/缓存哈希/LRU 驱逐 | 第 2 层 · 逻辑块池 |
| `vllm/vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`、`BlockHash`、空闲队列、block hash 计算工具；容量规划 `get_num_blocks`/`get_kv_cache_configs` | 第 2/1 层 |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | **NPU 物理显存申请**：`NPUModelRunner`，MLA 层 latent/rope **8:1 双池** int8 申请（2MiB 条件对齐）→ 纯 view 成 4D → 绑定 | 第 1 层 · 物理层 |
| `vllm-ascend/vllm_ascend/worker/worker.py` | **NPUWorker**：①②④ 三步的 RPC 实现侧（`get_kv_cache_spec`/`determine_available_memory`/`initialize_from_config`/`compile_or_warm_up_model`） | 第 1 层 · 物理层 |
| `vllm-ascend/vllm_ascend/attention/mla_v1.py` | `AscendMLABackend`：描述符恒 4D（:99-106）、kernel 分档 `[128]`（:117-118）；`AscendMLAImpl` 吸收式 decode（:707 起，`forward_mha/mqa` :1700-1721、`_v_up_proj` :898-906） | 第 1 层 · 物理层 |
| `vllm/vllm/v1/kv_cache_interface.py` | `KVCacheSpec` / `FullAttentionSpec` / `MLAAttentionSpec` / `KVCacheConfig` 定义（含 `real_page_size_bytes`） | 第 1 层 · 物理层 |
| `vllm-ascend/vllm_ascend/core/kv_cache_interface.py` | **`AscendMLAAttentionSpec`**（上游 MLA spec 的 NPU 扩展：latent/rope 语义、DSA sparse 字段）、`HiddenStateCacheSpec` 补丁（`register_ascend_kv_cache_specs` :305-315 注册） | 第 1 层 · 物理层（NPU 特有 spec） |

> 说明：上游 vllm 与 vllm-ascend 以插件方式协作——引擎进程全部跑上游代码（配置编排、逻辑侧第 2～5 层），worker 进程的平台行为由 `vllm_ascend` 插件提供（`NPUWorker`/`NPUModelRunner`）。两仓行号引用分别以 `vllm/`、`vllm-ascend/` 起始。**路径勘误**：`AscendMLAAttentionSpec` 实际定义于 `vllm_ascend/core/kv_cache_interface.py`。

---

## 4. 系统启动期：三侧装配

> 本节只讲**装配主线**：三侧对象按什么顺序、在哪个进程里被创建，彼此靠什么衔接。NPU 视角的逐步调用链、代码片段、2MiB 对齐与 KV 传输注册细节属于启动篇（`2_start_init.md`），块池内部结构属于类型篇（`1_kvcache_type.md`），本节均不展开。

### 4.1 装配全景：规格先行，物理先于逻辑

启动装配是一条单向流水线，横跨**引擎进程**与 **worker 进程**：配置侧 ①～③ 在引擎进程完成"规格推导"（全程不碰显存），物理侧 ④ RPC 到每卡（NPU）完成"物理落地"，RPC 返回后逻辑侧 ⑤ 才在引擎进程创建 Scheduler 及其逻辑三层。入口为 `EngineCore._initialize_kv_caches()`。

```
引擎进程                                                        worker 进程（每卡 NPU）
① 算规格  get_kv_cache_specs()          ── RPC ──▶  NPUModelRunner 遍历 attention 层
                                                        MLAAttention 层：上游 MLAAttentionSpec
                                                        → 包装为 AscendMLAAttentionSpec
                                                        (num_kv_heads=1, head_size=576)
② 测预算  determine_available_memory()  ── RPC ──▶  profile_run 实测本卡 KV 可用显存
                                                        （NPUGraph 内存不在此步扣，推迟到 4d 事后记账）
③ 做编排  get_kv_cache_configs()：27 层 MLA spec 同构合并为 1 个 group
          → 算 num_blocks → 多卡取 min 对齐，产出 KVCacheConfig（平台无关，引擎进程）
④ 落张量  initialize_from_config()      ── RPC ──▶  每层整池按 8:1 拆 latent 池(512) + rope 池(64)
                                                        两次 torch.zeros(int8)（开 KV 传输时 2MiB
                                                        对齐）→ view 成两张 4D → bind 到各 MLA 层
          第二个 RPC：compile_or_warm_up_model ──▶  _dummy_run 预热 + NPUGraph capture
                                                    （worker.py:706 → :732）
⑤ 建逻辑层 Scheduler(...)（引擎进程，与平台无关）
          └ KVCacheManager → Coordinator → BlockPool + FullAttentionManager
```

### 4.2 配置侧：① 算规格 ② 测预算 ③ 做编排（引擎进程发起，全程不碰显存）

| 步骤 | 产物 | 架构级结论（NPU 视角 · MLA） |
|------|------|-----------|
| ① 算规格 | 每层一个 `AscendMLAAttentionSpec` | `NPUWorker.get_kv_cache_spec()` 一行转发（worker.py:891）；`NPUModelRunner.get_kv_cache_spec()`（model_runner_v1.py:4909）重写了上游实现，**MLAAttention 层走 :4948-5000 分支**：dense 路径（:4987-4999）从上游 spec 转包装，冻结四字段 `block_size=128 / num_kv_heads=1 / head_size=576 / dtype=bf16`。**TP4 只切 Q 头、不切 latent → 4 rank 同层 spec 全等**（合并断言天然通过）。NPU 特有分流另有：CacheOnly→`HiddenStateCacheSpec`、Mamba 页对齐 |
| ② 测预算 | 每卡一个可用显存字节数 | `NPUWorker.determine_available_memory()`（worker.py:527）dummy forward 实测：总显存 × 利用率 − 权重 − 激活（**公式里没有图内存项**）；NPUGraph 内存推迟到 4d capture 后以"建议 `--kv-cache-memory`"日志事后记账（worker.py:741-771）。MLA 实测 **48.25 GiB**（权重 7.45 GB + 激活 0.15 + non-torch 0.51，图内存 0.57~0.58 GiB 事后核销） |
| ③ 做编排 | `KVCacheConfig`（含 `num_blocks`） | 平台无关：单层单块字节 `page_size = block_size × num_kv_heads × (head_size × dtype + …)`，MLA 实测 `128 × 1 × 576 × 2B = 147,456 B`；本卡 `num_blocks = 可用显存 // page_size // 本 worker 层数`，实测 `48.25 GiB // 147,456 // 27 = 13,013`；多卡取最小值，共享同一套 `block_id` 空间 |

引擎算出 `num_blocks` 后写入 `KVCacheConfig`，同一份配置下发物理侧与逻辑侧，两侧各自做自己的事。

### 4.3 物理侧：④ 落张量（worker 进程 · 每卡）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ④ 落张量 | 每 worker 每层 **2 张**物理张量（latent、rope 各一；PP 下只含本 worker 负责的层） | 先按字节数申请：MLA 层 `_get_attention_kv_cache_dims`（model_runner_v1.py:3987-3989，注释原文 `DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim`）取 `(512, 64)`，`calc_split_factor([512,64])`（vllm_ascend/utils.py:1611）得拆分比例 **(1.125, 9.0)**——latent 池占 8/9、rope 池占 1/9，两块存储**不连续**；每层 1.79 GiB 整池 → latent ≈1.59 GiB / rope ≈0.20 GiB。再零拷贝 `view` 成后端 4D shape；`NPUWorker.initialize_from_config()`（worker.py:907）先拉起 KV 传输组（`ensure_kv_transfer_initialized`），开 sleep mode 时在 `CaMemAllocator` 内存池上下文内分配 |

物理侧只按配置执行、不自行算块数，逐条执行细节见 4.5 衔接点与启动篇 §3。

### 4.4 逻辑侧：⑤ 建逻辑层（引擎进程）

| 步骤 | 产物 | 架构级结论 |
|------|------|-----------|
| ⑤ 建逻辑层 | 第 2～5 层全部对象 | 构造链路：`Scheduler → KVCacheManager → Coordinator → BlockPool + FullAttentionManager`——引擎进程内平台无关的上游代码。**MLA 单 group → 与纯 FA 完全同形**：Unitary 协调器 + 每 group 一个 manager（插件注册表为 `AscendMLAAttentionSpec` 指到 `FullAttentionManager`）；前缀缓存开启（实测 `enable_prefix_caching=True`） |

逻辑侧只建元数据对象，逐条执行细节见 4.5 衔接点。

### 4.5 衔接点：一份配置定容量，`block_id` 即行号

引擎算出 `num_blocks` 后写入 `KVCacheConfig` 并下发两侧，两侧各自做自己的事：

- **物理侧（NPU worker，只按配置执行，不自行算块数）**
  - 遍历 `KVCacheConfig.kv_cache_tensors`；MLA 层：取 `(k_dim, v_dim) = (512, 64)`，`latent_size = size // 1.125`（8/9）、`rope_size = size // 9.0`（1/9），各调一次 `_allocate_int8_cache_tensor`（model_runner_v1.py:4004：无 KV 传输时普通 `torch.zeros`；开传输时多申请 2MiB 后 `data_ptr` 向上对齐切片）；
  - 对照：等宽 dense 层（Llama 128/128）对半拆；例外分支 `linear_attn`/mamba/`cache_only_layers`/hidden-state 与 `use_compress` 压缩注意力走**单张量** int8 池（model_runner_v1.py:4118-4146）；sparse MLA（DSA/SFA）额外分配 indexer/scale 张量（2～4 元组登记，:4646-4680）；
  - 源码：`model_runner_v1.py:4082-4309`（分配执行）、`mla_v1.py:99-106`（后端 4D 描述符）、`:4605-4629`（MLA 双池构造：k_shape=(13013,128,1,512)、v_shape=(13013,128,1,64)）。
- **逻辑侧（引擎进程的 BlockPool，只建元数据对象）**
  - new 出 `num_blocks` 个 `KVCacheBlock`，`block_id` 取序号 `0 … n−1`；每个对象只带 `block_id`、引用计数等元数据，**不含任何 latent/rope 数据**——也不知道每行装的是 K/V 还是 latent/rope。源码：`vllm/vllm/v1/core/block_pool.py:162-164`。
- **桥接结果**：同一份配置保证两侧容量相等，"从 0 顺序编号"的约定让 `block_id` 直接等于物理张量行号——**latent 池与 rope 池各有一套同序号的行空间**，同一个 `block_id` 在两张张量里都对应本块的 latent 行与 rope 行。此后物理张量不再变动，分配/共享/驱逐只改引用计数和哈希表。**实测数字闭环**：`13,013 × 27 层 × 147,456 B/层 = 51,808,813,056 B = 48.25 GiB`，`13,013 × 128 = 1,665,664 tokens` 与启动日志一字不差（运行期再添 `0.1% = 12 块/13,013` 闭环）。
  - 特例：`block_id=0` 开池即留作 `null_block` 占位，不维护引用计数、不可分配，实际可分配块为 `n−1`。

---

## 5. 系统运行期：一条请求的 KV Cache 生命周期

### 5.1 宏观：请求状态机与两侧调用链

一条请求由 Scheduler 在两个队列间推进：`WAITING`（等待首次 prefill）→ `RUNNING`（持续 decode）→ 生成结束后释放。prefill 只做一次、处理整段 prompt，decode 每步生成 1 个 token；二者共用同一套 step 骨架（`schedule → execute_model → sample`），decode 只是规模更小并跳过前缀查找。调度器在共享 token 预算下按"先 RUNNING、后 WAITING"逐请求填充。

```
WAITING ──首次调度──▶ RUNNING ──每步 decode──▶ 生成结束 ──▶ 释放
      （prefill，一次）         （EOS / max_tokens）
```

分配决策全部在逻辑侧沿五层编排链完成，产出整数 `block_id` 列表（`block_table`）；第 1 层物理张量不在这条链上——物理侧收到下发的 `block_table` 后，才由 Ascend MLA 算子读写张量：

```
逻辑侧（引擎进程 · 平台无关）：
Scheduler ─▶ 第5层 KVCacheManager ─▶ 第4层 Coordinator ─▶ 第3层 FullAttentionManager ─▶ 第2层 BlockPool
                       只改元数据（引用计数 / 哈希表 / 空闲队列），产出 block_table
                                             │ block_table 随 SchedulerOutput 下发
                                             ▼
物理侧（worker 进程 · NPU）：
AscendMLAImpl 先清零本轮新块（latent 池 + rope 池两张各清一遍）
→ forward 时：
  unified_mla_kv_cache_update 按 slot_mapping 把新 (c_KV, k_rope) 分别 scatter 进
  latent_cache / rope_cache 双池 → 注意力核按 block_table 读历史 latent/rope，
  decode 端 latent 经 W_UK/W_UV 即时展开 16 头 K(nope)/V —— V 从不落地 cache
```

### 5.2 时间线：prefill 一次，decode 循环

四个阶段中，前缀查找只在 prefill 出现一次；分配 slot 与设备执行每个调度步都有；释放在生成结束时做一次。

**prefill（仅一次）：**

```
WAITING ── 入队：预计算链式哈希（不触碰任何管理对象；MLA 下同样只由 token 序列决定）
   │
   ▼ 首次调度
前缀查找 ── 沿哈希链只读询问 BlockPool：命中仅标记可复用，ref_cnt 不变
   ▼
分配 slot ── touch 命中块（ref_cnt++、移出空闲队列）＋ 弹空闲块，拼出 block_table；写满的块记入哈希
   ▼
NPU 执行 ── 新块清零（双池）→ 写整段 prompt 的 (c_KV, k_rope) 进 latent/rope 双池、
   │          注意力核按 block_table 读命中前缀（prefill 走非吸收式并行核）
   ▼
进入 RUNNING
```

**decode（每步循环，直到 EOS / max_tokens）：**

```
RUNNING
   ▼ 每步
分配 slot ── 弹空闲块拼出本步 block_table（每步新增 0 或 1 块）；写满的块记入哈希
   ▼
NPU 执行 ── 新块清零 → 吸收式核：读历史 latent+rope、latent 经 W_UK/W_UV 展开各头 K/V、
   │          写本步 1 个 token 的 (c_KV, k_rope) 进双池
   ├─ 未结束：回到分配 slot，进入下一步
   └─ EOS / max_tokens
   ▼
释放 ── 逆序归还、ref_cnt--；归零才回收，按 LRU 规则入空闲队列
```

### 5.3 空间分工：决策在逻辑侧，执行在物理侧

- **逻辑侧（第 2～5 层）承担全部分配决策**：前缀查找、分配 slot、释放只改引用计数、哈希表与空闲队列，全程不触碰物理张量——这部分是**引擎进程内的同一份平台无关代码**；MLA 与纯 FA 的请求在这条链上走完全相同的路径（实测：前缀缓存命中 61.3%→68.9%、命中率数字与块账本互证，见 [`3_runtime_sequence.md`](./3_runtime_sequence.md) §4）。
- **物理侧（第 1 层）只在设备执行时动张量**：worker 清零本轮新块（**latent 池与 rope 池两张各清一遍**）后，MLA 写入路径把新 token 的 `(c_KV, k_rope)` 按 `slot_mapping` 分别写进 latent_cache/rope_cache 双池（slot 公式 `block_id × block_size + 块内偏移` 与主线全同——**同一个 slot 值同时定位两池**）；注意力计算核按 `block_table` 索引物理行读历史 latent/rope。块的命运由此只由引用计数与空闲队列位置决定：前缀命中经 `touch` 零拷贝共享给新请求；请求结束只减计数；归零回收时，带哈希的块排到队尾受 LRU 保护、无哈希的块排到队首被优先复用——前缀缓存因此能跨请求持续积累。

---

## 6. 扩展：其他注意力类型概览

本文主线是 **MLA 模型**（DeepSeek-V2-Lite）。vllm-ascend 在五层骨架上同样支持以下场景（NPU 特有落点以 **粗体** 标注）：

| 类型 | 代表模型 | 主要差异 | 扩展落点（代码层） |
|------|---------|---------|---------|
| **Full Attention / GQA（对照主线）** | Llama-3 系列 | 逐头存 K/V；**NPU 上 K、V 双独立 4D 池（等宽对半）**，本系列作为纯 FA 对照示例贯穿各篇 | 第 1 层 dense 双 pool（attention_v1.py） |
| **MLA (Multi-head Latent Attention)（本系列主线）** | **DeepSeek-V2/V3** | KV 低秩压缩；**NPU 上 spec 为 `AscendMLAAttentionSpec`（num_kv_heads=1、head_size=576），物理张量为 latent(512)/rope(64) 两张独立 4D、8:1 拆分**（model_runner_v1.py:4605-4629；mla_v1.py backend `AscendMLABackend`）；**decode 吸收式：V 由 latent 经 W_UV 即时展开不落池**；sparse DSA/SFA 再挂 indexer/scale 张量（2~4 元组） | 第 1 层 MLA 规格与张量布局；编译算子 `vllm::unified_mla_attention_with_output / unified_mla_kv_cache_update / mla_forward` |
| Sliding Window Attention (SWA) | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐 | 第 3 层滑窗管理器、第 4 层协调器 |
| Mamba/SSM | Bamba、Jamba、Qwen3-Next | 递归状态无前缀缓存；**NPU 上走单张量分配分支**（`linear_attn` 层名命中，model_runner_v1.py:4118） | 第 3 层状态管理器，第 1 层张量布局随之改变 |
| 混合模型 (Full + SWA/Mamba) | Gemma3、Jamba、**Qwen3.5-4B（GDN，兄弟系列主线）** | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool；**NPU 侧 ① 算规格时把 attn 层 `page_size_padded` 对齐到 mamba 页**（model_runner_v1.py:5027-5036），块抬到 128 整数倍（实测 1024）、前缀缓存自动关闭 | 第 4 层 `HybridKVCacheCoordinator` |
| Cross-Attention | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | 第 3 层 `CrossAttentionManager` |
| 投机解码 (EAGLE/MTP) | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑；**NPU 上 CacheOnly 层以 `HiddenStateCacheSpec` 单独成组**（model_runner_v1.py:5005-5025） | 第 4 层协调器与第 5 层门面 |
| P/D 分离 / KV 传输 | Mooncake、ADXL | prefill/decode 实例间搬 KV；**NPU 上这是 latent/rope（dense K/V）双池拆分与 2MiB 对齐的动机**（buffer 注册要求 dim0=num_blocks、地址对齐） | 第 1 层分配（model_runner_v1.py:4004/:4087-4088）、worker 侧 `register_kv_caches`（:3883-3884） |

阅读建议：先吃透五层主线（第 1 层物理显存 → 第 2 层块池 → 第 3 层单类型管理 → 第 4 层协调器 → 第 5 层门面）——**MLA 与纯 FA 在第 2～5 层逐字相同**，再到本系列的 [`2_start_init.md`](./2_start_init.md) §3 精读 NPU 物理侧的 latent/rope 双池实现，最后按本表按需理解复杂场景（GDN 混合路线见兄弟系列 `../docs_npu_gdn/`）。
