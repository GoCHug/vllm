# vLLM-Ascend（NPU）系统启动期初始化

本文讲 Ascend NPU 上**系统启动期的 KVCache 管理初始化**，统一以 hybrid 模型 **Qwen3.5-4B**（架构 `Qwen3_5ForConditionalGeneration`）为例。

**为什么选 Qwen3.5-4B**：它是 Gated DeltaNet（线性注意力）与 Full Attention 的混合体——32 层里 24 层 `linear_attention`（GDN，产出 `MambaSpec`）+ 8 层 `full_attention`（产出 `FullAttentionSpec`）；两类 spec 无法 merge，且分组时要求每组层数相等（取 `min(8,24)=8`），于是把 24 层 mamba 平分成 3 组 → 全模型 **4 个 KV cache group（1 组 attn + 3 组 mamba）**（多组编排 + 多组协调器 + 两种物理形态），是启动链最完整、最能暴露混合模型差异的样本。

**模型配置**：

| 配置项 | 值 | 含义 / 对 KVCache 的影响 |
|---|---|---|
| `num_hidden_layers` | **32** | 24 `linear_attention` + 8 `full_attention`（`full_attention_interval=4`，每 3 层 GDN 插 1 层全注意力） |
| full attn：`num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 4 / 256 | GQA，4 个 KV 头；TP2 后每 rank 2 头（每 token K 页 = 2 × 256 × bf16 = 1 KB） |
| GDN：`linear_num_key_heads` / `linear_num_value_heads` / `head_dim` / `conv_kernel` | 16 / 32 / 128 / 4 | 递归状态；块内**存 state 而非 token K/V**（TP2 后 value 头 16） |
| `hidden_size` / `intermediate_size` / `vocab_size` | 2560 / 9216 / 248320 | 4B 级小模型 |
| `mamba_ssm_dtype` | float32 | GDN 状态精度（决定 temporal 状态页 = 1 MB） |
| `max_model_len` | 32768 | 前缀缓存关闭时 mamba 组的 `mamba_block_size`（§2） |
| `mtp_num_hidden_layers` | 1 | 含 1 层 MTP（本次未启投机解码） |

**环境一览**：

| 项 | 值 |
|---|---|
| vllm / vllm-ascend | 0.23.0 / 0.23.0 |
| 模型 | Qwen3.5-4B |
| 并行策略 | PP2 × TP2 |

服务启动命令如下：

```bash
vllm serve /xxx/qwen3-5-4b --pipeline-parallel-size 2 --tensor-parallel-size 2 --served-model-name qwen3.5-4b --port 8100 --max-model-len 32768 > vllm_qwen35_pp2tp2.log 2>&1 &
```

---

## 1. 系统启动期初始化总览

启动期的初始化分为两部分——物理张量初始化 / 逻辑管理器初始化：

### 1.1 vllm serve 服务启动全栈与进程拓扑（1 API server + 1 EngineCore + N × worker）

> **进程拓扑**：

| 进程 | 实测 pid | 谁拉起的 | 跑什么 |
|---|---|---|---|
| **[P0] API server 主进程** | 20424（APIServer） | shell | **CLI 入口也在此进程**（`vllm serve` 命令不单开进程，uvicorn 常驻到退出）；config 期、路由注册 |
| **[P1] EngineCore 进程** | 20736（EngineCore） | P0 spawn | 引擎调度 busy-loop；再 spawn workers |
| **[P2..P5] worker ×N**（N=PP×TP=4） | 20829-20832（Worker_PP0_TP0…PP1_TP1） | P1 的 executor spawn | S1 算规格 / S2 测预算 / S4 落张量 / S5 执行——全部 [vllm-ascend] 实现体 |

```text
$ vllm serve <qwen3.5-4b> --pipeline-parallel-size 2 --tensor-parallel-size 2 --port 8100
│
▼ 一、CLI 分派 ════════════════ 进程 [P0 20424]（CLI 与 API server 同进程）═══════  [vllm]
│  ServeSubcommand.cmd(args)                  cli/serve.py:74                [P0]
│  └─ 单 server 分支（DP=1 → api_server_count 默认 1）
│     uvloop.run(run_server(args))             cli/serve.py:148              [P0]
│
▼ 二、API server async 启动 ══════════ 仍进程 [P0 20424] ═══════════════════════  [vllm]
│  run_server(args) / setup_server(args)（绑端口）         api_server.py:652/664  [P0]
│  └─ run_server_worker(...)                              api_server.py:665     [P0]
│     ├─ build_async_engine_client(args)                  api_server.py:679      [P0] [vllm]
│     │  └─ build_async_engine_client_from_engine_args(...) api_server.py:99      [P0] [vllm]
│     │     ├─ AsyncEngineArgs.from_cli_args(args)（参数搬移，block_size=None） api_server.py:94   [P0]
│     │     ├─ engine_args.create_engine_config()  ★config 期整个栈        api_server.py:123 → §2.1 · 0a 逐帧树
│     │     └─ AsyncLLM.from_vllm_config(vllm_config, ...)                api_server.py:135  [P0] [vllm]
│     │        └─ ★进程边界 1：P0 spawn 子进程 →
│     │           ══ 进程 [P1 EngineCore 20736] ══ EngineCoreProc.run_engine_core
│     │           EngineCore.__init__               v1/engine/core.py:119          [P1] [vllm]
│     │           ├─ Executor选型 + spawn N=4 worker 子进程             core.py:122  [P1] [vllm]
│     │           │  └─ ★进程边界 2：P1 spawn N 个 →
│     │           │     ══ 进程 [P2..P5 workers 20829-20832] ══
│     │           │     NPUWorker 设备初始化/模型加载/S1、S2 实现体      vllm_ascend/worker/  [P2-P5] [vllm-ascend]
│     │           └─ _initialize_kv_caches S1–S5        v1/engine/core.py:132    [P1] [vllm]
│     │              S2/S4 RPC 逐 worker 下发；S4 在 P2-P5 内执行
│     │              （展开 = §1.3 总调用链；时间线 07:52:45→07:55:17 = 部署篇 §2.4）
│     └─ build_and_serve(engine_client, ...)             api_server.py:683        [P0] [vllm]
│        └─ serve_http(app, ...)（uvicorn）              api_server.py:585 → launcher.py  [P0]
│
▼ 三、服务就绪 ══════════════════ 进程 [P0]（uvicorn 常驻）═════════════
│  "Started server process → Application startup complete"（07:55:17；P2-P5 headless 无此日志属正常）
│
└─ 请求期跨进程链：chat 路由（P0）→ 队列 → EngineCore 调度（P1）→ workers 执行（P2-P5）→ 回包（P0）
```

### 1.2 启动期时序图（物理侧 / 逻辑侧）

`EngineCore.__init__` 内先调 `_initialize_kv_caches`（core.py:132，定义见 :236），由它完成**物理侧 S1–S4**（:290），并返回 scheduler 版 `kv_cache_config`（:317）；随后引擎凭该 `kv_cache_config` 进行**逻辑侧 S5** 初始化（core.py:145-157）。以主线 Qwen3.5-4B pp2tp2 为例（每 worker 16 层 = 12 linear + 4 full attn、full attn 2 KV 头；**本仓库 NPU 实测**：`Available KV cache memory: 51.90 GiB`、attn `block_size=1024`、`num_blocks≈6003`）。

#### 物理侧（物理张量初始化 · S1 算规格 → S2 测预算 → S3 做编排 → S4 落张量）

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "mirrorActors": true}}}%%
sequenceDiagram
    participant EngineCore
    participant ModelExecutor
    participant NPUWorker
    participant NPUModelRunner

    Note over EngineCore,NPUModelRunner: S1 算规格：NPUModelRunner 重写版遍历各层
    EngineCore->>ModelExecutor: get_kv_cache_specs()
    ModelExecutor->>NPUWorker: collective_rpc("get_kv_cache_spec")
    NPUWorker->>NPUModelRunner: get_kv_cache_spec()
    Note over NPUModelRunner: 遍历 attention 层（每 worker 16 层）<br/>4 个 full-attn 层原样收 FullAttentionSpec(block_size=1024, num_kv_heads=2, head_size=256)<br/>12 个 linear 层收 MambaSpec；并把 attn 层 page_size_padded 对齐到 mamba 页
    NPUModelRunner-->>NPUWorker: dict[layer, FullAttentionSpec | MambaSpec]
    NPUWorker-->>ModelExecutor: 本卡 spec（RPC 聚合各卡）
    ModelExecutor-->>EngineCore: kv_cache_specs

    Note over EngineCore,NPUModelRunner: S2 测预算：复用上游 profiling 工具，无图项
    EngineCore->>ModelExecutor: determine_available_memory()
    ModelExecutor->>NPUWorker: collective_rpc("determine_available_memory")
    NPUWorker->>NPUModelRunner: profile_run()
    Note over NPUWorker,NPUModelRunner: dummy forward 实测<br/>available = total×util − weights − activations<br/>（NPUGraph 不在此步扣，推迟到 4d 事后记账）
    NPUModelRunner-->>NPUWorker: dummy run 完成
    NPUWorker-->>ModelExecutor: 本卡 available（字节）
    ModelExecutor-->>EngineCore: available_npu_memory

    Note over EngineCore: S3 做编排（在 EngineCore 内完成，平台无关）
    EngineCore->>EngineCore: get_kv_cache_configs(...)
    Note over EngineCore: 两类 spec 无法 merge → 4 个 group<br/>num_blocks 按 projected 组计算、跨组 min 协调<br/>跨 worker 取 min 对齐；收口 deepcopy worker[0] → KVCacheConfig(num_blocks≈6003, 4 groups)

    Note over EngineCore,NPUModelRunner: S4 落张量：单张连续张量 → K/V 双池 + 状态视图 + NPUGraph
    EngineCore->>ModelExecutor: initialize_from_config(kv_cache_configs)
    ModelExecutor->>NPUWorker: collective_rpc("initialize_from_config")
    NPUWorker->>NPUWorker: ensure_kv_transfer_initialized()（P/D 分离传输组）
    NPUWorker->>NPUModelRunner: initialize_kv_cache(config)
    Note over NPUModelRunner: 4a 每张共享张量按整块（2,072 KB×nb）申请单张连续张量<br/>4b attn 切头 conv padding → 对半切 K/V 双池 → view(nb×8,128,2,256)；linear 层切 conv/ssm 状态<br/>4c 复用上游 bind_kv_cache
    NPUModelRunner-->>NPUWorker: 完成（张量落卡）
    NPUWorker-->>ModelExecutor: 完成
    ModelExecutor->>NPUWorker: collective_rpc("compile_or_warm_up_model")
    Note over NPUWorker: 4d 逐 size _dummy_run + NPUGraph capture<br/>＋ 图内存事后记账（打印 --kv-cache-memory 建议值）
    NPUWorker-->>ModelExecutor: 完成
    ModelExecutor-->>EngineCore: RPC 返回
    EngineCore-->>EngineCore: _initialize_kv_caches 返回 scheduler 版 kv_cache_config（:317）
```

#### 逻辑侧（逻辑管理器初始化 · S5，引擎进程，物理侧 RPC 全部返回后）

```mermaid
%%{init: {"themeVariables": {"actorFontSize": "11px", "messageFontSize": "11px", "noteFontSize": "11px"}, "sequence": {"actorMargin": 40, "messageMargin": 16, "noteMargin": 8, "boxMargin": 8, "mirrorActors": true}}}%%
sequenceDiagram
    participant EngineCore
    participant Scheduler
    participant KVCacheManager

    Note over EngineCore,KVCacheManager: S5 建逻辑管理器：引擎进程内（平台无关）
    EngineCore->>EngineCore: resolve_kv_cache_block_sizes → (32768, 32768)
    EngineCore->>Scheduler: Scheduler(vllm_config, kv_cache_config,<br/>block_size=32768, hash_block_size=32768)
    Scheduler->>KVCacheManager: 5a 门面 KVCacheManager(kv_cache_config, max_model_len, ...)
    Note over KVCacheManager: 5b 协调器 get_kv_cache_coordinator → KVCacheCoordinatorNoPrefixCache<br/>（关前缀缓存，支持任意多组）
    KVCacheManager->>KVCacheManager: 5c 块池 BlockPool(num_blocks)：KVCacheBlock(0..n-1)<br/>null_block 摘走块 0<br/>5d 每 group 1 个 manager（attn→FullAttentionManager、linear→MambaManager）
    KVCacheManager-->>Scheduler: block_pool / coordinator 就绪
    Scheduler-->>EngineCore: 逻辑层装配完成（freeze_gc_heap 冻结启动堆）
```

### 1.3 总调用链（物理侧 / 逻辑侧）

以混合模型 Qwen3.5-4B pp2tp2（每 worker 16 层 = 12 linear + 4 full attn、full attn 2 KV 头，合并后全模型 **4 个 group**：1 组 attn 8 层 + 3 组 linear 各 8 层）为例。

#### 物理侧（引擎进程发起 · S1 算规格 → S2 测预算 → S3 做编排 → S4 落张量）

```text
EngineCore.__init__                                     # engine/core.py:119 起
│
├─ model_executor = executor_class(vllm_config)         # core.py:122 拉起 worker 进程（平台插件选 NPUWorker）
│
├─ kv_cache_config = _initialize_kv_caches(vllm_config)  # core.py:132，内部跑完物理侧 S1–S4
│    │
│    ├─ ※  register_all_kvcache_specs(vllm_config)       # FullAttentionSpec↔FullAttentionManager 等注册表
│    │
│    ├─ √ S1 §3.1.1 算规格  model_executor.get_kv_cache_specs()
│    │     ├─ RPC 到 NPUWorker.get_kv_cache_spec()（worker.py:891，一行转发）
│    │     │    → NPUModelRunner.get_kv_cache_spec()（model_runner_v1.py:4909，重写：分流表见 §3.1.1）
│    │     │      · 4 个 full-attn 层 → FullAttentionSpec（block_size=1024）
│    │     │      · 12 个 linear 层 → MambaSpec，并把 attn 层 page_size_padded 对齐到 mamba 页
│    │     └─ 返回 -> list[dict[str, KVCacheSpec]]，每个 worker 上每层 Attention KV cache 类型
│    │
│    ├─ ※  扫描 spec.non_causal                          # 非因果层，关闭 chunked prefill / 前缀缓存（平台无关）
│    │
│    ├─ √ S2 §3.2.1 测预算  model_executor.determine_available_memory()
│    │     ├─ RPC 到 NPUWorker.determine_available_memory()（worker.py:527）
│    │     │    → NPUModelRunner.profile_run()（model_runner_v1.py:3684：预热 + super()）
│    │     └─ 返回每个 worker 的 available_kv_cache_memory_bytes 字节数 -> list[int]
│    │        （NPUGraph 内存推迟到 4d 事后记账，见 worker.py:734-771）
│    │
│    ├─ √ S3 §3.3 做编排  get_kv_cache_configs(...)       # 平台无关，引擎进程
│    │     ├─ 合并各 worker 的 spec → 生成 global groups（1 组 attn 8 层 + 3 组 linear 各 8 层 = 4 组）
│    │     ├─ _project 到每 worker 的 projected groups（1 组 attn 4 层 + 3 组 linear 各 4 层）
│    │     ├─ 基于 projected groups 算 num_blocks → 跨组/跨 worker 对齐 min num_blocks
│    │     ├─ 返回 -> list[KVCacheConfig]，每个 worker 上的 KVCacheConfig（统一 num_blocks）
│    │     └─ 收口：generate_scheduler_kv_cache_config（deepcopy）→ 回写 num_gpu_blocks
│    │          → validate_block_size（core.py:279-287；core.py:317 返回 scheduler 版配置）
│    │
│    └─ √ S4 §3.4 物理侧落张量  model_executor.initialize_from_config(...)
│          ├─ collective_rpc("initialize_from_config") → NPUWorker.initialize_from_config()（worker.py:907）
│          │    ├─ ensure_kv_transfer_initialized()（worker.py:909，P/D 分离）
│          │    ├─ sleep mode：CaMemAllocator kv_cache 内存池（worker.py:910-912）
│          │    └─ NPUModelRunner.initialize_kv_cache()（model_runner_v1.py:3843）
│          │         ├─ 4a _allocate_kv_cache_tensors（:4082）
│          │         │      full-attn：整块单张连续张量；linear：单张量（命中 linear_attn 分支 :4118）
│          │         ├─ 4b _reshape_kv_cache_tensors（:4353）
│          │         │      full-attn：切头 padding → 对半切 K/V 双池 → view(nb×8,128,2,256)
│          │         ├─ 4c bind_kv_cache                复用上游（:3950-3953），block_id == 张量行号
│          │         └─ 有 KV 传输组时 register_kv_caches（:3883-3884）
│          └─ collective_rpc("compile_or_warm_up_model") → NPUWorker.compile_or_warm_up_model()（worker.py:706）
│                └─ 4d 逐 size _dummy_run（:728）+ NPUGraph capture（:732）
│                      + 图内存事后记账与 --kv-cache-memory 建议（:734-771）+ ATB 预热（:775-776）
│
└─ （物理侧结束：返回 scheduler 版 kv_cache_config，交给逻辑侧 S5）
```

#### 逻辑侧（S5，引擎进程内，物理侧 RPC 全部返回后）

```text
EngineCore.__init__（承上，:145 起）                    # engine/core.py:145
│
└─ √ S5 §4 逻辑侧逻辑管理器（平台无关）
       ├─ resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)  # core.py:145 → (32768, 32768)
       ├─ Scheduler(vllm_config, kv_cache_config, block_size=32768, hash_block_size=32768)  # core.py:149
       │     └─ 5a 门面 KVCacheManager(kv_cache_config, max_model_len, ...)   # sched/scheduler.py:231
       │           ├─ 5b 协调器 get_kv_cache_coordinator → KVCacheCoordinatorNoPrefixCache
       │           │     （关前缀缓存：工厂第一分支，支持任意多组）
       │           │     ├─ 5c 块池 BlockPool(num_blocks)：KVCacheBlock(0..n-1)，null_block 摘走块 0
       │           │     └─ 5d 每 group 1 个 manager（attn→FullAttentionManager、linear→MambaManager）
       │           └─ self.block_pool = coordinator.block_pool
       ├─ request_block_hasher（前缀缓存开启时才生成请求级哈希器；本模型关闭 → 不装配）
       └─ freeze_gc_heap() / enable_envs_cache()（core.py:228 / :233，启动装配冻结，运行期进入定态）
```

## 2. 配置初始化：物理侧 / 逻辑侧之前

### 2.1 S0 页对齐

本节对应启动链的 **S0**（config 期，物理侧／逻辑侧都还没开始），下分三个子步：**0a** 装配 `block_size`、**0b** 定 `mamba_block_size`、**0c** 统一 `page_size_bytes`，前附一张量总表。三类量**怎么定**：`block_size` 与 `mamba_block_size` 在 config 期定型；`page_size_bytes` 在 config 期只给出数值（`cache_config.mamba_page_size_padded`），到 S1 算规格才成为两族全等的 `spec.page_size_bytes`。§3 再讲它们在物理侧怎么用（S1 垫值 → S3 断言 → S4 消费）。

**量总表**：三类量 + 两个派生量（`kernel_block_sizes`、`scheduler_block_size` / `hash_block_size`）：

| block_size | 层 | 单位 | 含义 | 本模型（Qwen3.5-4B）取值 | 使用者 | 出处 |
|---|---|---|---|---|---|---|
| `cache_config.block_size` | **attn** | **token** | 一个块装多少 token——缓存/编号/前缀命中的粒度；1 个逻辑块 = 1024 token = 2,072 KB | **1024**（上游默认 **16** → 混合模型抬到 `kernel_block_size`（128）的整数倍 → 实测 1024） | BlockPool / KVCacheManager（`block_id` 编号、`num_blocks`）、调度器块对齐分片 | 0a |
| `cache_config.mamba_block_size` | **mamba** | **token** | 与 attn 组同义（每块装多少 token）；但 mamba 的页字节与它无关，故它实际表示**状态快照间隔**而非容量；1 块 = 整序列状态（= `max_model_len`），每请求 1 块 | **32768**（`none` / `all`）；`align` 降为 **1024**（见 0b） | 同上 | patch :143-146 |
| `spec.page_size_bytes` | **attn + mamba**（共用） | **字节** | `KVCacheSpec.page_size_bytes`（`@property`）：一个块占多少字节——物理账本（张量 dim0 行大小、显存预算的乘数）；**由 attn 组 `block_size` 决定**（= `block_size` × 每 token 页 + conv 页），**与 `mamba_block_size` 无关**；`AttentionSpec` 版在 `page_size_padded` 非空时直接返回 padded | **2,072 KB**（两族统一页，两族 spec 全等） | 全部 KV cache 张量：每层 `size = page_size_bytes × num_blocks` | kv_cache_interface.py:105、:168-180、patch :110-117、0c |
| `self.kernel_block_sizes` | **attn** | **token** | 逐组一项的 `list[list[int]]`；attn 组取 `select_common_block_size(kv_manager_block_size, backends)` 的结果，算子寻址粒度：大块内部再按 128-token 行切（大块展开 8×128）；`1024 % 128 == 0` 是选档底线（选档链路见 §3.4） | **128** | `reshape_and_cache`（写）、`npu_fused_infer_attention_score`（读）、slot mapping | model_runner_v1.py:4758-4762、patch :58 |
| `self.kernel_block_sizes` | **mamba** | **token** | 同列表的 mamba 组项，固定 `[0]`——源码注释原文“set kernel_block_sizes to 0 to disable slotmapping computation of mamba block”；无子粒度，状态整块读写 | **0** | GDN 状态算子（conv / temporal 就地步进） | model_runner_v1.py:4766-4769 |
| `scheduler_block_size` / `hash_block_size` | **全局**（跨组） | **token** | `resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)` 返回的二元组，再作为 `Scheduler(block_size=…, hash_block_size=…)` 传入；多组时调度取各组 block_size 的 **LCM**、哈希取 **GCD**（本模型两者相等），跨组 token 对齐不变量；关缓存时 hash 粒度退回同值（§4.2） | **32768** / **32768** | Scheduler → KVCacheManager → 协调器 / managers | kv_cache_utils.py:593-609、core.py:155-156 |

#### 0a block size：16 → 128 → 1024 的装配链

各量的取值、使用者与出处见 §2.1 量总表，此处只展开 config 期装配链（截至 patch 结束）：

```text
│（承接 §1.1 全栈图：run_server_worker → build_async_engine_client）

└─ engine_args = AsyncEngineArgs.from_cli_args(args)                                              vllm/entrypoints/api_server.py:117
   └─ vllm_config = engine_args.create_engine_config(usage_context)                               vllm/v1/engine/async_llm.py:242
      ├─ current_platform.pre_register_and_update()                                               vllm/engine/arg_utils.py:1712
      │  └─ adapt_patch(is_global_patch=True)                                                     vllm_ascend/platform.py:188
      ├─ model_config = self.create_model_config()                                                vllm/engine/arg_utils.py:1735
      ├─ cache_config = CacheConfig(block_size=self.block_size, ...)                              :1759-1760
      │  │  （self.block_size ← AsyncEngineArgs.block_size：CLI --block-size 直通，省略时为 None）   arg_utils.py:504
      │  └─ _apply_block_size_default（model_validator，分叉点；_skip_none_validation 先放行 None）  vllm/config/cache.py:224-244
      │     ├─ 传参分支（--block-size 2048）：非 None → 保留用户值、user_specified=True             :240-241
      │     └─ 无传参分支（本部署）：None → 填 DEFAULT_BLOCK_SIZE = 16（此处即"默认 16"的来源）      :238-239、:46
      └─ config = VllmConfig(...)                                                                 vllm/engine/arg_utils.py:2229
         │
         └─ VllmConfig.__post_init__（pydantic 自动触发）                                          vllm/config/vllm.py:847
            ├─ self.try_verify_and_update_config()                                                :856
            │  └─ is_hybrid=True（混合模型）
            │     └─ HybridAttentionMambaModelConfig.verify_and_update_config(self)               :1903
            │        └─ verify_and_update_config（patch 版）                                      vllm-ascend/patch/platform/patch_mamba_config.py:31
            │           ├─ kernel_block_size = 128                                                :58
            │           ├─ cache_config.block_size = attn_block_size (=1024)                      :103
            │           ├─ cache_config.mamba_page_size_padded = attn_page_size + conv_page_size  :117
            │           ├─ 分支①：开缓存 且 mamba_cache_mode=align → mamba_block_size = cache_config.block_size   :143-144
            │           └─ 分支②：其余全部形态（关缓存、开缓存且 all）→ mamba_block_size = model_config.max_model_len :145-146
            └─ current_platform.check_and_update_config(self)                                     vllm/config/vllm.py:1335
               └─ refresh_block_size(vllm_config)                                                 vllm_ascend/platform.py:671
                  └─ 混合：is_hybrid=True → return，1024 保留（平台层不覆盖）                       vllm-ascend/utils.py:1397-1400
```

#### 0b mamba_block_size：mamba_cache_mode 三模式

`cache_config.mamba_block_size` 由 `mamba_cache_mode` 决定（patch :143-146）：`align` → 取 attn 组同值 **1024**；`none` / `all` → 取 `max_model_len` **32768**。三种模式的差别不止块大小，还体现在缓存内容与每请求持有块数上：

| 模式 | 何时被选中 | 缓存什么 | `mamba_block_size` | 每请求持有的 mamba 物理块 | 前缀命中 |
|---|---|---|---|---|---|
| `none` | 前缀缓存关闭（默认） | 不缓存，只保留当前运行态 | `max_model_len` = **32768**（一个块 = 整序列） | **`1 + num_spec`**，恒定到请求结束（`ref_cnt>0`） | 无 |
| `all` | 开缓存 **且** 模型声明 `supports_mamba_prefix_caching`——⚠️ **名义可选，NPU 上实际退化** | **每个** `i * block_size` 位置的状态 | `max_model_len` = **32768** | **`cdiv(num_tokens, bs) + num_spec`**（块表宽度按 `max_model_len` 预留，mamba_attn.py:111-120） | 命中集合 = 全部块边界；**强保证**（请求持有期间不可被驱逐） |
| `align` | 开缓存但模型不支持 `all`（降级） | **只**缓存“每步最后一个 token、且落在 `i * block_size`”的状态 | `cache_config.block_size` = **1024**（与 attn 组一致） | **命中块数 + `2 + num_spec`（峰值）**；状态块逐块前移，旧块滞后一步交还块池（逐轮 prefill / decode 流程见运行期篇 §4.3） | 命中集合 = 全部块边界；**best-effort**（`ref_cnt=0`，随时可被驱逐） |

#### 0c page_size_bytes：两族统一为 2,072 KB

Qwen3.5-4B 的 4 个 group 共用一个 `BlockPool`，前提是**统一页大小 `page_size_bytes`**，让 mamba 组把状态页塞进 attn 定义的页中。attn 页 = `block_size × 每 token 字节`，随 attn 的 block size 线性增长；mamba 的状态页固定 1,048 KB，与任何 block size 都无关——所以页大小只能**由 attn 的 block size 决定**。vllm-ascend 的 patch 用 `assert 每 token K 页 × attn_block_size == ssm 状态页`，要求 attn 的 K 页正好盛下 mamba 的 ssm 页，由此反推出 Qwen3.5-4B 的 attn block size = 1024、统一页 = 2,072 KB。数字链如下（TP2 / KV bf16 / ssm fp32）：

| 量 | 计算 | 值 |
|---|---|---|
| 每 token K 页 | head_dim(256) × num_kv_heads(TP2 后 4÷2=2) × bf16(2B) | 1 KB |
| temporal 状态页（ssm） | value 头(32÷TP2=16) × 128 × 128 × fp32(4B) | 1 MB（1,024 KB） |
| conv 状态页 | 卷积通道(2×k头16×128 + v头32×128) ÷ TP2 = 4,096 × (conv_kernel−1 = 3) × bf16(2B) | 24 KB（24,576 B） |
| **attn 块大小** | 128 × ⌈1 MB ÷ (128 × 1 KB)⌉ = 128 × 8 | **1024 token** |
| attn 原生页 | 2(K/V) × 1024 × 2 头 × 256 × 2B | 2 MB（2,048 KB） |
| mamba 原生页 | conv(24 KB) + ssm(1 MB) | 1,048 KB（= 1 MB + 24 KB） |
| **统一页（两族 `page_size_bytes` 全等）** | **attn 页 + conv 页**（patch :113-117） | **2,072 KB（= 2 MB + 24 KB）** |
| padding 比例 | conv 24 KB ÷ 2,072 KB | **1.16%**（= 日志原值） |

**成型时点与落地位置**：2,072 KB 只在 **config 期计算一次**，物理侧初始化 S1 期只把该值**附着**到两个 spec 上，S3 只断言、S4 只消费：

| 时点 | 发生什么 | 出处 |
|---|---|---|
| **config 期** | patch 算出 `cache_config.mamba_page_size_padded = attn 页 + conv 页 = 2,072 KB`，并用 `assert` 保证 attn 块的 K 半区正好罩住 ssm 页；此时**还没有任何 spec 实例** | patch :95、:110-117 |
| **S1 算规格 · mamba 侧** | `MambaSpec(..., page_size_padded=cache_config.mamba_page_size_padded)` 构造即带值；`page_size_bytes` 断言 padded ≥ 原生 1,048 KB 后返回 2,072 KB | abstract.py:45-51、kv_cache_interface.py:616-625 |
| **S1 算规格 · attn 侧** | `FullAttentionSpec` 不带 padded，原生页 2,048 KB；NPU 侧 `object.__setattr__(spec, "page_size_padded", 2,072 KB)` **事后垫上** → 两族 `spec.page_size_bytes` 首次全等 | model_runner_v1.py:5030-5036 |
| **S3 编排** | `get_uniform_page_size` 断言 `len(page_sizes) == 1`，按 2,072 KB 给 4 组算 `num_blocks`（详见 §3.3） | kv_cache_utils.py:977、:1688 |
| **S4 落张量** | `page_stride = page_size_padded // dtype_size` 定块步长，切头部 conv padding 后对半切 K/V 双池（详见 §3.4） | model_runner_v1.py:4522-4531、:4576-4579 |

**统一页的浪费结构（attn 空洞 / mamba 空洞）**：统一页 2,072 KB 按物理排布分三段——**conv 位 24 KB → K 位 1,024 KB → V 位 1,024 KB**，patch 断言（patch_mamba_config.py:95）保证 `K 位 == ssm 状态页`（1,024 KB），即 mamba 的 ssm 恰好**顶替 K 位**。attn 层每块浪费 conv 位的 24 KB；mamba 层每块浪费 V 位的 1,024 KB。两族的"使用 / 空洞"矩阵：

| 层类型（每块） | conv 位 24 KB | K 位 1,024 KB | V 位 1,024 KB |
|---|---|---|---|
| attn full-attn 层（8 层） | ✗ **空洞（浪费 24 KB/块，= 1.16%）** | ✓ 使用（K 池） | ✓ 使用（V 池） |
| mamba GDN 层（24 层） | ✓ 使用（conv 状态） | ✓ 使用（ssm 状态顶替此位） | ✗ **空洞（浪费 1,024 KB/块，≈ 49.4%）** |

示意图（一块 2,072 KB 的三段划分与两族占用；绿色 = 有效数据，灰色虚线 = 空洞）：

```mermaid
block-beta
  columns 3
  ac["attn · conv 位 24 KB<br/>空洞（1.16%）"]
  ak["attn · K 位 1,024 KB<br/>K 池"]
  av["attn · V 位 1,024 KB<br/>V 池"]
  mc["mamba · conv 位 24 KB<br/>conv 状态"]
  mk["mamba · K 位 1,024 KB<br/>ssm 状态顶替"]
  mv["mamba · V 位 1,024 KB<br/>空洞（49.4%）"]
  classDef used fill:#e8f5e9,stroke:#4caf50,color:#1b5e20
  classDef hole fill:#eeeeee,stroke:#bdbdbd,stroke-dasharray:4 3,color:#757575
  class ak,av,mc,mk used
  class ac,mv hole
```

**Qwen3.5 4B 实测**：

```text
[1] vllm CacheConfig() default block_size = 16
[2] refresh_block_size(hybrid, block=16) untouched -> 16           # is_hybrid 跳过，交给模型层
[3] attn_block_size = 1024 | attn_page = 2 MB | conv_page = 24 KB | padding = 1.16%
[4] mamba_block_size = 32768 (= max_model_len，前缀缓存关闭)

# verify_page.py（用 vllm 真实 spec 类的 page_size_bytes property 验证统一页）：
[P1] MambaSpec 原生页(conv+ssm)            = 1,048 KB（= 1 MB + 24 KB）
[P2] MambaSpec.page_size_bytes (padded)     = 2,072 KB（= 2 MB + 24 KB）
[P3] FullAttentionSpec real 页（1024 块）   = 2,048 KB = 2 MB
[P4] FullAttentionSpec.page_size_bytes（NPU S1 垫页后）= 2,072 KB（= 2 MB + 24 KB）
[P5] 两族全等: True | 统一页 = 2,072 KB（= 2 MB + 24 KB） | padding = 1.16%
```

**最后几个设计问题**

**为什么 mamba ssm 状态不能摊用 K+V 两段省掉空洞（设计取舍）**：

- ① ssm 状态视图 `(16,128,128)` 要求**单段连续嵌入**（GDN 算子 state buffer 布局）；
- ② V 段语义被双池算子 API 固化（`reshape_and_cache` 双池入参、`slot→偏移` 线性）；
- ③ 两族同构三段式 `[24KB | 1,024KB | 1,024KB]` 使块级操作（swap/注册/清零/块表）跨层一套逻辑。

**为什么是 K/V 双池（NPU）**：NPU 双池的直接动机是 **PD 分离 KV 传输注册**——`model_runner_v1.py:4087-4088` 注释原文："To support prefill disaggregation, we need to split kvcache tensor into k cache and v cache, and the addr of both are aligned by 2M"：
传输层注册 buffer 要求 **dim0 = num_blocks + 地址 2MiB 对齐**；
Ascend 注意力算子链（`reshape_and_cache` → `npu_fused_infer_attention_score`）的生态本来就是 K、V 两张独立 4D 入参。单机无 KV 传输时不强制 2MiB，但双池布局形态保留（含 `swap_blocks`/`copy_blocks` 双池搬行，attention_v1.py:114-140）。

**hybrid 模型 cache 的物理布局**（model_runner_v1.py:4495-4579）：
混合模式下 full-attn 层**不走纯 FullAttn 的"K/V 双独立池"两次独立申请**——而是**单张连续张量 `2,072 KB × num_blocks`**（:4503），reshape 时先切掉头部 conv padding（`raw[24KB×nb:]`，:4576-4577）再对半切成 K 池（`:attn页`）与 V 池（`[attn页:]`，:4578-4579），各自 `view` 成 `(num_blocks×8, 128, 2, 256)` bf16（:4561-4570）——管理块 1024-token、物理 block_table **展开 8 份 128-token 细块**；mamba 层同张量格式按状态切片成视图。头部集中的 padding 字节与 mamba 尾部 V 位空洞**物理分配但永不访问**。

## 3. 物理侧：物理张量初始化

物理侧的四步横跨两侧进程：S1/S2/S4 的实现主体经 collective RPC 落在各 worker 进程（vllm-ascend 的 `NPUWorker` / `NPUModelRunner`），S3 做纯 Python 编排跑在引擎进程 `_initialize_kv_caches`（core.py:236-317）内。前三步全程不申请任何 KV 显存，产出**唯一的容量仲裁结果** `KVCacheConfig`：一份完整版 list（供 S4 逐卡落张量）+ 一份 scheduler 版（供逻辑侧 S5 装配）；S4 才把张量落到 NPU 显存。

### 3.1 S1 · 算规格：各层产出 KVCacheSpec

这一步遍历本 worker 模型里的每个 Attention 层，问出"你需要什么形状的 KV cache"。对本模型，最终产出的是**两类 spec**：full-attn 层的 `FullAttentionSpec` + linear 层的 `MambaSpec`。

**可见范围：PP 切层、TP 切头**——S1 只遍历本 worker 负责的层，故每卡只产出 16 层 spec；切分规则如下：

- **PP 按层切分**：`config/model.py:1307-1320` `get_layers_start_end_indices()` 按 `pp_rank` 切层范围，`get_kv_cache_spec()` 只返回本 worker 负责的层（NPU 的 `NPUModelRunner.get_kv_cache_spec` 同样只遍历本 worker 层，model_runner_v1.py:4922）。
- **TP 按"头"切分**（两类层都切，但**切维不同**）：同一 PP stage 的不同 TP rank 存**同层、不同头子集**，且都**不复制**状态——
  - **full-attn 组**：按 **KV 头**切——`config/model.py:1284-1295` `get_num_kv_heads()` 除以 `tensor_parallel_size`（本模型 4÷2 = **2 头/卡**）；
  - **linear 组（GDN）**：两族状态的切维不同：
    - **temporal / SSM 状态**：按 **value 头**切 → `(num_v_heads/tp, head_v_dim, head_k_dim)`，**头内两维完整不切**（本模型 32÷2 = **16 头** × 128 × 128，即"页统一"一节里的 ssm 页）；
    - **conv 状态**：按 **卷积通道维**切 → `divide(2·k头·k头维 + v头·v头维, tp)`（短卷积作用在 q/k/v 上），kernel 长度维（`conv_kernel−1 + num_spec`）**不切**。
  - 因头内维度完整，**每个 rank 只更新自己那部分头的状态 → 无跨 TP 通信**（与 full-attn "按 KV 头切、无通信"同性质）。

**主线部署 · PP2 × TP2 = 4 卡（Ascend NPU）**（Qwen3.5-4B：32 层 = 24 linear + 8 full attn，`full_attention_interval=4`）

PP2 按层切成两段，每段 16 层（**12 linear + 4 full attn**）：

| worker | PP rank | TP rank | 负责层 | 组内层数 | full attn KV 头数（4 ÷ TP2） |
|--------|---------|---------|--------|----------|-------------------------------|
| W0 | 0 | 0 | L0–L15 | 12 linear + 4 full attn | 2 |
| W1 | 0 | 1 | L0–L15 | 12 linear + 4 full attn | 2 |
| W2 | 1 | 0 | L16–L31 | 12 linear + 4 full attn | 2 |
| W3 | 1 | 1 | L16–L31 | 12 linear + 4 full attn | 2 |

**分流规则：一轮遍历即产出两类 spec**，页对齐也在本步完成：

| 本卡的层 | 分流分支（model_runner_v1.py） | 产出 spec |
|---|---|---|
| **4 × full-attn** | :4943-4946 取到 attn 模块 spec 后**原样存入** | **`FullAttentionSpec`**：`block_size = 1024`、`num_kv_heads = 2`（4 头 ÷ TP2）、`head_size = 256`、bf16 |
| **12 × linear（GDN）** | :5002-5003 先收集 → :5027-5036 统一构造 | **`MambaSpec`**：`block_size = mamba_block_size = 32768`（关缓存分支）、`page_size_padded = 2,072 KB`；同一步里把 4 个 attn 层也垫到该页（:5034-5036） |

**调用链**：

```text
EngineCore._initialize_kv_caches(vllm_config)                       vllm/v1/engine/core.py:236
│
└─ S1 算规格  model_executor.get_kv_cache_specs()                      :243
   │        （[P1 引擎] RPC 下发 collective_rpc("get_kv_cache_spec")：引擎进程 → 逐 worker）
   │
   └─ ══ worker 进程 [P2..P5] ══  NPUWorker.get_kv_cache_spec()       vllm_ascend/worker/worker.py:891
      │       一行转发 → return self.model_runner.get_kv_cache_spec()
      │
      └─ NPUModelRunner.get_kv_cache_spec()                          vllm_ascend/worker/model_runner_v1.py:4909（重写上游同名方法）
         ├─ 遍历本卡 16 层（AttentionLayerBase，kv_sharing 跳过）                        :4922
         ├─ 4 层 full-attn → FullAttentionSpec（block_size=1,024、2 头、256 维、bf16）    :4943-4946（循环内当场写入）
         ├─ 12 层 linear（GDN）→ 先暂存 mamba_layers，循环结束再统一构造 MambaSpec      :5002-5031（尾块写入）
         ├─ 4 个 attn 层 page_size_padded 垫到 mamba 页（object.__setattr__）             :5033-5036
         └─ 返回 dict[layer_name, KVCacheSpec]（本卡 16 条 = 4 FullAttentionSpec 在前 + 12 MambaSpec 在后）
```

**`layer_name` 是什么**

它是 **attention 模块在模型里的注册名**（字符串），不是层号、也不是权重名。格式固定为 `<模型前缀>.layers.<i>.<模块属性名>`：

- **本模型实况**：full attn 为 `model.language_model.layers.3.self_attn`，GDN 为 `model.language_model.layers.0.linear_attn`。多一层 `language_model` 是因为服务用 `Qwen3_5ForConditionalGeneration`（VLM 包装，qwen3_5.py:586-587）；若换成纯文本的 `Qwen3_5ForCausalLM`，名字就是 `model.layers.<i>.…`。
- **谁登记**：模型构建时每个 attention 模块以模块属性名注册进 `static_forward_context`（`self.self_attn = Qwen3NextAttention(...)` / `self.linear_attn = QwenGatedDeltaNetAttention(...)`，qwen3_5.py:138-152）；`get_kv_cache_spec()` 遍历这些条目，**用同一个 key** 写进 spec dict，所以 dict 的键天然就是层名。
- **四个用途**：3a 合并时作 dict 键（按名字拼接 + 同层断言）、分组产物 `KVCacheGroupSpec.layer_names`、S4 落张量时 `kv_caches[layer_name] = (k_cache, v_cache)`（:797 起）、NPU 用 `"linear_attn" in layer_name` 判单张量路径（model_runner_v1.py:4118-4134）。

**契约 · 返回顺序是"attn 先、mamba 后"**（注意：**不是层号顺序**）——Qwen3.5-4B 的层序从 layer 0 起就是 GDN（mamba 在前），但 `get_kv_cache_spec()` 的返回顺序由契约规定：

> 源码注释（model_runner_v1.py:4923-4924）："Must process Attention/MLAAttention before MambaBase to maintain ordering expected by graph parameter update logic in attention backends."

落到本模型：

- **顺序怎么形成**：attn 层**循环内当场写入** `kv_cache_spec`（:4943-4946）；mamba 层只先暂存 `mamba_layers`，等循环结束由尾块统一构造（:5027-5031）→ 本卡 16 条 = **4 attn 在前 + 12 mamba 在后**，layer 0 虽是 GDN 仍排在 4 个 attn 之后（见上树）。
- **为什么不能乱**：ACL Graph 回放**按位置配对** —— `zip(forward_context.attn_metadata, graph_params.attn_params[num_tokens], ...)`（attention_v1.py:474-491），而捕获时记录的参数顺序就是 spec 注册顺序；顺序一变，`key_cache / block_table / seq_lens` 全部错位。
- **顺序即 group id**：合并各 worker 时按层名**首次出现顺序**拼接（kv_cache_utils.py:1994-2011），分组函数按 spec **首次出现顺序**建组（`defaultdict(list)`，kv_cache_utils.py:1142-1144）→ 本模型固定为：

| group id | 内容 | `block_size` | `kernel_block_size` |
|---|---|---|---|
| **0** | attn 组（8 层） | 1024 | `[128]` |
| **1 ~ 3** | mamba 三组（各 8 层） | 32768 | `[0]` |

（即 `block_sizes = [1024, 32768, 32768, 32768]`、`kernel_block_sizes = [[128], [0], [0], [0]]`，model_runner_v1.py:4731-4769）

- **group id 被谁消费**：每 group 一个 manager（kv_cache_coordinator.py:101-120）、`attn_groups[group_id]`（model_runner_v1.py:4756）、前缀缓存块哈希编入 group id（kv_cache_utils.py:56-65）、物理张量按"组内第 i 层"跨组共享（kv_cache_utils.py:1319-1326）；唯一不依赖编号的是上游 Hybrid 协调器——它自己按类型把 attn 组提到最前（kv_cache_coordinator.py:547）。

### 3.2 S2 · 测预算：profile 量出可用显存

这一步在每个 worker 上跑一次 dummy forward，量出权重+激活之外还剩多少字节给 KV cache。

**调用链**：

```text
EngineCore._initialize_kv_caches(vllm_config)                       vllm/v1/engine/core.py:236
│
└─ S2 测预算  model_executor.determine_available_memory()              :257
   │        （[P1 引擎] RPC 下发 collective_rpc("determine_available_memory") → 逐 worker）
   │
   └─ ══ worker 进程 [P2..P5] ══  NPUWorker.determine_available_memory()   vllm-ascend/worker/worker.py:527
      │        （覆写上游 Worker 同名方法，骨架见 vllm/v1/worker/gpu_worker.py:372）
      ├─ 快路径：--kv-cache-memory 已指定 → 只 profile_run()（触发编译）、跳过记账 → 直接返回该值   :540-553
      └─ 常规路径：
         ├─ with memory_profiling(init_snapshot, weights_memory=权重占用)：                     :557-560
         │     └─ NPUModelRunner.profile_run()                                                  vllm-ascend/worker/model_runner_v1.py:3684
         │        ├─ eplb_warmup() + MC2 容量判断（本模型不命中，走 super() 主路径）              :3685-3690
         │        ├─ PCP 时按 pcp_size 缩放 max_num_tokens（本模型 pcp=1 → 跳过）                :3694-3695
         │        └─ super().profile_run()  ← 上游 max-token dummy forward                      vllm/v1/worker/gpu_model_runner.py:6168
         │              （NPU 版不在此处估图内存：上游的 profile_cudagraph_memory() 未调用）
         ├─ context 内、图捕获前取 torch 峰值  torch.npu.memory_stats(...)["…all.peak"]           :567
         └─ 出 context 后记账                                                                    :569-574
            ├─ 覆写 torch_peak_increase = 预捕获峰值 − 记账前峰值（避免把图池算成激活）
            ├─ non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights_memory
            ├─ available = requested_memory − non_kv_cache_memory                               :590
            ├─ 存 peak_activation_memory / non_torch_memory（供 4d 事后记账复用）                :577-578
            ├─ 断言 init.free > profile 后 free（防同容器他进程腾显存）                           :581
            └─ 日志 info_once("Available KV cache memory: %.2f GiB", scope="local")             :593-595
```

`NPUWorker.determine_available_memory()`（vllm_ascend/worker/worker.py:527）整体结构沿用上游框架，多处直接复用上游工具：

1. **快路径 `--kv-cache-memory`**（worker.py:540-553）：用户显式指定 KV 字节数时，仍跑一次 `profile_run()`（触发编译），但跳过整个记账计算，直接返回用户指定值；
2. **同一套快照/上下文工具**：`MemorySnapshot` 与 `memory_profiling` 直接从上游 `vllm.utils.mem_utils` 导入（worker.py:47）——快照内部走 `current_platform` 平台抽象，NPU 下即 torch.npu 的设备统计；
3. **同一次 dummy forward**：`NPUModelRunner.profile_run()`（model_runner_v1.py:3684）只在前后加 NPU 专属预热，核心调用 **`super().profile_run()`**（:3696）即上游那套 max-token dummy forward（含 EPLB / MC2 / PCP 分支，本模型走 `super()` 主路径）；
4. **torch 峰值改走 npu API**：`torch.npu.memory_stats(self.device)["allocated_bytes.all.peak"]`（worker.py:567），取图捕获**之前**的预捕获值并覆写 `torch_peak_increase`（:571）。

**核心公式——没有图内存项**（worker.py:572-574 / :590）：

```text
available_kv_cache_memory = requested_memory                （total × gpu_memory_utilization）
                           − (non_torch_increase
                              + torch_peak_increase          （预捕获峰值）
                              + weights_memory)
```

**显存分配示意**（每卡一条账，宽度仅示意、非严格按比例）：

```mermaid
block-beta
  columns 8
  T["设备 total_memory"]:8
  R["requested = total × util ｜ vLLM 的记账范围"]:7
  G["total − requested<br/>未计入余量<br/>（4d 图内存出处）"]:1
  W["权重<br/>weights_memory"]:2
  P["峰值激活<br/>torch_peak_increase"]:2
  NT["非 torch<br/>HCCL / 驱动 / ACL"]:1
  AV["available_kv_cache_memory<br/>→ S3 算 num_blocks"]:2
  space:1
  S["4d 实测 npugraph_memory_bytes 后重算并给建议：<br/>--kv-cache-memory = requested − non_kv − 150 MiB"]:8
  classDef budget fill:#e3f2fd,stroke:#1e88e5,color:#0d47a1
  classDef used fill:#e8f5e9,stroke:#4caf50,color:#1b5e20
  classDef kv fill:#fff3e0,stroke:#ff9800,color:#e65100
  classDef hole fill:#eeeeee,stroke:#bdbdbd,stroke-dasharray:4 3,color:#757575
  class T,R budget
  class W,P,NT used
  class AV kv
  class G hole
```

**那"没算图内存"为什么通常也不会 OOM？**——先把账分成两类：**S2 只记"此刻已确定"的开销（权重 / 峰值激活 / 非 torch）；图内存属于"还没发生"的那类**——它要到 4d `capture_model()` 才产生，而 S2 跑在 S4 分配 KV **之前**，此刻物理上无从测量。既然记不了，就靠下面三道兜底，按发生顺序排列：

1. **分配前 · util 自带的隐式余量**：`requested = total × util`，只要 util < 1，`total − requested`（util=0.92 时即约 8% 的 total）就**从未被任何一项计入**；图池落地时优先从这块缓冲里出——所以"没记"不等于"没地方放"；
2. **捕获后 · 4d 用实测值校正并给建议**：`capture_model()` 返回真实 `npugraph_memory_bytes`（:732），立即把它并进 non_kv 重算，并打印两个建议值 `--kv-cache-memory = requested − non_kv − 150 MiB`（塞进 requested）/ `= init.free − non_kv − 150 MiB`（吃满空闲）（:743-751）；那 **150 MiB 冗余**正是给 `memory_profiling` 可能低估的 non-torch 分配留的（ACL context / HCCL buffer / 驱动层，:737-742）；
3. **万一仍不够 · 失败点在前**：S4 的顺序是**先落 KV 张量（4a-4c）再 capture（4d）**，所以真不够就是**启动期在 capture 处 OOM**——报错可见、可定位、可重试，不会拖到运行期才崩。

> 因此准确的说法是：**它不是"不算也安全"，而是"util 余量 + 事后校正"这套概率性方案**——余量托底，日志给出建议值，通常两次启动即可收敛到安全值。要一次性拿到确定性，就走 `--kv-cache-memory` 快路径（:540-553）：用户显式把图内存算进预算，整套记账直接跳过。

**结果日志**：`logger.info_once("Available KV cache memory: %.2f GiB", ..., scope="local")`（worker.py:593-595）——启动日志里的 `[worker.py:593] Available KV cache memory: 51.90 GiB` 即此行。

### 3.3 S3 · 做编排（平台无关）：合并 → 分组 → 自适应 + 校验 → 生成配置 → 对齐缩容

`get_kv_cache_configs()` 的 docstring（:1968-1979）把编排归纳为 5 步，本节 **3a–3e 与之一一对应**：

| docstring 步骤 | 本节 | 做什么 |
|---|---|---|
| 1. Merge the KV cache specs of all workers | **3a 合并** | 合并各 worker 的 KV cache spec → 全模型 spec |
| 2. Generate the KV cache groups based on the layer ratio | **3b 分组** | 按层比例生成 KV cache group，并处理混合模型的 spec 统一 |
| 3. Handle auto-fit max_model_len and memory checks using per-worker projected groups | **3c 自适应 + 校验** | 用 per-worker projected group 适配 `max_model_len` + 校验显存 |
| 4. Generate the KV cache configs for each worker | **3d 生成配置** | 逐 worker 生成 `KVCacheConfig`（含 `num_blocks`） |
| 5. Change the num_blocks of each worker to the smallest | **3e 对齐** | 取跨 worker `min(num_blocks)` + 张量等比缩 |

**调用链**：

```text
EngineCore._initialize_kv_caches(vllm_config)                              vllm/v1/engine/core.py:236
│
└─ S3 做编排  get_kv_cache_configs(vllm_config, kv_cache_specs, available)   vllm/v1/core/kv_cache_utils.py:1956
   │        （全部跑在引擎进程 [P1]，无 RPC——消费 S1/S2 上报的 spec / 预算）
   │
   ├─ 3a 合并：合并各 worker 的 KV cache spec（同层跨 TP 断言等值）            :1991-2003
   ├─ 3b 分组：按层比例生成 KV cache group（含混合模型 spec 统一）             :2011
   │     ├─ is_kv_cache_spec_uniform()：merge 试探 → 混合模型判 False          :874
   │     └─ unify_kv_cache_spec_page_size() → _get_kv_cache_groups_uniform_page_size()   :1074
   ├─ 3c 自适应 + 校验：用 per-worker projected group 适配 max_model_len 并校验显存 :2013-2057
   │     ├─ 投影  _project_kv_cache_groups_to_worker()（PP 分片）              :2016-2019
   │     ├─ num_gpu_blocks_override：available ← override × 每块字节           :2027-2041
   │     ├─ 自适应 max_model_len（仅 original_max_model_len == -1）             :2043-2046
   │     └─ 逐 worker 预算校验  _check_enough_kv_cache_memory()                :2049-2057 → :713
   ├─ 3d 生成配置：逐 worker 生成 KV cache config（含 num_blocks）             :2059-2070 → :1247
   │     └─ num_blocks = available // page_size // group_size                 :1309-1326
   └─ 3e 对齐：num_blocks 取所有 worker 最小值 + 张量等比缩                      :2072-2090
```

#### 3a 合并：合并各 worker 的 KV cache spec（`merged_kv_cache_specs`）

源码（kv_cache_utils.py:1991-2003）：

```python
merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
for kv_cache_spec_one_worker in kv_cache_specs:                # 每 worker 一份 dict
    for layer_name, layer_spec in kv_cache_spec_one_worker.items():
        if layer_name not in merged_kv_cache_specs:
            merged_kv_cache_specs[layer_name] = layer_spec     # 首次出现 → 登记
        else:
            assert merged_kv_cache_specs[layer_name] == layer_spec, (
                "The KV cache specs for the same layer are different "
                "across workers. This is not supported yet."
            )
```

- **键是"层名"而不是层号** → 合并天然只按名字拼接：PP 两段层名不相交，谁也不会覆盖谁；**同一 PP stage 的两个 TP rank 提交同名同 spec**，会走进 `else` 分支做等值断言（`KVCacheSpec` 是 frozen dataclass，`==` 是逐字段比较）。
- **值仍是 spec 对象** → 这里只做"登记 + 校验"，**不做任何归一化**；页统一、类型折叠都留到 3b。
- **插入顺序被保留**（Python dict 有序）→ 它决定后面分组的组号，见 §3.1「返回顺序是契约的一部分」；注意每个 worker 的返回都是"attn 先、mamba 后"。

**本模型实况（Qwen3.5-4B / pp2tp2 → 4 个 worker，每卡 16 层）**：

| 来源 | 提交内容（本卡 16 条） | 合并动作 |
|---|---|---|
| W0（PP0/TP0） | 4 × `…layers.{3,7,11,15}.self_attn` + 12 × `…layers.{i}.linear_attn`（i ∈ L0,1,2,4,5,6,8,9,10,12,13,14） | 全部首次出现 → 登记 16 条（键序：**4 attn → 12 mamba**） |
| W1（PP0/TP1） | 与 W0 同名、同 spec 16 条 | 16 次 `==` 断言全部通过，**零新增**（TP2 已把 4 头切成每卡 2 头，折进 spec，故两边自然相等） |
| W2（PP1/TP0） | 4 × `…layers.{19,23,27,31}.self_attn` + 12 × `…layers.{i}.linear_attn`（i ∈ L16,17,18,20,21,22,24,25,26,28,29,30） | 全部新层名 → 追加 16 条（键序：**4 attn → 12 mamba**） |
| W3（PP1/TP1） | 与 W2 同名、同 spec 16 条 | 16 次断言通过，**零新增** |
| **合并结果** | **32 条** | 8 条 `FullAttentionSpec` + 24 条 `MambaSpec` |

> 键序实际是 **`4 attn → 12 mamba → 4 attn → 12 mamba`**（按 worker 顺序拼接），并不是"8 attn 在前、24 mamba 在后"；但**组号只看 spec 类型的首次出现**，所以后面仍是 group 0 = attn、group 1~3 = mamba。
>
> **NPU 的 Mamba 层在 S1 阶段已是 `MambaSpec`**——"平台特有 spec"从这一步就进入平台无关的编排管线：3a 只按层名与 `==` 工作，对 spec 的具体类型一无所知。

#### 3b 分组：按层比例生成 KV cache group（`get_kv_cache_groups()`）

`get_kv_cache_groups(vllm_config, merged_kv_cache_specs)`（:2011 → 定义 :1629-1695）用一条**分支级联**把不同模型形态分流，只有落到最后的**兜底分支**才做"按层比例切组"：

| 分支 | 用途（各分支对应哪些模型） | 本模型 |
|---|---|---|
| A `disable_hybrid_kv_cache_manager` | 用户显式关掉 hybrid 管理器时，`unify_hybrid_kv_cache_specs` 把多类 spec 拍平成单一 spec（:1642-1643） | 未开 → 跳过 |
| B attention-free | encoder-only 模型（spec 为空）→ 返回空组（:1645-1647） | 非空 → 跳过 |
| C `is_kv_cache_spec_uniform` | 全层 spec 完全一致（纯 full-attn、或同窗口的纯滑窗）→ 直接 1 组（:874-896） | False（两类 spec 无共同类型，merge 断言失败） |
| D `UniformTypeKVCacheSpecs.from_specs` | 同一类注意力、仅窗口/形状不同（`block_size` 相同的滑窗混合）→ 按类型归一后分组（kv_cache_interface.py:742-767） | None（`block_size` 有 1024 / 32768 两值） |
| E `group_and_unify_kv_cache_specs` | DeepseekV4 的 MLA + 多种滑窗组合专用（:1431-1441） | None（无 MLA） |
| **兜底** | 以上都不命中 → 摘 hidden state、统一页、按层数比例切组 | **走这条** |

**兜底分支的源码（本模型真正执行的三行，:1669-1695）**

```python
# 1) 摘掉 HiddenStateCacheSpec：Mamba 的中间 hidden state，不参与页统一与分组
filtered_spec = {
    k: v for k, v in kv_cache_spec.items()
    if not isinstance(v, HiddenStateCacheSpec)          # 本模型没有 → 原样
}
# 2) 页统一：KVCacheManager 只能分配一种大小的页
filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
# 3) 按层数比例切组
groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)
return groups
```

> 第 2 步在此是**零成本**：`unify_kv_cache_spec_page_size` 只在 `page_sizes` 去重后 >1 时才通过"抬高小页的 `block_size`"归一，并要求大页能被小页整除（:1044-1065）；本模型两支页已被 S0 拉平为 2,072 KB，`len(page_sizes) <= 1` → 原样返回入参。**配置初始化期的页对齐收益在此兑现：4 组天然共享同一 block 账本。**

**切组源码（`_get_kv_cache_groups_uniform_page_size`，:1139-1193）**

```python
same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
for layer_name, layer_spec in kv_cache_spec.items():
    same_type_layers[layer_spec].append(layer_name)               # (1) 按 spec 值分桶

min_num_layers = min(len(v) for v in same_type_layers.values())   # (2) 8
group_size = min_num_layers
max_num_layers = max(len(v) for v in same_type_layers.values())   # 24
if max_num_layers < min_num_layers * 1.5:                         # 24 >= 12 → 不启用
    group_size = max_num_layers

for layers in same_type_layers.values():
    num_padding_layers = group_size - len(layers) % group_size    # (3) 0 → 不告警
    num_groups = cdiv(len(layers), group_size)                    # (4) 1 与 3
    for i in range(num_groups):
        grouped_layers.append(layers[i::num_groups])              # (5) 交错切
return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)
```

- **分桶 → 切组（(1)–(5) 汇总）**：`same_type_layers` 按 spec 值分桶（frozen dataclass 按值哈希），本模型正好两桶；五步结果如下（`i` 是桶内下标，层号来自 `layer_name`）：

| 桶 | spec 字段（桶内全等） | 条数（层号） | (2) 组大小 | (3) padding | (4) `num_groups` | (5) 切出的 group |
|---|---|---|---|---|---|---|
| **A** | `FullAttentionSpec`：`block_size=1024`、`num_kv_heads=2`、`head_size=256`、bf16、页 2,072 KB | 8（3, 7, 11, 15, 19, 23, 27, 31） | 8 | 0 | `cdiv(8, 8) = 1` | **group 0** = 3, 7, 11, 15, 19, 23, 27, 31 |
| **B** | `MambaSpec`：`block_size=32768`、state fp32、页 2,072 KB（原生 1,048 KB，S0 pad 后） | 24（0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30） | 8 | 0 | `cdiv(24, 8) = 3` | **group 1** = 0, 4, 8, 12, 16, 20, 24, 28<br>**group 2** = 1, 5, 9, 13, 17, 21, 25, 29<br>**group 3** = 2, 6, 10, 14, 18, 22, 26, 30 |

> 桶迭代顺序 = spec 类型首次出现顺序（3a 合并时第 1 条是 L3 的 `FullAttentionSpec`）→ **group 0 = attn、group 1~3 = mamba**。

最后，`create_kv_cache_group_specs()`（:844-871）对每组再 merge 一次，返回 4 个 `KVCacheGroupSpec`。

**返回**：`list[KVCacheGroupSpec]`（本模型 1 attn + 3 linear，共 4 个元素），每个元素三个字段（kv_cache_interface.py:840-851）：
- `layer_names`：该组 8 条层名；
- `kv_cache_spec`：组内 8 层 merge 后的 spec（group 0 为 `FullAttentionSpec`，group 1~3 为 `MambaSpec`）；
- `is_eagle_group = False`（本模型未启投机解码）。

#### 3c 自适应 + 校验：先投影到 per-worker，再过三道预算门

3b 的 4 组是**全模型视角**（每组 8 层），而 `num_blocks` 是 **per-worker 容量**——每个 worker 只物化本 rank 负责的层（PP 切层；TP 切头已折进各卡 spec 的 `page_size_bytes`）。所以先投影、再逐 worker 预算：

```python
# kv_cache_utils.py:2016-2019
projected_groups_per_worker = [
    _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
    for worker_spec in kv_cache_specs
]
```

投影只做一件事——**按本 worker 的层名过滤**（不新建 spec、不改 `block_size`，组数不变）：

```python
# kv_cache_utils.py:1932-1952（节选）
for group in global_kv_cache_groups:
    worker_layer_names = [
        layer_name for layer_name in group.layer_names if layer_name in worker_spec
    ]
    projected_groups.append(
        KVCacheGroupSpec(worker_layer_names, group.kv_cache_spec,
                         is_eagle_group=group.is_eagle_group and bool(worker_layer_names))
    )
```

本模型投影结果（PP2：PP0 = L0–15、PP1 = L16–31；TP 不切层，同 stage 两卡完全相同）：

| worker | group 0（attn） | group 1 | group 2 | group 3 |
|---|---|---|---|---|
| W0 / W1（PP0） | 3, 7, 11, 15 | 0, 4, 8, 12 | 1, 5, 9, 13 | 2, 6, 10, 14 |
| W2 / W3（PP1） | 19, 23, 27, 31 | 16, 20, 24, 28 | 17, 21, 25, 29 | 18, 22, 26, 30 |

投影之后是三道具预算门（:2027-2057），本模型只走最后一道：

| 门 | 触发条件 | 本模型 |
|---|---|---|
| ① `num_gpu_blocks_override` 改写 `available_memory` | `num_gpu_blocks_override is not None` | **未设**（`None`）→ 跳过 |
| ② 自适应 `max_model_len`（`_auto_fit_max_model_len`） | `original_max_model_len == -1` | 启动命令显式 `--max-model-len 32768` → **不触发** |
| ③ 逐 worker 显存校验（`_check_enough_kv_cache_memory`） | 无条件执行 | **通过**（见下） |

第 ③ 道门的判据是"按 `max_model_len` 算需求，够不够"：

```python
# kv_cache_utils.py:713 起（节选）
needed_memory = get_needed_memory()      # _max_memory_usage_bytes_from_groups
if needed_memory > available_memory:
    raise ValueError(...)                # 抛错时附带估算出的最大可支持长度
```

本模型的需求（通用分支，:1801-1812）：`blocks_needed = Σ cdiv(spec.max_memory_usage_bytes(max_model_len), page_size)`

| 组 | `max_memory_usage_bytes` | 折算块数 |
|---|---|---|
| attn（1 组） | `cdiv(32768, 1024) = 32` 页 | 32 |
| mamba（3 组） | `mamba_cache_mode=none` → 恒 1 块状态 | 1 × 3 = 3 |
| **合计** | — | **35 块** |

`needed = group_size(4) × 2,072 KB × 35 ≈ 283 MB`——不到 `available` 的 1%，校验通过。

#### 3d 生成配置：各 worker 4 张共享张量 + `num_blocks`

逐 worker 用 projected groups 调 `get_kv_cache_config_from_groups()`（:2059-2070 → :1247）。该函数按 groups 形态分四支，本模型走**通用分支**：

| 分支（:1263-1300） | 条件 | 适用模型 | 本模型 |
|---|---|---|---|
| 0 组 | `len(kv_cache_groups) == 0` | attention-free（`num_blocks=1` 只作 null_block 占位） | ✗ |
| 单组 `UniformTypeKVCacheSpecs` | 1 组且 spec 为 UniformType | MLA 多 hidden（DeepseekV2 系） | ✗ |
| 多组 `UniformTypeKVCacheSpecs` | 每组都是 UniformType | DeepseekV4 | ✗ |
| **通用** | 其余 | **本模型（4 组普通 spec）** | ✓ |

通用分支的两个关键量（:1309-1326）：

```python
# kv_cache_utils.py:1309-1326（节选）
group_size = max(len(group.layer_names) for group in kv_cache_groups)   # 4
page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])  # 2,072 KB
num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
kv_cache_tensors = [
    KVCacheTensor(size=page_size * num_blocks, shared_by=[...])
    for i in range(group_size)                                          # 4 张
]
```

`get_num_blocks`（:967）就是一行整除：`available_memory // page_size // num_layers`（`num_layers` 即 `group_size`）；

本模型四个量的落点：

| 量 | 计算式 | 本模型 |
|---|---|---|
| `group_size` | 投影后各个 worker 上最大组层数 | **4** |
| `page_size` | 两族统一页 | **2,072 KB** |
| `num_blocks` | `available // 2,072 KB // 4` | **6003**（启动日志实测） |
| `kv_cache_tensors` | `group_size` 张 | **4 张**，每张 `2,072 KB × 6003 ≈ 11.9 GiB` |

`shared_by` 的组装规则是"第 `i` 张 = 各组第 `i` 层"（:1319-1326），于是每张张量恰好由 **1 个 attn 层 + 3 个 mamba 层**共享：

| 张量 | `shared_by`（PP0） | `shared_by`（PP1） |
|---|---|---|
| T0 | attn L3 + mamba L0 / L1 / L2 | attn L19 + mamba L16 / L17 / L18 |
| T1 | attn L7 + mamba L4 / L5 / L6 | attn L23 + mamba L20 / L21 / L22 |
| T2 | attn L11 + mamba L8 / L9 / L10 | attn L27 + mamba L24 / L25 / L26 |
| T3 | attn L15 + mamba L12 / L13 / L14 | attn L31 + mamba L28 / L29 / L30 |

> 这张表是 4a 的直接前提：`_allocate_kv_cache_tensors` 遍历每张张量的 `shared_by`，一旦同时看到 `MambaSpec` 与 `AttentionSpec` 就把 `hybrid_with_attn_and_mamba` 置 True（model_runner_v1.py:4106-4114），于是**该张量的所有共享层（连 attn 层）都走单张量路径**，不做 K/V 两次独立申请。

#### 3e 对齐：各 worker 4 张卡取 `min(num_blocks)`，张量等比缩

同一请求的 4 组 `block_table` 会被所有 worker 共用（PP 按段索引、TP 只读本头子集），所以 `block_id` 必须在**每张卡**上都对应有效物理行。4 张卡的 profiling 结果可能因显存碎片略有差异，需要统一取最小值：

```python
# kv_cache_utils.py:2075-2085（节选）
min_num_blocks = min(cfg.num_blocks for cfg in kv_cache_configs)
for kv_cache_config in kv_cache_configs:
    num_blocks_old = kv_cache_config.num_blocks
    kv_cache_config.num_blocks = min_num_blocks
    for tensor in kv_cache_config.kv_cache_tensors:
        assert tensor.size % num_blocks_old == 0
        tensor.size = tensor.size // num_blocks_old * min_num_blocks
```

本例四卡实测 `num_blocks` 一致（**6003**），4 张张量的 `size` 不变；若某卡偏少，其余卡按 `min / old` 等比缩——先整除再乘，保证 `size` 仍是 `page_size` 的整数倍。

**收口 · 生成 scheduler 版配置并回写（core.py:265-287）**

`list[KVCacheConfig]` 就绪后、S4 发起前，`_initialize_kv_caches` 依次做三件事：

| # | 动作 | 位置 | 做什么 | 本模型落点 |
|---|---|---|---|---|
| 1 | max_model_len 同步 | core.py:265-277 | S3 若自动调小过 max_model_len，用 `collective_rpc("update_max_model_len")` 下发新值（worker 实现 worker.py:894-905） | 预算够，未触发 |
| 2 | 生成 scheduler 版 | core.py:279 → kv_cache_utils.py:1698 | 断言各份 `num_blocks` 相等 → `deepcopy(kv_cache_configs[0])`；`UniformTypeKVCacheSpecs` 组折叠为组内任一 spec | 4 组均为普通 spec，不折叠，直接 deepcopy |
| 3 | 回写与终检 | core.py:280-287 | `cache_config.num_gpu_blocks = num_blocks`（容量对外可见）；`cache_config.block_size` 取各组最小值；`validate_block_size()` | `num_gpu_blocks = 6003`、`block_size = 1024` |

产出 `KVCacheConfig`（kv_cache_interface.py:854 起）：`kv_cache_groups` **4 个**——1 组 `FullAttentionSpec`（8 层）+ 3 组 `MambaSpec`（各 8 层，合计 24 层；两族 spec 无法 merge，且要求每组层数相等，故取 `min(8,24)=8`）；`kv_cache_tensors` **4 张**，每张 `size = 2,072 KB × 6,003`。

### 3.4 S4 · 落张量 + 编译预热（NPU worker 进程）

S4 与前两步走相同的 RPC 骨架——`ModelExecutor.initialize_from_config()`（core.py:290）内部**连续两个 `collective_rpc`**：先 `initialize_from_config` 完成 4a–4c 落张量，再 `compile_or_warm_up_model` 完成 4d 编译预热。整条链摊开如下（实现主体是 vllm-ascend 的 `NPUWorker` / `NPUModelRunner`）：

```text
ModelExecutor.initialize_from_config()                        core.py:290
├─ collective_rpc("initialize_from_config")                   → 4 个 NPUWorker
│  └─ NPUWorker.initialize_from_config()                      worker.py:907
│     └─ NPUModelRunner.initialize_kv_cache(config)           model_runner_v1.py:3843
│        ├─ deepcopy(config)                                  :3850      不污染引擎进程共享对象
│        ├─ initialize_attn_backend(config)                   :3857
│        ├─ may_reinitialize_input_batch(config)              :3864
│        │  └─ select_common_block_size(...)                  :4759      → kernel_block_size = 128
│        ├─ initialize_kv_cache_tensors(config)               :3865      ← 4a + 4b + 4c
│        │  ├─ 4a _allocate_kv_cache_tensors()                :4082      4 张 int8 张量
│        │  ├─ 4b _reshape_kv_cache_tensors()                 :4353      切 K/V 双池 / conv+ssm 状态视图
│        │  └─ 4c bind_kv_cache()                             :3950-3953 按层名填入 kv_caches
│        └─ register_kv_caches()                              :3883-3884 有传输组时注册
└─ collective_rpc("compile_or_warm_up_model")                 → 4 个 NPUWorker
   └─ NPUWorker.compile_or_warm_up_model()                    worker.py:706  ← 4d
      ├─ _dummy_run(size) × 多档                              :728       触发编译
      ├─ capture_model()                                     :732       捕获 NPUGraph
      ├─ 图内存记账                                          :734-771   打印 --kv-cache-memory 建议值
```

落张量是在 `initialize_kv_cache_tensors` 中，四步各自做什么（每 worker 独立执行，4 张卡同构）：

| 步 | 动作 | 本模型落点 |
|---|---|---|
| 4a 分配 | 按 `kv_cache_tensors` 逐张 `torch.zeros(int8)` | **4 张**，各 `2,072 KB × 6003 ≈ 11.9 GiB` |
| 4b reshape | attn：切 conv padding → 对半切 K/V → view 4D<br>mamba：按 `MambaSpec.shapes` 顺序切 conv / ssm | attn 4 层各得 `(48,024, 128, 2, 256)` 的 K/V 双池；mamba 12 层各得 conv `(6,003, 3, 4,096)` bf16（24 KB/块）+ ssm `(6,003, 16, 128, 128)` fp32（1 MiB/块） |
| 4c 绑定 | 复用上游 `bind_kv_cache` | 16 个层条目（attn 各 `(K, V)`、mamba 各 `[conv, ssm]`）填入 `kv_caches` |
| 4d 编译预热 | `_dummy_run` + `capture_model` + 图内存记账 | 启动日志里的 `--kv-cache-memory` 建议行 |

**4a 之前的偏离点 · kernel 粒度取 128 而非 1024**：`initialize_kv_cache` 先调 `may_reinitialize_input_batch`（:3864），其中 `select_common_block_size`（vllm/v1/worker/utils.py:262-328）负责"取所有 backend 都支持、且能整除 manager 大块的最大档位"：

| 分支 | 条件 | 本模型 |
|---|---|---|
| Case 1（:303-306） | manager 块 1024 本身被所有 backend 支持 → 直接返回、免切分 | **落空**——`AscendAttentionBackend` 只声明 `[128]`（attention_v1.py:143-144） |
| Case 2（:308-327） | 否则把 int 档位从大到小试"能整除 1024 + 被所有 backend 支持"，命中即返回；都不满足 → `ValueError` | 命中 **128**（`1024 % 128 == 0` 且 128 被支持） |
| `MambaSpec` 组 | 走 else 分支 | `kernel_block_sizes = [0]`（:4763-4769），不参与 slot 映射 |

结论：**manager 以 1024 大块记账、kernel 以 128 写读**，一个大块被虚拟切成 8 个细块——这就是 4b 中 `view(nb×8, 128, 2, 256)` 的 8 倍来源；mamba 组不受影响，状态整块读写。

#### 4a 分配：4 张共享张量 → 16 个层视图

3d 下发的 `kv_cache_config` 里只有 **4 张 `KVCacheTensor`**，每张的 `shared_by` = 1 attn + 3 mamba。函数内部有 A / B / C 三条分配分支，hybrid模型**只走 A**：

```text
_allocate_kv_cache_tensors(kv_cache_config)                             # model_runner_v1.py:4082
│
├─ (1) alignment = 2 MiB · layer_kv_cache_spec = 16 层 spec          # :4101-4102
│
├─ (2) for kv_cache_tensor in kv_cache_config.kv_cache_tensors        # 本模型 4 张
│      │
│      ├─ 判形态：use_mamba / use_attn                               # :4108-4113；两者均 True
│      │  hybrid_with_attn_and_mamba = True                           # :4114，一置恒真
│      │
│      └─ for idx, layer_name in enumerate(shared_by)                # :4115；[attn, mamba, mamba, mamba]
│             │
│             ├─ A 单张量路径                                        # :4118-4134  ★ 本模型唯一命中
│             │    条件：hybrid_with_attn_and_mamba
│             │    动作：tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
│             │          for layer_name_inner in kv_cache_tensor.shared_by:
│             │              # shared the kvcache for all shared layers
│             │              kv_cache_raw_tensors[layer_name_inner] = tensor
│             ├─ B 单张量路径（压缩注意力）                          # :4135-4146
│             │    条件：attn∈名 and use_compress and 尚未登记
│             │    ✗ use_compress = hasattr(hf_config, "compress_ratios") → False（:283-285）
│             │
│             ├─ C K/V 分开申请（纯 FA / 稀疏 MLA / DSA）            # :4147-4309
│             │    条件：attn∈名 and 尚未登记 and not use_mamba
│             │    ✗ use_mamba = True → 不进
│             │
│             └─ 否则：无动作（已被前一轮登记覆盖）                  # idx 1..3 的 3 个 mamba 层
│
└─ (3) 断言：4 组层名集合 == kv_cache_raw_tensors 键集合              # :4310-4316；16 == 16
        return kv_cache_raw_tensors
```

**这一步的产物**：`kv_cache_raw_tensors` 里 **16 个层名 → 4 张张量**（4 层/张）。分支 A 只在每张张量的 `idx=0`（attn 层）触发一次，一次就把 `shared_by` 的 4 个层名全部登记到同一张——所以全函数只分配 4 次，却生成了 16 条层名映射：

| 值（张量，`kv_cache_raw_tensors` 的 value） | W0（PP0·TP0）的键 | W1（PP0·TP1）的键 | W2（PP1·TP0）的键 | W3（PP1·TP1）的键 |
|---|---|---|---|---|
| **T0**（`2,072 KB × 6003` ≈ 11.9 GiB） | attn L3 · mamba L0 / L1 / L2 | attn L3 · mamba L0 / L1 / L2 | attn L19 · mamba L16 / L17 / L18 | attn L19 · mamba L16 / L17 / L18 |
| **T1**（≈ 11.9 GiB） | attn L7 · mamba L4 / L5 / L6 | attn L7 · mamba L4 / L5 / L6 | attn L23 · mamba L20 / L21 / L22 | attn L23 · mamba L20 / L21 / L22 |
| **T2**（≈ 11.9 GiB） | attn L11 · mamba L8 / L9 / L10 | attn L11 · mamba L8 / L9 / L10 | attn L27 · mamba L24 / L25 / L26 | attn L27 · mamba L24 / L25 / L26 |
| **T3**（≈ 11.9 GiB） | attn L15 · mamba L12 / L13 / L14 | attn L15 · mamba L12 / L13 / L14 | attn L31 · mamba L28 / L29 / L30 | attn L31 · mamba L28 / L29 / L30 |

> 键写法：`attn L<i>` = `model.language_model.layers.<i>.self_attn`，`mamba L<i>` = `…layers.<i>.linear_attn`；每格 4 个键按 `shared_by` 的登记顺序（attn 先、3 个 mamba 后）全部指向本行那张张量，每 worker 合计 16 条 `layer_name → tensor`。W0/W1 同属 PP0、层名与 spec 相同，但各进程独立分配自己的 4 张张量。

- **表读法**：每行是一张池（本 worker 的一张 `int8` 张量），格内的 4 个层名就是 `kv_cache_raw_tensors` 中指向同一张量的 4 个键——它们拿到的是同一段内存的 4 个 **view**。
- 每格的 3 个 mamba 层分别来自 **mamba 组 1 / 2 / 3**——共享规则是"第 `i` 张池 = 各组第 `i` 层"，所以同一格 4 层的组内下标相同；
- **容量**：4 张池共 ≈47.4 GiB = 本 worker 的全部 KV 预算。

**与纯 FA 的差别**：`use_mamba=False` 的张量会落到分支 C，为每个 attn 层单独申请 K、V 两张 int8 张量；混合模型则把"拆分"推迟到 4b 的 reshape——4a 只分配一张整张量，K/V 与状态的分界由 4b 的切片决定。

#### 4b reshape：切头 → 对半切 K/V 双池 → view 4D 

`_reshape_kv_cache_tensors()`（:4353）只按 spec 类型分两支，**两支切的是同一张 `raw` 张量**；

调用链如下：

```text
_reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)          # model_runner_v1.py:4353
│
├─ (0) kv_caches = {} · layer_kv_cache_spec = _get_layer_kv_cache_specs(kv_cache_config)   # :4369-4370
│
├─ (1) for group in _kv_cache_spec_attn_group_iterator()                    # 本 worker 4 组：attn 1 + mamba 3
│      │  attn_backend = group.backend · current_kv_cache_spec = group.kv_cache_spec
│      │
│      └─ (2) for layer_name in group.layer_names                           # :4374；本 worker 16 层（attn 4 + mamba 12）
│             │  spec = layer_kv_cache_spec[layer_name]                     # :4378
│             │
│             ├─ A AttentionSpec：混合 full-attn（4 层）                    # :4436 → :4495-4582  ★
│             │    条件：use_hybrid_blocks and hybrid_with_attn_and_mamba   # :4495-4500；且非 cache_only/hidden
│             │    动作：raw_k_tensor = raw_v_tensor = kv_cache_raw_tensors[layer_name]   # :4503；同一张 raw！
│             │          sum_page_size_bytes = raw.numel()                  # :4504
│             │          num_blocks = sum_page_size_bytes // page_size_bytes = 6,003   # :4550
│             │          assert num_blocks >= kv_cache_config.num_blocks    # :4559
│             │    ├─ 取 kernel 细块 128                                    # :4561-4562
│             │    │    block_size_chunk = 1,024 // 128 = 8                 # :4564
│             │    │    kv_cache_shape = get_kv_cache_shape(6,003×8, 128, 2, 256)   # :4565-4570
│             │    │        = (2, 48,024, 128, 2, 256)                      # dim0=2 是 K/V 二元组
│             │    ├─ 切三段                                                # :4571-4579
│             │    │    attn_tensor_page_size = prod(shape[1:]) × 2 = nb × 1 MiB   # :4573；K 总量
│             │    │    conv_block_padding_size = raw.numel() − 2 × nb×1MiB = nb × 24 KB   # :4576
│             │    │    raw_kv_tensor = raw[conv_block_padding_size:]       # :4577；切掉段1
│             │    │    raw_k_tensor = raw_kv_tensor[:nb×1MiB]              # :4578；段2 = K
│             │    │    raw_v_tensor = raw_kv_tensor[nb×1MiB:]              # :4579；段3 = V
│             │    └─ view 4D                                               # :4599-4644
│             │          k_shape = shape[1:] = (48,024, 128, 2, 256)        # :4599-4600
│             │          v_shape = k_shape                                   # :4601-4604；本模型无 head_size_v
│             │          k_cache = raw_k_tensor.view(bf16).view(k_shape)    # :4639
│             │          v_cache = raw_v_tensor.view(bf16).view(v_shape)    # :4644
│             │          kv_caches[layer_name] = (k_cache, v_cache)         # :4682
│             │
│             ├─ B MambaSpec：12 个 linear 层                               # :4683-4716  ★
│             │    动作：raw_tensor = kv_cache_raw_tensors[layer_name]      # :4684
│             │          num_blocks = raw.numel() // page_size_bytes = 6,003   # :4687
│             │          assert num_blocks >= kv_cache_config.num_blocks    # :4688
│             │          for shape, dtype in zip(spec.shapes, spec.dtypes)  # :4707；顺序固定：(conv, ssm)
│             │              target_shape = (num_blocks, *shape)            # :4710
│             │              target_idx += prod(target_shape) × dtype_size  # :4712
│             │              tensor = raw[start_idx:target_idx].view(dtype).view(target_shape)   # :4713
│             │              start_idx = target_idx · state_tensors.append(tensor)   # :4714-4715
│             │          kv_caches[layer_name] = state_tensors → [conv, ssm] # :4716
│             │    shapes/dtypes 由 get_state_shape() 给出                   # qwen_gdn_linear_attn.py:421-432 → mamba_utils.py:213-234（TP2）
│             │        conv： (conv_kernel−1, conv_dim/tp) = (3, 4,096) bf16 → 24 KB/块
│             │        ssm ： (num_v_heads/tp, head_v_dim, head_k_dim) = (16, 128, 128) fp32 → 1 MiB/块
│
└─ (3) return kv_caches                                                     # :4720；16 个层条目
```

> `MambaSpec.page_size_bytes` 已被前置 patch 对齐成 2,072 KB（§2.1 · 0c），所以 `num_blocks` 与 attn 层算出同一个 6,003，三段视图按块号一一对齐。

**一张 raw 张量的三段分工**——两族各取所需（段内按块连续）：

| 段（slab） | 每块宽度 | attn 层 | mamba 层 |
|---|---|---|---|
| 段1 `[0, 24 KB × nb)` | 24 KB | **跳过**（conv padding，:4576-4577） | conv state `(nb, 3, 4,096)` bf16 |
| 段2 `[24 KB × nb, 24 KB × nb + 1 MiB × nb)` | 1 MiB | K `(nb×8, 128, 2, 256)` bf16（`1,024 tok × 2 头 × 256`） | ssm `(nb, 16, 128, 128)` fp32 |
| 段3 尾部 `1 MiB × nb` | 1 MiB | V，形状同 K | **未用**（mamba padding） |

- 段1 的 24 KB 不是浪费：attn 当 padding、mamba 当 conv state，**同一段内存两种解释**；段2 的 K 与 ssm、段3 的 V 与 mamba padding 同理——两族字段宽度一致，才能共享同一张 raw 张量。
- 逻辑 `block_size=1024`、算子 `kernel_block_size=128` → attn 视图把物理 block_table **展开 8 份 128-token 细块**；mamba 组 `block_size=32768`，每请求 1 块，不展开。
- 全程只有 `view`（int8 → bf16 / fp32 → 多维），**无 `permute`**。
- 哪个块号归谁用由 4 组各自的 block_table 决定：同一块号只会分给其中一个组（3d 的 `shared_by` + 5c 的单 `BlockPool`）。


**`for group` 里的 `group` 是什么**：`_kv_cache_spec_attn_group_iterator()`（gpu_model_runner.py:7034-7038）把 `self.attn_groups: list[list[AttentionGroup]]` 拍平成一维流（`kv_cache_config` 尚未就绪时直接产出空序列）——**外层下标 = KV cache group id**（与 `kv_cache_config.kv_cache_groups` 对齐），**内层 = 该组内再切出的 AttentionGroup**。内层切分键是 `(backend 类名, 每层 kv_cache_spec, num_heads_q)`（gpu_model_runner.py:6687-6703）：同组层可能因 backend 不同（滑动窗口 / chunked local attention / fast-prefill）、逐层 spec 不同（`UniformTypeKVCacheSpecs`）、Q 头数不同（draft 层）而分裂；Ascend 版 key 只保留前两项（model_runner_v1.py:4843）。每个 AttentionGroup 携带 `backend / layer_names / kv_cache_spec / kv_cache_group_id / metadata_builders`（utils.py:223-259），metadata builder 及其 scratch 按组各建一份。

Qwen3.5 每个 worker 恰好 4 个 KV cache group（1 个 FullAttentionSpec + 3 个 MambaSpec），组内 backend 与 spec 都一致 → **每组只切出 1 个 AttentionGroup**，所以 4b 的 `for group` 循环 4 次、每次 `group.layer_names` 4 条：

| worker | kv_group 0：attn（`self_attn`） | kv_group 1：mamba（`linear_attn`） | kv_group 2：mamba | kv_group 3：mamba |
|---|---|---|---|---|
| W0（PP0·TP0） | L3 / L7 / L11 / L15 | L0 / L4 / L8 / L12 | L1 / L5 / L9 / L13 | L2 / L6 / L10 / L14 |
| W1（PP0·TP1） | L3 / L7 / L11 / L15 | L0 / L4 / L8 / L12 | L1 / L5 / L9 / L13 | L2 / L6 / L10 / L14 |
| W2（PP1·TP0） | L19 / L23 / L27 / L31 | L16 / L20 / L24 / L28 | L17 / L21 / L25 / L29 | L18 / L22 / L26 / L30 |
| W3（PP1·TP1） | L19 / L23 / L27 / L31 | L16 / L20 / L24 / L28 | L17 / L21 / L25 / L29 | L18 / L22 / L26 / L30 |

> 每格 = 一个 AttentionGroup 的 `layer_names`。分组规律：层号 `i mod 4 = 3` 进 attn 组，`i mod 4 = 0/1/2` 依次进 3 个 mamba 组。attn 组 backend 为 Ascend attention backend、spec 为 `FullAttentionSpec`；3 个 mamba 组 backend 为 `AscendGDNAttentionBackend`（gdn.py:64）、spec 为 `MambaSpec(2072 KB)`。W0/W1（或 W2/W3）层名相同，但各进程独立建 backend/builder，所以每个 worker 的 `attn_groups` 都是 4 个内层列表、每个列表 1 个组。

**真实对象（W0 = PP0·TP0）**——展开后就是下面这样（等价表示；`metadata_builders` 在 Ascend 上是建组时直接 new 出的实例，各 1 个）：

```python
attn_spec = FullAttentionSpec(
    block_size=1024, num_kv_heads=2, head_size=256, head_size_v=256,
    dtype=torch.bfloat16, page_size_padded=2_121_728,        # 2,072 KB；逻辑页 2 MiB，NPU 侧对齐到 mamba 页
)
mamba_spec = MambaSpec(
    block_size=32768,                                        # mamba_block_size = max_model_len
    shapes=((3, 4096), (16, 128, 128)),                      # (conv, ssm)，TP2
    dtypes=(torch.bfloat16, torch.float32),
    page_size_padded=2_121_728,                              # 2,072 KB
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,           # gdn/base.py:50-51
)

self.attn_groups = [
    [AttentionGroup(                                         # kv_group 0（1 个组）
        backend=AscendAttentionBackend,                      # attention_v1.py:78
        layer_names=["model.language_model.layers.3.self_attn",
                     "model.language_model.layers.7.self_attn",
                     "model.language_model.layers.11.self_attn",
                     "model.language_model.layers.15.self_attn"],
        kv_cache_spec=attn_spec,
        kv_cache_group_id=0,
        metadata_builders=[AscendAttentionMetadataBuilder],   # 1 个实例
    )],
    [AttentionGroup(                                         # kv_group 1
        backend=AscendGDNAttentionBackend,                   # gdn_attn_builder.py:933
        layer_names=["model.language_model.layers.0.linear_attn",
                     "model.language_model.layers.4.linear_attn",
                     "model.language_model.layers.8.linear_attn",
                     "model.language_model.layers.12.linear_attn"],
        kv_cache_spec=mamba_spec,
        kv_cache_group_id=1,
        metadata_builders=[AscendGDNAttentionMetadataBuilder],
    )],
    [AttentionGroup(                                         # kv_group 2
        backend=AscendGDNAttentionBackend,
        layer_names=["model.language_model.layers.1.linear_attn",
                     "model.language_model.layers.5.linear_attn",
                     "model.language_model.layers.9.linear_attn",
                     "model.language_model.layers.13.linear_attn"],
        kv_cache_spec=mamba_spec,
        kv_cache_group_id=2,
        metadata_builders=[AscendGDNAttentionMetadataBuilder],
    )],
    [AttentionGroup(                                         # kv_group 3
        backend=AscendGDNAttentionBackend,
        layer_names=["model.language_model.layers.2.linear_attn",
                     "model.language_model.layers.6.linear_attn",
                     "model.language_model.layers.10.linear_attn",
                     "model.language_model.layers.14.linear_attn"],
        kv_cache_spec=mamba_spec,
        kv_cache_group_id=3,
        metadata_builders=[AscendGDNAttentionMetadataBuilder],
    )],
]
```

> W1（PP0·TP1）与 W0 层名完全相同（TP 各卡独立构造 backend/builder）；W2/W3（PP1）层号整体 +16：attn 组 L19/L23/L27/L31，mamba 组 1/2/3 分别为 L16/L20/L24/L28、L17/L21/L25/L29、L18/L22/L26/L30。上游 vLLM 的切分键还含 `num_heads_q`（gpu_model_runner.py:6687-6703），Ascend 版去掉它（model_runner_v1.py:4843），所以这里不会因头数再分裂。

#### 4c 绑定：直接复用上游 `bind_kv_cache`

绑定发生在 `initialize_kv_cache_tensors()` 内部（:3950-3953），以同一上游函数把每层对象按层序填入 `self.kv_caches` 并绑进 `static_forward_context`。forward 时：
- **full-attn 层**：`AscendAttentionBackendImpl.reshape_and_cache`（attention_v1.py:1430-1454）取 `kv_cache[0]/[1]` 分发给 NPU 双池算子（`DeviceOperator.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)`，:1444）+ `notify_kv_cache_written()`（:1453）；
- **linear 层**：GDN 状态算子读状态视图、就地步进。

同后端类还提供双池版 `swap_blocks`（:114-126）/`copy_blocks`（:128-140）——NPU 上块级 KV 搬移（P/D 传输、offload）的算子层入口。

#### 4d 编译与预热：NPUGraph `NPUWorker.compile_or_warm_up_model()`（worker.py:706）

逐 size `_dummy_run`（:728）触发编译 → `capture_model()`（:732）捕获 NPUGraph → **图内存事后记账**（:734-771）：把 `权重 + 峰值激活 + non-torch + NPUGraph + 150 MiB` 从 requested（或整机 free）里扣掉，打印两种 `--kv-cache-memory` 建议值 → ATB 首算预热（:775-776）→ 绑核（:779-783）→ 重置随机种子（:786）。

**图内存的"三层归属"**（最容易混的一点）：

- **量法**：`capture_model()` 靠**设备空闲差**度量 —— `start_free − end_free`（`gpu_model_runner.py:6547/6565/6583`），与 torch / non-torch 的分类无关；NPU 只是包了一层并透传返回值（`model_runner_v1.py:5101-5111`，内部 `torch.cuda.*` 经 `_torch_cuda_wrapper()` 桥接到 `torch.npu.*`）。
- **物理归因 → torch 侧**：图池由 torch_npu 的 graph pool 承载（`current_platform.graph_pool_handle()`，`gpu_model_runner.py:6415/6561`），会体现在 `torch_memory = memory_reserved()`；而 `non_torch_memory = cuda_memory − torch_memory` **不含**它。
- **账本归因 → 两个都不进，单独第四项**：不算 `torch_peak_increase`（S2 取的是**图捕获前**峰值，worker.py:567 注释），也不算 `non_torch_increase`（其时间窗早于本步的 capture），而是这里**单列**加到 non_kv（worker.py:743-748）。

> 与 §3.2 分配图里"把它画在 `total − requested` 那块余量上"不矛盾：那里说的是**预算出处**（util 预留），这里说的是**归因口径**——它由 torch 申请，只是迟到于 S2 的记账窗口。

至此 NPU 物理侧就绪：每 worker **两种形态并存**——full-attn 4 层的 K/V 双池（各 `(nb×8,128,2,256)` bf16）+ linear 12 层的状态张量（每层 conv `(nb,3,4,096)` bf16 与 ssm `(nb,16,128,128)` fp32 两视图）。

**不变的存储契约**（所以上层零改动）：同一份 `KVCacheConfig` 与跨 worker `min(num_blocks)` 对齐；一律从字节池起步、与 dtype 解耦；都用同一个 `bind_kv_cache` 绑定；逻辑侧 `block_id == 物理张量 dim0 行号` 的约定恒成立——S5 逻辑层（§4）、`BlockPool`、`KVCacheManager` 的分配/释放逻辑对此完全无感。

## 4. 逻辑侧：逻辑管理器初始化（引擎进程 · 纯元数据装配）

本侧特点：**全部对象都是纯元数据**（`KVCacheBlock` 只含 `block_id` 与引用计数等）、**启动期一次性建好**、**与平台完全无关**。四层对象（门面 / 协调器 / 单类型 manager / 块池）的装配入口与顺序见 §4.1。

### 4.1 S5 建逻辑管理器

S4 的 RPC 全部返回后，`_initialize_kv_caches`（core.py:317）把 scheduler 版 `KVCacheConfig` 交回 `EngineCore.__init__`，引擎进程随即在 **S5** 里把四层逻辑对象一次装好。整条链自上而下只有四跳：

```text
Scheduler.__init__(vllm_config, kv_cache_config, block_size=32768, ...)   sched/scheduler.py:66
└─ KVCacheManager.__init__(...)                                          kv_cache_manager.py:111     ← 5a 门面
   └─ get_kv_cache_coordinator(...)                                      kv_cache_coordinator.py:689 ← 5b 选型
      └─ KVCacheCoordinator.__init__(...)                                kv_cache_coordinator.py:65
         ├─ BlockPool(num_gpu_blocks=6003, ...)                          block_pool.py:149           ← 5c 块池
         └─ get_manager_for_kv_cache_spec(...) × 4                       single_type_kv_cache_manager.py:1311 ← 5d 单类型 manager
```

| 步 | 层 | 动作 | 本模型落点 |
|---|---|---|---|
| **5a 门面** | 第 5 层 | `Scheduler.__init__` 建 `KVCacheManager` | 1 个，`enable_caching=False` |
| **5b 协调器** | 第 4 层 | 工厂按"缓存开关 × 组数"选型 | `KVCacheCoordinatorNoPrefixCache`，三条整除断言全过 |
| **5c 块池** | 第 2 层 | `BlockPool(num_blocks)` 全量预建 `KVCacheBlock` | **6003 块**，块 0 摘作 `null_block`，可分配 6002 |
| **5d 单类型 manager** | 第 3 层 | 每组一个，注册表查表实例化 | **1 个 `FullAttentionManager` + 3 个 `MambaManager`** |

> 构造顺序是**自底向上**：5c、5d 两个共享对象先建好，5b 把它们包进协调器，5a 再把协调器交给 Scheduler。4 个 manager 共享同一个 `BlockPool`，但各自 `req_to_blocks` 独立 → 运行期同一请求 4 组 `block_table`。以下四节按 5a → 5d 展开。

#### 5a 门面：`Scheduler.__init__` 创建 `KVCacheManager`（第 5 层）

`Scheduler.__init__`（sched/scheduler.py:66）中创建逻辑侧顶层门面：

```python
# sched/scheduler.py:231-244（节选）
self.kv_cache_manager = KVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=self.max_model_len,
    enable_caching=self.cache_config.enable_prefix_caching,   # False
    ...
    scheduler_block_size=self.block_size,   # 32768
    hash_block_size=hash_block_size,        # 32768
)
```

三个联动点：

1. `enable_caching` 接 `cache_config.enable_prefix_caching`（False）——决定协调器选型（见 5b 工厂分支）；
2. KV connector 在场时 `connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)`（scheduler.py:247-248）；
3. Scheduler 接收运行标志 `has_mamba_layers` / `needs_kv_cache_zeroing`（scheduler.py:259-260，来自 `kv_cache_config`）——**本模型 `has_mamba_layers=True`**，据此调整调度。

#### 5b 协调器：工厂三分支与基类装配（第 4 层）

`KVCacheManager.__init__`（kv_cache_manager.py:111）核心动作是经工厂创建协调器并把句柄提级缓存。工厂 `get_kv_cache_coordinator()`（kv_cache_coordinator.py:689-741）按"是否开前缀缓存 × group 数"分三路：

```python
# kv_cache_coordinator.py:702-703（本模型命中）
if not enable_caching:
    return KVCacheCoordinatorNoPrefixCache(...)   # 支持任意组数，无前缀缓存特性
```

> **本模型走第一分支 `KVCacheCoordinatorNoPrefixCache`**（前缀缓存关闭——即使它是多组）。该类注释原文："supports arbitrary numbers of KV cache groups (including 0 groups). Does not implement any features related to prefix caching."，即**多组结构照常装配（4 组各一个 manager），但前缀查找恒空、不做任何缓存**（5d 与运行期篇 §4.1.2）。

| 分支 | 条件 | 适用 | 本模型 |
|---|---|---|---|
| `KVCacheCoordinatorNoPrefixCache` | `not enable_caching` | 关前缀缓存；支持任意组数（含 0 组） | ✓ |
| `UnitaryKVCacheCoordinator` | 1 组且开缓存 | 单组 + 前缀缓存 | ✗ |
| `HybridKVCacheCoordinator` | 多组且开缓存 | 多组 + 前缀缓存（组间命中求交） | ✗ |

协调器基类 `KVCacheCoordinator.__init__`（kv_cache_coordinator.py:65-127）完成两件装配动作：

```python
# kv_cache_coordinator.py:84-119（节选）
assert scheduler_block_size % hash_block_size == 0 and all(
    scheduler_block_size % g.kv_cache_spec.block_size == 0
    for g in kv_cache_config.kv_cache_groups
)                                                        # 粒度整除终检：32768%32768==0 ✓、32768%1024==0 ✓
self.block_pool = BlockPool(                             # 5c 建块池
    num_gpu_blocks=kv_cache_config.num_blocks,           # ≈6003
    enable_caching=enable_caching,
    hash_block_size=hash_block_size,
)
self.single_type_managers = tuple(                       # 5d 每 group 一个 → 本模型 4 个
    get_manager_for_kv_cache_spec(kv_cache_spec=kv_cache_group.kv_cache_spec, ...)
    for kv_cache_group in kv_cache_config.kv_cache_groups
)
```

> 三条断言全过：`32768 % 32768 == 0`、`32768 % 1024 == 0`、`32768 % 32768 == 0`——正是 S3 编排"4 组 + 统一页"在逻辑侧的呼应。

#### 5c 块池：`BlockPool.__init__` 一次建 num_blocks 个逻辑块（第 2 层）

`BlockPool`（block_pool.py:149）是逻辑侧唯一持有"块"的地方，全部启动期一次性装配：

```python
# block_pool.py:162-177（节选）
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx) for idx in range(num_gpu_blocks)   # ≈6003 个块，block_id = 序号 0..n-1
]
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
# To represent a placeholder block with block_id=0.
# The ref_cnt of null_block is not maintained, needs special care to
# avoid freeing it.
self.null_block = self.free_block_queue.popleft()        # 摘走队头（block_id=0）
self.null_block.is_null = True
```

| 属性 | 启动期值 | 运行期角色 |
|---|---|---|
| `blocks` | `[KVCacheBlock(0) … KVCacheBlock(6002)]`，每块仅含 `block_id`、`ref_cnt=0` 等元数据 | 全量预建，运行期永不 new 新块——分配只是改指针/计数 |
| `free_block_queue` | 全部块按 **block_id 升序**串成双向链表 | 分配 `popleft`、驱逐后回插；自研双向链表支持 O(1) 摘中间块，队序即驱逐序 |
| `cached_block_hash_to_block` | **空** dict（`BlockHashToBlockMap`） | 前缀缓存查表——**本模型 enable_caching=False，此表运行期始终为空** |
| `null_block` | 队头 pop 出的 `block_id=0`，`is_null=True` | 占位块，不维护引用计数、永不再分配 |

摘走 `null_block` 后实际可分配块为 **6002**，编号 1..6002——与物理侧"张量 dim0 的 0 号行闲置无人写"一一对应（§5）。**NPU 上逻辑块只有这一份：一个 `block_id` 同时命名 K/V 双池的行，以及 linear 层状态张量的对应块。**

#### 5d 单类型 manager：`FullAttentionManager` + `MambaManager`（第 3 层）

协调器基类为每个 group 调 `get_manager_for_kv_cache_spec()`（single_type_kv_cache_manager.py:1311），查 S1 前置阶段 `register_all_kvcache_specs` 填充的注册表：

- 1 个 attn 组（`FullAttentionSpec`）→ **`FullAttentionManager`**（类定义同文件 :521）；
- 3 个 linear 组（`MambaSpec`）→ **`MambaManager`**（类定义同文件 :955，各实例化一次）。

| 实例 | `kv_cache_spec` | `block_size` | 层数（全模型 / 每卡） | `req_to_blocks` |
|---|---|---|---|---|
| `FullAttentionManager` ×1 | `FullAttentionSpec` | 1024 | 8 / 4 | 每请求 1 条（attn 组 `block_table`） |
| `MambaManager` ×3 | `MambaSpec` | 32768 | 各 8 / 4 | 每请求各 1 条（3 个 linear 组 `block_table`） |

两者共享同一 `BlockPool`，但**各自 `block_size` 不同**（基类装配，single_type_kv_cache_manager.py:64-95）：

```python
# single_type_kv_cache_manager.py:64-95（基类 __init__ 节选）
self.scheduler_block_size = scheduler_block_size          # 32768（4 组相同）
self.block_size = kv_cache_spec.block_size                # attn 组=1024；linear 组=32768
self.kv_cache_spec = kv_cache_spec                        # FullAttentionSpec / MambaSpec
self.block_pool = block_pool                             # 共享的同一个块池（4 组共享第 2 层）
self.enable_caching = enable_caching                      # False
self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
                                                          # 每组各持一份：request_id → 本组块序列（= 本组 block_table）
```

> 4 个 manager 共享同一 `BlockPool`，但 `req_to_blocks` **各持一份** → 运行期同一请求有**4 组 block_table**（见运行期篇 §4.1.2）。`MambaManager` 的核心运行期方法（`find_longest_cache_hit`、`cache_blocks` 等）在本模型前缀缓存关闭时不产生额外行为；`FullAttentionManager.find_longest_cache_hit` 等同样不执行（协调器恒返回空）。

### 4.2 推导调度粒度：`resolve_kv_cache_block_sizes()`

**它在逻辑侧干什么**：逻辑侧同时存在三套"块"——各组 manager 块（attn 1024 / mamba 32768）、调度块、哈希块。这个函数在装配前把"多组各自的块大小"折算成**调度器只用一份**的两个粒度，作为 `Scheduler(...)` 的入参；它不创建任何对象，只定口径。

| 返回值 | 含义 | 本模型 | 谁消费 |
|---|---|---|---|
| `scheduler_block_size` | 跨组统一的 token 粒度 | **32768**（LCM） | Scheduler → `KVCacheManager` → 协调器整除断言（5b）、各 manager 的 `scheduler_block_size` |
| `hash_block_size` | `Request.block_hashes` 的计算粒度 | **32768**（等于上一行，哈希实际不生效） | `get_request_block_hasher`（本模型不装配） |

五个出口（kv_cache_utils.py:616-656），本模型走第 3 个：

| # | 条件 | 返回 | 本模型 |
|---|---|---|---|
| 1 | `len(groups) <= 1` | `(cache_block_size × dcp × pcp, 同)` | 4 组 → 跳过 |
| 2 | 多组且 `dcp/pcp != 1` | `ValueError`（多块大小不支持上下文并行） | 均为 1 → 跳过 |
| 3 | 多组、**既无前缀缓存也无 connector** | `(LCM, LCM)` | ✓ 命中（:633-634） |
| 4 | 多组、有缓存但 mamba 块 ≠ `cache_config.block_size` | `(LCM, LCM)`（哈希退化） | 若开缓存仍命中（32768 ≠ 1024） |
| 5 | 多组、有缓存且 mamba 块已对齐 | `(LCM, hash_block_size 或 GCD)` | ✗ |

本模型命中的就是这一段：

```python
# kv_cache_utils.py:626-634（节选）
group_block_sizes = [g.kv_cache_spec.block_size for g in groups]   # [1024, 32768]
scheduler_block_size = math.lcm(*group_block_sizes)               # 32768
connector_enabled = vllm_config.kv_transfer_config is not None
if not (cache_config.enable_prefix_caching or connector_enabled):
    return scheduler_block_size, scheduler_block_size             # (32768, 32768)
```

- **为什么取 LCM 而不是 GCD**：一份调度块要同时覆盖 1024 与 32768 两种组边界，取最小公倍数才能保证"任一组边界都落在调度块边界上"；取 GCD 会把 mamba 组切碎。
- **32768 在本模型只做两件事**：传给 `KVCacheManager` 的 `scheduler_block_size`（再被协调器拿去断言 `32768 % 1024 == 0`），以及作为 `self.block_size` 的兜底参数——**真正按 token 切块用的仍是 `cache_config.block_size` = 1024**（`_get_num_new_tokens`，sched/scheduler.py:319-334）。
- **哈希为什么也退成 32768**：本模型关前缀缓存且无 connector，直接返回 scheduler 值；即便开了缓存，`mamba_cache_mode != "align"` 时 mamba 组 32768 ≠ 1024 也会命中第 4 个出口退成同值——所以哈希粒度在本模型恒为 32768、且从不使用。

至此两侧全部装配完成：物理侧两类张量就绪、逻辑侧 n 个块与 4 个 manager（1 attn + 3 linear）就绪、两侧靠 `block_id == 行号` 约定桥接，引擎进入接收请求的定态。

## 5. 两侧对应关系：一份配置、一个编号约定

两侧之间**没有任何对象引用**，衔接只靠两样东西：**一份配置**（分组信息 + `num_blocks`）与**一套编号**（`block_id == 张量 dim0 行号`）。下面按"传了什么 → 怎么对上号 → `block_id` 的三层语义 → 运行期怎么读 → 落在哪张卡 → 本模型数字"六层递进。

### 5.1 交接清单：两侧之间传了什么

| 交付物 | 生产（侧 · 阶段） | 消费方 | 用途 |
|--------|------------------|--------|------|
| `kv_cache_configs`（list，每 worker 一份） | 物理侧 S3 | 物理侧 S4：各 NPUWorker | Worker 遍历 `kv_cache_tensors`，full-attn 层分 K/V 双池、linear 层按状态切片绑定 |
| `num_blocks`（跨 worker 对齐） | 物理侧 S3 | 逻辑侧 S5 的 `BlockPool` | `BlockPool.__init__` 建出 `KVCacheBlock(0 .. num_blocks-1)`（块 0 开池即留作 `null_block` 占位，实际可分配 `num_blocks-1`） |
| scheduler 版 `KVCacheConfig`（deepcopy 单份） | 物理侧 S3 收口（`_initialize_kv_caches` 返回值） | 逻辑侧 S5：Scheduler / KVCacheManager / Coordinator | 分组信息、`num_blocks`、`has_mamba_layers` / `needs_kv_cache_zeroing` 等运行标志 |
| `cache_config.num_gpu_blocks` 回写 | 物理侧 S3 | EngineCore / Scheduler / KV connector | KV 容量对外可见，供准入与 connector 握手 |
| `kv_caches[layer]` | 物理侧 S4 | Ascend attention 算子 + GDN 状态算子 + NPUModelRunner | **full-attn 层**：`(k_cache, v_cache)` 两张 `(nb×8,128,2,256)`——forward 时 `reshape_and_cache` 按 `slot_mapping` 写新 K/V 各自进池、注意力核按 `block_table` 沿第 0 维 gather 读历史；**linear 层**：状态张量，每步就地步进 |
| `KVCacheManager` 全套（Coordinator + BlockPool + manager×4） | 逻辑侧 S5 | Scheduler 运行期 | `get_computed_blocks` / `allocate_slots` / `free` 统一 API，产出 4 组 `block_table` |
| `scheduler_block_size` / `hash_block_size`（32768 / 32768） | 逻辑侧 S5 的 `resolve_kv_cache_block_sizes()` | Scheduler → KVCacheManager → 协调器 / 各 manager | 跨组 token 对齐不变量（协调器整除断言）；本模型关缓存且无 connector → 两者同值、哈希实际不使用 |

> 一条隐含约束：`kv_cache_configs`（交 S4 各 worker）与 scheduler 版 `KVCacheConfig`（deepcopy 单份交 S5）都由 S3 收口产出，因此两侧看到的分组与 `num_blocks` 必然一致——见 §5.2 约定 1。

### 5.2 三条约定：从配置到编号

1. **同一份配置定义容量**：S3 收口产出完整版 `list[KVCacheConfig]`（每 worker 一份、`num_blocks` 已 min 对齐）交给 S4；deepcopy 出的 scheduler 版单份交给 S5。`generate_scheduler_kv_cache_config` 的入口断言兜底保证两侧看到的 `num_blocks` 必然相等——物理侧各张量的 block 轴大小、逻辑侧 `BlockPool.blocks` 长度都由同一个数字推出。NPU 特有的一层保障：`NPUModelRunner.initialize_kv_cache` 还会 `deepcopy(config)`（model_runner_v1.py:3850），确保 worker 侧的任何配置增补不污染引擎进程共享的对象。
2. **从 0 顺序编号的约定**：逻辑侧 `BlockPool.__init__` 一次性建出 `num_blocks` 个 `KVCacheBlock`，`block_id` 取序号 `0 … n-1`；物理侧 reshape 后每张张量的 block 轴大小恰为同一 `num_blocks`。两侧在 block 轴上"位置等同"——`block_id` 即张量行号，fancy index 直接取行，无查表、无拷贝。
3. **NPU 的两种物理形态，同一套编号约定**：
   - **full-attn 层**（K/V 双池）：K 池与 V 池**各有一套同序号的行空间**——同一个 `block_id` 同时命中两池各自的第 `block_id` 行；索引方式 `kv_caches[layer][0][block_ids]`（K）/ `[1][block_ids]`（V）；
   - **linear 层**（单张量状态）：`block_id` 定位状态张量的对应块（conv + temporal 视图）；
   - 逻辑侧对形态差异完全无感，`block_table` 按组各一份。

### 5.3 `block_id` 的三层语义

“`block_id` 即行号”只解释第一层；块多大、怎么换算必须按组分看——三个层面依次是：

| 层面 | 由谁决定 | 表现 |
|---|---|---|
| **编号空间** | 共享的 `BlockPool`（全局序号 `0 … n-1`） | 4 组、全部 32 层张量共用同一空间 → `block_id == dim0 行号`；**与 attn block size 无关** |
| **序列位置 → block_id 的换算** | **各组自己的 `spec.block_size`**（`single_type_kv_cache_manager.py:66`，分配时 `cdiv(num_tokens, self.block_size)`） | attn 组按 1024 算、mamba 组按 32768 算 → 4 张 `block_table` 长度不同（序列 L：`cdiv(L,1024)` vs `cdiv(L,32768)`）；**同一个 `block_id=3` 在 attn 表里是第 3 个 1024-token 块，在 mamba 表里是第 3 个整序列状态块——同号不同义** |
| **每格多少字节（容量刻度）** | **attn 的 block size**（mamba 页被垫到同值） | attn 页 = `block_size × 每 token 字节`；patch 用 `assert 每 token K 页 × attn_block_size == ssm 页`（patch :95）反推出 1024 再垫成统一页，故 `num_blocks = 预算 ÷ 统一页 ÷ 组内层数` 的"格"由 attn 定 |

> 一句话：**编号共享、映射各算、刻度由 attn 定**——`block_id` 本身不含"这个块多大"的信息，块的 token 跨度必须按组查各组的 `block_size`（§2.1 量总表）。`scheduler_block_size`（32768）只用于命中/缓存边界，也不参与编号。

### 5.4 运行期怎么读：一次 forward 的取数路径

**双池形式下 block 轴恒为 dim0，无需哨兵值运行时探测**——这是 KV 传输注册约束（buffer dim0 = num_blocks、地址 2MiB 对齐）的直接副产物。

forward 伪代码（NPU，`block_ids` 即该请求**本组**的 `block_table`）：

```python
# full-attn 层（K/V 双池）：
block_ids = fm_blocks(request_id)                 # attn 组 block_id 列表，来自 FullAttentionManager.req_to_blocks
k = kv_caches[layer][0][block_ids]                # K 池：dim0 fancy indexing（每块 8 个 128 细块）
v = kv_caches[layer][1][block_ids]                # V 池：dim0 fancy indexing（同一批行号）
# 写路径入口：DeviceOperator.reshape_and_cache 按 slot_mapping scatter（attention_v1.py:1444）
# slot_mapping = block_id * 128 + 细块内偏移（每 token 一个槽位，同一值同时定位两张池）

# linear 层（单张量状态）：
state_blocks = gm_blocks(request_id)              # 3 个 linear 组各自 block_id 列表（每组每请求 1 块）
state = state_cache[layer][state_blocks]          # 按状态视图读写，每步就地步进（无 block_table 读历史）
```

> 4 组的 `req_to_blocks[request_id]` 分别来自 `FullAttentionManager` / `MambaManager`（S5 装配的产物，5d）。`null_block`（`block_id=0`）在 `BlockPool.__init__` 立即摘走作占位，实际可分配数为 `num_blocks-1`，物理 0 号行因此闲置。

### 5.5 并行布局：同一请求的 KV 落在哪

4 张卡各存 16 层（12 linear 状态 + 4 full-attn K/V 双池）；同一 PP stage 的两个 TP rank 存**同层、不同 KV 头子集**（full attn 各 2 头）。同一请求的 KV 被切成多段：`block_table` 跨 PP 按阶段分段索引，跨 TP 各 rank 只读自己的头子集。调度器仍只认 `block_id`（**4 组各一份 block_table**），对 PP/TP 布局与两种物理形态完全透明；`BlockPool` 在引擎进程**只有一份**。

### 5.6 本模型实测量（收口）

> §1–§4 的结论落到真实数字（Qwen3.5-4B，pp2tp2，attn block_size=1024）：

| 量 | 计算式 | 实测值 |
|---|---|---|
| 每卡 KV 可用显存 | `worker.py:593 info_once` 日志 | **51.90 GiB** |
| attn 统一页（两族全等） | attn 页 2 MB + conv 页 24 KB | **2,072 KB**（padding 1.16%） |
| attn 组 block_size | patch 页对齐 | **1024** |
| mamba 组 block_size | `patch_mamba_config.py:146`（前缀缓存关闭） | **32768**（= max_model_len） |
| `num_blocks` | `available // page(2,072 KB) // group_size(4)`，再跨 worker 取 min | **6003**（= 6,147,072 tokens ÷ 1024） |
| scheduler / hash block_size | `resolve_kv_cache_block_sizes`（多组 LCM / 关缓存退回） | **32768 / 32768** |
| 每 worker 张量 | S3 下发 **4 张**共享张量（每张 1 attn + 3 mamba 层）→ 4b 展开 | 分配 4 张（各 ≈11.9 GiB）、reshape 后 **16 个层条目 / 32 个张量视图**（8 K/V + 24 状态 = 12 层 × (conv, ssm)） |
| full-attn 单张 shape | `view(num_blocks×8, 128, 2, 256)` bf16 | 逻辑块 1024-token、物理 8 细块 |

## 6. 关键公式汇总（速查）

沿两侧管线顺序排列（Qwen3.5-4B pp2tp2 实测口径；单位 1024 进制）：

| 公式 | 所属阶段/侧 | 含义 | Qwen3.5-4B 实测 | 出处 |
|------|------------|------|------------------|------|
| `attn_block_size = kernel_block_size × cdiv(ssm_page, kernel_block_size × attn_k_page)` | 前置 patch | 按 mamba ssm 状态页反推 attn 块 token 数 | `128 × cdiv(1MB, 128×1KB) = 128×8 = 1024` | patch_mamba_config.py:94 |
| `统一页 = attn 页 + conv 页` | 前置 patch | 两族 `page_size_bytes` 全等（多组共享 block 账本的前提） | `2 MB + 24 KB = 2,072 KB`（padding 1.16%） | patch_mamba_config.py:113-117 |
| `mamba_block_size = max_model_len`（前缀缓存关闭） | 前置 patch | mamba 组每请求只占 1 个整序列状态块 | `32768` | patch_mamba_config.py:146 |
| 每请求 mamba 状态块数：`none`→`1+num_spec`；`all`→`cdiv(num_tokens, bs)+num_spec`；`align`→`命中块数+2+num_spec`（峰值） | 前置 patch / S5 逻辑侧 | 三模式内存账的唯一权威来源 = `MambaSpec.max_memory_usage_bytes`（`all` 的块表宽度按 `max_model_len` 预留；`align` 的旧块滞后一步释放，故比稳态多算 1 块） | 本模型 `none`：恒 1 块 | kv_cache_interface.py:627-636、§2 |
| `available = total × util − weights − activations` | S2 物理侧 | 单卡 KV 显存预算（profile_run 实测）；**NPU 无图项** | **51.90 GiB** | §3.2.1 |
| `num_blocks = 多组统一 block 空间协调 → min 对齐` | S3 物理侧 | per-worker 容量：attn 大页与 mamba 状态页共享预算后取 min | ≈ **6003** | §3.3d（公式）、§3.3e（min 对齐） |
| `KVCacheTensor.size = page_size_bytes × num_blocks` | S3 物理侧 | **每张**共享张量的字节数（每张由 1 attn + 3 mamba 层共用） | `2,072 KB × 6003 ≈ 11.9 GiB`（共 4 张） | §3.3d |
| `scheduler_block_size = LCM(各组 block_size)` | S5 逻辑侧 | 跨组 token 对齐不变量（命中掩码/缓存边界/offload 粒度；**非**调度分片——分片用 `cache_config.block_size`） | `LCM(1024, 32768) = 32768` | kv_cache_utils.py:627、sched/scheduler.py:319 |
| `hash_block_size = scheduler_block_size`（关缓存无 connector） | S5 逻辑侧 | 哈希粒度；关前缀缓存时退回 | `32768` | kv_cache_utils.py:633-634 |
| `num_new(attn 组) = max(cdiv(num_tokens, 1024) − 已持块数, 0)` | S5 逻辑侧 | attn 组每步新块数 | prefill 1 / decode 0~1 | 5d、运行期篇 §4.1.3 |
| `num_new(mamba 组) = max(cdiv(num_tokens, 32768) − 已持块数, 0)` | S5 逻辑侧 | mamba 组每步新块数 | 恒 0~1（3 组各 1 块） | 5d、运行期篇 §4.1.3 |
| `non_kv_memory = weights + peak_activation + non_torch + npugraph + 150MiB` | 4d 物理侧 | 图内存事后记账：给出两种 `--kv-cache-memory` 建议值 | 启动日志建议行 | worker.py:741-771 |
| `逻辑块数 == 每张张量行数 == num_blocks` | S5 ↔ S4 | 同一份配置保证两侧容量一致（断言兜底） | 6002 个可分配块（0 号为 null_block） | 5c/§5 |

---

## 7. 设计要点小结

1. **规格先行且平台透明**：所有显存计算源自 spec 的 `page_size_bytes`；NPU 的 Runner 重写负责两类 spec 分流与页对齐，S3 编排因此完全平台无关。
2. **五阶段单向管线、每卡并行、物理先于逻辑**：`spec → profile → 编排 → 落张量 → 建逻辑层`，S1/S2/S4 经 collective RPC 执行；S4 编译预热完成后 S5 才装配逻辑层，随后 `freeze_gc_heap` 封门。
3. **多 group 是 GDN 混合模型的核心特征**：两类 spec 无法 merge → 全模型 **4 个 KV group**（1 组 attn 8 层 + 3 组 linear 各 8 层；切分规则 `group_size = min(8,24) = 8`，见 §3.3b）；S5 因此走"每 group 一个 manager"路径（4 个 manager），且因前缀缓存关闭命中 **`KVCacheCoordinatorNoPrefixCache`**（支持多组但无缓存特性）。
4. **两类层 block_size 不同**：attn `1024`（页对齐产物）、mamba `32768`（= max_model_len）→ `scheduler_block_size = LCM = 32768`，且这一数字决定调度对齐与 hash 粒度。mamba 组的取值由 `mamba_cache_mode` 决定：`align` → 1024（与 attn 组对齐、状态按块入哈希），`none` / `all` → 32768（本部署 `none`）——三模式语义、块账与命名由来见 §2。
5. **容量是 per-worker、且跨组协调的**：`num_blocks` 在统一 block 空间下由 attn 大页与 mamba 状态页共享预算后取 min，再跨 worker `min` 对齐并等比缩 `KVCacheTensor.size`；收口断言保证下发两侧的配置必然相等。
6. **两种物理形态并存**：full-attn 层单张连续张量 → 切头 padding → 对半切 K/V 双池 → `view(nb×8,128,2,256)`（逻辑 1024-token、物理 8×128 细块）；linear 层单张量 → 状态视图。**`block_id == dim0 行号` 契约在两种形态上同时成立**。
7. **`block_id == 行号` 桥接、物理-逻辑分离**：调度决策零显存拷贝，对 PP/TP 布局与两种物理形态完全透明（§5）。
8. **配置是唯一权威**：`num_blocks` 等容量事实全部由物理侧一次性推导，物理侧与逻辑侧只消费、不自行计算。
9. **逻辑侧纯元数据、启动期一次装配、平台无关**：`BlockPool` 全量预建 n 个 `KVCacheBlock`、空闲队列是 O(1) 摘插的双向链表；装配完成后冻结启动堆。
10. **注册表闭环扩展点（插件同权）**：物理侧起点 `register_all_kvcache_specs` 注册 spec↔manager 映射，逻辑侧 S5 `get_manager_for_kv_cache_spec` 查表实例化——新增 spec（含 NPU 自定义）只需两头登记（§1.3/5d）。
11. **NPU 的两处"事后性"**：NPUGraph 内存不在 S2 预算公式内（capture 发生在其后的 4d），靠事后建议日志补偿——因此 NPU 生产上推荐直接指定 `--kv-cache-memory`。

---
