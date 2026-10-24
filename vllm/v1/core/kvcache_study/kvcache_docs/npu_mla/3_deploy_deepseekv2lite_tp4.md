# DeepSeek-V2-Lite（MLA 模型）单机 TP4 部署实测记录（NPU / vllm-ascend）

> 本篇是 KV Cache 学习系列 MLA 路线的**实战篇**：在昇腾 A3 4 卡 pod 上以 **TP4** 部署 **DeepSeek-V2-Lite** 并冒烟验证，记录全流程与关键实测数字，并把每一步启动日志证据回挂到系列文档的三侧五阶段模型上。
>
> 与系列各篇的关系：理论篇 [`0_kvcache_arch.md`](./0_kvcache_arch.md)（五层架构 + §6 MLA 扩展行）、类型篇 [`1_kvcache_type.md`](../../kvcache_docs_v2/1_kvcache_type.md)（`AscendMLAAttentionSpec` 冻结值 + latent/rope 双池）、启动篇 [`1_start_init.md`](./1_start_init.md)（五阶段全链 + 数字一字不差推演）、运行篇 [`2_runtime_sequence.md`](./2_runtime_sequence.md)（一条请求端到端 + 前缀缓存实测闭环）——本篇只做**部署操作与日志实录**，MLA 机制结论引用前四篇不复述。对照兄弟路线的 GDN 混合模型部署篇 [`../npu_hybrid/3_deploy_qwen35_pp2tp2.md`](../npu_hybrid/3_deploy_qwen35_pp2tp2.md)。

**实测环境一览**：

| 项 | 值 |
|---|---|
| 部署日期 | 2026-10-18（pod 日志 UTC = 本地 −8h，下文时间线统一 pod UTC） |
| 容器 | itask pod `gggtest2`（a3，4 × hpu910a3，32C / 400GB / boot-disk 200Gi 挂 `/data`；Pod IP 172.16.78.208，超节点 2f377d71） |
| 镜像 | `hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| vllm / vllm-ascend | 0.23.0 / 0.23.0（`pip show` 确认，`vllm 0.23.0+empty`） |
| 模型 | `modelhub_97542_deepseek-v2-lite:36500041_20260318110950`（创建 pod 时 `--model` 自动拉取） |
| 并行策略 | TP4 × PP1 = 1 pod × 4 NPU（单机 4 rank，HCCL `tcp://127.0.0.1`） |
| 服务 | OpenAI API，端口 8100；`--max-model-len 32768`（原生 163840 = yarn 40×4096，为缩短 profile/校验时间调小） |

---

## 1. 模型架构速览（config.json 实测解析）

架构 `DeepseekV2ForCausalLM`（纯文本 decoder-only；`auto_map` 带自定义代码但 vllm 走内置实现）。与 KV cache 相关的 MLA 三要素（config.json 原件归档：[`./logs/dsv2lite_config.json`](./logs/dsv2lite_config.json)）：

| 参数 | 值 | KV cache 含义 |
|---|---|---|
| `kv_lora_rank` | 512 | **latent 维 c_KV → latent 池宽（每块容量 8/9）** |
| `qk_rope_head_dim` | 64 | **rope 维 → rope 池宽（1/9）**；RoPE 旋转后吸收不进 latent，必须逐 token 存 |
| `q_lora_rank` | null | Q 不低秩压缩（V2 全量版才有），只影响 Q 投影路径 |
| `qk_nope_head_dim` / `v_head_dim` | 128 / 128 | K nope / V 逐头维——**decode 由 latent 经 `W_UK/W_UV` 即时展开，不占 cache** |
| `num_attention_heads` / `num_hidden_layers` | 16 / 27 | TP4 后每卡 4 头；27 层**全 MLA**（`first_k_dense_replace=1` 只换首层 FFN 非 MoE，与注意力无关） |
| MoE | 64 routed + 2 shared / 每 tok 6 / intermediate 1408 | FFN 侧并行；64+2 专家随 TP4 分片（权重 7.45 GB/卡的来源） |
| `max_position_embeddings` / yarn | 163840 / factor 40（orig 4096） | 部署用 `--max-model-len 32768` |
| `torch_dtype` / vocab | bfloat16 / 102400 | KV dtype 跟随 bf16 |

按类型篇的分类：27 层 `MLAAttention` 均产出 **`AscendMLAAttentionSpec`**（`num_kv_heads=1, head_size=576, block_size=128`，dense 路径 :4987-4999 转换）→ 同构 merge → **单 KV cache group，走 `UnitaryKVCacheCoordinator`**（对照 GDN 混合路线的 `HybridKVCacheCoordinator` 多组）。

**TP4 切分推演**（进程名 + npu-smi 实测印证）：TP 按 Q 头切 16 → 每 rank 4 头；**MLA 的 latent 是全头共享、不在头维切分**——每 rank 仍持有全部 27 层、`num_kv_heads=1` 不变，切分只作用于 Q 头与权重矩阵（MoE 专家分片）。4 个 worker：`W0..W3 = TP0..TP3`（进程名 `VLLMWorker_TP`、无 PP 后缀）。

---

## 2. 部署全流程

### 2.1 Step 1：创建容器（itask create）

```bash
itask create --name gggtest2 \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_97542_deepseek-v2-lite:36500041_20260318110950 \
--4card \
-t a3
```

- `--model` 让平台在 pod 内自动拉权重到**本地盘**（非共享 NFS）：`/home/admin/model-csi/models/modelhub_97542_deepseek-v2-lite-36500041_20260318110950/model`（`ls` 验证：4 个 safetensors 分片 + config.json + tokenizer 均就绪）。
- 保持默认 **Pod 网络**（勿用 `--hostnet`，会与同物理节点其它 pod 冲突 IP）。
- 创建即输出：PVC `itask-boot-disk-gggtest2-wsl02075301-20261018000905`（200Gi，alilocal-ssd）与 Headless Service `itask-svc-…`（支持全端口访问）。

**验证**（`itask list`）：`NODE-STATUS / TASK-STATUS` 双 `Running` 即可进入下一步（Pending 是镜像/权重拉取中，等 1~2 分钟）。

### 2.2 Step 2：环境健康检查

```bash
itask exec gggtest2 -- bash -c 'npu-smi info'          # Health=OK、无残留进程
itask exec gggtest2 -- bash -c 'ls /home/admin/model-csi/models/'   # 权重就绪
itask exec gggtest2 -- bash -c 'pip show vllm-ascend'  # 0.23.0 插件版对齐镜像
```

本次实测：NPU3/NPU4 各 die `Health=OK`、无残留 vllm 进程（首次拉起可跳过 kill_all 清理；复用旧 pod 时必须先杀净残进程并 `npu-smi` 验证）。

### 2.3 Step 3：启动服务（setsid 后台拉起）

```bash
itask exec gggtest2 -- bash -c 'mkdir -p /a3_inference/itask/workdir/wsl02075301/dsv2lite_tp4 && cd /a3_inference/itask/workdir/wsl02075301/dsv2lite_tp4 && setsid nohup vllm serve /home/admin/model-csi/models/modelhub_97542_deepseek-v2-lite-36500041_20260318110950/model --tensor-parallel-size 4 --served-model-name deepseek-v2-lite --port 8100 --max-model-len 32768 > vllm_dsv2lite_tp4_1018.log 2>&1 < /dev/null & echo LAUNCHED; sleep 5'
```

要点：

- **`setsid nohup … & sleep 5`** 缺一不可——`itask exec` 断连会杀普通后台进程，`sleep 5` 保证 setsid 脱离完成；
- 日志重定向用**全新文件名**（NFS 覆盖旧句柄会静默失败）；
- 单机 4 rank 由 vllm 自行 spawn（`multiproc_executor`，HCCL 走 `127.0.0.1`），无需手动配 master。
- 与 GDN 篇的唯一参数差：单 `--tensor-parallel-size 4`、无 `--pipeline-parallel-size`——MLA latent 全头共享不切、PP 切层无收益，走 TP 对齐全纯 FA 主线（Llama-3-8B）部署形态对照。

### 2.4 Step 4：监控启动（marker 驱动）

启动 10 秒后开始 `tail` 日志，按 marker 判断阶段（MLA 主线 marker 与纯 FA 同套；完整时间线见 [`./logs/startup_markers_1018.txt`](./logs/startup_markers_1018.txt)、日志头 [`./logs/startup_head_1018.txt`](./logs/startup_head_1018.txt)）。本次完整时间线（pod UTC；五阶段对照解析见启动篇 §1/§2.0）：

| 时刻（pod UTC） | marker | 对应阶段 |
|---|---|---|
| 16:13:07 | `Platform plugin ascend is activated`；`Breakable cudagraph is force disabled … DeepSeek V4 PIECEWISE not supported`（免责行） | 插件激活 |
| 16:13:15 | vllm 0.23.0 横幅；`non-default args: TP=4, port=8100, max_model_len=32768` | — |
| 16:13:26 | `Resolved architecture: DeepseekV2ForCausalLM`（**无 `is_hybrid`/无 mamba 页对齐 patch 行**） | 架构解析 |
| 16:13:38 | EngineCore init：`tensor_parallel_size=4`、**`enable_prefix_caching=True, enable_chunked_prefill=True`**、`splitting_ops` 含 **`vllm::unified_mla_attention_with_output / unified_mla_kv_cache_update / mla_forward**`（另共存 DSA 系算子） | 引擎 config |
| 16:14:00~08 | 4 worker HCCL init：`world_size=4 rank=0..3, backend=hccl, tcp://127.0.0.1:48609` | worker 拉起 |
| 16:14:08~14 | `Starting to load model` ×4 → **`Loading model weights took 7.4549 GB`** ×4（四卡一致） | 权重加载 |
| 16:14:24 | **`Available KV cache memory: 48.25 GiB`**（worker.py:593） | **② 测预算** |
| 16:14:24 | **`GPU KV cache size: 1,665,664 tokens`**（kv_cache_utils.py:1744） | **③ 做编排** |
| 16:14:42 | `Free memory on device (60.87~61.11/61.27 GiB) … 7.45 weights + 0.15 peak act + 0.51 non-torch + 0.57~0.58 NPU graph … --kv-cache-memory=51040165068 (47.53 GiB) / 55875203072 (52.04 GiB)` | **4d 图内存事后记账** |
| 16:14:42 | `cpu_bind_mode global_slice`：NPU0..3 → NUMA[0]/[2]/[4]/[6] | 绑核 |
| 16:14:49 | `init engine (profile, create kv cache, warmup model) took 35.27 s (compilation: 7.26 s)` | 引擎就绪 |
| 16:14:50 | `Asynchronous scheduling is enabled`；`Dynamic EPLB is False`、`redundant experts 0`；`fusions: norm_quant, act_quant`；`No quantization signature → float` | ⑤ 逻辑层 |
| 16:14:51 | `Application startup complete` | **服务就绪** |

总启动耗时 ≈ **96 秒**（16:13:15 横幅 → 16:14:51 ready；EngineCore 段 71 s，其中 init engine 35.27 s / compilation 7.26 s）——对照 GDN 篇的 ≈2.5 分钟：**MLA 无多模态 warmup、无 mamba 页对齐 patch**，启动管线与纯 FA 最接近。

### 2.5 Step 5：冒烟测试

```bash
itask exec gggtest2 -- bash -c "curl -s http://localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"deepseek-v2-lite\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，请用一句话介绍你自己\"}],\"max_tokens\":64}'"
```

**返回**（完整 JSON 归档：[`./logs/smoke_test_1018.json`](./logs/smoke_test_1018.json)）：HTTP 200，**`system_fingerprint: vllm-0.23.0-tp4-44ad530f`——TP4 切分生效的直接证据**；`usage: prompt_tokens=15, completion_tokens=64`，`finish_reason=length`（max_tokens 截满属预期）；中文回复非空。启动早期打过一次 `generation_config.json` override 警告（`temperature 0.3 / top_p 0.95`，deepseek 出厂默认采样，与 vllm 默认采样不同时注意）。

### 2.6 Step 6：运行期验证（前缀缓存 & 运行时态实测）

服务就绪后跑运行篇 §2 的实验（脚本归档：[`./logs/runtime_test_scripts_1018.py`](./logs/runtime_test_scripts_1018.py)）：同 prompt（970 token）三连发 + 长 decode 请求（975 + 512 token），采集三层证据：

- `/metrics`：`vllm:prefix_cache_queries_total=2925 / hits_total=1792`（token 账 = 970×3+15 与 896×2， Prometheus counter 与块叙事互相印证）
- stats 日志：`Prefix cache hit rate: 61.3% → 68.9%`、`Running: 1 reqs` 时 `GPU KV cache usage: 0.1%`（= cdiv(1487,128)=12 块 / 13,013，一字不差）
- **三个数字全部手算复现**——MLA 前缀缓存可用 + 容量账实的运维级证据，详证与解读见 [`2_runtime_sequence.md`](./2_runtime_sequence.md) §4/§5。

---

## 3. 启动日志 × 三侧五阶段对照（本篇核心）

把启动篇 §1 的五阶段管线与本实例日志逐段对上（单机 TP4，无 PD 分离、无 KV 传输组 → 无 2MiB 对齐/`register_kv_caches` 环节；单 spec 族 → 无 mamba 页对齐 patch、无`is_hybrid` 分岔）：

| 阶段 | 代码入口（系列文档行号） | 本实例日志/行为证据 |
|---|---|---|
| 前置插件 | platform.py 激活 | `Platform plugin ascend is activated`；**`Breakable cudagraph force disabled … V4 PIECEWISE`（免责行，每台 DeepSeek 系模型必现，与本模型无关——MLA 走 FULL_AND_PIECEWISE 图照常）**；yarn `rope_parameters' factor … got 40` 序列化告警（无功能影响） |
| ① 算规格 | worker.py:891 → model_runner_v1.py:4948-5000 | 27 层 `MLAAttention` 每层产出 `AscendMLAAttentionSpec`（dense 路径 :4987-4999：block_size=128, num_kv_heads=1, head_size=576, dtype=bf16）。**TP4 切 Q 头不动 latent → 4 rank 同层 spec 全等**（合并断言天然通过）→ 同构 merge → **单 KV cache group**。侧证：splitting_ops 含 `unified_mla_*` 三算子 |
| ② 测预算 | worker.py:527 → :593 | `Available KV cache memory: 48.25 GiB`；４d 事后分解：weights 7.45 + peak act 0.15 + non-torch 0.51 + graph 0.58 ≈ 8.69 GiB（61.27×0.92 − 8.69 ≈ 47.5 → profile 实测 48.25，公式无图项取整差在图项事后记账核销）；profile 前真实跑 `eplb_warmup` + MC2 容量试跑（MoE 模型新增预检，`Dynamic EPLB is False`） |
| ③ 做编排 | kv_cache_utils.py:1956 | `GPU KV cache size: 1,665,664 tokens`；**推演**（启动篇 §2.0 数字链）：`num_blocks = 51,808,813,056 B ÷ (27 × 147,456) = 13,013 块`，`13,013 × 128 = 1,665,664` ✓ 与日志**一字不差**；四卡同 spec → min 对齐恒等值 |
| ④ 落张量 | model_runner_v1.py:4605-4629 | MLA 双池：**k_cache=(13013,128,1,512) nope/latent、v_cache=(13013,128,1,64) rope**（源码注释 `k_cache: nope_cache    v_cache: rope_cache`），整层 1.79 GiB 按 `calc_split_factor([512,64])=(1.125,9.0)` **8:1 拆**两条独立 int8 字节池 → `view(bf16).view(4D)` 无 permute；block_id == 双池各自行号 |
| 4d 图捕获 | worker.py:706-776 | `0.57~0.58 GiB for NPU graph memory` + 两条 `--kv-cache-memory` 建议（47.53 / 52.04 GiB）——机制与 FA/GDN 两路线完全一致 |
| ⑤ 建逻辑层 | core.py:145-157 | `enable_prefix_caching=True`（**MLA 支持前缀缓存——与 GDN 自动关闭的关键分野，运行期实测命中 61.3%→68.9%**）、`Asynchronous scheduling is enabled`；单 group → `UnitaryKVCacheCoordinator` 直通 + `BlockPool(13013)` + 1 个 `FullAttentionManager`（注册表 :305-315 为 MLA spec 指到它）——第 2~5 层与纯 FA 逐字同 |

**佐证进程布局**（`npu-smi info` 全文：[`./logs/npu_smi_after_serve_1018.txt`](./logs/npu_smi_after_serve_1018.txt)）：4 个 `VLLMWorker_TP` 进程（PID 1990-1993）分别持有 NPU3 die0/1 + NPU4 die0/1；**TP4 全尺寸双池**（MLA 不切 latent，无"每卡头子集"现象——对照 Llama pp2tp2 的"同层不同头子集"）；HBM 61,498~61,744 / 65,536 MB（含 KV 48.25 GiB），进程内存 ≈58.66 GB/卡；服务空载时 AICore 0%。

---

## 4. MLA 特有行为（部署期实测到的 6 个点）

1. **KV 容量数字即 MLA 压缩率的最直接证据**：48.25 GiB ÷ 27 层 ÷ (576×2B) = 1,665,664 tokens（13,013 块）。对照"等价 MHA 直存"（16 头 × 320 元素 → 每 token 10,240 B/层）仅能容纳 ~187K tokens——同预算**容量放大 8.9 倍**（启动篇 §2.0 完整算式）。
2. **K/V 维宽 8:1 的"反直觉"拆分静默完成**：`_get_attention_kv_cache_dims`（:3987-3989，注释 `DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim`）→ **"V 池"装的不是 value 而是 k_rope**（value 在 decode 由 latent 经 `W_UV` 即时上采样，mla_v1.py:898-906）。日志侧不可见（默默完成），源码 + 容量数字闭环佐证（类型篇中篇）。
3. **前缀缓存开箱即用**：`enable_prefix_caching=True` 默认生效。latent/rope 逐 token 因果 → 链式哈希照常成立；实测三连发 + 长请求命中 61.3%→68.9%（token 账 2,925/1,792 → 3,900/2,688）——对照 Qwen3.5 GDN 篇的 `enable_prefix_caching=False` 自动关闭，两条扩展路线在逻辑侧的第一分野点（运行篇 §4）。
4. **MLA 专属编译图切分算子指纹**：`compilation_config.splitting_ops` 实测含 `vllm::unified_mla_attention_with_output`、`vllm::unified_mla_kv_cache_update`（写 KV）、`vllm::mla_forward`（吸收式 decode）；同列表共存 DSA 路线算子（`dsa_forward / deepseek_v4_attention / sparse_attn_indexer`）——**判断"跑的是 dense MLA 还是 sparse DSA 路线"看日志这一行**。
5. **DeepSeek 系启动即出现的两条已知噪声**（排障先排除）：`Breakable cudagraph is force disabled … DeepSeek V4 PIECEWISE cudagraph is not supported yet`（平台侧对 V4 特性预防性禁用，不作用于本模型的 FULL_AND_PIECEWISE 图）；`rope_parameters' factor field must be a float >= 1, got 40` ×多次（yarn factor 40 序列化告警，无功能影响）。
6. **MoE 侧平行状态**：`Dynamic EPLB is False`、`The number of redundant experts is 0`（专家不扩块账本——num_blocks 公式不因专家数变化）；`No quantization signature detected → float`（bf16 原精度，KV dtype auto→bf16）；cpu_binding global_slice 把 4 worker 各钉 NUMA 0/2/4/6（每 worker 主核 76 + acl 核 1）。

---

## 5. 与 Llama-3-8B / Qwen3.5-4B 主线实测对照（同规格 A3 4 卡、同镜像 0.23.0）

| 量 | Llama-3-8B 纯 FA（2_start_init §6） | Qwen3.5-4B GDN 混合（GDN 篇） | **DeepSeek-V2-Lite MLA（本篇）** |
|---|---|---|---|
| 每卡 KV 可用显存 | 51.96 GiB | 51.90 GiB | **48.25 GiB**（权重 7.45 GB + MoE 激活更贵） |
| block_size | 128 | 1024（mamba 页对齐 patch） | **128**（与纯 FA 一致，无级跳） |
| 每层 KV 物理 | K/V 等宽双 4D（128/128 对半） | attn 1024 大页 + GDN 状态张量并存 | **latent(512)/rope(64) 双 4D，8:1 拆** |
| KV 容量 | 13,291 块 / ~1.7M tokens | 6,147,276 tokens（~6003 大页） | **13,013 块 / 1,665,664 tokens** |
| 每 token 全模型 KV | 131,072 B（128 KiB） | —（跨类型不可比） | **31,104 B**（27 层 × 576 × 2B） |
| 每 token KV 压缩率 | 基线 | — | **×0.24（≈8.89× 放大）vs 等价 MHA** |
| 每 worker 权重 | ~8 GB（8B/TP4） | 2.70 GB（4B/PP2TP2 均摊） | **7.4549 GB**（15.7B/TP4：MoE 分片） |
| KV group | 1（Unitary） | 多（Hybrid 协调器） | **1（Unitary，与主线同形）** |
| 前缀缓存 | 可开 | 自动关（`enable_prefix_caching=False`） | **默认开 + 实测命中 61.3%→68.9%** |
| NPUGraph 事后建议 | 有 | 51.19/55.69 GiB | **47.53/52.04 GiB**（0.57~0.58 GiB 图内存，同机制） |
| 总启动耗时 | ~3 分钟 | ≈2.5 分钟 | **≈96 秒**（无多模态/无 mamba patch） |
| 冒烟 fingerprint | tp4 | tp2-pp2 | **tp4-44ad530f** |

三列之差异即三条路线的代价结构：纯 FA 吃层数与 KV 头数、GDN 吃状态页对齐与多组协调、**MLA 把显存压力从 KV 侧挪到权重/激活侧**（16B 级 MoE 权重 7.45 GB/卡换 8.89× KV 容量）。

---

## 6. 常用运维命令（本次实测汇总）

```bash
itask list                                  # pod 状态（Running/Healthy）
itask exec gggtest2                         # 交互进入容器
itask describe gggtest2                     # 物理节点 IP / 详情

# 服务健康与推理
itask exec gggtest2 -- bash -c 'curl -s http://localhost:8100/health'
itask exec gggtest2 -- bash -c 'curl -s http://localhost:8100/v1/models'
# 日志（NFS 路径，pod 内）
itask exec gggtest2 -- bash -c 'tail -50 /a3_inference/itask/workdir/wsl02075301/dsv2lite_tp4/vllm_dsv2lite_tp4_1018.log'
# 运行期证据（前缀命中 / KV 用量 / 吞吐）
itask exec gggtest2 -- bash -c 'curl -s http://localhost:8100/metrics | grep -iE "prefix_cache_(queries|hits)_total"'

# 重启服务（先杀干净再拉起；pkill 用括号防自匹配）
itask exec gggtest2 -- bash -c "pkill -f 'd[e]epseek-v2-lite' ; pkill -9 -f 'v[l]lm serve' ; sleep 3"
itask exec gggtest2 -- bash -c "pgrep -f 'v[l]lm serve' || echo CLEAN"

# 停止/删除容器（delete 会确认，name-only，拒绝 --user 组合）
itask stop gggtest2        # 保留 PVC/boot-disk
itask start gggtest2       # 带盘重启
yes | itask delete gggtest2
```

**注意事项**：

- **NPU 保活**：服务空载（本篇实测 AICore 0%）时平台可能因"利用率过低"回收 pod，源码/日志随 pod 销毁丢失。长时间调试请跑 NPU 保活脚本（占 AICore ~50%，HBM 仅 ~16MB/卡，可与 vllm 共存；**压测前必须停掉**）。
- **首次 `itask exec` 偶发"正在连接"挂起**，重试即可；复杂管道命令（`ps | grep` 级）经 exec 会静默失败，拆简单命令或落文件再 `cat`（本系列 marker/stats 采集均单命令 grep 模式）。
- **复现换模型**：替换 `--model`（权重路径同 pattern）与 `--served-model-name`；DSA 类（V3.2/V4）额外关注 `sparse_head_dim` 与 kv_cache[2]/[3] indexer 张量（未实测，见类型篇尾注）。

---

## 7. 小结

1. **部署结论**：DeepSeek-V2-Lite 在 A3 单机 4 卡 TP4 下经 vllm 0.23.0 + vllm-ascend 0.23.0**一次拉起成功**，启动 ≈96 秒，冒烟推理通过（`tp4-44ad530f` fingerprint 印证切分生效），中文回复内容正常；运行期前缀缓存与容量账全部闭环。
2. **系列结论被逐条实证**：MLA → `AscendMLAAttentionSpec`（① :4987-4999 冻结四字段）✓；物理张量 **latent/rope 双 4D 8:1**（④ :4605-4629，"V 池"装 k_rope 非 value）✓；单 KV group 走主线逻辑层 + `FullAttentionManager` ✓；**前缀缓存默认开且实测命中**（61.3%→68.9%，与 GDN 强关相对）✓；②测预算无图项 + 4d 事后记账并给 `--kv-cache-memory` 建议（47.53/52.04 GiB）✓。
3. **数字闭环（全系列的实证核心）**：`48.25 GiB ÷ 27 层 ÷ 147,456 B/层/块 = 13,013 块`，`×128 = 1,665,664 tokens` 与日志一字不差；运行期再添两组——`0.1% = 12 块/13,013`（long 请求峰值）与 `68.9% = 2,688/3,900`（累计命中）。**这是三次实测（FA 51.96 GiB / GDN 51.90 GiB / MLA 48.25 GiB）中 spec→运维数字链路最短的一次**：无页对齐 patch、无多组协调、图项不进②公式——MLA 的容量可纯手算复现。
4. **复现路径**：第 2 章六步命令从空 pod 到冒烟（+运行期实验）顺序可复跑；原始证据 9 件归档于 [`./logs/`](./logs/)（startup_markers / startup_head / smoke_test / npu_smi / config.json / runtime 前缀实验与解读 / metrics / stats / 实验脚本）。换模型复现对照见 §6 注意事项。
