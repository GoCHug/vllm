# Qwen3.5-4B（GDN 混合模型）单机 PP2×TP2 部署实测记录（NPU / vllm-ascend）

> 本篇是 KV Cache 学习系列的**实战篇**：在昇腾 A3 4 卡 pod 上以 **pp2tp2** 部署 **Qwen3.5-4B** 并冒烟验证，记录全流程与关键实测数字，并把每一步启动日志证据回挂到系列文档的三侧五阶段模型上。
>
> 与系列前几篇的关系：前四篇主线是**纯 Full Attention**（Llama-3-8B，单 KV cache group）；Qwen3.5-4B 正是 [`0_kvcache_arch.md`](./0_kvcache_arch.md) §6 扩展表与 [`1_start_init.md`](./1_start_init.md) §1 示例表中的 **GDN（Gated DeltaNet，线性注意力混合模型）代表**——本篇等于给"混合模型（Full + Mamba/GDN）多 KV group"这条扩展路线补上一次真实部署走查。混合相关的架构结论直接引用 [`1_kvcache_type.md`](../../kvcache_docs_v2/1_kvcache_type.md) §4.3 与 [`1_start_init.md`](./1_start_init.md) 扩展节，不再复述代码细节。

**实测环境一览**：

| 项 | 值 |
|---|---|
| 部署日期 | 2026-10-16 |
| 容器 | itask pod `gggtest3`（a3，4 × hpu910a3，32C / 400GB / boot-disk 200Gi 挂 `/data`） |
| 镜像 | `hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| vllm / vllm-ascend | 0.23.0 / 0.23.0（`pip show vllm-ascend` 确认） |
| 模型 | `modelhub_41700011_qwen3.5-4b:112600047_20260303111458`（创建 pod 时 `--model` 自动拉取） |
| 并行策略 | PP2 × TP2 = 1 pod × 4 NPU（单机 4 rank，HCCL `tcp://127.0.0.1`） |
| 服务 | OpenAI API，端口 8100；`--max-model-len 32768`（原生 262144，为缩短 profile/校验时间调小） |

---

## 1. 模型架构速览（config.json 实测解析）

架构 `Qwen3_5ForConditionalGeneration`（带 vision_config / image_token_id，多模态模型，但本次按纯文本路径冒烟）。`text_config` 关键参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `num_hidden_layers` | 32 | `layer_types` 显示 **24 × linear_attention + 8 × full_attention**，`full_attention_interval=4`（每 3 层 GDN 插 1 层全注意力） |
| full attn：`num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 4 / 256 | GQA 4 KV 头 |
| GDN：`linear_num_key/value_heads` / `head_dim` / `conv_kernel` | 16 / 32 / 128 / 128 / 4 | 门控 DeltaNet 递归状态，**块内存状态而非 token K/V** |
| `hidden_size` / `intermediate_size` / vocab | 2560 / 9216 / 248320 | 4B 级小模型 |
| `max_position_embeddings` | 262144 | 部署用 `--max-model-len 32768` |
| `mtp_num_hidden_layers` | 1 | 解码端含 1 层 MTP（本次未启投机解码） |
| `mamba_ssm_dtype` | float32 | GDN 状态精度 |

按 [`1_kvcache_type.md`](../../kvcache_docs_v2/1_kvcache_type.md) 上篇 §1.5 的分类：full_attention 层产出上游 `FullAttentionSpec`（NPU 上 K/V 双池）、linear_attention 层产出 `MambaSpec`（NPU 上单张量路径，`model_runner_v1.py:4118` 命中 `linear_attn` 层名分支）→ 两种 spec 无法 merge → **多 KV cache group，走 `HybridKVCacheCoordinator`**（对照 [`1_start_init.md`](./1_start_init.md) §4.4 工厂三分支）。

**pp2tp2 切分推演**（实测进程名印证）：PP2 按层切 32→每 stage 16 层（各含 12 linear + 4 full attn）；TP2 按 KV 头切 full attn 4→每 rank 2 头。4 个 worker：`W0=PP0/TP0，W1=PP0/TP1，W2=PP1/TP0，W3=PP1/TP1`。

---

## 2. 部署全流程

### 2.1 Step 1：创建容器（itask create）

```bash
itask create --name gggtest3 \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_41700011_qwen3.5-4b:112600047_20260303111458 \
--4card \
-t a3
```

- `--model` 让平台在 pod 内自动拉权重到**本地盘**（非共享 NFS）：`/home/admin/model-csi/models/<model_id>/model`（`ls` 验证含 2 个 safetensors 分片、config.json、tokenizer 均就绪）。
- 保持默认 **Pod 网络**（勿用 `--hostnet`，会与同物理节点其它 pod 冲突 IP）。
- 创建即输出：PVC `itask-boot-disk-gggtest3-…-20261016145802`（200Gi，alilocal-ssd）与 Headless Service `itask-svc-…`（支持全端口访问）。

**验证**（`itask list --user <user>`）：`NODE-STATUS / TASK-STATUS` 双 `Running` 即可进入下一步（Pending 是镜像/权重拉取中，等 1~2 分钟）。

### 2.2 Step 2：环境健康检查

```bash
itask exec gggtest3 -- bash -c 'npu-smi info'   # Health=OK、无残留进程
itask exec gggtest3 -- bash -c 'ls /home/admin/model-csi/models/'   # 权重就绪
```

本次实测：4 die 全 OK、无残留 vllm 进程（首次拉起可跳过 kill_all 清理；复用旧 pod 时必须先 `kill_all.sh` 2~3 次并 `npu-smi` 验证无残留）。NPU 0/1 偶有 `Alarm` 属良性。

### 2.3 Step 3：启动服务（setsid 后台拉起）

```bash
itask exec gggtest3 -- bash -c 'mkdir -p /a3_inference/itask/workdir/<user>/qwen35_pp2tp2 && cd /a3_inference/itask/workdir/<user>/qwen35_pp2tp2 && setsid nohup vllm serve /home/admin/model-csi/models/modelhub_41700011_qwen3-5-4b-112600047_20260303111458/model --pipeline-parallel-size 2 --tensor-parallel-size 2 --served-model-name qwen3.5-4b --port 8100 --max-model-len 32768 > vllm_qwen35_pp2tp2_1016.log 2>&1 < /dev/null & echo LAUNCHED; sleep 5'
```

要点：

- **`setsid nohup … & sleep 5`** 缺一不可——`itask exec` 断连会杀普通后台进程，`sleep 5` 保证 setsid 脱离完成；
- 日志重定向用**全新文件名**（NFS 覆盖旧句柄会静默失败）；
- 单机 4 rank 由 vllm 自行 spawn（`multiproc_executor`，HCCL 走 `127.0.0.1`），无需手动配 master。

### 2.4 Step 4：监控启动（marker 驱动）

启动 10 秒后开始 `tail` 日志，按 marker 判断阶段（Full Attention 主线 marker 见 [`1_start_init.md`](./1_start_init.md)；rank1+ headless 无 `Application startup complete` 属正常——单机部署 rank0 日志即总日志）。本次完整时间线：

| 时刻（pod UTC） | marker | 对应五阶段 |
|---|---|---|
| 07:52:45 | API server 拉起，ascend 插件激活 | — |
| 07:53:11 | `Resolved architecture: Qwen3_5ForConditionalGeneration`；**`Setting attention block size to 1024 tokens … >= mamba page size`；`Padding mamba page size by 1.16%`** | 前置（mamba 页对齐 patch） |
| 07:53:32 | EngineCore：`world_size=4, local_world_size=4`，V1 引擎 config（`enable_prefix_caching=False, enable_chunked_prefill=True`） | — |
| 07:53:56 | 4 个 worker HCCL init：`rank=0..3, backend=hccl, tcp://127.0.0.1:60243` | worker 拉起 |
| 07:54:04 | `Starting to load model`；`Using Triton/FLA GDN prefill kernel (head_k_dim=128)` | — |
| 07:54:05 | `Loading model weights took 2.6957 GB`（×4 worker 一致） | — |
| 07:54:27 | `Available KV cache memory: 51.90 GiB`（worker.py:593 info_once） | **② 测预算** |
| 07:54:29 | `GPU KV cache size: 6,147,276 tokens`（kv_cache_utils.py:1744） | **③ 做编排** |
| 07:54:49 | `Free memory on device (60.87/61.27 GiB) … 0.57 GiB for NPU graph memory … --kv-cache-memory=54964745420` 建议 | **4d 图内存事后记账** |
| 07:55:16 | 多模态 warmup 完成（12.56s） | — |
| 07:55:17 | `Application startup complete` | **就绪** |

总启动耗时 ≈ 2.5 分钟（架构解析 ~25s、worker 拉起+权重加载 ~30s、profile+编排 ~25s、图捕获+warmup ~25s）。

### 2.5 Step 5：冒烟测试

```bash
itask exec gggtest3 -- bash -c "curl -s http://localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，请用一句话介绍你自己\"}],\"max_tokens\":48}'"
```

**返回**（截选）：HTTP 200，`system_fingerprint: vllm-0.23.0-tp2-pp2-789ebada`——**并行策略生效的直接证据**；`usage: prompt_tokens=16, completion_tokens=48`；`finish_reason=length`（模型为 thinking 风格输出，48 tokens 未收尾属正常）。响应 body 含 `choices[0].message.content` 非空中文推理 → 冒烟通过。

---

## 3. 启动日志 × 三侧五阶段对照（本篇核心）

把 [`1_start_init.md`](./1_start_init.md) §1 的"①算规格→②测预算→③做编排→④落张量→⑤建逻辑层"管线与本实例日志逐段对上（单机 pp2tp2，无 PD 分离、无 KV 传输组 → 无 2MiB 对齐/`register_kv_caches` 环节）：

| 阶段 | 代码入口（系列文档行号） | 本实例日志/行为证据 |
|---|---|---|
| 前置 patch | `NPUModelRunner.get_kv_cache_spec`（model_runner_v1.py:5027-5036：attn 层 `page_size_padded` 对齐 mamba 页） | `patch_mamba_config.py:104/119` 两行日志即此 patch 的 config 侧前置：**attention block size 128→1024**，mamba 页 padding 1.16% 后两类页字节数严格相等 |
| ① 算规格 | worker.py:891 → model_runner_v1.py:4909 | full_attention 层每 rank 产出 `FullAttentionSpec`（4 KV 头÷TP2=2 头，block_size=1024 对齐后）；linear_attention 层产出 `MambaSpec`（GDN 状态页）。**两类 spec merge 失败 → 多 group**（`get_kv_cache_groups` 走混合分组，非主线单 group） |
| ② 测预算 | worker.py:527 profile_run（公式无图项，worker.py:593） | `Available KV cache memory: 51.90 GiB` ——与 Llama-3-8B 实测的 51.96 GiB 几乎一致（同规格 die、同 0.92 利用率，差异来自权重 2.7 vs ~8 GB 被同等放大的激活抵消） |
| ③ 做编排 | kv_cache_utils.py:1956 | `GPU KV cache size: 6,147,276 tokens`（≈6003 个 1024-token 块）。粗算：attn 组每 rank 4 层 × (2×1024×2×256×2B=2MB/块/层)=8MB/块，纯 attn 可容纳 51.9GiB/8MB≈6912 块；实际 6003 块是 **attn 组与 mamba 状态组在统一 block 空间下协调（分组混合布局 + min 对齐）后的结果**——mamba 状态页同样从 51.9GiB 预算扣减，这是混合模型区别于纯 FA 主线（13291 块全归 KV）的关键现象 |
| ④ 落张量 | model_runner_v1.py:4082/4118 | 无 KV 传输 → dense full-attn 层 K/V 双池普通 `torch.zeros`（无 2MiB 对齐）；**linear_attn 层按层名命中单张量路径**（`"linear_attn" in layer_name`，:4118-4134）——同一模型里两种物理形态并存（K/V 双 4D 池 + 状态扁平缓冲），均恒 dim0=block 行号 |
| 4d 图捕获 | worker.py:706-776 NPUGraph | `0.57 GiB for NPU graph memory`；随后打印两种 `--kv-cache-memory` 建议值（51.19 GiB 贴 requested / 55.69 GiB 贴整机 free，含 150MiB 余量）——事后记账机制的实证 |
| ⑤ 建逻辑层 | core.py:145-157（平台无关） | 引擎配置里 `enable_prefix_caching=False`（见 §4），逻辑侧走 `HybridKVCacheCoordinator` 多组分支 + 每组一个 manager、全组共享唯一 BlockPool；`Asynchronous scheduling is enabled` |

**佐证进程布局**（`npu-smi info`）：4 个 `VLLMWorker_PP0_TP0 / PP0_TP1 / PP1_TP0 / PP1_TP1` 进程（PID 20829-20832）分别持有 NPU1-2 各 die，进程内存 ≈57.8 GB / die，HBM 60.6-60.9 / 64 GB（含 KV cache 51.9 GiB）。

---

## 4. 混合模型特有行为（部署期实测到的 5 个点）

1. **mamba 页对齐 patch 实测生效**：`Setting attention block size to 1024 tokens…` + `Padding mamba page size by 1.16%…`。这正是 [`0_kvcache_arch.md`](./0_kvcache_arch.md) §6 混合模型行的"NPU 侧①算规格时把 attn 层 page_size_padded 对齐到 mamba 页"在 config 阶段的日志投影：attn 块被放大到与 mamba 状态页**字节严格相等**，两类页才能共享同一套 block 账本（`BlockPool` 唯一、`block_table` 按组独立）。对照纯 FA 主线（block_size=128）——**Qwen3.5-4B 的 attn KV 块是 1024-token 大页**。1024 的来源公式（`patch_mamba_config.py:94`）：`attn_block_size = 128 × cdiv(ssm_block_page_size, 128 × attn_single_token_k_page_size)`——128 是 kernel 分档基值，按 GDN ssm 状态页字节数（TP2 后 `linear_num_value_heads×head_dim²` 状态矩阵）反推出 8 倍放大。
2. **前缀缓存自动关闭**：引擎 config 中 `enable_prefix_caching=False`（默认本应开启）。GDN 递归状态无"逐 token 增量"语义，链式哈希前缀复用不成立；调度链上 `get_computed_blocks` 恒空，协调器退化为只走分配/释放。
3. **两类物理张量形态并存**（对照 [`1_kvcache_type.md`](../../kvcache_docs_v2/1_kvcache_type.md) 中篇 §5）：full-attn 层 K/V 双 int8 池 → `view` 成 4D（本例 `(num_blocks, 1024, 2, 256)` bf16）；linear-attn 层单张量路径 → 按 `MambaSpec.shapes` 切成 conv/temporal 状态视图。`block_id == dim0 行号` 契约在两种形态上同时成立。
4. **GDN 计算路径**：`Using Triton/FLA GDN prefill kernel (head_k_dim=128)`；编译 splitting_ops 里可见 `vllm::qwen_gdn_attention_core / mamba_mixer / linear_attention` 等混合算子（`compilation_config.splitting_ops`）。异步调度（async scheduling）开启。
5. **block size 双粒度 + attention 后端分派**（**16 → 128 → 1024 三级跳全链时序、页统一数字推导与 `verify_bs.py` 实测四联证见 [`1_start_init.md`](./1_start_init.md) §2**）：本次 full-attn 组默认走 `AscendAttentionBackend`（dense 主线，kernel 分档 `[128]`），linear-attn 组走 `AscendGDNAttentionBackend`（状态后端）；混合模型跳过平台层 `refresh_block_size`（`is_hybrid` 分支，utils.py:1397），`cache_config.block_size` 由 mamba 对齐层定为 1024，但 kernel 层 `select_common_block_size(kv_manager_block_size, backends)`（model_runner_v1.py:4759）仍在后端支持的 **128 粒度**上物理寻址——逻辑管理块 1024-token 对齐 mamba 状态页、物理 kernel 128-token 分档访问，两层粒度分离；`MambaSpec` 组 `kernel_block_sizes` 记 `[0]`（不按 token 切分，:4763-4769）。**即 NPU 的 block size 并非"都是 128"**：dense 主线默认 128，DSA 后端多档（2~128），deepseek_v4 推荐 32，混合 GDN 模型 attn 块为 128 的整数倍（本例 1024）。


---

## 5. 与 Llama-3-8B 主线实测对照（同规格 4 卡环境）

| 量 | Llama-3-8B 纯 FA（2_start_init §6） | Qwen3.5-4B GDN 混合（本篇） | 差异原因 |
|---|---|---|---|
| 每卡 KV 可用显存 | 51.96 GiB | 51.90 GiB | 同预算机制（②公式一致） |
| block_size | 128 | **1024**（attn，mamba 页对齐产物） | 混合模型页对齐 patch |
| KV 容量 | 13291 块（≈1.7M tokens） | 6,147,276 tokens（≈6003 个 1024-token 块） | 大页 + mamba 状态共享预算 + 混合编排 |
| 每 worker 权重 | 系列未记（8B/2=8GB 级） | 2.6957 GB | 4B 模型 / PP 切层 |
| 物理形态 | 每层 K/V 双 4D（16 层 × 2 张） | full-attn 双池 + linear-attn 单张量并存 | 混合架构 |
| KV group | 1（Unitary） | 多 group（Hybrid） | spec merge 失败 |
| 前缀缓存 | 可开 | 自动关闭（`enable_prefix_caching=False`） | GDN 状态语义 |
| NPUGraph 事后建议 | 有 | 有（51.19/55.69 GiB，0.57GiB 图内存） | 同一 4d 机制 |

---

## 6. 常用运维命令（本次实测汇总）

```bash
itask list --user <user>                       # pod 状态（Running/Healthy）
itask exec gggtest3                            # 交互进入容器
itask describe gggtest3                        # 物理节点 IP / 详情

# 服务健康与推理
itask exec gggtest3 -- bash -c 'curl -s http://localhost:8100/health'
itask exec gggtest3 -- bash -c 'curl -s http://localhost:8100/v1/models'
# 日志（NFS 路径，pod 内）
tail -f /a3_inference/itask/workdir/<user>/qwen35_pp2tp2/vllm_qwen35_pp2tp2_1016.log

# 重启服务（先杀干净再拉起；pkill 用括号防自匹配）
itask exec gggtest3 -- bash -c "pkill -f 'q[w]en3-5-4b' ; pkill -9 -f 'v[l]lm serve' ; sleep 3"
itask exec gggtest3 -- bash -c "pgrep -f 'v[l]lm serve' || echo CLEAN"

# 停止/删除容器（delete 会确认，name-only，拒绝 --user 组合）
itask stop gggtest3        # 保留 PVC/boot-disk
itask start gggtest3       # 带盘重启
yes | itask delete gggtest3
```

**注意事项**：

- **NPU 保活**：服务空载（AICore 0%）时平台可能因"利用率过低"回收 pod，源码/日志随 pod 销毁丢失。长时间调试请跑 NPU 保活脚本（fp16 matmul 占 AICore ~50%，HBM 仅 ~16MB/卡，可与 vllm 共存；**压测前必须停掉**）。
- 体感提示：混合 GDN 模型的 decode 步中 full-attn 层才有"读历史 KV"路径，linear 层只做状态步进——排查 KV cache 相关问题（块分配、显存账）时先分清层类型再 grep 日志。
- 首次 `itask exec` 偶发"正在连接"挂起，重试即可；复杂管道命令（`ps | grep` 级）经 exec 会静默失败，拆简单命令或落文件再 `cat`。

---

## 7. 小结

1. **部署结论**：Qwen3.5-4B 在 A3 单机 4 卡 pp2tp2 下经 vllm 0.23.0 + vllm-ascend 0.23.0 一次拉起成功，启动 ≈2.5 分钟，冒烟推理通过（`tp2-pp2` fingerprint 印证切分生效）。
2. **系列文档结论被逐条实证**：GDN=Qwen3.5-4B 走 `MambaSpec` 单张量路径 ✓；混合模型多 KV group（Hybrid 协调器）✓；NPU mamba 页对齐 patch（attn block 128→1024、mamba 页 padding 1.16%）✓；前缀缓存自动关闭 ✓；②测预算无图项 + 4d 图内存事后记账并给出 `--kv-cache-memory` 建议 ✓。
3. **混合模型的容量口径变化**：KV 容量从纯 FA 的"块数"视角变为"mamba 状态页与 attn 大页共享同一 block 账本"的协调视角（6.15M tokens / 51.9 GiB），粗算 attn 组理论块数(≈6912)与编排结果(≈6003)之差即 mamba 状态占用——这正是 [`1_kvcache_type.md`](../../kvcache_docs_v2/1_kvcache_type.md) 中篇"两种形态并存、逻辑侧无感"的数字体现。
4. **复现路径**：第 2 章五步命令从空 pod 到冒烟通过顺序可复跑；换模型只需替换 `--model`（权重拉取路径同 pattern）与 `--served-model-name`，纯 FA 模型可额外开 `--enable-prefix-caching` 对比观察组合切换。
