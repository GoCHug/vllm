# Qwen3.5-0.8B（hybrid attn+GDN/mamba）KVCache 端到端观测实验记录

- 实验日期：2026-10-20（整理）；首轮实验记录待补
- 实验平台：gggtest4（A3, hpu910a3，单卡即可起 qwen3.5-0.8B）
- 镜像：`hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64`（同 block-size 实验/兄弟 kvc 工作区）
- 模型：`modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902`（容器内 `/home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/`）
- vllm 版本：0.23.0（容器内 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend`，git 干净打补丁）
- 参照工作区：`kvc/`（llama3-8b 单组 attn 版，08 v2.5 归档）——本实验 01~07 号补丁与其完全一致，08 号为其 hybrid 演进版

---

## 1. 实验目的

1. **混合模型双组 KV cache 的物理观测**：qwen3.5-0.8B 的 full-attention 层（组 0）与 GDN/mamba 层（组 1）**共用同一张物理 raw tensor 页**（页内三视图布局：`tensor1: [(kv_padding), conv, ...]` / `tensor2: [k, ssm, ...]` / `tensor3: [v, (mamba_padding), ...]`，见 `_reshape_kv_cache_tensors` MambaSpec 分支），但**块表各自独立**。把请求结束时刻两组张量分别整块归档，验证管理侧块表与物理侧张量的对应关系。
2. **attn 组前缀复用 铁 证**：P 请求种 2 满块入哈希链，R 请求前缀命中（零拷贝复用），归档张量逐位比对 P 种块 vs R 命中块。
3. **mamba 组 slot 生命周期**：GDN 递推状态不参与哈希前缀复用，每请求独立槽位；观测 slot 分配/归还轨迹（[KVC][L2]/[L3] 管理侧打印）与状态张量本体（conv (3,6144) bf16 + ssm (16,128,128) fp32）。
4. 对照 block-size 实验（本工作区 README 根因分析）：不变量 `K 段宽 ≡ SSM 段宽` 成立时（默认启动，block_size 矫正为 1024）物理页内两族切窗对齐，归档张量应无语义污染。

## 2. 归档设计（08 号补丁 hybrid 版，schema kvt5-hybrid-raw）

- **时点**：每步前向后 `_kvc_rel_snapshot` 追踪（块表按组累积 + TERM 判定 `written >= prompt + max_tokens - 1`），TERM 时 `_kvc_kv_save` 整块快照；
- **层来源**：`self.attn_groups` 展开（组序 <-> `NewRequestData.block_ids` 下标），每层张量取 `forward_context[层名].kv_cache`（bind_kv_cache 绑定）：
  - attn 层 = `(k_cache, v_cache)` 元组 → 每层每块 `K[blk]/V[blk]` 整块（块-行映射，含未写槽位）；
  - mamba 层 = `[conv, ssm]` 列表 → 每层每 slot `M[slot] = [conv_t, ssm_t]` 整块（状态张量 dim0 行号 = slot id，无 token 维度）；
- **bundle**：`{"K": [attn层][块->t], "V": 同构, "M": [mamba层][slot->[conv,ssm]], "meta": {…双组块表/shape/dtype…}}`，kvt4 兼容字段另置顶层；
- **布局**：`tensors/req{seq}_{rid尾8}/kv_pp0tp0.pt`（一请求一子目录；单卡 TP1 下每请求 1 份）；
- **开关**：`KVC_SAVE_KV=1`（默认关），`KVC_SAVE_DIR=tensors`（start.sh 自动导出）。

## 3. 启动配置要点

- 单卡 TP1（0.8B 小模型），端口 8340，`--max-model-len 16384`，**不传 `--block-size`**（vllm-ascend `patch_mamba_config` 自动矫正 attention block_size=1024，日志可见 `Setting attention block size to 1024 tokens ...`）；
- 请求设计（`scripts/curl/gen_qwen_requests.py --gen`，qwen tokenizer 分词）：
  - **P**：中文长文 ~3.2k 字符 → tokens 落 (2048, 3072) = **2 满块 + 1 未满**，`max_tokens=1`，`ignore_eos`，种 2 满块入哈希链；
  - **R**：P 前缀 + 追问 → tokens 落 (3072, 4096) = **3 满块 + 第 4 块 X/1024**，前缀命中 P 的 2 满块；decode 前 `FILL=1024-X` 步填满第 4 块，步 `FILL+1` 跨界申请第 5 块；`max_tokens=FILL+9`。

## 4. 复现步骤（容器内）

```bash
cd /a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35   # gch 身份
# 参考: 本地源码仓用 apply_patches.sh / revert_patches.sh 直接跑(默认 mac 路径);
#       容器内 run_all 已自动导出容器路径。
mkdir -p logs/server
setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
tail -f logs/server/run_all_screen.log
# 六阶段: [0/6]req 生成 -> [1/6]apply 8 补丁 -> [2/6]起服务(就绪轮询) -> [3/6]P/R 双请求
#         -> [4/6]归档 2 个 .pt -> [5/6]容器内初检 -> [6/6]打包 qwen35_bundle.tar.gz
```

分步（手动）：

```bash
python3 scripts/curl/gen_qwen_requests.py --gen      # 请求体(qwen tokenizer)
bash scripts/patchs/apply_patches.sh                   # 8/8 (run_all 已传容器路径)
bash scripts/server/start.sh                          # 起服务 + KVC_SAVE_KV=1
bash scripts/curl/curl_p_r.sh                         # P -> 6s -> R, 轨迹拆解 + 等归档
python3 scripts/analysis/inspect_kv_tensors.py --dir tensors
python3 scripts/analysis/inspect_prefix.py    --dir tensors
bash scripts/server/stop.sh                           # 先 EngineCore 再 vllm serve
bash scripts/patchs/revert_patches.sh                  # 撤补丁, 源码归零
```

主机侧回收：

```bash
bash scripts/recover/pull_artifacts.sh fetch          # 经 5557 隧道, 2 .pt 核对 + 解包
python3 tensors/view_pt.py                            # 单件快速检视(K/V + mamba M)
```

## 5. 观测要点 checklist

| # | 观测项 | 载体 | 预期（待实验填写） |
|---|--------|------|--------------------|
| 1 | hybrid 共用物理页（单张 int8 raw） | `[KVC][L1] _allocate_kv_cache_tensors[hybrid]` | 每 KVCacheTensor 1 条, shared_by=attn+mamba 层名 |
| 2 | attn 组 K/V 布局 | `[KVC][L1] _reshape_kv_cache_tensors: ...` | shape=(num_blocks, 1024, 2, 256) bf16 之意向 |
| 3 | mamba 组状态视图 | `[KVC][L1] _reshape_kv_cache_tensors[mamba]` | conv (num_blocks,3,6144) bf16 + ssm (num_blocks,16,128,128) fp32 |
| 4 | attn 组前缀命中 2 满块 | `[KVC][L2]/[L3] 前缀查找` + `inspect_prefix.out` | R 第 1~2 块 == P 的满块, 逐位相等 |
| 5 | mamba 槽位独立不复用 | `[KVC][L2]/[L3]` 管理侧 + `inspect_prefix.out` 侧注 | R 槽表与 P 不共享 |
| 6 | TERM 归档横幅对 | `[KVS] ======== 开始/完成保存物理tensor` | 每 worker×请求两行, K=attn 层数, M=mamba 层数×slot |
| 7 | attn 块-行映射 + 末块未写槽位全零 | `inspect_kv_tensors.out` | K/V 网格 + 未写区 RMS≈0 |
| 8 | mamba slot 状态张量本体 | `inspect_kv_tensors.out` / `view_pt.py` | conv/ssm shape/dtype/RMS 合理(非全零/无污染迹象) |

## 6. 结果记录（首轮实验后填写）

> 实验日期：____
> - 启动日志：`logs/server/qwen3-5-0-8b.log`（Block size 矫正行：____；num_blocks=____；attn 层 ____ 个 / mamba 层 ____ 个）
> - P/R 输出：`resp_p.json`（finish=____）/ `resp_r.json`（finish=____，生成 ____ tok）
> - 归档：2 个 .pt（____ MiB）；meta.attn{layers=__, block_table=__, cov=__} / meta.mamba{layers=__, block_table=__}
> - 前缀复用：共享表头块 k=____（== 2 设计值？）；逐位比对：P.b{0,1} vs R 首两块 K/V ______
> - mamba slot：P 槽=____ / R 槽=____；slot 张量 RMS：conv=____ / ssm=____
> - 结论：____________

## 7. 与 kvc/（llama 版）工作区的差异速记

| 项 | kvc/（llama3-8b 单组） | qwen35/（本工作区） |
|----|----------------------|---------------------|
| 模型/层组 | 纯 full-attn 32 层，1 个 KV cache group | attn + GDN/mamba 双组，共 2 个 group |
| 并行/端口 | PP2×TP2 占 4 卡，:8000，每请求 4 份 .pt | 单卡 TP1，:8340，每请求 1 份 .pt |
| block_size | 128 | 1024（hybrid 页匹配矫正） |
| 08 归档 | K/V 单组（kvt4-raw，19 行） | attn K/V + mamba conv/ssm 双组（**kvt5-hybrid-raw**，23 行） |
| 展开路径 | `self.kv_caches`（list of (K,V)） | `self.attn_groups` + `forward_context[层名].kv_cache`（attn=tuple / mamba=state list） |
| 01~07 | vllm 管理侧 155 行，块表单组 | 同 155 行（对 HybridKVCacheManager 同样生效，块表双组） |
| apply/revert | EXPECT 总 174 行 | 总 178 行（vllm 155 + ascend 23） |
