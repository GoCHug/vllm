# Qwen3.5-0.8B KVCache 实验工作区

qwen3.5-0.8B（**hybrid full-attn + GDN/mamba 混合架构**）在 vllm-ascend v0.23.0 + 昇腾 A3 NPU 上的实验交付工作区。

> **文档职责**：本 README 为整个目录的介绍（概览 + 导航）；
> 两个实验的**全流程与根因/修复分析详解**分别见 `docs/` 下姊妹文档。

## 实验环境快照

| 项 | 值 |
|---|---|
| Pod / 卡 | gggtest4（A3, 4×hpu910a3）@ 172.16.145.250，gch02599191 身份 |
| 镜像 | `hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| 模型 | `modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902`（24 层 = 6 attn + 18 GDN） |
| vllm | 0.23.0（容器 `/vllm-workspace/vllm` + `/vllm-workspace/vllm-ascend`，editable） |
| 容器工作区 | `/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35`（NFS 持久化） |

## 两个实验一览

| 实验 | 日期 | 一句话结论 | 详解文档 |
|------|------|-----------|---------|
| **① block-size 精度问题** | 2026-10-19 | `--block-size 2048` 触发混合模型页内切窗错位 → 乱码；[PR 15408](https://github.com/vllm-project/vllm-ascend/pull/15408) 一行条件修复验证通过 | `docs/0_blocksize_precision_record.md` |
| **② 混合 KVCache 三轮观测** | 2026-10-20 | 物理张量层实证 ① 的根因链（2048 时 K 全段 NaN、mamba ssm 12× 污染）+ 15408 修复后归档与原生基线**完全一致** | `docs/0_hybrid_kvcache_record.md` |

## 目录结构

```
qwen35/
├── README.md                                  # 本文件：工作区介绍与导航
├── docs/                                      # 两个实验的详解文档
│   ├── 0_blocksize_precision_record.md        #   ① 精度问题根因/修复验证全流程
│   └── 0_hybrid_kvcache_record.md             #   ② KVCache 三轮观测全流程与分析
├── scripts/                                   # 操作脚本（两实验共用）
│   ├── server/                                #   start_normal.sh / start_abnormal.sh / kill.sh
│   │                                          #     （start 系内置 KVC_SAVE_KV=1 归档 env）
│   ├── curl/                                  #   curl.sh [A|B|C]（ignore_eos 保证 TERM 归档,
│   │                                          #     响应 tee 落 logs/curl_resp.json）
│   ├── patchs/                                #   01~07 号 [KVC] 补丁(同 kvc) + 08 号 hybrid 归档版
│   │                                          #     + apply/revert 脚本 + kvc_patch_locations.txt
│   │                                          #     + pr_15408.patch(修复补丁)
│   ├── analysis/                              #   inspect_kv_tensors.py(hybrid 查看器)
│   │                                          #     + inspect_prefix.py(前缀复用)
│   │                                          #     + compare_modes.py(三轮两两比对)
│   │                                          #     + overwrite_evidence.py(覆写四层证据)
│   │                                          #     + group_slots.py(4组block id全景)
│   └── recover/                               #   pull_artifacts.sh fetch(容器 tar → 主机拉回)
├── logs/                                      # 实验产物（日志/响应/分析报告）
│   ├── qwen35_normal.log                      #   轮 A 服务日志(含 L1 页几何 + [KVC]/[KVS] 轨迹)
│   ├── qwen35_abnormal_runB.log               #   轮 B 服务日志(2048 无修复)
│   ├── qwen35_abnormal_runC.log               #   轮 C 服务日志(2048+15408, 矫正行 matches)
│   ├── curl_resp.json                         #   三轮响应体(A 正常 / B 乱码 / C 正常)
│   └── analysis/                              #   三组对比报告 compare_*.out + 查看器输出
└── tensors/                                   # TERM 物理归档(kvt5-hybrid-raw)
    ├── view_pt.py                             #   单件快速检视器(K/V/M + meta)
    ├── normal/req1_90bfa9ce/kv_pp0tp0.pt     #   轮 A 归档(基线, 32 MB)
    ├── abnormal/req1_abdc0ba8/kv_pp0tp0.pt   #   轮 B 归档(污染, 45 MB)
    └── fixed/req1_8345cee7/kv_pp0tp0.pt      #   轮 C 归档(修复后, 32 MB)
```

## 快速上手

### 本地离线查看（torch CPU 即可）

```bash
cd qwen35
python3 tensors/view_pt.py tensors/normal/req1_90bfa9ce/kv_pp0tp0.pt    # 单件检视
python3 scripts/analysis/inspect_kv_tensors.py --dir tensors/normal     # 完整报告 → logs/analysis/
python3 scripts/analysis/compare_modes.py \
    --normal tensors/normal --abnormal tensors/abnormal                # 根因对比
python3 scripts/analysis/overwrite_evidence.py                       # 覆写四层证据(docs §6.2)
```

### 容器内复现（三轮规程详见 `docs/0_hybrid_kvcache_record.md`）

```bash
cd /a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
    bash scripts/patchs/apply_patches.sh     # [1] 打补丁(01~08, 两仓 [KVC]=178 行校验)
bash scripts/server/start_normal.sh          # [2] 起服务(默认 1024; 归档 env 内置)
bash scripts/curl/curl.sh A                  # [3] 发请求(ignore_eos → TERM 归档)
bash scripts/server/kill.sh                  # [5] 清理(先杀 EngineCore 再杀 serve)
VLLM_DIR=... bash scripts/patchs/revert_patches.sh    # [6] 撤补丁, 源码归零
```

完整三轮（A 正常基线 → B 2048 异常 → C 2048+15408 修复）含归档隔离/日志防覆盖/
15408 叠加与撤收顺序 → `docs/0_hybrid_kvcache_record.md` §8。

## 关键结论速查

| 指标 | A(默认 1024) | B(2048 无修复) | C(2048+15408) |
|------|-------------|----------------|---------------|
| 响应 | `1+14=15` | `1!!!!!!!!!` | `1+14=15` |
| meta block_size | 1024 | 2048 | 1024（矫正回） |
| K 已写区 RMS | 1.3992 | **NaN** | 1.3992 |
| K 未写区 RMS | 0.000000 | **NaN** | 0.000000 |
| M ssm RMS / Pearson | 0.0279 / — | **0.3357 / 0.086** | 0.0279 |

> 根因链与证据闭环的完整分析 → `docs/0_hybrid_kvcache_record.md` §6-§7；
> block-size 根因与页内切窗图示 → `docs/0_blocksize_precision_record.md` §2。

## 容器复用备忘（Pod 重建）

```bash
itask create --name gggtest4 --image $IMAGE --model ... --4card -t a3   # 详见 docs/0_blocksize_*.md §4
```
