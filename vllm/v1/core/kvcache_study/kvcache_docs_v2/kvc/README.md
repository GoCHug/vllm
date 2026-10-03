# kvc/ KVCache 实操工作区（v2.3 归档+横幅式日志版 · P/R 双请求 · llama3-8b · NPU）

> vLLM V1 KVCache 管理端到端实验交付物。本地/容器两侧同步：
> - **本地**：`vllm/vllm/v1/core/kvcache_study/kvcache_docs_v2/kvc/`
> - **容器**：`/a3_inference/itask/workdir/wsl02075301/kvc/`（gggtest pod；实验后已 stop + revert，`/vllm-workspace` 源码未改动）
>
> **本轮（2026-10-02 16:24，v2.3）**：横幅式两行日志——TERM 时把该请求在各 worker 的物理
> KV 块**整块原样**归档 `.pt`，打 `======== 开始保存物理tensor ========` /
> `======== 完成保存物理tensor ========` 横幅（与 L5/L1 的 `======== ========` 风格统一）。
>
> 版本沿革：v1 打印版（09-29）→ v2 归档版（10-02 早）→ v2.1 全链日志（532 行/轮）→
> v2.2 精简两行式（20 行/轮）→ **v2.3 横幅式（20 行/轮，当前）**。

## 0. 本轮核心结论（详见 docs/1 §3.3/§4.5/§5）

1. **8/8 归档结构 PASS**：4 worker × P/R 全就位（P [1,2,3]·324tok；R [1,2,4,5,6]·520tok）。
2. **前缀缓存零篡改（铁证）**：P 种块 vs R 命中块 [1,2] 在全 4 worker × 16 层 K/V **逐位相等**。
3. **横幅日志全链对账 PASS**：开始横幅 8/8（字段/块表/cov = meta）；完成横幅 8/8（字节 =
   manifest）；flush 33.4~71.6ms 平均 45.5ms。[KVS] 总行 **20**（v2.1 同实验 532 精简 26 倍）。
4. **重算仅 ULP 级差异**：P.b3 vs R.b4 L00 K 2/34816、V 18/34816 元素位翻转（Pearson=1.000000）；L15 残差流放大为大面积位级漂移，Pearson 仍 ≥0.99989。
5. **跨 run 位级确定**：本轮 L00 统计与 v1（09-29）打印记录值 4 位完全相同（五轮实验互证）。
6. **未写槽位全零**：块池新建基线为零。

## 1. 目录树

```
kvc/
├── README.md                          本文件
├── scripts/                           操作脚本
│   ├── start.sh                       服务启动（自动 export KVC_SAVE_KV=1 + KVC_SAVE_DIR）
│   ├── stop.sh                        杀服务（pkill + 确认归零）
│   ├── curl_p_r.sh                    P/R 双请求 + 打屏留痕 + 三段轨迹拆解 + 等归档
│   ├── gen_cn_requests.py             P/R 请求体生成器
│   ├── inspect_kv_tensors.py          ★ tensor 离线查看器（列表/深查/切片/逐位比对/selftest）
│   ├── pull_artifacts.sh              产物回收（容器 pack → 主机 fetch + md5 校验）
│   └── run_all.sh                     一键编排 [patch→serve→curl→验归档→初检]
├── patch/                             补丁与应用/回滚
│   ├── 01~07_vllm_*.patch             管理侧打印（155 行 [KVC]，各版通用）
│   ├── 08_vllm_ascend_worker_model_runner_v1.py.patch   ★ v2.3 归档+横幅式日志（19 行）
│   ├── apply_patches.sh / revert_patches.sh             dry-run 预检 + 计数(174) + py_compile; revert 幂等(干净目标自动 skip)
│   └── kvc_patch_locations.txt        v1 96 打印点位置清单（历史参照）
├── log/                               本轮（v2.3，16:24）全部产物
│   ├── llama-3-8b.log                 服务全量 1211 行（启动 398 + P 67 + R 746）
│   ├── kvc_p.log / kvc_r.log / kvc_startup.log          [KVC] 三段拆解（61/719/172 行）
│   ├── kvs_archive_lines.log          [KVS] 横幅日志 20 行（4 启用 + 8 开始 + 8 完成）
│   ├── curl_screen.log                curl 命令 + 响应打屏（req/resp json 同存）
│   ├── inspect_*.out                  离线分析原始输出 ×9（列表/深查/切片/比对/ULP/对账/审计）
│   └── run_all_screen.log             一键编排留痕
├── tensors/                           物理 tensor 归档（本轮 rid 尾8 b3cb18f2/81516ab1）
│   ├── kv_pp{pp}tp{tp}_s{seq}_{rid尾8}.pt × 8          4 worker × P/R（12/20 MiB）
│   └── manifest.json                  md5 + meta 摘要（fetch 后校验 8/8 OK）
└── docs/                              分析文档
    ├── 1_kvc_patch_apply_e2e_record.md                 E2E 全记录（启动期/P/R 生命周期 + v2.3
    │                                                   归档设计/横幅审计/六项验证，即原 3/4 并入）
    └── 2_kvc_cn_curl_case.md                           P/R 中文 curl 用例设计（通用）
```

## 2. 环境快照（本轮实测，2026-10-02 16:24）

| 项 | 值 |
|---|---|
| Pod / 卡 | gggtest（a3, 4×hpu910a3）@ gpuxdn010030015012.guian02 |
| 模型 / 基线 | Meta-Llama-3-8B bf16（32 层, kv_heads 8→4/TP2, block_size 128）；vllm @0fc695f + vllm-ascend @5cb98c（v0.23.0，git 干净打补丁） |
| 服务 | `vllm serve --enforce-eager -tp2 -pp2`（单实例占 4 卡）；就绪 50s；APIServer 94246 / EngineCore 94363 / worker 94580~94583 = npu:0~3 |
| KV 池 | num_blocks=13291；每 worker 每层 K/V 分离 `(13291,128,4,128)` bf16 |
| 归档开关 | `KVC_SAVE_KV=1`（默认关零侵入）+ `KVC_SAVE_DIR=<工作区>/tensors`（start.sh 自导出） |
| 节奏 | 16:24:58 run_all 启动 → 50s 就绪 → P 16:25:54（TERM 归档×4）→ 6s → R 16:26:01（TERM 归档×4）→ 16:26:43 初检 PASS → pack/fetch |

## 3. 离线查看快速上手（本地，torch CPU 即可）

```bash
cd kvc
python3 scripts/inspect_kv_tensors.py --selftest                                        # 查看器自检
python3 scripts/inspect_kv_tensors.py --dir tensors                                     # ① 归档清单
python3 scripts/inspect_kv_tensors.py --dir tensors --file kv_pp0tp0_s1_b3cb18f2.pt     # ② 深查
python3 scripts/inspect_kv_tensors.py --file kv_pp0tp0_s2_81516ab1.pt --layer 0 --block 6 --kv K  # ③ 切片
python3 scripts/inspect_kv_tensors.py --compare kv_pp0tp0_s1_b3cb18f2.pt kv_pp0tp0_s2_81516ab1.pt # ④ 前缀复用比对
```

## 4. 复现（本地源码仓用本地路径；容器内传环境变量或用 run_all 一键）

**本地源码仓**（apply/revert 默认路径已配置为本地 macOS 路径，直接跑无需传参）：
```bash
cd kvc
bash patch/apply_patches.sh          # 默认指向 /Users/wushanglun/Desktop/vllmgch/{vllm, vllm-ascend}
bash patch/revert_patches.sh         # 同上（逐文件幂等：已干净目标自动 skip；若 08 为旧版研究态会安全中止并给 git 恢复指引）
```

**容器内**（需显式传环境变量覆盖本地默认，或用 run_all.sh 一键）：
```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc
# 分步（手动传容器路径）：
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
    bash patch/apply_patches.sh      # 8/8 dry-run + 应用 + 174 行计数 + py_compile
bash scripts/start.sh                # 服务后台启动(自动导出 KVC_SAVE_KV=1 等归档 env)
bash scripts/curl_p_r.sh             # P → 6s → R；打屏/轨迹/归档全落盘
bash scripts/pull_artifacts.sh pack  # manifest + tar；主机侧 fetch(需 itask ssh-tunnel --port 5557)
bash scripts/stop.sh
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
    bash patch/revert_patches.sh     # 撤补丁，容器源码归零
# 或一键（run_all.sh 已自动导出容器路径，无需手动传）：
setsid nohup bash scripts/run_all.sh > log/run_all_screen.log 2>&1 < /dev/null &
```

## 5. Pod 重建备忘（gggtest 若被长时间空闲回收）

```bash
IMAGE=hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64
model-cli pull hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_74000048_meta-llama-3-8b:148700128_20260921221233
itask create --name gggtest --image $IMAGE \
  --model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_74000048_meta-llama-3-8b:148700128_20260921221233 \
  --4card -t a3
```

## 6. 兄弟工作区（kvcache_docs_v2 下）

| 工作区 | 主题 |
|---|---|
| `kvc/`（本区） | 单机 PP2×TP2，物理 KVCache 观测（v1 打印→v2 归档→v2.3 横幅式两行日志） |
| `kvc_offline/` | v3 原样归档 + C0~C3 检查器 |
| `kvc_pd/` 等 | PD 分离实验线（另有三个子工作区） |

## 7. 补丁要点速记（08 v2.3）

- 时点：每步前向后 `_kvc_rel_snapshot`；`written ≥ prompt+max_tokens-1` 触发 TERM（恰在 L5 释放横幅前）；
- 归档：每层每块 `kt[blk]` 整块 cpu().clone()（含未写槽位）→ 后台线程 torch.save；
- **日志**：每 worker×请求两行——SAVE（保存什么：K/V 池/shape/块表/文件名）+ SAVED
  （完成了：文件名/字节数/耗时）；失败兜底 ARCHIVE-FAIL；
- 安全：env 默认关；异常三层兜底；TERM 不等 IO（flush 平均 ~39ms 后台）。

更多细节：`docs/1`——§3.3/§4.5（v2.3 归档设计与离线互证）、§5（实证结论与横幅审计）、§1（补丁演进表）。
