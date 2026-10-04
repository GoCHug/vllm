# kvc/ KVCache 实操工作区（v2.5 块-行映射归档版 · P/R 双请求 · llama3-8b · NPU）

> vLLM V1 KVCache 管理端到端实验交付物。本地/容器两侧同步：
> - **本地**：`vllm/vllm/v1/core/kvcache_study/kvcache_docs_v2/kvc/`
> - **容器**：`/a3_inference/itask/workdir/wsl02075301/kvc/`（gggtest pod；实验后已 stop + revert，`/vllm-workspace` 源码未改动）
>
> **本轮（2026-10-03 15:16，v2.5）**：一个请求一个子目录 + **块-行映射**——TERM 时把该请求在各 worker 的物理
> KV 块**整块原样**归档 `tensors/req{seq}_{rid尾8}/kv_pp{pp}tp{tp}.pt`（目录内 4 worker 各一份；meta
> 新增 group_size=组内层数，block id = 组内每张 K/V 张量 dim0 的统一行号），横幅两行式
> `======== 开始/完成保存物理tensor ========`（v2.3 演进而来）；查看报告逐块直读「= 组内 16 张 K +
> 16 张 V 张量(dim0) 各取第 {blk} 行」：`python3 scripts/analysis/inspect_kv_tensors.py --dir tensors` → `logs/analysis/inspect_kv_tensors.out`。
>
> 版本沿革：v1 打印版（09-29）→ v2 归档版（10-02 早）→ v2.1 全链日志（532 行/轮）→
> v2.2 精简两行式（20 行/轮）→ v2.3 横幅式 → v2.4 请求子目录 + 报告式查看器 → **v2.5 + meta.group_size 块-行映射（20 行/轮，当前）**。

## 0. 本轮核心结论（详见 docs/0_kvcache_e2e_record.md §4.3/§5.5/§6）

1. **8/8 归档结构 PASS**：4 worker × P/R 全就位（P [1,2,3]·324tok；R [1,2,4,5,6]·520tok；meta.group_size=16 × 8 覆盖）。
2. **前缀缓存零篡改（铁证）**：P 种块 vs R 命中块 [1,2] 在全 4 worker × 16 层 K/V **逐位相等**。
3. **横幅日志全链对账 PASS**：开始横幅 8/8（字段/块表/cov/group_size = meta）；完成横幅 8/8（字节 =
   文件实测）；flush 32.0~47.9ms 平均 39.9ms。[KVS] 总行 **20**（v2.1 同实验 532 精简 26 倍；横幅对账为实验轮内 run_all [4/6] + fetch 复核）。
4. **重算仅 ULP 级差异**：P.b3 vs R.b4 L00 K 2/34816、V 18/34816 元素位翻转（Pearson=1.000001，与 10-02 轮一致=位级确定）；L15 残差流放大为大面积位级漂移（77%/88%），Pearson 0.999970/0.999887。
5. **跨 run 位级确定**：本轮 L00 统计与 v1（09-29）打印记录值 4 位完全相同（八轮实验互证：v1→v2→v2.1→v2.2→v2.3→v2.4→v2.5×2）。
6. **未写槽位全零**：块池新建基线为零。

## 1. 目录树

```
kvc/
├── README.md                          本文件
├── scripts/                           操作脚本 + 补丁 + 请求体（五子目录, 前四目录与 logs/ 一一对应）
│   ├── server/                        服务生命周期 + 一键编排（start.sh / stop.sh / run_all.sh：补丁→服务→curl→验归档→双初检→打包）
│   ├── patchs/                        补丁与应用/回滚（01~07 管理侧 155 行 + 08 v2.5 块-行映射归档 19 行 + apply/revert_patches.sh + 98 调用点清单）
│   ├── curl/                          P/R 请求（curl_p_r.sh 发送 + gen_cn_requests.py 生成 + req_p/req_r.json 请求体）
│   ├── analysis/                      离线检查双脚本（inspect_kv_tensors.py 查看器 v2.5 + inspect_prefix.py 前缀复用重算一致性检查 -> logs/analysis/*.out）
│   └── recover/pull_artifacts.sh      产物回收（主机侧 fetch 单命令；打包已并入 run_all [6/6]）
├── logs/                              本轮（v2.5，15:16）产物 · 四子目录, 与 scripts/ 前四目录一一对应
│   ├── server/                        llama-3-8b.log（服务全量 1182 行）+ run_all_screen.log（一键留痕）
│   ├── patchs/                        kvc_startup / kvc_p / kvc_r / kvs_archive_lines.log（patch 打印日志拆解轨迹 172/61/719/20 行）
│   ├── curl/                          resp_p / resp_r.json + curl_screen.log（响应与打屏；请求体在 scripts/curl/）
│   └── analysis/                      inspect_kv_tensors.out / inspect_prefix.out（查看报告 + 前缀复用重算一致性检查）
├── tensors/                           物理 tensor 归档（本轮 rid 尾8 P=bdd8c88d / R=9eb8d7fa）
│   └── req{seq}_{rid尾8}/kv_pp{pp}tp{tp}.pt × 2 目录 × 4 worker       一请求一子目录（12/20 MiB）
└── docs/                              分析文档
    └── 0_kvcache_e2e_record.md                          E2E 全记录（用例设计 + 启动期/P/R 生命周期
                                                          + v2.5 归档互证/六项验证，原 docs 1/2 合并）
```

## 2. 环境快照（本轮实测，2026-10-03 15:16）

| 项 | 值 |
|---|---|
| Pod / 卡 | gggtest（a3, 4×hpu910a3）@ gpuxdn010030015012.guian02 |
| 模型 / 基线 | Meta-Llama-3-8B bf16（32 层, kv_heads 8→4/TP2, block_size 128）；vllm @0fc695f + vllm-ascend @5cb98c（v0.23.0，git 干净打补丁） |
| 服务 | `vllm serve --enforce-eager -tp2 -pp2`（单实例占 4 卡）；就绪 ~50s；APIServer 5848 / EngineCore 5969 / worker 6070~6073 = npu:0~3 |
| KV 池 | num_blocks=13291；每 worker 每层 K/V 分离 `(13291,128,4,128)` bf16 |
| 归档开关 | `KVC_SAVE_KV=1`（默认关零侵入）+ `KVC_SAVE_DIR=<工作区>/tensors`（start.sh 自导出） |
| 节奏 | 15:16:36 run_all 启动 → 50s 就绪 → P 15:17:32（TERM 归档×4）→ 7s → R 15:17:39（TERM 归档×4）→ 15:18:10 产物就绪并打包 → 主机 fetch |

## 3. 离线查看快速上手（本地，torch CPU 即可）

```bash
cd kvc
python3 scripts/analysis/inspect_kv_tensors.py --dir tensors                 # 归档查看报告 -> logs/analysis/inspect_kv_tensors.out
python3 scripts/analysis/inspect_kv_tensors.py --dir tensors --preview 8     # tensor 预览前 8 值（默认 4）
python3 scripts/analysis/inspect_prefix.py --dir tensors                    # 前缀复用重算一致性检查 -> logs/analysis/inspect_prefix.out
```

## 4. 复现（本地源码仓用本地路径；容器内传环境变量或用 run_all 一键）

**本地源码仓**（apply/revert 默认路径已配置为本地 macOS 路径，直接跑无需传参）：
```bash
cd kvc
bash scripts/patchs/apply_patches.sh          # 默认指向 /Users/wushanglun/Desktop/vllmgch/{vllm, vllm-ascend}
bash scripts/patchs/revert_patches.sh         # 同上（逐文件幂等：已干净目标自动 skip；若 08 为旧版研究态会安全中止并给 git 恢复指引）
```

**容器内**（需显式传环境变量覆盖本地默认，或用 run_all.sh 一键）：
```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc
# 分步（手动传容器路径）：
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
    bash scripts/patchs/apply_patches.sh      # 8/8 dry-run + 应用 + 174 行计数 + py_compile
bash scripts/server/start.sh                # 服务后台启动(自动导出 KVC_SAVE_KV=1 等归档 env)
bash scripts/curl/curl_p_r.sh             # P → 6s → R；打屏/轨迹/归档全落盘
tar -czf kvc_bundle.tar.gz logs/ tensors/    # 产物打包（run_all 一键已含 [6/6]；主机侧 fetch 拉回）
bash scripts/server/stop.sh
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
    bash scripts/patchs/revert_patches.sh     # 撤补丁，容器源码归零
# 或一键（run_all.sh 已自动导出容器路径，末尾 [6/6] 自动打包）：
mkdir -p logs/server   # nohup 重定向目标需先存在
setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
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
| `kvc/`（本区） | 单机 PP2×TP2，物理 KVCache 观测（v1 打印→v2 归档→v2.3 横幅式→v2.4 请求子目录→v2.5 块-行映射） |
| `kvc_offline/` | v3 原样归档 + C0~C3 检查器 |
| `kvc_pd/` 等 | PD 分离实验线（另有三个子工作区） |

## 7. 补丁要点速记（08 v2.5）

- 时点：每步前向后 `_kvc_rel_snapshot`；`written ≥ prompt+max_tokens-1` 触发 TERM（恰在 L5 释放横幅前）；
- 归档：每层每块 `kt[blk]` 整块 cpu().clone()（含未写槽位）→ 后台线程 torch.save 到 `req{seq}_{rid尾8}/kv_pp{pp}tp{tp}.pt`（一请求一子目录，meta 记 group_size=组内层数——block id 即组内每张 K/V 张量 dim0 行号，查看报告逐块展示块-行映射）；
- **日志**：每 worker×请求两行——开始（保存什么：K/V 池/shape/块表/文件名）+ 完成
  （落盘回执：文件名/字节数/耗时）；失败兜底 ARCHIVE-FAIL；
- 安全：env 默认关；异常三层兜底；TERM 不等 IO（flush 平均 ~40ms 后台）。

更多细节：`docs/0_kvcache_e2e_record.md`——§4.3/§5.5（v2.5 归档设计与离线互证）、§6（实证结论与横幅审计）、§2.2（补丁演进表）。
