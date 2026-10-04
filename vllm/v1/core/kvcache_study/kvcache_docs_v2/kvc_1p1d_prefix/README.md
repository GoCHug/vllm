# kvc_1p1d_prefix —— 1P1D Prefix Cache 四象限实验工作区（独立自持）

> 本工作区为 **PD 分离 1P1D 形态下的 P/D prefix cache 开关矩阵实验**：1P(prefill producer 卡0/:8100) + 1D(decode consumer 卡1/:8200) + mooncake + load_balance proxy(:8000)。四象限 = P 开关 × D 开关（`--enable-prefix-caching` 默认开 / `--no-enable-prefix-caching` 注入），用 01 号 [PCM] 六打点补丁观察每个组合在"**算多少 / 传多少 / 驻留多少**"上的差异。
>
> **本区完全独立**：补丁/脚本/请求体/文档全套自带，零外部依赖（不依赖 ../kvc、../kvc_1p1d、../kvc_pd_prefix 任何文件；01 号补丁目标 mooncake_connector.py 与其他工作区补丁不重叠，可叠加）。
>
> **核心文档**：`docs/0_pd_prefix_matrix.md`（机制 / 四张场景卡 / 实测 / 成本模型 / 选型一站式）。**历史权威数据**：`logs/legacy/`（09-30 贵安 + 09-29 乌兰交叉）；**本轮实测**：`logs/q{1..4}/` + 汇总 `logs/analysis/matrix_report.out`（铁律核验）。

## 0. 30 秒结论（四条铁律）

| 象限 | P 开关 | D 开关 | P prefill | mooncake 传输 | D 跨请求驻留 |
|---|---|---|---|---|---|
| ① q1_p1d1（默认） | ✓ | ✓ | 230 tok | **32.0 MiB** 增量 | 保留（LRU） |
| ② q2_p1d0 | ✓ | ✗ | 230 tok | **64.0 MiB** 全量 | 无 |
| ③ q3_p0d1 | ✗ | ✓ | 486 tok 全算 | **32.0 MiB** 增量 | 保留 |
| ④ q4_p0d0 | ✗ | ✗ | 486 tok 全算 | **64.0 MiB** 全量 | 无 |

1. **P 的开关只影响"算多少"，D 的开关只影响"传多少"**（解耦铁证：卡③ 与卡① 传输量分毫不差）
2. vLLM v1 **默认双侧都开**（象限① = 线上默认态）
3. **P 上报块恒全量**（P 不感知 D 缓存）
4. 四格**正确性全等**（APC 只改"KV 从哪来"）

## 目录树

```
kvc_1p1d_prefix/
├── README.md                              本文档(工作区导航)
├── docs/
│   └── 0_pd_prefix_matrix.md              主文档: 机制底座 / 四张场景卡 / 实测 /
│                                           成本模型 / 选型 / 本轮复现
├── scripts/                               五子目录(前四目录与 logs/ 一一对应)
│   ├── server/                            生命周期与编排:
│   │   ├── start_p.sh / start_d.sh        参数化启动($1=象限日志子目录 $2=pc:1|0;
│   │   │                                  pc=0 注入 --no-enable-prefix-caching)
│   │   ├── start_proxy.sh / stop.sh       proxy 起停(:8000 -> P:8100/D:8200)
│   │   ├── run_quadrant.sh                单象限全流程(HBM 防抢占 + 崩溃早退 +
│   │   │                                  EXIT trap 兜底清理 + 证据落盘 12 文件)
│   │   ├── run_matrix.sh                  四象限一键(动态选干净卡 + 失败重试×2)
│   │   └── run_all.sh                     五阶段总编排(补丁→矩阵→汇总→打包→收尾提示)
│   ├── patchs/                            补丁与应用/回滚:
│   │   ├── 01_pcm_prefix_cache_matrix.patch   六打点补丁([PCM]×7, 目标唯一
│   │   │                                      mooncake_connector.py, 独立可叠加)
│   │   ├── apply_patches.sh / revert_patches.sh  应用/回退(防重/dry-run/计数/md5)
│   │   └── gen_01_pcm_patch.py            补丁生成器(锚点断言 + 三重自检)
│   ├── curl/                              请求体(req_p 324 tok 种 / req_r 486 tok
│   │                                      前缀复用 256) + gen_pd_requests.py
│   ├── analysis/
│   │   └── matrix_report.py               四象限对照总表 + 铁律核验(A/B/C/D)
│   └── recover/pull_artifacts.sh          产物回收(主机侧 fetch 单命令)
└── logs/
    ├── q1_p1d1/ ~ q4_p0d0/                本轮四象限产物(一象限一子目录):
    │                                      p/d_llama.log, resp_{p,r}.json, {p,d}_pcm.txt,
    │                                      d_transfer.txt, {p,d}_hitrate.txt,
    │                                      p_delayfree.txt, q_summary.md
    ├── server/                            matrix_screen.log / run_all_screen.log
    ├── analysis/                          matrix_report.out(汇总判读)
    └── legacy/                            历史权威轮(09-30 贵安 round_0930_guian +
                                           09-29 乌兰 round_legacy_0929am 交叉 + 旧总表)
```

## 快速上手（容器内）

```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc_1p1d_prefix
setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
tail -f logs/server/run_all_screen.log
# [1/5] 01 号 PCM 补丁([PCM]×7 -> mooncake_connector.py, 基线 md5 00baf169...)
# [2/5] 四象限矩阵(~2.5-3min/象限, 全程 ~12min; 失败重试×2)
# [3/5] 汇总: matrix_report.py -> logs/analysis/matrix_report.out(铁律核验)
# [4/5] 打包: kvc_1p1d_prefix_bundle.tar.gz(logs/) + md5
# 收尾(手动): bash scripts/server/stop.sh && VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
#             bash scripts/patchs/revert_patches.sh     # md5 回 00baf169...
# 回收(主机侧): bash scripts/recover/pull_artifacts.sh fetch   # 经 5557 隧道
```

单象限手工复跑（调试用）：
```bash
bash scripts/server/run_quadrant.sh q2_p1d0 1 0    # 只跑象限②(P开D关)
python3 scripts/analysis/matrix_report.py --dir logs   # 汇总复核
```

## 四张场景卡 & 证据文件对照

每象限 12 文件（grep 直达 `grep '\[PCM\]' logs/q*/{p,d}_llama.log`）：
- p_pcm.txt / d_pcm.txt = [PCM] 六打点全集（CFG/SCHED/ALLOC/PFINISH/XFER-entry+FULL-HIT/XFER-end）
- d_transfer.txt = 原生 `KV cache transfer ... took N ms`（传输耗时）
- {p,d}_hitrate.txt = 原生 Prometheus `Prefix cache hit rate`（命中率自证）
- q_summary.md = 一行式快照（十行关键打点）

四张卡（配置 × 打印全集 × 判读）见 `docs/0_pd_prefix_matrix.md` §3。

## 复现注意事项

1. **本区唯一补丁 01**：目标 mooncake_connector.py 与 ../kvc、../kvc_1p1d 的 01~08 无文件重叠——可独立 apply，也可与它们叠加（互不冲突）。
2. **P 必须先于 D 启动**（mooncake 会话建立次序）；proxy 在两侧就绪后启动。
3. **每象限独立起停**（进程间零状态污染）——四象限的对比才有意义；重跑单象限不影响其他象限产物。
4. **HBM 防抢占**：编排自动等卡空闲（<12GB）再启；外部租户瞬占会导致启动超时重试。
5. **apply/revert 单独跑**：容器内需传 `VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend`（run_all 已自动导出；默认路径是本地 macOS 的）。
6. 基线指纹：pristine 0.23.0 mooncake_connector.py md5 = `00baf169f48fb167b9f6dfe650ac0ea5`（revert 后应精确回此值）。

## 与兄弟工作区的关系

| 工作区 | 关系 |
|---|---|
| `../kvc/` | 单机 PP2×TP2 KVCache 观测区（本区 [PCM] 六打点方法论与其同族，但补丁目标不同文件、互不依赖） |
| `../kvc_1p1d/` | 1P1D KVCache 打印+物理归档+P→D 传输正确性（Tx 区逐位检查为本区"四格正确性全等"背书；两区补丁可叠加） |
| 本区（由 kvc_pd_prefix 重构而来） | 1P1D prefix 开关四象限：**传输正确性之外的另一维——算力/带宽/内存的行为矩阵** |

> 原 kvc_pd_prefix/（09-30 贵安权威轮 + 09-29 乌兰交叉轮）已重构为本区：四象限方法论保留、拓扑锚定 1P1D、目录结构与 kvc 家族对齐（scripts 五子目录 + logs 产物区 + docs 文档区）、历史实测数据迁存 logs/legacy/。
