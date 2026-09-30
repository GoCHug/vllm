# kvc_pd_offline —— PD 分离 KVCache 张量归档与离线全量检查工作区（v2 双证据链）

> **回答的问题**：`kvc_pd/`（v1 sha256 指纹链）已证明传输逐位无损（三轮 PASS）；本区回答取证级问题——**如果**出现不等，差在哪个 token/head/位？差多少？差的样子像重算还是 DMA 损伤？且**改检查项不重跑实验**。
>
> **方法**：TERM 快照原始张量 `.pt` 归档（10 号 dump patch，与 09 指纹同点双采）→ `check_kv_tensors.py` 五级离线全检 → `tensor_report.md/.json`。
> **判据**：L0 结构 ∧ **L1 两链指纹互证**（.pt 重算 sha256 = 日志 [FPB]）∧ **L2 Tx 区 torch.equal** ∧ L3 尾槽差异=exact 或重算一致（幅度判据：\|Δ\|≤5% 层幅值且无 NaN/Inf）∧ decode 区数值健康。
>
> **状态**：**✅ 已实施，首轮（2026-09-30 10:50 gggtest）PASS**——L1 互证 **1472/1472**、L2 Tx **128/128 对 torch.equal**、L3 尾槽差异全落重算带（0.02%~1.15% 层幅值）、decode 34 行健康；v1 指纹链同轮 448/448 PASS。方案见 [`docs/1_tensor_archive_design.md`](docs/1_tensor_archive_design.md)，实验记录见 [`docs/2_tensor_archive_experiment.md`](docs/2_tensor_archive_experiment.md)。
> 首轮落地两项实测修正（详见 docs/2 §2.4~§2.5）：L1 解析 bug、L3 判据从位模式（ULP-only）改为数值语义（重算差实测位谱=分布式多 bit 翻转，bf16 尾数仅 7 位）。

## 目录结构（实测，2026-09-30 首轮后）

```
kvc_pd_offline/
├── README.md                           本文档
├── docs/
│   ├── 1_tensor_archive_design.md      方案设计（§3 含 10 号 patch 全文与源码落点详解）
│   └── 2_tensor_archive_experiment.md  首轮实验记录（五级数字/发现/修复/复现）
├── patch/                              10 号归档补丁 + 三段式 apply/revert
│   ├── 10_pd_kv_tensor_dump.patch      TERM 张量归档（叠在 09 之上，env 开关；roundtrip 验证过）
│   ├── apply_offline_patches.sh        一键应用：kvc 01~08 → 09 → 10（[KVT] 标记 + py_compile 终验）
│   └── revert_offline_patches.sh       一键撤销（反向 10 → 09 → 01~08）
├── scripts/
│   ├── run_offline_all.sh              一键七阶段（含 KVC_DUMP_TENSORS=1 导出）
│   ├── start_p.sh / start_d.sh / start_proxy.sh / curl_pd.sh / stop_pd.sh   （拷贝自 kvc_pd，log 指向本区）
│   ├── compare_fp.py                   （拷贝；v1 链裁决同轮同目录产出）
│   ├── check_kv_tensors.py             五级离线检查器（L0~L4，仅 torch-cpu；--selftest 合成样例自测）
│   └── pull_tensors.sh                 归档回收（容器 pack / 主机 fetch 两模式 + md5 校验）
└── log/                                （首轮全产物已归档）
    ├── tensors/kv_{P,D}_{1,2}_{rid尾8}.pt ×4 + manifest.json   （207 MiB，md5 双侧一致）
    ├── tensor_report.md / tensor_report.json                    （verdict=PASS）
    ├── verdict.txt / kvc_*.log / {p,d}_llama.log / resp_* 等    （v1 套件，同轮）
    └── run_offline_screen.log / kvt_archive_lines.log           （七阶段主控/归档留痕）
# 依赖：kvc 01~08、kvc_pd 09 指纹 patch（跨 ../kvc/ 与 ../kvc_pd/ 引用）
# 双证据架构与链间互证（L1）设计详见 docs/1 §2
```

## 环境信息（规划，复用 v1 形态）

| 项 | 值 |
|---|---|
| 容器 | gggtest（itask 4×hpu910a3，与 kvc_pd 实验同形态） |
| 归档触发 | TERM（`_kvc_kv_dump`，09 指纹同点双采），`KVC_DUMP_TENSORS=1` 开关 |
| 单轮规模 | ~207 MiB .pt ×4 + manifest（40.5/40.5/61/65 MiB） |
| 检查环境 | 主机 torch-cpu 即可（无需 NPU），秒级出报告 |

## 操作步骤（规划）

```bash
# 容器内（kvc_pd_offline 同步后）
bash scripts/run_offline_all.sh          # 一键七阶段：apply(01~10) → P/D/proxy → 双请求 → 归档落盘 → 容器内初检 → 撤补丁
python3 scripts/compare_fp.py            # v1 链裁决（同轮产物）
python3 scripts/check_kv_tensors.py --dir log/tensors --fp-logs log/ --out log/tensor_report   # 离线五级全检
# 主机侧回收
bash scripts/pull_tensors.sh             # md5 → manifest → tar → scp 导入本地 log/
```

## 判据速查（详见 docs/1 §4.4；首轮实测修正后）

```
[PASS] L0 结构对齐 ∧ L1 .pt 重算指纹 = 日志[FPB](1472/1472) ∧ L2 Tx 全 torch.equal(128/128)
      ∧ L3 尾槽差异 = exact 或重算一致(幅度判据: |Δ|max ≤ 5% 层幅值 且 无 NaN/Inf)
      ∧ decode 区数值健康
[FAIL] 其他（附取证包：层/块/K·V/tok/head/dim/xor_bits/位翻转谱/相对差分布）
```

| 裁决对象 | 判别 | 结果 |
|---|---|---|
| L2 Tx 区（传输区） | torch.equal 零容忍 | 不等 = 疑似传输损伤（FAIL + 位翻转谱刻画：EXPONENT/成片） |
| L3 尾槽（重算槽） | **数值语义**：\|Δ\|max ≤ 5% 层幅值 且无 NaN/Inf | 预期（首轮实测 0.02%~1.15%）；位谱仅作取证包（实测为分布式多 bit 翻转，非 ULP-only——bf16 尾数仅 7 位，低幅值元素 1~2 尾数位翻转的相对差天然大） |

> L1 链间互证是双链架构的第一道场景：正常轮里 .pt 重算指纹与日志 [FPB] 全等即互证成立（详见 docs/1 §2）。

## 与相邻工作区的关系

| 工作区 | 关系 |
|---|---|
| `../kvc_pd/` | v1 在线指纹链（判等 + 三轮 PASS）；**判定前提（分区间框架）与其共享**，本区为其第二证据链 |
| `../kvc/` | 01~08 基础打印补丁共用（apply 脚本跨区引用，不改 kvc） |
| `../kvc_pd_prefix/` | prefix 四象限实验（独立主题，无依赖） |
