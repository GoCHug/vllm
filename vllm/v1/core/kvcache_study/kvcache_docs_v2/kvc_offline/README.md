# kvc_offline —— 单机 PP2×TP2 KVCache block 原样张量归档与离线对比工作区（v3）

> **回答的问题**：服务形态与 `../kvc/` 完全相同（**单实例 `--tensor-parallel-size 2 --pipeline-parallel-size 2` 占满 4 卡**，无 PD 分离、无 proxy、直发 :8000）。在此形态上把 TERM 快照的 KV block **原样物理张量**（不 gather、每块 `(128, 4, 128)` 整存、含未写槽位）归档为 `.pt`，离线全量检查：
>
> 1. **分片覆盖**——4 worker（pp0/1 × tp0/1，各持 16 层/4 kv_heads）归档联合覆盖 32 层全量；
> 2. **前缀缓存驻留一致性（裁决核心）**——req_p TERM 的种块（[1,2]）与 req_r TERM 前缀 HIT 复用的同块**逐位 torch.equal**，证明缓存命中复用零篡改；
> 3. **双链互证**——.pt 重算 09 同口径 sha256 与日志 [FPB] 集合匹配（单机无传输，故无 PD Tx 对账——那是 kvc_pd/kvc_pd_offline 的主题）。
>
> **状态**：**✅ v3 已实施，首轮（2026-09-30 14:08 gggtest）PASS**——C1 集合互证 **1152/1152** 全消解、C2 前缀缓存驻留 **4 worker × 64/64 torch.equal**（HIT 块复用零篡改）、残值槽全零；响应文本与 kvc 基线/PD 两轮逐字一致。方案见 [`docs/1_pp2tp2_block_archive_design.md`](docs/1_pp2tp2_block_archive_design.md)，实验记录见 [`docs/2_pp2tp2_experiment_record.md`](docs/2_pp2tp2_experiment_record.md)。
> 首轮落地要点：PP 层号本地序重号（4 worker [FPB] 同 L## 交织，集合匹配消解）；单机无补算故 C2 零 diff（对照 PD 尾槽 ULP 差——"bootstrap 补算"是 PD 特有行为）。

## 目录结构

```
kvc_offline/
├── README.md                                本文档
├── docs/
│   └── 1_pp2tp2_block_archive_design.md     方案设计（单机 PP2×TP2 语义 + block 原样 schema + 四级检查）
├── patch/                                   11 号原样归档补丁 + 三段式 apply/revert
│   ├── 11_pp2tp2_block_dump.patch           TERM block 原样归档（叠 09 上；+104 行单 hunk，roundtrip 逐字节还原验证过）
│   ├── apply_kvc_offline_patches.sh         一键应用：kvc 01~08 → 09 → 11（[KVB] 标记 + py_compile 终验）
│   └── revert_kvc_offline_patches.sh        一键撤销（反向 11 → 09 → 01~08）
├── scripts/
│   ├── run_all.sh                           一键六阶段（含 KVC_DUMP_BLOCKS=1 导出）
│   ├── start.sh                             启动服务（同 ../kvc/：TP2+PP2 单实例 -> log/llama-3-8b.log）
│   ├── stop.sh                              杀服务（同 ../kvc/）
│   ├── curl_pr.sh                           P/R 双请求直发 :8000 + 三段轨迹拆解 + [KVB] 留痕
│   ├── check_kv_blocks.py                   四级检查器（C0 分片覆盖/C1 集合互证/C2 缓存驻留/C3-C4 健康；--selftest）
│   └── pull_tensors.sh                      归档回收（容器 pack / 主机 fetch + md5 校验）
└── log/                                     （首轮全产物已归档）
    ├── tensors/kv_S{00,01,10,11}_{1,2}_{rid8}.pt ×8 + manifest.json
    ├── block_report.md / block_report.json
    └── llama-3-8b.log / kvc_{startup,p,r}.log / kvb_archive_lines.log / resp_*.json
```

## 环境信息

| 项 | 值 |
|---|---|
| 容器 | gggtest（itask 4×hpu910a3） |
| 服务 | 单实例 vllm serve llama-3-8b：`--enforce-eager --tensor-parallel-size 2 --pipeline-parallel-size 2` |
| worker 分片 | 4 worker = 2 PP stage（层 0~15 / 16~31）× 2 TP rank（kv_heads 8→4） |
| 归档触发 | TERM（`_kvc_kv_dump` 收尾，每 worker 进程独立执行、每请求 4 份） |
| 单轮规模 | 8 个 .pt（4 worker × 2 请求）+ manifest |
| 检查环境 | 主机 torch-cpu 即可（无需 NPU） |

## 操作步骤

```bash
# 容器内（kvc_offline 同步后）
setsid nohup bash scripts/run_all.sh > log/run_all_screen.log 2>&1 &
# 回收
bash scripts/pull_tensors.sh pack      # 容器内
bash scripts/pull_tensors.sh fetch     # 主机侧（经隧道 5557）
# 离线四级检查（本机）
python3 scripts/check_kv_blocks.py --dir log/tensors --logs log --out log/block_report
```

## 判据速查（详见 docs/1 §5）

```
[PASS] C0 结构+分片覆盖（4 worker × 2 seq 齐, layer_ids 拼合 0~31, 各 worker 块表一致）
     ∧ C1 集合互证全消解（.pt 重算 09 哈希 vs 日志 [FPB]/[FP] 4-way 候选集）
     ∧ C2 前缀缓存驻留一致（req_p↔req_r 公共块 [1,2] 同 worker 同层 torch.equal）
     ∧ C3 全量无 NaN/Inf
[FAIL] 其他（附取证包: 层/块/worker/K·V/tok/head/dim/xor_bits/位翻转谱/残值统计）
```

| 检查对象 | 判别 | 结果 |
|---|---|---|
| C1 集合匹配 | 4 worker [FPB] 交织（PP 层号重号 ×2、TP 切片 ×2 → 同 ltag 候选 4 条） | 消解失败 = .pt 与日志互证断裂 |
| C2 公共块 | HIT 块不重算不覆写 → seq1/seq2 归档逐位相等 | 不等 = 缓存驻留期间被写入（复用污染／驱逐重算异常） |
| C4 残值槽 | 原样整存独有：块内 cov 之后槽位实测值 | 仅报告（对比分配零填充假设） |

## 与相邻工作区的关系

| 工作区 | 关系 |
|---|---|
| `../kvc/` | **同服务形态基线**（start.sh 同参数）；01~08 打印补丁共用；请求体共用 |
| `../kvc_pd/`、`../kvc_pd_offline/` | PD 分离主题（TP1）——Tx 传输判等是它们的裁决；本区单机无传输，裁决换为缓存驻留一致性 |
