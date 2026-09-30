# 2. 张量归档实验记录（2026-09-30 10:50 gggtest 容器，P3 首轮）

> **结论：PASS（双证据链同轮互证）**——v1 指纹链 448/448 Tx 全等；v2 张量链五级全过：**L1 两链互证 1472/1472 条全等**（.pt 重算 sha256 = 日志 [FPB]）∧ **L2 Tx 区 128/128 对 torch.equal** ∧ L3 尾槽差异全部落重算带（0.02%~1.15% 层幅值）∧ L4 decode 区数值健康。张量链首次落地即兑现设计目标：**判据之外，差异的"形态"也可报告了**。

## 2.1 实验轮概览

| 项 | 值 |
|---|---|
| 时间 | 2026-09-30 10:50:26 ~ 10:52:43（全程 2 分 17 秒） |
| 容器 | gggtest（itask 4×hpu910a3，P=卡0/8100/producer，D=卡1/8200/consumer，proxy=8000） |
| 补丁链 | kvc 01~08（170 行 [KVC]）+ 09 指纹 + **10 张量归档（本轮首跑）** |
| 开关 | `KVC_DUMP_TENSORS=1` `KVC_DUMP_DIR=log/tensors`（随 run_offline_all.sh 导出） |
| 双请求 | req_p（324 tok, max_tokens=1, 纯传输探针）/ req_r（486 tok, max_tokens=35, 前缀命中+decode 探针） |
| 七阶段 | patch → P 就绪(50s) → D 就绪(40s) → proxy+双请求 → 归档落盘 → 容器内初检 → revert（源码归零验证过） |

## 2.2 归档产物（log/tensors/，manifest md5 全过）

| 文件 | side/seq | request_id 尾 8 | p_tok/w_tok | block_table / cov | 大小 / md5 |
|---|---|---|---|---|---|
| kv_P_1_bcd07435.pt | P seq1 | bcd07435 | 324/324 | [1,2,3] / [128,128,68] | 40.5 MiB / 0b71f340…9de03a |
| kv_D_1_aa9a175b.pt | D seq1 | aa9a175b | 324/324 | [1,2,3] / [128,128,68] | 40.5 MiB / 85774397…551d7e |
| kv_P_2_81a351d5.pt | P seq2 | 81a351d5 | 486/486 | [1,2,4,5] / [128,128,128,102] | 60.8 MiB / 45d4c223…c083cfe2 |
| kv_D_2_87786813.pt | D seq2 | 87786813 | 486/520 | [1,2,4,5,6] / [128,128,128,128,8] | 65.0 MiB / d0e17541…389c923a |

- meta 与 v1 日志完全对齐：P₂/D₂ 的 bloc 布局差异（P 缺 blk6、blk5 只到 102 槽）如实反映"P 不做 decode"；D₂ blk5 满块（102 prompt + 26 decode 写满）+ blk6[8]（34 decode 尾槽）与 kvc_pd v1 记录的块语义一致。
- 归档行为双侧日志原样留痕（kvt_archive_lines.log，[KVT] 行号：p_reqp:137 / d_reqp:158 / p_reqr:147 / d_reqr:812）：

```
INFO 09-30 10:52:00 [model_runner_v1.py:2687] [KVC][KVT] TERM P s1 张量归档(与指纹同点双证据): layers=32 w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] -> log/tensors/kv_P_1_bcd07435.pt (后台落盘)
INFO 09-30 10:52:00 [model_runner_v1.py:2687] [KVC][KVT] TERM D s1 张量归档(与指纹同点双证据): layers=32 w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] -> log/tensors/kv_D_1_aa9a175b.pt (后台落盘)
INFO 09-30 10:52:09 [model_runner_v1.py:2687] [KVC][KVT] TERM P s2 张量归档(与指纹同点双证据): layers=32 w_tok=486 p_tok=486 block_table=[1, 2, 4, 5] cov=[128, 128, 128, 102] -> log/tensors/kv_P_2_81a351d5.pt (后台落盘)
INFO 09-30 10:52:10 [model_runner_v1.py:2687] [KVC][KVT] TERM D s2 张量归档(与指纹同点双证据): layers=32 w_tok=520 p_tok=486 block_table=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8] -> log/tensors/kv_D_2_87786813.pt (后台落盘)
```

## 2.3 五级检查结果（tensor_report.json 摘录）

| 级 | 检查 | 结果 |
|---|---|---|
| L0 | 结构不变量×4 文件 + P/D p_tok 对账 | 全过（sum(cov)=w_tok、len(K)=layers、shape/dtype 实测=声明） |
| L1 | **两链互证**：.pt 重算 09 同口径 sha256 vs 日志 [FPB]/[FP] | **1472/1472 全等**（P₁320+D₁320+P₂384+D₂448）——同一 TERM 快照的双链互证成立：日志链与归档链任何一侧损坏/篡改都会在此暴露 |
| L2 | **Tx 逐位**：32 层×K/V torch.equal | **128/128 对全等**（seq1: Tx=323 行 ×64；seq2: Tx=485 行 ×64）——mooncake 传输逐位无损的张量级复证 |
| L3 | 尾槽（行 p_tok-1）重算一致 + decode 区健康 | 128 组：exact 6 / **recompute 122**（\|Δ\|max 带 0.02%~1.15% 层幅值，例 L00 V 3 元素 \|Δ\|0.0020 → L27 K 891 元素 \|Δ\|0.0176）；decode 34 行无 NaN/Inf |
| L4 | 报告落盘 | tensor_report.md/.json（本地/容器各一份，同轮同产物） |

对照 v1（同轮 compare_fp.py）：req_p 192/192 + req_r 256/256 Tx 全等 → 两链**同轮独立**得出一致裁决。

## 2.4 本轮科学发现：重算差的真实位谱（修正 L3 判据）

首跑时 L3 曾按设计文档原判据（位模式 ULP-only：xor≤2 且翻转落尾数低位）误报 122 条 FAIL。取证（张量链独有的能力——直接读尾槽元素）表明这是**判据设计错误而非传输损伤**：

- 尾槽元素 P/D 的 \|Δ\| 有界：max=0.0625 ≈ **0.43%×层幅值 14.56**；64 组中位相对差 0.47%；P 与 D 的数值分布同构（幅值域/极值一致，如 L03 K：P∈[0.0005,14.56] vs D∈[0.0003,14.56]，最大差处 P=-1.3906 vs D=-1.3750）。
- 但**位翻转谱**远超 ULP：{'ULP': 400~500, 'MANHIGH': 200~330, 'EXPONENT': 20~66, 'SIGN': 0~2}（元素总量 1024/组）——低幅值元素在 bf16 仅 7 位尾数下，1~2 尾数位的相对差天然巨大（0.0005 级元素翻 1 个尾数位 → 相对差 ~50%），多位翻转也常见。

**结论**：同一数学量经不同 kernel 路径（P 批量 prefill vs D 单 token 补算）的合法重算差，其判据应是**数值语义**（绝对差有界 + 双侧分布同构），而非位模式（ULP-only）。已将 L3 判据修正为：`|Δ|max ≤ 5% 层幅值 且 无 NaN/Inf` → recompute（预期）；位翻转谱降级为取证包内容。位模式判据保留给 **Tx 区**（L2 零容忍 + EXPONENT/成片签名刻画损伤形态）。selftest 双场景（正常重算差→PASS、DMA 3bit 损伤→FAIL）修正后仍全过。

> 这是双证据链的第一次实证价值：指纹链只能判"该层不等"（Xx 差 122 条），张量链直接给出"差多少、长什么样、为什么合法"——判据修正本身由张量数据驱动。

## 2.5 修复记录（本轮落地）

| 问题 | 修复 |
|---|---|
| L1 日志解析 0 条（setdefault 嵌套层级错——层字典插入 d 顶层而非 d["layers"]） | check_kv_tensors.py parse_trajs 修正；supertest 补充对账断言 |
| L3 位模式判据误报（见 §2.4） | 判据改数值语义（幅度 5% 阈值）；设计文档 §4.4 判别表同步修正 |
| 10 号 patch 函数体草稿残留（`VD_inj` 悬空行/int `.to()` 笔误） | staging 重生成，roundtrip + py_compile 验证后落盘 |

## 2.6 复现与文件索引

```
# 容器内（需 kvc/kvc_pd/kvc_pd_offline 三工作区就位, 见 README）
cd .../kvc_pd_offline && setsid nohup bash scripts/run_offline_all.sh > log/run_offline_screen.log 2>&1 &
# 回收: 容器 bash scripts/pull_tensors.sh pack | 主机 bash scripts/pull_tensors.sh fetch
# 离线检查(本机 torch-cpu 即可): python3 scripts/check_kv_tensors.py --dir log/tensors --logs log --out log/tensor_report
```

| 产物 | 说明 |
|---|---|
| log/tensors/*.pt ×4 + manifest.json | 207 MiB 归档（md5 双侧一致） |
| log/tensor_report.{md,json} | 五级检查报告（verdict=PASS） |
| log/verdict.txt | v1 指纹裁决（同轮，PASS） |
| log/kvc_{p,d}_{reqp,reqr}.log | 双侧 [KVC] 轨迹（含 [KVT] 归档行） |
| log/{p,d}_llama.log / proxy.log / curl_* / resp_* | 服务日志与请求响应（\r 已规范化） |
| log/run_offline_screen.log / kvt_archive_lines.log | 七阶段主控输出 / 归档行为留痕 |
