# 2. pp2tp2 混布 block 原样归档实验记录（2026-09-30 12:18 gggtest 容器，Q3 首轮）

> **结论：PASS（pp2tp2 双 rank 各自传输无损 + 块原样双链互证）**——B1 集合互证 **4864/4864 全消解**（.pt 重算 09 同口径哈希逐 rank 匹配日志候选）∧ B2 同 rank 对 **256/256 对 torch.equal**（P0↔D0、P1↔D1 各 128 对 Tx 逐位相等）∧ B3 尾槽差异全落重算带（0.02%~1.92% 层幅值）∧ decode 区双 rank 健康 ∧ **残值槽全零**（原样归档独有取证）。pp2tp2 拓扑首轮即验证：**TP 维 kv_heads 切分（8→4/rank）下的 mooncake 块传输同为逐位无损**。

## 2.1 实验轮概览

| 项 | 值 |
|---|---|
| 时间 | 2026-09-30 12:18:22 ~ 12:21:14（全程 2 分 52 秒） |
| 容器/拓扑 | gggtest 4×hpu910a3：**P=TP2(卡0,1/8100) + D=TP2(卡2,3/8200) + proxy(8000) 混布** |
| 补丁链 | kvc 01~08（170 行 [KVC]）+ 09 指纹 + **11 block 原样归档（本轮首跑）** |
| 开关 | `KVC_DUMP_BLOCKS=1` `KVC_DUMP_DIR=log/tensors` |
| 双请求 | req_p（324 tok, max_tokens=1）/ req_r（486 tok, max_tokens=35）——与 TP1 轮同输入 |
| 就绪 | P 12:18:52（`vllm-0.23.0-tp2` 指纹，双 rank 物理池横幅×2 ✓）；D 60s 就绪（横幅×2 ✓） |
| 输出一致性 | req_p 生成"为了"、req_r 35 tokens——与 kvc_pd/kvc_pd_offline TP1 实验逐字一致（seed=1024 跨拓扑稳定） |
| 收尾 | 七阶段全过：源码逐字节归零、零进程残留 |

## 2.2 归档产物（log/tensors/ 8 文件 ≈ 217 MiB，manifest md5 双侧一致）

| 文件 | rank/seq | p_tok/w_tok | block_table / cov | 大小 / md5 |
|---|---|---|---|---|
| kv_P0_1_86535257.pt / kv_P1_1_*.pt | P r0/r1 s1 | 324/324 | [1,2,3] / [128,128,68] | 24.1 MiB ×2 |
| kv_P0_2_816242b6.pt / kv_P1_2_*.pt | P r0/r1 s2 | 486/486 | [1,2,4,5] / [128,128,128,102] | 32.1 MiB ×2 |
| kv_D0_1_9b2b9e4f.pt / kv_D1_1_*.pt | D r0/r1 s1 | 324/324 | [1,2,3] / [128,128,68] | 24.1 MiB ×2 |
| kv_D0_2_abc09176.pt / kv_D1_2_*.pt | D r0/r1 s2 | 486/**520** | [1,2,4,5,6] / [128,128,128,128,8] | 40.1 MiB ×2 |

meta 三项跨拓扑对比（vs TP1 kvc_pd_offline 轮）：

- **kv_heads=4**（8/TP2）——模型结构不变、TP 切分生效；
- **block_table/cov 与 TP1 完全同构**（P [1,2,4,5] 展平 486、D [1,2,4,5,6] 延伸 520）——**块池编排跨 rank 对称**（P0=P1 同表、D0=D1 同表），term 语义逐位对齐 TP1（可靠证据：块池调度在 TP2 下仍按 rank 独立同构复制）；
- **P/D 两侧 request_id 尾 8 不同**（P:86535257/816242b6 vs D:9b2b9e4f/abc09176）——与 TP1 同一 proxy 改写模式（P 得哑请求 id，D 得原 id）。

[KVB] 归档行为留痕（kvb_archive_lines.log，8 行全；原始日志行号：p_llama.log:720,721,981,982 / d_llama.log:742,743,1673,1674——**Worker_TP0/Worker_TP1 双进程前缀**（pid 81657/81658 与 82682/82683）直接佐证 TP2 双 worker 进程结构）：

```
INFO 09-30 12:20:28 [model_runner_v1.py:2702] [KVC][KVB] TERM P r1 s1 block原样归档(与指纹同点双证据): layers=32 w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_P1_1_86535257.pt (后台落盘)
INFO 09-30 12:20:28 [model_runner_v1.py:2702] [KVC][KVB] TERM P r0 s1 block原样归档(与指纹同点双证据): layers=32 w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_P0_1_86535257.pt (后台落盘)
INFO 09-30 12:20:39 [model_runner_v1.py:2702] [KVC][KVB] TERM P r0 s2 block原样归档(与指纹同点双证据): layers=32 w_tok=486 p_tok=486 block_table=[1, 2, 4, 5] cov=[128, 128, 128, 102] kv_heads=4 -> log/tensors/kv_P0_2_816242b6.pt (后台落盘)
INFO 09-30 12:20:39 [model_runner_v1.py:2702] [KVC][KVB] TERM P r1 s2 block原样归档(与指纹同点双证据): layers=32 w_tok=486 p_tok=486 block_table=[1, 2, 4, 5] cov=[128, 128, 128, 102] kv_heads=4 -> log/tensors/kv_P1_2_816242b6.pt (后台落盘)
INFO 09-30 12:20:30 [model_runner_v1.py:2702] [KVC][KVB] TERM D r1 s1 block原样归档(与指纹同点双证据): layers=32 w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_D1_1_9b2b9e4f.pt (后台落盘)
INFO 09-30 12:20:30 [model_runner_v1.py:2702] [KVC][KVB] TERM D r0 s1 block原样归档(与指纹同点双证据): layers=32 w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_D0_1_9b2b9e4f.pt (后台落盘)
INFO 09-30 12:20:40 [model_runner_v1.py:2702] [KVC][KVB] TERM D r1 s2 block原样归档(与指纹同点双证据): layers=32 w_tok=520 p_tok=486 block_table=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8] kv_heads=4 -> log/tensors/kv_D1_2_abc09176.pt (后台落盘)
INFO 09-30 12:20:40 [model_runner_v1.py:2702] [KVC][KVB] TERM D r0 s2 block原样归档(与指纹同点双证据): layers=32 w_tok=520 p_tok=486 block_table=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8] kv_heads=4 -> log/tensors/kv_D0_2_abc09176.pt (后台落盘)
```

## 2.3 五级检查结果（block_report.json 摘录，容器/本地两次跑同 verdict）

| 级 | 检查 | 结果 |
|---|---|---|
| B0 | 结构不变量×8 + 跨侧 p_tok + 块表同构报告 | 全过（seq2 双 rank 的 cov/block_table 异构按预期记录：P 停在 486，D 延伸到 520——decode 只在 D 侧） |
| B1 | **两链互证（集合匹配法）** | **4864/4864 全消解**（D0s1/D1s1:512×2、D0s2/D1s2:768×2、P0s1/P1s1:512×2、P0s2/P1s2:640×2——每 rank 的 .pt 重算哈希各自在日志候选集消解一个匹配，TERM 行 (rid, dev) 双 rank 旁证齐备） |
| B2 | **同 rank 对 Tx 逐位**（行号按各自 block_table+cov 重建后切公共区间） | **256/256 全等**：s1 r0/r1 各 64 对（Tx=323 行）、s2 r0/r1 各 64 对（Tx=485 行）——**P0↔D0、P1↔D1 跨实例传输逐位无损** |
| B3 | 尾槽（行 p_tok-1，双 rank×2 请求×32 层×K/V=256 组） | exact 52 / recompute 204（\|Δ\|max 带 **0.02%~1.92%** 层幅值；TP1 轮为 0.02%~1.15%——TP2 重算差幅度同量级） |
| B4 | decode 区 + **残值槽统计（原样归档独有）** | decode 34 行×双 rank健康（r0 first3 [-5.375,1.164,-1.539]、r1 [0.953,-0.139,-0.412]——rank 间独立正常）；**残值槽全零**（s1 30720 槽/s2 61440 槽 K·V nonzero=0：块内 cov 之后的未写槽位实测为零值） |

**残值槽全零的取证价值**：原样整块存储（含未写槽位）首次给出"分配后未写区域"的实际内存值——NPU 池块分配时即为零填充（非随机残值/跨请求脏数据）。这对"块复用是否污染跨请求数据"这一类问题是唯一证据形态（gather 展平在 cov 处截断、根本看不到这些槽位）。

## 2.4 本轮方法论发现：TP2 下的对账适配（两项）

**① v1 compare_fp 在 TP2 的 FAIL 是解析局限所致（非传输错误）**。初检段日志里该工具报 `TX-MISMATCH`——机理：09 的 [FPB]/[FP] 行**不带 dev 标签**、TP2 双 rank 的行在同一日志文件交织，compare_fp 的 `d[layer].setdefault(...)` 逐层覆盖使其实际比较了"P 侧最后写入的 rank" vs "D 侧最后写入的 rank"（跨 rank 的不同 kv_heads 切片）→ 必然不等。修复面在解析侧而非传输侧：B1 用**集合匹配法**（每层每块收集双 rank 候选构成多重集合，对每个 rank 的 .pt 重算哈希消解匹配一个——rank0/rank1 是 kv_heads 不同切片、数据必然不同，匹配唯一可靠）；TERM 概览行（带 dev=npu:{r}）作 (request_id, rank) 存在性旁证。**设计文档 §4.4 预判（B1 集合匹配保留给 TP2、B2 Tx 逐位用于损伤取证）在真实数据上兑现。**

**② compare_fp 的裁决以 B1+B2 为准**——同一轮实验中 B1 4864/4864 + B2 256/256 的张量级证据直接覆盖 compare_fp 在 TP1 场景下的全部职能（这一atz设计已在 run_pp2tp2 脚本的初检段注释里预告，本轮按判读执行）。

## 2.5 修复记录（本轮实施过程，均已测试后落盘）

| 问题 | 修复 |
|---|---|
| B0 断言对 req_r 误判（cov 不等是"P 不做 decode"的预期形态、非错误） | 就地放宽为只断 p_tok，异构 cov/块表改为信息记录——这是从 kvc_pd_offline（v2 检查器）复制检查思路时套用了 TP1 的“cov 相同”前提 |
| selftest 构造 bug 四处（P/D 同 rank 不共享语料、块张量未按 BS 整块填充导致 shape 不符、rank1 块号 [11+1,12] 碰撞、场景 2 改内存不落盘） | 全数修复；selftest 两场景（正常+重算差→PASS、Tx 损伤→B1/B2 双层 FAIL）全过且首轮真实数据验证 |
| B1 消费端 side 大小写（meta 大写 P/D vs 日志文件名小写 p/d）导致 B1 全跳过 | 修 skey=side.lower()——selftest 从"S1 通过但 B1 0 条"变为 24/24（真实数据显示 512/768/640 条） |
| curl_pp2.sh 的 dev 标签 grep 模式（`dev=npu:0,`）与真实日志形态（`dev=npu:0 逐层按块:`）不匹配 | 改 grep -E `"dev=npu:${r}(,| |$)"`，分 rank 轨迹文件正确落盘 |

修复启示：selftest 合成数据用了“int. 序列标识行尾”的自造日志形态，真实日志是“`dev=npu:0 逐层...`”——合成样例的行尾形态要贴真实格式或正则做成形态鲁棒。

## 2.6 复现与文件索引

```
# 容器内(需 kvc/kvc_pd/kvc_offline 三工作区就位)
cd .../kvc_offline && setsid nohup bash scripts/run_pp2tp2_all.sh > log/run_pp2tp2_screen.log 2>&1 &
# 回收: 容器 bash scripts/pull_tensors.sh pack | 主机 bash scripts/pull_tensors.sh fetch
# 离线检查(本机 torch-cpu): python3 scripts/check_kv_blocks.py --dir log/tensors --logs log --out log/block_report
```

| 产物 | 说明 |
|---|---|
| log/tensors/kv_*.pt ×8 + manifest.json | 217 MiB 原样归档（块径 128 槽整存；md5 双侧一致） |
| log/block_report.{md,json} | 五级检查报告（verdict=PASS，本地/容器同) |
| log/kvb_archive_lines.log | [KVB] 归档行为 8 行留痕（含 rank/块表/文件对应） |
| log/kvc_{p,d}{0,1}_req{p,r}.log ×8 | 分 rank 物理轨迹（TERM dev 行） |
| log/kvc_{p,d}_req{p,r}.log ×4 | 共享轨迹（[FPB]/[FP] 双 rank 交织；B1 集合匹配数据源） |
| verdict.txt | v1 compare_fp 裁决（TP2 下仅参考——解析局限见 §2.4①） |
| {p,d}_llama.log / proxy.log / resp_*.json / curl_* / run_pp2tp2_screen.log | 服务日志与请求响应（\r 已规范化） |

## 2.7 上跨链对照（TP1 vs TP2 同输入同判据）

| 指标 | kvc_pd_offline（TP1, 10:50 轮） | **kvc_offline（TP2, 12:18 轮）** |
|---|---|---|
| 传输判等 | L2 128/128 | B2 **256/256**（双 rank ×64×2 请求） |
| 链间互证 | L1 1472/1472 | B1 **4864/4864** |
| 尾槽重算差幅度带 | 0.02%~1.15% | **0.02%~1.92%** |
| decode 健康 | 34 行无 NaN | 34 行×双 rank 无 NaN |
| 归档文件数/形态 | 4 文件（gather 展平） | **8 文件（块原样整存，含残值取证）** |
| 输出文本 | "为了" + 35 tokens | **逐字一致** |
