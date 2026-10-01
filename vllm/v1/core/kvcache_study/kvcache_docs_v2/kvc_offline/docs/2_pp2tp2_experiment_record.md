# 2. 单机 PP2×TP2 block 原样归档实验记录（2026-09-30 14:08 gggtest 容器，R3 首轮）

> **结论：PASS（分片覆盖完整 + 缓存驻留零篡改 + 双链互证成立）**——C1 集合互证 **1152/1152 全消解**（8 文件：4 worker × s1 112 + s2 176）∧ C2 前缀缓存驻留 **4 worker × 64/64 对 torch.equal**（req_p 种块 [1,2] 与 req_r 前缀 HIT 复用同块逐位一致）∧ decode 区 4×34 行健康 ∧ **残值槽全零**（61440 槽×8 K/V 路全零）。单机 PP2×TP2 混合并行（与 ../kvc/ 同形态）下：**4 worker 的 KV 分片归档联合无缺口覆盖 32 层全量；前缀缓存命中复用不改写任何数值**。
>
> 背景：本区初版按 PD 双实例理解实现，经澄清服务形态（"启动的服务跟 kvc 一样"——单实例 `--tensor-parallel-size 2 --pipeline-parallel-size 2`，4 卡混布无 PD 分离）后**全部重实现为 v3**；初版产物已废弃清除。

## 2.1 实验轮概览

| 项 | 值 |
|---|---|
| 时间 | 2026-09-30 14:08:15 ~ 14:10:05（全程 1 分 50 秒；六阶段） |
| 容器 | gggtest（itask 4×hpu910a3；会话中途 pod 曾被回收 Stopped，`itask start` 保环境重启，workdir 完整） |
| 服务 | **单实例 vllm serve llama-3-8b：`--enforce-eager --tensor-parallel-size 2 --pipeline-parallel-size 2`**（同 ../kvc/scripts/start.sh 参数；60s 就绪） |
| 补丁链 | kvc 01~08（170 行 [KVC]）+ 09 指纹 + **11 block 原样归档 v3（本轮首跑）** |
| 开关 | `KVC_DUMP_BLOCKS=1` `KVC_DUMP_DIR=log/tensors` |
| 双请求 | req_p（324 tok, max_tokens=1）/ req_r（486 tok, max_tokens=35）——拷贝自 ../kvc/ |
| 响应 | req_p 生成"为了"（completion=1）、req_r 35 tokens（"://www.zhihu…"）——**与 kvc TP1 基线及 PD 两轮逐字一致**（seed=1024 跨全部拓扑稳定） |
| 收尾 | 六阶段全过：源码逐字节归零、零进程残留 |

## 2.2 归档产物（log/tensors/ 8 文件 = 128 MiB，manifest md5 双侧一致）

| 文件模式（×4 worker） | seq | p_tok/w_tok | block_table / cov | 单文件大小 |
|---|---|---|---|---|
| kv_{S00,S01,S10,S11}_1_b9c6961a.pt | 1 | 324/324 | [1,2,3] / [128,128,68] | 12.0 MiB |
| kv_{S00,S01,S10,S11}_2_bae2bb90.pt | 2 | 486/**520** | [1,2,4,5,6] / [128,128,128,128,8] | 20.0 MiB |

meta 关键事实（相对 PD 轮的形态差异）：

- **每 worker 16 层**（kv_heads=4）：PP2 把 32 层切为 stage0[0~15]/stage1[16~31]，TP2 再切 kv_heads 8→4——单文件是 16 层 × 4 heads 的分片，**4 文件联合才是全量**；
- **layer_ids 全部为本地序 0~15**（pp1 的全局层 16~31 在 kv_caches 枚举中也是 0~15）——两个 PP stage 的 [FPB] L 标签在日志中**重号**（L00~L15 各出现 4 次：2 PP × 2 TP），这正是 C1 必须用集合匹配的原因（设计文档 §5 预判兑现）；
- **4 worker 块表完全一致**（blk [1,2,3] / [1,2,4,5,6]）——单机块池由 EngineCore 全局调度，各分片同构复制；
- cov 形态与 PD 轮同构：req_r 的 w_tok=520（486 prompt + 34 decode 输入槽），blk5 满块（102+26）+ blk6[8]。

[KVB] 归档行为留痕（kvb_archive_lines.log，8 行全；每 worker 独立 TERM 触发，时间同秒）：

```
INFO 09-30 14:09:21 [model_runner_v1.py:2702] [KVC][KVB] TERM pp0 tp0 s1 block原样归档(与指纹同点双证据): layers=16[0~15] w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_S00_1_b9c6961a.pt (后台落盘)
INFO 09-30 14:09:21 [model_runner_v1.py:2702] [KVC][KVB] TERM pp0 tp1 s1 block原样归档(与指纹同点双证据): layers=16[0~15] w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_S01_1_b9c6961a.pt (后台落盘)
INFO 09-30 14:09:21 [model_runner_v1.py:2702] [KVC][KVB] TERM pp1 tp1 s1 block原样归档(与指纹同点双证据): layers=16[0~15] w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_S11_1_b9c6961a.pt (后台落盘)
INFO 09-30 14:09:21 [model_runner_v1.py:2702] [KVC][KVB] TERM pp1 tp0 s1 block原样归档(与指纹同点双证据): layers=16[0~15] w_tok=324 p_tok=324 block_table=[1, 2, 3] cov=[128, 128, 68] kv_heads=4 -> log/tensors/kv_S10_1_b9c6961a.pt (后台落盘)
INFO 09-30 14:09:29 [model_runner_v1.py:2702] [KVC][KVB] TERM pp0 tp0 s2 block原样归档(与指纹同点双证据): layers=16[0~15] w_tok=520 p_tok=486 block_table=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8] kv_heads=4 -> log/tensors/kv_S00_2_bae2bb90.pt (后台落盘)
（pp0tp1 / pp1tp0 / pp1tp1 对称同构，完整 8 行见 log/kvb_archive_lines.log）
```

## 2.3 四级检查结果（block_report.json；容器/本地两次跑同 verdict=PASS）

| 级 | 检查 | 结果 |
|---|---|---|
| C0 | 结构 + 分片覆盖 | 8 文件不变量全过；4 worker 齐备（{pp0,pp1}×{tp0,tp1}）；同 seq 各 worker p_tok/cov/block_table 一致；layer_ids 拼合覆盖 0~31 |
| C1 | **两链互证（集合匹配）** | **1152/1152 全消解**（每 worker：s1 112 条 + s2 176 条 = 块指纹 Tx/Xx + 层指纹；候选池同 L## 4 条——PP 重号 × TP 切片——各 worker 的 .pt 重算哈希各自消解一个，零剩余） |
| C2 | **前缀缓存驻留一致性（裁决核心）** | **4 worker × 64/64 对 torch.equal**：公共块 [1,2]（req_p 种入 → req_r 前缀 HIT 复用）在每 worker 16 层 × K/V 下逐位相等——**缓存命中复用零篡改**（HIT 块不重算不覆写，驻留期间（释放挂队尾 + LRU + 哈希挂链）数值不变） |
| C3/C4 | 健康 + 统计 | 全 8 文件 NaN/Inf=0；decode 区 4×34 行健康（各 worker first3 各异——分片数据独立正常）；**残值槽全零**（每 worker K/V 各 61440 未写槽 nonzero=0——NPU 池块分配即零填充） |

cf. 跨实验对照（同输入 req_p/req_r、同判据框架）：

| 指标 | kvc_pd_offline（PD TP1） | kvc_offline v3（单机 PP2×TP2） |
|---|---|---|
| 裁决主题 | L2 Tx 传输判等 128/128 | C2 缓存驻留判等 256/256（64×4 worker） |
| 链间互证 | L1 1472/1472 | C1 1152/1152 |
| 尾槽/重算差 | 0.02%~1.15%（PD 补算覆写） | **无**（单机无补算——req_p 的尾 token 在本机前向即最终值，无跨实例 kernel 路径差） |
| 残值槽 | 全零 | 全零 |
| 响应文本 | "为了" + 35 tok | 逐字一致 |

**单机语境的独特发现**：C2 全等且**无任何重算差**——PD 架构里 D 侧尾槽单 token 补算造成的 ULP 级差在单机下不存在（req_r 的末 prompt token 与增量 token 由同一实例同一 kernel 路径一次前向完成），全 256 对比较零 diff。这是"PD bootstrap 补算"是 PD 特有行为（而非 vLLM 通用重算）的又一张量级佐证。

## 2.4 重实现说明（v1 误向 → v3 澄清）

| 项 | 初版（误向，已废弃） | **v3（本轮）** |
|---|---|---|
| 服务理解 | P=TP2 + D=TP2 双实例 PD 分离 | **单实例 TP2+PP2 混合并行（同 kvc/）** |
| 归档身份 | kv_{P,D}{rank}（kv_role+tp_rank） | **kv_S{pp}{tp}（get_pp/tp_group）** |
| 裁决语义 | B2 同 rank 对 Tx 判等（传输） | **C2 缓存驻留判等（无传输）** |
| 日志挑战 | TP2 双 rank 交织 | **PP 层号重号 × TP 切片 = 同 L## 4 候选** |
| 产物 | 8 文件 217 MiB（PD 语料） | 8 文件 128 MiB |

初版全部产物（patch/脚本/docs/log）已删除重写；本记录仅对应 v3。

## 2.5 复现与文件索引

```
# 容器内(需 kvc/kvc_pd/kvc_offline 三工作区就位; 单实例服务无 proxy)
cd .../kvc_offline && setsid nohup bash scripts/run_all.sh > log/run_all_screen.log 2>&1 &
# 回收: 容器 bash scripts/pull_tensors.sh pack | 主机 bash scripts/pull_tensors.sh fetch
# 离线四级检查(本机 torch-cpu): python3 scripts/check_kv_blocks.py --dir log/tensors --logs log --out log/block_report
```

| 产物 | 说明 |
|---|---|
| log/tensors/kv_S??_?_*.pt ×8 + manifest.json | 128 MiB 原样归档（md5 双侧一致） |
| log/block_report.{md,json} | 四级检查报告（verdict=PASS） |
| log/kvb_archive_lines.log | [KVB] 归档行为 8 行留痕（pp/tp/s/块表/文件对应） |
| log/kvc_{startup,p,r}.log | 三段 [KVC] 轨迹（四 worker 交织；C1 集合匹配数据源） |
| log/llama-3-8b.log / resp_*.json / curl 留痕 / run_all_screen.log | 服务日志与请求响应（\r 已规范化 9 处） |
