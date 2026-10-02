# kvc v2.3 实验分析与 E2E 记录（2026-10-02 15:16 轮）

> 本轮 = 08 号补丁 v2.3（横幅两行式日志版）完整 E2E：打补丁 → 起服务 → P/R 双请求 →
> TERM 物理KV原样归档（开始/完成保存横幅各 1 行/worker）→ 产物回收 → 杀服务 + revert。
> 全部结论均出自本轮产物（log/ + tensors/，rid 994a2ef5/81604d58），无历史遗留数据。
> 补丁设计见同目录 `3_kvc_v2_tensor_archive_design.md`。

## 1. 时间线（gggtest 容器内，UTC）

| 时刻 | 事件 |
|---|---|
| 15:16:18 | run_all 启动 |
| 15:16:5x | 8/8 补丁应用并验证（[KVC] 174 行 = vllm 155 + vllm-ascend 19；py_compile OK） |
| 14:43:5x | vllm serve 后台启动（KVC_SAVE_KV=1, KVC_SAVE_DIR=tensors） |
| 14:44:2x | [KVC][CFG/L1] 池初始化：num_blocks=13291，K/V=(13291,128,4,128) bf16 |
| 15:17:1x | 服务就绪（等待 50s；Worker pid=65595~65598 = npu:0~3） |
| 15:17:14 | curl P（324 tok, max_tokens=1）→ 响应 "为了"（finish=length） |
| 15:17:14 | P TERM：4 worker 各打 **1 行开始横幅 + 1 行完成横幅**（12.0 MiB/份, flush 32.1~32.7ms） |
| 15:17:15 | [KVC][L5] 释放横幅（kvc_p.log:55）：归还 [3,2,1] reversed 挂队尾 |
| 15:17:21 | curl R（486 tok, max_tokens=35） |
| 14:44:43 | R 前缀查找 HIT [1,2]；S3 新块 [4,5]；decode 第27步跨界 [6] |
| 15:17:21 | R TERM：4 worker 开始+完成横幅（20.0 MiB/份, flush 42~45ms） |
| 15:18:03 | 容器初检 DONE（selftest + 列表 + 四 worker 比对 PASS） |
| 15:19:xx | pack（88MB）→ fetch 回本地（md5 8/8 OK）→ stop + revert → 容器源码归零 |

实验窗口 ~2 分钟；服务 log 1185 行；**[KVS] 全部仅 20 行**（v2.1 为 532；v2.2 起 20 行，v2.3 横幅化）。

## 2. 环境快照

| 项 | 值 |
|---|---|
| Pod / 卡 | gggtest（a3, 4×hpu910a3, wsl02075301）@ gpuxdn010030015012.guian02 |
| 模型 / 基线 | Meta-Llama-3-8B bf16（32 层, kv_heads 8→4/TP2, block_size 128）；vllm @0fc695f + vllm-ascend @5cb98c（v0.23.0，git 干净打补丁） |
| 服务 | `vllm serve --enforce-eager -tp2 -pp2` 单实例占 4 卡；就绪 50s |
| worker | pp0tp0=npu:0/pid65595, pp0tp1=npu:1/65596, pp1tp0=npu:2/65597, pp1tp1=npu:3/65598 |
| KV 池 | num_blocks=13291；每 worker 每层 K/V 分离 (13291,128,4,128) bf16 |
| 落盘 | 8 归档 .pt + manifest.json；tar 88368766 B（md5 fcb081ffbb51dcb762688f7dcd0e3733） |

## 3. P/R 双请求块生命周期（log 拆解 + 归档互证）

```
P (324 tok, max_tokens=1)                     R (486 tok, max_tokens=35)
入队: 链式哈希 ×2                               入队: 链式哈希 ×3 (前 2 级与 P 完全相同)
前缀查找: MISS (冷池)                           前缀查找: HIT [1,2] hit=256
S3 新块 popleft_n [1,2,3]                       S2 touch [1,2]; S3 新块 [4,5]
S4 满块 [1,2] 双级哈希入缓存表                   S4 满块哈希维护
TERM 归档: [1,2,3] cov=[128,128,68]             decode 第27步跨界 [6] (num_tokens=513)
  SAVE + SAVED × 4 worker                       TERM 归档: [1,2,4,5,6] cov=[128,...,8]
L5 释放: [3,2,1] reversed 挂队尾                  SAVE + SAVED × 4 worker
                                              (w_tok=520=486+34; 第35个输出仅采样不写卡)
```

## 4. v2.3 横幅日志审计（本轮专属验证，全文存 log/inspect_savelog_audit.out）

| 审计项 | 结果 |
|---|---|
| 开始保存横幅 | 8/8（4 worker × P/R，grep '======== 开始保存物理tensor' 即得）✓ |
| 完成保存横幅 | 8/8（grep '======== 完成保存物理tensor'）✓ |
| [KVS] 总行 | **20**（4 启用 + 8 开始 + 8 完成；v2.1 为 532）✓ |
| 开始横幅字段 | PASS 8/8：shape=(128,4,128) bf16、16 层、TERM/seq/req尾8 齐全 ✓ |
| 开始横幅块表/cov vs meta | 8/8 OK ✓ |
| 完成横幅字节 vs manifest | 8/8 OK（12,612,247 / 21,019,411 B）✓ |
| 落盘 flush 耗时 | min 32.1 / max 45.1 / avg 38.6 ms（后台不阻塞）✓ |
| 归档在释放前 | kvc_p.log: 开始横幅 40~48 行 < 释放横幅 55 行 ✓ |
| LATE 兜底 | 0 次 ✓ |

**日志即收据**：`======== 开始保存物理tensor worker=PP?_TP? ... -> 文件名 ========`
声明保存开始（身份/shape/块表齐全）；`======== 完成保存物理tensor ... 落盘 Xms ========`
为完成回执——与 L5 的 `======== 释放 ========` 横幅同一风格，可用同一 grep 习惯定位。

## 5. 六项验证结论（本地离线分析，命令输出存 log/inspect_*.out）

### 5.1 归档结构与 worker 覆盖（log/inspect_list.out）
8/8 结构校验 PASS：P [1,2,3]·324tok·cov=[128,128,68]；R [1,2,4,5,6]·520tok；
{(pp,tp)}={0,1}×{0,1} 全覆盖；同 seq 四 worker 块表/cov/p_tok/w_tok 一致。

### 5.2 前缀缓存命中零篡改（log/inspect_compare.out）
P(s1) vs R(s2) 公共块 [1,2] 在全部 4 worker × 16 层 K/V，前 128 行**逐位相等**（bf16 位模式，
每 worker 2,097,152 元素；本轮 rid 994a2ef5/81604d58）—— 命中即 P 归档时同批物理字节。

### 5.3 重算一致性量化：仅 ULP 级差异（log/inspect_recalc_diff.out）
P.b3[:68] vs R.b4[:68]（同一 token 段、不同 prefill 形状）：
K 2/34816、V 18/34816 元素 1~4 位翻转；Pearson=1.000001；跨轮（v2/v2.1/v2.2 三轮）谱完全一致。

### 5.4 跨 run 对账（log/inspect_stats_crosscheck.out）
L00 按 v1 打印口径重算 vs v1（09-29）记录值：mean/std/min/max **4 位完全相同**
（K: -0.0219/1.394/-10.38/10.81；V: 0.0007659/0.03501/-0.2539/0.3223）。
跨 v1→v2→v2.1→v2.2→v2.3 五轮实验位级确定性互证。

### 5.5 未写槽位全零
P b3 行 68~127 与 R b6 行 8~127 的未写槽位 K/V 全零——块池新建基线为零。

### 5.6 数值分布健康（log/inspect_detail_p/r.out）
16 层有效区无 NaN/Inf；K std 1.398(L00)→2.074(L15)、V std 0.035→0.248（深层展宽）；
R b6 = 8 行纯 decode KV。

## 6. 收尾状态（容器未改动纪律）

- stop.sh：pkill → 0 进程；
- revert_patches.sh：8 文件 [KVC] 归零 + py_compile OK；两仓 `git status` 干净；
- 清除 .orig / bundle tar；容器 kvc/ 与本地终态一致。

## 7. 本轮产物索引

```
log/llama-3-8b.log          服务全量 1185 行（启动 392 + P 62 + R ~730）
log/kvc_startup.log / kvc_p.log / kvc_r.log    [KVC] 三段拆解（172/61/719 行）
log/kvs_archive_lines.log   [KVS] 横幅日志 20 行（4 启用 + 8 开始 + 8 完成）
log/curl_screen.log         curl 命令 + 响应打屏；req/resp json 同存
log/inspect_*.out           离线分析原始输出（列表/深查×2/切片×2/比对/对账/ULP/SAVE 审计）
log/run_all_screen.log / pack_screen.log       容器侧编排/打包留痕
tensors/kv_pp?tp?_s?_*.pt   8 归档（rid 994a2ef5/81604d58）+ manifest.json（md5 8/8 OK）
```
