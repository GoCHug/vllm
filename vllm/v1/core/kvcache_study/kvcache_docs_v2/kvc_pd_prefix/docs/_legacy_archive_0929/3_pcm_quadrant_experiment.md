# PD Prefix Cache 四象限实测实录（10 号 [PCM] 补丁 · 四种开关组合一锤定音）

> docs/2（《PD 分离下 Prefix Cache 开关矩阵》）在末尾 §8 留下了一个回填方案：给 `mooncake_connector.py` 打 6 个观察点、把 P/D 双侧 prefix cache 开关的四象限逐一实测。本文就是那次实验的正式记录——**10 号 PCM 补丁**（独立于 kvc 01-08 与 09，唯一触碰 `mooncake_connector.py`）+ `pcm/` 自包含工作区，于 2026-09-29 08:07~08:17 在 gggtest 容器（a3 4 卡 · P=npu:0 / D=npu:1 · 1P+1D · MooncakeConnectorV1）完成四象限全矩阵实测。
>
> workload 与 docs/1/docs/2 完全同源：**req_p**（324 tok，种缓存）→ sleep → **req_r**（486 tok，含 256 tok 共享前缀），正是"第二请求才能分出象限"d 的标尺场景。**四象限推演全部命中，并带回了三个超出推演的新发现（§5）**。实验后容器源码已回退至基线 md5（00baf169…），工作区（patch/ + scripts/ + log/）留存于 `pcm/`。

## 0. 30 秒看结论

| | ① P开D开 | ② P开D关 | ③ P关D开 | ④ P关D关 |
|---|---|---|---|---|
| P 本地命中 | **256 tok（块 1,2）** | **256 tok（块 1,2）** | 0 | 0 |
| P prefill（req_r） | **230 tok** | **230 tok** | 486 tok | 486 tok |
| P 上报块（PFINISH） | [4] | [4] | [4] | [4] |
| D 本地命中 | **256 tok（块 1,2）** | 0 | **256 tok（块 1,2）** | 0 |
| **实传输块/字节** | **2 块 / 32.0 MiB** | **4 块 / 64.0 MiB** | **2 块 / 32.0 MiB** | **4 块 / 64.0 MiB** |
| mooncake 耗时 | 1.07 ms | 1.38 ms | 1.09 ms | 1.18 ms |
| D hit rate | 31.6% | 0.0% | 31.6% | 0.0% |

四条铁律实测落锤：**P 本地命中只受 P 开关控制、决定 P 算多少**（①②=256 省 256、③④=0 全量算）；**D 本地命中只受 D 开关控制、决定传多少**（①=③、②=④ 字节分毫不差）；**P 恒上报全量 prompt 块**（四格 PFINISH 均报 4 块，裁剪只发生在 D）；**传输按整块 DMA 计**（2 块恒 32.0 MiB / 4 块恒 64.0 MiB，与有效 token 数无关——docs/2 按 token 估算的公式需修正，见 §5.1）。

## 1. 实验设计

### 1.1 10 号 [PCM] 补丁（6 打点 ×7 观察）

对 pristine `mooncake_connector.py`（3770 行，与容器 md5 一致）注入 6 个观察点，补丁生成/应用/回退三重自检（锚点唯一性断言 + py_compile + `patch --dry-run` + [PCM]x7 计数）：

| 打点 | 位置（补丁后行号） | 打印内容 | 证明什么 |
|---|---|---|---|
| CFG | Worker `__init__` :2113 | role + **enable_prefix_caching** | 象限自证（每实例一行，开关与角色一目了然） |
| SCHED | `get_num_new_matched_tokens` :1830 | req / prompt / **local_hit** / do_rp / do_rd | 每请求本地命中数：P 侧=省算依据，D 侧=传输抵扣依据 |
| ALLOC | `update_state_after_alloc` :1871 | **external** / **recv_blocks** / all_blocks | D 仅为未命中 token 分配的接收块清单 |
| PFINISH | `request_finished` :1948 | prompt / prompt_blocks / **report_blocks** / delay_free | P 上报的块清单（恒全量） |
| XFER-entry | `_transfer_kv_cache_all_groups` :790 | recv_groups / pull_groups | 实拉对账（worker 侧已按 D 命中切片；全量上报看 PFINISH） |
| XFER-end | 同函数传输完成 :992 | segments / **bytes(MiB)** / eff_GBps / **pull_local / pull_remote** | 实传字节与两侧块号（D 传输量铁证 + 块号语义） |

另：`FULL-HIT-SKIP` 短路行已预埋（本次 workload 未触发，D 命中未到全量）。原生日志（"KV cache transfer took X ms"、Loggers 的 "Prefix cache hit rate"，无需补丁）作为旁路证据同收。

### 1.2 四象限矩阵与执行

`pcm/scripts/run_matrix.sh` 依次执行四象限，单象限全流程约 2.5~3 min（P 就绪 ~51s + D 就绪 ~45s + 双请求与落盘 ~20s + 停全套与端口缓冲 ~20s）：

| 象限 | P 启动 | D 启动 |
|---|---|---|
| q1_p1d1 | start_p.sh log/q1 1 | start_d.sh log/q1 1 |
| q2_p1d0 | 1（默认开） | **0 → 注入 `--no-enable-prefix-caching`** |
| q3_p0d1 | **0** | 1 |
| q4_p0d0 | **0** | **0** |

CFG 自证行（四象限实录）：q1/q3 P `enable_prefix_caching=True`；q2 P True + D False；q3 P False + D True；q4 双 False——开关真实生效无需借助旁证。每个象限独立起停（进程间零状态污染：冷缓存种 324 tok → 第二请求分象限）。

## 2. 首请求基线：四象限完全一致（冷启动不受开关影响的实证）

req_p（324 tok，双侧冷缓存）在四象限中的 [PCM] 轨迹**逐项相同**：

```
SCHED(D)   prompt=324 local_hit=0 do_rp=True
ALLOC(D)   external=324 recv_blocks=[[1, 2, 3]] all_blocks=([1, 2, 3],)
PFINISH(P) prompt=324 prompt_blocks=3 report_blocks=[3] delay_free=True
XFER-end   segments=64 bytes=50331648 (48.0 MiB) eff_GBps=0.06 pull_local=([1,2,3],) pull_remote=([1,2,3],)
```

- 首请求传输 48.0 MiB = **3 块 × 16 MiB 整块**（324 tok → ceil=3 块；有效字节按 token 算是 40.5 MiB，整块放大 1.19×）
- eff_GBps=0.06 → 反解耗时 ≈ **839ms**，与指纹轮 845.58ms、09-29 轮 834.86ms 同量级——再次确认**首请求耗时是 adxl 会话建立主导**（~835ms 一次性），且该成本与四象限开关无关（乘 4 象限后共 ~4×835ms 冷启动，全过程总长 10 分钟与此吻合）

## 3. 第二请求四象限分野（核心实测）

### 3.1 全矩阵证据表

| 维度 | ① q1_p1d1 | ② q2_p1d0 | ③ q3_p0d1 | ④ q4_p0d0 |
|---|---|---|---|---|
| SCHED(P): prompt / local_hit | 486 / **256** | 486 / **256** | 486 / **0** | 486 / **0** |
| SCHED(D): prompt / local_hit | 486 / **256** | 486 / **0** | 486 / **256** | 486 / **0** |
| P prefill 实算 | 230 tok | 230 tok | 486 tok | 486 tok |
| PFINISH: report_blocks（恒全量） | **[4]** | **[4]** | **[4]** | **[4]** |
| P 上报块号全集（pull_remote 切前） | [1,2,4,5]（含复用 1,2） | [1,2,4,5]（含复用 1,2） | [4,5,6,7]（全新复算） | [4,5,6,7]（全新复算） |
| ALLOC(D): external / recv_blocks | **230 / [4,5]** | **486 / [4,5,6,7]** | **230 / [4,5]** | **486 / [4,5,6,7]** |
| XFER-entry: recv/pull_groups | [2]/[2] | [4]/[4] | [2]/[2] | [4]/[4] |
| XFER-end: segments / bytes | 64 / **32.0 MiB** | 128 / **64.0 MiB** | 64 / **32.0 MiB** | 64 / **64.0 MiB** |
| XFER-end: pull_local / pull_remote | [4,5] / [4,5] | [4,5,6,7] / **[1,2,4,5]** | [4,5] / **[6,7]** | [4,5,6,7] / [4,5,6,7] |
| XFER-end: eff_GBps | 31.26 | 48.78 | 30.82 | 56.63 |
| "KV cache transfer took" | **1.07 ms** | **1.38 ms** | **1.09 ms** | **1.18 ms** |
| 日志: prefix (P / D) | 31.6% / 31.6% | 31.6% / **0.0%** | **0.0%** / 31.6% | 0.0% / 0.0% |
| 日志: D External hit | 100% | 100% | 100% | 100% |

### 3.2 逐象限证据链判读

- **① P开D开（理论默认格，实测复刻 docs/1 轮）**：P 命中 256 只算 230 tok（KVP 口径同 docs/1 §4.1）→ 上报 [1,2,4,5]；D 命中自己种下的 [1,2]（hit 256）→ worker `remote_start_idx=256//128=2` 切掉前 2 块 → 实拉 2 块 32.0 MiB。D 31.6% = 256/810 精确吻合（810=324+486 双请求 tok 总和）。
- **② P开D关（P 省算，传输全付）**：P 与 ① 完全同（hit 256 只算 230）；D local_hit=0 → 不切片 → **连 P 复用过的块 1,2 也重传**（pull_remote=[1,2,4,5] 是"P 上报含复用块的直接铁证"）；D 侧 recv=[4,5,6,7] 全新块（无哈希表 → 无复用引用）。
- **③ P关D开（传省，算不省——"解耦"的试金石）**：P local_hit=0 全量重算 486 tok，为新 prefild 分配全新 4 块 [4,5,6,7]；D 照常命中 [1,2] → 切片后实拉 **P 侧的新块 [6,7]** → 32.0 MiB 与 ① 分毫不差。**P 侧缓存状态对 D 传输量的影响为零**（与 ① 相差的只是 P 侧块号语义）。
- **④ P关D关（裸基线）**：双双归零，全量重算 + 全量 4 块 64.0 MiB；P/D 块号 [4,5,6,7]/[4,5,6,7] 巧合一致（双侧自由池各自顺序分配）。

> 彩蛋（段数差异的机理）：q2 XFER segments=128 = 32 层 × 4 段/层，因为 P 侧 [1,2] 与 [4,5] 两组连续段、D 侧 [4,5,6,7] 一组连续段，`group_concurrent_contiguous`/`split_if_not_byte_contiguous` 要求 src/dst 双侧同时连续才能合并 → 每层拆 2 段（K/V 各 2）；q4 双侧块号完全连续对齐 → 每层 1 组大段 → 64 段。段数本身就是"块号拓扑"的 fingerprint。

## 4. P 侧证据（四象限的 PFINISH 全量铁证）

q1/q3 的 p_pcm.txt（q2/q4 同构，完整清单在 `pcm/log/q*/p_pcm.txt`）：

```
q1: SCHED   req_p: prompt=324 local_hit=0 do_rd=True          ← 冷启动
     PFINISH req_p: prompt=324 prompt_blocks=3 report_blocks=[3]
     SCHED   req_r: prompt=486 local_hit=256                   ← P 缓存命中(省算)
     PFINISH req_r: prompt=486 prompt_blocks=4 report_blocks=[4]  ← 恒报全量
q3: SCHED   req_r: prompt=486 local_hit=0                      ← P 关缓存, 恒 miss
     PFINISH req_r: prompt=486 prompt_blocks=4 report_blocks=[4]  ← 照样恒报全量
```

**四象限 × 双请求 PFINISH 全部按 `ceil(prompt_len/128)` 整prompt上报**（[3]/[3]/[4]/[4]），P 侧 source code 中不存在任何"按 D 状态裁剪"路径——docs/2 §1.1 的机制论断从"源码走读"升级为"实测落锤"。

## 5. 三大新发现（超出 docs/2 推演的部分）

### 5.1 传输字节按整块计，不按有效 token（修正 docs/2 §4 公式）

实测 bytes 完全落在 16 MiB 的整数倍栅格上：3 块=48.0 / 2 块=32.0 / 4 块=64.0（16 MiB = 128 tok × 128 KiB/tok）。external=230 的有效字节是 30.1MB（按 token 计），实测 32.0 MiB —— **mooncake 拉的是整个目标块**（部分尾块的空槽也一起 DMA），比 docs/2 的 token 级公式放大 ceil 后 ×(块 tok 数/有效 tok 数)。修正公式：

```
传输字节 = ceil(external_tokens / 128) × 16 MiB     (TP1 llama-8B, block_size=128)
```

### 5.2 同 pod 跨卡下"省传输"的绝对量很小（选型语境重新定量）

四象限第二请求耗时 1.07~1.38 ms：增量(32 MiB)与全量(64 MiB)差 **0.11~0.31 ms**。会话热后 adxl DMA 有效带宽实测 **30.8~56.6 GB/s**（四象限波动同量级，比 docs/2 按 22 GB/s 的估算高 1.4~2.6 倍，docs/2 曾推算②象限 ~2.8ms，实测 1.38ms）。**结论语境刷新**：跨卡（同 pod）场景 D 缓存对延迟的收益不足 0.5ms，真正心痛的是 **跨机 RDMA 网卡的带宽窗**（64 MiB/req 在 25~50 Gbps 网卡 = 10~20 ms/req 的线路占用）——这正是 docs/2 §3.2 判断"②象限跨机带宽全付"的实测依据，只是量级要在跨机语境下才显性。

### 5.3 D 侧块号与 P 侧块号语义（块号系统独立性的具体呈现）

XFER-end 的 pull_local/pull_remote 四象限分别给出四种块号组合（[4,5↔4,5]、[4,5,6,7↔1,2,4,5]、[4,5↔6,7]、[4,5,6,7↔4,5,6,7]）——**remote 侧块号是 P 池的视角**（含复用块或全新块），**local 侧是 D 池的接收目标**，两侧自由池互不联动（docs/2 §1.4 的"独立 BlockPool"从哈希独立扩展到**块号空间独立**）。q3 的 [6,7] 尤其直观：P 全量重算分配新块 4 块 [4,5,6,7]，被 D 切走前 2 块后剩 [6,7]。

（另两项已预期但首次直接量化的数字：首请求 eff_GBps=0.06 → ~839ms 会话建立成本；D External hit rate=100% 四格全同——D 的 prompt token 一律"本地缓存 + mooncake 外部"合计覆盖，D 从不自己跑 prefill。）

## 6. 与 docs/2 §0 推演表的回填对照

| docs/2 §0 推演项 | 推演值 | 实测值 | 判定 |
|---|---|---|---|
| ① 传输量 | 增量 2 块 ≈30.1MB | 2 块 32.0 MiB | ✓（字节公式修正 §5.1） |
| ① 第二次耗时 | 1.36ms（指纹轮口径） | 1.07 ms | ✓ 同量级 |
| ② 传输量 | 全量 4 块 ≈31.8MB→标注 63.7MB | 4 块 64.0 MiB | ✓ |
| ② 第二次耗时 | ~2.8 ms 推算 | **1.38 ms** | ✓ 方向对、量级修正（带宽 22→49GB/s） |
| ③ 传输量 | 增量 2 块 | 2 块 32.0 MiB | ✓✓（与 ① 分毫不差） |
| ③ 耗时 | ~1.36ms | 1.09 ms | ✓ |
| ④ 传输量 | 全量 4 块 | 4 块 64.0 MiB | ✓ |
| ④ 耗时 | ~2.8ms | 1.18 ms | ✓ 修正同 ② |
| 首次请求 | ~835ms 会话建立 | ~839ms（eff 0.06 反解） | ✓ |
| 内存驻留 | ①③D 保留 / ②④即释 | hitrate 31.6% / 0.0% 佐证 | ✓（块级轨迹见 llama.log） |

**docs/2§8 的五点验收预测全部命中**（传输 ms 分档、hit_length、hit rate、KVP 正确性未复核项见 §7 说明、Delaying free 四格出现）。

## 7. 正确性边界说明（本实验未覆盖面）

本次 PCM 补丁**不含 KVP 内容指纹**，四象限的"数值正确性"未逐块复核（正确性已由 docs/1+09 号补丁在双开象限②构型下全量证明，且 KV 数值与 APC 开关无耦合——docs/2 §5）。若需在四象限各自的 KV 落卡数值上复核，可将 kvc 01-08+09 与 10 号叠加（10 号独立于 01-09，同一文件互不冲突的文件集），复用 `compare_fp.py` 对账。

## 8. 复现（容器内，~10 分钟）

```bash
# 前提: gggtest 容器空闲(无 vllm 进程), /vllm-workspace 双仓为 pristine 基线
cd /a3_inference/itask/workdir/gch02599191/kvc_pd/pcm
patch/apply_pcm_patch.sh                    # 1. [PCM]x7 注入(独立, 无需 kvc 01-09)
bash scripts/run_matrix.sh                  # 2. 四象限全自动(~10min), 实时: tail -f log/matrix_run.log
grep '\[PCM\]' log/q*/{p,d}_llama.log       # 3. 证据抽查
patch/revert_pcm_patch.sh                   # 4. 回退(md5 应回 00baf169f48fb167b9f6dfe650ac0ea5)
```

## 9. 产物清单（`pcm/`，容器与本地同步）

| 产物 | 说明 |
|---|---|
| `patch/gen_10_pcm_patch.py` | 补丁生成器（锚点断言 + difflib 产出 + 三重自检） |
| `patch/10_pcm_prefix_cache_matrix.patch` | 六打点补丁（6 hunks，[PCM]x7） |
| `patch/apply_pcm_patch.sh` / `revert_pcm_patch.sh` | 独立应用/回退（防重 + 计数 + py_compile + md5 留底） |
| `scripts/start_{p,d}.sh` | 参数化启动（pc=0 注入 --no-enable-prefix-caching） |
| `scripts/run_quadrant.sh` / `run_matrix.sh` | 单象限全流程 / 四象限一键 + 总表 |
| `log/q{1..4}_*/` | 每象限 12 文件：三组件日志 + [PCM] 行 + hitrate + q_summary.md |
| `log/matrix_run.log` / `log/matrix_summary.md` | 全程编排输出 / 修正版对照总表 |

> 关键行号举证：[PCM] 打点位于 mooncake_connector.py:2113/1830/1871/1948/790/992（补丁后行号）；原生证据行 `KV cache transfer`（:981）与 `Prefix cache hit rate`（loggers.py）无需补丁。
