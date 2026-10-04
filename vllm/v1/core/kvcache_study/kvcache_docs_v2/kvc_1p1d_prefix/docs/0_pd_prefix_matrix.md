# kvc_1p1d —— PD Prefix Cache 开关矩阵：机制、四场景实测与场景卡（1P1D）

> **工作区定位**：1P1D（1P prefill producer + 1D decode consumer + mooncake + load_balance proxy）形态下的 **P/D prefix cache 开关四象限实验**——用 01 号 [PCM] 六打点补丁观察四个开关组合（`--enable-prefix-caching` 默认开 / `--no-enable-prefix-caching` 注入）在"算多少 / 传多少 / 驻留多少"上的行为差异。本目录**独立自持**（补丁/脚本/请求体/文档全套自带，零外部依赖）。
>
> **证据体系**：实测产物在 `logs/q{1..4}/`（run_all 五阶段产出，一象限一子目录），汇总判读在 `logs/analysis/matrix_report.out`（铁律核验 PASS/FAIL）——本文 §0/§3/§4 数值与该报告一一对应。
>
> 统一口径：req_p 324 tok（种缓存）→ req_r 486 tok = 4 块（块 1,2 = 256 tok 共享前缀 + 块 4,5 = 230 tok 新增）；块 128 tok；TP1 bf16。

## 0. 30 秒结论：四象限总表（本轮 gggtest 容器实测；数值与 matrix_report.out 一致）

| 象限 | P 本地命中 | P prefill 实算 | P 上报块(恒全量) | D 本地命中 | mooncake 实际传输 | 第二请求耗时* | 跨请求显存驻留 |
|---|---|---|---|---|---|---|---|
| ① q1_p1d1 **(默认)** | **256 tok(块 1,2)** | 230 tok | 4 块 [1,2,4,5] | **256 tok(块 1,2)** | 增量 2 块 = **32.0 MiB** | **1.17 ms** | D 池保留命中块(LRU) |
| ② q2_p1d0 | **256 tok(块 1,2)** | 230 tok | 4 块 [1,2,4,5] | 0(无哈希表) | 全量 4 块 = **64.0 MiB** | **1.46 ms** | 无(D 请求完即释放) |
| ③ q3_p0d1 | 0 | **486 tok(全量重算)** | 4 块 [4,5,6,7]† | **256 tok(块 1,2)** | 增量 2 块 = **32.0 MiB** | **1.16 ms** | D 池保留命中块 |
| ④ q4_p0d0 | 0 | **486 tok(全量重算)** | 4 块 | 0 | 全量 4 块 = **64.0 MiB** | **1.18 ms** | 无 |

\* 耗时指 mooncake `KV cache transfer` 行，会话热后纯 DMA；首请求 260.75~309.37 ms（adxl 会话建立一次性成本，四象限一致）。† P✗ 时 P 复用断链、全新复算的块号与 P✓ 时不同（实测 pull_remote=[6,7]，详见 §3 卡③）。

**四条铁律**（本轮 matrix_report.py 自动核验的判据）：
1. **P 的开关只影响"算多少"，D 的开关只影响"传多少"**——二者完全解耦（§1）；
2. vLLM v1 **默认双侧都开**（`enable_prefix_caching: bool = True`，vllm/vllm/config/cache.py:92）——线上与象限①同态；
3. **P 上报块恒为全量 prompt 块**——P 不感知 D 的缓存（§1.2）；
4. 四格**正确性全等**（§6）：APC 改变的只是"KV 从哪来"。

## 1. 机制底座：prefix cache 在一条 PD 请求里的三个出场位置

### 1.1 P 侧：省 prefill 计算（与单机 APC 语义相同）

P 调度器对每个新请求先做本地前缀查找（`KVCacheManager.get_computed_blocks`）——命中即跳过这部分 token 的 forward。双请求后 P 侧 hit rate = 256/810 = 31.6%（810 = 324+486）。P 的 `request_finished` 上报给 D 的块清单与 P 是否命中无关——恒为"全部 prompt 块"。

### 1.2 D 侧：省传输量（增量裁剪链，唯一的传输裁剪变量）

1. **调度器本地查找** → `num_computed_tokens`（D✓ 时 req_r 命中 256；D✗ 恒 0）
2. **connector 报增量**：`get_num_new_matched_tokens` (mc:1786-1822) `count = max(actual - num_computed_tokens, 0)` → external = 486 − D命中
3. **worker 裁剪远端块清单**：`_get_kernel_block_ids` (mc:2587-2618) 按输出 `remote_start_idx = num_computed_tokens // remote_kernel_token_size` 切头——注释原话 "Skip prefix-cached remote kernels (D-side already holds them)"
4. **极端：全命中零传输**（mc:786-790 `num_local_blocks == 0` 直接 return， PCM 补丁另有 FULL-HIT-SKIP 行）

D 落卡接收目标是 `update_state_after_alloc` 里的**未哈希新块**（`get_unhashed_block_ids_all_groups`, mc:1837）。

### 1.3 协议中枢：kv_transfer_params 经 proxy 从 P 响应流入 D 请求

proxy 双发次序：改写副本（`build_prefill_request` proxy:790-806，`max_tokens=1, stream=False, do_remote_decode=True`）→ P prefill 收官（LENGTH_CAPPED）→ `request_finished` 三连门槛（mc:1897-1902）返回 kv_transfer_params → proxy 提取（proxy:925）→ 注入原始请求发 D。详见同族 `../kvc_1p1d_prefix/docs/0_pd_request_lifecycle.md`（如需生命周期细节见 `../kvc_1p1d/docs/`，本区不依赖）。

### 1.4 双侧独立 BlockPool、独立哈希链——两个开关解耦的结构性前提

同内容 token 的块哈希**构造链头 NONE_HASH 是进程级随机盐**（`os.urandom(32)`）——P/D 两进程各摸一个，同 token 序列双侧哈希必然不同。**D 的命中只能来自 D 自己接收过并落卡的块**，P 种下的哈希对 D 不可见；传输搬运的是 KV 数据字节，不带任何哈希、块号。

## 2. 实验方法：01 号 PCM 补丁与四象限编排

### 2.1 六打点（[PCM] × 7 观察位，唯一触碰 mooncake_connector.py）

| 打点 | 位置 | 打印内容 | 证明什么 |
|---|---|---|---|
| CFG | Worker `__init__` | role + **enable_prefix_caching** | 象限自证：开关真实生效的每实例一行 |
| SCHED | `get_num_new_matched_tokens` | req / prompt / **local_hit** / do_rp / do_rd | 每请求本地命中数：P=省算依据，D=传输抵扣依据 |
| ALLOC | `update_state_after_alloc` | **external** / **recv_blocks** / all_blocks | D 仅为未命中 token 分配的接收块清单 |
| PFINISH | `request_finished` | prompt / **report_blocks** / delay_free | P 上报的块清单（**恒全量**铁证） |
| XFER-entry | `_transfer_kv_cache_all_groups` 入口 | recv_groups / pull_groups | 实拉对账（post-slice；全命中另有 FULL-HIT-SKIP） |
| XFER-end | 同函数传输完成 | segments / **bytes(MiB)** / eff_GBps / **pull 块** | 实传字节与两侧块号（D 传输量铁证） |

补丁独立（mooncake_connector.py 不与 kvc/kvc_1p1d 的 01~08 任何目标重叠——可叠加互不冲突）；`gen_01_pcm_patch.py` 锚点断言 + py_compile + dry-run + [PCM]x7 计数四重自检。

### 2.2 四象限编排（run_matrix.sh，单象限 ~2.5-3 min）

| 象限 | P 启动 | D 启动 | 开关注入 |
|---|---|---|---|
| q1_p1d1 | `scripts/server/start_p.sh logs/q1 1` | `start_d.sh logs/q1 1` | 无（默认双开） |
| q2_p1d0 | 1 | **0** | D 侧 `--no-enable-prefix-caching` |
| q3_p0d1 | **0** | 1 | P 侧 `--no-enable-prefix-caching` |
| q4_p0d0 | **0** | **0** | 双侧注入 |

每象限**独立起停**（进程间零状态污染：冷缓存种 324 tok → 第二请求分象限观查）；`run_quadrant.sh` 自带 HBM 防抢占等待 + 崩溃早退 + EXIT trap 兜底清理 + 重试×2。

## 3. 四张场景卡：配置 × [PCM] 打印全集 × 判读

> 引文为 [PCM] 打印的原样形态（取自 `logs/q*/`，req 为缩略示意）；grep 直达各象限同位文件。

### 场景卡①：q1_p1d1（P✓ D✓，默认基线）

**配置**：双侧不传任何开关（v1 默认）。

```
P 侧(p_pcm.txt):
[PCM] CFG role=kv_producer enable_prefix_caching=True ...
[PCM] SCHED req=<req_r> prompt=486 local_hit=256 do_rp=False do_rd=True
[PCM] PFINISH req=<req_r> prompt=486 prompt_blocks=4 report_blocks=[4] delay_free=True
D 侧(d_pcm.txt):
[PCM] CFG role=kv_consumer enable_prefix_caching=True ...
[PCM] SCHED req=<req_r> prompt=486 local_hit=256 do_rp=True do_rd=False
[PCM] ALLOC req=<req_r> external=230 recv_blocks=[[4, 5]] all_blocks=([1, 2, 4, 5],)
[PCM] XFER-entry req=<req_r> recv_groups=[2] pull_groups=[2]
[PCM] XFER-end req=<req_r> segments=64 bytes=33554432 (32.0 MiB) eff_GBps=28.79 pull_local=([4, 5],) pull_remote=([4, 5],)
KV cache transfer for request <req_r> took 1.17 ms.(原生行)
```

**判读**：P 命中 256（只算 230）→ 上报全 4 块；D 命中 256 → 切掉前 2 块只拉 [4,5] 32 MiB。**P 命中 + D 命中 = 算力带宽双省**。

### 场景卡②：q2_p1d0（P✓ 省算、传输全付）

**配置**：D 侧 `--no-enable-prefix-caching`（P 默认）。CFG 行自证：`D: enable_prefix_caching=False`。

```
P 侧:与①完全相同(local_hit=256 / 只算 230 / PFINISH report_blocks=[4])
D 侧:
[PCM] SCHED req=<req_r> prompt=486 local_hit=0 do_rp=True do_rd=False      ← 无哈希表,恒 miss
[PCM] ALLOC req=<req_r> external=486 recv_blocks=[[4, 5, 6, 7]] all_blocks=([4, 5, 6, 7],)
[PCM] XFER-end req=<req_r> segments=128 bytes=67108864 (64.0 MiB) eff_GBps=45.81 pull_local=([4, 5, 6, 7],) pull_remote=([1, 2, 4, 5],)
KV cache transfer ... took 1.46 ms.
```

**判读**：P 上报含**复用块 [1,2]**（pull_remote 铁证"恒全量"）；D 无表不裁剪 → 连 P 复用过的块也重传。segments=128（≠④ 64）：P 侧两组连续块与 D 侧一组对不齐、src/dst 段无法合并——**段数是块号拓扑的指纹**。

### 场景卡③：q3_p0d1（传省、算不省——解耦性试金石）

**配置**：P 侧 `--no-enable-prefix-caching`（D 默认）。

```
P 侧:
[PCM] SCHED req=<req_r> prompt=486 local_hit=0 do_rd=True                  ← P 恒 miss,全量重算
[PCM] PFINISH req=<req_r> prompt=486 prompt_blocks=4 report_blocks=[4]
D 侧:与①完全相同(ALLOC external=230 recv=[4,5];XFER-end 32.0 MiB)
[PCM] XFER-end ... bytes=33554432 (32.0 MiB) eff_GBps=28.84 pull_local=([4, 5],) pull_remote=([6, 7],)   ← P 全新复算块号
KV cache transfer ... took 1.16 ms.
```

**判读**：**传输量与①分毫不差（32.0 MiB/2 块）——P 侧缓存状态对 D 传输量的影响为零**（差异只在 pull_remote 块号：P 全量复算时自分配 [4,5,6,7]，被 D 切前 2 后剩 [6,7]）。**"计算量只看 P 开关、传输量只看 D 开关"的最直接实证**。

### 场景卡④：q4_p0d0（双关基线）

**配置**：双侧 `--no-enable-prefix-caching`（CFG 行双 False）。

```
P 侧:同③(local_hit=0 / 全量算 / report_blocks=[4])
D 侧:同②(local_hit=0 / external=486 / recv 4 块)
[PCM] XFER-end req=<req_r> segments=64 bytes=67108864 (64.0 MiB) eff_GBps=56.93 pull_local=([4, 5, 6, 7],) pull_remote=([4, 5, 6, 7],)
KV cache transfer ... took 1.18 ms.
```

**判读**：全量重算 + 全量重传 + 双侧不驻留——行为最可预测的**排障基线**（消除一切缓存路径变量）。即便双关 **Delaying free 仍出现**（传输协议需要，非缓存特性）。

## 4. 实测结果分析（本轮 gggtest 容器实测：2026-10-04 09:44:52 → 09:53:37；判读与 matrix_report.out 一致）

### 4.1 首请求基线：四象限完全一致（冷启动不受开关影响）

req_p（324 tok 双侧冷缓存）四格 [PCM] 轨迹逐项相同：`local_hit=0 → external=324 → recv=[1,2,3] → XFER-entry recv/pull=3 组 → 3 块 48.0 MiB`（首请求 took 260.75~309.37 ms：adxl 会话建立+首建成本，一次性，四象限一致——q1 309.37 / q2 260.75 / q3 272.73 / q4 270.61）。

### 4.2 第二请求全矩阵证据表

| 维度 | ① q1 | ② q2 | ③ q3 | ④ q4 |
|---|---|---|---|---|
| SCHED(P) local_hit | **256** | **256** | 0 | 0 |
| SCHED(D) local_hit | **256** | 0 | **256** | 0 |
| P prefill 实算 | 230 tok | 230 tok | 486 tok | 486 tok |
| PFINISH report_blocks(**恒全量**) | **[4]** | **[4]** | **[4]** | **[4]** |
| P 上报块号全集 | [1,2,4,5](含复用) | [1,2,4,5](含复用) | [4,5,6,7](全新) | [4,5,6,7](全新) |
| ALLOC(D) external / recv | **230 / [4,5]** | **486 / [4,5,6,7]** | **230 / [4,5]** | **486 / [4,5,6,7]** |
| XFER-end segments / bytes | 64 / **32.0 MiB** | 128 / **64.0 MiB** | 64 / **32.0 MiB** | 64 / **64.0 MiB** |
| pull_local / pull_remote | [4,5] / [4,5] | [4,5,6,7] / **[1,2,4,5]** | [4,5] / **[6,7]** | [4,5,6,7] / [4,5,6,7] |
| eff_GBps | 28.79 | 45.81 | 28.84 | 56.93 |
| took(第二请求) | **1.17 ms** | **1.46 ms** | **1.16 ms** | **1.18 ms** |
| 日志 prefix hit(P/D) | 31.6% / 31.6% | 31.6% / **0.0%** | **0.0%** / 31.6% | 0.0% / 0.0% |
| D External hit | 100% | 100% | 100% | 100% |

### 4.3 三大发现（超出理论推演部分）

1. **传输字节按整块计**：`bytes = ceil(external/128) × 16 MiB` 栅格（230 tok → 32.0 非 30.1）——mooncake 拉整块，部分尾块空槽一起 DMA。
2. **同 pod 跨卡下"省传"绝对量小**：增量(33.9MB) vs 全量(67.1MB) 差 0.02~0.29ms；有效带宽 29~57 GB/s。真正的痛在跨机 RDMA 带宽窗（25-50Gbps 网卡上 64 MiB/req = 10-20ms 线路占用）。
3. **块号系统独立**：pull_local/pull_remote 四种组合——remote 是 P 池视角、local 是 D 池接收目标，两侧自由池互不联动。

## 5. 成本模型速查（TP1 bf16 · 块 128 tok；实测修正版）

| 量 | 公式 | req_r 例值 |
|---|---|---|
| 每 token KV 字节 | 2(KV) × 32层 × 8头 × 128维 × 2B = **128 KiB** | — |
| 传输字节 | **ceil(external_tokens/128) × 16 MiB**（整块栅格） | ①③ **32.0 MiB**；②④ **64.0 MiB** |
| P prefill tokens | prompt_len − P_hit（P✓ 时） | ①② 230；③④ 486 |
| 首次请求耗时 | adxl 会话建立（一次性）+ DMA | **260.75~309.37 ms**，四象限一致 |
| 后续请求耗时 | ≈ 传输字节 / (29~57 GB/s 实测有效) | 32MiB→1.16/1.17ms、64MiB→1.18/1.46ms |
| 命中率监控 | `Prefix cache hit rate`(P/D 各打各的) + D 侧 `External prefix cache hit rate` | P/D 31.6%；D External=100%（D prompt 全由"本地缓存+mooncake"覆盖，从不自算 prefill） |

## 6. 正确性与边界

- KV 比特级与"从哪来"无关：跨卡 DMA 路径与 D 本地复用路径的逐位对账在 `../kvc_1p1d/`（Tx 区 torch.equal 全 PASS）已覆盖 → 四象限组合路径都在已验证空间内。
- D✗ 只是"接收块不入哈希表"（字节不变）；P✗ 只是"每轮重 prefill"（确定性计算、同 KV）。
- 尾 token 重算与象限无关：D 首步恒重算最后 1 个 prompt token（bootstrap 语义）。
- **边界与坑**：①D 池 LRU 驱逐回退（缓存块被逐→退化②行为，正确性无损）；②满块才入哈希表（复用粒度 128 tok 整块）；③P 延迟释放兜底 480s（`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`，D 不来拉时强制 free）；④adxl 会话/元数据一次性；⑤proxy 轮询稀释命中（assign_instances 每请求选实例），D 抢占走"recomputed"重试。

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 常规负载(对话/agent 共享 system+few-shot) | ①(默认双开) | 算力、带宽双省 |
| D 内存极紧/无共享前缀 | ② | D 零驻留；代价为同 pod 0.3ms/跨机 64MiB 每请求 |
| P→D 带宽硬瓶颈、P 算力富余 | ③ | 传输仍增量；TTFT 恒全量 |
| 排障/纯独立请求 | ④ | 消除全部缓存路径变量 |

一句话：**默认①别动**；要关就明确省什么（②省 D 内存、③省 P 内存）放弃什么（②放弃跨请求带宽复用、③放弃 prefill 省算）。

## 8. 本轮复现（gggtest 容器，2026-10-04 重构后形态）

```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc_1p1d_prefix
setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
# [1/5] 01号 PCM 补丁 -> [2/5] 四象限矩阵(~12min) -> [3/5] matrix_report 汇总 -> [4/5] 打包 -> [5/5] 收尾提示
tail -f logs/server/run_all_screen.log
# 铁律核验(本地拉回后同样可跑): python3 scripts/analysis/matrix_report.py --dir logs
```

> 编排内置防护：HBM 防抢占等待 + 动态选卡 + 失败重试×2 + EXIT trap 兜底清理（任一象限失败不污染其余象限）。

## 9. 产物索引（kvc_1p1d_prefix/）

| 产物 | 说明 |
|---|---|
| `scripts/patchs/{01_pcm_prefix_cache_matrix.patch, apply, revert, gen_01_pcm_patch.py}` | 六打点补丁 + 生成器 + 应用/回退（防重/dry-run/[PCM]x7 计数/py_compile/md5） |
| `scripts/server/{start_p,start_d,start_proxy,stop}.sh` | 生命周期(参数 = 象限日志子目录 + pc:1\|0) |
| `scripts/server/{run_quadrant,run_matrix,run_all}.sh` | 单象限全流程 / 四象限一键 / 五阶段总编排 |
| `scripts/curl/req_{p,r}.json` + `gen_pd_requests.py` | 请求体自持(324/486 tok, 同族 workload) |
| `scripts/analysis/matrix_report.py` | 四象限对照总表 + 铁律核验 → logs/analysis/matrix_report.out |
| `scripts/recover/pull_artifacts.sh` | 主机侧 fetch 单命令(经 5557 隧道) |
| `logs/q{1..4}_{p0|p1}{d0|d1}/` | **本轮**四象限产物(一象限一子目录, 12 文件) |
| `logs/analysis/matrix_report.out` | 本轮汇总判读(总表 + 铁律核验 PASS/FAIL) |

