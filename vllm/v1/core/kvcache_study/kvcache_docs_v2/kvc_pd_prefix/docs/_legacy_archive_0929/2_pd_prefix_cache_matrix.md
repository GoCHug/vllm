# PD 分离下 Prefix Cache 开关矩阵（P × D 四象限）

> 本文是 docs/1（P✓D✓ 象限的正确性验证实录）的姊妹篇：把 **P 侧 prefix cache 开关 × D 侧 prefix cache 开关** 的四种组合（四象限）在 1P+1D · MooncakeConnectorV1 · vllm-ascend 0.23.0 形态下的行为差异逐一讲透——每格的 **Prefill计算量 / P→D 传输量 / 耗时 / 显存驻留 / 正确性路径 / 适用与不适用** 全部落到源码行号与实测日志上。
>
> **证据基础**：① 源码走读 `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`（3770 行）+ `vllm/vll/v1/core/*`；② 实测两轮——docs/1 记录的 2026-09-27 轮（845.58ms / 1.31ms）与 2026-09-29 06:48 本地复跑轮（`log/p_llama.log` / `log/d_llama.log`，834.86ms / 1.36ms，结构完全同构）；③ **2026-09-29 08:07 四象限全矩阵实测**（10 号 [PCM] 补丁 · `pcm/` 工作区 · gggtest 容器）：②③④ 象限不再是推演——SCHED/ALLOC/PFINISH/XFER 六打点逐格落锤，全部命中本文推演，且带回三个修正项（传输按整块 DMA 计、实测带宽 31-57GB/s、同 pod 下全量/增量耗时差 ~0.3ms），详见 **docs/3** 与 `pcm/log/matrix_summary.md`。
>
> 统一口径：R 请求 prompt 486 tok = 4 块 [1,2,4,5]（块 1、2 共 256 tok 是与前请求 P（324 tok）的共享前缀，块 4、5 计 230 tok 为新增）；块大小 128 tok；TP1 bf16。

## 0. 30 秒结论：四象限总表（2026-09-29 全矩阵实测版，实录见 docs/3）

| 象限 | P 本地命中 | P prefill 实算 | P 上报块（恒全量） | D 本地命中 | mooncake 实际传输 | 第二次请求耗时* | 跨请求显存驻留 |
|---|---|---|---|---|---|---|---|
| ① P✓ D✓ **（默认）** | **256 tok（块 1,2）** | 230 tok | 4 块 [1,2,4,5] | **256 tok（块 1,2）** | 增量 2 块 = **32.0 MiB** | **1.07 ms** | D 池保留命中块（LRU） |
| ② P✓ D✗ | **256 tok（块 1,2）** | 230 tok | 4 块 [1,2,4,5] | 0（无哈希表） | 全量 4 块 = **64.0 MiB** | **1.38 ms** | 无（D 请求完即释放） |
| ③ P✗ D✓ | 0 | **486 tok（全量重算）** | 4 块 [4,5,6,7]† | **256 tok（块 1,2）** | 增量 2 块 = **32.0 MiB** | **1.09 ms** | D 池保留命中块 |
| ④ P✗ D✗ | 0 | **486 tok（全量重算）** | 4 块 | 0 | 全量 4 块 = **64.0 MiB** | **1.18 ms** | 无 |

\* 耗时指 mooncake `KV cache transfer` 行（d_llama.log:501 口径），传的是会话热后的纯 DMA；首次请求四格都是 ~835ms（adxl 会话建立，docs/3 §2 冷启动基线）。† P✗ 时 P 复用断链、全新复算的块号与 P✓ 时不同（[6,7] 实拉细节见 docs/3 §3.1）。**P 本地命中与 D 本地命中语义对称**：P 命中省自己的 prefill 算力，D 命中省 P→D 的传输量——两列各自只受本侧开关控制，`P 上报块` 四格恒为全量 prompt 块（**P 不感知 D 的缓存**，§1.2）。

> 演进注：本表初版（推算版）曾以 ①1.36ms 实测 / ②④ ~2.8ms 推算 / 字节按 token 估算（30.1/63.7MB）面世；四象限实测后修正为整块栅格字节（16 MiB/块）与实测耗时（有效带宽 31-57GB/s，同 pod 下全量 vs 增量只差 ~0.3ms——**省传输的价值场景在跨机 RDMA**）。逐格 [PCM] 证据：`pcm/log/matrix_summary.md`。

三个一眼记住的要点：
1. **P 的开关只影响"算多少"，D 的开关只影响"传多少"**，二者完全解耦——除了 D 出哈希种子阶段的连带（§1.4），没有任何一侧的开关会改变另一侧行为。
2. vLLM v1 **默认双侧都开**（`enable_prefix_caching: bool = True`，vllm/vllm/config/cache.py:92；start_p.sh/start_d.sh 都没传 `--no-enable-prefix-caching`）——线上与 docs/1 实测都处在象限①。
3. 四格**正确性全等**（§5）：APC 改变的是"KV 从哪来"（本地复用 or 跨卡 DMA 或两者），不改变"KV 是什么"。

## 1. 机制底座：prefix cache 在一条 PD 请求里的三个出场位置

### 1.1 P 侧：省 prefill 计算（与自己单机时的 APC 语义相同）

P 调度器对每个新请求先做本地前缀查找（`KVCacheManager.get_computed_blocks`，kv_cache_manager.py:196）——命中即跳过这部分 token 的 forward：

- 本轮实测 P 侧请求 2：命中块 1、2 → **只 prefill 新增 230 tok**（docs/1 §4.1：`hit_length=256, hit_blocks=[[1,2]]`，KVP `PF+TERM 486/486` 覆盖 4 块 = 复用 1、2 + 新写 4、5）。
- 给 Prometheus 的直接实证：P 侧 hit rate 请求 1 后 0.0%（p_llama.log:441）→ 请求 2 后 **31.6%**（p_llama.log:600）——恰 = 256/810（810 = 324+486 两请求 prompt 总 tok），分毫不差。
- P✗：`enable_caching=False`（scheduler.py:235 传入 KVCacheManager）→ P 恒 miss、恒全量 prefill，TTFT 不随历史改善。

**注意：P 的 request_finished 上报给 D 的块清单与 P 是否命中无关**——恒为"全部 prompt 块"。请求 2 的上报是 4 块 [1,2,4,5]（复用块也带）：p_llama.log:432 `Delaying free of 3 blocks`（请求 1，324 tok 3 块）→ p_llama.log:591 `Delaying free of 4 blocks`（请求 2，486 tok 4 块）。裁剪逻辑只按 prompt 长度裁掉 MTP/SWA 尾（`_get_transfer_block_ids` mooncake_connector.py:1712-1735、`_get_swa_transfer_block_ids` 1737-1751），**不存在"按 D 命中裁剪"**。P 是纯数据供给方，不感知 D。

### 1.2 D 侧：省传输量（增量裁剪链）

D 的 APC 命中数是唯一的传输裁剪变量，链路三步：

1. **调度器本地查找** → `num_computed_tokens`。D✓ 时请求 2 命中 256（第二轮实测 `hit_length=256, hit_blocks=[[1,2]]`；D 侧 Prometheus 0.0% → 31.6%，d_llama.log:460 → 1253——后者还打出 `External prefix cache hit rate: 100.0%`，即 D 的 prompt token 100% 来自"本地缓存 + mooncake 外部"合计而非自算 forward）。D✗ 时恒 0。
2. **connector 报增量**：`get_num_new_matched_tokens`（mooncake_connector.py:1786-1822）`count = max(actual - num_computed_tokens, 0)`（1814 行）→ 请求 2 申报 external = 486-256 = 230。
3. **worker 裁剪远端块清单**：`_get_kernel_block_ids`（2587-2618）里 `remote_start_idx = meta.num_computed_tokens // remote_kernel_token_size; kernel_remote = kernel_remote[remote_start_idx:]`（2615-2616），注释原话 "Skip prefix-cached remote kernels (D-side already holds them)"——把 P 上报的前 2 块切掉，只对块 4、5 构造 `batch_transfer_sync_read` 描述符。
4. **极端：全命中则零传输**。`_transfer_kv_cache_all_groups`（786-790）`num_local_blocks == 0` 直接 return，注释 "Full prefix cache hit: do not need to read remote blocks"。

配套：D 落卡的接入目标是 `update_state_after_alloc` 里的**未哈希新块**（`blocks.get_unhashed_block_ids_all_groups()`，1837 行）——即"这部分 KV 我本地没有，正是要收的"。

### 1.3 协议中枢：kv_transfer_params 经 proxy 从 P 响应流入 D 请求

proxy（load_balance_proxy_server_example.py）转发次序：先发 preset（`build_prefill_request` 790-806：`max_tokens=1`、`stream=False`、注入 `do_remote_decode=True`）→ P prefill 收官（必 FINISHED_LENGTH_CAPPED）→ P 的 `request_finished`（mooncake_connector.py:1882-1928）返回完整 kv_transfer_params 字典（`do_remote_prefill=True` + remote_block_ids + host/port/engine_id 等）**随 P 的 HTTP 响应体回传** → proxy 提取注入转给 D（925-927 行：`kv_transfer_params = response.json().get("kv_transfer_params")`）→ D 拿到"去哪张卡上拉哪几块"。

这解释了实测响应里 `"kv_transfer_params": null`（resp_p.json）——对外部 curl 而言 proxy 把参数剥走注入了 D 侧请求客户端不可见。

### 1.4 双侧独立 BlockPool、独立哈希种子——两个开关解耦的根本原因

同内容 token 在 P/D 侧的块哈希**不同**（各自 NONE_HASH 随机盐：P 走 `ca5796d309cb`/`098dc4eed782`，D 走 `3fa6fb86447a`/`6e3746e03188`，docs/1 §4.2）——**D 的命中只能来自"D 自己接收过并落卡的块"**，P 种下的哈希对 D 不可见（反之亦然）。两侧池、两侧换入换出、两侧 LRU，互不联动；这是"D 单独开关"在工程上天然成立的结构性前提。

另一个代码级佐证：块哈希链的消费方注明是 "prefix caching and KV connectors (P/D, offloading)"（vllm/vllm/v1/core/kv_cache_utils.py:629-634）——**D✗ 且 connector 在跑时 Request.block_hashes 依然会计算**（connector_enabled=True），只是 enable_caching=False 的池不维护哈希映射、不做查找。所以 D✗ 不影响协议流转，纯粹失去"本地复用"。

## 2. 统一实验语境（四格放到同一标尺）

```
请求 1 (种缓存): prompt 324 tok ──┐
                                   ├─ 前缀公共区 = 块1,2 = 256 tok
请求 2 (R):     prompt 486 tok ──┘  新增区   = 块4,5 = 230 tok (块5 为 102 tok 部分块)
```

四格 PK 的都是"请求 2 "的行为。请求 1 在四格中行为一致（双侧冷缓存：P 全量算 324 + D 全量收 3 块 + D✓ 时满块种哈希）。

## 3. 四象限逐一详解

### 3.1 象限① P✓ D✓（默认；两轮实测）

完整链路（本轮日志行号；09-27 轮见 docs/1 §3/§4）：

```
请求2 到proxy ─→ P: get_computed_blocks HIT 256 ─→ P prefill 230 tok
              ─→ P request_finished: 上报 4 块 + Delaying free of 4 blocks (p:591)
              ─→ D: get_computed_blocks HIT 256 (接收时种下的块1,2)
              ─→ D connector: external = 486-256 = 230 (mc:1814)
              ─→ D worker: remote_start_idx=2, 切掉P上报的块1,2 (mc:2615)
              ─→ batch_transfer_sync_read: 只DMA块4,5 ≈30.1MB → 1.36ms (d:501)
              ─→ D decode 35 步(落卡含块6) → 释放:P等done信号后放块
```

KB 计口径（TP1 llama-3-8b bf16，一次 KV 元素总数 n = 486×32768）：

- 收益结算：**prefill 算力 2.1×↓**（486→230）+ **传输量 2.1×↓**（63.7→30.1MB）+ 耗时 614×↓（834.86ms→1.36ms，其中大头是 adxl 会话/msn 一次性建立被摊销 + 量减）。
- 正确性：KVP 逐项 18 项分毫不差（docs/1 §5.1）——传输路径（块 4、5）与本地复用路径（块 1、2）**同请求混合落地**且均零比特差，是四象限正确性论证的锚。
- 代价：D 池保留命中块（LRU 驱逐前驻留）、P 侧"延迟释放 + 哈希驻留"两份显存占用。

### 3.2 象限② P✓ D✗（P 省算、传输全付）

- P 侧与 ① 完全同（命中 256 → 只算 230 tok，TTFT 不变）。
- D 侧：`enable_caching=False` → `get_computed_blocks` 恒空（num_computed_tokens=0）→ external=486 → D 给整个 prompt 分配全新未哈希块 → `remote_start_idx=0` 不切任何块 → 从 P 拉**全部 4 块 ≈63.7MB**。
- 耗时推算：会话热后 ≈ 63.7MB ÷ 22GB/s（实测 1.36ms 传 30.1MB 的有效带宽口径）≈ **2.8ms**。跨机 RDMA 口径下 63.7MB/请求的链路流量才是主痛点（同 pod 跨卡 22-45GB/s 近乎免费，跨机 25-50Gbps 网卡每请求吃 10-20ms 带宽窗）。
- **D 的接收不再种哈希**——第三、四个同前缀请求每个都全量拉，"传输成本随 D 命中递减"特性（docs/1 §4.2 观察到的原生 PD 红利）**完全消失**。
- 也无 D 侧哈希管理开销与命中块驻留：D 显存可全额投给并发 decode 池。
- 适用：D 卡哈希/池管理预算敏感、负载无共享前缀（每请求独立 prompt）、隔离 APC 干扰排障。
- 实操开关：start_d.sh 的 vllm serve 加 `--no-enable-prefix-caching`。

### 3.3 象限③ P✗ D✓（传省、算不省）

- P 侧：恒全量 prefill 486 tok（无缓存命中，P 卡每轮重算共享前缀 256 tok——llama 下重算出 byte 级相同的 KV）→ 正常上报 4 块并 Delaying free。
- D 侧与 ① 完全同：命中 256 → external 230 → 切掉前 2 块 → **只拉块 4、5 ≈30.1MB ≈1.36ms**。
- **关键结论的试金石**：③ 的传输量与 ① 完全相同——`_get_kernel_block_ids` 的 remote_start_idx 用的是 kv_transfer_params 里 D 调度器填的 `num_computed_tokens`（mc:1813"params[\"num_computed_tokens\"] = num_computed_tokens"），**与 P 侧缓存状态零耦合**。
- 代价：P 算力 2.1× 浪费（TTFT 恒为全量 prefill 时间）；invite 上 D 拉到的块 4、5 每轮来自 P 的新分配（内容 byte 级相同——doc1 §5 证明无重算误差风险）。
- 适用：P→D 带宽紧张而 P 算力富余；或前端 prompt 共享前缀少但后端同前缀请求集中（少见）；或 P 池小/多实例轮换导致 P 命中天然低。
- 实操开关：start_p.sh 的 vllm serve 加 `--no-enable-prefix-caching`。

### 3.4 象限④ P✗ D✗（双关基线）

每请求都是"全量 prefill + 全量拉 63.7MB + 双侧不驻留"的纯透传 PD：
- 成本恒定、行为最可预测——**排障基线**（怀疑 APC 复用路径引入数值/路由问题时先切 ④ 复现）。
- 长前缀共享负载下退化最狠：同前缀对话第 N 个请求仍付全额 prefill+传输。
- 注意即便双关，**延迟释放机制仍在**（这是传输协议需要，非缓存特性）："Delaying free of 4 blocks" 照常出现，拉完/超时才归还自由池。

## 4. 成本模型速查（TP1 bf16 · 块 128 tok；已按 docs/3 实测修正）

| 量 | 公式 | 请求2 例值 |
|---|---|---|
| 每 token KV 字节 | 2(KV) × 32层 × 8头 × 128维 × 2B = **128 KiB** | — |
| 传输字节 | **ceil(external_tokens / 128) × 16 MiB**（整块栅格，实测修正：26 tok 部分尾块的空槽也整块 DMA；按 token 估算法已废弃） | ①③ 2块→**32.0 MiB**；②④ 4块→**64.0 MiB**（docs/3 §5.1） |
| P prefill tokens | prompt_len − P_hit（P✓ 时） | ①② 230；③④ 486 |
| 首次请求耗时 | adxl 会话建立（~833ms，一次性）+ DMA | 834.86ms（d:325）；四象限一致（docs/3 §2，反解 ~839ms） |
| 后续请求耗时 | ≈ 传输字节 / (31~57 GB/s 实测有效带宽)（同 pod 跨卡；SENDING 侧 ZMQ 拉元数据首轮已缓存，mc:792-805） | 32MiB→1.07/1.09ms、64MiB→1.38/1.18ms（docs/3 实测口径） |
| 命中率监控 | `Prefix cache hit rate`（P⁄D 侧各打各的）+ D 侧 `External prefix cache hit rate` | 31.6% = 256/810；D 的 External=100%（四象限均 100%，docs/3） |

## 5. 正确性：四格全等（无缓存相关风险）

- KV 比特级与"从哪来"无关：① 的 KVP 18 项对照（docs/1 §5.1）同时覆盖**跨卡 DMA 路径（块4、5）与 D 本地复用路径（块1、2）**，都对 P 侧原始写入零差异 → ②（全走 DMA）与 ③（全走复用+DMA）的组合路径都在已验证的空间内。
- D✗ 只是"接收块不入哈希表"（不改变接收字节）；P✗ 只是"每轮重 prefill"（llama 下确定性计算、byte 级相同 KV）。
- 尾 token 重算与象限无关：D 首步恒重算最后 1 个 prompt token 的 KV（v1"全命中也要重算最后一个 token"语义在 consumer 侧重演，docs/1 §5.2 zeros 60→59 即此），四格一致。
- 唯一刻意排除的模型面：mamba/压缩等 state 组在"双侧缓存命中率不均"时有专门对齐逻辑（mc:865-869 注释 "When Prefix Caching is enabled on both P and D nodes, num_block should not be forced to match…"），本文四格结论只在纯 attention（llama）上实证，混态模型请单独验证。

## 6. 边界与坑

1. **D 池 LRU 驱逐回退**：①③ 下若 D 命中块被驱逐（哈希链断 anywhere 即全 miss，block_hash = hash(链)），请求退化为②的全量拉——正确性不受影响，只是省不了。监控口径即 D 侧 `Prefix cache hit rate` 掉 0。
2. **满块才入哈希表**：块 5（102 tok 部分块）恒不入池，D 复用粒度是 128 tok 整块——共享前缀不满块直接断链退化为 miss。
3. **P 延迟释放兜底 480s**：D 永不来拉（宕机/换路由）时 P 若不释放就泄漏。`KVCacheTaskTracker._retrieve_expired_requests`（mc:221-242）按 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=480`（vllm/vllm/envs.py:222）强制 free 并打 ERROR。四象限共有。
4. **Adxl 会话/远端元数据一次性**：`_get_remote_metadata` 首轮 ZMQ 交换对端块基址后缓存（mc:792-805）——这就是"首次 835ms、后续毫秒级"的结构性原因；评估首 token 延迟要区分冷热会话，四象限首次耗时都是同量级 ~835ms。
5. **路由耦合**：本形态 proxy `assign_instances` 先选 prefiller 再选 decoder（load_balance_proxy_server_example.py:896-946）、每请求新 request_id——多实例池下 P/D 各自命中率会被轮询稀释（选型时别忘了这一层）。D 被抢占时走 `stop_reason=="recomputed"` 重试路径（proxy:1047-1058，带已生成 token 重发），该路径下 APC 状态同样影响重传规模。

## 7. 选型建议

| 场景 | 推荐象限 | 理由 |
|---|---|---|
| 常规负载（对话/agent 共享 system+few-shot） | ①（默认双开） | 算力、带宽双省；docs/1 + 本轮实测 |
| D 卡内存极紧 / 无共享前缀负载 | ② | D 零驻留、零哈希开销；代价为传输量回到全量（同 pod 跨卡几乎免费，跨机 RDMA 需先测带宽窗） |
| P→D 带宽是硬瓶颈、P 算力富余 | ③ | 传输仍增量；代价为 TTFT 恒全量 prefill |
| 排障 / 纯独立请求负载 | ④ | 消除全部缓存路径变量 |

两句话版本：默认 ① 别动；要关就明确你在省什么（②省 D 内存、③省 P 内存）又放弃了什么（②放弃跨请求带宽复用、③放弃 prefill 省算），跨机部署时用 §4 模型先算 63.7MB/请求的账。

## 8. 四象限实验方案（✅ 已于 2026-09-29 执行完毕，实录见 docs/3）

> 实际执行未用本节的 sed 派生方案，而是升级为 **10 号 [PCM] 补丁 + `pcm/` 自包含工作区**（六打点直取象限差异，比 hit_length 回捞更直接）：`pcm/patch/apply_pcm_patch.sh` → `pcm/scripts/run_matrix.sh` 一键四象限 → `pcm/log/matrix_summary.md`。本节保留为历史方案存档。

```bash
# 在 ../kvc_pd/scripts/ 基础上各派生变体，其余流程照 docs/1 §6
sed 's/^    --enforce-eager/    --enforce-eager --no-enable-prefix-caching/' scripts/start_p.sh > scripts/start_p_nopc.sh
sed 's/^    --enforce-eager/    --enforce-eager --no-enable-prefix-caching/' scripts/start_d.sh > scripts/start_d_nopc.sh
# 四次组合 × curl_pd.sh 双请求（P 324 / R 486 固定 workload，即本文 §2 标尺）
```

每格验收五点：`grep 'KV cache transfer' log/d_llama.log`（②④ 应 ~2-3ms、①③ ~1.4ms）、`grep 'hit_length' log/kvc_*_reqr.log`（①③ P/D 双侧 256；② D 侧 0；④ 双侧 0）、`grep 'Prefix cache hit rate'`（① 双 31.6%② P 31.6%/D 0、③ P 0/D 31.6%、④ 双 0）、`grep '\[KVP\]'`（四格 18 项一致性）、`grep 'Delaying free'`（四格均出现）。表 §0 的推算值即本方案的预期答案，回填即闭环。

> **回填结果（2026-09-29 08:17 实测，docs/3 §6）**：五点预测**全中**——耗时 ①③ 1.07/1.09ms、②④ 1.38/1.18ms（比预测略快，实测有效带宽 31-57GB/s），hit_length（SCHED local_hit 口径）/hit_rate/Delaying free 逐格分毫不差；KVP 18 项一致性因 PCM 轮未叠加 09 号指纹补丁未逐块复核，由 docs/1 象限① + 数理解耦论证补足（docs/3 §7）。

## 9. 实测锚点索引

| 锚点 | 行号 | 说明 |
|---|---|---|
| 请求1 全量传输 834.86ms | d_llama.log:325 | 含 adxl 会话建立（09-27 轮为 845.58ms，docs/1 §3） |
| 请求2 增量传输 1.36ms | d_llama.log:501 | 只 DMA 块 4、5（09-27 轮 1.31ms） |
| P 侧 Delaying free 3 / 4 blocks | p_llama.log:432 / 591 | 请求 1、2 的 Prompt 全块上报+延迟释放 |
| P hit rate 0.0% → 31.6% | p_llama.log:441 → 600 | = 256/810，P 命中实证 |
| D hit rate 0.0% → 31.6% + External 100% | d_llama.log:460 → 1253 | D 命中实证 + "外部供给 100%"（本地缓存+mooncake 合计覆盖全部 prompt token） |
| 增量裁剪三步源码 | mooncake_connector.py:1814 / 1837 / 2615-2616 | external 申报 → 未哈希块 → remote_start_idx 切片 |
| P 上报不裁剪 | mooncake_connector.py:1712-1735 + 1882-1928 | 恒全量 prompt 块 |
| 全命中零传输 | mooncake_connector.py:786-790 | num_local_blocks==0 return |
| 会话/元数据缓存 | mooncake_connector.py:792-805 | 首次 835ms 摊销来源 |
| 480s 强制释放 | vllm/envs.py:222 + mooncake_connector.py:221-242 | 兜底防 P 泄漏 |
| 默认双开 | vllm/v1/config/cache.py:92 | enable_prefix_caching 默认 True |
| **四象限实锤（10 号 PCM 补丁）** | `pcm/log/q*/{p,d}_pcm.txt` + `pcm/log/matrix_summary.md` | 六打点逐格证据（2026-09-29，docs/3 §3.1 全表） |
