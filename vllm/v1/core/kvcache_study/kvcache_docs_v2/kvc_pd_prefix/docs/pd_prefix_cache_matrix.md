# PD Prefix Cache 开关矩阵：机制、四场景实测与场景卡

> 本文整合 kvc_pd 原 `docs/2_pd_prefix_cache_matrix.md`（理论推演）与原 `docs/3_pcm_quadrant_experiment.md`（PCM 实测实录）为单一文档，归档于 `kvc_pd_prefix/` 工作区。**理论、六打点补丁、四场景卡（每象限的配置 × [PCM] 打印全集 × 判读）、实测分析、成本模型、选型建议一站式覆盖**。
>
> **实测轮**:**09-30 贵安轮(权威,`log/round_0930_guian/`)**——gchtest 容器(贵安 a3 4 卡 10.30.13.28 / cloudideworkspaceapp-testaigz06a,P=npu:0/D=npu:1),03:25-03:31 四象限 trial-1 全 PASS,1P+1D MooncakeConnectorV1,workload 与 docs/1 同源(req_p 324 tok 种缓存 → req_r 486 tok 前缀复用 256 tok)。**09-29 乌兰轮(`log/round_legacy_0929am/`)同参数复测全部结论一致**(相差仅 ms 级抖动),作为交叉验证保留;乌兰下午重跑遭集群节点故障(附录 §8),已随 09-30 itask 切贵安全新复现收官。
>
> 统一口径:req_r prompt 486 tok = 4 块(块 1、2 = 256 tok 共享前缀,块 4、5 = 230 tok 新增);块 128 tok;TP1 bf16。

## 0. 30 秒结论:四象限总表(09-30 贵安轮;括号内 09-29 乌兰轮交叉值)

| 象限 | P 本地命中 | P prefill 实算 | P 上报块(恒全量) | D 本地命中 | mooncake 实际传输 | 第二次请求耗时* | 跨请求显存驻留 |
|---|---|---|---|---|---|---|---|
| ① P✓ D✓ **(默认)** | **256 tok(块 1,2)** | 230 tok | 4 块 [1,2,4,5] | **256 tok(块 1,2)** | 增量 2 块 = **32.0 MiB** | **1.12 ms**(1.07) | D 池保留命中块(LRU) |
| ② P✓ D✗ | **256 tok(块 1,2)** | 230 tok | 4 块 [1,2,4,5] | 0(无哈希表) | 全量 4 块 = **64.0 MiB** | **1.49 ms**(1.38) | 无(D 请求完即释放) |
| ③ P✗ D✓ | 0 | **486 tok(全量重算)** | 4 块 [4,5,6,7]† | **256 tok(块 1,2)** | 增量 2 块 = **32.0 MiB** | **1.11 ms**(1.09) | D 池保留命中块 |
| ④ P✗ D✗ | 0 | **486 tok(全量重算)** | 4 块 | 0 | 全量 4 块 = **64.0 MiB** | **1.19 ms**(1.18) | 无 |

\* 耗时指 mooncake `KV cache transfer` 行,会话热后纯 DMA;首次请求贵安轮 261-289ms(乌兰轮 838-855ms,adxl 会话建立+首建成本,均为一次性,§4.1 冷启动基线)。† P✗ 时 P 复用断链、全新复算的块号与 P✓ 时不同(实测两轮均如此:[6,7] 实拉细节见 §3 场景卡③)。

**四条铁律**:
1. **P 的开关只影响"算多少",D 的开关只影响"传多少"**——二者完全解耦(§1 机制底座);
2. vLLM v1 **默认双侧都开**(`enable_prefix_caching: bool = True`,vllm/vllm/config/cache.py:92)——线上与实测都处在象限①;
3. `P 上报块` 四格恒为全量 prompt 块——**P 不感知 D 的缓存**(§1.2);
4. 四格**正确性全等**(§6):APC 改变的是"KV 从哪来"。

## 1. 机制底座:prefix cache 在一条 PD 请求里的三个出场位置

### 1.1 P 侧:省 prefill 计算(与自己单机时的 APC 语义相同)

P 调度器对每个新请求先做本地前缀查找(`KVCacheManager.get_computed_blocks`,kv_cache_manager.py:196)——命中即跳过这部分 token 的 forward。实测 P 侧 hit rate 双请求后 **31.6%** = 256/810(810 = 324+486 两请求 prompt 总 tok),分毫不差。**P 的 request_finished 上报给 D 的块清单与 P 是否命中无关**——恒为"全部 prompt 块"(裁剪只按 prompt 长度裁掉 MTP/SWA 尾,`_get_transfer_block_ids` mc:1712-1735,不存在"按 D 命中裁剪")。

### 1.2 D 侧:省传输量(增量裁剪链,唯一的传输裁剪变量)

1. **调度器本地查找** → `num_computed_tokens`(D✓ 时 req_r 命中 256;D✗ 恒 0)
2. **connector 报增量**:`get_num_new_matched_tokens`(mc:1786-1822)`count = max(actual - num_computed_tokens, 0)`(1814 行)→ external = 486−D命中
3. **worker 裁剪远端块清单**:`_get_kernel_block_ids`(mc:2587-2618)`remote_start_idx = num_computed_tokens // remote_kernel_token_size; kernel_remote = kernel_remote[remote_start_idx:]`(2615-2616),注释原话 "Skip prefix-cached remote kernels (D-side already holds them)"——P 上报块清单按 D 命中数切头
4. **极端:全命中则零传输**(mc:786-790,`num_local_blocks == 0` 直接 return)

D 落卡接收目标是 `update_state_after_alloc` 里的**未哈希新块**(`blocks.get_unhashed_block_ids_all_groups()`,mc:1837)。

### 1.3 协议中枢:kv_transfer_params 经 proxy 从 P 响应流入 D 请求

proxy 双发次序:改写副本(`build_prefill_request` proxy:790-806,`max_tokens=1, stream=False, do_remote_decode=True`)→ P prefill 收官(FINISHED_LENGTH_CAPPED)→ `request_finished`(mc:1882-1928)三连门槛(mc:1897-1902,**必须 LENGTH_CAPPED 才交块**)返回 kv_transfer_params → proxy 提取(proxy:925)→ 注入原始请求发 D。详见 kvc_pd docs/4(请求生命周期)。

### 1.4 双侧独立 BlockPool、独立哈希链——两个开关解耦的结构性前提

同内容 token 的块哈希**构造链头 NONE_HASH 是进程级随机盐**(kv_cache_utils.py:99-113,`PYTHONHASHSEED` 未设时 `os.urandom(32)`)——P/D 两进程各摸一个,同 token 序列双侧哈希必然不同(实测:P `78e45fa0b2cc` 链 vs D `e482d4993ed0` 链)。**D 的命中只能来自 D 自己接收过并落卡的块**,P 种下的哈希对 D 不可见。传输搬运的是 KV 数据字节,不带任何哈希、块号(§4.3 块号语义)。另一佐证:块哈希链的消费方注明 "prefix caching and KV connectors"(kv_cache_utils.py:629-634)——D✗ 且 connector 在跑时 block_hashes 依然计算,只是不查表。

## 2. 实验方法:10 号 PCM 补丁与四象限编排

### 2.1 六打点([PCM] × 7 观察位,唯一触碰 mooncake_connector.py)

| 打点 | 位置(补丁后行号) | 打印内容 | 证明什么 |
|---|---|---|---|
| CFG | Worker `__init__` :2113 | role + **enable_prefix_caching** | 象限自证:开关真实生效的每实例一行 |
| SCHED | `get_num_new_matched_tokens` :1830 | req / prompt / **local_hit** / do_rp / do_rd | 每请求本地命中数:P 侧=省算依据,D 侧=传输抵扣依据 |
| ALLOC | `update_state_after_alloc` :1871 | **external** / **recv_blocks** / all_blocks | D 仅为未命中 token 分配的接收块清单 |
| PFINISH | `request_finished` :1948 | prompt / **report_blocks** / delay_free | P 上报的块清单(**恒全量**铁证) |
| XFER-entry | `_transfer_kv_cache_all_groups` :790 | recv_groups / pull_groups | 实拉对账(post-slice) |
| XFER-end | 同函数传输完成 :992 | segments / **bytes(MiB)** / eff_GBps / **pull_local/pull_remote** | 实传字节与两侧块号(D 传输量铁证) |

补丁独立于 kvc 01-09(同一文件不同内容集,可单独 apply 也可叠加);`gen_10_pcm_patch.py` 锚点断言 + py_compile + `patch --dry-run` + [PCM]x7 计数四重自检。

### 2.2 四象限编排(`scripts/run_matrix.sh`,单象限 ~2.5-3 min)

| 象限 | P 启动 | D 启动 | 开关注入 |
|---|---|---|---|
| q1_p1d1 | `start_p.sh log/q1 1` | `start_d.sh log/q1 1` | 无(默认双开) |
| q2_p1d0 | 1 | **0** | D 侧 `--no-enable-prefix-caching` |
| q3_p0d1 | **0** | 1 | P 侧 `--no-enable-prefix-caching` |
| q4_p0d0 | **0** | **0** | 双侧注入 |

每象限独立起停(进程间零状态污染:冷缓存种 324 tok → 第二请求分象限)。

## 3. 四张场景卡:配置 × [PCM] 打印全集 × 判读

> 每张卡 = 该象限第二请求(req_r)的**实际日志全集**(P 侧 5 行中 req_r 相关 + D 侧 9 行中 req_r 相关,原样从 `log/round_0930_guian/q*/{p,d}_pcm.txt` 提取)。grep 直达:`grep '\[PCM\]' log/q*/{p,d}_llama.log`。

### 场景卡①:P✓ D✓(默认基线)

**配置**:双侧不传任何开关(v1 默认)。

```
P 侧(grep 自 p_pcm.txt):
[PCM] CFG role=kv_producer enable_prefix_caching=True ...
[PCM] SCHED req=<req_r> prompt=486 local_hit=256 do_rp=False do_rd=True
[PCM] PFINISH req=<req_r> prompt=486 prompt_blocks=4 report_blocks=[4] delay_free=True

D 侧(grep 自 d_pcm.txt):
[PCM] CFG role=kv_consumer enable_prefix_caching=True ...
[PCM] SCHED req=<req_r> prompt=486 local_hit=256 do_rp=True do_rd=False
[PCM] ALLOC req=<req_r> external=230 recv_blocks=[[4, 5]] all_blocks=([1, 2, 4, 5],)
[PCM] XFER-entry req=<req_r> recv_groups=[2] pull_groups=[2]
[PCM] XFER-end req=<req_r> segments=64 bytes=33554432 (32.0 MiB) eff_GBps=29.94 pull_local=([4, 5],) pull_remote=([4, 5],)
KV cache transfer for request <req_r> took 1.12 ms.(原生行)
```

**判读**:P 命中 256(省算 230)→ 上报全 4 块;D 命中 256 → 切掉前 2 块,只拉 [4,5] 32 MiB;两侧块号巧合一致(P 新写/D 新分配)。**P 命中+D 命中=算力带宽双省**。

### 场景卡②:P✓ D✗(P 省算、传输全付)

**配置**:D 侧 `--no-enable-prefix-caching`(P 默认)。CFG 行自证:`D: enable_prefix_caching=False`。

```
P 侧:与①完全相同(local_hit=256 / 只算 230 / PFINISH report_blocks=[4])
D 侧:
[PCM] SCHED req=<req_r> prompt=486 local_hit=0 do_rp=True do_rd=False      ← 无哈希表,恒 miss
[PCM] ALLOC req=<req_r> external=486 recv_blocks=[[4, 5, 6, 7]] all_blocks=([4, 5, 6, 7],)
[PCM] XFER-end req=<req_r> segments=128 bytes=67108864 (64.0 MiB) eff_GBps=45.10 pull_local=([4, 5, 6, 7],) pull_remote=([1, 2, 4, 5],)
KV cache transfer ... took 1.49 ms.
```

**判读**:P 上报含**复用块 [1,2]**(pull_remote 铁证"恒全量");D 无表不裁剪 → 连 P 复用过的块也重传,分配全新 4 块。**跨请求零复用,同 pod 差 0.31ms,跨机 RDMA 下每请求 64 MiB 线路占用才是主代价**。另注意 segments=128(≠④ 的 64):P 侧 [1,2]+[4,5] 两组连续块与 D 侧 [4,5,6,7] 一组连续块对不齐,src/dst 段无法合并——段数本身是"块号拓扑"的指纹。

### 场景卡③:P✗ D✓(传省、算不省——解耦性的试金石)

**配置**:P 侧 `--no-enable-prefix-caching`(D 默认)。

```
P 侧:
[PCM] SCHED req=<req_r> prompt=486 local_hit=0 do_rd=True                  ← P 恒 miss,全量重算
[PCM] PFINISH req=<req_r> prompt=486 prompt_blocks=4 report_blocks=[4]
D 侧:与①完全相同(ALLOC external=230 recv=[4,5];XFER-end 32.0 MiB)
[PCM] XFER-end ... bytes=33554432 (32.0 MiB) eff_GBps=30.19 pull_local=([4, 5],) pull_remote=([6, 7],)   ← P 全新复算块号
KV cache transfer ... took 1.11 ms.
```

**判读**:**传输量与①分毫不差(32.0 MiB/2 块)——P 侧缓存状态对 D 传输量的影响为零**(差异只在 pull_remote 块号:P 无缓存全量复算时给自己分配 [4,5,6,7],被 D 切前 2 后剩 [6,7])。P 算力 2.1× 浪费。**"计算量只看 P 开关、传输量只看 D 开关"的最直接实证**。

### 场景卡④:P✗ D✗(双关基线)

**配置**:双侧 `--no-enable-prefix-caching`(CFG 行双 False)。

```
P 侧:同③(local_hit=0 / 全量算 / report_blocks=[4])
D 侧:同②(local_hit=0 / external=486 / recv 4 块)
[PCM] XFER-end req=<req_r> segments=64 bytes=67108864 (64.0 MiB) eff_GBps=56.28 pull_local=([4, 5, 6, 7],) pull_remote=([4, 5, 6, 7],)
KV cache transfer ... took 1.19 ms.
```

**判读**:全量重算+全量重传+双侧不驻留;双侧块号巧合一致(各自顺序分配)。行为最可预测的**排障基线**(消除一切缓存路径变量)。注意即便双关,**Delaying free 仍出现**(传输协议需要,非缓存特性)。

## 4. 实测结果分析(09-30 贵安轮;09-29 乌兰轮交叉一致)

### 4.1 首请求基线:四象限完全一致(冷启动不受开关影响)

req_p(324 tok 双侧冷缓存)四格 [PCM] 轨迹逐项相同:`local_hit=0 → external=324 → recv=[1,2,3] → XFER 3 块 48.0 MiB, eff_GBps≈0.2`(反解贵安轮 261~289ms / 乌兰轮 838~855ms——adxl 会话建立占绝对大头,一次性成本;贵安轮会话首建明显更快)。

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
| eff_GBps | 29.94 | 45.10 | 30.19 | 56.28 |
| eff_GBps(乌兰交叉) | 31.26 | 48.78 | 30.82 | 56.63 |
| took(第二请求) | **1.12 ms** | **1.49 ms** | **1.11 ms** | **1.19 ms** |
| took(乌兰交叉) | 1.07 ms | 1.38 ms | 1.09 ms | 1.18 ms |
| 日志 prefix hit(P/D) | 31.6% / 31.6% | 31.6% / **0.0%** | **0.0%** / 31.6% | 0.0% / 0.0% |
| D External hit | 100% | 100% | 100% | 100% |

### 4.3 三大新发现(超出理论推演部分)

1. **传输字节按整块计**:`bytes = ceil(external/128) × 16 MiB` 栅格(230 tok → 32.0 非 30.1;324→48.0;486→64.0)。mooncake 拉的是整个目标块,部分尾块空槽一起 DMA。**修正了原 docs/2 的 token 级公式**。
2. **同 pod 跨卡下"省传"绝对量小**:增量(33.9MB)与全量(67.1MB)差 0.11-0.31ms;实测有效带宽 30.8-56.6 GB/s(四象限波动同量级)。**真正的痛在跨机 RDMA 带宽窗**(25-50Gbps 网卡上 64 MiB/req = 10-20ms 线路占用)。
3. **块号系统独立**:pull_local/pull_remote 四种组合([4,5↔4,5]、[4,5,6,7↔1,2,4,5]、[4,5↔6,7]、[4,5,6,7↔4,5,6,7])——remote 是 P 池视角、local 是 D 池接收目标,两侧自由池互不联动。

### 4.4 推演→实测回填对照(全中)

传输量/耗时/命中率/P 上报恒全量等五点验收全部命中;原②④ "~2.8ms" 推算修正为 1.38/1.18ms(带宽 22→31-57GB/s);原 token 级字节公式修正为整块栅格;①耗时 1.36(指纹轮口径)→1.07(PCM 轮,同量级)。

## 5. 成本模型速查(TP1 bf16 · 块 128 tok;实测修正版)

| 量 | 公式 | req_r 例值 |
|---|---|---|
| 每 token KV 字节 | 2(KV) × 32层 × 8头 × 128维 × 2B = **128 KiB** | — |
| 传输字节 | **ceil(external_tokens/128) × 16 MiB**(整块栅格) | ①③ **32.0 MiB**;②④ **64.0 MiB** |
| P prefill tokens | prompt_len − P_hit(P✓ 时) | ①② 230;③④ 486 |
| 首次请求耗时 | adxl 会话建立(一次性)+ DMA | 贵安 261~289ms / 乌兰 838~855ms,四象限一致 |
| 后续请求耗时 | ≈ 传输字节 / (30~57 GB/s 实测有效) | 32MiB→1.11/1.12ms、64MiB→1.19/1.49ms |
| 命中率监控 | `Prefix cache hit rate`(P/D 各打各的)+ D 侧 `External prefix cache hit rate` | P/D 31.6%;D 的 External=100%(四格均 100%——D prompt 全由"本地缓存+mooncake"覆盖,从不自算 prefill) |

## 6. 正确性与边界

- KV 比特级与"从哪来"无关:①的 KVP 18 项对照(kvc_pd docs/1 §5.1)同时覆盖**跨卡 DMA 路径(块 4,5)与 D 本地复用路径(块 1,2)**,都对 P 原始写入零差异 → ②③ 的组合路径都在已验证空间内。
- D✗ 只是"接收块不入哈希表"(字节不变);P✗ 只是"每轮重 prefill"(确定性计算、同 KV)。
- 尾 token 重算与象限无关:D 首步恒重算最后 1 个 prompt token(docs/1 §1.2 bootstrap 语义)。
- 唯一排除面:mamba/混合"双侧命中率不均"有专门对齐逻辑(mc:865-869 注释),结论只在纯 attention(llama)上实证。

**边界与坑**:①D 池 LRU 驱逐回退(缓存块被逐→退化②行为,正确性无损);②满块才入哈希表(复用粒度 128 tok 整块,不满块断链全 miss);③P 延迟释放兜底 480s(`VLLM_KOUCAKE`…`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,envs.py:222`,D 不来拉时强制 free 防 P 泄漏);④adxl 会话/元数据一次性(首轮 ZMQ 交换后缓存,mc:792-805);⑤proxy 轮询稀释命中(assign_instances 每请求选实例,proxy:896-946),D 抢占走"recomputed"重试(proxy:1047-1058)。

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 常规负载(对话/agent 共享 system+few-shot) | ①(默认双开) | 算力、带宽双省 |
| D 内存极紧/无共享前缀 | ② | D 零驻留;代价为同 pod 0.3ms/跨机 64MiB 每请求 |
| P→D 带宽硬瓶颈、P 算力富余 | ③ | 传输仍增量;TTFT 恒全量 |
| 排障/纯独立请求 | ④ | 消除全部缓存路径变量 |

一句话:**默认①别动**;要关就明确省什么(②省 D 内存、③省 P 内存)放弃什么(②放弃跨请求带宽复用、③放弃 prefill 省算)。

## 8. 环境事故记录与复现(2026-09-29 下午)

**重跑尝试受阻记录**(数据已留 `log/` 容器侧,grep 可查):
- 15:26 轮(gggtest 迁移后):四象限 P 就绪全部超时,根因 `EngineCore` 初始化时 `*** stack smashing detected ***`——**迁移到的新节点驱动用户态(CANN 25.2.1)与镜像工具链 ABI 错配**(torch_npu import 即崩,npu-smi symbol lookup error);150s×4 全灭,源码与补丁零嫌疑(裸 vllm serve 同崩)。
- 16:36 轮(gggprobe 新建 2 卡):驱动兼容(ACL-OK)但**两卡 HBM 被外部僵尸进程占满**(~57GB/65536MB,`No running processes found`),vllm `Free memory(7.71/61.27 GiB) < 需求(49.02)`——节点级残留显存,容器内无法释放(npu-smi reset 权限不足)。
- 17:0x 轮(ggg4c/4d/4e 连摇):全部落在同一坏节点 33.215.118.32(调度器短期粘性),同样 HBM 脏,止损删除。

**恢复途径**(任一):等待节点 HBM 释放/平台清理后重跑;换时段(上午集群干净);联系平台 reset 节点。

**✅ 09-30 后记:贵安收官**——乌兰环境反复受阻后(当日 4 轮探测:2 卡池 x86 架构与 aarch64 镜像不匹配;4 卡池要么驱动 ABI 坏、要么 HBM 时净时脏——探测后 20 分钟内被动态租户吃掉 12-46 GiB;期间还暴露自身编排缺陷并修复:run_quadrant 失败路径不清理残留 vllm 致连环占卡、300s 干等超时),随 itask CLI 切贵安集群新建 gchtest(4 卡 A3,节点 10.30.13.28 全净、驱动 25.5.1.1 健康、模型 CSI 秒拉),修复版编排(HBM 防抢占等待 + 动态选卡 + 失败重试×2 + EXIT trap 兜底清理)一次跑通四象限 trial-1 全 PASS(03:25-03:31,约 6.5 分钟)——即本文档权威数据。

**复现命令**(环境就绪时,gchtest 或任意 2 卡健康 a3 pod):
```bash
cd /a3_inference/itask/workdir/gch02599191/kvc_pd_prefix
model-cli pull hcr...modelhub_74000048_meta-llama-3-8b:148700128_20260921221233   # CSI 秒级(层缓存)
patch/apply_pcm_patch.sh                       # 10 号补丁([PCM]x7)
bash scripts/run_matrix.sh                     # 四象限全自动 ~12min, tail -f log/matrix_run.log
patch/revert_pcm_patch.sh                      # 回退(md5 应回 00baf169f48fb167b9f6dfe650ac0ea5)
```

## 9. 产物索引(`kvc_pd_prefix/`,容器 NFS 与本地同步)

| 产物 | 说明 |
|---|---|
| `patch/gen_10_pcm_patch.py` + `10_pcm_prefix_cache_matrix.patch` | 补丁生成器(锚点断言/三重自检)与产物 |
| `patch/apply_pcm_patch.sh` / `revert_pcm_patch.sh` | 独立应用/回退 |
| `scripts/start_{p,d}.sh` | 参数化启动($2=0 注入 --no-epc) |
| `scripts/run_quadrant.sh` / `run_matrix.sh` | 单象限全流程 / 四象限一键 + 总表 |
| `log/round_0930_guian/` | **权威实测轮**(09-30 03:25-03:31):q1-q4 全套 12 文件 + matrix_run screen,trial-1 全 PASS |
| `log/round_legacy_0929am/` | **交叉验证轮**(09-29 08:07):q1-q4 全套,结论与贵安轮全一致(乌兰环境,含 §8 事故语境) |
| `log/matrix_summary_legacy.md` | 旧总表(历史) |
| `backup_before_nodereplace/` | 换节点前环境备份(usr_local_upper.tgz 470M + pip freeze) |

> 相关文档:kvc_pd `docs/1`(KV 传输正确性/KVP 指纹)、`docs/4`(请求生命周期与 token 归属)。
