# PD 分离 KVCache 传输正确性检查实验记录（1P+1D + mooncake）

> **实验目的**：检查 PD 分离场景下 KVCache 传输是否正确——P 侧（prefill）算出的 KVCache 经 mooncake 传到 D 侧（decode）后，与 P 侧**逐位一致**吗？传输路径是否丢字节、错位、篡改？
>
> **结论（2026-09-30 gggtest 容器实测轮，本轮）：PASS，传输零差错**——传输区 448/448 对块级 sha256 指纹全等（req_p 192 + req_r 256）；全部"不等"都落在 D 侧本地生成槽位（尾 token 补算 + decode 新写），本就不经过传输。
> **复现性**：09-29 首测轮（旧容器）与本轮两次独立实验 verdict.txt **逐字节一致（MD5 相同）**，生成文本逐 token 相同（§8）。

---

## 0. 一屏概览

| 项 | 值 |
|---|---|
| 检查对象 | mooncake adxl device 直传链路（P 卡0 → D 卡1）上的 KV 数据 |
| 判定手段 | 双侧对**同一 token 区间**的 KV 原始位取 sha256，逐层逐块对账（09 补丁） |
| 判定结果 | Tx 指纹 448/448 全等 → **传输逐位无损** |
| 架构 | 1P+1D：P=卡0/:8100/producer(rank0,20001)，D=卡1/:8200/consumer(rank1,20002)，proxy=:8000 同 id 双发 |
| 传输引擎 | mooncake-transfer-engine-npu 0.3.11.post1（adxl 直传，kv_buffer_device=npu） |
| 模型/配置 | Meta-Llama-3-8B bf16，TP1×2 实例（kv_heads=8, head_dim=128, block_size=128），seed=1024，prefix_caching=True |
| 软件栈 | vllm 0.23.0 + vllm-ascend 0.23.0（/vllm-workspace 源码仓直链） |
| 双请求 | req_p：324 tok，max_tokens=1——**纯传输探针**；req_r：486 tok（前缀继承 req_p），max_tokens=35——**前缀命中+decode 递进探针** |
| 一键复现 | `bash scripts/run_all.sh` → `python3 scripts/compare_fp.py`（自动裁决 → log/verdict.txt） |

---

## 1. 检查方法设计

> 本章回答两组问题：**检查什么、哪些区域该等/不该等**（§1.2 判定框架）与**怎么检**（§1.3 指纹探针 + §1.4 自动判据）。

### 1.1 为什么统计值不够，必须用"内容指纹"

旧一轮（09-27，旧版 08 补丁）用**统计值**（n/mean/std/min/max）对比双侧：完全一致。但这**不能证明逐位相等**——统计值是"多位数值的平均"，单 bit 翻转几乎不影响 4 位有效数字。旧轮其实已有可疑信号：req_p 的 V 张量 **zeros 计数 P=60 vs D=59，差 1**——至少一个元素两侧不同，统计值却全同。

要严格检查"传输是否有误"，最直接的办法是对**同一 token 区间的原始位模式**取哈希：两侧 sha256 相同 ⇔ 每一位都相同。09 补丁在 P/D 两实例跑同一份代码（`_kvc_kv_dump()`），对同区间 KV 原始字节（bf16 按 uint8 位视角）各取 sha256 前 16 hex。

### 1.2 三个必须预先厘清的比对前提

正确性检查不是"双侧全部字节拉平比对"——PD 架构决定了两侧的 KV **天然不该全等**。先明确边界，判定才有意义：

| # | 比对前提 | 实证依据 |
|---|---|---|
| 1 | **D 只接收前 p_tok-1 个 token 的 KV**——最后一个 prompt token（第 324 个，0 基 index 323）由 D 本地前向补算（PD bootstrap 惯例，D 以"正在计算"的状态衔接 prefill→decode） | 载入步布局行（kvc_d_reqp.log:19）：`|<comp>=0 |<ext_comp>=324(P传D) |<new>=0|` 账面全量声明 → 分配 3 块后，mooncake 异步写入 P 侧 **323 个 token**（index 0~322，=p_tok-1 物理整块 DMA；账面 324 按整块上限声明 vs 物理 323 差 1 tok）；补算步布局行（kvc_d_reqp.log:45）：`|<comp>=323 |<new>=1|` |
| 2 | **D 的 decode 新写块 P 侧不存在**（P 不做 decode），无比对对象 | 架构事实：decode 新块/新槽仅 D 写 |
| 3 | **两侧 BlockHash 不同**——prefix cache 是实例本地机制，跨侧一致性只能靠内容指纹、不能靠哈希对账 | 实例哈希盐独立：req_p 首块 P=bf3264861b0c vs D=6445dcc64516（§3.2/§3.4 入队日志） |

**前提 1 的两个深层细节**（免传为何是必然而非优化、两种口径为何不矛盾）：

| 细节 | 论证链 |
|---|---|
| **免传不是优化、传了也作废** | ① D 首步要产出第 1 个生成 token，必须把**最后一个 prompt token（第 324 个，index 323）**喂进 forward 拿 **logits**——KV cache 里只有注意力的 K/V、没有输出头结果，"零 forward 启动 decode"不存在；② QKV projection 是一个融合 GEMM——Q(末 token) 必算，**K/V(末 token) 是同一刀的免费副产品**；③ 标准 attention 内核语义是 **query 位的 K/V 由 projection 现算并写入槽位**（非 query 位才从 cache 读——末 token 的 forward 正是"1 个 query 位 + 读前 323 个 token 的 KV cache"）——不存在"读现成 KV(末 token)、只算 Q"的默认路径，拆开融合 GEMM 写专用内核得不偿失；④ vLLM 因此**完全**复用"差 1 tok 没算完的普通请求"路径（`num_computed=323 + num_new=1` → 一条普通 1-token prefill，零特判）。D 覆写值与原值 bf16 等价（批量 vs 单 tok kernel tiling 的 ~ULP 差，§1.1 zeros 60→59 即此） |
| **物理/逻辑口径区分** | 物理上 DMA 按整块传（3 块 48.0 MiB，../kvc_pd_prefix/docs/pd_prefix_cache_matrix.md §4.1 实测），末 prompt token（index 323）的槽位随尾块一起到达、随后被 D 覆写；"323"只是调度器逻辑账（num_computed_tokens），不矛盾。同一惯例三处同现：单机全命中也重算最后 1 tok / 本行 D 侧补算 / mamba 模型 P 侧 `_truncate_request_for_prefill` 显式只算到 h(N-1)（mc:1760-1784） |

由此得出**分区间判定框架**：

| 区间 | 写入者 | 是否应相等 |
|---|---|---|
| **Tx 传输区**（前 p_tok-1 tok 的槽位） | P prefill 写 → **本次**传输 → D 落卡 | **必须逐位相等**（不等 = 传输有误） |
| **D 缓存命中块**（如 req_r 的 blk1/2，本次未传输） | P 本地缓存驻留 vs D"前次传输落卡 → 缓存驻留" | **必须逐位相等**（不等 = 某侧缓存驻留路径有误——传输嫌疑已被前次请求排除） |
| 尾块最后 1 槽（第 p_tok 个 token） | P 批量 prefill vs D 单 token 补算 | 允许不等（同一数学量、不同 kernel 路径的 ULP 级差） |
| decode 新块/新槽 | 仅 D 写 | 不比对 |

> 缓存命中行是**链式论证**：req_p 轮先证"同源数据经 DMA 无损落卡"，req_r 轮再证"两侧各自驻留（P 池保留 / D 落卡+哈希挂链）期间无篡改"（blk1/2 的 128 对指纹全等即证此层）——这正是双请求递进设计的用意。

### 1.3 09 补丁的三层指纹探针

| 行标 | 内容 | 用途 |
|---|---|---|
| `[KVC][KVP][FPB]`（块指纹） | 每层每块两条：**Tx**=该块传输区（前 `p_tok-1` tok 覆盖的槽位）、**Xx**=该块全部已写槽位 | **裁决核心**，定位到块粒度 |
| `[KVC][KVP][FP]`（层指纹） | 每层 K.prompt/K.all、V.prompt/V.all | 层粒度总对账（prompt 区含补算槽，全等层数少是预期） |
| 统计行/首 3 值（08 基线） | mean/std/min/max + K示/V示 | 弱校验参考；ULP 级差不可见，反衬指纹必要性 |

### 1.4 自动判据（scripts/compare_fp.py）

```
[PASS] 所有层所有块的 Tx 指纹 P/D 全等
      且 Xx 不等仅出现于"prompt 尾块最后槽 + decode 新块"（本地生成区）
[FAIL] 任一 Tx 指纹不等（传输丢字节/错位），或传输区内 Xx 不等
```

---

## 2. 部署与时间线

**拓扑**（容器 gggtest，itask 4×hpu910a3 pod，workdir /a3_inference/itask/workdir/wsl02075301，pod IP 172.16.210.194）：

| 实例 | 卡 | 端口 | kv 角色 | adxl engine | KV 显存/块数 |
|---|---|---|---|---|---|
| P | npu:0 | 8100 | producer rank0 / listening tcp://172.16.210.194:20001 | 172.16.210.194:20238 | 33.78 GiB / 2161 块 |
| D | npu:1 | 8200 | consumer rank1 / 20002 | 172.16.210.194:20377 | 33.79 GiB / 2162 块 |
| proxy | — | 8000 | 同 request body 双发 P/D | — | — |

**时间线（run_all_screen.log，全程 2 分 02 秒）**：

| 时刻 | 事件 |
|---|---|
| 07:59:27 | run_all 开始：打补丁 kvc 01~08 + 09（8 文件 170 行 [KVC] + 指纹，dry-run 全过；04 补丁含 allocate_slots 五段布局行增强版） |
| 07:59:59~08:00:01 | P 建池（33.78 GiB / 2161 块）→ adxl 注册(:20238) → SendingThread 监听 20001（p:232）→ 就绪（等待 50s，p:296 startup complete） |
| 08:00:47 | D 建池（33.79 GiB / 2162 块, adxl 172.16.210.194:20377）→ 就绪（等待 40s，d:294） |
| 08:00:58 | proxy 就绪（1 prefill + 1 decode client, worker 98662） |
| 08:01:01~02 | req_p 完成（P TERM 快照 → D 载入 ext_comp=324 / DMA 282.94ms(d:330) → 补算"为了" → 双侧释放） |
| 08:01:10~11 | req_r 完成（双侧 HIT[1,2] → P 增量 230、D 载入 ext_comp=230 / DMA 1.29ms(d:508) → decode 35 步 → 释放） |
| 08:01:29 | 杀服务 → revert 09→08→01~07，源码 [KVC] 全归零，零进程残留 |

启动期 [CFG]/[L1]/[L2~L5] 打印与单机实验同构，不再展开；本档聚焦**运行期传输对账**。

---

## 3. 检查一：req_p（324 tok，max_tokens=1 —— 纯传输探针）

**设计意图**：max_tokens=1 让 D 侧几乎不做 decode——323 tok 的 KV 全部来自传输，把"传输正确性"从其他变量里单独剥离出来。

**全流程时间线**（P/D/proxy 三组件对照；行号以 log/ 落盘文件为准——p/d_llama.log 启动期各含 5 段 tqdm 进度条产生的孤立 `\r`，已规范化为独立行）：

| 时刻 | 组件 | 事件 |
|---|---|---|
| 07:59:27 | run_all | 开始：打补丁 kvc 01~08 + 09（8 文件 170 行 [KVC] + 指纹，dry-run 全过；04 补丁本轮增强——allocate_slots 新增**五段布局行** comp/new_comp/ext_comp/new/lookahead，ext_comp 即"P 传 D 的 KV tok 账面"） |
| 07:59:59~08:00:01 | P | 建池 33.78 GiB / 2161 块 → adxl 注册(172.16.210.194:20238) → SendingThread 监听 20001（p:232）→ 就绪（等待 50s，p:296） |
| 08:00:47 | D | 建池 33.79 GiB / 2162 块 → adxl 注册(:20377) → 就绪（等待 40s，d:294） |
| 08:00:58 | proxy | 就绪（Initialized 1 prefill + 1 decode client, worker 98662） |
| 08:01:01~02 | req_p | P 全量 prefill 324 tok + TERM 快照 → Delaying free 3 块(p:437) → 200 OK 串行点(p:438)；D 载入步 `ext_comp=324` 备块[1,2,3] → mooncake DMA 282.94ms(d:330) → 补算步出首 token"为了" → 双侧释放（P p:439~445 / D d:458~464） |
| 08:01:10~11 | req_r | 双侧前缀 HIT [1,2]（hit_length=256）→ P 只算增量 230 tok；D 载入步 `ext_comp=230`、DMA 1.29ms(d:508) → 补算 + decode 35 步（comp 485→520，:568 跨界申请 blk6）→ TERM blocks=[1,2,4,5,6] region=520/520 → 双侧释放（P p:598~605 / D d:820~824） |
| 08:01:29 | 收尾 | 杀服务 → revert 09→08→01~07，源码 [KVC] 全归零（全程 2 分 02 秒） |

### 3.1 proxy：收单与改写双发（proxy.log）

```
INFO:     ::1:45036 - "GET /healthcheck HTTP/1.1" 200 OK
INFO:     ::1:45046 - "POST /v1/completions HTTP/1.1" 200 OK        ← req_p：客户端只打 proxy(:8000) 一枪
```

流程：proxy 收单后复制改写出**哑请求**（max_tokens=1、stream=False、do_remote_decode=True）发 P:8100；P 响应后提取 `kv_transfer_params`(:925)，再把**原始请求**注入发 D:8200。改写是 DEBUG 级不可见，铁证在 3.2/3.4 两侧入队参数对照（proxy:790-806 只改 `req_data.copy()` 副本）。

### 3.2 P：哑请求一步走完（kvc_p_reqp.log:2~39 原样）

```
p_reqp:2     [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=bf3264861b0c
p_reqp:3     [KVC][ENQ] 入队 hash_block_tokens: parent=bf3264861b0c, tokens=128 -> BlockHash=fca18393830e
p_reqp:4     [KVC][ENQ] Request(request_id=cmpl-...-82f39907) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['bf3264861b0c', 'fca18393830e']
p_reqp:11    [KVC][L3] 前缀查找   第 1 块 MISS: BlockHash=bf3264861b0c -> break
p_reqp:18    [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-...-82f39907, num_new_tokens=324(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=0(comp), request.num_tokens=324, delay_cache_blocks=False
p_reqp:19    [KVC][L5] 分配布局: |<comp>=0 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=324 |<lookahead>=0| num_local_computed_tokens=0 total_computed_tokens=0 to_be_computed=324
p_reqp:25    [KVC][L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=2157
p_reqp:31    [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=bf3264861b0c, group_id=0) <- KVCacheBlock(block_id=1), map size=1
p_reqp:32    [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=fca18393830e, group_id=0) <- KVCacheBlock(block_id=2), map size=2
p_reqp:39    [KVC][KVP] TERM req=cmpl-...-82f39907 dev=npu:0 逐层按块: layers=32 blocks=[1, 2, 3] region=324/324 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=8, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
```

流程讲解：① 入队 :2~:4 算出两条满块哈希（盐链 bf32648…/fca183…），:11 前缀查找首块 MISS（P 冷启动）；② :18/:19 进入分配——**布局行 `|<comp>=0 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=324 |`：P 不收外部 KV，全量 324 tok 本地前向**（to_be_computed=324，ext_comp 恒 0——P 是供给方）；③ :25 新块 [1,2,3] 全新分配，:31/:32 两条满块哈希种入 P 本地表；④ :39 TERM blocks=[1,2,3] region=324/324——释放前物理快照（**指纹对账的 P 侧数据源**；哑 token 已采样丢弃）。

### 3.3 P 收官：上报块清单 → 延迟释放 → done 回流真正释放（p_llama.log:437~445）

```
p:437  [mooncake_connector.py:1910] Delaying free of 3 blocks for request cmpl-...-82f39907
p:438  INFO: 127.0.0.1:46638 - "POST /v1/completions HTTP/1.1" 200 OK
p:439  [kv_cache_manager.py:566] [KVC][L5] ======== 释放 ========
p:440  [kv_cache_manager.py:568] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-...-82f39907, 释放前持有 block_table=([1, 2, 3],)
p:441  [single_type_kv_cache_manager.py:405] [KVC][L3] 释放 SingleTypeKVCacheManager.free: req=cmpl-...-82f39907, 持有 blocks=[1, 2, 3] (reversed 后释放)
p:442  [block_pool.py:521] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(3, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 3 块 [3, 2, 1], append_n -> 队尾(LRU保护)
p:443  [kv_cache_utils.py:391] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[3, 2, 1]), num_free_blocks=2160
p:444  [kv_cache_coordinator.py:299] [KVC][L4] 释放 KVCacheCoordinator.free: req=cmpl-...-82f39907 已逐组下放第3层释放
p:445  [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

流程（"延迟释放"的真实含义与兑现）：

- **:437 延迟登记**——`request_finished` 三连门槛（mc:1897-1902：params 非 None + do_remote_decode + FINISHED_LENGTH_CAPPED）通过：块 [1,2,3] 清单（remote_block_ids 等 11 字段）随 :438 响应体（kv_transfer_params）交 proxy → 注入 D；块**驻留 P 池等 DMA**——"延迟"本体；:438 是**串行点**（proxy 此后才联系 D，D 侧首条日志与 P 收官同秒）。
- **:439~:445 真正释放**——触发器是 **D 拉完回 done**（D 在 d:330 拉完 282.94ms，经 side channel 回流；mc:755-758 发送 / mc:181-192 弹登记）：五层栈 `KVCacheManager.free(:440) → SingleType…free(:441) → BlockPool.free_blocks(:442) → FreeKVCacheBlockQueue.append_n(:443) → Coordinator 完成(:445)`，**延迟窗口 ≈1s**。兜底：D 失联超 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`（默认 480s，vllm/envs.py:222）强制释放防泄漏（mc:221-246），本轮未触发。
- **:442/:443 数字细节**——三块 ref_cnt−1 归零回收 [3,2,1]；归还后 `num_free_blocks=2160`（P 池 2161 − mooncake 保留 1）。**释放 ≠ 失效**——块 append 队尾(LRU) + 哈希映射保留，req_r 时 P 前缀 HIT [1,2] 由此来（§4.1）。

req_r 侧同构（p_llama.log:597~605，同一秒内完成，D 拉取仅 1.29ms）：`Delaying free of 4 blocks(:597, 块[1,2,4,5]) → 同一释放栈(:600~605，归还 [5,4,2,1]、num_free_blocks=2160)`。

### 3.4 D：载入步收 DMA + 补算步出首 token（kvc_d_reqp.log + d_llama.log:330 原样）

```
d_reqp:2     [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=6445dcc64516
d_reqp:3     [KVC][ENQ] 入队 hash_block_tokens: parent=6445dcc64516, tokens=128 -> BlockHash=ae2c4445bc09
d_reqp:4     [KVC][ENQ] Request(request_id=cmpl-...-98406ba9) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['6445dcc64516', 'ae2c4445bc09']
d_reqp:11    [KVC][L3] 前缀查找   第 1 块 MISS: BlockHash=6445dcc64516 -> break
d_reqp:18    [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-...-98406ba9, num_new_tokens=0(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=324(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=0(comp), request.num_tokens=324, delay_cache_blocks=True
d_reqp:19    [KVC][L5] 分配布局: |<comp>=0 |<new_comp>=0 |<ext_comp>=324(P传D) |<new>=0 |<lookahead>=0| num_local_computed_tokens=0 total_computed_tokens=324 to_be_computed=0
d_reqp:28    [KVC][L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=2158
d_reqp:38    [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=6445dcc64516, group_id=0) <- KVCacheBlock(block_id=1), map size=1
d_reqp:39    [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=ae2c4445bc09, group_id=0) <- KVCacheBlock(block_id=2), map size=2
d:330   [mooncake_connector.py:973] KV cache transfer for request cmpl-...-82f39907 took 282.94 ms. local_ip 172.16.210.194 local_device_id 0 remote_session_id 172.16.210.194:16615
d_reqp:44    [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-...-98406ba9, num_new_tokens=1(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=323(comp), request.num_tokens=324, delay_cache_blocks=False
d_reqp:45    [KVC][L5] 分配布局: |<comp>=323 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=1 |<lookahead>=0| num_local_computed_tokens=323 total_computed_tokens=323 to_be_computed=1
d_reqp:53    [KVC][L5] S3 块未满, 无需分配新块 (req=cmpl-...-98406ba9, num_new_tokens=1)
d_reqp:60    [KVC][KVP] TERM req=cmpl-...-98406ba9 dev=npu:0 逐层按块: layers=32 blocks=[1, 2, 3] region=324/324 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=8, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
d_reqp:61    [KVC][KVP] TERM L00 blk=1[满:128](128,8,128) blk=2[满:128](128,8,128) blk=3[未满:68](68,8,128) | K示(首块首token前3)=[0.5078, 0.9336, 0.9219] 统计[n=331776] mean=0.01352 std=1.453 min=-14.56 max=10.81 | V示(首块首token前3)=[0.01538, 0.0008049, 0.03345] 统计[n=331776] mean=0.0003884 std=0.0374 min=-0.3477 max=0.4707
```

流程讲解（载入 + 补算两步、与 P 的三处对照）：

- **入队哈希 6445d…/ae2c4… ≠ P 的 bf326…/fca183…**——同 token 序列、两进程两盐两链（§1.2 前提 3 的原始出处）。
- **载入步 :18/:19**——`num_new_tokens=0` + **布局行 `|<comp>=0 |<new_comp>=0 |<ext_comp>=324(P传D) |<new>=0 |`：ext_comp 账面全量声明 P 传 324 tok**（ext_comp=connector 外部已算 = PD 分离下 P→D 的 KV 抵扣账）；:28 三块接收目标 [1,2,3] 全新分配（S3 新块为空——块不由 forward 写），**d:330 mooncake DMA 实拉 48.0 MiB / 282.94ms**（含 adxl 会话建立）；:38/:39 落卡后满块种入 **D 自己的**哈希表——req_r 前缀命中种子。**三层口径对账：ext_comp 账面 324 / 物理指纹实测 323（Tx=前 p_tok-1 tok）/ 补算步落账 323**——账面按整块上限声明、物理整块 DMA、逻辑差 1 tok 留给补算（§1.2 前提 1）。
- **补算步 :44/:45**——布局行 `|<comp>=323 |<new>=1 |`：调度器把请求摆成"差 1 tok 没算完的普通请求"，D 本地 forward 第 324 个 token（index 323，attention 读前 323 tok 传输 KV）→ 采样首 token——resp_p.json 的"为了"；:53 块未满无需新块；:60/:61 TERM blocks=[1,2,3] region=324/324，blk3[未满:68] = 载入 67 + 补算 1 互证。

### 3.5 D 收官：TERM 指纹快照 + 同一释放栈（d_llama.log:453~464）

```
d:453  [model_runner_v1.py:2611] [KVC][KVP] ======== 请求结束, 物理 cache 打印完毕 ========
d:458  [kv_cache_manager.py:566] [KVC][L5] ======== 释放 ========
d:459  [kv_cache_manager.py:568] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-...-98406ba9, 释放前持有 block_table=([1, 2, 3],)
d:461  [block_pool.py:521] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(3, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 3 块 [3, 2, 1], append_n -> 队尾(LRU保护)
d:462  [kv_cache_utils.py:391] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[3, 2, 1]), num_free_blocks=2161
d:464  [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

流程：D 与 P 走**完全相同**的五层释放栈（此时 P 已归还完毕）；:453 TERM 打印完毕 = **指纹对账的 D 侧数据源**（P 侧对应 p:432）。:461 归零回收 [3,2,1]、:462 归还后 `num_free_blocks=2161`（D 池 2162 − mooncake 接收缓冲 1；对照 P :443 的 2160——D 的 mooncake 缓冲张量大 256KiB）。LRU 队尾 + 哈希映射保留 → req_r 时 D 前缀 HIT 的种子（§4.1）。

### 3.6 块级指纹对账（32 层 × 3 块 × K/V = 192 对 Tx）——双侧原样

```
kvc_p_reqp.log:41  [KVC][KVP][FPB] TERM L00 块指纹: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/K.Xx=759d20483db8a502 blk3K.Tx=2a2c157641418bfa/K.Xx=85d795e93bd122e1 | blk1V.Tx=5909248f73a62d39/V.Xx=5909248f73a62d39 blk2V.Tx=5db451ce3b0ae393/V.Xx=5db451ce3b0ae393 blk3V.Tx=04fd71d2dcd7e433/V.Xx=7635560bb2914987
kvc_d_reqp.log:62  [KVC][KVP][FPB] TERM L00 块指纹: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/K.Xx=759d20483db8a502 blk3K.Tx=2a2c157641418bfa/K.Xx=85d795e93bd122e1 | blk1V.Tx=5909248f73a62d39/V.Xx=5909248f73a62d39 blk2V.Tx=5db451ce3b0ae393/V.Xx=5db451ce3b0ae393 blk3V.Tx=04fd71d2dcd7e433/V.Xx=7635560bb2914987
kvc_p_reqp.log:44  [KVC][KVP][FPB] TERM L01 块指纹: blk1K.Tx=9582307cb75a1af8/K.Xx=9582307cb75a1af8 blk2K.Tx=cf340e4c1e28b0be/K.Xx=cf340e4c1e28b0be blk3K.Tx=95658bda922a290b/K.Xx=6cb10022565c349a | blk1V.Tx=ef8da3ddd73cd3f5/V.Xx=ef8da3ddd73cd3f5 blk2V.Tx=084a8e4d37c792aa/V.Xx=084a8e4d37c792aa blk3V.Tx=b8c3c0d0c4e0f78b/V.Xx=8cec55805753eda9
kvc_d_reqp.log:65  [KVC][KVP][FPB] TERM L01 块指纹: blk1K.Tx=9582307cb75a1af8/K.Xx=9582307cb75a1af8 blk2K.Tx=cf340e4c1e28b0be/K.Xx=cf340e4c1e28b0be blk3K.Tx=95658bda922a290b/K.Xx=36c4fb7d3ebe8950 | blk1V.Tx=ef8da3ddd73cd3f5/V.Xx=ef8da3ddd73cd3f5 blk2V.Tx=084a8e4d37c792aa/V.Xx=084a8e4d37c792aa blk3V.Tx=b8c3c0d0c4e0f78b/V.Xx=f986f9bc9f9f78ab
```

流程讲解（以上述原文为准）：

- **L00（kvc_p_reqp.log:41 vs kvc_d_reqp.log:62）**——12 个指纹字段（3 块 × K/V × Tx/Xx）**逐字符全同**，连 blk3 的 Xx 都相同（32 层中唯一）——D 补算重写第 68 槽时该层恰好 bit 级命中，kernel tiling 差异未改变 bf16 结果。
- **L01 起其余 31 层（:44 vs :65）**——blk1/blk2 的 Tx 与 Xx 全等（整块都来自传输）；**blk3 Tx 同、Xx 异**：blk3K.Tx=2a2c157641418bfa 两侧逐字相同（Tx 只哈希前 323 tok 覆盖的 67 槽载入区 ⇒ 传输无损——"D 落卡字节 = P 原始字节"的 sha256 级证据），而 K.Xx=6cb10022565c349a(P) vs 36c4fb7d3ebe8950(D)、V.Xx=8cec55805753eda9(P) vs f986f9bc9f9f78ab(D)（Xx 哈希全部 68 槽，第 68 槽是 D 补算重写的 ULP 级差值 ⇒ 必然不同）。

| 区间 | 槽位数 | P 侧来源 | D 侧来源 | 对账 |
|---|---|---|---|---|
| blk1 Tx（tok 0~127） | 128 | P prefill | mooncake 传输 | **32 层全等** |
| blk2 Tx（tok 128~255） | 128 | P prefill | mooncake 传输 | **32 层全等** |
| blk3 Tx（tok 256~322） | 67 | P prefill | mooncake 传输 | **32 层全等** |
| blk3 Xx 第 68 槽（index 323） | 1 | P 批量 prefill | D 单 token 补算 | L00 恰好同，L01~L31 异（ULP 级） |

**192/192 Tx 全等 → 本轮传输无误差**。两个佐证细节：

- 层指纹 `K.prompt` 全等层数仅 1/32（恰好 L00）——层指纹的 prompt 区含补算槽，每层 1 个 token 的 bit 差就毁掉整层哈希；而**统计行 32/32 全同**——ULP 级差对统计不可见。两相对照正是"必须用指纹"的实证。
- 响应：resp_p.json 生成"为了"（completion_tokens=1）——D 侧基于"传输 KV+补算槽"采样，语义正确。

---

## 4. 检查二：req_r（486 tok，max_tokens=35 —— 前缀命中 + decode 递进探针）

**设计意图**：在"纯传输"成立后叠加两个新变量——（a）双侧**本地前缀缓存**命中参与供数（KV 不再全部直接来自本次传输）；（b）D 侧 35 步 decode 持续写入新块。检验混合来源下传输数据是否仍然一致。

### 4.1 双侧前缀查找与分工（kvc_p_reqr.log / kvc_d_reqr.log / d_llama.log:508，08:01:10~11）

**P 侧——命中自己种的块，只算增量（kvc_p_reqr.log:2~50 原样）：**

```
p_reqr:2     [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=bf3264861b0c
p_reqr:3     [KVC][ENQ] 入队 hash_block_tokens: parent=bf3264861b0c, tokens=128 -> BlockHash=fca18393830e
p_reqr:4     [KVC][ENQ] 入队 hash_block_tokens: parent=fca18393830e, tokens=128 -> BlockHash=ef148858d679
p_reqr:5     [KVC][ENQ] Request(request_id=cmpl-...-b8af9dad) 入队: num_prompt_tokens=486, max_tokens=1, 满块链式哈希 BlockHash × 3: ['bf3264861b0c', 'fca18393830e', 'ef148858d679']
p_reqr:13    [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=fca18393830e, group_id=0) -> HIT KVCacheBlock(block_id=2)
p_reqr:16    [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=ef148858d679, group_id=0) -> MISS
p_reqr:18    [KVC][L3] 前缀查找   第 3 块 MISS: BlockHash=ef148858d679 -> break
p_reqr:20    [KVC][L4] 前缀查找 UnitaryKVCacheCoordinator.find_longest_cache_hit 返回: hit_blocks=[[1, 2]], hit_length=256
p_reqr:25    [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-...-b8af9dad, num_new_tokens=230(new), num_new_computed_tokens=256(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=0(comp), request.num_tokens=486, delay_cache_blocks=False
p_reqr:26    [KVC][L5] 分配布局: |<comp>=0 |<new_comp>=256 |<ext_comp>=0(P传D) |<new>=230 |<lookahead>=0| num_local_computed_tokens=256 total_computed_tokens=256 to_be_computed=230
p_reqr:33    [KVC][L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
p_reqr:36    [KVC][L2] S3 BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=2156
p_reqr:37    [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-...-b8af9dad, num_tokens=486, block_size=128, 需 4 块 - 已有 2 = 新分配 2 块 [4, 5], 持有 req_blocks=[1, 2, 4, 5]
p_reqr:49    [KVC][KVP] TERM req=cmpl-...-b8af9dad dev=npu:0 逐层按块: layers=32 blocks=[1, 2, 4, 5] region=486/486 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=8, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
p_reqr:50    [KVC][KVP] TERM L00 blk=1[满:128](128,8,128) blk=2[满:128](128,8,128) blk=4[满:128](128,8,128) blk=5[未满:102](102,8,128) | K示(首块首token前3)=[0.5078, 0.9336, 0.9219] 统计[n=497664] mean=0.02175 std=1.457 min=-14.56 max=10.81 | V示(首块首token前3)=[0.01538, 0.0008049, 0.03345] 统计[n=497664] mean=0.0003654 std=0.03763 min=-0.3477 max=0.4707
```

流程讲解：① 入队 :2~:5 算出 3 条满块哈希链（首两条与 req_p 完全相同——同 token 同盐必然同链；第 3 条 ef148858d679 是 230 tok 新增量链）；② :13/:16 命中对照——HIT blk1、HIT blk2、第 3 块 MISS，:20 返回 hit_length=256；③ :25/:26 进入分配——**布局行 `|<comp>=0 |<new_comp>=256 |<ext_comp>=0 |<new>=230 |`：P 命中 256 后只本地前向 230 tok**（P 侧 ext_comp 恒 0）；④ :33 touch [1,2] 保活（ref_cnt+1）、:36 新块 [4,5]、:42 第三满块种入 P 表（map size=3）、:45 S4 提交 num_computed_tokens=486；⑤ :49/:50 TERM blocks=[1,2,4,5] region=486/486（2 复用+2 新算），终态 blk5[未满:102]。P 的 max_tokens=1 哑请求口径不变。

**D 侧——命中自己接收时种的块，只收增量（kvc_d_reqr.log + d_llama.log:508 原样）：**

```
d_reqr:2     [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=6445dcc64516
d_reqr:3     [KVC][ENQ] 入队 hash_block_tokens: parent=6445dcc64516, tokens=128 -> BlockHash=ae2c4445bc09
d_reqr:4     [KVC][ENQ] 入队 hash_block_tokens: parent=ae2c4445bc09, tokens=128 -> BlockHash=57a37c2a699b
d_reqr:5     [KVC][ENQ] Request(request_id=cmpl-...-9f3122d1) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 BlockHash × 3: ['6445dcc64516', 'ae2c4445bc09', '57a37c2a699b']
d_reqr:10    [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=6445dcc64516, group_id=0) -> HIT KVCacheBlock(block_id=1)
d_reqr:13    [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=ae2c4445bc09, group_id=0) -> HIT KVCacheBlock(block_id=2)
d_reqr:18    [KVC][L3] 前缀查找   第 3 块 MISS: BlockHash=57a37c2a699b -> break
d_reqr:20    [KVC][L4] 前缀查找 UnitaryKVCacheCoordinator.find_longest_cache_hit 返回: hit_blocks=[[1, 2]], hit_length=256
d_reqr:25    [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-...-9f3122d1, num_new_tokens=0(new), num_new_computed_tokens=256(new_comp), num_external_computed_tokens=230(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=0(comp), request.num_tokens=486, delay_cache_blocks=True
d_reqr:26    [KVC][L5] 分配布局: |<comp>=0 |<new_comp>=256 |<ext_comp>=230(P传D) |<new>=0 |<lookahead>=0| num_local_computed_tokens=256 total_computed_tokens=486 to_be_computed=0
d_reqr:33    [KVC][L3] S2 SingleTypeKVCacheManager.allocate_new_computed_blocks: req=cmpl-...-9f3122d1, touch 命中块 [1, 2], 此前请求持有块数=0
d_reqr:34    [KVC][L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
d_reqr:35    [KVC][L2] S3 BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=2157
d_reqr:39    [KVC][L5] S3 allocate_new_blocks: req=cmpl-...-9f3122d1, num_tokens_need_slot=486 -> 新块 []
d:508   [mooncake_connector.py:973] KV cache transfer for request cmpl-...-b8af9dad took 1.29 ms. local_ip 172.16.210.194 local_device_id 0 remote_session_id 172.16.210.194:16615
d_reqr:45    [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=57a37c2a699b, group_id=0) <- KVCacheBlock(block_id=4), map size=3
d_reqr:50    [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-...-9f3122d1, num_new_tokens=1(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=485(comp), request.num_tokens=486, delay_cache_blocks=False
d_reqr:51    [KVC][L5] 分配布局: |<comp>=485 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=1 |<lookahead>=0| num_local_computed_tokens=485 total_computed_tokens=485 to_be_computed=1
d_reqr:59    [KVC][L5] S3 块未满, 无需分配新块 (req=cmpl-...-9f3122d1, num_new_tokens=1)
d_reqr:67    [KVC][L5] 分配布局: |<comp>=486 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=1 |<lookahead>=0| num_local_computed_tokens=486 total_computed_tokens=486 to_be_computed=1
d_reqr:568   [KVC][L2] S3 BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=2156
d_reqr:714   [KVC][KVP] TERM req=cmpl-...-9f3122d1 dev=npu:0 逐层按块: layers=32 blocks=[1, 2, 4, 5, 6] region=520/520 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=8, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
d_reqr:715   [KVC][KVP] TERM L00 blk=1[满:128](128,8,128) blk=2[满:128](128,8,128) blk=4[满:128](128,8,128) blk=5[满:128](128,8,128) blk=6[未满:8](8,8,128) | K示(首块首token前3)=[0.5078, 0.9336, 0.9219] 统计[n=532480] mean=0.02146 std=1.457 min=-14.56 max=10.81 | V示(首块首token前3)=[0.01538, 0.0008049, 0.03345] 统计[n=532480] mean=0.0003453 std=0.03747 min=-0.3477 max=0.4746
d_reqr:820   [KVC][L5] ======== 释放 ========
d_reqr:823   [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(6, 0), (5, 0), (4, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 5 块 [6, 5, 4, 2, 1], append_n -> 队尾(LRU保护)
d_reqr:824   [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[6, 5, 4, 2, 1]), num_free_blocks=2161
```

（:60~:713 为补算步后 34 个 decode 步的分配循环，每步约 19 行同构——`num_new_tokens=1` 逐 token 推进 comp 486→520；:717~:819 为 L01~L31 逐层指纹打印，§4.2 摘 L00 对照；完整原文见 kvc_d_reqr.log）

流程讲解：

- **D 的 HIT 哈希（:10/:13，6445dcc64516/ae2c4445bc09）= §3.4 载入步 :38/:39 种的 D 链**——伏笔回收：命中块 [1,2] 本次**不传输**，mooncake 只拉 D miss 的块 4、5（30 MiB 增量，d:508 `took 1.29 ms`，对比 req_p 全量 282.94ms——会话已热 + 传输量减半）。
- **载入步 :25/:26 布局行 `|<comp>=0 |<new_comp>=256 |<ext_comp>=230(P传D) |<new>=0 |`：D 本地命中 256 + P 传 230 = 486 prompt 全账**——ext_comp 正是"P 传给 D 的 KV 抵扣"，与 §3.4 载入步的 ext_comp=324（全量）对照，同一字段两种取值直观展示**"传输量由 D 命中决定"**（../kvc_pd_prefix/docs/pd_prefix_cache_matrix.md §1.2）；:34/:35 备块 [4,5]（S3 新块为空，KV 由 mooncake DMA 写入），:39 落卡后第三满块种入 D 表（map size=3）。
- **补算步 :50/:51 布局行 `|<comp>=485 |<new>=1 |`**（256+230−1=485 落账）：末 prompt token 本地 forward 出首 token；:59 块未满无需新块。
- **decode 循环（:60~:713）**：每步 comp 递增（:67 首步 `|<comp>=486 |<new>=1|`），blk5 由 102 prompt 槽 +26 decode 槽写满 128 后 :568 跨界申请 blk6。
- **:714/:715 TERM blocks=[1, 2, 4, 5, 6] region=520/520**（486 prompt 槽 + 34 新 decode 输入槽 = 4×128 + blk6 前 8 槽）；终态 blk5[满:128] blk6[未满:8]——对照 P :50 的 blk5[未满:102]：26 个 decode 槽写满了它。:820/:823/:824 五块归零回收 append 队尾（LRU 保护、哈希映射保留——§1.2 判定框架表第 4 行"D 池保留命中块(LRU)"的日志原貌）。

### 4.2 块级指纹对账（32 层 × 4 共有块 × K/V = 256 对 Tx）——双侧 L00 原样

```
kvc_p_reqr.log:51  [KVC][KVP][FPB] TERM L00 块指纹: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/K.Xx=759d20483db8a502 blk4K.Tx=cd55014ec99844e4/K.Xx=cd55014ec99844e4 blk5K.Tx=ff59cb965b6b778f/K.Xx=31b081d62d392f88 | blk1V.Tx=5909248f73a62d39/V.Xx=5909248f73a62d39 blk2V.Tx=5db451ce3b0ae393/V.Xx=5db451ce3b0ae393 blk4V.Tx=25ac60aa285f19c4/V.Xx=25ac60aa285f19c4 blk5V.Tx=a54bf1bba8ee4d87/V.Xx=5f7deeb82178057d
kvc_d_reqr.log:716  [KVC][KVP][FPB] TERM L00 块指纹: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/K.Xx=759d20483db8a502 blk4K.Tx=cd55014ec99844e4/K.Xx=cd55014ec99844e4 blk5K.Tx=ff59cb965b6b778f/K.Xx=fe64baae2eeffc72 blk6K.Tx=-/K.Xx=1392dbef7f47a461 | blk1V.Tx=5909248f73a62d39/V.Xx=5909248f73a62d39 blk2V.Tx=5db451ce3b0ae393/V.Xx=5db451ce3b0ae393 blk4V.Tx=25ac60aa285f19c4/V.Xx=25ac60aa285f19c4 blk5V.Tx=a54bf1bba8ee4d87/V.Xx=4f1bd19f6cbf22b6 blk6V.Tx=-/V.Xx=407b9af2efde1926

verdict.txt:71 ===== kvc_p_reqr.log vs kvc_d_reqr.log =====
verdict.txt:73 p_tok: P=486 D=486  传输区 Tx = 双侧各取前 p_tok-1=485 tok
verdict.txt:74 Tx 指纹对账: 256/256 对全等
```

流程讲解（L00 为例，对账结论以 verdict 为准）：

- **blk1/blk2**（P 本地缓存驻留 vs D 传输落卡驻留）：Tx 与 Xx 逐字段全同——两条**独立缓存路径**的同源数据（源头都是 req_p 轮传输）逐位一致；注意 **blk1K.Tx=4f99980bf97ed995 与 req_p 轮 §3.6 完全相同**——同一块两次 TERM 打的同一哈希，互证指纹计算无状态泄漏。
- **blk4**（P prefill 新算 vs mooncake 新传）：Tx、Xx 全同——本次新传输无误差。
- **blk5**：Tx 同（blk5K.Tx=ff59cb965b6b778f 两侧逐字相同，Tx 只覆盖前 485 tok 的 101 槽纯传输区 ⇒ 传输无损）/ Xx 异——P.Xx=31b081d62d392f88 vs D.Xx=fe64baae2eeffc72（Xx 含 D 补算 1 槽+decode 26 槽）；blk5V.Xx 同理（P=5f7deeb82178057d vs D=4f1bd19f6cbf22b6）。
- **blk6**：D 独有 decode 新块，`Tx=-` 无传输区不参与 Tx 对账（K.Xx=1392dbef7f47a461 / V.Xx=407b9af2efde1926 仅 D 有）。
- 层指纹 K.prompt 全等 2/32（verdict.txt:75）——同 §3.6 机理。

| 区间 | P 侧来源 | D 侧来源 | 对账 | 含义 |
|---|---|---|---|---|
| blk1、blk2 Tx/Xx | **P 本地缓存** | **D 本地缓存**（req_p 传输落卡后缓存） | **全等** | 两条独立缓存路径的同源数据仍逐位一致 |
| blk4 Tx/Xx（满块） | P prefill 新算 | mooncake 传输 | **全等** | 本次新传输无误差 |
| blk5 Tx（前 101 槽） | P prefill 新算 | mooncake 传输 | **全等** | 本次新传输无误差 |
| blk5 Xx（传输 101+补算 1+decode 26） | P prefill | D 补算+decode | L01~L31 异 | 本地生成槽位，预期 |
| blk6（D 独有，Tx=- 无传输区） | — | D decode 循环 | 不比对 | D 独有 |
| 层指纹 K.prompt | | | 全等 2/32 | 同 §3.6 机理 |

**256/256 Tx 全等 → 混合来源（本地缓存+新传输）下传输依然无损**。

响应核对：resp_r.json 生成 35 token `"://www.zhihu.com/question/404201526\n1. 什么是前缀缓存？\n2. 前缀缓存的工作原"`（completion_tokens=35），与 kvc_d_reqr.log 中 35 组 decode 步循环对账吻合。

---

## 5. 裁决：全部差异归因清单

两轮请求的所有"不等"逐项归因，**无一落在传输区**：

| 差异点 | P 侧写入者 | D 侧写入者 | 性质 |
|---|---|---|---|
| 尾块最后 1 槽（第 p_tok 个 token） | 批量 prefill（324/486 tok 前向） | 单 token 补算前向 | 同一数学量、不同 kernel 路径 → bf16 ULP 级 bit 差；L00 偶然 bit 相同 |
| decode 新块/新槽 | （P 不做 decode） | D decode 循环写入 | D 独有，无比对对象 |

> **最终答复**：P 侧计算并交出的 KV（前 p_tok-1 个 token，共 448 对块级指纹），D 侧**逐位一致**——mooncake adxl 传输在这两个请求上**零字节差错**。"P/D 两侧 kvcache 是否相等"的严格表述是：**该相等的部分（传输区）完全相等；不相等的部分恰好都是 D 侧本地生成、从未经过传输的槽位。**

---

## 6. 附：两个边界观察

1. **哈希盐**：注意 req_r 前缀查找（§4.1）：P 查 bf3264861b0c、D 查 6445dcc64516——同一 token 序列在 P/D 的 BlockHash 不同（P 盐链≠D 盐链，请求变量 mix-in）。两侧没有任何哈希对齐操作（每实例仅需在本地连续命中即可）——这使得**指纹必须基于 kv 块内容（sha256）独立对账**、而**绝对不能基于 BlockHash 判等**（本文件即）。
2. **池尺寸**：本轮 P num_blocks=2161 / D=2162——两池块数由各实例独立 measure-available-memory 决定，D 侧因 mooncake completed-buffer 占用一个块位（33.79 vs 33.78 GiB）而恒定少 alloc 1 块；**传输量按请求实际 KV（48 MiB / 30 MiB 整块）计算，两侧池尺寸一致性非前提**（跨轮波动 2161↔2162，三轮对照见 §8）。

---

## 7. 复现与文件索引

```
# 容器内（kvc_pd 同步后）
bash scripts/run_all.sh            # 一键：打 01~09 → 起双实例 → 发双请求 → 收轨迹 → 杀服务 → 撤补丁(源码还原干净)
python3 scripts/compare_fp.py      # 自动对账 → log/verdict.txt（Tx/Xx/层指纹三级 + PASS/FAIL）
```

| 产物 | 内容 |
|---|---|
| log/verdict.txt | 自动裁决全文（本轮：Tx 448/448 全等 → PASS） |
| log/kvc_{p,d}_{startup,reqp,reqr}.log | 双侧 [KVC] 轨迹六段（含 [FPB]/[FP] 指纹） |
| log/{p,d}_llama.log | 双侧服务全量日志 |
| log/proxy.log | proxy(8000) 启动与健康检查 |
| log/curl_{p,r}_screen.txt + resp_{p,r}.json | 请求命令/屏显/响应体 |
| log/run_all_screen.log | 一键脚本全程录屏 |
| patch/09_pd_kv_fingerprint.patch | 指纹探针（apply/revert 见 patch/ 下脚本） |

> log/ 只保留最新一轮产物，重跑实验直接原地覆盖更新。

---

## 8. 附：跨轮复现对照（三轮，旧轮产物已释出）

本实验已完整跑过三轮（历史轮容器回收/产物释出，log/ 只留最新一轮；本文全文以最新轮为准）：

| 对比项 | R1: 09-29 旧容器首测 | R2: 09-30 上午复测 | R3: 09-30 08:01 本轮（04 补丁增强版） |
|---|---|---|---|
| patch 04 | 168 行基础版 | 同 R1 | **170 行五段布局行增强版**（allocate_slots 新增 comp/new_comp/ext_comp/new/lookahead 布局行） |
| verdict.txt | 448/448 Tx 全等 | **MD5 全同（b4cf086b…）** | **MD5 三轮全同（b4cf086b…）** |
| 生成文本 | "为了" / 35 tok 知乎文本 | 同左 | 同左（逐 token 全同） |
| ext_comp 实证 | 无此打印（靠 num_new_computed_tokens 间接推断） | 无 | **直读：载入步 324（全量）/ 230（命中抵扣后）**（kvc_d_reqp:19 / kvc_d_reqr:26） |
| transfer 耗时 | 834.86ms / 1.36ms | 289.96ms / 1.33ms | 282.94ms / 1.29ms |
| P/D 池块数 | 2161 / 2162 | 2162 / 2162 | 2161 / 2162 |
| 物理卡 | 旧容器（回收） | gggtest 首次 | gggtest 二次（同卡） |

判定意义：

1. **MD5 三轮一致 + FPB 指纹值逐轮全同**（blk1K.Tx=4f99980bf97ed995 等跨轮不变）——内容指纹由 KV 数据决定而非环境，seed=1024 + enforce_eager 下位级确定，跨容器/跨响应已验证（三轮 request_id 全不同，指纹值照常全同，反证与请求 ID 无关）。
2. **三轮 BlockHash 链各不相同**（R3: P=bf32648… vs D=6445dcc…）——哈希盐含 request_id 属预期；轮内 P/D 同 ID 同盐对账才是判据。
3. **池块数波动（2161↔2162）不影响结论**——传输按请求实际 KV 整块（48 / 30 MiB）计算。
4. **R3 新增的第 4 证据维度**：布局行把"P 传 D"从间接推断（对比两步字段变化）升级为**直读账面**——ext_comp 字段显式打印，再见 §3.4/§4.1 讲解。

