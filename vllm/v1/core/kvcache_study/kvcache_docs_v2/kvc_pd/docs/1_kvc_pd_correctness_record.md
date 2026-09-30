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

### 1.1 为什么统计值不够，必须用"内容指纹"

旧一轮（09-27，旧版 08 补丁）用**统计值**（n/mean/std/min/max）对比双侧：完全一致。但这**不能证明逐位相等**——统计值是"多位数值的平均"，单 bit 翻转几乎不影响 4 位有效数字。旧轮其实已有可疑信号：req_p 的 V 张量 **zeros 计数 P=60 vs D=59，差 1**——至少一个元素两侧不同，统计值却全同。

要严格检查"传输是否有误"，唯一可靠的办法是对**同一 token 区间的原始位模式**取哈希：两侧 sha256 相同 ⇔ 每一位都相同。09 补丁在 P/D 两实例跑同一份代码（`_kvc_kv_dump()`），对同区间 KV 原始字节（bf16 按 uint8 位视角）各取 sha256 前 16 hex。

### 1.2 三个必须预先厘清的比对前提

正确性检查不是"双侧全部字节拉平比对"——PD 架构决定了两侧的 KV **天然不该全等**。先明确边界，判定才有意义：

1. **D 只接收前 p_tok-1 个 token 的 KV**。D 侧载入步实测（kvc_d_reqp.log:26）：`num_new_tokens=0` 分配 3 块后，mooncake 异步写入 P 侧 **323 个 token**（=p_tok-1，p_tok=324）的 KV；随后补算步 `num_new_tokens=1, num_computed_tokens=323`——**最后一个 prompt token 由 D 本地前向补算**（PD bootstrap 惯例，D 以"正在计算"的状态衔接 prefill→decode）。
   - **为什么免传不是优化、传了也作废**：D 首步要产出第 1 个生成 token，必须把 token 323 喂进 forward 拿 **logits**（KV cache 里只有注意力的 K/V，没有输出头结果，"零 forward 启动 decode"不存在）；而 QKV projection 是一个融合 GEMM——Q(323) 必算，**K/V(323) 是同一刀的免费副产品**；标准 attention 内核语义是 **query 位的 K/V 由 projection 现算并写入槽位**（非 query 位才从 cache 读）——不存在"读现成 KV(323)、只算 Q"的默认路径，拆开融合 GEMM 写专用内核得不偿失，vLLM 而是**完全**复用"差 1 tok 没算完的普通请求"路径（`num_computed=323 + num_new=1`→一条普通 1-token prefill，零特判）。D 覆写值与原值 bf16 等价（批量 vs 单 tok kernel tiling 的 ~ULP 差，§5.2 zeros 60→59 即此）。
   - **物理/逻辑口径区分**：物理上 DMA 按整块传（3 块 48.0 MiB，../kvc_pd_prefix/docs/pd_prefix_cache_matrix.md §4.1 实测），token 323 的槽位随尾块一起到达、随后被 D 覆写；"323"只是调度器逻辑账（num_computed_tokens），不矛盾。同一惯例三处同现：单机全命中也重算最后 1 tok / 本行 D 侧补算 / mamba 模型 P 侧 `_truncate_request_for_prefill` 显式只算到 h(N-1)（mc:1760-1784）。
2. **D 的 decode 新写块 P 侧不存在**（P 不做 decode），无比对对象。
3. **两侧 BlockHash 不同**（实例哈希盐独立，req_p: P=6232d7d69383 vs D=16c6db3ef24b）——prefix cache 是实例本地机制，跨侧一致性只能靠内容指纹，不能靠哈希对账。

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
| P | npu:0 | 8100 | producer rank0 / listening tcp://172.16.210.194:20001 | 172.16.210.194:20262 | 33.78 GiB / 2162 块 |
| D | npu:1 | 8200 | consumer rank1 / 20002 | 172.16.210.194:20344 | 33.79 GiB / 2162 块 |
| proxy | — | 8000 | 同 request body 双发 P/D | — | — |

**时间线（run_all_screen.log，全程 2 分 31 秒）**：

| 时刻 | 事件 |
|---|---|
| 04:10:59 | run_all 开始：打补丁 kvc 01~08 + 09（8 文件 168 行 [KVC] + 指纹，dry-run 全过） |
| 04:11:56~58 | P 建池（33.78 GiB / 2162 块）→ adxl 注册 → 04:12:09 就绪（等待 70s，p_llama.log:291 startup complete） |
| 04:12:41 | D 建池（33.79 GiB / 2162 块, adxl 172.16.210.194:20344）→ 04:12:4x 就绪（等待 40s，d_llama.log:289） |
| 04:12:46 | proxy 就绪（1 prefill + 1 decode client） |
| 04:12:53~04:13:11+ | req_p、req_r 相继完成（双侧 KVP 指纹打印） |
| 04:13:30 | 杀服务 → revert 09→08→01~07，源码 [KVC] 全归零，零进程残留 |

启动期 [CFG]/[L1]/[L2~L5] 打印与单机实验同构，不再展开；本档聚焦**运行期传输对账**。

---

## 3. 检查一：req_p（324 tok，max_tokens=1 —— 纯传输探针）

**设计意图**：max_tokens=1 让 D 侧几乎不做 decode——323 tok 的 KV 全部来自传输，把"传输正确性"从其他变量里单独剥离出来。

### 3.1 P 侧：哑请求一步走完（kvc_p_reqp.log 原样节选，04:12:53~54）

```
[ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=6232d7d69383
[ENQ] 入队 hash_block_tokens: parent=6232d7d69383, tokens=128 -> BlockHash=91124d5b7fc5
[ENQ] Request(...-876d38d7) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['6232d7d69383', '91124d5b7fc5']
[L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=(hash=6232d7d69383, group_id=0) -> MISS
[L3] 前缀查找   第 1 块 MISS: BlockHash=6232d7d69383 -> break
[L5] 前缀查找 get_computed_blocks 返回: KVCacheBlocks(blocks=([],)), num_computed_tokens=0
[L5] 分配 allocate_slots 进入: num_new_tokens=324, num_new_computed_tokens=0, request.num_computed_tokens=0, num_tokens=324
[L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=2158
[L2] S4 BlockHashToBlockMap.insert: key=(hash=6232d7d69383, group_id=0) <- KVCacheBlock(block_id=1), map size=1
[L2] S4 BlockHashToBlockMap.insert: key=(hash=91124d5b7fc5, group_id=0) <- KVCacheBlock(block_id=2), map size=2
[KVP] TERM req=...-876d38d7 dev=npu:0 逐层按块: layers=32 blocks=[1, 2, 3] region=324/324 tok
```

逐行讲解：① `max_tokens=1` 是 proxy 改写的哑请求（P 的任务只是 prefill+收官交块）；② 前缀查找对首块哈希即 MISS——P 冷启动，324 tok 全量 prefill，块 [1,2,3] 全新分配；③ **两满块的哈希种入 P 本地表**（注意盐：6232d…，这是 P 进程的 NONE_HASH 链）；④ TERM 是"请求结束、KV 即将释放"前的物理快照——P 随后 `request_finished` 把块 [1,2,3] 上报给 D 并延迟释放（p_llama.log:431 `Delaying free of 3 blocks`）。

### 3.2 D 侧：载入步 + 补算步（kvc_d_reqp.log 原样节选，04:12:57~58）

```
[ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=16c6db3ef24b
[ENQ] 入队 hash_block_tokens: parent=16c6db3ef24b, tokens=128 -> BlockHash=17d3af4991da
[L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=(hash=16c6db3ef24b, group_id=0) -> MISS
[L5] 前缀查找 get_computed_blocks 返回: KVCacheBlocks(blocks=([],)), num_computed_tokens=0
──── 载入步 ────
[L5] 分配 allocate_slots 进入: num_new_tokens=0, num_new_computed_tokens=0, request.num_computed_tokens=0, num_tokens=324
[L4] S1 get_num_blocks_to_allocate: req=... -> 需分配 3 块(含touch需腾挪的块)
[L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=2158
[L5] S3 allocate_new_blocks: num_tokens_need_slot=324 -> 新块 []          ← S3 新块为空!
[L5] ======== 分配完成 ========
[L5] 提交 cache_blocks: num_computed_tokens=324 (async 步末输出路径)
[L2] S4 BlockHashToBlockMap.insert: (hash=16c6db3ef24b) <- block_id=1    ← 传输落卡后种入 D 表
[L2] S4 BlockHashToBlockMap.insert: (hash=17d3af4991da) <- block_id=2
──── 补算步 ────
[L5] 分配 allocate_slots 进入: num_new_tokens=1, num_new_computed_tokens=0, request.num_computed_tokens=323, num_tokens=324
[L5] S3 块未满, 无需分配新块 (num_new_tokens=1)
[L5] 分配 返回: ... 当前完整 block_table=([1, 2, 3],)
[KVP] TERM req=...-800e2db8 dev=npu:0: layers=32 blocks=[1, 2, 3] region=324/324 tok
[KVP] TERM L00 blk=1[满:128] blk=2[满:128] blk=3[未满:68] | 统计[n=331776] ...
```

逐段讲解：

- **入队的哈希 16c6d… ≠ P 的 6232d…**——同 token 序列、两进程两个盐、两条链（§1.2 前提 3 的原始出处）。
- **载入步 `num_new_tokens=0`**：D 本请求不算任何新 token；3 块接收目标 [1,2,3] 全新分配，但 S3"新块为空"——块本身不由 forward 写，**KV 由 mooncake 异步从 P 卡 DMA 写入**（d_llama.log:324 实证：`KV cache transfer for request ...-876d38d7 took 289.96 ms`，48.0 MiB 整块含 adxl 会话建立）。
- **载入步的 S4 玄机**：满块 [1,2] 立刻种入 **D 自己的**哈希表（盐 16c6d… 链）——这就是 req_r 时 D 前缀命中的种子（§4 将看到它们 HIT）。
- **补算步 `num_new_tokens=1, num_computed_tokens=323`**：调度器把请求摆成"差 1 个 token 没算完的普通请求"；D 本地 forward 第 324 个 token（index 323）写入 blk3 第 68 槽，logits 采样出首 token——`resp_p.json` 的"为了"。blk3 覆盖 68 槽 = 载入 67 + 补算 1，与 TERM 行 `[未满:68]` 互证。
- TERM region=324/324：P/D 双侧 TERM 头完全同构（blocks/层数/region 逐字段一致），为指纹对账给出同构骨架。

### 3.3 块级指纹对账（32 层 × 3 块 × K/V = 192 对 Tx）——双侧原样

```
P L00 [FPB]: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/K.Xx=759d20483db8a502 blk3K.Tx=2a2c157641418bfa/K.Xx=85d795e93bd122e1 | blk1V.Tx=5909248f73a62d39/V.Xx=5909248f73a62d39 blk2V.Tx=5db451ce3b0ae393/V.Xx=5db451ce3b0ae393 blk3V.Tx=04fd71d2dcd7e433/V.Xx=7635560bb2914987
D L00 [FPB]: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/K.Xx=759d20483db8a502 blk3K.Tx=2a2c157641418bfa/K.Xx=85d795e93bd122e1 | ...（与 P 逐字段全同，含 Xx）
P L01 [FPB]: blk1K.Tx=9582307cb75a1af8/K.Xx=9582307cb75a1af8 blk2K.Tx=cf340e4c1e28b0be/K.Xx=cf340e4c1e28b0be blk3K.Tx=95658bda922a290b/K.Xx=6cb10022565c349a
D L01 [FPB]: blk1K.Tx=9582307cb75a1af8(同) blk2K.Tx=cf340e4c1e28b0be(同)；blk3K.Tx=95658bda922a290b(同!) / K.Xx=36c4fb7d3ebe8950(异!)
```

读法讲解（L01 为例）：**blk1/blk2 的 Tx 与 Xx 全等**（整块都来自传输）；**blk3 的 Tx 同、Xx 异**——Tx 只哈希前 323 tok 覆盖的 67 槽（载入区），两侧相等 ⇒ 传输无损；Xx 哈希全部 68 槽，第 68 槽是 D 补算重写的 ULP 级差值 ⇒ Xx 必然不同。`Tx=95658bda922a290b` 两侧逐字相同就是"D 落卡字节 = P 原始字节"的 sha256 级证据。

| 区间 | 槽位数 | P 侧来源 | D 侧来源 | 对账 |
|---|---|---|---|---|
| blk1 Tx（tok 0~127） | 128 | P prefill | mooncake 传输 | **32 层全等** |
| blk2 Tx（tok 128~255） | 128 | P prefill | mooncake 传输 | **32 层全等** |
| blk3 Tx（tok 256~322） | 67 | P prefill | mooncake 传输 | **32 层全等** |
| blk3 Xx 第 68 槽（index 323） | 1 | P 批量 prefill | D 单 token 补算 | L00 恰好同，L01~L31 异（ULP 级） |

**192/192 Tx 全等 → 本轮传输无误差**。两个佐证细节：

- 层指纹 `K.prompt` 全等层数仅 1/32（恰好 L00）——层指纹的 prompt 区含补算槽，每层 1 个 token 的 bit 差就毁掉整层哈希；而**统计行 32/32 全同**——ULP 级差对统计不可见。两相对照正是"必须用指纹"的实证。
- 响应：resp_p.json 生成"为了"（completion_tokens=1），与单机实验同 prompt 输出一致——D 侧基于"传输 KV+补算槽"采样，语义正确。

---

## 4. 检查二：req_r（486 tok，max_tokens=35 —— 前缀命中 + decode 递进探针）

**设计意图**：在"纯传输"成立后叠加两个新变量——（a）双侧**本地前缀缓存**命中参与供数（KV 不再全部直接来自本次传输）；（b）D 侧 35 步 decode 持续写入新块。检验混合来源下传输数据是否仍然一致。

### 4.1 双侧前缀查找与分工（原样日志节选，04:13:11~12）

**P 侧（kvc_p_reqr.log）——命中自己种的块，只算增量：**

```
[ENQ] 入队 hash_block_tokens: parent=NONE_HASH → 6232d7d69383; parent=6232d7d69383 → 91124d5b7fc5; parent=91124d5b7fc5 → 77db31e3301e
[ENQ] Request(...-ab437267) 入队: num_prompt_tokens=486, max_tokens=1, 满块链式哈希 BlockHash × 3: ['6232d7d69383', '91124d5b7fc5', '77db31e3301e']
[L2] 前缀查找 get_one_block: key=(hash=6232d7d69383, group_id=0) -> HIT KVCacheBlock(block_id=1)
[L2] 前缀查找 get_one_block: key=(hash=91124d5b7fc5, group_id=0) -> HIT KVCacheBlock(block_id=2)
[L3] 前缀查找   第 3 块 MISS: BlockHash=77db31e3301e -> break
[L4] 前缀查找 返回: hit_blocks=[[1, 2]], hit_length=256
[L5] S2 allocate_new_computed_blocks: new_computed_blocks=[[1, 2]]
[L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
[L3] S3 SingleType...: 需 4 块 - 已有 2 = 新分配 2 块 [4, 5], 持有 req_blocks=[1, 2, 4, 5]
[L5] S4 cache_blocks: ... num_computed_tokens=486
```

讲解：P 入队把 486 tok 算成 3 条满块哈希链（首两条与 req_p 完全相同——同 token 同盐必然同链）→ 命中 req_p 种下的块 [1,2]（touch +1 保活）→ **只新 prefill 230 tok**（块 4、5），TERM blocks=[1,2,4,5] region=486/486（2 复用+2 新算）。P 的 max_tokens 仍是 proxy 改写的 1（哑请求口径不变）。

**D 侧（kvc_d_reqr.log）——命中自己接收时种的块，只收增量：**

```
[ENQ] 入队 hash_block_tokens: parent=NONE_HASH → 16c6db3ef24b; → 17d3af4991da; → b7fcbb3406a7   ← D 盐链(≠P 链)
[ENQ] Request(...-87426429) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 × 3: ['16c6db3ef24b', '17d3af4991da', 'b7fcbb3406a7']
[L2] 前缀查找 get_one_block: key=(hash=16c6db3ef24b, group_id=0) -> HIT KVCacheBlock(block_id=1)    ← §3.2 载入步 S4 种下的伏笔回收
[L2] 前缀查找 get_one_block: key=(hash=17d3af4991da, group_id=0) -> HIT KVCacheBlock(block_id=2)
[L3] 前缀查找   第 3 块 MISS: BlockHash=b7fcbb3406a7 -> break
[L4] 前缀查找 返回: hit_blocks=[[1, 2]], hit_length=256
──── 载入步(增量接收) ────
[L5] 分配 allocate_slots 进入: num_new_tokens=0, num_new_computed_tokens=256, request.num_computed_tokens=0, num_tokens=486
[L2] S2 touch blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
[L2] S3 BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=2157
[d_llama.log:500] KV cache transfer for request ...-ab437267 took 1.33 ms. remote_session_id 172.16.210.194:15284
──── 补算步 → decode 35 步 → 跨界新块 ────
[L5] 分配 进入: num_new_tokens=1, num_new_computed_tokens=0, request.num_computed_tokens=485, num_tokens=486   ← 尾 tok 本地算
[L5] 分配 进入: num_new_tokens=1, request.num_computed_tokens=486, num_tokens=486(每个 decode 步重复此形态, 共 35 步)
[L2] S3 BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=2156   ← decode 写满 blk5 后跨界申请
[KVP] TERM req=...-87426429: layers=32 blocks=[1, 2, 4, 5, 6] region=520/520 tok
[L2] 释放 BlockPool.free_blocks: blocks=[(6, 0), (5, 0), (4, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 5 块, append_n -> 队尾(LRU保护)
```

讲解：

- **D 的 HIT 哈希 = req_p 载入步 S4 种的 D 链**（16c6d…/17d3a…）——命中块 [1,2] 本次**不传输**：mooncake 只拉 D miss 的块 4、5（30 MiB 增量，`took 1.33 ms`，对比 req_p 全量 289.96ms——会话已热+传输量减半）。
- **载入步 `num_new_tokens=0, num_new_computed_tokens=256`**：D 侧声明的 external = 486−256（命中率就是传输抵扣）——对比 reqp 载入步的 `num_new_computed_tokens=0`（external=324 全量），**同一字段的两种取值直观展示"传输量由 D 命中决定"**（../kvc_pd_prefix/docs/pd_prefix_cache_matrix.md §1.2）。
- **region=520/520 的账**：486 prompt 槽 + 34 个新 decode 输入槽（35 步 decode 里第 1 步是补算重写槽 485、其余 34 步各写 1 新槽）= 520 = 4×128 + blk6 前 8 槽；blk5 由 102 prompt 槽写到满 128（+26 decode），跨界进 blk6。
- **释放 LRU 尾插**：块 [6,5,4,2,1] 归零后 append 到空闲队列**队尾**——块 1、2 释放后仍留在 BlockHashToBlockMap 哈希映射中、且排在最"新"端，**下次同前缀请求仍会命中**（这就是 docs/1 §1.3 框架表第 4 行"D 池保留命中块(LRU)"的日志原貌）。

### 4.2 块级指纹对账（32 层 × 4 共有块 × K/V = 256 对 Tx）——双侧 L00 原样

```
P L00 [FPB] TERM L00 块指纹: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/K.Xx=759d20483db8a502 blk4K.Tx=cd55014ec99844e4/K.Xx=cd55014ec99844e4 blk5K.Tx=ff59cb965b6b778f/K.Xx=31b081d62d392f88 | blk1V.Tx=5909248f73a62d39/...(同构) blk4V.Tx=25ac60aa285f19c4/... blk5V.Tx=a54bf1bba8ee4d87/...
D L00 [FPB] TERM L00 块指纹: blk1K.Tx=4f99980bf97ed995(同) blk2K.Tx=759d20483db8a502(同) blk4K.Tx=cd55014ec99844e4(同) blk5K.Tx=ff59cb965b6b778f(同!) / blk5K.Xx=fe64baae2eeffc72(异:101 传输槽+27 本地新写槽) | blk6K.Tx=-/K.Xx=1392dbef7f47a461(D 独有,无传输区)

===== kvc_p_reqr.log vs kvc_d_reqr.log =====            (verdict.txt 原样)
p_tok: P=486 D=486  传输区 Tx = 双侧各取前 p_tok-1=485 tok
Tx 指纹对账: 256/256 对全等
```

读法讲解（L00 为例）：**blk1/blk2**（P 缓存 vs D 缓存）与 **blk4**（P 新算 vs mooncake 新传）**Tx、Xx 逐字段全同**——两条独立缓存路径与新传输路径的 KV 都零位差；**blk5 Tx 同 / Xx 异**（Tx 只覆盖前 485 tok 的 101 槽纯传输区，Xx 含 D 补算 1 槽+decode 26 槽）；**blk6 `Tx=-`** decode 新块无传输区不参与 Tx 对账。注意 **blk1 的 Tx=4f99980bf97ed995 与 req_p 轮完全相同**——两轮指纹自洽（同一块两次 TERM 打的同一哈希），互证指纹计算无状态泄漏。

| 区间 | P 侧来源 | D 侧来源 | 对账 | 含义 |
|---|---|---|---|---|
| blk1、blk2 Tx/Xx | **P 本地缓存** | **D 本地缓存**（req_p 传输落卡后缓存） | **全等** | 两条独立缓存路径的同源数据仍逐位一致（源头都是 req_p 的传输） |
| blk4 Tx/Xx（满块） | P prefill 新算 | mooncake 传输 | **全等** | 本次新传输无误差 |
| blk5 Tx（前 101 槽） | P prefill 新算 | mooncake 传输 | **全等** | 本次新传输无误差 |
| blk5 Xx（传输 101+补算 1+decode 26） | P prefill | D 补算+decode | L01~L31 异 | 本地生成槽位，预期 |
| blk6（D 独有，Tx=- 无传输区） | — | D decode 循环 | 不比对 | D 独有 |
| 层指纹 K.prompt | | | 全等 2/32 | 同 §3.3 机理 |

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

1. **哈希盐**：同一 token 序列在 P/D 的 BlockHash 不同（req_p: 6232d7d69383 vs 16c6db3ef24b）——prefix cache 是实例本地机制，跨侧一致性不依赖哈希对齐（这正是用内容指纹而非哈希对账的原因）。
2. **池尺寸**：本轮 P 2162 / D 2162（09-29 首测轮曾 P 2161 / D 2162，D 的 mooncake 缓冲张量大 256KiB）——池块数随建池时显存余量碎片级波动——传输量按请求实际 KV 计算，不依赖两侧池尺寸一致。

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

## 8. 附：跨轮复现对照（09-29 首测轮 vs 本轮，旧轮产物已释出）

本实验在两个独立容器各完整跑过一次（09-29 首测轮容器已回收、产物已清理；本文全文以 09-30 本轮为准，此处仅保留两轮对照结论）。本轮为 itask 4 卡容器 gggtest（image v0.23.0-a3-openeuler-20260818163431，openEuler 24.03 LTS-SP3，workdir `/a3_inference/itask/workdir/wsl02075301/kvc_pd`，物理卡与首测轮不同）：`setsid nohup bash scripts/run_all.sh` 全程 2 分 31 秒（补丁 dry-run 全过 → P 就绪 70s → D 就绪 40s → proxy ok → 双请求 → 源码干净归零）→ `python3 scripts/compare_fp.py`。

**结果：逐字节级复现 PASS**

| 对比项 | 首测轮（09-29 旧 pod，gch02599191） | 本轮（09-30 gggtest，wsl02075301） |
|---|---|---|
| verdict.txt | 448/448 Tx 全等，Xx 差异 62+64 条全落尾块/decode | **MD5 全同（b4cf086bdde4372431ef73a7a25840e2）** |
| req_p 生成 token | "为了" | "为了"（prompt_tokens=324, completion=1） |
| req_r 生成 35 tok | `://www.zhihu.com/question/404201526\n1. 什么是前缀缓存？\n2. 前缀缓存的工作原` | **逐 token 全同**（prompt_tokens=486, completion=35） |
| P 侧 KV 池 | 2161 块 | 2162 块（唯一实质差异，显存碎片级波动） |
| 全流程时长 | 2 分 12 秒 | 2 分 31 秒 |

判定意义：

1. **MD5 级一致远强于"结论同为 PASS"**——裁决文本含全部 126 条 Xx 差异明细（每条的层号/块号/K·V/双侧哈希值全量打印）逐字节相同,意味着两侧 ULP 级差的位置与数值在换容器、换物理卡后**全部复现**（seed=1024 + enforce_eager 的确定性执行链）。
2. **BlockHash 链两轮不同属预期**（req_p 首块 09-29 轮 78e45fa0b2cc vs 09-30 轮 6232d7d69383）——哈希盐含 request_id，每轮新请求 ID 不同；轮内 P/D 同 ID 同盐对账才是判据，跨轮比对哈希值无意义。
3. **池尺寸是环境性取值**（首测轮 P 2161 / D 2162，本轮双侧同为 2162；当时对两轮 kvc_p_startup 逐行 diff，去时间戳/PID 后实质差异仅 num_blocks 一处）——不影响按请求实际 KV 的传输量与指纹，verdict 不受影响。
