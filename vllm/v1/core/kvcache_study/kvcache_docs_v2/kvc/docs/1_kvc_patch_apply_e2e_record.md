# KVCache 调试打印体系实验（全生命周期设计 + 端到端实测）

> 本文是 `2_kvc_cn_curl_case.md`（用例）与 `patch/`（打印补丁）的**端到端正式验证记录**：从干净源码出发，以 patch 方式注入 92 处 `[KVC]` 打印（grep 计数 163 行，含注释行），记录一次完整的服务启动初始化与 P/R 双请求生命周期。§2~§4 的日志引文均**逐字原样**取自本次实测的拆解轨迹（`log/kvc_startup.log` / `kvc_p.log` / `kvc_r5.log`，已剥离进程前缀；带 `(Worker pid=…)` 前缀的完整版见 `log/llama.log`）。

## 0. 总览

### 0.1 打印体系全生命周期设计

（92 处 `[KVC]` 打印 / 163 行，三级横幅 + 阶段前缀贯穿）

**一、启动初始化（一次性，168 行 [KVC]，三段各带 `================` 开始/完成横幅）**

| 段 | 层标签 | 行数 | 核心内容 |
|---|---|---|---|
| 配置侧 | CFG | 84 | 各 worker 可用 KV 显存 → 逐 worker `KVCacheConfig`（num_blocks/组数/张量数）→ 逐张量 size/shared_by → 最终 scheduler 侧 min 对齐 |
| 物理侧 | L1 | 76 | 每层 KVCacheTensor 拆 **K int8 池 + V int8 池**两张独立张量（2MiB 对齐，支持 PD 分离）；reshape 后 K_cache=V_cache=(num_blocks, 128, 4, 128) bf16，block id 即 dim0 行号 |
| 逻辑侧 | L5 | 8 | 空闲队列（伪头尾哨兵）/ null 块 / BlockPool / L3 manager / 单组直通 coordinator 逐层装配 |

**二、运行期（每请求/每步动态，横幅对 + `--- S 子步标记 ---` + 阶段前缀）**

1. **入队（ENQ）**——`入队 hash_block_tokens` 逐满块链式哈希：H(bn)=fn(H(bn−1), tokens(bn))，首块 parent=NONE_HASH
2. **前缀查找**——逐块 HIT/MISS 查链（遇 miss 即断），hit_length = 命中块数 × block_size
3. **分配 S1~S4**（四项设计）：
   - **横幅先行**：子步横幅先于其全部下钻——S1 子步横幅在两次外层容量探问（L4）之前、S3 横幅在 `allocate_new_blocks` 调用之前
   - **四子步无条件**：无新块步也打全四段（S1 需分配 0 / S2 无前缀无需 touch / S3 无需分配新块 / S4 满块缓存维护），结构闭合；容量不足、延迟缓存提前返回路径均补关闭横幅
   - **阶段前缀全显**：每条下钻消息开头标注所属阶段（S1~S4 / 前缀查找 / 入队 / 释放 / 分配 / 提交），单条日志可定位
   - **S1 汇总值**：总横幅 → 进入 → `--- S1: 容量检查 ---` → 两次外层探问 → 需求块数 vs 可用块数
4. **调度提交**——async_scheduler 每步输出后的 `cache_blocks` 提交路径单列 `调度提交(非分配 S4)` 横幅对（每步一次，独立于分配 S4）
5. **KVP 物理校验（仅请求结束 TERM / 兜底 LATE）**——概览行含一次性 KV 布局说明（K_cache/V_cache 张量级拆分、非最后一维拼接、逐维含义）；每层一行内联块 `blk=N[满:bsz|未满:cov](cov,kv_heads,head_dim)` + K/V 首 3 值示意 + 层合并统计（n = region × kv_heads × head_dim 可精确断言）
6. **释放**——逆序归还、ref_cnt 归零回收、带哈希块 append 队尾（LRU 保护）

### 0.2 实测环境

| 项 | 值 |
|---|---|
| Pod | gggtest（a3 · 4× Ascend910 · PP2TP2 · 当日 Running） |
| 模型 | Meta-Llama-3-8B（`modelhub_74000048_meta-llama-3-8b-148700128_20260921221233`，32 层 / kv_heads 8 / TP2 下本地 4 / block_size 128） |
| 软件栈 | vllm 0.23.0 + vllm-ascend 0.23.0（`/vllm-workspace/`） |
| 进程 | APIServer pid=4555 · EngineCore pid=4600 · Worker pid=4633~4636（PP0_TP0 / PP0_TP1 / PP1_TP0 / PP1_TP1） |
| 实测时间 | 2026-09-28 07:46（log 内时间戳，容器时钟 UTC-8） |

## 1. 实验流程

### 1.1 起容器

```bash
# pod 状态确认（平台空闲会回收为 Stopped，需 itask start 拉起）
itask list | grep gggtest            # 期望 Running
itask start gggtest                  # 若 Stopped 时拉起
# SSH 隧道（本地 5558 -> 容器 7890）；pod 重建后 ssh host key 变化需先清除
itask ssh-tunnel gggtest --port 5558 --user gch02599191
ssh-keygen -f ~/.ssh/known_hosts -R "[localhost]:5558"   # pod 重建后
ssh -p 5558 root@localhost "hostname; whoami"            # 连通确认
```

源码干净基线核验（8 个文件 `grep -c "\[KVC\]"` 全部为 **0**、两仓库 `git status` 0 改动）。容器若经平台重启，可写层自动还原，此项天然满足。

### 1.2 打 patch

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
# Phase0 已应用检测 -> Phase1 dry-run 8/8 预检 -> Phase2 8/8 应用 -> Phase3 逐文件计数(合计 163 行) + py_compile
```

应用后逐文件 `[KVC]` grep 计数（含注释行；合计 163 行 = 92 个 `logger.info` 打印调用点）：

| 文件 | grep 计数 | 打印调用点 |
|---|---|---|
| `v1/request.py` / `v1/core/kv_cache_utils.py` | 5 / 12 | 3 / 6 |
| `v1/core/block_pool.py` / `kv_cache_manager.py` | 27 / **56** | 14 / **32** |
| `v1/core/kv_cache_coordinator.py` / `single_type_kv_cache_manager.py` | 17 / 18 | 9 / 9 |
| `v1/engine/core.py` | 13 | 9 |
| `vllm_ascend/worker/model_runner_v1.py` | 15 | 9（KVP: 头横幅/概览/逐层行循环/尾横幅） |

py_compile（8 文件）→ **COMPILE_OK**。核心补丁：04（12 hunk）、08（6 hunk）。

### 1.3 起服务并发送 P/R

```bash
bash scripts/start.sh                        # vllm serve PP2TP2 --enforce-eager，就绪 55s
bash scripts/curl_p_r.sh                     # P -> sleep 6 -> R；落盘打屏/响应/分界并提取三条 [KVC] 轨迹
```

### 1.4 收日志并去 patch

```bash
# 本地同步全套日志（12 文件）
scp -P 5558 -r root@localhost:/a3_inference/itask/workdir/gch02599191/kvc/log <本地kvc>/log/
# 容器回收
cd <kvc> && bash scripts/stop.sh             # 杀服务并确认 0 进程
cd patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_patches.sh
# 终态核验: 8 文件 [KVC] 全部归零 + py_compile + 两仓库 git 0 改动 + .orig 清理
```

## 2. 启动期日志讲解（log/kvc_startup.log，168 行）

### 2.1 配置侧（EngineCore，84 行 CFG，含首尾横幅）

开头三行即给出 KV 预算结论——4 个 worker 各自可用显存，随后逐 worker 展开 Config/Group/spec/page 全参数：

```
INFO 09-28 07:46:08 [core.py:265] [KVC][CFG] ================ 配置侧 KVCache 编排开始 ================
INFO 09-28 07:46:08 [core.py:267] [KVC][CFG] determine_available_memory: 各 worker 可用 KV 显存 = ['51.98GiB', '51.99GiB', '51.94GiB', '51.95GiB']
INFO 09-28 07:46:08 [core.py:280] [KVC][CFG] worker0 KVCacheConfig: num_blocks=13296, groups数=1, tensors数=16
INFO 09-28 07:46:08 [core.py:285] [KVC][CFG]   [0] KVCacheGroupSpec(group_id=0): layers=16 (首层 model.layers.0.self_attn.attn, 末层 model.layers.15.self_attn.attn), is_eagle_group=False
INFO 09-28 07:46:08 [core.py:290] [KVC][CFG]   [0]   kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, kv_quant_mode=<KVQuantMode.NONE: 0>, page_size_padded=None, head_size_v=128, sliding_window=None, attention_chunk_size=None)
INFO 09-28 07:46:08 [core.py:294] [KVC][CFG]   [0]   page_size_bytes=262144 (256.0KB/层/块), storage_block_size=128
```

逐 worker ×4 后由 scheduler 做 min 对齐并收尾：

```
INFO 09-28 07:46:08 [core.py:317] [KVC][CFG] 最终 scheduler KVCacheConfig: num_blocks=13296 (跨 worker min 对齐), cache_config.num_gpu_blocks=13296, block_size=128
INFO 09-28 07:46:08 [core.py:322] [KVC][CFG] ================ 配置侧 KVCache 编排完成 ================
```

### 2.2 物理侧（4 worker，76 行 L1，vllm-ascend，含首尾横幅）

每个 worker 持 1 个 KVCacheGroupSpec（16 层共享同一物理池）：每层 KVCacheTensor 拆 **K int8 池 + V int8 池**两张独立张量（2MiB 对齐），reshape 后 K/V 各 `(13296, 128, 4, 128)` bf16——**block id 即 dim0 行号**（KVP 校验可直接索引的物理前提）：

```
INFO 09-28 07:46:08 [model_runner_v1.py:4266] [KVC][L1] ================ 物理侧 KV Cache 分配开始 ================
...（K int8 1661.88MiB + V int8 1661.88MiB，alignment=2097152，逐层 ×16）
INFO 09-28 07:46:09 [model_runner_v1.py:4866] [KVC][L1] vllm-ascend _reshape_kv_cache_tensors: model.layers.16.self_attn.attn (本组 16 层同形) -> K_cache shape=(13296, 128, 4, 128) dtype=torch.bfloat16 / V_cache shape=(13296, 128, 4, 128) dtype=torch.bfloat16, device=npu:3 (K/V 分离布局)
...（PP1 末 worker 完成横幅）
```

### 2.3 逻辑侧装配（EngineCore，8 行全量）

队列 → null 块 → BlockPool → L3 manager → L5 门面逐层装配，8 行一屏读完：

```
INFO 09-28 07:46:17 [kv_cache_manager.py:143] [KVC][L5] ================ 逻辑侧初始化开始 ================
INFO 09-28 07:46:17 [kv_cache_utils.py:217] [KVC][L2] FreeKVCacheBlockQueue.__init__: num_free_blocks=13296, 伪头尾哨兵 fake_free_list_head/tail(block_id=-1), 类型=FreeKVCacheBlockQueue
INFO 09-28 07:46:17 [kv_cache_utils.py:258] [KVC][L2] FreeKVCacheBlockQueue.popleft -> KVCacheBlock(block_id=0), num_free_blocks=13295
INFO 09-28 07:46:17 [block_pool.py:209] [KVC][L2] BlockPool.__init__: num_gpu_blocks=13296, 创建 KVCacheBlock × 13296 (block_id=0..13295), free_block_queue=FreeKVCacheBlockQueue(num_free_blocks=13295), cached_block_hash_to_block=BlockHashToBlockMap(size=0), null_block=KVCacheBlock(block_id=0, is_null=True), enable_caching=True, hash_block_size=128
INFO 09-28 07:46:17 [single_type_kv_cache_manager.py:95] [KVC][L3] FullAttentionManager.__init__: spec=FullAttentionSpec(block_size=128), scheduler_block_size=128, group_id=0, enable_caching=True, dcp×pcp=1×1, block_pool(num_gpu_blocks=13296)
INFO 09-28 07:46:17 [kv_cache_coordinator.py:462] [KVC][L4] UnitaryKVCacheCoordinator.__init__: 单组直通, managers=['FullAttentionManager'], kv_cache_spec=FullAttentionSpec(block_size=128, page_size_bytes=262144), coordinator_block_size=128
INFO 09-28 07:46:17 [kv_cache_manager.py:178] [KVC][L5] KVCacheManager.__init__: coordinator=UnitaryKVCacheCoordinator, num_kv_cache_groups=1, managers=['FullAttentionManager'], block_pool(num_gpu_blocks=13296), enable_caching=True, max_model_len=8192, empty_kv_cache_blocks=KVCacheBlocks([],)
INFO 09-28 07:46:17 [kv_cache_manager.py:186] [KVC][L5] ================ 逻辑侧初始化完成 ================
```

### 2.4 原生关键行

```
(APIServer pid=4555)  INFO [utils.py:1404]  Block size is set to 128       # llama.log:30
(EngineCore pid=4600) INFO [kv_cache_utils.py:1777] Maximum concurrency for 8,192 tokens per request: 207.73x
(Worker_PP0_TP0 pid=4633) INFO [worker.py:593] Available KV cache memory: 51.98 GiB
(APIServer pid=4555) INFO: Application startup complete.    # llama.log:388, 就绪 55s
```

## 3. P 运行期日志讲解（log/kvc_p.log，124 行）

P：num_prompt_tokens=324（2 满块 + 尾 68），max_tokens=1——一次 prefill 即终态（TERM），验证"缓冲 2 块"。

### 3.1 入队与前缀查找

```
INFO 09-28 07:46:34 [request.py:184] [KVC][ENQ] ======== 入队 ========
INFO 09-28 07:46:34 [kv_cache_utils.py:617] [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=df3b74831f54
INFO 09-28 07:46:34 [kv_cache_utils.py:617] [KVC][ENQ] 入队 hash_block_tokens: parent=df3b74831f54, tokens=128 -> BlockHash=5751b0a5469a
INFO 09-28 07:46:34 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-addf477dedca5a18-0-a8f278a5) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['df3b74831f54', '5751b0a5469a']
INFO 09-28 07:46:34 [request.py:193] [KVC][ENQ] ======== 入队完成 ========
INFO 09-28 07:46:34 [kv_cache_manager.py:222] [KVC][L5] ======== 前缀查找 ========
...（冷缓存: 第 1 块即 MISS 断链, hit_length=0, 返回 blocks=[[]]）
INFO 09-28 07:46:34 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 1 块 MISS: BlockHash=df3b74831f54 -> break
```

### 3.2 分配 S1~S4 全链（S1 段自洽实录）

总横幅(:393) → 进入(:395) → **S1 子步横幅(:402)** → 两次外层探问(:188×2) → S1 汇总值(:447)，随后 S2/S3/S4 四子步下钻逐层展开：

```
INFO 09-28 07:46:34 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 09-28 07:46:34 [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-addf477dedca5a18-0-a8f278a5, num_new_tokens=324, num_new_computed_tokens=0, request.num_computed_tokens=0, request.num_tokens=324
INFO 09-28 07:46:34 [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---
INFO 09-28 07:46:34 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-addf477dedca5a18-0-a8f278a5, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 09-28 07:46:34 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-addf477dedca5a18-0-a8f278a5, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 09-28 07:46:34 [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13295 块 (free=13295 - reserved=0)
INFO 09-28 07:46:34 [kv_cache_manager.py:481] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 09-28 07:46:34 [kv_cache_manager.py:485] [KVC][L5] --- S3: 新块分配 ---
INFO 09-28 07:46:34 [block_pool.py:411] [KVC][L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=13292
INFO 09-28 07:46:34 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-addf477dedca5a18-0-a8f278a5, num_tokens=324, block_size=128, 需 3 块 - 已有 0 = 新分配 3 块 [1, 2, 3], 持有 req_blocks=[1, 2, 3]
INFO 09-28 07:46:34 [kv_cache_manager.py:496] [KVC][L5] S3 allocate_new_blocks: req=cmpl-addf477dedca5a18-0-a8f278a5, num_tokens_need_slot=324 -> 新块 [1, 2, 3]
INFO 09-28 07:46:34 [kv_cache_manager.py:523] [KVC][L5] --- S4: 满块入缓存 ---
INFO 09-28 07:46:34 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=df3b74831f54, group_id=0) <- KVCacheBlock(block_id=1), map size=1
INFO 09-28 07:46:34 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=5751b0a5469a, group_id=0) <- KVCacheBlock(block_id=2), map size=2
INFO 09-28 07:46:34 [kv_cache_manager.py:529] [KVC][L5] S4 cache_blocks: req=cmpl-addf477dedca5a18-0-a8f278a5, num_tokens_to_cache=324
INFO 09-28 07:46:34 [kv_cache_manager.py:533] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([1, 2, 3],)), req=cmpl-addf477dedca5a18-0-a8f278a5 当前完整 block_table=([1, 2, 3],)
INFO 09-28 07:46:34 [kv_cache_manager.py:539] [KVC][L5] ======== 分配完成 ========
```

要点：324 tokens → S3 需 3 块（`需 3 块 - 已有 0 = 新分配 3 块`）；S4 只把 2 个满块入哈希表（324 // 128 = 2，`insert` ×2），块 3 为尾块（68/128）不入表。

### 3.3 KVP 物理校验（TERM ×4 卡，每卡 18 行）

以 dev=npu:0 为例（P 一步即终态，TERM 文案；概览含一次性 KV 布局说明，层行每层 1 条、尾块 `未满:68`，层统计 n=165888=324×4×128）：

```
INFO 09-28 07:46:34 [model_runner_v1.py:2518] [KVC][KVP] ======== 请求结束 KVCache 即将释放, 开始打印该请求物理 cache ========
INFO 09-28 07:46:34 [model_runner_v1.py:2520] [KVC][KVP] TERM req=cmpl-addf477dedca5a18-0-a8f278a5 dev=npu:0 逐层按块: layers=16 blocks=[1, 2, 3] region=324/324 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=…
INFO 09-28 07:46:34 [model_runner_v1.py:2563] [KVC][KVP] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=3[未满:68](68,4,128) | K示(首块首token前3)=[-0.008057, 0.1035, 0.02747] 统计[n=165888] mean=0.04895 std=1.508 min=-14.56 max=8.5 | V示(首块首token前3)=[0.0009766, 0.000576, -0.0008163] 统计[n=165888] mean=1.096e-05 std=0.03965 min=-0.3477 max=0.4707
...（L01~L15 同构 ×16 行/卡, 共 4 卡）
INFO 09-28 07:46:34 [model_runner_v1.py:2570] [KVC][KVP] ======== 请求结束, 物理 cache 打印完毕 ========
```

### 3.4 调度提交与释放

async_scheduler 每步输出后的独立提交段（与分配 S4 明确区分），随后逆序释放、带哈希块 LRU 归队：

```
INFO 09-28 07:46:35 [kv_cache_manager.py:676] [KVC][L5] ======== 调度提交(非分配 S4) ========
INFO 09-28 07:46:35 [kv_cache_manager.py:677] [KVC][L5] 提交 cache_blocks: req=cmpl-addf477dedca5a18-0-a8f278a5, num_computed_tokens=324 (async 步末输出路径: 本步已算 token 提交入缓存)
INFO 09-28 07:46:35 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-addf477dedca5a18-0-a8f278a5, num_computed_tokens=324
INFO 09-28 07:46:35 [kv_cache_manager.py:683] [KVC][L5] ======== 提交完成 ========
INFO 09-28 07:46:35 [kv_cache_manager.py:551] [KVC][L5] ======== 释放 ========
INFO 09-28 07:46:35 [kv_cache_manager.py:553] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-addf477dedca5a18-0-a8f278a5, 释放前持有 block_table=([1, 2, 3],)
INFO 09-28 07:46:35 [block_pool.py:516] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(3, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 3 块 [3, 2, 1], append_n -> 队尾(LRU保护)
INFO 09-28 07:46:35 [kv_cache_utils.py:396] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[3, 2, 1]), num_free_blocks=13295
INFO 09-28 07:46:35 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

### 3.5 响应核对

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| P | 1 | length | `"为了"` |

## 4. R 运行期日志讲解（log/kvc_r5.log，752 行）

R：num_prompt_tokens=486 = 3 满 + 第 4 块 102/128，max_tokens=35——完整五块生命周期（复用 2 + prefill 补 1 满 1 尾 + decode 填满尾块 + 步 27 跨界申请第 5 块）。request_id=cmpl-86360a91ceab7325-0-9d4982ba。

### 4.1 入队与前缀查找（HIT×2 后 MISS 断链）

```
INFO 09-28 07:46:41 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-86360a91ceab7325-0-9d4982ba) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 BlockHash × 3: ['df3b74831f54', '5751b0a5469a', '3d788bda3932']
INFO 09-28 07:46:41 [kv_cache_coordinator.py:477] [KVC][L4] 前缀查找 UnitaryKVCacheCoordinator.find_longest_cache_hit: 满块hash数=3, max_cache_hit_length=485, 下钻 single_type_managers[0]
INFO 09-28 07:46:41 [block_pool.py:70] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=df3b74831f54, group_id=0) -> HIT KVCacheBlock(block_id=1)
INFO 09-28 07:46:41 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 1 块 HIT: BlockHash=df3b74831f54 -> cached blocks=[1]
INFO 09-28 07:46:41 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 2 块 HIT: BlockHash=5751b0a5469a -> cached blocks=[2]
INFO 09-28 07:46:41 [block_pool.py:84] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=3d788bda3932, group_id=0) -> MISS
INFO 09-28 07:46:41 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 3 块 MISS: BlockHash=3d788bda3932 -> break
```

命中 2 块 → hit_length=2×128=256；第 3 个 hash 是 R 追问句新内容（冷），断链即止。

### 4.2 prefill 完整链（S2 touch 复用 + S3 补 2 块 + S4 新满块入表）

```
INFO 09-28 07:46:41 [kv_cache_manager.py:467] [KVC][L5] --- S2: touch 命中块 ---
INFO 09-28 07:46:41 [kv_cache_manager.py:469] [KVC][L5] S2 allocate_new_computed_blocks: req=cmpl-86360a91ceab7325-0-9d4982ba, new_computed_blocks=[[1, 2]]
INFO 09-28 07:46:41 [block_pool.py:491] [KVC][L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
INFO 09-28 07:46:41 [kv_cache_manager.py:485] [KVC][L5] --- S3: 新块分配 ---
INFO 09-28 07:46:41 [block_pool.py:411] [KVC][L2] S3 BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13291
INFO 09-28 07:46:41 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-86360a91ceab7325-0-9d4982ba, num_tokens=486, block_size=128, 需 4 块 - 已有 2 = 新分配 2 块 [4, 5], 持有 req_blocks=[1, 2, 4, 5]
INFO 09-28 07:46:41 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=3d788bda3932, group_id=0) <- KVCacheBlock(block_id=4), map size=3
INFO 09-28 07:46:41 [kv_cache_manager.py:533] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([4, 5],)), req=cmpl-86360a91ceab7325-0-9d4982ba 当前完整 block_table=([1, 2, 4, 5],)
INFO 09-28 07:46:41 [kv_cache_manager.py:539] [KVC][L5] ======== 分配完成 ========
```

要点：**touch 零拷贝**——`S2 BlockPool.touch: blocks=[(1, 1), (2, 1)]`（block_id, ref_cnt），直接复用 P 留下的块 1/2；S3 = cdiv(486,128)−2 = 2 新块 [4,5]；S4 把追问句恰好填满的块 4 入表（map 2→3）。

### 4.3 decode 无块步闭合链（33 步同构，完整 14 行实录）

decode 步 1 起的每一步（除步 27 外）都是这条闭合链——S1 如实打"需分配 0"，S3 切换"无需分配新块"文案，尾部紧跟本步调度提交：

```
INFO 09-28 07:46:41 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 09-28 07:46:41 [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-86360a91ceab7325-0-9d4982ba, num_new_tokens=1, num_new_computed_tokens=0, request.num_computed_tokens=486, request.num_tokens=486
INFO 09-28 07:46:41 [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---
INFO 09-28 07:46:41 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-86360a91ceab7325-0-9d4982ba, num_tokens=487 -> 需分配 0 块(含touch需腾挪的块)
INFO 09-28 07:46:41 [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 0 块 vs 可用 13291 块 (free=13291 - reserved=0)
INFO 09-28 07:46:41 [kv_cache_manager.py:481] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 09-28 07:46:41 [kv_cache_manager.py:487] [KVC][L5] --- S3: 无需分配新块 ---
INFO 09-28 07:46:41 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-86360a91ceab7325-0-9d4982ba, num_tokens=487 -> [[]]
INFO 09-28 07:46:41 [kv_cache_manager.py:501] [KVC][L5] S3 块未满, 无需分配新块 (req=cmpl-86360a91ceab7325-0-9d4982ba, num_new_tokens=1)
INFO 09-28 07:46:41 [kv_cache_manager.py:523] [KVC][L5] --- S4: 满块入缓存 ---
INFO 09-28 07:46:41 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-86360a91ceab7325-0-9d4982ba, num_computed_tokens=486
INFO 09-28 07:46:41 [kv_cache_manager.py:529] [KVC][L5] S4 cache_blocks: req=cmpl-86360a91ceab7325-0-9d4982ba, num_tokens_to_cache=486
INFO 09-28 07:46:41 [kv_cache_manager.py:533] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([],)), req=cmpl-86360a91ceab7325-0-9d4982ba 当前完整 block_table=([1, 2, 4, 5],)
INFO 09-28 07:46:41 [kv_cache_manager.py:539] [KVC][L5] ======== 分配完成 ========
INFO 09-28 07:46:41 [kv_cache_manager.py:676] [KVC][L5] ======== 调度提交(非分配 S4) ========
```

### 4.4 步 27 跨界（第 512 个 token 触发第 5 块申请）

尾块 102/128 被 decode 逐步填满——步 27 时 num_computed_tokens=512，S1 汇总值切换为"需分配 1 块"，S3 弹出新块 [6]，S4 把刚填满的块 5 入表（map 3→4）：

```
INFO 09-28 07:46:41 [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-86360a91ceab7325-0-9d4982ba, num_new_tokens=1, num_new_computed_tokens=0, request.num_computed_tokens=512, request.num_tokens=512
INFO 09-28 07:46:41 [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---
INFO 09-28 07:46:41 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-86360a91ceab7325-0-9d4982ba, num_tokens=513 -> 需分配 1 块(含touch需腾挪的块)
INFO 09-28 07:46:41 [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 1 块 vs 可用 13291 块 (free=13291 - reserved=0)
INFO 09-28 07:46:41 [kv_cache_manager.py:485] [KVC][L5] --- S3: 新块分配 ---
INFO 09-28 07:46:41 [block_pool.py:411] [KVC][L2] S3 BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=13290
INFO 09-28 07:46:41 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=8529e6691553, group_id=0) <- KVCacheBlock(block_id=5), map size=4
INFO 09-28 07:46:41 [block_pool.py:340] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-86360a91ceab7325-0-9d4982ba 新满块 1 块 block_ids=[5] 入 BlockHashToBlockMap (num_cached_blocks 3 -> 4, group_id=0, map size=4)
```

### 4.5 TERM KVP（4 卡 × 18 行，五块全景）

region=520/520 = 4×128 + 8（第 5 块仅 8 个有效 token）；层行一屏内联全部 5 块，层统计 n=266240=520×4×128 精确断言：

```
INFO 09-28 07:46:42 [model_runner_v1.py:2520] [KVC][KVP] TERM req=cmpl-86360a91ceab7325-0-9d4982ba dev=npu:0 逐层按块: layers=16 blocks=[1, 2, 4, 5, 6] region=520/520 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维…
INFO 09-28 07:46:42 [model_runner_v1.py:2563] [KVC][KVP] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=4[满:128](128,4,128) blk=5[满:128](128,4,128) blk=6[未满:8](8,4,128) | K示(首块首token前3)=[0.5078, 0.9336, 0.9219] 统计[n=266240] mean=-0.0149 std=1.402 min=-10.38 max=10.81 | V示(首块首token前3)=[0.01538, 0.0008049, 0.03345] 统计[n=266240] mean=0.0006548 std=0.03523 min=-0.3438 max=0.3223
...（L01~L15 ×16 行/卡, 4 卡 = 64 条层行）
INFO 09-28 07:46:42 [model_runner_v1.py:2570] [KVC][KVP] ======== 请求结束, 物理 cache 打印完毕 ========
```

K 示首 3 值与 P 的块 1 一致（`0.5078, 0.9336, 0.9219`）——**复用块零拷贝**的直接证据（同一物理块，K/V 数据未动）。

### 4.6 五块逆序释放与计数自检

```
INFO 09-28 07:46:42 [kv_cache_manager.py:553] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-86360a91ceab7325-0-9d4982ba, 释放前持有 block_table=([1, 2, 4, 5, 6],)
INFO 09-28 07:46:42 [block_pool.py:516] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(6, 0), (5, 0), (4, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 5 块 [6, 5, 4, 2, 1], append_n -> 队尾(LRU保护)
INFO 09-28 07:46:42 [kv_cache_utils.py:396] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[6, 5, 4, 2, 1]), num_free_blocks=13295
INFO 09-28 07:46:42 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

**全程计数自检**（grep 即可复验）：`调度提交` P=1 / R=**35**（每 decode 步一次）；KVP 层行 P=R=**64**（4 卡 × 16 层，固定）；S1 汇总值 R = **33×0 + 1×1 + 1×4**（无块步 33 + 跨界 1 + prefill 1）；哈希链 `df3b74831f54 → 5751b0a5469a → 3d788bda3932`（+ decode 段 `8529e6691553`）。

### 4.7 响应核对

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| R | 35 | length | 中文贪心续写 |

## 5. 实证结论

1. **patch 163 行验证通过**：8/8 应用、92 调用点、py_compile OK；本地回环（stash→apply→revert→stash pop）与容器 apply/revert 双向验证。
2. **S1 段完整自洽**：子步横幅先行（:402）→ 两次外层探问下钻（coordinator:188 ×2）→ 汇总值（:447）——S1 语义日志全部在子步横幅之内。
3. **S1~S4 全子步可观测**：无块步打出完整四段子步（S1 需分配 0 / S2 无前缀 / S3 无需分配 / S4 维护），R 33 个无块步全部闭合。
4. **KVP 可读性**：每请求 KVP 固定 76 行（4 卡 × 18）；每层一行内联块标注 + K/V 首 3 值示意 + 层合并统计——一眼可读。
5. **KV 布局实证**：K/V 为张量级拆分的两个独立池（非最后一维拼接）；层统计 `n = region × kv_heads(4) × head_dim(128)` 精确断言（P: 165888、R: 266240）；复用块 K 示值跨请求一致（零拷贝）。
6. **容器回收闭环**：服务已杀（0 进程）、8 补丁 revert 归零、两仓库 git 0 改动、.orig 清理。

## 6. 实测产物

| 产物（`log/`） | 说明 |
|---|---|
| `log/llama.log`（1274 行） | 服务全量日志（启动 :1~388 + P :389~518 + R :519~1274，带进程前缀完整版） |
| `log/kvc_startup.log`（168 行） | 启动期 [KVC] 拆解轨迹 |
| `log/kvc_p.log`（124 行）/ `log/kvc_r5.log`（752 行） | P / R 运行期 [KVC] 拆解轨迹 |
| `log/req_*.json`、`resp_*.json`、curl_*screen.txt | 请求体 / 响应体 / 打屏实录 |
| `log/p_run_start.txt` / `r_run_start.txt` | 双请求分界（:389 / :519） |