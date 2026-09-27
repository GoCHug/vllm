# 端到端实录：还原 → patch 应用验证 → 启动期 KVCache 初始化全流程 → P/R 运行期全流程

> 本文是 `2_kvc_cn_curl_case.md`（用例）与 `patch/`（打印补丁）的**端到端正式验证记录**：从干净源码出发，以 patch 方式注入 79 处 `[KVC]` 打印（grep 计数 141 行，含注释行；横幅/子步标记升级后实测），记录一次完整的服务启动初始化与 P/R 双请求生命周期。实测 2026-09-27（log 内时间戳 09-26 17:51~17:52，容器时钟为 UTC-8，比北京时间慢 16 小时）。
>
> 环境：gggtest（PP2TP2 4 卡 Ascend910，pod 当日从 Stopped 重新拉起）、vllm 0.23.0（`/vllm-workspace/vllm`）+ vllm-ascend 0.23.0（`/vllm-workspace/vllm-ascend`）。进程：APIServer pid=1750、EngineCore pid=1789、Worker pid=1936~1939（PP0_TP0/PP0_TP1/PP1_TP0/PP1_TP1）。

## 1. 还原源码至原始状态

```bash
cd /vllm-workspace/vllm && git checkout -- vllm/v1/request.py vllm/v1/core/kv_cache_utils.py vllm/v1/core/block_pool.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/engine/core.py vllm/v1/worker/gpu_model_runner.py
cd /vllm-workspace/vllm-ascend && git checkout -- vllm_ascend/worker/model_runner_v1.py
```

验证：9 个文件 `grep -c "\[KVC\]"` 全部为 **0**（干净基线）。

## 2. 以 patch 方式应用打印代码（验证 patch 文件正确可用）

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
# 一键完成: Phase0 状态检查 -> Phase1 dry-run 9/9 预检 -> Phase2 9/9 应用 -> Phase3 逐文件计数(合计 141 行) + py_compile
# 回滚: ./revert_patches.sh（patch -R 反向应用, 不依赖备份）
```

应用后逐文件 `[KVC]` grep 计数（含注释行；合计 141 行 = 79 个 `logger.info` 打印调用点）：

| 文件 | grep 计数 | 打印调用点 |
|---|---|---|
| `v1/request.py` / `v1/core/kv_cache_utils.py` | 5 / 12 | 3 / 6 |
| `v1/core/block_pool.py` / `kv_cache_manager.py` | 27 / 37 | 14 / 23 |
| `v1/core/kv_cache_coordinator.py` / `single_type_kv_cache_manager.py` | 17 / 18 | 9 / 9 |
| `v1/engine/core.py` / `v1/worker/gpu_model_runner.py` | 13 / 4 | 9 / 2 |
| `vllm_ascend/worker/model_runner_v1.py` | 8 | 4 |

`python3 -m py_compile`（9 文件）→ **COMPILE_OK**。结论：patch 文件在"原始代码"上一发命中、无 fuzz、无 reject。9 个 patch 仅含 logger 打印插入 + logger 导入 + 7 处值捕获重写（清单见 `../patch/README.md` §5 第 5 条），无任何其他源码改动。

**本轮升级：三级横幅 + S1~S4 子步标记**（141 行中新增约 19 个调用点）：

| 级别 | 样式 | 标记的阶段 |
|---|---|---|
| 初始化横幅 | `================ 阶段名 ================` | 配置侧/物理侧/逻辑侧三个一次性装配阶段的开始与完成 |
| 阶段横幅 | `======== 阶段名 ========` | 每请求的入队、前缀查找、释放（及其"完成"） |
| 分配小横幅 | `-------- 分配 S1~S4 --------` / `-------- 分配完成 --------` | 每次调度分配（每请求每步一对） |
| 子步标记 | `--- S1: 容量检查---` 等 | S1 容量检查 / S2 touch 命中块 / S3 新块分配 / S4 满块入缓存 |

> S2 子步仅在存在命中块时出现：P 冷启动无命中，P 轨迹中 S2 不打；R 借助 P 种下的缓存，首个 prefill 分配出现一次 `--- S2: touch 命中块 ---`。

## 3. 启动期 KVCache 初始化全流程（log/kvc_startup.log，168 行）

启动命令与就绪标志：

```bash
bash scripts/start.sh        # 服务后台启动, 日志 -> log/llama.log; 就绪标志: grep 'Application startup complete' log/llama.log
head -389 log/llama.log | grep '\[KVC\]' > log/kvc_startup.log    # 启动段 168 行纯 [KVC]（389 = P 请求起始行 p_run_start）
```

分层统计：`CFG 84 + L1 76 + 逻辑侧装配 8 = 168`（逻辑侧 8 = 首尾横幅 2 + L2 3 + L3 1 + L4 1 + L5 1）——**配置侧→物理侧→逻辑侧装配全链路打印齐全，且三段首尾均有开始/完成横幅**。

### 3.1 配置侧（EngineCore，84 行 CFG，含首尾横幅）

```
INFO [core.py:265] [KVC][CFG] ================ 配置侧 KVCache 编排开始 ================
INFO [core.py:267] [KVC][CFG] determine_available_memory: 各 worker 可用 KV 显存 = ['51.98GiB', '51.99GiB', '51.94GiB', '51.95GiB']   # ① 测预算(profile_run)
INFO [core.py:280] [KVC][CFG] worker0 KVCacheConfig: num_blocks=13295, groups数=1, tensors数=16          # ③ 编排产物(逐 worker ×4)
INFO [core.py:285] [KVC][CFG]   [0] KVCacheGroupSpec(group_id=0): layers=16 (首层 model.layers.0.self_attn.attn, 末层 model.layers.15.self_attn.attn), is_eagle_group=False
INFO [core.py:290] [KVC][CFG]   [0]   kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, ...)
INFO [core.py:294] [KVC][CFG]   [0]   page_size_bytes=262144 (256.0KB/层/块), storage_block_size=128
INFO [core.py:300] [KVC][CFG]   [0] KVCacheTensor: size=3485204480 bytes (3323.75MiB), shared_by=1 层 (model.layers.0.self_attn.attn)   # ×16 张; 单层直显层名(不再有 "X .. X" 冗余)
INFO [core.py:317] [KVC][CFG] 最终 scheduler KVCacheConfig: num_blocks=13295 (跨 worker min 对齐), cache_config.num_gpu_blocks=13295, block_size=128
INFO [core.py:322] [KVC][CFG] ================ 配置侧 KVCache 编排完成 ================
```

主线：① 算规格 → ② 测预算 → ③ `KVCacheSpec → KVCacheGroupSpec → KVCacheTensor → KVCacheConfig` → min 对齐下发。本轮 profile 实测 **13295 块**（profile 跨次启动存在 ±1~2 块的测量波动，上一轮为 13296）。

### 3.2 物理侧（4 个 worker 进程，76 行 L1，vllm-ascend 路径，含首尾横幅）

```
[Worker][model_runner_v1.py:4097] [KVC][L1] ================ 物理侧 KV Cache 分配开始 ================        # ×4 worker
[Worker][model_runner_v1.py:4259] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.0.self_attn.attn: KVCacheTensor(size=3485204480 bytes = 3323.75MiB) -> K int8 1661.88MiB + V int8 1661.88MiB (alignment=2097152, device=npu:0~3)   # ×16 层/worker
[Worker][model_runner_v1.py:4697] [KVC][L1] vllm-ascend _reshape_kv_cache_tensors: model.layers.0(或16).self_attn.attn (本组 16 层同形) -> K_cache shape=(13295, 128, 4, 128) dtype=torch.bfloat16 / V_cache 同形 (K/V 分离布局)
[Worker][model_runner_v1.py:4748] [KVC][L1] ================ 物理侧 KV Cache 分配完成 ================        # ×4 worker
```

4 worker 设备映射：PP0_TP0→npu:0、PP0_TP1→npu:1、PP1_TP0→npu:2、PP1_TP1→npu:3；`block_id` 索引 K/V 两张张量各自第 0 维。

### 3.3 逻辑侧装配（EngineCore，8 行，含首尾横幅与 L3 初始化打印）

```
INFO [kv_cache_manager.py:143] [KVC][L5] ================ 逻辑侧初始化开始 ================
INFO [kv_cache_utils.py:217] [KVC][L2] FreeKVCacheBlockQueue.__init__: num_free_blocks=13295, 伪头尾哨兵 fake_free_list_head/tail(block_id=-1)
INFO [kv_cache_utils.py:258] [KVC][L2] FreeKVCacheBlockQueue.popleft -> KVCacheBlock(block_id=0), num_free_blocks=13294      # null_block 摘取
INFO [block_pool.py:209] [KVC][L2] BlockPool.__init__: num_gpu_blocks=13295, 创建 KVCacheBlock × 13295 (block_id=0..13294), free_block_queue=FreeKVCacheBlockQueue(num_free_blocks=13294), null_block=KVCacheBlock(block_id=0, is_null=True), enable_caching=True, hash_block_size=128
INFO [single_type_kv_cache_manager.py:95] [KVC][L3] FullAttentionManager.__init__: spec=FullAttentionSpec(block_size=128), scheduler_block_size=128, group_id=0, enable_caching=True, dcp×pcp=1×1, block_pool(num_gpu_blocks=13295)    # 本轮新增: L3 也入镜
INFO [kv_cache_coordinator.py:462] [KVC][L4] UnitaryKVCacheCoordinator.__init__: 单组直通, managers=['FullAttentionManager'], kv_cache_spec=FullAttentionSpec(block_size=128, page_size_bytes=262144), coordinator_block_size=128
INFO [kv_cache_manager.py:178] [KVC][L5] KVCacheManager.__init__: coordinator=UnitaryKVCacheCoordinator, num_kv_cache_groups=1, managers=['FullAttentionManager'], block_pool(num_gpu_blocks=13295), enable_caching=True, max_model_len=8192, empty_kv_cache_blocks=KVCacheBlocks([],)
INFO [kv_cache_manager.py:186] [KVC][L5] ================ 逻辑侧初始化完成 ================
```

### 3.4 原生关键行（log/llama.log 启动段）

```
(APIServer pid=1750)    INFO [utils.py:1404]  Block size is set to 128 if prefix cache or chunked prefill is enabled.    # llama.log:30
(EngineCore pid=1789)   INFO [kv_cache_utils.py:1777] Maximum concurrency for 8,192 tokens per request: 207.73x
(Worker_PP0_TP0 pid=1936) INFO [worker.py:593] Available KV cache memory: 51.98 GiB
(各 Worker pid=1936~1939) Loading model weights took 3.7454 GB
(APIServer pid=1750) INFO: Application startup complete.    # llama.log:389, 就绪耗时 ~54s
```

## 4. 运行期：P/R 双请求全流程

请求体 `log/req_p.json` / `log/req_r5.json`（与 `2_kvc_cn_curl_case.md` §3 字节级一致）：

```
bash scripts/curl_p_r.sh     # 依次发送 P、R；自动记录起始行/打屏/响应, 并提取三条 [KVC] 轨迹到 log/
```

### 4.1 P：缓冲 2 块（log/kvc_p.log，44 行）

```
INFO [request.py:184] [KVC][ENQ] ======== 入队 ========
INFO [kv_cache_utils.py:617] [KVC][ENQ] hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=1b158fb27097         # 链首
INFO [kv_cache_utils.py:617] [KVC][ENQ] hash_block_tokens: parent=1b158fb27097, tokens=128 -> BlockHash=a5323e08231a
INFO [request.py:187] [KVC][ENQ] Request(...) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['1b158fb27097', 'a5323e08231a']
INFO [request.py:193] [KVC][ENQ] ======== 入队完成 ========
INFO [kv_cache_manager.py:222] [KVC][L5] ======== 前缀查找 ========                             # 冷缓存
INFO [block_pool.py:84] [KVC][L2] get_one_block: key=(hash=1b158fb27097, group_id=0) -> MISS
INFO [single_type_kv_cache_manager.py:607] [KVC][L3]   第 1 块 MISS: BlockHash=1b158fb27097 -> break                        # 冷缓存, 首块断链
INFO [kv_cache_manager.py:367] [KVC][L5] -------- 分配 S1~S4 --------
INFO [kv_cache_manager.py:439] [KVC][L5] --- S1: 容量检查---
INFO [kv_cache_manager.py:441] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13294 块
INFO [block_pool.py:411] [KVC][L2] BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=13291
INFO [kv_cache_manager.py:478] [KVC][L5] --- S3: 新块分配 ---
INFO [block_pool.py:108] [KVC][L2] insert: (hash=1b158fb27097) <- KVCacheBlock(block_id=1), map size=1    # 满块1 入表
INFO [block_pool.py:108] [KVC][L2] insert: (hash=a5323e08231a) <- KVCacheBlock(block_id=2), map size=2    # 满块2 入表
INFO [block_pool.py:340] [KVC][L2] cache_full_blocks: 新满块 2 块 block_ids=[1, 2] 入表 (num_cached_blocks 0 -> 2)    # 缓冲 2 块!
INFO [kv_cache_manager.py:499] [KVC][L5] --- S4: 满块入缓存 ---
INFO [kv_cache_manager.py:508] [KVC][L5] allocate_slots 返回: KVCacheBlocks(blocks=([1, 2, 3],)), block_table=([1, 2, 3],)
INFO [kv_cache_manager.py:513] [KVC][L5] -------- 分配完成 --------
INFO [kv_cache_manager.py:525] [KVC][L5] ======== 释放 ========
INFO [kv_cache_manager.py:527] [KVC][L5] free: 释放前持有 block_table=([1, 2, 3],)
INFO [block_pool.py:516] [KVC][L2] free_blocks: [(3,0),(2,0),(1,0)] 归零回收 3 块 [3, 2, 1], append_n -> 队尾
INFO [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

尾块 3（68/128）未满不入表；释放后 1/2/3 挂队尾带哈希，缓存表留存 2 个 hash。**P 无 `--- S2 ---` 标记（0 命中，S2 分支整段跳过）**。

### 4.2 R：五块生命周期（log/kvc_r5.log，537 行）

横幅/标记统计：`入队/入队完成/前缀查找/前缀查找完成/释放/释放完成 各 ×1；分配 S1~S4 与 分配完成 各 ×35（1 次 prefill + 34 次 decode 步）；S1×35 / S2×1 / S3×35 / S4×35`。

```
① 复用 2 块 (第 3 hash MISS 断链):
INFO [request.py:187] [KVC][ENQ] Request(...) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 × 3: ['1b158fb27097', 'a5323e08231a', '0dc651ebed07']
INFO [kv_cache_coordinator.py:477] [KVC][L4] find_longest_cache_hit: 满块hash数=3, max_cache_hit_length=485
INFO [block_pool.py:70] [KVC][L2] get_one_block: (hash=1b158fb27097) -> HIT KVCacheBlock(block_id=1)
INFO [single_type_kv_cache_manager.py:599] [KVC][L3]   第 1 块 HIT: BlockHash=1b158fb27097 -> cached blocks=[1]
INFO [block_pool.py:70] [KVC][L2] get_one_block: (hash=a5323e08231a) -> HIT KVCacheBlock(block_id=2)
INFO [single_type_kv_cache_manager.py:599] [KVC][L3]   第 2 块 HIT: BlockHash=a5323e08231a -> cached blocks=[2]
INFO [block_pool.py:84] [KVC][L2] get_one_block: (hash=0dc651ebed07) -> MISS
INFO [single_type_kv_cache_manager.py:607] [KVC][L3]   第 3 块 MISS: BlockHash=0dc651ebed07 -> break                       # P 只种了前 2 块, 中间断链
INFO [kv_cache_coordinator.py:495] [KVC][L4] find_longest_cache_hit 返回: hit_blocks=[[1, 2]], hit_length=256
② prefill: touch 复用 + 新申请 2 块 (1 满 + 1 尾):
INFO [kv_cache_manager.py:459] [KVC][L5] --- S2: touch 命中块 ---                              # 本轮唯一一次 S2
INFO [single_type_kv_cache_manager.py:243] [KVC][L3] allocate_new_computed_blocks: touch 命中块 [1, 2]
INFO [block_pool.py:491] [KVC][L2] BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)              # 零拷贝共享
INFO [kv_cache_manager.py:441] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 4 块 vs 可用 13294 块            # S1 报总需求 cdiv(486,128)=4(含待 touch 2)
INFO [block_pool.py:411] [KVC][L2] BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13290   # 实际只新弹 2 块
INFO [block_pool.py:108] [KVC][L2] insert: (hash=0dc651ebed07) <- KVCacheBlock(block_id=4), map size=3   # 块4 追问句恰填满
INFO [block_pool.py:340] [KVC][L2] cache_full_blocks: 新满块 1 块 block_ids=[4] 入表 (num_cached_blocks 2 -> 3)     # 块5 (102/128) 未满不入
INFO [kv_cache_manager.py:508] [KVC][L5] allocate_slots 返回: ... block_table=([1, 2, 4, 5],)
decode 步 1~26: S1 恒 "需分配 0 块", 尾块 102 → 128 (每步一对 分配 S1~S4/分配完成 小横幅)
③ decode 填满 + 步 27 跨界申请第 5 块:
INFO [kv_cache_coordinator.py:188] [KVC][L4] get_num_blocks_to_allocate: num_tokens=513 -> 需分配 1 块             # cdiv(513,128)-4 = 1
INFO [kv_cache_manager.py:441] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 1 块 vs 可用 13290 块
INFO [block_pool.py:411] [KVC][L2] BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=13289  # 第 5 块!
INFO [block_pool.py:108] [KVC][L2] insert: (hash=1a4a87d90103) <- KVCacheBlock(block_id=5), map size=4   # 刚满的块5 与跨界申请合并发生在步 27
decode 步 28~34: 块 6 装 8/128 未满不入表 (第 35 个输出仅采样)
结束释放 (五块逆序):
INFO [kv_cache_manager.py:525] [KVC][L5] ======== 释放 ========
INFO [kv_cache_manager.py:527] [KVC][L5] free: 释放前持有 block_table=([1, 2, 4, 5, 6],)                        # 2 复用 + prefill 2 + decode 1
INFO [block_pool.py:516] [KVC][L2] free_blocks: [(6,0),(5,0),(4,0),(2,0),(1,0)] 归零回收 5 块 [6,5,4,2,1], append_n -> 队尾
INFO [kv_cache_utils.py:396] [KVC][L2] FreeKVCacheBlockQueue.append_n(blocks=[6, 5, 4, 2, 1]), num_free_blocks=13294
INFO [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

### 4.3 响应核对

| 请求 | completion_tokens | finish_reason |
|---|---|---|
| P | 1 | length |
| R | 35 | length |

## 5. 本轮实证结论

1. **patch 文件正确可用**：9 个 patch 在原始代码上 dry-run/apply 一发命中、计数（141 行）/编译/运行三重验证通过——`../patch/README.md` 的应用与回滚方法成立。
2. **启动期五层全流程带首尾横幅**：168 行 [KVC] 完整覆盖"配置侧（84）→ 物理侧（76）→ 逻辑侧装配（8，含 L3 打印）"，block_size=128、num_blocks=13295、K/V 分离 `(13295,128,4,128)×2` bf16、可用 KV 显存 51.98GiB；三段装配的开始/完成各有一条 `================` 横幅，日志肉眼可分段。
3. **运行期全流程带阶段横幅与 S1~S4 子步标记**：P 一次演示"断链种块"（44 行，全程无 S2——冷缓存无命中可 touch），R 一次演示"复用 2 + prefill 2（1满1尾）+ decode 填满跨界"完整五块生命周期（537 行，1 次 S2 + 35 对分配横幅）——`2_kvc_cn_curl_case.md` 的全部设计论断均在本轮拿到直接日志证据。

## 6. 本轮产物（容器与本地 `kvc/` 同步；目录结构见 `../README.md`）

| 产物（相对 `kvc/` 根，全部在 `log/`） | 说明 |
|---|---|
| `log/llama.log`（979 行） | 本轮服务全量日志（启动 + 双请求；P 起始 :389，R 起始 :439） |
| `log/kvc_startup.log`（168 行） | 启动期 [KVC] 拆解轨迹（原生关键行见 §3.4） |
| `log/kvc_p.log`（44 行）/ `log/kvc_r5.log`（537 行） | P / R 运行期 [KVC] 拆解轨迹 |
| `log/req_p.json`、`log/req_r5.json` | 请求体 |
| `log/resp_p.json`、`log/resp_r5.json` | 响应体（P=1 token / R=35 tokens，均 finish=length） |
| `log/curl_p_screen.txt`、`log/curl_r5_screen.txt` | curl 命令与终端打屏实录 |
| `log/p_run_start.txt` / `log/r_run_start.txt` | 双请求在 log/llama.log 中的起始行（389 / 439） |

> 容器可读化命令：`grep '\[KVC\]' log/llama.log`；权威数据以本文档与上表文件为准（同一 token 序列的哈希链在单次服务进程内完全一致，服务重启后因 NONE_HASH 种子随机而变化——见 `2_kvc_cn_curl_case.md` §7 注 4）。