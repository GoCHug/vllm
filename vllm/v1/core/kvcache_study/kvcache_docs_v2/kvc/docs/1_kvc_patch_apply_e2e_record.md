# 端到端实录：还原 → patch 应用验证 → 启动期 KVCache 初始化全流程 → P/R 运行期全流程

> 本文是 `2_kvc_cn_curl_case.md`（用例）与 `patch/`（打印补丁）的**端到端正式验证记录**：从干净源码出发，以 patch 方式注入 40 处 `[KVC]` 打印，记录一次完整的服务启动初始化与 P/R 双请求生命周期。实测时间 2026-09-23 09:13~09:15（容器时钟）。
>
> 环境：gggtest（PP2TP2 4 卡 Ascend910）、vllm 0.23.0（`/vllm-workspace/vllm`）+ vllm-ascend 0.23.0（`/vllm-workspace/vllm-ascend`）。

## 1. 还原源码至原始状态

```bash
cd /vllm-workspace/vllm && git checkout -- vllm/v1/request.py vllm/v1/core/kv_cache_utils.py vllm/v1/core/block_pool.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/engine/core.py vllm/v1/worker/gpu_model_runner.py
cd /vllm-workspace/vllm-ascend && git checkout -- vllm_ascend/worker/model_runner_v1.py
```

验证：9 个文件 `grep -c "\[KVC\]"` 全部为 **0**（干净基线）。

## 2. 以 patch 方式应用打印代码（验证 patch 文件正确可用）

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc/patch && ./apply_patches.sh
# 一键完成: Phase0 状态检查 -> Phase1 dry-run 9/9 预检 -> Phase2 9/9 应用 -> Phase3 逐文件计数(合计 113 行) + py_compile
# 回滚: ./revert_patches.sh（patch -R 反向应用, 不依赖备份）
```

应用后逐文件 `[KVC]` 行数（与设计清单一致，合计 40 个打印调用点）：

| 文件 | grep 计数 |
|---|---|
| `v1/request.py` / `v1/core/kv_cache_utils.py` | 2 / 12 |
| `v1/core/block_pool.py` / `kv_cache_manager.py` | 27 / 22 |
| `v1/core/kv_cache_coordinator.py` / `single_type_kv_cache_manager.py` | 16 / 16 |
| `v1/engine/core.py` / `v1/worker/gpu_model_runner.py` | 10 / 4 |
| `vllm_ascend/worker/model_runner_v1.py` | 4 |

`python3 -m py_compile`（9 文件）→ **COMPILE_OK**。结论：patch 文件在"原始代码"上一发命中、无 fuzz、无 reject。9 个 patch 仅含 logger 打印插入 + logger 导入 + 7 处值捕获重写（清单见 `../patch/README.md` §5 第 5 条），无任何其他源码改动；手工逐个 patch 的等价命令见 `../patch/README.md` §0 快速使用。

## 3. 启动期 KVCache 初始化全流程（kvc_startup.log，155 行）

启动命令与就绪标志：

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc && setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model --enforce-eager --tensor-parallel-size 2 --pipeline-parallel-size 2 > ./llama.log 2>&1 < /dev/null &
# 等待 ~100s: grep "Application startup complete" llama.log  -> 1
head -375 llama.log | grep '\[KVC\]' > startup/kvc_startup.log    # 155 行纯 [KVC]
```

分层统计：`CFG 82 + L1 68 + L2 3 + L4 1 + L5 1 = 155`——**配置侧→物理侧→逻辑侧装配全链路打印齐全**。

### 3.1 配置侧（EngineCore，82 行 CFG）

```
INFO [core.py:265] [KVC][CFG] determine_available_memory: 各 worker 可用 KV 显存 = ['51.98GiB', '51.99GiB', '51.94GiB', '51.95GiB']   # ① 测预算(profile_run)
INFO [core.py:278] [KVC][CFG] worker0 KVCacheConfig: num_blocks=13295, groups数=1, tensors数=16          # ③ 编排产物(逐 worker)
INFO [core.py:283] [KVC][CFG]   [0] KVCacheGroupSpec(group_id=0): layers=16 (首层 model.layers.0.self_attn.attn, 末层 model.layers.15.self_attn.attn), is_eagle_group=False
INFO [core.py:288] [KVC][CFG]   [0]   kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, ...)
INFO [core.py:292] [KVC][CFG]   [0]   page_size_bytes=262144 (256.0KB/层/块), storage_block_size=128
INFO [core.py:298] [KVC][CFG]   [0] KVCacheTensor: size=3485204480 bytes (3323.75MiB), shared_by=1 层   # ×16 张
INFO [core.py:315] [KVC][CFG] 最终 scheduler KVCacheConfig: num_blocks=13295 (跨 worker min 对齐), cache_config.num_gpu_blocks=13295, block_size=128
```

主线：① 算规格 → ② 测预算 → ③ `KVCacheSpec → KVCacheGroupSpec → KVCacheTensor → KVCacheConfig` → min 对齐下发。本轮 profile 实测 13295 块（`determine_available_memory` 跨次启动存在 ±1~2 块的测量波动）。

### 3.2 物理侧（4 个 worker 进程，68 行 L1，vllm-ascend 路径）

```
[Worker][model_runner_v1.py:4257] [KVC][L1] _allocate_kv_cache_tensors[dense]: model.layers.0.self_attn.attn: KVCacheTensor(size=3323.75MiB) -> K int8 1661.88MiB + V int8 1661.88MiB (alignment=2097152, device=npu:0~3)   # ×16 层/worker
[Worker][model_runner_v1.py:4695] [KVC][L1] _reshape_kv_cache_tensors: model.layers.0(或16).self_attn.attn (本组 16 层同形) -> K_cache shape=(13295, 128, 4, 128) dtype=torch.bfloat16 / V_cache 同形 (K/V 分离布局)
```

4 worker 设备映射：PP0_TP0→npu:0、PP0_TP1→npu:1、PP1_TP0→npu:2、PP1_TP1→npu:3；`block_id` 索引 K/V 两张张量各自第 0 维。

### 3.3 逻辑侧装配（EngineCore，L2/L4/L5 共 5 行）

```
INFO [kv_cache_utils.py:217] [KVC][L2] FreeKVCacheBlockQueue.__init__: num_free_blocks=13295, 伪头尾哨兵 fake_free_list_head/tail(block_id=-1)
INFO [kv_cache_utils.py:258] [KVC][L2] FreeKVCacheBlockQueue.popleft -> KVCacheBlock(block_id=0), num_free_blocks=13294      # null_block 摘取
INFO [block_pool.py:209] [KVC][L2] BlockPool.__init__: num_gpu_blocks=13295, KVCacheBlock × 13295, cached_block_hash_to_block=BlockHashToBlockMap(size=0), null_block=(0, is_null=True)
INFO [kv_cache_coordinator.py:461] [KVC][L4] UnitaryKVCacheCoordinator.__init__: 单组直通, managers=['FullAttentionManager'], spec block_size=128, page_size_bytes=262144
INFO [kv_cache_manager.py:175] [KVC][L5] KVCacheManager.__init__: coordinator=UnitaryKVCacheCoordinator, num_kv_cache_groups=1, enable_caching=True, max_model_len=8192
```

### 3.4 原生关键行（llama.log 启动段）

```
(APIServer)   INFO [utils.py:1404]   Block size is set to 128 if prefix cache or chunked prefill is enabled.
(EngineCore)  INFO [kv_cache_utils.py:1777] Maximum concurrency for 8,192 tokens per request: 207.73x
(Worker_PP0_TP0) INFO [worker.py:593] Available KV cache memory: 51.98 GiB
(各 Worker) Loading model weights took 3.7454 GB
```

## 4. 运行期：P/R 双请求全流程

请求体 `p/req_cn_p.json` / `r5/req_cn_r5.json` 由 `scripts/gen_cn_requests.py --gen` 生成（与 `2_kvc_cn_curl_case.md` §3 字节级一致）：

```
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @p/req_cn_p.json   > p/resp_cn_p.json
sleep 6
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @r5/req_cn_r5.json > r5/resp_cn_r5.json
```

### 4.1 P：缓冲 2 块（kvc_cn_p.log，33 行）

```
INFO [request.py:185] [KVC][ENQ] Request(...) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['dcaeded48257', '8c4d2a2ea88b']
INFO [kv_cache_coordinator.py:476] [KVC][L4] find_longest_cache_hit: 满块hash数=2, max_cache_hit_length=323
INFO [block_pool.py:84] [KVC][L2] get_one_block: key=(hash=dcaeded48257, group_id=0) -> MISS
INFO [single_type_kv_cache_manager.py:598] [KVC][L3]   第 1 块 MISS: BlockHash=dcaeded48257 -> break                        # 冷缓存, 首块断链
INFO [kv_cache_coordinator.py:494] [KVC][L4] find_longest_cache_hit 返回: hit_blocks=[[]], hit_length=0
INFO [kv_cache_manager.py:431] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13294 块
INFO [block_pool.py:411] [KVC][L2] BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=13291
INFO [block_pool.py:108] [KVC][L2] insert: (hash=dcaeded48257) <- KVCacheBlock(block_id=1), map size=1    # 满块1 入表
INFO [block_pool.py:108] [KVC][L2] insert: (hash=8c4d2a2ea88b) <- KVCacheBlock(block_id=2), map size=2    # 满块2 入表
INFO [block_pool.py:340] [KVC][L2] cache_full_blocks: 新满块 2 块 block_ids=[1, 2] 入表 (num_cached_blocks 0 -> 2)    # 缓冲 2 块!
INFO [kv_cache_manager.py:495] [KVC][L5] allocate_slots 返回: KVCacheBlocks(blocks=([1, 2, 3],)), block_table=([1, 2, 3],)
INFO [kv_cache_manager.py:511] [KVC][L5] free: 释放前持有 block_table=([1, 2, 3],)
INFO [block_pool.py:516] [KVC][L2] free_blocks: [(3,0),(2,0),(1,0)] 归零回收 3 块 [3, 2, 1], append_n -> 队尾
```

尾块 3（68/128）未满不入表；释放后 1/2/3 挂队尾带哈希，缓存表留存 2 个 hash。

### 4.2 R：五块生命周期（kvc_cn_r5.log，355 行）

```
① 复用 2 块 (第 3 hash MISS 断链):
INFO [request.py:185] [KVC][ENQ] Request(...) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 × 3: ['dcaeded48257', '8c4d2a2ea88b', '2807a32a7a28']
INFO [kv_cache_coordinator.py:476] [KVC][L4] find_longest_cache_hit: 满块hash数=3, max_cache_hit_length=485
INFO [block_pool.py:70] [KVC][L2] get_one_block: (hash=dcaeded48257) -> HIT KVCacheBlock(block_id=1)
INFO [single_type_kv_cache_manager.py:590] [KVC][L3]   第 1 块 HIT: BlockHash=dcaeded48257 -> cached blocks=[1]
INFO [block_pool.py:70] [KVC][L2] get_one_block: (hash=8c4d2a2ea88b) -> HIT KVCacheBlock(block_id=2)
INFO [single_type_kv_cache_manager.py:590] [KVC][L3]   第 2 块 HIT: BlockHash=8c4d2a2ea88b -> cached blocks=[2]
INFO [block_pool.py:84] [KVC][L2] get_one_block: (hash=2807a32a7a28) -> MISS
INFO [single_type_kv_cache_manager.py:598] [KVC][L3]   第 3 块 MISS: BlockHash=2807a32a7a28 -> break                       # P 只种了前 2 块, 中间断链
INFO [kv_cache_coordinator.py:494] [KVC][L4] find_longest_cache_hit 返回: hit_blocks=[[1, 2]], hit_length=256
INFO [block_pool.py:491] [KVC][L2] BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)              # 零拷贝共享
② prefill 新申请 2 块 (1 满 + 1 尾):
INFO [kv_cache_manager.py:431] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 4 块 vs 可用 13294 块            # S1 报总需求 cdiv(486,128)=4(含待 touch 2)
INFO [block_pool.py:411] [KVC][L4] S3... [KVC][L2] BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13290   # 实际只新弹 2 块
INFO [block_pool.py:108] [KVC][L2] insert: (hash=2807a32a7a28) <- KVCacheBlock(block_id=4), map size=3   # 块4 追问句恰填满
INFO [block_pool.py:340] [KVC][L2] cache_full_blocks: 新满块 1 块 block_ids=[4] 入表 (num_cached_blocks 2 -> 3)     # 块5 (102/128) 未满不入
INFO [kv_cache_manager.py:495] [KVC][L5] allocate_slots 返回: KVCacheBlocks(blocks=([4, 5],)), block_table=([1, 2, 4, 5],)
decode 步 1~26: S1 恒 "需分配 0 块", 尾块 102 → 128
③ decode 填满 + 步 27 跨界申请第 5 块:
INFO [kv_cache_coordinator.py:188] [KVC][L4] get_num_blocks_to_allocate: num_tokens=513 -> 需分配 1 块             # cdiv(513,128)-4 = 1
INFO [kv_cache_manager.py:431] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 1 块 vs 可用 13290 块
INFO [block_pool.py:411] [KVC][L2] BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=13289  # 第 5 块!
INFO [block_pool.py:108] [KVC][L2] insert: (hash=4324a28794de) <- KVCacheBlock(block_id=5), map size=4   # 刚满的块5 与跨界申请合并发生在步 27
decode 步 28~34: 块 6 装 8/128 未满不入表 (第 35 个输出仅采样)
结束释放 (五块逆序):
INFO [kv_cache_manager.py:511] [KVC][L5] free: 释放前持有 block_table=([1, 2, 4, 5, 6],)                        # 2 复用 + prefill 2 + decode 1
INFO [block_pool.py:516] [KVC][L2] free_blocks: [(6,0),(5,0),(4,0),(2,0),(1,0)] 归零回收 5 块 [6,5,4,2,1], append_n -> 队尾
```

### 4.3 响应核对

| 请求 | completion_tokens | finish_reason |
|---|---|---|
| P | 1 | length |
| R | 35 | length |

## 5. 本轮实证结论

1. **patch 文件正确可用**：9 个 patch 在原始代码上 dry-run/apply 一发命中、计数/编译/运行三重验证通过——`../patch/README.md` 的应用与回滚方法成立。
2. **启动期五层全流程**：155 行 [KVC] 完整覆盖"配置侧（82）→ 物理侧（68）→ 逻辑侧装配（5）"，block_size=128、num_blocks=13295、K/V 分离 `(13295,128,4,128)×2` bf16、可用 KV 显存 51.98GiB。
3. **运行期全流程**：P 一次演示"断链种块"（33 行），R 一次演示"复用 2 + prefill 2（1满1尾）+ decode 填满跨界"完整五块生命周期（355 行）——`2_kvc_cn_curl_case.md` 的全部设计论断均在本轮拿到直接日志证据。

## 6. 本轮产物（容器与本地 `kvc/` 同步；目录结构见 `../README.md`）

| 产物（相对 `kvc/` 根） | 说明 |
|---|---|
| `startup/kvc_startup.log`（155 行） | **启动期 KVCache 初始化全流程**（155 行纯 [KVC] 口径，原生行见 §3.4） |
| `p/kvc_cn_p.log`（33 行）/ `r5/kvc_cn_r5.log`（355 行） | P / R 运行期全流程轨迹（本轮 patch 版实测） |
| `llama.log`（774 行） | 本轮服务全量日志（启动 + 双请求） |
| `p/req_cn_p.json`、`r5/req_cn_r5.json` | 请求体 |
| `p/resp_cn_p.json`、`r5/resp_cn_r5.json` | 响应体 |
| `p/p_run_start.txt` / `r5/r_run_start.txt` | 双请求在 llama.log 中的起始行 |

> 容器可读化命令：`grep '\[KVC\]' llama.log`；权威数据以本文档与上表文件为准（同一 token 序列的哈希链在单次服务进程内完全一致，服务重启后因 NONE_HASH 种子随机而变化——见 `2_kvc_cn_curl_case.md` §7 注 4）。
