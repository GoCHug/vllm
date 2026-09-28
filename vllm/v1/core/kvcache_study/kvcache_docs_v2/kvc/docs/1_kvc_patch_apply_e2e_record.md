# 端到端实录：还原 → patch 应用验证 → 启动期 KVCache 初始化全流程 → P/R 运行期全流程（S1 子步横幅先行 + KVP 每层一行与 KV 布局说明）

> 本文是 `2_kvc_cn_curl_case.md`（用例）与 `patch/`（打印补丁）的**端到端正式验证记录**：从干净源码出发，以 patch 方式注入 94 处 `[KVC]` 打印（grep 计数 167 行，含注释行；本轮升级：①`--- S1: 容量检查 ---` 子步横幅**上移至 full-fit 预检前**——两次外层 S1 容量探问的 L4 下钻全部落在子步横幅之后；②KVP 改为**每层一行**（块内联 + 层统计），且概览行新增**一次性 KV 布局说明**——K_cache 与 V_cache 为两个独立张量池（张量级拆分，**不是最后一维拼接**），每块每层 shape(bsz, kv_heads, head_dim)），记录一次完整的服务启动初始化与 P/R 双请求生命周期。实测 2026-09-28（log 内 09-28 06:54，容器时钟 UTC-8）。
>
> 环境：gggtest（PP2TP2 4 卡 Ascend910，pod 当日 Running）；vllm 0.23.0（`/vllm-workspace/vllm`）+ vllm-ascend 0.23.0（`/vllm-workspace/vllm-ascend`）。进程：APIServer pid=1711、EngineCore pid=1749、Worker pid=1783~1786（PP0_TP0/PP0_TP1/PP1_TP0/PP1_TP1）。

## 1. 还原源码至原始状态

```bash
cd /vllm-workspace/vllm && git checkout -- vllm/v1/request.py vllm/v1/core/kv_cache_utils.py vllm/v1/core/block_pool.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/engine/core.py vllm/v1/worker/gpu_model_runner.py
cd /vllm-workspace/vllm-ascend && git checkout -- vllm_ascend/worker/model_runner_v1.py
```

验证：9 个文件 `grep -c "\[KVC\]"` 全部为 **0**（干净基线）。本轮结束时同样核验：revert 归零 + 两仓库 0 改动 + 0 进程 + .orig 清理。

## 2. 以 patch 方式应用打印代码（验证 patch 文件正确可用）

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
# Phase0 状态检查 -> Phase1 dry-run 9/9 预检 -> Phase2 9/9 应用 -> Phase3 逐文件计数(合计 167 行) + py_compile
```

应用后逐文件 `[KVC]` grep 计数（含注释行；合计 167 行 = 94 个 `logger.info` 打印调用点）：

| 文件 | grep 计数 | 打印调用点 |
|---|---|---|
| `v1/request.py` / `v1/core/kv_cache_utils.py` | 5 / 12 | 3 / 6 |
| `v1/core/block_pool.py` / `kv_cache_manager.py` | 27 / **56** | 14 / **32**（调用点数不变; S1 横幅上移净增 1 注释行） |
| `v1/core/kv_cache_coordinator.py` / `single_type_kv_cache_manager.py` | 17 / 18 | 9 / 9 |
| `v1/engine/core.py` / `v1/worker/gpu_model_runner.py` | 13 / 4 | 9 / 2 |
| `vllm_ascend/worker/model_runner_v1.py` | 15 | 9（KVP 头横幅/概览/逐层行循环/尾横幅） |

py_compile（9 文件）→ **COMPILE_OK**。改动补丁：04（12 hunk）、09（6 hunk）。

**本轮升级（相对上一轮的两个变化）**：

| 变化 | 旧行为（第六轮） | 新行为（实测） |
|---|---|---|
| S1 子步横幅位置 | `--- S1: 容量检查---` 在两次外层探问**之后**打（探问下钻游离在子步横幅外） | **上移到 full-fit 预检前（:402）**：总横幅(:393) → 进入(:395) → **S1 子步横幅(:402)** → 外层探问 ×2(:407/:434 下钻) → S1 汇总值(:447) |
| KVP 输出格式 | 每层每块一行（前 10 值 + 块级统计；R 320 行、P 192 行） | **每层一行**（块内联 + 层统计；固定 64 行/请求）+ 概览行含 **KV 布局一次性说明**（张量级拆分、非最后一维拼接） |

## 3. 启动期 KVCache 初始化全流程（log/kvc_startup.log，168 行）

分层统计：`CFG 84 + L1 76 + 逻辑侧装配 8 = 168`（本轮未动启动期打印）。

### 3.1 配置侧（EngineCore，84 行 CFG，含首尾横幅）

```
INFO [core.py:265] [KVC][CFG] ================ 配置侧 KVCache 编排开始 ================
INFO [core.py:267] [KVC][CFG] determine_available_memory: 各 worker 可用 KV 显存 = ['51.98GiB', '51.99GiB', '51.94GiB', '51.95GiB']
INFO [core.py:280] [KVC][CFG] worker0 KVCacheConfig: num_blocks=13295, groups数=1, tensors数=16          # 逐 worker ×4
INFO [core.py:300] [KVC][CFG]   [0] KVCacheTensor: size=3485204480 bytes (3323.75MiB), shared_by=1 层 (model.layers.0.self_attn.attn)
INFO [core.py:317] [KVC][CFG] 最终 scheduler KVCacheConfig: num_blocks=13295 (跨 worker min 对齐), cache_config.num_gpu_blocks=13295, block_size=128
INFO [core.py:322] [KVC][CFG] ================ 配置侧 KVCache 编排完成 ================
```

### 3.2 物理侧（4 worker，76 行 L1，vllm-ascend，含首尾横幅）

```
[Worker][model_runner_v1.py:4264] [KVC][L1] ================ 物理侧 KV Cache 分配开始 ================   # ×4 worker
[Worker][model_runner_v1.py:4426] [KVC][L1] ...K int8 1661.88MiB + V int8 1661.88MiB (alignment=2097152, device=npu:0~3)   # ×16 层/worker
[Worker][model_runner_v1.py:4864] [KVC][L1] ...K_cache shape=(13295, 128, 4, 128) dtype=torch.bfloat16 / V_cache 同形 (K/V 分离布局)
[Worker][model_runner_v1.py:4915] [KVC][L1] ================ 物理侧 KV Cache 分配完成 ================   # ×4 worker
```

### 3.3 逻辑侧装配（EngineCore，8 行，含 L3 init）

```
INFO [kv_cache_manager.py:143] [KVC][L5] ================ 逻辑侧初始化开始 ================
INFO [kv_cache_utils.py:217] ...FreeKVCacheBlockQueue.__init__: num_free_blocks=13295, 伪头尾哨兵...
INFO [kv_cache_utils.py:258] ...popleft -> KVCacheBlock(block_id=0), num_free_blocks=13294
INFO [block_pool.py:209] ...BlockPool.__init__: num_gpu_blocks=13295, null_block=(0, is_null=True), enable_caching=True
INFO [single_type_kv_cache_manager.py:95] ...FullAttentionManager.__init__: spec=FullAttentionSpec(block_size=128)...
INFO [kv_cache_coordinator.py:462] ...UnitaryKVCacheCoordinator.__init__: 单组直通...
INFO [kv_cache_manager.py:178] ...KVCacheManager.__init__: coordinator=UnitaryKVCacheCoordinator, num_kv_cache_groups=1...
INFO [kv_cache_manager.py:186] [KVC][L5] ================ 逻辑侧初始化完成 ================
```

### 3.4 原生关键行

```
(APIServer pid=1711)  INFO [utils.py:1404]  Block size is set to 128       # llama.log:30
(EngineCore pid=1749) INFO [kv_cache_utils.py:1777] Maximum concurrency for 8,192 tokens per request: 207.73x
(Worker_PP0_TP0 pid=1783) INFO [worker.py:593] Available KV cache memory: 51.98 GiB
(APIServer pid=1711) INFO: Application startup complete.    # llama.log:388, 就绪 60s
```

## 4. 运行期：P/R 双请求全流程

### 4.1 P：缓冲 2 块（log/kvc_p.log，124 行）

分配链 S1 顺序（本轮核心修复点）实测：

```
INFO [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========          ← 总横幅
INFO [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-a58429f021d08c49-0-9c31f51a, num_new_tokens=324, ...
INFO [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---                    ← 本轮: 子步横幅先行!
INFO [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: ... 需分配 3 块   ← 探问 #1 落在横幅后 ✓
INFO [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: ... 需分配 3 块   ← 探问 #2 ✓
INFO [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13294 块   ← 汇总值收尾
INFO [kv_cache_manager.py:481] --- S2: 无前缀缓冲, 无需 touch ---
INFO [kv_cache_manager.py:485] --- S3: 新块分配 (横幅先行) → [L2] S3 BlockPool.get_new_blocks(3) -> [1, 2, 3], 剩余 13291
INFO [kv_cache_manager.py:523] --- S4: 满块入缓存 → [L2] S4 insert dc1b17e68cb7<-1, 0cd9eb7d9f2e<-2
INFO [kv_cache_manager.py:533/539] 分配...返回 block_table=([1, 2, 3],) / ======== 分配完成 ========
[KVP:2518] ======== 请求结束 KVCache 即将释放, 开始打印该请求物理 cache ========  (TERM ×4 卡)
[KVP:2520] TERM req=cmpl-a584...0-9c31f51a dev=npu:x 逐层按块: layers=16 blocks=[1, 2, 3] region=324/324 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
[KVP:2563] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=3[未满:68](68,4,128) | K示(首块首token前3)=[-0.008057, 0.1035, 0.02747] 统计[n=165888] mean=0.04895 std=1.508 min=-14.56 max=8.5 | V示(首块首token前3)=[0.0009766, 0.000576, -0.0008163] 统计[n=165888] ...
... L01~L15 同构 (每 worker 16 行) ...
[KVP:2570] ======== 请求结束, 物理 cache 打印完毕 ========
INFO [kv_cache_manager.py:676/677/683] 调度提交(非分配 S4) / 提交 cache_blocks: num_computed_tokens=324 / 提交完成   (P=1 次)
INFO [kv_cache_manager.py:551/553] 释放 ... / [L2] 释放 free_blocks 归零回收 [3, 2, 1]
```

层统计交叉验证：P n=165888 = 324 tok × 4 kv_heads × 128 head_dim —— 正是 region×(kv_heads×head_dim) 的精确展开。

### 4.2 R：五块生命周期（log/kvc_r5.log，752 行）

阶段计数实测：
- `调度提交(非分配 S4)`：**35 次**（async_scheduler 每步输出后各一次）
- S1 探问值分布（汇总行）：**33× 需分配 0 块** + **1× 需分配 1 块**（步 27 跨界，vs 可用 13290）+ **1× 需分配 4 块**（prefill）
- 块分配：`S3 get_new_blocks(2) -> [4, 5]` 剩余 13290、跨界 `get_new_blocks(1) -> [6]` 剩余 13289
- 哈希 insert 链：`dc1b17e68cb7→1`、`0cd9eb7d9f2e→2`、`aee282abd7e4→4`（prefill 满）、`9430724f6da3→5`（decode 步 27 填满）
- 释放：`[6, 5, 4, 2, 1]`，`num_free_blocks=13294`
- KVP：概览 4 条 + 层行 **64 条**（4 worker × 16 层固定）；R 层统计 n=266240 = 520 tok × 4 × 128

**TERM 层行实测样本**（R，region=520/520，blocks=[1,2,4,5,6]）：

```
[KVP:2563] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=4[满:128](128,4,128) blk=5[满:128](128,4,128) blk=6[未满:8](8,4,128) | K示(首块首token前3)=[0.5078, 0.9336, 0.9219] 统计[n=266240] mean=-0.0149 std=1.402 min=-10.38 max=10.81 | V示(首块首token前3)=[...] 统计[n=266240] ...
```

### 4.3 本轮升级的形式语义

| 输出形态 | 触发 | 行数 | 关键行 |
|---|---|---|---|
| 分配链 S1 段 | allocate_slots 开头 | 6 行 | :393 总横幅 → :395 进入 → **:402 S1 子步横幅** → 探问×2(:188) → :447 汇总值 |
| KVP TERM(每卡) | 请求终态 | **18 行/卡**（1 头 + 1 概览 + 16 层行 + 1 尾），全请求固定 | :2518/:2520/:2563/:2570 |
| 调度提交 | async 每步输出后 | 3 行 | :676/:677/:683 |

### 4.4 响应核对

| 请求 | completion_tokens | finish_reason |
|---|---|---|
| P | 1（"为了"） | length |
| R | 35 | length |

## 5. 本轮实证结论

1. **patch 167 行验证通过**：9/9 应用、94 调用点、py_compile OK；本地回环（stash→apply→revert→stash pop）与容器双验证。
2. **S1 顺序完整修复**：子步横幅先行（:402）→ 两次外层探问下钻（coordinator:188 ×2）→ 汇总值（:447）——S1 语义日志全部在子步横幅之内，顺序自洽（用户"S1: 容量检查顺序不对"根治）。
3. **KVP 可读性重构**：每请求 KVP 行数从 ~390 行降至 **76 行**（4 卡 × 18）；每层一行内联块标注（`blk=N[满:128](128,4,128)` / `blk=6[未满:8](8,4,128)`）+ K/V 首 3 值示意 + 层合并统计——一眼可读。
4. **KV 布局澄清**：用户问"最后一维拆 kv"——实测说明 K/V 是**张量级拆分**（K_cache 与 V_cache 两个独立池），不是最后一维拼接；概览行的布局说明给出了每维含义（第1维=token 槽位、第2维=kv_heads=8/TP2、最后一维=head_dim=128），并以层统计 n=region×4×128 精确交叉验证。
5. **容器回收闭环**：服务已杀（0 进程）、9 补丁 revert 归零、两仓库 git 0 改动、.orig 清理。

## 6. 本轮产物

| 产物（`log/`） | 说明 |
|---|---|
| `log/llama.log`（1273 行） | 服务全量日志（启动 :1~388 + P :389~517 + R :518~1273） |
| `log/kvc_startup.log`（168 行） | 启动期 [KVC] 拆解轨迹 |
| `log/kvc_p.log`（124 行）/ `log/kvc_r5.log`（752 行） | P / R 运行期 [KVC] 拆解轨迹 |
| `log/req_*.json`、`resp_*.json`、curl_*screen.txt | 请求体 / 响应体 / 打屏实录 |
| `log/p_run_start.txt` / `r_run_start.txt` | 双请求分界（:389 / :518） |