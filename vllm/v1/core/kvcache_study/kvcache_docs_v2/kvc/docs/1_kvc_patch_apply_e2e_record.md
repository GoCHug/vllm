# 端到端实录：还原 → patch 应用验证 → 启动期 KVCache 初始化全流程 → P/R 运行期全流程

> 本文是 `2_kvc_cn_curl_case.md`（用例）与 `patch/`（打印补丁）的**端到端正式验证记录**：从干净源码出发，以 patch 方式注入 92 处 `[KVC]` 打印（grep 计数 163 行，含注释行），记录一次完整的服务启动初始化与 P/R 双请求生命周期。实测 2026-09-28（log 内 09-28 07:46，容器时钟 UTC-8）。
>
> 打印体系核心设计：
> 1. **分配 S1 段自洽**——总横幅 → 进入 → `--- S1: 容量检查 ---` 子步横幅 → 两次外层容量探问（L4 下钻）→ S1 汇总值
> 2. **S1~S4 四子步无条件打印**——无新块步也打全四段（需分配 0 / 无前缀无需 touch / 无需分配新块 / 满块缓存维护），结构完整闭合
> 3. **阶段前缀全显**——每条下钻消息开头标注所属阶段（S1/S2/S3/S4/前缀查找/入队/释放/分配/提交），单条日志可定位
> 4. **调度提交独立包裹**——async_scheduler 每步输出后的 `cache_blocks` 提交路径单列 `调度提交(非分配 S4)` 横幅对
> 5. **KVP 仅请求结束打印（TERM/LATE）**——概览行含一次性 KV 布局说明（K_cache/V_cache 两个独立张量池=张量级拆分，非最后一维拼接）；逐层单行内联块（`blk=N[满:bsz|未满:cov](cov,kv_heads,head_dim)`）+ K/V 首块首 token 前 3 值示意 + 层合并统计
>
> 环境：gggtest（PP2TP2 4 卡 Ascend910，pod 当日 Running）；vllm 0.23.0（`/vllm-workspace/vllm`）+ vllm-ascend 0.23.0（`/vllm-workspace/vllm-ascend`）。进程：APIServer pid=4555、EngineCore pid=4600、Worker pid=4633~4636（PP0_TP0/PP0_TP1/PP1_TP0/PP1_TP1）。

## 1. 还原源码至原始状态

```bash
cd /vllm-workspace/vllm && git checkout -- vllm/v1/request.py vllm/v1/core/kv_cache_utils.py vllm/v1/core/block_pool.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/engine/core.py
cd /vllm-workspace/vllm-ascend && git checkout -- vllm_ascend/worker/model_runner_v1.py
```

验证：8 个文件 `grep -c "\[KVC\]"` 全部为 **0**（干净基线）。实验结束时同样核验：revert 归零 + 两仓库 0 改动 + 0 进程 + .orig 清理。

## 2. 以 patch 方式应用打印代码（验证 patch 文件正确可用）

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
# Phase0 状态检查 -> Phase1 dry-run 8/8 预检 -> Phase2 8/8 应用 -> Phase3 逐文件计数(合计 163 行) + py_compile
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

## 3. 启动期 KVCache 初始化全流程（log/kvc_startup.log，168 行）

分层统计：`CFG 84 + L1 76 + 逻辑侧装配 8 = 168`。

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
[Worker][model_runner_v1.py:4426] [KVC][L1] ...KVCacheTensor(size=3485204480 bytes = 3323.75MiB) -> K int8 1661.88MiB + V int8 1661.88MiB (alignment=2097152, device=npu:0~3)   # ×16 层/worker
[Worker][model_runner_v1.py:4864] [KVC][L1] ...K_cache shape=(13295, 128, 4, 128) dtype=torch.bfloat16 / V_cache 同形 (K/V 分离布局: 每层两张量池, block id 即 dim0 行号)
[Worker][model_runner_v1.py:4915] [KVC][L1] ================ 物理侧 KV Cache 分配完成 ================   # ×4 worker
```

KV 布局要点：vllm-ascend 将每层 KVCacheTensor（3323.75MiB）**拆成 K、V 两个独立的 int8 张量池**（各 1661.88MiB、2MiB 地址对齐，支持 PD 分离），叠加视角等价于上游 GPU FlashAttention 后端的单张量 `(num_blocks, 2, block_size, num_kv_heads, head_size)`；每块每层 K_cache[blk] 与 V_cache[blk] 各 `(128, 4, 128)`。

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
(APIServer pid=4555)  INFO [utils.py:1404]  Block size is set to 128       # llama.log:30
(EngineCore pid=4600) INFO [kv_cache_utils.py:1777] Maximum concurrency for 8,192 tokens per request: 207.73x
(Worker_PP0_TP0 pid=4633) INFO [worker.py:593] Available KV cache memory: 51.98 GiB
(APIServer pid=4555) INFO: Application startup complete.    # llama.log:388, 就绪 55s
```

## 4. 运行期：P/R 双请求全流程

### 4.1 P：缓冲 2 块（log/kvc_p.log，124 行）

```
INFO [request.py:187] [KVC][ENQ] Request(request_id=cmpl-addf477dedca5a18-0-a8f278a5) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['df3b74831f54', '5751b0a5469a']
INFO [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========          ← 总横幅
INFO [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: ...
INFO [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---                    ← 子步横幅先行
INFO [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: ... 需分配 3 块   ← 探问 #1 落在子步横幅后
INFO [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: ... 需分配 3 块   ← 探问 #2
INFO [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13294 块   ← 汇总值收尾
INFO [kv_cache_manager.py:481] --- S2: 无前缀缓冲, 无需 touch ---
INFO [kv_cache_manager.py:485] --- S3: 新块分配 (横幅先行) → [L2] S3 BlockPool.get_new_blocks(3) -> [1, 2, 3], 剩余 13291
INFO [kv_cache_manager.py:523] --- S4: 满块入缓存 → [L2] S4 insert df3b74831f54<-1, 5751b0a5469a<-2
INFO [kv_cache_manager.py:533/539] 分配...返回 block_table=([1, 2, 3],) / ======== 分配完成 ========
[KVP:2518] ======== 请求结束 KVCache 即将释放, 开始打印该请求物理 cache ========  (TERM ×4 卡)
[KVP:2520] TERM req=cmpl-addf4...0-a8f278a5 dev=npu:0 逐层按块: layers=16 blocks=[1, 2, 3] region=324/324 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
[KVP:2563] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=3[未满:68](68,4,128) | K示(首块首token前3)=[...] 统计[n=165888] mean=... std=... min=... max=... | V示(首块首token前3)=[...] 统计[n=165888] ...
   (每 worker 16 行层行; n=165888 = 324 tok × 4 kv_heads × 128 head_dim 精确闭合)
[KVP:2570] ======== 请求结束, 物理 cache 打印完毕 ========
INFO [kv_cache_manager.py:676/677/683] ======== 调度提交(非分配 S4) ======== / 提交 cache_blocks: num_computed_tokens=324 / 提交完成   (P=1 次)
INFO [kv_cache_manager.py:551/553] 释放 ... / [L2] 释放 free_blocks 归零回收 [3, 2, 1]
```

### 4.2 R：五块生命周期（log/kvc_r5.log，752 行）

阶段计数实测：
- `调度提交(非分配 S4)`：**35 次**（async_scheduler 每步输出后各一次）
- S1 汇总值分布：**33× 需分配 0 块** + **1× 需分配 1 块**（步 27 跨界，vs 可用 13290）+ **1× 需分配 4 块**（prefill）
- 块分配：`S3 get_new_blocks(2) -> [4, 5]` 剩余 13290、跨界 `get_new_blocks(1) -> [6]` 剩余 13289
- 哈希 insert 链：`df3b74831f54→1`、`5751b0a5469a→2`、`3d788bda3932→4`（prefill 满）、`8529e6691553→5`（decode 步 27 填满）
- 释放：`[6, 5, 4, 2, 1]`，`num_free_blocks=13294`
- KVP：概览 4 条 + 层行 **64 条**（4 worker × 16 层固定）；层统计 n=266240 = 520 tok × 4 × 128

**TERM 层行实测样本**（R，region=520/520，blocks=[1,2,4,5,6]）：

```
[KVP:2563] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=4[满:128](128,4,128) blk=5[满:128](128,4,128) blk=6[未满:8](8,4,128) | K示(首块首token前3)=[...] 统计[n=266240] mean=-0.0149 std=1.402 min=... max=... | V示(首块首token前3)=[...] 统计[n=266240] ...
```

### 4.3 输出形态语义

| 输出形态 | 触发 | 行数 | 关键行 |
|---|---|---|---|
| 分配链 S1 段 | allocate_slots 开头 | 6 行 | :393 总横幅 → :395 进入 → :402 S1 子步横幅 → 探问×2(:188) → :447 汇总值 |
| 无块步闭合链 | decode 无新块步 | ~14 行 | S1 需分配 0 → S2 无前缀 → S3 无需分配(空下钻+块未满) → S4 维护 → 返回/完成 |
| KVP TERM(每卡) | 请求终态 | **18 行/卡**（1 头 + 1 概览 + 16 层行 + 1 尾） | :2518/:2520/:2563/:2570 |
| 调度提交 | async 每步输出后 | 3 行 | :676/:677/:683 |

### 4.4 响应核对

| 请求 | completion_tokens | finish_reason |
|---|---|---|
| P | 1（"为了"） | length |
| R | 35 | length |

## 5. 实证结论

1. **patch 163 行验证通过**：8/8 应用、92 调用点、py_compile OK；本地回环（stash→apply→revert→stash pop）与容器 apply/revert 双向验证。
2. **S1 段完整自洽**：子步横幅先行（:402）→ 两次外层探问下钻（coordinator:188 ×2）→ 汇总值（:447）——S1 语义日志全部在子步横幅之内。
3. **S1~S4 全子步可观测**：无块步打出完整四段子步（S1 需分配 0 / S2 无前缀 / S3 无需分配 / S4 维护），R 33 个无块步全部闭合。
4. **KVP 可读性**：每请求 KVP 固定 76 行（4 卡 × 18）；每层一行内联块标注 + K/V 首 3 值示意 + 层合并统计——一眼可读。
5. **KV 布局实证**：K/V 为张量级拆分的两个独立池（非最后一维拼接）；层统计 `n = region × kv_heads(4) × head_dim(128)` 精确断言（P: 165888、R: 266240）；4 卡各持不同层段与 kv_heads 切片。
6. **容器回收闭环**：服务已杀（0 进程）、9 补丁 revert 归零、两仓库 git 0 改动、.orig 清理。

## 6. 实测产物

| 产物（`log/`） | 说明 |
|---|---|
| `log/llama.log`（1274 行） | 服务全量日志（启动 :1~388 + P :389~518 + R :519~1274） |
| `log/kvc_startup.log`（168 行） | 启动期 [KVC] 拆解轨迹 |
| `log/kvc_p.log`（124 行）/ `log/kvc_r5.log`（752 行） | P / R 运行期 [KVC] 拆解轨迹 |
| `log/req_*.json`、`resp_*.json`、curl_*screen.txt | 请求体 / 响应体 / 打屏实录 |
| `log/p_run_start.txt` / `r_run_start.txt` | 双请求分界（:389 / :519） |