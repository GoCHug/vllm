# KVCache 调试打印体系实验（全生命周期设计 + 端到端实测）

> 本文是 `2_kvc_cn_curl_case.md`（用例）与 `patch/`（打印补丁）的**端到端正式验证记录**：从干净源码出发，以 patch 方式注入 95 处 `[KVC]` 打印（grep 计数 168 行，含注释行），记录一次完整的服务启动初始化与 P/R 双请求生命周期。§2~§4 的日志引文均**逐字原样**取自本次实测的拆解轨迹（`log/kvc_startup.log` / `kvc_p.log` / `kvc_r.log`，已剥离进程前缀；带 `(Worker pid=…)` 前缀的完整版见 `log/llama-3-8b.log`）。

## 0. 总览

### 0.1 打印体系全生命周期设计

（95 处 `[KVC]` 打印 / 168 行，三级横幅 + 阶段前缀贯穿）

**一、启动初始化（一次性，172 行 [KVC]，三段各带 `================` 开始/完成横幅）**

| 段 | 层标签 | 行数 | 核心内容 |
|---|---|---|---|
| 配置侧 | CFG | 85 | **① 算规格（紧跟 get_kv_cache_specs() 调用，本轮前移至此）** → ② 各 worker 可用 KV 显存 → 逐 worker `KVCacheConfig`（num_blocks/组数/张量数）→ 逐张量 size/shared_by → 最终 scheduler 侧 min 对齐 |
| 物理侧 | L1 | 76 | 每层 KVCacheTensor 拆 **K int8 池 + V int8 池**两张独立张量（2MiB 对齐）；reshape 后 K_cache=V_cache=(num_blocks, 128, 4, 128) bf16，block id 即 dim0 行号 |
| 逻辑侧 | L2~L5 | 8 | 自底向上逐组件 `__init__完成：`——L2 空闲队列（伪头尾哨兵）/ L2 BlockHashToBlockMap / L2 BlockPool（null 块摘取）/ L3 manager / L4 单组直通 coordinator / L5 门面 |

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
| 模型 | Meta-Llama-3-8B（`modelhub_74000048_meta-llama-3-8b-148700128_20260921221233`，32 层 / kv_heads 8 / TP2 下本地 4 / block_size 128）https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B |
| 软件栈 | vllm 0.23.0 + vllm-ascend 0.23.0（`/vllm-workspace/`） |
| 进程 | APIServer pid=1008 · EngineCore pid=1047 · Worker pid=1077~1080（PP0_TP0 / PP0_TP1 / PP1_TP0 / PP1_TP1） |
| 实测时间 | 2026-09-29 02:27~02:30（log 内时间戳，容器时钟 UTC-8） |

## 1. 实验流程

### 1.1 起容器

```bash
# pod 状态确认（平台空闲会回收为 Stopped，需 itask start 拉起）
itask list | grep gggtest            # 期望 Running
itask start gggtest                  # 若 Stopped 时拉起
# SSH 隧道（本地 5557 -> 容器 7890）；pod 重建后 ssh host key 变化需先清除
itask ssh-tunnel gggtest --port 5557 --user gch02599191
ssh-keygen -f ~/.ssh/known_hosts -R "[localhost]:5557"   # pod 重建后
ssh -p 5557 root@localhost "hostname; whoami"            # 连通确认
```

源码干净基线核验（8 个文件 `grep -c "\[KVC\]"` 全部为 **0**、两仓库 `git status` 0 改动）。容器若经平台重启，可写层自动还原，此项天然满足。

### 1.2 打 patch

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
# Phase0 已应用检测 -> Phase1 dry-run 8/8 预检 -> Phase2 8/8 应用 -> Phase3 逐文件计数(合计 168 行) + py_compile
```

- **168 行**——apply 脚本 Phase 3 用 `grep -c "\[KVC\]"` 逐文件统计、并与预期值比对的口径。它由两部分构成：**73 行注释**（`# [KVC][L5] …` 设计说明，不产生日志）+ **95 个打印语句**（`logger.info(…)` 调用首行；多行调用的 f-string 续行不带 `[KVC]` 标记，不重复计数）。
- **92 个调用点**——真正会在运行时输出日志的 `logger.info` 语句数。

| 补丁 | 文件 | grep 行数（含注释） | 其中注释 | 打印调用点 |
|---|---|---|---|---|
| 01 | `vllm/v1/request.py` | 5 | 2 | 3 |
| 02 | `v1/core/kv_cache_utils.py` | 10 | 5 | 5 |
| 03 | `v1/core/block_pool.py` | 29 | 14 | 15 |
| 04 | `v1/core/kv_cache_manager.py` | **56** | 24 | **32** |
| 05 | `v1/core/kv_cache_coordinator.py` | 17 | 8 | 9 |
| 06 | `v1/core/single_type_kv_cache_manager.py` | 18 | 9 | 9 |
| 07 | `v1/engine/core.py` | 18 | 5 | 13 |
| 08 | `vllm_ascend/worker/model_runner_v1.py` | 15 | 6 | 9 |
| **合计** | 8 文件 | **165** | **73** | **95** |

> **本轮（07 v2）变更**：`07_vllm_v1_engine_core.py.patch` 的 CFG 编排开始横幅 + **① 算规格打印**（每 worker 的 spec 摘要 `_spec_parts`）从 `determine_available_memory` 之后的 assert 处**前移**到 `model_executor.get_kv_cache_specs()` 调用后紧跟处（core.py:245/:256），算规格产物在 profile_run 之前即可观测；启动轨迹 CFG 段 84→85 行、kvc_startup 168→172 行，[KVC] 计数 engine 13→15 行、总 163→168 行。


### 1.3 起服务并发送 P/R

```bash
bash scripts/start.sh                        # vllm serve PP2TP2 --enforce-eager，就绪 59s
bash scripts/curl_p_r.sh                     # P -> sleep 6 -> R；落盘打屏/响应/分界并提取三条 [KVC] 轨迹
```

### 1.4 收日志并去 patch

```bash
# 本地同步全套日志（12 文件）
scp -P 5557 -r root@localhost:/a3_inference/itask/workdir/gch02599191/kvc/log <本地kvc>/log/
# 容器回收
cd <kvc> && bash scripts/stop.sh             # 杀服务并确认 0 进程
cd patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_patches.sh
# 终态核验: 8 文件 [KVC] 全部归零 + py_compile + 两仓库 git 0 改动 + .orig 清理
```

## 2. 启动期日志讲解（log/kvc_startup.log，172 行）

### 2.1 配置侧（EngineCore，88 行 CFG，含首尾横幅）

对应理论文档初始化四阶段管线（① 算规格 → ② 测预算 → ③ 做编排 → ④ 落张量）中的 **① 算规格 + ② 测预算 + ③ 做编排**——本轮以 ①②③ **子步横幅**（`--- ①: 算规格 ---` / `--- ②: 测预算 ---` / `--- ③: 做编排 ---`）把配置侧日志划分成三个小步，① 打印紧跟 `model_executor.get_kv_cache_specs()` 调用。④ 落张量在 §2.2 物理侧。88 行结构地图：

| 行号 | 日志内容 | 对应理论阶段 |
|---|---|---|
| :1 | `================ 配置侧 KVCache 编排开始 ================` | 开始横幅（紧跟 `get_kv_cache_specs()` 调用） |
| :2 | `--- ①: 算规格 ---` | **① 子步横幅**（本轮新增） |
| :3 | `① 算规格 get_kv_cache_specs: worker0=16×FullAttentionSpec(…); …共 4 worker` | **① 算规格**（每 worker 层 spec 摘要：类型×层数×首末层名） |
| :4 | `--- ②: 测预算 ---` | **② 子步横幅**（先于 profile_run） |
| :5 | `determine_available_memory: 各 worker 可用 KV 显存` | **② 测预算**（`profile_run` 实测） |
| :6 | `--- ③: 做编排 ---` | **③ 子步横幅**（先于 get_kv_cache_configs） |
| :7~:86 | worker0~3 各 20 行（Config + GroupSpec + spec + page + KVCacheTensor ×16） | **③ 做编排**（合并 → 分组 → 投影 → num_blocks） |
| :87 | `最终 scheduler KVCacheConfig: num_blocks=… (跨 worker min 对齐)` | ③⑤ 多 worker min 对齐 |
| :88 | `================ 配置侧 KVCache 编排完成 ================` | 收尾横幅 |

**(a) ① 算规格（第 1~3 行，3 条：开始横幅 + ① 子步横幅 + ① 打印；紧跟 `get_kv_cache_specs()` 调用）**

```
INFO 09-29 02:27:44 [core.py:245] [KVC][CFG] ================ 配置侧 KVCache 编排开始 ================
INFO 09-29 02:27:44 [core.py:246] [KVC][CFG] --- ①: 算规格 ---
INFO 09-29 02:27:44 [core.py:257] [KVC][CFG] ① 算规格 get_kv_cache_specs: worker0=16×FullAttentionSpec(首 model.layers.0.self_attn.attn, 末 model.layers.15.self_attn.attn); worker1=16×FullAttentionSpec(首 model.layers.0.self_attn.attn, 末 model.layers.15.self_attn.attn); worker2=16×FullAttentionSpec(首 model.layers.16.self_attn.attn, 末 model.layers.31.self_attn.attn); worker3=16×FullAttentionSpec(首 model.layers.16.self_attn.attn, 末 model.layers.31.self_attn.attn)
```

① 打印的是 `get_kv_cache_specs()` 的返回值摘要——EngineCore 向每 worker 收集本 rank 各层 KVCacheSpec（`list[dict[层名, spec]]`），一行读完全部 4 worker：各 **16×FullAttentionSpec**（PP2 按层切：PP0 两卡 `layers.0~15`、PP1 两卡 `layers.16~31`；TP2 切 kv_heads 不切层，spec 各 rank 同形）。四 worker spec 字段全等是 (c) 中 `is_kv_cache_spec_uniform=true → 全模型单 group` 的直接依据（理论 §2.3②）。

**(b) ② 测预算（第 4~5 行，2 条：② 子步横幅 + ② 打印；横幅先行，先于 `profile_run`）**

```
INFO 09-29 02:27:44 [core.py:259] [KVC][CFG] --- ②: 测预算 ---
INFO 09-29 02:27:48 [core.py:280] [KVC][CFG] determine_available_memory: 各 worker 可用 KV 显存 = ['51.98GiB', '51.99GiB', '51.94GiB', '51.95GiB']
```

② 横幅（14:43:14）先落，随后各 worker 并行跑一次 `profile_run()`（dummy forward 量峰值，实测约 3s），结果 14:43:17 才打印：`available = 总显存 × 利用率 − 权重 − 激活 − 大图预留`（理论 §2.2）。四卡实测 51.94~51.99GiB；**最小者 51.94GiB（worker2）将决定 (c) 的 min 对齐**。

**(c) ③ 做编排（第 6~88 行——③ 横幅 + 80 行 worker + min 对齐 + 尾横幅）——worker0 段全量原样**

```
INFO 09-29 02:27:48 [core.py:284] [KVC][CFG] --- ③: 做编排 ---
INFO 09-29 02:27:48 [core.py:294] [KVC][CFG] worker0 KVCacheConfig: num_blocks=13296, groups数=1, tensors数=16
INFO 09-29 02:27:48 [core.py:299] [KVC][CFG]   [0] KVCacheGroupSpec(group_id=0): layers=16 (首层 model.layers.0.self_attn.attn, 末层 model.layers.15.self_attn.attn), is_eagle_group=False
INFO 09-29 02:27:48 [core.py:304] [KVC][CFG]   [0]   kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, kv_quant_mode=<KVQuantMode.NONE: 0>, page_size_padded=None, head_size_v=128, sliding_window=None, attention_chunk_size=None)
INFO 09-29 02:27:48 [core.py:308] [KVC][CFG]   [0]   page_size_bytes=262144 (256.0KB/层/块), storage_block_size=128
INFO 09-29 02:27:48 [core.py:314] [KVC][CFG]   [0] KVCacheTensor: size=3485466624 bytes (3324.00MiB), shared_by=1 层 (model.layers.0.self_attn.attn)
INFO 09-29 02:27:48 [core.py:314] [KVC][CFG]   [0] KVCacheTensor: size=3485466624 bytes (3324.00MiB), shared_by=1 层 (model.layers.1.self_attn.attn)
...（:13~:26 其余 14 条 KVCacheTensor 同构，仅层名 layers.2 ~ layers.15 逐层一列，每层一张）
```

逐行解读（对照理论 §2.3）：

| 行 | 字段 | 含义与公式验算 |
|---|---|---|
| Config（core.py:294） | `num_blocks=13296, groups数=1, tensors数=16` | **num_blocks = available ÷ page_size ÷ 16**（16 = projected 后本 worker 层数，PP2 下 32÷2；除数不是合并后的 32——理论 §2.3③"容量是 per-worker 的"） |
| GroupSpec（core.py:299） | `layers=16 (model.layers.0 ~ layers.15), is_eagle_group=False` | 32 层 spec 字段全等 → `is_kv_cache_spec_uniform=true` → **全模型单 group**（理论 §2.3②）；`layers=16` 是 `_project_kv_cache_groups_to_worker()` 从 32 层投影到本 rank 的结果 |
| spec（core.py:304） | `FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, bf16, …)` | **① 算规格的产物**：`num_kv_heads=4`（8 头 ÷ TP2，理论 §3；TP 切头不切层）；`block_size=128`（NPU 存储页）；无滑窗/无 padding |
| page（core.py:308） | `page_size_bytes=262144 (256KB/层/块)` | **一页字节 = block_size × num_kv_heads × head_size × dtype × 2(K/V) = 128 × 4 × 128 × 2B × 2 = 262,144**（理论 §4 公式 1，系数 2 对应 K/V 两份）精确闭合 |
| Tensor ×16（core.py:314） | `size=3485466624 bytes (3324.00MiB), shared_by=1 层` | **每层一张张量：size = page_size × num_blocks = 262,144 × 13,296 = 3,485,466,624** 精确闭合；主线 FullAttention 非打包 → `shared_by` 恒单层 |

四个 worker 段完全同构，仅两处不同（其余 78 行一致）：

| worker（行区间） | PP·TP | GroupSpec 层段 | Tensor 16 条 |
|---|---|---|---|
| worker0（:7~:26） | PP0·TP0 | `layers.0 ~ layers.15` | layers.0~.15 各一条 |
| worker1（:27~:46） | PP0·TP1 | `layers.0 ~ layers.15` | layers.0~.15 各一条 |
| worker2（:47~:66） | PP1·TP0 | `layers.16 ~ layers.31` | layers.16~.31 各一条 |
| worker3（:67~:86） | PP1·TP1 | `layers.16 ~ layers.31` | layers.16~.31 各一条 |

PP 按层切（每 worker 16 层）、TP 各 rank **同层同 spec**（理论 §3"spec 相等 ≠ 物理相同"——同一层的 4 个 KV 头子集在两个 TP rank 各自独立物化）。

③ 段尾（第 87~88 行）——跨 worker min 对齐 + 完成横幅：

```
INFO 09-29 02:27:48 [core.py:331] [KVC][CFG] 最终 scheduler KVCacheConfig: num_blocks=13296 (跨 worker min 对齐), cache_config.num_gpu_blocks=13296, block_size=128
INFO 09-29 02:27:48 [core.py:336] [KVC][CFG] ================ 配置侧 KVCache 编排完成 ================
```

集中式调度要求同一 `block_table` 对所有 rank 有效 → 取四 worker `num_blocks` 的 **min**（本例四卡恰好同为 13296，由最小预算 51.94GiB 卡定出）作为全局统一值（理论 §2.3⑤），并等比缩小各 `KVCacheTensor.size`，最后写回 `cache_config.num_gpu_blocks=13296`。

**预算闭合验算**：每 worker 16 张 × 3324.00MiB = 53,184MiB = **51.94GiB** 恰等于最小 worker 的 `available`——KV 预算被本 worker 的张量恰好铺满，编排无浪费。此后 §2.2（物理侧每层按此 size 建池）与 §2.3（逻辑侧 BlockPool 建 13,296 块）**消费同一份 KVCacheConfig**——两侧容量由同一配置锁定，是 `block_id == 张量行号` 桥接的编排前提。

### 2.2 物理侧（4 worker 并行，各 19 行 L1，vllm-ascend）——对应理论"④ 落张量"

对应理论 `../1_init_physical_memory.md` §2.4 的 ④ 落张量：`EngineCore → initialize_from_config() → initialize_kv_cache()`（4a/4b/4c 落张量 + 4d 编译预热两个 collective_rpc；4c/4d 无 [KVC] 打印）。4 卡并行各自执行；以下用 **Worker_PP0_TP0（pid=1077）的真实日志全量**示例（另 3 卡同构，仅 pid/device/层段不同——PP0 两卡打 `layers.0~15`、PP1 两卡打 `layers.16~31`）。

**(a) 4a 分配 int8 字节池（开始横幅 + 16 层 dense，每层 K/V 两个独立池）**

```
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4266] [KVC][L1] ================ 物理侧 KV Cache 分配开始 ================
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.0.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.1.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.2.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.3.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.4.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.5.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.6.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.7.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.8.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.9.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.10.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.11.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.12.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.13.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.14.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4428] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.15.self_attn.attn: KVCacheTensor(size=3485466624 bytes = 3324.00MiB) -> K int8 1662.00MiB + V int8 1662.00MiB (alignment=2097152, device=npu:0)
```

| 字段 | 值 | 验算（对照理论 §2.4 4a） |
|---|---|---|
| KVCacheTensor.size | 3,485,466,624 B = 3324.00MiB | = page 262,144 × num_blocks 13,296——**直接消费 (c) ③ 做编排产出的 KVCacheConfig** |
| K int8 池 | 1662.00MiB | = size ÷ 2（K/V 各半）= 1,742,733,312 B，恰为 alignment 的整数倍 |
| V int8 池 | 1662.00MiB | 同上——**每层 K/V 两个独立的 int8 池（张量级拆分）** |
| alignment | 2,097,152（2MiB） | NPU 张量 2MiB 对齐 |
| device | npu:0（本卡常驻） | 16 层逐层打印，每层一行 |

> **为什么用 int8？**（理论 §2.4）：与 dtype 解耦——先按字节量申请，之后 4b reshape 时再 `view(dtype)` 转回 bf16，同一分配逻辑适配任意 dtype。

**(b) 4b 零拷贝 reshape（K/V 张量就位 + 完成横幅）**

```
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4866] [KVC][L1] vllm-ascend _reshape_kv_cache_tensors: model.layers.0.self_attn.attn (本组 16 层同形) -> K_cache shape=(13296, 128, 4, 128) dtype=torch.bfloat16 / V_cache shape=(13296, 128, 4, 128) dtype=torch.bfloat16, device=npu:0 (K/V 分离布局)
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:48 [model_runner_v1.py:4917] [KVC][L1] ================ 物理侧 KV Cache 分配完成 ================
```

对照理论 4b：int8 → dtype → shape 的 **view 零拷贝**（`raw.view(dtype).view(shape)` 普通路径，理论 §2.4），再 permute 成后端逻辑布局——全程无数据拷贝，int8 池地址即最终张量地址。`K_cache shape=(13296, 128, 4, 128) bf16` 四维含义：dim0=num_blocks（**block id == 张量行号**，理论 §5 桥接）、dim1=block_size 128、dim2=kv_heads 4（8÷TP2）、dim3=head_dim 128；`K/V 分离布局`——两组独立的 `(num_blocks, 128, 4, 128)` 张量，区别于上游 GPU 的 K/V packed 单张量（理论 §5 表）。

### 2.3 逻辑侧装配（EngineCore，8 行全量）

**自底向上**逐组件装配，每组件一条 `__init__完成：` 打印统一格式；

```
INFO 09-29 02:27:58 [kv_cache_manager.py:143] [KVC][L5] ================ 逻辑侧初始化开始 ================
INFO 09-29 02:27:58 [kv_cache_utils.py:217] [KVC][L2] FreeKVCacheBlockQueue.__init__完成：num_free_blocks=13296, 伪头尾哨兵 fake_free_list_head/tail(block_id=-1), 类型=FreeKVCacheBlockQueue
INFO 09-29 02:27:58 [block_pool.py:62] [KVC][L2] BlockHashToBlockMap.__init__完成：底容器 size=0, value=KVCacheBlock | dict[block_id→KVCacheBlock](不去重 append-only)
INFO 09-29 02:27:58 [block_pool.py:214] [KVC][L2] BlockPool.__init__完成：num_gpu_blocks=13296, 创建 KVCacheBlock × 13296 (block_id=0..13295), free_block_queue=FreeKVCacheBlockQueue(num_free_blocks=13295), cached_block_hash_to_block=BlockHashToBlockMap(size=0), null_block=KVCacheBlock(block_id=0, is_null=True), enable_caching=True, hash_block_size=128
INFO 09-29 02:27:58 [single_type_kv_cache_manager.py:95] [KVC][L3] FullAttentionManager.__init__完成：spec=FullAttentionSpec(block_size=128), scheduler_block_size=128, group_id=0, enable_caching=True, dcp×pcp=1×1, block_pool(num_gpu_blocks=13296)
INFO 09-29 02:27:58 [kv_cache_coordinator.py:462] [KVC][L4] UnitaryKVCacheCoordinator.__init__完成：单组直通, managers=['FullAttentionManager'], kv_cache_spec=FullAttentionSpec(block_size=128, page_size_bytes=262144), coordinator_block_size=128
INFO 09-29 02:27:58 [kv_cache_manager.py:178] [KVC][L5] KVCacheManager.__init__完成：coordinator=UnitaryKVCacheCoordinator, num_kv_cache_groups=1, managers=['FullAttentionManager'], block_pool(num_gpu_blocks=13296), enable_caching=True, max_model_len=8192, empty_kv_cache_blocks=KVCacheBlocks([],)
INFO 09-29 02:27:58 [kv_cache_manager.py:186] [KVC][L5] ================ 逻辑侧初始化完成 ================
```

### 2.4 启动期原生关键行（vllm 自带，非 [KVC] patch）

这一节收的是 **[KVC] 调试打印之外、vllm 自己打的结论性行**：启动期有任何一步（算规格→测预算→做编排→落张量→装配）出问题，服务起不来或数值对不上，这 6 行是排障的第一落点。按日志出现序：

```
(APIServer pid=1008) INFO 09-29 02:27:10 [utils.py:1404] Block size is set to 128 if prefix cache or chunked prefill is enabled.
(Worker_PP0_TP0 pid=1077) INFO 09-29 02:27:47 [worker.py:593] Available KV cache memory: 51.98 GiB
(EngineCore pid=1047) INFO 09-29 02:27:48 [kv_cache_utils.py:1771] GPU KV cache size: 1,701,888 tokens
(EngineCore pid=1047) INFO 09-29 02:27:48 [kv_cache_utils.py:1772] Maximum concurrency for 8,192 tokens per request: 207.75x
(EngineCore pid=1047) INFO 09-29 02:27:57 [core.py:369] init engine (profile, create kv cache, warmup model) took 13.37 s
(APIServer pid=1008) INFO:     Application startup complete.
```

（6 行分别位于 `llama-3-8b.log` :29 / :148 / :151 / :152 / :331 / :387；最后一行即就绪标志，02:27:02 -> 02:28:00 共 58s）

逐行详解：

| 行号 | 谁打的（源码） | 说什么 / 怎么算 | 与 [KVC] 的对账 |
|---|---|---|---|
| :29 | APIServer（utils.py:1404） | **block_size=128**：开了 prefix cache / chunked prefill 时 vllm 把存储分块粒径定成 128（NPU 页对齐默认）——后面所有"满块 128 tokens、按 128 切哈希"的分母 | [KVC][L1] `K_cache=(num_blocks, 128, 4, 128)` 的 dim1=128 即此值 |
| :148 | Worker_PP0_TP0（worker.py:593） | worker 0 卡**实测可分配 KV 显存 51.98 GiB**（整卡可用 − 权重 − 激活 − warmup 峰值）；4 worker 各打 1 行，本卡只是第一行 | §2.1 (b) [KVC] `determine_available_memory` 的 51.98/51.99/51.94/51.95——同一测量、[KVC] 四卡一屏对比，原生行只逐卡逐行 |
| :151 | EngineCore（kv_cache_utils.py:1771） | **总 KV 容量 1,701,888 tokens** = num_blocks × block_size | 13,296 × 128 = 1,701,888——与 §2.1 (c) 最终 num_blocks 完全一致 |
| :152 | EngineCore（kv_cache_utils.py:1772） | **maximum concurrency 207.75x**：满载 8,192-token 长请求时的并发上限（理论上限，实际受连续批处理调度影响） | 1,701,888 ÷ 8,192 = 207.75——同源两行连算，max_model_len=8192 为分母 |
| :331 | EngineCore（core.py:369） | **init engine 13.37 s**：profile（=② 测预算的 dummy forward）+ create kv cache（=③ 做编排 + ④ 落张量）+ warmup 的总耗时 | [KVC] 各步（02:27:44 编排 → 02:27:58 逻辑侧完成）全落在这 13.37s 内——[KVC] 打印对启动时延的净增量可用此行与无 patch 版对比衡量 |
| :387 | APIServer（uvicorn） | **服务就绪标志**：路由全部挂载、uvicorn 开始收请求 | P 分界行号 387 与本行同号——§3 的 P/R 轨迹分界从这条起算（分界号由 curl_p_r.sh 运行时取行数，不落盘）；start.sh 的就绪探测就是 `grep 'Application startup complete' log/llama-3-8b.log` |

## 3. P 运行期日志讲解（log/kvc_p.log，124 行）

P：num_prompt_tokens=324（2 满块 + 尾 68），max_tokens=1——一次 prefill 即终态（TERM），验证"缓冲 2 块"。

> 以下按理论 `../0_runtime_sequence.md` §4 分阶段详解的时序组织：**入队 → 首次调度（①前缀查找 + ②allocate_slots）→ GPU 写 KV → （③ decode）→ ④ 结束释放**。GPU 写 KV 与 P 的 decode 环节无 [KVC] 打印（相应性由 KVP TERM 回读兜底验证）。

| 理论时序（0_runtime_sequence §4） | 日志小节 | P 实测要点 |
|---|---|---|
| §4.1 入队（预计算链式哈希 → WAITING） | §3.1 | 2 个满块哈希入队（NONE_HASH 起链） |
| §4.2 首次调度 ① get_computed_blocks（前缀查找） | §3.1 | 冷缓存第 1 hash 即 MISS，hit_length=0 |
| §4.2 首次调度 ② allocate_slots（S1~S4） | §3.2 | S1 需 3 vs 可用 13295 → S3 [1,2,3] → S4 双满块入表 |
| §4.3 GPU 写 KV（forward） | —（无 [KVC] 打印） | 写 324 token K/V；物理正确性由 KVP TERM 回读验证（§3.3） |
| ③ decode | — | P max_tokens=1，首 token 即终态，无 decode 循环 |
| ④ 结束释放（§4.5） | §3.4 | [3,2,1] 逆序 append 队尾（LRU 保护） |

### 3.1 入队 + ①前缀查找（对照理论 §4.1、§4.2-①）

```
INFO 09-29 02:29:58 [request.py:184] [KVC][ENQ] ======== 入队 ========
INFO 09-29 02:29:58 [kv_cache_utils.py:617] [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=3375832d2d59
INFO 09-29 02:29:58 [kv_cache_utils.py:617] [KVC][ENQ] 入队 hash_block_tokens: parent=3375832d2d59, tokens=128 -> BlockHash=34459d7f8362
INFO 09-29 02:29:58 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-995a13ddcc5595ca-0-a5c7908a) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['3375832d2d59', '34459d7f8362']
INFO 09-29 02:29:58 [request.py:193] [KVC][ENQ] ======== 入队完成 ========
INFO 09-29 02:29:58 [kv_cache_manager.py:222] [KVC][L5] ======== 前缀查找 ========
...（冷缓存: 第 1 块即 MISS 断链, hit_length=0, 返回 blocks=[[]]）
INFO 09-29 02:29:58 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 1 块 MISS: BlockHash=3375832d2d59 -> break
```

### 3.2 ②分配 S1~S4 全链（对照理论 §4.2-②；S1 段自洽实录）

总横幅(:393) → 进入(:395) → **S1 子步横幅(:402)** → 两次外层探问(:188×2) → S1 汇总值(:447)，随后 S2/S3/S4 四子步下钻逐层展开：

```
INFO 09-29 02:29:58 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 09-29 02:29:58 [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_new_tokens=324, num_new_computed_tokens=0, request.num_computed_tokens=0, request.num_tokens=324
INFO 09-29 02:29:58 [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---
INFO 09-29 02:29:58 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 09-29 02:29:58 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 09-29 02:29:58 [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13295 块 (free=13295 - reserved=0)
INFO 09-29 02:29:58 [kv_cache_manager.py:481] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 09-29 02:29:58 [kv_cache_manager.py:485] [KVC][L5] --- S3: 新块分配 ---
INFO 09-29 02:29:58 [block_pool.py:411] [KVC][L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=13292
INFO 09-29 02:29:58 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_tokens=324, block_size=128, 需 3 块 - 已有 0 = 新分配 3 块 [1, 2, 3], 持有 req_blocks=[1, 2, 3]
INFO 09-29 02:29:58 [kv_cache_manager.py:496] [KVC][L5] S3 allocate_new_blocks: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_tokens_need_slot=324 -> 新块 [1, 2, 3]
INFO 09-29 02:29:58 [kv_cache_manager.py:523] [KVC][L5] --- S4: 满块入缓存 ---
INFO 09-29 02:29:58 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=3375832d2d59, group_id=0) <- KVCacheBlock(block_id=1), map size=1
INFO 09-29 02:29:58 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=34459d7f8362, group_id=0) <- KVCacheBlock(block_id=2), map size=2
INFO 09-29 02:29:58 [kv_cache_manager.py:529] [KVC][L5] S4 cache_blocks: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_tokens_to_cache=324
INFO 09-29 02:29:58 [kv_cache_manager.py:533] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([1, 2, 3],)), req=cmpl-995a13ddcc5595ca-0-a5c7908a 当前完整 block_table=([1, 2, 3],)
INFO 09-29 02:29:58 [kv_cache_manager.py:539] [KVC][L5] ======== 分配完成 ========
```

要点：324 tokens → S3 需 3 块（`需 3 块 - 已有 0 = 新分配 3 块`）；S4 只把 2 个满块入哈希表（324 // 128 = 2，`insert` ×2），块 3 为尾块（68/128）不入表。

### 3.3 KVP 物理校验（实装侧校验点，理论时序之外；TERM ×4 卡，每卡 18 行）

以 dev=npu:0 为例（P 一步即终态，TERM 文案；概览含一次性 KV 布局说明，层行每层 1 条、尾块 `未满:68`，层统计 n=165888=324×4×128）：

```
INFO 09-29 02:29:58 [model_runner_v1.py:2518] [KVC][KVP] ======== 请求结束 KVCache 即将释放, 开始打印该请求物理 cache ========
INFO 09-29 02:29:58 [model_runner_v1.py:2520] [KVC][KVP] TERM req=cmpl-995a13ddcc5595ca-0-a5c7908a dev=npu:0 逐层按块: layers=16 blocks=[1, 2, 3] region=324/324 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=…
INFO 09-29 02:29:58 [model_runner_v1.py:2563] [KVC][KVP] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=3[未满:68](68,4,128) | K示(首块首token前3)=[-0.008057, 0.1035, 0.02747] 统计[n=165888] mean=0.04895 std=1.508 min=-14.56 max=8.5 | V示(首块首token前3)=[0.0009766, 0.000576, -0.0008163] 统计[n=165888] mean=1.096e-05 std=0.03965 min=-0.3477 max=0.4707
...（L01~L15 同构 ×16 行/卡, 共 4 卡）
INFO 09-29 02:29:58 [model_runner_v1.py:2570] [KVC][KVP] ======== 请求结束, 物理 cache 打印完毕 ========
```

### 3.4 ④调度提交与结束释放（对照理论 §4.5）

async_scheduler 每步输出后的独立提交段（与分配 S4 明确区分），随后逆序释放、带哈希块 LRU 归队：

```
INFO 09-29 02:29:59 [kv_cache_manager.py:676] [KVC][L5] ======== 调度提交(非分配 S4) ========
INFO 09-29 02:29:59 [kv_cache_manager.py:677] [KVC][L5] 提交 cache_blocks: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_computed_tokens=324 (async 步末输出路径: 本步已算 token 提交入缓存)
INFO 09-29 02:29:59 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-995a13ddcc5595ca-0-a5c7908a, num_computed_tokens=324
INFO 09-29 02:29:59 [kv_cache_manager.py:683] [KVC][L5] ======== 提交完成 ========
INFO 09-29 02:29:59 [kv_cache_manager.py:551] [KVC][L5] ======== 释放 ========
INFO 09-29 02:29:59 [kv_cache_manager.py:553] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-995a13ddcc5595ca-0-a5c7908a, 释放前持有 block_table=([1, 2, 3],)
INFO 09-29 02:29:59 [block_pool.py:516] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(3, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 3 块 [3, 2, 1], append_n -> 队尾(LRU保护)
INFO 09-29 02:29:59 [kv_cache_utils.py:396] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[3, 2, 1]), num_free_blocks=13295
INFO 09-29 02:29:59 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

### 3.5 响应核对

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| P | 1 | length | `"为了"` |

## 4. R 运行期日志讲解（log/kvc_r.log，752 行）

R：num_prompt_tokens=486 = 3 满 + 第 4 块 102/128，max_tokens=35——完整五块生命周期（复用 2 + prefill 补 1 满 1 尾 + decode 填满尾块 + 步 27 跨界申请第 5 块）。request_id=cmpl-b7272ef6037e383c-0-8264eafe。

> 按理论 `../0_runtime_sequence.md` §4 时序组织：R 覆盖**全部阶段**（理论时序图以 70 token/16 块示例推演，此处为 486 token/128 块全量实测）。

| 理论时序（0_runtime_sequence §4） | 日志小节 | R 实测要点 |
|---|---|---|
| §4.1 入队 | §4.1 | 3 个满块哈希入队（P 链延伸 + 追问句新段） |
| §4.2 ① 前缀查找（get_computed_blocks） | §4.1 | HIT×2 → 第 3 hash MISS 断链，hit_length=2×128=256 |
| §4.2 ② allocate_slots（S1~S4） | §4.2 | S2 touch[(1,1),(2,1)] 零拷贝复用；S3 [4,5]；S4 新满块入表 |
| §4.3 GPU 写 KV（forward） | —（无 [KVC] 打印） | 跨 4 块写 486 token（复用块不重算） |
| §4.4 ③ decode·情况 A（需 0 块） | §4.3 | 33 步同构闭合链 + 每步调度提交 |
| §4.4 ③ decode·情况 B（需 1 块） | §4.4 | 步 27 跨界：新块 [6]，刚填满的块 5 入表 |
| KVP 物理校验（实装侧，理论之外） | §4.5 | TERM 五块全景；K 示值与 P 一致 = 零拷贝证据 |
| ④ 结束释放（§4.5） | §4.6 | 五块逆序归队 [6,5,4,2,1] |

### 4.1 入队 + ①前缀查找（对照理论 §4.1、§4.2-①；HIT×2 后 MISS 断链）

```
INFO 09-29 02:30:05 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-b7272ef6037e383c-0-8264eafe) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 BlockHash × 3: ['3375832d2d59', '34459d7f8362', '5bbf6f30dff0']
INFO 09-29 02:30:05 [kv_cache_coordinator.py:477] [KVC][L4] 前缀查找 UnitaryKVCacheCoordinator.find_longest_cache_hit: 满块hash数=3, max_cache_hit_length=485, 下钻 single_type_managers[0]
INFO 09-29 02:30:05 [block_pool.py:70] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=3375832d2d59, group_id=0) -> HIT KVCacheBlock(block_id=1)
INFO 09-29 02:30:05 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 1 块 HIT: BlockHash=3375832d2d59 -> cached blocks=[1]
INFO 09-29 02:30:05 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 2 块 HIT: BlockHash=34459d7f8362 -> cached blocks=[2]
INFO 09-29 02:30:05 [block_pool.py:84] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=5bbf6f30dff0, group_id=0) -> MISS
INFO 09-29 02:30:05 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 3 块 MISS: BlockHash=5bbf6f30dff0 -> break
```

命中 2 块 → hit_length=2×128=256；第 3 个 hash 是 R 追问句新内容（冷），断链即止。

### 4.2 ②prefill 完整链（对照理论 §4.2-②；S2 touch 复用 + S3 补 2 块 + S4 新满块入表）

```
INFO 09-29 02:30:05 [kv_cache_manager.py:467] [KVC][L5] --- S2: touch 命中块 ---
INFO 09-29 02:30:05 [kv_cache_manager.py:469] [KVC][L5] S2 allocate_new_computed_blocks: req=cmpl-b7272ef6037e383c-0-8264eafe, new_computed_blocks=[[1, 2]]
INFO 09-29 02:30:05 [block_pool.py:491] [KVC][L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
INFO 09-29 02:30:05 [kv_cache_manager.py:485] [KVC][L5] --- S3: 新块分配 ---
INFO 09-29 02:30:05 [block_pool.py:411] [KVC][L2] S3 BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13291
INFO 09-29 02:30:05 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-b7272ef6037e383c-0-8264eafe, num_tokens=486, block_size=128, 需 4 块 - 已有 2 = 新分配 2 块 [4, 5], 持有 req_blocks=[1, 2, 4, 5]
INFO 09-29 02:30:05 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=5bbf6f30dff0, group_id=0) <- KVCacheBlock(block_id=4), map size=3
INFO 09-29 02:30:05 [kv_cache_manager.py:533] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([4, 5],)), req=cmpl-b7272ef6037e383c-0-8264eafe 当前完整 block_table=([1, 2, 4, 5],)
INFO 09-29 02:30:05 [kv_cache_manager.py:539] [KVC][L5] ======== 分配完成 ========
```

要点：**touch 零拷贝**——`S2 BlockPool.touch: blocks=[(1, 1), (2, 1)]`（block_id, ref_cnt），直接复用 P 留下的块 1/2；S3 = cdiv(486,128)−2 = 2 新块 [4,5]；S4 把追问句恰好填满的块 4 入表（map 2→3）。随后的 SchedulerOutput 携带 `new_block_ids_to_zero=[4, 5]`（对应理论 §4.3"调度输出附清零块 id"）——GPU forward 写 KV 无 [KVC] 日志，物理正确性由 §4.5 的 KVP 回读实测验证。

### 4.3 ③decode 无块步——理论"情况 A：需 0 块"（对照 §4.4；33 步同构，完整 14 行实录）

理论 §4.4 把 decode 每步 allocate_slots 分两种情况：**情况 A**·当前块未满需 0 块（token 直接续写）/ **情况 B**·已满需 1 块（token 落进下一块）。本节是情况 A 的每步实录；decode 步 1 起的每一步（除步 27 外）都是这条闭合链——S1 如实打"需分配 0"，S3 切换"无需分配新块"文案，尾部紧跟本步调度提交：

```
INFO 09-29 02:30:05 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 09-29 02:30:05 [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-b7272ef6037e383c-0-8264eafe, num_new_tokens=1, num_new_computed_tokens=0, request.num_computed_tokens=486, request.num_tokens=486
INFO 09-29 02:30:05 [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---
INFO 09-29 02:30:05 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-b7272ef6037e383c-0-8264eafe, num_tokens=487 -> 需分配 0 块(含touch需腾挪的块)
INFO 09-29 02:30:05 [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 0 块 vs 可用 13291 块 (free=13291 - reserved=0)
INFO 09-29 02:30:05 [kv_cache_manager.py:481] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 09-29 02:30:05 [kv_cache_manager.py:487] [KVC][L5] --- S3: 无需分配新块 ---
INFO 09-29 02:30:05 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-b7272ef6037e383c-0-8264eafe, num_tokens=487 -> [[]]
INFO 09-29 02:30:05 [kv_cache_manager.py:501] [KVC][L5] S3 块未满, 无需分配新块 (req=cmpl-b7272ef6037e383c-0-8264eafe, num_new_tokens=1)
INFO 09-29 02:30:05 [kv_cache_manager.py:523] [KVC][L5] --- S4: 满块入缓存 ---
INFO 09-29 02:30:05 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-b7272ef6037e383c-0-8264eafe, num_computed_tokens=486
INFO 09-29 02:30:05 [kv_cache_manager.py:529] [KVC][L5] S4 cache_blocks: req=cmpl-b7272ef6037e383c-0-8264eafe, num_tokens_to_cache=486
INFO 09-29 02:30:05 [kv_cache_manager.py:533] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([],)), req=cmpl-b7272ef6037e383c-0-8264eafe 当前完整 block_table=([1, 2, 4, 5],)
INFO 09-29 02:30:05 [kv_cache_manager.py:539] [KVC][L5] ======== 分配完成 ========
INFO 09-29 02:30:05 [kv_cache_manager.py:676] [KVC][L5] ======== 调度提交(非分配 S4) ========
```

### 4.4 ③decode 步 27 跨界——理论"情况 B：需 1 块"（对照 §4.4；第 512 个 token 触发第 5 块申请）

尾块 102/128 被 decode 逐步填满——步 27 时 num_computed_tokens=512，S1 汇总值切换为"需分配 1 块"，S3 弹出新块 [6]，S4 把刚填满的块 5 入表（map 3→4）：

```
INFO 09-29 02:30:05 [kv_cache_manager.py:395] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-b7272ef6037e383c-0-8264eafe, num_new_tokens=1, num_new_computed_tokens=0, request.num_computed_tokens=512, request.num_tokens=512
INFO 09-29 02:30:05 [kv_cache_manager.py:402] [KVC][L5] --- S1: 容量检查---
INFO 09-29 02:30:05 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-b7272ef6037e383c-0-8264eafe, num_tokens=513 -> 需分配 1 块(含touch需腾挪的块)
INFO 09-29 02:30:05 [kv_cache_manager.py:447] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 1 块 vs 可用 13291 块 (free=13291 - reserved=0)
INFO 09-29 02:30:05 [kv_cache_manager.py:485] [KVC][L5] --- S3: 新块分配 ---
INFO 09-29 02:30:05 [block_pool.py:411] [KVC][L2] S3 BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=13290
INFO 09-29 02:30:05 [block_pool.py:108] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=58c7c704c9b6, group_id=0) <- KVCacheBlock(block_id=5), map size=4
INFO 09-29 02:30:05 [block_pool.py:340] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-b7272ef6037e383c-0-8264eafe 新满块 1 块 block_ids=[5] 入 BlockHashToBlockMap (num_cached_blocks 3 -> 4, group_id=0, map size=4)
```

### 4.5 KVP 物理校验（实装侧校验点，理论时序之外；4 卡 × 18 行，五块全景）

region=520/520 = 4×128 + 8（第 5 块仅 8 个有效 token）；层行一屏内联全部 5 块，层统计 n=266240=520×4×128 精确断言：

```
INFO 09-29 02:30:06 [model_runner_v1.py:2520] [KVC][KVP] TERM req=cmpl-b7272ef6037e383c-0-8264eafe dev=npu:0 逐层按块: layers=16 blocks=[1, 2, 4, 5, 6] region=520/520 tok | KV 布局: K_cache 与 V_cache 是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=128), 第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维…
INFO 09-29 02:30:06 [model_runner_v1.py:2563] [KVC][KVP] TERM L00 blk=1[满:128](128,4,128) blk=2[满:128](128,4,128) blk=4[满:128](128,4,128) blk=5[满:128](128,4,128) blk=6[未满:8](8,4,128) | K示(首块首token前3)=[0.5078, 0.9336, 0.9219] 统计[n=266240] mean=-0.0149 std=1.402 min=-10.38 max=10.81 | V示(首块首token前3)=[0.01538, 0.0008049, 0.03345] 统计[n=266240] mean=0.0006548 std=0.03523 min=-0.3438 max=0.3223
...（L01~L15 ×16 行/卡, 4 卡 = 64 条层行）
INFO 09-29 02:30:06 [model_runner_v1.py:2570] [KVC][KVP] ======== 请求结束, 物理 cache 打印完毕 ========
```

K 示首 3 值与 P 的块 1 一致（`0.5078, 0.9336, 0.9219`）——**复用块零拷贝**的直接证据（同一物理块，K/V 数据未动）。

### 4.6 ④结束释放（对照理论 §4.5：五块逆序释放与计数自检）

```
INFO 09-29 02:30:06 [kv_cache_manager.py:553] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-b7272ef6037e383c-0-8264eafe, 释放前持有 block_table=([1, 2, 4, 5, 6],)
INFO 09-29 02:30:06 [block_pool.py:516] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(6, 0), (5, 0), (4, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 5 块 [6, 5, 4, 2, 1], append_n -> 队尾(LRU保护)
INFO 09-29 02:30:06 [kv_cache_utils.py:396] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[6, 5, 4, 2, 1]), num_free_blocks=13295
INFO 09-29 02:30:06 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

**全程计数自检**（grep 即可复验）：`调度提交` P=1 / R=**35**（每 decode 步一次）；KVP 层行 P=R=**64**（4 卡 × 16 层，固定）；S1 汇总值 R = **33×0 + 1×1 + 1×4**（无块步 33 + 跨界 1 + prefill 1）；哈希链 `3375832d2d59 → 34459d7f8362 → 5bbf6f30dff0`（+ decode 段 `58c7c704c9b6`）。

### 4.7 响应核对

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| R | 35 | length | 中文贪心续写 |

## 5. 实证结论

1. **patch 168 行验证通过**：8/8 应用、91 调用点、py_compile OK；本地回环（stash→apply→revert→stash pop）与容器 apply/revert 双向验证。
2. **S1 段完整自洽**：子步横幅先行（:402）→ 两次外层探问下钻（coordinator:188 ×2）→ 汇总值（:447）——S1 语义日志全部在子步横幅之内。
3. **S1~S4 全子步可观测**：无块步打出完整四段子步（S1 需分配 0 / S2 无前缀 / S3 无需分配 / S4 维护），R 33 个无块步全部闭合。
4. **KVP 可读性**：每请求 KVP 固定 76 行（4 卡 × 18）；每层一行内联块标注 + K/V 首 3 值示意 + 层合并统计——一眼可读。
5. **KV 布局实证**：K/V 为张量级拆分的两个独立池（非最后一维拼接）；层统计 `n = region × kv_heads(4) × head_dim(128)` 精确断言（P: 165888、R: 266240）；复用块 K 示值跨请求一致（零拷贝）。
6. **容器回收闭环**：服务已杀（0 进程）、8 补丁 revert 归零、两仓库 git 0 改动、.orig 清理。

## 6. 实测产物

| 产物（`log/`） | 说明 |
|---|---|
| `log/llama-3-8b.log`（1272 行） | 服务全量日志（启动 :1~387 + P :388~517 + R :518~1272，带进程前缀完整版） |
| `log/kvc_startup.log`（168 行） | 启动期 [KVC] 拆解轨迹 |
| `log/kvc_p.log`（124 行）/ `log/kvc_r.log`（752 行） | P / R 运行期 [KVC] 拆解轨迹 |
| `log/req_*.json`、`resp_*.json` | 请求体 / 响应体（curl 命令见 docs/2 §3） |
| （P/R 分界行号 387 / 517 由 curl_p_r.sh 运行时确定，不落盘） |