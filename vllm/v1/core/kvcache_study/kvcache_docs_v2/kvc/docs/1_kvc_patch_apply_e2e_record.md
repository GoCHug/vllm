# KVCache 调试体系实验（全生命周期打印轨迹 + TERM 物理KV原样归档）

> 本文是 `2_kvc_cn_curl_case.md`（用例）与 `patch/`（打印补丁）的**端到端正式验证记录**：从干净源码出发，以 patch 方式注入 `[KVC]` 调试体系（grep 计数 174 行 = vllm 155 + vllm-ascend 19，含注释行；**08 号补丁为 v2.3 归档版——请求结束把各 worker 的物理 KV tensor 原样 save 成 .pt 而非打印**），记录一次完整的服务启动初始化、P/R 双请求生命周期及 8 份物理 KV 归档的离线验证。§2~§4 的日志引文均**逐字原样**取自本次实测的拆解轨迹（`log/kvc_startup.log` / `kvc_p.log` / `kvc_r.log`，已剥离进程前缀；带 `(Worker pid=…)` 前缀的完整版见 `log/llama-3-8b.log`）。

## 0. 总览

### 0.1 调试体系全生命周期设计

（grep 口径 174 行 = vllm 155 + vllm-ascend 19，三级横幅 + 阶段前缀贯穿）

**一、启动初始化（一次性，172 行 [KVC]，三段各带 `================` 开始/完成横幅）**

| 段 | 层标签 | 行数 | 核心内容 |
|---|---|---|---|
| 配置侧 | CFG | 88 | **① 算规格（紧跟 get_kv_cache_specs() 调用，本轮前移至此）** → ② 各 worker 可用 KV 显存 → 逐 worker `KVCacheConfig`（num_blocks/组数/张量数）→ 逐张量 size/shared_by → 最终 scheduler 侧 min 对齐 |
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
5. **KVS 物理归档（仅请求结束 TERM / 兜底 LATE 告警）**——TERM 时各 worker 把该请求全部物理 KV 块**整块原样**（`.cpu().clone()` 位级快照，含未写槽位）交后台线程 `torch.save` 为 tensors/*.pt；日志仅横幅两行：`======== 开始保存物理tensor ========`（worker/dev/shape/dtype/层×块/blk/cov/文件名）+ `======== 完成保存物理tensor ========`（文件名/字节/落盘耗时）；逐层逐块粒度由 .pt meta + `scripts/inspect_kv_tensors.py` 离线承载
6. **释放**——逆序归还、ref_cnt 归零回收、带哈希块 append 队尾（LRU 保护）

### 0.2 实测环境

| 项 | 值 |
|---|---|
| Pod | gggtest（a3 · 4× Ascend910 · PP2TP2 · 当日 Running） |
| 模型 | Meta-Llama-3-8B（`modelhub_74000048_meta-llama-3-8b-148700128_20260921221233`，32 层 / kv_heads 8 / TP2 下本地 4 / block_size 128）https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B |
| 软件栈 | vllm 0.23.0 + vllm-ascend 0.23.0（`/vllm-workspace/`） |
| 进程 | APIServer pid=94246 · EngineCore pid=94363 · Worker pid=94580~94583（PP0_TP0 / PP0_TP1 / PP1_TP0 / PP1_TP1） |
| 实测时间 | 2026-10-02 16:24~16:27（log 内时间戳，容器时钟 UTC-8） |

## 1. 实验流程

### 1.1 起容器

```bash
# pod 状态确认（平台空闲会回收为 Stopped，需 itask start 拉起）
itask list | grep gggtest            # 期望 Running
itask start gggtest                  # 若 Stopped 时拉起
# SSH 隧道（本地 5557 -> 容器 7890）；pod 重建后 ssh host key 变化需先清除
itask ssh-tunnel gggtest --port 5557
ssh-keygen -f ~/.ssh/known_hosts -R "[localhost]:5557"   # pod 重建后
ssh -p 5557 root@localhost "hostname; whoami"            # 连通确认
```

源码干净基线核验（8 个文件 `grep -c "\[KVC\]"` 全部为 **0**、两仓库 `git status` 0 改动）。容器若经平台重启，可写层自动还原，此项天然满足。

### 1.2 打 patch

```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
# Phase0 已应用检测 -> Phase1 dry-run 8/8 预检 -> Phase2 8/8 应用 -> Phase3 逐文件计数(合计 174 行) + py_compile
```

- **174 行**——apply 脚本 Phase 3 用 `grep -c "[KVC]"` 逐文件统计、并与预期值比对的口径，= **vllm 155（01~07 管理侧打印）+ vllm-ascend 19（08 归档版：[L1] 8 行 + [KVS] 11 行）**，均含注释行（`# [KVC]…` 设计说明，不产生日志；打印语句与注释的拆分随补丁演进略有浮动，以 grep 口径为准）。
- **[KVS] 输出量恒定**——运行期归档日志每请求每 worker 2 行 + 每 worker 启用 1 行，不随层×块数增长。

| 补丁 | 文件 | grep 行数（含注释） |
|---|---|---|
| 01 | `vllm/v1/request.py` | 5 |
| 02 | `v1/core/kv_cache_utils.py` | 10 |
| 03 | `v1/core/block_pool.py` | 29 |
| 04 | `v1/core/kv_cache_manager.py` | **58** |
| 05 | `v1/core/kv_cache_coordinator.py` | 17 |
| 06 | `v1/core/single_type_kv_cache_manager.py` | 18 |
| 07 | `v1/engine/core.py` | 18 |
| 08 | `vllm_ascend/worker/model_runner_v1.py` | **19**（[L1] 8 + [KVS] 11，v2.3 归档版） |
| **合计** | 8 文件 | **174** = vllm 155 + vllm-ascend 19 |

**08 号补丁版本演进**（本套补丁仅 08 随轮次演进，01~07 管理侧打印不变）：

| 版本 | TERM 行为 | [KVS] 行/轮(P+R) |
|---|---|---|
| v1 打印版（09-29，历史） | 逐层统计打印（%.4g 文本，不可比对） | 152（76 行/请求） |
| v2 归档版（10-02 早） | 归档 .pt，1 行汇总（细节不可见） | 12 |
| v2.1 全链版（10-02 午） | 归档 + 逐层逐块行 + 回执 | 532（刷屏） |
| v2.2 精简版（10-02 午） | 归档 + SAVE/SAVED 两行式 | 20 |
| **v2.3（当前）** | 归档 + **横幅式**两行（与 L5/L1 风格统一） | **20** |

> **补丁演进**：07 号（历史 v2）把 CFG 编排开始横幅 + ① 算规格打印**前移**到 `model_executor.get_kv_cache_specs()` 调用后紧跟处（core.py:245/:256），算规格产物在 profile_run 之前即可观测（CFG 段 84→88 行）；04 号现 58 行（含“分配布局”分段行）。**08 号由 v1 打印版（15 行 [KVP]，TERM 逐层统计打印）重设计为 v2.3 归档版（19 行 = [L1] 8 + [KVS] 11）——“打印”换成“save 下来处理”，01~07 管理侧打印原样保留。**


### 1.3 起服务并发送 P/R

```bash
bash scripts/start.sh                        # vllm serve PP2TP2 --enforce-eager，就绪 50s
bash scripts/curl_p_r.sh                     # P -> sleep 6 -> R；落盘打屏/响应/分界并提取三条 [KVC] 轨迹
```

### 1.4 收产物并去 patch

```bash
# 产物回收: 容器内 pack -> 主机侧 fetch (log/ + tensors/ 全套, manifest md5 校验)
bash scripts/pull_artifacts.sh pack     # 容器内 kvc/: 等归档就位 -> manifest.json -> tar
bash scripts/pull_artifacts.sh fetch    # 主机侧 kvc/: 经 5557 隧道拉回 -> 解包 -> md5 8/8 OK
# 容器回收
cd <kvc> && bash scripts/stop.sh             # 杀服务并确认 0 进程
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend bash patch/revert_patches.sh
# 本地离线分析(输出留痕 log/inspect_*.out): python3 scripts/inspect_kv_tensors.py --dir tensors
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
INFO 10-02 16:25:37 [core.py:245] [KVC][CFG] ================ 配置侧 KVCache 编排开始 ================
INFO 10-02 16:25:38 [core.py:246] [KVC][CFG] --- ①: 算规格 ---
INFO 10-02 16:25:38 [core.py:257] [KVC][CFG] ① 算规格 get_kv_cache_specs: worker0=16×FullAttentionSpec(首 model.layers.0.self_attn.attn, 末 model.layers.15.self_attn.attn); worker1=16×FullAttentionSpec(首 model.layers.0.self_attn.attn, 末 model.layers.15.self_attn.attn); worker2=16×FullAttentionSpec(首 model.layers.16.self_attn.attn, 末 model.layers.31.self_attn.attn); worker3=16×FullAttentionSpec(首 model.layers.16.self_attn.attn, 末 model.layers.31.self_attn.attn)
```

① 打印的是 `get_kv_cache_specs()` 的返回值摘要——EngineCore 向每 worker 收集本 rank 各层 KVCacheSpec（`list[dict[层名, spec]]`），一行读完全部 4 worker：各 **16×FullAttentionSpec**（PP2 按层切：PP0 两卡 `layers.0~15`、PP1 两卡 `layers.16~31`；TP2 切 kv_heads 不切层，spec 各 rank 同形）。四 worker spec 字段全等是 (c) 中 `is_kv_cache_spec_uniform=true → 全模型单 group` 的直接依据（理论 §2.3②）。

**(b) ② 测预算（第 4~5 行，2 条：② 子步横幅 + ② 打印；横幅先行，先于 `profile_run`）**

```
INFO 10-02 16:25:38 [core.py:259] [KVC][CFG] --- ②: 测预算 ---
INFO 10-02 16:25:40 [core.py:280] [KVC][CFG] determine_available_memory: 各 worker 可用 KV 显存 = ['51.96GiB', '51.97GiB', '51.92GiB', '51.93GiB']
```

② 横幅（16:25:38）先落，随后各 worker 并行跑一次 `profile_run()`（dummy forward 量峰值，实测约 2s），结果 16:25:40 才打印：`available = 总显存 × 利用率 − 权重 − 激活 − 大图预留`（理论 §2.2）。四卡实测 51.92~51.97GiB；**最小者 51.92GiB（worker2）将决定 (c) 的 min 对齐**。

**(c) ③ 做编排（第 6~88 行——③ 横幅 + 80 行 worker + min 对齐 + 尾横幅）——worker0 段全量原样**

```
INFO 10-02 16:25:40 [core.py:284] [KVC][CFG] --- ③: 做编排 ---
INFO 10-02 16:25:40 [core.py:294] [KVC][CFG] worker0 KVCacheConfig: num_blocks=13291, groups数=1, tensors数=16
INFO 10-02 16:25:40 [core.py:299] [KVC][CFG]   [0] KVCacheGroupSpec(group_id=0): layers=16 (首层 model.layers.0.self_attn.attn, 末层 model.layers.15.self_attn.attn), is_eagle_group=False
INFO 10-02 16:25:40 [core.py:304] [KVC][CFG]   [0]   kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, kv_quant_mode=<KVQuantMode.NONE: 0>, page_size_padded=None, head_size_v=128, sliding_window=None, attention_chunk_size=None)
INFO 10-02 16:25:40 [core.py:308] [KVC][CFG]   [0]   page_size_bytes=262144 (256.0KB/层/块), storage_block_size=128
INFO 10-02 16:25:40 [core.py:314] [KVC][CFG]   [0] KVCacheTensor: size=3484155904 bytes (3322.75MiB), shared_by=1 层 (model.layers.0.self_attn.attn)
INFO 10-02 16:25:40 [core.py:314] [KVC][CFG]   [0] KVCacheTensor: size=3484155904 bytes (3322.75MiB), shared_by=1 层 (model.layers.1.self_attn.attn)
...（:13~:26 其余 14 条 KVCacheTensor 同构，仅层名 layers.2 ~ layers.15 逐层一列，每层一张）
```

逐行解读（对照理论 §2.3）：

| 行 | 字段 | 含义与公式验算 |
|---|---|---|
| Config（core.py:294） | `num_blocks=13291, groups数=1, tensors数=16` | **num_blocks = available ÷ page_size ÷ 16**（16 = projected 后本 worker 层数，PP2 下 32÷2；除数不是合并后的 32——理论 §2.3③"容量是 per-worker 的"） |
| GroupSpec（core.py:299） | `layers=16 (model.layers.0 ~ layers.15), is_eagle_group=False` | 32 层 spec 字段全等 → `is_kv_cache_spec_uniform=true` → **全模型单 group**（理论 §2.3②）；`layers=16` 是 `_project_kv_cache_groups_to_worker()` 从 32 层投影到本 rank 的结果 |
| spec（core.py:304） | `FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, bf16, …)` | **① 算规格的产物**：`num_kv_heads=4`（8 头 ÷ TP2，理论 §3；TP 切头不切层）；`block_size=128`（NPU 存储页）；无滑窗/无 padding |
| page（core.py:308） | `page_size_bytes=262144 (256KB/层/块)` | **一页字节 = block_size × num_kv_heads × head_size × dtype × 2(K/V) = 128 × 4 × 128 × 2B × 2 = 262,144**（理论 §4 公式 1，系数 2 对应 K/V 两份）精确闭合 |
| Tensor ×16（core.py:314） | `size=3484155904 bytes (3322.75MiB), shared_by=1 层` | **每层一张张量：size = page_size × num_blocks = 262,144 × 13,291 = 3,484,155,904** 精确闭合；主线 FullAttention 非打包 → `shared_by` 恒单层 |

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
INFO 10-02 16:25:40 [core.py:331] [KVC][CFG] 最终 scheduler KVCacheConfig: num_blocks=13291 (跨 worker min 对齐), cache_config.num_gpu_blocks=13291, block_size=128
INFO 10-02 16:25:40 [core.py:336] [KVC][CFG] ================ 配置侧 KVCache 编排完成 ================
```

集中式调度要求同一 `block_table` 对所有 rank 有效 → 取四 worker `num_blocks` 的 **min**（本例四卡恰好同为 13291，由最小预算 51.92GiB 卡定出）作为全局统一值（理论 §2.3⑤），并等比缩小各 `KVCacheTensor.size`，最后写回 `cache_config.num_gpu_blocks=13291`。

**预算闭合验算**：每 worker 16 张 × 3322.75MiB = 53,164MiB = **51.92GiB** 恰等于最小 worker 的 `available`——KV 预算被本 worker 的张量恰好铺满，编排无浪费。此后 §2.2（物理侧每层按此 size 建池）与 §2.3（逻辑侧 BlockPool 建 13,291 块）**消费同一份 KVCacheConfig**——两侧容量由同一配置锁定，是 `block_id == 张量行号` 桥接的编排前提。

### 2.2 物理侧（4 worker 并行，各 19 行 L1，vllm-ascend）——对应理论"④ 落张量"

对应理论 `../1_init_physical_memory.md` §2.4 的 ④ 落张量：`EngineCore → initialize_from_config() → initialize_kv_cache()`（4a/4b/4c 落张量 + 4d 编译预热两个 collective_rpc；4c/4d 无 [KVC] 打印）。4 卡并行各自执行；以下用 **Worker_PP0_TP0（pid=94580）的真实日志**示例（另 3 卡同构，仅 pid/device/层段不同——PP0 两卡打 `layers.0~15`、PP1 两卡打 `layers.16~31`）。

**(a) 4a 分配 int8 字节池（开始横幅 + 16 层 dense，每层 K/V 两个独立池）**

```
INFO 10-02 16:25:40 [model_runner_v1.py:4326] [KVC][L1] ================ 物理侧 KV Cache 分配开始 ================
INFO 10-02 16:25:40 [model_runner_v1.py:4488] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.0.self_attn.attn: KVCacheTensor(size=3484155904 bytes = 3322.75MiB) -> K int8 1661.38MiB + V int8 1661.38MiB (alignment=2097152, device=npu:0)
INFO 10-02 16:25:40 [model_runner_v1.py:4488] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.1.self_attn.attn: KVCacheTensor(size=3484155904 bytes = 3322.75MiB) -> K int8 1661.38MiB + V int8 1661.38MiB (alignment=2097152, device=npu:0)
...（layers.2~14 同构, 每层一行; 另 3 卡同构, 仅 pid/device/层段不同）
INFO 10-02 16:25:45 [model_runner_v1.py:4488] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.15.self_attn.attn: KVCacheTensor(size=3484155904 bytes = 3322.75MiB) -> K int8 1661.38MiB + V int8 1661.38MiB (alignment=2097152, device=npu:0)
```

| 字段 | 值 | 验算（对照理论 §2.4 4a） |
|---|---|---|
| KVCacheTensor.size | 3,484,155,904 B = 3322.75MiB | = page 262,144 × num_blocks 13,291——**直接消费 (c) ③ 做编排产出的 KVCacheConfig** |
| K int8 池 | 1661.38MiB | = size ÷ 2（K/V 各半）= 1,742,077,952 B |
| V int8 池 | 1661.38MiB | 同上——**每层 K/V 两个独立的 int8 池（张量级拆分）** |
| alignment | 2,097,152（2MiB） | NPU 张量 2MiB 对齐 |
| device | npu:0（本卡常驻） | 16 层逐层打印，每层一行 |

> **为什么用 int8？**（理论 §2.4）：与 dtype 解耦——先按字节量申请，之后 4b reshape 时再 `view(dtype)` 转回 bf16，同一分配逻辑适配任意 dtype。

**(b) 4b 零拷贝 reshape（K/V 张量就位 + 完成横幅）**

```
INFO 10-02 16:25:45 [model_runner_v1.py:4926] [KVC][L1] vllm-ascend _reshape_kv_cache_tensors: model.layers.0.self_attn.attn (本组 16 层同形) -> K_cache shape=(13291, 128, 4, 128) dtype=torch.bfloat16 / V_cache shape=(13291, 128, 4, 128) dtype=torch.bfloat16, device=npu:0 (K/V 分离布局)
INFO 10-02 16:25:45 [model_runner_v1.py:4977] [KVC][L1] ================ 物理侧 KV Cache 分配完成 ================
```

对照理论 4b：int8 → dtype → shape 的 **view 零拷贝**（`raw.view(dtype).view(shape)` 普通路径，理论 §2.4），再 permute 成后端逻辑布局——全程无数据拷贝，int8 池地址即最终张量地址。`K_cache shape=(13291, 128, 4, 128) bf16` 四维含义：dim0=num_blocks（**block id == 张量行号**，理论 §5 桥接）、dim1=block_size 128、dim2=kv_heads 4（8÷TP2）、dim3=head_dim 128；`K/V 分离布局`——两组独立的 `(num_blocks, 128, 4, 128)` 张量，区别于上游 GPU 的 K/V packed 单张量（理论 §5 表）。

### 2.3 逻辑侧装配（EngineCore，8 行全量）

**自底向上**逐组件装配，每组件一条 `__init__完成：` 打印统一格式；

```
INFO 10-02 16:25:49 [kv_cache_manager.py:143] [KVC][L5] ================ 逻辑侧初始化开始 ================
INFO 10-02 16:25:49 [kv_cache_utils.py:217] [KVC][L2] FreeKVCacheBlockQueue.__init__完成：num_free_blocks=13291, 伪头尾哨兵 fake_free_list_head/tail(block_id=-1), 类型=FreeKVCacheBlockQueue
INFO 10-02 16:25:49 [block_pool.py:62] [KVC][L2] BlockHashToBlockMap.__init__完成：底容器 size=0, value=KVCacheBlock | dict[block_id→KVCacheBlock](不去重 append-only)
INFO 10-02 16:25:49 [block_pool.py:214] [KVC][L2] BlockPool.__init__完成：num_gpu_blocks=13291, 创建 KVCacheBlock × 13291 (block_id=0..13290), free_block_queue=FreeKVCacheBlockQueue(num_free_blocks=13290), cached_block_hash_to_block=BlockHashToBlockMap(size=0), null_block=KVCacheBlock(block_id=0, is_null=True), enable_caching=True, hash_block_size=128
INFO 10-02 16:25:49 [single_type_kv_cache_manager.py:95] [KVC][L3] FullAttentionManager.__init__完成：spec=FullAttentionSpec(block_size=128), scheduler_block_size=128, group_id=0, enable_caching=True, dcp×pcp=1×1, block_pool(num_gpu_blocks=13291)
INFO 10-02 16:25:49 [kv_cache_coordinator.py:462] [KVC][L4] UnitaryKVCacheCoordinator.__init__完成：单组直通, managers=['FullAttentionManager'], kv_cache_spec=FullAttentionSpec(block_size=128, page_size_bytes=262144), coordinator_block_size=128
INFO 10-02 16:25:49 [kv_cache_manager.py:178] [KVC][L5] KVCacheManager.__init__完成：coordinator=UnitaryKVCacheCoordinator, num_kv_cache_groups=1, managers=['FullAttentionManager'], block_pool(num_gpu_blocks=13291), enable_caching=True, max_model_len=8192, empty_kv_cache_blocks=KVCacheBlocks([],)
INFO 10-02 16:25:49 [kv_cache_manager.py:186] [KVC][L5] ================ 逻辑侧初始化完成 ================
```

### 2.4 启动期原生关键行（vllm 自带，非 [KVC] patch）

这一节收的是 **[KVC] 调试打印之外、vllm 自己打的结论性行**：启动期有任何一步（算规格→测预算→做编排→落张量→装配）出问题，服务起不来或数值对不上，这 6 行是排障的第一落点。按日志出现序：

```
(APIServer pid=94246) INFO 10-02 16:25:09 [utils.py:1404] Block size is set to 128 if prefix cache or chunked prefill is enabled.
(Worker_PP0_TP0 pid=94580) INFO 10-02 16:25:39 [worker.py:593] Available KV cache memory: 51.96 GiB
(EngineCore pid=94363) INFO 10-02 16:25:40 [kv_cache_utils.py:1771] GPU KV cache size: 1,701,248 tokens
(EngineCore pid=94363) INFO 10-02 16:25:40 [kv_cache_utils.py:1772] Maximum concurrency for 8,192 tokens per request: 207.67x
(EngineCore pid=94363) INFO 10-02 16:25:48 [core.py:369] init engine (profile, create kv cache, warmup model) took 10.71 s
(APIServer pid=94246) INFO:     Application startup complete.
```

（6 行分别位于 `llama-3-8b.log` :32 / :159 / :162 / :163 / :342 / :398；最后一行即就绪标志，16:24:58 run_all 启动 → 16:25:48 就绪共 50s）

逐行详解：

| 行号 | 谁打的（源码） | 说什么 / 怎么算 | 与 [KVC] 的对账 |
|---|---|---|---|
| :32 | APIServer（utils.py:1404） | **block_size=128**：开了 prefix cache / chunked prefill 时 vllm 把存储分块粒径定成 128（NPU 页对齐默认）——后面所有"满块 128 tokens、按 128 切哈希"的分母 | [KVC][L1] `K_cache=(num_blocks, 128, 4, 128)` 的 dim1=128 即此值 |
| :159 | Worker_PP0_TP0（worker.py:593） | worker 0 卡**实测可分配 KV 显存 51.96 GiB**（整卡可用 − 权重 − 激活 − warmup 峰值）；4 worker 各打 1 行，本卡只是第一行 | §2.1 (b) [KVC] `determine_available_memory` 的 51.96/51.97/51.92/51.93——同一测量、[KVC] 四卡一屏对比，原生行只逐卡逐行 |
| :162 | EngineCore（kv_cache_utils.py:1771） | **总 KV 容量 1,701,248 tokens** = num_blocks × block_size | 13,291 × 128 = 1,701,248——与 §2.1 (c) 最终 num_blocks 完全一致 |
| :163 | EngineCore（kv_cache_utils.py:1772） | **maximum concurrency 207.67x**：满载 8,192-token 长请求时的并发上限（理论上限，实际受连续批处理调度影响） | 1,701,248 ÷ 8,192 = 207.67——同源两行连算，max_model_len=8192 为分母 |
| :342 | EngineCore（core.py:369） | **init engine 10.71 s**：profile（=② 测预算的 dummy forward）+ create kv cache（=③ 做编排 + ④ 落张量）+ warmup 的总耗时 | [KVC] 各步（16:25:37 编排 → 16:25:49 逻辑侧完成）全落在这 10.71s 内——[KVC] 打印对启动时延的净增量可用此行与无 patch 版对比衡量 |
| :398 | APIServer（uvicorn） | **服务就绪标志**：路由全部挂载、uvicorn 开始收请求 | P 分界行号 399 与本行紧邻（:398 就绪标志，:399 即 P 入队）——§3 的 P/R 轨迹分界从这条起算（分界号由 curl_p_r.sh 运行时取行数，不落盘）；start.sh 的就绪探测就是 `grep 'Application startup complete' log/llama-3-8b.log` |

## 3. P 运行期日志讲解（log/kvc_p.log，61 行）

P：num_prompt_tokens=324（2 满块 + 尾 68），max_tokens=1——一次 prefill 即终态（TERM），验证"缓冲 2 块"。

> 以下按理论 `../0_runtime_sequence.md` §4 分阶段详解的时序组织：**入队 → 首次调度（①前缀查找 + ②allocate_slots）→ GPU 写 KV → （③ decode）→ ④ 结束释放**。GPU 写 KV 与 P 的 decode 环节无 [KVC] 打印（正确性由 KVS TERM 归档离线比对兜底验证，§3.3）。

| 理论时序（0_runtime_sequence §4） | 日志小节 | P 实测要点 |
|---|---|---|
| §4.1 入队（预计算链式哈希 → WAITING） | §3.1 | 2 个满块哈希入队（NONE_HASH 起链） |
| §4.2 首次调度 ① get_computed_blocks（前缀查找） | §3.1 | 冷缓存第 1 hash 即 MISS，hit_length=0 |
| §4.2 首次调度 ② allocate_slots（S1~S4） | §3.2 | S1 需 3 vs 可用 13290 → S3 [1,2,3] → S4 双满块入表 |
| §4.3 GPU 写 KV（forward） | —（无 [KVC] 打印） | 写 324 token K/V；物理正确性由 KVS TERM 归档离线验证（§3.3） |
| ③ decode | — | P max_tokens=1，首 token 即终态，无 decode 循环 |
| ④ 结束释放（§4.5） | §3.4 | [3,2,1] 逆序 append 队尾（LRU 保护） |

### 3.1 入队 + ①前缀查找（对照理论 §4.1、§4.2-①）

```
INFO 10-02 16:25:54 [request.py:184] [KVC][ENQ] ======== 入队 ========
INFO 10-02 16:25:54 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=c727e7e5d059
INFO 10-02 16:25:54 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=c727e7e5d059, tokens=128 -> BlockHash=d99d77210268
INFO 10-02 16:25:54 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-a0c5ce849a3b6868-0-b3cb18f2) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['c727e7e5d059', 'd99d77210268']
INFO 10-02 16:25:54 [request.py:193] [KVC][ENQ] ======== 入队完成 ========
INFO 10-02 16:25:54 [kv_cache_manager.py:222] [KVC][L5] ======== 前缀查找 ========
...（冷缓存: 第 1 块即 MISS 断链, hit_length=0, 返回 blocks=[[]]）
INFO 10-02 16:25:54 [block_pool.py:89] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=c727e7e5d059, group_id=0) -> MISS
INFO 10-02 16:25:54 [block_pool.py:247] [KVC][L2] 前缀查找 BlockPool.get_cached_block: MISS BlockHash=c727e7e5d059 (group_id=0 未命中) -> None
INFO 10-02 16:25:54 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 1 块 MISS: BlockHash=c727e7e5d059 -> break
```

### 3.2 ②分配 S1~S4 全链（对照理论 §4.2-②；S1 段自洽实录）

总横幅(:393) → 进入(:399, 全 7 字段) → 分配布局(:409) → **S1 子步横幅(:417)** → 两次外层探问(:188×2) → S1 汇总值(:462)，随后 S2/S3/S4 四子步下钻逐层展开：

```
INFO 10-02 16:25:54 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 10-02 16:25:54 [kv_cache_manager.py:399] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_new_tokens=324(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=0(comp), request.num_tokens=324, delay_cache_blocks=False
INFO 10-02 16:25:54 [kv_cache_manager.py:409] [KVC][L5] 分配布局: |<comp>=0 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=324 |<lookahead>=0| num_local_computed_tokens=0 total_computed_tokens=0 to_be_computed=324
INFO 10-02 16:25:54 [kv_cache_manager.py:417] [KVC][L5] --- S1: 容量检查---
INFO 10-02 16:25:54 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 10-02 16:25:54 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 10-02 16:25:54 [kv_cache_manager.py:462] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13290 块 (free=13290 - reserved=0)
INFO 10-02 16:25:54 [kv_cache_manager.py:496] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 10-02 16:25:54 [kv_cache_manager.py:500] [KVC][L5] --- S3: 新块分配 ---
INFO 10-02 16:25:54 [block_pool.py:416] [KVC][L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=13287
INFO 10-02 16:25:54 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_tokens=324, block_size=128, 需 3 块 - 已有 0 = 新分配 3 块 [1, 2, 3], 持有 req_blocks=[1, 2, 3]
INFO 10-02 16:25:54 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_tokens=324 -> [[1, 2, 3]]
INFO 10-02 16:25:54 [kv_cache_manager.py:511] [KVC][L5] S3 allocate_new_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_tokens_need_slot=324 -> 新块 [1, 2, 3]
INFO 10-02 16:25:54 [kv_cache_manager.py:538] [KVC][L5] --- S4: 满块入缓存 ---
INFO 10-02 16:25:54 [single_type_kv_cache_manager.py:357] [KVC][L3] S4 SingleTypeKVCacheManager.cache_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_tokens=324, block_size=128, 已缓存 0 块 -> 满块数 2
INFO 10-02 16:25:54 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=c727e7e5d059, group_id=0) <- KVCacheBlock(block_id=1), map size=1
INFO 10-02 16:25:54 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=d99d77210268, group_id=0) <- KVCacheBlock(block_id=2), map size=2
INFO 10-02 16:25:54 [block_pool.py:345] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2 新满块 2 块 block_ids=[1, 2] 入 BlockHashToBlockMap (num_cached_blocks 0 -> 2, group_id=0, map size=2)
INFO 10-02 16:25:54 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_computed_tokens=324
INFO 10-02 16:25:54 [kv_cache_manager.py:544] [KVC][L5] S4 cache_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_tokens_to_cache=324
INFO 10-02 16:25:54 [kv_cache_manager.py:548] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([1, 2, 3],)), req=cmpl-a0c5ce849a3b6868-0-b3cb18f2 当前完整 block_table=([1, 2, 3],)
INFO 10-02 16:25:54 [kv_cache_manager.py:554] [KVC][L5] ======== 分配完成 ========
```

要点：324 tokens → S3 需 3 块（`需 3 块 - 已有 0 = 新分配 3 块 [1, 2, 3]`）；S4 只把 2 个满块入哈希表（324 // 128 = 2，`insert` ×2），块 3 为尾块（68/128）不入表。

### 3.3 KVS 物理归档（实装侧校验点，理论时序之外；TERM ×4 worker，横幅两行式）

TERM 判定成立（worker 内 written ≥ prompt_len + max_tokens − 1，即该请求最后一次写卡已完成）时，各 worker 把该请求全部物理 KV 块**整块原样**归档：每层每块 `kt[blk].cpu().clone()`（位级快照，**含未写槽位**），同步快照完成后交后台 daemon 线程 `torch.save` 落盘 `.pt`，不阻塞推理。日志仅横幅两行（替代 v1 打印版每请求 76 行的逐层统计），逐层逐块粒度全部由 `.pt` 内 meta + 离线查看器（`scripts/inspect_kv_tensors.py`）承载：

```
INFO 10-02 16:25:54 [model_runner_v1.py:2426] [KVC][KVS] 物理KV原样归档已启用: KVC_SAVE_KV=1, 输出目录=/a3_inference/itask/workdir/wsl02075301/kvc/tensors (相对 worker cwd)
INFO 10-02 16:25:54 [model_runner_v1.py:2557] [KVC][KVS] ======== 开始保存物理tensor worker=PP0_TP0 dev=npu:0 TERM seq=1 req尾8=b3cb18f2: K_cache/V_cache(双独立张量池, 每块 shape=(128,4,128) torch.bfloat16, 含未写槽位) 16 层 × 3 块 blk=[1, 2, 3] cov=[128, 128, 68] -> kv_pp0tp0_s1_b3cb18f2.pt ========
INFO 10-02 16:25:54 [model_runner_v1.py:2612] [KVC][KVS] ======== 完成保存物理tensor worker=PP0_TP0 dev=npu:0 seq=1 req尾8=b3cb18f2: kv_pp0tp0_s1_b3cb18f2.pt (K_cache/V_cache, 16 层 × 3 块, 12612247 B = 12.0 MiB) 落盘 34.2 ms ========
```

另 3 卡同构（PP0_TP1 / PP1_TP0 / PP1_TP1 → npu:1/2/3，flush 33.4~36.0 ms）。启用横幅每 worker 首个前向后各 1 行——P 段 [KVS] 合计 12 行（4 启用 + 4 开始 + 4 完成），双请求全程共 20 行。两行横幅承载字段：

| 横幅 | 打印时点 | 承载字段 |
|---|---|---|
| 开始（:2557） | 归档启动（同步） | worker/dev/TERM/seq/req尾8——**保存什么**：K_cache/V_cache（双独立张量池）· 每块 shape=(128,4,128) bf16 · 16 层 × 3 块 · blk/cov → 目标文件名 |
| 完成（:2612） | 落盘完成（后台线程异步） | **save 完成回执**：文件名 + 12612247 B (12.0 MiB) + 落盘 34.2 ms（失败兜底 ARCHIVE-FAIL 单行，绝不影响服务） |

**归档时点语义**：free() 在 EngineCore 进程、物理张量在各 worker 进程，跨进程不可直读——以“最后一次写卡完成”为等价时点。本轮 kvc_p.log 实测：4 卡开始横幅 :40~48 全部先于 L5 释放横幅 :55，同步快照先于块归还；LATE（结束后兜底）不归档仅一行告警，本轮 0 次。

**.pt 格式（schema kvt4-raw）**：`{"K": [16 层, {块号: (128,4,128) bf16}], "V": 同构, "meta": {pp/tp/seq/request_id/p_tok/w_tok/final/block_table/cov/layers/layer_ids/kv_heads/head_dim/dtype/dev/ts}}`；文件名 `kv_pp{pp}tp{tp}_s{seq}_{rid尾8}.pt`；层序为 worker 本地序（全局层号 = pp×16 + 本地序，4 worker 联合覆盖 32 层）。

**离线对账**（原始输出存 log/inspect_*.out）：深查见块 3 尾块 `未写槽位[60行] K/V n=30720 mean=0`（未写槽位全零，块池新建基线为零）；L00 region 统计按 v1 打印口径重算，与 v1（09-29 打印版轮）记录值**4 位完全相同**（K: mean=-0.0219 std=1.394 min=-10.38 max=10.81；V: 0.0007659/0.03501/-0.2539/0.3223）——归档与打印位级等价、跨 run 确定（`log/inspect_stats_crosscheck.out`）。

### 3.4 ④调度提交与结束释放（对照理论 §4.5）

async_scheduler 每步输出后的独立提交段（与分配 S4 明确区分），随后逆序释放、带哈希块 LRU 归队（KVS 开始/完成横幅 :39~50 已先行，见 §3.3）：

```
INFO 10-02 16:25:55 [kv_cache_manager.py:691] [KVC][L5] ======== 调度提交(非分配 S4) ========
INFO 10-02 16:25:55 [kv_cache_manager.py:692] [KVC][L5] 提交 cache_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_computed_tokens=324 (async 步末输出路径: 本步已算 token 提交入缓存)
INFO 10-02 16:25:55 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, num_computed_tokens=324
INFO 10-02 16:25:55 [kv_cache_manager.py:698] [KVC][L5] ======== 提交完成 ========
INFO 10-02 16:25:55 [kv_cache_manager.py:566] [KVC][L5] ======== 释放 ========
INFO 10-02 16:25:55 [kv_cache_manager.py:568] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, 释放前持有 block_table=([1, 2, 3],)
INFO 10-02 16:25:55 [single_type_kv_cache_manager.py:405] [KVC][L3] 释放 SingleTypeKVCacheManager.free: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2, 持有 blocks=[1, 2, 3] (reversed 后释放)
INFO 10-02 16:25:55 [block_pool.py:521] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(3, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 3 块 [3, 2, 1], append_n -> 队尾(LRU保护)
INFO 10-02 16:25:55 [kv_cache_utils.py:391] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[3, 2, 1]), num_free_blocks=13290
INFO 10-02 16:25:55 [kv_cache_coordinator.py:299] [KVC][L4] 释放 KVCacheCoordinator.free: req=cmpl-a0c5ce849a3b6868-0-b3cb18f2 已逐组下放第3层释放
INFO 10-02 16:25:55 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

### 3.5 响应核对

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| P | 1 | length | `"为了"` |

## 4. R 运行期日志讲解（log/kvc_r.log，719 行）

R：num_prompt_tokens=486 = 3 满 + 第 4 块 102/128，max_tokens=35——完整五块生命周期（复用 2 + prefill 补 1 满 1 尾 + decode 填满尾块 + 步 27 跨界申请第 5 块）。request_id=cmpl-b5d15266be8d362d-0-81516ab1。

> 按理论 `../0_runtime_sequence.md` §4 时序组织：R 覆盖**全部阶段**（理论时序图以 70 token/16 块示例推演，此处为 486 token/128 块全量实测）。

| 理论时序（0_runtime_sequence §4） | 日志小节 | R 实测要点 |
|---|---|---|
| §4.1 入队 | §4.1 | 3 个满块哈希入队（P 链延伸 + 追问句新段） |
| §4.2 ① 前缀查找（get_computed_blocks） | §4.1 | HIT×2 → 第 3 hash MISS 断链，hit_length=2×128=256 |
| §4.2 ② allocate_slots（S1~S4） | §4.2 | S2 touch[(1,1),(2,1)] 零拷贝复用；S3 [4,5]；S4 新满块入表 |
| §4.3 GPU 写 KV（forward） | —（无 [KVC] 打印） | 跨 4 块写 486 token（复用块不重算） |
| §4.4 ③ decode·情况 A（需 0 块） | §4.3 | 33 步同构闭合链 + 每步调度提交 |
| §4.4 ③ decode·情况 B（需 1 块） | §4.4 | 步 27 跨界：新块 [6]，刚填满的块 5 入表 |
| KVS 物理归档与离线互证（实装侧，理论之外） | §4.5 | TERM 五块归档 8/8 + 公共块 [1,2] 逐位相等 = 零拷贝证据 |
| ④ 结束释放（§4.5） | §4.6 | 五块逆序归队 [6,5,4,2,1] |

### 4.1 入队 + ①前缀查找（对照理论 §4.1、§4.2-①；HIT×2 后 MISS 断链）

```
INFO 10-02 16:26:01 [request.py:184] [KVC][ENQ] ======== 入队 ========
INFO 10-02 16:26:01 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=c727e7e5d059
INFO 10-02 16:26:01 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=c727e7e5d059, tokens=128 -> BlockHash=d99d77210268
INFO 10-02 16:26:01 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=d99d77210268, tokens=128 -> BlockHash=366a05d2bbcc
INFO 10-02 16:26:01 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-b5d15266be8d362d-0-81516ab1) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 BlockHash × 3: ['c727e7e5d059', 'd99d77210268', '366a05d2bbcc']
INFO 10-02 16:26:01 [request.py:193] [KVC][ENQ] ======== 入队完成 ========
INFO 10-02 16:26:01 [block_pool.py:75] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=c727e7e5d059, group_id=0) -> HIT KVCacheBlock(block_id=1)
INFO 10-02 16:26:01 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 1 块 HIT: BlockHash=c727e7e5d059 -> cached blocks=[1]
INFO 10-02 16:26:01 [block_pool.py:75] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=d99d77210268, group_id=0) -> HIT KVCacheBlock(block_id=2)
INFO 10-02 16:26:01 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 2 块 HIT: BlockHash=d99d77210268 -> cached blocks=[2]
INFO 10-02 16:26:01 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 3 块 MISS: BlockHash=366a05d2bbcc -> break
INFO 10-02 16:26:01 [kv_cache_coordinator.py:495] [KVC][L4] 前缀查找 UnitaryKVCacheCoordinator.find_longest_cache_hit 返回: hit_blocks=[[1, 2]], hit_length=256
```

命中 2 块 → hit_length=2×128=256；第 3 个 hash 是 R 追问句新内容（冷），断链即止。

### 4.2 ②prefill 完整链（对照理论 §4.2-②；S2 touch 复用 + S3 补 2 块 + S4 新满块入表）

```
INFO 10-02 16:26:01 [kv_cache_manager.py:482] [KVC][L5] --- S2: touch 命中块 ---
INFO 10-02 16:26:01 [kv_cache_manager.py:484] [KVC][L5] S2 allocate_new_computed_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, new_computed_blocks=[[1, 2]]
INFO 10-02 16:26:01 [block_pool.py:496] [KVC][L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
INFO 10-02 16:26:01 [kv_cache_manager.py:500] [KVC][L5] --- S3: 新块分配 ---
INFO 10-02 16:26:01 [block_pool.py:416] [KVC][L2] S3 BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13286
INFO 10-02 16:26:01 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens=486, block_size=128, 需 4 块 - 已有 2 = 新分配 2 块 [4, 5], 持有 req_blocks=[1, 2, 4, 5]
INFO 10-02 16:26:01 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=366a05d2bbcc, group_id=0) <- KVCacheBlock(block_id=4), map size=3
INFO 10-02 16:26:01 [block_pool.py:345] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1 新满块 1 块 block_ids=[4] 入 BlockHashToBlockMap (num_cached_blocks 2 -> 3, group_id=0, map size=3)
INFO 10-02 16:26:01 [kv_cache_manager.py:548] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([4, 5],)), req=cmpl-b5d15266be8d362d-0-81516ab1 当前完整 block_table=([1, 2, 4, 5],)
INFO 10-02 16:26:01 [kv_cache_manager.py:554] [KVC][L5] ======== 分配完成 ========
```

要点：**touch 零拷贝**——`S2 BlockPool.touch: blocks=[(1, 1), (2, 1)]`（block_id, ref_cnt），直接复用 P 留下的块 1/2；S3 = cdiv(486,128)−2 = 2 新块 [4,5]；S4 把追问句恰好填满的块 4 入表（map 2→3）。随后的 SchedulerOutput 携带 `new_block_ids_to_zero=[4, 5]`（对应理论 §4.3“调度输出附清零块 id”）——GPU forward 写 KV 无 [KVC] 日志，物理正确性由 §4.5 的 KVS 归档离线比对实测验证。

### 4.3 ③decode 无块步——理论“情况 A：需 0 块”（对照 §4.4；33 步同构，完整实录）

理论 §4.4 把 decode 每步 allocate_slots 分两种情况：**情况 A**·当前块未满需 0 块（token 直接续写）/ **情况 B**·已满需 1 块（token 落进下一块）。本节是情况 A 的每步实录；decode 步 1 起的每一步（除步 27 外）都是这条闭合链——S1 如实打“需分配 0”，S3 切换“无需分配新块”文案，尾部紧跟本步调度提交：

```
INFO 10-02 16:26:01 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 10-02 16:26:01 [kv_cache_manager.py:399] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-b5d15266be8d362d-0-81516ab1, num_new_tokens=1(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=486(comp), request.num_tokens=486, delay_cache_blocks=False
INFO 10-02 16:26:01 [kv_cache_manager.py:409] [KVC][L5] 分配布局: |<comp>=486 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=1 |<lookahead>=0| num_local_computed_tokens=486 total_computed_tokens=486 to_be_computed=1
INFO 10-02 16:26:01 [kv_cache_manager.py:417] [KVC][L5] --- S1: 容量检查---
INFO 10-02 16:26:01 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens=487 -> 需分配 0 块(含touch需腾挪的块)
INFO 10-02 16:26:01 [kv_cache_manager.py:462] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 0 块 vs 可用 13286 块 (free=13286 - reserved=0)
INFO 10-02 16:26:01 [kv_cache_manager.py:496] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 10-02 16:26:01 [kv_cache_manager.py:502] [KVC][L5] --- S3: 无需分配新块 ---
INFO 10-02 16:26:01 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens=487 -> [[]]
INFO 10-02 16:26:01 [kv_cache_manager.py:516] [KVC][L5] S3 块未满, 无需分配新块 (req=cmpl-b5d15266be8d362d-0-81516ab1, num_new_tokens=1)
INFO 10-02 16:26:01 [kv_cache_manager.py:538] [KVC][L5] --- S4: 满块入缓存 ---
INFO 10-02 16:26:01 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_computed_tokens=486
INFO 10-02 16:26:01 [kv_cache_manager.py:544] [KVC][L5] S4 cache_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens_to_cache=486
INFO 10-02 16:26:01 [kv_cache_manager.py:548] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([],)), req=cmpl-b5d15266be8d362d-0-81516ab1 当前完整 block_table=([1, 2, 4, 5],)
INFO 10-02 16:26:01 [kv_cache_manager.py:554] [KVC][L5] ======== 分配完成 ========
INFO 10-02 16:26:01 [kv_cache_manager.py:691] [KVC][L5] ======== 调度提交(非分配 S4) ========
```

### 4.4 ③decode 步 27 跨界——理论“情况 B：需 1 块”（对照 §4.4；第 512 个 token 触发第 5 块申请）

尾块 102/128 被 decode 逐步填满——步 27 时 num_computed_tokens=512，S1 汇总值切换为“需分配 1 块”，S3 弹出新块 [6]，S4 把刚填满的块 5 入表（map 3→4）：

```
INFO 10-02 16:26:01 [kv_cache_manager.py:399] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-b5d15266be8d362d-0-81516ab1, num_new_tokens=1(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=512(comp), request.num_tokens=512, delay_cache_blocks=False
INFO 10-02 16:26:01 [kv_cache_manager.py:409] [KVC][L5] 分配布局: |<comp>=512 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=1 |<lookahead>=0| num_local_computed_tokens=512 total_computed_tokens=512 to_be_computed=1
INFO 10-02 16:26:01 [kv_cache_manager.py:417] [KVC][L5] --- S1: 容量检查---
INFO 10-02 16:26:01 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens=513 -> 需分配 1 块(含touch需腾挪的块)
INFO 10-02 16:26:01 [kv_cache_manager.py:462] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 1 块 vs 可用 13286 块 (free=13286 - reserved=0)
INFO 10-02 16:26:01 [kv_cache_manager.py:496] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 10-02 16:26:01 [kv_cache_manager.py:500] [KVC][L5] --- S3: 新块分配 ---
INFO 10-02 16:26:01 [block_pool.py:416] [KVC][L2] S3 BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=13285
INFO 10-02 16:26:01 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens=513, block_size=128, 需 5 块 - 已有 4 = 新分配 1 块 [6], 持有 req_blocks=[1, 2, 4, 5, 6]
INFO 10-02 16:26:01 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens=513 -> [[6]]
INFO 10-02 16:26:01 [kv_cache_manager.py:511] [KVC][L5] S3 allocate_new_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens_need_slot=513 -> 新块 [6]
INFO 10-02 16:26:01 [kv_cache_manager.py:538] [KVC][L5] --- S4: 满块入缓存 ---
INFO 10-02 16:26:01 [single_type_kv_cache_manager.py:357] [KVC][L3] S4 SingleTypeKVCacheManager.cache_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1, num_tokens=512, block_size=128, 已缓存 3 块 -> 满块数 4
INFO 10-02 16:26:01 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=36f5e361457d, group_id=0) <- KVCacheBlock(block_id=5), map size=4
INFO 10-02 16:26:01 [block_pool.py:345] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-b5d15266be8d362d-0-81516ab1 新满块 1 块 block_ids=[5] 入 BlockHashToBlockMap (num_cached_blocks 3 -> 4, group_id=0, map size=4)
```

### 4.5 KVS 物理归档与离线互证（实装侧校验点，理论时序之外；4 worker × 横幅两行，五块全景）

R TERM 同样每 worker 横幅两行：blk=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8]（region=520/520 = 4×128 + 8；w_tok=520 = 486 + 34，第 35 个输出 token 仅采样不写卡），每 worker 21,019,411 B = 20.0 MiB：

```
INFO 10-02 16:26:01 [model_runner_v1.py:2557] [KVC][KVS] ======== 开始保存物理tensor worker=PP0_TP0 dev=npu:0 TERM seq=2 req尾8=81516ab1: K_cache/V_cache(双独立张量池, 每块 shape=(128,4,128) torch.bfloat16, 含未写槽位) 16 层 × 5 块 blk=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8] -> kv_pp0tp0_s2_81516ab1.pt ========
INFO 10-02 16:26:01 [model_runner_v1.py:2612] [KVC][KVS] ======== 完成保存物理tensor worker=PP0_TP0 dev=npu:0 seq=2 req尾8=81516ab1: kv_pp0tp0_s2_81516ab1.pt (K_cache/V_cache, 16 层 × 5 块, 21019411 B = 20.0 MiB) 落盘 58.9 ms ========
```

**时序细节**（与 P 的差异，kvc_r.log 行号）：4 卡开始横幅 :697~704 仍先于 L5 释放横幅 :709（同步快照先行）；但完成横幅 :716~719 落在释放横幅**之后**——R 的 flush 45.7~71.6 ms 由后台线程执行，EngineCore 已先行归还块，落盘早晚互不影响（数据在开始横幅前的同步 clone 已位级脱离物理池）。

**离线互证**（log/inspect_*.out）——归档取代 v1 的“K 示值跨请求一致”打印证据，并升级为逐位断言：

| 验证项 | 结果 | 原始输出 |
|---|---|---|
| 归档结构与 worker 覆盖 | 8/8 PASS：{(pp,tp)} 全 4 组合 × P/R；同 seq 四 worker 块表/cov/p_tok/w_tok 一致 | inspect_list.out |
| **前缀复用零篡改** | P(s1) vs R(s2) 公共块 [1, 2] 在全 4 worker × 16 层 K/V 前 min(cov) 有效行**逐位相等**（bf16 位模式，每 worker 2,097,152 元素）——命中块即 P 归档的同批物理字节，零拷贝铁证 | inspect_compare.out |
| 重算一致性（ULP 级） | 同 token 段 P.b3[:68]（324-tok prefill）vs R.b4[:68]（486-tok prefill）：L00 仅 K 2/34816、V 18/34816 元素位翻转（Pearson=1.000000）；L15 经 0~14 层残差流放大呈大面积位级漂移（77%/88%），Pearson 仍 ≥0.99989——bf16 数值敏感性的自然现象，非数据错误 | inspect_recalc_diff.out |
| 首块首 token 跨证 | K 前 3 值 [0.5078, 0.9336, 0.9219] 与 v1（09-29）打印记录完全一致——五轮实验（v1→v2→v2.1→v2.2→v2.3）位级确定性互证 | inspect_stats_crosscheck.out |

### 4.6 ④结束释放（对照理论 §4.5：五块逆序释放与计数自检）

```
INFO 10-02 16:26:01 [kv_cache_manager.py:566] [KVC][L5] ======== 释放 ========
INFO 10-02 16:26:01 [kv_cache_manager.py:568] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-b5d15266be8d362d-0-81516ab1, 释放前持有 block_table=([1, 2, 4, 5, 6],)
INFO 10-02 16:26:01 [single_type_kv_cache_manager.py:405] [KVC][L3] 释放 SingleTypeKVCacheManager.free: req=cmpl-b5d15266be8d362d-0-81516ab1, 持有 blocks=[1, 2, 4, 5, 6] (reversed 后释放)
INFO 10-02 16:26:01 [block_pool.py:521] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(6, 0), (5, 0), (4, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 5 块 [6, 5, 4, 2, 1], append_n -> 队尾(LRU保护)
INFO 10-02 16:26:01 [kv_cache_utils.py:391] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[6, 5, 4, 2, 1]), num_free_blocks=13290
INFO 10-02 16:26:01 [kv_cache_coordinator.py:299] [KVC][L4] 释放 KVCacheCoordinator.free: req=cmpl-b5d15266be8d362d-0-81516ab1 已逐组下放第3层释放
INFO 10-02 16:26:01 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

**全程计数自检**（grep 即可复验）：`调度提交` P=1 / R=**35**（每 decode 步一次）；[KVS] 横幅 P=**12**（4 启用 + 4 开始 + 4 完成）/ R=**8**（4 开始 + 4 完成）；S1 汇总值 R = **33×0 + 1×1 + 1×4**（无块步 33 + 跨界 1 + prefill 1）；哈希链 `c727e7e5d059 → d99d77210268 → 366a05d2bbcc`（+ decode 段 `36f5e361457d`）。

### 4.7 响应核对

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| R | 35 | length | `://www.zhihu.com/question/404202526\n1. 什么是前缀缓存…`（中文贪心续写） |

## 5. 实证结论

1. **patch 174 行验证通过**：8/8 应用（vllm 155 + vllm-ascend 19 = [L1] 8 + [KVS] 11，[KVC] 计数与 py_compile 均过）；实验后 revert 8 文件 [KVC] 归零、两仓库 git 0 改动——**补丁可双向复现，容器源码始终未污染**。
2. **S1 段完整自洽**：子步横幅先行（:417）→ 两次外层探问下钻（coordinator:188 ×2）→ 汇总值（:462）——S1 语义日志全部在子步横幅之内。
3. **S1~S4 全子步可观测**：无块步打出完整四段子步（S1 需分配 0 / S2 无前缀 / S3 无需分配 / S4 维护），R 33 个无块步全部闭合。
4. **KVS 横幅两行式的可读性**：每请求每 worker 仅 2 行（开始声明 + 完成回执），**不随层×块增长**（v1 打印版 76 行/请求、v2.1 逐块版 532 行/轮 → v2.3 每轮 20 行）；横幅字段与 .pt meta / manifest 全链对账 16/16 PASS（`log/inspect_savelog_audit.out`：块表/cov vs meta 8/8、字节 vs manifest 8/8、flush 33.4~71.6 ms 后台不阻塞、归档时序先于释放）。
5. **KV 布局与零拷贝实证**：归档张量直接承载布局证据——每块 (128,4,128) bf16、K/V 双独立张量池、未写槽位全零（P b3[68:128] 与 R b6[8:128]）；前缀复用 = 同批物理字节（公共块 [1,2] 四 worker × 16 层 × 2 池逐位相等）；L00 统计跨 run 与 v1（09-29）打印记录 4 位相同——五轮实验位级确定性互证。
6. **容器回收闭环**：服务已杀（0 进程）、8 补丁 revert 归零、两仓库 git 0 改动、.orig/打包残留清理。

## 6. 实测产物

产物（log/ + tensors/，容器时钟 2026-10-02 16:24~16:27，本轮 rid 尾8：P=b3cb18f2 / R=81516ab1）：

| 产物 | 说明 |
|---|---|
| `log/llama-3-8b.log`（1211 行） | 服务全量日志（启动 :1~398 + P :399~465 + R :466~1211，带进程前缀完整版） |
| `log/kvc_startup.log`（172 行） | 启动期 [KVC] 拆解轨迹（CFG 88 + L1 76 + 逻辑侧 8） |
| `log/kvc_p.log`（61 行）/ `log/kvc_r.log`（719 行） | P / R 运行期 [KVC]+[KVS] 拆解轨迹 |
| `log/kvs_archive_lines.log`（20 行） | [KVS] 横幅留痕（4 启用 + 8 开始 + 8 完成） |
| `log/curl_screen.log` + `req_*.json` / `resp_*.json` | curl 命令与响应打屏（用例设计见 docs/2） |
| `log/inspect_*.out`（9 个） | 离线分析原始输出（列表 / 深查×2 / 切片×2 / 比对 / ULP 重算 / v1 口径对账 / 横幅审计） |
| `log/run_all_screen.log` | 容器侧一键编排留痕（patch → serve → curl → 初检） |
| `tensors/kv_pp?tp?_s?_*.pt` **× 8** + `manifest.json` | 物理 KV 归档（P 12.0 MiB + R 20.0 MiB 每 worker；fetch 后 manifest md5 8/8 OK） |

（P/R 分界行号 399 / 466 由 curl_p_r.sh 运行时确定，不落盘）
