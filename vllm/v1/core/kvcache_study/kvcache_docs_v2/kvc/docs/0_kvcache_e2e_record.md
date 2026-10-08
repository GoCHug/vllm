# KVCache 调试体系实验 E2E 全记录

## 0. 总览

### 0.1 文档导读

本文是 `[KVC]` 调试体系的**完整实验文档**（用例设计与端到端验证记录合并而成），章节安排：

| 章节 | 内容 |
|---|---|
| **§1 实验设计** | 启动期打印（CFG/L1/逻辑侧三段 172 行）→ 运行期用例请求设计（P/R 双请求与五块生命周期推演）→ 运行期打印（入队 → 前缀查找 → 分配 S1~S4 → 调度提交 → KVS 归档 → 释放 六步） |
| **§2 实验流程与复现** | 容器 / 补丁 / 服务 / 产物回收全步骤 |
| **§3 启动期日志讲解**（172 行） | 服务启动初始化：配置侧 88 行 CFG、物理侧 76 行 L1、逻辑侧装配 8 行 |
| **§4 P 运行期日志讲解**（61 行） | P 请求全生命周期：入队 / 前缀查找 / 分配 S1~S4 / 物理归档 / 释放 |
| **§5 R 运行期日志讲解**（719 行） | R 请求全生命周期及 8 份物理 KV 归档的离线互证：HIT×2 后 MISS 断链、prefill 补 2 块、decode 第 27 步跨界第 5 块、五块逆序释放 |
| **§6 实证结论** | 补丁双向复现、S1~S4 全子步可观测、KVS 横幅对账、KV 布局与零拷贝位级互证等六条结论 |
| **§7 实测产物** | `logs/` 四子目录 + `tensors/` 8 份物理 KV 归档清单 |

### 0.2 实测环境

| 项 | 值 |
|---|---|
| Pod | gggtest（a3 · 4× Ascend910 · PP2TP2 · 当日 Running） |
| 模型 | Meta-Llama-3-8B（`modelhub_74000048_meta-llama-3-8b-148700128_20260921221233`，32 层 / kv_heads 8 / TP2 下本地 4 / block_size 128）https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B |
| 软件栈 | vllm 0.23.0 + vllm-ascend 0.23.0（`/vllm-workspace/`） |
| 进程 | APIServer pid=5848 · EngineCore pid=5969 · Worker pid=6070~6073（PP0_TP0 / PP0_TP1 / PP1_TP0 / PP1_TP1） |
| 实测时间 | 2026-10-03 15:16~15:18（log 内时间戳，容器时钟 UTC-8） |

## 1. 实验设计

> 调试体系共注入 **174 行 `[KVC]`**（grep 口径 = vllm 155 + vllm-ascend 19，含注释行；三级横幅 + 阶段前缀贯穿全程）。本章按实验的时间顺序组织三块设计：**§1.1 启动期打印**与**§1.3 运行期打印**规定注入的日志在两个阶段分别输出什么，**§1.2 运行期用例请求的设计**规定用 P/R 两个请求去触发哪些代码路径。

### 1.1 启动期打印

服务启动时一次性输出 **172 行 `[KVC]`**，三段各带 `================` 开始/完成横幅：

| 段 | 层标签 | 行数 | 核心内容 |
|---|---|---|---|
| 配置侧 | CFG | 88 | **① 算规格（紧跟 get_kv_cache_specs() 调用）** → ② 各 worker 可用 KV 显存 → 逐 worker `KVCacheConfig`（num_blocks/组数/张量数）→ 逐张量 size/shared_by → 最终 scheduler 侧 min 对齐 |
| 物理侧 | L1 | 76 | 每层 KVCacheTensor 拆 **K int8 池 + V int8 池**两张独立张量（2MiB 对齐）；reshape 后 K_cache=V_cache=(num_blocks, 128, 4, 128) bf16，block id 即 dim0 行号 |
| 逻辑侧 | L2~L5 | 8 | 自底向上逐组件 `__init__完成：`——L2 空闲队列（伪头尾哨兵）/ L2 BlockHashToBlockMap / L2 BlockPool（null 块摘取）/ L3 manager / L4 单组直通 coordinator / L5 门面 |

### 1.2 运行期用例请求的设计

> R 的 prompt 设计为 **486 tokens（3 个满块 + 第 4 块 102/128，非恰好边界）**：prefill 复用 2 块后新申请 **2 块（1 满 + 1 尾）**；decode **前 26 步填满尾块、第 27 步跨界申请第 5 块**；max_tokens=35。

#### 用例总览

| # | 请求 | prompt | max_tokens | 验证目标 |
|---|---|---|---|---|
| P | "种缓存" | 394 字 → **324 tokens**（2 满 + 尾 68） | 1 | **缓冲 2 块**：满块带哈希入缓存表 |
| R | P 全文 + 加长追问句 | 591 字 → **486 tokens = 3 满 + 第 4 块 102/128** | **35** | **① 复用 2 块（第 3 hash MISS 断链）② 申请 2 = 1 满 + 1 尾 ③ decode 前 26 步填尾块、第 27 步跨界第 5 块** |

#### 设计原理

| prompt | 字符 | tokens | 满块结构（block_size=128） |
|---|---|---|---|
| P | 394 | **324** | 2 满（256）+ 尾 68/128 |
| R | 591 | **486 = 3 满 + X** | 前 256 与 P 一致 → 复用 2；X = 486−384 = **102** |

推演：

1. **复用 2 块 + 断链**：入队满 hash × 3；`max_cache_hit_length = 485 → 3` 个查找：1、2 HIT，**第 3 个 MISS → break**
2. **prefill 申请 2 块**：`cdiv(486,128)=4 − 2 = 2`
3. **decode 跨界**：前 `26` 步填尾块 102→128；**步 27（需分配 1 块）跨界**；步 28~34 落第 5 块 8/128

#### curl 命令与请求体（可直接复制）

##### 请求 1（P：缓冲 2 块）

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "/home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model",
  "prompt": "大语言模型的推理服务需要同时处理许多并发请求。每个请求都会带来一段中文提示词，引擎首先执行预填充计算，把输入文本的全部令牌一次性算完，随后进入解码阶段，逐个生成后续的文字。预填充产生的键值会写入显存中的缓存块，之后每生成一个新令牌，注意力计算都要读取这些已缓存的键值。为了减少碎片，系统把每相邻的一百二十八个令牌放进同一个块，块由调度器统一编号、分配和回收。请求结束时，写满的块连同内容哈希一起留在缓存池中，后续请求只要前缀相同，就可以直接复用这些块，省去重复计算，这正是前缀缓存机制的核心。调度器的每一次操作都可以在日志里观察到，块的编号、引用计数、内容哈希以及命中与否，都会逐行打印，方便对照理论逐条验证。当第二条请求到达时，前缀查找会沿着第一条请求留下的哈希链逐块比对，命中即标记复用，未命中则立即中断查找，剩余部分重新计算并写入新的块。本文用于缓存实验，后面的每个字都会参与哈希。",
  "max_tokens": 1, "temperature": 0, "ignore_eos": true
}' > logs/curl/resp_p.json
```

##### 请求 2（R：五块生命周期）

```bash
sleep 6
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "/home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model",
  "prompt": "大语言模型的推理服务需要同时处理许多并发请求。……（P 全文）……本文用于缓存实验，后面的每个字都会参与哈希。现在请结合上面介绍，逐条详细回答后面的每个问题：第一，本次推理的前缀查找到底复用了缓存池中的哪两个块，这算不算零拷贝共享？第二，预填充阶段新申请了几个块，哪一个恰好被追问句写满并且连同内容哈希记入映射表，哪一个尚未写满？第三，解码阶段的生成需要多少步才能把未满块填到一百二十八，又是从哪一步开始申请第五个块？第四，请求结束后这些块按什么顺序归还，归还之后哪些块还能被下一个请求命中？请认真作答。",
  "max_tokens": 35, "temperature": 0, "ignore_eos": true
}' > logs/curl/resp_r.json
```

文件方式（实测所用）：

```bash
python3 scripts/curl/gen_cn_requests.py --gen
bash scripts/curl/curl_p_r.sh
```

#### 响应样例（temperature=0）

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| P | 1 | length | `"为了"` |
| R | 35 | length | `://www.zhihu.com/question/404202526\n1. 什么是前缀缓存…`（中文贪心续写） |
> 实测（15:17 轮）响应与上表逐字一致（temperature=0 确定性）——P：completion=1 / finish=length / “为了”；R：completion=35 / finish=length（响应原文见 `logs/curl/resp_p.json` / `resp_r.json`）。

### 1.3 运行期打印

每请求 / 每步动态输出，统一用横幅对 + `--- S 子步标记 ---` + 阶段前缀，按请求生命周期分六步：

1. **入队（ENQ）**——`入队 hash_block_tokens` 逐满块链式哈希：H(bn)=fn(H(bn−1), tokens(bn))，首块 parent=NONE_HASH
2. **前缀查找**——逐块 HIT/MISS 查链（遇 miss 即断），hit_length = 命中块数 × block_size
3. **分配 S1~S4**（四项设计）：
   - **横幅先行**：子步横幅先于其全部下钻——S1 子步横幅在两次外层容量探问（L4）之前、S3 横幅在 `allocate_new_blocks` 调用之前
   - **四子步无条件**：无新块步也打全四段（S1 需分配 0 / S2 无前缀无需 touch / S3 无需分配新块 / S4 满块缓存维护），结构闭合；容量不足、延迟缓存提前返回路径均补关闭横幅
   - **阶段前缀全显**：每条下钻消息开头标注所属阶段（S1~S4 / 前缀查找 / 入队 / 释放 / 分配 / 提交），单条日志可定位
   - **S1 汇总值**：总横幅 → 进入 → `--- S1: 容量检查 ---` → 两次外层探问 → 需求块数 vs 可用块数
4. **调度提交**——async_scheduler 每步输出后的 `cache_blocks` 提交路径单列 `调度提交(非分配 S4)` 横幅对（每步一次，独立于分配 S4）
5. **KVS 物理归档（仅请求结束 TERM / 兜底 LATE 告警）**——TERM 时各 worker 把该请求全部物理 KV 块**整块原样**（`.cpu().clone()` 位级快照，含未写槽位）交后台线程 `torch.save` 为 tensors/req{seq}_{rid尾8}/kv_pp{p}tp{t}.pt（一请求一子目录，meta.group_size=组内层数——block id 即组内每张 K/V 张量 dim0 的统一行号）；日志仅横幅两行：`======== 开始保存物理tensor ========`（worker/dev/shape/dtype/层×块/blk/cov/文件名）+ `======== 完成保存物理tensor ========`（文件名/字节/落盘耗时）；逐层逐块粒度由 .pt meta + `scripts/analysis/inspect_kv_tensors.py` 离线承载
6. **释放**——逆序归还、ref_cnt 归零回收、带哈希块 append 队尾（LRU 保护）

## 2. 实验流程与复现

### 2.1 起容器

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

### 2.2 打 patch

```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc/scripts/patchs && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
# Phase0 已应用检测 -> Phase1 dry-run 8/8 预检 -> Phase2 8/8 应用 -> Phase3 逐文件计数(合计 174 行) + py_compile
```

- **174 行**——apply 脚本 Phase 3 用 `grep -c "[KVC]"` 逐文件统计、并与预期值比对的口径，= **vllm 155（01~07 管理侧打印）+ vllm-ascend 19（08 归档版：[L1] 8 行 + [KVS] 11 行）**，均含注释行（`# [KVC]…` 设计说明，不产生日志）。
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
| 08 | `vllm_ascend/worker/model_runner_v1.py` | **19**（[L1] 8 + [KVS] 11，块-行映射归档版） |
| **合计** | 8 文件 | **174** = vllm 155 + vllm-ascend 19 |

> **08 号补丁（vllm-ascend，19 行 = [L1] 8 + [KVS] 11）**：请求结束不做逐层统计打印，而是把物理 KV 整块归档——横幅两行 + 一请求一子目录 `req{seq}_{rid尾8}/kv_pp{p}tp{t}.pt` + `meta.group_size` 块-行映射（block id = 组内 gs 张 K + gs 张 V 张量 dim0 各一行，查看报告逐块直读），01~07 管理侧打印保持不变。另：07 号把 CFG 编排开始横幅 + ① 算规格打印放在 `model_executor.get_kv_cache_specs()` 调用后紧跟处（core.py:245/:256），算规格产物在 profile_run 之前即可观测；04 号 58 行含“分配布局”分段行。

### 2.3 起服务并发送 P/R

```bash
bash scripts/server/start.sh                        # vllm serve PP2TP2 --enforce-eager，就绪 ~50s
bash scripts/curl/curl_p_r.sh                     # P -> sleep 6 -> R；落盘打屏/响应/分界并提取三条 [KVC] 轨迹
```

### 2.4 收产物并去 patch

```bash
# 产物回收: run_all [6/6] 已在容器内自动打包 -> 主机侧 fetch 拉回 (logs/ + tensors/ 全套, tar md5 备查)
bash scripts/recover/pull_artifacts.sh fetch    # 主机侧 kvc/: 经 5557 隧道拉回 -> 解包 -> 8 .pt 核对
# 容器回收
cd <kvc> && bash scripts/server/stop.sh             # 杀服务并确认 0 进程
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend bash scripts/patchs/revert_patches.sh
# 本地离线分析(输出留痕 logs/analysis/inspect_*.out): python3 scripts/analysis/inspect_kv_tensors.py --dir tensors
# 终态核验: 8 文件 [KVC] 全部归零 + py_compile + 两仓库 git 0 改动 + .orig 清理
```
### 2.5 复现注意事项

1. **生效前提**：服务带 [KVC] 调试体系运行（174 行 = vllm 155 + vllm-ascend 19，08 号为块-行映射归档版；应用方法见 §2.2 与 `../README.md`）。
2. **区间鲁棒**：R 落在 (384,512) 任意位置皆成立。快速断言：`调度提交` R=35、[KVS] 归档横幅 P=**12**（4 启用 + 4 开始 + 4 完成）/ R=**8**（4 开始 + 4 完成，双请求共 20 行）、S1 汇总值 R = 33×0 + 1×1 + 1×4——详 §5.6 计数自检。
3. **第 3 hash 的 MISS 断链**：R 的追问句内容须在缓存中不存在——与 P 仅共享前 256 token 的设计保证（§1.2「设计原理」）。
4. **哈希值每次服务重启变化**（种子随机）：本轮链值见 §5.6 哈希链。
5. **冷/热缓存对 P 的影响**：要完整复现“缓冲 → 五块生命周期”，先重启服务清缓存再依次发 P、R。
6. **KV 布局与归档校验要点**：K/V 为**张量级拆分**的两独立池（非最后一维拼接）；block id 即池张量 dim0 行号（.pt 离线 `K["K"][layer][blk]` 索引）；4 卡各持不同层段与 kv_heads 切片，同层同位值卡间不同属正常（TP2 切 kv_heads）——详 §4.3（.pt 格式）与 §6 结论 5。
7. **阶段前缀即导航**：`grep -- '--- S' logs/patchs/kvc_r.log`（子步横幅）、`grep '开始保存物理tensor' logs/patchs/kvc_r.log`（KVS 归档横幅）、`grep '调度提交'`（每步提交）、`grep '\[未满'`（未满块过滤）。

## 3. 启动期日志讲解（logs/patchs/kvc_startup.log，172 行）

### 3.1 配置侧（EngineCore，88 行 CFG，含首尾横幅）

对应理论文档初始化四阶段管线（① 算规格 → ② 测预算 → ③ 做编排 → ④ 落张量）中的 **① 算规格 + ② 测预算 + ③ 做编排**——以 ①②③ **子步横幅**（`--- ①: 算规格 ---` / `--- ②: 测预算 ---` / `--- ③: 做编排 ---`）把配置侧日志划分成三个小步，① 打印紧跟 `model_executor.get_kv_cache_specs()` 调用。④ 落张量在 §3.2 物理侧。88 行结构地图：

| 行号 | 日志内容 | 对应理论阶段 |
|---|---|---|
| :1 | `================ 配置侧 KVCache 编排开始 ================` | 开始横幅（紧跟 `get_kv_cache_specs()` 调用） |
| :2 | `--- ①: 算规格 ---` | **① 子步横幅** |
| :3 | `① 算规格 get_kv_cache_specs: worker0=16×FullAttentionSpec(…); …共 4 worker` | **① 算规格**（每 worker 层 spec 摘要：类型×层数×首末层名） |
| :4 | `--- ②: 测预算 ---` | **② 子步横幅**（先于 profile_run） |
| :5 | `determine_available_memory: 各 worker 可用 KV 显存` | **② 测预算**（`profile_run` 实测） |
| :6 | `--- ③: 做编排 ---` | **③ 子步横幅**（先于 get_kv_cache_configs） |
| :7~:86 | worker0~3 各 20 行（Config + GroupSpec + spec + page + KVCacheTensor ×16） | **③ 做编排**（合并 → 分组 → 投影 → num_blocks） |
| :87 | `最终 scheduler KVCacheConfig: num_blocks=… (跨 worker min 对齐)` | ③ 做编排 · 多 worker min 对齐 |
| :88 | `================ 配置侧 KVCache 编排完成 ================` | 收尾横幅 |

**(a) ① 算规格（第 1~3 行，3 条：开始横幅 + ① 子步横幅 + ① 打印；紧跟 `get_kv_cache_specs()` 调用）**

```
INFO 10-03 15:17:18 [core.py:245] [KVC][CFG] ================ 配置侧 KVCache 编排开始 ================
INFO 10-03 15:17:18 [core.py:246] [KVC][CFG] --- ①: 算规格 ---
INFO 10-03 15:17:18 [core.py:257] [KVC][CFG] ① 算规格 get_kv_cache_specs: worker0=16×FullAttentionSpec(首 model.layers.0.self_attn.attn, 末 model.layers.15.self_attn.attn); worker1=16×FullAttentionSpec(首 model.layers.0.self_attn.attn, 末 model.layers.15.self_attn.attn); worker2=16×FullAttentionSpec(首 model.layers.16.self_attn.attn, 末 model.layers.31.self_attn.attn); worker3=16×FullAttentionSpec(首 model.layers.16.self_attn.attn, 末 model.layers.31.self_attn.attn)
```

① 打印的是 `get_kv_cache_specs()` 的返回值摘要——EngineCore 向每 worker 收集本 rank 各层 KVCacheSpec（`list[dict[层名, spec]]`），一行读完全部 4 worker：各 **16×FullAttentionSpec**（PP2 按层切：PP0 两卡 `layers.0~15`、PP1 两卡 `layers.16~31`；TP2 切 kv_heads 不切层，spec 各 rank 同形）。四 worker spec 字段全等是 (c) 中 `is_kv_cache_spec_uniform=true → 全模型单 group` 的直接依据（理论 §2.3②）。

**(b) ② 测预算（第 4~5 行，2 条：② 子步横幅 + ② 打印；横幅先行，先于 `profile_run`）**

```
INFO 10-03 15:17:18 [core.py:259] [KVC][CFG] --- ②: 测预算 ---
INFO 10-03 15:17:20 [core.py:280] [KVC][CFG] determine_available_memory: 各 worker 可用 KV 显存 = ['51.96GiB', '51.97GiB', '51.92GiB', '51.93GiB']
```

② 横幅（15:17:18）先落，随后各 worker 并跑一次 `profile_run()`（dummy forward 量峰值，实测约 2s），结果 15:17:20 才打印：`available = 总显存 × 利用率 − 权重 − 激活 − 大图预留`（理论 §2.2）。四卡实测 51.92~51.97GiB；**最小者 51.92GiB（worker2）将决定 (c) 的 min 对齐**。

**(c) ③ 做编排（第 6~88 行——③ 横幅 + 80 行 worker + min 对齐 + 尾横幅）——worker0 段全量原样**

```
INFO 10-03 15:17:20 [core.py:284] [KVC][CFG] --- ③: 做编排 ---
INFO 10-03 15:17:20 [core.py:294] [KVC][CFG] worker0 KVCacheConfig: num_blocks=13291, groups数=1, tensors数=16
INFO 10-03 15:17:20 [core.py:299] [KVC][CFG]   [0] KVCacheGroupSpec(group_id=0): layers=16 (首层 model.layers.0.self_attn.attn, 末层 model.layers.15.self_attn.attn), is_eagle_group=False
INFO 10-03 15:17:20 [core.py:304] [KVC][CFG]   [0]   kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16, kv_quant_mode=<KVQuantMode.NONE: 0>, page_size_padded=None, head_size_v=128, sliding_window=None, attention_chunk_size=None)
INFO 10-03 15:17:20 [core.py:308] [KVC][CFG]   [0]   page_size_bytes=262144 (256.0KB/层/块), storage_block_size=128
INFO 10-03 15:17:20 [core.py:314] [KVC][CFG]   [0] KVCacheTensor: size=3484155904 bytes (3322.75MiB), shared_by=1 层 (model.layers.0.self_attn.attn)
INFO 10-03 15:17:20 [core.py:314] [KVC][CFG]   [0] KVCacheTensor: size=3484155904 bytes (3322.75MiB), shared_by=1 层 (model.layers.1.self_attn.attn)
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
INFO 10-03 15:17:20 [core.py:331] [KVC][CFG] 最终 scheduler KVCacheConfig: num_blocks=13291 (跨 worker min 对齐), cache_config.num_gpu_blocks=13291, block_size=128
INFO 10-03 15:17:20 [core.py:336] [KVC][CFG] ================ 配置侧 KVCache 编排完成 ================
```

集中式调度要求同一 `block_table` 对所有 rank 有效 → 取四 worker `num_blocks` 的 **min**（本例四卡恰好同为 13291，由最小预算 51.92GiB 卡定出）作为全局统一值（理论 §2.3⑤），并等比缩小各 `KVCacheTensor.size`，最后写回 `cache_config.num_gpu_blocks=13291`。

**预算闭合验算**：每 worker 16 张 × 3322.75MiB = 53,164MiB = **51.92GiB** 恰等于最小 worker 的 `available`——KV 预算被本 worker 的张量恰好铺满，编排无浪费。此后 §3.2（物理侧每层按此 size 建池）与 §3.3（逻辑侧 BlockPool 建 13,291 块）**消费同一份 KVCacheConfig**——两侧容量由同一配置锁定，是 `block_id == 张量行号` 桥接的编排前提。

### 3.2 物理侧（4 worker 并行，各 19 行 L1，vllm-ascend）

对应理论 `../../1_init_physical_memory.md` §2.4 的 ④ 落张量：`EngineCore → initialize_from_config() → initialize_kv_cache()`（4a/4b/4c 落张量 + 4d 编译预热两个 collective_rpc；4c/4d 无 [KVC] 打印）。4 卡并行各自执行；以下用 **Worker_PP0_TP0（pid=6070）的真实日志**示例（另 3 卡同构，仅 pid/device/层段不同——PP0 两卡打 `layers.0~15`、PP1 两卡打 `layers.16~31`）。

**(a) 4a 分配 int8 字节池（开始横幅 + 16 层 dense，每层 K/V 两个独立池）**

```
INFO 10-03 15:17:20 [model_runner_v1.py:4336] [KVC][L1] ================ 物理侧 KV Cache 分配开始 ================
INFO 10-03 15:17:21 [model_runner_v1.py:4498] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.0.self_attn.attn: KVCacheTensor(size=3484155904 bytes = 3322.75MiB) -> K int8 1661.38MiB + V int8 1661.38MiB (alignment=2097152, device=npu:0)
INFO 10-03 15:17:21 [model_runner_v1.py:4498] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.1.self_attn.attn: KVCacheTensor(size=3484155904 bytes = 3322.75MiB) -> K int8 1661.38MiB + V int8 1661.38MiB (alignment=2097152, device=npu:0)
...（layers.2~14 同构, 每层一行; 另 3 卡同构, 仅 pid/device/层段不同）
INFO 10-03 15:17:25 [model_runner_v1.py:4498] [KVC][L1] vllm-ascend _allocate_kv_cache_tensors[dense]: model.layers.15.self_attn.attn: KVCacheTensor(size=3484155904 bytes = 3322.75MiB) -> K int8 1661.38MiB + V int8 1661.38MiB (alignment=2097152, device=npu:0)
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
INFO 10-03 15:17:25 [model_runner_v1.py:4936] [KVC][L1] vllm-ascend _reshape_kv_cache_tensors: model.layers.0.self_attn.attn (本组 16 层同形) -> K_cache shape=(13291, 128, 4, 128) dtype=torch.bfloat16 / V_cache shape=(13291, 128, 4, 128) dtype=torch.bfloat16, device=npu:0 (K/V 分离布局)
INFO 10-03 15:17:25 [model_runner_v1.py:4987] [KVC][L1] ================ 物理侧 KV Cache 分配完成 ================
```

对照理论 4b：int8 → dtype → shape 的 **view 零拷贝**（`raw.view(dtype).view(shape)` 普通路径，理论 §2.4），再 permute 成后端逻辑布局——全程无数据拷贝，int8 池地址即最终张量地址。`K_cache shape=(13291, 128, 4, 128) bf16` 四维含义：dim0=num_blocks（**block id == 张量行号**，理论 §5 桥接）、dim1=block_size 128、dim2=kv_heads 4（8÷TP2）、dim3=head_dim 128；`K/V 分离布局`——两组独立的 `(num_blocks, 128, 4, 128)` 张量，区别于上游 GPU 的 K/V packed 单张量（理论 §5 表）。

### 3.3 逻辑侧装配（EngineCore，8 行全量）

**自底向上**逐组件装配，每组件一条 `__init__完成：` 打印，格式统一（5 条装配 + 首尾横幅共 8 行）：

```
INFO 10-03 15:17:30 [kv_cache_manager.py:143] [KVC][L5] ================ 逻辑侧初始化开始 ================
INFO 10-03 15:17:30 [kv_cache_utils.py:217] [KVC][L2] FreeKVCacheBlockQueue.__init__完成：num_free_blocks=13291, 伪头尾哨兵 fake_free_list_head/tail(block_id=-1), 类型=FreeKVCacheBlockQueue
INFO 10-03 15:17:30 [block_pool.py:62] [KVC][L2] BlockHashToBlockMap.__init__完成：底容器 size=0, value=KVCacheBlock | dict[block_id→KVCacheBlock](不去重 append-only)
INFO 10-03 15:17:30 [block_pool.py:214] [KVC][L2] BlockPool.__init__完成：num_gpu_blocks=13291, 创建 KVCacheBlock × 13291 (block_id=0..13290), free_block_queue=FreeKVCacheBlockQueue(num_free_blocks=13290), cached_block_hash_to_block=BlockHashToBlockMap(size=0), null_block=KVCacheBlock(block_id=0, is_null=True), enable_caching=True, hash_block_size=128
INFO 10-03 15:17:30 [single_type_kv_cache_manager.py:95] [KVC][L3] FullAttentionManager.__init__完成：spec=FullAttentionSpec(block_size=128), scheduler_block_size=128, group_id=0, enable_caching=True, dcp×pcp=1×1, block_pool(num_gpu_blocks=13291)
INFO 10-03 15:17:30 [kv_cache_coordinator.py:462] [KVC][L4] UnitaryKVCacheCoordinator.__init__完成：单组直通, managers=['FullAttentionManager'], kv_cache_spec=FullAttentionSpec(block_size=128, page_size_bytes=262144), coordinator_block_size=128
INFO 10-03 15:17:30 [kv_cache_manager.py:178] [KVC][L5] KVCacheManager.__init__完成：coordinator=UnitaryKVCacheCoordinator, num_kv_cache_groups=1, managers=['FullAttentionManager'], block_pool(num_gpu_blocks=13291), enable_caching=True, max_model_len=8192, empty_kv_cache_blocks=KVCacheBlocks([],)
INFO 10-03 15:17:30 [kv_cache_manager.py:186] [KVC][L5] ================ 逻辑侧初始化完成 ================
```

### 3.4 启动期原生关键行（vllm 自带，非 [KVC] patch）

这一节收的是 **[KVC] 调试打印之外、vllm 自己打的结论性行**：启动期有任何一步（算规格→测预算→做编排→落张量→装配）出问题，服务起不来或数值对不上，这 6 行是排障的第一落点。按日志出现序：

```
(APIServer pid=5848) INFO 10-03 15:16:49 [utils.py:1404] Block size is set to 128 if prefix cache or chunked prefill is enabled.
(Worker_PP0_TP0 pid=6070) INFO 10-03 15:17:20 [worker.py:593] Available KV cache memory: 51.96 GiB
(EngineCore pid=5969) INFO 10-03 15:17:20 [kv_cache_utils.py:1771] GPU KV cache size: 1,701,248 tokens
(EngineCore pid=5969) INFO 10-03 15:17:20 [kv_cache_utils.py:1772] Maximum concurrency for 8,192 tokens per request: 207.67x
(EngineCore pid=5969) INFO 10-03 15:17:29 [core.py:369] init engine (profile, create kv cache, warmup model) took 10.78 s
(APIServer pid=5848) INFO:     Application startup complete.
```

（6 行分别位于 `llama-3-8b.log` :32 / :155 / :158 / :159 / :338 / :394；最后一行即就绪标志，15:16:36 run_all 启动 → 50s 轮询检测就绪（init engine 10.78s 止于 15:17:29））

逐行详解：

| 行号 | 谁打的（源码） | 说什么 / 怎么算 | 与 [KVC] 的对账 |
|---|---|---|---|
| :32 | APIServer（utils.py:1404） | **block_size=128**：开了 prefix cache / chunked prefill 时 vllm 把存储分块粒径定成 128（NPU 页对齐默认）——后面所有"满块 128 tokens、按 128 切哈希"的分母 | [KVC][L1] `K_cache=(num_blocks, 128, 4, 128)` 的 dim1=128 即此值 |
| :155 | Worker_PP0_TP0（worker.py:593） | worker 0 卡**实测可分配 KV 显存 51.96 GiB**（整卡可用 − 权重 − 激活 − warmup 峰值）；4 worker 各打 1 行，本卡只是第一行 | §3.1 (b) [KVC] `determine_available_memory` 的 51.96/51.97/51.92/51.93——同一测量、[KVC] 四卡一屏对比，原生行只逐卡逐行 |
| :158 | EngineCore（kv_cache_utils.py:1771） | **总 KV 容量 1,701,248 tokens** = num_blocks × block_size | 13,291 × 128 = 1,701,248——与 §3.1 (c) 最终 num_blocks 完全一致 |
| :159 | EngineCore（kv_cache_utils.py:1772） | **maximum concurrency 207.67x**：满载 8,192-token 长请求时的并发上限（理论上限，实际受连续批处理调度影响） | 1,701,248 ÷ 8,192 = 207.67——同源两行连算，max_model_len=8192 为分母 |
| :338 | EngineCore（core.py:369） | **init engine 10.78 s**：profile（=② 测预算的 dummy forward）+ create kv cache（=③ 做编排 + ④ 落张量）+ warmup 的总耗时 | [KVC] 各步（15:17:18 编排 → 15:17:25 落张量完成）全落在这 10.78s 内；逻辑侧装配（§3.3，15:17:30）紧随其后——[KVC] 打印对启动时延的净增量可用此行与无 patch 版对比衡量 |
| :394 | APIServer（uvicorn） | **服务就绪标志**：路由全部挂载、uvicorn 开始收请求 | P 分界行号 395 与本行紧邻（:394 就绪标志，:395 即 P 入队）——§4 的 P/R 轨迹分界从这条起算（分界号由 curl_p_r.sh 运行时取行数，不落盘）；start.sh 的就绪探测就是 `grep 'Application startup complete' logs/server/llama-3-8b.log` |

## 4. P 运行期日志讲解（logs/patchs/kvc_p.log，61 行）

P：num_prompt_tokens=324（2 满块 + 尾 68），max_tokens=1——一次 prefill 即终态（TERM），验证"缓冲 2 块"。

> 以下按理论 `../../0_runtime_sequence.md` §4 分阶段详解的时序组织：**入队 → 首次调度（①前缀查找 + ②allocate_slots）→ GPU 写 KV → （③ decode）→ ④ 结束释放**。GPU 写 KV 与 P 的 decode 环节无 [KVC] 打印（正确性由 KVS TERM 归档离线比对兜底验证，§4.3）。

| 理论时序（0_runtime_sequence §4） | 日志小节 | P 实测要点 |
|---|---|---|
| §4.1 入队（预计算链式哈希 → WAITING） | §4.1 | 2 个满块哈希入队（NONE_HASH 起链） |
| §4.2 首次调度 ① get_computed_blocks（前缀查找） | §4.1 | 冷缓存第 1 hash 即 MISS，hit_length=0 |
| §4.2 首次调度 ② allocate_slots（S1~S4） | §4.2 | S1 需 3 vs 可用 13290 → S3 [1,2,3] → S4 双满块入表 |
| §4.3 GPU 写 KV（forward） | —（无 [KVC] 打印） | 写 324 token K/V；物理正确性由 KVS TERM 归档离线验证（§4.3） |
| ③ decode | — | P max_tokens=1，首 token 即终态，无 decode 循环 |
| ④ 结束释放（§4.5） | §4.4 | [3,2,1] 逆序 append 队尾（LRU 保护） |

### 4.1 入队与前缀查找

```
INFO 10-03 15:17:32 [request.py:184] [KVC][ENQ] ======== 入队 ========
INFO 10-03 15:17:32 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=754104f1d6be
INFO 10-03 15:17:32 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=754104f1d6be, tokens=128 -> BlockHash=70ae0ea07102
INFO 10-03 15:17:32 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-b6b18c35b4fb325e-0-bdd8c88d) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['754104f1d6be', '70ae0ea07102']
INFO 10-03 15:17:32 [request.py:193] [KVC][ENQ] ======== 入队完成 ========
INFO 10-03 15:17:32 [kv_cache_manager.py:222] [KVC][L5] ======== 前缀查找 ========
...（冷缓存: 第 1 块即 MISS 断链, hit_length=0, 返回 blocks=[[]]）
INFO 10-03 15:17:32 [block_pool.py:89] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=754104f1d6be, group_id=0) -> MISS
INFO 10-03 15:17:32 [block_pool.py:247] [KVC][L2] 前缀查找 BlockPool.get_cached_block: MISS BlockHash=754104f1d6be (group_id=0 未命中) -> None
INFO 10-03 15:17:32 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 1 块 MISS: BlockHash=754104f1d6be -> break
```

### 4.2 分配 S1~S4 全链

总横幅（kv_cache_manager.py:393，本节括号行号均为日志中打印的**源码行号**，不是 kvc_p.log 文件行号）→ 进入（:399, 全 7 字段）→ 分配布局（:409）→ **S1 子步横幅（:417）** → 两次外层探问（kv_cache_coordinator.py:188×2）→ S1 汇总值（:462），随后 S2/S3/S4 四子步下钻逐层展开：

```
INFO 10-03 15:17:32 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 10-03 15:17:32 [kv_cache_manager.py:399] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_new_tokens=324(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=0(comp), request.num_tokens=324, delay_cache_blocks=False
INFO 10-03 15:17:32 [kv_cache_manager.py:409] [KVC][L5] 分配布局: |<comp>=0 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=324 |<lookahead>=0| num_local_computed_tokens=0 total_computed_tokens=0 to_be_computed=324
INFO 10-03 15:17:32 [kv_cache_manager.py:417] [KVC][L5] --- S1: 容量检查---
INFO 10-03 15:17:32 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 10-03 15:17:32 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_tokens=324 -> 需分配 3 块(含touch需腾挪的块)
INFO 10-03 15:17:32 [kv_cache_manager.py:462] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13290 块 (free=13290 - reserved=0)
INFO 10-03 15:17:32 [kv_cache_manager.py:496] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 10-03 15:17:32 [kv_cache_manager.py:500] [KVC][L5] --- S3: 新块分配 ---
INFO 10-03 15:17:32 [block_pool.py:416] [KVC][L2] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=13287
INFO 10-03 15:17:32 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_tokens=324, block_size=128, 需 3 块 - 已有 0 = 新分配 3 块 [1, 2, 3], 持有 req_blocks=[1, 2, 3]
INFO 10-03 15:17:32 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_tokens=324 -> [[1, 2, 3]]
INFO 10-03 15:17:32 [kv_cache_manager.py:511] [KVC][L5] S3 allocate_new_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_tokens_need_slot=324 -> 新块 [1, 2, 3]
INFO 10-03 15:17:32 [kv_cache_manager.py:538] [KVC][L5] --- S4: 满块入缓存 ---
INFO 10-03 15:17:32 [single_type_kv_cache_manager.py:357] [KVC][L3] S4 SingleTypeKVCacheManager.cache_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_tokens=324, block_size=128, 已缓存 0 块 -> 满块数 2
INFO 10-03 15:17:32 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=754104f1d6be, group_id=0) <- KVCacheBlock(block_id=1), map size=1
INFO 10-03 15:17:32 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=70ae0ea07102, group_id=0) <- KVCacheBlock(block_id=2), map size=2
INFO 10-03 15:17:32 [block_pool.py:345] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d 新满块 2 块 block_ids=[1, 2] 入 BlockHashToBlockMap (num_cached_blocks 0 -> 2, group_id=0, map size=2)
INFO 10-03 15:17:32 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_computed_tokens=324
INFO 10-03 15:17:32 [kv_cache_manager.py:544] [KVC][L5] S4 cache_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_tokens_to_cache=324
INFO 10-03 15:17:32 [kv_cache_manager.py:548] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([1, 2, 3],)), req=cmpl-b6b18c35b4fb325e-0-bdd8c88d 当前完整 block_table=([1, 2, 3],)
INFO 10-03 15:17:32 [kv_cache_manager.py:554] [KVC][L5] ======== 分配完成 ========
```

要点：324 tokens → S3 需 3 块（`需 3 块 - 已有 0 = 新分配 3 块 [1, 2, 3]`）；S4 只把 2 个满块入哈希表（324 // 128 = 2，`insert` ×2），块 3 为尾块（68/128）不入表。

### 4.3 KVS 物理归档（TERM ×4 worker，横幅两行式）

TERM 判定成立（worker 内 written ≥ prompt_len + max_tokens − 1，即该请求最后一次写卡已完成）时，各 worker 把该请求全部物理 KV 块**整块原样**归档：每层每块 `kt[blk].cpu().clone()`（位级快照，**含未写槽位**），同步快照完成后交后台 daemon 线程 `torch.save` 落盘 `.pt`，不阻塞推理。日志仅横幅两行，不做逐层统计打印，逐层逐块粒度全部由 `.pt` 内 meta + 离线查看器（`scripts/analysis/inspect_kv_tensors.py`）承载：

```
INFO 10-03 15:17:32 [model_runner_v1.py:2441] [KVC][KVS] 物理KV原样归档已启用: KVC_SAVE_KV=1, 输出目录=/a3_inference/itask/workdir/wsl02075301/kvc/tensors (相对 worker cwd)
INFO 10-03 15:17:32 [model_runner_v1.py:2573] [KVC][KVS] ======== 开始保存物理tensor worker=PP0_TP0 dev=npu:0 TERM seq=1 req尾8=bdd8c88d: K_cache/V_cache(双独立张量池, 每块 shape=(128,4,128) torch.bfloat16, 含未写槽位) 16 层 × 3 块 blk=[1, 2, 3] cov=[128, 128, 68] -> req1_bdd8c88d/kv_pp0tp0.pt ========
INFO 10-03 15:17:32 [model_runner_v1.py:2623] [KVC][KVS] ======== 完成保存物理tensor worker=PP0_TP0 dev=npu:0 seq=1 req尾8=bdd8c88d: req1_bdd8c88d/kv_pp0tp0.pt (K_cache/V_cache, 16 层 × 3 块, 12611023 B = 12.0 MiB) 落盘 35.0 ms ========
```

另 3 卡同构（PP0_TP1 / PP1_TP0 / PP1_TP1 → npu:1/2/3，flush 32.0~36.8 ms）。启用横幅每 worker 首个前向后各 1 行——P 段 [KVS] 合计 12 行（4 启用 + 4 开始 + 4 完成），双请求全程共 20 行。两行横幅承载字段：

| 横幅 | 打印时点 | 承载字段 |
|---|---|---|
| 开始（model_runner_v1.py:2573，源码行号） | 归档启动（同步） | worker/dev/TERM/seq/req尾8——**保存什么**：K_cache/V_cache（双独立张量池）· 每块 shape=(128,4,128) bf16 · 16 层 × 3 块 · blk/cov → 目标文件名 |
| 完成（model_runner_v1.py:2623，源码行号） | 落盘完成（后台线程异步） | **save 完成回执**：文件名 + 12611023 B (12.0 MiB) + 落盘 35.0 ms（失败兜底 ARCHIVE-FAIL 单行，绝不影响服务） |

**归档时点语义**：free() 在 EngineCore 进程、物理张量在各 worker 进程，跨进程不可直读——以“最后一次写卡完成”为等价时点。本轮 kvc_p.log 实测：4 卡开始横幅（kvc_p.log :40~48）全部先于 L5 释放横幅（kvc_p.log :55），同步快照先于块归还；LATE（结束后兜底）不归档仅一行告警，本轮 0 次。

**.pt 格式（schema kvt4-raw）**：`{"K": [16 层, {块号: (128,4,128) bf16}], "V": 同构, "meta": {pp/tp/seq/request_id/p_tok/w_tok/final/block_table/cov/layers/layer_ids/group_size/kv_heads/head_dim/dtype/dev/ts}}`（group_size=组内层数：block id 即组内每张 K/V 张量 dim0 的行号，块-行映射）；文件名 `req{seq}_{rid尾8}/kv_pp{pp}tp{tp}.pt`（一请求一子目录）；层序为 worker 本地序（全局层号 = pp×16 + 本地序，4 worker 联合覆盖 32 层）。

**离线对账**：开始/完成横幅 vs meta/字节 16/16 PASS（实验轮内对账：run_all [4/6] 归档核对 + fetch 后横幅 vs .pt 复核）；未写槽位全零（P.b3[68:128]、R.b6[8:128]，块池新建基线为零）。L00 region 统计与历史轮打印记录值**4 位完全相同**（K: mean=-0.0219 std=1.394 min=-10.38 max=10.81；V: 0.0007659/0.03501/-0.2539/0.3223）——归档与打印位级等价、跨 run 确定（多轮互证，详 §5.5 表）。

### 4.4 调度提交与结束释放

async_scheduler 每步输出后的独立提交段（与分配 S4 明确区分），随后逆序释放、带哈希块 LRU 归队（KVS 开始/完成横幅在 kvc_p.log :39~50，已先行，见 §4.3）：

```
INFO 10-03 15:17:33 [kv_cache_manager.py:691] [KVC][L5] ======== 调度提交(非分配 S4) ========
INFO 10-03 15:17:33 [kv_cache_manager.py:692] [KVC][L5] 提交 cache_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_computed_tokens=324 (async 步末输出路径: 本步已算 token 提交入缓存)
INFO 10-03 15:17:33 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, num_computed_tokens=324
INFO 10-03 15:17:33 [kv_cache_manager.py:698] [KVC][L5] ======== 提交完成 ========
INFO 10-03 15:17:33 [kv_cache_manager.py:566] [KVC][L5] ======== 释放 ========
INFO 10-03 15:17:33 [kv_cache_manager.py:568] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, 释放前持有 block_table=([1, 2, 3],)
INFO 10-03 15:17:33 [single_type_kv_cache_manager.py:405] [KVC][L3] 释放 SingleTypeKVCacheManager.free: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d, 持有 blocks=[1, 2, 3] (reversed 后释放)
INFO 10-03 15:17:33 [block_pool.py:521] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(3, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 3 块 [3, 2, 1], append_n -> 队尾(LRU保护)
INFO 10-03 15:17:33 [kv_cache_utils.py:391] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[3, 2, 1]), num_free_blocks=13290
INFO 10-03 15:17:33 [kv_cache_coordinator.py:299] [KVC][L4] 释放 KVCacheCoordinator.free: req=cmpl-b6b18c35b4fb325e-0-bdd8c88d 已逐组下放第3层释放
INFO 10-03 15:17:33 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

## 5. R 运行期日志讲解（logs/patchs/kvc_r.log，719 行）

R：num_prompt_tokens=486 = 3 满 + 第 4 块 102/128，max_tokens=35——完整五块生命周期（复用 2 + prefill 补 1 满 1 尾 + decode 填满尾块 + 步 27 跨界申请第 5 块）。request_id=cmpl-863603840a5003dc-0-9eb8d7fa。

> 按理论 `../../0_runtime_sequence.md` §4 时序组织：R 覆盖**全部阶段**（理论时序图以 70 token/16 块示例推演，此处为 486 token/128 块全量实测）。

| 理论时序（0_runtime_sequence §4） | 日志小节 | R 实测要点 |
|---|---|---|
| §4.1 入队 | §5.1 | 3 个满块哈希入队（P 链延伸 + 追问句新段） |
| §4.2 ① 前缀查找（get_computed_blocks） | §5.1 | HIT×2 → 第 3 hash MISS 断链，hit_length=2×128=256 |
| §4.2 ② allocate_slots（S1~S4） | §5.2 | S2 touch[(1,1),(2,1)] 零拷贝复用；S3 [4,5]；S4 新满块入表 |
| §4.3 GPU 写 KV（forward） | —（无 [KVC] 打印） | 跨 4 块写 486 token（复用块不重算） |
| §4.4 ③ decode·情况 A（需 0 块） | §5.3 | 33 步同构闭合链 + 每步调度提交 |
| §4.4 ③ decode·情况 B（需 1 块） | §5.4 | 步 27 跨界：新块 [6]，刚填满的块 5 入表 |
| KVS 物理归档与离线互证（实装侧，理论之外） | §5.5 | TERM 五块归档 8/8 + 横幅对账 + 重算 ULP 一致性 |
| ④ 结束释放（§4.5） | §5.6 | 五块逆序归队 [6,5,4,2,1] |

### 5.1 入队与前缀查找（HIT×2 后 MISS 断链）

```
INFO 10-03 15:17:39 [request.py:184] [KVC][ENQ] ======== 入队 ========
INFO 10-03 15:17:39 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=754104f1d6be
INFO 10-03 15:17:39 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=754104f1d6be, tokens=128 -> BlockHash=70ae0ea07102
INFO 10-03 15:17:39 [kv_cache_utils.py:612] [KVC][ENQ] 入队 hash_block_tokens: parent=70ae0ea07102, tokens=128 -> BlockHash=1839e465e0eb
INFO 10-03 15:17:39 [request.py:187] [KVC][ENQ] Request(request_id=cmpl-863603840a5003dc-0-9eb8d7fa) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 BlockHash × 3: ['754104f1d6be', '70ae0ea07102', '1839e465e0eb']
INFO 10-03 15:17:39 [request.py:193] [KVC][ENQ] ======== 入队完成 ========
INFO 10-03 15:17:39 [block_pool.py:75] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=754104f1d6be, group_id=0) -> HIT KVCacheBlock(block_id=1)
INFO 10-03 15:17:39 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 1 块 HIT: BlockHash=754104f1d6be -> cached blocks=[1]
INFO 10-03 15:17:39 [block_pool.py:75] [KVC][L2] 前缀查找 BlockHashToBlockMap.get_one_block: key=BlockHashWithGroupId(hash=70ae0ea07102, group_id=0) -> HIT KVCacheBlock(block_id=2)
INFO 10-03 15:17:39 [single_type_kv_cache_manager.py:599] [KVC][L3] 前缀查找   第 2 块 HIT: BlockHash=70ae0ea07102 -> cached blocks=[2]
INFO 10-03 15:17:39 [single_type_kv_cache_manager.py:607] [KVC][L3] 前缀查找   第 3 块 MISS: BlockHash=1839e465e0eb -> break
INFO 10-03 15:17:39 [kv_cache_coordinator.py:495] [KVC][L4] 前缀查找 UnitaryKVCacheCoordinator.find_longest_cache_hit 返回: hit_blocks=[[1, 2]], hit_length=256
```

命中 2 块 → hit_length=2×128=256；第 3 个 hash 是 R 追问句新内容（冷），断链即止。

### 5.2 prefill 完整链（S2 touch 复用 + S3 补 2 块 + S4 新满块入表）

```
INFO 10-03 15:17:39 [kv_cache_manager.py:482] [KVC][L5] --- S2: touch 命中块 ---
INFO 10-03 15:17:39 [kv_cache_manager.py:484] [KVC][L5] S2 allocate_new_computed_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, new_computed_blocks=[[1, 2]]
INFO 10-03 15:17:39 [block_pool.py:496] [KVC][L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)
INFO 10-03 15:17:39 [kv_cache_manager.py:500] [KVC][L5] --- S3: 新块分配 ---
INFO 10-03 15:17:39 [block_pool.py:416] [KVC][L2] S3 BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13286
INFO 10-03 15:17:39 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens=486, block_size=128, 需 4 块 - 已有 2 = 新分配 2 块 [4, 5], 持有 req_blocks=[1, 2, 4, 5]
INFO 10-03 15:17:39 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=1839e465e0eb, group_id=0) <- KVCacheBlock(block_id=4), map size=3
INFO 10-03 15:17:39 [block_pool.py:345] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa 新满块 1 块 block_ids=[4] 入 BlockHashToBlockMap (num_cached_blocks 2 -> 3, group_id=0, map size=3)
INFO 10-03 15:17:39 [kv_cache_manager.py:548] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([4, 5],)), req=cmpl-863603840a5003dc-0-9eb8d7fa 当前完整 block_table=([1, 2, 4, 5],)
INFO 10-03 15:17:39 [kv_cache_manager.py:554] [KVC][L5] ======== 分配完成 ========
```

要点：**touch 零拷贝**——`S2 BlockPool.touch: blocks=[(1, 1), (2, 1)]`（block_id, ref_cnt），直接复用 P 留下的块 1/2；S3 = cdiv(486,128)−2 = 2 新块 [4,5]；S4 把追问句恰好填满的块 4 入表（map 2→3）。随后的 SchedulerOutput 携带 `new_block_ids_to_zero=[4, 5]`（对应理论 §4.3“调度输出附清零块 id”）——GPU forward 写 KV 无 [KVC] 日志，物理正确性由 §5.5 的 KVS 归档离线比对实测验证。

> **新块为什么是 [4,5] 而不是 [3,4]？** P 释放时把 [3,2,1] 整体 `append_n` 到空闲队列**队尾**（§4.4，LRU 保护），队列队头仍是顺序在后的 4,5,6,…；新分配一律从队头 `popleft`，所以 R 拿到 [4,5] 而非块 3——P 的尾块 3 没有哈希、不可被前缀命中，排在队尾等待后续被覆写。同理，decode 跨界时 R 的**第 5 个逻辑块**从队头拿到的物理 id 是 **6**（§5.4），不是 5。

### 5.3 decode 无块步（需 0 块，33 步同构）

理论 §4.4 把 decode 每步 allocate_slots 分两种情况：**情况 A**·当前块未满需 0 块（token 直接续写）/ **情况 B**·已满需 1 块（token 落进下一块）。本节是情况 A 的每步实录；decode 步 1 起的每一步（除步 27 外）都是这条闭合链——S1 如实打“需分配 0”，S3 切换“无需分配新块”文案，尾部紧跟本步调度提交：

```
INFO 10-03 15:17:39 [kv_cache_manager.py:393] [KVC][L5] ======== 分配 S1~S4 ========
INFO 10-03 15:17:39 [kv_cache_manager.py:399] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_new_tokens=1(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=486(comp), request.num_tokens=486, delay_cache_blocks=False
INFO 10-03 15:17:39 [kv_cache_manager.py:409] [KVC][L5] 分配布局: |<comp>=486 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=1 |<lookahead>=0| num_local_computed_tokens=486 total_computed_tokens=486 to_be_computed=1
INFO 10-03 15:17:39 [kv_cache_manager.py:417] [KVC][L5] --- S1: 容量检查---
INFO 10-03 15:17:39 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens=487 -> 需分配 0 块(含touch需腾挪的块)
INFO 10-03 15:17:39 [kv_cache_manager.py:462] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 0 块 vs 可用 13286 块 (free=13286 - reserved=0)
INFO 10-03 15:17:39 [kv_cache_manager.py:496] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 10-03 15:17:39 [kv_cache_manager.py:502] [KVC][L5] --- S3: 无需分配新块 ---
INFO 10-03 15:17:39 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens=487 -> [[]]
INFO 10-03 15:17:39 [kv_cache_manager.py:516] [KVC][L5] S3 块未满, 无需分配新块 (req=cmpl-863603840a5003dc-0-9eb8d7fa, num_new_tokens=1)
INFO 10-03 15:17:39 [kv_cache_manager.py:538] [KVC][L5] --- S4: 满块入缓存 ---
INFO 10-03 15:17:39 [kv_cache_coordinator.py:284] [KVC][L4] S4 KVCacheCoordinator.cache_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_computed_tokens=486
INFO 10-03 15:17:39 [kv_cache_manager.py:544] [KVC][L5] S4 cache_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens_to_cache=486
INFO 10-03 15:17:39 [kv_cache_manager.py:548] [KVC][L5] 分配 KVCacheManager.allocate_slots 返回: KVCacheBlocks(blocks=([],)), req=cmpl-863603840a5003dc-0-9eb8d7fa 当前完整 block_table=([1, 2, 4, 5],)
INFO 10-03 15:17:39 [kv_cache_manager.py:554] [KVC][L5] ======== 分配完成 ========
INFO 10-03 15:17:39 [kv_cache_manager.py:691] [KVC][L5] ======== 调度提交(非分配 S4) ========
```

### 5.4 decode 步 27 跨界（需 1 块，第 512 个 token 触发第 5 块申请）

尾块 102/128 被 decode 逐步填满——步 27 时 num_computed_tokens=512，S1 汇总值切换为“需分配 1 块”，S3 弹出新块 [6]，S4 把刚填满的块 5 入表（map 3→4）：

```
INFO 10-03 15:17:39 [kv_cache_manager.py:399] [KVC][L5] 分配 KVCacheManager.allocate_slots 进入: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_new_tokens=1(new), num_new_computed_tokens=0(new_comp), num_external_computed_tokens=0(ext_comp=P传D_KV), num_encoder_tokens=0, num_lookahead_tokens=0, request.num_computed_tokens=512(comp), request.num_tokens=512, delay_cache_blocks=False
INFO 10-03 15:17:39 [kv_cache_manager.py:409] [KVC][L5] 分配布局: |<comp>=512 |<new_comp>=0 |<ext_comp>=0(P传D) |<new>=1 |<lookahead>=0| num_local_computed_tokens=512 total_computed_tokens=512 to_be_computed=1
INFO 10-03 15:17:39 [kv_cache_manager.py:417] [KVC][L5] --- S1: 容量检查---
INFO 10-03 15:17:39 [kv_cache_coordinator.py:188] [KVC][L4] S1 KVCacheCoordinator.get_num_blocks_to_allocate: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens=513 -> 需分配 1 块(含touch需腾挪的块)
INFO 10-03 15:17:39 [kv_cache_manager.py:462] [KVC][L5] S1 get_num_blocks_to_allocate: 需分配 1 块 vs 可用 13286 块 (free=13286 - reserved=0)
INFO 10-03 15:17:39 [kv_cache_manager.py:496] [KVC][L5] --- S2: 无前缀缓冲, 无需 touch ---
INFO 10-03 15:17:39 [kv_cache_manager.py:500] [KVC][L5] --- S3: 新块分配 ---
INFO 10-03 15:17:39 [block_pool.py:416] [KVC][L2] S3 BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=13285
INFO 10-03 15:17:39 [single_type_kv_cache_manager.py:303] [KVC][L3] S3 SingleTypeKVCacheManager.allocate_new_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens=513, block_size=128, 需 5 块 - 已有 4 = 新分配 1 块 [6], 持有 req_blocks=[1, 2, 4, 5, 6]
INFO 10-03 15:17:39 [kv_cache_coordinator.py:261] [KVC][L4] S3 KVCacheCoordinator.allocate_new_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens=513 -> [[6]]
INFO 10-03 15:17:39 [kv_cache_manager.py:511] [KVC][L5] S3 allocate_new_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens_need_slot=513 -> 新块 [6]
INFO 10-03 15:17:39 [kv_cache_manager.py:538] [KVC][L5] --- S4: 满块入缓存 ---
INFO 10-03 15:17:39 [single_type_kv_cache_manager.py:357] [KVC][L3] S4 SingleTypeKVCacheManager.cache_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa, num_tokens=512, block_size=128, 已缓存 3 块 -> 满块数 4
INFO 10-03 15:17:39 [block_pool.py:113] [KVC][L2] S4 BlockHashToBlockMap.insert: key=BlockHashWithGroupId(hash=8b76b45d8148, group_id=0) <- KVCacheBlock(block_id=5), map size=4
INFO 10-03 15:17:39 [block_pool.py:345] [KVC][L2] S4 BlockPool.cache_full_blocks: req=cmpl-863603840a5003dc-0-9eb8d7fa 新满块 1 块 block_ids=[5] 入 BlockHashToBlockMap (num_cached_blocks 3 -> 4, group_id=0, map size=4)
```

### 5.5 KVS 物理归档与离线互证（4 worker 横幅两行，五块全景）

R TERM 同样每 worker 横幅两行：blk=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8]（region=520/520 = 4×128 + 8；w_tok=520 = 486 + 34，第 35 个输出 token 仅采样不写卡），每 worker 21,017,419 B = 20.0 MiB：

```
INFO 10-03 15:17:39 [model_runner_v1.py:2573] [KVC][KVS] ======== 开始保存物理tensor worker=PP0_TP0 dev=npu:0 TERM seq=2 req尾8=9eb8d7fa: K_cache/V_cache(双独立张量池, 每块 shape=(128,4,128) torch.bfloat16, 含未写槽位) 16 层 × 5 块 blk=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8] -> req2_9eb8d7fa/kv_pp0tp0.pt ========
INFO 10-03 15:17:40 [model_runner_v1.py:2623] [KVC][KVS] ======== 完成保存物理tensor worker=PP0_TP0 dev=npu:0 seq=2 req尾8=9eb8d7fa: req2_9eb8d7fa/kv_pp0tp0.pt (K_cache/V_cache, 16 层 × 5 块, 21017419 B = 20.0 MiB) 落盘 45.8 ms ========
```

**时序细节**（与 P 的差异，kvc_r.log 行号）：4 卡开始横幅 :701~704 仍先于最后一步调度提交（:705~708）与释放横幅 :709（同步快照先行）；完成横幅 4 条（:716~719）全部落在释放完成（:715）**之后**——R 的 flush 42.9~47.9 ms 由后台线程执行，EngineCore 已先行归还块，落盘早晚互不影响（数据在开始横幅前的同步 clone 已位级脱离物理池）。

**离线互证**（logs/analysis/inspect_prefix.out + logs/analysis/inspect_kv_tensors.out）：

| 验证项 | 结果 | 原始输出 |
|---|---|---|
| 横幅 vs meta / 字节对账 | 开始横幅 worker/blk/cov/group_size = .pt meta **8/8**（group_size=16）；完成横幅字节数 = 实际文件 **8/8** | 实验轮内对账（[4/6] 归档 8/8 + fetch 后横幅 vs meta/字节复核） |
| 首块首 token 跨证 | K 前 3 值 [0.5078, 0.9336, 0.9219] 多轮独立重跑逐位一致——位级确定性互证 | inspect_kv_tensors.out |

### 5.6 结束释放（五块逆序释放与计数自检）

```
INFO 10-03 15:17:40 [kv_cache_manager.py:566] [KVC][L5] ======== 释放 ========
INFO 10-03 15:17:40 [kv_cache_manager.py:568] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-863603840a5003dc-0-9eb8d7fa, 释放前持有 block_table=([1, 2, 4, 5, 6],)
INFO 10-03 15:17:40 [single_type_kv_cache_manager.py:405] [KVC][L3] 释放 SingleTypeKVCacheManager.free: req=cmpl-863603840a5003dc-0-9eb8d7fa, 持有 blocks=[1, 2, 4, 5, 6] (reversed 后释放)
INFO 10-03 15:17:40 [block_pool.py:521] [KVC][L2] 释放 BlockPool.free_blocks: blocks=[(6, 0), (5, 0), (4, 0), (2, 0), (1, 0)] (ref_cnt 已 -1), 归零回收 5 块 [6, 5, 4, 2, 1], append_n -> 队尾(LRU保护)
INFO 10-03 15:17:40 [kv_cache_utils.py:391] [KVC][L2] 释放 FreeKVCacheBlockQueue.append_n(blocks=[6, 5, 4, 2, 1]), num_free_blocks=13290
INFO 10-03 15:17:40 [kv_cache_coordinator.py:299] [KVC][L4] 释放 KVCacheCoordinator.free: req=cmpl-863603840a5003dc-0-9eb8d7fa 已逐组下放第3层释放
INFO 10-03 15:17:40 [kv_cache_coordinator.py:302] [KVC][L4] ======== 释放完成 ========
```

**全程计数自检**（grep 即可复验）：`调度提交` P=1 / R=**35**（prefill 1 步 + 34 个写卡 decode 步；第 35 个输出 token 仅采样不写卡、无提交）；[KVS] 横幅 P=**12**（4 启用 + 4 开始 + 4 完成）/ R=**8**（4 开始 + 4 完成）；S1 汇总值 R = **33×0 + 1×1 + 1×4**（无块步 33 + 跨界 1 + prefill 1）；哈希链 `754104f1d6be → 70ae0ea07102 → 1839e465e0eb`（+ decode 段 `8b76b45d8148`）。

## 6. 实证结论

1. **patch 174 行验证通过**：8/8 应用（vllm 155 + vllm-ascend 19 = [L1] 8 + [KVS] 11，[KVC] 计数与 py_compile 均过）；实验后 revert 8 文件 [KVC] 归零、两仓库 git 0 改动——**补丁可双向复现，容器源码始终未污染**。
2. **S1 段完整自洽**：子步横幅先行（源码 kv_cache_manager.py:417）→ 两次外层探问下钻（coordinator.py:188 ×2）→ 汇总值（kv_cache_manager.py:462）——S1 语义日志全部在子步横幅之内。
3. **S1~S4 全子步可观测**：无块步打出完整四段子步（S1 需分配 0 / S2 无前缀 / S3 无需分配 / S4 维护），R 33 个无块步全部闭合。
4. **KVS 横幅两行式的可读性**：每请求每 worker 仅 2 行（开始声明 + 完成回执），**不随层×块增长**（每轮固定 20 行：P 12 = 4 启用 + 4 开始 + 4 完成，R 8 = 4 开始 + 4 完成）；横幅字段与 .pt meta / 文件字节全链对账 16/16 PASS（实验轮内对账——run_all [4/6] 归档核对 + fetch 后横幅 vs .pt 复核：块表/cov/group_size vs meta 8/8、字节 vs 文件 8/8；flush 32.0~47.9 ms 后台不阻塞、开始横幅先于释放）。
5. **KV 布局与零拷贝实证**：归档张量直接承载布局证据——每块 (128,4,128) bf16、K/V 双独立张量池、未写槽位全零（P b3[68:128] 与 R b6[8:128]）；前缀复用 = 同批物理字节（公共块 [1,2] 四 worker × 16 层 × 2 池逐位相等）；L00 统计跨 run 与历史轮打印记录 4 位相同——多轮独立重跑位级确定性互证。
6. **容器回收闭环**：服务已杀（0 进程）、8 补丁 revert 归零、两仓库 git 0 改动、.orig/打包残留清理。

## 7. 实测产物

产物（logs/ 四子目录（与 scripts/ 前四目录一一对应）+ tensors/，容器时钟 2026-10-03 15:16~15:18，本轮 rid 尾8：P=bdd8c88d / R=9eb8d7fa）：

| 产物 | 说明 |
|---|---|
| `logs/server/llama-3-8b.log`（1182 行） | 服务全量日志（启动 :1~394 + P :395~460 + R :461~1182，带进程前缀完整版） |
| `logs/patchs/kvc_startup.log`（172 行） | 启动期 [KVC] 拆解轨迹（CFG 88 + L1 76 + 逻辑侧 8） |
| `logs/patchs/kvc_p.log`（61 行）/ `logs/patchs/kvc_r.log`（719 行） | P / R 运行期 [KVC]+[KVS] 拆解轨迹 |
| `logs/patchs/kvs_archive_lines.log`（20 行） | [KVS] 横幅留痕（4 启用 + 8 开始 + 8 完成） |
| `logs/curl/curl_screen.log` + `req_*.json` / `resp_*.json` | curl 命令与响应打屏（用例设计见 §1.2） |
| `logs/analysis/inspect_kv_tensors.out` | 归档查看报告（块-行映射查看器：逐请求 × worker × block 的块-行映射网格（1 block 竖跨 group_size 层逐层列行号）+ 第 0 层张量 shape/dtype/预览示例） |
| `logs/analysis/inspect_prefix.out` | 前缀复用关系 pairwise 列表（共享表头块/命中 tokens/早→晚块表对照）；横幅 vs meta/字节对账在实验轮内完成（[4/6] + fetch 复核） |
| `logs/server/run_all_screen.log` | 容器侧一键编排留痕（patch → serve → curl → 初检 → [6/6] 打包） |
| `tensors/req{seq}_{rid尾8}/kv_pp?tp?.pt` **× 8**（一请求一子目录 × 4 worker）| 物理 KV 归档（P 12.0 MiB + R 20.0 MiB 每 worker，meta.group_size=16；fetch 后 8/8 就位核对） |

（P/R 分界行号 395 / 460 由 curl_p_r.sh 运行时确定，不落盘）
