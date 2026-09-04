# D 端到端：一条请求的 KV Cache 管理全流程（Llama-3-8B · pp2tp2）

> **格子**：`D1`～`D6`。场景锚点：**Full Attention · Llama-3-8B · PP2×TP2**（→ 00 章 0.5.1）。
> **一句话**：把 00/A/B/C 串成一条链——**准入（D2）→ 前缀查找（D3）→ 分配（D4）→ 执行（D5）→ 收尾（D6）**，读完能回答"任一时刻系统在做什么、哪些状态在迁移"。
> **配图**：`draw/D_端到端.drawio`（P1～P5，与小节对应见章末"配图"表）。
> **收束章纪律**：本章只讲"何时发生、按什么顺序发生"；组件结构（→ B5）、块池与前缀缓存机制（→ C1～C3）、写/读路径（→ C5/C6）**只留链接不重讲**；横向差异（混合/并行/Connector）→ E 章。
>
> **源码路径约定**（行号均已在本仓库逐行核实）：
> - `vllm/v1/core/kv_cache_manager.py`（五层门面，`KVCacheManager:117`）
> - `vllm/v1/core/sched/scheduler.py`（调度器，`schedule:427`）
> - `vllm/v1/core/kv_cache_utils.py`（启动期预算校验与块元数据）
> - 下文缩写：KCM=`kv_cache_manager.py`、SCH=`sched/scheduler.py`、KCU=`kv_cache_utils.py`。

---

## D1 示例请求 + 总时序图定义

### D1.1 示例请求与前置状态

全章共用一个示例（沿用旧材料口径，数字与 B 章/C 章示例同为一条主线，不重复推导）：

**前置请求 P**（先于 R 服务、已正常结束）：

- prompt = 共享前缀 SP（32 token）+ P 自己的追问（与 R 不同）；
- 服务时把 SP 写成满块 0/1，写满即哈希入前缀缓存表（→ C3.4）；
- 结束后块 0/1 成为**带哈希缓存块**：释放时 `ref_cnt` 归 0 后 append 进 free 队列队尾（LRU 保护），并登记在 `cached_block_hash_to_block`（→ C1.4）。

**示例请求 R**（贯穿 D2～D6）：

| 项 | 值 | 说明 |
|---|---|---|
| prompt | SP（32 token）+ 追加问题（38 token）= **70 token** | 前 32 token 与 P 完全相同 |
| `max_tokens` | **32** | 31 个落 KV，第 32 个只采样（D5.4） |
| `block_size` | 16 | → 00 章 0.3 |
| 预期块占用 | 最终 7 块（块 0..6） | 5 块 prefill + 2 块 decode 跨界 |

- **块号约定**：块 0/1/2… 仅为叙述示意；实际 `block_id=0` 是 `null_block`（→ C1.5），真实分配避开它，不影响流程。
- 部署口径（→ B6）：32 层按 PP2 切成每 worker 16 层，8 个 KV 头按 TP2 切成每 worker 4 头；`page_size_bytes=32 KiB`、假设 `num_blocks=4096`（→ B3.4）。调度器视角全天真为**单 group**，对布局透明。

### D1.2 宏观路径与每步骨架

- R 的宏观路径一行：**入队（WAITING）→ 首次调度 prefill（命中 2 块 + 新分 3 块）→ RUNNING** → decode 续写 31 步 → 完成 → 释放。
- 引擎每步由 `EngineCore.step()` 驱动 `schedule() → execute_model() → sample_tokens()`：
  - **调度半步**（CPU，只动 `block_id`）：遍历请求做 D2 门控、D3 查前缀、D4 分配，产出 `SchedulerOutput`；
  - **执行半步**（GPU）：Worker 清零新块、按 `block_table` 写/读 KV、采样（D5）。
- 调度没有独立的 prefill/decode 全局阶段：RUNNING 与 WAITING 共享 `token_budget`（SCH:447），先 RUNNING 后 WAITING（D2.1）。

### D1.3 全程时间线表（P1 图的叙事版）

| # | 时刻 | 系统 in flight | 状态迁移 | 归属 |
|---|---|---|---|---|
| 0 | `add_request`(SCH:2157) | 构造 Request，`update_block_hashes()` 预计算 4 个链式哈希（`request.py:257`）→ 入 WAITING（SCH:2007） | 无块，纯哈希 | D2 |
| 1 | `schedule()` 门控 | `num_computed_tokens==0` → 查前缀 + `allocate_slots(full_sequence_must_fit)` | WAITING →（准入）→ RUNNING | D2 |
| 2 | 前缀查找 | `get_computed_blocks`：4 哈希查表，命中块 0/1，`hit_length=32` | 只读不写 | D3 |
| 3 | 分配 | `allocate_slots` 三阶段：检查 → touch 0/1 → 新分 2/3/4 → 满块 2/3 入哈希 | `req_to_blocks[R]=[0,1,2,3,4]` | D4 |
| 4 | 执行 | `SchedulerOutput`（清零 2/3/4）→ 清零 → 写 38 token KV（命中 32 不重算）→ sample | 第 1 个输出 token | D5 |
| 5 | decode ×31 | 每步 1 token：0 块或 1 块；步 10/26 块满补录入表；步 11/27 各弹 1 块清零 | 块表长到 [0..6] | D5 |
| 6 | finish | `check_stop` → `FINISHED_*` → 从 RUNNING 移除 → free | RUNNING → 结束 | D6 |
| 7 | 释放 | 逆序归还 6→0；有哈希进队尾、无哈希进队首、共享块只减计数 | 块回池，前缀缓存仍然活着 | D6 |

> 配图：P1 · 端到端总览时序（归属 D1，兼作 D2～D6 的地图）。

---

## D2 准入与调度：WAITING → RUNNING 门控

### D2.1 schedule() 骨架：先 RUNNING 后 WAITING

`schedule()`（SCH:427）每步按统一预算依次处理两类请求：

```python
token_budget = self.max_num_scheduled_tokens        # SCH:447
# 第一遍：RUNNING 请求（decode / chunked prefill 中间片）   SCH:473
while req_index < len(self.running) and token_budget > 0: ...
# 第二遍：WAITING 请求（新请求首次 prefill）                SCH:671
while (self.waiting or self.skipped_waiting) and token_budget > 0: ...
```

- 调度注释明说：**没有"prefill 阶段 / decode 阶段"**（SCH:429-438），每个请求只有 `num_computed_tokens` 追赶 `num_tokens` 的进度条；
- 推论：RUNNING 里可能躺着 chunked prefill 的中间片（`is_prefill_chunk`），它们**优先于**新 WAITING 消耗预算（D5.3）。

### D2.2 WAITING 门控链（每个请求过三道门）

| 门 | 源码 | 检查内容 | 失败出口 |
|---|---|---|---|
| ① 结构门 | SCH:675-710 | `max_num_running_reqs` / blocked 状态 + LoRA 上限 | pop 出队 → 进 `step_skipped_waiting`，本轮跳过 |
| ② 前缀查找 | SCH:718-739 | `num_computed_tokens==0` 时 `get_computed_blocks`（D3） | （只读，无失败出口） |
| ③ 空间门 | SCH:946-958 | `allocate_slots(..., full_sequence_must_fit=True)`（D4） | 返回 `None` → `break`（SCH:960-967），留在 WAITING 下轮再试 |

- 空间门参数注入点：`full_sequence_must_fit=self.scheduler_reserve_full_isl`（SCH:955）、`reserved_blocks`（异步 KV 加载预留，主线 0，→ E4）、`has_scheduled_reqs=bool(self.running)`（SCH:957，控制 watermark 生效与否）。
- 三道全过：`request.status = RequestStatus.RUNNING`（SCH:1047）+ `num_computed_tokens` 落账（SCH:1048），进 `scheduled_new_reqs`。

### D2.3 full_sequence_must_fit 检查（按"完整序列"先算一笔账）

`allocate_slots` 内部的准入支路（KCM:472-488）：

```python
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)   # 整条序列
    num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(
        num_tokens=full_num_tokens, ..., apply_admission_cap=True)  # 按"放得下整条"算
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():     # 对比"当前全池空闲"
        return None                                                 # 整序列放不下 → 拒绝
```

- 动机（docstring 原意）：chunked prefill 若只检查**第一个 chunk**，可能放行一条永远放不下的请求，之后反复抢占抖动；
- R 的演算：`full_num_tokens=70`，命中 2 块 → 需要 `cdiv(70,16)−2 = 3` 块（+watermark），对 4096 池轻松通过；
- 开关：`scheduler_reserve_full_isl` 默认 `True`（`config/scheduler.py:130`），即**默认开启**这条准入门。

### D2.4 watermark headroom（给在跑请求留余量）

```python
# KCM:168-171：构造期把比例换算成块数
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
# KCM:463-470：分配期按请求状态临时叠加
watermark_blocks = 0
if has_scheduled_reqs and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
    watermark_blocks = self.watermark_blocks
```

- **语义**：WAITING/PREEMPTED 请求准入时，额外要求"空闲块里还得留出 `watermark_blocks`"，防止准入把池子榨干、随后在跑请求集体无块可用而频繁抢占；
- 只影响**准入判定**两张比较表（D2.3 的 1a 与 D4.1 的 1e），**不改 `num_blocks`、不占实物块**（启动期语义 → B3.5）；
- 生效条件两个都要满足：`has_scheduled_reqs=True`（本步已有请求在跑）且请求状态是 WAITING/PREEMPTED；
- 默认关闭：`watermark: float = 0.0`（`config/scheduler.py:136`），即默认不启用该余量（R 的演算里 `watermark_blocks=0`）；
- 启动期关联：`check_enough_kv_cache_memory`（KCU:854）保证池子至少装下一条 `max_model_len` 序列、`may_override_num_blocks`（KCU:962）允许手工覆盖块数——两者都只发生在 B 章启动期，运行期不再碰。

### D2.5 可用块的完整口径

```python
# KCM:521-527（1e，非准入支路）
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks  = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks: return None
```

| 分量 | 含义 | 主线取值 |
|---|---|---|
| `get_num_free_blocks()` | 空闲双链长度（含队尾带哈希的待命块——它们随时可被驱逐顶上，→ C1.3/C3.5） | 动态 |
| `reserved_blocks` | 给其他 in-flight 序列保底（异步 KV 加载场景） | 0（无 connector 步） |
| `watermark_blocks` | D2.4 的准入余量 | 0（默认） |

- 结论一句话：**主线上"能不能调度"几乎就是"空闲块够不够"**；两个附加项是给极端场景插的安全销。

> 配图：P2 · 准入门控流程（归属 D2）。

---

## D3 prefill 首步：`get_computed_blocks` 前缀查找

### D3.1 触发条件与短路

- 只有**首次调度**满足 `request.num_computed_tokens == 0`（SCH:718）才触发；RUNNING 的 decode/续 chunk 步直接跳过本节；
- 短路出口：`prefix_cache_lookup_enabled`（KCM:214-216）为假（前缀缓存关闭 / 请求要求跳过 KV 读，如 prompt logprobs）→ 直接返回 `empty, 0, 0`（KCM:250-251），不进查找。

### D3.2 查找流程（机制 → C3.3，此处只给镜头）

`get_computed_blocks`（KCM:229-295）把请求的链式哈希交给协调器逐块查表：

| 步骤 | 锚点 | R 的落点 |
|---|---|---|
| ① 命中上限 = `num_tokens − 1` | KCM:253-259 | 69：即使全命中也要重算最后 1 个 token 拿 logits |
| ② `coordinator.find_longest_cache_hit`（单组直通） | `kv_cache_coordinator.py:486-503` | 透传给 `FullAttentionManager` |
| ③ 逐块查 `BlockPool.get_cached_block`，miss 即 break | `block_pool.py:198-223` | `hash(t0-15)`→块0，`hash(t16-31)`→块1，`hash(t32-47)`→miss 断链 |
| ④ 返回 `(KVCacheBlocks, hit_length, 0)` | KCM:294-295 | `([块0, 块1], 32, 0)` |

- 命中的是**满块**且**只读不写**：`ref_cnt` 不动、临时 key 用完即弃（→ C3.1 三级哈希形态）；真正的引用固化等 D4.2 的 touch；
- **ext_comp=0 简化**：主线无 KV Connector，调度器走 `get_computed_blocks`（SCH:739）而非 `get_computed_blocks_for_connector`，`num_external_computed_tokens` 全程 0——connector 注入的外部已算 token 分支（SCH:742-799）整条跳过 → E4；
- **out-of-date 处理**：命中块若之后发生"部分条目 → 满块晋升"，入表前会清旧哈希防过期别名残留（→ C3.4），本主线是升满块的通用规则，查找侧无感知。
- **decode 为什么不查**：续写的是全新 token，无新前缀可命；`comp>0` 与 `new_comp/ext_comp>0` 运行时互斥（D4.0）。

> 配图：P3 的入口段 + P1 的 ① 行（归属 D3）。

---

## D4 `allocate_slots` 三阶段（本章重心）

### D4.0 统一的入参视角：块布局 5 段

`allocate_slots`（KCM:344-565）的 docstring 把一条 token 序列切成 5 段（KCM:390-422）：

```text
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
```

| 段 | 含义 | **R prefill** | **R decode 步 N** |
|---|---|---|---|
| `comp` | 前面步已算完（`request.num_computed_tokens`） | 0 | 70+N−1 |
| `new_comp` | 本步新命中的前缀（= 命中块数 × block_size） | 32 | 0（不再查表） |
| `ext_comp` | connector 侧已算（→ E4） | 0 | 0 |
| `new` | 本步要 GPU 真算（含未验证 draft token） | 38 | 1 |
| `lookahead` | EAGLE 投机前瞻（→ A 章 spec 支） | 0 | 0 |

- 段间规则：`comp>0` 时后两段必为 0（RUNNING 不再看前缀），5 段**永不同时非零**；
- 全部三阶段都围绕一个等式：`to_be_allocated + to_be_computed` 必须**齐**：已有块 + 命中块 + 新块 ≥ 需要的块，否则 `return None`。

### D4.1 阶段①：空间检查（"放得下才动手"）

| 子步骤 | 做什么 | 代码锚点 | 失败出口 |
|---|---|---|---|
| 1a | `full_sequence_must_fit`：按**完整序列**算需求并对比全池空闲（D2.3 已展开），`apply_admission_cap=True` | KCM:472-488 | `return None`（准入拒绝） |
| 1b | 算本轮需要槽位的 token：`num_tokens_need_slot = min(comp + new_comp + new + lookahead, max_model_len)` | KCM:490-493 | —（纯计算） |
| 1c | `remove_skipped_blocks`：先释放窗口外不再需要的块（SWA/R-SWA，待算 token 不受影响） | KCM:504-508 | —（Full Attention 恒 no-op → A4） |
| 1d | `coordinator.get_num_blocks_to_allocate`：需求 = `max(cdiv(需槽位 token, block_size) − (已持块+命中块), 0)`，RUNNING 走快路径只算差额 | KCM:510-519（下钻 `single_type_kv_cache_manager.py:144-230`） | —（纯计算） |
| 1e | 空间判决：`required(需求+watermark) > available(free−reserved)` → 不放行 | KCM:521-527 | `return None`（空间不足） |

- R 的账：`need_slot=70`，需求 `5−2=3` 块 → 通过；decode 步的账（快路径）：`cdiv(70+N,16) − len(块表)`，多数为 0，跨块边界时为 1。
- 1c 放在分配之前的原因（源码注释）：提前释放能减少随后需要驱逐的块数，且**即使本轮最终调度失败，这个释放也是安全的**。

### D4.2 阶段②：前缀处理（touch 命中块）

| 子步骤 | 做什么 | 代码锚点 | 失败出口 |
|---|---|---|---|
| 2a | 进入条件（满足其一）：有命中块 / `num_external_computed_tokens>0` | KCM:529-532 | 不满足 → 整段跳过（decode 常态） |
| 2b | `allocate_new_computed_blocks` 两段式：先逐组 `add_local_computed_blocks`（把命中块挂进 `req_to_blocks`、记 `num_cached_block`），必要时再处理 ext token（主线无） | `kv_cache_coordinator.py:192-236` | — |
| 2c | `BlockPool.touch`：`ref_cnt==0` 的命中块先从 free 队列摘出，再 `ref_cnt+=1`（零拷贝共享） | `block_pool.py:702-717`（机制 → C2.2） | — |

- **两阶段协议（先 touch 后分新块）**是 KCM:529-532 与 KCM:542-547 顺序不能颠倒的原因：新块分配可能触发驱逐（弹带哈希块 → C2.1 第③步），若命中块还没 `ref_cnt++` 会被误驱逐（修复 issue #33775）；
- R 的账：块 0/1 `ref_cnt: 1→2`（P 仍持 1、R 挂 1），`req_to_blocks[R] = [块0, 块1]`，`num_cached_block[R]=2`。

### D4.3 阶段③：新块分配 + 顺手缓存

| 子步骤 | 做什么 | 代码锚点 | 失败出口 |
|---|---|---|---|
| 3a | `coordinator.allocate_new_blocks` → `BlockPool.get_new_blocks(n)`：队首弹出 n 块、清旧哈希（隐式驱逐）、`ref_cnt=1`、append 进块表、记入 `new_block_ids` 待清零清单 | KCM:542-547；`block_pool.py:647-677`（→ C2.1） | 上层 1e 已保证够用，池内不足直接 `ValueError`（防御断言 `:658`） |
| 3b | `delay_cache_blocks`（P/D 远程传输未落地的块先不入缓存） | KCM:549-552 | 命中 → 提前 return，跳过 3c（→ E4） |
| 3c | `cache_blocks`：`num_tokens_to_cache = min(已缓存+本轮, request.num_tokens)`（cap 排除未定稿 draft token）→ 满块切片入哈希表，**幂等** | KCM:554-563（机制 → C3.4） | — |
| 3d | 返回 `KVCacheBlocks(new_blocks)`；`new_block_ids` 由 `take_new_block_ids`（KCM:796）在打包输出时 drain（SCH:1233-1245）→ `SchedulerOutput.new_block_ids_to_zero` | KCM:565 | — |

- 命中块早退：块 0/1 的哈希早已在表，2c 已把它们记入 `num_cached_block[R]`，幂等闸门直接跳过（→ C3.4 第①步）；
- R 的账：`get_new_blocks(3)` 弹 2/3/4 → `block_table=[0,1,2,3,4]`；3c 把满块 2/3（`hash(t32-47)`/`hash(t48-63)`）入表，块 4 只有 6/16 不入；`new_block_ids=[2,3,4]` 等待清零（D5.2）。

### D4.4 失败出口全表：`return None` 之后发生什么

| 谁拿到 `None` | 位置 | 转写 |
|---|---|---|
| WAITING 门控 | SCH:960-967 | 解 `encoder cache` 占用后 `break`——请求**留在 WAITING**，下一轮重试（选择权交给下轮调度） |
| RUNNING 循环 | SCH:565-613（`while True` 抢占循环） | 抢占一个请求腾块（PRIORITY 取 `(priority, arrival_time)` 最大者；默认 FCFS 直接 `running.pop()` 队尾）→ `_preempt_request` 转写为重算态 → **重试** `allocate_slots`；若把自己也抢完仍为 `None` → `break` 放弃本请求 |

- **GRACEFUL/CHECK 转写**：v0 时代 allocate 不足会区分 `PreemptionMode.GRACEFUL/CHECK`（先试"再分配"，退让失败才重算）；V1 收敛后只剩一种语义——**RECOMPUTE**（块全部回池 + `num_computed_tokens` 清零 + 回 WAITING 重头 prefill），见 D6.2；
- 这也是 D2 门控存在的意义：准入门拦得越准，走到这条抢占重算路径的次数越少。

### D4.5 R 的三阶段数字账本（prefill 首步收口）

| 阶段 | 变化 |
|---|---|
| ① 检查 | `need_slot=70`；需求 `3` 块；`free ≫ 3+0` → 通过 |
| ② 前缀 | 块 0/1 `ref_cnt 1→2`，`req_to_blocks[R]=[0,1]` |
| ③ 分配 | 弹 2/3/4 `ref_cnt=1`，`block_table=[0,1,2,3,4]`；满 2/3 入哈希；`new_block_ids=[2,3,4]` |
| 状态 | `req_to_blocks[R]` 长 5；待清零 3 块；新扣 `token_budget-=38` |

> 配图：P3 · allocate_slots 三阶段流程图（归属 D4）。

---

## D5 执行：SchedulerOutput 下发 → 写块 → 抽样（与 C5/C6 链接）

### D5.1 SchedulerOutput 下发（打包三路块数据）

调度半步的收尾把 KV 相关数据打进一个不可变结构（组装点 SCH:1096-1202，闭环细节 → C4）：

| 三路 | 内容 | R 首步的值 |
|---|---|---|
| 新请求 | `NewRequestData.block_ids`（整表） | `([0,1,2,3,4],)` |
| running 增量 | `CachedRequestData.new_block_ids`（行尾追加） | decode 步的 `[块5]` / `[块6]` |
| 清零清单 | `new_block_ids_to_zero`（drain `take_new_block_ids`） | `[2,3,4]` |

- 下发经 EngineCore 到**所有 worker**；同一 `block_id` 在 4 张卡上各自落地为本地张量的行（PP 切层/TP 切头，→ E2/C4.3）。

### D5.2 prefill 写入（清零 → slot_mapping → kernel）

| 步骤 | 动作 | 锚点 |
|---|---|---|
| ① 清零新块 | Worker forward 前对 `new_block_ids_to_zero` memset：物理块可能残留上一请求旧 KV，**必须先擦后写** | `gpu_model_runner.py:1147`（→ C4.4） |
| ② 摊 slot | `slot = block_id × block_size + offset`（唯一公式 → C5.1），`slot_mapping` 汇总本步每个新 token 的落位 | `block_table.py:153-182` |
| ③ 写 + 读 | attention kernel 按需把 38 个新 token 的 K/V `store` 进对应槽，并 `load` 命中块 0/1 与同批次历史的 KV | `flash_attn.py:933`（→ C6.1） |
| ④ 采样 | `sample_tokens` → 第 1 个输出 token 回灌 | SCH 上层 `update_from_output` |

- 确认写入与分配的咬合：38 个 `new` token 恰好填满块 2/3 并占用块 4 的 6 个槽——**块写满于 slot 写入的瞬间**，满块的哈希却早在 D4.3 的 3c（forward 之前）就登记了。哈希基于 token ID（内容未算也可入表），满块判定只依赖"块容量"而非"KV 已写完"（→ C3.1）。

### D5.3 chunked prefill 续步（长 prompt 的分段方式）

- 切片规则：`num_new_tokens = request.num_tokens − num_computed_tokens`，经 `long_prefill_token_threshold` 截断、`enable_chunked_prefill` 开关约束，再 `min(num_new_tokens, token_budget)`（SCH:848-887）；
- 续步仍是每步一次完整循环 `allocate_slots`（RUNNING 路径 SCH:566），块表按 chunk 步进增长、每步槽位是该 chunk 的连续区段；
- 中间片被标记 `is_prefill_chunk`（SCH:1290），继续占用 RUNNING 槽位，并抑制同批新 WAITING 的准入（D2.1）；
- 与首步唯一的机制差异：`num_computed_tokens>0` → **不再查前缀**（D3.1）；`comp` 段开始累积。
- R 的 prompt=70 未触发 chunk 切分（70 < budget 阈值），故 R 只有 1 步 prefill——chunked 流程对 R 的上图是对"如果它更长会怎样"的通道说明。

### D5.4 decode 追加 1 token（31 步的减配循环）

- 每步调度**所有** RUNNING 请求，每请求每轮只 append 1 token；全部请求分配完后 `execute_model + sample_tokens` 批内一把完成；
- `allocate_slots` 走快路径（D4.1 / 1d）：当前块未满 → 0 块（`new` slot 落块内空槽）；已满 → 1 块（弹新块 + 清零）；
- `cache_blocks` 每步照调（幂等 → C3.4）：**某块恰好写满的那一步，它的哈希入表**——前缀缓存就是这样在 decode 里持续生长的；
- 第 32 个输出达到 `max_tokens` 只采样、不再落 KV：31 步共落 31 槽（块 4 补 10、块 5 装 16、块 6 装 5）。

### D5.5 prefill vs chunked vs decode：同一次预算的不同写量

| 维度 | prefill（WAITING 首步） | chunked 续步 | decode（RUNNING） |
|---|---|---|---|
| 处理 token 数 | 整段 prompt（受 budget 截断） | 每 chunk 一段 | 每步 1 |
| 前缀查找 | `get_computed_blocks` 有 | 否（`comp>0`） | 否 |
| 分配块数 | 命中不重分 + 一次多块 | 按 chunk 增量补 | 0 或 1 |
| 写入方式 | 一次调度写满整段 slot（清零 = 新块） | 写段内连续 slot | 写 1 个 slot |
| `cache_blocks` | 新满块入表 | 满块逐 chunk 入表 | 块满步补录 |

- **统一心智（本节收尾）**：三种形态都是同一个循环的 `分配块（调度）+ 写 slot（C5）+ 读块（C6）`，差异只在**量的刻度**——这解释了为什么调度器可以为它们共享同一条 `allocate_slots` 路径。

> 配图：P4 · prefill vs decode vs chunked 三栏对比（归属 D5）。

---

## D6 收尾：finish / 抢占 / free 三路径

### D6.1 三条出口一览

| 出口 | 触发 | 状态 | 块的去向 | 锚点 |
|---|---|---|---|---|
| finish | `check_stop`（EOS/stop 串/`max_tokens` 满） | `FINISHED_*` | 全部 free（D6.3） | SCH:1850-1852 |
| 抢占 | `allocate_slots` 不足 + 有可抢占对象 | `PREEMPTED`（可复活） | 全部 free + `num_computed_tokens=0` 重排队（D6.2） | SCH:1247-1269 |
| abort | 客户端断开 / 外部取消 | `FINISHED_ABORTED` 等 | 同 finish（P/D 远程 KV 在途时延迟释放） | SCH:2181-2242 |

- 三条路**殊途同归**：最终都调 `_free_request` → `_free_request_blocks`（SCH:2285-2298）→ `KVCacheManager.free`；是否立即释放由 `defer_block_free` 决定（本步 in-flight 还要写它 → 先 `pop_blocks_for_free` 押后，`deferred_frees` 阶梯释放，SCH:2311-2324）。

### D6.2 抢占：RECOMPUTE 语义（块回池 + 重排队）

```
_preempt_request（SCH:1247-1269）：
  _free_request_blocks(request)            # ① 块全部回池（KV 数据不迁移，重算要重做）
  request.status = RequestStatus.PREEMPTED # ②
  request.num_computed_tokens = 0          # ③ 前缀查找也允许从 0 重来
  self.waiting.prepend_request(request)    # ④ 插回 WAITING 队头（FCFS 优先复活）
```

- 被抢请求从头重新走 D2～D3：重排队后重新 `get_computed_blocks`，自己刚写满的块**很可能反手被自己命中**（→ C3.5），把重算代价降到最低；
- 与 D4.4 的链接：RUNNING 抢占循环里 `num_computed_tokens` 清零正是 RECOMPUTE 全貌——V1 没有半途的"只抢占部分块"模式；
- 恢复路径：重新调度时请求的状态是 `PREEMPTED`，仍然走 D2 门控（watermark 对它生效，D2.4）。

### D6.3 finish 与 free：逆序归还

```
KVCacheManager.free（KCM:567-578）
  ├─ 先释放 partial-tail pins（主线无 → connector 场景 → E4）
  └─ coordinator.free → FullAttentionManager.free（single_type_kv_cache_manager.py:519）→ BlockPool.free_blocks
       # 必须 reversed：尾块先进队首，最近用的最先被复用（→ C2.3）
```

| 释放的块 | 属性 | free 后去向 | R 的数值 |
|---|---|---|---|
| 块 6 | 无哈希（6/16 未满，从未入表） | `ref_cnt 1→0` → **prepend 队首**（最先被复用） | 立即可复用 |
| 块 5/4/3/2 | 有哈希（已入前缀缓存表） | **append 队尾**（LRU 保留） | 哈希条目仍在表 |
| 块 0/1 | 共享命中块（`ref_cnt=2`，P 也在用） | 仅 `ref_cnt 2→1`，**不回收** | 归 P 独占 |

- **"满块已被缓存 → ref_cnt 清零后自然留存"**：`free` 只减引用不删哈希（→ C2.4 驱逐是独立操作），所以块 2/3/4/5 即便回收进队列，其 `cached_block_hash_to_block` 条目依然健在——下一个带同样前缀的请求在 D3 就能命中它们，**这就是 P → R 这种零成本复用闭环的最后一环**。

### D6.4 R 的完整一生（块视角收口）

| 时刻 | 块 0/1 | 块 2/3 | 块 4/5 | 块 6 |
|---|---|---|---|---|
| prefill 首步 | 命中复用，`ref_cnt 1→2` | 新分并入表 | 块 4 新分（6/16） | — |
| decode 31 步 | 不变 | 不变 | 块 4 步 10 填满→补录；步 11 弹块 5 顶上，步 26 填满→补录 | 步 27 弹入（5/16 未满） |
| free | `ref_cnt 2→1`（留 P） | 逆序 append 队尾 | 同左 | prepend 队首 |
| 之后 | 原地驻留 | 表内待命中 | 表内待命中 | 无哈希，最先被复用 |

> 配图：P5 · 释放 / 抢占 / finish 三路径图（归属 D6）。

---

## 配图

`draw/D_端到端.drawio`，共 5 页（每页左上角标注归属格子）：

| 页 | 标题 | 归属 | 要点 |
|---|---|---|---|
| P1 | 端到端总览时序 | D1（兼 D2～D6 地图） | Engine/Scheduler · KVCacheManager · Worker · KV Caches **四泳道**，从 `add_request` 到 `finish` 的全链消息流，左侧标 D2～D6 泳道区间 |
| P2 | 准入门控流程 | D2 | WAITING → 结构门 → `full_sequence_must_fit` 判定 → watermark/reserved 加码 → `return None` 分支 → `status=RUNNING` |
| P3 | allocate_slots 三阶段 | D4 | 阶段①空间检查（1a/1b/1c/1d/1e）→ 阶段②touch 命中 → 阶段③新块 + 缓存；`return None` 的 WAITING/RUNNING 两个失败出口与抢占转写 |
| P4 | prefill vs decode vs chunked | D5 | 三栏对比：请求切分量 / 前缀查找 / 分配块数 / 写入方式（slot 数与连续性）/ `cache_blocks` / 清零清单差异 |
| P5 | 释放 / 抢占 / finish 三路径 | D6 | RUNNING 之后的三条出口；free 逆序归还 + 三类块的池内去向；RECOMPUTE 重算闭环 |

**完成标准自查**：合上文档，顺着 P1 能口头回答——
R 为什么在 D2 被放行（整序列 3 块余量足）；D3 为什么只命中 32 token（链式哈希第 3 块断链 + `num_tokens−1` 上限）；D4 的 `touch` 为什么必须先于 `get_new_blocks`（issue #33775 驱逐竞态）；D5 里 prefill/chunked/decode 为什么共用三阶段（量的刻度不同，机制同套）；D6 里块 6 为什么只进队首而块 2/3/4/5 能被复用（哈希条目随块留存，ref_cnt 归零后自然可命中）。
