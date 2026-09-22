# Issue #39734 | Admission 死锁与每请求准入上限（`max_admission_blocks_per_request`）

| 项 | 内容 |
|---|---|
| Issue | [#39734](https://github.com/vllm-project/vllm/issues/39734) "Scheduler deadlocks when request exceeds KV cache capacity but within max_model_len"（2026 年报告；2×H100 TP=2，Gemma-4-31B，vLLM 0.19.1rc1.dev221） |
| 源码引用 | `single_type_kv_cache_manager.py:178-191`（`get_num_blocks_to_allocate` 的准入钳制；注释原文 "Drift between the two would re-introduce the deadlock from issue #39734 or, worse, mid-prefill OOM."） |
| 本仓库关联修复 | PR [#40946](https://github.com/vllm-project/vllm/pull/40946) "Cap SWA/chunked-local runtime admission to startup pool-sizing bound"（`84c276d7ea`，2026-04-26，引入 `max_admission_blocks_per_request`；本地 `git log -S` 实查）；后续同域修正 PR [#41282](https://github.com/vllm-project/vllm/pull/41282) "[Bugfix] Fix failure to allocate KV blocks error"（`fa8accac62`，2026-04-29） |
| 上游脉络 | issue [#55800](https://github.com/vllm-project/vllm/issues/55800) 的梳理："#39734: origin of the apply_admission_cap mechanism"，其后继修复为 "[Bugfix] Resolve admission deadlock for blocked-waiting statuses" |
| 类型 | 调度死锁 / **准入（admission）与容量一致性** |

## 一句话摘要

请求 token 数超过 KV cache 总容量、但仍小于 `max_model_len` 时，该请求永远无法被准入却也永远不被拒绝，卡死在 waiting 队列头部阻塞所有后续请求（head-of-line blocking）；更一般地，一旦"**运行时准入估计**"与"**启动期池容量估计**"发生漂移，就会得到 admission 死锁（所有请求集体 blocked、无一能完成释放）或 mid-prefill OOM。本仓库的修复：SWA/chunked-local 的每请求准入需求被钳制到与启动池容量估算**同一个**"单一事实来源"上。

---

## 一、问题现象（issue #39734 原始报告）

- 模型 Gemma-4-31B（60 层、大 head dim，每 token KV 成本高）在 2×H100 TP=2 下：`GPU KV cache size: 76,640 tokens`，而 `max_model_len` 默认 `262,144`；
- 发送一个介于两者之间的请求（如 ~100k tokens）后：

```text
Waiting: 1 reqs, Running: 0     ← 零吞吐，永久停滞
```

- 该请求**卡在 waiting 队列头**，所有后续推理请求被阻塞；`/v1/models` 等非推理端点正常；
- 只有客户端超时断开（abort）才恢复；
- issue 给出的 workaround：`--no-scheduler-reserve-full-isl`，或把 `--max-model-len` 设为 KV cache 容量。

## 二、根因分析

### 2.1 直接根因（issue 报告的 scheduler 形态）

waiting 队列的调度循环中，`scheduler_reserve_full_isl=True`（默认）时 `can_fit_full_sequence()` 对**永远装不下**的请求返回 False，调度器 break 出循环但**不 pop 该请求**→ 下一轮调度再次遇到同一请求、再次失败、再次 break → 队列头永久阻塞（issue 作者指出根因在 `vllm/v1/core/sched/scheduler.py` 的 waiting 调度循环）。

### 2.2 泛化根因（本仓库注释将 #39734 定为这一族的代表）

死锁的深层条件是**准入估计与容量池估计不一致（drift）**：

- 启动期：池容量按各类 spec 的 `max_memory_usage_bytes` 估算。对 SWA/chunked-local 这类**回收感知（recycling-aware）** spec，每请求真实占用会收敛到"窗口 + 在途 tokens"的常数——`remove_skipped_blocks` 在 `allocate_slots` 里、每个 chunk 的 `get_num_blocks_to_allocate` 之前运行，把窗外已处理块持续还回池里；
- 运行时：若准入/预留统计按**全序列长度**估算这类请求的需求（如 200k token 的 SWA 请求按 `cdiv(200k, block_size)` 计），会严重虚高；
- 后果一（死锁）：预留统计显示池耗尽 ⇒ 在跑的 prefill 拿不到新块、WAITING 请求进不来、也**没有任何请求能完成并释放块** ⇒ 集体 blocked-waiting 死锁；
- 后果二（OOM）：反向漂移（低估需求/高估容量）⇒ prefill 中途无块可分 ⇒ mid-prefill OOM。

### 2.3 为什么 full attention 型 spec 不需要 cap

full attention 持有全部块直到请求结束，"全序列块数"**就是**真实峰值——钳制它反而会错误放行装不下的请求（即 #39734 报告的那类"永远装不下"场景，须由调度器显式拒绝）。只有"真实峰值 ≪ 全序列"的回收感知 spec 才需要（也才应该）用窗口上限钳制。

## 三、解决办法

### 3.1 单一事实来源：`max_admission_blocks_per_request`

定义在 spec 上（`vllm/v1/kv_cache_interface.py`），同一 cap 同时服务**启动池容量估算**和**运行时准入门**：

| Spec | 公式 | 位置 |
|---|---|---|
| `SlidingWindowSpec` | `cdiv(min(sliding_window - 1 + max_in_flight_tokens, max_model_len), block_size) + 1`（+1：窗口起点可能不在块边界；例：块 4 token、6-token 窗口 [CDEF] 需 [XXCD][EF] 两块） | `kv_cache_interface.py:567-588` |
| `ChunkedLocalAttentionSpec` | `cdiv(min(attention_chunk_size + max_in_flight_tokens, max_model_len), block_size)` | `kv_cache_interface.py:498-519` |

- `max_in_flight_tokens`：已调度未结算的最大 token 数（每并发步一批）；
- 启动侧：`max_memory_usage_bytes = max_blocks * page_size_bytes`（`:521-526`、`:590-598`）；
- docstring 明示设计意图：
  > "Single source of truth for both startup pool sizing (`max_memory_usage_bytes`) and the runtime admission gate, so requests admitted by startup can also be admitted at runtime."

### 3.2 运行时准入门

- `SingleTypeKVCacheManager` 构造参数 `max_admission_blocks_per_request`（`single_type_kv_cache_manager.py:54`，仅回收感知 spec 传入；`None` = 不钳制；`:83` 存储）；
- `get_num_blocks_to_allocate`（`:148-191`）在 `apply_admission_cap=True` 时：

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
if apply_admission_cap and self._max_admission_blocks_per_request is not None:
    # Recycling-aware specs (SWA, chunked-local) cap the per-request
    # reservation here so admission matches the startup pool sizer
    # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
    # chunked-local counterpart). `remove_skipped_blocks` runs from
    # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
    # so per-request peak real-held blocks <= this cap, which keeps
    # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
    # Drift between the two would re-introduce the deadlock from
    # issue #39734 or, worse, mid-prefill OOM.
    num_required_blocks = min(num_required_blocks,
                              self._max_admission_blocks_per_request)
```

**进展保证（不变量）**：对回收感知 spec，每请求真实峰值 ≤ cap；准入门用同一 cap ⇒ `sum(reservations) ≤ pool ⟺ sum(peak_real_held) ≤ pool` ⇒ 每个已被准入的请求**总能推进到完成**（完成即释放）—— admission 死锁与 mid-prefill OOM 同时被排除。

### 3.3 接线：谁以 `apply_admission_cap=True` 调用

| 调用点 | 用途 |
|---|---|
| `kv_cache_manager.py:472-488`（`full_sequence_must_fit` 分支） | 准入硬检查：`required_blocks = num_blocks_to_allocate + watermark_blocks` 与 `get_num_free_blocks()` 比较，放不下直接拒绝（返回 None） |
| `scheduler.py:2559-2571` `_request_remaining_blocks` → `:2573-2578` `_inflight_prefill_reserved_blocks` | 在途 prefill 的剩余预留统计——钳制后 SWA/chunked-local 的预留不再虚高，调度器据此决定还能放进多少新请求 |

## 四、修复脉络（本地 git 实查 + 上游梳理）

| 时间 | 事件 |
|---|---|
| 2026 年 | issue #39734 报告 Gemma-4-31B 上 waiting 队头死锁（76k 容量 vs 262k max_model_len）；建议方向：立即拒绝"空池也放不下"的请求、或启动时对齐 max_model_len 与容量 |
| 2026-04-26 | PR #40946（`84c276d7ea`）"Cap SWA/chunked-local runtime admission to startup pool-sizing bound"——引入 `max_admission_blocks_per_request` 机制（本仓库源码注释即引 #39734） |
| 2026-04-29 | PR #41282（`fa8accac62`）"[Bugfix] Fix failure to allocate KV blocks error"——同域后续修正 |
| 其后 | #55800 梳理：#39734 是 apply_admission_cap 机制的源头；后继修复 "[Bugfix] Resolve admission deadlock for blocked-waiting statuses" 继续演进该族 |

> 阅读提示：issue 原文聚焦 scheduler 层的 never-fit-not-pop 死锁；本仓库引用的是池管理这一侧的同族修复——把准入估算与池容量估算对齐。两侧合起来构成对 "admission 死锁" 类问题的完整回答。

## 五、关联源码（当前 main，2026-09 实查）

| 位置 | 说明 |
|---|---|
| `single_type_kv_cache_manager.py:148-191` | `get_num_blocks_to_allocate` + 准入钳制（**#39734 注释在此**） |
| `single_type_kv_cache_manager.py:54, 83` | cap 的构造注入与存储（仅回收感知 spec） |
| `kv_cache_interface.py:498-526` | `ChunkedLocalAttentionSpec.max_admission_blocks_per_request` / `max_memory_usage_bytes` |
| `kv_cache_interface.py:538-598` | `SlidingWindowSpec` 同上（含边界 +1 注释） |
| `kv_cache_manager.py:472-488` | 准入硬检查（watermark + 自由块比较） |
| `sched/scheduler.py:2559-2578` | 在途 prefill 预留统计（`apply_admission_cap=True`） |

## 六、易混点

- **cap ≠ 全序列块数**：只对 SWA / chunked-local 生效，数值由"窗口/块大小 + 在途 token"决定；
- 与 #33775（驱逐竞态）、#32802（EAGLE 螺旋）机制上完全无关，属"容量-准入一致性"这一独立维度；
- 触发姿势常见于长上下文模型（Gemma-4 等大 KV/token 成本 + `max_model_len` 默认远大于 KV 容量）与 hybrid SWA 模型的 chunked prefill。
