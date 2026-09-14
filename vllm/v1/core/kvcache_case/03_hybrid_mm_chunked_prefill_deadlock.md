# Case 03 | Hybrid 模型 + 双大图 + Chunked Prefill：Encoder Cache 调度死锁

| 项 | 内容 |
|---|---|
| 出处 | [vllm issue #40707](https://github.com/vllm-project/vllm/issues/40707)（vLLM v0.19.0，2×A800，Qwen3.5-35B-A3B）；关联 PR：[#30877](https://github.com/vllm-project/vllm/pull/30877)（引入 Mamba prefix caching align 模式）、[#31699](https://github.com/vllm-project/vllm/pull/31699)、[#31857](https://github.com/vllm-project/vllm/pull/31857)（encoder cache 泄漏/释放时机修复） |
| 类型 | 调度死锁（hang） |
| NPU 相关性 | 高：Qwen3.5 是 hybrid（FullAttention + GatedDeltaNet）架构，vllm-ascend 0.18.0 起支持（[vllm-ascend#7103](https://github.com/vllm-project/vllm-ascend/pull/7103) hybrid prefix cache），调度器逻辑对所有后端一视同仁 |
| 难度 | 中高（需要能装下 35B 的卡，A3/多卡 A2） |

## 摘要

hybrid 模型在 `mamba-cache-mode align` 下，prefill chunk 的结束位置必须落在
Mamba 状态可缓存的块边界上。当 chunked prefill 的剩余预算不足以推进到下一个块边界时，
`_mamba_block_aligned_split` 会把 `num_new_tokens` **截断为 0**；
调度器看到空中止就跳过该请求 —— 于是 encoder cache（视觉编码结果）永不释放，
第二张图永远进不了预填充队列，引擎进入永久 hang（显存占用高、算力 0%）。

## 问题现象（issue 原始记录）

- 模型：`Qwen/Qwen3.5-35B-A3B`，默认 `max_num_batched_tokens=8192`；
- 输入：一个请求带**两张 3024×4032 图片**（单张小图或多张小图都正常）；
- 现象：请求永不完成，两张卡显存占用约 89%、**算力 0%**，引擎永不恢复。

## 根因（三机制交互，issue 给出完整数值账）

1. **Encoder cache 预算只够一张图**
   `encoder_cache_size = max(scheduler_config.encoder_cache_size, max_tokens_per_mm_item)`
   （本地 `vllm/v1/core/encoder_cache_manager.py:273` `compute_mm_encoder_budget`）。
   本例 = max(8192, 16384) = 16384。单张 3024×4032 图经 Qwen3.5 视觉编码
   （patch16/merge2/factor32，smart_resize 后 3008×4032）≈ 11844 个视觉 token；
   11844 ≤ 16384 放得下四舍五入一张，两张 23688 > 16384 放不下。

2. **Chunked prefill 下视觉 token 是"边消费边腾位置"**
   图 1 的 encoder 输出占着 cache，要等它的占位 token 全部被消费完才释放；
   图 2 必须等 cache 腾出来才能加载。

3. **Mamba block 对齐把剩余 gap 截成 0（零塌缩）**
   `vllm/v1/core/sched/scheduler.py:357` `_mamba_block_aligned_split`：
   在 `end < last_cache_position` 时 `end = end // block_size * block_size`。
   到图 2 之前只剩 4 个文本 token（ < block_size=16）时被截成 0，
   调度器 `if num_new_tokens == 0: continue` 跳过请求 → 无进展 → 死锁。

issue 中的推演表（图 1 前 30 个文本 token、两图间 3 个文本 token）：

| Chunk | computed | num_new_tokens（对齐前） | 对齐后 | 结果 |
|---|---|---|---|---|
| 1 | 0 | 8192 | 8192 | 正常，图 1 入 cache |
| 2 | 8192 | 3684 | 3680 | 正常，图 1 仍占 cache |
| 3 | 11872 | 4 | **0** | 死锁 |

若保留 4 不截断：`computed = 11876` ≥ 图 1 所需 11874 → 释放，流程继续。

**本地 main 现状**：`scheduler.py:357` 的函数注释明文写着
"May yield an empty chunk (budget cannot reach the next boundary); the caller then skips the request."
—— 零塌缩行为仍在，该问题在当前代码上具备复现条件（这也是把它选进案例库的原因）。

## 如何修复

- issue 作者建议（最小修复）：`num_new_tokens > 0` 且对齐后为 0 时**保留原值**。
  子块级 chunk 的 Mamba 运行状态由 `preprocess_mamba` 的 `mamba_state_idx` 维护，
  只是没有做块边界 checkpoint（与现有注释"小于 block_size 的段不需要特殊处理"一致）。
- 相关已合入的同类修复：
  - [PR #31699](https://github.com/vllm-project/vllm/pull/31699)：encoder cache 提前到
    `_update_after_schedule` 释放，避免调度空洞演变成死锁；
  - [PR #31857](https://github.com/vllm-project/vllm/pull/31857)：waiting 请求的
    encoder cache 泄漏导致 CPU 调度卡死，修复泄漏。
- 此前修复尝试 [PR #36844](https://github.com/vllm-project/vllm/pull/36844)
  （Guard mamba prefill split fragmentation）因 rebase 问题关闭，说明该问题修复在社区有共识但过程反复。

## NPU 复现实验

环境：A3 或 TP4 A2，vllm-ascend 主线（hybrid + prefix caching 支持，参考 #7103）。

服务端（按 issue 复刻，speculative 可去掉以隔离变量）：

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 \
  --max-model-len 32768 \
  --trust-remote-code \
  --enforce-eager
```

客户端：

```bash
# 先准备两张大图（3024x4032 量级，内容任意但两张不同）
python scripts/gen_large_images.py --w 3024 --h 4032 --n 2 --out /tmp/imgs
python scripts/client_two_images_deadlock.py \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.5-35B-A3B \
  --images /tmp/imgs/a.jpg /tmp/imgs/b.jpg --timeout 300
```

### 观测与判定

| 观测点 | 死锁特征 |
|---|---|
| `npu-smi info` | NPU 利用率长期 0%，显存占用高 |
| `/metrics` | `vllm:num_requests_running` 停在 1（或一直 waiting），请求 TTFT 无限等待 |
| 服务端日志 | 该 request 反复不被调度；无崩溃栈 |
| `py-spy dump`（API server 进程） | 调度循环仍在空转（区分"卡在 HCCL/计算"） |

Leak/死锁后恢复手段：重启进程；或把 `--max-num-batched-tokens` 调大到
`>= 单图视觉 token 数 × 2` 观察是否不再触发（对照组）。

## 关联源码（本地 main）

| 位置 | 说明 |
|---|---|
| `vllm/v1/core/sched/scheduler.py:309-318` | `need_mamba_block_aligned_split` / `mamba_partial_cache_hit` 触发条件 |
| `vllm/v1/core/sched/scheduler.py:357-440` | `_mamba_block_aligned_split`（零塌缩行为 + 四类强制停止点） |
| `vllm/v1/core/sched/scheduler.py:540` | 调度主循环里的调用点 |
| `vllm/v1/core/encoder_cache_manager.py:273` | `compute_mm_encoder_budget` |
| `vllm/v1/core/encoder_cache_manager.py:95/185/244` | `check_and_update_cache` / `allocate` / `free`（视觉缓存生命周期） |

## 延伸阅读

- [vllm#36627](https://github.com/vllm-project/vllm/issues/36627)：qwen3.5 vs qwen3 性能讨论，最早发现该调度空转。
- 案例 05（abort 风暴）会把 encoder cache 的"申请/释放配对"做成持续观测，
  属于本案例第 2、3 类根因的通用探测器。
