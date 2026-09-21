# vllm 上游 | Prefix Cache / KV 管理类已修复问题合集（源码注释之外）

> 本文收录 **vllm-project/vllm** 中与 KV cache / prefix caching 相关、**已有修复合入**的问题（不含本目录 01-04 已覆盖的 #33775/#32802/#39734/#42082）。
>
> 数据来源：vllm 仓库本地 git 历史实查（提交号均为本地可验证）+ 工作区 `vllm-ascend/0_topic/precision/0_kvcache.md`（158 条 KV cache 精度问题全景清单，状态截至 2026-08-18，下称 **[KVCACHE 清单]**）。
>
> 收录时间：2026-09-21。

| # | Issue | 一句话问题 | 修复 PR（本地 git 实查） | 状态 |
|---|---|---|---|---|
| 1 | [#40707](https://github.com/vllm-project/vllm/issues/40707) | hybrid + 多模态双大图 + chunked prefill → encoder cache 泄漏，调度器 CPU 核长时间空转卡死 | [#31857](https://github.com/vllm-project/vllm/pull/31857)（`a01a1c0d69`，2026-01-10） | 已修复 |
| 2 | —（#43559 的修复线之一） | MTP + prefix caching + hybrid Mamba：Mamba cache 保留部分接受的末 page，后续请求命中"半有效"状态 → 精度崩溃（GSM8K 降 ~20%） | [#47861](https://github.com/vllm-project/vllm/pull/47861) "Fix MTP prefix cache correctness for hybrid Mamba models"（closed 2026-07-07，"[KVCACHE 清单]"） | 已修复 |
| 3 | —（#32802 族续篇） | hybrid + KV connector 下各组 prefix-hit 长度分歧未对齐 | [#48425](https://github.com/vllm-project/vllm/pull/48425)（`229e01e9e1`，2026-07-23） | 已修复 |
| 4 | —（#39734 同域后续） | 部分场景 KV block 分配失败（failure to allocate KV blocks） | [#41282](https://github.com/vllm-project/vllm/pull/41282)（`fa8accac62`，2026-04-29） | 已修复 |
| 5 | —（packed 布局误路由） | uniform-page-size 的 MLA+SWA 模型被误路由进 DeepseekV4 packing → 布局错误 | [#48256](https://github.com/vllm-project/vllm/pull/48256)（`56a357ed33`，2026-07-13） | 已修复 |
| 6 | —（packed 布局校验） | packed KV cache specs 混入不同 dtype 未被检测 → silent 精度损失 | [#49623](https://github.com/vllm-project/vllm/pull/49623)（`275556c35c`，2026-07-23） | 已修复 |

---

## 1. #40707 — Hybrid 多模态 chunked prefill：encoder cache 泄漏死锁

**现象**（issue 报告：vLLM v0.19.0，2×A800，Qwen3.5-35B-A3B hybrid 模型）：

- 单请求携带两张大图（多模态）+ chunked prefill 开启，请求推进缓慢直至停滞；
- 症状是调度器 CPU 侧长时间空转卡死（"stuck in CPU scheduling"），表现为吞吐归零但进程不退出；
- 关掉多模态或关掉 chunked prefill 则不出现。

**根因**：

- Hybrid 模型走统一调度，多模态 encoder 块由 `encoder_cache_manager` 单独管理；
- chunked prefill 多轮推进时，**waiting/preempted 状态请求的 encoder cache 条目未被释放**——请求被抢占/重排后其 encoder cache 泄漏（占用槽位却无人认领）；
- 泄漏积累使 encoder cache 满 → 新 chunk 的 encoder 块无法分配 → 该请求与后续请求整体卡死。

**修复**（PR #31857，`a01a1c0d69`）：

```text
[Bugfix] fix encoder cache leak of waiting requests in scheduler
to solve stuck in CPU scheduling (#31857)
```

调度器在请求回到 waiting（或被抢占）路径上释放其 encoder cache 条目，保证 encoder cache 与请求生命周期一致；关联特性 [#30877](https://github.com/vllm-project/vllm/pull/30877)（`5206e5e28c`，2026-01-24，"[V1][Hybrid] Mamba Prefix Caching with align mode"）同窗口合入，vllm-ascend 的 `model_runner_v1.py:2187` 注释仍引用它。

**关联源码**（本仓库）：`vllm/v1/core/encoder_cache_manager.py`（encoder cache 管理）、`vllm/v1/core/sched/scheduler.py`（抢占/重排路径）。

## 2. MTP + Prefix Caching + hybrid Mamba：部分接受 page 污染（#43559 修复线）

**现象**（[KVCACHE 清单] §3.2，原 issue [#43559](https://github.com/vllm-project/vllm/issues/43559)）：Qwen3.5-35B-A3B（hybrid: attention+Mamba）开 `--enable-prefix-caching` + MTP 投机解码，GSM8K 准确率下降约 20%。

**根因**：

```text
MTP 投机 → 最后一个 page 的 draft token 只有部分被接受 → 该 page 本应整体失效
但 Mamba state cache 仍保留了"最后一个未被完全接受的 page"
→ 后续请求 prefix 命中这些部分有效的 Mamba cache page
→ hidden states 从半接受状态继续传播 → 精度崩溃
```

**修复**：PR [#47861](https://github.com/vllm-project/vllm/pull/47861) "Fix MTP prefix cache correctness for hybrid Mamba models"（closed 2026-07-07）——投机接受后对末 page 做整页失效，禁止部分接受 page 进入可复用集合。同族回归测试由 [#48970](https://github.com/vllm-project/vllm/pull/48970)（e2e hybrid-Mamba prefix-cache corruption 回归）固化。

> 注：#43559 本身在清单中仍标 open（长尾验证中），但其核心正确性缺陷已由 #47861 闭合；相关未竟项见清单。

## 3. #48425 — hybrid + KV connector 的 per-group 命中分歧

PR（`229e01e9e1`，2026-07-23）：

```text
[BugFix] Handle per-group prefix-hit divergence for hybrid models with KV connector (#48425)
```

- **问题**：hybrid 模型各 attention group 的 prefix 命中长度可能不同（full 与 SWA/mamba 组各自匹配），带 KV connector（PD 分离/离屏）时外部块加载长度与各组本地命中长度未对齐，产生错误截断或多余传输；
- **修复**：协调器对各组命中分歧显式处理（与本目录 `02` 文档 `find_longest_cache_hit` 的 `num_uncached_common_prefix_tokens` 机制同一谱系——#32802 修复后的续篇）；
- **关联**：本目录 [02_32802_hybrid_eagle_prefix_cache_zero.md](02_32802_hybrid_eagle_prefix_cache_zero.md) §六 提到的 "#48425 per-group prefix-hit divergence" 即此。

## 4. #41282 — KV block 分配失败（admission cap 同域后续）

PR（`fa8accac62`，2026-04-29）：

```text
[Bugfix] Fix failure to allocate KV blocks error (#41282)
```

对 #40946 引入的 `max_admission_blocks_per_request` 准入钳制域的后续修错（某些合法场景被错误拒绝/报 "failure to allocate KV blocks"）。详见本目录 [03_39734_admission_deadlock_cap.md](03_39734_admission_deadlock_cap.md) §四 修复脉络。

## 5. #48256 — MLA+SWA 均匀页模型误入 DSv4 packing

PR（`56a357ed33`，2026-07-13）：

```text
[Bugfix][KV Cache] Don't route uniform-page-size MLA+SWA models into DeepseekV4 packing (#48256)
```

- **问题**：`_use_packed_kv_cache_config` 判定过宽，把 *均匀页大小* 的 MLA+SWA hybrid 模型也路由进为 DSv4（非均匀页）设计的 packed 分支 → 布局/reshape 错误；
- **修复**：增加 uniform-page-size 排除条件；
- **关联**：与本目录 [04_42082_cross_layers_blocks_packed_layout.md](04_42082_cross_layers_blocks_packed_layout.md) 所述 packed 布局同域——该文档 §三.3 已把"路由边界"列为实验期活跃问题。

## 6. #49623 — packed KV cache specs 混合精度检测

PR（`275556c35c`，2026-07-23）：

```text
[Bugfix] Detect mixed precision in packed KV cache specs (#49623)
```

- **问题**：packed 布局假设各组同 dtype；不同 dtype 的组被 pack 进同一 slab 时不报错但按错误 stride/解释读取 → silent 精度损失；
- **修复**：布局规划期显式检测并拒绝 mixed-precision 组合（快速失败）。

---

## 附：同域仍未修复（跟踪中，供对照）

| Issue | 问题 | 状态（[KVCACHE 清单]，2026-08-18） |
|---|---|---|
| [#49125](https://github.com/vllm-project/vllm/issues/49125) | full-block promotion 后 stale partial prefix-hash 复活 | open（修复 PR #49145 进行中） |
| [#48401](https://github.com/vllm-project/vllm/issues/48401) | hybrid Mamba2 模型 `max_model_len < block_size` 时 prefix cache 静默失效 | open |
| [#48435](https://github.com/vllm-project/vllm/issues/48435) | hybrid-SWA 多会话轮询下命中率坍塌 0（~25% 池占用即触发） | open |
| [#48489](https://github.com/vllm-project/vllm/issues/48489) | 缓释 free 路径丢失 hybrid 配置的 per-group 驱逐顺序 | open |

> 若需逐条跟踪全部 158 条（含 open/closed、严重度、根因分析），见工作区 `vllm-ascend/0_topic/precision/0_kvcache.md`。
