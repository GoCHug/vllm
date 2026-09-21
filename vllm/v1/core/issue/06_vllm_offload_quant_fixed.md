# vllm 上游 | KV Offload / 量化 KV Cache 类已修复问题合集

> 本文收录 **vllm-project/vllm** 中 KV cache 卸载（OffloadingConnector）、量化 KV cache（fp8/int8_per_token_head 等）相关**已有修复合入**的问题。
>
> 数据来源：vllm 仓库本地 git 历史实查（提交号均为本地可验证）+ 工作区 `vllm-ascend/0_topic/precision/0_kvcache.md`（下称 **[KVCACHE 清单]**，状态截至 2026-08-18）。
>
> 收录时间：2026-09-21。

| # | 修复 PR（本地 git 实查） | 一句话问题 | 根因 | 状态 |
|---|---|---|---|---|
| 1 | [#48596](https://github.com/vllm-project/vllm/pull/48596)（`f38f3d11fb`，2026-07-17） | 请求结束 offload 末块与调度器复用该块竞态 → 数据损坏 | offload 未完成前 block 已被回收复用 | 已修复 |
| 2 | [#48911](https://github.com/vllm-project/vllm/pull/48911)（`fbfe58133d`，2026-07-20） | hybrid SWA 组 offload 破坏"可达尾部"块 | 卸载策略不识别窗口内仍可达的尾块 | 已修复 |
| 3 | [#48530](https://github.com/vllm-project/vllm/pull/48530)（`12f2c515a7`，2026-07-16） | packed 非均匀页 KV cache 的 offload `set_` 溢出 | packed offset 计算按均匀页假设 | 已修复 |
| 4 | [#47574](https://github.com/vllm-project/vllm/pull/47574)（`8ce53a616e`，2026-07-20） | 量化 + SWA hybrid：新分配块未清零 → 未初始化数据被算子消费 | 量化路径假设块内容已初始化 | 已修复 |
| 5 | [#47716](https://github.com/vllm-project/vllm/pull/47716)（`04adc8843b`，2026-07-07） | DSV4 `fp8_ds_mla` 启动即崩（reshap­e 按 head_size vs 656B/token 页） | MLA 页大小与 head_size 量纲混用 | 已修复（修 [#48378](https://github.com/vllm-project/vllm/issues/48378)） |
| 6 | [#45363](https://github.com/vllm-project/vllm/pull/45363)（closed 2026-06-12） | `CuMemAllocator.sleep()` unmap GPU VA 时与 in-flight offload 传输竞态 | sleep 未排空在途传输 | 已修复 |
| 7 | [#51094](https://github.com/vllm-project/vllm/issues/51094)（closed 2026-08-05） | OffloadingConnector 在 chunk 边界静默输出错误（`mamba_cache_mode=all`） | Mamba 状态按 chunk 边界切分语义错误 | 已修复 |
| 8 | [#49716](https://github.com/vllm-project/vllm/issues/49716)（closed 2026-07-24） | `int8_per_token_head` KV cache 高负载下损坏 Gemma-4 (hybrid) 输出（Triton） | 量化 KV 布局/索引在 hybrid 高并发下错位 | 已修复 |

---

## 1. #48596 — 请求结束 offload 末块的复用竞态

**问题**：请求结束时最后一个 block 需要被 offload；offload 完成前，调度器可能已回收该 block 并分配给新请求——新请求写入与 offload 读取发生竞争 → 数据损坏。

**修复**（`f38f3d11fb`）："Offload last block at request finish and prevent reuse race"——末块的 offload 完成后才允许复用（复用闸门随 offload 完成回调打开）。

## 2. #48911 — hybrid SWA 组的可达尾部保护

**问题**：hybrid 模型 SWA 组中，部分尾部块仍在滑动窗口"可达"范围内；offload 装载/驱逐路径未识别 reachable tails，把这些窗口内仍要用的块当作可驱逐/可覆盖对象 → attention 读到缺口。

**修复**（`fbfe58133d`）："Preserve reachable tails for hybrid SWA groups"——offload 决策以"窗口可达性"为准绳，保留可达尾块。

## 3. #48530 — packed 非均匀页 offload `set_` 溢出

**问题**：packed 布局（见本目录 [04_42082_cross_layers_blocks_packed_layout.md](04_42082_cross_layers_blocks_packed_layout.md)）叠加非均匀 page size（各层每 token 字节不同）时，offload 的 `set_` 操作按均匀页假设计算 offset → 越界写入/读错位置 → 数据损坏。

**修复**（`12f2c515a7`）：offload 的偏移计算改为按 per-layer 页大小精确累积。同族后续见 PR [#48878](https://github.com/vllm-project/vllm/pull/48878)（`blocks_per_chunk` 概念，针对 DSv4-Flash/Gemma-4 混合组离屏分块）。

## 4. #47574 — 量化 + SWA hybrid 的新块清零

**问题**：量化（quantized）+ 滑动窗口 + hybrid 三重组合下，新分配 KV block 未被清零，块内残留随机数据被量化算子当作有效数据消费 → 精度异常（silent）。

**修复**（`8ce53a616e`）：新分配 block 显式清零。

**关联本仓库源码**：本仓库 `single_type_kv_cache_manager.py:84-89` 的 `_record_new_block_ids`（"Record newly allocated block ids only when worker-side zeroing will consume them..."）与 `:327-328`、`:367-368` 的 `new_block_ids` 通路正是该修复的落地——manager 记录新块 ID，worker 侧统一 zeroing。

## 5. #47716 / #48378 — DSV4 `fp8_ds_mla` reshape 崩溃

**问题**（issue #48378，closed 2026-07-12）：DeepSeek-V3.2 / GLM DSA + `--kv-cache-dtype fp8_ds_mla` 引擎初始化即崩——KV cache reshape 以 `head_size`（576）计算视图，而该格式实际页 656 B/token，量纲不一致。

**修复**（`04adc8843b`）：reshape 改用 fp8_ds_mla 的真实页大小。MLA 系同族已修复问题（closed，[KVCACHE 清单] §1.1）：#47935（DP/EP + FlashMLA fp8）、#47905（Nemotron+FlashInfer fp8 启动断言）、#48439（MLA fp8 startup crash）、#49435（SM100 fp8_ds_mla scales，open PR）。

## 6. #45363 — `CuMemAllocator.sleep()` 与在途 offload 竞态

**问题**：`CuMemAllocator.sleep()` 会 unmap GPU 虚拟地址；若此刻仍有 in-flight KV offload 传输访问这些 VA → 崩溃/损坏。

**修复**：sleep 前先排空（drain）所有 in-flight KV offload 传输。

## 7. #51094 — OffloadingConnector chunk 边界 Mamba 状态错误

**问题**（closed 2026-08-05）：mamba_cache_mode=all 时，OffloadingConnector 在 chunk 边界处静默返回错误输出——Mamba 状态的 chunk 切分未按状态边界对齐。

**深剖**：工作区 `vllm-ascend/0_topic/precision/cases/1_单机单卡/09_issue51094_mamba_offload_chunk.md`。

## 8. #49716 — `int8_per_token_head` 损坏 Gemma-4 hybrid 输出

**问题**（closed 2026-07-24，🔴 极高）：Triton 后端高负载下，per-token-head 粒度量化 KV cache 损坏 Gemma-4（hybrid）输出。

**深剖**：工作区 `vllm-ascend/0_topic/precision/cases/1_单机单卡/12_issue49716_int8_kv_layout_refactor.md`（含 int8 KV 布局重构分析）。

---

## 附：同域仍未修复高危（跟踪中，供对照）

| Issue | 问题 | 状态（[KVCACHE 清单]，2026-08-18） |
|---|---|---|
| [#48412](https://github.com/vllm-project/vllm/issues/48412) | OffloadingConnector 跨层分配缺 per-token-head scale packing → 输出完全腐败（silent corruption，🔴 极高） | open |
| [#44238](https://github.com/vllm-project/vllm/issues/44238) | MooncakeConnector 并发 PD 传输竞态（batch_transfer_sync_write race）→ 尾部全零/头部损坏 | open（深剖：`vllm-ascend/0_topic/precision/7_issue44238_mooncake_batch_transfer_race.md`） |
| [#49176](https://github.com/vllm-project/vllm/issues/49176) | 二级 tier load 失败致 livelock（async lookup cache 未失效） | open |
| [#49261](https://github.com/vllm-project/vllm/issues/49261) | 持久化 KV offload cache 未按 model revision 隔离 | open |
| [#51313](https://github.com/vllm-project/vllm/issues/51313) | Kimi-K3 `--kv-cache-dtype fp8` 非 Blackwell 上不可用 | open |
