# vllm-ascend | KV Cache 精度类已修复问题合集

> 本文收录 **vllm-project/vllm-ascend** 中与 KV cache 相关、**已有修复合入**的精度/正确性问题。
>
> 数据来源：vllm-ascend 仓库本地 git 历史实查（标注提交号者均为本地可验证）+ 工作区 `vllm-ascend/0_topic/precision/0_kvcache.md`（[KVCACHE 清单]，158 条全景，状态截至 2026-08-18；深剖案例编号即该文档 §10 的"案例 N"）。单条深剖见 `vllm-ascend/0_topic/precision/` 下同名文档。
>
> 收录时间：2026-09-21。

## 总表（已修复）

| # | Issue/PR | 类别 | 一句话问题 | 修复 | 深剖 |
|---|---|---|---|---|---|
| 1 | [#10253](https://github.com/vllm-project/vllm-ascend/issues/10253) | PD 传输 | SWA KV 传输先 clip 后 trim，选到 dirty 尾块 → D 端全 NaN | [#10255](https://github.com/vllm-project/vllm-ascend/pull/10255)（`9baaf7428`，2026-06-10，"Trim SWA transfer blocks before clipping"） | 清单案例 2；`10_issue10253_swa_stale_nan.md` |
| 2 | [#8540](https://github.com/vllm-project/vllm-ascend/pull/8540) | PD 传输 | P/D 端 TP 不等时 MTP 层 KV head 归属错位 | `c3b1d409a`（#8541 cherry-pick，2026-04-23） | 清单案例 6；`1_pr8540_tp_unequal_mtp_kv.md` |
| 3 | [#10901](https://github.com/vllm-project/vllm-ascend/pull/10901) | ACL Graph | ACL Graph + MTP + DP 三重组合精度错误（Qwen3.5） | `5270d8467`（2026-06-25） | 清单案例 5 |
| 4 | [#11127](https://github.com/vllm-project/vllm-ascend/issues/11127) | KV Pool | KV Pool + MTP 压测 1-2h 渐进退化（接受率→<1%） | 修复链含 [#11829](https://github.com/vllm-project/vllm-ascend/pull/11829)（`46eccd02b`，2026-07-11） | 清单案例 1 |
| 5 | [#11470](https://github.com/vllm-project/vllm-ascend/pull/11470) | C8 布局 | Qwen3.x 多层 KV cache binding 错误 | `2f8934ae3`（2026-07-11） | — |
| 6 | [#8845](https://github.com/vllm-project/vllm-ascend/pull/8845) | A5 特有 | A5 `npu_scatter_pa_kv_cache` 默认 cache_mode 与 ND 布局不兼容 → 静默数据不一致 | "set cache_mode to 'Norm'"（closed 2026-04-30） | 清单案例 7 |
| 7 | [#10885](https://github.com/vllm-project/vllm-ascend/pull/10885) | C8 量化 | Sparse C8 hybrid rank mapping 错误 → 反量化读错位置 | closed 2026-06-24 | 清单案例 4 |
| 8 | [#10756](https://github.com/vllm-project/vllm-ascend/pull/10756) | C8 量化 | Sparse C8 索引不匹配（GLM 模型） | closed 2026-06-18 | — |
| 9 | [#11408](https://github.com/vllm-project/vllm-ascend/pull/11408) | ACL Graph | 310P MTP + ACLGraph 精度问题 | closed 2026-07-03 | — |
| 10 | [#11228](https://github.com/vllm-project/vllm-ascend/pull/11228) | packed 布局 | SFA C8 统一 packed KV cache 布局（A3） | closed 2026-07-01 | — |
| 11 | [#12183](https://github.com/vllm-project/vllm-ascend/pull/12183) | Mooncake | 非连续 PA cache 输入导致传输读取错位 | closed 2026-07-16 | `2_pr12183_non_contiguous_mooncake_pa.md` |
| 12 | [#11886](https://github.com/vllm-project/vllm-ascend/pull/11886) | Mooncake | 传输组未带显式 total KV heads → head 拆分错 | closed 2026-07-12 | `3_pr11886_total_kv_heads_transfer.md` |
| 13 | [#11601](https://github.com/vllm-project/vllm-ascend/pull/11601) | Mooncake | split metadata 未用 cache group ids → 跨组混淆 | closed 2026-07-08 | `4_pr11601_cache_group_ids_metadata.md` |
| 14 | [#9500](https://github.com/vllm-project/vllm-ascend/pull/9500) | PD 传输 | DSV4 PD `kv_cache_tensor.shared_by` 可能为空 → 传输失效 | closed 2026-05-25 | `5_pr9500_shared_by_empty.md` |
| 15 | [#7792](https://github.com/vllm-project/vllm-ascend/issues/7792) | Mooncake | mooncake kv_both+GLM5 传输 503900 报错 | closed 2026-03-28 | `11_issue7792_mooncake_glm5_503900.md` |
| 16 | [#12885](https://github.com/vllm-project/vllm-ascend/pull/12885) | NetLoader | processed-layout P2P 传 INT8_CACHE=no 出错 | closed 2026-07-26 | — |
| 17 | [#13195](https://github.com/vllm-project/vllm-ascend/pull/13195) | 分页注意力 | 恢复 paged attention fallback，修 PD/PCP/DCP 精度回归 | closed 2026-07-30 | — |
| 18 | [#9400](https://github.com/vllm-project/vllm-ascend/issues/9400) / [#10413](https://github.com/vllm-project/vllm-ascend/issues/10413) | 精度 | DSv4-Flash BFCL / GPQA（4P1D 128K）精度低 | closed | — |

---

## 重点深剖

### 1. #10253 — SWA KV 传输顺序错误产生 NaN（清单案例 2）

- **配置**：PD 分离 + SWA（hybrid 模型），并发越高触发概率越大；
- **现象**：部分请求输出全 NaN，逐层传播；
- **根因**：`request_finished_all_groups()` **先执行 SWA tail clip、后执行 prompt trim**（顺序颠倒），且 `_compute_transfer_block_ids()` 对 SWA 组跳过 prompt-trim → SWA tail clip 可能选中未写入/已过期的 dirty 尾块 → dirty block 传到 D 端 → attention 消费 NaN → 全 NaN 传播；
- **修复**（#10255，`9baaf7428`）："Trim SWA transfer blocks before clipping"——先 trim 后 clip，传输块集合一律先做 prompt-trim 再裁剪。

### 2. #8540 — TP 不等时 MTP 层 KV cache 归属错位（清单案例 6）

- **配置**：PD 分离，P 端与 D 端 TP 度数不同（如 P=4、D=8）+ MTP；
- **根因**：TP 度数不同 → KV head 分片方式不同（TP=4 时每 rank 2 个 head，TP=8 时每 rank 1 个）→ 直接传输后 D 端 rank 拿到的 head 集合与自己应持有不匹配 → attention 用错 K/V；
- **修复**（#8540/#8541，`c3b1d409a`）：TP 不等场景对 MTP 层 KV cache 传输前后按两端 TP 差异重分片。

### 3. #11127 + #11829 — KV Pool + MTP 渐进性精度退化（清单案例 1）

- **配置**：GLM5.1 w8a8 + PD 分离 + KV Pool 池化 + MTP；压测 1-2 小时后 MTP 接受率从 ~60-70% 跌至 <1%，且伴随精度问题；重启恢复，复现周期固定；
- **根因**：layerwise KV Pool 的索引计算未正确计入 MTP 每层额外增加的 KV cache → 每次迭代引入微小索引偏移 → 长时间累积后读取错位（渐进性故障，CI 无法覆盖）；
- **修复**（#11829，`46eccd02b`）："Fix layerwise KV pool IndexError when MTP is enabled"——索引计算计入 MTP 附加 KV；配套建议 4h+ soak test。

### 4. #10901 — ACL Graph + MTP + DP 三重组合（清单案例 5）

- **配置**：Qwen3.5 + ACL Graph + MTP + DP，三者**同时**开启才触发（任一关闭即正常）；
- **根因方向**：图优化融合/重排 KV cache 读写算子与 MTP 多步更新、DP 通信时序交互，KV 状态不一致；
- **修复**（`5270d8467`，2026-06-25）：ACL Graph 模式下对 MTP 相关 KV cache 操作特殊处理并保证 DP 边界状态同步。排查方法论（关 graph 对比 eager）已沉淀进 [KVCACHE 清单] §11.5。

### 5. #8845 — A5 `scatter_pa_kv_cache` cache_mode（清单案例 7）

- **硬件**：Ascend 950（A5）独有，A3 正常；
- **现象**：ND KVCache 布局下 prefix cache 命中后输出不稳定、无报错的静默精度问题；
- **根因**：`npu_scatter_pa_kv_cache` 在 A5 上的默认 cache mode 非 'Norm'，L1/L2 写回不及时 → 读到旧值；
- **修复**：ND 布局下显式 set cache_mode='Norm'（closed 2026-04-30）。

### 6. #10885 — Sparse C8 hybrid rank mapping（清单案例 4）

- **配置**：Sparse C8 INT8 KV cache + hybrid 模型（attention + Mamba/SSM 两类 cache 结构）；
- **根因**：rank mapping 假设所有层一致，但 hybrid 中 attention 层与 mamba 层映射方式不同 → 数据排错位置 → 反量化读错；
- **修复**：按层类型分别实现 rank mapping（closed 2026-06-24）。同族：#10756（GLM Sparse C8 索引不匹配）、#11470（Qwen3.x 多层 binding）、#11856（长上下文 C8 scale 溢出，**仍未合入**）。

---

## 附：同域仍未修复代表（跟踪中，供对照）

| Issue | 问题 | 状态 |
|---|---|---|
| [#7707](https://github.com/vllm-project/vllm-ascend/issues/7707) | 100 并发 KVCache chain 断裂（Ds3.2+A3） → 读到不完整 KV | open（深剖 `8_issue7707_chain_breakage_concurrency.md`） |
| [#9111](https://github.com/vllm-project/vllm-ascend/issues/9111) | DSv4-Flash 大 `num_speculative_tokens` 精度损失 | open |
| [#11856](https://github.com/vllm-project/vllm-ascend/pull/11856) | C8 量化长上下文 scale 溢出 | open PR |
| [#14339](https://github.com/vllm-project/vllm-ascend/issues/14339) | 310P Qwen3.5 系列 MTP + prefix cache 精度异常 | open |
| [#12390](https://github.com/vllm-project/vllm-ascend/issues/12390) | AscendStore KV Pool v0.23.0 known issues 汇总 | open（跟踪页） |
| [#10048](https://github.com/vllm-project/vllm-ascend/issues/10048) | D 实例 KV cache 占用率显示负数 | open |
| [#9168](https://github.com/vllm-project/vllm-ascend/issues/9168) / [#11478](https://github.com/vllm-project/vllm-ascend/issues/11478) | AscendStore "Failed to get key" / PP2 producer-put TRANSFER_FAIL | open |
