# vllm-ascend | Prefix Cache 命中率 0% 与 PD 分离 Prefix Cache 问题合集

> 本文收录 **vllm-project/vllm-ascend** 中 prefix cache 命中率坍塌、PD 分离下 prefix cache 生效与否的问题族。**注意：该族状态混合**——一部分已完整修复（有本地 git 实证），一部分修复推进中（如实标注 open）。
>
> 数据来源：vllm-ascend 仓库本地 git 历史实查 + `vllm_ascend/patch/__init__.py` bitbridge 索引 + 工作区 `vllm-ascend/0_topic/precision/0_kvcache.md`（[KVCACHE 清单]，状态截至 2026-08-18）+ ascend 官方文档（`docs/source/tutorials/models/Qwen3.5-397B-A17B.md`）。
>
> 收录时间：2026-09-21。

| # | Issue | 一句话问题 | 修复 | 状态 |
|---|---|---|---|---|
| 1 | [ascend#7722](https://github.com/vllm-project/vllm-ascend/issues/7722) | PD 分离下 P 节点命中 prefix cache 后服务 hang | [#7675](https://github.com/vllm-project/vllm-ascend/pull/7675)、[#7814](https://github.com/vllm-project/vllm-ascend/pull/7814)（`988c2aa5c`，2026-03-31）、[#7796](https://github.com/vllm-project/vllm-ascend/pull/7796)（`ab928ed58`，2026-03-31） | **已修复**（本地 git 实证） |
| 2 | [ascend#7944](https://github.com/vllm-project/vllm-ascend/issues/7944) | PD 分离 D 节点 prefix cache 不生效 / 精度退化 | 上游方案 vllm PR [#42524](https://github.com/vllm-project/vllm/pull/42524)、[#44243](https://github.com/vllm-project/vllm/pull/44243)，ascend 以 bitbridge 补丁落地 | **部分修复**（Qwen3.5-397B 官方文档仍标 known issue） |
| 3 | [ascend#10710](https://github.com/vllm-project/vllm-ascend/issues/10710)、[#11324](https://github.com/vllm-project/vllm-ascend/issues/11324)、[#10970](https://github.com/vllm-project/vllm-ascend/issues/10970)、[#8977](https://github.com/vllm-project/vllm-ascend/issues/8977)（总纲 RFC [#10517](https://github.com/vllm-project/vllm-ascend/issues/10517)） | DSv4 系列 prefix cache 命中率 0% 族 | 上游修复链 vllm PR [#40860](https://github.com/vllm-project/vllm/pull/40860)/[#44082](https://github.com/vllm-project/vllm/pull/44082)/[#45845](https://github.com/vllm-project/vllm/pull/45845)/[#43447](https://github.com/vllm-project/vllm/pull/43447)，ascend PR [#11383](https://github.com/vllm-project/vllm-ascend/pull/11383) | **修复推进中**（#10710/#10517 在 [KVCACHE 清单] 中仍标 open，2026-08-18） |
| 4 | [ascend#2304](https://github.com/vllm-project/vllm-ascend/issues/2304) | 早期"缓存命中为 0"用户报告 | 配置类根因（正确开启 prefix caching 即恢复） | 已澄清（非代码缺陷） |
| 5 | [ascend#10569](https://github.com/vllm-project/vllm-ascend/issues/10569) | DSv4 Pro PD 分离 D 节点 `mooncake_hybrid_connector` KV 传输失败 | 跟踪中 | open |

---

## 1. #7722 — PD 分离 P 侧命中 prefix cache 后服务 hang（已修复）

**现象**：PD 分离 + hybrid 模型（如 Qwen3.5-A3B），请求在 **P 节点**命中 prefix cache 后，服务 hang（不再推进生成）。

**根因方向**：Layerwise/PD connector 不支持 Mamba（hybrid 状态）部分的 prefill prefix caching——P 侧命中后交给 D 侧的元数据与传输块集合对 Mamba 组不闭合，D 侧等待永远不会到达的数据。

**修复**（git 实证，两条链同一特性分别进 main 与 v0.18.0 分支）：

```text
988c2aa5c 2026-03-31 [P/D][Feature] Layerwise connector supports Mamba prefill prefix caching (#7814)
ab928ed58 2026-03-31 [v0.18.0][P/D][Feature] Layerwise connector supports Mamba prefill prefix caching (#7796)
```

Layerwise connector 增加对 Mamba prefill 状态的 prefix caching 支持，P→D 传输对状态部分闭合。另一修复 #7675 同窗口合入（P 侧命中路径的 hang 修复）。

## 2. #7944 — PD 分离 D 节点 prefix cache 不生效 / 精度退化（部分修复）

**现象**：PD 分离下 **D 节点**的 prefix cache 不生效（`num_cached_tokens` 恒 0 或命中后精度退化），单机正常。

**上游方案机制**（ascend 以 bitbridge 补丁落地，`vllm_ascend/patch/__init__.py:133-147` 原文摘录）：

```text
# FullAttention-only prefix cache reuse on the D side.
# How:
#    For Mamba hybrid models, num_new_local_computed_tokens should be the FA hit
#    length. This value is passed to the connector's get_num_new_matched_tokens
#    which computes: external = total - local_computed.
#    Using the FA hit skips re-transferring FA blocks already cached on D-side.
# Related PR:
#    https://github.com/vllm-project/vllm/pull/42524
#    https://github.com/vllm-project/vllm/pull/44243
```

即：Mamba hybrid 模型在 D 侧只对 **FullAttention 组**做 prefix cache 复用——`num_new_local_computed_tokens` 取 FA 命中长度，connector 的 `external = total - local_computed` 便只传输 D 侧尚未缓存的块，跳过 D 侧已有的 FA 块重传。

**状态说明**：该方案解决了"重复传输/不生效"主体问题；但 ascend 官方文档（`docs/source/tutorials/models/Qwen3.5-397B-A17B.md:552/730`）仍写明：

> "`--no-enable-prefix-caching` disables prefix caching. For PD disaggregation, the D-node prefix-cache known issue is tracked in #7944."

即 PD 分离下 D 节点 prefix cache 仍属**已知限制**，特定场景建议关闭前缀缓存。

## 3. DSv4 命中率 0% 族（#10710 / #11324 / #10970 / #8977，RFC #10517）

**现象**（总纲 RFC #10517 "Improve Prefix-Caching Hit Rate for Hybrid Models" 收录的四类）：

| Issue | 形态 |
|---|---|
| #10710 | DSv4-Flash-w8a8-mtp：**完全相同的串行请求** prefix cache 命中率始终 0% |
| #11324 | block size 升级到 32 后命中率回归为 0 |
| #10970 | retention（稀疏保留）interval 之后命中归 0 |
| #8977 | prefix 长度不足时不命中 |

**根因方向**（与 vllm 上游 #32802 论述的机制同族，见本目录 [02_32802_hybrid_eagle_prefix_cache_zero.md](02_32802_hybrid_eagle_prefix_cache_zero.md)）：

- DSv4 是**复杂 hybrid**（多个 KV cache group，full + SWA/retention）：上游 #33524 只 special-case 了 simple hybrid（1 Full + 1 SWA），复杂 hybrid 的收敛循环/EAGLE 处理在 #40860（`eagle_verified`）之前仍有缺陷；
- 驱逐器可能破坏"单存条目"（retention 组只存一份的条目被逐出后不可重建）；
- 调度块大小被 LCM 拉大后，短 prefix 凑不满一个调度块（#8977/block size=32 形态）。

**修复链**：

- 上游：vllm PR #40860（DeepSeek V4 Rebased，`4d51588e23`，引入 `eagle_verified`，本仓库 `kv_cache_coordinator.py:725` 即其代码）、#44082（SWA lookahead mask）、#45845、#43447；
- ascend：PR #11383 等；并以 `platform/patch_kv_cache_utils.py` bitbridge 适配 #40860 对 hybrid KV cache groups 的限制（`patch/__init__.py:149-154`，"** 7. File: platform/patch_kv_cache_utils.py** ... vLLM PR #40860 added a restriction that hybrid KV cache groups with ..."）。

**状态**：[KVCACHE 清单]（2026-08-18）中 #10710 与 RFC #10517 仍标 **open**——上游机制已就位、逐步消化中，但 ascend 侧端到端命中率目标尚未完全达成。引用时请以 issue 当前状态为准。

## 4. #2304 — 早期"命中为 0"报告（配置类澄清）

早期用户报告 prefix cache 命中为 0 的经典根因是**配置**而非代码：vllm-ascend 各版本默认未必开启 `--enable-prefix-caching`（且老版本需显式传参）、TP/多机部署下指标口径不同等。属于排障时优先排除的假阳性。

## 5. #10569 — DSv4 Pro PD 分离 mooncake 传输失败（open）

`mooncake_hybrid_connector` 在 PD 分离 D 节点报 KV cache 传输失败（回退单机正常）。深剖：工作区 `vllm-ascend/0_topic/precision/9_issue10569_mooncake_hybrid_connector_fail.md`。尚未修复。

---

## 与本目录其它文档的关系

- 命中率坍塌机制的"上游版本"（#32802 + #40860 修复线）：见 [02_32802_hybrid_eagle_prefix_cache_zero.md](02_32802_hybrid_eagle_prefix_cache_zero.md)；
- packed 布局（DSv4 默认启用，与 #11324 的 block size 语义相关）：见 [04_42082_cross_layers_blocks_packed_layout.md](04_42082_cross_layers_blocks_packed_layout.md)；
- 上游 prefix cache 其余已修复问题：见 [05_vllm_prefix_cache_fixed.md](05_vllm_prefix_cache_fixed.md)。
