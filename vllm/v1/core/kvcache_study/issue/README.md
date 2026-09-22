# vllm/v1/core 源码中引用的上游 Issue 整理

本目录分两批收录 KV cache 相关问题的整理文档，均给出问题现象、根因分析、解决办法（修复机制 + 代码走读）与修复提交溯源：

- **第一批（01–04）**：`vllm/vllm/v1/core/*.py` 源码注释中显式引用的上游 issue，逐条深析；
- **第二批（05–08）**：扩展收集的 KV cache 相关**已修复**问题合集（vllm 上游 prefix cache / offload / 量化族 + vllm-ascend 精度 / 命中率 / PD 族，含仍在推进中的部分，均已如实标注状态）。

行号均以本仓库当前 main（2026-09 实查）为准。

## 收录范围与索引

| # | Issue | 源码引用位置 | 一句话问题 | 修复 | 本文 |
|---|---|---|---|---|---|
| 1 | [#33775](https://github.com/vllm-project/vllm/issues/33775) | `kv_cache_coordinator.py:219-222`、`single_type_kv_cache_manager.py:299-302` | 多 KV cache group + KV connector 场景下，前组申请外部块时驱逐了后组**尚未 touch 的前缀命中块**（跨组驱逐竞态） | 两阶段分配（PR [#44409](https://github.com/vllm-project/vllm/pull/44409)，取代原 #33775 方案） | [01_33775_two_phase_allocation.md](01_33775_two_phase_allocation.md) |
| 2 | [#32802](https://github.com/vllm-project/vllm/issues/32802) | `kv_cache_coordinator.py:722-724` | hybrid 模型 + EAGLE 投机解码下，收敛循环中 EAGLE drop 被**重复叠加**，prefix cache 命中率螺旋坍塌至 0.0% | `is_simple_hybrid` 单轮退出（PR [#33524](https://github.com/vllm-project/vllm/pull/33524)）+ `eagle_verified` 每候选长度至多 drop 一次（PR [#40860](https://github.com/vllm-project/vllm/pull/40860)） | [02_32802_hybrid_eagle_prefix_cache_zero.md](02_32802_hybrid_eagle_prefix_cache_zero.md) |
| 3 | [#39734](https://github.com/vllm-project/vllm/issues/39734) | `single_type_kv_cache_manager.py:178-191` | 请求 token 数超过 KV cache 容量但小于 max_model_len 时死锁在调度队列头部（head-of-line blocking）；更一般地：**运行时准入估计与启动期池容量估计漂移** → admission 死锁 / mid-prefill OOM | SWA/chunked-local 的每请求准入上限 `max_admission_blocks_per_request`（启动容量估算与运行时准入门共享同一"单一事实来源"） | [03_39734_admission_deadlock_cap.md](03_39734_admission_deadlock_cap.md) |
| 4 | [#42082](https://github.com/vllm-project/vllm/issues/42082) | `kv_cache_utils.py:1301-1302` | 非缺陷引用：`enable_cross_layers_blocks`（跨层 packed KV cache 布局）是**实验性 API**，其形态/默认策略将随 #42082 演进 | 无需修复；文档化其机制、演进史（#30207 → revert #33241 → V2 #33339 → DSv4 默认启用）与已知边界问题 | [04_42082_cross_layers_blocks_packed_layout.md](04_42082_cross_layers_blocks_packed_layout.md) |

> 收录依据：对 `vllm/v1/core/**/*.py` 全量扫描 `issue` / `issues/\d+` 引用（2026-09-21），`*.py` 源码中恰为以上 4 个 issue（#33775 被两处引用）。

## 扩展收录（第二批，2026-09-21 增补）：KV cache 相关已修复问题合集

| # | 范围 | 主题 | 覆盖问题（重点） | 本文 |
|---|---|---|---|---|
| 5 | vllm 上游 | Prefix cache / KV 管理 | #40707 encoder cache 泄漏死锁（修 #31857）、MTP+hybrid Mamba prefix cache 正确性（#47861）、per-group 命中分歧（#48425）、#41282、MLA+SWA 误入 DSv4 packing（#48256）、packed 混合精度检测（#49623） | [05_vllm_prefix_cache_fixed.md](05_vllm_prefix_cache_fixed.md) |
| 6 | vllm 上游 | KV offload / 量化 KV cache | #48596 末块复用竞态、#48911 SWA 可达尾、#48530 packed 溢出、#47574 量化+SWA 新块清零、#47716 fp8_ds_mla reshape、#45363 / #51094 / #49716 | [06_vllm_offload_quant_fixed.md](06_vllm_offload_quant_fixed.md) |
| 7 | vllm-ascend | KV cache 精度类已修复（18 项总表） | #10253 SWA 传输 NaN（修 #10255）、#8540 TP 不等 MTP KV、#10901 ACL Graph+MTP+DP、#11127+#11829 KV Pool 渐进退化、#8845 A5 cache_mode、#10885 Sparse C8 rank 映射、#12183/#11886/#11601/#9500 Mooncake/PD 族等 | [07_ascend_precision_fixed.md](07_ascend_precision_fixed.md) |
| 8 | vllm-ascend | 命中率 0% / PD prefix cache | #7722 P 侧命中后 hang（已修：#7675/#7814/#7796）、#7944 D 侧不生效（部分修：上游 #42524/#44243 bitbridge）、DSv4 命中率 0 族 #10710/#11324/#10970/#8977 + RFC #10517（推进中）、#2304 配置类假阳性 | [08_ascend_prefix_cache_pd.md](08_ascend_prefix_cache_pd.md) |

> 第二批来源：vllm 与 vllm-ascend 两仓库本地 git 历史实查（标注提交号者均可本地验证）+ 工作区 `vllm-ascend/0_topic/precision/0_kvcache.md`（**[KVCACHE 清单]**：158 条 KV cache 精度问题全景，含 open/closed 状态、严重度与根因分析，状态截至 2026-08-18）。

## 与既有文档的关系

- `../kvcache_docs/`、`../kvcache_docs_v2/`：KV cache 子系统的架构走读（`4_kv_cache_coordinator.md`、`5_kv_cache_manager.md` 等从机制角度覆盖了 #33775 的两阶段分配）。
- 工作区 `vllm-ascend/0_topic/precision/`：**KV cache 精度问题全景与深剖**——`0_kvcache.md`（158 条全景清单 + 8 个深剖案例）、`1_pr8540_tp_unequal_mtp_kv.md` … `11_issue7792_mooncake_glm5_503900.md`（单 issue 深剖）及 `cases/` 下 28 个分类案例；第二批文档 07/08 与其互为索引。
- `vllm-ascend/vllm_ascend/patch/__init__.py`：vllm-ascend 对上游 vllm 修复的 bitbridge 补丁索引（如 #42524/#44243 的 D 侧 FA-only prefix cache 复用、#40860 相关 kv_cache_utils patch）。

## 修复机制速览

| Issue | 修复核心 | 所在代码 |
|---|---|---|
| #33775 | Coordinator 先对**所有组** touch 命中块（`ref_cnt++`、移出 free 队列），再对**所有组**做外部块分配（`get_new_blocks` 可能驱逐 LRU） | `kv_cache_coordinator.py:219-236` |
| #32802 | 收敛循环中每个 EAGLE 组对同一候选长度只 drop 一次（`eagle_verified`），长度收缩时清空复验；simple hybrid（1 full + 1 other）单轮退出 | `kv_cache_coordinator.py:718-724, 780-796` |
| #39734 | `max_admission_blocks_per_request` 同为准入钳制与启动池容量估算的上限（SWA = `cdiv(min(sliding_window-1+in_flight, max_len), block_size)+1`；chunked-local 类似） | `single_type_kv_cache_manager.py:178-191`、`kv_cache_interface.py:498-598` |
| #42082 | （实验特性）多组 KV cache 共享一个 packed block slab：同一 block ID 任一时刻只属于一个 group，各组布局可重叠、别名同一物理后备内存 | `kv_cache_utils.py:1262-1340` |
