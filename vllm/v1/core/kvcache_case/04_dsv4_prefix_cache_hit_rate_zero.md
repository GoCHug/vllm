# Case 04 | DSv4 系列：完全相同的串行请求，Prefix Cache 命中率恒为 0

| 项 | 内容 |
|---|---|
| 出处 | vllm-ascend 系列 issue（均收录在 RFC [#10517](https://github.com/vllm-project/vllm-ascend/issues/10517)）：[#10710](https://github.com/vllm-project/vllm-ascend/issues/10710)（命中率 0）、[#11324](https://github.com/vllm-project/vllm-ascend/issues/11324)（block size=32 升级后回归）、[#10970](https://github.com/vllm-project/vllm-ascend/issues/10970)（retention interval 后命中为 0）、[#8977](https://github.com/vllm-project/vllm-ascend/issues/8977)（prefix 长度不足不命中）；上游关联 [vllm#44082](https://github.com/vllm-project/vllm/pull/44082)、[#45845](https://github.com/vllm-project/vllm/pull/45845)、[#43447](https://github.com/vllm-project/vllm/pull/43447)、[#10354](https://github.com/vllm-project/vllm-ascend/issues/10354)、[#11383](https://github.com/vllm-project/vllm-ascend/pull/11383) |
| 类型 | 性能回归 / 命中率为 0（不 crash、输出正确，但缓存失效） |
| 模型 | DeepSeek-V4-Flash（DSA 压缩注意力 + MTP + SWA/MLA 混合层） |
| 难度 | 高（模型权重与多卡门槛）；但"命中率观测基线"部分单卡即可做 |

## 摘要

DSv4 的注意力是异构分组（MLA + 滑窗 + 压缩索引 + MTP lookahead），每种组各自有
KV cache spec，prefix cache 命中要求**所有组的命中率对齐**——
任何一组的 block size / 对齐单元 / mask 规则不一致，整条请求的命中就会被拉平为 0。
vllm-ascend 在 0.21→0.22 升级期间连续出现多个"命中率 0"回归，根因各不相同但都长在
`single_type_kv_cache_manager` 的"组长缓存单元"抽象上，非常适合学习。

## 问题现象（四个子案例）

| 子案例 | 现象 |
|---|---|
| #10710 | DeepSeek-V4-Flash-w8a8-**mtp**：完全相同的串行请求，prefix cache 命中率始终 0% |
| #11324 | 同配置 block size=32：v0.21.0rc1 下前缀重复率 90% 能命中，升级 v0.22.1rc1 后无法命中 |
| #10970 | 开启 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 后，prefix 不仅没被保留优化，直接优化为 0 |
| #8977 | DSv4-Flash：prefix 长度不够长时永远不命中（对齐粒度损耗） |

共通的可观测特征：服务正常、输出正确，只有 `vllm:prefix_cache_hits` 不涨。

## 根因

1. **MTP lookahead 块未进 SWA prefix-cache mask**（#10710 主因）
   MTP 的 lookahead 块占一格 slot，SWA 组做 `find_longest_cache_hit` 时
   以当前 token 视角倒着扣滑窗，lookahead 块被算成"还没算出来的块"，
   每次都把命中长度砍掉 → 命中率 0。
   上游修复：[vllm#44082](https://github.com/vllm-project/vllm/pull/44082)
   "Cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask"；
   vllm-ascend 侧对应 [#11107](https://github.com/vllm-project/vllm-ascend/pull/11107)。

2. **SlidingWindowManager 的 scheduler block size 与 LCM 块大小不对齐**（#11324）
   0.22 改造后 SWA 组的对齐单元 `alignment_tokens != lcm_block_size`，
   命中长度在对齐截断处被清零。
   修复：[#11383](https://github.com/vllm-project/vllm-ascend/pull/11383)
   "Align SlidingWindowManager scheduler block size with LCM block size"。

3. **Compressor block size 硬编码 128**（#8977）
   DSv4 压缩 KV 的块大小曾被固定为 128，prefix 对齐粒度被抬高，
   短前缀永远凑不满一个"可命中单元"。
   背景：[#10354](https://github.com/vllm-project/vllm-ascend/issues/10354)
   把压缩块大小做成 [32, 64, 128] 可配置，缩小对齐要求提升命中率。

4. **Retention interval 生效后的留置策略**（#10970）
   `PREFIX_CACHE_RETENTION_INTERVAL` 与 SWA 层的选择性留置
   （上游 [#43447](https://github.com/vllm-project/vllm/pull/43447)、
   [#45845](https://github.com/vllm-project/vllm/pull/45845)）
   的版本适配出现空档，留置区间内的块未按要求保留。

映射到本地代码：

- 通用机制层：`vllm/v1/core/single_type_kv_cache_manager.py` 的
  `SlidingWindowManager` / `FullAttentionManager` 各自的
  `find_longest_cache_hit`（多组取最小者作为最终命中长度）；
- DSv4 特有：`vllm-ascend/vllm_ascend/core/single_type_kv_cache_manager.py:32`
  `CompressAttentionManager`（DSA 压缩 KV 的专用 manager，
  覆写了 `get_num_blocks_to_allocate` / `allocate_new_computed_blocks` /
  `cache_blocks` / `find_longest_cache_hit`）。

## 如何修复

| 子案例 | 修复 |
|---|---|
| #10710 | vllm#44082 + vllm-ascend#11107：把 MTP lookahead 块纳入 SWA mask 的命中推演 |
| #11324 | vllm-ascend#11383：对齐单元取所有组 block size 的 LCM |
| #8977 | vllm-ascend#10354：压缩块大小可配置 32/64/128 |
| #10970 | 上游 retention interval 语义对齐（vllm#43447/#45845），ascend 侧不需额外适配（RFC 结论） |

## NPU 复现实验

### 第 0 步（单卡即可）：建立命中率观测基线

任何 dense 模型（Qwen2.5-0.5B），正常命中行为：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enforce-eager
# 用同一长 system prompt + 不同问题，串行发 5 个请求，每个 max_tokens=8
bash scripts/observe_metrics.sh http://127.0.0.1:8000 300
```

期望：`vllm:prefix_cache_hits` 在第 2 个请求起快速增长；
日志中第二个请求的 `num_cached_tokens` 约等于共享前缀长度。
**基线都打不出来，先查服务姿态（--enable-prefix-caching 是否被关）、再谈模型侧。**

### 第 1 步（DSv4，多卡）：复现"串行同请求命中 0"

```bash
# 伪命令，按实际集群调整；DSv4-Flash-w8a8 + MTP
vllm serve DeepSeek/DeepSeek-V4-Flash-W8A8 \
  --tensor-parallel-size 8 \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}' \
  --enforce-eager
# 用固定长 prompt 串行发 10 次相同请求，观察 metrics
```

判定：hits/queries 一直为 0，而权重/后端无异常 → 按
"是否 MTP → 是否 SWA 组 → retention 环境变量 → block size" 的排查树
定位到具体子案例。

### 排查树（命中率 0 的标准诊断路径）

```
命中率 0 ──┬─ 加了 MTP/MTP+SWA?        → lookahead 块 mask（vllm#44082）
           ├─ hybrid/SWA 模型?          → LCM 对齐（#11383）
           ├─ 设了 RETENTION_INTERVAL?  → retention 适配（#10970/#45845）
           ├─ DSv4 压缩层?              → compressor block size（#10354/#8977）
           └─ 普通 dense 模型也不命中?  → 服务姿态/配置问题（参考 vllm-ascend#2304）
```

## 延伸阅读

- [vllm-ascend#2304](https://github.com/vllm-project/vllm-ascend/issues/2304)：早期版本的"缓存命中为 0"用户报告（配置类根因）。
- [vllm#43587](https://github.com/vllm-project/vllm/issues/43587)：hybrid + 增量多模态的"0 命中"参考。
- RFC #10517 是 vllm-ascend 侧 prefix cache 命中率问题的总索引，新问题先来这里对号。
