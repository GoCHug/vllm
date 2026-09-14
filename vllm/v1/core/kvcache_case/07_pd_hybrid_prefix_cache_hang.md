# Case 07 | Hybrid 模型 PD 分离：D/P 节点 Prefix Cache 挂死与精度退化

| 项 | 内容 |
|---|---|
| 出处 | [vllm-ascend#7722](https://github.com/vllm-project/vllm-ascend/issues/7722)（P 节点命中 prefix cache 后服务 hang）、[#7944](https://github.com/vllm-project/vllm-ascend/issues/7944)（D 节点 prefix cache 不生效/精度退化）；修复 [#7675](https://github.com/vllm-project/vllm-ascend/pull/7675)、[#7814](https://github.com/vllm-project/vllm-ascend/pull/7814)、[#7796](https://github.com/vllm-project/vllm-ascend/pull/7796)；总纲 RFC [#10517](https://github.com/vllm-project/vllm-ascend/issues/10517)；上游方案 [vllm#42524](https://github.com/vllm-project/vllm/pull/42524)、[#44243](https://github.com/vllm-project/vllm/pull/44243) |
| 类型 | 分布式正确性 + 可用性（hang） |
| 模型 | Qwen3.5/3.6-Next（hybrid：FullAttention + GDN/Mamba 线性注意力） |
| 难度 | 高（1P1D 多卡 + MooncakeLayerwiseConnector 环境） |

## 摘要

PD 分离后，P（prefill）节点与 D（decode）节点各自维护 KV cache 管理器，
跨节点靠 KV connector 传输层同步。hybrid 模型的 KV 被分成多个"组"
（FA 组可复用、Mamba/GDN 组是流式状态），**两组对"命中"的语义并不一致**：
D 侧把 Mamba 组也算进命中账目并对远端 KV 做复用假设时，
与 P 侧实际传输/状态不对齐，轻则 hang（等待一个永远不会来的块），
重则精度退化（错把不能复用的流式状态当已算好来用）。

## 问题现象

issue #7722 记录（vLLM v0.17.0rc1 + vllm-ascend，Qwen3.5-35B-A3B）：

- 部署形态：1P1D，TP4，`MooncakeLayerwiseConnector`（P 侧 kv_producer）；
- 关键参数：`--enable-prefix-caching`、`--mamba-cache-mode align`、
  `--no-disable-hybrid-kv-cache-manager`、`--additional-config '{"recompute_scheduler_enable": true}'`；
- 现象：P 节点命中 prefix cache 后服务挂死不再返回（无响应、无崩溃栈）；
- 后续：#7675 解除 hang，但命中场景下精度退化（#7944），最终由 #7814/#7796 修复。

## 根因

1. D 节点的 KV 命中率**记账**把 Mamba 组包含进去了（ RFC #10517 的表述：
   "exclude the Mamba group from D-side KV cache hit rate accounting under specific conditions"），
   于是调度依据的 hit 长度大于实际可复用长度，消费端等待/复用错位的块；
2. `mamba-cache-mode align` 下 Mamba 状态只在块边界整体成立，
   跨节点只传了 FA 组块，D 侧当成"整条都算好了"；
3. 解 hang 后的精度问题同根：部分组被错误地视为已缓存。

一句话：**多组 KV spec 上，"最长可复用前缀 = min(各组命中)"+ 传输契约，
任何一组没对齐，账目和事实就分家了**（呼应案例 04 的命中率被拉平原理，
那是"没 crash 的形态"，这里是"hang 的形态"）。

## 如何修复

- 短期（vllm-ascend 平台补丁）：`--additional-config '{"recompute_scheduler_enable": true}'`
  使用 RecomputeScheduler，让 D 侧不做无效记账而是重算 Mamba 组
  ——本地代码即 `vllm-ascend/vllm_ascend/core/recompute_scheduler.py:54`
  `RecomputeSchedulerConfig` / `:94` `RecomputeScheduler`，
  配置开关在 `vllm_ascend/ascend_config.py:875`。
  RFC 约定：上游合入 vllm#42524/#44243（KV consumer partial-group caching）
  或 0.23.1rc0 之后移除补丁。
- 长期（上游）：D 侧按组部分缓存（partial-group caching）+ Mamba 组的
  PD 分离命中规则（vllm#44243 "Fix Mamba prefix cache hit rate in PD disaggregation"）。

## NPU 复现实验（进阶）

环境：两台卡（或同机两组卡），每侧 TP4，走 MooncakeLayerwiseConnector。

P 节点（复刻 issue 脚本要点）：

```bash
export HCCL_BUFFSIZE=256
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
vllm serve /path/Qwen3.5-35B-A3B \
  --tensor-parallel-size 4 --enforce-eager \
  --max-model-len 40960 --max-num-batched-tokens 16384 \
  --enable-prefix-caching --mamba-cache-mode align \
  --no-disable-hybrid-kv-cache-manager \
  --additional-config '{"recompute_scheduler_enable": true}' \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector","kv_role":"kv_producer","kv_port":"36010", ...}'
```

D 节点：同构命令，`kv_role` 换 `kv_consumer`，加 `--headless`。

### 观测与判定

| 步骤 | 期望 |
|---|---|
| 连发两条共享长前缀的请求 | 第二条命中 prefix cache 时出现挂死（历史上）→ 记录 hang 点日志 |
| 打开 recompute_scheduler（对照） | 不挂死（缓解生效） |
| 关闭 P 侧 prefix caching（对照） | 不挂死（锁定问题在命中路径） |

观测点：P/D 两侧日志中 KV connector 的块传输记录、
`vllm:num_requests_running`、超时请求的 TTFT；`npu-smi info` 确认算力形态。

## 关联源码

| 位置 | 说明 |
|---|---|
| `vllm_ascend/core/recompute_scheduler.py` | RecomputeScheduler（D 侧重算策略） |
| `vllm_ascend/ascend_config.py:875` | `recompute_scheduler_enable` 开关 |
| `vllm/v1/core/kv_cache_coordinator.py` | 多组 KV spec 的命中聚合（组间取 min 的通用逻辑） |
| `vllm/v1/core/kv_cache_manager.py:297` | `get_computed_blocks_for_connector`（连接器视角的命中块） |

## 延伸阅读

- RFC #10517 中 "Qwen 3.5/3.6/3.7 Partial hit (FA)" 小节完整记录了
  本案例的修复时间线与新模型（DSv4）同构问题的后续。
- 与案例 03 的共同教训：hybrid 模型上所有"以块为最小单位"的机制
  （对齐切分、组对齐、PD 传输）都必须显式处理 Mamba 组的流式状态语义。
