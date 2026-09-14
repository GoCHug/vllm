# Case 02 | Cache Hit 与 Miss 数值路径分叉：温度 0 下的输出漂移

| 项 | 内容 |
|---|---|
| 出处 | [vllm issue #33123](https://github.com/vllm-project/vllm/issues/33123)（2026 年报告，AMD MI355X/gfx950 + Qwen3-0.6B，分析翔实）；NPU 版实验为**自设计移植**，并给出可复用脚本 |
| 类型 | 数值一致性（不是调度器 bug，但直接决定 prefix caching 的可用性） |
| 难度 | 低（单卡小模型，脚本自动化） |

## 摘要

Prefix caching 的前提是"复用缓存块 = 数学上重新计算"。
但全量 prefill 和"缓存前缀 + 只算新 token 的 paged prefill"是**两条不同的内核路径**，
在部分硬件/内核组合下浮点结果不同，bf16 下差异足以让 argmax 在某个 token 翻转，
温度 0 输出从此完全分叉。这类问题在 v1 切换内核后端（每层的 kernel 选择、tile 大小、
累加顺序）时最容易冒出来。

## 问题现象（issue #33123 原始记录）

MI355X 上，同一 prompt、temperature=0，串行发 5 次：

| 次数 | 缓存状态 | 输出 |
|---|---|---|
| Run 1 | miss，全量 prefill | `The European Union consists of 27 member states` |
| Run 2 ~ 5 | hit，partial prefill + cached KV | `The European Union (EU) consists of **2` |

分叉点在第 3 个输出 token（`consists` vs `(`）。
同一版本在 MI325X 上两条路径数值一致，所有 run 输出相同；
关闭 prefix caching 后所有 run 也一致。这证明：
调度与缓存本身没有错（KV 内容正确），错的是**两条计算路径不等价**。

## 根因

1. Run 1（miss）：注意力内核对整段序列一次算完（无分页、无前缀拼接）；
2. Run 2+（hit）：q 只含新 token，K/V 从 paged KV cache 里按 block table 拼前缀。

两条路径的 kernel 实现（tile、mask、softmax 累加顺序、MLA/GQA 展开方式）不同，
浮点上只是"几乎相等"。绝大多数 token 上 argmax 稳定，
但接近打平的位置会被 `1e-3` 量级差异翻转，之后自回归地把分歧放大成整段不同的输出。

本质：**prefix caching 保证数学等价，不保证浮点等价**。
所以它属于"平台/内核层"问题 —— 换后端、换硬件、换精度都可能复现或消失，
这正是它能移植到 NPU 上做实验的原因。

## 如何修复

没有调度器侧的"一行修复"，方向有三类：

1. 内核对齐：让 hit/miss 两条路径走同一实现（或用数值上更稳定的算子/更高精度累加）；
   vLLM 部分后端曾以"全部走 paged 路径"统一过。
2. 业务侧接受：对确定性敏感的场景（RL rollout、评测）关闭 prefix caching。
3. 平台侧确定性：vllm-ascend 面向 RL 训推一致性提供了 batch invariant 支持
   （本地 `vllm-ascend/vllm_ascend/batch_invariant.py`，0.18.0 起），思路就是
   "同一语义的多种计算路径强制产出位级一致结果"，与本案例的诉求同源。

## NPU 复现实验（自设计移植）

目标不是"复现一次算完"，而是**建立一条量化管线**：
在 NPU 上测 hit/miss 两条路径的输出一致性，并沉淀为后端升级的回归用例。

服务端（单卡即可）：

```bash
MODEL=Qwen/Qwen3-0.6B   # 或 Qwen2.5-0.5B-Instruct
vllm serve $MODEL \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager   # 关 ACL Graph，先排除图捕获引入的变量
```

客户端：

```bash
python scripts/client_hit_vs_miss_divergence.py \
  --base-url http://127.0.0.1:8000 --model $MODEL --runs 5 --max-tokens 96
```

脚本：同一 prompt（temperature=0，`logprobs` 拉回 top-5）串行 N 次，
逐位置对比第 1 次（miss）与后续（hit）的 top-1 token 与 logprob 差值，
输出：首个 token 分叉位置、logprob 最大偏差、是否发生输出分叉。

### 观测与判定

| 现象 | 结论 |
|---|---|
| 所有 run 输出一致，max logprob diff < 1e-2 | NPU 内核两路径一致（实验"通过"，也是有价值的负结果） |
| 输出在固定位置分叉，且关闭 prefix caching 后一致 | 复现（等同 #33123 的 NPU 版） |
| 多次相同 run 之间输出也不一致 | 先修服务级随机性问题（ACL Graph/采样器），再谈本案例 |

注意控制变量：`--enforce-eager` 固定、temperature=0、seed 固定、
请求串行（避免 batch 变化引入 batch size 相关数值差异）。

## 关联源码（本地 main）

| 位置 | 说明 |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | `schedule()` 计算命中前缀 → 生成 partial prefill 的输入 |
| `vllm-ascend/vllm_ascend/attention/` | NPU 侧 prefill / paged prefill 内核选择逻辑 |
| `vllm-ascend/vllm_ascend/batch_invariant.py` | 批次不变形状结果的平台方案（RL 场景） |

## 延伸阅读

- #33123 里作者用 HF（`use_cache=False` vs `use_cache=True`）做了第三方对照组，
  该方法可直接搬到 NPU 复现实验中（HF on CPU/NPU 对照）。
- 与之相关的方向：`vllm_ascend` 的 RL/评测流程里常配
  `--no-enable-prefix-caching + --enforce-eager` 以获得可复现 rollout。
