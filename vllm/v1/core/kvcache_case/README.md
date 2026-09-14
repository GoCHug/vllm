# KV Cache 问题案例库（kvcache_case）

本目录是 `v1/core` KV cache 管理机制的**配套问题案例库**：把 `../kvcache_docs/` 里的机制笔记
（block_pool → single_type_kv_cache_manager → kv_cache_coordinator → kv_cache_manager）
落到"真实 bug + 可在 NPU 上复现的实验"上。

每个案例统一按以下结构组织（也是收集新案例的模板）：

```
1. 出处        —— issue/PR/设计文档/自设计，附链接与版本
2. 适用版本与环境
3. 问题现象     —— 用户可见的日志/指标/输出
4. 根因        —— 精确到本仓库源码位置（file:line 以本地 main 为准）
5. 如何修复     —— 对应 PR 的修复思路，以及本地代码现状验证
6. NPU 复现实验 —— 服务端启动命令 + 客户端脚本 + 负载构造
7. 观测与判定   —— 指标名、日志关键字、pass/fail 标准
8. 关联源码
9. 延伸阅读
```

## 案例索引

| 编号 | 主题 | 类型 | 出处 | 难度（资源） | 配套脚本 |
|---|---|---|---|---|---|
| 01 | 多模态占位 token 哈希碰撞 → 回答张冠李戴 | 正确性 / hash 冲突 | [vllm#20261](https://github.com/vllm-project/vllm/issues/20261) | 低（单卡 VL 模型） | `scripts/repro_mm_hash_collision_demo.py`、`scripts/client_mm_hash_two_images.py` |
| 02 | cache hit 与 miss 数值路径分叉 → 温度 0 下输出漂移 | 数值一致性 | [vllm#33123](https://github.com/vllm-project/vllm/issues/33123)（NPU 版为自设计移植） | 低（单卡小模型） | `scripts/client_hit_vs_miss_divergence.py` |
| 03 | hybrid 模型 + 双大图 + chunked prefill → encoder cache 死锁 | 调度死锁 | [vllm#40707](https://github.com/vllm-project/vllm/issues/40707) | 中高（hybrid 模型 35B） | `scripts/client_two_images_deadlock.py` |
| 04 | DSv4 系列 prefix cache 命中率 0% | 命中率回归 | [vllm-ascend#10710](https://github.com/vllm-project/vllm-ascend/issues/10710)、[#11324](https://github.com/vllm-project/vllm-ascend/issues/11324)、[#10970](https://github.com/vllm-project/vllm-ascend/issues/10970)、RFC [#10517](https://github.com/vllm-project/vllm-ascend/issues/10517) | 高（DSv4 权重） | `scripts/observe_metrics.sh` |
| 05 | abort 风暴下的泄漏不变量检测 | 健壮性 / 泄漏（自设计） | 自设计（机制背景 vllm#31857 等） | 低-中（单卡） | `scripts/client_abort_storm.py` |
| 06 | cache_salt 多租户隔离验证 | 特性验证（自设计） | 官方设计文档 + 自设计 | 低（单卡） | `scripts/client_cache_salt_test.py` |
| 07 | hybrid 模型 PD 分离下 prefix cache 挂死 | 分布式 / PD 分离 | [vllm-ascend#7722](https://github.com/vllm-project/vllm-ascend/issues/7722)、[#7944](https://github.com/vllm-project/vllm-ascend/issues/7944) | 高（1P1D 多卡） | 手工步骤见文档 |

建议顺序：01/02/06（门槛最低）→ 05（压力观察能力）→ 03（调度交互）→ 04/07（大型/分布式）。

## 通用观测方法

### 1. /metrics（Prometheus，首选）

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'prefix_cache|gpu_cache_usage'
```

关键指标（定义见 `../metrics/loggers.py:584` 附近）：

| 指标 | 含义 |
|---|---|
| `vllm:prefix_cache_queries` | 发起的 prefix cache 查询次数（累计 token 数） |
| `vllm:prefix_cache_hits` | 命中的 token 数；命中率 = hits / queries |
| `vllm:gpu_cache_usage_perc` | KV cache 使用率（0~1） |
| `vllm:num_requests_running` / `vllm:num_requests_waiting` | 排队状态，死锁类案例的关键观测点 |

### 2. 服务端日志

- 每 10s 的 stat 日志中带 prefix cache 命中率与 `Preempted` 数：
  `Running: x reqs, Waiting: y reqs ... Preempted: z reqs`
- 内存不足告警：`Enough free tokens ... but less than watermark ... preemption`
- 事件日志（需要开 KV events）。

### 3. 引擎内部事件（离线模式）

`LLM.get_kv_cache_events()` 可拿到 `BlockStored` / `BlockRemoved` / `AllBlocksCleared`
等事件（实现分发在 `kv_cache_manager.py` 的 `take_events`，`block_pool.py:373` 的
`emit_cached_block_events` / `block_pool.py:592` 的 `_emit_block_removed_events`）。
用来验证"某个哈希块何时被存入/驱逐"，是 hash 类案例最直接的证据链。

### 4. 排查工具

- 挂死：`py-spy dump --pid <api_server_pid>` 看调度循环栈；
  `npu-smi info` 看 AICore 算力是否为 0（区分"卡死在计算"与"卡死在调度"）。
- 哈希算法：v0.11 起 prefix cache 默认 sha256（`--prefix-caching-hash-algo`），
  之前用 Python 内建 `hash()`，存在理论碰撞面（见案例 01 延伸）。

## 上游必读

- vLLM APC 设计文档: <https://docs.vllm.ai/en/stable/design/v1/prefix_caching.html>
- vLLM 自动前缀缓存 RFC: <https://github.com/vllm-project/vllm/issues/2614>（hash 换 trie 的动机、LRU+引用计数策略）
- vllm-ascend prefix cache 优化 RFC: <https://github.com/vllm-project/vllm-ascend/issues/10517>

## 收集新案例

复制任意一个 case 文件，按 9 段模板填写；出处不确定 PR 号时只给 issue 链接，
不要编造编号；`file:line` 引用以本地 main 为准并注明日期。
