# Case 05 | 自设计实验：Abort 风暴下的 KV Cache 泄漏不变量检测

| 项 | 内容 |
|---|---|
| 出处 | 自设计（压力观测实验）。机制背景为历史上的真实泄漏类修复：[vllm#31857](https://github.com/vllm-project/vllm/pull/31857)（waiting 请求 encoder cache 泄漏导致调度卡死）、vllm-ascend release notes 中多个 "KV cache edge cases" 修复（如 0.20.2rc1 的 `#9456 #9487 #9488 #9500`） |
| 类型 | 健壮性 / 泄漏探测 |
| 难度 | 低-中（单卡小模型 + 脚本，还能当 CI 用） |

## 摘要

KV cache 管理器的一切正确性最终都收敛到一个不变量：
**每个块要么在 free 队列、要么 ref_cnt>0 被 pending/running 请求持有，两者不重不漏。**
`touch`（引用 +1、出 free 队）与 `free`（减到 0 回队）只要有一处路径漏调——
终止请求、abort、preempt、waiting 队列异常 —— 就会产生"幽灵块"：
内存长期不归还、命中率虚高、最终 OOM 或死锁。

本实验不依赖知道具体 bug，而是用 abort 风暴把所有"异常终止路径"都打到，
然后盯住三个不变量。它是案例 03 中 encoder cache 释放类问题的**通用探测器**。

## 问题现象（探测器要抓什么）

健康系统：负载冲击后idle，`gpu_cache_usage_perc` 应回落到基线水平；
prefix cache 里保留一些热点块（LRU 语义）属正常。

泄漏系统（历史同类 bug 的共同表现）：

1. idle 后 usage 停在高位不再回落（块既不在 free 队也不再被任何请求引用）；
2. 随周期性负载，usage 低点逐周期抬升（只进不出的锯齿）；
3. 新请求开始出现 `preemption`（明明没有活跃负载）。

## 实验设计

三阶段，全程每 2s 抓一次 `/metrics`：

```
Phase 1  基线    ：N 个请求正常跑完 → 记录稳态 usage_u，验证命中率上升
Phase 2  风暴    ：N 轮请求，每轮 60~80% 在首 token 到达后主动断连（abort）
Phase 3  静置    ：停发 120s，观察 usage 是否回落到 base + cache 驻留容忍量
```

### 判定标准

| 检查 | PASS | 疑似泄漏 |
|---|---|---|
| Phase3 usage 回落 | usage ≤ usage_u + 驻留容忍（如 20%） | 高位不回落 |
| usage 低点趋势 | 静止且低于告警线 | 低点单调抬升 |
| 日志 | 无异常栈 | 出现 assertion / 块计数不符 |

泄漏定位后的黄金验证：开启 `--enforce-eager` 在泄漏复现路径下把
`BlockPool` 的统计打出来（离线模式可用 `LLM.llm_engine` 直接读
`free_queue` 长度 + 汇总 `ref_cnt>0` 块数，两者之和应恒等于总块数）。

## 根因教学（为什么 abort 路径最容易漏）

对照本地 main：

| 位置 | 职责 |
|---|---|
| `vllm/v1/core/block_pool.py:702` `touch` | 命中块 ref_cnt+1 并移出 free 队 |
| `vllm/v1/core/block_pool.py:719` `free_blocks` | 归还块：ref_cnt-1，归零时入队尾（逆序入队，深前缀先逐出） |
| `vllm/v1/core/block_pool.py:679` `_maybe_evict_cached_block` | 复用前逐出旧哈希块 |
| `vllm/v1/core/kv_cache_manager.py:567` `free` | 请求结束时统一归还全部组 |
| `vllm/v1/core/kv_cache_manager.py:599` `pop_blocks_for_free` | 归还前的块表快照 |

泄漏的经典形态 = 有 `touch`/`allocate`、没有配对的 `free`：
- 请求在 waiting 队列就 abort，encoder cache 已扣预算但没走到 `free`（#31857 修的形态）；
- preempt 与 async 调度交错时，归还顺序被 fence 打乱（相关机制见
  `scheduler.py` 的 `deferred_frees` 与 `_pause_state`）；
- 客户端断连后 abort 没触发（前端开关姿态不同）。

修复范式都是同一句话：**把"归还"收敛到请求生命周期的统一出口**
（scheduler 的 finish/abort 分支），并给 encoder cache 类二级缓存补 FE/RE 平衡。

## NPU 复现步骤

服务端（复现用小模型 + 小 KV，让水位变化肉眼可见）：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.25 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --enforce-eager
# （低 gpu-memory-utilization 是故意把 KV 池缩小，usage 波动更灵敏）
```

客户端：

```bash
python scripts/client_abort_storm.py \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-0.5B-Instruct \
  --rounds 6 --concurrency 24 --abort-ratio 0.7 --output usage.csv
```

脚本做的事：每轮并发发长 prompt 流式请求（prompt 长度 ≥ 上千 token，
保证 KV 占用可观），按比例随机在某时刻 `response.close()` 模拟客户端断连，
后台线程抓 `/metrics` 写 CSV，结束时打印三阶段判定摘要。

### 结果解读与下一步

| 出现的形态 | 下一步 |
|---|---|
| PASS | 把脚本加进回归 CI，换 hybrid/MM 模型再跑（多模态会多一条 encoder cache 泄漏面） |
| usage 不回落 | 二分变量：关 prefix caching / 关 MM / 换 eager，缩到最小复现组合，再对 `free` 调用链加日志 |
| 低点抬升 | 同上，重点观察 waiting 请求 abort 时序（#31857 形态） |

## 关联源码（本地 main）

| 位置 | 说明 |
|---|---|
| `vllm/v1/core/block_pool.py:647` `get_new_blocks` / `:679` | 分配与逐出 |
| `vllm/v1/core/block_pool.py:702` / `:719` | touch / free（不变量的两根支柱） |
| `vllm/v1/core/encoder_cache_manager.py:68-267` | encoder cache 全套申请/释放（MM 泄漏面） |
| `vllm/v1/core/sched/scheduler.py` `deferred_frees` | 异步释放 fence（preempt/abort 交错点） |

## 延伸阅读

- 同类思路在工程里的变体：给 KVCacheManager 写一个
  pytest fixture 版校验器（每步 schedule 后断言不变量），历史多个泄漏 bug 都能被它提前拦下。
- 案例 03 是本实验某一泄漏面的具体化；案例 04 是"泄漏不发生但命中归零"的姊妹篇。
