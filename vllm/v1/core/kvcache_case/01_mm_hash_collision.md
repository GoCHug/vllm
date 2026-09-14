# Case 01 | 多模态占位 token 哈希碰撞：Prefix Caching 回答"张冠李戴"

| 项 | 内容 |
|---|---|
| 出处 | [vllm issue #20261](https://github.com/vllm-project/vllm/issues/20261)（vLLM 0.9.1，2025-06-30）；修复随 v0.10 前后合入上游，机制见下方"如何修复" |
| 类型 | 正确性 / 缓存键冲突 |
| 硬件相关性 | 与硬件无关（bug 在调度器纯 Python 的 hash 层），NPU 上照常复现 |
| 难度 | 低（单卡即可，需要一个多模态模型） |

## 一句话摘要

v1 的 prefix cache 只按"块内 token ids"做缓存键，而多模态输入里的图片会被替换成
相同的占位 token 序列 —— 于是**内容完全不同的两张图会命中同一个缓存块**，
第二个请求直接复用第一张图的 KV，输出乱码或"看图说错话"。

## 问题现象

issue #20261 报告（Qwen2.5-VL，vLLM 0.9.1，prefix cache 默认开启）：

- 所有请求使用**相同文本 prompt，但图片不同**（不同结构的迷宫图）；
- 高并发下输出出现重复、截断、乱码，prefix cache 命中率约 40%（异常偏高）；
- 加 `--no-enable-prefix-caching` 后问题完全消失。

单个请求串行发也可复现：请求 A（图 1 + 文本 T）正常回答；请求 B（图 2 + 同文本 T）
的第二问直接沿用 A 的视觉信息，或输出与视觉无关的重复 token。

## 根因

v1 prefix cache 的缓存键是三元组哈希（父块哈希 → token ids → extra keys）：

```text
BlockHash = hash(parent_hash, block_tokens, extra_keys)
```

修复前 `extra_keys` 为空：图片在 tokenizer 阶段被展开成 `<image>` 占位 token，
**不同图片的占位 token 序列完全相同**，文本部分也相同，于是两个请求从首块开始
逐块同哈希 → 请求 B `get_computed_blocks()` 直接命中请求 A 留下的 KV 块。

对照本地 main 的代码（已含修复）理解数据流：

- `vllm/v1/core/kv_cache_utils.py:430` `need_extra_keys()`：判断请求是否需要 extra keys；
- `vllm/v1/core/kv_cache_utils.py:455` `_gen_mm_extra_hash_keys()`：把
  `(mm_item.identifier, 在 prompt 中的偏移 offset)` 追加进 extra_keys；
- `vllm/v1/core/kv_cache_utils.py:596` `hash_block_tokens()`：最终
  `hash(parent_hash, tuple(token_ids), tuple(extra_keys))`。

## 如何修复

把多模态输入的**内容哈希 + 位置偏移**注入缓存键（即文档中 "Extra hashes" 一节）：

```text
Block 0
 Parent hash: None
 Token IDs: 1, 3, ..., <p>, ..., <p>
 Extra hash: <image hash>        # ← 修复点
Block 1
 Parent hash: Block 0 hash       # 链式继承，后续每块都带上
 Token IDs: <p>, ..., <p>
 Extra hash: <image hash>
```

本地 main 已包含该修复（`kv_cache_utils.py` 的 `_gen_mm_extra_hash_keys`，
extra keys 同时覆盖 mm、LoRA、cache_salt 三类）。同一时期还把哈希算法从
Python 内建 `hash()` 升级为 sha256（v0.11 起默认，`--prefix-caching-hash-algo`
可选 `sha256_cbor` / `xxhash`），消除内建哈希跨进程不稳定与理论碰撞面。

## NPU 复现实验

两种玩法：

- A（复现真 bug）：用 vLLM ≤ 0.9.x / vllm-ascend 引入 mm hash 之前的版本；
- B（修复验证 + 机制演示，推荐）：用当前版本，验证"两图不再互相命中"。

服务端（vllm-ascend，单卡 A2/A3 即可，Qwen2.5-VL-3B 保守起见用 TP1）：

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enforce-eager
```

客户端：

```bash
python scripts/client_mm_hash_two_images.py \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-VL-3B-Instruct
```

脚本行为：生成两张随机迷宫图 → 用完全相同的文本 prompt 各发一次（temperature=0）
→ 连续各发第二次（模拟前缀已入缓存）→ 打印每次回答与 `usage.prompt_tokens_details.cached_tokens`。

### 观测与判定

| 期望 | 判定 |
|---|---|
| 修复后：同图第二次请求 `cached_tokens > 0`（命中自己）；换图请求 `cached_tokens == 0` | PASS |
| 旧版 bug：换图请求 `cached_tokens > 0`，且答案与图不符/乱码 | 复现成功 |

离线机制演示（不需要 NPU 和模型）：

```bash
python scripts/repro_mm_hash_collision_demo.py
```

它复刻"仅 token 哈希"与"token + mm extra keys"两种键构造，
打印两张不同图片前缀块的哈希：旧方案同哈希、新方案不同哈希。
真实实现对照 `kv_cache_utils.py` 的 `_gen_mm_extra_hash_keys`。

## 关联源码（本地 main）

| 位置 | 说明 |
|---|---|
| `vllm/v1/core/kv_cache_utils.py:430` | `need_extra_keys`：mm/LoRA/cache_salt 判定 |
| `vllm/v1/core/kv_cache_utils.py:455-514` | `_gen_mm_extra_hash_keys`：mm 内容哈希 + 偏移注入 |
| `vllm/v1/core/kv_cache_utils.py:558` | `generate_block_hash_extra_keys`：extra keys 总装 |
| `vllm/v1/core/kv_cache_utils.py:596` | `hash_block_tokens` |
| `vllm/v1/core/sched/scheduler.py` | `schedule()` 中 `get_computed_blocks` → 命中路径入口 |

## 延伸阅读

- 同族问题（版本演进中陆续修）：
  - [vllm#43587](https://github.com/vllm-project/vllm/issues/43587)：Qwen3.5（hybrid）多轮对话"每轮多一张图"场景 `num_cached_tokens` 恒为 0；
  - [vllm#52583](https://github.com/vllm-project/vllm/issues/52583)：大多模态输入 + prefix caching hash 对齐逻辑挂死；
  - [vllm#9790](https://github.com/vllm-project/vllm/issues/9790)：最早直接禁用"多模态 + prefix caching"的历史包袱。
- hash 碰撞与缓存串读的安全讨论：vLLM 论坛 "Avoiding hash collisions in prefix cache"。
