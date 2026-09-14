# Case 06 | 自设计实验：cache_salt 多租户隔离验证

| 项 | 内容 |
|---|---|
| 出处 | 官方设计文档 [Automatic Prefix Caching → Cache Isolation for Security](https://docs.vllm.ai/en/stable/design/v1/prefix_caching.html) + 本地实现；实验为**自设计** |
| 类型 | 特性验证（教学：block hash 的链式构成） |
| 难度 | 低（单卡，10 分钟） |

## 摘要

Prefix cache 是进程内全局共享的：所有人只要文本前缀相同就能复用同一份 KV。
共享部署/多租户 SaaS 必须防止"我用拼写变化探测缓存命中时序，
侧信道推断另一个租户的系统提示词"。cache_salt 的机制就是**在首块的哈希里混入租户盐**，
靠父哈希链式传导，让不同 salt 的整条前缀链天然不命中。

这个实验帮助理解 block hash 的三元组构成
（parent_hash, block_tokens, extra_keys）——案例 01 的多模态修复、
LoRA 区分、本案例的 salt，全都走同一个 extra keys 通道。

## 机制

实现位置（本地 main）：

- `vllm/v1/core/kv_cache_utils.py:446`：`need_extra_keys` 把 `cache_salt` 列为需要 extra keys 的情形之一；
- `vllm/v1/core/kv_cache_utils.py:579-586`：
  `generate_block_hash_extra_keys` 仅当 `start_token_idx == 0`（首块）时
  追加 `extra_keys = [request.cache_salt]`；
- 后续块的 extra keys 为空，但 parent_hash 里已经"掺了盐"，
  整条链的哈希全部改变 → 跨 salt 零命中。

请求侧：OpenAI API 里给每个请求带 `cache_salt` 字段（字符串）。
文档示例：约 95% 流量带 `salt="trusted"`，其余每请求随机盐，隔离不可信流量。

## 实验设计

材料：同一段 ≥ 1000 token 的共享长前缀（模拟系统提示）+ 三个短问题。

| 组 | 请求 | 预期 |
|---|---|---|
| A1 | 前缀+Q1，salt="tenant-A" | miss，TTFT 长 |
| A2 | 前缀+Q2，salt="tenant-A" | **hit**（TTFT 显著下降，cached_tokens≈前缀长） |
| B1 | 前缀+Q1，salt="tenant-B" | miss（虽然 token 完全相同） |

TTFT 用流式 `stream=True`、首个 chunk 到达时间测；
同时对照 `/metrics` 的 `vllm:prefix_cache_hits` 增量。

## NPU 复现步骤

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --gpu-memory-utilization 0.85 --enforce-eager
```

```bash
python scripts/client_cache_salt_test.py \
  --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
  --prefix-tokens 1024
```

脚本自动生成长前缀（重复 pattern 保证 tokenizer 后仍足够长）、
按 A1→A2→B1 顺序计时发请求，输出 TTFT 表与 pass/fail。

### 判定

```
PASS: TTFT(A2) << TTFT(A1) 且 TTFT(B1) ≈ TTFT(A1)；
      且 metrics 上 A2 贡献了一次 hits 跳变，B1 没有。
FAIL: B1 也命中（salt 失效）→ 检查版本是否带 cache_salt 支持（0.9+）。
```

顺带验证一个负例：不带 salt 的请求与带 salt 的请求互相不命中（salt 语义是双向隔离）。

## 关联源码（本地 main）

| 位置 | 说明 |
|---|---|
| `vllm/v1/core/kv_cache_utils.py:446` | salt 进入 need_extra_keys 判定 |
| `vllm/v1/core/kv_cache_utils.py:579-586` | 首块注入 `[cache_salt]` |
| `vllm/v1/core/kv_cache_utils.py:596` | `hash_block_tokens`（三元组哈希） |

## 延伸阅读

- 与多模态（案例 01）对照学习：mm hash 注入**每个含视觉 token 的块**，
  salt 只注入**首块**——原因都是"父哈希链式传导"，注入点选择取决于污染范围。
- 哈希算法选择的安全注记（文档 Note）：`xxhash` 等非密码学哈希理论上增加碰撞
  与跨租户串读风险，多租户场景用默认 sha256。
