# Issue #32802 | Hybrid 模型 + EAGLE 投机解码：Prefix Cache 命中率坍塌为 0

| 项 | 内容 |
|---|---|
| Issue | [#32802](https://github.com/vllm-project/vllm/issues/32802)（GPT-OSS-120B + Eagle3，2026-01-30 报告；报告者同时提交实验性 PR [#32801](https://github.com/vllm-project/vllm/pull/32801)，未被采用） |
| 引入提交 | `cd4a95e3aa`（PR [#31707](https://github.com/vllm-project/vllm/pull/31707)，2026-01-09：Hybrid Coordinator 支持多 KV cache group，引入收敛 `while` 循环——bug 的必要条件） |
| 修复 PR | [#33524](https://github.com/vllm-project/vllm/pull/33524)（`a01ef3fa51`，2026-02-01，`is_simple_hybrid` 单轮退出，首发 `v0.15.2rc0+`）；复杂 hybrid 由 [#40860](https://github.com/vllm-project/vllm/pull/40860)（`4d51588e23`，引入 `eagle_verified`，源码注释即引 #32802）与 [#44082](https://github.com/vllm-project/vllm/pull/44082)（SWA lookahead mask，`drop_eagle_block` 参数化）继续演进 |
| 源码引用 | `kv_cache_coordinator.py:722-724`（`eagle_verified` 注释） |
| 类型 | 命中率回归 / 协调器逻辑（纯 Python 层，无硬件相关性） |
| 深度资料 | [../kvcache_case/09_hybrid_eagle_prefix_cache_zero.md](../kvcache_case/09_hybrid_eagle_prefix_cache_zero.md)（本仓库 git worktree A/B 确定性复现 + 版本时间线） |

## 一句话摘要

`HybridKVCacheCoordinator.find_longest_cache_hit()` 的收敛循环里，EAGLE drop 被**重复叠加**：full attention 先 drop 末块（N→N-1），缩小后的长度传给 SWA → SWA 在缩小的范围内再次 pop 且命中门槛还要 +1 → 返回更短乃至 0 → 触发新一轮迭代……螺旋下降直至命中归零——GPT-OSS + EAGLE 场景 prefix cache 命中率 0.0%。修复分两层：simple hybrid 单轮退出（#33524）+ `eagle_verified` 保证每个 EAGLE 组对同一候选长度至多 drop 一次（#40860）。

---

## 一、问题现象（issue 原始报告）

```text
GPT-OSS-120B + Eagle3 (num_speculative_tokens=3)  → Prefix cache hit rate: 0.0%
GPT-OSS-120B（不开投机解码）                        → 88.7%
GPT-OSS-120B + Eagle3 + --disable-hybrid-kv-cache-manager → ~80%（workaround）
Qwen3 + EAGLE                                       → 正常（非 hybrid，不走 HybridCoordinator）
```

- 负载：`vllm serve openai/gpt-oss-120b --speculative-config '{"model": "nvidia/gpt-oss-120b-Eagle3-long-context", "num_speculative_tokens": 3, "method": "eagle3"}'` 跑 `tests/evals/gsm8k/gsm8k_eval.py`，重复前缀命中率应 ~80%，实测 0.0%；
- 关 EAGLE 立即恢复、`--disable-hybrid-kv-cache-manager` 也能恢复 ⇒ 指向 **hybrid 协调器 × EAGLE 的组合**；
- 微缩形态（单测级）：同前缀请求重查 `get_computed_blocks()`，bug 态返回 `([], [], [])`——0 块命中。

触发条件组合（缺一不可）：

```text
① hybrid attention 模型（≥2 种 attention 类型 → HybridKVCacheCoordinator）
② EAGLE/MTP 投机解码（use_eagle=True → 各 manager 的 drop/阈值逻辑被激活）
③ prefix caching 开启（默认开启）
```

### EAGLE drop 的正确动机（为什么 manager 要丢一块）

EAGLE/MTP 的 draft head 需要**生成点之前若干 token 的 hidden states**；而 KV 命中意味着这些 token 不再过 forward——所以最后一个命中块必须丢掉强制重算。单 manager（unitary）路径下每个只执行一次，行为正确；bug 出在协调器把 drop **叠乘**。

---

## 二、根因分析

预修复 `find_longest_cache_hit`（`a01ef3fa51^`）的收敛循环要义：

```python
while True:
    curr_hit_length = hit_length
    for spec, group_ids, manager_cls in self.attention_groups:
        if is_full_attn and cached_blocks is not None:
            # 仅 full attn 有"后续迭代只需截断"的 shortcut
            curr_hit_length = (curr_hit_length // block_size) * block_size
            del blocks[num_blocks:]
        else:
            # SWA 没有等价 shortcut：每轮都以(缩小的)curr_hit_length
            # 重新完整调用，且 use_eagle=True 原样再传!
            hit_blocks = manager_cls.find_longest_cache_hit(
                ..., max_length=curr_hit_length, use_eagle=self.use_eagle)
    if curr_hit_length < hit_length:
        hit_length = curr_hit_length     # 收缩 → 下一轮
    else:
        break
```

螺旋坍塌一轮内的因果链（issue 作者分析，与修复 PR 验证一致）：

```text
迭代 k:   full attn 命中 N 块 → EAGLE pop → 返回 N-1 块
          SWA 收到 max_length = (N-1) * block_size，右到左扫 [0, N-1)
          SWA 窗内缓存块在 [N-w, N-1]——位置 N-1 被切在搜索范围外
          且 SWA 命中门槛因 EAGLE +1 本就更紧 → 连续尾部块不足 ⇒ 命中大减/归零
          SWA 自己又 pop 一块（use_eagle 再一次生效）
迭代 k+1: full 被截到更短 → SWA 范围再缩 → ……
循环直到 hit_length 归零（实测场景返回 ([], [], []) 收敛）
```

三个缺陷叠加：

1. **FullAttentionManager 的 pop**：`if use_eagle: computed.pop()`（N→N-1），正确动机但每轮都被触发前的状态影响；
2. **SWA 的阈值 +1 与"末块排除"**：SWA 以 full 缩短后的长度为 `max_length`，窗内尾部块被切在界外，门槛 `cdiv(sliding_window-1, block_size)+1` 又要求连续尾部块；
3. **收敛循环对非 full 组的重复调用**（核心缺陷）：full 有二次迭代 shortcut，SWA 没有——每轮完整重查且 `use_eagle` 原样再传 ⇒ 每轮再 pop 一块（螺旋）；full 的 `del blocks[...]` 也在循环内反复执行。

> 版本窗口：**v0.10 ~ v0.15.1 之间含 #31707 的主线构建**。v0.9.x 的 coordinator 是"full 先查带 drop → 以缩小长度查 SWA 一次"的**两步直调**结构（无循环），SWA 只 pop 一次——无螺旋的结构条件，不受影响。

---

## 三、解决办法

### 3.1 PR #33524（`a01ef3fa51`）：simple hybrid 单轮退出

- 新增 `is_simple_hybrid`（恰 2 个 attention group 且首个是 `FullAttentionSpec`，即 GPT-OSS 等绝大多数 hybrid 的形态）——循环体执行**一轮后直接 break**（现行代码 `kv_cache_coordinator.py:795-796`）：full 只 pop 一次、SWA 只被调用一次只 pop 一次，螺旋条件被摘除；
- full attention 的截断从"循环内反复 del"移到**循环结束后**统一按最终 `hit_length` 截断（`:798-808`）。

### 3.2 PR #40860（`4d51588e23`）：`eagle_verified` 闭合复杂 hybrid

#33524 留下了 "complex hybrid（>2 个 attention group，如 DSv4 的 5 个 KV group）仍有 EAGLE spiral" 的 FIXME，由 `eagle_verified` 闭合。现行代码（`kv_cache_coordinator.py:685-817`）：

```python
# :725  Attention-group indices whose EAGLE drop is verified at the current
#       ``curr_hit_length``. Each eagle group applies the drop at most once
#       per candidate length (see issue #32802).
eagle_verified: set[int] = set()

while True:
    ...
    # :747  本轮是否对该组执行 drop：未验证过的 eagle 组才 drop
    drop_eagle_block = use_eagle and idx not in eagle_verified

    # :755-765  Eagle 匹配多带一个 drop 单位（细粒度管理器为一个 hash 单位，
    #           否则一个 cache block）后再丢掉，落回候选长度；mamba 从不 drop
    #           （draft 模型没有 mamba 层）
    if drop_eagle_block and not isinstance(spec, MambaSpec):
        eagle_margin = (...)
        _max_length = min(curr_hit_length + eagle_margin, max_cache_hit_length)

    hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
        ..., drop_eagle_block=drop_eagle_block, ...)

    # :780-784
    if drop_eagle_block:
        eagle_verified.add(idx)          # 该组在"当前候选长度"下的 drop 已验证
    elif _new_hit_length < curr_hit_length:
        eagle_verified.clear()           # 长度收缩 → 在新长度下复验（重新允许 drop）

    if curr_hit_length >= hit_length:
        break
    hit_length = curr_hit_length
    if is_simple_hybrid:
        break
```

语义要点：

- **每个 eagle 组对同一候选长度至多 drop 一次**：收敛循环若在同一长度上重查该组，`idx in eagle_verified` 使 drop 关闭，匹配结果与上一轮一致——螺旋被切断；
- **长度收缩时 `eagle_verified.clear()`**：候选长度变小后重新允许 drop 并复验，保证较短长度上的匹配仍带上"丢一块以取 hidden state"的语义；
- `eagle_margin`：允许 manager 在 drop 前多匹配一个单位（回退到候选长度），避免 drop 后长度错位入窗。

### 3.3 后续演进

- **PR #44082**（`e9e08c49b9`）："Cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask"——`drop_eagle_block` 参数化，SWA 端 pop 语义演进（lookahead 块入 mask 而非简单丢弃）；
- **PR #48425**：per-group prefix-hit divergence 的继续修补（同族收敛正确性）。

---

## 四、关联源码（当前 main，2026-09 实查）

| 位置 | 说明 |
|---|---|
| `kv_cache_coordinator.py:685-817` | `HybridKVCacheCoordinator.find_longest_cache_hit`（固定点算法 + `eagle_verified`） |
| `kv_cache_coordinator.py:718-720` | `is_simple_hybrid` 判定——PR #33524 的核心 |
| `kv_cache_coordinator.py:725, 747, 780-784` | `eagle_verified` 机制（#40860，注释引 #32802） |
| `kv_cache_coordinator.py:750-765` | `eagle_margin`（mamba 不 drop；细粒度 hash 单位） |
| `kv_cache_coordinator.py:795-796` | simple hybrid 单轮退出 |
| `kv_cache_coordinator.py:798-808` | full attn 截断移至循环后执行 |
| `kv_cache_coordinator.py:810-813` | `longest_hit_length - hit_length` → `num_uncached_common_prefix_tokens`（稀疏保留组未缓存的共享前缀） |
| `single_type_kv_cache_manager.py` | FullAttentionManager 的 EAGLE drop（`use_eagle` 字段与 pop 逻辑）、SWA 的 `get_num_required_blocks`（阈值 +1 预占）与 find（右到左扫描、pop 后 re-align） |
| `kv_cache_manager.py`（`use_eagle` 传递） | `use_eagle` 一路传递到 coordinator/manager |

## 五、验证与复现

- 回归测试（随 #33524 合入，`tests/v1/core/test_prefix_caching.py`）：`test_prefill_hybrid_model_eagle`、`test_prefill_hybrid_model_combinations_eagle`（后者注释即 "More complex hybrid models with EAGLE are not yet supported (see issue #32802)"）；
- L1 确定性 A/B（纯 CPU、秒级）：`bash ../kvcache_case/scripts/run_hybrid_eagle_ab_test.sh`——bug 提交 `7320ca3942` 2 failed（`([], [], [])` 断言失败）/ fix 提交 `a01ef3fa51` 2 passed。完整步骤、环境踩坑与版本实查见 [../kvcache_case/09_hybrid_eagle_prefix_cache_zero.md](../kvcache_case/09_hybrid_eagle_prefix_cache_zero.md)。

## 六、修复方案取舍（issue 原文三方案）

| 方案 | 内容 | 结局 |
|---|---|---|
| Option 1 | 把 EAGLE 逻辑全部上收 coordinator，manager 不感知 | 改动最大，未立即采用；主线后来部分走向该方向（#44082/#40860 把 drop 决策参数化 + coordinator 统一控制） |
| Option 2 | EagleMode enum（作者实验 PR #32801） | 未采用 |
| Option 3（实际） | 不动各 manager，修协调器对循环的滥用（单轮退出 + 至多 drop 一次） | #33524 + #40860 合入 |

> 防误读：#33524 **不是**"把 EAGLE 逻辑从各 manager 移除"，修的是协调器收敛循环对 SWA 的重复调用与 full 截断时机；manager 端的 drop/threshold 逻辑全部保留并经 #44082 继续演进。
