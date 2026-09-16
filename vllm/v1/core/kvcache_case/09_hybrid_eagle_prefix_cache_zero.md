# Case 09 | Hybrid 模型 + EAGLE 投机解码：Prefix Cache 命中率坍塌为 0

| 项 | 内容 |
|---|---|
| Issue | [#32802](https://github.com/vllm-project/vllm/issues/32802)（GPT-OSS-120B + Eagle3，2026-01-30 报告；issue 作者同时提交了实验性 PR [#32801](https://github.com/vllm-project/vllm/pull/32801)） |
| 引入提交 | `cd4a95e3aa`（PR [#31707](https://github.com/vllm-project/vllm/pull/31707)，2026-01-09：Hybrid Coordinator 支持多 KV cache group，引入收敛 `while` 循环——bug 的必要条件） |
| 修复 PR | [#33524](https://github.com/vllm-project/vllm/pull/33524)（本仓库 `a01ef3fa51`，2026-02-01，首发 tag `v0.15.2rc0+`，`git tag --contains` 实查）；后续演进：#44082（`e9e08c49b9`，SWA lookahead mask）、#40860（`4d51588e23`，`eagle_verified`） |
| 类型 | 命中率回归 / 调度器协调逻辑（纯 Python 层，无硬件相关性） |
| 难度 | 单测级 **低（纯 CPU、秒级、确定性）**；服务级高（GPT-OSS-120B + Eagle3 draft 模型且需 bug 窗口内版本） |
| 实测 | 2026-09-15 已在本仓库 git worktree 上 A/B 确定性复现（bug 态 FAILED / fix 态 PASSED），见 §2.2 / §7 |

## 一句话摘要

`HybridKVCacheCoordinator.find_longest_cache_hit()` 的收敛循环里，EAGLE drop 被**重复叠加**：full attention 先丢最后一块（N→N-1）并把搜索范围传给 SWA → SWA 窗口内的尾部缓存块（位置 N-1）被排除在范围外，且阈值还要 +1 → SWA 返回更短乃至 0 → 触发新一轮迭代，SWA 在缩小的范围里再做一次 EAGLE pop，full 再被截断……螺旋下降直至命中归零——GPT-OSS + EAGLE 场景 prefix cache 命中率 0.0%。

服务端表现（issue #32802 原始报告）：

```text
GPT-OSS-120B + Eagle3 (num_speculative_tokens=3)  → Prefix cache hit rate: 0.0%
GPT-OSS-120B（不开投机解码）                        → 88.7%
GPT-OSS-120B + Eagle3 + --disable-hybrid-kv-cache-manager → ~80%（workaround）
Qwen3 + EAGLE                                       → 正常（非 hybrid，不走 HybridCoordinator）
```

---

## 一、问题现象

- `vllm serve openai/gpt-oss-120b --speculative-config '{"model": "nvidia/gpt-oss-120b-Eagle3-long-context", "num_speculative_tokens": 3, "method": "eagle3"}'` 跑 `tests/evals/gsm8k/gsm8k_eval.py`，重复前缀的问答命中率应为 ~80%，实测 **0.0%**；
- 关掉 EAGLE 立即恢复 88.7%；加 `--disable-hybrid-kv-cache-manager` 恢复 ~80%；
- Qwen3 + EAGLE 正常 ⇒ 排除 EAGLE 本身，指向 **hybrid 协调器 × EAGLE 的组合**；
- 上游后续评论确认 "Seems like this was fixed by #33524"。

刮掉服务层外壳后的微缩形态（本 Case 实测，§2.2）：**同前缀请求重发，`get_computed_blocks()` 返回 `([], [], [])`——0 块命中**。

---

## 二、复现方法

### 2.1 复现结论

**可以复现，且成本极低**——本仓库 git 历史自带完整 bug 现场（引入提交、bug 态、修复提交、回归测试都在），分三层：

| 层 | 路径 | 资源 | 状态 |
|---|---|---|---|
| L1 | 单测级 A/B：bug 提交 vs 修复提交，跑修复 PR 自带回归测试 | 纯 CPU、秒级 | **2026-09-15 已实测 REPRO** |
| L2 | 服务级原始路径：GPT-OSS-120B + Eagle3 + gsm8k | GPT-OSS 权重（多卡 GPU）+ bug 窗口内 vllm 版本 | 未执行（§2.4 说明） |
| L3 | 指标观测面：`/metrics` 命中率 + 日志关键字 | 任意 vllm 服务 | 服务级判定入口 |

### 2.2 L1 单测级 A/B（推荐，本次实测）

**原理**：在 `a01ef3fa51^`（bug 态，2026-02-01 前）与 `a01ef3fa51`（PR #33524 修复态）各建 git worktree；回归测试 `test_prefill_hybrid_model_eagle` / `test_prefill_hybrid_model_combinations_eagle` 是**随修复 PR 合入**的，把它同步到 bug 态后两边跑同一条测试——只有 coordinator 的修复差异在中间：

```bash
bash scripts/run_hybrid_eagle_ab_test.sh            # 默认定位本仓库, 产物在 /tmp/kvc_case08
bash scripts/run_hybrid_eagle_ab_test.sh /path/to/vllm   # 或显式指定检出
```

实测输出（2026-09-15，macOS arm64 + Python 3.12.3 + torch 2.7.1 CPU）：

```text
bug       = 7320ca3942d0 (2026-02-01)   = a01ef3fa51^
fix       = a01ef3fa51   (2026-02-01)   = PR #33524
--- bug 态(未修复) ---
    2 failed, 1 warning in 7.12s
--- fix 态(PR #33524) ---
    2 passed, 1 warning in 5.74s
==================== 判定 ====================
REPRO: bug 态 eagle 回归测试失败(同前缀请求 get_computed_blocks 返回 ([],[],[])),
       fix 态通过 —— issue #32802(命中率坍塌为 0)在本仓库确定性复现
```

bug 态的关键失败断言（`tests/v1/core/test_prefix_caching.py:557`，完整 traceback 见 `/tmp/eagle_bug_fail.txt` 形态）：

```text
E       assert ([], [], []) == ([1, 2, 3, 4], [0, 9, 10, 11], [0, 16, 17, 18])
```

场景是**教科书式微缩**：前一个请求已把 6 个完整块写入缓存（full 组 + 两个 SWA 组），新请求带**完全相同的前缀**再查 `get_computed_blocks()`——修复态命中 3 组 × 4 块（full 6 块被 EAGLE drop 与 SWA 交集各截 1 块），bug 态**一组都命中不了**。这就是服务端 0.0% 命中率的单机形态：慢不是问题，命中等价于全 miss。

**环境依赖与踩坑**（本机缺谁就装谁，详见 §7.1）：

| 坑 | 现象 | 解决 |
|---|---|---|
| torch 版本 | `ModuleNotFoundError: No module named 'torch._inductor.custom_graph_pass'` | 2026-02 代码需 torch ≥ 2.6；实测装 `torch==2.7.1`（CPU wheel 即可） |
| 依赖集 | `numpy/regex/psutil/pyzmq/msgspec/pydantic/hf_hub/transformers/gguf/...` 逐个报错 | `git show a01ef3fa51:requirements/common.txt > /tmp/common.txt && pip install -r /tmp/common.txt` 一把装齐 |
| conftest 重链 | 跑 pytest 拖起 `tests/conftest.py` → engine/arg_utils 全家桶 | 加 `--noconftest`（脚本已内置） |
| 走错检出 | `import vllm` 命中其它安装 | `PYTHONPATH` 指向对应 worktree（脚本已内置） |

### 2.3 L1 手工版（不依赖脚本，便于移植到任何机器/容器）

```bash
git worktree add /tmp/wt_bug a01ef3fa51^
git worktree add /tmp/wt_fix a01ef3fa51
git -C /tmp/wt_bug checkout a01ef3fa51 -- tests/v1/core/test_prefix_caching.py
for d in /tmp/wt_bug /tmp/wt_fix; do
  ( cd "$d" && PYTHONPATH="$d" python3 -m pytest \
      "tests/v1/core/test_prefix_caching.py::test_prefill_hybrid_model_eagle" \
      "tests/v1/core/test_prefix_caching.py::test_prefill_hybrid_model_combinations_eagle" \
      --noconftest -q )
done   # 预期: bug 2 failed, fix 2 passed
```

### 2.4 L2 服务级原始路径（重资源，本 Case 未执行）

原始 issue 的命令，供在 **bug 窗口内版本** 上验证服务级现象（窗口 = 含 #31707 且未含 #33524，约 2026-01-09 ~ 2026-02-01 的主线）：

```bash
# bug 组: 期望 0.0%
vllm serve openai/gpt-oss-120b \
  --speculative-config '{"model": "nvidia/gpt-oss-120b-Eagle3-long-context", "num_speculative_tokens": 3, "method": "eagle3"}'
# 对照 1: 期望 ~88%
vllm serve openai/gpt-oss-120b
# 对照 2 workaround: 期望 ~80%
vllm serve openai/gpt-oss-120b \
  --speculative-config '{...同上...}' --disable-hybrid-kv-cache-manager
```

负载与判定：`python tests/evals/gsm8k/gsm8k_eval.py`（固定 few-shot 前缀，天然高重复前缀），观测见 §2.5。

### 2.5 NPU 复现评估（结论：不需要、也不匹配）

- 本地常备 NPU 栈 vllm-ascend 0.9.1（基座 vllm **v0.9.1**，2025-06）：其 `HybridKVCacheCoordinator` 是**两步直调**结构（`v0.9.1:kv_cache_coordinator.py:207-345`：full 先查带 drop，再以缩小后的长度查 SWA 一次，无收敛 `while` 循环）——**没有螺旋坍塌的结构条件，不复现**。v0.9.2 同构。这不是巧合而是结构差异：循环是 #31707（2026-01-09）才引入的。
- 服务级复现此 bug 还需 GPT-OSS-120B（≈120B MoE）+ Eagle3 draft 模型同时可用，资源组合同量级于案例 04，且需把 vllm/vllm-ascend 对齐到 bug 窗口版本——投入产出比远逊于 L1。
- **结论：NPU 上复现此 issue 无必要**。bug 在调度器纯 Python 层（与硬件完全无关），L1 的 CPU 单测 A/B 已是"确定性、可审计"的最强复现；NPU 验证留给它作为回归基座的价值——魔改 vllm/vllm-ascend 后在 NPU 容器里跑 L1（pytest 可直接在容器里运行），可验证协调器逻辑未被改坏。

### 2.6 观测与判定（服务级，若走 L2）

| 面 | 内容 |
|---|---|
| `/metrics` | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`；命中率 = hits/queries（定义见 `../metrics/loggers.py`，README 通用观测一节） |
| 服务端日志 | 每 10s stat 行 `Prefix cache hit rate: 0.0%`（正常应为 70%+） |
| API | 响应 `usage.prompt_tokens_details.cached_tokens` 恒为 0/null（需 `--enable-prompt-tokens-details`，见案例 01 §2.1 双条件坑） |
| pass/fail | EAGLE 开 → 0% **且** EAGLE 关 → ~80% **且** `--disable-hybrid-kv-cache-manager` → ~80% ⇒ 三点齐中即命中 #32802 |

---

## 三、机制与版本演进

### 3.1 触发条件组合（缺一不可）

```text
① hybrid attention 模型（≥2 种 attention 类型 → HybridKVCacheCoordinator）
   GPT-OSS 系列 = full attention + sliding window 交替
② EAGLE/MTP 投机解码（use_eagle=True → 各 manager 的 drop/阈值逻辑被激活）
③ prefix caching 开启（默认开启）
```

Qwen3（全 full attention）只有 ②③ 没有 ①→ 走 `UnitaryKVCacheCoordinator` 直调一次，无问题；`--disable-hybrid-kv-cache-manager` 把 hybrid 模型切到"全 full attention 协调"路径，绕开 ① 同样恢复。

### 3.2 EAGLE drop 的正确动机

EAGLE/MTP 的 draft head 输入需要**生成点之前若干 token 的 hidden states**，而 KV cache 命中意味着这些 token 不再过 forward——所以最后一个命中块必须丢掉强制重算。三处独立实现：

| 位置（预修复 a01ef3fa51^） | 行为 |
|---|---|
| FullAttentionManager | 命中 N 块后 `computed.pop()` → N-1 |
| SlidingWindowManager 阈值 | `sliding_window_contiguous_blocks += 1`（把需要重算的那块"预占"进门槛） |
| SlidingWindowManager 尾部 | 命中后再 `computed.pop()` 丢一块 |

单 manager（unitary）下每个只执行一次，正确。**Bug 出在协调器的循环把这三件事叠乘**。

### 3.3 螺旋坍塌过程（以实测场景为数字底本）

预修复 `find_longest_cache_hit` 的收敛循环（`a01ef3fa51^:kv_cache_coordinator.py:491-525` 要义）：

```python
while True:
    curr_hit_length = hit_length
    for spec, group_ids, manager_cls in self.attention_groups:
        if is_full_attn and cached_blocks is not None:
            # 仅 full attn 有"二次迭代只需截断"的 shortcut
            curr_hit_length = (curr_hit_length // block_size) * block_size
            del blocks[num_blocks:]
        else:
            # SWA 没有等价 shortcut: 每轮都以(缩小的)curr_hit_length
            # 重新调用, 且 use_eagle=True 原样再传!
            hit_blocks = manager_cls.find_longest_cache_hit(
                ..., max_length=curr_hit_length, use_eagle=self.use_eagle)
    if curr_hit_length < hit_length:
        hit_length = curr_hit_length     # 收缩 → 下一轮
    else:
        break
```

一轮内的因果链（issue 作者的根因分析，与修复 PR 验证一致）：

```text
迭代 k:  full attn 命中 N 块 → EAGLE pop → 返回 N-1 块
         SWA 收到 max_length = (N-1) * block_size，右到左扫 [0, N-1)
         SWA 窗内缓存块在 [N-w, N-1]——位置 N-1 被排除在搜索范围外
         且 SWA 阈值因 EAGLE +1 本就更紧 → 连续块不足门槛 → 命中大减/归零
         SWA 自己又 pop 一块（use_eagle 又一次生效）
迭代 k+1: full 被截到更短 → SWA 范围再缩 → ……
循环直到 hit_length 归零（实测场景: 第三组返回 ([], [], []) 收敛）
```

对照数字（实测）：同一缓存现场，bug 态返回 `([], [], [])`，修复态返回 full `[1,2,3,4]` / SWA `[null,9,10,11]` / `[null,16,17,18]`（4 块×3 组 = 64 tokens 命中，即"80% 命中率"形态的微缩）。

### 3.4 版本时间线（本地 git 实查 2026-09-15）

| 时间 | 事件 |
|---|---|
| 2025-06-10（v0.9.1） | hybrid coordinator 两步直调：full（带 drop）→ 以缩小长度查 SWA 一次后截断 —— **无循环**，SWA 只 pop 一次，无螺旋形态 |
| 2025-07-06（v0.9.2） | 结构同 v0.9.1；SWA 的 use_eagle 阈值/pop 已接线 |
| 2026-01-09 | PR #31707（`cd4a95e3aa`）引入多 KV group 支持 + **收敛 while 循环**，bug 的必要条件就位 |
| 2026-01-30 | issue #32802 报告 GPT-OSS+EAGLE 命中率 0.0%；同作者实验 PR #32801（EagleMode enum 方案，未采用） |
| 2026-02-01 | PR #33524（`a01ef3fa51`）：simple hybrid 单轮退出——**bug 窗口关闭**；修复首发于 `v0.15.2rc0+`（`git tag --contains` 实查） |
| 2026 | PR #44082（`e9e08c49b9`）："Cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask"——`drop_eagle_block` 参数化，SWA 端 pop 语义演进 |
| 2026 | PR #40860（`4d51588e23`，DSv4 Rebased）：`eagle_verified` 集合进了 coordinator——修复 PR #33524 留下的 "complex hybrid 仍有 EAGLE spiral" FIXME 被闭合（代码注释直接引用 #32802） |

> 版本窗口一句话：**v0.10 ~ v0.15.1 之间含 #31707 的主线构建**（准确窗口以 #31707 所发 tag 与 v0.15.2rc0 之间的发布为准），hybrid + EAGLE 组合被命中。v0.9.x 是两步结构，vllm-ascend 0.9.1 基座不受影响。

---

## 四、根因分析

三个必要缺陷叠加（预修复代码，`a01ef3fa51^`）：

1. **FullAttentionManager 的 pop**（`single_type_kv_cache_manager.py` 预修复）：

   ```python
   if use_eagle and computed_blocks[0]:
       for computed in computed_blocks:
           computed.pop()      # N → N-1
   ```

2. **SWA 的阈值 +1 与"末块排除"**（issue 根因步骤 2-5）：协调器把 full 缩短后的长度作为 `max_length` 传给 SWA，SWA 的搜索范围 `max_num_blocks = max_length // block_size` 天然把位置 N-1 的窗内缓存块切在界外；而它的命中门槛 `cdiv(sliding_window-1, block_size) + 1`（+1 为 EAGLE 预占）又要求一段**连续**尾部块——被切掉一块后连续段不足 ⇒ 命中坍塌。

3. **收敛循环对非 full 组的重复调用**（核心缺陷，修复 PR 的改动正对这里）：full attn 有"第二轮起只需截断"的 shortcut，SWA **没有**——每轮都以缩小后的 `curr_hit_length` 被**完整重新调用**，`use_eagle=True` 原样再传 ⇒ 每轮再 pop 一块（螺旋）；循环内 full 块的 `del blocks[num_blocks:]` 也在循环内反复执行，加剧不收敛。

修复对比（`git show a01ef3fa51 -- vllm/v1/core/kv_cache_coordinator.py`，恰好 30 行）：**单删 4 处 / 新增 10 处**——

- 新增 `is_simple_hybrid`（恰好 2 个 attention group 且首个是 FullAttentionSpec，即 GPT-OSS 等绝大多数 hybrid 模型的形态）：循环体执行**一轮后直接 break**。full 只 pop 一次、SWA 只被调用一次只 pop 一次，螺旋条件被摘除；
- full attn 的截断从"循环内重复 del"移到"循环结束后统一按最终 hit_length 截断"；
- FIXME 留言：complex hybrid（>2 个 attention group）仍可能有螺旋——由后续 #40860 的 `eagle_verified` 闭合：**每个 eagle 组对同一候选长度至多 drop 一次**（`kv_cache_coordinator.py:725` 附近），长度收缩时 `eagle_verified.clear()` 复验，并为细粒度模式预留 `eagle_margin` 一个单位。

> 三个修复方案的取舍（issue 原文）：Option 1（把 EAGLE 逻辑全部上收 coordinator）最干净但改动大；Option 2（EagleMode enum，PR #32801，未采用）；**实际采用接近 Option 3 的思路**——不动各 manager，修协调器对循环的滥用。主线后来还是走向了 Option 1 的精神（#44082/#40860 把 drop 决策参数化 + coordinator 端统一控制）。

---

## 五、关联源码

本地 main（行号 2026-09-15 实查）与预修复 `a01ef3fa51^` 对照：

| 本地 main 位置 | 预修复对应 | 说明 |
|---|---|---|
| `kv_cache_coordinator.py:685-817` | `:479-525` | `HybridKVCacheCoordinator.find_longest_cache_hit`（现状为固定点算法 + `eagle_verified`） |
| `kv_cache_coordinator.py:718-720` | （新增即修复） | `is_simple_hybrid` 判定——PR #33524 的核心 |
| `kv_cache_coordinator.py:725, 774-793` | 无 | `eagle_verified` 集合（#40860），每 eagle 组每候选长度至多 drop 一次 |
| `kv_cache_coordinator.py:798-808` | 循环内反复 del | full attn 截断移至循环后执行 |
| `single_type_kv_cache_manager.py:108-112` | `:196` 附近 | manager 上 `use_eagle` 字段 |
| `single_type_kv_cache_manager.py:768-776` | `if use_eagle: computed.pop()` | FullAttentionManager 的 EAGLE drop |
| `single_type_kv_cache_manager.py:885-889` | `:478-483`（threshold+1 内联） | SWA `get_num_required_blocks`（+1 预占） |
| `single_type_kv_cache_manager.py:904-995` | `:485-561` | SWA find：右到左扫描、`post_pop_blocks`、pop 后 re-align |
| `kv_cache_manager.py:463` | `use_eagle=self.use_eagle` | `use_eagle` 一路传递到 coordinator/manager |

回归测试（随修复 PR 合入，`tests/v1/core/test_prefix_caching.py`）：`test_prefill_hybrid_model_eagle`（本 Case 的 L1 主证据）、`test_prefill_hybrid_model_combinations_eagle`（参数化多 spec 组合，源码注释即 "More complex hybrid models with EAGLE are not yet supported (see issue #32802)"）。

---

## 六、延伸阅读

- issue #32802 全文（含三方案讨论与 `--disable-hybrid-kv-cache-manager` workaround）与作者的实验 PR #32801（EagleMode enum）。
- PR #33524 的测试组：把"窗口外驱逐 / full 首块驱逐 / 末块驱逐 / SWA 倒数第二块在 EAGLE drop 后驱逐"等 8 种部分缓存态的期望命中长度固化为断言，是本 Case L1 复现的 oracle。
- 同族问题（命中率坍塌家族）：
  - 案例 04（vllm-ascend 上 DSv4 系列 0% 命中）——机制路径不同（驱逐单存条目破坏），现象同型；
  - #42948：DSv4-Flash 复杂 hybrid（5 KV group）下幸存的单存条目驱逐变体，其 issue 明言 "#33524 special-cased the simple 1-Full+1-SWA topology" 之后复杂 hybrid 仍需 #40860 时代的修复；
  - 复杂 hybrid 的收敛正确性后来由 #40860（`eagle_verified`）与 #48425（per-group prefix-hit divergence）继续修补。
- EAGLE lookahead 与 SWA 交互的专门修复：#44082（`e9e08c49b9`）。

---

## 七、实测记录（2026-09-15，本机 A/B）

### 7.1 环境

| 项 | 值 |
|---|---|
| 机器 | macOS 26.6.2, arm64 |
| Python | 3.12.3（Framework），系统站点包 |
| torch | 2.7.1（CPU wheel；**2.5.1 不行**——缺 `torch._inductor.custom_graph_pass`） |
| 测试栈 | pytest 9.1.1 + `a01ef3fa51:requirements/common.txt` 依赖集（msgspec/pydantic/transformers/cbor2 等） |
| 仓库 | `/Users/wushanglun/Desktop/vllmgch/vllm`（main 含上游全历史） |
| worktree | `/tmp/vllm_bug`（7320ca3942）、`/tmp/vllm_fix`（a01ef3fa51）；脚本版产物 `/tmp/kvc_case08/{bug,fix}` |

### 7.2 时间线与结果

| # | 动作 | 结果 |
|---|---|---|
| 1 | 手工 worktree A/B：fix 态跑 `test_prefill_hybrid_model_eagle` + `combinations_eagle` | **2 passed** |
| 2 | bug 态同步修复版测试文件后复跑 | **2 failed**（`([ ], [], []) == ([1,2,3,4],[0,9,10,11],[0,16,17,18])` 断言失败）⇒ **REPRO** |
| 3 | 编写 `scripts/run_hybrid_eagle_ab_test.sh` 并端到端复跑（独立 worktree） | REPRO 判定矩阵输出正常（摆平 bash 语法/路径两处 bug 后） |
| 4 | 验证 v0.9.1 / v0.9.2 结构 | 两步直调、无 `while` 收敛循环——确认 NPU 常备栈不在 bug 窗口（§2.5） |

### 7.3 结论

1. **可以复现，确定性且零 GPU 成本**：同一测试在 bug 提交失败、修复提交通过，中间只有 PR #33524 的 coordinator 改动——因果链唯一，无环境噪声。
2. 复现的最小充分条件 = 本仓库 git 历史 + CPU Python 环境；服务级（GPT-OSS + Eagle3）现象由上游报告背书，本 Case 不重复投入（资源/版本窗口不匹配 NPU 常备栈）。
3. 该 Case 的持续价值：`run_hybrid_eagle_ab_test.sh` 可作为**协调器回归基线**——任何对 `kv_cache_coordinator.py` / `single_type_kv_cache_manager.py` 的魔改或后端移植后跑一遍，若 bug 态不再 FAIL 说明历史被改写，若 fix 态不再 PASS 说明协调器逻辑被改坏。
4. 旧版本文档防误读：#33524 **不是**"把 EAGLE 逻辑从各 manager 移除"（那是未采用的 Option 1 原案），修的是协调器收敛循环对 SWA 的重复调用与 full 截断时机；manager 端的 drop/threshold 逻辑全部保留并经 #44082 继续演进。
