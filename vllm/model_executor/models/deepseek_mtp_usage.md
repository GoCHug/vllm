# DeepSeek-V3.2 MTP 在 vLLM 中的运行机制

> **本文档以 DeepSeek-V3.2 + MTP=1 和 DeepSeek-V3.2 + MTP=2 为例，结合 vLLM 源码，先讲解MTP=1的情况，再讲解MTP=2的情况，由浅入深讲解 MTP 技术。**
>
> **适合读者：对 LLM 推理有基本了解，想深入理解 MTP 原理和 vLLM 实现的朋友。** 

# 第一部分：背景篇 —— 为什么需要 MTP？

## 1.1 传统生成有多慢？看一个例子

用户问：**"一句话介绍北京"**，模型要回答 **"北京是中国的首都，历史悠久"**。这句话被拆成 8 个 token 逐步生成。每生成一个 token，61 层 Transformer 就要全部算一遍。

```
传统自回归的架构——每次只能产 1 个 token：

┌───────────────────────────────────────┐
│            DeepSeek-V3.2              │
│  ┌──────┐┌──────┐       ┌──────┐    │
│  │Layer0││Layer1│  ...  │Layer60│    │  ← 61 层全跑
│  └──┬───┘└──┬───┘       └──┬───┘    │
│     └───────┴──────────────┘         │
│                 ↓                     │
│          采样出 1 个 token             │
│          "北京"或"是"或"中国"...       │
└─────────────────┬─────────────────────┘
                  │
      下次生成下一个 token 时，
      整个模型再跑一遍（共 8 次循环）
```

具体过程：

```
用户："一句话介绍北京"
  └→ [61 层] → "北京"           ← 前向 1

用户："一句话介绍北京" + "北京"
  └→ [61 层] → "是"             ← 前向 2

... + "是"
  └→ [61 层] → "中国"           ← 前向 3

... + "中国"
  └→ [61 层] → "的"             ← 前向 4

... + "的"
  └→ [61 层] → "首都"           ← 前向 5

... + "首都"
  └→ [61 层] → "，"             ← 前向 6

... + "，"
  └→ [61 层] → "历史"           ← 前向 7

... + "历史"
  └→ [61 层] → "悠久"           ← 前向 8

  输出 8 个 token = 8 × 61 = 488 层计算
```

每多一个 token，就要再跑一遍全模型。这就是 LLM 生成慢的根本原因——**"一次只能生一个，生一个就要全模型跑一次"**。

## 1.2 投机解码：引入一个小模型来"猜"

能不能让大模型少跑几次？投机解码的思路是：**找一个便宜的小模型（Draft），先快速猜出几个可能的后续 token，然后大模型（Target）一次性验证它们**。

还是上面的聊天场景，额外加一个只有 12 层的 Draft 小模型。猜 2 个 draft（`num_speculative_tokens=2`）。

在 vLLM 中，整个推理过程分为 **Prefill（预填充）** 和 **Decode（解码）** 两个阶段。Prefill 处理完整的 prompt，生成第一个 token；Decode 在后续循环中进行"验证+解码合并"。

### Prefill 阶段架构（第 1 次 `execute_model`）

处理 prompt、产出第一个 token、生成第一批 draft。

```
═══════════════════════════════════════════
  投机解码 —— Prefill 阶段
═══════════════════════════════════════════

  源码路径:
    Target 前向: model_runner.py → execute_model() → self.model()
    Target 采样: model_runner.py → sample_tokens() → self.sample()
    Draft 生成:   model_runner.py → sample_tokens() → speculator.propose()
                  └→ speculator.py → propose() → prefill()

  ┌──────────────────────────────────────────────┐
  │ Target 主模型 (61层, 独立权重)                 │
  │                                              │
  │  输入: "一句话介绍北京"                         │
  │        (全部 prompt tokens, 一次性并行处理)     │
  │  ┌──┐┌──┐┌──┐         ┌──┐                  │
  │  │L0││L1││L2│  ...    │L60│                  │
  │  └──┘└──┘└──┘         └──┘                  │
  │       全部 prompt tokens 并行通过 61 层       │
  │       每层为每个 position 写入 KV cache        │
  │       → 每层写入与 prompt 长度相同的 KV entries │
  │                                              │
  │  输出① sampled token → "北京"                 │
  │        (取最后 prompt 位置的 logits 采样)       │
  │  输出② hidden_states                          │
  │        (所有 prompt 位置的隐藏态)               │
  └──────────┬───────────────┬───────────────────┘
             │               │
             │ samled_token   │ hidden_states
             │ ="北京"        │ (取最后位置)
             ▼               ▼
  ┌──────────────────────────────────────────────┐
  │ Draft 小模型 (12层, 独立权重)                   │
  │ ┌──┐┌──┐         ┌──┐                       │
  │ │L0││L1│  ...    │L11│                       │
  │ └──┘└──┘         └──┘                       │
  │                                              │
  │  第1遍前向:                                    │
  │    输入① embedding("北京") ← 主模型采样的 token │
  │    输入② hidden_states    ← 主模型最后层的隐藏态 │
  │         → 猜 draft_1 = "是"                    │
  │                                              │
  │  第2遍前向:                                    │
  │    输入① embedding("是")   ← 第1遍猜出的 token │
  │    输入② hidden_states_1  ← 第1遍输出的隐藏态  │
  │         → 猜 draft_2 = "中国"                  │
  └──────────────────────────────────────────────┘

  本轮 Target 前向次数: 1  |  Draft 前向次数: 2
  本次输出: "北京" (1 token)
```

> **Prefill 的特点**：Target 一次性处理全部 prompt tokens（不是逐 token），为所有 prompt 位置计算 KV cache。Draft 模型用 Target 的 `last_sampled_token` 作为起始输入、`hidden_states` 作为上下文来猜测后续。

### Decode 阶段架构（第 2 次起 `execute_model`）

**验证+解码合并**：上一次的 draft tokens 拼接在输入后面，一趟 Target 前向同时完成验证和采样 bonus。

```
═══════════════════════════════════════════
  投机解码 —— Decode 阶段（验证+解码 合并）
═══════════════════════════════════════════

  源码路径:
    输入拼接:      input_batch.py → combine_sampled_and_draft_tokens()
                      将 last_sampled + draft_tokens 写入 input_ids
    Target 前向:   model_runner.py → execute_model() → self.model()
                      一趟前向同时计算所有位置 logits
    拒绝采样验证:   rejection_sampler.py → accept_prob = min(1, P_target/P_draft)
    Bonus 采样:    model_runner.py → sample_tokens() → self.sample()
    Draft 生成:    model_runner.py → sample_tokens() → speculator.propose()
                      生成下一批 draft tokens

  以第 2 次 Target 前向为例:

  combine_sampled_and_draft_tokens()
  input_ids = [..., "北京"(last_sampled), "是"(draft_1), "中国"(draft_2)]
                │                         └──────┬──────┘
                │                         上一次 Draft 的猜测
                └── 上一次 Target 的采样结果

  ┌──────────────────────────────────────────────┐
  │ Target 主模型 (61层, 一趟前向)                  │
  │                                              │
  │  输入: ["北京", "是", "中国"]                   │
  │        (3 tokens = 1 last_sampled + 2 draft) │
  │        combine_sampled_and_draft_tokens() 拼接 │
  │  ┌──┐┌──┐         ┌──┐                       │
  │  │L0││L1│  ...    │L60│                       │
  │  └──┘└──┘         └──┘                       │
  │       3 tokens 并行通过 61 层                  │
  │       → 写入 3×61 条 KV cache entries          │
  │                                              │
  │  内部处理 (每层输出 3 个位置的 hidden_states):   │
  │  位置1 ("北京") → 不重采样 (已在上轮输出)        │
  │  位置2 ("是")   → P_target vs P_draft → ✓ 接受  │  ← 拒绝采样
  │  位置3 ("中国") → P_target vs P_draft → ✓ 接受  │  ← 拒绝采样
  │  ── 以上全部接受 → 额外采样 bonus ──              │
  │  位置4 (bonus)  → 采样 → "的"                  │  ← bonus 解码
  │                                              │
  │  输出① bonus_token → "的"                      │
  │        (从 bonus 位置 logits 采样)              │
  │  输出② hidden_states → [3, hidden_dim]        │
  │        (3 个新位置的隐藏态)                      │
  └──────────┬───────────────┬───────────────────┘
             │               │
             │ bonus_token   │ hidden_states
             │ ="的"         │ (取 bonus 位置)
             ▼               ▼
  ┌──────────────────────────────────────────────┐
  │ Draft 小模型 (12层)                            │
  │                                              │
  │  第1遍前向:                                    │
  │    输入① embedding("的")   ← bonus token      │
  │    输入② hidden_states    ← 主模型 bonus 位隐藏态│
  │         → 猜 draft_1 = "首都"                  │
  │                                              │
  │  第2遍前向:                                    │
  │    输入① embedding("首都") ← 第1遍猜出的 token │
  │    输入② hidden_states_1  ← 第1遍输出的隐藏态  │
  │         → 猜 draft_2 = "，"                   │
  └──────────────────────────────────────────────┘

  本轮 Target 前向次数: 1  |  Draft 前向次数: 2
  本次输出: "是", "中国", "的" (3 tokens)

  ... 第 3 次, 第 4 次 Target 前向同样模式循环 ...
```

### 完整流程对比（8 个 token 输出）

```
Prefill (第 1 次 execute_model):
  [Target 61 层] → "北京"
  [Draft 12 层]  → 猜 "是", "中国"
  输出 1 token

Decode (第 2 次 execute_model, 验证+解码 合并):
  喂入: "北京" + ["是", "中国"]
  [Target 61 层] → 验证 "是"✓, "中国"✓, bonus→"的"
  [Draft 12 层]  → 猜 "首都", "，"
  输出 3 tokens

Decode (第 3 次 execute_model, 验证+解码 合并):
  喂入: "的" + ["首都", "，"]
  [Target 61 层] → 验证 "首都"✓, "，"✓, bonus→"历史"
  [Draft 12 层]  → 猜 "悠久", ...
  输出 3 tokens

Decode (第 4 次 execute_model):
  喂入: "历史" + ["悠久", ...]
  [Target 61 层] → 验证 "悠久"✓, bonus→EOS
  输出 1 token

  总计: 4×61(Target) + 3×12(Draft) = 280 层
  对比传统 488 层: 节省 43%

  两个模型文件: draft_model.safetensors + model.safetensors (671GB)
  两份显存, 两份维护成本
```

> **Prefill vs Decode 总结**：Prefill 处理全部 prompt，Target 只做正常解码 + 输出 hidden_states 给 Draft。Decode 每次 Target 前向的输入 = 1 个 last_sampled + N 个 draft，"验证"和"下一轮解码"在同一趟 Target 前向中完成，额外多产一个 bonus token。

## 1.3 DeepSeek MTP：把"猜"的能力直接嵌进主模型

DeepSeek 的做法更巧妙：既然要猜，**何必搞一个独立模型？在主模型自带的权重文件里塞一层轻量的 MTP 层**（Layer 61），专门负责"猜"。同样分为 Prefill 和 Decode 两个阶段。

### Prefill 阶段架构（第 1 次 `execute_model`）

```
═══════════════════════════════════════════
  DeepSeek MTP —— Prefill 阶段
═══════════════════════════════════════════

  源码路径:
    Target 前向: model_runner.py → execute_model() → self.model()
                  └→ deepseek_v2.py → DeepseekV2ForCausalLM.forward()
                 Layer 0~60 前向, 输出 hidden_states
    采样:        model_runner.py → sample_tokens() → self.sample()
    MTP 生成:    model_runner.py → sample_tokens() → speculator.propose()
                  └→ speculator.py → propose() → prefill()
                      └→ DeepSeekMTP.forward() → Layer 61 (第1遍)
                  └→ ... → multi_step_decode()
                      └→ DeepSeekMTP.forward() → Layer 61 (第2遍)

  ┌──────────────────────────────────────────────────┐
  │            一份 model.safetensors 权重文件          │  ← 只需一个文件
  │                                                  │
  │  ┌────────────────────────────────────────────┐  │
  │  │ Target 主模型 (Layer 0~60)                  │  │
  │  │                                            │  │
  │  │  输入: "一句话介绍北京"                       │  │
  │  │        (全部 prompt tokens, 一次性并行处理)   │  │
  │  │  ┌──┐┌──┐┌──┐         ┌──┐                │  │
  │  │  │L0││L1││L2│  ...    │L60│                │  │
  │  │  └──┘└──┘└──┘         └──┘                │  │
  │  │       全部 prompt tokens 并行通过 60 层     │  │
  │  │       每层为每个 position 写入 KV cache      │  │
  │  │       → 每层写入与 prompt 长度相同的 KV entries│  │
  │  │                                            │  │
  │  │  输出① sampled token → "北京"               │  │
  │  │        (取最后 prompt 位置的 logits 采样)     │  │
  │  │  输出② hidden_states                        │  │
  │  │        (所有 prompt 位置的隐藏态)             │  │
  │  └──────────┬───────────────┬─────────────────┘  │
  │             │               │                    │
  │             │ samled_token  │ hidden_states      │
  │             │ ="北京"       │ (取最后位置)        │
  │             ▼               ▼                    │
  │  ┌────────────────────────────────────────────┐  │
  │  │ MTP Layer 61 (同一层跑 2 遍, 只有 1 份权重) │  │
  │  │                                            │  │
  │  │  第1遍前向:                                  │  │
  │  │    输入① embedding("北京") ← 主模型采样的 token│  │
  │  │    输入② hidden_states    ← 主模型最后层的隐藏态│  │
  │  │         → 猜 draft_1 = "是"                  │  │
  │  │         → 输出 hidden_states_1               │  │
  │  │                                            │  │
  │  │  第2遍前向 (同一层权重, 不同 position):       │  │
  │  │    输入① embedding("是")  ← 第1遍猜出的 token│  │
  │  │    输入② hidden_states_1 ← 第1遍输出的隐藏态 │  │
  │  │         → 猜 draft_2 = "中国"                │  │
  │  └────────────────────────────────────────────┘  │
  │                                                  │
  │  Target 和 MTP 共享 embedding 和 lm_head           │
  └──────────────────────────────────────────────────┘

  本轮 Target 前向次数: 1  |  MTP 前向次数: 2
  本次输出: "北京" (1 token)
```

### Decode 阶段架构（第 2 次起 `execute_model`）

```
═══════════════════════════════════════════
  DeepSeek MTP —— Decode 阶段（验证+解码 合并）
═══════════════════════════════════════════

  源码路径:
    输入拼接:      input_batch.py → combine_sampled_and_draft_tokens()
    Target 前向:   model_runner.py → execute_model() → self.model()
                    一趟前向验证 draft + 采样 bonus + 输出 hidden_states
    拒绝采样验证:   rejection_sampler.py
    MTP 新 draft:  model_runner.py → sample_tokens() → speculator.propose()

  以第 2 次 Target 前向为例:

  combine_sampled_and_draft_tokens()
  input_ids = [..., "北京", "是", "中国"]
                │       └──┬──┘
                │     上一次 MTP 的猜测
                └── 上一次 Target 的采样

  ┌──────────────────────────────────────────────────┐
  │          同一份 model.safetensors 权重文件          │
  │                                                  │
  │  ┌────────────────────────────────────────────┐  │
  │  │ Target 主模型 (Layer 0~60, 一趟前向)         │  │
  │  │                                            │  │
  │  │  输入: ["北京", "是", "中国"]                 │  │
  │  │        (3 tokens = 1 last_sampled + 2 draft)│  │
  │  │        combine_sampled_and_draft_tokens() 拼接│  │
  │  │  ┌──┐┌──┐         ┌──┐                     │  │
  │  │  │L0││L1│  ...    │L60│                     │  │
  │  │  └──┘└──┘         └──┘                     │  │
  │  │       3 tokens 并行通过 60 层                │  │
  │  │       → 写入 3×60 条 KV cache entries        │  │
  │  │                                            │  │
  │  │  内部处理 (每层输出 3 个位置的 hidden_states): │  │
  │  │  位置1 ("北京") → 不重采样 (已在上轮输出)      │  │
  │  │  位置2 ("是")   → 拒绝采样 → ✓ 接受          │  │  ← 验证
  │  │  位置3 ("中国") → 拒绝采样 → ✓ 接受          │  │  ← 验证
  │  │  ── 以上全部接受 → 额外采样 bonus ──            │  │
  │  │  位置4 (bonus)  → 采样 → "的"               │  │  ← 解码
  │  │                                            │  │
  │  │  输出① bonus_token → "的"                    │  │
  │  │        (从 bonus 位置 logits 采样)            │  │
  │  │  输出② hidden_states → [3, hidden_dim]      │  │
  │  │        (3 个新位置的隐藏态, 给 MTP 用)        │  │
  │  └──────────┬───────────────┬─────────────────┘  │
  │             │               │                    │
  │             │ bonus_token   │ hidden_states      │
  │             │ ="的"         │ (取 bonus 位置)     │
  │             ▼               ▼                    │
  │  ┌────────────────────────────────────────────┐  │
  │  │ MTP Layer 61 (同一层跑 2 遍)                │  │
  │  │                                            │  │
  │  │  第1遍前向:                                  │  │
  │  │    输入① embedding("的")   ← bonus token    │  │
  │  │    输入② hidden_states    ← 主模型 bonus 位隐藏态│  │
  │  │         → 猜 draft_1 = "首都"                │  │
  │  │         → 输出 hidden_states_1               │  │
  │  │                                            │  │
  │  │  第2遍前向:                                  │  │
  │  │    输入① embedding("首都") ← 第1遍猜出的 token│  │
  │  │    输入② hidden_states_1  ← 第1遍输出的隐藏态│  │
  │  │         → 猜 draft_2 = "，"                 │  │
  │  └────────────────────────────────────────────┘  │
  │                                                  │
  │  Target 和 MTP 共享 embedding 和 lm_head           │
  └──────────────────────────────────────────────────┘

  本轮 Target 前向次数: 1  |  MTP 前向次数: 2
  本次输出: "是", "中国", "的" (3 tokens)

  ... 第 3 次, 第 4 次 Target 前向同样模式循环 ...
```

> **关键事实**：DeepSeek-V3.2 的权重文件中 MTP 权重只有 **1 层**（`num_nextn_predict_layers=1`）。`num_speculative_tokens=2` 时，**同一个 Layer 61 循环执行 2 遍**：第 1 遍猜 draft_1，第 2 遍拿第 1 遍的输出再猜 draft_2。

### 完整流程对比（8 个 token 输出）

```
Prefill (第 1 次 execute_model):
  [Target 61 层]     → "北京"
  [MTP L61 第1遍]     → 猜 "是"
  [MTP L61 第2遍]     → 猜 "中国"
  输出 1 token

Decode (第 2 次 execute_model, 验证+解码 合并):
  喂入: "北京" + ["是", "中国"]
  [Target 61 层]     → 验证 "是"✓, "中国"✓, bonus→"的"
  [MTP L61 第1遍]     → 猜 "首都"
  [MTP L61 第2遍]     → 猜 "，"
  输出 3 tokens

Decode (第 3 次 execute_model, 验证+解码 合并):
  喂入: "的" + ["首都", "，"]
  [Target 61 层]     → 验证 "首都"✓, "，"✓, bonus→"历史"
  [MTP L61 × 2遍]     → 猜 "悠久", ...
  输出 3 tokens

Decode (第 4 次 execute_model):
  喂入: "历史" + ["悠久", ...]
  [Target 61 层]     → 验证 "悠久"✓, bonus→EOS
  输出 1 token

  总计: 4×61(Target) + 3×2(MTP) = 250 层
  对比传统 488 层: 节省 49%
```

**三种方案横向对比**：

```
任务：回答 "一句话介绍北京"（8 个 token 输出）

                  Prefill        Decode × 3        总计          模型文件数
  传统自回归:      1×61=61       7×61=427         488 层         1 个
  投机解码:       1×61+2×12      3×61+6×12        280 层         2 个 ✗
  DeepSeek MTP:  1×61+2         3×61+6×2         250 层         1 个 ✓

  节省比例: 投机解码 43%, MTP 49%
```

> **规律**：Prefill 阶段 Target 处理全部 prompt 并生成第 1 个 token，Draft/MTP 基于 hidden_states 猜测后续。Decode 阶段 Target 每次前向验证上一轮的 draft + 采样 bonus——**验证和"下一轮解码"在同一趟 Target 前向中完成**。输出越长，Decode 循环越多，加速效果越明显。

MTP 的本质：把"猜"的能力作为模型的一部分一起训练出来，省掉独立 Draft 模型。

---

# 第二部分：概念篇 —— MTP 的核心概念

## 2.1 先理解一个关键概念：hidden_states 是什么？

在深入 MTP 之前，必须先理解 **hidden_states**。这是 MTP 整个机制运转的核心"信息载体"。

```
输入 "我 爱" → [主模型 61 层 Transformer]
                     │
                     │  每一层都是一个巨大的向量运算
                     │  比如: [4096, 7168] 维度的向量在层间传递
                     │
                     ▼
              最后一层的输出（hidden_states）
              
              这个向量编码了"输入文本的完整语义理解"
              它不是 token，而是 token 在高维语义空间中的位置
```

**通俗理解**：hidden_states 就是模型"思考过程中的脑电波"。主模型看完了 "我 爱" 两个 token，经过 61 层 Transformer 的处理，最后输出的 hidden_states 蕴含了丰富的语义信息："接下来应该接一个名词/形容词/动词..."

**为什么 MTP 需要 hidden_states？** 因为 hidden_states 比单个 token 包含更多上下文信息。如果 MTP 只用 token 来预测，信息太少了；结合 hidden_states，就相当于看到了主模型的"思考过程"，预测准确率大幅提升。

## 2.2 三个核心角色

| 角色 | 说明 | 对应 DeepSeek-V3.2 中的部分 |
|---|---|---|
| **Target Model（目标模型）** | 真正干活的主力模型，负责验证 draft token 是否正确 | DeepSeek-V3.2 的前 61 层 Transformer |
| **Draft Model（草稿模型）** | 快速猜测下一个 token 的模型 | MTP 层（Layer 61，只有 1 层） |
| **Rejection Sampler（拒绝采样器）** | 比较 Target 和 Draft 的概率，决定接受还是拒绝 | vLLM 的 `RejectionSampler` |

### 关键理解

- **Draft Model 不是独立模型**。MTP 层是加载在主模型权重中的，它们复用主模型的 embedding 层和整个 lm_head。
- **Checkpoint 中 MTP 权重只有 1 层**：`config.num_nextn_predict_layers = 1`。这意味着 DeepSeek-V3.2 只在 Layer 61 处有 MTP 相关的权重（enorm、hnorm、eh_proj、mtp_block、shared_head）。
- **`num_speculative_tokens` 决定运行时行为**：这个参数是 vLLM 启动时配置的，它告诉 vLLM "你要猜几个 draft token"。因为只有 1 层 MTP 权重，所以 `num_speculative_tokens=2` 时，**同一个 Layer 61 会被执行 2 遍**。第 1 遍基于主模型的 hidden_states 猜 draft_1；第 2 遍拿第 1 遍输出的 hidden_states 再猜 draft_2。
- **模运算实现循环**：代码中 `current_step_idx = spec_step_idx % num_mtp_layers`，因为 `num_mtp_layers=1`，`0%1=0`、`1%1=0`，无论 `spec_step_idx` 是多少都路由到 Layer 61。
- **MTP 层的编号**从主模型最后一层之后开始：主模型有 61 层（编号 0~60），MTP 层编号为 61。
- **重要：两次执行如何区分？** `EagleSpeculator.run_model()` 调用 `self.model(...)` 时并**没有传递** `spec_step_idx` 参数，因此该参数始终默认为 0。第 1 遍和第 2 遍执行的**是同一层、同一个路由**（`0%1=0 → Layer 61`）。两次的区别纯粹在于**输入数据不同**：第 1 遍的 `previous_hidden_states` 来自主模型，`input_ids` 是主模型刚采样的 token；第 2 遍的 `previous_hidden_states` 是 Layer 61 第 1 遍的输出，`input_ids` 是第 1 遍猜出的 draft_1。层路径完全不变，变的是喂进去的数据。
- **hidden_states 流向**：每遍 MTP 接收的是 **上一遍传来的 hidden_states**：
  - 第 1 遍收到的是 **主模型 Layer 60 输出的 hidden_states**（包含"刚刚生成的 token 是什么"的上下文信息）
  - 第 2 遍收到的是 **Layer 61 第 1 遍输出的 hidden_states**（包含"第 1 个 draft token 是什么"的信息）

### 2.2.1 hidden_states 的流动路径（一张图看懂）

```
  主模型 Layer 60 输出 hidden_states
              │
              │  传给 Layer 61 第1遍作为 previous_hidden_states
              ▼
  Layer 61 (第1遍): previous_hidden_states + next_token_embedding → 前向 → 输出 hidden_states_1
              │                                        输出 draft_token_1
              │  传给 Layer 61 第2遍作为 previous_hidden_states
              ▼
  Layer 61 (第2遍): previous_hidden_states + draft_token_1_embedding → 前向 → 输出 hidden_states_2
                                                         输出 draft_token_2
  
  同一层 MTP 的输入 = 【上一遍的 hidden_states】+ 【当前位置的 token 的 embedding】
```

> **为什么同一层跑两遍能猜出不同的 token？** 因为两次的输入不同。第 1 遍拿到的是主模型的 hidden_states（包含完整上下文的语义），输入 token 是主模型刚生成的；第 2 遍拿到的是第 1 遍自己的 hidden_states（包含"第 1 个 draft 是什么"的信息），输入 token 是第 1 遍刚猜的 draft_1。两次的前向计算完全相同（同一个权重矩阵），但输入变了，输出自然也变了。

## 2.3 核心流程（一句话版）

> **Target Model 前向（第 N 次）→ 生成 token + hidden_states → Layer 61（第1遍）基于 hidden_states 猜 draft_1 → Layer 61（第2遍）基于第1遍的 hidden_states 猜 draft_2 → 下次 Target 前向（第 N+1 次）将 draft 拼在输入里一起验证 + 采样 bonus → 接受的输出，拒绝的用 Target 重采样。**

## 2.4 接受率 —— MTP 效率的关键

验证算法公式（第六部分会详解）：

```
接受概率 = min(1, P_target(token) / P_draft(token))
```

**通俗理解**：主模型（Target）和 MTP 层（Draft）各自对"下一个 token 应该是什么"有自己的判断。如果 MTP 层非常确信某个 token（概率很高），而主模型也觉得合理（概率不低于 MTP），那就直接用 MTP 的猜测。如果 MTP 非常确信但主模型不太同意（概率相差较大），那就以一定概率拒绝。

**这个公式保证了**：无论你接受还是拒绝，最终输出的 token 分布**精确等同于主模型自回归生成的分布**——数学上已经证明。所以 MTP 不改变生成质量，只改变速度。

#### 影响接受率的因素

| 因素 | 影响方向 | 原因 |
|---|---|---|
| `num_speculative_tokens` 越大 | 接受率越低 | 猜得越多越容易出错 |
| 内容越有规律 | 接受率越高 | 规律性内容（如代码、表格）容易预测 |
| temperature 越低 | 接受率越高 | 低温度下模型更"确定"，Draft 和 Target 更容易一致 |

#### 实际接受率大概多少？

根据社区经验，DeepSeek-V3.2 的 MTP 接受率通常在 **80%~95%** 之间。也就是说 2 个 draft token 平均能接受约 **1.6~1.9 个**。加上 Bonus token（全部接受时的额外奖励），实际加速比约在 **1.5× ~ 2.0×** 范围内。

> **为什么 MTP 的接受率这么高？** 因为 MTP 层和主模型是联合训练的——主模型在训练时就知道后面有 MTP 层在"帮忙"，所以它会输出更适合 MTP 使用的 hidden_states。这是一种"配合默契"。

---

# 第三部分：架构篇 —— 代码结构总览

## 3.1 源码文件地图

> **核心 MTP 模型定义**（本文 第四部分 深入拆解）

```
vllm/model_executor/models/
├── deepseek_v2.py                  ← 主模型 (Layer 0~60 的 Transformer 层)
└── deepseek_mtp.py                 ← MTP 核心实现 (Layer 61 的定义和计算，仅1层)
     ├── DeepSeekMultiTokenPredictorLayer  ← 单层 MTP
     ├── DeepSeekMultiTokenPredictor       ← MTP 层管理器
     └── DeepSeekMTP                       ← 对外入口 + 权重加载
```

> **vLLM 推理引擎中的 MTP 调度与执行**（本文 第五、六部分 深入拆解）

```
vllm/v1/
├── config/speculative.py                    ← SpeculativeConfig: MTP 参数的解析
├── engine/
│   ├── core.py                              ← EngineCore: 引擎主循环，step() 中调度 MTP
│   └── async_llm.py                         ← AsyncLLM: API 层到 EngineCore 的桥梁
├── core/sched/scheduler.py                  ← Scheduler: 调度 MTP draft tokens 的验证
├── worker/gpu/
│   ├── model_runner.py                      ← GPUModelRunner: 执行 Target 前向 + 调用 MTP propose
│   └── spec_decode/
│       ├── __init__.py                      ← init_speculator(): 创建 EagleSpeculator
│       └── eagle/
│           ├── speculator.py                ← EagleSpeculator: MTP 的 propose() 和 generate_draft()
│           └── utils.py                     ← load_eagle_model(): MTP 模型创建 + 权重共享
├── sample/
│   └── rejection_sampler.py                 ← RejectionSampler: 拒绝采样验证 draft tokens
├── entrypoints/openai/
│   ├── chat_completion/
│   │   ├── api_router.py                    ← FastAPI 路由: POST /v1/chat/completions
│   │   └── serving.py                       ← OpenAIServingChat: 请求处理 + 响应组装
│   └── serve/__main__.py                    ← vllm serve 命令行入口
```

> **补充说明**：vLLM v1 中还存在旧版投机解码代码（`vllm/v1/spec_decode/llm_base_proposer.py` 等），但这些代码在 v1 新架构中已被 `vllm/v1/worker/gpu/spec_decode/eagle/` 中的实现取代。本文以新架构为准。

## 3.2 类层次结构

在 [deepseek_mtp.py](file:///c:/study/vllm_vllmascend/vllm/vllm/model_executor/models/deepseek_mtp.py) 中定义了 3 个核心类（由底向上）：

```
DeepSeekMultiTokenPredictorLayer     ← 单层 MTP 的计算单元
    │
    ├── enorm:  归一化当前 token 的 embedding
    ├── hnorm:  归一化从上一层传来的 hidden_states
    ├── eh_proj: 融合 embedding + hidden_states (2H → H)
    ├── mtp_block: 一层标准 Transformer Decoder Layer
    └── shared_head: 输出层 (RMSNorm + ParallelLMHead)
    
DeepSeekMultiTokenPredictor          ← MTP 层的管理器
    │
    ├── self.layers: ModuleDict {"61": Layer}（仅1层）
    ├── self.embed_tokens: Embedding 层（复用主模型权重）
    └── self.logits_processor: 将 hidden_states 转为 logits

DeepSeekMTP                          ← 对外入口，同时用于 load_weights
    │
    └── self.model = DeepSeekMultiTokenPredictor(...)
```

### 3.2.1 类关系图

```
                    vLLM v1 投机解码框架
                    ═════════════════════
                    
     EagleSpeculator                RejectionSampler
     (v1/worker/gpu/spec_decode/    (v1/sample/rejection_sampler.py)
      eagle/speculator.py)
            │                              ▲
            │ self.model(...)              │ forward()
            │ 调用 DeepSeekMTP             │ 验证 draft token
            ▼                              │ 决定接受/拒绝
     ┌──────────────┐             ┌──────────────────┐
     │  DeepSeekMTP  │             │  对每个 draft:    │
     │  (主入口)      │             │  接受或修正重采样  │
     └──────┬───────┘             └──────────────────┘
            │
            │ self.model
            ▼
     ┌──────────────────────┐
     │ DeepSeekMultiToken   │
     │ Predictor (管理层)    │
     │ ┌──────────────────┐ │
     │ │ Layer 61（仅1层）  │ │  num_speculative_tokens=2 时
     │ │                   │ │  循环执行2遍:
     │ │ 第1遍: spec_step=0 │ │  收到主模型 hidden_states
     │ │ 第2遍: spec_step=1 │ │  收到第1遍的 hidden_states
     │ │    (模运算路由到   │ │  spec_step%1=0，始终走Layer61)
     │ │     同一层)        │ │
     │ └──────────────────┘ │
     └──────┬───────────────┘
            │ 每层包含（见第四部分详解）
            ▼
     ┌────────────────────────────┐
     │ DeepSeekMultiToken          │
     │ PredictorLayer (计算层)      │
     │ ┌────────────────────────┐  │
     │ │ ① enorm:  归一化 token  │  │
     │ │    embedding            │  │
     │ │ ② hnorm:  归一化 hidden │  │
     │ │    _states              │  │
     │ │ ③ eh_proj: 融合两者     │  │
     │ │    [B,2H] → [B,H]      │  │
     │ ├────────────────────────┤  │
     │ │ ④ mtp_block:            │  │
     │ │    Transformer 层       │  │
     │ ├────────────────────────┤  │
     │ │ ⑤ shared_head:          │  │
     │ │    输出头 (共享权重)     │  │
     │ └────────────────────────┘  │
     └─────────────────────────────┘
```

---

# 第四部分：源码篇 —— 核心类逐层拆解

## 4.1 DeepSeekMultiTokenPredictorLayer —— 单层 MTP 计算

源码位置：[deepseek_mtp.py:L56-L111](file:///c:/study/vllm_vllmascend/vllm/vllm/model_executor/models/deepseek_mtp.py#L56-L111)

这是 MTP 最基本的计算单元。每一层 MTP 包含以下组件：

```python
class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        # ① 两个 RMSNorm: 分别归一化 embedding 和 hidden_states
        self.enorm = RMSNorm(hidden_size)     # embedding normalization
        self.hnorm = RMSNorm(hidden_size)     # hidden_state normalization
        
        # ② 融合层: 将 embedding 和 hidden_states 拼接后降维
        #    input:  (H + H) = 2H
        #    output: H
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # ③ 一层标准的 Transformer Decoder Layer (复用 deepseek_v2 的实现)
        self.mtp_block = DeepseekV2DecoderLayer(vllm_config, ...)
        
        # ④ 输出头: RMSNorm + ParallelLMHead (与主模型共享权重)
        self.shared_head = SharedHead(config)
```

### 4.1.1 forward 计算步骤

> **prefill vs decode**：MTP Layer 的 forward 在处理两种场景时，`input_ids` 的 shape 不同：
> - **prefill**（第 1 遍）：`input_ids` 是**完整的 shifted 序列**（例如 `[T1,T2,T3,T4,T5,T6]`，共 `query_len` 个 token），所有位置并行计算，但只有最后位置（`last_token_indices`）的 `hidden_states` 被用于预测 draft token。
> - **decode**（第 2 遍及以后）：`input_ids` 只有 **1 个 token**（上一轮猜出的 draft token），每个 request 只跑 1 个 position。
>
> 下面以 decode 阶段的单个 token 为例讲解 forward 流程。

```python
def forward(
    self,
    input_ids: torch.Tensor,          # 当前位置的 token id，shape: [num_tokens]
    positions: torch.Tensor,          # 每个 token 的位置编号，shape: [num_tokens]
    previous_hidden_states: torch.Tensor,  # 上一层传来的 hidden_states
                                           # shape: [num_tokens, hidden_size]
    inputs_embeds: torch.Tensor | None = None,  # 预计算的 token embedding（可选）
    spec_step_index: int = 0,         # 当前是第几个 MTP 层（DeepSeek-V3.2 中始终为 0）
):
    # Step 1: 屏蔽 position=0 的输入
    #   为什么这样做？在 prefill 阶段，MTP 的输入是 target input 左移一位后的序列，
    #   例如 target 输入 [T0,T1,T2...] → MTP 输入 [T1,T2,T3...]
    #   position=0 的原 T0 已经不存在了，此时 position=0 的位置是 padding，
    #   需要将其 embedding 置零，防止干扰计算
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    
    # Step 2: 分别归一化
    #   enorm: 归一化当前 token 的 embedding（让 token 信息更稳定）
    #   hnorm: 归一化上层传来的 hidden_states（让语义信息更稳定）
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = self.hnorm(previous_hidden_states)
    
    # Step 3: 融合 —— 在最后一维拼接后线性投影
    #   concat: inputs_embeds [B, H] + previous_hidden_states [B, H] = [B, 2H]
    #   eh_proj: [B, 2H] → [B, H]，把"当前位置"和"历史语义"融合成一个向量
    hidden_states = self.eh_proj(
        torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
    )
    
    # Step 4: 经过一层标准 Transformer Decoder Layer
    #   mtp_block 内部包含 self-attention + MLP（可能带 MoE）
    #   它会对融合后的 hidden_states 做语义加工
    hidden_states, residual = self.mtp_block(
        positions=positions, hidden_states=hidden_states, residual=None
    )
    hidden_states = residual + hidden_states  # 残差连接，防止梯度消失
    
    return hidden_states  # shape: [num_tokens, hidden_size]
```

### 4.1.2 融合操作图解

```
                             eh_proj
  inputs_embeds [B, H] ─────┐
                             ├─→ concat [B, 2H] ─→ Linear(2H → H) ─→ [B, H]
  previous_hidden [B, H] ───┘
  
  "当前 token 说了什么" + "上一层在想什么" → 融合后的表示
```

核心直觉：**MTP 层同时看到了"当前要预测的 token"（通过 embedding）和"主模型在生成上个 token 时的思考过程"（通过 hidden_states），从而能做出更好的预测。**

### 4.1.3 SharedHead

```python
class SharedHead(nn.Module):
    def __init__(self, config, prefix, quant_config=None):
        self.norm = RMSNorm(hidden_size)
        self.head = ParallelLMHead(vocab_size, hidden_size)
    
    def forward(self, hidden_states):
        return self.norm(hidden_states)  # 只做归一化，logits 计算在外部
```

> `SharedHead` 的 `head`（ParallelLMHead，词表投影矩阵）与主模型的 `lm_head` 共享同一份权重——这保证了 MTP 的预测与主模型的输出在同一个语义空间中。但 `SharedHead` 中的 `norm`（RMSNorm，归一化层）是 MTP 层独有的权重，不共享。

---

## 4.2 DeepSeekMultiTokenPredictor —— MTP 层管理器

源码位置：[deepseek_mtp.py:L114-L172](file:///c:/study/vllm_vllmascend/vllm/vllm/model_executor/models/deepseek_mtp.py#L114-L172)

```python
class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        # 主模型有 61 层 (0~60)，MTP 从第 61 层开始
        self.mtp_start_layer_idx = config.num_hidden_layers    # = 61
        self.num_mtp_layers = config.num_nextn_predict_layers  # = 1 (checkpoint中仅1层!)
        
        # 创建 MTP 层字典: {"61": Layer}（仅1层）
        self.layers = torch.nn.ModuleDict({
            str(idx): DeepSeekMultiTokenPredictorLayer(vllm_config, ...)
            for idx in range(
                self.mtp_start_layer_idx,
                self.mtp_start_layer_idx + self.num_mtp_layers,  # 只循环一次
            )
        })
        # Embedding 层: 将 token id 转为向量
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        # Logits 处理器: 将 hidden_states 转为概率分布
        self.logits_processor = LogitsProcessor(vocab_size)
```

### 4.2.1 forward: MTP 层的路由逻辑

```python
def forward(self, input_ids, positions, previous_hidden_states,
            inputs_embeds=None, spec_step_idx=0):
    # 如果没有预计算的 embedding，就用 embed_tokens 从 token id 转换
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)  # token id → 向量 [H]

    # 关键路由逻辑：spec_step_idx 决定用哪个 MTP 层
    #   因为 num_mtp_layers=1，spec_step_idx%1 始终为 0，总是路由到 Layer 61
    #   注意：EagleSpeculator.run_model() 不传 spec_step_idx，所以始终为默认值 0
    #   两次执行用同一层，区别在于传入的 previous_hidden_states 和 input_ids 不同
    #   num_speculative_tokens 更大时继续循环复用 Layer 61
    current_step_idx = spec_step_idx % self.num_mtp_layers
    return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
        input_ids, positions, previous_hidden_states,
        inputs_embeds, current_step_idx,
    )
```

> **疑问**：为什么 `spec_step_idx` 设计为外部传入的参数？这是为了支持多 MTP 层的通用架构——如果某个模型有 2 层 MTP（`num_nextn_predict_layers=2`），`spec_step_idx` 就能区分路由到 Layer 61 还是 Layer 62。但对于 DeepSeek-V3.2（只有 1 层 MTP），`EagleSpeculator.run_model()` 调用 `self.model(...)` 时并**不传递** `spec_step_idx`，因此该参数始终为默认值 0，路由永远是 `0%1=0 → Layer 61`。详见第六部分 6.7。所以第 1 遍和第 2 遍的区别不在路由，而在于输入数据不同（`previous_hidden_states` 和 `input_ids`）。

### 4.2.2 compute_logits: 将 hidden_states 转为 token 概率

```python
def compute_logits(self, hidden_states, spec_step_idx=0):
    # 拿到当前 MTP 层
    current_step_idx = spec_step_idx % self.num_mtp_layers
    mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
    
    # 两步走：
    # ① shared_head(hidden_states) → RMSNorm 归一化
    # ② logits_processor(shared_head.head, 归一化结果) → 投影到词表大小 → logits
    logits = self.logits_processor(
        mtp_layer.shared_head.head,        # ← 这个 head 和主模型的 lm_head 是同一个对象
        mtp_layer.shared_head(hidden_states)  # ← 先做 RMSNorm
    )
    return logits  # shape: [num_tokens, vocab_size]
```


## 4.3 DeepSeekMTP —— 对外入口

源码位置：[deepseek_mtp.py:L175-L244](file:///c:/study/vllm_vllmascend/vllm/vllm/model_executor/models/deepseek_mtp.py#L175-L244)

```python
class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config, prefix=""):
        self.model = DeepSeekMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()  # 初始化 MoE 相关参数

    def forward(self, input_ids, positions, hidden_states,
                intermediate_tensors=None, inputs_embeds=None,
                spec_step_idx=0):
        # 直接委托给 self.model (DeepSeekMultiTokenPredictor)
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return self.model.compute_logits(hidden_states, spec_step_idx)
```


`DeepSeekMTP` 还承担了 **权重加载** 的重任（`load_weights` 方法）。它的关键功能包括：

- **权重复写**：将 checkpoint 中的 `model.layers.{spec_layer}.*` 映射到正确的参数名（如添加 `mtp_block` 前缀）
- **共享权重去重**：MTP 层共享主模型的 `embed_tokens`，只加载第一份权重
- **加载验证**：确保每个 MTP 层的权重都被正确加载

---

# 第五部分：流程篇 —— vLLM 启动 DeepSeek 服务的 MTP 流程

> **本节目标**：理解从敲下 `vllm serve` 命令到服务就绪的全过程中，MTP 是如何被一步步初始化出来的。
## 5.1 启动流程总览

═══════════════════════════════════════════════════════════════════════════
                   vLLM 启动时 MTP 相关的完整初始化链路
═══════════════════════════════════════════════════════════════════════════
```bash
  vllm serve deepseek-ai/DeepSeek-V3.2 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
  ───→ SpeculativeConfig 解析     (vllm\vllm\config\speculative.py)
       │  ├─ method = "mtp"
       │  ├─ num_speculative_tokens = 2
       │  └─ model = same as target (MTP无需额外模型)
       │
  ───→ VllmConfig 组装             (vllm\vllm\config\vllm.py)
       │  └─ speculative_config = SpeculativeConfig(...)
       │
  ───→ EngineCore.__init__()       (vllm\vllm\v1\engine\core.py)
       │  └─ self.use_spec_decode = vllm_config.speculative_config is not None
       │
  ───→ GPUModelRunner.__init__()   (vllm\vllm\v1\worker\gpu\model_runner.py)
       │  └─ self.speculator = init_speculator(...) → EagleSpeculator
       │
  ───→ GPUModelRunner.load_model() (vllm\vllm\v1\worker\gpu\model_runner.py)
       │  └─ self.speculator.load_model(self.model)
       │     └─ load_eagle_model() → 创建 MTP 模型 + 共享权重
       │        ├─ 加载 DeepSeekMTP 模型（仅 Layer 61，1层）
       │        ├─ embed_tokens 共享（主模型 → MTP）
       │        ├─ lm_head 共享（主模型 → MTP）
       │        └─ shared_head.head 修正（指向共享 head）
       │
  ───→ EagleSpeculator 捕获 CUDA Graph   (speculator.py)
       │  └─ capture_decode_cudagraphs() / capture_prefill_cudagraphs()
       └── 启动完成，等待请求
```

下面逐一展开每个步骤，给出对应的源码位置和关键逻辑。

## 5.2 Step 1: 命令行参数 → SpeculativeConfig

### 5.2.1 入口：命令行解析

启动 vLLM 时，用户通过 `--speculative-config` 传入 MTP 配置：

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

整个链路分 **5 个环节**，从命令行字符串到最终 `VllmConfig` 中的 `SpeculativeConfig` 对象：

```
  vllm serve deepseek-ai/DeepSeek-V3.2 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
                                      │
    ① argparse 解析                   │  --speculative-config → dict
       entrypoints/cli/main.py        │  FlexibleArgumentParser
                                      │
    ② EngineArgs 组装                 │  from_cli_args(args)
       engine/arg_utils.py            │  speculative_config = {"method":"mtp", ...}
                                      │
    ③ 自动探测 speculators_config     │  maybe_override_with_speculators()
       transformers_utils/config.py   │  解析 HuggingFace config.json
                                      │
    ④ SpeculativeConfig 构造          │  create_speculative_config()
       engine/arg_utils.py            │  合并 --spec-method/--spec-model/--spec-tokens
       config/speculative.py          │  → SpeculativeConfig(**dict)
                                      │
    ⑤ 挂载到 VllmConfig               │  VllmConfig.speculative_config = ...
       engine/arg_utils.py            │  → create_engine_config() 返回
       config/vllm.py                 │
```

下面逐环节展开源码。

---

**环节①：命令行 → argparse.Namespace**

真正的 CLI 入口是 [entrypoints/cli/main.py](file:///c:/study/vllm_vllmascend/vllm/vllm/entrypoints/cli/main.py)，`main()` 函数中通过 `FlexibleArgumentParser` 创建子命令解析器，每个 `CMD_MODULES` 的 `cmd_init()` 返回子命令列表（包括 `ServeSubcommand`）：

```python
# entrypoints/cli/main.py : main()
import vllm.entrypoints.cli.serve   # ← 注册 serve 子命令

CMD_MODULES = [
    vllm.entrypoints.cli.openai,
    vllm.entrypoints.cli.serve,     # ← 包含 ServeSubcommand
    ...
]
subparsers = parser.add_subparsers()
for cmd_module in CMD_MODULES:
    for cmd in cmd_module.cmd_init():
        cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
args = parser.parse_args()
args.dispatch_function(args)        # ← 最终调用 ServeSubcommand.cmd(args)
```

**环节②：参数注册 + EngineArgs 提取**

`ServeSubcommand.subparser_init()` 调用 `make_arg_parser()`（[entrypoints/openai/cli_args.py](file:///c:/study/vllm_vllmascend/vllm/vllm/entrypoints/openai/cli_args.py)），后者通过 `AsyncEngineArgs.add_cli_args(parser)` 注册所有引擎参数——包括 `--speculative-config`：

```python
# engine/arg_utils.py : EngineArgs.add_cli_args()
vllm_kwargs = get_kwargs(VllmConfig)

# 关键: --speculative-config 的 type=json.loads
# 这意味着命令行传入的 JSON 字符串会被自动解析为 Python dict
vllm_kwargs["speculative_config"]["type"] = optional_type(json.loads)
vllm_group.add_argument(
    "--speculative-config", "-sc", **vllm_kwargs["speculative_config"]
)

# 同时注册快捷别名（与 --speculative-config 互斥）
vllm_group.add_argument("--spec-method", ...)   # 等价于 {"method": ...}
vllm_group.add_argument("--spec-model", ...)    # 等价于 {"model": ...}
vllm_group.add_argument("--spec-tokens", ...)   # 等价于 {"num_speculative_tokens": ...}
```

执行 `vllm serve ... --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'` 后，argparse 将 JSON 字符串解析为 dict 存入 `args.speculative_config`。

`ServeSubcommand.cmd(args)` 最终进入 [entrypoints/openai/api_server.py](file:///c:/study/vllm_vllmascend/vllm/vllm/entrypoints/openai/api_server.py) 的 `run_server()`：

```python
# entrypoints/openai/api_server.py : run_server()
engine_args = AsyncEngineArgs.from_cli_args(args)   # ← 从 Namespace 提取

# engine/arg_utils.py : EngineArgs.from_cli_args()
@classmethod
def from_cli_args(cls, args: argparse.Namespace):
    attrs = [attr.name for attr in dataclasses.fields(cls)]
    return cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})
    # → EngineArgs(speculative_config={"method": "mtp", "num_speculative_tokens": 2}, ...)
```

此时 `EngineArgs.speculative_config` 是一个 **普通 dict**，尚未经历 Pydantic 验证——验证被延迟到 `create_engine_config()` 中。

---

**环节③：自动探测 speculators_config（`maybe_override_with_speculators`）**

在 `create_engine_config()` 的最开始（[engine/arg_utils.py:L1719-L1732](file:///c:/study/vllm_vllmascend/vllm/vllm/engine/arg_utils.py#L1719-L1732)），vLLM 会尝试读取模型的 `config.json`，查找是否有 `speculators_config` 字段：

```python
# engine/arg_utils.py : create_engine_config()
if not is_cloud_storage(self.model):
    (self.model, self.tokenizer, self.speculative_config) = (
        maybe_override_with_speculators(
            model=self.model,
            tokenizer=self.tokenizer,
            vllm_speculative_config=self.speculative_config,  # ← 用户传入的 dict
            ...
        )
    )
```

`maybe_override_with_speculators()`（[transformers_utils/config.py:L590](file:///c:/study/vllm_vllmascend/vllm/vllm/transformers_utils/config.py#L590)）：
1. 读取模型 `config.json` 中的 `"speculators_config"` 字段
2. 如果不存在 → 直接返回原始参数（不做任何修改）
3. 如果存在 → 从 `speculators_config` 中提取 `verifier.name_or_path` 作为新的 `model`，并构造 `speculative_config`

> **对于 DeepSeek-V3.2**：其 `config.json` 中**没有** `speculators_config` 字段（MTP 层直接嵌在 model.safetensors 中），所以 `maybe_override_with_speculators` 走步骤 2 的快速返回路径，`speculative_config` 保持用户传入的 `{"method": "mtp", "num_speculative_tokens": 2}`。

---

**环节④：构造 SpeculativeConfig 对象**

`create_engine_config()` 完成所有子 Config 的创建后，调用 `create_speculative_config()`（[engine/arg_utils.py:L1665-L1700](file:///c:/study/vllm_vllmascend/vllm/vllm/engine/arg_utils.py#L1665-L1700)）：

```python
# engine/arg_utils.py : EngineArgs.create_speculative_config()
def create_speculative_config(self, target_model_config, target_parallel_config):
    # Step A: 合并 --spec-method / --spec-model / --spec-tokens（快捷别名）
    for flag, key, value in (
        ("--spec-method", "method", self.spec_method),
        ("--spec-model", "model", self.spec_model),
        ("--spec-tokens", "num_speculative_tokens", self.spec_tokens),
    ):
        if value is not None:
            if self.speculative_config is None:
                self.speculative_config = {}
            self.speculative_config[key] = value

    # Step B: 如果完全没有 speculative 配置 → 返回 None（不启用投机解码）
    if self.speculative_config is None:
        return None

    # Step C: 注入目标模型信息（这些字段不来自 CLI，而是由 create_engine_config 传入）
    self.speculative_config.update({
        "target_model_config": target_model_config,   # ← 主模型完整配置
        "target_parallel_config": target_parallel_config,
    })

    # Step D: 构造 Pydantic 模型 → 触发 __post_init__ 验证
    return SpeculativeConfig(**self.speculative_config)
```

**Step D** 触发了 [config/speculative.py](file:///c:/study/vllm_vllmascend/vllm/vllm/config/speculative.py) 中 `SpeculativeConfig` 的 Pydantic 验证和 `__post_init__()`：

```python
# config/speculative.py : SpeculativeConfig.__post_init__()
def __post_init__(self):
    if self.method == "mtp" and self.model is None:
        # MTP 不需要额外模型，draft model 就在主模型 checkpoint 中
        self.model = self.target_model_config.model   # ← "deepseek-ai/DeepSeek-V3.2"
        if not self.quantization:
            self.quantization = self.target_model_config.quantization  # ← 量化对齐
    # ... 其他方法的后处理 ...
```

**进入 `__post_init__` 时的 SpeculativeConfig 状态**：
```
SpeculativeConfig(
    method                  = "mtp",          ← 来自 --speculative-config
    num_speculative_tokens  = 2,              ← 来自 --speculative-config
    model                   = None,           ← 未设置，MTP 不需要额外模型
    target_model_config     = ModelConfig(    ← 由 create_speculative_config 注入
        model = "deepseek-ai/DeepSeek-V3.2",
        ...
    ),
    target_parallel_config  = ParallelConfig(...),
)
```

**`__post_init__` 执行后**：`model` 被设为 `"deepseek-ai/DeepSeek-V3.2"`（与主模型相同），`quantization` 与主模型对齐。

---

**环节⑤：挂载到 VllmConfig**

回到 `create_engine_config()`（[engine/arg_utils.py:L2238](file:///c:/study/vllm_vllmascend/vllm/vllm/engine/arg_utils.py#L2238)）：

```python
# engine/arg_utils.py : create_engine_config() 末尾
return VllmConfig(
    model_config=model_config,
    cache_config=cache_config,
    ...
    speculative_config=speculative_config,  # ← SpeculativeConfig 对象挂载于此
    ...
)
```

`VllmConfig`（[config/vllm.py](file:///c:/study/vllm_vllmascend/vllm/vllm/config/vllm.py)）是整个 vLLM 引擎的**总配置容器**，后续所有模块（`EngineCore`、`GPUModelRunner`、`EagleSpeculator`）都通过 `vllm_config.speculative_config` 来访问 MTP 配置。

---

**汇总：DeepSeek-V3.2 的 `vllm serve` → SpeculativeConfig 完整数据流**

```
  用户敲下:
  vllm serve deepseek-ai/DeepSeek-V3.2 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

  argparse 解析:
    args.speculative_config = {"method": "mtp", "num_speculative_tokens": 2}

  from_cli_args:
    EngineArgs(speculative_config={"method": "mtp", "num_speculative_tokens": 2})

  maybe_override_with_speculators:
    读取 config.json → 无 speculators_config → 原样返回  [快速路径]

  create_speculative_config:
    注入 target_model_config + target_parallel_config
    → SpeculativeConfig(method="mtp", num_speculative_tokens=2, model=None, ...)
    → __post_init__: model = "deepseek-ai/DeepSeek-V3.2"  [自动设为与主模型相同]

  create_engine_config 返回:
    VllmConfig(
        model_config       = ModelConfig(model="deepseek-ai/DeepSeek-V3.2"),
        speculative_config = SpeculativeConfig(
            method                  = "mtp",
            num_speculative_tokens  = 2,
            model                   = "deepseek-ai/DeepSeek-V3.2",  ← 复用主模型
            quantization            = None,                         ← 复用主模型
            target_model_config     = <ModelConfig>,
            target_parallel_config  = <ParallelConfig>,
        )
    )
```

### 5.2.2 SpeculativeConfig 的构造

源码位置：[config/speculative.py:L118-L253](file:///c:/study/vllm_vllmascend/vllm/vllm/config/speculative.py)

关键代码：

```python
# speculative.py: SpeculativeConfig 类定义
class SpeculativeConfig:
    method: SpeculativeMethod | None = None  # ← 如 "mtp"
    num_speculative_tokens: int = Field(default=None, gt=0)  # ← 如 2
    model: str | None = None  # ← 无额外模型时为 None
    # ... 其他字段
```

`SpeculativeConfig.__post_init__()` 中的最关键逻辑（[speculative.py:L173-L176](file:///c:/study/vllm_vllmascend/vllm/vllm/config/speculative.py)）：

```python
def __post_init__(self):
    if self.method == "mtp" and self.model is None:
        # MTP 不需要额外模型，draft model 就在主模型 checkpoint 中
        self.model = self.target_model_config.model  # ← 复用主模型名
        if not self.quantization:
            self.quantization = self.target_model_config.quantization  # ← 量化对齐
```

核心要点：**MTP 的 `model` 和 `quantization` 都直接复用主模型的配置**，因为 MTP 层就嵌入在主模型 checkpoint 中，不需要单独下载模型文件。

### 5.2.3 hf_config_override：模型类型重写

源码位置：[config/speculative.py:L137-L151](file:///c:/study/vllm_vllmascend/vllm/vllm/config/speculative.py)

在初始化 `SpeculativeConfig` 时，还会调用 `hf_config_override()` 来修改 HuggingFace config，使得 vLLM 能正确识别 MTP 模型：

```python
@staticmethod
def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    if hf_config.model_type in ("deepseek_v3", "deepseek_v32"):
        hf_config.model_type = "deepseek_mtp"  # ← 改为 MTP 类型
    if hf_config.model_type == "deepseek_mtp":
        n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
        hf_config.update(
            {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
        )
```

这里的 `"deepseek_mtp"` 和 `"DeepSeekMTPModel"` 会映射到 vLLM 中 [deepseek_mtp.py](file:///c:/study/vllm_vllmascend/vllm/vllm/model_executor/models/deepseek_mtp.py) 里的 `DeepSeekMTP` 类，这是整个 MTP 模型的入口。

## 5.3 Step 2: 引擎创建 EngineCore

> **这一步做什么**：`VllmConfig` 组装完成后，创建 `LLMEngine` → `EngineCoreClient` → `EngineCore`。这是推理引擎的核心调度器，负责请求调度和模型执行协调。它只需要知道自己需不需要跑 MTP（通过一个布尔标志位），具体怎么跑 MTP 由下层组件（`GPUModelRunner` / `EagleSpeculator`）负责。

**完整创建链路**：

```
  api_server.py: run_server()
    │  engine_args.create_engine_config() → VllmConfig
    │
    └→ LLMEngine.from_vllm_config(vllm_config)       [v1/engine/llm_engine.py]
         │
         │  # Step A: 根据并行配置决定 Executor 类型
         │  executor_class = Executor.get_class(vllm_config)
         │    ├─ "mp" (默认)        → MultiprocExecutor   [v1/executor/multiproc_executor.py]
         │    ├─ "ray"              → RayDistributedExecutor
         │    └─ "uni" / None       → UniProcExecutor
         │
         │  # Step B: 根据部署模式选择 EngineCoreClient
         └→ EngineCoreClient.make_client(...)              [v1/engine/core_client.py]
              │
              ├─ multiprocess=False  → InprocClient        ← EngineCore 在同一进程中
              ├─ multiprocess=True, async=False → SyncMPClient   ← 独立进程, 同步通信
              └─ multiprocess=True, async=True  → AsyncMPClient  ← 独立进程, 异步通信
                    │
                    └→ EngineCore(vllm_config, executor_class, ...)   [v1/engine/core.py]
```

下面逐步展开源码。

---

### 5.3.1 LLMEngine：引擎入口

[LLMEngine.from_vllm_config()](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/llm_engine.py#L137-L155) 是创建 `EngineCore` 的入口：

```python
# v1/engine/llm_engine.py : LLMEngine.from_vllm_config()
@classmethod
def from_vllm_config(cls, vllm_config, ...) -> "LLMEngine":
    return cls(
        vllm_config=vllm_config,
        executor_class=Executor.get_class(vllm_config),  # ← 决定用哪个 Executor
        multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        ...
    )
```

`LLMEngine.__init__()` 中（[llm_engine.py:L104-L110](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/llm_engine.py#L104-L110)）：

```python
# v1/engine/llm_engine.py : LLMEngine.__init__()
self.engine_core = EngineCoreClient.make_client(
    multiprocess_mode=multiprocess_mode,
    asyncio_mode=False,
    vllm_config=vllm_config,         # ← 包含 speculative_config
    executor_class=executor_class,
    log_stats=self.log_stats,
)
```

其中 `Executor.get_class()`（[v1/executor/abstract.py:L48-L82](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/executor/abstract.py#L48-L82)）根据 `parallel_config.distributed_executor_backend` 决定 Executor：

```
  "mp"   → MultiprocExecutor       ← 单机多 GPU（最常见）
  "ray"  → RayDistributedExecutor  ← 多机分布式
  "uni"  → UniProcExecutor         ← 单 GPU，同一进程
```

---

### 5.3.2 EngineCoreClient.make_client：选择部署模式

[EngineCoreClient.make_client()](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/core_client.py#L82-L104) 根据 `multiprocess_mode` 和 `asyncio_mode` 分派：

```python
# v1/engine/core_client.py : EngineCoreClient.make_client()
@staticmethod
def make_client(multiprocess_mode, asyncio_mode, vllm_config, executor_class, log_stats):
    if multiprocess_mode and asyncio_mode:
        return AsyncMPClient(vllm_config, executor_class, log_stats)
    if multiprocess_mode and not asyncio_mode:
        return SyncMPClient(vllm_config, executor_class, log_stats)
    return InprocClient(vllm_config, executor_class, log_stats)  # ← 默认单进程
```

三种模式的区别：

| 模式 | EngineCore 位置 | 通信方式 | 适用场景 |
|------|----------------|---------|---------|
| **InprocClient** | 当前进程内 | 直接 Python 调用 | 开发/测试、embedding 模型 |
| **SyncMPClient** | 独立子进程 | ZMQ + multiprocessing | 生产环境（同步 API） |
| **AsyncMPClient** | 独立子进程 | ZMQ + asyncio | 生产环境（OpenAI API Server） |

无论哪种模式，最终都会调用 `EngineCore.__init__()`——区别只是调用发生在**当前进程**还是**子进程**中。

---

### 5.3.3 EngineCore.__init__：MTP 被感知

[EngineCore.__init__()](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/core.py#L93-L225) 执行了最关键的 **5 个初始化步骤**：

```python
# v1/engine/core.py : EngineCore.__init__()
class EngineCore:
    def __init__(self, vllm_config, executor_class, log_stats, ...):
        self.vllm_config = vllm_config                              # ① 保存配置
                                                        
        self.model_executor = executor_class(vllm_config)           # ② 创建 Executor
            # ↑ executor_class 就是前面选的 MultiprocExecutor / UniProcExecutor
            # 在 Executor 内部，会进一步创建 GPUModelRunner / EagleSpeculator
            
        kv_cache_config = self._initialize_kv_caches(vllm_config)   # ③ 初始化 KV Cache
                                                                    #    profile 显存, 确定 block 数量

        self.scheduler = Scheduler(                                 # ④ 创建调度器
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            ...
        )

        self.use_spec_decode = vllm_config.speculative_config is not None  # ← ⑤ MTP 感知点
```

**第⑤步是 MTP 被引擎"感知"的唯一位置**——仅一行代码，只检查 `speculative_config is not None`，不关心具体是 MTP / EAGLE / Medusa 还是其他 speculate 方法。

---

### 5.3.4 use_spec_decode 标志位的两个作用

这个简单的布尔标志位在 `EngineCore` 的两个关键方法中驱动 MTP 行为：

**作用一：`post_step()` — 每一步 Decode 后回收 draft token IDs**

[core.py:L470-L478](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/core.py#L470-L478)：

```python
def post_step(self, model_executed: bool) -> None:
    if not self.async_scheduling and self.use_spec_decode and model_executed:
        # Take the draft token ids from model executor
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
```

时序关系：

```
  step() 执行:
  ① scheduler.schedule()         → 生成 SchedulerOutput（含 draft token slots）
  ② model_executor.execute_model() → Target 模型 forward + MTP 层生成 draft tokens
  ③ scheduler.update_from_output()  → 用 Target 模型结果更新请求状态

  post_step() 执行:
  ④ model_executor.take_draft_token_ids() → 取出 MTP 生成的 draft token IDs
  ⑤ scheduler.update_draft_token_ids()    → 注入到调度器，供下一轮 schedule 使用
```

> **关键理解**：draft tokens 在 **当前 step** 的 `execute_model()` 中由 MTP 层生成，但要在 **下一个 step** 的 `schedule()` 中才被加入 batch——所以需要一个"跨 step 传递"的机制。`take_draft_token_ids` + `update_draft_token_ids` 就是这个传递通道。

**作用二：`step_with_batch_queue()` — 异步调度模式下的 grammar bitmask 对齐**

[core.py:L573-L591](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/core.py#L573-L591)：

```python
if self.use_spec_decode:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    assert draft_token_ids is not None
    # 用 draft tokens 过滤掉无效的 speculative tokens（pad 成 -1）
    self.scheduler.update_draft_token_ids_in_output(
        draft_token_ids, deferred_scheduler_output
    )
```

当 speculative decoding 与 **structured output**（grammar-constrained generation）同时使用时，需要在计算 grammar bitmask 之前知道哪些 draft tokens 是有效的、哪些需要被过滤。这个分支确保了 structured output + MTP 的正确协同。

---

### 5.3.5 完整数据流总览

```
  VllmConfig 就绪（含 speculative_config）
        │
        ▼
  LLMEngine.from_vllm_config()
        │  executor_class = Executor.get_class(vllm_config)
        │
        ▼
  EngineCoreClient.make_client()
        │  → InprocClient / SyncMPClient / AsyncMPClient
        │
        ▼
  EngineCore.__init__()
        │
        ├── model_executor = executor_class(vllm_config)
        │     └── 内部: GPUModelRunner.__init__()
        │           └── self.speculator = init_speculator(vllm_config)  [见 5.4 节]
        │
        ├── _initialize_kv_caches(vllm_config)
        │     └── 根据 max_model_len + num_speculative_tokens 计算 KV Cache 容量
        │
        ├── scheduler = Scheduler(vllm_config, kv_cache_config, ...)
        │     └── 调度器根据 num_speculative_tokens 预留 draft token slots
        │
        └── self.use_spec_decode = True  ← 引擎层面唯一的 MTP 感知点
              │
              ├── 触发 post_step() 中 take_draft_token_ids + update_draft_token_ids
              └── 触发 step_with_batch_queue() 中 grammar bitmask 的 draft token 过滤
```

**核心设计哲学**：`EngineCore` 是"引擎调度层"，它通过一个布尔标志位 `use_spec_decode` 来驱动 **draft token 的跨 step 传递逻辑**，但完全不知道也不关心 draft tokens 是怎么生成的（那是下一层 `GPUModelRunner` / `EagleSpeculator` 的职责）。这种分层设计使得 EngineCore 可以支持任意 speculative decode 方法（MTP、EAGLE、Medusa 等），无需修改调度代码。

## 5.4 Step 3: GPUModelRunner 初始化 Speculator

> **这一步做什么**：`GPUModelRunner` 是真正执行 GPU 推理的组件。它初始化时发现配置了 MTP，就创建一个 `EagleSpeculator` 作为 MTP 的"执行器"。注意：此时只创建了对象，模型权重还没加载（那是下一步的事）。

### 5.4.1 GPUModelRunner.__init__()

源码位置：[v1/worker/gpu/model_runner.py:L217-L224](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/model_runner.py)

```python
class GPUModelRunner(LoRAModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # ... 初始化主模型的 ModelRunner 基础设施 ...

        self.speculator = None
        if self.speculative_config is not None:
            if self.is_last_pp_rank:
                # 只在最后一个 Pipeline Parallel rank 上创建 speculator
                self.speculator = init_speculator(self.vllm_config, self.device)
```

### 5.4.2 init_speculator：创建 EagleSpeculator

源码位置：[v1/worker/gpu/spec_decode/\_\_init\_\_.py:L13-L19](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/__init__.py)

```python
def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_eagle():
        # MTP 走 Eagle 分支（EagleSpeculator 是统一的 speculator 实现）
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
```

`EagleSpeculator` 是 vLLM 中统一的投机解码抽象，MTP、EAGLE、EAGLE2、EAGLE3 都复用它。它内部会根据 `self.method`（这里为 `"mtp"`）做适配。

### 5.4.3 EagleSpeculator 初始化

源码位置：[v1/worker/gpu/spec_decode/eagle/speculator.py:L59-L112](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

```python
class EagleSpeculator:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.speculative_config = vllm_config.speculative_config
        self.method = self.speculative_config.method           # "mtp"
        self.num_speculative_steps = (
            self.speculative_config.num_speculative_tokens     # 2
        )
        # ... 预分配 buffer、初始化 CUDA Graph manager 等
```

## 5.5 Step 4: 加载 MTP 模型

> **这一步做什么**：这是 MTP 初始化的**最关键步骤**。主模型（Layer 0~60）加载完毕后，`EagleSpeculator` 开始加载 MTP 模型。它做了三件重要的事：① 创建 MTP 模型实例（仅 Layer 61，1层），② 把 embedding 和 lm_head 共享给 MTP 模型以节省显存，③ 从 checkpoint 加载 MTP 独有的权重。

### 5.5.1 GPUModelRunner.load_model()

源码位置：[v1/worker/gpu/model_runner.py:L228-L231](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/model_runner.py)

```python
def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
    # ... 加载主模型 (Layer 0~60) ...

    if self.speculator is not None:
        self.speculator.load_model(self.model)  # ← 将主模型传入
```

### 5.5.2 EagleSpeculator.load_model()

源码位置：[v1/worker/gpu/spec_decode/eagle/speculator.py:L152-L165](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

```python
def load_model(self, target_model: nn.Module) -> None:
    # 获取主模型的 attention 层名称（用于区分主模型和 MTP 层的 KV cache）
    target_attn_layer_names = get_layers_from_vllm_config(
        self.vllm_config, AttentionLayerBase
    ).keys()

    # 加载 MTP 模型（调用 load_eagle_model）
    self.model = load_eagle_model(target_model, self.vllm_config)

    # 找出 MTP 层独有的 attention 层名称（用于 MTP 层自己的 KV cache 管理）
    all_attn_layers = get_layers_from_vllm_config(...).keys()
    self.draft_attn_layer_names = set(all_attn_layers) - set(target_attn_layer_names)
```

### 5.5.3 load_eagle_model：创建 MTP 模型并共享权重

源码位置：[v1/worker/gpu/spec_decode/eagle/utils.py:L33-L84](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/utils.py)

这是 MTP 初始化中最核心的一步——将主模型的 embedding 层和 lm_head 共享给 MTP 模型，避免重复加载显存：

```python
def load_eagle_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    draft_model_config = speculative_config.draft_model_config
    with set_model_tag("eagle_head"):
        # 创建 MTP 模型实例：这里会触发 DeepSeekMTP.__init__(),
        # 构建 Layer 61 的 enorm/hnorm/eh_proj/mtp_block/shared_head（仅1层）
        eagle_model = get_model(vllm_config=vllm_config, model_config=draft_model_config)

    target_language_model = target_model.get_language_model()
    target_inner = target_language_model.model  # 主模型的内部结构
    draft_inner = eagle_model.model             # MTP 模型的内部结构

    # ① 共享 embed_tokens（单卡模式，Pipeline Parallel 下不共享）
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None)
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and draft_embed is not None:
            del draft_inner.embed_tokens         # ← 删除 MTP 自己的 embedding
            draft_inner.embed_tokens = target_embed  # ← 指向主模型的 embedding

    # ② 共享 lm_head
    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(eagle_model, "lm_head", None)
    if target_lm_head is not None and draft_lm_head is not None:
        del eagle_model.lm_head
        eagle_model.lm_head = target_lm_head   # ← 指向主模型的 lm_head

        # MTP 层的 shared_head.head 也需要指向同一个 lm_head
        layers = getattr(draft_inner, "layers", None)
        if layers is not None:
            items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
            for layer in items:
                sh = getattr(layer, "shared_head", None)
                if sh is not None and hasattr(sh, "head"):
                    del sh.head
                    sh.head = target_lm_head   # ← shared_head.head 也指向主模型的 lm_head

    # ③ 共享 topk_indices_buffer
    if hasattr(target_inner, "topk_indices_buffer"):
        if hasattr(draft_inner, "topk_indices_buffer"):
            del draft_inner.topk_indices_buffer
        draft_inner.topk_indices_buffer = target_inner.topk_indices_buffer

    return eagle_model
```

**关键理解**：`embed_tokens` 和 `lm_head` 通过 Python 对象的引用共享，底层 Tensor 指向同一块 GPU 显存。MTP 层不需要自己的 embedding 和 lm_head，直接将主模型的拿来用。只有 MTP 独有的权重（enorm、hnorm、eh_proj、mtp_block、shared_head.norm）才需要额外加载。

## 5.6 Step 5: MTP 权重的实际加载

> **这一步做什么**：`load_eagle_model()` 创建了 MTP 模型的结构，但里面的权重还是随机的。现在 `DeepSeekMTP.load_weights()` 开始从 checkpoint 中筛选 MTP 相关的权重并加载到正确的参数中。它需要判断每个权重属于主模型还是 MTP 层，并对 MTP 层的权重进行路径重写。

### 5.6.1 DeepSeekMTP.load_weights：筛选并重写 MTP 权重

源码位置：[model_executor/models/deepseek_mtp.py:L239-L455](file:///c:/study/vllm_vllmascend/vllm/vllm/model_executor/models/deepseek_mtp.py)

当 vLLM 的权重加载器遍历 checkpoint 所有权重时，`DeepSeekMTP.load_weights()` 通过 `get_spec_layer_idx_from_weight_name()` 判断该权重是否属于 MTP 层：

- **主模型权重**（Layer 0~60）→ `get_spec_layer_idx_from_weight_name` 返回 `None` → **跳过不加载**（主模型已有自己的加载逻辑）
- **MTP 层权重**（Layer 61，仅1层）→ 返回值 >= `mtp_start_layer_idx` → 继续处理

### 5.6.2 _rewrite_spec_layer_name：权重名重写

源码位置：[model_executor/models/deepseek_mtp.py:L458-L488](file:///c:/study/vllm_vllmascend/vllm/vllm/model_executor/models/deepseek_mtp.py)

```python
def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
    spec_layer_weight_names = ["embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"]
    shared_weight_names = ["embed_tokens"]

    # 判断权重是否属于 MTP 独有的模块
    for weight_name in spec_layer_weight_names:
        if weight_name in name:
            spec_layer_weight = True
            shared_weight = weight_name in shared_weight_names
            break

    if not spec_layer_weight:
        # 非 MTP 独有模块 → 加上 mtp_block 前缀
        # 例如: model.layers.61.mlp.gate_proj → model.layers.61.mtp_block.mlp.gate_proj
        name = name.replace(
            f"model.layers.{spec_layer}.",
            f"model.layers.{spec_layer}.mtp_block."
        )
    elif shared_weight:
        # embed_tokens 共享 → 提升到 model 级别
        # model.layers.61.embed_tokens → model.embed_tokens
        name = name.replace(f"model.layers.{spec_layer}.", "model.")

    return name
```

权重名重写对照表：

```
Checkpoint 中的原始名称                        →  重写后加载到的参数名
─────────────────────────────────────────────────────────────────────────
model.layers.61.enorm.weight                  →  model.layers.61.enorm.weight         (不变)
model.layers.61.hnorm.weight                  →  model.layers.61.hnorm.weight         (不变)
model.layers.61.eh_proj.weight                →  model.layers.61.eh_proj.weight       (不变)
model.layers.61.shared_head.norm.weight       →  model.layers.61.shared_head.norm.weight (不变)
model.layers.61.shared_head.head.weight       →  model.layers.61.shared_head.head.weight (不变)
model.layers.61.embed_tokens.weight           →  model.embed_tokens.weight            (只加载第一次)
model.layers.61.self_attn.q_proj.weight       →  model.layers.61.mtp_block.self_attn.q_proj.weight
model.layers.61.mlp.gate_proj.weight          →  model.layers.61.mtp_block.mlp.gate_proj.weight
```

> **注意**：DeepSeek-V3.2 的 checkpoint **只有 `model.layers.61.*` 的权重**，没有 `model.layers.62.*`。vLLM 创建模型时也只会创建 Layer 61 这 1 层 MTP。

## 5.7 Step 6: CUDA Graph 预热

> **这一步做什么**：CUDA Graph 是 NVIDIA GPU 的一个加速技术——它把一系列 GPU 操作"录制"下来，后续直接回放，避免每次推理都重新提交内核（kernel launch）。MTP 层也需要自己的 CUDA Graph，分为两个阶段：
> **Prefill Graph**：处理 prompt（批量输入），所有 spec_step（第1遍、第2遍...）都用 Layer 61
> **Decode Graph**：处理单个 token 逐个生成，同样都用 Layer 61

源码位置：[v1/worker/gpu/spec_decode/eagle/speculator.py:L397-L448](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

加载完模型后，`EagleSpeculator` 会为 MTP 的 prefill 和 decode 两个阶段分别捕获 CUDA Graph，以加速后续推理：

```python
def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
    # Prefill CUDA Graph（draft position 0）
    self.prefill_cudagraph_manager = PrefillEagleCudaGraphManager(...)

    # Decode CUDA Graph（draft positions > 0）
    self.decode_cudagraph_manager = DecodeEagleCudaGraphManager(...)
```

这一步类似主模型的 CUDA Graph 预热，只是为 MTP 层单独捕获。预热完成后，vLLM 的 MTP 初始化链路全部完成，可以开始处理请求。

---

# 第六部分：流程篇 —— curl 请求到输出文字的全流程

> **本节目标**：跟踪一个完整的 curl 请求从发出到收到回复的全过程，重点关注 MTP 在各个环节中做了什么。理解请求流程有助于你理解为什么 MTP 能加速、以及什么情况下加速效果最好。

## 6.1 整体流程

```
════════════════════════════════════════════════════════════════════════════════════
                   一次 curl 请求的完整 MTP 生命周期
════════════════════════════════════════════════════════════════════════════════════

  curl ──→ API Server (接收请求)
           │  [api_router.py:L44-L78] 路由到 /v1/chat/completions
           │
           ├─→ OpenAIServingChat.create_chat_completion()
           │   [chat_completion/serving.py:L228-L427] 解析请求 + 构造 generator
           │
           ├─→ AsyncLLM.generate()
           │   [v1/engine/async_llm.py:L524-L623] add_request() → EngineCore
           │
           ├─→ EngineCore.step()  ← 循环执行，直到请求完成
           │   [v1/engine/core.py:L560-L590]
           │   │
           │   ├─→ Scheduler.schedule()  ← 调度准备执行
           │   │   [v1/core/sched/scheduler.py]
           │   │
           │   ├─→ GPUModelRunner.execute_model()  ← 执行推理
           │   │   [v1/worker/gpu/model_runner.py]
           │   │   │
           │   │   ├─→ ① Target Model 前向 (Layer 0~60)
           │   │   │   │  输入: prompt tokens
           │   │   │   │  输出: sampled_token + hidden_states
           │   │   │   │
           │   │   ├─→ ② EagleSpeculator.propose()  ← MTP Draft 生成
           │   │   │   [spec_decode/eagle/speculator.py:L449-L600]
           │   │   │   │
           │   │   │   ├─ prefill: 用 target hidden_states + next_token → draft_token_1
           │   │   │   │            [speculator.py:L185-L270]
           │   │   │   │   └─ 内部调用 self.model(...) → DeepSeekMTP.forward()
           │   │   │   │      (路由到 Layer 61，第 1 遍执行)
           │   │   │   │
           │   │   │   └─ multi_step_decode: draft_token_1 + hidden_states_1 → draft_token_2
           │   │   │                [speculator.py:L272-L345]
           │   │   │       └─ 内部调用 self.model(...) → DeepSeekMTP.forward()
           │   │   │          (路由到 Layer 61，第 2 遍执行——同一层不同数据)
           │   │   │
           │   │   ├─→ ③ Target Model 第二遍前向 (验证)
           │   │   │   │  输入: [original_tokens] + [draft_tokens]
           │   │   │   │  输出: 每个 draft 位置的概率 P_target
           │   │   │   │
           │   │   └─→ ④ RejectionSampler 拒绝采样
           │   │       [v1/sample/rejection_sampler.py]
           │   │       验证 draft tokens，接受或拒绝
           │   │
           │   ├─→ Scheduler.update_from_output()
           │   │   [v1/core/sched/scheduler.py:L1350-L1390]
           │   │   记录 accept/reject 统计
           │   │
           │   └─→ EngineCoreOutputs 输出
           │
           ├─→ output_handler 后台循环
           │   [v1/engine/async_llm.py:L637-L703]
           │   从 EngineCore 拉取输出 → OutputProcessor 处理 → 放入 AsyncStream
           │
           └─→ chat_completion_stream_generator / chat_completion_full_generator
               [chat_completion/serving.py:L436-L520]
               组装 OpenAI 格式响应 → StreamingResponse / JSONResponse
```

下面逐一展开关键步骤，给出对应的源码位置和代码逻辑。

## 6.2 Step 1: curl 请求 → API Server 接收

> **这一步做什么**：用户发送的 curl 请求通过 FastAPI 路由进入 `OpenAIServingChat`，这是 OpenAI 兼容 API 的入口。MTP 在这一步完全透明——API 层不知道也不会关心底层是否使用了 MTP。

### 6.2.1 FastAPI 路由注册

源码位置：[chat_completion/api_router.py:L44-L78](file:///c:/study/vllm_vllmascend/vllm/vllm/entrypoints/openai/chat_completion/api_router.py)

当用户发送 curl 请求 `POST /v1/chat/completions`，FastAPI 将其路由到 `create_chat_completion` 函数：

```python
@router.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = chat(raw_request)  # 获取 OpenAIServingChat 实例
    generator = await handler.create_chat_completion(request, raw_request)
    # 根据 generator 类型返回 JSONResponse 或 StreamingResponse
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), ...)
    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump(), ...)
    return StreamingResponse(content=generator, media_type="text/event-stream")
```

### 6.2.2 OpenAIServingChat.create_chat_completion()

源码位置：[chat_completion/serving.py:L228-L427](file:///c:/study/vllm_vllmascend/vllm/vllm/entrypoints/openai/chat_completion/serving.py)

这个函数做了以下事情：
1. **渲染 chat 模板**：将 messages 转换为 prompt token IDs
2. **构造 SamplingParams**：max_tokens、temperature 等
3. **调用引擎**：

```python
# serving.py: ~L350
generator = self.engine_client.generate(
    engine_input,
    sampling_params,
    sub_request_id,
    lora_request=lora_request,
    ...
)
```

这里的 `self.engine_client` 是 `AsyncLLM` 实例，`generate()` 返回一个 `AsyncGenerator[RequestOutput, None]`。

## 6.3 Step 2: AsyncLLM.generate() → EngineCore

> **这一步做什么**：API Server 拿到请求后，不直接推理，而是通过 `AsyncLLM` 将请求送入 `EngineCore`（推理引擎核心）。`AsyncLLM` 作为中间层，负责异步通信和输出流转发。和 Step 1 一样，MTP 在此处是透明的。

### 6.3.1 将请求加入 EngineCore

源码位置：[v1/engine/async_llm.py:L537-L622](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/async_llm.py)

```python
async def generate(self, prompt, sampling_params, request_id, ...):
    q = await self.add_request(
        request_id, prompt, sampling_params, ...  # ← 将请求加入 EngineCore
    )
    # 循环等待 EngineCore 输出
    finished = False
    while not finished:
        out = q.get_nowait() or await q.get()
        finished = out.finished
        if out is not STREAM_FINISHED:
            yield out  # ← 每次 yield，外层 API server 就发送一次 SSE 事件
```

`add_request()` 内部会构造 `EngineCoreRequest` 并调用 `self.engine_core.add_request(request)` 将请求送入 `EngineCore`。

### 6.3.2 后台 output_handler

源码位置：[v1/engine/async_llm.py:L654-L703](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/async_llm.py)

`output_handler` 是一个后台 asyncio 任务，持续从 `EngineCore` 拉取输出，经过 `OutputProcessor` 处理后，推入对应请求的 `AsyncStream`（即上面 `generate()` 中的 `q`）：

```python
async def output_handler():
    while True:
        outputs = await engine_core.get_output_async()
        processed_outputs = output_processor.process_outputs(
            outputs_slice, outputs.timestamp, iteration_stats
        )
```

## 6.4 Step 3: EngineCore.step() —— 调度 + 执行

> **这一步做什么**：进入推理引擎的核心循环。`EngineCore.step()` 是推理引擎的"心跳"——它每次迭代从 scheduler 拿到本轮要跑的任务，送到 GPU 执行推理，执行完后再把结果交给 scheduler 更新状态。MTP 在这里开始介入：每次 step 完，EngineCore 会收集本轮生成的 draft tokens 传给 scheduler，供下轮使用。

源码位置：[v1/engine/core.py:L560-L590](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/engine/core.py)

`EngineCore.step()` 是引擎的每次迭代入口：

```python
class EngineCore:
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        # ① 调度：决定哪些请求在本轮执行
        scheduler_output = self.scheduler.schedule()
        # ② 执行模型
        model_output = self.model_executor.execute_model(scheduler_output)
        # ③ MTP 相关：将新生成的 draft tokens 传递给 scheduler
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)
```

关键点：**每次 step 执行完后，如果启用了 MTP，会把本轮生成的 draft token IDs 传给 scheduler，以便下一次调度时使用**。

## 6.5 Step 4: Scheduler 调度——MTP 相关逻辑

> **这一步做什么**：Scheduler 是推理引擎的"大脑"——它决定每个请求在本轮应该处理多少 token、是否需要用 MTP 来加速。对于启用了 MTP 的请求，scheduler 会把上一轮的 draft tokens 打包进本轮的调度结果中，这样 GPUModelRunner 执行时就知道要验证哪些 draft tokens。

源码位置：[v1/core/sched/scheduler.py:L500-L515](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/core/sched/scheduler.py)

在 `Scheduler.schedule()` 中，对每个 running 请求，scheduler 会计算本轮需要验证的 spec tokens：

```python
# scheduler.py: ~L500-515
if request.spec_token_ids:
    num_scheduled_spec_tokens = (
        num_new_tokens
        + request.num_computed_tokens
        - request.num_tokens
        - request.num_output_placeholders
    )
    if num_scheduled_spec_tokens > 0:
        spec_token_ids = request.spec_token_ids
        if len(spec_token_ids) > num_scheduled_spec_tokens:
            spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
        scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

    # 清空旧的 spec tokens，下一轮会从 model_runner 获得新的 draft tokens
    request.spec_token_ids = []
```

### 6.5.1 update_draft_token_ids

源码位置：[v1/core/sched/scheduler.py:L1691-L1712](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/core/sched/scheduler.py)

EngineCore step 完后调用 `update_draft_token_ids()`，将本轮 Target 模型前向验证后输出的新 draft tokens 存入对应请求：

```python
def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
    for request_id, token_ids in draft_token_ids.items():
        request = self.requests.get(request_id)
        if request is not None:
            request.spec_token_ids = list(token_ids)
```

## 6.6 Step 5: GPUModelRunner.execute_model() —— MTP 执行核心

> **这一步做什么**：这是 MTP 投机解码的**核心执行逻辑**。`GPUModelRunner` 拿到 scheduler 的调度结果后，会把请求送到 GPU 执行，包含完整的四个阶段：Target 前向 → MTP Draft 生成 → Target 验证 → 拒绝采样。理解这一步就理解了 MTP 的整个运行时行为。
> 
> **再回顾一下论文的数据流**：请看第二部分 2.3 的 hidden_states 流程图，这里的执行就是那段逻辑的工程实现。

这是 MTP 投机解码的**核心执行逻辑**，包含了 Target 前向 → Draft 生成 → Target 验证 → 拒绝采样的完整过程。

源码位置：[v1/worker/gpu/model_runner.py:L1200-L1250](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/model_runner.py)

（以下是简化逻辑，真实代码涉及 CUDA Graph、Pipeline Parallel 等细节）

```
GPUModelRunner.execute_model()
│
├── ① Target Model 第一遍前向
│   │  input: input_ids (包含上一次接受的 tokens + 新的 prompt tokens)
│   │  forward: Layer 0~60
│   │  output: sampled_token_ids [batch_size] + hidden_states [num_tokens, H]
│   │
│   ├── self.model(...)  →  Target 模型前向
│   └── self.sampler(logits)  →  采样得到 next token
│
├── ② EagleSpeculator.propose()  ← MTP Draft 生成
│   [详细见 §6.7]
│   │  用 Target hidden_states 生成 num_speculative_tokens 个 draft tokens
│   │  draft_tokens shape: [num_reqs, num_speculative_tokens]
│   │
│   └── self.req_states.draft_tokens[...] = draft_tokens
│
├── ③ Target Model 第二遍前向（验证）
│   │  input: [original_tokens] + [draft_tokens]
│   │  forward: Layer 0~60 (验证每个 draft token)
│   │  output: logits for each draft position
│   │
│   └── 这一步和普通的 decoding forward 合并在一起执行的
│
└── ④ RejectionSampler 拒绝采样
    [详细见 §6.8]
    │  对每个 draft token 做 accept/reject 判断
    └── output: 最终 token ids
```

## 6.7 Step 6: EagleSpeculator.propose() —— MTP Draft 生成详细过程

> **这一步做什么**：`EagleSpeculator` 是 MTP 的"执行引擎"。它从 Target 模型拿到最后一层的 `hidden_states`（语义信息）和刚采样的 `token`（位置信息），逐步调用 MTP 层 Layer 61 生成 draft tokens。因为 `num_speculative_tokens=2` 而只有 1 层 MTP，同一层 Layer 61 会被**循环执行 2 遍**。输入是语义+位置的融合，输出是猜测的 token IDs。

源码位置：[v1/worker/gpu/spec_decode/eagle/speculator.py:L449-L600](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

这是 MTP 最核心的执行代码，完整展示了如何用 Target 的 hidden_states 逐步生成 draft tokens：

```python
@torch.inference_mode()
def propose(
    self,
    input_batch: InputBatch,
    attn_metadata: dict[str, Any],
    slot_mappings: dict[str, torch.Tensor],
    last_hidden_states: torch.Tensor,   # ← Target Model 最后一层的 hidden_states
    aux_hidden_states: list[torch.Tensor] | None,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,         # ← Target Model 刚采样的 token ids
    next_prefill_tokens: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    ...
) -> torch.Tensor:
```

### 6.7.1 阶段一：prefill（生成第 1 个 draft token）

源码位置：[eagle/speculator.py:L505-L550](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

```python
# ① 准备输入：复制 Target 的 hidden_states
self.hidden_states[:num_tokens].copy_(hidden_states)

# ② 准备输入：确定输入给 MTP 的 token ids
prepare_eagle_inputs(
    self.last_token_indices,
    self.current_draft_step,
    self.input_buffers,       # ← 包含 input_ids, positions, query_start_loc 等
    input_batch,
    num_sampled,
    num_rejected,
    last_sampled,             # ← Target 刚采样的 token
    next_prefill_tokens,
    self.max_num_reqs,
)

# ↓ prepare_eagle_inputs 内部做的事情（_prepare_eagle_inputs_kernel）：
#   1. 将 target 的 input_ids 整体左移 1 位：
#      eagle_input_ids = target_input_ids[1:] + [last_sampled]
#      例如 target 输入 [T0,T1,T2,T3,T4,T5]，target 采样 T6
#      → eagle 输入变成 [T1,T2,T3,T4,T5,T6]
#   2. 记录 last_token_indices（最后位置的索引，用于提取 hidden_states）
#   3. 复制 positions（与 target 相同）
#
# 关键理解：MTP prefill 阶段处理的不是"单个 token"，
# 而是整个被左移了一位、末尾拼接了 last_sampled 的完整 prompt 序列。
# MTP Layer 61 会为序列中每个位置都计算 hidden_states，
# 但后续只取 last_token_indices 位置的 hidden_states 用于预测 draft token。

# ③ 执行 prefill（CUDA Graph 或 Eager 模式）
#  内部调用 self.prefill() → self.generate_draft()
#  → self.run_model() → self.model(...) → DeepSeekMTP.forward()
#  用 current_draft_step=0 路由到 Layer 61
if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
    self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
else:
    self.prefill(num_reqs, prefill_batch_desc.num_tokens,
                 attn_metadata, slot_mappings, ...)

if self.num_speculative_steps == 1:
    return self.draft_tokens[:num_reqs, :1]  # 只有 1 个 draft token，提前返回
```

### 6.7.2 阶段二：multi_step_decode（生成第 2~N 个 draft token）

源码位置：[eagle/speculator.py:L552-L598](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

```python
# ④ 准备 decode 阶段的输入：用第 1 个 draft token 作为下一个 MTP 层的输入
#  与 prefill 不同：decode 阶段每 request 只输入 1 个 token（上一轮猜出的 draft token）
#  不处理整个序列，MTP Layer 61 每个 request 只跑 1 个 position
#
#  ⚠️ 关键：prepare_eagle_decode 不仅设置了 input_ids，还做了两件事：
#     - 将 position 递增 1（从 prefill 最后位置 → 下一个位置）
#     - 将 seq_len 设置为 target_seq_len - num_rejected + 1
#   这一步保证了 MTP 的 KV cache 写入正确的 position slot
prepare_eagle_decode(
    self.draft_tokens[:num_reqs, 0],  # ← 第 1 个 draft token
    input_batch.seq_lens,
    num_rejected,
    self.input_buffers,
    self.max_model_len,
    self.max_num_reqs,
)

# ⑤ 执行 multi-step decode（CUDA Graph 或 Eager 模式）
#  对 spec_step=1..N-1 逐遍生成（num_speculative_tokens=2 时只循环 1 次，执行 Layer 61 第2遍）
if decode_batch_desc.cg_mode == CUDAGraphMode.FULL:
    self.decode_cudagraph_manager.run_fullgraph(decode_batch_desc)
else:
    self.multi_step_decode(num_reqs, dummy_run, decode_batch_desc, ...)
    # → 内部循环调用 self.generate_draft()，更新 hidden_states 和 draft tokens
```

### 6.7.3 generate_draft：单次 MTP 层的执行

源码位置：[eagle/speculator.py:L341-L385](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

```python
def generate_draft(self, num_reqs, num_tokens_padded, attn_metadata,
                   slot_mappings, ...):
    # ① 执行 MTP 模型前向
    #   self.model → DeepSeekMTP
    #   self.run_model() → self.model(input_ids=..., ...)
    #   → DeepSeekMTP.forward() 根据 self.current_draft_step 路由
    last_hidden_states, hidden_states = self.run_model(...)

    # ② 用 MTP 层输出的 hidden_states 计算 logits
    logits = self.model.compute_logits(last_hidden_states)
    #   内部: shared_head(last_hidden_states) → lm_head → logits

    # ③ 从 logits 中采样 draft token
    draft_tokens = self._sample_draft(logits, ...)
    #   使用 temperature 做简单采样（draft 阶段不执行 top_k/top_p）

    # ④ 更新下一轮的输入
    update_eagle_draft_inputs(
        draft_tokens,                          # ← 本轮采样的 draft token
        self.current_draft_step,               # ← 当前 draft 步数（0-based，仅用于索引记录）
        hidden_states,                         # ← 本轮 MTP 层输出的 hidden_states
        self.draft_tokens,                     # ← 累积所有 draft tokens
        self.hidden_states,                    # ← 更新输入 hidden_states（下一层用）
        self.input_buffers,
        num_reqs, self.max_model_len,
        self.num_speculative_steps,
    )
```

**关键链路**：`self.model(...)` → `DeepSeekMTP.forward()` → 根据 `self.current_draft_step` 选择 `self.model.layers[mtp_start + step_idx]` → `DeepSeekMultiTokenPredictorLayer.forward()` → shared_head → lm_head → 采样 draft token。

`update_eagle_draft_inputs` 除了更新 input_ids 和 hidden_states，还在 GPU kernel 内部做了两件事：

- **`position += 1`**：每遍 MTP Layer 61 前向处理不同的序列位置
- **`seq_len += 1`**：递增序列长度，使下一遍 attention 能看到之前所有 KV（包括前面自己写的）

源码：[speculator.py:L795-L864](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

### 6.7.4 KV cache 机制：同一层跑两遍，KV 为何不冲突

这是理解 MTP 的核心问题：**MTP 只有 1 层权重（Layer 61），但 `num_speculative_tokens=2` 时需要跑 2 遍，KV cache 会不会互相覆盖？**

答案：**不会**。因为 KV cache 的 key 是 `(layer_id, position)`，而不是 `(layer_weight)`。

```
MTP Layer 61 每次执行时的 position 变化:

prefill 阶段:
  prepare_eagle_inputs → positions = [0, 1, 2, ..., S-1]（完整序列）
  prefill() 前向 → Layer 61 为每个 position 写 KV cache
  prefill() 最后 → positions[:num_reqs] = last_token_indices（只保留最后位置）

transition (prepare_eagle_decode):
  position += 1  →  position = S
  seq_len = target_seq_len - num_rejected + 1  →  seq_len = S+1

decode step 1 (multi_step_decode[0]):
  generate_draft → Layer 61 在 position=S 执行
  → KV cache 写入: KV[Layer61][pos=S]  ← 第 1 遍的结果
  update_eagle_draft_inputs:
    position += 1  →  position = S+1
    seq_len += 1   →  seq_len = S+2

decode step 2 (multi_step_decode[1], 如果 num_speculative_tokens=3):
  generate_draft → Layer 61 在 position=S+1 执行
  → KV cache 写入: KV[Layer61][pos=S+1]  ← 第 2 遍的结果，不同位置！
  注意: 这遍 attention 可以看到 pos=0..S 的所有 KV
        包括第 1 遍刚写的 KV[Layer61][pos=S]
```

**关键点**：

1. **MTP 有自己的 KV cache**：代码中通过 `draft_attn_layer_names` 区分 MTP 层独有的 attention 层，这些层分配独立的 KV cache block。源码：[speculator.py:L169-L173](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

2. **每遍重建 slot_mappings**：`multi_step_decode` 的循环中，每遍都重新调用 `self.block_tables.compute_slot_mappings(positions=...)` 构建 slot_mappings。因为 position 变了，映射到不同的 cache slot。源码：[speculator.py:L305-L336](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py)

3. **seq_len 递增**：`update_eagle_draft_inputs` 每遍 `seq_len += 1`，确保下一遍的 attention 能 attend 到前面自己写的 KV。

这跟主模型处理不同 position 的 token 是同一个道理——同一个 Transformer Block 在每个 position 都写 KV cache，不同 position 之间不会互相覆盖。

## 6.8 Step 7: RejectionSampler 拒绝采样验证

> **这一步做什么**：Target 模型对每个 draft token 位置都跑了一遍前向，得到了"主模型认为应该输出什么"的概率分布 `P_target`。`RejectionSampler` 将 MTP 猜的 `P_draft` 与主模型的 `P_target` 逐位比较，用接受概率公式 `min(1, P_target / P_draft)` 决定接受还是拒绝。这是保证 MTP **不改变生成质量**的关键。

源码位置：[v1/sample/rejection_sampler.py:L61-L165](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/sample/rejection_sampler.py)

```python
class RejectionSampler(nn.Module):
    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        # ① Bonus token: 如果所有 draft tokens 全部被接受，
        #    从 Target 验证 forward 的最后一个 logits 采样额外 token
        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_token_ids = self.sampler(logits=bonus_logits, ...)

        # ② Target 模型对 draft token 概率
        target_logits = logits[metadata.target_logits_indices]

        # ③ 拒绝采样（GPU kernel 实现）
        output_token_ids = rejection_sample(
            metadata.draft_token_ids,    # ← MTP 猜的 draft tokens
            draft_probs,                 # ← MTP 给出的概率 (P_draft)
            target_logits,               # ← Target 模型给出的概率 (P_target)
            bonus_token_ids,
            ...
        )
        return SamplerOutput(sampled_token_ids=output_token_ids, ...)
```

### 6.8.1 rejection_sample 核心算法

源码位置：[v1/sample/rejection_sampler.py:L392-L470](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/sample/rejection_sampler.py)

对每个 draft token `x`，GPU kernel 执行以下逻辑：

1. 从均匀分布采样 `u ~ Uniform(0, 1)`
2. 计算接受概率：
   - 若 `P_target(x) >= P_draft(x)` → 肯定接受（因为 `min(1, >=1) = 1`）
   - 若 `P_target(x) < P_draft(x)` → 以概率 `P_target(x) / P_draft(x)` 接受
3. 若拒绝：从 `max(0, P_target - P_draft)` 的修正分布中重新采样

```
直观示例：
  ┌──────────────────────────────────────────────────────┐
  │  P_target("def") = 0.9    P_draft("def") = 0.3      │
  │  → P_target > P_draft → 肯定接受 ✓                  │
  │                                                      │
  │  P_target("def") = 0.2    P_draft("def") = 0.8      │
  │  → P_draft > P_target → 以 0.2/0.8 = 25% 概率接受  │
  │  → 75% 概率拒绝，从修正分布重新采样                   │
  └──────────────────────────────────────────────────────┘
```

**数学上可以证明：这个算法保证输出的 token 分布与纯 Target Model 自回归生成的分布完全一致。** MTP 只是加速了生成过程，不改变生成质量。

### 6.8.2 接受还是拒绝？

假设 `num_speculative_tokens = 2`。以下描述的"本次 Target 前向"接收上一次 Target 前向的 token（"Hello"）+ 上一次 MTP 的 draft tokens 作为输入，在同一趟前向中完成验证和 bonus 采样。

```
情况 1：全部接受 ✓✓
  上一次 Target 前向的 token:  "Hello"  ← 上一趟产出的 token
  上一次 MTP 的 draft:        "World"  ← 本趟验证
                              "!"      ← 本趟验证
  本次 Target 前向验证:       "World"✓, "!"✓
  本次 bonus 采样:            "How"
  本次净输出: "World" + "!" + "How" = 3 个 token
  （"Hello" 在上一次 Target 前向已输出）
  等效传统推理 3 次 forward，Target 只跑 1 次 forward

情况 2：部分接受 ✓✗
  上一次 Target 前向的 token:  "Hello"
  上一次 MTP 的 draft:        "World"  ← 接受 ✓
                              "nice"   ← 拒绝 ✗（从修正分布重新采样）
  本次 bonus 采样:            接着 "nice" 的修正分布继续
  本次净输出: 验证通过的 draft + 修正 token + bonus
  等效节省部分 Target forward（draft 全错时无加速）

情况 3：全部拒绝 ✗✗
  上一次 Target 前向的 token:  "Hello"
  上一次 MTP 的 draft:        "xyz"    ← 拒绝 ✗
                              "abc"    ← 拒绝 ✗
  从 Target 的修正分布重新采样所有位置
  这种情况 MTP 没有任何加速（还额外消耗了 MTP forward 的计算）
```

## 6.9 Step 8: 输出处理与响应返回

### 6.9.1 Scheduler.update_from_output

源码位置：[v1/core/sched/scheduler.py:L1350-L1390](file:///c:/study/vllm_vllmascend/vllm/vllm/v1/core/sched/scheduler.py)

RejectionSampler 输出 token ids 后，scheduler 会更新请求状态，记录 MTP 统计：

```python
# scheduler.py: ~L1350-1390
def update_from_output(self, scheduler_output, model_runner_output):
    for req_id in ...:
        scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
        if scheduled_spec_token_ids:
            num_draft_tokens = len(scheduled_spec_token_ids)
            num_accepted = self._count_accepted(...)
            num_rejected = num_draft_tokens - num_accepted
            # 记录统计信息
```

### 6.9.2 EngineCore → AsyncLLM → API Server

`EngineCoreOutputs` 包含本轮生成的新 token IDs，通过以下链路返回给用户：

```
      EngineCore. get_output()
          │
          ▼
      OutputProcessor. process_outputs()
          │   └─→ 将新 tokens 放入 Request 的 AsyncStream
          │
          ▼
      AsyncLLM. generate()   (yield RequestOutput)
          │
          ▼
      OpenAIServingChat. chat_completion_stream_generator()
          │   ├─ [chat_completion/serving.py:L436-L520]
          │   │   逐个 token 组装 SSE 格式:
          │   │   data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}
          │   │
          │   └─ StreamingResponse(content=generator, ...)
          │
          ▼
      curl 收到 StreamingResponse 的 SSE 流
```

（非 streaming 模式则走 `chat_completion_full_generator()`，收集所有 token 后一次性返回 JSON）

## 6.10 配置参数

在 vLLM 启动参数中启用 MTP：

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

| 参数 | 说明 | 推荐值 |
|---|---|---|
| `method` | 投机解码方法 | `"mtp"` |
| `num_speculative_tokens` | 每次猜几个 draft token（运行时循环 Layer 61） | 2（与 DeepSeek-V3.2 训练一致） |
| `model` | MTP 权重来源 | 不填（自动复用主模型） |

---

# 第七部分：进阶篇 —— MTP 与其他投机解码的区别

> **如果你只想理解 MTP**：Part 2~6 已经完整覆盖了 MTP 的原理、实现和工程流程。Part 7 是可选内容，帮助有好奇心的读者了解 MTP 和其他投机解码方案的异同。

| 方案 | 一句话总结 | 与 MTP 的关系 |
|---|---|---|
| **EAGLE** | 独立的 Draft Model，从 Target 提取 hidden states 后预测 | MTP 的 vLLM 工程实现沿用了 EAGLE 的 `EagleSpeculator` 框架 |
| **Medusa** | 多个并行的分类头同时预测不同位置的 token | MTP 是串行的，Medusa 是并行的，但都是"多头预测"思路 |
| **标准投机解码** | 小模型猜 + 大模型验证，两个独立模型 | MTP 把"小模型"内置到了"大模型"里，省显存、省维护 |

## 7.1 EAGLE vs DeepSeek MTP

| | EAGLE | DeepSeek MTP |
|---|---|---|
| 原理 | 独立的 Eagle Draft Model，从 Target 提取 hidden states 后预测 | 内置 MTP 层，与主模型共享权重 |
| 模型文件 | 需要单独的 Eagle 权重 | MTP 权重包含在主模型 checkpoint 中 |
| Draft 层数 | 通常 1 层 | Checkpoint 中只有 1 层，运行时循环执行 |
| 训练方式 | 需要额外训练 Eagle 模型 | 与主模型联合训练 |

## 7.2 Medusa vs DeepSeek MTP

| | Medusa | DeepSeek MTP |
|---|---|---|
| 原理 | 多个并行的分类头分别预测不同位置的 token | 串行的 MTP 层逐层预测 |
| 预测方式 | 并行（所有头同时预测） | 串行（同一层循环执行，hidden_states 传递） |
| 信息传递 | 各头独立 | 上一遍的 hidden_states 传给下一遍 |

---

# 总结

## 核心知识地图

```
                                MTP 核心知识地图
                                ════════════════
                                    
  ① 背景: LLM 自回归生成慢 → 投机解码提速 → MTP 无需额外模型
  
  ② 概念: Target Model (主模型) + Draft Model (MTP层) + RejectionSampler (验证)
  
  ③ 架构: DeepSeekMTP ← DeepSeekMultiTokenPredictor ← DeepSeekMultiTokenPredictorLayer
     DeepSeek-V3.2: 主模型 61 层 (Layer 0~60) + MTP 仅 1 层 (Layer 61)
     Checkpoint 中只有 Layer 61 的权重，num_speculative_tokens=2 时循环执行 2 遍
  
  ④ 源码: enorm/hnorm → eh_proj(融合) → mtp_block(Transformer) → shared_head(输出)
  
  ⑤ 启动流程: 命令行 → SpeculativeConfig → EngineCore → GPUModelRunner → EagleSpeculator
     → load_eagle_model(创建MTP模型+共享embedding/lm_head) → load_weights(加载MTP权重)
     → CUDA Graph 预热
  
  ⑥ 请求全流程: curl → API Server → AsyncLLM.generate() → EngineCore.step()
     → Schedule → execute_model (Target前向→MTP propose→Target验证→RejectionSampler)
     → output_handler → StreamingResponse/JSONResponse
  
  ⑦ 配置: --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
  
  ⑧ 关键文件:
     - config/speculative.py                        ← SpeculativeConfig 配置解析
     - v1/engine/core.py                            ← EngineCore 引擎主循环
     - v1/worker/gpu/model_runner.py                ← GPUModelRunner 执行推理
     - v1/worker/gpu/spec_decode/eagle/speculator.py ← EagleSpeculator MTP propose
     - v1/worker/gpu/spec_decode/eagle/utils.py     ← load_eagle_model 权重共享
     - v1/core/sched/scheduler.py                   ← Scheduler MTP 调度
     - v1/sample/rejection_sampler.py               ← 拒绝采样验证
     - model_executor/models/deepseek_mtp.py         ← MTP 模型定义+权重加载
     - entrypoints/openai/chat_completion/api_router.py ← API 路由入口
     - entrypoints/openai/chat_completion/serving.py    ← 请求处理+响应组装
     - v1/engine/async_llm.py                       ← AsyncLLM generate + output_handler
```