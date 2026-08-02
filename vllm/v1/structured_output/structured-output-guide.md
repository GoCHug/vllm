# vLLM V1 结构化输出 (Structured Output) 与状态机原理详解

> 本文基于 `vllm/v1/structured_output/` 目录源码，系统讲解结构化输出的设计、状态机原理、各后端实现，以及与投机解码 (speculative decoding) 和推理模式 (reasoning/thinking) 的交互。

---

## 目录

1. [什么是结构化输出](#1-什么是结构化输出)
2. [整体架构与分层](#2-整体架构与分层)
3. [核心抽象：backend_types.py](#3-核心抽象backend_typespy)
4. [状态机原理(核心)](#4-状态机原理核心)
5. [请求生命周期与状态机交互](#5-请求生命周期与状态机交互)
6. [四种后端实现对比](#6-四种后端实现对比)
7. [推理模式 (reasoning) 与结构化输出的交互](#7-推理模式-reasoning-与结构化输出的交互)
8. [投机解码集成](#8-投机解码集成)
9. [编译、缓存与安全](#9-编译缓存与安全)
10. [apply_grammar_bitmask 实现细节](#10-apply_grammar_bitmask-实现细节)
11. [端到端时序总览](#11-端到端时序总览)
12. [关键文件索引](#12-关键文件索引)

---

## 1. 什么是结构化输出

### 1.1 问题背景

LLM 是**自回归 (autoregressive)** 生成：每一步根据已生成的前缀，预测下一个 token 的概率分布（logits），再从中采样。原始采样过程**没有任何语法约束**，因此：

- 要求输出合法 JSON 时，模型可能生成多余的逗号、未闭合的括号、甚至是自然语言解释；
- 要求匹配某个正则时，可能整体偏离；
- 要求从固定选项 (choice) 中选时，可能附带额外解释文字。

### 1.2 目标

**结构化输出**的目标是：保证模型最终生成的 token 序列**一定**满足用户给定的规格 (spec)，例如 JSON Schema、正则表达式、BNF/Lark 文法、候选选项等。

### 1.3 核心思想：分布层面的约束 (Constrained Decoding)

结构化输出**不改变模型架构**，而是在采样**之前**对 logits 做约束：

1. 在每个解码步，根据"到目前为止已生成的前缀"，计算出"下一个合法 token 集合"；
2. 用一个 **bitmask（位掩码）** 表示该集合：词表里每个 token 对应一个 bit，合法为 1，非法为 0；
3. 把非法 token 的 logit 设为 `-inf`（即概率归零），这样采样器**不可能**采到非法 token。

```
                        ┌──────────────┐
   前缀 token 序列 ───▶ │  状态机/文法  │ ──▶ 当前合法 token 集合
                        └──────────────┘          │
                                                  ▼
                                            位掩码 bitmask
                                                  │
                                                  ▼
        原始 logits  ─────────────────────▶  masked logits ──▶ 采样 ──▶ 下一个 token
        (vocab_size)     非法位 -> -inf         (只剩合法 token)
```

关键点：**"到目前为止已生成的前缀"决定了合法集合，但前缀可能无限长**。状态机 (FSM) 的作用就是把"任意长的前缀"压缩成一个**有限的当前状态**，使得"从该状态出发的合法集合"可计算。这就是为什么需要状态机。

---

## 2. 整体架构与分层

结构化输出分为**引擎级 (engine-level)** 和**请求级 (request-level)** 两层，二者通过抽象基类解耦，支持多种可插拔后端。

```
                        ┌─────────────────────────────────────────────┐
                        │         StructuredOutputManager              │  引擎级管理器
                        │  (vllm/v1/structured_output/__init__.py)     │  每个引擎一个
                        │                                             │
                        │  - grammar_init()      : 编译文法            │
                        │  - grammar_bitmask()   : 批量生成 bitmask    │
                        │  - should_advance()    : 是否推进状态机      │
                        │  - should_fill_bitmask(): 是否填充掩码       │
                        └───────────────┬─────────────────────────────┘
                                        │ 持有单个实例
                                        ▼
                ┌──────────────────────────────────────────────────────┐
                │        StructuredOutputBackend  (ABC, 引擎级)         │
                │  compile_grammar() / allocate_token_bitmask()/destroy│
                └──────────────────────────────────────────────────────┘
                                        ▲ 抽象基类
                ┌────────────┬──────────┴───────────┬─────────────────┐
                │            │                      │                 │
        ┌───────┴──────┐ ┌───┴────────┐ ┌───────────┴────────┐ ┌──────┴──────────┐
        │ Xgrammar     │ │ Outlines   │ │ Guidance(llguidance)│ │LMFormatEnforcer │
        │ Backend      │ │ Backend    │ │ Backend             │ │ Backend         │
        └───────┬──────┘ └───┬────────┘ └───────────┬────────┘ └──────┬──────────┘
                │            │                      │                 │
                ▼            ▼                      ▼                 ▼
        ┌────────────┐ ┌──────────┐          ┌────────────┐    ┌──────────────────┐
        │Xgrammar    │ │Outlines  │          │Guidance    │    │LMFormatEnforcer  │  请求级
        │ Grammar    │ │ Grammar  │          │ Grammar    │    │ Grammar          │  状态机实例
        │(GrammarMatch│(Guide/DFA│           │(LLMatcher) │    │(TokenEnforcer)   │  每请求一个
        │ er)        │ │)         │          │            │    │                  │
        └────────────┘ └──────────┘          └────────────┘    └──────────────────┘
                ▲            ▲                      ▲                 ▲
                │            │                      │                 │
                └────────────┴──────────────────────┴─────────────────┘
                                        │ 实现
                                        ▼
                ┌──────────────────────────────────────────────────────┐
                │      StructuredOutputGrammar  (ABC, 请求级)           │
                │ accept_tokens/validate_tokens/rollback/              │
                │ fill_bitmask/is_terminated/reset                     │
                └──────────────────────────────────────────────────────┘
```

### 2.1 两层抽象

| 层级 | 抽象类 | 生命周期 | 职责 | 位置 |
|------|--------|----------|------|------|
| 引擎级 | `StructuredOutputBackend` | 整个引擎一个 | 编译文法、分配 bitmask 缓冲 | `backend_types.py:98` |
| 请求级 | `StructuredOutputGrammar` | 每个请求一个 | **就是状态机**：推进、回滚、生成掩码 | `backend_types.py:31` |

> 命名约定：`Backend` = 引擎级（持有 compiler/cache/tokenizer），`Grammar` = 请求级（每个请求的 matcher/状态机实例）。

### 2.2 Manager 的职责

`StructuredOutputManager` (`__init__.py:35`) 是引擎级协调者，**不实现状态机逻辑本身**，而是：

- 懒加载并持有唯一的 `backend` 实例（`grammar_init`, `__init__.py:114`）；
- 异步编译文法（`executor` 线程池，`__init__.py:77`）；
- 在每个调度步批量生成 bitmask（`grammar_bitmask`, `__init__.py:212`）；
- 决定是否推进状态机、是否填充掩码（与 reasoning 模式相关）。

### 2.3 单后端约束

```python
# __init__.py:127 附近注释
# NOTE: We only support a single backend. We do NOT support different
# backends on a per-request basis in V1 (for now, anyway...).
```

V1 中**整个引擎只用一个后端**，由 `sampling_params.structured_outputs._backend` 在请求校验阶段确定，第一次需要时懒加载。

---

## 3. 核心抽象：backend_types.py

### 3.1 规格类型枚举

```python
# backend_types.py:19
class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()           # JSON Schema
    JSON_OBJECT = enum.auto()    # 仅 {"type":"object"}
    REGEX = enum.auto()          # 正则
    GRAMMAR = enum.auto()        # BNF/Lark 文法
    CHOICE = enum.auto()         # 候选选项
    STRUCTURAL_TAG = enum.auto() # 结构化标签(如 <tool_call>)

StructuredOutputKey = tuple[StructuredOutputOptions, str]  # (类型, 规格字符串)
```

`request.py:82` 的 `get_structured_output_key` 把 `SamplingParams.structured_outputs` 归一成 `(类型, 字符串)` 二元组，作为编译/缓存 key。注意 `choice` 会被转成字面量文法（见下文各后端）。

### 3.2 请求级状态机接口 `StructuredOutputGrammar`

这是整个系统的**状态机抽象**，定义了 6 个方法 (`backend_types.py:31`)：

| 方法 | 语义 | 是否改状态 |
|------|------|-----------|
| `accept_tokens(req_id, tokens)` | 推进状态机（消费 token，转移到下一状态） | **是** |
| `validate_tokens(tokens)` | 验证 token 序列是否合法，返回可接受前缀 | **否**（探测用） |
| `rollback(num_tokens)` | 回滚 N 步状态 | **是**（撤销） |
| `fill_bitmask(bitmask, idx)` | 把"当前状态下合法 token 集合"写入 bitmask 的第 idx 行 | 否（只读状态） |
| `is_terminated()` | 状态机是否到达接受/终止状态 | 否 |
| `reset()` | 重置到初始状态 | 是 |

> 这 6 个方法构成了"可回滚的状态机"接口。`accept`/`rollback`/`fill_bitmask` 三者配合，是支持投机解码的关键（见 §8）。

### 3.3 引擎级后端接口 `StructuredOutputBackend`

```python
# backend_types.py:98
@dataclass
class StructuredOutputBackend(ABC):
    vllm_config / tokenizer / vocab_size
    compile_grammar(request_type, grammar_spec) -> StructuredOutputGrammar  # 工厂方法
    allocate_token_bitmask(max_num_seqs) -> torch.Tensor                    # 分配位掩码缓冲
    destroy()                                                               # 清理
```

---

## 4. 状态机原理(核心)

### 4.1 为什么是状态机

约束解码需要在每步回答："给定前缀，下一个合法 token 是哪些？"

朴素做法是**回放整个前缀**重新计算，开销随长度线性增长。状态机把"前缀 → 合法集合"这个映射**压缩**：

- 文法规格 (JSON Schema/regex/grammar) 编译成一个**自动机**（DFA 或 CFG 的逐字符 matcher）；
- 自动机的**当前状态**唯一确定"接下来允许哪些字符（进而哪些 token）"；
- 前缀的消费过程 = 状态机的状态转移过程。

于是每步只需：`当前状态 → 合法 token 集合`，与历史长度无关。

### 4.2 字符级 vs Token 级

文法/正则天然是**字符级 (character-level)** 的，但 LLM 生成的是 **token**（一个 token 可能对应多个字符，例如 `"name"` 可能是单 token）。桥接方式：

1. **构建"缩减词表" (reduced vocabulary)**：把词表里每个 token 映射到它对应的字节串（`utils.py:308` `_reduced_vocabulary`，处理 BPE/byte-fallback/llama `<0xXX>` 等特殊情况）。
2. 对每个状态，问："从这个状态出发，词表里哪些 token 的字节串是完全合法前缀？" 这些 token 的 bit 置 1。
3. 生成的 bitmask 形状 = `(num_rows, (vocab_size+31)//32)`，每 32 个 token 打包进一个 int32。

> 一个 token 可能"部分合法但不完整"（例如当前状态期望 `"name"`，token ` "na` 是合法前缀）——这类 token **仍然允许**，因为下一步会在新的更精细状态下继续筛选。这就是为什么约束解码需要多步逐步收敛，而不是一步到位。

### 4.3 一个解码步的状态机闭环

```
                 ┌─────────────────────────────┐
   步开始 ──────▶│ 状态机位于状态 S             │
                 └──────────────┬──────────────┘
                                │ fill_bitmask
                                ▼
                 ┌─────────────────────────────┐
                 │ bitmask[idx] = 允许的 token  │  (基于 S)
                 └──────────────┬──────────────┘
                                │ apply to logits  (非法 -> -inf)
                                ▼
                 ┌─────────────────────────────┐
                 │ 采样得到 token t             │
                 └──────────────┬──────────────┘
                                │ accept_tokens([t])
                                ▼
                 ┌─────────────────────────────┐
                 │ 状态机转移 S ──▶ S'          │  (消费 t)
                 └──────────────┬──────────────┘
                                │ is_terminated()?
                                ▼
                       终止? ──▶ 是: 结束请求
                              ─▶ 否: 进入下一步, S'
```

### 4.4 四种"状态机"实现路径

不同后端用不同的自动机/匹配器，但都实现同一个 `StructuredOutputGrammar` 接口：

| 后端 | 规格编译路径 | 状态机载体 |
|------|-------------|-----------|
| **xgrammar** | JSON Schema/regex/grammar → **CFG (BNF)** | `xgr.GrammarMatcher` |
| **outlines** | JSON Schema/choice → **regex** → **DFA** (regex-automata, Rust) | `oc.Guide` (走 DFA) |
| **guidance** | 各规格 → llguidance 序列化文法 → **通用 parser** | `llguidance.LLMatcher` |
| **lm-format-enforcer** | JSON/regex/choice → **字符级 parser** | `lmformatenforcer.TokenEnforcer` |

> xgrammar 和 outlines/guidance 的核心差别：xgrammar 基于**上下文无关文法 (CFG)**，能力更强但匹配开销略高；outlines 把一切先转成**正则**再编译成 **DFA**（确定性有限自动机），状态数固定、查询 O(1)，但正则表达能力弱于 CFG（不支持递归/嵌套结构，故 outlines 不支持 `grammar` 规格）。

### 4.5 三类操作的形式化

设状态机为 `M`，当前状态 `S`：

- **`fill_bitmask(S)`**：计算 `allowed(S) = { token ∈ Vocab | S --token--> S' 存在合法转移 }`，写入位掩码。**只读不改状态。**
- **`accept_tokens([t1..tn])`**：依次执行 `S --t1--> S1 --t2--> ... --> Sn`，若中途任一 token 非法则停止并返回 False。**推进状态。**
- **`rollback(k)`**：从当前状态回退 k 步状态。前三个后端内部维护了**状态历史栈**，lm-format-enforcer 则靠重新基于 `current_tokens_prefix` 计算天然支持回滚。

> `validate_tokens` 与 `accept_tokens` 的区别：`validate` 只探测、不真正推进，用于"假设性"校验（例如检查草稿是否合法），探测完会 `rollback` 回原状态。xgrammar 的实现 (`backend_xgrammar.py:173`) 探测后立即 `self.matcher.rollback(len(accepted))`。

---

## 5. 请求生命周期与状态机交互

### 5.1 文法编译 (grammar_init)

入口 `StructuredOutputManager.grammar_init` (`__init__.py:114`)：

1. 首次调用时按 `_backend` 字段懒加载具体 `Backend` 实例（xgrammar/guidance/outlines/lm-format-enforcer）；
2. 调用 `_create_grammar` (`__init__.py:177`) → `backend.compile_grammar(request_type, grammar_spec)`；
3. **异步编译**：默认提交到 `executor` 线程池返回 `Future`（`__init__.py:167`），避免阻塞调度器；
4. 把 `Future` 存入 `request.structured_output_request.grammar`。

`request.py:50` 的 `_check_grammar_completion` 用 100µs 超时轮询 Future：未就绪则请求停留在 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 状态；编译失败则 Future 携带异常，只让**本请求**失败，不影响其他请求。

> `external_launcher` 分布式后端下会**禁用异步编译**（`__init__.py:52`），因为各 TP rank 的编译完成时间不同会破坏 `external_launcher` 依赖的确定性。

### 5.2 每个调度步：grammar_bitmask

`grammar_bitmask` (`__init__.py:212`) 是调度器在采样前调用的核心方法，负责为本步所有结构化输出请求生成 bitmask。

**处理流程：**

1. 懒分配 bitmask 张量：`max_batch_size * (1 + max_num_spec_tokens)` 行（每行一个解码位置，含投机位置 + 1 个 bonus/非投机位置），见 `__init__.py:225-234`；
2. **大批量 + 非投机**时（批大小 > `fill_bitmask_parallel_threshold=128`），并行分组填充（`__init__.py:244`），用 `executor_for_fillmask` 线程池，每批 16 个；
3. **小批量或投机解码**时走串行路径（`__init__.py:272`）——这条路径尤为重要，因为它要为**每个投机草稿位置**生成独立掩码（见 §8）；
4. 裁剪未用尾部行，转成 `np.ndarray` 返回（numpy 序列化比 tensor 高效，`__init__.py:356`）。

### 5.3 是否填充掩码：should_fill_bitmask

`should_fill_bitmask` (`__init__.py:361`) 决定某请求**本步**是否真的应用语法约束：

- 无 reasoner → 恒为 True（全程约束）；
- 有 reasoner 且 `enable_in_reasoning=True` → True（连推理过程也要约束）；
- 否则取决于 `reasoning_ended`：**推理未结束时不约束**（让模型自由思考，见 §7）。

不约束时，对应行被填成 `full_mask`（全 1，即不屏蔽任何 token，`__init__.py:205`）。

### 5.4 是否推进状态：should_advance

`should_advance` (`__init__.py:381`) 决定采样后**是否把新 token 喂给状态机推进**：

- 无 reasoner → 推进；
- `enable_in_reasoning` → 推进；
- 推理已结束 → 推进；
- 推理在本步**刚好结束**（`is_reasoning_end_streaming` 命中）→ 标记 `reasoning_ended=True`，记录边界 `reasoning_end_token_index`，并推进；
- 推理仍在进行 → **不推进**（这些 token 是思考内容，不属于文法内容）。

> 注意：`grammar_bitmask` 里的"模拟推进"和 `should_advance` 里的"真实推进"是两回事——前者只为生成草稿位置的掩码而临时推进并随后 rollback，后者才是把主模型接受的真实 token 提交给状态机。

---

## 6. 四种后端实现对比

### 6.1 Xgrammar (`backend_xgrammar.py`)

- **引擎级** `XgrammarBackend` (`backend_xgrammar.py:36`)：
  - 用 `xgr.TokenizerInfo` 封装 tokenizer（Mistral 走特殊 `VocabType` 分支，`backend_xgrammar.py:42`）；
  - 构造 `xgr.GrammarCompiler`（`max_threads=8`，带字节级缓存，`backend_xgrammar.py:65`）；
  - `compile_grammar` 按类型分发到 `compile_json_schema`/`compile_grammar`/`compile_regex`/`compile_structural_tag`（`backend_xgrammar.py:78`）。
- **请求级** `XgrammarGrammar` (`backend_xgrammar.py:135`)，核心是 `xgr.GrammarMatcher`：
  - `accept_tokens` → `matcher.accept_token`，记录 `num_processed_tokens`，`backend_xgrammar.py:152`；
  - `fill_bitmask` → `matcher.fill_next_token_bitmask`，`backend_xgrammar.py:195`；
  - `max_rollback_tokens = num_speculative_tokens`，支持投机解码回滚。
- **校验** `validate_xgrammar_grammar` (`backend_xgrammar.py:272`)：把 `choice` 转成 EBNF、检测不支持的 JSON 特性、Lark→EBNF 转换。

特点：基于 CFG，表达能力最强（唯一支持自定义 `grammar` 的"文法"后端之一），Rust 实现，性能好。

### 6.2 Outlines (`backend_outlines.py`)

- **引擎级** `OutlinesBackend` (`backend_outlines.py:53`)：
  - 构建缩减词表 `oc.Vocabulary`（`backend_outlines.py:55`）；
  - `compile_grammar` 把 **JSON Schema 转成正则**（`json_schema.build_regex_from_schema`），`choice` 也转成正则，`backend_outlines.py:73`；
  - `_compile_index` 用 `(vocab_hash, regex)` 作 key 缓存 `oc.Index`（DFA），`backend_outlines.py:58`。
- **请求级** `OutlinesGrammar` (`backend_outlines.py:111`)，核心是 `oc.Guide`：
  - `accept_tokens` 先 `accepts_tokens` 判断，再逐个 `advance`，`backend_outlines.py:123`；
  - `fill_bitmask` → `guide.write_mask_into` 直接写内存，`backend_outlines.py:155`；
  - 终止判定有个**延后一步**的技巧 `_prev_finished`（`backend_outlines.py:159`）：DFA 到达接受态时 vLLM 还想发 EOS，所以把 finished 标志延迟一拍。
- **校验** `validate_structured_output_request_outlines` (`backend_outlines.py:171`) + `validate_regex_is_buildable` (`backend_outlines.py:303`)：
  - 检查正则不含回溯引用、lookaround、unicode 边界（regex-automata 不支持）；
  - 检查"通用起始状态"（不能依赖前文上下文，如 `^` 锚点前置）；
  - **不支持 `grammar` 规格**（正则无法表达 CFG），`backend_outlines.py:200`。

特点：把一切归约为正则→DFA，DFA 查询 O(1) 非常数级稳定，但不支持递归文法。

### 6.3 Guidance / llguidance (`backend_guidance.py`)

- **引擎级** `GuidanceBackend` (`backend_guidance.py:88`)：
  - 构造 `ll_tokenizer`（Mistral/HF/通用三分支，`backend_guidance.py:97`）；
  - `compile_grammar` 把规格序列化为 llguidance 文法 JSON，再 `LLMatcher` 构造匹配器，`backend_guidance.py:108`。
- **请求级** `GuidanceGrammar` (`backend_guidance.py:143`)，核心是 `llguidance.LLMatcher`：
  - `accept_tokens` 处理 EOS 停止、`consume_tokens` 推进，`backend_guidance.py:158`；
  - `validate_tokens` → `ll_matcher.validate_tokens` 返回可接受数量，`backend_guidance.py:186`；
  - `rollback` 有 `rollback_lag` 修正（EOS 的延迟一拍），`backend_guidance.py:203`；
  - `fill_bitmask` → `llguidance_torch.fill_next_token_bitmask`，`backend_guidance.py:210`。
- **特性处理**：`_walk_json_for_additional_properties` 自动给 `properties` 补 `additionalProperties:False`（`backend_guidance.py:36`）；不支持 `patternProperties`。

特点：Rust 实现的通用 parser，支持 JSON/regex/grammar/choice/structural_tag 全类型；预留了 jump-forward（快进）解码接口（`backend_guidance.py:173` 注释）。

### 6.4 LM-Format-Enforcer (`backend_lm_format_enforcer.py`)

- **引擎级** `LMFormatEnforcerBackend` (`backend_lm_format_enforcer.py:95`)：缓存构造 `TokenEnforcerTokenizerData`。
- **请求级** `LMFormatEnforcerGrammar` (`backend_lm_format_enforcer.py:44`)，核心是 `TokenEnforcer` + 自己维护 `current_tokens_prefix`：
  - `accept_tokens`：逐个 `get_allowed_tokens(...).is_token_allowed`，`backend_lm_format_enforcer.py:48`；
  - `fill_bitmask`：直接把 `allowed_tokens` 列表写入 bitmask 行，`backend_lm_format_enforcer.py:76`；
  - `rollback`：截断 `current_tokens_prefix`，`backend_lm_format_enforcer.py:73`。
- **限制**：**不支持投机解码**（`backend_lm_format_enforcer.py:130` 显式报错）；不支持 `grammar`。

特点：纯 Python（带 C 扩展），字符级 parser，实现直观但性能弱于 Rust 系后端，不支持投机解码。

### 6.5 后端能力矩阵

| 能力 | xgrammar | outlines | guidance | lm-format-enforcer |
|------|:-------:|:-------:|:--------:|:-----------------:|
| JSON Schema | ✅ | ✅ | ✅ | ✅ |
| JSON Object | ✅ | ✅ | ✅ | ✅ |
| Regex | ✅ | ✅ | ✅ | ✅ |
| Choice | ✅(转EBNF) | ✅(转regex) | ✅ | ✅ |
| Grammar (BNF/Lark) | ✅ | ❌ | ✅ | ❌ |
| Structural Tag | ✅ | ❌ | ✅ | ❌ |
| 投机解码 (spec decode) | ✅ | ✅ | ✅ | ❌ |
| 实现语言 | Rust(CFG) | Rust(DFA) | Rust(parser) | Python+C |
| 默认/推荐 | ✅ 默认 | | | |

> 从 `__init__.py:133` 的分发顺序可见，xgrammar 是 V1 默认后端。

---

## 7. 推理模式 (reasoning) 与结构化输出的交互

支持"思考型"模型（如 DeepSeek-R1、gpt-oss）时，输出分为 **reasoning 内容**（`<think>...</think>`）和**正式答案**两部分。结构化输出**只应约束正式答案**，不能约束思考过程。

### 7.1 关键字段 (request.py:22)

```python
class StructuredOutputRequest:
    reasoning_ended: bool | None            # 推理是否已结束
    reasoning_end_token_index: int | None   # 推理结束标记的绝对 token 索引
    reasoner: ReasoningParser | None        # 请求级推理解析器(懒构造)
    reasoning_parser_kwargs: dict | None
```

### 7.2 三种配置形态

- **无 reasoner / 无思考**：全程约束，状态机一直推进。
- **`enable_in_reasoning=True`**：连思考过程也受文法约束（少见）。
- **默认思考模式**：思考阶段**不约束 + 不推进**；`reasoner` 检测到推理结束后，从那一刻起开始约束并推进。

### 7.3 三个方法的协作

| 方法 | 作用 | 触发时机 |
|------|------|----------|
| `should_fill_bitmask` | 推理中→返回 False (填 full_mask 不约束) | 采样前生成 bitmask 时 |
| `should_advance` | 推理中→返回 False (不把思考 token 喂状态机) | 采样后决定推进时 |
| `trim_reasoning_for_advance` | 推理在一步中结束时，剔除该步里的思考 token 再推进 | 真正推进前 |

`trim_reasoning_for_advance` (`__init__.py:462`) 解决一个微妙问题：当推理在**某一步中间**结束（思考标记在本步 token 之中），该步的 token 序列前半段是思考内容、后半段才是正式答案。直接整段喂给状态机会被文法拒绝（思考标记不合法）而杀死请求（注释引用 #44006）。因此用 `reasoning_end_token_index` 计算并剔除前缀思考 token，只把答案后缀喂给状态机。

`_find_reasoning_end_index` (`__init__.py:441`) 逐 token 扫描定位推理结束的确切位置；`should_advance` 用 `is_reasoning_end_streaming` 在"增量窗口"上检测结束（注释说明异步调度+投机解码下窗口边界处理的坑，#43388）。

---

## 8. 投机解码集成

投机解码 (speculative decoding) 用小模型/草稿模型一次提议多个 token，主模型并行验证。结构化输出必须为**每个草稿位置**都提供合法掩码——草稿说的是"如果 token t1 被接受、紧接着 t2...，那么每个位置合法集合分别是什么"。

### 8.1 多行 bitmask

`grammar_bitmask` 串行路径 (`__init__.py:272-350`) 的核心循环：

```
对每个请求 req:
  for i, token in enumerate(req_tokens):      # req_tokens = 草稿 tokens
      fill_bitmask(grammar, cumulative_index, apply_bitmask)   # 写当前位置掩码
      if token != -1 and apply_bitmask:
          accept_tokens([token])              # 模拟推进到下一位置
          state_advancements += 1
      cumulative_index += 1
  # bonus / 非投机位置多写一行
  fill_bitmask(grammar, cumulative_index, bonus_apply)
  cumulative_index += 1
  if state_advancements > 0:
      rollback(state_advancements)            # ★ 关键: 全部回滚!
```

**关键点：** 这里对状态机的 `accept_tokens` 只是**模拟推进**，目的是让每个草稿位置的掩码反映"假设前面草稿都被接受"的状态；循环结束后 `rollback(state_advancements)` 把状态机**恢复到本步开始的状态**。真正的推进要等主模型验证后由 `should_advance` 触发——因为草稿可能被拒绝，状态机不能贸然前进。

### 8.2 max_rollback_tokens

各后端在构造 matcher 时传入 `max_rollback_tokens = num_speculative_tokens`：

- xgrammar: `XgrammarBackend.__post_init__` (`backend_xgrammar.py:72`) → `GrammarMatcher(max_rollback_tokens=...)`；
- outlines: `compile_grammar` (`backend_outlines.py:89`) → `Guide(max_rollback=...)`；
- guidance: 内部由 LLMatcher 支持；
- lm-format-enforcer: **不支持**，编译期检测到 `>0` 直接抛错 (`backend_lm_format_enforcer.py:130`)。

`-1` 草稿 token 的处理：代表"无效/被拒绝"位置，置 `apply_bitmask=False` 且不推进（`__init__.py:300`）。

### 8.3 bonus token 行

每个请求在所有草稿位置之后多一行用于 bonus token（非投机场景就是唯一的采样位置）。`bonus_apply` 综合判断：本步开始推理已结束，或推理在窗口中途结束并被翻转（`__init__.py:346`）。扩散 LLM 不采 bonus token，跳过该行 (`__init__.py:335`)。

---

## 9. 编译、缓存与安全

### 9.1 文法编译缓存

- **xgrammar**：编译器内置 `cache_enabled=True`，上限 `VLLM_XGRAMMAR_CACHE_MB`（`backend_xgrammar.py:68`）。
- **outlines**：两级缓存——
  - 内存 LRU（`maxsize=128`）或 SQLite 磁盘缓存（`VLLM_V1_USE_OUTLINES_CACHE`，`utils.py:282`）；
  - `OutlinesDiskCache` (`utils.py:218`) 用 SQLite + outlines_core 原生二进制序列化（Rust serde），**刻意避免 pickle** 以消除反序列化任意代码执行风险；
  - cache key = `vocabulary_hash + regex`，词表变化时自动失效。

### 9.2 词表与缩减词表

`_reduced_vocabulary` (`utils.py:308`) 把 HF tokenizer 的 vocab 归约为 `{bytes: [token_ids]}`，处理：

- BPE 字节型 token；
- llama 风格 `<0xXX>` 单字节 token（`re_llama_byte_token`）；
- GPT2 风格 unicode→bytes 反查（`unicode_to_bytes`）；
- 无效 UTF-8 替换符 `�` 的特殊处理（`re_replacement_seq`）；
- 跳过 special tokens 和 EOS。

词表带 sha256 hash（`OutlinesVocabulary`, `utils.py:178`）作为缓存 key，并缓存到 tokenizer 对象上避免重复计算。

### 9.3 防 ReDoS

`compile_regex_with_timeout` (`utils.py:48`)：用户提供的正则可能含嵌套量词（如 `(a+)+b`）导致 DFA 状态空间指数爆炸。用单线程 `ThreadPoolExecutor` + `VLLM_REGEX_COMPILATION_TIMEOUT_S` 超时保护，超时则取消并抛 `ValueError`。xgrammar、outlines、lm-format-enforcer 的正则编译都经过它。

---

## 10. apply_grammar_bitmask 实现细节

`apply_grammar_bitmask` (`utils.py:86`) 在 **GPU worker 侧**执行，把调度器产出的紧凑 bitmask 应用到模型 logits。

### 10.1 重排序对齐 batch

调度器返回的 bitmask **只含结构化输出请求**，且顺序不一定等于 runner batch 顺序。必须重排：

1. 遍历 `input_batch.req_ids`，按 `cumulative_offset` 累加每个请求的 spec token 数，计算每个结构化请求的 `logit_index`（`utils.py:117-121`）；
2. 按 `grammar_output.structured_output_request_ids` 顺序，把对应行拷到重排后的 `sorted_bitmask`，每个请求占 `1 + num_spec_tokens` 行（`utils.py:134-141`）；
3. 记录 `out_indices`（哪些 logit 行需要应用掩码）。

### 10.2 应用掩码到 logits

调用 `xgr.apply_token_bitmask_in_place(logits, grammar_bitmask, indices)`（`utils.py:161`）：

- **GPU**：`index_tensor` 用 `async_tensor_h2d` 异步拷到 device，避免 CPU 同步；当 `out_indices` 覆盖全部 logit 时 `skip_out_indices` 省去 index 参数。
- **CPU**：用 python list；老版本 xgrammar CPU kernel 要求 float32，做 dtype 转换往返 (`utils.py:168-175`，对应 issue #31901)。

效果：bitmask 中为 0 的位对应 token 的 logit 被置为 `-inf`，采样器无法选到。

> 即便用 outlines/guidance/lm-format-enforcer 后端，最终的 logits 掩码应用仍复用 xgrammar 的 `apply_token_bitmask_inplace` kernel——bitmask 格式是统一的 `(rows, vocab_int32_words)`。

---

## 11. 端到端时序总览

```
请求到达
  │
  ├─ Processor._validate_structured_output
  │    选定 backend 名; 校验规格合法性(各 validate_*_grammar)
  │
  ├─ StructuredOutputRequest.from_sampling_params(request.py:39)
  │    构造请求级对象
  │
  ▼
调度器把请求加入 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  │
  ├─ StructuredOutputManager.grammar_init(__init__.py:114)
  │    懒加载 Backend; 提交 Future 编译文法
  │
  ▼
Future 就绪? 否→等待 / 是→grammar=StructuredOutputGrammar(状态机实例)
  │
  ▼ ───────────── 每个调度步 ─────────────
  │
  ├─ StructuredOutputManager.grammar_bitmask(__init__.py:212)
  │    对每个结构化请求:
  │      should_fill_bitmask? ─ 否→填 full_mask(推理中)
  │                          └ 是→fill_bitmask 写当前状态掩码
  │      (投机解码: 逐草稿 accept_tokens 模拟推进+逐行写掩码, 末尾 rollback)
  │    → 返回 np.ndarray bitmask
  │
  ├─ Scheduler→GPU runner: GrammarOutput 携带 bitmask
  │
  ├─ GPU runner: 模型前向→logits
  │
  ├─ apply_grammar_bitmask(utils.py:86)
  │    重排对齐 batch; apply_token_bitmask_in_place 把非法 logit→-inf
  │
  ├─ 采样(只能采到合法 token)
  │
  ├─ should_advance(__init__.py:381)? ─ 否(推理中)→不推进
  │                                  └ 是→
  │       trim_reasoning_for_advance 剔除思考 token(如需)
  │       grammar.accept_tokens(new_tokens) 真实推进状态机
  │
  ├─ is_terminated()? → 是→结束请求
  │
  ▼ ─────────────────────────────────────────
  下一步(状态机已在 S')
```

---

## 12. 关键文件索引

| 文件 | 作用 | 关键符号 |
|------|------|----------|
| `__init__.py` | 引擎级管理器 | `StructuredOutputManager` |
| `backend_types.py` | 两层抽象基类 + 类型枚举 | `StructuredOutputGrammar` / `StructuredOutputBackend` / `StructuredOutputOptions` |
| `request.py` | 请求级对象 | `StructuredOutputRequest` / `get_structured_output_key` |
| `utils.py` | 通用工具(掩码应用、缓存、词表、正则安全、Lark转换) | `apply_grammar_bitmask` / `_reduced_vocabulary` / `OutlinesDiskCache` / `compile_regex_with_timeout` / `convert_lark_to_ebnf` |
| `backend_xgrammar.py` | xgrammar 后端(默认) | `XgrammarBackend` / `XgrammarGrammar` / `validate_xgrammar_grammar` |
| `backend_outlines.py` | outlines 后端(regex→DFA) | `OutlinesBackend` / `OutlinesGrammar` / `validate_regex_is_buildable` |
| `backend_guidance.py` | llguidance 后端(通用 parser) | `GuidanceBackend` / `GuidanceGrammar` / `serialize_guidance_grammar` |
| `backend_lm_format_enforcer.py` | lm-format-enforcer 后端(字符级) | `LMFormatEnforcerBackend` / `LMFormatEnforcerGrammar` |

---

## 附：核心心智模型一句话总结

> **结构化输出 = 把规格编译成"可回滚的状态机"，每个解码步用状态机算出合法 token 位掩码屏蔽 logits，采样后再用真实 token 推进状态机；投机解码靠"模拟推进 + rollback"为每个草稿位置生成掩码，推理模式靠"延迟推进 + 剔除思考 token"让约束只作用于正式答案。**
