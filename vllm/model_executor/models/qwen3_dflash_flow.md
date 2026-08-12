# vLLM 跑 Qwen3.5 + DFlash 完整流程详解

> 本文档基于源代码 [qwen3_dflash.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py)、[qwen3_5.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_5.py) 以及 vLLM V1 speculative decoding 框架整理。

---

## 目录

1. [DFlash 是什么](#1-dflash-是什么)
2. [整体架构概览](#2-整体架构概览)
3. [模型启动与初始化流程](#3-模型启动与初始化流程)
4. [权重加载流程](#4-权重加载流程)
5. [核心机制详解](#5-核心机制详解)
6. [推理全流程（Decode 阶段）](#6-推理全流程decode-阶段)
7. [关键数据结构与 Tensor Shape](#7-关键数据结构与-tensor-shape)
8. [配置参数说明](#8-配置参数说明)
9. [代码调用关系图](#9-代码调用关系图)

---

## 1. DFlash 是什么

**DFlash** 是一种用于 speculative decoding（投机解码）的 draft model 架构，专为 Qwen3 系列模型设计。它的核心思想是：

- **Target Model（目标模型）**：Qwen3.5，大模型，负责最终验证
- **Draft Model（草稿模型）**：DFlashQwen3ForCausalLM，小模型，负责快速生成候选 token

与传统的 EAGLE/MTP 等投机解码方法不同，DFlash 有几个关键特点：

| 特点 | 说明 |
|------|------|
| **Mask Token 并行草稿** | 使用一个特殊的 `[MASK]` token 占位，一次 forward 预测 N 个未来 token |
| **Context KV 预计算** | Target model 的 hidden states 直接投影成 draft model 的 K/V，提前写入 KV cache |
| **非因果注意力** | Draft 层内部使用 bidirectional attention（非因果），让 mask token 之间互相可见 |
| **单次 Forward** | 所有 N 个 speculative token 在一次 forward 中同时预测，不需要自回归循环 |
| **EAGLE3 特征融合** | 支持使用 target model 多层 hidden states 拼接作为 draft 输入 |

### 生活化类比

把 LLM 推理想象成"考试做阅读理解题"：

- **Target model** = 学霸，做题慢但准确率高
- **Draft model (DFlash)** = 学霸的考前小抄，提前把上下文的关键信息（KV）整理好
- **Context KV 预计算** = 小抄提前把课文重点抄好，考试时直接看
- **Mask token** = 小抄上留的空格，一次猜 N 个答案
- **非因果注意力** = 猜答案时可以同时看所有空格，互相参考
- **投机解码验证** = 学霸快速检查小抄的答案，对的直接用，错的自己改

---

## 2. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM Engine                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │   Target Model       │      │    DFlash Draft Model        │ │
│  │   Qwen3.5ForCausalLM │      │    DFlashQwen3ForCausalLM   │ │
│  │                      │      │                              │ │
│  │  ┌────────────────┐  │      │  ┌────────────────────────┐  │ │
│  │  │ embed_tokens   │  │      │  │ embed_tokens (共享)    │  │ │
│  │  ├────────────────┤  │      │  ├────────────────────────┤  │ │
│  │  │ Layer 0..N     │  │      │  │ DFlash Layer 0..M      │  │ │
│  │  │ (混合注意力)    │  │      │  │ (非因果注意力)          │  │ │
│  │  │ - linear_attn  │  │      │  │ - precomputed KV cache │  │ │
│  │  │ - full_attn    │  │      │  │ - query only forward   │  │ │
│  │  ├────────────────┤  │      │  ├────────────────────────┤  │ │
│  │  │ norm + lm_head │  │      │  │ fc (特征融合)          │  │ │
│  │  └────────────────┘  │      │  │ norm + lm_head (共享)  │  │ │
│  └──────────┬───────────┘      │  └────────────────────────┘  │ │
│             │                  └──────────────┬───────────────┘ │
│             │ hidden_states + aux_hidden      │                 │
│             ▼                                 ▼                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              DFlashProposer / Speculator                 │    │
│  │  1. 准备 context positions + slot mapping               │    │
│  │  2. 调用 precompute_and_store_context_kv()              │    │
│  │  3. 构造 query tokens (bonus + N masks)                 │    │
│  │  4. 运行 draft model forward                            │    │
│  │  5. 采样 N 个 draft tokens                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心文件清单

| 文件路径 | 作用 |
|---------|------|
| [qwen3_5.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_5.py) | Target 模型（Qwen3.5），混合线性注意力+全注意力 |
| [qwen3_dflash.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py) | Draft 模型（DFlash），含 KV 预计算优化 |
| [dflash.py (proposer)](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/spec_decode/dflash.py) | Scheduler 侧的 DFlash proposer |
| [speculator.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py) | Worker 侧的 DFlash speculator，含 Triton kernel |
| [utils.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/utils.py) | Draft 模型加载、embedding/lm_head 共享逻辑 |
| [cudagraph.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py) | DFlash CUDA Graph 管理 |

---

## 3. 模型启动与初始化流程

### 3.1 配置识别

启动命令示例：

```bash
vllm serve z-lab/Qwen3.5-9B-DFlash-Target \
  --speculative-model z-lab/Qwen3.5-9B-DFlash \
  --speculative-method dflash \
  --num-speculative-tokens 5
```

当 `speculative_config.method == "dflash"` 时：

1. **Worker 侧**：[spec_decode/__init__.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/__init__.py) 根据 method 创建 `DFlashSpeculator`
2. **Scheduler 侧**：创建 `DFlashProposer`

### 3.2 DFlashSpeculator 初始化

位置：[speculator.py#L31-L96](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L31-L96)

```python
class DFlashSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)

        # hidden states 缓冲区：存储 target model 输出
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, ...)

        # 每个 request 输出 (1 bonus + N mask) 个 query token
        self.num_query_per_req = 1 + self.num_speculative_steps

        # mask token 的 token id
        self.parallel_drafting_token_id = get_parallel_drafting_token_id(...)

        # 检查是否需要非因果注意力后端
        self.requires_non_causal = dflash_has_any_non_causal(
            self.draft_model_config.hf_config)

        # context positions 缓冲区（用于 KV 预计算）
        self.context_positions = torch.zeros(self.max_num_tokens, ...)

        # 采样相关缓冲区
        self.sample_indices = torch.zeros(max_num_sampled_tokens, ...)
        self.sample_pos = torch.zeros(max_num_sampled_tokens, ...)
        self.sample_idx_mapping = torch.zeros(max_num_sampled_tokens, ...)
        self.sample_col = torch.arange(self.num_speculative_steps).repeat(max_num_reqs)
```

### 3.3 Draft 模型加载

位置：[utils.py#L14-L74](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/utils.py#L14-L74)

```python
def load_dflash_model(target_model, vllm_config):
    # 1. 根据 draft config 创建 attention 配置（支持 non-causal）
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(..., use_non_causal=..., backend=...),
        cache_config=replace(..., cache_dtype=...),
    )

    # 2. 加载 DFlash draft 模型
    with set_model_tag("dflash_head"):
        dflash_model = get_model(vllm_config=draft_vllm_config, ...)

    # 3. 共享 embedding（省显存）
    if get_pp_group().world_size == 1:
        if _should_share(...):
            draft_inner.embed_tokens = target_embed

    # 4. 共享 lm_head（省显存）
    if _should_share(...):
        dflash_model.lm_head = target_lm_head

    return dflash_model
```

### 3.4 DFlashQwen3ForCausalLM 初始化

位置：[qwen3_dflash.py#L664-L696](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L664-L696)

```python
class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config

        # 核心：创建 DFlashQwen3Model
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,  # 层号偏移，避免与 target 冲突
        )

        # lm_head（可能与 target 共享）
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            ...)
        self.logits_processor = LogitsProcessor(...)

        # 如果 draft vocab 和 target vocab 大小不同，需要映射表
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(...)
```

### 3.5 DFlashQwen3Model 初始化

位置：[qwen3_dflash.py#L346-L430](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L346-L430)

关键组件：

| 组件 | 作用 |
|------|------|
| `embed_tokens` | Token embedding（可能与 target 共享） |
| `mask_embedding` | 可选的独立 mask token embedding（部分 checkpoint 有） |
| `layers` | `DFlashQwen3DecoderLayer` 列表 |
| `fc` | EAGLE3 特征融合层，将多层 aux hidden states 投影到 hidden_size |
| `hidden_norm` | Context KV 预计算前使用的 RMSNorm |
| `norm` | 最终输出 RMSNorm |

### 3.6 注意力层因果性判定

位置：[qwen3_dflash.py#L58-L146](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L58-L146)

函数 `_resolve_layer_attention(config, layer_idx)` 决定每层的注意力模式：

```
┌──────────────────────┬──────────────────┬──────────────────────┐
│ Config               │ layer_type       │ causal               │
├──────────────────────┼──────────────────┼──────────────────────┤
│ layer_types 有设置    │ SWA 层           │ True（因果）          │
│                      │ Full 层          │ False（非因果）       │
├──────────────────────┼──────────────────┼──────────────────────┤
│ layer_types=None     │ 根据 use_swa     │ False（非因果）       │
│ + use_swa=True       │ 全部 SWA         │                      │
├──────────────────────┼──────────────────┼──────────────────────┤
│ dflash_config.causal │ 覆盖所有层       │ 使用配置指定的值      │
│ 显式设置             │                  │                      │
└──────────────────────┴──────────────────┴──────────────────────┘
```

> **注意**：标准的 Qwen3.5-DFlash checkpoint（z-lab/Qwen3.5-9B-DFlash）使用全 full attention，所有层都是**非因果**的。

---

## 4. 权重加载流程

位置：[qwen3_dflash.py#L772-L816](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L772-L816)

### 4.1 权重名称映射

DFlash 复用了 EAGLE 的权重命名约定，通过 `hf_to_vllm_mapper` 进行转换：

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_substr={"midlayer.": "layers.0."},
    orig_to_new_stacked={
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    },
)
```

### 4.2 特殊权重处理

`load_weights` 方法处理：

1. **`d2t` 权重**：重命名为 `draft_id_to_target_id`（vocab 映射表）
2. **`t2d` 权重**：跳过（target-to-draft，不需要）
3. **非 lm_head 权重**：添加 `model.` 前缀
4. **mask_embedding.pt**：如果 checkpoint 中包含单独的 mask embedding 文件，加载它
5. **跳过缺失权重**：根据配置动态跳过 `fc`、`embed_tokens`、`draft_id_to_target_id` 等

### 4.3 Fused KV Buffer 构建

权重加载完成后调用 `_build_fused_kv_buffers()`（[qwen3_dflash.py#L462-L503](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L462-L503)）：

这是 DFlash 性能优化的关键！将所有层的 KV 投影权重融合成一个大矩阵：

```
原始：每层独立计算 KV = hidden @ W_kv^T
      Layer 0: [kv_size, hidden_size]
      Layer 1: [kv_size, hidden_size]
      ...
      Layer L: [kv_size, hidden_size]

融合后：一次 GEMM 计算所有层的 KV
      _fused_kv_weight: [L * 2 * kv_size, hidden_size]
      _fused_kv_bias:   [L * 2 * kv_size]
      _k_norm_weights:  [L, head_dim]  (K-norm 权重堆叠)
```

---

## 5. 核心机制详解

### 5.1 Mask Token 机制

DFlash 使用特殊的 `[MASK]` token（`parallel_drafting_token_id`）来表示"待预测位置"。

每个 decode step，每个 request 的 draft 输入布局：

```
位置:     [ctx_0, ctx_1, ..., ctx_n,  bonus,  mask_1, mask_2, ..., mask_N]
                                                                        ↑
Input IDs: [...target tokens...,  last_token,  MASK,   MASK,  ..., MASK]
角色:     └────── Context ──────┘  └──────── Query (1+N tokens) ────────┘
                                    │         │       │              │
                                    │         │       └─ 预测 token 3 │
                                    │         └─ 预测 token 2         │
                                    └─ bonus = 已确定的最新 token     └─ 预测 token N+1
```

关键点：
- **Context tokens 的 KV 来自 target model**，提前预计算并写入 cache
- **Query tokens 才是 draft model 实际需要 forward 的**
- **Bonus token** 位置放的是 target model 最新采样出的 token（已知）
- **Mask tokens** 位置放的是 `[MASK]`，它们的 hidden states 通过非因果注意力互相看到

### 5.2 Context KV 预计算（核心优化）

位置：[qwen3_dflash.py#L548-L619](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L548-L619)

这是 DFlash 最核心的优化。传统 draft model 需要自己对 context tokens 做完整 forward 来构建 KV cache，但 DFlash 直接用 **target model 的 hidden states** 投影出 draft model 的 K/V：

```
Target model hidden_states [num_ctx, hidden_size]
           │
           ▼
    ┌──────────────┐
    │ hidden_norm  │  RMSNorm
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Fused GEMM   │  一次矩阵乘法算出所有层的 K 和 V
    │ (all layers) │  output: [num_ctx, L*2*kv_size]
    └──────┬───────┘
           │ reshape + permute
           ▼
    all_k [L, num_ctx, nkv, hd], all_v [L, num_ctx, nkv, hd]
           │
           ▼
    ┌──────────────┐
    │ Grouped      │  对 K 做 RMSNorm（所有层一次 kernel）
    │ K-RMSNorm    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Fused RoPE   │  对所有层的 K 应用旋转位置编码
    └──────┬───────┘
           │
           ▼
    逐层写入 KV cache（通过 do_kv_cache_update）
```

**为什么能这么做？**

因为 DFlash draft model 的结构和 target model 相似（都是 Qwen3 架构），但 draft model 的 KV 投影权重是独立训练的。Target model 的 hidden states 经过 draft model 自己的 KV 投影层，就能得到 draft model 视角下的 K/V，无需重新跑一遍 draft model 的前几层。

### 5.3 非因果注意力（Bidirectional Attention）

传统 decoder-only 模型使用因果掩码：token i 只能看到位置 ≤ i 的 token。

DFlash draft model 使用非因果注意力的原因：

```
因果注意力（传统）：          非因果注意力（DFlash）：
                            ┌─────────────────────┐
bonus   ──▶ mask_1          bonus   ◀──▶ mask_1   │
  │         │                │  │      │  │       │
  ▼         ▼                │  ▼      ▼  │       │
mask_1  ──▶ mask_2          mask_1  ◀──▶ mask_2   │
  │         │                │  │      │  │       │
  ▼         ▼                │  ▼      ▼  │       │
mask_2  ──▶ mask_3          mask_2  ◀──▶ mask_3   │
                            └─────────────────────┘
每个 mask 只能看前面的         所有 mask 可以互相看到，
信息，串行依赖                 支持并行预测
```

由于 mask tokens 都是"未来位置"，它们之间没有真实的因果关系——都是需要预测的未知 token。让它们互相注意可以提供更丰富的上下文信息，提高预测准确率。

代码中通过 `causal=False` 传递给 attention 后端（[qwen3_dflash.py#L234](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L234)）。

### 5.4 EAGLE3 特征融合（fc 层）

位置：[qwen3_dflash.py#L750-L770](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L750-L770)

当 `use_aux_hidden_state=True`（默认）时，DFlash 使用 target model 多层的 hidden states 拼接后通过 `fc` 层投影：

```python
def combine_hidden_states(self, hidden_states):
    # hidden_states: [num_tokens, target_hidden_size * num_aux_layers]
    # fc 输入大小 = target_hidden_size × num_features_to_use
    result = self.model.fc(hidden_states)  # → [num_tokens, draft_hidden_size]
    return result
```

辅助层的选择由 `get_eagle3_aux_layers_from_config` 决定，默认使用 target model 的第 2 层、中间层、倒数第 3 层。

Target 模型侧通过 `SupportsEagle3` 接口在这些层输出 hidden states（见 [qwen3_next.py#L696-L698](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_next.py#L696-L698)）。

### 5.5 独立 Mask Embedding

部分 checkpoint（如 XiaomiMiMo）附带 `mask_embedding.pt` 文件，包含专门训练的 mask token embedding。加载后在 `embed_input_ids` 中替换 `embed_tokens[mask_token_id]` 的值（[qwen3_dflash.py#L432-L438](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L432-L438)）。

---

## 6. 推理全流程（Decode 阶段）

下面详细描述一个 decode step 中，从 target model 输出到 draft tokens 生成的完整过程。

### 步骤 1：Target Model Forward

```
输入: input_ids, positions, ...
  ↓
Qwen3.5ForCausalLM.forward()
  ↓
Qwen3_5Model.forward()
  ├─ embed_tokens
  ├─ 逐层 forward（线性注意力层 + 全注意力层混合）
  ├─ 在 aux_hidden_state_layers 指定层收集 hidden states
  └─ norm
  ↓
输出:
  - last_hidden_states: [num_tokens, hidden_size]
  - aux_hidden_states: list of [num_tokens, hidden_size]（可选）
```

Target 模型同时输出：
- 采样后的 `next_token_ids`（bonus token）
- `last_hidden_states`
- `aux_hidden_states`（如果启用）

### 步骤 2：Propose 入口

位置：[speculator.py#L299-L469](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L299-L469)

```python
def propose(self, input_batch, attn_metadata, slot_mappings,
            last_hidden_states, aux_hidden_states,
            num_sampled, num_rejected, last_sampled, next_prefill_tokens,
            temperature, seeds, ...):

    num_reqs = input_batch.num_reqs
    num_target_tokens = input_batch.num_tokens
    num_query_tokens = num_reqs * self.num_query_per_req  # B * (1+N)
```

### 步骤 3：特征融合（如有 aux_hidden_states）

```python
if aux_hidden_states:
    # 拼接多层特征: [num_tokens, hidden_size * num_layers]
    hidden_states = self.model.combine_hidden_states(
        torch.cat(aux_hidden_states, dim=-1)
    )
else:
    hidden_states = last_hidden_states

# 复制到 speculator 缓冲区
self.hidden_states[:num_target_tokens].copy_(hidden_states)
```

### 步骤 4：Triton Kernel 准备输入

位置：[speculator.py#L621-L687](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L621-L687)

调用 `prepare_dflash_inputs()`，启动 `_prepare_dflash_inputs_kernel` Triton kernel，并行处理每个 request：

对于每个 request，kernel 输出：

| 输出 Buffer | 内容 | Shape |
|------------|------|-------|
| `input_ids` | query token IDs：`[bonus, MASK, MASK, ...]` | `[B*(1+N)]` |
| `positions`（query） | query 位置：`[last_pos+1, last_pos+2, ...]` | `[B*(1+N)]` |
| `query_slot_mapping` | query token 对应的 KV cache 槽位 | `[B*(1+N)]` |
| `context_positions` | context token 的位置 | `[num_ctx]` |
| `context_slot_mapping` | context token 对应的 KV cache 槽位 | `[num_ctx]` |
| `sample_indices` | 哪些 query 位置需要采样（mask 位置） | `[B*N]` |
| `sample_pos` | 采样位置（用于 Gumbel 采样 verification key） | `[B*N]` |
| `sample_idx_mapping` | 每个采样属于哪个 request | `[B*N]` |

**Kernel 内部逻辑（每个 request 一个 program）：**

```python
# 1. 获取 context 范围
ctx_start = query_start_loc[req_idx]
ctx_end = query_start_loc[req_idx + 1]
num_ctx = ctx_end - ctx_start
valid_ctx_end = ctx_end - num_rejected  # 排除被拒绝的 token

# 2. 获取 bonus token
if num_sampled > 0:
    bonus_token = last_sampled[req_state_idx]
else:
    bonus_token = next_prefill_tokens[req_state_idx]  # chunked prefill

# 3. 填充 context positions 和 slot mapping
for j in ctx positions:
    ctx_pos = target_positions[ctx_start + j]
    ctx_block = block_table[req_idx][ctx_pos // block_size]
    ctx_slot = ctx_block * block_size + ctx_pos % block_size
    store context_positions, context_slot_mapping

# 4. 填充 query tokens
last_valid_pos = target_positions[valid_ctx_end - 1]
for q_off in 0..num_query_per_req:
    query_pos = last_valid_pos + 1 + q_off
    if q_off == 0:
        input_id = bonus_token      # 第一个位置放 bonus
    else:
        input_id = MASK_TOKEN       # 其余位置放 mask

    q_block = block_table[req_idx][query_pos // block_size]
    q_slot = q_block * block_size + query_pos % block_size
    store input_ids, query_positions, query_slot_mapping

# 5. 填充采样索引（只在 mask 位置采样，bonus 位置不采样）
for q_off in 1..N:
    sample_idx = req_idx * N + (q_off - 1)
    sample_indices[sample_idx] = query_base + q_off
    sample_pos[sample_idx] = query_pos   # DFlash: 在 mask 自身位置采样
    sample_idx_mapping[sample_idx] = req_state_idx
```

### 步骤 5：Context KV 预计算并写入 Cache

位置：[speculator.py#L404-L421](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L404-L421)

```python
# 根据是否有多层 KV cache group，确定 context_slots
if self._layer_group_idx is not None:
    context_slots = [
        self._context_slot_mappings[gidx][:num_target_tokens]
        for gidx in self._layer_group_idx
    ]
else:
    context_slots = self._context_slot_mappings[0][:num_target_tokens]

# 核心调用：预计算 KV 并写入 cache
self.model.precompute_and_store_context_kv(
    self.hidden_states[:num_target_tokens],   # target hidden states
    self.context_positions[:num_target_tokens], # context 位置
    context_slots,                              # KV cache 槽位
)
```

`precompute_and_store_context_kv` 内部流程（[qwen3_dflash.py#L548-L619](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L548-L619)）：

1. **RMSNorm**：对 context hidden states 做 `hidden_norm`
2. **Fused GEMM**：一次线性变换算出所有层的 K+V
3. **Reshape**：将 flat output 拆成 `all_k [L, num_ctx, nkv, hd]` 和 `all_v [L, num_ctx, nkv, hd]`
4. **K Norm**：对所有层的 K 做 grouped RMSNorm
5. **Fused RoPE**：将所有层的 K 展平成 `[L*num_ctx, kv]`，一次 RoPE kernel 处理
6. **写入 Cache**：逐层调用 `attn.impl.do_kv_cache_update()` 将 K/V 写入对应层的 KV cache

> 注意：V 不需要做 Norm 和 RoPE，只有 K 需要。

### 步骤 6：Draft Model Forward（Query Only）

位置：[qwen3_dflash.py#L621-L640](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L621-L640)

```python
def forward(self, input_ids, positions, input_embeds=None):
    if input_embeds is None:
        input_embeds = self.embed_input_ids(input_ids)

    hidden_states = input_embeds
    residual = None

    # 逐层 forward，只处理 query tokens（context KV 已在 cache 中）
    for layer in self.layers:
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states
```

`DFlashQwen3DecoderLayer.forward`（[qwen3_dflash.py#L323-L342](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L323-L342)）：

```python
def forward(self, positions, hidden_states, residual):
    # 1. Input RMSNorm
    if residual is not None:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    else:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

    # 2. Self Attention（只计算 Q，KV 从 cache 读取）
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    # 3. Post-attention RMSNorm + MLP
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

`DFlashQwen3Attention.forward`（[qwen3_dflash.py#L238-L263](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_dflash.py#L238-L263)）：

```python
def forward(self, positions, hidden_states):
    # 只对 query tokens 做 QKV 投影
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # Per-head RMSNorm（QK-norm）
    q = self.q_norm(q.view(..., num_heads, head_dim)).view(q_shape)
    k = self.k_norm(k.view(..., num_kv_heads, head_dim)).view(k_shape)

    # RoPE
    q, k = self.rotary_emb(positions, q, k)

    # Attention: Q 来自当前输入，K/V 来自 cache（含预计算的 context KV）
    attn_output = self.attn(q, k, v)

    output, _ = self.o_proj(attn_output)
    return output
```

> 注意：这里的 attention 会从 KV cache 中读取**所有 context token 的 K/V**（步骤 5 预计算的）加上**当前 query tokens 的 K/V**，然后执行非因果注意力。

### 步骤 7：采样 Draft Tokens

位置：[speculator.py#L242-L274](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py#L242-L274)

```python
def _generate_draft(self, num_reqs, num_tokens_padded, attn_metadata, ...):
    # 1. 运行 draft model forward
    last_hidden_states = self._run_model(...)

    # 2. 取出 mask 位置的 hidden states（跳过 bonus 位置）
    num_sample = num_reqs * self.num_speculative_steps
    sample_hidden_states = last_hidden_states[
        self.sample_indices[:num_sample]
    ]

    # 3. Gumbel 采样，生成 draft tokens
    draft_tokens = self.sample_draft(
        sample_hidden_states,
        self.sample_pos[:num_sample] - 2,  # verification key position
        self.sample_idx_mapping[:num_sample],
        self.temperature,
        self.seeds,
        self.sample_col[:num_sample],      # per-token 列索引
        self.draft_logits,
    )

    # 4. reshape 成 [num_reqs, num_speculative_steps]
    self.draft_tokens[:num_reqs] = draft_tokens.view(
        num_reqs, self.num_speculative_steps
    )
```

### 步骤 8：Target Model 验证

生成的 `draft_tokens [B, N]` 返回给 vLLM 的 speculative decoding 框架，框架会：

1. 将 bonus + N 个 draft tokens 拼成一个 sequence
2. 运行 target model 一次 forward 验证所有 token
3. 检查每个位置 target 采样结果是否与 draft 一致
4. 接受匹配的 token，拒绝第一个不匹配位置之后的所有 token
5. 被拒绝的 token 数量通过 `num_rejected` 传回，影响下一轮 context 的有效长度

### 6.1 Prefill 阶段

Prefill 阶段 DFlash 不生成 draft tokens（因为需要先有 target hidden states）。但 `dummy_run` 会执行内存预分配和 CUDA graph capture：

位置：[dflash.py#L205-L262](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/spec_decode/dflash.py#L205-L262)

```python
def dummy_run(self, num_tokens, use_cudagraphs=True, ...):
    # 1. 确定 query token 数量（远小于 context）
    num_query_tokens = min(num_tokens, self.max_query_tokens)

    # 2. 运行 KV 预计算（内存分析用）
    context_states = self.hidden_states[:num_tokens]
    context_positions = self._context_positions_buffer[:num_tokens]
    self.model.precompute_and_store_context_kv(context_states, context_positions)

    # 3. 运行 draft model forward（CUDA graph capture）
    with set_forward_context(...):
        self.model(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self._get_positions(num_input_tokens),
        )
```

### 6.2 CUDA Graph 优化

位置：[cudagraph.py](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py)

DFlash 使用 `FULL_DECODE_ONLY` CUDA graph 模式：
- Context KV 预计算**不**在 CUDA graph 中（因为 context 长度每步变化）
- Query forward（固定 `1+N` tokens per request）**在** CUDA graph 中
- 这样既享受了 CUDA graph 的延迟优化，又避免了动态 context 长度的问题

---

## 7. 关键数据结构与 Tensor Shape

假设：
- Batch size = B
- Target context tokens = T（所有 request 的 token 总数）
- Speculative tokens = N
- Draft hidden size = H_d
- Target hidden size = H_t
- Num attention heads = n_heads
- Num KV heads = n_kv
- Head dim = hd
- Num draft layers = L

### 7.1 Target 输出

| Tensor | Shape | 说明 |
|--------|-------|------|
| `last_hidden_states` | `[T, H_t]` | Target model 最后一层输出 |
| `aux_hidden_states` | list of `[T, H_t]` | EAGLE3 辅助层输出（可选） |
| `next_token_ids` | `[B]` | Bonus token IDs |

### 7.2 融合后 Hidden States

| Tensor | Shape | 说明 |
|--------|-------|------|
| `hidden_states` (after fc) | `[T, H_d]` | 输入给 KV 预计算的 context states |

### 7.3 Context KV 预计算中间结果

| Tensor | Shape | 说明 |
|--------|-------|------|
| `normed_context_states` | `[T, H_d]` | RMSNorm 后 |
| `all_kv_flat` | `[T, L*2*n_kv*hd]` | Fused GEMM 输出 |
| `all_k` | `[L, T, n_kv, hd]` | 所有层的 K |
| `all_v` | `[L, T, n_kv, hd]` | 所有层的 V |
| `all_k_normed` | `[L, T, n_kv, hd]` | K-norm 后 |
| `all_k_flat` | `[L*T, n_kv*hd]` | RoPE 前展平 |

### 7.4 Query Tokens

| Tensor | Shape | 说明 |
|--------|-------|------|
| `input_ids` | `[B*(1+N)]` | Query token IDs（bonus + masks） |
| `positions` | `[B*(1+N)]` | Query 位置编码 |
| `slot_mapping` | `[B*(1+N)]` | Query 的 KV cache 槽位 |

### 7.5 Draft 输出

| Tensor | Shape | 说明 |
|--------|-------|------|
| `last_hidden_states` | `[B*(1+N), H_d]` | Draft model 输出 |
| `sample_hidden_states` | `[B*N, H_d]` | mask 位置的 hidden states |
| `draft_tokens` | `[B, N]` | 最终采样的 draft tokens |

---

## 8. 配置参数说明

### 8.1 dflash_config（Draft 模型 config.json 中）

```json
{
  "dflash_config": {
    "mask_token_id": 151668,
    "use_swa": false,
    "swa_window_size": null,
    "causal": null,
    "attention_sink_bias": false,
    "use_aux_hidden_state": true
  }
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mask_token_id` | int | 必填 | `[MASK]` token 的 ID |
| `use_swa` | bool | false | 是否启用滑动窗口注意力 |
| `swa_window_size` | int | null | SWA 窗口大小 |
| `causal` | bool/null | null | 是否强制所有层因果注意力（null=自动判断） |
| `attention_sink_bias` | bool | false | 是否使用 attention sink bias |
| `use_aux_hidden_state` | bool | true | 是否使用 EAGLE3 多层特征融合 |

### 8.2 eagle_config（复用 EAGLE 配置）

DFlash 复用 EAGLE3 的配置结构：

| 参数 | 说明 |
|------|------|
| `target_layer_ids` | Draft 从 target 哪些层取 hidden states |
| `draft_vocab_size` | Draft 模型的 vocab 大小（可能与 target 不同） |
| `logit_scale` | Logits 缩放因子 |

### 8.3 启动参数

| 参数 | 说明 |
|------|------|
| `--speculative-model` | Draft 模型路径 |
| `--speculative-method dflash` | 指定使用 DFlash 方法 |
| `--num-speculative-tokens N` | 每步预测 N 个 token |
| `--speculative-attention-backend` | Draft 模型使用的 attention 后端 |

---

## 9. 代码调用关系图

### 9.1 初始化阶段

```
vLLM Engine 启动
    │
    ├─▶ 创建 DFlashSpeculator (worker 侧)
    │       ├─▶ load_dflash_model()
    │       │       ├─▶ get_model() → DFlashQwen3ForCausalLM
    │       │       │       └─▶ DFlashQwen3Model
    │       │       │               ├─▶ embed_tokens
    │       │       │               ├─▶ layers × L (DFlashQwen3DecoderLayer)
    │       │       │               ├─▶ fc (EAGLE3 融合)
    │       │       │               ├─▶ hidden_norm, norm
    │       │       │               └─▶ mask_embedding (可选)
    │       │       ├─▶ 共享 target embed_tokens
    │       │       └─▶ 共享 target lm_head
    │       ├─▶ init_cudagraph_manager()
    │       └─▶ capture() → dummy_run()
    │               ├─▶ precompute_and_store_context_kv()
    │               └─▶ model.forward() [CUDA Graph capture]
    │
    └─▶ 创建 DFlashProposer (scheduler 侧)
            └─▶ _create_draft_vllm_config()
                    └─▶ 设置 use_non_causal
```

### 9.2 Decode 阶段

```
Target Model forward
    │
    ├─▶ 输出 last_hidden_states + aux_hidden_states
    │
    ▼
DFlashSpeculator.propose()
    │
    ├─▶ 1. combine_hidden_states() (EAGLE3 特征融合)
    │       └─▶ model.fc() → hidden_states
    │
    ├─▶ 2. prepare_dflash_inputs() [Triton kernel]
    │       ├─▶ 生成 context_positions, context_slot_mapping
    │       ├─▶ 生成 query input_ids [bonus, MASK, ...]
    │       ├─▶ 生成 query positions, slot_mapping
    │       └─▶ 生成 sample_indices, sample_pos
    │
    ├─▶ 3. model.precompute_and_store_context_kv()
    │       ├─▶ RMSNorm(hidden_states)
    │       ├─▶ Fused GEMM → all_k, all_v
    │       ├─▶ Grouped K-RMSNorm
    │       ├─▶ Fused RoPE on K
    │       └─▶ 逐层写入 KV cache
    │
    ├─▶ 4. _generate_draft()
    │       ├─▶ _run_model()
    │       │       └─▶ DFlashQwen3ForCausalLM.forward()
    │       │               └─▶ DFlashQwen3Model.forward()
    │       │                       ├─▶ embed_input_ids()
    │       │                       └─▶ 逐层 DFlashQwen3DecoderLayer
    │       │                               ├─▶ input_layernorm
    │       │                               ├─▶ DFlashQwen3Attention
    │       │                               │       ├─▶ qkv_proj (只算 Q 部分有意义)
    │       │                               │       ├─▶ q_norm, k_norm
    │       │                               │       ├─▶ rotary_emb
    │       │                               │       └─▶ attn() (读取预计算 KV)
    │       │                               ├─▶ post_attention_layernorm
    │       │                               └─▶ mlp
    │       │
    │       └─▶ sample_draft() → draft_tokens [B, N]
    │
    └─▶ 返回 draft_tokens
            │
            ▼
    Target Model 验证 (标准 speculative decoding 流程)
```

---

## 附：Qwen3.5 Target 模型的混合注意力结构

Qwen3.5 是混合架构模型，包含两种注意力层：

| 层类型 | 类 | 说明 |
|--------|-----|------|
| `linear_attention` | `QwenGatedDeltaNetAttention` | Gated DeltaNet 线性注意力，有 recurrent state |
| `full_attention` | `Qwen3NextAttention` | 标准全注意力，有 KV cache |

参见 [qwen3_5.py#L137-L155](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/model_executor/models/qwen3_5.py#L137-L155)。

DFlash draft 模型只使用全注意力层（不使用线性注意力），因为 draft model 的 KV 是从 target hidden states 投影来的，不需要维护线性注意力的 recurrent state。Target model 通过 `SupportsEagle3` 接口在全注意力层之后输出 aux hidden states 供 DFlash 使用。
