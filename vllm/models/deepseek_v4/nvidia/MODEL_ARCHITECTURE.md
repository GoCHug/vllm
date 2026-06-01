# DeepSeek V4 NVIDIA 模型结构文档

> 本文档梳理了 DeepSeek V4 在 vLLM 中 NVIDIA 平台的完整模型结构，涵盖主模型、注意力机制、MoE、压缩器、多 token 预测（MTP）等全部核心组件。

---

## 1. 整体架构概览

```
DeepseekV4ForCausalLM
├── model: DeepseekV4Model
│   ├── embed_tokens: VocabParallelEmbedding
│   ├── layers: [DeepseekV4DecoderLayer × N]
│   │   ├── attn_norm / ffn_norm: RMSNorm
│   │   ├── attn: DeepseekV4Attention
│   │   │   ├── fused_wqa_wkv: MergedColumnParallelLinear   (Q投影 + KV投影)
│   │   │   ├── q_norm / kv_norm: RMSNorm
│   │   │   ├── wq_b: ColumnParallelLinear
│   │   │   ├── wo_a: ColumnParallelLinear (bmm)
│   │   │   ├── wo_b: RowParallelLinear
│   │   │   ├── rotary_emb: RoPE
│   │   │   ├── indexer: DeepseekV4Indexer          (compress_ratio=4 时)
│   │   │   ├── compressor: DeepseekCompressor      (compress_ratio>1 时)
│   │   │   ├── swa_cache_layer: DeepseekV4SWACache
│   │   │   └── mla_attn: DeepseekV4MLAAttention
│   │   │       └── impl: DeepseekV4FlashMLASparseImpl
│   │   ├── ffn: DeepseekV4MoE
│   │   │   ├── gate: GateLinear
│   │   │   ├── shared_experts: DeepseekV4MLP       (可选)
│   │   │   └── experts: DeepseekV4MegaMoEExperts   (MegaMoE 路径)
│   │   │       或 FusedMoE                          (标准 MoE 路径)
│   │   └── HC 参数 (MHC multi-head compression):
│   │       ├── hc_attn_fn / hc_ffn_fn
│   │       ├── hc_attn_base / hc_ffn_base
│   │       └── hc_attn_scale / hc_ffn_scale
│   ├── norm: RMSNorm
│   ├── hc_head_fn / hc_head_base / hc_head_scale
│   └── _mtp_hidden_buffer                         (MTP 残差缓存)
├── lm_head: ParallelLMHead
└── logits_processor: LogitsProcessor
```

---

## 2. 核心组件详解

### 2.1 DeepseekV4ForCausalLM（顶层模型）

**文件位置**: [model.py](model.py)

顶层因果语言模型，继承 `nn.Module` 和 `SupportsPP`（支持流水线并行）。

```python
class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")  # 默认 FP4 映射
```

**关键方法**:

| 方法 | 说明 |
|------|------|
| `forward(input_ids, positions, ...)` | 调用 `self.model(...)` 返回 hidden_states |
| `compute_logits(hidden_states)` | 通过 `lm_head` + `LogitsProcessor` 计算 logits |
| `embed_input_ids(input_ids)` | Token 嵌入 |
| `load_weights(weights)` | 使用 `AutoWeightsLoader` + 权重映射器加载权重 |
| `get_mtp_target_hidden_states()` | 返回 MTP draft 模型所需的 pre-hc_head 残差 |

---

### 2.2 DeepseekV4Model（模型主干）

**文件位置**: [model.py](model.py)

模型主干，包含 Token Embedding、N 层 Decoder Layer、最终 Norm 和 HyperCompressed Head。

```python
class DeepseekV4Model(nn.Module):
    embed_tokens: VocabParallelEmbedding     # Token 嵌入
    layers: ModuleDict                       # Decoder 层 (PP 分片)
    norm: RMSNorm                            # 最终 LayerNorm
    hc_head_fn/base/scale                    # HyperCompressed Head 参数
    _mtp_hidden_buffer                       # MTP 使用的 pre-hc_head 残差流
```

**HC Head 流程（HyperCompressed Head）**:

在训练/推理时，hidden_states 会被复制到 `hc_mult` 个流中执行 MHC（Multi-Head Compression），最终通过 `hc_head_fused_kernel_tilelang` 将 (N, hc_mult, D) 压缩回 (N, D)。

```
hidden_states (N, D)
    ↓ unsqueeze(-2).repeat(1, hc_mult, 1)
(N, hc_mult, D)
    ↓ [Decoder Layers] → 每层内部 MHC pre/post
(N, hc_mult, D)
    ↓ hc_head_fused_kernel_tilelang
(N, D)
    ↓ norm (RMSNorm)
(N, D) → lm_head → logits
```

**Pipeline Parallel 支持**:

- 首 PP rank: 持有 `embed_tokens`，执行嵌入 + 扩展为 (N, hc_mult, D)
- 中间 PP rank: 仅持有 `layers` 分片
- 末 PP rank: 持有 `norm` + `hc_head` 参数 + `_mtp_hidden_buffer`

---

### 2.3 DeepseekV4DecoderLayer（解码层）

**文件位置**: [model.py](model.py)

每个 Decoder Layer 包含：
- **MHC pre**（Multi-Head Compression 前处理）：结合 attn norm
- **Self-Attention**: `DeepseekV4Attention`
- **MHC post + pre**：attn 后处理与 ffn 前处理的融合
- **MoE FFN**: `DeepseekV4MoE`

```
输入 x (N, hc_mult, D)
    ↓ mhc_pre_tilelang / mhc_fused_post_pre_tilelang  ← 融合 attn_norm
    ↓ DeepseekV4Attention
    ↓ mhc_fused_post_pre_tilelang                      ← 融合 ffn_norm
    ↓ DeepseekV4MoE
输出 x, residual, post_mix, res_mix
```

**HC 层参数**:

| 参数 | 形状 | 说明 |
|------|------|------|
| `hc_attn_fn` | (mix_hc, hc_dim) | Attn 子层的 hypercompression 函数矩阵 |
| `hc_ffn_fn` | (mix_hc, hc_dim) | FFN 子层的 hypercompression 函数矩阵 |
| `hc_attn_base` | (mix_hc,) | Attn 子层偏置 |
| `hc_ffn_base` | (mix_hc,) | FFN 子层偏置 |
| `hc_attn_scale` | (3,) | Attn 子层缩放因子 |
| `hc_ffn_scale` | (3,) | FFN 子层缩放因子 |

其中 `mix_hc = (2 + hc_mult) * hc_mult`，`hc_dim = hc_mult * hidden_size`。

---

### 2.4 DeepseekV4Attention（注意力机制）

**文件位置**: [model.py](model.py)、[attention.py](../attention.py)、[flashmla.py](flashmla.py)

DeepSeek V4 使用 **Multi-Head Latent Attention (MLA)**，结合稀疏注意力（SWA + 压缩 KV）。

#### 核心投影

| 投影 | 类型 | 输入 → 输出 |
|------|------|-------------|
| `fused_wqa_wkv` | MergedColumnParallelLinear | hidden_size → [q_lora_rank, head_dim] |
| `q_norm` | RMSNorm | q_lora_rank |
| `wq_b` | ColumnParallelLinear | q_lora_rank → n_heads × head_dim |
| `kv_norm` | RMSNorm | head_dim |
| `wo_a` | ColumnParallelLinear (bmm) | n_heads × head_dim // n_groups → n_groups × o_lora_rank |
| `wo_b` | RowParallelLinear | n_groups × o_lora_rank → hidden_size |

#### 注意力头规格

```
head_dim = 512 (448 NoPE + 64 RoPE)
q_lora_rank = 1536
kv_lora_rank = 512
o_lora_rank = 1536
n_groups = 1 (MQA, 所有 Q 头共享 1 组 KV)
```

#### 三种压缩模式

| 模式 | compress_ratio | 索引器 | 说明 |
|------|---------------|--------|------|
| SWA-only | 1 | 无 | 纯滑动窗口注意力 |
| C4A | 4 | 有 (DeepseekV4Indexer) | 每层独立索引，稀疏注意力 |
| C128A | 128 | 无 | 预计算全局 topk 索引 |

#### 注意力计算流程

```
hidden_states
    ↓ attn_gemm_parallel_execute (流水线并行 GEMM)
    ├─ default stream: fused_wqa_wkv → qr, kv
    ├─ aux stream 0: compressor.kv_score       (compress_ratio > 1)
    ├─ aux stream 1: indexer.weights_proj      (C4A 模式)
    └─ aux stream 2: indexer.compressor        (C4A 模式)
    ↓
qr, kv → fused_q_kv_rmsnorm → qr (归一化), kv (归一化)
    ↓ wq_b → q → 变形 + Q head norm + RoPE + KV cache insert
    ↓ compressor(kv_score) → KV 压缩 + 写入 compressed KV cache
    ↓ indexer(qr, indexer_kv_score, indexer_weights) → 更新 topk_indices_buffer (C4A)
    ↓
DeepseekV4MLAAttention.forward(q, kv, positions, output)
    ↓ DeepseekV4FlashMLASparseImpl.forward_mqa
    ├─ Prefill: dequantize + gather KV → flash_mla_sparse_fwd
    └─ Decode:  flash_mla_with_kvcache
    ↓
output → fused_inv_rope_fp8_quant → fp8_einsum (wo_a) → wo_b → 返回
```

#### FlashMLA 集成（NVIDIA 专有）

**文件**: [flashmla.py](flashmla.py)

```
DeepseekV4FlashMLASparseBackend          ← 后端注册
DeepseekV4FlashMLASparseImpl             ← 实现类
```

- **KV Cache 格式**: `fp8_ds_mla`，每个 token 584 字节 = 448 (NoPE fp8) + 128 (RoPE bf16) + 8 (scale)
- **Q Head 填充**: 填充至 64 或 128 头以适配 FP8 decode kernel
- **Prefill 分块**: `PREFILL_CHUNK_SIZE = 4`，分块处理 prefill tokens
- **Decode**: 调用 `flash_mla_with_kvcache`，支持 SWA + compressed KV 双路注意力
- **Tile Scheduler**: 为三种压缩模式分别维护 `tile_sched_swaonly` / `tile_sched_c4a` / `tile_sched_c128a`

---

### 2.5 DeepseekV4MoE（混合专家层）

**文件位置**: [model.py](model.py)

支持两种 MoE 后端：

#### MegaMoE 路径 (`moe_backend="deep_gemm_mega_moe"`)

```
hidden_states
    ↓ GateLinear → router_logits
    ↓ fused_topk_bias → topk_weights, topk_ids
    ↓ prepare_megamoe_inputs (Triton kernel)
       ├─ hidden_states → fp8 quant (E8M0 scale)
       ├─ topk_ids → int64 布局重排
       └─ topk_weights → float32 布局重排
    ↓ deep_gemm.fp8_fp4_mega_moe
       ├─ L1 weight: w13 (FP4 experts)
       ├─ L2 weight: w2 (FP4 experts)
       └─ Symmetric buffer 用于 EP 通信
    ↓ + shared_experts (可选)
输出
```

**要求**: SM100 GPU，启用 expert parallel

#### 标准 FusedMoE 路径

使用 vLLM 的 `FusedMoE` 模块，支持：
- 标准 topk 路由
- Hash MoE 路由（前 `num_hash_layers` 层）：通过 `tid2eid` 表将 token ID 映射到专家
- `noaux_tc` 路由：使用 `e_score_correction_bias` 辅助评分

**路由参数**:

| 参数 | 说明 |
|------|------|
| `n_routed_experts` | 路由专家总数 |
| `num_experts_per_tok` | 每个 token 激活的专家数 (top_k) |
| `scoring_func` | 评分函数，MegaMoE 仅支持 `"sqrtsoftplus"` |
| `norm_topk_prob` | 是否归一化 topk 概率 |
| `routed_scaling_factor` | 路由缩放因子 |
| `swiglu_limit` | SwiGLU 激活值截断上限 |

---

### 2.6 DeepseekV4MegaMoEExperts（MegaMoE 专家）

**文件位置**: [model.py](model.py)

使用 DeepGEMM 的 FP4 MegaMoE 实现，将专家权重存储为：

| 参数 | 形状 | 数据类型 | 量化方法 |
|------|------|----------|----------|
| `w13_weight` | (n_local_experts, 2×intermediate, hidden//2) | uint8 (int8视图) | FP4 |
| `w13_weight_scale` | (n_local_experts, 2×intermediate, hidden//32) | uint8 (E8M0) | block |
| `w2_weight` | (n_local_experts, hidden, intermediate//2) | uint8 (int8视图) | FP4 |
| `w2_weight_scale` | (n_local_experts, hidden, intermediate//32) | uint8 (E8M0) | block |

**finalize_weights()**: 将原始参数转换为 DeepGEMM 所需的布局：
- 通过 `deep_gemm.transform_sf_into_required_layout` 转换 scale
- 通过 `deep_gemm.transform_weights_for_mega_moe` 转换权重
- 转换后释放原始 Parameter（节省显存）

**Symmetric Buffer**: 按 `(group, device, num_experts, max_tokens, top_k, hidden_size, intermediate_size)` 缓存，跨层共享。

---

### 2.7 DeepseekV4MLP（标准 MLP）

**文件位置**: [model.py](model.py)

```python
class DeepseekV4MLP(nn.Module):
    gate_up_proj: MergedColumnParallelLinear    # hidden → [intermediate, intermediate]
    act_fn: SiluAndMul | SiluAndMulWithClamp    # SwiGLU 激活
    down_proj: RowParallelLinear                # intermediate → hidden
```

支持 `swiglu_limit` 参数进行激活值截断（`SiluAndMulWithClamp`）。

---

### 2.8 DeepseekCompressor（KV 压缩器）

**文件位置**: [compressor.py](../compressor.py)

负责将 KV cache 压缩存储，结合 norm → RoPE → FP8 量化 → cache 写入。

#### 核心组件

| 组件 | 说明 |
|------|------|
| `fused_wkv_wgate` | MergedColumnParallelLinear，输出 KV 和 gate score |
| `norm` | RMSNorm |
| `ape` | 可学习的绝对位置编码参数 |
| `state_cache` (CompressorStateCache) | 存储部分状态 (kv_state + score_state) |

#### 压缩流程

```
hidden_states
    ↓ fused_wkv_wgate → kv, score
    ↓ save_partial_states (APE + 状态存储到 state_cache)
    ↓ compress_norm_rope_store_xxx (融合 kernel)
       ├─ 从 state_cache 读取状态
       ├─ RMSNorm
       ├─ RoPE (GPT-J 风格)
       ├─ FP8 量化 (UE8M0)
       └─ 写入 KV cache
```

#### 两种实现路径

| GPU | head_dim | 实现 |
|-----|----------|------|
| NVIDIA | 512 | `compress_norm_rope_store_cutedsl` (cutedsl kernel) |
| NVIDIA | 128 (Indexer) | `compress_norm_rope_store_triton` (Triton kernel) |
| AMD | 全部 | `compress_norm_rope_store_triton` (Triton kernel) |

---

### 2.9 DeepSeekV4MultiTokenPredictor（多 Token 预测 / MTP）

**文件位置**: [mtp.py](mtp.py)

DeepSeek V4 的 speculative decoding draft 模型。

#### 架构层次

```
DeepSeekV4MTP
└── model: DeepSeekV4MultiTokenPredictor
    ├── embed_tokens: VocabParallelEmbedding
    ├── logits_processor: LogitsProcessor
    └── layers: [DeepSeekV4MultiTokenPredictorLayer × num_mtp_layers]
        ├── enorm / hnorm: RMSNorm
        ├── e_proj / h_proj: ReplicatedLinear (FP8)
        ├── hc_head_fn / hc_head_base / hc_head_scale
        ├── shared_head: SharedHead
        └── mtp_block: DeepseekV4DecoderLayer
```

#### V4 MTP 与 V3 的区别

| 特性 | DeepSeek V3 | DeepSeek V4 |
|------|-------------|-------------|
| 嵌入投影 | 融合 `eh_proj` | 分离 `e_proj` + `h_proj` (FP8 量化) |
| Logits 计算 | 标准 head | `hc_head` (HyperCompressed head) 压缩 |
| Decoder 层 | 自定义 MTP 层 | 复用 `DeepseekV4DecoderLayer` |

#### MTP 前向流程

```
input_ids, positions, previous_hidden_states, inputs_embeds
    ↓ embed_input_ids (首步) 或使用传入的 inputs_embeds
    ↓ fused_mtp_input_rmsnorm (融合: mask + enorm + hnorm)
    ↓ h_proj(previous_hidden_states) + e_proj(inputs_embeds).unsqueeze(-2)
    ↓ mtp_block (DeepseekV4DecoderLayer: attn + ffn + MHC)
    ↓ mhc_post_tilelang
    ↓ flatten → pre-hc_head 残差
    ↓ (compute_logits 阶段)
    ↓ hc_head_fused_kernel_tilelang
    ↓ mtp_shared_head_rmsnorm
    ↓ LogitsProcessor(shared_head.head)
    ↓ logits
```

当 `num_speculative_tokens > 1` 时，pre-hc_head 残差作为下一步的 `previous_hidden_states` 被循环使用。

---

### 2.10 DeepseekV4Indexer（稀疏注意力索引器）

**文件位置**: [attention.py](../attention.py)

仅存在于 `compress_ratio = 4` 的层（C4A 模式），用于为压缩 KV cache 选择 topk 索引。

```
输入: hidden_states, qr, indexer_kv_score, indexer_weights, positions, rotary_emb
    ↓ weights_proj(hidden_states) → 计算压缩 KV 的注意力权重
    ↓ compressor(indexer_kv_score, positions, rotary_emb) → 压缩存储 indexer KV
    ↓ fused_indexer_q_rope_quant → 处理 Q 并做 RoPE + FP4 量化
    ↓ SparseAttnIndexer → 选择 topk 索引写入 topk_indices_buffer
```

---

## 3. NVIDIA 专属算子

**文件位置**: [ops/](ops/)

| 文件 | 算子 | 说明 |
|------|------|------|
| `prepare_megamoe.py` | `prepare_megamoe_inputs` | Triton kernel：hidden states FP8 量化 + topk 布局重排，为 DeepGEMM MegaMoE 准备输入 |
| `sparse_attn_compress_cutedsl.py` | `compress_norm_rope_store_cutedsl` | cutedsl kernel：压缩器主路径 (head_dim=512) |
| `fused_indexer_q_cutedsl.py` | Indexer Q 融合操作 | cutedsl kernel：Indexer Q 的 RoPE + quant |
| `dequant_gather_k_cutedsl.py` | K cache 解量化和收集 | cutedsl kernel：解量化并收集 KV cache |

---

## 4. 通用算子 (common ops)

**文件位置**: [common/ops/](../common/ops/)

| 算子 | 说明 |
|------|------|
| `fused_q_kv_rmsnorm` | 融合 Q 和 KV 的 RMSNorm |
| `fused_indexer_q_rope_quant` | 融合 Indexer Q 的 RoPE 和 FP4 量化 |
| `fused_inv_rope_fp8_quant` | 融合逆 RoPE 和 FP8 量化（O 投影前） |
| `fused_mtp_input_rmsnorm` | MTP 输入融合 norm（mask + enorm + hnorm） |
| `mtp_shared_head_rmsnorm` | MTP shared head 的 RMSNorm |
| `save_partial_states` | 压缩器状态部分保存 |
| `compress_norm_rope_store_triton` | Triton 版压缩+norm+RoPE+存储 |
| `combine_topk_swa_indices` | 合并 topk 和 SWA 索引 |
| `compute_global_topk_indices_and_lens` | 计算全局 topk 索引和长度 |
| `dequantize_and_gather_k_cache` | 解量化并收集 K cache |
| `quantize_and_insert_k_cache` | 量化并插入 K cache |

---

## 5. 量化配置

**文件位置**: [quant_config.py](../quant_config.py)

```python
class DeepseekV4FP8Config(Fp8Config):
    # FP8 block quantization for linear/attention layers
```

**专家数据类型** (由 `hf_config.expert_dtype` 控制):

| expert_dtype | 说明 | MoE 量化方法 | Scale 格式 |
|-------------|------|-------------|------------|
| `"fp4"` (默认) | MXFP4 专家 | `Mxfp4MoEMethod` | E8M0 (e8m0fnu) |
| `"fp8"` | FP8 block 专家 | `Fp8MoEMethod` (block_quant=True) | float32 |

**MoE 量化算法** (`moe_quant_algo`):
- `""`: 默认，使用 `Mxfp4MoEMethod`
- `"NVFP4"`: 使用 NVIDIA 的 `ModelOptNvFp4FusedMoE`

**KV Cache 格式**: `"fp8_ds_mla"` — DeepSeek 定制的 FP8 KV cache 布局。

---

## 6. 权重加载与映射

### 6.1 权重映射器

`_make_deepseek_v4_weights_mapper(expert_dtype)` 创建 `WeightsMapper`，处理 checkpoint 到模型参数的名称映射：

| 映射类型 | 示例 |
|----------|------|
| 前缀映射 | `"layers."` → `"model.layers."` |
| 后缀映射 | `"embed.weight"` → `"embed_tokens.weight"` |
| 子串替换 | `".attn.compressor."` → `".attn.mla_attn.compressor."` |
| Scale 正则 | `.scale$` → `.weight_scale` (fp4) / `.weight_scale_inv` (fp8) |

### 6.2 权重堆叠映射

```python
stacked_params_mapping = [
    ("gate_up_proj", "w1", 0),
    ("gate_up_proj", "w3", 1),
    ("attn.fused_wqa_wkv", "attn.wq_a", 0),
    ("attn.fused_wqa_wkv", "attn.wkv", 1),
    ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
    ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
]
```

### 6.3 专家参数映射

`make_deepseek_v4_expert_params_mapping(num_experts)` 生成 MegaMoE 专家映射：

```
experts.{expert_id}.w1 → experts.w13_[shard 0]
experts.{expert_id}.w3 → experts.w13_[shard 1]
experts.{expert_id}.w2 → experts.w2_
```

---

## 7. 推理流水线

### 7.1 主模型前向

```
Embedding: input_ids → embed_tokens → (N, D)
    → unsqueeze+repeat → (N, hc_mult, D)
    ↓
MHC Pre (Layer 0): mhc_pre_tilelang (一次性初始化 residual/post_mix/res_mix)
    ↓
[Decoder Layer × N]:
    ├─ mhc_fused_post_pre_tilelang (attn: post前层 + pre本层)
    ├─ DeepseekV4Attention
    ├─ mhc_fused_post_pre_tilelang (ffn: post attn + pre ffn)
    └─ DeepseekV4MoE
    ↓
MHC Post (最后一层): mhc_post_tilelang
    ↓
hc_head: hc_head_fused_kernel_tilelang → (N, D)
    ↓
Final Norm: RMSNorm
    ↓
lm_head → logits
```

### 7.2 MTP Draft 前向

```
Main model 输出 pre-hc_head residual (N, hc_mult*D)
    ↓
[spec_step = 0..num_speculative_tokens-1]:
    ├─ embed_input_ids (首步) / 使用上步 logits 采样的 token
    ├─ fused_mtp_input_rmsnorm
    ├─ e_proj + h_proj → hidden_states
    ├─ mtp_block (DeepseekV4DecoderLayer)
    ├─ mhc_post_tilelang → pre-hc_head residual
    └─ hc_head + shared_head → logits (用于下一步采样)
```

---

## 8. 关键配置参数

| 参数 | 说明 |
|------|------|
| `hc_mult` | HyperCompressed 多路复用系数 |
| `hc_eps` | HC 数值稳定 epsilon |
| `hc_sinkhorn_iters` | Sinkhorn 迭代次数 |
| `compress_ratios` | 每层的压缩比列表 |
| `index_topk` | Indexer 选出的 topk 压缩块数 |
| `sliding_window` | 滑动窗口大小 |
| `num_hash_layers` | 使用 Hash MoE 路由的前 N 层 |
| `expert_dtype` | 专家数据类型 ("fp4" / "fp8") |
| `swiglu_limit` | SwiGLU 激活值截断上限 |
| `routed_scaling_factor` | 路由权重缩放因子 |
| `norm_topk_prob` | 是否归一化 topk 概率 |
| `scoring_func` | 路由评分函数 |

---

## 9. 依赖关系图

```
DeepseekV4ForCausalLM
└── DeepseekV4Model
    ├── VocabParallelEmbedding
    ├── DeepseekV4DecoderLayer
    │   ├── MHC kernels (mhc_pre / mhc_fused_post_pre / mhc_post tilelang)
    │   ├── DeepseekV4Attention
    │   │   ├── DeepseekV4MultiHeadLatentAttentionWrapper
    │   │   │   ├── MergedColumnParallelLinear (fused_wqa_wkv)
    │   │   │   ├── ColumnParallelLinear (wq_b, wo_a)
    │   │   │   ├── RowParallelLinear (wo_b)
    │   │   │   ├── RMSNorm (q_norm, kv_norm)
    │   │   │   ├── RoPE
    │   │   │   ├── DeepseekV4Indexer (C4A)
    │   │   │   ├── DeepseekCompressor
    │   │   │   │   ├── MergedColumnParallelLinear (fused_wkv_wgate)
    │   │   │   │   ├── RMSNorm
    │   │   │   │   ├── CompressorStateCache
    │   │   │   │   ├── save_partial_states
    │   │   │   │   └── compress_norm_rope_store_xxx
    │   │   │   ├── DeepseekV4SWACache
    │   │   │   └── DeepseekV4MLAAttention
    │   │   │       └── DeepseekV4FlashMLASparseImpl (NVIDIA)
    │   │   │           ├── flash_mla_with_kvcache (decode)
    │   │   │           └── flash_mla_sparse_fwd (prefill)
    │   │   └── fused_q_kv_rmsnorm / fused_inv_rope_fp8_quant / fp8_einsum
    │   └── DeepseekV4MoE
    │       ├── GateLinear
    │       ├── DeepseekV4MegaMoEExperts (MegaMoE 路径)
    │       │   ├── deep_gemm.transform_sf_into_required_layout
    │       │   ├── deep_gemm.transform_weights_for_mega_moe
    │       │   ├── prepare_megamoe_inputs
    │       │   └── deep_gemm.fp8_fp4_mega_moe
    │       ├── FusedMoE (标准路径)
    │       └── DeepseekV4MLP (shared_experts)
    │           └── MergedColumnParallelLinear + SiluAndMul + RowParallelLinear
    ├── RMSNorm (final)
    ├── hc_head (hc_head_fused_kernel_tilelang)
    └── _mtp_hidden_buffer
├── ParallelLMHead
└── LogitsProcessor
```