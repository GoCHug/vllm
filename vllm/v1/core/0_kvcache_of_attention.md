# 各类 Attention 的 KV Cache 存储信息总览

> 源文件索引：
> - Spec 定义：`vllm/v1/kv_cache_interface.py`
> - Backend shape：`vllm/v1/attention/backends/*.py`、`vllm/v1/attention/backends/mla/*.py`
> - 状态 shape：`vllm/model_executor/layers/mamba/mamba_utils.py`（`MambaStateShapeCalculator`、`MambaStateDtypeCalculator`）
> - 抽象层：`vllm/model_executor/layers/mamba/abstract.py`（`MambaBase.get_kv_cache_spec`）
> - 分配/reshape：`vllm/v1/worker/gpu_model_runner.py`（`_reshape_kv_cache_tensors`）
>
> 本文聚焦"每种 attention/SSM 类型的 KV cache **物理 tensor 最终是什么 shape**、**存的是什么数据**"。

---

## 速查总表

### Attention 系列（基于 `AttentionSpec`）

| Attention 类型 | Spec 类 | 典型逻辑 shape | K/V 存放方式 | 典型模型 |
|---|---|---|---|---|
| Full Attention | `FullAttentionSpec` | `(num_blocks, num_kv_heads, block_size, 2*head_size)` | 最后一维前半 K 后半 V | Llama、Qwen、Mistral |
| Full Attention (Diff KV) | `FullAttentionSpec` (`head_size_v ≠ head_size`) | `(num_blocks, num_kv_heads, block_size, head_size+head_size_v)` | 最后一维前 head_size 为 K，后 head_size_v 为 V | MiMo-V2 |
| Full Attention (ROCm) | `FullAttentionSpec` | `(2, num_blocks, block_size, num_kv_heads, head_size)` | dim 0 的 2 分别 K/V | Llama on AMD GPU |
| Full Attention (HPC) | `FullAttentionSpec` | `(num_blocks, 2, block_size, num_kv_heads, head_size)` | dim 1 的 2 分别 K/V | Hopper GPU (SM90+) |
| Sliding Window | `SlidingWindowSpec` | 同 Full Attention | 同 Full Attention | Gemma3 |
| MLA | `MLAAttentionSpec` | `(num_blocks, block_size, head_size)` | 单一 latent 向量，无独立 K/V 分离 | DeepSeek V2/V3 |
| MLA (fp8_ds_mla V3.2) | `MLAAttentionSpec` | `(num_blocks, block_size, 656)` | 512B NoPE + 16B scale + 128B RoPE | DeepSeek V3.2 |
| MLA (DeepSeek V4) | `MLAAttentionSpec` | `(num_blocks, block_size, 584)` | 448B NoPE + 128B RoPE + 8B fp8 scale | DeepSeek V4 |
| SWA + MLA | `SlidingWindowMLASpec` | `(num_blocks, block_size, head_size)` 或 584/656 | 同 MLA | DeepSeek V4 SWA |
| Cross Attention | `CrossAttentionSpec` | 同 Full Attention（取决于后端） | 静态 encoder KV | Whisper |
| Sink Attention | `SinkFullAttentionSpec` | 同 Full Attention | sink block 常驻 | — |
| RSWEA | `RSWASpec` | 同 Full Attention | gap block 驱逐 | — |
| Chunked Local | `ChunkedLocalAttentionSpec` | 同 Full Attention | 块内局部注意力 | GLM-4v |
| TurboQuant | `TQFullAttentionSpec` | `(num_blocks, num_kv_heads, block_size, slot_size_aligned)` | K+V 交织打包 | — |
| Encoder-Only | `EncoderOnlyAttentionSpec` | 无 KV cache（max_memory=0） | — | BERT |

### SSM 系列（基于 `MambaSpec`，非 `AttentionSpec`）

| SSM 类型 | Spec `mamba_type` | 物理 tensor shape | 状态子张量 shapes | 典型模型 |
|---|---|---|---|---|
| Mamba1 | `MAMBA1` | `(num_blocks, 1, 1, page_size_bytes)` | conv `(intermediate//tp, conv_kernel-1)` + ssm `(intermediate//tp, state_size)` | Mamba、Jamba |
| Mamba2 | `MAMBA2` | `(num_blocks, 1, 1, page_size_bytes)` | conv `(conv_dim//tp, conv_kernel-1+num_spec)` + ssm `(num_heads//tp, head_dim, state_size)` | Mamba2、Falcon-Mamba |
| GDN | `GDN_ATTN` | `(num_blocks, 1, 1, page_size_bytes)` | conv `(conv_dim//tp, conv_kernel-1+num_spec)` + temporal `(num_v_heads//tp, head_v_dim, head_k_dim)` | Qwen3-Next、OLMo-Hybrid |
| Short Conv | `SHORT_CONV` | `(num_blocks, 1, 1, page_size_bytes)` | conv `(intermediate//tp, conv_kernel-1)` | — |
| Linear Attn | `LINEAR` | `(num_blocks, 1, 1, page_size_bytes)` | state `(num_heads//tp, head_dim, head_dim)` | — |
| KDA | (CUSTOM/注册) | `(num_blocks, 1, 1, page_size_bytes)` | conv `(conv_dim//tp, conv_kernel-1)` + recurrent `(num_heads//tp, head_dim, head_dim)` | Kimi-Linear |

> **关键区别**：Attention 系列的 KV cache 每个 block 存储 `block_size` 个 token 的 K/V 数据，张量有 `num_kv_heads` 和 `head_size` 维度；SSM 系列的 cache 每个 block 存储一份**递归状态**（conv state + ssm/temporal state），张量是一个扁平字节缓冲，`bind_kv_cache` 时再按 state shape 切分 view。

---

## 一、Full Attention

### 1.1 基本 Shape

Full Attention 的物理 KV cache shape 由 **attention backend** 的 `get_kv_cache_shape()` 决定，有三种主流形式：

#### 形式 A：K/V packed in content dim（最常见）

```
(num_blocks, num_kv_heads, block_size, 2 * head_size)
     ↑           ↑              ↑              ↑
   块编号      KV 头数      每块 token 数    K 和 V 在最后一维拼接
                                  前 head_size 为 K，后 head_size 为 V
```

**使用此 shape 的 backend**：

| Backend | 源码位置 | 备注 |
|---|---|---|
| FlashAttention | `flash_attn.py:144` | `block_size % 16 == 0` |
| FlashInfer | `flashinfer.py:408` | NVFP4 时 shape 不同（见 1.3） |
| CPU | `cpu_attn.py:101` | — |
| Triton | `triton_attn.py:351` | per-token-head 量化时 shape 不同 |
| FlexAttention | `flex_attention.py:138` | — |
| ROCm Aiter FA | `rocm_aiter_fa.py:775` | — |
| ROCm Aiter Unified | `rocm_aiter_unified_attn.py:91` | — |

#### 形式 B：K/V as separate dim

```
(2, num_blocks, block_size, num_kv_heads, head_size)
 ↑      ↑           ↑              ↑           ↑
K/V   块编号     每块 token 数    KV 头数     头维度
```

**使用此 shape 的 backend**：

| Backend | 源码位置 | 备注 |
|---|---|---|
| ROCm Attn | `rocm_attn.py:256` | `block_size % 16 == 0` |

#### 形式 C：HPC（Hopper）

```
(num_blocks, 2, block_size, num_kv_heads, head_size)
     ↑       ↑      ↑              ↑           ↑
   块编号   K/V   每块 token 数    KV 头数     头维度
```

| Backend | 源码位置 | 备注 |
|---|---|---|
| HPC Attn | `hpc_attn.py:293` | 仅 SM90+，仅 head_size=128 |

### 1.2 Diff-KV 变体（K 和 V 维度不同）

当 `head_size_v ≠ head_size` 时（如 MiMo-V2），K 和 V 的头维度不同但仍打包在最后一维：

```
(num_blocks, num_kv_heads, block_size, head_size + head_size_v)
     ↑           ↑              ↑                    ↑
   块编号      KV 头数       每块 token 数     前 head_size 为 K，后 head_size_v 为 V
```

| Backend | 源码位置 |
|---|---|
| FlashAttn DiffKV | `flash_attn_diffkv.py:88-93` |
| Triton DiffKV | `triton_attn_diffkv.py:108-113` |

### 1.3 量化对 shape 的影响

| 量化模式 | backend | 逻辑 shape 变化 | 说明 |
|---|---|---|---|
| 无量化 (bf16/fp16) | 所有 | 同基本 shape | dtype 为 bf16/fp16 |
| FP8 / INT8 | 所有 | 同基本 shape | dtype 变为 uint8/int8，head_size 不变 |
| INT4 per-token-head | Triton | `(num_blocks, num_kv_heads, block_size, 2 * (head_size//2 + 4))` | 2×int4 打包 1 字节 + fp32 scale (4B) 内联 |
| NVFP4 | FlashInfer | `(num_blocks, 2 * num_kv_heads, block_size, nvfp4_kv_cache_full_dim(head_size))` | head 数翻倍，head_dim = head_size//2 + head_size//16 |

> **INT4 per-token-head** 的 `4` 来自 `get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)` = `4 // 1 = 4`，即 fp32 scale 占 4 个 cache_dtype 元素位。

> **NVFP4** 的 `nvfp4_kv_cache_full_dim(head_size)` = `head_size//2 + head_size//16`，包含量化数据 + block scale。

### 1.4 page_size_bytes（字节数计算）

```python
# FullAttentionSpec.real_page_size_bytes (kv_cache_interface.py:327-342)
last_dim = head_size + head_size_v  # 非量化
#   INT4:  last_dim = head_size//2 + head_size_v//2
#   NVFP4: last_dim = nvfp4_kv_cache_full_dim(head_size) + nvfp4_kv_cache_full_dim(head_size_v)

page_size_bytes = block_size * num_kv_heads * last_dim * get_dtype_size(dtype)
```

> 注意：`real_page_size_bytes` **不含** `2 ×` 因为 K 和 V 的维度已经加在 `last_dim` 里。`AttentionSpec.real_page_size_bytes` 才有 `2 ×`，因为它只算单个 K 或 V。

### 1.5 stride order（物理布局）

逻辑 shape 和物理 stride 可以不同。`_reshape_attention_kv_cache` 先 `view` 出物理 contiguous 的 permuted shape，再 `permute` 回逻辑 shape。

| layout | stride order | 物理布局 | shape 不变 |
|---|---|---|---|
| HND | `(0, 1, 2, 3)` | `(B, H, N, 2*D)` | 是 |
| NHD | `(0, 2, 1, 3)` | `(B, N, H, 2*D)` | 是（shape 仍为 `(B, H, N, 2*D)`） |

由 `get_kv_cache_layout()` 全局设置控制，FlashInfer/FlashAttention 均支持。

---

## 二、MLA（Multi-head Latent Attention）

### 2.1 存储 shape

MLA 的核心区别：**不存储分离的 K 和 V，而是存储压缩后的 latent 向量**，`num_kv_heads` 固定为 1，不需要在 shape 中体现。

#### 标准（非量化 / fp8）

```
(num_blocks, block_size, head_size)
     ↑          ↑          ↑
   块编号    每块 token  latent 向量维度（如 576）
```

| Backend | 源码位置 |
|---|---|
| FlashMLA | `mla/flashmla_sparse.py:142` |
| FlashAttn MLA | `mla/flashattn_mla_sparse.py:114` |
| FlashInfer MLA (SM90) | `mla/flashinfer_mla_sparse.py:134` |
| FlashInfer MLA (SM120) | `mla/flashinfer_mla_sparse.py:229` |
| ROCm Aiter MLA | `mla/rocm_aiter_mla_sparse.py:303` |
| XPU MLA | `mla/xpu_mla_sparse.py:77` |

#### fp8_ds_mla（DeepSeek V3.2）

```
(num_blocks, block_size, 656)
```

656 字节 = 512B NoPE latent + 16B fp8 scale + 128B RoPE latent。

DeepSeek V3.2 MLA 的 `kv_lora_rank=512`、`qk_rope_head_dim=64`，head_size 语义上是 576（512+64），但物理存储为自定义 656B 布局。

| Backend | 源码位置 |
|---|---|
| FlashMLA | `mla/flashmla_sparse.py:140` |
| FlashInfer MLA (SM120) | `mla/flashinfer_mla_sparse.py:229` |

#### fp8_ds_mla（DeepSeek V4）

```
(num_blocks, storage_block_size, 584)
```

584 字节 = 448B NoPE + 128B RoPE + 8B fp8 scale。`storage_block_size = block_size // compress_ratio`。

| Backend | 源码位置 |
|---|---|
| FlashMLA (V4) | `mla/flashmla_sparse.py:140`（model_version == "deepseek_v4"） |
| Sparse SWA (V4) | `mla/sparse_swa.py:149` |

### 2.2 page_size_bytes

```python
# MLAAttentionSpec.real_page_size_bytes (kv_cache_interface.py:397-416)
if cache_dtype_str == "fp8_ds_mla":
    if model_version == "deepseek_v4":
        return storage_block_size * 584        # V4: 584B/token
    return block_size * 656                    # V3.2: 656B/token
if kv_quant_mode == INT4_PER_TOKEN_HEAD:
    head_dim = head_size // 2
else:
    head_dim = head_size
return storage_block_size * num_kv_heads * head_dim * get_dtype_size(dtype)
```

### 2.3 compress_ratio

```python
# MLAAttentionSpec.storage_block_size (kv_cache_interface.py:394-395)
return block_size // compress_ratio
```

DeepSeek V4 引入 `compress_ratio`，把逻辑 block_size 压缩到更小的存储 block_size。

---

## 三、Sliding Window Attention

### 3.1 物理 shape

与 Full Attention **完全相同**——因为 SWA 在 KV cache 布局层面与 Full Attention 一致，区别仅在于 attention 计算时只看最近 `sliding_window` 个 token。

```
取决于 backend：
  形式 A: (num_blocks, num_kv_heads, block_size, 2 * head_size)
  形式 B: (2, num_blocks, block_size, num_kv_heads, head_size)
```

### 3.2 与 Full Attention 的区别

| 维度 | Full Attention | Sliding Window |
|---|---|---|
| Spec 类 | `FullAttentionSpec` | `SlidingWindowSpec` |
| page_size_bytes | 相同 | 相同 |
| 驻留 block 数 | 全部 token 的 block | 最多 `sliding_window - 1 + in_flight` 个 token 的 block |
| `max_admission_blocks_per_request` | — | `cdiv(min(sliding_window - 1 + max_in_flight, max_model_len), block_size) + 1` |

> 当 `--disable-hybrid-kv-cache-manager` 开启时，SWA 层使用 `FullAttentionSpec`（缓存所有 token），仅计算时按窗口读取。

---

## 四、Sliding Window + MLA（SlidingWindowMLASpec）

DeepSeek V4 的滑动窗口 MLA 层，使用 MLA 的 latent 存储格式 + 滑动窗口的驱逐策略。

### 物理 shape

```
标准:      (num_blocks, block_size, head_size)
V4 fp8:    (num_blocks, storage_block_size, 584)
```

| 版本 | shape | 语义 |
|---|---|---|
| 标准 | `(num_blocks, block_size, head_size)` | 同 MLA |
| DeepSeek V4 fp8_ds_mla | `(num_blocks, storage_block_size, 584)` | 448B NoPE + 128B RoPE + 8B scale |

源码：`mla/sparse_swa.py:145-151`

---

## 五、SSM 系列（MambaSpec）

### 5.1 物理 tensor shape

所有 SSM 类型（Mamba1/Mamba2/GDN/ShortConv/LinearAttn）的物理 KV cache tensor 形状都是**相同的扁平结构**：

```python
# gpu_model_runner.py:7446-7448
kv_caches[layer_name] = raw_tensor[:num_blocks * page_size_bytes].view(
    num_blocks, 1, 1, page_size_bytes
)
```

```
(num_blocks, 1, 1, page_size_bytes)
     ↑                    ↑
   块编号          每 block 的扁平字节缓冲区（int8 dtype）
```

**关键**：与 Attention 不同，SSM 的 KV cache **没有 `num_kv_heads`/`head_size`/`block_size(token数)` 维度**。每个 block 是一个 `page_size_bytes` 字节的扁平缓冲区，由 `bind_kv_cache` 在 forward 时按 state shape 切分为 conv state 和 ssm/temporal state 的 view。

### 5.2 page_size_bytes

```python
# MambaSpec.page_size_bytes (kv_cache_interface.py:698-707)
page_size = sum(
    prod(shape) * get_dtype_size(dtype)
    for (shape, dtype) in zip(self.shapes, self.dtypes)
)
```

即 `page_size_bytes = Σ (各 state 的元素总数 × 各 state dtype 的字节数)`。

### 5.3 各 SSM 类型的 state shapes

#### Mamba1（`MAMBA1`）

```
conv_state:     (intermediate_size // tp, conv_kernel - 1)
temporal_state: (intermediate_size // tp, state_size)
```

源码：`mamba_utils.py:159-171`、`mamba_mixer.py:472-478`

dtype：`(conv_state_dtype, temporal_state_dtype)`，通常为 `(bf16, bf16)` 或 `(fp8, bf16)`。

`page_size_bytes = (intermediate//tp) × (conv_kernel-1) × dtype_size + (intermediate//tp) × state_size × dtype_size`

#### Mamba2（`MAMBA2`）

```
conv_state:     (conv_dim // tp, conv_kernel - 1 + num_spec)
temporal_state: (num_heads // tp, head_dim, state_size)
```

其中 `conv_dim = intermediate_size + 2 * n_groups * state_size`（含 A/B 门控投影）。

源码：`mamba_utils.py:173-199`、`mamba_mixer2.py:1119-1139`

> Mamba2 的 conv_state 比 Mamba1 多了 `num_spec` 列（投机解码用），且维度计算含 n_groups。

#### GDN / Gated Delta Net（`GDN_ATTN`）

```
conv_state:      (conv_dim // tp, conv_kernel_size - 1 + num_spec)
temporal_state:  (num_v_heads // tp, head_v_dim, head_k_dim)
```

其中 `conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads`。

源码：`mamba_utils.py:247-268`、`gdn/qwen_gdn_linear_attn.py:343-354`

> GDN 的 temporal_state 是一个 3D 矩阵 `(v_heads, v_dim, k_dim)`，本质是一个**门控的 delta-rule 更新矩阵**，而非传统 SSM 的 `(heads, head_dim, state_size)` 状态矩阵。

dtype：`(conv_state_dtype, temporal_state_dtype)`——与 Mamba 相同的 `_mamba_state_dtype`。

#### Short Conv（`SHORT_CONV`）

```
conv_state: (intermediate_size // tp, conv_kernel - 1)
```

**只有 conv state**，没有 temporal/ssm state。本质是一个短卷积注意力，仅缓存卷积滑窗。

源码：`mamba_utils.py:224-232`、`short_conv.py:324-329`

dtype：`(conv_state_dtype,)` — 仅一种 dtype。

#### Linear Attention（`LINEAR`）

```
state: (num_heads // tp, head_dim, head_dim)
```

**只有 recurrent state**，没有 conv state。是一个 `(heads, key_dim, value_dim)` 的外积矩阵。

源码：`mamba_utils.py:142-149`、`linear/base.py:63-66`

dtype：`(state_dtype,)` — 仅一种 dtype。

#### KDA / Kimi Delta Attention

```
conv_state:          (conv_dim // tp, conv_kernel - 1)
recurrent_state:     (num_heads // tp, head_dim, head_dim)
```

其中 `conv_dim = num_heads * head_dim + 2 * num_k_heads * head_k_dim`。

源码：`mamba_utils.py:271-294`

### 5.4 bind_kv_cache：从扁平字节缓冲到 `self.kv_cache`

上一节列出了各 SSM 类型的 state shapes，但 **物理 tensor 是一个扁平的 `(num_blocks, 1, 1, page_size_bytes)` int8 缓冲区**。`bind_kv_cache` 负责把这个扁平缓冲区切分成各 state 的独立 view，存入 `self.kv_cache`。

#### 5.4.1 切分逻辑（`MambaBase.bind_kv_cache`）

```python
# abstract.py:29-43
def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
    """Unpack a raw [B, 1, 1, C] int8 page view into per-state views."""
    pages = kv_cache.squeeze(dim=(1, 2))        # (num_blocks, page_size_bytes) int8
    states: list[torch.Tensor] = []
    offset = 0
    for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype()):
        nbytes = prod(shape) * get_dtype_size(dtype)
        # 1. 按 offset 切出该 state 的字节范围
        state_bytes = pages[:, offset : offset + nbytes]
        # 2. view 成目标 dtype
        state = state_bytes.view(dtype)
        # 3. reshape 为 (num_blocks, *state_shape)
        states.append(state.view(-1, *shape))
        offset += nbytes
    self.kv_cache = tuple(states)
```

关键步骤：
1. **squeeze**：`(B, 1, 1, C)` → `(B, C)`，去掉多余维度
2. **逐 state 切片**：按各 state 的 `prod(shape) × dtype_size` 字节数从 page 中顺序切出
3. **dtype view**：`int8` 字节 → 目标 dtype（bf16/fp8/fp32）
4. **reshape**：展平后恢复为 `(num_blocks, *state_shape)`，num_blocks 自动推导

> 切分是 **zero-copy view**——不拷贝数据，只是创建不同 dtype/shape 的视图，指向同一块物理内存。

#### 5.4.2 GDN 的 `self.kv_cache` 结构

GDN 的 `get_state_shape()` 返回 2 元组 `(conv_state_shape, temporal_state_shape)`，`get_state_dtype()` 返回 `(conv_dtype, temporal_dtype)`，因此：

```python
self.kv_cache = (
    conv_state,   # self.kv_cache[0]  shape: (num_blocks, conv_dim//tp, conv_kernel-1+num_spec)
    ssm_state,    # self.kv_cache[1]  shape: (num_blocks, num_v_heads//tp, head_v_dim, head_k_dim)
)
```

以 Qwen3-Next 配置（`num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, tp=1, bf16`）为例：

```
conv_dim = 128 * 8 * 2 + 128 * 8 = 3072

self.kv_cache[0]  (conv_state):
  shape: (num_blocks, 3072, 3)     # SD 布局: (state_len, dim)
  或     (num_blocks, 3, 3072)     # DS 布局: (dim, state_len) — AMD AITER 等
  dtype: bf16
  字节: 3072 * 3 * 2 = 18,432 B per block

self.kv_cache[1]  (ssm_state):
  shape: (num_blocks, 8, 128, 128)  # (blocks, v_heads, v_dim, k_dim)
  dtype: bf16
  字节: 8 * 128 * 128 * 2 = 262,144 B per block

page_size_bytes = 18,432 + 262,144 = 280,576 B
```

#### 5.4.3 GDN forward 中如何使用 `self.kv_cache`

在 forward 中（以 vllm-ascend `AscendGatedDeltaNetAttention` 为例），直接通过索引访问各 state：

```python
# vllm-ascend/ops/gdn.py:174-175
self_kv_cache = self.kv_cache    # (conv_state, ssm_state) tuple
ssm_state = self_kv_cache[1]     # (num_blocks, v_heads, v_dim, k_dim)

# 1. Conv1d：读取/更新 conv_state
#    self_kv_cache[0] 作为 conv_state 传入算子
torch.ops._C_ascend.npu_causal_conv1d_custom(
    ...,
    conv_state=self_kv_cache[0],      # (num_blocks, conv_dim, conv_kernel-1)
    cache_indices_opt=cache_indices,  # block_id 列表，索引第 0 维
    ...
)

# 2. Recurrent delta rule：读取/更新 ssm_state
#    decode 路径 — 逐 token 递归更新
core_attn_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query, key=key, value=value,
    g=g, beta=beta,
    state=ssm_state,                    # (num_blocks, v_heads, v_dim, k_dim)
    ssm_state_indices=state_indices,    # block_id 列表，索引第 0 维
    ...
)

#    prefill 路径 — chunk 分块计算
initial_state = ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()
core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
    q=query, k=key, v=value, g=g, beta=beta,
    initial_state=initial_state,        # 从 ssm_state 按 block_id 取出
    output_final_state=True,
    ...
)
# 写回最终状态
ssm_state[prefill_state_indices] = last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
```

**Qwen GDN（非 Ascend）** 的用法完全一致，区别仅在于 conv_state 的 DS/SD 布局处理：

```python
# qwen_gdn_linear_attn.py:1228-1236
self_kv_cache = self.kv_cache
conv_state = (
    self_kv_cache[0]                        # DS 布局: (blocks, dim, width-1) — 直接用
    if is_conv_state_dim_first()
    else self_kv_cache[0].transpose(-1, -2) # SD 布局: 物理是 (blocks, width-1, dim)，转置
)
ssm_state = self_kv_cache[1]
```

#### 5.4.4 不同 SSM 类型的 `self.kv_cache` 元组长度

| 类型 | `self.kv_cache` 元组 | 各元素含义 |
|---|---|---|
| Mamba1 | 2-tuple | `(conv_state, ssm_state)` |
| Mamba2 | 2-tuple | `(conv_state, ssm_state)` |
| Mamba2 + ReplaySSM | 5-tuple | `(conv_state, ssm_state, x_cache, dt_cache, B_cache)` |
| GDN | 2-tuple | `(conv_state, ssm_state)` |
| Short Conv | 1-tuple | `(conv_state,)` |
| Linear Attn | 1-tuple | `(state,)` |
| KDA | 2-tuple | `(conv_state, recurrent_state)` |

> Mamba2 + ReplaySSM 是唯一超过 2 元素的情况，额外 3 个 buffer（`x_cache`, `dt_cache`, `B_cache`）用于 ReplaySSM 投机解码的环形缓存，由 `append_replayssm_ring()` 在 `get_state_shape()` 中追加。

#### 5.4.5 block 索引机制

所有 state view 的第 0 维都是 `num_blocks`，与 `block_table` 中的 `block_id` 一一对应：

```
conv_state[block_id]  →  (conv_dim, conv_kernel-1)     该 block 的卷积滑窗状态
ssm_state[block_id]   →  (v_heads, v_dim, k_dim)       该 block 的递归状态矩阵
```

forward 时通过 `cache_indices` / `state_indices`（即 `block_table` 中的 `block_id` 列表）做 fancy indexing 取出对应请求的状态。这与 Attention 系列的 `block_table[block_id]` 索引 `kv_caches[layer][block_id]` 机制完全一致，区别仅在于 SSM 每个 block 存的是**递归状态**而非 token 的 K/V。

### 5.5 conv state 布局方向

```python
# mamba_utils.py:152-156
def _orient_conv_shape(dim, state_len):
    if is_conv_state_dim_first():  # DS 布局
        return (dim, state_len)
    return (state_len, dim)        # SD 布局
```

conv state 的两维顺序取决于 `is_conv_state_dim_first()`——某些设备（如 AMD AITER）使用 dim-first (DS) 布局，其他使用 state_len-first (SD)。

### 5.6 Mamba cache mode

```python
# mamba_utils.py:709-718
if mamba_cache_mode == "all":
    # 每个 token 位置都有独立 state block
    max_blocks = cdiv(max_model_len, block_size) + num_speculative_blocks
elif mamba_cache_mode == "align":
    # 仅 2 + num_speculative_blocks 个 state block 常驻
    max_blocks = 2 + num_speculative_blocks
else:  # "none"
    max_blocks = 1 + num_speculative_blocks
```

| cache mode | 常驻 block 数 | 说明 |
|---|---|---|
| `none` (默认) | `1 + num_spec` | 仅当前步状态 |
| `align` | `2 + num_spec` | 前一步 + 当前步 + 投机 |
| `all` | `cdiv(max_model_len, block_size) + num_spec` | 类似 attention，每 token 一 block（支持 prefix caching） |

---

## 六、其他 Attention 变体

### 6.1 Cross Attention（Encoder-Decoder）

```python
# CrossAttentionSpec (kv_cache_interface.py:749-759)
# 继承 AttentionSpec，物理 shape 同 Full Attention（取决于后端）
# 区别：缓存的是 encoder 输出，不释放
max_memory = cdiv(max_encoder_len, block_size) * page_size_bytes
```

### 6.2 Sink Attention

```python
# SinkFullAttentionSpec(FullAttentionSpec) (kv_cache_interface.py:762-813)
# 物理 shape 同 Full Attention
# 区别：前 sink_len 个 token 的 block 永久驻留不驱逐
# 额外字段：sink_len: int | None
```

### 6.3 RSWEA（Rotating SWA）

```python
# RSWASpec(FullAttentionSpec) (kv_cache_interface.py:458-496)
# 物理 shape 同 Full Attention
# 区别：prefill token 全局可见，最近 rswa_window 个生成 token 保留，
#       gap block 在每次 decode 步驱逐
# 额外字段：rswa_window: int
```

### 6.4 Chunked Local Attention

```python
# ChunkedLocalAttentionSpec(AttentionSpec) (kv_cache_interface.py:498-536)
# 物理 shape 同 Full Attention
# 区别：长序列切分为 attention_chunk_size 的 chunk 独立计算
# 块内局部注意力，不跨 chunk
# 额外字段：attention_chunk_size: int
```

### 6.5 TurboQuant

```python
# TQFullAttentionSpec(FullAttentionSpec) (kv_cache_interface.py:354-377)
# shape: (num_blocks, num_kv_heads, block_size, tq_config.slot_size_aligned)
# K+V 交织打包到单个 slot，无 K/V 分离维度
# slot = [key_packed | value_packed | padding]
# page_size_bytes 用 tq_slot_size 而非 head_size * dtype 公式
```

| 模式 | shape | page_size_bytes |
|---|---|---|
| 标准 | `(B, H, N, 2*D)` | `block_size * num_kv_heads * (head_size + head_size_v) * dtype_size` |
| TurboQuant | `(B, H, N, slot_size_aligned)` | `block_size * num_kv_heads * tq_slot_size` |

### 6.6 Encoder-Only Attention

```python
# EncoderOnlyAttentionSpec(AttentionSpec) (kv_cache_interface.py:742-746)
# 无 KV cache！
max_memory_usage_bytes = 0
```

### 6.7 Hidden State Cache

```python
# HiddenStateCacheSpec(MLAAttentionSpec) (kv_cache_interface.py:451-455)
# 继承 MLA 的 shape，但缓存的是 hidden state 而非 K/V latent
```

---

## 七、Spec 继承关系图

```
KVCacheSpec (frozen dataclass, block_size)
├── AttentionSpec (num_kv_heads, head_size, dtype, kv_quant_mode, ...)
│   ├── FullAttentionSpec (head_size_v, sliding_window, attention_chunk_size, non_causal)
│   │   ├── TQFullAttentionSpec (tq_slot_size)
│   │   ├── MLAAttentionSpec (cache_dtype_str, alignment, compress_ratio, model_version)
│   │   │   └── HiddenStateCacheSpec
│   │   ├── RSWASpec (rswa_window)
│   │   └── SinkFullAttentionSpec (sink_len)
│   ├── SlidingWindowSpec (sliding_window, head_size_v)
│   │   └── SlidingWindowMLASpec (cache_dtype_str, alignment, compress_ratio, model_version)
│   ├── ChunkedLocalAttentionSpec (attention_chunk_size)
│   ├── CrossAttentionSpec
│   └── EncoderOnlyAttentionSpec  (max_memory = 0)
├── MambaSpec (shapes, dtypes, mamba_type, mamba_cache_mode, ...)
│   └── 用于 Mamba1/Mamba2/GDN/ShortConv/LinearAttn/KDA
└── UniformTypeKVCacheSpecs (kv_cache_specs: dict) — 跨层统一类型但参数不同
```

---

## 八、block_dim 与 stride 汇总

`block_dim` 是 `num_blocks` 在 shape 中的位置索引，决定 `block_table` 用来 fancy index 的维度。

```python
# attention/backend.py:100-117
_S = 1234567
shape = cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size, ...)
return shape.index(_S)  # 0 = blocks-first, 1 = kv-first
```

| Backend / 类型 | 逻辑 shape | block_dim | 说明 |
|---|---|---|---|
| FlashAttention / FlashInfer / CPU / Triton / Flex | `(B, H, N, 2*D)` | **0** | blocks-first |
| ROCm Attn | `(2, B, N, H, D)` | **1** | kv-first |
| HPC | `(B, 2, N, H, D)` | **0** | blocks-first |
| MLA 系列 | `(B, N, D)` | **0** | blocks-first |
| TurboQuant | `(B, H, N, slot)` | **0** | blocks-first |
| MambaSpec 系列 | `(B, 1, 1, page_bytes)` | **0** | blocks-first |

> 混合模型中若同时存在 block_dim=0 和 block_dim=1 的层，`_update_hybrid_attention_mamba_layout()` 会将 block_dim=1 的层通过 `as_strided_()` 转为 block_dim=0，保证统一索引。

---

## 九、典型模型 KV cache shape 速查

### Llama-7B（Full Attention, FlashInfer, bf16, block_size=16）

```
单层: (num_blocks, 32, 16, 256)  # bf16, 32 KV heads, 128 head_size
#       B      H    N   2*128
# page_size_bytes = 16 * 32 * 256 * 2 = 262,144 B = 256 KB
```

### DeepSeek V3（MLA, FlashMLA, bf16, block_size=64）

```
# kv_lora_rank=512, qk_rope_head_dim=64, head_size=576
单层: (num_blocks, 64, 576)  # bf16
# page_size_bytes = 64 * 576 * 2 = 73,728 B = 72 KB
```

### DeepSeek V3.2（MLA, fp8_ds_mla, block_size=64）

```
单层: (num_blocks, 64, 656)  # 512B NoPE + 16B scale + 128B RoPE, uint8
# page_size_bytes = 64 * 656 * 1 = 41,984 B = 41 KB
```

### DeepSeek V4（MLA, fp8_ds_mla, compress_ratio=2, block_size=64）

```
storage_block_size = 64 // 2 = 32
单层: (num_blocks, 32, 584)  # 448B NoPE + 128B RoPE + 8B scale, uint8
# page_size_bytes = 32 * 584 * 1 = 18,688 B = 18.25 KB
```

### Mamba2（MambaSpec, bf16, block_size=64）

```
# 假设 intermediate_size=2048, n_groups=8, num_heads=128, head_dim=64,
#         state_size=128, conv_kernel=4, tp=1
# conv_dim = 2048 + 2 * 8 * 128 = 4096
# conv_state: (4096, 3)  →  4096 * 3 * 2 = 24,576 B
# temporal_state: (128, 64, 128)  →  128 * 64 * 128 * 2 = 2,097,152 B
# page_size_bytes = 24,576 + 2,097,152 = 2,121,728 B ≈ 2 MB
物理 tensor: (num_blocks, 1, 1, 2121728)
```

### GDN（MambaSpec, bf16, block_size=64, Qwen3-Next 配置）

```
# 假设 num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128,
#         conv_kernel_size=4, tp=1, num_spec=0
# conv_dim = 128 * 8 * 2 + 128 * 8 = 3072
# conv_state: (3072, 3)  →  3072 * 3 * 2 = 18,432 B
# temporal_state: (8, 128, 128)  →  8 * 128 * 128 * 2 = 262,144 B
# page_size_bytes = 18,432 + 262,144 = 280,576 B = 274 KB
物理 tensor: (num_blocks, 1, 1, 280576)
```

### Whisper（Cross Attention, FlashInfer, bf16, block_size=16）

```
# encoder 侧输出缓存，不释放
单层: (num_blocks, num_kv_heads, 16, 2 * head_size)  # 同 Full Attention
# max_memory = cdiv(max_encoder_len, block_size) * page_size_bytes
```

---

## 十、设计要点小结

1. **两大家族**：Attention 系列（`AttentionSpec`）按 token 存 K/V，有 `num_kv_heads × head_size` 维度；SSM 系列（`MambaSpec`）按递归状态存，是扁平字节缓冲。

2. **Shape 由 backend 决定**：同一 Spec 在不同 backend 下可能有不同逻辑 shape（如 FlashInfer 的 `(B, H, N, 2*D)` vs ROCm 的 `(2, B, N, H, D)`），但 `page_size_bytes`（字节数）一致。

3. **MLA 是特例**：不存储分离的 K/V，而是存储 latent 向量 `(B, N, D)`，无 `num_kv_heads` 维度（假设为 1）。fp8_ds_mla 使用自定义字节布局（656B 或 584B）。

4. **Mamba/GDN 扁平存储**：物理 tensor `(num_blocks, 1, 1, page_size_bytes)`，每 block 是 `page_size_bytes` 字节的扁平缓冲，`bind_kv_cache` 时按 conv_state + ssm_state 的 shape 切分 view。

5. **量化改变维度而非 dtype**：物理 dtype 通常固定为 uint8，量化模式改变 `head_dim`（INT4 减半，NVFP4 展开）或在最后一维内联 scale。

6. **stride 与 shape 分离**：`_reshape_attention_kv_cache` 先 view 出物理 contiguous 的 permuted shape，再 permute 回逻辑 shape，让 kernel 获得最优内存访问模式而 shape 保持不变。
