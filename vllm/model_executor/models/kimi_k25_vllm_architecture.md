# Kimi-K2.5 架构文档

***

## 一、完整架构概览

```
KimiK25ForConditionalGeneration (模型入口类)
├── vision_tower: MoonViT3dPretrainedModel
│   ├── patch_embed: MoonVision3dPatchEmbed
│   │   ├── proj (Conv2d)
│   │   └── pos_emb (Learnable2DInterpPosEmbDivided_fixed)
│   └── encoder: MoonViT3dEncoder
│       ├── rope_2d (Rope2DPosEmbRepeated)
│       ├── blocks[]: MoonViTEncoderLayer × N
│       │   ├── norm0 (LayerNorm)
│       │   ├── wqkv (QKVParallelLinear)
│       │   ├── wo (RowParallelLinear)
│       │   ├── attn (MMEncoderAttention)
│       │   ├── norm1 (LayerNorm)
│       │   └── mlp (MLP2)
│       └── final_layernorm (LayerNorm)
├── mm_projector: KimiK25MultiModalProjector
│   ├── pre_norm (LayerNorm)
│   ├── linear_1 (ReplicatedLinear)
│   ├── act (GELUActivation)
│   └── linear_2 (ReplicatedLinear)
└── language_model: DeepseekV2ForCausalLM (DeepseekV3 文本模型)
```

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25.py#L299`

***

## 二、核心组件详解

### 2.1 KimiK25ForConditionalGeneration

**模型入口类**, 继承自 `nn.Module`, 实现 `SupportsMultiModal`, `SupportsPP`, `SupportsQuant`, `SupportsEagle`, `SupportsEagle3` 接口。

**初始化组件**:

| 组件                | 类型                          | 说明                          |
| ----------------- | --------------------------- | --------------------------- |
| `vision_tower`    | `MoonViT3dPretrainedModel`  | 视觉编码器 (3D ViT)              |
| `mm_projector`    | `KimiK25MultiModalProjector`| 多模态投影器, 将视觉特征映射到文本空间       |
| `language_model`  | `DeepseekV2ForCausalLM`     | 文本语言模型 (基于 DeepseekV3 架构)   |
| `config`          | `KimiK25Config`             | 模型配置                        |
| `quant_config`    | `QuantizationConfig`        | 量化配置                        |
| `use_data_parallel` | `bool`                    | 视觉编码器是否使用数据并行               |
| `media_placeholder` | `int`                     | 媒体占位符 token ID              |

**forward 输入参数说明**:

`forward` 方法接收来自 vLLM model runner 阶段的输入参数。以下是从 vLLM Scheduler 到模型 forward 的完整调用链路:

```
Scheduler → model_runner.execute_model() → prepare_inputs() → self.model(**model_inputs) → KimiK25ForConditionalGeneration.forward()
```

| 参数                     | 类型                            | 形状                         | 来源与说明                                                                                                                                                |
| ---------------------- | ----------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_ids`            | `torch.Tensor`                | `(num_tokens,)`            | 展平的 token ID 序列, 由 vLLM model runner 从多个请求聚合而来。采用 **continuous batching** 机制, 将多个请求的 token 平铺成一维序列                                                                                   |
| `positions`            | `torch.Tensor`                | `(num_tokens,)`            | 每个 token 在序列中的绝对位置索引, 用于 RoPE 旋转位置编码。`positions[i] = num_computed_tokens[req_idx] + offset_in_req`                                                                                          |
| `intermediate_tensors` | `IntermediateTensors \| None` | `(num_tokens, H)`          | PP (流水线并行) 时传递的中间隐藏状态, 来自前一个 PP stage 的输出                                                                                                                                              |
| `inputs_embeds`        | `torch.Tensor \| None`        | `(num_tokens, H)`          | 可选的预计算嵌入, 优先于 `input_ids`。**场景**: 多模态模型的 encoder 输出或多模态嵌入。**优先级**: `inputs_embeds > input_ids`。当 `intermediate_tensors is not None` 时, `inputs_embeds` 被置为 `None` |

**关键说明**:

- `input_ids` 的形状为 `(num_tokens,)` 而非 `(batch_size, seq_len)`, 这是因为 vLLM 采用 **continuous batching** (连续批处理) 机制, 将多个请求的 token 平铺成一维序列
- `num_tokens = sum(seq_len for each request in batch)`, 即所有请求的 token 总数
- `positions` 用于旋转位置编码 (RoPE), 表示每个 token 在其原始序列中的位置 (从 0 开始)

**forward 流程**:

| 阶段  | 操作                                | 形状变化                          | 作用                  |
| --- | --------------------------------- | ----------------------------- | ------------------- |
| 1   | 检查 `intermediate_tensors`         | -                             | PP 阶段时清空 `inputs_embeds` |
| 2   | 调用 `self.language_model(...)`    | `(num_tokens,)` → `(num_tokens, H)` | 文本模型前向传播            |
| 3   | 返回 `hidden_states`               | `(num_tokens, H)`             | 返回隐藏状态或中间张量         |

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25.py#L439`

***

### 2.2 MoonViT3dPretrainedModel (视觉塔)

**视觉编码器主模型**, 实现 3D patch embedding、RoPE 位置编码和视频块的时间池化。

**初始化组件**:

| 组件               | 类型                          | 说明                                       |
| ---------------- | --------------------------- | ---------------------------------------- |
| `patch_embed`    | `MoonVision3dPatchEmbed`    | 3D patch 嵌入层, 将图像转换为 patch 并添加位置编码        |
| `encoder`        | `MoonViT3dEncoder`          | ViT 编码器, 包含多层 `MoonViTEncoderLayer`      |
| `merge_kernel_size` | `tuple[int, int]`        | patch 合并核大小 (默认 `(2, 2)`)                |
| `patch_size`     | `int`                       | patch 大小 (默认 `14`)                       |
| `merge_type`     | `str`                       | 合并类型 (默认 `sd2_tpool`, 空间下采样 2x + 时间池化)    |

**forward 流程**:

| 阶段 | 操作                          | 形状变化                                          | 作用                          |
| -- | --------------------------- | --------------------------------------------- | --------------------------- |
| 1  | `patch_embed(pixel_values, grid_thws)` | `(N, 3, H, W)` → `(num_patches, hidden_size)` | 图像分块 + 位置编码                 |
| 2  | `encoder(hidden_states, grid_thws)` | `(num_patches, hidden_size)` → `(num_patches, hidden_size)` | ViT 编码器前向传播                 |
| 3  | `tpool_patch_merger(...)`   | `(num_patches, hidden_size)` → `list[(num_merged, kh*kw, hidden_size)]` | 时间池化 + 空间下采样 2x             |

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L548`

***

### 2.3 MoonVision3dPatchEmbed (3D Patch 嵌入)

**3D patch 嵌入层**, 将输入图像转换为 patch 并添加 2D + 时间位置编码。

**初始化组件**:

| 组件         | 类型                                  | 说明                                |
| ---------- | ----------------------------------- | --------------------------------- |
| `proj`     | `nn.Conv2d`                         | 2D 卷积, 将图像分块并投影到 `out_dim` 维度      |
| `pos_emb`  | `Learnable2DInterpPosEmbDivided_fixed` | 2D 可学习位置嵌入 + 时间 sincos 位置嵌入        |
| `patch_size` | `tuple[int, int]`                 | patch 大小 (高度, 宽度)                  |

**forward 流程**:

```
输入: pixel_values (N, 3, H, W), grid_thws (B, 3)
  ↓
proj 卷积分块: (N, 3, H, W) → (N, out_dim, H/ps, W/ps)
  ↓
展平: (N, out_dim, H/ps, W/ps) → (N, num_patches, out_dim)
  ↓
位置编码: + 2D 可学习位置嵌入 + 时间 sincos 位置嵌入
  ↓
输出: (num_patches, out_dim)
```

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L183`

***

### 2.4 Learnable2DInterpPosEmbDivided_fixed (2D 可学习位置嵌入)

**2D 可学习位置嵌入**, 支持时间维度扩展和双三次插值。

**初始化组件**:

| 组件                  | 类型             | 说明                          |
| ------------------- | -------------- | --------------------------- |
| `weight`            | `nn.Parameter` | 2D 可学习位置嵌入权重, 形状 `(H, W, dim)` |
| `time_weight`       | `buffer`       | 时间维度 sincos 位置嵌入             |
| `interpolation_mode` | `str`         | 插值模式 (默认 `bicubic`)         |

**forward 流程**:

| 步骤 | 操作                                                                 | 说明                          |
| -- | ------------------------------------------------------------------ | --------------------------- |
| 1  | 对每个 `(t, h, w)` 网格: 若 `(h, w) == weight.shape[:-1]`, 直接使用权重; 否则双三次插值 | 2D 位置嵌入                     |
| 2  | 若 `t > 1`: `pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + time_weight[0:t]` | 时间维度扩展 + 时间位置嵌入             |
| 3  | 拼接所有 patch 的位置嵌入, 与输入相加                                            | `x + pos_embs`              |

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L126`

***

### 2.5 MoonViT3dEncoder (ViT 编码器)

**ViT 编码器堆叠**, 包含多层 `MoonViTEncoderLayer` 和 2D RoPE 位置编码。

**初始化组件**:

| 组件                | 类型                          | 说明                          |
| ----------------- | --------------------------- | --------------------------- |
| `rope_2d`         | `Rope2DPosEmbRepeated`     | 2D 旋转位置编码, 最大尺寸 `512×512`   |
| `blocks`          | `nn.ModuleList`             | `MoonViTEncoderLayer` 层堆叠   |
| `final_layernorm`  | `nn.LayerNorm`             | 最终层归一化                      |
| `video_attn_type` | `str`                       | 视频注意力类型 (默认 `spatial_temporal`) |

**forward 流程**:

| 阶段 | 操作                                                | 形状变化                                          | 说明                          |
| -- | ------------------------------------------------- | --------------------------------------------- | --------------------------- |
| 1  | `rope_2d.get_freqs_cis(grid_thws, device)`       | -                                             | 计算 2D RoPE 频率               |
| 2  | 计算 `cu_seqlens` (累积序列长度)                          | -                                             | 用于 packed attention         |
| 3  | 遍历 `blocks`: `block(hidden_states, cu_seqlens, rope_freqs_cis)` | `(num_patches, H)` → `(num_patches, H)` | 编码器层前向传播                   |
| 4  | `final_layernorm(hidden_states)`                  | `(num_patches, H)` → `(num_patches, H)`       | 最终归一化                       |

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L463`

***

### 2.6 Rope2DPosEmbRepeated (2D 旋转位置编码)

**2D 旋转位置编码**, 支持多分辨率。

**初始化参数**:

| 参数           | 类型    | 说明                       |
| ------------ | ----- | ------------------------ |
| `dim`        | `int` | 头维度 (必须被 4 整除)           |
| `max_height` | `int` | 最大高度 (默认 `512`)         |
| `max_width`  | `int` | 最大宽度 (默认 `512`)         |
| `theta_base` | `int` | RoPE 基础频率 (默认 `10000`)  |

**频率计算流程**:

```
1. 计算 flat_pos = [0, 1, ..., N-1], N = max_height * max_width
2. x_pos = flat_pos % max_width
3. y_pos = flat_pos // max_width
4. dim_range = [0, 4, 8, ..., dim-4] (dim/4 个)
5. freqs = 1.0 / (theta_base ^ (dim_range / dim))
6. x_freqs = outer(x_pos, freqs), y_freqs = outer(y_pos, freqs)
7. x_cis = polar(1, x_freqs), y_cis = polar(1, y_freqs)
8. freqs_cis = cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
9. reshape 为 (max_height, max_width, dim/2)
```

**get_freqs_cis 方法**: 根据 `grid_thws` 提取对应位置的频率, 并按时间维度 `t` 重复。

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L228`

***

### 2.7 MoonViTEncoderLayer (ViT 编码器层)

**单层 ViT 编码器**, 包含自注意力和 MLP, 支持 TP/DP。

**初始化组件**:

| 组件                            | 类型                          | 说明                                |
| ----------------------------- | --------------------------- | --------------------------------- |
| `norm0`                       | `nn.LayerNorm`             | 注意力前归一化                           |
| `wqkv`                        | `QKVParallelLinear`        | QKV 合并投影 (支持 TP/DP)               |
| `wo`                          | `RowParallelLinear`        | 输出投影 (支持 TP/DP)                   |
| `attn`                        | `MMEncoderAttention`      | 多模态编码器注意力                         |
| `norm1`                       | `nn.LayerNorm`             | MLP 前归一化                          |
| `mlp`                         | `MLP2`                     | 两层 MLP                            |
| `num_heads`                   | `int`                      | 注意力头数                             |
| `hidden_dim`                  | `int`                      | 隐藏层维度                             |
| `hidden_size_per_attention_head` | `int`                   | 每头维度                              |
| `tp_size`                     | `int`                      | 张量并行大小                            |
| `num_attention_heads_per_partition` | `int`                 | 每分区头数                             |
| `use_data_parallel`           | `bool`                     | 是否使用数据并行                          |

**forward 流程**:

```
┌─────────────────────────────────────────────────────────────┐
│  注意力子流程: norm0 → attention_qkvpacked → residual add     │
│  FFN子流程: norm1 → mlp → residual add                       │
└─────────────────────────────────────────────────────────────┘
```

**attention_qkvpacked 流程**:

| 步骤 | 操作                                                              | 形状变化                                              |
| -- | --------------------------------------------------------------- | ------------------------------------------------- |
| 1  | `wqkv(x)`                                                       | `(seqlen, hidden_dim)` → `(seqlen, 3*nheads*headdim)` |
| 2  | reshape & unbind                                                | → `xq, xk, xv` 各为 `(seqlen, nheads, headdim)`      |
| 3  | `apply_rope(xq, xk, rope_freqs_cis)`                            | 应用 2D RoPE                                       |
| 4  | `attn(xq, xk, xv, cu_seqlens, max_seqlen)`                     | packed attention                                 |
| 5  | reshape & `wo(attn_out)`                                        | 输出投影                                             |

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L340`

***

### 2.8 MLP2 (两层 MLP)

**两层 MLP**, 支持张量并行。

**初始化组件**:

| 组件           | 类型                   | 说明                    |
| ------------ | -------------------- | --------------------- |
| `fc0`        | `ColumnParallelLinear` | 第一层 (升维), 支持 TP/DP    |
| `fc1`        | `RowParallelLinear`   | 第二层 (降维), 支持 TP/DP    |
| `activation` | `callable`            | 激活函数                  |

**forward 流程**:

```python
x, _ = self.fc0(x)        # 升维
x = self.activation(x)   # 激活
x, _ = self.fc1(x)        # 降维
return x
```

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L300`

***

### 2.9 tpool_patch_merger (时间池化 patch 合并)

**时间池化 patch 合并器**, 实现空间下采样 2x 和时间维度池化。

**参数**:

| 参数                 | 类型             | 说明                  |
| ------------------ | -------------- | ------------------- |
| `x`                | `torch.Tensor` | 输入特征, 形状 `(total_patches, d)` |
| `grid_thws`        | `torch.Tensor` | 网格尺寸 `(B, 3)`       |
| `merge_kernel_size` | `tuple[int, int]` | 合并核大小 `(kh, kw)` (默认 `(2, 2)`) |

**forward 流程**:

```
输入: x (total_patches, d), grid_thws (B, 3)
  ↓
按媒体项切分: x.split(lengths, dim=0)
  ↓
对每个媒体项 (seq, (t, h, w)):
  1. reshape: (t, h, w, d) → (t, nh, kh, nw, kw, d), 其中 nh=h//kh, nw=w//kw
  2. 时间池化: v.mean(dim=0) → (nh, kh, nw, kw, d)
  3. 空间重排: (nh, kh, nw, kw, d) → (nh*nw, kh*kw, d)
  ↓
输出: list[(num_merged_patches, kh*kw, d)]
```

**核心机制**: 先对时间维度取平均 (时间池化), 再对空间维度进行 2x2 分组, 将每个 `kh*kw` (默认 `4`) 个 patch 合并为一个输出 token, 后续通过 MM projector 展平处理。

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L524`

***

### 2.10 KimiK25MultiModalProjector (多模态投影器)

**多模态投影器**, 将视觉特征映射到文本隐藏空间。

**初始化组件**:

| 组件           | 类型                   | 说明                                       |
| ------------ | -------------------- | ---------------------------------------- |
| `pre_norm`   | `nn.LayerNorm`      | 预归一化层                                    |
| `linear_1`   | `ReplicatedLinear`   | 第一层线性变换 (输入: `hidden_size * kh * kw`)    |
| `linear_2`   | `ReplicatedLinear`   | 第二层线性变换 (输出: `mm_hidden_size`)           |
| `act`        | `GELUActivation`     | GELU 激活函数                                |
| `hidden_size` | `int`               | 合并后的隐藏维度 = `config.hidden_size * merge_h * merge_w` |
| `use_data_parallel` | `bool`        | 是否使用数据并行                                 |

**forward 流程**:

```python
hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
hidden_states, _ = self.linear_1(hidden_states)   # 升维/降维
hidden_states = self.act(hidden_states)           # GELU 激活
hidden_states, _ = self.linear_2(hidden_states)   # 投影到文本空间
return hidden_states
```

**形状变化**:

- 输入: `(num_merged_patches, kh*kw, hidden_size)` → 展平为 `(num_merged_patches, hidden_size * kh * kw)`
- 输出: `(num_merged_patches, mm_hidden_size)`, 其中 `mm_hidden_size` = 文本模型隐藏维度

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L657`

***

### 2.11 vision_tower_forward (视觉塔前向传播)

**视觉塔前向传播函数**, 支持数据并行分片。

**参数**:

| 参数                | 类型             | 说明                          |
| ----------------- | -------------- | --------------------------- |
| `vision_tower`    | `Any`          | 视觉塔模型                        |
| `pixel_values`    | `torch.Tensor` | 像素值                          |
| `grid_thw`        | `torch.Tensor` | 网格尺寸                         |
| `mm_projector`    | `Any`          | 多模态投影器                       |
| `use_data_parallel` | `bool`       | 是否使用数据并行                     |

**forward 流程**:

```
if use_data_parallel:
    vt_outputs = run_dp_sharded_mrope_vision_model(...)  # DP 分片
else:
    vt_outputs = vision_tower(pixel_values, grid_thw)    # 普通前向
  ↓
tensors = mm_projector_forward(mm_projector, list(vt_outputs))  # 投影器前向
  ↓
return list(tensors)
```

**mm_projector_forward**: 将视觉塔输出批量拼接, 经过投影器处理后, 按原始大小切分返回。

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25_vit.py#L631`

***

## 三、多模态处理流程

### 3.1 整体处理流程

```
用户输入 (文本 + 图像/视频)
  ↓
KimiK25MultiModalProcessor._call_hf_processor()
  ↓
KimiK25Processor.__call__()
  ├── image_processor.preprocess(vision_chunks)  → pixel_values, grid_thws
  └── tokenizer(text) → input_ids
       └── 媒体占位符扩展: 1 个 <|media_pad|> → N 个 <|media_pad|>
  ↓
KimiK25ForConditionalGeneration
  ├── embed_multimodal(pixel_values, grid_thws)
  │   ├── _parse_and_validate_media_input()
  │   ├── vision_tower_forward() → vision_embeddings
  │   └── 返回 NestedTensors
  └── forward(input_ids, positions, ...)
      └── language_model(...) → hidden_states
```

### 3.2 模态类型

| 模态            | 类型             | 说明                                       |
| ------------- | -------------- | ---------------------------------------- |
| `image`       | `VisionChunkImage` | 单张图像, 占位符 `<|media_begin|>image<|media_content|><|media_pad|><|media_end|>` |
| `video_chunk` | `VisionChunkVideo` | 视频块 (多帧图像), 通过时间池化处理                     |

### 3.3 媒体占位符

- **媒体占位符 token**: `<|media_pad|>`, token ID 默认 `163605`
- **占位符字符串**:
  - image: `<|media_begin|>image<|media_content|><|media_pad|><|media_end|>`
  - video: `<|kimi_k25_video_placeholder|>` (待替换)

### 3.4 输入数据结构

**KimiK25MediaPixelInputs**:

| 字段            | 类型                          | 形状                          | 说明                          |
| ------------- | --------------------------- | --------------------------- | --------------------------- |
| `pixel_values` | `torch.Tensor | list[torch.Tensor]` | `(np, 3, ps, ps)`           | 所有 patch 拼接, `np` = patch 总数 |
| `grid_thws`   | `torch.Tensor`              | `(nm, 3)`                   | 每个媒体项的网格尺寸 `[N_t, N_h, N_w]` |

**维度说明**:

- `np`: patch 总数 (所有媒体项展平)
- `ps`: patch 大小
- `nm`: 媒体项数量
- `N_t * N_h * N_w`: 单个媒体项的 patch 数

### 3.5 _parse_and_validate_media_input 流程

| 步骤 | 操作                                                              | 说明                          |
| -- | --------------------------------------------------------------- | --------------------------- |
| 1  | `pixel_values = kwargs.pop("pixel_values", None)`              | 提取像素值                       |
| 2  | 若 `pixel_values` 为 list, `torch.cat(pixel_values, dim=0)`       | 拼接                          |
| 3  | 若形状为 5D 或 3D, reshape 为 4D                                     | 展平 batch 维度                 |
| 4  | 转换为视觉塔 dtype                                                   | 类型转换                        |
| 5  | `grid_thws.reshape(-1, grid_thws.shape[-1])`                   | 展平中间维度                      |
| 6  | 断言 `grid_thws.ndim == 2 and grid_thws.size(1) == 3`           | 形状校验                        |

***

## 四、配置详解

### 4.1 KimiK25Config (主配置)

**文件位置**: `vllm/vllm/transformers_utils/configs/kimi_k25.py#L56`

| 参数                          | 类型                          | 默认值          | 说明                          |
| --------------------------- | --------------------------- | ------------ | --------------------------- |
| `vision_config`             | `KimiK25VisionConfig`       | 默认实例         | 视觉塔和投影器配置                   |
| `text_config`               | `DeepseekV3Config`          | 默认实例         | 文本模型配置 (DeepseekV3)         |
| `ignore_index`              | `int`                       | `-100`       | 损失函数忽略索引                    |
| `media_placeholder_token_id` | `int`                      | `163605`     | 媒体占位符 token ID              |
| `pad_token_id`              | `int`                       | `0`          | 填充 token ID                 |
| `use_unified_vision_chunk`  | `bool`                      | `False`      | 是否使用统一视频块                   |
| `video_placeholder`          | `str`                       | `<|kimi_k25_video_placeholder|>` | 视频占位符字符串                    |

**属性**:

- `hidden_size`: 从 `text_config.hidden_size` 获取
- `vocab_size`: 从 `text_config.vocab_size` 获取
- `quantization_config`: 从 `text_config.quantization_config` 传播

### 4.2 KimiK25VisionConfig (视觉配置)

**文件位置**: `vllm/vllm/transformers_utils/configs/kimi_k25.py#L9`

**视觉塔参数**:

| 参数                    | 类型             | 默认值      | 说明                          |
| --------------------- | -------------- | -------- | --------------------------- |
| `patch_size`          | `int`          | `14`     | patch 大小                    |
| `init_pos_emb_height` | `int`          | `64`     | 初始位置嵌入高度                    |
| `init_pos_emb_width`  | `int`          | `64`     | 初始位置嵌入宽度                    |
| `init_pos_emb_time`   | `int`          | `4`      | 初始位置嵌入时间                    |
| `pos_emb_type`        | `str`          | `divided_fixed` | 位置嵌入类型                      |
| `num_attention_heads` | `int`          | `16`     | 注意力头数                       |
| `num_hidden_layers`   | `int`          | `27`     | 编码器层数                       |
| `hidden_size`         | `int`          | `1152`   | 隐藏层维度                       |
| `intermediate_size`   | `int`          | `4304`   | MLP 中间层维度                   |
| `merge_kernel_size`   | `tuple[int, int]` | `(2, 2)` | patch 合并核大小                 |
| `video_attn_type`     | `str`          | `spatial_temporal` | 视频注意力类型                     |
| `merge_type`          | `str`          | `sd2_tpool` | 合并类型 (空间下采样 2x + 时间池化)      |

**MM 投影器参数**:

| 参数                      | 类型             | 默认值      | 说明                          |
| ----------------------- | -------------- | -------- | --------------------------- |
| `mm_projector_type`     | `str`          | `patchmerger` | 投影器类型                       |
| `mm_hidden_size`        | `int \| None`  | `None`   | 投影器输出维度 (默认 = `text_config.hidden_size`) |
| `projector_hidden_act`  | `str`          | `gelu`   | 投影器激活函数                     |
| `projector_ln_eps`      | `float`        | `1e-5`   | LayerNorm epsilon            |

***

## 五、多模态处理器

### 5.1 KimiK25ProcessingInfo

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25.py#L98`

**初始化流程**:

| 步骤 | 操作                                                              | 说明                          |
| -- | --------------------------------------------------------------- | --------------------------- |
| 1  | `hf_config = self.get_hf_config()`                              | 获取 HF 配置                     |
| 2  | `tokenizer = self.get_tokenizer()`                              | 获取 tokenizer                 |
| 3  | `image_processor = cached_get_image_processor(...)`             | 获取图像处理器                      |
| 4  | 解析 `media_token_id`: 从 tokenizer 解析 `<|media_pad|>`              | 处理 transformers v5 token ID 重映射 |
| 5  | 若解析值有效且与 config 不一致, 使用 tokenizer 值并 patch config               | 一致性处理                       |
| 6  | 初始化 `KimiK25Processor(tokenizer, image_processor, media_token_id)` | HF 处理器                      |
| 7  | `media_tokens_calculator = image_processor.media_tokens_calculator` | 媒体 token 计算器                |

**get_supported_mm_limits**: 返回 `{"vision_chunk": None}`, 表示 `vision_chunk` 模态数量无限制。

### 5.2 KimiK25DummyInputsBuilder

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25.py#L165`

**功能**: 为模型 profiling 构建虚拟输入。

**get_dummy_text**: 返回 `media_token * num_media`。

**get_dummy_mm_items**:

- 构建虚拟视频块 (4 帧, `3000×3000`) 和虚拟图像 (`3000×3000`)
- 计算各自的 token 数, 返回 token 数较大的那个

**MaxImageTokenMeta**: `width=3000, height=3000`。

### 5.3 KimiK25MultiModalProcessor

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25.py#L215`

**_get_mm_fields_config**: 定义媒体输入字段的切片方式。

| 字段             | 配置                                    | 说明                          |
| -------------- | ------------------------------------- | --------------------------- |
| `pixel_values` | `MultiModalFieldConfig.flat_from_sizes("vision_chunk", grid_sizes)` | 按 `grid_sizes` 切片, `grid_sizes = grid_thws.prod(-1)` |
| `grid_thws`    | `MultiModalFieldConfig.batched("vision_chunk")` | 批量维度                        |

**_get_prompt_updates**: 定义媒体占位符的替换规则。

- `target`: `[media_token_id]`
- `replacement`: 根据 `media_tokens_calculator(media[item_idx])` 返回 `[media_token_id] * num_media_token`

***

## 六、权重映射

### 6.1 hf_to_vllm_mapper

**文件位置**: `vllm/vllm/model_executor/models/kimi_k25.py#L322`

| 原始前缀                          | 新前缀                              | 说明                          |
| ----------------------------- | -------------------------------- | --------------------------- |
| `language_model.layers.`      | `language_model.model.layers.`   | NVFP4 checkpoint 兼容性         |
| `mm_projector.proj.0`         | `mm_projector.linear_1`          | MM 投影器第一层                   |
| `mm_projector.proj.2`         | `mm_projector.linear_2`          | MM 投影器第二层                   |

**参考**: https://github.com/vllm-project/vllm/pull/33346#issuecomment-3851475033

***

## 七、并行策略

### 7.1 支持的并行模式

```
├── TP (Tensor Parallel): 张量并行 (语言模型 + 视觉塔)
├── PP (Pipeline Parallel): 流水线并行 (SupportsPP)
├── DP (Data Parallel): 数据并行 (视觉塔, mm_encoder_tp_mode="data")
└── Quant: 量化支持 (SupportsQuant)
```

### 7.2 视觉塔并行

| 模式                | 配置                                          | 说明                          |
| ----------------- | ------------------------------------------- | --------------------------- |
| 张量并行 (TP)         | `mm_encoder_tp_mode != "data"`              | `QKVParallelLinear`, `RowParallelLinear` 启用 TP              |
| 数据并行 (DP)         | `mm_encoder_tp_mode == "data"`              | `disable_tp=True`, 使用 `run_dp_sharded_mrope_vision_model`   |

### 7.3 量化配置处理

**_maybe_ignore_quant_config**: 若 `quant_config` 为 `CompressedTensorsConfig`, 返回 `None` (视觉塔和投影器不量化)。

### 7.4 EAGLE 支持

- `SupportsEagle`: 支持 EAGLE 投机解码
- `SupportsEagle3`: 支持 EAGLE3
- `set_aux_hidden_state_layers`: 设置辅助隐藏状态层
- `get_eagle3_aux_hidden_state_layers`: 获取 EAGLE3 辅助隐藏状态层
- `supports_encoder_tp_data = True`: 支持编码器 TP 数据

***

## 八、文件位置参考

| 文件/模块                | 路径                                                              |
| -------------------- | --------------------------------------------------------------- |
| 主模型文件                | `vllm/vllm/model_executor/models/kimi_k25.py`                   |
| 视觉塔文件                | `vllm/vllm/model_executor/models/kimi_k25_vit.py`               |
| 模型配置                 | `vllm/vllm/transformers_utils/configs/kimi_k25.py`              |
| HF 处理器               | `vllm/vllm/transformers_utils/processors/kimi_k25.py`           |
| 多模态输入                | `vllm/vllm/multimodal/inputs.py`                                |
| 多模态处理                | `vllm/vllm/multimodal/processing.py`                            |
| MM 编码器注意力            | `vllm/vllm/model_executor/layers/attention/mm_encoder_attention.py` |
| 视觉工具                 | `vllm/vllm/model_executor/models/vision.py`                     |
| 线性层                  | `vllm/vllm/model_executor/layers/linear.py`                     |
| 量化配置                 | `vllm/vllm/model_executor/layers/quantization/`                 |

***

## 九、架构关系图

```
                    ┌─────────────────────────────────────────┐
                    │    KimiK25ForConditionalGeneration      │
                    │         (模型入口类)                       │
                    └─────────────────────┬─────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
┌───────▼───────────────┐    ┌────────────▼─────────────┐    ┌─────────────▼─────────────┐
│   vision_tower        │    │      mm_projector        │    │      language_model       │
│ (MoonViT3dPretrained  │    │ (KimiK25MultiModal      │    │  (DeepseekV2ForCausalLM)  │
│       Model)          │    │     Projector)          │    │                           │
└──────────┬───────────┘    └────────────┬─────────────┘    └───────────────────────────┘
           │                             │
    ┌──────▼──────┐               ┌──────▼──────┐
    │ patch_embed │               │  pre_norm   │
    │   encoder   │               │  linear_1   │
    └──────┬──────┘               │    act      │
           │                      │  linear_2   │
    ┌──────▼────────────────┐     └─────────────┘
    │  MoonViT3dEncoder    │
    │  ├── rope_2d         │
    │  ├── blocks[]        │
    │  └── final_layernorm │
    └──────┬───────────────┘
           │
    ┌──────▼──────────────────┐
    │ MoonViTEncoderLayer ×N │
    │ ├── norm0 → wqkv → wo  │
    │ ├── attn               │
    │ └── norm1 → mlp        │
    └────────────────────────┘
```

**数据流**:

```
pixel_values (N, 3, H, W)
    │
    ▼
┌─────────────────────────────────────────────┐
│ vision_tower                                │
│   patch_embed: (N, 3, H, W) → (np, hidden)  │
│   encoder: (np, hidden) → (np, hidden)      │
│   tpool_patch_merger: → list[(nm, kh*kw, d)]│
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ mm_projector                                │
│   pre_norm → linear_1 → act → linear_2     │
│   (nm, kh*kw, d) → (nm, mm_hidden_size)    │
└────────────────────┬────────────────────────┘
                     │
                     ▼
              vision_embeddings
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ language_model (DeepseekV2)                 │
│   input_ids + positions → hidden_states     │
└─────────────────────────────────────────────┘
```
