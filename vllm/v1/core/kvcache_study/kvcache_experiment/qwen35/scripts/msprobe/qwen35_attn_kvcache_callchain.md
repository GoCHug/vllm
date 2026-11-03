# Qwen3.5 KVCache 读写调用链（full-attention + GDN 双分支）

Qwen3.5-0.8B 在 vLLM 0.23.0 + vllm-ascend 0.23.0 上，两条模型分支到 NPU 算子之间，
KVCache 的写入与读取分别发生在哪个函数、哪个算子：

- **full-attention 分支**（6/24 层）：读写的 KVCache 是 `key_cache` / `value_cache`；
- **GDN 分支**（18/24 层）：读写的 KVCache 是 `conv_state` / `ssm_state`。

> 一句话结论：模型层传进 `self.attn` 的 `q/k/v` **只是本步新算出来的当前 token**，不含任何历史 KV；
> 历史 KV 的**写**在 `AscendAttentionBackendImpl.reshape_and_cache`，**读**在 `forward_fused_infer_attention` /
> `forward_paged_attention` 里，并且读的是**整张 cache 张量 + block_table 分页寻址**，不是拼接好的连续 KV。
> 另一条分支 GDN 层（`linear_attention`，18/24 层）不写 attn 的 K/V，读写的是 **conv_state + ssm_state**，
> 与 full-attention 共用同一套 block/页几何。

源码位置约定：
`vllm/` = `/vllm-workspace/vllm`，`vllm-ascend/` = `/vllm-workspace/vllm-ascend`。

---

## 1. 总调用树

### 1.1 init：模块构建

```
Qwen3_5DecoderLayer.__init__                                 vllm/model_executor/models/qwen3_5.py:120
│
├── self.self_attn    = Qwen3NextAttention(...)              qwen3_next.py:349（类定义 :207）
│     │  forward 被 patch 成 AscendQwen3NextAttention.forward   patch_qwen3_5.py:196 → :42
│     └── 内部：qkv_proj = QKVParallelLinear（:242）、o_proj = RowParallelLinear（:252）、
│           rotary_emb = get_rope（M-RoPE，:260）、attn = Attention（:267，impl=AscendAttentionBackendImpl）、
│           q_norm / k_norm = Qwen3NextRMSNorm（:283-284）
├── self.linear_attn  = QwenGatedDeltaNetAttention(...)      qwen3_next.py:342
│     │  forward / _forward_core / _warmup_prefill_kernels 被 patch 成 AscendGatedDeltaNetAttention 版本
│     │  （patch_qwen3_5.py:207 / :208 / :209；310P：_forward_core = AscendGatedDeltaNetAttention310，:204）
│     └── 另 patch _split_ba_for_tp / get_state_shape / get_attn_backend（:197-199）
├── self.mlp                 = Qwen3NextSparseMoeBlock / Qwen3NextMLP        qwen3_next.py:366-371
├── self.input_layernorm     = Qwen3NextRMSNorm              qwen3_next.py:379
├── self.post_attention_layernorm = Qwen3NextRMSNorm         qwen3_next.py:382
└── self.layer_scale → self.attn_layer_scale / self.ffn_layer_scale（Parameter）  qwen3_next.py:386-395
```

### 1.2 forward：执行

```
Qwen3_5DecoderLayer(Qwen3NextDecoderLayer)                     vllm/model_executor/models/qwen3_5.py:120
└── AscendQwen3_5DecoderLayer.forward(hidden_states, residual, positions)    patch_qwen3_5.py:91
        │
        ├── self.layer_type == "full_attention"（6/24 层）：self.self_attn(positions, output, hidden_states)  :119-124
        │   └── self.self_attn.forward(positions, output, hidden_states)  # patch_qwen3_5.py:42 AscendQwen3NextAttention
        │       ├── qkv, _ = self.qkv_proj(hidden_states)                               :43
        │       ├── q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(qkv, self.heads, self.head_size)  :44
        │       ├── self.attn(q, k, v)                    # vllm/model_executor/layers/attention/attention.py:437
        │       │   └── torch.ops.vllm.unified_attention_with_output(query, key, value, output,...)  :521-528 → 注册的 Python 实现 :734
        │       │             ├── attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)   :749
        │       │             └── self.impl.forward(self, q, k, v, kv_cache, attn_metadata, output)   :751
        │       │                 └── AscendAttentionBackendImpl.forward(...)  # vllm_ascend/attention/attention_v1.py:1479
        │       │                     ├── self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]  :1527
        │       │                     ├──【写 KVCache】self.reshape_and_cache(q, k, v, kv_cache, meta, out)  :1532
        │       │                     │   └── AscendAttentionBackendImpl.reshape_and_cache  :1430
        │       │                     │       ├── slots = attn_metadata.slot_mapping
        │       │                     │       ├── DeviceOperator.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping=slots)  :1444
        │       │                     │       │   └── torch_npu._npu_reshape_and_cache(...)  # device_op.py:49  ★ 写算子(A2/A3)
        │       │                     │       └── notify_kv_cache_written()  :1453
        │       │                     ├──【读 KVCache】forward_impl(q, k, v, kv_cache, meta, out)  :1456
        │       │                     │   ├── 条件：DecodeOnly && sliding_window is None && using_paged_attention
        │       │                     │   │   └── forward_paged_attention  :1363
        │       │                     │   │       └── torch_npu._npu_paged_attention(query, key_cache, value_cache,
        │       │                     │   │               block_table=block_tables, context_lens=seq_lens, out=output)  ★ 读算子(PA)
        │       │                     │   └── 否则 forward_fused_infer_attention  :1234
        │       │                     │       ├── _get_fia_params(key, value, meta, kv_cache)  :1174
        │       │                     │       │     PrefillNoCache : block_table=None, key/value = 当前 token
        │       │                     │       │     PrefillCacheHit: key = self.key_cache.view(num_block, block_size, -1)  :1204
        │       │                     │       │     DecodeOnly    : key = self.key_cache.view(num_block, block_size, -1)  :1213
        │       │                     │       └── DeviceOperator.npu_fused_infer_attention_score(...)  # device_op.py:55
        │       │                     │           └── torch_npu.npu_fused_infer_attention_score(query, key, value, ...)  :92  ★ 读算子(FIA)
        │       │                     └── output[:num_tokens] = attn_output[:num_tokens]; return output  :1546-1547
        │       └── output[:], _ = self.o_proj(attn_output)                                 :87
        │
        └── self.layer_type == "linear_attention"（18/24 层）：self.linear_attn(hidden_states, output)  :114-118
            └── QwenGatedDeltaNetAttention.forward(hidden_states, output)  qwen3_next.py:342
                └── self._forward_method → _forward_core  qwen_gdn_linear_attn.py:844 / :849 / :1260
                    └── AscendGatedDeltaNetAttention._forward_core(...)  vllm-ascend/vllm_ascend/ops/gdn.py:147
                        ├──【1. 取状态】conv_state / ssm_state = self.kv_cache[0] / [1]  :172 / :1308
                        ├──【2. conv 卷积】torch.ops._C_ascend.npu_causal_conv1d_custom  ★ 读写 conv_state（原地）
                        ├──【3. 门控】DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias) → g, beta  :313
                        ├──【4. q/k 归一】l2norm_fwd(query/key)  :339 / :367 / :423
                        ├──【5. decode 递归】torch.ops._C_ascend.npu_recurrent_gated_delta_rule  ★ 读写 ssm_state（原地）
                        └──【6. prefill 递归】chunk_gated_delta_rule（分块并行）  :402
```

---

## 2. attn 层：KVCache 读写

### 2.1 写路径

| 项 | 内容 |
|---|---|
| 写入位置 | `AscendAttentionBackendImpl.forward` → `reshape_and_cache`（`attention_v1.py:1430`） |
| 写什么 | 只有**当前 token** 的 K/V（`key[:num_actual_tokens]`） |
| 写到哪里 | `key_cache` / `value_cache`，形状 `[num_blocks, block_size, num_kv_heads, head_size]`（`kv_cache` 整体 `[2, ...]`） |
| 定位方式 | `slot_mapping`（每个 token 一个 slot 下标），由 scheduler 的 block 分配决定 |
| 底层算子 | A2/A3：`torch_npu._npu_reshape_and_cache`；A5：`torch_npu.npu_scatter_pa_kv_cache`（`cache_mode="Norm"`） |
| 特性 | **原地写**（in-place），Python 层拿不到"写后副本"，只能看到同一块张量被改写 |

### 2.2 读路径

| 项 | FIA（`forward_fused_infer_attention`） | PA（`forward_paged_attention`） |
|---|---|---|
| 触发条件 | 默认 / prefill / cache-hit | `DecodeOnly` + 无 sliding window + `using_paged_attention()` |
| 传入的 key/value | `_get_fia_params` 的返回值 | 不走 `key/value`，直接传 `key_cache`/`value_cache` |
| cache 形态 | CacheHit/Decode 时 `self.key_cache.view(num_block, block_size, -1)` | 原样整张 cache |
| 寻址 | `block_table` + `actual_seq_lengths_kv` | `block_table=attn_metadata.block_tables` + `context_lens=attn_metadata.seq_lens` |
| 底层算子 | `torch_npu.npu_fused_infer_attention_score` | `torch_npu._npu_paged_attention` |


---

## 3. GDN 层：conv_state / ssm_state 读写

### 3.1 写路径

| 项 | 内容 |
|---|---|
| 写入位置 | `AscendGatedDeltaNetAttention._forward_core`（`vllm-ascend/vllm_ascend/ops/gdn.py:147`，patch 替换而来） |
| conv_state | `torch.ops._C_ascend.npu_causal_conv1d_custom` 原地读写（gdn.py:200/247/270/291） |
| ssm_state | decode：`torch.ops._C_ascend.npu_recurrent_gated_delta_rule` 原地（:345/369/427）；prefill：`chunk_gated_delta_rule` 读拷贝 → 清初态 → 整块写回（:400-415） |
| 定位方式 | `cache_indices` / `ssm_state_indices`（每序列一个状态行，无 token 级 slot） |
| 特性 | 同样**原地写**；prefill 是"读拷贝、写回"两段式，不是 kernel 内 in-place |

### 3.2 读路径

同样把整张状态张量交给算子，按 `cache_indices` / `ssm_state_indices` 选行；
decode 由算子直接读原地状态，prefill 由 Python 层先拷出 `initial_state`（gdn.py:400）
再喂给 `chunk_gated_delta_rule`。

---


## 4. KVCache 张量形态与寻址

```
kv_cache                     : [2, num_blocks, block_size, num_kv_heads, head_size]
  kv_cache[0] = key_cache    : [num_blocks, block_size, num_kv_heads, head_size]
  kv_cache[1] = value_cache  : 同上
写入定位  slot_mapping[t]    : token t 落到 (block_id, offset) → 压平成 slot 下标
读取定位  block_table[q, i]  : 第 q 条序列的第 i 个逻辑块 → 物理 block_id
有效长度  seq_lens / actual_seq_lengths_kv
```

full-attn 层与 GDN 层**共用同一个 BlockPool**，但各自持有独立的 block_table；页宽由 attn 的 `block_size` 决定。

---

## 5. dump / 观测挂点

### 5.1 msprobe 算子级 dump（已在 `1024_dump` / `2048_dump` 中验证可用）

| 想看什么 | dump 目标（msprobe 里的名字） | 说明 |
|---|---|---|
| 写入的当前 token K/V | `NPU._npu_reshape_and_cache.*.forward.input.key` / `.input.value` | 本步新算的 KV，即写入源 |
| 写入目标整张 cache | `NPU._npu_reshape_and_cache.*.forward.input.key_cache` / `.input.value_cache` | **原地写，dump 到的是写入前内容** |
| 写入落点 | 同算子的 `input.slot_indices` | 配合 cache 形状可反推 (block, offset) |
| attention 读到的 cache | `NPU.npu_fused_infer_attention_score.*.forward.input.key` / `.input.value` | CacheHit/Decode 下就是整张 cache，需按 block_table 解码 |
| PA 路径读到的 cache | `NPU._npu_paged_attention.*.forward.input.key_cache` / `.input.value_cache` | 同上 |
| mamba 状态 | `_C_ascend.npu_causal_conv1d_custom`、`_C_ascend.npu_recurrent_gated_delta_rule` 的 `input.conv_state` / `input.state` | GDN 侧 KV 等价物 |

**原地写的坑**：`_npu_reshape_and_cache` dump 出来的是调用时刻快照（= 写之前）。
想要"写后内容"只能：① 看下一次读算子（FIA/PA）的 input；② 或在 Python 层显式 `clone()`。

### 5.2 Python 层打点（更易解码）

在 `AscendAttentionBackendImpl.reshape_and_cache`（`attention_v1.py:1430`）`DeviceOperator.reshape_and_cache`
**之后**加 `self.key_cache.clone()` / `self.value_cache.clone()`，配合 `attn_metadata.slot_mapping` 落盘，
拿到的就是"写后"真实内容，且不用反解 block_table。
（本工作区早期 `[KVC]` 补丁的 `KVC_SAVE_KV=1` 走的就是这条路线，产物 schema `kvt5-hybrid-raw`：
attn 组 K/V + mamba 组 conv/ssm + 双组块表 meta。）
