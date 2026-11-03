# Qwen3.5 Attn 读写 KVCache 调用链

Qwen3.5-0.8B 在 vLLM 0.23.0 + vllm-ascend 0.23.0 上，**full-attention 层**从 `self.attn(q, k, v)` 到 NPU 算子之间，
KVCache 的写入与读取分别发生在哪个函数、哪个算子。

> 一句话结论：模型层传进 `self.attn` 的 `q/k/v` **只是本步新算出来的当前 token**，不含任何历史 KV；
> 历史 KV 的**写**在 `AscendAttentionBackendImpl.reshape_and_cache`，**读**在 `forward_fused_infer_attention` /
> `forward_paged_attention` 里，并且读的是**整张 cache 张量 + block_table 分页寻址**，不是拼接好的连续 KV。

源码位置约定：
`vllm/` = `/vllm-workspace/vllm`，`vllm-ascend/` = `/vllm-workspace/vllm-ascend`。

---

## 1. 总调用树

```
Qwen3NextDecoderLayer.forward                                  vllm/model_executor/models/qwen3_next.py:324
│   layer_type == "linear_attention" → QwenGatedDeltaNetAttention（见 §5，另一条分支）
└── Qwen3NextAttention.forward(positions, output, hidden_states)          qwen3_next.py:286
    ├── self.qkv_proj(hidden_states) → qkv                                :292
    ├── qkv.split(...) → q / k / v（当前 token）                          :304
    │     attn_output_gate 时：q_gate → chunk → q, gate                    :294-302
    ├── self.q_norm(q) / self.k_norm(k)                                   :306-311
    ├── self.rotary_emb(positions, q, k) → q, k（加 RoPE）                 :313
    ├── self.attn(q, k, v)  ────────────────────────┐                    :315
    │                                               │
    │   Attention.forward(query, key, value)        ▼    vllm/model_executor/layers/attention/attention.py:437
    │   ├── view 成 3D: query/key/value = .view(-1, heads, head_size)
    │   ├── unified_kv_cache_update(key, value, layer_name)               :498 / :518
    │   │     └── def unified_kv_cache_update                             :691
    │   │         └──（Ascend 上写 cache 走的是 impl 内部，见下方 ⚠）
    │   └── unified_attention_with_output(query, key, value, output, ...) :501 / :521
    │         └── def unified_attention_with_output                        :734
    │             ├── get_attention_context(layer_name) → attn_metadata, self, kv_cache   :749
    │             └── self.impl.forward(self, q, k, v, kv_cache, attn_metadata, output)   :751
    │                   │
    │                   ▼  AscendAttentionBackendImpl.forward   vllm-ascend/vllm_ascend/attention/attention_v1.py:1479
    │                 ├── self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]     :1527
    │                 │
    │                 ├──【写 KVCache】self.reshape_and_cache(q, k, v, kv_cache, meta, out) :1532
    │                 │   └── AscendAttentionBackendImpl.reshape_and_cache                :1430
    │                 │       ├── slots = attn_metadata.slot_mapping
    │                 │       ├── DeviceOperator.reshape_and_cache(key, value,
    │                 │       │        key_cache=self.key_cache, value_cache=self.value_cache,
    │                 │       │        slot_mapping=slots)                                :1444
    │                 │       │   └── BaseDeviceAdaptor.reshape_and_cache   vllm-ascend/vllm_ascend/device/device_op.py:49
    │                 │       │       └── torch_npu._npu_reshape_and_cache(...)  ← ★ 写算子(A2/A3)
    │                 │       │           （A5 走 A5DeviceAdaptor，device_op.py:1119
    │                 │       │             → torch_npu.npu_scatter_pa_kv_cache）
    │                 │       └── notify_kv_cache_written()                              :1453
    │                 │
    │                 ├──【读 KVCache】forward_impl(q, k, v, kv_cache, meta, out)          :1456
    │                 │   │
    │                 │   ├── 条件：DecodeOnly && sliding_window is None && using_paged_attention
    │                 │   │   └── forward_paged_attention                                 :1363
    │                 │   │       └── torch_npu._npu_paged_attention(
    │                 │   │               query, key_cache=self.key_cache,
    │                 │   │               value_cache=self.value_cache,
    │                 │   │               block_table=attn_metadata.block_tables,
    │                 │   │               context_lens=attn_metadata.seq_lens, out=output)  ← ★ 读算子(PA)
    │                 │   │
    │                 │   └── 否则 forward_fused_infer_attention                          :1234
    │                 │       ├── _get_fia_params(key, value, meta, kv_cache)             :1174
    │                 │       │     PrefillNoCache : block_table=None, key/value = 当前 token
    │                 │       │     PrefillCacheHit: key = self.key_cache.view(num_block, block_size, -1)  :1204
    │                 │       │     DecodeOnly    : key = self.key_cache.view(num_block, block_size, -1)  :1213
    │                 │       └── DeviceOperator.npu_fused_infer_attention_score(...)     device_op.py:55
    │                 │           └── torch_npu.npu_fused_infer_attention_score(
    │                 │                   query, key, value, ...)                        :92  ← ★ 读算子(FIA)
    │                 │
    │                 └── output[:num_tokens] = attn_output[:num_tokens]; return output   :1546-1547
    │
    ├── gate = torch.sigmoid(gate); attn_output *= gate（attn_output_gate 时） qwen3_next.py:317-319
    └── output[:], _ = self.o_proj(attn_output)                                 :321
```

> ⚠ `unified_kv_cache_update`（vllm 侧，:691）是通用抽象。Ascend 后端在
> `AscendAttentionBackendImpl.forward` 内部自己做写（:1532），走的是 `reshape_and_cache`，
> 而不是这个 op 的默认实现；调试时以 `_npu_reshape_and_cache` 为准。

---

## 2. 写路径要点

| 项 | 内容 |
|---|---|
| 写入位置 | `AscendAttentionBackendImpl.forward` → `reshape_and_cache`（`attention_v1.py:1430`） |
| 写什么 | 只有**当前 token** 的 K/V（`key[:num_actual_tokens]`） |
| 写到哪里 | `key_cache` / `value_cache`，形状 `[num_blocks, block_size, num_kv_heads, head_size]`（`kv_cache` 整体 `[2, ...]`） |
| 定位方式 | `slot_mapping`（每个 token 一个 slot 下标），由 scheduler 的 block 分配决定 |
| 底层算子 | A2/A3：`torch_npu._npu_reshape_and_cache`；A5：`torch_npu.npu_scatter_pa_kv_cache`（`cache_mode="Norm"`） |
| 特性 | **原地写**（in-place），Python 层拿不到"写后副本"，只能看到同一块张量被改写 |

---

## 3. 读路径要点

| 项 | FIA（`forward_fused_infer_attention`） | PA（`forward_paged_attention`） |
|---|---|---|
| 触发条件 | 默认 / prefill / cache-hit | `DecodeOnly` + 无 sliding window + `using_paged_attention()` |
| 传入的 key/value | `_get_fia_params` 的返回值 | 不走 `key/value`，直接传 `key_cache`/`value_cache` |
| cache 形态 | CacheHit/Decode 时 `self.key_cache.view(num_block, block_size, -1)` | 原样整张 cache |
| 寻址 | `block_table` + `actual_seq_lengths_kv` | `block_table=attn_metadata.block_tables` + `context_lens=attn_metadata.seq_lens` |
| 底层算子 | `torch_npu.npu_fused_infer_attention_score` | `torch_npu._npu_paged_attention` |

两种方式都**不做 KV 拼接**：整张 cache 交给算子，算子按 block_table 索引，有效长度由 seq_lens 决定。

---

## 4. 三种 `attn_state` 下 K/V 的来源

| attn_state | 写入 cache | 传给 attention 的 K/V | block_table |
|---|---|---|---|
| `PrefillNoCache`（首次 prefill，cache 还没内容） | 有 | **当前 token 的 K/V**（不读 cache） | `None` |
| `PrefillCacheHit`（前缀命中） | 有 | **整张 key_cache/value_cache 的 view** | `attn_metadata.block_tables` |
| `DecodeOnly`（逐 token 解码） | 有 | **整张 key_cache/value_cache 的 view** | `attn_metadata.block_tables` |

对应源码：`_get_fia_params`（`attention_v1.py:1174`），分支在 `:1194` / `:1200` / `:1211`。

---

## 5. 并排分支：GDN 层（`layer_type == "linear_attention"`）

Qwen3.5 的 24 层里 18 层是 GDN，它们不写 attn 的 K/V，而是写 **conv_state + ssm_state**，
但两者**共享同一套 block/页几何**（这正是 block-size 问题里 `K 段宽 ≡ SSM 段宽` 不变量的由来）。

```
Qwen3NextDecoderLayer.forward                     qwen3_next.py:324
└── self.linear_attn = QwenGatedDeltaNetAttention                    :342
    └── forward(hidden_states, output)            vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:844
        └── self._forward_method(...)             :849   （forward_native / forward_cuda / ...，:307-311 选择）
            └── _forward_core(...)                :1260
                ├── self_kv_cache = self.kv_cache               :1308
                ├── conv_state = self_kv_cache[0]               :1311（是否 transpose 由 is_conv_state_dim_first 决定）
                ├── ssm_state  = self_kv_cache[1]               :1316
                └── 传 conv_state / ssm_state 进入 GDN 计算，状态原地更新   :1346 起
```

---

## 6. KVCache 张量形态与寻址

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

## 7. dump / 观测挂点

### 7.1 msprobe 算子级 dump（已在 `1024_dump` / `2048_dump` 中验证可用）

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

### 7.2 Python 层打点（更易解码）

在 `AscendAttentionBackendImpl.reshape_and_cache`（`attention_v1.py:1430`）`DeviceOperator.reshape_and_cache`
**之后**加 `self.key_cache.clone()` / `self.value_cache.clone()`，配合 `attn_metadata.slot_mapping` 落盘，
拿到的就是"写后"真实内容，且不用反解 block_table。
（本工作区早期 `[KVC]` 补丁的 `KVC_SAVE_KV=1` 走的就是这条路线，产物 schema `kvt5-hybrid-raw`：
attn 组 K/V + mamba 组 conv/ssm + 双组块表 meta。）

---

## 8. 源码索引

| 环节 | 文件:行 |
|---|---|
| 模型层 attention forward | `vllm/model_executor/models/qwen3_next.py:286` |
| `self.attn(q,k,v)` | `qwen3_next.py:315`（`Attention` 实例定义于 `:267`） |
| DecoderLayer 分支（attn / GDN） | `qwen3_next.py:324` / `:342` / `:349` |
| vLLM `Attention.forward` | `vllm/model_executor/layers/attention/attention.py:437` |
| `unified_kv_cache_update` | `attention.py:691`（调用点 `:498` / `:518`） |
| `unified_attention_with_output` | `attention.py:734`（调用点 `:501` / `:521`） |
| impl 分发 | `attention.py:751` → `self.impl.forward(...)` |
| Ascend impl forward | `vllm-ascend/vllm_ascend/attention/attention_v1.py:1479` |
| 写 cache 调用点 | `attention_v1.py:1532` → `:1430` → `:1444` |
| 写算子（A2/A3） | `vllm-ascend/vllm_ascend/device/device_op.py:49` → `torch_npu._npu_reshape_and_cache` |
| 写算子（A5） | `device_op.py:1119` → `torch_npu.npu_scatter_pa_kv_cache` |
| FIA 参数选择 | `attention_v1.py:1174`（`:1194` / `:1200` / `:1211`） |
| FIA 读路径 | `attention_v1.py:1234` → `device_op.py:55` → `:92` |
| PA 读路径 | `attention_v1.py:1363` → `torch_npu._npu_paged_attention` |
| forward_impl 分派 | `attention_v1.py:1456`（`:1468` 判断 PA / FIA） |
| GDN 层 forward | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:844` → `:1260` |
| GDN 状态读写 | 同文件 `:1308`（`kv_cache`）、`:1311`（conv_state）、`:1316`（ssm_state） |