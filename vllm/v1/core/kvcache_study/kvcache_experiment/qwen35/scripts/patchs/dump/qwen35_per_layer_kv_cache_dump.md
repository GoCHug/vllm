# Qwen3.5 逐层 KVCache / Slot 落盘方案

> 配套补丁：[qwen35_per_layer_kv_cache_dump.patch](./qwen35_per_layer_kv_cache_dump.patch)
> 目标文件：`vllm/vllm/model_executor/models/qwen3_next.py`

## 1. 目标

在一次前向（= 一个 scheduler step）里，把每一层刚刚写完的 KV 状态取出来，落盘保存，用于：

- 逐层比对 vllm 与 vllm-ascend 的 KVCache 内容（对齐 msprobe 抓到的算子入参）；
- 观察混合模型（full attention 层 + GDN 层）在同一 step 内的状态推进；
- 排查 dtype 域穿越、padding、块复用等精度/布局问题。

## 2. 结论

**可以**放在 `Qwen3NextModel.forward` 的 DecoderLayer 循环里，但有三个前提：

1. **必须是 eager 模式**（`--enforce-eager` 或 `cudagraph_mode=NONE`）。
   `Qwen3NextModel` 被 `@support_torch_compile` 装饰；`CompilationMode.NONE` 时该装饰器直接 `return`，`forward` 就是普通 Python 函数，循环每 step 执行一次。开了 ACL Graph 后 FULL 模式下整个 forward 只在 capture 时跑一次，循环里的落盘代码不会按 step 执行。
2. **必须 clone / index_select**。`kv_cache` 是跨 step 复用的持久显存，下个 step 会原地覆写。
3. **层名要反查**。`Attention` 实例和 GDN 实例都不保存 `prefix`，只能通过 `get_forward_context().no_compile_layers`（层名 → 层实例）做 id 反查。

## 3. 为什么不放在 Attention / GDN 内部

- 放在 `attention.py` 或 `attention_v1.py` 的 `forward` 里：同样受编译/图捕获影响，且拿不到「这一层整体算完」的视角，还要改上游 vllm 或 vllm-ascend 两个仓库。
- 放在 model runner 的 `execute_model` 之后：能覆盖所有层，但拿不到「逐层刚算完」的时序（不过同 step 内没有相互覆盖，数据其实等价）。
- 放在模型结构的循环里：改动集中在一个文件，层对象、`self_attn` / `linear_attn`、层名都能就地拿到。

## 4. Attention 层与 GDN 层的差异

| 项目 | Attention 层 | GDN 层（linear attention） |
| --- | --- | --- |
| 层对象路径 | `layer.self_attn.attn` | `layer.linear_attn` |
| 层名前缀 | `model.layers.{i}.self_attn.attn` | `model.layers.{i}.linear_attn` |
| `kv_cache` 类型 | `tuple (k_cache, v_cache)` | `list [conv_state, ssm_state]` |
| 张量形状 | `(num_blocks, block_size, H, D)` | `(num_blocks, *state_shape)` |
| 索引来源 | `forward_context.slot_mapping[层名]` | `forward_context.attn_metadata[层名].*_state_indices_tensor` |
| 索引语义 | 扁平 token 槽 = `block_id * block_size + offset` | **块号**（每请求一个），不是 token 槽 |
| 选取方式 | `t.flatten(0, 1).index_select(0, slots)` | `t.index_select(0, ids)` |
| 无效值 | `-1`（PAD_SLOT_ID，图 padding / CP 非本 rank） | `NULL_BLOCK_ID = 0`（mamba 元数据 padding 行） |

### 为什么 GDN 不能用 slot_mapping

`slot_mapping` 是按 `kv_cache_groups` 的每个 `layer_name` 都填的（mamba 组也有），但对 mamba 组它等于 `block_id * mamba_block_size + offset`：

- `mamba_block_size` 在 `none` 模式下是 `max_model_len`（如 32768），`align` 模式下是 `block_size`（如 1024）；
- 而 `conv_state` / `ssm_state` 的第 0 维只有 `num_blocks`。

直接 `index_select(0, slot_mapping)` 必然越界。GDN 真正的索引是 `GDNAttentionMetadata` 里的 `spec_state_indices_tensor` / `non_spec_state_indices_tensor`，其源头是 `block_table_tensor[:, 0]`，与 GDN 算子入参 `state_indices` 同源。

## 5. 实现要点

```
get_forward_context()
  ├─ no_compile_layers : dict[层名 -> 层实例]        # 反查层名
  ├─ slot_mapping      : dict[层名 -> Tensor]        # 仅 Attention 层可用
  └─ attn_metadata     : dict[层名 -> AttentionMetadata]
        └─ GDN 层 -> GDNAttentionMetadata.state_indices
```

1. **反查层名**：`{id(module): name for name, module in no_compile_layers.items()}`，每 step 建一次（几十条，开销可忽略）。
2. **跳过 profile / warmup**：此时 `attn_metadata is None`，直接不落盘。
3. **过滤 padding**：Attention 侧 `slots >= 0`；GDN 侧按真实请求数截断或容忍 `NULL_BLOCK_ID`。
4. **维度**：Attention 必须先 `flatten(0, 1)` 让 dim0 变成扁平槽号，再 `index_select`。
5. **拷贝语义**：`index_select` 本身返回新张量；`slots` / `ids` 来自本 step 复用 buffer，需 `clone()`。

## 6. 落盘格式

```
<OUT_DIR>/
  step0/
    layer00_attn.pt
    layer00_gdn.pt
    layer01_gdn.pt
    ...
  step1/
    ...
```

Attention 层 payload：

```python
{
    "kind": "attn",
    "layer": "model.layers.0.self_attn.attn",
    "slots": Tensor[num_tokens],        # 本 step 写入的扁平槽号（已过滤 -1）
    "k_cache": Tensor[num_tokens, H, D],
    "v_cache": Tensor[num_tokens, H, D],
}
```

GDN 层 payload：

```python
{
    "kind": "gdn",
    "layer": "model.layers.0.linear_attn",
    "state_indices": Tensor[num_reqs],   # 块号（投机解码时已 flatten）
    "conv_state": Tensor[num_reqs, ...],
    "ssm_state": Tensor[num_reqs, ...],
}
```

## 7. 使用方式

```bash
export VLLM_QWEN35_KV_DUMP_DIR=/tmp/qwen35_kv
export VLLM_QWEN35_KV_DUMP_STEPS=2      # 0 = 不限步数

vllm serve <model> --enforce-eager ...
```

不打这两个环境变量时，`_kv_dump_begin()` 直接返回 `None`，循环里只多一次判空，无副作用。

## 8. 已知限制

- 只在 eager 下逐 step 生效；ACL Graph 下是 no-op。
- PP > 1 时每个 rank 只持有 `[start_layer, end_layer)` 的层，只能落本 rank 的层。
- 开了投机解码（MTP / eagle）时，一个 scheduler step 里会有 draft model 的额外前向，step 计数会比 scheduler step 多。
- `index_select` 只取「本 step 写入的部分」。若要还原「整条序列的完整 KV」，需要结合 block table 按请求拼块。
- 单步全层落盘数据量大，建议配合 `VLLM_QWEN35_KV_DUMP_STEPS` 限制步数。

## 9. 与 msprobe 的分工

| | msprobe | 本补丁 |
| --- | --- | --- |
| 抓取点 | 算子级（`npu_reshape_and_cache` / `npu_recurrent_gated_delta_rule` 等） | 层级（每层 forward 结束后） |
| 数据内容 | 算子入参 / 出参的原始张量 | 按 slot / state_index 归位后的 KV 状态 |
| 覆盖范围 | 依赖配置，易漏层 | 天然覆盖全部层 |
| 额外依赖 | 需要 msprobe + aclgraph_dump | 无，纯 Python + 环境变量 |

两者可以交叉验证：msprobe 的 `_npu_reshape_and_cache.*.kwargs.slot_indices` 应与本补丁的 `slots` 一致；`npu_recurrent_gated_delta_rule.*.kwargs.ssm_state_indices` 应与 `state_indices` 一致。

## 10. 验证方法

```python
import torch

a = torch.load("/tmp/qwen35_kv/step0/layer00_attn.pt")
print(a["kind"], a["layer"], a["slots"].shape, a["k_cache"].shape)

g = torch.load("/tmp/qwen35_kv/step0/layer00_gdn.pt")
print(g["kind"], g["layer"], g["state_indices"], g["ssm_state"].shape)
```

校验点：

1. `slots` 无 `-1`，且每个值 `< num_blocks * block_size`；
2. `k_cache` dim0 == `slots.numel()`；
3. GDN 的 `state_indices` 均 `< num_blocks`；
4. 与 msprobe 同名算子入参逐元素比对。
