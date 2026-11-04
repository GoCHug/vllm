"""pool_map — hybrid 模型 pool ↔ layer 映射（4-group 共享模型）。

vLLM 对 hybrid KVCache 的切分（kv_cache_utils.py:1146-1190）：
    1. 按 kv_cache_spec 的 type 分组            → attn 组、mamba 组
    2. 每个 type 组再"交错"切成等长小组（不足补 padding）
    3. group_size = 每个小组的层数；共 group_size 个共享 pool
       每个 pool 由各 group 的同序号层共享（4 个 group 各 1 层）

Qwen3.5-0.8B：24 层 = 6 × [L,L,L,A]
    attn 6 层  → 1 个 group（6 层）
    mamba 18 层 → 交错切成 3 个 group（每组 6 层）
    合计 4 个 group × 6 层 → 6 个 pool
    结论：pool k ← layer[4k .. 4k+3]，即 attn 的 .k 与 GDN 的 .3k/.3k+1/.3k+2
"""
from math import ceil

# Qwen3.5-0.8B 的 layer_types（full_attention_interval=4，24 层）
QWEN35_LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"] + \
                     ["linear_attention"] * 3 + ["full_attention"] + \
                     ["linear_attention"] * 3 + ["full_attention"] + \
                     ["linear_attention"] * 3 + ["full_attention"] + \
                     ["linear_attention"] * 3 + ["full_attention"] + \
                     ["linear_attention"] * 3 + ["full_attention"]


def build(layer_types=None, attn_groups=1):
    """返回 pool 列表：[{pool, layers, attn_call, gdn_calls}]。

    attn_groups: attn 侧被切成的 group 数（Qwen3.5 为 1）。
    """
    types = list(layer_types or QWEN35_LAYER_TYPES)
    attn_layers = [i for i, t in enumerate(types) if t == "full_attention"]
    mamba_layers = [i for i, t in enumerate(types) if t != "full_attention"]

    # group_size = 单个 group 的层数。attn 侧按 attn_groups 切，mamba 侧对齐成同样层数。
    group_size = ceil(len(attn_layers) / attn_groups)
    mamba_groups = ceil(len(mamba_layers) / group_size) if mamba_layers else 0

    pools = []
    for k in range(group_size):
        layers = []
        attn_call = None
        gdn_calls = []
        # attn: 第 k 个 group 的第 k 层（attn_groups=1 时即第 k 层）
        if k < len(attn_layers):
            attn_call = k
            layers.append(attn_layers[k])
        # mamba: 交错分组后，每个 group 的第 k 层 → 原序号 k*group 数 + g
        for g in range(mamba_groups):
            pos = k * mamba_groups + g
            if pos < len(mamba_layers):
                gdn_calls.append(pos)
                layers.append(mamba_layers[pos])
        pools.append({
            "pool": k,
            "layers": sorted(layers),
            "attn_call": attn_call,
            "gdn_calls": gdn_calls,
        })
    return pools


def attn_call_to_pool(call_index):
    """dump 里 NPU._npu_reshape_and_cache.{i} 的 i → pool 号。"""
    return call_index


def gdn_call_to_pool(call_index, pools=None):
    """dump 里 _C_ascend.npu_*_custom.{j} 的 j → pool 号（每 3 个 GDN 层一个 pool）。"""
    for p in pools or build():
        if call_index in p["gdn_calls"]:
            return p["pool"]
    raise KeyError(f"GDN 调用序号越界: {call_index}")


def validate(pools=None, n_layers=None):
    """断言：pool 数 × 每 pool 层数 = 全部层，且无重复/遗漏。"""
    pools = pools or build()
    total = sum(len(p["layers"]) for p in pools)
    flat = [l for p in pools for l in p["layers"]]
    assert len(set(flat)) == len(flat), f"层重复: {sorted(flat)}"
    assert total == len(n_layers) if n_layers else total == len(QWEN35_LAYER_TYPES), \
        f"层数不匹配: 映射覆盖 {total} 层"
    assert sorted(flat) == list(range(total)), f"层不连续: {sorted(flat)}"
    return True


if __name__ == "__main__":
    ps = build()
    validate(ps)
    for p in ps:
        print(f"pool{p['pool']}: layers={p['layers']} "
              f"attn_call={p['attn_call']} gdn_calls={p['gdn_calls']}")
    print(f"共 {len(ps)} 个 pool，覆盖 {sum(len(p['layers']) for p in ps)} 层")