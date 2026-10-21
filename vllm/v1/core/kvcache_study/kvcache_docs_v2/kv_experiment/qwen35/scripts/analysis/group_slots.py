#!/usr/bin/env python3
"""group_slots.py — 归档 4 组 block id 分配全景（离线校正器）

08 v1.1 的 meta 只记录单一 mamba block_table（取自首个 mamba 组），丢掉另外两组分配；
本工具直接枚举归档 K[层]/M[层] 的 dict 键（= 持有的块/slot id），按组还原真实分配。

块表结构（kvt5-hybrid-raw）:
  K: [attn 层][管理块号 -> tensor]    6 层, 键 = attn 组实际持块
  M: [mamba 层][slot id -> states]   18 层, 键 = 该层实际持 slot
M 层迭代序 = 组优先: [0..5]=mamba 组A(layers.0/4/8/12/16/20, interleaved 0::3)
             [6..11]=组B(layers.1/5/9/13/17/21)  [12..17]=组C(layers.2/6/10/14/18/22)

用法（容器内相对 qwen35 根）:
    python3 tensors/group_slots.py tensors/abnormal/req1_abdc0ba8/kv_pp0tp0.pt \
        tensors/prefix2048/req1_9cdd21bf/kv_pp0tp0.pt
"""
import sys

import torch

for path in sys.argv[1:]:
    b = torch.load(path, map_location="cpu", weights_only=False)
    K, M = b["K"], b["M"]
    meta = b["meta"]
    mb = meta.get("mamba", {})
    print(f"== {path}")
    print(f"  meta.attn.block_table = {meta.get('attn', {}).get('block_table')}"
          f"  meta.mamba.block_table = {mb.get('block_table')}   <- v1.1 单一表(仅首组)")
    at_keys = sorted({k for d in K for k in d.keys()})
    print(f"  K 全层持块并集 = {at_keys}")
    row = " ".join(f"K[{li}]={sorted(d.keys())}" for li, d in enumerate(K))
    print(f"  {row}")
    for li, d in enumerate(M):
        print(f"  M[{li:02d}] keys={sorted(d.keys())}")
    gs = {g: sorted({s for li in range(g * 6, g * 6 + 6) for s in M[li].keys()})
          for g in range(3)}
    print(f"  ★ 组级还原: attn组={at_keys}  mambaA={gs[0]}  mambaB={gs[1]}  mambaC={gs[2]}")
    print()
