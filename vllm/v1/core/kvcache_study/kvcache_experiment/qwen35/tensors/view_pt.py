#!/usr/bin/env python3
"""kv_*.pt 张量查看器(hybrid 版): 加载 .pt 文件并打印里面存了什么.

支持 schema:
  kvt5-hybrid-raw: {"K": [attn层][块->t], "V": 同构, "M": [mamba层][slot->[conv,ssm]],
                    "meta": {...attn/mamba 双组块表...}}
  kvt4-raw(kvc 兼容): {"K","V","meta"} (无 M 键, 纯 attn 单组)
  其他 .pt: 通用递归遍历(逐张量 shape/dtype/RMS)

用法(相对路径以本脚本所在 tensors/ 为基准):
  python3 view_pt.py                               # 无参数: 自动选第一个 req*/kv_*.pt
  python3 view_pt.py req1_bdd8c88d/kv_pp0tp0.pt    # 指定归档文件
  python3 view_pt.py /abs/path/xxx.pt              # 任意其他 .pt
"""
import glob
import os
import sys

import torch

BASE = os.path.dirname(os.path.abspath(__file__))


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # 老版本 torch 没有 weights_only 形参
        return torch.load(path, map_location="cpu")


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def view_hybrid(b, path):
    """qwen35 hybrid 归档格式: {"K","V","M"(可选),"meta"} —— attn 组 + mamba 组双节。"""
    m, K, V, M = b["meta"], b["K"], b["V"], b.get("M") or []
    schema = m.get("schema", "kvt4-raw")
    print(f"== {schema} KV 快照: {os.path.relpath(path, BASE)} ({os.path.getsize(path)} B) ==")
    print("== meta ==")
    for k in sorted(m):
        print(f"  {k}: {m[k]}")

    print("== 张量: attn 组 ==")
    bt = m.get("attn", {}).get("block_table", m.get("block_table", []))
    if not K:
        print("  (无 attn 层归档)")
    else:
        t0 = K[0][bt[0]] if bt and bt[0] in K[0] else next(iter(K[0].values())) \
            if K[0] else None
        if t0 is not None:
            print(f"K: {len(K)} 层; 块表 {bt}; 单块 shape={tuple(t0.shape)} dtype={t0.dtype}")
        print(f"V: {len(V)} 层; 块表与 K 逐层一致: "
              f"{all(list(a) == list(c) for a, c in zip(K, V))}")
        kb = sum(x.numel() * x.element_size() for lay in K for x in lay.values())
        vb = sum(x.numel() * x.element_size() for lay in V for x in lay.values())
        print(f"字节对账: 纯 attn K/V 张量 ≈ {kb + vb} B")
        bsz = m.get("block_size", 0)
        cov = m.get("cov") or []
        if bt and cov and 0 < cov[-1] < bsz and bt[-1] in K[0]:
            tk, tv = K[0][bt[-1]], V[0][bt[-1]]
            print(f"末块(block {bt[-1]}) cov={cov[-1]}/{bsz}: K已写区RMS={rms(tk[:cov[-1]]):.4f} "
                  f"未写区RMS={rms(tk[cov[-1]:]):.6f}(应≈0, 快照含未写槽位); "
                  f"V已写区RMS={rms(tv[:cov[-1]]):.4f}")

    print("== 张量: mamba 组 ==")
    if not M:
        print("  (无 M 数据: kvt4 纯 attn 归档或 mamba 组未命中)")
    else:
        sbt = m.get("mamba", {}).get("block_table", [])
        gts = m.get("mamba", {}).get("group_block_tables")  # 08 v1.2 meta
        if gts is None:  # v1.1 旧归档: 从 M 层键按组还原(M 迭代序=组优先, 每 6 层一组)
            gts = [sorted({s for li in range(g * 6, min((g + 1) * 6, len(M)))
                          for s in M[li].keys()}) for g in range((len(M) + 5) // 6)]
        at_bt = m.get("attn", {}).get("block_table") or m.get("block_table", [])
        gs = " ".join(f"mamba{chr(65 + g)}={t}" for g, t in enumerate(gts))
        print(f"M: {len(M)} 层({len(gts)} 个 mamba 组); ★4 组 block id 全景: "
              f"attn组={at_bt} {gs}")
        print(f"每层每 slot = 一整套状态张量(qwen3.5-0.8B: conv + ssm)")
        if gts:
            sbt = gts[0]  # 详查仍取首组表(与 v1.1 对齐)
        if sbt and sbt[0] in M[0]:
            for i, t in enumerate(M[0][sbt[0]]):
                nm = (m.get("mamba", {}).get("state_shapes") or ["?"] * len(M[0][sbt[0]]))[i]
                print(f"  slot={sbt[0]} state[{i}]({tuple(nm) if not isinstance(nm, tuple) else nm}): "
                      f"shape={tuple(t.shape)} dtype={t.dtype} RMS={rms(t):.4f} "
                      f"预览前4={t.reshape(-1)[:4].tolist()}")
        mb = sum(x.numel() * x.element_size() for lay in M for s in lay.values() for x in s)
        n_mamba_slots = len(list(next(iter(M), {}).keys())) if M else 0
        print(f"字节对账: mamba 状态张量 ≈ {mb} B (slot 数 x 状态张量 {n_mamba_slots} 套 x 层数)")
        print(f"两组共用同一物理 raw 页(块号命名空间独立) —— 组语义详见 meta.attn/meta.mamba")


def view_generic(obj, path):
    """非绘制归档: 递归遍历, 逐张量打印 shape/dtype/RMS."""
    print(f"== 通用 .pt 检视: {path} ==")

    def walk(o, prefix=""):
        if torch.is_tensor(o):
            r = rms(o) if o.numel() else None
            print(f"  {prefix}: {tuple(o.shape)} {o.dtype}" + (f" RMS={r:.4f}" if r is not None else ""))
        elif isinstance(o, dict):
            for k in o:
                walk(o[k], f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(o, (list, tuple)):
            for i, x in enumerate(o):
                walk(x, f"{prefix}[{i}]")
        else:
            print(f"  {prefix}: {type(o).__name__} = {o!r}")

    walk(obj)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isabs(path):
            path = os.path.join(BASE, path)
    else:
        cands = sorted(glob.glob(os.path.join(BASE, "req*", "kv_*.pt")))
        if not cands:
            sys.exit("用法: view_pt.py [req*/kv_*.pt 或任意 .pt 路径]")
        path = cands[0]
        print(f"(无参数, 自动选择: {os.path.relpath(path, BASE)})")
    if not os.path.isfile(path):
        sys.exit(f"文件不存在: {path}")
    b = load_pt(path)
    if isinstance(b, dict) and {"K", "V", "meta"} <= set(b):
        view_hybrid(b, path)
    else:
        view_generic(b, path)


if __name__ == "__main__":
    main()
