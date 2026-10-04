#!/usr/bin/env python3
"""kv_*.pt 张量查看器: 加载 .pt 文件并打印里面存了什么.

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


def view_kvt4(b, path):
    """kvc 归档格式: {"K": [层 -> {块号: 张量}], "V": [...], "meta": {...}}"""
    m, K, V = b["meta"], b["K"], b["V"]
    print(f"== kvt4-raw KV 快照: {os.path.relpath(path, BASE)} ({os.path.getsize(path)} B) ==")
    print("== meta ==")
    for k in sorted(m):
        print(f"  {k}: {m[k]}")
    print("== 张量 ==")
    blocks = m["block_table"]
    t = K[0][blocks[0]]
    print(f"K: {len(K)} 层; 每层块表 {blocks}; 单块 shape={tuple(t.shape)} dtype={t.dtype}")
    print(f"V: {len(V)} 层; 块表与 K 逐层一致: {all(list(a) == list(c) for a, c in zip(K, V))}")
    kv_heads, head_dim, bsz = t.shape[1], t.shape[2], t.shape[0]
    kb = sum(x.numel() * x.element_size() for lay in K for x in lay.values())
    print(f"字节对账: 纯 K/V 张量 ≈ {2 * kb} B vs 文件 {os.path.getsize(path)} B (差=zip容器+meta开销)")
    cov = m.get("cov") or []
    if cov:
        last, w = blocks[-1], cov[-1]
        tk, tv = K[0][last], V[0][last]
        if 0 < w < bsz:
            print(f"末块(block {last}) cov={w}/{bsz}: K已写区RMS={rms(tk[:w]):.4f} "
                  f"未写区RMS={rms(tk[w:]):.6f}(应≈0, 快照含未写槽位); V已写区RMS={rms(tv[:w]):.4f}")


def view_generic(obj, path):
    """非 kvc 归档: 递归遍历, 逐张量打印 shape/dtype/RMS."""
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
        view_kvt4(b, path)
    else:
        view_generic(b, path)


if __name__ == "__main__":
    main()
