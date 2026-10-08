#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# inspect_prefix.py —— kvc prefix cache 前缀复用关系检查器(v2.5 归档, pairwise)
#
# 输入: --dir tensors  (与 inspect_kv_tensors.py 同输入: 一请求一子目录布局)
#
# 逻辑:
#   1. 枚举全部请求归档, 读 meta(seq/request_id/block_table/cov/...)
#   2. 按 seq 排序, 两两判定 "前缀复用关系":
#      —— 晚请求 block_table 与早请求 block_table 的公共头部块 k 个
#         (早请求种块, 晚请求命中 -> prefix cache 复用; k=0 视为无关系跳过)
#
# 输出: logs/analysis/inspect_prefix.out（报告式, 逐对一节 + 汇总）
# 用法:
#   python3 scripts/analysis/inspect_prefix.py --dir tensors   # -> logs/analysis/inspect_prefix.out
#   python3 scripts/analysis/inspect_prefix.py --dir tensors --out xx.out
# ==============================================================================
import argparse
import sys
import time
from pathlib import Path

import torch

W = 76   # 报告横线宽度


def load_bundle(path):
    b = torch.load(str(path), map_location="cpu", weights_only=False)
    for k in ("K", "V", "meta"):
        if k not in b:
            raise ValueError(f"非 kvc 归档结构(缺 {k}): {path}")
    return b


class Req:
    """一个请求的全部 worker 归档 + meta。"""

    def __init__(self, d: Path):
        self.dir = d
        self.files = sorted(d.glob("kv_pp*.pt"))
        if not self.files:
            raise ValueError(f"{d} 下无 kv_pp*.pt")
        m = load_bundle(self.files[0])["meta"]
        self.meta = m
        self.seq = m["seq"]
        self.rid = str(m.get("request_id", d.name))
        self.round = m.get("p_tok", 0)      # 归档时的 region/kv 表长
        self.table = list(m.get("block_table", []))
        self.cov = list(m.get("cov", []))
        self.bs = m.get("block_size", 128)
        self.layers = m.get("layers", len(m.get("layer_ids", [])))


def shared_head(a, b):
    """早请求 a 与晚请求 b 的公共表头块数 k(b.table 前 k 项 == a.table 前 k 项)。"""
    k = 0
    for x, y in zip(a.table, b.table):
        if int(x) != int(y):
            break
        k += 1
    return k


def report(args) -> int:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"[ERROR] 归档目录不存在: {root}")
        return 2
    req_dirs = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("req")],
                      key=lambda d: d.name)
    try:
        reqs = [Req(d) for d in req_dirs]
    except Exception as e:
        print(f"[ERROR] 归档加载失败: {e}")
        return 2
    if len(reqs) < 1:
        print(f"[ERROR] {root} 下无请求子目录(req*)")
        return 2
    reqs.sort(key=lambda r: r.seq)

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    def say(s=""):
        lines.append(s)

    say("=" * W)
    say("kvc prefix cache 前缀复用关系检查(08 v2.5 归档 · pairwise)")
    say(f"归档目录: {root} | 请求: {len(reqs)} 个 | 生成: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * W)
    for r in reqs:
        say(f"  {r.dir.name}/  seq={r.seq}  块表={r.table}  cov={r.cov}")
    say()

    # ---------- 两两判定前缀复用关系 ----------
    pairs = []
    for i in range(len(reqs)):
        for j in range(i + 1, len(reqs)):
            k = shared_head(reqs[i], reqs[j])
            if k > 0:
                pairs.append((reqs[i], reqs[j], k))

    if not pairs:
        say("未发现前缀复用关系(任意两请求无公共表头块) —— 无可检查对。")
    for idx, (a, b, k) in enumerate(pairs, 1):
        hit = k * a.bs
        shared = a.table[:k]
        say("=" * W)
        say(f"[{idx}/{len(pairs)}] 前缀复用: {a.dir.name}(seq={a.seq}) --种块--> "
            f"{b.dir.name}(seq={b.seq}) 命中共享表头块")
        say("=" * W)
        say(f"  共享表头块: {shared} (k={k}, 命中 tokens ≈ {hit})")
        say(f"  早请求: {a.rid}  块表={a.table}  cov={a.cov}")
        say(f"  晚请求: {b.rid}  块表={b.table}  cov={b.cov}")
        say()

    # ---------- 汇总 ----------
    say("=" * W)
    if not pairs:
        say("[DONE] 无前缀复用关系对; 归档请求如上。")
    else:
        say(f"[DONE] 前缀复用关系对 {len(pairs)} 个。")
    say("=" * W)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[DONE] 报告已落盘: {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="kvc prefix cache 前缀复用关系检查器(v2.5 pairwise)")
    ap.add_argument("--dir", default="tensors",
                    help="tensors 归档根目录(默认 tensors, 含 req*/ 子目录)")
    ap.add_argument("--out", default="logs/analysis/inspect_prefix.out",
                    help="输出 out 产物路径(默认 logs/analysis/inspect_prefix.out)")
    args = ap.parse_args()
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
