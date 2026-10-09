#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# inspect_prefix_p.py —— kvc_1p1d P 侧前缀复用关系检查器(pairwise)
# P 侧语义: 早请求种块, 晚请求(prefix 命中)复用; A 重算一致性 =
# 同 token 段两次独立 prefill 计算(均在本 P 实例)的 ULP 级对比。
#
# 输入: --dir tensors  (08 号 PD 归档版, 根目录下 req{seq}/{P,D}/ 一请求一目录布局;
#       本检查器只扫请求下的 P/ 子目录)
#
# 逻辑:
#   1. 枚举全部请求归档, 读 meta(seq/request_id/block_table/cov/...)
#   2. 按 seq 排序, 两两判定 "前缀复用关系":
#      —— 晚请求 block_table 与早请求 block_table 的公共头部块 k 个
#         (早请求种块, 晚请求命中 -> prefix cache 复用; k=0 视为无关系跳过)
#   3. 对每一对(早 -> 晚)做重算一致性检查:
#      A) 重算一致性(ULP): 若早请求第 k 块为部分块(尾块)且晚请求第 k 块为
#         对应重算新块, 比较同 token 段 [0:cov早] 的位翻转与 Pearson
#         —— 分层统计(L0 与末层): L0 仅 ULP 级, 深层呈残差流放大
#
# 输出: logs/analysis/inspect_prefix.out（报告式, 逐对一节 + 汇总）
# 用法:
#   python3 scripts/analysis/inspect_prefix_p.py --dir tensors   # -> logs/analysis/inspect_prefix_p.out
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


def sig16(t):
    return t.view(torch.int16)


def pearson(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()


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
        self.rid = str(m.get("request_id", d.parent.name))
        self.round = m.get("p_tok", 0)      # 归档时的 region/kv 表长
        self.table = list(m.get("block_table", []))
        self.cov = list(m.get("cov", []))
        self.bs = m.get("block_size", 128)
        self.layers = m.get("layers", len(m.get("layer_ids", [])))
        self._cache = {}

    def b(self, f):
        if f not in self._cache:
            self._cache[f] = load_bundle(f)
        return self._cache[f]

    def block(self, f, kv, layer, blk):
        return self.b(f)[kv][layer].get(int(blk))


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
    req_dirs = sorted([d / "P" for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("req")],
                      key=lambda d: d.name)
    try:
        reqs = [Req(d) for d in req_dirs]
    except Exception as e:
        print(f"[ERROR] 归档加载失败: {e}")
        return 2
    if len(reqs) < 1:
        print(f"[ERROR] {root} 下无请求(req*/P)")
        return 2
    reqs.sort(key=lambda r: r.seq)

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    def say(s=""):
        lines.append(s)

    say("=" * W)
    say("kvc_1p1d P 侧前缀复用关系检查(08 PD 归档 · pairwise)")
    say(f"归档目录: {root} | 请求: {len(reqs)} 个 | 生成: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * W)
    for r in reqs:
        say(f"  {r.dir.parent.name}/{r.dir.name}/  seq={r.seq}  块表={r.table}  cov={r.cov}")
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
        say(f"[{idx}/{len(pairs)}] 前缀复用: {a.dir.parent.name}/{a.dir.name}(seq={a.seq}) --种块--> "
            f"{b.dir.parent.name}/{b.dir.name}(seq={b.seq}) 命中共享表头块")
        say("=" * W)
        say(f"  共享表头块: {shared} (k={k}, 命中 tokens ≈ {hit})")
        say(f"  早请求: {a.rid}  块表={a.table}  cov={a.cov}")
        say(f"  晚请求: {b.rid}  块表={b.table}  cov={b.cov}")

        # ---- A: 重算一致性(ULP, 同 token 段) ----
        # 早请求第 k 块(部分块) vs 晚请求第 k 块(重算新块), 比较 [:cov_k]
        if k < len(a.table) and k < len(b.table):
            ca = a.cov[k] if k < len(a.cov) else a.bs
            blk_a, blk_b = a.table[k], b.table[k]
            if ca < a.bs and int(blk_a) != int(blk_b):
                say(f"  A) 重算一致性(同 token 段前 {ca} 槽): 早 req b{blk_a} vs 晚 req b{blk_b}:")
                for lay in (0, a.layers - 1):
                    for kv in ("K", "V"):
                        ta = a.block(a.files[0], kv, lay, blk_a)
                        tb = b.block(b.files[0], kv, lay, blk_b)
                        if ta is None or tb is None:
                            continue
                        x, y = ta[:ca], tb[:ca]
                        neq = int((sig16(x) != sig16(y)).sum())
                        pr = pearson(x, y)
                        n = x.numel()
                        note = "仅 ULP 级" if neq / n < 0.002 else "深层残差流放大属正常"
                        say(f"     L{lay:02d} {kv}: 位翻转 {neq}/{n} ({neq / n:.2%}), "
                            f"Pearson={pr:.6f} ({note})")
            else:
                say("  A) 无重叠重算段(早请求第 k 块非部分块或块号相同), 跳过")
        else:
            say("  A) 单块请求或长短不足, 跳过")
        say()

    # ---------- 汇总 ----------
    say("=" * W)
    if not pairs:
        say("[DONE] 无前缀复用关系对; 归档请求如上。")
    else:
        say(f"[DONE] 前缀复用关系对 {len(pairs)} 个; A 重算一致性统计如上" 
            "(信息性: L00 仅 ULP 级 / 深层残差流放大属正常)。")
    say("=" * W)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[DONE] 报告已落盘: {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="kvc_1p1d P 侧前缀复用关系检查器(PD 归档版 pairwise)")
    ap.add_argument("--dir", default="tensors",
                    help="归档根目录(默认 tensors, 含 req*/P 子目录)")
    ap.add_argument("--out", default="logs/analysis/inspect_prefix_p.out",
                    help="输出 out 产物路径(默认 logs/analysis/inspect_prefix.out)")
    args = ap.parse_args()
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
