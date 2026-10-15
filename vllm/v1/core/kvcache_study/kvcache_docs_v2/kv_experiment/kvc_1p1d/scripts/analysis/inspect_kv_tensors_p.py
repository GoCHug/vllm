#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# inspect_kv_tensors_p.py —— kvc_1p1d P 侧(prefill producer) KV 归档离线查看器
#
# 输入: --dir tensors  (08 号 PD 归档版, 根目录下一请求一目录 req{seq}/, 请求内分 P/D;
#       本查看器只读请求下的 P/ 子目录)、P 侧语义不变
#       P 侧语义: TERM 时 P 实例把该请求全部物理块整块归档(w_tok=p_tok,
#       纯 prefill, 无 decode 槽) —— P→D 传输源头证据(供 inspect_p2d.py 比对)。
#   tensors/
#     req{seq}/              <- 一个业务请求一个目录
#       P/kv_pp{p}tp{t}.pt   <- P 侧归档(TP1 下 1 份)
#       D/kv_pp{p}tp{t}.pt   <- 对侧归档(本查看器不读)
#
# .pt bundle 结构(schema=kvt4-raw):
#   K / V: [层序 list, 每层 dict{块号: (block_size, kv_heads, head_dim) 整块张量}]
#   meta:  pp/tp/seq/request_id/tag/p_tok/w_tok/final/block_size/block_table/
#          cov(每块有效槽位)/layers/layer_ids/kv_heads/head_dim/dtype/dev/ts
#
# 报告内容(逐请求一节, 基础打印 + 完整落 out 文件):
#   开始加载请求 xxx 的 kvcache 物理张量(身份: request_id/tag/token 进度)
#   该请求使用了 block id [...] (num_blocks=N, block_size=B)
#   逐 worker -> 逐 block:
#     block id=N 头行(有效槽位)
#     块-行映射网格: 1 个 block 竖跨 group_size 层 —— L00/L01/.../L15 每层
#       K_cache[i]/V_cache[i] 张量(dim0) 各占第 N 行(层张量独立, 池块统一编址)
#     第 0 层示例: "第 0 层, block id=N 对应张量第 N 行, 张量信息如下"
#       —— k/v cache 的 shape/dtype/tensor 预览(其余层同构)
#   (tensor 预览 = 该块首 token 首 head 的前 N 维, --preview 可调; 完整张量请 torch.load)
#
# 用法:
#   python3 scripts/analysis/inspect_kv_tensors_p.py --dir tensors                # 报告 -> logs/analysis/inspect_kv_tensors_p.out
#   python3 scripts/analysis/inspect_kv_tensors_p.py --dir tensors --out xx.out     # 指定 out 产物
#   python3 scripts/analysis/inspect_kv_tensors_p.py --dir tensors --preview 8       # tensor 预览前 8 值
# ==============================================================================
import argparse
import glob
import sys
import time
from pathlib import Path

import torch

W = 78   # 报告横线宽度


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def load_bundle(path):
    """ 加载并校验 bundle(K/V/meta 三键)。"""
    b = torch.load(str(path), map_location="cpu", weights_only=False)
    if not (isinstance(b, dict) and "K" in b and "V" in b and "meta" in b):
        raise ValueError(f"非 kvc 归档结构: {path}")
    return b


def fmt_vals(t, n):
    """张量前 n 个值 -> [v1, v2, ...] 字符串(bf16 友好 %.4g)。"""
    vals = t.reshape(-1)[:n].tolist()
    return "[" + ", ".join(f"{v:+.4g}" for v in vals) + ", ...]"


def rid_label(meta, dir_name):
    rid = str(meta.get("request_id", "?"))
    return f"{dir_name}  ({rid})"


def layer_grid_rows(gs, blk):
    """块-行映射网格行: 1 个 block 竖跨 gs 层, 每层 K/V 张量(dim0) 各占第 blk 行。

    gs<=4 时逐层全列; 否则列首两层 + 省略行 + 末层, 中间层同构省略。
    """
    def row(li):
        return (f"      L{li:02d}   K_cache[{li:2d}] 第 {blk} 行   |   "
                f"V_cache[{li:2d}] 第 {blk} 行")

    if gs <= 0:
        return []
    if gs <= 4:
        return [row(li) for li in range(gs)]
    return [row(0), row(1),
            f"      ...   (L02~L{gs - 2:02d} 共 {gs - 3} 层与 L00 同构, 略)",
            row(gs - 1)]


# ---------------------------------------------------------------------------
# 报告主逻辑
# ---------------------------------------------------------------------------
def report(args) -> int:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"[ERROR] 归档目录不存在: {root}")
        return 2

    # 请求侧目录(req{seq}/P); 同时侦测旧版平铺归档并提示
    req_dirs = sorted([d / "P" for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("req") and (d / "P").is_dir()])
    old_flat = sorted(glob.glob(str(root / "kv_*.pt")))
    if old_flat:
        print(f"[WARN] 跳过 {len(old_flat)} 个旧版平铺归档(v2.3 及以前布局), "
              f"如需查看请升级到请求子目录布局(v2.4)")
    if not req_dirs:
        print(f"[ERROR] {root} 下无请求(req*/P): {root}")
        return 2

    # out 产物: 默认 logs/analysis/inspect_kv_tensors.out(相对 cwd), 建父目录
    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []          # 收集全部输出, 最后一次性写文件 + 回显

    def say(s=""):
        lines.append(s)

    def worker_section(f: Path):
        """单 worker 归档 -> 报告段(逐 block 的块-行映射网格 + 第 0 层张量示例)。"""
        b = load_bundle(f)
        m = b["meta"]
        K, V = b["K"], b["V"]
        cov = m.get("cov", [None] * len(m.get("block_table", [])))
        gs = m.get("group_size", m["layers"])   # 组内层数(v2.5 meta; 旧归档回退 layers)
        say(f"---- worker PP{m['pp']}_TP{m['tp']}  "
            f"[{f.name} | 组内 {gs} 层张量 | {m['dtype']} | {m.get('dev','?')}] ----")
        L0_K, L0_V = K[0], V[0]          # 示例取第 0 层(其余层 shape 同构)
        for bi, blk in enumerate(m.get("block_table", [])):
            k = L0_K.get(int(blk))
            v = L0_V.get(int(blk))
            if k is None and v is None:
                continue                  # cov=0 未归档的块
            cv = cov[bi] if bi < len(cov) else "?"
            say(f"  block id={blk}  (有效槽位 {cv}/{m['block_size']})")
            say(f"    块-行映射 · 1 个 block 竖跨 {gs} 层(group_size={gs}): "
                f"每层 K/V 张量(dim0) 各占第 {blk} 行")
            for r in layer_grid_rows(gs, blk):
                say(r)
            say()
            say(f"    第 0 层, block id={blk} 对应张量第 {blk} 行, "
                f"张量信息如下 (其余 {gs - 1} 层同构):")
            if k is not None:
                say(f"      k cache  shape={tuple(k.shape)}  dtype={k.dtype}")
                say(f"        k tensor  = {fmt_vals(k, args.preview)}")
            if v is not None:
                say(f"      v cache  shape={tuple(v.shape)}  dtype={v.dtype}")
                say(f"        v tensor  = {fmt_vals(v, args.preview)}")
            say()
        say()

    # ---------------- 报告头 ----------------
    say("=" * W)
    say("kvc_1p1d P 侧(prefill/producer)物理 KV 张量离线查看报告(08 PD 归档版)")
    say(f"归档目录: {root} | 请求: {len(req_dirs)} 个 | "
        f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * W)
    for rd in req_dirs:
        say(f"  {rd.parent.name}/{rd.name}/  ({len(list(rd.glob('kv_*.pt')))} 个 worker 归档)")
    say()

    # ---------------- 逐请求报告 ----------------
    for i, rd in enumerate(req_dirs, 1):
        try:
            files = sorted(rd.glob("kv_pp*.pt"))
            if not files:
                say(f"[WARN] {rd.parent.name}/{rd.name}/ 下无 kv_*.pt, 跳过")
                say()
                continue
            m0 = load_bundle(files[0])["meta"]
        except Exception as e:
            say(f"[WARN] 加载 {rd.name} 首个归档失败, 跳过: {e}")
            say()
            continue

        say("=" * W)
        say(f"[{i}/{len(req_dirs)}] 开始加载请求 {rid_label(m0, f"{rd.parent.name}/{rd.name}")} 的 kvcache 物理张量")
        say("=" * W)
        bt = m0.get("block_table", [])
        say(f"  request_id : {m0.get('request_id')}")
        say(f"  tag={m0.get('tag')}  prompt={m0.get('p_tok')} tok  "
            f"written={m0.get('w_tok')} tok  dev={m0.get('dev')}")
        say(f"  该请求使用了 block id {bt}  (num_blocks={len(bt)}, "
            f"block_size={m0['block_size']})")
        say(f"  块有效槽位 cov={m0.get('cov')}  "
            f"(kv_heads={m0.get('kv_heads')}, head_dim={m0.get('head_dim')}, "
            f"含未写槽位原样归档)")
        say()
        for f in files:
            try:
                worker_section(f)
            except Exception as e:
                say(f"[WARN] {f.name} 解析失败, 跳过: {e}")
                say()

    say("=" * W)
    say("[END] 报告完毕 —— tensor 预览为首 token 首 head 前 "
        f"{args.preview} 值(第 0 层); 完整张量请 torch.load(<归档路径>)['K'/'V']")

    # ---------------- 落盘 + 终端回显 ----------------
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[DONE] 报告已落盘: {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="kvc_1p1d P 侧 KV 归档离线查看器(PD 归档版, 简洁报告版)")
    ap.add_argument("--dir", default="tensors",
                    help="归档根目录(默认 tensors, 含 req*/P 子目录)")
    ap.add_argument("--out", default="logs/analysis/inspect_kv_tensors_p.out",
                    help="输出 out 产物路径(默认 logs/analysis/inspect_kv_tensors.out)")
    ap.add_argument("--preview", type=int, default=4,
                    help="tensor 预览前 N 个值(默认 4)")
    args = ap.parse_args()
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
