#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# inspect_p2d.py —— kvc_1p1d P→D KVCache 传输正确性检查器(离线, 逐位级)
#
# 输入: --dir tensors   (根目录, 一请求一目录 req{seq}/, 请求内分 P/D)
#   tensors/req{seq}/P/kv_pp0tp0.pt   P(prefill producer) 侧归档
#   tensors/req{seq}/D/kv_pp0tp0.pt   D(decode consumer)  侧归档
#
# 配对: 按 seq(P/D 各自独立递增, proxy 双发保证完成序一致)。
#   rid 尾8 两侧不同(proxy 改写 request_id) —— seq 即目录名, 配对天然成立, p_tok 相等性做哨兵。
#
# 检查(每对一节, 逐层逐池):
#   A) 结构对齐  : side/p_tok/层数/heads/dim 一致; 重建 token-major 长度 = w_tok
#   B) Tx 区逐位 : 前 p_tok-1 个 token(P 侧产出并迁移加载到 D) 全层 K/V
#                  torch.equal —— 不等 = 传输损伤(FAIL, 附首差 token/head/dim
#                  定位 + 位翻转统计)。 期望: 两侧逐位相等(传输无损)。
#   C) 尾槽对比  : [p_tok-1, w_tok_P) 的重叠槽 —— P 侧(纯 prefill 末 token
#                  forward 产物) vs D 侧(bootstrap 补算真 prev token) —— 预期
#                  存在 ULP 级数值差(两侧算法路径不同), 报告位翻转/Pearson/
#                  幅度比(|Δ|max / P 层幅值, 参考判据 ≤5% 属“重算一致”)。
#   D) D 独有段  : [w_tok_P, w_tok_D) = D 本地 decode 新写槽(P 无对应) ——
#                  数值健康检查(无 NaN/Inf) + 统计行, 不做对比。
#
# 输出: --out logs/analysis/inspect_p2d.out(报告式; [DONE] 汇总 verdict)
# 用法:
#   python3 scripts/analysis/inspect_p2d.py --dir tensors                 # -> logs/analysis/inspect_p2d.out
# ==============================================================================
import argparse
import sys
import time
from pathlib import Path

import torch

W = 78


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


def seq_of(name):
    # req{seq}_{rid尾8}
    try:
        return int(name.split("_")[0][3:])
    except ValueError:
        return None


def rebuild(pool, meta):
    """块结构 kvt4-raw -> token-major: 按 block_table 顺序 cat 各块 [:cov]。"""
    bt = [int(b) for b in meta["block_table"]]
    cov = [int(c) for c in meta["cov"]]
    parts = []
    for e, blk in enumerate(bt):
        c = cov[e] if e < len(cov) else 0
        if c <= 0:
            continue
        t = pool.get(blk)
        if t is None:
            raise ValueError(f"块 {blk} 缺失于归档")
        parts.append(t[:c])
    return torch.cat(parts, dim=0) if parts else None


def report(args) -> int:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"[ERROR] 归档根目录不存在: {root}")
        return 2

    p_reqs = sorted([d / "P" for d in root.iterdir()
                     if d.is_dir() and d.name.startswith("req") and (d / "P").is_dir()],
                     key=lambda d: seq_of(d.parent.name) or 0)
    d_reqs = sorted([d / "D" for d in root.iterdir()
                     if d.is_dir() and d.name.startswith("req") and (d / "D").is_dir()],
                     key=lambda d: seq_of(d.parent.name) or 0)
    p_map = {seq_of(d.parent.name): d for d in p_reqs}
    d_map = {seq_of(d.parent.name): d for d in d_reqs}

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    def say(s=""):
        lines.append(s)

    say("=" * W)
    say("kvc_1p1d P→D KVCache 传输正确性检查(块结构 kvt4-raw · seq 配对)")
    say(f"归档根: {root} | P 侧 {len(p_reqs)} 个 / D 侧 {len(d_reqs)} 个 | "
        f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * W)
    for d in p_reqs:
        say(f"  {d.parent.name}/{d.name}/")
    for d in d_reqs:
        say(f"  {d.parent.name}/{d.name}/")
    say()

    pairs = sorted(set(p_map) & set(d_map))
    only_p = sorted(set(p_map) - set(d_map))
    only_d = sorted(set(d_map) - set(p_map))
    for s in only_p:
        say(f"  [WARN] seq={s} 仅 P 侧有归档({p_map[s].parent.name}/{p_map[s].name}), 无法配对")
    for s in only_d:
        say(f"  [WARN] seq={s} 仅 D 侧有归档({d_map[s].parent.name}/{d_map[s].name}), 无法配对")
    if not pairs:
        say("[ERROR] 无可配对请求(P/D 侧 seq 无交集)")
    say()

    n_pass = n_fail = 0
    for idx, s in enumerate(pairs, 1):
        fp = sorted(p_map[s].glob("kv_*.pt"))
        fd = sorted(d_map[s].glob("kv_*.pt"))
        if len(fp) != 1 or len(fd) != 1:
            say(f"[WARN] seq={s}: 归档文件数异常 P={len(fp)} D={len(fd)}, 跳过")
            continue
        bp = load_bundle(fp[0])
        bd = load_bundle(fd[0])
        mp, md = bp["meta"], bd["meta"]

        say("=" * W)
        say(f"[{idx}/{len(pairs)}] seq={s}: {p_map[s].parent.name}/{p_map[s].name}  --传输-->  {d_map[s].parent.name}/{d_map[s].name}")
        say("=" * W)

        # ---- A) 结构对齐 ----
        bad = []
        if mp.get("side") != "P":
            bad.append(f"P 侧 meta.side={mp.get('side')!r}")
        if md.get("side") != "D":
            bad.append(f"D 侧 meta.side={md.get('side')!r}")
        if int(mp.get("p_tok", -1)) != int(md.get("p_tok", -2)):
            bad.append(f"p_tok 不等 P={mp.get('p_tok')} D={md.get('p_tok')}")
        for k in ("layers", "kv_heads", "head_dim", "block_size", "dtype"):
            if str(mp.get(k)) != str(md.get(k)):
                bad.append(f"{k} 不一致 P={mp.get(k)} D={md.get(k)}")
        p_tok = int(mp.get("p_tok", 0) or 0)
        w_p, w_d = int(mp["w_tok"]), int(md["w_tok"])
        say(f"  A) 结构: side P/D ✓  p_tok={p_tok}(哨兵等)  层×heads×dim="
            f"{mp.get('layers')}×{mp.get('kv_heads')}×{mp.get('head_dim')}  "
            f"w_tok P={w_p}(纯prefill) / D={w_d}({'迁移'+str(p_tok-1)+' + bootstrap+decode '+str(w_d-p_tok) if w_d > p_tok else '迁移+补算'})")
        if bad:
            say(f"     [FAIL-A] " + "; ".join(bad))
        nlay = int(mp.get("layers", 0) or 0)
        say(f"     块表: P {mp.get('block_table')} cov={mp.get('cov')}")
        say(f"           D {md.get('block_table')} cov={md.get('cov')}")

        # B/C/D 三段都按层分桶统计(全层汇总 + 首差层定位), 逐层逐池计算
        tx = max(0, p_tok - 1)
        seg_c_lo, seg_c_hi = tx, min(w_p, w_d)       # C 段: 同位重叠尾槽(P 哑 token vs D bootstrap)
        seg_d_lo, seg_d_hi = w_p, w_d                 # D 段: D 独有 decode 槽
        say(f"  区间划分: Tx=[0,{tx})  重算槽=[{seg_c_lo},{seg_c_hi})  D独有decode=[{seg_d_lo},{seg_d_hi})")

        eq_layers = 0
        diff_layers = []
        c_reports = []
        d_bad = []
        for li in range(nlay):
            for kv in ("K", "V"):
                tp = rebuild(bp[kv][li], mp)
                td = rebuild(bd[kv][li], md)
                if tp is None or td is None:
                    continue
                # ---- B) Tx 区逐位 ----
                if tx > 0 and tx <= tp.shape[0] and tx <= td.shape[0]:
                    x, y = tp[:tx], td[:tx]
                    if torch.equal(sig16(x), sig16(y)):
                        eq_layers += 1
                    else:
                        neq = (sig16(x) != sig16(y))
                        nflip = int(neq.sum())
                        pos = neq.nonzero()[0]
                        tok_, h_, d_ = int(pos[0]), int(pos[1]), int(pos[2])
                        diff_layers.append(f"L{li:02d}{kv} 位翻转 {nflip}/{x.numel()} "
                                           f"首差 token={tok_} head={h_} dim={d_}")
                # ---- C) 重算槽(同位) ----
                if seg_c_hi > seg_c_lo:
                    n = min(seg_c_hi, tp.shape[0], td.shape[0]) - seg_c_lo
                    if n > 0:
                        x, y = tp[seg_c_lo:seg_c_lo + n], td[seg_c_lo:seg_c_lo + n]
                        nf = int((sig16(x) != sig16(y)).sum())
                        pr = pearson(x, y)
                        rel = float((x.float() - y.float()).abs().max() / (x.float().abs().max() + 1e-9))
                        c_reports.append((li, kv, n, nf, pr, rel))
                # ---- D) D 独有段健度 ----
                if seg_d_hi > seg_d_lo and seg_d_hi <= td.shape[0]:
                    z = td[seg_d_lo:seg_d_hi].float()
                    if torch.isnan(z).any() or torch.isinf(z).any():
                        d_bad.append(f"L{li:02d}{kv} 含 NaN/Inf")

        tot_pairs = nlay * 2
        ok_b = eq_layers == tot_pairs and tot_pairs > 0
        say(f"  B) Tx 区逐位(前 {tx} tok): {eq_layers}/{tot_pairs} 层池 torch.equal "
            f"{'PASS(传输无损)' if ok_b else 'FAIL(传输损伤!)'}")
        for m in diff_layers[:8]:
            say(f"     [TX-DIFF] {m}")
        if len(diff_layers) > 8:
            say(f"     ... 其余 {len(diff_layers)-8} 处略")

        # C 汇总
        if c_reports:
            nf_tot = sum(r[3] for r in c_reports)
            pr_min = min(r[4] for r in c_reports)
            rel_max = max(r[5] for r in c_reports)
            n_slots = c_reports[0][2]
            note = "重算一致(正常)" if rel_max <= 0.05 else "幅度超参考带(请复核)"
            say(f"  C) 重算槽(P 哑token前向 vs D bootstrap 补算, {n_slots} 槽): "
                f"位翻转 {nf_tot} 元素槽, Pearson≥{pr_min:.6f}, |Δ|max/层幅值 max={rel_max:.2%} —— {note}")
            worst = max(c_reports, key=lambda r: r[5])
            say(f"     最差层池: L{worst[0]:02d}{worst[1]} (rel={worst[5]:.2%})")
        else:
            say("  C) 重算槽: 无(P w_tok <= p_tok-1?)")

        # D 汇总
        n_dslot = seg_d_hi - seg_d_lo
        if n_dslot > 0:
            say(f"  D) D 独有 decode 段({n_dslot} 槽): "
                f"{'健康(无 NaN/Inf)' if not d_bad else 'FAIL: ' + ';'.join(d_bad[:4])}")
        else:
            say("  D) D 独有 decode 段: 无(w_tok_D == w_tok_P)")

        if ok_b and not bad and not d_bad:
            n_pass += 1
            say("  ==> 本对 verdict: PASS")
        else:
            n_fail += 1
            say("  ==> 本对 verdict: FAIL")
        say()

    say("=" * W)
    if not pairs:
        say("[DONE] 无配对请求, 检查未执行。")
    else:
        verdict = "PASS" if n_fail == 0 else "FAIL"
        say(f"[DONE] 传输正确性: {n_pass}/{len(pairs)} 对 PASS"
            + (f", {n_fail} 对 FAIL" if n_fail else "")
            + f" —— Tx 区逐位相等即 P→D KVCache 传输无损; 尾槽/decode 段为 D 本地语义(信息性)。")
    say("=" * W)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[DONE] 报告已落盘: {out_path}")
    return 0 if n_fail == 0 and pairs else (1 if pairs else 2)


def main():
    ap = argparse.ArgumentParser(
        description="kvc_1p1d P→D KVCache 传输正确性检查器(seq 配对, Tx 逐位)")
    ap.add_argument("--dir", default="tensors",
                    help="归档根目录(默认 tensors, 含 P/ 与 D/ 子目录)")
    ap.add_argument("--out", default="logs/analysis/inspect_p2d.out",
                    help="输出 out 产物路径(默认 logs/analysis/inspect_p2d.out)")
    args = ap.parse_args()
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
