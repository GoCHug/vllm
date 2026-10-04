#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""matrix_report.py —— kvc_1p1d_prefix 四象限对照总表 + 铁律核验

扫描 logs/q{1..4}_*/ 的 PCM 证据(第二请求 req_r 486 tok), 生成:
  1. 四象限对照总表(P/D local_hit, external, recv, 传输 MiB, took, eff 带宽, 命中率)
  2. 铁律核验 PASS/FAIL —— 本实验核心判据(workload: req_p 324 种缓存 -> req_r 486 前缀复用 256):
     铁律A: P 本地命中只看 P 开关(P✓=256 tok / P✗=0)
     铁律B: D 本地命中与传输量只看 D 开关(D✓=256 命中+32.0 MiB 增量 / D✗=0 命中+64.0 MiB 全量, 整块栅格)
     铁律C: P 上报块恒全量(四象限 PFINISH report_blocks 组数恒 4, 不感知 D 缓存)
     铁律D: 输出正确性(①~④ resp_r completion=35 / finish=length, 与象限无关)

用法: python3 scripts/analysis/matrix_report.py --dir logs   (产物 -> logs/analysis/matrix_report.out)
"""
import argparse
import ast
import json
import re
from pathlib import Path

QNAMES = ["q1_p1d1", "q2_p1d0", "q3_p0d1", "q4_p0d0"]


def grab(pattern, text, default="?"):
    ms = re.findall(pattern, text)
    return ms[-1] if ms else default


def read_side(qdir, prefix):
    """读一侧证据文本: 全量日志优先(llama.log), 缺失时回退抽取件拼装(legacy 轮形态)"""
    log = qdir / f"{prefix}_llama.log"
    if log.exists():
        return log.read_text(errors="ignore")
    parts = []
    for name in (f"{prefix}_pcm.txt", f"{prefix}_transfer.txt", f"{prefix}_hitrate.txt",
                 f"{prefix}_delayfree.txt"):
        f = qdir / name
        if f.exists():
            parts.append(f.read_text(errors="ignore"))
    return "\n".join(parts)


def load_q(qdir):
    d = {"ok": False}
    pt = read_side(qdir, "p")
    dt = read_side(qdir, "d")

    for key, fname in (("resp_p", "resp_p.json"), ("resp_r", "resp_r.json")):
        p = qdir / fname
        d[key] = None
        if p.exists():
            try:
                j = json.loads(p.read_text())
                ch = (j.get("choices") or [{}])[0]
                u = j.get("usage", {})
                d[key] = {"fin": ch.get("finish_reason"), "cmpl": u.get("completion_tokens"),
                          "prompt": u.get("prompt_tokens"), "text": (ch.get("text") or "")[:12]}
            except Exception:
                pass

    d["cfg_p"] = grab(r"\[PCM\] CFG role=\S+ enable_prefix_caching=(\S+)", pt)
    d["cfg_d"] = grab(r"\[PCM\] CFG role=\S+ enable_prefix_caching=(\S+)", dt)
    d["phit"] = grab(r"\[PCM\] SCHED \S+ prompt=486 local_hit=(\d+)", pt)
    d["dhit"] = grab(r"\[PCM\] SCHED \S+ prompt=486 local_hit=(\d+)", dt)

    m = re.findall(r"\[PCM\] ALLOC \S+ external=(\d+) recv_blocks=(\[.*?\]\]) all_blocks=", dt)
    if m:
        d["ext"], d["recv"] = m[-1]
    m = re.findall(r"\[PCM\] PFINISH \S+ prompt=486 prompt_blocks=\d+ report_blocks=(\[.*?\])", pt)
    if m:
        d["pfin"] = m[-1]
    m = re.findall(r"\[PCM\] XFER-end \S+ segments=(\d+) bytes=(\d+) \(([\d.]+) MiB\) "
                   r"eff_GBps=([\d.]+) pull_local=(.*?) pull_remote=(.*)", dt)
    if m:
        d["seg"], d["bytes"], d["mib"], d["gbps"], d["plocal"], d["premote"] = m[-1]
    d["took"] = grab(r"KV cache transfer for request \S+ took ([\d.]+) ms", dt)
    d["prate"] = grab(r"Prefix cache hit rate: *([\d.]+)%?", pt)
    d["drate"] = grab(r"Prefix cache hit rate: *([\d.]+)%?", dt)

    # 成功判定: req_r 响应就位 + ALLOC(external) 证据在
    d["ok"] = bool(d.get("resp_r") and d.get("ext"))
    return d


def parse_flags(qn):
    """'q1_p1d1' -> (P开, D开): 尾段 p{0|1}d{0|1}"""
    flags = qn.split("_")[-1]
    return "p1" in flags, flags.endswith("d1")


def verdict(qn, d):
    """铁律核验 -> [(标签, 实测, 期望, bool)]"""
    out = []
    p_on, d_on = parse_flags(qn)

    phit = d.get("phit", "?")
    out.append(("A-P本地命中", phit, "256" if p_on else "0", phit == ("256" if p_on else "0")))

    dhit = d.get("dhit", "?")
    out.append(("B-D本地命中", dhit, "256" if d_on else "0", dhit == ("256" if d_on else "0")))
    mib = d.get("mib")
    if mib and mib != "?":
        exp = 32.0 if d_on else 64.0
        out.append(("B-传输MiB", mib, f"{exp:.1f}", abs(float(mib) - exp) < 0.1))

    pfin = d.get("pfin")
    if pfin:
        try:
            # report_blocks=[4] = "1 组共 4 块" -> 总块数恒等 prompt 满块数(ceil(486/128)=4)
            n = sum(ast.literal_eval(pfin))
            out.append(("C-P上报总块数", n, 4, n == 4))
        except Exception:
            pass

    rr = d.get("resp_r")
    if rr:
        out.append(("D-输出tok", rr.get("cmpl"), 35, rr.get("cmpl") == 35 and rr.get("fin") == "length"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs")
    args = ap.parse_args()
    root = Path(args.dir)
    bc = "=" * 100
    L = []
    L.append(bc)
    L.append("kvc_1p1d_prefix 1P1D Prefix Cache 四象限汇总(PCM 六打点 · 第二请求 req_r=486 tok)")
    L.append(f"扫描根: {root} | 象限: 4 个(缺则标 MISS) | workload: req_p 324(种) -> req_r 486(复用 256)")
    L.append(bc)
    for qn in QNAMES:
        qd = root / qn
        L.append(f"  {qn}/  [{'有产物' if qd.exists() and any(qd.iterdir()) else 'MISS'}]")

    ds = {}
    L.append("")
    L.append("[表] 四象限对照总表")
    L.append("| 象限 | pc(P/D) | P.hit | D.hit | ext | recv | MiB | seg | eff(GB/s) | took(ms) | hit%(P/D) | resp_r(cmpl) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for qn in QNAMES:
        qd = root / qn
        d = load_q(qd) if qd.exists() else {"ok": False}
        ds[qn] = d
        if not d.get("ok"):
            L.append(f"| {qn} | — | (未跑/未完成) | | | | | | | | | |")
            continue
        p_on, d_on = parse_flags(qn)
        pn, dn = ("✓" if p_on else "✗"), ("✓" if d_on else "✗")
        row = [qn, f"P{pn}/D{dn}", d.get("phit", "?"), d.get("dhit", "?"), d.get("ext", "?"),
               d.get("recv", "?"), d.get("mib", "?"), d.get("seg", "?"), d.get("gbps", "?"),
               d.get("took", "?"), f"{d.get('prate', '—')}/{d.get('drate', '—')}",
               (d.get("resp_r") or {}).get("cmpl", "?")]
        L.append("| " + " | ".join(str(x) for x in row) + " |")

    L.append("")
    L.append("> 注: hit% 为 Prometheus 末次采样值(10s 窗口可能错过末次命中——如权威轮 q1 的 P 侧实际"
             " 31.6% 被窗口错过仍显示 0.0%; 命中权威判据 = SCHED local_hit / (324+486))。")

    L.append("")
    L.append("[表] 铁律核验(核心判据) ——")
    n_all = n_pass = 0
    for qn in QNAMES:
        d = ds[qn]
        if not d.get("ok"):
            continue
        L.append(f"  {qn}:")
        for tag, val, exp, ok in verdict(qn, d):
            n_all += 1
            n_pass += int(bool(ok))
            L.append(f"    {tag:12s} 实测={val}  期望={exp}  {'PASS' if ok else 'FAIL'}")

    L.append("")
    if n_all and n_all == n_pass:
        L.append(f"[DONE] 铁律核验 {n_pass}/{n_all} 项 PASS —— 四条铁律全中: "
                 "P 开关只管算多少 / D 开关只管传多少 / P 上报恒全量 / 正确性与象限无关")
    elif n_all:
        L.append(f"[WARN] 铁律核验 {n_pass}/{n_all} 项 PASS —— 存在 FAIL 行, 逐条核对上表")
    else:
        L.append("[WARN] 无可核验象限(四象限均未跑) —— 先 bash scripts/server/run_matrix.sh")

    out = root / "analysis" / "matrix_report.out"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] 报告已落盘: {out}")


if __name__ == "__main__":
    main()
