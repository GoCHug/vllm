#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_tensors.py — 用三个实验保存下来的归档独立核验可视化结论.

运行:  cd kvcache_docs_v2/viz && python3 verify_tensors.py
       (路径按 __file__ 解析: viz_data 取脚本同级, 三实验目录取上级, 任意 CWD 皆可)

原则: 不复用 viz_export.py 的任何计算 —— 直接读 .pt 归档与原始 PCM 日志
重新推导全部核验量, 再把结果与 viz_data/manifest.json(页面数据源)对账:

  A. kvc(单机 PP2×TP2, 8 个 .pt)
     - 前缀复用: req1(P 种块) vs req2(R 命中) 共享表头块 [1,2]
       逐 worker×层×池 torch.equal  → 预期全等(零篡改)
     - 未写槽位基线零(cov 之外)
     - TP 分头 sanity: 同块不同 TP/PP 的张量应互不相等
     - viz_data 对账: 核验布尔 / 统计(rms±2e-5) / 热图字节(逐位)
  B. kvc_1p1d(PD 分离, 4 个 .pt)
     - P→D Tx 区 [0,p_tok-1) 逐层逐池 torch.equal → 预期全等(传输无损)
     - 块级区间划分(rowsTx/重算槽/D独有)独立重算
     - 重算槽 ULP 统计(位翻转/Pearson/|Δ|max)独立重算
     - D 独有 decode 段健康(有限性)
     - viz_data 对账: 上述全部 + 热图字节
  C. kvc_1p1d_prefix(无 tensors 归档 —— PCM 六打点即实验证据)
     - 从 logs/q*_p?d?/ 原始文本独立重解四象限关键量 + 铁律 A/B1/B2/C/D
     - viz_data 对账: 象限字段与铁律判定
  D. kvcFlow(kvc 全流程状态机回放)
     - 把 P/R 归档步的块表/cov 与 .pt meta 逐位对账(与 A 节双路互证)
     - 空闲队列守恒(首末 13290) / 哈希表终态 / touch·popleft 计数演进
"""
import base64
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent   # 本脚本所在 viz/
BASE = ROOT.parent                          # 上级 kvcache_docs_v2(kvc 等三实验所在)
MAN = json.loads((ROOT / "viz_data" / "manifest.json").read_text())
POOLS = ("K", "V")

NP_, NF_ = [0], [0]


def ck(name, cond, extra=""):
    if cond:
        NP_[0] += 1
        print(f"  PASS: {name}")
    else:
        NF_[0] += 1
        print(f"  FAIL: {name}  {extra}")


def sep(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def load_pt(p):
    b = torch.load(str(p), map_location="cpu", weights_only=False)
    for k in ("K", "V", "meta"):
        if k not in b:
            raise ValueError(f"非 kvt4-raw 归档(缺 {k}): {p}")
    return b


def plain(v):
    return v.tolist() if torch.is_tensor(v) else v


def starts_of(cov):
    s, st = 0, []
    for c in cov:
        st.append(s)
        s += int(c)
    return st


def pearson(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def heat_expect(bundle, pool, blk):
    """独立重算 32×32 下采样热图(头0通道 4×4 平均池化, 逐层)."""
    lay = bundle[pool]
    hs = []
    for i in range(len(lay)):
        x = lay[i][blk][0:128, 0, 0:128].float()
        hs.append(x.view(32, 4, 32, 4).mean(dim=(1, 3)))
    return torch.stack(hs).numpy()


def file_json(href):
    return json.loads((ROOT / "viz_data" / href).read_text())


def heat_array(fj, pool, blk):
    a = np.frombuffer(base64.b64decode(fj["heat"][pool][blk]), dtype="<f4")
    return a.reshape(a.size // (32 * 32), 32, 32)


def close(x, y, tol=2e-5):
    return abs(float(x) - float(y)) <= tol * max(1.0, abs(float(y)))


# ================================================================ A. kvc
def section_kvc():
    sep("A. kvc 单机 PP2×TP2 —— 前缀复用逐位核验(独立重算 vs viz_data)")
    tdir = BASE / "kvc" / "tensors"
    reqs = sorted([d for d in tdir.iterdir() if d.is_dir() and d.name.startswith("req")])
    ck("归档请求目录 = 2", len(reqs) == 2, f"got {len(reqs)}")
    bundles = {}
    for d in reqs:
        for f in sorted(d.glob("kv_*.pt")):
            b = load_pt(f)
            w = f"pp{plain(b['meta'].get('pp', 0))}tp{plain(b['meta'].get('tp', 0))}"
            bundles.setdefault(d.name, {})[w] = b
    nfiles = sum(len(v) for v in bundles.values())
    ck(".pt 文件数 = 8(2 请求 × 4 worker)", nfiles == 8, f"got {nfiles}")

    names = sorted(bundles, key=lambda n: int(n.split("_")[0][3:]))
    early, late = names[0], names[1]
    be0 = next(iter(bundles[early].values()))
    bl0 = next(iter(bundles[late].values()))
    bt = [int(x) for x in be0["meta"]["block_table"]]
    cv = [int(x) for x in be0["meta"]["cov"]]
    btL = [int(x) for x in bl0["meta"]["block_table"]]
    cvL = [int(x) for x in bl0["meta"]["cov"]]

    # 共享表头块独立推导
    shared, k = [], 0
    while (k < len(bt) and k < len(btL) and bt[k] == btL[k]
           and min(cv[k], cvL[k]) == 128):
        shared.append(bt[k])
        k += 1
    hit = sum(min(cv[i], cvL[i]) for i in range(k))
    ck("共享表头块 = [1, 2]", shared == [1, 2], f"got {shared}")
    ck("命中 tokens = 256", hit == 256, f"got {hit}")

    # 逐 worker×层×池 torch.equal(独立重算)
    eq_ind = {}
    nfail = 0
    for w in sorted(bundles[early]):
        bE, bL = bundles[early][w], bundles[late][w]
        nL = len(bE["K"])
        ck(f"worker {w}: 16 层", nL == 16, f"got {nL}")
        for p in POOLS:
            for blk in shared:
                for i in range(nL):
                    te, tl = bE[p][i].get(blk), bL[p][i].get(blk)
                    ok = (te is not None and tl is not None
                          and te.shape == tl.shape and bool(torch.equal(te, tl)))
                    eq_ind[(w, p, str(blk), i)] = ok
                    nfail += 0 if ok else 1
    ntot = len(eq_ind)
    ck(f"前缀复用逐位相等 {ntot - nfail}/{ntot}(期望全等 = 零篡改)",
       nfail == 0 and ntot == 256, f"nfail={nfail} ntot={ntot}")

    # viz_data 对账: manifest 核验布尔
    mkvc = MAN["scenarios"]["kvc"]
    mpf = mkvc["prefix"]["pairs"][0]
    ck("manifest: sharedBlocks/hitTokens/nFail 一致",
       mpf["sharedBlocks"] == shared and mpf["hitTokens"] == hit
       and mpf["nFail"] == nfail and bool(mpf["overallPass"]) is (nfail == 0))
    mism = 0
    for w, c in mpf["checks"].items():
        for p in POOLS:
            for blk in shared:
                for i in range(c["nLayers"]):
                    if bool(c[p][str(blk)][i]) != eq_ind[(w, p, str(blk), i)]:
                        mism += 1
    ck("manifest 核验布尔与独立重算 256/256 一致", mism == 0, f"mism={mism}")

    # 未写槽位基线零: 两个请求的末块
    for name, lastblk, cov_end in ((early, bt[-1], cv[-1]), (late, btL[-1], cvL[-1])):
        zeros_ok = True
        for w, b in bundles[name].items():
            for p in POOLS:
                for i in range(len(b["K"])):
                    if b[p][i][lastblk][cov_end:128].abs().max().item() != 0:
                        zeros_ok = False
        ck(f"{name} 末块 blk{lastblk} cov[{cov_end}:128) 未写槽位全零(全 worker×层×池)",
           zeros_ok)

    # TP 分头 / PP 分层组 sanity
    b00 = bundles[early]["pp0tp0"]["K"][0][shared[0]]
    b01 = bundles[early]["pp0tp1"]["K"][0][shared[0]]
    b10 = bundles[early]["pp1tp0"]["K"][0][shared[0]]
    ck("TP 分头 sanity: pp0tp0 vs pp0tp1 同块候不同值(head 片不同)",
       not torch.equal(b00, b01))
    ck("PP 分层组 sanity: pp0tp0 vs pp1tp0 同块候不同值(层组不同)",
       not torch.equal(b00, b10))

    # stats + 热图字节对账(全部 worker × 池 × 块 × 层)
    stats_bad, heat_bad, heat_diffs = 0, 0, []
    for r in mkvc["requests"]:
        for f in r["files"]:
            b = bundles[r["name"]][f["worker"]]
            fj = file_json(f["href"])
            for p in POOLS:
                for blk in r["blockTable"]:
                    bs = str(blk)
                    for i in range(b["nLayers"] if "nLayers" in b else len(b["K"])):
                        t = b[p][i][blk].float()
                        direct = {"rms": float((t * t).mean().sqrt()),
                                  "mean": float(t.mean()),
                                  "min": float(t.min()),
                                  "max": float(t.max())}
                        for key in ("rms", "mean", "min", "max"):
                            if not close(fj["stats"][p][bs][key][i], direct[key]):
                                stats_bad += 1
                    diff = float(np.abs(heat_array(fj, p, bs)
                                        - heat_expect(b, p, blk)).max())
                    heat_diffs.append(diff)
                    if diff > 1e-6:
                        heat_bad += 1
    ck("viz_data stats 对账: 全部 层×块×池×4 指标 |Δ|≤2e-5",
       stats_bad == 0, f"bad={stats_bad}")
    ck("viz_data 热图对账: 全部 块×池 逐位一致(max|Δ|≤1e-6)",
       heat_bad == 0, f"bad={heat_bad}, maxdiff={max(heat_diffs):.3e}")
    print(f"  [info] kvc 热图最大偏差 = {max(heat_diffs):.3e}")
    return mkvc


# ================================================================ B. 1p1d
def rebuild(bundle, bt, cov):
    """逐层 token-major rebuild: {pool: [层 -> [Σcov, heads, dim]]}."""
    out = {}
    for p in POOLS:
        lay = bundle[p]
        rows = []
        for i in range(len(lay)):
            parts = [lay[i][blk][:int(c)] for blk, c in zip(bt, cov) if int(c) > 0]
            rows.append(torch.cat(parts, dim=0))
        out[p] = rows
    return out


def section_p1d():
    sep("B. kvc_1p1d PD 分离 —— P→D 传输正确性(独立重算 vs viz_data)")
    td = BASE / "kvc_1p1d" / "tensors"
    pairs_ind = []
    pt_by = {}
    for d in sorted([x for x in td.iterdir() if x.is_dir() and x.name.startswith("req")]):
        seq = int(d.name[3:].split("_")[0])
        ppath = sorted((d / "P").glob("kv_*.pt"))[0]
        dpath = sorted((d / "D").glob("kv_*.pt"))[0]
        pb, db = load_pt(ppath), load_pt(dpath)
        pt_by[seq] = {"P": pb, "D": db, "paths": (ppath, dpath)}
        mp, md = pb["meta"], db["meta"]
        pTok = int(plain(mp.get("p_tok", -1)))
        nL = len(pb["K"])
        btP = [int(x) for x in mp["block_table"]]
        cvP = [int(x) for x in mp["cov"]]
        btD = [int(x) for x in md["block_table"]]
        cvD = [int(x) for x in md["cov"]]
        wP, wD = sum(cvP), sum(cvD)

        ck(f"seq={seq} 结构: side P/D · p_tok 等哨兵",
           plain(mp.get("side")) == "P" and plain(md.get("side")) == "D"
           and int(plain(md.get("p_tok", -2))) == pTok)
        ck(f"seq={seq} 结构: 32 层 × kv_heads 8 × head_dim 128",
           nL == 32 and int(plain(mp.get("kv_heads"))) == 8
           and int(plain(mp.get("head_dim"))) == 128
           and int(plain(md.get("layers"))) == 32)

        rP, rD = rebuild(pb, btP, cvP), rebuild(db, btD, cvD)

        # Tx 区逐层逐池 torch.equal
        tx_bad = 0
        for p in POOLS:
            for i in range(nL):
                if not bool(torch.equal(rP[p][i][:pTok - 1], rD[p][i][:pTok - 1])):
                    tx_bad += 1
        ck(f"seq={seq} Tx 区 [0,{pTok-1}) 逐层逐池 torch.equal 64/64",
           tx_bad == 0, f"bad={tx_bad}")

        # 块级区间划分独立重算 + 逐块等值
        stP, stD = starts_of(cvP), starts_of(cvD)
        posP = {b: i for i, b in enumerate(btP)}
        grid_ind = []
        grid_eq_bad = 0
        for iD, blk in enumerate(btD):
            if blk not in posP:
                grid_ind.append({"blk": blk, "kind": "donly", "rows": cvD[iD],
                                 "startD": stD[iD]})
                continue
            ip = posP[blk]
            cp, cd = cvP[ip], cvD[iD]
            rowsTx = max(0, min(cp, cd, (pTok - 1) - stP[ip]))
            rowsRe = 1 if stP[ip] <= pTok - 1 < stP[ip] + min(cp, cd) else 0
            rowsDonly = max(0, cd - max(cp, rowsTx + rowsRe))
            eq = {"K": None, "V": None}
            if rowsTx > 0:
                for p in POOLS:
                    eq[p] = [bool(torch.equal(pb[p][i][blk][:rowsTx],
                                              db[p][i][blk][:rowsTx]))
                             for i in range(nL)]
                    grid_eq_bad += 0 if all(eq[p]) else 1
            grid_ind.append({"blk": blk, "kind": "shared", "rowsTx": rowsTx,
                             "rowsRe": rowsRe, "rowsDonly": rowsDonly,
                             "eq": eq})
        ck(f"seq={seq} 块级 Tx 等值(逐层)全 PASS",
           grid_eq_bad == 0, f"bad={grid_eq_bad}")

        # 重算槽 ULP 统计
        rec = {}
        for p in POOLS:
            bit, pea, rel = [], [], []
            worst = {"rel": 0.0, "pool": "-"}
            for i in range(nL):
                a = rP[p][i][pTok - 1: pTok]
                b = rD[p][i][pTok - 1: pTok]
                bit.append(int((a.view(torch.int16) != b.view(torch.int16)).sum()))
                pea.append(pearson(a, b))
                rv = float((a.float() - b.float()).abs().max() / (a.abs().max() or 1e-9))
                rel.append(rv)
                if rv > worst["rel"]:
                    worst = {"rel": rv, "pool": f"L{i:02d}{p}"}
            rec[p] = {"bit": bit, "pea": pea, "rel": rel, "worst": worst}
        bitAll = sum(rec[p]["bit"] for p in POOLS and ["K", "V"]) \
            if False else sum(rec["K"]["bit"]) + sum(rec["V"]["bit"])
        peaMin = min(rec["K"]["pea"] + rec["V"]["pea"])
        relMax = max(rec["K"]["rel"] + rec["V"]["rel"])
        worst = max((rec[p]["worst"] for p in POOLS), key=lambda x: x["rel"])
        ck(f"seq={seq} 重算槽 ULP: Pearson min={peaMin:.6f}, |Δ|max={relMax*100:.2f}%"
           f" @ {worst['pool']} (≤5% 属正常)",
           peaMin >= 0.9998 and relMax <= 0.05)

        # D 独有 decode 段
        ds = {"tok": max(0, wD - wP), "healthy": True}
        if wD > wP:
            for p in POOLS:
                for i in range(nL):
                    if not bool(torch.isfinite(rD[p][i][wP:].float()).all()):
                        ds["healthy"] = False
        ck(f"seq={seq} D 独有 decode 段 {ds['tok']} tok 健康(无 NaN/Inf)",
           ds["healthy"])

        pairs_ind.append({"seq": seq, "pTok": pTok, "wTokP": wP, "wTokD": wD,
                          "grid": grid_ind, "bitAll": bitAll, "peaMin": peaMin,
                          "relMax": relMax, "worst": worst, "ds": ds,
                          "verdict": "PASS" if tx_bad == 0 and not grid_eq_bad else "FAIL"})

    # ---- viz_data 对账 ----
    mp1d = MAN["scenarios"]["p1d"]
    ck("manifest pairs 数与独立重算数一致",
       len(mp1d["pairs"]) == len(pairs_ind))
    for ind, m in zip(pairs_ind, mp1d["pairs"]):
        tag = f"req{ind['seq']}"
        ck(f"{tag} manifest pTok/wTokP/wTokD/verdict 一致",
           m["pTok"] == ind["pTok"] and m["wTokP"] == ind["wTokP"]
           and m["wTokD"] == ind["wTokD"] and m["verdict"] == ind["verdict"])
        tv = m["txLayerVerdict"]
        arr_flat = [bool(tv[p][i]) for p in POOLS for i in range(len(tv["K"]))]
        ck(f"{tag} manifest 整层 Tx 布尔 64/64 全 true", all(arr_flat)
           and len(arr_flat) == 64)
        gbad = 0
        for gi, gm in zip(ind["grid"], m["grid"]):
            if gi["kind"] == "donly":
                if gm.get("kind") != "donly" or gm.get("rows") != gi["rows"]:
                    gbad += 1
                continue
            if (gm.get("rowsTx") != gi["rowsTx"]
                    or gm.get("rowsRe") != gi["rowsRe"]
                    or gm.get("rowsDonly") != gi["rowsDonly"]):
                gbad += 1
                continue
            for p in POOLS:
                mk_arr = gm["eqK"] if p == "K" else gm["eqV"]
                if mk_arr is None or [bool(x) for x in mk_arr] != gi["eq"][p]:
                    gbad += 1
        ck(f"{tag} manifest 块级区间划分/等值布尔逐块全对齐", gbad == 0, f"bad={gbad}")
        r = m["recomp"]
        ck(f"{tag} manifest 重算槽统计对齐(位翻转 {ind['bitAll']}, worst {ind['worst']['pool']})",
           r["bitFlipsTotal"] == ind["bitAll"] and r["worst"]["pool"] == ind["worst"]["pool"]
           and close(r["pearsonMin"], ind["peaMin"], 1e-5)
           and close(r["relDeltaMax"], ind["relMax"], 1e-5))
        ck(f"{tag} manifest decode 段对齐(tok={ind['ds']['tok']})",
           m["decodeSeg"]["tok"] == ind["ds"]["tok"]
           and bool(m["decodeSeg"]["healthy"]) == ind["ds"]["healthy"])
        # 热图字节对账(P/D 两侧首块 × K/V × 32 层)
        hbad, hdiffs = 0, []
        for side_key in ("p", "d"):
            fj = file_json(m[side_key]["href"])
            bd = pt_by[ind["seq"]]["P" if side_key == "p" else "D"]
            blk0 = int(m[side_key]["blockTable"][0])
            for p in POOLS:
                diff = float(np.abs(heat_array(fj, p, str(blk0))
                                    - heat_expect(bd, p, blk0)).max())
                hdiffs.append(diff)
                if diff > 1e-6:
                    hbad += 1
        ck(f"{tag} viz_data 侧文件热图字节对齐(P/D 首块×K/V×32 层)",
           hbad == 0, f"bad={hbad}, maxdiff={max(hdiffs):.3e}")
    return mp1d


# ================================================================ C. matrix
TOOK_RE = re.compile(r"took ([\d.]+) ms")


def last_line(text, tag):
    for ln in reversed(text.splitlines()):
        if f"[PCM] {tag} " in ln:
            return ln
    return None


def qnum(text, key):
    ms = re.findall(key + r"=([\d.]+)", text + " ")
    return ms[-1] if ms else None


def section_matrix():
    sep("C. kvc_1p1d_prefix 四象限 —— PCM 日志独立重解(本实验无 tensors 归档)")
    lroot = BASE / "kvc_1p1d_prefix" / "logs"
    mm = MAN["scenarios"]["matrix"]
    qdir_by = {q["name"]: q for q in mm["quadrants"]}
    ck("viz_data 象限数 = 4", len(mm["quadrants"]) == 4)
    all_law_ok = True
    for qdir in sorted([x for x in lroot.iterdir()
                        if x.is_dir() and re.match(r"q\d+_p\dd\d", x.name)]):
        name = qdir.name
        p_pc = (qdir / "p_pcm.txt").read_text(errors="ignore")
        d_pc = (qdir / "d_pcm.txt").read_text(errors="ignore")
        cfg_p = last_line(p_pc, "CFG")
        cfg_d = last_line(d_pc, "CFG")
        sch_p = last_line(p_pc, "SCHED")
        sch_d = last_line(d_pc, "SCHED")
        alloc = last_line(d_pc, "ALLOC")
        xend = last_line(d_pc, "XFER-end")
        pfin = last_line(p_pc, "PFINISH")
        d_tr_txt = (qdir / "d_transfer.txt").read_text(errors="ignore")
        took = re.findall(r"took ([\d.]+) ms", d_tr_txt)

        pOn = "enable_prefix_caching=True" in cfg_p
        dOn = "enable_prefix_caching=True" in cfg_d
        pHit = int(qnum(sch_p, "local_hit"))
        dHit = int(qnum(sch_d, "local_hit"))
        promptR = int(qnum(sch_p, "prompt"))
        recvTok = int(qnum(alloc, "external"))
        mib = round(int(qnum(xend, "bytes")) / 1048576.0, 1)
        repN = int(re.search(r"report_blocks=\[(\d+)\]", pfin).group(1))
        rj = json.loads((qdir / "resp_r.json").read_text())
        genTok = int(rj["usage"]["completion_tokens"])

        laws = [
            ("A P本地命中", pHit, 256 if pOn else 0),
            ("B1 D本地命中", dHit, 256 if dOn else 0),
            ("B2 传输MiB", mib, 32.0 if dOn else 64.0),
            ("C P上报块数", repN, 4),
            ("D 输出tokens", genTok, 35),
        ]
        qok = all(g == w for _, g, w in laws)
        all_law_ok &= qok
        bad = "; ".join(f"{n}: got {g}, want {w}" for n, g, w in laws if g != w)
        ck(f"{name} 铁律独立复核 5/5 (P{'on' if pOn else 'off'}/D{'on' if dOn else 'off'}: "
           f"算 {promptR - pHit} tok / 传 {mib} MiB / D驻留 {dHit})",
           qok, bad)
        m = qdir_by.get(name, {})
        sync = (m.get("pOn") == pOn and m.get("dOn") == dOn
                and m.get("pHit") == pHit and m.get("dHit") == dHit
                and m.get("mib") == mib and m.get("reportBlocksN") == repN
                and m.get("genTok") == genTok and m.get("recvTok") == recvTok
                and took and abs(float(m.get("tookMs", -1)) - float(took[-1])) < 1e-9)
        ck(f"{name} viz_data 象限字段与日志独立重解一致", bool(sync))
    lawN = sum(1 for q in mm["quadrants"] for L in q["laws"] if L["pass"])
    ck("viz_data 铁律核验 20/20 全 PASS(与官方 matrix_report 口径一致)",
       lawN == 20 and all_law_ok, f"viz_data pass={lawN}/20")
    print("  [note] 该实验为日志驱动(README: PCM 六打点即证据), 无 tensors 归档;")
    print("         传输正确性维度由 B 节 kvc_1p1d 的 Tx 逐位核验背书。")


# ================================================================ D. kvcFlow
def section_flow():
    sep("D. kvc 全流程(kvcFlow) —— 状态机回放快照 vs .pt ground truth 对账")
    mf = MAN["scenarios"].get("kvcFlow")
    if not mf:
        ck("manifest 含 kvcFlow", False, "缺 scenarios.kvcFlow")
        return
    steps = mf["runtime"]["steps"]
    ck("manifest 含 kvcFlow(runtime steps=%d, 期望 17)" % len(steps),
       len(steps) == 17)
    ck("空闲队列守恒: 首末步 free 均 13290",
       steps[0]["state"]["free"] == steps[-1]["state"]["free"] == 13290)
    arcP = next(x for x in steps if x["req"] == "P" and x["phase"] == "archive")
    arcR = next(x for x in steps if x["req"] == "R" and x["phase"] == "archive")
    for tag, arc, ptdir in (("P", arcP, "req1_8e22e700"),
                            ("R", arcR, "req2_8d83db92")):
        b = load_pt(BASE / "kvc" / "tensors" / ptdir / "kv_pp0tp0.pt")
        bt = [int(x) for x in plain(b["meta"]["block_table"])]
        cv = [int(x) for x in plain(b["meta"]["cov"])]
        bs = {x["id"]: x["cov"] for x in arc["state"]["blocks"]}
        ck(f"{tag} 归档步块表/cov 与 .pt meta 逐位一致(块表 {bt})",
           [bs[i] for i in bt] == cv,
           f"flow={[bs.get(i) for i in bt]} vs pt={cv}")
    ck("R block_table 演进到位 [1, 2, 4, 5, 6]",
       arcR["state"]["btab"]["R"] == [1, 2, 4, 5, 6])
    ck("哈希表终态 3 项 blk[1,2,4](P 种 h1/h2 + R 满块 h3)",
       [(m["blk"], m["by"]) for m in steps[-1]["state"]["map"]]
       == [(1, "P"), (2, "P"), (4, "R")])
    ts = next(x for x in steps if x["phase"] == "touch")
    ax = next(x for x in steps if x["req"] == "R" and x["phase"] == "alloc"
              and x["title"].endswith("队头出队)"))
    ck("touch 后 free=13288 / R 首次 popleft 后=13286(与日志剩余数一致)",
       ts["state"]["free"] == 13288 and ax["state"]["free"] == 13286)
    ck("startup 编排数字(num_blocks=13291 · page=256.0KB/层块)",
       mf["startup"]["numBlocks"] == 13291
       and float(mf["startup"]["pageKb"]) == 256.0)
    print("  [note] kvcFlow 由 kvc/logs/patchs/{kvc_startup,kvc_p,kvc_r}.log 回放导出,")
    print("         本节把回放末态与 .pt meta 对账 —— 与 A 节 .pt 同源双路互证。")


# ================================================================ main
def main():
    t0 = time.time()
    nj = len(list((ROOT / "viz_data").rglob("*.json")))
    print(f"独立核验开始 | viz_data 生成于 {MAN.get('generated')} | 伴随 JSON {nj} 个")
    section_kvc()
    section_p1d()
    section_matrix()
    section_flow()
    sep(f"汇总: PASS {NP_[0]} / FAIL {NF_[0]}  |  耗时 {time.time() - t0:.1f}s")
    if NF_[0]:
        print("存在 FAIL —— 可视化结论与 tensors 事实不一致, 需排查!")
        return 1
    print("全部通过 —— 三个实验的 tensors/日志独立重算结果与 viz.html 数据完全一致:")
    print("  · A. kvc 前缀复用零篡改(req1 种块 vs req2 命中块 256/256 逐位相等)")
    print("  · B. 1P1D P→D 传输无损(Tx 区 2 对 × 64 层池 torch.equal 全等)")
    print("  · C. 四象限铁律 5×4 全 PASS(算多少/传多少/上报恒全量/输出全等)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
