#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viz_export.py — kvcache_docs_v2 可视化数据导出器.

把三个工作区的实验产物转换成 viz.html 可消费的 JSON:

  1. kvc/            单机 PP2×TP2 物理 KV 归档(.pt)
       -> 每归档文件: 统计(rms/mean/min/max 逐层逐块) + 32×32 下采样热图(base64 f32)
       -> manifest: 跨请求 prefix 复用逐位核验(同 worker 同层同块 torch.equal)
  2. kvc_1p1d/       PD 分离(1P1D) P/D 两侧归档
       -> 每归档文件同上; manifest: seq 配对 P→D 传输正确性
          (Tx 区逐层逐块 torch.equal + 重算槽 ULP 统计 + D 独有 decode 段健康)
  3. kvc_1p1d_prefix/  prefix cache 四象限 PCM 打点日志
       -> manifest: 四象限解析(CFG/SCHED/ALLOC/PFINISH/XFER 打点 + 铁律核验)

用法(路径均按脚本位置解析, 全程推荐在 viz/ 目录下运行):
  cd kvcache_docs_v2/viz
  python3 viz_export.py                   # 重新生成 ./viz_data/
  python3 viz_export.py --no-heat         # 跳过热图(快速跑通, 只出统计/核验)
然后: python3 -m http.server 8000  ->  http://localhost:8000/viz.html
(默认 --root = 脚本上级 kvcache_docs_v2, 即 kvc/kvc_1p1d/kvc_1p1d_prefix 所在;
 --out 默认 = 脚本同级 viz_data/)
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
HEAT_N = 32   # 热图边长(块张量头0通道 128×128 做 4×4 平均池化 -> 32×32)
POOLS = ("K", "V")


# ---------------------------------------------------------------- 基础工具
def load_pt(path):
    b = torch.load(str(path), map_location="cpu", weights_only=False)
    for k in ("K", "V", "meta"):
        if k not in b:
            raise ValueError(f"非 kvt4-raw 归档(缺 {k}): {path}")
    return b


def plain(v):
    if torch.is_tensor(v):
        v = v.tolist()
    return v


def meta_json(m):
    return {str(k): plain(v) for k, v in m.items()}


def blocks_of(bundle):
    return [int(b) for b in bundle["meta"]["block_table"]]


def cov_of(bundle):
    return [int(c) for c in bundle["meta"]["cov"]]


def starts_of(cov):
    s, starts = 0, []
    for c in cov:
        starts.append(s)
        s += int(c)
    return starts


def worker_of(meta):
    return f"pp{plain(meta.get('pp', 0))}tp{plain(meta.get('tp', 0))}"


def stats_of(t):
    f = t.float()
    return {"rms": float(f.pow(2).mean().sqrt()), "mean": float(f.mean()),
            "min": float(f.min()), "max": float(f.max())}


def heat_of(pools):
    """每层一张 [128,128](头0通道) -> 4×4 平均池化 -> 32×32; 返回 [nL,32,32]."""
    hs = []
    for t in pools:
        x = t[0:128, 0, 0:128].float()
        hs.append(x.view(HEAT_N, 4, HEAT_N, 4).mean(dim=(1, 3)))
    return torch.stack(hs)


def b64f32(t):
    return base64.b64encode(t.contiguous().numpy().tobytes()).decode()


def pearson(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def write_json(base: Path, rel: Path, obj):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p.stat().st_size


def export_bundle_json(bundle, with_heat: bool):
    """kvt4-raw bundle -> 文件级 JSON(统计+热图+meta), 供 viz.html 懒加载."""
    m, K, V = bundle["meta"], bundle["K"], bundle["V"]
    nL = len(K)
    blocks = blocks_of(bundle)
    stats = {p: {} for p in POOLS}
    heat = {p: {} for p in POOLS} if with_heat else None
    for p, pool in zip(POOLS, (K, V)):
        if len(pool) != nL:
            raise ValueError("K/V 层数不一致")
        for blk in blocks:
            st = {"rms": [], "mean": [], "min": [], "max": []}
            for i in range(nL):
                if blk not in pool[i]:
                    raise ValueError(f"层{i} 缺块 {blk}")
                s = stats_of(pool[i][blk])
                for kk in st:
                    st[kk].append(round(s[kk], 5))
            stats[p][str(blk)] = st
            if with_heat:
                heat[p][str(blk)] = b64f32(
                    heat_of([pool[i][blk] for i in range(nL)]))
    out = {"meta": meta_json(m), "stats": stats,
           "nLayers": nL, "heatmapN": HEAT_N,
           "blockTable": blocks, "cov": cov_of(bundle)}
    if with_heat:
        out["heat"] = heat
    return out


# ---------------------------------------------------------------- 1) kvc 单机
def export_kvc(root: Path, out: Path, with_heat: bool):
    tdir = root / "kvc" / "tensors"
    if not tdir.is_dir():
        return None
    bundles = {}   # reqdir -> {worker: {"path", "bundle"}}
    requests = []
    for d in sorted([x for x in tdir.iterdir()
                     if x.is_dir() and x.name.startswith("req")]):
        files = {}
        for f in sorted(d.glob("kv_*.pt")):
            b = load_pt(f)
            files[worker_of(b["meta"])] = {"path": f, "bundle": b}
        if not files:
            continue
        bundles[d.name] = files
        first = next(iter(files.values()))["bundle"]
        m = first["meta"]
        bt, cv = blocks_of(first), cov_of(first)
        mm = meta_json(m)
        requests.append({
            "name": d.name, "seq": int(plain(m.get("seq", 0)) or 0),
            "rid": d.name.split("_")[-1], "requestId": mm.get("request_id", ""),
            "ts": mm.get("ts", ""),
            "role": "",
            "blockTable": bt, "cov": cv, "blockStarts": starts_of(cv),
            "pTok": int(plain(m.get("p_tok", 0)) or 0),
            "wTok": int(plain(m.get("w_tok", sum(cv)))),
            "groupSize": int(plain(m.get("group_size", 0)) or 0),
            "kvHeads": int(plain(m.get("kv_heads", 0)) or 0),
            "headDim": int(plain(m.get("head_dim", 0)) or 0),
            "blockSize": int(plain(m.get("block_size", 0)) or 0),
            "model": "Meta-Llama-3-8B bf16 · PP2×TP2",
            "files": [],
        })
    # 逐请求逐 worker 落文件 JSON
    for req in requests:
        for w, fw in sorted(bundles[req["name"]].items()):
            rel = Path("kvc") / req["name"] / (fw["path"].stem + ".json")
            fj = export_bundle_json(fw["bundle"], with_heat)
            fj["worker"] = w
            size = write_json(out, rel, fj)
            req["files"].append({
                "worker": w, "pp": int(w[2]), "tp": int(w[5]),
                "href": str(rel), "nLayers": fj["nLayers"],
                "globalLayerBase": int(w[2]) * int(req["groupSize"] or 16),
                "size": size, "stats": fj["stats"],
            })
    requests.sort(key=lambda r: r["seq"])
    for i, r in enumerate(requests):
        r["role"] = ("P·种块(先发 324 tok)" if i == 0
                     else f"R·前缀命中(后发 {r['wTok']} tok, 复用 {requests[0]['name']})")

    # -- prefix 复用核验: 早请求种块 vs 晚请求命中共享表头块 --
    prefix = {"pairs": []}
    if len(requests) >= 2:
        early, late = requests[0], requests[1]
        shared, k = [], 0
        while (k < len(early["blockTable"]) and k < len(late["blockTable"])
               and early["blockTable"][k] == late["blockTable"][k]
               and min(early["cov"][k], late["cov"][k]) == 128):
            shared.append(early["blockTable"][k])
            k += 1
        if not shared:  # 退化: 块号交集保序
            sl = set(late["blockTable"])
            shared = [b for b in early["blockTable"] if b in sl]
        hit_tok = sum(min(early["cov"][i], late["cov"][i])
                      for i, b in enumerate(early["blockTable"][:k])) if k else 0
        checks, nfail = {}, 0
        for w in sorted(set(bundles[early["name"]]) & set(bundles[late["name"]])):
            be = bundles[early["name"]][w]["bundle"]
            bl = bundles[late["name"]][w]["bundle"]
            nL = len(be["K"])
            per = {p: {} for p in POOLS}
            nck = 0
            for p in POOLS:
                for blk in shared:
                    eqs = []
                    for i in range(nL):
                        te, tl = be[p][i].get(blk), bl[p][i].get(blk)
                        ok = (te is not None and tl is not None
                              and te.shape == tl.shape and bool(torch.equal(te, tl)))
                        eqs.append(ok)
                        nck += 1
                    per[p][str(blk)] = eqs
            nf = sum(0 if x else 1 for p in POOLS for arr
                     in per[p].values() for x in arr)
            nfail += nf
            checks[w] = {"K": per["K"], "V": per["V"],
                         "nLayers": nL, "nChecked": nck, "nFail": nf,
                         "pass": nf == 0}
        prefix["pairs"].append({
            "early": early["name"], "late": late["name"],
            "roleNote": "早请求(P)种块 -> 晚请求(R)命中共享表头块, 逐 worker × 逐层 × 逐池 torch.equal",
            "sharedBlocks": shared, "hitTokens": hit_tok,
            "checks": checks, "nFail": nfail, "overallPass": nfail == 0,
        })
    return {"requests": requests, "prefix": prefix,
            "label": "单机 PP2×TP2 · 物理 KV 归档 + 前缀复用核验",
            "desc": ("4 worker(pp∈{0,1}×tp∈{0,1}) 每层组 16 × kv_heads 4; "
                     "物理块整块快照(bf16), TERM 时归档")}


# ------------------------------------------------- 2) kvc_1p1d PD 分离
def _rebuild_layer(lay, bt, cov):
    """单层块字典 -> token-major [Σcov, heads, dim](顺序按块表, 前 cov 行)."""
    parts = []
    for e, blk in enumerate(bt):
        c = int(cov[e])
        if c > 0 and blk in lay:
            parts.append(lay[blk][:c])
    return torch.cat(parts, dim=0)


def export_1p1d(root: Path, out: Path, with_heat: bool):
    td = root / "kvc_1p1d" / "tensors"
    if not td.is_dir():
        return None
    pairs = []
    for d in sorted([x for x in td.iterdir()
                     if x.is_dir() and x.name.startswith("req")],
                    key=lambda x: int(x.name[3:].split("_")[0])):
        seq = int(d.name[3:].split("_")[0])
        sides = {}
        for side in ("P", "D"):
            fs = sorted((d / side).glob("kv_*.pt"))
            if fs:
                sides[side] = {"path": fs[0], "bundle": load_pt(fs[0])}
        if "P" not in sides or "D" not in sides:
            continue
        pb, db = sides["P"]["bundle"], sides["D"]["bundle"]
        mp, md = pb["meta"], db["meta"]
        pTok = int(plain(mp.get("p_tok", -1)) or -1)
        nL = len(pb["K"])

        def side_entry(bundle, side, wtok_meta):
            m = bundle["meta"]
            bt, cv = blocks_of(bundle), cov_of(bundle)
            rel = Path("p1d") / d.name / side / (
                sides[side]["path"].stem + ".json")
            fj = export_bundle_json(bundle, with_heat)
            fj["side"] = side
            write_json(out, rel, fj)
            return {"side": side, "href": str(rel),
                    "blockTable": bt, "cov": cv, "blockStarts": starts_of(cv),
                    "wTok": int(plain(m.get("w_tok", sum(cv)))),
                    "requestId": plain(m.get("request_id", "")),
                    "ts": plain(m.get("ts", "")),
                    "nLayers": nL, "kvHeads": int(plain(m.get("kv_heads", 0)) or 0),
                    "headDim": int(plain(m.get("head_dim", 0)) or 0),
                    "blockSize": int(plain(m.get("block_size", 0)) or 0),
                    "stats": fj["stats"], "size": (out / rel).stat().st_size}

        entry = {"seq": seq, "pTok": pTok,
                 "p": side_entry(pb, "P", "w_tok"), "d": side_entry(db, "D", "w_tok")}
        entry["wTokP"], entry["wTokD"] = entry["p"]["wTok"], entry["d"]["wTok"]
        btP, cvP = entry["p"]["blockTable"], entry["p"]["cov"]
        btD, cvD = entry["d"]["blockTable"], entry["d"]["cov"]
        stP, stD = entry["p"]["blockStarts"], entry["d"]["blockStarts"]

        # A) 结构对齐
        bad = []
        if plain(mp.get("side")) != "P" or plain(md.get("side")) != "D":
            bad.append(f"meta.side P/D = {plain(mp.get('side'))}/{plain(md.get('side'))}")
        if int(plain(md.get("p_tok", -2)) or -2) != pTok:
            bad.append(f"p_tok 不等 {pTok}/{plain(md.get('p_tok'))}")
        for k in ("layers", "kv_heads", "head_dim", "block_size", "dtype"):
            if str(plain(mp.get(k))) != str(plain(md.get(k))):
                bad.append(f"{k} 不等 {plain(mp.get(k))}/{plain(md.get(k))}")
        entry["struct"] = bad

        # B) Tx 区 [0, pTok-1): 逐层逐池 torch.equal(官方口径)
        rbP = {p: [_rebuild_layer(pb[p][i], btP, cvP) for i in range(nL)]
               for p in POOLS}
        rbD = {p: [_rebuild_layer(db[p][i], btD, cvD) for i in range(nL)]
               for p in POOLS}
        txLayer = {}
        for p in POOLS:
            txLayer[p] = [bool(torch.equal(rbP[p][i][:pTok - 1],
                                           rbD[p][i][:pTok - 1]))
                          for i in range(nL)]

        # 块级 Tx 等值网格(更细粒度: 行=层, 列=共同块)
        posP = {b: i for i, b in enumerate(btP)}
        grid = []
        for blk in btD:
            iD = btD.index(blk)
            if blk not in posP:
                grid.append({"blk": blk, "kind": "donly",
                             "rows": cvD[iD], "startD": stD[iD],
                             "eqK": None, "eqV": None})
                continue
            ip = posP[blk]
            cp, cd = cvP[ip], cvD[iD]
            rowsTx = max(0, min(cp, cd, (pTok - 1) - stP[ip]))
            eqK = [bool(torch.equal(pb["K"][i][blk][:rowsTx],
                                    db["K"][i][blk][:rowsTx]))
                   for i in range(nL)] if rowsTx > 0 else None
            eqV = [bool(torch.equal(pb["V"][i][blk][:rowsTx],
                                    db["V"][i][blk][:rowsTx]))
                   for i in range(nL)] if rowsTx > 0 else None
            # 重算槽: 全局 token pTok-1 落在本块已写区
            rowsRe = 1 if stP[ip] <= pTok - 1 < stP[ip] + min(cp, cd) else 0
            rowsD = max(0, cd - max(cp, rowsTx + rowsRe))   # D 独有行
            grid.append({"blk": blk, "kind": "shared",
                         "startP": stP[ip], "startD": stD[iD],
                         "rowsTx": rowsTx, "rowsRe": rowsRe, "rowsDonly": rowsD,
                         "eqK": eqK, "eqV": eqV})

        # C) 重算槽 [pTok-1, min(wP,wD)): ULP 级统计(正常)
        recomp = None
        if pTok - 1 < min(entry["wTokP"], entry["wTokD"]):
            rec = {}
            worst = {"rel": 0.0, "pool": "-"}
            for p in POOLS:
                bit, pea, rel = [], [], []
                for i in range(nL):
                    a = rbP[p][i][pTok - 1: pTok]
                    b = rbD[p][i][pTok - 1: pTok]
                    bit.append(int((a.view(torch.int16) != b.view(torch.int16)).sum()))
                    pea.append(round(pearson(a, b), 6))
                    denom = float(a.abs().max()) or 1e-9
                    rv = float((a.float() - b.float()).abs().max()) / denom
                    rel.append(round(rv, 6))
                    if rv > worst["rel"]:
                        worst = {"rel": round(rv, 6), "pool": f"L{i:02d}{p}"}
                rec[p] = {"bitFlips": bit, "pearson": pea, "relDelta": rel}
            relAll = rec["K"]["relDelta"] + rec["V"]["relDelta"]
            peaAll = rec["K"]["pearson"] + rec["V"]["pearson"]
            recomp = {
                "slot": f"[{pTok-1}, {min(entry['wTokP'], entry['wTokD'])})",
                "note": "P 哑 token 纯前向 vs D bootstrap 补算, ULP 级差属正常路径",
                "K": rec["K"], "V": rec["V"],
                "bitFlipsTotal": sum(rec["K"]["bitFlips"]) + sum(rec["V"]["bitFlips"]),
                "pearsonMin": min(peaAll), "relDeltaMax": max(relAll),
                "worst": worst,
                "consistentULP": max(relAll) <= 0.05,   # 参考判据 ≤5%
            }

        # D) D 独有 decode 段 [wP, wD): 健康检查
        wP, wD = entry["wTokP"], entry["wTokD"]
        decodeSeg = {"tok": max(0, wD - wP), "healthy": True}
        if wD > wP:
            rmsAll, badfin = [], False
            for p in POOLS:
                for i in range(nL):
                    t = rbD[p][i][wP:]
                    if not bool(torch.isfinite(t.float()).all()):
                        badfin = True
                    rmsAll.append(round(float(t.float().pow(2).mean().sqrt()), 4))
            decodeSeg["healthy"] = not badfin
            decodeSeg["rmsRange"] = [min(rmsAll), max(rmsAll)] if rmsAll else None

        tx_ok = all(all(v) for v in txLayer.values())
        for g in grid:
            if g["eqK"] is not None and not all(g["eqK"]):
                tx_ok = False
            if g["eqV"] is not None and not all(g["eqV"]):
                tx_ok = False
        entry.update({
            "verdict": "PASS" if (not bad and tx_ok) else "FAIL",
            "txLayerVerdict": txLayer, "grid": grid,
            "recomp": recomp, "decodeSeg": decodeSeg,
            "model": "Meta-Llama-3-8B bf16 · TP1 · mooncake 跨卡迁移",
        })
        pairs.append(entry)
    return {"pairs": pairs,
            "label": "PD 分离 1P1D · P→D KVCache 传输逐位核验",
            "desc": ("P(npu:0) --mooncake--> D(npu:1); Tx 区[0,pTok-1) 逐位 torch.equal "
                     "= 传输无损; 重算槽 ULP 差 = bootstrap 补算正常; D 独有段 = 本地 decode")}



# ------------------------------------------ 2.5) kvc 全流程(启动期+运行期)
def _kvc_lines(path):
    out = []
    for ln in path.read_text(errors="ignore").splitlines():
        m = re.search(r"INFO (\d\d-\d\d) (\d\d:\d\d:\d\d) \[[\w.]+\.py:\d+\] (.*)$", ln)
        if m:
            out.append({"t": m.group(2), "s": m.group(3)})
    return out


def export_kvc_flow(base: Path):
    """kvc 全流程: 启动期三侧(配置/物理/逻辑) + 运行期 P/R 状态机回放.

    输入: kvc/logs/patchs/{kvc_startup,kvc_p,kvc_r}.log 的 [KVC] 打点.
    输出: 页面零逻辑消费的 steps(每步附 哈希表/空闲队列/块池 快照),
    表尾自检与 .pt 归档的块表/cov 对账.
    """
    pdir = base / "kvc" / "logs" / "patchs"
    for f in ("kvc_startup.log", "kvc_p.log", "kvc_r.log"):
        if not (pdir / f).is_file():
            return None
    sup = _kvc_lines(pdir / "kvc_startup.log")
    whole = "\n".join(e["s"] for e in sup)

    def m1(pat):
        m = re.search(pat, whole)
        return m.group(1) if m else None

    def t_of(pat):
        for e in sup:
            if re.search(pat, e["s"]):
                return e["t"]
        return ""

    # ---- 启动期三侧时间轴(4 worker 重复行去重为 1 条) ----
    budget = m1(r"各 worker 可用 KV 显存 = \[(.*?)\]")
    num_blocks = int(m1(r"KVCacheConfig: num_blocks=(\d+)") or 0)
    page_kb = m1(r"page_size_bytes=\d+ \(([\d.]+)KB/层/块\)")
    tensor_mib = m1(r"KVCacheTensor: size=\d+ bytes \(([\d.]+)MiB\)")
    kshape = m1(r"K_cache shape=\(([\d, ]+)\)")
    maxlen = m1(r"max_model_len=(\d+)")
    side_events = [
        ("cfg", t_of(r"配置侧 KVCache 编排开始"), "配置侧编排开始",
         "core.py 驱动: get_kv_cache_config -> 物理分配 -> 逻辑初始化"),
        ("cfg", t_of(r"--- ①: 算规格 ---"), "① 算规格",
         "4 worker × 16×FullAttentionSpec(pp0=L0~L15 ·2TP / pp1=L16~L31 ·2TP)"),
        ("cfg", t_of(r"--- ②: 测预算 ---"), "② 测预算",
         f"profile 实测: 各 worker 可用 KV 显存 = [{budget}]"),
        ("cfg", t_of(r"--- ③: 做编排 ---"), "③ 做编排",
         f"num_blocks={num_blocks} · groups=1 · tensors=16 · page={page_kb}KB/层/块"
         f" · 每层张量 {tensor_mib}MiB ×16 层 ≈ {16 * float(tensor_mib):.0f}MiB/worker"),
        ("cfg", t_of(r"配置侧 KVCache 编排完成"), "配置侧编排完成",
         "生成 KVCacheConfig 下发物理侧与逻辑侧"),
        ("phy", t_of(r"物理侧 KV Cache 分配开始"), "物理侧分配开始(×4 worker)",
         "vllm-ascend model_runner 依 config 裸分配 16 张层张量"),
        ("phy", t_of(r"_reshape_kv_cache_tensors"), "K/V 池 reshape",
         f"每 worker K/V 分离两池: shape=({kshape}) bf16 ×16 层 ×4 worker · device=npu:0~3"),
        ("phy", t_of(r"物理侧 KV Cache 分配完成"), "物理侧分配完成(×4 worker)",
         f"{num_blocks} 块/worker · 新建池全零基线(未写槽位=0, 已由核验器 PASS)"),
        ("log", t_of(r"逻辑侧初始化开始"), "逻辑侧初始化开始", "五层装配链开启"),
        ("log", t_of(r"FreeKVCacheBlockQueue"), "FreeKVCacheBlockQueue",
         "num_free_blocks=13291 · 伪头尾哨兵 block_id=-1(双链表 LRU 序)"),
        ("log", t_of(r"BlockHashToBlockMap"), "BlockHashToBlockMap",
         "底容器 dict size=0 —— 满块哈希表(prefix cache): hash -> KVCacheBlock, append-only"),
        ("log", t_of(r"BlockPool.__init__"), "BlockPool",
         f"KVCacheBlock ×{num_blocks} (id 0..{num_blocks - 1}) · null_block=0 独占"
         " · hash_block_size=128 · free 队列实际可分配 13290"),
        ("log", t_of(r"FullAttentionManager.__init__"), "Manager / Coordinator / Manager 装配",
         f"block_size=128 · enable_caching=True · max_model_len={maxlen} · Unitary 单组直通"),
        ("log", t_of(r"逻辑侧初始化完成"), "逻辑侧初始化完成", "系统就绪, 等待请求"),
    ]

    # ---- 运行期: P/R 状态机回放 ----
    FREE0 = 13290
    free = FREE0
    mapp = []
    btab = {"P": [], "R": []}
    blocks = {0: {"id": 0, "req": None, "cov": None, "null": True}}
    steps = []

    def snap(hit_len=0, last=None, ids=None, hit_marks=()):
        blkset = {m["blk"] for m in mapp}
        return {"free": free,
                "map": [dict(m, hit=m["h"] in hit_marks) for m in mapp],
                "blocks": [dict(b, shared=(b["id"] in blkset))
                           for b in blocks.values()],
                "btab": {k: list(v) for k, v in btab.items()},
                "hitLen": hit_len,
                "queue": {"last": last, "ids": [str(x) for x in (ids or [])]}}

    def emit(req, phase, title, desc, text, st):
        steps.append({"req": req, "phase": phase, "title": title,
                      "desc": desc, "text": text, "state": st})

    emit(None, "init", "|就绪| 启动期末态",
         f"空闲队列 {FREE0}/{num_blocks}(null_block=0 独占) · 哈希表空 · 无已分配块",
         "—— 进入运行期 ——", snap())

    for name, fname in (("P", "kvc_p.log"), ("R", "kvc_r.log")):
        lines = _kvc_lines(pdir / fname)
        whole_r = "\n".join(e["s"] for e in lines)
        texts = [e["s"] for e in lines]

        def first_r(pat):
            return next((x for x in texts if re.search(pat, x)), "")

        # 一 · 入队: 满块链式哈希
        mr = re.search(r"Request\(request_id=(\S+?)\) 入队: num_prompt_tokens=(\d+)"
                       r".*?满块链式哈希 BlockHash × (\d+): \[(.*?)\]", whole_r)
        hrs = [x.strip("' ") for x in mr.group(4).split(",")]
        emit(name, "enq", f"{name}·入队(满块链式哈希)",
             f"prompt {mr.group(2)} tok · 哈希 ×{mr.group(3)}: "
             + " → ".join(h[:8] for h in hrs),
             first_r(r"Request\(request_id=\S+\) 入队: num_prompt_tokens"), snap())

        # 二 · 前缀查哈希表
        hits = []
        for x in texts:
            mh = re.search(r"第 (\d+) 块 (HIT|MISS): BlockHash=(\w+) -> (.*)", x)
            if mh:
                b = None
                if mh.group(2) == "HIT":
                    nb = re.findall(r"\[(\d+)\]", mh.group(4))
                    b = int(nb[0]) if nb else None
                hits.append({"h": mh.group(3), "b": b})
        mm = re.search(r"get_computed_blocks 返回: KVCacheBlocks\(blocks=\((.*?)\)\),"
                       r" num_computed_tokens=(\d+)", whole_r)
        hit_len = int(mm.group(2)) if mm else 0
        emit(name, "lookup", f"{name}·前缀查哈希表",
             " | ".join(f"#{i + 1} {'HIT→blk' + str(b['b']) if b['b'] is not None else 'MISS'}"
                        f"({b['h'][:8]})" for i, b in enumerate(hits))
             + f" → hit_length={hit_len}",
             "\n".join(x for x in texts
                        if "前缀查找 BlockHashToBlockMap.get_one_block" in x)[:400],
             snap(hit_len, "lookup", [b["h"] for b in hits],
                  hit_marks=[b["h"] for b in hits if b["b"] is not None]))

        # 三 · S2 touch 命中块(离队, ref+1)
        mt = re.search(r"S2 BlockPool.touch: blocks=\[(.*?)\]", whole_r)
        if mt and hit_len > 0:
            touched = [int(x) for x in re.findall(r"\((\d+),", mt.group(1))]
            free -= len(touched)
            btab[name].extend(touched)
            emit(name, "touch", f"{name}·S2 touch 命中块",
                 f"touch {touched}: ref_cnt+1 且从空闲队列出队 · free→{free}",
                 first_r(r"S2 BlockPool.touch"),
                 snap(hit_len, "touch", touched))

        # 四 · S3 popleft_n(可多次: prefill一次 + decode 跨块)
        for x in texts:
            mo = re.search(r"get_new_blocks\((\d+)\): popleft_n -> block_ids=\[(.*?)\]"
                           r", 剩余 num_free_blocks=(\d+)", x)
            if not mo:
                continue
            got = [int(v) for v in mo.group(2).replace(" ", "").split(",") if v]
            f0 = free
            free = int(mo.group(3))
            for b in got:
                blocks[b] = {"id": b, "req": name, "cov": None, "null": False}
                btab[name].append(b)
            if name == "R" and len(got) == 1:
                title = f"{name}·decode 跨块出队"
                desc = f"decode 续写跨块边界: popleft_n {got} · free {f0}→{free}"
            else:
                title = f"{name}·S3 新块分配(队头出队)"
                desc = f"popleft_n {got} ← 队头 · free {f0}→{free}"
            emit(name, "alloc", title, desc, x, snap(hit_len, "pop", got))

        # 五 · S4 满块入哈希表
        ins_texts = []
        n0 = len(mapp)
        for x in texts:
            mi = re.search(r"insert: key=BlockHashWithGroupId\(hash=(\w+), group_id=\d+\)"
                           r" <- KVCacheBlock\(block_id=(\d+)\), map size=(\d+)", x)
            if mi and not any(m["h"] == mi.group(1) for m in mapp):
                mapp.append({"h": mi.group(1), "blk": int(mi.group(2)), "by": name})
                ins_texts.append(x)
        ins_texts += [x for x in texts
                      if "S4 BlockPool.cache_full_blocks" in x and x not in ins_texts]
        if ins_texts:
            emit(name, "cache", f"{name}·S4 满块入哈希表",
                 f"map size {n0}→{len(mapp)}: "
                 + ", ".join(f"{m['h'][:8]}→blk{m['blk']}({m['by']})" for m in mapp),
                 "\n".join(ins_texts)[:600], snap(hit_len, "insert", []))

        # 六 · 块表对账(以日志最后一次 allocate_slots 返回为准)
        mb = [re.findall(r"当前完整 block_table=\(\[(.*?)\]\)", x) for x in texts]
        mb = [q for y in mb for q in y if q]
        if mb:
            btab[name] = [int(v) for v in mb[-1].replace(" ", "").split(",") if v]

        # 七 · TERM 归档(cov 真值)
        mv = re.search(r"开始保存物理tensor worker=PP0_TP0 .*blk=\[(.*?)\] cov=\[(.*?)\]",
                       whole_r)
        if mv:
            blks = [int(v) for v in mv.group(1).replace(" ", "").split(",")]
            covs = [int(v) for v in mv.group(2).replace(" ", "").split(",")]
            for b, c in zip(blks, covs):
                blocks[b]["cov"] = c
            emit(name, "archive", f"{name}·TERM 物理归档(4 worker)",
                 f"块表 {blks} · cov {covs}" + (" · 末块含未写槽位" if covs[-1] < 128 else "")
                 + " -> tensors/req*/kv_*.pt(张量浏览器数据源)",
                 first_r(r"开始保存物理tensor worker=PP0_TP0"), snap(hit_len))

        # 八 · 步末提交
        mj = re.search(r"提交 cache_blocks: req=\S+, num_computed_tokens=(\d+)", whole_r)
        emit(name, "commit", f"{name}·步末提交(异步输出路径)",
             f"num_computed_tokens={mj.group(1) if mj else '-'}",
             first_r(r"提交 cache_blocks"), snap(hit_len))

        # 九 · 释放归还
        mf = re.search(r"FreeKVCacheBlockQueue\.append_n\(blocks=\[(.*?)\]\),"
                       r" num_free_blocks=(\d+)", whole_r)
        if mf:
            back = [int(v) for v in mf.group(1).replace(" ", "").split(",") if v]
            free = int(mf.group(2))
            for b in back:
                blocks[b]["req"] = None
            btab[name] = []
            emit(name, "free", f"{name}·释放(逆序 append 队尾)",
                 f"append_n {back} -> 队尾(LRU 保护, ref_cnt 归零回收) · free→{free}"
                 " · 满块哈希表保留(种块可被后续请求继续命中)",
                 "\n".join(x for x in texts if "释放 FreeKVCacheBlockQueue" in x
                            or "释放前持有" in x)[:450],
                 snap(hit_len, "append", back))

    # ---- 导出层自检(与 .pt ground truth 对账) ----
    assert steps[0]["state"]["free"] == steps[-1]["state"]["free"] == FREE0, "free 首末不一致"
    arcP = next(x for x in steps if x["phase"] == "archive" and x["req"] == "P")
    gotP = [b["cov"] for b in arcP["state"]["blocks"] if b["id"] in (1, 2, 3)]
    assert gotP == [128, 128, 68], f"P 归档 cov 不符: {gotP}"
    arcR = next(x for x in steps if x["phase"] == "archive" and x["req"] == "R")
    gotR = [b["cov"] for b in arcR["state"]["blocks"] if b["id"] in (1, 2, 4, 5, 6)]
    assert gotR == [128, 128, 128, 128, 8], f"R 归档 cov 不符: {gotR}"
    assert arcR["state"]["btab"]["R"] == [1, 2, 4, 5, 6], "R 块表不符"

    return {
        "label": "单机 PP2×TP2 · 启动期初始化 + 运行期 KV 管理全流程",
        "desc": "启动三侧(配置/物理/逻辑)真实打点时间轴; 运行期 P/R 逐步入场:"
                " 入队链式哈希 -> 前缀查表 -> touch/队头出队 -> 满块入哈希表 ->"
                " 归档 -> 提交 -> 释放; 每步附哈希表/空闲队列/块池快照, 与 .pt 对账",
        "model": "Meta-Llama-3-8B bf16 · PP2×TP2 · 4×hpu910a3 · vLLM V1 · "
                 "block_size=128 · enforce-eager · seed=1024",
        "startup": {"timeline": [{"side": a, "t": b, "title": c, "detail": d}
                                  for a, b, c, d in side_events],
                    "numBlocks": num_blocks, "pageKb": page_kb,
                    "tensorMib": tensor_mib, "freeTotal": FREE0},
        "runtime": {"steps": steps, "freeTotal": FREE0,
                    "note": "free=空闲队列剩余(13291-null_block0); "
                            "touch=命中块出队; popleft=队头分配; append_n=逆序归还队尾"},
    }


# ------------------------------------------------- 3) prefix 四象限日志
PCM_RE = re.compile(r"\[(PCM)\] ([\w-]+)(.*)")
T_RE = re.compile(r"INFO (\d\d-\d\d) (\d\d:\d\d:\d\d)")
KV_RE = re.compile(r"(\w+)=([\w.,%\-\[\]()/]+)")


def parse_pcm_events(text, side):
    evs = []
    for line in text.splitlines():
        mm = PCM_RE.search(line)
        if not mm:
            continue
        tt = T_RE.search(line)
        evs.append({"t": tt.group(2) if tt else "", "side": side,
                    "tag": mm.group(2), "text": line.strip()})
    return evs


def last_by(events, side, tag):
    for e in reversed(events):
        if e["side"] == side and e["tag"] == tag:
            return e
    return None


def num_kv(e, key, cast=int, default=None):
    try:
        return cast(e["text"].split(key + "=")[1].split()[0].rstrip(",."))
    except Exception:
        return default


def pl(text, pattern, cast=str):
    ms = re.findall(pattern, text)
    return cast(ms[-1]) if ms else None


def export_matrix(root: Path):
    lroot = root / "kvc_1p1d_prefix" / "logs"
    if not lroot.is_dir():
        return None
    quads = []
    for qdir in sorted([x for x in lroot.iterdir()
                        if x.is_dir() and re.match(r"q\d+_p\dd\d", x.name)]):
        p_pc = (qdir / "p_pcm.txt").read_text(errors="ignore")
        d_pc = (qdir / "d_pcm.txt").read_text(errors="ignore")
        evs = parse_pcm_events(p_pc, "P") + parse_pcm_events(d_pc, "D")
        evs.sort(key=lambda e: e["t"])

        cfg_p = last_by(evs, "P", "CFG")
        cfg_d = last_by(evs, "D", "CFG")
        sch_p = last_by(evs, "P", "SCHED")
        sch_d = last_by(evs, "D", "SCHED")
        alloc = last_by(evs, "D", "ALLOC")
        xend = last_by(evs, "D", "XFER-end")
        pfin = last_by(evs, "P", "PFINISH")

        pOn = "enable_prefix_caching=True" in (cfg_p["text"] if cfg_p else "")
        dOn = "enable_prefix_caching=True" in (cfg_d["text"] if cfg_d else "")
        pHit = num_kv(sch_p, "local_hit") if sch_p else 0
        dHit = num_kv(sch_d, "local_hit") if sch_d else 0
        promptR = num_kv(sch_p, "prompt") if sch_p else 0
        recvTok = num_kv(alloc, "external") if alloc else 0
        recvBlocks = None
        m = re.search(r"recv_blocks=(\[\[.*?\]\])",
                      alloc["text"]) if alloc else None
        if m:
            recvBlocks = m.group(1)
        repN = None
        if pfin:
            mm = re.search(r"report_blocks=\[(.*?)\]", pfin["text"])
            if mm:
                # "[4]" = 1 组共 4 块(括号内数字即块数, 恒等 ceil(prompt/128))
                repN = sum(int(x) for x in re.findall(r"\d+", mm.group(1)))
        mib = seg = eff = None
        if xend:
            mib = round(int(num_kv(xend, "bytes", float)) / 1048576.0, 1)
            seg = num_kv(xend, "segments")
            eff = num_kv(xend, "eff_GBps", float)

        d_tr = (qdir / "d_transfer.txt").read_text(errors="ignore")
        tookMs = pl(d_tr, r"took ([\d.]+) ms", float)
        hitPctP = pl((qdir / "p_hitrate.txt").read_text(errors="ignore"),
                     r"Prefix cache hit rate: ([\d.]+)%", float)
        hitPctD = pl((qdir / "d_hitrate.txt").read_text(errors="ignore"),
                     r"Prefix cache hit rate: ([\d.]+)%", float)
        genTok = None
        try:
            rj = json.loads((qdir / "resp_r.json").read_text())
            u = rj.get("usage", {})
            genTok = int(u.get("completion_tokens", 0))
        except Exception:
            pass

        laws = [
            {"key": "A", "name": "P 本地命中 tok", "got": pHit,
             "want": 256 if pOn else 0},
            {"key": "B1", "name": "D 本地命中 tok", "got": dHit,
             "want": 256 if dOn else 0},
            {"key": "B2", "name": "P→D 传输量 MiB", "got": mib,
             "want": 32.0 if dOn else 64.0},
            {"key": "C", "name": "P 上报总块数(恒全量)", "got": repN, "want": 4},
            {"key": "D", "name": "输出 tokens", "got": genTok, "want": 35},
        ]
        for L in laws:
            L["pass"] = (L["got"] == L["want"])

        quads.append({
            "name": qdir.name, "pOn": pOn, "dOn": dOn,
            "promptR": promptR, "pHit": pHit, "dHit": dHit,
            "pRecompute": promptR - pHit,
            "recvTok": recvTok, "recvBlocks": recvBlocks,
            "mib": mib, "segments": seg, "effGbps": eff, "tookMs": tookMs,
            "hitPctP": hitPctP, "hitPctD": hitPctD,
            "genTok": genTok, "reportBlocksN": repN,
            "events": evs,
            "laws": laws, "allPass": all(L["pass"] for L in laws),
        })
    return {"quadrants": quads,
            "label": "1P1D Prefix Cache 四象限开关矩阵",
            "desc": "P 开关×D 开关 = 算多少(P prefill) / 传多少(P→D MiB) / 驻留多少(D 命中)"}


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT.parent))
    ap.add_argument("--out", default=str(ROOT / "viz_data"))
    ap.add_argument("--no-heat", action="store_true")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with_heat = not args.no_heat
    t0 = time.time()

    manifest = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "heatmapN": HEAT_N, "withHeat": with_heat, "scenarios": {}}

    print("[1/4] kvc(单机 PP2×TP2) ...")
    kvc = export_kvc(root, out, with_heat)
    if kvc:
        manifest["scenarios"]["kvc"] = kvc
        print(f"  requests={len(kvc['requests'])} "
              f"prefixPairs={len(kvc['prefix']['pairs'])} "
              f"prefixPass={kvc['prefix']['pairs'][0]['overallPass'] if kvc['prefix']['pairs'] else '-'}")

    print("[2/4] kvc_1p1d(PD 分离 P→D 传输正确性) ...")
    p1d = export_1p1d(root, out, with_heat)
    if p1d:
        manifest["scenarios"]["p1d"] = p1d
        print(f"  pairs={len(p1d['pairs'])} "
              f"verdicts={[e['verdict'] for e in p1d['pairs']]}")

    print("[3/4] kvc_1p1d_prefix(四象限矩阵) ...")
    print("[4/4] kvc 全流程(启动期+运行期状态机) ...")
    flow = export_kvc_flow(root)
    if flow:
        manifest["scenarios"]["kvcFlow"] = flow
        print(f"  startup events={len(flow['startup']['timeline'])} "
              f"runtime steps={len(flow['runtime']['steps'])}")

    mx = export_matrix(root)
    if mx:
        manifest["scenarios"]["matrix"] = mx
        print(f"  quadrants={len(mx['quadrants'])} "
              f"allPass={[q['allPass'] for q in mx['quadrants']]}")

    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    njson = len(list(out.rglob("*.json")))
    size = sum(p.stat().st_size for p in out.rglob("*.json"))
    print(f"[DONE] {time.time()-t0:.1f}s | {njson} 个 JSON / {size/1048576:.1f} MiB"
          f" | manifest -> {mpath}")
    print("下一步: cd " + str(ROOT) + " && python3 -m http.server 8000"
          "  ->  http://localhost:8000/viz.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
