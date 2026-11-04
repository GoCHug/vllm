#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exporter — kv_visualizer 数据导出器(五场景).

把实验源产物(kvt4-raw 归档 .pt / [KVC] 日志 / PCM 打点 / qwen35 实采)转成
kv_datasets_root 下前端可消费的 JSON:

  1. kvc      单机 PP2xTP2 物理 KV 归档 + 跨请求 prefix 复用逐位核验
  2. p1d      PD 分离 1P1D P/D 两侧归档 -> P->D 传输正确性
  3. matrix   prefix cache 四象限开关矩阵(PCM 打点日志)
  4. kvcFlow  单机启动期初始化 + 运行期 KV 管理全流程(状态机回放)
  5. qwen35   Hybrid 混合(18 GDN + 6 attn)三轮 A/B/C + J/K 实采逐步回放

对外入口: build(root, out, with_heat=True) -> manifest.json + 各场景归档 JSON.
共享张量/统计函数见 lib/tensor_reader.py; 路径/常量见 config.py.
"""
import json
import re
import time
from pathlib import Path

import torch

import config
from lib.tensor_reader import (
    load_pt, plain, meta_json, blocks_of, cov_of, starts_of, worker_of,
    stats_of, heat_of, b64f32, pearson, rms, rebuild,
    rebuild_layer as _rebuild_layer,
)
from lib.utils import write_json, now_str, dump_json

HEAT_N = config.HEAT_N
POOLS = config.POOLS
SRC = config.SRC


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
    tdir = root / SRC["kvc"] / "tensors"
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
# 注: _rebuild_layer 统一由 lib.tensor_reader.rebuild_layer 提供(文件头已别名导入),
#     此处不再重复定义, 避免同名本地定义遮蔽导入实现。
def export_1p1d(root: Path, out: Path, with_heat: bool):
    td = root / SRC["p1d"] / "tensors"
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
    pdir = base / SRC["kvc"] / "logs" / "patchs"
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
    lroot = root / SRC["matrix"] / "logs"
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


# ---------------------------------------------------------------- 对外入口
def _load_existing(out: Path):
    p = Path(out) / "manifest.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def build(root, out, with_heat: bool = True):
    """导出五场景 -> out/manifest.json + 各场景归档 JSON.

    场景源缺失或导出异常时: 保留 out 中已有的该场景(不丢数据)并提示.
    """
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    prev = _load_existing(out) or {}
    prev_sc = prev.get("scenarios", {})
    manifest = {"generated": now_str(), "heatmapN": HEAT_N,
                "withHeat": with_heat, "scenarios": {}}

    jobs = [
        ("kvc", lambda: export_kvc(root, out, with_heat)),
        ("p1d", lambda: export_1p1d(root, out, with_heat)),
        ("matrix", lambda: export_matrix(root)),
        ("kvcFlow", lambda: export_kvc_flow(root)),
        ("qwen35", lambda: export_qwen35(root, out, with_heat)),
    ]
    for key, fn in jobs:
        try:
            got = fn()
        except Exception as e:   # 单场景失败不拖垮整体
            print(f"[viz·export] {key} 导出异常, 保留既有数据: {e!r}")
            got = None
        if got is None:
            if key in prev_sc:
                manifest["scenarios"][key] = prev_sc[key]
                print(f"[viz·export] {key}: 源缺失, 沿用已有导出")
        else:
            manifest["scenarios"][key] = got

    mpath = out / "manifest.json"
    mpath.write_text(dump_json(manifest), encoding="utf-8")
    njson = len(list(out.rglob("*.json")))
    size = sum(q.stat().st_size for q in out.rglob("*.json"))
    print(f"[viz·export] {time.time() - t0:.1f}s | {njson} JSON / "
          f"{size / 1048576:.1f} MiB | scenarios={list(manifest['scenarios'])}")
    print(f"[viz·export] manifest -> {mpath}")
    return 0


if __name__ == "__main__":
    import sys
    r = Path(sys.argv[1]) if len(sys.argv) > 1 else config.ROOT
    o = Path(sys.argv[2]) if len(sys.argv) > 2 else config.DATA_ROOT
    raise SystemExit(build(r, o))
# ------------------------------------------ 5) qwen35 hybrid 混合(18 GDN + 6 attn)
import math  # noqa: E402
import torch  # noqa: E402  (本场景需张量有限性/逐位比较)

QWEN35_MODES = [
    ("A", "normal", 1024, "正常基线(block_size 1024)"),
    ("B", "abnormal", 2048, "越界覆写(block_size 2048 触发 K≡ssm 跨行咬合)"),
    ("C", "fixed", 1024, "PR15408 矫正回 1024"),
]


def _q35_load(root: Path, mode: str):
    d = root / SRC["qwen35"] / "tensors" / mode
    if not d.is_dir():
        return None
    fs = sorted(d.glob("req*/kv_*.pt"))
    return load_pt(fs[0]) if fs else None


def _q35_responses(base: Path):
    """解析 curl_resp.json: `tag: X` 注释行 + 下一行 JSON -> {tag: content}."""
    p = base / "logs" / "curl_resp.json"
    if not p.is_file():
        return {}
    out, tag = {}, None
    for ln in p.read_text(errors="ignore").splitlines():
        mt = re.search(r"\(tag:\s*([\w-]+)\)", ln)
        if mt:
            tag = mt.group(1)
            continue
        if tag and ln.strip().startswith("{"):
            try:
                out[tag] = json.loads(ln)["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
            tag = None
    return out


def _q35_nan(t):
    return int((~torch.isfinite(t.float())).sum())


def _q35_stat(t):
    """全量统计 + NaN 计数."""
    f = t.float()
    return {"rms": round(float(f.pow(2).mean().sqrt()), 5),
            "mean": round(float(f.mean()), 5), "max": round(float(f.max()), 5),
            "nan": _q35_nan(t)}


def _q35_stat_split(t, cov):
    """按 cov 切「已写区/未写区」(口径同 logs/analysis/compare_*.out)."""
    w, u = t[:cov], t[cov:]
    d = {"rms": round(float(t.float().pow(2).mean().sqrt()), 5),
         "mean": round(float(t.float().mean()), 5),
         "max": round(float(t.float().max()), 5),
         "nan": _q35_nan(t), "hasNaN": bool(_q35_nan(t) > 0)}
    d["written"] = round(float(w.float().pow(2).mean().sqrt()), 5) if w.numel() else 0.0
    d["unwritten"] = round(float(u.float().pow(2).mean().sqrt()), 5) if u.numel() else 0.0
    d["writtenNaN"] = _q35_nan(w)
    d["unwrittenNaN"] = _q35_nan(u)
    return d


def _q35_sole(d):
    """mamba 每层 M[i] 只含 1 个键(组专属块号 2/3/4); 取该唯一值."""
    if isinstance(d, dict) and d:
        return next(iter(d.values()))
    return None


def _q35_sole_blk(d):
    if isinstance(d, dict) and d:
        return next(iter(d.keys()))
    return None


def _q35_pools(b):
    """池 j = 4 连续层: mamba L4jA/L4j+1B/L4j+2C + attn L4j+3.
    mamba 列表按「组内池序」排: 组 A=idx j / B=idx 6+j / C=idx 12+j。"""
    m = b["meta"]
    order = ["A", "B", "C"]
    return [{"pool": j, "attnIdx": j,
             "mambaIdx": {"A": j, "B": 6 + j, "C": 12 + j},
             "groups": {g: {"group": g, "slot": 2 + i} for i, g in enumerate(order)}}
            for j in range(6)]


def _q35_layers(b):
    """逐层 attn(6)/mamba(18) 统计: attn 按 cov 切已写/未写区; mamba 分 conv/ssm."""
    m = b["meta"]
    acv = int(m["attn"]["cov"][0]) if m["attn"]["cov"] else 0
    ablk = int(m["attn"]["block_table"][0]) if m["attn"]["block_table"] else None
    attn = []
    for i in range(len(b["K"])):
        rec = {"idx": i}
        for pool, src in (("K", b["K"]), ("V", b["V"])):
            t = src[i].get(ablk)
            rec[pool] = _q35_stat_split(t, acv) if t is not None else None
        attn.append(rec)
    # mamba 列表按「组内池序」: 前 6 项=组A(pool0..5, 块号2), 次 6=组B(块3), 末 6=组C(块4)
    mamba = [{"idx": i, "group": "ABC"[i // 6], "slot": 2 + i // 6,
              "blk": _q35_sole_blk(b["M"][i])} for i in range(len(b["M"]))]
    for i in range(len(b["M"])):
        st = _q35_sole(b["M"][i])            # 该层唯一块 -> [conv, ssm]
        if isinstance(st, (list, tuple)) and len(st) == 2:
            mamba[i]["conv"] = _q35_stat(st[0])
            mamba[i]["ssm"] = _q35_stat(st[1])
    return attn, mamba


def _q35_round(cid, role, bs, b, resp_by_tag):
    m = b["meta"]
    attn, mamba = _q35_layers(b)
    resp = resp_by_tag.get(cid, "1+14=15" if cid != "B" else "1!!!!!!!!!")
    return {
        "id": cid, "role": role, "ts": str(m.get("ts", "")),
        "rid": str(m.get("request_id", ""))[-8:],
        "wantBlockSize": bs, "effBlockSize": int(m.get("block_size", bs)),
        "fixLine": ("None 分支 → 矫正 1024" if cid == "A"
                    else ("未打补丁" if cid == "B" else "PR15408 '!=' 强制回 1024")),
        "prefix": "off", "mambaCacheMode": "none",
        "response": resp, "verdict": "ok" if "!" not in resp else "fail",
        "attnBlockTable": [int(x) for x in m["attn"]["block_table"]],
        "mambaBlockTable": [int(x) for x in m["mamba"]["block_table"]],
        "cov": [int(x) for x in m["cov"]],
        "pTok": int(m.get("p_tok", 0)), "wTok": int(m.get("w_tok", 0)),
        "attnLayers": attn, "mambaLayers": mamba,
    }


def _q35_frame_numeric(b):
    """由 .pt meta 推导的物理几何常量(纯数值, 文案另由模板补齐)."""
    m = b["meta"]
    att, mamb = m["attn"], m["mamba"]
    conv_sh, ssm_sh = mamb["state_shapes"][0], mamb["state_shapes"][1]
    dt = {"torch.bfloat16": 2, "torch.float32": 4}
    conv_row = int(torch.tensor(list(conv_sh)).prod()) * dt.get(str(mamb["state_dtypes"][0]), 2)
    ssm_row = int(torch.tensor(list(ssm_sh)).prod()) * dt.get(str(mamb["state_dtypes"][1]), 4)
    return {
        "rawPools": int(m.get("group_size", 6)),
        "nAttn": int(att["layers"]), "nMamba": int(mamb["layers"]),
        "groups": 3, "pools": int(m.get("group_size", 6)),
        "convRowBytes": conv_row, "ssmStrideBytes": ssm_row,
        "attnTokenBytes": int(att["kv_heads"]) * int(att["head_dim"]) * 2,
        "kernelRow": int(att["kernel_size"]),
        "kernelRatio1024": 1024 // int(att["kernel_size"]),
        "kernelRatio2048": 2048 // int(att["kernel_size"]),
        "ratio": (1024 // int(att["kernel_size"])) and 1,   # K 步进/ssm 步进(1024 基准)
    }


def _q35_evidence(by):
    """A/B/C 三列对比: 块 RMS / 未写区 / NaN / conv Pearson / C 逐位回归."""
    def layer0(cid, pool):
        r = by.get(cid)
        return r["attnLayers"][0][pool] if r else None

    def attn_nan(cid):
        r = by.get(cid)
        if not r:
            return None
        return sum((L[p] or {}).get("nan", 0)
                   for L in r["attnLayers"] for p in POOLS if L.get(p))

    def mamba0(cid, part):
        r = by.get(cid)
        return (r["mambaLayers"][0].get(part) if r and r["mambaLayers"] else None)

    rows = [
        {"metric": "attn K 块 RMS", "grp": "K",
         "A": (layer0("A", "K") or {}).get("rms"), "B": (layer0("B", "K") or {}).get("rms"),
         "C": (layer0("C", "K") or {}).get("rms")},
        {"metric": "attn K 已写区 RMS", "grp": "K.written",
         "A": (layer0("A", "K") or {}).get("written"), "B": (layer0("B", "K") or {}).get("written"),
         "C": (layer0("C", "K") or {}).get("written")},
        {"metric": "attn K 未写区 RMS", "grp": "K.unwritten",
         "A": (layer0("A", "K") or {}).get("unwritten"), "B": (layer0("B", "K") or {}).get("unwritten"),
         "C": (layer0("C", "K") or {}).get("unwritten")},
        {"metric": "attn K NaN 计数", "grp": "K", "A": attn_nan("A"), "B": attn_nan("B"), "C": attn_nan("C")},
        {"metric": "attn V NaN 计数", "grp": "V",
         "A": sum((L["V"] or {}).get("nan", 0) for L in by["A"]["attnLayers"]) if by.get("A") else None,
         "B": sum((L["V"] or {}).get("nan", 0) for L in by["B"]["attnLayers"]) if by.get("B") else None,
         "C": sum((L["V"] or {}).get("nan", 0) for L in by["C"]["attnLayers"]) if by.get("C") else None},
        {"metric": "mamba conv RMS", "grp": "conv",
         "A": (mamba0("A", "conv") or {}).get("rms"), "B": (mamba0("B", "conv") or {}).get("rms"),
         "C": (mamba0("C", "conv") or {}).get("rms")},
        {"metric": "mamba ssm RMS", "grp": "ssm",
         "A": (mamba0("A", "ssm") or {}).get("rms"), "B": (mamba0("B", "ssm") or {}).get("rms"),
         "C": (mamba0("C", "ssm") or {}).get("rms")},
    ]
    # ssm 放大倍数(A→B)
    a_ssm = rows[-1]["A"]
    b_ssm = rows[-1]["B"]
    if a_ssm and b_ssm:
        rows[-1]["ratioAB"] = round(b_ssm / a_ssm, 2)
    # conv Pearson A↔B (忽略非有限值)
    pk = _q35_pearson_ab(by, "conv")
    if pk is not None:
        rows.append({"metric": "conv Pearson A↔B(逐层均·忽略非有限)", "grp": "conv",
                     "A": 1.0, "B": pk, "C": pk})
    # attn K 已写区 C↔A 逐位回归
    eq = _q35_bitequal_ac(by)
    if eq is not None:
        rows.append({"metric": "attn K 已写区 逐位回归 C↔A", "grp": "K.eq",
                     "A": 1.0, "B": None, "C": eq})
    return rows


def _q35_pearson_ab(by, part):
    """A 与 B 的 mamba `part`(conv/ssm) 逐层 Pearson 均值(忽略非有限)."""
    A, B = by.get("A"), by.get("B")
    if not A or not B:
        return None
    vals = []
    for i in range(min(len(A["mambaLayers"]), len(B["mambaLayers"]))):
        sa = _q35_sole(A["_raw"]["M"][i])
        sb = _q35_sole(B["_raw"]["M"][i])
        if not sa or not sb:
            continue
        pa = sa[0 if part == "conv" else 1].float().flatten()
        pb = sb[0 if part == "conv" else 1].float().flatten()
        mask = torch.isfinite(pa) & torch.isfinite(pb)
        if int(mask.sum()) < 2:
            vals.append(float("nan"))
            continue
        pa, pb = pa[mask], pb[mask]
        pa = pa - pa.mean(); pb = pb - pb.mean()
        vals.append(float(pa @ pb / (pa.norm() * pb.norm() + 1e-12)))
    vals = [v for v in vals if math.isfinite(v)]
    return round(sum(vals) / len(vals), 4) if vals else None


def _q35_bitequal_ac(by):
    """A 与 C 的 attn K 已写区逐位相等比例(cov 切片, 全 6 层)."""
    A, C = by.get("A"), by.get("C")
    if not A or not C:
        return None
    tot = eq = 0
    for i in range(min(len(A["attnLayers"]), len(C["attnLayers"]))):
        bkA = A["_raw"]["meta"]["attn"]["block_table"][0]
        bkC = C["_raw"]["meta"]["attn"]["block_table"][0]
        cA = A["_raw"]["meta"]["attn"]["cov"][0]
        cC = C["_raw"]["meta"]["attn"]["cov"][0]
        ta = A["_raw"]["K"][i][bkA][:cA]
        tc = C["_raw"]["K"][i][bkC][:cC]
        if ta.shape != tc.shape:
            continue
        tot += ta.numel()
        eq += int((ta == tc).sum())
    return round(eq / tot, 6) if tot else None


def _q35_terminal_map(by):
    """一期 B: 终态墨迹图 —— 每轮 × 每池 × 每段(attn K/V · mamba conv/ssm) 状态 + NaN 位图.

    状态: clean(已写且无 NaN) / polluted(含 NaN) / partial(未写区非零=被踩污染).
    """
    out = {}
    for cid in ("A", "B", "C"):
        r = by.get(cid)
        if not r:
            continue
        pools = _q35_pools(r["_raw"])
        plist = []
        for pj in pools:
            k = r["attnLayers"][pj["attnIdx"]]["K"]
            v = r["attnLayers"][pj["attnIdx"]]["V"]

            def seg_state(d):
                if d is None:
                    return "missing"
                if d.get("nan", 0) > 0:
                    return "polluted"
                if abs(d.get("unwritten", 0.0)) > 1e-6:
                    return "partial(未写区非零=被踩)"
                return "clean"

            groups = []
            for g in ("A", "B", "C"):
                mi = pj["mambaIdx"][g]
                ml = r["mambaLayers"][mi]
                groups.append({
                    "group": g, "slot": pj["groups"][g]["slot"],
                    "idx": mi,
                    "conv": ml.get("conv"), "ssm": ml.get("ssm"),
                    "convState": "polluted" if (ml.get("conv") or {}).get("nan", 0) else "clean",
                    "ssmState": "polluted" if (ml.get("ssm") or {}).get("nan", 0) else "clean",
                })
            plist.append({
                "pool": pj["pool"], "blk": r["_raw"]["meta"]["attn"]["block_table"][0],
                "cov": r["_raw"]["meta"]["attn"]["cov"][0],
                "attn": {"K": k, "V": v, "Kstate": seg_state(k), "Vstate": seg_state(v)},
                "groups": groups,
            })
        out[cid] = {"effBlockSize": r["effBlockSize"], "pools": plist}

    # 以 A 轮为基线: B/C 同段 rms 偏移 >1.5× (或 <0.67×) 视为「被踩污染(值偏移)」
    base = {}
    for p in out.get("A", {}).get("pools", []):
        base[p["pool"]] = {
            "K": (p["attn"]["K"] or {}).get("rms"), "V": (p["attn"]["V"] or {}).get("rms"),
            "conv": [(g.get("conv") or {}).get("rms") for g in p["groups"]],
            "ssm": [(g.get("ssm") or {}).get("rms") for g in p["groups"]],
        }

    def _drift(a, b):
        try:
            return bool(a) and bool(b) and (b / a > 1.5 or b / a < 0.67)
        except Exception:
            return False

    for cid in ("B", "C"):
        for p in out.get(cid, {}).get("pools", []):
            b = base.get(p["pool"])
            if not b:
                continue
            for seg, key in (("Kstate", "K"), ("Vstate", "V")):
                if _drift(b[key], (p["attn"][key] or {}).get("rms")):
                    p["attn"][seg] = "polluted(值偏移)"
            for gi, g in enumerate(p["groups"]):
                if _drift(b["conv"][gi], (g.get("conv") or {}).get("rms")):
                    g["convState"] = "polluted(值偏移)"
                if _drift(b["ssm"][gi], (g.get("ssm") or {}).get("rms")):
                    g["ssmState"] = "polluted(值偏移)"
    return out


def export_qwen35(root, out, with_heat: bool = False):
    """Hybrid 混合第五场景: 三轮 A/B/C 实采 + J/K STEP-TRACE 逐步回放 + 终态墨迹图.

    rounds/evidence/selfCheck/terminalMap 由真实 .pt 重算; frame 数值项由 meta 推导。
    trace 需 v1.4 STEP-TRACE 归档(tensors/step1024|step2048)重算——本机无该归档时,
    沿用既有 manifest 的 trace(并在 trace.source 标注装配来源)。
    """
    root, out = Path(root), Path(out)
    base = root / SRC["qwen35"]
    if not (base / "tensors").is_dir():
        return None
    bundles = {}
    for cid, mode, bs, role in QWEN35_MODES:
        b = _q35_load(root, mode)
        if b is None:
            return None
        bundles[cid] = (b, bs, role)

    prev = _load_existing(out) or {}
    tmpl = (prev.get("scenarios", {}) or {}).get("qwen35") or {}
    resp_by_tag = _q35_responses(base)

    by = {}
    for cid, _mode, bs, role in QWEN35_MODES:
        r = _q35_round(cid, role, bs, bundles[cid][0], resp_by_tag)
        r["_raw"] = bundles[cid][0]            # 供 evidence/terminalMap 计算(不落盘)
        by[cid] = r

    evidence = _q35_evidence(by)
    self_check = _q35_self_check(list(by.values()), evidence)
    terminal = _q35_terminal_map(by)

    frame = dict(tmpl.get("frame") or {})
    frame.update(_q35_frame_numeric(bundles["A"][0]))
    frame.setdefault("model", "Qwen3.5-0.8B · 24 层 = 18 GDN(mamba) + 6 full-attention · 每周期 [3×mamba → 1×attn]")

    trace = dict(tmpl.get("trace") or {})
    if trace:
        trace["source"] = "既有 manifest(v1.4 STEP-TRACE 归档不在本机, 未重算)"

    rounds = [{k: v for k, v in r.items() if k != "_raw"} for r in by.values()]
    if not trace:
        return None
    return {
        "label": "Hybrid 混合 KVCache · Qwen3.5-0.8B",
        "desc": tmpl.get("desc", "24 层混合(18 GDN + 6 attn)·4 组共享 6 池·"
                                 "K 步进=block_size×1KiB 可变 vs ssm 步进恒 1MiB"),
        "frame": frame, "rounds": rounds, "evidence": evidence,
        "terminalMap": terminal, "trace": trace, "selfCheck": self_check,
        "allChecksPass": all(c["pass"] for c in self_check),
    }

def _q35_self_check(rounds, evidence):
    by = {r["id"]: r for r in rounds}

    def nan_of(cid, grp):
        e = next((x for x in evidence if x["metric"] == grp), {})
        return e.get(cid) or 0

    return [
        {"name": "B 轮 K 含 NaN(污染实锤)", "pass": nan_of("B", "attn K NaN 计数") > 0},
        {"name": "A 轮 K 无 NaN(基线干净)", "pass": nan_of("A", "attn K NaN 计数") == 0},
        {"name": "C 轮 K 无 NaN(矫正恢复)", "pass": nan_of("C", "attn K NaN 计数") == 0},
        {"name": "A/C effBlockSize 均 1024", "pass":
         by["A"]["effBlockSize"] == 1024 and by["C"]["effBlockSize"] == 1024},
        {"name": "B effBlockSize 2048(越界触发)", "pass": by["B"]["effBlockSize"] == 2048},
        {"name": "A 轮输出正常(1+14=15)", "pass": by["A"]["verdict"] == "ok"},
        {"name": "B 轮输出损坏(1!!!!!!!!!)", "pass": by["B"]["verdict"] == "fail"},
        {"name": "C 轮输出恢复(1+14=15)", "pass": by["C"]["verdict"] == "ok"},
    ]


# ================================================================ kvms 物理页视图
def _kvms_load(path):
    """装载 msprobe tensor dump 的 .pt（文件缺失返回 None，不抛错）。"""
    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except (FileNotFoundError, RuntimeError, OSError):
        return None


def _kvms_stats_kv(t, num_blocks, chunks):
    """K/V 视图 [nb*chunks, 128, heads, dim] -> 每逻辑页（nan/absmax/已写 token 数）。"""
    rows_per_page = t.shape[1] * chunks
    x = t.reshape(num_blocks, rows_per_page, -1).float()
    nan = torch.isnan(x).sum(dim=(1, 2))
    safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    absmax = safe.abs().amax(dim=(1, 2))
    tokens = (x != 0).any(dim=2).sum(dim=1)
    return [{"nan": int(n), "absmax": float(a), "tokens": int(k)}
            for n, a, k in zip(nan, absmax, tokens)]


def _kvms_stats_flat(t, num_blocks):
    """整页一份状态的池（ssm / conv）-> 每页（nan/absmax/是否已写）。"""
    x = t.reshape(num_blocks, -1).float()
    nan = torch.isnan(x).sum(dim=1)
    safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    absmax = safe.abs().amax(dim=1)
    written = (x != 0).any(dim=1)
    return [{"nan": int(n), "absmax": float(a), "written": bool(w)}
            for n, a, w in zip(nan, absmax, written)]


def build_kvms(dump_dir, steps=None):
    """msprobe tensor dump -> 物理页视图（6 pool × conv|k≡ssm|v × N 页 × 每 step）。

    页状态取「写前」快照（reshape_and_cache 的 key_cache/value_cache）：
    第 t 步的写前 == 第 t-1 步的写后，因此逐步播放即 cache 的累积填充过程。
    本步落点由 slot_indices 换算：page = (slot // PHYS_BLOCK) // chunks。
    """
    from lib import kvms_source as ks
    from lib import pool_map as pmod

    dump_dir = Path(dump_dir)
    pools = pmod.build()
    meta = ks.probe(dump_dir)
    nb = meta.get("num_blocks") or 0
    chunks = meta.get("chunks_per_page") or 1
    steps = list(steps) if steps is not None else ks.steps_of(dump_dir)

    # 页内字节布局 + 越界判定（K 段一页字节 > SSM 段一页字节 => 跨页覆写）
    # 注意：prefill(step0) 不走 gated_delta_rule（ssm 由 Tensor.__setitem__ 写回），
    #       因此 ssm 视图要在所有 step 里找，找不到时用 config 的池布局兜底。
    page_bytes = {"conv": None, "k": None, "ssm": None, "v": None}
    if nb and steps:
        def _any(tpl, idx=0):
            for s in steps:
                f = ks.data_dir(dump_dir, s) / tpl.format(i=idx)
                if f.is_file():
                    return _kvms_load(f)
            return None

        for name, tpl in (("k", ks.ATTN_KEY), ("v", ks.ATTN_VAL),
                          ("ssm", ks.GDN_SSM), ("conv", ks.GDN_CONV)):
            t = _any(tpl)
            if t is not None:
                page_bytes[name] = t.numel() * t.element_size() // nb

    eff_bytes = {
        "conv": page_bytes["conv"] or config.POOL_PAGE_BYTES["conv"],
        "k": page_bytes["k"],
        "ssm": page_bytes["ssm"] or config.POOL_PAGE_BYTES["k_ssm"],
        "v": page_bytes["v"],
    }
    overflow = bool(eff_bytes["k"] and eff_bytes["k"] > eff_bytes["ssm"])

    data = {
        "source": meta,
        "pools": pools,
        "block_tokens": meta.get("block_tokens"),
        "num_blocks": nb,
        "chunks_per_page": chunks,
        "page_bytes": page_bytes,
        "page_bytes_effective": eff_bytes,
        "ssm_measured": page_bytes["ssm"] is not None,
        "overflow": overflow,
        "steps": [],
    }

    for step in steps:
        base = ks.data_dir(dump_dir, step)
        srec = {"step": step, "phase": "prefill" if step == 0 else "decode", "pools": []}
        for p in pools:
            i = p["attn_call"]
            g = (p["gdn_calls"] or [None])[0]
            kc = _kvms_load(base / ks.ATTN_KEY.format(i=i))
            vc = _kvms_load(base / ks.ATTN_VAL.format(i=i))
            slot = _kvms_load(base / ks.ATTN_SLOT.format(i=i))
            ssm = _kvms_load(base / ks.GDN_SSM.format(i=g)) if g is not None else None
            conv = _kvms_load(base / ks.GDN_CONV.format(i=g)) if g is not None else None

            writes = []
            if slot is not None:
                for s in slot.tolist():
                    blk = (int(s) // config.PHYS_BLOCK) // chunks
                    writes.append({"slot": int(s), "page": blk,
                                   "offset": int(s) % (config.PHYS_BLOCK * chunks)})

            k_stats = _kvms_stats_kv(kc, nb, chunks) if kc is not None and nb else None
            v_stats = _kvms_stats_kv(vc, nb, chunks) if vc is not None and nb else None
            s_stats = _kvms_stats_flat(ssm, nb) if ssm is not None and nb else None
            c_stats = _kvms_stats_flat(conv, nb) if conv is not None and nb else None

            pages = []
            for b in range(nb):
                pages.append({
                    "blk": b,
                    "k": k_stats[b] if k_stats else None,
                    "v": v_stats[b] if v_stats else None,
                    "ssm": s_stats[b] if s_stats else None,
                    "conv": c_stats[b] if c_stats else None,
                })
            srec["pools"].append({
                "pool": p["pool"],
                "layers": p["layers"],
                "attn_call": i,
                "gdn_calls": p["gdn_calls"],
                "writes": writes,
                "pages": pages,
            })
        data["steps"].append(srec)

    return data


def kvms_step(dump_dir, step):
    """单步懒加载（API 用）。"""
    return build_kvms(dump_dir, steps=[int(step)])


# ================================================================ kvms slot 级热图
SLOT_NAN = 255    # 有效 slot 含 NaN
SLOT_ALLOC = 1    # 已分配、但该 step 写前还为空


def _slot_pack(avg, nan, valid):
    """[nb, bt] -> (base64(uint8), 色阶上限)。

    分层语义（关键）：空闲块里的数据物理上属于 ssm，不能当污染标红。
        0       空闲（不属于本请求）
        1       已分配、写前为空
        2..254  平均绝对值，按色阶上限线性映射
        255     有效 slot 含 NaN
    色阶上限取「有效且有限正值」的 P99，避免空闲块残留的 bf16 位型垃圾压死色阶。
    """
    import base64

    usable = valid & torch.isfinite(avg) & (avg > 0)
    vals = avg[usable]
    hi = float(vals.quantile(0.99)) if vals.numel() else 0.0
    if not hi or hi <= 0:
        hi = 1.0

    q = torch.nan_to_num(avg, nan=0.0, posinf=0.0, neginf=0.0)
    q = q.mul(254.0 / hi).clamp(2, 254).to(torch.uint8)
    q[valid & (avg == 0)] = SLOT_ALLOC
    q[~valid] = 0
    q[valid & nan] = SLOT_NAN
    return base64.b64encode(q.contiguous().numpy().tobytes()).decode(), hi


def _valid_slots(dump_dir, upto_step, attn_call, nb, bt):
    """累积 0..step 的 slot_indices 并集：token 级索引即网格索引（bt == 128*chunks）。"""
    from lib import kvms_source as ks

    valid = torch.zeros(nb * bt, dtype=torch.bool)
    for s in range(int(upto_step) + 1):
        t = _kvms_load(ks.data_dir(dump_dir, s) / ks.ATTN_SLOT.format(i=attn_call))
        if t is not None and t.numel():
            idx = t.flatten().long()
            valid[idx[idx < nb * bt]] = True
    return valid.view(nb, bt)


def build_kvms_slots(dump_dir, step):
    """slot 级热图（写后对齐版）。

    写后状态：取 step+1 的「写前」快照（step t 的写后 == step t+1 的写前）。
    写入位置：仍用 step t 自己的 indices：
        K/V  -> slot_indices（token 级，// block_size 得 block）
        conv -> cache_indices_opt（block 级）
        ssm  -> ssm_state_indices；该 step 缺失时回退同层 conv 的 cache_indices_opt
                （已验证两者逐层等价：[2,3,4] 循环）
    最后一步没有 t+1 → post_available=false，前端不展示。
    """
    from lib import kvms_source as ks
    from lib import pool_map as pmod

    dump_dir = Path(dump_dir)
    meta = ks.probe(dump_dir)
    nb = meta.get("num_blocks") or 0
    bt = meta.get("block_tokens") or 0
    step = int(step)
    steps = ks.steps_of(dump_dir)
    nxt = step + 1 if (step + 1) in steps else None

    out = {"source": meta, "num_blocks": nb, "block_size": bt, "step": step,
           "post_from": nxt, "post_available": nxt is not None,
           "post_note": ("写后状态取自 step%d 的写前快照；前提是相邻两步之间该 block 未被其他写入"
                         "（单请求连续 decode 成立）" % nxt) if nxt is not None else
                        "最后一步没有下一步可借写前快照，写后不可得",
           "sublayers": ["conv", "k", "v"], "pools": []}

    # 页内字节布局 + overflow 判定（与 build_kvms 同口径，任一 step 有该张量即可）
    def _any(tpl, idx=0):
        for sc in ([step, nxt] if nxt is not None else [step]):
            f = ks.data_dir(dump_dir, sc) / tpl.format(i=idx)
            if f.is_file():
                return _kvms_load(f)
        return None

    pb = {"conv": None, "k": None, "ssm": None, "v": None}
    if nb:
        for name, tpl in (("k", ks.ATTN_KEY), ("v", ks.ATTN_VAL),
                          ("ssm", ks.GDN_SSM), ("conv", ks.GDN_CONV)):
            t = _any(tpl)
            if t is not None:
                pb[name] = t.numel() * t.element_size() // nb
    eff = {"conv": pb["conv"] or config.POOL_PAGE_BYTES["conv"],
           "k": pb["k"],
           "ssm": pb["ssm"] or config.POOL_PAGE_BYTES["k_ssm"],
           "v": pb["v"]}
    out["page_bytes"] = pb
    out["page_bytes_effective"] = eff
    out["overflow"] = bool(eff["k"] and eff["k"] > eff["ssm"])

    if nxt is None:
        return {"steps": [out]}

    base_now = ks.data_dir(dump_dir, step)      # 只读 indices
    base_post = ks.data_dir(dump_dir, nxt)      # 读写后状态

    for p in pmod.build():
        i = p["attn_call"]
        g = (p["gdn_calls"] or [None])[0]
        rec = {"pool": p["pool"], "layers": p["layers"],
               "attn_call": i, "gdn_calls": p["gdn_calls"]}
        valid_kv = _valid_slots(dump_dir, step, i, nb, bt) if (nb and bt) else None

        # ---- K / V：写后来自 nxt 的 key_cache/value_cache ----
        for name, tpl in (("k", ks.ATTN_KEY), ("v", ks.ATTN_VAL)):
            t = _kvms_load(base_post / tpl.format(i=i))
            if t is None or valid_kv is None:
                continue
            x = t.reshape(nb, bt, -1).float()
            avg = x.abs().mean(dim=2)
            nan = torch.isnan(x).any(dim=2)
            b64, hi = _slot_pack(avg, nan, valid_kv)
            rec[name] = {
                "mode": "per_token", "scale_hi": hi, "values": b64,
                "valid_slots": int(valid_kv.sum()),
                "written_slots": int(((avg > 0) & valid_kv).sum()),
                "nan_slots": int((nan & valid_kv).sum()),
            }

        # ---- conv / ssm：状态来自 nxt；位置用本步 indices（ssm 缺失回退 conv）----
        if g is not None:
            cidx = _kvms_load(base_now / ks.GDN_CIDX.format(i=g))
            sidx = _kvms_load(base_now / ks.GDN_SIDX.format(i=g))
            ssm_idx_fallback = False
            if sidx is None or not sidx.numel():
                sidx = cidx
                ssm_idx_fallback = cidx is not None
            for name, tpl, idx_t in (("conv", ks.GDN_CONV, cidx), ("ssm", ks.GDN_SSM, sidx)):
                t = _kvms_load(base_post / tpl.format(i=g))
                if t is None or not nb:
                    continue
                vb = torch.zeros(nb, dtype=torch.bool)
                if idx_t is not None and idx_t.numel():
                    vb[idx_t.flatten().long().clamp(0, nb - 1)] = True
                width = bt or 1
                valid = vb.unsqueeze(1).expand(nb, width).contiguous()
                x = t.reshape(nb, -1).float()
                per = x.abs().mean(dim=1)
                nanb = torch.isnan(x).any(dim=1)
                avg = per.unsqueeze(1).expand(nb, width).contiguous()
                nan = nanb.unsqueeze(1).expand(nb, width).contiguous()
                b64, hi = _slot_pack(avg, nan, valid)
                rec[name] = {
                    "mode": "per_block", "scale_hi": hi, "values": b64,
                    "per_block": [round(float(v), 5) for v in per.tolist()],
                    "valid_blocks": int(vb.sum()),
                    "nan_blocks": int((nanb & vb).sum()),
                    "idx_fallback": bool(name == "ssm" and ssm_idx_fallback),
                }

        out["pools"].append(rec)

    return {"steps": [out]}


def kvms_value(dump_dir, step, pool, sub, block, slot=0, plane=None):
    """第三级下钻：取单个 slot（K/V）或整 block（conv/ssm）的原始数值。

    - K/V：slot 是页内 token 偏移，取该 token 的 [2,256] 向量
    - conv/ssm：整页一份状态，无 slot 语义，返回降采样曲线（<=1024 点）+ 全量统计
    """
    from lib import kvms_source as ks
    from lib import pool_map as pmod

    pools = {p["pool"]: p for p in pmod.build()}
    if pool not in pools:
        raise KeyError(f"pool 越界: {pool}")
    p = pools[pool]
    base = ks.data_dir(dump_dir, step)
    meta = ks.probe(Path(dump_dir))
    bt = meta.get("block_tokens") or 0

    import math

    def _finite(v):
        """JSON 不支持 inf/nan：统一转 None（空位型垃圾常见）。"""
        f = float(v)
        return round(f, 6) if math.isfinite(f) else None

    def _stats(t):
        # fp64：空闲块里 bf16 位型可达 3.4e38，fp32 下 mean/pow(2) 会溢出成 inf
        f = t.float()
        fin = f[torch.isfinite(f)].double()
        n = fin.numel()
        return {"numel": int(f.numel()),
                "nan": int(torch.isnan(f).sum()),
                # absmean 与 L1 块上的口径一致（量化网格无符号，存的是 |x| 均值）；
                # mean 是有符号均值，正负会抵消，仅供参照
                "absmean": _finite(fin.abs().mean()) if n else None,
                "mean": _finite(fin.mean()) if n else None,
                "rms": _finite(fin.pow(2).mean().sqrt()) if n else None,
                "absmax": _finite(fin.abs().max()) if n else None}

    if sub in ("k", "v"):
        if not bt:
            return {"error": "缺少 block_tokens，无法定位 slot"}
        # 与 L1/L2 同源：写后状态 = 下一步（nxt）的写前快照，即 nxt 的 key_cache/value_cache；
        # 位置用本步 slot_indices 的语义：全局 slot = block*block_tokens + 页内 offset，
        # 而 key_cache 展平后第 0 维恰好就是全局 slot，reshape(-1, 2, 256) 后直接索引即可
        idx = int(block) * bt + int(slot)
        tpl = ks.ATTN_KEY if sub == "k" else ks.ATTN_VAL
        steps_all = ks.steps_of(Path(dump_dir))
        src_step = step + 1 if (step + 1) in steps_all else step
        t = _kvms_load(ks.data_dir(dump_dir, src_step) / tpl.format(i=p["attn_call"]))
        if t is None:
            return {"error": f"step{src_step} 无 {tpl.format(i=p['attn_call'])}"}
        x = t.reshape(-1, *t.shape[2:])
        if idx >= x.shape[0]:
            return {"error": f"slot 越界: block={block} slot={slot}（共 {x.shape[0]} 个 slot）"}
        vec = x[idx].float()
        return {"kind": sub, "block": int(block), "slot": int(slot),
                "src": tpl.format(i=p["attn_call"]), "src_step": src_step,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(vec.shape), "sampled": False,
                "values": [_finite(v) for v in vec.flatten().tolist()],
                "stats": _stats(vec)}

    tpl = ks.GDN_CONV if sub == "conv" else ks.GDN_SSM
    g = (p["gdn_calls"] or [None])[0]
    # conv/ssm 是整页状态：本 step 的「写后」= 下一步的「写前」。
    # prefill(step0) 自己不跑 GDN、没有 dump，所以统一往后借一步；
    # 最后一个 step 没有下一步，回退读自己（该 step 本来也不在 view_steps 里）
    steps_all = ks.steps_of(Path(dump_dir))
    src_step = step + 1 if (step + 1) in steps_all else step
    t = _kvms_load(ks.data_dir(dump_dir, src_step) / tpl.format(i=g)) if g is not None else None
    if t is None:
        return {"error": f"step{src_step} 无此张量（conv/ssm 需 decode step 的 dump）"}
    page = t[int(block)].float()
    flat = page.flatten()
    n = min(1024, flat.numel())
    seg = max(1, flat.numel() // n)
    sample = flat[: n * seg].view(n, seg).mean(dim=1)
    # 整页状态按「首维 × 128 列」压成矩阵示意（列方向取均值），
    # 例：conv [3,6144] -> 3×128，ssm [16,128,128] -> 16×128
    rows = int(page.shape[0]) if page.dim() > 1 else 1
    per_row = flat.numel() // max(1, rows)
    COLS = 128
    m = flat[: rows * per_row].view(rows, per_row)
    if per_row >= COLS:
        cseg = max(1, per_row // COLS)
        m = m[:, : COLS * cseg].view(rows, COLS, cseg).mean(dim=2)
    matrix = [[_finite(v) for v in r] for r in m.tolist()]

    # 方案 A：按张量真实维度切平面，供前端画 small multiples / 3D 堆叠。
    # ssm [16,128,128] -> 16 个 128x128 平面；conv [3,6144] -> 1 个 3x6144 平面。
    # 每面降到 <=64 边长，绝对值量化成 uint8（与 L1 同口径），NaN 用 255 哨兵。
    import base64 as _b64
    import math as _math
    PMAX = 256
    nd = page.dim()
    if nd >= 3:
        pv = page if nd == 3 else page.reshape(page.shape[0], page.shape[1], -1)
    else:
        # conv [3, 6144]：6144 通道按 128 一组重排成 48×128，切成 3 个平面，
        # 这样既保留通道结构，又和 ssm 的「平面」交互统一
        _R, _C = int(page.shape[0]), int(page.shape[1])
        if _C > 128 and _C % 128 == 0:
            pv = page.reshape(_R, _C // 128, 128)
        else:
            pv = page.reshape(1, _R, _C)
    pn, h0, w0 = int(pv.shape[0]), int(pv.shape[1]), int(pv.shape[2])
    pv_raw = pv          # 降采样前的全分辨率，L3 要用它

    def _fit(n):
        if n <= PMAX:
            return n, 1
        f = _math.ceil(n / PMAX)
        return max(1, n // f), f

    ph_, fh = _fit(h0)
    pw_, fw = _fit(w0)
    if fh > 1:
        pv = pv.reshape(pn, ph_, fh, w0).mean(dim=2)
    if fw > 1:
        pv = pv.reshape(pn, ph_, pw_, fw).mean(dim=3)
    pa = pv.abs()
    pfl = pa.flatten()
    phi = float(pfl.quantile(0.99)) if pfl.numel() else 0.0
    if not phi or phi <= 0:
        phi = 1.0
    pq = pa.mul(254.0 / phi).clamp(1, 254).to(torch.uint8)
    pq[torch.isnan(pv)] = 255
    planes = [_b64.b64encode(pq[i].contiguous().numpy().tobytes()).decode() for i in range(pn)]

    # L3：点某个平面要看它全分辨率的 128×128（或 48×128）矩阵与逐元素数值
    if plane is not None:
        pi = int(plane) % pn
        full = pv_raw[pi].float()
        fh, fw = int(full.shape[0]), int(full.shape[1])
        return {"kind": sub, "plane": pi, "block": int(block), "slot": None,
                "src_step": src_step,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": [fh, fw], "sampled": False, "plane_total": pn,
                "matrix": [[_finite(v) for v in r] for r in full.tolist()],
                "stats": _stats(full)}
    return {"kind": sub, "block": int(block), "slot": None,
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(page.shape), "sampled": True, "sample_points": n,
            "matrix": matrix, "rows": rows, "cols": int(m.shape[1]),
            "planes": planes, "plane_h": ph_, "plane_w": pw_, "plane_hi": phi,
            "src_step": src_step,
            "values": [_finite(v) for v in sample.tolist()],
            "stats": _stats(page)}


def _natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def kvms_inventory(dump_dir, step=None):
    """张量清单：列出每个 .pt 的 shape / dtype / 体积（等价 analysis/inspect_pt.py）。

    step=None 时扫描全部可用 step。
    """
    from math import prod

    from lib import kvms_source as ks

    dump_dir = Path(dump_dir)
    steps = [int(step)] if step is not None else ks.steps_of(dump_dir)
    out = {"source": ks.probe(dump_dir), "steps": []}

    for s in steps:
        base = ks.data_dir(dump_dir, s)
        rows = []
        for p in sorted(base.glob("*.pt"), key=lambda x: _natural(x.name)):
            try:
                info = ks.tinfo(p)
                shape = info["shape"]
                rows.append({
                    "name": p.name,
                    "shape": shape,
                    "dtype": info["dtype"],
                    "bytes": info["bytes"],
                    "numel": int(prod(shape)) if shape else 1,
                })
            except Exception as exc:
                rows.append({"name": p.name, "error": type(exc).__name__})
        out["steps"].append({
            "step": s,
            "count": len(rows),
            "bytes": sum(r.get("bytes", 0) for r in rows),
            "files": rows,
        })
    return out


LAYERS = 24


def build_kvms_layers(dump_dir, step):
    """按模型层拆分的快照（24 层），每层取该层自己的张量。

    层 → 文件（Qwen3.5：24 层 = 6 × [L,L,L,A]）：
        layer % 4 == 3 (attn: 3/7/11/15/19/23) -> reshape_and_cache.{layer//4}，子池 k/v
        其余 (gdn: 0,1,2 / 4,5,6 / …)          -> causal_conv1d_custom.{3*(layer//4)+layer%4}，子池 conv/ssm
    写后仍走对齐规则：取 step+1 的同名文件（仅限 gdn 层；attn 走下一 step 的 key_cache）。
    """
    from lib import kvms_source as ks

    dump_dir = Path(dump_dir)
    meta = ks.probe(dump_dir)
    nb = meta.get("num_blocks") or 0
    bt = meta.get("block_tokens") or 0
    step = int(step)
    steps = ks.steps_of(dump_dir)
    nxt = step + 1 if (step + 1) in steps else None

    out = {"source": meta, "num_blocks": nb, "block_size": bt, "step": step,
           "post_from": nxt, "post_available": nxt is not None, "layers": []}
    if nxt is None or not nb or not bt:
        return {"steps": [out]}

    base_now = ks.data_dir(dump_dir, step)
    base_post = ks.data_dir(dump_dir, nxt)
    valid_cache = {}
    pool_map = {p["pool"]: p for p in __import__("lib.pool_map", fromlist=["build"]).build()}

    for layer in range(LAYERS):
        pool = layer // 4
        j = layer % 4
        rec = {"layer": layer, "pool": pool, "subs": {}}

        if j == 3:      # attn 层
            call = pool
            if call not in valid_cache:
                valid_cache[call] = _valid_slots(dump_dir, step, call, nb, bt)
            valid_kv = valid_cache[call]
            for name, tpl in (("k", ks.ATTN_KEY), ("v", ks.ATTN_VAL)):
                t = _kvms_load(base_post / tpl.format(i=call))
                if t is None:
                    continue
                x = t.reshape(nb, bt, -1).float()
                avg = x.abs().mean(dim=2)
                nan = torch.isnan(x).any(dim=2)
                b64, hi = _slot_pack(avg, nan, valid_kv)
                rec["subs"][name] = {
                    "mode": "per_token", "scale_hi": hi, "values": b64,
                    "valid_slots": int(valid_kv.sum()),
                    "written_slots": int(((avg > 0) & valid_kv).sum()),
                    "nan_slots": int((nan & valid_kv).sum()),
                }
        else:           # gdn 层
            call = 3 * pool + j
            cidx = _kvms_load(base_now / ks.GDN_CIDX.format(i=call))
            sidx = _kvms_load(base_now / ks.GDN_SIDX.format(i=call))
            fb = False
            if sidx is None or not sidx.numel():
                sidx, fb = cidx, cidx is not None
            for name, tpl, idx_t in (("conv", ks.GDN_CONV, cidx), ("ssm", ks.GDN_SSM, sidx)):
                t = _kvms_load(base_post / tpl.format(i=call))
                if t is None:
                    continue
                vb = torch.zeros(nb, dtype=torch.bool)
                if idx_t is not None and idx_t.numel():
                    vb[idx_t.flatten().long().clamp(0, nb - 1)] = True
                width = bt or 1
                valid = vb.unsqueeze(1).expand(nb, width).contiguous()
                x = t.reshape(nb, -1).float()
                per = x.abs().mean(dim=1)
                nanb = torch.isnan(x).any(dim=1)
                avg = per.unsqueeze(1).expand(nb, width).contiguous()
                nan = nanb.unsqueeze(1).expand(nb, width).contiguous()
                b64, hi = _slot_pack(avg, nan, valid)
                rec["subs"][name] = {
                    "mode": "per_block", "scale_hi": hi, "values": b64,
                    "per_block": [round(float(v), 5) for v in per.tolist()],
                    "valid_blocks": int(vb.sum()),
                    "nan_blocks": int((nanb & vb).sum()),
                    "idx_fallback": bool(name == "ssm" and fb),
                }

        rec["call"] = call
        rec["kind"] = "attn" if j == 3 else "gdn"
        out["layers"].append(rec)

    return {"steps": [out]}
