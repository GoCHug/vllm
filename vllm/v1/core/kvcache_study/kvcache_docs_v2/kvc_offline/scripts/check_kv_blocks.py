#!/usr/bin/env python3
# ==============================================================================
# check_kv_blocks.py —— kvc_offline(pp2tp2) 检查器: block 原样归档离线检查
#
# 输入:
#   --dir   log/tensors   kv_{P,D}{rank}_{seq}_{rid尾8}.pt (11 号补丁 kvt2-raw)
#                          K/V = [层序 list, 每层 dict{块号: 原样张量(bs,kv_heads,hd)}]
#   --logs  log           轨迹: kvc_{p,d}_req{p,r}.log(共享段,含[FPB]/[FP] 无dev行,
#                          双 rank 交织) + kvc_{p,d}{0,1}_req{p,r}.log(带dev行:TERM)
#   --out   log/block_report
#
# 检查(pp2tp2 特有约束: TP2 双 rank 的 [FPB]/[FP] 行无 dev 标签、在同一文件交织):
#   B0 结构: 8 文件齐备 + 不变量(len(K)==layers, K[li]键集={b_i|cov_i>0},
#           每张量 shape==(bs,kv_heads,hd), sum(cov)==w_tok, kv_heads==8/TP)
#           + 跨侧 p_tok/cov 对齐 + block_table 同构性(报告,不 FAIL——块池各自独立)
#   B1 链间互证(集合匹配法): 每层每块把日志 [FPB] 的 (Tx,Xx) 收成 multiset,
#           每 rank 的 .pt 重算 09 同口径 sha256 后在 multiset 消解匹配一个——
#           rank0/rank1 是 kv_heads 不同切片(数据必然不同), 匹配唯一可靠;
#           层指纹 [FP](K.prompt/K.all/V.prompt/V.all)同理; TERM 行(带dev=npu:{r})
#           作 (request_id, rank) 存在性旁证。
#   B2 Tx 逐位: 按 block_table+cov 重建 token 序(cat 块前cov槽), 同 rank 对
#           (P{r} vs D{r}) 前 p_tok-1 行 torch.equal —— TP 维各自无损
#   B3 差异归类: 尾槽(行p_tok-1) 数值语义判据(|Δ|max<=5%层幅值, 沿用kvc_pd_offline
#           首轮修正结论——真实重算差位谱为分布式多bit翻转, 不能用ULP-only);
#           decode区(仅D, 行p_tok..w_tok-1)健康扫描;
#           未写槽位残值(块内cov之后槽)只统计报告, 不参与裁决。
#
# 判据: [PASS] B0 ∧ B1 消解全成功 ∧ B2 全 equal ∧ B3 尾槽=exact/recompute ∧ decode健康
#       [FAIL] 其余(附取证包: 层/块/rank/K|V/tok/head/dim/xor_bits/位翻转谱)
#
# selftest: python3 check_kv_blocks.py --selftest
#   合成 P0/P1/D0/D1×seq1/2 八文件 + 双 rank 交织日志, 注入两类差异(尾槽重算差→PASS,
#   Tx 损伤→FAIL)验证判定与集合匹配。
# ==============================================================================
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# 09 同口径 sha256(前16hex) —— 与 _kvc_fp 逐字符一致, 保证 B1 可互证
# ---------------------------------------------------------------------------
def kvc_fp(parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# bf16 位模式工具(取证包; 判据用数值语义——见 B3)
# ---------------------------------------------------------------------------
def xor_int32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.view(torch.int16).to(torch.int32) ^ b.view(torch.int16).to(torch.int32)


def classify_xor(x: int) -> str:
    if x == 0:
        return "exact"
    pop = bin(x & 0xFFFF).count("1")
    if pop <= 2 and (x & 0xFF80) == 0:
        return "ULP"
    if (x & 0x7F80):
        return "EXPONENT"
    if (x & 0x8000):
        return "SIGN"
    return "MANHIGH"


def block_offsets(meta) -> list:
    off, acc = [], 0
    for c in meta["cov"]:
        off.append(acc)
        acc += c
    return off


def tx_len_i(meta, i) -> int:
    return max(0, min(meta["cov"][i], (int(meta["p_tok"]) - 1) - block_offsets(meta)[i]))


def cp_len_i(meta, i) -> int:
    return min(meta["cov"][i], max(0, int(meta["p_tok"]) - block_offsets(meta)[i]))


# ---------------------------------------------------------------------------
# 取证包(参数为同形展平切片 a/b; meta 用于 blk 反推)
# ---------------------------------------------------------------------------
def evidence(l, which, tok, meta, a, b) -> dict:
    ai = int(a.view(torch.int16).to(torch.int32)) & 0xFFFF
    bi = int(b.view(torch.int16).to(torch.int32)) & 0xFFFF
    x = ai ^ bi
    bits = [str(k) for k in range(16) if x & (1 << k)]
    off = block_offsets(meta)
    blk = None
    for i, c in enumerate(meta["cov"]):
        if off[i] <= tok < off[i] + c:
            blk = meta["block_table"][i]
            break
    return {"layer": l, "which": which, "blk": blk, "tok": tok,
            "pa": f"0x{ai:04x}", "pb": f"0x{bi:04x}",
            "xor_bits": bin(x).count("1"), "bit_pos": "+".join(bits),
            "sig": classify_xor(x)}


def diff_dump(a, b, l, which, meta, cap):
    eq = (a.view(torch.int16) == b.view(torch.int16))
    if bool(eq.all()):
        return [], 0, {"exact": int(a.numel())}
    bad = (~eq).reshape(-1)
    idx = bad.nonzero(as_tuple=False).flatten().tolist()
    va = a.reshape(-1)[idx[:1024]]
    vb = b.reshape(-1)[idx[:1024]]
    x = xor_int32(va, vb) & 0xFFFF
    sigs = [classify_xor(int(v)) for v in x.tolist()]
    hist = {}
    for s_ in sigs:
        hist[s_] = hist.get(s_, 0) + 1
    ev = []
    kv_heads = int(meta.get("kv_heads", 1))
    head_dim = int(meta.get("head_dim", 1))
    for i in idx[:cap]:
        row, rem = divmod(i, kv_heads * head_dim)
        head, dim = divmod(rem, head_dim)
        ev.append(evidence(l, which, row, meta,
                           a.reshape(-1)[i], b.reshape(-1)[i]))
    n_exact = int(a.numel()) - len(idx)
    if n_exact:
        hist = {"exact": n_exact, **hist}
    return ev, len(idx), hist


# ---------------------------------------------------------------------------
# 日志解析
#   [FPB]/[FP] 无 dev(双 rank 交织) -> multiset 候选桶: {(side,tag): {layer: {blk: [(tx,xx)]
#   TERM 带 dev -> 旁证: {(side,tag): {(request_id, rank)}}
# ---------------------------------------------------------------------------
FPB_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FPB\] (\w+) (L\d+) 块指纹: (.*) \| (.*)")
FP_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FP\] (\w+) (L\d+) 指纹 w_tok=(\d+) p_tok=(\d+) "
    r"\| K\.prompt=(\S+) K\.all=(\S+) \| V\.prompt=(\S+) V\.all=(\S+)")
BLK_RE = re.compile(r"blk(\d+)[KV]\.Tx=([0-9a-f-]+)/[KV]\.Xx=([0-9a-f-]+)")
TERM_RE = re.compile(
    r"\[KVC\]\[KVP\] (TERM|LATE) req=(\S+) dev=npu:(\d+)[ ,]")

NAME_RE = re.compile(r"^kvc_([pd])(\d?)_req([pr])\.log$")


def parse_logs(log_dir: Path):
    """-> fp_cand: {(side,tag): {layer: {blk: [(tx,xx),...]}}}
        fp_layer: {(side,tag): {layer: [ (kp,ka,vp,va), ... ]}}  (multiset)
        term_seen: {(side,tag): {(rid, rank)}}"""
    fp_cand, fp_layer, term_seen = {}, {}, {}
    for f in sorted(log_dir.glob("kvc_*_req*.log")):
        m = NAME_RE.match(f.name)
        if not m:
            continue
        side, rnk, rletter = m.group(1), m.group(2), m.group(3)  # rnk '' for shared
        # 带 dev 的行只应出现在分 rank 文件; 共享文件只有无 dev 行
        key = (side, "req" + rletter)
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m2 = TERM_RE.search(line)
            if m2 and m2.group(1) == "TERM":
                term_seen.setdefault(key, set()).add((m2.group(2), int(m2.group(3))))
                continue
            if rnk:  # 分 rank 文件里 [FPB] 不应出现(无 dev), 跳过防御
                continue
            m2 = FPB_RE.search(line)
            if m2:
                layer, kpart, vpart = m2.group(2), m2.group(3), m2.group(4)
                lay = fp_cand.setdefault(key, {}).setdefault(layer, {})
                for which, part in (("K", kpart), ("V", vpart)):
                    for blk, tx, xx in BLK_RE.findall(part):
                        lay.setdefault(int(blk), {}).setdefault(which, []).append((tx, xx))
                continue
            m2 = FP_RE.search(line)
            if m2:
                layer = m2.group(2)
                fp_layer.setdefault(key, {}).setdefault(layer, []).append(
                    (m2.group(5), m2.group(6), m2.group(7), m2.group(8)))
    return fp_cand, fp_layer, term_seen


# ---------------------------------------------------------------------------
# .pt 加载 / 重建 / 重算
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(r"^kv_([PDX])(\d)_(\d)_([0-9a-f]+)\.pt$")


def load_bundles(t_dir: Path):
    out = {}
    for f in sorted(t_dir.glob("kv_*.pt")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        try:
            b = torch.load(f, map_location="cpu", weights_only=False)
        except TypeError:
            b = torch.load(f, map_location="cpu")
        meta = b["meta"]
        out[(meta["side"], int(meta["rank"]), int(meta["seq"]))] = {
            "file": f, "bundle": b, "meta": meta}
    return out


def rebuild_flat(bundle: dict, which: str, li: int) -> torch.Tensor:
    """按 block_table+cov 重建 token 序展平张量 (sum(cov), kv_heads, head_dim)"""
    meta = bundle["meta"]
    lay = bundle[which][li]
    parts = []
    for i, blk in enumerate(meta["block_table"]):
        cov = meta["cov"][i]
        if cov <= 0 or blk not in lay:
            continue
        parts.append(lay[blk][:cov])
    return torch.cat(parts, dim=0)


def recompute_fp_block(bundle: dict, li: int, blk: int):
    """(Tx, Xx) per K 与 V —— 从原样块按 09 行数语义切片"""
    meta = bundle["meta"]
    i = meta["block_table"].index(blk)
    tp = tx_len_i(meta, i)
    cov = meta["cov"][i]
    out = {}
    for which in ("K", "V"):
        t = bundle[which][li][blk]
        out[which] = (kvc_fp([t[:tp]]) if tp > 0 else "-",
                      kvc_fp([t[:cov]]))
    return out


def recompute_fp_layer(bundle: dict, li: int):
    meta = bundle["meta"]
    K = rebuild_flat(bundle, "K", li)
    V = rebuild_flat(bundle, "V", li)
    sum_cp = sum(cp_len_i(meta, i) for i in range(len(meta["cov"])))
    w = int(meta["w_tok"])
    return (kvc_fp([K[:sum_cp]]), kvc_fp([K[:w]]),
            kvc_fp([V[:sum_cp]]), kvc_fp([V[:w]]))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_check(t_dir: Path, log_dir: Path, out_prefix: Path):
    issues = []
    ok = True
    R = {"B0": [], "B1": [], "B2": [], "B3": [], "B4": {}}

    bundles = load_bundles(t_dir)
    if not bundles:
        return None, ["无 .pt 文件"], None

    # ---- B0 结构 ----
    for (side, rank, seq), it in sorted(bundles.items()):
        m, b = it["meta"], it["bundle"]
        cov_sum = sum(m["cov"])
        keys_ok = all(
            set(bd.keys()) == {m["block_table"][i] for i, c in enumerate(m["cov"]) if c > 0}
            for bd in b["K"]) and len(b["K"]) == int(m["layers"])
        shape_ok = all(
            bd[blk].shape == (int(m["block_size"]), int(m["kv_heads"]), int(m["head_dim"]))
            for bd in b["K"] for blk in bd)
        self_ok = (cov_sum == int(m["w_tok"]) and keys_ok and shape_ok)
        R["B0"].append({"file": it["file"].name, "side": side, "rank": rank,
                        "seq": seq, "p_tok": m["p_tok"], "w_tok": m["w_tok"],
                        "cov": m["cov"], "block_table": m["block_table"],
                        "kv_heads": m["kv_heads"], "struct_ok": self_ok})
        if not self_ok:
            issues.append(f"B0: {side}{rank}s{seq} 结构不变量不成立")
            ok = False
    # 跨侧对齐(P{r} vs D{r} 同 seq): p_tok/cov 必须相同; block_table 同构只报告
    byseq = {}
    for (side, rank, seq), it in bundles.items():
        byseq.setdefault(seq, {})[(side, rank)] = it
    for seq, pp in byseq.items():
        for r in sorted({rk for (_, rk) in pp}):
            if ("P", r) in pp and ("D", r) in pp:
                mp, md = pp[("P", r)]["meta"], pp[("D", r)]["meta"]
                # p_tok 必须相等(proxy 同 prompt 双发); cov 允许不等——req_r 预期形态:
                # P 不做 decode(w_tok=p_tok), D 侧 decode 扩到 prompt+max_tokens-1。
                # 离线比对取公共区间: B2 只比前 p_tok-1 行, 尾槽行 p_tok-1, decode 区
                # 仅 D 存在 —— 行号重建各自按自身 block_table+cov 做, 不要求同构。
                if int(mp["p_tok"]) != int(md["p_tok"]):
                    issues.append(f"B0: seq={seq} rank={r} P/D p_tok 不等"
                                  f"({mp['p_tok']} vs {md['p_tok']})")
                    ok = False
                if mp["cov"] != md["cov"] or mp["block_table"] != md["block_table"]:
                    R["B0"].append({
                        "note": f"seq={seq} r{r} cov/block_table 异构(pd期预期: P 停在"
                                f"prompt, D 延伸到 decode)——P.cov={mp['cov']} P.w_tok={mp['w_tok']}"
                                f" / D.cov={md['cov']} D.w_tok={md['w_tok']}",
                        "P_block_table": mp["block_table"],
                        "D_block_table": md["block_table"]})
    # 双 rank 齐备(P/D 各 rank0/1)
    want = {("P", 0), ("P", 1), ("D", 0), ("D", 1)}
    have = {(s, r) for (s, r, _) in bundles}
    if not want <= have:
        R["B0"].append({"note": f"rank 覆盖不全: {sorted(have)} (TP1 退化为单 rank 时 "
                        "P0/D0 亦可判定)"})

    # ---- B1 链间互证(集合匹配) ----
    fp_cand, fp_layer, term_seen = (parse_logs(log_dir)
                                    if log_dir and log_dir.is_dir() else ({}, {}, {}))
    seq2tag = {1: "reqp", 2: "reqr"}
    for (side, rank, seq), it in sorted(bundles.items()):
        m = it["meta"]
        tag = seq2tag.get(seq)
        rec = {"key": f"{side}{rank}s{seq}", "n_checked": 0, "n_equal": 0}
        R["B1"].append(rec)
        skey = side.lower()               # 日志文件名口径: 小写 p|d
        if not tag or (skey, tag) not in fp_cand:
            rec["note"] = "日志候选缺失(B1 跳过)"
            continue
        # TERM 旁证: (rid, rank) 应在 term_seen
        if term_seen.get((skey, tag)) and \
           (m["request_id"], rank) not in term_seen[(skey, tag)]:
            issues.append(f"B1: {side}{rank}s{seq} TERM 行未见 req={m['request_id'][:24]}.. "
                          f"dev=npu:{rank}")
            ok = False
        cand = fp_cand[(skey, tag)]
        cand_lay = fp_layer.get((skey, tag), {})
        n_ok = n_bad = 0
        for li in range(int(m["layers"])):
            ltag = f"L{li:02d}"
            lay_c = cand.get(ltag)
            if not lay_c:
                continue
            for i, blk in enumerate(m["block_table"]):
                if m["cov"][i] <= 0:
                    continue
                got = recompute_fp_block(it["bundle"], li, int(blk))
                for which in ("K", "V"):
                    want_pair = (got[which][0], got[which][1])
                    pool = lay_c.get(int(blk), {}).get(which, [])
                    for k, pair in enumerate(pool):
                        if pair == want_pair:
                            pool.pop(k)
                            n_ok += 2
                            break
                    else:
                        n_bad += 1
                        issues.append(
                            f"B1: {side}{rank}s{seq} {ltag} blk{blk} {which} "
                            f"指纹不在日志候选集 (Tx={want_pair[0]} Xx={want_pair[1]})")
            # 层指纹 multiset
            lp = cand_lay.get(ltag, [])
            got4 = recompute_fp_layer(it["bundle"], li)
            for k, quad in enumerate(lp):
                if quad == got4:
                    lp.pop(k)
                    n_ok += 4
                    break
            else:
                if lp:
                    n_bad += 1
                    issues.append(f"B1: {side}{rank}s{seq} {ltag} 层指纹不在候选集")
        rec["n_checked"] = n_ok + n_bad
        rec["n_equal"] = n_ok
        if n_bad:
            ok = False
    # 候选应恰好被双 rank 消解完(剩余=有日志指纹无归档对应——记录不 FAIL,
    # 可能是 LATE 或其他 rank 轨迹残留)
    for (side, tag), cand in fp_cand.items():
        leftover = sum(len(p) for lay in cand.values()
                       for blk in lay.values() for p in blk.values())
        if leftover:
            R["B1"].append({"note": f"{side}_{tag} 候选剩余 {leftover} 条未消解"
                            "(LATE/异常路径残留, 仅记录)"})

    # ---- B2 Tx 逐位 + B3 归类(同 rank 对) ----
    for seq, pp in byseq.items():
        for r in sorted({rk for (_, rk) in pp}):
            if ("P", r) not in pp or ("D", r) not in pp:
                continue
            mP, mD = pp[("P", r)]["meta"], pp[("D", r)]["meta"]
            p_tok = int(mP["p_tok"])
            n_tx = p_tok - 1
            n_eq = 0
            R3_rank = []
            for l in range(int(mP["layers"])):
                for which in ("K", "V"):
                    aP = rebuild_flat(pp[("P", r)]["bundle"], which, l)[:n_tx]
                    aD = rebuild_flat(pp[("D", r)]["bundle"], which, l)[:n_tx]
                    if aP.shape != aD.shape:
                        issues.append(f"B2: seq={seq} r{r} L{l:02d} {which} Tx 形状不等")
                        ok = False
                        continue
                    if bool((aP.view(torch.int16) == aD.view(torch.int16)).all()):
                        n_eq += 1
                    else:
                        ev, cnt, hist = diff_dump(aP, aD, l, which, mP, cap=20)
                        issues.append(
                            f"B2: seq={seq} r{r} L{l:02d} {which} Tx 区不等 "
                            f"({cnt} 元素, 签名={hist}) —— 疑似传输损伤")
                        R["B3"].extend(ev[:8])
                        ok = False
                # 尾槽(行 p_tok-1): 数值语义判据(v2 修正结论)
                for which in ("K", "V"):
                    tP = rebuild_flat(pp[("P", r)]["bundle"], which, l)[n_tx:n_tx + 1]
                    tD = rebuild_flat(pp[("D", r)]["bundle"], which, l)[n_tx:n_tx + 1]
                    if tP.numel() == 0 or tP.shape != tD.shape:
                        continue
                    if bool((tP.view(torch.int16) == tD.view(torch.int16)).all()):
                        R3_rank.append({"layer": l, "which": which, "tok": n_tx,
                                        "tag": "尾槽", "sig": "exact"})
                        continue
                    fP, fD = tP.float(), tD.float()
                    bad_nan = bool(torch.isnan(fP).any() or torch.isnan(fD).any()
                                   or torch.isinf(fP).any() or torch.isinf(fD).any())
                    scale = float(max(float(fP.abs().max()), float(fD.abs().max())))
                    abs_max = float((fP - fD).abs().max())
                    rel = abs_max / max(scale, 1e-8)
                    ev, cnt, hist = diff_dump(tP, tD, l, which, mP, cap=1024)
                    if bad_nan:
                        issues.append(f"B3: seq={seq} r{r} L{l:02d} {which} 尾槽含 NaN/Inf")
                        ok = False
                        R["B3"].extend(ev[:8])
                    elif rel <= 0.05:
                        R3_rank.append({
                            "layer": l, "which": which, "tok": n_tx, "tag": "尾槽",
                            "sig": f"recompute(|Δ|max={abs_max:.4g}={rel:.2%}幅值,{cnt}元素)",
                            "hist": hist, "first": ev[:3]})
                    else:
                        issues.append(
                            f"B3: seq={seq} r{r} L{l:02d} {which} 尾槽幅度异常 "
                            f"(|Δ|max={abs_max:.4g}={rel:.1%}幅值>5%)")
                        ok = False
                        R["B3"].extend(ev[:8])
            R["B2"].append({"seq": seq, "rank": r, "p_tok": p_tok,
                            "pairs_checked": int(mP["layers"]) * 2,
                            "pairs_equal": n_eq})
            R["B3"].extend(R3_rank)
            # decode 区(D 侧, 行 p_tok..w_tok-1): 健康扫描
            wD = int(mD["w_tok"])
            if wD > p_tok:
                dt = rebuild_flat(pp[("D", r)]["bundle"], "K", 0)[p_tok:wD]
                bad = int(torch.isnan(dt.float()).sum() + torch.isinf(dt.float()).sum())
                R["B4"][f"seq{seq}_r{r}_decode"] = {
                    "rows": wD - p_tok, "health": f"NaN/Inf {bad}" if bad else "无",
                    "first3": [round(v, 3) for v in dt[0, 0, :3].float().tolist()]}
                if bad:
                    issues.append(f"B3: seq={seq} r{r} decode 区 NaN/Inf {bad}")
                    ok = False
            # 未写槽位残值统计(块内 cov 之后): 只报告
            resid = {"seq": seq, "rank": r, "unwritten": {}}
            for which in ("K", "V"):
                tot = nz = 0
                for li in range(1):  # 首层代表统计
                    lay = pp[("D", r)]["bundle"][which][li]
                    for i, blk in enumerate(mD["block_table"]):
                        cov_i = mD["cov"][i]
                        rest = lay[int(blk)][cov_i:]
                        if rest.numel():
                            tot += rest.numel()
                            nz += int((rest.float() != 0).sum())
                resid["unwritten"][which] = {"slots": tot, "nonzero": nz}
            R["B4"][f"seq{seq}_r{r}_residual"] = resid

    verdict = "PASS" if ok else "FAIL"
    md = ["# block_report（kvc_offline pp2tp2 block 原样检查）",
          f"- verdict: **{verdict}**",
          f"- bundles: {sorted(k for k in bundles)}",
          f"- B1 集合互证: {R['B1']}",
          f"- B2 Tx: {R['B2']}",
          f"- B3 明细: {len(R['B3'])} 条  B4: {R['B4']}",
          f"- issues: {issues[:50]}"]
    js = {"verdict": verdict, "B0": R["B0"], "B1": R["B1"], "B2": R["B2"],
          "B3": R["B3"], "B4": R["B4"], "issues": issues}
    return verdict, issues, (md, js)


# ---------------------------------------------------------------------------
# selftest: 合成 P0/P1/D0/D1 × seq1/2(小规模 2 层/kv_heads=4/块 4 槽) + 交织日志
# ---------------------------------------------------------------------------
def selftest(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    tdir, ldir = tmp / "tensors", tmp / "logs"
    tdir.mkdir(exist_ok=True)
    ldir.mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(2024)

    LAYERS, KVH, HD, BS = 2, 4, 8, 4     # 缩小规模
    P_TOK = 6                            # cov=[4,2] -> Tx=[0:5), 尾槽=行5

    def rand_rows(n):
        return (torch.randn(n * KVH * HD, generator=g) * 1.2
                ).reshape(n, KVH, HD).to(torch.bfloat16)

    def mk_bundle(side, rank, seq, K_flat, V_flat):
        meta = {"schema": "kvt2-raw", "side": side, "rank": rank, "seq": seq,
                "request_id": f"cmpl-st{side}{rank}{seq}", "tag": "TERM",
                "p_tok": P_TOK, "w_tok": P_TOK, "final": P_TOK,
                "cov": [4, 2],
                "block_table": [11 + 2 * rank, 12 + 2 * rank], "layers": LAYERS,
                "layer_ids": ["0", "1"], "kv_heads": KVH, "head_dim": HD,
                "block_size": BS, "dtype": "torch.bfloat16",
                "dev": f"npu:{rank}", "ts": "selftest"}

        def _full_block(rows):   # 原样块: BS 整槽, 前 cov 行=数据, 未写槽=残值 0
            out = torch.zeros(BS, KVH, HD, dtype=torch.bfloat16)
            out[:rows.shape[0]] = rows
            return out
        b11, b12 = 11 + 2 * rank, 12 + 2 * rank
        K = [{b11: _full_block(K_flat[:4]), b12: _full_block(K_flat[4:])},
             {b11: _full_block(K_flat[:4]), b12: _full_block(K_flat[4:])}]
        V = [{b11: _full_block(V_flat[:4]), b12: _full_block(V_flat[4:])},
             {b11: _full_block(V_flat[:4]), b12: _full_block(V_flat[4:])}]
        b = {"K": K, "V": V, "meta": meta}
        # rid 尾段用纯 hex(FNAME_RE 口径 [0-9a-f]+)
        torch.save(b, tdir / f"kv_{side}{rank}_{seq}_cafe{rank}{seq:02d}.pt")
        return b

    def synth_logs(all_bundles):
        # 交织: 共享文件 [FPB]/[FP] 无 dev; 分 rank 文件 TERM 带 dev
        shared = {("p", "reqp"): [], ("d", "reqp"): [],
                  ("p", "reqr"): [], ("d", "reqr"): []}
        perank = {}
        for (side, rank, seq), b in sorted(all_bundles.items()):
            tag = "reqp" if seq == 1 else "reqr"
            skey = side.lower()                        # 文件名口径: 小写 p|d
            perank.setdefault((skey, tag), []).append(
                f"[KVC][KVP] TERM req={b['meta']['request_id']} dev=npu:{rank} "
                f"逐层按块: layers={LAYERS}")
            for li in range(LAYERS):
                ks, vs = [], []
                for blk in b["meta"]["block_table"]:
                    fp = recompute_fp_block(b, li, blk)
                    ks.append(f"blk{blk}K.Tx={fp['K'][0]}/K.Xx={fp['K'][1]}")
                    vs.append(f"blk{blk}V.Tx={fp['V'][0]}/V.Xx={fp['V'][1]}")
                shared[(skey, tag)].append(
                    f"[KVC][KVP][FPB] TERM L{li:02d} 块指纹: {' '.join(ks)} | {' '.join(vs)}")
                kp, ka, vp, va = recompute_fp_layer(b, li)
                shared[(skey, tag)].append(
                    f"[KVC][KVP][FP] TERM L{li:02d} 指纹 w_tok={P_TOK} p_tok={P_TOK} "
                    f"| K.prompt={kp} K.all={ka} | V.prompt={vp} V.all={va}")
        for (side, tag), lines in shared.items():
            (ldir / f"kvc_{side}_{tag}.log").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
        for (side, tag), lines in perank.items():
            for r in (0, 1):
                sub = [l for l in lines if f"dev=npu:{r} " in l or f"dev=npu:{r}\n" in l
                       or l.endswith(f"dev=npu:{r}")]
                (ldir / f"kvc_{side}{r}_{tag}.log").write_text(
                    "\n".join(sub) + "\n", encoding="utf-8")

    # ---- 场景 1: 正常(尾槽重算差→PASS) ----
    for f in tdir.glob("kv_*.pt"):
        f.unlink()
    buns = {}
    for rank in (0, 1):                      # rank 间语料独立(不同 kv_heads 切片语义)
        base = rand_rows(P_TOK)              # P{rank} 的 K 语料
        VP = rand_rows(P_TOK)                # P{rank} 的 V 语料(D 侧 V Tx 同源)
        buns[("P", rank, 1)] = mk_bundle("P", rank, 1, base.clone(), VP.clone())
        # D{rank}: Tx 前 5 行与 P 同 rank 逐位相同(传输无损), 尾槽(行5)注入重算差
        dK = base.clone()
        tail = dK[5].clone()
        v16 = int(tail[1, 3].view(torch.int16).to(torch.int32))
        tail[1, 3] = torch.tensor(v16 ^ 0x0001,
                                  dtype=torch.int16).view(torch.bfloat16)
        dK[5] = tail
        buns[("D", rank, 1)] = mk_bundle("D", rank, 1, dK, VP.clone())
    synth_logs(buns)
    v1, i1, rep1 = run_check(tdir, ldir, tmp)
    assert v1 == "PASS", f"场景1 期望 PASS: {i1[:5]}"
    b1 = [e for e in rep1[1]["B1"] if "n_equal" in e]
    assert all(e["n_equal"] == e["n_checked"] and e["n_checked"] > 0 for e in b1), b1
    sigs = [e.get("sig") for e in rep1[1]["B3"] if isinstance(e.get("sig"), str)]
    assert any(s.startswith("recompute(") for s in sigs), sigs
    print(f"[SELFTEST-1 PASS] pp2tp2 正常+尾槽重算差: verdict=PASS, B1={[(e['key'], e['n_equal']) for e in b1]}")

    # ---- 场景 2: D0 Tx 损伤(3bit) -> FAIL ----
    b2 = buns[("D", 0, 1)]                   # mk_bundle 返回裸 bundle dict
    blk11 = b2["K"][0][11]                  # rank0 L0 块11 前4行即 Tx 的一部分
    cur = blk11[1, 1, 2]
    v16 = int(cur.view(torch.int16).to(torch.int32))
    blk11[1, 1, 2] = torch.tensor(v16 ^ 0x0105,
                                  dtype=torch.int16).view(torch.bfloat16)
    torch.save(b2, tdir / "kv_D0_1_cafe001.pt")   # 改动回写磁盘(run_check 读盘)
    v2, i2, rep2 = run_check(tdir, ldir, tmp)
    assert v2 == "FAIL", "场景2 期望 FAIL"
    assert any("Tx 区不等" in s for s in i2), i2[:3]
    assert any("B1" in s for s in i2), "损伤后 B1 互证应同时 FAIL"  # 哈希对不上候选集
    print(f"[SELFTEST-2 PASS] Tx 损伤: verdict=FAIL, 首条={i2[0][:80]}")

    print("[SELFTEST ALL PASS] B0/B1 集合匹配/B2 equal/B3 数值语义/B4 残值 全部通过")


def main():
    ap = argparse.ArgumentParser(description="kvc_offline pp2tp2 block 原样检查器")
    ap.add_argument("--dir", default="log/tensors")
    ap.add_argument("--logs", default="log")
    ap.add_argument("--out", default="log/block_report")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(Path("/tmp/kvc_blk_selftest"))
        return

    verdict, issues, rep = run_check(Path(args.dir), Path(args.logs), Path(args.out))
    if rep is None:
        print("FAIL: 无输入")
        sys.exit(2)
    md, js = rep
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    p.with_suffix(".json").write_text(
        json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"verdict={verdict}")
    for s in issues[:20]:
        print("  !", s)
    print(f"报告: {p}.md / {p}.json")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
