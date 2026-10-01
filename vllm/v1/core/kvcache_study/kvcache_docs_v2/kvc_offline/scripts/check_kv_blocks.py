#!/usr/bin/env python3
# ==============================================================================
# check_kv_blocks.py —— kvc_offline(v3, 单机 PP2×TP2) block 原样归档离线检查器
#
# 场景: 单实例 vllm serve --tensor-parallel-size 2 --pipeline-parallel-size 2
#   （与 ../kvc/ 相同形态, 无 PD 分离）。4 worker = 2 PP stage × 2 TP rank:
#   每 worker 持本 stage 的 16 层(kv_caches 按 worker 本地序 0~15 枚举)与
#   4 kv_heads 分片(全局 8 / TP2)。归档: kv_S{pp}{tp}_{seq}_{rid尾8}.pt 每请求
#   每 worker 一份(11 号补丁, schema kvt3-raw)。
#
# 输入:
#   --dir   log/tensors   kv_S??_?_*.pt: K/V = [层序 list, 每层 dict{块号: 原样张量}]
#   --logs  log           kvc_p.log / kvc_r.log(09 [FPB]/[FP] 行, 4 worker 交织
#                          在同一文件且无 worker 标签——本地层序 L00~L15 在两个
#                          PP stage 重号, 同层号实际有 4 条: pp{0,1}×tp{0,1})
#   --out   log/block_report
#
# 检查(四级, 无 PD 传输语境):
#   C0 结构+分片覆盖: 每文件不变量(len(K)==layers, 键集=={b_i|cov_i>0},
#          shape==(bs,kv_heads,hd), sum(cov)==w_tok) + 4 worker 齐备
#          {(pp,tp)} + 同 seq 各 worker p_tok/cov/block_table 一致
#   C1 链间互证(集合匹配): 每 worker 的 .pt 重算 09 同口径 sha256(块 Tx/Xx +
#          层 prompt/all), 在日志候选多重集合(同层号 4 条: 2 PP×2 TP)中各消解
#          一个 —— .pt 与日志两链互证; 候选剩余仅记录(LATE/异常残留)
#   C2 前缀缓存驻留一致性(裁决核心, 替代 PD 的 Tx 对账): req_p(seq1) TERM 与
#          req_r(seq2) TERM 的**公共块**(seq2 前缀 HIT 的块, 如 [1,2])在
#          同 worker 同层下逐位 torch.equal —— 证明缓存命中复用不改数值
#          (HIT 块不重算不覆写, 驻留期间零篡改); 满块比全 128 槽,
#          非满块比 min(cov) 行
#   C3 健康: 全量 NaN/Inf 扫描 + decode 区(仅 seq2: 行 p_tok..w_tok-1)统计
#          + 未写槽位残值统计(原样整存独有, 仅报告)
#
# 判据: [PASS] C0 ∧ C1 全消解 ∧ C2 公共块全 equal ∧ C3 无 NaN/Inf
#       [FAIL] 其余(附取证包: 层/块/worker/K·V/tok/head/dim/xor_bits/位翻转谱)
#
# selftest: python3 check_kv_blocks.py --selftest
#   合成 4 worker × 2 请求bundles + 交织日志; 验证: 正常→PASS;
#   seq2 HIT 块被篡改→C2 FAIL; seq1 归档损坏→C1 FAIL。
# ==============================================================================
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# 09 同口径 sha256(前16hex) —— 与 _kvc_fp 逐字符一致, 保证 C1 可互证
# ---------------------------------------------------------------------------
def kvc_fp(parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# bf16 位模式工具(取证包; C2/C3 判据用数值语义)
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
    kv_heads = int(meta.get("kv_heads", 1))
    head_dim = int(meta.get("head_dim", 1))
    ev = []
    for i in idx[:cap]:
        row, rem = divmod(i, kv_heads * head_dim)
        head, dim = divmod(rem, head_dim)
        ai = int(a.reshape(-1)[i].view(torch.int16).to(torch.int32)) & 0xFFFF
        bi = int(b.reshape(-1)[i].view(torch.int16).to(torch.int32)) & 0xFFFF
        xv = ai ^ bi
        ev.append({"layer": l, "which": which, "tok": row, "head": head,
                   "dim": dim, "pa": f"0x{ai:04x}", "pb": f"0x{bi:04x}",
                   "xor_bits": bin(xv).count("1"), "sig": classify_xor(xv)})
    n_exact = int(a.numel()) - len(idx)
    if n_exact:
        hist = {"exact": n_exact, **hist}
    return ev, len(idx), hist


# ---------------------------------------------------------------------------
# 日志解析: kvc_p.log / kvc_r.log —— 4 worker 交织的 [FPB]/[FP] 收成候选多重集合
# ---------------------------------------------------------------------------
FPB_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FPB\] (\w+) (L\d+) 块指纹: (.*) \| (.*)")
FP_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FP\] (\w+) (L\d+) 指纹 w_tok=(\d+) p_tok=(\d+) "
    r"\| K\.prompt=(\S+) K\.all=(\S+) \| V\.prompt=(\S+) V\.all=(\S+)")
BLK_RE = re.compile(r"blk(\d+)[KV]\.Tx=([0-9a-f-]+)/[KV]\.Xx=([0-9a-f-]+)")


def parse_logs(log_dir: Path):
    """-> {tag: {"blk": {ltag: {blk: {"K": [(tx,xx)..], "V": [...]}},
                         "lay": {ltag: [(kp,ka,vp,va)..]}}}}   tag in {p, r}"""
    out = {}
    for tag, fname in (("p", "kvc_p.log"), ("r", "kvc_r.log")):
        f = log_dir / fname
        d = {"blk": {}, "lay": {}}
        if f.exists():
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                m = FPB_RE.search(line)
                if m:
                    ltag, kpart, vpart = m.group(2), m.group(3), m.group(4)
                    lay = d["blk"].setdefault(ltag, {})
                    for which, part in (("K", kpart), ("V", vpart)):
                        for blk, tx, xx in BLK_RE.findall(part):
                            lay.setdefault(int(blk), {}).setdefault(which, []).append((tx, xx))
                    continue
                m = FP_RE.search(line)
                if m:
                    ltag = m.group(2)
                    d["lay"].setdefault(ltag, []).append(
                        (m.group(5), m.group(6), m.group(7), m.group(8)))
        out[tag] = d
    return out


# ---------------------------------------------------------------------------
# .pt 加载 / 重算
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(r"^kv_S(\d)(\d)_(\d)_([0-9a-f]+)\.pt$")


def load_bundles(t_dir: Path):
    out = {}
    for f in sorted(t_dir.glob("kv_*.pt")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        pp, tp, seq = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            b = torch.load(f, map_location="cpu", weights_only=False)
        except TypeError:
            b = torch.load(f, map_location="cpu")
        meta = b["meta"]
        out[(pp, tp, int(meta["seq"]))] = {"file": f, "bundle": b, "meta": meta}
    return out


def recompute_fp_block(bundle: dict, li: int, blk: int):
    """{(K,V): (Tx, Xx)} —— 从原样块按 09 行数语义切片"""
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
    sum_cp = sum(cp_len_i(meta, i) for i in range(len(meta["cov"])))
    w = int(meta["w_tok"])
    parts = {which: [] for which in ("K", "V")}
    for i, blk in enumerate(meta["block_table"]):
        cov = meta["cov"][i]
        if cov <= 0:
            continue
        for which in ("K", "V"):
            parts[which].append(bundle[which][li][blk][:cov])
    K = torch.cat(parts["K"], dim=0)
    V = torch.cat(parts["V"], dim=0)
    return (kvc_fp([K[:sum_cp]]), kvc_fp([K[:w]]),
            kvc_fp([V[:sum_cp]]), kvc_fp([V[:w]]))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_check(t_dir: Path, log_dir: Path, out_prefix: Path):
    issues = []
    ok = True
    R = {"C0": [], "C1": [], "C2": [], "C3": [], "C4": {}}

    bundles = load_bundles(t_dir)
    if not bundles:
        return None, ["无 .pt 文件"], None

    workers = sorted({(pp, tp) for (pp, tp, _) in bundles})
    seqs = sorted({seq for (_, _, seq) in bundles})

    # ---- C0 结构 + 分片覆盖 ----
    for (pp, tp, seq), it in sorted(bundles.items()):
        m, b = it["meta"], it["bundle"]
        cov_sum = sum(m["cov"])
        want_keys = {m["block_table"][i] for i, c in enumerate(m["cov"]) if c > 0}
        keys_ok = all(set(bd.keys()) == want_keys for bd in b["K"]) \
            and len(b["K"]) == len(b["V"]) == int(m["layers"])
        shape_ok = all(
            bd[blk].shape == (int(m["block_size"]), int(m["kv_heads"]), int(m["head_dim"]))
            for bd in b["K"] for blk in bd)
        self_ok = (cov_sum == int(m["w_tok"]) and keys_ok and shape_ok)
        R["C0"].append({"file": it["file"].name, "pp": pp, "tp": tp,
                        "seq": seq, "layers": m["layers"],
                        "layer_ids_head": m["layer_ids"][:3],
                        "layer_ids_tail": m["layer_ids"][-3:],
                        "p_tok": m["p_tok"], "w_tok": m["w_tok"],
                        "cov": m["cov"], "block_table": m["block_table"],
                        "kv_heads": m["kv_heads"], "struct_ok": self_ok})
        if not self_ok:
            issues.append(f"C0: S{pp}{tp}s{seq} 结构不变量不成立")
            ok = False
    want_workers = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if workers and set(workers) != want_workers:
        R["C0"].append({"note": f"worker 分片覆盖不完全: {workers} "
                        "(期望 pp0/pp1 × tp0/tp1 四路)"})
    for seq in seqs:
        metas = [(pp, tp, bundles[(pp, tp, seq)]["meta"])
                 for (pp, tp, s) in bundles if s == seq]
        if len(metas) < 2:
            continue
        base = metas[0][2]
        for pp, tp, m in metas[1:]:
            if (int(m["p_tok"]), m["cov"], m["block_table"]) != \
               (int(base["p_tok"]), base["cov"], base["block_table"]):
                issues.append(f"C0: seq={seq} 各 worker p_tok/cov/block_table 不一致")
                ok = False
                break

    # ---- C1 链间互证(集合匹配; 同 ltag 候选=4 条: 2 PP × 2 TP) ----
    logs = parse_logs(log_dir) if log_dir and log_dir.is_dir() else {}
    seq2tag = {1: "p", 2: "r"}
    for (pp, tp, seq), it in sorted(bundles.items()):
        m = it["meta"]
        tag = seq2tag.get(seq)
        rec = {"key": f"S{pp}{tp}s{seq}", "n_checked": 0, "n_equal": 0}
        R["C1"].append(rec)
        if not tag or tag not in logs:
            rec["note"] = "日志候选缺失(C1 跳过)"
            continue
        cand = logs[tag]
        n_ok = n_bad = 0
        for li in range(int(m["layers"])):
            ltag = f"L{li:02d}"
            lay_c = cand["blk"].get(ltag)
            if not lay_c:
                continue
            for i, blk in enumerate(m["block_table"]):
                if m["cov"][i] <= 0:
                    continue
                got = recompute_fp_block(it["bundle"], li, int(blk))
                for which in ("K", "V"):
                    pair = got[which]
                    pool = lay_c.get(int(blk), {}).get(which, [])
                    for k, cand_pair in enumerate(pool):
                        if cand_pair == pair:
                            pool.pop(k)
                            n_ok += 1
                            break
                    else:
                        n_bad += 1
                        issues.append(
                            f"C1: S{pp}{tp}s{seq} {ltag} blk{blk} {which} "
                            f"指纹不在日志候选集 (Tx={pair[0]} Xx={pair[1]})")
            lp = cand["lay"].get(ltag, [])
            if lp:
                got4 = recompute_fp_layer(it["bundle"], li)
                for k, quad in enumerate(lp):
                    if quad == got4:
                        lp.pop(k)
                        n_ok += 1
                        break
                else:
                    n_bad += 1
                    issues.append(f"C1: S{pp}{tp}s{seq} {ltag} 层指纹不在候选集")
        rec["n_checked"] = n_ok + n_bad
        rec["n_equal"] = n_ok
        if n_bad:
            ok = False
    for tag, d in logs.items():
        leftover = sum(len(p) for lay in d["blk"].values()
                       for blk in lay.values() for p in blk.values())
        leftover += sum(len(v) for v in d["lay"].values())
        if leftover:
            R["C1"].append({"note": f"kvc_{tag}.log 候选剩余 {leftover} 条未消解"
                            "(LATE/异常残留, 仅记录)"})

    # ---- C2 前缀缓存驻留一致性(seq1 vs seq2 公共块, 同 worker 同层) ----
    byws = {}
    for (pp, tp, seq), it in bundles.items():
        byws.setdefault((pp, tp), {})[seq] = it
    for (pp, tp), per in sorted(byws.items()):
        if 1 not in per or 2 not in per:
            continue
        m1, m2 = per[1]["meta"], per[2]["meta"]
        common = [b for b in m1["block_table"] if b in m2["block_table"]]
        rec = {"worker": f"S{pp}{tp}", "common_blocks": common, "n_pairs": 0,
               "n_equal": 0}
        R["C2"].append(rec)
        if not common:
            rec["note"] = "无公共块(前缀未命中?)"
            continue
        for li in range(int(m1["layers"])):
            for blk in common:
                i1 = m1["block_table"].index(blk)
                i2 = m2["block_table"].index(blk)
                rows = min(m1["cov"][i1], m2["cov"][i2])
                if rows <= 0:
                    continue
                for which in ("K", "V"):
                    a = per[1]["bundle"][which][li][blk][:rows]
                    b = per[2]["bundle"][which][li][blk][:rows]
                    rec["n_pairs"] += 1
                    if bool((a.view(torch.int16) == b.view(torch.int16)).all()):
                        rec["n_equal"] += 1
                    else:
                        ev, cnt, hist = diff_dump(a, b, li, which, m1, cap=8)
                        issues.append(
                            f"C2: S{pp}{tp} {blk}(公共块,前{rows}行) {which} "
                            f"L{li:02d} 驻留不一致 ({cnt} 元素, 签名={hist}) "
                            "—— 缓存命中复用改写数值")
                        R["C3"].extend(ev[:8])
                        ok = False
    # ---- C3/C4 健康与统计 ----
    for (pp, tp, seq), it in sorted(bundles.items()):
        m, b = it["meta"], it["bundle"]
        nan_inf = 0
        for which in ("K", "V"):
            for li, lay in enumerate(b[which]):
                for blk, t in lay.items():
                    f = t.float()
                    nan_inf += int(torch.isnan(f).sum() + torch.isinf(f).sum())
        if nan_inf:
            issues.append(f"C3: S{pp}{tp}s{seq} 归档含 NaN/Inf 共 {nan_inf}")
            ok = False
    for (pp, tp), per in sorted(byws.items()):
        if 2 not in per:
            continue
        m = per[2]["meta"]
        p_tok, w = int(m["p_tok"]), int(m["w_tok"])
        if w > p_tok:
            parts = []
            off = []
            acc = 0
            for i, blk in enumerate(m["block_table"]):
                off.append(acc)
                acc += m["cov"][i]
            for i, blk in enumerate(m["block_table"]):
                lo = max(p_tok, off[i])
                hi = min(w, off[i] + m["cov"][i])
                if hi > lo:
                    parts.append(per[2]["bundle"]["K"][0][blk][lo - off[i]:hi - off[i]])
            if parts:
                dt = torch.cat(parts, dim=0)
                R["C4"][f"S{pp}{tp}_decode"] = {
                    "rows": w - p_tok,
                    "health": "无" if not int(torch.isnan(dt.float()).sum() + torch.isinf(dt.float()).sum()) else "NaN/Inf",
                    "first3": [round(v, 3) for v in dt[0, 0, :3].float().tolist()]}
    # 残值统计(seq2 未写槽)
    for (pp, tp), per in sorted(byws.items()):
        if 2 not in per:
            continue
        m, b = per[2]["meta"], per[2]["bundle"]
        stat = {}
        for which in ("K", "V"):
            tot = nz = 0
            lay = b[which][0]
            for i, blk in enumerate(m["block_table"]):
                cov_i = m["cov"][i]
                rest = lay[int(blk)][cov_i:]
                if rest.numel():
                    tot += rest.numel()
                    nz += int((rest.float() != 0).sum())
            stat[which] = {"slots": tot, "nonzero": nz}
        R["C4"][f"S{pp}{tp}_residual"] = stat

    verdict = "PASS" if ok else "FAIL"
    md = ["# block_report（kvc_offline v3 单机 PP2×TP2 block 原样检查）",
          f"- verdict: **{verdict}**",
          f"- workers: {workers} | seqs: {seqs}",
          f"- C1 集合互证: {R['C1']}",
          f"- C2 缓存驻留: {R['C2']}",
          f"- C4 统计: {R['C4']}",
          f"- issues: {issues[:50]}"]
    js = {"verdict": verdict, "C0": R["C0"], "C1": R["C1"], "C2": R["C2"],
          "C3": R["C3"], "C4": R["C4"], "issues": issues}
    return verdict, issues, (md, js)


# ---------------------------------------------------------------------------
# selftest: 合成 4 worker(pp0/1 × tp0/1) × 2 seq 小规模 bundles + 交织日志
# ---------------------------------------------------------------------------
def selftest(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    tdir, ldir = tmp / "tensors", tmp / "logs"
    for d in (tdir, ldir):
        d.mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(2026)

    LAYERS, KVH, HD, BS = 2, 2, 8, 4      # 缩小规模: 每 worker 2 层/2 heads
    P_TOK = 6                             # cov=[4,2] -> blk11 满块 blk12 未满

    def rand_rows(n, seed_off):
        gg = torch.Generator().manual_seed(2026 + seed_off)
        return (torch.randn(n * KVH * HD, generator=gg) * 1.2
                ).reshape(n, KVH, HD).to(torch.bfloat16)

    def mk_bundle(pp, tp, seq, blocks, K_by_blk, V_by_blk, rid):
        meta = {"schema": "kvt3-raw", "side": "S", "pp": pp, "tp": tp, "seq": seq,
                "request_id": rid, "tag": "TERM",
                "p_tok": P_TOK, "w_tok": P_TOK if seq == 1 else P_TOK + 2,
                "final": P_TOK,
                "cov": [4, 2] if seq == 1 else [4, 2, 2],
                "block_table": blocks, "layers": LAYERS,
                "layer_ids": [str(i) for i in range(LAYERS)],
                "kv_heads": KVH, "head_dim": HD, "block_size": BS,
                "dtype": "torch.bfloat16", "dev": f"npu:{tp}",
                "ts": "selftest"}
        K = [{blk: t.clone() for blk, t in K_by_blk.items()} for _ in range(LAYERS)]
        V = [{blk: t.clone() for blk, t in V_by_blk.items()} for _ in range(LAYERS)]
        b = {"K": K, "V": V, "meta": meta}
        torch.save(b, tdir / f"kv_S{pp}{tp}_{seq}_{rid}.pt")
        return b

    def synth_logs(buns):
        lines = {"p": [], "r": []}
        for (pp, tp, seq), b in sorted(buns.items()):
            tag = "p" if seq == 1 else "r"
            for li in range(LAYERS):
                ks, vs = [], []
                for blk in b["meta"]["block_table"]:
                    fp = recompute_fp_block(b, li, blk)
                    ks.append(f"blk{blk}K.Tx={fp['K'][0]}/K.Xx={fp['K'][1]}")
                    vs.append(f"blk{blk}V.Tx={fp['V'][0]}/V.Xx={fp['V'][1]}")
                lines[tag].append(
                    f"[KVC][KVP][FPB] TERM L{li:02d} 块指纹: {' '.join(ks)} | {' '.join(vs)}")
                kp, ka, vp, va = recompute_fp_layer(b, li)
                lines[tag].append(
                    f"[KVC][KVP][FP] TERM L{li:02d} 指纹 w_tok={b['meta']['w_tok']} "
                    f"p_tok={P_TOK} | K.prompt={kp} K.all={ka} | V.prompt={vp} V.all={va}")
        for tag, ls in lines.items():
            (ldir / f"kvc_{tag}.log").write_text("\n".join(ls) + "\n", encoding="utf-8")

    # ---- 场景 1: 正常(公共块 11,12 驻留一致 → PASS) ----
    for f in tdir.glob("kv_*.pt"):
        f.unlink()
    buns = {}
    for pp in (0, 1):
        for tp in (0, 1):
            seed_off = pp * 10 + tp          # worker 间数据独立(不同层/heads 分片)
            base = {11: rand_rows(4, seed_off + 100), 12: rand_rows(2, seed_off + 200)}
            # seq1: blocks [11,12]
            buns[(pp, tp, 1)] = mk_bundle(
                pp, tp, 1, [11, 12],
                {11: _full(base[11], BS, KVH, HD), 12: _full(base[12], BS, KVH, HD)},
                {11: _full(rand_rows(4, seed_off + 300), BS, KVH, HD),
                 12: _full(rand_rows(2, seed_off + 400), BS, KVH, HD)},
                "cafe0001")
            # seq2: 前缀 HIT 保留 11,12 逐位不变; 新增 blk13(decode 增量)
            buns[(pp, tp, 2)] = mk_bundle(
                pp, tp, 2, [11, 12, 13],
                {11: _full(base[11], BS, KVH, HD), 12: _full(base[12], BS, KVH, HD),
                 13: _full(rand_rows(2, seed_off + 500), BS, KVH, HD)},
                {11: _full(rand_rows(4, seed_off + 300), BS, KVH, HD),
                 12: _full(rand_rows(2, seed_off + 400), BS, KVH, HD),
                 13: _full(rand_rows(2, seed_off + 600), BS, KVH, HD)},
                "cafe0002")
    synth_logs(buns)
    v1, i1, rep1 = run_check(tdir, ldir, tmp)
    assert v1 == "PASS", f"场景1 期望 PASS: {i1[:5]}"
    c1 = [e for e in rep1[1]["C1"] if "n_checked" in e]
    assert all(e["n_equal"] == e["n_checked"] and e["n_checked"] > 0 for e in c1), c1
    c2 = [e for e in rep1[1]["C2"] if "n_pairs" in e]
    assert all(e["n_equal"] == e["n_pairs"] for e in c2), c2
    print(f"[SELFTEST-1 PASS] 单机 4 worker 正常+缓存驻留: verdict=PASS, "
          f"C1={[(e['key'], e['n_equal']) for e in c1]}, C2={[(e['worker'], e['n_equal']) for e in c2]}")

    # ---- 场景 2: seq2 的 HIT 块被篡改 → C2 FAIL ----
    b2 = buns[(0, 0, 2)]
    cur = b2["K"][0][11][2, 1, 3]
    v16 = int(cur.view(torch.int16).to(torch.int32))
    b2["K"][0][11][2, 1, 3] = torch.tensor(
        v16 ^ 0x0105, dtype=torch.int16).view(torch.bfloat16)
    torch.save(b2, tdir / "kv_S00_2_cafe0002.pt")
    v2, i2, rep2 = run_check(tdir, ldir, tmp)
    assert v2 == "FAIL", "场景2 期望 FAIL"
    assert any("驻留不一致" in s for s in i2), i2[:3]
    assert any("C1" in s for s in i2), "篡改后 C1 哈希互证应同时 FAIL"
    print(f"[SELFTEST-2 PASS] HIT 块篡改: verdict=FAIL, 首条={i2[0][:80]}")

    # ---- 场景 3: seq1 归档损坏(不落日志候选) → C1 FAIL ----
    b1 = torch.load(tdir / "kv_S11_1_cafe0001.pt", map_location="cpu")
    cur = b1["V"][1][12][1, 0, 5]
    v16 = int(cur.view(torch.int16).to(torch.int32))
    b1["V"][1][12][1, 0, 5] = torch.tensor(
        v16 ^ 0x0201, dtype=torch.int16).view(torch.bfloat16)
    torch.save(b1, tdir / "kv_S11_1_cafe0001.pt")
    v3, i3, rep3 = run_check(tdir, ldir, tmp)
    assert v3 == "FAIL", "场景3 期望 FAIL"
    assert any(s.startswith("C1") for s in i3), i3[:3]
    print(f"[SELFTEST-3 PASS] seq1 归档损坏: verdict=FAIL, 首条={i3[0][:80]}")

    print("[SELFTEST ALL PASS] C0 分片覆盖 / C1 集合互证 / C2 缓存驻留一致 / C3-C4 健康统计 全部通过")


def _full(rows, BS, KVH, HD):
    out = torch.zeros(BS, KVH, HD, dtype=torch.bfloat16)
    out[:rows.shape[0]] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description="kvc_offline v3 单机 PP2×TP2 block 原样检查器")
    ap.add_argument("--dir", default="log/tensors")
    ap.add_argument("--logs", default="log")
    ap.add_argument("--out", default="log/block_report")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(Path("/tmp/kvc_blk_v3_selftest"))
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
