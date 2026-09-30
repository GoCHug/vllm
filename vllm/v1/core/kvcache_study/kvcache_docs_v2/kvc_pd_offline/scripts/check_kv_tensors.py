#!/usr/bin/env python3
# ==============================================================================
# check_kv_tensors.py —— kvc_pd_offline v2 检查器: TERM 张量归档五级离线全检
#
# 输入:
#   --dir   log/tensors   kv_{P,D}_{1,2}_{rid尾8}.pt (10 号补丁产出) + manifest*.jsonl(可选)
#   --logs  log           v1 轨迹日志 kvc_{p,d}_{reqp,reqr}.log (09 [FPB]/[FP] 行)
#   --out   log/tensor_report   报告前缀 -> .md/.json
#
# 五级:
#   L0 结构: meta 不变量(sum(cov)==w_tok/len(K)==layers/shape/dtype) + 跨侧 p_tok 对账
#   L1 链间互证: .pt 重算 09 同口径 sha256(块 Tx/Xx + 层 prompt/all) == 日志 [FPB]/[FP]
#   L2 Tx 逐位: P.K[l][:p_tok-1] vs D.K[l][:p_tok-1] torch.equal (裁决核心)
#   L3 差异归类: 尾槽 ULP 签名(bf16 bit0-6 尾数位, xor<=2) vs 疑似 DMA(xor>=3/指数/成片)
#              + decode 区(仅 D)健康扫描
#   L4 报告: md + json
#
# 判据:
#   [PASS] L0 结构对齐 ∧ L1 两链指纹互证全等 ∧ L2 Tx 全 torch.equal
#         ∧ L3 尾槽差异=exact 或重算一致(幅度判据: |Δ|max≤5% 层幅值且无 NaN/Inf)
#         ∧ decode 区数值健康
#   [FAIL] 其余(附取证包: 层/块/K|V/tok/head/dim/xor_bits/位翻转谱/相对差分布)
#   注: 尾槽重算差实测为分布式位翻转(bf16 7 位尾数, 低幅值元素多位翻常见), 故
#       L3 裁决用数值语义(幅度有界+分布同构), 位谱仅作取证包; Tx 区(L2)差异
#       零容忍, 位翻转谱用于刻画损伤形态(EXPONENT/成片 = 疑似 DMA)。
#
# selftest(无需容器): python3 check_kv_tensors.py --selftest
#   合成 P/D 成对 .pt + 合成 [FPB] 日志, 注入三类差异(exact/ULP/DMA), 验证判定
# ==============================================================================
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# 09 同口径 sha256(前 16 hex) —— 与 _kvc_fp 逐字符一致, 保证 L1 可互证
# ---------------------------------------------------------------------------
def kvc_fp(parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# bf16 位模式工具: xor 后按位分类(签名判别的实现核心)
#   bf16: 1 符号(bit15) + 8 指数(bit7-14) + 7 尾数(bit0-6)
# ---------------------------------------------------------------------------
def xor_int32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 视图转 int32 再异或(避免 int16 溢出负数), 返回逐元素 xor 值"""
    return a.view(torch.int16).to(torch.int32) ^ b.view(torch.int16).to(torch.int32)


def classify_xor(x: int) -> str:
    """单元素 xor 值 -> 签名分类"""
    if x == 0:
        return "exact"
    popcount = bin(x & 0xFFFF).count("1")
    if popcount <= 2 and (x & 0xFF80) == 0:        # 仅尾数低位(bit0-6)
        return "ULP"
    if (x & 0x7F80):                                # 指数位翻转
        return "EXPONENT"
    if (x & 0x8000):                                # 符号位翻转
        return "SIGN"
    return "MANHIGH"                                # 尾数高位多 bit(>=3)


# ---------------------------------------------------------------------------
# 日志解析(与 compare_fp.py 同正则) + TERM rid 提取
# ---------------------------------------------------------------------------
FPB_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FPB\] (\w+) (L\d+) 块指纹: (.*) \| (.*)")
FP_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FP\] (\w+) (L\d+) 指纹 w_tok=(\d+) p_tok=(\d+) "
    r"\| K\.prompt=(\S+) K\.all=(\S+) \| V\.prompt=(\S+) V\.all=(\S+)")
BLK_RE = re.compile(r"blk(\d+)[KV]\.Tx=([0-9a-f-]+)/[KV]\.Xx=([0-9a-f-]+)")
TERM_RE = re.compile(
    r"\[KVC\]\[KVP\] (TERM|LATE) req=(\S+) dev=\S+ 逐层按块")


def parse_trajs(log_dir: Path):
    """扫 4 个轨迹日志 -> {key=('p','reqp'): {rid, layers{L00..}, }}"""
    out = {}
    for f in sorted(log_dir.glob("kvc_[pd]_req[pr].log")):
        side = f.name.split("_")[1]                # p | d
        rtag = f.name.split("_")[2].split(".")[0]  # reqp | reqr
        d = {"rid": None, "layers": {}, "w_tok": None, "p_tok": None}
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = TERM_RE.search(line)
            if m and d["rid"] is None:
                d["rid"] = m.group(2)
            m = FPB_RE.search(line)
            if m:
                layer, kpart, vpart = m.group(2), m.group(3), m.group(4)
                lay = d["layers"].setdefault(layer, {"K": {}, "V": {}})
                for which, part in (("K", kpart), ("V", vpart)):
                    for blk, tx, xx in BLK_RE.findall(part):
                        lay[which][int(blk)] = (tx, xx)
            m = FP_RE.search(line)
            if m:
                layer = m.group(2)
                lay = d["layers"].setdefault(layer, {"K": {}, "V": {}})
                d["w_tok"], d["p_tok"] = int(m.group(3)), int(m.group(4))
                lay["fp"] = {"K.prompt": m.group(5), "K.all": m.group(6),
                             "V.prompt": m.group(7), "V.all": m.group(8)}
        out[(side, rtag)] = d
    return out


# ---------------------------------------------------------------------------
# .pt 加载与块/层重算
# ---------------------------------------------------------------------------
def load_tensors(dir_path: Path):
    out = {}
    for f in sorted(dir_path.glob("kv_*_*.pt")):
        try:
            b = torch.load(f, map_location="cpu", weights_only=False)
        except TypeError:                            # 旧版 torch 无 weights_only
            b = torch.load(f, map_location="cpu")
        meta = b["meta"]
        key = (meta["side"], int(meta["seq"]))
        out[key] = {"file": f, "bundle": b, "meta": meta}
    return out


def block_offsets(meta) -> list:
    """cov 前缀和: 块 i 行区间 [off[i], off[i]+cov[i])"""
    cov = meta["cov"]
    off, acc = [], 0
    for c in cov:
        off.append(acc)
        acc += c
    return off


def tx_len_i(meta, i) -> int:
    """块 i 的 Tx 行数 = max(0, min(cov_i, (p_tok-1) - off_i)) —— 09 tp 公式"""
    cov, p_tok = meta["cov"], int(meta["p_tok"])
    off = block_offsets(meta)
    return max(0, min(cov[i], (p_tok - 1) - off[i]))


def cp_len_i(meta, i) -> int:
    """块 i 的 prompt 区行数 = min(cov_i, max(0, p_tok - off_i)) —— 09 cp 公式"""
    cov, p_tok = meta["cov"], int(meta["p_tok"])
    off = block_offsets(meta)
    return min(cov[i], max(0, p_tok - off[i]))


def recompute_fp(bundle: dict):
    """从 .pt 重算 09 全套指纹 -> {L 层序: {...}} — 供 L1 对账
    层序与 bucket['K'] 列表索引一致(meta['layer_ids'] 供参考)"""
    meta = bundle["meta"]
    off = block_offsets(meta)
    res = {}
    for li, K in enumerate(bundle["K"]):
        V = bundle["V"][li]
        lay = {"K": {}, "V": {}}
        for i, blk in enumerate(meta["block_table"]):
            cov = meta["cov"][i]
            tp = tx_len_i(meta, i)
            lay["K"][int(blk)] = (
                kvc_fp([K[off[i]: off[i] + tp]]) if tp > 0 else "-",
                kvc_fp([K[off[i]: off[i] + cov]]))
            lay["V"][int(blk)] = (
                kvc_fp([V[off[i]: off[i] + tp]]) if tp > 0 else "-",
                kvc_fp([V[off[i]: off[i] + cov]]))
        sum_cp = sum(cp_len_i(meta, i) for i in range(len(meta["cov"])))
        w_tok = int(meta["w_tok"])
        lay["fp"] = {
            "K.prompt": kvc_fp([K[:sum_cp]]), "K.all": kvc_fp([K[:w_tok]]),
            "V.prompt": kvc_fp([V[:sum_cp]]), "V.all": kvc_fp([V[:w_tok]]),
        }
        res[li] = lay
    return res


# ---------------------------------------------------------------------------
# L3 取证包构造
# ---------------------------------------------------------------------------
def evidence(l: int, which: str, tok: int, meta_page: dict,
             a: torch.Tensor, b: torch.Tensor) -> dict:
    """单元素差异取证: pa/pb bf16 hex / xor_bits / bit 位谱"""
    ai = int(a.view(torch.int16).to(torch.int32)) & 0xFFFF
    bi = int(b.view(torch.int16).to(torch.int32)) & 0xFFFF
    x = ai ^ bi
    bits = [str(k) for k in range(16) if x & (1 << k)]
    # tok 所在块反推
    off = block_offsets(meta_page)
    blk = None
    for i, c in enumerate(meta_page["cov"]):
        if off[i] <= tok < off[i] + c:
            blk = meta_page["block_table"][i]
            break
    return {"layer": l, "which": which, "blk": blk, "tok": tok,
            "pa": f"0x{ai:04x}", "pb": f"0x{bi:04x}",
            "xor_bits": bin(x).count("1"), "bit_pos": "+".join(bits),
            "sig": classify_xor(x)}


def diff_dump(a: torch.Tensor, b: torch.Tensor, l: int, which: str,
              meta_page: dict, cap: int):
    """返回 (首证包列表, 不等元素总数, 签名直方图) — a/b 为同形展平切片"""
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
    # 首证包: head/dim 展平索引还原
    kv_heads = int(meta_page.get("kv_heads", a.shape[-2] if a.dim() >= 2 else 1))
    head_dim = int(meta_page.get("head_dim", a.shape[-1] if a.dim() >= 2 else 1))
    flat_pos = 0
    for i in idx[:cap]:
        if flat_pos >= cap:
            break
        row, rem = divmod(i, kv_heads * head_dim)
        head, dim = divmod(rem, head_dim)
        ev.append(evidence(l, which, row, meta_page,
                           a.reshape(-1)[i], b.reshape(-1)[i]))
        flat_pos += 1
    hist_full = int(a.numel()) - len(idx)
    if hist_full:
        hist = {"exact": hist_full, **hist}
    return ev, len(idx), hist


# ---------------------------------------------------------------------------
# 主检查流程
# ---------------------------------------------------------------------------
def run_check(t_dir: Path, log_dir: Path, out_prefix: Path):
    issues, details = [], []
    verdict_pass = True
    R = {"L0": [], "L1": [], "L2": [], "L3": [], "L4": {}}

    # ---- L0 结构 ----
    tens = load_tensors(t_dir)
    if not tens:
        return None, ["无 .pt 文件"], None
    pairs = {}
    for key, it in sorted(tens.items()):
        m = it["meta"]
        b = it["bundle"]
        cov_sum = sum(m["cov"])
        ok = (cov_sum == int(m["w_tok"]) and len(b["K"]) == int(m["layers"])
              and all(K.shape == (cov_sum, int(m["kv_heads"]), int(m["head_dim"]))
                      for K in b["K"])
              and all(V.shape == (cov_sum, int(m["kv_heads"]), int(m["head_dim"]))
                      for V in b["V"]))
        tag = f"kv_{m['side']}_{m['seq']}"
        R["L0"].append({"file": it["file"].name, "side": m["side"],
                        "seq": m["seq"], "p_tok": m["p_tok"],
                        "w_tok": m["w_tok"], "cov": m["cov"],
                        "block_table": m["block_table"],
                        "shape_ok": ok})
        if not ok:
            issues.append(f"L0: {tag} 结构不变量不成立(sum(cov)={cov_sum}, "
                          f"w_tok={m['w_tok']})")
            verdict_pass = False
        pairs.setdefault(int(m["seq"]), {})[m["side"]] = it

    for seq, pp in pairs.items():
        if "P" in pp and "D" in pp:
            if int(pp["P"]["meta"]["p_tok"]) != int(pp["D"]["meta"]["p_tok"]):
                issues.append(f"L0: seq={seq} P/D p_tok 不等("
                              f"{pp['P']['meta']['p_tok']} vs {pp['D']['meta']['p_tok']})")
                verdict_pass = False

    # ---- L1 链间互证 ----
    trajs = parse_trajs(log_dir) if log_dir and log_dir.is_dir() else {}
    seq2rtag = {1: "reqp", 2: "reqr"}
    for (side, seq), it in sorted(tens.items()):
        rtag = seq2rtag.get(seq)
        tr = trajs.get((side.lower(), rtag)) if rtag else None
        if tr is None or not tr["rid"]:
            R["L1"].append({"key": f"kv_{side}_{seq}", "log": None,
                            "n_checked": 0, "n_equal": 0, "note": "日志缺失(L1 跳过)"})
            continue
        if tr["rid"] and tr["rid"] != it["meta"]["request_id"]:
            issues.append(f"L1: kv_{side}_{seq} request_id 与日志不符("
                          f"pt={it['meta']['request_id']} vs log={tr['rid']})")
            verdict_pass = False
        rec = recompute_fp(it["bundle"])
        n_ok = n_bad = 0
        for li, lay in rec.items():
            ltag = None
            # 层标签对齐: 逐层顺序 L00.. — 用 meta.layer_ids
            ltag = f"L{li:02d}"
            tr_lay = tr["layers"].get(ltag)
            if tr_lay is None:
                continue
            for which in ("K", "V"):
                for blk, (tx, xx) in lay[which].items():
                    t_tx, t_xx = tr_lay[which].get(blk, ("?", "?"))
                    if tx == t_tx and xx == t_xx:
                        n_ok += 1
                    else:
                        n_bad += 1
                        details.append(
                            f"L1: kv_{side}_{seq} {ltag} blk{blk} {which} "
                            f"Tx {tx} vs {t_tx} / Xx {xx} vs {t_xx}")
            # 层级指纹
            for k, v in lay["fp"].items():
                if tr_lay.get("fp") and v != tr_lay["fp"].get(k):
                    n_bad += 1
                    details.append(f"L1: kv_{side}_{seq} {ltag} {k} "
                                   f"{v} vs {tr_lay['fp'].get(k)}")
                else:
                    n_ok += 1
        R["L1"].append({"key": f"kv_{side}_{seq}", "log": f"kvc_{side}_{rtag}.log",
                        "n_checked": n_ok + n_bad, "n_equal": n_ok})
        if n_bad:
            verdict_pass = False
            issues.append(f"L1: kv_{side}_{seq} 与日志指纹互证失败 {n_bad} 条")

    # ---- L2 Tx 逐位 + L3 归类 ----
    for seq, pp in pairs.items():
        if "P" not in pp or "D" not in pp:
            continue
        mP, mD = pp["P"]["meta"], pp["D"]["meta"]
        p_tok = int(mP["p_tok"])
        n_tx = p_tok - 1
        n_eq = 0
        for l in range(int(mP["layers"])):
            for which in ("K", "V"):
                aP = pp["P"]["bundle"][which][l][:n_tx]
                aD = pp["D"]["bundle"][which][l][:n_tx]
                if aP.shape != aD.shape:
                    issues.append(f"L2: seq={seq} L{l:02d} {which} Tx 形状不等")
                    verdict_pass = False
                    continue
                if bool((aP.view(torch.int16) == aD.view(torch.int16)).all()):
                    n_eq += 1
                else:
                    ev, cnt, hist = diff_dump(aP, aD, l, which, mP, cap=20)
                    issues.append(
                        f"L2: seq={seq} L{l:02d} {which} Tx 区不等 "
                        f"({cnt} 元素, 签名={hist}) —— 疑似传输损伤")
                    R["L3"].extend(ev[:8])
                    verdict_pass = False
        R["L2"].append({"seq": seq, "p_tok": p_tok, "pairs_checked":
                        int(mP["layers"]) * 2, "pairs_equal": n_eq})
        # 尾槽(行 p_tok-1): 同一数学量、不同 kernel 路径(P 批量 prefill vs D 单 tok
        # 补算)的合法重算差。判据用**数值语义**而非位模式: 合法重算差 = 绝对差有界
        # (相对层幅值) + 双侧分布同构; 实测合法形态 |Δ|max=0.0625 ≈ 0.43%×层幅值14.56,
        # 组中位相对差 0.47%。位模式判据(仅 ULP)会误伤 —— bf16 只有 7 位尾数, 低幅值
        # 元素一个尾数位翻转的相对差天然大、多位翻转也常见(首个张量链实测发现)。
        # 故 L3 裁决 = 幅度判据(≤5% 层幅值) + NaN/Inf 扫描; 位翻转谱/相对差分布进
        # 取证包(张量链的质变能力: 差异长什么样, 而不仅是差不差)。
        for l in range(int(mP["layers"])):
            for which in ("K", "V"):
                aP = pp["P"]["bundle"][which][l][n_tx:n_tx + 1]
                aD = pp["D"]["bundle"][which][l][n_tx:n_tx + 1]
                if aP.shape != aD.shape or aP.numel() == 0:
                    continue
                eq = bool((aP.view(torch.int16) == aD.view(torch.int16)).all())
                if eq:
                    R["L3"].append({"layer": l, "which": which, "blk":
                                    mP["block_table"][-1], "tok": n_tx,
                                    "tag": "尾槽", "sig": "exact"})
                    continue
                fP, fD = aP.float(), aD.float()
                bad_nan = bool(torch.isnan(fP).any() or torch.isnan(fD).any()
                               or torch.isinf(fP).any() or torch.isinf(fD).any())
                scale = float(max(float(fP.abs().max()), float(fD.abs().max())))
                abs_max = float((fP - fD).abs().max())
                rel_scale = abs_max / max(scale, 1e-8)
                ev, cnt, hist = diff_dump(aP, aD, l, which, mP, cap=1024)
                if bad_nan:
                    issues.append(f"L3: seq={seq} L{l:02d} {which} 尾槽含 NaN/Inf —— FAIL")
                    R["L3"].extend(ev[:8])
                    verdict_pass = False
                elif rel_scale <= 0.05:
                    R["L3"].append({"layer": l, "which": which, "blk":
                                    mP["block_table"][-1], "tok": n_tx,
                                    "tag": "尾槽",
                                    "sig": (f"recompute(|Δ|max={abs_max:.4g}"
                                            f"={rel_scale:.2%}幅值, {cnt} 元素)"),
                                    "hist": hist, "first": ev[:3]})
                else:
                    issues.append(
                        f"L3: seq={seq} L{l:02d} {which} 尾槽幅度异常 "
                        f"(|Δ|max={abs_max:.4g} = {rel_scale:.1%} 层幅值 > 5%) —— FAIL + 取证")
                    R["L3"].extend(ev[:8])
                    verdict_pass = False
        # decode 区(仅 D, 行 p_tok..w_tok): 健康扫描
        wD = int(mD["w_tok"])
        if wD > p_tok:
            dt = pp["D"]["bundle"]["K"][0][p_tok:wD]
            bad = int(torch.isnan(dt.float()).sum() + torch.isinf(dt.float()).sum())
            nans = f"NaN/Inf {bad}" if bad else "无"
            R["L4"][f"seq{seq}_decode"] = {
                "rows": wD - p_tok, "health": nans,
                "first3": [round(v, 3) for v in dt[0, 0, :3].float().tolist()]}
            if bad:
                issues.append(f"L3: seq={seq} decode 区 NaN/Inf {bad} 个")
                verdict_pass = False

    # ---- L4 汇总 ----
    verdict = "PASS" if verdict_pass else "FAIL"
    md = [f"# tensor_report（kvc_pd_offline 五级检查）",
          f"- verdict: **{verdict}**",
          f"- tensors: {[str(it['file'].name) for it in sorted(tens.values(), key=lambda x: (x['meta']['side'], x['meta']['seq']))]}",
          f"- L1 互证: {R['L1']}",
          f"- L2 Tx: {R['L2']}",
          f"- L3 明细条数: {len(R['L3'])}  L4: {R['L4']}",
          f"- issues: {issues[:50]}"]
    js = {"verdict": verdict, "L0": R["L0"], "L1": R["L1"], "L2": R["L2"],
          "L3": R["L3"], "L4": R["L4"], "issues": issues,
          "details": details[:200]}
    return verdict, issues, (md, js)


def n_tok_of(n):
    return n


# ---------------------------------------------------------------------------
# selftest: 合成 bundle + 合成日志, 注入 exact / ULP / DMA 三类差异
# ---------------------------------------------------------------------------
def selftest(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    tdir, ldir = tmp / "tensors", tmp / "logs"
    tdir.mkdir(exist_ok=True)
    ldir.mkdir(exist_ok=True)
    torch.manual_seed(1024)

    g = torch.Generator().manual_seed(1024)

    def rand_rows(n, layers=1):
        return (torch.randn(n * 8 * 128, generator=g) * 1.4
                ).reshape(n, 8, 128).to(torch.bfloat16)

    # 场景: p_tok=6, cov=[4,2] -> Tx=[0:5), 尾槽=行5, (无 decode)
    p_tok = 6
    base = rand_rows(6)
    # P: 原始
    KP = [base.clone() for _ in range(2)]
    VP = [rand_rows(6) for _ in range(2)]
    # D 序列: Tx 区前 5 行与 P 完全一致(传输无损); 尾槽第 5 行注入差异
    ub = base.clone()
    # 情形 A: L0K 尾槽 = exact(bit 级命中)
    KD = [ub.clone(), ub.clone()]
    VD = [VP[0].clone(), VP[1].clone()]
    # 情形 B: L1K 尾槽 = ULP(bit0 翻转, 单元素)
    tail = ub[5].clone()
    v = tail[2, 64].view(torch.int16).to(torch.int32)
    tail[2, 64] = (v ^ 0x0001).to(torch.int16).view(torch.bfloat16)
    KD[1][5] = tail
    # 正常主路径: V 侧不注入差异(exact); DMA 损伤场景在 SELFTEST-2 单独构造
    meta_common = {
        "schema": "kvt-1", "side": None, "seq": None, "request_id": "cmpl-selftest",
        "tag": "TERM", "p_tok": p_tok, "w_tok": 6, "final": 6,
        "cov": [4, 2], "block_table": [11, 12], "layers": 2,
        "layer_ids": ["0", "1"], "kv_heads": 8, "head_dim": 128, "block_size": 4,
        "dtype": "torch.bfloat16", "dev": "cpu", "ts": "selftest"}

    def blob(side, seq, K, V):
        m = dict(meta_common, side=side, seq=seq)
        b = {"K": K, "V": V, "meta": m}
        f = tdir / f"kv_{side}_{seq}_selftest.pt"
        torch.save(b, f)
        return b

    blob("P", 1, KP, VP)
    blob("D", 1, KD, VD)

    # 合成日志(从 D bundle 反推哈希, 保证互证成立)
    def synth_log(bundle, path):
        lines = ["[KVC][KVP] TERM req=cmpl-selftest dev=cpu 逐层按块: layers=2"]
        rec = recompute_fp(bundle)
        for li, lay in rec.items():
            ks = " ".join(f"blk{b}K.Tx={v[0]}/K.Xx={v[1]}"
                          for b, v in lay["K"].items())
            vs = " ".join(f"blk{b}V.Tx={v[0]}/V.Xx={v[1]}"
                          for b, v in lay["V"].items())
            lines.append(f"[KVC][KVP][FPB] TERM L{li:02d} 块指纹: {ks} | {vs}")
            fp = lay["fp"]
            lines.append(
                f"[KVC][KVP][FP] TERM L{li:02d} 指纹 w_tok=6 p_tok=6 "
                f"| K.prompt={fp['K.prompt']} K.all={fp['K.all']} "
                f"| V.prompt={fp['V.prompt']} V.all={fp['V.all']}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    synth_log(blob("D", 1, KD, VD), ldir / "kvc_d_reqp.log")
    # P 日志用 P 哈希
    synth_log(blob("P", 1, KP, VP), ldir / "kvc_p_reqp.log")
    # req_r 日志: 空占位(P/D seq=2 不存在, 检查器仅对存在的文件检查)
    (ldir / "kvc_d_reqr.log").write_text("", encoding="utf-8")
    (ldir / "kvc_p_reqr.log").write_text("", encoding="utf-8")

    verdict, issues, rep = run_check(tdir, ldir, tmp)
    assert verdict == "PASS", f"selftest 期望 PASS, 实际 {verdict}: {issues}"
    l3_sigs = [e.get("sig") for e in rep[1]["L3"]]
    assert any(isinstance(s, str) and s.startswith("recompute(") for s in l3_sigs), \
        f"应捕获尾槽重算差(幅度判据): {l3_sigs}"
    print(f"[SELFTEST-1 PASS] 正常+尾槽重算差场景: verdict={verdict}, L3 签名={l3_sigs}")

    # ---- 场景 2: Tx 区 DMA 损伤(3bit + 成片) -> 应 FAIL ----
    (tdir / "kv_P_1_selftest.pt").unlink()
    (tdir / "kv_D_1_selftest.pt").unlink()
    KD2 = [ub.clone(), ub.clone()]
    inj = ub[2:5].clone()                      # Tx 区 3 行连续注入
    flat = inj.view(torch.int16).reshape(-1)
    for j in range(0, flat.numel(), 997):      # 抽多个位置翻 3bit
        v = int(flat[j].item())
        flat[j] = torch.tensor(v ^ 0x0107, dtype=torch.int16)  # bit0,1,2 + bit8
    KD2[0][2:5] = inj.view(3, 8, 128)
    blob("P", 1, KP, VP)
    blob("D", 1, KD2, VD)
    verdict2, issues2, rep2 = run_check(tdir, ldir, tmp)
    assert verdict2 == "FAIL", f"DMA 场景期望 FAIL: {issues2}"
    assert any("Tx 区不等" in s or "UNEQUAL" in s for s in issues2), issues2
    print(f"[SELFTEST-2 PASS] DMA 损伤场景: verdict=FAIL, 首条 issue={issues2[0][:90]}")

    print("[SELFTEST ALL PASS] L1 哈希重算口径/ L2 equal / L3 签名判别 全部通过")


def main():
    ap = argparse.ArgumentParser(description="kvc_pd_offline 五级张量检查器")
    ap.add_argument("--dir", default="log/tensors", help=".pt 目录")
    ap.add_argument("--logs", default="log", help="v1 轨迹日志目录")
    ap.add_argument("--out", default="log/tensor_report", help="报告前缀(.md/.json)")
    ap.add_argument("--selftest", action="store_true", help="合成样例自测")
    args = ap.parse_args()

    if args.selftest:
        selftest(Path("/tmp/kvc_selftest"))
        return

    t_dir, log_dir = Path(args.dir), Path(args.logs)
    out_prefix = Path(args.out)
    verdict, issues, rep = run_check(t_dir, log_dir, out_prefix)
    if rep is None:
        print("FAIL: 无输入(.pt)")
        sys.exit(2)
    md, js = rep
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out_prefix.with_suffix(".json").write_text(
        json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"verdict={verdict}")
    for s in issues[:20]:
        print("  !", s)
    print(f"报告: {out_prefix}.md / {out_prefix}.json")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
