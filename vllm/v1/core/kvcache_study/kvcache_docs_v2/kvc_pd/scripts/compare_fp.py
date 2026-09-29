#!/usr/bin/env python3
# ==============================================================================
# compare_fp.py —— P/D 双侧 KVP 指纹自动比对, 产出正确性实验裁决 log/verdict.txt
#
# 输入: log/kvc_p_req*.log / log/kvc_d_req*.log (curl_pd.sh 提取的 [KVC] 轨迹)
# 逐层解析 [FPB] 块指纹(blk{N}K.Tx=/K.Xx= 与 V 同构) 与 [FP] 层指纹,
# P/D 成对对账:
#   Tx = 传输区指纹(前 p_tok-1 tok, D 侧异步加载覆盖范围) —— 必须逐位一致
#   Xx = 本块全部已写槽位指纹 —— P/D 各自写入者不同(prefill vs 本地重算/decode)
#
# 裁决规则:
#   [PASS] 所有层所有块 Tx 指纹 P/D 全等  => mooncake 传输逐位无损
#          Xx 不等仅出现在"prompt 尾块 + decode 新块" => 归因本地生成, 非传输损伤
#   [FAIL] 任何 Tx 指纹不等 => 传输路径丢字节/错位
# ==============================================================================
import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "log"

FPB_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FPB\] (\w+) (L\d+) 块指纹: (.*) \| (.*)")
FP_RE = re.compile(
    r"\[KVC\]\[KVP\]\[FP\] (\w+) (L\d+) 指纹 w_tok=(\d+) p_tok=(\d+) "
    r"\| K\.prompt=(\S+) K\.all=(\S+) \| V\.prompt=(\S+) V\.all=(\S+)")
BLK_RE = re.compile(r"blk(\d+)[KV]\.Tx=([0-9a-f-]+)/[KV]\.Xx=([0-9a-f-]+)")


def parse(path: Path):
    """-> {layer: {"tag":.., "w_tok":.., "p_tok":.., "K": {blk: (tx, xx)}, "V": {...}}}"""
    if not path.exists():
        return None
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = FPB_RE.search(line)
        if m:
            tag, layer, kpart, vpart = m.groups()
            d = out.setdefault(layer, {"K": {}, "V": {}, "fp": None})
            d["tag"] = tag
            for which, part in (("K", kpart), ("V", vpart)):
                for blk, tx, xx in BLK_RE.findall(part):
                    d[which][int(blk)] = (tx, xx)
            continue
        m = FP_RE.search(line)
        if m:
            tag, layer, w, p, kp, ka, vp, va = m.groups()
            d = out.setdefault(layer, {"K": {}, "V": {}, "fp": None})
            d["w_tok"], d["p_tok"] = int(w), int(p)
            d["fp"] = {"K.prompt": kp, "K.all": ka,
                       "V.prompt": vp, "V.all": va}
    return out


def cmp_req(name_p: str, name_d: str):
    P, D = parse(LOG / name_p), parse(LOG / name_d)
    lines = []
    if not P or not D:
        return [f"!! 缺文件: {name_p}={bool(P)} {name_d}={bool(D)}"], False
    title = f"{name_p} vs {name_d}"
    lines += [f"===== {title} =====",
              f"层集合: P={len(P)} D={len(D)}"]
    ok = True
    layers = sorted(set(P) & set(D), key=lambda s: int(s[1:]))
    p_ptok = next(iter(P.values())).get("p_tok")
    d_ptok = next(iter(D.values())).get("p_tok")
    lines += [f"p_tok: P={p_ptok} D={d_ptok}  传输区 Tx = 双侧各取前 p_tok-1={int(p_ptok)-1} tok"]

    tx_total = tx_match = 0
    xx_rows = []          # (layer, blk, which, P_xx, D_xx, explain)
    for layer in layers:
        pd_, dd_ = P[layer], D[layer]
        for which in ("K", "V"):
            for blk in sorted(set(pd_[which]) & set(dd_[which])):
                ptx, pxx = pd_[which][blk]
                dtx, dxx = dd_[which][blk]
                tx_total += 1
                if ptx == dtx:
                    tx_match += 1
                else:
                    ok = False
                    xx_rows.append((layer, blk, which, ptx, dtx, "TX-MISMATCH"))
                if pxx != dxx:
                    if blk * 128 >= int(p_ptok) - 1:  # 尾块或 decode 新块
                        expl = "尾块/本地重算槽位或 decode 新块(预期)"
                        xx_rows.append((layer, blk, which, pxx, dxx, expl))
                    else:
                        ok = False
                        xx_rows.append((layer, blk, which, pxx, dxx,
                                        "Xx-MISMATCH-传输区内(异常!)"))
    # FP 层指纹对账(参考信息)
    fp_kp = sum(1 for l in layers
                if P[l]["fp"] and D[l]["fp"]
                and P[l]["fp"]["K.prompt"] == D[l]["fp"]["K.prompt"])
    lines += [f"Tx 指纹对账: {tx_match}/{tx_total} 对全等",
              f"层指纹 K.prompt 全等层数: {fp_kp}/{len(layers)}"]
    if xx_rows:
        lines.append(f"Xx 不等明细 ({len(xx_rows)} 条, 均应落于尾块/decode 块):")
        for layer, blk, which, a, b, expl in xx_rows[:80]:
            lines.append(f"  {layer} blk{blk}{which}: P={a} D={b}  ({expl})")
    verdict = (tx_match == tx_total) and ok
    lines.append(f"--> 裁决: {'[PASS] mooncake 传输逐位无损, 差异均来自本地生成区' if verdict else '[FAIL] 传输区指纹存在不一致!'}")
    return lines, verdict


def main():
    out, allok = [], True
    pairs = [("kvc_p_req1.log", "kvc_d_req1.log"),
             ("kvc_p_req2.log", "kvc_d_req2.log")]
    for p, d in pairs:
        if (LOG / p).exists() and (LOG / d).exists():
            lines, ok = cmp_req(p, d)
            out += lines + [""]
            allok = allok and ok
    out.append(f"====== 总结论: {'PASS' if allok else 'FAIL'} ======")
    text = "\n".join(out)
    (LOG / "verdict.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
