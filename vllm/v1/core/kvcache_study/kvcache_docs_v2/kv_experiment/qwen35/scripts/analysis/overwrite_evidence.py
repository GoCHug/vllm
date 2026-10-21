#!/usr/bin/env python3
"""2048 步进失配覆写证据分析: 对照 A(1024 正常)/B(2048 异常) 归档,
从 宏观RMS → token级状态 → head级污染分布 → 字节级互找 四层闭环定位覆写.

用法(在 qwen35 目录下执行):
    python3 scripts/analysis/overwrite_evidence.py
"""
from itertools import groupby
from collections import Counter

import numpy as np
import torch

NORMAL = "tensors/normal/req1_90bfa9ce/kv_pp0tp0.pt"
ABNORMAL = "tensors/abnormal/req1_abdc0ba8/kv_pp0tp0.pt"
LAYER, BLOCK, SLOT = 0, 1, 2
COV = 130  # K 实际写入 token 数(prompt + decode)


def rms(t):
    # double 防溢出: bf16 误读 fp32 指数位可产生 ~1e38 量级, f32 平方会溢出成 inf
    return float(t.float().double().pow(2).mean().sqrt())


def raw(t):
    """任意 dtype tensor -> 原始字节流. bf16 需先 view(uint16), numpy 不认 bf16."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def clean(f):
    return f[~torch.isnan(f) & ~torch.isinf(f)]


def tok_state(t):
    f = t.float()
    if bool(torch.isnan(f).any()):
        return "NaN"
    if bool(torch.isinf(f).any()):
        return "Inf"
    return "ZERO" if float(f.abs().max()) == 0.0 else "VALID"


def window_hits(src: bytes, dst: bytes, win=256, span=None):
    """在 dst 中查找 src 的非全零 256B 窗口, 返回 (命中数, 命中窗口偏移差样本)."""
    hits, offs = 0, []
    s, e = span if span else (0, len(src))
    for i in range(s, min(e, len(src)) - win + 1, win):
        w = src[i : i + win]
        if w.count(0) == len(w):
            continue  # 跳过全零窗口避免假阳性
        p = dst.find(w)
        if p >= 0:
            hits += 1
            offs.append(p - i)
    return hits, offs


def main():
    A = torch.load(NORMAL, map_location="cpu", weights_only=False)
    B = torch.load(ABNORMAL, map_location="cpu", weights_only=False)

    kn, ka = A["K"][LAYER][BLOCK], B["K"][LAYER][BLOCK]  # bf16 (1024|2048, 2, 256)
    sn, sa = A["M"][LAYER][SLOT][1], B["M"][LAYER][SLOT][1]  # fp32 (16, 128, 128)

    print("════════ 层级1 · 宏观差异 ════════")
    wa, wb = kn[:COV].float(), ka[:COV].float()
    ua, ub = kn[COV:].float(), ka[COV:].float()
    print(
        f"K 写区[0:{COV})    A: RMS={rms(wa):.4f} NaN=0 Inf=0   "
        f"B: NaN={int(torch.isnan(wb).sum())} Inf={int(torch.isinf(wb).sum())} "
        f"干净元素RMS={rms(clean(wb)):.4f}"
    )
    print(
        f"K 未写区[{COV}:)  A: RMS={rms(ua):.8f} 非零率={float((ua != 0).float().mean()):.6f}   "
        f"B: 非零率={float(((ub != 0) & ~torch.isnan(ub)).float().mean()):.4f} "
        f"NaN={int(torch.isnan(ub).sum())} Inf={int(torch.isinf(ub).sum())}"
    )
    print("→ B 轮未写区非全零(A 轮全零) = K 视角之外另有写入者(ssm) ─ 覆写证据一\n")

    print("════════ 层级2a · token 级混合指纹(fp32 位型被当 bf16 读) ════════")
    sts = [tok_state(ka[i]) for i in range(ka.shape[0])]
    runs = [(k, len(list(g))) for k, g in groupby(sts)]
    if len(runs) <= 16:
        seg = runs
    else:
        seg = runs[:8] + [("...", sum(n for _, n in runs[8:-4]))] + runs[-4:]
    print("状态段: " + "  ".join(f"{k}×{n}" for k, n in seg))
    print(f"总段数={len(runs)}   计数={dict(Counter(sts))}")
    print("→ fp32(4B) 对齐拆成 2×bf16(2B): 一半元素指数字段坍缩成 Inf/NaN → 交替指纹\n")

    print("════════ 层级2b · ssm 头级污染分布(每 head 64KiB ↔ 64 个 K token) ════════")
    print("head   ssm字节区间(K视角)    K tokens     A RMS   B RMS    倍数   判定")
    for h in range(8):
        lo = h * 65536
        tl, th = lo // 1024, (lo + 65536) // 1024 - 1
        r = rms(sa[h]) / (rms(sn[h]) + 1e-12)
        v = "★★重度污染" if r > 10 else ("★中度污染" if r > 2 else "干净")
        print(
            f"h[{h:02d}]  [{lo:6d},{lo + 65536:6d})  [{tl:4d},{th:4d}]   "
            f"{rms(sn[h]):.4f}  {rms(sa[h]):.4f}  {r:5.1f}×  {v}"
        )
    print("→ 污染集中在 K 写区命中的 head / 未命中处干净 ─ 精确相交几何证据\n")

    print("════════ 层级3 · 字节级指纹与互找(决定性) ════════")
    print(
        "K tok[64] 前8值(bf16):   "
        + ", ".join(f"{float(v):+.6f}" for v in ka[64].flatten()[:8])
    )
    print(
        "ssm h1 前8值(fp32,B轮): "
        + ", ".join(f"{float(v):+.6f}" for v in sa[1].flatten()[:8])
    )
    print(
        "ssm h1 前8值(fp32,A轮): "
        + ", ".join(f"{float(v):+.6f}" for v in sn[1].flatten()[:8])
    )
    print(f"字节hex: K-tok[64]={raw(ka[64])[:12].hex()}   ssm-h1(B)={raw(sa[1])[:12].hex()}")
    print()

    kb, sb = raw(ka), raw(sa)  # B 轮 K block 2MiB / ssm 槽 1MiB
    kk, ss = raw(kn), raw(sn)  # A 轮对照

    b_hits, b_offs = window_hits(kb, sb, 256, span=(0, COV * 1024))
    a_hits, _ = window_hits(kk, ss, 256, span=(0, COV * 1024))
    print(f"B 轮: K 写区 256B 窗口在 ssm 流命中 {b_hits} 个   偏移差样例 {b_offs[:6]}")
    print(f"A 轮对照: 同口径命中 {a_hits} 个 (预期≈0, 1024 下两者不交叠)")

    r_hits, r_offs = window_hits(sb, kb, 256)
    print(f"B 轮反向: ssm 1MiB 共 {len(sb) // 256} 窗, 命中 {r_hits} 个   偏移差样例 {r_offs[:6]}")
    if len(b_offs) >= 2:
        print(f"→ 偏移差是否恒定: {b_offs[:8]}  (恒定=可反推两视角基址相位)")

    print()
    print("── 层级3b · K 后半块(tokens 1024-2047)的写入者定位 ──")
    print(
        f"M 归档结构: mamba 层数={len(B['M'])}  "
        f"每层 slot 数={[len(B['M'][i]) for i in range(len(B['M']))]}"
    )
    half = len(kb) // 2
    tail_bytes = kb[half:]  # K 第二个 1MiB(tokens 1024-2047 地盘)
    hit_any = False
    for li in range(len(B["M"])):
        for sid, entry in B["M"][li].items():
            sb2 = raw(entry[1])
            h2, _ = window_hits(tail_bytes, sb2, 256)
            h_head, _ = window_hits(
                kb[:half], sb2, 256, span=(0, min(COV * 1024, half))
            )
            if h2 > 0 or h_head > 0:
                hit_any = True
                print(
                    f"  M[{li}][slot{sid}]: 与 K 前半(写区)命中 {h_head} / "
                    f"与 K 后半命中 {h2} 窗口"
                )
    if not hit_any:
        print("  K 后半块与所有归档 mamba slot 都不重合(写入者另有其他视图)")

    print()
    print("════════ 证据链总结 ════════")
    print("① B 轮 K 未写区非全零 vs A 轮全零 → 有外部写入(覆写)")
    print("② token 级 VALID/NaN/Inf 交替 → fp32 位型被 bf16 视角读走的指纹")
    print("③ ssm head 污染分布与 K 写区 token→head 几何映射吻合")
    print("④ ssm h1 数值 ≈ K tok[64] 数值 → 双视图同一物理字节")
    print("⑤ B 轮字节互找命中且 A 轮不命中 → 1024 不交叠 / 2048 交叠, 根因=步进失配")


if __name__ == "__main__":
    main()
