#!/usr/bin/env python3
"""seg_check.py — v1.3 SEG 三区段窗口正确性判定（按 block id 直切对比）

v1.3 归档的 SEG.pools[池] 保存三区段的前 8 个 block/slot 窗口:
  A 段 conv[j] : (8, 3, 6144) bf16           —— mamba conv 视角(独立区段)
  B 段 k[j]    : (8*ratio, 128, H, D) bf16    —— attn K 视角(kernel 行)
  B 段 ssm[j]  : (8, 16, 128, 128) fp32       —— mamba ssm 视角(同段字节叠加)
  C 段 v[j]    : 同 k                          —— attn V 视角

判据体系(NaN 不滤——NaN/mega 值本身即污染证据):
  ① 持有 attn 块 g 的 K: 写区 RMS(1024 基线≈1.4) / 未写区 RMS(基线=0)
  ② 持有 mamba slot s 的 ssm RMS(1024 基线 0.0279 量级; 2048 被踩 ≈0.3357)
  ③ conv 行 s RMS(A 段独立——独立对照, 两轮同量级)
  ④ 池净哨兵=行 0 系: K[0] 全段与未持有 ssm 行, 两轮都应≈0
     (注意: 未持有 K 块 g≥2 的字节区间=已持有 ssm 行, 位型非零属
      K(g)≡ssm(g) 同号行设计共享常态, 不作判据——§2.3)
  ⑤ 2048 专列: attn K 块覆盖的 ss 行区间提示(同段字节双写几何)

综合判定: ①写区健康 ∧ ①未写≈0 ∧ ④K哨兵≈0 ∧ ②基线 ∧ ④ssm未持≈0

用法: python3 seg_check.py <kv_pp0tp0.pt> [...]
"""
import sys

import torch


def rms(t):
    """NaN 自然传播(不滤)——2048 的 NaN/mega 值本身就是污染证据."""
    if t.numel() == 0:
        return float("nan")
    return float(t.float().double().pow(2).mean().sqrt())


def has_nan(t):
    return bool(torch.isnan(t.float()).any())


def check(path):
    b = torch.load(path, map_location="cpu", weights_only=False)
    meta = b.get("meta", {})
    seg = b.get("SEG")
    print(f"== {path}")
    if seg is None or not (seg.get("pools") or []):
        print("  [SKIP] 无 SEG 键(非 v1.3 归档)\n")
        return None
    bsz = int(meta.get("block_size", 0)
              or meta.get("attn", {}).get("block_size", 0) or 0)
    ratio = int(meta.get("kernel_ratio", 0)
                or meta.get("attn", {}).get("kernel_ratio", 0) or 1)
    cov = int((meta.get("cov") or meta.get("attn", {}).get("cov") or [0])[0])
    seg_req = meta.get("req_ids") or []
    pools = seg.get("pools") or []
    print(f"  meta: blk={bsz} ratio={ratio} cov={cov} req_ids={seg_req} "
          f"pools={len(pools)} win={seg.get('win')}")

    verdicts = []
    for j, pool in enumerate(pools[:2]):        # 判定主池 0 + 证池 1
        k, v, conv, ssm = pool["k"], pool["v"], pool["conv"], pool["ssm"]
        at_ids = sorted(int(i) for i in meta.get("attn", {}).get("block_table", []))
        mb_tbls = meta.get("mamba", {}).get("group_block_tables") or []
        slots = sorted({int(s) for t in mb_tbls for s in t})
        print(f"  ---- pool-{j}: k={tuple(k.shape)} {k.dtype} | "
              f"ssm={tuple(ssm.shape)} {ssm.dtype}")
        # ①⑤ 持有 attn K 块(写区/未写区 + V 写区观察 + 2048 越界几何)
        for g in at_ids:
            lo, hi = g * ratio, (g + 1) * ratio
            kb = k[lo:hi].reshape(bsz, *k.shape[2:])
            w_r, u_r = rms(kb[:cov]), rms(kb[cov:])
            vb = v[lo:hi].reshape(bsz, *v.shape[2:])
            line = (f"  ① K[{g}] 写区[:{cov}] RMS={w_r:.4f} 未写区 RMS={u_r:.6f}"
                    f" NaN={has_nan(kb)} | V[{g}]写区 RMS={rms(vb[:cov]):.4f}")
            if bsz > 1024:
                s0, s1 = g * (bsz // 1024), (g + 1) * (bsz // 1024)
                line += f" | ⑤越界覆盖 ss 行=[{s0},{s1})"
            print(line)
            verdicts.append(("K写区", w_r))
            verdicts.append(("K未写", u_r))
        # ②③ 持有 mamba slots
        for s in slots:
            if s >= int(ssm.shape[0]):
                continue
            sv = rms(ssm[s])
            print(f"  ② ssm[{s}] RMS={sv:.4f} (基线≈0.0279, 2048被踩≈0.3357)")
            print(f"  ③ conv[{s}] RMS={rms(conv[s]):.4f} (A 段独立, 两轮同量级)")
            verdicts.append(("ssm.", sv))
        # ④ 池净哨兵: 双组无主行应≈0 ——行 r 被 attn 块 g 覆盖(span 行,
        #   1024→1 行/块, 2048→2 行/块)或被 mamba slot 持有即"有主";
        #   有主位字节属 K(g)≡ssm(g) 同号行设计共享, 非零是常态, 不作判据(§2.3)
        span = max(1, bsz // 1024)                    # attn 块跨 ssm 行数
        owned = set(slots)
        for g in at_ids:
            owned.update(range(g * span, min((g + 1) * span, 8)))
        if 0 not in owned:                            # K[0] 哨兵仅当块 0 系无主
            fr = rms(k[0:min(ratio, int(k.shape[0]))])
            print(f"  ④ K[0](池净哨兵) 全段 RMS={fr:.6f}")
            verdicts.append(("K哨兵", fr))
        for s in range(min(8, int(ssm.shape[0]))):
            if s not in owned:
                sr = rms(ssm[s])
                print(f"  ④ ssm[{s}](无主哨兵行) RMS={sr:.6f}")
                verdicts.append(("ssm哨兵", sr))

    # 综合判定
    kw = [x for n_, x in verdicts if n_ == "K写区"]
    ku = [x for n_, x in verdicts if n_ == "K未写"]
    ks = [x for n_, x in verdicts if n_ == "K哨兵"]
    sm = [x for n_, x in verdicts if n_ == "ssm."]
    su = [x for n_, x in verdicts if n_ == "ssm哨兵"]
    ok = (bool(kw) and all(0.05 < w < 20 for w in kw)      # K 写区健康量级
          and bool(ku) and all(u < 1e-3 for u in ku)       # 持有块未写区≈0
          and bool(ks) and all(f_ < 1e-3 for f_ in ks)      # K[0] 哨兵≈0
          and bool(sm) and all(s_ < 0.1 for s_ in sm)       # ssm 基线量级
          and all(s_ < 1e-3 for s_ in su if su))            # 哨兵行 ssm≈0
    print(f"  ── 判定(pool0+1 综合): "
          f"{'✅ 正确(基线干净)' if ok else '❌ 错误(区段污染/步进失配)'}\n")
    return ok


def main():
    if len(sys.argv) < 2:
        sys.exit("用法: python3 seg_check.py <kv_pp0tp0.pt> [...]  (需 v1.3 带 SEG 归档)")
    for p in sys.argv[1:]:
        check(p)


if __name__ == "__main__":
    main()
