#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# inspect_kv_tensors.py —— kvc (v2 物理 KV 原样归档) 离线查看器
#
# 背景: 08 号补丁 v2 在每个请求结束(TERM, 恰在释放前)把该请求覆盖的全部物理块
#   整块(含未写槽位) .cpu().clone() 后 torch.save 归档。PP2×TP2 下 4 worker 各存
#   一份: kv_pp{p}tp{t}_s{seq}_{rid尾8}.pt (每 worker 独立递增 seq)。
#
# .pt 结构(bundle, schema=kvt4-raw):
#   K / V: [层序 list, 每层 dict{块号: 整块张量 (block_size, kv_heads, head_dim)}]
#          —— 与 NPU 池块逐字节同构, 保留原始 dtype(bf16), 未写槽位一并保存
#   meta:  pp/tp/seq/request_id/tag/p_tok/w_tok/final/block_size/block_table/
#          cov(每块有效槽位)/layers/layer_ids/kv_heads/head_dim/dtype/dev/ts
#
# 用法:
#   python3 inspect_kv_tensors.py --dir tensors            # ① 列出全部归档 + meta 摘要
#   python3 inspect_kv_tensors.py --dir tensors --file kv_pp0tp0_s1_*.pt
#                                                          # ② 单文件深查: 结构校验 +
#                                                          #    逐层逐块 shape/统计/首值
#   python3 inspect_kv_tensors.py --dir tensors --file ... --layer 0 --block 1 --kv K
#                                                          # ③ 看具体张量(默认前 8 行)
#   python3 inspect_kv_tensors.py --dir tensors --file ... --layer 0 --block 1 --kv K --rows 60:68
#                                                          #    指定行区间切片
#   python3 inspect_kv_tensors.py --dir tensors --compare kv_pp0tp0_s1_x.pt kv_pp0tp0_s2_y.pt
#                                                          # ④ 两归档公共块逐位比对
#                                                          #    (前缀缓存复用验证: P 种块
#                                                          #     vs R 命中块, 期望全等)
#   python3 inspect_kv_tensors.py --selftest               # 合成数据自检
#
# 说明:
#   - 有效区 = 每块前 cov 行(已写 token); cov..block_size-1 行为未写槽位(残值仅供参考)
#   - 统计均在有效区上计算(n/mean/std/min/max + NaN/Inf 扫描)
#   - 比对用 bf16 位模式(int16 视图)逐位比较, NaN 也视为相等
#   - 容器/本地均可运行(仅依赖 torch)
# ==============================================================================
import argparse
import glob
import sys
import tempfile
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def load_bundle(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    try:
        b = torch.load(str(p), map_location="cpu")
    except Exception:
        b = torch.load(str(p), map_location="cpu", weights_only=False)
    for k in ("K", "V", "meta"):
        if k not in b:
            raise ValueError(f"{p.name}: bundle 缺少键 {k}")
    return b


def rid_tail(meta) -> str:
    return str(meta.get("request_id", "?")).split("-")[-1][:8]


def cov_of(meta, blk: int) -> int:
    """该块的有效槽数(按 block_table 中的位置查 cov)。"""
    for i, b in enumerate(meta["block_table"]):
        if int(b) == int(blk):
            return int(meta["cov"][i])
    return 0


# ---------------------------------------------------------------------------
# ① 列表模式: 每文件一行 meta 摘要
# ---------------------------------------------------------------------------
def cmd_list(args):
    files = sorted(glob.glob(str(Path(args.dir) / "*.pt")))
    if not files:
        print(f"[空] {args.dir} 下无 .pt 归档")
        return 1
    print(f"== 归档列表 ({len(files)} 个, dir={args.dir}) ==")
    print(f"{'file':<28} {'pp':>2} {'tp':>2} {'s':>2} {'rid尾8':>8} "
          f"{'p_tok':>5} {'w_tok':>5} {'blocks':<16} {'cov':<14} "
          f"{'layers':>6} {'kvh':>3} {'hd':>3} {'dtype':<8} {'MiB':>6}")
    for f in files:
        m = load_bundle(f)["meta"]
        mib = Path(f).stat().st_size / 1024 / 1024
        print(f"{Path(f).name:<28} {m['pp']:>2} {m['tp']:>2} {m['seq']:>2} "
              f"{rid_tail(m):>8} {m['p_tok']:>5} {m['w_tok']:>5} "
              f"{str(m['block_table']):<16} {str(m['cov']):<14} "
              f"{m['layers']:>6} {m['kv_heads']:>3} {m['head_dim']:>3} "
              f"{m['dtype']:<8} {mib:>6.1f}")
    # 汇总: worker 覆盖与请求对账
    workers = set()
    seqs = {}
    for f in files:
        m = load_bundle(f)["meta"]
        workers.add((m["pp"], m["tp"]))
        seqs.setdefault(m["seq"], []).append((m["pp"], m["tp"]))
    print(f"== worker 覆盖: {sorted(workers)} ==")
    for s in sorted(seqs):
        print(f"   seq={s}: {sorted(seqs[s])}")
    return 0


# ---------------------------------------------------------------------------
# ② 单文件深查: 结构校验 + 逐层逐块统计
# ---------------------------------------------------------------------------
def region_stats(t: torch.Tensor) -> dict:
    x = t.reshape(-1).float()
    n = int(x.numel())
    s = float(x.sum())
    ss = float((x * x).sum())
    mean = s / n
    std = max(ss / n - mean * mean, 0.0) ** 0.5
    return {
        "n": n, "mean": mean, "std": std,
        "min": float(x.min()), "max": float(x.max()),
        "finite": bool(torch.isfinite(x).all()),
    }


def fmt_stats(st: dict) -> str:
    fin = "" if st["finite"] else " [!NaN/Inf]"
    return (f"n={st['n']} mean={st['mean']:.4g} std={st['std']:.4g} "
            f"min={st['min']:.4g} max={st['max']:.4g}{fin}")


def check_structure(b: dict, name: str) -> bool:
    m, K, V = b["meta"], b["K"], b["V"]
    problems = []
    if len(K) != m["layers"] or len(V) != m["layers"]:
        problems.append(f"层 list 长度 {len(K)}/{len(V)} != layers={m['layers']}")
    want = {int(blk) for blk, c in zip(m["block_table"], m["cov"]) if c > 0}
    bs, kh, hd = m["block_size"], m["kv_heads"], m["head_dim"]
    for li, (kd, vd) in enumerate(zip(K, V)):
        if set(kd.keys()) != want or set(vd.keys()) != want:
            problems.append(f"L{li}: 键集 {sorted(kd.keys())} != 有效块集 {sorted(want)}")
        for blk, t in kd.items():
            if tuple(t.shape) != (bs, kh, hd):
                problems.append(f"L{li} b{blk}: shape {tuple(t.shape)} != ({bs},{kh},{hd})")
        for blk, t in vd.items():
            if tuple(t.shape) != (bs, kh, hd):
                problems.append(f"L{li} b{blk}(V): shape {tuple(t.shape)} != ({bs},{kh},{hd})")
    if sum(m["cov"]) != m["w_tok"]:
        problems.append(f"sum(cov)={sum(m['cov'])} != w_tok={m['w_tok']}")
    if problems:
        print(f"[STRUCT_FAIL] {name}:")
        for p in problems[:10]:
            print(f"    - {p}")
        return False
    return True


def cmd_detail(args):
    files = [args.file] if Path(args.file).exists() \
        else sorted(glob.glob(str(Path(args.dir) / args.file)))
    if not files:
        print(f"[ERROR] 未找到文件: {args.file} (dir={args.dir})")
        return 1
    f = files[0]
    b = load_bundle(f)
    m = b["meta"]
    print(f"== {Path(f).name} 单文件深查 (schema={m.get('schema','?')}) ==")
    print(f"  worker: pp={m['pp']} tp={m['tp']} seq={m['seq']} "
          f"req尾8={rid_tail(m)} tag={m['tag']} dev={m['dev']} ts={m.get('ts','?')}")
    print(f"  token:  p_tok={m['p_tok']} w_tok={m['w_tok']} final={m['final']}")
    print(f"  pool:   block_size={m['block_size']} kv_heads={m['kv_heads']} "
          f"head_dim={m['head_dim']} dtype={m['dtype']} layers={m['layers']}")
    print(f"  池布局: K/V 两个独立张量池(张量级拆分); 第0维=块号, 第1维=token槽位, "
          f"第2维=kv_heads, 第3维=head_dim")
    print(f"  块表:   block_table={m['block_table']}")
    print(f"  覆盖:   cov={m['cov']} (有效区=每块前 cov 行; 其后为未写槽位, 保留原样)")
    ok = check_structure(b, Path(f).name)
    print(f"  结构校验: {'PASS' if ok else 'FAIL'}")

    # 逐层逐块
    print(f"== 逐层逐块 (有效区统计; K/V 各一行) ==")
    hdr = True
    for li, (kd, vd) in enumerate(zip(b["K"], b["V"])):
        if hdr:
            print(f"  L{li:02d}  " + " | ".join(
                f"b{blk}[{'满' if cov_of(m, blk) == m['block_size'] else '未满' }]"
                f"{tuple(kd[blk].shape)}" for blk in m["block_table"]
                if blk in kd))
            hdr = False
        for blk in m["block_table"]:
            if blk not in kd:
                continue
            cov = cov_of(m, blk)
            kst = region_stats(kd[blk][:cov])
            vst = region_stats(vd[blk][:cov])
            kdemo = ", ".join(f"{x:.4g}" for x in
                              kd[blk][0].reshape(-1)[:3].float().tolist())
            vdemo = ", ".join(f"{x:.4g}" for x in
                              vd[blk][0].reshape(-1)[:3].float().tolist())
            print(f"    b{blk:<6} K[{cov},{m['kv_heads']},{m['head_dim']}] "
                  f"首3=[{kdemo}] {fmt_stats(kst)}")
            print(f"    {'':<6} V[{cov},{m['kv_heads']},{m['head_dim']}] "
                  f"首3=[{vdemo}] {fmt_stats(vst)}")
            # 未写槽位残值(整块归档独有信息; 满块无)
            if cov < m["block_size"]:
                res_k = region_stats(kd[blk][cov:])
                res_v = region_stats(vd[blk][cov:])
                print(f"    {'':<6} 未写槽位[{m['block_size'] - cov}行] "
                      f"K:{fmt_stats(res_k)}")
                print(f"    {'':<6} {'':<15} "
                      f"V:{fmt_stats(res_v)}")
    print("[DONE] 深查完成 (张量切片: --layer/--block/--kv/--rows/--head)")
    return 0


# ---------------------------------------------------------------------------
# ③ 张量切片查看
# ---------------------------------------------------------------------------
def cmd_tensor(args):
    files = [args.file] if Path(args.file).exists() \
        else sorted(glob.glob(str(Path(args.dir) / args.file)))
    if not files:
        print(f"[ERROR] 未找到文件: {args.file}")
        return 1
    b = load_bundle(files[0])
    m = b["meta"]
    li = args.layer
    if li is None or li >= m["layers"]:
        print(f"[ERROR] --layer 必须指定 0~{m['layers'] - 1}")
        return 1
    pool = b[args.kv]
    valid = [blk for blk in m["block_table"] if blk in pool[li]]
    blk = args.block if args.block is not None else (valid[0] if valid else None)
    if blk is None or blk not in pool[li]:
        print(f"[ERROR] --block 无效; L{li} 现有块: {valid}")
        return 1
    t = pool[li][blk]
    cov = cov_of(m, blk)
    rows = t.shape[0]
    a, bnd = (0, min(args.head or 8, rows))
    if args.rows:
        parts = args.rows.split(":")
        a = int(parts[0]) if parts[0] else 0
        bnd = int(parts[1]) if len(parts) > 1 and parts[1] else rows
    sl = t[a:bnd]
    print(f"== {Path(files[0]).name} L{li:02d} {args.kv}[blk={blk}] "
          f"shape={tuple(t.shape)} dtype={t.dtype} 有效区=[0:{cov}) "
          f"切片=[{a}:{bnd}) ==")
    print(f"  形状注记: 第1维=token槽位(block_size={t.shape[0]}), "
          f"第2维=kv_heads({t.shape[1]}), 第3维=head_dim({t.shape[2]})")
    print(f"  每行(head_dim={t.shape[2]} 值, ', ' 分隔):")
    for r in range(sl.shape[0]):
        mark = "" if r + a < cov else "  (未写槽位)"
        vals = ", ".join(f"{x:.5g}" for x in sl[r].reshape(-1)[:16].float().tolist())
        more = " ..." if t.shape[2] > 16 else ""
        print(f"    row{r + a:>3}: [{vals}{more}]{mark}")
    return 0


# ---------------------------------------------------------------------------
# ④ 两归档公共块逐位比对(前缀缓存复用验证)
# ---------------------------------------------------------------------------
def bit_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    if t1.dtype == t2.dtype == torch.bfloat16:
        return bool(torch.equal(t1.view(torch.int16), t2.view(torch.int16)))
    return bool(torch.equal(t1, t2))


def cmd_compare(args):
    def resolve(spec):
        p = Path(spec)
        if p.exists():
            return str(p)
        g = sorted(glob.glob(str(Path(args.dir) / spec)))
        if not g:
            raise FileNotFoundError(f"未找到: {spec} (dir={args.dir})")
        return g[0]

    f1, f2 = resolve(args.compare[0]), resolve(args.compare[1])
    b1, b2 = load_bundle(f1), load_bundle(f2)
    m1, m2 = b1["meta"], b2["meta"]
    print(f"== 公共块逐位比对 ==")
    print(f"  A: {Path(f1).name} pp{m1['pp']}tp{m1['tp']} s{m1['seq']} "
          f"req尾8={rid_tail(m1)} w_tok={m1['w_tok']} blocks={m1['block_table']}")
    print(f"  B: {Path(f2).name} pp{m2['pp']}tp{m2['tp']} s{m2['seq']} "
          f"req尾8={rid_tail(m2)} w_tok={m2['w_tok']} blocks={m2['block_table']}")
    if (m1["pp"], m1["tp"]) != (m2["pp"], m2["tp"]):
        print("  [WARN] 两个归档来自不同 worker —— 比对仅对同 worker 同分片有意义")
        return 1
    common = [int(x) for x in m1["block_table"] if int(x) in set(m2["block_table"])]
    if not common:
        print("  [结果] 无公共块, 无可比")
        return 0
    print(f"  公共块: {common}")
    all_equal = True
    for blk in common:
        for li in range(min(m1["layers"], m2["layers"])):
            k1 = b1["K"][li].get(blk)
            k2 = b2["K"][li].get(blk)
            v1 = b1["V"][li].get(blk)
            v2 = b2["V"][li].get(blk)
            if k1 is None or k2 is None or v1 is None or v2 is None:
                continue
            rows_a = min(cov_of(m1, blk), cov_of(m2, blk))
            k_eq = bit_equal(k1[:rows_a], k2[:rows_a])
            v_eq = bit_equal(v1[:rows_a], v2[:rows_a])
            if not (k_eq and v_eq):
                all_equal = False
                print(f"    [DIFF] L{li:02d} b{blk}: K={'等' if k_eq else '不等'} "
                      f"V={'等' if v_eq else '不等'} (前 {rows_a} 有效行)")
                # 首个差异定位
                for tag, t1x, t2x in (("K", k1[:rows_a], k2[:rows_a]),
                                      ("V", v1[:rows_a], v2[:rows_a])):
                    neq = (t1x.view(torch.int16) != t2x.view(torch.int16))
                    if bool(neq.any()):
                        idx = neq.nonzero(as_tuple=False)[0].tolist()
                        print(f"      {tag} 首差异位置: dim{idx} "
                              f"A={t1x[tuple(idx)].item():.6g} "
                              f"B={t2x[tuple(idx)].item():.6g}")
                        break
    if all_equal:
        cells = sum(min(cov_of(m1, b), cov_of(m2, b))
                    for b in common) * m1["layers"] * m1["kv_heads"] * m1["head_dim"]
        print(f"  [PASS] 全部公共块 {common} 在 {m1['layers']} 层 K/V "
              f"前 min(cov) 有效行逐位相等 (共 {cells} 元素) —— 前缀缓存命中块"
              f"内容零篡改(复用=读共享, 不重算不覆写)")
        return 0
    print("  [FAIL] 存在差异 —— 见上方 [DIFF] 行")
    return 1


# ---------------------------------------------------------------------------
# selftest: 合成 2 worker × 2 请求归档, 验证四模式
# ---------------------------------------------------------------------------
def cmd_selftest(args):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bs, kh, hd, layers = 8, 2, 4, 2
        torch.manual_seed(7)

        def mk(pp, tp, seq, blocks, covs, mutate_blk=None):
            # 块内容由 (块号, 层号) 确定性生成 —— 同一块在不同请求/worker 间相同,
            # 模拟前缀缓存语义(同块号=同内容); mutate_blk 注入差异模拟篡改
            K, V = [], []
            for li in range(layers):
                kd, vd = {}, {}
                for blk, cov in zip(blocks, covs):
                    gk = torch.Generator().manual_seed(1000 + 17 * int(blk) + 31 * li)
                    gv = torch.Generator().manual_seed(5000 + 17 * int(blk) + 31 * li)
                    kt = torch.randn(bs, kh, hd, generator=gk).to(torch.bfloat16)
                    vt = torch.randn(bs, kh, hd, generator=gv).to(torch.bfloat16)
                    if mutate_blk == blk and li == 0:
                        kt[0, 0, 0] += 1.0
                    kd[int(blk)] = kt
                    vd[int(blk)] = vt
                K.append(kd)
                V.append(vd)
            meta = {
                "schema": "kvt4-raw", "pp": pp, "tp": tp, "seq": seq,
                "request_id": f"cmpl-test-{pp}{tp}{seq}", "tag": "TERM",
                "p_tok": sum(covs), "w_tok": sum(covs), "final": sum(covs),
                "block_size": bs, "block_table": list(map(int, blocks)),
                "cov": list(map(int, covs)), "layers": layers,
                "layer_ids": [str(i) for i in range(layers)],
                "kv_heads": kh, "head_dim": hd,
                "dtype": "torch.bfloat16", "dev": "npu:0",
                "ts": "selftest",
            }
            torch.save({"K": K, "V": V, "meta": meta},
                       str(td / f"kv_pp{pp}tp{tp}_s{seq}_test{pp}{tp}{seq}.pt"))

        # P 种块 1,2; R 命中 1,2 + 新块 3; 另一 worker 同构
        mk(0, 0, 1, [1, 2], [8, 6])
        mk(0, 0, 2, [1, 2, 3], [8, 8, 4])
        mk(1, 1, 1, [1, 2], [8, 6])
        mk(1, 1, 2, [1, 2, 3], [8, 8, 4])

        class NS:  # 简易参数命名空间
            dir = str(td)
            file = None
            layer = None
            block = None
            kv = "K"
            rows = None
            head = 8
            compare = None

        ns = NS()
        print("---- [selftest] ① 列表模式 ----")
        assert cmd_list(ns) == 0
        print("---- [selftest] ② 单文件深查 ----")
        ns.file = "kv_pp0tp0_s1_*.pt"
        assert cmd_detail(ns) == 0
        print("---- [selftest] ③ 张量切片 ----")
        ns.file = "kv_pp0tp0_s1_*.pt"
        ns.layer, ns.block, ns.kv = 0, 1, "K"
        assert cmd_tensor(ns) == 0
        print("---- [selftest] ④ 比对: 同 worker s1 vs s2 公共块应全等 ----")
        ns.compare = ["kv_pp0tp0_s1_*.pt", "kv_pp0tp0_s2_*.pt"]
        assert cmd_compare(ns) == 0
        print("---- [selftest] ⑤ 比对: 注入 1 bit 翻转后应 FAIL ----")
        mk(9, 9, 1, [1, 2], [8, 6])
        mk(9, 9, 2, [1, 2], [8, 6], mutate_blk=1)
        ns.compare = ["kv_pp9tp9_s1_*.pt", "kv_pp9tp9_s2_*.pt"]
        assert cmd_compare(ns) == 1
        print("[DONE] selftest 全部通过 (列表/深查/切片/比对/异常注入)")
        return 0


def main():
    ap = argparse.ArgumentParser(
        description="kvc v2 物理 KV 归档(.pt)查看器")
    ap.add_argument("--dir", default="tensors", help="归档目录 (默认 tensors)")
    ap.add_argument("--file", help="单文件名(glob 可)深查或切片")
    ap.add_argument("--layer", type=int, help="层序 (切片模式)")
    ap.add_argument("--block", type=int, help="块号 (切片模式, 默认首块)")
    ap.add_argument("--kv", choices=["K", "V"], default="K", help="K 或 V 池")
    ap.add_argument("--rows", help="行区间 a:b (切片模式)")
    ap.add_argument("--head", type=int, default=8, help="切片默认前 N 行")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="两归档公共块逐位比对")
    ap.add_argument("--selftest", action="store_true", help="合成数据自检")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(cmd_selftest(args))
    if args.compare:
        sys.exit(cmd_compare(args))
    if args.file and args.layer is not None:
        sys.exit(cmd_tensor(args))
    if args.file:
        sys.exit(cmd_detail(args))
    sys.exit(cmd_list(args))


if __name__ == "__main__":
    main()
