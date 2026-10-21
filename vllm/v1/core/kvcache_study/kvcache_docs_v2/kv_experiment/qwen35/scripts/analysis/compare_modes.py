#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# compare_modes.py —— qwen35 hybrid 归档对比器(A 基线侧 vs B 被检侧)
#
# 输入: --normal   <dir>   (A 侧归档根 = 正常/修复轮, 含 req*/kv_pp0tp0.pt)
#       --abnormal <dir>   (B 侧归档根 = 异常/被检轮)
#
# 三轮实验(docs/0_hybrid_kvcache_record.md §8)的用法 — A 侧传"更正常的那轮", B 侧传"被检的那轮":
#   ① 根因分析  : --normal tensors/normal    --abnormal tensors/abnormal  (基线 vs 2048 污染)
#   ② 修复效果  : --normal tensors/fixed     --abnormal tensors/abnormal  (2048+15408 修复 vs 污染)
#   ③ 修复后基线: --normal tensors/normal    --abnormal tensors/fixed     (修复后应与原生一致, 判定应 PASS)
#
# 目的: 定位 --block-size 2048 精度异常的 cache 根因 —— 同 prompt 同 token 数的
#   两轮归档并排比对:
#   ① meta 对照: block_size / 块表 / cov / 层数 / mamba states(不变量 K 段宽==SSM 段宽);
#   ② attn K/V 首块双列 stats;
#   ③ K 已写区对比: 前 w 个 token 的逐元素差 / Pearson(两轮同 prompt, K 分布应一致;
#      B 失真即 Q·K^T 失真的上游证据);
#   ④ K 未写槽区 RMS(★ 根因直接证据): A 新块基线应≈0; B 若显著非零 =
#      mamba conv/ssm 写入落进 attn 视图未写区(页内重叠区污染, README §2 根因链);
#   ⑤ mamba conv/ssm 对照: RMS + 元素差(检验 attn K 写入是否反向污染 mamba 状态)。
#
# 输出: 终端 + logs/analysis/compare_modes.out
# 用法:
#   python3 scripts/analysis/compare_modes.py --normal tensors/normal --abnormal tensors/abnormal
#   (两轮归档按 README §7.4 规程隔离保存后再比对)
# ==============================================================================
import argparse
import sys
import time
from pathlib import Path

import torch

W = 78


def load_first(root: Path):
    """加载归档根下第一个 req*/kv_pp*.pt 的 bundle + 目录/文件名。"""
    rd = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("req")])
    if not rd:
        raise FileNotFoundError(f"{root} 下无 req*/ 子目录")
    files = sorted(rd[0].glob("kv_pp*.pt"))
    if not files:
        raise FileNotFoundError(f"{rd[0]} 下无 kv_pp*.pt")
    b = torch.load(str(files[0]), map_location="cpu", weights_only=False)
    for k in ("K", "V", "meta"):
        if k not in b:
            raise ValueError(f"非归档结构(缺 {k}): {files[0]}")
    return b, rd[0].name, files[0].name


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def stats_line(say, tag, t):
    if t is None or t.numel() == 0:
        say(f"  {tag}: (无数据)")
        return
    tf = t.float()
    say(f"  {tag}: shape={tuple(t.shape)} dtype={t.dtype} RMS={rms(t):.5f} "
        f"mean={tf.mean():+.4g} |min|={tf.abs().min():.4g} |max|={tf.abs().max():.4g}")


def pearson(x, y):
    x = x.float().flatten()
    y = y.float().flatten()
    if x.numel() < 2 or x.numel() != y.numel():
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    s = (xm.pow(2).sum() * ym.pow(2).sum()).sqrt()
    return float((xm * ym).sum() / s) if s > 0 else float("nan")


def get_meta(m, leaf, top="attn"):
    """meta 顶层 kvt4 兼容字段优先级: groups 节 > 兼容顶层。"""
    return (m.get(top) or {}).get(leaf, m.get(leaf))


def report(args) -> int:
    n_dir, a_dir = Path(args.normal), Path(args.abnormal)
    nb, nname, nfile = load_first(n_dir)
    ab, aname, afile = load_first(a_dir)
    nm, am = nb["meta"], ab["meta"]

    lines = []

    def say(s=""):
        lines.append(s)

    say("=" * W)
    say("qwen35 hybrid 归档对比: A(基线侧 --normal) vs B(被检侧 --abnormal)")
    say(f"A normal  : {n_dir}/{nname}/ ({nfile})  schema={nm.get('schema')}")
    say(f"B abnormal: {a_dir}/{aname}/ ({afile})  schema={am.get('schema')}")
    say(f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * W)

    # ---------------- ① meta 对照 ----------------
    say("① meta 对照(不变量 K段宽==SSM段宽 的物理载体; 同 prompt 两轮请求)")
    rows = [
        ("block_size(attn 组)", "block_size", "attn"),
        ("attn 层数", "layers", "attn"),
        ("attn 块表", "block_table", "attn"),
        ("attn cov(每块已写槽)", "cov", "attn"),
        ("kv_heads / head_dim", None, None),
        ("mamba 层数", "layers", "mamba"),
        ("mamba state shapes", "state_shapes", "mamba"),
        ("mamba state dtypes", "state_dtypes", "mamba"),
    ]
    for name, leaf, top in rows:
        if leaf is None:
            say(f"  {name}: A=({get_meta(nm, 'kv_heads')}, {get_meta(nm, 'head_dim')}) "
                f"B=({get_meta(am, 'kv_heads')}, {get_meta(am, 'head_dim')})")
        else:
            say(f"  {name}:")
            say(f"    A: {get_meta(nm, leaf, top)}")
            say(f"    B: {get_meta(am, leaf, top)}")

    # ★ mamba 组级 slot 表(08 v1.2 meta.group_block_tables; 旧 v1.1 归档从 M 层键
    #   按组还原: M 迭代序=组优先, qwen3.5= 3 mamba 组 × 6 层) —— 4 组 block id
    #   分配全景: attn 组先(组 0), mamba A/B/C 按注册序从共享池顺序取号
    def group_tables(bundle, mm):
        gts = get_meta(mm, "group_block_tables", "mamba")
        if gts:
            return list(gts)
        M = bundle.get("M") or []
        n = (len(M) + 5) // 6
        return [sorted({s for li in range(g * 6, min((g + 1) * 6, len(M)))
                       for s in M[li].keys()}) for g in range(n)] if M else []
    say("  mamba 组级 slot 表(4 组 block id 全景):")
    say(f"    A: attn={list(get_meta(nm, 'block_table', 'attn') or [])} "
        f"mambaA/B/C={group_tables(nb, nm)}")
    say(f"    B: attn={list(get_meta(am, 'block_table', 'attn') or [])} "
        f"mambaA/B/C={group_tables(ab, am)}")
    say()

    # ---------------- 张量定位: 首 attn 层首块 ----------------
    nbt = list(get_meta(nm, "block_table", "attn") or [])
    abt = list(get_meta(am, "block_table", "attn") or [])
    ncov = list(get_meta(nm, "cov", "attn") or [])
    acov = list(get_meta(am, "cov", "attn") or [])
    if not nbt or not abt:
        say("[ERROR] 任一轮缺 attn 块表, 无法比对")
        return 3
    kn = nb["K"][0].get(int(nbt[0]))
    ka = ab["K"][0].get(int(abt[0]))
    vn = nb["V"][0].get(int(nbt[0]))
    va = ab["V"][0].get(int(abt[0]))
    if kn is None or ka is None:
        say("[ERROR] 两轮 K 首块任一缺失, 跳过张量对比")
        return 3

    bsz_n, bsz_a = int(kn.shape[0]), int(ka.shape[0])
    wn = int(ncov[0]) if ncov else bsz_n
    wa = int(acov[0]) if acov else bsz_a
    w = min(wn, wa)

    # ---------------- ② 首块 K/V stats ----------------
    say("② attn K/V 首块对照(首 attn 层, 同 prompt 同写入 token 序)")
    stats_line(say, "A K[blk0]", kn)
    stats_line(say, "B K[blk0]", ka)
    stats_line(say, "A V[blk0]", vn)
    stats_line(say, "B V[blk0]", va)
    say(f"  块粒度: A bsz={bsz_n} (块内已写 {wn}) | B bsz={bsz_a} (块内已写 {wa})")
    say()

    # ---------------- ③ 已写区对比 ----------------
    say(f"③ K 已写区对比(前 w={w} 个 token = 两轮共同写入段)")
    stats_line(say, "A K[:w]", kn[:w])
    stats_line(say, "B K[:w]", ka[:w])
    diff = kn[:w].float() - ka[:w].float()
    say(f"  逐元素差: mean={diff.mean():+.4g} RMS={rms(diff):.5g} |max|={diff.abs().max():.4g}"
        f"  Pearson(A,B)={pearson(kn[:w], ka[:w]):.6f}")
    say("  (Pearson≈1 + 差≈ULP: 分布一致; Pearson 低/差显著: B 读回的 K 已失真)")
    say()

    # ---------------- ④ 未写槽区 RMS(根因直接证据) ----------------
    say("④ K 未写槽区 RMS(★ 根因直接证据: A 基线应≈0; B 污染则显著非零)")
    r_un_n = rms(kn[wn:]) if wn < bsz_n else 0.0
    r_un_a = rms(ka[wa:]) if wa < bsz_a else 0.0
    say(f"    A 未写区 [{wn}:{bsz_n}]: RMS={r_un_n:.6f}")
    say(f"    B 未写区 [{wa}:{bsz_a}]: RMS={r_un_a:.6f}")
    if wa < bsz_a:
        lo, hi = 36, min(1040, bsz_a)
        if hi > lo and hi > wa:
            seg = ka[max(wa, lo):hi]
            say(f"    B 页内重叠段提示: mamba conv/ssm 物理落点 ≈ 块内 token[{lo}:{hi}]"
                f" 区间, 该段 RMS={rms(seg):.6f}")
    say()

    # ---------------- ⑤ mamba 状态对照 ----------------
    say("⑤ mamba 状态张量对照(首 mamba 层首 slot; 检验 attn K 写入是否反向污染 mamba)")
    M_n, M_a = nb.get("M") or [], ab.get("M") or []
    ns = (get_meta(nm, "block_table", "mamba") or [])
    asl = (get_meta(am, "block_table", "mamba") or [])
    if M_n and M_a and ns and asl:
        st_n = (M_n[0].get(int(ns[0])) or [])
        st_a = (M_a[0].get(int(asl[0])) or [])
        for i, (tn, ta) in enumerate(zip(st_n, st_a)):
            stats_line(say, f"A state[{i}]", tn)
            stats_line(say, f"B state[{i}]", ta)
            if tn.shape == ta.shape:
                d = tn.float() - ta.float()
                say(f"  逐元素差: mean={d.mean():+.4g} RMS={rms(d):.5g} "
                    f"Pearson={pearson(tn, ta):.6f}")
            say()
    else:
        say("  (缺 mamba M 数据, 跳过 —— 两轮均无 M 时先确认 08 补丁版本)")
    say()

    # ---------------- 判定 ----------------
    say("=" * W)
    say("判定(对应 README §2 根因链: K段宽≠SSM段宽 -> 页内重叠区两族互写 -> 读回失真 -> 乱码)")
    say(f"  A 未写区 RMS={r_un_n:.6f} -> " + ("基线干净 PASS" if r_un_n < 1e-3 else "CHECK: A 基线未全零"))
    if r_un_a > max(1e-3, 10 * max(r_un_n, 1e-9)):
        say(f"  B 未写区 RMS={r_un_a:.6f} ≫ A -> 页内重叠区污染证据成立 ★")
    else:
        say(f"  B 未写区 RMS={r_un_a:.6f} 未显著异常 -> 未见 mamba 溢出 attn 未写区,"
            f" 结合 ③ Pearson 与 ⑤ mamba 对照判断")
    pr = pearson(kn[:w], ka[:w])
    say(f"  K 已写区 Pearson(A,B)={pr:.6f} -> " + ("两轮分布一致" if pr > 0.999 else "B 相对 A 已失真"))
    say("  结论请结合: 响应体(logs/curl_resp.json) + 启动日志 L1 页几何 + 本报告四组数值")
    say("=" * W)

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[DONE] 对比报告已落盘: {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="qwen35 hybrid 两模式归档对比(A 1024 正常 vs B 2048 异常; 根因分析)")
    ap.add_argument("--normal", required=True, help="正常轮归档根(含 req*/)")
    ap.add_argument("--abnormal", required=True, help="异常轮归档根(含 req*/)")
    ap.add_argument("--out", default="logs/analysis/compare_modes.out",
                    help="输出报告路径(默认 logs/analysis/compare_modes.out)")
    args = ap.parse_args()
    try:
        return report(args)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
