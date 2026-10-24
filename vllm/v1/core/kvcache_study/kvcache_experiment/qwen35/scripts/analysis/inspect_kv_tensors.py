#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# inspect_kv_tensors.py —— qwen35 hybrid 物理 KV 归档离线查看器(kvt5-hybrid-raw)
#
# 输入: --dir tensors  (08 号补丁 hybrid 版归档根目录, 一请求一子目录布局)
#   tensors/
#     req{seq}_{rid尾8}/     <- 一个请求一个目录
#       kv_pp{p}tp{t}.pt     <- 每 worker 一份(单卡 TP1 下 1 份; 多卡时多份)
#
# .pt bundle 结构(schema kvt5-hybrid-raw):
#   K / V: [attn 层序 list, 每层 dict{块号: (block_size, kv_heads, head_dim) 整块张量}]
#   M:     [mamba 层序 list, 每层 dict{slot号: [conv, ssm, ...] 状态张量列表}]
#   meta:  pp/tp/seq/request_id/tag/p_tok/w_tok/final
#          + attn{layers/layer_names/block_size/block_table/cov/kv_heads/head_dim/dtype}
#          + mamba{layers/layer_names/block_table/state_shapes/state_dtypes}
#          + kvt4 兼容顶层字段(block_size/block_table/cov/...)
#
# 报告内容(逐请求一节, 基础打印 + 完整落 out 文件):
#   开始加载请求 xxx 的 kvcache 物理张量(身份: request_id/tag/token 进度)
#   ── attn 组 ── 该请求使用了 block id [...] (num_blocks=N, block_size=B)
#     逐 worker -> 逐 block: 块-行映射网格(L00/L01/.../L{n-1} 各层 K_cache/V_cache
#       张量(dim0) 各占第 {blk} 行); 第 0 层示例 k/v cache shape/dtype/tensor 预览
#   ── mamba 组 ── 该请求使用了 state slot [...] (每 slot=状态张量 dim0 行号)
#     逐 worker -> 逐 slot: 网格(各层 M 张量第 {slot} 行, 即该层此请求的整套状态);
#       第 0 层示例 conv/ssm shape/dtype/tensor 预览(RMS)
#   (tensor 预览 = 前 N 维; 完整张量请 torch.load)
#
# 用法:
#   python3 scripts/analysis/inspect_kv_tensors.py --dir tensors                  # 报告 -> logs/analysis/inspect_kv_tensors.out
#   python3 scripts/analysis/inspect_kv_tensors.py --dir tensors --out xx.out     # 指定 out 产物
#   python3 scripts/analysis/inspect_kv_tensors.py --dir tensors --preview 8      # tensor 预览前 8 值
# ==============================================================================
import argparse
import sys
import time
from pathlib import Path

import torch

W = 78   # 报告横线宽度


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def load_bundle(path):
    """ 加载并校验 bundle(K/V/M/meta 四键, M 允许缺席=纯 attn 模型兼容)。"""
    b = torch.load(str(path), map_location="cpu", weights_only=False)
    if not (isinstance(b, dict) and "K" in b and "V" in b and "meta" in b):
        raise ValueError(f"非 kvc 归档结构(缺 K/V/meta): {path}")
    return b


def fmt_vals(t, n):
    """张量前 n 个值 -> [v1, v2, ...] 字符串(bf16/fp32 友好 %.4g)。"""
    vals = t.reshape(-1)[:n].tolist()
    return "[" + ", ".join(f"{v:+.4g}" for v in vals) + ", ...]"


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def grid_rows(prefix, names):
    """块-行映射网格行头(逐层名)。"""
    return [f"{prefix}{i:02d}({nm.split('.')[-1][:14] if nm else '?'})"
            for i, nm in enumerate(names)] or [f"{prefix}00"]


# ---------------------------------------------------------------------------
# 报告主逻辑
# ---------------------------------------------------------------------------
def report(args) -> int:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"[ERROR] 归档目录不存在: {root}")
        return 2

    req_dirs = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("req")])
    if not req_dirs:
        print(f"[ERROR] {root} 下无请求子目录(req*): {root}")
        return 2

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    def say(s=""):
        lines.append(s)

    def attn_block_section(f: Path, m: dict):
        """attn 组一节: 块-行映射网格 + 第 0 层 K/V 预览。"""
        bt = m.get("block_table", [])
        if not bt:
            say("  [attn] 无 attn 组块表(纯 mamba 归档?)")
            return
        say(f"  [attn] 该请求使用了 block id {bt}  (num_blocks={len(bt)}, "
            f"block_size={m['block_size']})")
        say(f"  [attn] 块有效槽位 cov={m.get('cov')}  "
            f"(kv_heads={m.get('kv_heads')}, head_dim={m.get('head_dim')}, "
            f"含未写槽位原样归档)")
        b = load_bundle(f)
        K, V = b["K"], b["V"]
        grid = grid_rows("L", m.get("layer_ids", []))
        say(f"  [attn] 块-行映射: 1 个 block 竖跨 {len(K)} 层 —— "
            + " ".join(grid[:6]) + (" ..." if len(grid) > 6 else ""))
        bsz = m["block_size"]
        say(f"  [attn] 第 0 层示例: block id={bt[0]} 对应张量第 {bt[0]} 行, 张量信息如下")
        k0, v0 = K[0].get(bt[0]), V[0].get(bt[0])
        if k0 is not None:
            say(f"    K_cache[{bt[0]}]: shape={tuple(k0.shape)} dtype={k0.dtype} "
                f"RMS={rms(k0):.4f} 预览{fmt_vals(k0, args.preview)}")
            say(f"    V_cache[{bt[0]}]: shape={tuple(v0.shape)} dtype={v0.dtype} "
                f"RMS={rms(v0):.4f} 预览{fmt_vals(v0, args.preview)}")
            w = (m.get("cov") or [bsz])[0]
            if 0 < w < bsz:
                say(f"    首块 cov={w}/{bsz}: 未写区 RMS={rms(k0[w:]):.6f} (应≈0, "
                    f"快照含未写槽位原样取证)")
        say(f"    (其余 {len(K) - 1} 层同构; 每层块表相同 "
            f"{all(list(kd.keys()) == list(K[0].keys()) for kd in K) if K else '?'})")

    def mamba_block_section(f: Path, m: dict):
        """mamba 组一节: slot 网格 + 第 0 层 conv/ssm 预览。"""
        mb = m.get("mamba") or {}
        if not mb or not mb.get("layer_names"):
            say("  [mamba] 无 mamba 组(纯 attn 归档 / kvt4 旧格式) —— 跳过")
            return
        sbt = mb.get("block_table", [])
        say(f"  [mamba] 该请求使用了 state slot {sbt}  (num_slots={len(sbt)}; "
            f"每 slot = 一整套递推状态, 无 token 维度)")
        say(f"  [mamba] states: shapes={mb.get('state_shapes')} "
            f"dtypes={mb.get('state_dtypes')}  (qwen3.5-0.8B: conv (3,6144) bf16 + "
            f"ssm (16,128,128) fp32)")
        b = load_bundle(f)
        M = b.get("M") or []
        if not M:
            say("  [mamba] bundle 无 M 数据 —— 跳过")
            return
        grid = grid_rows("L", mb.get("layer_names", []))
        say(f"  [mamba] 块-行映射: 1 个 slot 竖跨 {len(M)} 层 —— "
            + " ".join(grid[:6]) + (" ..." if len(grid) > 6 else ""))
        for slot in sbt:
            if M[0].get(int(slot)) is not None:
                states = M[0][int(slot)]
                parts = []
                for i, t in enumerate(states):
                    nm = (mb.get("state_shapes") or ["?"] * len(states))[i]
                    parts.append(f"state{i}({tuple(nm) if not isinstance(nm, tuple) else nm})")
                say(f"  [mamba] slot={slot}: " + " + ".join(parts))
        say(f"  [mamba] 第 0 层示例: slot={sbt[0]} 对应张量第 {sbt[0]} 行, 张量信息如下")
        for i, t in enumerate(M[0][sbt[0]]):
            say(f"    state[{i}]: shape={tuple(t.shape)} dtype={t.dtype} "
                f"RMS={rms(t):.4f} 预览{fmt_vals(t, args.preview)}")
        say(f"    (其余 {len(M) - 1} 层同构; conv/ssm 与 attn K/V 共用同一物理页, "
            f"块表/组语义各自独立)")

    def worker_section(f: Path):
        b = load_bundle(f)
        m = b["meta"]
        say("-" * W)
        say(f"  worker 归档: {f.name}  (pp={m.get('pp')} tp={m.get('tp')} "
            f"dev={m.get('dev')} ts={m.get('ts')})")
        attn_block_section(f, m)
        mamba_block_section(f, m)
        say()

    # ---------------- 报告头 ----------------
    say("=" * W)
    say("qwen35 hybrid 物理 KV Cache 张量离线查看报告(08 hybrid: attn K/V + mamba conv/ssm)")
    say(f"归档目录: {root} | 请求: {len(req_dirs)} 个 | "
        f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say("=" * W)
    for rd in req_dirs:
        say(f"  {rd.name}/  ({len(list(rd.glob('kv_*.pt')))} 个 worker 归档)")
    say()

    # ---------------- 逐请求报告 ----------------
    for i, rd in enumerate(req_dirs, 1):
        try:
            files = sorted(rd.glob("kv_pp*.pt"))
            if not files:
                say(f"[WARN] {rd.name}/ 下无 kv_*.pt, 跳过")
                say()
                continue
            m0 = load_bundle(files[0])["meta"]
        except Exception as e:
            say(f"[WARN] 加载 {rd.name} 首个归档失败, 跳过: {e}")
            say()
            continue

        say("=" * W)
        say(f"[{i}/{len(req_dirs)}] 开始加载请求 {rd.name} ({m0.get('request_id')}) 的 "
            f"kvcache 物理张量")
        say("=" * W)
        say(f"  request_id : {m0.get('request_id')}")
        say(f"  tag={m0.get('tag')}  prompt={m0.get('p_tok')} tok  "
            f"written={m0.get('w_tok')} tok  final={m0.get('final')}  "
            f"schema={m0.get('schema')}")
        attn_meta = m0.get("attn") or {}
        mamba_meta = m0.get("mamba") or {}
        say(f"  组结构: attn {attn_meta.get('layers', m0.get('layers'))} 层 × "
            f"{len(attn_meta.get('block_table', m0.get('block_table', [])))} 块; "
            f"mamba {mamba_meta.get('layers')} 层 × "
            f"{len(mamba_meta.get('block_table', []))} slot")
        say()
        for f in files:
            try:
                worker_section(f)
            except Exception as e:
                say(f"[WARN] {f.name} 解析失败, 跳过: {e}")
                say()

    say("=" * W)
    say("[END] 报告完毕 —— tensor 预览为首 token 首 head 前 "
        f"{args.preview} 值(第 0 层); 完整张量请 torch.load(<归档路径>)['K'/'V'/'M']")

    # ---------------- 落盘 + 终端回显 ----------------
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"[DONE] 报告已落盘: {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="qwen35 hybrid 归档查看器(kvt5-hybrid-raw: attn K/V + mamba conv/ssm)")
    ap.add_argument("--dir", default="tensors",
                    help="tensors 归档根目录(默认 tensors, 含 req*/ 子目录)")
    ap.add_argument("--out", default="logs/analysis/inspect_kv_tensors.out",
                    help="输出 out 产物路径(默认 logs/analysis/inspect_kv_tensors.out)")
    ap.add_argument("--preview", type=int, default=4,
                    help="tensor 预览前 N 值(默认 4)")
    args = ap.parse_args()
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
