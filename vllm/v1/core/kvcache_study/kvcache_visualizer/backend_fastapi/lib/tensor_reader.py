"""tensor_reader — .pt 归档(kvt4-raw)读取与基础张量/统计.

exporter 与 verify 共用, 保证两侧口径一致:
    load_pt / plain / meta_json        归档装载与 JSON 化
    blocks_of / cov_of / starts_of     块表 / 覆盖 / 起始偏移
    worker_of / rebuild_layer          拓扑 / 逐层 token-major 重建
    stats_of / heat_of / b64f32        统计 / 32×32 热图 / base64 打包
    pearson / rms                      数值比较指标
"""
import base64
from pathlib import Path

import torch

import config

HEAT_N = config.HEAT_N
POOLS = config.POOLS


def load_pt(path):
    """装载 kvt4-raw 归档(CPU); 缺 K/V/meta 抛错."""
    b = torch.load(str(path), map_location="cpu", weights_only=False)
    for k in ("K", "V", "meta"):
        if k not in b:
            raise ValueError(f"非 kvt4-raw 归档(缺 {k}): {path}")
    return b


def plain(v):
    return v.tolist() if torch.is_tensor(v) else v


def meta_json(m):
    return {str(k): plain(v) for k, v in m.items()}


def blocks_of(bundle):
    return [int(b) for b in bundle["meta"]["block_table"]]


def cov_of(bundle):
    return [int(c) for c in bundle["meta"]["cov"]]


def starts_of(cov):
    """覆盖序列 -> 每块起始 token 偏移."""
    s, starts = 0, []
    for c in cov:
        starts.append(s)
        s += int(c)
    return starts


def worker_of(meta):
    return f"pp{plain(meta.get('pp', 0))}tp{plain(meta.get('tp', 0))}"


def rebuild_layer(lay, bt, cov):
    """单层块字典 -> token-major [Σcov, heads, dim](顺序按块表, 取前 cov 行)."""
    parts = []
    for e, blk in enumerate(bt):
        c = int(cov[e])
        if c > 0 and blk in lay:
            parts.append(lay[blk][:c])
    return torch.cat(parts, dim=0)


def rebuild(bundle, bt, cov):
    """逐层 token-major 重建: {pool: [层 -> [Σcov, heads, dim]]}."""
    out = {}
    for p in POOLS:
        lay = bundle[p]
        out[p] = [rebuild_layer(lay[i], bt, cov) for i in range(len(lay))]
    return out


def stats_of(t):
    f = t.float()
    return {"rms": float(f.pow(2).mean().sqrt()), "mean": float(f.mean()),
            "min": float(f.min()), "max": float(f.max())}


def heat_of(pools):
    """每层一张 [128,128](头0通道) -> 4×4 平均池化 -> 32×32; 返回 [nL,32,32]."""
    hs = []
    for t in pools:
        x = t[0:128, 0, 0:128].float()
        hs.append(x.view(HEAT_N, 4, HEAT_N, 4).mean(dim=(1, 3)))
    return torch.stack(hs)


def b64f32(t):
    return base64.b64encode(t.contiguous().numpy().tobytes()).decode()


def pearson(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def rms(t):
    return float(t.float().pow(2).mean().sqrt())