"""kvms_source — msprobe tensor dump 数据源发现（目录扫描 + 形状反推元信息）。

目录约定（msprobe task=tensor 的产物）：
    <KVMS_ROOT>/<name>_dump_tensor/step{N}/rank0/dump_tensor_data/*.pt

元信息全部从张量形状反推，无需额外配置：
    num_blocks      = conv_state.shape[0]
    chunks_per_page = key_cache.shape[0] / num_blocks
    block_tokens    = chunks_per_page × PHYS_BLOCK(128)
"""
import re
from pathlib import Path

import torch

import config

NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+_dump_tensor$")
STEP_RE = re.compile(r"^step(\d+)$")

# dump 里的算子文件名模板（{i} = 层内调用序号）
ATTN_KEY = "NPU._npu_reshape_and_cache.{i}.forward.kwargs.key_cache.pt"
ATTN_VAL = "NPU._npu_reshape_and_cache.{i}.forward.kwargs.value_cache.pt"
ATTN_SLOT = "NPU._npu_reshape_and_cache.{i}.forward.kwargs.slot_indices.pt"
FIA_KEY = "NPU.npu_fused_infer_attention_score.{i}.forward.kwargs.key.pt"
FIA_VAL = "NPU.npu_fused_infer_attention_score.{i}.forward.kwargs.value.pt"
GDN_CONV = "_C_ascend.npu_causal_conv1d_custom.{i}.forward.kwargs.conv_state.pt"
GDN_CIDX = "_C_ascend.npu_causal_conv1d_custom.{i}.forward.kwargs.cache_indices_opt.pt"
GDN_SSM = "_C_ascend.npu_recurrent_gated_delta_rule.{i}.forward.kwargs.state.pt"
GDN_SIDX = "_C_ascend.npu_recurrent_gated_delta_rule.{i}.forward.kwargs.ssm_state_indices.pt"


def tinfo(path):
    """只取 shape/dtype/字节数（mmap，不真正读数据）。"""
    t = torch.load(str(path), map_location="cpu", mmap=True)
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "bytes": t.numel() * t.element_size()}


def resolve(source_id, root=None):
    """source_id → 目录（含目录穿越校验）。"""
    root = Path(root or config.KVMS_ROOT).resolve()
    if not NAME_RE.match(source_id or ""):
        raise ValueError(f"非法数据源名: {source_id}")
    target = (root / source_id).resolve()
    if target.parent != root:
        raise ValueError(f"路径越界: {source_id}")
    if not target.is_dir():
        raise FileNotFoundError(source_id)
    return target


def steps_of(dump_dir):
    """有张量落盘的 step 列表（step10 这类只有 json 的不会被计入）。"""
    out = []
    for d in Path(dump_dir).iterdir():
        m = STEP_RE.fullmatch(d.name)
        if not m:
            continue
        data = d / "rank0" / "dump_tensor_data"
        if data.is_dir() and any(data.glob("*.pt")):
            out.append(int(m.group(1)))
    return sorted(out)


def data_dir(dump_dir, step):
    return Path(dump_dir) / f"step{step}" / "rank0" / "dump_tensor_data"


def dir_bytes(dump_dir):
    return sum(p.stat().st_size for p in Path(dump_dir).rglob("*.pt"))


def _fmt_size(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def probe(dump_dir):
    """反推数据源元信息；形状缺失时对应字段为 None。"""
    dump_dir = Path(dump_dir)
    steps = steps_of(dump_dir)
    # view_steps：真正能展示「写后」的 step —— 最后一个 step 没有下一步可借写前快照，
    # 所以排除（有张量的 step 数 - 1）
    info = {"id": dump_dir.name, "steps": steps,
            "view_steps": steps[:-1] if len(steps) > 1 else list(steps),
            "empty": not steps}
    if not steps:
        info["size"] = "0 B"
        return info

    num_blocks = chunks = None
    attn_page_bytes = ssm_page_bytes = conv_page_bytes = None

    # conv 先扫一遍拿 num_blocks —— attn 侧每块字节要除以它
    for st in steps:
        f = data_dir(dump_dir, st) / GDN_CONV.format(i=0)
        if f.is_file():
            cs = tinfo(f)
            num_blocks = cs["shape"][0]
            conv_page_bytes = 1
            for d in cs["shape"][1:]:
                conv_page_bytes *= d
            conv_page_bytes *= 2          # bf16
            info["conv_shape"] = cs["shape"]
            break

    # ssm 只在 decode step 才 dump（prefill 不跑 gated_delta_rule），必须往后找，
    # 否则会把 attn 的页大小当成 ssm，2048 时误差一倍
    for st in steps:
        base = data_dir(dump_dir, st)
        if attn_page_bytes is None and (base / ATTN_KEY.format(i=0)).is_file():
            kc = tinfo(base / ATTN_KEY.format(i=0))
            info["key_cache_shape"] = kc["shape"]
            if num_blocks:
                chunks = kc["shape"][0] // num_blocks
                attn_page_bytes = kc["bytes"] // num_blocks
        if ssm_page_bytes is None and (base / GDN_SSM.format(i=0)).is_file():
            stt = tinfo(base / GDN_SSM.format(i=0))
            ssm_page_bytes = 1
            for d in stt["shape"][1:]:
                ssm_page_bytes *= d
            ssm_page_bytes *= 4           # fp32
            info["ssm_shape"] = stt["shape"]
            info["ssm_from_step"] = st
        if attn_page_bytes and ssm_page_bytes:
            break

    block_tokens = chunks * config.PHYS_BLOCK if chunks else None
    info.update({
        "num_blocks": num_blocks,
        "chunks_per_page": chunks,
        "block_tokens": block_tokens,
        # attn_k / ssm 是两个视角各自的块大小：attn 随 block_size 变，ssm 固定；
        # k_ssm 是两者共用的那段物理内存，按较大者算
        "page_bytes": {
            "conv": conv_page_bytes,
            "attn_k": attn_page_bytes,
            "ssm": ssm_page_bytes,
            "k_ssm": max(attn_page_bytes or 0, ssm_page_bytes or 0) or None,
            "v": attn_page_bytes,
        },
        "size": _fmt_size(dir_bytes(dump_dir)),
        "label": (f"block_tokens {block_tokens} · {num_blocks} 页" if block_tokens
                  else "形状未能反推"),
    })
    return info


def list_sources(root=None):
    """扫描根目录下所有可用数据源，按名称排序。"""
    root = Path(root or config.KVMS_ROOT)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and NAME_RE.match(d.name):
            out.append(probe(d))
    return out