#!/usr/bin/env python3
"""读取 msprobe tensor dump 中每个 .pt 的 shape / dtype / 体积。

用法:
    python3 inspect_pt.py <dump_dir> [step ...]
    python3 inspect_pt.py ../1024_dump_tensor            # 默认 step0 step1
    python3 inspect_pt.py ../2048_dump_tensor 0 1 5      # 指定 step
    python3 inspect_pt.py ../1024_dump_tensor all        # 全部 step

目录约定（msprobe 产物）:
    <dump_dir>/step{N}/rank0/dump_tensor_data/<算子名>.forward.<参数>.pt
"""
import glob
import os
import re
import sys

import torch


def natural_key(path):
    """让 layer 序号按数值排序（.2 排在 .10 前面）。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def size_str(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024


def inspect_step(step_dir):
    files = sorted(glob.glob(os.path.join(step_dir, "*.pt")), key=natural_key)
    if not files:
        print("  (无 .pt 文件)")
        return 0, 0, 0

    total_elems = total_bytes = 0
    print(f"  {'文件':<74} {'shape':<24} {'dtype':<14} {'体积':>10}")
    print("  " + "-" * 124)
    for path in files:
        name = os.path.basename(path)
        display = name[:73] + "…" if len(name) > 74 else name
        try:
            obj = torch.load(path, map_location="cpu", mmap=True)
        except Exception:
            try:
                obj = torch.load(path, map_location="cpu")
            except Exception as exc:
                print(f"  {display:<74} <读取失败: {type(exc).__name__}>")
                continue

        if isinstance(obj, torch.Tensor):
            nbytes = obj.numel() * obj.element_size()
            total_elems += obj.numel()
            total_bytes += nbytes
            print(f"  {display:<74} {str(tuple(obj.shape)):<24} {str(obj.dtype):<14} {size_str(nbytes):>10}")
        else:
            print(f"  {display:<74} <{type(obj).__name__}>")

    return len(files), total_elems, total_bytes


def resolve_steps(dump_dir, args):
    available = sorted(
        int(m.group(1))
        for d in os.listdir(dump_dir)
        if (m := re.fullmatch(r"step(\d+)", d)) and os.path.isdir(os.path.join(dump_dir, d))
    )
    if not args:
        return [s for s in (0, 1) if s in available]
    if args[0] == "all":
        return available
    wanted = []
    for a in args:
        try:
            wanted.append(int(a))
        except ValueError:
            print(f"[WARN] 忽略非法 step 参数: {a}")
    return [s for s in wanted if s in available]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dump_dir = sys.argv[1].rstrip("/")
    if not os.path.isdir(dump_dir):
        print(f"[ERROR] 目录不存在: {dump_dir}")
        sys.exit(1)

    steps = resolve_steps(dump_dir, sys.argv[2:])
    if not steps:
        print(f"[ERROR] 没有可用的 step 目录（{dump_dir}）")
        sys.exit(1)

    print(f"dump 目录: {dump_dir}")
    print(f"扫描 step: {steps}\n")

    grand_files = grand_elems = grand_bytes = 0
    for step in steps:
        step_dir = os.path.join(dump_dir, f"step{step}", "rank0", "dump_tensor_data")
        print(f"=== step{step} ===")
        if not os.path.isdir(step_dir):
            print("  (目录不存在)\n")
            continue
        n, elems, nbytes = inspect_step(step_dir)
        grand_files += n
        grand_elems += elems
        grand_bytes += nbytes
        print(f"  小计: {n} 个文件, {elems} 元素, {size_str(nbytes)}\n")

    print("=" * 60)
    print(f"合计: {grand_files} 个文件, {grand_elems} 元素, {size_str(grand_bytes)}")


if __name__ == "__main__":
    main()