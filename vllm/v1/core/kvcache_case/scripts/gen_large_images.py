#!/usr/bin/env python3
"""Case 03 辅助：批量生成分辨率极大的不同图片（3024x4032 量级）。

依赖：pip install pillow
"""
import argparse
import random

from PIL import Image, ImageDraw


def gen(w, h, seed, path):
    random.seed(seed)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    cell = 84
    for y in range(0, h - cell, cell):
        for x in range(0, w - cell, cell):
            r = random.random()
            if r < 0.2:
                d.line([(x, y), (x + cell, y)], "black", 5)
            elif r < 0.4:
                d.line([(x, y), (x, y + cell)], "black", 5)
            elif r < 0.45:
                d.ellipse([x + 10, y + 10, x + cell - 10, y + cell - 10],
                          fill="black")
    img.save(path, format="JPEG", quality=85)
    print(f"wrote {path} ({w}x{h})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=3024)
    ap.add_argument("--h", type=int, default=4032)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="/tmp/imgs")
    args = ap.parse_args()
    import os
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        gen(args.w, args.h, seed=i + 1,
            path=os.path.join(args.out, f"big_{i}.jpg"))


if __name__ == "__main__":
    main()
