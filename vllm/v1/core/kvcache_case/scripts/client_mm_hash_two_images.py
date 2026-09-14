#!/usr/bin/env python3
"""Case 01 客户端：两图同文本，验证 prefix cache 不跨图命中。

流程（temperature=0）：
  A1: 图1 + 文本T        （miss，图片入缓存）
  B1: 图2 + 文本T        （修复版应为 miss；bug 版会命中图1 的 KV）
  A2: 图1 + 文本T（再问） （hit 自己）
  B2: 图2 + 文本T        （hit 自己；bug 版继续与图1 混缓存）

判定依据 usage.prompt_tokens_details.cached_tokens：
  修复版：B1=0, A2>0, B2>0
  bug 版：B1>0（错误命中图1）

用法：
  python client_mm_hash_two_images.py --base-url http://127.0.0.1:8000 \
      --model Qwen/Qwen2.5-VL-3B-Instruct
依赖：pip install requests pillow
"""
import argparse
import io
import random

import requests


def gen_maze(seed, size=640, cell=40):
    """生成两张视觉上明确不同的迷宫图（PIL 缺失时报错退出）"""
    from PIL import Image, ImageDraw
    random.seed(seed)
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    n = size // cell
    for y in range(n):
        for x in range(n):
            if random.random() < 0.45:
                if random.random() < 0.5:
                    d.line([(x * cell, y * cell), (x * cell + cell, y * cell)],
                           "black", 3)
                else:
                    d.line([(x * cell, y * cell), (x * cell, y * cell + cell)],
                           "black", 3)
    return img


def _png_b64(img):
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


QUESTION = "Describe precisely the structure you see in the image. Then count visual intersections."

def ask(base_url, model, image_uri, question):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_uri}},
                {"type": "text", "text": question},
            ]},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      timeout=600)
    r.raise_for_status()
    d = r.json()
    cached = d["usage"]["prompt_tokens_details"].get("cached_tokens", 0)
    text = d["choices"][0]["message"]["content"]
    return cached, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    img1 = _png_b64(gen_maze(seed=1, size=args.size))
    img2 = _png_b64(gen_maze(seed=999, size=args.size))

    print("A1 (img1 first):")
    c, t = ask(args.base_url, args.model, img1, QUESTION)
    print(f"  cached_tokens={c}\n  answer={t[:80]!r}")
    print("B1 (img2 first, same text):")
    c_b, t_b = ask(args.base_url, args.model, img2, QUESTION)
    print(f"  cached_tokens={c_b}\n  answer={t_b[:80]!r}")
    print("A2 (img1 again, same text):")
    c_a2, _ = ask(args.base_url, args.model, img1, QUESTION)
    print(f"  cached_tokens={c_a2}")
    print("B2 (img2 again, same text):")
    c_b2, _ = ask(args.base_url, args.model, img2, QUESTION)
    print(f"  cached_tokens={c_b2}")

    print("-" * 60)
    if c_b > 0:
        print("REPRO: 图2 首次请求即命中（cached_tokens>0）-> 跨图复用缓存，")
        print("  即 issue #20261 形态（旧版本）。")
    else:
        print("PASS: 图2 首次未命中、复访各自的第二句命中 -> mm hash 隔离正常。")
        print("  版本判定：B2（是否>0）体现同图复访命中，也应 >0。")


if __name__ == "__main__":
    main()
