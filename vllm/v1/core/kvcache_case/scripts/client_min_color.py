#!/usr/bin/env python3
"""Case 01 最简复现用例：两张纯色图 + 同一问题，两步判定跨图串读。

图1 = 纯红 448x448；图2 = 纯蓝 448x448（Qwen2.5-VL 下各约 256 个视觉 token，
prompt ~275 tokens ≈ 17 个完整块；块足够多，跨图命中才会显著）。
问题："What is the dominant color of the image? Answer with one word."（temperature=0）

判定（B1 = 蓝图首次请求）：
  修复态（mm 内容哈希参与缓存键）：B1 cached_tokens=0，答案 blue（正常看图）
  bug 态  （mm 内容不参与缓存键）  ：B1 cached_tokens>0，答案 red （串读红图 1）

注意：图不宜太小——128x128 纯色图（prompt ~45 tokens、2 个完整块）在
0.9.1 上不出现跨图命中（小图边缘 case，命中机制上不去），448 及以上可靠。

用法：
  python client_min_color.py --base-url http://127.0.0.1:8000 --model <served-model-name>
依赖：pip install requests pillow
"""
import argparse
import base64
import io

import requests

Q = "What is the dominant color of the image? Answer with one word."


def png_b64(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (448, 448), rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(base_url, model, uri):
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": Q},
            ]}],
            "max_tokens": 16,
            "temperature": 0.0,
        }, timeout=600)
    r.raise_for_status()
    d = r.json()
    usage = d.get("usage", {})
    ptd = usage.get("prompt_tokens_details") or {}
    cached = ptd.get("cached_tokens", 0) if isinstance(ptd, dict) else 0
    return cached, usage.get("prompt_tokens", -1), d["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    red = png_b64((255, 0, 0))
    blue = png_b64((0, 0, 255))

    c1, p1, t1 = ask(a.base_url, a.model, red)
    print(f"A1 red  : prompt_tokens={p1:4d}  cached_tokens={c1:4d}  answer={t1!r}")
    c2, p2, t2 = ask(a.base_url, a.model, blue)
    print(f"B1 blue : prompt_tokens={p2:4d}  cached_tokens={c2:4d}  answer={t2!r}")

    print("-" * 60)
    if c2 > 0 and "red" in t2.lower():
        print("=> REPRO: 蓝图请求命中红图缓存且答 red —— 跨图串读（issue #20261 形态）")
    elif c2 > 0:
        print(f"=> REPRO(cached_tokens={c2}): 蓝图跨图命中红图缓存；答案 {t2!r} 非本人内容")
    elif c2 == 0 and t2:
        print("=> PASS: 蓝图未命中红图缓存，答案正常 —— mm 隔离生效")
    else:
        print("=> WARN/UNKNOWN: 请检查 V1 引擎（VLLM_USE_V1=1）与 --enable-prompt-tokens-details")


if __name__ == "__main__":
    main()
