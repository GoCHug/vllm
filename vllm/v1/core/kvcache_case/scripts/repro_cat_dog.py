#!/usr/bin/env python3
"""Case 01 复现脚本：猫狗图片 + 相同文本，验证 prefix cache 是否跨图命中。

流程（temperature=0）：
  A1: 猫图 + "What is this?"     （miss，猫图入缓存）
  B1: 狗图 + "What is this?"     （修复版应为 miss；bug 版会命中猫图的 KV → 回答"猫"）
  A2: 猫图 + "What is this?"     （hit 自己）
  B2: 狗图 + "What is this?"     （hit 自己）

判定：
  bug 版：B1 cached_tokens>0 且 B1 回答"cat"（跨图命中，张冠李戴）
  修复版：B1 cached_tokens=0 且 B1 回答"dog"，A2/B2 cached_tokens>0

用法：
  python repro_cat_dog.py --base-url http://127.0.0.1:8000 --model <model_path>
依赖：pip install requests
"""
import argparse
import io
import base64

import requests


def make_image(color_name):
    """生成一张纯色图片（不依赖 PIL，用最小依赖）。
    如果 PIL 可用则生成更复杂的图；否则用 1x1 像素的 PNG。
    """
    try:
        from PIL import Image
        colors = {
            "cat": (255, 100, 100),   # 红色调
            "dog": (100, 100, 255),   # 蓝色调
        }
        rgb = colors.get(color_name, (128, 128, 128))
        img = Image.new("RGB", (336, 336), rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # 无 PIL：用 1x1 纯色 PNG
        # 构造最小合法 PNG（3字节 RGB pixel）
        import struct
        import zlib
        colors = {
            "cat": (255, 0, 0),
            "dog": (0, 0, 255),
        }
        r, g, b = colors.get(color_name, (128, 128, 128))
        # PNG 签名
        sig = b'\x89PNG\r\n\x1a\n'
        # IHDR
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        # IDAT
        raw = b'\x00' + bytes([r, g, b])
        compressed = zlib.compress(raw)
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
        # IEND
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        png = sig + ihdr + idat + iend
        return "data:image/png;base64," + base64.b64encode(png).decode()


QUESTION = "What animal do you see in this image? Answer in one word."


def ask(base_url, model, image_uri, question, dump_usage=False):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": image_uri}},
                {"type": "text", "text": question},
            ]},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      timeout=600)
    r.raise_for_status()
    d = r.json()
    usage = d.get("usage", {})
    ptd = usage.get("prompt_tokens_details") or {}
    cached = ptd.get("cached_tokens", 0) if isinstance(ptd, dict) else 0
    text = d["choices"][0]["message"]["content"]
    if dump_usage:
        import json as _json
        print(f"  [debug] usage={_json.dumps(usage, ensure_ascii=False)}")
    return cached, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--debug", action="store_true",
                    help="打印完整 usage 结构")
    args = ap.parse_args()

    cat_img = make_image("cat")
    dog_img = make_image("dog")

    print("=" * 60)
    print("Case 01 复现：猫狗图片 + 相同文本，验证 prefix cache 跨图命中")
    print("=" * 60)

    print("\nA1 (cat first):")
    c_a1, t_a1 = ask(args.base_url, args.model, cat_img, QUESTION, dump_usage=args.debug)
    print(f"  cached_tokens={c_a1}")
    print(f"  answer={t_a1!r}")

    print("\nB1 (dog, same text):")
    c_b1, t_b1 = ask(args.base_url, args.model, dog_img, QUESTION, dump_usage=args.debug)
    print(f"  cached_tokens={c_b1}")
    print(f"  answer={t_b1!r}")

    print("\nA2 (cat again):")
    c_a2, t_a2 = ask(args.base_url, args.model, cat_img, QUESTION, dump_usage=args.debug)
    print(f"  cached_tokens={c_a2}")
    print(f"  answer={t_a2!r}")

    print("\nB2 (dog again):")
    c_b2, t_b2 = ask(args.base_url, args.model, dog_img, QUESTION, dump_usage=args.debug)
    print(f"  cached_tokens={c_b2}")
    print(f"  answer={t_b2!r}")

    print("\n" + "=" * 60)
    print("判定结果：")
    print("-" * 60)

    if c_b1 > 0:
        print("[REPRO] B1 cached_tokens>0 -> 跨图命中！")
        print(f"  B1 用狗图却命中了猫图的缓存（cached_tokens={c_b1}）")
        if "cat" in t_b1.lower():
            print(f'  B1 回答="cat"（张冠李戴）-> 确认 issue #20261')
        else:
            print(f"  B1 回答={t_b1!r}（可能是乱码或重复 token）")
    elif c_a2 > 0 and c_b2 > 0:
        print("[PASS] B1 未命中，A2/B2 命中自己 -> mm hash 隔离正常")
        print(f"  A2 cached_tokens={c_a2}, B2 cached_tokens={c_b2}")
    elif c_a2 == 0 and c_b2 == 0:
        print("[WARN] 所有请求 cached_tokens=0，prefix cache 可能未启用")
        print("  请检查:")
        print("    1) 是否设置 VLLM_USE_V1=1（V0 对多模态禁用 prefix caching）")
        print("    2) 启动是否加了 --enable-prompt-tokens-details（否则该字段恒为 null）")
        print("    3) 加 --debug 查看完整 usage 结构")
    else:
        print(f"[UNKNOWN] A2={c_a2}, B2={c_b2}，需人工分析")


if __name__ == "__main__":
    main()
