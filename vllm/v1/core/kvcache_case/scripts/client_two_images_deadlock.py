#!/usr/bin/env python3
"""Case 03 客户端：双大图请求，观察 encoder cache 调度死锁（hang）。

一个请求携带两张大图（视觉 token 数 > encoder 预算的一半，详见 case 文档），
在 `--max-num-batched-tokens 8192` 的服务上预期触发 vllm#40707 死锁：
请求永不完成、NPU 算力 0%。

用法：
  python client_two_images_deadlock.py --base-url http://127.0.0.1:8000 \
      --model Qwen/Qwen3.5-35B-A3B --images /tmp/imgs/big_0.jpg /tmp/imgs/big_1.jpg \
      --timeout 300
依赖：pip install requests
"""
import argparse
import time

import requests

PROMPT = "Compare the two images you are given and count their differences."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs=2, required=True)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    content = []
    for p in args.images:
        with open(p, "rb") as f:
            import base64
            uri = ("data:image/jpeg;base64," +
                   base64.b64encode(f.read()).decode())
        content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": PROMPT})

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    print(f"sending two-image request to {args.base_url}; "
          f"timeout={args.timeout}s ...")
    t0 = time.time()
    try:
        r = requests.post(f"{args.base_url}/v1/chat/completions",
                          json=payload, timeout=args.timeout)
        r.raise_for_status()
        d = r.json()
        print(f"OK in {time.time() - t0:.1f}s -> "
              f"{d['choices'][0]['message']['content'][:120]!r}")
        print("未复现死锁（版本可能已修，或预算参数未命中触发条件）。")
    except requests.exceptions.Timeout:
        print(f"TIMEOUT after {args.timeout}s -> 疑似 dead lock。"
              f" elapsed={time.time() - t0:.1f}s")
        print("死锁指纹（case 03 文档）：")
        print("  1. npu-smi info：AICore 算力 ~0%，显存占用高；")
        print("  2. /metrics：vllm:num_requests_running 卡在 1，TTFT 不再推进；")
        print("  3. py-spy dump --pid <api_server>：调度循环空转；")
        print("  4. 重启恢复；对照：调大 --max-num-batched-tokens 复测。")


if __name__ == "__main__":
    main()
