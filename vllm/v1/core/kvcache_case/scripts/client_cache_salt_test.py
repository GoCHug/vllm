#!/usr/bin/env python3
"""Case 06 客户端：cache_salt 多租户隔离验证。

顺序（前缀完全相同，仅 salt/问题不同）：
  A1: salt=tenant-A, Q1   -> 预期 miss（TTFT 基准）
  A2: salt=tenant-A, Q2   -> 预期 hit  （TTFT 明显降低）
  B1: salt=tenant-B, Q1   -> 预期 miss（salt 隔离生效）

用法：
  python client_cache_salt_test.py --base-url http://127.0.0.1:8000 \
      --model Qwen/Qwen2.5-1.5B-Instruct --prefix-tokens 1024
依赖：pip install requests
"""
import argparse
import time

import requests

QA = [
    ("Q1", "Repeat the last sentence of the passage above verbatim."),
    ("Q2", "How many sentences are in the passage above? Answer with a number."),
]


def ttft_and_tokens(base_url, model, prefix, question, salt):
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": prefix + "\n\n" + question}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    t0 = time.time()
    ttft = None
    prompt_cached = None
    with requests.post(f"{base_url}/v1/chat/completions", json=payload,
                       stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode()
            if s.startswith("data:") and ttft is None:
                ttft = time.time() - t0
            if s.startswith("data:") and '"usage"' in s:
                import json
                d = json.loads(s[5:])
                u = d.get("usage", {})
                pt = u.get("prompt_tokens_details") or {}
                prompt_cached = pt.get("cached_tokens")
    return ttft, prompt_cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-tokens", type=int, default=1024)
    args = ap.parse_args()

    # ~20 token/句，拼出足够长的稳定前缀
    sent = ("Alpha bravo charlie delta echo foxtrot golf hotel india juliet "
            "kilo lima mike november oscar papa quebec romeo sierra tango. ")
    prefix = sent * (args.prefix_tokens // 20)

    results = {}
    for tag, q in QA:
        ttft, cached = ttft_and_tokens(args.base_url, args.model, prefix, q,
                                       "tenant-A")
        results[tag] = (ttft, cached)
        print(f"{tag} salt=tenant-A  TTFT={ttft:.2f}s  cached_tokens={cached}")
    ttft_b, cached_b = ttft_and_tokens(args.base_url, args.model, prefix,
                                       QA[0][1], "tenant-B")
    results["B1"] = (ttft_b, cached_b)
    print(f"B1 salt=tenant-B  TTFT={ttft_b:.2f}s  cached_tokens={cached_b}")

    print("-" * 60)
    a1, a2, b1 = results["Q1"][0], results["Q2"][0], results["B1"][0]
    hit_a = results["Q2"][1]
    if a2 is not None and a1 is not None and b1 is not None:
        if a2 < a1 * 0.6 and b1 >= a1 * 0.8:
            print("PASS: 同 salt 二次命中（A2 提速），跨 salt 不命中（B1 不提速）。")
        elif hit_a == 0:
            print("FAIL/信息量低: A2 也未命中 -> 先确认服务版本支持 cache_salt"
                  "（0.9+），且前缀足够长（>= --prefix-tokens）。")
        else:
            print("FAIL: 命中形态异常（B1 可能也命中了）-> 检查版本是否忽略"
                  " cache_salt 字段。")
    else:
        print("数据不完整，请直接对照四次 TTFT 数值人工判读。")


if __name__ == "__main__":
    main()
