#!/usr/bin/env python3
"""Case 02 客户端：cache hit 与 miss 数值路径一致性检测（温度 0）。

串行发 N 次同一请求（temperature=0, logprobs 回收 top-1）：
  Run1  cache miss（全量 prefill）
  Run2+ cache hit（复用前缀 + 只算新 token）
对比每次输出的 token 序列与 top-1 logprob，定位首个分叉位置与最大偏差。

用法：
  python client_hit_vs_miss_divergence.py --base-url http://127.0.0.1:8000 \
      --model Qwen/Qwen3-0.6B --runs 5 --max-tokens 96
依赖：pip install requests
"""
import argparse
import json

import requests

PROMPT = (
    "Question: In one rigorous paragraph, explain why the European Union "
    "expanded eastward in 2004, naming three concrete institutional "
    "mechanisms. Answer:\n"
)


def one_run(base_url, model, max_tokens, seed):
    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "logprobs": 1,
        "echo": False,
    }
    r = requests.post(f"{base_url}/v1/completions", json=payload, timeout=600)
    r.raise_for_status()
    d = r.json()
    choice = d["choices"][0]
    tokens = [t["text"] for t in choice.get("logprobs", {}).get("tokens", [])]
    lps = [
        vals[next(iter(vals))]["logprob"]
        for vals in [t["top_logprobs"][0] for t in
                     choice.get("logprobs", {}).get("tokens", [])]
    ]
    return choice["text"], tokens, lps


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()

    runs = []
    for i in range(args.runs):
        # 固定 seed，排除采样器引入的随机性
        text, toks, lps = one_run(args.base_url, args.model, args.max_tokens,
                                  seed=1234)
        runs.append((text, toks, lps))
        print(f"[run {i + 1}] tokens={len(toks)} text={text[:64]!r}...")

    base = runs[0]
    print("-" * 60)
    print(f"run1 = baseline (cache MISS); run2..{args.runs} = cache HIT path")
    all_same = True
    for i in range(1, len(runs)):
        r = runs[i]
        dp = first_divergence(base[1], r[1])
        if dp is not None:
            all_same = False
            print(f"[run {i + 1}] DIVERGED at token {dp}: "
                  f"{base[1][dp]!r} (miss) vs {r[1][dp]!r} (hit)")
        if len(r[2]) and len(base[2]):
            n = min(len(base[2]), len(r[2]))
            diff = max(abs(a - b) for a, b in zip(base[2][:n], r[2][:n]))
            print(f"[run {i + 1}] max top-1 logprob diff = {diff:.4f}")
    print("-" * 60)
    if all_same:
        print("PASS: hit/miss 两条数值路径在此 NPU 内核组合下输出一致。")
        print("  （负结果同样有归档价值，换内核/关图捕获后再验证）")
    else:
        print("REPRO: 复现 hit/miss 分叉（对应 vllm#33123 的 NPU 版）。")
        print("  对照组：服务端加 --no-enable-prefix-caching 后相同实验，")
        print("  若全部一致则锁定为前缀复用路径的数值差异。")


if __name__ == "__main__":
    main()
