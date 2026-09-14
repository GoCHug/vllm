#!/usr/bin/env python3
"""Case 05 客户端：abort 风暴 + KV cache 泄漏不变量检测。

三阶段（详见 case 文档）：
  Phase1 基线：全部请求正常完成，静置后记录 baseline usage
  Phase2 风暴：多轮并发，其中 --abort-ratio 比例的请求在收到首个流式
              chunk 后主动断连（模拟客户端 abort）
  Phase3 静置：停发 --idle-sec 秒，观察 usage 是否回落

判定（打印为摘要，人工/CI 皆可用）：
  idle_usage <= base_usage + --tolerance   -> PASS
  否则                                     -> 疑似泄漏（幽灵块）

monitor.csv 逐秒记录 (ts, gpu_cache_usage_perc, prefix_cache_hits, queries)。

用法：
  python client_abort_storm.py --base-url http://127.0.0.1:8000 \
      --model Qwen/Qwen2.5-0.5B-Instruct --rounds 6 --concurrency 24 \
      --abort-ratio 0.7 --idle-sec 120 --output usage.csv
依赖：pip install requests
"""
import argparse
import csv
import random
import re
import threading
import time

import requests

SENT = ("The quick brown fox jumps over the lazy dog. The weather today is "
        "surprisingly mild and the city market opens at nine. ")
QUESTION = " Summarize everything above in exactly two sentences."


def build_prompt(prefix_tokens):
    # 估算：该句子 ~20 token，按需重复
    n = max(1, prefix_tokens // 20)
    return SENT * n + QUESTION


class Monitor(threading.Thread):
    NAMES = ("gpu_cache_usage_perc", "prefix_cache_hits",
             "prefix_cache_queries")

    def __init__(self, base_url, out_csv, every=2.0, stop_evt=None):
        super().__init__(daemon=True)
        self.base_url, self.every, self.stop_evt = base_url, every, \
            stop_evt or threading.Event()
        self.rows = []
        self._f = open(out_csv, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["time", "usage", "hits", "queries"])

    @staticmethod
    def _parse(metrics_text, name):
        # 兼容 Prometheus counter 的 _total 后缀
        pat = re.compile(rf"^vllm:{name}(_total)?(\{{.*?\}})?\s+(.+)$", re.M)
        for m in pat.finditer(metrics_text):
            try:
                return float(m.group(3).strip())
            except ValueError:
                if m.group(1):
                    continue
        return None

    def sample(self):
        try:
            t = requests.get(f"{self.base_url}/metrics", timeout=5).text
        except Exception:
            return
        vals = [self._parse(t, n) for n in self.NAMES]
        self.rows.append((time.time(), *vals))
        self._w.writerow([f"{time.time():.0f}", *vals])
        self._f.flush()

    def run(self):
        while not self.stop_evt.is_set():
            self.sample()
            time.sleep(self.every)
        self.sample()   # 收尾一帧
        self._f.close()

    def values(self, idx):
        return [r[idx] for r in self.rows if r[idx] is not None]

    def last(self, idx):
        vals = self.values(idx)
        return vals[-1] if vals else None


def worker(base_url, model, prompt, abort, out_queue):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                          stream=True, timeout=(5, 600))
        got_first = False
        for _ in r.iter_lines():
            got_first = True
            if abort:
                break
        if abort:
            time.sleep(random.uniform(0.2, 2.0))
        r.close()
        out_queue.append(("aborted" if abort else "done", got_first))
    except Exception as e:  # noqa: BLE001
        out_queue.append(("error", str(e)[:80]))


def run_round(args, prompt, abort_ratio):
    results = []
    threads = []
    for i in range(args.concurrency):
        abort = random.random() < abort_ratio
        t = threading.Thread(target=worker,
                             args=(args.base_url, args.model, prompt,
                                   abort, results))
        t.start()
        threads.append(t)
        if i % 4 == 0:
            time.sleep(0.05)   # 错峰打入
    for t in threads:
        t.join()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--abort-ratio", type=float, default=0.7)
    ap.add_argument("--prefix-tokens", type=int, default=1500)
    ap.add_argument("--idle-sec", type=int, default=120)
    ap.add_argument("--tolerance", type=float, default=0.2,
                    help="prefix cache LRU 驻留容忍量（usage 绝对值）")
    ap.add_argument("--output", default="usage.csv")
    args = ap.parse_args()

    prompt = build_prompt(args.prefix_tokens)
    mon = Monitor(args.base_url, args.output)
    mon.start()

    print("Phase1 基线（全部正常完成）...")
    run_round(args, prompt, abort_ratio=0.0)
    time.sleep(20)
    base_usage = mon.last(1)
    hits0 = mon.last(2) or 0
    print(f"  baseline usage = {base_usage}")

    print(f"Phase2 风暴（abort-ratio={args.abort_ratio}）...")
    peak = 0.0
    for r in range(args.rounds):
        res = run_round(args, prompt)
        aborted = sum(1 for s, _ in res if s == "aborted")
        errs = sum(1 for s, _ in res if s == "error")
        us = mon.last(1)
        peak = max(peak, us or 0)
        print(f"  round {r + 1}: aborted={aborted}/{len(res)} err={errs} "
              f"usage={us} hits={mon.last(2)}")

    print(f"Phase3 静置 {args.idle_sec}s ...")
    time.sleep(args.idle_sec)
    idle_usage = mon.last(1)
    hits1 = mon.last(2) or 0

    print("=" * 64)
    print(f"baseline_usage={base_usage}  peak_usage={peak:.3f}  "
          f"idle_usage={idle_usage}")
    print(f"prefix_cache_hits 增量 = {hits1 - hits0:.0f} "
          f"（queries={mon.last(3)}）")
    if base_usage is None or idle_usage is None:
        print("判定：指标缺失，请检查服务是否开了 /metrics。")
        return
    if idle_usage <= base_usage + args.tolerance:
        print(f"PASS: 静置后 usage 回落至基线(+{args.tolerance} 容忍)以内，"
              f"未发现幽灵块。建议进 CI 并在 hybrid/多模态模型上复测。")
    else:
        print("疑似泄漏：静置后 usage 未回落。下一步：")
        print("  1) 关 --enable-prefix-caching 复测（二分变量）；")
        print("  2) 复测时让部分请求带多模态输入（encoder cache 泄漏面）；")
        print("  3) 对 free/touch 调用链加日志（case 05 文档 '根因教学' 节）。")


if __name__ == "__main__":
    main()
