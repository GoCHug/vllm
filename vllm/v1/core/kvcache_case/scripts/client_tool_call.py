#!/usr/bin/env python3
"""issue #1093 (Qwen3-VL repo) 复现用例：Qwen2.5-VL tool call（vLLM + hermes parser）。

issue 现象：Qwen2.5-VL-7B 在 vLLM + tools 场景下忽略工具，输出乱码
（'addCriterion' / 表情符号 / 缺闭合 tool 标签）；同 setup 下非 VLM 的
Qwen2.5-Instruct 正常 -> VLM 特有。

社区 workaround（DavidCatalano / SamuelBG13 验证）：tool_choice=required。

判定：
  auto     态：tool_calls 为空、content 乱码或答非所问 -> REPRO（issue 形态）
  required 态：tool_calls 正确返回 get_weather(location=London) -> workaround 生效

用法：
  python client_tool_call.py --base-url http://127.0.0.1:8000 --model <served-model-name>
依赖：pip install requests
"""
import argparse

import requests

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name e.g. London",
                }
            },
            "required": ["location"],
        },
    },
}]


def ask(base_url, model, tool_choice):
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "user",
                 "content": "What is the weather like in London today?"},
            ],
            "tools": TOOLS,
            "tool_choice": tool_choice,
            "temperature": 0.0,
            "max_tokens": 256,
        }, timeout=600)
    r.raise_for_status()
    d = r.json()
    m = d["choices"][0]["message"]
    calls = None
    if m.get("tool_calls"):
        calls = [(c["function"]["name"], c["function"]["arguments"])
                 for c in m["tool_calls"]]
    return calls, m.get("content"), d["choices"][0].get("finish_reason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    print("question: What is the weather like in London today?  tool: get_weather")

    c_a, t_a, f_a = ask(a.base_url, a.model, "auto")
    print("[tool_choice=auto    ] finish=%s" % f_a)
    print("  tool_calls=%s" % (c_a,))
    print("  content=%r" % ((t_a or "")[:200],))
    c_r, t_r, f_r = ask(a.base_url, a.model, "required")
    print("[tool_choice=required] finish=%s" % f_r)
    print("  tool_calls=%s" % (c_r,))
    print("  content=%r" % ((t_r or "")[:200],))

    print("-" * 60)
    if not c_a:
        print("REPRO: auto 态未产生 tool_calls —— issue #1093 形态"
              "（7B 忽略 tools / 输出乱码）")
    else:
        print("NO-REPRO: auto 态也产生了 tool_calls（issue 未复现，"
              "检查模板与启动参数）")
    if c_r:
        print("WORKAROUND_OK: required 态 tool_calls 正常 —— "
              "社区 workaround 生效")
    else:
        print("WORKAROUND_FAIL: required 态也无 tool_calls")


if __name__ == "__main__":
    main()
