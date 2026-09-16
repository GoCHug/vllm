#!/usr/bin/env python3
"""issue #1093 顺序效应验证：fgolemo 的观察（Qwen2.5-VL-7B + vLLM）。

三种时序，全部 tool_choice=auto：
  A. 直接发 tool use（无前置请求）      fgolemo: bad
  B. 先发 [图片请求，无 tools]，再 tool use  fgolemo: bad
  C. 先发 [图片请求，带 tools]，再 tool use  fgolemo: good

现象一旦复现 -> 说明"前置请求内容影响后继 tool call 输出"，
与 vllm#20261 的"高并发乱码"现象同源（该报告人就是这么关联的）。

用法：
  python client_seq_order.py --base-url http://127.0.0.1:8000 --model <model>
依赖：pip install requests pillow
"""
import argparse
import base64
import io

import requests

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name e.g. London"}
            },
            "required": ["location"],
        },
    },
}]


def png_b64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (448, 448), (200, 30, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(base_url, model, messages, tools=None, tool_choice=None):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    d = r.json()
    m = d["choices"][0]["message"]
    calls = None
    if m.get("tool_calls"):
        calls = [(c["function"]["name"], c["function"]["arguments"])
                 for c in m["tool_calls"]]
    return calls, (m.get("content") or "")[:150]


def tool_use(base_url, model):
    return chat(base_url, model,
                [{"role": "user", "content": "What is the weather like in London today?"}],
                tools=TOOLS, tool_choice="auto")


def image_req(base_url, model, with_tools):
    content = [
        {"type": "image_url", "image_url": {"url": png_b64()}},
        {"type": "text", "text": "Briefly describe this image in one sentence."},
    ]
    if with_tools:
        return chat(base_url, model, [{"role": "user", "content": content}], tools=TOOLS)
    return chat(base_url, model, [{"role": "user", "content": content}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    print("[A] tool use directly (no prior request)")
    c, t = tool_use(a.base_url, a.model)
    print("    tool_calls=%s content=%r" % (c, t))

    print("[B] image (no tools) -> then tool use")
    c0, t0 = image_req(a.base_url, a.model, with_tools=False)
    print("    pre  image: tool_calls=%s content=%r" % (c0, t0))
    c, t = tool_use(a.base_url, a.model)
    print("    next tool : tool_calls=%s content=%r" % (c, t))

    print("[C] image (with tools payload) -> then tool use")
    c0, t0 = image_req(a.base_url, a.model, with_tools=True)
    print("    pre  image: tool_calls=%s content=%r" % (c0, t0))
    c, t = tool_use(a.base_url, a.model)
    print("    next tool : tool_calls=%s content=%r" % (c, t))


if __name__ == "__main__":
    main()
