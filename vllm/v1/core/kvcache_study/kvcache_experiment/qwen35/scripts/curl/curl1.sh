#!/bin/bash
# curl 测试请求脚本（单请求版; 需先启动 qwen3.5 服务, 端口 8340）
#   与 curl.sh 唯一差异: content 前置 "请你" 两字(其余请求体完全一致), 作对照请求
# KV 观测适配: 请求体带 ignore_eos=true 且一次吐满 max_tokens —— 保证请求以 length
#   结束(而非提前 stop), 08 号补丁的 TERM 判定(written>=final)才会触发归档;
#   若提前 EOS 停止则请求结束只见 LATE 告警不归档(块已还池, 非证据级)。
#   prompt 短(约几十 token): attn 组单块未满+未写槽位原样归档, mamba 组 1 slot;
#   需满块/多块观测可仿此请求体自行加长 content。
# 响应落盘: logs/curl_resp.json (追加式, 带时间戳+轮次标签分隔) — 三轮(A 正常回答 /
#   B 2048 乱码 / C 2048+15408 修复后正常)的响应体是对照证据链之一环。
# 用法: bash scripts/curl/curl1.sh [A|B|C]   (TAG 标注轮次, 默认 run; 任意 cwd 可跑)
TAG="${1:-run}"
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
{
  echo "=== [$(date '+%F %T')] qwen3.5 curl resp (tag: $TAG) ==="
  curl -s http://127.0.0.1:8340/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen3.5",
      "messages": [
        {
          "role": "user",
          "content": "请你直接给出答案：1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=6, 1+6=7, 1+7=8, 1+8=9, 1+9=10, 1+10=11, 1+11=12, 1+12=13, 1+13=14, 1+14=?"
        }
      ],
      "max_tokens": 10,
      "ignore_eos": true,
      "temperature": 0,
      "chat_template_kwargs": {
        "enable_thinking": false
      }
    }'
  echo
} | tee -a logs/curl_resp.json
