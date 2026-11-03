#!/bin/bash
# tensor dump 配套请求 - 打 start_tensor_2048.sh 起的 8341 服务

curl -s http://127.0.0.1:8341/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [
      {
        "role": "user",
        "content": "直接给出答案：1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=6, 1+6=7, 1+7=8, 1+8=9, 1+9=10, 1+10=11, 1+11=12, 1+12=13, 1+13=14, 1+14=?"
      }
    ],
    "max_tokens": 10,
    "ignore_eos": true,
    "temperature": 0,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }' > ./logs/curl_tensor_2048.log