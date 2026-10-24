#!/bin/bash
# start_proxy.sh —— 启动 PD 负载均衡代理: localhost:8000 -> P(8100) / D(8200) 同 request_id 双发
# 使用 vllm-ascend 官方示例 examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
# 依赖: fastapi/httpx/uvicorn(pip 已装); 先启动 P、D 两侧并就绪后再启动本代理
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs/server

PROXY=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py

setsid nohup python3 $PROXY \
    --host localhost --port 8000 \
    --prefiller-hosts localhost --prefiller-ports 8100 \
    --decoder-hosts localhost  --decoder-ports 8200 > logs/server/proxy.log 2>&1 < /dev/null &

echo "proxy 已后台启动 (localhost:8000 -> P:8100 / D:8200) -> logs/server/proxy.log"
echo "健康检查: curl -s http://localhost:8000/healthcheck"
