#!/usr/bin/env bash
# 通用观测：抓取 vLLM /metrics 中 KV cache 关键指标（case 04/05 等共用）
#
# 用法:
#   ./observe_metrics.sh [BASE_URL] [INTERVAL_SEC]
# 例:
#   ./observe_metrics.sh http://127.0.0.1:8000        # 单次快照
#   ./observe_metrics.sh http://127.0.0.1:8000 5      # 每 5s 循环
#
# 关键指标说明（定义见 vllm/v1/metrics/loggers.py）:
#   vllm:prefix_cache_queries  prefix cache 查询的 token 数(累计)
#   vllm:prefix_cache_hits     命中的 token 数(累计); 命中率 = hits/queries
#   vllm:gpu_cache_usage_perc  KV cache 使用率 0~1
#   vllm:num_requests_running  正在跑的请求数（死锁类: 卡住不动）
#   vllm:num_requests_waiting  排队请求数
BASE_URL="${1:-http://127.0.0.1:8000}"
INTERVAL="${2:-0}"

snapshot() {
  echo "===== $(date '+%F %T') ====="
  curl -sf "$BASE_URL/metrics" \
    | grep -E '^vllm:(prefix_cache_queries|prefix_cache_hits|gpu_cache_usage_perc|num_requests_running|num_requests_waiting|num_preemptions)(\{|$)' \
    | sed 's/{.*} / /'
}

if [ "$INTERVAL" -gt 0 ]; then
  while true; do snapshot; sleep "$INTERVAL"; done
else
  snapshot
fi
