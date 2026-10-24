#!/bin/bash
# ==============================================================================
# stop.sh —— 停止 PD 分离全部组件: 先杀 proxy, 再杀两个 vllm 实例(P/D), 确认零进程
# 用法: bash scripts/server/stop.sh
# (run_quadrant.sh 每象限自带同模式清理; 本脚本用于手动兜底/全局收尾)
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1

echo "停止 proxy..."
pkill -f "load_balance_proxy_server_example" 2>/dev/null
sleep 2

echo "停止 vllm 实例(P:8100 / D:8200)..."
pkill -f "v[l]lm serve" 2>/dev/null
sleep 5

REMAIN_PROXY=$(pgrep -f "load_balance_proxy_server_example" 2>/dev/null | wc -l | tr -d " ")
REMAIN_VLLM=$(pgrep -f "v[l]lm serve" 2>/dev/null | wc -l | tr -d " ")
echo "剩余 proxy 进程数: $REMAIN_PROXY"
echo "剩余 vllm 进程数(P/D): $REMAIN_VLLM"
if [ "$REMAIN_PROXY" = "0" ] && [ "$REMAIN_VLLM" = "0" ]; then
  echo "[OK] PD 全部组件已退出 —— 下一步: bash scripts/patchs/revert_patches.sh"
else
  echo "[WARN] 仍有进程残留, 再次强制清理..."
  pkill -9 -f "load_balance_proxy_server_example" 2>/dev/null
  pkill -9 -f "v[l]lm serve" 2>/dev/null
  sleep 3
  pgrep -af "v[l]lm serve|load_balance_proxy" || echo "[OK] 强制清理后归零"
fi
