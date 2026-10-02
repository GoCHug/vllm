#!/bin/bash
# ==============================================================================
# stop.sh —— 杀掉容器内的 vllm 服务（kvc 实验收尾第一步）
# 之后还需: bash patch/revert_patches.sh 撤补丁, 保持容器源码未改动
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
pkill -f "v[l]lm serve" 2>/dev/null && echo "已发送 kill 信号, 等待进程退出..." || echo "无运行中的 vllm 服务"
sleep 5
N=$(ps -ef | grep "[v]llm serve" | wc -l | tr -d " ")
echo "剩余 vllm 进程数: $N"
[ "$N" = 0 ] && echo "[OK] 服务已完全退出 —— 下一步: bash patch/revert_patches.sh" \
             || echo "[WARN] 仍有进程, 可重跑本脚本或 ps -ef | grep vllm 检查"
