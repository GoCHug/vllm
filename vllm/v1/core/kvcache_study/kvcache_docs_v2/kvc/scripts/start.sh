#!/bin/bash
# start.sh —— 启动 vllm 服务（gggtest 容器内；前提: patch/ 已应用, 见 patch/apply_patches.sh）
# 日志统一输出到 log/llama2.log；从任意目录执行均可（自动定位到 kvc/ 根）
cd "$(dirname "$0")/.." || exit 1
mkdir -p log
setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 > log/llama2.log 2>&1 < /dev/null &
sleep 5   # 保持父进程存活 5s, 让 setsid 分离完成 (itask exec 断连杀会话的前 5s 窗口)
echo "服务已后台启动 -> log/llama2.log（首次约 100s 就绪）"
echo "就绪标志: grep 'Application startup complete' log/llama2.log   # 出现 1 行即就绪"
