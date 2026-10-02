#!/bin/bash
# ==============================================================================
# start.sh —— 启动 kvc 实验服务（gggtest 容器内；前提: patch/apply_patches.sh 已应用）
#
# 形态：单机 PP2×TP2 混合并行 —— 一个 vllm serve 实例占满 4 卡
#   （--tensor-parallel-size 2 --pipeline-parallel-size 2，无 PD 分离、无 proxy）
# 4 个 worker 进程 = 2 PP stage × 2 TP rank：
#   pp0: 层 0~15（每 TP rank 持 4 kv_heads 分片）；pp1: 层 16~31
#
# 关键 env（export 后随进程树继承到 4 个 worker，08 号补丁 v2 读取）：
#   KVC_SAVE_KV=1                     开启 TERM 物理 KV 原样归档（默认关, 零侵入）
#   KVC_SAVE_DIR=<工作区>/tensors     归档目录（绝对路径，与 worker cwd 解耦）
# 日志 -> log/llama-3-8b.log
# 归档 -> tensors/kv_pp{p}tp{t}_s{seq}_{rid尾8}.pt（每 worker 每请求一份, 后台落盘）
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
mkdir -p log tensors
export KVC_SAVE_KV=1
export KVC_SAVE_DIR="$(pwd)/tensors"
echo "[env] KVC_SAVE_KV=$KVC_SAVE_KV  KVC_SAVE_DIR=$KVC_SAVE_DIR"
setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 > log/llama-3-8b.log 2>&1 < /dev/null &
sleep 5   # 保持父进程存活 5s, 让 setsid 分离完成（itask exec 断连杀会话的前 5s 窗口）
echo "服务已后台启动（单机 PP2×TP2 占 4 卡, TERM 归档 -> $KVC_SAVE_DIR）-> log/llama-3-8b.log（首次约 100s 就绪）"
echo "就绪标志: grep 'Application startup complete' log/llama-3-8b.log   # 出现 1 行即就绪"
echo "归档启用横幅(每 worker 一行, 共 4): grep '物理KV原样归档已启用' log/llama-3-8b.log"
echo "TERM 归档行为(每 worker×请求一行): grep '\[KVS\]' log/llama-3-8b.log"
