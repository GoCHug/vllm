#!/bin/bash
# 异常启动脚本（额外指定 --block-size 2048，用于复现 kvcache 异常）
# 日志输出: qwen35_abnormal.log（写入实验根目录 qwen35/）
# 用法: bash scripts/server/start_abnormal.sh
# 混合 KV 观测: 已内置 KVC_SAVE_KV=1 + KVC_SAVE_DIR=tensors —— 源码带 08 号补丁时,
#   请求结束(TERM)自动归档 attn K/V + mamba conv/ssm 至 tensors/req*/kv_pp0tp0.pt;
#   与 start_normal 对照可观察 --block-size 2048 不变量破坏下的张量污染,
#   未打补丁时这两个 env 无任何影响(乱码复现照常, 归档默认关零侵入)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p tensors
export KVC_SAVE_KV=1
export KVC_SAVE_DIR="tensors"
echo "[env] KVC_SAVE_KV=$KVC_SAVE_KV  KVC_SAVE_DIR=$KVC_SAVE_DIR (serve cwd=$(pwd))"
vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8340 \
    --block-size 2048 \
    --max-model-len 16384 > qwen35_abnormal.log 2>&1 &
