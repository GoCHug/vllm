#!/bin/bash
# F 轮启动脚本（--enable-prefix-caching + --block-size 2048，无修复）
#   与 start_abnormal.sh 唯一差异 = 显式 --enable-prefix-caching
#   （A/B/C 轮不传该参数, 实测 hybrid 默认 enable_prefix_caching=False / NoPrefixCache;
#    F 轮目的: 开 prefix 时 2048 步进失配是否仍触发 → (prefix, block-size) 矩阵补全）
# 日志输出: qwen35_prefix_runF.log（写入实验根目录 qwen35/，轮末归位 logs/）
# 用法: bash scripts/server/start_prefix_2048.sh
# 混合 KV 观测: 已内置 KVC_SAVE_KV=1 + KVC_SAVE_DIR=tensors —— 源码带 08 号补丁时,
#   请求结束(TERM)自动归档 attn K/V + mamba conv/ssm 至 tensors/req*/kv_pp0tp0.pt;
#   未打补丁时这两个 env 无任何影响(复现照常, 归档默认关零侵入)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p tensors
export KVC_SAVE_KV=1
export KVC_SAVE_DIR="tensors"
echo "[env] KVC_SAVE_KV=$KVC_SAVE_KV  KVC_SAVE_DIR=$KVC_SAVE_DIR (serve cwd=$(pwd))"
vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8340 \
    --block-size 2048 \
    --enable-prefix-caching \
    --max-model-len 16384 > qwen35_prefix_runF.log 2>&1 &
