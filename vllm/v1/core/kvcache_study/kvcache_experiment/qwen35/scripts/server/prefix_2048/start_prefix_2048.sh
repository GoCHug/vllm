#!/bin/bash
# 异常启动脚本

vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8343 \
    --block-size 2048 \
    --enable-prefix-caching \
    --max-model-len 16384 > ./logs/qwen35_prefix_2048.log 2>&1 &
