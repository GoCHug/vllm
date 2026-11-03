#!/bin/bash
# 异常启动脚本

vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8341 \
    --block-size 2048 \
    --max-model-len 16384 \
    --enforce-eager \
    --additional-config '{
        "dump_config": {
        "task": "statistics",
        "level": "mix",
        "step": ["0-1"],
        "dump_path": "./2048_dump",
        "statistics": {
            "list": []
        }
        }
    }' > ./logs/qwen35_2048_dump.log 2>&1 &
