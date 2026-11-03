#!/bin/bash
# 正常启动脚本

vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8340 \
    --max-model-len 16384 \
    --enforce-eager \
    --additional-config '{
        "dump_config": {
        "task": "statistics",
        "level": "mix",
        "step": ["0-1"],
        "dump_path": "./1024_dump",
        "statistics": {
            "list": []
        }
        }
    }' > ./logs/qwen35_1024_dump.log 2>&1 &