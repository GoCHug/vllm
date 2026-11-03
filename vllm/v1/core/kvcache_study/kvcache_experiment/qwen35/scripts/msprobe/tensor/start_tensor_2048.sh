#!/bin/bash
# tensor dump - 2048 异常（msprobe task=tensor，落 .npy 张量本体）
#
# 与 start_2048.sh 的差别同 start_tensor_1024.sh：
#   task 改 tensor + tensor.list 限定算子 + --num-gpu-blocks-override 压缩块池
#
# 产物: ./2048_dump_tensor/step{0,1}/rank0/{dump.json, dump_tensor_data/*.npy}

vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8341 \
    --block-size 2048 \
    --max-model-len 16384 \
    --enforce-eager \
    --num-gpu-blocks-override 64 \
    --additional-config '{
        "dump_config": {
            "task": "tensor",
            "level": "mix",
            "step": ["0-1"],
            "dump_path": "./2048_dump_tensor",
            "tensor": {
                "list": [
                    "NPU._npu_reshape_and_cache",
                    "NPU.npu_fused_infer_attention_score"
                ]
            }
        }
    }' > ./logs/qwen35_2048_tensor.log 2>&1 &