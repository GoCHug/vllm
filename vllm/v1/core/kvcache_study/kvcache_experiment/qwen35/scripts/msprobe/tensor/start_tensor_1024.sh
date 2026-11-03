#!/bin/bash
# tensor dump - 1024 基线（msprobe task=tensor，落 .npy 张量本体）
#
# 与 start_1024.sh 的差别：
#   1) task: statistics -> tensor，dump 出真实张量而非统计标量
#   2) tensor.list 只挑 kvcache 读写两个算子，避免全量张量撑爆盘
#   3) --num-gpu-blocks-override 压缩块池：整张 key_cache 从 ~4.7GB 降到几 MiB
#      （块池大小不改变页宽几何，K 段≡SSM 段的不变量不受影响）
#
# 产物: ./1024_dump_tensor/step{0,1}/rank0/{dump.json, dump_tensor_data/*.npy}

vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8340 \
    --max-model-len 16384 \
    --enforce-eager \
    --num-gpu-blocks-override 64 \
    --additional-config '{
        "dump_config": {
            "task": "tensor",
            "level": "mix",
            "step": ["0-1"],
            "dump_path": "./1024_dump_tensor",
            "tensor": {
                "list": [
                    "NPU._npu_reshape_and_cache",
                    "NPU.npu_fused_infer_attention_score"
                ]
            }
        }
    }' > ./logs/qwen35_1024_tensor.log 2>&1 &