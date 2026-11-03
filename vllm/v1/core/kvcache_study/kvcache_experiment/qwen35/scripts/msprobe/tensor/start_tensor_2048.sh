#!/bin/bash
# tensor dump - 2048 异常（msprobe task=tensor，落 .npy 张量本体）
#
# 与 start_tensor_1024.sh 同构，只差 --block-size 2048 与端口/dump 目录。
# 2048 下 1 逻辑页 = 16 个 128-token 物理块（1024 下是 8 个），
# 即 attn 一页 K 段 2 MiB 而 mamba 一 slot 的 ssm 段仍 1 MiB —— 复现覆写的关键。
#
# 块池同为 20 逻辑页（max_model_len=16384 的启动下限）：2048 页更大，物理块数翻倍。
#
# 各算子的 cache 快照含义：
#   _npu_reshape_and_cache.input.key_cache/value_cache   attn 写前（原地写，output 为 null）
#   npu_fused_infer_attention_score.input.key/value      attn 写后（同 step 内先写后读）
#   npu_causal_conv1d_custom.input.conv_state            GDN conv 写前
#   npu_recurrent_gated_delta_rule.input.state           GDN ssm  写前
#   （GDN 写后状态本 step 无读者，需用下一 step 的写前快照配对；最后一个 step 的写后缺失）
#
# 产物: ./2048_dump_tensor/step{0..10}/rank0/{dump.json, dump_tensor_data/*.npy}
# 体积: 约 14 GiB（attn 单张 40 MiB，是 1024 轮的两倍）

vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8341 \
    --block-size 2048 \
    --max-model-len 16384 \
    --enforce-eager \
    --num-gpu-blocks-override 20 \
    --additional-config '{
        "dump_config": {
            "task": "tensor",
            "level": "mix",
            "step": ["0-10"],
            "dump_path": "./2048_dump_tensor",
            "tensor": {
                "list": [
                    "NPU._npu_reshape_and_cache",
                    "NPU.npu_fused_infer_attention_score",
                    "_C_ascend.npu_causal_conv1d_custom",
                    "_C_ascend.npu_recurrent_gated_delta_rule"
                ]
            }
        }
    }' > ./logs/qwen35_2048_tensor.log 2>&1 &