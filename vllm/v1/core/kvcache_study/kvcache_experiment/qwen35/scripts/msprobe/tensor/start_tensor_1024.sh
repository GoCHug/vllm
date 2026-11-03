#!/bin/bash
# tensor dump - 1024 基线（msprobe task=tensor，落 .npy 张量本体）
#
# 与 start_1024.sh 的差别：
#   1) task: statistics -> tensor，dump 出真实张量而非统计标量
#   2) tensor.list 限定 4 个 kvcache 读写算子（attn 写/读 + GDN conv/ssm）
#   3) --num-gpu-blocks-override 20：块池按逻辑页计（1 页 = 1024 token = 8 个 128-token 物理块）。
#      max_model_len=16384 要求至少约 19 页才能启动（131 token 本身只占 1 页）；
#      块池大小不改变页宽几何，K 段≡SSM 段的不变量不受影响
#   4) step 0-10 覆盖全量：1 次 prefill + 10 次 decode
#
# 各算子的 cache 快照含义：
#   _npu_reshape_and_cache.input.key_cache/value_cache   attn 写前（原地写，output 为 null）
#   npu_fused_infer_attention_score.input.key/value      attn 写后（同 step 内先写后读）
#   npu_causal_conv1d_custom.input.conv_state            GDN conv 写前
#   npu_recurrent_gated_delta_rule.input.state           GDN ssm  写前
#   （GDN 写后状态本 step 无读者，需用下一 step 的写前快照配对；最后一个 step 的写后缺失）
#
# 产物: ./1024_dump_tensor/step{0..10}/rank0/{dump.json, dump_tensor_data/*.npy}
# 体积: 约 9 GiB（attn 单张 20 MiB；需要时可在容器内按有效块裁剪后再回传）

vllm serve /home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/ \
    --served-model-name "qwen3.5" \
    --port 8340 \
    --max-model-len 16384 \
    --enforce-eager \
    --num-gpu-blocks-override 20 \
    --additional-config '{
        "dump_config": {
            "task": "tensor",
            "level": "mix",
            "step": ["0-10"],
            "dump_path": "./1024_dump_tensor",
            "tensor": {
                "list": [
                    "NPU._npu_reshape_and_cache",
                    "NPU.npu_fused_infer_attention_score",
                    "_C_ascend.npu_causal_conv1d_custom",
                    "_C_ascend.npu_recurrent_gated_delta_rule"
                ]
            }
        }
    }' > ./logs/qwen35_1024_tensor.log 2>&1 &