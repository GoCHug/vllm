#!/bin/bash
# start_d.sh —— 启动 PD 分离 D 侧(decode consumer)实例: 卡1 / localhost:8200 / kv consumer(20002, rank 1)
# 依赖: 补丁已应用, P 侧已就绪; 日志 -> logs/server/d_llama.log
# kvc_1p1d 增强: 自动导出 KVC_SAVE_KV=1 + KVC_SAVE_DIR=<工作区>/tensors —— TERM 归档到 tensors/req{seq}/D/
# 参考: vllm-ascend/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs/server

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
# HCCL_IF_IP 不设置: 指南示例的 localhost 是非法值(HCCL 要求 ip[ifname] 格式), 会导致 adxl 连接失败 103900
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export ASCEND_RT_VISIBLE_DEVICES=1

# [KVS] TERM 归档开关: D 侧归档到 tensors/req{seq}/D/(side 由 08 补丁按 kv_role 自动判定)
export KVC_SAVE_KV=1
export KVC_SAVE_DIR="$(pwd)/tensors"

setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --host localhost --port 8200 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --seed 1024 \
    --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": 20002,
  "kv_rank": 1,
  "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 1},
            "decode":  {"dp_size": 1, "tp_size": 1}
      }
  }' > logs/server/d_llama.log 2>&1 < /dev/null &

echo "D 侧已后台启动 (卡1=物理 npu:1 / localhost:8200 / kv_consumer rank1 port20002) -> logs/server/d_llama.log"
echo "  TERM 归档 -> $KVC_SAVE_DIR/req{seq}/D/kv_pp0tp0.pt (KVC_SAVE_KV=1)"
echo "就绪标志: grep 'Application startup complete' logs/server/d_llama.log"
