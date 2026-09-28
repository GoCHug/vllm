#!/bin/bash
# start_d.sh —— 启动 PD 分离 D 侧(decode consumer)实例: 卡1 / localhost:8200 / kv consumer(20002, rank 1)
# 依赖: 9 个 [KVC] 补丁已应用(用 ../kvc/patch/apply_patches.sh), 日志统一输出到 log/d_llama.log; 先启动 P 侧
# 参考: vllm-ascend/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md
cd "$(dirname "$0")/.." || exit 1
mkdir -p log

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
# HCCL_IF_IP 不设置: 指南示例的 localhost 是非法值(HCCL 要求 ip[ifname] 格式), 会导致 adxl 连接失败 103900
# (vllm-ascend 部署指南踩坑: D 侧连 P 数据端口时 Config_Error_Invalid_Environment_Variable(EI0001))
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export ASCEND_RT_VISIBLE_DEVICES=1

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
  }' > log/d_llama.log 2>&1 < /dev/null &

echo "D 侧已后台启动 (卡1=物理 npu:1 / localhost:8200 / kv_consumer rank1 port20002) -> log/d_llama.log"
echo "就绪标志: grep 'Application startup complete' log/d_llama.log"