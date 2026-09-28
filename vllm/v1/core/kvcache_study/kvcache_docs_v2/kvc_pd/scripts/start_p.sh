#!/bin/bash
# start_p.sh —— 启动 PD 分离 P 侧(prefill producer)实例: 卡0 / localhost:8100 / kv producer(20001, rank 0)
# 依赖: 9 个 [KVC] 补丁已应用(用 ../kvc/patch/apply_patches.sh), 日志统一输出到 log/p_llama.log
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
export ASCEND_RT_VISIBLE_DEVICES=0

setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --host localhost --port 8100 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --seed 1024 \
    --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": 20001,
  "kv_rank": 0,
  "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 1},
            "decode":  {"dp_size": 1, "tp_size": 1}
      }
  }' > log/p_llama.log 2>&1 < /dev/null &

echo "P 侧已后台启动 (卡0=物理 npu:0 / localhost:8100 / kv_producer rank0 port20001) -> log/p_llama.log"
echo "就绪标志: grep 'Application startup complete' log/p_llama.log"