#!/bin/bash
# ==============================================================================
# start_p.sh —— kvc_1p1d_prefix 四象限实验 P 侧启动(参数 = 1:PC开启 | 0:PC关闭)
#
# 用法: bash scripts/server/start_p.sh <象限日志子目录> <pc:1|0>
#   例: bash scripts/server/start_p.sh logs/q2_p1d0 1      # P侧开prefix(象限②)
# 拓扑: P = prefill producer, 卡 $PCM_P_NPU(默认0) / localhost:8100 / kv_producer(20001, rank0)
#   pc=0 时注入 --no-enable-prefix-caching(象限开关)
# 日志: <象限日志子目录>/p_llama.log (就绪探活: "Application startup complete")
# 教训继承: 勿设 HCCL_IF_IP(部署指南示例 localhost 为非法值, adxl 连接失败 103900)
# 本工作区独立自持: 启动参数与 ../kvc_1p1d 已实测基线一致(TP1/enforce_eager/seed1024)
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1          # kvc_1p1d_prefix/ 根
PCM_Q="${1:?用法: start_p.sh <象限日志子目录> <pc:1|0>}"
PCM_PC="${2:?用法: start_p.sh <象限日志子目录> <pc:1|0>}"
mkdir -p "$PCM_Q"

PCFLAG=""
[ "$PCM_PC" = "0" ] && PCFLAG="--no-enable-prefix-caching"
PCNAME="on";  [ "$PCM_PC" = "0" ] && PCNAME="off"

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export ASCEND_RT_VISIBLE_DEVICES=${PCM_P_NPU:-0}

setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --host localhost --port 8100 \
    --tensor-parallel-size 1 \
    --enforce-eager $PCFLAG \
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
  }' > "$PCM_Q/p_llama.log" 2>&1 < /dev/null &

echo "[P 启动] pc=${PCNAME} flag='$PCFLAG' npu${ASCEND_RT_VISIBLE_DEVICES} -> $PCM_Q/p_llama.log (就绪探活: Application startup complete)"
