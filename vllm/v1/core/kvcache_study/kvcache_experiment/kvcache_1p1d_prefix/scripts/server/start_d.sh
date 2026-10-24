#!/bin/bash
# ==============================================================================
# start_d.sh —— kvc_1p1d_prefix 四象限实验 D 侧启动(参数 = 1:PC开启 | 0:PC关闭)
#
# 用法: bash scripts/server/start_d.sh <象限日志子目录> <pc:1|0>
# 拓扑: D = decode consumer, 卡 $PCM_D_NPU(默认1) / localhost:8200 / kv_consumer(20002, rank1)
#   pc=0 时注入 --no-enable-prefix-caching(象限开关)
# 日志: <象限日志子目录>/d_llama.log (就绪探活: "Application startup complete")
# 前提: P 侧已就绪(mooncake 会话建立次序: P 先于 D)
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1          # kvc_1p1d_prefix/ 根
PCM_Q="${1:?用法: start_d.sh <象限日志子目录> <pc:1|0>}"
PCM_PC="${2:?用法: start_d.sh <象限日志子目录> <pc:1|0>}"
mkdir -p "$PCM_Q"

PCFLAG=""
[ "$PCM_PC" = "0" ] && PCFLAG="--no-enable-prefix-caching"
PCNAME="on";  [ "$PCM_PC" = "0" ] && PCNAME="off"

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export ASCEND_RT_VISIBLE_DEVICES=${PCM_D_NPU:-1}

setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --host localhost --port 8200 \
    --tensor-parallel-size 1 \
    --enforce-eager $PCFLAG \
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
  }' > "$PCM_Q/d_llama.log" 2>&1 < /dev/null &

echo "[D 启动] pc=${PCNAME} flag='$PCFLAG' npu${ASCEND_RT_VISIBLE_DEVICES} -> $PCM_Q/d_llama.log (就绪探活: Application startup complete)"
