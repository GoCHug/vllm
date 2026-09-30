#!/bin/bash
# ==============================================================================
# start_p.sh —— PCM 四象限实验 P 侧启动（参数 = 1:PC开启 | 0:PC关闭）
#
# 与 ../kvc_pd/scripts/start_p.sh 同源（卡/端口/rank/握手全一致），仅两处差异：
#   1) $2=0 时注入 --no-enable-prefix-caching（象限开关）
#   2) 日志落 PCM 象限目录 $PCM_Q（由调用方传入，相对 kvc_pd_prefix/ 根）
# 教训继承: 勿设 HCCL_IF_IP（vllm-ascend 0.23.0 部署指南 bug，见 ../kvc_pd/docs/1 §2）
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1          # kvc_pd_prefix/ 根
PCM_Q="${1:?用法: start_p.sh <象限日志目录> <pc:1|0>}"
PCM_PC="${2:?用法: start_p.sh <象限日志目录> <pc:1|0>}"
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

echo "[P 启动] pc=${PCNAME} flag='$PCFLAG' -> $PCM_Q/p_llama.log (就绪探活: Application startup complete)"
