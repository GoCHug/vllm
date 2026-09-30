#!/bin/bash
# start_pp2_d.sh —— 启动 pp2tp2 混布 D 侧实例: TP2(卡2,卡3) / localhost:8200 / kv consumer(20002, rank 1)
# 依赖: kvc_offline 补丁已应用; 先启动 P 侧并就绪。
# 注意: ASCEND_RT_VISIBLE_DEVICES=2,3 —— 容器内进程视角逻辑卡号 0,1 映射物理卡 2,3,
#       故 TERM 日志行 dev=npu:0/npu:1 在 D 侧对应物理卡 2/3(r0/r1)。
cd "$(dirname "$0")/.." || exit 1
mkdir -p log

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export ASCEND_RT_VISIBLE_DEVICES=2,3        # ★ pp2tp2: D 占卡2,卡3 (TP1 版为 1)

setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --host localhost --port 8200 \
    --tensor-parallel-size 2 \
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
            "prefill": {"dp_size": 1, "tp_size": 2},
            "decode":  {"dp_size": 1, "tp_size": 2}
      }
  }' > log/d_llama.log 2>&1 < /dev/null &

echo "D 侧已后台启动 (物理卡2+3=TP2 / localhost:8200 / kv_consumer rank1 port20002) -> log/d_llama.log"
echo "就绪标志: grep 'Application startup complete' log/d_llama.log"
echo "双 rank 验证: grep -c '物理侧 KV Cache 分配开始' log/d_llama.log (应为 2)"
echo "归档验证: grep '\[KVC\]\[KVB\]' log/d_llama.log (TERM 后应有 r0/r1 两路)"
