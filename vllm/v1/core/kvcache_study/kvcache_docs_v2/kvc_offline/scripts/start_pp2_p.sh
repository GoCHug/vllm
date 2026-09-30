#!/bin/bash
# start_pp2_p.sh —— 启动 pp2tp2 混布 P 侧实例: TP2(卡0,卡1) / localhost:8100 / kv producer(20001, rank 0)
# 依赖: kvc_offline 补丁已应用(bash patch/apply_kvc_offline_patches.sh = kvc 01~08 + 09 指纹 + 11 原样归档),
#       KVC_DUMP_BLOCKS=1 已导出(run 脚本导出)。
# 参考: vllm-ascend/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md
#       (其示例 DP2xTP2; 本脚本 dp 默认 1, TP2 双卡)
cd "$(dirname "$0")/.." || exit 1
mkdir -p log

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
# HCCL_IF_IP 不设置: 指南示例的 localhost 是非法值(HCCL 要求 ip[ifname] 格式), 会导致 adxl 连接失败 103900
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export ASCEND_RT_VISIBLE_DEVICES=0,1        # ★ pp2tp2: P 占卡0,卡1 (TP1 版为 0)

setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --host localhost --port 8100 \
    --tensor-parallel-size 2 \
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
            "prefill": {"dp_size": 1, "tp_size": 2},
            "decode":  {"dp_size": 1, "tp_size": 2}
      }
  }' > log/p_llama.log 2>&1 < /dev/null &

echo "P 侧已后台启动 (物理卡0+1=TP2 / localhost:8100 / kv_producer rank0 port20001) -> log/p_llama.log"
echo "就绪标志: grep 'Application startup complete' log/p_llama.log"
echo "双 rank 验证: grep -c '物理侧 KV Cache 分配开始' log/p_llama.log (应为 2)"
echo "归档验证: grep '\[KVC\]\[KVB\]' log/p_llama.log (TERM 后应有 r0/r1 两路)"
