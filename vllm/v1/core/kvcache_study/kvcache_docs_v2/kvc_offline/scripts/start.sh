#!/bin/bash
# ==============================================================================
# start.sh —— 启动 kvc_offline 实验服务（gggtest 容器内；与 ../kvc/ 相同形态）
#
# 形态：单机 PP2×TP2 混合并行 —— 一个 vllm serve 实例占满 4 卡
#   （--tensor-parallel-size 2 --pipeline-parallel-size 2，无 PD 分离、无 proxy）
# 4 个 worker 进程 = 2 PP stage × 2 TP rank：
#   pp0: 层 0~15（每 TP rank 持 4 kv_heads 分片）；pp1: 层 16~31
# 前提: patch/apply_kvc_offline_patches.sh 已应用（kvc 01~08 + 09 指纹 + 11 归档）
#       且 KVC_DUMP_BLOCKS=1 已导出（run_all.sh 导出）
# 日志 -> log/llama-3-8b.log（4 worker [KVC] 行交织在同一文件, 检查器集合匹配消解）
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
mkdir -p log
setsid nohup vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 > log/llama-3-8b.log 2>&1 < /dev/null &
sleep 5   # 保持父进程存活 5s, 让 setsid 分离完成（itask exec 断连杀会话的前 5s 窗口）
echo "服务已后台启动（单机 PP2×TP2 占 4 卡）-> log/llama-3-8b.log（首次约 100s 就绪）"
echo "就绪标志: grep 'Application startup complete' log/llama-3-8b.log   # 出现 1 行即就绪"
echo "四 worker 验证: grep -c '物理侧 KV Cache 分配开始' log/llama-3-8b.log (应 4——每 worker 各一)"
echo "归档验证: grep '\[KVC\]\[KVB\]' log/llama-3-8b.log（TERM 后应有 pp0tp0/pp0tp1/pp1tp0/pp1tp1 四路 × 每请求）"
