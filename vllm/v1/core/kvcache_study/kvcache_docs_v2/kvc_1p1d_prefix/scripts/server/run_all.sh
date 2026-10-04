#!/bin/bash
# ==============================================================================
# run_all.sh —— kvc_1p1d_prefix 1P1D Prefix Cache 四象限实验一键执行(容器内)
#
# 拓扑: 1P(prefill producer, 卡0/:8100) + 1D(decode consumer, 卡1/:8200) + proxy(:8000 双发)
# 五阶段:
#   [1/5] patch:     apply_patches.sh(01 号 PCM 六打点注入 mooncake_connector.py)
#   [2/5] matrix:    run_matrix.sh —— 四象限 q1~q4(每象限独立起停 P/D/proxy, 冷缓存种 324
#                    tok → req_r 486 tok 前缀复用), 失败重试×2 + 动态选卡
#   [3/5] 汇总:      analysis/matrix_report.py -> logs/analysis/matrix_report.out
#                    (四象限对照总表 + 铁律核验: 传输量只看 D 开关 / prefill 只看 P 开关)
#   [4/5] 打包:      tar logs/ -> kvc_1p1d_prefix_bundle.tar.gz + md5(主机侧 recover 拉回)
#   [5/5] 提示:      手动收尾(stay 优雅退出) —— stop.sh + revert_patches.sh
#
# 日志布局(四象限一目录, 共 12 文件/象限):
#   logs/q1_p1d1/ ~ logs/q4_p0d0/   p_llama.log / d_llama.log / proxy.log / resp_{p,r}.json
#                                  p_pcm.txt / d_pcm.txt / d_transfer.txt / {p,d}_hitrate.txt
#                                  p_delayfree.txt / q_summary.md
#   logs/server/                    matrix_screen.log / run_all_screen.log(本脚本留痕)
#   logs/analysis/                  matrix_report.out(四象限汇总)
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc_1p1d_prefix
#   setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
#   tail -f logs/server/run_all_screen.log
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1
# 容器路径显式导出(apply/revert 脚本默认路径为本地 macOS 路径, 容器内必须覆盖)
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
mkdir -p logs/server logs/analysis
echo "===== [run_all] $(date '+%F %T') kvc_1p1d_prefix 1P1D Prefix 四象限实验开始 ====="
[ -f scripts/curl/req_p.json ] && [ -f scripts/curl/req_r.json ] || { echo "[FATAL] scripts/curl/req_p/req_r 缺失"; exit 1; }
rm -f logs/server/matrix_screen.log 2>/dev/null

echo "===== [1/5] 打补丁（01 号 PCM 六打点注入 mooncake_connector.py） ====="
bash scripts/patchs/apply_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/5] 四象限矩阵（每象限~2.5-3min, 全程~12min; 失败重试×2） ====="
bash scripts/server/run_matrix.sh 2>&1 | tee logs/server/matrix_screen.log
QMATRIX_DONE=$(grep -c "MATRIX-DONE" logs/server/matrix_screen.log 2>/dev/null || echo 0)
[ "$QMATRIX_DONE" = "0" ] && echo "[FATAL] 四象限矩阵未完成(见 logs/server/matrix_screen.log)" && exit 1

echo "===== [3/5] 四象限汇总（对照总表 + 铁律核验） ====="
python3 scripts/analysis/matrix_report.py --dir logs 2>/dev/null || echo "[WARN] 矩阵汇总异常, 产物保留于 logs/qN/ 可手动核对"

echo "===== [4/5] 打包产物 (kvc_1p1d_prefix_bundle.tar.gz = logs/, 供主机侧 fetch) ====="
tar -czf kvc_1p1d_prefix_bundle.tar.gz logs/
echo "[DONE] $(ls -la kvc_1p1d_prefix_bundle.tar.gz | awk '{print $5}') B -> kvc_1p1d_prefix_bundle.tar.gz"
md5sum kvc_1p1d_prefix_bundle.tar.gz 2>/dev/null || md5 -q kvc_1p1d_prefix_bundle.tar.gz

echo "===== [DONE] $(date '+%F %T') kvc_1p1d_prefix 四象限产物就绪并已打包: logs/ ====="
echo "       回收: 主机侧 bash scripts/recover/pull_artifacts.sh fetch (经 5557 隧道拉回)"
echo "       收尾: bash scripts/server/stop.sh && bash scripts/patchs/revert_patches.sh (保持容器源码未改动)"
