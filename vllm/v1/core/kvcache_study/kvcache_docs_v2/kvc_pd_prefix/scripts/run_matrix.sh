#!/bin/bash
# ==============================================================================
# run_matrix.sh —— 四象限全矩阵编排：动态选干净卡对 + 失败重试×2 + 结果核验
#
# 用法: bash scripts/run_matrix.sh
#   可用环境变量: PCM_P_NPU / PCM_D_NPU(默认 0/1) + 动态选卡掩码 PCM_AUTO_NPU=1
# 防护: 每象限前等双侧 HBM<12GB(120s) ; 象限失败(无 XFER-entry/resp_r)清理重试
# 产物: log/q{1..4}_{p0|p1}{d0|d1}/ + log/matrix_run_<date>.screen
# ==============================================================================
set -uo pipefail
BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"
export PCM_P_NPU="${PCM_P_NPU:-0}" PCM_D_NPU="${PCM_D_NPU:-1}"
export PCM_AUTO_NPU="${PCM_AUTO_NPU:-1}"

hbm_used_mb() { npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *65536" | sed -n "$(( $1 + 1 ))p" | tr -d ' ' | cut -d/ -f1; }

# 动态选卡: 若未显式指定, 跳过外部租户占用的卡(从 0-3 里挑空闲对)
if [ "${PCM_AUTO_NPU:-1}" = "1" ]; then
  CLEAN=()
  for c in 0 1 2 3; do
    u=$(hbm_used_mb "$c" 2>/dev/null)
    if [ -n "$u" ] && [ "$u" -lt 12000 ] 2>/dev/null; then CLEAN+=("$c"); fi
  done
  if [ "${#CLEAN[@]}" -ge 2 ]; then
    export PCM_P_NPU=${CLEAN[0]} PCM_D_NPU=${CLEAN[1]}
    echo "[MATRIX] 动态选卡: P=npu$PCM_P_NPU D=npu$PCM_D_NPU (clean: ${CLEAN[*]})"
  else
    echo "[MATRIX] WARN 找不到两张 12GB 内的卡, 维持默认 $PCM_P_NPU/$PCM_D_NPU"
  fi
fi

q_ok() {  # 成功判定: 双请求响应落盘 + D 侧有 XFER-entry 打点
  [ -s "log/$1/resp_r.json" ] && grep -q "XFER-entry" "log/$1/d_llama.log" 2>/dev/null
}

for q in "q1_p1d1 1 1" "q2_p1d0 1 0" "q3_p0d1 0 1" "q4_p0d0 0 0"; do
  set -- $q; QN="$1"; PPC="$2"; DPC="$3"
  PASS=0
  for try in 1 2; do
    echo "== [MATRIX] $QN trial#$try 于 $(date +%T) =="
    bash scripts/run_quadrant.sh "$QN" "$PPC" "$DPC"
    if q_ok "$QN"; then echo "== [MATRIX] $QN PASS (trial $try)"; PASS=1; break; fi
    echo "== [MATRIX] $QN 未完成(trial $try) —— rm 残迹, 清场缓 20s 后重试"
    rm -rf "log/$QN"; sleep 20
  done
  [ "$PASS" = "0" ] && echo "== [MATRIX] $QN FAIL ×2 —— 弃, 进下一象限 =="
done
echo "MATRIX-DONE $(date +%T)"
