#!/bin/bash
# ==============================================================================
# run_matrix.sh —— 四象限一键：q1 P1D1(双开·默认) → q2 P1D0 → q3 P0D1 → q4 P0D0(双关·基线)
#
# 用法(容器内, pcm/ 根): bash scripts/run_matrix.sh
# 全程 ~15min(4×[P就绪~60s + D就绪~50s + 双请求~30s + 收尾~20s])
# 产物: log/q{1..4}_*/ 各象限全套证据 + log/matrix_summary.md 四象限对照总表
# ==============================================================================
set -uo pipefail
BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"

echo "===== PCM Prefix Cache 四象限实验 ====="
echo "开始: $(date '+%F %T')"

bash scripts/run_quadrant.sh q1_p1d1 1 1
bash scripts/run_quadrant.sh q2_p1d0 1 0
bash scripts/run_quadrant.sh q3_p0d1 0 1
bash scripts/run_quadrant.sh q4_p0d0 0 0

# ---------- 四象限对照总表 ----------
{
  echo "# PCM 四象限对照总表（自动生成 $(date '+%F %T')）"
  echo
  echo "| 象限 | P·D | SCHED(P) req_r | SCHED(D) req_r | PFINISH req_r | XFER-entry req_r | XFER-end req_r | mooncake 耗时 | D hitrate |"
  echo "|---|---|---|---|---|---|---|---|---|"
  i=0
  for q in q1_p1d1 q2_p1d0 q3_p0d1 q4_p0d0; do
    i=$((i+1))
    d="log/$q"
    pd=$(grep '\[PCM\] CFG' "$d/p_llama.log" 2>/dev/null | head -1 | grep -o 'enable_prefix_caching=[A-Za-z]*' | cut -d= -f2)
    dd=$(grep '\[PCM\] CFG' "$d/d_llama.log" 2>/dev/null | head -1 | grep -o 'enable_prefix_caching=[A-Za-z]*' | cut -d= -f2)
    sp=$(grep '\[PCM\] SCHED' "$d/p_llama.log" 2>/dev/null | tail -1 | sed 's/.*SCHED //')
    sd=$(grep '\[PCM\] SCHED' "$d/d_llama.log" 2>/dev/null | tail -1 | sed 's/.*SCHED //')
    pf=$(grep 'PFINISH' "$d/p_llama.log" 2>/dev/null | tail -1 | sed 's/.*PFINISH //')
    xe=$(grep 'XFER-entry' "$d/d_llama.log" 2>/dev/null | tail -1 | sed 's/.*XFER-entry //')
    xn=$(grep 'XFER-end' "$d/d_llama.log" 2>/dev/null | tail -1 | sed 's/.*XFER-end //')
    tt=$(grep 'KV cache transfer' "$d/d_llama.log" 2>/dev/null | tail -1 | grep -o 'took [0-9.]* ms' | head -1)
    hr=$(tail -1 "$d/d_hitrate.txt" 2>/dev/null | grep -o 'Prefix cache hit rate: [0-9.]*%' | head -1 | sed 's/Prefix cache hit rate: //')
    echo "| $i | ${pd:0:-4}/${dd:0:-4} | ${sp:-<fail>} | ${sd:-<fail>} | ${pf:-<fail>} | ${xe:-<fail>} | ${xn:-<fail>} | ${tt:-<fail>} | ${hr:--} |"
  done
} > log/matrix_summary.md
echo
cat log/matrix_summary.md
echo
echo "===== 全部完成: $(date '+%F %T') ====="
