#!/usr/bin/env bash
# ==============================================================================
# pull_artifacts.sh —— kvc_1p1d_prefix 产物回收（仅在主机侧执行, 唯一模式 fetch）
#
# 容器侧无需调用本脚本: run_all.sh [4/5] 已自动把 logs/ 打成
#   kvc_1p1d_prefix_bundle.tar.gz 并打印 tar md5 备查。
# 主机侧(工作区根执行): bash scripts/recover/pull_artifacts.sh fetch
#   -> scp 经 SSH 隧道拉回 tar -> 打印 tar md5(与容器侧 [4/5] 输出人工对照)
#   -> 解包覆盖本地 logs/(四象限 q1~q4 + server/ + analysis/)
#   -> 四象限产物完整性核对(每象限 q_summary.md + resp_r.json + d_pcm.txt)
#
# 隧道: itask ssh-tunnel gggtest --port 5557 (端口可用 PORT 环境变量覆盖)
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
[ "${1:-}" = "fetch" ] || {
  echo "用法: bash scripts/recover/pull_artifacts.sh fetch   (容器侧打包已并入 run_all [4/5], 无需手动)"; exit 1; }
cd "$HERE"
PORT="${PORT:-5557}"
POD_WS="/a3_inference/itask/workdir/wsl02075301/kvc_1p1d_prefix"

echo "== [fetch] 经隧道 $PORT 拉回容器产物 =="
scp -P "$PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    "root@localhost:$POD_WS/kvc_1p1d_prefix_bundle.tar.gz" /tmp/kvc_1p1d_prefix_bundle.tar.gz

echo "== [fetch] tar md5 (对照容器侧 run_all [4/5] 输出) =="
md5 -q /tmp/kvc_1p1d_prefix_bundle.tar.gz 2>/dev/null || md5sum /tmp/kvc_1p1d_prefix_bundle.tar.gz

echo "== [fetch] 解包到本地 (logs/q1~q4 + server/ + analysis/) =="
tar -xzf /tmp/kvc_1p1d_prefix_bundle.tar.gz

echo "== [fetch] 四象限产物核对 (expect: 每象限 q_summary.md + resp_r.json + d_pcm.txt 有值) =="
n=0
for q in q1_p1d1 q2_p1d0 q3_p0d1 q4_p0d0; do
  if [ -s "logs/$q/resp_r.json" ] && [ -s "logs/$q/d_pcm.txt" ]; then
    echo "  [OK] $q ($(wc -l < logs/$q/d_pcm.txt | tr -d ' ') 行 [PCM])"; n=$((n+1))
  else
    echo "  [WARN] $q 产物缺/空"
  fi
done
echo "  [SUM] $n/4 象限就位"

echo "[DONE] 产物已回收 logs/{q1~q4,server,analysis}"
echo "       汇总复核: python3 scripts/analysis/matrix_report.py --dir logs -> logs/analysis/matrix_report.out"
[ "$n" = "4" ] || exit 1
