#!/bin/bash
# ==============================================================================
# run_quadrant.sh —— 跑一个象限: 起 P→D→proxy → req_p(种缓存)+req_r(前缀复用) → 收证据 → 停全套
#
# 用法: ./run_quadrant.sh <qname> <p_pc:1|0> <d_pc:1|0>
#   例: ./run_quadrant.sh q1_p1d1 1 1     # 象限① 双开
#       ./run_quadrant.sh q2_p1d0 1 0     # 象限② P开D关
#
# 证据落盘( kvc_pd_prefix/log/<qname>/ ):
#   p_llama.log / d_llama.log / proxy.log      三组件全量日志
#   resp_p.json / resp_r.json                  双请求响应
#   p_pcm.txt / d_pcm.txt                      [PCM] 观察行(6 打点)
#   d_transfer.txt                             mooncake 传输耗时行(原生)
#   p_hitrate.txt / d_hitrate.txt              Prometheus 命中率行(原生)
#   q_summary.md                               本象限一行式快照
# 前提: 10 号 PCM 补丁已应用; 4 卡空闲(无 vllm 进程); 本区 log/req_*.json 就位(源自 kvc_1p1d, 原 kvc_pd 已删)
# ==============================================================================
set -uo pipefail   # 不用 -e: 单步失败仍要保证收尾清理
BASE="$(cd "$(dirname "$0")/.." && pwd)"       # kvc_pd_prefix/ 根
REQS="$BASE/log"                              # 请求体已自持(原 kvc_pd/log, 2026-10-04 迁入)
PROXY=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py

Q="$1"; PPC="$2"; DPC="$3"
QDIR="$BASE/log/$Q"; mkdir -p "$QDIR"
PCM_P_NPU="${PCM_P_NPU:-0}"; PCM_D_NPU="${PCM_D_NPU:-1}"

PN="on"; [ "$PPC" = "0" ] && PN="off"
DN="on"; [ "$DPC" = "0" ] && DN="off"
echo "== [$Q] P(pc=$PN, npu$PCM_P_NPU) D(pc=$DN, npu$PCM_D_NPU) begin $(date +%T) =="

# 兜底清理: 失败路径 exit 也不能把 vllm 残留留给下一象限(否则连环占卡)
cleanup() {
  pkill -f "load_balance_proxy_server_example" 2>/dev/null; sleep 2
  pkill -f "v[l]lm serve" 2>/dev/null; sleep 6
  pkill -9 -f "v[l]lm serve" 2>/dev/null; pkill -9 -f "load_balance_[p]roxy" 2>/dev/null; sleep 3
}
trap cleanup EXIT

hbm_used_mb() {  # $1=容器内卡号 → npu-smi 第 n 条 HBM 已用 MB（空=测量失败）
  npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *65536" | sed -n "$(( $1 + 1 ))p" | tr -d ' ' | cut -d/ -f1
}
guard_hbm() {  # $1=卡号 $2=最多等秒 —— 外部租户瞬占防护(阈值≈12GB: 恒留 49GiB 需求裕量)
  local card="$1" tlimit="$2" t0 u
  t0=$(date +%s)
  while :; do
    u=$(hbm_used_mb "$card")
    if [ -n "$u" ] && [ "$u" -lt 12000 ] 2>/dev/null; then echo "[HBM] [$Q] npu$card used=${u}MB 就绪"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt "$tlimit" ]; then echo "[WARN] [$Q] npu$card HBM used=${u:-?}MB 等待 ${tlimit}s 未清，放行试跑"; return 1; fi
    sleep 10
  done
}

wait_ready() {  # $1=日志文件 $2=超时秒
  local log="$1" tlimit="$2" t0
  t0=$(date +%s)
  while ! grep -q "Application startup complete" "$log" 2>/dev/null; do
    if grep -qE "Engine core initialization failed|smashing detected" "$log" 2>/dev/null; then
      echo "[ERROR] [$Q] 启动崩溃(见 $log): $log —— 早退收尾"
      return 1
    fi
    if [ $(( $(date +%s) - t0 )) -gt "$tlimit" ]; then
      echo "[ERROR] [$Q] 就绪超时(${tlimit}s): $log —— 保留现场, 跳过后续请求"
      return 1
    fi
    sleep 5
  done
  echo "[OK] [$Q] $(basename "$log") 就绪 (耗时 $(( $(date +%s) - t0 ))s)"
  return 0
}

# ---------- 0. P 卡防抢占 ----------
guard_hbm "$PCM_P_NPU" 120 || true

# ---------- 1. 起 P → (D 卡) → D → proxy ----------
bash "$BASE/scripts/start_p.sh" "log/$Q" "$PPC"
wait_ready "$QDIR/p_llama.log" 300 || { cp "$QDIR/p_llama.log" "$QDIR/p_llama.timeout" 2>/dev/null; exit 1; }

guard_hbm "$PCM_D_NPU" 120 || true
bash "$BASE/scripts/start_d.sh" "log/$Q" "$DPC"
wait_ready "$QDIR/d_llama.log" 300 || { cp "$QDIR/d_llama.log" "$QDIR/d_llama.timeout" 2>/dev/null; exit 1; }

setsid nohup python3 "$PROXY" --host localhost --port 8000 \
    --prefiller-hosts localhost --prefiller-ports 8100 \
    --decoder-hosts localhost  --decoder-ports 8200 > "$QDIR/proxy.log" 2>&1 < /dev/null &
t0=$(date +%s)
until curl -s --max-time 3 http://localhost:8000/healthcheck >/dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 60 ] && { echo "[ERROR] [$Q] proxy 就绪超时"; break; }
  sleep 2
done
echo "[OK] [$Q] proxy 就绪"

# ---------- 2. 双请求(req_p 种缓存 324tok / req_r 前缀复用 486tok) ----------
curl -s --max-time 120 http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" -d @"$REQS/req_p.json" > "$QDIR/resp_p.json" || echo "[WARN] req_p 失败"
sleep 8    # 等 P→D KV 迁移与双侧收尾日志落盘
curl -s --max-time 180 http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" -d @"$REQS/req_r.json" > "$QDIR/resp_r.json" || echo "[WARN] req_r 失败"
sleep 8

# ---------- 3. 收 [PCM] 证据 ----------
grep "\[PCM\]" "$QDIR/p_llama.log" > "$QDIR/p_pcm.txt" 2>/dev/null || true
grep "\[PCM\]" "$QDIR/d_llama.log" > "$QDIR/d_pcm.txt" 2>/dev/null || true
grep "KV cache transfer" "$QDIR/d_llama.log" > "$QDIR/d_transfer.txt" 2>/dev/null || true
grep "Prefix cache hit rate" "$QDIR/p_llama.log" | tail -2 > "$QDIR/p_hitrate.txt" 2>/dev/null || true
grep "Prefix cache hit rate" "$QDIR/d_llama.log" | tail -2 > "$QDIR/d_hitrate.txt" 2>/dev/null || true
grep "Delaying free" "$QDIR/p_llama.log" > "$QDIR/p_delayfree.txt" 2>/dev/null || true

{
  echo "[$Q] P(pc=$PN) D(pc=$DN) —— $(date +%F) $(date +%T)"
  echo "CFG(P):      $(grep '\[PCM\] CFG' "$QDIR/p_llama.log" | head -1)"
  echo "CFG(D):      $(grep '\[PCM\] CFG' "$QDIR/d_llama.log" | head -1)"
  echo "SCHED(P)末:  $(grep '\[PCM\] SCHED' "$QDIR/p_llama.log" | tail -1)"
  echo "SCHED(D)末:  $(grep '\[PCM\] SCHED' "$QDIR/d_llama.log" | tail -1)"
  echo "ALLOC(D)末:  $(grep '\[PCM\] ALLOC' "$QDIR/d_llama.log" | tail -1)"
  echo "PFINISH 末:  $(grep '\[PCM\] PFINISH' "$QDIR/p_llama.log" | tail -1)"
  echo "XFERentry末: $(grep 'XFER-entry' "$QDIR/d_llama.log" | tail -1)"
  echo "XFERend末:   $(grep 'XFER-end' "$QDIR/d_llama.log" | tail -1)"
  echo "transfer:    $(grep 'KV cache transfer' "$QDIR/d_llama.log" | tail -1)"
  echo "hitrate(P):  $(tail -1 "$QDIR/p_hitrate.txt" 2>/dev/null)"
  echo "hitrate(D):  $(tail -1 "$QDIR/d_hitrate.txt" 2>/dev/null)"
} > "$QDIR/q_summary.md" 2>/dev/null
cat "$QDIR/q_summary.md"

# ---------- 4. 停全套(保清理, 沿用原 kvc_pd/scripts/stop_pd.sh 模式, 该区已删) ----------
pkill -f "load_balance_proxy_server_example" 2>/dev/null; sleep 2
pkill -f "v[l]lm serve" 2>/dev/null; sleep 6
REMAIN=$(pgrep -f "v[l]lm serve" | wc -l | tr -d " ")
if [ "$REMAIN" = "0" ]; then
  echo "[OK] [$Q] 组件已归零"
else
  echo "[WARN] [$Q] 残留 $REMAIN 进程, 强制清理"
  pkill -9 -f "v[l]lm serve" 2>/dev/null; pkill -9 -f "load_balance_proxy" 2>/dev/null; sleep 3
fi
sleep 5   # 端口释放缓冲
echo "== [$Q] DONE $(date +%T) =="
