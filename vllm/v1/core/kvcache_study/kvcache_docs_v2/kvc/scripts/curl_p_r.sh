#!/bin/bash
# ==============================================================================
# curl_p_r.sh —— 依次发送 P、R 双请求，落盘响应体与 curl 打屏记录，拆解三条
#               [KVC] 轨迹，并等待 [KVS] 物理KV原样归档落盘
# 前提: 服务已就绪(scripts/start.sh, 且 KVC_SAVE_KV=1 已随进程树生效)
# 产物(log/):  resp_p.json / resp_r.json     响应体
#              curl_screen.log              curl 命令 + 响应 打屏完整留痕
#              kvc_startup.log / kvc_p.log / kvc_r.log    [KVC] 三段轨迹(pid 剥离)
#              kvs_archive_lines.log        [KVS] TERM 归档行为留痕
# 产物(tensors/): kv_pp{p}tp{t}_s{seq}_{rid8}.pt  每 worker×请求一份(4 worker × 2 = 8 个)
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
REC=log/curl_screen.log
[ -f log/req_p.json ] && [ -f log/req_r.json ] || {
  echo "[ERROR] 缺少 log/req_p.json / req_r.json (本工作区已自带; 或 python3 scripts/gen_cn_requests.py --gen)"; exit 1; }
: > "$REC"
say() { echo "$*" | tee -a "$REC"; }

say "=== $(date '+%F %T') kvc P/R 双请求实验 (v2: TERM 物理KV原样归档) ==="

# ---------- 请求 1: P（种子前缀 324 tok, 2 满块+1 未满, max_tokens=1） ----------
P1=$(wc -l < log/llama-3-8b.log | tr -d " ")
say "--- 请求 1: P ---"
say '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json'
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json | tee log/resp_p.json | tee -a "$REC"
say ""
say "[P done] (P 分界行号 llama-3-8b.log:$P1)"

sleep 6   # 等 P 结束: 满块带哈希留在缓存池, 全部块释放挂队尾

# ---------- 请求 2: R（前缀命中 2 块 + 五块生命周期, max_tokens=35） ----------
R1=$(wc -l < log/llama-3-8b.log | tr -d " ")
say "--- 请求 2: R ---"
say '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r.json'
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r.json | tee log/resp_r.json | tee -a "$REC"
say ""
say "[R done] (R 分界行号 llama-3-8b.log:$R1)"

sleep 10  # 等收尾日志([KVC] 释放轨迹) + [KVS] 后台落盘线程 flush

# ---------- [KVC] 轨迹拆解（llama-3-8b.log -> 三条轨迹, 剥离进程 pid 前缀） ----------
TOT=$(wc -l < log/llama-3-8b.log | tr -d " ")
STRIP='s/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //'
head -n "$P1" log/llama-3-8b.log              | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_startup.log
sed -n "$((P1+1)),${R1}p" log/llama-3-8b.log  | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_p.log
sed -n "$((R1+1)),${TOT}p" log/llama-3-8b.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_r.log
grep "\[KVS\]" log/llama-3-8b.log | sed -E "$STRIP" > log/kvs_archive_lines.log
say "== [KVC] 轨迹提取完成(三段 + [KVS] 留痕) =="
wc -l log/kvc_startup.log log/kvc_p.log log/kvc_r.log log/kvs_archive_lines.log | tee -a "$REC"

# ---------- 等待 [KVS] 归档落盘（期望 8 个: 4 worker × P/R 双请求） ----------
for i in $(seq 1 12); do
  N=$(ls tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 8 ] && break
  say "  ... 归档落盘 $N/8, 10s 后重试"
  sleep 10
done
N=$(ls tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
say "[KVS] 归档文件数: $N/8 (4 worker × P/R); 明细见 log/kvs_archive_lines.log"
[ "$N" -ge 8 ] || say "[WARN] 归档不足 8 个 —— 检查 grep '\[KVS\]' log/llama-3-8b.log 与 KVC_SAVE_KV 环境变量"
ls -la tensors/ 2>/dev/null | tee -a "$REC"
say "== DONE 打屏/curl 响应/三段轨迹/归档清单 全部落盘 (log/ + tensors/) =="
