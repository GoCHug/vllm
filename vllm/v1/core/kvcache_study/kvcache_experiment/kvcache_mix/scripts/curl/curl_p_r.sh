#!/bin/bash
# ==============================================================================
# curl_p_r.sh —— 依次发送 P、R 双请求，落盘响应体与 curl 打屏记录，拆解三条
#               [KVC] 轨迹，并等待 [KVS] 物理KV原样归档落盘
# 前提: 服务已就绪(scripts/server/start.sh, 且 KVC_SAVE_KV=1 已随进程树生效)
# 产物(logs/ 四子目录, 与 scripts/ 一一对应):  curl/resp_p.json / resp_r.json   响应体
#              curl/curl_screen.log               curl 命令 + 响应 打屏完整留痕
#              patchs/kvc_startup.log / kvc_p.log / kvc_r.log   patch 打印日志三段轨迹(pid 剥离)
#              patchs/kvs_archive_lines.log      [KVS] TERM 归档行为留痕
# 输入(本目录 scripts/curl/): req_p.json / req_r.json   请求体(gen_cn_requests.py 生成, 工作区自带)
# 产物(tensors/): req{seq}_{rid尾8}/kv_pp{p}tp{t}.pt  一请求一子目录, 每 worker 一份(4 worker × 2 = 8 个)
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1
REC=logs/curl/curl_screen.log
[ -f scripts/curl/req_p.json ] && [ -f scripts/curl/req_r.json ] || {
  echo "[ERROR] 缺少 scripts/curl/req_p.json / req_r.json (本工作区已自带; 或 python3 scripts/curl/gen_cn_requests.py --gen)"; exit 1; }
mkdir -p logs/curl logs/patchs   # 服务日志在 logs/server(start.sh 建); patch 打印轨迹落 patchs
: > "$REC"
say() { echo "$*" | tee -a "$REC"; }

say "=== $(date '+%F %T') kvc P/R 双请求实验 (v2: TERM 物理KV原样归档) ==="

# ---------- 请求 1: P（种子前缀 324 tok, 2 满块+1 未满, max_tokens=1） ----------
P1=$(wc -l < logs/server/llama-3-8b.log | tr -d " ")
say "--- 请求 1: P ---"
say '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_p.json'
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_p.json | tee logs/curl/resp_p.json | tee -a "$REC"
say ""
say "[P done] (P 分界行号 llama-3-8b.log:$P1)"

sleep 6   # 等 P 结束: 满块带哈希留在缓存池, 全部块释放挂队尾

# ---------- 请求 2: R（前缀命中 2 块 + 五块生命周期, max_tokens=35） ----------
R1=$(wc -l < logs/server/llama-3-8b.log | tr -d " ")
say "--- 请求 2: R ---"
say '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_r.json'
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_r.json | tee logs/curl/resp_r.json | tee -a "$REC"
say ""
say "[R done] (R 分界行号 llama-3-8b.log:$R1)"

sleep 10  # 等收尾日志([KVC] 释放轨迹) + [KVS] 后台落盘线程 flush

# ---------- [KVC] 轨迹拆解（llama-3-8b.log -> 三条轨迹, 剥离进程 pid 前缀） ----------
TOT=$(wc -l < logs/server/llama-3-8b.log | tr -d " ")
STRIP='s/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //'
head -n "$P1" logs/server/llama-3-8b.log              | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_startup.log
sed -n "$((P1+1)),${R1}p" logs/server/llama-3-8b.log  | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_p.log
sed -n "$((R1+1)),${TOT}p" logs/server/llama-3-8b.log | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_r.log
grep "\[KVS\]" logs/server/llama-3-8b.log | sed -E "$STRIP" > logs/patchs/kvs_archive_lines.log
say "== [KVC] 轨迹提取完成(三段 + [KVS] 留痕) =="
wc -l logs/patchs/kvc_startup.log logs/patchs/kvc_p.log logs/patchs/kvc_r.log logs/patchs/kvs_archive_lines.log | tee -a "$REC"

# ---------- 等待 [KVS] 归档落盘（期望 8 个: 4 worker × P/R, 一请求一子目录） ----------
for i in $(seq 1 12); do
  N=$(ls tensors/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 8 ] && break
  say "  ... 归档落盘 $N/8, 10s 后重试"
  sleep 10
done
N=$(ls tensors/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
say "[KVS] 归档文件数: $N/8 (tensors/req{seq}_{rid尾8}/kv_pp{p}tp{t}.pt); 明细见 logs/patchs/kvs_archive_lines.log"
[ "$N" -ge 8 ] || say "[WARN] 归档不足 8 个 —— 检查 grep '\\[KVS\\]' logs/server/llama-3-8b.log 与 KVC_SAVE_KV 环境变量"
ls -laR tensors/ 2>/dev/null | tee -a "$REC"
say "== DONE 打屏/curl 响应/三段轨迹/归档清单 全部落盘 (logs/ + tensors/) =="
