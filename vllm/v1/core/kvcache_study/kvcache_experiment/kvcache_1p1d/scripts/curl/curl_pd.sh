#!/bin/bash
# ==============================================================================
# curl_pd.sh —— 经 proxy(8000) 依次发送 P、R 双请求, 落盘 curl 打屏/响应, 并提取双侧 [KVC] 轨迹
#
# 前提: P(8100)/D(8200)/proxy(8000) 三者均就绪; scripts/curl/req_p.json 与 req_r.json 已就位
# 产物:
#   logs/curl/   resp_{p,r}.json + curl_{p,r}_screen.txt(打屏)
#   logs/patchs/ 双侧各三段 [KVC] 轨迹(sed 剥进程前缀, 覆盖更新):
#     kvc_p_{startup,reqp,reqr}.log + kvc_d_{startup,reqp,reqr}.log
# 边界用 shell 变量记录(不落盘临时文件); 轨迹固定命名, 每次重跑覆盖更新
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs/curl logs/patchs

[ -f scripts/curl/req_p.json ] && [ -f scripts/curl/req_r.json ] || {
  echo "[ERROR] 缺少 scripts/curl/req_p.json / req_r.json"
  echo "        先执行: python3 scripts/curl/gen_pd_requests.py --gen"
  exit 1
}

# 健康检查: proxy 与两侧后端
HC=$(curl -s --max-time 5 http://localhost:8000/healthcheck 2>/dev/null)
[ -n "$HC" ] || { echo "[ERROR] proxy(8000) 未响应, 先启动: bash scripts/server/start_proxy.sh"; exit 1; }
echo "proxy healthcheck: $HC"

# ---------- 请求 1: P(种缓存, 324 tok, max_tokens=1) ----------
PP1=$(wc -l < logs/server/p_llama.log | tr -d " ")   # P 请求开始时 P 侧行数
DP1=$(wc -l < logs/server/d_llama.log | tr -d " ")   # P 请求开始时 D 侧行数
{
  echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_p.json'
  curl -s --max-time 120 http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_p.json | tee logs/curl/resp_p.json
  echo
} 2>&1 | tee logs/curl/curl_p_screen.txt
echo "[P done] (P 分界行号 p_llama.log:$PP1 / d_llama.log:$DP1)"

sleep 8   # 等 P 侧 KV 迁移完成 + 两侧收尾日志落盘(含 D 侧 decode 与双侧释放/归档)

# ---------- 请求 2: R(五块生命周期, 486 tok, max_tokens=35) ----------
PR1=$(wc -l < logs/server/p_llama.log | tr -d " ")
DR1=$(wc -l < logs/server/d_llama.log | tr -d " ")
{
  echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_r.json'
  curl -s --max-time 180 http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @scripts/curl/req_r.json | tee logs/curl/resp_r.json
  echo
} 2>&1 | tee logs/curl/curl_r_screen.txt
echo "[R done] (R 分界行号 p_llama.log:$PR1 / d_llama.log:$DR1)"

sleep 10  # 等收尾日志落盘 + [KVS] 落盘线程 flush

# ---------- [KVC] 轨迹拆解(每侧三段: startup / reqp / reqr, 覆盖更新) ----------
PP_TOT=$(wc -l < logs/server/p_llama.log | tr -d " ")
DP_TOT=$(wc -l < logs/server/d_llama.log | tr -d " ")
STRIP='s/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //'

head -n "$PP1"                 logs/server/p_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_p_startup.log
sed -n "$((PP1+1)),${PR1}p"     logs/server/p_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_p_reqp.log
sed -n "$((PR1+1)),${PP_TOT}p"  logs/server/p_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_p_reqr.log

head -n "$DP1"                 logs/server/d_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_d_startup.log
sed -n "$((DP1+1)),${DR1}p"     logs/server/d_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_d_reqp.log
sed -n "$((DR1+1)),${DP_TOT}p"  logs/server/d_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > logs/patchs/kvc_d_reqr.log

# [KVS] 归档横幅留痕(双侧 [KVS] 均保留)
grep "\[KVS\]" logs/server/p_llama.log | sed -E "$STRIP" > logs/patchs/kvs_p_archive_lines.log
grep "\[KVS\]" logs/server/d_llama.log | sed -E "$STRIP" > logs/patchs/kvs_d_archive_lines.log

echo "== [KVC] 双侧轨迹提取完成(六段 + 双侧 [KVS] 留痕) =="
wc -l logs/patchs/kvc_p_startup.log logs/patchs/kvc_d_startup.log \
      logs/patchs/kvc_p_reqp.log logs/patchs/kvc_p_reqr.log \
      logs/patchs/kvc_d_reqp.log logs/patchs/kvc_d_reqr.log \
      logs/patchs/kvs_p_archive_lines.log logs/patchs/kvs_d_archive_lines.log 2>/dev/null | tail -9
