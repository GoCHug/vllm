#!/bin/bash
# curl_pd.sh —— 经 proxy(8000) 依次发送 P、R 双请求, 落盘 curl 打屏/响应, 并提取双侧 [KVC] 轨迹
# 前提: P(8100)/D(8200)/proxy(8000) 三者均就绪; log/req_p.json 与 log/req_r5.json 已就位
# 产物(log/): 边界文件 ×4 + resp/screen ×2 + 双侧各三段轨迹(startup/req1/req2)共 6 个轨迹文件
cd "$(dirname "$0")/.." || exit 1

[ -f log/req_p.json ] && [ -f log/req_r5.json ] || {
  echo "[ERROR] 缺少 log/req_p.json / req_r5.json"
  echo "        先执行: cp ../kvc/log/req_p.json ../kvc/log/req_r5.json log/"
  exit 1
}

# 健康检查: proxy 与两侧后端
HC=$(curl -s --max-time 5 http://localhost:8000/healthcheck 2>/dev/null)
[ -n "$HC" ] || { echo "[ERROR] proxy(8000) 未响应, 先启动: bash scripts/start_proxy.sh"; exit 1; }
echo "proxy healthcheck: $HC"

# ---------- 请求 1: P(种缓存, 324 tok, max_tokens=1) ----------
wc -l < log/p_llama.log | tr -d " " > log/p_start_p.txt   # P 请求开始时 P 侧行数
wc -l < log/d_llama.log | tr -d " " > log/d_start_p.txt   # P 请求开始时 D 侧行数
{
  echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json'
  curl -s --max-time 120 http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json | tee log/resp_p.json
  echo
} 2>&1 | tee log/curl_p_screen.txt
echo "[P done]"

sleep 8   # 等 P 侧 KV 迁移完成 + 两侧收尾日志落盘(含 D 侧 decode 与双侧释放)

# ---------- 请求 2: R(五块生命周期, 486 tok, max_tokens=35) ----------
wc -l < log/p_llama.log | tr -d " " > log/p_start_r.txt
wc -l < log/d_llama.log | tr -d " " > log/d_start_r.txt
{
  echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r5.json'
  curl -s --max-time 180 http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r5.json | tee log/resp_r5.json
  echo
} 2>&1 | tee log/curl_r5_screen.txt
echo "[R done]"

sleep 5   # 等收尾日志落盘

# ---------- [KVC] 轨迹拆解(每侧三段: startup / req1 / req2) ----------
PP1=$(cat log/p_start_p.txt); PR1=$(cat log/p_start_r.txt)
DP1=$(cat log/d_start_p.txt); DR1=$(cat log/d_start_r.txt)
PP_TOT=$(wc -l < log/p_llama.log | tr -d " ")
DP_TOT=$(wc -l < log/d_llama.log | tr -d " ")
STRIP='s/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //'

head -n "$PP1"              log/p_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_p_startup.log
sed -n "$((PP1+1)),${PR1}p"  log/p_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_p_req1.log
sed -n "$((PR1+1)),${PP_TOT}p" log/p_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_p_req2.log

head -n "$DP1"              log/d_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_d_startup.log
sed -n "$((DP1+1)),${DR1}p"  log/d_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_d_req1.log
sed -n "$((DR1+1)),${DP_TOT}p" log/d_llama.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_d_req2.log

echo "== [KVC] 双侧轨迹提取完成 =="
wc -l log/kvc_p_startup.log log/kvc_d_startup.log \
      log/kvc_p_req1.log log/kvc_p_req2.log \
      log/kvc_d_req1.log log/kvc_d_req2.log