#!/bin/bash
# curl_p_r.sh —— 依次发送 P、R 双请求, 落盘响应体, 并提取三条 [KVC] 拆解轨迹
# 前提: 服务已就绪(scripts/start.sh); log/req_p.json 与 log/req_r.json 已生成(scripts/gen_cn_requests.py --gen)
# 产物(log/): resp_p.json, resp_r.json, kvc_startup.log, kvc_p.log, kvc_r.log
# （P/R 分界行号仅作内存变量参与轨迹切分, 不落盘）
cd "$(dirname "$0")/.." || exit 1
[ -f log/req_p.json ] && [ -f log/req_r.json ] || { echo "[ERROR] 缺少 log/req_p.json / log/req_r.json, 先执行: python3 scripts/gen_cn_requests.py --gen"; exit 1; }

# ---------- 请求 1: P（缓冲 2 块） ----------
P1=$(wc -l < log/llama-3-8b.log | tr -d " ")   # P 分界行号(启动段结束)
echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json'
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json | tee log/resp_p.json
echo
echo "[P done] (P 分界行号: $P1)"

sleep 6   # 等 P 结束: 满块带哈希留在缓存池, 全部块释放挂队尾

# ---------- 请求 2: R（五块生命周期） ----------
R1=$(wc -l < log/llama-3-8b.log | tr -d " ")   # R 分界行号(P 段结束)
echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r.json'
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r.json | tee log/resp_r.json
echo
echo "[R done] (R 分界行号: $R1)"

sleep 3   # 等收尾日志(释放逆序)落盘

# ---------- [KVC] 轨迹拆解（llama-3-8b.log -> 三条轨迹, 剥离进程 pid 前缀） ----------
TOT=$(wc -l < log/llama-3-8b.log | tr -d " ")
STRIP='s/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //'
head -n "$P1" log/llama-3-8b.log              | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_startup.log
sed -n "$((P1+1)),${R1}p" log/llama-3-8b.log  | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_p.log
sed -n "$((R1+1)),${TOT}p" log/llama-3-8b.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_r.log
echo "== [KVC] 轨迹提取完成 =="
wc -l log/kvc_startup.log log/kvc_p.log log/kvc_r.log