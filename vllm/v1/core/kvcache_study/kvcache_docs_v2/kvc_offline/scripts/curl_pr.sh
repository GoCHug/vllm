#!/bin/bash
# ==============================================================================
# curl_pr.sh —— 依次发送 P、R 双请求（直发本机 :8000 单实例, 无 proxy），
#              落盘响应体, 并提取三条 [KVC] 拆解轨迹（同 ../kvc/scripts/curl_p_r.sh 模式）
# 前提: 服务已就绪(scripts/start.sh); log/req_p.json 与 log/req_r.json 已拷贝就位
# 产物(log/): resp_p/r.json + kvc_startup.log + kvc_p.log + kvc_r.log
#   + kvb_archive_lines.log（[KVB] 归档行为留痕, 行内自带 pp/tp 标签）
# 注意: 单机 PP2×TP2 四 worker 的 [KVC] 行交织在同一轨迹文件（[FPB]/[FP] 无
#       worker 标签)—— 检查器 C1 用集合匹配消解 worker 归属, 无需按行拆分。
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
[ -f log/req_p.json ] && [ -f log/req_r.json ] || { echo "[ERROR] 缺少 log/req_p.json / req_r.json"; echo "        先执行: cp ../kvc/log/req_p.json ../kvc/log/req_r.json log/"; exit 1; }

# ---------- 请求 1: P（前缀种块, 324 tok, max_tokens=1） ----------
P1=$(wc -l < log/llama-3-8b.log | tr -d " ")
echo "curl -s http://localhost:8000/v1/completions -H \"Content-Type: application/json\" -d @log/req_p.json"
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json | tee log/resp_p.json
echo
echo "[P done] (P 分界行号: $P1)"

sleep 6   # 等 P 结束: 满块带哈希留在缓存池, 全部块释放挂队尾

# ---------- 请求 2: R（五块生命周期: 前缀 HIT + 35 decode 步） ----------
R1=$(wc -l < log/llama-3-8b.log | tr -d " ")
echo "curl -s http://localhost:8000/v1/completions -H \"Content-Type: application/json\" -d @log/req_r.json"
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r.json | tee log/resp_r.json
echo
echo "[R done] (R 分界行号: $R1)"

sleep 10  # 等收尾日志(释放逆序 + [KVB] 后台落盘线程 flush)落盘

# ---------- [KVC] 轨迹拆解（llama-3-8b.log -> 三条轨迹, 剥离进程 pid 前缀） ----------
TOT=$(wc -l < log/llama-3-8b.log | tr -d " ")
STRIP='s/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //'
head -n "$P1" log/llama-3-8b.log              | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_startup.log
sed -n "$((P1+1)),${R1}p" log/llama-3-8b.log  | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_p.log
sed -n "$((R1+1)),${TOT}p" log/llama-3-8b.log | grep "\[KVC\]" | sed -E "$STRIP" > log/kvc_r.log

# [KVB] 归档行为行单独留痕（行内自带 pp/tp/s 标签, 不分 worker）
grep "\[KVB\]" log/llama-3-8b.log | sed -E "$STRIP" > log/kvb_archive_lines.log

echo "== [KVC] 轨迹提取完成（三段 + 归档留痕） =="
wc -l log/kvc_startup.log log/kvc_p.log log/kvc_r.log log/kvb_archive_lines.log
