#!/bin/bash
# curl_pp2.sh —— 经 proxy(8000) 发 P/R 双请求, 落盘响应, 提取双侧 [KVC] 轨迹(按 rank 拆分)
# pp2tp2 特有: TP2 双 rank worker 进程日志交织在同一个 {p,d}_llama.log —— 按
# [KVC] 行内的 dev=npu:{r} 标签把轨迹拆成 kvc_{p,d}{0,1}_{reqp,reqr}.log 八路。
# (EngineCore pid 前缀不可靠: 调度在 EngineCore(单), 物理打印在 worker(双) —— 同一 [KVC]
#  行的 logger 是 worker 进程, 但行首 pid 前缀是 multiprocessing 日志转发统一格式,
#  不能区分 rank; dev= 是 08 号补丁打印里自带的设备标签, 可靠。)
# 注意: 入队 [ENQ]/调度类打印不带 dev 标签(EngineCore 进程打印) —— 只归 rank 无差别侧
#       轨迹文件 kvc_{p,d}_req{p,r}.log 中按"有 dev 标签的行才分 rank"处理:
#       带 dev 的行 → kvc_{side}{r}_... ; 不带的行 → 落到两路共享 kvc_{side}_req{tag}.log
#       (检查器 B1 只用带 dev 的 [FPB]/[FP]/TERM 行, 对账不受影响)
cd "$(dirname "$0")/.." || exit 1

[ -f log/req_p.json ] && [ -f log/req_r.json ] || {
  echo "[ERROR] 缺少 log/req_p.json / req_r.json"
  echo "        先执行: cp ../kvc/log/req_p.json ../kvc/log/req_r.json log/"
  exit 1
}

HC=$(curl -s --max-time 5 http://localhost:8000/healthcheck 2>/dev/null)
[ -n "$HC" ] || { echo "[ERROR] proxy(8000) 未响应, 先启动: bash scripts/start_proxy.sh"; exit 1; }
echo "proxy healthcheck: $HC"

# ---------- 请求 1: P(种缓存, 324 tok, max_tokens=1) ----------
PP1=$(wc -l < log/p_llama.log | tr -d " ")
DP1=$(wc -l < log/d_llama.log | tr -d " ")
{
  echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json'
  curl -s --max-time 120 http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_p.json | tee log/resp_p.json
  echo
} 2>&1 | tee log/curl_p_screen.txt
echo "[P done]"

sleep 8   # 等 KV 迁移 + 两侧收尾(含 [KVB] 后台落盘)

# ---------- 请求 2: R(五块生命周期, 486 tok, max_tokens=35) ----------
PR1=$(wc -l < log/p_llama.log | tr -d " ")
DR1=$(wc -l < log/d_llama.log | tr -d " ")
{
  echo '$ curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r.json'
  curl -s --max-time 180 http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @log/req_r.json | tee log/resp_r.json
  echo
} 2>&1 | tee log/curl_r_screen.txt
echo "[R done]"

sleep 5

# ---------- [KVC] 轨迹拆解: 每侧三段 × 分 rank ----------
PP_TOT=$(wc -l < log/p_llama.log | tr -d " ")
DP_TOT=$(wc -l < log/d_llama.log | tr -d " ")
STRIP='s/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //'

extract() {  # extract <side> <start> <end> <rtag>
  local side=$1 start=$2 end=$3 rtag=$4
  local seg
  seg=$(sed -n "${start},${end}p" log/${side}_llama.log | grep "\[KVC\]" | sed -E "$STRIP")
  # 共享段(无 dev 标签行: 入队/调度/[FPB]/[FP] —— 09 指纹行不带 dev, 双 rank 交织,
  # 由检查器 B1 集合匹配消解 rank)
  echo "$seg" | grep -v "dev=npu:" > log/kvc_${side}_${rtag}.log
  # 分 rank 段: TERM 概览行带 dev=npu:{r}(真实形态 "dev=npu:0 逐层按块:": 空格续接)
  for r in 0 1; do
    echo "$seg" | grep -E "dev=npu:${r}(,| |$)" > log/kvc_${side}${r}_${rtag}.log
  done
}

extract p "$PP1" "$PR1" reqp
extract p "$((PR1+1))" "$PP_TOT" reqr
extract d "$DP1" "$DR1" reqp
extract d "$((DR1+1))" "$DP_TOT" reqr

# [KVB] 归档行为行单独留痕(不分 rank, 行内自带 r{N} 标记)
sed -n "1,${PP_TOT}p" log/p_llama.log | grep "\[KVB\]" | sed -E "$STRIP" > log/kvb_archive_lines.log
sed -n "1,${DP_TOT}p" log/d_llama.log | grep "\[KVB\]" | sed -E "$STRIP" >> log/kvb_archive_lines.log

echo "== [KVC] 轨迹提取完成(八路 + 共享两路 + 归档留痕) =="
wc -l log/kvc_p_reqp.log log/kvc_p0_reqp.log log/kvc_p1_reqp.log \
      log/kvc_d_reqp.log log/kvc_d0_reqp.log log/kvc_d1_reqp.log 2>/dev/null
