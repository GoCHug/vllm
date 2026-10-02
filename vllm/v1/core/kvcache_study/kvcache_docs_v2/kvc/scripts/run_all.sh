#!/bin/bash
# ==============================================================================
# run_all.sh —— kvc 一键实验（容器内执行；产物回收/杀服务/撤补丁为独立后续步骤）
#
# 拓扑: 单机 PP2×TP2（与 ../kvc/、../kvc_offline/ 同形态, 占满 4 卡, 直发 :8000）
# 五阶段:
#   [1/5] patch:   apply_patches.sh（kvc 01~07 管理侧打印 + 08 v2 物理KV原样归档）
#   [2/5] 起服务:  start.sh（自动 export KVC_SAVE_KV=1 + KVC_SAVE_DIR=<工作区>/tensors）
#                  → 就绪（500s 超时; 四 worker 物理池横幅×4）
#   [3/5] 发请求:  curl_p_r.sh（P/R 双请求 + 打屏留痕 + 三段 [KVC] 轨迹 + 等归档）
#   [4/5] 验归档:  8 个 .pt（4 worker × P/R）; [KVS] 行为留痕
#   [5/5] 容器内初检: inspect_kv_tensors.py 列表模式 + 自检 + P/R 公共块比对
# （后续手动步骤: scripts/pull_artifacts.sh pack|fetch → scripts/stop.sh →
#   patch/revert_patches.sh —— 实验完毕保持容器源码未改动）
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc
#   setsid nohup bash scripts/run_all.sh > log/run_all_screen.log 2>&1 < /dev/null &
#   tail -f log/run_all_screen.log
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
mkdir -p log tensors
echo "===== [run_all] $(date '+%F %T') kvc 物理KV原样归档实验开始 ====="
[ -f log/req_p.json ] && [ -f log/req_r.json ] || { echo "[FATAL] req_p/req_r 缺失"; exit 1; }
rm -f log/kv_*.pt log/kvc_*.log log/curl_screen.log log/resp_*.json log/kvs_archive_lines.log log/inspect_*.out 2>/dev/null
rm -f tensors/kv_*.pt tensors/manifest.json 2>/dev/null

echo "===== [1/5] 打补丁（kvc 01~07 管理侧 + 08 v2 TERM 物理KV原样归档） ====="
bash patch/apply_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/5] 启动服务（单机 PP2×TP2 占 4 卡, KVC_SAVE_KV=1） ====="
bash scripts/start.sh
for i in $(seq 1 50); do
  sleep 10
  if grep -q "Application startup complete" log/llama-3-8b.log 2>/dev/null; then
    echo "[OK] 服务就绪 (等待 $((i*10))s)"
    NR=$(grep -c "物理侧 KV Cache 分配开始" log/llama-3-8b.log 2>/dev/null || true)
    [ "$NR" = 4 ] && echo "  四 worker 物理池分配横幅: ${NR} ✓" \
                  || echo "  [WARN] 物理池横幅 ${NR}/4 (PP2×TP2 应为 4)"
    KS=$(grep -c "物理KV原样归档已启用" log/llama-3-8b.log 2>/dev/null || true)
    echo "  [KVS] 归档启用横幅: $KS/4 (首个前向后打印)"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" log/llama-3-8b.log 2>/dev/null; then
    echo "[FATAL] 启动失败, 见 log/llama-3-8b.log 尾部:"; tail -30 log/llama-3-8b.log; exit 1
  fi
  [ "$i" = 50 ] && { echo "[FATAL] 500s 未就绪"; tail -30 log/llama-3-8b.log; exit 1; }
done

echo "===== [3/5] 发送 P/R 双请求（直发 :8000） ====="
bash scripts/curl_p_r.sh || echo "[WARN] 请求发送返回非零, 继续检查归档"

echo "===== [4/5] 检查归档落盘 (expect: 8 个 kv_pp?tp?_s?_*.pt) ====="
for i in $(seq 1 12); do
  N=$(ls tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 8 ] && break
  echo "  ... 等待归档 flush ($N/8), 10s 后重试"
  sleep 10
done
N=$(ls tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
if [ "$N" -lt 8 ]; then
  echo "[WARN] 归档仅 $N/8 —— [KVS] 行为检查: "
  grep "\[KVS\]" log/llama-3-8b.log 2>/dev/null | tail -10 || true
else
  echo "[OK] 8 个归档就位"
fi
ls -la tensors/ 2>/dev/null || true

echo "===== [5/5] 容器内初检 ====="
echo "-- 查看器 selftest --"
python3 scripts/inspect_kv_tensors.py --selftest 2>&1 | tail -2 || echo "[WARN] selftest 异常"
echo "-- 归档列表 --"
python3 scripts/inspect_kv_tensors.py --dir tensors || echo "[WARN] 列表异常"
echo "-- P/R 公共块比对 (前缀缓存复用一致性, 每 worker 一次) --"
for P_F in $(ls tensors/*_s1_*.pt 2>/dev/null); do
  PP=$(echo "$P_F" | grep -oE "pp[0-9]tp[0-9]")
  R_F=$(ls tensors/kv_${PP}_s2_*.pt 2>/dev/null | head -1)
  [ -n "$R_F" ] && { echo "  [worker $PP]"; python3 scripts/inspect_kv_tensors.py --dir tensors --compare "$P_F" "$R_F" 2>/dev/null | tail -4; }
done

echo "===== [DONE] $(date '+%F %T') kvc 产物就绪: log/ + tensors/ ====="
echo "       回收: 容器内 bash scripts/pull_artifacts.sh pack | 主机侧 bash scripts/pull_artifacts.sh fetch"
echo "       收尾: bash scripts/stop.sh && bash patch/revert_patches.sh (保持容器源码未改动)"
