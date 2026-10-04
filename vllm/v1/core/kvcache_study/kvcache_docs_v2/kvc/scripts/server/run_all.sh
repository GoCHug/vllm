#!/bin/bash
# ==============================================================================
# run_all.sh —— kvc 一键实验（容器内执行；产物自动打包, 主机 fetch/杀服务/撤补丁为后续步骤）
#
# 拓扑: 单机 PP2×TP2（占满 4 卡, 直发 :8000）
# 六阶段:
#   [1/6] patch:   apply_patches.sh（kvc 01~07 管理侧打印 + 08 v2.5 物理KV原样归档, 请求子目录 + 块-行映射）
#   [2/6] 起服务:  server/start.sh（自动 export KVC_SAVE_KV=1 + KVC_SAVE_DIR=<工作区>/tensors）
#                  → 就绪（500s 超时; 四 worker 物理池横幅×4）
#   [3/6] 发请求:  curl/curl_p_r.sh（P/R 双请求 + 打屏留痕 + 三段 [KVC] 轨迹 + 等归档）
#   [4/6] 验归档:  tensors/req{seq}_{rid尾8}/kv_pp{p}tp{t}.pt 共 8 个（4 worker × P/R）
#   [5/6] 容器内初检: analysis/ 两个检查器 —— inspect_kv_tensors.py(查看 -> logs/analysis/inspect_kv_tensors.out)
#                  + inspect_prefix.py(前缀复用关系检查 -> logs/analysis/inspect_prefix.out)
#   [6/6] 打包:    tar logs/ + tensors/ -> kvc_bundle.tar.gz + md5（主机侧 recover/pull_artifacts.sh fetch 拉回）
#
# 日志布局（logs/ 四子目录, 与 scripts/ 一一对应）:
#   server/    llama-3-8b.log(服务全量) + run_all_screen.log(本脚本留痕)        <- scripts/server/
#   patchs/    kvc_startup/kvc_p/kvc_r/kvs_archive_lines.log（patch 打印日志）  <- scripts/patchs/
#   curl/      resp_p/resp_r.json + curl_screen.log（响应与打屏）                <- scripts/curl/
#   analysis/  inspect_kv_tensors.out / inspect_prefix.out（离线检查产物）       <- scripts/analysis/
# （后续手动步骤: 主机侧 scripts/recover/pull_artifacts.sh fetch 回收产物 -> 容器 scripts/server/stop.sh
#   -> scripts/patchs/revert_patches.sh —— 实验完毕保持容器源码未改动; 本脚本已导出容器路径）
#
# 注: apply/revert 补丁脚本默认仓库路径为本地 macOS 路径(开箱即用于本地);
#     容器内由本脚本/手动按 README 传 VLLM_DIR=/vllm-workspace/vllm 等覆盖。
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc
#   mkdir -p logs/server   # nohup 重定向目标需先存在
#   setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
#   tail -f logs/server/run_all_screen.log
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1
# 容器路径显式导出(apply/revert 脚本默认路径为本地 macOS 路径, 容器内必须覆盖;
# 本地直接跑 apply/revert 时无需任何传参)
export VLLM_DIR=/vllm-workspace/vllm
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
mkdir -p logs/server logs/patchs logs/curl logs/analysis tensors
echo "===== [run_all] $(date '+%F %T') kvc 物理KV原样归档实验开始 ====="
[ -f scripts/curl/req_p.json ] && [ -f scripts/curl/req_r.json ] || { echo "[FATAL] scripts/curl/req_p/req_r 缺失"; exit 1; }
# 清上一轮产物(保留 logs/server/run_all_screen.log —— 本轮正被 nohup 写入)
rm -f logs/server/llama-3-8b.log logs/patchs/* logs/curl/* logs/analysis/* 2>/dev/null
rm -rf log 2>/dev/null   # 兼容旧 log/ 平铺布局残留(现为 logs/)
rm -rf tensors/req* 2>/dev/null; rm -f tensors/kv_*.pt 2>/dev/null

echo "===== [1/6] 打补丁（kvc 01~07 管理侧 + 08 v2.5 TERM 物理KV原样归档, 请求子目录 + 块-行映射） ====="
bash scripts/patchs/apply_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/6] 启动服务（单机 PP2×TP2 占 4 卡, KVC_SAVE_KV=1） ====="
bash scripts/server/start.sh
for i in $(seq 1 50); do
  sleep 10
  if grep -q "Application startup complete" logs/server/llama-3-8b.log 2>/dev/null; then
    echo "[OK] 服务就绪 (等待 $((i*10))s)"
    NR=$(grep -c "物理侧 KV Cache 分配开始" logs/server/llama-3-8b.log 2>/dev/null || true)
    [ "$NR" = 4 ] && echo "  四 worker 物理池分配横幅: ${NR} ✓" \
                  || echo "  [WARN] 物理池横幅 ${NR}/4 (PP2×TP2 应为 4)"
    KS=$(grep -c "物理KV原样归档已启用" logs/server/llama-3-8b.log 2>/dev/null || true)
    echo "  [KVS] 归档启用横幅: $KS/4 (首个前向后打印)"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" logs/server/llama-3-8b.log 2>/dev/null; then
    echo "[FATAL] 启动失败, 见 logs/server/llama-3-8b.log 尾部:"; tail -30 logs/server/llama-3-8b.log; exit 1
  fi
  [ "$i" = 50 ] && { echo "[FATAL] 500s 未就绪"; tail -30 logs/server/llama-3-8b.log; exit 1; }
done

echo "===== [3/6] 发送 P/R 双请求（直发 :8000） ====="
bash scripts/curl/curl_p_r.sh || echo "[WARN] 请求发送返回非零, 继续检查归档"

echo "===== [4/6] 检查归档落盘 (expect: tensors/req{seq}_{rid尾8}/kv_pp{p}tp{t}.pt × 8) ====="
for i in $(seq 1 12); do
  N=$(ls tensors/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 8 ] && break
  echo "  ... 等待归档 flush ($N/8), 10s 后重试"
  sleep 10
done
N=$(ls tensors/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
if [ "$N" -lt 8 ]; then
  echo "[WARN] 归档仅 $N/8 —— [KVS] 行为检查: "
  grep "\[KVS\]" logs/server/llama-3-8b.log 2>/dev/null | tail -10 || true
else
  echo "[OK] 8 个归档就位（一请求一子目录, 目录内 4 worker 各 1 份）"
fi
ls -laR tensors/ 2>/dev/null || true

echo "===== [5/6] 容器内初检 ====="
echo "-- 归档查看报告（逐请求逐 worker 逐 block 的 K/V shape/dtype/tensor 预览） --"
python3 scripts/analysis/inspect_kv_tensors.py --dir tensors || echo "[WARN] 报告生成异常"
echo "-- 前缀复用关系检查（pairwise: 命中关系 + 重算段 ULP 一致性） --"
python3 scripts/analysis/inspect_prefix.py --dir tensors || echo "[WARN] 前缀检查生成异常"

echo "===== [6/6] 打包产物 (kvc_bundle.tar.gz = logs/ + tensors/, 供主机侧 fetch) ====="
tar -czf kvc_bundle.tar.gz logs/ tensors/
echo "[DONE] $(ls -la kvc_bundle.tar.gz | awk '{print $5}') B -> kvc_bundle.tar.gz"
md5sum kvc_bundle.tar.gz 2>/dev/null || md5 -q kvc_bundle.tar.gz

echo "===== [DONE] $(date '+%F %T') kvc 产物就绪并已打包: logs/ + tensors/ ====="
echo "       回收: 主机侧 bash scripts/recover/pull_artifacts.sh fetch (经 5557 隧道拉回)"
echo "       收尾: bash scripts/server/stop.sh && bash scripts/patchs/revert_patches.sh (保持容器源码未改动)"
