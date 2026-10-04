#!/bin/bash
# ==============================================================================
# run_all.sh —— kvc_1p1d PD 分离(1P+1D) KVCache 打印/归档实验一键执行(容器内)
#
# 拓扑: 1P(prefill producer, 卡0/:8100) + 1D(decode consumer, 卡1/:8200) + proxy(:8000 双发)
# 八阶段:
#   [1/8] patch:   apply_patches.sh(kvc 01~07 管理侧打印 + 08 PD 归档版块结构 kvt4-raw)
#   [2/8] P 侧:    start_p.sh(自动 export KVC_SAVE_KV=1 + KVC_SAVE_DIR) → 就绪
#   [3/8] D 侧:    start_d.sh → 就绪
#   [4/8] proxy:   start_proxy.sh(:8000 → P:8100/D:8200) → healthcheck
#   [5/8] 发请求:  curl_pd.sh(P->8s->R; 双侧 [KVC] 六段轨迹 + [KVS] 留痕落 logs/patchs/)
#   [6/8] 验归档:  tensors/{P,D}/req{seq}_{rid尾8}/kv_pp0tp0.pt 共 4 个(P×2 + D×2)
#   [7/8] 容器内初检: analysis/ 5 个检查器 —— inspect_kv_tensors_{p,d}.py(查看器)
#                  + inspect_prefix_{p,d}.py(侧内前缀复用/重算一致性)
#                  + inspect_p2d.py(P→D 传输正确性: Tx 区逐位) -> logs/analysis/*.out
#   [8/8] 打包:    tar logs/ + tensors/ -> kvc_1p1d_bundle.tar.gz + md5(主机侧 recover/pull_artifacts.sh fetch 拉回)
#
# 日志布局(logs/ 四子目录, 与 scripts/ 前四目录一一对应):
#   server/  p_llama.log / d_llama.log / proxy.log / run_all_screen.log(本脚本留痕)
#   patchs/  kvc_{p,d}_{startup,reqp,reqr}.log + kvs_{p,d}_archive_lines.log([KVC]/[KVS] 拆解轨迹)
#   curl/    resp_{p,r}.json + curl_{p,r}_screen.txt
#   analysis/  inspect_kv_tensors_{p,d}.out / inspect_prefix_{p,d}.out / inspect_p2d.out
# tensors/  P/req{seq}_{rid尾8}/ + D/req{seq}_{rid尾8}/ —— 同 seq 跨侧配对(rid 尾8 两侧不同)
#
# 注: apply/revert 补丁脚本默认仓库路径为本地 macOS 路径(开箱即用于本地);
#     容器内由本脚本自动导出容器路径。
# 停服务/撤补丁为手动收尾(不自动做): bash scripts/server/stop.sh && bash scripts/patchs/revert_patches.sh
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc_1p1d
#   mkdir -p logs/server   # nohup 重定向目标需先存在
#   setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
#   tail -f logs/server/run_all_screen.log
# ==============================================================================
cd "$(dirname "$0")/../.." || exit 1
# 容器路径显式导出(apply/revert 脚本默认路径为本地 macOS 路径, 容器内必须覆盖)
export VLLM_DIR=/vllm-workspace/vllm
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
mkdir -p logs/server logs/patchs logs/curl logs/analysis tensors
echo "===== [run_all] $(date '+%F %T') kvc_1p1d PD 分离(1P+1D) KVCache 实验开始 ====="
[ -f scripts/curl/req_p.json ] && [ -f scripts/curl/req_r.json ] || { echo "[FATAL] scripts/curl/req_p/req_r 缺失"; exit 1; }
# 清上一轮产物(保留 logs/server/run_all_screen.log —— 本轮正被 nohup 写入)
rm -f logs/server/p_llama.log logs/server/d_llama.log logs/server/proxy.log \
      logs/patchs/* logs/curl/* logs/analysis/* 2>/dev/null
rm -rf tensors/P tensors/D 2>/dev/null

echo "===== [1/8] 打补丁（kvc 01~07 管理侧 + 08 PD 归档版 v2 块结构） ====="
bash scripts/patchs/apply_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/8] 启动 P 侧 (卡0/8100/producer, TERM 归档 -> tensors/P/) ====="
bash scripts/server/start_p.sh
for i in $(seq 1 30); do
  sleep 10
  if grep -q "Application startup complete" logs/server/p_llama.log 2>/dev/null; then
    echo "[OK] P 侧就绪 (等待 $((i*10))s)"
    KS=$(grep -c "物理KV原样归档已启用" logs/server/p_llama.log 2>/dev/null || true)
    echo "  [KVS] 归档启用横幅: $KS/1"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" logs/server/p_llama.log 2>/dev/null; then
    echo "[FATAL] P 侧启动失败, 见 logs/server/p_llama.log 尾部:"; tail -30 logs/server/p_llama.log; exit 1
  fi
  [ "$i" = 30 ] && { echo "[FATAL] P 侧 300s 未就绪"; tail -30 logs/server/p_llama.log; exit 1; }
done

echo "===== [3/8] 启动 D 侧 (卡1/8200/consumer, TERM 归档 -> tensors/D/) ====="
bash scripts/server/start_d.sh
for i in $(seq 1 30); do
  sleep 10
  if grep -q "Application startup complete" logs/server/d_llama.log 2>/dev/null; then
    echo "[OK] D 侧就绪 (等待 $((i*10))s)"
    KS=$(grep -c "物理KV原样归档已启用" logs/server/d_llama.log 2>/dev/null || true)
    echo "  [KVS] 归档启用横幅: $KS/1"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" logs/server/d_llama.log 2>/dev/null; then
    echo "[FATAL] D 侧启动失败, 见 logs/server/d_llama.log 尾部:"; tail -30 logs/server/d_llama.log; exit 1
  fi
  [ "$i" = 30 ] && { echo "[FATAL] D 侧 300s 未就绪"; tail -30 logs/server/d_llama.log; exit 1; }
done

echo "===== [4/8] 启动 proxy (8000 -> P:8100/D:8200) ====="
bash scripts/server/start_proxy.sh
sleep 3
for i in $(seq 1 10); do
  HC=$(curl -s --max-time 5 http://localhost:8000/healthcheck 2>/dev/null)
  [ -n "$HC" ] && { echo "[OK] proxy 就绪: $HC"; break; }
  sleep 3
  [ "$i" = 10 ] && { echo "[FATAL] proxy 30s 未就绪"; tail -20 logs/server/proxy.log; exit 1; }
done

echo "===== [5/8] 发送 P/R 双请求 + 提取双侧 [KVC] 轨迹 ====="
bash scripts/curl/curl_pd.sh || echo "[WARN] 请求发送返回非零, 继续检查归档"

echo "===== [6/8] 检查归档落盘 (expect: tensors/{P,D}/req{seq}_{rid尾8}/kv_pp0tp0.pt × 4) ====="
for i in $(seq 1 12); do
  NP=$(ls tensors/P/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  ND=$(ls tensors/D/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$NP" -ge 2 ] && [ "$ND" -ge 2 ] && break
  echo "  ... 等待归档 flush (P $NP/2, D $ND/2), 10s 后重试"
  sleep 10
done
NP=$(ls tensors/P/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
ND=$(ls tensors/D/req*/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
if [ "$NP" -lt 2 ] || [ "$ND" -lt 2 ]; then
  echo "[WARN] 归档 P $NP/2 / D $ND/2 —— [KVS] 行为检查: "
  grep "\[KVS\]" logs/server/p_llama.log 2>/dev/null | tail -5 || true
  grep "\[KVS\]" logs/server/d_llama.log 2>/dev/null | tail -5 || true
else
  echo "[OK] 双侧归档就位 (P×$NP + D×$ND, 一请求一 side 子目录)"
fi
ls -laR tensors/ 2>/dev/null || true

echo "===== [7/8] 容器内初检（analysis/ 五个检查器） ====="
echo "-- P 侧 & D 侧归档查看报告（逐请求逐 block 的 K/V shape/dtype/tensor 预览） --"
python3 scripts/analysis/inspect_kv_tensors_p.py --dir tensors/P 2>/dev/null || echo "[WARN] P 查看器异常"
python3 scripts/analysis/inspect_kv_tensors_d.py --dir tensors/D 2>/dev/null || echo "[WARN] D 查看器异常"
echo "-- P 侧 & D 侧前缀复用关系（pairwise: 共享表头块 + 重算一致性） --"
python3 scripts/analysis/inspect_prefix_p.py --dir tensors/P 2>/dev/null || echo "[WARN] P 前缀检查异常"
python3 scripts/analysis/inspect_prefix_d.py --dir tensors/D 2>/dev/null || echo "[WARN] D 前缀检查异常"
echo "-- P→D 传输正确性（Tx 区逐位 torch.equal + 对端段归因） --"
python3 scripts/analysis/inspect_p2d.py --dir tensors 2>/dev/null || echo "[WARN] p2d 检查异常"

echo "===== [8/8] 打包产物 (kvc_1p1d_bundle.tar.gz = logs/ + tensors/, 供主机侧 fetch) ====="
tar -czf kvc_1p1d_bundle.tar.gz logs/ tensors/
echo "[DONE] $(ls -la kvc_1p1d_bundle.tar.gz | awk '{print $5}') B -> kvc_1p1d_bundle.tar.gz"
md5sum kvc_1p1d_bundle.tar.gz 2>/dev/null || md5 -q kvc_1p1d_bundle.tar.gz

echo "===== [DONE] $(date '+%F %T') kvc_1p1d 产物就绪并已打包: logs/ + tensors/ ====="
echo "       回收: 主机侧 bash scripts/recover/pull_artifacts.sh fetch (经 5557 隧道拉回)"
echo "       收尾: bash scripts/server/stop.sh && bash scripts/patchs/revert_patches.sh (保持容器源码未改动)"
