#!/bin/bash
# ==============================================================================
# run_all.sh —— PD 分离 KVCache 正确性实验一键执行(容器内; 可 nohup 后台跑)
#
# 全流程:
#   1. 打 PD 补丁(kvc 01~08 + 09 指纹)
#   2. 起 P(卡0/8100/producer) -> 就绪 -> 起 D(卡1/8200/consumer) -> 就绪 -> 起 proxy(8000)
#   3. 准备 req_p/req_r -> curl_pd.sh 发双请求(含双侧 [KVC] 轨迹提取)
#   4. stop_pd.sh 杀全部服务
#   5. revert_pd_patches.sh 撤补丁(源码还原干净)
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc_pd
#   setsid nohup bash scripts/run_all.sh > log/run_all_screen.log 2>&1 < /dev/null &
#   tail -f log/run_all_screen.log    # 观察进度
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
mkdir -p log
export VLLM_DIR=/vllm-workspace/vllm
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend

echo "===== [run_all] $(date '+%F %T') PD KVCache 正确性实验开始 ====="
[ -f log/req_p.json ] || cp ../kvc/log/req_p.json log/
[ -f log/req_r.json ] || cp ../kvc/log/req_r.json log/
[ -f log/req_p.json ] && [ -f log/req_r.json ] || { echo "[FATAL] req_p/req_r 缺失"; exit 1; }

echo "===== [1/6] 打 PD 补丁 (kvc 01~08 + 09 指纹) ====="
bash patch/apply_pd_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/6] 启动 P 侧 (卡0/8100/producer) ====="
bash scripts/start_p.sh
for i in $(seq 1 30); do
  sleep 10
  if grep -q "Application startup complete" log/p_llama.log 2>/dev/null; then
    echo "[OK] P 侧就绪 (等待 $((i*10))s)"; break
  fi
  if grep -qE "Traceback|EngineCore failed" log/p_llama.log 2>/dev/null; then
    echo "[FATAL] P 侧启动失败, 见 log/p_llama.log 尾部:"; tail -30 log/p_llama.log; exit 1
  fi
  [ "$i" = 30 ] && { echo "[FATAL] P 侧 300s 未就绪"; tail -30 log/p_llama.log; exit 1; }
done

echo "===== [3/6] 启动 D 侧 (卡1/8200/consumer) ====="
bash scripts/start_d.sh
for i in $(seq 1 30); do
  sleep 10
  if grep -q "Application startup complete" log/d_llama.log 2>/dev/null; then
    echo "[OK] D 侧就绪 (等待 $((i*10))s)"; break
  fi
  if grep -qE "Traceback|EngineCore failed" log/d_llama.log 2>/dev/null; then
    echo "[FATAL] D 侧启动失败, 见 log/d_llama.log 尾部:"; tail -30 log/d_llama.log; exit 1
  fi
  [ "$i" = 30 ] && { echo "[FATAL] D 侧 300s 未就绪"; tail -30 log/d_llama.log; exit 1; }
done

echo "===== [4/6] 启动 proxy (8000 -> P:8100/D:8200) ====="
bash scripts/start_proxy.sh
sleep 3
for i in $(seq 1 10); do
  HC=$(curl -s --max-time 5 http://localhost:8000/healthcheck 2>/dev/null)
  [ -n "$HC" ] && { echo "[OK] proxy 就绪: $HC"; break; }
  sleep 3
  [ "$i" = 10 ] && { echo "[FATAL] proxy 30s 未就绪"; tail -20 log/proxy.log; exit 1; }
done

echo "===== [5/6] 发送 P/R 双请求 + 提取双侧 [KVC] 轨迹 ====="
bash scripts/curl_pd.sh || { echo "[FATAL] 请求发送失败"; }
sleep 5

echo "===== [6/6] 收尾: 杀服务 + 撤补丁 ====="
bash scripts/stop_pd.sh
bash patch/revert_pd_patches.sh || echo "[WARN] 撤补丁异常, 请手动检查"
pgrep -af "v[l]lm serve" || echo "[OK] 无 vllm 进程残留"
grep -q "\[KVC\]" /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py \
  && echo "[WARN] vllm-ascend 源码仍有 [KVC] 残留" \
  || echo "[OK] vllm-ascend 源码干净"
grep -q "\[KVC\]" /vllm-workspace/vllm/vllm/v1/request.py \
  && echo "[WARN] vllm 源码仍有 [KVC] 残留" \
  || echo "[OK] vllm 源码干净"

echo "===== [DONE] $(date '+%F %T') 指纹文件: log/kvc_p_req*.log / log/kvc_d_req*.log ====="
