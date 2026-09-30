#!/bin/bash
# ==============================================================================
# run_offline_all.sh —— kvc_pd_offline 张量归档实验一键执行(容器内; 可 nohup 后台跑)
#
# 与 kvc_pd/scripts/run_all.sh 的差异(七阶段):
#   [1/7] patch:  apply_offline_patches.sh(kvc 01~08 → 09 指纹 → 10 张量归档)
#   [2/7] 起 P 卡0/8100 → 就绪   (KVC_DUMP_TENSORS=1 在本脚本导出, 随 start 脚本继承)
#   [3/7] 起 D 卡1/8200 → 就绪
#   [4/7] 起 proxy :8000 + 发 req_p/req_r 双请求(轨迹提取同 v1)
#   [5/7] 检查归档落盘: log/tensors/kv_{P,D}_{1,2}_*.pt 4 文件就位(轮询后台线程 flush)
#   [6/7] 容器内初检: compare_fp.py(v1 裁决) + check_kv_tensors.py --selftest
#         (正式五级检回收后本机跑; 容器内只做 L0 级 .pt 可读快检)
#   [7/7] 收尾: stop_pd.sh + revert_offline_patches.sh(源码归零)
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc_pd_offline
#   setsid nohup bash scripts/run_offline_all.sh > log/run_offline_screen.log 2>&1 < /dev/null &
#   tail -f log/run_offline_screen.log
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
mkdir -p log log/tensors
export VLLM_DIR=/vllm-workspace/vllm
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
export KVC_DUMP_TENSORS=1          # 10 号补丁开关: TERM 张量归档(随子进程继承)
export KVC_DUMP_DIR=log/tensors    # 相对 cwd=本工作区根

echo "===== [run_offline_all] $(date '+%F %T') kvc_pd_offline 张量归档实验开始 ====="
[ -f log/req_p.json ] || cp ../kvc/log/req_p.json log/
[ -f log/req_r.json ] || cp ../kvc/log/req_r.json log/
[ -f log/req_p.json ] && [ -f log/req_r.json ] || { echo "[FATAL] req_p/req_r 缺失"; exit 1; }
# 归档目录清空(只留本轮产物)
rm -f log/tensors/*.pt log/tensors/manifest*.json 2>/dev/null

echo "===== [1/7] 打 offline 补丁 (kvc 01~08 + 09 指纹 + 10 张量归档) ====="
bash patch/apply_offline_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/7] 启动 P 侧 (卡0/8100/producer) ====="
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

echo "===== [3/7] 启动 D 侧 (卡1/8200/consumer) ====="
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

echo "===== [4/7] 启动 proxy (8000) + 发送 P/R 双请求 ====="
bash scripts/start_proxy.sh
sleep 3
for i in $(seq 1 10); do
  HC=$(curl -s --max-time 5 http://localhost:8000/healthcheck 2>/dev/null)
  [ -n "$HC" ] && { echo "[OK] proxy 就绪: $HC"; break; }
  sleep 3
  [ "$i" = 10 ] && { echo "[FATAL] proxy 30s 未就绪"; tail -20 log/proxy.log; exit 1; }
done

bash scripts/curl_pd.sh || echo "[WARN] 请求发送返回非零, 继续检查归档"
sleep 10   # 等最后一个 TERM 归档的后台线程 flush 完

echo "===== [5/7] 检查归档落盘 (expect: kv_P_1 / kv_D_1 / kv_P_2 / kv_D_2) ====="
for i in $(seq 1 12); do
  N=$(ls log/tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 4 ] && break
  echo "  ... 等待归档 flush ($N/4), 10s 后重试"
  sleep 10
done
ls -la log/tensors/ || true
N=$(ls log/tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
if [ "$N" -lt 4 ]; then
  echo "[WARN] 归档仅 $N/4 个 —— 检查 [KVT] 行: "
  grep -h "\[KVT\]" log/kvc_p_req*.log log/kvc_d_req*.log 2>/dev/null | tail -8
else
  echo "[OK] 4 个归档就位"
fi
# [KVT] 归档行为双侧日志留痕(轨迹文件在 curl_pd.sh 已提取, 这里补 grep 上下文)
grep -h "\[KVT\]" log/kvc_p_req*.log log/kvc_d_req*.log 2>/dev/null \
  > log/kvt_archive_lines.log || true
echo "  [KVT] 行数: $(wc -l < log/kvt_archive_lines.log 2>/dev/null || echo 0)"

echo "===== [6/7] 容器内初检 ====="
echo "-- v1 指纹裁决 (compare_fp.py) --"
python3 scripts/compare_fp.py 2>&1 | tail -12 || echo "[WARN] compare_fp 异常"
echo "-- 五级检查器 selftest --"
python3 scripts/check_kv_tensors.py --selftest 2>&1 | tail -5 || echo "[WARN] selftest 异常"
echo "-- .pt L0 快检(可读/形状) --"
python3 - <<'PYEOF'
import torch, glob, json
files = sorted(glob.glob("log/tensors/kv_*.pt"))
for f in files:
    b = torch.load(f, map_location="cpu")
    m = b["meta"]
    ok = (sum(m["cov"]) == m["w_tok"] and len(b["K"]) == m["layers"]
          and b["K"][0].shape == (m["w_tok"], m["kv_heads"], m["head_dim"]))
    print(f"  {f}: side={m['side']} seq={m['seq']} p_tok={m['p_tok']} "
          f"w_tok={m['w_tok']} blocks={m['block_table']} cov={m['cov']} "
          f"{'OK' if ok else 'STRUCT_FAIL'}")
PYEOF

echo "===== [7/7] 收尾: 杀服务 + 撤补丁 ====="
bash scripts/stop_pd.sh
bash patch/revert_offline_patches.sh || echo "[WARN] 撤补丁异常, 请手动检查"
pgrep -af "v[l]lm serve" || echo "[OK] 无 vllm 进程残留"
grep -q "\[KVC\]" /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py \
  && echo "[WARN] vllm-ascend 源码仍有 [KVC] 残留" \
  || echo "[OK] vllm-ascend 源码干净"

echo "===== [DONE] $(date '+%F %T') 双链产物: log/ (v1 轨迹+verdict) + log/tensors/ (v2 归档) ====="
echo "       回收: 容器内 bash scripts/pull_tensors.sh pack | 主机侧 bash scripts/pull_tensors.sh fetch"
