#!/bin/bash
# ==============================================================================
# run_pp2tp2_all.sh —— kvc_offline pp2tp2 混布 block 原样归档实验一键执行(容器内)
#
# 拓扑: P=TP2(卡0,1/8100/producer) + D=TP2(卡2,3/8200/consumer) + proxy(8000)
# 七阶段:
#   [1/7] patch:  apply_kvc_offline_patches.sh(kvc 01~08 → 09 指纹 → 11 block 原样)
#   [2/7] 起 P TP2 卡0,1 → 就绪(双 rank: 物理侧分配横幅 ×2)
#   [3/7] 起 D TP2 卡2,3 → 就绪(同上)
#   [4/7] 起 proxy + 发 req_p/req_r 双请求(curl_pp2.sh: 轨迹按 dev 拆双 rank)
#   [5/7] 检查归档落盘: 8 文件(kv_{P0,P1,D0,D1}_{1,2}_*.pt)
#   [6/7] 容器内初检: compare_fp.py(v1,TP2 仅参考) + check_kv_blocks --selftest
#         + L0 快检(meta 可读/双 rank 齐全)
#   [7/7] 收尾: stop_pd.sh + revert_kvc_offline_patches.sh(源码归零)
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc_offline
#   setsid nohup bash scripts/run_pp2tp2_all.sh > log/run_pp2tp2_screen.log 2>&1 < /dev/null &
#   tail -f log/run_pp2tp2_screen.log
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
mkdir -p log log/tensors
export VLLM_DIR=/vllm-workspace/vllm
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
export KVC_DUMP_BLOCKS=1             # 11 号补丁开关: TERM block 原样归档(随子进程继承)
export KVC_DUMP_DIR=log/tensors      # 相对 cwd=本工作区根

echo "===== [run_pp2tp2_all] $(date '+%F %T') kvc_offline pp2tp2 block 原样归档实验开始 ====="
[ -f log/req_p.json ] || cp ../kvc/log/req_p.json log/
[ -f log/req_r.json ] || cp ../kvc/log/req_r.json log/
[ -f log/req_p.json ] && [ -f log/req_r.json ] || { echo "[FATAL] req_p/req_r 缺失"; exit 1; }
rm -f log/tensors/*.pt log/tensors/manifest*.json 2>/dev/null

echo "===== [1/7] 打 kvc_offline 补丁 (kvc 01~08 + 09 指纹 + 11 block 原样) ====="
bash patch/apply_kvc_offline_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/7] 启动 P TP2 (卡0,1/8100/producer) ====="
bash scripts/start_pp2_p.sh
for i in $(seq 1 35); do
  sleep 10
  if grep -q "Application startup complete" log/p_llama.log 2>/dev/null; then
    echo "[OK] P 侧就绪 (等待 $((i*10))s)"
    NR=$(grep -c "物理侧 KV Cache 分配开始" log/p_llama.log 2>/dev/null || true)
    [ "$NR" = 2 ] && echo "  双 rank 物理池分配横幅: ${NR} ✓" || echo "  [WARN] 物理池横幅 ${NR}/2 (TP rank 进程数待核)"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" log/p_llama.log 2>/dev/null; then
    echo "[FATAL] P 侧启动失败, 见 log/p_llama.log 尾部:"; tail -30 log/p_llama.log; exit 1
  fi
  [ "$i" = 35 ] && { echo "[FATAL] P 侧 350s 未就绪"; tail -30 log/p_llama.log; exit 1; }
done

echo "===== [3/7] 启动 D TP2 (卡2,3/8200/consumer) ====="
bash scripts/start_pp2_d.sh
for i in $(seq 1 35); do
  sleep 10
  if grep -q "Application startup complete" log/d_llama.log 2>/dev/null; then
    echo "[OK] D 侧就绪 (等待 $((i*10))s)"
    NR=$(grep -c "物理侧 KV Cache 分配开始" log/d_llama.log 2>/dev/null || true)
    [ "$NR" = 2 ] && echo "  双 rank 物理池分配横幅: ${NR} ✓" || echo "  [WARN] 物理池横幅 ${NR}/2"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" log/d_llama.log 2>/dev/null; then
    echo "[FATAL] D 侧启动失败, 见 log/d_llama.log 尾部:"; tail -30 log/d_llama.log; exit 1
  fi
  [ "$i" = 35 ] && { echo "[FATAL] D 侧 350s 未就绪"; tail -30 log/d_llama.log; exit 1; }
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

bash scripts/curl_pp2.sh || echo "[WARN] 请求发送返回非零, 继续检查归档"
sleep 10   # 等最后一个 TERM 的后台落盘线程 flush 完(每 rank 独立 4 MiB/块级别)

echo "===== [5/7] 检查归档落盘 (expect: kv_{P0,P1,D0,D1}_{1,2}_*.pt 共 8 个) ====="
for i in $(seq 1 12); do
  N=$(ls log/tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 8 ] && break
  echo "  ... 等待归档 flush ($N/8), 10s 后重试"
  sleep 10
done
ls -la log/tensors/ 2>/dev/null || true
N=$(ls log/tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
if [ "$N" -lt 8 ]; then
  echo "[WARN] 归档仅 $N/8 个 —— [KVB] 行为检查: "
  grep -h "\[KVB\]" log/kvc_p_req*.log log/kvc_d_req*.log log/kvc_p[01]_req*.log \
       log/kvc_d[01]_req*.log 2>/dev/null | tail -10 || true
else
  echo "[OK] 8 个归档就位"
fi
grep -h "\[KVB\]" log/p_llama.log log/d_llama.log 2>/dev/null \
  | sed -E 's/^\((EngineCore|Worker_[A-Za-z0-9_]+) pid=[0-9]+\) //' \
  > log/kvb_archive_lines.log || true
echo "  [KVB] 行数: $(wc -l < log/kvb_archive_lines.log 2>/dev/null || echo 0)"

echo "===== [6/7] 容器内初检 ====="
echo "-- v1 指纹裁决 (compare_fp.py; TP2 下日志双 rank 交织, 逐层覆盖仅末 rank, 仅作参考) --"
python3 scripts/compare_fp.py 2>&1 | tail -6 || echo "[WARN] compare_fp 异常(TP2 参考)"
echo "-- block 检查器 selftest --"
python3 scripts/check_kv_blocks.py --selftest 2>&1 | grep -v "LD_PRELOAD\|FunctionLoader" | tail -4 || echo "[WARN] selftest 异常"
echo "-- .pt L0 快检(可读/双 rank/shape) --"
python3 - <<'PYEOF'
import torch, glob
files = sorted(glob.glob("log/tensors/kv_*.pt"))
for f in files:
    b = torch.load(f, map_location="cpu")
    m = b["meta"]
    bs, kh = int(m["block_size"]), int(m["kv_heads"])
    ok = (sum(m["cov"]) == m["w_tok"] and len(b["K"]) == m["layers"]
          and all(bd[blk].shape == (bs, kh, int(m["head_dim"]))
                  for bd in b["K"] for blk in bd))
    print(f"  {f}: {m['side']}r{m['rank']} s{m['seq']} p_tok={m['p_tok']} "
          f"w_tok={m['w_tok']} kv_heads={kh} blocks={m['block_table']} "
          f"cov={m['cov']} {'OK' if ok else 'STRUCT_FAIL'}")
PYEOF

echo "===== [7/7] 收尾: 杀服务 + 撤补丁 ====="
bash scripts/stop_pd.sh
bash patch/revert_kvc_offline_patches.sh || echo "[WARN] 撤补丁异常, 请手动检查"
pgrep -af "v[l]lm serve" || echo "[OK] 无 vllm 进程残留"
grep -q "\[KVC\]" /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py \
  && echo "[WARN] vllm-ascend 源码仍有 [KVC] 残留" \
  || echo "[OK] vllm-ascend 源码干净"

echo "===== [DONE] $(date '+%F %T') pp2tp2 双链产物: log/ (v1 轨迹) + log/tensors/ (8 block 原样归档) ====="
echo "       正式检查: python3 scripts/check_kv_blocks.py --dir log/tensors --logs log --out log/block_report"
echo "       回收: 容器内 bash scripts/pull_tensors.sh pack | 主机侧 bash scripts/pull_tensors.sh fetch"
