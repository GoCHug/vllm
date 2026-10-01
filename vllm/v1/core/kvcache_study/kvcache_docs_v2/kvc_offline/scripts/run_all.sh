#!/bin/bash
# ==============================================================================
# run_all.sh —— kvc_offline v3 单机 PP2×TP2 block 原样归档实验一键执行(容器内)
#
# 拓扑: 单实例 vllm serve --tensor-parallel-size 2 --pipeline-parallel-size 2
#       （与 ../kvc/ 相同形态, 占满 4 卡, 无 PD 分离/无 proxy, 直发 :8000）
# 六阶段:
#   [1/6] patch:  apply_kvc_offline_patches.sh（kvc 01~08 → 09 指纹 → 11 block 原样）
#   [2/6] 起服务（TP2+PP2 单实例）→ 就绪（四 worker 物理池横幅）
#   [3/6] 发 P/R 双请求（curl_pr.sh 直发 :8000 + 三段轨迹提取 + [KVB] 留痕）
#   [4/6] 检查归档落盘: 8 文件（kv_S{00,01,10,11}_{1,2}_*.pt = 4 worker × 2 请求）
#   [5/6] 容器内初检: check_kv_blocks --selftest + L0 快检（meta 可读/四路齐/层段正确）
#   [6/6] 收尾: stop.sh + revert_kvc_offline_patches.sh（源码归零）
#
# 用法(容器内):
#   cd /a3_inference/itask/workdir/wsl02075301/kvc_offline
#   setsid nohup bash scripts/run_all.sh > log/run_all_screen.log 2>&1 < /dev/null &
#   tail -f log/run_all_screen.log
# ==============================================================================
cd "$(dirname "$0")/.." || exit 1
mkdir -p log log/tensors
export VLLM_DIR=/vllm-workspace/vllm
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
export KVC_DUMP_BLOCKS=1             # 11 号补丁开关: TERM block 原样归档(随子进程继承)
export KVC_DUMP_DIR=log/tensors      # 相对 cwd=本工作区根

echo "===== [run_all] $(date '+%F %T') kvc_offline v3 单机 PP2×TP2 block 原样归档实验开始 ====="
[ -f log/req_p.json ] || cp ../kvc/log/req_p.json log/
[ -f log/req_r.json ] || cp ../kvc/log/req_r.json log/
[ -f log/req_p.json ] && [ -f log/req_r.json ] || { echo "[FATAL] req_p/req_r 缺失"; exit 1; }
rm -f log/tensors/*.pt log/tensors/manifest*.json 2>/dev/null

echo "===== [1/6] 打 kvc_offline 补丁 (kvc 01~08 + 09 指纹 + 11 block 原样) ====="
bash patch/apply_kvc_offline_patches.sh || { echo "[FATAL] 补丁应用失败"; exit 1; }

echo "===== [2/6] 启动服务（单机 TP2+PP2 占 4 卡）====="
bash scripts/start.sh
for i in $(seq 1 30); do
  sleep 10
  if grep -q "Application startup complete" log/llama-3-8b.log 2>/dev/null; then
    echo "[OK] 服务就绪 (等待 $((i*10))s)"
    NR=$(grep -c "物理侧 KV Cache 分配开始" log/llama-3-8b.log 2>/dev/null || true)
    [ "$NR" = 4 ] && echo "  四 worker 物理池分配横幅: ${NR} ✓" \
                  || echo "  [WARN] 物理池横幅 ${NR}/4 (worker 进程数待核; PP2×TP2 应为 4)"
    break
  fi
  if grep -qE "Traceback|EngineCore failed" log/llama-3-8b.log 2>/dev/null; then
    echo "[FATAL] 启动失败, 见 log/llama-3-8b.log 尾部:"; tail -30 log/llama-3-8b.log; exit 1
  fi
  [ "$i" = 30 ] && { echo "[FATAL] 300s 未就绪"; tail -30 log/llama-3-8b.log; exit 1; }
done

echo "===== [3/6] 发送 P/R 双请求（直发 :8000）====="
bash scripts/curl_pr.sh || echo "[WARN] 请求发送返回非零, 继续检查归档"
sleep 10   # 等最后一个 TERM 的后台落盘线程 flush 完(每 worker 独立)

echo "===== [4/6] 检查归档落盘 (expect: kv_S{00,01,10,11}_{1,2}_*.pt 共 8 个) ====="
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
  grep "\[KVB\]" log/llama-3-8b.log 2>/dev/null | tail -10 || true
else
  echo "[OK] 8 个归档就位"
fi

echo "===== [5/6] 容器内初检 ====="
echo "-- block 检查器 selftest --"
python3 scripts/check_kv_blocks.py --selftest 2>&1 | grep -v "LD_PRELOAD\|FunctionLoader" | tail -4 || echo "[WARN] selftest 异常"
echo "-- .pt L0 快检(可读/四路/层段) --"
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
    lids = m["layer_ids"]
    print(f"  {f}: pp{m['pp']} tp{m['tp']} s{m['seq']} layers={m['layers']}"
          f"[{lids[0]}~{lids[-1]}] p_tok={m['p_tok']} w_tok={m['w_tok']} "
          f"kv_heads={kh} blocks={m['block_table']} cov={m['cov']} "
          f"{'OK' if ok else 'STRUCT_FAIL'}")
PYEOF

echo "===== [6/6] 收尾: 杀服务 + 撤补丁 ====="
bash scripts/stop.sh
bash patch/revert_kvc_offline_patches.sh || echo "[WARN] 撤补丁异常, 请手动检查"
pgrep -af "v[l]lm serve" || echo "[OK] 无 vllm 进程残留"
grep -q "\[KVC\]" /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py \
  && echo "[WARN] vllm-ascend 源码仍有 [KVC] 残留" \
  || echo "[OK] vllm-ascend 源码干净"

echo "===== [DONE] $(date '+%F %T') v3 双链产物: log/（轨迹+归档） ====="
echo "       正式检查: python3 scripts/check_kv_blocks.py --dir log/tensors --logs log --out log/block_report"
echo "       回收: 容器内 bash scripts/pull_tensors.sh pack | 主机侧 bash scripts/pull_tensors.sh fetch"
