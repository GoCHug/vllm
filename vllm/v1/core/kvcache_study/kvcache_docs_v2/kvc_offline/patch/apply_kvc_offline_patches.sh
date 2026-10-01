#!/bin/bash
# ==============================================================================
# apply_kvc_offline_patches.sh —— kvc_offline(v3 单机 PP2×TP2) block 原样归档补丁套装
#                                （kvc 01~08 → kvc_pd 09 指纹 → 本区 11 block 归档）
#
# 11: 11_pp2tp2_block_dump.patch —— 在 09 之上叠加 TERM block 原样张量归档
#     (_kvc_kv_dump 收尾处调 _kvc_block_dump: 每块整存 kt[blk] 不 gather,
#      env 开关 KVC_DUMP_BLOCKS=1 / KVC_DUMP_DIR; 单机 PP2×TP2 下每 worker
#      (pp×tp 4 路)各自归档, 文件 kv_S{pp}{tp}_{seq}_{rid尾8}.pt)
#
# 用法(容器内):
#   VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
#     ./apply_kvc_offline_patches.sh
#
# 注意: 11 与 10(kvc_pd_offline) 挂同一位点但互为本轮独立使用, 两者不同轮共存;
#       hunk 行号基于"已应用 09"的文件, 必须 01~08 → 09 → 11 顺序, 撤销反向。
# ==============================================================================
set -euo pipefail

OFF_PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"

[ -f "$OFF_PATCH_DIR/11_pp2tp2_block_dump.patch" ] || { echo "[ERROR] 缺少 11_pp2tp2_block_dump.patch"; exit 1; }

echo "== Step 1+2: 应用 kvc 01~08 + 09 指纹 (调 kvc_pd/patch/apply_pd_patches.sh) =="
PD_APPLY="$OFF_PATCH_DIR/../../kvc_pd/patch/apply_pd_patches.sh"
[ -f "$PD_APPLY" ] || { echo "[ERROR] 找不到 $PD_APPLY"; exit 1; }
bash "$PD_APPLY"

echo "== Step 3: 应用 11 (block 原样归档, 叠在 09 之上) =="
ASCEND_FILE="$VLLM_ASCEND_DIR/vllm_ascend/worker/model_runner_v1.py"
if grep -q "_kvc_block_dump" "$ASCEND_FILE" 2>/dev/null; then
  echo "[ABORT] 11 block 归档已存在, 拒绝重复应用"
  exit 1
fi
if (cd "$VLLM_ASCEND_DIR" && patch -p1 --dry-run < "$OFF_PATCH_DIR/11_pp2tp2_block_dump.patch" >/dev/null 2>&1); then
  (cd "$VLLM_ASCEND_DIR" && patch -p1 < "$OFF_PATCH_DIR/11_pp2tp2_block_dump.patch" >/dev/null 2>&1) \
    && echo "  applied: 11_pp2tp2_block_dump.patch"
else
  echo "[ABORT] 11 dry-run 失败 —— 请确认 01~09 已正确应用后再试"
  exit 1
fi

echo "== Step 4: 验证 =="
n=$(grep -c "\[KVC\]\[KVB\]" "$ASCEND_FILE" 2>/dev/null || true)
[ "$n" -ge 2 ] && echo "  model_runner_v1.py [KVB] 归档相关行: $n 处" \
              || { echo "[ERROR] [KVB] 行数异常($n)"; exit 1; }
python3 -m py_compile "$ASCEND_FILE" && echo "  py_compile OK"
echo "[DONE] kvc_offline 补丁套装应用完成: kvc 01~08 + 09 指纹 + 11 block 原样归档。"
echo "       运行时开关: KVC_DUMP_BLOCKS=1 (默认关); 输出目录 KVC_DUMP_DIR=log/tensors"
