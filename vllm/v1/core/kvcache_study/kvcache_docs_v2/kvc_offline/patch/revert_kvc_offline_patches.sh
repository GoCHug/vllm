#!/bin/bash
# ==============================================================================
# revert_kvc_offline_patches.sh —— 撤销 kvc_offline 补丁套装
#                                  （11 先撤 → 调 kvc_pd 撤 09 → 调 kvc 撤 01~08）
#
# 用法(容器内):
#   VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
#     ./revert_kvc_offline_patches.sh
# ==============================================================================
set -euo pipefail

OFF_PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
ASCEND_FILE="$VLLM_ASCEND_DIR/vllm_ascend/worker/model_runner_v1.py"

echo "== Step 1: 检测并撤销 11 =="
if grep -q "_kvc_block_dump" "$ASCEND_FILE" 2>/dev/null; then
  if (cd "$VLLM_ASCEND_DIR" && patch -R -p1 --dry-run < "$OFF_PATCH_DIR/11_pp2tp2_block_dump.patch" >/dev/null 2>&1); then
    (cd "$VLLM_ASCEND_DIR" && patch -R -p1 < "$OFF_PATCH_DIR/11_pp2tp2_block_dump.patch" >/dev/null 2>&1) \
      && echo "  reverted: 11_pp2tp2_block_dump.patch"
  else
    echo "[ABORT] 11 反向 dry-run 失败, 源码疑似被修改, 未做任何更改"
    exit 1
  fi
else
  echo "  11 未检测到归档代码, 跳过(可能已撤)"
fi

echo "== Step 2: 撤销 09 + 01~08 (调 kvc_pd/patch/revert_pd_patches.sh) =="
PD_REVERT="$OFF_PATCH_DIR/../../kvc_pd/patch/revert_pd_patches.sh"
[ -f "$PD_REVERT" ] || { echo "[ERROR] 找不到 $PD_REVERT"; exit 1; }
bash "$PD_REVERT"

echo "== Step 3: 终验 =="
n=$(grep -c "\[KVC\]" "$ASCEND_FILE" 2>/dev/null || true)
[ "$n" = 0 ] && echo "  model_runner_v1.py [KVC] 归零 OK" || { echo "[ERROR] 仍有 $n 行残留"; exit 1; }
python3 -m py_compile "$ASCEND_FILE" && echo "  py_compile OK"
echo "[DONE] kvc_offline 补丁已全部撤销, 源码还原干净。"
