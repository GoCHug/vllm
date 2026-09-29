#!/bin/bash
# ==============================================================================
# revert_pd_patches.sh —— 撤销 PD 补丁套装（09 先撤, 再调 ../kvc/patch 撤 01~08）
#
# 用法(容器内):
#   VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_pd_patches.sh
#
# 行为:
#   Step 1  检测 09 是否在(指纹打印存在) -> patch -R -p1 反向撤 09
#   Step 2  调 ../kvc/patch/revert_patches.sh 撤 01~08(幂等: 已干净则跳过)
#   Step 3  验证: model_runner_v1.py [KVC] 归零 + py_compile
# ==============================================================================
set -euo pipefail

PD_PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
KVC_PATCH_DIR="${KVC_PATCH_DIR:-$PD_PATCH_DIR/../../kvc/patch}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"

ASCEND_FILE="$VLLM_ASCEND_DIR/vllm_ascend/worker/model_runner_v1.py"

echo "== Step 1: 检测并撤销 09 =="
if grep -q "指纹" "$ASCEND_FILE" 2>/dev/null; then
  if (cd "$VLLM_ASCEND_DIR" && patch -R -p1 --dry-run < "$PD_PATCH_DIR/09_pd_kv_fingerprint.patch" >/dev/null 2>&1); then
    (cd "$VLLM_ASCEND_DIR" && patch -R -p1 < "$PD_PATCH_DIR/09_pd_kv_fingerprint.patch" >/dev/null 2>&1) \
      && echo "  reverted: 09_pd_kv_fingerprint.patch"
  else
    echo "[ABORT] 09 反向 dry-run 失败, 源码疑似被修改, 未做任何更改"
    exit 1
  fi
else
  echo "  09 未检测到指纹打印, 跳过(可能已撤)"
fi

echo "== Step 2: 撤销 kvc 01~08 (调 ../kvc/patch/revert_patches.sh) =="
KVC_PATCH_DIR_NORM="$(cd "$KVC_PATCH_DIR" && pwd)"
bash "$KVC_PATCH_DIR_NORM/revert_patches.sh"

echo "== Step 3: 终验 =="
n=$(grep -c "\[KVC\]" "$ASCEND_FILE" 2>/dev/null || true)
[ "$n" = 0 ] && echo "  model_runner_v1.py [KVC] 归零 ✓" || { echo "[ERROR] 仍有 $n 行残留"; exit 1; }
python3 -m py_compile "$ASCEND_FILE" && echo "  py_compile OK"
echo "[DONE] PD 补丁已全部撤销, 源码还原干净。"
