#!/bin/bash
# ==============================================================================
# apply_pd_patches.sh —— PD 正确性实验补丁套装（kvc 01-08 + PD 专用 09）
#
# 01~08 与单机实验共用 ../kvc/patch/（[KVC] 调试打印 8 补丁）
# 09   09_pd_kv_fingerprint.patch：在 vllm-ascend 08 之上叠加 KVP 内容指纹
#      （_kvc_kv_dump 每层新增 sha256 原始位指纹, 对 P/D 两实例逐层逐位比对）
#
# 用法(容器内):
#   VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_pd_patches.sh
#
# 行为:
#   Step 1  调 ../kvc/patch/apply_patches.sh 应用 01~08(dry-run 预检+计数验证)
#   Step 2  09 dry-run 预检 -> patch -p1 应用(依赖 08 已应用, 上下文含 08 增量)
#   Step 3  验证: model_runner_v1.py 出现 [KVP] 指纹调用 + py_compile
#
# 注意:
#   - 09 hunk 行号基于"已应用 08"的文件, 必须先 01~08 后 09
#   - revert 顺序相反: 先 09 (本目录 revert_pd_patches.sh 会自动处理)
# ==============================================================================
set -euo pipefail

PD_PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
KVC_PATCH_DIR="${KVC_PATCH_DIR:-$PD_PATCH_DIR/../../kvc/patch}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"

[ -d "$KVC_PATCH_DIR" ] || { echo "[ERROR] kvc 补丁目录不存在: $KVC_PATCH_DIR"; exit 1; }
[ -f "$PD_PATCH_DIR/09_pd_kv_fingerprint.patch" ] || { echo "[ERROR] 缺少 09_pd_kv_fingerprint.patch"; exit 1; }

echo "== Step 1: 应用 kvc 01~08 (调 ../kvc/patch/apply_patches.sh) =="
KVC_PATCH_DIR_NORM="$(cd "$KVC_PATCH_DIR" && pwd)"
bash "$KVC_PATCH_DIR_NORM/apply_patches.sh"

echo "== Step 2: 应用 09 (PD 内容指纹, 叠在 08 之上) =="
ASCEND_FILE="$VLLM_ASCEND_DIR/vllm_ascend/worker/model_runner_v1.py"
if grep -q "\[KVP\] .*指纹" "$ASCEND_FILE" 2>/dev/null; then
  echo "[ABORT] 09 指纹打印已存在, 拒绝重复应用"
  exit 1
fi
if (cd "$VLLM_ASCEND_DIR" && patch -p1 --dry-run < "$PD_PATCH_DIR/09_pd_kv_fingerprint.patch" >/dev/null 2>&1); then
  (cd "$VLLM_ASCEND_DIR" && patch -p1 < "$PD_PATCH_DIR/09_pd_kv_fingerprint.patch" >/dev/null 2>&1) \
    && echo "  applied: 09_pd_kv_fingerprint.patch"
else
  echo "[ABORT] 09 dry-run 失败 —— 请确认 01~08(尤其 08)已正确应用后再试"
  exit 1
fi

echo "== Step 3: 验证 =="
n=$(grep -c "指纹" "$ASCEND_FILE" 2>/dev/null || true)
[ "$n" -ge 2 ] && echo "  model_runner_v1.py 指纹相关行: $n 处 (注释 2 + 打印 1)" \
              || { echo "[WARN] 指纹行数异常($n), 请人工核对"; }
python3 -m py_compile "$ASCEND_FILE" && echo "  py_compile OK"
echo "[DONE] PD 补丁套装应用完成: kvc 01~08 + 09。"
