#!/bin/bash
# ==============================================================================
# apply_offline_patches.sh —— kvc_pd_offline 张量归档补丁套装
#                            （kvc 01~08 → kvc_pd 09 指纹 → 本区 10 张量归档）
#
# 10: 10_pd_kv_tensor_dump.patch —— 在 09 之上叠加 TERM 张量归档
#     (_kvc_kv_dump 收尾处调 _kvc_tensor_dump: cpu().clone() 位级快照
#      + 后台线程 torch.save, env 开关 KVC_DUMP_TENSORS=1 / KVC_DUMP_DIR)
#
# 用法(容器内):
#   VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
#     ./apply_offline_patches.sh
#
# 行为:
#   Step 1  调 ../../kvc/patch/apply_patches.sh        应用 01~08
#   Step 2  调 ../../kvc_pd/patch/apply_pd_patches.sh  应用 09   (内含上面的 01~08 幂等检测)
#   Step 3  本区 10 dry-run -> patch -p1 应用 -> 验证(_kvc_tensor_dump 存在 + py_compile)
#
# 注意: 10 hunk 行号基于"已应用 09"的文件, 必须 01~08 → 09 → 10 顺序。
#       revert 顺序相反: 先 10 (本目录 revert_offline_patches.sh 自动处理)。
# ==============================================================================
set -euo pipefail

OFF_PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"

[ -f "$OFF_PATCH_DIR/10_pd_kv_tensor_dump.patch" ] || { echo "[ERROR] 缺少 10_pd_kv_tensor_dump.patch"; exit 1; }

echo "== Step 1+2: 应用 kvc 01~08 + 09 指纹 (调 kvc_pd/patch/apply_pd_patches.sh) =="
PD_APPLY="$OFF_PATCH_DIR/../../kvc_pd/patch/apply_pd_patches.sh"
[ -f "$PD_APPLY" ] || { echo "[ERROR] 找不到 $PD_APPLY"; exit 1; }
bash "$PD_APPLY"

echo "== Step 3: 应用 10 (张量归档, 叠在 09 之上) =="
ASCEND_FILE="$VLLM_ASCEND_DIR/vllm_ascend/worker/model_runner_v1.py"
if grep -q "_kvc_tensor_dump" "$ASCEND_FILE" 2>/dev/null; then
  echo "[ABORT] 10 张量归档已存在, 拒绝重复应用"
  exit 1
fi
if (cd "$VLLM_ASCEND_DIR" && patch -p1 --dry-run < "$OFF_PATCH_DIR/10_pd_kv_tensor_dump.patch" >/dev/null 2>&1); then
  (cd "$VLLM_ASCEND_DIR" && patch -p1 < "$OFF_PATCH_DIR/10_pd_kv_tensor_dump.patch" >/dev/null 2>&1) \
    && echo "  applied: 10_pd_kv_tensor_dump.patch"
else
  echo "[ABORT] 10 dry-run 失败 —— 请确认 01~09 已正确应用后再试"
  exit 1
fi

echo "== Step 4: 验证 =="
n=$(grep -c "\[KVC\]\[KVT\]" "$ASCEND_FILE" 2>/dev/null || true)
[ "$n" -ge 2 ] && echo "  model_runner_v1.py [KVT] 归档相关行: $n 处" \
              || { echo "[ERROR] [KVT] 行数异常($n)"; exit 1; }
python3 -m py_compile "$ASCEND_FILE" && echo "  py_compile OK"
echo "[DONE] offline 补丁套装应用完成: kvc 01~08 + 09 指纹 + 10 张量归档。"
echo "       运行时开关: KVC_DUMP_TENSORS=1 (默认关); 输出目录 KVC_DUMP_DIR=log/tensors"
