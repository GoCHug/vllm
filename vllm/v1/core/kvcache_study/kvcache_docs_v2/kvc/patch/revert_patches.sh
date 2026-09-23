#!/bin/bash
# ==============================================================================
# revert_patches.sh —— 一键撤销 9 个 [KVC] 调试打印补丁（还原为干净源码）
#
# 用法:
#   容器内默认目录:      ./revert_patches.sh
#   自定义仓库位置:      VLLM_DIR=/path/to/vllm VLLM_ASCEND_DIR=/path/to/vllm-ascend ./revert_patches.sh
#
# 行为:
#   Phase 0  状态检查 —— 源码中无 [KVC] 时提示已干净并退出
#   Phase 1  patch -R --dry-run 预检(反向打必须全部可干净执行)
#   Phase 2  patch -R -p1 逐个反转 9 个补丁
#   Phase 3  验证: 9 文件 [KVC] 全部归 0 + py_compile
#
# 说明:
#   - 采用 patch -R 反向应用, 不依赖 .orig 备份
# ==============================================================================
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
# VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
# VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
# 本地跑(releases/v0.23.0 基线 9/9 实测通过): 直接用环境变量, 或注释上面两行改用下面两行:
VLLM_DIR="${VLLM_DIR:-/Users/wushanglun/Desktop/vllmgch/vllm}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/Users/wushanglun/Desktop/vllmgch/vllm-ascend}"

VLLM_FILES=(
  vllm/v1/request.py
  vllm/v1/core/kv_cache_utils.py
  vllm/v1/core/block_pool.py
  vllm/v1/core/kv_cache_manager.py
  vllm/v1/core/kv_cache_coordinator.py
  vllm/v1/core/single_type_kv_cache_manager.py
  vllm/v1/engine/core.py
  vllm/v1/worker/gpu_model_runner.py
)
ASCEND_FILE="vllm_ascend/worker/model_runner_v1.py"

[ -d "$VLLM_DIR" ]        || { echo "[ERROR] vllm 仓库不存在: $VLLM_DIR (用 VLLM_DIR=... 指定)"; exit 1; }
[ -d "$VLLM_ASCEND_DIR" ] || { echo "[ERROR] vllm-ascend 仓库不存在: $VLLM_ASCEND_DIR (用 VLLM_ASCEND_DIR=... 指定)"; exit 1; }

kvc_count() {  # BSD grep -c 无匹配时输出 0 且 exit 1 —— 只压码值, 不再追加输出(避免 "0\n0")
  grep -c "\[KVC\]" "$1" 2>/dev/null || true
}

echo "== Phase 0: 状态检查 =="
if ! grep -q "\[KVC\]" "$VLLM_DIR/${VLLM_FILES[0]}" 2>/dev/null \
   && ! grep -q "\[KVC\]" "$VLLM_ASCEND_DIR/$ASCEND_FILE" 2>/dev/null; then
  echo "  源码中未发现 [KVC] 打印, 已是干净状态, 无需撤销。"
  exit 0
fi
cur=$(kvc_count "$VLLM_DIR/${VLLM_FILES[0]}")
cur2=$(kvc_count "$VLLM_ASCEND_DIR/$ASCEND_FILE")
echo "  检测到 [KVC] 打印(request.py:$cur / model_runner_v1.py:$cur2), 开始撤销。"

echo "== Phase 1: dry-run(-R) 预检 =="
fail=0
for f in "$PATCH_DIR"/0[1-8]_vllm_*.patch; do
  if (cd "$VLLM_DIR" && patch -R -p1 --dry-run < "$f" >/dev/null 2>&1); then
    echo "  ok: $(basename "$f")"
  else
    echo "  FAIL: $(basename "$f")"; fail=1
  fi
done
if (cd "$VLLM_ASCEND_DIR" && patch -R -p1 --dry-run < "$PATCH_DIR"/09_*.patch >/dev/null 2>&1); then
  echo "  ok: $(basename "$PATCH_DIR"/09_*.patch)"
else
  echo "  FAIL: $(basename "$PATCH_DIR"/09_*.patch)"; fail=1
fi
[ "$fail" = 0 ] || { echo "[ABORT] 反向 dry-run 未通过: 补丁已被修改或非本套补丁产物, 未做任何更改"; exit 1; }

echo "== Phase 2: 反向应用 =="
for f in "$PATCH_DIR"/0[1-8]_vllm_*.patch; do
  (cd "$VLLM_DIR" && patch -R -p1 < "$f" >/dev/null 2>&1) && echo "  reverted: $(basename "$f")"
done
(cd "$VLLM_ASCEND_DIR" && patch -R -p1 < "$PATCH_DIR"/09_*.patch >/dev/null 2>&1) && echo "  reverted: 09_vllm_ascend_*.patch"

echo "== Phase 3: 验证 =="
bad=0
for f in "${VLLM_FILES[@]}"; do
  n=$(kvc_count "$VLLM_DIR/$f")
  [ "$n" = 0 ] || { echo "  残留: $f ($n 行)"; bad=1; }
done
n=$(kvc_count "$VLLM_ASCEND_DIR/$ASCEND_FILE")
[ "$n" = 0 ] || { echo "  残留: $ASCEND_FILE ($n 行)"; bad=1; }
[ "$bad" = 0 ] && echo "  9 文件 [KVC] 全部归零 ✓" || { echo "[WARN] 有残留, 请人工检查"; exit 1; }

cd "$VLLM_DIR"
python3 -m py_compile "${VLLM_FILES[@]}" "$VLLM_ASCEND_DIR/$ASCEND_FILE" && echo "  py_compile OK (9 files)"
echo "[DONE] 9 个补丁已全部撤销, 源码还原干净。"