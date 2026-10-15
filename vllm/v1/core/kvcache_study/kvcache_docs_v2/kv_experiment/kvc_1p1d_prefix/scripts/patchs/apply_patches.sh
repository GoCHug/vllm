#!/bin/bash
# ==============================================================================
# apply_patches.sh —— 应用 01 号 [PCM] Prefix Cache 矩阵观察补丁（独立、可单独 apply）
#
# 目标文件: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py（唯一触碰文件）
# 独立性:   本区唯一补丁 —— mooncake_connector.py 不与 ../kvc(01~08)/../kvc_1p1d(01~08)
#           的任何目标重叠, 可在 pristine 源码上直接 apply, 也可与它们叠加(互不冲突)。
#
# 用法(容器内): VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
#      (run_all.sh 已自动导出容器路径; 单独在容器内跑时才需手动传)
# 行为: 防重复检测 -> dry-run 预检 -> patch -p1 -> [PCM]x7 计数验证 -> py_compile
# ==============================================================================
set -euo pipefail
PD="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/Users/wushanglun/Desktop/vllmgch/vllm-ascend}"
REL=vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
FILE="$VLLM_ASCEND_DIR/$REL"

[ -f "$FILE" ] || { echo "[ERROR] 目标文件不存在: $FILE (用 VLLM_ASCEND_DIR=... 指定)"; exit 1; }

if grep -q "\[PCM\]" "$FILE" 2>/dev/null; then
  echo "[ABORT] 源码已含 [PCM] 打印, 拒绝重复应用 —— 回退用 ./revert_patches.sh"
  exit 1
fi

if (cd "$VLLM_ASCEND_DIR" && patch -p1 --dry-run < "$PD/01_pcm_prefix_cache_matrix.patch" >/dev/null 2>&1); then
  (cd "$VLLM_ASCEND_DIR" && patch -p1 < "$PD/01_pcm_prefix_cache_matrix.patch" >/dev/null)
else
  echo "[ABORT] dry-run 失败 —— 源码已非 pristine 0.23.0 基线或被改动(人工核对 $FILE)"
  exit 1
fi

n=$(grep -c "\[PCM\]" "$FILE" || true)
if [ "$n" = "7" ]; then
  echo "[OK] [PCM] x7 注入完成: CFG / SCHED / ALLOC / PFINISH / XFER-entry / FULL-HIT / XFER-end"
else
  echo "[WARN] [PCM] 计数 ${n}(预期 7), 请人工核对"
fi
python3 -m py_compile "$FILE" && echo "[OK] py_compile 通过"
md5sum "$FILE" 2>/dev/null || md5 -q "$FILE"
echo "[DONE] 01 号 PCM 补丁已应用。"
