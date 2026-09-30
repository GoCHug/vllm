#!/bin/bash
# ==============================================================================
# apply_pcm_patch.sh —— 应用 10 号 [PCM] Prefix Cache 矩阵观察补丁（独立、可单独 apply）
#
# 目标文件: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py（唯一触碰文件）
# 独立性:   不依赖 kvc 01-08 / pd 09 —— mooncake_connector.py 不在任何既有补丁的目标里,
#           可在 pristine 源码上直接 apply(实验推荐形态), 也可与 01-09 叠加(互不冲突)。
#
# 用法(容器内): VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_pcm_patch.sh
# 行为: 防重复检测 -> dry-run 预检 -> patch -p1 -> [PCM]x7 计数验证 -> py_compile
# ==============================================================================
set -euo pipefail
PD="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
REL=vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
FILE="$VLLM_ASCEND_DIR/$REL"

[ -f "$FILE" ] || { echo "[ERROR] 目标文件不存在: $FILE"; exit 1; }

if grep -q "\[PCM\]" "$FILE" 2>/dev/null; then
  echo "[ABORT] 源码已含 [PCM] 打印, 拒绝重复应用 —— 回退用 ./revert_pcm_patch.sh"
  exit 1
fi

if (cd "$VLLM_ASCEND_DIR" && patch -p1 --dry-run < "$PD/10_pcm_prefix_cache_matrix.patch" >/dev/null 2>&1); then
  (cd "$VLLM_ASCEND_DIR" && patch -p1 < "$PD/10_pcm_prefix_cache_matrix.patch" >/dev/null)
else
  echo "[ABORT] dry-run 失败 —— 源码已非 pristine 0.23.0 基线(见 md5 对照)或被改动"
  exit 1
fi

n=$(grep -c "\[PCM\]" "$FILE" || true)
if [ "$n" = "7" ]; then
  echo "[OK] [PCM] x7 注入完成: CFG / SCHED / ALLOC / PFINISH / XFER-entry / FULL-HIT / XFER-end"
else
  echo "[WARN] [PCM] 计数 ${n}(预期 7), 请人工核对"
fi
python3 -m py_compile "$FILE" && echo "[OK] py_compile 通过"
md5sum "$FILE"
echo "[DONE] 10 号 PCM 补丁已应用。"
