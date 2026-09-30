#!/bin/bash
# ==============================================================================
# revert_pcm_patch.sh —— 回退 10 号 [PCM] 补丁（与 apply_pcm_patch.sh 互逆）
#
# 用法(容器内): VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_pcm_patch.sh
# 行为: 确认 [PCM] 存在 -> patch -R -p1 -> 零 [PCM] 验证 -> py_compile -> 容器源码还原基线
# ==============================================================================
set -euo pipefail
PD="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
REL=vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
FILE="$VLLM_ASCEND_DIR/$REL"

[ -f "$FILE" ] || { echo "[ERROR] 目标文件不存在: $FILE"; exit 1; }

if ! grep -q "\[PCM\]" "$FILE" 2>/dev/null; then
  echo "[SKIP] 源码无 [PCM] 打印, 无需回退(已是基线)"
  exit 0
fi

(cd "$VLLM_ASCEND_DIR" && patch -R -p1 < "$PD/10_pcm_prefix_cache_matrix.patch" >/dev/null)

n=$(grep -c "\[PCM\]" "$FILE" || true)
[ "$n" = "0" ] && echo "[OK] [PCM] 归零, 源码已还原" || { echo "[ERROR] 残留 [PCM] x${n}!"; exit 1; }
python3 -m py_compile "$FILE" && echo "[OK] py_compile 通过"
md5sum "$FILE"
echo "[DONE] 10 号 PCM 补丁已回退(基线 md5 应为 00baf169f48fb167b9f6dfe650ac0ea5)。"
