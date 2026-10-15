#!/bin/bash
# ==============================================================================
# revert_patches.sh —— 回退 01 号 [PCM] 补丁（与 apply_patches.sh 互逆）
#
# 用法(容器内): VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_patches.sh
#      (run_all.sh 收尾提示单独跑时需手动传; 本地默认 macOS 路径)
# 行为: 确认 [PCM] 存在 -> patch -R -p1 -> 零 [PCM] 验证 -> py_compile -> 容器源码还原基线
# 基线 md5: 00baf169f48fb167b9f6dfe650ac0ea5 (pristine 0.23.0 mooncake_connector.py)
# ==============================================================================
set -euo pipefail
PD="$(cd "$(dirname "$0")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/Users/wushanglun/Desktop/vllmgch/vllm-ascend}"
REL=vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
FILE="$VLLM_ASCEND_DIR/$REL"

[ -f "$FILE" ] || { echo "[ERROR] 目标文件不存在: $FILE (用 VLLM_ASCEND_DIR=... 指定)"; exit 1; }

if ! grep -q "\[PCM\]" "$FILE" 2>/dev/null; then
  echo "[SKIP] 源码无 [PCM] 打印, 无需回退(已是基线)"
  exit 0
fi

(cd "$VLLM_ASCEND_DIR" && patch -R -p1 < "$PD/01_pcm_prefix_cache_matrix.patch" >/dev/null)

n=$(grep -c "\[PCM\]" "$FILE" || true)
[ "$n" = "0" ] && echo "[OK] [PCM] 归零, 源码已还原" || { echo "[ERROR] 残留 [PCM] x${n}!"; exit 1; }
python3 -m py_compile "$FILE" && echo "[OK] py_compile 通过"
md5sum "$FILE" 2>/dev/null || md5 -q "$FILE"
echo "[DONE] 01 号 PCM 补丁已回退(基线 md5 应为 00baf169f48fb167b9f6dfe650ac0ea5)。"
