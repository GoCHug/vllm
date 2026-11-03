#!/bin/bash
# 一键应用当前目录下所有 *.patch
#
# 用法:
#   ./apply_patches.sh                      # patch 目录 = 脚本所在目录
#   ./apply_patches.sh /path/to/patches     # 指定 patch 目录
#   VLLM_DIR=... VLLM_ASCEND_DIR=... ./apply_patches.sh
#
# 每个 patch 自动判断目标仓库: 谁 dry-run 能干净应用就打到谁那里。
set -uo pipefail

PATCH_DIR="$(cd "${1:-$(dirname "$0")}" && pwd)"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"

# 找出能干净应用该 patch 的仓库
# 注意 -N: 已应用过的补丁直接忽略(rc=1), 否则 patch 会交互式反向应用
pick_repo() {
  for d in "$VLLM_DIR" "$VLLM_ASCEND_DIR"; do
    [ -d "$d" ] || continue
    (cd "$d" && patch -N -p1 --dry-run < "$1" >/dev/null 2>&1) && { echo "$d"; return 0; }
  done
  return 1
}

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
[ ${#patches[@]} -gt 0 ] || { echo "没有 *.patch: $PATCH_DIR"; exit 0; }

fail=0
for p in "${patches[@]}"; do
  name="$(basename "$p")"
  if repo="$(pick_repo "$p")"; then
    if (cd "$repo" && patch -N -p1 < "$p" >/dev/null); then
      echo "OK    $name -> $(basename "$repo")"
    else
      echo "FAIL  $name"; fail=1
    fi
  else
    echo "SKIP  $name (已应用, 或两个仓库都对不上)"; fail=1
  fi
done

[ "$fail" = 0 ] && echo "全部应用完成" || echo "有补丁未应用, 请检查"
exit $fail