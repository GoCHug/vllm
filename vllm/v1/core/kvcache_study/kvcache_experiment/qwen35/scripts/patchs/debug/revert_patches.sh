#!/bin/bash
# 一键撤销当前目录下所有 *.patch (反向应用)
#
# 用法:
#   ./revert_patches.sh                     # patch 目录 = 脚本所在目录
#   ./revert_patches.sh /path/to/patches    # 指定 patch 目录
#   VLLM_DIR=... VLLM_ASCEND_DIR=... ./revert_patches.sh
#
# 每个 patch 自动判断打在哪个仓库: 谁反向 dry-run 能通过就撤谁那里的。
# 已被撤过(未应用)的补丁会 SKIP, 可重复执行。
set -uo pipefail

PATCH_DIR="$(cd "${1:-$(dirname "$0")}" && pwd)"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"

# 找出能反向干净撤销该 patch 的仓库
# 注意 -N: 已撤销过的补丁直接忽略(rc=1), 否则 patch 会交互式正向应用
pick_repo() {
  for d in "$VLLM_DIR" "$VLLM_ASCEND_DIR"; do
    [ -d "$d" ] || continue
    (cd "$d" && patch -R -N -p1 --dry-run < "$1" >/dev/null 2>&1) && { echo "$d"; return 0; }
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
    if (cd "$repo" && patch -R -N -p1 < "$p" >/dev/null); then
      echo "OK    $name <- $(basename "$repo")"
    else
      echo "FAIL  $name"; fail=1
    fi
  else
    echo "SKIP  $name (未应用, 无需撤销)"
  fi
done

[ "$fail" = 0 ] && echo "全部撤销完成" || echo "有补丁撤销失败, 请检查"
exit $fail