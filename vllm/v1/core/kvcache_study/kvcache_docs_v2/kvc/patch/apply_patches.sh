#!/bin/bash
# ==============================================================================
# apply_patches.sh —— 一键应用 8 个 [KVC] KVCache 调试打印补丁（vllm 0.23.0 基线）
#
# 用法:
#   容器内执行:          VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
#   本地(默认路径已配):  ./apply_patches.sh        # 或用 VLLM_DIR=... VLLM_ASCEND_DIR=... 自定义仓库位置
#
# 行为:
#   Phase 1  dry-run 预检 —— 8 个 patch 全部通过才继续, 任一失败则中止(不落盘)
#   Phase 2  patch -p1 应用 (01~07 -> vllm, 08 -> vllm-ascend)
#   Phase 3  验证: 每文件 [KVC] 计数 + 总数(预期 168 行/95 打印调用点, 含 CFG ①②③ 子步横幅、S1 子步横幅先行覆盖外层容量探问、调度提交包裹横幅、KVP 仅 TERM/LATE 且逐层按块展开) + py_compile
#
# 注意:
#   - vllm 0.23.0 + vllm-ascend 0.23.0 基线 8/8 干净命中(容器实测通过)
#   - 容器重启会丢可写层改动, 需重新执行本脚本(见 patch/README.md §6)
# ==============================================================================
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
# VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
# VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
# 本地跑(releases/v0.23.0 基线 8/8 实测通过): 直接用环境变量, 或注释上面两行改用下面两行:
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
)
ASCEND_FILES=(vllm_ascend/worker/model_runner_v1.py)
EXPECT=(5 10 29 56 17 18 18)
ASCEND_EXPECT=15                  # 08 补丁预期 [KVC] 行(含 KVP 结束期逐层按块打印: 头横幅/概览/逐块行/尾横幅)

[ -d "$VLLM_DIR" ]        || { echo "[ERROR] vllm 仓库不存在: $VLLM_DIR (用 VLLM_DIR=... 指定)"; exit 1; }
[ -d "$VLLM_ASCEND_DIR" ] || { echo "[ERROR] vllm-ascend 仓库不存在: $VLLM_ASCEND_DIR (用 VLLM_ASCEND_DIR=... 指定)"; exit 1; }

kvc_count() {  # BSD grep -c 无匹配时输出 0 且 exit 1 —— 只压码值, 不再追加输出(避免 "0\n0")
  grep -c "\[KVC\]" "$1" 2>/dev/null || true
}

echo "== Phase 0: 已应用检测 =="
if grep -q "\[KVC\]" "$VLLM_DIR/${VLLM_FILES[0]}" 2>/dev/null \
   || grep -q "\[KVC\]" "$VLLM_ASCEND_DIR/${ASCEND_FILES[0]}" 2>/dev/null; then
  echo "[ABORT] 检测到源码已带 [KVC] 打印, 拒绝重复应用 —— 要移除请用: $0 的兄弟脚本 revert_patches.sh"
  exit 1
fi
echo "  ok: 源码为干净状态"

echo "== Phase 1: dry-run 预检 =="
fail=0
for f in "$PATCH_DIR"/0[1-7]_vllm_*.patch; do
  if (cd "$VLLM_DIR" && patch -p1 --dry-run < "$f" >/dev/null 2>&1); then
    echo "  ok: $(basename "$f")"
  else
    echo "  FAIL: $(basename "$f")"; fail=1
  fi
done
if (cd "$VLLM_ASCEND_DIR" && patch -p1 --dry-run < "$PATCH_DIR"/08_*.patch >/dev/null 2>&1); then
  echo "  ok: $(basename "$PATCH_DIR"/08_*.patch)"
else
  echo "  FAIL: $(basename "$PATCH_DIR"/08_*.patch)"; fail=1
fi
[ "$fail" = 0 ] || { echo "[ABORT] dry-run 未全部通过, 未做任何修改 (0.23.0 基线应 8/8 通过)"; exit 1; }

echo "== Phase 2: 应用 =="
for f in "$PATCH_DIR"/0[1-7]_vllm_*.patch; do
  (cd "$VLLM_DIR" && patch -p1 < "$f" >/dev/null 2>&1) && echo "  applied: $(basename "$f")"
done
(cd "$VLLM_ASCEND_DIR" && patch -p1 < "$PATCH_DIR"/08_*.patch >/dev/null 2>&1) && echo "  applied: 08_vllm_ascend_*.patch"

echo "== Phase 3: 验证 =="
bad=0; total=0
for i in "${!VLLM_FILES[@]}"; do
  n=$(kvc_count "$VLLM_DIR/${VLLM_FILES[$i]}")
  flag=ok; [ "$n" = "${EXPECT[$i]}" ] || { flag="MISMATCH(expect ${EXPECT[$i]})"; bad=1; }
  printf "  %-60s %s 行 %s\n" "${VLLM_FILES[$i]}" "$n" "[$flag]"
  total=$((total + n))
done
for f in "${ASCEND_FILES[@]}"; do
  n=$(kvc_count "$VLLM_ASCEND_DIR/$f")
  flag=ok; [ "$n" = "$ASCEND_EXPECT" ] || { flag="MISMATCH(expect $ASCEND_EXPECT)"; bad=1; }
  total=$((total + n))
  printf "  %-60s %s 行 %s\n" "$f" "$n" "[$flag]"
done
echo "  [KVC] 总匹配行: $total (预期 168 行)"
[ "$bad" = 0 ] || { echo "[WARN] 部分文件计数与预期不符, 请人工核对"; }

cd "$VLLM_DIR"
python3 -m py_compile "${VLLM_FILES[@]}" "$VLLM_ASCEND_DIR/${ASCEND_FILES[0]}" && echo "  py_compile OK (8 files)"
echo "[DONE] 8 个补丁已应用并验证。打印位置索引: $PATCH_DIR/kvc_patch_locations.txt"