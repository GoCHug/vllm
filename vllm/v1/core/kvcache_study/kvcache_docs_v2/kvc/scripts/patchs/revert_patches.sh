#!/bin/bash
# ==============================================================================
# revert_patches.sh —— 撤销 [KVC] 调试补丁(逐文件幂等版, 撤的是本工作区当前补丁套)
#
# 用法:
#   本地(默认路径):     ./revert_patches.sh
#   容器内(需指定):     VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_patches.sh
#   (容器内 run_all.sh 已自动导出容器路径; 单独在容器内跑时才需手动传)
#
# 行为:
#   Phase 0  状态检查 —— 8 个目标文件全部无 [KVC] 时提示已干净并直接退出
#   Phase 1  逐补丁反向预检 —— 目标文件已无 [KVC] 的补丁自动 skip(幂等, 支持
#            "部分已还原"的混合态); 有 [KVC] 者必须 patch -R --dry-run 通过
#   Phase 2  反向应用(仅实际有 [KVC] 的目标; skip 者不动)
#   Phase 3  验证: 8 文件 [KVC] 全部归 0 + py_compile
#
# 版本不匹配的处理:
#   若 08 目标文件残留旧版研究态(如 v1 打印版 / 中间实验版, 特征 def _kvc_kv_dump
#   等), 而本套 08 为 v2.5 归档版 —— 反向 dry-run 必然 FAIL; 脚本会安全中止(未做
#   任何更改)并给出 git 恢复指引: 先备份该文件的研究态 diff, 再 checkout 还原
#   基线, 然后重跑本脚本即可对其余补丁正常撤销。
# ==============================================================================
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
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
ASCEND_FILE="vllm_ascend/worker/model_runner_v1.py"

[ -d "$VLLM_DIR" ]        || { echo "[ERROR] vllm 仓库不存在: $VLLM_DIR (用 VLLM_DIR=... 指定)"; exit 1; }
[ -d "$VLLM_ASCEND_DIR" ] || { echo "[ERROR] vllm-ascend 仓库不存在: $VLLM_ASCEND_DIR (用 VLLM_ASCEND_DIR=... 指定)"; exit 1; }

kvc_count() {  # BSD grep -c 无匹配时输出 0 且 exit 1 —— 只压码值
  grep -c "\[KVC\]" "$1" 2>/dev/null || true
}

# ----------------------------------------------------------------------------
echo "== Phase 0: 状态检查 =="
dirty=0
for f in "${VLLM_FILES[@]}"; do
  n=$(kvc_count "$VLLM_DIR/$f")
  [ "$n" = 0 ] || { echo "  [KVC] $f: $n 行"; dirty=1; }
done
n=$(kvc_count "$VLLM_ASCEND_DIR/$ASCEND_FILE")
[ "$n" = 0 ] || { echo "  [KVC] $ASCEND_FILE: $n 行"; dirty=1; }
if [ "$dirty" = 0 ]; then
  echo "  8 个目标文件均无 [KVC], 已是干净状态, 无需撤销。"
  exit 0
fi

# ----------------------------------------------------------------------------
preflight_one() {  # $1=repo $2=patch_file $3=target_rel $4=tag(=08 时启用版本诊断)
  local repo="$1" pf="$2" tgt="$3" tag="$4"
  local base
  base="$(basename "$pf")"
  if ! grep -q "\[KVC\]" "$repo/$tgt" 2>/dev/null; then
    echo "  skip(目标已干净): $base"
    P_SKIP=$((P_SKIP+1))
    return 0
  fi
  if (cd "$repo" && patch -R -p1 --dry-run < "$pf" >/dev/null 2>&1); then
    echo "  ok: $base"
    P_OK=$((P_OK+1))
  else
    echo "  FAIL: $base"
    if [ "$tag" = "08" ]; then
      if grep -q "def _kvc_kv_dump" "$repo/$tgt" 2>/dev/null \
         || ! grep -q "def _kvc_kv_save" "$repo/$tgt" 2>/dev/null; then
        echo "    诊断: $ASCEND_FILE 为旧版研究态(v1 打印版/中间实验版), 而本套 08 为 v2.5 归档版(19 行), 无法反向。"
        echo "    恢复指引(先备份研究态 diff 再还原基线, 然后重跑本脚本):"
        echo "      cd $repo && git diff -- $tgt > /tmp/kvc_08_researchstate_backup.diff"
        echo "      cd $repo && git checkout -- $tgt"
      else
        echo "    诊断: 目标含 [KVC] 但反向 dry-run 不通过 —— 源码或补丁被改动过, 请人工核对。"
      fi
    fi
    P_FAIL=$((P_FAIL+1))
  fi
}

echo "== Phase 1: 逐补丁反向预检(干净目标自动跳过) =="
P_OK=0; P_SKIP=0; P_FAIL=0
for i in 1 2 3 4 5 6 7; do
  pf=$(ls "$PATCH_DIR"/0${i}_vllm_*.patch 2>/dev/null | head -1 || true)
  if [ -z "$pf" ]; then
    echo "  FAIL: 0${i}_vllm_*.patch 不存在"
    P_FAIL=$((P_FAIL+1))
    continue
  fi
  preflight_one "$VLLM_DIR" "$pf" "${VLLM_FILES[$((i-1))]}" ""
done
pf8=$(ls "$PATCH_DIR"/08_*.patch 2>/dev/null | head -1 || true)
if [ -n "$pf8" ]; then
  preflight_one "$VLLM_ASCEND_DIR" "$pf8" "$ASCEND_FILE" "08"
else
  echo "  FAIL: 08_*.patch 不存在"
  P_FAIL=$((P_FAIL+1))
fi
if [ "$P_FAIL" != 0 ]; then
  echo "[ABORT] 反向预检 $P_FAIL 项未通过(见上), 未做任何更改。"
  exit 1
fi

# ----------------------------------------------------------------------------
revert_one() {  # $1=repo $2=patch_file $3=target_rel $4=suffix
  local repo="$1" pf="$2" tgt="$3" suffix="$4"
  if ! grep -q "\[KVC\]" "$repo/$tgt" 2>/dev/null; then
    echo "  skip(目标已干净): $(basename "$pf")"
    return 0
  fi
  if (cd "$repo" && patch -R -p1 < "$pf" >/dev/null 2>&1); then
    echo "  reverted: $(basename "$pf")$suffix"
  else
    echo "  FAIL: $(basename "$pf")"
    R_FAIL=$((R_FAIL+1))
  fi
}

echo "== Phase 2: 反向应用(预检 ok $P_OK 个 / skip $P_SKIP 个) =="
R_FAIL=0
for i in 1 2 3 4 5 6 7; do
  pf=$(ls "$PATCH_DIR"/0${i}_vllm_*.patch 2>/dev/null | head -1 || true)
  if [ -n "$pf" ]; then
    revert_one "$VLLM_DIR" "$pf" "${VLLM_FILES[$((i-1))]}" ""
  fi
done
pf8=$(ls "$PATCH_DIR"/08_*.patch 2>/dev/null | head -1 || true)
if [ -n "$pf8" ]; then
  revert_one "$VLLM_ASCEND_DIR" "$pf8" "$ASCEND_FILE" " (v2.5 块-行映射归档版)"
fi
if [ "$R_FAIL" != 0 ]; then
  echo "[ERROR] Phase 2 有 $R_FAIL 项失败, 请人工检查。"
  exit 1
fi

# ----------------------------------------------------------------------------
echo "== Phase 3: 验证 =="
bad=0
for f in "${VLLM_FILES[@]}"; do
  n=$(kvc_count "$VLLM_DIR/$f")
  [ "$n" = 0 ] || { echo "  残留: $f ($n 行)"; bad=1; }
done
n=$(kvc_count "$VLLM_ASCEND_DIR/$ASCEND_FILE")
[ "$n" = 0 ] || { echo "  残留: $ASCEND_FILE ($n 行)"; bad=1; }
if [ "$bad" != 0 ]; then
  echo "[WARN] 有残留, 请人工检查"
  exit 1
fi
echo "  8 文件 [KVC] 全部归零 ✓"

cd "$VLLM_DIR"
python3 -m py_compile "${VLLM_FILES[@]}" "$VLLM_ASCEND_DIR/$ASCEND_FILE" && echo "  py_compile OK (8 files)"
kvd=$(grep -c "def _kvc_kv_save\|def _kvc_rel_snapshot\|def _kvc_kv_dump" "$VLLM_ASCEND_DIR/$ASCEND_FILE" || true)
[ "$kvd" = 0 ] && echo "  kvc 方法已全部移除, 源码还原干净 ✓" || echo "  [WARN] 仍存在 kvc 方法($kvd 处)"
echo "[DONE] 撤销完成(reverted $P_OK / skipped $P_SKIP): 8 文件 [KVC] 归零, 源码回到无补丁状态。"
