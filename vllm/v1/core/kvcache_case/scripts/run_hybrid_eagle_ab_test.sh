#!/usr/bin/env bash
# Case 09 | issue #32802（hybrid + EAGLE → prefix cache 命中率归零）确定性 A/B 复现
#
# 原理: 本仓库 git 历史自带 bug 现场:
#   bug = a01ef3fa51^ (= 7320ca3942, 2026-02-01, 含 #31707 引入的收敛循环, 未修复)
#   fix = a01ef3fa51    (PR #33524, 2026-02-01, is_simple_hybrid 单轮退出修复)
# 修复 PR 自带 eagle 回归测试(测试文件本身随修复合入), 把它同步到 bug 态,
# 同一条测试在两态跑: bug FAIL / fix PASS ⇒ 命中坍塌到 0 的确定性复现(REPRO)。
#
# 依赖(纯 CPU, 不需要 GPU/NPU/模型权重):
#   python3 + torch>=2.6(实测 2.7.1; 2.5.1 会缺 torch._inductor.custom_graph_pass)
#   pytest + 修复提交的 requirements/common.txt 依赖集, 一条命令:
#     cd <REPO> && git show a01ef3fa51:requirements/common.txt > /tmp/common.txt \
#       && pip install -r /tmp/common.txt && pip install pytest
#
# 用法: bash run_hybrid_eagle_ab_test.sh [REPO_PATH]
#   REPO_PATH 默认 = 脚本所在目录上溯 5 级(kvcache_case/scripts → vllm 仓库根)
#   worktree 产物目录默认 /tmp/kvc_case08(可用 WORK=... 覆盖), 跑完保留供人工复查

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
WORK="${WORK:-/tmp/kvc_case08}"
FIX_COMMIT="a01ef3fa51"   # PR #33524
TESTS="tests/v1/core/test_prefix_caching.py"

[ -d "${REPO}/.git" ] || { echo "ERR: 不是 git 仓库: ${REPO}"; exit 2; }
git -C "${REPO}" cat-file -e "${FIX_COMMIT}^{commit}" 2>/dev/null \
  || { echo "ERR: 仓库缺少提交 ${FIX_COMMIT}(PR #33524), 无法 A/B"; exit 2; }

BUG_COMMIT="$(git -C "${REPO}" rev-parse "${FIX_COMMIT}^")"
echo "repo      = ${REPO}"
echo "bug       = ${BUG_COMMIT::12} ($(git -C "${REPO}" log -1 --format=%as "${BUG_COMMIT}"))"
echo "fix       = ${FIX_COMMIT}   ($(git -C "${REPO}" log -1 --format=%as "${FIX_COMMIT}"))"
echo "workdir   = ${WORK}"
mkdir -p "${WORK}"

setup_worktree() {  # $1=dir $2=commit
  mkdir -p "$(dirname "$1")"
  if [ -e "$1/.git" ]; then
    # 先丢弃上次同步测试文件带来的改动, 否则切提交会被本地改动挡住
    git -C "$1" checkout -q -- . || true
    git -C "$1" checkout --detach -q "$2" --
  else
    git -C "${REPO}" worktree add --detach "$1" "$2" >/dev/null
  fi
}
setup_worktree "${WORK}/bug" "${BUG_COMMIT}"
setup_worktree "${WORK}/fix" "${FIX_COMMIT}"

# 回归测试文件随修复 PR 合入, bug 态本来没有 ⇒ 同步过去(只影响 worktree, 不动主检出)
git -C "${WORK}/bug" checkout "${FIX_COMMIT}" -- "${TESTS}"
git -C "${WORK}/fix" checkout "${FIX_COMMIT}" -- "${TESTS}"

run_eagle_tests() {  # $1=dir; 输出 pytest 最后一行摘要(passed/failed 计数)
  ( cd "$1" && PYTHONPATH="$1" python3 -m pytest \
      "${TESTS}::test_prefill_hybrid_model_eagle" \
      "${TESTS}::test_prefill_hybrid_model_combinations_eagle" \
      --noconftest -q 2>&1 || true ) | tail -1
}

echo
echo "--- bug 态(未修复) ---"
bug_out="$(run_eagle_tests "${WORK}/bug")"
echo "    ${bug_out}"
echo "--- fix 态(PR #33524) ---"
fix_out="$(run_eagle_tests "${WORK}/fix")"
echo "    ${fix_out}"

count() { grep -oE "([1-9][0-9]*) ${1}" <<< "$2" | head -1 | cut -d' ' -f1 || true; }
bug_failed=$(count failed "${bug_out}"); bug_failed=${bug_failed:-0}
fix_passed=$(count passed "${fix_out}"); fix_passed=${fix_passed:-0}
fix_failed=$(count failed "${fix_out}"); fix_failed=${fix_failed:-0}

echo
echo "==================== 判定 ===================="
echo "bug 态: ${bug_failed:-0} failed | fix 态: ${fix_passed} passed, ${fix_failed} failed"
if [ "${bug_failed:-0}" -gt 0 ] && [ "${fix_passed}" -gt 0 ] && [ "${fix_failed}" -eq 0 ]; then
  echo "REPRO: bug 态 eagle 回归测试失败(同前缀请求 get_computed_blocks 返回 ([],[],[])),"
  echo "       fix 态通过 —— issue #32802(命中率坍塌为 0)在本仓库确定性复现"
  echo "失败断言细节: cd ${WORK}/bug && PYTHONPATH=${WORK}/bug python3 -m pytest \\"
  echo "    ${TESTS}::test_prefill_hybrid_model_eagle --noconftest -q"
elif [ "${bug_failed:-0}" -eq 0 ] && [ "${fix_passed}" -gt 0 ] && [ "${fix_failed}" -eq 0 ]; then
  echo "PASS: 两态均通过 —— 未复现(检查 REPO 是否指向含这段历史的 vllm 检出)"
else
  echo "WARN: 两态都失败或输出异常 —— 多为环境问题: 先运行"
  echo "     cd ${REPO} && git show ${FIX_COMMIT}:requirements/common.txt > /tmp/common.txt && pip install -r /tmp/common.txt"
fi
