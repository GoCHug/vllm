#!/bin/bash
# ==============================================================================
# apply_patches.sh —— qwen35 hybrid 一键应用 8 个 [KVC] KVCache 调试补丁(v0.23.0 基线)
#
# qwen3.5-0.8B(hybrid: full-attn 层 + GDN/mamba 层)混合模型 KVCache 端到端观测:
#   01~07 与 kvc/ 工作区完全一致 —— vllm 管理侧打印(入队/哈希链/BlockPool/
#     Manager 四子步/Coordinator 逐层下放/SingleType/EngineCore 编排六步),
#     对混合模型同样生效(HybridKVCacheManager 继承关系下打点为基类方法);
#   08 为 hybrid 改造版(kvc v2.5 的混合模型演进) ——
#     · 层分组: self.attn_groups 展开(attn 组 / mamba 组), 组序与
#       NewRequestData.block_ids 下标一一对应;
#     · TERM 归档: attn 组每层每块 K/V 整块 + mamba(GDN) 组每层每 slot 全部
#       状态张量(conv+ssm)整块 cpu().clone() 位级快照 —— 两组共享同一张
#       物理 raw tensor(页级共用) 但块表独立, 按各组块表分别取证;
#     · bundle schema kvt5-hybrid-raw: {"K"(attn), "V"(attn), "M"(mamba
#       状态), "meta"(双组块表/shape/dtype)}; kvt4 兼容字段另置顶层;
#     · 归档布局 req{seq}_{rid尾8}/kv_pp{p}tp{t}.pt(一请求一子目录, 每 worker 一份);
#     · L1 区新增 hybrid 单张 raw 页打印 + mamba 状态视图(conv/ssm 三视图布局)打印。
#   env 开关: KVC_SAVE_KV=1 启用(默认关, 零侵入); KVC_SAVE_DIR 输出目录。
#
# 用法:
#   本地(默认路径):     ./apply_patches.sh
#   容器内(需指定):     VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
#
# 行为:
#   Phase 0  已应用检测(源码带 [KVC] 即中止)
#   Phase 1  dry-run 预检 8 补丁(任一失败则不落盘)
#   Phase 2  patch -p1 应用(01~07 -> vllm, 08 -> vllm-ascend)
#   Phase 3  验证: [KVC] 计数(178 行 = vllm 155 + vllm-ascend 23) + py_compile
#   Phase 4  [KVS] 归档开关提示(scripts/server/start.sh 会自动导出)
#
# 注意:
#   - vllm 0.23.0 + vllm-ascend 0.23.0 基线 8/8 干净命中(本地实测通过)
#   - 容器重启会丢可写层改动, 需重新执行本脚本
# ==============================================================================
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="${VLLM_DIR:-/Users/wushanglun/Desktop/vllmgch/vllm}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/Users/wushanglun/Desktop/vllmgch/vllm-ascend}"
# VLLM_DIR=/c/Users/89517/Desktop/github/vllm-npu/vllm VLLM_ASCEND_DIR=/c/Users/89517/Desktop/github/vllm-npu/vllm-ascend ./apply_patches.sh

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
EXPECT=(5 10 29 58 17 18 18)
ASCEND_EXPECT=23                 # 08 hybrid 版实测: KVS 归档 11(attn K/V + mamba M + kvt5-hybrid-raw) + L1 物理池 12(横幅2 + hybrid页2 + dense2 + attn reshape2 + mamba视图3 + 注释1)

[ -d "$VLLM_DIR" ]        || { echo "[ERROR] vllm 仓库不存在: $VLLM_DIR (用 VLLM_DIR=... 指定)"; exit 1; }
[ -d "$VLLM_ASCEND_DIR" ] || { echo "[ERROR] vllm-ascend 仓库不存在: $VLLM_ASCEND_DIR (用 VLLM_ASCEND_DIR=... 指定)"; exit 1; }

kvc_count() {  # BSD grep -c 无匹配时输出 0 且 exit 1 —— 只压码值
  grep -c "\[KVC\]" "$1" 2>/dev/null || true
}

echo "== Phase 0: 已应用检测 =="
if grep -q "\[KVC\]" "$VLLM_DIR/${VLLM_FILES[0]}" 2>/dev/null \
   || grep -q "\[KVC\]" "$VLLM_ASCEND_DIR/${ASCEND_FILES[0]}" 2>/dev/null; then
  echo "[ABORT] 检测到源码已带 [KVC] 打印, 拒绝重复应用 —— 要移除请用兄弟脚本 revert_patches.sh"
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
  echo "  ok: $(basename "$PATCH_DIR"/08_*.patch) (hybrid 版: TERM attn K/V + mamba conv/ssm 归档, kvt5-hybrid-raw)"
else
  echo "  FAIL: $(basename "$PATCH_DIR"/08_*.patch)"; fail=1
fi
[ "$fail" = 0 ] || { echo "[ABORT] dry-run 未全部通过, 未做任何修改 (0.23.0 基线应 8/8 通过)"; exit 1; }

echo "== Phase 2: 应用 =="
for f in "$PATCH_DIR"/0[1-7]_vllm_*.patch; do
  (cd "$VLLM_DIR" && patch -p1 < "$f" >/dev/null 2>&1) && echo "  applied: $(basename "$f")"
done
(cd "$VLLM_ASCEND_DIR" && patch -p1 < "$PATCH_DIR"/08_*.patch >/dev/null 2>&1) && echo "  applied: 08_vllm_ascend_*.patch (hybrid 归档版, kvt5-hybrid-raw)"

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
echo "  [KVC] 总匹配行: $total (预期 178 行 = vllm 155 + vllm-ascend 23)"
[ "$bad" = 0 ] || { echo "[WARN] 部分文件计数与预期不符, 请人工核对"; }

cd "$VLLM_DIR"
python3 -m py_compile "${VLLM_FILES[@]}" "$VLLM_ASCEND_DIR/${ASCEND_FILES[0]}" && echo "  py_compile OK (8 files)"
kvs_line=$(grep -c "def _kvc_kv_save" "$VLLM_ASCEND_DIR/${ASCEND_FILES[0]}" || true)
[ "$kvs_line" = 1 ] && echo "  hybrid 特征方法 _kvc_kv_save 存在 ✓" || echo "  [WARN] 未找到 _kvc_kv_save 方法"
kvt5_line=$(grep -c "kvt5-hybrid-raw" "$VLLM_ASCEND_DIR/${ASCEND_FILES[0]}" || true)
[ "$kvt5_line" -ge 1 ] && echo "  bundle schema kvt5-hybrid-raw 标记 ✓" || echo "  [WARN] 未找到 kvt5-hybrid-raw 标记"
kvp_line=$(grep -c "def _kvc_kv_dump" "$VLLM_ASCEND_DIR/${ASCEND_FILES[0]}" || true)
[ "$kvp_line" = 0 ] && echo "  纯 attention 版打印方法 _kvc_kv_dump 不存在(llama 版专属, 正确) ✓" || echo "  [WARN] 仍存在 _kvc_kv_dump(旧版残留)"

echo "== Phase 4: 归档开关提示 =="
echo "  qwen35 hybrid 实验的 TERM 归档由 env 控制:"
echo "    KVC_SAVE_KV=1                    # 开启( scripts/server/start.sh 已自动导出 )"
echo "    KVC_SAVE_DIR=<目录>              # 输出目录(默认 log/tensors; start.sh 用相对路径 tensors, 基准=实验根)"
echo "[DONE] 8 个补丁已应用并验证(08 为 hybrid 归档版, kvt5-hybrid-raw: attn K/V + mamba conv/ssm)。"
