#!/usr/bin/env bash
# ==============================================================================
# pull_artifacts.sh —— kvc_1p1d 产物回收（仅在主机侧执行, 唯一模式 fetch）
#
# 容器侧无需调用本脚本: run_all.sh [8/8] 已自动把 logs/ + tensors/ 打成
#   kvc_1p1d_bundle.tar.gz 并打印 tar md5 备查。
# 主机侧(工作区根执行): bash scripts/recover/pull_artifacts.sh fetch
#   -> scp 经 SSH 隧道拉回 tar -> 打印 tar md5(与容器侧 [8/8] 输出人工对照)
#   -> 解包覆盖本地 logs/{server,patchs,curl,analysis} + tensors/{P,D}
#   -> \r 规范化(p/d_llama.log 中 tqdm 孤立 \r) -> 4 .pt 数量/字节核对
#
# 隧道: itask ssh-tunnel gggtest --port 5557 (端口可用 PORT 环境变量覆盖)
# 完整性: tar md5 对照 + 4 .pt 就位核对即可; .pt 逐字节正确性由拉回后
#   五个 analysis 检查器(加载全部张量)承载。
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
[ "${1:-}" = "fetch" ] || {
  echo "用法: bash scripts/recover/pull_artifacts.sh fetch   (容器侧打包已并入 run_all [8/8], 无需手动)"; exit 1; }
cd "$HERE"
PORT="${PORT:-5557}"
POD_WS="/a3_inference/itask/workdir/wsl02075301/kvc_1p1d"

echo "== [fetch] 经隧道 $PORT 拉回容器产物 =="
scp -P "$PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    "root@localhost:$POD_WS/kvc_1p1d_bundle.tar.gz" /tmp/kvc_1p1d_bundle.tar.gz

echo "== [fetch] tar md5 (对照容器侧 run_all [8/8] 输出) =="
md5 -q /tmp/kvc_1p1d_bundle.tar.gz 2>/dev/null || md5sum /tmp/kvc_1p1d_bundle.tar.gz

echo "== [fetch] 解包到本地 (logs/{server,patchs,curl,analysis} + tensors/{P,D}) =="
tar -xzf /tmp/kvc_1p1d_bundle.tar.gz

echo "== [fetch] \r 规范化 (logs/server/{p,d}_llama.log 中 tqdm 孤立 \r -> \n) =="
python3 - <<'PYEOF'
from pathlib import Path
for name in ("logs/server/p_llama.log", "logs/server/d_llama.log"):
    f = Path(name)
    if f.exists():
        raw = f.read_bytes()
        norm = raw.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if norm != raw:
            f.write_bytes(norm)
            print(f"  {f}: 规范化 {raw.count(b'\\r')} 个 \\r")
        else:
            print(f"  {f}: 无需规范化")
    else:
        print(f"  [WARN] {f} 不存在")
PYEOF

echo "== [fetch] 归档核对 (expect 4 .pt: P x2 + D x2) =="
python3 - <<'PYEOF'
import glob
from pathlib import Path
files = sorted(glob.glob("tensors/P/req*/kv_*.pt")) + sorted(glob.glob("tensors/D/req*/kv_*.pt"))
n = len(files)
for s in sorted({Path(f).stat().st_size for f in files}):
    k = sum(1 for f in files if Path(f).stat().st_size == s)
    print(f"  {s/1048576:.1f} MiB x {k}")
print(f"  [OK] 归档 {n}/4 就位 (P + D)" if n == 4 else f"  [WARN] 归档 {n}/4 (期望 4)")
raise SystemExit(0 if n == 4 else 1)
PYEOF

echo "[DONE] 产物已回收 logs/{server,patchs,curl,analysis} + tensors/{P,D}"
echo "       P 查看器: python3 scripts/analysis/inspect_kv_tensors_p.py --dir tensors/P -> logs/analysis/inspect_kv_tensors_p.out"
echo "       D 查看器: python3 scripts/analysis/inspect_kv_tensors_d.py --dir tensors/D -> logs/analysis/inspect_kv_tensors_d.out"
echo "       P 前缀:   python3 scripts/analysis/inspect_prefix_p.py     --dir tensors/P -> logs/analysis/inspect_prefix_p.out"
echo "       D 前缀:   python3 scripts/analysis/inspect_prefix_d.py     --dir tensors/D -> logs/analysis/inspect_prefix_d.out"
echo "       传输检查: python3 scripts/analysis/inspect_p2d.py          --dir tensors   -> logs/analysis/inspect_p2d.out"
