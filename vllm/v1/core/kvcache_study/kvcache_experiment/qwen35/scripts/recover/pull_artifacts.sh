#!/usr/bin/env bash
# ==============================================================================
# pull_artifacts.sh —— qwen35 hybrid 产物回收（仅主机侧执行, 唯一模式 fetch）
#
# 前置（容器内手动打包, 实验收尾 revert_patches 之前执行）:
#   cd /a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35
#   tar -czf qwen35_bundle.tar.gz tensors/ qwen35_normal.log qwen35_abnormal.log 2>/dev/null
#   md5sum qwen35_bundle.tar.gz   # 记下 md5 备查
# 主机侧（实验根执行）: bash scripts/recover/pull_artifacts.sh fetch
#   -> scp 经 SSH 隧道拉回 tar -> 打印 tar md5（与容器侧 md5sum 输出人工对照）
#   -> 解包覆盖本地 tensors/ + 服务日志（qwen35_normal.log / qwen35_abnormal.log）
#   -> \r 规范化（tqdm 孤立 \r） -> .pt 数量/字节清单核对
#
# 隧道: itask ssh-tunnel gggtest4 --port 5557（端口可用 PORT 环境变量覆盖）
#   （gggtest4 为 gch 身份容器 —— 若本机 itask 仍为他人身份, 先 itask auth login 切回 gch）
# 完整性: tar md5 对照 + .pt 明细清单即可; .pt 逐字节正确性由拉回后
#   python3 scripts/analysis/inspect_kv_tensors.py --dir tensors（加载全部张量）承载。
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
[ "${1:-}" = "fetch" ] || {
  echo "用法: 容器内 tar 打包后, 主机侧 bash scripts/recover/pull_artifacts.sh fetch"; exit 1; }
cd "$HERE"
PORT="${PORT:-5557}"
POD_WS="/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35"

echo "== [fetch] 经隧道 $PORT 拉回容器产物 =="
scp -P "$PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=/dev/null \
    "root@localhost:$POD_WS/qwen35_bundle.tar.gz" /tmp/qwen35_bundle.tar.gz

echo "== [fetch] tar md5（对照容器侧 md5sum 输出） =="
md5 -q /tmp/qwen35_bundle.tar.gz 2>/dev/null || md5sum /tmp/qwen35_bundle.tar.gz

echo "== [fetch] 解包到本地（tensors/ + 服务日志） =="
tar -xzf /tmp/qwen35_bundle.tar.gz

echo "== [fetch] \r 规范化（qwen35_normal/abnormal.log 中 tqdm 孤立 \r -> \n） =="
python3 - <<'PYEOF'
from pathlib import Path
for name in ("qwen35_normal.log", "qwen35_abnormal.log"):
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
        print(f"  [INFO] {f} 不存在（该启动模式本轮未运行, 属正常）")
PYEOF

echo "== [fetch] 归档清单（tensors/req*/kv_*.pt, 每次 curl.sh 归档 1 份） =="
python3 - <<'PYEOF'
import glob
from pathlib import Path
files = sorted(glob.glob("tensors/req*/kv_*.pt"))
for s in sorted({Path(f).stat().st_size for f in files}):
    k = sum(1 for f in files if Path(f).stat().st_size == s)
    print(f"  {s/1048576:.1f} MiB x {k}")
n = len(files)
print(f"  [OK] 归档 {n} 份就位（每请求 1 份: K=attn / V=attn / M=mamba conv+ssm / meta）"
      if n else "  [WARN] 无归档 —— 检查容器侧 KVC_SAVE_KV=1 与 [KVS] 横幅")
raise SystemExit(0)
PYEOF

echo "[DONE] 产物已回收 tensors/（+服务日志）"
echo "       报告: python3 scripts/analysis/inspect_kv_tensors.py --dir tensors -> logs/analysis/inspect_kv_tensors.out"
echo "       前缀: python3 scripts/analysis/inspect_prefix.py    --dir tensors -> logs/analysis/inspect_prefix.out"
echo "       单件: python3 tensors/view_pt.py（归档 .pt 快速检视: K/V + mamba M）"
