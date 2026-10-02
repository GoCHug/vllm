#!/usr/bin/env bash
# ==============================================================================
# pull_artifacts.sh —— kvc 产物回收（容器 pack → 主机 fetch）
#
# 容器内: bash scripts/pull_artifacts.sh pack
#   -> 等 8 个 .pt 归档就位 -> 写 tensors/manifest.json(md5+meta 摘要, 由
#      inspect_kv_tensors.py 逐个加载) -> tar 打包 log/ + tensors/
# 主机侧(工作区根执行): bash scripts/pull_artifacts.sh fetch
#   -> scp 经 SSH 隧道拉回 -> 解包覆盖本地 log/ + tensors/ -> \r 规范化
#      -> manifest md5 校验
#
# 隧道: itask ssh-tunnel gggtest --port 5557 --user wsl02075301 (fetch 依赖,
#       端口可用 PORT 环境变量覆盖)
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-pack}"
POD_PORT="${PORT:-5557}"
POD_WS="/a3_inference/itask/workdir/wsl02075301/kvc"

case "$MODE" in
pack)
  cd "$HERE"
  mkdir -p log tensors
  echo "== [pack] 等待归档就位 (expect 8 .pt: 4 worker × P/R) =="
  for i in $(seq 1 12); do
    N=$(ls tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
    [ "$N" -ge 8 ] && break
    echo "  ... $N/8, 10s 后重试"; sleep 10
  done
  N=$(ls tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 1 ] || { echo "[ERROR] 无归档产物"; exit 1; }
  [ "$N" = 8 ] || echo "[WARN] 归档数量 $N != 8, 继续打包"

  echo "== [pack] 写 tensors/manifest.json (md5 + meta 摘要) =="
  python3 - <<'PYEOF'
import hashlib, json, glob, torch
from pathlib import Path
mani = {"files": []}
for f in sorted(glob.glob("tensors/kv_*.pt")):
    h = hashlib.md5(Path(f).read_bytes()).hexdigest()
    m = torch.load(f, map_location="cpu")["meta"]
    mani["files"].append({
        "file": Path(f).name, "md5": h, "bytes": Path(f).stat().st_size,
        "pp": m["pp"], "tp": m["tp"], "seq": m["seq"],
        "request_id": m["request_id"],
        "p_tok": m["p_tok"], "w_tok": m["w_tok"], "layers": m["layers"],
        "layer_ids_head": m["layer_ids"][:2], "layer_ids_tail": m["layer_ids"][-2:],
        "cov": m["cov"], "block_table": m["block_table"],
        "kv_heads": m["kv_heads"]})
Path("tensors/manifest.json").write_text(
    json.dumps(mani, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(mani, ensure_ascii=False, indent=1))
PYEOF

  echo "== [pack] 打 tar 包 (log/ + tensors/) =="
  tar -czf kvc_bundle.tar.gz log/ tensors/
  echo "[DONE] $(ls -la kvc_bundle.tar.gz)"
  md5sum kvc_bundle.tar.gz 2>/dev/null || md5 -q kvc_bundle.tar.gz
  ;;

fetch)
  cd "$HERE"
  echo "== [fetch] 经隧道 $POD_PORT 拉回容器产物 =="
  scp -P "$POD_PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
      "root@localhost:$POD_WS/kvc_bundle.tar.gz" \
      /tmp/kvc_bundle.tar.gz
  echo "== [fetch] 校验 tar md5 (对照容器侧 pack 输出) =="
  md5 -q /tmp/kvc_bundle.tar.gz 2>/dev/null || md5sum /tmp/kvc_bundle.tar.gz
  echo "== [fetch] 解包到本地 (覆盖更新 log/ + tensors/) =="
  tar -xzf /tmp/kvc_bundle.tar.gz
  echo "== [fetch] \r 规范化 (tqdm 孤立 \r -> \n) =="
  python3 - <<'PYEOF'
from pathlib import Path
f = Path("log/llama-3-8b.log")
if f.exists():
    raw = f.read_bytes()
    norm = raw.replace(b"\r\r\n", b"\n").replace(b"\r\n", b"\n") \
              .replace(b"\r", b"\n")
    if norm != raw:
        f.write_bytes(norm)
        print(f"  {f}: 规范化 {raw.count(b'\r')} 个 \r")
    else:
        print(f"  {f}: 无需规范化")
PYEOF
  echo "== [fetch] 校验 manifest md5 =="
  python3 - <<'PYEOF'
import hashlib, json
from pathlib import Path
mani = json.loads(Path("tensors/manifest.json").read_text(encoding="utf-8"))
bad = 0
for e in mani["files"]:
    f = Path("tensors") / e["file"]
    if not f.exists():
        print(f"  [MISS] {e['file']}"); bad += 1; continue
    md5 = hashlib.md5(f.read_bytes()).hexdigest()
    ok = "OK" if md5 == e["md5"] else "MD5_MISMATCH"
    if ok != "OK":
        bad += 1
    print(f"  [{ok}] {e['file']} ({e['bytes']/1048576:.1f} MiB)")
raise SystemExit(1 if bad else 0)
PYEOF
  echo "[DONE] 产物已回收 — 查看: python3 scripts/inspect_kv_tensors.py --dir tensors"
  echo "       深查: python3 scripts/inspect_kv_tensors.py --file kv_pp0tp0_s1_*.pt"
  ;;

*)
  echo "用法: $0 pack(容器内) | fetch(主机侧, 需先建 SSH 隧道)"
  exit 1
  ;;
esac
