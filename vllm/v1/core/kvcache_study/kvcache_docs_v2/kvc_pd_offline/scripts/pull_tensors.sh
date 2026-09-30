#!/bin/bash
# ==============================================================================
# pull_tensors.sh —— v2 双链产物回收（容器内 pack → 主机侧 fetch 两模式）
#
# 容器内(打包):
#   bash scripts/pull_tensors.sh pack
#   -> 等 .pt 全就位 -> md5sum 写 tensors/manifest.json -> tar.gz 整个 log/
#
# 主机侧(经隧道 5557 拉回, 在本工作区根执行):
#   bash scripts/pull_tensors.sh fetch
#   -> scp 容器内 log_offline_bundle.tar.gz 解开到本地 log/ (覆盖更新)
#
# 产物完整性: manifest.json 记录 md5, fetch 后自动校验; \r 规范化同 v1 纪律
# (p/d_llama.log 的 tqdm 孤立 \r 会在 fetch 侧统一转 \n, 保持行号口径)
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-pack}"

case "$MODE" in
pack)
  cd "$HERE"
  mkdir -p log/tensors
  echo "== [pack] 等待归档就位 (expect 4 .pt) =="
  for i in $(seq 1 12); do
    N=$(ls log/tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
    [ "$N" -ge 4 ] && break
    echo "  ... $N/4, 10s 后重试"; sleep 10
  done
  N=$(ls log/tensors/kv_*.pt 2>/dev/null | wc -l | tr -d " ")
  [ "$N" -ge 1 ] || { echo "[ERROR] 无归档产物"; exit 1; }
  [ "$N" = 4 ] || echo "[WARN] 归档数量 $N != 4 (P/D × req_p/req_r), 继续打包"

  echo "== [pack] 写 manifest.json (md5 + meta 摘要) =="
  python3 - <<'PYEOF'
import hashlib, json, glob, torch
from pathlib import Path
mani = {"files": []}
for f in sorted(glob.glob("log/tensors/kv_*.pt")):
    h = hashlib.md5(Path(f).read_bytes()).hexdigest()
    m = torch.load(f, map_location="cpu")["meta"]
    mani["files"].append({
        "file": Path(f).name, "md5": h, "bytes": Path(f).stat().st_size,
        "side": m["side"], "seq": m["seq"], "request_id": m["request_id"],
        "p_tok": m["p_tok"], "w_tok": m["w_tok"],
        "cov": m["cov"], "block_table": m["block_table"]})
Path("log/tensors/manifest.json").write_text(
    json.dumps(mani, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(mani, ensure_ascii=False, indent=1))
PYEOF

  echo "== [pack] 打 tar 包 =="
  tar -czf log_offline_bundle.tar.gz log/
  echo "[DONE] $(ls -la log_offline_bundle.tar.gz)"
  md5sum log_offline_bundle.tar.gz
  ;;

fetch)
  cd "$HERE"
  PORT="${PORT:-5557}"
  echo "== [fetch] 经隧道 $PORT 拉回容器产物 =="
  scp -P "$PORT" -o ConnectTimeout=10 \
      "root@localhost:/a3_inference/itask/workdir/wsl02075301/kvc_pd_offline/log_offline_bundle.tar.gz" \
      /tmp/kvc_offline_bundle.tar.gz
  echo "== [fetch] 校验 tar md5 (对照容器侧 pack 输出) =="
  md5 -q /tmp/kvc_offline_bundle.tar.gz || md5sum /tmp/kvc_offline_bundle.tar.gz
  echo "== [fetch] 解包到本地 log/ (覆盖更新) =="
  tar -xzf /tmp/kvc_offline_bundle.tar.gz
  echo "== [fetch] \r 规范化 (tqdm 孤立 \r -> \n, 同 v1 纪律) =="
  python3 - <<'PYEOF'
from pathlib import Path
for f in (Path("log/p_llama.log"), Path("log/d_llama.log")):
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
mani = json.loads(Path("log/tensors/manifest.json").read_text(encoding="utf-8"))
bad = 0
for e in mani["files"]:
    f = Path("log/tensors") / e["file"]
    if not f.exists():
        print(f"  [MISS] {e['file']}"); bad += 1; continue
    md5 = hashlib.md5(f.read_bytes()).hexdigest()
    ok = "OK" if md5 == e["md5"] else "MD5_MISMATCH"
    if ok != "OK":
        bad += 1
    print(f"  [{ok}] {e['file']} ({e['bytes']/1048576:.1f} MiB)")
raise SystemExit(1 if bad else 0)
PYEOF
  echo "[DONE] 产物已回收到本地 log/ — 可跑: python3 scripts/check_kv_tensors.py"
  ;;

*)
  echo "用法: $0 pack(容器内) | fetch(主机侧)"
  exit 1
  ;;
esac
