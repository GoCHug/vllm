"""pack — standalone 单文件打包器.

把 frontend_vue/dist/index.html 与 kv_datasets_root 全部 JSON 内嵌成
一个可双击打开(或任意 http 服务)的单文件 HTML:

    <script id="viz-embedded" type="application/json">{manifest + 归档}</script>

前端 legacy.js 检测到 #viz-embedded 即走 embedded 模式(零后端)。
产物不驻留工作区, 用完可删, 需要时秒级重建。
"""
import json
from pathlib import Path

from lib.utils import dump_json

MARK = '</head>'


def _collect(data: Path):
    """{相对路径(不含 .json): 对象} —— manifest 用 'manifest' 键."""
    bundle = {}
    for f in sorted(Path(data).rglob("*.json")):
        if f.name.startswith("."):
            continue                      # 跳过 .viz_sig 等内部文件
        rel = f.relative_to(data).with_suffix("").as_posix()
        try:
            bundle[rel] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
    return bundle


def build(page_dir, data, out):
    page_dir, data, out = Path(page_dir), Path(data), Path(out)
    src = page_dir / "dist" / "index.html"
    if not src.is_file():
        print(f"[viz·pack] 缺前端产物: {src} —— 先 cd frontend_vue && npm run build")
        return 1
    html = src.read_text(encoding="utf-8")
    bundle = _collect(data)
    payload = dump_json(bundle)   # 净化 NaN/Inf, 否则单文件内嵌数据同样非法
    tag = f'<script id="viz-embedded" type="application/json">{payload}</script>'
    if MARK in html:
        html = html.replace(MARK, tag + MARK, 1)
    else:
        html = tag + html
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    print(f"[viz·pack] {out} ({out.stat().st_size / 1048576:.1f} MiB, "
          f"{len(bundle)} 个 JSON 内嵌)")
    return 0