"""utils — 后端通用工具: 路径安全 / 目录签名 / JSON 落盘.

- safe_api_rel   校验 /api/<rel> 相对路径(拒绝绝对路径与 .. 逃逸)
- sources_signature  目录清单的 mtime+size 签名(新鲜度判定)
- write_json     写 JSON(自动建父目录, UTF-8 不转义)
"""
import json
import math
import time
from pathlib import Path

import config


def safe_api_rel(rel: str):
    """把 /api/<rel> 归一为安全相对路径; 非法返回 None(拒绝绝对路径 / .. / 空)."""
    if not rel:
        return None
    rel = rel.strip("/")
    if not rel:
        return None
    p = Path(rel)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        return None
    return p


def sources_signature(root: Path, globs=None):
    """按 SOURCE_GLOBS 遍历 root, 返回 {相对路径: (mtime_ns, size)} 的签名字典."""
    root = Path(root)
    globs = globs if globs is not None else config.SOURCE_GLOBS
    sig = {}
    for g in globs:
        for f in sorted(root.glob(g)):
            if f.is_file():
                st = f.stat()
                sig[str(f.relative_to(root))] = (st.st_mtime_ns, st.st_size)
    return sig


def jsonable(o):
    """递归把 NaN/+-Infinity 换成 None —— Python json.dumps 会写成非法的 `NaN` 字面量,
    浏览器 JSON.parse 会直接报 "Unexpected token 'N'", 必须预先净化。"""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def dump_json(obj):
    """合法 JSON 字符串(已净化 NaN/Inf)。"""
    return json.dumps(jsonable(obj), ensure_ascii=False)


def write_json(base: Path, rel, obj):
    """写 JSON 到 base/rel, 返回字节数(NaN/Inf 已净化)."""
    p = Path(base) / Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_json(obj), encoding="utf-8")
    return p.stat().st_size


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")