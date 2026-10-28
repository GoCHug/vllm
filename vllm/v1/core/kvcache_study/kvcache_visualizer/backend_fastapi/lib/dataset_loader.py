"""dataset_loader — kv_datasets_root 数据集层.

- ensure_fresh(root, data, auto)  实验源签名变化时重导(懒生成); 返回是否重导
- api_verify(root, data)          在线核验 A~D, 供 /api/verify 返回

新鲜度: 用 lib.utils.sources_signature 对 SOURCE_GLOBS 求 (mtime_ns,size) 签名,
与 data/.viz_sig.json 比对; 一致则跳过(不重复导出)。全程 _LOCK 串行化。
"""
import json
import threading
from pathlib import Path

import config
from lib import exporter, verify
from lib.utils import sources_signature

_LOCK = threading.Lock()
_SIG_NAME = ".viz_sig.json"


def _load_sig(data: Path):
    p = Path(data) / _SIG_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dump_sig(data: Path, sig):
    p = Path(data) / _SIG_NAME
    try:
        p.write_text(json.dumps({k: list(v) for k, v in sig.items()}),
                     encoding="utf-8")
    except Exception:
        pass


def ensure_fresh(root, data, auto=True):
    """签名变化 / 首次 时重导数据; 返回 True 表示本次发生了重导."""
    if not auto:
        return False
    root, data = Path(root), Path(data)
    try:
        sig = sources_signature(root)
    except Exception:
        return False
    if not sig and (data / "manifest.json").is_file():
        return False                      # 源不可见: 直接用已有数据
    cur = {k: list(v) for k, v in sig.items()}
    with _LOCK:
        if _load_sig(data) == cur and (data / "manifest.json").is_file():
            return False
        exporter.build(root, data, with_heat=True)
        _dump_sig(data, sig)
        return True


def api_verify(root, data):
    """在线核验 -> JSON(供 /api/verify)."""
    np_, nf_, items = verify.run(root, data, quiet=True)
    return {"ok": nf_ == 0, "pass": np_, "fail": nf_,
            "total": np_ + nf_, "items": items}