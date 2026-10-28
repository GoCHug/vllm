"""server_stdlib — 纯标准库 HTTP 回退后端(缺 fastapi/uvicorn 时 serve 自动启用).

路由与 FastAPI 版一致:
    GET /            -> frontend_vue/dist/index.html(缺产物 404 + build 指引)
    GET /viz.html    -> 同上(历史 URL 兼容)
    GET /api/verify  -> A~D 核验 JSON
    GET /api/<rel>   -> kv_datasets_root/<rel>.json(源变化自动重导)
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from lib.dataset_loader import api_verify, ensure_fresh
from lib.utils import safe_api_rel

MIME = {".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8",
        ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def _make_handler(root, data, page_dir, auto):
    root, data, page_dir = Path(root), Path(data), Path(page_dir)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):   # 静音默认访问日志
            pass

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                       MIME[".json"])

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/viz.html"):
                dist = page_dir / "dist" / "index.html"
                if dist.is_file():
                    return self._send(200, dist.read_bytes(), MIME[".html"])
                return self._send(404,
                                  "frontend 构建产物缺失: cd frontend_vue && "
                                  "npm install && npm run build".encode(),
                                  MIME[".html"])
            if path == "/api/verify":
                try:
                    return self._json(200, api_verify(root, data))
                except Exception as e:
                    return self._json(500, {"ok": False, "error": repr(e)})
            if path.startswith("/api/"):
                rel = safe_api_rel(path[len("/api/"):])
                if rel is None:
                    return self._json(400, {"error": "bad path"})
                target = (data / rel).resolve()
                if not str(target).startswith(str(data.resolve())):
                    return self._json(400, {"error": "bad path"})
                ensure_fresh(root, data, auto)
                if target.is_file():
                    ctype = MIME.get(target.suffix, "application/octet-stream")
                    return self._send(200, target.read_bytes(), ctype)
                return self._json(404, {"error": "not found"})
            self._json(404, {"error": "not found"})

    return H


def serve(root, data, page_dir, port=8000, bind="127.0.0.1", auto=True):
    ensure_fresh(root, data, auto)
    httpd = ThreadingHTTPServer((bind, port), _make_handler(root, data, page_dir, auto))
    print(f"[viz·stdlib] http://{bind}:{port}/ | 实验根: {root} | 数据: {data}")
    print("[viz·stdlib] 纯标准库后端(pip install -r requirements.txt 可启用 FastAPI)")
    print("[viz·stdlib] Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz·stdlib] 已停止")
    finally:
        httpd.server_close()
    return 0