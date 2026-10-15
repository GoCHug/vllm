#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py — kv_visualizer 后端入口(FastAPI 所有接口定义 + CLI).

路由(GET):
  /            -> Vue 页面(frontend_vue/dist/index.html; 缺产物 404+build 指引)
  /viz.html    -> 同上(历史 URL 兼容)
  /docs        -> FastAPI 自动接口文档
  /api/verify  -> A~D 60 项核验 JSON
  /api/<rel>   -> kv_datasets_root/<rel>.json(源 mtime 变化自动重导)

CLI(路径默认见 config.py):
  python3 main.py serve  [--port 8000] [--bind 127.0.0.1] [--no-auto]
  python3 main.py export [--no-heat] [--root DIR] [--out DIR]
  python3 main.py verify [--root DIR] [--data DIR]
  python3 main.py pack   [--out viz_standalone.html]
"""
import argparse
import sys
from pathlib import Path

import config
from lib import exporter, verify
from lib.dataset_loader import api_verify, ensure_fresh
from lib.utils import safe_api_rel

ROOT_DEF = config.ROOT
PAGE_DIR = config.PAGE_DIR
DATA_ROOT = config.DATA_ROOT


# ================================================================ FastAPI 接口
def create_app(root, data, page_dir, auto=True):
    """装配 FastAPI 应用(所有 GET 路由)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    root, data, page_dir = Path(root), Path(data), Path(page_dir)
    app = FastAPI(title="kv_visualizer", docs_url="/docs")
    data_abs = str(data.resolve())

    @app.get("/", include_in_schema=False)
    @app.get("/viz.html", include_in_schema=False)
    def index():
        dist = page_dir / "dist" / "index.html"
        if dist.is_file():
            return FileResponse(dist, media_type="text/html")
        raise HTTPException(404, "frontend 构建产物缺失: cd frontend_vue && npm install && npm run build")

    @app.get("/api/verify")
    def verify_route():
        return api_verify(root, data)

    @app.get("/api/{rel:path}")
    def api_json(rel: str):
        rel = safe_api_rel(rel)
        if rel is None:
            raise HTTPException(400, "bad path")
        target = (data / rel).resolve()
        if not str(target).startswith(data_abs):
            raise HTTPException(400, "bad path")
        ensure_fresh(root, data, auto)
        if not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target, media_type="application/json")

    return app


def _serve_fastapi(root, data, page_dir, port, bind, auto):
    """启动 FastAPI + uvicorn; 阻塞直至 Ctrl+C. 返回退出码 0."""
    import uvicorn

    ensure_fresh(root, data, auto)
    app = create_app(root, data, page_dir, auto)
    print(f"[viz·fastapi] http://{bind}:{port}/ | docs: http://{bind}:{port}/docs")
    print(f"[viz·fastapi] 实验根: {root} | 数据: {data} | 源变更自动重导: {'开' if auto else '关'}")
    print("[viz·fastapi] Ctrl+C 停止")
    try:
        uvicorn.run(app, host=bind, port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n[viz·fastapi] 已停止")
    return 0


# ================================================================ CLI
def cmd_export(args):
    root = Path(args.root) if args.root else ROOT_DEF
    out = Path(args.out) if args.out else DATA_ROOT
    exporter.build(root, out, with_heat=not args.no_heat)
    return 0


def cmd_verify(args):
    root = Path(args.root) if args.root else ROOT_DEF
    if args.data:
        data = Path(args.data)
    elif args.out:
        data = Path(args.out)
    else:
        data = DATA_ROOT
    np_, nf_, _ = verify.run(root, data)
    return 0 if nf_ == 0 else 1


def cmd_serve(args):
    root = Path(args.root) if args.root else ROOT_DEF
    data = Path(args.out) if args.out else DATA_ROOT
    try:
        return _serve_fastapi(root, data, PAGE_DIR,
                              args.port, args.bind, args.auto)
    except ImportError:
        from lib import server_stdlib
        print("[viz] fastapi/uvicorn 未安装, 回退 stdlib 后端"
              "(pip install -r requirements.txt 可启用)")
        return server_stdlib.serve(root=root, data=data, page_dir=PAGE_DIR,
                                   port=args.port, bind=args.bind,
                                   auto=args.auto)


def cmd_pack(args):
    from lib import pack
    out = Path(args.out) if args.out else HERE_CALLER() / "viz_standalone.html"
    return pack.build(page_dir=PAGE_DIR, data=DATA_ROOT, out=out)


def HERE_CALLER():
    return Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(prog="kv-viz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="本地看页面(后端模式)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--root", default=None, help=f"实验根(默认 {ROOT_DEF})")
    p.add_argument("--out", default=None, help=f"数据目录(默认 {DATA_ROOT})")
    p.add_argument("--no-auto", dest="auto", action="store_false",
                   help="禁用源变更自动重导(仅服务现有数据)")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("export", help="静态导出到 kv_datasets_root")
    p.add_argument("--no-heat", action="store_true", help="跳过 32×32 热图")
    p.add_argument("--root", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("verify", help="独立核验(A~D 60 项)")
    p.add_argument("--root", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None, help="verify 的 --data 别名")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("pack", help="生成 standalone 单文件 HTML")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_pack)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
