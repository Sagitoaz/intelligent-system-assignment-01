"""Máy chủ localhost cho các demo Agent 01 và Agent 02 (chỉ dùng thư viện chuẩn).

Chạy:  python agents/demo_app/server.py [--port 8100] [--open]
Thêm demo mới: tạo demos/<tên>.py (META + handle) và static/demos/<tên>.js; không cần sửa tệp khác.
"""
from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
import os
import pkgutil
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import demos  # noqa: E402

AGENTS = {
    "agent01": dict(title="Agent 01 — Mèo đuổi chuột & trợ lý đơn hàng", subtitle="Q-learning trên bản đồ lớn, BFS làm chuẩn, trợ lý dùng công cụ trên 60.000 đơn"),
    "agent02": dict(title="Agent 02 — Hai người bán cạnh tranh", subtitle="Q-learning so với luật cố định trong thị trường có cầu thay đổi"),
}
DEMOS = {}
ORDER = ["a1_chase", "a1_play", "a1_compare", "a1_orders", "a1_real", "a1_data", "a2_about", "a2_train", "a2_opponents", "a2_episode", "a2_sweep", "a2_market", "a2_real", "a2_data"]


def discover():
    DEMOS.clear()
    for info in pkgutil.iter_modules(demos.__path__):
        mod = importlib.import_module(f"demos.{info.name}")
        if hasattr(mod, "META"):
            DEMOS[mod.META["id"]] = mod


def manifest():
    return dict(agents=[dict(id=a, **meta, demos=[{k: v for k, v in m.META.items()}
                                                   for m in sorted(DEMOS.values(), key=lambda m: ORDER.index(m.META["id"]) if m.META["id"] in ORDER else 99)
                                                   if m.META["agent"] == a])
                        for a, meta in AGENTS.items()])


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/manifest":
            return self._send(200, manifest())
        if url.path.startswith("/api/demo/"):
            demo = DEMOS.get(url.path.rsplit("/", 1)[-1])
            if not demo:
                return self._send(404, dict(error="Không có demo này"))
            try:
                return self._send(200, demo.handle(dict(parse_qsl(url.query))))
            except Exception as exc:  # trả lỗi dạng JSON để giao diện hiển thị được
                traceback.print_exc()
                return self._send(400, dict(error=f"{type(exc).__name__}: {exc}"))
        rel = "index.html" if url.path in ("", "/") else url.path.lstrip("/")
        full = os.path.realpath(os.path.join(HERE, "static", rel))
        if not full.startswith(os.path.join(HERE, "static")) or not os.path.isfile(full):
            return self._send(404, b"Not found", "text/plain")
        with open(full, "rb") as f:
            self._send(200, f.read(), (mimetypes.guess_type(full)[0] or "text/plain") + "; charset=utf-8")

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    discover()
    import shared
    for key in shared.TRAIN_EPISODES:  # nạp (hoặc huấn luyện lần đầu) Q-table của mèo cho cả ba bản đồ
        print(f"Chuẩn bị bản đồ {key} ...", flush=True)
        shared.get_cat(key)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"{len(DEMOS)} demo sẵn sàng tại {url}  (Ctrl+C để dừng)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
