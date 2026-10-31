"""Demo gộp của tiểu luận (7 chương) trên localhost: một địa chỉ, mỗi chương một tab.

Chạy:  .venv\\Scripts\\python.exe tieuluan_demo/server.py [--port 8500] [--open] [--no-children]
Máy chủ này mở trang tổng hợp và tự khởi động các demo con trên 127.0.0.1:
  Chương 2–4 · Streamlit        cổng 8502  (deployment/tieuluan/app.py, lọc tab theo ?chapter=)
  Chương 5, 7 · FastAPI         cổng 8044  (deployment/chapter04/app.py: hỏi đáp KG-RAG và /agents)
  Chương 6 · Deep Graph         cổng 8300  (deepgraph_demo/server.py)
Chương 1 do chính máy chủ này phục vụ (dòng thời gian AI và thống kê ba chỉ số, tính lại từ dữ liệu).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
APP_DATA = os.path.join(ROOT, "data", "tieuluan", "app")
PY = sys.executable

CHILDREN = [
    dict(name="streamlit", port=8502, check="/_stcore/health", chapters="2, 3, 4",
         cmd=[PY, "-m", "streamlit", "run", "deployment/tieuluan/app.py", "--server.port", "8502", "--server.headless", "true", "--browser.gatherUsageStats", "false"]),
    dict(name="fastapi", port=8044, check="/health", chapters="5, 7",
         cmd=[PY, "-m", "uvicorn", "deployment.chapter04.app:app", "--host", "127.0.0.1", "--port", "8044"]),
    dict(name="deepgraph", port=8300, check="/api/graph", chapters="6",
         cmd=[PY, "deepgraph_demo/server.py", "--port", "8300"]),
]
PROCS: list[subprocess.Popen] = []
INDEXES = {"sp500": ("S&P 500", 252, "#2589e0"), "vnindex": ("VN-Index", 252, "#f0603a"), "btc": ("Bitcoin", 365, "#e8a317")}
CACHE: dict = {}


def healthy(port: int, path: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_children():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for c in CHILDREN:
        if healthy(c["port"], c["check"]):
            print(f"  {c['name']}: đã chạy sẵn ở cổng {c['port']}", flush=True)
            continue
        print(f"  khởi động {c['name']} (cổng {c['port']}) ...", flush=True)
        log = open(os.path.join(tempfile.gettempdir(), f"tieuluan_demo_{c['name']}.log"), "wb")  # log từng demo con, xem khi có lỗi
        PROCS.append(subprocess.Popen(c["cmd"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))


def stop_children():
    for p in PROCS:
        try:
            p.terminate()
        except Exception:
            pass


def ch1_data() -> dict:
    if "ch1" in CACHE:
        return CACHE["ch1"]
    milestones = json.load(open(os.path.join(HERE, "ch1_milestones.json"), encoding="utf-8"))
    stats, series, hist = [], {}, {}
    for key, (name, days, _) in INDEXES.items():
        df = pd.read_csv(os.path.join(APP_DATA, f"{key}_prices.csv"), parse_dates=["date"])
        r = np.log(df["close"]).diff().dropna()
        stats.append(dict(key=key, name=name, first=str(df.date.iloc[0].date()), last=str(df.date.iloc[-1].date()), n=int(len(df)), mean=float(r.mean() * 100),
                          std=float(r.std() * 100), annual=float(r.std() * np.sqrt(days) * 100), min=float(r.min() * 100), max=float(r.max() * 100),
                          kurt=float(r.kurt()), up=float((r > 0).mean() * 100)))
        step = max(1, len(df) // 700)
        series[key] = [[str(d.date()), round(float(c), 2)] for d, c in zip(df.date.iloc[::step], df.close.iloc[::step])]
        lim = float(r.std() * 5)
        counts, edges = np.histogram(r.clip(-lim, lim), bins=41, range=(-lim, lim))
        centers = (edges[:-1] + edges[1:]) / 2
        normal = np.exp(-0.5 * ((centers - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi)) * (edges[1] - edges[0]) * len(r)
        hist[key] = dict(centers=(centers * 100).round(4).tolist(), counts=counts.tolist(), normal=normal.round(2).tolist())
    CACHE["ch1"] = dict(milestones=milestones, stats=stats, series=series, hist=hist, colors={k: v[2] for k, v in INDEXES.items()})
    return CACHE["ch1"]


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/ch1":
            return self._json(200, ch1_data())
        if path == "/api/status":
            return self._json(200, {c["name"]: healthy(c["port"], c["check"]) for c in CHILDREN})
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC, rel))
        if not full.startswith(os.path.realpath(STATIC)) or not os.path.isfile(full):
            return self._json(404, dict(error="Không tìm thấy"))
        data = open(full, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(full)[0] or "application/octet-stream") + ("; charset=utf-8" if full.endswith((".html", ".js", ".css")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--no-children", action="store_true", help="không tự khởi động các demo con (đã chạy riêng)")
    args = ap.parse_args()
    print("Demo tiểu luận (7 chương)", flush=True)
    if not args.no_children:
        start_children()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if socket.has_ipv6:
        class V6(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        try:
            threading.Thread(target=V6(("::1", args.port), Handler).serve_forever, daemon=True).start()
        except OSError:
            pass
    url = f"http://localhost:{args.port}"
    print(f"Mở {url}  (các tab Chương 2–7 cần vài chục giây để các demo con khởi động; Ctrl+C để dừng tất cả)", flush=True)
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_children()


if __name__ == "__main__":
    main()
