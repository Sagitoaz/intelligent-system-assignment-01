"""Demo localhost cho Chương 6 (Deep Graph): đồ thị cổ phiếu VN30, lan truyền thông điệp và GCN đoán ngành.

Chạy:  .venv\\Scripts\\python.exe deepgraph_demo/server.py [--port 8300] [--open]
Cần torch, numpy, pandas, scikit-learn (có sẵn trong .venv). Dữ liệu giá lấy từ data/tieuluan/raw/vn_stocks.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from torch import nn  # noqa: E402

from scripts.tieuluan import complete_gnn as g  # noqa: E402

STATIC = os.path.join(HERE, "static")
DATA = {}
LOCK = threading.Lock()  # torch huấn luyện một lượt tại một thời điểm cho gọn


def knn_adj(corr: np.ndarray, k: int) -> np.ndarray:
    n = len(corr)
    c = corr.copy()
    np.fill_diagonal(c, -1)
    a = np.zeros((n, n))
    for i in range(n):
        for j in np.argsort(-c[i])[:k]:
            a[i, j] = a[j, i] = 1
    return a


def load():
    close, tickers, f, label, corr = g.sector_data()
    logp = np.log(close)
    r5 = (logp.iloc[-1] - logp.iloc[-6]).values
    r5 = (r5 - r5.mean()) / (r5.std() + 1e-9)
    sector_of = {t: s for s, ts in g.SECTORS.items() for t in ts}
    DATA.update(close=close, tickers=tickers, f=f, label=label, corr=corr, r5=r5, sector=[sector_of[t] for t in tickers], last=str(close.index[-1].date()))


def api_graph() -> dict:
    return dict(tickers=DATA["tickers"], sector=DATA["sector"], sectors=list(g.SECTORS), label=DATA["label"].tolist(), corr=np.round(DATA["corr"], 3).tolist(),
                feature=np.round(DATA["r5"], 3).tolist(), feature_name="lợi suất 5 phiên gần nhất (chuẩn hóa)", last=DATA["last"],
                node_features=["độ biến động", "beta", "độ lệch", "độ nhọn", "log khối lượng", "tự tương quan bậc 1"])


def train_predict(model, x, yt, tr_idx, seed, epochs=150):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(x)[0][tr_idx], yt[tr_idx]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x)[0]).numpy()


def api_sector(body: dict) -> dict:
    n = len(DATA["tickers"])
    labelled = sorted({int(i) for i in body.get("labelled", []) if 0 <= int(i) < n})
    k = max(1, min(10, int(body.get("k", 4))))
    label = DATA["label"]
    if len(labelled) < 2 or len({int(label[i]) for i in labelled}) < 2:
        raise ValueError("Cần ít nhất 2 mã đã biết nhãn, gồm cả ngân hàng và không phải ngân hàng")
    test = [i for i in range(n) if i not in labelled]
    x = torch.tensor(DATA["f"], dtype=torch.float32)[None]
    yt = torch.tensor(label, dtype=torch.float32)
    a_corr = knn_adj(DATA["corr"], k)
    with LOCK:
        p_mlp = train_predict(g.NodeMLP(x.shape[-1], hid=16, drop=0.2), x, yt, labelled, 0)
        p_gcn = train_predict(g.GCN(g.norm_adj(a_corr), x.shape[-1], hid=16, drop=0.2), x, yt, labelled, 0)
    acc = lambda p: float(np.mean([(p[i] > 0.5) == bool(label[i]) for i in test]))  # noqa: E731
    return dict(labelled=labelled, k=k, edges=int(a_corr.sum() / 2), mlp=dict(prob=np.round(p_mlp, 3).tolist(), accuracy=acc(p_mlp)),
                gcn=dict(prob=np.round(p_gcn, 3).tolist(), accuracy=acc(p_gcn)), majority=float(max(label[test].mean(), 1 - label[test].mean())), n_test=len(test))


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
        if path == "/api/graph":
            return self._json(200, api_graph())
        full = os.path.realpath(os.path.join(STATIC, "index.html" if path in ("", "/") else path.lstrip("/")))
        if not full.startswith(os.path.realpath(STATIC)) or not os.path.isfile(full):
            return self._json(404, dict(error="Không tìm thấy"))
        data = open(full, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(full)[0] or "application/octet-stream") + ("; charset=utf-8" if full.endswith((".html", ".js", ".css")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if urlparse(self.path).path != "/api/sector":
            return self._json(404, dict(error="Không có API này"))
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            return self._json(200, api_sector(body))
        except (ValueError, TypeError) as exc:
            return self._json(400, dict(error=str(exc)))

    def log_message(self, fmt, *args):
        pass


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    print("Đang nạp dữ liệu VN30 ...", flush=True)
    load()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    import socket
    if socket.has_ipv6:
        class V6(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        try:
            threading.Thread(target=V6(("::1", args.port), Handler).serve_forever, daemon=True).start()
        except OSError:
            pass
    url = f"http://localhost:{args.port}"
    print(f"Demo Deep Graph sẵn sàng tại {url}  (Ctrl+C để dừng)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
