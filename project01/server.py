"""Máy chủ localhost của "Phòng khám AI" (hệ đa agent cho bệnh tiểu đường).

Chạy:  .venv\\Scripts\\python.exe project01/server.py [--port 8200] [--open]
Cần numpy và matplotlib (có sẵn trong .venv). Mô hình Deep Learning và Q-table đã huấn luyện nằm trong project01/artifacts/.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from project01.mas import rl  # noqa: E402
from project01.mas.orchestrator import Clinic  # noqa: E402

STATIC = os.path.join(HERE, "static")
CLINIC: Clinic | None = None


def life_start(body: dict) -> dict:
    prof, dl, pat = CLINIC.patient(body.get("profile") or {})
    state = rl.new_state(pat, random.Random(int(body.get("seed", 7))))
    return dict(patient=pat, state=state, actions=[{k: a[k] for k in ("id", "label", "emoji", "desc")} for a in rl.ACTIONS],
                allowed=rl.allowed(state["hour"]), advice=rl.advice(state, CLINIC.coach), risk=dl["risk"])


def life_step(body: dict) -> dict:
    state = body["state"]
    action = body["action"]
    if action not in rl.allowed(state["hour"]):
        raise ValueError("Hành động không khả dụng vào giờ này (ban đêm chỉ được nghỉ hoặc ăn nhẹ có đường)")
    seed = int(body.get("seed", 7)) * 100 + state["hour"]
    new, reward, info = rl.step(state, action, random.Random(seed))
    events = []
    if info["zone"] == "low":
        events.append("Đường huyết xuống thấp, bệnh nhân run và đói.")
    elif info["zone"] == "danger":
        events.append("Đường huyết ở mức nguy hiểm!")
    elif info["zone"] == "high":
        events.append("Đường huyết đang cao hơn mức an toàn.")
    if info["rise"] > 25:
        events.append(f"Bữa ăn làm đường huyết tăng khoảng {info['rise']:.0f} mg/dL.")
    if info["drop"] > 15:
        events.append(f"Vận động và thuốc giúp đường huyết giảm khoảng {info['drop']:.0f} mg/dL.")
    log = new["log"]
    tir = sum(70 <= r["glucose"] <= 180 for r in log) / len(log)
    return dict(state=new, glucose_before=state["glucose"], glucose_after=new["glucose"], reward=round(reward, 2), zone=info["zone"], events=events,
                advice=rl.advice(new, CLINIC.coach) if not rl.done(new) else None, done=rl.done(new), allowed=rl.allowed(new["hour"]) if not rl.done(new) else [],
                score=dict(time_in_range=round(tir, 3), hypo_hours=sum(r["glucose"] < 70 for r in log), hyper_hours=sum(r["glucose"] > 180 for r in log),
                           total_reward=None))


def life_compare(body: dict) -> dict:
    prof, dl, pat = CLINIC.patient(body.get("profile") or {})
    out = rl.compare(pat, CLINIC.coach, days=max(5, min(60, int(body.get("days", 30)))), seed=int(body.get("seed", 7)))
    out["patient"] = pat
    return out


ROUTES_POST = {
    "/api/consult": lambda b: CLINIC.consult(b.get("profile") or {}),
    "/api/kg/search": lambda b: kg_search(b),
    "/api/life/start": life_start,
    "/api/life/step": life_step,
    "/api/life/compare": life_compare,
}


def kg_search(body: dict) -> dict:
    q = str(body.get("q", ""))[:300]
    r = CLINIC.kb.search(q, 3)
    answer = ""
    if r["passages"]:
        answer = r["passages"][0]["quote"][:320]
    return dict(nodes=r["nodes"], seeds=r["seeds"], passages=r["passages"], facts=r["facts"], answer=answer)


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
        if path == "/api/meta":
            return self._json(200, CLINIC.meta())
        if path == "/api/kg":
            return self._json(200, CLINIC.kb.graph())
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC, rel))
        if not full.startswith(os.path.realpath(STATIC)) or not os.path.isfile(full):
            return self._json(404, dict(error="Không tìm thấy"))
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(full)[0] or "application/octet-stream") + ("; charset=utf-8" if full.endswith((".html", ".js", ".css", ".json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        fn = ROUTES_POST.get(path)
        if not fn:
            return self._json(404, dict(error="Không có API này"))
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            return self._json(200, fn(body))
        except (ValueError, KeyError, TypeError) as exc:
            return self._json(400, dict(error=f"Dữ liệu không hợp lệ: {exc}"))
        except Exception as exc:  # trả lỗi dạng JSON để giao diện hiển thị
            traceback.print_exc()
            return self._json(500, dict(error=f"{type(exc).__name__}: {exc}"))

    def log_message(self, fmt, *args):
        pass


def main():
    global CLINIC
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    print("Đang nạp mô hình Deep Learning, đồ thị tri thức và Q-table ...", flush=True)
    CLINIC = Clinic()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Phòng khám AI sẵn sàng tại {url}  (Ctrl+C để dừng)", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
