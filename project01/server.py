"""Máy chủ localhost của "Phòng khám AI" (hệ đa agent cho bệnh tiểu đường).

Chạy:  .venv\\Scripts\\python.exe project01/server.py [--port 8200] [--open]
Cần numpy và matplotlib (có sẵn trong .venv). Mô hình Deep Learning và Q-table đã huấn luyện nằm trong project01/artifacts/.

======================================================================================================
BẢN ĐỒ FILE (đọc phần này để biết "cái này nằm ở đâu")
======================================================================================================
Vai trò: đây là lớp WEB mỏng, chỉ dùng thư viện chuẩn http.server. Nó KHÔNG chứa thuật toán AI; mọi AI nằm trong
project01/mas/. Server nhận JSON từ trình duyệt (project01/static/*.js), gọi hàm trong mas/, trả JSON (hoặc PDF/MP3).

Luồng chạy:  main() -> tạo Clinic() (nạp MLP, KG, Q-table) -> load_extensions() (đăng ký thêm route) -> serve_forever().

Các hàm/lớp chính trong file này:
  - life_start / life_step / life_compare : 3 API của mô phỏng "Sống cùng bệnh" (RL coach) -> gọi mas/rl.py
  - kg_search                             : API tra đồ thị tri thức + RAG -> gọi mas/kb.py (Clinic.kb.search)
  - load_extensions                       : nạp các module mở rộng trong EXTENSIONS, mỗi module tự khai báo route
  - Handler (do_GET, do_POST, _tts, ...)  : xử lý HTTP; trả file tĩnh trong static/ nếu không phải API
  - main                                  : đọc tham số --port/--open, khởi động server (cả IPv4 và IPv6)

BẢNG ENDPOINT  ->  HÀM XỬ LÝ  ->  NƠI CÀI ĐẶT THẬT SỰ
  GET  /api/meta            Clinic.meta()                       mas/orchestrator.py  (danh sách agent, triệu chứng, thông số mô hình)
  GET  /api/kg              Clinic.kb.graph()                   mas/kb.py            (toàn bộ nút/cạnh của đồ thị tri thức)
  GET  /api/tts/status      SPEAKER.enabled                     mas/tts.py           (có khóa Azure Speech hay không)
  GET  /api/stt/status      (lambda trong stt.routes)           mas/stt.py           (có khóa nhận dạng giọng nói hay không)
  GET  /api/trust           trust._trust                        mas/trust.py         (số liệu độ tin cậy/hiệu chuẩn của mô hình DL)
  GET  /api/trust/fairness  trust._fairness                     mas/trust.py         (độ công bằng theo nhóm tuổi/giới...)
  POST /api/consult         Clinic.consult(profile)             mas/orchestrator.py  (HỘI CHẨN: gọi lần lượt các agent)
  POST /api/kg/search       kg_search (file này)                mas/kb.py            (Graph-RAG: nút liên quan + đoạn trích NIDDK)
  POST /api/life/start      life_start (file này)               mas/rl.py            (bắt đầu một ngày mô phỏng)
  POST /api/life/step       life_step (file này)                mas/rl.py            (một giờ mô phỏng theo hành động đã chọn)
  POST /api/life/compare    life_compare (file này)             mas/rl.py            (so sánh RL / luật / không huấn luyện viên)
  POST /api/tts             Handler._tts                        mas/tts.py           (đọc văn bản thành giọng nói, trả MP3)
  POST /api/ask             ask.routes                          mas/ask.py           ("Hỏi bác sĩ": RAG trên corpus 492 đoạn)
  POST /api/queue           followup.routes                     mas/followup.py      (hàng đợi bệnh nhân, phân loại ưu tiên)
  POST /api/followup        followup.routes                     mas/followup.py      (theo dõi xu hướng 4 tuần)
  POST /api/report/pdf      pdf_report.routes                   mas/pdf_report.py    (xuất báo cáo khám dạng PDF)
  POST /api/whatif          whatif.routes                       mas/whatif.py        (giả định: đổi BMI/thói quen thì nguy cơ ra sao)
  POST /api/whatif/curve    whatif.routes                       mas/whatif.py        (đường cong nguy cơ theo BMI)
  POST /api/stt             stt.routes                          mas/stt.py           (nhận dạng giọng nói bằng Azure STT)
  POST /api/stt/parse       stt.routes                          mas/stt.py           (chỉ phân tích câu chữ bằng luật, không cần mạng)
  GET  /<bất kỳ>            Handler.do_GET                      static/              (index.html, app.js, *.css, fixtures/)
Ghi chú: tên route của các module mở rộng (ask, trust, followup, pdf_report, stt, whatif) lấy từ hàm routes() ở cuối mỗi file đó.
Nếu thầy hỏi "chức năng X gọi API nào", tìm đường dẫn /api/... trong static/app.js (và ask.js, queue.js, explain.js) rồi tra bảng trên.
======================================================================================================
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import sys
import socket
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # thêm thư mục gốc repo vào sys.path để import được package "project01"
from project01.mas import rl  # noqa: E402
from project01.mas.tts import Speaker  # noqa: E402
from project01.mas.orchestrator import Clinic  # noqa: E402

STATIC = os.path.join(HERE, "static")  # thư mục giao diện web (HTML/JS/CSS)
CLINIC: Clinic | None = None  # đối tượng "Bác sĩ trưởng": giữ MLP + KG + Q-table, được tạo một lần trong main()
SPEAKER = Speaker()  # đối tượng đọc giọng nói (Azure TTS, tùy chọn)


def life_start(body: dict) -> dict:
    """API /api/life/start: bắt đầu một ngày mô phỏng cho bệnh nhân ảo.

    Đầu vào: body = {profile: hồ sơ bệnh nhân, seed: hạt giống ngẫu nhiên (mặc định 7)}.
    Đầu ra: bệnh nhân, trạng thái ban đầu (rl.new_state), danh sách hành động, hành động được phép vào giờ này,
    lời khuyên của Huấn luyện viên RL (rl.advice) và mức nguy cơ do MLP ước lượng.
    """
    prof, dl, pat = CLINIC.patient(body.get("profile") or {})
    state = rl.new_state(pat, random.Random(int(body.get("seed", 7))))
    return dict(patient=pat, state=state, actions=[{k: a[k] for k in ("id", "label", "emoji", "desc")} for a in rl.ACTIONS],
                allowed=rl.allowed(state["hour"]), advice=rl.advice(state, CLINIC.coach), risk=dl["risk"])


def life_step(body: dict) -> dict:
    """API /api/life/step: thực hiện MỘT giờ mô phỏng với hành động người chơi chọn.

    Đầu vào: body = {state: trạng thái hiện tại, action: id hành động, seed}.
    Đầu ra: trạng thái mới, đường huyết trước/sau, reward, vùng đường huyết (low/normal/high/danger),
    các sự kiện bằng lời, lời khuyên tiếp theo và điểm (time_in_range, số giờ hạ/tăng đường huyết).
    Nếu hành động không hợp lệ (ban đêm chỉ được nghỉ/ăn nhẹ có đường) thì ném ValueError -> do_POST trả HTTP 400.
    """
    state = body["state"]
    action = body["action"]
    if action not in rl.allowed(state["hour"]):
        raise ValueError("Hành động không khả dụng vào giờ này (ban đêm chỉ được nghỉ hoặc ăn nhẹ có đường)")
    seed = int(body.get("seed", 7)) * 100 + state["hour"]  # seed phụ thuộc giờ để mỗi bước có nhiễu khác nhau nhưng tái lập được
    new, reward, info = rl.step(state, action, random.Random(seed))
    events = []  # mô tả bằng lời những gì vừa xảy ra, để giao diện hiển thị cho người xem
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
    tir = sum(70 <= r["glucose"] <= 180 for r in log) / len(log)  # time in range: tỷ lệ giờ đường huyết trong ngưỡng 70–180 mg/dL
    return dict(state=new, glucose_before=state["glucose"], glucose_after=new["glucose"], reward=round(reward, 2), zone=info["zone"], events=events,
                advice=rl.advice(new, CLINIC.coach) if not rl.done(new) else None, done=rl.done(new), allowed=rl.allowed(new["hour"]) if not rl.done(new) else [],
                score=dict(time_in_range=round(tir, 3), hypo_hours=sum(r["glucose"] < 70 for r in log), hyper_hours=sum(r["glucose"] > 180 for r in log),
                           total_reward=None))


def life_compare(body: dict) -> dict:
    """API /api/life/compare: chạy nhiều ngày mô phỏng và so sánh 3 phương án (RL, luật, không huấn luyện viên).

    Đầu vào: body = {profile, days (kẹp trong 5..60, mặc định 30), seed}.
    Đầu ra: kết quả rl.compare (time_in_range, đường huyết trung bình, số giờ hạ/tăng...) kèm thông tin bệnh nhân.
    """
    prof, dl, pat = CLINIC.patient(body.get("profile") or {})
    out = rl.compare(pat, CLINIC.coach, days=max(5, min(60, int(body.get("days", 30)))), seed=int(body.get("seed", 7)))
    out["patient"] = pat
    return out


# Các mô-đun mở rộng (project01/mas/<tên>.py): mỗi mô-đun có thể định nghĩa routes(clinic, speaker) -> (get, post).
# Handler POST nhận dict JSON; trả dict (JSON) hoặc tuple (content_type, bytes) cho nội dung nhị phân (PDF...).
EXTENSIONS = ["ask", "trust", "followup", "pdf_report", "stt", "whatif"]
ROUTES_GET: dict = {}  # bảng route GET do các module mở rộng đăng ký (đường dẫn -> hàm không tham số)


def load_extensions():
    """Nạp từng module trong EXTENSIONS và gộp route của chúng vào ROUTES_GET/ROUTES_POST.

    Module nào chưa tồn tại thì bỏ qua (không làm sập server); lỗi import khác thì ném lại để lộ bug.
    Gọi một lần trong main(), SAU khi CLINIC đã được tạo vì routes() cần clinic.
    """
    import importlib
    for name in EXTENSIONS:
        try:
            mod = importlib.import_module(f"project01.mas.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"project01.mas.{name}":
                raise
            continue
        if hasattr(mod, "routes"):
            g, p = mod.routes(CLINIC, SPEAKER)
            ROUTES_GET.update(g)
            ROUTES_POST.update(p)


# Bảng route POST cốt lõi (đường dẫn -> hàm nhận dict JSON). Các route mở rộng được thêm vào đây bởi load_extensions().
ROUTES_POST = {
    "/api/consult": lambda b: CLINIC.consult(b.get("profile") or {}),  # hội chẩn đa agent -> mas/orchestrator.py
    "/api/kg/search": lambda b: kg_search(b),
    "/api/life/start": life_start,
    "/api/life/step": life_step,
    "/api/life/compare": life_compare,
}


def kg_search(body: dict) -> dict:
    """API /api/kg/search: hỏi đáp trên đồ thị tri thức bằng Graph-RAG (mas/kb.py).

    Đầu vào: body = {q: câu hỏi} (cắt tối đa 300 ký tự).
    Đầu ra: nodes/seeds (nút liên quan trong đồ thị), passages (đoạn trích NIDDK có nguồn), facts (quan hệ),
    answer (câu trả lời ngắn = trích đoạn đầu tiên, tối đa 320 ký tự).
    """
    q = str(body.get("q", ""))[:300]
    r = CLINIC.kb.search(q, 3)
    answer = ""
    if r["passages"]:
        answer = r["passages"][0]["quote"][:320]
    return dict(nodes=r["nodes"], seeds=r["seeds"], passages=r["passages"], facts=r["facts"], answer=answer)


class Handler(BaseHTTPRequestHandler):
    """Bộ xử lý HTTP: GET trả JSON hoặc file tĩnh; POST gọi hàm trong ROUTES_POST; log tắt để console sạch."""

    def _json(self, code, obj):
        """Gửi phản hồi JSON UTF-8 với mã HTTP `code` (không cache)."""
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_result(self, res):
        """Gửi kết quả của route: tuple (content_type, bytes[, Content-Disposition]) -> file nhị phân (PDF); còn lại -> JSON."""
        if isinstance(res, tuple):
            ctype, data = res[0], res[1]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if len(res) > 2:
                self.send_header("Content-Disposition", res[2])
            self.end_headers()
            self.wfile.write(data)
            return
        return self._json(200, res)

    def do_GET(self):
        """Xử lý GET: API cố định -> route mở rộng -> file tĩnh trong static/ (chặn đọc ra ngoài thư mục static)."""
        path = urlparse(self.path).path
        if path == "/api/meta":
            return self._json(200, CLINIC.meta())
        if path == "/api/tts/status":
            return self._json(200, dict(enabled=SPEAKER.enabled))
        if path == "/api/kg":
            return self._json(200, CLINIC.kb.graph())
        if path in ROUTES_GET:
            try:
                return self._send_result(ROUTES_GET[path]())
            except Exception as exc:
                traceback.print_exc()
                return self._json(500, dict(error=f"{type(exc).__name__}: {exc}"))
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC, rel))
        # chống path traversal (../): đường dẫn thật phải nằm trong STATIC
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

    def _tts(self):
        """POST /api/tts: đọc văn bản thành MP3 (mas/tts.py). 400 nếu thiếu nội dung, 503 nếu chưa có khóa/lỗi mạng."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            audio = SPEAKER.synthesize(str(body.get("text", "")), str(body.get("agent", "doctor")))
        except ValueError as exc:
            return self._json(400, dict(error=str(exc)))
        except RuntimeError as exc:
            return self._json(503, dict(error=str(exc)))
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(audio)

    def do_POST(self):
        """Xử lý POST: tra ROUTES_POST rồi gọi hàm với body JSON.

        Quy ước mã lỗi: 400 = dữ liệu vào sai (ValueError/KeyError/TypeError); 503 = dịch vụ ngoài thiếu khóa/mạng
        (RuntimeError); 500 = lỗi không lường trước (in traceback ra console, trả JSON lỗi cho giao diện).
        """
        path = urlparse(self.path).path
        if path == "/api/tts":
            return self._tts()
        fn = ROUTES_POST.get(path)
        if not fn:
            return self._json(404, dict(error="Không có API này"))
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            return self._send_result(fn(body))
        except (ValueError, KeyError, TypeError) as exc:
            return self._json(400, dict(error=f"Dữ liệu không hợp lệ: {exc}"))
        except RuntimeError as exc:  # thiếu khóa/lỗi mạng của dịch vụ ngoài (Azure STT...)
            return self._json(503, dict(error=str(exc)))
        except Exception as exc:  # trả lỗi dạng JSON để giao diện hiển thị
            traceback.print_exc()
            return self._json(500, dict(error=f"{type(exc).__name__}: {exc}"))

    def log_message(self, fmt, *args):
        pass


def main():
    """Điểm vào: đọc tham số, nạp mô hình (Clinic), đăng ký route mở rộng, mở server trên 127.0.0.1 và ::1 (IPv6)."""
    global CLINIC
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252 không in được tiếng Việt
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    print("Đang nạp mô hình Deep Learning, đồ thị tri thức và Q-table ...", flush=True)
    CLINIC = Clinic()  # nạp dl_model.npz, kg.json + corpus, rl_qtable.json (xem mas/orchestrator.py)
    load_extensions()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if socket.has_ipv6:  # "localhost" thường phân giải sang ::1 trước; không nghe thì trình duyệt chờ rất lâu
        class V6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        try:
            threading.Thread(target=V6Server(("::1", args.port), Handler).serve_forever, daemon=True).start()
        except OSError:
            pass
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
