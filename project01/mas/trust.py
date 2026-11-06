"""Độ tin cậy của mô hình: nạp artifacts/trust_metrics.json (do project01/train/eval_trust.py tạo) và phục vụ qua API.

=====================  BẢN ĐỒ FILE (thầy hỏi "bảng độ tin cậy / công bằng (fairness) ở đâu" thì mở file này)  =====================
Không tính toán gì: file này chỉ ĐỌC kết quả đã tính sẵn và đưa lên API. Phần tính thật nằm ở train/eval_trust.py
(đánh giá độ tin cậy của MLP trên đúng tập kiểm thử 50.736 dòng, cùng phép chia train/test với train_dl.py; file kết quả có
khóa "groups" = số liệu theo từng nhóm và "fairness" = chỉ số công bằng giữa các nhóm).

Thành phần chính:
  - load()      : đọc trust_metrics.json một lần rồi giữ trong bộ nhớ (_CACHE).
  - _trust()    : toàn bộ số liệu độ tin cậy; nếu chưa có file thì trả {"error": ...} thay vì làm sập server.
  - _fairness() : chỉ phần "groups" và "fairness" (phục vụ bảng công bằng trên giao diện).
  - routes()    : đăng ký 2 endpoint GET: /api/trust và /api/trust/fairness (server.py ghép vào bảng route chung).

Dữ liệu đọc: artifacts/trust_metrics.json. Được gọi từ: server.py (qua routes()).
===============================================================================================================================
"""
from __future__ import annotations

import json
import os

PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "trust_metrics.json")
_CACHE: dict = {}  # bộ nhớ đệm: file chỉ đọc đĩa một lần


def load() -> dict:
    """Đọc (lần đầu) và trả về nội dung trust_metrics.json; ném FileNotFoundError nếu chưa chạy eval_trust.py."""
    if not _CACHE:
        with open(PATH, encoding="utf-8") as f:
            _CACHE.update(json.load(f))
    return _CACHE


def _trust() -> dict:
    """Handler của /api/trust: số liệu độ tin cậy, hoặc dict báo lỗi thân thiện nếu thiếu file."""
    try:
        return load()
    except FileNotFoundError:
        return dict(error="Chưa có trust_metrics.json. Hãy chạy project01/train/eval_trust.py.")


def _fairness() -> dict:
    """Handler của /api/trust/fairness: chỉ lấy các nhóm (groups) và chỉ số công bằng (fairness)."""
    d = _trust()
    if "error" in d:
        return d
    return dict(groups=d["groups"], fairness=d["fairness"])


def routes(clinic, speaker):
    """Trả về (route GET, route POST) theo quy ước của server.py; module này chỉ có route GET. clinic/speaker không dùng."""
    return {"/api/trust": _trust, "/api/trust/fairness": _fairness}, {}
