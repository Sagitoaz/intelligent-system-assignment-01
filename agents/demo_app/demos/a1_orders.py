import csv
import os
import re

import shared

META = dict(agent="agent01", id="a1_orders", title="Trợ lý đơn hàng", js="a1_orders.js",
            description="Agent đọc yêu cầu tiếng Việt, tra 60.000 đơn hàng trong CSV và quyết định hoàn tiền, hủy hay tra cứu theo chính sách.")
_orders = {}
INTENTS = [("refund", r"hoàn tiền|trả lại tiền"), ("cancel", r"hủy|huỷ|không muốn mua"),
           ("complaint", r"khiếu nại|lỗi|thiếu hàng"), ("status", r"đang ở đâu|trạng thái|bao giờ"),
           ("policy", r"chính sách|bảo hành|phí vận chuyển")]


def _load():
    if not _orders:
        with open(os.path.join(shared.DATA_DIRS["agent01"], "orders_large.csv"), encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                _orders[r["id"]] = r
    return _orders


def handle(p):
    text, owner = p.get("text", ""), p.get("owner", "").strip()
    low = text.lower()
    intent = next((name for name, pat in INTENTS if re.search(pat, low)), "unknown")
    m = re.search(r"DH\d{6}", text.upper())
    steps = [f"Nhận diện ý định: {intent}"]
    if intent == "policy":
        return dict(intent=intent, steps=steps,
                    answer="Đổi trả trong 30 ngày với đơn đã giao; hủy được khi đơn chưa giao.")
    if intent == "unknown" or not m:
        return dict(intent=intent, steps=steps + ["Thiếu ý định hoặc mã đơn → hỏi lại người dùng"],
                    answer="Mình chưa rõ yêu cầu. Bạn cho mình mã đơn (ví dụ DH100123) và việc cần làm nhé.")
    order = _load().get(m.group())
    if not order:
        return dict(intent=intent, steps=steps + ["Không tìm thấy đơn"], answer=f"Không có đơn {m.group()}.")
    steps.append(f"Tra đơn {order['id']}: trạng thái {order['status']}, chủ đơn {order['owner']}, {order['age_days']} ngày")
    if owner and owner != order["owner"]:
        return dict(intent=intent, order=order, steps=steps + ["Chủ đơn khác người đăng nhập → từ chối"],
                    answer="Bạn không có quyền với đơn này.")
    ok = {"refund": order["status"] == "delivered" and int(order["age_days"]) <= 30,
          "cancel": order["status"] in ("processing", "shipping"),
          "complaint": True, "status": True}[intent]
    steps.append("Kiểm tra chính sách: " + ("được phép" if ok else "không đủ điều kiện"))
    msg = {"refund": "Đã tạo yêu cầu hoàn tiền." if ok else "Đơn chưa giao hoặc quá 30 ngày nên không hoàn tiền được.",
           "cancel": "Đã hủy đơn." if ok else "Đơn đã giao hoặc đã hủy, không thể hủy.",
           "complaint": "Đã ghi nhận khiếu nại, nhân viên sẽ liên hệ.",
           "status": f"Đơn {order['id']} đang ở trạng thái {order['status']}."}[intent]
    return dict(intent=intent, order=order, steps=steps, answer=msg)
