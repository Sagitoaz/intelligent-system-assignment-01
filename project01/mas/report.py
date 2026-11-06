"""Thư ký báo cáo: tự sinh hình (matplotlib) và bảng cho mỗi lượt hội chẩn.

================================ BẢN ĐỒ FILE (đọc phần này trước) ================================
Agent "Thư ký báo cáo" (reporter) của phòng khám: biến kết quả của các agent khác thành 3 HÌNH và 7 BẢNG hiển thị ở tab kết quả.
Không có endpoint riêng: build_report() được Clinic.consult() (orchestrator.py) gọi cuối mỗi lượt hội chẩn, kết quả nằm trong khóa "report"
của phản hồi POST /api/consult. File pdf_report.py (POST /api/report/pdf) nhận lại đúng cấu trúc này để dựng PDF.
Hình được vẽ ngay trên máy chủ bằng matplotlib (backend "Agg", không cần màn hình) rồi mã hóa base64 PNG nhúng thẳng vào JSON.
Dữ liệu: không đọc/ghi file; dùng kết quả MLP (dl), luật (rule_list), kế hoạch RL (plan, cmp), đoạn tài liệu (passages) và metrics huấn luyện.
Các hàm chính:
    b64(fig)            : đổi hình matplotlib thành chuỗi base64 PNG (rồi đóng hình để khỏi rò bộ nhớ)
    fig_risk()          : hình 1 — xác suất 3 lớp (không bệnh / tiền tiểu đường / tiểu đường) so với tỉ lệ trong dữ liệu BRFSS
    fig_factors()       : hình 2 — yếu tố đẩy nguy cơ lên/xuống (drivers) + what-if (hành động giúp giảm nguy cơ)
    fig_day()           : hình 3 — đường huyết một ngày mô phỏng của 3 kiểu huấn luyện viên (RL, luật, không có) so với ngưỡng an toàn 70–180
    build_report()      : ghép 7 bảng (hồ sơ, nguy cơ, what-if, luật, kế hoạch RL, so sánh 3 coach, nguồn trích dẫn) + 3 hình
Thầy hỏi "biểu đồ sinh ra ở đâu?" -> fig_risk / fig_factors / fig_day (file này); "bảng ở đâu?" -> danh sách tables trong build_report().
================================================================================================
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import FEATURES, SYMPTOMS, bmi  # noqa: E402
from .rl import ACTIONS, AID, START_HOUR  # noqa: E402

# Bảng màu thống nhất cho mọi hình: INK chữ, MUTE chữ phụ, GRID đường lưới; C = màu theo lớp bệnh và theo loại huấn luyện viên.
INK, MUTE, GRID = "#101828", "#3d4c63", "#dce8f3"
C = dict(none="#0d9f8f", pre="#e8a317", diabetes="#e8503a", rl="#f0603a", rule="#7060d8", nocoach="#7f95ad", sky="#2589e0")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#8fb0cf", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False})
# Tên lớp tiếng Việt hiển thị trên hình và bảng; khóa trùng với đầu ra của MLP (none/pre/diabetes).
CLASS_VI = {"none": "Không tiểu đường", "pre": "Tiền tiểu đường", "diabetes": "Tiểu đường"}


def b64(fig) -> str:
    """Lưu hình matplotlib thành PNG (130 dpi, nền trắng) trong bộ nhớ, đóng hình, trả về chuỗi base64 để nhúng vào JSON/PDF."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_risk(dl: dict, metrics: dict) -> str:
    """Hình 1: thanh ngang xác suất MLP gán cho ba lớp; vạch đứt đánh dấu tỉ lệ thực của từng lớp trong BRFSS (để người xem so mình với dân số)."""
    counts = metrics["class_counts"]
    total = sum(counts.values())
    pop = {k: counts[k] / total for k in ("none", "pre", "diabetes")}
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    keys = ["none", "pre", "diabetes"]
    vals = [dl["risk"][k] * 100 for k in keys]
    bars = ax.barh([CLASS_VI[k] for k in keys][::-1], vals[::-1], color=[C[k] for k in keys][::-1], height=0.55)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontweight="bold", color=INK)
    for k, y in zip(keys[::-1], range(3)):
        ax.plot([pop[k] * 100] * 2, [y - 0.34, y + 0.34], color=INK, lw=2, ls=(0, (3, 2)))
    ax.plot([], [], color=INK, lw=2, ls=(0, (3, 2)), label="Tỉ lệ trong dữ liệu BRFSS")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Xác suất theo mô hình (%)")
    ax.set_title("Nguy cơ của bạn theo mô hình Deep Learning", fontweight="bold", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    return b64(fig)


def fig_factors(dl: dict) -> str:
    """Hình 2 (hai ô). Trái: effect của từng yếu tố = chênh lệch nguy cơ khi thay yếu tố bằng giá trị trung bình dân số (đỏ = đẩy lên, xanh = kéo xuống).
    Phải: 4 hành động what-if giúp giảm nhiều nhất (gain); nếu không còn gì cải thiện thì hiện thông báo lối sống đã gần lý tưởng."""
    drivers = sorted(dl["drivers"], key=lambda d: d["effect"])
    wi = dl["what_if"][:4]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    ax.barh([d["vi"] for d in drivers], [d["effect"] * 100 for d in drivers], color=[C["diabetes"] if d["effect"] > 0 else C["none"] for d in drivers], height=0.55)
    ax.axvline(0, color=INK, lw=1)
    ax.set_title("Yếu tố nào đẩy nguy cơ lên/xuống?", fontweight="bold", loc="left", fontsize=11)
    ax.set_xlabel("Điểm phần trăm chỉ số nguy cơ\n(so với giá trị trung bình dân số)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax = axes[1]
    if wi:
        ax.barh([w["action"] for w in wi][::-1], [w["gain"] * 100 for w in wi][::-1], color=C["sky"], height=0.55)
        ax.set_xlabel("Điểm phần trăm nguy cơ có thể giảm")
    else:
        ax.text(0.5, 0.5, "Lối sống của bạn đã gần mức lý tưởng\ntheo các yếu tố mô hình xét", ha="center", va="center", color=MUTE)
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Nếu thay đổi được thì sao? (what-if)", fontweight="bold", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return b64(fig)


def fig_day(cmp: dict) -> str:
    """Hình 3: đường huyết trung bình theo giờ trong 24 giờ mô phỏng (từ START_HOUR) của ba kiểu huấn luyện viên; dải xanh là ngưỡng an toàn 70–180 mg/dL.
    Dữ liệu cmp lấy từ rl.compare() (trung bình 20 ngày). Đây là mô phỏng đồ chơi, không phải y khoa thật."""
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    hours = list(range(START_HOUR, START_HOUR + 24))
    ax.axhspan(70, 180, color="#d3f3ee", alpha=0.7, label="Ngưỡng an toàn 70–180")
    col = {"rl": C["rl"], "rule": C["rule"], "none": C["nocoach"]}
    for c in cmp["coaches"]:
        ax.plot(hours, c["curve"], color=col[c["id"]], lw=2.6 if c["id"] == "rl" else 1.8, label=f"{c['name']} ({c['time_in_range']*100:.0f}% trong ngưỡng)")
    ax.set_xticks(hours[::3])
    ax.set_xticklabels([f"{h % 24}h" for h in hours[::3]])
    ax.set_ylabel("Đường huyết trung bình (mg/dL)")
    ax.set_title("Một ngày mô phỏng: ba kiểu huấn luyện (trung bình 20 ngày)", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    return b64(fig)


VALUE_FMT = {"HighBP": "Có/Không", "HighChol": "Có/Không"}


def build_report(prof: dict, dl: dict, rule_list: list[dict], plan: list[dict], cmp: dict, passages: list[dict], pop_index: float, metrics: dict) -> dict:
    """Dựng báo cáo. Đầu vào: hồ sơ chuẩn hóa, kết quả MLP, danh sách luật kích hoạt, kế hoạch RL, so sánh coach, đoạn trích NIDDK, pop_index (chỉ số nguy cơ dân số), metrics huấn luyện.
    Đầu ra: dict(figures=[{id, title, caption, png_base64}], tables=[{id, title, columns, rows}]). Mỗi bảng có một "id" cố định để giao diện/PDF tham chiếu."""
    yesno = lambda v: "Có" if v else "Không"
    # Cột "Trung bình BRFSS" lấy từ feature_stats lúc huấn luyện (artifacts/dl_metrics.json) để so hồ sơ với dân số khảo sát 253.680 người.
    stats = metrics["feature_stats"]
    profile_rows = [
        ["Tuổi", f"{prof['age_years']}", "—"], ["Giới tính", "Nam" if prof["sex"] == "male" else "Nữ", f"{stats['Sex']['mean']*100:.0f}% nam"],
        ["BMI", f"{dl['bmi']:.1f}", f"{stats['BMI']['mean']:.1f}"], ["Huyết áp cao", yesno(prof["high_bp"]), f"{stats['HighBP']['mean']*100:.0f}% có"],
        ["Mỡ máu cao", yesno(prof["high_chol"]), f"{stats['HighChol']['mean']*100:.0f}% có"], ["Hút thuốc", yesno(prof["smoker"]), f"{stats['Smoker']['mean']*100:.0f}% có"],
        ["Có vận động", yesno(prof["phys_activity"]), f"{stats['PhysActivity']['mean']*100:.0f}% có"], ["Tự đánh giá sức khỏe (1 tốt – 5 kém)", f"{prof['gen_health']}", f"{stats['GenHlth']['mean']:.1f}"],
        ["Đường huyết đói", f"{prof['fasting_glucose']:.0f} mg/dL" if prof["fasting_glucose"] else "Chưa đo", "—"],
        ["Triệu chứng", ", ".join(dict(SYMPTOMS)[s] for s in prof["symptoms"]) or "Không", "—"],
    ]
    # 7 bảng: profile, risk, whatif, rules, plan, coaches, sources (chú ý: bảng rỗng được thay bằng một dòng giải thích để không để trống).
    tables = [
        dict(id="profile", title="Hồ sơ của bạn so với dân số trong dữ liệu", columns=["Chỉ số", "Của bạn", "Trung bình BRFSS"], rows=profile_rows),
        dict(id="risk", title="Nguy cơ theo mô hình", columns=["Nhóm", "Xác suất", "Tỉ lệ trong BRFSS"],
             rows=[[CLASS_VI[k], f"{dl['risk'][k]*100:.1f}%", f"{metrics['class_counts'][k] / metrics['rows'] * 100:.1f}%"] for k in ("none", "pre", "diabetes")]),
        dict(id="whatif", title="Thay đổi nào giúp giảm nguy cơ nhiều nhất", columns=["Hành động", "Giảm chỉ số nguy cơ"],
             rows=[[w["action"].capitalize(), f"{w['gain']*100:.1f} điểm %"] for w in dl["what_if"]] or [["Không có yếu tố lối sống nào còn cải thiện được", "—"]]),
        dict(id="rules", title="Cảnh báo của Hộ lý an toàn", columns=["Luật", "Mức", "Nội dung"],
             rows=[[r["id"], {"danger": "Khẩn", "warn": "Lưu ý", "info": "Nhắc"}[r["level"]], r["text"]] for r in rule_list] or [["—", "—", "Không có luật nào được kích hoạt"]]),
        dict(id="plan", title="Một ngày gợi ý của Huấn luyện viên RL", columns=["Giờ", "Hành động", "Đường huyết sau đó (mg/dL)"],
             rows=[[f"{p['hour']}h", f"{p['emoji']} {p['label']}", f"{p['glucose_after']:.0f}"] for p in plan] or [["—", "Duy trì thói quen hiện tại", "—"]]),
        dict(id="coaches", title="So sánh ba kiểu huấn luyện trong mô phỏng", columns=cmp["table"]["columns"], rows=cmp["table"]["rows"]),
        dict(id="sources", title="Nguồn trích dẫn (NIDDK)", columns=["Mã đoạn", "Nguồn"], rows=[[p["id"], p["url"]] for p in passages]),
    ]
    # 3 hình + chú thích tiếng Việt giải thích cách đọc (caption) để người không chuyên cũng hiểu.
    figures = [
        dict(id="risk_bars", title="Nguy cơ theo mô hình", caption="Ba cột là xác suất mô hình gán cho bạn; vạch đứt là tỉ lệ thực trong dữ liệu BRFSS 2015.", png_base64=fig_risk(dl, metrics)),
        dict(id="factors", title="Yếu tố tác động và what-if", caption="Trái: thay từng yếu tố bằng giá trị trung bình dân số để xem nó đẩy nguy cơ lên hay xuống. Phải: nguy cơ giảm được nếu cải thiện lối sống.", png_base64=fig_factors(dl)),
        dict(id="day", title="Một ngày mô phỏng", caption="Đường huyết trung bình theo giờ khi có huấn luyện viên RL, huấn luyện viên theo luật và khi không có ai (mô phỏng đồ chơi, không phải y khoa thật).", png_base64=fig_day(cmp)),
    ]
    return dict(figures=figures, tables=tables)
