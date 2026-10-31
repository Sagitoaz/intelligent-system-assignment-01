"""Agent luật an toàn (rule-based): cảnh báo khẩn và khuyến nghị tối thiểu, chạy trước mọi lời khuyên khác."""
from __future__ import annotations

from .data import SYMPTOMS, bmi, normalize

SYM = dict(SYMPTOMS)


def evaluate(profile: dict, risk_index: float | None = None, p_diabetes: float | None = None) -> list[dict]:
    """Trả về danh sách luật đã kích hoạt: id, mức (info|warn|danger), nội dung, căn cứ."""
    p = normalize(profile)
    g = p["fasting_glucose"]
    sym = p["symptoms"]
    out = []

    def add(rid, level, text, basis):
        out.append(dict(id=rid, level=level, text=text, basis=basis))

    classic = [s for s in ("thirst", "frequent_urination", "weight_loss", "blurred_vision") if s in sym]
    if g is not None and g >= 250:
        add("R1", "danger", f"Đường huyết {g:.0f} mg/dL rất cao. Hãy đi khám hoặc đến cơ sở y tế trong ngày, đặc biệt nếu khát nhiều, tiểu nhiều, buồn nôn hoặc thở nhanh.",
            "Ngưỡng cảnh báo của demo (≥ 250 mg/dL); đường huyết ngẫu nhiên ≥ 200 mg/dL kèm triệu chứng gợi ý tiểu đường (ADA/NIDDK)")
    elif g is not None and g >= 200 and classic:
        add("R2", "danger", f"Đường huyết {g:.0f} mg/dL kèm triệu chứng ({', '.join(SYM[s].lower() for s in classic)}). Cần xét nghiệm xác nhận sớm tại cơ sở y tế.",
            "Đường huyết ngẫu nhiên ≥ 200 mg/dL kèm triệu chứng (ADA/NIDDK)")
    elif g is not None and g >= 126:
        add("R3", "warn", f"Đường huyết đói {g:.0f} mg/dL đạt ngưỡng của tiểu đường (≥ 126 mg/dL). Cần lặp lại xét nghiệm (đường huyết đói, A1C) để xác nhận.",
            "Chuẩn ADA: đường huyết đói ≥ 126 mg/dL")
    elif g is not None and g >= 100:
        add("R4", "warn", f"Đường huyết đói {g:.0f} mg/dL thuộc vùng tiền tiểu đường (100–125 mg/dL). Nên thay đổi lối sống và xét nghiệm lại.",
            "Chuẩn ADA: đường huyết đói 100–125 mg/dL")
    elif g is not None and g < 70:
        add("R5", "danger", f"Đường huyết {g:.0f} mg/dL thấp. Nếu đang dùng thuốc hạ đường huyết hãy ăn ngay đường hấp thu nhanh và báo bác sĩ.",
            "NIDDK: đường huyết dưới 70 mg/dL là quá thấp ở người bệnh tiểu đường")
    if len(classic) >= 3 and (g is None or g < 200):
        add("R6", "warn", f"Có {len(classic)} triệu chứng điển hình ({', '.join(SYM[s].lower() for s in classic)}). Nên đi xét nghiệm đường huyết dù chưa đo được.",
            "NIDDK: người có triệu chứng tiểu đường nên được xét nghiệm")
    b = bmi(p)
    if b >= 30:
        add("R7", "warn", f"BMI {b:.1f} thuộc nhóm béo phì, yếu tố nguy cơ chính của tiểu đường típ 2.", "NIDDK: thừa cân, béo phì là yếu tố nguy cơ")
    if p["high_bp"] and p["high_chol"] and (p["heart_disease"] or p["stroke"]):
        add("R8", "warn", "Huyết áp cao, mỡ máu cao và bệnh tim/đột quỵ cùng tồn tại: nguy cơ biến chứng tim mạch cao, cần theo dõi cùng bác sĩ.", "NIDDK: tiểu đường làm tăng nguy cơ bệnh tim và đột quỵ")
    if p["age_years"] >= 45 and not p["chol_check"]:
        add("R9", "info", "Từ 45 tuổi trở lên nên tầm soát định kỳ đường huyết và mỡ máu.", "NIDDK: tầm soát định kỳ, kết quả bình thường nên làm lại sau mỗi 3 năm")
    if risk_index is not None and p_diabetes is not None and p_diabetes >= 0.5 and not any(r["level"] == "danger" for r in out):
        add("R10", "warn", f"Mô hình ước lượng khả năng tiểu đường {p_diabetes*100:.0f}%. Nên làm xét nghiệm xác nhận (A1C hoặc đường huyết đói).", "Quy tắc của demo: xác suất ≥ 50% thì khuyên xét nghiệm xác nhận")
    return out


def top_level(rules: list[dict]) -> str:
    return "danger" if any(r["level"] == "danger" for r in rules) else "warn" if any(r["level"] == "warn" for r in rules) else "info"
