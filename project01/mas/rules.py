"""Agent luật an toàn (rule-based): cảnh báo khẩn và khuyến nghị tối thiểu, chạy trước mọi lời khuyên khác.

=====================  BẢN ĐỒ FILE (thầy hỏi "phần luật / rule-based / cảnh báo an toàn ở đâu" thì mở file này)  =====================
Thuộc agent: "Hộ lý an toàn" (id="guardian", vai trò Luật rule-based). Luật luôn được ƯU TIÊN hơn dự đoán của mô hình học máy.

Cách hoạt động: không học gì cả; là một chuỗi if/elif cố định dựa trên ngưỡng xét nghiệm của ADA/NIDDK. Mỗi luật có
  id (R1..R10), level ("info" | "warn" | "danger"), text (lời khuyên tiếng Việt) và basis (căn cứ, nguồn ngưỡng).

Bảng luật:
  R1  đường huyết ≥ 250 mg/dL                                  → danger (đi khám trong ngày)
  R2  đường huyết ≥ 200 kèm ≥ 1 triệu chứng điển hình           → danger
  R3  đường huyết đói ≥ 126                                     → warn  (ngưỡng tiểu đường ADA, cần xét nghiệm lặp lại)
  R4  đường huyết đói 100–125                                   → warn  (vùng tiền tiểu đường)
  R5  đường huyết < 70                                          → danger (quá thấp, ăn đường nhanh)
  R6  ≥ 3 triệu chứng điển hình (và đường huyết < 200 / chưa đo) → warn
  R7  BMI ≥ 30                                                  → warn  (béo phì)
  R8  huyết áp cao + mỡ máu cao + bệnh tim/đột quỵ              → warn
  R9  ≥ 45 tuổi mà chưa đo mỡ máu                               → info
  R10 mô hình ước lượng P(tiểu đường) ≥ 50% và chưa có luật danger → warn (khuyên xét nghiệm xác nhận)
  (R1–R5 loại trừ nhau theo thứ tự elif: chỉ một luật về đường huyết được kích hoạt.)

Thành phần chính: evaluate() (chạy toàn bộ luật cho một hồ sơ), top_level() (mức nghiêm trọng cao nhất trong các luật đã kích hoạt).
Được gọi từ: mas/orchestrator.py (hội chẩn: evaluate rồi top_level để đặt mức cảnh báo), mas/critic.py (đối chiếu kết luận của MLP với kết quả luật).
=========================================================================================================================
"""
from __future__ import annotations

from .data import SYMPTOMS, bmi, normalize

SYM = dict(SYMPTOMS)  # id triệu chứng -> nhãn tiếng Việt, dùng để in tên triệu chứng trong lời khuyên


def evaluate(profile: dict, risk_index: float | None = None, p_diabetes: float | None = None) -> list[dict]:
    """Trả về danh sách luật đã kích hoạt: id, mức (info|warn|danger), nội dung, căn cứ.
    Đầu vào: hồ sơ thân thiện; risk_index và p_diabetes (xác suất tiểu đường từ MLP) là tùy chọn, chỉ dùng cho R10."""
    p = normalize(profile)
    g = p["fasting_glucose"]
    sym = p["symptoms"]
    out = []

    def add(rid, level, text, basis):
        out.append(dict(id=rid, level=level, text=text, basis=basis))

    # "triệu chứng điển hình" = 4 triệu chứng kinh điển của tiểu đường mà người dùng đã chọn
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
    """Mức nghiêm trọng cao nhất trong danh sách luật: "danger" > "warn" > "info" (không có luật nào cũng trả "info")."""
    return "danger" if any(r["level"] == "danger" for r in rules) else "warn" if any(r["level"] == "warn" for r in rules) else "info"
