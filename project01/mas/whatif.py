"""Giải thích kết quả: "đổi một chỉ số thì nguy cơ đổi bao nhiêu" bằng cách chạy lại chính mô hình MLP với hồ sơ đã sửa.

Mô hình không dùng đường huyết làm đầu vào, nên các chỉ số có thể đổi ở đây là cân nặng, hút thuốc, vận động, trái cây, rau, rượu bia,
huyết áp, mỡ máu. Kết quả là thay đổi theo mô hình trên dữ liệu khảo sát, không phải lời hứa về hiệu quả điều trị với từng người.
"""
from __future__ import annotations

from .data import bmi, normalize

KNOBS = {"weight_kg": float, "smoker": bool, "phys_activity": bool, "fruits_daily": bool, "veggies_daily": bool, "heavy_alcohol": bool,
         "high_bp": bool, "high_chol": bool}
LABEL = {"weight_kg": "cân nặng", "smoker": "hút thuốc", "phys_activity": "vận động", "fruits_daily": "ăn trái cây", "veggies_daily": "ăn rau",
         "heavy_alcohol": "uống nhiều rượu bia", "high_bp": "huyết áp cao", "high_chol": "mỡ máu cao"}


def _summary(pred: dict) -> dict:
    return dict(risk=pred["risk"], risk_index=round(pred["risk_index"], 4), label=pred["label"], bmi=round(pred["bmi"], 1))


def what_if(clinic, profile: dict, changes: dict) -> dict:
    prof = normalize(profile)
    new = dict(prof)
    for k, v in (changes or {}).items():
        if k not in KNOBS:
            raise ValueError(f"Không đổi được chỉ số '{k}'")
        new[k] = float(v) if KNOBS[k] is float else bool(v)
    new = normalize(new)
    a, b = clinic.dl.predict(prof), clinic.dl.predict(new)
    delta = b["risk_index"] - a["risk_index"]
    moved = [LABEL[k] for k in (changes or {}) if prof[k] != new[k]]
    if not moved:
        text = "Chưa đổi chỉ số nào nên nguy cơ giữ nguyên."
    else:
        word = "giảm" if delta < -0.0005 else "tăng" if delta > 0.0005 else "gần như không đổi"
        text = f"Nếu thay đổi {', '.join(moved)}, chỉ số nguy cơ của mô hình {word}" + (f" khoảng {abs(delta) * 100:.1f} điểm phần trăm.".replace(".", ",") if word != "gần như không đổi" else ".")
    return dict(base=_summary(a), new=_summary(b), delta_index=round(delta, 4), delta_diabetes=round(b["risk"]["diabetes"] - a["risk"]["diabetes"], 4), changed=moved, text=text)


def bmi_curve(clinic, profile: dict) -> dict:
    """Chỉ số nguy cơ khi BMI chạy từ 18 đến 40 (các yếu tố khác giữ nguyên): thấy rõ nguy cơ tăng nhanh ở đâu."""
    prof = normalize(profile)
    h = prof["height_cm"] / 100
    pts = []
    for b in range(18, 41):
        w = min(180.0, max(35.0, b * h * h))
        p = clinic.dl.predict(dict(prof, weight_kg=w))
        pts.append(dict(bmi=round(w / h / h, 1), risk_index=round(p["risk_index"], 4), diabetes=round(p["risk"]["diabetes"], 4)))
    return dict(points=pts, current=dict(bmi=round(bmi(prof), 1), risk_index=round(clinic.dl.predict(prof)["risk_index"], 4)))


def routes(clinic, speaker):
    return {}, {"/api/whatif": lambda b: what_if(clinic, b.get("profile") or {}, b.get("changes") or {}),
                "/api/whatif/curve": lambda b: bmi_curve(clinic, b.get("profile") or {})}
