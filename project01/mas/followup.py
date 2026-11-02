"""Hàng đợi bệnh nhân (agent phân loại) và theo dõi xu hướng 4 tuần kèm "RL điều chỉnh gợi ý".

DỮ LIỆU XU HƯỚNG LÀ MẪU MÔ PHỎNG: bộ sinh có seed cố định xuất phát từ hồ sơ (đường huyết đói và cân nặng hiện tại),
cộng một xu hướng nhẹ và nhiễu. Đó không phải số đo thật của bất kỳ ai.
"""
from __future__ import annotations

import random
import zlib

from . import rl, rules
from .data import PERSONAS, bmi

DAYS, WEEKS = 28, 4
SIM_NOTE = ("Dữ liệu mẫu minh họa: chuỗi 28 ngày này do chương trình tự sinh (có seed cố định) từ hồ sơ mẫu, "
            "không phải số đo thật của ai.")
LEVEL_COLOR = {"thấp": "#0d9f8f", "trung bình": "#e8a317", "cao": "#e8503a"}
ACT_VI = {a["id"]: a["label"].lower() for a in rl.ACTIONS}


# ------------------------------------------------------------------ hàng đợi
def triage(clinic, profiles: list | None = None) -> dict:
    items = profiles if profiles else [dict(id=p["id"], name=p["name"], story=p["story"], profile=p["profile"]) for p in PERSONAS]
    if not isinstance(items, list) or len(items) > 12:
        raise ValueError("Hàng đợi cần một danh sách tối đa 12 hồ sơ")
    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError("Mỗi hồ sơ phải là một đối tượng JSON")
        raw = it["profile"] if isinstance(it.get("profile"), dict) else it
        prof, dl, _ = clinic.patient(raw)
        rule_list = rules.evaluate(prof, dl["risk_index"], dl["risk"]["diabetes"])
        level = clinic.level(dl, rule_list)
        flags = [dict(id=r["id"], level=r["level"], text=r["text"]) for r in rule_list if r["level"] != "info"]
        danger = any(r["level"] == "danger" for r in flags)
        out.append(dict(id=str(it.get("id") or f"p{i + 1}"), name=str(it.get("name") or prof["name"]), story=it.get("story", ""), profile=raw,
                        risk=dl["risk"], risk_index=round(dl["risk_index"], 4), level=level, level_color=LEVEL_COLOR[level],
                        danger=danger, red_flags=flags, fasting_glucose=prof["fasting_glucose"], bmi=round(dl["bmi"], 1), age=prof["age_years"]))
    order = sorted(range(len(out)), key=lambda i: (not out[i]["danger"], -out[i]["risk_index"]))
    ranked = []
    for rank, i in enumerate(order, 1):
        o = out[i]
        if o["danger"]:
            top = next(f for f in o["red_flags"] if f["level"] == "danger")
            why = f"Xếp trước vì có cờ đỏ mức nguy hiểm ({top['id']}): {top['text']}"
        else:
            n_warn = sum(f["level"] == "warn" for f in o["red_flags"])
            why = (f"Không có cờ đỏ nguy hiểm; xếp theo chỉ số nguy cơ {o['risk_index'] * 100:.0f}% (nguy cơ {o['level']})"
                   + (f", có {n_warn} lưu ý cần theo dõi." if n_warn else ", không có lưu ý nào."))
        ranked.append(dict(o, rank=rank, reason=why))
    rule = ("Quy tắc xếp hàng của agent phân loại: bệnh nhân có cờ đỏ mức nguy hiểm được khám trước; "
            "những người còn lại xếp theo chỉ số nguy cơ từ cao xuống thấp. Đây chỉ là công cụ học tập, không phải phân loại y khoa thật.")
    return dict(queue=ranked, rule=rule)


# ------------------------------------------------------------------ dữ liệu mẫu mô phỏng
def sample_series(prof: dict, pat_base: float, seed: int = 2024) -> dict:
    """Sinh 28 ngày đường huyết đói và cân nặng; ngày cuối khớp (xấp xỉ) số liệu hiện tại của hồ sơ."""
    rng = random.Random(zlib.crc32(prof["name"].encode("utf-8")) ^ (int(seed) * 7919))
    g_now = prof["fasting_glucose"] or pat_base
    w_now = prof["weight_kg"]
    heavy = bmi(prof) >= 25
    inactive = not prof["phys_activity"]
    if inactive and (heavy or g_now >= 100):
        gs, ws = 0.45, 0.045   # ít vận động: xu hướng nhẹ tăng
    elif prof["phys_activity"] and not heavy:
        gs, ws = -0.05, 0.0
    elif prof["phys_activity"]:
        gs, ws = -0.2, -0.03
    else:
        gs, ws = 0.15, 0.015
    glucose, weight = [], []
    for d in range(DAYS):
        k = d - (DAYS - 1)
        glucose.append(round(max(55.0, min(450.0, g_now + gs * k + rng.gauss(0, max(3.0, 0.035 * g_now)))), 1))
        weight.append(round(max(30.0, w_now + ws * k + rng.gauss(0, 0.25)), 1))
    return dict(glucose=glucose, weight=weight)


def _clean(vals, name, lo, hi) -> list[float]:
    if not isinstance(vals, list) or not (8 <= len(vals) <= 120):
        raise ValueError(f"Chuỗi {name} cần từ 8 đến 120 giá trị")
    try:
        v = [float(x) for x in vals]
    except (TypeError, ValueError):
        raise ValueError(f"Chuỗi {name} phải là các số")
    if any(not (lo <= x <= hi) for x in v):
        raise ValueError(f"Chuỗi {name} có giá trị ngoài khoảng {lo:g}–{hi:g}")
    return v


def _slope(y: list[float]) -> float:
    n = len(y)
    mx, my = (n - 1) / 2, sum(y) / n
    den = sum((i - mx) ** 2 for i in range(n))
    return sum((i - mx) * (v - my) for i, v in enumerate(y)) / den if den else 0.0


def _chunks(v: list[float]) -> list[list[float]]:
    n = len(v)
    return [v[round(n * w / WEEKS):round(n * (w + 1) / WEEKS)] for w in range(WEEKS)]


def _mean(v):
    return sum(v) / len(v)


def _plan_map(plan: list[dict]) -> dict:
    return {p["hour"]: p["action"] for p in plan}


def _hour_txt(h: int) -> str:
    part = "sáng" if 5 <= h < 11 else "trưa" if h < 14 else "chiều" if h < 18 else "tối" if h < 23 else "đêm"
    return f"{h}h ({part})"


def _diff(prev: dict, cur: dict) -> list[dict]:
    a, b = _plan_map(prev["plan"]), _plan_map(cur["plan"])
    out = []
    for h in sorted(set(a) | set(b)):
        if a.get(h) == b.get(h):
            continue
        if h in a and h in b:
            out.append(dict(hour=h, kind="đổi", before=a[h], after=b[h], text=f"{_hour_txt(h)}: từ “{ACT_VI[a[h]]}” sang “{ACT_VI[b[h]]}”"))
        elif h in b:
            out.append(dict(hour=h, kind="thêm", before=None, after=b[h], text=f"{_hour_txt(h)}: thêm “{ACT_VI[b[h]]}”"))
        else:
            out.append(dict(hour=h, kind="bỏ", before=a[h], after=None, text=f"{_hour_txt(h)}: bỏ “{ACT_VI[a[h]]}”"))
    return out


def _modal_plan(pt: dict, coach, days: int = 15) -> list[dict]:
    """Kế hoạch ngày ổn định: với mỗi giờ lấy hành động RL chọn nhiều nhất qua `days` ngày mô phỏng (seed cố định), bỏ giờ nghỉ.
    Dùng nhiều ngày để một thay đổi nhỏ do nhiễu ngẫu nhiên không bị hiểu nhầm là gợi ý mới."""
    runs = [rl.run_day(coach, pt, 3 + d)["rows"] for d in range(days)]
    plan = []
    for i in range(rl.DAY_STEPS):
        cnt: dict = {}
        for r in runs:
            cnt[r[i]["action"]] = cnt.get(r[i]["action"], 0) + 1
        act = max(cnt, key=lambda a: (cnt[a], a != "rest"))
        if act != "rest" and cnt[act] * 2 > days:
            a = rl.ACTIONS[rl.AID[act]]
            plan.append(dict(hour=rl.clock(runs[0][i]["hour"]), action=act, label=a["label"], emoji=a["emoji"], share=round(cnt[act] / days, 2)))
    return plan


def _count(plan, ids):
    return sum(p["action"] in ids for p in plan)


def _reasons(prev: dict, cur: dict, diffs: list[dict]) -> list[str]:
    """Lý do chỉ dựa trên số liệu thật đã tính: đường huyết trung bình tuần, cân nặng và trạng thái bệnh nhân RL."""
    r = []
    dg = cur["mean_glucose"] - prev["mean_glucose"]
    dw = cur["mean_weight"] - prev["mean_weight"]
    verb = "tăng" if dg > 0.5 else "giảm" if dg < -0.5 else "gần như không đổi"
    r.append(f"Đường huyết đói trung bình {verb}: tuần trước {prev['mean_glucose']:.0f}, tuần này {cur['mean_glucose']:.0f} mg/dL ({dg:+.0f}).")
    r.append(f"Cân nặng trung bình đổi {dw:+.1f} kg (từ {prev['mean_weight']:.1f} sang {cur['mean_weight']:.1f} kg).")
    ps, cs = prev["rl_patient"], cur["rl_patient"]
    r.append(f"Trong mô phỏng của huấn luyện viên RL, đường huyết nền của bệnh nhân ảo đổi từ {ps['baseline_glucose']:.0f} sang {cs['baseline_glucose']:.0f} mg/dL "
             f"và độ nhạy insulin từ {ps['insulin_sensitivity']:.2f} sang {cs['insulin_sensitivity']:.2f}.")
    if diffs:
        w_prev, w_cur = _count(prev["plan"], {"walk", "workout"}), _count(cur["plan"], {"walk", "workout"})
        m_prev, m_cur = _count(prev["plan"], {"light_meal"}), _count(cur["plan"], {"light_meal"})
        if w_cur != w_prev:
            r.append(f"Số lần vận động trong ngày gợi ý đổi từ {w_prev} sang {w_cur}.")
        if m_cur != m_prev:
            r.append(f"Số bữa nhẹ trong ngày gợi ý đổi từ {m_prev} sang {m_cur}.")
    else:
        r.append("Với số liệu của hai tuần này, kế hoạch RL tính ra giống hệt nhau nên không có gợi ý nào đổi.")
    return r


def analyze(clinic, body: dict) -> dict:
    prof, dl, pat = clinic.patient(body.get("profile") or {})
    seed = int(body.get("seed", 2024))
    ser = body.get("series")
    simulated = not ser
    if simulated:
        ser = sample_series(prof, pat["baseline_glucose"], seed)
        g, w = ser["glucose"], ser["weight"]
    else:
        g, w = _clean(ser.get("glucose"), "đường huyết", 30, 600), _clean(ser.get("weight"), "cân nặng", 25, 250)
    gw, ww = _chunks(g), _chunks(w)
    h_m = prof["height_cm"] / 100.0
    weeks = []
    for i in range(WEEKS):
        mg, mw = _mean(gw[i]), _mean(ww[i])
        pt = rl.patient_from_risk(dl["risk_index"], mw / h_m ** 2, mg, prof["name"])
        st = rl.new_state(pt, random.Random(7))
        plan = _modal_plan(pt, clinic.coach)
        adv = rl.advice(st, clinic.coach)["coach"]
        weeks.append(dict(week=i + 1, mean_glucose=round(mg, 1), mean_weight=round(mw, 2), n=len(gw[i]), bmi=round(mw / h_m ** 2, 1),
                          pct_ge_126=round(100 * sum(x >= 126 for x in gw[i]) / len(gw[i])), rl_patient=pt, plan=plan,
                          first_hour=dict(action=adv["action"], label=ACT_VI[adv["action"]], why=adv["why"])))
    for i, wk in enumerate(weeks):
        if i == 0:
            wk["changes"], wk["reasons"] = [], ["Đây là tuần đầu tiên nên chưa có tuần trước để so sánh."]
        else:
            wk["changes"] = _diff(weeks[i - 1], wk)
            wk["reasons"] = _reasons(weeks[i - 1], wk, wk["changes"])
    n = len(g)
    sg, sw = _slope(g), _slope(w)
    stats = dict(glucose_slope_per_day=round(sg, 3), glucose_slope_per_week=round(sg * 7, 1), weight_slope_per_week=round(sw * 7, 2),
                 pct_ge_126=round(100 * sum(x >= 126 for x in g) / n), pct_ge_100=round(100 * sum(x >= 100 for x in g) / n),
                 weight_change_4w=round(weeks[-1]["mean_weight"] - weeks[0]["mean_weight"], 2),
                 glucose_change_4w=round(weeks[-1]["mean_glucose"] - weeks[0]["mean_glucose"], 1), days=n)
    trend = "tăng" if stats["glucose_slope_per_week"] > 2 else "giảm" if stats["glucose_slope_per_week"] < -2 else "gần như đi ngang"
    comment = (f"{prof['name']}: đường huyết đói {trend} (độ dốc {stats['glucose_slope_per_week']:+.1f} mg/dL mỗi tuần); trung bình tuần 1 là "
               f"{weeks[0]['mean_glucose']:.0f} và tuần {WEEKS} là {weeks[-1]['mean_glucose']:.0f} mg/dL. "
               f"{stats['pct_ge_126']}% số ngày từ 126 mg/dL trở lên, {stats['pct_ge_100']}% số ngày từ 100 mg/dL trở lên. "
               f"Cân nặng đổi {stats['weight_change_4w']:+.1f} kg sau {WEEKS} tuần.")
    if simulated:
        comment += " Đây là dữ liệu mẫu minh họa."
    n_changes = sum(len(wk["changes"]) for wk in weeks[1:])
    rl_note = "Gợi ý RL điều chỉnh qua các tuần: " + (f"có {n_changes} thay đổi trong kế hoạch ngày." if n_changes else "kế hoạch ngày không đổi qua 4 tuần.")
    return dict(name=prof["name"], simulated=simulated, note=SIM_NOTE if simulated else "Chuỗi do người dùng cung cấp.", seed=seed,
                series=dict(glucose=g, weight=w), weeks=weeks, stats=stats, comment=comment, rl_note=rl_note,
                thresholds=dict(normal=100, diabetes=126))


def routes(clinic, speaker):
    return {}, {"/api/queue": lambda b: triage(clinic, b.get("profiles")), "/api/followup": lambda b: analyze(clinic, b)}
