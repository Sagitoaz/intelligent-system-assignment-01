"""Agent Reinforcement Learning: huấn luyện viên lối sống cho bệnh nhân ảo.

MÔ PHỎNG ĐỒ CHƠI (không phải mô hình y khoa): mỗi bước là 1 giờ, một ngày gồm 24 bước từ 6 giờ sáng đến 5 giờ sáng hôm sau.
Q-learning dạng bảng học cách chọn hành động (ăn, đi bộ, tập...) để giữ đường huyết trong ngưỡng 70–180 mg/dL.
So sánh với huấn luyện viên theo luật và với người không có huấn luyện viên.
"""
from __future__ import annotations

import json
import math
import os
import random

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
START_HOUR, DAY_STEPS = 6, 24

ACTIONS = [
    dict(id="rest", label="Nghỉ ngơi", emoji="🛋️", desc="Không ăn, không vận động", carbs=0, act=0.0, fast=False),
    dict(id="sweet", label="Ăn nhẹ có đường", emoji="🍬", desc="Nước ngọt, kẹo hoặc chuối: đường lên nhanh, dùng khi đường huyết thấp", carbs=20, act=0.0, fast=True),
    dict(id="light_meal", label="Bữa nhẹ", emoji="🥗", desc="Salad, sữa chua, trái cây: ít tinh bột", carbs=30, act=0.0, fast=False),
    dict(id="normal_meal", label="Bữa chính", emoji="🍚", desc="Cơm, thịt cá và rau cân đối", carbs=60, act=0.0, fast=False),
    dict(id="heavy_meal", label="Bữa nhiều tinh bột", emoji="🍔", desc="Cơm lớn, đồ chiên, tráng miệng ngọt", carbs=100, act=0.0, fast=False),
    dict(id="walk", label="Đi bộ nhanh", emoji="🚶", desc="Đi bộ khoảng 30 phút", carbs=0, act=1.0, fast=False),
    dict(id="workout", label="Tập thể dục", emoji="🏃", desc="Chạy hoặc đạp xe khoảng 40 phút", carbs=0, act=2.0, fast=False),
]
AID = {a["id"]: i for i, a in enumerate(ACTIONS)}
MEALS = {"light_meal", "normal_meal", "heavy_meal"}
NIGHT_ALLOWED = ["rest", "sweet"]


def clock(hour: int) -> int:
    return hour % 24


def is_night(hour: int) -> bool:
    return clock(hour) >= 23 or clock(hour) < 6


def allowed(hour: int) -> list[str]:
    return NIGHT_ALLOWED if is_night(hour) else [a["id"] for a in ACTIONS]


def zone(g: float) -> str:
    if g < 55 or g > 250:
        return "danger"
    if g < 70:
        return "low"
    if g <= 180:
        return "ok"
    return "high"


def zone_reward(g: float) -> float:
    if g > 250:
        return -2.5
    if g > 180:
        return -0.7
    if g > 140:
        return 0.3
    if g >= 70:
        return 1.0
    if g >= 55:
        return -1.0
    return -3.0


# ------------------------------------------------------------------ bệnh nhân và môi trường
def patient_from_risk(risk_index: float, bmi: float, fasting: float | None, name: str = "Bệnh nhân") -> dict:
    s = max(0.3, min(1.15, 1.12 - 1.5 * risk_index - 0.012 * max(0.0, bmi - 25)))
    base = 90 + 100 * risk_index
    if fasting:
        base = 0.5 * fasting + 0.5 * base
    base = max(80.0, min(200.0, base))
    return dict(name=name, insulin_sensitivity=round(s, 3), baseline_glucose=round(base, 1), medication=10.0 if risk_index >= 0.35 else 0.0)


def random_patient(rng: random.Random) -> dict:
    s = rng.uniform(0.3, 1.15)
    base = max(80.0, min(200.0, 90 + 55 * (1.15 - s) / 0.85 * 1.2 + rng.gauss(0, 8)))
    return dict(name="Bệnh nhân ngẫu nhiên", insulin_sensitivity=round(s, 3), baseline_glucose=round(base, 1),
                medication=10.0 if (s < 0.6 and rng.random() < 0.6) else 0.0)


def new_state(patient: dict, rng: random.Random | None = None) -> dict:
    g = patient["baseline_glucose"] + (rng.gauss(0, 4) if rng else 0.0)
    return dict(hour=START_HOUR, glucose=round(g, 1), carbs=0.0, act=0.0, since_meal=2, exercise=0, patient=patient, log=[])


def step(state: dict, action: str, rng: random.Random) -> tuple[dict, float, dict]:
    """Một giờ mô phỏng. Trả về (trạng thái mới, phần thưởng, thông tin)."""
    p = state["patient"]
    s, base, med = p["insulin_sensitivity"], p["baseline_glucose"], p.get("medication", 0.0)
    act = ACTIONS[AID[action]]
    g, c, a = state["glucose"], state["carbs"], state["act"]
    c_total = c + act["carbs"]
    absorbed = c_total * (0.75 if act["fast"] else 0.45)
    c_after = c_total - absorbed
    rise = absorbed * 1.35 * (1.7 - s)
    a_total = a + act["act"]
    drop_act = a_total * 14 * (0.5 + 0.5 * s) * max(0.0, min(1.0, (g - 62) / 60))
    pull = (g - base) * 0.16 * s
    drop_med = med * max(0.0, min(1.0, (g - 55) / 60))
    g2 = g + rise - drop_act - pull - drop_med + rng.gauss(0, 3)
    g2 = max(35.0, min(420.0, g2))
    meal = action in MEALS
    since = 0 if meal else state["since_meal"] + 1
    reward = zone_reward(g2)
    night = is_night(state["hour"] + 1)
    if since >= 6 and not night:
        reward -= 0.6
    ex = state["exercise"] + (1 if act["act"] > 0 else 0)
    if act["act"] > 0 and state["exercise"] < 2:
        reward += 0.25
    new = dict(hour=state["hour"] + 1, glucose=round(g2, 1), carbs=round(c_after, 2), act=round(a_total * 0.5, 3), since_meal=since,
               exercise=ex, patient=p, log=state["log"] + [dict(hour=state["hour"], action=action, glucose=round(g2, 1))])
    return new, reward, dict(zone=zone(g2), delta=round(g2 - g, 1), rise=round(rise, 1), drop=round(drop_act + drop_med, 1))


def done(state: dict) -> bool:
    return state["hour"] >= START_HOUR + DAY_STEPS


def encode(state: dict) -> tuple:
    g, c, a, sm, p = state["glucose"], state["carbs"], state["act"], state["since_meal"], state["patient"]
    gb = 0 if g < 70 else 1 if g < 100 else 2 if g < 140 else 3 if g < 180 else 4 if g < 250 else 5
    cb = 0 if c < 5 else 1 if c < 40 else 2
    ab = 0 if a < 0.3 else 1
    hb = 0 if sm <= 2 else 1 if sm <= 4 else 2
    s = p["insulin_sensitivity"]
    sb = 0 if s < 0.55 else 1 if s < 0.85 else 2
    block = min(5, max(0, (state["hour"] - START_HOUR) // 4))
    return (block, gb, cb, ab, hb, sb, 1 if p.get("medication", 0) > 0 else 0)


# ------------------------------------------------------------------ các huấn luyện viên
def rule_coach(state: dict) -> tuple[str, str]:
    g, h, sm, a = state["glucose"], clock(state["hour"]), state["since_meal"], state["act"]
    if is_night(state["hour"]):
        return ("sweet", "Đường huyết thấp về đêm: ăn nhẹ có đường rồi ngủ tiếp.") if g < 80 else ("rest", "Giờ ngủ: nghỉ ngơi, không ăn thêm.")
    if g < 75:
        return "sweet", "Luật 1: đường huyết dưới 75 mg/dL thì ăn nhẹ có đường ngay."
    if sm >= 5:
        return ("light_meal", "Luật 2: đã lâu chưa ăn nhưng đường huyết đang cao, chọn bữa nhẹ.") if g > 150 else ("normal_meal", "Luật 2: đã hơn 5 giờ chưa ăn, ăn bữa chính.")
    if h in (7, 12, 19) and sm >= 3:
        return ("light_meal", "Luật 3: đến giờ ăn nhưng đường huyết cao, ăn nhẹ.") if g > 150 else ("normal_meal", "Luật 3: đến giờ ăn chính (7, 12, 19 giờ).")
    if g > 220 and a < 0.3:
        return "workout", "Luật 4: đường huyết trên 220 mg/dL, vận động mạnh để hạ."
    if g > 160 and a < 0.3:
        return "walk", "Luật 4: đường huyết trên 160 mg/dL, đi bộ nhanh để hạ."
    return "rest", "Không có luật nào kích hoạt: nghỉ ngơi."


def no_coach(state: dict) -> tuple[str, str]:
    h = clock(state["hour"])
    if is_night(state["hour"]):
        return "rest", "Đi ngủ."
    if h == 7 or h == 12:
        return "normal_meal", "Ăn bữa chính như thường lệ."
    if h == 19:
        return "heavy_meal", "Bữa tối nhiều tinh bột, ăn xong ngồi xem tivi."
    if h in (10, 15, 21):
        return "sweet", "Ăn vặt đồ ngọt."
    return "rest", "Ngồi làm việc, ít vận động."


class QCoach:
    def __init__(self, q: dict | None = None):
        self.q = q if q is not None else {}

    @classmethod
    def load(cls):
        with open(os.path.join(ART, "rl_qtable.json"), encoding="utf-8") as f:
            raw = json.load(f)
        return cls({tuple(int(x) for x in k.split(",")): v for k, v in raw["q"].items()})

    def values(self, key):
        return self.q.setdefault(key, [0.0] * len(ACTIONS))

    def act(self, state: dict, epsilon: float = 0.0, rng: random.Random | None = None) -> str:
        legal = allowed(state["hour"])
        if rng is not None and rng.random() < epsilon:
            return rng.choice(legal)
        v = self.values(encode(state))
        best = max(legal, key=lambda a: (v[AID[a]], -AID[a]))
        return best

    def advise(self, state: dict) -> tuple[str, str]:
        a = self.act(state)
        g = state["glucose"]
        why = {"sweet": "Đường huyết đang thấp hoặc sắp thấp, cần đường hấp thu nhanh.",
               "light_meal": "Cần ăn nhưng chọn ít tinh bột để đường huyết không vọt lên.",
               "normal_meal": "Đến lúc ăn bữa chính; khẩu phần vừa phải giúp đường huyết ổn định.",
               "heavy_meal": "Bữa lớn: chỉ nên chọn khi đường huyết thấp và còn nhiều giờ để vận động sau đó.",
               "walk": "Đi bộ nhẹ giúp kéo đường huyết xuống và đã học được là lựa chọn an toàn.",
               "workout": "Đường huyết cao nên vận động mạnh hơn để hạ nhanh.",
               "rest": "Đường huyết đang ổn, nghỉ ngơi và chờ giờ ăn tiếp theo."}[a]
        return a, f"{why} (G = {g:.0f} mg/dL)"


# ------------------------------------------------------------------ huấn luyện và đánh giá
def train(episodes: int = 60000, seed: int = 11, alpha: float = 0.15, gamma: float = 0.97, log_every: int = 2000):
    rng = random.Random(seed)
    coach = QCoach()
    curve = []
    window = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - 0.95 * ep / (0.7 * episodes))
        st = new_state(random_patient(rng), rng)
        total = 0.0
        while not done(st):
            key = encode(st)
            a = coach.act(st, eps, rng)
            nst, r, _ = step(st, a, rng)
            fin = done(nst)
            target = r if fin else r + gamma * max(coach.values(encode(nst))[AID[x]] for x in allowed(nst["hour"]))
            v = coach.values(key)
            v[AID[a]] += alpha * (target - v[AID[a]])
            st, total = nst, total + r
        window.append(total)
        if (ep + 1) % log_every == 0:
            curve.append(dict(episode=ep + 1, mean_return=round(sum(window) / len(window), 3), epsilon=round(eps, 3)))
            window = []
    return coach, curve


def run_day(policy, patient: dict, seed: int) -> dict:
    rng = random.Random(seed)
    st = new_state(patient, rng)
    total, rows = 0.0, []
    while not done(st):
        a = policy(st)[0] if not isinstance(policy, QCoach) else policy.act(st)
        nst, r, info = step(st, a, rng)
        total += r
        rows.append(dict(hour=st["hour"], action=a, glucose=nst["glucose"], zone=info["zone"], reward=round(r, 2)))
        st = nst
    gl = [r["glucose"] for r in rows]
    return dict(rows=rows, total=total, tir=sum(70 <= g <= 180 for g in gl) / len(gl), mean=sum(gl) / len(gl),
                hypo=sum(g < 70 for g in gl), hyper=sum(g > 180 for g in gl), danger=sum((g < 55 or g > 250) for g in gl))


def policies(qcoach: QCoach) -> dict:
    return {"rl": ("Huấn luyện viên RL", qcoach), "rule": ("Huấn luyện viên theo luật", rule_coach), "none": ("Không có huấn luyện viên", no_coach)}


def compare(patient: dict, qcoach: QCoach, days: int = 30, seed: int = 7) -> dict:
    out = []
    for pid, (name, pol) in policies(qcoach).items():
        runs = [run_day(pol, patient, seed * 1000 + d) for d in range(days)]
        n = len(runs)
        out.append(dict(id=pid, name=name, time_in_range=round(sum(r["tir"] for r in runs) / n, 3), mean_glucose=round(sum(r["mean"] for r in runs) / n, 1),
                        hypo_hours=round(sum(r["hypo"] for r in runs) / n, 2), hyper_hours=round(sum(r["hyper"] for r in runs) / n, 2),
                        danger_hours=round(sum(r["danger"] for r in runs) / n, 2), mean_reward=round(sum(r["total"] for r in runs) / n, 2),
                        curve=[round(sum(r["rows"][i]["glucose"] for r in runs) / n, 1) for i in range(DAY_STEPS)]))
    cols = ["Huấn luyện viên", "Thời gian trong ngưỡng 70–180", "Đường huyết trung bình (mg/dL)", "Giờ hạ đường huyết/ngày", "Giờ tăng cao/ngày", "Điểm thưởng/ngày"]
    rows = [[o["name"], f"{o['time_in_range']*100:.0f}%", f"{o['mean_glucose']:.0f}", f"{o['hypo_hours']:.1f}", f"{o['hyper_hours']:.1f}", f"{o['mean_reward']:.1f}"] for o in out]
    return dict(coaches=out, table=dict(columns=cols, rows=rows), days=days)


def advice(state: dict, qcoach: QCoach) -> dict:
    a, why = qcoach.advise(state)
    ra, rwhy = rule_coach(state)
    return dict(coach=dict(action=a, why=why), rule=dict(action=ra, why=rwhy))


def plan_for(patient: dict, qcoach: QCoach, seed: int = 3) -> list[dict]:
    """Kế hoạch một ngày do RL gợi ý (rút gọn: bỏ các giờ nghỉ liên tiếp)."""
    day = run_day(qcoach, patient, seed)
    plan = []
    for r in day["rows"]:
        if r["action"] != "rest":
            a = ACTIONS[AID[r["action"]]]
            plan.append(dict(hour=clock(r["hour"]), action=r["action"], label=a["label"], emoji=a["emoji"], glucose_after=r["glucose"]))
    return plan
