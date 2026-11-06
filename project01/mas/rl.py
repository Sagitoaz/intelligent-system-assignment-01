"""Agent Reinforcement Learning: huấn luyện viên lối sống cho bệnh nhân ảo.

MÔ PHỎNG ĐỒ CHƠI (không phải mô hình y khoa): mỗi bước là 1 giờ, một ngày gồm 24 bước từ 6 giờ sáng đến 5 giờ sáng hôm sau.
Q-learning dạng bảng học cách chọn hành động (ăn, đi bộ, tập...) để giữ đường huyết trong ngưỡng 70–180 mg/dL.
So sánh với huấn luyện viên theo luật và với người không có huấn luyện viên.

=====================  BẢN ĐỒ FILE (thầy hỏi "phần Reinforcement Learning ở đâu" thì mở file này)  =====================
Thuộc agent: "Huấn luyện viên" (id="coach", vai trò Reinforcement Learning) trong hệ 7 agent của Project 01.

Mô hình RL (viết tay, không dùng thư viện): Q-learning dạng bảng (tabular) trên môi trường mô phỏng đường huyết.
  - Trạng thái (state)  : dict gồm giờ, đường huyết, tinh bột còn trong ruột (carbs), vận động còn tác dụng (act), số giờ từ bữa ăn
                          gần nhất (since_meal), số lần tập (exercise), bệnh nhân (độ nhạy insulin, đường huyết nền, thuốc).
  - Trạng thái rời rạc  : encode() nén state thành tuple 7 số (khối giờ, mức đường huyết, mức tinh bột, mức vận động, mức đói,
                          nhóm nhạy insulin, có dùng thuốc) làm khóa của bảng Q  (tổng cộng ~1535 trạng thái đã gặp khi huấn luyện).
  - Hành động (action)  : 7 hành động trong ACTIONS (nghỉ, ăn đường nhanh, bữa nhẹ/chính/nhiều tinh bột, đi bộ, tập).
  - Phần thưởng (reward): step() cộng zone_reward(đường huyết mới) (+1 nếu 70–180, âm nếu thấp/cao/nguy hiểm), trừ 0,6 nếu quá 6 giờ không ăn
                          ban ngày, cộng 0,25 cho 2 lần vận động đầu tiên trong ngày.
  - Cập nhật Q (train) : Q(s,a) ← Q(s,a) + α·[ r + γ·max_a' Q(s',a') − Q(s,a) ],  α = 0,15, γ = 0,97, epsilon-greedy giảm 1,0 → 0,05.

Thành phần chính (tìm theo tên):
  - ACTIONS, AID, MEALS, NIGHT_ALLOWED : danh sách hành động và ràng buộc (ban đêm chỉ được nghỉ hoặc ăn đường nhanh).
  - clock(), is_night(), allowed()     : giờ trong ngày, có phải ban đêm không, hành động nào hợp lệ.
  - zone(), zone_reward()              : phân vùng đường huyết (danger/low/ok/high) và phần thưởng theo vùng.
  - patient_from_risk(), random_patient(), new_state() : tạo bệnh nhân (từ hồ sơ thật hoặc ngẫu nhiên khi huấn luyện) và trạng thái đầu ngày.
  - step(), done()                     : MÔI TRƯỜNG mô phỏng 1 giờ (động học đường huyết) và điều kiện kết thúc ngày (24 bước).
  - encode()                           : rời rạc hóa trạng thái.
  - rule_coach(), no_coach()           : hai đối chứng — huấn luyện viên theo luật (4 luật) và người sinh hoạt thường ngày.
  - class QCoach                       : bảng Q; load() nạp rl_qtable.json, act() chọn hành động epsilon-greedy, advise() lời khuyên cho người chơi.
  - train()                            : vòng huấn luyện Q-learning (được gọi bởi train/train_rl.py).
  - run_day(), policies(), compare()   : chạy 1 ngày / so sánh 3 chính sách (RL, luật, không có) trên nhiều ngày.
  - advice(), plan_for()               : lời khuyên tại một thời điểm và kế hoạch cả ngày do RL gợi ý.

Dữ liệu đọc: artifacts/rl_qtable.json (bảng Q đã huấn luyện). Dữ liệu ghi: không (train/train_rl.py ghi rl_qtable.json + rl_metrics.json).
Được gọi từ: mas/orchestrator.py (Clinic: self.coach = rl.QCoach.load(); hội chẩn gọi rl.plan_for, rl.compare),
             server.py (API /api/life/start, /api/life/step, /api/life/compare), train/train_rl.py (huấn luyện + đánh giá).
Lưu ý: đây là MÔ PHỎNG ĐỒ CHƠI, không phải mô hình y khoa; kết quả "trong ngưỡng 99,9%" là trong môi trường mô phỏng này.
=====================================================================================================================
"""
from __future__ import annotations

import json
import math
import os
import random

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")  # thư mục chứa rl_qtable.json, rl_metrics.json
START_HOUR, DAY_STEPS = 6, 24  # một ngày mô phỏng: bắt đầu 6 giờ sáng, 24 bước (mỗi bước 1 giờ)

# 7 hành động. carbs = gam tinh bột đưa vào; act = mức vận động (0 = không, 1 = đi bộ, 2 = tập mạnh);
# fast = tinh bột hấp thu nhanh (75%/giờ) hay chậm (45%/giờ) trong step().

ACTIONS = [
    dict(id="rest", label="Nghỉ ngơi", emoji="🛋️", desc="Không ăn, không vận động", carbs=0, act=0.0, fast=False),
    dict(id="sweet", label="Ăn nhẹ có đường", emoji="🍬", desc="Nước ngọt, kẹo hoặc chuối: đường lên nhanh, dùng khi đường huyết thấp", carbs=20, act=0.0, fast=True),
    dict(id="light_meal", label="Bữa nhẹ", emoji="🥗", desc="Salad, sữa chua, trái cây: ít tinh bột", carbs=30, act=0.0, fast=False),
    dict(id="normal_meal", label="Bữa chính", emoji="🍚", desc="Cơm, thịt cá và rau cân đối", carbs=60, act=0.0, fast=False),
    dict(id="heavy_meal", label="Bữa nhiều tinh bột", emoji="🍔", desc="Cơm lớn, đồ chiên, tráng miệng ngọt", carbs=100, act=0.0, fast=False),
    dict(id="walk", label="Đi bộ nhanh", emoji="🚶", desc="Đi bộ khoảng 30 phút", carbs=0, act=1.0, fast=False),
    dict(id="workout", label="Tập thể dục", emoji="🏃", desc="Chạy hoặc đạp xe khoảng 40 phút", carbs=0, act=2.0, fast=False),
]
AID = {a["id"]: i for i, a in enumerate(ACTIONS)}  # id hành động -> chỉ số cột trong bảng Q
MEALS = {"light_meal", "normal_meal", "heavy_meal"}  # các hành động tính là "đã ăn" (đặt lại since_meal)
NIGHT_ALLOWED = ["rest", "sweet"]  # từ 23 giờ đến 6 giờ chỉ được nghỉ hoặc ăn đường nhanh (khi tụt đường)


def clock(hour: int) -> int:
    """Giờ trong ngày 0–23 (hour có thể vượt 24 vì đếm liên tục từ START_HOUR)."""
    return hour % 24


def is_night(hour: int) -> bool:
    """True nếu là ban đêm (23 giờ đến trước 6 giờ)."""
    return clock(hour) >= 23 or clock(hour) < 6


def allowed(hour: int) -> list[str]:
    """Danh sách id hành động hợp lệ tại giờ này (ban đêm bị hạn chế)."""
    return NIGHT_ALLOWED if is_night(hour) else [a["id"] for a in ACTIONS]


def zone(g: float) -> str:
    """Phân vùng đường huyết g (mg/dL): "danger" (<55 hoặc >250), "low" (<70), "ok" (70–180), "high" (>180)."""
    if g < 55 or g > 250:
        return "danger"
    if g < 70:
        return "low"
    if g <= 180:
        return "ok"
    return "high"


def zone_reward(g: float) -> float:
    """Phần thưởng theo đường huyết g: 70–140 → +1,0 (tốt nhất); 140–180 → +0,3; 180–250 → −0,7; >250 → −2,5; 55–70 → −1,0; <55 → −3,0.
    Phạt nặng hơn ở vùng nguy hiểm để agent học tránh hạ/tăng đường huyết quá mức."""
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
    """Tạo bệnh nhân mô phỏng từ hồ sơ thật: risk_index (từ MLP) và BMI quyết định độ nhạy insulin s ∈ [0,3; 1,15]
    (nguy cơ/BMI càng cao thì s càng thấp = kháng insulin); đường huyết nền = 90 + 100·risk (trộn 50/50 với đường huyết đói nếu có);
    risk ≥ 0,35 thì cho dùng thuốc (medication = 10). Trả về dict bệnh nhân dùng trong step()."""
    s = max(0.3, min(1.15, 1.12 - 1.5 * risk_index - 0.012 * max(0.0, bmi - 25)))
    base = 90 + 100 * risk_index
    if fasting:
        base = 0.5 * fasting + 0.5 * base
    base = max(80.0, min(200.0, base))
    return dict(name=name, insulin_sensitivity=round(s, 3), baseline_glucose=round(base, 1), medication=10.0 if risk_index >= 0.35 else 0.0)


def random_patient(rng: random.Random) -> dict:
    """Bệnh nhân ngẫu nhiên cho HUẤN LUYỆN (độ nhạy insulin ngẫu nhiên 0,3–1,15, đường huyết nền tương ứng) để Q-table tổng quát hóa."""
    s = rng.uniform(0.3, 1.15)
    base = max(80.0, min(200.0, 90 + 55 * (1.15 - s) / 0.85 * 1.2 + rng.gauss(0, 8)))
    return dict(name="Bệnh nhân ngẫu nhiên", insulin_sensitivity=round(s, 3), baseline_glucose=round(base, 1),
                medication=10.0 if (s < 0.6 and rng.random() < 0.6) else 0.0)


def new_state(patient: dict, rng: random.Random | None = None) -> dict:
    """Trạng thái đầu ngày (6 giờ): đường huyết = nền (± nhiễu nhỏ nếu có rng), chưa có tinh bột/vận động, 2 giờ kể từ bữa trước."""
    g = patient["baseline_glucose"] + (rng.gauss(0, 4) if rng else 0.0)
    return dict(hour=START_HOUR, glucose=round(g, 1), carbs=0.0, act=0.0, since_meal=2, exercise=0, patient=patient, log=[])


def step(state: dict, action: str, rng: random.Random) -> tuple[dict, float, dict]:
    """Một giờ mô phỏng. Trả về (trạng thái mới, phần thưởng, thông tin).

    Động học đường huyết (đây là MÔI TRƯỜNG của RL):
      g' = g + rise − drop_act − pull − drop_med + nhiễu N(0, 3²),  kẹp trong [35, 420] mg/dL
        rise     = tinh bột hấp thu · 1,35 · (1,7 − s)         (s = độ nhạy insulin; s thấp thì đường lên nhiều hơn)
        drop_act = vận động · 14 · (0,5 + 0,5·s) · hệ số an toàn (không hạ thêm khi g đã gần 62)
        pull     = (g − nền) · 0,16 · s                         (cơ thể tự kéo đường huyết về mức nền)
        drop_med = thuốc · hệ số an toàn
    Phần thưởng = zone_reward(g') − 0,6 nếu >= 6 giờ chưa ăn ban ngày + 0,25 cho 2 lần vận động đầu tiên."""
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
    """Hết ngày khi đã đi đủ DAY_STEPS (24) bước kể từ START_HOUR."""
    return state["hour"] >= START_HOUR + DAY_STEPS


def encode(state: dict) -> tuple:
    """Rời rạc hóa trạng thái thành tuple 7 số (khóa hàng của bảng Q): (khối 4 giờ trong ngày 0–5, mức đường huyết 0–5, mức tinh bột 0–2,
    có vận động còn tác dụng 0/1, mức đói 0–2, nhóm nhạy insulin 0–2, dùng thuốc 0/1). Gom nhóm để bảng Q đủ nhỏ mà vẫn học được."""
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
    """Đối chứng 1 — huấn luyện viên THEO LUẬT: duyệt luật theo thứ tự ưu tiên (đêm → <75 ăn đường → ≥5 giờ chưa ăn → giờ ăn 7/12/19
    → >220 tập mạnh → >160 đi bộ). Trả về (id hành động, lý do bằng tiếng Việt)."""
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
    """Đối chứng 2 — người KHÔNG có huấn luyện viên: thói quen cố định (ăn nhiều tinh bột bữa tối, ăn vặt đồ ngọt, ít vận động).
    Trả về (id hành động, mô tả)."""
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
    """Huấn luyện viên RL: giữ bảng Q (dict khóa = encode(state), giá trị = list 7 giá trị Q, một giá trị cho mỗi hành động)."""

    def __init__(self, q: dict | None = None):
        self.q = q if q is not None else {}

    @classmethod
    def load(cls):
        """Nạp bảng Q đã huấn luyện từ artifacts/rl_qtable.json (khóa lưu dạng chuỗi "a,b,c,..." được đổi lại thành tuple số nguyên)."""
        with open(os.path.join(ART, "rl_qtable.json"), encoding="utf-8") as f:
            raw = json.load(f)
        return cls({tuple(int(x) for x in k.split(",")): v for k, v in raw["q"].items()})

    def values(self, key):
        """Danh sách Q-value của trạng thái `key`; trạng thái chưa gặp được khởi tạo toàn 0."""
        return self.q.setdefault(key, [0.0] * len(ACTIONS))

    def act(self, state: dict, epsilon: float = 0.0, rng: random.Random | None = None) -> str:
        """Chọn hành động epsilon-greedy: với xác suất epsilon chọn NGẪU NHIÊN một hành động hợp lệ (khám phá),
        còn lại chọn hành động có Q cao nhất (khai thác; hòa thì lấy hành động có chỉ số nhỏ hơn). epsilon = 0 khi chạy thật."""
        legal = allowed(state["hour"])
        if rng is not None and rng.random() < epsilon:
            return rng.choice(legal)
        v = self.values(encode(state))
        best = max(legal, key=lambda a: (v[AID[a]], -AID[a]))
        return best

    def advise(self, state: dict) -> tuple[str, str]:
        """Gợi ý cho người chơi: lấy hành động tốt nhất của bảng Q nhưng không khuyên đồ ngọt khi đường huyết đã cao."""
        g = state["glucose"]
        v = self.values(encode(state))
        legal = [x for x in allowed(state["hour"]) if not (x == "sweet" and g >= 140)] or allowed(state["hour"])
        a = max(legal, key=lambda x: (v[AID[x]], -AID[x]))
        if a == "sweet":
            why = "Đường huyết đang thấp hoặc sắp thấp, cần đường hấp thu nhanh."
        elif a == "rest":
            why = "Đường huyết đang ổn, nghỉ ngơi và chờ giờ ăn tiếp theo." if g <= 180 else "Đường huyết đang cao nhưng chưa nên làm gì mạnh; theo dõi thêm một giờ."
        else:
            why = {"light_meal": "Cần ăn nhưng chọn ít tinh bột để đường huyết không vọt lên.",
                   "normal_meal": "Đến lúc ăn bữa chính; khẩu phần vừa phải giúp đường huyết ổn định.",
                   "heavy_meal": "Bữa lớn: chỉ nên chọn khi đường huyết thấp và còn nhiều giờ để vận động sau đó.",
                   "walk": "Đi bộ nhẹ giúp kéo đường huyết xuống và đã học được là lựa chọn an toàn.",
                   "workout": "Đường huyết cao nên vận động mạnh hơn để hạ nhanh."}[a]
        return a, f"{why} (G = {g:.0f} mg/dL)"


# ------------------------------------------------------------------ huấn luyện và đánh giá
def train(episodes: int = 60000, seed: int = 11, alpha: float = 0.15, gamma: float = 0.97, log_every: int = 2000):
    """Huấn luyện Q-learning. Mỗi episode = 1 ngày của 1 bệnh nhân ngẫu nhiên. epsilon giảm tuyến tính 1,0 → 0,05 trong 70% số episode đầu.
    Trả về (QCoach đã học, đường cong học: trung bình tổng thưởng mỗi `log_every` episode)."""
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
            # TD target: r (hết ngày) hoặc r + γ·max Q(s', a' hợp lệ); rồi Q(s,a) += α·(target − Q(s,a))
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
    """Chạy trọn 1 ngày với một chính sách (QCoach hoặc hàm rule_coach/no_coach). Trả về từng giờ (rows) và các chỉ số:
    tir = tỉ lệ giờ trong ngưỡng 70–180 (time in range), mean, số giờ hạ (hypo) / tăng cao (hyper) / nguy hiểm (danger), tổng thưởng."""
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
    """Ba chính sách để so sánh: rl (bảng Q), rule (luật), none (không huấn luyện viên): {id: (tên, chính sách)}."""
    return {"rl": ("Huấn luyện viên RL", qcoach), "rule": ("Huấn luyện viên theo luật", rule_coach), "none": ("Không có huấn luyện viên", no_coach)}


def compare(patient: dict, qcoach: QCoach, days: int = 30, seed: int = 7) -> dict:
    """So sánh 3 chính sách trên `days` ngày (cùng bệnh nhân, mỗi ngày một seed): thời gian trong ngưỡng, đường huyết trung bình,
    giờ hạ/tăng cao/nguy hiểm, điểm thưởng và đường cong đường huyết trung bình theo giờ. Trả về dict(coaches, table, days) cho giao diện/báo cáo."""
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
    """Lời khuyên tại một trạng thái: gợi ý của RL (kèm lý do) và gợi ý của huấn luyện viên theo luật, để người chơi đối chiếu."""
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
