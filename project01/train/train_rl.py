"""Huấn luyện huấn luyện viên RL (Q-learning) và đánh giá so với luật và không có huấn luyện viên.

Chạy: python project01/train/train_rl.py   (chỉ dùng thư viện chuẩn, mất khoảng 1–2 phút)
Kết quả: project01/artifacts/rl_qtable.json và rl_metrics.json

======================================================================================================
BẢN ĐỒ FILE
======================================================================================================
Chức năng: huấn luyện agent "Huấn luyện viên" (Reinforcement Learning) và so sánh với 2 đối chứng. File này chỉ là KỊCH BẢN CHẠY;
môi trường mô phỏng và thuật toán nằm trong project01/mas/rl.py (hàm rl.train, rl.step, rl.encode, lớp rl.QCoach).
Thuật toán: Q-learning dạng BẢNG (tabular), 60.000 episode (mỗi episode = 1 ngày = 24 bước, mỗi bước = 1 giờ), seed 11.
  - Trạng thái (rl.encode): nhóm giờ trong ngày, mức đường huyết (6 bậc), lượng carb đang tiêu hóa, mức vận động,
    số giờ từ bữa ăn trước, độ nhạy insulin (3 bậc), có dùng thuốc hay không -> 1.535 trạng thái thực sự gặp.
  - Hành động (7): nghỉ, ăn nhẹ có đường, bữa nhẹ, bữa thường, bữa lớn, đi bộ, tập mạnh. Ban đêm chỉ được nghỉ hoặc ăn nhẹ có đường.
  - Reward: cao khi đường huyết trong 70-180 mg/dL, phạt nặng khi hạ đường huyết hoặc vùng nguy hiểm (xem rl.zone_reward).
  - Siêu tham số: alpha 0,15 (tốc độ học), gamma 0,97 (coi trọng hậu quả dài hạn), epsilon-greedy giảm tuyến tính từ 1,0 xuống 0,05
    trong 70% số episode đầu (khám phá nhiều lúc đầu, khai thác về sau).
Đánh giá: 210 bệnh nhân ngẫu nhiên CHƯA thấy khi huấn luyện (seed 2026), chia 3 nhóm độ nhạy insulin x 70 người, mỗi người 5 ngày.
  So 3 chính sách (rl.policies): "Huấn luyện viên RL", "Huấn luyện viên theo luật" (rl.rule_coach), "Không có huấn luyện viên" (rl.no_coach).
  Chỉ số: time_in_range (tỉ lệ giờ trong 70-180), đường huyết TB, số giờ hạ/tăng đường huyết, số giờ nguy hiểm, reward TB.
Ghi : artifacts/rl_qtable.json (bảng Q: khóa "block,gb,cb,ab,hb,sb,med" -> 7 giá trị Q; mas/rl.py QCoach.load đọc file này khi chạy demo)
      artifacts/rl_metrics.json (đường cong học + bảng so sánh; báo cáo và giao diện đọc file này)
Lưu ý để trả lời thầy: bệnh nhân là MÔ PHỎNG đồ chơi, không phải mô hình sinh lý y khoa; kết quả chỉ chứng minh RL học được chính sách
tốt TRONG môi trường mô phỏng đó.
======================================================================================================
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from project01.mas import rl  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def main():
    """Huấn luyện Q-learning rồi đánh giá và lưu bảng Q + số liệu. Không có đầu vào; chạy khoảng 1-2 phút."""
    t0 = time.time()
    coach, curve = rl.train(episodes=60000, seed=11)  # curve: lợi suất TB mỗi 2.000 episode, dùng vẽ đường cong học
    seconds = round(time.time() - t0, 1)
    print("huấn luyện xong", seconds, "s;", len(coach.q), "trạng thái đã gặp")
    # đánh giá trên 200 bệnh nhân ngẫu nhiên chưa thấy khi huấn luyện (seed khác), mỗi người 5 ngày, ba nhóm độ nhạy insulin
    # lấy mẫu cho tới khi mỗi nhóm độ nhạy insulin đủ 70 người (nhóm hiếm sẽ được lấp dần) -> so sánh công bằng giữa 3 nhóm
    rng = random.Random(2026)
    groups = {"nhạy insulin tốt (s ≥ 0,85)": [], "trung bình (0,55 ≤ s < 0,85)": [], "kháng insulin (s < 0,55)": []}
    while min(len(v) for v in groups.values()) < 70:
        p = rl.random_patient(rng)
        key = "nhạy insulin tốt (s ≥ 0,85)" if p["insulin_sensitivity"] >= 0.85 else "trung bình (0,55 ≤ s < 0,85)" if p["insulin_sensitivity"] >= 0.55 else "kháng insulin (s < 0,55)"
        if len(groups[key]) < 70:
            groups[key].append(p)
    results = {}
    for gname, plist in groups.items():
        row = {}
        for pid, (name, pol) in rl.policies(coach).items():
            # mỗi chính sách chạy cùng bộ bệnh nhân và cùng seed (900000 + ...) nên khác biệt chỉ do chính sách, không do may rủi
            runs = [rl.run_day(pol, p, 900000 + 10 * i + d) for i, p in enumerate(plist) for d in range(5)]
            n = len(runs)
            row[pid] = dict(name=name, time_in_range=round(sum(r["tir"] for r in runs) / n, 4), mean_glucose=round(sum(r["mean"] for r in runs) / n, 1),
                            hypo_hours=round(sum(r["hypo"] for r in runs) / n, 2), hyper_hours=round(sum(r["hyper"] for r in runs) / n, 2),
                            danger_hours=round(sum(r["danger"] for r in runs) / n, 2), mean_reward=round(sum(r["total"] for r in runs) / n, 2), runs=n)
        results[gname] = row
        print(gname, {k: (v["time_in_range"], v["mean_reward"]) for k, v in row.items()})
    os.makedirs(OUT, exist_ok=True)
    # lưu bảng Q: khóa tuple trạng thái được nối thành chuỗi "a,b,c,..." vì JSON không có khóa dạng tuple
    with open(os.path.join(OUT, "rl_qtable.json"), "w", encoding="utf-8") as f:
        json.dump(dict(q={",".join(map(str, k)): [round(x, 3) for x in v] for k, v in coach.q.items()}), f)
    with open(os.path.join(OUT, "rl_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(episodes=60000, alpha=0.15, gamma=0.97, seed=11, states_visited=len(coach.q), train_seconds=seconds, actions=[a["id"] for a in rl.ACTIONS],
                       curve=curve, evaluation=results, eval_patients_per_group=70, eval_days_per_patient=5), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
