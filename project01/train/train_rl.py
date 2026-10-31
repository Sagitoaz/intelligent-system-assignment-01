"""Huấn luyện huấn luyện viên RL (Q-learning) và đánh giá so với luật và không có huấn luyện viên.

Chạy: python project01/train/train_rl.py   (chỉ dùng thư viện chuẩn, mất khoảng 1–2 phút)
Kết quả: project01/artifacts/rl_qtable.json và rl_metrics.json
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
    t0 = time.time()
    coach, curve = rl.train(episodes=60000, seed=11)
    seconds = round(time.time() - t0, 1)
    print("huấn luyện xong", seconds, "s;", len(coach.q), "trạng thái đã gặp")
    # đánh giá trên 200 bệnh nhân ngẫu nhiên chưa thấy khi huấn luyện (seed khác), mỗi người 5 ngày, ba nhóm độ nhạy insulin
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
            runs = [rl.run_day(pol, p, 900000 + 10 * i + d) for i, p in enumerate(plist) for d in range(5)]
            n = len(runs)
            row[pid] = dict(name=name, time_in_range=round(sum(r["tir"] for r in runs) / n, 4), mean_glucose=round(sum(r["mean"] for r in runs) / n, 1),
                            hypo_hours=round(sum(r["hypo"] for r in runs) / n, 2), hyper_hours=round(sum(r["hyper"] for r in runs) / n, 2),
                            danger_hours=round(sum(r["danger"] for r in runs) / n, 2), mean_reward=round(sum(r["total"] for r in runs) / n, 2), runs=n)
        results[gname] = row
        print(gname, {k: (v["time_in_range"], v["mean_reward"]) for k, v in row.items()})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rl_qtable.json"), "w", encoding="utf-8") as f:
        json.dump(dict(q={",".join(map(str, k)): [round(x, 3) for x in v] for k, v in coach.q.items()}), f)
    with open(os.path.join(OUT, "rl_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dict(episodes=60000, alpha=0.15, gamma=0.97, seed=11, states_visited=len(coach.q), train_seconds=seconds, actions=[a["id"] for a in rl.ACTIONS],
                       curve=curve, evaluation=results, eval_patients_per_group=70, eval_days_per_patient=5), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
