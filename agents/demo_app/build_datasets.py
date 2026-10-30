"""Tạo/tải về các tệp CSV cho Agent 01 và Agent 02 vào thư mục data/agent01 và data/agent02.

Không huấn luyện lại mô hình nào: các kết quả đã có được gộp lại, phần dữ liệu tổng hợp được sinh
bằng seed cố định (chạy lại cho kết quả giống hệt).
Chạy: python agents/demo_app/build_datasets.py
"""
from __future__ import annotations

import csv
import glob
import os
import random
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import engine  # noqa: E402

A1 = os.path.join(ROOT, "data", "agent01")
A2 = os.path.join(ROOT, "data", "agent02")
ART = os.path.join(ROOT, "agents", "agent01", "expanded", "artifacts")
A2_SRC = os.path.join(ROOT, "agents", "agent02")


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        names = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=names, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"{os.path.relpath(path, ROOT):55s} {len(rows):>8,} dòng")


def orders(n=60000):
    rng = random.Random(861)
    statuses = ["delivered", "shipping", "processing", "cancelled", "refunded"]
    weights = [0.55, 0.15, 0.12, 0.10, 0.08]
    channels = ["web", "mobile", "cửa hàng", "đại lý"]
    rows = []
    for i in range(n):
        rows.append(dict(id=f"DH{100000 + i}", owner=f"U{rng.randint(1, 4000):04d}", tenant=f"T{rng.randint(1, 25):02d}",
                         paid=rng.randrange(30000, 4000000, 1000), status=rng.choices(statuses, weights)[0],
                         age_days=rng.randint(0, 120), channel=rng.choice(channels), note="Dữ liệu tổng hợp"))
    return rows


def intents(n=12000):
    rng = random.Random(7)
    templates = {
        "refund": ["trả lại tiền đơn {o}", "tôi muốn hoàn tiền đơn hàng {o}", "hoàn tiền giao dịch {o} giúp tôi"],
        "cancel": ["hủy đơn {o}", "tôi không muốn mua {o} nữa", "xin huỷ đơn hàng {o}"],
        "status": ["đơn {o} đang ở đâu", "kiểm tra trạng thái đơn {o}", "bao giờ đơn {o} đến"],
        "policy": ["chính sách đổi trả thế nào", "bảo hành bao lâu", "phí vận chuyển tính ra sao"],
        "complaint": ["đơn {o} giao thiếu hàng", "sản phẩm trong đơn {o} bị lỗi", "tôi muốn khiếu nại đơn {o}"],
    }
    prefix = ["", "", "bạn ơi, ", "cho mình hỏi, ", "alo shop, ", "xin chào, "]
    rows = []
    for i in range(n):
        label = rng.choice(list(templates))
        text = rng.choice(prefix) + rng.choice(templates[label]).format(o=f"DH{rng.randint(100000, 159999)}")
        rows.append(dict(id=i + 1, text=text, intent=label))
    return rows


def map_cells():
    rows = []
    for key in engine.MAP_SPECS:
        m = engine.ChaseMap(key)
        for r in range(m.h):
            for c in range(m.w):
                rows.append(dict(map=key, row=r, col=c, cell="open" if m.grid[r][c] == "." else "wall"))
    return rows


def chase_episodes(per_policy=1500):
    """Nhật ký ván đuổi bắt: mỗi bản đồ × ba chính sách × per_policy ván, dùng Q-table đã cache nếu có."""
    from shared import get_cat

    rows = []
    for key in engine.MAP_SPECS:
        cmap, agent, _ = get_cat(key)
        for kind in ("random", "q", "bfs"):
            for i in range(per_policy):
                r = engine.play_episode(kind, cmap, agent, 1_000_000 + i)
                rows.append(dict(map=key, policy=kind, episode=i + 1, caught=int(r["caught"]), steps=r["steps"]))
    return rows


def market_steps(episodes=3000):
    rng_seed = 2026
    rows = []
    pairs = {"luật vs luật": (engine.rule_action, engine.rule_action),
             "luôn giảm vs luật": (lambda s: 1, engine.rule_action),
             "không giảm vs luật": (lambda s: 0, engine.rule_action),
             "luật vs luôn giảm": (engine.rule_action, lambda s: 1)}
    for name, (pa, pb) in pairs.items():
        env = engine.MarketEnv(random.Random(rng_seed))
        for ep in range(episodes):
            st = env.reset()
            for t in range(engine.SellerConfig.steps):
                a, b = pa(st[0]), pb(st[1])
                nxt, rw, info = env.step([a, b])
                rows.append(dict(scenario=name, episode=ep + 1, step=t + 1, demand=info["demand"], action_a=a, action_b=b,
                                 sales_a=info["sales"][0], sales_b=info["sales"][1],
                                 profit_a=round(rw[0], 2), profit_b=round(rw[1], 2)))
                st = nxt
    return rows


def merge_training_logs():
    rows = []
    for path in sorted(glob.glob(os.path.join(A2_SRC, "extra_results", "*_training.csv"))):
        parts = os.path.basename(path)[:-len("_training.csv")].rsplit("_", 1)
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(config=parts[0], seed=parts[1], **r))
    return rows


def main():
    for name in ("synthetic_orders.csv", "evaluation_cases.csv", "intent_predictions.csv", "rl_summary.csv",
                 "rl_training_history.csv", "rl_trained_seeds.csv", "rl_oracle.csv", "rl_qtable_example.csv"):
        src = os.path.join(ART, name)
        if os.path.exists(src):
            os.makedirs(os.path.join(A1, "notebook_results"), exist_ok=True)
            shutil.copy(src, os.path.join(A1, "notebook_results", name))
    for sub in ("agent02_results", "extra_results"):
        for src in glob.glob(os.path.join(A2_SRC, sub, "*.csv")):
            if sub == "extra_results" and os.path.basename(src).endswith(("_training.csv", "_evaluation.csv")):
                continue
            os.makedirs(os.path.join(A2, "experiment_results"), exist_ok=True)
            shutil.copy(src, os.path.join(A2, "experiment_results", sub[:5] + "_" + os.path.basename(src)))
    write(os.path.join(A1, "orders_large.csv"), orders())
    write(os.path.join(A1, "intent_utterances.csv"), intents())
    write(os.path.join(A1, "cat_mouse_maps.csv"), map_cells())
    write(os.path.join(A1, "cat_mouse_episodes.csv"), chase_episodes())
    write(os.path.join(A2, "market_steps.csv"), market_steps())
    write(os.path.join(A2, "q_learning_training_all.csv"), merge_training_logs())


if __name__ == "__main__":
    main()
