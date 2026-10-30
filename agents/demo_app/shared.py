"""Tiện ích dùng chung cho các demo: cache Q-table, đọc CSV theo trang, thư mục dữ liệu."""
from __future__ import annotations

import csv
import json
import os
import threading

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
DATA_DIRS = {"agent01": os.path.join(ROOT, "data", "agent01"), "agent02": os.path.join(ROOT, "data", "agent02")}
TRAIN_EPISODES = {"garden": 3000, "maze": 5000, "city": 8000}

_maps, _cats, _lock = {}, {}, threading.Lock()


def get_map(key):
    with _lock:
        if key not in _maps:
            _maps[key] = engine.ChaseMap(key)
        return _maps[key]


def get_cat(key):
    """Trả về (bản đồ, mèo đã huấn luyện, lịch sử). Q-table được lưu ở cache/ nên lần sau mở app không cần train lại."""
    cmap = get_map(key)
    with _lock:
        if key in _cats:
            return cmap, *_cats[key]
        path = os.path.join(CACHE, f"cat_{key}.json")
        agent = engine.CatQAgent(seed=0)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            agent.q = {int(k): v for k, v in blob["q"].items()}
            history = blob["history"]
        else:
            agent, history = engine.train_cat(cmap, episodes=TRAIN_EPISODES[key])
            os.makedirs(CACHE, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"q": agent.q, "history": history}, f)
        _cats[key] = (agent, history)
        return cmap, agent, history


def list_csv(agent):
    folder = DATA_DIRS[agent]
    out = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.endswith(".csv"):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, folder).replace("\\", "/")
                with open(full, encoding="utf-8-sig", errors="replace") as f:
                    rows = sum(1 for _ in f) - 1
                out.append(dict(file=rel, rows=rows, kb=round(os.path.getsize(full) / 1024)))
    return out


def read_csv_page(agent, rel, offset=0, limit=20, query=""):
    folder = os.path.realpath(DATA_DIRS[agent])
    full = os.path.realpath(os.path.join(folder, rel))
    if not full.startswith(folder) or not full.endswith(".csv"):
        raise ValueError("Tệp không hợp lệ")
    rows, total, query = [], 0, query.lower()
    with open(full, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if query and query not in ",".join(row).lower():
                continue
            if offset <= total < offset + limit:
                rows.append(row)
            total += 1
    return dict(header=header, rows=rows, total=total, offset=offset)
