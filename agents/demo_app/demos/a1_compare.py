import engine
import shared

META = dict(agent="agent01", id="a1_compare", title="So sánh chính sách", js="a1_compare.js",
            description="Tỉ lệ bắt được chuột và số bước trung bình của ngẫu nhiên, Q-learning và BFS trên ba bản đồ.")
_cache = {}


def handle(p):
    key = p.get("map", "garden")
    if key not in _cache:
        cmap, agent, history = shared.get_cat(key)
        win = 100
        curve = [round(sum(h["caught"] for h in history[i:i + win]) / win, 3)
                 for i in range(0, len(history) - win + 1, win)]
        _cache[key] = dict(map=cmap.public(), rows=engine.evaluate_cat(cmap, agent, 150), curve=curve,
                           episodes=len(history), window=win)
    return _cache[key]
