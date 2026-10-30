import engine
from demos.a2_train import run

META = dict(agent="agent02", id="a2_opponents", title="Ba kiểu đối thủ", js="a2_opponents.js",
            description="Cùng một Q-learning đấu với đối thủ theo luật, luôn giảm giá và không bao giờ giảm giá.")


def handle(p):
    items = []
    for key, (title, _) in engine.OPPONENTS.items():
        r = run(0.1, 0.9, 0.2, 3000, key, 42)
        items.append(dict(opponent=key, title=title, summary=r["summary"], policy=r["policy"], curve=r["curve"]))
    return dict(items=items)
