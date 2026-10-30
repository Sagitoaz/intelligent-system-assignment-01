import functools

import engine

META = dict(agent="agent02", id="a2_sweep", title="Quét siêu tham số", js="a2_sweep.js",
            description="Bảng lợi nhuận mỗi bước của Q-learning theo alpha và epsilon ban đầu.")


@functools.lru_cache(maxsize=4)
def _sweep(opponent):
    rows = []
    for alpha in (0.01, 0.1, 0.5):
        for eps in (0.05, 0.2, 0.5):
            r = engine.run_agent02(alpha=alpha, epsilon_start=eps, episodes=1500, eval_episodes=200, opponent=opponent)
            rows.append(dict(alpha=alpha, epsilon=eps, profit=r["summary"][0]["profit_per_step"],
                             discount_rate=r["summary"][0]["discount_rate"],
                             rule_profit=r["summary"][1]["profit_per_step"]))
    return dict(rows=rows)


def handle(p):
    return _sweep(p.get("opponent", "rule"))
