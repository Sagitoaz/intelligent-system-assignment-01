import functools

import engine

META = dict(agent="agent02", id="a2_train", title="Huấn luyện & so sánh", js="a2_train.js",
            description="Chỉnh alpha, gamma, epsilon và đối thủ rồi xem đường học, Q-table và lợi nhuận so với luật cố định.")


@functools.lru_cache(maxsize=64)
def run(alpha, gamma, eps, episodes, opponent, seed):
    return engine.run_agent02(alpha=alpha, gamma=gamma, epsilon_start=eps, episodes=episodes,
                              opponent=opponent, seed=seed)


def handle(p):
    return run(float(p.get("alpha", 0.1)), float(p.get("gamma", 0.9)), float(p.get("eps", 0.2)),
               min(int(p.get("episodes", 3000)), 10000), p.get("opponent", "rule"), int(p.get("seed", 42)))
