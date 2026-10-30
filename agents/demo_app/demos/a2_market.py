import random

import engine

META = dict(agent="agent02", id="a2_market", title="Bạn là người bán", js="a2_market.js",
            description="Chọn giảm giá hay giữ giá mỗi vòng, đấu với đối thủ theo luật cố định trong 20 vòng.")


def handle(p):
    seed, rnd = int(p.get("seed", 1)), int(p.get("round", 0))
    if p.get("op") == "new":
        return dict(demand=random.Random(seed).randrange(2))
    demand = int(p["demand"])
    prev_a, prev_b = int(p.get("prev_a", 0)), int(p.get("prev_b", 0))
    a = int(p["action"])
    b = engine.rule_action((demand, prev_a))
    env = engine.MarketEnv(random.Random(seed * 1000 + rnd))
    env.demand, env.prev = demand, [prev_a, prev_b]
    _, rw, info = env.step([a, b])
    return dict(round=rnd + 1, demand=demand, you=a, bot=b, sales=info["sales"],
                profit=[round(x, 2) for x in rw], next_demand=env.demand)
