import random

import engine
import shared

META = dict(agent="agent01", id="a1_play", title="Bạn là chuột", js="a1_play.js",
            description="Tự điều khiển chuột bằng phím mũi tên, mèo AI sẽ đuổi theo bạn.")


def handle(p):
    cmap, agent, _ = shared.get_cat(p.get("map", "garden"))
    if p.get("op") == "new":
        env = engine.ChaseEnv(cmap, random.Random(int(p.get("seed", 1))))
        env.reset()
        return dict(map=cmap.public(), cat=cmap.cells[env.cat], mouse=cmap.cells[env.mouse], caught=False)
    cat = cmap.index[tuple(map(int, p["cat"].split(",")))]
    mouse = cmap.index[tuple(map(int, p["mouse"].split(",")))]
    return engine.interactive_step(p.get("policy", "q"), cmap, agent, cat, mouse, int(p["dir"]), int(p.get("seed", 1)))
