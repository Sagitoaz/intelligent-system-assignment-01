import engine
import shared

META = dict(agent="agent01", id="a1_chase", title="Mèo đuổi chuột", js="a1_chase.js",
            description="Xem mèo (Q-learning, BFS hoặc ngẫu nhiên) đuổi chuột trên bản đồ lớn có tường.")


def handle(p):
    cmap, agent, _ = shared.get_cat(p.get("map", "garden"))
    ep = engine.play_episode(p.get("policy", "q"), cmap, agent, int(p.get("seed", 1)))
    return dict(map=cmap.public(), **ep)
