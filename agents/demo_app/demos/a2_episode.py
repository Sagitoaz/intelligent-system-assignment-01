from demos.a2_train import run

META = dict(agent="agent02", id="a2_episode", title="Ván 20 bước", js="a2_episode.js",
            description="Xem từng bước một ván cạnh tranh: cầu, hành động, số bán và lợi nhuận của hai bên.")


def handle(p):
    r = run(0.1, 0.9, 0.2, 3000, p.get("opponent", "rule"), int(p.get("seed", 42)))
    return dict(trace=r["trace"], policy=r["policy"], opponent_title=r["opponent_title"])
