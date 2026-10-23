"""Execute all notebook cells and check observable RL semantics and outputs."""
from pathlib import Path
import collections
import json
import math
import random
import runpy
import statistics
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    module = Path(__file__).with_name("rl_cells.py")
    assert module.exists(), "The RL notebook cell module has not been implemented"
    cells = runpy.run_path(str(module))["RL_CELLS"]
    assert 12 <= len(cells) <= 14
    for cell in cells:
        assert {"title", "source", "explanation", "discussion"} <= cell.keys()
        assert len(cell["source"].splitlines()) <= 45, cell["title"]
        compile(cell["source"], cell["title"], "exec")
    with tempfile.TemporaryDirectory(prefix="rl-expanded-check-") as directory:
        env = dict(json=json, random=random, math=math, statistics=statistics,
                   time=time, np=np, pd=pd, plt=plt, Path=Path,
                   defaultdict=collections.defaultdict, deque=collections.deque,
                   ARTIFACTS=Path(directory), display=lambda value: None)
        started = time.perf_counter()
        for i, cell in enumerate(cells, 1):
            exec(compile(cell["source"], cell["title"], "exec"), env)
            print(f"CELL {i:02d} PASS: {cell['title']}")
        # Goal/hole terminate; collision stays put; episode cutoff is truncation.
        rng = random.Random(902)
        assert env["rl_step"](("SG...", ".....", ".....", ".....", "....."),
                              (0, 0), 1, rng) == ((0, 1), 10.0, True)
        assert env["rl_step"](("SH...", ".....", ".....", ".....", "....G"),
                              (0, 0), 1, rng) == ((0, 1), -10.0, True)
        assert env["rl_step"](env["rl_maps"]["A_open"], (0, 0), 0, rng)[0] == (0, 0)
        assert env["rl_target"](10, True, [999] * 4, 0.95) == 10
        assert env["rl_target"](-0.05, False, [2] * 4, 0.95) == 1.8499999999999999
        ties = [env["rl_greedy"]([0] * 4, rng) for _ in range(100)]
        assert set(ties) == {0, 1, 2, 3}
        repeat1 = env["rl_train"](env["rl_maps"]["A_open"], 7, episodes=50)
        repeat2 = env["rl_train"](env["rl_maps"]["A_open"], 7, episodes=50)
        assert repeat1 == repeat2, "Training must reproduce exactly for a fixed seed"
        q = env["rl_models"][("A_open", "Q-learning", 0)]
        frozen_q = {state: values[:] for state, values in q.items()}
        env["rl_evaluate"](env["rl_maps"]["A_open"], q, 5, 909)
        assert q == frozen_q, "Evaluation must not update or expand the Q table"
        truncated = env["rl_rollout"](env["rl_maps"]["A_open"], None,
                                       random.Random(0), max_steps=1)
        assert truncated["truncated"] and not truncated["terminated"]
        assert all(len(path) > 1 for path in env["rl_oracle_paths"].values())
        assert len(env["rl_trained_rows"]) == 30
        assert len(env["rl_ablation_rows"]) == 60
        assert len(env["rl_robustness_rows"]) == 30
        payload = json.loads((Path(directory) / "rl_extended_metrics.json").read_text("utf-8"))
        assert len(payload["summary"]) == 15
        assert len(payload["figure_paths"]) == 4
        assert all((Path(directory) / Path(p).name).exists() for p in payload["figure_paths"])
        assert len(list(Path(directory).glob("*.csv"))) >= 5
        assert "NaN" not in json.dumps(payload, allow_nan=False)
        print("SEMANTICS PASS: terminal targets, truncation, ties, reproducibility, frozen evaluation")
        print("SUMMARY", json.dumps(payload["summary"], ensure_ascii=False))
        print("ROBUSTNESS", env["rl_robust_df"].groupby("map")["success_rate"].agg(["mean", "min", "max"]).to_json())
        print(f"VERIFIED {len(cells)} cells, 4 PNG, {len(list(Path(directory).glob('*.csv')))} CSV in {time.perf_counter()-started:.2f}s")
        plt.close("all")


if __name__ == "__main__":
    main()
