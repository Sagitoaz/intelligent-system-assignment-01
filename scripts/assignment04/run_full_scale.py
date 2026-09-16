"""Run final full-scale Improved CNN experiments (PyTorch and TensorFlow only)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assignment04.full_runner import run_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("diabetes", "house", "comments", "all"), default="all")
    args = parser.parse_args()
    tasks = ("diabetes", "house", "comments") if args.task == "all" else (args.task,)
    print(run_all(ROOT, tasks=tasks).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
