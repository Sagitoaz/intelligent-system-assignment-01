"""Materialize the already-frozen master splits for final A04 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assignment04.config import COMMENTS_TEST, COMMENTS_TRAIN, DIABETES_DATA, HOUSE_DATA
from src.assignment04.full_data import prepare_comments, prepare_diabetes, prepare_house


MANIFESTS = ROOT / "results/assignment04/manifests"
OUTPUT = ROOT / "results/assignment04/full_scale/prepared"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("diabetes", "house", "comments", "all"), default="all")
    args = parser.parse_args()
    jobs = {
        "diabetes": lambda: prepare_diabetes(ROOT / DIABETES_DATA, MANIFESTS / "diabetes.index.csv",
                                               MANIFESTS / "diabetes_preprocessing.json", OUTPUT),
        "house": lambda: prepare_house(ROOT / HOUSE_DATA, MANIFESTS / "house.index.csv",
                                        MANIFESTS / "house_preprocessing.json", OUTPUT),
        "comments": lambda: prepare_comments(ROOT / COMMENTS_TRAIN, ROOT / COMMENTS_TEST,
                                               MANIFESTS / "comments", MANIFESTS / "comments_preprocessing.json", OUTPUT),
    }
    summary = {}
    for name, job in jobs.items():
        if args.task in (name, "all"):
            print(f"PREPARING FULL {name}", flush=True)
            summary[name] = job()
            print(summary[name], flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "split_counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
