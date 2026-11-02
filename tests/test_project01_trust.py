"""Kiểm thử số liệu độ tin cậy (trust_metrics.json) của agent Deep Learning."""
import json
import os

import pytest

from project01.mas import trust

ART = os.path.join(os.path.dirname(__file__), "..", "project01", "artifacts")


@pytest.fixture(scope="module")
def tm():
    path = os.path.join(ART, "trust_metrics.json")
    assert os.path.exists(path), "chưa chạy project01/train/eval_trust.py"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_accuracy_matches_training_metrics(tm):
    with open(os.path.join(ART, "dl_metrics.json"), encoding="utf-8") as f:
        dm = json.load(f)
    assert abs(tm["accuracy"] - dm["accuracy"]) < 1e-3
    assert tm["test_rows"] == dm["test_rows"]
    assert tm["confusion_matrix"] == dm["confusion_matrix"]
    for c, r in zip(tm["per_class"], dm["recall_per_class"]):
        assert abs(c["recall"] - r) < 1e-3


def test_groups_sum_to_test_rows(tm):
    for key, g in tm["groups"].items():
        assert sum(r["n"] for r in g["rows"]) == tm["test_rows"], key
        for r in g["rows"]:
            lo, hi = r["accuracy"]["ci"]
            assert lo <= r["accuracy"]["value"] <= hi


def test_probabilities_and_calibration_sane(tm):
    for k in ("diabetes", "pre", "index"):
        c = tm["calibration"][k]
        assert sum(b["n"] for b in c["bins"]) == tm["test_rows"]
        assert 0 <= c["ece"] <= 1 and 0 <= c["brier"] <= 1
        for b in c["bins"]:
            if b["n"]:
                assert 0 <= b["observed"] <= 1 and 0 <= b["mean_pred"] <= 1
    assert tm["per_class"][1]["recall"] < 0.5
    assert tm["baseline_always_none"]["accuracy"] > tm["accuracy"]


def test_alert_precision_consistent(tm):
    a = tm["alerts"]["main"]
    assert a["threshold"] == 0.25
    p = a["precision_diabetes"]
    assert abs(p["k"] / p["n"] - p["value"]) < 1e-3
    assert tm["alerts"]["loose"]["alerts"] >= a["alerts"]
    assert tm["limitations"] and tm["fairness"]["conclusions"]


def test_routes_return_data():
    get, post = trust.routes(None, None)
    assert get["/api/trust"]()["accuracy"] == trust.load()["accuracy"]
    assert "groups" in get["/api/trust/fairness"]()
