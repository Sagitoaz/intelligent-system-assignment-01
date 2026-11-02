"""Kiểm thử hàng đợi bệnh nhân và theo dõi xu hướng 4 tuần (dữ liệu mẫu mô phỏng)."""
import pytest

from project01.mas import followup
from project01.mas.data import PERSONAS
from project01.mas.orchestrator import Clinic


@pytest.fixture(scope="module")
def clinic():
    return Clinic()


def test_queue_triage_puts_danger_first(clinic):
    q = followup.triage(clinic)["queue"]
    assert len(q) == len(PERSONAS) and [o["rank"] for o in q] == list(range(1, len(q) + 1))
    assert q[0]["id"] == "hung" and q[0]["danger"] and "cờ đỏ" in q[0]["reason"]
    rest = [o["risk_index"] for o in q if not o["danger"]]
    assert rest == sorted(rest, reverse=True)
    assert all(o["level"] in ("thấp", "trung bình", "cao") and "profile" in o for o in q)


def test_queue_custom_profiles_and_limits(clinic):
    out = followup.triage(clinic, [PERSONAS[0]["profile"], PERSONAS[1]["profile"]])["queue"]
    assert [o["name"] for o in out] == ["Anh Minh", "Chị Lan"]
    with pytest.raises(ValueError):
        followup.triage(clinic, ["x"])


def test_sample_series_is_deterministic_and_labelled(clinic):
    a = followup.analyze(clinic, dict(profile=PERSONAS[1]["profile"]))
    b = followup.analyze(clinic, dict(profile=PERSONAS[1]["profile"]))
    assert a["series"] == b["series"] and a["simulated"] and "mẫu minh họa" in a["note"]
    assert len(a["series"]["glucose"]) == len(a["series"]["weight"]) == 28
    assert followup.analyze(clinic, dict(profile=PERSONAS[1]["profile"], seed=5))["series"] != a["series"]


def test_followup_stats_match_series(clinic):
    r = followup.analyze(clinic, dict(profile=PERSONAS[1]["profile"]))
    g = r["series"]["glucose"]
    assert r["stats"]["pct_ge_126"] == round(100 * sum(x >= 126 for x in g) / 28)
    assert abs(r["stats"]["glucose_slope_per_week"] - 7 * followup._slope(g)) < 0.06
    assert len(r["weeks"]) == 4 and r["weeks"][0]["changes"] == []
    assert abs(r["weeks"][0]["mean_glucose"] - sum(g[:7]) / 7) < 0.06


def test_rl_changes_only_come_from_computed_plans(clinic):
    r = followup.analyze(clinic, dict(profile=PERSONAS[1]["profile"]))
    wk = r["weeks"]
    assert any(w["changes"] for w in wk[1:]), "Anh Minh có xu hướng tăng nên gợi ý RL phải đổi ở ít nhất một tuần"
    for prev, cur in zip(wk, wk[1:]):
        before = {p["hour"]: p["action"] for p in prev["plan"]}
        after = {p["hour"]: p["action"] for p in cur["plan"]}
        assert {c["hour"] for c in cur["changes"]} == {h for h in set(before) | set(after) if before.get(h) != after.get(h)}
        assert cur["reasons"]


def test_custom_series_and_validation(clinic):
    ser = dict(glucose=[100 + i for i in range(28)], weight=[70.0] * 28)
    r = followup.analyze(clinic, dict(profile=PERSONAS[0]["profile"], series=ser))
    assert not r["simulated"] and r["stats"]["glucose_slope_per_day"] == 1.0
    with pytest.raises(ValueError):
        followup.analyze(clinic, dict(profile=PERSONAS[0]["profile"], series=dict(glucose=[1, 2], weight=[1, 2])))
