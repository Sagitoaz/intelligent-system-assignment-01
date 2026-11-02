"""Kiểm thử bác sĩ phản biện (critic) và giải thích what-if: bằng chứng chạy được, tranh luận tối đa 2 vòng, ưu tiên số đo trực tiếp."""
import pytest

from project01.mas import critic, rules, whatif
from project01.mas.data import PERSONAS
from project01.mas.orchestrator import Clinic


@pytest.fixture(scope="module")
def clinic():
    return Clinic()


def persona(pid):
    return next(p for p in PERSONAS if p["id"] == pid)["profile"]


def test_conflict_between_low_mlp_and_high_glucose_escalates_after_two_rounds(clinic):
    r = clinic.consult(persona("hoa"))
    cr = r["summary"]["critique"]
    assert cr["mlp_level"] == "thấp" and cr["escalated"] and cr["final_level"] == "cao" and cr["rounds"] == 2
    assert r["summary"]["level"] == "cao"
    deb = [e for e in r["events"] if e.get("round")]
    assert {e["round"] for e in deb} == {1, 2}
    assert {"critic", "predictor", "guardian", "knowledge"} <= {e["agent"] for e in deb}
    assert all(len(e["text"]) <= 320 for e in r["events"])  # vừa một lượt đọc của giọng nói
    assert [e["i"] for e in r["events"]] == list(range(len(r["events"])))


def test_evidence_is_real_not_self_agreement(clinic):
    """Mô hình không có đường huyết trong đầu vào: đổi đường huyết không làm đổi dự đoán, nhưng làm đổi kết luận phản biện."""
    low = dict(persona("hoa"), fasting_glucose=90)
    high = dict(persona("hoa"), fasting_glucose=190)
    assert clinic.dl.predict(low)["risk_index"] == clinic.dl.predict(high)["risk_index"]
    cr_low, cr_high = clinic.consult(low)["summary"]["critique"], clinic.consult(high)["summary"]["critique"]
    assert cr_low["outcome"] == "agree" and cr_high["outcome"] == "escalated"
    assert cr_high["quote"] and cr_high["quote"]["url"].startswith("https://")
    assert any(c["id"] == "C1" and c["status"] == "conflict" for c in cr_high["checks"])


def test_no_conflict_means_single_agreement_event(clinic):
    r = clinic.consult(persona("lan"))
    assert r["summary"]["critique"]["outcome"] == "agree" and r["summary"]["critique"]["rounds"] == 0
    assert [e for e in r["events"] if e["agent"] == "critic" and e["type"] == "verdict"]


def test_debate_never_exceeds_max_rounds(clinic):
    for p in PERSONAS:
        r = clinic.consult(p["profile"])
        assert max([e.get("round", 0) for e in r["events"]] + [0]) <= critic.MAX_ROUNDS


def test_symptom_conflict_triggers_when_model_says_low(clinic):
    prof = dict(persona("lan"), symptoms=["thirst", "frequent_urination", "blurred_vision"], fasting_glucose=None)
    pred = clinic.dl.predict(prof)
    rv = critic.review(clinic, prof, pred, rules.evaluate(prof, pred["risk_index"], pred["risk"]["diabetes"]), "trung bình")
    assert "C2" in rv["hard"]


def test_what_if_reruns_the_model(clinic):
    prof = persona("minh")
    out = whatif.what_if(clinic, prof, {"weight_kg": 70, "smoker": False})
    assert out["delta_index"] < 0 and out["new"]["bmi"] < out["base"]["bmi"]
    assert whatif.what_if(clinic, prof, {})["delta_index"] == 0
    with pytest.raises(ValueError):
        whatif.what_if(clinic, prof, {"fasting_glucose": 90})
    pts = whatif.bmi_curve(clinic, prof)["points"]
    assert len(pts) == 23 and pts[-1]["risk_index"] > pts[0]["risk_index"]
