"""Kiểm thử hệ đa agent tiểu đường (project01): mô hình, luật an toàn, đồ thị tri thức, RL và kịch bản hội chẩn."""
import os
import random

import pytest

from project01.mas import rl, rules
from project01.mas.data import PERSONAS, normalize, to_vector, FEATURES
from project01.mas.orchestrator import Clinic


@pytest.fixture(scope="module")
def clinic():
    return Clinic()


def test_profile_vector_matches_brfss_schema():
    v = to_vector(normalize(PERSONAS[1]["profile"]))
    assert len(v) == len(FEATURES) == 21
    assert 1 <= v[FEATURES.index("Age")] <= 13


def test_probabilities_sum_to_one_and_ordering(clinic):
    low = clinic.dl.predict(PERSONAS[0]["profile"])
    high = clinic.dl.predict(PERSONAS[2]["profile"])
    assert abs(sum(low["risk"].values()) - 1) < 1e-5
    assert high["risk_index"] > low["risk_index"]


def test_guardian_rules_trigger_on_dangerous_glucose():
    out = rules.evaluate(dict(PERSONAS[2]["profile"]), 0.5, 0.6)
    assert any(r["id"] == "R1" and r["level"] == "danger" for r in out)
    assert rules.evaluate(dict(PERSONAS[0]["profile"]), 0.02, 0.01) == []


def test_knowledge_graph_is_consistent_and_rag_cites_sources(clinic):
    ids = set(clinic.kb.nodes)
    assert all(e["source"] in ids and e["target"] in ids for e in clinic.kb.edges)
    r = clinic.kb.search("What organ makes insulin?", 3)
    assert r["passages"] and r["passages"][0]["url"].startswith("https://www.niddk.nih.gov")


def test_rl_coach_beats_no_coach_in_simulation(clinic):
    patient = rl.patient_from_risk(0.4, 30, None)
    cmp = rl.compare(patient, clinic.coach, days=10, seed=3)
    tir = {c["id"]: c["time_in_range"] for c in cmp["coaches"]}
    assert tir["rl"] > tir["none"]


def test_night_hours_restrict_actions():
    assert rl.allowed(24) == ["rest", "sweet"]
    assert "walk" in rl.allowed(12)


def test_consult_script_contract(clinic):
    r = clinic.consult(PERSONAS[2]["profile"])
    agents = {e["agent"] for e in r["events"]}
    assert {"doctor", "predictor", "knowledge", "coach", "guardian", "reporter", "critic"} <= agents
    assert r["events"][-1]["type"] == "final"
    assert r["summary"]["level"] == "cao" and r["summary"]["red_flags"]
    assert len(r["report"]["figures"]) == 3 and all(f["png_base64"] for f in r["report"]["figures"])
    assert [e["i"] for e in r["events"]] == list(range(len(r["events"])))


def test_tts_text_cleaning_and_voice_map():
    from project01.mas.tts import Speaker, VOICES
    assert Speaker.clean("Chào 😀 bạn!  Xem https://x.org/a ok") == "Chào bạn! Xem ok"
    assert len(Speaker.clean("a" * 1000)) <= 320
    assert set(VOICES) == {"doctor", "predictor", "knowledge", "coach", "guardian", "reporter", "critic"}


def test_deepgraph_demo_sector_api_runs():
    from deepgraph_demo import server as dg
    dg.load()
    n = len(dg.DATA["tickers"])
    banks = [i for i in range(n) if dg.DATA["label"][i] == 1][:4]
    others = [i for i in range(n) if dg.DATA["label"][i] == 0][:5]
    out = dg.api_sector({"labelled": banks + others, "k": 4})
    assert out["n_test"] == n - 9 and 0 <= out["gcn"]["accuracy"] <= 1 and len(out["gcn"]["prob"]) == n
    try:
        dg.api_sector({"labelled": banks[:1], "k": 4})
        raise AssertionError("cần báo lỗi khi thiếu một lớp")
    except ValueError:
        pass


def test_tieuluan_hub_has_six_chapters_and_no_chapter1():
    from tieuluan_demo import server as hub
    assert not hasattr(hub, "ch1_data")
    assert {c["name"] for c in hub.CHILDREN} == {"streamlit", "fastapi", "deepgraph"}
    html = open(os.path.join(hub.STATIC, "index.html"), encoding="utf-8").read()
    assert html.count("{ n: ") == 6 and "{ n: 1," not in html and "ch1" not in html
    assert not os.path.exists(os.path.join(hub.STATIC, "ch1.html"))


def test_deepgraph_report_numbers_match_results_json():
    from deepgraph_demo import server as dg
    dg.load()
    rep = dg.api_graph()["report"]
    assert abs(rep["sector"]["GCN, đồ thị tương quan"] - 0.7306) < 1e-3 and rep["splits"] == 40
    assert abs(rep["majority"] - 10 / 18) < 1e-9
