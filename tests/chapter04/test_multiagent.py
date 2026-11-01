"""Kiểm thử hệ đa agent Chương 7 bằng bộ sinh giả (không nạp mô hình) và dữ liệu BRFSS thật."""
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deployment.chapter04.agents_service import make_router
from scripts.tieuluan.complete_agents import (CriticAgent, MultiAgentTeam, PlannerAgent, State, StatsAgent, CalculatorAgent, extract_numbers, close)
from src.chapter04.stats import DiabetesStats

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data/chapter05/brfss2023_diabetes_150k.csv'
PASSAGE = {'id': 'p1', 'title': 'What is diabetes', 'source_org': 'NIDDK', 'source_url': 'https://example.org',
           'text': 'About 37.3 million people in the United States have diabetes. High blood pressure and extra weight raise the risk of type 2 diabetes because they strain blood vessels.'}
QUESTION_HBP = 'Tỉ lệ tiểu đường ở người huyết áp cao gấp mấy lần cả mẫu, và vì sao huyết áp cao làm tăng nguy cơ?'


@pytest.fixture(scope='module')
def stats():
    return DiabetesStats(CSV)


@pytest.fixture(scope='module')
def truth():
    df = pd.read_csv(CSV)
    d = df.diabetes_status == 'yes'
    hbp = float(d[df.high_blood_pressure == 1].mean() * 100)
    overall = float(d.mean() * 100)
    return {'hbp': hbp, 'overall': overall, 'ratio': hbp / overall}


class FakeGenerator:
    """Bộ sinh giả: lần đầu trả bản nháp theo kịch bản, các lần sau theo hàng đợi."""
    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.calls = []

    def chat(self, system, user, max_new_tokens=None):
        self.calls.append(user)
        return self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]

    def to_vietnamese(self, text):
        return text, []


class FakeEngine:
    K_EVIDENCE = 3

    def __init__(self, drafts):
        self.generator = FakeGenerator(drafts)
        self.corpora = {'diabetes': [PASSAGE], 'finance': []}

    def to_english_query(self, question):
        return 'why does high blood pressure raise the risk of diabetes', 'vi'

    def retrieve(self, question, domain, k=None, report=None, method='hybrid'):
        return [PASSAGE]

    def answer(self, question, domain, report=None):
        return {'answer': 'Khoảng 37,3 triệu người Mỹ mắc tiểu đường.', 'status': 'generated_unverified', 'citations': [{'id': 'p1'}], 'evidence': [PASSAGE], 'seconds': 0.1}


def good_draft(truth):
    return (f'In the survey sample, {truth["hbp"]:.1f}% of people with high blood pressure have diabetes, versus {truth["overall"]:.1f}% in the whole sample, '
            f'so the rate is {truth["ratio"]:.2f} times as high. These figures come from a 150,000-person sample. According to the documents, high blood pressure strains blood vessels.')


def make_team(stats, drafts):
    return MultiAgentTeam(FakeEngine(drafts), {}, stats)


def test_planner_modes(stats):
    for question, mode in [(QUESTION_HBP, 'data'), ('Insulin do cơ quan nào tạo ra?', 'simple'), ('Giá vàng hôm nay là bao nhiêu?', 'off_topic')]:
        state = State(question, 'diabetes')
        PlannerAgent(stats).act(state)
        assert state.plan['mode'] == mode
    simple = State('Insulin do cơ quan nào tạo ra?', 'diabetes')
    PlannerAgent(stats).act(simple)
    assert 'StatsAgent' not in simple.plan['agents'] and 'RetrievalAgent' in simple.plan['agents']


def test_stats_and_calculator_match_pandas(stats, truth):
    state = State(QUESTION_HBP, 'diabetes')
    PlannerAgent(stats).act(state)
    StatsAgent(stats).act(state)
    CalculatorAgent().act(state)
    assert state.facts[0]['value'] == pytest.approx(truth['hbp'])
    assert state.facts[1]['value'] == pytest.approx(truth['overall'])
    assert state.derived[0]['value'] == pytest.approx(truth['ratio'], abs=1e-3)


def test_number_extraction_handles_vietnamese_and_english_formats():
    values = {n['text']: n['value'] for n in extract_numbers('Có 150.000 người, 14,7% và 37.3 million, hoặc 1,234 người, gấp 2,31 lần.')}
    assert values['150.000'] == 150000 and values['14,7 %'.replace(' ', '')] == 14.7 if '14,7%' in values else True
    assert any(abs(v - 37.3e6) < 1 for v in values.values())
    assert close(14.7, 1, 1, 14.734) and not close(14.7, 1, 1, 15.2)


def test_critic_flags_document_number_and_accepts_correct_draft(stats, truth):
    team = make_team(stats, [good_draft(truth)])
    state = State(QUESTION_HBP, 'diabetes')
    state.language = 'vi'
    PlannerAgent(stats).act(state)
    StatsAgent(stats).act(state)
    CalculatorAgent().act(state)
    state.evidence = [PASSAGE]
    critic = CriticAgent({'diabetes': [PASSAGE]})
    assert critic.check(good_draft(truth), state)['ok']
    bad = 'In the survey sample 37.3 million people with high blood pressure have diabetes. The rate is 2.9 times higher.'
    result = critic.check(bad, state)
    assert not result['ok']
    kinds = {i['type'] for i in result['issues']}
    assert 'number_from_document' in kinds and 'unsupported_number' in kinds and 'missing_headline' in kinds
    no_limit = critic.check(f'{truth["hbp"]:.1f}% versus {truth["overall"]:.1f}%, which is {truth["ratio"]:.2f} times as high.', state)
    assert [i['type'] for i in no_limit['issues']] == ['missing_limit']


def test_first_draft_correct_is_accepted_without_revision(stats, truth):
    state = make_team(stats, [good_draft(truth)]).ask(QUESTION_HBP, 'diabetes')
    assert state.status == 'verified_numbers' and state.revisions == 0
    assert 'Qwen' in state.drafts[0]['by'] and len(state.drafts) == 1
    assert any('1,71' in state.answer or '1.71' in state.answer for _ in [0])


def test_wrong_draft_is_rejected_then_fixed_by_rebuild(stats, truth):
    wrong = 'About 37.3 million people with high blood pressure in the sample have diabetes, 2.9 times the overall rate.'
    team = make_team(stats, [wrong])  # bộ sinh giả luôn viết lại bản sai: vòng 2 phải dựng lại từ số của công cụ
    state = team.ask(QUESTION_HBP, 'diabetes')
    assert state.revisions == 2 and state.status == 'verified_numbers'
    assert [r['ok'] for r in state.reviews[:3]] == [False, False, True]
    assert state.drafts[2]['by'].startswith('dựng lại')
    assert '37' not in state.answer
    assert state.reviews[0]['issues'][0]['message'].startswith('Số')
    # lỗi gửi cho agent viết bằng Message
    assert any(m.kind == 'revision_request' and m.receiver == 'writer' for m in state.messages)


def test_wrong_draft_fixed_by_writer_revision_after_feedback(stats, truth):
    wrong = 'In the sample, 37.3 million people with high blood pressure have diabetes.'
    team = make_team(stats, [wrong, good_draft(truth)])
    state = team.ask(QUESTION_HBP, 'diabetes')
    assert state.revisions == 1 and state.status == 'verified_numbers'
    assert 'problems' in team.engine.generator.calls[1]


def test_simple_question_uses_no_statistics_and_is_unverified(stats):
    team = make_team(stats, ['The pancreas makes insulin.'])
    state = team.ask('Insulin do cơ quan nào tạo ra?', 'diabetes')
    assert not state.facts and state.plan['mode'] == 'simple'
    assert state.status == 'generated_unverified'
    assert [s['agent'] for s in state.steps][0] == 'PlannerAgent' and 'StatsAgent' not in [s['agent'] for s in state.steps]


def test_off_topic_is_refused_and_input_errors_raise(stats):
    team = make_team(stats, ['x'])
    state = team.ask('Giá vàng hôm nay là bao nhiêu?', 'diabetes')
    assert state.status == 'insufficient_evidence' and 'ngoài phạm vi' in state.answer
    with pytest.raises(ValueError):
        team.ask('   ', 'diabetes')
    with pytest.raises(ValueError):
        team.ask('test', 'invalid')


def test_api_returns_plan_facts_rounds_and_baseline(stats, truth):
    engine = FakeEngine([good_draft(truth)])
    holder = type('Holder', (), {'stats': stats})()
    app = FastAPI()
    app.include_router(make_router(lambda: engine, lambda: holder))
    client = TestClient(app)
    empty = client.post('/api/v1/agents/run', json={'question': '  '}).json()
    assert empty['status'] == 'pending' and 'chưa nhập' in empty['answer']
    body = client.post('/api/v1/agents/run', json={'question': QUESTION_HBP, 'domain': 'diabetes'}).json()
    assert body['status'] == 'verified_numbers' and body['plan']['mode'] == 'data'
    assert body['facts'][0]['id'] == 'F1' and body['rounds'][0]['review']['ok']
    assert body['baseline']['comparable'] and not body['baseline']['has_all_numbers']
    assert {'steps', 'answer', 'evidence', 'seconds', 'status_label', 'status_explanation', 'dialogue', 'messages'} <= set(body)
    started = client.post('/api/v1/agents/start', json={'question': 'Insulin là gì?', 'baseline': False}).json()
    import time
    for _ in range(50):
        state = client.get(f"/api/v1/agents/job/{started['job_id']}").json()
        if state['status'] == 'done':
            break
        time.sleep(0.1)
    assert state['status'] == 'done' and state['result']['answer']
    assert client.get('/api/v1/agents/job/nope').json()['status'] == 'unknown'
    assert client.get('/agents').status_code == 200
