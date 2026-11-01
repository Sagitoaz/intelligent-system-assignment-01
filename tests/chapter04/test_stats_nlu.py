import os
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.chapter04.agent import ToolAgent
from src.chapter04.nlu import find_company, parse_calculation
from src.chapter04.rag import INSUFFICIENT_VI, lexical_support
from src.chapter04.stats import DiabetesStats, detect_language

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'data/chapter05/brfss2023_diabetes_150k.csv'


@pytest.fixture(scope='module')
def stats():
    return DiabetesStats(CSV)


@pytest.fixture(scope='module')
def frame():
    return pd.read_csv(CSV)


def test_dataset_is_new_and_large(frame):
    assert 100_000 <= len(frame) <= 200_000
    assert {'diabetes_status', 'age_group', 'sex', 'bmi', 'high_blood_pressure'} <= set(frame.columns)


def test_count_and_rate_match_pandas(stats, frame):
    assert stats.answer('Có bao nhiêu người trong dữ liệu khảo sát?')['value'] == len(frame)
    rate = stats.answer('Tỉ lệ người mắc tiểu đường là bao nhiêu?')
    assert rate['value'] == pytest.approx((frame.diabetes_status == 'yes').mean() * 100)
    assert 'tiểu đường' in rate['answer'] and '%' in rate['answer']


def test_group_and_compare_have_tables(stats, frame):
    by_age = stats.answer('Tỉ lệ tiểu đường theo nhóm tuổi')
    assert len(by_age['table']['rows']) == frame.age_group.nunique()
    cmp_ = stats.answer('So sánh tỉ lệ tiểu đường giữa nam và nữ')
    assert [r[0] for r in cmp_['table']['rows']] == ['nam', 'nữ']
    en = stats.answer('Compare diabetes rate between men and women')
    assert en['table']['columns'][0] == 'Sex' and 'gap' in en['answer']


def test_filters_and_mean_bmi(stats, frame):
    hbp = stats.answer('Tỉ lệ tiểu đường ở người huyết áp cao')
    sub = frame[frame.high_blood_pressure == 1]
    assert hbp['value'] == pytest.approx((sub.diabetes_status == 'yes').mean() * 100)
    bmi = stats.answer('BMI trung bình của người bị tiểu đường là bao nhiêu?')
    assert bmi['value'] == pytest.approx(frame[frame.diabetes_status == 'yes'].bmi.mean())
    old = stats.answer('How many people over 65 have diabetes?')
    assert old['value'] == len(frame[frame.age_group.isin(['65-69', '70-74', '75-79', '80+']) & (frame.diabetes_status == 'yes')])


def test_unknown_question_is_polite_not_error(stats):
    assert stats.answer('Hello there') is None
    assert 'Tỉ lệ' in stats.help_text('vi')
    assert not DiabetesStats.looks_statistical('How many hours should a person fast?')
    assert not DiabetesStats.looks_statistical('Tiểu đường là gì?')


def test_calculator_reads_natural_sentences():
    assert parse_calculation('Tính tăng trưởng doanh thu từ 100 lên 130') == ('growth', [100, 130])
    assert parse_calculation('250 chia 5') == ('divide', [250, 5])
    assert parse_calculation('What is 12 plus 30?') == ('add', [12, 30])
    assert parse_calculation('Doanh thu năm 2008 là bao nhiêu') is None


def test_agent_answers_in_sentences_without_generator(stats):
    graph = nx.DiGraph()
    graph.add_node('company:V', type='Company', ticker='V')
    graph.add_node('report:V/2008/p.pdf:abc', type='Report', year='2008', filename='V/2008/p.pdf')
    graph.add_node('passage:x', type='Passage')
    graph.add_edge('company:V', 'report:V/2008/p.pdf:abc')
    graph.add_edge('report:V/2008/p.pdf:abc', 'passage:x')
    agent = ToolAgent(None, stats, {'finance': graph})
    calc = agent.ask('Tính tăng trưởng doanh thu từ 100 lên 130', 'finance')
    assert calc['route'] == 'calculate' and calc['value'] == pytest.approx(30) and 'tăng 30%' in calc['answer']
    g = agent.ask('Công ty V có bao nhiêu báo cáo?', 'finance')
    assert g['route'] == 'graph' and 'Công ty V có 1 báo cáo' in g['answer'] and '2008' in g['answer']
    s = agent.ask('Tỉ lệ tiểu đường theo nhóm tuổi', 'diabetes')
    assert s['route'] == 'statistics' and s['table'] and s['confidence']['kind'] == 'computed'
    odd = agent.ask('xin chào bạn', 'diabetes', task='statistics')
    assert odd['status'] == 'unsupported' and 'Tỉ lệ' in odd['answer']
    assert find_company('Doanh thu của AAPL?', {'AAPL', 'V'}) == 'AAPL'


def test_language_detection_and_lexical_guard():
    assert detect_language('Tiểu đường là gì?') == 'vi'
    assert detect_language('What is diabetes?') == 'en'
    evidence = [{'text': 'Insulin is a hormone made by the pancreas.'}]
    assert lexical_support('What makes insulin?', evidence)
    assert not lexical_support('current price of gold', evidence)
    assert 'Không đủ' in INSUFFICIENT_VI


def test_api_gives_sentences_not_422(monkeypatch):
    os.environ['CH5_NO_WARMUP'] = '1'
    from fastapi.testclient import TestClient
    from deployment.chapter04.app import app
    client = TestClient(app)
    r = client.post('/api/v1/knowledge/ask', json={'domain': 'diabetes', 'question': 'Tỉ lệ người mắc tiểu đường là bao nhiêu?'})
    assert r.status_code == 200 and r.json()['route'] == 'statistics' and '%' in r.json()['answer']
    r = client.post('/api/v1/knowledge/ask', json={'domain': 'finance', 'question': '   '})
    assert r.status_code == 200 and r.json()['status'] == 'unsupported'
    r = client.post('/api/v1/knowledge/ask', json={'domain': 'finance', 'question': 'Công ty V có bao nhiêu báo cáo?'})
    assert r.json()['route'] == 'graph' and 'báo cáo' in r.json()['answer']
    assert client.get('/api/v1/status').status_code == 200
