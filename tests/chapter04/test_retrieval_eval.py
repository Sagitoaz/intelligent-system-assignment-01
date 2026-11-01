import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location('eval_diabetes', ROOT / 'scripts/chapter05/evaluate_diabetes_retrieval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hit_at_k_and_mrr_are_computed_correctly():
    score = _load().score
    rankings = [['a', 'b', 'c'], ['x', 'y', 'z'], ['q', 'r', 's', 't', 'u']]
    golds = [['a'], ['z'], ['none']]
    result = score(rankings, golds)
    assert result['questions'] == 3
    assert abs(result['hit_at_1'] - 1 / 3) < 1e-9
    assert abs(result['hit_at_3'] - 2 / 3) < 1e-9
    assert abs(result['mrr_at_5'] - (1 + 1 / 3) / 3) < 1e-9


def test_qa_set_is_labelled_on_existing_chunks():
    corpus = {json.loads(x)['id'] for x in (ROOT / 'data/chapter04/diabetes_corpus.jsonl').read_text(encoding='utf-8').splitlines()}
    qa = json.loads((ROOT / 'data/chapter04/diabetes_qa_demo.json').read_text(encoding='utf-8'))
    assert len(qa) >= 30
    for item in qa:
        assert item['gold_ids'] and set(item['gold_ids']) <= corpus
        assert item['needle']
    result = json.loads((ROOT / 'results/chapter04/diabetes_retrieval_eval.json').read_text(encoding='utf-8'))
    assert result['questions'] == len(qa) and result['corpus_chunks'] == len(corpus)
