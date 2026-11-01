import json
from pathlib import Path

from src.chapter04.corpus import chunk_page, extract_blocks
from src.chapter04.textguard import foreign_script, fix_units, is_list_question, strip_foreign, translation_problems
from src.chapter04.suggest import explain_refusal

ROOT = Path(__file__).resolve().parents[2]
HTML = b'''<html><body><main><h1>Symptoms of Diabetes</h1>
<h2>What are the symptoms?</h2>
<p>Common symptoms of diabetes include:</p>
<ul><li>increased urination</li><li>feeling very thirsty</li><li>blurred vision</li><li>fatigue</li></ul>
<p>Some people have no symptoms at all and only find out during a routine blood test at the clinic.</p>
</main></body></html>'''


def test_chunker_keeps_bullet_list_with_lead_sentence_and_heading():
    title, blocks = extract_blocks(HTML)
    chunks = chunk_page('demo', title, blocks, 'https://example.org', '2026-01-01', 'NIDDK')
    assert chunks[0]['id'] == 'demo:0'
    text = chunks[0]['text']
    assert 'What are the symptoms?' in text
    assert 'Common symptoms of diabetes include: increased urination; feeling very thirsty; blurred vision; fatigue' in text
    assert chunks[0]['source_url'] == 'https://example.org' and chunks[0]['accessed'] == '2026-01-01'


def test_real_corpus_has_symptom_list_and_enough_pages():
    corpus = [json.loads(x) for x in (ROOT / 'data/chapter04/diabetes_corpus.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len({c['source_url'] for c in corpus}) >= 15
    assert len({c['id'] for c in corpus}) == len(corpus)
    symptom = [c for c in corpus if c['id'].startswith('symptoms-causes:') and 'increased urination' in c['text'] and 'feeling very thirsty' in c['text']]
    assert symptom
    assert all(c.get('source_url') and c.get('accessed') for c in corpus)


def test_foreign_script_is_detected_and_removed():
    assert foreign_script('青少年 trẻ vị thành niên')
    assert not foreign_script('Đường huyết 180 mg/dL, tuyến tụy.')
    assert '青' not in strip_foreign('Ai có nguy cơ 青少年 mắc bệnh?')
    assert strip_foreign('Triệu chứng: khát nước') == 'Triệu chứng: khát nước'


def test_units_and_numbers_are_preserved_or_flagged():
    assert fix_units('dưới 180 miligam/lượng glucose'.replace('miligam/lượng', 'miligam trên lượng')) .count('mg/dL') == 1
    assert fix_units('70 mg/dl') == '70 mg/dL'
    assert 'missing_number' in translation_problems('Below 70 mg/dL', 'Dưới mức thấp mg/dL')
    assert 'missing_unit' in translation_problems('Below 70 mg/dL', 'Dưới 70')
    assert translation_problems('Below 70 mg/dL', 'Dưới 70 mg/dL') == []
    assert 'foreign_script' in translation_problems('Teens', 'thanh 青少年')


def test_list_question_detection():
    assert is_list_question('What are the symptoms of diabetes?')
    assert is_list_question('What are the complications of diabetes?')
    assert not is_list_question('What is the A1C test?')


def test_refusal_explanations_and_suggestions():
    out = explain_refusal('Giá vàng hôm nay là bao nhiêu?', 'diabetes')
    assert out['refusal_kind'] == 'out_of_scope' and out['suggestions'] and 'tiểu đường' in out['why_refused']
    wrong = explain_refusal('Doanh thu của Visa là bao nhiêu?', 'diabetes')
    assert wrong['refusal_kind'] == 'wrong_domain' and wrong['switch_domain'] == 'finance'
    close = explain_refusal('Metformin có gây hại thận không?', 'diabetes')
    assert close['refusal_kind'] == 'not_in_documents' and close['switch_domain'] is None
