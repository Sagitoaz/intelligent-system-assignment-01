"""Kiểm thử agent hỏi đáp có trích dẫn (project01/mas/ask.py) và đo hit@k thật trên 40 câu của data/chapter04/diabetes_qa_demo.json."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from project01.mas import ask  # noqa: E402
from project01.mas.kb import KnowledgeBase  # noqa: E402

QA = json.load(open(os.path.join(ROOT, "data", "chapter04", "diabetes_qa_demo.json"), encoding="utf-8"))
BY_ID = {q["id"]: q for q in QA}
# 12 câu tiếng Việt do người viết, mỗi câu gắn với đoạn kỳ vọng của câu tiếng Anh tương ứng trong diabetes_qa_demo.json
VI = {
    "DM01": "Cơ quan nào trong cơ thể tạo ra insulin?", "DM03": "Các triệu chứng thường gặp của bệnh tiểu đường là gì?",
    "DM07": "Nguyên nhân gây tiểu đường típ 1 là gì?", "DM13": "Tiểu đường có thể gây những vấn đề gì về mắt?",
    "DM18": "Tôi nên kiểm tra bàn chân bao lâu một lần?", "DM24": "Mỗi tuần nên tập thể dục bao nhiêu phút?",
    "DM26": "Mức A1C bình thường là bao nhiêu?", "DM29": "Theo dõi đường huyết liên tục là gì?",
    "DM36": "Tiền tiểu đường là gì?", "DM20": "Làm gì khi đường huyết xuống dưới 70 mg/dL?",
    "DM34": "Làm sao để giảm nguy cơ mắc tiểu đường típ 2?", "DM39": "Tăng đường huyết là gì?",
}


@pytest.fixture(scope="module")
def engine():
    e = ask.AskEngine(type("C", (), {"kb": KnowledgeBase()})())
    e.retriever.dense_ready.wait(timeout=float(os.environ.get("ASK_TEST_DENSE_WAIT", "600")))  # nạp torch + MiniLM có thể chậm khi máy bận
    return e


def _hits(engine, q, k=3):
    return [h["id"] for h in engine.retriever.search(q, k)["hits"]]


def test_corpus_is_the_new_492_passages(engine):
    assert len(engine.retriever.passages) == 492
    assert {p["org"] for p in engine.retriever.passages} == {"NIDDK", "CDC", "MedlinePlus"}


def test_vietnamese_term_mapping():
    q = ask.vi_to_en_terms("Triệu chứng của tiểu đường típ 2 là gì?")
    assert {"symptoms", "type", "2", "diabetes"} <= set(q)
    assert "feet" in ask.vi_to_en_terms("Cách chăm sóc bàn chân")


def test_answer_has_citations_and_disclaimer(engine):
    r = engine.ask("Cơ quan nào tạo ra insulin?", mode="retrieval")
    assert r["mode"] == "retrieval" and r["citations"] and "[1]" in r["answer"]
    assert all(c["url"].startswith("https://") and c["id"] for c in r["citations"])
    assert "không thay thế bác sĩ" in r["answer"]
    assert r["notes"] and "trích đoạn" in " ".join(r["notes"])


def test_personalised_intro(engine):
    r = engine.ask("Triệu chứng của tiểu đường là gì?", profile=dict(age_years=55, height_cm=165, weight_kg=75, high_bp=True), mode="retrieval")
    assert r["answer"].startswith("Với hồ sơ của bạn (55 tuổi, BMI khoảng 27.5")


@pytest.mark.parametrize("q", ["Thời tiết hôm nay thế nào?", "Cách nấu phở ngon", "Who won the football world cup?", "How do I bake a chocolate cake?"])
def test_refuses_off_topic(engine, q):
    r = engine.ask(q, mode="retrieval")
    assert r.get("refused") == "nohit" and not r["citations"]


@pytest.mark.parametrize("q", ["Tôi có bị tiểu đường không?", "Tôi nên tiêm bao nhiêu đơn vị insulin?", "Liều metformin cho tôi là bao nhiêu?",
                               "Do I have diabetes?", "Should I stop my insulin?"])
def test_refuses_personal_diagnosis_or_dose(engine, q):
    r = engine.ask(q, mode="retrieval")
    assert r.get("refused") == "personal" and not r["citations"] and "bác sĩ" in r["answer"]


def test_emergency_is_redirected(engine):
    assert engine.ask("Bố tôi bị hôn mê, đường huyết rất cao", mode="retrieval")["refused"] == "emergency"


def test_bad_input(engine):
    with pytest.raises(ValueError):
        engine.ask("", mode="retrieval")
    with pytest.raises(ValueError):
        engine.ask("tiểu đường là gì", mode="bogus")


def test_qwen_falls_back_when_busy(engine):
    """Khóa Qwen đang bị giữ => không khởi chạy lượt thứ hai, trả lời bằng chế độ trích đoạn kèm thông báo trung thực."""
    if not ask.QwenRunner.available():
        pytest.skip("Máy không có Qwen")
    assert engine.qwen.lock.acquire(blocking=False)
    try:
        r = engine.ask("Triệu chứng của tiểu đường là gì?", mode="qwen")
    finally:
        engine.qwen.lock.release()
    assert r["mode"] == "retrieval" and "đang bận" in " ".join(r["notes"])


def test_retrieval_eval_writes_artifact(engine):
    """Đo hit@1/hit@3 thật (chế độ trích đoạn) trên 40 câu tiếng Anh và 12 câu tiếng Việt; ghi project01/artifacts/ask_eval.json."""
    def run(items):
        h1 = h3 = 0
        for q, gold in items:
            ids = _hits(engine, q, 3)
            h1 += bool(ids and ids[0] in gold)
            h3 += any(i in gold for i in ids)
        n = len(items)
        return dict(questions=n, hit_at_1=round(h1 / n, 3), hit_at_3=round(h3 / n, 3), hits_at_1=h1, hits_at_3=h3)
    en = run([(q["question"], q["gold_ids"]) for q in QA])
    vi = run([(VI[i], BY_ID[i]["gold_ids"]) for i in VI])
    both = dict(questions=en["questions"] + vi["questions"], hits_at_1=en["hits_at_1"] + vi["hits_at_1"], hits_at_3=en["hits_at_3"] + vi["hits_at_3"])
    both["hit_at_1"], both["hit_at_3"] = round(both["hits_at_1"] / both["questions"], 3), round(both["hits_at_3"] / both["questions"], 3)
    hybrid = engine.retriever.dense is not None
    out = dict(corpus="data/chapter04/diabetes_corpus.jsonl (492 đoạn, 28 trang)", qa="data/chapter04/diabetes_qa_demo.json",
               retriever="hybrid: BM25 + mở rộng đồ thị tri thức + vector MiniLM (RRF)" if hybrid else "BM25 + mở rộng đồ thị tri thức (mô hình vector không nạp được)",
               hybrid=hybrid, english=en, vietnamese=vi, combined=both,
               note="Đoạn kỳ vọng do người viết gắn tay (gold_ids); một câu có thể có nhiều đoạn đúng nhưng chỉ tính các đoạn đã gắn nên số liệu là cận dưới. "
                    "Câu tiếng Việt do người dịch tay từ 12 câu tiếng Anh và đi qua bảng thuật ngữ Việt-Anh, không qua Qwen.")
    with open(os.path.join(ROOT, "project01", "artifacts", "ask_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    assert en["questions"] == 40 and vi["questions"] == 12
    assert en["hit_at_3"] >= 0.5 and vi["hit_at_3"] >= 0.3
