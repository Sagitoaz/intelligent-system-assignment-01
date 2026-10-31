"""Đánh giá truy xuất của agent tri thức trên 16 câu hỏi có đáp án (data/chapter04/diabetes_qa_demo.json).

So sánh BM25 thuần và Graph-RAG (mở rộng truy vấn qua đồ thị tri thức). Chỉ số: hit@1, hit@3 theo đoạn gold, MRR.
Chạy: python project01/train/eval_rag.py  -> project01/artifacts/rag_metrics.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from project01.mas.kb import KnowledgeBase  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "project01", "artifacts", "rag_metrics.json")


def main():
    kb = KnowledgeBase()
    qa = json.load(open(os.path.join(ROOT, "data", "chapter04", "diabetes_qa_demo.json"), encoding="utf-8"))
    res = {}
    for name, graph in (("BM25", False), ("Graph-RAG", True)):
        h1 = h3 = 0
        rr = 0.0
        rows = []
        for q in qa:
            got = [p["id"] for p in kb.search(q["question"], 5, graph=graph)["passages"]]
            rank = next((i + 1 for i, pid in enumerate(got) if pid in q["gold_ids"]), None)
            h1 += rank == 1
            h3 += rank is not None and rank <= 3
            rr += 1 / rank if rank else 0.0
            rows.append(dict(id=q["id"], question=q["question"], rank=rank))
        n = len(qa)
        res[name] = dict(hit_at_1=round(h1 / n, 4), hit_at_3=round(h3 / n, 4), mrr=round(rr / n, 4), questions=n, detail=rows)
        print(name, {k: v for k, v in res[name].items() if k != "detail"})
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
