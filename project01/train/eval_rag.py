"""Đánh giá truy xuất của agent tri thức trên 16 câu hỏi có đáp án (data/chapter04/diabetes_qa_demo.json).

So sánh BM25 thuần và Graph-RAG (mở rộng truy vấn qua đồ thị tri thức). Chỉ số: hit@1, hit@3 theo đoạn gold, MRR.
Chạy: python project01/train/eval_rag.py  -> project01/artifacts/rag_metrics.json

======================================================================================================
BẢN ĐỒ FILE
======================================================================================================
Chức năng: đo chất lượng TRUY XUẤT (retrieval) của agent tri thức, trả lời câu hỏi "Graph-RAG có tốt hơn BM25 thuần không?".
Đọc : data/chapter04/diabetes_qa_demo.json (16 câu hỏi, mỗi câu có `gold_ids` = các đoạn tài liệu chứa đáp án đúng);
      KnowledgeBase trong mas/kb.py (corpus NIDDK 183 đoạn + kg.json).
Ghi : artifacts/rag_metrics.json (hit@1, hit@3, MRR và thứ hạng từng câu; giao diện/báo cáo đọc file này).
Hàm : main() - chạy hai chế độ: "BM25" (graph=False) và "Graph-RAG" (graph=True: mở rộng truy vấn bằng nút láng giềng trong KG).
Giải thích chỉ số (nên thuộc khi bảo vệ):
  - rank   : vị trí đầu tiên trong 5 đoạn trả về mà đoạn đó thuộc gold_ids (None nếu không có đoạn đúng trong top 5)
  - hit@1  : tỉ lệ câu có đoạn đúng đứng ĐẦU ; hit@3 : tỉ lệ câu có đoạn đúng nằm trong 3 đoạn đầu
  - MRR    : Mean Reciprocal Rank = trung bình của 1/rank (rank None tính 0); càng gần 1 càng tốt
Kết quả đã lưu: BM25 hit@1 0,375 / hit@3 0,5625 / MRR 0,4844 ; Graph-RAG hit@1 0,4375 / hit@3 0,625 / MRR 0,5208 (16 câu).
Cỡ mẫu nhỏ (16 câu) nên chênh lệch chỉ mang tính minh họa, không đủ để kết luận có ý nghĩa thống kê.
======================================================================================================
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from project01.mas.kb import KnowledgeBase  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "project01", "artifacts", "rag_metrics.json")


def main():
    """Chạy đánh giá cả hai chế độ truy xuất trên 16 câu hỏi, in kết quả và ghi rag_metrics.json."""
    kb = KnowledgeBase()
    qa = json.load(open(os.path.join(ROOT, "data", "chapter04", "diabetes_qa_demo.json"), encoding="utf-8"))
    res = {}
    for name, graph in (("BM25", False), ("Graph-RAG", True)):
        h1 = h3 = 0
        rr = 0.0
        rows = []
        for q in qa:
            got = [p["id"] for p in kb.search(q["question"], 5, graph=graph)["passages"]]  # top-5 đoạn trả về, theo thứ tự xếp hạng
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
