"""Demo "KB_diabet trên Neo4j + RAG" để trình bày với thầy: in ra từng bước, kèm câu Cypher thật đã chạy.

Chạy: python project01/train/demo_kb_neo4j.py [--file]      (--file: ép dùng kg.json, không cần Neo4j)

Bốn phần trình diễn (khớp 4 ý của thầy):
  1. KB trên Neo4j      : đếm KBNode / KBSource / KBPassage / MENTIONS đang có trên Neo4j và in 2 truy vấn Cypher mẫu.
  2. Nguồn dữ liệu      : liệt kê số đoạn theo nhóm link (NIDDK, CDC, MedlinePlus) và PDF.
  3. RAG <-> KB bằng code: với mỗi câu hỏi in nút khớp, các câu Cypher đã chạy (MATCH ... MENTIONS), đoạn tài liệu tìm được kèm link/PDF.
  4. Hỏi đáp có trích dẫn: gọi AskEngine (chế độ trích đoạn, không cần Qwen) để thấy câu trả lời cuối cùng và nguồn.
Ghi kết quả vào project01/artifacts/demo_kb_neo4j.json (để báo cáo trích số liệu thật).

NÓI ĐƠN GIẢN - cách trình bày với thầy (khoảng 3 phút), chạy file này rồi chỉ vào từng khối in ra:
  "Thầy xem, KB nằm trên Neo4j: 41 khái niệm, 29 nguồn (28 trang web + 1 PDF), 514 đoạn tài liệu, 123 liên kết bằng chứng."   (khối 1, 2)
  "Em hỏi 1 câu, RAG đi qua KB: tìm khái niệm khớp -> chạy 2 câu Cypher lên Neo4j -> lấy đoạn bằng chứng -> xếp hạng."        (khối 3)
  "Kết quả là đoạn trích có link nguồn, có cả đoạn lấy từ file PDF (nhãn [PDF])."                                           (khối 3, 4)
  Nếu thầy hỏi "không có mạng thì sao?" -> chạy lại với --file: vẫn ra kết quả giống, chỉ không có dòng Cypher.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = ["Which test diagnoses prediabetes?", "Xét nghiệm A1C dùng để làm gì?",   # trúng các trang web NIDDK
             "What are the four steps to manage diabetes?", "How often should I get routine care for diabetes?"]  # trúng đoạn từ file PDF


def main() -> int:
    from project01.mas.ask import AskEngine
    from project01.mas.kb import KnowledgeBase

    # KnowledgeBase() tự thử kết nối Neo4j (KB_BACKEND=auto). Truyền "file" để ép dùng kg.json (không cần mạng).
    kb = KnowledgeBase("file" if "--file" in sys.argv else None)
    out: dict = dict(backend=kb.backend, note=kb.backend_note)
    print(f"\n=== Backend đồ thị tri thức: {kb.backend.upper()} " + (f"({kb.neo.host})" if kb.neo else f"({kb.backend_note or 'kg.json'})"))

    print("\n=== 1. KB_diabet trên Neo4j")
    if kb.neo:
        for label in ("KBNode", "KBSource", "KBPassage"):
            n = kb.neo._run(f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]
            print(f"  (:{label}): {n}")
            out[label] = n
        n = kb.neo._run("MATCH (:KBNode)-[r:MENTIONS]->(:KBPassage) RETURN count(r) AS c")[0]["c"]
        out["MENTIONS"] = n
        print(f"  [:MENTIONS]: {n}")
        print("  Cypher mẫu 1 - triệu chứng của bệnh tiểu đường:")
        for r in kb.neo._run("MATCH (s:KBNode)-[:SYMPTOM_OF]->(:KBNode {id:'diabetes'}) RETURN s.vi AS vi ORDER BY vi"):
            print("     -", r["vi"])
        print("  Cypher mẫu 2 - bằng chứng của 'Xét nghiệm A1C' (kèm link nguồn):")
        for r in kb.neo._run("MATCH (:KBNode {id:'a1c'})-[m:MENTIONS]->(p:KBPassage)<-[:HAS_PASSAGE]-(s:KBSource) "
                             "RETURN m.rank AS rank, p.id AS id, s.url AS url ORDER BY rank"):
            print(f"     #{r['rank']} {r['id']}  {r['url']}")
    else:
        print("  (đang dùng file nên bỏ qua; đặt NEO4J_* hoặc file Neo4j-*.txt rồi chạy lại để xem phần này)")

    # Khối 2: đếm đoạn theo nguồn, chứng minh kho dữ liệu gồm cả 3 nhóm link (NIDDK, CDC, MedlinePlus) lẫn PDF.
    print("\n=== 2. Nguồn dữ liệu (3 nhóm link + PDF)")
    by = Counter()
    for p in kb.passages:
        by["PDF NIDDK" if p["type"] == "pdf" else p["org"]] += 1
    pages = Counter(p["url"] for p in kb.passages)
    for k, v in by.items():
        print(f"  {k:12s} {v:4d} đoạn")
    print(f"  Tổng {len(kb.passages)} đoạn từ {len(pages)} nguồn")
    out["passages_by_source"], out["passages"], out["sources"] = dict(by), len(kb.passages), len(pages)

    print("\n=== 3 + 4. RAG đi qua KB (mỗi câu hỏi) và câu trả lời có trích dẫn")
    # Khối 3 + 4: với mỗi câu hỏi, in nút KB khớp, số câu Cypher đã chạy (kb.neo.queries tăng lên), đoạn tìm được (web/PDF) và câu trả lời cuối.
    engine = AskEngine(type("C", (), {"kb": kb})())
    out["questions"] = []
    for q in QUESTIONS:
        q0 = kb.neo.queries if kb.neo else 0
        found = engine.retriever.search(q, 3, use_dense=False)  # BM25 + đồ thị (nhanh, không cần tải mô hình vector)
        ans = engine.ask(q, mode="retrieval")
        used = (kb.neo.queries - q0) if kb.neo else 0
        print(f"\n  Câu hỏi: {q}")
        print(f"   - nút KB khớp: {', '.join(kb.nodes[n]['vi'] for n in found['seeds']) or '(không)'}")
        if kb.neo:
            print(f"   - số câu Cypher đã chạy trên Neo4j cho câu này: {used}; câu gần nhất: {kb.neo.last_cypher[:110]}... ({kb.neo.last_ms} ms)")
        for h in found["hits"]:
            kind = "PDF" if h["url"].lower().endswith(".pdf") else "web"
            print(f"   - [{kind}] {h['id']}  {h['url']}")
        print(f"   - trả lời ({ans.get('mode')}): {ans['answer'][:230]}...")
        out["questions"].append(dict(question=q, seeds=found["seeds"], cypher_queries=used, hits=[h["id"] for h in found["hits"]],
                                     backend=kb.backend, citations=[c["url"] for c in ans["citations"]]))
    if kb.neo:
        kb.neo.close()
    with open(os.path.join(ROOT, "project01", "artifacts", "demo_kb_neo4j.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # thoát ngay: luồng nạp mô hình vector chạy nền (torch) có thể làm chương trình treo lúc tắt
