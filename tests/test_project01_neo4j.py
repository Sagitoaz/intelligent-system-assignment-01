"""Kiểm thử KB_diabet trên Neo4j: file Cypher sinh ra đúng và nhất quán với kg.json / kho tài liệu (không cần mạng),
cộng thêm test sống kiểm tra RAG truy vấn Neo4j thật (tự bỏ qua nếu không kết nối được)."""
import json
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from project01.mas import kb as kbmod  # noqa: E402
from project01.mas import neo4j_kb  # noqa: E402

CYPHER = os.path.join(ROOT, "db_diabet_cypher")


def _read(name):
    return open(os.path.join(CYPHER, name), encoding="utf-8").read()


def test_cypher_folder_has_three_ordered_files():
    assert sorted(f for f in os.listdir(CYPHER) if f.endswith(".cypher")) == ["01_kb_concepts.cypher", "02_sources_passages.cypher", "03_evidence_links.cypher"]
    assert os.path.isfile(os.path.join(CYPHER, "README.md"))


def test_concepts_cypher_matches_kg_json():
    kg = json.load(open(kbmod.KG_PATH, encoding="utf-8"))
    text = _read("01_kb_concepts.cypher")
    assert len(re.findall(r"^CREATE \(:KBNode ", text, re.M)) == len(kg["nodes"]) == 41
    assert len(re.findall(r"^MATCH \(a:KBNode .*CREATE \(a\)-\[:", text, re.M)) == len(kg["edges"]) == 54
    assert "aliases_en:[" in text  # RAG cần từ khóa tiếng Anh của nút khi lấy đồ thị từ Neo4j


def test_every_cypher_statement_is_one_line():
    """push_to_neo4j.py đọc theo dòng: mỗi dòng không phải chú thích phải là một câu lệnh kết thúc bằng ';'."""
    for name in os.listdir(CYPHER):
        if name.endswith(".cypher"):
            for line in _read(name).splitlines():
                assert not line.strip() or line.startswith("//") or line.rstrip().endswith(";"), (name, line[:80])


def test_passages_cypher_covers_web_and_pdf_sources():
    rows = kbmod.load_passages()
    text = _read("02_sources_passages.cypher")
    assert len(re.findall(r"CREATE \(:KBSource ", text)) == len({r["url"] for r in rows})
    assert "source_type:'pdf'" in text and ".pdf" in text
    for pid in ("what-is-diabetes:0", "niddk-4steps-pdf:0"):
        assert f"id:'{pid}'" in text
    assert text.count("id:'") >= len(rows)  # mọi đoạn đều có mặt


def test_evidence_links_point_to_existing_passages():
    kg = json.load(open(kbmod.KG_PATH, encoding="utf-8"))
    ids = {p["id"] for p in kbmod.load_passages()}
    linked = {pid for n in kg["nodes"] for pid in n["passages"]}
    assert linked <= ids
    assert _read("03_evidence_links.cypher").count("MENTIONS") >= 1
    assert sum(len(n["passages"]) for n in kg["nodes"]) == len(re.findall(r"\[\'[a-z0-9_]+\', \'[^\']+\', \d\]", _read("03_evidence_links.cypher")))


def test_pdf_passages_are_real_pdf_text():
    rows = [json.loads(x) for x in open(kbmod.PDF_CORPUS, encoding="utf-8")]
    assert 15 <= len(rows) <= 40
    assert all(r["source_type"] == "pdf" and r["source_url"].endswith(".pdf") and len(r["text"].split()) >= 40 for r in rows)
    assert any("insulin" in r["text"].lower() for r in rows)


def test_file_backend_never_touches_network():
    kb = kbmod.KnowledgeBase(backend="file")
    assert kb.backend == "file" and kb.neo is None and len(kb.nodes) == 41
    assert kb.search("What is prediabetes?", 3)["backend"] == "file"


@pytest.fixture(scope="module")
def live():
    try:
        kb = kbmod.KnowledgeBase(backend="neo4j")
    except Exception as exc:  # không có cấu hình / mất mạng / Neo4j rỗng
        pytest.skip(f"Không kết nối được Neo4j: {type(exc).__name__}")
    yield kb
    kb.neo.close()


def test_live_graph_matches_file_graph(live):
    """Đồ thị đọc từ Neo4j trùng khớp kg.json: cùng nút, cùng cạnh, cùng danh sách đoạn bằng chứng."""
    kg = json.load(open(kbmod.KG_PATH, encoding="utf-8"))
    assert live.backend == "neo4j"
    assert set(live.nodes) == {n["id"] for n in kg["nodes"]}
    assert {(e["source"], e["target"], e["rel"]) for e in live.edges} == {(e["source"], e["target"], e["rel"]) for e in kg["edges"]}
    assert all(live.nodes[n["id"]]["passages"] == n["passages"] for n in kg["nodes"])


def test_live_search_runs_cypher_and_matches_file_ranking(live):
    file_kb = kbmod.KnowledgeBase(backend="file")
    before = live.neo.queries
    for q in ("Which test diagnoses prediabetes?", "What causes kidney disease?"):
        a, b = live.search(q, 5), file_kb.search(q, 5)
        assert [p["id"] for p in a["passages"]] == [p["id"] for p in b["passages"]] and a["backend"] == "neo4j"
    assert live.neo.queries > before and "MATCH" in live.neo.last_cypher
