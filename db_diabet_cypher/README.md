# db_diabet_cypher - KB_diabet (đồ thị tri thức tiểu đường) dạng Cypher cho Neo4j

Thư mục này là **KB_diabet** của Project 01: toàn bộ đồ thị tri thức, nguồn tài liệu và bằng chứng, viết bằng Cypher.
Ba file **do `project01/train/build_kg.py` sinh ra** (đừng sửa tay: sửa danh sách nút/cạnh `N`, `E` trong script rồi chạy lại).

| File | Nội dung | Số lượng |
|---|---|---|
| `01_kb_concepts.cypher` | `(:KBNode)` khái niệm y khoa (bệnh, triệu chứng, xét nghiệm, yếu tố nguy cơ, biến chứng, điều trị, lối sống) và quan hệ giữa chúng (`IS_A`, `RAISES_RISK`, `DIAGNOSED_BY`, ...) | 41 nút, 54 cạnh |
| `02_sources_passages.cypher` | `(:KBSource)` trang web hoặc PDF; `(:KBPassage)` từng đoạn văn bản nguyên văn; `(KBSource)-[:HAS_PASSAGE]->(KBPassage)` | 29 nguồn (28 trang web + 1 PDF), 514 đoạn (492 web + 22 PDF) |
| `03_evidence_links.cypher` | `(KBNode)-[:MENTIONS {rank}]->(KBPassage)`: đoạn tài liệu làm bằng chứng cho khái niệm (rank 1 = khớp nhất theo BM25) | 123 liên kết |

## Nguồn tài liệu (3 nhóm link + 1 PDF, đều là tài liệu công khai của chính phủ Mỹ)

1. NIDDK: https://www.niddk.nih.gov/health-information/diabetes/overview/ (20 trang)
2. CDC: https://www.cdc.gov/diabetes/ (7 trang)
3. MedlinePlus: https://medlineplus.gov/hyperglycemia.html (1 trang)
4. PDF NIDDK "4 Steps to Manage Your Diabetes for Life" (`data/project01/niddk_4_steps_manage_diabetes.pdf`, 20 trang -> 22 đoạn)

## Nạp vào Neo4j

```bash
# cách 1: script của project (dùng file Neo4j-*.txt mới nhất ở thư mục gốc hoặc biến NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD/NEO4J_DATABASE)
python scripts/push_to_neo4j.py --only kb

# cách 2: cypher-shell, theo thứ tự 01 -> 02 -> 03
cypher-shell -a neo4j+s://<id>.databases.neo4j.io -u <user> -p <password> -d <database> -f 01_kb_concepts.cypher
cypher-shell ... -f 02_sources_passages.cypher
cypher-shell ... -f 03_evidence_links.cypher
```

Mỗi file tự xóa dữ liệu cũ của chính nó (`DETACH DELETE`) nên chạy lại nhiều lần an toàn.

## Truy vấn mẫu (dán vào Neo4j Browser / Aura Query)

```cypher
// 1. Toàn bộ đồ thị khái niệm
MATCH (a:KBNode)-[r]->(b:KBNode) RETURN a, r, b;

// 2. Triệu chứng của bệnh tiểu đường
MATCH (s:KBNode)-[:SYMPTOM_OF]->(d:KBNode {id:'diabetes'}) RETURN s.vi AS trieu_chung, s.desc AS mo_ta;

// 3. Yếu tố nguy cơ của tiểu đường típ 2
MATCH (f:KBNode)-[:RAISES_RISK]->(:KBNode {id:'type2'}) RETURN f.vi AS yeu_to_nguy_co;

// 4. Bằng chứng: các đoạn tài liệu (kèm link nguồn) cho khái niệm "Xét nghiệm A1C"
MATCH (:KBNode {id:'a1c'})-[m:MENTIONS]->(p:KBPassage)<-[:HAS_PASSAGE]-(s:KBSource)
RETURN m.rank AS hang, p.id AS doan, s.org AS nguon, s.url AS link, left(p.text, 160) AS trich_doan ORDER BY hang;

// 5. Các đoạn lấy từ file PDF
MATCH (s:KBSource {source_type:'pdf'})-[:HAS_PASSAGE]->(p:KBPassage) RETURN s.url, p.id, p.section, left(p.text, 120) LIMIT 10;

// 6. Đường đi từ yếu tố nguy cơ tới biến chứng (tối đa 3 bước)
MATCH path = (:KBNode {id:'overweight'})-[*..3]->(c:KBNode {type:'complication'}) RETURN path LIMIT 10;
```

## RAG nối với KB này như thế nào

`project01/mas/neo4j_kb.py` mở kết nối tới Neo4j; `project01/mas/kb.py` (lớp `KnowledgeBase`) nạp đồ thị từ Neo4j khi khởi động
và, **mỗi câu hỏi**, chạy hai câu Cypher: láng giềng của các nút khớp câu hỏi (`MATCH (n)--(m)`) và đoạn bằng chứng của chúng
(`MATCH (n)-[:MENTIONS]->(p)`). Các đoạn bằng chứng được cộng điểm trước khi xếp hạng BM25 + vector (`project01/mas/ask.py`).
Biến `KB_BACKEND`: `auto` (mặc định, thử Neo4j rồi rơi về file), `neo4j` (bắt buộc), `file` (chỉ dùng `project01/artifacts/kg.json`, không cần mạng).
Chạy thử có in câu Cypher: `python project01/train/demo_kb_neo4j.py`.
