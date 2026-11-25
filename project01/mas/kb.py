"""Agent tri thức: đồ thị tri thức (KB) + RAG trích dẫn tài liệu NIDDK.

- KB: đồ thị nút/cạnh về tiểu đường KB_diabet. Nguồn thật nằm trên NEO4J (nạp từ db_diabet_cypher/*.cypher); artifacts/kg.json là bản dự phòng khi không có mạng.
- RAG: BM25 trên kho 514 đoạn = 492 đoạn từ 28 trang web (NIDDK/CDC/MedlinePlus) + 22 đoạn từ 1 file PDF (NIDDK "4 Steps to Manage Your Diabetes for Life").
  Graph-RAG mở rộng truy vấn bằng các nút liên quan trong đồ thị (truy vấn Cypher khi có Neo4j) trước khi truy xuất, rồi trả về đoạn trích có nguồn.

=====================  BẢN ĐỒ FILE (thầy hỏi "KB / RAG nằm ở đâu" thì mở file này)  =====================
Thuộc agent: "Thủ thư tri thức" (id="knowledge", vai trò KB + RAG) trong hệ 7 agent của Project 01.

Thành phần chính (tìm theo tên):
  - strip_accents(), tokens()   : chuẩn hóa văn bản (bỏ dấu tiếng Việt, tách từ, bỏ stop-word) để so khớp không phân biệt dấu.
  - load_rows(), load_passages(): nạp kho 514 đoạn (492 web + 22 PDF), lọc đoạn quá ngắn / đoạn tham khảo / đoạn chân trang.
  - class BM25                  : bộ truy xuất BM25 tự cài (không dùng thư viện): __init__ tính TF, độ dài, IDF; scores() chấm điểm; search() lấy top-k.
  - class KnowledgeBase         : đồ thị tri thức + RAG.
        .backend       -> "neo4j" (đồ thị đọc từ Neo4j, mỗi câu hỏi chạy Cypher) hoặc "file" (kg.json). Chọn bằng KB_BACKEND (mas/neo4j_kb.py).
        .graph()       -> trả toàn bộ nút/cạnh cho giao diện vẽ đồ thị (API /api/kg).
        .match_nodes() -> tìm nút khớp câu hỏi (bước 1 của Graph-RAG).
        .expand()      -> mở rộng sang nút láng giềng 1 bước (bước 2 của Graph-RAG); với Neo4j là một câu Cypher MATCH (n)--(m).
        .evidence()    -> các đoạn tài liệu mà nút MENTIONS (với Neo4j là một câu Cypher MATCH (n)-[:MENTIONS]->(p)).
        .search()      -> Graph-RAG hoàn chỉnh: nút khớp -> mở rộng -> BM25 có cộng điểm -> đoạn trích + sự kiện đồ thị (API /api/kg/search).
        .passages_for()-> (gắn vào lớp ở cuối file) lấy đoạn trích cho tập nút khi hội chẩn.
  - REL_VI                      : từ điển dịch tên quan hệ trong đồ thị sang tiếng Việt để in thành câu.

Dữ liệu đọc:  Neo4j (hoặc artifacts/kg.json: 41 nút, 54 cạnh), data/chapter04/diabetes_corpus.jsonl (492 đoạn web),
              artifacts/diabetes_pdf_passages.jsonl (22 đoạn PDF). Không ghi file nào.
Được gọi từ:  mas/orchestrator.py (Clinic.__init__ tạo KnowledgeBase; hội chẩn gọi kb.nodes / kb.edges / kb.passages_for),
              server.py (/api/kg, /api/kg/search), mas/critic.py (dùng lại lớp BM25 + ROOT để chạy trên corpus 492 đoạn),
              train/eval_rag.py (đo Hit@1, Hit@3, MRR của BM25 so với Graph-RAG; kết quả ở artifacts/rag_metrics.json).
Lưu ý: văn bản đoạn tài liệu (BM25) đọc từ file jsonl cho nhanh, còn ĐỒ THỊ (nút, cạnh, bằng chứng MENTIONS) đọc/truy vấn từ Neo4j. Neo4j cũng có nút KBPassage
       nên cùng một id đoạn xuất hiện ở cả hai nơi và có thể kiểm tra chéo.

--------------------  KB NỐI VỚI RAG Ở ĐÂU?  (tìm bằng Ctrl+F theo tên hàm)  --------------------
  KnowledgeBase.search()      : Graph-RAG. Chỗ KB (đồ thị) tác động lên RAG (BM25) là vòng lặp "for nid in nodes: ... boost[pid] += weight":
                                đoạn gắn với nút khớp câu hỏi +1,0; đoạn gắn với nút láng giềng +0,3, rồi bm25.search(query, k, boost).
  KnowledgeBase.match_nodes() : bước 1 - câu hỏi -> nút đồ thị.   KnowledgeBase.expand() : bước 2 - nút -> láng giềng 1 bước.
  _passages_for()             : cùng ý tưởng nhưng dùng khi hội chẩn (orchestrator.py gọi kb.passages_for).
  Trường "passages" trong mỗi nút của kg.json = danh sách id đoạn NIDDK gắn với nút đó (do train/build_kg.py gắn bằng BM25) -> đây là "dây nối" KB-RAG.
  Bản hỏi đáp tự do (tab "Hỏi bác sĩ") nối KB với RAG ở mas/ask.py -> Retriever.build_query() (dùng kb.match_nodes lấy alias tiếng Anh làm từ khóa mở rộng).

--------------------  NGUỒN DỮ LIỆU / ĐƯỜNG LINK LẤY DB  --------------------
  3 NHÓM link + 1 PDF, tất cả là tài liệu công khai của chính phủ Mỹ:
    (1) NIDDK        https://www.niddk.nih.gov/health-information/diabetes/overview/  (20 trang: what-is-diabetes, symptoms-causes, tests-diagnosis, managing-diabetes, ...)
    (2) CDC          https://www.cdc.gov/diabetes/                                    (7 trang)
    (3) MedlinePlus  https://medlineplus.gov/hyperglycemia.html                       (1 trang)
    (4) PDF          NIDDK "4 Steps to Manage Your Diabetes for Life" (data/project01/niddk_4_steps_manage_diabetes.pdf, 20 trang -> 22 đoạn;
                     dựng bằng project01/train/build_pdf_corpus.py)
  Kho web 492 đoạn dựng bằng scripts/chapter04/build_diabetes_corpus.py; danh sách URL từng trang nằm trong biến PAGES của file đó và trong
    data/chapter04/diabetes_corpus_manifest.json. Mỗi đoạn trong jsonl có trường "source_url" trỏ về trang/PDF gốc.
  KB_diabet trên Neo4j: thư mục db_diabet_cypher/ (do train/build_kg.py sinh ra); đẩy lên Aura bằng: python scripts/push_to_neo4j.py --only kb.
=============================================================================================================
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

from project01.mas.neo4j_kb import connect as neo4j_connect

# ROOT = thư mục gốc repo (đi lên 2 cấp từ project01/mas). critic.py import ROOT này để tìm data/chapter04/diabetes_corpus.jsonl.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS = os.path.join(ROOT, "data", "chapter04", "diabetes_corpus.jsonl")                 # 492 đoạn từ 28 trang web
PDF_CORPUS = os.path.join(ROOT, "project01", "artifacts", "diabetes_pdf_passages.jsonl")  # 22 đoạn từ file PDF NIDDK
KG_PATH = os.path.join(ROOT, "project01", "artifacts", "kg.json")
# Stop-word tiếng Anh bị loại khỏi truy vấn/đoạn văn (corpus NIDDK là tiếng Anh) để BM25 không bị nhiễu bởi từ phổ biến.
STOP = set("the a an of to and or in is are for with you your can may be it as that this on by from at have has if not more most".split())


def strip_accents(s: str) -> str:
    """Chữ thường + bỏ dấu tiếng Việt (đ -> d) bằng chuẩn hóa Unicode NFD rồi loại ký tự dấu (category "Mn").
    Đầu vào: chuỗi bất kỳ. Đầu ra: chuỗi không dấu. Dùng để so khớp "tiểu đường" với "tieu duong"."""
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokens(text: str) -> list[str]:
    """Tách văn bản thành danh sách từ: bỏ dấu, lấy chuỗi chữ-số, bỏ stop-word và từ 1 ký tự. Đây là bước "tokenize" của BM25."""
    return [t for t in re.findall(r"[a-z0-9]+", strip_accents(text)) if t not in STOP and len(t) > 1]


def load_rows() -> list[dict]:
    """Đọc NGUYÊN các dòng của kho tài liệu dùng chung (492 đoạn web rồi 22 đoạn PDF); mỗi dòng là dict đủ trường (id, text, title, section, source_url, ...)."""
    rows = []
    for path in (CORPUS, PDF_CORPUS):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def load_passages() -> list[dict]:
    """Nạp corpus RAG (kho 514 đoạn). Lọc: đoạn < 60 ký tự, đoạn trích dẫn tham khảo dạng "[12]", và đoạn chân trang
    "service of the National Institute". Trả về list dict(id, text, url, title, section, org, type) để BM25 lập chỉ mục."""
    rows = []
    for r in load_rows():
        text = r["text"].strip()
        if len(text) >= 60 and not re.match(r"^\[\d+\]", text) and "service of the National Institute" not in text:
            rows.append(dict(id=r["id"], text=text, url=r["source_url"], title=r.get("title", ""), section=r.get("section", ""),
                             org=r.get("source_org", ""), type=r.get("source_type", "web")))
    return rows


class BM25:
    """Truy xuất BM25 (Okapi BM25) tự cài đặt, dùng cho RAG.
    score(q, d) = Σ_{w∈q} IDF(w) · f(w,d)·(k1+1) / ( f(w,d) + k1·(1 − b + b·|d|/avgdl) )
      - f(w,d): số lần từ w xuất hiện trong đoạn d (TF); |d|: độ dài đoạn; avgdl: độ dài trung bình các đoạn.
      - k1 = 1.4: mức bão hòa TF (từ lặp nhiều lần chỉ tăng điểm tới một mức).
      - b = 0.75: mức chuẩn hóa theo độ dài (đoạn dài bị phạt nhẹ để không thắng chỉ vì dài).
      - IDF(w) = ln(1 + (N − df + 0.5)/(df + 0.5)): từ hiếm có trọng số cao hơn từ phổ biến."""

    def __init__(self, docs: list[dict], k1: float = 1.4, b: float = 0.75):
        """Lập chỉ mục: đếm TF từng đoạn, độ dài, độ dài trung bình và IDF của mọi từ."""
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokens(d["text"])) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(docs)
        df = Counter(w for t in self.tf for w in t)  # df(w) = số đoạn chứa từ w
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        """Chấm điểm BM25 của truy vấn (đã tokenize) với TỪNG đoạn; trả về danh sách điểm cùng thứ tự self.docs."""
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for w in query_tokens:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out

    def search(self, query: str, k: int = 3, boost: dict[str, float] | None = None) -> list[dict]:
        """Lấy top-k đoạn cho câu hỏi. boost = {id đoạn: điểm cộng thêm} do Graph-RAG truyền vào (đoạn gắn với nút đồ thị liên quan).
        Trả về list dict(id, quote = nguyên văn đoạn, url = nguồn, score); chỉ giữ đoạn có điểm > 0."""
        sc = self.scores(tokens(query))
        if boost:
            sc = [s + boost.get(d["id"], 0.0) for s, d in zip(sc, self.docs)]
        # bỏ các đoạn bị cụt (bắt đầu bằng chữ thường, thường là mục gạch đầu dòng) khi chọn đoạn để trích dẫn
        order = [i for i in sorted(range(len(sc)), key=lambda i: -sc[i]) if not self.docs[i]["text"][0].islower()][:k]
        return [dict(id=self.docs[i]["id"], quote=self.docs[i]["text"], url=self.docs[i]["url"], score=round(sc[i], 3)) for i in order if sc[i] > 0]


class KnowledgeBase:
    """Đồ thị tri thức tiểu đường (KB) + Graph-RAG. Một đối tượng dùng chung cho cả hệ (tạo trong Clinic.__init__)."""

    def __init__(self, backend: str | None = None):
        """Nạp corpus + dựng chỉ mục BM25; nạp đồ thị từ NEO4J (nếu kết nối được) hoặc từ kg.json; dựng dict nút (self.nodes), danh sách cạnh (self.edges)
        và danh sách kề vô hướng (self.adj); alias_tokens = tập từ khóa (tên VN, tên EN, biệt danh) của từng nút.
        backend: "auto" | "neo4j" | "file" (mặc định lấy biến môi trường KB_BACKEND, nếu không có là "auto")."""
        self.passages = load_passages()
        self.bm25 = BM25(self.passages)
        self.by_id = {p["id"]: p for p in self.passages}
        self.neo = neo4j_connect(backend)  # Neo4jKB hoặc None (file). Chế độ "neo4j" mà lỗi thì ném ngoại lệ; "auto" tự rơi về file
        self.backend_note = ""             # lý do rơi về file giữa chừng (nếu có), để hiện cho người dùng
        kg = None
        if self.neo is not None:
            try:
                nodes, edges = self.neo.load_graph()
                kg = dict(nodes=nodes, edges=edges)
            except Exception as exc:  # mất mạng giữa chừng: dùng bản file thay vì làm hỏng hệ
                self.backend_note = f"Neo4j lỗi ({type(exc).__name__}), dùng kg.json"
                self.neo = None
        if kg is None:
            with open(KG_PATH, encoding="utf-8") as f:
                kg = json.load(f)
        self.nodes = {n["id"]: n for n in kg["nodes"]}
        self.edges = kg["edges"]
        self.adj = defaultdict(set)  # đồ thị vô hướng: cạnh nào cũng đi được cả hai chiều khi mở rộng
        for e in self.edges:
            self.adj[e["source"]].add(e["target"])
            self.adj[e["target"]].add(e["source"])
        self.alias_tokens = {nid: set(tokens(" ".join([n["vi"], n["label"]] + n.get("aliases", [])))) for nid, n in self.nodes.items()}

    @property
    def backend(self) -> str:
        """"neo4j" nếu đang truy vấn Neo4j, ngược lại "file"."""
        return "neo4j" if self.neo is not None else "file"

    def _fallback(self, exc: Exception) -> None:
        """Mất kết nối khi đang chạy: ghi lý do và chuyển hẳn sang file (đồ thị đã nạp sẵn trong bộ nhớ nên không mất gì).
        Nói đơn giản: "cầu chì". Đang demo mà rớt mạng thì hệ không báo lỗi mà lặng lẽ dùng bản sao trong bộ nhớ, kết quả vẫn đúng."""
        self.backend_note = f"Neo4j lỗi giữa chừng ({type(exc).__name__}), dùng đồ thị trong bộ nhớ"
        self.neo = None

    def evidence(self, node_ids: list[str]) -> dict[str, list[str]]:
        """Id các đoạn tài liệu gắn với từng nút (quan hệ MENTIONS). Neo4j: chạy Cypher; file: đọc trường "passages" của nút."""
        if self.neo is not None:
            try:
                return self.neo.evidence(node_ids)
            except Exception as exc:
                self._fallback(exc)
        return {nid: list(self.nodes[nid].get("passages", [])) for nid in node_ids if nid in self.nodes}

    def graph(self) -> dict:
        """Toàn bộ đồ thị (id, label, vi, type, desc cho nút; nguyên cạnh) để giao diện vẽ — endpoint GET /api/kg."""
        return dict(nodes=[{k: n[k] for k in ("id", "label", "vi", "type", "desc")} for n in self.nodes.values()], edges=self.edges)

    def match_nodes(self, query: str, limit: int = 6) -> list[str]:
        """Bước 1 Graph-RAG — tìm các nút đồ thị khớp câu hỏi. Điểm = số từ trùng + 2 nếu cả cụm tên/biệt danh xuất hiện trong câu
        (so khớp sau khi bỏ dấu). Trả về tối đa `limit` id nút, điểm cao trước.
        Nói đơn giản: "câu hỏi này đang nói về khái niệm nào?". Vd "Which test diagnoses prediabetes?" -> nút "Tiền tiểu đường", "Xét nghiệm A1C"...
        Hạn chế (nên nói thật nếu thầy hỏi): khớp theo từ khóa nên có lúc nhầm, vd từ "high" trong "high blood sugar" khớp nhầm nút "Huyết áp cao"."""
        q = set(tokens(query))
        qs = strip_accents(query)
        scored = []
        for nid, toks in self.alias_tokens.items():
            hit = len(q & toks)
            n = self.nodes[nid]
            phrase = any(strip_accents(a) in qs for a in [n["vi"], n["label"]] + n.get("aliases", []) if len(a) > 3)
            if hit or phrase:
                scored.append((hit + (2 if phrase else 0), nid))
        scored.sort(reverse=True)
        return [nid for _, nid in scored[:limit]]

    def expand(self, seeds: list[str], hops: int = 1) -> list[str]:
        """Bước 2 Graph-RAG — duyệt rộng (BFS) từ các nút hạt giống `seeds` ra `hops` bước (mặc định 1) theo self.adj.
        Trả về seeds đứng đầu, sau đó là nút láng giềng (sắp xếp theo id để kết quả ổn định).
        Nói đơn giản: từ khái niệm tìm được, đi thêm MỘT bước theo mũi tên để lấy các khái niệm "hàng xóm" (vd Tiền tiểu đường -> A1C, FPG, ăn uống lành mạnh).
        Có Neo4j thì bước đi này là một câu Cypher; không có thì dùng danh sách kề self.adj trong bộ nhớ. Hai cách cho kết quả giống hệt nhau."""
        seen, frontier = list(seeds), set(seeds)
        for _ in range(hops):
            nxt = set()
            if self.neo is not None and frontier:  # Neo4j: một câu Cypher lấy láng giềng của cả tập nút
                try:
                    nxt = set(self.neo.neighbors(sorted(frontier)))
                except Exception as exc:
                    self._fallback(exc)
            if self.neo is None:
                for s in frontier:
                    nxt |= self.adj[s]
            for s in sorted(nxt - set(seen)):
                seen.append(s)
            frontier = nxt
        return seen

    def search(self, query: str, k: int = 3, graph: bool = True) -> dict:
        """Graph-RAG: tìm nút khớp truy vấn -> mở rộng 1 bước -> BM25 trên câu hỏi, cộng điểm cho đoạn gắn với các nút đó.
        Nói đơn giản - 4 việc, mỗi việc 1 dòng code bên dưới:
          (1) match_nodes: câu hỏi nói về khái niệm nào?   (2) expand: lấy thêm khái niệm hàng xóm.
          (3) evidence: các khái niệm đó có bằng chứng ở đoạn tài liệu nào? (Cypher MENTIONS nếu có Neo4j) -> cộng điểm cho các đoạn đó
              (nút khớp +1,0; nút hàng xóm +0,3).   (4) BM25 chấm điểm từ khóa + điểm cộng -> lấy top-k đoạn kèm link nguồn.
        graph=False = tắt (1)-(3), chỉ còn BM25 thuần: dùng để so sánh "có đồ thị" với "không đồ thị" (train/eval_rag.py)."""
        # Thứ tự: (1) seeds = nút khớp câu hỏi; (2) mở rộng 1 bước, giữ tối đa 14 nút; (3) tính boost; (4) BM25 + boost; (5) gom "sự kiện" từ cạnh.
        # graph=False tắt toàn bộ phần đồ thị -> chính là BM25 thuần, dùng làm đối chứng khi đo (train/eval_rag.py).
        seeds = self.match_nodes(query) if graph else []
        nodes = self.expand(seeds)[:14] if seeds else []
        boost = {}
        ev = self.evidence(nodes) if nodes else {}  # bằng chứng của các nút: Cypher MATCH (n)-[:MENTIONS]->(p) khi dùng Neo4j
        for nid in nodes:  # đoạn gắn với nút khớp truy vấn được cộng 1,0; đoạn gắn với nút láng giềng được cộng 0,3
            weight = 1.0 if nid in seeds else 0.3
            for pid in ev.get(nid, []):
                boost[pid] = boost.get(pid, 0) + weight
        passages = self.bm25.search(query, k, boost if graph else None)
        facts = []  # câu sự kiện dạng "A — quan hệ — B" cho mọi cạnh chạm vào nút khớp (tối đa 8 câu trả về)
        for e in self.edges:
            if e["source"] in seeds or e["target"] in seeds:
                a, b = self.nodes[e["source"]]["vi"], self.nodes[e["target"]]["vi"]
                facts.append(f"{a} — {REL_VI.get(e['rel'], e['rel'])} — {b}")
        return dict(nodes=nodes, seeds=seeds, passages=passages, facts=facts[:8], backend=self.backend)


# Hàm này được GẮN vào KnowledgeBase ở dòng `KnowledgeBase.passages_for = _passages_for` bên dưới (đặt tên ngoài lớp
# để tách khỏi Graph-RAG phục vụ tìm kiếm; orchestrator.py gọi self.kb.passages_for(...) khi hội chẩn).
def _passages_for(self, node_ids: list[str], k: int = 3) -> list[dict]:
    """Truy xuất đoạn trích NIDDK cho một tập nút (dùng khi hội chẩn): ưu tiên đoạn đã gắn với nút, rồi BM25 theo từ khóa tiếng Anh."""
    boost, words = {}, []
    for rank, nid in enumerate(node_ids):
        node = self.nodes[nid]
        words.extend(node.get("aliases_en", [node["label"]]))
        for pid in node.get("passages", [])[:2]:
            boost[pid] = boost.get(pid, 0) + 3.0 / (1 + 0.15 * rank)
    return self.bm25.search(" ".join(words), k, boost)


KnowledgeBase.passages_for = _passages_for

# Tên quan hệ trong kg.json -> câu tiếng Việt, dùng khi in "sự kiện" đồ thị (kb.search) và trong báo cáo hội chẩn (orchestrator.py).
REL_VI = {"is_a": "là một dạng của", "raises_risk": "làm tăng nguy cơ", "symptom_of": "là triệu chứng của", "diagnosed_by": "được chẩn đoán bằng",
          "prevented_by": "có thể phòng ngừa bằng", "treated_by": "được kiểm soát bằng", "complication_of": "là biến chứng của", "causes": "gây ra",
          "monitors": "được theo dõi bằng", "helps": "giúp cải thiện"}
