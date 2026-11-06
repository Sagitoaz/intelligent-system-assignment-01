"""Agent tri thức: đồ thị tri thức (KB) + RAG trích dẫn tài liệu NIDDK.

- KB: đồ thị nút/cạnh về tiểu đường (artifacts/kg.json), có thể nạp vào Neo4j bằng artifacts/kg_neo4j.cypher.
- RAG: BM25 trên 183 đoạn tài liệu NIDDK (project01/artifacts/diabetes_corpus_niddk183.jsonl, bản đóng băng của kho 183 đoạn). Graph-RAG mở rộng truy vấn bằng các nút liên quan
  trong đồ thị trước khi truy xuất, rồi trả về đoạn trích có nguồn.

=====================  BẢN ĐỒ FILE (thầy hỏi "KB / RAG nằm ở đâu" thì mở file này)  =====================
Thuộc agent: "Thủ thư tri thức" (id="knowledge", vai trò KB + RAG) trong hệ 7 agent của Project 01.

Thành phần chính (tìm theo tên):
  - strip_accents(), tokens()   : chuẩn hóa văn bản (bỏ dấu tiếng Việt, tách từ, bỏ stop-word) để so khớp không phân biệt dấu.
  - load_passages()             : nạp corpus 183 đoạn NIDDK từ file jsonl, lọc đoạn quá ngắn / đoạn tham khảo / đoạn chân trang.
  - class BM25                  : bộ truy xuất BM25 tự cài (không dùng thư viện): __init__ tính TF, độ dài, IDF; scores() chấm điểm; search() lấy top-k.
  - class KnowledgeBase         : đồ thị tri thức + RAG.
        .graph()       -> trả toàn bộ nút/cạnh cho giao diện vẽ đồ thị (API /api/kg).
        .match_nodes() -> tìm nút khớp câu hỏi (bước 1 của Graph-RAG).
        .expand()      -> mở rộng sang nút láng giềng 1 bước (bước 2 của Graph-RAG).
        .search()      -> Graph-RAG hoàn chỉnh: nút khớp -> mở rộng -> BM25 có cộng điểm -> đoạn trích + sự kiện đồ thị (API /api/kg/search).
        .passages_for()-> (gắn vào lớp ở cuối file) lấy đoạn trích cho tập nút khi hội chẩn.
  - REL_VI                      : từ điển dịch tên quan hệ trong đồ thị sang tiếng Việt để in thành câu.

Dữ liệu đọc:  artifacts/kg.json (41 nút, 54 cạnh), artifacts/diabetes_corpus_niddk183.jsonl (183 đoạn). Không ghi file nào.
Được gọi từ:  mas/orchestrator.py (Clinic.__init__ tạo KnowledgeBase; hội chẩn gọi kb.nodes / kb.edges / kb.passages_for),
              server.py (/api/kg, /api/kg/search), mas/critic.py (dùng lại lớp BM25 + ROOT để chạy trên corpus 492 đoạn),
              train/eval_rag.py (đo Hit@1, Hit@3, MRR của BM25 so với Graph-RAG; kết quả ở artifacts/rag_metrics.json).
Lưu ý: RAG ở đây chạy hoàn toàn trên file, KHÔNG gọi Neo4j. Neo4j chỉ là bản sao đồ thị để xem/truy vấn.
=============================================================================================================
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

# ROOT = thư mục gốc repo (đi lên 2 cấp từ project01/mas). critic.py import ROOT này để tìm data/chapter04/diabetes_corpus.jsonl.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS = os.path.join(ROOT, "project01", "artifacts", "diabetes_corpus_niddk183.jsonl")
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


def load_passages() -> list[dict]:
    """Nạp corpus RAG từ CORPUS (jsonl, mỗi dòng 1 đoạn NIDDK). Lọc: đoạn < 60 ký tự, đoạn trích dẫn tham khảo dạng "[12]",
    và đoạn chân trang "service of the National Institute". Trả về list dict(id, text, url) để BM25 lập chỉ mục."""
    rows = []
    for line in open(CORPUS, encoding="utf-8"):
        r = json.loads(line)
        text = r["text"].strip()
        if len(text) >= 60 and not re.match(r"^\[\d+\]", text) and "service of the National Institute" not in text:
            rows.append(dict(id=r["id"], text=text, url=r["source_url"]))
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

    def __init__(self):
        """Nạp corpus + dựng chỉ mục BM25, nạp kg.json thành dict nút (self.nodes), danh sách cạnh (self.edges)
        và danh sách kề vô hướng (self.adj) để duyệt láng giềng; alias_tokens = tập từ khóa (tên VN, tên EN, biệt danh) của từng nút."""
        self.passages = load_passages()
        self.bm25 = BM25(self.passages)
        self.by_id = {p["id"]: p for p in self.passages}
        with open(KG_PATH, encoding="utf-8") as f:
            kg = json.load(f)
        self.nodes = {n["id"]: n for n in kg["nodes"]}
        self.edges = kg["edges"]
        self.adj = defaultdict(set)  # đồ thị vô hướng: cạnh nào cũng đi được cả hai chiều khi mở rộng
        for e in self.edges:
            self.adj[e["source"]].add(e["target"])
            self.adj[e["target"]].add(e["source"])
        self.alias_tokens = {nid: set(tokens(" ".join([n["vi"], n["label"]] + n.get("aliases", [])))) for nid, n in self.nodes.items()}

    def graph(self) -> dict:
        """Toàn bộ đồ thị (id, label, vi, type, desc cho nút; nguyên cạnh) để giao diện vẽ — endpoint GET /api/kg."""
        return dict(nodes=[{k: n[k] for k in ("id", "label", "vi", "type", "desc")} for n in self.nodes.values()], edges=self.edges)

    def match_nodes(self, query: str, limit: int = 6) -> list[str]:
        """Bước 1 Graph-RAG — tìm các nút đồ thị khớp câu hỏi. Điểm = số từ trùng + 2 nếu cả cụm tên/biệt danh xuất hiện trong câu
        (so khớp sau khi bỏ dấu). Trả về tối đa `limit` id nút, điểm cao trước."""
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
        Trả về seeds đứng đầu, sau đó là nút láng giềng (sắp xếp theo id để kết quả ổn định)."""
        seen, frontier = list(seeds), set(seeds)
        for _ in range(hops):
            nxt = set()
            for s in frontier:
                nxt |= self.adj[s]
            for s in sorted(nxt - set(seen)):
                seen.append(s)
            frontier = nxt
        return seen

    def search(self, query: str, k: int = 3, graph: bool = True) -> dict:
        """Graph-RAG: tìm nút khớp truy vấn -> mở rộng 1 bước -> BM25 trên câu hỏi, cộng điểm cho đoạn gắn với các nút đó."""
        # Thứ tự: (1) seeds = nút khớp câu hỏi; (2) mở rộng 1 bước, giữ tối đa 14 nút; (3) tính boost; (4) BM25 + boost; (5) gom "sự kiện" từ cạnh.
        # graph=False tắt toàn bộ phần đồ thị -> chính là BM25 thuần, dùng làm đối chứng khi đo (train/eval_rag.py).
        seeds = self.match_nodes(query) if graph else []
        nodes = self.expand(seeds)[:14] if seeds else []
        boost = {}
        for nid in nodes:  # đoạn gắn với nút khớp truy vấn được cộng 1,0; đoạn gắn với nút láng giềng được cộng 0,3
            weight = 1.0 if nid in seeds else 0.3
            for pid in self.nodes[nid].get("passages", []):
                boost[pid] = boost.get(pid, 0) + weight
        passages = self.bm25.search(query, k, boost if graph else None)
        facts = []  # câu sự kiện dạng "A — quan hệ — B" cho mọi cạnh chạm vào nút khớp (tối đa 8 câu trả về)
        for e in self.edges:
            if e["source"] in seeds or e["target"] in seeds:
                a, b = self.nodes[e["source"]]["vi"], self.nodes[e["target"]]["vi"]
                facts.append(f"{a} — {REL_VI.get(e['rel'], e['rel'])} — {b}")
        return dict(nodes=nodes, seeds=seeds, passages=passages, facts=facts[:8])


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
