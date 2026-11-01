"""Agent tri thức: đồ thị tri thức (KB) + RAG trích dẫn tài liệu NIDDK.

- KB: đồ thị nút/cạnh về tiểu đường (artifacts/kg.json), có thể nạp vào Neo4j bằng artifacts/kg_neo4j.cypher.
- RAG: BM25 trên 183 đoạn tài liệu NIDDK (project01/artifacts/diabetes_corpus_niddk183.jsonl, bản đóng băng của kho 183 đoạn). Graph-RAG mở rộng truy vấn bằng các nút liên quan
  trong đồ thị trước khi truy xuất, rồi trả về đoạn trích có nguồn.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS = os.path.join(ROOT, "project01", "artifacts", "diabetes_corpus_niddk183.jsonl")
KG_PATH = os.path.join(ROOT, "project01", "artifacts", "kg.json")
STOP = set("the a an of to and or in is are for with you your can may be it as that this on by from at have has if not more most".split())


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", strip_accents(text)) if t not in STOP and len(t) > 1]


def load_passages() -> list[dict]:
    rows = []
    for line in open(CORPUS, encoding="utf-8"):
        r = json.loads(line)
        text = r["text"].strip()
        if len(text) >= 60 and not re.match(r"^\[\d+\]", text) and "service of the National Institute" not in text:
            rows.append(dict(id=r["id"], text=text, url=r["source_url"]))
    return rows


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokens(d["text"])) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(docs)
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
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
        sc = self.scores(tokens(query))
        if boost:
            sc = [s + boost.get(d["id"], 0.0) for s, d in zip(sc, self.docs)]
        # bỏ các đoạn bị cụt (bắt đầu bằng chữ thường, thường là mục gạch đầu dòng) khi chọn đoạn để trích dẫn
        order = [i for i in sorted(range(len(sc)), key=lambda i: -sc[i]) if not self.docs[i]["text"][0].islower()][:k]
        return [dict(id=self.docs[i]["id"], quote=self.docs[i]["text"], url=self.docs[i]["url"], score=round(sc[i], 3)) for i in order if sc[i] > 0]


class KnowledgeBase:
    def __init__(self):
        self.passages = load_passages()
        self.bm25 = BM25(self.passages)
        self.by_id = {p["id"]: p for p in self.passages}
        with open(KG_PATH, encoding="utf-8") as f:
            kg = json.load(f)
        self.nodes = {n["id"]: n for n in kg["nodes"]}
        self.edges = kg["edges"]
        self.adj = defaultdict(set)
        for e in self.edges:
            self.adj[e["source"]].add(e["target"])
            self.adj[e["target"]].add(e["source"])
        self.alias_tokens = {nid: set(tokens(" ".join([n["vi"], n["label"]] + n.get("aliases", [])))) for nid, n in self.nodes.items()}

    def graph(self) -> dict:
        return dict(nodes=[{k: n[k] for k in ("id", "label", "vi", "type", "desc")} for n in self.nodes.values()], edges=self.edges)

    def match_nodes(self, query: str, limit: int = 6) -> list[str]:
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
        seeds = self.match_nodes(query) if graph else []
        nodes = self.expand(seeds)[:14] if seeds else []
        boost = {}
        for nid in nodes:  # đoạn gắn với nút khớp truy vấn được cộng 1,0; đoạn gắn với nút láng giềng được cộng 0,3
            weight = 1.0 if nid in seeds else 0.3
            for pid in self.nodes[nid].get("passages", []):
                boost[pid] = boost.get(pid, 0) + weight
        passages = self.bm25.search(query, k, boost if graph else None)
        facts = []
        for e in self.edges:
            if e["source"] in seeds or e["target"] in seeds:
                a, b = self.nodes[e["source"]]["vi"], self.nodes[e["target"]]["vi"]
                facts.append(f"{a} — {REL_VI.get(e['rel'], e['rel'])} — {b}")
        return dict(nodes=nodes, seeds=seeds, passages=passages, facts=facts[:8])


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

REL_VI = {"is_a": "là một dạng của", "raises_risk": "làm tăng nguy cơ", "symptom_of": "là triệu chứng của", "diagnosed_by": "được chẩn đoán bằng",
          "prevented_by": "có thể phòng ngừa bằng", "treated_by": "được kiểm soát bằng", "complication_of": "là biến chứng của", "causes": "gây ra",
          "monitors": "được theo dõi bằng", "helps": "giúp cải thiện"}
