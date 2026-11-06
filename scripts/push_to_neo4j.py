"""Đẩy toàn bộ đồ thị tri thức của project lên Neo4j (Aura hoặc local).

Nguồn dữ liệu:
  1. knowledge_graph/cypher/diabetes_knowledge_graph.cypher  -> hệ thống/mô hình tiểu đường (domain=diabetes_assignment_01)
  2. project01/artifacts/kg_neo4j.cypher                     -> KG khái niệm tiểu đường của Project 01 (nhãn KBNode)
  3. knowledge_graph/chapter04/*.json.gz                     -> đồ thị chương 4: tài liệu NIDDK (bệnh án 130 BV nạp bởi push_datasets_to_neo4j.py)

Kết nối: đọc biến môi trường NEO4J_URI/NEO4J_USER|NEO4J_USERNAME/NEO4J_PASSWORD/NEO4J_DATABASE,
hoặc file Neo4j-*.txt ở thư mục gốc (file Aura tải về), hoặc backend/.env. Không bao giờ in mật khẩu.

Idempotent: dùng MERGE, chạy lại nhiều lần an toàn. Dữ liệu chương 4 mang nhãn Ch4Node + nhãn theo loại,
thuộc tính `graph` cho biết thuộc đồ thị nào, `uid` = graph|id (duy nhất).

Chạy:  python scripts/push_to_neo4j.py [--only cypher|project01|ch4] [--reset-ch4]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
BATCH = 1000
GRAPHS = {
    "diabetes_documents_graph.json.gz": "ch4_diabetes_docs",
}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_config() -> dict[str, str]:
    cfg: dict[str, str] = {}
    for p in [ROOT / "backend" / ".env", *map(Path, glob.glob(str(ROOT / "Neo4j-*.txt"))), ROOT / "neo4j.txt"]:
        cfg.update(_read_env_file(p))
    cfg.update({k: v for k, v in os.environ.items() if k.startswith("NEO4J_")})
    user = cfg.get("NEO4J_USER") or cfg.get("NEO4J_USERNAME") or "neo4j"
    return {
        "uri": cfg["NEO4J_URI"],
        "user": user,
        "password": cfg["NEO4J_PASSWORD"],
        "database": cfg.get("NEO4J_DATABASE") or "neo4j",
    }


def run_cypher_file(session, path: Path, split_lines: bool = False) -> int:
    text = path.read_text(encoding="utf-8")
    if split_lines:  # file một câu lệnh/dòng, kết thúc bằng ';'
        stmts = [ln.strip().rstrip(";") for ln in text.splitlines() if ln.strip() and not ln.startswith("//")]
    else:
        stmts = [s.strip() for s in re.sub(r"(?m)^\s*//.*$", "", text).split(";\n") if s.strip().rstrip(";")]
        stmts = [s.rstrip(";") for s in stmts]
    for s in stmts:
        session.run(s).consume()
    return len(stmts)


def clean_props(node: dict) -> dict:
    props = {}
    for k, v in node.items():
        if k in ("id", "type"):
            continue
        props[k] = v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, ensure_ascii=False)
    return {k: v for k, v in props.items() if v is not None}


def push_ch4(session, only_reset: bool) -> dict[str, tuple[int, int]]:
    session.run("CREATE CONSTRAINT ch4_uid IF NOT EXISTS FOR (n:Ch4Node) REQUIRE n.uid IS UNIQUE").consume()
    result = {}
    for fname, graph in GRAPHS.items():
        if only_reset:
            session.run("MATCH (n:Ch4Node {graph:$g}) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS", g=graph).consume()
        data = json.load(gzip.open(ROOT / "knowledge_graph" / "chapter04" / fname, "rt", encoding="utf-8"))
        by_type: dict[str, list[dict]] = {}
        for n in data["nodes"]:
            by_type.setdefault(n["type"], []).append(
                {"uid": f"{graph}|{n['id']}", "id": n["id"], "props": clean_props(n)}
            )
        for t, rows in by_type.items():
            assert IDENT.match(t), t
            for i in range(0, len(rows), BATCH):
                session.run(
                    f"UNWIND $rows AS r MERGE (n:Ch4Node {{uid:r.uid}}) SET n:{t}, n.id=r.id, n.graph=$g, n += r.props",
                    rows=rows[i : i + BATCH], g=graph,
                ).consume()
        by_rel: dict[str, list[dict]] = {}
        for e in data["edges"]:
            by_rel.setdefault(e["relation"], []).append(
                {"s": f"{graph}|{e['source']}", "t": f"{graph}|{e['target']}"}
            )
        for rel, rows in by_rel.items():
            assert IDENT.match(rel), rel
            for i in range(0, len(rows), BATCH):
                session.run(
                    f"UNWIND $rows AS r MATCH (a:Ch4Node {{uid:r.s}}), (b:Ch4Node {{uid:r.t}}) MERGE (a)-[:{rel}]->(b)",
                    rows=rows[i : i + BATCH],
                ).consume()
        result[graph] = (len(data["nodes"]), len(data["edges"]))
        print(f"  {graph}: {len(data['nodes'])} nút, {len(data['edges'])} cạnh đã nạp", flush=True)
    return result


def report(session) -> None:
    print("\n== Kiểm tra trên Neo4j ==")
    q = {
        "System KG (domain=diabetes_assignment_01)": "MATCH (n {domain:'diabetes_assignment_01'}) RETURN count(n) AS c",
        "Project01 KBNode": "MATCH (n:KBNode) RETURN count(n) AS c",
    }
    for k, v in q.items():
        print(f"  {k}: {session.run(v).single()['c']} nút")
    for r in session.run("MATCH (n:Ch4Node) RETURN n.graph AS g, count(n) AS c ORDER BY g"):
        rels = session.run("MATCH (a:Ch4Node {graph:$g})-[r]->() RETURN count(r) AS c", g=r["g"]).single()["c"]
        print(f"  {r['g']}: {r['c']} nút, {rels} cạnh")
    tot = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    totr = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
    print(f"  TỔNG: {tot} nút, {totr} cạnh (Aura Free giới hạn 200k/400k)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cypher", "project01", "ch4"])
    ap.add_argument("--reset-ch4", action="store_true", help="xóa đồ thị chương 4 trước khi nạp lại")
    a = ap.parse_args()
    cfg = load_config()
    print(f"Kết nối {cfg['uri']} (database={cfg['database']}, user={cfg['user']})")
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    driver.verify_connectivity()
    with driver.session(database=cfg["database"]) as s:
        if a.only in (None, "cypher"):
            n = run_cypher_file(s, ROOT / "knowledge_graph" / "cypher" / "diabetes_knowledge_graph.cypher")
            print(f"[1] diabetes_knowledge_graph.cypher: {n} câu lệnh")
        if a.only in (None, "project01"):
            n = run_cypher_file(s, ROOT / "project01" / "artifacts" / "kg_neo4j.cypher", split_lines=True)
            print(f"[2] project01 kg_neo4j.cypher: {n} câu lệnh")
        if a.only in (None, "ch4"):
            print("[3] Đồ thị chương 4:")
            push_ch4(s, a.reset_ch4)
        report(s)
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
