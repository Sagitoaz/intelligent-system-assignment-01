"""Kết nối Neo4j (Aura hoặc local) để RAG của Project 01 truy vấn đồ thị tri thức KB_diabet bằng Cypher.

NÓI ĐƠN GIẢN (đọc phần này trước):
  - Neo4j là "cuốn sổ tay quan hệ": mỗi khái niệm y khoa (vd "Tiền tiểu đường") là một nút, mối liên hệ ("được chẩn đoán bằng") là một mũi tên.
  - Cypher là ngôn ngữ để hỏi Neo4j, giống SQL nhưng cho đồ thị. Ví dụ: MATCH (a)-[:DIAGNOSED_BY]->(b) RETURN a, b = "tìm cặp a, b nối bằng mũi tên DIAGNOSED_BY".
  - File này chỉ làm 1 việc: mở kết nối tới Neo4j rồi chạy 3 câu hỏi Cypher (nạp đồ thị, hỏi láng giềng, hỏi bằng chứng). RAG (mas/kb.py) gọi nó
    để biết "câu hỏi này liên quan khái niệm nào, và đoạn tài liệu nào chứng minh khái niệm đó".
  - Nếu Neo4j hỏng/không có mạng thì file này báo lỗi, KnowledgeBase tự chuyển sang đọc kg.json -> demo không bao giờ chết.

=====================  BẢN ĐỒ FILE (thầy hỏi "RAG nối với KB qua code ở đâu?" thì mở file này)  =====================
Chức năng: lớp mỏng bọc driver `neo4j`. mas/kb.py (KnowledgeBase) gọi các hàm ở đây để:
    - load_graph()  : nạp toàn bộ nút KBNode + cạnh khi khởi động (nguồn thật của đồ thị là Neo4j, kg.json chỉ là bản dự phòng);
    - neighbors()   : mỗi câu hỏi, lấy láng giềng 1 bước của các nút khớp câu hỏi bằng Cypher;
    - evidence()    : mỗi câu hỏi, lấy các đoạn tài liệu (KBPassage) mà nút đó MENTIONS (có kèm nguồn URL/PDF) bằng Cypher;
    - ping()        : kiểm tra kết nối (tối đa ~6 giây) để quyết định dùng Neo4j hay rơi về file.
Cấu hình kết nối (không bao giờ in mật khẩu), theo thứ tự: biến môi trường NEO4J_URI / NEO4J_USERNAME|NEO4J_USER / NEO4J_PASSWORD /
    NEO4J_DATABASE; file Neo4j-*.txt ở thư mục gốc (file Aura tải về, đã chặn trong .gitignore); backend/.env.
Chế độ chọn bằng biến KB_BACKEND: "auto" (mặc định: thử Neo4j, lỗi thì dùng file), "neo4j" (bắt buộc Neo4j), "file" (chỉ dùng kg.json, không mạng).
Lược đồ đồ thị (do train/build_kg.py sinh vào db_diabet_cypher/):
    (:KBNode {id, label, vi, type, desc, aliases, aliases_en}) -[:IS_A|RAISES_RISK|...]-> (:KBNode)
    (:KBNode)-[:MENTIONS {rank}]->(:KBPassage {id, text, title, section, url, org, source_type})
    (:KBSource {url, title, org, source_type})-[:HAS_PASSAGE]->(:KBPassage)
=====================================================================================================================
"""
from __future__ import annotations

import glob
import os
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONNECT_TIMEOUT = float(os.environ.get("NEO4J_TIMEOUT", "6"))  # giây; quá hạn thì rơi về file để demo không bị treo


def _read_env_file(path: str) -> dict[str, str]:
    """Đọc file dạng KEY=VALUE (bỏ dòng trống và dòng #). Trả {} nếu file không tồn tại."""
    out: dict[str, str] = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_config() -> dict[str, str] | None:
    """Gom cấu hình kết nối; trả None nếu thiếu URI hoặc mật khẩu (khi đó KB dùng file).
    Nói đơn giản: địa chỉ + tài khoản + mật khẩu Neo4j không nằm trong code (để không lộ lên git) mà đọc từ file Neo4j-*.txt tải từ Aura.
    Nguồn đọc theo thứ tự, nguồn sau ĐÈ nguồn trước: backend/.env < file Neo4j-*.txt (mới nhất) < biến môi trường."""
    cfg: dict[str, str] = {}
    aura = sorted(glob.glob(os.path.join(ROOT, "Neo4j-*.txt")), key=os.path.getmtime)  # file Aura tải về: file MỚI NHẤT được ưu tiên (instance cũ có thể đã bị xóa)
    for p in [os.path.join(ROOT, "backend", ".env"), *aura, os.path.join(ROOT, "neo4j.txt")]:
        cfg.update(_read_env_file(p))
    if aura:  # tên đăng nhập trong file Aura đi cùng URI/mật khẩu của nó, không để NEO4J_USER cũ trong backend/.env đè lên
        user_in_file = _read_env_file(aura[-1]).get("NEO4J_USERNAME")
        if user_in_file:
            cfg["NEO4J_USER"] = user_in_file
    cfg.update({k: v for k, v in os.environ.items() if k.startswith("NEO4J_")})
    if not cfg.get("NEO4J_URI") or not cfg.get("NEO4J_PASSWORD"):
        return None
    return dict(uri=cfg["NEO4J_URI"], user=cfg.get("NEO4J_USER") or cfg.get("NEO4J_USERNAME") or "neo4j",
                password=cfg["NEO4J_PASSWORD"], database=cfg.get("NEO4J_DATABASE") or "neo4j")


def make_driver(cfg: dict[str, str], **kwargs):
    """Tạo driver Neo4j. Với URI mã hóa (neo4j+s / bolt+s) dùng bộ chứng chỉ gốc của gói certifi thay vì kho chứng chỉ Windows, vì Aura hiện ký bằng
    SSL.com mà nhiều máy Windows chưa có root này ("self-signed certificate in certificate chain"). VẪN xác thực chứng chỉ đầy đủ (không dùng +ssc)."""
    from neo4j import GraphDatabase

    uri = cfg["uri"]                      # vd "neo4j+s://xxxx.databases.neo4j.io": "+s" nghĩa là kết nối được mã hóa (TLS)
    scheme, rest = uri.split("://", 1)
    if scheme.endswith("+s"):             # nói đơn giản: giống "ổ khóa https"; ta chỉ đổi nơi lấy danh sách "nhà cấp chứng chỉ tin cậy", KHÔNG tắt kiểm tra
        try:
            import certifi
            from neo4j import TrustCustomCAs

            uri = f"{scheme[:-2]}://{rest}"
            kwargs.update(encrypted=True, trusted_certificates=TrustCustomCAs(certifi.where()))
        except ImportError:  # không có certifi: dùng kho chứng chỉ hệ thống như mặc định
            pass
    return GraphDatabase.driver(uri, auth=(cfg["user"], cfg["password"]), **kwargs)


class Neo4jKB:
    """Một kết nối Neo4j dùng chung; mọi truy vấn đi qua _run() để đếm số lần gọi và lưu câu Cypher cuối (phục vụ demo).
    Nói đơn giản: giống một "quầy hỏi đáp" tới Neo4j. Có 3 câu hỏi chuẩn: load_graph (lấy cả đồ thị), neighbors (ai là láng giềng), evidence (bằng chứng ở đâu)."""

    def __init__(self, cfg: dict[str, str]):
        self.cfg = cfg  # neo4j được nạp lười trong make_driver: máy không cài neo4j vẫn chạy được bằng file
        self.driver = make_driver(cfg, connection_timeout=CONNECT_TIMEOUT, max_connection_lifetime=300)
        self.queries = 0            # số câu Cypher đã chạy từ lúc khởi động
        self.last_cypher = ""       # câu Cypher chạy gần nhất (hiện trong demo)
        self.last_ms = 0.0
        self._lock = threading.Lock()

    @property
    def host(self) -> str:
        """Tên máy chủ Neo4j (không gồm mật khẩu) để hiện trong giao diện/ghi chú."""
        return self.cfg["uri"].split("://", 1)[-1]

    def _run(self, cypher: str, **params) -> list[dict]:
        """Chạy một câu Cypher chỉ đọc, trả danh sách dict. Lỗi mạng/xác thực sẽ ném ngoại lệ để KB rơi về file."""
        t0 = time.perf_counter()
        with self.driver.session(database=self.cfg["database"]) as s:
            rows = [r.data() for r in s.run(cypher, **params)]
        with self._lock:
            self.queries += 1
            self.last_cypher, self.last_ms = " ".join(cypher.split()), round((time.perf_counter() - t0) * 1000, 1)
        return rows

    def ping(self) -> None:
        """Kiểm tra kết nối VÀ đồ thị đã được nạp (có nút KBNode); ném RuntimeError nếu Neo4j rỗng để KB dùng file thay thế."""
        n = self._run("MATCH (n:KBNode) RETURN count(n) AS c")[0]["c"]
        if n == 0:
            raise RuntimeError("Neo4j chưa có đồ thị KBNode - chạy: python scripts/push_to_neo4j.py --only kb")

    def load_graph(self) -> tuple[list[dict], list[dict]]:
        """Nạp toàn bộ đồ thị: nút (kèm aliases, aliases_en và id các đoạn bằng chứng theo thứ hạng) và cạnh (source, target, rel viết thường)."""
        nodes = self._run(
            "MATCH (n:KBNode) OPTIONAL MATCH (n)-[m:MENTIONS]->(p:KBPassage) "
            "WITH n, p, m ORDER BY m.rank "
            "RETURN n.id AS id, n.label AS label, n.vi AS vi, n.type AS type, n.desc AS desc, "
            "coalesce(n.aliases, []) AS aliases, coalesce(n.aliases_en, []) AS aliases_en, collect(p.id) AS passages ORDER BY id")
        edges = self._run("MATCH (a:KBNode)-[r]->(b:KBNode) RETURN a.id AS source, b.id AS target, toLower(type(r)) AS rel ORDER BY source, target, rel")
        return nodes, edges

    def neighbors(self, ids: list[str]) -> list[str]:
        """Láng giềng 1 bước (cả hai chiều) của các nút `ids`, không trùng, theo thứ tự id.
        Ví dụ: ids=["prediabetes"] -> ["a1c", "fpg", "type2", ...] (các khái niệm nối trực tiếp với "Tiền tiểu đường").
        Cypher: (n)--(m) = "n và m nối nhau bằng BẤT KỲ mũi tên nào, chiều nào"; $ids là tham số truyền vào (an toàn, không ghép chuỗi)."""
        rows = self._run("MATCH (n:KBNode)--(m:KBNode) WHERE n.id IN $ids RETURN DISTINCT m.id AS id ORDER BY id", ids=ids)
        return [r["id"] for r in rows]

    def evidence(self, ids: list[str]) -> dict[str, list[str]]:
        """Với mỗi nút trong `ids`: danh sách id đoạn tài liệu mà nút MENTIONS, theo thứ hạng (1 = khớp nhất).
        Ví dụ: ids=["a1c"] -> {"a1c": ["a1c-test:0", "a1c-test:19", "a1c-test:20"]}.
        Đây chính là "dây nối KB -> RAG": khái niệm trong đồ thị trỏ tới đoạn văn thật (web hoặc PDF) làm bằng chứng. RAG dùng nó để cộng điểm cho đoạn đó."""
        rows = self._run("MATCH (n:KBNode)-[m:MENTIONS]->(p:KBPassage) WHERE n.id IN $ids "
                         "RETURN n.id AS nid, p.id AS pid ORDER BY nid, m.rank", ids=ids)
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["nid"], []).append(r["pid"])
        return out

    def close(self) -> None:
        """Đóng kết nối (gọi khi tắt máy chủ hoặc sau khi chạy demo)."""
        self.driver.close()


def connect(mode: str | None = None) -> Neo4jKB | None:
    """Tạo kết nối theo KB_BACKEND. file -> None. auto -> Neo4jKB nếu kết nối + có dữ liệu, ngược lại None. neo4j -> bắt buộc, lỗi thì ném ngoại lệ.
    Nói đơn giản: trả về None nghĩa là "đừng dùng Neo4j, đọc file". Chế độ auto là cầu chì an toàn: có mạng thì dùng Neo4j, không thì dùng file."""
    mode = (mode or os.environ.get("KB_BACKEND", "auto")).lower()
    if mode == "file":
        return None
    cfg = load_config()
    if cfg is None:
        if mode == "neo4j":
            raise RuntimeError("KB_BACKEND=neo4j nhưng thiếu NEO4J_URI/NEO4J_PASSWORD (đặt biến môi trường hoặc để file Neo4j-*.txt ở thư mục gốc)")
        return None
    try:
        kb = Neo4jKB(cfg)
        kb.ping()
        return kb
    except Exception:
        if mode == "neo4j":
            raise
        return None
