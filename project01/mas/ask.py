"""Agent tri thức trả lời câu hỏi tự do về bệnh tiểu đường, có trích dẫn nguồn (NIDDK / CDC / MedlinePlus).

Kho tri thức: data/chapter04/diabetes_corpus.jsonl (492 đoạn tiếng Anh, 28 trang). Truy hồi: BM25 (cùng cài đặt với kb.py) trên truy vấn
đã được mở rộng bằng đồ thị tri thức (alias tiếng Việt -> từ khóa tiếng Anh của nút). Câu hỏi tiếng Việt được chuyển sang từ khóa tiếng Anh
bằng bảng thuật ngữ (GLOSSARY của src/chapter04/rag.py đảo chiều + bảng bổ sung bên dưới), không cần mô hình.

Sinh câu trả lời có hai chế độ:
- "qwen": Qwen2.5-1.5B-Instruct (ChatGenerator của Chương 4/5, nạp bản int8 tiết kiệm bộ nhớ qua scripts/chapter05/lowmem_llm.py).
  Nạp lười, một tiến trình, một khóa: không bao giờ chạy hai lượt sinh cùng lúc. Quá ngân sách thời gian hoặc lỗi bộ nhớ thì rơi về "retrieval".
- "retrieval": trích các câu liên quan nhất từ đoạn tìm được, kèm [1], [2] ánh xạ tới đoạn và đường dẫn nguồn. Luôn nhanh, không bịa.

================================ BẢN ĐỒ FILE (đọc phần này trước) ================================
Chức năng của phòng khám: tab "Hỏi bác sĩ" (RAG hỏi đáp tự do về tiểu đường, có trích dẫn). Giao diện: static/ask.js + ask.css.
Endpoint (đăng ký qua routes() ở cuối file; server.py nạp mô-đun "ask" trong danh sách EXTENSIONS):
    POST /api/ask         -> post_ask()  -> AskEngine.ask()      (câu hỏi + hồ sơ tùy chọn + mode auto|qwen|retrieval)
    GET  /api/ask/status  -> get_status()                        (Qwen đã sẵn sàng chưa, vector đã nạp chưa, số đoạn tài liệu)
Dữ liệu đọc: data/chapter04/diabetes_corpus.jsonl (492 đoạn); models/chapter04/embedding (MiniLM); results/chapter04/diabetes_embeddings.npy
             (cache vector); models/chapter05/llm (Qwen2.5-1.5B). Không ghi file nào.
Luồng một lượt hỏi (AskEngine.ask):
    1) làm sạch câu hỏi -> 2) chặn câu CẤP CỨU / CÁ NHÂN (liều thuốc, chẩn đoán) -> 3) thử Qwen (có ngân sách thời gian)
    -> 4) kiểm tra câu trả lời Qwen có bám tài liệu không (answer_problem) -> 5) nếu không đạt/không có Qwen: trích câu nguyên văn
    (extractive_answer) -> 6) gắn lời nhắc "chỉ để học tập" + danh sách trích dẫn.
Các thành phần chính (tìm số dòng bằng Ctrl+F tên hàm):
    TERMS / vi_to_en_terms()  : bảng thuật ngữ Việt->Anh (dịch câu hỏi KHÔNG dùng mô hình dịch)
    PERSONAL_RX, EMERGENCY_RX : luật an toàn chặn câu hỏi chẩn đoán/liều thuốc và dấu hiệu cấp cứu
    load_corpus(), Retriever  : nạp corpus; truy hồi hybrid BM25 (+mở rộng bằng đồ thị tri thức kb.py) + vector, hợp nhất bằng RRF
    QwenRunner                : chạy LLM Qwen int8 một luồng, có khóa và ngân sách thời gian
    answer_problem()          : "kiểm chứng bám nguồn" (chống bịa) cho câu trả lời của Qwen
    extractive_answer()       : phương án dự phòng, trích nguyên văn
    AskEngine                 : ghép tất cả; refusal() tạo câu từ chối an toàn
Thầy hỏi "BM25 nằm ở đâu?" -> class BM25 ở kb.py (import ở đầu file này), được dùng trong Retriever.search().
Thầy hỏi "chống bịa/hallucination ở đâu?" -> answer_problem() + chế độ retrieval + từ chối "nohit" trong AskEngine.ask().
================================================================================================
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import threading
import time

from project01.mas.kb import BM25, KnowledgeBase, strip_accents, tokens

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS = os.path.join(ROOT, "data", "chapter04", "diabetes_corpus.jsonl")
LLM_PATH = os.path.join(ROOT, "models", "chapter05", "llm")
QWEN_BUDGET = float(os.environ.get("ASK_QWEN_BUDGET", "60"))  # giây; vượt thì rơi về chế độ trích đoạn
DISCLAIMER = "Thông tin chỉ để học tập, không thay thế bác sĩ."

_IMPORT_LOCK = threading.Lock()


def import_ml() -> None:  # nạp torch/transformers an toàn đa luồng; được gọi từ Retriever._load_dense và QwenRunner._gen
    """Nạp torch/transformers trong một khóa: hai luồng cùng import transformers lần đầu có thể gây lỗi ImportError (nạp lười chạy đua)."""
    with _IMPORT_LOCK:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: F401


# --- bảng thuật ngữ Việt -> Anh (khóa không dấu; khớp cụm dài trước) -----------------------------------------------------------------
# VÌ SAO có bảng này: câu hỏi tiếng Việt phải thành từ khóa tiếng Anh để khớp corpus NIDDK (tiếng Anh). Dùng bảng thủ công thay cho mô hình dịch
# vì Qwen int8 dịch Việt-Anh không đáng tin. Khóa viết KHÔNG dấu (đã strip_accents) để khớp cả khi người dùng gõ thiếu dấu.
TERMS = {
    "tieu duong thai ky": "gestational diabetes", "tieu duong tip 1": "type 1 diabetes", "tieu duong tip 2": "type 2 diabetes",
    "tieu duong type 1": "type 1 diabetes", "tieu duong type 2": "type 2 diabetes", "tieu duong loai 1": "type 1 diabetes",
    "tieu duong loai 2": "type 2 diabetes", "tien tieu duong": "prediabetes", "tieu duong": "diabetes", "dai thao duong": "diabetes",
    "tip 1": "type 1", "tip 2": "type 2", "loai 1": "type 1", "loai 2": "type 2",
    "duong huyet thap": "low blood glucose hypoglycemia", "ha duong huyet": "hypoglycemia low blood glucose",
    "duong huyet cao": "high blood glucose hyperglycemia", "tang duong huyet": "hyperglycemia high blood glucose",
    "duong huyet": "blood glucose", "duong trong mau": "blood glucose", "luong duong": "blood sugar", "muc duong": "blood glucose level",
    "nhiem toan ceton": "diabetic ketoacidosis ketones", "ceton": "ketones",
    "xet nghiem a1c": "A1C test", "a1c": "A1C", "hba1c": "A1C", "xet nghiem": "test diagnosis", "chan doan": "diagnosis tests",
    "theo doi lien tuc": "continuous glucose monitoring", "may theo doi duong huyet": "continuous glucose monitor",
    "tuy nhan tao": "artificial pancreas", "tuyen tuy": "pancreas", "tuy": "pancreas",
    "tinh trang khang insulin": "insulin resistance", "khang insulin": "insulin resistance", "insulin": "insulin",
    "benh than": "kidney disease", "than": "kidney", "suy than": "kidney failure kidney disease",
    "vong mac": "retinopathy eye", "bien chung ve mat": "eye problems eye disease", "benh ve mat": "eye disease", "mat": "eye eyes vision",
    "nhin mo": "blurry vision", "thi luc": "vision",
    "benh than kinh": "neuropathy nerve damage", "ton thuong than kinh": "nerve damage neuropathy", "than kinh": "nerve neuropathy",
    "ban chan": "feet foot", "cham soc chan": "foot care feet", "chan": "feet foot", "loet": "foot sores ulcers", "tim mach": "heart disease",
    "nhoi mau co tim": "heart attack", "dot quy": "stroke", "tim": "heart", "huyet ap": "blood pressure", "cholesterol": "cholesterol",
    "mo mau": "cholesterol blood fats",
    "trieu chung": "symptoms signs", "dau hieu": "signs symptoms", "bien chung": "complications problems", "nguyen nhan": "causes",
    "yeu to nguy co": "risk factors", "nguy co": "risk factors", "ai bi": "who is more likely", "phong ngua": "prevent lower chances",
    "ngan ngua": "prevent lower chances", "phong tranh": "prevent", "dieu tri": "treatment medicines manage", "chua khoi": "cure",
    "chua duoc": "cure", "kiem soat": "manage managing", "quan ly": "manage managing", "thuoc": "medicines medicine",
    "uong thuoc": "medicines", "tiem insulin": "insulin shots injections", "metformin": "metformin", "benh nhan": "people with diabetes",
    "khat nuoc": "thirst very thirsty", "khat": "thirsty", "tieu nhieu": "urinate often urination", "di tieu nhieu": "urinate often urination",
    "dai nhieu": "urinate often", "met moi": "tired fatigue", "met": "tired", "sut can": "losing weight weight loss", "giam can": "weight loss losing weight",
    "tang can": "weight gain", "thua can": "overweight weight", "beo phi": "overweight obesity", "can nang": "weight", "doi": "hungry hunger",
    "vet thuong": "sores cuts", "lau lanh": "slow healing", "ngua": "itchy skin",
    "an gi": "eat foods healthy eating", "che do an": "healthy eating meal plan", "an uong": "healthy eating food", "thuc pham": "food foods",
    "do uong": "drinks beverages", "nuoc ngot": "sugary drinks", "ruou": "alcohol", "bia": "alcohol", "carbohydrate": "carbohydrates carbs",
    "tinh bot": "carbohydrates starch", "duong": "sugar", "rau": "vegetables", "trai cay": "fruit", "ngu coc": "whole grains grains",
    "phuong phap dia": "plate method", "bua an": "meal meals", "bua an nhe": "snack", "an nhe": "snack",
    "tap the duc": "physical activity exercise", "van dong": "physical activity", "tap luyen": "physical activity exercise",
    "di bo": "walking physical activity", "the duc": "physical activity", "phut": "minutes", "tuan": "week",
    "thai ky": "pregnancy gestational", "mang thai": "pregnancy pregnant", "tre em": "children kids", "tre": "children",
    "nguoi lon": "adults", "nguoi cao tuoi": "older adults", "benh": "disease",
    "benh nhan": "people", "bao nhieu": "how much how many", "khi nao": "when", "bao lau": "how often", "the nao": "how",
    "tai sao": "why causes", "la gi": "what is", "co nghia la": "what is", "binh thuong": "normal", "muc tieu": "target targets goals",
    "gia tri": "level", "do": "measure", "xuong duoi 70": "below 70", "70": "70",
    "kiem tra": "check", "co quan": "organ", "tao ra": "makes produces", "san xuat": "makes produces", "thuong gap": "common", "xuong duoi": "below",
    "giam nguy co": "lower chances risk", "giam": "reduce lower", "tang": "increase", "tot cho": "healthy", "nen an": "eat", "nen lam gi": "what should I do",
    "lam gi": "what should do", "lam sao": "how can", "cach": "ways how", "kiem soat duong huyet": "manage blood glucose",
    "gay ra": "cause causes", "anh huong": "affect", "co the": "can", "hoc": "",
}
# tên hiển thị tiếng Việt cho chủ đề khi gợi ý
_PHRASES = sorted(TERMS, key=len, reverse=True)  # cụm dài khớp trước ("tieu duong tip 2" trước "tieu duong") để không bị cắt nhỏ sai

# KEEP_EN: từ/ký hiệu tiếng Anh giữ nguyên khi gặp trong câu hỏi (A1C, BMI, mg, dl...); STOP_VI: danh sách từ dừng tiếng Việt không dấu.
KEEP_EN = {"a1c", "cgm", "mg", "dl", "bmi", "ldl", "hdl", "ketones", "insulin", "metformin", "glucose", "type"}
STOP_VI = set("la gi co khong bi cua cho toi minh em anh chi ban the nao nhu nay kia va hay hoac nhung thi se da dang duoc o trong tren voi mot cac nhieu rat".split())

# câu hỏi cần chẩn đoán cá nhân hoặc liều thuốc: không trả lời, chỉ chỉ dẫn đi khám
# LUẬT AN TOÀN số 1: nhận diện câu hỏi "tôi có bị bệnh không?", "uống bao nhiêu mg?", "đổi thuốc..." -> từ chối, hướng dẫn đi khám (xem refusal()).
PERSONAL_PATTERNS = [
    r"\b(toi|minh|em|bo toi|me toi|chong toi|vo toi)\b.{0,40}\b(co bi|co phai bi|bi .{0,20} khong|co mac|mac benh .{0,12} khong)",
    r"\bchan doan (cho|giup|xem)",
    r"\b(lieu|lieu luong|bao nhieu don vi|bao nhieu mg|bao nhieu vien|uong may vien|tiem may don vi|tiem bao nhieu|ke don|ke thuoc|doi thuoc|bo thuoc|ngung thuoc|giam lieu|tang lieu)\b",
    r"\bshould i (take|stop|change|inject|skip)\b",
    r"\bhow (much|many units) (insulin|metformin).{0,30}(should i|do i)",
    r"\b(do i have diabetes|am i diabetic|what dose|how many units|dosage for me|diagnose me|prescribe)\b",
]
PERSONAL_RX = re.compile("|".join("(?:%s)" % x for x in PERSONAL_PATTERNS), re.I)
# LUẬT AN TOÀN số 2: dấu hiệu cấp cứu (hôn mê, co giật, đau ngực...) -> khuyên gọi 115, không trả lời bằng tài liệu.
EMERGENCY_RX = re.compile(r"(hon me|bat tinh|co giat|kho tho|dau nguc|nguy kich|cap cuu|unconscious|seizure|chest pain|emergency)", re.I)


def detect_vi(text: str) -> bool:
    """Đoán câu hỏi có phải tiếng Việt không: có chữ cái có dấu đặc trưng, hoặc chứa từ Việt phổ biến khi gõ không dấu. Dùng để quyết định có dịch bằng TERMS không."""
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", text.lower()):
        return True
    q = " " + strip_accents(text) + " "
    return any(w in q for w in (" toi ", " bi ", " khong ", " tieu duong ", " nhu the nao ", " la gi", " duoc khong"))


def vi_to_en_terms(question: str) -> list[str]:
    """Chuyển câu hỏi tiếng Việt thành danh sách từ khóa tiếng Anh bằng bảng thuật ngữ (khớp cụm dài trước, không chồng lấn)."""
    q = " " + re.sub(r"[^a-z0-9 ]+", " ", strip_accents(question)) + " "
    out: list[str] = []
    for ph in _PHRASES:
        pat = " " + ph + " "
        if pat in q:
            out.extend(TERMS[ph].split())
            q = q.replace(pat, " # ", 1)
    # từ tiếng Anh/số còn sót lại trong câu (ví dụ "A1C", "70 mg/dL")
    for w in q.split():
        if (w.isdigit() or w in KEEP_EN) and w not in out:
            out.append(w)
    return out


# --- kho tri thức + truy hồi ---------------------------------------------------------------------------------------------------------
def load_corpus(path: str = CORPUS) -> list[dict]:
    """Đọc corpus jsonl (492 đoạn); bỏ đoạn quá ngắn (<60 ký tự). Trả về danh sách dict(id, text, url, title, section, org); org là nguồn: NIDDK/CDC/MedlinePlus."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r["text"].strip()
            if len(text) >= 60:
                rows.append(dict(id=r["id"], text=text, url=r["source_url"], title=r.get("title", ""), section=r.get("section", ""), org=r.get("source_org", "")))
    return rows


def split_sentences(text: str) -> list[str]:
    """Tách đoạn thành các câu (cắt sau . ! ? khi câu sau bắt đầu bằng chữ hoa/số) để extractive_answer chọn câu."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


class Retriever:
    """BM25 (có mở rộng bằng đồ thị tri thức) kết hợp vector (MiniLM, cache embedding của Chương 4) bằng RRF, như "hybrid" của Chương 4."""
    DENSE_MIN = 0.30     # cosine tối thiểu của đoạn đầu (câu ngoài chủ đề thường < 0,25; câu về tiểu đường thường > 0,45)
    BM25_MIN_ONLY = 12.0  # khi chưa có vector: điểm BM25 tối thiểu

    def __init__(self, kb: KnowledgeBase | None = None, dense: bool = True):
        """kb: đồ thị tri thức để mở rộng truy vấn (có thể None). dense=True thì nạp mô hình vector ở luồng nền (không chặn khởi động; trong lúc chờ dùng BM25)."""
        self.rows = [json.loads(line) for line in open(CORPUS, encoding="utf-8")]
        self.passages = load_corpus()
        self.bm25 = BM25(self.passages)
        self.kb = kb
        self.by_id = {p["id"]: p for p in self.passages}
        self.dense = None
        self.dense_error: str | None = None
        self.dense_ready = threading.Event()
        if dense and importlib.util.find_spec("torch") is not None:
            threading.Thread(target=self._load_dense, daemon=True).start()
        else:
            self.dense_ready.set()

    def _load_dense(self) -> None:
        """Luồng nền: nạp Encoder MiniLM + DenseRetriever (cache vector .npy). Lỗi thì ghi dense_error và vẫn chạy bằng BM25; luôn set dense_ready khi xong."""
        try:
            if ROOT not in sys.path:
                sys.path.insert(0, ROOT)
            import_ml()
            from src.chapter04.retrieval import DenseRetriever, Encoder
            enc = Encoder(os.path.join(ROOT, "models", "chapter04", "embedding"))
            self.dense = DenseRetriever(self.rows, enc, os.path.join(ROOT, "results", "chapter04", "diabetes_embeddings.npy"))
        except Exception as exc:  # không có mô hình vector: vẫn chạy được bằng BM25
            self.dense_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.dense_ready.set()

    def build_query(self, question: str, english: str | None = None) -> tuple[list[str], list[str], str, list[str]]:
        """Trả về (token BM25, nút đồ thị khớp, câu tiếng Anh cho vector, từ mở rộng từ đồ thị). Từ khóa chính lặp 2 lần; từ mở rộng từ đồ thị lặp 1 lần."""
        # Bước 1: có câu tiếng Anh sẵn thì dùng; nếu là tiếng Việt thì dịch bằng bảng TERMS; nếu đã là tiếng Anh thì giữ nguyên.
        base = english if english else (" ".join(vi_to_en_terms(question)) if detect_vi(question) else question)
        main = tokens(base)
        # Bước 2 (Graph-RAG): khớp câu hỏi với tối đa 3 nút trong đồ thị tri thức rồi lấy alias tiếng Anh của nút làm từ khóa mở rộng.
        seeds = self.kb.match_nodes(question + " " + base, 3) if self.kb else []
        extra: list[str] = []
        for nid in seeds:
            if nid != "diabetes":  # nút chung chung "diabetes" kéo mọi đoạn lên, không giúp phân biệt
                extra += tokens(" ".join(self.kb.nodes[nid].get("aliases_en", [self.kb.nodes[nid]["label"]])))
        return main * 2, seeds, base, [t for t in dict.fromkeys(extra) if t not in main]

    def search(self, question: str, k: int = 3, english: str | None = None, use_dense: bool = True) -> dict:
        """Truy hồi top-k đoạn. Trả về dict(hits=[{id, quote, url, score, bm25, title, section, org}], seeds=nút đồ thị khớp, query, hybrid, relevant).
        relevant=False nghĩa là không đủ liên quan -> AskEngine từ chối thay vì đoán (chống bịa)."""
        qt, seeds, base, extra = self.build_query(question, english)
        if not qt:
            return dict(hits=[], seeds=seeds, query=[], hybrid=False, relevant=False)
        # Nhánh từ khóa: điểm BM25 của từ khóa gốc (đã nhân đôi trọng số ở build_query).
        sc = self.bm25.scores(qt)
        if extra:  # mở rộng truy vấn bằng đồ thị tri thức, trọng số thấp hơn từ khóa gốc
            sc = [a + 0.35 * b for a, b in zip(sc, self.bm25.scores(extra))]
        lex = [i for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:40] if sc[i] > 0 and not self.passages[i]["text"][0].islower()]
        # Nhánh ngữ nghĩa (nếu mô hình vector đã nạp): lấy top 40 theo cosine, rồi hợp nhất với top 40 từ khóa bằng RRF
        # (Reciprocal Rank Fusion: điểm = tổng 1/(60+hạng)). VÌ SAO: BM25 mạnh về từ khóa chính xác, vector mạnh về đồng nghĩa; RRF không cần chỉnh thang điểm.
        dense = self.dense if (use_dense and self.dense is not None) else None
        cos: dict[int, float] = {}
        if dense is not None:
            vec = dense.encoder.encode([base])[0]
            sims = dense.vectors @ vec
            sem = sorted(range(len(sims)), key=lambda i: -float(sims[i]))[:40]
            cos = {i: float(sims[i]) for i in sem}
            rrf: dict[int, float] = {}
            for ranking in (lex, sem):
                for r, i in enumerate(ranking, 1):
                    rrf[i] = rrf.get(i, 0.0) + 1 / (60 + r)
            order = sorted(rrf, key=lambda i: -rrf[i])
            top_cos = float(sims[order[0]]) if order else 0.0
            # Ngưỡng liên quan: cosine tối thiểu DENSE_MIN, hoặc BM25 đủ cao; câu ngoài chủ đề (hỏi về bóng đá...) sẽ bị loại ở đây.
            relevant = bool(order) and max(cos.values()) >= self.DENSE_MIN and (top_cos >= self.DENSE_MIN * 0.8 or (lex and sc[lex[0]] >= 8))
        else:
            order = lex
            relevant = bool(order) and sc[order[0]] >= self.BM25_MIN_ONLY
        hits = []
        for i in order[:k]:
            p = self.passages[i]
            hits.append(dict(id=p["id"], quote=p["text"], url=p["url"], score=round(cos[i], 3) if i in cos else round(sc[i], 3), bm25=round(sc[i], 3),
                             title=p["title"], section=p.get("section", ""), org=p["org"]))
        return dict(hits=hits, seeds=seeds, query=sorted(set(qt)), hybrid=dense is not None, relevant=relevant)


# --- Qwen (một thể hiện duy nhất cho cả tiến trình) ----------------------------------------------------------------------------------
class QwenRunner:
    """Bọc mô hình sinh Qwen2.5-1.5B-Instruct: nạp lười, chỉ một thể hiện, một khóa (không chạy hai lượt cùng lúc để khỏi tràn RAM)."""
    def __init__(self):
        self.gen = None
        self.lock = threading.Lock()   # đúng một lượt sinh tại một thời điểm
        self.broken: str | None = None  # lý do không dùng được (thiếu bộ nhớ, thiếu gói...)
        self.last_seconds: float | None = None

    @staticmethod
    def available() -> bool:
        """True nếu máy có thư mục mô hình (config.json) và đủ gói torch + transformers."""
        return (os.path.isfile(os.path.join(LLM_PATH, "config.json")) and importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None)

    def _gen(self):
        """Tạo ChatGenerator lần đầu (nạp int8 qua scripts/chapter05/lowmem_llm.py để đỉnh RAM ~2,5 GB thay vì 6,5 GB); các lần sau dùng lại."""
        if self.gen is None:
            if ROOT not in sys.path:
                sys.path.insert(0, ROOT)
            import_ml()
            from scripts.chapter05 import lowmem_llm
            from src.chapter04.rag import ChatGenerator
            lowmem_llm.install()  # nạp int8: đỉnh ~2,5 GB thay vì 6,5 GB
            self.gen = ChatGenerator(LLM_PATH, max_new_tokens=130, threads=max(2, (os.cpu_count() or 4) // 2))
        return self.gen

    def run(self, question: str, retriever: Retriever, k: int, box: dict) -> None:
        """Chạy trong luồng nền; ghi kết quả vào box. Dùng acquire không chặn: nếu đang bận thì box['busy']=True và thoát.
        Các khóa box: found (kết quả truy hồi), q_en (truy vấn tiếng Anh), weak (không đủ liên quan), full (câu trả lời Qwen),
        grounded (lý do bị loại bởi answer_problem, rỗng = đạt), error, busy, done. Luồng: tìm đoạn -> nếu liên quan thì Qwen viết lại -> kiểm bám nguồn."""
        if not self.lock.acquire(blocking=False):
            box["busy"] = True
            box["done"] = True
            return
        try:
            gen = self._gen()
            gen.load()
            vi = detect_vi(question)
            # Qwen int8 dịch Việt-Anh không đáng tin (từng dịch thành câu vô nghĩa), nên dùng bảng thuật ngữ để tạo truy vấn tiếng Anh
            q_en = " ".join(vi_to_en_terms(question)) if vi else question
            found = retriever.search(question, k, english=q_en if vi else None)
            box["found"], box["q_en"] = found, q_en
            if not found["relevant"]:
                box["weak"] = True
                return
            full = gen.generate_full(question, [dict(id=h["id"], text=h["quote"]) for h in found["hits"]], "vi" if vi else "en", q_en)
            box["full"] = full
            box["grounded"] = answer_problem(full, [h["quote"] for h in found["hits"]])
        except MemoryError as exc:
            self.broken = "thiếu bộ nhớ"
            box["error"] = repr(exc)
        except Exception as exc:  # lỗi nạp mô hình/bộ nhớ ảo: ghi lại và rơi về chế độ trích đoạn
            if "memory" in repr(exc).lower() or "alloc" in repr(exc).lower():
                self.broken = "thiếu bộ nhớ"
            box["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            box["done"] = True
            self.lock.release()


# --- Hợp nhất ------------------------------------------------------------------------------------------------------------------------
# Cụm câu "xin lỗi/không thể giúp" của LLM: nếu xuất hiện nghĩa là Qwen không dùng tài liệu -> bị loại, chuyển sang trích nguyên văn.
BAD_PHRASES = re.compile(r"(i'?m sorry|i am sorry|could you (please )?(provide|clarify|rephrase)|as an ai|i can'?t provide|ai assistant|i'?m here to help|end of period|không thể trả lời|xin lỗi)", re.I)


def answer_problem(full: dict, quotes: list[str]) -> str:
    """Kiểm tra câu trả lời của Qwen có bám sát tài liệu không; trả về lý do loại (chuỗi rỗng nếu đạt).
    Đây là cơ chế CHỐNG BỊA (hallucination guard): (1) không chứa câu xin lỗi/chung chung, (2) không quá dài, (3) ít nhất 70% từ khóa của câu trả lời
    phải có trong các đoạn tìm được, (4) mọi con số trong câu trả lời phải xuất hiện trong tài liệu (trừ 1, 2, 3 là số thứ tự trích dẫn)."""
    eng = (full.get("english_answer") or "").strip()
    vi = (full.get("answer") or "").strip()
    if full.get("insufficient") or not eng:
        return ""
    if BAD_PHRASES.search(eng) or BAD_PHRASES.search(vi):
        return "câu trả lời chung chung hoặc xin lỗi thay vì dùng tài liệu"
    if len(eng) > 1400 or len(vi) > 1800:
        return "câu trả lời quá dài và lan man"
    pool = set(tokens(" ".join(quotes)))
    words = [w for w in tokens(eng) if not w.isdigit()]
    if not words:
        return "câu trả lời rỗng"
    cover = sum(w in pool for w in words) / len(words)
    nums = set(re.findall(r"\d+(?:[.,]\d+)?", eng)) - set(re.findall(r"\d+(?:[.,]\d+)?", " ".join(quotes)))
    if cover < 0.7:
        return f"chỉ {cover:.0%} từ khóa có trong các đoạn tìm được"
    if nums - {"1", "2", "3"}:
        return "có con số không xuất hiện trong tài liệu"
    return ""


def trim_sentence(text: str) -> str:
    """Câu trả lời bị cắt do hết số từ tối đa thì bỏ phần câu dở dang cuối cùng."""
    text = text.strip()
    if text and text[-1] not in ".!?…":
        cut = max(text.rfind(c) for c in ".!?")
        if cut > len(text) * 0.4:
            return text[: cut + 1]
    return text


def bmi_text(profile: dict) -> str:
    """Tính BMI từ height_cm, weight_kg của hồ sơ; thiếu/sai dữ liệu thì trả chuỗi rỗng."""
    try:
        h, w = float(profile.get("height_cm")), float(profile.get("weight_kg"))
        return f"BMI khoảng {w / (h / 100) ** 2:.1f}"
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


def personal_intro(profile: dict | None) -> str:
    """Câu mở đầu cá nhân hóa nhẹ (tuổi, BMI, huyết áp, hút thuốc) nếu người dùng gửi hồ sơ; luôn nhấn mạnh đây là thông tin chung, không phải lời khuyên riêng."""
    if not profile:
        return ""
    bits = []
    if profile.get("age_years"):
        bits.append(f"{int(profile['age_years'])} tuổi")
    b = bmi_text(profile)
    if b:
        bits.append(b)
    if profile.get("high_bp"):
        bits.append("có huyết áp cao")
    if profile.get("smoker"):
        bits.append("có hút thuốc")
    if not bits:
        return ""
    return f"Với hồ sơ của bạn ({', '.join(bits)}), đây là thông tin chung từ tài liệu, không phải lời khuyên riêng cho bạn."


def body_text(h: dict) -> str:
    """Bỏ phần "Tiêu đề trang — Tên mục" mà bộ chia đoạn gắn ở đầu mỗi đoạn, chỉ giữ nội dung."""
    t, title, section = h["quote"], h.get("title", ""), h.get("section", "")
    for pre in (f"{title} — {section}", title):
        if pre and t.startswith(pre):
            t = t[len(pre):].lstrip(" —:").strip()
            break
    return t or h["quote"]


def extractive_answer(question: str, hits: list[dict], retriever: Retriever) -> str:
    """Chọn câu có nhiều từ khóa truy vấn nhất trong từng đoạn: 2 câu từ đoạn [1], 1 câu từ mỗi đoạn tiếp theo."""
    qt = set(retriever.build_query(question)[0])  # (token, nút, câu) -> lấy token
    parts = []
    for n, h in enumerate(hits, 1):
        body = body_text(h)
        sents = split_sentences(body) or [body]
        scored = sorted(range(len(sents)), key=lambda i: (-len(qt & set(tokens(sents[i]))), i))
        take = sorted(scored[: 2 if n == 1 else 1])
        text = " ".join(sents[i] for i in take)
        if len(text) > 420:
            text = text[:420].rsplit(" ", 1)[0] + "…"
        parts.append(f"{text} [{n}]")
    return "Đây là các đoạn liên quan nhất trong tài liệu (nguyên văn tiếng Anh): " + " ".join(parts)


class AskEngine:
    """Bộ máy hỏi đáp: ghép Retriever (truy hồi) + QwenRunner (sinh) + luật an toàn. Một thể hiện duy nhất cho cả máy chủ (xem routes())."""
    def __init__(self, clinic=None):
        """clinic: đối tượng Clinic của orchestrator.py; lấy clinic.kb (đồ thị tri thức) để Graph-RAG mở rộng truy vấn."""
        kb = getattr(clinic, "kb", None)
        self.retriever = Retriever(kb)
        self.qwen = QwenRunner()

    def refusal(self, kind: str, started: float, profile) -> dict:
        """Tạo phản hồi từ chối an toàn. kind: "personal" (cần bác sĩ), "emergency" (gọi 115), "nohit" (không có đoạn đủ liên quan nên không đoán). Không có trích dẫn."""
        msgs = {
            "personal": "Câu hỏi này cần bác sĩ đánh giá riêng cho bạn (chẩn đoán hoặc liều thuốc), nên mình không trả lời để tránh gây hại. "
                        "Bạn có thể dùng tab \"Khám bệnh\" để xem mức nguy cơ tham khảo, rồi mang kết quả đến gặp bác sĩ.",
            "emergency": "Các dấu hiệu bạn mô tả có thể là cấp cứu. Hãy gọi cấp cứu (115 tại Việt Nam) hoặc đến cơ sở y tế gần nhất ngay, đừng chờ câu trả lời từ công cụ này.",
            "nohit": "Mình chưa tìm thấy đoạn tài liệu nào đủ liên quan để trả lời chắc chắn, nên mình không đoán. Bạn thử hỏi cụ thể hơn về tiểu đường "
                     "(ví dụ triệu chứng, nguyên nhân, xét nghiệm A1C, ăn uống, vận động).",
        }
        return dict(answer=msgs[kind] + " " + DISCLAIMER, citations=[], mode="retrieval", refused=kind, seconds=round(time.perf_counter() - started, 2),
                    notes=["Không trả lời vì " + {"personal": "cần chẩn đoán/liều thuốc cá nhân", "emergency": "có dấu hiệu cấp cứu", "nohit": "không có đoạn đủ liên quan"}[kind] + "."],
                    speak=msgs[kind])

    def ask(self, question: str, profile: dict | None = None, mode: str = "auto") -> dict:
        """Trả lời một câu hỏi. mode: "auto" (thử Qwen rồi dự phòng), "qwen", "retrieval" (chỉ trích đoạn).
        Trả về dict(answer, citations[{id, quote, url, score, title, org}], mode đã dùng, seconds, notes giải thích cho người dùng, speak cho TTS, graph_nodes)."""
        started = time.perf_counter()
        question = re.sub(r"\s+", " ", str(question or "")).strip()[:400]
        if len(question) < 3:
            raise ValueError("Hãy nhập câu hỏi (ít nhất vài chữ)")
        if mode not in ("auto", "qwen", "retrieval"):
            raise ValueError("mode phải là auto, qwen hoặc retrieval")
        # Luật an toàn chạy TRƯỚC mọi truy hồi: cấp cứu và câu hỏi cá nhân bị chặn ngay.
        plain = strip_accents(question)
        if EMERGENCY_RX.search(plain):
            return self.refusal("emergency", started, profile)
        if PERSONAL_RX.search(plain) or PERSONAL_RX.search(question.lower()):
            return self.refusal("personal", started, profile)

        notes: list[str] = []
        k = 3
        answer = None
        used = "retrieval"
        found = None
        # Nhánh 1: thử Qwen trong luồng riêng, chờ tối đa QWEN_BUDGET giây (mặc định 60); mọi trường hợp lỗi/bận/quá giờ đều ghi note và rơi xuống nhánh 2.
        if mode != "retrieval":
            if self.qwen.broken:
                notes.append(f"Đang dùng chế độ trích đoạn (Qwen không chạy được: {self.qwen.broken}).")
            elif not QwenRunner.available():
                notes.append("Đang dùng chế độ trích đoạn (máy chưa có mô hình Qwen hoặc thiếu torch).")
            else:
                box: dict = {}
                t = threading.Thread(target=self.qwen.run, args=(question, self.retriever, k, box), daemon=True)
                t.start()
                t.join(timeout=QWEN_BUDGET)
                if box.get("busy"):
                    notes.append("Đang dùng chế độ trích đoạn (Qwen đang bận với câu hỏi trước).")
                elif not box.get("done"):
                    notes.append(f"Đang dùng chế độ trích đoạn: Qwen chưa xong sau {QWEN_BUDGET:.0f} giây (vẫn đang chạy nền, hỏi lại sau sẽ nhanh hơn).")
                elif box.get("error"):
                    notes.append("Đang dùng chế độ trích đoạn: Qwen gặp lỗi (" + box["error"][:120] + ").")
                elif box.get("weak"):
                    notes.append(f"Qwen dịch câu hỏi thành “{box.get('q_en', '')}” nhưng chưa tìm được đoạn đủ liên quan; đang thử lại bằng bảng thuật ngữ.")
                elif box.get("full") and box.get("grounded"):
                    notes.append(f"Qwen trả lời chưa đạt kiểm tra ({box['grounded']}), nên mình hiển thị trích đoạn nguyên văn thay thế để không đưa thông tin sai.")
                elif box.get("full") and box["full"].get("insufficient"):
                    notes.append("Qwen cho rằng các đoạn tìm được chưa đủ để trả lời chắc chắn, nên hiển thị trích đoạn nguyên văn thay thế.")
                elif box.get("full"):
                    full = box["full"]
                    found = box["found"]
                    nums = "".join(f" [{i}]" for i in range(1, len(found["hits"]) + 1))
                    answer = trim_sentence(full["answer"]) + nums
                    used = "qwen"
                    if box.get("q_en") and box["q_en"] != question:
                        notes.append(f"Qwen dịch câu hỏi sang tiếng Anh để tìm tài liệu: “{box['q_en']}”.")
                    notes.append("Câu trả lời do Qwen2.5-1.5B-Instruct (bản int8, chạy trên máy này) viết lại từ các đoạn tìm được; có thể còn sai sót, hãy đối chiếu trích dẫn.")
                    if full.get("translation_problems"):
                        notes.append("Bản dịch tiếng Việt chưa đạt kiểm tra nên được giữ nguyên văn tiếng Anh.")
        # Nhánh 2 (dự phòng, luôn nhanh và không bịa): trích nguyên văn các câu liên quan nhất. Không đủ liên quan -> từ chối "nohit".
        if used == "retrieval":
            found = self.retriever.search(question, k)
            hits = found["hits"]
            if not found["relevant"]:
                return self.refusal("nohit", started, profile)
            answer = extractive_answer(question, hits, self.retriever)
            if mode == "retrieval" or not notes:
                notes.append("Đang dùng chế độ trích đoạn: các câu liên quan nhất được trích nguyên văn (tiếng Anh) từ tài liệu, không qua mô hình sinh.")
            if not found.get("hybrid"):
                notes.append("Tìm kiếm tạm chỉ dùng từ khóa (BM25) vì mô hình vector chưa nạp xong; kết quả có thể kém chính xác hơn.")
        hits = found["hits"]
        intro = personal_intro(profile)
        if intro:
            answer = intro + " " + answer
        answer += " " + DISCLAIMER
        speak = ("Mình tìm thấy %d đoạn tài liệu liên quan; bạn xem phần trích dẫn bên dưới nhé. " % len(hits) + DISCLAIMER) if used == "retrieval" else answer
        if used == "qwen":
            speak = re.sub(r"\[\d+\]", "", speak)
        return dict(answer=answer, citations=[dict(id=h["id"], quote=h["quote"], url=h["url"], score=h["score"], title=h["title"], org=h["org"]) for h in hits],
                    mode=used, seconds=round(time.perf_counter() - started, 2), notes=notes, speak=speak,
                    graph_nodes=found.get("seeds", []))


_ENGINE: AskEngine | None = None
_ENGINE_LOCK = threading.Lock()


def routes(clinic, speaker):
    """Hook cho server.py: trả về (bảng GET, bảng POST) = ({"/api/ask/status": get_status}, {"/api/ask": post_ask}). speaker (TTS) không dùng ở đây."""
    def engine() -> AskEngine:
        global _ENGINE
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = AskEngine(clinic)
            return _ENGINE

    threading.Thread(target=engine, daemon=True).start()  # dựng sẵn chỉ mục (và nạp mô hình vector nền) ngay khi máy chủ khởi động

    def post_ask(body: dict) -> dict:
        """POST /api/ask, body {question, profile?, mode?} -> kết quả AskEngine.ask()."""
        profile = body.get("profile") if isinstance(body.get("profile"), dict) else None
        return engine().ask(body.get("question", ""), profile, str(body.get("mode", "auto")))

    def get_status() -> dict:
        """GET /api/ask/status: giao diện dùng để hiện trạng thái Qwen/vector (đang nạp hay sẵn sàng) và số đoạn trong kho."""
        e = engine()
        return dict(qwen_available=QwenRunner.available() and not e.qwen.broken, qwen_loaded=bool(e.qwen.gen and e.qwen.gen.ready),
                    passages=len(e.retriever.passages), budget_seconds=QWEN_BUDGET, vector_ready=e.retriever.dense is not None, vector_loading=not e.retriever.dense_ready.is_set())

    return {"/api/ask/status": get_status}, {"/api/ask": post_ask}
