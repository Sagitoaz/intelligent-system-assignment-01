"""Biến file PDF nguồn của KB tiểu đường thành các đoạn (passage) để RAG và Neo4j dùng chung với kho 492 đoạn từ web.

Chạy: python project01/train/build_pdf_corpus.py   (không cần mạng nếu PDF đã nằm trong data/project01/)

Nguồn PDF (tài liệu công khai của chính phủ Mỹ, NIDDK/NIH, 20 trang):
    "4 Steps to Manage Your Diabetes for Life"
    https://www.niddk.nih.gov/-/media/Files/Health-Information/Health-Professionals/Diabetes/health-care-professionals/4StepsToManageDiabetes-English_508.pdf
File được lưu ở data/project01/niddk_4_steps_manage_diabetes.pdf; nếu thiếu, script tự tải (User-Agent chung, không gửi thông tin cá nhân).

Đọc : data/project01/niddk_4_steps_manage_diabetes.pdf (PyMuPDF trích chữ từng trang).
Ghi : project01/artifacts/diabetes_pdf_passages.jsonl - mỗi dòng 1 đoạn, cùng lược đồ với data/chapter04/diabetes_corpus.jsonl
      (id, text, title, section, source_url, source_org, domain, language, accessed) cộng thêm "source_type": "pdf" và "page": số trang.
Tách đoạn: gom các dòng của mỗi trang thành đoạn ~55-150 từ, không cắt ngang câu; bỏ đầu/chân trang lặp lại và đoạn quá ngắn.
Được dùng bởi: mas/kb.py (load_rows), mas/ask.py (load_corpus), train/build_kg.py (xuất Cypher có nút nguồn PDF).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PDF_URL = ("https://www.niddk.nih.gov/-/media/Files/Health-Information/Health-Professionals/Diabetes/"
           "health-care-professionals/4StepsToManageDiabetes-English_508.pdf")
PDF_PATH = os.path.join(ROOT, "data", "project01", "niddk_4_steps_manage_diabetes.pdf")
OUT = os.path.join(ROOT, "project01", "artifacts", "diabetes_pdf_passages.jsonl")
TITLE = "4 Steps to Manage Your Diabetes for Life"
MIN_WORDS, MAX_WORDS = 40, 150
SKIP_TEXT = re.compile(r"(provided as a service|Step 1: Learn about diabetes \d+ Step 2)", re.I)  # chân trang nhà xuất bản và mục lục
NOISE = re.compile(r"(^\d+$|^national diabetes education program|^1-800|^www\.|^ndep|^niddk|^nih publication|^\W+$)", re.I)


def ensure_pdf() -> None:
    """Tải PDF về nếu chưa có (kiểm tra đúng là PDF bằng 4 byte đầu)."""
    if os.path.isfile(PDF_PATH):
        return
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    req = urllib.request.Request(PDF_URL, headers={"User-Agent": "Mozilla/5.0 (educational coursework)"})
    data = urllib.request.urlopen(req, timeout=90).read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError("URL không trả về file PDF")
    with open(PDF_PATH, "wb") as f:
        f.write(data)


def clean(text: str) -> str:
    """Gộp khoảng trắng, đổi ký tự gạch đầu dòng lạ của PDF (thường thành �) thành dấu gạch ngang, nối từ bị ngắt dòng."""
    text = text.replace("�", "-").replace("•", "-").replace("\xa0", " ")
    text = re.sub(r"-\s*\n\s*", "", text) if re.search(r"[a-z]-\n[a-z]", text) else text
    return re.sub(r"\s+", " ", text).strip()


def page_paragraphs(page) -> list[str]:
    """Trả về các khối chữ của một trang theo thứ tự đọc (PyMuPDF 'blocks'), bỏ khối nhiễu."""
    out = []
    for b in sorted(page.get_text("blocks"), key=lambda b: (round(b[1] / 8), b[0])):
        t = clean(b[4]) if b[6] == 0 else ""
        if t and not NOISE.search(t) and len(t.split()) >= 3:
            out.append(t)
    return out


def main() -> int:
    import pymupdf

    sys.stdout.reconfigure(encoding="utf-8")

    ensure_pdf()
    doc = pymupdf.open(PDF_PATH)
    accessed = time.strftime("%Y-%m-%d")
    rows = []
    for pno, page in enumerate(doc, 1):
        buf: list[str] = []

        def flush():
            text = " ".join(buf).strip()
            if len(text.split()) >= MIN_WORDS and not SKIP_TEXT.search(text):
                rows.append(dict(id=f"niddk-4steps-pdf:{len(rows)}", text=f"{TITLE} - trang {pno}: {text}", title=TITLE, section=f"Trang {pno}",
                                 source_url=PDF_URL, domain="diabetes", language="en", accessed=accessed, source_org="NIDDK",
                                 source_type="pdf", page=pno))
            buf.clear()

        for para in page_paragraphs(page):
            if len(" ".join(buf + [para]).split()) > MAX_WORDS:
                flush()
            buf.append(para)
        flush()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    words = [len(r["text"].split()) for r in rows]
    print(f"{len(doc)} trang PDF -> {len(rows)} đoạn ({min(words)}-{max(words)} từ), ghi {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
