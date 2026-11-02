"""Kiểm thử xuất báo cáo PDF của Phòng khám AI."""
import pytest

from project01.mas import pdf_report
from project01.mas.data import PERSONAS
from project01.mas.orchestrator import Clinic


@pytest.fixture(scope="module")
def consult():
    return Clinic().consult(PERSONAS[2]["profile"])


@pytest.fixture(scope="module")
def pdf(consult):
    return pdf_report.build_pdf(dict(summary=consult["summary"], report=consult["report"], events=consult["events"]))


def test_pdf_header_size_and_pages(pdf):
    assert pdf.startswith(b"%PDF")
    assert 40_000 < len(pdf) < 8_000_000
    assert pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") + pdf.count(b"/Type/Page") >= 2 or b"/Count" in pdf


def test_pdf_text_has_vietnamese_and_disclaimer(pdf):
    fitz = pytest.importorskip("pymupdf")
    d = fitz.open(stream=pdf, filetype="pdf")
    assert d.page_count >= 2
    for page in d:
        assert "Chỉ để học tập, không phải chẩn đoán y khoa" in page.get_text()
    text = "\n".join(p.get_text() for p in d)
    for s in ("Bác Hùng", "Mức nguy cơ", "Cờ đỏ", "Lời khuyên", "Nguồn trích dẫn"):
        assert s in text
    # bảng không tràn lề phải của trang A4
    for page in d:
        for b in page.get_text("blocks"):
            assert b[2] <= page.rect.width - 20


def test_route_returns_pdf_tuple_and_rejects_bad_input(consult):
    _, post = pdf_report.routes(None, None)
    res = post["/api/report/pdf"](dict(summary=consult["summary"], report=consult["report"], patient_name="Bác Hùng"))
    assert res[0] == "application/pdf" and res[1].startswith(b"%PDF") and "bao-cao-kham.pdf" in res[2]
    with pytest.raises(ValueError):
        post["/api/report/pdf"]({})
    with pytest.raises(ValueError):
        post["/api/report/pdf"](dict(summary=consult["summary"], report=dict(figures=[dict(title="x", caption="y", png_base64="AAAA")])))
