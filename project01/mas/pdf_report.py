"""Xuất báo cáo khám dạng PDF cho bệnh nhân (reportlab, nhúng font DejaVu Sans để hiển thị đủ dấu tiếng Việt).

POST /api/report/pdf  {summary, report:{figures,tables}, events?, patient_name?}  ->  ("application/pdf", bytes, Content-Disposition)
Đây là công cụ học tập: mọi trang đều có dòng cảnh báo "Chỉ để học tập, không phải chẩn đoán y khoa".
"""
from __future__ import annotations

import base64
import binascii
import io
import os
import re
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle)

DISCLAIMER = "Chỉ để học tập, không phải chẩn đoán y khoa"
FONT, FONT_B = "DejaVuSans", "DejaVuSans-Bold"
INK, MUTED, LINE = colors.HexColor("#14203d"), colors.HexColor("#4a5a7c"), colors.HexColor("#d5e0f2")
LEVEL_COLORS = {"thấp": "#0d9f8f", "trung bình": "#e8a317", "cao": "#e8503a"}
RISK_ROWS = [("none", "Không tiểu đường", "#0d9f8f"), ("pre", "Tiền tiểu đường", "#e8a317"), ("diabetes", "Tiểu đường", "#e8503a")]
MAX_FIGS, MAX_TABLES, MAX_ROWS, MAX_PNG = 10, 12, 300, 6_000_000
_fonts_ready = False


def _register_fonts():
    global _fonts_ready
    if _fonts_ready:
        return
    d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont(FONT, os.path.join(d, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_B, os.path.join(d, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_B, italic=FONT, boldItalic=FONT_B)
    _fonts_ready = True


def _styles():
    s = lambda name, **kw: ParagraphStyle(name, fontName=kw.pop("fontName", FONT), textColor=kw.pop("textColor", INK), alignment=TA_LEFT, **kw)
    return dict(
        body=s("body", fontSize=10, leading=14.5, spaceAfter=3),
        h1=s("h1", fontName=FONT_B, fontSize=19, leading=24, spaceAfter=2, textColor=colors.white),
        sub=s("sub", fontSize=10, leading=14, textColor=colors.white),
        h2=s("h2", keepWithNext=1, fontName=FONT_B, fontSize=13, leading=17, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#1b5fb0")),
        h3=s("h3", keepWithNext=1, fontName=FONT_B, fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=3),
        cap=s("cap", fontSize=8.6, leading=12, textColor=MUTED, spaceAfter=8),
        cell=s("cell", fontSize=8.4, leading=11.2),
        cellh=s("cellh", fontName=FONT_B, fontSize=8.4, leading=11.2, textColor=colors.white),
        quote=s("quote", fontSize=8.8, leading=12.5, textColor=colors.HexColor("#2b3a5e"), leftIndent=8),
        bullet=s("bullet", fontSize=10, leading=14.5, leftIndent=14, bulletIndent=2, spaceAfter=3),
    )


_EMOJI = re.compile("[𐀀-􏿿☀-➿️‍]")


def _clean(text) -> str:
    """Font DejaVu không có emoji: bỏ biểu tượng để PDF không hiện ô vuông."""
    return re.sub(r"[ 	]{2,}", " ", _EMOJI.sub("", str(text if text is not None else ""))).strip()


def _p(text, style):
    return Paragraph(escape(_clean(text)), style)


def _patient_name(body: dict) -> str:
    n = str(body.get("patient_name") or "").strip()
    if n:
        return n[:80]
    for e in body.get("events") or []:
        m = re.match(r"Chào (.+?)!", str((e or {}).get("text", "")))
        if m:
            return m.group(1)[:80]
    return "Bệnh nhân"


def _risk_bars(risk: dict, width: float) -> Drawing:
    h = 22
    d = Drawing(width, h * len(RISK_ROWS))
    lab_w, val_w = 108, 44
    bar_w = width - lab_w - val_w - 6
    for i, (k, label, col) in enumerate(RISK_ROWS):
        y = h * (len(RISK_ROWS) - 1 - i)
        v = max(0.0, min(1.0, float(risk.get(k, 0))))
        d.add(String(0, y + 6, label, fontName=FONT_B, fontSize=9.5, fillColor=INK))
        d.add(Rect(lab_w, y + 3, bar_w, 13, rx=6, ry=6, fillColor=colors.HexColor("#e8eefa"), strokeColor=None))
        d.add(Rect(lab_w, y + 3, max(3.0, bar_w * v), 13, rx=6, ry=6, fillColor=colors.HexColor(col), strokeColor=None))
        d.add(String(lab_w + bar_w + 6, y + 6, f"{v * 100:.0f}%" if v >= 0.1 else f"{v * 100:.1f}%", fontName=FONT_B, fontSize=9.5, fillColor=INK))
    return d


def _png_image(b64: str, max_w: float, max_h: float):
    try:
        raw = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError):
        raise ValueError("Hình trong báo cáo không đúng định dạng base64")
    if not raw.startswith(b"\x89PNG") or len(raw) > MAX_PNG:
        raise ValueError("Hình trong báo cáo phải là PNG và không quá lớn")
    iw, ih = ImageReader(io.BytesIO(raw)).getSize()
    k = min(max_w / iw, max_h / ih)
    return Image(io.BytesIO(raw), width=iw * k, height=ih * k)


def _col_widths(columns, rows, total):
    n = len(columns)
    weight = []
    for j in range(n):
        longest = max([len(str(columns[j])) * 0.8] + [len(str(r[j])) if j < len(r) else 0 for r in rows[:60]])
        weight.append(min(60.0, max(8.0, longest)) ** 0.8)
    s = sum(weight)
    w = [max(total * 0.1, total * x / s) for x in weight]
    k = total / sum(w)
    return [x * k for x in w]


def _table(t: dict, st: dict, width: float):
    cols = [str(c) for c in (t.get("columns") or [])]
    rows = [list(r) for r in (t.get("rows") or [])][:MAX_ROWS]
    if not cols:
        return None
    head = [Paragraph(escape(c), st["cellh"]) for c in cols]
    data = [head] + [[Paragraph(escape(_clean(r[j]) if j < len(r) else ""), st["cell"]) for j in range(len(cols))] for r in rows]
    tb = Table(data, colWidths=_col_widths(cols, rows, width), repeatRows=1, splitByRow=1)
    tb.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2589e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7ff")]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tb


def build_pdf(body: dict) -> bytes:
    summary = body.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("risk"), dict) or "level" not in summary:
        raise ValueError("Thiếu kết quả khám (summary) để lập báo cáo")
    report = body.get("report") if isinstance(body.get("report"), dict) else {}
    figures = [f for f in (report.get("figures") or []) if isinstance(f, dict)][:MAX_FIGS]
    tables = [t for t in (report.get("tables") or []) if isinstance(t, dict)][:MAX_TABLES]
    _register_fonts()
    st = _styles()
    name = _patient_name(body)
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    level = str(summary.get("level"))
    lcol = colors.HexColor(summary.get("level_color") if re.fullmatch(r"#[0-9a-fA-F]{6}", str(summary.get("level_color") or "")) else LEVEL_COLORS.get(level, "#4a5a7c"))

    buf = io.BytesIO()
    W, H = A4
    mx, top, bottom = 18 * mm, 16 * mm, 20 * mm
    cw = W - 2 * mx

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(mx, 14 * mm, W - mx, 14 * mm)
        canvas.setFillColor(colors.HexColor("#8a5a00"))
        canvas.setFont(FONT_B, 8.5)
        canvas.drawString(mx, 9 * mm, DISCLAIMER)
        canvas.setFillColor(MUTED)
        canvas.setFont(FONT, 8.5)
        canvas.drawRightString(W - mx, 9 * mm, f"{name} · Trang {doc.page}")
        if doc.page > 1:
            canvas.setFont(FONT, 8)
            canvas.drawString(mx, H - 10 * mm, f"Phòng khám AI · Báo cáo khám của {name} · {now}")
        canvas.restoreState()

    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=mx, rightMargin=mx, topMargin=top, bottomMargin=bottom,
                          title=f"Báo cáo khám - {name}", author="Phòng khám AI", subject=DISCLAIMER)
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(mx, bottom, cw, H - top - bottom - 2 * mm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page)])

    el = []
    banner = Table([[Paragraph("Báo cáo khám · Phòng khám AI", st["h1"])], [Paragraph(f"Bệnh nhân: <b>{escape(name)}</b> · Ngày lập: {now}", st["sub"])]], colWidths=[cw])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2589e0")), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                ("TOPPADDING", (0, 0), (0, 0), 12), ("BOTTOMPADDING", (0, -1), (-1, -1), 12), ("ROUNDEDCORNERS", [8, 8, 8, 8])]))
    el += [banner, Spacer(1, 8)]
    badge = Table([[Paragraph(f"<b>Mức nguy cơ: {escape(level.upper())}</b>", ParagraphStyle("b", fontName=FONT_B, fontSize=12, leading=16, textColor=colors.white)),
                    _p(summary.get("headline", ""), st["body"])]], colWidths=[cw * 0.34, cw * 0.66])
    badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), lcol), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#f6f9ff")), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    el += [badge, Spacer(1, 6)]
    bits = []
    if summary.get("bmi") is not None:
        bits.append(f"BMI {float(summary['bmi']):.1f}")
    if summary.get("risk_index") is not None:
        bits.append(f"chỉ số nguy cơ {float(summary['risk_index']) * 100:.0f}%")
    if bits:
        el.append(_p("Chỉ số tóm tắt: " + ", ".join(bits) + ".", st["body"]))

    el += [Paragraph("Xác suất theo mô hình", st["h2"]), _risk_bars(summary["risk"], cw),
           _p("Ba thanh là xác suất mô hình học máy gán cho từng nhóm; đây là ước lượng thống kê, không phải kết luận bệnh.", st["cap"])]

    flags = [f for f in (summary.get("red_flags") or []) if isinstance(f, dict)]
    el.append(Paragraph("Cờ đỏ cần chú ý", st["h2"]))
    if flags:
        for f in flags:
            danger = f.get("level") == "danger"
            col = "#e5342d" if danger else "#b57a00"
            el.append(Paragraph(f"<font color='{col}'><b>[{escape(str(f.get('id', '')))}] {'NGUY HIỂM' if danger else 'Lưu ý'}</b></font> {escape(_clean(f.get('text', '')))}", st["bullet"], bulletText="●"))
    else:
        el.append(_p("Không có cờ đỏ nào được kích hoạt.", st["body"]))

    el.append(Paragraph("Lời khuyên", st["h2"]))
    for a in summary.get("advice") or []:
        el.append(Paragraph(escape(_clean(a)), st["bullet"], bulletText="•"))

    if figures:
        for k, f in enumerate(figures):
            el.append(KeepTogether(([Paragraph("Hình do thư ký báo cáo vẽ", st["h2"])] if k == 0 else []) + [_p(f.get("title", ""), st["h3"]), _png_image(str(f.get("png_base64", "")), cw, 82 * mm), Spacer(1, 3), _p(f.get("caption", ""), st["cap"])]))

    first = True
    for t in tables:
        tb = _table(t, st, cw)
        if tb is None:
            continue
        head = [Paragraph("Các bảng tóm tắt", st["h2"])] if first else []
        first = False
        title = _p(t.get("title", ""), st["h3"])
        if len(t.get("rows") or []) <= 8:
            el.append(KeepTogether(head + [title, tb]))
        else:
            el.append(KeepTogether(head + [title]))
            el.append(tb)
        el.append(Spacer(1, 6))

    cites = [c for c in (summary.get("citations") or []) if isinstance(c, dict)]
    for k, c in enumerate(cites):
        url = str(c.get("url", ""))
        link = f" <font color='#0b7f72'>{escape(url)}</font>" if url else ""
        el.append(KeepTogether(([Paragraph("Nguồn trích dẫn", st["h2"])] if k == 0 else []) + [Paragraph(f"<b>{escape(str(c.get('id', '')))}</b>{link}", st["body"]), _p("“" + str(c.get("quote", "")).strip() + "”", st["quote"]), Spacer(1, 4)]))

    warn = Table([[_p(DISCLAIMER + ". Báo cáo do chương trình học tập tự sinh; hãy gặp bác sĩ để được tư vấn và xét nghiệm xác nhận.", st["body"])]], colWidths=[cw])
    warn.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8dc")), ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#f2d77a")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    el += [Spacer(1, 8), KeepTogether([warn])]
    doc.build(el)
    return buf.getvalue()


def routes(clinic, speaker):
    def pdf(body: dict):
        return "application/pdf", build_pdf(body), 'attachment; filename="bao-cao-kham.pdf"'
    return {}, {"/api/report/pdf": pdf}
