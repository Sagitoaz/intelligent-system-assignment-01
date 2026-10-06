"""Mở báo cáo bằng Microsoft Word để cập nhật mục lục/danh mục hình–bảng, rồi đo số trang từng chương.

Chạy sau build_report:  .venv\\Scripts\\python.exe -m scripts.tieuluan.finalize_docx [--pdf DUONG_DAN.pdf]
Cần Windows + Microsoft Word (pywin32). Tùy chọn --pdf chỉ để tự kiểm tra bố cục, không ghi vào repo.
"""

import argparse
import json
import sys

from scripts.tieuluan.build_report import DOCS, NAME


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None, help="xuất thêm PDF tạm (đường dẫn ngoài repo) để xem bố cục")
    args = ap.parse_args()
    import win32com.client

    path = (DOCS / f"{NAME}.docx").resolve()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(path), ReadOnly=False, AddToRecentFiles=False)
        if doc.ReadOnly:
            raise RuntimeError("File đang được mở ở nơi khác (Word). Hãy đóng file rồi chạy lại.")
        for toc in doc.TablesOfContents:
            toc.Update()
        doc.Fields.Update()
        doc.Repaginate()
        for toc in doc.TablesOfContents:   # cập nhật lần 2 sau khi số trang đã ổn định
            toc.Update()
        heads = []
        for p in doc.Paragraphs:
            if p.OutlineLevel == 1:
                text = p.Range.Text.strip()
                if text:
                    heads.append((text, p.Range.Information(1), p.Range.Information(3)))  # trang hiển thị, trang tuyệt đối
        total = doc.ComputeStatistics(2)
        doc.Save()
        if args.pdf:
            doc.ExportAsFixedFormat(args.pdf, 17)
        doc.Close(False)
    finally:
        word.Quit()
    report = {"total_pages": total, "chapters": []}
    for (title, shown, absolute), nxt in zip(heads, heads[1:] + [(None, None, total + 1)]):
        report["chapters"].append({"title": title[:70], "start_page": shown, "pages": nxt[2] - absolute})
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
