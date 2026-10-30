"""Tải bộ dữ liệu thật UCI Online Retail (giao dịch thương mại điện tử Anh, 2010-2011) và đổi sang CSV.

Nguồn: https://archive.ics.uci.edu/dataset/352/online+retail (CC BY 4.0, Chen 2015).
Chỉ dùng thư viện chuẩn; tệp tải về chỉ được đọc như dữ liệu, không chạy mã trong đó.
Chạy: python agents/demo_app/fetch_external_data.py
Kết quả:
  data/agent02/external/online_retail_lines.csv   (từng dòng hàng hóa: giá, số lượng, ngày)
  data/agent02/external/online_retail_daily_price_demand.csv   (nhu cầu theo ngày và mức giá)
  data/agent01/external/online_retail_invoices.csv   (mỗi hóa đơn một dòng, hóa đơn bắt đầu bằng C là đơn hủy/hoàn)
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx_rows(xlsx_bytes):
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as z:
        shared = [("".join(t.text or "" for t in si.iter(NS + "t")))
                  for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si")]
        with z.open("xl/worksheets/sheet1.xml") as f:
            for _, row in ET.iterparse(f):
                if row.tag != NS + "row":
                    continue
                cells = {}
                for c in row.iter(NS + "c"):
                    col = re.match(r"[A-Z]+", c.get("r")).group()
                    v = c.find(NS + "v")
                    if v is None:
                        continue
                    cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
                yield [cells.get(k, "") for k in "ABCDEFGH"]
                row.clear()


def main():
    print("Đang tải", URL)
    with urllib.request.urlopen(URL, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        xlsx = z.read(next(n for n in z.namelist() if n.lower().endswith(".xlsx")))
    out2 = os.path.join(ROOT, "data", "agent02", "external")
    out1 = os.path.join(ROOT, "data", "agent01", "external")
    os.makedirs(out2, exist_ok=True)
    os.makedirs(out1, exist_ok=True)

    invoices = {}
    daily = defaultdict(lambda: [0, 0.0])  # (ngày, mức giá) -> [số lượng, số dòng]
    n = 0
    with open(os.path.join(out2, "online_retail_lines.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country"])
        rows = read_xlsx_rows(xlsx)
        next(rows)  # dòng tiêu đề
        for inv, code, desc, qty, date, price, cust, country in rows:
            try:
                when = dt.datetime(1899, 12, 30) + dt.timedelta(days=float(date))
                qty, price = int(float(qty)), float(price)
            except ValueError:
                continue
            cust = cust[:-2] if cust.endswith(".0") else cust
            w.writerow([inv, code, desc, qty, when.strftime("%Y-%m-%d %H:%M"), price, cust, country])
            n += 1
            rec = invoices.setdefault(inv, dict(invoice=inv, customer_id=cust, country=country, date=when.strftime("%Y-%m-%d"),
                                                lines=0, items=0, total=0.0, cancelled=int(inv.startswith("C"))))
            rec["lines"] += 1
            rec["items"] += qty
            rec["total"] += qty * price
            if qty > 0 and price > 0 and code and not code.startswith(("POST", "D", "M", "BANK")):
                key = (code, when.strftime("%Y-%m-%d"), round(price, 2))
                daily[key][0] += qty
                daily[key][1] += 1
    print(f"online_retail_lines.csv: {n:,} dòng")
    with open(os.path.join(out1, "online_retail_invoices.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["invoice", "customer_id", "country", "date", "lines", "items", "total", "cancelled"])
        w.writeheader()
        for rec in invoices.values():
            rec["total"] = round(rec["total"], 2)
            w.writerow(rec)
    print(f"online_retail_invoices.csv: {len(invoices):,} hóa đơn")
    with open(os.path.join(out2, "online_retail_daily_price_demand.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["stock_code", "date", "unit_price", "units_sold", "order_lines"])
        for (code, day, price), (units, lines) in sorted(daily.items()):
            w.writerow([code, day, price, units, int(lines)])
    print(f"online_retail_daily_price_demand.csv: {len(daily):,} dòng")


if __name__ == "__main__":
    main()
