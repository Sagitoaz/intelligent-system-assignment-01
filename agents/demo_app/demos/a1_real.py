import csv
import os
from collections import defaultdict

import shared

META = dict(agent="agent01", id="a1_real", title="Dữ liệu thật: hóa đơn & hủy đơn", js="a1_real.js",
            description="25.900 hóa đơn bán lẻ thật (UCI Online Retail): tỉ lệ đơn hủy theo quốc gia và theo tháng.")
_cache = {}


def _load():
    if _cache:
        return _cache
    path = os.path.join(shared.DATA_DIRS["agent01"], "external", "online_retail_invoices.csv")
    country, month = defaultdict(lambda: [0, 0, 0.0]), defaultdict(lambda: [0, 0])
    total = cancelled = 0
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c, m, can = country[r["country"]], month[r["date"][:7]], int(r["cancelled"])
            c[0] += 1; c[1] += can; m[0] += 1; m[1] += can
            if not can:
                c[2] += float(r["total"])
            total += 1; cancelled += can
    top = sorted(country.items(), key=lambda kv: -kv[1][0])[:8]
    _cache.update(total=total, cancelled=cancelled,
                  countries=[dict(country=k, invoices=v[0], cancelled=v[1], revenue=round(v[2])) for k, v in top],
                  months=[dict(month=k, invoices=v[0], cancelled=v[1]) for k, v in sorted(month.items())])
    return _cache


def handle(p):
    return _load()
