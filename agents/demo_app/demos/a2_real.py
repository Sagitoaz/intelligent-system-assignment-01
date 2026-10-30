import csv
import os
import statistics
from collections import defaultdict

import shared

META = dict(agent="agent02", id="a2_real", title="Dữ liệu thật: giá và doanh số", js="a2_real.js",
            description="Dữ liệu bán lẻ thật (UCI Online Retail): ngày có giá thấp hơn mức thường có bán nhiều hơn không?")
_cache = {}


def _load():
    if _cache:
        return _cache
    path = os.path.join(shared.DATA_DIRS["agent02"], "external", "online_retail_daily_price_demand.csv")
    by_product = defaultdict(list)  # mã hàng -> [(giá, số lượng bán trong ngày)]
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            by_product[r["stock_code"]].append((float(r["unit_price"]), int(r["units_sold"])))
    rows = []
    for code, obs in by_product.items():
        if len(obs) < 60:
            continue
        typical = statistics.median(p for p, _ in obs)
        low = [u for p, u in obs if p < 0.95 * typical]
        normal = [u for p, u in obs if 0.95 * typical <= p <= 1.05 * typical]
        if len(low) >= 5 and len(normal) >= 5:
            rows.append(dict(code=code, typical_price=round(typical, 2), low_days=len(low), normal_days=len(normal),
                             low_units=round(statistics.mean(low), 1), normal_units=round(statistics.mean(normal), 1)))
    higher = sum(r["low_units"] > r["normal_units"] for r in rows)
    rows.sort(key=lambda r: -(r["low_days"] + r["normal_days"]))
    _cache.update(products=len(by_product), compared=len(rows), higher=higher, rows=rows[:20])
    return _cache


def handle(p):
    return _load()
