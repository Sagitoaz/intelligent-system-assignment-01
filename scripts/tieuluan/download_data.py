"""Tải toàn bộ dữ liệu thô cho tiểu luận và ghi provenance + báo cáo chất lượng.

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.download_data [--force]
Kết quả: data/tieuluan/raw/**.csv, data/tieuluan/provenance.json, data/tieuluan/data_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import pandas as pd

from src.tieuluan import config
from src.tieuluan.data import sources

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main(force: bool = False, only_index: str | None = None) -> None:
    config.ensure_dirs()
    prov_path = config.DATA_DIR / "provenance.json"
    qc_path = config.DATA_DIR / "data_quality.json"
    provenance = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {}
    quality = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    start, end = config.SNAPSHOT_START, config.SNAPSHOT_END

    def need(path):
        if only_index is not None:
            return path == config.RAW_DIR / "indices" / f"{only_index}.csv"
        return force or not path.exists()

    # 1) Ba chỉ số thị trường
    for key, spec in config.MARKET_INDICES.items():
        path = config.RAW_DIR / "indices" / f"{key}.csv"
        if not need(path):
            continue
        if spec["source"] == "yahoo":
            frame = sources.fetch_yahoo(spec["symbol"], start, end)
            meta = {"source": "Yahoo Finance (yfinance, auto_adjust=True)", "symbol": spec["symbol"],
                    "url": f"https://finance.yahoo.com/quote/{spec['symbol']}/history"}
        else:
            frame, report = sources.build_vnindex(start, end)
            quality["vnindex_cross_source"] = report
            meta = {"source": "SSI iBoard (nguồn chính từ 2009-07) + DNSE Entrade (trước 2009-07 và bù các phiên SSI thiếu), "
                              "đối chiếu VNDirect", "symbol": "VNINDEX", "url": "https://iboard.ssi.com.vn/"}
        frame = frame[(frame.date >= start) & (frame.date <= end)].reset_index(drop=True)
        if key == "vnindex":   # số dòng cuối cùng tính SAU khi cắt theo mốc chốt dữ liệu
            quality["vnindex_cross_source"].update(rows_final=int(len(frame)), first_date=str(frame.date.iloc[0].date()),
                                                   last_date=str(frame.date.iloc[-1].date()))
        sources.save_frame(frame, path, meta, provenance)
        quality[f"indices/{key}"] = sources.ohlc_quality(frame)
        print(f"[index] {key:8s} rows={len(frame):5d} {frame.date.iloc[0].date()} → {frame.date.iloc[-1].date()}")

    # 2) Cổ phiếu Mỹ (DJIA 30)
    for ticker in config.US_STOCKS:
        path = config.RAW_DIR / "us_stocks" / f"{ticker}.csv"
        if not need(path):
            continue
        frame = sources.fetch_yahoo(ticker, start, end)
        sources.save_frame(frame, path, {"source": "Yahoo Finance (yfinance, auto_adjust=True)", "symbol": ticker}, provenance)
        quality[f"us_stocks/{ticker}"] = sources.ohlc_quality(frame)
        print(f"[us] {ticker:6s} rows={len(frame):5d} from {frame.date.iloc[0].date()}")
        time.sleep(0.4)

    # 3) Cổ phiếu Việt Nam (VN30)
    for ticker in config.VN_STOCKS:
        path = config.RAW_DIR / "vn_stocks" / f"{ticker}.csv"
        if not need(path):
            continue
        frame, check = sources.fetch_vn_stock(ticker, start, end)
        frame = frame[(frame.date >= start) & (frame.date <= end)].reset_index(drop=True)
        sources.save_frame(frame, path, {"source": "SSI iBoard (đối chiếu DNSE Entrade)", "symbol": ticker}, provenance)
        quality[f"vn_stocks/{ticker}"] = {**sources.ohlc_quality(frame), **check}
        print(f"[vn] {ticker:6s} rows={len(frame):5d} from {frame.date.iloc[0].date()} agree={check['close_agreement_ratio']:.4f}")
        time.sleep(0.6)

    # 4) Tiền mã hóa
    for ticker in config.CRYPTO_ASSETS:
        path = config.RAW_DIR / "crypto" / f"{ticker}.csv"
        if not need(path):
            continue
        frame = sources.fetch_yahoo(ticker, start, end)
        sources.save_frame(frame, path, {"source": "Yahoo Finance (yfinance)", "symbol": ticker}, provenance)
        quality[f"crypto/{ticker}"] = sources.ohlc_quality(frame)
        print(f"[crypto] {ticker:9s} rows={len(frame):5d} from {frame.date.iloc[0].date()}")
        time.sleep(0.4)

    # 5) Hai bộ dữ liệu bảng của UCI
    for name, spec in config.UCI_DATASETS.items():
        target = config.RAW_DIR / "uci" / name
        if only_index is not None or (not force and target.exists() and any(target.iterdir())):
            continue
        files = sources.download_uci(name, spec["url"], target)
        provenance[f"raw/uci/{name}"] = {"source": "UCI Machine Learning Repository", "uci_id": spec["id"],
                                         "url": spec["url"], "page": spec["page"], "files": files}
        print(f"[uci] {name}: {files}")

    sources.write_json(provenance, prov_path)
    sources.write_json(quality, qc_path)
    print("Đã ghi", prov_path, "và", qc_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="tải lại kể cả khi file đã tồn tại")
    parser.add_argument("--only-index", choices=sorted(config.MARKET_INDICES), default=None,
                        help="chỉ tải lại một chỉ số (ghi đè file của chỉ số đó), giữ nguyên các file khác")
    args = parser.parse_args()
    main(args.force, args.only_index)
