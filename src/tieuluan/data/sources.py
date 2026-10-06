"""Các hàm tải dữ liệu thô từ nguồn công khai, kèm thông tin nguồn gốc (provenance).

Nguồn sử dụng:
- Yahoo Finance qua thư viện ``yfinance``: S&P 500, Bitcoin, cổ phiếu DJIA, tiền mã hóa.
- API biểu đồ công khai của SSI iBoard, DNSE (Entrade) và VNDirect: VN-Index, cổ phiếu VN30.
  Ba nguồn được đối chiếu chéo để phát hiện phiên bị ghi sai (xem ``build_vnindex``).
- UCI Machine Learning Repository: hai bộ dữ liệu bảng cho Chương 2.

Không dùng gói ``vnstock`` vì tại thời điểm thực hiện (10/2026) dự án này đang bị PyPI
đặt ở trạng thái "quarantined" (cách ly do nghi ngờ mã độc).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
OHLCV = ["open", "high", "low", "close", "volume"]


def _epoch(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def _get_json(url: str, retries: int = 4, pause: float = 1.5):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=40)
            response.raise_for_status()
            return response.json()
        except Exception as error:  # mạng chập chờn hoặc bị giới hạn tần suất
            last_error = error
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Không tải được {url}: {last_error}")


def _udf_to_frame(payload: dict) -> pd.DataFrame:
    """Chuyển định dạng TradingView-UDF {t,o,h,l,c,v} thành DataFrame theo ngày giao dịch VN."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert("Asia/Ho_Chi_Minh").date,
        "open": payload["o"], "high": payload["h"], "low": payload["l"],
        "close": payload["c"], "volume": payload["v"],
    })
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return frame


def fetch_ssi(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://iboard-api.ssi.com.vn/statistics/charts/history?resolution=1D"
        f"&symbol={symbol}&from={_epoch(start)}&to={_epoch(end) + 86_399}"
    )
    payload = _get_json(url)
    return _udf_to_frame(payload["data"]).assign(source="SSI iBoard")


def fetch_dnse(symbol: str, start: str, end: str, kind: str = "index") -> pd.DataFrame:
    url = (
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{kind}?resolution=1D"
        f"&symbol={symbol}&from={_epoch(start)}&to={_epoch(end) + 86_399}"
    )
    return _udf_to_frame(_get_json(url)).assign(source="DNSE Entrade")


def fetch_vndirect(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://dchart-api.vndirect.com.vn/dchart/history?resolution=D"
        f"&symbol={symbol}&from={_epoch(start)}&to={_epoch(end) + 86_399}"
    )
    return _udf_to_frame(_get_json(url)).assign(source="VNDirect dchart")


def fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Giá ngày từ Yahoo Finance; OHLC đã điều chỉnh chia tách & cổ tức (auto_adjust=True)."""
    import yfinance as yf

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(4):
        raw = yf.download(
            ticker, start=start, end=end_exclusive, auto_adjust=True,
            progress=False, threads=False,
        )
        if raw is not None and len(raw):
            break
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Yahoo Finance trả về rỗng cho {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw.rename(columns=str.lower)[OHLCV].reset_index().rename(columns={"Date": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    return frame.dropna(subset=["close"]).reset_index(drop=True).assign(source="Yahoo Finance")


def build_vnindex(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Ghép VN-Index từ ba nguồn và trả về báo cáo kiểm định chất lượng.

    Quy tắc ghép theo từng phiên:
    1. SSI là nguồn chính kể từ 2009-07-01 (trước mốc này SSI chỉ có vài dòng rời rạc,
       trong đó một dòng năm 2007 bị lỗi bước nhảy ~+499%).
    2. Phiên nào SSI thiếu (kể cả lỗ hổng 20/07–24/08/2009 của SSI) thì lấy từ DNSE.
    3. Phiên SSI và DNSE lệch giá đóng cửa: dùng VNDirect làm "trọng tài" và ghi lại kết quả.
    Cột ``source`` cho biết mỗi phiên lấy từ nguồn nào. DNSE lưu giá giai đoạn 2003–06/2009
    ở dạng làm tròn tới số nguyên, nên thực nghiệm chỉ dùng VN-Index từ 2010 (xem config).
    """
    ssi = fetch_ssi("VNINDEX", start, end)
    dnse = fetch_dnse("VNINDEX", start, end, kind="index")
    vnd = fetch_vndirect("VNINDEX", start, end)

    ssi_cut = pd.Timestamp("2009-07-01")
    ssi_main = ssi[ssi.date >= ssi_cut]
    merged = ssi_main.merge(dnse, on="date", how="inner", suffixes=("_ssi", "_dnse"))
    disagree = merged[(merged.close_ssi - merged.close_dnse).abs() > 0.01]
    check = disagree[["date", "close_ssi", "close_dnse"]].merge(
        vnd[["date", "close"]].rename(columns={"close": "close_vnd"}), on="date", how="left"
    )
    vnd_votes_ssi = int(((check.close_vnd - check.close_ssi).abs() < 0.015).sum())
    vnd_votes_dnse = int(((check.close_vnd - check.close_dnse).abs() < 0.015).sum())

    main = ssi_main[["date"] + OHLCV].assign(source="SSI iBoard")
    fill = dnse[~dnse.date.isin(main.date)][["date"] + OHLCV].assign(source="DNSE Entrade")
    combined = pd.concat([fill, main], ignore_index=True).sort_values("date").reset_index(drop=True)
    filled_after_cut = fill[fill.date >= ssi_cut]

    overlap_vnd = main.merge(vnd[["date", "close"]], on="date", suffixes=("", "_vnd"))
    report = {
        "rows_ssi": int(len(ssi)), "rows_dnse": int(len(dnse)), "rows_vndirect": int(len(vnd)),
        "ssi_primary_from": str(ssi_cut.date()),
        "overlap_ssi_dnse": int(len(merged)),
        "close_disagreements_ssi_vs_dnse": int(len(disagree)),
        "disagreement_years": sorted({int(d.year) for d in disagree.date}),
        "vndirect_votes_for_ssi": vnd_votes_ssi,
        "vndirect_votes_for_dnse": vnd_votes_dnse,
        "dnse_rows_filling_ssi_gaps_after_cut": int(len(filled_after_cut)),
        "dnse_fill_dates_after_cut": [str(d.date()) for d in filled_after_cut.date],
        "overlap_ssi_vndirect": int(len(overlap_vnd)),
        "close_disagreements_ssi_vs_vndirect": int(((overlap_vnd.close - overlap_vnd.close_vnd).abs() > 0.015).sum()),
        "rows_final": int(len(combined)),
        "first_date": str(combined.date.iloc[0].date()), "last_date": str(combined.date.iloc[-1].date()),
    }
    return combined, report


def fetch_vn_stock(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """Cổ phiếu VN: SSI là nguồn chính (lịch sử dài hơn), DNSE dùng để đối chiếu giá đóng cửa."""
    ssi = fetch_ssi(symbol, start, end)
    try:
        dnse = fetch_dnse(symbol, start, end, kind="stock")
        merged = ssi.merge(dnse, on="date", suffixes=("_ssi", "_dnse"))
        rel = (merged.close_ssi - merged.close_dnse).abs() / merged.close_ssi
        agree = float((rel < 0.005).mean()) if len(merged) else float("nan")
        overlap = int(len(merged))
    except Exception:
        agree, overlap = float("nan"), 0
    return ssi[["date"] + OHLCV].assign(source="SSI iBoard"), {
        "rows": int(len(ssi)), "overlap_with_dnse": overlap, "close_agreement_ratio": agree,
    }


def download_uci(name: str, url: str, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    archive.extractall(target_dir)
    return sorted(archive.namelist())


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_frame(frame: pd.DataFrame, path: Path, meta: dict, provenance: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")
    key = str(path.relative_to(path.parents[2])).replace("\\", "/")
    provenance[key] = {
        **meta,
        "rows": int(len(frame)),
        "first_date": str(pd.Timestamp(frame.date.iloc[0]).date()) if "date" in frame and len(frame) else None,
        "last_date": str(pd.Timestamp(frame.date.iloc[-1]).date()) if "date" in frame and len(frame) else None,
        "sha256": sha256_of(path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def ohlc_quality(frame: pd.DataFrame) -> dict:
    """Thống kê chất lượng: dòng OHLC mâu thuẫn, dòng phẳng (O=H=L=C), lợi suất cực trị."""
    hi_bad = frame.high < frame[["open", "close"]].max(axis=1) - 1e-9
    lo_bad = frame.low > frame[["open", "close"]].min(axis=1) + 1e-9
    flat = (frame.open == frame.high) & (frame.high == frame.low) & (frame.low == frame.close)
    ret = np.log(frame.close).diff()
    return {
        "inconsistent_ohlc_rows": int((hi_bad | lo_bad).sum()),
        "flat_ohlc_rows": int(flat.sum()),
        "zero_volume_rows": int((frame.volume <= 0).sum()),
        "max_abs_log_return": float(np.nanmax(np.abs(ret))) if len(frame) > 1 else None,
    }


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
