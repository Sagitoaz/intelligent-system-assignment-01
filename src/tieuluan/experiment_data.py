"""Tạo bộ dữ liệu thực nghiệm có thể tái lập.

Hai nhóm dữ liệu:
- Ba chỉ số thị trường (S&P 500, VN-Index, Bitcoin): mỗi mẫu là cửa sổ 20 quan sát kết thúc
  tại ngày t; nhãn là dấu lợi suất close-to-close của quan sát kế tiếp (t+1). Từ cùng một cửa sổ
  tạo hai biểu diễn: chuỗi log-return (Chương 4, RNN) và ảnh GASF 20×20 (Chương 3, CNN).
- Hai bảng UCI (phá sản doanh nghiệp, vỡ nợ thẻ tín dụng) cho Chương 2.

Nguyên tắc chống rò rỉ: chia theo thời gian và loại mẫu có nhãn vượt ranh giới tập; mọi phép
chuẩn hóa/điền khuyết chỉ học trên tập train; ảnh GASF chỉ chuẩn hóa trong chính cửa sổ quá khứ.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ANALYSIS_START, PROJECT_ROOT, SNAPSHOT_END, TRAIN_END, VAL_END

ROOT = PROJECT_ROOT
RAW = ROOT / "data/tieuluan/raw"
OUT = ROOT / "data/tieuluan/processed"
WINDOW = 20
MARKETS = ("sp500", "vnindex", "btc")
TABULAR = ("taiwan_bankruptcy", "credit_default")


def gaf_image(prices) -> np.ndarray:
    """Gramian Angular Summation Field: G[i, j] = cos(φᵢ + φⱼ), φ = arccos(x̃).

    x̃ là giá chuẩn hóa min–max về [0, 1] *trong chính cửa sổ* nên không dùng thông tin tương lai.
    Chọn [0, 1] thay vì [−1, 1]: với [−1, 1], đổi x̃ → −x̃ cho φ → π − φ và cos(φᵢ + φⱼ) không đổi,
    tức một cửa sổ tăng và ảnh phản chiếu (giảm) của nó cho CÙNG một ảnh — mất thông tin chiều giá.
    Trên [0, 1], φ ∈ [0, π/2] và phép đổi sang góc là đơn ánh (Wang & Oates, 2015).
    cos(φᵢ + φⱼ) = x̃ᵢx̃ⱼ − √(1−x̃ᵢ²)·√(1−x̃ⱼ²). Cửa sổ có giá không đổi được quy ước x̃ = 0.
    """
    p = np.asarray(prices, dtype=np.float64)
    span = np.ptp(p)
    z = np.zeros_like(p) if span < 1e-12 else (p - p.min()) / span
    z = np.clip(z, 0, 1)
    s = np.sqrt(np.maximum(0, 1 - z * z))
    return (np.outer(z, z) - np.outer(s, s)).astype(np.float32)


def split_by_dates(dates, target_dates) -> dict[str, np.ndarray]:
    """Mặt nạ train/validation/test theo thời gian; cả ngày quan sát lẫn ngày nhãn phải nằm trong tập."""
    d = np.asarray(dates, dtype="datetime64[D]")
    t = np.asarray(target_dates, dtype="datetime64[D]")
    train_end, val_end, end = (np.datetime64(x) for x in (TRAIN_END, VAL_END, SNAPSHOT_END))
    val_start = train_end + np.timedelta64(1, "D")
    test_start = val_end + np.timedelta64(1, "D")
    return {
        "train": (d <= train_end) & (t <= train_end),
        "val": (d >= val_start) & (d <= val_end) & (t <= val_end),
        "test": (d >= test_start) & (t <= end),
    }


def market_samples(frame: pd.DataFrame, window: int = WINDOW) -> dict[str, np.ndarray]:
    """Mỗi mốc t (anchor) sinh một mẫu: 20 log-return đến t, ảnh GASF của 20 giá đóng cửa đến t,
    nhãn y = 1 nếu C(t+1) > C(t)."""
    f = frame.copy()
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date")
    f = f[(f.date <= pd.Timestamp(SNAPSHOT_END)) & (f.close > 0)].dropna(subset=["close"])
    if f.date.duplicated().any():
        raise ValueError("Duplicate dates must be resolved before sampling")
    p = f.close.to_numpy(float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite prices")
    dates = f.date.to_numpy(dtype="datetime64[D]")
    returns = np.diff(np.log(p))                 # returns[j] = ln(C[j+1] / C[j])
    anchors = np.arange(window, len(f) - 1)      # cần 20 lợi suất trước t và một quan sát sau t
    sequence = np.array([returns[i - window:i, None] for i in anchors], dtype=np.float32)
    images = np.array([gaf_image(p[i - window + 1:i + 1])[None] for i in anchors], dtype=np.float32)
    future = p[anchors + 1] / p[anchors] - 1
    return {"sequence": sequence, "image": images, "y": (future > 0).astype(np.float32),
            "return": future, "date": dates[anchors], "target_date": dates[anchors + 1]}


def _prepare_market(key: str) -> dict:
    path = RAW / "indices" / f"{key}.csv"
    frame = pd.read_csv(path)
    analysed = frame[pd.to_datetime(frame.date) >= pd.Timestamp(ANALYSIS_START[key])]
    samples = market_samples(analysed)
    masks = split_by_dates(samples["date"], samples["target_date"])
    scaler = StandardScaler().fit(samples["sequence"][masks["train"]].reshape(-1, 1))
    sequence = scaler.transform(samples["sequence"].reshape(-1, 1)).reshape(samples["sequence"].shape).astype(np.float32)
    payload, stats = {}, {}
    for split, mask in masks.items():
        payload[f"{split}_sequence"] = sequence[mask]
        payload[f"{split}_image"] = samples["image"][mask]
        for field in ("y", "date", "target_date", "return"):
            payload[f"{split}_{field}"] = samples[field][mask]
        stats[split] = {"n": int(mask.sum()), "positive": int(samples["y"][mask].sum()),
                        "first": str(samples["date"][mask][0]), "last_target": str(samples["target_date"][mask][-1])}
    payload["scaler_mean"], payload["scaler_scale"] = scaler.mean_, scaler.scale_
    np.savez_compressed(OUT / f"{key}.npz", **payload)
    return {"raw_rows": len(frame), "analysis_start": ANALYSIS_START[key], "analysed_rows": len(analysed),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "first": str(analysed.date.iloc[0]), "last": str(analysed.date.iloc[-1]), "splits": stats,
            "window": WINDOW, "dropped_unlabelled_or_boundary": int(len(samples["y"]) - sum(s["n"] for s in stats.values())),
            "raw_sample": analysed.head(3).to_dict("records"),
            "return_quantiles": dict(zip(["min", "q25", "median", "q75", "max"],
                                         np.quantile(samples["return"], [0, .25, .5, .75, 1]).tolist()))}


def _prepare_tabular(key: str) -> dict:
    if key == "taiwan_bankruptcy":
        path = RAW / "uci" / key / "data.csv"
        f = pd.read_csv(path)
        y = f.iloc[:, 0].to_numpy(np.float32)       # cột đầu "Bankrupt?"
        x = f.iloc[:, 1:]
    else:
        path = next((RAW / "uci" / key).glob("*.xls"))
        f = pd.read_excel(path, header=1)           # hàng 1 là X1..X23, hàng 2 là tên biến
        y = f.iloc[:, -1].to_numpy(np.float32)      # "default payment next month"
        x = f.iloc[:, 1:-1]                         # bỏ ID (mã định danh, không phải đặc trưng)
    ids = np.arange(len(y))
    # Dữ liệu cắt ngang (mỗi dòng một doanh nghiệp/khách hàng) → chia ngẫu nhiên phân tầng 60/20/20.
    tr, rest = train_test_split(ids, test_size=.4, stratify=y, random_state=2026)
    va, te = train_test_split(rest, test_size=.5, stratify=y[rest], random_state=2026)
    imputer = SimpleImputer(strategy="median").fit(x.iloc[tr])
    scaler = StandardScaler().fit(imputer.transform(x.iloc[tr]))
    scaled = scaler.transform(imputer.transform(x)).astype(np.float32)
    payload = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "imputer_median": imputer.statistics_,
               "feature_names": np.array(x.columns.astype(str), dtype=str)}
    stats = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        payload.update({f"{name}_tabular": scaled[idx], f"{name}_y": y[idx], f"{name}_ids": idx})
        stats[name] = {"n": len(idx), "positive": int(y[idx].sum())}
    np.savez_compressed(OUT / f"{key}.npz", **payload)
    return {"raw_rows": len(f), "features": x.shape[1], "missing": int(x.isna().sum().sum()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "positive": int(y.sum()), "splits": stats,
            "raw_sample": f.iloc[:3, :6].to_dict("records")}


def prepare_datasets() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = {key: _prepare_market(key) for key in MARKETS}
    inventory.update({key: _prepare_tabular(key) for key in TABULAR})
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False, default=str), encoding="utf8")
    return inventory


if __name__ == "__main__":
    print(json.dumps(prepare_datasets(), ensure_ascii=False, indent=2, default=str))
