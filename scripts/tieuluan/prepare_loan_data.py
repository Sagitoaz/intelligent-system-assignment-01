"""Chuẩn bị bộ dữ liệu Chương 2: Lending Club (khoản vay 36 tháng, 2012–2015, kết cục đến 2018).

Nguồn:  Hugging Face `codesignal/lending-club-loan-accepted` (bản sao của Lending Club
        "accepted_2007_to_2018Q4", giấy phép CC0-1.0, 2.260.701 dòng, 151 cột).
Bước 1: đọc tệp gốc theo khối; giữ khoản vay 36 tháng có kết cục cuối cùng (Fully Paid / Charged Off / Default),
        phát hành 01/2012–06/2015 (đủ 36 tháng + thời gian chờ nên không còn khoản "đang trả" làm lệch nhãn).
Bước 2: lấy mẫu ngẫu nhiên 150.000 dòng (seed 2026) và lưu CSV có cột thô tại
        data/tieuluan/raw/lending_club/loans_36m_2012_2015_150k.csv.
Bước 3: chia THEO THỜI GIAN (train/val/test), điền khuyết + chuẩn hóa chỉ học trên train,
        lưu data/tieuluan/processed/loan_default.npz và tài nguyên nhỏ cho demo.

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.prepare_loan_data <đường dẫn accepted_2007_to_2018Q4.csv>
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.tieuluan.loan import FEATURES, INPUT_COLUMNS, build_features

ROOT = Path(__file__).resolve().parents[2]
RAW_OUT = ROOT / "data/tieuluan/raw/lending_club/loans_36m_2012_2015_150k.csv"
PROCESSED = ROOT / "data/tieuluan/processed"
APP = ROOT / "data/tieuluan/app"
SAMPLE_SIZE = 150_000
ISSUE_FROM, ISSUE_TO = "2012-01-01", "2015-06-30"
TRAIN_END, VAL_END = "2014-08-31", "2014-12-31"      # test: 2015-01 … 2015-06
SOURCE = "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted"
USECOLS = ["loan_amnt", "term", "int_rate", "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
           "verification_status", "issue_d", "loan_status", "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
           "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
           "total_acc", "pub_rec_bankruptcies"]


def load_filtered(path):
    kept, seen = [], 0
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=200_000, low_memory=False):
        seen += len(chunk)
        chunk = chunk[chunk.term.str.strip().eq("36 months")
                      & chunk.loan_status.isin(["Fully Paid", "Charged Off", "Default"])]
        chunk["issue_date"] = pd.to_datetime(chunk.issue_d, format="%b-%Y", errors="coerce")
        chunk = chunk[(chunk.issue_date >= ISSUE_FROM) & (chunk.issue_date <= ISSUE_TO)]
        kept.append(chunk)
    return pd.concat(kept, ignore_index=True), seen


def tidy(frame):
    """Chuyển cột thô về đúng kiểu/đơn vị; loại giá trị vô lý bằng quy tắc CỐ ĐỊNH (không học từ dữ liệu)."""
    f = frame.copy()
    f["issue_date"] = pd.to_datetime(f.issue_d, format="%b-%Y")
    f["fico"] = (f.fico_range_low + f.fico_range_high) / 2
    cr = pd.to_datetime(f.earliest_cr_line, format="%b-%Y", errors="coerce")
    f["credit_history_years"] = ((f.issue_date - cr).dt.days / 365.25).clip(0, 60)
    emp = f.emp_length.astype(str).str.extract(r"(\d+)")[0].astype(float)
    f["emp_length"] = emp                                   # "10+ years" -> 10; thiếu -> NaN
    f.loc[frame.emp_length.astype(str).str.startswith("< 1"), "emp_length"] = 0.0
    f["int_rate"] = pd.to_numeric(f.int_rate.astype(str).str.rstrip("%"), errors="coerce")
    f["revol_util"] = pd.to_numeric(f.revol_util.astype(str).str.rstrip("%"), errors="coerce").clip(0, 150)
    f["dti"] = f.dti.where((f.dti >= 0) & (f.dti <= 60))
    f["annual_inc"] = f.annual_inc.clip(0, 1_000_000)
    f["default"] = (f.loan_status != "Fully Paid").astype(int)
    return f


def main(source):
    frame, seen = load_filtered(source)
    print("Dòng gốc đã duyệt:", seen, "| khoản vay 36 tháng 2012–06/2015 có kết cục:", len(frame))
    eligible_rows = len(frame)
    sample = tidy(frame.sample(SAMPLE_SIZE, random_state=2026)).sort_values("issue_date", kind="stable").reset_index(drop=True)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    keep = ["issue_date", "loan_status", "grade", "sub_grade"] + INPUT_COLUMNS + ["default"]
    sample[keep].rename(columns={"issue_date": "issue_d"}).to_csv(RAW_OUT, index=False, date_format="%Y-%m-%d")
    sha = hashlib.sha256(RAW_OUT.read_bytes()).hexdigest()

    csv = pd.read_csv(RAW_OUT, parse_dates=["issue_d"])        # đọc lại đúng tệp CSV đã lưu: pipeline bắt đầu từ CSV
    x = build_features(csv)
    y = csv["default"].to_numpy(np.float32)
    masks = {"train": (csv.issue_d <= TRAIN_END).to_numpy(),
             "val": ((csv.issue_d > TRAIN_END) & (csv.issue_d <= VAL_END)).to_numpy(),
             "test": (csv.issue_d > VAL_END).to_numpy()}
    train_x = x[masks["train"]]
    median = train_x.median().to_numpy()
    filled = x.fillna(pd.Series(median, index=x.columns))
    mean = filled[masks["train"]].mean().to_numpy()
    scale = filled[masks["train"]].std(ddof=0).replace(0, 1.0).to_numpy()
    scaled = ((filled.to_numpy() - mean) / scale).astype(np.float32)
    payload = {"scaler_mean": mean, "scaler_scale": scale, "imputer_median": median,
               "feature_names": np.array(FEATURES, dtype=str)}
    stats = {}
    for name, m in masks.items():
        payload[f"{name}_tabular"], payload[f"{name}_y"] = scaled[m], y[m]
        payload[f"{name}_date"] = csv.issue_d[m].to_numpy().astype("datetime64[D]")
        stats[name] = {"n": int(m.sum()), "positive": int(y[m].sum()), "first": str(csv.issue_d[m].min().date()),
                       "last": str(csv.issue_d[m].max().date())}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / "loan_default.npz", **payload)

    APP.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(APP / "loan_default.npz", **{k: v for k, v in payload.items() if k.startswith(("scaler_", "imputer_", "feature"))})
    csv[masks["test"]][["issue_d"] + INPUT_COLUMNS + ["grade", "default"]].to_csv(APP / "loan_test.csv", index=False, date_format="%Y-%m-%d")

    missing = {c: float(csv[c].isna().mean()) for c in INPUT_COLUMNS if csv[c].isna().any()}
    info = {"source": SOURCE, "license": "CC0-1.0", "origin": "Lending Club accepted loans 2007–2018Q4",
            "source_rows": int(seen), "eligible_rows": int(eligible_rows), "raw_rows": int(len(csv)),
            "features": len(FEATURES), "raw_columns": len(INPUT_COLUMNS), "positive": int(y.sum()),
            "base_rate": float(y.mean()), "issue_range": [str(csv.issue_d.min().date()), str(csv.issue_d.max().date())],
            "split": "theo ngày phát hành: train ≤ 2014-08-31; val 2014-09…2014-12; test 2015-01…2015-06",
            "splits": stats, "missing_share": missing, "sha256": sha,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_sample": csv.head(3).astype(str).to_dict("records")}
    inv_path = PROCESSED / "inventory.json"
    inventory = json.loads(inv_path.read_text(encoding="utf8")) if inv_path.exists() else {}
    inventory["loan_default"] = info
    inv_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False, default=str), encoding="utf8")
    prov_path = ROOT / "data/tieuluan/provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf8"))
    prov["raw/lending_club/loans_36m_2012_2015_150k.csv"] = {k: info[k] for k in
        ("source", "license", "source_rows", "eligible_rows", "raw_rows", "sha256")} | {"downloaded_at": info["created_at"]}
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(info, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main(sys.argv[1])
