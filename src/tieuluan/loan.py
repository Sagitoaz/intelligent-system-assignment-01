"""Bài toán Chương 2: dự đoán khoản vay có bị "xóa nợ" (không trả được) hay không.

Dữ liệu: Lending Club (cho vay ngang hàng, Mỹ), khoản vay 36 tháng phát hành 01/2012–06/2015,
kết cục đã biết đến hết 2018. Mọi đặc trưng ở đây đều biết TẠI THỜI ĐIỂM XIN VAY; không dùng
cột nào sinh ra sau khi khoản vay bắt đầu (số tiền đã trả, ngày thanh toán cuối, v.v.).

Cùng một hàm ``build_features`` được dùng khi huấn luyện và khi demo nhập tay, nên không có
chuyện hai pipeline lệch nhau. Tệp này chỉ cần numpy/pandas để app chạy nhẹ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TERM_MONTHS = 36

HOME = {"RENT": "Thuê nhà", "MORTGAGE": "Đang trả góp mua nhà", "OWN": "Sở hữu nhà", "OTHER": "Khác"}
VERIFICATION = {"Not Verified": "Chưa xác minh thu nhập", "Source Verified": "Đã xác minh qua nguồn",
                "Verified": "Đã xác minh thu nhập"}
PURPOSE = {"debt_consolidation": "Gộp các khoản nợ cũ", "credit_card": "Trả nợ thẻ tín dụng",
           "home_improvement": "Sửa nhà", "major_purchase": "Mua sắm lớn", "small_business": "Kinh doanh nhỏ",
           "car": "Mua xe", "medical": "Chi phí y tế", "moving": "Chuyển nhà", "other": "Mục đích khác"}

# key, nhãn tiếng Việt, đơn vị, nhỏ nhất, lớn nhất, mặc định, bước, giải thích
NUMERIC_FIELDS = [
    ("loan_amnt", "Số tiền xin vay", "USD", 1000, 40000, 12000, 500, "Số tiền người vay xin trong hồ sơ."),
    ("int_rate", "Lãi suất khoản vay", "%/năm", 5.0, 31.0, 13.0, 0.25,
     "Lãi suất do nền tảng ấn định sau khi chấm hồ sơ; hồ sơ rủi ro hơn bị áp lãi cao hơn."),
    ("annual_inc", "Thu nhập hằng năm", "USD", 10000, 500000, 65000, 1000, "Thu nhập người vay tự khai."),
    ("dti", "Tỷ lệ nợ trên thu nhập (DTI)", "%", 0.0, 50.0, 18.0, 0.5,
     "Tổng tiền trả nợ hằng tháng chia cho thu nhập hằng tháng. Càng cao càng căng."),
    ("fico", "Điểm tín dụng FICO", "điểm", 600, 850, 690, 5,
     "Thang 300–850 của Mỹ; trên 740 là rất tốt, dưới 650 là yếu."),
    ("emp_length", "Số năm đi làm", "năm", 0, 10, 5, 1, "10 nghĩa là từ 10 năm trở lên."),
    ("delinq_2yrs", "Số lần trễ hạn 30 ngày trở lên (2 năm qua)", "lần", 0, 10, 0, 1, ""),
    ("inq_last_6mths", "Số lần bị tra cứu tín dụng (6 tháng qua)", "lần", 0, 8, 1, 1,
     "Mỗi lần xin vay hoặc mở thẻ ở nơi khác thường tạo một lượt tra cứu."),
    ("open_acc", "Số tài khoản tín dụng đang mở", "tài khoản", 1, 40, 11, 1, ""),
    ("total_acc", "Tổng số tài khoản tín dụng từng mở", "tài khoản", 2, 80, 24, 1, ""),
    ("pub_rec", "Số hồ sơ xấu công khai", "hồ sơ", 0, 5, 0, 1, "Ví dụ phán quyết của tòa án về nợ."),
    ("pub_rec_bankruptcies", "Số lần từng phá sản cá nhân", "lần", 0, 3, 0, 1, ""),
    ("revol_bal", "Dư nợ quay vòng (thẻ tín dụng…)", "USD", 0, 200000, 14000, 500, ""),
    ("revol_util", "Mức dùng hạn mức tín dụng quay vòng", "%", 0.0, 120.0, 50.0, 1.0,
     "Dư nợ chia hạn mức; dùng sát hạn mức là dấu hiệu căng thẳng tài chính."),
    ("credit_history_years", "Số năm lịch sử tín dụng", "năm", 0.0, 60.0, 16.0, 1.0,
     "Từ lúc mở tài khoản tín dụng đầu tiên đến lúc xin vay."),
]
FIELD = {f[0]: f for f in NUMERIC_FIELDS}
CATEGORICAL_FIELDS = [("home_ownership", "Tình trạng nhà ở", HOME), ("verification_status", "Xác minh thu nhập", VERIFICATION),
                      ("purpose", "Mục đích vay", PURPOSE)]
INPUT_COLUMNS = [f[0] for f in NUMERIC_FIELDS] + [c[0] for c in CATEGORICAL_FIELDS]

NUMERIC_FEATURES = [f[0] for f in NUMERIC_FIELDS] + ["installment", "payment_to_income", "log_annual_inc", "log_revol_bal"]
FEATURES = NUMERIC_FEATURES + [f"{col}={val}" for col, _, mapping in CATEGORICAL_FIELDS for val in mapping]

# Tên dễ hiểu cho từng yếu tố khi giải thích "vì sao" (gom đặc trưng dẫn xuất về yếu tố gốc)
FACTOR_NAME = {f[0]: f[1] for f in NUMERIC_FIELDS} | {c[0]: c[1] for c in CATEGORICAL_FIELDS}


def monthly_payment(amount, annual_rate_pct, months=TERM_MONTHS):
    """Tiền góp hằng tháng theo công thức niên kim: A·r / (1 − (1+r)^−n)."""
    r = np.asarray(annual_rate_pct, dtype=float) / 1200.0
    amount = np.asarray(amount, dtype=float)
    safe = np.where(r > 0, r, 1.0)
    return np.where(r > 0, amount * safe / (1 - (1 + safe) ** -months), amount / months)


def normalise_categories(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["home_ownership"] = f["home_ownership"].where(f["home_ownership"].isin(HOME), "OTHER")
    f["purpose"] = f["purpose"].where(f["purpose"].isin(PURPOSE), "other")
    f["verification_status"] = f["verification_status"].where(f["verification_status"].isin(VERIFICATION), "Not Verified")
    return f


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Bảng đầu vào thô (các cột INPUT_COLUMNS, có thể thiếu giá trị) -> ma trận đặc trưng chưa chuẩn hóa."""
    f = normalise_categories(frame[INPUT_COLUMNS])
    out = pd.DataFrame(index=f.index)
    for key in (x[0] for x in NUMERIC_FIELDS):
        out[key] = pd.to_numeric(f[key], errors="coerce")
    out["installment"] = monthly_payment(out.loan_amnt, out.int_rate)
    income = out.annual_inc.clip(lower=1000)
    out["payment_to_income"] = (out.installment * 12 / income).clip(upper=2.0)
    out["log_annual_inc"] = np.log1p(out.annual_inc.clip(lower=0))
    out["log_revol_bal"] = np.log1p(out.revol_bal.clip(lower=0))
    for col, _, mapping in CATEGORICAL_FIELDS:
        for val in mapping:
            out[f"{col}={val}"] = (f[col] == val).astype(float)
    return out[FEATURES].astype(float)


def preprocess(features, params):
    """Điền khuyết bằng trung vị TRAIN rồi chuẩn hóa bằng trung bình/độ lệch chuẩn TRAIN."""
    x = np.atleast_2d(np.asarray(features, dtype=np.float64))
    median, mean, scale = (np.asarray(params[k], dtype=float) for k in ("imputer_median", "scaler_mean", "scaler_scale"))
    if x.shape[1] != median.shape[0]:
        raise ValueError("Số đặc trưng không khớp mô hình đã lưu.")
    x = np.where(np.isnan(x), median, x)
    return ((x - mean) / np.where(scale > 0, scale, 1.0)).astype(np.float32)


def risk_band(probability, base_rate):
    """Ba mức theo tỉ lệ so với mức trung bình của cả tập: dưới 0,7×, 0,7–1,5×, trên 1,5×."""
    if probability < 0.7 * base_rate:
        return "thấp"
    return "trung bình" if probability < 1.5 * base_rate else "cao"
