"""Tiền xử lý tín dụng bằng tham số đã học; tên và giải thích dùng chung cho giao diện."""
import numpy as np

PAY_CODES = {-2: 'Không phát sinh chi tiêu (diễn giải phổ biến)', -1: 'Trả đủ',
             0: 'Chỉ trả tối thiểu (diễn giải phổ biến)',
             **{i: f'Trễ {i} tháng' for i in range(1, 9)}, 9: 'Trễ từ 9 tháng trở lên'}
CATEGORIES = {
    'SEX': {1: 'Nam', 2: 'Nữ'},
    'EDUCATION': {0: 'Không rõ (mã 0)', 1: 'Sau đại học', 2: 'Đại học', 3: 'Trung học', 4: 'Khác', 5: 'Không rõ (mã 5)', 6: 'Không rõ (mã 6)'},
    'MARRIAGE': {0: 'Không rõ', 1: 'Đã kết hôn', 2: 'Độc thân', 3: 'Khác'}}
LABELS = {'LIMIT_BAL': 'Hạn mức tín dụng (TWD)', 'AGE': 'Tuổi', 'SEX': 'Giới tính',
          'EDUCATION': 'Học vấn', 'MARRIAGE': 'Tình trạng hôn nhân'}
for i, pay in enumerate(['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']):
    LABELS[pay] = f'Trả nợ tháng {9-i}/2005'
    LABELS[f'BILL_AMT{i+1}'] = f'Dư nợ tháng {9-i}/2005 (TWD)'
    LABELS[f'PAY_AMT{i+1}'] = f'Tiền đã trả tháng {9-i}/2005 (TWD)'


def preprocess_credit(values, params):
    x = np.atleast_2d(np.asarray(values, dtype=np.float64))
    median, mean, scale = [np.asarray(params[k], dtype=float) for k in ('imputer_median', 'scaler_mean', 'scaler_scale')]
    if x.shape[1:] != median.shape or mean.shape != median.shape or scale.shape != median.shape:
        raise ValueError('Số đặc trưng không khớp mô hình đã lưu.')
    x = np.where(np.isnan(x), median, x)
    if not np.all(np.isfinite(x)) or np.any(scale <= 0):
        raise ValueError('Thông tin nhập hoặc tham số chuẩn hóa không hợp lệ.')
    return ((x - mean) / scale).astype(np.float32)
