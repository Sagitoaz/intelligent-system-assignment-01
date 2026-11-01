# Ứng dụng minh họa của tiểu luận

Ứng dụng Streamlit cho người chưa biết gì về AI: mỗi chương mở đầu bằng khung "Demo này làm gì?" và "Cách đọc kết quả",
kết luận là câu văn, có thanh đo và biểu đồ màu sáng. Chọn chương bằng `?chapter=2|3|4` (`&embed=true` để nhúng iframe).

- **Chương 2**: nhập hồ sơ vay bằng thanh trượt hoặc lấy một hồ sơ thật trong tập kiểm tra (Lending Club, 36 tháng, 2012–2015);
  hiện khả năng không trả được nợ, ba yếu tố ảnh hưởng nhất, so sánh ba mô hình, kiểm chứng với kết cục thật; tab so sánh và tab dữ liệu.
- **Chương 3**: chọn thị trường và một ngày đã biết; đồ thị giá có vùng 20 phiên được tô màu, ảnh GASF, xác suất tăng, kết quả thật, tỉ lệ đúng
  trên cả giai đoạn kiểm tra so với đoán mò; dự báo phiên sắp tới; so sánh mô hình; dữ liệu và giới hạn.
- **Chương 4**: như Chương 3 với 20 mức thay đổi giá và LSTM, thêm backtest 100 triệu đồng trước/sau phí so với mua và giữ.

Mã chia thành `app.py` (điều hướng), `common.py` (định dạng, thanh đo), `ch2.py`, `market.py` (Chương 3 và 4).
Ứng dụng chỉ suy luận bằng trọng số đã lưu, không huấn luyện lại, không đổi kết quả thực nghiệm.

## Chạy tại máy

Từ thư mục gốc dự án:

```powershell
.venv\Scripts\python.exe -m pip install -r deployment/tieuluan/requirements.txt
.venv\Scripts\python.exe -m streamlit run deployment/tieuluan/app.py --server.port 8502
.venv\Scripts\python.exe -m scripts.tieuluan.prepare_loan_data <accepted_2007_to_2018Q4.csv>   # dữ liệu Chương 2
.venv\Scripts\python.exe -m scripts.tieuluan.run_loan                                          # huấn luyện Chương 2
.venv\Scripts\python.exe -m pytest tests/test_tieuluan.py tests/test_tieuluan_app.py tests/test_tieuluan_live.py tests/test_tieuluan_credit.py tests/test_tieuluan_loan.py -q
```

Ứng dụng đọc `results/tieuluan/full`, `models/tieuluan/full` (hạt giống 11, 22, 33),
`data/tieuluan/processed` và `data/tieuluan/app`. Biến `TIEULUAN_ROOT` cho phép đổi thư mục tài nguyên.
Chạy `prepare_app_assets` bằng venv đầy đủ khi cần sinh lại tài nguyên: script đọc Excel/CSV gốc theo
`test_ids`, lưu tham số, hồ sơ thật ở phân vị 10/50/90% và chọn ngưỡng tín dụng từ validation.
Không chạy script chuẩn bị dữ liệu hoặc huấn luyện khi triển khai ứng dụng.

## Dự báo bằng giá mới

| Thị trường | Nguồn chính | Dự phòng |
|---|---|---|
| S&P 500 | Yahoo Finance chart, ^GSPC | FRED SP500 |
| Bitcoin | Yahoo Finance chart, BTC-USD | Coinbase Exchange BTC-USD |
| VN-Index | SSI iBoard | DNSE Entrade |

Chỉ tải khi bấm nút; cache 30 phút, nút tải lại xóa cache. Mỗi nguồn có timeout 9 giây,
tối đa một lần thử lại. Nguồn và thời điểm thực tế luôn hiển thị; không thay dữ liệu lỗi bằng dữ liệu giả.
Bỏ phiên VN-Index trước 15:00 giờ Việt Nam, S&P 500 trước 16:30 giờ New York, và nến Bitcoin ngày UTC hiện tại.
Cần 21 giá hợp lệ; cảnh báo khi phiên cuối cũ hơn 5 ngày (chứng khoán) hoặc 2 ngày (Bitcoin).
LSTM dùng 20 lợi suất log chuẩn hóa với tham số cũ; CNN4 dùng ảnh GASF của 20 giá.
Dự báo mới là trung bình xác suất ba hạt giống. Ngưỡng thị trường và khoảng tin cậy lấy từ
`analysis.json` (phân tích trung bình ba bản PyTorch); độ chính xác cân bằng lấy từ ba bản NumPy.
Ngưỡng tín dụng được sinh riêng từ trung bình validation NumPy. Tất cả xác suất chưa được hiệu chuẩn.

Các nguồn công khai có thể chậm, chặn truy cập hoặc đổi định dạng. Việc tải được trên máy cá nhân
không bảo đảm cùng nguồn luôn truy cập được ở mọi lúc. Mô hình không tự học từ dữ liệu mới.

## Kiểm chứng và ảnh báo cáo

`data/tieuluan/app/verification.json` ghi commit ứng dụng đã thử bằng `git archive`, nguồn mạng thực tế
và số đo RAM của môi trường chỉ cài requirements. Đây là bằng chứng trên máy Windows.

Các kiểm thử pytest giả lập tải mạng; ứng dụng chỉ gọi mạng thật khi bấm nút tải giá.
Biểu mẫu dùng dấu chấm phân cách hàng nghìn, dấu phẩy thập phân. Ba mức tín dụng là quy ước minh họa
quanh ngưỡng validation: dưới nửa ngưỡng, từ nửa ngưỡng đến dưới ngưỡng, từ ngưỡng trở lên;
không phải thang tín dụng đã kiểm định.

Giao diện sáng và cấu hình localhost nằm trong `.streamlit/config.toml` ở thư mục gốc. Xem `RUN.md` để biết mọi lệnh chạy.
