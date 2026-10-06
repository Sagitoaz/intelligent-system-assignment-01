# PHỤ LỤC

## Phụ lục A. Mã nguồn và cách tái lập thực nghiệm

Toàn bộ mã nguồn, dữ liệu đã xử lý, kết quả và trọng số mô hình nằm trong kho mã của tiểu luận ([github.com/Sagitoaz/intelligent-system-assignment-01](https://github.com/Sagitoaz/intelligent-system-assignment-01)). Phần dành cho tiểu luận được tổ chức như sau:

```text
src/tieuluan/
  config.py              mốc thời gian, danh sách tài sản, đường dẫn
  data/sources.py        tải dữ liệu: Yahoo Finance, SSI, DNSE, VNDirect, UCI
  experiment_data.py     cửa sổ 20 phiên, ảnh GASF, chia dữ liệu
  experiment_models.py   kiến trúc, chép trọng số, vòng huấn luyện
  scratch/               thư viện NumPy tự viết (Dense, Conv2d, LSTM, GRU…)
  analysis.py            bootstrap khoảng tin cậy, backtest có phí
scripts/tieuluan/        các bước chạy (Bảng A.1)
deployment/tieuluan/     ứng dụng Streamlit và Dockerfile
notebooks/tieuluan/      notebook minh họa cho Chương 2, 3, 4
tests/                   kiểm thử tự động (pytest)
```

Bảng A.1. Trình tự tái lập toàn bộ kết quả (chạy từ thư mục gốc, Windows PowerShell).
| Bước | Lệnh (.venv\Scripts\python.exe -m …) | Kết quả |
|---|---|---|
| 1 | scripts.tieuluan.download_data | Dữ liệu gốc, nguồn gốc và kiểm tra chất lượng (data/tieuluan) |
| 2 | src.tieuluan.experiment_data | Năm bộ dữ liệu đã chia và tiền xử lý (data/tieuluan/processed) |
| 3 | scripts.tieuluan.run_experiments | {{v:n_records}} bản ghi kết quả, dự báo từng mẫu, trọng số (results, models) |
| 4 | scripts.tieuluan.analyze_results | Khoảng tin cậy bootstrap và backtest (analysis.json) |
| 5 | scripts.tieuluan.verify_implementations | Kiểm tra gradient và song trùng ba bản cài đặt |
| 6 | scripts.tieuluan.sensitivity_tabular, scripts.tieuluan.ablation_gasf | Hai phép thử phụ ở mục 2.11 và 3.9 |
| 7 | scripts.tieuluan.audit_experiments | Kiểm toán chỉ đọc: tính lại mọi thước đo, nạp lại các mô hình đã lưu |
| 8 | scripts.tieuluan.build_notebooks --execute | Ba notebook minh họa Chương 2, 3, 4 (có kết quả chạy) |
| 9 | scripts.tieuluan.build_report | Báo cáo Word với mọi số liệu, bảng và hình sinh từ kết quả |

Bước 3 chạy trên CPU; tổng thời gian huấn luyện của {{v:n_trained}} lượt là khoảng 9 phút. Mỗi bản ghi lưu kèm hạt giống, cấu hình, lịch sử huấn luyện và "dấu vân tay" của lần chạy; file results/tieuluan/full/run_manifest.json ghi mã băm SHA-256 của dữ liệu đã xử lý và của mã nguồn tạo ra kết quả, nên có thể kiểm tra một con số trong báo cáo đến từ phiên bản dữ liệu và mã nào. Bước 7 tính lại toàn bộ {{v:audit.records}} bản ghi từ dự báo đã lưu và nạp lại {{v:audit.reloaded}} mô hình đã huấn luyện; xác suất dự báo lại {{v:audit.match}} file đã lưu. Mỗi notebook ở bước 8 huấn luyện lại hạt giống 11 bằng cả ba cách cài đặt ngay trong notebook và tái lập đúng ROC-AUC đã báo cáo. Các kiểm thử tự động, gồm kiểm tra gradient, kiểm tra không rò rỉ dữ liệu qua ranh giới thời gian và kiểm tra ứng dụng minh họa, được chạy bằng lệnh:

```text
.venv\Scripts\python.exe -m pytest tests/test_tieuluan.py tests/test_tieuluan_app.py -q
```

## Phụ lục B. Ứng dụng minh họa và triển khai

Ứng dụng web được viết bằng Streamlit, gồm bốn thẻ: **Thử mô hình** chạy mô hình NumPy tự viết trên một mẫu bất kỳ của tập test và đối chiếu với dự báo đã lưu (Hình B.1); **So sánh thực nghiệm** tổng hợp {{v:n_records}} bản ghi theo mô hình và cách cài đặt; **Backtest** vẽ giá trị danh mục của các chiến lược (Hình B.2); **Dữ liệu và giới hạn** nêu nguồn dữ liệu và các lưu ý khi đọc kết quả. Ứng dụng chỉ đọc kết quả đã lưu: không huấn luyện lại, không chọn lại ngưỡng, và chỉ cần ba thư viện NumPy, pandas, Streamlit.

![Hình B.1. Thẻ "Thử mô hình": LSTM tự viết dự báo một mẫu VN-Index của tập test và tự đối chiếu với dự báo đã lưu.](../../figures/tieuluan/appendix_demo_app.png)

![Hình B.2. Thẻ "Backtest": giá trị danh mục trên tập test của VN-Index và bảng chỉ số trước, sau phí.](../../figures/tieuluan/appendix_demo_backtest.png)

Ứng dụng được triển khai thành một dịch vụ web miễn phí trên Render [@render_free]. Render lấy mã trực tiếp từ kho GitHub công khai của tiểu luận, và tự cài đặt, khởi động lại ứng dụng mỗi khi có mã mới trên nhánh `main`. Gói miễn phí có 512 MB bộ nhớ; dịch vụ tự "ngủ" sau 15 phút không có truy cập và mất khoảng một phút để thức dậy ở lượt truy cập kế tiếp. Các bước triển khai:

1. Đăng nhập render.com, chọn **New → Web Service**, chọn thẻ **Public Git Repository**, dán địa chỉ kho mã GitHub của tiểu luận rồi bấm **Connect**.
2. Chọn **Language: Python 3**, **Branch: main**, gói **Free**. Phiên bản Python 3.12 được khai báo sẵn trong file `.python-version` ở gốc kho mã.
3. **Build Command:** `pip install -r deployment/tieuluan/requirements.txt` (chỉ gồm NumPy, pandas và Streamlit).
4. **Start Command:** `streamlit run deployment/tieuluan/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`.
5. Bấm **Create Web Service**, chờ trạng thái **Live** rồi mở địa chỉ dạng `https://<tên-dịch-vụ>.onrender.com`.

Trước khi triển khai, ứng dụng được chạy thử trong một môi trường Python sạch chỉ có ba thư viện trên, từ đúng nội dung đã đẩy lên GitHub: cả tám lựa chọn mô hình và bốn thẻ đều hoạt động, bộ nhớ sử dụng dưới 200 MB. Kho mã vẫn giữ gói Docker cho Hugging Face Spaces (`scripts.tieuluan.package_demo`), nhưng từ tháng 7/2026 Space dạng Docker chỉ tạo được với tài khoản trả phí [@hf_spaces2026]. Để chạy tại máy, dùng lệnh dưới đây rồi mở địa chỉ http://localhost:8501:

```text
.venv\Scripts\python.exe -m streamlit run deployment/tieuluan/app.py
```
