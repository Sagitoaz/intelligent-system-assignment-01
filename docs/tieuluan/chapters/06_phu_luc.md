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
.venv\Scripts\python.exe -m pytest tests/test_tieuluan.py tests/test_tieuluan_app.py tests/test_tieuluan_live.py tests/test_tieuluan_credit.py -q
```

## Phụ lục B. Ứng dụng minh họa và triển khai

Ứng dụng Streamlit dành cho người chưa học tài chính, gồm bảy thẻ, mỗi thẻ có ô “Cách đọc”. **Giới thiệu** giải thích hai bài toán và kết luận chính; **Thử trên dữ liệu kiểm tra** chọn ngày hoặc hồ sơ thật để đối chiếu dự báo; **Dự báo phiên tới** tải giá mới; **Tự chấm điểm tín dụng** nhận thông tin người dùng nhập; **So sánh mô hình** tổng hợp {{v:n_records}} bản ghi và biểu đồ ROC-AUC, với vạch đoán mò; **Backtest** mô phỏng vốn bằng tiền trước, sau phí; **Dữ liệu và giới hạn** giải thích nguồn và phạm vi sử dụng. Ứng dụng không huấn luyện lại và không thay đổi kết quả Chương 2–4.

**Đối chiếu lịch sử.** Chọn thị trường, CNN4 hoặc LSTM và ngày cuối cửa sổ; biểu đồ hiển thị giá đóng cửa theo ngày thật, dự báo được so với biến động phiên tiếp theo (Hình B.1). Với dữ liệu bảng, chọn hoặc lấy ngẫu nhiên một dòng kiểm tra, xem thông tin gốc và kết quả thật. Ảnh GASF và vector chuẩn hóa nằm trong phần mở rộng; dự báo NumPy hạt giống 11 được đối chiếu với xác suất đã lưu.

![Hình B.1. Chọn ngày để đối chiếu dự báo LSTM với diễn biến phiên tiếp theo của VN-Index.](../../figures/tieuluan/appendix_demo_app.png){w=12}

**Dự báo phiên tới.** Nút tải lấy giá ngày từ Yahoo Finance cho S&P 500 và Bitcoin [@yahoo_live2026], dự phòng lần lượt FRED [@fred_live2026] và Coinbase Exchange [@coinbase_live2026]; VN-Index dùng SSI iBoard [@ssi_live2026], dự phòng DNSE [@dnse_live2026]. Giao diện ghi nguồn thực tế, phiên cuối và giờ cập nhật Việt Nam (Hình B.2). Chỉ dùng phiên đã đóng cửa: bỏ phiên hôm nay trước 15:00 giờ Việt Nam với VN-Index, trước 16:30 giờ New York với S&P 500 và bỏ nến ngày UTC hiện tại với Bitcoin. UTC là giờ quốc tế, chậm hơn Việt Nam bảy giờ. Cần ít nhất 21 giá hợp lệ; giá quá cũ được cảnh báo, lỗi mạng không được thay bằng dữ liệu giả.

LSTM nhận 20 lợi suất logarit, tức thay đổi giá trên thang logarit, với tham số chuẩn hóa của tập học; CNN4 nhận ảnh GASF của 20 giá cuối. Công thức và thứ tự ép kiểu được kiểm thử khớp đặc trưng thực nghiệm. Dự báo lấy trung bình xác suất ba hạt giống NumPy; ngưỡng và khoảng tin cậy ROC-AUC lấy từ analysis.json, vốn phân tích trung bình ba bản PyTorch gần tương đương NumPy. Độ chính xác cân bằng hiển thị là trung bình ba bản NumPy trên tập kiểm tra, không phải độ tin cậy riêng của phiên mới. Tải giá không cập nhật trọng số; nguồn dự phòng có thể khác nguồn huấn luyện.

![Hình B.2. Dự báo phiên tới từ giá thật: nguồn, phiên cuối và xác suất của LSTM, CNN4.](../../figures/tieuluan/appendix_live.png){w=12}

**Chấm điểm tín dụng.** Chọn một hồ sơ thật ở nhóm xác suất thấp, giữa hoặc cao trong tập kiểm tra, sửa hạn mức, tuổi, lịch sử trả nợ và số tiền rồi bấm chấm điểm (Hình B.3). Các thông tin khác nằm trong phần mở rộng. Đơn vị là Đài tệ (TWD), không phải VND; dữ liệu thuộc Đài Loan năm 2005. Mã trả nợ được giải thích cạnh biểu mẫu; hai mã −2 và 0 dùng diễn giải phổ biến nhưng không được tài liệu UCI gốc mô tả. MLP sử dụng đủ đặc trưng theo thứ tự gốc, điền thiếu và chuẩn hóa bằng tham số tập học, rồi lấy trung bình ba xác suất. Ngưỡng {{v:app.credit.threshold}} được chọn trước từ trung bình dự báo validation; tỷ lệ vỡ nợ chung để so sánh là {{v:positive_rate.credit_default}}. Ba mức thấp, trung bình, cao lần lượt ứng với xác suất dưới nửa ngưỡng, từ nửa ngưỡng đến dưới ngưỡng và từ ngưỡng trở lên; đây là quy ước minh họa chưa được kiểm định như thang tín dụng. Giới tính, tuổi và tình trạng hôn nhân có thể gây phân biệt đối xử nếu dùng để quyết định tín dụng thực tế; ứng dụng chỉ phục vụ học tập.

![Hình B.3. Hồ sơ tín dụng thật làm mẫu cho biểu mẫu nhập và chấm điểm bằng MLP.](../../figures/tieuluan/appendix_credit.png){w=12}

**Mô phỏng bằng tiền.** Backtest là thử cách mua bán trên dữ liệu quá khứ. Hình B.4 quy đổi về vốn ban đầu 100 triệu đồng, so sánh mua và giữ, theo mô hình và lặp lại hướng phiên trước; bảng trình bày số tiền trước, sau phí. Đây là vốn quy đổi để dễ đọc, không tính tỷ giá và không khẳng định có thể mua trực tiếp chỉ số. Mọi dự báo đều kèm kết quả lịch sử và câu “Không phải khuyến nghị đầu tư”; riêng S&P 500, Bitcoin nêu rõ không tốt hơn đoán mò trên giai đoạn kiểm tra.

![Hình B.4. Backtest VN-Index diễn đạt bằng triệu đồng, với bảng kết quả trước và sau phí.](../../figures/tieuluan/appendix_demo_backtest.png){w=12}

**Chạy và triển khai.** Tài nguyên nhẹ trong data/tieuluan/app được tạo trước bằng lệnh `.venv\Scripts\python.exe -m scripts.tieuluan.prepare_app_assets`; script đọc dữ liệu gốc theo chỉ số dòng kiểm tra, tính ngưỡng từ validation và lưu mẫu, tham số, thống kê. Máy chủ chỉ cài NumPy, pandas, Streamlit; không đọc Excel, không nạp thư viện huấn luyện. Tệp và mô hình có bộ nhớ đệm; giá tải mạng được giữ tối đa 30 phút. Nút tải lại xóa bộ nhớ đệm giá. Mỗi nguồn chờ tối đa chín giây mỗi lần, chỉ thử lại một lần.

Ứng dụng đã được thử từ bản git archive trong môi trường Python sạch chỉ cài requirements triển khai: Edge đi qua các thẻ, lựa chọn mô hình, hồ sơ mẫu và dự báo mạng thật; đỉnh bộ nhớ tiến trình máy chủ đo bằng Get-Process là {{v:app.peak_memory_mb}} MB. Đây là phép đo trên Windows, không phải cam kết bộ nhớ cho mọi mức truy cập trên Render.

Ứng dụng trực tuyến: [financial-investment-ai.onrender.com](https://financial-investment-ai.onrender.com/). Render lấy mã trực tiếp từ GitHub và tự triển khai khi nhánh main được cập nhật [@render_free]. Các bước:

1. Đăng nhập render.com, chọn **New → Web Service → Public Git Repository**, dán địa chỉ kho mã ở Phụ lục A rồi bấm **Connect**.
2. Chọn **Language: Python 3**, **Branch: main**, gói **Free**; Python 3.12 khai báo tại `.python-version`.
3. **Build Command:** `pip install -r deployment/tieuluan/requirements.txt`.
4. **Start Command:** `streamlit run deployment/tieuluan/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`.
5. Bấm **Create Web Service**, chờ **Live** rồi mở địa chỉ Render được cấp.

Gói Free có 512 MB bộ nhớ, ngủ sau 15 phút không truy cập; nên mở trước khi trình bày khoảng một phút. Gói Docker cho Hugging Face vẫn có thể tạo bằng `scripts.tieuluan.package_demo`, gồm tài nguyên và trọng số ba hạt giống. Chạy tại máy bằng `.venv\Scripts\python.exe -m streamlit run deployment/tieuluan/app.py`, rồi mở http://localhost:8501.
