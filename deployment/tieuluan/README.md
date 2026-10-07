# Ứng dụng minh họa của tiểu luận

Ứng dụng trực tuyến: [financial-investment-ai.onrender.com](https://financial-investment-ai.onrender.com/).

Ứng dụng Streamlit có bảy tab: **Giới thiệu**, **Thử trên dữ liệu kiểm tra**, **Dự báo phiên tới**,
**Tự chấm điểm tín dụng**, **So sánh mô hình**, **Backtest**, **Dữ liệu và giới hạn**.
Chọn ngày để đối chiếu giá, dự báo và kết quả thật; nhập hồ sơ tín dụng để chạy MLP;
xem mô phỏng 100 triệu đồng trước và sau phí. Câu chữ dành cho người chưa học tài chính.
Ứng dụng chỉ suy luận bằng NumPy, không huấn luyện lại, không đổi kết quả thực nghiệm.
Ba thư viện trực tiếp trong requirements giữ nguyên; requests và Altair là phụ thuộc của Streamlit.

## Chạy tại máy

Từ thư mục gốc dự án:

```powershell
.venv\Scripts\python.exe -m pip install -r deployment/tieuluan/requirements.txt
.venv\Scripts\python.exe -m streamlit run deployment/tieuluan/app.py
.venv\Scripts\python.exe -m scripts.tieuluan.prepare_app_assets
.venv\Scripts\python.exe -m pytest tests/test_tieuluan.py tests/test_tieuluan_app.py tests/test_tieuluan_live.py tests/test_tieuluan_credit.py -q
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
không bảo đảm cùng nguồn luôn truy cập được từ Render. Mô hình không tự học từ dữ liệu mới.

## Kiểm chứng và ảnh báo cáo

`data/tieuluan/app/verification.json` ghi commit ứng dụng đã thử bằng `git archive`, nguồn mạng thực tế
và số đo RAM của môi trường chỉ cài requirements. Đây là bằng chứng trên máy Windows, không phải cam kết
tài nguyên cho mọi lượng truy cập trên Render. Gói Space được kiểm tra thêm bằng Streamlit AppTest.

Để thử giao diện bằng Edge và chụp lại ảnh (dùng venv phát triển có Playwright và Pillow):

```powershell
.venv\Scripts\python.exe -m streamlit run deployment/tieuluan/app.py --server.port 8517 --client.toolbarMode minimal
.venv\Scripts\python.exe -m scripts.tieuluan.check_app_browser --url http://localhost:8517 --live --screenshots
```

Lệnh trình duyệt gọi mạng thật khi có `--live`; các kiểm thử pytest đều giả lập tải mạng.
Biểu mẫu dùng dấu chấm phân cách hàng nghìn, dấu phẩy thập phân. Ba mức tín dụng là quy ước minh họa
quanh ngưỡng validation: dưới nửa ngưỡng, từ nửa ngưỡng đến dưới ngưỡng, từ ngưỡng trở lên;
không phải thang tín dụng đã kiểm định.

## Triển khai miễn phí trên Render (cách đang dùng)

Render lấy mã thẳng từ kho GitHub công khai, không cần đóng gói hay tải file lên.

1. Đăng nhập <https://render.com> (có thể đăng nhập bằng tài khoản GitHub).
2. **New → Web Service** → thẻ **Public Git Repository** → dán
   `https://github.com/Sagitoaz/intelligent-system-assignment-01` → **Connect**.
3. Điền:
   - **Name:** tên tùy chọn, ví dụ `ai-dau-tu-tai-chinh` (thành địa chỉ `https://ai-dau-tu-tai-chinh.onrender.com`).
   - **Language:** `Python 3`; **Branch:** `main`; **Region:** `Singapore` (gần Việt Nam nhất); **Root Directory:** để trống.
   - **Build Command:** `pip install -r deployment/tieuluan/requirements.txt`
   - **Start Command:**
     `streamlit run deployment/tieuluan/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Instance Type:** `Free`.
4. Bấm **Create Web Service**, chờ log báo build xong và trạng thái **Live** (lần đầu vài phút).

Python 3.12 được khai báo trong file `.python-version` ở gốc kho mã nên không cần đặt biến `PYTHON_VERSION`.
Gói Free có 512 MB RAM, 0,1 CPU; dịch vụ ngủ sau 15 phút không có truy cập và mất khoảng một phút để thức dậy,
nên hãy mở link trước khi trình bày khoảng một phút. Mỗi lần push lên `main`, Render tự build lại.

## Phương án khác

- **Streamlit Community Cloud** (miễn phí, chuyên cho Streamlit): đăng nhập <https://share.streamlit.io> bằng GitHub →
  **Create app** → chọn repo `Sagitoaz/intelligent-system-assignment-01`, nhánh `main`,
  file `deployment/tieuluan/app.py` → **Advanced settings**: Python 3.12 → **Deploy**.
- **Hugging Face Spaces (Docker)**: từ tháng 7/2026 Hugging Face chỉ cho tài khoản PRO (hoặc Team/Enterprise) tạo Space
  dạng Docker/Gradio; tài khoản miễn phí chỉ tạo được Space tĩnh. Nếu có tài khoản trả phí, chạy
  `.venv\Scripts\python.exe -m scripts.tieuluan.package_demo` để tạo `deployment/tieuluan/tieuluan_hf_space.zip`
  (Dockerfile, cổng 7860, người dùng UID 1000), tạo Space **Docker → Blank** rồi tải toàn bộ nội dung đã giải nén lên.
