# Ứng dụng minh họa của tiểu luận

Ứng dụng Streamlit gồm bốn tab: **Thử mô hình** (chạy mô hình NumPy tự viết trên một mẫu của tập test),
**So sánh thực nghiệm** (128 bản ghi, ba cách cài đặt, ba hạt giống), **Backtest** (giá trị danh mục có tính phí)
và **Dữ liệu và giới hạn**. Ứng dụng chỉ đọc kết quả đã lưu: không tải TensorFlow/PyTorch, không huấn luyện,
không chọn lại ngưỡng. Chỉ cần ba thư viện trong `requirements.txt` (NumPy, pandas, Streamlit).

## Chạy tại máy

Từ thư mục gốc dự án:

```powershell
.venv\Scripts\python.exe -m pip install -r deployment/tieuluan/requirements.txt
.venv\Scripts\python.exe -m streamlit run deployment/tieuluan/app.py
.venv\Scripts\python.exe -m pytest tests/test_tieuluan_app.py -q
```

Ứng dụng đọc `results/tieuluan/full`, `models/tieuluan/full` (các file `*__scratch__11.npz`) và
`data/tieuluan/processed`. Biến môi trường `TIEULUAN_ROOT` cho phép trỏ tới một thư mục khác có cùng cấu trúc.

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
