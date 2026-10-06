# Ứng dụng minh họa của tiểu luận

Ứng dụng Streamlit gồm bốn tab: **Thử mô hình** (chạy mô hình NumPy tự viết trên một mẫu của tập test),
**So sánh thực nghiệm** (128 bản ghi, ba cách cài đặt, ba hạt giống), **Backtest** (giá trị danh mục có tính phí)
và **Dữ liệu và giới hạn**. Ứng dụng chỉ đọc kết quả đã lưu: không tải TensorFlow/PyTorch, không huấn luyện,
không chọn lại ngưỡng.

## Chạy tại máy

Từ thư mục gốc dự án:

```powershell
.venv\Scripts\python.exe -m pip install -r deployment/tieuluan/requirements.txt
.venv\Scripts\python.exe -m streamlit run deployment/tieuluan/app.py
.venv\Scripts\python.exe -m pytest tests/test_tieuluan_app.py -q
```

Ứng dụng đọc `results/tieuluan/full`, `models/tieuluan/full` (các file `*__scratch__11.npz`) và
`data/tieuluan/processed`. Biến môi trường `TIEULUAN_ROOT` cho phép trỏ tới một thư mục khác có cùng cấu trúc.

## Đóng gói cho Hugging Face Spaces (Docker)

```powershell
.venv\Scripts\python.exe -m scripts.tieuluan.package_demo
```

Lệnh tạo thư mục `deployment/tieuluan/hf_space/` và file `deployment/tieuluan/tieuluan_hf_space.zip`. Gói chỉ chứa
mã suy luận, trọng số NumPy hạt giống 11, kết quả đánh giá và **phần test** của dữ liệu đã tiền xử lý
(không có dữ liệu gốc, train hay validation), kèm `SHA256_MANIFEST.json`.

Các bước tải lên:

1. Đăng nhập <https://huggingface.co>, chọn **New → Space**.
2. Đặt tên Space, chọn **SDK: Docker → Blank**, phần cứng **CPU basic (miễn phí)**. Nên để **Private** cho tới khi
   kiểm tra xong điều khoản phân phối dữ liệu thị trường (xem `DATA_SOURCES.md` trong gói).
3. Trong Space vừa tạo: **Files → Add file → Upload files**. Giải nén file ZIP và kéo **toàn bộ nội dung bên trong**
   (gồm `Dockerfile`, `README.md`, các thư mục `src`, `deployment`, `data`, `models`, `results`) vào, rồi **Commit**.
   Không tải nguyên file ZIP và không tạo thêm tầng thư mục `hf_space`.
4. Chờ trạng thái chuyển từ *Building* sang *Running* (vài phút), mở tab **App** và thử cả bốn tab.

`README.md` trong gói đã có phần khai báo `sdk: docker` và `app_port: 7860`; `Dockerfile` chạy bằng người dùng
UID 1000 theo [hướng dẫn Docker Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-docker).
Có thể build thử tại máy trước khi tải lên:

```text
cd deployment/tieuluan/hf_space
docker build -t tieuluan-demo .
docker run --rm -p 7860:7860 tieuluan-demo
```
