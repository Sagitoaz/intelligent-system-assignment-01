# Cách chạy các phần của dự án (localhost)

Tất cả ứng dụng chỉ chạy **trên máy cá nhân** (`localhost`), giao diện nền sáng. Không cần tài khoản đám mây hay API key.
Mở **PowerShell tại thư mục gốc dự án** rồi chạy lệnh của phần cần xem. Dừng ứng dụng bằng `Ctrl + C`.

| Phần | Địa chỉ mở trên trình duyệt | Cổng |
|---|---|---|
| Agent 01 & Agent 02 (demo mèo đuổi chuột, hai người bán) | http://localhost:8100 | 8100 |
| Project 01: Phòng khám AI 3D (hệ đa agent tiểu đường) | http://localhost:8200 | 8200 |
| Assignment 01–02: API (FastAPI) | http://localhost:8000/docs | 8000 |
| Assignment 01–02: Web (React) | http://localhost:5173 | 5173 |
| Assignment 06: Dự báo RNN (Streamlit) | http://localhost:8501 | 8501 |
| Chương 4: Knowledge & RAG agent | http://localhost:8044 | 8044 |
| Tiểu luận: Hiểu AI qua dữ liệu tài chính (Streamlit) | http://localhost:8502 | 8502 |

Các cổng khác nhau nên có thể chạy nhiều phần cùng lúc, mỗi phần trong một cửa sổ PowerShell riêng.

## 0. Chuẩn bị một lần

```powershell
# Python: dùng môi trường ảo có sẵn .venv (Python 3.12)
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt        # chỉ khi thiếu thư viện

# Node.js 20+ (cho web và mobile của Assignment 02)
cd web; npm install; cd ..
```

Nếu không kích hoạt venv, thay `python` bằng `.\.venv\Scripts\python.exe` trong mọi lệnh bên dưới.

## 0. Project 01: Phòng khám AI 3D (hệ đa agent cho bệnh tiểu đường)

```powershell
python project01/server.py --open                # mở http://localhost:8200
python project01/server.py --open --port 8210    # đổi cổng nếu bị trùng
python -m pytest tests/test_project01.py -q      # 7 kiểm thử
# Huấn luyện/đánh giá lại (tùy chọn, kết quả lưu ở project01/artifacts/)
python project01/train/train_dl.py ; python project01/train/build_kg.py ; python project01/train/train_rl.py ; python project01/train/eval_rag.py
```

Thêm `?mock=1` vào địa chỉ để xem dữ liệu mẫu khi chưa chạy máy chủ API.

Giọng nói (tùy chọn): đặt `API_MODEL_TTS` (khóa Azure Speech) và `TTS_ENDPOINT` (địa chỉ khu vực, ví dụ `https://southeastasia.api.cognitive.microsoft.com/`) trong tệp `.env` ở thư mục gốc (tệp này đã được git bỏ qua). Có khóa thì phòng khám vừa chạy phụ đề vừa đọc; không có khóa thì chỉ có chữ. Phím tắt: Dấu cách tạm dừng, V bật/tắt tiếng.

## 1. Agent 01 và Agent 02 (demo mèo đuổi chuột, hai người bán)

```powershell
python agents/demo_app/server.py --open          # mở http://localhost:8100
python agents/demo_app/server.py --port 8110     # đổi cổng nếu bị trùng
```

Chỉ dùng thư viện chuẩn của Python. Q-table của mèo đã lưu sẵn trong `agents/demo_app/cache/`, nên mở lên là chạy, không huấn luyện lại.
Muốn thêm demo mới: tạo `agents/demo_app/demos/<tên>.py` và `agents/demo_app/static/demos/<tên>.js` (xem `agents/README.md`).

Tạo lại các tệp CSV hoặc tải thêm dữ liệu thật (không huấn luyện lại mô hình):

```powershell
python agents/demo_app/build_datasets.py          # tạo lại CSV mô phỏng trong data/agent01 và data/agent02
python agents/demo_app/fetch_external_data.py     # tải UCI Online Retail (cần internet)
```

## 2. Assignment 01–02: ba hệ thống Diabetes, House Price, E-commerce

Cần chạy **API trước, web sau** (hai cửa sổ PowerShell).

```powershell
# Cửa sổ 1: API
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

```powershell
# Cửa sổ 2: Web
cd web
npm run dev                                       # mở http://localhost:5173
```

Web gọi API tại `http://localhost:8000` theo mặc định. Muốn đổi, tạo `web/.env` với dòng
`VITE_API_BASE_URL=http://localhost:<cổng>`.

Ứng dụng di động (Expo), tùy chọn:

```powershell
cd mobile
npm install
npx expo start                                    # quét mã QR bằng Expo Go hoặc nhấn a để mở giả lập Android
```

Mặc định gọi `http://localhost:8000`. Giả lập Android cần địa chỉ `http://10.0.2.2:8000`, điện thoại thật cần IP của máy tính:
đặt biến `EXPO_PUBLIC_API_BASE_URL` trong `mobile/.env`.

Knowledge graph trong web cần Neo4j (tùy chọn). Khi không bật, các trang dự đoán vẫn chạy bình thường:

```powershell
Copy-Item .env.example .env                       # lần đầu, rồi đổi mật khẩu trong .env
docker compose up -d neo4j
$env:NEO4J_ENABLED = "true"                       # đặt trước khi chạy API ở mục trên
```

Kiểm thử API:

```powershell
python -m pytest backend/tests -q
```

## 3. Assignment 03, 04, 05 (chỉ có notebook)

Ba bài này không có ứng dụng web; xem kết quả bằng notebook.

```powershell
python -m jupyter lab notebooks/assignment03      # mạng nơ-ron (Diabetes, House Price, Comments)
python -m jupyter lab notebooks/assignment04      # CNN NumPy, PyTorch, TensorFlow
python -m jupyter lab notebooks/assignment05      # MNIST, Fashion-MNIST, Diabetes với 4 kiến trúc CNN
```

Mở notebook rồi chọn **Kernel → Restart Kernel and Run All**. Thư viện riêng của từng bài nằm trong
`requirements-assignment03.txt`, `requirements-assignment04.txt` (cài bằng `python -m pip install -r <tệp>` khi cần).

## 4. Assignment 06: dự báo giá bằng RNN (Streamlit)

```powershell
python -m streamlit run deployment/assignment06/app.py --server.port 8501
```

Ứng dụng đọc trọng số đã lưu trong `models/assignment06/`, không huấn luyện lại khi mở. Lần mở đầu mất 20–40 giây để nạp PyTorch và TensorFlow.
Notebook của bài: `python -m jupyter lab notebooks/assignment06`.

## 5. Chương 4: Knowledge Graph và RAG agent

```powershell
python -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8044
```

Mở http://localhost:8044 để hỏi đáp (có sẵn ba câu hỏi mẫu); tài liệu API tại http://localhost:8044/docs.
Lần hỏi RAG đầu tiên có thể mất 30–60 giây vì phải nạp mô hình sinh văn bản chạy trên CPU.
Notebook của chương: `python -m jupyter lab notebooks/chapter04`.

## 6. Tiểu luận: ứng dụng minh họa "Hiểu AI qua dữ liệu tài chính"

```powershell
python -m streamlit run deployment/tieuluan/app.py --server.port 8502
```

Ứng dụng chỉ suy luận bằng trọng số đã lưu, không huấn luyện lại. Tab "Dự báo phiên tới" cần internet để tải giá mới
(chỉ tải khi bấm nút); các tab còn lại chạy offline. Notebook: `python -m jupyter lab notebooks/tieuluan`.

## 7. Chạy nhiều phần cùng lúc (tùy chọn)

```powershell
$py = ".\.venv\Scripts\python.exe"
Start-Process $py "agents/demo_app/server.py","--port","8100"
Start-Process $py "-m","uvicorn","app.main:app","--app-dir","backend","--port","8000"
Start-Process $py "-m","uvicorn","deployment.chapter04.app:app","--port","8044"
Start-Process $py "-m","streamlit","run","deployment/assignment06/app.py","--server.port","8501"
Start-Process $py "-m","streamlit","run","deployment/tieuluan/app.py","--server.port","8502"
Start-Process -WorkingDirectory web powershell "-NoExit","-Command","npm run dev"
```

Tắt tất cả ứng dụng đang giữ các cổng trên:

```powershell
8000,8044,8100,8200,8501,8502,5173 | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force } }
```

## 8. Gặp lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Address already in use` / cổng bị chiếm | Đổi cổng (`--port` hoặc `--server.port`) hoặc dùng lệnh tắt ở mục 7 |
| `ModuleNotFoundError` | Chưa kích hoạt `.venv`: dùng `.\.venv\Scripts\python.exe` thay cho `python` |
| Web báo lỗi gọi API | API ở mục 2 chưa chạy hoặc đang ở cổng khác (đặt `VITE_API_BASE_URL`) |
| Streamlit mở trang trắng một lúc | Đang nạp mô hình; chờ 20–40 giây rồi tải lại |
| Streamlit hỏi email lần đầu | Nhấn Enter để bỏ qua (đã tắt thu thập số liệu trong `.streamlit/config.toml`) |

Giao diện sáng của Streamlit đặt trong `.streamlit/config.toml`; giao diện web nằm ở `web/src/styles.css`, di động ở `mobile/src/theme.ts`.
