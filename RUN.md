# Cách chạy các phần của dự án (localhost)

Tất cả ứng dụng chỉ chạy **trên máy cá nhân** (`localhost`), giao diện nền sáng. Không cần tài khoản đám mây hay API key.
Mở **PowerShell tại thư mục gốc dự án** rồi chạy lệnh của phần cần xem. Dừng ứng dụng bằng `Ctrl + C`.

| Phần | Địa chỉ mở trên trình duyệt | Cổng |
|---|---|---|
| Agent 01 & Agent 02 (demo mèo đuổi chuột, hai người bán) | http://localhost:8100 | 8100 |
| **Tiểu luận: demo gộp 6 chương, Chương 2–7 (sidebar bên trái)** | http://localhost:8500 | 8500 (tự mở thêm 8502, 8044, 8300) |
| Project 01: Phòng khám AI 3D (hệ đa agent tiểu đường) | http://localhost:8200 | 8200 |
| Tiểu luận Chương 6: Deep Graph (GCN trên đồ thị VN30) | http://localhost:8300 | 8300 |
| Assignment 01–02: API (FastAPI) | http://localhost:8000/docs | 8000 |
| Assignment 01–02: Web (React) | http://localhost:5173 | 5173 |
| Assignment 06: Dự báo RNN (Streamlit) | http://localhost:8501 | 8501 |
| Chương 5 (KG, RAG) và Chương 7 (hệ đa agent): `/` và `/agents` | http://localhost:8044 | 8044 |
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

## 0a. Tiểu luận: demo gộp sáu chương, Chương 2–7 (một lệnh, sidebar bên trái)

```powershell
python tieuluan_demo/server.py --open            # mở http://localhost:8500
```

Máy chủ tự khởi động các demo con trên 127.0.0.1 (Streamlit 8502 cho Chương 2–4, FastAPI 8044 cho Chương 5 và 7, Deep Graph 8300 cho Chương 6). Chương 1 (lịch sử AI) không có demo.
Sidebar bên trái liệt kê sáu chương (chấm xanh lá là demo đã sẵn sàng, cam là đang khởi động); bấm một chương để xem phần "Demo này làm gì?" và "Nên thử". Nút mũi tên trên cùng thu gọn sidebar thành thanh hẹp (nhớ lựa chọn), nút "Đối chiếu báo cáo" ở góc dưới bên phải mở cột mục, hình, bảng của báo cáo (mặc định ẩn). Mở thẳng một chương bằng `http://localhost:8500/#c6`.
Mỗi chương một notebook, theo thứ tự trong `notebooks/tieuluan/complete/`: `02_ml.ipynb`, `03_cnn.ipynb`, `04_rnn_lstm.ipynb`, `05_knowledge_rag.ipynb`, `06_deep_graph_gcn.ipynb`, `07_multi_agent.ipynb`.
 Log của từng demo con nằm ở thư mục tạm (`tieuluan_demo_<tên>.log`).
Lần đầu các tab cần vài chục giây; lần hỏi RAG đầu tiên (Chương 5, 7) cần chờ mô hình ngôn ngữ nạp xong (khoảng 1–2 phút kể từ khi máy chủ 8044 khởi động) rồi mất thêm 15–60 giây trên CPU.
Nếu đã chạy riêng một demo con thì thêm `--no-children`.

## 0. Project 01: Phòng khám AI 3D (hệ đa agent cho bệnh tiểu đường)

```powershell
python project01/server.py --open                # mở http://localhost:8200
python project01/server.py --open --port 8210    # đổi cổng nếu bị trùng
python -m pytest tests/test_project01*.py -q     # kiểm thử tự động (64 kiểm thử)
python -m jupyter lab notebooks/project01        # notebook thực hành (đã chạy sẵn, có giải thích từng ô)
# Huấn luyện/đánh giá lại (tùy chọn, kết quả lưu ở project01/artifacts/)
python project01/train/train_dl.py ; python project01/train/build_kg.py ; python project01/train/train_rl.py ; python project01/train/eval_rag.py ; python project01/train/eval_trust.py
```

Thêm `?mock=1` vào địa chỉ để xem dữ liệu mẫu khi chưa chạy máy chủ API.

Sáu tab (phím 1–6): **Khám bệnh** (có bác sĩ phản biện tranh luận tối đa 2 vòng, biểu đồ yếu tố tác động, what-if kéo thả, tải PDF; nút 🎤 nhập hồ sơ bằng giọng nói), **Sống 24 giờ**, **Bản đồ tri thức**, **Hỏi bác sĩ** (RAG có trích dẫn trên kho 492 đoạn), **Độ tin cậy** (độ chính xác, hiệu chuẩn, công bằng theo tuổi/giới/thu nhập), **Hàng đợi** (phân loại và xu hướng 4 tuần, dữ liệu mẫu).
Ghi chú: giọng nói (TTS) và nhập liệu giọng nói (STT) dùng khóa Azure trong `.env` (`API_MODEL_TTS`, `TTS_ENDPOINT`), không có khóa thì vẫn chạy bằng chữ. Câu trả lời của Hỏi bác sĩ mặc định là trích đoạn; Qwen2.5-1.5B (ô "Cho Qwen viết lại") chậm 40–140 giây và chỉ chạy một tiến trình, đừng bật cùng lúc với demo khác dùng Qwen. Gói PDF cần `reportlab` (có trong `requirements.txt`).

Giọng nói (tùy chọn): đặt `API_MODEL_TTS` (khóa Azure Speech) và `TTS_ENDPOINT` (địa chỉ khu vực, ví dụ `https://southeastasia.api.cognitive.microsoft.com/`) trong tệp `.env` ở thư mục gốc (tệp này đã được git bỏ qua). Có khóa thì phòng khám vừa chạy phụ đề vừa đọc; không có khóa thì chỉ có chữ. Phím tắt: Dấu cách tạm dừng, V bật/tắt tiếng.

## 0b. Tiểu luận Chương 6: demo Deep Graph (GCN trên đồ thị cổ phiếu VN30)

```powershell
python deepgraph_demo/server.py --open           # mở http://localhost:8300
python -m scripts.tieuluan.complete_gnn          # chạy lại thực nghiệm (khoảng 4 phút) -> data/tieuluan/gnn/results.json
```

Trang có ba màn hình: đồ thị tương quan, lan truyền thông điệp qua các lớp GCN, và trò chơi đoán ngành bằng GCN so với MLP.

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

## 5. Chương 5 và 7: Knowledge Graph, RAG và hệ đa agent (cùng một ứng dụng FastAPI)

Lần đầu, tải mô hình sinh câu trả lời và dữ liệu thống kê (một lần, cần internet; mô hình nặng khoảng 3 GB nên không đưa lên Git):

```powershell
python scripts/chapter05/download_llm.py      # Qwen/Qwen2.5-1.5B-Instruct -> models/chapter05/llm
python scripts/chapter05/prepare_brfss.py     # CDC BRFSS 2023 -> data/chapter05/brfss2023_diabetes_150k.csv (đã có sẵn trong repo)
```

Chạy ứng dụng:

```powershell
python -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8044
```

- http://localhost:8044 là giao diện **Chương 5** (khung hội thoại, câu mẫu tiếng Việt; trả lời bằng văn xuôi kèm "cách agent xử lý", thẻ nguồn và nhãn tin cậy; JSON chỉ nằm trong mục "Chi tiết kỹ thuật").
- http://localhost:8044/agents là giao diện **Chương 7** (nhóm agent điều phối, thống kê, truy hồi, tính toán, đồ thị, viết và phản biện). Mỗi lượt chạy hiện đường thời gian từng agent, cuộc trao đổi viết - phản biện và hai cột so sánh "một RAG đơn" với "nhóm agent"; mất khoảng 40–90 giây trên CPU. API: `POST /api/v1/agents/run` (chạy một mạch), `POST /api/v1/agents/start` + `GET /api/v1/agents/job/{id}` (theo dõi tiến độ).
- Tài liệu API (Swagger): http://localhost:8044/docs. Trạng thái nạp mô hình: `/api/v1/status`.
- Khi máy chủ khởi động, mô hình ngôn ngữ và dữ liệu được nạp ở luồng nền (khoảng 1–2 phút); giao diện hiện thanh trạng thái "đang nạp mô hình". Một câu hỏi RAG sau đó mất khoảng 15–60 giây trên CPU; câu thống kê, tính toán, đồ thị trả lời dưới 1 giây.
- Câu hỏi tiếng Việt được dịch sang truy vấn tiếng Anh (kho tài liệu bằng tiếng Anh); giao diện hiển thị truy vấn đã dùng. Thống kê dùng CDC BRFSS 2023 (150.000 người), tính bằng pandas.
- Muốn tắt nạp nền (ví dụ khi chạy kiểm thử): đặt biến môi trường `CH5_NO_WARMUP=1`.

Notebook: `notebooks/tieuluan/complete/05_knowledge_rag.ipynb` (Chương 5) và `07_multi_agent.ipynb` (Chương 7); dựng và chạy lại bằng
`python scripts/chapter05/build_notebooks.py` (Chương 5) và `python scripts/chapter05/build_agent_notebook.py` (Chương 7); dừng máy chủ 8044 trước để tiết kiệm RAM.
Đánh giá định lượng Chương 7 (một RAG đơn so với nhóm agent, 21 câu phức hợp + 5 câu đơn giản + 4 câu ngoài phạm vi; cần máy chủ 8044 đang chạy, mất khoảng 1–2 giờ): `python scripts/chapter05/eval_multiagent.py --resume`, kết quả ở `results/chapter05/multiagent_eval.json`.
Nếu máy thiếu bộ nhớ ảo khiến mô hình float32 không nạp được (tiến trình sập, `access violation`), mở máy chủ bằng `python scripts/chapter05/lowmem_llm.py --serve 8044` (nạp lượng tử hóa int8, đỉnh khoảng 2,5 GB). Notebook bài tập cũ của Chương 4: `python -m jupyter lab notebooks/chapter04`.

## 6. Tiểu luận: ứng dụng minh họa "Hiểu AI qua dữ liệu tài chính"

```powershell
python -m streamlit run deployment/tieuluan/app.py --server.port 8502
```

Ứng dụng chỉ suy luận bằng trọng số đã lưu, không huấn luyện lại. Mỗi chương mở bằng `?chapter=2|3|4` (không có tham số thì
chọn chương bằng nút radio; thêm `&embed=true` để ẩn thanh công cụ). Chương 2 chấm điểm một hồ sơ vay (Lending Club, 150.000 khoản vay);
Chương 3 và 4 thử trên một ngày đã biết kết quả, dự báo phiên sắp tới và (Chương 4) backtest 100 triệu đồng.
Tab "Dự báo phiên sắp tới" cần internet để tải giá mới (chỉ tải khi bấm nút); các tab còn lại chạy offline.
Notebook mỗi chương một tệp: `python -m jupyter lab notebooks/tieuluan/complete`.

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
8000,8044,8100,8200,8300,8500,8501,8502,5173 | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force } }
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
