# Chương 5 và 7: Knowledge Graph, RAG và hệ đa agent (FastAPI, localhost)

Chạy từ thư mục gốc dự án:

```powershell
.venv/Scripts/python.exe -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8044
```

| Địa chỉ | Nội dung |
|---|---|
| http://127.0.0.1:8044/ | Giao diện Chương 5: khung hội thoại, câu mẫu tiếng Việt, trả lời văn xuôi kèm cách xử lý, nguồn, nhãn tin cậy |
| http://127.0.0.1:8044/agents | Giao diện Chương 7: bốn agent (truy hồi, đồ thị, viết, kiểm tra) làm việc nhóm |
| http://127.0.0.1:8044/docs | Swagger UI |
| `GET /api/v1/status` | Trạng thái nạp mô hình ở luồng nền (`loading` / `ready` / `error`) |
| `POST /api/v1/knowledge/ask` | Hỏi đáp; luôn trả 200 với `answer` là câu văn, `route`, `steps`, `sources`, `confidence`, `table`, `trace` |
| `POST /api/v1/agents/run` | Chạy bốn agent; mỗi bước có `summary` tiếng Việt, `stats_check` khi là câu số liệu |

## Dữ liệu và mô hình

- Bộ sinh: `Qwen/Qwen2.5-1.5B-Instruct` ở `models/chapter05/llm` (tải bằng `scripts/chapter05/download_llm.py`, không commit). Nếu thư mục
  này không có, `RAGEngine(generator='auto')` quay về flan-t5-small cũ. Mô hình nhỏ đọc tiếng Anh tốt hơn tiếng Việt nên viết câu trả lời
  tiếng Anh từ các đoạn tiếng Anh rồi dịch sang tiếng Việt bằng bảng thuật ngữ; câu hỏi tiếng Việt cũng được dịch sang truy vấn tiếng Anh trước khi truy hồi.
- Thống kê: CDC BRFSS 2023, mẫu 150.000 người, `data/chapter05/brfss2023_diabetes_150k.csv` (chuẩn bị bằng `scripts/chapter05/prepare_brfss.py`).
- Kho tài liệu: 183 đoạn NIDDK (tiểu đường) và 9.177 đoạn FinQA (tài chính), embedding và đồ thị tri thức ở `results/chapter04` và `knowledge_graph/chapter04`.

## Phạm vi và giới hạn

Agent là bộ định tuyến công cụ theo luật và từ khóa (RAG, thống kê, máy tính, đồ thị), không phải agent LLM tự lập kế hoạch. Câu trả lời do mô hình viết
có trạng thái `generated_unverified`: mã nguồn trích dẫn hợp lệ nhưng nội dung chưa được kiểm chứng và không phải lời khuyên y tế hoặc đầu tư.
Chỉ con số do công cụ thống kê hoặc máy tính tính ra mới chính xác. Khi các đoạn truy hồi không chứa đáp án, hệ thống từ chối (`insufficient_evidence`).
Biến môi trường `CH5_NO_WARMUP=1` tắt việc nạp mô hình ở luồng nền (dùng cho kiểm thử). Xem RUN.md ở thư mục gốc cho mọi lệnh.
