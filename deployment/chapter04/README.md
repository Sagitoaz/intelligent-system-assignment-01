# Chương 5 và 7: Knowledge Graph, RAG và hệ đa agent (FastAPI, localhost)

Chạy từ thư mục gốc dự án:

```powershell
.venv/Scripts/python.exe -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8044
```

| Địa chỉ | Nội dung |
|---|---|
| http://127.0.0.1:8044/ | Giao diện Chương 5: khung hội thoại, câu mẫu tiếng Việt, trả lời văn xuôi kèm cách xử lý, nguồn, nhãn tin cậy |
| http://127.0.0.1:8044/agents | Giao diện Chương 7: nhóm agent (điều phối, thống kê, truy hồi, tính toán, đồ thị, viết, phản biện) kèm so sánh với một RAG đơn |
| http://127.0.0.1:8044/docs | Swagger UI |
| `GET /api/v1/status` | Trạng thái nạp mô hình ở luồng nền (`loading` / `ready` / `error`) |
| `POST /api/v1/knowledge/ask` | Hỏi đáp; luôn trả 200 với `answer` là câu văn, `route`, `steps`, `sources`, `confidence`, `table`, `trace` |
| `POST /api/v1/agents/run` | Chạy nhóm agent (và một RAG đơn để so sánh nếu `baseline` = true). Trả `plan`, `steps` (mỗi bước có `summary` tiếng Việt), `facts`, `derived`, `rounds` (bản nháp và lời phản biện từng vòng), `dialogue`, `critic`, `baseline`, `messages`, `status`, `answer`; luôn trả 200, lỗi nằm trong `status = error` |
| `POST /api/v1/agents/start`, `GET /api/v1/agents/job/{id}?since=n` | Chạy nền và lấy tiến độ từng agent (giao diện dùng cặp này) |

## Dữ liệu và mô hình

- Bộ sinh: `Qwen/Qwen2.5-1.5B-Instruct` ở `models/chapter05/llm` (tải bằng `scripts/chapter05/download_llm.py`, không commit). Nếu thư mục
  này không có, `RAGEngine(generator='auto')` quay về flan-t5-small cũ. Mô hình nhỏ đọc tiếng Anh tốt hơn tiếng Việt nên viết câu trả lời
  tiếng Anh từ các đoạn tiếng Anh rồi dịch sang tiếng Việt bằng bảng thuật ngữ; câu hỏi tiếng Việt cũng được dịch sang truy vấn tiếng Anh trước khi truy hồi.
- Thống kê: CDC BRFSS 2023, mẫu 150.000 người, `data/chapter05/brfss2023_diabetes_150k.csv` (chuẩn bị bằng `scripts/chapter05/prepare_brfss.py`).
- Kho tài liệu: 492 đoạn từ 28 trang y tế công khai của Mỹ (NIDDK 20 trang, CDC 7 trang, MedlinePlus 1 trang; thông tin nhà nước Mỹ, ghi nguồn, URL, ngày truy cập và giấy phép trong `data/chapter04/diabetes_corpus_manifest.json`) và 9.177 đoạn FinQA (tài chính). Embedding và đồ thị tri thức ở `results/chapter04` và `knowledge_graph/chapter04`.
- Dựng lại kho tiểu đường: `python scripts/chapter04/build_diabetes_corpus.py [--refresh]` (tải HTML vào `data/chapter04/raw_pages/`, tách đoạn bằng `src/chapter04/corpus.py`: giữ danh sách gạch đầu dòng cùng câu dẫn, tiêu đề mục nằm trong đoạn, 55-150 từ, id `trang:số`). Rồi `python scripts/chapter05/build_diabetes_qa.py` (bộ câu hỏi đánh giá có nhãn) và `python scripts/chapter05/evaluate_diabetes_retrieval.py` (hit@k, MRR, làm mới embedding và đồ thị). Project 01 dùng bản đóng băng 183 đoạn ở `project01/artifacts/diabetes_corpus_niddk183.jsonl`, không bị ảnh hưởng.
- Câu bị từ chối kèm `why_refused`, `suggestions`, `switch_domain` (hàm `explain_refusal` trong `src/chapter04/suggest.py`).

## Phạm vi và giới hạn

Agent là bộ định tuyến công cụ theo luật và từ khóa (RAG, thống kê, máy tính, đồ thị), không phải agent LLM tự lập kế hoạch. Câu trả lời do mô hình viết
có trạng thái `generated_unverified`: mã nguồn trích dẫn hợp lệ nhưng nội dung chưa được kiểm chứng và không phải lời khuyên y tế hoặc đầu tư.
Chỉ con số do công cụ thống kê hoặc máy tính tính ra mới chính xác. Khi các đoạn truy hồi không chứa đáp án, hệ thống từ chối (`insufficient_evidence`).
Chương 7 là nhóm agent có kiểm tra độc lập: agent thống kê (pandas trên BRFSS) và agent tính toán (`safe_calculate`) tạo sự kiện, agent viết (Qwen) soạn nháp, agent phản biện (bộ kiểm tra chạy được, không phải mô hình) đối chiếu từng con số với sự kiện và đoạn tài liệu rồi yêu cầu sửa, tối đa 2 vòng. Agent phản biện chỉ kiểm con số, không kiểm ngữ nghĩa. Mã ở `scripts/tieuluan/complete_agents.py`, dịch vụ ở `deployment/chapter04/agents_service.py`, đánh giá ở `scripts/chapter05/eval_multiagent.py`.
Biến môi trường `CH5_NO_WARMUP=1` tắt việc nạp mô hình ở luồng nền (dùng cho kiểm thử). Xem RUN.md ở thư mục gốc cho mọi lệnh.
