# Bốn notebook Chương 4: Knowledge & RAG

1. `01_kb_kg.ipynb`: thống kê toàn bộ UCI/FinQA; KG bệnh án mẫu 5.000 lượt; KG tài liệu tiểu đường và tài chính; lưu/nạp và query; biểu đồ.
2. `02_diabetes_rag.ipynb`: 16 câu hỏi neo nguồn NIDDK; BM25/dense/hybrid; FLAN-T5-small sinh thật; không context/RAG/gold context; EM/F1 và đầu ra sai.
3. `03_finance_rag.ipynb`: 8 cấu hình trên 883 FinQA dev; 24 câu sinh cố định; chấm số có đơn vị phần trăm; audit qa.answer/exe_ans; calculator từ dòng bảng thực tế.
4. `04_agent_deployment.ipynb`: tool traces; process uvicorn thật; health, Swagger UI, POST RAG/thống kê/graph/calculate; lỗi đầu vào; response và server log.

Các notebook đã thực thi và giữ output. Mở trực tiếp trong Jupyter hoặc VS Code. Để chạy lại từ kernel sạch:

```powershell
.venv/Scripts/python.exe scripts/chapter04/execute_notebooks.py
```

Script dùng đúng interpreter đang chạy để khởi động kernel; không phụ thuộc kernel Python toàn hệ thống. Nếu chạy thủ công, chọn interpreter `.venv/Scripts/python.exe`. Dataset, model và artifact đã tải trong dự án. Khi di chuyển sang máy khác, cài `scripts/chapter04/requirements.txt`, chạy `prepare_data.py` và `download_models.py` trước.

Corpus/model và câu hỏi benchmark tiếng Anh; phần giải thích/report tiếng Việt. Tập 16 câu tiểu đường do người viết xây để minh họa, không phải thẩm định y khoa độc lập. Report-filtered FinQA dùng nguồn oracle. Agent là định tuyến công cụ có giới hạn, chưa là LLM planner. Chỉ kiểm chứng triển khai HTTP cục bộ.
