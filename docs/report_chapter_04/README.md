# Chương 4: Knowledge & RAG

Báo cáo cuối: `CHUONG_04_KNOWLEDGE_RAG.docx`. Không xuất PDF.

Đã thực hiện: tải UCI Diabetes 130-US Hospitals, ba tập FinQA và bốn trang NIDDK; tải MiniLM và FLAN-T5-small; xây/lưu/truy vấn 3 KG; chạy 4 notebook với output; so sánh BM25/dense/hybrid; sinh trả lời thật; kiểm tra Agent API bằng HTTP cục bộ. DOCX gồm code thực thi, bảng kết quả, hình và phân tích lỗi.

KG đã chạy mô tả quan hệ dữ liệu và tài liệu. Agent là tool router xác định; API chạy cục bộ và có Swagger UI. Không tuyên bố đã triển khai cloud, ontology y khoa đầy đủ, kiểm định lâm sàng hoặc hiệu quả đầu tư.

Từ thư mục gốc:

```powershell
.venv/Scripts/python.exe -m pip install -r scripts/chapter04/requirements.txt
.venv/Scripts/python.exe scripts/chapter04/prepare_data.py
.venv/Scripts/python.exe scripts/chapter04/download_models.py
.venv/Scripts/python.exe scripts/chapter04/execute_notebooks.py
.venv/Scripts/python.exe scripts/chapter04/build_report.py
.venv/Scripts/python.exe scripts/chapter04/verify_artifacts.py
```

Nguồn soạn: `CHUONG_04_KNOWLEDGE_RAG.md`; bản đã điền số liệu: `CHUONG_04_KNOWLEDGE_RAG_HOAN_CHINH.md`. Dataset tại `data/chapter04/`; kết quả tại `results/chapter04/`; notebook tại `notebooks/chapter04/`; model local tại `models/chapter04/` (khoảng 400 MB). Không cần khóa API. Xem `deployment/chapter04/README.md` để chạy Agent bằng uvicorn.
