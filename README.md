# Intelligent System Development: bài tập, chương và tiểu luận

Kho mã của môn Phát triển Hệ thống Thông minh (PTIT). Mọi ứng dụng chạy **trên máy cá nhân (localhost)**, giao diện nền sáng.
**Muốn chạy phần nào, xem [RUN.md](RUN.md)** (lệnh, cổng, cách xử lý lỗi).

## Các phần của dự án

| Phần | Nội dung | Báo cáo | Chạy thử |
|---|---|---|---|
| Assignment 01–02 | Ba hệ thống: Diabetes, House Price, E-commerce (FastAPI + React + Expo) | `docs/report_assignment_02/`, `docs/archive/assignment_01/` | RUN.md mục 2 |
| Assignment 03 | Mạng nơ-ron (MLP) trên ba bài toán | `docs/report_assignment_03/` | notebook |
| Assignment 04 | CNN NumPy, PyTorch, TensorFlow | `docs/report_assignment_04/` | notebook |
| Assignment 05 | Bốn kiến trúc CNN trên MNIST, Fashion-MNIST, Diabetes | `docs/report_assignment_05/` | notebook |
| Assignment 06 | RNN dự báo giá AAPL và vàng | `docs/report_assignment_06/` | RUN.md mục 4 |
| Chương 4 | Knowledge Graph, RAG và agent dùng công cụ | `docs/report_chapter_04/` | RUN.md mục 5 |
| Tiểu luận | AI trong đầu tư tài chính (6 chương) | `docs/tieuluan/`, `deliverables/tieuluan_hoan_chinh_6_chuong/` | RUN.md mục 6 |
| Project 01 | Hệ đa agent tiểu đường: Deep Learning, KB + RAG, RL, luật, phòng khám 3D | `project01/`, `docs/project01/` | RUN.md mục 0 |
| Agent 01, Agent 02 | AI agent, RL so với rule-based (mèo đuổi chuột, hai người bán) | `agents/agent01/`, `agents/agent02/` | RUN.md mục 1 |

Báo cáo là file Word (`.docx`) là bản chính; file PDF nộp bài do tác giả tự xuất.

## Cấu trúc thư mục

```text
agents/        Agent 01 và Agent 02: báo cáo, notebook, và demo localhost (agents/demo_app)
backend/       API FastAPI của Assignment 02
web/  mobile/  Ứng dụng React/Vite và Expo của Assignment 02
deployment/    Ứng dụng Streamlit/FastAPI của Assignment 06, Chương 4 và tiểu luận
notebooks/     Notebook theo từng bài (assignment03..06, chapter04, tieuluan, ...)
src/           Mã dùng chung của các bài (assignment03..06, chapter04, tieuluan)
scripts/       Chuẩn bị dữ liệu, huấn luyện, chạy thí nghiệm
data/          Dữ liệu (CSV, npz, corpus)
models/        Mô hình đã huấn luyện (joblib, npz, trọng số)
results/       Kết quả thí nghiệm đã lưu (JSON, CSV)
figures/       Hình đánh giá và hình minh họa
docs/          Báo cáo Word và tài liệu API/kiến trúc
deliverables/  Bản tiểu luận 6 chương và notebook kèm theo
tests/         Kiểm thử tự động
.streamlit/    Giao diện sáng và cấu hình localhost cho Streamlit
RUN.md         Lệnh chạy tất cả các phần
```

## Assignment 01–02: ba hệ thống

1. **Diabetes**: phân loại nhị phân từ 6 chỉ số bệnh nhân; Random Forest Pipeline đã lưu.
2. **Vietnam House Price**: hồi quy từ 11 thuộc tính nhà đất; Random Forest Pipeline đã lưu.
3. **E-commerce**: phân loại mức yêu thích từ nội dung đánh giá và số lượt hữu ích; TF-IDF + Logistic Regression Pipeline đã lưu.

| Hệ thống | Dữ liệu gốc | Biểu diễn số cuối cùng | Mô hình | Kết quả trên tập kiểm tra |
|---|---|---|---|---|
| Diabetes | 6 giá trị CSV | điền khuyết trung vị, chuẩn hóa, `B × 6` | Random Forest Classifier | Accuracy 0,7468; F1 0,6061 |
| House | 11 giá trị hỗn hợp | điền khuyết, chuẩn hóa, one-hot, `B × 83` | Random Forest Regressor | MAE 1,2529; RMSE 1,6018; R² 0,4738 |
| E-commerce | Summary, Text, 2 số lượt hữu ích | TF-IDF `V=12.000` + 5 đặc trưng = `B × 12.005` | Logistic Regression | Accuracy 0,9566; F1 0,9746; ROC-AUC 0,9842 |

Nguồn dữ liệu: [Kaggle Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database),
[Kaggle Vietnam Housing Dataset 2024](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024),
[Kaggle Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews).
Tệp `Reviews.csv` của E-commerce rất lớn (hơn 300 MB) nên không đưa vào Git; tải về và đặt đúng tại `data/ecommerce/Reviews.csv` nếu muốn chạy lại notebook.
API đã nạp sẵn Pipeline đã lưu nên không cần tệp này.

Các endpoint chính của API (Swagger tại http://localhost:8000/docs, chi tiết trong [docs/API.md](docs/API.md)):
`POST /api/v1/diabetes/predict`, `POST /api/v1/house-price/predict`, `POST /api/v1/ecommerce/predict`, `GET /api/v1/models`.

## Chạy kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

## Giới hạn cần nhớ

Diabetes không phải chẩn đoán y khoa; House không phải định giá chuyên nghiệp; E-commerce dự đoán một nhãn suy ra từ điểm đánh giá chứ không đo ý định mua thật.
Bộ tiền xử lý chỉ được fit trên tập huấn luyện, tập kiểm tra không dùng để chỉnh tham số. Ứng dụng tiểu luận và Chương 4 chỉ minh họa kỹ thuật,
không phải khuyến nghị đầu tư hay lời khuyên y tế.
