# Notebook bổ sung cho tiểu luận bảy chương

Mỗi chương của tiểu luận có đúng **một** notebook trong thư mục này (Chương 1 không có notebook và demo).
Notebook cũ nằm ở `notebooks/chapter04/` (bài Chương 04 của môn học) không bị thay thế hoặc ghi đè.

## Thứ tự theo chương của tiểu luận

| Chương | Notebook | Nội dung |
|---|---|---|
| 1. Lịch sử AI | không có notebook | số liệu và dòng thời gian tính trong demo gộp (tab Chương 1) |
| 2. Kỹ thuật ML cơ bản | `02_ml.ipynb` | dự đoán khoản vay không trả được nợ (Lending Club, 150.000 dòng, chia theo thời gian): hồi quy logistic, rừng ngẫu nhiên, MLP bằng NumPy/PyTorch/Keras |
| 3. CNN | `03_cnn.ipynb` | phần A thực nghiệm CNN trên S&P 500, VN-Index, Bitcoin (ảnh GASF); phần B CNN 4 Conv + 2 Dense bằng Scratch/Keras/PyTorch |
| 4. RNN, LSTM | `04_rnn_lstm.ipynb` | phần A thực nghiệm RNN/LSTM/GRU và backtest; phần B RNN và phần C LSTM 5 lớp hồi quy + Dense, ba công cụ |
| 5. Knowledge Graph và RAG | `05_knowledge_rag.ipynb` | KG, truy vấn nguồn, RRF, truy hồi/sinh và calculator |
| 6. Deep Graph | `06_deep_graph_gcn.ipynb` | GCN trên đồ thị cổ phiếu VN30: dự báo, phân loại nút, over-smoothing |
| 7. Hệ đa Agent | `07_multi_agent.ipynb` | State, Message, bốn Agent, trace và kiểm tra lỗi |

Mỗi notebook gồm Markdown giải thích, định nghĩa hàm, cell chạy và output.
Notebook mô hình chứa định nghĩa ba bản cài đặt, cách chép trọng số, vòng học,
trace kích thước tensor và biểu đồ loss. Mã Scratch sử dụng thư viện NumPy
tự viết ở `src/tieuluan/scratch`, có forward/backward riêng.

## Chạy và giải thích

1. Mở từ dự án `intelligent-system-assignment-01`, chọn kernel Python của `.venv`.
2. Giữ dữ liệu `data/tieuluan/raw/lending_club/` và `data/tieuluan/processed/` (Chương 2–4), corpus `data/chapter04`,
   hai mô hình `models/chapter04` và đồ thị `knowledge_graph/chapter04` có sẵn.
3. Chọn **Restart Kernel and Run All** để chạy độc lập từng notebook.
4. Đọc các ô Markdown cạnh cell để trình bày đầu vào → công thức → hàm → đầu ra.
5. Dùng nhãn cell (ví dụ CNN-03, RNN-05, KG-02, AGENT-04) trong metadata
   để đối chiếu những cell được trích vào báo cáo Word.

Chương 2 huấn luyện đúng cấu hình của kết quả đã lưu (mini-batch 256, tối đa 25 epoch) nên chạy trong 3–5 phút.
Ở Chương 3 và 4, phần A huấn luyện lại bản rút gọn (tối đa 12 epoch) và phần B/C học ba epoch trên 256 mẫu
train, chọn checkpoint bằng 128 mẫu validation, rồi đánh giá trên 256 mẫu test; các tập giữ ranh giới
thời gian đã định trong dự án. Đây là demo trên tập con VN-Index thật; không gán kết quả cho benchmark
toàn bộ tập dữ liệu hoặc lợi nhuận giao dịch. Notebook đã được chạy và lưu sẵn output.

Notebook KG/Agent chạy mô hình cục bộ, không cần API key. Agent kiểm tra id
nguồn không có nghĩa đã xác minh nội dung; xem ví dụ đối chiếu bộ sinh với
công cụ thống kê. Kết quả chạy được lưu ở `results/tieuluan/complete/notebooks/`.

Tạo và chạy lại toàn bộ (notebook Chương 6 có script dựng riêng; số liệu 5 seed của nó sinh bởi `complete_gnn`).
Lệnh `build_complete_notebooks` mặc định chỉ dựng ba notebook Chương 2–4 (`--only 03_cnn` để chọn một; `--legacy-extra`
dựng lại 05 và 07 nếu thật sự cần, vì sẽ ghi đè). Dữ liệu Chương 2 chuẩn bị bằng
`scripts.tieuluan.prepare_loan_data` và huấn luyện bằng `scripts.tieuluan.run_loan`:

```powershell
.venv/Scripts/python.exe -m scripts.tieuluan.build_deepgraph_notebook
.venv/Scripts/python.exe -m scripts.tieuluan.build_complete_notebooks
.venv/Scripts/python.exe -m scripts.tieuluan.build_complete_report
.venv/Scripts/python.exe -m scripts.tieuluan.finalize_complete_report
```

Nếu nhận gói ZIP bổ sung, giải nén theo đúng cấu trúc thư mục vào dự án này.
Gói chỉ kèm notebook, mã và kết quả nhỏ; dữ liệu và trọng số mô hình dùng
những tệp đã có trong dự án, không phải môi trường độc lập được đóng gói sẵn.
