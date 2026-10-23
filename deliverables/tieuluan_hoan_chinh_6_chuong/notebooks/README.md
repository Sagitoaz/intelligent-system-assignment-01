# Notebook bổ sung cho tiểu luận sáu chương

Các notebook này bổ sung cho notebook thực nghiệm cũ trong `notebooks/tieuluan/`
và `notebooks/chapter04/`; không thay thế hoặc ghi đè chúng.

| Notebook | Nội dung |
|---|---|
| `03_cnn_6_layers.ipynb` | CNN 4 Conv + 2 Dense, Scratch/Keras/PyTorch |
| `04a_rnn_6_layers.ipynb` | RNN 5 lớp hồi quy + Dense, ba công cụ |
| `04b_lstm_6_layers.ipynb` | LSTM 5 lớp hồi quy + Dense, ba công cụ |
| `05_knowledge_rag.ipynb` | KG, truy vấn nguồn, RRF, truy hồi/sinh và calculator |
| `06_multi_agent.ipynb` | State, Message, bốn Agent, trace và kiểm tra lỗi |

Mỗi notebook gồm Markdown giải thích, định nghĩa hàm, cell chạy và output.
Notebook mô hình chứa định nghĩa ba bản cài đặt, cách chép trọng số, vòng học,
trace kích thước tensor và biểu đồ loss. Mã Scratch sử dụng thư viện NumPy
tự viết ở `src/tieuluan/scratch`, có forward/backward riêng.

## Chạy và giải thích

1. Mở từ dự án `intelligent-system-assignment-01`, chọn kernel Python của `.venv`.
2. Giữ dữ liệu `data/tieuluan/processed/vnindex.npz`, corpus `data/chapter04`,
   hai mô hình `models/chapter04` và đồ thị `knowledge_graph/chapter04` có sẵn.
3. Chọn **Restart Kernel and Run All** để chạy độc lập từng notebook.
4. Đọc các ô Markdown cạnh cell để trình bày đầu vào → công thức → hàm → đầu ra.
5. Dùng nhãn cell (ví dụ CNN-03, RNN-05, KG-02, AGENT-04) trong metadata
   để đối chiếu những cell được trích vào báo cáo Word.

Ba notebook mô hình học ba epoch trên 256 mẫu train, chọn checkpoint bằng
128 mẫu validation, rồi đánh giá trên 256 mẫu test. Các tập giữ ranh giới
thời gian đã định trong dự án. Đây là demo trên tập con VN-Index thật;
không gán kết quả cho benchmark toàn bộ tập dữ liệu hoặc lợi nhuận giao dịch.

Notebook KG/Agent chạy mô hình cục bộ, không cần API key. Agent kiểm tra id
nguồn không có nghĩa đã xác minh nội dung; xem ví dụ đối chiếu bộ sinh với
công cụ thống kê. Kết quả chạy được lưu ở `results/tieuluan/complete/notebooks/`.

Tạo và chạy lại toàn bộ:

```powershell
.venv/Scripts/python.exe -m scripts.tieuluan.build_complete_notebooks
.venv/Scripts/python.exe -m scripts.tieuluan.build_complete_report
.venv/Scripts/python.exe -m scripts.tieuluan.finalize_complete_report
```

Nếu nhận gói ZIP bổ sung, giải nén theo đúng cấu trúc thư mục vào dự án này.
Gói chỉ kèm notebook, mã và kết quả nhỏ; dữ liệu và trọng số mô hình dùng
những tệp đã có trong dự án, không phải môi trường độc lập được đóng gói sẵn.
