# Chạy tiểu luận bằng notebook

Ba notebook chính đã đổi sang thực nghiệm lớn và **chưa chạy huấn luyện đầy đủ**:

1. `02_ml.ipynb`: Covertype, MiniBooNE; logistic, SVM tuyến tính, MLP.
2. `03_cnn.ipynb`: EMNIST Digits, Kuzushiji-49; CNN 4 bộ lọc, CNN 8 bộ lọc, CNN hai khối.
3. `04_rnn.ipynb`: Jena và điện năng; SimpleRNN, LSTM, GRU.

Mỗi notebook có hai bộ, mỗi bộ có hơn 100.000 mẫu train sau làm sạch; cả ba mô hình đều có NumPy tự viết, Keras, PyTorch. Dữ liệu đọc từ `data/tieuluan/course_large/csv/`. Bản tài chính cũ được lưu tại tag `tieuluan-finance-v1-20261008`.

## Thao tác trong VS Code

1. Chọn kernel Python ở `.venv\Scripts\python.exe`.
2. Mở notebook, xem các ô cấu hình và nguồn dữ liệu. Giữ mặc định để chạy đầy đủ; lưu Ctrl+S.
3. Restart Kernel rồi Run All. Ô huấn luyện ở mục 6 sẽ chạy lâu. Các mục trước tải/đọc CSV, xem phân bố và kiểm tra đầu ra ba cách cài đặt.
4. Sau mỗi epoch có checkpoint. Nếu dừng kernel, chạy lại từ đầu; các bước học đã lưu được tiếp tục và các lượt hoàn thành được bỏ qua.
5. Sau khi chạy xong, lưu notebook để giữ output. Mục 7 sinh CSV kết quả, đường học, ma trận nhầm lẫn và Markdown có mã trích từ notebook.

Mã được đặt trực tiếp trong notebook, gồm cả lan truyền ngược NumPy và vòng huấn luyện, không chỉ gọi script bên ngoài. Các file Python giúp kiểm thử và sinh bản ban đầu; không chạy lại `build_large_notebooks` sau khi tự sửa notebook, trừ khi đã lưu bản sao. Công cụ mặc định từ chối ghi đè notebook đã sửa hoặc có output.

## Chạy từng phần và tiếp tục

Có thể chọn một phần `MODEL_KINDS` hoặc `FRAMEWORKS` trong ô cấu hình để chạy theo đợt. Giữ `RUN_ID`, cấu hình học và mã thuật toán không đổi để tiếp tục. Khi đổi kiến trúc, cách xử lý dữ liệu, seed hoặc siêu tham số, hãy dùng `RUN_ID` mới. Không trộn kết quả khác cấu hình vào một bảng.

Các checkpoint lưu cả Adam, bộ sinh ngẫu nhiên, vòng học và mô hình tốt nhất. Nếu ngắt giữa epoch, phần epoch đang chạy sẽ làm lại. Chỉ dùng checkpoint cục bộ do notebook tạo; không mở file pickle nhận từ người khác.

Thời gian chạy phụ thuộc CPU, số vòng thực chạy, kích cỡ mạng và dữ liệu. Bộ dữ liệu lớn không bảo đảm mô hình tốt, và chạy lâu không phải chỉ tiêu chất lượng. Chỉ số có ý nghĩa khi so với đường cơ sở và đọc lỗi trên test.

## Sau khi huấn luyện

- Kết quả: `results/tieuluan/course_large/<RUN_ID>/`.
- Trọng số: `models/tieuluan/course_large/<RUN_ID>/`.
- Hình: `figures/tieuluan/course_large/<RUN_ID>/`.
- Bảng và mã phục vụ báo cáo: `docs/tieuluan/course_large/generated/<RUN_ID>/`.

Phần xuất báo cáo kiểm tra đủ 18 lượt mỗi chương (54 lượt cho cả ba), trên 100.000 mẫu train mỗi bộ, và mã hiện tại khớp mã đã chạy. Nếu thiếu kết quả, nó báo rõ và dừng. Phần Word hoàn chỉnh, nhận xét và kiểm tra số trang cần làm sau khi có kết quả thật; bản Word cũ không được coi là báo cáo của thực nghiệm mới. Deploy tạm gác theo yêu cầu.
