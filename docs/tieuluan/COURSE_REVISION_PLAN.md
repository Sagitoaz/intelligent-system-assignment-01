# Kế hoạch sửa tiểu luận theo yêu cầu môn học

> Kế hoạch tập nhỏ này đã được thay thế theo yêu cầu tiếp theo của người dùng: mỗi train trên 100.000 mẫu, đọc CSV, người dùng tự chạy notebook, tạm gác deploy. Xem `COURSE_LARGE_HANDOFF.md` và `notebooks/tieuluan/README.md`. Không dùng kết quả tập nhỏ cho báo cáo mới.

## Mục tiêu và mốc bảo toàn

Bản tài chính đã lưu trên `origin/main` tại `f7bd064`. Không ghi đè dữ liệu, mã hoặc kết quả nằm trong manifest thực nghiệm cũ. Bản mới chỉ trình bày Mở đầu và Chương 1–4; tài liệu tham khảo và hướng dẫn triển khai là phần bổ trợ. Người dùng đã giao quyền chọn dữ liệu, tải dữ liệu, sửa báo cáo, chạy thực nghiệm và commit.

## Thiết kế

- Chương 1: lịch sử AI, ví dụ đa lĩnh vực; giới thiệu dữ liệu bảng, ảnh, chuỗi để nối sang thực nghiệm.
- Chương 2: Breast Cancer Wisconsin Diagnostic và Credit Default; hồi quy logistic, SVM tuyến tính, MLP. Cả ba có bản NumPy tự viết, Keras và PyTorch; SVM dùng hinge loss, không gọi SVM thư viện trong bản tự viết.
- Chương 3: MNIST và Fashion-MNIST, phân loại đủ mười lớp; CNN một khối 4 bộ lọc, một khối 8 bộ lọc, hai khối 4–8 bộ lọc. Giữ ảnh gốc 28×28. Cả ba kiến trúc có đủ ba bản cài đặt.
- Chương 4: nhiệt độ Jena và VN-Index; SimpleRNN, LSTM, GRU, đủ ba bản cài đặt. Dự đoán hướng thay đổi bước kế tiếp, có đối chứng luôn chọn lớp đa số và lặp lại hướng bước trước.
- Từng chương 2–4 tự chứa đường dẫn nguồn, kích cỡ gốc và thực dùng, mẫu có nhãn, phân bố, cách chia, công thức, mã cốt lõi, bảng so sánh, hình học/nhầm lẫn và nhận xét giới hạn.
- Ngân sách ban đầu: ảnh lấy mẫu phân tầng từ tập train chính thức để bản NumPy chạy được trên CPU; giữ riêng validation, dùng toàn bộ test chính thức. Ghi chính xác kích cỡ trong manifest và báo cáo. Một seed chung cho đối chiếu triển khai; không diễn giải chênh lệch nhỏ thành kết luận phổ quát.
- Khởi tạo chung, thứ tự batch chung, ngăn rò rỉ dữ liệu; chọn checkpoint theo validation, không theo test. Số liệu và mã trong Word được trích tự động từ artifact/mã chạy thật.
- Ứng dụng bổ sung minh họa ba nhóm bài toán; giữ tính năng tài chính đã hoạt động. Render chỉ nạp NumPy khi suy luận.

## Đường dẫn mới

`src/tieuluan/course/`: dữ liệu, mô hình, thực nghiệm độc lập; tái sử dụng các lớp scratch cũ bằng import, không sửa chúng.

`scripts/tieuluan/prepare_course_data.py`, `run_course_experiments.py`, `build_course_report.py`: tải dữ liệu, huấn luyện và sinh báo cáo.

`data/tieuluan/course/`, `models/tieuluan/course/`, `results/tieuluan/course/`, `figures/tieuluan/course/`: tách khỏi thực nghiệm cũ.

## Các bước và nghiệm thu

- [x] Xác nhận bản cũ đã commit và push.
- [ ] Lưu kế hoạch và mốc Git bản cũ.
- [ ] Chuẩn bị dữ liệu, nguồn, checksum, định danh mẫu; kiểm tra chia tập và chuẩn hóa.
- [ ] Kiểm tra đạo hàm loss mới, đồng nhất logits ba framework trước khi huấn luyện.
- [ ] Chạy đủ 3 mô hình × 2 bộ dữ liệu × 3 framework cho mỗi chương; lưu dự báo, lịch sử và trọng số.
- [ ] Bổ sung giao diện minh họa phù hợp và kiểm thử không mạng.
- [ ] Viết lại báo cáo có mã, hình, bảng thực nghiệm; thêm nguồn tham khảo.
- [ ] Sinh Word, kiểm tra số trang (Mở đầu 3–4, Chương 1: 8, Chương 2–4: 10–15 mỗi chương), mã và chú thích không tràn trang.
- [ ] Kiểm tra artifact mới; audit xác nhận thực nghiệm cũ không thay đổi.
- [ ] Commit đúng phạm vi, push và báo rõ kết quả cùng giới hạn.
