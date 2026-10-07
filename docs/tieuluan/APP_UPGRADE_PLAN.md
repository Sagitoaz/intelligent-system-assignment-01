# Kế hoạch nâng cấp ứng dụng minh họa

Đặc tả: yêu cầu A–E, kiểm thử và báo cáo do người dùng cung cấp ngày 06/10/2026.
Thực hiện trực tiếp trong repo theo yêu cầu; không chạm các tệp trong manifest thực nghiệm.

- [x] Viết kiểm thử đối chiếu đặc trưng thị trường, tín dụng, suy luận và tải mạng giả lập.
- [x] Sinh tài nguyên nhỏ bằng `prepare_app_assets.py`: dòng gốc, tham số, ngưỡng validation, mẫu và thống kê.
- [x] Tạo `live.py`, `credit.py`: chỉ NumPy/pandas/requests/thư viện chuẩn; nạp mô hình tự viết.
- [x] Tổ chức bảy tab, cache đọc tệp/mô hình/mạng; tiếng Việt và định dạng số thống nhất.
- [x] Cập nhật Docker, gói Space; chạy pytest và AppTest cả gói.
- [x] Commit phạm vi nhiệm vụ, thử git archive với venv tối giản, Playwright, mạng thật và đo RAM.
- [x] Chụp hình, sửa Phụ lục B/Mở đầu/tài liệu tham khảo; build Word, kiểm tra trang và tham chiếu.
- [x] Kiểm toán manifest, rà soát diff và chuẩn bị các commit thuộc nhiệm vụ; tích hợp bằng push origin main theo yêu cầu (đối chiếu lịch sử Git).

Điểm kiểm soát: không dùng nhãn test làm đầu vào; không đọc Excel trên máy chủ; chỉ dùng phiên đã đóng;
nguồn dự phòng phải ghi tên thật; thiếu dữ liệu phải báo rõ; khoảng tin cậy lấy từ phân tích PyTorch đã lưu.
Mô hình cache dùng khóa khi suy luận vì lớp NumPy giữ trạng thái trung gian.
Địa chỉ Render do người dùng cung cấp ngày 07/10/2026: https://financial-investment-ai.onrender.com/. Đã bổ sung vào Phụ lục B và README. Ba mức tín dụng dùng quy ước minh họa đã thông báo: dưới nửa ngưỡng, từ nửa ngưỡng đến dưới ngưỡng, từ ngưỡng trở lên.

Bằng chứng: 41 kiểm thử ứng dụng/mô hình đạt; thêm kiểm thử danh mục nguồn Phụ lục đạt. Kiểm toán nạp lại 52 mô hình, sai số bằng 0, files_changed_since_run rỗng. Bản archive 808d69d chạy với requirements tối giản; Edge hoàn tất 37 bước và thử nhập tiền hợp lệ/sai. Đỉnh RAM ghi tại data/tieuluan/app/verification.json.

Rà soát độc lập không tìm thấy lỗi chặn; đề nghị bổ sung kiểm thử luồng dự báo trực tiếp thành công đã thực hiện. Lỗi kiểm tra trình duyệt chờ quá ngắn đã sửa bằng chờ phần tử kết quả. Bộ dựng Word được sửa để nguồn chỉ trích dẫn trong Phụ lục có trong danh mục tham khảo.

Word đã cập nhật mục lục: Mở đầu 4 trang; Chương 1: 8; Chương 2: 12; Chương 3: 11; Chương 4: 11; Phụ lục: 6. Đã soát PDF tạm: không còn chỗ giữ chỗ, mọi hình/bảng đánh số liên tục và được nhắc trong văn bản.

Nghiệm thu cuối: 42 kiểm thử đạt. Gói Space cuối mở đủ bảy thẻ. Đã dừng hai máy chủ thử nghiệm và xóa thư mục tạm chứa archive, venv, PDF; Word cuối giữ tại đường dẫn báo cáo, không đưa vào Git.

Rà soát bổ sung ngày 07/10/2026: đã thử trực tiếp URL Render bằng Edge, hoàn tất 37 bước gồm bảy thẻ, mô hình lịch sử, hồ sơ tín dụng và tải mạng thật. Yahoo hoạt động cho S&P 500/Bitcoin; VN-Index dùng nguồn dự phòng DNSE. Word đã dựng lại, liên kết Render có trong hyperlink và PDF; số trang giữ nguyên. Kiểm tra toàn tài liệu: 22 hình, 25 bảng đều liên tục và có câu nhắc; 126 mục tham khảo; không còn ký hiệu chưa thay hoặc chữ tràn trang. Kiểm toán lặp lại: 128 bản ghi, 52 trọng số, sai số nạp lại bằng 0, không tệp thực nghiệm nào đổi. Chỉ còn hai ô NhomLop/NhomTL trên bìa để người dùng điền trước khi xuất PDF.
