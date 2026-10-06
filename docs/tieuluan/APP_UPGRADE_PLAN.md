# Kế hoạch nâng cấp ứng dụng minh họa

Đặc tả: yêu cầu A–E, kiểm thử và báo cáo do người dùng cung cấp ngày 06/10/2026.
Thực hiện trực tiếp trong repo theo yêu cầu; không chạm các tệp trong manifest thực nghiệm.

- [ ] Viết kiểm thử đối chiếu đặc trưng thị trường, tín dụng, suy luận và tải mạng giả lập.
- [ ] Sinh tài nguyên nhỏ bằng `prepare_app_assets.py`: dòng gốc, tham số, ngưỡng validation, mẫu và thống kê.
- [ ] Tạo `live.py`, `credit.py`: chỉ NumPy/pandas/requests/thư viện chuẩn; nạp mô hình tự viết.
- [ ] Tổ chức bảy tab, cache đọc tệp/mô hình/mạng; tiếng Việt và định dạng số thống nhất.
- [ ] Cập nhật Docker, gói Space; chạy pytest và AppTest cả gói.
- [ ] Commit phạm vi nhiệm vụ, thử git archive với venv tối giản, Playwright, mạng thật và đo RAM.
- [ ] Chụp hình, sửa Phụ lục B/Mở đầu/tài liệu tham khảo; build Word, kiểm tra trang và tham chiếu.
- [ ] Kiểm toán manifest, rà soát diff, commit và push origin main.

Điểm kiểm soát: không dùng nhãn test làm đầu vào; không đọc Excel trên máy chủ; chỉ dùng phiên đã đóng;
nguồn dự phòng phải ghi tên thật; thiếu dữ liệu phải báo rõ; khoảng tin cậy lấy từ phân tích PyTorch đã lưu.
Mô hình cache dùng khóa khi suy luận vì lớp NumPy giữ trạng thái trung gian.
Địa chỉ Render và cách chia ba mức tín dụng đang được hỏi người dùng.
