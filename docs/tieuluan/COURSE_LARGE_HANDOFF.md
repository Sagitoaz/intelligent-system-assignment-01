# Bàn giao thực nghiệm lớn: trạng thái trước huấn luyện

## Yêu cầu đang áp dụng

Mở đầu đến Chương 4, mỗi Chương 2–4 có hai bộ dữ liệu và ba mô hình, đủ mã NumPy/Keras/PyTorch. **Mỗi tập train phải trên 100.000 mẫu thực tế. Người dùng tự chạy notebook. Dữ liệu làm việc phải là CSV. Deploy tính sau.**

Kế hoạch tập nhỏ tại `COURSE_REVISION_PLAN.md` đã bị thay thế. Những kết quả thử ở `results/tieuluan/course/` không dùng trong báo cáo mới. Bản tài chính hoàn chỉnh vẫn ở tag `tieuluan-finance-v1-20261008`.

## Đã chuẩn bị

- Sáu bộ nguồn công khai, có URL và SHA-256. Đã chuyển thành CSV chia train/validation/test.
- Notebook có mã thật, EDA, nguồn/kích cỡ, mô hình ba cách cài đặt, kiểm tra logits, checkpoint, đánh giá và xuất hình/bảng/mã.
- Tách toàn bộ đường dẫn kết quả mới thành `course_large/`, không ghi đè thực nghiệm `full/`.
- Kiểm thử nhỏ xác minh số mẫu, chia thời gian, chuyển đổi CSV, mô hình 49 lớp và khôi phục Adam; không coi kiểm thử là huấn luyện đầy đủ.

## Còn chờ người dùng chạy

Chưa có điểm số test của thực nghiệm lớn, chưa có nhận xét thực nghiệm cuối cùng, chưa sinh lại Word theo dữ liệu mới. Các biểu đồ, bảng kết quả và mã trích được tạo ở ô cuối notebook sau khi chạy đủ. Sau đó cần tích hợp với lý thuyết, rà soát trích dẫn, hình/bảng và số trang: Mở đầu 3–4; Chương 1 đúng 8; Chương 2–4 mỗi chương 10–15.

Không điền số liệu từ thực nghiệm nhỏ hoặc dùng số liệu của tài liệu khác như kết quả tự chạy. Không triển khai bản web mới trong đợt này.

## Nguồn

- Covertype: https://archive.ics.uci.edu/dataset/31/covertype
- MiniBooNE: https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification
- EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
- Kuzushiji-49: https://github.com/rois-codh/kmnist
- Jena: https://weather.bgc-jena.mpg.de/weather_data.html (bản dữ liệu 2009–2016 phân phối bởi TensorFlow).
- Điện năng: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Truy cập và kiểm tra tải ngày 08/10/2026. Các con số chính xác, lớp và checksum nằm trong `data/tieuluan/course_large/inventory.json`.
