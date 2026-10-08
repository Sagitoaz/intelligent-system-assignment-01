# Dữ liệu CSV cho thực nghiệm trên 100.000 mẫu train

Các file nằm ở `csv/<tên_bộ>/train.csv`, `validation.csv`, `test.csv`.
Đây là CSV được chuyển đổi từ định dạng gốc của nhà cung cấp, không phải tất cả nhà cung cấp đều phát hành CSV sẵn.

| Chương | Bộ dữ liệu | Train thực tế | Validation | Test |
|---|---|---:|---:|---:|
| 2 | Covertype | 464.809 | 58.101 | 58.102 |
| 2 | MiniBooNE | 103.676 | 12.960 | 12.960 |
| 3 | EMNIST Digits | 216.000 | 24.000 | 40.000 |
| 3 | Kuzushiji-49 | 209.128 | 23.237 | 38.547 |
| 4 | Jena 10 phút | 294.437 | 62.972 | 62.623 |
| 4 | Điện năng hộ gia đình | 143.160 | 30.315 | 29.684 |

Số liệu được xác nhận bằng `inventory.json` sinh từ việc đọc lại CSV. Bộ MiniBooNE đã loại các dòng toàn -999. Không dùng tăng cường hay nhân bản để đạt số mẫu.

- Ảnh: mỗi dòng là một ảnh, `pixel_000` … `pixel_783` là điểm ảnh 0–255 theo thứ tự từng hàng; `label` là nhãn. EMNIST có 10 lớp, K49 có 49 lớp.
- Bảng: `feature_*` là đặc trưng chuẩn hóa bằng thống kê train. Covertype thực hiện bài toán nhị phân lớp gốc 2 so với các loại rừng khác; không nhận là benchmark bảy lớp.
- Chuỗi: `step_000` … `step_023` là 24 quan trắc 10 phút đã chuẩn hóa; nhãn là bước kế tiếp tăng hay không tăng. `observations.csv` lưu thời gian và giá trị trước chuẩn hóa. Cửa sổ trong cùng tập có thể chồng lấn; ba tập không dùng chung quan trắc.
- `sample_id`, `label`, `previous_direction` không phải đặc trưng. `previous_direction` chỉ phục vụ mô hình cơ sở lặp hướng trước.
- CSV dùng dấu phẩy ngăn cột, dấu chấm thập phân để pandas đọc đúng. Notebook định dạng phần trình bày bằng tiếng Việt.

Notebook đọc CSV thực sự. `csv_cache/` chỉ là bộ đệm trên đĩa, tạo lại khi hash CSV thay đổi để tránh nạp hàng GB vào RAM. `prepared/` là bước trung gian phục vụ kiểm tra chuyển đổi.

CSV lớn không đưa trực tiếp vào Git. Mã tải/chuyển đổi, nguồn, checksum và kiểm kê được commit. Chạy lại trên máy khác:

```powershell
.venv\Scripts\python.exe -m scripts.tieuluan.export_large_csv
.venv\Scripts\python.exe -m scripts.tieuluan.verify_large_csv
```

Lệnh không huấn luyện. Giữ trống vài GB ổ đĩa cho nguồn, CSV, cache và checkpoint.
