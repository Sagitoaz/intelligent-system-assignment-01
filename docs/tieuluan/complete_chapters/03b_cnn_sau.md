## 3.12. CNN trên 5 lớp và biểu diễn bằng hợp các hàm

Để đáp ứng yêu cầu mô hình phải trên 5 layer, phần bổ sung sử dụng cách đếm chặt: **chỉ đếm lớp có tham số học được**, không tính Input, ReLU, Permute, Flatten hay sigmoid. CNN có bốn lớp tích chập và hai lớp Dense, tổng cộng sáu lớp học được. Các mô hình CNN4, CNN8 và CNN sâu hai khối ở phần thực nghiệm trước vẫn là các cấu hình đối chứng; số liệu của chúng không được gán cho kiến trúc mới.

Bảng 3.8. Kích thước và tham số của CNN sáu lớp học được, đầu vào một ảnh GASF 20 × 20.
| Lớp học được | Biến đổi | Đầu ra mỗi mẫu (C, H, W) | Số tham số |
|---|---|---|---|
| 1 | Conv2D, 1 → 4, kernel 3 × 3, valid | 4 × 18 × 18 | 40 |
| 2 | Conv2D, 4 → 4, kernel 3 × 3, valid | 4 × 16 × 16 | 148 |
| 3 | Conv2D, 4 → 8, kernel 3 × 3, valid | 8 × 14 × 14 | 296 |
| 4 | Conv2D, 8 → 8, kernel 3 × 3, valid | 8 × 12 × 12 | 584 |
| 5 | Flatten rồi Dense, 1.152 → 8 | 8 | 9.224 |
| 6 | Dense, 8 → 1 | 1 logit | 9 |
| Tổng | Bốn Conv và hai Dense | Xác suất sau sigmoid | 10.301 |

Với kernel 3, stride 1 và không padding, cạnh ảnh giảm hai đơn vị sau mỗi Conv; sau bốn lớp là 20 − 2 × 4 = 12. Tại lớp thứ năm, số chiều vào bằng 8 × 12 × 12 = 1.152. Công thức đếm tham số Conv là C_out × (C_in × 3 × 3 + 1); Dense là d_out × (d_in + 1). Mỗi Conv và Dense ẩn đi kèm ReLU. Không dùng pooling ở kiến trúc minh họa để giữ việc kiểm tra kích thước đơn giản; đây là lựa chọn thiết kế, không phải bằng chứng rằng bỏ pooling sẽ tốt hơn.

Gọi C₁, C₂, C₃, C₄ là bốn phép tích chập có bias; ρ là ReLU; V là Flatten; D₅, D₆ là hai phép affine. Mạng trả về logit z và xác suất p như sau:

$$z = (D_6 \circ \rho \circ D_5 \circ V \circ \rho \circ C_4 \circ \rho \circ C_3 \circ \rho \circ C_2 \circ \rho \circ C_1)(X), \qquad p=\sigma(z)$$ (3.12)

Đây là **hợp các hàm**, đọc từ phải sang trái: ảnh đi qua C₁ trước, D₆ sau cùng. Dấu ∘ không phải phép cộng hay phép nhân trọng số. Các hàm nối nhau vì miền giá trị của hàm trước phù hợp với miền xác định của hàm sau. Viết thành từng bước giúp kiểm tra: A₁ = ρ(C₁(X)); A₂ = ρ(C₂(A₁)); A₃ = ρ(C₃(A₂)); A₄ = ρ(C₄(A₃)); u = V(A₄); v = ρ(D₅(u)); z = D₆(v).

### 3.12.1. Giải thích từng hàm riêng

**Cᵢ – tích chập.** Mỗi bộ lọc tạo một kênh đặc trưng. Ở vị trí (u, v), đầu ra là tổng các tích giữa cửa sổ đầu vào và trọng số, cộng bias. Trong mã, thao tác thường gọi là convolution thực chất dùng tương quan chéo: không lật kernel trước khi nhân. Các bản Scratch, Keras và PyTorch phải cùng quy ước. Scratch tự thực hiện im2col, nhân ma trận và cộng dồn gradient của các cửa sổ chồng lấp trong backward.

**ρ – ReLU.** Hàm ρ(a) = max(0, a) giữ số dương, đưa số âm về không. Ví dụ (−2; 0; 3) thành (0; 0; 3). Backward nhân gradient với mặt nạ a > 0; tại a = 0 mã chọn đạo hàm bằng 0. ReLU tạo phi tuyến nhưng không thêm tham số học được.

**V – Flatten.** Hàm này chỉ đổi hình dạng 8 × 12 × 12 thành vector 1.152 phần tử. Nó không học trọng số và không trộn giá trị. Thứ tự trải kênh phải giống nhau ở ba bản. Keras dùng hai Permute để chuyển NCHW sang NHWC trước Conv và đưa về NCHW trước Flatten.

**D₅ và D₆ – Dense.** D₅(u) = W₅u + b₅ tổng hợp các đặc trưng cục bộ thành tám đặc trưng toàn cục. D₆(v) = W₆v + b₆ tạo một logit thực. Bias được tính đúng một lần ở mỗi phép affine.

**σ – sigmoid.** σ(z) = 1/(1 + exp(−z)). Logit 0 tương ứng xác suất 0,5; logit khoảng 1,099 tương ứng 0,75. Khi huấn luyện, truyền z trực tiếp vào BCEWithLogits, không sigmoid trước rồi lại dùng mất mát từ logit. Khi hiển thị demo mới áp dụng sigmoid để người xem đọc xác suất.

### 3.12.2. Ba bản cài đặt và cách kiểm chứng

Mã đầy đủ dựng kiến trúc nằm trong Phụ lục C, được đưa trực tiếp từ tệp `scripts/tieuluan/complete_models.py`. Scratch sử dụng thư viện NumPy tự viết trong `src/tieuluan/scratch/`; các lớp này tự cài forward và backward, không gọi autograd. Keras khai báo Functional API; PyTorch ghép các nn.Module. Cả ba đều có đầu vào batch × 1 × 20 × 20 và đầu ra batch × 1.

Việc dùng cùng seed chưa đủ để tạo cùng trọng số giữa ba thư viện. Hàm `align_weights` sao chép trọng số Scratch sang hai bản còn lại, chuyển trục kernel Conv và chuyển vị ma trận Dense theo quy ước tương ứng. Phép kiểm tra trong Phụ lục C so sánh logit trước khi học, gradient Scratch–PyTorch và ba bước tối ưu thật. Dữ liệu kiểm tra là mini-batch tổng hợp để phát hiện sai kích thước, sai thứ tự trọng số và sai đạo hàm; nó không dùng để kết luận khả năng dự báo thị trường.

### 3.12.3. Kịch bản biểu diễn CNN

Khi trình bày, trước hết chọn một cửa sổ 20 phiên trong ứng dụng tiểu luận, hiển thị chuỗi lợi suất và ảnh GASF. Giải thích vì sao cùng một cửa sổ tạo được ảnh 20 × 20; việc chuyển thành ảnh không tạo thêm thông tin gốc. Tiếp theo dùng bảng 3.8 để đi qua sáu lớp của bản bổ sung và chỉ rõ đầu vào–đầu ra ở mỗi bước. Chạy lệnh kiểm chứng Phụ lục C để thấy ba bản cho logit gần nhau. Cuối cùng quay về bảng thực nghiệm cũ để bàn luận AUC và phí giao dịch, nêu đúng tên kiến trúc đã tạo ra kết quả đó. Không trình bày xác suất tăng như một lệnh mua bán.
