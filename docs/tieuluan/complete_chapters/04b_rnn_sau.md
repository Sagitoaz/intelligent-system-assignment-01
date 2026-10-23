## 4.13. RNN và LSTM xếp chồng trên 5 lớp

Mạng hồi quy có hai chiều độ sâu khác nhau: độ dài thời gian T và số lớp xếp chồng L. Đọc 20 bước thời gian bằng một lớp RNN không có nghĩa là đã xây dựng 20 lớp học độc lập. Để đáp ứng rõ yêu cầu trên 5 layer, tiểu luận bổ sung **năm lớp hồi quy xếp chồng và một lớp Dense**, tức sáu lớp học được, cho cả SimpleRNN lẫn LSTM. Cách đếm này áp dụng đồng nhất cho Scratch, Keras và PyTorch.

Đầu vào X có kích thước B × 20 × 1. Lớp hồi quy thứ nhất đổi một đặc trưng thành tám trạng thái ở mỗi bước; bốn lớp tiếp theo giữ tám chiều. Bốn lớp đầu phải trả về toàn bộ chuỗi. Lớp cuối được lấy trạng thái ở bước 20 rồi chiếu qua Dense(8, 1). Keras dùng `return_sequences=True` ở bốn lớp đầu; PyTorch lấy tensor đầu ra của từng RNN/LSTM; Scratch dùng các lớp tự viết vốn trả về cả chuỗi [@keras_lstm_api; @torch_rnn_api].

Bảng 4.8. Kiến trúc chung của RNN sáu lớp và LSTM sáu lớp, theo một bias hiệu dụng mỗi lớp hồi quy.
| Vị trí | SimpleRNN | LSTM | Dạng tensor ra |
|---|---|---|---|
| 1 | RNN(1 → 8), 80 tham số | LSTM(1 → 8), 320 tham số | B × 20 × 8 |
| 2–5 | Mỗi RNN(8 → 8), 136 tham số | Mỗi LSTM(8 → 8), 544 tham số | B × 20 × 8 |
| Lấy bước cuối | LastStep, không tham số | LastStep, không tham số | B × 8 |
| 6 | Dense(8 → 1), 9 tham số | Dense(8 → 1), 9 tham số | B × 1 |
| Tổng | 633 tham số học được | 2.505 tham số học được | Một logit mỗi mẫu |

PyTorch và thư viện Scratch lưu hai bias cho mỗi lớp hồi quy. Để khớp số tham số học với Keras, bias hồi quy thứ hai được giữ bằng không và không cập nhật. Trong Scratch, gradient của bias này được xóa trước bước Adam; trong PyTorch đặt `requires_grad_(False)`. Hai bias vẫn có thể tồn tại trong trạng thái lưu, nhưng chỉ một bias hiệu dụng được học.

### 4.13.1. Hợp các hàm của RNN sâu

Gọi Rₗ là hàm biến đổi toàn chuỗi của lớp hồi quy l, S là LastStep, D là Dense và σ là sigmoid. Hợp hàm của mô hình là:

$$z=(D\circ S\circ R_5\circ R_4\circ R_3\circ R_2\circ R_1)(X),\qquad p=\sigma(z)$$ (4.13)

Trong mỗi Rₗ, phép cập nhật tại bước t là:

$$h_t^{(l)}=\tanh\left(W_x^{(l)}h_t^{(l-1)}+W_h^{(l)}h_{t-1}^{(l)}+b^{(l)}\right),\qquad h_t^{(0)}=x_t$$ (4.14)

Có hai đầu vào khác nhau: h ở lớp dưới cùng thời điểm t và h ở cùng lớp tại thời điểm t − 1. Nối dọc tạo khả năng học biểu diễn nhiều mức; nối ngang giữ quan hệ thời gian. Khi trình bày cần vẽ rõ hai mũi tên này, tránh nhầm trạng thái của lớp dưới với trạng thái của phiên trước.

**R₁ đến R₅.** Mỗi hàm nhận một chuỗi và trả về một chuỗi. Trọng số được dùng chung theo thời gian trong cùng một lớp; trọng số của năm lớp khác nhau. Trạng thái ban đầu bằng không khi bắt đầu một cửa sổ mới. Việc đặt lại trạng thái giúp tránh mang thông tin của mẫu trước sang mẫu sau ngoài thiết kế.

**tanh.** Hàm đưa số thực vào (−1, 1), có đạo hàm 1 − tanh²(a). Khi giá trị gần ±1, đạo hàm nhỏ, nên RNN sâu vẫn có thể tiêu biến gradient. Xếp thêm lớp không bảo đảm cải thiện chất lượng; đó là lý do cần giữ mô hình nông làm đối chứng.

**S – LastStep.** S(H) = H[:, T − 1, :]. Gradient được đưa về đúng lát cắt cuối, các vị trí còn lại của tensor gradient ban đầu bằng không. BPTT sau đó vẫn truyền gradient từ trạng thái cuối về các thời điểm trước thông qua quan hệ hồi quy. Lấy bước cuối không đồng nghĩa bỏ qua lịch sử.

**D và σ.** Dense tạo logit từ trạng thái cuối; sigmoid đổi logit thành xác suất. Dùng chung cách tính BCE từ logit như CNN để so sánh ba công cụ trên cùng bài toán phân loại nhị phân.

### 4.13.2. Hợp các hàm của LSTM sâu

Thay mỗi Rₗ bằng Lₗ, một phép biến đổi toàn chuỗi bằng ô LSTM. Các công thức cổng ở mục 4.4 áp dụng độc lập cho từng lớp:

$$z=(D\circ S\circ L_5\circ L_4\circ L_3\circ L_2\circ L_1)(X),\qquad p=\sigma(z)$$ (4.15)

**Cổng quên f.** Hàm sigmoid của đầu vào và trạng thái cũ quyết định phần ô nhớ cũ được giữ lại. f gần một cho phép thông tin đi qua nhiều bước; f gần không xóa thông tin. Đây là cơ chế điều tiết số học, không phải thao tác xóa một dòng văn bản.

**Cổng vào i và ứng viên g.** i nằm trong (0, 1), g nằm trong (−1, 1). Tích từng phần tử i ⊙ g là lượng thông tin mới ghi vào ô nhớ. Ví dụ c cũ = 0,4, f = 0,8, i = 0,3 và g = 0,5 cho c mới = 0,8 × 0,4 + 0,3 × 0,5 = 0,47.

**Cổng ra o.** Hàm h = o ⊙ tanh(c) quyết định phần ô nhớ được đưa ra thành trạng thái ẩn. Nếu o = 0,6 và c = 0,47 thì h xấp xỉ 0,263. Trong LSTM sâu, chuỗi h được chuyển sang lớp kế; c là bộ nhớ nội bộ của chính lớp đó, không chuyển thẳng làm c của lớp trên.

**BPTT và quy tắc dây chuyền.** Gradient đi qua cả năm lớp và qua các bước thời gian. Với hợp hàm F = fₖ ∘ … ∘ f₁, Jacobian của F là tích các Jacobian theo đúng thứ tự. Trong LSTM, đường qua c có thành phần f ở từng bước, giúp duy trì gradient trong một số điều kiện, nhưng không loại bỏ hoàn toàn vấn đề tiêu biến hay bùng nổ. Mã áp dụng cắt chuẩn gradient toàn cục ở mức 1.

### 4.13.3. Quy trình huấn luyện, đánh giá và giải thích hàm

`scratch_deep`, `keras_deep`, `torch_deep` dựng cùng đặc tả kiến trúc. `align_weights` khớp trọng số hiệu dụng, bao gồm thứ tự cổng (i, f, g, o) và một bias hiệu dụng. `train_batch` thực hiện một lần forward, tính loss, backward, cắt gradient và cập nhật. Chức năng từng nhánh được diễn giải ở Phụ lục C. Mã kiểm chứng mới chỉ xác nhận triển khai và khả năng thực hiện bước học trên mini-batch tổng hợp; các chỉ số AUC, bootstrap và backtest ở mục 4.7–4.12 thuộc mô hình cũ đã được huấn luyện trên dữ liệu thị trường.

Nếu mở rộng thực nghiệm thị trường cho mạng sáu lớp, phải giữ nguyên cách chia thời gian, scaler chỉ fit trên train, cùng số lượt thử siêu tham số và cùng dữ liệu test như mô hình nông. Chọn checkpoint bằng validation, sau đó mới tính AUC test, AP và phí giao dịch. Không dùng loss giảm trên bốn mẫu để suy ra mạng tổng quát hóa tốt hơn.

Demo bắt đầu từ một chuỗi 20 phiên, thể hiện shape B × 20 × 1, rồi lần lượt năm tensor B × 20 × 8 và vector B × 8. Một thành viên giải thích hợp hàm; thành viên tiếp theo tính ví dụ ô LSTM bằng số; thành viên phụ trách mã chạy đối chiếu ba framework. Khi thảo luận, dùng hai câu hỏi kiểm tra: vì sao `return_sequences=False` ở lớp đầu làm hỏng xếp chồng, và vì sao 20 thời điểm không được đếm là 20 lớp học được?
