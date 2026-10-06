# CHƯƠNG 4. MẠNG NƠ-RON HỒI QUY (RNN)

## 4.1. Dữ liệu chuỗi và vì sao thứ tự quan trọng

Hai chuỗi lợi suất +1%, +1%, −1%, −1% và +1%, −1%, +1%, −1% có cùng trung bình bằng 0, nhưng kể hai câu chuyện khác nhau: một bên là hai phiên tăng nối tiếp rồi đảo chiều, bên kia là giá giằng co luân phiên. MLP ở Chương 2 nhận đầu vào như một bảng số không có thứ tự; CNN ở Chương 3 phải biến chuỗi thành ảnh. **Mạng nơ-ron hồi quy** (recurrent neural network – RNN) đọc chuỗi theo đúng thứ tự thời gian và mang theo một **trạng thái ẩn** tóm tắt những gì đã đọc [@elman1990].

> **Hiểu nhanh.** RNN đọc chuỗi như người đọc một câu chuyện: đọc từng câu, sau mỗi câu cập nhật lại "trí nhớ" về diễn biến, và dùng trí nhớ đó để đoán điều gì xảy ra tiếp theo. Trạng thái ẩn chính là "trí nhớ" ấy – một vector số được học, không phải một bản ghi chép hoàn hảo.

## 4.2. Mạng hồi quy đơn giản (SimpleRNN)

Tại mỗi bước t, SimpleRNN kết hợp đầu vào mới xₜ với trạng thái cũ hₜ₋₁ để tạo trạng thái mới:

$$\mathbf{h}_t = \tanh\left(W_x\mathbf{x}_t + W_h\mathbf{h}_{t-1} + \mathbf{b}\right)$$ (4.1)

Ma trận Wₓ quyết định thông tin mới được đưa vào thế nào, Wₕ quyết định trí nhớ cũ được giữ lại thế nào, còn tanh giữ mỗi phần tử trạng thái trong khoảng (−1, 1). Điểm mấu chốt là **cùng một bộ trọng số** được dùng lại ở mọi bước thời gian. Nhờ vậy chuỗi dài hay ngắn thì số tham số vẫn như nhau: với d đầu vào và h phần tử trạng thái, lớp có h·d + h² + h tham số. Trong thực nghiệm (d = 1, h = 8) là 8 + 64 + 8 = 80 tham số. Khi "trải" mạng theo thời gian (Hình 4.1), ta thấy RNN thực chất là một mạng rất sâu, mỗi bước thời gian là một lớp, và các lớp dùng chung trọng số. Với bài toán "đọc 20 phiên, dự báo một lần", chỉ trạng thái cuối h₂₀ được đưa qua lớp kết nối đầy đủ và hàm sigmoid để ra xác suất tăng.

![Hình 4.1. Mạng hồi quy ở dạng vòng lặp (trái) và dạng trải theo thời gian (phải).](../../figures/tieuluan/diagram_rnn_unrolled.png)

**Ví dụ tính tay** với trạng thái một chiều: Wₓ = 1, Wₕ = 0,5, b = 0, h₀ = 0 và chuỗi đầu vào 0,1; −0,2; 0,3 (Bảng 4.1). Trạng thái cuối 0,2218 không chỉ phụ thuộc x₃ mà còn mang dấu vết của x₁ và x₂; nếu đảo thứ tự ba giá trị, trạng thái cuối sẽ khác. Đây chính là khả năng "nhớ thứ tự" mà phép lấy trung bình không có.

Bảng 4.1. Lan truyền xuôi của SimpleRNN một chiều (Wₓ = 1, Wₕ = 0,5, b = 0, h₀ = 0).
| Bước t | xₜ | Tổng trước tanh | Trạng thái hₜ |
|---|---|---|---|
| 1 | 0,1 | 0,1 + 0,5 × 0 = 0,1 | 0,0997 |
| 2 | −0,2 | −0,2 + 0,5 × 0,0997 = −0,1502 | −0,1490 |
| 3 | 0,3 | 0,3 + 0,5 × (−0,1490) = 0,2255 | 0,2218 |

## 4.3. Lan truyền ngược theo thời gian và bài toán gradient

RNN được huấn luyện bằng **lan truyền ngược theo thời gian** (backpropagation through time – BPTT): trải mạng thành chuỗi các bước rồi áp dụng lan truyền ngược như một mạng sâu bình thường [@werbos1990]. Vì Wₕ được dùng ở mọi bước, gradient của nó là tổng đóng góp từ mọi bước. Để biết đầu vào ở bước đầu ảnh hưởng đến kết quả ra sao, gradient phải đi ngược qua tích của nhiều đạo hàm:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_1} = \prod_{t=2}^{T}\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$ (4.2)

Nếu mỗi thừa số có độ lớn khoảng 0,5, sau 20 bước tích còn 0,5²⁰ ≈ 0,00000095: tín hiệu học từ quá khứ xa gần như biến mất. Đó là hiện tượng **tiêu biến gradient** (vanishing gradient), được Bengio và cộng sự phân tích từ năm 1994 [@bengio1994]. Ngược lại, nếu mỗi thừa số khoảng 1,5, tích lên tới 1,5²⁰ ≈ 3.325, gây **bùng nổ gradient**: một bước cập nhật quá lớn có thể phá hỏng cả mô hình.

> **Hiểu nhanh.** Tiêu biến gradient giống trò chơi "tam sao thất bản": thông điệp truyền qua 20 người, mỗi người chỉ nói lại được một nửa, đến người cuối thì gần như không còn gì. Mạng vì thế khó học được rằng một sự kiện từ 20 phiên trước có liên quan đến hôm nay.

Bùng nổ gradient được xử lý đơn giản bằng **cắt chuẩn gradient** (gradient clipping) [@pascanu2013]: nếu độ dài vector gradient vượt ngưỡng c thì co nó lại về đúng độ dài c, giữ nguyên hướng.

$$\mathbf{g} \leftarrow \mathbf{g}\cdot\min\left(1,\ \frac{c}{\Vert\mathbf{g}\Vert}\right)$$ (4.3)

Ví dụ g = (6, 8) có độ dài 10; với c = 5, gradient được co thành (3, 4). Mọi mô hình trong thực nghiệm đều dùng c = 1. Tiêu biến gradient thì khó hơn nhiều, và lời giải nổi tiếng nhất là LSTM.

## 4.4. LSTM: bộ nhớ có cổng điều khiển

LSTM (long short-term memory) của Hochreiter và Schmidhuber [@hochreiter1997], với cổng quên do Gers và cộng sự bổ sung [@gers2000], thêm một **ô nhớ** cₜ chạy dọc theo chuỗi như một "băng chuyền", cùng ba **cổng** điều khiển việc ghi, xóa và đọc (Hình 4.2). Mỗi cổng là một lớp sigmoid cho giá trị từ 0 (đóng) đến 1 (mở), tính từ hₜ₋₁ và xₜ:

$$\mathbf{f}_t,\ \mathbf{i}_t,\ \mathbf{o}_t = \sigma\left(W_{\{f,i,o\}}[\mathbf{h}_{t-1},\mathbf{x}_t] + \mathbf{b}_{\{f,i,o\}}\right)$$
$$\mathbf{g}_t = \tanh\left(W_g[\mathbf{h}_{t-1},\mathbf{x}_t] + \mathbf{b}_g\right)$$ (4.4)

$$\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\mathbf{g}_t, \qquad \mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{c}_t)$$ (4.5)

Ký hiệu ⊙ là phép nhân từng phần tử. **Cổng quên** fₜ quyết định giữ lại bao nhiêu phần trí nhớ cũ; **cổng vào** iₜ quyết định ghi bao nhiêu thông tin mới gₜ; **cổng ra** oₜ quyết định đưa bao nhiêu phần trí nhớ ra ngoài thành hₜ.

![Hình 4.2. Cấu trúc một ô LSTM: băng chuyền ô nhớ cₜ (phía trên) và bốn khối tính cổng (phía dưới).](../../figures/tieuluan/diagram_lstm_cell.png)

**Ví dụ.** Giả sử cₜ₋₁ = 0,8; fₜ = 0,9; iₜ = 0,2; gₜ = −0,5; oₜ = 0,7. Ô nhớ mới cₜ = 0,9 × 0,8 + 0,2 × (−0,5) = 0,62, và hₜ = 0,7 × tanh(0,62) ≈ 0,386. Thông tin mới mang dấu âm nhưng không xóa sạch trí nhớ, vì cổng quên giữ lại 90%. Nếu fₜ chỉ bằng 0,1, cₜ sẽ còn −0,02, tức mô hình gần như "quên" quá khứ để chạy theo thông tin mới.

Vì sao LSTM giảm được tiêu biến gradient? Trên băng chuyền ô nhớ, đạo hàm của cₜ theo cₜ₋₁ (theo đường trực tiếp) chính là fₜ. Khi cổng quên gần 1, gradient đi ngược qua nhiều bước mà không bị co lại theo cấp số nhân như ở (4.2). Cái giá phải trả là số tham số gấp bốn lần SimpleRNN: 4h(d + h + 1), tức 320 tham số với d = 1, h = 8.

> **Hiểu nhanh.** LSTM giống người ghi sổ tay có ba thói quen: trước khi ghi điều mới, cân nhắc xóa bớt điều đã cũ (cổng quên); chỉ ghi những gì thật sự quan trọng (cổng vào); và khi được hỏi, chỉ đọc phần liên quan (cổng ra).

## 4.5. GRU: phiên bản gọn hơn

GRU (gated recurrent unit) của Cho và cộng sự [@cho2014] gộp ô nhớ và trạng thái ẩn làm một, chỉ dùng hai cổng: **cổng đặt lại** rₜ (bao nhiêu trí nhớ cũ được dùng khi tạo ứng viên) và **cổng cập nhật** zₜ (trộn trí nhớ cũ với ứng viên mới theo tỉ lệ nào). Theo quy ước của PyTorch và cuDNN mà thực nghiệm sử dụng:

$$\mathbf{n}_t = \tanh\left(W_{in}\mathbf{x}_t + \mathbf{b}_{in} + \mathbf{r}_t\odot(W_{hn}\mathbf{h}_{t-1} + \mathbf{b}_{hn})\right)$$
$$\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{n}_t + \mathbf{z}_t\odot\mathbf{h}_{t-1}$$ (4.6)

Ví dụ hₜ₋₁ = 0,8; ứng viên nₜ = −0,2; zₜ = 0,75 cho hₜ = 0,25 × (−0,2) + 0,75 × 0,8 = 0,55: trạng thái giữ 75% quá khứ. Bảng 4.2 so sánh ba kiến trúc hồi quy.

Bảng 4.2. So sánh SimpleRNN, LSTM và GRU (số tham số tính với d = 1, h = 8, chưa kể lớp đầu ra).
| Đặc điểm | SimpleRNN | LSTM | GRU |
|---|---|---|---|
| Trạng thái mang theo | hₜ | hₜ và ô nhớ cₜ | hₜ |
| Cổng | Không có | Quên, vào, ra | Đặt lại, cập nhật |
| Số tham số | 80 | 320 | 264 (hai vector bias) |
| Phụ thuộc dài hạn | Khó học | Tốt | Tốt |
| Chi phí tính toán | Thấp nhất | Cao nhất | Trung bình |

## 4.6. Chuẩn bị chuỗi tài chính cho mạng hồi quy

**Dự báo lợi suất, không dự báo mức giá.** Một sai lầm phổ biến là cho mạng dự báo trực tiếp giá đóng cửa ngày mai rồi khoe hệ số xác định R² rất cao. Vì giá hôm nay đã rất gần giá ngày mai, ngay cả dự báo "ngây thơ" Ĉ(t+1) = C(t) – không học gì – cũng đạt R² = {{v:naive_r2.sp500}} trên tập test của S&P 500 (và {{v:naive_r2.vnindex}} với VN-Index). R² cao khi dự báo mức giá vì thế không chứng minh mô hình có khả năng dự báo. Tiểu luận dùng **lợi suất logarit** rₜ = ln(Cₜ/Cₜ₋₁), đại lượng có thang đo ổn định theo thời gian, và đặt bài toán là đoán **hướng** của phiên kế tiếp. Đây cũng chính là bài toán của Chương 3, nên có thể so sánh trực tiếp CNN với RNN.

**Chuẩn hóa không nhìn tương lai.** Mỗi mẫu là cửa sổ 20 lợi suất đến ngày t, chuẩn hóa z-score bằng trung bình và độ lệch chuẩn tính trên **tập train**. Validation và test dùng lại nguyên hai con số đó, vì lúc dự báo thật ta không biết thống kê của tương lai. Cách chia theo thời gian, cách loại mẫu ở ranh giới và hai đường cơ sở giống hệt Chương 3 (Bảng 3.4).

**Không trộn thời gian.** Hai cửa sổ liền kề chung 19/20 quan sát. Nếu trộn ngẫu nhiên các cửa sổ rồi mới chia train/test, gần như mọi cửa sổ test đều có "anh em sinh đôi" trong train, và kết quả sẽ đẹp một cách giả tạo [@lopezdeprado2018]. Việc trộn thứ tự mini-batch **bên trong** tập train thì vẫn hợp lệ, vì không có thông tin nào vượt khỏi tập train.

## 4.7. Cài đặt bằng ba cách

Mô hình đại diện của chương là LSTM với 8 phần tử trạng thái: chuỗi 20 × 1 → LSTM(8) → trạng thái cuối → lớp kết nối đầy đủ → logit, tổng cộng {{v:params.sp500.lstm.pytorch}} tham số học. Bản NumPy tự viết toàn bộ lan truyền xuôi (4.4)–(4.5) và BPTT, với bố cục trọng số và thứ tự cổng (i, f, g, o) trùng PyTorch. Nhờ đó có thể chép nguyên trọng số giữa ba bản cài đặt; riêng GRU của Keras xếp cổng theo thứ tự (z, r, n) nên phải hoán vị khi chép. PyTorch có hai vector bias cộng dồn (b_ih + b_hh) còn Keras chỉ có một, nên ở SimpleRNN và LSTM, vector b_hh dư thừa được cố định bằng 0, để ba bản có cùng số tham số học. Đoạn mã dưới là một bước của LSTM tự viết:

```python
z = xw[:, t] + h @ self.weight_hh.value.T + self.bias_hh.value   # 4 khối cổng cùng lúc
i, f = sigmoid(z[:, :H]), sigmoid(z[:, H:2*H])                    # cổng vào, cổng quên
g, o = np.tanh(z[:, 2*H:3*H]), sigmoid(z[:, 3*H:])               # ứng viên, cổng ra
c = f * c + i * g                                                 # cập nhật ô nhớ (4.5)
h = o * np.tanh(c)                                                # trạng thái ẩn mới
```

Kiểm tra gradient cho sai số tương đối lớn nhất {{v:gc.rnn}} với SimpleRNN, {{v:gc.lstm}} với LSTM và {{v:gc.gru}} với GRU. Mức 10⁻⁵ của LSTM đến từ một số thành phần gradient rất nhỏ, nơi sai số làm tròn của phép sai phân trở nên đáng kể; tính trên toàn vector gradient, sai số tương đối chỉ còn {{v:gcnorm.lstm}}. Trước huấn luyện, logit của ba bản cài đặt lệch nhau tối đa {{v:parity.lstm}}.

## 4.8. Kết quả và thảo luận

**Ba bản cài đặt.** Bảng 4.3 và Hình 4.3 cho thấy mức trùng khớp còn cao hơn Chương 2 và 3: ba bản LSTM dừng ở cùng epoch với mọi hạt giống, và ROC-AUC theo từng hạt giống chênh nhau {{v:aucdiff.lstm.keras}}. Xác suất dự báo trên tập test của bản NumPy và PyTorch lệch nhau tối đa {{v:parity_trained.sp500.lstm.pytorch}} (S&P 500), của NumPy và Keras tối đa {{v:parity_trained.sp500.lstm.keras}}. Về thời gian, bản Keras chậm nhất vì chi phí điều phối cho mỗi lần gọi `train_on_batch` lớn so với một mạng chỉ có vài trăm tham số.

{{t:ch4_frameworks:4.3}}

![Hình 4.3. BCE trên tập validation theo epoch của ba bản cài đặt LSTM (hạt giống 11); ba đường gần như trùng nhau.](../../figures/tieuluan/ch4_learning_curves.png)

**Ba kiến trúc hồi quy.** Bảng 4.4 và Hình 4.4 so sánh SimpleRNN, LSTM và GRU với hai đường cơ sở. Bức tranh giống Chương 3. Ở S&P 500, cả ba mô hình đều có ROC-AUC dưới 0,5 (LSTM: {{v:ci.sp500.lstm}}). Ở Bitcoin, GRU cao nhất với {{v:ci.btc.gru}}, nhưng khoảng tin cậy vẫn chứa 0,5. Chỉ ở VN-Index, LSTM ({{v:ci.vnindex.lstm}}) và GRU ({{v:ci.vnindex.gru}}) có khoảng tin cậy nằm hoàn toàn trên 0,5 và cao hơn đường cơ sở "lặp lại hướng phiên trước" ({{v:ci.vnindex.persistence}}). Khác biệt rõ nhất giữa các kiến trúc là **độ ổn định**: ROC-AUC của SimpleRNN dao động mạnh giữa các hạt giống ({{v:aucsd.vnindex.rnn.pytorch}} ở VN-Index), trong khi LSTM và GRU gần như không đổi ({{v:aucsd.vnindex.lstm.pytorch}} và {{v:aucsd.vnindex.gru.pytorch}}). Cơ chế cổng giúp quá trình học ít phụ thuộc vào điểm xuất phát, đúng như phân tích ở mục 4.3–4.5. Giữa LSTM và GRU không có khác biệt đáng kể; GRU đạt mức tương đương với ít tham số hơn.

{{t:ch4_architectures:4.4}}

![Hình 4.4. ROC-AUC trên tập test và khoảng tin cậy 95% (bootstrap theo khối 20 phiên) của SimpleRNN, LSTM, GRU và đường cơ sở "lặp lại hướng phiên trước".](../../figures/tieuluan/ch4_auc_ci.png)

**Vì sao VN-Index khác hai thị trường còn lại?** Câu trả lời nằm ngay trong dữ liệu. Hệ số tự tương quan bậc một của lợi suất ngày VN-Index là {{v:ac1.vnindex}}: sau một phiên tăng, xác suất phiên kế tiếp cũng tăng là {{v:pupup.vnindex}}, còn sau một phiên giảm chỉ là {{v:pupdown.vnindex}}. Với S&P 500, hệ số này âm ({{v:ac1.sp500}}), còn với Bitcoin gần bằng 0 ({{v:ac1.btc}}). Nói cách khác, VN-Index có **quán tính ngắn hạn**. Hiện tượng lợi suất chỉ số tự tương quan dương đã được Lo và MacKinlay ghi nhận trên thị trường Mỹ từ cuối thập niên 1980 [@lo1988]; một cách giải thích phổ biến là giá cổ phiếu nhỏ, ít thanh khoản phản ứng với thông tin chậm hơn cổ phiếu lớn [@lo1990]. Đường cơ sở "lặp lại hướng phiên trước" khai thác đúng quán tính này, còn LSTM và GRU chỉ nhỉnh hơn nó một chút. Tín hiệu các mạng học được vì vậy chủ yếu là quán tính đơn giản, chứ không phải một quy luật phức tạp.

**So sánh với CNN.** Trên cùng dữ liệu, cùng nhãn và cùng cách chia, LSTM và GRU đọc trực tiếp chuỗi lợi suất cho kết quả nhỉnh hơn và ổn định hơn CNN đọc ảnh GASF ở VN-Index (ROC-AUC {{v:aucsd.vnindex.lstm.pytorch}} của LSTM so với {{v:aucsd.vnindex.cnn4.pytorch}} của CNN4), dù chênh lệch còn nhỏ so với độ rộng khoảng tin cậy. Ở S&P 500 và Bitcoin, không mô hình nào vượt được đoán ngẫu nhiên. Kết quả ủng hộ nhận định của chính Jiang, Kelly và Xiu rằng một mô hình chuỗi thời gian tốt có thể vượt CNN trên ảnh [@jiang2023]. Nó cũng cho thấy với thông tin chỉ gồm giá quá khứ, kiến trúc mạng không thể tạo ra tín hiệu ở nơi tín hiệu không tồn tại.

## 4.9. Từ dự báo đến quyết định đầu tư

Một ROC-AUC có ý nghĩa thống kê chưa chắc đem lại lợi nhuận. Để kiểm tra, tiểu luận mô phỏng một chiến lược đơn giản trên tập test: cuối mỗi phiên, nếu xác suất tăng (trung bình ba hạt giống) không thấp hơn ngưỡng đã chọn trên validation thì nắm giữ chỉ số trong phiên kế tiếp, ngược lại đứng ngoài và giữ tiền mặt. Mỗi lần mua hoặc bán trả phí {{v:cost.sp500}} với S&P 500 (giao dịch quỹ ETF), {{v:cost.vnindex}} với VN-Index (phí môi giới khoảng 0,15% mỗi chiều cộng thuế 0,1% trên giá trị bán, bình quân 0,2% mỗi chiều) và {{v:cost.btc}} với Bitcoin (phí sàn giao ngay phổ biến). Chiến lược so sánh là **mua và giữ** suốt giai đoạn test. Đây là mô phỏng lạc quan: giả định giao dịch được đúng giá đóng cửa vừa quan sát và không tính lãi tiền gửi khi đứng ngoài.

{{t:ch4_backtest:4.5}}

![Hình 4.5. Giá trị danh mục trên tập test của chiến lược mua và giữ so với các chiến lược theo tín hiệu LSTM, GRU và CNN4 (đã trừ phí).](../../figures/tieuluan/ch4_equity.png)

Bảng 4.5 và Hình 4.5 cho thấy sau khi trừ phí, không chiến lược theo tín hiệu nào thắng mua và giữ một cách thuyết phục. Ở S&P 500, LSTM cho kết quả gần trùng mua và giữ (lợi suất năm {{v:bt.sp500.lstm.ret}} so với {{v:bt.sp500.buy_hold.ret}}) đơn giản vì nó hầu như luôn dự báo "tăng" và nắm giữ tới {{v:bt.sp500.lstm.exposure}} thời gian: mô hình học lại xu hướng tăng dài hạn chứ không phải một tín hiệu ngắn hạn. Ở VN-Index, nơi có ROC-AUC tốt nhất, cột "trước phí" cho thấy tín hiệu là có thật: chỉ nắm giữ {{v:bt.vnindex.gru.exposure}} thời gian, chiến lược theo GRU đạt {{v:bt.vnindex.gru.retgross}} mỗi năm với Sharpe {{v:bt.vnindex.gru.sharpegross}}, vượt mua và giữ ({{v:bt.vnindex.buy_hold.retgross}}, Sharpe {{v:bt.vnindex.buy_hold.sharpegross}}). Nhưng sau {{v:bt.vnindex.gru.entries}} lần mua và bán với mức phí của thị trường Việt Nam, tổng phí cộng dồn tương đương {{v:bt.vnindex.gru.cost}} giá trị danh mục, và lợi suất năm chỉ còn {{v:bt.vnindex.gru.ret}}, thua mua và giữ. Đường cơ sở "lặp lại hướng phiên trước" còn rõ hơn: trước phí đạt {{v:bt.vnindex.persistence.retgross}} mỗi năm, sau phí lỗ {{v:bt.vnindex.persistence.ret}} vì phải mua {{v:bt.vnindex.persistence.entries}} lần. LSTM trên VN-Index, dù ROC-AUC tương đương GRU, lại có ngưỡng (chọn trên validation 2020–2022) quá cao so với giai đoạn test nên chỉ nắm giữ {{v:bt.vnindex.lstm.exposure}} thời gian. ROC-AUC đo khả năng **xếp hạng**, không bảo đảm một ngưỡng quyết định tốt. Ở Bitcoin, mua và giữ đạt {{v:bt.btc.buy_hold.ret}} mỗi năm trong giai đoạn test, mức mà không chiến lược đứng ngoài nào theo kịp.

Kết quả này trùng với bài học của Fischer và Krauss. Trong nghiên cứu năm 2018, LSTM dự báo cổ phiếu nào sẽ vượt trung vị thị trường với độ chính xác khoảng 54,3% và tạo lợi nhuận 0,46% mỗi ngày trước chi phí trên giai đoạn 1992–2015. Nhưng từ khoảng năm 2010, lợi nhuận sau chi phí chỉ còn dao động quanh 0, vì thị trường đã "học" và triệt tiêu quy luật [@fischer2018]. Riêng với Việt Nam còn một rào cản thực tế: chu kỳ thanh toán T+2 khiến cổ phiếu hay chứng chỉ quỹ mua hôm nay chưa thể bán ngay hôm sau. Một chiến lược vào – ra theo từng phiên vì thế chỉ thực hiện được qua hợp đồng tương lai, mà hợp đồng tương lai lại theo chỉ số VN30 chứ không phải VN-Index.

## 4.10. Xu hướng mới: sự trở lại của mạng hồi quy

Từ năm 2017, Transformer gần như thay thế RNN trong xử lý ngôn ngữ nhờ khả năng tính song song trên toàn chuỗi [@vaswani2017]. Nhưng Transformer có chi phí tăng theo bình phương độ dài chuỗi, và điều này mở đường cho một **"thời kỳ phục hưng" của mạng hồi quy** từ năm 2023. Ý tưởng chung là thiết kế phép cập nhật trạng thái sao cho vừa huấn luyện song song được như Transformer, vừa suy luận tuần tự với bộ nhớ cố định như RNN (Bảng 4.6).

Bảng 4.6. Một số hướng mạng hồi quy và mô hình chuỗi thời gian mới (2023–2026).
| Mô hình | Năm, nơi công bố | Ý tưởng chính |
|---|---|---|
| Mamba [@gu2023mamba] | 2023 | Mô hình không gian trạng thái có cơ chế "chọn lọc" thông tin theo nội dung |
| Mamba-2, Mamba-3 [@dao2024mamba2; @lahoti2026mamba3] | ICML 2024; ICLR 2026 | Liên hệ chặt với attention; cải tiến cách cập nhật trạng thái |
| xLSTM [@beck2024xlstm] | NeurIPS 2024 | LSTM với cổng hàm mũ và ô nhớ dạng ma trận |
| minLSTM, minGRU [@feng2024minrnn] | 2024 | Bỏ phụ thuộc của cổng vào trạng thái cũ để huấn luyện song song |
| RWKV-7 [@peng2025rwkv7] | 2025 | RNN cạnh tranh với Transformer ở quy mô mô hình ngôn ngữ |
| Titans [@behrouz2025titans] | 2025 | Bộ nhớ dài hạn được học ngay trong lúc suy luận |
| Chronos, Chronos-2 [@ansari2024chronos; @ansari2025chronos2] | 2024; 2025 | Mô hình nền tảng dự báo chuỗi thời gian, mã hóa giá trị thành token |
| TimesFM [@das2024timesfm] | ICML 2024 | Mô hình nền tảng dạng decoder cho chuỗi thời gian |
| TiRex [@auer2025tirex] | NeurIPS 2025 | Dự báo zero-shot dựa trên xLSTM |
| Kronos [@shi2026kronos] | AAAI 2026 | Mô hình nền tảng cho dữ liệu nến giá của 45 sàn |

Đáng chú ý, minLSTM và minGRU cho thấy chỉ cần bỏ sự phụ thuộc của các cổng vào trạng thái cũ hₜ₋₁, LSTM và GRU có thể huấn luyện song song bằng thuật toán quét song song (parallel scan) mà vẫn cạnh tranh với các kiến trúc hiện đại [@feng2024minrnn]. Đồng thời, TiRex – một bộ dự báo xây trên xLSTM – cho thấy ý tưởng bộ nhớ có cổng của LSTM năm 1997 vẫn là nền tảng của các mô hình dự báo hàng đầu năm 2025 [@auer2025tirex]. Với tài chính, Kronos mã hóa mỗi cây nến (mở, cao, thấp, đóng, khối lượng) thành token và tiền huấn luyện trên hơn 12 tỉ bản ghi nến, mở ra khả năng dự báo "zero-shot" cho cả những thị trường ít dữ liệu như Việt Nam [@shi2026kronos]. Một lưu ý khi đánh giá các mô hình nền tảng: nếu dữ liệu tiền huấn luyện đã chứa giai đoạn được dùng để kiểm tra, kết quả sẽ lạc quan giả tạo. Kiểm tra trên giai đoạn sau ngày phát hành mô hình là cách đánh giá công bằng nhất.

## 4.11. Tiểu kết chương

Chương 4 đã trình bày mạng hồi quy từ SimpleRNN đến LSTM, GRU, cùng hai vấn đề cốt lõi của chúng: tiêu biến và bùng nổ gradient. Thực nghiệm khẳng định ba điều. Thứ nhất, LSTM tự viết bằng NumPy, kể cả phần BPTT, cho kết quả trùng với Keras và PyTorch. Thứ hai, trên cùng bài toán, mạng hồi quy đọc chuỗi lợi suất nhỉnh hơn và ổn định hơn CNN đọc ảnh, nhưng chỉ ở VN-Index mới có khả năng dự báo nhỏ, có ý nghĩa thống kê, phần lớn đến từ quán tính ngắn hạn của chỉ số. Thứ ba, khả năng dự báo đó có thật nhưng nhỏ: trước phí, chiến lược theo GRU thắng mua và giữ ở VN-Index; sau phí giao dịch thì thua, chưa kể các ràng buộc thực tế như chu kỳ thanh toán T+2. Các hướng mới như xLSTM, Mamba, minGRU hay mô hình nền tảng Kronos cho thấy ý tưởng "bộ nhớ có cổng" vẫn đang tiếp tục phát triển, và là đối tượng đánh giá phù hợp cho các nghiên cứu tiếp theo.
