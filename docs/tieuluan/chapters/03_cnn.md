# CHƯƠNG 3. MẠNG NƠ-RON TÍCH CHẬP (CNN)

## 3.1. Vì sao cần mạng tích chập?

MLP ở Chương 2 nối **mọi** đầu vào với **mọi** nơ-ron. Với dữ liệu bảng vài chục cột, điều đó không thành vấn đề. Nhưng một ảnh xám nhỏ 20 × 20 đã có 400 điểm ảnh; nối thẳng tới 64 nơ-ron ẩn cần 400 × 64 + 64 = 25.664 tham số. Với ảnh màu 224 × 224, con số đó vượt chín triệu cho riêng lớp đầu. Tệ hơn, MLP coi một nét gạch ở góc trái và cùng nét gạch đó ở giữa ảnh là hai thứ hoàn toàn khác nhau, nên phải học lại từ đầu ở mỗi vị trí.

Mạng nơ-ron tích chập (convolutional neural network – CNN) giải quyết vấn đề bằng hai ý tưởng lấy cảm hứng từ thị giác sinh học. Hubel và Wiesel phát hiện mỗi tế bào ở vỏ não thị giác chỉ phản ứng với một vùng nhỏ của trường nhìn [@hubel1962]; Fukushima hiện thực hóa ý tưởng này trong mạng Neocognitron [@fukushima1980]; LeCun và cộng sự kết hợp nó với lan truyền ngược thành LeNet [@lecun1998]. Ý tưởng thứ nhất là **kết nối cục bộ**: mỗi nơ-ron chỉ nhìn một vùng nhỏ của ảnh. Ý tưởng thứ hai là **dùng chung trọng số**: cùng một bộ lọc nhỏ được trượt qua mọi vị trí, nên một mẫu hình học được ở một chỗ sẽ được nhận ra ở mọi chỗ khác.

> **Hiểu nhanh.** CNN giống một người cầm kính lúp soi từng ô nhỏ của bức ảnh để tìm một chi tiết quen thuộc, chẳng hạn một cạnh thẳng hay một góc nhọn. Mỗi "chiếc kính lúp" (bộ lọc) chuyên tìm một loại chi tiết. Các lớp đầu tìm chi tiết đơn giản; các lớp sau ghép chúng thành hình phức tạp hơn, như nét ghép thành chữ và chữ ghép thành từ.

## 3.2. Phép tích chập

### 3.2.1. Bộ lọc trượt trên ảnh

Một **bộ lọc** (kernel) là một bảng trọng số nhỏ, ví dụ 3 × 3. Đặt bộ lọc lên một vùng của ảnh, nhân từng cặp số tương ứng rồi cộng lại (cộng thêm hệ số chặn b) ta được một số. Trượt bộ lọc sang vị trí kế tiếp và lặp lại, ta thu được một **bản đồ đặc trưng** (feature map). Với ảnh nhiều kênh (ảnh màu có 3 kênh), bộ lọc có độ sâu bằng số kênh và tổng được lấy trên cả các kênh:

$$Y[i,j] = \sum_{c}\sum_{u}\sum_{v} X[c,\,i+u,\,j+v]\;K[c,u,v] + b$$ (3.1)

Hình 3.1 minh họa phép tính trên ảnh 3 × 3 với bộ lọc 2 × 2. Ô đầu tiên của đầu ra bằng 1·1 + 2·0 + 0·0 + 1·(−1) = 0; trượt sang phải ta được 2·1 + 0·0 + 1·0 + 3·(−1) = −1; tương tự hai ô hàng dưới là −1 và 1. Bộ lọc này cho giá trị lớn khi điểm ảnh trên-trái sáng hơn điểm ảnh dưới-phải, tức nó là một "máy dò" cạnh chéo đơn giản. Trong CNN, các trọng số bộ lọc không do con người chọn mà được **học** từ dữ liệu. Về mặt toán học, phép tính trong các thư viện là tương quan chéo (không lật bộ lọc), nhưng theo thông lệ vẫn được gọi là tích chập [@goodfellow2016].

![Hình 3.1. Ví dụ phép tích chập: bộ lọc 2 × 2 trượt trên ảnh 3 × 3, sau đó đi qua hàm ReLU.](../../figures/tieuluan/diagram_conv_example.png)

### 3.2.2. Số tham số và tính dùng chung trọng số

Một lớp tích chập có C_in kênh vào, C_out bộ lọc kích thước K × K thì có (K·K·C_in + 1)·C_out tham số. Ví dụ, 8 bộ lọc 3 × 3 trên ảnh một kênh chỉ cần (9 + 1)·8 = 80 tham số, bất kể ảnh lớn đến đâu. So với 25.664 tham số của MLP ở mục 3.1, mức tiết kiệm là hơn 300 lần. Đây chính là lợi ích của việc dùng chung trọng số.

### 3.2.3. Bước trượt, phần đệm và kích thước đầu ra

**Bước trượt** S (stride) là số ô bộ lọc dịch sau mỗi lần tính; S = 2 làm đầu ra nhỏ đi khoảng một nửa. **Phần đệm** P (padding) là số hàng/cột số 0 thêm quanh viền để giữ kích thước hoặc để các điểm ở mép được xét đủ. Với ảnh cạnh N, bộ lọc cạnh K, kích thước đầu ra là [@dumoulin2016]:

$$O = \left\lfloor \frac{N + 2P - K}{S} \right\rfloor + 1$$ (3.2)

Bảng 3.1 áp dụng công thức này cho ảnh 20 × 20 dùng trong thực nghiệm.

Bảng 3.1. Kích thước đầu ra của một kênh với ảnh vào 20 × 20 (tính theo công thức 3.2).
| Bộ lọc K | Bước S | Đệm P | Đầu ra | Ghi chú |
|---|---|---|---|---|
| 3 × 3 | 1 | 0 | 18 × 18 | Cấu hình "valid" dùng trong thực nghiệm |
| 3 × 3 | 1 | 1 | 20 × 20 | Cấu hình "same": giữ nguyên kích thước |
| 3 × 3 | 2 | 0 | 9 × 9 | Bước 2 giảm kích thước khoảng một nửa |
| 5 × 5 | 1 | 0 | 16 × 16 | Bộ lọc lớn hơn nhìn vùng rộng hơn |

### 3.2.4. Vùng tiếp nhận

**Vùng tiếp nhận** (receptive field) của một nơ-ron là phần ảnh đầu vào có thể ảnh hưởng tới nó. Một lớp 3 × 3 nhìn vùng 3 × 3; hai lớp 3 × 3 chồng lên nhau nhìn vùng 5 × 5; ba lớp nhìn vùng 7 × 7. Nhờ vậy, mạng sâu ghép các mẫu nhỏ thành cấu trúc lớn mà mỗi bộ lọc vẫn nhỏ và ít tham số. Đây là ý tưởng cốt lõi của các kiến trúc như VGG [@simonyan2015].

## 3.3. Hàm kích hoạt, phép gộp và lớp phân loại

Sau mỗi lớp tích chập, giá trị đi qua hàm kích hoạt phi tuyến, phổ biến nhất là ReLU, giữ số dương và đưa số âm về 0 (ô cuối Hình 3.1). Tiếp theo thường là lớp **gộp** (pooling) để giảm kích thước và giữ lại phản ứng nổi bật. Gộp cực đại (max pooling) 2 × 2 lấy giá trị lớn nhất trong mỗi ô 2 × 2; với vùng [[1, 3], [2, 0]], gộp cực đại cho 3, còn gộp trung bình cho 1,5. Phép gộp không có tham số nhưng giúp mạng ít nhạy với dịch chuyển nhỏ của mẫu hình. Cuối cùng, các bản đồ đặc trưng được **duỗi phẳng** (flatten) thành một vector rồi đưa qua lớp kết nối đầy đủ và hàm sigmoid để ra xác suất, giống phần cuối của MLP ở Chương 2.

## 3.4. Huấn luyện CNN và kỹ thuật im2col

CNN được huấn luyện bằng lan truyền ngược như MLP, với một điểm đặc biệt: vì mỗi trọng số của bộ lọc được dùng ở **mọi** vị trí, gradient của nó là **tổng** đóng góp từ mọi vị trí. Ví dụ, nếu trọng số w nhân với các điểm ảnh 2, 1 và −1 ở ba vị trí, và gradient từ lớp sau tại ba vị trí đều bằng 1, thì ∂L/∂w = 2 + 1 − 1 = 2; với tốc độ học 0,01, w = 0,50 được cập nhật thành 0,48. Qua lớp gộp cực đại, gradient chỉ đi về đúng vị trí đã đạt cực đại; qua ReLU, gradient bị chặn ở những nơi đầu vào âm.

Cài đặt tích chập bằng bốn vòng lặp lồng nhau trong Python rất chậm. Bản NumPy của tiểu luận dùng kỹ thuật **im2col**: mỗi vùng ảnh mà bộ lọc phủ lên được "duỗi" thành một hàng của một ma trận lớn, nhờ đó cả phép tích chập trở thành **một phép nhân ma trận** – thao tác mà NumPy thực hiện rất nhanh. Lan truyền ngược làm điều ngược lại ("col2im"): cộng gradient của từng vị trí trong bộ lọc trở về đúng điểm ảnh gốc. Đoạn mã dưới trích từ lớp `Conv2d` tự viết:

```python
# windows: mọi vùng KH×KW của ảnh, lấy bằng "cửa sổ trượt" không sao chép dữ liệu
windows = as_strided(xp, shape=(n, oh, ow, c, kh, kw), strides=(...))
cols = windows.reshape(n * oh * ow, c * kh * kw)          # im2col: mỗi vùng là một hàng
out = cols @ self.weight.value.reshape(f, -1).T + self.bias.value
# lan truyền ngược: gradient trọng số cũng chỉ là một phép nhân ma trận
self.weight.grad += (g.T @ cols).reshape(self.weight.value.shape)
```

Kiểm tra gradient (công thức 2.9) trên mạng tích chập tự viết cho sai số tương đối lớn nhất {{v:gc.cnn}}, khẳng định phần lan truyền ngược được cài đặt đúng.

## 3.5. Các kiến trúc CNN tiêu biểu và xu hướng mới

Bảng 3.2 tóm tắt những bước tiến chính. LeNet-5 đặt nền móng cho mẫu "tích chập – gộp – kết nối đầy đủ" [@lecun1998]. AlexNet chứng minh sức mạnh của mạng sâu khi có GPU và dữ liệu lớn [@krizhevsky2012]. VGG cho thấy chồng nhiều bộ lọc 3 × 3 là đủ [@simonyan2015]. GoogLeNet ghép nhiều kích thước bộ lọc song song [@szegedy2015]. ResNet thêm **kết nối tắt** (đầu ra = đầu vào + phần dư) để huấn luyện được mạng hàng trăm lớp [@he2016]. EfficientNet tìm cách tăng đồng thời chiều sâu, chiều rộng và độ phân giải [@tan2019].

Năm 2021, Vision Transformer (ViT) cho thấy kiến trúc Transformer áp lên các mảnh ảnh có thể vượt CNN khi được huấn luyện trên dữ liệu rất lớn [@dosovitskiy2021], làm dấy lên câu hỏi CNN còn cần thiết không. Câu trả lời của cộng đồng là "có": ConvNeXt (2022) hiện đại hóa ResNet theo các lựa chọn thiết kế của Transformer và đạt kết quả ngang bằng [@liu2022convnext]; ConvNeXt V2 (2023) bổ sung tiền huấn luyện tự giám sát [@woo2023convnextv2]. Đến năm 2025, MambaOut (CVPR 2025) chỉ ra rằng với bài toán phân loại ảnh, có thể bỏ hẳn khối Mamba (mô hình không gian trạng thái) khỏi các kiến trúc thị giác dựa trên Mamba mà kết quả còn tốt hơn; mô hình thu được chủ yếu dựa trên tích chập [@yu2025mambaout]. CNN cũng được dùng cho chuỗi thời gian: mạng tích chập thời gian (TCN) cạnh tranh tốt với mạng hồi quy [@bai2018tcn], còn TimesNet (ICLR 2023) biến chuỗi một chiều thành ảnh hai chiều theo chu kỳ rồi xử lý bằng tích chập 2D [@wu2023timesnet]. Ý tưởng "biến chuỗi thành ảnh" này cũng chính là cách tiếp cận của thực nghiệm trong chương.

Bảng 3.2. Một số kiến trúc CNN tiêu biểu.
| Năm | Kiến trúc | Ý tưởng chính | Bài học |
|---|---|---|---|
| 1998 | LeNet-5 [@lecun1998] | Tích chập – gộp – kết nối đầy đủ | Khuôn mẫu chung của CNN |
| 2012 | AlexNet [@krizhevsky2012] | Mạng sâu, ReLU, huấn luyện trên GPU | Dữ liệu lớn và GPU tạo đột phá |
| 2015 | VGG [@simonyan2015] | Chồng nhiều bộ lọc 3 × 3 | Bộ lọc nhỏ, mạng sâu |
| 2016 | ResNet [@he2016] | Kết nối tắt học phần dư | Huấn luyện được hàng trăm lớp |
| 2019 | EfficientNet [@tan2019] | Mở rộng cân đối sâu–rộng–phân giải | Hiệu quả trên mỗi tham số |
| 2022 | ConvNeXt [@liu2022convnext] | CNN thiết kế lại theo Transformer | CNN vẫn cạnh tranh với ViT |
| 2025 | MambaOut [@yu2025mambaout] | Bỏ khối Mamba, giữ tích chập | Tích chập vẫn rất mạnh cho ảnh |

## 3.6. CNN trong tài chính: bài học từ "(Re-)Imag(in)ing Price Trends"

Nghiên cứu nổi bật nhất về CNN trong đầu tư là công trình của Jiang, Kelly và Xiu đăng trên *The Journal of Finance* năm 2023 [@jiang2023]. Thay vì để nhà nghiên cứu tự định nghĩa tín hiệu kỹ thuật (đường trung bình, động lượng…), các tác giả vẽ lịch sử giá của từng cổ phiếu thành **ảnh đen trắng** giống biểu đồ mà nhà phân tích kỹ thuật vẫn xem, rồi để CNN tự học mẫu hình nào báo hiệu giá sắp tăng. Bảng 3.3 tóm tắt thiết kế và kết quả chính.

Bảng 3.3. Tóm tắt nghiên cứu của Jiang, Kelly và Xiu (2023).
| Thành phần | Thiết kế / kết quả |
|---|---|
| Dữ liệu | Toàn bộ cổ phiếu NYSE, AMEX, NASDAQ giai đoạn 1993–2019 |
| Ảnh đầu vào | Thanh giá mở – cao – thấp – đóng (OHLC), mỗi ngày rộng 3 điểm ảnh, kèm đường trung bình động và khối lượng; ảnh 5, 20, 60 ngày có kích thước 32 × 15, 64 × 60, 96 × 180 điểm ảnh |
| Nhãn | 1 nếu lợi suất 5, 20 hoặc 60 ngày tiếp theo dương |
| Chia dữ liệu | Huấn luyện và xác thực trên 1993–2000; kiểm tra ngoài mẫu 2001–2019 |
| Mạng CNN | Khối tích chập 5 × 3 → chuẩn hóa theo lô → LeakyReLU → gộp 2 × 1; 64 bộ lọc ở khối đầu, nhân đôi qua mỗi khối; mạng cho ảnh 20 ngày có 708.866 tham số |
| Kết quả | Danh mục mua nhóm 10% cổ phiếu có xác suất tăng cao nhất, bán nhóm 10% thấp nhất theo tuần, trọng số đều: Sharpe năm 7,15 trước chi phí; khoảng 4,0 sau chi phí 10–20 điểm cơ bản |
| So sánh | Vượt các tín hiệu động lượng, đảo chiều; không quy tắc nào trong 7.846 quy tắc kỹ thuật vượt được CNN ở kỳ hạn tuần |

Ba phát hiện của nghiên cứu này đặc biệt có giá trị khi áp dụng vào thị trường như Việt Nam. Thứ nhất, mô hình học trên dữ liệu Mỹ khi **chuyển giao** sang 26 thị trường quốc tế cho kết quả tốt hơn mô hình học tại chỗ ở 21 thị trường, nhưng kém hơn ở Trung Quốc, Ấn Độ và Hàn Quốc. Với các thị trường mới nổi lớn, việc học trên dữ liệu địa phương vẫn quan trọng. Thứ hai, các tác giả cho thấy yếu tố tạo khác biệt chủ yếu là **cách chuẩn hóa** dữ liệu khi vẽ ảnh: một CNN một chiều nhận chuỗi số được chuẩn hóa giống ảnh cho kết quả ngang CNN hai chiều. Nói cách khác, biểu diễn đầu vào quan trọng không kém kiến trúc. Thứ ba, chính các tác giả thừa nhận một mô hình chuỗi thời gian được thiết kế tốt như LSTM có thể vượt CNN, nên kết quả của họ chỉ là "cận dưới" của những gì học máy có thể khai thác [@jiang2023]. Nhận định này dẫn thẳng tới Chương 4.

Một hướng khác là biến chuỗi giá thành ảnh bằng phép biến đổi toán học thay vì vẽ biểu đồ. Wang và Oates đề xuất **trường góc Gram** (Gramian Angular Field – GAF) [@wang2015gaf]; Sezer và Ozbayoglu ghép 15 chỉ báo kỹ thuật trong 15 ngày thành ảnh 15 × 15 để CNN đưa ra tín hiệu mua – bán – giữ [@sezer2018]. Thực nghiệm của tiểu luận theo hướng GAF.

## 3.7. Dữ liệu thực nghiệm: ảnh GASF của ba chỉ số

### 3.7.1. Bài toán và lý do chọn GASF

Từ ba chuỗi chỉ số ở mục 1.8, mỗi mẫu là một **cửa sổ 20 phiên** kết thúc tại ngày t; nhãn y = 1 nếu giá đóng cửa phiên kế tiếp tăng (C(t+1) > C(t)), ngược lại y = 0. Đây là bài toán dự báo hướng đi ngắn hạn mà nhiều nhà đầu tư cá nhân quan tâm. Ảnh chỉ dùng **giá đóng cửa** vì hai lý do. Một là dữ liệu mở – cao – thấp của VN-Index trong lịch sử không đáng tin (mục 1.8), nên vẽ biểu đồ OHLC kiểu Jiang, Kelly và Xiu sẽ tạo ra thông tin giả. Hai là dùng cùng một quy trình cho cả ba thị trường giúp phép so sánh công bằng.

### 3.7.2. Cách tạo ảnh GASF

Cửa sổ giá x = (x₁, …, x₂₀) được chuẩn hóa min–max trong chính cửa sổ, đổi thành góc, rồi tạo ma trận 20 × 20 theo dạng tổng góc (GASF):

$$\tilde{x}_t = \frac{x_t - \min(\mathbf{x})}{\max(\mathbf{x}) - \min(\mathbf{x})}, \qquad \phi_t = \arccos(\tilde{x}_t)$$ (3.3)

$$G_{ij} = \cos(\phi_i + \phi_j) = \tilde{x}_i\tilde{x}_j - \sqrt{1-\tilde{x}_i^{2}}\,\sqrt{1-\tilde{x}_j^{2}}$$ (3.4)

Ví dụ với ba mức giá 100, 105, 110: chuẩn hóa được 0; 0,5; 1, tương ứng các góc 90°, 60°, 0°. Ô G₁₁ = cos(180°) = −1, ô G₃₃ = cos(0°) = 1, ô G₁₃ = cos(90°) = 0. Ảnh mã hóa quan hệ giữa **từng cặp thời điểm** trong cửa sổ: điểm ảnh hàng i, cột j cho biết giá ở hai thời điểm i và j cùng cao, cùng thấp hay trái chiều. Do chỉ dùng min và max của chính cửa sổ đến ngày t, phép chuẩn hóa không nhìn thấy tương lai.

**Một lựa chọn thiết kế quan trọng.** Wang và Oates cho phép chuẩn hóa về [−1, 1] hoặc [0, 1] [@wang2015gaf]. Phân tích công thức (3.4) cho thấy với khoảng [−1, 1], khi đổi dấu toàn bộ chuỗi chuẩn hóa (x̃ → −x̃) thì φ → π − φ, nên cos(φᵢ + φⱼ) không đổi. Hệ quả là một cửa sổ **tăng đều** và cửa sổ **giảm đều** đối xứng với nó cho ra **cùng một ảnh**: ảnh đánh mất đúng thông tin về chiều giá mà bài toán cần dự báo. Với khoảng [0, 1], góc chỉ nằm trong [0°, 90°], phép đổi sang góc là đơn ánh và hiện tượng trên không xảy ra. Tiểu luận vì vậy chọn [0, 1], và bộ kiểm thử tự động có một phép thử khẳng định ảnh của chuỗi tăng khác ảnh của chuỗi giảm. Hình 3.3 cho thấy ảnh GASF thật của ba thị trường: ô màu đỏ ứng với hai phiên cùng ở gần mức giá cao nhất của cửa sổ, ô xanh đậm ứng với hai phiên cùng ở gần mức thấp nhất, ô nhạt ứng với một phiên cao và một phiên thấp.

![Hình 3.2. Quy trình của mô hình CNN4: cửa sổ 20 giá → ảnh GASF → tích chập → gộp → xác suất tăng.](../../figures/tieuluan/diagram_cnn_pipeline.png)

![Hình 3.3. Cửa sổ 20 phiên đầu tiên của tập test ở mỗi thị trường (trên) và ảnh GASF tương ứng (dưới).](../../figures/tieuluan/ch3_gasf_examples.png)

### 3.7.3. Kích thước và phân bố dữ liệu

Dữ liệu được chia **theo thời gian**: train đến hết 2019, validation 2020–2022 (giai đoạn có cú sốc COVID-19 và đợt tăng lãi suất 2022), test từ 01/2023 đến 30/09/2026. Mẫu nào có ngày nhãn rơi sang tập sau thì bị loại khỏi tập trước, để nhãn không bao giờ "vượt rào". Bảng 3.4 cho thấy tỉ lệ nhãn "tăng" dao động quanh 50–58%; tỉ lệ này trên tập train được dùng làm đường cơ sở "luôn đoán lớp đa số".

{{t:ch3_splits:3.4}}

## 3.8. Cài đặt bằng ba cách

Mô hình đại diện **CNN4** gồm một lớp tích chập 4 bộ lọc 3 × 3 (không đệm, bước 1) → ReLU → gộp cực đại 2 × 2 → duỗi phẳng 4 × 9 × 9 = 324 số → lớp kết nối đầy đủ → logit, tổng cộng {{v:params.sp500.cnn4.pytorch}} tham số (Hình 3.2). Mạng được giữ nhỏ có chủ đích: mỗi thị trường chỉ có vài nghìn ảnh huấn luyện, tín hiệu rất yếu, và bản NumPy phải huấn luyện được trên CPU. CNN4 được cài đặt bằng NumPy, Keras và PyTorch với cùng giao thức như Chương 2: cùng trọng số khởi tạo, cùng thứ tự mini-batch, cùng Adam và dừng sớm. Một chi tiết kỹ thuật là Keras mặc định xếp kênh ở cuối (cao × rộng × kênh) còn PyTorch và bản NumPy xếp kênh ở đầu. Bản Keras vì thế đổi trục trước và sau khối tích chập để phép duỗi phẳng cho cùng thứ tự với hai bản kia; nếu bỏ qua, trọng số chép sang sẽ bị đặt sai vị trí.

```python
# PyTorch: ảnh vào dạng [batch, kênh, cao, rộng]
model = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Flatten(), nn.Linear(4 * 9 * 9, 1))
# Keras: đổi trục sang [cao, rộng, kênh] rồi đổi lại trước khi duỗi phẳng
z = keras.layers.Permute((2, 3, 1))(inputs)
z = keras.layers.MaxPooling2D(2)(keras.layers.Conv2D(4, 3, activation="relu")(z))
z = keras.layers.Flatten()(keras.layers.Permute((3, 1, 2))(z))
outputs = keras.layers.Dense(1)(z)
```

Để so sánh kiến trúc, tiểu luận huấn luyện thêm (bằng PyTorch) **CNN8** (8 bộ lọc) và **CNN sâu** (hai khối tích chập 4 và 8 bộ lọc), cùng hai đường cơ sở không cần học: luôn đoán lớp đa số của tập train, và "lặp lại hướng phiên trước" (dự báo tăng nếu phiên gần nhất tăng).

## 3.9. Kết quả và thảo luận

**Ba bản cài đặt.** Bảng 3.5 và Hình 3.4 cho thấy ba bản cài đặt CNN4 gần như trùng khớp: cùng epoch tốt nhất với mọi hạt giống, ROC-AUC theo từng hạt giống chênh nhau {{v:aucdiff.cnn4.keras}}. Trên tập test, xác suất dự báo của bản NumPy và PyTorch lệch nhau tối đa {{v:parity_trained.vnindex.cnn4.pytorch}} (VN-Index). Đây là bằng chứng mạnh cho thấy lớp tích chập, gộp và lan truyền ngược tự viết là đúng.

{{t:ch3_frameworks:3.5}}

![Hình 3.4. BCE trên tập validation theo epoch của ba bản cài đặt CNN4 (hạt giống 11).](../../figures/tieuluan/ch3_learning_curves.png)

**Khả năng dự báo.** Bảng 3.6 và Hình 3.5 trả lời câu hỏi chính của chương. Mỗi đoạn ngang trong Hình 3.5 là khoảng tin cậy 95% của ROC-AUC, ước lượng bằng **bootstrap theo khối**: lấy mẫu lại từng khối 20 phiên liên tiếp để giữ tính phụ thuộc giữa các ngày gần nhau, thay vì coi mỗi ngày là độc lập [@kunsch1989]. Nếu đoạn ngang cắt qua đường 0,5 thì không thể khẳng định mô hình tốt hơn đoán ngẫu nhiên.

{{t:ch3_architectures:3.6}}

![Hình 3.5. ROC-AUC trên tập test và khoảng tin cậy 95% (bootstrap theo khối 20 phiên) của các CNN và đường cơ sở "lặp lại hướng phiên trước".](../../figures/tieuluan/ch3_auc_ci.png)

**Đọc kết quả.** Ở S&P 500 và Bitcoin, khoảng tin cậy của cả ba CNN đều chứa 0,5, chẳng hạn CNN4 đạt {{v:ci.sp500.cnn4}} trên S&P 500 và {{v:ci.btc.cnn4}} trên Bitcoin: không có bằng chứng CNN dự báo hướng đi của hai thị trường này tốt hơn tung đồng xu. Ở VN-Index, cả ba CNN đều nhỉnh hơn đường cơ sở "lặp lại hướng phiên trước" ({{v:ci.vnindex.persistence}}), nhưng chỉ CNN8 có khoảng tin cậy nằm hẳn trên 0,5 ({{v:ci.vnindex.cnn8}}); CNN4 ({{v:ci.vnindex.cnn4}}) và CNN sâu ({{v:ci.vnindex.cnndeep}}) ở sát ranh giới. Kết quả của CNN8 cần được đọc thận trọng: khi kiểm định nhiều mô hình cùng lúc, một khoảng tin cậy nằm trên 0,5 vẫn có thể xuất hiện do may mắn, nên đây chỉ là tín hiệu yếu. Thêm bộ lọc hay thêm lớp cũng không cải thiện một cách nhất quán: CNN sâu kém CNN4 ở cả S&P 500 lẫn Bitcoin. Epoch tốt nhất của CNN4 phần lớn chỉ từ 1 đến 6 (Bảng 3.5), tức là chỉ sau vài lượt mạng đã bắt đầu học thuộc nhiễu của tập train và cơ chế dừng sớm phải dừng lại.

**Phép thử về cách chuẩn hóa GASF.** Để kiểm tra lựa chọn [0, 1] ở mục 3.7.2, CNN4 được huấn luyện lại trên ảnh GASF chuẩn hóa về [−1, 1], giữ nguyên mọi thiết lập khác (PyTorch, 3 hạt giống). ROC-AUC trung bình lần lượt là {{v:abl.sp500}}, {{v:abl.vnindex}} và {{v:abl.btc}} với S&P 500, VN-Index và Bitcoin, so với {{v:auc.sp500.cnn4.pytorch}}, {{v:auc.vnindex.cnn4.pytorch}} và {{v:auc.btc.cnn4.pytorch}} khi dùng [0, 1]; các khoảng tin cậy chồng lấn nhau hoàn toàn. Như vậy, dù về lý thuyết [−1, 1] làm mất thông tin chiều giá, trong thực nghiệm này khác biệt không đo được. Đây thêm một dấu hiệu cho thấy ảnh GASF của giá đóng cửa chứa rất ít thông tin về hướng đi của phiên kế tiếp. Tiểu luận vẫn giữ [0, 1] vì đúng về nguyên tắc.

Có ba lý do khiến kết quả khiêm tốn hơn nhiều so với Jiang, Kelly và Xiu. Một là **quy mô dữ liệu**: họ có hàng triệu ảnh từ hàng nghìn cổ phiếu, còn mỗi thị trường ở đây chỉ có vài nghìn ảnh của một chỉ số. Hai là **bản chất bài toán**: họ xếp hạng cổ phiếu với nhau trong cùng một tuần, nên phần biến động chung của thị trường bị triệt tiêu; ở đây phải dự báo chính hướng đi của cả thị trường – thứ chịu ảnh hưởng của tin tức vĩ mô vốn không có trong ảnh. Ba là **kỳ hạn một ngày** rất ngắn và nhiều nhiễu. Kết quả này phù hợp với giả thuyết thị trường hiệu quả ở dạng yếu đối với các thị trường lớn [@fama1970]. Khả năng dự báo nhỉnh hơn ở VN-Index cũng hợp lý: thị trường nhỏ hơn, có biên độ giá trần – sàn và tỉ trọng nhà đầu tư cá nhân cao, nên thông tin có thể được phản ánh vào giá chậm hơn.

## 3.10. Tiểu kết chương

Chương 3 đã trình bày nguyên lý của CNN – kết nối cục bộ, dùng chung trọng số, phép gộp – cùng các kiến trúc từ LeNet đến ConvNeXt và MambaOut, và nghiên cứu tiêu biểu về CNN trong đầu tư của Jiang, Kelly và Xiu. Về thực nghiệm, CNN tự viết bằng NumPy (với kỹ thuật im2col) cho kết quả trùng với Keras và PyTorch. Khi đọc ảnh GASF của chuỗi giá, CNN không dự báo được hướng đi của S&P 500 và Bitcoin tốt hơn đoán ngẫu nhiên, và chỉ cho tín hiệu yếu ở VN-Index. Chương cũng chỉ ra một điểm dễ bị bỏ qua về biểu diễn dữ liệu: cách chuẩn hóa [−1, 1] phổ biến của GASF làm mất thông tin về chiều giá, dù trong thực nghiệm này khác biệt về ROC-AUC không đáng kể. Chương 4 sẽ tiếp cận cùng dữ liệu theo cách tự nhiên hơn: đọc chuỗi lợi suất theo đúng thứ tự thời gian bằng mạng hồi quy.
