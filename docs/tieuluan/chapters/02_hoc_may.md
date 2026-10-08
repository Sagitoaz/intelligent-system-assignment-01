# CHƯƠNG 2. CÁC KỸ THUẬT HỌC MÁY CƠ BẢN

## 2.1. Học máy là gì?

Theo định nghĩa kinh điển của Mitchell, một chương trình được gọi là **học** từ kinh nghiệm E đối với nhiệm vụ T và thước đo P nếu hiệu quả của nó trên T, đo bằng P, tăng lên nhờ E [@mitchell1997]. Áp vào bài toán của chương này: nhiệm vụ T là dự báo doanh nghiệp nào sắp phá sản; kinh nghiệm E là hàng nghìn hồ sơ doanh nghiệp cũ đã biết kết quả; thước đo P là khả năng xếp đúng doanh nghiệp rủi ro lên trên doanh nghiệp an toàn. Điều cốt lõi là máy không được "dạy" quy tắc nào cụ thể: nó tự rút ra quy luật từ ví dụ.

Có ba kiểu học chính (Bảng 2.1). **Học có giám sát** dùng dữ liệu có đáp án; nếu đáp án là một con số (giá nhà, tổn thất) thì đó là bài toán **hồi quy**, nếu đáp án là một nhóm (vỡ nợ / không vỡ nợ) thì đó là bài toán **phân loại**. **Học không giám sát** chỉ có dữ liệu mà không có đáp án; máy tự tìm cấu trúc như các nhóm khách hàng giống nhau. **Học tăng cường** cho một tác tử thử – sai trong môi trường và học từ phần thưởng, ví dụ một tác tử giao dịch được "thưởng" khi danh mục tăng giá trị.

Bảng 2.1. Ba kiểu học máy và ví dụ trong tài chính – đầu tư.
| Kiểu học | Dữ liệu huấn luyện | Ví dụ tài chính | Đầu ra |
|---|---|---|---|
| Có giám sát – phân loại | Hồ sơ kèm nhãn | Khách hàng có vỡ nợ tháng sau không? | Xác suất và nhãn |
| Có giám sát – hồi quy | Hồ sơ kèm giá trị số | Mức tổn thất của khoản vay là bao nhiêu? | Một con số |
| Không giám sát | Chỉ có đặc trưng | Chia nhà đầu tư thành các nhóm hành vi | Mã nhóm |
| Tăng cường | Tương tác và phần thưởng | Tác tử học cách phân bổ danh mục | Hành động |

> **Hiểu nhanh.** Học có giám sát giống học sinh luyện đề có đáp án ở cuối sách: làm bài, so với đáp án, sửa cách làm. Học không giám sát giống việc tự xếp một chồng ảnh thành các nhóm "trông giống nhau" mà không ai cho biết tên nhóm. Học tăng cường giống tập đi xe đạp: không ai đưa đáp án, chỉ có cảm giác "ngã" hay "giữ được thăng bằng".

## 2.2. Quy trình xây dựng một mô hình học máy

Một dự án học máy thường đi qua sáu bước: (1) xác định bài toán và nhãn; (2) thu thập và làm sạch dữ liệu; (3) chia dữ liệu thành tập huấn luyện (train), tập xác thực (validation) và tập kiểm tra (test); (4) tiền xử lý như điền giá trị thiếu và chuẩn hóa; (5) huấn luyện và chọn cấu hình; (6) đánh giá lần cuối rồi triển khai. Ba tập dữ liệu có ba vai trò tách bạch: train để mô hình học tham số, validation để người làm chọn cấu hình (số vòng học, ngưỡng quyết định), còn test chỉ được mở **một lần** ở cuối để ước lượng hiệu quả trên dữ liệu mới.

Mục tiêu thật sự của học máy là **tổng quát hóa**: dự báo đúng trên dữ liệu chưa từng thấy. Mô hình quá đơn giản sẽ **thiếu khớp** (underfitting), không nắm được quy luật ngay cả trên dữ liệu học. Mô hình quá phức tạp có thể **quá khớp** (overfitting), tức học thuộc cả nhiễu ngẫu nhiên của tập train nên dự báo kém trên dữ liệu mới. Dấu hiệu dễ nhận ra của quá khớp là sai số trên train tiếp tục giảm trong khi sai số trên validation bắt đầu tăng.

> **Hiểu nhanh.** Quá khớp giống học sinh "học vẹt": thuộc lòng đáp án của đúng các đề đã luyện nên điểm luyện tập rất cao, nhưng gặp đề mới thì lúng túng. Tập test chính là "đề thi thật", và đề thi thật chỉ được làm một lần.

Đặc biệt nguy hiểm trong tài chính là **rò rỉ dữ liệu** (data leakage): mô hình vô tình được nhìn thấy thông tin mà khi dự báo thật sẽ không có. Ba dạng thường gặp là: dùng biến chỉ biết sau sự kiện (như tình trạng thu hồi nợ để dự báo vỡ nợ); chuẩn hóa dữ liệu bằng trung bình tính trên toàn bộ dữ liệu, kể cả tập test; và trộn ngẫu nhiên dữ liệu chuỗi thời gian khiến tương lai lọt vào tập train [@lopezdeprado2018]. Toàn bộ thực nghiệm của tiểu luận được thiết kế để loại trừ cả ba dạng này.

## 2.3. Hồi quy tuyến tính, hàm mất mát và hạ gradient

Mô hình đơn giản nhất là **hồi quy tuyến tính**: dự báo là tổng có trọng số của các đặc trưng cộng với một hệ số chặn [@hastie2009].

$$\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_dx_d + b = \mathbf{w}^{\top}\mathbf{x} + b$$ (2.1)

Để biết bộ trọng số nào "tốt", ta cần một **hàm mất mát** đo mức sai của dự báo. Với hồi quy, lựa chọn phổ biến là sai số bình phương trung bình (MSE):

$$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i\right)^2$$ (2.2)

Huấn luyện chính là tìm trọng số làm hàm mất mát nhỏ nhất. Thuật toán nền tảng là **hạ gradient** (gradient descent): gradient ∇L chỉ hướng làm mất mát tăng nhanh nhất, nên ta bước một bước nhỏ theo hướng ngược lại. Độ dài bước η gọi là **tốc độ học**.

$$\theta \leftarrow \theta - \eta\,\nabla_{\theta}L(\theta)$$ (2.3)

**Ví dụ tính tay.** Xét mô hình một tham số ŷ = w·x, một mẫu x = 2, y = 6, mất mát L = (wx − y)²/2. Khởi đầu w = 1: dự báo ŷ = 2, mất mát L = 8. Đạo hàm ∂L/∂w = (wx − y)·x = (2 − 6)·2 = −8. Với η = 0,1, trọng số mới là w = 1 − 0,1·(−8) = 1,8; dự báo mới 3,6 và mất mát giảm còn 2,88. Dấu âm của đạo hàm cho biết tăng w sẽ làm mất mát giảm, và thuật toán đã tự "tìm" ra hướng đó. Trong thực tế, gradient được ước lượng trên từng nhóm nhỏ dữ liệu (mini-batch); đi hết một lượt tập train gọi là một **epoch**. Với 6.400 mẫu và batch 64, mỗi epoch có 100 lần cập nhật.

> **Hiểu nhanh.** Hạ gradient giống người xuống núi trong sương mù dày đặc: không thấy chân núi, chỉ cảm nhận được độ dốc dưới chân, nên mỗi bước đều đi theo hướng dốc xuống nhất. Bước quá dài dễ trượt qua đáy thung lũng; bước quá ngắn thì rất lâu mới tới nơi.

## 2.4. Hồi quy logistic cho xác suất rủi ro

Với bài toán phân loại "có rủi ro / không có rủi ro", ta cần đầu ra là một xác suất trong khoảng (0, 1). **Hồi quy logistic** tính một điểm tuyến tính z rồi "nén" nó vào khoảng (0, 1) bằng hàm sigmoid σ:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \mathbf{w}^{\top}\mathbf{x} + b$$ (2.4)

Ví dụ: giả sử mô hình học được z = −2 + 0,8·m, với m là số tháng trễ hạn. Khách hàng trễ ba tháng có z = 0,4, nên p = σ(0,4) ≈ 0,599. Nếu ngưỡng quyết định là 0,5, khách hàng này được xếp vào nhóm rủi ro; nếu nâng ngưỡng lên 0,7 thì không. Ngưỡng là quyết định kinh doanh, tách biệt với mô hình.

Hàm mất mát phù hợp cho xác suất là **entropy chéo nhị phân** (binary cross-entropy – BCE):

$$L = -\frac{1}{n}\sum_{i=1}^{n}\left[\,y_i\ln p_i + (1 - y_i)\ln(1 - p_i)\,\right]$$ (2.5)

Với một khách hàng thực sự vỡ nợ (y = 1), dự báo p = 0,8 bị phạt −ln 0,8 ≈ 0,223, còn dự báo p = 0,2 bị phạt −ln 0,2 ≈ 1,609. BCE phạt nặng những dự báo "tự tin nhưng sai". Hồi quy logistic gọn, nhanh và dễ giải thích, nên là đường cơ sở tiêu chuẩn trong chấm điểm tín dụng. Hạn chế của nó là ranh giới quyết định tuyến tính, không tự nắm được tương tác kiểu "dư nợ cao chỉ đáng lo khi lịch sử trả nợ cũng xấu".

## 2.5. Cây quyết định và phương pháp tổ hợp

**Cây quyết định** chia dữ liệu bằng chuỗi câu hỏi có/không, chẳng hạn "trễ hạn quá 1 tháng?", rồi "tỷ lệ nợ trên 0,6?" [@breiman1984]. Mỗi lá của cây chứa một nhóm hồ sơ, và tỉ lệ vỡ nợ trong nhóm là xác suất dự báo. Để chọn câu hỏi, thuật toán tìm cách chia làm các nút con "thuần nhất" nhất, đo bằng chỉ số Gini:

$$G = 1 - \sum_{k} p_k^{2}$$ (2.6)

Nút có 50% vỡ nợ và 50% không vỡ nợ có G = 1 − (0,5² + 0,5²) = 0,5, mức hỗn tạp lớn nhất; nút chỉ gồm một lớp có G = 0. Cây đơn lẻ dễ đọc nhưng dễ quá khớp. **Rừng ngẫu nhiên** (random forest) khắc phục bằng cách trồng hàng trăm cây, mỗi cây học trên một mẫu bootstrap (lấy mẫu có hoàn lại) và chỉ được xét một tập con ngẫu nhiên các đặc trưng ở mỗi lần chia; dự báo cuối là trung bình của các cây [@breiman2001]. Vì các cây mắc lỗi khác nhau, phép lấy trung bình làm giảm phương sai: ba cây dự báo 0,2; 0,4; 0,6 cho kết quả chung 0,4. **Tăng cường gradient** (gradient boosting) trồng cây nối tiếp nhau, cây sau sửa phần sai của các cây trước [@friedman2001]. Các thư viện như XGBoost đã đưa phương pháp này thành tiêu chuẩn trên dữ liệu bảng [@chen2016xgboost]. Một nghiên cứu đối chuẩn năm 2022 xác nhận các mô hình dạng cây vẫn thường vượt học sâu trên dữ liệu bảng điển hình [@grinsztajn2022]. Đây là giả thuyết sẽ được kiểm chứng ở mục 2.10.

## 2.6. Một số thuật toán cơ bản khác

Bảng 2.2 tóm tắt bốn thuật toán kinh điển khác. Chúng không được đưa vào thực nghiệm của chương nhưng là kiến thức nền cần biết.

Bảng 2.2. Bốn thuật toán học máy kinh điển khác.
| Thuật toán | Ý tưởng chính | Ví dụ tài chính | Lưu ý |
|---|---|---|---|
| k láng giềng gần nhất (kNN) [@cover1967] | Dự báo theo k mẫu giống nhất | Xếp hạng tín dụng theo hồ sơ tương tự | Phải chuẩn hóa thang đo |
| Máy vector hỗ trợ (SVM) [@cortes1995] | Tìm ranh giới có lề rộng nhất | Phân loại doanh nghiệp an toàn/rủi ro | Điểm số chưa phải xác suất |
| k-means [@macqueen1967] | Gom điểm quanh k tâm cụm | Phân nhóm nhà đầu tư theo hành vi | Phải chọn trước số cụm k |
| Phân tích thành phần chính (PCA) [@pearson1901] | Nén nhiều biến tương quan thành ít trục | Tóm tắt hàng chục chỉ số tài chính | Không nhìn nhãn khi nén |

## 2.7. Mạng nơ-ron nhiều lớp và lan truyền ngược

Một **nơ-ron nhân tạo** tính tổng có trọng số của đầu vào rồi đưa qua một **hàm kích hoạt** phi tuyến (Hình 2.1a). Xếp nhiều nơ-ron thành lớp và nối các lớp với nhau ta được **mạng nơ-ron nhiều lớp** (multilayer perceptron – MLP), như Hình 2.1b. Với một lớp ẩn dùng hàm ReLU(a) = max(0, a) [@nair2010] và đầu ra sigmoid, MLP được viết là:

$$\mathbf{h} = \mathrm{ReLU}(W_1\mathbf{x} + \mathbf{b}_1), \qquad p = \sigma(\mathbf{w}_2^{\top}\mathbf{h} + b_2)$$ (2.7)

![Hình 2.1. (a) Một nơ-ron nhân tạo; (b) mạng nơ-ron nhiều lớp (MLP) với một lớp ẩn.](../../figures/tieuluan/diagram_neuron_mlp.png)

Ví dụ: một nơ-ron có trọng số 0,8 và 0,4, hệ số chặn −0,3, nhận đầu vào (1; 0,5) sẽ tính 0,8·1 + 0,4·0,5 − 0,3 = 0,7, rồi ReLU giữ nguyên 0,7. Hàm kích hoạt phi tuyến là bắt buộc: nếu bỏ nó đi, chồng bao nhiêu lớp tuyến tính cũng chỉ tương đương một phép biến đổi tuyến tính duy nhất.

Mạng được huấn luyện bằng **lan truyền ngược** [@rumelhart1986]: sau khi tính dự báo (lan truyền xuôi), ta áp dụng quy tắc đạo hàm của hàm hợp (chain rule) đi ngược từ đầu ra về đầu vào để biết mỗi trọng số đóng góp bao nhiêu vào sai số. Với sigmoid kết hợp BCE, đạo hàm theo điểm z ở lớp ra rút gọn rất đẹp:

$$\frac{\partial L}{\partial z} = p - y$$ (2.8)

Nếu khách hàng vỡ nợ (y = 1) mà mô hình chỉ cho p = 0,2, đạo hàm bằng −0,8, và cập nhật sẽ đẩy xác suất của mẫu này lên. Trong tiểu luận, trọng số được khởi tạo theo Glorot [@glorot2010] và được cập nhật bằng thuật toán Adam – một biến thể của hạ gradient tự điều chỉnh độ dài bước cho từng tham số [@kingma2015adam].

Khi tự viết lan truyền ngược, rất dễ sai một chỉ số hay một dấu cộng mà chương trình vẫn chạy bình thường. Cách kiểm chứng chuẩn là **kiểm tra gradient**: so đạo hàm giải tích với đạo hàm số tính bằng sai phân trung tâm [@goodfellow2016]:

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}$$ (2.9)

## 2.8. Đánh giá mô hình phân loại

Kết quả phân loại được tóm tắt trong **ma trận nhầm lẫn** gồm bốn ô: dương thật (TP – phát hiện đúng ca rủi ro), âm giả (FN – bỏ sót ca rủi ro), dương giả (FP – báo động nhầm) và âm thật (TN). Giả sử tập test có 100 hồ sơ, trong đó 10 hồ sơ vỡ nợ; mô hình phát hiện đúng 6, bỏ sót 4 và báo động nhầm 9. Bảng 2.3 tính các thước đo từ ví dụ này.

Bảng 2.3. Các thước đo phân loại, minh họa với TP = 6, FN = 4, FP = 9, TN = 81.
| Thước đo | Công thức | Ý nghĩa | Giá trị |
|---|---|---|---|
| Accuracy | (TP + TN) / N | Tỉ lệ dự báo đúng | 87% |
| Precision | TP / (TP + FP) | Trong số bị báo động, bao nhiêu là thật | 40% |
| Recall | TP / (TP + FN) | Trong số ca rủi ro, phát hiện được bao nhiêu | 60% |
| Specificity | TN / (TN + FP) | Nhận đúng ca an toàn | 90% |
| F1 | 2·Precision·Recall / (Precision + Recall) | Cân bằng precision và recall | 48% |
| Balanced accuracy | (Recall + Specificity) / 2 | Trung bình độ đúng của hai lớp | 75% |

Một mô hình "luôn đoán an toàn" trong ví dụ trên đạt accuracy 90%, cao hơn mô hình đang xét, nhưng recall bằng 0 vì không phát hiện được ca nào. Với dữ liệu mất cân bằng, accuracy vì thế rất dễ gây hiểu lầm. Tiểu luận dùng các thước đo không phụ thuộc tỉ lệ lớp. **ROC-AUC** là xác suất một mẫu dương ngẫu nhiên được mô hình chấm điểm cao hơn một mẫu âm ngẫu nhiên: 0,5 tương đương đoán ngẫu nhiên, 1,0 là hoàn hảo [@hanley1982]. **AP** (average precision) tóm tắt đường precision–recall và nhạy hơn ROC-AUC khi lớp dương rất hiếm [@saito2015]. **Balanced accuracy** là trung bình độ đúng trên hai lớp. ROC-AUC và AP dùng xác suất liên tục nên không phụ thuộc ngưỡng; balanced accuracy, F1 và recall phụ thuộc ngưỡng, và ngưỡng này được chọn trên tập validation.

## 2.9. Dữ liệu thực nghiệm

Chương này dùng hai bộ dữ liệu công khai của kho UCI, đều đại diện cho các quyết định đầu tư có thật (Bảng 2.4; Bảng 2.5 trích ba dòng đầu của mỗi bộ).

- **Phá sản doanh nghiệp Đài Loan** ([archive.ics.uci.edu/dataset/572](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)) [@uci572]: 95 chỉ số tài chính (khả năng sinh lời, đòn bẩy, thanh khoản, dòng tiền…) của các doanh nghiệp niêm yết giai đoạn 1999–2009; nhãn 1 nếu doanh nghiệp phá sản theo quy định của Sở Giao dịch Chứng khoán Đài Loan [@liang2016]. Đây là bài toán **sàng lọc cổ phiếu và trái phiếu** của nhà đầu tư theo phân tích cơ bản.
- **Vỡ nợ thẻ tín dụng Đài Loan** ([archive.ics.uci.edu/dataset/350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)) [@uci350]: 23 đặc trưng của 30.000 khách hàng (hạn mức, nhân khẩu học, trạng thái trả nợ 6 tháng, dư nợ và số tiền đã trả); nhãn 1 nếu vỡ nợ ở tháng kế tiếp [@yeh2009]. Đây là bài toán **đầu tư tín dụng**: ngân hàng hay nhà đầu tư cho vay ngang hàng cần biết khoản vay nào rủi ro.

{{t:ch2_datasets:2.4}}

{{t:ch2_samples:2.5}}

![Hình 2.2. Phân bố lớp của hai bộ dữ liệu (trái) và hai đặc trưng tiêu biểu: lãi ròng trên tổng tài sản theo tình trạng phá sản (giữa); tỉ lệ vỡ nợ theo trạng thái trả nợ tháng gần nhất PAY_0, số trên cột là số khách hàng (phải).](../../figures/tieuluan/ch2_data_overview.png)

**Nhận xét.** Hai bộ dữ liệu đều **mất cân bằng**: chỉ {{v:positive_rate.taiwan_bankruptcy}} doanh nghiệp phá sản và {{v:positive_rate.credit_default}} khách hàng vỡ nợ. Một mô hình luôn đoán "an toàn" đạt accuracy {{v:majority_acc.taiwan_bankruptcy}} và {{v:majority_acc.credit_default}} mà không phát hiện được rủi ro nào, nên accuracy không được dùng làm thước đo chính. Hình 2.2 cho thấy dữ liệu chứa tín hiệu rõ: doanh nghiệp phá sản tập trung ở vùng lãi ròng trên tổng tài sản thấp; tỉ lệ vỡ nợ khoảng 13–17% ở nhóm trả đúng hạn, nhưng tăng lên 34% khi trễ hạn một tháng và khoảng 70% khi trễ từ hai tháng (PAY_0 ≥ 2). Khi khảo sát dữ liệu còn phát hiện ba điểm cần lưu ý. Thứ nhất, 71/95 chỉ số của bộ phá sản nằm trong khoảng [0, 1], nhưng 24 chỉ số còn lại trộn hai thang đo: phần lớn giá trị nhỏ, xen lẫn giá trị cực lớn tới 10¹⁰ (chẳng hạn 88% giá trị của tốc độ tăng tổng tài sản vượt 1.000). Đây là dấu hiệu của khác biệt đơn vị hoặc mã lỗi ở nguồn; ảnh hưởng của nó được kiểm tra riêng ở mục 2.11. Thứ hai, các cột như giới tính, học vấn, trạng thái trả nợ là mã nhóm chứ không phải số đo; tiểu luận giữ chúng ở dạng số cho cả ba mô hình để phép so sánh công bằng. Thứ ba, cả hai bộ dữ liệu không có ô thiếu.

**Tiền xử lý và chia dữ liệu.** Mỗi dòng là một doanh nghiệp hay một khách hàng độc lập, không có trục thời gian, nên dữ liệu được chia ngẫu nhiên **phân tầng** theo tỉ lệ 60/20/20 (giữ tỉ lệ nhãn như nhau ở ba tập, hạt giống 2026). Giá trị trung vị để điền khuyết và trung bình, độ lệch chuẩn để chuẩn hóa z = (x − μ)/σ đều chỉ tính trên tập train rồi áp dụng nguyên cho validation và test (Bảng 2.6).

{{t:ch2_splits:2.6}}

## 2.10. Cài đặt bằng ba cách: NumPy, Keras và PyTorch

Mô hình đại diện của chương là MLP có một lớp ẩn 16 nơ-ron: d đầu vào → Dense(16) → ReLU → Dense(1) → logit, với hàm mất mát BCE (2.5). Cùng một kiến trúc được cài đặt theo ba cách. Bản **NumPy tự viết** [@harris2020numpy] cài đặt thủ công từng phép tính xuôi và ngược. Bản **Keras** (giao diện cấp cao của TensorFlow [@chollet2015keras; @abadi2016]) chỉ cần khai báo các lớp. Bản **PyTorch** [@paszke2019] khai báo mô hình và tự viết vòng huấn luyện, để thư viện tự tính đạo hàm. Đoạn mã dưới trích từ mã nguồn thực nghiệm (thư mục `src/tieuluan`).

Lớp kết nối đầy đủ tự viết: lan truyền xuôi y = xWᵀ + b, lan truyền ngược trả về gradient của trọng số, hệ số chặn và đầu vào.

```python
class Dense(Module):
    def forward(self, x):
        self._x = x
        return x @ self.weight.value.T + self.bias.value
    def backward(self, grad):                      # grad = ∂L/∂y
        self.weight.grad += grad.T @ self._x         # ∂L/∂W
        self.bias.grad += grad.sum(axis=0)           # ∂L/∂b
        return grad @ self.weight.value              # ∂L/∂x, chuyển cho lớp trước
```

Cùng MLP trong Keras và PyTorch:

```python
# Keras (backend TensorFlow)
inputs = keras.Input(shape=(d,))
z = keras.layers.Dense(16, activation="relu")(inputs)
model = keras.Model(inputs, keras.layers.Dense(1)(z))
# PyTorch
model = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
loss = nn.BCEWithLogitsLoss()(model(x), y); loss.backward(); optimizer.step()
```

**Thiết kế để so sánh công bằng.** Nếu mỗi thư viện tự khởi tạo trọng số theo cách riêng, khác biệt cuối cùng có thể chỉ do "điểm xuất phát" khác nhau. Vì vậy trọng số được rút **một lần** bằng NumPy rồi chép nguyên sang Keras và PyTorch; cả ba dùng cùng thứ tự mini-batch, cùng Adam (tốc độ học 0,001), cùng cắt chuẩn gradient ở mức 1,0, batch {{v:env.batch}}, tối đa {{v:env.epochs}} epoch và dừng sớm khi BCE trên validation không giảm sau {{v:env.patience}} epoch. Mỗi cấu hình chạy với ba hạt giống 11, 22, 33. Môi trường gồm Python {{v:env.python}}, NumPy {{v:env.numpy}}, Keras {{v:env.keras}} với TensorFlow {{v:env.tensorflow}}, PyTorch {{v:env.torch}} và scikit-learn {{v:env.sklearn}} [@pedregosa2011], chạy trên CPU.

**Kiểm chứng tính đúng.** Kiểm tra gradient (2.9) trên MLP tự viết cho sai số tương đối lớn nhất {{v:gc.mlp}}. Trước huấn luyện, với cùng trọng số và cùng đầu vào, logit của ba bản cài đặt lệch nhau tối đa {{v:parity.mlp}}, đúng bằng cỡ sai số làm tròn của số thực 32-bit.

## 2.11. Kết quả và thảo luận

**So sánh ba bản cài đặt.** Bảng 2.7 và Hình 2.3 cho thấy ba bản cài đặt cho kết quả gần như đồng nhất. Bản NumPy và PyTorch dừng ở cùng epoch với cả ba hạt giống, ROC-AUC theo từng hạt giống chênh nhau {{v:aucdiff.mlp.pytorch}}; sau huấn luyện, xác suất dự báo của chúng trên tập test chỉ lệch tối đa {{v:parity_trained.taiwan_bankruptcy.mlp.pytorch}} (bộ phá sản). Bản Keras lệch nhiều hơn một chút: ROC-AUC chênh {{v:aucdiff.mlp.keras}}, xác suất lệch tới {{v:parity_trained.taiwan_bankruptcy.mlp.keras}}, và đôi khi dừng ở epoch khác. Nguyên nhân là Keras đặt hằng số ε của Adam ở vị trí hơi khác và thực hiện phép toán theo thứ tự khác; sai khác cực nhỏ ở mỗi bước tích lũy dần qua hàng nghìn lần cập nhật. Về tốc độ, bản NumPy nhanh nhất trên mạng rất nhỏ này vì không phải trả chi phí điều phối của thư viện cho mỗi batch; kết quả này chỉ đúng với mô hình nhỏ chạy trên CPU và không phải bảng xếp hạng chung về tốc độ của các thư viện.

{{t:ch2_frameworks:2.7}}

![Hình 2.3. BCE trên tập validation theo epoch của ba bản cài đặt MLP (hạt giống 11); đường chấm dọc đánh dấu epoch tốt nhất. Đường NumPy và PyTorch trùng khít nhau.](../../figures/tieuluan/ch2_learning_curves.png)

**So sánh ba thuật toán.** Bảng 2.8 và Hình 2.4 so sánh hồi quy logistic, rừng ngẫu nhiên và MLP với đường cơ sở luôn đoán lớp đa số. Cả ba mô hình đều vượt xa đường cơ sở (AUC 0,5), với khoảng tin cậy 95% nằm hoàn toàn trên 0,7. Ở bộ phá sản, rừng ngẫu nhiên tốt nhất với AUC {{v:auc.taiwan_bankruptcy.random_forest.sklearn}}, so với {{v:auc.taiwan_bankruptcy.mlp.pytorch}} của MLP và {{v:auc.taiwan_bankruptcy.logistic.sklearn}} của hồi quy logistic. Tuy nhiên, khoảng tin cậy của hiệu AUC giữa rừng ngẫu nhiên và MLP là {{v:ucidiff.taiwan_bankruptcy.rf_minus_mlp}}, vẫn chạm 0: với chỉ {{v:npos.taiwan_bankruptcy.test}} doanh nghiệp phá sản trong tập test, chưa đủ bằng chứng để khẳng định ưu thế. Ở bộ vỡ nợ, rừng ngẫu nhiên ({{v:auc.credit_default.random_forest.sklearn}}) và MLP ({{v:auc.credit_default.mlp.pytorch}}) gần như ngang nhau và cùng vượt hồi quy logistic ({{v:auc.credit_default.logistic.sklearn}}); hiệu AUC giữa MLP và logistic là {{v:ucidiff.credit_default.mlp_minus_logistic}}, khác 0 có ý nghĩa thống kê. Điều này cho thấy quan hệ giữa các đặc trưng và rủi ro vỡ nợ có tính phi tuyến, như bước nhảy của tỉ lệ vỡ nợ theo PAY_0 ở Hình 2.2.

{{t:ch2_algorithms:2.8}}

![Hình 2.4. Đường cong ROC trên tập test của ba thuật toán; đường chéo nét đứt là đoán ngẫu nhiên.](../../figures/tieuluan/ch2_roc.png)

Kết quả phù hợp với nhận định của Grinsztajn và cộng sự: trên dữ liệu bảng cỡ vừa, mô hình dạng cây là đối thủ rất mạnh của học sâu [@grinsztajn2022]. Về mặt thực tiễn, với ngưỡng chọn trên validation, MLP phát hiện được trung bình {{v:recallpct.taiwan_bankruptcy.mlp.pytorch}} số doanh nghiệp phá sản và {{v:recallpct.credit_default.mlp.pytorch}} số khách hàng vỡ nợ trong tập test. Đổi lại là một tỉ lệ báo động nhầm đáng kể: đây chính là đánh đổi giữa chi phí bỏ sót một khoản vay xấu và chi phí từ chối nhầm một khách hàng tốt.

**Độ nhạy theo cách tiền xử lý.** Mục 2.9 đã chỉ ra rằng 24 chỉ số của bộ phá sản trộn hai thang đo, còn các cột số tiền của bộ vỡ nợ lệch phải rất mạnh. Để kiểm tra kết luận trên có phụ thuộc vào cách xử lý này hay không, ba thuật toán được huấn luyện lại sau khi biến đổi log có dấu x′ = sign(x)·ln(1 + |x|) trước bước chuẩn hóa, giữ nguyên cách chia dữ liệu và mọi thiết lập khác (Bảng 2.9). Rừng ngẫu nhiên gần như không đổi, đúng như lý thuyết: cây chỉ so sánh thứ tự các giá trị nên không bị ảnh hưởng bởi một phép biến đổi đơn điệu. Hồi quy logistic cải thiện ở bộ vỡ nợ ({{v:auc.credit_default.logistic.sklearn}} → {{v:sens.credit_default.logistic}}) vì phép log làm các cột số tiền bớt lệch, nhưng giảm nhẹ ở bộ phá sản ({{v:auc.taiwan_bankruptcy.logistic.sklearn}} → {{v:sens.taiwan_bankruptcy.logistic}}). MLP thay đổi chưa tới 0,01. Thứ hạng của ba thuật toán vì thế được giữ nguyên, và ưu thế của rừng ngẫu nhiên ở bộ phá sản không phải là hệ quả của cách xử lý thang đo.

{{t:ch2_sensitivity:2.9}}

## 2.12. Xu hướng mới: mô hình nền tảng cho dữ liệu bảng

Nếu học sâu từng thua cây trên dữ liệu bảng, thì giai đoạn 2025–2026 chứng kiến một hướng tiếp cận khác: **mô hình nền tảng cho dữ liệu bảng**. TabPFN, công bố trên tạp chí *Nature* tháng 01/2025, là một Transformer được tiền huấn luyện trên hàng triệu bài toán bảng tổng hợp. Khi gặp bảng dữ liệu mới, nó không cần huấn luyện lại mà "đọc" các dòng có nhãn như ngữ cảnh rồi dự báo dòng mới ngay trong một lần suy luận; với bảng đến 10.000 mẫu, nó vượt các mô hình cây đã được tinh chỉnh nhiều giờ [@hollmann2025]. TabPFN-2.5 (11/2025) mở rộng giới hạn lên 50.000 mẫu và 2.000 đặc trưng [@tabpfn25], đủ bao trùm cả bộ dữ liệu vỡ nợ 30.000 dòng của chương này. Ở hướng khác, TabM (ICLR 2025) cho thấy một MLP được "tổ hợp hóa" khéo léo, nhiều nhánh dùng chung phần lớn tham số, có thể cạnh tranh với cây [@gorishniy2025tabm]; mạng Kolmogorov–Arnold (KAN) đề xuất đặt hàm kích hoạt học được lên các cạnh thay vì nút của mạng [@liu2025kan]. Thử nghiệm TabPFN-2.5 trên hai bộ dữ liệu rủi ro là một hướng mở rộng tự nhiên của chương này.

## 2.13. Tiểu kết chương

Chương 2 đã trình bày các viên gạch nền của học máy: quy trình train – validation – test, hàm mất mát và hạ gradient, hồi quy logistic, cây và rừng ngẫu nhiên, MLP và lan truyền ngược, cùng các thước đo phù hợp cho dữ liệu mất cân bằng. Thực nghiệm trên hai bộ dữ liệu rủi ro cho thấy ba điều. Thứ nhất, một MLP tự viết bằng NumPy, nếu được kiểm tra gradient cẩn thận, cho kết quả trùng với Keras và PyTorch. Thứ hai, trên dữ liệu bảng, rừng ngẫu nhiên là đối thủ rất mạnh: tốt nhất ở bộ phá sản và ngang MLP ở bộ vỡ nợ. Thứ ba, khi có tín hiệu thật, như lịch sử trả nợ, cả ba thuật toán đạt ROC-AUC từ {{v:auc.credit_default.logistic.sklearn}} đến {{v:auc.taiwan_bankruptcy.random_forest.sklearn}}, cao hơn hẳn mức 0,5 của đoán ngẫu nhiên. Câu hỏi của hai chương tiếp theo khó hơn nhiều: liệu tín hiệu như vậy có tồn tại trong chính chuỗi giá thị trường hay không.
