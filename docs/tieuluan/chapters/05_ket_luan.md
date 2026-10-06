# KẾT LUẬN

## 1. Trả lời các câu hỏi nghiên cứu

**CH1 – Ba bản cài đặt có tương đương không?** Có. Khi dùng chung trọng số khởi tạo và thứ tự mini-batch, bản NumPy tự viết và bản PyTorch cho ROC-AUC theo từng hạt giống chênh nhau {{v:aucdiff.mlp.pytorch}} với MLP, {{v:aucdiff.cnn4.pytorch}} với CNN4 và {{v:aucdiff.lstm.pytorch}} với LSTM. Kiểm tra gradient bằng sai phân cho sai số tương đối (tính trên toàn vector gradient) chỉ từ {{v:gcnorm.cnn}} đến {{v:gcnorm.lstm}}. Bản Keras lệch nhiều hơn một chút, ROC-AUC của MLP chênh {{v:aucdiff.mlp.keras}}, do hằng số ε của Adam được đặt ở vị trí khác và thứ tự phép tính khác; mức lệch này vẫn nhỏ hơn nhiều so với chênh lệch giữa các hạt giống. Khác biệt thực sự giữa ba cách nằm ở công sức và tốc độ. Bản NumPy buộc người viết hiểu từng đạo hàm, và nhanh nhất với MLP rất nhỏ ({{v:time.credit_default.mlp.scratch}} giây mỗi lượt huấn luyện trên bộ vỡ nợ, so với {{v:time.credit_default.mlp.pytorch}} giây của PyTorch và {{v:time.credit_default.mlp.keras}} giây của Keras). PyTorch nhanh nhất với CNN và LSTM. Keras cần ít dòng mã nhất nhưng thường chậm nhất khi huấn luyện theo từng batch nhỏ.

**CH2 – Thuật toán nào phân biệt rủi ro tốt nhất trên dữ liệu bảng?** Rừng ngẫu nhiên là lựa chọn tốt nhất, nhưng không bỏ xa MLP. Ở bộ phá sản, rừng ngẫu nhiên đạt ROC-AUC {{v:auc.taiwan_bankruptcy.random_forest.sklearn}}, so với {{v:auc.taiwan_bankruptcy.mlp.pytorch}} của MLP và {{v:auc.taiwan_bankruptcy.logistic.sklearn}} của hồi quy logistic. Tuy vậy, với chỉ {{v:npos.taiwan_bankruptcy.test}} ca phá sản trong tập test, ưu thế so với MLP chưa đạt ý nghĩa thống kê. Ở bộ vỡ nợ, rừng ngẫu nhiên và MLP ngang nhau ({{v:auc.credit_default.random_forest.sklearn}} và {{v:auc.credit_default.mlp.pytorch}}) và cùng vượt hồi quy logistic một cách có ý nghĩa. Thứ hạng này giữ nguyên khi thay đổi cách tiền xử lý (mục 2.11).

**CH3 – CNN và RNN có dự báo được hướng đi của phiên kế tiếp không?** Với S&P 500 và Bitcoin thì không: mọi khoảng tin cậy 95% của ROC-AUC đều chứa 0,5. Với VN-Index thì có, nhưng rất nhỏ. LSTM ({{v:ci.vnindex.lstm}}) và GRU ({{v:ci.vnindex.gru}}) có khoảng tin cậy nằm trên 0,5, còn các CNN ở sát ranh giới. Tín hiệu này chủ yếu đến từ quán tính ngắn hạn của VN-Index (hệ số tự tương quan bậc một {{v:ac1.vnindex}}), mà một quy tắc đơn giản như "lặp lại hướng phiên trước" cũng đã khai thác được một phần. Trên cùng bài toán, mạng hồi quy đọc chuỗi lợi suất nhỉnh hơn và ổn định hơn CNN đọc ảnh GASF.

**CH4 – Khả năng dự báo có chuyển thành lợi nhuận sau phí không?** Không, với thiết kế của tiểu luận. Trước phí, chiến lược theo GRU trên VN-Index đạt {{v:bt.vnindex.gru.retgross}} mỗi năm, vượt mua và giữ ({{v:bt.vnindex.buy_hold.retgross}}). Sau phí, con số giảm còn {{v:bt.vnindex.gru.ret}} và thua mua và giữ ({{v:bt.vnindex.buy_hold.ret}}). Ở S&P 500 và Bitcoin, không chiến lược nào theo kịp mua và giữ. Kết quả này phù hợp với giả thuyết thị trường hiệu quả hiểu theo nghĩa thực tế: tín hiệu nhỏ có thể tồn tại, nhưng không đủ lớn để bù chi phí khai thác nó [@fama1970; @fischer2018].

## 2. Bài học rút ra

Thứ nhất, **cách đánh giá quan trọng không kém mô hình**. Chỉ cần chia dữ liệu ngẫu nhiên thay vì theo thời gian, hoặc dự báo mức giá thay vì lợi suất, một mô hình vô dụng cũng có thể đạt con số đẹp: dự báo "giá mai bằng giá hôm nay" đã có R² = {{v:naive_r2.sp500}}. Đường cơ sở, khoảng tin cậy và chi phí giao dịch là ba "phép thử" giúp tránh tự đánh lừa.

Thứ hai, **chất lượng dữ liệu quyết định kết quả**. Chuỗi VN-Index từ một nguồn duy nhất chứa một mức "tăng 23%" không có thật do thiếu dữ liệu năm 2009. Lỗi này chỉ lộ ra khi đối chiếu ba nguồn. Nếu không phát hiện, mô hình sẽ học từ một sự kiện chưa từng xảy ra.

Thứ ba, **hiểu nền tảng giúp phát hiện lỗi tinh vi**. Việc tự cài đặt bằng NumPy và đối chiếu với hai thư viện đã làm lộ ra những chi tiết dễ bị bỏ qua: thứ tự cổng của GRU trong Keras khác PyTorch, PyTorch có hai vector bias cho mỗi lớp hồi quy, và cách chuẩn hóa [−1, 1] của GASF làm mất thông tin về chiều giá.

## 3. Hạn chế

- **Thông tin đầu vào hẹp.** Các mô hình thị trường chỉ dùng giá đóng cửa của ba chỉ số; chưa dùng khối lượng, tin tức, dữ liệu vĩ mô hay dữ liệu từng cổ phiếu.
- **Mô hình nhỏ và không tinh chỉnh siêu tham số một cách hệ thống.** Đây là lựa chọn có chủ đích, để giảm nguy cơ "học thuộc" tập validation vốn nhỏ, nhưng có thể chưa khai thác hết khả năng của từng kiến trúc.
- **Kiểm định nhiều mô hình cùng lúc.** Với 18 cặp mô hình – thị trường, nếu không có tín hiệu thật, trung bình vẫn có khoảng một kết quả "có ý nghĩa" ở mức 5% chỉ do ngẫu nhiên [@harvey2016]. Tiểu luận vì thế báo cáo toàn bộ kết quả thay vì chọn mô hình tốt nhất trên tập test [@bailey2014]. Kết luận về VN-Index đáng tin hơn một phát hiện đơn lẻ vì nó nhất quán qua cả sáu kiến trúc (ROC-AUC của dự báo trung bình từ {{v:ciauc.vnindex.cnn4}} đến {{v:ciauc.vnindex.gru}}) và được giải thích bởi một đặc điểm độc lập của dữ liệu là tự tương quan dương.
- **Backtest đơn giản hóa.** Mô phỏng giả định khớp lệnh đúng giá đóng cửa, không tính trượt giá, lãi tiền gửi và chu kỳ thanh toán T+2; kết quả chỉ mang tính minh họa.
- **Dữ liệu bảng không phải của Việt Nam.** Hai bộ dữ liệu rủi ro đến từ Đài Loan, trong các giai đoạn 1999–2009 và 2005, nên kết quả chưa chắc đúng với doanh nghiệp và khách hàng Việt Nam hiện nay.

## 4. Hướng phát triển

Phần hai của tiểu luận sẽ mở rộng theo ba chương còn lại, tiếp tục chủ đề đầu tư tài chính và tái sử dụng dữ liệu đã thu thập (cổ phiếu Mỹ, rổ VN30 và 20 tài sản số):

- **Chương 5 – Knowledge Graph và RAG:** xây dựng đồ thị tri thức về các doanh nghiệp VN30 (ngành, cổ đông lớn, quan hệ công ty mẹ – con) [@hogan2021kg], kết hợp truy xuất báo cáo tài chính và tin tức theo hướng sinh văn bản có truy xuất (RAG) [@lewis2020rag] để trả lời câu hỏi đầu tư kèm nguồn dẫn.
- **Chương 6 – Học sâu trên đồ thị:** dùng mạng nơ-ron đồ thị [@kipf2017gcn] trên đồ thị quan hệ giữa các cổ phiếu, để kiểm tra liệu thông tin từ các cổ phiếu "hàng xóm" có bổ sung tín hiệu mà chuỗi giá riêng lẻ không có.
- **Chương 7 – Hệ đa tác tử:** xây dựng nhóm tác tử mô phỏng một bộ phận phân tích đầu tư (phân tích cơ bản, kỹ thuật, tin tức, quản trị rủi ro) theo hướng TradingAgents [@xiao2024tradingagents], và đánh giá bằng đúng quy trình của phần một: chia theo thời gian, có đường cơ sở và tính phí giao dịch.

Ngoài ra, hai thử nghiệm có thể làm ngay là áp dụng TabPFN-2.5 [@tabpfn25] cho hai bộ dữ liệu rủi ro của Chương 2, và đánh giá mô hình nền tảng Kronos [@shi2026kronos] cho VN-Index trên giai đoạn sau ngày mô hình được công bố.
