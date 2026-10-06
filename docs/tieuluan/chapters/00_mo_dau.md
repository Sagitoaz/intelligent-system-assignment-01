# MỞ ĐẦU

## 1. Lý do chọn đề tài

Năm 2026 là một năm đáng nhớ đối với cả công nghệ lẫn thị trường vốn Việt Nam. Ngày 01/3/2026, Luật Trí tuệ nhân tạo (Luật số 134/2025/QH15) chính thức có hiệu lực, trở thành văn bản luật đầu tiên của Việt Nam điều chỉnh toàn diện lĩnh vực trí tuệ nhân tạo (Artificial Intelligence – AI) [@vn_ai_law2025]. Ngày 21/9/2026, thị trường chứng khoán Việt Nam được FTSE Russell nâng hạng từ "cận biên" lên "mới nổi thứ cấp", sau quyết định công bố ngày 07/10/2025 [@ftse2025]. Nâng hạng đồng nghĩa với việc các quỹ đầu tư theo chỉ số trên toàn cầu bắt đầu phân bổ vốn vào cổ phiếu Việt Nam, và thị trường sẽ ngày càng chuyên nghiệp, cạnh tranh hơn. Trong bối cảnh đó, câu hỏi "AI có thể giúp gì cho người làm đầu tư, và giới hạn của nó nằm ở đâu?" không còn là câu hỏi lý thuyết.

Tài chính cũng là lĩnh vực đi cùng AI từ rất sớm. Lý thuyết danh mục của Markowitz năm 1952 đặt nền móng cho đầu tư định lượng [@markowitz1952]; mô hình Z-score của Altman năm 1968 dùng thống kê để cảnh báo phá sản doanh nghiệp [@altman1968]; đến những năm 2020, học máy đã được dùng để định giá tài sản [@gu2020], mạng tích chập được dạy "đọc" biểu đồ giá giống nhà phân tích kỹ thuật [@jiang2023], còn các mô hình nền tảng chuyên cho dữ liệu nến giá xuất hiện từ năm 2025 [@shi2026kronos]. Mỗi bước tiến của AI gần như đều nhanh chóng được thử nghiệm trên thị trường tài chính.

Tuy vậy, tài chính lại là "bài kiểm tra khó" của AI. Theo giả thuyết thị trường hiệu quả, giá đã phản ánh phần lớn thông tin công khai, nên tín hiệu dự báo, nếu có, thường rất nhỏ so với nhiễu [@fama1970]. Dữ liệu tài chính còn chứa nhiều bẫy: dùng nhầm thông tin của tương lai, trộn ngẫu nhiên dữ liệu theo thời gian, hay đánh giá lợi nhuận mà bỏ quên phí giao dịch [@lopezdeprado2018]. Một mô hình đạt độ chính xác 97% có thể vô dụng nếu lớp cần phát hiện chỉ chiếm 3%; một chiến lược đoán đúng 55% số ngày vẫn có thể thua lỗ. Vì thế, học AI qua bài toán tài chính buộc người học phải hiểu cả **cách mô hình hoạt động** lẫn **cách đánh giá mô hình một cách trung thực**. Đó là lý do tiểu luận chọn chủ đề "Trí tuệ nhân tạo trong đầu tư tài chính" làm sợi chỉ đỏ xuyên suốt các chương của môn Phát triển hệ thống thông minh.

## 2. Mục tiêu nghiên cứu

Phần một của tiểu luận (Mở đầu – Chương 4) hướng tới năm mục tiêu:

1. Hệ thống hóa lịch sử phát triển của AI và lịch sử song hành của AI trong tài chính – đầu tư, từ năm 1943 đến năm 2026.
2. Trình bày các kỹ thuật học máy cơ bản, mạng nơ-ron tích chập (CNN) và mạng nơ-ron hồi quy (RNN) theo trình tự trực giác → ví dụ số tính tay → công thức → mã nguồn, để người không chuyên cũng có thể theo dõi.
3. Cài đặt cùng một mô hình theo ba cách – tự viết bằng NumPy, Keras và PyTorch – và chứng minh định lượng rằng ba bản cài đặt tương đương nhau.
4. Đánh giá các mô hình trên năm bộ dữ liệu tài chính thật bằng quy trình chống rò rỉ dữ liệu, có đường cơ sở, khoảng tin cậy và phân tích kinh tế có tính phí giao dịch.
5. Đóng gói mô hình thành ứng dụng minh họa có thể triển khai lên Internet.

## 3. Câu hỏi nghiên cứu

Các mục tiêu trên được cụ thể hóa thành bốn câu hỏi, sẽ được trả lời bằng số liệu ở phần Kết luận:

- **CH1.** Ba bản cài đặt (NumPy, Keras, PyTorch) của cùng một kiến trúc có cho kết quả tương đương không? Chúng khác nhau ở điểm nào?
- **CH2.** Trên dữ liệu bảng về rủi ro doanh nghiệp và rủi ro tín dụng, hồi quy logistic, rừng ngẫu nhiên hay mạng nơ-ron nhiều lớp phân biệt rủi ro tốt nhất?
- **CH3.** CNN (học từ ảnh của chuỗi giá) và RNN (học trực tiếp từ chuỗi lợi suất) có dự báo hướng đi ngày kế tiếp của S&P 500, VN-Index và Bitcoin tốt hơn đoán ngẫu nhiên không?
- **CH4.** Nếu có khả năng dự báo, khả năng đó có chuyển thành lợi nhuận sau phí giao dịch không?

## 4. Đối tượng và phạm vi nghiên cứu

**Đối tượng nghiên cứu** là các kỹ thuật học máy cơ bản (hồi quy logistic, cây quyết định và rừng ngẫu nhiên, mạng nơ-ron nhiều lớp), mạng tích chập và các mạng hồi quy SimpleRNN, LSTM, GRU, cùng cách áp dụng chúng vào hai nhóm bài toán đầu tư: đánh giá rủi ro trên dữ liệu bảng và dự báo hướng biến động của chỉ số thị trường.

**Phạm vi dữ liệu** gồm năm bộ dữ liệu: hai bộ dữ liệu bảng của kho UCI (phá sản doanh nghiệp Đài Loan; vỡ nợ thẻ tín dụng Đài Loan) và ba chuỗi chỉ số đại diện cho ba kiểu thị trường – S&P 500 (thị trường phát triển), VN-Index (thị trường Việt Nam) và Bitcoin (tài sản số giao dịch 24/7). Dữ liệu thị trường được chốt đến ngày 30/09/2026. **Phạm vi nội dung** dừng ở Chương 4; Knowledge Graph và RAG, học sâu trên đồ thị và hệ đa tác tử thuộc phần hai của tiểu luận. Tiểu luận không xây dựng hệ thống giao dịch thật và không đưa ra khuyến nghị đầu tư.

## 5. Phương pháp nghiên cứu

Tiểu luận kết hợp hai phương pháp. **Nghiên cứu tài liệu** dựa trên các công trình gốc, giáo trình kinh điển [@russell2021; @goodfellow2016] và các công bố mới nhất đến năm 2026; mọi thông tin về công trình đều được dẫn nguồn và đối chiếu với nơi xuất bản. **Thực nghiệm định lượng** được thiết kế như Hình 0.1: từ năm bộ dữ liệu, tiểu luận tạo ba cách biểu diễn (bảng, ảnh, chuỗi), huấn luyện các mô hình tương ứng với từng chương, mỗi mô hình đại diện được cài đặt theo ba cách, rồi đánh giá và triển khai.

![Hình 0.1. Thiết kế nghiên cứu của phần một tiểu luận: từ dữ liệu đến triển khai.](../../figures/tieuluan/diagram_study_design.png)

Toàn bộ thực nghiệm tuân thủ năm nguyên tắc: (i) dữ liệu theo thời gian được chia theo thời gian, nhãn không được vượt ranh giới giữa các tập; (ii) mọi phép tiền xử lý chỉ "học" trên tập huấn luyện; (iii) ngưỡng quyết định và số vòng huấn luyện chỉ được chọn trên tập xác thực, tập kiểm tra chỉ dùng một lần ở cuối; (iv) mọi mô hình đều được so với đường cơ sở đơn giản; (v) mỗi cấu hình chạy với ba hạt giống ngẫu nhiên, kèm khoảng tin cậy bootstrap. Tổng cộng có {{v:n_records}} bản ghi thực nghiệm ({{v:n_trained}} lượt huấn luyện và 8 đường cơ sở), tất cả được lưu kèm mã nguồn và dự báo từng mẫu để có thể kiểm tra lại.

## 6. Đóng góp của tiểu luận

- Xây dựng bộ dữ liệu VN-Index được **kiểm chứng chéo giữa ba nguồn** (SSI, DNSE, VNDirect). Quá trình này phát hiện và xử lý ba vấn đề: {{v:vnq.close_disagreements_ssi_vs_dnse}} phiên lệch giá giữa các nguồn, một lỗ hổng {{v:vnq.gap2009}} phiên năm 2009 tạo ra mức "tăng 23%" không có thật, và giai đoạn giá bị làm tròn trước năm 2009.
- Tự cài đặt một thư viện học sâu nhỏ bằng NumPy (lớp kết nối đầy đủ, tích chập, gộp, SimpleRNN, LSTM, GRU, Adam). Thư viện được kiểm tra gradient bằng sai phân và đối chiếu với Keras, PyTorch khi dùng **cùng trọng số khởi tạo, cùng thứ tự dữ liệu**.
- Đưa ra kết quả trung thực, có kiểm định thống kê: mô hình phân biệt rủi ro tín dụng rất tốt trên dữ liệu bảng; với thị trường, chỉ VN-Index cho thấy khả năng dự báo nhỏ nhưng có ý nghĩa thống kê, còn S&P 500 và Bitcoin thì không; và sau phí giao dịch, không chiến lược nào vượt rõ chiến lược mua và giữ.
- Đóng gói mô hình thành ứng dụng web Streamlit trên Render: đối chiếu dự báo theo ngày, tải giá mới để dự báo phiên tới, nhập hồ sơ tín dụng và xem mô phỏng bằng tiền; giải thích kết quả và giới hạn cho người không chuyên.

## 7. Cấu trúc tiểu luận

Ngoài Mở đầu, Kết luận, Tài liệu tham khảo và Phụ lục, phần một gồm bốn chương. **Chương 1** trình bày lịch sử phát triển của AI và của AI trong tài chính, đồng thời giới thiệu ba chuỗi chỉ số dùng xuyên suốt. **Chương 2** trình bày các kỹ thuật học máy cơ bản và thực nghiệm trên hai bộ dữ liệu rủi ro. **Chương 3** trình bày mạng tích chập và thử nghiệm "đọc" ảnh của chuỗi giá. **Chương 4** trình bày mạng hồi quy, so sánh với Chương 3 và phân tích giá trị kinh tế của dự báo. Phụ lục hướng dẫn tái lập thực nghiệm và triển khai ứng dụng.
