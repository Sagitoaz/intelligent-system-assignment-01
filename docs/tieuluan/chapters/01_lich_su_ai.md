# CHƯƠNG 1. LỊCH SỬ PHÁT TRIỂN CỦA TRÍ TUỆ NHÂN TẠO

## 1.1. Trí tuệ nhân tạo là gì?

Năm 1950, Alan Turing mở đầu bài báo "Computing Machinery and Intelligence" bằng câu hỏi "Máy móc có thể suy nghĩ không?". Vì "suy nghĩ" khó định nghĩa, ông đề xuất một phép thử: nếu qua trao đổi bằng văn bản, người hỏi không phân biệt được đâu là máy, đâu là người, thì có thể coi máy đã hành xử thông minh [@turing1950]. Năm năm sau, cụm từ "artificial intelligence" lần đầu xuất hiện trong đề xuất hội thảo Dartmouth của McCarthy, Minsky, Rochester và Shannon [@mccarthy1955].

Ngày nay, giáo trình phổ biến nhất của ngành định nghĩa AI theo bốn hướng: máy suy nghĩ như người, hành động như người, suy nghĩ hợp lý và hành động hợp lý; hướng được dùng rộng rãi nhất là xem AI như một **tác tử hợp lý** (rational agent) – hệ thống nhận thông tin từ môi trường và chọn hành động tốt nhất theo mục tiêu đã đặt [@russell2021]. Trong AI có hai nhánh con quen thuộc. **Học máy** (machine learning – ML) là cách xây dựng hệ thống thông minh bằng cách cho máy học từ dữ liệu thay vì viết sẵn mọi quy tắc. **Học sâu** (deep learning – DL) là nhánh của học máy dùng mạng nơ-ron nhiều tầng, tự học ra các đặc trưng từ dữ liệu thô [@goodfellow2016].

> **Hiểu nhanh.** Hãy hình dung một ngân hàng muốn tự động duyệt hồ sơ vay. Cách "AI cổ điển" là mời chuyên gia viết ra hàng trăm quy tắc *nếu – thì* ("nếu trễ hạn quá 90 ngày thì từ chối"). Cách "học máy" là đưa cho máy hàng chục nghìn hồ sơ cũ đã biết kết quả để máy tự tìm ra quy luật. Cách "học sâu" đi xa hơn: máy tự học cả việc nên chú ý đến những đặc điểm nào, kể cả từ dữ liệu thô như ảnh hay chuỗi thời gian.

Các hệ thống hiện nay đều là **AI hẹp**: giỏi một nhóm nhiệm vụ cụ thể như nhận dạng ảnh, dịch thuật hay chấm điểm tín dụng. AI tổng quát, có năng lực ngang con người trên mọi nhiệm vụ, vẫn là mục tiêu nghiên cứu và chủ đề tranh luận.

## 1.2. Giai đoạn hình thành và lạc quan (1943–1969)

Mốc khởi đầu thường được nhắc đến là mô hình nơ-ron nhân tạo của McCulloch và Pitts năm 1943: một nơ-ron được mô tả như một phép toán logic, "bật" khi tổng tín hiệu đầu vào vượt ngưỡng [@mcculloch1943]. Mùa hè năm 1956, hội thảo Dartmouth quy tụ những người sáng lập ngành với tham vọng mô tả mọi khía cạnh của trí thông minh chính xác đến mức máy có thể mô phỏng [@mccarthy1955]. Năm 1958, Rosenblatt công bố **perceptron**, mô hình đầu tiên tự điều chỉnh trọng số từ ví dụ để phân loại [@rosenblatt1958]. Năm 1966, chương trình hội thoại ELIZA của Weizenbaum khiến nhiều người dùng tin rằng máy "hiểu" họ, dù thực chất nó chỉ khớp mẫu câu và lặp lại từ khóa [@weizenbaum1966].

Những thành công ban đầu tạo ra làn sóng lạc quan: nhiều nhà nghiên cứu dự đoán máy sẽ đạt trí thông minh như người chỉ trong một thế hệ. Bài học đầu tiên của lịch sử AI nằm ở đây: thành công trên các bài toán "đồ chơi" thường bị ngoại suy quá mức sang thế giới thực.

## 1.3. Hệ chuyên gia và hai "mùa đông AI" (1969–1993)

Năm 1969, cuốn sách *Perceptrons* của Minsky và Papert chứng minh perceptron một lớp không thể học những quan hệ đơn giản như phép XOR [@minsky1969]. Năm 1973, báo cáo Lighthill gửi Hội đồng Nghiên cứu Khoa học Anh kết luận AI chưa đạt được các hứa hẹn ban đầu [@lighthill1973]. Kinh phí bị cắt giảm mạnh ở cả Anh và Mỹ, mở ra **mùa đông AI** thứ nhất, kéo dài khoảng từ giữa những năm 1970 đến đầu những năm 1980.

AI hồi sinh nhờ **hệ chuyên gia**: tri thức của chuyên gia được mã hóa thành các quy tắc *nếu – thì*. MYCIN (Đại học Stanford) tư vấn chẩn đoán nhiễm trùng máu và kê kháng sinh [@shortliffe1976]; hệ R1/XCON giúp hãng DEC cấu hình máy tính VAX theo đơn đặt hàng [@mcdermott1982]. Những năm 1980 chứng kiến làn sóng thương mại hóa, kể cả trong ngân hàng và bảo hiểm. Nhưng hệ chuyên gia sớm bộc lộ điểm yếu: tốn kém khi thu thập và bảo trì tri thức, "giòn" (brittle) trước tình huống nằm ngoài các quy tắc, và không tự học thêm được. Khi thị trường máy tính chuyên dụng cho AI sụp đổ năm 1987, mùa đông thứ hai bắt đầu và kéo dài tới đầu những năm 1990.

## 1.4. Học máy thống kê và sự trở lại của mạng nơ-ron (1986–2011)

Trong mùa đông, một số hướng đi quan trọng vẫn âm thầm phát triển. Năm 1986, Rumelhart, Hinton và Williams phổ biến thuật toán **lan truyền ngược** (backpropagation), cho phép huấn luyện mạng nơ-ron nhiều lớp và vượt qua giới hạn mà Minsky – Papert chỉ ra [@rumelhart1986]. LeCun và cộng sự phát triển mạng tích chập LeNet; hệ thống đọc séc dùng mạng này đã được triển khai thương mại và xử lý hàng triệu tờ séc ngân hàng mỗi ngày [@lecun1998]. Đây có lẽ là một trong những ứng dụng học sâu thực tế đầu tiên, và nó thuộc ngành tài chính.

Thập niên 1990–2000 là thời của **học máy thống kê**: máy vector hỗ trợ (SVM) [@cortes1995], rừng ngẫu nhiên [@breiman2001] và các phương pháp có nền tảng toán học chặt chẽ. Năm 1997 có hai sự kiện đáng chú ý. Siêu máy tính Deep Blue của IBM thắng nhà vô địch cờ vua thế giới Garry Kasparov, chủ yếu nhờ tìm kiếm quy mô lớn [@campbell2002]. Cùng năm, Hochreiter và Schmidhuber công bố LSTM, kiến trúc mạng hồi quy sẽ trở thành trọng tâm của Chương 4 [@hochreiter1997]. Năm 2006, Hinton và cộng sự chỉ ra cách huấn luyện hiệu quả mạng nhiều tầng [@hinton2006], làm sống lại thuật ngữ "học sâu". Năm 2009, bộ dữ liệu ImageNet với hơn một triệu ảnh có nhãn ra đời [@deng2009]. Dữ liệu lớn, sức mạnh tính toán của GPU và thuật toán tốt hơn trở thành ba trụ cột cho giai đoạn tiếp theo.

## 1.5. Kỷ nguyên học sâu (2012–2021)

Năm 2012, mạng tích chập AlexNet thắng cuộc thi ImageNet với tỉ lệ lỗi top-5 là 15,3%, trong khi đội xếp thứ hai có 26,2% [@krizhevsky2012]. Khoảng cách lớn đến mức cộng đồng thị giác máy tính gần như chuyển hẳn sang học sâu. Các năm sau đó liên tục có đột phá: mạng đối nghịch sinh dữ liệu (GAN) năm 2014 [@goodfellow2014gan]; ResNet với kết nối tắt cho phép huấn luyện mạng hơn 150 lớp [@he2016]; AlphaGo thắng kỳ thủ cờ vây hàng đầu Lee Sedol năm 2016 [@silver2016]; kiến trúc Transformer năm 2017 dựa hoàn toàn vào cơ chế chú ý (attention) [@vaswani2017], làm nền cho BERT [@devlin2019] và GPT-3 với 175 tỉ tham số [@brown2020]. Năm 2020–2021, AlphaFold 2 dự đoán cấu trúc protein với độ chính xác gần thực nghiệm [@jumper2021], cho thấy AI bắt đầu đóng góp trực tiếp vào khoa học.

## 1.6. AI tạo sinh, mô hình nền tảng và tác tử (2022–2026)

Ngày 30/11/2022, OpenAI ra mắt ChatGPT [@openai2022chatgpt] và đưa AI tạo sinh đến hàng trăm triệu người dùng. GPT-4 (2023) mở rộng sang đầu vào hình ảnh [@openai2023gpt4]. Từ cuối năm 2024, các **mô hình suy luận** được huấn luyện để "nghĩ" nhiều bước trước khi trả lời; DeepSeek-R1 (01/2025) cho thấy học tăng cường quy mô lớn có thể tự hình thành năng lực suy luận nhiều bước, và mô hình được công bố mở trọng số [@deepseek2025]. Tháng 7/2025, một phiên bản của Gemini Deep Think giải được 5/6 bài thi Olympic Toán quốc tế, đạt 35/42 điểm, tương đương huy chương vàng, chỉ bằng ngôn ngữ tự nhiên [@deepmind2025imo].

Giới khoa học cũng chính thức ghi nhận đóng góp của AI. Giải Nobel Vật lý 2024 được trao cho John Hopfield và Geoffrey Hinton vì những phát minh nền tảng cho học máy bằng mạng nơ-ron [@nobel2024physics]; Giải Nobel Hóa học 2024 được trao cho David Baker, Demis Hassabis và John Jumper, trong đó hai người sau được vinh danh nhờ AlphaFold [@nobel2024chemistry]. Giải Turing năm 2024, công bố ngày 05/3/2025, thuộc về Andrew Barto và Richard Sutton – những người đặt nền móng cho học tăng cường [@acm2025turing].

Song song với công nghệ là khung pháp lý. Đạo luật AI của Liên minh châu Âu (Quy định 2024/1689) là bộ luật toàn diện đầu tiên về AI trên thế giới [@euaiact2024]. Tại Việt Nam, sau Chiến lược quốc gia về AI năm 2021 [@vn_ai_strategy2021], Quốc hội thông qua Luật Trí tuệ nhân tạo ngày 10/12/2025, có hiệu lực từ 01/3/2026 [@vn_ai_law2025]. Tháng 2/2026, Hội nghị Thượng đỉnh Tác động AI tại New Delhi kết thúc với Tuyên bố New Delhi được 88 quốc gia và tổ chức tán thành [@newdelhi2026]. Về kỹ thuật, xu hướng nổi bật của giai đoạn 2024–2026 là **tác tử AI** (AI agent) tự lập kế hoạch và dùng công cụ, cùng sự trở lại của các kiến trúc hồi quy hiện đại như Mamba và xLSTM – chủ đề sẽ quay lại ở mục 4.10.

Hình 1.1 và Bảng 1.1 tóm tắt các mốc chính; dải gạch chéo ở Hình 1.1 đánh dấu hai mùa đông AI.

![Hình 1.1. Dòng thời gian phát triển của AI (phía trên) và của AI trong tài chính – đầu tư (phía dưới).](../../figures/tieuluan/diagram_ai_timeline.png)

Bảng 1.1. Một số mốc phát triển tiêu biểu của AI và ý nghĩa của chúng.
| Năm | Sự kiện | Ý nghĩa |
|---|---|---|
| 1943 | Nơ-ron McCulloch–Pitts [@mcculloch1943] | Mô hình toán học đầu tiên của nơ-ron |
| 1950 | Phép thử Turing [@turing1950] | Đặt câu hỏi "máy có thể suy nghĩ không?" |
| 1956 | Hội thảo Dartmouth [@mccarthy1955] | Khai sinh ngành AI và thuật ngữ AI |
| 1958 | Perceptron [@rosenblatt1958] | Mô hình đầu tiên tự học trọng số từ ví dụ |
| 1974–1980 | Mùa đông AI thứ nhất [@lighthill1973] | Kỳ vọng vượt xa năng lực, kinh phí bị cắt |
| 1980–1987 | Hệ chuyên gia [@shortliffe1976; @mcdermott1982] | AI thương mại dựa trên luật *nếu – thì* |
| 1986 | Lan truyền ngược [@rumelhart1986] | Huấn luyện được mạng nhiều lớp |
| 1997 | Deep Blue; LSTM [@campbell2002; @hochreiter1997] | Máy thắng vua cờ; bộ nhớ dài cho chuỗi |
| 2012 | AlexNet [@krizhevsky2012] | Mở đầu kỷ nguyên học sâu |
| 2017 | Transformer [@vaswani2017] | Nền tảng của các mô hình ngôn ngữ lớn |
| 2022 | ChatGPT [@openai2022chatgpt] | AI tạo sinh phổ cập đến công chúng |
| 2024 | Nobel Vật lý và Hóa học [@nobel2024physics; @nobel2024chemistry] | Khoa học chính thức ghi nhận AI |
| 2026 | Luật Trí tuệ nhân tạo có hiệu lực [@vn_ai_law2025] | Hành lang pháp lý cho AI tại Việt Nam |

## 1.7. AI trong tài chính – đầu tư: một lịch sử song hành

Tài chính là một trong những lĩnh vực áp dụng tư duy định lượng sớm nhất. Năm 1952, Markowitz chứng minh nhà đầu tư nên đánh giá danh mục bằng cả lợi nhuận kỳ vọng lẫn rủi ro (phương sai) [@markowitz1952]. Năm 1968, Altman dùng phân tích phân biệt trên các chỉ số tài chính để dự báo phá sản doanh nghiệp, tiền thân của bài toán ở Chương 2 [@altman1968]. Năm 1970, Fama tổng kết giả thuyết thị trường hiệu quả [@fama1970]: nếu giá đã phản ánh thông tin sẵn có thì việc dự báo giá bằng dữ liệu quá khứ gần như không mang lại lợi thế. Giả thuyết này là "đối thủ" mà mọi mô hình dự báo ở Chương 3 và 4 phải vượt qua.

Mạng nơ-ron được đưa vào tài chính ngay khi chúng hồi sinh: năm 1990, Odom và Sharda dùng mạng nơ-ron dự báo phá sản và cho kết quả tốt hơn phân tích phân biệt truyền thống [@odom1990]. Giao dịch bằng thuật toán phát triển mạnh trong những năm 2000 và đi kèm rủi ro mới: ngày 06/5/2010, chỉ số Dow Jones mất gần 1.000 điểm trong vài phút rồi hồi phục trong sự kiện "Flash Crash", nơi các nhà giao dịch tần suất cao đóng vai trò khuếch đại biến động [@kirilenko2017]. Với học sâu, Krauss và cộng sự (2017) so sánh mạng nơ-ron sâu, cây tăng cường và rừng ngẫu nhiên trong chiến lược kinh doanh chênh lệch giá trên S&P 500 [@krauss2017]. Fischer và Krauss (2018) cho thấy LSTM vượt các phương pháp khác, nhưng lợi nhuận suy giảm mạnh từ khoảng năm 2010 [@fischer2018]. Nghiên cứu này là minh chứng rõ cho việc thị trường "học" và triệt tiêu các quy luật dễ khai thác. Gu, Kelly và Xiu (2020) cho thấy học máy, đặc biệt là cây hồi quy và mạng nơ-ron, đem lại lợi ích kinh tế lớn trong định giá tài sản – trong một số trường hợp tăng gấp đôi hiệu quả của các chiến lược dựa trên hồi quy truyền thống [@gu2020].

Giai đoạn gần đây đánh dấu bước chuyển sang mô hình lớn. Năm 2023, Jiang, Kelly và Xiu dạy mạng tích chập "đọc" ảnh biểu đồ giá và đạt hiệu quả vượt các tín hiệu kỹ thuật truyền thống [@jiang2023]; cùng năm, Bloomberg công bố BloombergGPT, mô hình ngôn ngữ 50 tỉ tham số chuyên cho tài chính [@wu2023bloomberggpt]. Năm 2024, TradingAgents mô phỏng cả một công ty đầu tư bằng nhiều tác tử mô hình ngôn ngữ lớn đóng vai nhà phân tích, nhà nghiên cứu và nhà giao dịch [@xiao2024tradingagents]. Năm 2025, Kronos – mô hình nền tảng mã nguồn mở đầu tiên cho dữ liệu nến giá (K-line), được tiền huấn luyện trên hơn 12 tỉ bản ghi nến của 45 sàn giao dịch – được chấp nhận tại hội nghị AAAI 2026 [@shi2026kronos].

Nhìn lại, mỗi làn sóng AI đều nhanh chóng được thử nghiệm trong tài chính, nhưng cũng mang theo rủi ro mới: mô hình học thuộc quá khứ, chiến lược bị khai thác cạn, hay hành vi tự động khuếch đại biến động. Đây là lý do các chương sau luôn đặt kết quả cạnh đường cơ sở và tính đến chi phí giao dịch.

## 1.8. Dữ liệu minh họa: ba chỉ số thị trường

Theo yêu cầu của tiểu luận, mỗi chương đều gắn với dữ liệu thật. Chương này giới thiệu ba chuỗi chỉ số được dùng xuyên suốt ở Chương 3 và Chương 4; hai bộ dữ liệu bảng được giới thiệu ở Chương 2.

**Nguồn và cách thu thập.** S&P 500 (mã ^GSPC, đại diện 500 doanh nghiệp lớn của Mỹ) và Bitcoin (mã BTC-USD) được tải từ Yahoo Finance [@yahoo2026]. VN-Index (chỉ số của Sở Giao dịch Chứng khoán TP. Hồ Chí Minh) không có trên Yahoo Finance nên được tải từ API biểu đồ công khai của ba công ty chứng khoán SSI, DNSE và VNDirect [@ssi2026]. Mỗi file đều được lưu kèm thời điểm tải và mã băm SHA-256 để có thể kiểm tra lại. Dữ liệu chốt đến ngày 30/09/2026.

**Kiểm chứng chéo VN-Index.** Khi đặt các nguồn cạnh nhau, tiểu luận phát hiện ba vấn đề:

- Trên {{v:vnq.overlap_ssi_dnse}} phiên có ở cả SSI và DNSE, giá đóng cửa khớp ở {{v:vnq.agree_pct}} số phiên. {{v:vnq.close_disagreements_ssi_vs_dnse}} phiên còn lại lệch nhau (đều thuộc 2020–2023). Dùng VNDirect làm "trọng tài", cả {{v:vnq.vndirect_votes_for_ssi}}/{{v:vnq.close_disagreements_ssi_vs_dnse}} phiên đều khớp với SSI, nên SSI được chọn làm nguồn chính.
- SSI thiếu dữ liệu từ 20/07 đến 24/08/2009, khiến phiên 25/08/2009 hiển thị một mức "tăng 23%" giả. Lỗ hổng này và các phiên lẻ SSI bỏ sót ({{v:vnq.dnse_rows_filling_ssi_gaps_after_cut}} phiên) được bổ sung từ DNSE.
- Trước tháng 7/2009, chỉ DNSE có dữ liệu, nhưng giá đã bị làm tròn đến số nguyên. Hệ quả là hàng trăm phiên "đứng giá" giả (riêng năm 2003 có 106 phiên) và giá mở cửa, cao nhất, thấp nhất trùng nhau. Vì vậy thực nghiệm chỉ dùng VN-Index từ năm 2010, năm đầy đủ đầu tiên có dữ liệu chất lượng tốt.

Sau khi làm sạch, giai đoạn phân tích của S&P 500, VN-Index và Bitcoin lần lượt có {{v:rows.sp500}}, {{v:rows.vnindex}} và {{v:rows.btc}} phiên. Bảng 1.2 tóm tắt thống kê mô tả, Bảng 1.3 trích các dòng dữ liệu gốc, còn Hình 1.2 và Hình 1.3 cho thấy diễn biến giá và phân phối lợi suất.

{{t:ch1_market_stats:1.2}}

{{t:ch1_market_samples:1.3}}

![Hình 1.2. Diễn biến giá (thang logarit) của ba chỉ số và ranh giới các tập Train – Validation – Test.](../../figures/tieuluan/ch1_market_history.png)

![Hình 1.3. Phân phối lợi suất theo ngày so với phân phối chuẩn có cùng trung bình và độ lệch chuẩn.](../../figures/tieuluan/ch1_return_distribution.png)

**Nhận xét.** Thứ nhất, Bitcoin biến động mạnh hơn hẳn hai chỉ số chứng khoán: độ biến động năm {{v:vol.btc}}, gấp khoảng {{v:vol_ratio}} lần S&P 500 ({{v:vol.sp500}}) và VN-Index ({{v:vol.vnindex}}), đồng thời giao dịch cả cuối tuần nên "một phiên" của Bitcoin là một ngày lịch. Thứ hai, cả ba chuỗi đều có "đuôi béo": độ nhọn dư lớn hơn 0 và Hình 1.3 cho thấy các ngày biến động cực đoan xuất hiện thường xuyên hơn nhiều so với phân phối chuẩn. Đây là lý do mô hình tài chính không nên giả định lợi suất có phân phối chuẩn. Thứ ba, tỉ lệ phiên tăng của cả ba chuỗi đều trên 50%, phản ánh xu hướng tăng dài hạn. Một mô hình "luôn đoán tăng" vì thế đã đạt độ chính xác trên 50% mà không cần học gì. Do đó các chương sau luôn so sánh với đường cơ sở và dùng các thước đo không bị đánh lừa bởi tỉ lệ lớp như ROC-AUC.

## 1.9. Tiểu kết chương

Lịch sử AI để lại ba bài học cho phần còn lại của tiểu luận. Thứ nhất, các chu kỳ "hưng phấn – thất vọng" nhắc nhở rằng thành công trên một bài toán hẹp không đảm bảo thành công ở mọi nơi; tài chính, với tín hiệu yếu và thị trường luôn thích nghi, là nơi bài học này thể hiện rõ nhất. Thứ hai, các bước nhảy vọt đều đến từ sự kết hợp của dữ liệu, sức tính toán và thuật toán, chứ không đến từ thuật toán đơn lẻ; chất lượng dữ liệu, như trường hợp VN-Index ở trên, cũng quan trọng không kém mô hình. Thứ ba, các phương pháp cũ hơn không biến mất mà trở thành nền móng: học máy thống kê vẫn là đường cơ sở mạnh trên dữ liệu bảng, còn ý tưởng hồi quy của LSTM đang được tái sinh trong các kiến trúc mới. Chương 2 bắt đầu từ chính nền móng đó: các kỹ thuật học máy cơ bản.
