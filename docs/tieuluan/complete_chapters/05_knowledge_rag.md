# CHƯƠNG 5. KNOWLEDGE GRAPH VÀ RAG

## 5.1. Mở đầu

### 5.1.1. Knowledge Graph, Knowledge Base và RAG là gì?

Các chương trước học một ánh xạ từ dữ liệu sang nhãn. Khi cần trả lời “chỉ tiêu này xuất hiện trong báo cáo nào?” hay “có bao nhiêu lượt nhập viện tái nhập trong 30 ngày?”, một xác suất do CNN hoặc LSTM tạo ra chưa giải quyết được nhu cầu tra cứu. Hệ thống cần tổ chức tri thức, tìm bằng chứng và trả lời theo nguồn. Chương này bổ sung ba thành phần: cơ sở tri thức, đồ thị tri thức và sinh câu trả lời có truy hồi.

**Knowledge Base (KB)** là kho tri thức có tổ chức: có thể gồm bảng, luật, tài liệu và đồ thị. **Knowledge Graph (KG)** biểu diễn các thực thể và quan hệ giữa chúng dưới dạng đồ thị. Ví dụ, bộ ba (Công ty A, công bố, Báo cáo 2023) nối hai thực thể bằng một quan hệ có tên. KG là một cách tổ chức KB; hai thuật ngữ liên quan nhưng không đồng nghĩa hoàn toàn [@hogan2021kg].

**Retrieval-Augmented Generation (RAG)** kết hợp truy hồi bằng chứng từ kho ngoài với mô hình sinh văn bản. Câu hỏi được dùng để tìm các đoạn liên quan; các đoạn này đi vào ngữ cảnh để mô hình tạo câu trả lời. Tri thức không chỉ nằm trong trọng số mô hình: có thể cập nhật kho tài liệu mà không huấn luyện lại toàn bộ bộ sinh [@lewis2020rag].

### 5.1.2. Vai trò và ý nghĩa của KB và RAG

KB cung cấp dữ liệu đã tổ chức; KG giúp đi theo quan hệ và nguồn gốc; RAG giúp hỏi bằng ngôn ngữ tự nhiên. Trong bài toán báo cáo tài chính, đồ thị hỗ trợ xác định công ty, niên độ và báo cáo, còn truy hồi tìm dòng bảng hoặc đoạn giải thích. Trong bài toán bệnh tiểu đường, bảng dữ liệu hỗ trợ thống kê lượt khám; tài liệu nguồn giải thích ý nghĩa trường dữ liệu. Cách chọn công cụ phải phù hợp với câu hỏi: đếm chính xác nên dùng mã thống kê, thay vì yêu cầu mô hình ngôn ngữ đoán số.

Bảng 5.1. Phân biệt các thành phần của hệ hỏi đáp có căn cứ.
| Thành phần | Dữ liệu đầu vào | Đầu ra | Vai trò trong tiểu luận |
|---|---|---|---|
| KB | Tài liệu, bảng và metadata | Kho tri thức có nguồn | Lưu corpus và bảng thống kê |
| KG | Thực thể, quan hệ, thuộc tính | Đồ thị truy vấn được | Liên kết công ty–báo cáo–đoạn trích |
| Retriever | Câu hỏi và chỉ mục | Danh sách đoạn xếp hạng | BM25, dense và hybrid |
| Generator | Câu hỏi cùng bằng chứng | Văn bản trả lời | FLAN-T5-small chạy CPU |
| Calculator | Phép toán và toán hạng có đơn vị | Số và công thức | Tách tính toán khỏi sinh văn bản |
| Agent | Yêu cầu có cấu trúc | Chọn công cụ và trace | Triển khai ở cuối chương và Chương 6 |

Tiểu luận thống nhất số chương theo đề cương tổng thể: CNN ở Chương 3, RNN–LSTM ở Chương 4, KG–RAG ở Chương 5 và Hệ đa Agent ở Chương 6. Thư mục mã `src/chapter04` giữ tên lịch sử của bài thực hành “Knowledge & RAG”; đường dẫn này không làm thay đổi số chương của bản báo cáo.

## 5.2. Knowledge Graph và cơ sở tri thức

### 5.2.1. Mô hình dữ liệu, thực thể và quan hệ

KG có thể viết G = (V, E), trong đó V là tập nút và E là tập cạnh có nhãn. Thuộc tính của nút mô tả thực thể, thuộc tính cạnh mô tả mối liên hệ. Trong hệ minh họa, nút Company có ticker; Report có year, filename và source_url; Passage hoặc TableRow có chunk_id và text. Mã định danh ổn định giúp phân biệt hai báo cáo cùng tên công ty nhưng khác niên độ.

Ví dụ minh họa cấu trúc, không phải số liệu kinh doanh thật: (company:A, PUBLISHED, report:A:2023) và (report:A:2023, HAS_PASSAGE, passage:A:2023:row_5). Câu hỏi về công ty A năm 2023 cần đi đến đúng Report trước khi tìm row_5. Một giá trị doanh thu không được đặt trên nút Company mà thiếu năm và đơn vị, vì cùng một doanh nghiệp có nhiều giá trị theo thời gian.

Kho bệnh tiểu đường có hai lớp dữ liệu: đồ thị tài liệu và đồ thị lượt khám. Ở đồ thị lượt khám, Patient nối Encounter bằng HAS_ENCOUNTER; Encounter nối DiagnosisCode bằng HAS_RECORDED_CODE và nối Dataset bằng FROM_DATASET. Quan hệ “có mã chẩn đoán được ghi” mô tả dữ liệu quan sát, không suy ra quan hệ nhân quả giữa hai bệnh.

### 5.2.2. Quy trình xây dựng và quản lý nguồn gốc

Quy trình gồm xác định câu hỏi → thiết kế schema → đọc dữ liệu → chuẩn hóa mã → tạo nút/cạnh → kiểm tra ràng buộc → lưu đồ thị. Tệp `src/chapter04/knowledge.py` cài các hàm `build_document_graph`, `build_encounter_graph`, `save_graph` và `load_graph` bằng NetworkX. Đồ thị được lưu JSON nén gzip; bản minh họa không yêu cầu máy chủ Neo4j.

Đối với báo cáo, mã `report_key` kết hợp định danh báo cáo với hash nội dung để tránh trộn tài liệu cùng tên. `build_finance_chunks` đưa cả tiêu đề cột vào dòng bảng, nhờ đó một ô số đi kèm tên chỉ tiêu và năm. Metadata tối thiểu gồm id, domain, report, source_url và locator. Nếu bỏ locator, đường dẫn về cả kho GitHub vẫn khó giúp người đọc tìm đúng dòng bằng chứng.

Với dữ liệu thiếu, dấu “?” được nhận diện là giá trị thiếu. Không tạo nút DiagnosisCode có mã “?” rồi xem đó là một bệnh mới. Với tập dữ liệu có nhiều lượt khám của cùng bệnh nhân, số lượt khám và số người phải được tính riêng. Để kiểm tra, lấy một Patient có nhiều Encounter và truy vết các cạnh, sau đó đối chiếu với dòng nguồn.

### 5.2.3. Truy vấn và giới hạn suy luận

Một truy vấn đồ thị có thể lấy tất cả Report là nút kế của Company, rồi lấy Passage là nút kế của Report. Ví dụ Cypher dưới đây diễn tả cùng ý tưởng nếu chuyển schema sang Neo4j; mã thực tế của chương sử dụng NetworkX:

```cypher
MATCH (c:Company {ticker: $ticker})-[:PUBLISHED]->(r:Report)
MATCH (r)-[:HAS_PASSAGE]->(p:Passage)
WHERE r.year = $year
RETURN r.filename, p.chunk_id, p.text, p.source_url
LIMIT 10
```

Truy vấn phải truyền tham số thay vì nối trực tiếp chuỗi người dùng. Schema tài chính hiện tại chủ yếu tổ chức tài liệu; nó chưa tự động trích xuất mọi sự kiện doanh thu, lợi nhuận và quan hệ sở hữu. Vì vậy đây là KG tài liệu có truy vết nguồn, chưa phải đồ thị tri thức tài chính đầy đủ. Không suy diễn rằng cạnh HAS_PASSAGE chứng minh một nhận định kinh doanh nào đó.

## 5.3. RAG: truy hồi, sinh và đánh giá

### 5.3.1. Hai luồng xử lý

Luồng lập chỉ mục chạy khi nạp hoặc cập nhật dữ liệu: đọc tài liệu, chia đoạn, gắn metadata, tạo chỉ mục từ khóa, tính embedding và lưu bộ nhớ đệm. Luồng trả lời chạy cho từng câu hỏi: kiểm tra đầu vào, chọn miền, lọc phạm vi nếu có, truy hồi, ghép ngữ cảnh, sinh câu trả lời và đính kèm nguồn.

Gọi I là hàm tạo chỉ mục của kho D; R là truy hồi; B là dựng prompt; G là sinh; A là đóng gói đáp án. Với I(D) đã có, hệ trả lời được biểu diễn bằng hợp hàm:

$$E_q=R(q,I(D)),\qquad y=A\left(G\left(B(q,E_q)\right),E_q\right)$$ (5.1)

R trả về danh sách bằng chứng, B trả về văn bản đầu vào, G trả về văn bản sinh và A trả về cấu trúc chứa answer, citations và status. Việc phân biệt kiểu dữ liệu này giúp giải thích từng hàm thay vì gọi toàn bộ hệ thống là một “mô hình RAG” khó kiểm tra.

![Hình 5.1. Luồng lập chỉ mục và luồng trả lời của hệ KG–RAG trong tiểu luận.](../../figures/tieuluan/complete_rag.png)

### 5.3.2. BM25, dense retrieval và hợp nhất thứ hạng

**BM25** dùng tần suất từ trong đoạn, độ hiếm của từ trong kho và độ dài đoạn để chấm điểm. Mã tự viết tạo postings từ token tới các đoạn chứa token. Nó phù hợp khi câu hỏi chứa đúng tên công ty, tên chỉ tiêu hoặc cụm từ xuất hiện trong tài liệu. Với từ chưa xuất hiện trong chỉ mục, phương pháp không tự hiểu các cách diễn đạt đồng nghĩa.

**Dense retrieval** đưa câu hỏi và đoạn vào cùng không gian vector bằng all-MiniLM-L6-v2. Mã lấy trung bình có attention mask trên trạng thái token rồi chuẩn hóa L2. Khi vector đã có độ dài một, tích vô hướng bằng cosine similarity. Truy hồi dense có thể đưa các đoạn gần về nghĩa lên cao nhưng không bảo đảm phân biệt chính xác các con số hoặc niên độ gần nhau [@minilm_model].

**Hybrid** lấy 30 ứng viên từ mỗi nhánh rồi dùng Reciprocal Rank Fusion (RRF). Điểm của đoạn d là tổng nghịch đảo hạng đã làm trơn:

$$s(d)=\sum_{j\in\{lexical,dense\}}\frac{1}{60+rank_j(d)}$$ (5.2)

Đoạn không có trong một danh sách không nhận đóng góp từ danh sách đó. Ví dụ d ở hạng 2 của BM25 và hạng 5 của dense có điểm 1/62 + 1/65, xấp xỉ 0,03151. Hợp nhất hạng tránh cộng trực tiếp hai loại điểm có thang đo khác nhau; nó không khẳng định hai hệ truy hồi có độ tin cậy giống nhau [@rrf2009].

### 5.3.3. Dựng ngữ cảnh và sinh câu trả lời

`build_prompt` đặt câu hỏi trước ngữ cảnh, kèm yêu cầu chỉ trả lời theo nguồn và nói unknown khi không có đáp án. Bộ sinh `google/flan-t5-small` chạy cục bộ, tối đa 512 token đầu vào và 64 token sinh mới, không lấy mẫu ngẫu nhiên. Các cấu hình này đọc trực tiếp từ `src/chapter04/rag.py`; việc đặt câu hỏi trước giúp giảm rủi ro mất câu hỏi khi cắt ngữ cảnh dài [@flan_model].

`RAGEngine.answer` lấy ba đoạn, gọi bộ sinh rồi đính kèm id, source_url và locator. Khi không có bằng chứng, hệ trả `insufficient_evidence`. Khi có câu trả lời được sinh, trạng thái là `generated_unverified`. Nguồn đính kèm là các đoạn đã truy hồi, chưa phải ánh xạ từng mệnh đề trong câu trả lời tới đoạn chứng minh. Kiểm tra id hợp lệ chỉ xác nhận nguồn tồn tại, không xác nhận câu trả lời được nguồn hỗ trợ.

### 5.3.4. Chỉ số và thiết kế thí nghiệm

Hit@k đo có ít nhất một đoạn đúng trong k kết quả; Recall@k đo phần bằng chứng chuẩn đã thu hồi; MRR@k đo vị trí của kết quả đúng đầu tiên. Với một câu cần hai đoạn chuẩn và truy hồi chỉ có một đoạn, Hit@5 vẫn bằng một nhưng Recall@5 chỉ bằng 0,5. Complete@5 yêu cầu có đủ tất cả đoạn chuẩn nên phù hợp hơn khi câu hỏi cần nhiều toán hạng.

Phần sinh được đo riêng: Exact Match so chuỗi chuẩn hóa; token-F1 đo mức trùng token; numeric accuracy kiểm tra kết quả số theo quy trình đánh giá đã lưu. So sánh ba điều kiện: không ngữ cảnh, ngữ cảnh chuẩn và RAG thực tế. “Gold context oracle” cung cấp bằng chứng chuẩn trực tiếp để chẩn đoán lỗi bộ sinh, không đại diện một hệ triển khai phải tự tìm tài liệu.

### 5.3.5. Giải thích từng hàm trong mã KG–RAG

Các hàm dưới được nhóm theo luồng dữ liệu để người trình bày có thể nối từ công thức đến mã. Đầu ra mỗi hàm là đầu vào của bước kế; chỉ bộ sinh trả văn bản, các bước truy hồi chủ yếu trả cấu trúc dữ liệu và danh sách bằng chứng.

- `tokenize(text)` chuyển chữ về dạng thường và tách token theo biểu thức chính quy. Bản hiện tại ưu tiên chữ/số Latin nên không được mặc định là bộ tách từ tiếng Việt hoàn chỉnh.
- `report_key(item)` tạo id báo cáo từ tên và hash nội dung. Hàm giúp phân biệt các ngữ cảnh báo cáo khác nhau, không chấm điểm mức liên quan của tài liệu.
- `build_finance_chunks(items, table_headers=True)` nhận các mẫu FinQA, tách đoạn văn và dòng bảng, loại đoạn trùng id, gắn công ty/năm/nguồn/locator. Khi bật table_headers, ô số đi kèm tiêu đề cột.
- `BM25.__init__(chunks)` tạo độ dài đoạn, độ dài trung bình và chỉ mục đảo postings. `BM25.search(query, k, allowed)` tính điểm theo các token câu hỏi, lọc id được phép rồi trả top k; allowed không phải nhãn đáp án.
- `Encoder.encode(texts, batch_size)` tokenize thành tensor, chạy mô hình embedding, lấy trung bình có attention mask và chuẩn hóa L2. Đầu ra là ma trận số đoạn × số chiều embedding, không phải câu trả lời.
- `DenseRetriever.__init__` tính hoặc đọc embedding cache; hash corpus giúp tránh dùng cache không đúng nội dung. `DenseRetriever.search` tính tích vô hướng với vector câu hỏi, áp dụng bộ lọc id và trả danh sách xếp hạng.
- `reciprocal_rank_fusion(rankings, k, c)` nhận các danh sách id và cộng 1/(c + hạng), bỏ qua đóng góp của danh sách không chứa id đó. `hybrid_search` lấy ứng viên từ BM25 và dense, hợp nhất hạng rồi khôi phục nội dung các đoạn.
- `retrieval_metrics(rankings, golds)` so danh sách truy hồi với id bằng chứng chuẩn, tính Hit@1/5, Recall@5, MRR@5 và Complete@5. Hàm dùng nhãn chuẩn để đánh giá, không được đưa gold vào luồng trả lời người dùng.
- `build_prompt(question, evidence)` ghép yêu cầu trả lời, câu hỏi và nội dung bằng chứng. Khi evidence là None, nó dựng điều kiện đối chứng không ngữ cảnh; đây là một điều kiện khác với corpus không có kết quả phù hợp.
- `LocalGenerator.generate(question, evidence)` dựng prompt, tokenize tối đa 512 token, sinh tối đa 64 token và decode. Hàm trả văn bản; nó không tự chứng minh từng mệnh đề được bằng chứng hỗ trợ.
- `RAGEngine.retrieve(question, domain, k, report, method)` chọn chỉ mục miền, tạo tập id được phép theo report nếu có và gọi BM25/dense/hybrid. `RAGEngine.answer` gọi retrieve, xử lý rỗng, gọi generate và đóng gói answer, nguồn cùng status.
- `normalized_answer(text)` chuẩn hóa chuỗi để so khớp. `token_f1(prediction, reference)` đếm token chung có xét số lần xuất hiện, tính precision/recall rồi trung bình điều hòa; token-F1 không kiểm tra đầy đủ số học hoặc ngữ nghĩa.
- `build_document_graph(chunks, domain)` tạo các nút và cạnh nguồn tài liệu. `build_encounter_graph(frame)` tạo Patient–Encounter–DiagnosisCode từ dòng dữ liệu; không tự suy ra quan hệ nhân quả hay chỉ định điều trị.
- `save_graph(graph, path)` ghi JSON node-link nén gzip; `load_graph(path)` đọc lại cùng cấu trúc. Hai hàm phục vụ lưu trữ, không huấn luyện embedding đồ thị.
- `safe_calculate(operation, values, units)` kiểm tra phép toán/toán hạng/đơn vị, tính kết quả và trả value, formula, unit. Nguồn của từng toán hạng vẫn phải do bên gọi cung cấp, hàm không tự trích số từ văn bản.

## 5.4. KB và RAG cho bệnh tiểu đường

### 5.4.1. Dữ liệu và câu hỏi

Dữ liệu UCI Diabetes 130-US Hospitals gồm các lượt điều trị nội trú trong giai đoạn 1999–2008 [@uci_diabetes296]. Kết quả xử lý cục bộ ghi nhận 101.766 lượt khám, 71.518 bệnh nhân phân biệt và 50 cột. Nhãn tái nhập viện có 54.864 lượt NO, 35.545 lượt >30 và 11.357 lượt <30. Đây là số liệu toàn bộ bảng; đồ thị minh họa chỉ lấy mẫu 5.000 lượt với seed 42, trong đó có 570 lượt <30. Hai con số 570 và 11.357 thuộc hai phạm vi khác nhau.

Corpus hỏi đáp gồm 183 đoạn; bộ câu hỏi minh họa gồm 16 câu tự xây dựng. Câu “How many encounters are in the dataset?” cần lấy thống kê từ bảng. Câu hỏi giải thích trường `readmitted` cần truy hồi mô tả bộ dữ liệu. Hệ không dùng những câu hỏi này để suy ra phác đồ hay hướng dẫn điều trị cá nhân.

### 5.4.2. Kết quả và phân tích

{{complete:diabetes_tables}}

Hybrid tìm được ít nhất một đoạn đúng trong top 5 cho 14/16 câu, nhưng Exact Match của toàn bộ câu trả lời chỉ 5/16. Khi đưa ngữ cảnh chuẩn, Exact Match tăng lên 9/16; không ngữ cảnh chỉ đạt 1/16. Khoảng cách giữa truy hồi và sinh cho thấy phải đánh giá hai khâu riêng. Token-F1 cao hơn Exact Match không có nghĩa mọi chi tiết số đều đúng.

Với câu hỏi đếm, dùng `ToolAgent` nhánh statistics cho kết quả xác định: 11.357 lượt tái nhập dưới 30 ngày. Nó trả trạng thái computed và trace dataset_statistics. Câu trả lời được lấy bằng truy vấn dữ liệu, không gán cho chất lượng suy luận của FLAN-T5-small. Bộ 16 câu quá nhỏ và do người triển khai biên soạn nên kết quả chỉ minh họa kỹ thuật, chưa đủ chứng minh hiệu quả trong môi trường lâm sàng.

## 5.5. KB và RAG cho chứng khoán, báo cáo tài chính

### 5.5.1. Dữ liệu FinQA và biểu diễn bảng

FinQA cung cấp câu hỏi cần suy luận số trên báo cáo doanh nghiệp, kèm văn bản, bảng và chương trình tính đáp án [@finqa2021]. Bản dữ liệu cục bộ có 6.251 mẫu train, 883 dev và 1.147 test. Thí nghiệm của chương đánh giá truy hồi trên **883 câu dev**, tạo 9.177 đoạn thuộc 299 báo cáo phân biệt. Đây là tập phát triển để phân tích phương pháp; không trình bày kết quả như một đánh giá cuối cùng trên test độc lập.

Hai cách chia đoạn được so sánh: dòng bảng thô và dòng bảng gắn tiêu đề cột. Ví dụ minh họa, dòng “2022: 100; 2023: 120” giữ được ngữ cảnh thời gian tốt hơn chuỗi “100 | 120”. Thông tin metadata về công ty, niên độ và id báo cáo được lưu cùng đoạn để có thể truy vết.

### 5.5.2. Kết quả truy hồi và điều kiện oracle

{{complete:finance_tables}}

Hybrid toàn kho đạt Hit@5 khoảng 68,29%, so với khoảng 51,98% của BM25 trên dòng thô. BM25 gắn tiêu đề cột đạt khoảng 61,95%, thể hiện ích lợi của tiền xử lý bảng ngay cả khi chưa đổi mô hình embedding. Khi cung cấp sẵn báo cáo đúng, hybrid đạt khoảng 92,64%. Mức này là **oracle_report**: biết sẵn báo cáo đúng theo dữ liệu chuẩn, nên không được dùng thay cho kết quả tìm kiếm toàn kho của người dùng mới.

Complete@5 thấp hơn Hit@5 vì nhiều câu cần nhiều bằng chứng. Điều này có ý nghĩa trực tiếp với tính toán: tìm được doanh thu năm nay mà thiếu năm trước chưa đủ để tính tăng trưởng. Có thể dùng KG để lọc báo cáo khi người dùng thật sự cung cấp công ty và niên độ, nhưng bộ định tuyến không được âm thầm lấy report chuẩn từ nhãn đánh giá.

### 5.5.3. Sinh câu trả lời và tính toán có kiểm soát

Thử nghiệm sinh trên 24 câu được lưu trong `finance_rag_results.json`. Numeric accuracy bằng 0/24 ở cả ba điều kiện không ngữ cảnh, ngữ cảnh chuẩn và hybrid RAG trong báo cáo chuẩn. Đây là một kết quả âm: mô hình sinh nhỏ chưa giải được bài toán suy luận số trong mẫu thử, ngay cả khi đã có ngữ cảnh đúng. Không lấy Hit@5 cao để tuyên bố hệ đã trả lời tài chính chính xác.

Giải pháp kỹ thuật được minh họa là tách phép tính sang `safe_calculate`. Với số giả định doanh thu từ 100 lên 120, growth trả ((120 − 100)/100) × 100 = 20%. Hàm kiểm tra phép toán nằm trong danh sách cho phép, hai toán hạng hữu hạn, mẫu số khác không và các đơn vị phù hợp. Không dùng eval để chạy mã tự do từ văn bản.

Trong bản hiện tại, người gọi phải truyền toán hạng có cấu trúc; hệ chưa tự động trích số đáng tin cậy từ đoạn báo cáo. Vì thế việc máy tính cho kết quả đúng khi được nhập đúng số không sửa được kết quả 0/24 của bộ sinh. Một hệ mở rộng phải có bước xác định chỉ tiêu–niên độ–đơn vị–nguồn cho từng toán hạng và được đánh giá đầu cuối trên câu hỏi chưa từng dùng để phát triển.

RAG trên FinQA trả lời về tài liệu doanh nghiệp, còn CNN/RNN dự báo hướng thị trường. Đây là hai bài toán khác nhau về đầu vào, mục tiêu và chỉ số. Không dùng độ chính xác trả lời FinQA để suy ra khả năng chọn cổ phiếu hoặc lợi nhuận giao dịch.

## 5.6. Deploy với Agent

Ứng dụng FastAPI ở `deployment/chapter04/app.py` cung cấp `/health`, `/docs` và POST `/api/v1/knowledge/ask`. Bộ định tuyến `ToolAgent.resolve_task` chọn một trong bốn công cụ: statistics, calculate, graph và rag. Domain chỉ nhận diabetes hoặc finance; câu hỏi rỗng, phép toán sai và yêu cầu không hợp lệ bị từ chối. Bản triển khai đã có log HTTP cục bộ trong `results/chapter04/agent_deployment_results.json`; log ghi rõ tiến trình kiểm tra đã được dừng sau thử nghiệm.

```powershell
.venv/Scripts/python.exe -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8044
```

Mở http://127.0.0.1:8044/docs, thử yêu cầu có cấu trúc:

```json
{
  "domain": "diabetes",
  "question": "How many encounters were readmitted within 30 days?",
  "task": "statistics"
}
```

Kết quả dự kiến từ bảng đã xử lý là answer = “11357”, status = computed, nguồn uci296 và trace cho công cụ thống kê. Tiếp theo thử calculate với operation = growth, values = [100, 120], units = ["USD", "USD"], rồi thử mẫu số bằng không để thấy đường xử lý lỗi. Với RAG, phải xem cả evidence và trạng thái generated_unverified.

Gói Docker có sẵn để đóng gói môi trường nhưng kết quả xác nhận của bài là triển khai HTTP cục bộ, chưa khẳng định đã triển khai dịch vụ KG–RAG trên cloud. Agent ở bước này là một bộ định tuyến xác định; Chương 6 mở rộng thành các tác tử có vai trò, trạng thái và thông điệp riêng.

## 5.7. Tiểu kết chương

KG tổ chức quan hệ và nguồn gốc; truy hồi chọn bằng chứng; bộ sinh diễn đạt câu trả lời. Các kết quả cho thấy hybrid và tiền xử lý bảng cải thiện khả năng tìm bằng chứng trong thiết kế này, nhưng câu trả lời số vẫn là nút thắt. Phân tách chức năng thành các hàm có kiểu dữ liệu rõ ràng giúp xác định lỗi nằm ở chọn miền, tìm báo cáo, tìm đoạn, sinh hay tính toán, đồng thời tạo nền tảng cho hệ đa Agent ở chương tiếp theo.
