# CHƯƠNG 6. HỆ ĐA AGENT

## 6.1. Từ Agent đơn đến hệ đa Agent

Một tác tử (agent) nhận thông tin từ môi trường, duy trì trạng thái và chọn hành động theo mục tiêu. Hệ đa Agent gồm nhiều tác tử trao đổi thông tin để phối hợp hoặc cạnh tranh. Tác tử có thể dựa trên luật, mô hình học máy hoặc mô hình ngôn ngữ; không bắt buộc mỗi tác tử là một LLM riêng [@russell2021].

Trong bài toán của tiểu luận, câu hỏi tài chính thường cần tìm báo cáo, theo dõi nguồn, đọc văn bản và kiểm tra kết quả. Nếu giao toàn bộ cho một lời gọi sinh, lỗi rất khó định vị. Tách vai trò tạo ra các điểm kiểm tra riêng: tác tử truy hồi chịu trách nhiệm danh sách bằng chứng, tác tử đồ thị chịu trách nhiệm quan hệ tài liệu, tác tử viết tạo câu trả lời và tác tử kiểm tra rà soát cấu trúc. AutoGen là một hướng nghiên cứu tổ chức tương tác giữa các tác tử thông qua hội thoại; chương này dùng một minh họa Python nhỏ để làm rõ cơ chế, không phụ thuộc framework đó [@wu2023autogen].

Bảng 6.1. Mức độ tổ chức Agent trong các phần của bài.
| Cách tổ chức | Cơ chế quyết định | Cơ chế trao đổi | Phạm vi đã cài đặt |
|---|---|---|---|
| Hàm RAG đơn | Truy hồi rồi sinh | Truyền đối số | RAGEngine ở Chương 5 |
| Agent dùng công cụ | Luật chọn một công cụ | Trace công cụ | ToolAgent và API cục bộ |
| Hệ tác tử theo vai trò | Bốn tác tử có trách nhiệm riêng | Message, State và điều phối | Mã minh họa Chương 6 |
| Nhóm LLM tự lập kế hoạch | LLM quyết định người nói và hành động | Hội thoại nhiều vòng | Hướng mở rộng, chưa đánh giá |

Không đồng nhất nhiều hàm với hệ tự trị: bản cài đặt ở chương này có luồng phối hợp cố định và bốn lượt xử lý, nên được mô tả chính xác là **hệ đa tác tử theo vai trò, điều phối xác định**. Nó minh họa thông điệp, trách nhiệm và điểm dừng; chưa chứng minh ưu thế của các nhóm LLM tự tổ chức.

## 6.2. Kiến trúc, vai trò và dữ liệu trao đổi

### 6.2.1. Đặc tả bốn tác tử

**RetrievalAgent** nhận câu hỏi, miền và bộ lọc report. Nó gọi hybrid retrieval, lấy tối đa ba đoạn và gửi danh sách id cho tác tử đồ thị. Tác tử không có nhiệm vụ tự suy ra đáp án. Nếu bộ lọc không khớp tài liệu nào, nó giữ danh sách rỗng để các bước sau xử lý thiếu bằng chứng.

**GraphAgent** nhận id các đoạn, tra nút passage tương ứng và lấy các nút cha trong đồ thị tài liệu. Đầu ra graph_context ghi id cùng nguồn cha. Ở bản minh họa, tác tử này bổ sung thông tin truy vết; nó chưa xếp hạng lại kết quả hay cung cấp suy luận nhiều bước cho bộ sinh. Với miền chưa nạp đồ thị, kết quả graph_context có thể rỗng mà luồng RAG vẫn tiếp tục.

**WriterAgent** nhận câu hỏi và bằng chứng để gọi cùng bộ sinh FLAN-T5-small đã dùng ở Chương 5. Dùng chung mô hình giúp làm rõ thay đổi về tổ chức phần mềm; không dùng thêm mô hình lớn để tạo ra một so sánh không công bằng. Nếu không có bằng chứng, tác tử bỏ qua sinh và chuyển trạng thái thiếu bằng chứng.

**VerifierAgent** kiểm tra các id có thuộc corpus đã chọn không, có ít nhất một nguồn và văn bản trả lời có rỗng không. Nó gửi cờ citation_ids_valid, semantic_support_checked và trạng thái cho bộ điều phối. Cờ semantic_support_checked luôn bằng false trong bản này vì chưa cài đánh giá hỗ trợ ngữ nghĩa từng mệnh đề. Câu trả lời vượt kiểm tra cấu trúc vẫn có trạng thái generated_unverified.

Bảng 6.2. Hợp đồng đầu vào–đầu ra và điều kiện thất bại của từng tác tử.
| Vai trò | Đầu vào chính | Đầu ra chính | Điều kiện cần theo dõi |
|---|---|---|---|
| Truy hồi | question, domain, report | evidence và id nguồn | Không có đoạn phù hợp |
| Đồ thị | id đoạn và graph | graph_context | Đoạn không có nút tương ứng |
| Viết | question và evidence | answer | Hết ngữ cảnh hoặc câu trả lời rỗng |
| Kiểm tra | answer, evidence, tập id hợp lệ | status và cờ kiểm tra | Id sai hoặc thiếu bằng chứng |
| Điều phối | Yêu cầu ban đầu | State và danh sách Message | Domain sai, câu hỏi rỗng |

### 6.2.2. Trạng thái dùng chung và thông điệp

`State` chứa question, domain, report, evidence, graph_context, answer, status và messages. `Message` chứa sender, receiver và payload. Mỗi bước ghi lại thông tin cần thiết cho tác tử tiếp theo, đồng thời lưu dấu vết để người xem biết ai tạo đầu ra nào. Không sao chép toàn bộ tài liệu vào mọi Message; phần bằng chứng dùng chung nằm trong State.

Trong môi trường triển khai nhiều tiến trình, State cần được chuyển thành dữ liệu có schema và cơ chế đồng bộ. Bản bài tập chạy tuần tự trong một tiến trình nên không phát sinh cập nhật đồng thời. Đây là chủ ý để dễ giải thích quá trình phối hợp. Mỗi yêu cầu tạo State mới; không dùng chung danh sách messages giữa các yêu cầu, nhờ `default_factory=list` trong dataclass.

![Hình 6.1. Bộ điều phối và các thông điệp của hệ bốn tác tử theo vai trò.](../../figures/tieuluan/complete_agents.png)

## 6.3. Hợp các hàm và điều kiện dừng

Gọi S₀ là trạng thái ban đầu. R, K, W và V lần lượt là phép chuyển trạng thái của tác tử truy hồi, đồ thị, viết và kiểm tra. Hàm O đóng gói trạng thái cuối thành kết quả để trình bày. Hệ có dạng:

$$S_4=(V\circ W\circ K\circ R)(S_0),\qquad output=O(S_4)$$ (6.1)

Mỗi hàm nhận State và cập nhật phần thuộc trách nhiệm của mình. Các Message là kết quả quan sát được sau từng phép chuyển, giúp biến hợp hàm từ một công thức trừu tượng thành chuỗi bước có thể theo dõi. Ví dụ R thêm evidence; K thêm graph_context; W thêm answer; V quyết định status. O không được tự sửa một câu trả lời sai thành “đã xác minh” chỉ vì có nhiều tác tử tham gia.

Điều phối gọi mỗi tác tử đúng một lần. Do đó một yêu cầu hợp lệ có tối đa bốn lượt tác tử ở tầng ứng dụng; số token sinh vẫn bị chặn bởi max_new_tokens ở bộ sinh. Không có vòng lặp “hãy thử lại đến khi đồng ý”. Nếu sau này thêm phản biện và truy hồi lại, phải đặt ngân sách lượt, timeout và tiêu chí dừng rõ ràng, chẳng hạn dừng khi không có bằng chứng mới hoặc đã đạt số lượt tối đa.

### 6.3.1. Giải thích các hàm và phương thức riêng

`State.send(sender, receiver, **payload)` tạo một Message và nối vào danh sách. Hàm chỉ ghi nhật ký thông điệp; việc điều phối thực tế do MultiAgentTeam quyết định, không phải mạng truyền tin phân tán.

`RetrievalAgent.act(state)` gọi `engine.retrieve` với k = 3 và bộ lọc đang có. Nó sao chép id vào payload để tác tử sau biết phạm vi tài liệu đang xử lý. Hàm không đọc gold report từ dữ liệu đánh giá.

`GraphAgent.act(state)` dùng đồ thị của domain. Với mỗi id, nó kiểm tra nút passage có tồn tại, lấy predecessor và ghi số trường hợp khớp. Hàm không suy ra quan hệ bệnh học hay dự báo xu hướng giá từ cạnh tài liệu.

`WriterAgent.act(state)` sinh khi có evidence; nếu evidence rỗng, đặt trạng thái insufficient_evidence. Nó gửi answer cho verifier ngay cả khi answer rỗng để đường lỗi được ghi lại và kiểm tra.

`VerifierAgent.act(state)` so id với tập id đã tạo từ corpus của miền tương ứng. Nếu id không hợp lệ hoặc answer rỗng, nó thay câu trả lời hiển thị bằng “Insufficient evidence”. Khi kiểm tra cấu trúc đạt, giữ generated_unverified, đồng thời ghi rõ chưa kiểm tra hỗ trợ ngữ nghĩa.

`MultiAgentTeam.ask(question, domain, report)` kiểm tra domain và câu hỏi, tạo State mới rồi gọi tuần tự các tác tử. Domain ngoài diabetes/finance hoặc câu hỏi chỉ chứa khoảng trắng gây ValueError trước khi tải tài liệu hay sinh. Đây là kiểm tra đầu vào ở cấp ứng dụng.

`MultiAgentTeam.__init__` gắn các phụ thuộc engine, generator, corpora và graphs cho từng vai trò. Dùng chung generator không làm bốn vai trò biến thành bốn mô hình độc lập; tính đa tác tử nằm ở phân công, giao thức trạng thái và thông điệp trong thiết kế này.

## 6.4. Mã nguồn và biểu diễn luồng phối hợp

Mã đầy đủ của `scripts/tieuluan/complete_agents.py` được chèn ở Phụ lục D. Cách khởi tạo trên corpus và mô hình thật đang có trong dự án:

```python
from deployment.chapter04.app import rag
from src.chapter04.knowledge import load_graph
from scripts.tieuluan.complete_agents import MultiAgentTeam

graphs = {'finance': load_graph(
    'knowledge_graph/chapter04/finance_graph.json.gz')}
team = MultiAgentTeam(rag(), graphs)
state = team.ask('What is the percentage change in revenue?', 'finance')
print(state.status, state.answer)
for message in state.messages:
    print(message.sender, '->', message.receiver, message.payload)
```

Câu hỏi tài chính trong ví dụ chưa chỉ định doanh nghiệp và năm, nên mục tiêu demo là quan sát truy hồi toàn kho và dấu vết trao đổi. Không lấy câu trả lời của ví dụ làm nhận định về một doanh nghiệp cụ thể. Để truy vấn đúng thực thể, người dùng cần chỉ định báo cáo hoặc hệ phải có bước làm rõ danh tính trước khi tính chỉ tiêu.

{{complete:agent_results}}

Các lần chạy dùng corpus, chỉ mục embedding và FLAN-T5-small cục bộ. Kiểm tra yêu cầu bốn Message theo đúng thứ tự; với câu hỏi tài chính, các đoạn khớp đồ thị được ghi trong kết quả. Các trường hợp thiếu report, câu hỏi rỗng và domain sai được kiểm tra riêng. Đây là xác nhận luồng phần mềm và đường lỗi, không phải benchmark chứng minh nhiều tác tử trả lời đúng hơn một tác tử.

## 6.5. Đánh giá hệ đa Agent và phân tích lỗi

### 6.5.1. Cần đo những gì?

Đánh giá kỹ thuật cần đo số lượt tác tử, độ trễ đầu cuối, số token, nguồn được truy vết và tỷ lệ từ chối đúng. Đánh giá nội dung cần tập câu hỏi độc lập, đáp án chuẩn, bằng chứng chuẩn và tiêu chí hỗ trợ từng mệnh đề. Với tài chính cần xác định sai số số học, đơn vị và niên độ; với câu hỏi dữ liệu bệnh cần phân biệt số người với số lượt khám.

So sánh có ý nghĩa là cùng mô hình sinh, cùng corpus, cùng câu hỏi và ngân sách tài nguyên: RAG đơn → Agent công cụ → đa Agent. Một phép loại bỏ thành phần có thể tắt GraphAgent để xem mất khả năng truy vết nào; không mặc định độ chính xác giảm vì graph hiện chỉ thêm metadata. Nếu muốn cải thiện chất lượng, cần đưa kiểm tra niên độ, đơn vị hoặc bằng chứng vào quyết định, rồi đo trên tập giữ lại.

### 6.5.2. Các nhóm lỗi cần giải thích khi bảo vệ

**Lỗi truy hồi.** Đoạn đúng không có trong top k. Các tác tử sau không thể khôi phục một con số chưa được cung cấp. Biện pháp kiểm tra là so evidence id với gold và phân biệt lỗi chọn báo cáo với lỗi chọn đoạn.

**Lỗi nguồn.** Id có thật nhưng đoạn không hỗ trợ câu trả lời. Verifier cấu trúc hiện tại chưa phát hiện được trường hợp này. Cần một kiểm tra mệnh đề–bằng chứng hoặc chấm thủ công; thêm một nhãn “Verifier” không tự tạo ra năng lực xác minh.

**Lỗi số học.** Bộ sinh nhầm dấu, tỷ lệ hoặc đơn vị. Calculator có thể xử lý phép toán nhưng chỉ khi nhận đúng toán hạng. Cần lưu từng số gốc, nguồn và phép chuyển đổi, ví dụ USD triệu sang USD tỷ, trước khi tính.

**Lỗi phối hợp.** Hai tác tử dùng khác report, hoặc câu trả lời được tạo trước khi bằng chứng cập nhật. State chung và luồng tuần tự giảm loại lỗi này trong demo. Khi chuyển sang chạy song song, cần khóa phiên bản bằng chứng và schema thông điệp, không chỉ tăng số tiến trình.

**Lỗi đánh giá.** Dùng câu hỏi đã điều chỉnh prompt để kết luận trên dữ liệu mới, dùng oracle report như truy hồi toàn kho hoặc dùng kết quả computed của calculator để thay cho kết quả generator. Báo cáo giữ riêng các điều kiện để tránh trộn các năng lực khác nhau.

## 6.6. Triển khai, demo và tổ chức trình bày nhóm

### 6.6.1. Kịch bản demo có thể tái lập

Bước một, giới thiệu đầu vào và phạm vi dữ liệu bằng sơ đồ tổng thể. Chọn một trong hai miền diabetes hoặc finance; nếu có báo cáo cụ thể, ghi rõ bộ lọc report do người dùng cung cấp.

Bước hai, chạy `python -m scripts.tieuluan.verify_complete` trong môi trường dự án. Phần đầu xác nhận chín mô hình sáu lớp; phần sau khởi tạo hệ bốn tác tử trên mô hình và corpus cục bộ. Đọc tệp kết quả `results/tieuluan/complete/verification.json` để thấy dữ liệu đầu vào, id bằng chứng, Message và trạng thái cuối.

Bước ba, đối chiếu một trace thành công về cấu trúc với trường hợp report không tồn tại. Ở trường hợp sau, phải thấy insufficient_evidence thay vì một câu trả lời được bịa. Sau đó thử domain sai và câu hỏi rỗng để chỉ ra kiểm tra đầu vào.

Bước bốn, mở Swagger của Agent công cụ Chương 5 để biểu diễn thống kê hoặc tính toán chính xác. Nêu rõ đây là API ToolAgent đã triển khai; bản MultiAgentTeam được gọi bằng Python trong demo Chương 6, chưa được tích hợp thành endpoint riêng. Không gọi URL `/health` là bằng chứng rằng hệ đa Agent đã chạy.

### 6.6.2. Phân công trình bày và họp nhóm

Phần này là **kế hoạch tổ chức và tiêu chí trình bày**, không phải biên bản xác nhận các cuộc họp đã diễn ra. Tên và số thành viên do nhóm điền theo tình hình thực tế; tài liệu không tự tạo danh tính hay thời gian họp.

Bảng 6.3. Phân công theo nhóm công việc để mọi phần đều được giải thích.
| Nhóm công việc | Nội dung phụ trách | Kết quả mang đến buổi họp | Câu hỏi phải giải thích được |
|---|---|---|---|
| Dữ liệu và nền tảng | Chương 1–2, nguồn dữ liệu, chia tập | Sơ đồ dữ liệu và bảng chất lượng | Vì sao không chia ngẫu nhiên chuỗi? |
| CNN | Chương 3 và ba bản sáu lớp | Bảng shape, hợp hàm, log đối chiếu | ReLU/Flatten có được đếm là lớp học? |
| RNN–LSTM | Chương 4 và BPTT | Ví dụ cổng, tensor qua từng lớp | Chiều thời gian khác chiều sâu thế nào? |
| KG–RAG | Chương 5, schema và truy hồi | Một truy vấn nguồn và bảng chỉ số | Hit@5 cao có bảo đảm đáp án đúng? |
| Đa Agent và demo | Chương 6, Message và điều phối | Trace thành công/lỗi, kịch bản chạy | Verifier hiện kiểm tra được điều gì? |

Họp chuẩn bị thứ nhất chốt schema, đường dẫn dữ liệu và cách đếm lớp; sản phẩm là bảng yêu cầu và giao diện hàm thống nhất. Họp thứ hai đối chiếu cùng mẫu đầu vào giữa ba framework và ghép luồng KG–RAG–Agent; ghi lại khác biệt trọng số, kiểu tensor và lỗi tích hợp. Họp thứ ba chạy thử bảo vệ với giới hạn thời gian, đổi người hỏi chéo giữa các nhóm công việc và kiểm tra khả năng chạy lại từ lệnh đã ghi. Chỉ khi thực sự tổ chức mới lập biên bản kèm ngày, người tham gia và việc đã hoàn thành.

### 6.6.3. Hợp các hàm như một cách giải thích chung

Khi báo cáo, mỗi nhóm dùng cùng mẫu: nêu đầu vào → viết hợp hàm → tách từng hàm → tính ví dụ nhỏ → chỉ đoạn code → đọc kết quả và giới hạn. CNN minh họa hợp các phép biến đổi ảnh; RNN/LSTM minh họa hợp theo lớp và cập nhật theo thời gian; RAG minh họa hợp các bước trên dữ liệu khác kiểu; đa Agent minh họa hợp các phép chuyển trạng thái có thông điệp.

Ví dụ câu hỏi kiểm tra chéo: nếu bỏ sigmoid ở lúc hiển thị thì logit 2 có phải xác suất 200% không? Nếu trả `return_sequences=False` quá sớm thì lớp sau nhận sai tensor thế nào? Nếu một citation id tồn tại thì vì sao câu trả lời vẫn có thể sai? Nếu hai tác tử cùng đồng ý về một số không xuất hiện trong nguồn thì đó có phải xác minh không? Mỗi câu buộc người trình bày liên hệ công thức, hàm và hành vi thực tế.

## 6.7. Tiểu kết chương

Hệ bốn tác tử làm rõ việc phân công và truy vết: truy hồi lấy đoạn, đồ thị bổ sung nguồn, bộ viết tạo văn bản và bộ kiểm tra xác nhận cấu trúc. Minh họa có điểm dừng xác định, thông điệp và trạng thái lỗi rõ ràng. Chất lượng nội dung vẫn bị giới hạn bởi truy hồi và bộ sinh, trong khi kiểm tra ngữ nghĩa chưa được triển khai. Giá trị của phần cài đặt là tạo một cấu trúc có thể kiểm tra và mở rộng; hiệu quả của hệ tự lập kế hoạch hoặc phản biện nhiều vòng cần thực nghiệm riêng.
