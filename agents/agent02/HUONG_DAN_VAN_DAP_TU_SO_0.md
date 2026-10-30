# HƯỚNG DẪN VẤN ĐÁP AGENT02 TỪ SỐ 0

Tài liệu này dành cho người mới bắt đầu và cần giải thích bài trong một phút. Mục tiêu là hiểu để tự nói được bằng lời của mình; không học thuộc một câu khẳng định vượt quá số liệu. Bài có hai tầng: bài lab gốc so sánh Q-learning với luật; phần nghiên cứu mở rộng xây người bán có tồn kho, học nhu cầu và phản ứng đối thủ, rồi lập kế hoạch bán trong một mùa hữu hạn. Hai tầng dùng môi trường và đơn vị báo cáo khác nhau. Đừng lấy lợi nhuận mỗi bước của bài cũ so trực tiếp với lợi nhuận cả mùa của bài mới.

## 1. Nếu chỉ còn một phút trước khi thầy hỏi

Có thể nói đoạn sau trong khoảng một phút, tùy tốc độ nói:

> Bài của em nghiên cứu cách người bán chọn giá khi có đối thủ. Bài gốc dùng Q-learning để học từ lợi nhuận, so với người bán làm theo luật. Em thực hiện các thí nghiệm về tốc độ học, khám phá và đánh giá chính sách sau khi học. Kết quả một seed không chứng minh RL luôn hơn luật: ở baseline cả hai cùng giữ giá, phần chênh lợi nhuận đo được chủ yếu là nhiễu. Em còn phân tích việc quan sát ban đầu thiếu lịch sử hành động của chính mình. Phần mở rộng đưa thêm tồn kho hữu hạn, thời gian còn lại, nhu cầu có tính mùa và đối thủ có trí nhớ. Tác tử học mô hình từ doanh số quan sát được, dự báo các khả năng rồi dùng quy hoạch động để cân bằng bán ngay với giữ hàng cho tương lai. Em đánh giá trên seed riêng, đóng băng chính sách, so với luật mạnh, Q-learning cùng thông tin và oracle. Đây là hệ mô phỏng và ứng dụng các phương pháp có sẵn, chưa phải triển khai thị trường thật hay thuật toán mới được chứng minh vượt mọi đối thủ.

Nếu bị ngắt sau 20 giây, giữ ba ý: **bài toán chọn giá**, **học để tối ưu lợi nhuận dài hạn**, **so sánh công bằng và không tuyên bố luôn thắng**. Nếu thầy hỏi đúng một phần, trả lời phần đó trước; không đọc nguyên bài giới thiệu.

### Lộ trình đọc từ số 0

- Có 15 phút: đọc mục 1, mục 2, ví dụ Q = 4,65 ở mục 4, ví dụ lợi nhuận tồn kho ở mục 7 và 10 câu hỏi nền ở mục 12.
- Có một giờ: đọc đến hết mục 9 và bảng số liệu đã đo ở mục 15; tự che đáp án và giải thích lại mỗi đoạn “Trả lời ngắn”.
- Có một buổi: đọc bản đồ mã, mở HTML/notebook và đối chiếu ba con số hoặc hình với tệp dữ liệu. Làm lại hai phép tính bằng giấy.
- Trước giờ vấn đáp: xác định thầy đang hỏi **bài gốc**, **mở rộng trạng thái cũ** hay **framework tồn kho mới**. Nói tên phạm vi trước khi đưa con số.

### Mười điều đọc trước nếu thầy hỏi ngẫu nhiên

1. **Bài làm gì:** chọn giá để tối đa lợi nhuận khi có đối thủ; bài gốc và tồn kho mới là hai môi trường khác nhau.
2. **Ba thành phần cơ bản:** state là tình huống, action là giá, reward là lời/lỗ; policy là cách chọn giá theo tình huống.
3. **Bài cũ có tám Q:** bốn quan sát nhân hai action; giảm giá làm margin từ 5 xuống 3.
4. **Một phép tính Q:** target 10,5, sai số 6,5, alpha 0,1 nên từ 4 thành 4,65.
5. **Đừng nhận thắng từ baseline:** 35,1275 và 34,9685 là mẫu mỗi bước; hai bên cùng giữ giá, kỳ vọng cùng 35.
6. **Tại sao thêm state:** own action trước ảnh hưởng đối thủ hôm nay; h và kho ảnh hưởng cơ hội ngày mai.
7. **Môi trường mới:** 20 bước, kho mặc định 50, bốn giá 10/9,5/9/8; chỉ kho cửa hàng tiêu điểm hữu hạn.
8. **Hai ví dụ bắt buộc hiểu:** stock 2 bán 2 chỉ cho cầu ít nhất 2; nhập 10 giá vốn 5, bán 6 giá 10, thu hồi 4 và giữ kho 2 thì lời 12.
9. **Phương pháp và bằng chứng:** thu dữ liệu → học phân phối → DP kỳ vọng → test đóng băng; 100.000 tương tác mỗi learner, 10 seed × 200 mùa, tape ghép cặp.
10. **Kết quả và giới hạn:** nominal planner/luật kho/Q khoảng 209,76/203,37/184,33 mỗi mùa; thắng hai baseline trong sáu kịch bản thử, chưa chứng minh mọi thị trường hay dữ liệu thật.
Một câu trả lời tốt thường gồm bốn mảnh: việc đang làm là gì; cơ chế đơn giản; một ví dụ hoặc bằng chứng; một giới hạn. Chẳng hạn: “Em tắt khám phá khi kiểm tra để đo chính sách đã học. Mã không gọi hàm learn. Tuy vậy doanh số vẫn ngẫu nhiên nên cần nhiều seed.” Cách nói này chắc hơn “em tắt random nên kết quả hoàn toàn chính xác”.

**Trả lời ngắn 20–40 giây:** “Bài có phần lab gốc và phần mở rộng. Lab cho em hiểu Q-learning và cách đánh giá. Mở rộng đặt thêm tồn kho và mùa bán hữu hạn để quyết định hôm nay có ảnh hưởng thật đến cơ hội ngày mai. Em học các phân phối từ dữ liệu quan sát rồi lập kế hoạch, và chỉ kết luận trong những kịch bản đã kiểm tra.”

## 2. Từ điển bằng ví dụ cửa hàng

Hãy tưởng tượng em quản lý một quầy bán hàng trong 20 ngày. Mỗi sáng em nhìn số hàng còn lại, tín hiệu nhu cầu và lịch sử giá, rồi chọn mức giá. Khách mua ít hay nhiều tạo lợi nhuận. Máy tính lặp lại việc đó nhiều lần để tìm cách quyết định tốt hơn.

| Từ | Hình dung đời thường | Nghĩa chính xác trong bài |
|---|---|---|
| Agent, tác tử | Người bán có cách ra quyết định riêng | Thành phần quan sát, chọn hành động và có mục tiêu. Không phải cứ có một hàm Python là có một agent. |
| Environment, môi trường | Chợ, khách và quy luật giao dịch | Bộ mô phỏng nhận các hành động, sinh doanh số, reward và quan sát kế tiếp. |
| State, trạng thái | Những thông tin cần để quyết định buổi sáng | Biến mô tả tình hình. Một quan sát rút gọn được gọi là state trong code chưa chắc đủ tính Markov. |
| Observation, quan sát | Những gì người bán thật sự nhìn thấy | Phần thông tin cung cấp cho tác tử; không bao gồm nhu cầu tiềm ẩn hay tương lai. |
| Action, hành động | Hôm nay bán giá nào | Bài cũ: giữ giá hoặc giảm. Bài mới: một trong bốn mức giá. |
| Reward, thưởng | Tiền lời hoặc lỗ sau giao dịch | Tín hiệu số để học/tối ưu. Ở đây gắn với lợi nhuận, không phải điểm thưởng tùy ý. |
| Policy, chính sách | Quy tắc “gặp tình huống này thì làm gì” | Ánh xạ từ state sang hành động hoặc phân phối hành động. Không phải chuỗi giá cố định cho mọi mùa. |
| Step, bước | Một ngày ra quyết định | Một lần quan sát → chọn giá → môi trường trả kết quả. |
| Episode, lượt/mùa | Một mùa bán hàng trọn vẹn | Chuỗi 20 bước, rồi reset. Bài mới trả hàng đầu mùa về mức của kịch bản. |
| Return, tổng thưởng | Tổng tiền lời cả mùa | Tổng reward; có thể chiết khấu bằng gamma hoặc cộng thẳng theo mục tiêu. |
| Q-value | Sổ ước lượng: làm việc này trong tình huống kia đáng bao nhiêu | Ước lượng tổng thưởng tương lai từ state–action rồi tiếp tục theo policy thích hợp; không chỉ là lời hôm nay. |
| Q-learning | Thử bán, ghi kết quả và sửa sổ | Phương pháp học trực tiếp Q từ chuyển tiếp thực nghiệm, không cần bảng xác suất môi trường tường minh. |
| Alpha | Độ mạnh khi sửa sổ sau một trải nghiệm | Tốc độ học. Alpha 0,1 dịch Q 10% khoảng cách đến target của mẫu đó. |
| Gamma | Mức coi trọng tiền lời ở các bước sau | Hệ số nhân reward tương lai. Gamma = 1 hợp lệ trong mùa hữu hạn có terminal. |
| Epsilon | Tỷ lệ chủ động thử một cách khác | Xác suất vào nhánh chọn ngẫu nhiên; không bằng tỷ lệ chắc chắn chọn hành động kém. |
| Exploration, khám phá | Thử mức giá chưa biết rõ | Thu thập trải nghiệm; có thể hy sinh lợi nhuận trước mắt. |
| Exploitation, khai thác | Dùng lựa chọn hiện được đánh giá tốt nhất | Chọn hành động có Q cao nhất hoặc policy đã lập. |
| RNG | Bộ rút thăm của chương trình | Bộ sinh số giả ngẫu nhiên để tạo nhu cầu, khách mua, hành động khám phá. |
| Seed | Mã khởi tạo bộ rút thăm | Cho phép tái lập quy trình với cùng phần mềm. Không bảo đảm kết quả đúng ngoài đời. |
| Markov | Bản ghi hôm nay đủ để dự đoán ngày mai | Khi đã biết state và action hiện tại, lịch sử xa hơn không bổ sung thông tin về phân phối kết quả kế tiếp. Đây là thuộc tính của biểu diễn và mô hình. |
| DP, quy hoạch động | Tính từ ngày cuối ngược về ngày đầu | Dùng giá trị các bài toán còn ít bước để giải bài toán còn nhiều bước. |
| Bellman | So tiền hôm nay cộng giá trị phần việc còn lại | Phương trình đệ quy của giá trị tối ưu; cần kỳ vọng khi có ngẫu nhiên. |
| Oracle | Giám khảo biết đúng luật sinh dữ liệu | Chuẩn tham chiếu biết xác suất thật. Không biết kết quả rút thăm tương lai trong bài này. |
| Model-free | Học trực tiếp sổ giá trị | Q-learning dùng mẫu reward/next state; không học riêng công thức nhu cầu rồi giải DP. |
| Model-based | Học bản đồ rồi tính đường đi | Học phân phối nhu cầu/đối thủ/chuyển trạng thái, sau đó dùng chúng để lập kế hoạch. |
| Myopic, thiển cận | Chỉ nhìn lời hôm nay | Chọn theo kỳ vọng reward trước mắt, không cộng giá trị cơ hội các ngày sau. |
| SD, độ lệch chuẩn | Các lần làm lại chênh nhau nhiều hay ít | Độ phân tán giữa seed. SD lớn nghĩa kết quả ít đồng đều hơn, không tự nói ai tốt hơn. |
| CI, khoảng tin cậy | Độ bất định của trung bình ước lượng | Ở đây là khoảng t 95% qua seed; khác SD và khác khoảng chứa 95% từng mùa. |
| Censored sales, doanh số bị chặn | Chỉ còn hai món nên bán được tối đa hai | Khi bán hết, không biết khách thật sự muốn mua bao nhiêu vượt tồn kho. |
| Opportunity cost, chi phí cơ hội | Bán món cuối hôm nay thì ngày mai không còn để bán giá cao | Giá trị tương lai bị mất khi dùng hàng ngay; không phải một khoản tiền mặt cộng tùy tiện vào hóa đơn. |
| Ablation, loại bỏ thành phần | Tắt một bộ phận để xem nó giúp gì | Giữ phần còn lại gần như cũ và bỏ một nguồn thông tin/cơ chế. Cần nói rõ có thay đổi đồng thời yếu tố nào khác. |
| Frozen evaluation | Thi xong mới chấm, không sửa bài lúc chấm | Đóng băng Q/policy, không khám phá hoặc cập nhật từ dữ liệu test. |
| Exact và Monte Carlo | Tính trung bình theo mọi khả năng và đo qua nhiều lần chơi | Exact tích phân/tổng theo mô hình; Monte Carlo lấy mẫu. Hai số có thể khác vì nhiễu lấy mẫu. |

Hai chữ dễ lẫn: **ngẫu nhiên** nghĩa kết quả từng lần thay đổi dù quy luật cố định; **không dừng** nghĩa quy luật phân phối hiệu dụng thay đổi. Một đồng xu công bằng cho lúc sấp lúc ngửa vẫn là quy luật dừng. Đối thủ đang học và đổi policy có thể làm môi trường hiệu dụng không dừng.

**Trả lời ngắn 20–40 giây:** “State là tình huống em biết, action là giá em chọn, reward là lời lỗ, còn policy là cách chọn giá theo tình huống. Một step tương ứng một lần bán; một episode là cả mùa. Q-learning sửa ước lượng giá trị từ trải nghiệm. Model-based học quy luật trước rồi dùng quy luật đó tính kế hoạch.”

## 3. Bài lab gốc: hai người bán, bốn quan sát, hai hành động

### 3.1. Mỗi bên biết gì và làm gì?

Có hai người bán A và B. Trong baseline, A dùng Q-learning; B dùng luật. Tại bước t, người bán i thấy `(d_t, hành động của đối thủ ở t−1)`. `d=0` nghĩa nhu cầu thấp, `d=1` nghĩa cao. Hành động bằng 0 là giữ giá, bằng 1 là giảm 20%.

Bốn quan sát có thể có là `(0,0), (0,1), (1,0), (1,1)`. Mỗi quan sát có hai lựa chọn nên bảng có `4 × 2 = 8` giá trị Q. Hình dung một bảng bốn hàng, hai cột: hàng là tình huống, cột là giữ/giảm giá. Các con số Q thay đổi trong khi học, còn luật B giữ nguyên.

Luật B rất ngắn: nhu cầu thấp thì không giảm; nhu cầu cao và A đã giảm ở bước trước thì giảm; các trường hợp còn lại giữ giá. B có phản ứng với tình huống nhưng **không học**: nhận lãi hay lỗ cũng không tự sửa luật.

Hai bên chọn hành động hiện tại trước khi môi trường tính thưởng. Tên hàm có thể gọi A trước rồi B, nhưng B chỉ nhận quan sát từ bước trước. Thứ tự gọi trong mã không cho phép B nhìn giá mới của A. Nếu cho xem rồi mới quyết định, bài toán chiến lược đã khác.

### 3.2. Khách mua và lợi nhuận được tạo thế nào?

Số khách tiềm năng N là 10 khi nhu cầu thấp, 20 khi cao. Xác suất một khách mua của người bán i là:

`p_i = clip(0,30 + 0,25d + 0,15a_i − 0,10a_đối_thủ; 0,05; 0,95)`.

`clip` là chặn số vào khoảng cho phép. Ví dụ nhu cầu thấp, cả hai giữ giá thì p = 0,30. Doanh số S được lấy mẫu từ nhị thức `Binomial(N,p)`: hiểu như N lần thử mua/không mua, mỗi lần có xác suất p. Không phải ngày nào cũng bán đúng Np; Np là trung bình lý thuyết khi lặp rất nhiều ngày cùng điều kiện.

Giá gốc 10, chi phí đơn vị 5. Giữ giá lời 5 trên món; giảm 20% thì bán giá 8, lời 3. Reward là `S × [10(1−0,20a_i)−5]`. Reward khác doanh thu: bán 10 món giá 8 thu 80 nhưng phần lời theo mô hình chỉ 30. Nếu dùng doanh số làm reward thì hệ có thể giảm giá nhiều hơn vì không bị phạt mất biên lợi nhuận.

Giữ đối thủ ở action 0 để tính một bước: nhu cầu thấp, giữ giá có kỳ vọng `10 × 0,30 × 5 = 15`, giảm có `10 × 0,45 × 3 = 13,5`. Nhu cầu cao, giữ giá có `20 × 0,55 × 5 = 55`, giảm có `20 × 0,70 × 3 = 42`. Số 42 là **kỳ vọng một bước khi nhu cầu cao và A giảm còn B không giảm**, không phải seed 42, không phải kết quả baseline và không phải chứng minh tối ưu dài hạn.

Bài cũ không có tồn kho: mỗi bước đều có thể bán theo quy luật. Doanh số hai bên được lấy mẫu riêng nên chưa bảo toàn một nhóm khách chỉ mua ở một cửa hàng. Đây là mô phỏng học tập, không phải mô hình đã được kiểm chứng kinh tế ngoài đời.

### 3.3. Cấu hình cần biết

| Tham số baseline | Giá trị và ý nghĩa |
|---|---|
| Train | 5.000 episode × 20 bước = 100.000 tương tác |
| Alpha | 0,10 |
| Gamma | 0,90 |
| Epsilon | `max(0,02; 0,20 × 0,995^episode)` |
| Seed môi trường / policy | 42 / 10042 |
| Evaluation | 500 episode × 20 bước, seed 500042 |
| Khi đánh giá | `explore=False`, không gọi `learn` |

Đừng gọi seed là “số mẫu”. Seed 42 chỉ là mã khởi tạo. Khi muốn biết lượng dữ liệu, nói số episode, số bước, số seed độc lập.

**Trả lời ngắn 30–50 giây:** “Bài gốc có một người bán học và một người bán theo luật. Quan sát gồm nhu cầu và hành động trước của đối thủ nên có bốn tổ hợp; hai lựa chọn tạo tám Q. Giảm giá tăng xác suất mua nhưng giảm lời mỗi món từ 5 xuống 3. Vì reward là lợi nhuận, bán nhiều hơn chưa chắc tốt hơn. Cả hai quyết định trước khi thấy hành động hiện tại của nhau.”
## 4. Q-learning: hiểu một phép cập nhật thay vì thuộc công thức

### 4.1. Máy học gì?

Ban đầu máy chưa biết giữ giá hay giảm giá tốt, nên tám số Q bằng 0. Sau một bước, máy có bộ trải nghiệm `(s, a, r, s′, terminal)`. Nó dùng reward vừa nhận cộng ước lượng cơ hội tương lai để sửa Q của đúng ô đã chọn. Không có giáo viên đưa nhãn “hôm nay phải giảm giá”. Phản hồi đến từ lời/lỗ trong mô phỏng.

Công thức là:

`Q_mới(s,a) = Q_cũ(s,a) + alpha × [r + gamma × max Q(s′,a′) − Q_cũ(s,a)]`.

Đọc thành tiếng: “giá trị mới bằng giá trị cũ cộng tốc độ học nhân sai số giữa mục tiêu mới và ước lượng cũ”. Dấu `max` nghĩa nhìn các hành động ở tình huống kế và lấy giá trị lớn nhất. Phần trong ngoặc vuông gọi là TD error. “TD” ở đây chỉ việc sửa từ chênh lệch giữa các ước lượng theo thời gian; không cần giải thích bằng thuật ngữ dài nếu thầy chỉ hỏi phép tính.

Ví dụ Q cũ = 4, alpha = 0,1, reward = 6, gamma = 0,9, Q tốt nhất ở s′ = 5:

1. Giá trị tương lai: `0,9 × 5 = 4,5`.
2. Target: `6 + 4,5 = 10,5`.
3. Sai số: `10,5 − 4 = 6,5`.
4. Sửa thêm: `0,1 × 6,5 = 0,65`.
5. Q mới: `4 + 0,65 = 4,65`.

Máy chỉ đi 10% quãng đường đến target vì một lần bán có thể may/rủi. Alpha lớn phản ứng mạnh hơn, nhưng cũng dễ chạy theo nhiễu. Alpha nhỏ sửa ít mỗi lần, thường cần thêm trải nghiệm. Không thể từ cơ chế này khẳng định alpha nào luôn cho lợi nhuận cao nhất.

Nếu đây là bước cuối thật sự, target chỉ bằng r. Ngày mai thuộc mùa mới, không được cộng Q của mùa mới như thể còn tiếp tục bán hàng cũ. Vì vậy nhánh `terminal` quan trọng. Mã mẫu xử lý terminal ở bước 20; một hạn chế khác là state baseline không chứa thời gian còn lại.

Q-learning được gọi là off-policy vì target dùng `max Q` của hành động tốt nhất, dù hành động thực sự tiếp theo khi đang khám phá có thể khác. Nó dùng bootstrapping: lấy ước lượng Q kế để sửa ước lượng Q hiện tại. “Bootstrap” ở đây không phải phương pháp bootstrap thống kê để tạo khoảng tin cậy.

### 4.2. Epsilon-greedy không có nghĩa chọn sai đúng epsilon phần trăm

Máy cần thử cả những lựa chọn chưa được đánh giá tốt, nếu không một kết quả xấu ban đầu có thể khiến nó bỏ một hành động mãi mãi. Với xác suất epsilon, nó chọn ngẫu nhiên; phần còn lại chọn greedy, tức hành động đang có Q cao nhất.

Nếu có hai hành động, một hành động greedy duy nhất và epsilon = 0,20, nhánh ngẫu nhiên vẫn chọn greedy một nửa số lần. Xác suất thực sự chọn hành động còn lại là `0,20/2 = 0,10`, không phải 20%. Với bốn hành động và một greedy duy nhất, tỷ lệ chọn một hành động không greedy trong nhánh tổng thể là `epsilon × 3/4`. Khi Q hòa thì còn phải xét cách phá hòa.

Bài cũ `act` phá hòa ngẫu nhiên giữa các Q được xem là gần bằng nhau; hàm xuất `greedy_policy` dùng argmax chỉ số nhỏ nhất. Vì vậy khi cần tính kỳ vọng đúng policy đánh giá, phần mở rộng phải mô phỏng đúng xác suất phá hòa. Bài tồn kho mới dùng `np.argmax` cố định khi khai thác; không mặc nhiên mang quy tắc phá hòa cũ sang mới.

**Trả lời ngắn 30–50 giây:** “Q-learning cập nhật một ô Q từ reward hôm nay và ước lượng tốt nhất ngày mai. Với ví dụ của đề, target là 10,5, sai số 6,5, alpha 0,1 nên Q tăng từ 4 lên 4,65. Epsilon tạo nhánh thử ngẫu nhiên; thử ngẫu nhiên vẫn có thể trúng hành động tốt nhất. Khi đánh giá, em tắt khám phá và không học thêm.”

## 5. Đọc thí nghiệm A–E và kết quả cũ mà không nói quá

### 5.1. Năm nhóm việc của đề

| Nhóm | Cần làm | Học được điều gì |
|---|---|---|
| A – Baseline | Chạy cấu hình chuẩn, xem Q và policy, so RL với Rule | Một mốc tái lập để hiểu môi trường và đầu ra |
| B – Đường học | Vẽ cumulative reward và reward trung bình theo cửa sổ | Quá trình học diễn ra thế nào; không dùng riêng đường học để chứng minh hội tụ |
| C – Learning rate | So alpha 0,01; 0,10; 0,50, giữ các điều kiện còn lại | Ảnh hưởng của mức cập nhật với ngân sách hữu hạn |
| D – Exploration | So ít nhất ba thiết lập khám phá | Đánh đổi thu thập thông tin và khai thác lợi nhuận |
| E – Frozen evaluation | Đánh giá riêng trên môi trường/seed mới, không học | Chất lượng policy sau train, tách khỏi việc thử hành động trong train |

Phần kiểm soát nhiều seed dùng 7 cấu hình × 5 seed. Ngoài baseline và hai alpha khác, có lịch khám phá low/high và epsilon cố định 0,05/0,30. Lịch low là `(start 0,05; min 0,02; decay 0,999)`, baseline `(0,20; 0,02; 0,995)`, high `(0,50; 0,05; 0,995)`. Các đường cửa sổ 100 episode giúp nhìn xu hướng dễ hơn. Làm mượt chỉ thay cách trình bày; không tạo thêm dữ liệu và có thể che dao động ngắn hạn.

Cumulative reward thường tăng vì cứ cộng thêm reward dương; việc nó tăng không tự chứng minh agent đang học tốt hơn. Hãy xem độ dốc, reward trung bình cửa sổ và frozen evaluation. Cũng đừng chọn phương pháp chỉ vì đường nhìn mượt: có thể một policy ít thay đổi cho đường mượt nhưng lợi nhuận thấp.

### 5.2. Con số baseline quan trọng nhất

Baseline seed 42 cho reward trung bình mỗi bước của RL là **35,1275**, của Rule là **34,9685**. Chênh lệch đo được là `0,1590`. Nhưng ở evaluation này **cả hai đều không giảm giá**. Với nhu cầu thấp/cao cân bằng của mô hình và hai bên cùng giữ giá, kỳ vọng mỗi bước là `(15+55)/2 = 35` cho mỗi bên.

Vì vậy cách nói đúng là: “Trong một lần đánh giá, RL có trung bình mẫu cao hơn một chút; hành vi hai bên giống nhau và kỳ vọng mô hình bằng nhau, nên con số này không chứng minh chính sách RL tốt hơn.” Cách nói sai là: “RL thông minh nên chắc chắn thắng luật 0,159 điểm.” Chênh lệch giữa hai mẫu doanh số ngẫu nhiên là điều bình thường ngay cả khi hai policy tạo cùng phân phối lợi nhuận.

Các con số 35,1275 và 34,9685 là **mỗi bước**, không phải tổng mỗi mùa. Bộ đánh giá có 10.000 bước. Nếu muốn đổi đơn vị, phải nhân đúng số bước và nói rõ phép đổi; không lấy 35 này đặt cạnh profit cả mùa của tồn kho mới rồi nói hệ mới “tăng gấp mấy lần”.

### 5.3. Mở rộng cũ đã làm và giới hạn của bằng chứng

Phần cũ thay mức giảm giá 20% bằng 10% và 5%, giữ các hệ số chuyển đổi của môi trường. Policy `reference_transfer` học tại 20% rồi mang nguyên sang môi trường mới; `reference_retrained` học lại từ đầu; `horizon_q` dùng state đầy đủ hơn; `oracle_dp` biết mô hình. Phân biệt “mang đi dùng ngay” với “được học lại” vì hai việc trả lời hai câu hỏi khác nhau.

Oracle cũ có kỳ vọng tối ưu mỗi bước lần lượt **35,00; 35,10; 37,35** ứng với giảm **20%; 10%; 5%**. Đây là giá trị exact của mô hình hữu hạn, không phải trung bình Monte Carlo và không phải số ngoài thị trường. Giảm phần trăm mà vẫn giữ nguyên phần cộng 0,15 vào xác suất mua là một giả định khiến giảm giá hấp dẫn hơn; chưa có dữ liệu thật chứng minh giả định đó.

Ở mức 10%, biến thể `local_gamma1` giữ quan sát cục bộ rồi chỉ đổi gamma giúp kiểm tra một phần. Tuy vậy `horizon_q` còn đổi state, thời gian và lịch alpha; đây chưa phải thí nghiệm tách sạch từng nguyên nhân. Nếu được hỏi “do thêm state hay do gamma?”, câu trả lời là bằng chứng hiện tại cho toàn bộ gói và có ablation một phần; chưa xác định riêng hoàn toàn từng đóng góp.

### 5.4. Hai bảng cũ để tập đọc số liệu thật

Bảng dưới trích `extra_results/controlled_summary.csv`, làm tròn bốn chữ số. Tất cả là reward evaluation mỗi bước, mean và SD qua năm seed; CI là khoảng t 95%. Đây là sweep cùng thiết kế, không phải baseline đơn seed 42.

| Cấu hình | Mean | SD | CI 95% |
|---|---:|---:|---:|
| baseline | 34,8919 | 0,3132 | [34,5030; 35,2808] |
| alpha 0,01 | 34,8919 | 0,3132 | [34,5030; 35,2808] |
| alpha 0,50 | 33,6299 | 1,6924 | [31,5286; 35,7313] |
| explore low | 34,3675 | 1,0857 | [33,0194; 35,7155] |
| explore high | 32,4223 | 4,1852 | [27,2257; 37,6190] |
| epsilon cố định 0,05 | 34,3675 | 1,0857 | [33,0194; 35,7155] |
| epsilon cố định 0,30 | 28,7730 | 3,8280 | [24,0199; 33,5261] |

Đọc mẫu một dòng: baseline có mean khoảng 34,89 và SD 0,31 giữa seed; không nói mọi episode đều gần 34,89. Alpha 0,50 có mean thấp hơn và phân tán lớn hơn trong sweep này; không suy thành alpha lớn luôn xấu. Hai cấu hình có bảng tổng hợp trùng nhau không chứng minh mọi quá trình học từng bước trùng, chỉ cho cùng thống kê cuối đã lưu.

Bảng robustness cũ dưới đây trích `extra_results/robustness_summary.csv`, vẫn là **Monte Carlo mỗi bước**, không phải exact oracle. Các cột lần lượt cho transfer/retrain/full-state/oracle:

| Mức giảm | Transfer | Retrain | Horizon Q | Oracle rollout |
|---|---:|---:|---:|---:|
| 20% | 34,8010 | 34,8010 | 33,2881 | 34,8010 |
| 10% | 34,8010 | 33,6427 | 34,0319 | 34,8995 |
| 5% | 34,8010 | 36,7061 | 36,3134 | 37,1664 |

`local_gamma1` tại 10% có mean 34,2399 và SD 0,8268. So bảng này với exact oracle 35; 35,1; 37,35 sẽ thấy rollout oracle không bằng exact vì nó là mẫu. Các seed test ở robustness khác sweep nên baseline 34,8919 và transfer 34,8010 không mâu thuẫn. Điều đáng học là full-state Q cũ không thắng transfer ở 20%, không thắng retrain ở 5%; thêm biểu diễn đúng hơn không xóa vấn đề hữu hạn mẫu. Nếu cần phân tích gap, mở `robustness_runs.csv` và dùng `oracle_mean − exact_mean`, không trừ lẫn hai loại số.
**Trả lời ngắn 30–50 giây:** “A đến E lần lượt là mốc chuẩn, đường học, so alpha, so khám phá và đánh giá đóng băng. Baseline RL đạt 35,1275 còn luật 34,9685 mỗi bước, nhưng hai bên đều giữ giá và kỳ vọng đều là 35. Em dùng kết quả này để tránh kết luận thắng từ một lần chạy. Phần robustness cũ còn so chuyển policy, học lại và chuẩn oracle ở ba mức giảm giá.”

## 6. Vì sao phải bàn về Markov và thêm trạng thái?

### 6.1. Phản ví dụ chỉ cần nhớ một tình huống

Giả sử A đang quan sát `(d=1, B_trước=0)`: nhu cầu cao và B hôm qua giữ giá. Có hai lịch sử khác nhau cùng tạo quan sát đó:

- Lịch sử thứ nhất: A hôm qua giữ giá, tức `A_trước=0`. Luật B hôm nay nhìn A hôm qua nên B hôm nay giữ giá.
- Lịch sử thứ hai: A hôm qua giảm giá, tức `A_trước=1`. Luật B hôm nay nhìn A hôm qua nên B hôm nay giảm giá.

A nhìn cùng một quan sát rút gọn nhưng phân phối reward hôm nay khác nhau vì hành động B hiện tại khác. Thông tin bị thiếu là hành động trước của chính A. Không phải vì B “đang học”: B vẫn tuân thủ đúng một luật cố định. Vấn đề nằm ở cách mô tả state của A.

Thêm `A_trước` cho trạng thái vật lý `(d, A_trước, B_trước)` tạo 8 tổ hợp và 16 Q nếu chưa xét thời gian. Nếu mục tiêu là tổng lợi nhuận của đúng 20 bước, cần thêm h, số bước còn lại: cùng tình huống vật lý nhưng chỉ còn một ngày có thể nên quyết khác khi còn 20 ngày.

### 6.2. Vì sao 320 và 336 đều xuất hiện?

Agent `horizon_q` cũ dùng `(h,d,A_trước,B_trước)`. Với riêng luật B cũ, B_trước là thông tin dư cho việc dự đoán action hiện tại vì B chỉ cần d và A_trước; nó được giữ trong cấu trúc mô tả hai bên. Ở đối thủ xác suất mới, B_trước thực sự có thể ảnh hưởng phân phối b. Với `h=1..20`, có `20 × 2 × 2 × 2 × 2 = 320` Q hoạt động, trong đó số 2 cuối là action hiện tại. Mảng lưu thêm `h=0` làm lớp terminal nên có `21 × 2 × 2 × 2 × 2 = 336` ô lưu. Nói 320 là số hoạt động, 336 là số lưu vật lý; không tự sửa một số thành số kia như thể tài liệu mâu thuẫn.

Agent này dùng gamma = 1 và alpha theo số lần thăm `N(s,a)^(-0,6)`. Gamma = 1 không gây tổng vô hạn vì episode kết thúc sau 20 bước. Nhưng bảng lớn hơn cần nhiều dữ liệu hơn; 100.000 bước chia cho nhiều ô không đảm bảo ô nào cũng được học tốt. Có state về mặt chỉ số nhưng hiếm hoặc không đến từ trạng thái khởi đầu, nên đếm số ô cũng không phải số ô được thăm đều.

### 6.3. Đối thủ cố định, đối thủ đang học và giới hạn kết luận

Với đối thủ cố định và state đủ, quy luật có thể dừng dù từng mẫu reward dao động. Nếu cả A và B liên tục học và đổi policy, phân phối A gặp phụ thuộc policy B đang thay đổi. Khi đó bảo đảm của Q-learning đơn tác tử trong MDP dừng không được áp dụng nguyên vẹn.

Bộ cũ và framework tồn kho đều không phải thử nghiệm hai learner đồng thời. Có thể đề xuất thử RL–RL và đối chiếu với đối thủ snapshot đã đóng băng, nhưng phải nói đây là hướng phát triển. Không dùng dữ liệu hiện có để kết luận đã giải quyết non-stationarity do hai learner cùng thích nghi.

**Trả lời ngắn 30–50 giây:** “Quan sát cũ thiếu hành động trước của chính mình. Cùng nhu cầu cao và B hôm qua giữ giá, B hôm nay có thể giữ hoặc giảm tùy A hôm qua. Thêm lịch sử của mình làm rõ phản ứng memory-one; thêm h để mô tả mùa hữu hạn. Agent cũ mở rộng có 320 Q hoạt động, 336 ô lưu cả terminal. State đầy đủ hơn có ích về biểu diễn nhưng không đảm bảo thắng khi dữ liệu huấn luyện còn ít.”
## 7. Framework mới: bán hàng có tồn kho và thời hạn

### 7.1. Vì sao cần bài toán này?

Trong bài cũ, bán thêm hôm nay không làm thiếu hàng ngày mai. Bài mới đặt câu hỏi thực tế hơn: nếu hàng có hạn và cuối mùa bán không hết chỉ thu hồi được ít tiền, khi nào giữ giá để tiết kiệm hàng, khi nào giảm để tránh tồn? Khi nhu cầu đầu mùa thấp nhưng cuối mùa có thể tăng, một quyết định có giá trị tương lai rõ ràng.

Một mùa có **20 bước**, đầu mùa thông thường có **50 sản phẩm**; khung hỗ trợ đến **70**, và kịch bản khan hiếm/dư thừa dùng 30/70. Bốn mức giảm là `0%; 5%; 10%; 20%`, tương ứng giá **10; 9,5; 9; 8** khi giá gốc 10. Chi phí nhập một món là 5, giá thu hồi cuối mùa là 1, phí giữ hàng là 0,02 trên mỗi món còn lại sau từng bước. Chỉ cửa hàng đang nghiên cứu có tồn kho hữu hạn; nguồn cung của đối thủ được đơn giản hóa là đủ bán. Mã mới không mô hình hóa doanh số/lợi nhuận riêng của đối thủ và không giải cân bằng Nash. Xác suất mua chỉ mô tả mua ở cửa hàng tiêu điểm hoặc không mua ở đó, không phân bổ độc quyền doanh số chung cho hai seller. Đừng mô tả đây là mô hình hai kho cùng cạn hàng.

Trạng thái quan sát là `(h, I, d, a_trước, b_trước)`:

| Thành phần | Người bán dùng để hiểu điều gì? |
|---|---|
| h | Còn bao nhiêu lần có thể bán? |
| I | Còn bao nhiêu hàng thật sự có thể bán? |
| d | Tín hiệu nhu cầu hiện tại thấp/cao? |
| a_trước | Mình vừa đặt mức giá nào ở bước trước? |
| b_trước | Đối thủ vừa đặt mức giá nào ở bước trước? |

Pha đầu/cuối mùa được suy ra từ h và horizon, nên không cần thêm một chỉ số pha độc lập vào policy. Khi I = 0, không bán thêm được và action được quy ước về 0. Mùa vẫn có các bước còn lại để giữ cấu trúc thời gian; zero stock không cung cấp quan sát nhu cầu bán hàng hữu ích.

### 7.2. Trình tự một ngày: ai biết gì ở thời điểm nào?

1. Người bán nhận state hiện tại, gồm giá hai bên của **bước trước**.
2. Người bán chọn a; đối thủ chọn b theo luật xác suất phụ thuộc nhu cầu và lịch sử.
3. Sau khi cả hai chọn xong, môi trường sinh số khách muốn mua, chặn bởi tồn kho, tính reward.
4. Dữ liệu ghi nhận gồm doanh số, mức hàng có thể bán, giá hiện tại hai bên và tín hiệu nhu cầu kế.
5. State mới có tồn kho đã giảm và lịch sử a, b vừa thực hiện.

Được thấy b sau bước bán để học mô hình không có nghĩa được dùng b trước khi chọn a trong chính bước đó. Đây là hàng rào thời gian quan trọng nhất. Tác tử dự báo **phân phối** của b khi ra quyết định; nó không đọc chính xác b hiện tại từ mô phỏng.

`potential`, lượng khách thật sự muốn mua trước khi chặn kho, là biến nội bộ môi trường. Learner không nhận nó. Biến `lost_units_private` phục vụ giám khảo đo mất cơ hội bán; có thể có trong báo cáo đánh giá nhưng không được đưa vào huấn luyện. Một chỉ số được tính bởi simulator không tự nhiên trở thành thông tin mà người bán được phép biết.

### 7.3. Nhu cầu được mô phỏng thế nào?

Mỗi bước có 12 khách tiềm năng. Với từng người, xác suất mua p phụ thuộc tín hiệu d, mức giảm của mình và của đối thủ. Hàm logistic biến một tổng điểm bất kỳ thành xác suất từ 0 đến 1:

`p = 1 / (1 + exp(−z))`,

`z = theta_0 + theta_d × d + theta_a × giảm_giá_mình + theta_b × giảm_giá_đối_thủ`.

Không cần nhớ hàm mũ để hiểu: tăng tổng điểm thì xác suất mua tăng; đầu ra không vượt 0..1. Cấu hình mô phỏng nominal dùng `theta = (−2,4; 1,8; 8,0; −4,5)`. Hệ số giảm giá mình dương nên giảm giá kéo khách về; hệ số giảm giá đối thủ âm nên đối thủ giảm làm mình mất khách. Giá trị giảm là 0,05 hay 0,20, không phải 5 hay 20; nhập sai đơn vị làm mô hình khác hoàn toàn.

Nhu cầu tiềm ẩn K có phân phối `Binomial(12,p)`. Doanh số thật là `Y = min(I,K)`. Có 12 khách và p = 0,5 không có nghĩa ngày nào cũng bán 6; trung bình tiềm ẩn là 6, còn tồn kho có thể chặn thấp hơn nữa.

Tín hiệu d chuyển theo chuỗi Markov hai trạng thái. Ma trận chuyển của pha đầu khác pha cuối, nên thị trường có mùa vụ được khai báo. Ví dụ hàng đầu `[0,88; 0,12]` nghĩa đang thấp thì bước kế có 88% thấp, 12% cao. Hàng thứ hai là xác suất bước kế khi hiện tại đang cao. Mỗi hàng cộng thành 1. Đó là xác suất chuyển tín hiệu, không phải xác suất một khách mua.

### 7.4. Sáu kịch bản kiểm tra

| Tên trong dữ liệu | Cách hiểu | Khác với nominal |
|---|---|---|
| `nominal` | Thị trường chuẩn | Kho 50; nhu cầu cuối mùa thường thuận lợi hơn đầu mùa |
| `scarce` | Khan hàng | Kho đầu mùa 30 |
| `abundant` | Dư hàng | Kho đầu mùa 70 |
| `late_surge` | Nhu cầu tăng mạnh về cuối | Ma trận chuyển thiên về thấp ở đầu và cao ở cuối rõ hơn |
| `early_peak` | Cao điểm đầu mùa | Đảo xu hướng thời gian, nhiều cơ hội sớm hơn |
| `aggressive_rival` | Đối thủ giảm mạnh hơn | Đổi cơ chế xác suất chọn giá của đối thủ |

Cả sáu kịch bản vẫn giữ cùng họ binomial–logistic và cùng theta thật; chỉ đổi tồn kho, chuyển tín hiệu hoặc luật đối thủ. Chưa có kiểm tra sai dạng hàm cầu hay dữ liệu thật. Các learner cuối cùng được train trong nominal; đánh giá sáu thị trường sau khi đóng băng để xem khả năng chuyển sang điều kiện khác. Không được nói đã retrain theo từng kịch bản test. Oracle biết quy luật đúng của từng kịch bản nên đo chuẩn cho từng thị trường, nhưng có lợi thế thông tin so với learner dùng mô hình nominal đã học.

Hai kịch bản chính được khai báo trong manifest là nominal và late_surge. Các kịch bản còn lại giúp kiểm tra độ bền. Nếu nhìn nhiều bảng rồi chỉ kể kịch bản thắng đẹp nhất, kết luận sẽ thiếu công bằng. Khi nói kết quả, nêu cả nơi mạnh, nơi yếu và loại thay đổi thị trường.

**Trả lời ngắn 30–50 giây:** “Bài mới có mùa 20 bước, kho mặc định 50 món và bốn giá 10; 9,5; 9; 8. State gồm thời gian, kho, nhu cầu và hai giá trước. Khách muốn mua theo nhị thức với xác suất logistic, nhưng bán không vượt kho. Đối thủ hiện tại chỉ được thấy sau khi hai bên đã chọn. Sáu kịch bản đổi kho, mùa vụ hoặc đối thủ để kiểm tra khả năng chuyển giao.”

## 8. Reward tồn kho: tiền đã nhập hàng và tiền thu hồi

### 8.1. Cách tính tổng lời dễ hiểu nhất

Với hàng mua sẵn đầu mùa, tổng lợi nhuận kinh tế là:

`tiền bán hàng − chi phí nhập toàn bộ kho đầu mùa − tổng phí giữ hàng + tiền thu hồi kho cuối mùa`.

Ví dụ minh họa, độc lập với kết quả thí nghiệm: nhập 10 món, mỗi món giá vốn 5; bán 6 món giá 10; cuối mùa còn 4 món, thu hồi 1 mỗi món; tổng phí giữ hàng qua mùa là 2. Khi đó:

`6×10 − 10×5 + 4×1 − 2 = 60 − 50 + 4 − 2 = 12`.

Ở đây phí giữ hàng tổng cộng bằng 2 là giả định của ví dụ để tính dễ, không phải tự tính từ một mùa thử nghiệm cụ thể. Bốn món tồn không phải không tốn vốn: đã trả tiền mua chúng. Vì vậy chỉ lấy doanh thu trừ vốn của 6 món đã bán sẽ làm lợi nhuận trông cao giả nếu quên lỗ hàng tồn.

### 8.2. Vì sao code không trừ toàn bộ giá vốn ngay đầu mùa?

Code phân bổ phần thưởng theo bước để việc học thuận tiện:

- Mỗi bước: `reward = Y × (giá_bán − cost) − holding × I_sau_bán`.
- Bước cuối trừ thêm `(cost − salvage) × I_cuối`.

Ở ví dụ trên, lời trên hàng đã bán là `6×(10−5)=30`; trừ giữ kho 2; trừ lỗ thu hồi của bốn món là `4×(5−1)=16`; còn `30−2−16=12`. Hai cách bằng nhau.

Chứng minh ngắn: tổng số đã bán bằng `I_đầu−I_cuối`. Tổng reward là `doanh_thu−cost×số_đã_bán−holding_tổng−(cost−salvage)×I_cuối`, rút gọn thành `doanh_thu−cost×I_đầu−holding_tổng+salvage×I_cuối`. Vì thế code đang hạch toán đủ giá vốn, dù không có một dòng âm rất lớn ở reset.

Trong bài cũ không có hàng nhập sẵn bị tồn cuối mùa nên biên lợi nhuận theo món đủ cho reward cũ. Bài mới có phần lỗ hàng tồn nên cần terminal adjustment. Đừng đem công thức cũ áp vào tồn kho rồi bỏ quên salvage.

### 8.3. Tại sao giữ hàng cũng có thể tốt?

Nếu hôm nay nhu cầu thấp, giảm giá bán nhanh có thể làm hết hàng trước đợt nhu cầu cao. Giữ một món để bán giá tốt ngày mai có giá trị cơ hội. Nhưng nếu chỉ còn một ngày và nhiều hàng, giữ giá cao có thể để hàng tồn bị thu hồi rẻ. Vì vậy cùng I và d nhưng h khác nhau có thể chọn giá khác.

“Bán hết” chưa chắc tối ưu: bán hết quá sớm ở giá thấp có thể lỡ cơ hội bán lời hơn. “Còn hàng” cũng chưa chắc sai: đôi khi không thể thu hồi toàn bộ giá vốn do cầu thấp; chỉ số leftovers phải đọc cùng profit và cơ chế. Mục tiêu là lợi nhuận, không phải cố đạt tồn cuối bằng 0 bằng mọi giá.

**Trả lời ngắn 30–50 giây:** “Em tính lợi nhuận cả mùa bằng doanh thu trừ vốn kho đầu mùa, trừ giữ kho, cộng thu hồi cuối mùa. Code phân bổ vốn vào từng món bán rồi phạt phần vốn mất của hàng còn cuối mùa, nên tổng tương đương. Ví dụ nhập 10 món giá vốn 5, bán 6 món giá 10, bốn món thu hồi 1 và phí giữ kho 2 thì lời 12. Bán nhanh quá có thể mất cơ hội bán giá tốt về sau.”

## 9. Học mô hình rồi lập kế hoạch: hiểu từng khâu

### 9.1. Doanh số bị chặn: vì sao bán được 2 không có nghĩa nhu cầu bằng 2?

Giả sử chỉ còn 2 món và bán hết 2. Có thể đúng 2 khách muốn mua; cũng có thể 3, 5 hay 12. Dữ liệu chỉ nói `K ≥ 2`. Nếu coi K đúng bằng 2, mô hình sẽ đánh giá thấp nhu cầu ở những ngày hết hàng, rồi có thể đặt giá không phù hợp.

Nếu còn 5 món mà chỉ bán 2, không hết kho: khi đó biết K = 2 trong mô hình này. Vì thế likelihood dùng xác suất đúng `P(K=2)` cho trường hợp không bị chặn và dùng xác suất đuôi `P(K≥2)` khi bán bằng giới hạn. “Likelihood” có thể hiểu là mức mô hình giải thích được dữ liệu đã quan sát; tối ưu nó là tìm tham số khiến các quan sát đã xảy ra bớt bất ngờ.

Nếu kho bằng 0, bán 0 là điều bắt buộc dù cầu cao hay thấp. Dòng đó không nói nhu cầu bằng 0; bộ fit loại khỏi việc ước lượng chuyển đổi. Nó vẫn có thể chứa thông tin về chuyển tín hiệu hoặc giá đối thủ sau bước, nên không phải cứ zero stock là toàn bộ dòng vô dụng cho mọi mô hình con.

Khi cap bằng 12 và bán đủ 12, biến cố `K≥12` bằng `K=12` vì K tối đa 12. Việc code dùng nhánh tail vẫn đúng. Cần chú ý cap là `min(kho trước bán,12)`, không phải kho sau bán.

### 9.2. Học ba phần gì?

Từ dữ liệu train, `ForecastModel.fit` học:

1. **Nhu cầu mua:** bốn hệ số logistic. Các quan sát cùng điều kiện được gom với số lần xuất hiện; hàm tối ưu dùng likelihood đúng cho dữ liệu bị chặn và điều chuẩn nhỏ.
2. **Nhu cầu chuyển thấp/cao:** đếm chuyển tiếp d→d′ theo pha đầu/cuối, rồi chuẩn hóa thành xác suất.
3. **Hành vi đối thủ:** đếm b theo `(d,a_trước,b_trước)`, rồi chuẩn hóa thành phân phối bốn hành động.

Các bảng đếm có giá trị khởi đầu dương để tránh xác suất 0 tuyệt đối chỉ do chưa nhìn thấy một trường hợp. Đây là smoothing; không có nghĩa mô hình đã quan sát thêm khách thật. Trong code, chuyển tín hiệu khởi tạo 0,5 mỗi ô, bảng đối thủ 0,25 mỗi ô.

Bộ học không biết theta thật và không được gọi luật đối thủ thật để ra quyết định. Tuy nhiên nó **được giả định trước đúng họ nhị thức–logistic và dấu tác động giá** thông qua dạng mô hình và ràng buộc hệ số. Đó là tri thức cấu trúc có lợi. Q-learning không có tri thức ấy, dù hai bên cùng observation, action và ngân sách tương tác. Vì vậy nói “cùng thông tin tương tác” được; nói “hai phương pháp hoàn toàn bằng nhau về mọi prior” là không chính xác.

Đối thủ được dự báo bằng bảng memory-one: chỉ dùng nhu cầu hiện tại và hai action trước. “Có trí nhớ” không có nghĩa mạng nơ-ron nhớ vô hạn hay phân tích toàn bộ lịch sử. Nếu đối thủ thật phụ thuộc năm bước quá khứ, bảng hiện tại có thể thiếu state và dự báo sai.

### 9.3. DP tính từ cuối mùa về đầu mùa

Đặt `V_h(I,d,a_trước,b_trước)` là kỳ vọng lợi nhuận tốt nhất khi còn h bước và đang ở state đó. Ở mỗi action a, bộ lập kế hoạch xem mọi khả năng b của đối thủ, mọi K = 0..12 và mọi d′ = 0/1. Nó tính số bán `min(I,K)`, kho còn và reward, cộng giá trị phần còn lại, nhân xác suất, rồi cộng tất cả. Cuối cùng chọn a có tổng lớn nhất.

Có thể viết gọn:

`Q_h(s,a) = E_b,K,d′ [reward(s,a,b,K) + V_(h−1)(s′)]`,

`V_h(s) = max_a Q_h(s,a)`.

Ở h=1 không còn giá trị tương lai, nhưng vẫn tính lỗ hàng tồn cuối mùa. H bắt đầu từ 1 rồi tăng đến 20, vì muốn biết 20 bước phải biết 19 bước, muốn biết 19 phải biết 18... Máy tính giải từ phần ngắn nhất trước. “Tính ngược” không có nghĩa xem dữ liệu tương lai đã xảy ra; nó dùng các xác suất ước lượng.

Ví dụ trực giác, không phải số đo: nếu một mức giá có hai kết quả lời 10 với xác suất 0,75 và lời 2 với xác suất 0,25 thì kỳ vọng là `0,75×10 + 0,25×2 = 8`. Không lấy trung bình không trọng số bằng 6. Trong Bellman, việc nhân trọng số còn lặp qua giá đối thủ, nhu cầu mua và tín hiệu ngày mai. Chọn duy nhất nhánh “đối thủ có xác suất cao nhất” rồi giải như chắc chắn không phải thuật toán đang dùng.

Model-based khác oracle ở **mô hình đưa vào**, không phải việc có tính DP hay không. Learned planner dùng phân phối fit từ dữ liệu. Oracle dùng theta, ma trận chuyển và luật đối thủ thật. Cả hai tối ưu kỳ vọng theo mô hình được cấp; oracle không đọc sẵn kết quả rút thăm ngày 17 của một mùa cụ thể.

### 9.4. Học sai thì lập kế hoạch cũng có thể sai

Nếu dự báo cuối mùa có cầu cao nhưng test thực tế cao điểm đã xảy ra đầu mùa, planner có thể giữ hàng quá lâu. Nếu dự báo đối thủ từ nominal mà test đối thủ giảm mạnh hơn, xác suất mua của mình có thể bị ước lượng cao. Nếu số liệu ít ở mức giá nào đó, dự báo tại mức giá đó có thể yếu. Một phép DP giải chính xác **mô hình học được** không chứng minh mô hình học được đúng.

Đây là đánh đổi model-based/model-free: mô hình cấu trúc giúp chia sẻ thông tin giữa nhiều state và có thể tiết kiệm mẫu, nhưng sai dạng hoặc dịch chuyển thị trường gây sai lệch; Q-learning ít giả định dạng nhu cầu hơn nhưng bảng lớn khó phủ với ngân sách hữu hạn. Không có quy tắc “có mô hình luôn thắng” hay “không mô hình luôn thích nghi tốt hơn”.

**Trả lời ngắn 40–60 giây:** “Em không dùng doanh số hết hàng như nhu cầu đúng. Nếu còn hai món và bán hết, em chỉ biết cầu ít nhất hai nên dùng xác suất đuôi. Từ dữ liệu quan sát, em học logistic mua hàng, chuyển nhu cầu theo pha và bảng phản ứng đối thủ. DP sau đó cộng reward trước mắt với giá trị tương lai trên mọi khả năng có trọng số. Learner dùng mô hình ước lượng; oracle dùng quy luật thật nhưng không biết trước lần rút thăm. Sai mô hình hoặc đổi thị trường vẫn có thể làm planner kém.”
## 10. So sánh công bằng và đọc bảng/hình

### 10.1. So với ai mới có ý nghĩa?

Một planner chỉ hơn luật cũ yếu chưa đủ thuyết phục. Framework có chín dòng policy để trả lời những câu hỏi khác nhau:

| Tên trong bảng | Nó làm gì? | Vì sao có mặt? |
|---|---|---|
| `original_rule` | Chuyển ý tưởng luật phản ứng từ bài cũ sang bốn mức; nhu cầu cao và đối thủ đã giảm thì dùng mức 20% | Mốc dễ hiểu, nhưng không tận dụng kho và thời gian đầy đủ |
| `stock_rule` | Chọn mức giảm theo hàng còn trên số bước còn lại, điều chỉnh theo d và gần cuối mùa | Đối chứng luật mạnh hơn có ý thức tồn kho; hệ số nhịp bán được chọn bằng validation |
| `fixed_price` | Giữ một mức giá đã chọn qua validation | Kiểm tra liệu phương pháp phức tạp có hơn một giá cố định tốt hay không |
| `q_learning` | Tabular Q với cùng state đầy đủ, bốn action, gamma 1 | Đối chứng model-free; không cố ý ép Q dùng state thiếu kho |
| `myopic_model` | Dùng cùng mô hình đã học nhưng không cộng giá trị tương lai | Kiểm tra vai trò của planning nhiều bước |
| `no_opponent_history` | Thay bảng phản ứng theo lịch sử bằng bảng không phân biệt lịch sử, rồi giải DP | Kiểm tra việc sử dụng thông tin đối thủ |
| `no_stock_feedback` | Tra bảng ở một mức kho cố định thay vì kho thực tế | Kiểm tra phản hồi tồn kho trong quyết định |
| `learned_planner` | Học mô hình từ quan sát rồi giải stochastic DP đầy đủ | Phương pháp đang nghiên cứu |
| `oracle_model` | Giải DP với các xác suất thật của kịch bản | Mốc tối ưu theo thông tin/mô hình, có lợi thế so với learner |

Cần nói chính xác về hai ablation. `no_opponent_history` dùng trung bình **đều** của 16 hàng điều kiện lịch sử cho mỗi d; đây không phải trung bình có trọng số theo tần suất lịch sử thật. Vì thế kết quả vừa phản ánh bỏ lịch sử vừa phụ thuộc cách thay thế phân phối. `no_stock_feedback` vẫn ở trong môi trường có kho, vẫn bị chặn doanh số và vẫn trả phí; nó chỉ không điều chỉnh action theo kho thực tế khi tra policy. Nó không biến môi trường thành vô hạn hàng.

Nếu myopic kém, đó là bằng chứng planning có ích trong thiết kế này. Nếu bỏ lịch sử gần như không giảm kết quả ở một kịch bản, hãy thừa nhận lợi ích của phần đó nhỏ trong kịch bản ấy. Không phải mọi thành phần phải có đóng góp dương rõ trong mọi thị trường thì framework mới có giá trị phân tích.

### 10.2. Train, validation, test là ba việc khác nhau

Train là nơi học tham số hoặc Q. Validation là nơi chọn cấu hình như mức giá cố định, nhịp luật kho, alpha của Q. Test là nơi chấm cuối. Nếu xem điểm test rồi đổi cấu hình cho đẹp, test đã trở thành validation và không còn phép kiểm tra độc lập như dự định.

Validation dùng seed 101, 102, 103. Nó so giá cố định, các scale 0,7/1/1,3 cho luật và alpha 0,1/0,3 cho Q. Q trong validation train 2.000 mùa để giảm chi phí chọn cấu hình; lần chạy cuối train 5.000 mùa. Alpha tốt ở 2.000 mùa chưa chắc tối ưu ở 5.000; đây là giới hạn của bước chọn cấu hình, không nên gọi là Q đã được tối ưu tham số toàn diện.

Final training dùng 10 seed từ 1001 đến 1010. Mỗi seed, calibration random-price để học mô hình nhận 5.000×20 = **100.000 tương tác**, Q-learning cũng nhận **100.000 tương tác** riêng. Cộng cả hai quá trình là 200.000 bước simulator cho một lần lặp, nhưng mỗi learner chỉ có 100.000; các ablation dùng lại mô hình đã fit, không thu thêm dữ liệu. Đây là ngân sách bằng nhau cho mỗi learner, không phải một bên được lấy cả dữ liệu lẫn Q của bên kia. Kho đầu mùa train được thay trong khoảng 30 đến 70 để tăng phạm vi quan sát.

Test dùng seed môi trường tách khỏi train/validation, mỗi seed đánh giá **200 mùa** cho mỗi policy trong mỗi kịch bản, policy đã đóng băng. Có 10 kết quả trung bình cấp seed cho mỗi cặp kịch bản–policy. Nếu đủ 6×10×9 thì có 540 dòng tổng hợp cấp run; mỗi dòng là trung bình của 200 mùa. Các phép tính exact là đánh giá bổ sung bằng mô hình thật sau khi policy đã cố định, không phải thông tin cấp cho quá trình train.

### 10.3. “Cùng seed” chưa đủ; random tape giải quyết chuyện gì?

Trong mã cũ, doanh số và nhu cầu lấy số từ cùng RNG tuần tự. Hai policy có thể tiêu thụ random khác nhau, nên cùng mã seed chưa đảm bảo thấy đúng cùng một chuỗi môi trường theo thời gian. Bài mới sinh trước các mảng chỉ số theo mùa, ngày, khách: tín hiệu nhu cầu, số ngẫu nhiên mua hàng và số ngẫu nhiên chọn giá đối thủ.

Các policy dùng cùng vị trí tape cho cùng mùa/ngày/khách. Giống như cho hai người làm bài dưới cùng bộ tình huống may/rủi để so kết quả đỡ nhiễu. Do giá khác, xác suất mua khác nên doanh số vẫn có thể khác; do lịch sử khác, phân phối đối thủ khác nên b vẫn có thể khác. “Cùng random tape” không có nghĩa ép các policy có cùng doanh số hoặc cùng action đối thủ.

Tape tồn tại trong simulator để tái lập; policy không được mở tape xem tương lai. Tách quyền này là khác biệt giữa mô phỏng dùng số ngẫu nhiên đã sinh sẵn và tác tử gian lận thấy đáp án.

### 10.4. Chênh lệch ghép cặp, SD và CI

Với mỗi seed, lấy profit trung bình 200 mùa của planner trừ profit trung bình của comparator trên cùng tape. Sau đó tính mean/SD/CI trên **10 chênh lệch cấp seed**. Đây là paired comparison: mỗi lần train có một cặp kết quả tương ứng. Dương nghĩa planner hơn comparator; âm nghĩa kém.

SD mẫu dùng mẫu số n−1. CI t 95% dùng `mean ± t_(0,975,n−1) × SD/√n`. Bài cũ n=5 nên t khoảng 2,776; bài mới n=10 nên t khoảng 2,262. Không lấy hệ số của 5 seed áp sang 10 seed. Không lấy 200 mùa hay 2.000 mùa test làm số lần train độc lập để thu CI nhỏ giả; các mùa cùng seed chia sẻ policy đã học.

Nếu paired CI nằm hoàn toàn trên 0, dữ liệu ủng hộ chênh lệch trung bình dương theo quy trình này. Nếu cắt 0, chưa đủ rõ dấu chênh lệch ở độ bất định đang báo; không thể nói hai phương pháp chắc chắn bằng nhau. CI là thủ tục ước lượng qua lặp mẫu, không phải xác suất 95% một policy bất kỳ sẽ thắng ở mọi mùa. Với nhiều kịch bản và comparator, nên xem đầy đủ bức tranh và kích thước hiệu ứng; không săn một ô vừa qua ngưỡng rồi gọi phát hiện phổ quát.

### 10.5. Cách đọc từng nhóm cột

| Cột | Đơn vị/cách đọc | Bẫy thường gặp |
|---|---|---|
| `profit` | Lợi nhuận trung bình **cả mùa** Monte Carlo | So trực tiếp với reward mỗi bước 35 của bài cũ |
| `exact_profit` | Kỳ vọng cả mùa của policy đóng băng dưới mô hình thật | Tưởng learner được biết mô hình thật lúc chọn policy |
| `oracle_exact` | Giá trị tối ưu exact của kịch bản | Tưởng oracle biết trước mọi khách tương lai |
| `exact_gap` | Oracle exact trừ policy exact; nhỏ hơn tốt | Trừ oracle exact với profit Monte Carlo rồi coi là gap chính sách chính xác |
| `leftover` | Kho cuối mùa trung bình | Tự cho bằng 0 là tối ưu dù phải bán quá rẻ |
| `early_stockout` | Tỷ lệ hết kho ở hoặc trước 75% mùa | Tự cho hết kho càng sớm càng tốt |
| `active_discount` | Mức giảm trung bình khi còn hàng | Lẫn với trung bình bao gồm ngày hết hàng quy ước action 0 |
| `lost_units_private` | Cầu tiềm ẩn vượt hàng, chỉ evaluator biết | Đưa vào input learner, làm rò rỉ thông tin |
| `conversion_rmse_referee` | Sai số xác suất mua so với mô hình thật | Tưởng bộ fit dùng sai số đó để chọn theta |
| `censored_rate` | Tỷ lệ quan sát hữu ích bị giới hạn kho theo định nghĩa code | Tưởng mọi sales=cap đều chứng minh cầu lớn hơn cap |
| `fit_converged` | Bộ tối ưu số báo đã thỏa điều kiện dừng | Tưởng điều đó chứng minh học đúng mô hình hoặc policy tối ưu thật |

Exact policy không thể vượt exact oracle nếu cùng state/action và đúng quy luật tối ưu; code kiểm tra điều đó. Nhưng một mẫu profit Monte Carlo có thể vượt oracle exact do may mắn. Hai câu trên không mâu thuẫn: một bên là kỳ vọng, bên kia là mẫu.

### 10.6. Cách nhìn hình và tìm chỗ trong báo cáo

Tìm theo **tên mục**, tên hình hoặc tên CSV; không học thuộc số trang vì dàn trang báo cáo có thể đổi khi bổ sung nghiên cứu. Các điểm nên mở là “Môi trường và tác tử”, “Cấu hình thí nghiệm”, “Kết quả baseline”, “Mở rộng trạng thái và oracle”, “Framework tồn kho”, “Kết quả đánh giá ghép cặp”, “Ablation”, “Giới hạn”.

Với đường học: đọc trục x là episode hay bước, trục y là reward mỗi bước hay profit mỗi mùa, dữ liệu thô hay cửa sổ. Với biểu đồ cột: đọc tên scenario, comparator, độ cao, error bar là SD hay CI. Với heatmap policy: màu biểu diễn action/giá, không nhất thiết biểu diễn lợi nhuận; h nhỏ thường nghĩa gần cuối nếu trục dùng số bước còn lại. Với trace một mùa: xem kho trước/sau, giá, sales và cumulative profit cùng nhau. Một trace minh họa cơ chế; nó không thay thế trung bình nhiều seed.

Hình học được mô hình: sai số chuyển đổi nhỏ là hữu ích nhưng chưa đủ, vì quyết định còn phụ thuộc phân phối đối thủ và chuyển tín hiệu. Hình giá theo h/I: quan sát biên nơi policy đổi giá, giải thích bằng áp lực hết mùa hoặc khan hàng, rồi nhớ đó là policy của mô hình đã học, không phải luật thị trường tổng quát.

**Trả lời ngắn 40–60 giây:** “Em tách train, validation và test. Mỗi learner có 100.000 tương tác; Q cũng nhận đủ state và bốn action. Test đóng băng trên 10 seed, mỗi seed 200 mùa, dùng tape ghép cặp. Em tính chênh lợi nhuận theo từng seed và CI qua 10 seed, không coi từng mùa là một lần huấn luyện độc lập. Oracle chỉ là chuẩn biết mô hình; model-based còn có lợi thế prior về họ logistic nên công bằng ngân sách không có nghĩa mọi giả định giống nhau.”
## 11. Bản đồ tệp và cách đọc mã mà không bị ngợp

Thư mục chính là `E:\WINDOW\Project\intelligent-system-assignment-01\ASIGNMENT-AGENT_02`. Các liên kết dưới đây đi theo vị trí tương đối trong thư mục đó, nên vẫn thuận tiện khi chuyển cả bộ bài sang máy khác.

### 11.1. Mỗi tệp dùng khi nào?

| Tệp/thư mục | Mở khi cần |
|---|---|
| [README.md](README.md) và [DEMO.md](DEMO.md) | Lệnh tái lập và kịch bản demo phần gốc |
| [AGENT02_notebook.html](AGENT02_notebook.html) | Đọc code, output và hình phần cũ, không cần chạy Python |
| [AGENT02_notebook.ipynb](AGENT02_notebook.ipynb) | Đối chiếu cell phần cũ hoặc chạy lại có chủ ý |
| [agent02_experiment.py](agent02_experiment.py) | Source baseline, Q-learning và A–E |
| [agent02_extensions.py](agent02_extensions.py) | Multi-seed, full state, exact/oracle và robustness cũ |
| [extra_results/controlled_summary.csv](extra_results/controlled_summary.csv) | Trung bình/SD/CI của sweep cũ |
| [extra_results/robustness_runs.csv](extra_results/robustness_runs.csv) | Dòng đo từng seed và exact của robustness cũ |
| [research_inventory/inventory_pricing.py](research_inventory/inventory_pricing.py) | Toàn bộ cơ chế môi trường mới, học mô hình, DP, đối chứng và vẽ |
| [research_inventory/INVENTORY_PRICING.ipynb](research_inventory/INVENTORY_PRICING.ipynb) | Notebook nghiên cứu mới, giữ giải thích và output |
| [research_inventory/results](research_inventory/results) | CSV, model/policy lưu sẵn, biểu đồ mới và manifest |
| [work/function_explanations.json](work/function_explanations.json) | Bản giải thích từng hàm của phần cũ |
| [research_inventory/work/inventory_function_explanations.json](research_inventory/work/inventory_function_explanations.json) | Bản giải thích từng hàm của phần mới |
| [work/questions_vi.json](work/questions_vi.json) | Mười câu trả lời đầy đủ theo đề |
| [work/reference_memo.md](work/reference_memo.md) | Đối chiếu tài liệu gốc, phạm vi slide và các giới hạn cần nói đúng |
| [research_inventory/work/literature.md](research_inventory/work/literature.md) | Nguồn nghiên cứu và rà soát thiết kế mở rộng |
| [research_inventory/work/DESIGN_PLAN.md](research_inventory/work/DESIGN_PLAN.md) | Mục tiêu, giả định và giao thức đã chọn cho phần mới |

[Báo cáo Word gốc](agent02_D23CTPM01_Nhom7_NguyenThanhTrung.docx) giữ nguyên 64 trang đã có. Báo cáo mở rộng `agent02_D23CTPM01_Nhom7_NguyenThanhTrung_MO_RONG.docx` bổ sung framework, phương pháp và kết quả mới ở **chương 9**, nguyên notebook mới ở **phụ lục D**, và phần 21 kiểm thử ở **phụ lục E**. [HTML notebook mới](research_inventory/INVENTORY_PRICING.html) thuận tiện khi chỉ cần đọc output. Khi tìm nội dung trong Word, dùng tên mục thay vì số trang. Source và notebook không thay thế báo cáo: source cho biết làm bằng cách nào; bảng/hình cho biết đã đo được gì; phần thảo luận cho biết kết luận được tới đâu.

### 11.2. Mã cũ: đọc theo luồng dữ liệu

Nhóm 1 – **mô tả chợ**. `Config` nhận/thể hiện các tham số, không tự học. `ECommerceEnvironment.__init__` nhận config và seed để tạo RNG và bộ nhớ. `reset` đặt lịch sử về 0, lấy nhu cầu khởi đầu và trả quan sát. `get_states` trả đúng quan sát cho từng bên bằng cách đảo vai mình/đối thủ. `step(actions)` nhận hai action, kiểm tra hợp lệ, tính xác suất mua và doanh số, trả reward rồi cập nhật lịch sử/nhu cầu. Đây là nơi cần mở khi thầy hỏi “lợi nhuận sinh từ đâu?”.

Nhóm 2 – **người quyết định**. `QLearningAgent.__init__` tạo tám Q bằng 0 cùng alpha/gamma/epsilon và RNG. `act(state, explore)` nhận một quan sát, trả action theo epsilon-greedy; `learn` nhận chuyển tiếp và sửa một ô Q; `greedy_policy` xuất bảng action để đọc. `RuleBasedAgent.act` nhận state và trả action bằng điều kiện, không có bước cập nhật. Đây là nơi chỉ ra bằng mã khác biệt giữa học và luật.

Nhóm 3 – **tổ chức cuộc chơi**. `train_rl_against_rule` tạo môi trường/agents, giảm epsilon, cho cả hai chọn rồi chỉ A học; đầu ra là agent đã học và log episode. `evaluate_pair` nhận cặp policy, tạo môi trường đánh giá mới, không gọi learn; trả tổng hợp reward và action cùng log. `main` điều phối toàn bộ. Không nhầm hàm runner với một tác tử thứ ba có mục tiêu kinh doanh.

Nhóm 4 – **lưu và trình bày**. `write_csv` nhận các dictionary và đường dẫn, ghi bảng có header. `save_q_table` ghi tám Q và lựa chọn greedy. `moving_average` nhận chuỗi/cửa sổ, trả đường làm mượt. `plot_training_rewards` và `plot_bar_comparison` nhận dữ liệu và tạo hình, không thay policy. `run_baseline`, `run_learning_rate_experiment`, `run_exploration_experiment`, `run_multi_seed_experiment` tạo cấu hình tương ứng, gọi train/eval và lưu kết quả. Khi nhìn thấy tên `run`, thường đó là điều phối thí nghiệm chứ không phải một công thức học mới.

Nhóm 5 – **mở rộng phân tích** trong `agent02_extensions.py`. `validate_config` kiểm tra đầu vào. `expected_profit` nhận trạng thái/hành động, trả kỳ vọng một bước không lấy mẫu. `oracle_dp` dùng mô hình đã biết giải finite-horizon Bellman. `action_probabilities` chuyển Q thành phân phối greedy đúng với cách phá hòa; `exact_evaluation` trả kỳ vọng policy đã cố định. `HorizonQLearningAgent.__init__/act/learn` lần lượt tạo bảng đầy đủ, chọn action và cập nhật từ mẫu. `train_horizon` dùng lịch sử quan sát để học; `evaluate_extended` đo với Q không đổi. `summarize` gom seed thành mean/SD/CI; `run_controlled_sweep` chạy bảy cấu hình; `run_robustness` đổi mức giảm và so transfer/retrain/full-state/oracle. `load_numeric_log` đọc log đã đo, `make_figures` vẽ từ log. Hàm vẽ không được dùng để tạo một “kết quả đẹp” không tồn tại trong CSV.

### 11.3. Mã mới có bảy khối: đầu vào → xử lý → đầu ra

| Khối | Hàm/lớp | Đầu vào và đầu ra, giải thích dễ hiểu |
|---|---|---|
| 01 – Cấu hình | `PricingConfig`, `__post_init__`, `prices`, `phase` | Nhận horizon, chi phí, kho, mức giảm; kiểm tra điều kiện kinh tế; đổi discount thành giá; đổi ngày đã qua thành pha 0/1. `Scenario` chứa định nghĩa một thị trường test. |
| 02 – Ngẫu nhiên và môi trường | `rival_probabilities` | Nhận loại đối thủ, d và lịch sử, trả bốn xác suất thật. Chỉ simulator/oracle dùng; không gọi làm dự báo cho learner. |
| 02 – Ngẫu nhiên và môi trường | `RandomTape.generate` | Nhận config/scenario/seed/số mùa, trả mảng nhu cầu và các uniform được đánh chỉ số trước. Mục đích tái lập và ghép cặp. |
| 02 – Ngẫu nhiên và môi trường | `MarketEpisode.__init__`, `observe`, `step` | Khởi tạo một mùa; trả state hiện tại; nhận a, sinh b và doanh số có chặn, cập nhật kho, trả next state/reward/terminal/dữ liệu quan sát. |
| 03 – Học dự báo | `censored_loglike` | Nhận sales/cap/số khách/p, trả log xác suất đúng cho quan sát điểm hoặc biến cố đuôi. Fit loại kho 0 khỏi phần ước lượng nhu cầu. |
| 03 – Học dự báo | `ForecastModel.fit` và hàm lồng `objective` | Nhận các dòng quan sát chín cột, trả theta, ma trận chuyển d, phân phối b và chẩn đoán fit. `objective` trả loss/gradient để bộ tối ưu số cập nhật theta. |
| 03 – Học dự báo | `ForecastModel.pmf` | Nhận config và d,a,b, trả 13 xác suất cho K=0..12; không chỉ trả K trung bình. |
| 03 – Mô hình đặc quyền | `perfect_model` | Nhận scenario, dựng model từ kernel thật cho benchmark/evaluator. Không nối vào dữ liệu train của learner. |
| 04 – Lập kế hoạch | `DPSolution` | Gói đầu ra Q, V, policy và thời gian tính; không phải thuật toán độc lập. |
| 04 – Lập kế hoạch | `one_step_values` | Nhận một state và model, trả bốn kỳ vọng reward một bước; thuận tiện kiểm tra terminal bằng phép tính riêng. |
| 04 – Lập kế hoạch | `solve_inventory_dp` | Nhận config/model và tùy chọn, trả bảng Bellman. `planning=False` bỏ future cho myopic. `fixed_policy` giữ action đã cho để chỉ đánh giá giá trị, không tối ưu lại. |
| 05 – Q và luật | `InventoryQLearner.__init__`, `action`, `learn` | Tạo Q cùng shape đầy đủ; chọn epsilon-greedy; nhận một chuyển tiếp để cập nhật với gamma 1 và terminal đúng. |
| 05 – Huấn luyện | `train_models` | Nhận seed/alpha, chạy calibration random-price và Q với ngân sách bằng nhau; trả model, learner Q, dữ liệu quan sát, đường train và thời gian. |
| 05 – Chuẩn hóa policy | `build_policy_table` | Nhận tên phương pháp/model/Q hoặc tham số luật, trả bảng action trên cùng hệ chỉ số. Nó tạo cả luật, giá cố định và các ablation. |
| 05 – Đánh giá | `evaluate_policy` | Nhận scenario/tape/policy, chạy không học, trả metrics, từng mùa và trace minh họa; kiểm tra policy không đổi. |
| 05 – Đánh giá exact | `exact_policy_profit` | Nhận model thật và policy đã khóa, trả kỳ vọng profit từ trạng thái đầu. Dành cho giám khảo sau học. |
| 05 – Xuất dữ liệu | `save_rows`, `aggregate_runs` | Một hàm ghi CSV; hàm kia nhóm các run rồi tính mean/SD/CI theo seed, không train thêm. |
| 06 – Validation | `choose_validation` | Nhận config/thư mục output, thử lựa chọn trên seed dành riêng, lưu `locked_choices.json` và trả cấu hình khóa. |
| 06 – Thí nghiệm cuối | `run_inventory_experiments` | Nhận lựa chọn đã khóa, train 10 seed và đánh giá chín policy trên sáu kịch bản; trả/lưu raw runs, paired differences, diagnostic, manifest. |
| 07 – Hình | `inventory_figures` và hàm lồng `save` | Nhận kết quả thật, tạo tám PNG; `save` chỉ chỉnh bố cục, lưu và đóng hình. Các hàm nhỏ `mean/sd/groups` phục vụ gom và vẽ dữ liệu. |

Tổng số ô của Q mới lớn hơn nhiều: mảng gồm `(21,71,2,4,4,4)`, tức 190.848 Q lưu cả h=0; số ô ở h=1..20 là 181.760 nếu tính cả kho 0; riêng h>0 và I>0 là 179.200 ô. Đây là đếm theo giới hạn cấu hình, bao gồm các state có thể hiếm/không gặp. Từ đó hiểu vì sao Q toàn trạng thái có thể cần nhiều mẫu; không dùng kích thước bảng để chứng minh Q chắc chắn kém.

### 11.4. Theo một dòng dữ liệu để kiểm tra không rò rỉ

Một dòng train chứa `phase,d,a,b,sales,cap,next_d,own_previous,rival_previous`. b hiện tại được ghi **sau** bước để fit. Không có theta thật, `potential`, `lost_units`, hay một d của hai ngày sau. Quy trình fit chỉ đọc các cột này. Khi chọn giá, policy chỉ nhận năm biến state hiện tại. Khi đánh giá exact, giám khảo mới dùng true model. Hãy phân biệt ba “vai”: simulator được biết nhiều để tạo thế giới; learner được thấy ít như hợp đồng; evaluator có quyền biết thêm để chấm nhưng không gửi ngược thông tin.

### 11.5. Demo mà không huấn luyện lại

Đối với phần cũ, mở HTML lưu sẵn output, hình `extra_results/evaluation_ci.png`, Q-table và `robustness.png`. Chỉ rõ công thức update và baseline cùng giữ giá. Với phần mới, mở output của notebook đã thực thi hoặc các PNG trong `research_inventory/results`, rồi mở CSV cùng tên kịch bản để đối chiếu một cột. Một trace lưu sẵn đủ để giải thích kho giảm theo ngày. Model/policy cũng được lưu trong `model_seed_*.npz` và `learned_policy_*.npy`; không cần học từ đầu để chứng minh có kết quả.

Đừng bấm “Run All” chỉ để xem hình trong giờ vấn đáp: nó khởi động lại thí nghiệm, có thể mất thời gian và ghi lại output. Xem output đã lưu là demo kết quả có dấu vết; nếu thầy yêu cầu tái lập, chỉ lệnh và giao thức trong README. Có thể chạy kiểm thử khi môi trường Python đã sẵn; đọc rõ output vừa chạy trước khi nói pass.

Một demo 60 giây có thể đi theo bốn điểm: 10 giây nêu state/action; 15 giây chỉ một ô lợi nhuận và đơn vị; 20 giây chỉ trace kho/giá để giải thích cơ chế; 15 giây nói giới hạn và so sánh. Nếu chỉ có một phút, không dành nửa thời gian cài thư viện.

**Trả lời ngắn 30–50 giây:** “Source mới chia bảy khối: cấu hình, môi trường, học dự báo, DP, đối chứng, thí nghiệm và hình. Em lần theo state đi vào action, môi trường trả quan sát, model fit từ quan sát, rồi policy đóng băng đem test. File CSV và output notebook đã lưu đủ để demo; không cần retrain. Mô hình thật chỉ ở simulator/oracle/evaluator, không ở input của learner.”
## 12. Mười câu hỏi của đề: khung trả lời đúng trọng tâm

Bảng này giúp nối câu hỏi gốc với phần đã học; không thay thế diễn giải trong báo cáo. Khi thầy hỏi một câu, hãy lấy ý đầu rồi thêm đúng một ví dụ.

| Câu | Ý cần trả lời | Ví dụ và giới hạn nên nhớ |
|---|---|---|
| 1. Mô hình tác tử/môi trường | Hai seller, quan sát cục bộ, hai action, thưởng profit, lựa chọn đồng thời | Bốn quan sát × hai action = tám Q; local observation chưa chắc Markov |
| 2. Học so với luật | Q sửa theo trải nghiệm; Rule có điều kiện cố định | Rule phản ứng không có nghĩa Rule học; RL cần ngân sách train |
| 3. Q-learning | Cũ + alpha × sai số target; target có reward và future | Tính 4,65; bước terminal bỏ future |
| 4. Khám phá/khai thác | Thử để biết thêm và dùng hành động tốt hiện tại | Epsilon 0,2 với hai action không đồng nghĩa 20% action khác greedy |
| 5. Reward kinh doanh | Chọn profit thay doanh số vì margin thay đổi khi giảm | Margin 5 xuống 3; bán nhiều chưa chắc lời hơn |
| 6. Đánh giá | Mean reward test; tắt khám phá, không learn, nhiều seed | Baseline 35,1275/34,9685 không chứng minh policy khác nhau |
| 7. Alpha/ổn định | Alpha lớn sửa nhanh và nhạy nhiễu; cần nhìn final eval | Ba mức 0,01/0,10/0,50; đường phẳng không chứng minh hội tụ |
| 8. Đa tác tử/không dừng | Action bên kia đổi reward; hai learner cùng đổi làm môi trường hiệu dụng thay đổi | Fixed opponent chưa bảo đảm state rút gọn Markov; chưa chạy hai learner đồng thời |
| 9. Hạn chế và sửa đã làm | Nêu giả định toy, rồi chỉ modification thực thi | Cũ đã đổi mức giảm 20/10/5 và tách transfer/retrain; mới thêm inventory framework |
| 10. Cải tiến và kiểm chứng | State/time/reward/method cùng giao thức test và robustness | Full-state Q cũ và learned planner mới là hai gói khác; phải nói đúng ablation và ưu thế thông tin oracle |

Đề AGENT02 trực tiếp yêu cầu A–E, mười câu trả lời, một modification được đề xuất **và thực thi** ở câu 9, cùng báo cáo đủ thiết lập/kết quả/thảo luận. Phần tồn kho, học mô hình bị chặn, dự báo đối thủ và stochastic DP là hướng mở rộng của bài này. Không trình bày mọi ý tưởng trong slide MAS như điều kiện bắt buộc của đề riêng nếu giảng viên chưa giao thêm.

Slide tutorial có bài ba agent, catalog/recommender/coordinator, classifier PyTorch và lỗi thông điệp. Bài hai seller không tự nhiên trở thành tutorial LLM/PyTorch đó. Bộ cũ bổ sung README, sơ đồ, kiểm thử và demo để tăng tính tái lập, nhưng không tuyên bố hoàn thành những bài ứng dụng khác. Nếu thầy hỏi vì sao không có LLM: mục tiêu ở đây là quyết định giá trên state nhỏ, đo reward rõ; LLM không cần thiết cho cơ chế đang kiểm tra.

**Trả lời ngắn 20–40 giây:** “Phần đề bắt buộc là triển khai Q-learning đấu luật, A–E, mười câu và một thay đổi có thực thi. Em giữ phần đó rồi bổ sung nghiên cứu tồn kho với học mô hình và planning. Đây là mở rộng ứng dụng và kiểm chứng của bài lab, không phải phát minh Q-learning hoặc quy hoạch động mới.”

## 13. Ngân hàng 50 câu vấn đáp: luyện hỏi ngẫu nhiên

Hãy đọc câu hỏi rồi che đáp án, nói trong 20–40 giây. Nếu không nhớ số, nói đúng phương pháp và mở bảng, thay vì đoán. Các đáp án dưới đây là điểm tựa; được nói bằng lời tự nhiên.

### Nhóm A – Nền tảng và bài cũ

**1. Em đang giải quyết bài toán gì?**  
Chọn giá để tối đa lợi nhuận khi có người bán cạnh tranh. Bài gốc so người bán học Q-learning với người bán theo luật. Bản mở rộng thêm kho hữu hạn và mùa bán, học nhu cầu/phản ứng đối thủ rồi tính chính sách cân bằng lợi nhuận hiện tại với cơ hội tương lai.

**2. Vì sao gọi là hệ đa tác tử nếu không gửi tin nhắn?**  
Có hai thực thể chọn hành động theo policy riêng và có lợi ích riêng. Hành động giảm giá của một bên làm đổi xác suất bán của bên còn lại. Tương tác qua thị trường chung cũng là tương tác đa tác tử; không bắt buộc có hội thoại hoặc LLM.

**3. State, action, reward của baseline là gì?**  
Quan sát là nhu cầu thấp/cao và action trước của đối thủ. Action là giữ hoặc giảm giá 20%. Reward là số bán nhân biên lợi nhuận. Có bốn quan sát, hai action nên tám Q. Tuy vậy biểu diễn này thiếu own previous action để thành state đầy đủ theo luật phản ứng đang có.

**4. Rule-based có phải không thông minh?**  
Không nên đánh giá bằng nhãn đó. Rule có thể hợp lý, dễ giải thích và rất mạnh nếu luật phù hợp. Khác biệt là nó không tự sửa từ reward. Trong baseline, giữ giá là hành vi có kỳ vọng tốt nên luật không nhất thiết kém Q-learning.

**5. Q khác reward như thế nào?**  
Reward là tiền lời của bước vừa xảy ra. Q ước lượng tổng giá trị từ một lựa chọn trong state, gồm ảnh hưởng các bước sau. Một reward xấu vì may rủi không đủ nói Q phải âm hoặc action đó luôn xấu.

**6. Giải thích phép tính 4,65.**  
Reward 6 cộng gamma 0,9 nhân Q tốt nhất kế là 5 cho target 10,5. Q hiện tại 4 nên sai số 6,5. Alpha 0,1 lấy 10% sai số, cộng 0,65 vào 4 được 4,65. Đây là một lần cập nhật, không phải lợi nhuận kinh doanh 4,65.

**7. Alpha lớn có chắc học tốt hơn?**  
Không. Nó cập nhật mạnh hơn, có thể phản ứng nhanh nhưng nhạy với mẫu sales ngẫu nhiên. Alpha nhỏ cần nhiều mẫu để thay đổi. Em so ba mức trên cùng điều kiện, xem frozen evaluation và biến thiên seed thay vì chỉ kết luận từ trực giác.

**8. Gamma 0,9 có nghĩa gì? Vì sao có lúc dùng 1?**  
Gamma 0,9 làm giá trị ở bước sau có trọng số 90% so với cùng lượng thưởng ngay bây giờ. Trong mùa hữu hạn, gamma 1 tối ưu tổng profit không chiết khấu và vẫn hữu hạn nhờ terminal. Đó là thay mục tiêu nên cần phân biệt với chỉ thêm state.

**9. Epsilon 0,2 có phải 20% thời gian chọn sai?**  
Không. Nó là xác suất chọn ngẫu nhiên. Với hai action, nhánh ngẫu nhiên vẫn có thể chọn greedy; nếu greedy duy nhất thì tỷ lệ chọn action còn lại là 10%. Tốt/xấu còn là điều chưa biết chính xác khi đang học.

**10. Vì sao cần exploration khi chỉ có hai action?**  
Chỉ hai action vẫn có doanh số nhiễu. Nếu một action tình cờ kém lúc đầu và không bao giờ được thử lại, Q có thể giữ đánh giá sai. Exploration cho dữ liệu đối chiếu, nhưng quá nhiều exploration làm lợi nhuận train thấp vì cứ thử.

**11. Seed là gì? Đặt 42 có làm kết quả tốt hơn không?**  
Seed khởi tạo bộ sinh số giả ngẫu nhiên để tái lập. Số 42 không có tính tốt đặc biệt. Nếu chọn seed vì nó cho kết quả đẹp thì làm sai đánh giá; cần nhiều seed được khai báo và giữ cả kết quả không đẹp.

**12. Vì sao giảm giá bán nhiều hơn chưa chắc lời hơn?**  
Trong bài cũ, lời mỗi món giảm từ 5 xuống 3 khi giảm 20%. Phần tăng số bán phải đủ bù giảm margin. Ví dụ nhu cầu cao và đối thủ giữ giá: giữ có kỳ vọng 55, giảm có 42. Đây là phép tính một bước dưới giả định mô hình.

**13. Baseline RL có thắng không?**  
Trong một đánh giá seed 42, mean mẫu là 35,1275 so với 34,9685 mỗi bước. Nhưng cả hai không giảm giá và kỳ vọng đều 35. Nói có chênh mẫu; không dùng số đó chứng minh RL có policy tốt hơn hay thắng phổ quát.

**14. Cumulative reward đi lên có chứng minh học được không?**  
Không, cộng nhiều reward dương tự làm tổng tăng. Cần xem độ dốc, trung bình cửa sổ, policy cuối và frozen evaluation. Đường làm mượt giúp đọc nhưng có thể che dao động và không thêm bằng chứng độc lập.

**15. Frozen evaluation làm những gì?**  
Tạo môi trường đánh giá riêng, tắt nhánh exploration, không gọi learn và kiểm tra Q/policy trước sau giống nhau. Môi trường vẫn ngẫu nhiên. Như vậy em đo policy sau train, không trộn việc học thêm hoặc lợi nhuận do thử hành động vào bài kiểm tra.

### Nhóm B – State, Markov và oracle

**16. Em tìm ra thiếu Markov như thế nào?**  
Lấy hai lịch sử cùng cho quan sát nhu cầu cao, đối thủ trước giữ giá, nhưng own previous action khác nhau. Luật B hiện tại sẽ chọn khác, nên reward distribution khác. Quan sát rút gọn bỏ thông tin cần thiết; fixed opponent chưa đủ để kết luận nó Markov.

**17. 320 Q và 336 Q số nào đúng?**  
Cả hai đúng theo cách đếm. Có 20 mức h hoạt động, hai nhu cầu, hai action trước mỗi bên và hai action hiện tại nên 320. Mảng có thêm h=0 terminal nên lưu 336. Cần nói rõ “hoạt động” hay “lưu cả terminal”.

**18. Thêm state có chắc luôn tốt?**  
Nó có thể làm biểu diễn đúng hơn nhưng tăng số ô cần dữ liệu. Với ngân sách hữu hạn, nhiều state hiếm làm Q ước lượng kém. Kết quả cũ đã cho thấy gói full-state không thắng mọi trường hợp; không được suy từ state đầy đủ sang ưu thế thực nghiệm tự động.

**19. Ngẫu nhiên với không dừng khác nhau ra sao?**  
Ngẫu nhiên là mỗi lần sales khác dưới cùng quy luật. Không dừng là quy luật phân phối hiệu dụng đổi theo thời gian, ví dụ đối thủ cùng học và đổi policy. Pha mùa vụ có thể được đưa vào state thời gian để mô hình hóa chuyển đổi có cấu trúc; không nên gọi mọi dao động là đối thủ học.

**20. Oracle có gian lận không?**  
Nếu dùng làm đối thủ ngang quyền để tuyên bố learner kém thì không công bằng về thông tin. Ở đây oracle được ghi rõ là mốc biết mô hình thật. Nó không biết trước các random realization; chỉ tối ưu kỳ vọng. Khoảng cách đến nó đo còn thiếu bao nhiêu so với chuẩn có thông tin tốt hơn.

**21. Vì sao exact oracle có thể thấp hơn profit Monte Carlo của learner?**  
Một bên là kỳ vọng, một bên là mẫu may/rủi. Một tập mùa may mắn có thể cao hơn kỳ vọng tối ưu. Khi tính exact cho cả policy learner lẫn oracle dưới cùng mô hình và action set, learner không được vượt oracle ngoài sai số số học.

**22. Oracle cũ 35; 35,1; 37,35 nói gì?**  
Đó là kỳ vọng tối ưu mỗi bước của episode hữu hạn ở các mức giảm 20%, 10%, 5%. Môi trường giữ nguyên phần tăng chuyển đổi khi thay phần trăm giảm. Nó cho mốc phân tích của mô hình, không phải lợi nhuận đo ở thị trường thật.

**23. Ablation cũ đã chứng minh riêng tác dụng của state chưa?**  
Chưa hoàn toàn. Full-state agent còn đổi h, gamma và alpha. `local_gamma1` ở mức 10% chỉ kiểm tra một phần thay mục tiêu. Vì vậy kết luận chính về gói thay đổi; muốn tách riêng từng yếu tố cần thêm biến thể kiểm soát.

### Nhóm C – Framework tồn kho và dữ liệu

**24. Vì sao phải biết kho và số ngày còn lại?**  
Cùng nhu cầu nhưng ít hàng nên cân nhắc giữ giá để dành cơ hội; nhiều hàng gần hết mùa nên cân nhắc giảm để tránh thu hồi rẻ. h và I quyết định áp lực bán. Thiếu chúng, một policy khó phân biệt các tình huống kinh tế khác nhau.

**25. Bốn action mới là những giá nào?**  
Với giá gốc 10, giảm 0%, 5%, 10%, 20% tương ứng 10; 9,5; 9; 8. Action là chỉ số 0..3. Hệ không được tự chọn một giá bất kỳ ngoài tập đó, nên oracle cũng bị giới hạn cùng bốn action.

**26. Có phải cả hai seller đều có kho hữu hạn?**  
Không. Kho hữu hạn áp dụng cho cửa hàng tiêu điểm đang nghiên cứu; đối thủ được giả định nguồn cung đủ. Đây là đơn giản hóa cần nói rõ. Mô hình chưa nghiên cứu đồng thời hai kho cạn, nhập hàng lại hoặc cả hai doanh nghiệp cùng tối ưu inventory.

**27. Nếu chỉ bán được 2 với kho 2, em học demand = 2 à?**  
Không, em chỉ biết cầu ít nhất 2. Likelihood phải dùng xác suất đuôi từ 2 trở lên. Nếu kho 5 và bán 2 thì biết cầu đúng 2 theo mô hình này. Nhầm hai tình huống gây đánh giá cầu thấp khi hay hết hàng.

**28. Hết kho rồi bán 0 có làm mô hình học cầu thấp đi không?**  
Không nên. Kho 0 ép sales 0 ở mọi mức cầu nên không có thông tin về chuyển đổi mua; fit lọc những dòng đó khỏi phần nhu cầu. Tín hiệu d và giá đối thủ vẫn có thể giúp bảng chuyển/đối thủ, nên cần phân biệt loại thông tin.

**29. Nhị thức và logistic mỗi cái làm nhiệm vụ gì?**  
Logistic đổi tín hiệu và giá thành xác suất một khách mua, nằm 0..1. Nhị thức dùng xác suất ấy và 12 khách để tạo phân phối tổng khách muốn mua. Sau đó lấy min với kho. Logistic không trực tiếp trả số món bán.

**30. Agent mới có biết hệ số thật không?**  
Learner không biết theta thật; nó fit từ capped sales và giá đã công bố sau bước. Nhưng nó biết trước họ logistic/nhị thức và dấu tác động giá. Cần thừa nhận đó là prior cấu trúc thuận lợi so với Q, không nói hai phương pháp ngang nhau mọi tri thức.

**31. Nhìn giá đối thủ sau bước có vi phạm chọn đồng thời không?**  
Không, miễn là giá đó chỉ dùng để cập nhật mô hình hoặc state cho các bước sau. Khi chọn a hiện tại, learner chỉ có phân phối dự báo b từ d và lịch sử. Nếu lén dùng b hiện tại trước quyết định thì đổi hợp đồng thông tin và làm kết quả sai.

**32. Reward mới có tính trùng giá vốn không?**  
Không. Mỗi món đã bán bị trừ cost trong margin; món chưa bán cuối mùa bị trừ cost−salvage. Tổng bằng doanh thu trừ toàn bộ vốn kho đầu, cộng thu hồi cuối, trừ giữ kho. Ví dụ 60−50+4−2=12 kiểm tra trực tiếp được.

**33. Vì sao không phạt lost sales thật trong reward?**  
Nhu cầu vượt kho không quan sát đầy đủ. Nếu dùng lost demand nội bộ để thưởng/phạt rồi cho learner học, nó nhận thêm thông tin không hợp đồng. Bài chỉ dùng lost_units_private cho evaluator phân tích; reward dựa vào khoản thu chi quan sát được.

**34. Planner dự báo một con số demand rồi dùng con số đó à?**  
Không. Nó dùng phân phối K=0..12, phân phối b và d′, nhân trọng số rồi cộng. Vì kho và giá trị tương lai phi tuyến, lấy `min(E[K],I)` thường không bằng `E[min(K,I)]`. Một dự báo điểm không thay phép Bellman ngẫu nhiên hiện tại.

**35. Tại sao cần planning, không chỉ chọn lời lớn nhất hôm nay?**  
Bán một món làm giảm kho ngày sau và có thể đổi phản ứng đối thủ. Planning tính giá trị cơ hội đó; myopic bỏ phần future. Gần cuối mùa, terminal còn tính lỗ hàng tồn. Có đối chứng myopic dùng cùng model để kiểm tra lợi ích nhiều bước.

**36. Nếu dự báo mô hình rất chính xác, tại sao vẫn có thể kém?**  
Sai số nhỏ ở chuyển đổi nominal không đảm bảo dự báo đối thủ hoặc mùa vụ đúng ngoài nominal. Ngoài ra phải xem tính phù hợp mục tiêu và action set. DP tối ưu đúng mô hình đầu vào; khi thị trường đổi khác mô hình, quyết định tối ưu dự báo có thể không tối ưu thực tế test.

### Nhóm D – Công bằng và thống kê

**37. So với Q có công bằng không?**  
Q có cùng năm biến quan sát, cùng bốn action, reward và 100.000 tương tác. Tuy nhiên model-based có prior logistic/nhị thức, còn cách thu dữ liệu random-price khác online Q. Em kiểm soát ngân sách và hợp đồng quan sát, không tuyên bố mọi giả định và cơ chế học hoàn toàn ngang nhau.

**38. Vì sao phải có luật kho mạnh và giá cố định?**  
Để kiểm tra phương pháp phức tạp có hơn lựa chọn đơn giản hợp lý hay chỉ hơn một luật quá yếu. Scale luật và fixed price được chọn trên validation riêng. Dùng baseline mạnh làm kết luận khó hơn nhưng đáng tin hơn.

**39. Ba tập train/validation/test giúp tránh gì?**  
Train học tham số, validation chọn cấu hình, test chấm cuối. Nếu đổi cấu hình sau khi nhìn test thì thông tin test rò vào quyết định. Seed tách riêng và lựa chọn được lưu trước test giúp giữ quy trình kiểm tra độc lập.

**40. Q đã được tune tốt nhất chưa?**  
Không thể khẳng định. Validation chỉ so một tập nhỏ alpha và dùng 2.000 mùa; final dùng 5.000. Cấu hình khóa là lựa chọn theo ngân sách đó, không phải tối ưu toàn diện mọi alpha/epsilon. Kết luận so sánh gắn với implementation và tuning đã nêu.

**41. Vì sao tape ghép cặp tốt hơn chỉ ghi cùng seed?**  
Nó gán trước số ngẫu nhiên theo mùa/ngày/khách, nên policy khác không làm lệch thứ tự lấy số vì tiêu thụ RNG khác. Hai bên gặp cùng nền may/rủi nhưng do action khác vẫn có sales/rival khác. Tape không được cấp cho learner để xem tương lai.

**42. Vì sao CI tính từ 10 seed, không từ 2.000 mùa?**  
Các mùa cùng seed dùng chung policy được học. Nếu muốn phản ánh độ biến thiên quá trình học, đơn vị độc lập chính là seed train. Mỗi seed lấy trung bình 200 mùa rồi gom 10 trung bình/chênh lệch. Coi mọi mùa độc lập về huấn luyện sẽ làm bất định trông nhỏ quá.

**43. SD và CI khác thế nào?**  
SD mô tả kết quả giữa seed phân tán đến đâu. CI thể hiện bất định của mean theo thủ tục thống kê, thường nhỏ hơn khi có thêm seed. CI không phải khoảng chứa 95% mùa và không cho xác suất 95% planner thắng mọi tình huống.

**44. Paired CI cắt 0 thì kết luận gì?**  
Chưa có đủ rõ ràng về dấu mean difference theo mức bất định đó. Không chứng minh hai phương pháp bằng nhau. Cần xem kích thước chênh, SD, số seed, exact và cơ chế; cũng tránh săn ý nghĩa ở một trong rất nhiều ô.

**45. `no_opponent_history` có phải chỉ đơn giản quên đối thủ?**  
Nó vẫn dùng một phân phối action đối thủ, nhưng bỏ phân biệt lịch sử bằng cách lấy trung bình đều 16 hàng điều kiện. Không phải bảng unconditional có trọng số tần suất. Cách thay thế này cần nêu khi diễn giải ablation; không gán mọi chênh lệch cho riêng một nguyên nhân lý tưởng.

**46. Agent mới có luôn tốt hơn không?**  
Không có lý do bảo đảm vậy. Nó có thể được lợi trong mùa đúng cấu trúc học nhưng bị hại khi nhu cầu đổi pha hoặc đối thủ thay luật. Hãy đọc đủ sáu kịch bản và báo cả thất bại. Kết quả tốt ở nominal không biến thành định lý thắng mọi thị trường.

### Nhóm E – Đóng góp, nguồn và trình bày

**47. Em có phát minh thuật toán mới không?**  
Không tuyên bố như vậy. Q-learning, quy hoạch động, học mô hình và likelihood cho dữ liệu bị chặn đều là ý tưởng đã có. Đóng góp của bài là xây mô phỏng, kết hợp chúng theo hợp đồng thông tin rõ, triển khai đối chứng/ablation và phân tích kết quả trong phạm vi bài lab.

**48. Tài liệu LLM MAS có liên quan thế nào?**  
Nó cung cấp bối cảnh về agents, tương tác, cấu trúc cạnh tranh và đánh giá hệ nhiều agent. Bài hiện tại không dùng prompt, debate hoặc LLM. Không lấy thành tích benchmark LLM làm bằng chứng Q-learning hoặc planner bán hàng tốt hơn; phân loại model-based của hai ngữ cảnh cũng không hoàn toàn đồng nghĩa.

**49. Em chưa làm những gì?**  
Chưa triển khai thị trường thật, chưa dùng dữ liệu bán hàng thật, chưa có LLM/DQN, chưa hai learner đồng thời, chưa hai kho hữu hạn cùng tối ưu, chưa nhập thêm hàng và chưa chứng minh tính mới học thuật hoặc ưu thế phổ quát. Chưa làm không phải lỗi nếu phạm vi đã nói rõ.

**50. Nếu chỉ có một phút để demo em mở gì?**  
Mở output đã lưu hoặc hình profit/paired difference, nêu đơn vị và comparator; mở trace kho–giá để giải thích cơ chế; chỉ source state/step hoặc Bellman nếu được hỏi. Không retrain trong giờ. Kết bằng giới hạn phù hợp, chẳng hạn cấu trúc nhu cầu đã giả định và đổi thị trường có thể làm policy kém.

**Trả lời ngắn 20–40 giây:** “Em luyện trả lời theo ý nghĩa, ví dụ và giới hạn. Những câu dễ nhầm nhất là doanh số khác nhu cầu khi hết kho, cùng ngân sách khác cùng prior, oracle biết luật khác biết tương lai, và một CI hay một seed không chứng minh thắng phổ quát. Nếu quên con số, em mở bảng có đơn vị thay vì đoán.”

## 14. Phạm vi nguồn, điều chưa làm và cách tự kiểm tra hiểu thật

Ba tài liệu cũ được dùng đúng phạm vi: bài tổng quan RL của Naeem và cộng sự giúp giải thích state/action/reward, Q-learning và các hạn chế; survey LLM MAS của Tran và cộng sự giúp đặt bối cảnh hệ nhiều agent; slide tutorial cung cấp cách đặc tả, kiểm thử và trình bày. Đề AGENT02 mới là nguồn yêu cầu trực tiếp của bài này. Bản ghi nguồn trong `work/reference_memo.md` nêu rõ tài liệu không ghi tác giả/năm nào, vị trí trích và điểm không được suy rộng.

Phần nghiên cứu mới dựa vào các hướng đã có. Gallego và van Ryzin (1994) nghiên cứu giá động với kho hữu hạn và hạn bán, nhưng mô hình gốc không giống hoàn toàn mô phỏng rời rạc có đối thủ ở đây. Bu, Simchi-Levi và Wang nghiên cứu học nhu cầu với dữ liệu bị chặn; đó không phải bằng chứng trực tiếp cho thuật toán logistic trong bài. Dyna của Sutton là tiền lệ ý tưởng kết hợp học mô hình và lập kế hoạch. Quy trình ở đây là thu dữ liệu theo lô → fit model → giải toàn bộ DP hữu hạn → đánh giá đóng băng; nó không phải triển khai canonical Dyna-Q cập nhật Q online bằng chuyển tiếp mô phỏng. Dyna chỉ được dẫn như một ý tưởng thuộc họ phương pháp; phần ghi nguồn chỉ xác nhận metadata/abstract cho tài liệu đó khi không đọc được toàn văn. Cần nói đúng mức đã đọc, không nhận đã đọc mọi bài mà một survey trích dẫn.

Phạm vi hiện tại là hai người bán tương tác qua thị trường mô phỏng; một learner đối diện đối thủ xác suất cố định. Có seasonal signal được mô hình hóa theo pha, nhưng chưa có sự thích nghi liên tục của learner trong test. Policy test đã đóng băng. Bản DP được tính trước, sau đó mỗi bước tra policy bằng state mới thực tế; nó không cố định một chuỗi giá bất chấp kho, không tối ưu nhìn lại kết quả đã xảy ra và không phải tìm kiếm kịch bản Monte Carlo. Không có neural network, không GPU bắt buộc, không bộ sinh ngôn ngữ, không kênh tin nhắn tự nhiên, không cập nhật giá trong một hệ thống thương mại thật. Số liệu simulator giúp kiểm tra ý tưởng trong giả định đã công bố; không dự báo được lợi nhuận doanh nghiệp thật chỉ bằng đổi đơn vị tiền.

Tự kiểm tra bằng năm việc nhỏ:

1. Vẽ bảng bốn quan sát và hai action, tự chỉ tám ô Q.
2. Tính được 4,65 mà không nhìn đáp án; nói terminal thay target như thế nào.
3. Nói lại phản ví dụ Markov bằng hai lịch sử A hôm qua khác nhau.
4. Tính lời tồn kho 12 bằng cả hai cách hạch toán, giải thích vì sao sales 2/stock 2 chỉ cho cầu ≥2.
5. Mở một dòng CSV, nói được scenario, agent, seed, metric và đơn vị; giải thích nó là mẫu hay exact, một mùa hay trung bình nhiều mùa.

Nếu làm được năm việc này và nói rõ giới hạn, em đã có nền để ứng biến. Không cần thuộc tất cả tên hàm. Khi được hỏi source, tìm đúng khối và lần theo đầu vào–đầu ra. Khi được hỏi kết quả, chỉ đúng bảng. Khi được hỏi tương lai, phân biệt đề xuất với việc đã thực hiện. Khi một kịch bản trái kỳ vọng, giải thích điều dữ liệu cho thấy thay vì sửa câu chuyện để giữ lời hứa “luôn tốt hơn”.

**Trả lời ngắn 20–40 giây:** “Đây là bài triển khai và đánh giá trong mô phỏng. Em dùng nguồn để xây nền khái niệm và thiết kế, còn kết luận hiệu năng phải từ kết quả của chính code. Em không nhận đã dùng LLM, chạy thị trường thật hoặc phát minh phương pháp mới. Điều em cần hiểu là dữ liệu nào agent biết, lợi nhuận tính ra sao và bằng chứng đủ cho kết luận nào.”



## 15. Kết quả mới đã đo: biết con số, hiểu điều nó chứng minh

Phần này đọc từ bộ kết quả cuối của **10 seed**, không phải seed đầu tiên hay số giả định. `inventory_manifest.json` ghi 540 run = 6 kịch bản × 10 seed × 9 policy; mỗi run đánh giá 200 mùa. Mỗi learner train 100.000 tương tác. Validation đã khóa **fixed action 1**, tức giá 9,5; **scale luật kho 1,3**; **alpha Q 0,1**. Các kết luận dưới đây gắn với cấu hình và phạm vi sáu kịch bản đã khai báo. Họ binomial–logistic và theta đều giữ đúng trong cả sáu, nên chưa phải bằng chứng chống sai dạng mô hình cầu hoặc hiệu quả trên dữ liệu thật.

### 15.1. Bảng nominal: nên đọc từ trái sang phải

Mỗi con số profit là lợi nhuận **cả mùa 20 bước**. Mean/SD/CI lấy qua 10 seed, mỗi seed trung bình 200 mùa. Exact là kỳ vọng policy đã đóng băng dưới mô hình thật, do evaluator tính sau học.

| Policy | Profit MC mean | SD giữa seed | CI t 95% của mean | Exact profit mean |
|---|---:|---:|---:|---:|
| Luật gốc | 179,1463 | 5,9495 | [174,8902; 183,4023] | 178,9486 |
| Luật kho mạnh | 203,3746 | 2,8666 | [201,3239; 205,4252] | 203,5300 |
| Giá cố định đã chọn | 195,8642 | 2,6433 | [193,9733; 197,7551] | 195,9988 |
| Q-learning full state | 184,3252 | 2,0144 | [182,8842; 185,7661] | 183,7896 |
| Myopic cùng model | 145,5223 | 0,1844 | [145,3904; 145,6542] | 145,5261 |
| Bỏ lịch sử đối thủ | 205,4574 | 2,2001 | [203,8835; 207,0312] | 205,2638 |
| Bỏ phản hồi kho | 161,9880 | 1,6445 | [160,8116; 163,1645] | 162,1631 |
| Learned planner | **209,7607** | 2,5950 | [207,9044; 211,6171] | **209,9600** |
| Oracle biết mô hình | 209,9235 | 2,6360 | [208,0378; 211,8092] | 209,9634 |

Cách nói: “Ở nominal, planner đạt khoảng 209,76 profit một mùa, hơn luật kho khoảng 6,39 và hơn Q khoảng 25,44 theo chênh lệch ghép cặp. Exact policy gần oracle trong mô hình chuẩn.” Không nói “vượt oracle” vì nhìn một seed hoặc so một cột mẫu với cột exact khác. Không nói Q-learning nói chung chỉ được 184; đây là Q tabular và giao thức tuning cụ thể của thí nghiệm.

Exact gap nominal của planner là **0,003421 profit/mùa** trung bình qua seed. Nó rất nhỏ so với oracle 209,9634, cho thấy policy fit tốt khi thị trường test đúng cấu trúc nominal đã học. Con số nhỏ này không đảm bảo từng hệ số, từng state hoặc mọi thị trường tương lai đều hoàn hảo. Model-based đang có prior đúng họ nhu cầu, cùng nguồn dữ liệu calibration đa dạng.

### 15.2. Chênh lệch ghép cặp: hai kịch bản chính

| Kịch bản | So sánh planner trừ… | Mean chênh profit/mùa | CI t 95% trên 10 chênh lệch |
|---|---|---:|---:|
| nominal | Luật kho | +6,3862 | [5,8718; 6,9006] |
| nominal | Q-learning | +25,4356 | [24,5136; 26,3575] |
| nominal | Myopic cùng model | +64,2384 | [62,4696; 66,0073] |
| nominal | Bỏ lịch sử đối thủ | +4,3034 | [3,8454; 4,7614] |
| nominal | Bỏ phản hồi kho | +47,7727 | [45,9261; 49,6193] |
| late_surge | Luật kho | +8,5387 | [8,0824; 8,9949] |
| late_surge | Q-learning | +29,9852 | [28,9374; 31,0329] |
| late_surge | Myopic cùng model | +72,2427 | [71,6292; 72,8562] |
| late_surge | Bỏ lịch sử đối thủ | +8,7036 | [8,3260; 9,0812] |
| late_surge | Bỏ phản hồi kho | +63,3636 | [61,9112; 64,8159] |

Ở đây CI là CI của **hiệu ghép cặp**, không phải lấy hai CI độc lập trừ bừa hai đầu. Thí dụ nominal planner và luật có độ nhiễu profit riêng khoảng vài đơn vị, nhưng dùng cùng tape làm chênh cặp ổn định hơn. Có thể kiểm tra từng seed trong `paired_differences.csv`, rồi xem tổng hợp ở `paired_summary.csv`.

Các khoảng trên đều dương trong hai thị trường chính, ủng hộ ưu thế trung bình của planner trước những đối chứng đó theo giao thức này. Đây không phải hiệu năng thị trường thật, cũng không phải phân tích đã hiệu chỉnh đầy đủ cho mọi phép so sánh nhiều kịch bản. Hãy trình bày mean, độ bất định và cơ chế cùng nhau.

### 15.3. Đọc toàn bộ sáu thị trường, giữ cả khoảng cách còn lại

| Kịch bản | Planner MC | Luật kho MC | Q MC | Oracle MC | Exact gap planner |
|---|---:|---:|---:|---:|---:|
| nominal | 209,7607 | 203,3746 | 184,3252 | 209,9235 | 0,0034 |
| scarce | 143,2861 | 141,6265 | 118,8404 | 143,2896 | 0,0001 |
| abundant | 231,1357 | 204,9990 | 173,8932 | 231,1768 | 0,0108 |
| late_surge | 217,6177 | 209,0790 | 187,6325 | 219,8899 | 2,3238 |
| early_peak | 232,0865 | 231,3396 | 206,2981 | 235,4798 | 3,5338 |
| aggressive_rival | 177,4123 | 152,4515 | 158,0273 | 187,1963 | 9,1239 |

Trong cả sáu kịch bản này, mean ghép cặp planner trừ luật kho và trừ Q đều dương. Không có thất bại trước hai baseline đó để dựng một câu chuyện “đã thua” giả. Tuy nhiên ưu thế trước luật kho rất nhỏ ở early_peak: **+0,7468**, CI **[0,3894; 1,1043]**; scarce là **+1,6597**, CI **[1,4980; 1,8213]**. Mức cải thiện thay đổi đáng kể theo thị trường.

Sự suy giảm khi chuyển thị trường thể hiện rõ hơn ở **exact gap tới oracle**: từ gần 0 ở nominal/kho thay đổi lên khoảng 2,32 khi tăng cầu cuối, 3,53 khi cao điểm đầu và 9,12 khi đổi đối thủ. Planner vẫn hơn các baseline đã chọn nhưng mất một phần cơ hội so với người biết mô hình đúng của thị trường mới. Câu kết đúng là “khả năng chuyển giao có ích nhưng không hoàn hảo”, không phải “đã thích nghi biết hết thị trường mới”. Test đang đóng băng, không học thêm để nhận ra luật mới.

Đừng so profit scarce 143 với abundant 231 để nói scarce có thuật toán kém hơn: lượng vốn và số hàng ban đầu khác nhau. So policy với policy **trong cùng kịch bản** trước; nếu so xuyên kịch bản phải giải thích thay điều kiện kinh tế.

### 15.4. Cơ chế trong số liệu: vì sao nhìn trước và biết kho giúp?

Ở nominal, myopic giảm giá trung bình khi còn hàng khoảng **19,13%**, gần mức giảm mạnh nhất; planner khoảng **4,34%**. Myopic có tỷ lệ hết hàng tại hoặc trước ngày 15 là **75,20%**, planner **2,75%**. Profit myopic chỉ 145,52 dù cuối mùa gần như không còn hàng; planner 209,76 với tồn cuối trung bình 0,8335 món. Điều đó minh họa việc “bán hết, bán nhanh” không tự tối đa profit.

Biến thể bỏ phản hồi kho có nominal profit 161,99 và early_stockout **51,30%**. Nó vẫn dự báo/planning nhưng chọn giá như thể kho vẫn ở mức đầu, nên không điều chỉnh đúng khi hàng cạn. Biến thể bỏ lịch sử đối thủ còn profit 205,46, thấp hơn planner khoảng 4,30; phần này giúp nhưng tác động nominal nhỏ hơn nhiều so với bỏ planning hoặc phản hồi kho. Nhớ cách lấy trung bình đều trong ablation để không diễn giải quá sạch.

Ở late_surge, planner early_stockout khoảng **0,35%**, myopic **75,20%**; đây là nơi giữ cơ hội đến cuối mùa rất có ý nghĩa. Ở early_peak, planner hết sớm **31,00%**, nhưng profit vẫn cao: cơ hội nằm đầu mùa nên không nhất thiết phải giữ kho đến sát ngày 20. Myopic hết sớm **99,80%** nhưng profit còn thấp hơn rất nhiều vì bán biên mỏng. Early_stockout là tín hiệu cần giải thích theo kịch bản, không phải mục tiêu/penalty riêng.

Khi đối thủ aggressive, planner tồn cuối trung bình **2,8175 món**, cao hơn nominal. Cùng lúc exact gap lớn nhất trong sáu kịch bản. Điều này phù hợp với việc policy học từ đối thủ nominal chưa phản ánh đúng thị trường mới; chỉ nhìn tồn cuối chưa đủ phân biệt mọi nguyên nhân, nên đọc cùng profit, gap và cơ chế xác suất.

### 15.5. Mô hình đã học tốt đến đâu, tính toán mất bao lâu?

Trung bình bốn hệ số fit qua 10 seed là khoảng **(−2,3982; 1,7997; 8,0007; −4,5078)**, gần các hệ số simulator nominal **(−2,4; 1,8; 8; −4,5)**. RMSE xác suất chuyển đổi trung bình do evaluator tính khoảng **0,000907**; tỷ lệ quan sát nhu cầu hữu ích bị chặn theo định nghĩa log trung bình khoảng **4,80%**. Tất cả 10 lần fit có cờ hội tụ bộ tối ưu bằng 1. Các số này hỗ trợ chất lượng fit trong cấu trúc nominal; không chứng minh khả năng nhận dạng ở mọi bộ dữ liệu hoặc mọi mức censoring.

Trên máy chạy lần này, thời gian fit trung bình khoảng **1,16 giây**, giải DP khoảng **0,074 giây**; phần `train_models` thu dữ liệu và train Q trung bình khoảng **28,72 giây** mỗi seed. Thời gian đó không bao gồm toàn bộ exact evaluation, sinh hình, dựng notebook và báo cáo. Nó là đo trên môi trường hiện tại, không phải cam kết mọi máy sẽ nhanh như nhau. Khi vận hành policy đã có, mỗi bước chủ yếu là tra bảng theo state mới, không chạy lại fit/DP toàn bộ.

Để nói trong vấn đáp, không cần thuộc sáu chữ số: “mô hình nominal học rất gần quy luật mô phỏng, DP tính nhanh trên không gian nhỏ; nhưng prior đúng dạng và test thay đối thủ làm gap lớn hơn.” Phần số chính nên nhớ là **209,76 / 203,37 / 184,33** cho planner/luật kho/Q ở nominal, cùng đơn vị profit một mùa và bằng chứng ghép cặp.

### 15.6. Mở đúng hình để giải thích đúng điều

- [01_nominal_profit.png](research_inventory/results/01_nominal_profit.png): bảng xếp profit nominal với CI t qua 10 seed. Dùng để chỉ đối chứng mạnh và đơn vị một mùa.
- [02_paired_advantage.png](research_inventory/results/02_paired_advantage.png): hiệu planner trừ luật/Q theo từng thị trường với CI ghép cặp; vạch 0 cho biết hướng chênh.
- [03_market_quality.png](research_inventory/results/03_market_quality.png): exact profit chia oracle của từng thị trường. Ô gần 100% thể hiện gần chuẩn trong mô hình; không phải tỷ lệ “dự báo đúng khách”. Nhãn phần trăm được làm tròn số nguyên, nên planner và oracle có thể cùng hiện 100% dù exact gap nominal của planner vẫn 0,003421. **100% do làm tròn không có nghĩa đạt tối ưu tuyệt đối.**
- [04_ablation_gap.png](research_inventory/results/04_ablation_gap.png): gap exact khi bỏ thành phần; thanh lỗi ở hình này là **±1 SD**, không phải CI 95%.
- [05_inventory_trace.png](research_inventory/results/05_inventory_trace.png): một mùa test đã khai báo của late_surge, seed 1001; đường kho và mức giảm giúp hiểu cơ chế, không đại diện thống kê mọi mùa.
- [06_policy_map.png](research_inventory/results/06_policy_map.png): giá theo kho và h, tách nhu cầu thấp/cao, cố định hai action trước bằng 0 và dùng model seed 1001. Trục x là kho; trục h tăng lên trên nghĩa còn nhiều bước hơn, không phải ngày đã qua. Nhiều hàng hoặc ít bước còn lại thường tạo áp lực giảm mạnh hơn để bán kịp. Đây là lát cắt policy, không phải mọi lịch sử.
- [07_demand_calibration.png](research_inventory/results/07_demand_calibration.png): so xác suất thật với xác suất học, error bar ±1 SD giữa seed; true conversion chỉ dành evaluator.
- [08_training_profit.png](research_inventory/results/08_training_profit.png): profit trong thu dữ liệu random-price và Q train theo cửa sổ 100 mùa. Đường calibration không phải profit của policy planner sau khi fit; phải xem frozen evaluation để đánh giá planner.

Các bảng gốc để kiểm tra lại là [inventory_summary.csv](research_inventory/results/inventory_summary.csv), [paired_summary.csv](research_inventory/results/paired_summary.csv), [model_diagnostics.csv](research_inventory/results/model_diagnostics.csv) và [inventory_manifest.json](research_inventory/results/inventory_manifest.json). Có thể đối chiếu từng seed ở `inventory_runs.csv` và từng mùa trong các tệp `episodes_...csv`.

**Trả lời ngắn 40–60 giây:** “Ở nominal, planner đạt khoảng 209,76 profit một mùa, luật kho 203,37 và Q 184,33; chênh ghép cặp với luật khoảng 6,39, CI 95% từ 5,87 đến 6,90. Myopic hay giảm mạnh và hết hàng sớm nên lợi nhuận thấp, cho thấy giá trị của planning và phản hồi kho. Planner hơn hai baseline này trong sáu kịch bản đã thử, nhưng mức hơn luật rất nhỏ ở early_peak, và exact gap tăng đến khoảng 9,12 khi đối thủ aggressive. Vì vậy em kết luận trong phạm vi mô phỏng đã đo, không nói luôn thắng.”



