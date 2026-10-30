# Demo AGENT02 trong 6 đến 8 phút

1. Mở sơ đồ, giải thích hai người bán chọn hành động đồng thời. Nêu state, action, profit và luật của đối thủ.
2. Mở cell định nghĩa QLearningAgent. Giải thích epsilon-greedy, TD target và terminal. Tính 4 + 0,1 × (6 + 0,9 × 5 − 4) = 4,65.
3. Mở baseline evaluation: RL 35,1275 và Rule 34,9685, nhưng cả hai đều không giảm giá. Kỳ vọng hai bên cùng bằng 35. Không kết luận RL luôn thắng.
4. Mở alpha_curves và evaluation_ci. So sánh alpha 0,01 / 0,10 / 0,50, giải thích mean và SD giữa năm seed.
5. Mở exploration_curves. Phân biệt reward training với greedy evaluation; epsilon 0,30 không có nghĩa 30% hành động chắc chắn khác greedy.
6. Nêu phản ví dụ Markov: cùng (d=1, b_previous=0), own_previous bằng 0 hoặc 1 làm đối thủ hiện tại chọn 0 hoặc 1. Mở HorizonQLearningAgent và giải thích trạng thái mới.
7. Mở robustness.png và bảng CSV. Phân biệt chuyển policy, huấn luyện lại và oracle; giải thích exact với Monte Carlo. Thừa nhận agent mới có thể kém hơn do bảng thưa và ngân sách hữu hạn.
8. Chạy `python -m unittest discover -s work -p test_agent02.py -v`: có 14 kiểm thử. Ba nhóm lỗi là đầu vào mô phỏng, không giả timeout mạng.
9. Kết bằng giới hạn mô hình và hướng kiểm tra tiếp: đối thủ cùng học, nhu cầu Markov hoặc ablation từng yếu tố. Không trình bày các ý tưởng đó như thí nghiệm đã làm.

Có thể dùng notebook hoặc HTML lưu sẵn output để demo nhanh. Không cần chạy lại 77 lượt train ngay trong giờ trình bày. README ghi đầy đủ lệnh tái lập.
