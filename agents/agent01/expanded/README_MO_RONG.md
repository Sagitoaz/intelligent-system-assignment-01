# DISCOVERING AI AGENT — Bộ bài thực hành mở rộng

Nguyễn Thành Trung — B23DCCN861 — D23CTPM01.

## Tệp nên mở

- Báo cáo Word: agent01_D23CTPM01_NguyenThanhTrung_B23DCCN861_MO_RONG.docx.
- Notebook đã chạy: agent01_notebook_mo_rong.ipynb.
- Bản xem trình duyệt: agent01_notebook_mo_rong.html.
- Số đo, dữ liệu và hình: artifacts/.

## Quy mô

41 ô mã Python, 815 dòng mã, phần giải thích trước và sau từng ô; 31 kiểm tra chức năng; 400 yêu cầu tổng hợp cho mỗi trong ba cấu hình; 90 mô hình Q-learning/ablation, tổng 135.000 episode huấn luyện.

## Chạy lại

Dùng Python 3.12 cùng numpy, pandas, matplotlib, scikit-learn. requirements-notebook.txt ghi phiên bản đã chạy. Với Jupyter, cần kernel Python đúng môi trường. Mở notebook, Restart Kernel & Run All. Notebook tự tạo thư mục artifacts và dựng lại tệp SQLite thử nghiệm ở đường dẫn đã định nghĩa.

Notebook tự chứa toàn bộ mã; không cần các tệp build_*.py hoặc base_cells.py để chạy. Các tệp này chỉ giúp tác giả tạo và kiểm tra báo cáo.

## Nội dung

1. Phản xạ, mô hình nội bộ, mục tiêu và độ hữu dụng.
2. Dữ liệu 360 đơn, 400 kịch bản; chính sách và truy xuất tài liệu.
3. SQLite, quyền theo phiên mô phỏng, phê duyệt quản lý, giao dịch và chống trùng.
4. Timeout trước/sau commit, rollback, đọc cũ, ghi chú không tin cậy.
5. Benchmark và ablation retry; tách phản hồi khỏi giao dịch đã ghi.
6. TF-IDF + Logistic Regression cho ý định tiếng Việt; tách họ mẫu; coverage và guard phủ định.
7. RL ba bản đồ, mười seed, BFS, random, epsilon ablation, gamma ablation và nhiễu hành động.

## Phạm vi

Dữ liệu tổng hợp, danh tính phiên mô phỏng, chưa thanh toán bên ngoài, planner theo quy tắc, không sử dụng LLM. Truy xuất chính sách theo từ khóa chưa phải một hệ thống RAG có mô hình sinh. Các guard ngôn ngữ chỉ bao phủ mẫu được định nghĩa, chưa bảo đảm hiểu ngữ nghĩa tổng quát.

Các tỷ lệ tốt trong tập kiểm thử hữu hạn không chứng minh khả năng tổng quát ngoài môi trường đã mô tả. Kết quả ablation được giữ nguyên kể cả khi không xác nhận giả định ban đầu. Báo cáo giải thích chi tiết các giới hạn này.
