# AGENT02 — Hướng dẫn sử dụng

## Sản phẩm

- `agent02_D23CTPM01_Nhom7_NguyenThanhTrung.docx`: báo cáo tiếng Việt, đủ 10 câu hỏi, giải thích từng hàm và cell, kèm nguyên notebook.
- `AGENT02_notebook.ipynb`: notebook đã thực thi, giữ mã, output và hình.
- `AGENT02_notebook.html`: bản đọc nhanh không cần Jupyter.
- `agent02_experiment.py`: bản nguyên mẫu từ `agent02_code.tex`, có đối chiếu SHA256.
- `agent02_extensions.py`: thí nghiệm nhiều seed, tác tử trạng thái mở rộng, oracle và kiểm tra độ bền.
- `agent02_results/`: baseline và thí nghiệm mẫu.
- `extra_results/`: log train/eval, Q-table, summary CSV, hình và manifest.

## Chạy lại

Mã mẫu dùng cú pháp Python từ 3.10. Môi trường ghim phiên bản để tái lập lần chạy này cần **Python 3.12 trở lên**, vì NumPy 2.5.2 yêu cầu phiên bản đó.

Trong thư mục chứa notebook:

```powershell
python -m venv .venv_agent02
.\.venv_agent02\Scripts\Activate.ps1
python -m pip install -r requirements_agent02.txt
$env:PYTHONIOENCODING='utf-8'
python work/execute_notebook.py
python -m unittest discover -s work -p test_agent02.py -v
```

Bộ chạy tự tạo kernelspec cục bộ theo Python đang dùng. Có thể mở notebook trong Jupyter hoặc VS Code, chọn Python của môi trường trên rồi Restart Kernel and Run All. Cell thí nghiệm có đầy đủ thuật toán; cell kiểm thử dùng tệp phụ trợ đã kèm. Toàn bộ quy trình gồm 77 lượt huấn luyện, mỗi lượt 100.000 bước, nên cần nhiều phút tùy CPU.

Chỉ chạy bản mẫu: `python agent02_experiment.py`.

Tái tạo notebook từ nguồn bằng `python work/build_notebook.py`, sau đó `python work/execute_notebook.py`. Việc tái tạo sẽ xóa output cũ; nên giữ bản đã chạy trước khi làm.

## Giao thức

Baseline dùng 5.000 episode × 20 bước, alpha 0,10, gamma 0,90 và epsilon = max(0,02; 0,20 × 0,995^episode). Seed môi trường là 42, seed policy 10042. Evaluation có 500 episode × 20 bước, seed 500042, explore=False và không cập nhật Q.

Sweep gồm 7 cấu hình × 5 seed: 11, 22, 33, 44, 55. Seed đánh giá bằng seed train + 800000. Robustness thay mức giảm giá 20%, 10%, 5%, dùng năm seed và seed evaluation bằng seed train + 900000. Có 65 lượt đánh giá robustness, gồm ablation một phần tại 10%.

SD mẫu và CI t 95% tính giữa năm seed; biến thiên chứa cả training và evaluation. Cùng seed tạo quy trình tương ứng, không bảo đảm mọi policy gặp chuỗi nhu cầu giống hệt nhau vì doanh số và nhu cầu dùng chung RNG.

- `reference_transfer`: Q học tại 20%, đóng băng khi chuyển mức giảm giá.
- `reference_retrained`: học lại từ đầu trong môi trường mới.
- `horizon_q`: state gồm h, nhu cầu và hai hành động trước; gamma=1, alpha=N^(-0.6), học từ mẫu.
- `oracle_dp`: biết mô hình, làm chuẩn hữu hạn; có lợi thế thông tin so với learner.
- `local_gamma1`: state cục bộ, chỉ đổi gamma, dùng ở 10% để làm ablation một phần.

`exact_mean` là kỳ vọng giải tích của policy đã học; `mean_reward` là reward đo bằng rollout Monte Carlo. `regret` = oracle exact − policy exact; không trừ trực tiếp reward lấy mẫu khỏi oracle để đánh giá gap.

## Báo cáo và bài nộp

Mọi code cell trong Word được đối chiếu nguyên chuỗi với notebook. Thông tin bìa lấy từ báo cáo AGENT01 có sẵn: Nguyễn Thành Trung, B23DCCN861, D23CTPM01, Nhóm 7. Kiểm tra lại trước khi nộp.

`work/make_report_figures.py` tạo hình tích lũy từ log và sơ đồ kiến trúc. `work/build_report.py` dựng Word; `work/report_body.py` giữ nội dung chính; `work/questions_vi.json` giữ 10 câu trả lời. Cần python-docx và đủ kết quả trước khi dựng Word. Báo cáo được kiểm tra bố cục bằng Word trên Windows và ảnh PNG từ pipeline render_docx/Poppler. Các tệp QA trung gian không phải bài nộp.

Sau khi chỉnh sửa Word, cập nhật mục lục rồi xuất PDF theo mẫu `agent02_class_team_name.pdf`. Tên gợi ý là `agent02_D23CTPM01_Nhom7_NguyenThanhTrung.pdf`. Hạn theo đề: **23:30 ngày 29/10/2026**. Bộ bàn giao giữ DOCX để người dùng tự xuất PDF.

## Phạm vi tài liệu slide

Đã đọc năm PDF và mã nguồn trong thư mục. Các bài tutorial về ít nhất ba agents, intent classifier PyTorch và lỗi thông điệp thuộc ứng dụng khác với AGENT02 hai người bán. Bộ này bổ sung phần tương thích: PEAS, sơ đồ, README, 14 kiểm thử có ba nhóm đầu vào lỗi và hướng dẫn demo. Không tuyên bố đã triển khai tutorial LLM, PyTorch hoặc kiểm thử inventory cũ.
