# Project 01: thầy hỏi gì thì mở file nào

Mỗi file `.py` trong `project01/` đều có khối **"BẢN ĐỒ FILE"** ở đầu (chức năng, dữ liệu đọc/ghi, nơi được gọi) và docstring cho từng hàm. Số dòng dưới đây theo bản đã chú thích.

## 1. Câu hỏi thường gặp

| Thầy hỏi | Mở file | Hàm / vị trí |
|---|---|---|
| Mô hình Deep Learning là gì, kiến trúc ra sao? | `train/train_dl.py` | `build_model` (50); các bước chia dữ liệu, chuẩn hóa, baseline, huấn luyện, early stopping, `decision_scale`, đánh giá trong `main` (77) |
| Khi chạy demo, MLP suy luận thế nào (không cần torch)? | `mas/dl_agent.py` | `_forward` (62), `predict` (78), `index` (75) |
| 21 đặc trưng đầu vào là gì, chuẩn hóa ở đâu? | `mas/data.py` | `FEATURES`, `normalize` (97), `to_vector` (130) |
| Kết quả mô hình (accuracy, F1, AUC)? | `artifacts/dl_metrics.json` | do `train/train_dl.py` ghi |
| Knowledge Base (đồ thị tri thức) nằm đâu? | `train/build_kg.py` | `N` (39, 41 nút), `E` (83, 54 cạnh), `main` (102) ghi `artifacts/kg.json` và `kg_neo4j.cypher` |
| RAG hoạt động thế nào? BM25 là gì? | `mas/kb.py` | `BM25` (70), `scores` (88), `search` (100) |
| Graph-RAG khác BM25 thường chỗ nào? | `mas/kb.py` | `match_nodes` (134), `expand` (149), `search` (162) |
| RAG đo chất lượng bằng gì? | `train/eval_rag.py` | `main` (33): Hit@1, Hit@3, MRR; kết quả ở `artifacts/rag_metrics.json` |
| Chức năng "Hỏi bác sĩ" (RAG + Qwen)? | `mas/ask.py` | `Retriever` (177), `AskEngine.ask` (439), `QwenRunner` (264), `extractive_answer` (402) |
| Chống bịa, chặn câu hỏi cá nhân hoặc cấp cứu? | `mas/ask.py` | `PERSONAL_RX` (129), `EMERGENCY_RX` (131), `answer_problem` (329), `BAD_PHRASES` (326) |
| Agent Reinforcement Learning là gì? | `mas/rl.py` | môi trường `step` (135), thưởng `zone_reward` (92), `QCoach` (227), `train` (274) |
| Huấn luyện Q-learning ở đâu, siêu tham số? | `train/train_rl.py` | `main` (39): alpha 0,15, gamma 0,97, 60.000 episode |
| So sánh RL với luật và không có huấn luyện viên? | `mas/rl.py` | `rule_coach` (193), `no_coach` (212), `compare` (324) |
| Luật an toàn (rule-based)? | `mas/rules.py` | `evaluate` (33): luật R1–R10 |
| Bác sĩ phản biện, tranh luận hoạt động ra sao? | `mas/critic.py` | `review` (104): kiểm tra C1–C4; `_debate` (182): tối đa 2 vòng |
| Các agent phối hợp thế nào? | `mas/orchestrator.py` | `AGENTS` (37), `Clinic.consult` (113) |
| What-if (đổi BMI thì nguy cơ đổi sao)? | `mas/whatif.py` | `what_if` (35), `bmi_curve` (56) |
| Độ tin cậy và công bằng của mô hình? | `train/eval_trust.py`, `mas/trust.py` | `reliability` (87, ECE và Brier), `group_table` (127); API ở `trust.routes` |
| Hàng đợi bệnh nhân, theo dõi 4 tuần? | `mas/followup.py` | `triage` (40), `analyze` (208) |
| Báo cáo hình và bảng, xuất PDF? | `mas/report.py`, `mas/pdf_report.py` | `build_report` (120), `build_pdf` (171) |
| Nhập bằng giọng nói? | `mas/stt.py` | `parse_speech` (193), `recognize` (340, gọi Azure) |
| Đọc kết quả bằng giọng nói? | `mas/tts.py` | `VOICES` (37), `Speaker.synthesize` (95) |
| API có những endpoint nào? | `server.py` | bảng endpoint ở đầu file; `Handler` (182), `ROUTES_POST` (158) |

## 2. Hai điểm dễ bị hỏi

- **Hai corpus RAG.** Tab Thủ thư (`kb.py`) dùng `artifacts/diabetes_corpus_niddk183.jsonl` (183 đoạn, bản đóng băng). "Hỏi bác sĩ" (`ask.py`) và Bác sĩ phản biện (`critic.py`) dùng `data/chapter04/diabetes_corpus.jsonl` (492 đoạn, 28 trang). Số Hit@1 trong báo cáo đo trên bản 183 đoạn.
- **Tranh luận của Bác sĩ phản biện** (`critic._debate`) dựng hội thoại từ mẫu câu có sẵn dựa trên bằng chứng đã tính, không phải hai mô hình ngôn ngữ nói chuyện với nhau.
- Truy hồi của `kb.py` là BM25 theo từ khóa, không có embedding. Riêng `ask.py` có thêm truy hồi vector, hợp nhất bằng RRF.

## 3. Dữ liệu và kết quả đã lưu (`artifacts/`)

| File | Nội dung |
|---|---|
| `dl_model.npz`, `dl_metrics.json` | Trọng số MLP và số đo |
| `rl_qtable.json`, `rl_metrics.json` | Q-table và số đo của huấn luyện viên RL |
| `kg.json`, `kg_neo4j.cypher` | Đồ thị tri thức (41 nút, 54 cạnh) |
| `diabetes_corpus_niddk183.jsonl` | Corpus 183 đoạn cho Thủ thư |
| `rag_metrics.json`, `ask_eval.json`, `trust_metrics.json` | Số đo RAG, hỏi bác sĩ, tin cậy và công bằng |

## 4. Chạy lại

Xem `RUN.md` mục 0 để chạy phòng khám. Huấn luyện và đánh giá lại (chạy từ thư mục gốc):
`python project01/train/train_dl.py` (cần torch), `train_rl.py`, `build_kg.py`, `eval_rag.py`, `eval_trust.py`.
