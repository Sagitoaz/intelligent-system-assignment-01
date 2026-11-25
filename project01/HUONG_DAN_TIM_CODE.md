# Project 01: thầy hỏi gì thì mở file nào

Mỗi file `.py` trong `project01/` đều có khối **"BẢN ĐỒ FILE"** ở đầu (chức năng, dữ liệu đọc/ghi, nơi được gọi) và docstring cho từng hàm. Số dòng dưới đây theo bản đã chú thích.

## 1. Câu hỏi thường gặp

| Thầy hỏi | Mở file | Hàm / vị trí |
|---|---|---|
| Mô hình Deep Learning là gì, kiến trúc ra sao? | `train/train_dl.py` | `build_model` (50); các bước chia dữ liệu, chuẩn hóa, baseline, huấn luyện, early stopping, `decision_scale`, đánh giá trong `main` (77) |
| Khi chạy demo, MLP suy luận thế nào (không cần torch)? | `mas/dl_agent.py` | `_forward` (62), `predict` (78), `index` (75) |
| 21 đặc trưng đầu vào là gì, chuẩn hóa ở đâu? | `mas/data.py` | `FEATURES`, `normalize` (97), `to_vector` (130) |
| Kết quả mô hình (accuracy, F1, AUC)? | `artifacts/dl_metrics.json` | do `train/train_dl.py` ghi |
| Knowledge Base (đồ thị tri thức) nằm đâu? | `train/build_kg.py` | `N` (41 nút), `E` (54 cạnh), `main` ghi `artifacts/kg.json`; `write_cypher` ghi 3 file Cypher vào thư mục `db_diabet_cypher/` (xem `db_diabet_cypher/README.md`) |
| KB lưu trên Neo4j thế nào, nạp ra sao? | `db_diabet_cypher/`, `scripts/push_to_neo4j.py` | `python scripts/push_to_neo4j.py --only kb`: nút `KBNode`, `KBSource`, `KBPassage`, quan hệ `MENTIONS` (bằng chứng) |
| Nguồn dữ liệu: 3 link + PDF lấy ở đâu? | `mas/kb.py` (docstring đầu file), `train/build_pdf_corpus.py` | web: `scripts/chapter04/build_diabetes_corpus.py` (28 URL); PDF: `data/project01/niddk_4_steps_manage_diabetes.pdf` -> `artifacts/diabetes_pdf_passages.jsonl` |
| RAG nối với KB qua code ở đâu? (Neo4j) | `mas/neo4j_kb.py`, `mas/kb.py` | `Neo4jKB.neighbors`, `Neo4jKB.evidence` (Cypher); `KnowledgeBase.expand`, `KnowledgeBase.evidence`; chọn backend bằng `KB_BACKEND` |
| Demo KB + RAG trên Neo4j | `train/demo_kb_neo4j.py` | in số nút/đoạn trên Neo4j, câu Cypher đã chạy và trích dẫn web/PDF |
| RAG hoạt động thế nào? BM25 là gì? | `mas/kb.py` | `BM25`, `BM25.scores`, `BM25.search` |
| Graph-RAG khác BM25 thường chỗ nào? | `mas/kb.py` | `match_nodes`, `expand`, `evidence`, `KnowledgeBase.search` |
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

- **Một kho RAG duy nhất.** `kb.py` (Thủ thư), `ask.py` (Hỏi bác sĩ) và nút KBPassage trên Neo4j cùng dùng kho 514 đoạn = 492 đoạn web (`data/chapter04/diabetes_corpus.jsonl`) + 22 đoạn PDF (`artifacts/diabetes_pdf_passages.jsonl`). (`critic.py` vẫn đọc riêng 492 đoạn web.) Trước đây Thủ thư dùng bản 183 đoạn cũ nên số đo bị lệch; nay đã gộp.
- **Neo4j là nguồn thật của đồ thị khi có kết nối.** `KB_BACKEND=auto` (mặc định) thử Neo4j rồi rơi về `kg.json`; `neo4j` bắt buộc; `file` không cần mạng. Văn bản đoạn để BM25 vẫn đọc từ file jsonl cho nhanh.
- **Tranh luận của Bác sĩ phản biện** (`critic._debate`) dựng hội thoại từ mẫu câu có sẵn dựa trên bằng chứng đã tính, không phải hai mô hình ngôn ngữ nói chuyện với nhau.
- Truy hồi của `kb.py` là BM25 theo từ khóa, không có embedding. Riêng `ask.py` có thêm truy hồi vector, hợp nhất bằng RRF.
- Trên 40 câu hỏi, Graph-RAG của `kb.py` **không hơn** BM25 thuần (xem `artifacts/rag_metrics.json`); báo cáo ghi trung thực kết quả này.

## 3. Dữ liệu và kết quả đã lưu (`artifacts/`)

| File | Nội dung |
|---|---|
| `dl_model.npz`, `dl_metrics.json` | Trọng số MLP và số đo |
| `rl_qtable.json`, `rl_metrics.json` | Q-table và số đo của huấn luyện viên RL |
| `kg.json` | Đồ thị tri thức (41 nút, 54 cạnh), bản dự phòng khi không có Neo4j; bản Cypher nằm ở `db_diabet_cypher/` |
| `diabetes_pdf_passages.jsonl` | 22 đoạn trích từ file PDF NIDDK (nguồn PDF của KB) |
| `demo_kb_neo4j.json` | Kết quả chạy `train/demo_kb_neo4j.py` (số nút/đoạn trên Neo4j, câu Cypher mỗi câu hỏi) |
| `rag_metrics.json`, `ask_eval.json`, `trust_metrics.json` | Số đo RAG, hỏi bác sĩ, tin cậy và công bằng |

## 4. Chạy lại

Xem `RUN.md` mục 0 để chạy phòng khám. Huấn luyện và đánh giá lại (chạy từ thư mục gốc):
`python project01/train/train_dl.py` (cần torch), `train_rl.py`, `build_pdf_corpus.py`, `build_kg.py`, `eval_rag.py`, `eval_trust.py`;
rồi nạp KB lên Neo4j: `python scripts/push_to_neo4j.py --only kb`; demo: `python project01/train/demo_kb_neo4j.py`.
