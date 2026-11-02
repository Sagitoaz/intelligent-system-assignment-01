"""Dựng notebook thực hành cho Project 01 (hệ đa agent cho bệnh tiểu đường) rồi chạy để lưu kết quả.

Chạy:  .venv\\Scripts\\python.exe scripts/build_project01_notebook.py
Kết quả: notebooks/project01/01_diabetes_multi_agent.ipynb (đã chạy, có output).
"""
import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "notebooks", "project01", "01_diabetes_multi_agent.ipynb")
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =====================================================================================================================
md(r"""
# Project 01: Hệ đa agent cho bệnh tiểu đường

**Môn:** Phát triển Hệ thống Thông minh · **Sinh viên:** Nguyễn Thành Trung (B23DCCN861) · **Lớp:** D23CTPM01-B · **Nhóm lớp CT, nhóm 07**

Notebook này chạy lại từng thành phần của hệ đa agent trong thư mục `project01/` và giải thích kỹ từng đoạn mã. Nó đi cùng báo cáo
`project01_CT_07_TrungNT.861` và demo "Phòng khám AI" 3D (`python project01/server.py --open`, cổng 8200).

## Bài toán và ý tưởng

Bệnh tiểu đường tiến triển chậm nên nhiều người không biết mình có nguy cơ. Một công cụ hỗ trợ tốt cần làm bốn việc cùng lúc:
**dự đoán** nguy cơ từ chỉ số sức khỏe, **giải thích** bằng kiến thức y khoa có nguồn, **gợi ý hành động** mỗi ngày và **cảnh báo** khi nguy hiểm.
Mỗi việc cần một kỹ thuật khác nhau, nên hệ chia thành các agent chuyên trách và một bộ điều phối ghép kết quả lại.

| Agent | Kỹ thuật | Tệp mã | Phần của notebook |
|---|---|---|---|
| Bác sĩ dữ liệu | Deep Learning (MLP) huấn luyện trên BRFSS 2015 | `mas/dl_agent.py`, `train/train_dl.py` | Phần 1 |
| Thủ thư tri thức | Đồ thị tri thức + RAG (BM25 + Graph-RAG) | `mas/kb.py`, `train/build_kg.py` | Phần 2 |
| Huấn luyện viên | Reinforcement Learning (Q-learning) | `mas/rl.py`, `train/train_rl.py` | Phần 3 |
| Hộ lý an toàn | Luật cố định (rule-based) | `mas/rules.py` | Phần 4 |
| Bác sĩ trưởng + Thư ký báo cáo | Điều phối theo kịch bản, tự sinh hình và bảng | `mas/orchestrator.py`, `mas/report.py` | Phần 5 |

## Cách chạy và cách đọc

1. Chọn **Kernel → Restart Kernel and Run All** (khoảng 1–2 phút; mô hình đã huấn luyện được nạp từ `project01/artifacts/`).
2. Mỗi phần có thứ tự: **ô Markdown giải thích → ô mã → kết quả**. Hãy đọc phần giải thích trước khi nhìn mã.
3. Các ô "chạy thử thu nhỏ" huấn luyện lại một bản nhỏ để thấy vòng học thật sự chạy; kết quả chính thức dùng mô hình đã huấn luyện đầy đủ.

> **Lưu ý:** đây là sản phẩm học tập. Hệ thống không chẩn đoán và không thay thế bác sĩ; mô phỏng 24 giờ là mô hình đồ chơi.
""")

md(r"""
## 0. Thiết lập môi trường

**Mục đích:** tìm thư mục gốc của dự án để `import project01...` hoạt động dù notebook được mở từ đâu, đặt kiểu biểu đồ, và liệt kê các tệp kết quả
đã huấn luyện (`artifacts/`).

**Giải thích từng dòng:**
- Vòng `while` đi ngược lên các thư mục cha cho tới khi gặp thư mục có `project01/mas`. Cách này tránh phải sửa đường dẫn khi chạy từ `notebooks/project01/`.
- `sys.path.insert(0, ROOT)` cho phép `from project01.mas ... import ...`.
- `show_source(obj)` in mã nguồn của một hàm hoặc lớp bằng `inspect.getsource`, để đọc đúng đoạn mã đang chạy thay vì chép tay.
""")
code(r"""
import os, sys, json, inspect, random, time, base64, textwrap, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = Path.cwd()
while not (ROOT / "project01" / "mas").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
ART = ROOT / "project01" / "artifacts"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})

def show_source(obj):
    print(textwrap.dedent(inspect.getsource(obj)))

print("Thư mục dự án:", ROOT)
for f in sorted(ART.iterdir()):
    print(f"{f.name:22s}{f.stat().st_size / 1024:9.1f} KB")
""")

# =====================================================================================================================
md(r"""
# Phần 1. Agent Deep Learning: dự đoán nguy cơ từ chỉ số sức khỏe

## 1.1. Dữ liệu BRFSS 2015

Dữ liệu là khảo sát sức khỏe BRFSS 2015 của CDC (Hoa Kỳ): **253.680 hồ sơ**, mỗi hồ sơ có **21 chỉ số** (huyết áp cao, mỡ máu, BMI, hút thuốc, vận động,
tự đánh giá sức khỏe, tuổi theo 13 nhóm, học vấn, thu nhập...) và một nhãn ba mức: `0` không tiểu đường, `1` tiền tiểu đường, `2` tiểu đường.

Điều cần chú ý là **mất cân bằng lớp**: nhóm tiền tiểu đường rất hiếm. Nếu huấn luyện thẳng, mô hình sẽ gần như không bao giờ đoán nhóm này. Ô dưới đây đọc
thống kê lớp từ tệp `dl_metrics.json` do `train_dl.py` ghi lại.
""")
code(r"""
M = json.load(open(ART / "dl_metrics.json", encoding="utf-8"))
cc, total = M["class_counts"], M["rows"]
df_cls = pd.DataFrame({"Nhóm": ["Không tiểu đường", "Tiền tiểu đường", "Tiểu đường"], "Số hồ sơ": [cc["none"], cc["pre"], cc["diabetes"]]})
df_cls["Tỉ lệ"] = (df_cls["Số hồ sơ"] / total * 100).round(2).astype(str) + "%"
display(df_cls)

fig, ax = plt.subplots(figsize=(6.5, 2.4))
ax.barh(df_cls["Nhóm"][::-1], df_cls["Số hồ sơ"][::-1], color=["#e5342d", "#f5a623", "#22b05a"])
for y, v in enumerate(df_cls["Số hồ sơ"][::-1]):
    ax.text(v, y, f" {v:,}".replace(",", "."), va="center")
ax.set_xlim(0, total * 1.12); ax.set_xlabel("Số hồ sơ"); ax.set_title(f"Phân bố ba nhóm trong {total:,} hồ sơ".replace(",", "."))
plt.tight_layout(); plt.show()
""")

md(r"""
## 1.2. Mô hình MLP và cách huấn luyện

Mô hình là mạng nơ-ron nhiều lớp (MLP) `21 → 128 → 64 → 32 → 3`: ba lớp ẩn dùng ReLU, có Dropout để giảm quá khớp, lớp cuối cho ba logit, qua softmax thành xác suất ba nhóm.

**Các quyết định thiết kế (đều nằm trong `train_dl.py`):**
- **Chuẩn hóa:** trừ trung bình, chia độ lệch chuẩn *tính trên tập huấn luyện*; tập kiểm tra dùng cùng thông số để không rò rỉ thông tin.
- **Hàm mất mát có trọng số lớp:** `CrossEntropyLoss(weight=...)` với trọng số `(n / (3·n_lớp))^0,75`. Số mũ 0,75 (thay vì 1) cân bằng một phần, tránh làm mô hình quá đoán lớp hiếm.
- **Chọn epoch** theo AUC trên tập *validation* riêng, không nhìn tập kiểm tra.
- **Hệ số điều chỉnh ngưỡng:** chỉ dùng để *gán nhãn* (không đổi xác suất hiển thị), chọn trên validation để cực đại macro-F1.
- **Suy luận bằng numpy:** sau huấn luyện, trọng số được xuất ra `dl_model.npz`; demo chạy chỉ cần numpy (không cần PyTorch). Hàm `numpy_forward` làm lại đúng phép tính của mạng
  và có phép kiểm tra `np.allclose` so với PyTorch.

Ô dưới in mã của hai hàm cốt lõi.
""")
code(r"""
from project01.train import train_dl
show_source(train_dl.build_model)
show_source(train_dl.numpy_forward)
""")

md(r"""
### Chạy thử thu nhỏ vòng huấn luyện

Huấn luyện đầy đủ (14 epoch trên 253.680 hồ sơ) mất vài phút, nên ô này chạy một **bản thu nhỏ**: lấy ngẫu nhiên 40.000 hồ sơ, huấn luyện 5 epoch.
Mục đích là nhìn thấy vòng lặp thật sự chạy: mỗi epoch xáo trộn dữ liệu, chia mini-batch 1.024 mẫu, tính loss, lan truyền ngược (`loss.backward()`), cập nhật trọng số (`opt.step()`),
rồi đo AUC và macro-F1 trên validation. Kết quả chính thức của hệ thống dùng mô hình đầy đủ ở ô sau.
""")
code(r"""
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

df = pd.read_csv(ROOT / "data" / "diabetes" / "diabetes_012_health_indicators_BRFSS2015.csv").sample(40000, random_state=0)
y = df["Diabetes_012"].astype(int).values
X = df[train_dl.FEATURES].astype("float32").values
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6          # thống kê chỉ tính trên tập huấn luyện
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

torch.manual_seed(0)
model = train_dl.build_model(len(train_dl.FEATURES))
w = torch.tensor((len(ytr) / (3 * np.bincount(ytr))) ** 0.75, dtype=torch.float32)   # trọng số lớp
loss_fn = torch.nn.CrossEntropyLoss(weight=w)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
Xt, Yt, Xv = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xva)

rows = []
for epoch in range(1, 6):
    model.train()
    perm, tot = torch.randperm(len(Xt)), 0.0
    for i in range(0, len(Xt), 1024):
        idx = perm[i:i + 1024]
        opt.zero_grad()
        loss = loss_fn(model(Xt[idx]), Yt[idx])
        loss.backward()
        opt.step()
        tot += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        pv = torch.softmax(model(Xv), 1).numpy()
    rows.append(dict(epoch=epoch, train_loss=round(tot / len(Xt), 4),
                     val_auc=round(roc_auc_score(yva, pv, multi_class="ovr"), 4),
                     val_macro_f1=round(f1_score(yva, pv.argmax(1), average="macro"), 4)))
display(pd.DataFrame(rows))
""")

md(r"""
### Kết quả của mô hình đầy đủ

Ô dưới vẽ đường học (loss huấn luyện, AUC và macro-F1 trên validation; đường đứt là epoch tốt nhất) và so sánh MLP với **hồi quy logistic** cân bằng lớp làm đường cơ sở.

**Đọc kết quả trung thực:** MLP chỉ hơn hồi quy logistic một chút về AUC. Độ nhạy (recall) của nhóm *tiền tiểu đường* thấp, vì dữ liệu khảo sát không có xét nghiệm máu nên
ranh giới giữa "không" và "tiền" rất mờ. Vì vậy hệ thống dùng *chỉ số nguy cơ kết hợp* và luôn khuyên xét nghiệm xác nhận, thay vì coi nhãn dự đoán là kết luận.
""")
code(r"""
h = pd.DataFrame(M["history"])
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3))
ax[0].plot(h.epoch, h.train_loss, "o-", color="#f0603a"); ax[0].set_title("Loss huấn luyện"); ax[0].set_xlabel("epoch")
ax[1].plot(h.epoch, h.val_auc_ovr, "o-", color="#2589e0", label="AUC"); ax[1].plot(h.epoch, h.val_macro_f1, "s-", color="#0d9f8f", label="macro-F1")
ax[1].axvline(M["best_epoch"], ls="--", color="gray"); ax[1].legend(); ax[1].set_title(f"Validation (epoch tốt nhất = {M['best_epoch']})"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.show()

b = M["baseline"]; rec = M["recall_per_class"]
display(pd.DataFrame([["Logistic Regression (cân bằng lớp)", b["accuracy"], b["macro_f1"], b["auc_ovr"]],
                      ["MLP 21-128-64-32-3 (có hệ số ngưỡng)", M["accuracy"], M["macro_f1"], M["auc_ovr"]]],
                     columns=["Mô hình", "Accuracy", "Macro-F1", "AUC (một-đấu-tất-cả)"]).round(3))
print("Độ nhạy theo nhóm (không / tiền / tiểu đường):", [f"{r:.0%}" for r in rec])
""")
code(r"""
cm = np.array(M["confusion_matrix"]); names = ["Không", "Tiền", "Tiểu đường"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center", color="black", fontsize=9)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(names); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(names)
ax[0].set_xlabel("Dự đoán"); ax[0].set_ylabel("Thực tế"); ax[0].set_title("Ma trận nhầm lẫn (màu: tỉ lệ theo hàng)")
for key, label, col in [("none", "Không", "#22b05a"), ("pre", "Tiền", "#f5a623"), ("diabetes", "Tiểu đường", "#e5342d")]:
    r = M["roc"][key]; ax[1].plot(r["fpr"], r["tpr"], color=col, label=f"{label} (AUC {r['auc']:.3f})")
ax[1].plot([0, 1], [0, 1], "--", color="gray"); ax[1].legend(loc="lower right"); ax[1].set_title("Đường ROC trên tập kiểm tra")
ax[1].set_xlabel("Tỉ lệ dương giả"); ax[1].set_ylabel("Tỉ lệ dương thật")
plt.tight_layout(); plt.show()
""")

md(r"""
## 1.3. Từ mô hình thành agent: `DLAgent.predict`

Một con số xác suất chưa đủ để tư vấn. `DLAgent.predict(profile)` làm bốn việc:
1. **Ánh xạ hồ sơ** thân thiện (tuổi, chiều cao, cân nặng, có hút thuốc không...) sang 21 chỉ số BRFSS (`data.normalize`, `data.to_vector`). Ví dụ BMI = cân nặng / chiều cao²; tuổi đổi thành nhóm 1–13.
2. **Chạy mạng** bằng numpy, ra xác suất ba nhóm; nhãn lấy theo xác suất nhân với hệ số ngưỡng.
3. **Chỉ số nguy cơ** `risk_index = P(tiểu đường) + 0,5·P(tiền tiểu đường)`: gộp hai nhóm có nguy cơ thành một số duy nhất để so sánh và đặt mức thấp / trung bình / cao.
4. **Giải thích:** *yếu tố tác động* (thay giá trị của từng yếu tố bằng trung bình dân số rồi xem chỉ số nguy cơ đổi bao nhiêu) và *what-if* (nếu đổi một yếu tố lối sống sang mức khỏe mạnh thì giảm bao nhiêu).

Lưu ý: hai phần giải thích mô tả những gì mô hình học từ dữ liệu quan sát, **không phải bằng chứng nhân quả**.
""")
code(r"""
from project01.mas.dl_agent import DLAgent
show_source(DLAgent.predict)
""")
code(r"""
from project01.mas.data import PERSONAS
dl = DLAgent()
rows = []
for p in PERSONAS:
    r = dl.predict(p["profile"])
    rows.append({"Vai": p["name"], "Không": f"{r['risk']['none']:.1%}", "Tiền": f"{r['risk']['pre']:.1%}", "Tiểu đường": f"{r['risk']['diabetes']:.1%}",
                 "Nhãn": r["label"], "Chỉ số nguy cơ": round(r["risk_index"], 3), "BMI": round(r["bmi"], 1)})
display(pd.DataFrame(rows))
""")
code(r"""
r = dl.predict(PERSONAS[2]["profile"])          # Bác Hùng, 67 tuổi
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3))
d = pd.DataFrame(r["drivers"])[::-1]
ax[0].barh(d["vi"], d["effect"] * 100, color=np.where(d["effect"] > 0, "#e5342d", "#22b05a"))
ax[0].set_title("Yếu tố tác động (điểm % chỉ số nguy cơ)"); ax[0].axvline(0, color="black", lw=0.8)
if r["what_if"]:
    w = pd.DataFrame(r["what_if"])[::-1]
    ax[1].barh(w["action"], w["gain"] * 100, color="#2589e0"); ax[1].set_title("What-if: giảm được bao nhiêu điểm %")
plt.tight_layout(); plt.show()
""")

# =====================================================================================================================
md(r"""
# Phần 2. Agent tri thức: đồ thị tri thức và RAG

## 2.1. Đồ thị tri thức (Knowledge Graph)

Mô hình học sâu ở Phần 1 không biết "vì sao" nguy cơ tăng. Agent tri thức cung cấp phần đó bằng một **đồ thị tri thức**: nút là khái niệm y khoa (bệnh, triệu chứng, yếu tố nguy cơ,
xét nghiệm, điều trị, lối sống, biến chứng), cạnh là quan hệ có hướng (`raises_risk`, `symptom_of`, `diagnosed_by`...). Đồ thị do nhóm biên soạn (41 nút) dựa trên bốn trang của NIDDK,
mỗi nút gắn với các **đoạn trích** liên quan nhất (`passages`) để mọi câu trả lời đều có nguồn. Đồ thị được xuất ra Cypher (`kg_neo4j.cypher`) để nạp vào Neo4j.

Ô dưới đếm nút theo loại, cạnh theo quan hệ và in vài sự kiện dạng "A — quan hệ — B".
""")
code(r"""
from project01.mas.kb import KnowledgeBase, REL_VI, BM25
kb = KnowledgeBase()
types = pd.Series([n["type"] for n in kb.nodes.values()]).value_counts().rename("Số nút").to_frame()
rels = pd.Series([REL_VI.get(e["rel"], e["rel"]) for e in kb.edges]).value_counts().rename("Số cạnh").to_frame()
display(types, rels)
print(f"Tổng: {len(kb.nodes)} nút, {len(kb.edges)} cạnh, {len(kb.bm25.docs) if hasattr(kb.bm25, 'docs') else '?'} đoạn trích")
for e in kb.edges[:6]:
    print("-", kb.nodes[e["source"]]["vi"], "—", REL_VI.get(e["rel"], e["rel"]), "—", kb.nodes[e["target"]]["vi"])
""")
code(r"""
import networkx as nx
G = nx.DiGraph()
for nid, n in kb.nodes.items():
    G.add_node(nid, vi=n["vi"], type=n["type"])
for e in kb.edges:
    G.add_edge(e["source"], e["target"])
palette = {"condition": "#e5342d", "symptom": "#f5a623", "risk_factor": "#7060d8", "test": "#2589e0", "treatment": "#0d9f8f", "lifestyle": "#22b05a", "complication": "#d6458a"}
pos = nx.spring_layout(G.to_undirected(), seed=3, k=0.9)
fig, ax = plt.subplots(figsize=(10, 6.2)); ax.axis("off")
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c4d0e6", arrows=False, width=1.2)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[palette.get(G.nodes[n]["type"], "#8896b0") for n in G], node_size=260)
nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["vi"] for n in G}, ax=ax, font_size=6.2)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in palette.items()], loc="lower left", fontsize=7, frameon=False)
plt.show()
""")

md(r"""
## 2.2. RAG: BM25 và Graph-RAG

**RAG (Retrieval-Augmented Generation)** trả lời bằng cách *tìm bằng chứng trước rồi mới nói*. Project này **chỉ trích nguyên văn** đoạn tài liệu kèm đường dẫn nguồn, không sinh văn bản tự do,
nên không có nguy cơ bịa nội dung y khoa.

- **BM25** là cách xếp hạng từ khóa kinh điển: điểm của một đoạn tăng khi chứa từ của câu hỏi (tần suất `tf`), giảm khi từ đó quá phổ biến (`idf`), có hiệu chỉnh theo độ dài đoạn (`k1`, `b`).
- **Graph-RAG** thêm đồ thị: (1) tìm các **nút khớp** câu hỏi (theo tên và từ đồng nghĩa), (2) mở rộng **một bước** sang nút láng giềng, (3) chạy BM25 như thường nhưng **cộng điểm**
  cho đoạn gắn với các nút đó (nút khớp +1,0; nút láng giềng +0,3). Bước (3) chính là lan truyền thông điệp cố định trên đồ thị.

Ô dưới in hàm `KnowledgeBase.search` để thấy đúng ba bước này.
""")
code(r"""
show_source(KnowledgeBase.search)
""")
code(r"""
questions = ["What organ makes insulin?", "What are the symptoms of high blood sugar?", "How can type 2 diabetes be prevented?"]
rows = []
for q in questions:
    plain = kb.search(q, 3, graph=False)["passages"][0]
    g = kb.search(q, 3, graph=True)
    rows.append({"Câu hỏi": q, "Nút khớp": ", ".join(kb.nodes[s]["vi"] for s in g["seeds"][:3]) or "(không)",
                 "BM25: đoạn xếp 1": plain["id"], "Graph-RAG: đoạn xếp 1": g["passages"][0]["id"]})
display(pd.DataFrame(rows))
g = kb.search(questions[1], 3)
print("Trích dẫn xếp 1:", g["passages"][0]["quote"][:260], "...")
print("Nguồn:", g["passages"][0]["url"])
print("Sự kiện từ đồ thị:", g["facts"][:3])
""")
md(r"""
**Đọc kết quả cụ thể:** trong ba câu mẫu, Graph-RAG chỉ đổi đoạn xếp 1 ở **câu thứ hai** (từ `tests-diagnosis:50` sang `managing-diabetes:66`, đoạn nói về hyperglycemia nên hợp ý hơn);
hai câu còn lại giữ nguyên thứ hạng của BM25. Cũng thấy một **hạn chế**: ở câu thứ hai, các nút khớp gồm "Huyết áp cao" vì từ khóa `high` trùng với tên nút, tức khớp nút bằng từ khóa còn thô.
Vì vậy đừng suy ra Graph-RAG luôn tốt hơn: lợi ích chỉ xuất hiện ở một phần câu hỏi và phụ thuộc chất lượng khớp nút.
""")
code(r"""
RAG = json.load(open(ART / "rag_metrics.json", encoding="utf-8"))
tab = pd.DataFrame([[k, RAG[k]["hit_at_1"], RAG[k]["hit_at_3"], RAG[k]["mrr"]] for k in ("BM25", "Graph-RAG")], columns=["Phương pháp", "hit@1", "hit@3", "MRR"])
display(tab.round(3))
fig, ax = plt.subplots(figsize=(6, 2.8)); x = np.arange(3)
for i, (k, c) in enumerate([("BM25", "#8896b0"), ("Graph-RAG", "#2589e0")]):
    ax.bar(x + i * 0.38 - 0.19, [RAG[k]["hit_at_1"], RAG[k]["hit_at_3"], RAG[k]["mrr"]], 0.38, label=k, color=c)
ax.set_xticks(x); ax.set_xticklabels(["hit@1", "hit@3", "MRR"]); ax.legend(); ax.set_title(f"Đánh giá trên {RAG['BM25']['questions']} câu hỏi có đáp án")
plt.tight_layout(); plt.show()
""")
md(r"""
**Đọc kết quả thận trọng:** Graph-RAG nhỉnh hơn BM25 thuần, nhưng bộ đánh giá chỉ có 16 câu tiếng Anh và trọng số cộng điểm được chọn thử trên chính bộ này,
nên đây chỉ là xu hướng, không phải bằng chứng chắc chắn.
""")

# =====================================================================================================================
md(r"""
# Phần 3. Agent Reinforcement Learning: huấn luyện viên lối sống

## 3.1. Bài toán và môi trường mô phỏng

Không thể thử nghiệm lên người thật, nên nhóm xây một **bệnh nhân ảo** đơn giản. Mỗi bước là **một giờ**, một ngày có 24 bước (6 giờ sáng đến 5 giờ sáng hôm sau).

| Thành phần RL | Trong project |
|---|---|
| **Trạng thái** | giờ trong ngày, đường huyết, lượng tinh bột đang tiêu hóa, mức vận động gần đây, số giờ từ bữa ăn trước, độ nhạy insulin, có dùng thuốc không (rời rạc hóa thành 3.888 trạng thái) |
| **Hành động** | 7 hành động: nghỉ, ăn nhẹ có đường, bữa nhẹ, bữa chính, bữa nhiều tinh bột, đi bộ, tập thể dục (ban đêm chỉ được nghỉ hoặc ăn nhẹ có đường) |
| **Phần thưởng (reward)** | theo vùng đường huyết: +1 khi 70–140 mg/dL, giảm dần khi cao, âm mạnh khi dưới 55 hoặc quá cao; phạt khi quá lâu không ăn; thưởng nhỏ cho vận động |
| **Mục tiêu** | giữ đường huyết trong ngưỡng 70–180 mg/dL càng lâu càng tốt (time-in-range) |

Ô dưới in bảng hành động và bảng phần thưởng theo mức đường huyết.
""")
code(r"""
from project01.mas import rl
display(pd.DataFrame(rl.ACTIONS)[["id", "label", "carbs", "act", "desc"]].rename(columns={"carbs": "tinh bột", "act": "vận động"}))
display(pd.DataFrame([{"Đường huyết (mg/dL)": g, "Vùng": rl.zone(g), "Phần thưởng": rl.zone_reward(g)} for g in (45, 65, 100, 170, 230, 320)]))
""")
md(r"""
## 3.2. Một bước mô phỏng: `rl.step`

`step(state, action, rng)` tính đường huyết của giờ kế tiếp: bữa ăn làm tăng (nhanh hơn với đồ ngọt, mạnh hơn khi kháng insulin), vận động và thuốc làm giảm,
đường huyết tự kéo dần về mức nền, thêm một chút nhiễu ngẫu nhiên. Hàm trả về trạng thái mới, phần thưởng và thông tin (vùng, mức tăng, mức giảm).
Các tham số được đặt theo trực giác, **không hiệu chỉnh từ số liệu bệnh nhân thật**. Ô sau in mã và thử ba hành động từ cùng một trạng thái.
""")
code(r"""
show_source(rl.step)
""")
code(r"""
patient = rl.patient_from_risk(0.4, 30.0, None, "Bệnh nhân mẫu")
print("Bệnh nhân:", patient)
state = rl.new_state(patient, random.Random(1))
print(f"Trạng thái đầu: {rl.clock(state['hour'])}h, đường huyết {state['glucose']} mg/dL\n")
rows = []
for action in ["normal_meal", "walk", "heavy_meal", "rest"]:
    new, reward, info = rl.step(state, action, random.Random(5))
    rows.append({"Hành động": action, "Đường huyết trước": state["glucose"], "Sau 1 giờ": new["glucose"], "Vùng": info["zone"], "Tăng": round(info["rise"], 1),
                 "Giảm": round(info["drop"], 1), "Reward": round(reward, 2)})
display(pd.DataFrame(rows))
""")
md(r"""
## 3.3. Q-learning

Huấn luyện viên học bằng **Q-learning dạng bảng**: bảng `Q[trạng thái][hành động]` ước lượng tổng phần thưởng kỳ vọng nếu làm hành động đó rồi hành xử tốt về sau.
Sau mỗi bước, giá trị được cập nhật theo công thức

$$Q(s,a) \leftarrow Q(s,a) + \alpha \,\big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big]$$

với `α` = 0,15 là tốc độ học, `γ` = 0,97 là hệ số chiết khấu. Khi huấn luyện dùng chiến lược **ε-greedy**: lúc đầu ε = 1 (thử hành động ngẫu nhiên để khám phá),
giảm dần tới 0,05 (chủ yếu khai thác điều đã học). Mỗi "episode" là một ngày của một bệnh nhân ngẫu nhiên (`random_patient`). Ô dưới in hàm `train`.
""")
code(r"""
show_source(rl.train)
""")
md(r"""
### Chạy thử thu nhỏ

Huấn luyện đầy đủ dùng 60.000 ngày. Ô này chỉ chạy **4.000 ngày** (vài giây) để thấy phần thưởng trung bình tăng dần khi ε giảm; mô hình chính thức nằm trong `rl_qtable.json`.
""")
code(r"""
t0 = time.time()
mini, curve = rl.train(episodes=4000, seed=11, log_every=500)
print(f"Xong sau {time.time() - t0:.1f} giây; đã gặp {len(mini.q)} trạng thái")
c = pd.DataFrame(curve)
fig, ax = plt.subplots(figsize=(6.5, 2.8)); ax.plot(c.episode, c.mean_return, "o-", color="#2589e0"); ax2 = ax.twinx(); ax2.plot(c.episode, c.epsilon, "--", color="#f0603a")
ax.set_xlabel("số ngày huấn luyện"); ax.set_ylabel("reward trung bình mỗi ngày"); ax2.set_ylabel("ε (xanh: reward, cam: ε)"); ax2.spines["right"].set_visible(True)
plt.tight_layout(); plt.show()
""")
md(r"""
## 3.4. So sánh ba huấn luyện viên

Ba chính sách được so sánh trên cùng bệnh nhân và cùng chuỗi nhiễu:
- **RL:** hành động có `Q` lớn nhất ở trạng thái hiện tại (`QCoach.act`).
- **Luật (`rule_coach`):** ăn đúng giờ, ăn nhẹ có đường khi dưới 75, đi bộ khi trên 160, tập mạnh khi trên 220.
- **Không có huấn luyện viên (`no_coach`):** ăn đúng giờ nhưng hay ăn vặt đồ ngọt, bữa tối nhiều tinh bột, ít vận động.

Ô sau chạy một ngày với mỗi chính sách cho bệnh nhân Anh Minh và vẽ đường đường huyết (vùng xanh là ngưỡng an toàn 70–180).
""")
code(r"""
coach = rl.QCoach.load()
minh = PERSONAS[1]["profile"]
r = dl.predict(minh)
pat = rl.patient_from_risk(r["risk_index"], r["bmi"], minh.get("fasting_glucose"), minh["name"])
print("Bệnh nhân ảo suy ra từ hồ sơ:", pat)
fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axhspan(70, 180, color="#d7f3df", alpha=0.7, label="ngưỡng 70–180")
rows = []
for pid, (name, pol) in rl.policies(coach).items():
    day = rl.run_day(pol, pat, seed=3)
    ax.plot([x["hour"] for x in day["rows"]], [x["glucose"] for x in day["rows"]], label=name)
    rows.append({"Chính sách": name, "Thời gian trong ngưỡng": f"{day['tir']:.0%}", "Giờ quá thấp": day["hypo"], "Giờ quá cao": day["hyper"], "Tổng reward": round(day["total"], 1)})
ax.set_xlabel("giờ (bước mô phỏng)"); ax.set_ylabel("đường huyết (mg/dL)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show(); display(pd.DataFrame(rows))
""")
code(r"""
RLM = json.load(open(ART / "rl_metrics.json", encoding="utf-8"))
rows = []
for gname, grp in RLM["evaluation"].items():
    for pid in ("rl", "rule", "none"):
        x = grp[pid]; rows.append({"Nhóm bệnh nhân": gname, "Chính sách": x["name"], "Trong ngưỡng": f"{x['time_in_range']:.0%}", "Đường huyết TB": x["mean_glucose"], "Giờ cao": x["hyper_hours"], "Điểm/ngày": x["mean_reward"]})
display(pd.DataFrame(rows))
""")
md(r"""
**Đọc kết quả trung thực:** huấn luyện viên RL giữ đường huyết trong ngưỡng gần như toàn bộ thời gian, luật làm tốt với bệnh nhân nhạy insulin nhưng kém dần khi kháng insulin tăng,
còn trường hợp không có huấn luyện viên rất thấp ở nhóm kháng insulin. Con số gần 100% của RL phản ánh việc môi trường mô phỏng **khá đơn giản và có thể học thuộc**,
không có nghĩa agent sẽ đạt kết quả như vậy với người thật.
""")

# =====================================================================================================================
md(r"""
# Phần 4. Agent luật an toàn

Trong y tế, hành vi an toàn cần **dự đoán được và giải thích được**, nên phần cảnh báo dùng luật cố định do con người viết (không học). Hộ lý an toàn có 10 luật (R1–R10), mỗi luật có mức
(`danger` khẩn, `warn` lưu ý, `info` nhắc) và căn cứ (ADA/NIDDK). Ví dụ R1: đường huyết ≥ 250 mg/dL; R5: đường huyết < 70 mg/dL; R7: BMI ≥ 30.
Agent chạy **độc lập với mô hình học**: nếu có luật mức khẩn thì bác sĩ trưởng luôn nâng mức nguy cơ lên *cao* dù mô hình nói gì.
""")
code(r"""
from project01.mas import rules
show_source(rules.evaluate)
""")
code(r"""
for p in (PERSONAS[0], PERSONAS[2]):
    r = dl.predict(p["profile"])
    out = rules.evaluate(dict(p["profile"]), r["risk_index"], r["risk"]["diabetes"])
    print(f"=== {p['name']}: {len(out)} luật được kích hoạt, mức cao nhất = {rules.top_level(out) if out else '(không)'}")
    if out:
        display(pd.DataFrame(out)[["id", "level", "text", "basis"]])
""")

# =====================================================================================================================
md(r"""
# Phần 5. Bác sĩ trưởng: điều phối và báo cáo tự sinh

`Clinic.consult(profile)` (trong `mas/orchestrator.py`) chạy các agent theo thứ tự cố định: bác sĩ dữ liệu → thủ thư tri thức → hộ lý an toàn → bác sĩ phản biện → huấn luyện viên → thư ký báo cáo.
Các agent **không gọi trực tiếp nhau**; bác sĩ trưởng mời lần lượt từng agent và nhận kết quả có cấu trúc, nên mỗi bước đều truy vết được.

Kết quả trả về là một **danh sách sự kiện** theo giao thức thống nhất, giao diện 3D diễn hoạt từng sự kiện:

| Loại sự kiện | Ý nghĩa |
|---|---|
| `speak` | một agent phát biểu |
| `handoff` | bác sĩ trưởng chuyển lượt cho agent khác |
| `tool` | agent chạy công cụ (mô hình, vẽ hình) |
| `cite` | thủ thư tri thức trích dẫn tài liệu |
| `alert` | hộ lý an toàn cảnh báo |
| `final` | kết luận cuối |

Mức nguy cơ do `Clinic.level` quyết định: chỉ số nguy cơ dưới 10% là *thấp*, dưới 25% là *trung bình*, từ 25% là *cao*; cảnh báo khẩn luôn nâng lên *cao*, cảnh báo lưu ý nâng *thấp* thành *trung bình*.
""")
code(r"""
from project01.mas.orchestrator import Clinic
clinic = Clinic()
show_source(Clinic.level)
""")
code(r"""
t0 = time.time()
result = clinic.consult(PERSONAS[2]["profile"])         # Bác Hùng
print(f"Một lượt hội chẩn mất {time.time() - t0:.1f} giây, {len(result['events'])} sự kiện\n")
ev = pd.DataFrame([{"i": e["i"], "agent": e["agent"], "loại": e["type"], "nội dung": e["text"][:110] + ("…" if len(e["text"]) > 110 else "")} for e in result["events"]])
display(ev)
""")
code(r"""
s = result["summary"]
print("Mức nguy cơ:", s["level"], "| chỉ số nguy cơ:", s["risk_index"], "| nhãn mô hình:", s["label"])
print("Kết luận:", s["headline"])
print("\nCờ đỏ:"); [print("  -", f["id"], f["text"]) for f in s["red_flags"]]
print("\nLời khuyên:"); [print("  -", a) for a in s["advice"]]
print("\nTrích dẫn:"); [print("  -", c["id"], c["url"]) for c in s["citations"]]
""")
md(r"""
## Báo cáo tự sinh: hình và bảng

Thư ký báo cáo (`mas/report.py`) vẽ bằng matplotlib ba hình (xác suất ba nhóm so với tỉ lệ trong dữ liệu, yếu tố tác động cùng what-if, một ngày mô phỏng của ba huấn luyện viên)
rồi mã hóa PNG sang base64 để gửi qua API; bảng gồm hồ sơ so với dân số, nguy cơ, what-if, cảnh báo, kế hoạch ngày, so sánh huấn luyện viên và nguồn trích dẫn.
Ô dưới giải mã và hiển thị chúng ngay trong notebook.
""")
code(r"""
rep = result["report"]
for f in rep["figures"]:
    print(f["title"], "-", f["caption"])
    display(Image(data=base64.b64decode(f["png_base64"]), width=640))
for t in rep["tables"][:3]:
    print("\n" + t["title"]); display(pd.DataFrame(t["rows"], columns=t["columns"]))
print("\nTổng cộng:", len(rep["figures"]), "hình và", len(rep["tables"]), "bảng")
""")

# =====================================================================================================================
md(r"""
# Phần 5b. Bác sĩ phản biện: kiểm tra chéo và tranh luận

**Vấn đề:** mô hình MLP chỉ nhìn 21 chỉ số khảo sát và **không có đường huyết trong đầu vào**. Một người trông khỏe mạnh nhưng xét nghiệm đường huyết đói 190 mg/dL vẫn có thể được mô hình báo nguy cơ thấp.
Nếu chỉ hỏi lại chính mô hình thì nó sẽ tự đồng ý với mình, nên agent thứ bảy, **Bác sĩ phản biện** (`mas/critic.py`), dùng ba nguồn bằng chứng *chạy được và độc lập với mô hình*:

1. **Luật an toàn** (`rules.py`) cùng ngưỡng ADA mà luật dựa vào (đường huyết đói ≥ 126 mg/dL).
2. **Số liệu thật của hồ sơ** (đường huyết, triệu chứng) và **phép thử độ nhạy** chạy lại chính mô hình khi cân nặng lệch ±5 kg.
3. **Đoạn trích có nguồn** từ kho tri thức 492 đoạn NIDDK/CDC/MedlinePlus (`data/chapter04/diabetes_corpus.jsonl`).

Bốn phép thử: C1 đường huyết so với mô hình, C2 triệu chứng so với mô hình, C3 độ chắc chắn (hai nhóm đứng đầu cách nhau bao nhiêu), C4 độ nhạy với sai số đo.
Khi có **mâu thuẫn thật** các agent tranh luận **tối đa 2 vòng**: vòng 1 bác sĩ phản biện nêu bằng chứng và bác sĩ dữ liệu bảo vệ; vòng 2 hộ lý an toàn nêu luật, thủ thư tri thức đọc nguyên văn tài liệu, bác sĩ phản biện kết luận.
Nguyên tắc kết luận: khi mô hình và **số đo trực tiếp** mâu thuẫn thì ưu tiên số đo và nâng mức nguy cơ (vẫn đề nghị xét nghiệm xác nhận). Trên giao diện 3D, các lượt tranh luận hiện thành bong bóng hội thoại và được đọc bằng giọng nói.
""")
code(r"""
from project01.mas import critic
show_source(critic.review)
""")
code(r"""
hoa = next(p for p in PERSONAS if p["id"] == "hoa")           # cô Hoa: đường huyết đói 190, hồ sơ lối sống lành mạnh
res_hoa = clinic.consult(hoa["profile"])
cr = res_hoa["summary"]["critique"]
print(f"Mô hình nói: {cr['mlp_level']} | sau luật: {cr['base_level']} | sau phản biện: {cr['final_level']} | kết cục: {cr['outcome']} | số vòng: {cr['rounds']}\n")
display(pd.DataFrame([{"phép thử": c["id"], "tên": c["title"], "kết quả": c["status"], "bằng chứng": c["evidence"], "nguồn": c["source"]} for c in cr["checks"]]))
for e in res_hoa["events"]:
    if e.get("round") is not None:
        print(f"[vòng {e['round']}] {e['agent']:<10} ({e['type']}): {e['text']}\n")
""")
md(r"""
**Kiểm tra rằng bằng chứng là thật:** hai hồ sơ giống hệt nhau, chỉ khác đường huyết đói (90 và 190). Vì mô hình không dùng đường huyết nên nguy cơ nó tính ra **bằng nhau**, nhưng bác sĩ phản biện cho **hai kết luận khác nhau**.
Điều đó chứng tỏ phản biện dựa vào luật và số đo chứ không phải mô hình đồng ý với chính nó.
""")
code(r"""
for g in (90, 190):
    prof = dict(hoa["profile"], fasting_glucose=g)
    r = clinic.consult(prof); c = r["summary"]["critique"]
    print(f"Đường huyết {g:>3}: chỉ số nguy cơ của mô hình = {dl.predict(prof)['risk_index']:.4f} | phản biện: {c['outcome']:<9} | mức cuối: {c['final_level']}")
""")

# =====================================================================================================================
md(r"""
# Phần 5c. Giải thích kết quả: yếu tố tác động và "đổi một chỉ số thì nguy cơ đổi bao nhiêu"

Với mỗi bệnh nhân, giao diện hiện hai thứ (đều tính bằng cách **chạy lại chính mô hình**, `mas/dl_agent.py` và `mas/whatif.py`):

- **Biểu đồ yếu tố đẩy nguy cơ lên/xuống:** thay từng yếu tố bằng giá trị trung bình của dân số rồi xem chỉ số nguy cơ đổi bao nhiêu *điểm phần trăm*. Thanh đỏ là yếu tố đang đẩy nguy cơ lên, thanh xanh là yếu tố đang kéo xuống.
- **What-if tương tác:** kéo cân nặng hoặc bật/tắt hút thuốc, vận động, trái cây, rau, rượu bia; mô hình tính lại ngay. Có thêm đường cong *nguy cơ theo BMI* để thấy nguy cơ tăng nhanh ở đâu. Nút "Nghe giải thích" đọc kết quả bằng giọng nói.

Đây là thay đổi **theo mô hình trên dữ liệu khảo sát**, không phải lời hứa về hiệu quả điều trị với từng người; đường huyết không nằm trong mô hình nên không đổi được ở đây.
""")
code(r"""
from project01.mas import whatif
minh = PERSONAS[1]["profile"]
base = dl.predict(minh)
drivers = pd.DataFrame(base["drivers"])[["vi", "effect"]]
drivers["hướng"] = ["đẩy lên" if v > 0 else "kéo xuống" for v in drivers["effect"]]
display(drivers.rename(columns={"vi": "yếu tố", "effect": "tác động (điểm phần trăm = ×100)"}))
for ch in ({"weight_kg": 78}, {"smoker": False, "phys_activity": True}, {"weight_kg": 78, "smoker": False, "phys_activity": True}):
    out = whatif.what_if(clinic, minh, ch)
    print(ch, "->", out["text"], f"(chỉ số {out['base']['risk_index']:.3f} → {out['new']['risk_index']:.3f})")
curve = whatif.bmi_curve(clinic, minh)["points"]
plt.figure(figsize=(6, 3)); plt.plot([p["bmi"] for p in curve], [p["risk_index"] for p in curve], color="#7060d8", lw=2.5)
plt.axvline(whatif.bmi_curve(clinic, minh)["current"]["bmi"], color="#e8503a", ls="--", label="BMI hiện tại"); plt.xlabel("BMI"); plt.ylabel("chỉ số nguy cơ"); plt.legend(); plt.tight_layout(); plt.show()
""")

# =====================================================================================================================
md(r"""
# Phần 5d. Hỏi bác sĩ bằng câu hỏi tự do (RAG có trích dẫn)

Tab "Hỏi bác sĩ" (`mas/ask.py`, `POST /api/ask`) trả lời câu hỏi tiếng Việt hoặc tiếng Anh dựa trên **kho 492 đoạn** NIDDK/CDC/MedlinePlus (`data/chapter04/diabetes_corpus.jsonl`).
Quy trình: (1) dịch từ khóa y khoa Việt→Anh bằng bảng thuật ngữ; (2) truy hồi lai BM25 + đồ thị tri thức + vector MiniLM; (3) soạn câu trả lời **kèm trích dẫn [1], [2]** ánh xạ tới đoạn và URL.
Có thể nhờ Qwen2.5-1.5B-Instruct viết lại, nhưng máy thiếu bộ nhớ ảo nên mặc định dùng **câu trả lời trích đoạn**; Qwen chạy chậm (40–140 giây) và bộ kiểm tra bám tài liệu thường loại câu sinh ra, khi đó hệ thống nói rõ đang dùng chế độ trích đoạn.
Hệ thống **từ chối** khi cần chẩn đoán/liều thuốc cá nhân, khi có dấu hiệu cấp cứu, hoặc khi không có đoạn đủ liên quan.
""")
code(r"""
from project01.mas.ask import AskEngine
engine = AskEngine(clinic)
engine.retriever.dense_ready.wait(600)          # chờ nạp xong mô hình vector nền (nếu chưa, chỉ dùng BM25)
print("Vector sẵn sàng:", engine.retriever.dense is not None)
for q in ["Triệu chứng của bệnh tiểu đường là gì?", "What is the A1C test?", "Tôi nên uống bao nhiêu mg metformin?"]:
    t0 = time.time(); r = engine.ask(q, None, "retrieval")
    print(f"Q: {q}\n  chế độ: {r['mode']} | {time.time() - t0:.1f}s | trích dẫn: {[c['id'] for c in r['citations']]}\n  A: {r['answer'][:260]}\n")
ev = json.load(open(ART / "ask_eval.json", encoding="utf-8"))
display(pd.DataFrame([{"tập": k, "số câu": ev[k]["questions"], "hit@1": ev[k]["hit_at_1"], "hit@3": ev[k]["hit_at_3"]} for k in ("english", "vietnamese", "combined")]))
print(ev["note"])
""")

# =====================================================================================================================
md(r"""
# Phần 5e. Bảng độ tin cậy và kiểm tra công bằng

`project01/train/eval_trust.py` chạy mô hình trên **đúng tập kiểm tra 50.736 hồ sơ** và ghi `artifacts/trust_metrics.json`: độ chính xác và recall từng nhóm, biểu đồ hiệu chuẩn xác suất, tỉ lệ cảnh báo đúng, và độ chính xác theo **nhóm tuổi, giới, thu nhập** (có khoảng tin cậy Wilson 95%).
Mục đích là **nói thẳng điểm yếu**: độ chính xác 75% thấp hơn cả baseline "luôn đoán không tiểu đường"; chỉ bắt khoảng 17% ca tiền tiểu đường; người trẻ và người thu nhập cao bị bỏ sót nhiều hơn.
""")
code(r"""
TM = json.load(open(ART / "trust_metrics.json", encoding="utf-8"))
print(f"Độ chính xác {TM['accuracy']:.3f} (khoảng tin cậy {TM['accuracy_ci']}) | baseline luôn đoán 'không': {TM['baseline_always_none']['accuracy']:.3f}")
display(pd.DataFrame(TM["per_class"])[["name", "support", "precision", "recall", "f1"]])
m = TM["alerts"]["main"]
print(f"Cảnh báo nguy cơ cao (chỉ số ≥ {m['threshold']}): {m['alerts']} người ({m['alert_rate']:.0%}); đúng {m['precision_diabetes']['value']:.1%}, bắt được {m['recall_diabetes']['value']:.1%} ca tiểu đường, báo nhầm {m['false_alarm_rate']['value']:.1%} người không bệnh")
rows = []
for key, g in TM["groups"].items():
    for r in g["rows"]:
        rows.append({"nhóm": g["title"], "giá trị": r["group"], "n": r["n"], "chính xác": r["accuracy"]["value"], "recall tiểu đường": r["recall_diabetes"]["value"], "tỉ lệ bị cảnh báo": r["alert_rate"]["value"]})
display(pd.DataFrame(rows))
print("\nKết luận công bằng:"); [print("  -", c) for c in TM["fairness"]["conclusions"]]
cal = TM["calibration"]["diabetes"]
print("\nHiệu chuẩn (lớp tiểu đường):", {k: cal[k] for k in cal if k in ("ece", "brier")})
""")

# =====================================================================================================================
md(r"""
# Phần 5f. Nhiều bệnh nhân, xu hướng 4 tuần, báo cáo PDF và nhập liệu bằng giọng nói

- **Hàng đợi** (`mas/followup.py`, `POST /api/queue`): agent phân loại sắp hồ sơ theo mức khẩn (cờ đỏ khẩn trước, rồi chỉ số nguy cơ giảm dần) và nêu lý do.
- **Xu hướng 4 tuần** (`POST /api/followup`): chuỗi 28 ngày đường huyết đói và cân nặng. **Đây là dữ liệu mẫu sinh bằng seed cố định, không phải số đo thật** (có thể gửi chuỗi thật qua `series`).
  Mỗi tuần dựng lại bệnh nhân RL từ đường huyết và BMI của tuần đó, lấy kế hoạch ngày phổ biến nhất của huấn luyện viên RL qua 15 ngày mô phỏng, rồi so kế hoạch giữa các tuần để chỉ ra giờ nào **đổi**, kèm lý do bằng số.
- **PDF** (`mas/pdf_report.py`, `POST /api/report/pdf`): ghép hình và bảng thư ký đã sinh thành báo cáo cho bệnh nhân (font DejaVu nhúng nên đủ dấu tiếng Việt, chân trang ghi "chỉ để học tập").
- **Giọng nói vào** (`mas/stt.py`): Azure STT nhận WAV 16 kHz; bộ phân tích luật `parse_speech` rút tuổi, giới, chiều cao, cân nặng, đường huyết, bệnh nền, triệu chứng (hiểu cả số nói bằng chữ) để tự điền biểu mẫu.
""")
code(r"""
from project01.mas import followup, pdf_report, stt
q = followup.triage(clinic)
display(pd.DataFrame([{k: v for k, v in x.items() if k in ("name", "level", "risk_index", "rank", "reason")} for x in q.get("patients", q.get("queue", []))]))
fu = followup.analyze(clinic, {"profile": PERSONAS[1]["profile"]})
print("Dữ liệu mô phỏng:", fu.get("simulated"), "| khóa kết quả:", list(fu.keys()))
""")
code(r"""
pdf = pdf_report.build_pdf({"summary": result["summary"], "report": result["report"], "events": result["events"], "patient_name": "Bác Hùng"})
print("PDF:", pdf[:5], f"{len(pdf) / 1024:.0f} KB")
for s in ["Tôi năm mươi lăm tuổi, nam, cao một mét sáu lăm, nặng 70 ký, đường huyết 190, bị huyết áp cao và hay khát nước",
          "Em nữ 24 tuổi cao 158 cm nặng 51 kg không hút thuốc"]:
    print(s, "\n  ->", stt.parse_speech(s))
""")

# =====================================================================================================================
md(r"""
# Phần 6. Chạy như một dịch vụ và kiểm thử

**Demo web:** `python project01/server.py --open` mở phòng khám 3D tại http://localhost:8200. Máy chủ chỉ dùng thư viện chuẩn của Python (`http.server`) và các API:
`POST /api/consult` (hội chẩn, trả về đúng cấu trúc `result` ở Phần 5), `GET /api/kg` và `POST /api/kg/search` (đồ thị và Graph-RAG), `POST /api/life/start|step|compare` (chế độ "Sống 24 giờ").
Giọng nói là tùy chọn (Azure Speech qua `.env`); nếu không có khóa thì demo vẫn chạy bằng chữ.

**Kiểm thử tự động:** ô dưới chạy các kiểm thử của Project 01: lược đồ hồ sơ khớp 21 chỉ số; xác suất cộng bằng 1; luật R1 kích hoạt với đường huyết nguy hiểm; đồ thị nhất quán và RAG trích đúng nguồn;
huấn luyện viên RL tốt hơn không có huấn luyện viên trong mô phỏng; ban đêm chỉ được nghỉ hoặc ăn nhẹ; kịch bản hội chẩn đủ bảy agent (có bác sĩ phản biện) kết thúc bằng sự kiện `final`.
""")
code(r"""
res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_project01.py", "tests/test_project01_critic.py", "tests/test_project01_trust.py", "tests/test_project01_ask.py", "tests/test_project01_stt.py", "tests/test_project01_followup.py", "tests/test_project01_pdf.py", "-q", "-k", "not hub and not deepgraph"], cwd=ROOT, capture_output=True, text=True)
print((res.stdout or res.stderr).strip().splitlines()[-1])
""")
md(r"""
# Tổng kết, hạn chế và câu hỏi trình bày

**Điều rút ra:** tách bài toán thành các agent nhỏ, mỗi agent có đầu ra có cấu trúc, và luôn để một agent luật độc lập kiểm soát an toàn. Cách này giải thích được từng bước với người dùng không chuyên.

**Hạn chế cần nói rõ khi trình bày**
- MLP chỉ đạt AUC ≈ 0,78 và gần như không phân biệt được tiền tiểu đường: dữ liệu khảo sát không có xét nghiệm máu.
- BRFSS là khảo sát tự khai ở Hoa Kỳ năm 2015, chỉ phản ánh tương quan, không đại diện cho mọi quần thể.
- Đồ thị tri thức nhỏ (41 nút), tài liệu nguồn bằng tiếng Anh, bộ đánh giá RAG chỉ có 16 câu.
- Mô phỏng 24 giờ là mô hình đồ chơi với tham số đặt theo trực giác; kết quả RL gần tối đa chỉ có ý nghĩa trong môi trường này.

**Câu hỏi trình bày**
1. Vì sao dùng trọng số lớp với số mũ 0,75 thay vì 1? Điều gì xảy ra với độ nhạy của nhóm tiền tiểu đường?
2. `risk_index = P(tiểu đường) + 0,5·P(tiền tiểu đường)`: vì sao gộp như vậy và hệ số 0,5 có ý nghĩa gì?
3. Graph-RAG khác BM25 thuần ở bước nào? Vì sao nói đó là một dạng lan truyền thông điệp cố định trên đồ thị?
4. Trong công thức Q-learning, `α`, `γ` và ε-greedy mỗi thứ điều khiển điều gì?
5. Vì sao agent luật phải độc lập với mô hình học, và nó can thiệp vào mức nguy cơ cuối cùng như thế nào?
6. Vì sao kết quả RL gần 100% không thể xem là bằng chứng về hiệu quả với người thật?
""")

# =====================================================================================================================
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(OUT)}})
client.execute()
nbf.write(nb, OUT)
print("da ghi", OUT, "voi", len(cells), "o")
