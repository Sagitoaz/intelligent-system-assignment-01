"""Dựng đồ thị tri thức tiểu đường (KB) và xuất Cypher cho Neo4j.

Đồ thị do nhóm biên soạn (nhãn tiếng Việt) dựa trên kho tài liệu công khai: 28 trang web (NIDDK, CDC, MedlinePlus) + 1 file PDF của NIDDK;
các ngưỡng xét nghiệm theo chuẩn ADA (bảng trong trang NIDDK bị mất khi thu thập corpus nên được ghi là tri thức bổ sung).
Mỗi nút được gắn tối đa 3 đoạn trích gần nhất bằng BM25 (bằng chứng; quan hệ MENTIONS trong Neo4j).
Chạy: python project01/train/build_kg.py   (nên chạy project01/train/build_pdf_corpus.py trước nếu PDF thay đổi)
Nguồn: 3 nhóm link + 1 PDF (chi tiết ở docstring mas/kb.py):
    NIDDK       https://www.niddk.nih.gov/health-information/diabetes/overview/
    CDC         https://www.cdc.gov/diabetes/
    MedlinePlus https://medlineplus.gov/hyperglycemia.html
    PDF NIDDK   "4 Steps to Manage Your Diabetes for Life" (data/project01/niddk_4_steps_manage_diabetes.pdf)
Các file Cypher do script này SINH RA, lưu ở thư mục db_diabet_cypher/ (đừng sửa tay; sửa N/E rồi chạy lại):
    01_kb_concepts.cypher       nút KBNode (khái niệm y khoa) + cạnh giữa các nút
    02_sources_passages.cypher  nút KBSource (trang web / PDF) + KBPassage (từng đoạn tài liệu) + quan hệ HAS_PASSAGE
    03_evidence_links.cypher    quan hệ (KBNode)-[:MENTIONS {rank}]->(KBPassage): bằng chứng của từng khái niệm
Nạp vào Neo4j: python scripts/push_to_neo4j.py --only kb   (hoặc cypher-shell -f từng file theo thứ tự 01, 02, 03).

======================================================================================================
BẢN ĐỒ FILE
======================================================================================================
Chức năng: ĐÂY LÀ NƠI ĐỊNH NGHĨA đồ thị tri thức (KB) của Project 01. Toàn bộ nút và cạnh viết tay trong hai danh sách:
  - N (dòng "N = [") : 41 NÚT = khái niệm y khoa. Mỗi nút: id, tên EN, tên VN, loại (condition / symptom / risk_factor / test /
                        lifestyle / treatment / complication), mô tả tiếng Việt, từ khóa tiếng Việt (để khớp câu hỏi) và từ khóa
                        tiếng Anh (để truy xuất đoạn tài liệu NIDDK bằng BM25).
  - E (dòng "E = [") : 54 CẠNH có hướng (nguồn, đích, quan hệ): is_a, symptom_of, raises_risk, causes, diagnosed_by,
                        prevented_by, treated_by, monitors, helps, complication_of.
  - main()           : dựng file; với mỗi nút tìm tối đa 3 đoạn NIDDK khớp nhất (BM25, điểm > 2,0) để gắn "bằng chứng" cho nút.
Đọc : kho 514 đoạn qua mas/kb.py (load_rows/load_passages: 492 đoạn web + 22 đoạn PDF).
Ghi : artifacts/kg.json            (bản dự phòng của đồ thị khi không có Neo4j; mas/kb.py đọc khi KB_BACKEND=file hoặc mất mạng)
      ../../db_diabet_cypher/*.cypher  (KB_diabet dưới dạng Cypher: nguồn thật của đồ thị khi chạy với Neo4j)
Muốn THÊM/SỬA kiến thức: sửa N hoặc E rồi chạy lại file này, sau đó python scripts/push_to_neo4j.py --only kb.
Được dùng bởi: mas/kb.py (KnowledgeBase), mas/orchestrator.py (Thủ thư tri thức), train/eval_rag.py (đo chất lượng RAG).
Lưu ý để trả lời thầy: đây là KG "nhóm tự biên soạn" dựa trên tài liệu NIDDK/CDC/MedlinePlus, KHÔNG phải KG trích xuất tự động; bằng chứng đi kèm
là các đoạn trích thật (web hoặc PDF). Các ngưỡng xét nghiệm (A1C 6,5%, FPG 126 mg/dL, OGTT 200 mg/dL) theo chuẩn ADA.
======================================================================================================
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from project01.mas.kb import BM25, load_passages  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "project01", "artifacts")
CYPHER_DIR = os.path.join(ROOT, "db_diabet_cypher")  # thư mục chứa KB_diabet dạng Cypher (thầy: "lưu vào folder, vd db_diabet_cypher")

# (id, label_en, vi, type, mô tả tiếng Việt, từ khóa tìm kiếm tiếng Việt/đồng nghĩa, từ khóa truy xuất tiếng Anh)
N = [
    ("diabetes", "Diabetes", "Bệnh tiểu đường", "condition", "Bệnh xảy ra khi đường huyết (glucose) trong máu quá cao vì cơ thể thiếu hoặc không dùng tốt insulin.", ["tiểu đường", "đái tháo đường", "diabetes"], ["diabetes blood glucose insulin"]),
    ("type1", "Type 1 diabetes", "Tiểu đường típ 1", "condition", "Hệ miễn dịch phá hủy tế bào beta của tụy nên cơ thể gần như không tạo insulin; hay gặp ở trẻ em và người trẻ.", ["tip 1", "typ 1", "type 1", "loại 1"], ["type 1 diabetes immune system beta cells"]),
    ("type2", "Type 2 diabetes", "Tiểu đường típ 2", "condition", "Dạng phổ biến nhất (khoảng 90–95% ca): tế bào kháng insulin, tụy sản xuất không đủ; tiến triển chậm qua nhiều năm.", ["tip 2", "typ 2", "type 2", "loại 2"], ["type 2 diabetes insulin resistance"]),
    ("prediabetes", "Prediabetes", "Tiền tiểu đường", "condition", "Đường huyết cao hơn bình thường nhưng chưa đủ để chẩn đoán tiểu đường típ 2; còn cơ hội đảo ngược nếu thay đổi lối sống.", ["tien tieu duong", "prediabetes"], ["prediabetes blood glucose higher than normal"]),
    ("gestational", "Gestational diabetes", "Tiểu đường thai kỳ", "condition", "Xuất hiện khi mang thai, thường hết sau sinh nhưng làm tăng nguy cơ tiểu đường típ 2 về sau.", ["thai ky", "mang thai", "gestational"], ["gestational diabetes pregnancy"]),
    ("insulin_res", "Insulin resistance", "Kháng insulin", "condition", "Tế bào không đáp ứng tốt với insulin nên glucose khó vào tế bào và ở lại trong máu.", ["khang insulin", "insulin resistance"], ["insulin resistance cells glucose"]),
    ("thirst", "Increased thirst", "Khát nước nhiều", "symptom", "Đường huyết cao kéo nước ra khỏi mô khiến người bệnh khát liên tục.", ["khat nuoc", "khat"], ["thirsty thirst"]),
    ("urination", "Frequent urination", "Đi tiểu nhiều", "symptom", "Thận đào thải bớt glucose qua nước tiểu nên phải đi tiểu thường xuyên, kể cả ban đêm.", ["di tieu nhieu", "tieu nhieu"], ["urinate more often"]),
    ("fatigue", "Fatigue", "Mệt mỏi", "symptom", "Glucose không vào được tế bào để tạo năng lượng nên người bệnh mệt mỏi, uể oải.", ["met moi", "met", "uể oải"], ["tired fatigue feeling very tired"]),
    ("blurred", "Blurred vision", "Nhìn mờ", "symptom", "Đường huyết cao làm thay đổi hình dạng thấu kính mắt nên nhìn mờ.", ["nhin mo", "mo mat"], ["blurry vision"]),
    ("slow_heal", "Slow-healing sores", "Vết thương lâu lành", "symptom", "Đường huyết cao làm giảm lưu thông máu và khả năng chống nhiễm trùng nên vết loét, vết cắt lâu lành.", ["vet thuong", "lau lanh", "loet"], ["sores that heal slowly infections"]),
    ("weight_loss", "Unexplained weight loss", "Sụt cân không rõ lý do", "symptom", "Cơ thể đốt mỡ và cơ để lấy năng lượng khi thiếu insulin; thường gặp ở típ 1.", ["sut can", "gay sut", "giam can"], ["weight loss unexplained"]),
    ("dka", "Diabetic ketoacidosis", "Nhiễm toan ceton", "complication", "Biến chứng cấp cứu khi cơ thể thiếu insulin nặng và tích tụ ceton trong máu; cần đi cấp cứu ngay.", ["nhiem toan", "ceton", "dka"], ["diabetic ketoacidosis DKA ketones"]),
    ("overweight", "Overweight / obesity", "Thừa cân, béo phì", "risk_factor", "Mỡ thừa, nhất là mỡ bụng, làm tăng kháng insulin; là yếu tố nguy cơ chính có thể thay đổi được.", ["thua can", "beo phi", "bmi", "can nang"], ["overweight obesity weight"]),
    ("inactivity", "Physical inactivity", "Ít vận động", "risk_factor", "Ít hoạt động thể chất làm tăng nguy cơ tiểu đường típ 2; vận động giúp hạ đường huyết, huyết áp và mỡ máu.", ["it van dong", "van dong", "the duc", "ngoi nhieu"], ["physically active less than 3 times a week"]),
    ("age", "Age 35 or older", "Tuổi từ 35 trở lên", "risk_factor", "Nguy cơ tiểu đường típ 2 tăng dần theo tuổi; người lớn tuổi nên được tầm soát định kỳ.", ["tuoi", "lon tuoi", "cao tuoi"], ["age 35 or older risk increases as a person gets older"]),
    ("family", "Family history", "Tiền sử gia đình", "risk_factor", "Có cha mẹ hoặc anh chị em ruột bị tiểu đường làm tăng nguy cơ; không thay đổi được.", ["gia dinh", "di truyen", "cha me"], ["family history parent sibling"]),
    ("high_bp", "High blood pressure", "Huyết áp cao", "risk_factor", "Huyết áp cao thường đi cùng tiểu đường típ 2 và làm tăng nguy cơ biến chứng tim, thận.", ["huyet ap", "cao huyet ap"], ["high blood pressure"]),
    ("high_chol", "High cholesterol", "Mỡ máu cao", "risk_factor", "HDL thấp và triglyceride cao là yếu tố nguy cơ; mỡ máu cần được kiểm soát cùng đường huyết.", ["mo mau", "cholesterol"], ["cholesterol triglycerides HDL"]),
    ("smoking", "Smoking", "Hút thuốc", "risk_factor", "Hút thuốc làm tăng nguy cơ tiểu đường típ 2 và làm biến chứng mạch máu nặng hơn; bỏ thuốc rất có lợi.", ["hut thuoc", "thuoc la"], ["smoking quit smoking"]),
    ("hx_gdm", "History of gestational diabetes", "Từng bị tiểu đường thai kỳ", "risk_factor", "Phụ nữ từng bị tiểu đường thai kỳ hoặc sinh con trên 4 kg có nguy cơ cao hơn.", ["tieu duong thai ky", "sinh con"], ["history of gestational diabetes baby weighing 9 pounds"]),
    ("a1c", "A1C test", "Xét nghiệm A1C", "test", "Đo đường huyết trung bình 2–3 tháng. Theo chuẩn ADA: từ 6,5% trở lên là tiểu đường; 5,7–6,4% là tiền tiểu đường.", ["a1c", "hba1c", "duong huyet trung binh"], ["A1C test average blood glucose"]),
    ("fpg", "Fasting plasma glucose", "Đường huyết lúc đói (FPG)", "test", "Đo sau nhịn ăn ít nhất 8 giờ. Theo chuẩn ADA: từ 126 mg/dL trở lên là tiểu đường; 100–125 mg/dL là tiền tiểu đường.", ["luc doi", "fpg", "nhin an", "duong huyet doi"], ["fasting plasma glucose FPG fasted 8 hours"]),
    ("ogtt", "Oral glucose tolerance test", "Nghiệm pháp dung nạp glucose (OGTT)", "test", "Đo đường huyết sau uống dung dịch glucose 2 giờ. Theo chuẩn ADA: từ 200 mg/dL là tiểu đường; 140–199 mg/dL là tiền tiểu đường.", ["ogtt", "dung nap glucose", "uong duong"], ["oral glucose tolerance test OGTT"]),
    ("rpg", "Random plasma glucose", "Đường huyết ngẫu nhiên", "test", "Đo vào bất kỳ lúc nào khi có triệu chứng; từ 200 mg/dL kèm triệu chứng gợi ý tiểu đường.", ["ngau nhien", "bat ky luc nao"], ["random plasma glucose test symptoms"]),
    ("screening", "Routine screening", "Tầm soát định kỳ", "test", "Người có yếu tố nguy cơ nên xét nghiệm định kỳ; kết quả bình thường vẫn nên làm lại sau mỗi 3 năm.", ["tam soat", "xet nghiem dinh ky", "kham"], ["routine testing risk factors retested every 3 years"]),
    ("weight_mgmt", "Healthy weight", "Giữ cân nặng hợp lý", "lifestyle", "Giảm một phần cân nặng giúp giảm kháng insulin; hỏi nhóm chăm sóc sức khỏe về kế hoạch phù hợp.", ["giam can", "can nang hop ly", "giu can"], ["manage your weight lose weight"]),
    ("activity", "Physical activity", "Vận động đều đặn", "lifestyle", "Nên ít nhất 150 phút mỗi tuần hoạt động cường độ vừa như đi bộ nhanh; có thể thêm bài tập sức mạnh.", ["van dong", "di bo", "tap the duc", "150 phut"], ["physical activity 150 minutes brisk walking"]),
    ("healthy_eating", "Healthy eating", "Ăn uống lành mạnh", "lifestyle", "Ưu tiên rau, trái cây, ngũ cốc nguyên hạt, đạm nạc; hạn chế đồ uống có đường và thức ăn nhiều đường tinh luyện.", ["an uong", "che do an", "thuc don", "bua an"], ["healthy eating meal plan vegetables"]),
    ("quit_smoking", "Quitting smoking", "Bỏ thuốc lá", "lifestyle", "Bỏ thuốc giúp giảm nguy cơ biến chứng tim mạch và cải thiện kiểm soát đường huyết.", ["bo thuoc"], ["quit smoking"]),
    ("glucose_monitor", "Blood glucose monitoring", "Tự theo dõi đường huyết", "treatment", "Đo đường huyết tại nhà (hoặc máy theo dõi liên tục CGM) giúp tránh đường huyết quá cao hoặc quá thấp.", ["theo doi duong huyet", "do duong huyet", "cgm"], ["check blood glucose level at home CGM time in range"]),
    ("time_in_range", "Time in range", "Thời gian trong ngưỡng an toàn", "treatment", "Phần trăm thời gian đường huyết nằm trong khoảng mục tiêu; là chỉ số dùng trong demo huấn luyện viên RL.", ["trong nguong", "time in range"], ["time in range percentage of time glucose"]),
    ("medicines", "Diabetes medicines", "Thuốc điều trị tiểu đường", "treatment", "Gồm thuốc uống và thuốc tiêm do bác sĩ chỉ định; không tự ý dùng hoặc ngừng.", ["thuoc", "dieu tri"], ["diabetes medicines insulin"]),
    ("insulin_tx", "Insulin therapy", "Điều trị bằng insulin", "treatment", "Bắt buộc với típ 1 và một số người típ 2; liều do bác sĩ quyết định.", ["tiem insulin", "insulin"], ["insulin take insulin"]),
    ("hypo", "Hypoglycemia", "Hạ đường huyết", "complication", "Đường huyết dưới 70 mg/dL ở người bệnh tiểu đường: run, đói, vã mồ hôi; cần ăn nhanh đường hấp thu nhanh và báo bác sĩ.", ["ha duong huyet", "duong huyet thap"], ["hypoglycemia blood glucose too low below 70 mg/dL"]),
    ("hyper", "Hyperglycemia", "Tăng đường huyết", "complication", "Đường huyết cao kéo dài gây khát, tiểu nhiều và tổn thương mạch máu, thần kinh về lâu dài.", ["tang duong huyet", "duong huyet cao"], ["high blood glucose hyperglycemia"]),
    ("heart", "Heart disease", "Bệnh tim mạch", "complication", "Tiểu đường làm tăng nguy cơ nhồi máu cơ tim và đột quỵ.", ["tim", "tim mach", "nhoi mau", "dot quy"], ["heart disease stroke"]),
    ("kidney", "Kidney disease", "Bệnh thận", "complication", "Đường huyết cao lâu ngày làm tổn thương các mạch máu nhỏ ở thận.", ["than", "suy than"], ["kidney disease damage"]),
    ("eye", "Eye damage", "Tổn thương mắt", "complication", "Có thể dẫn đến bệnh võng mạc đái tháo đường và mù lòa nếu không kiểm soát và khám mắt định kỳ.", ["mat", "vong mac"], ["eye damage retinopathy"]),
    ("nerve", "Nerve damage", "Tổn thương thần kinh", "complication", "Gây tê, ngứa ran hoặc đau ở bàn chân, bàn tay; làm tăng nguy cơ loét chân.", ["than kinh", "te chan", "te tay"], ["nerve damage neuropathy feet"]),
    ("foot", "Foot problems", "Biến chứng bàn chân", "complication", "Giảm cảm giác và lưu thông máu dễ gây loét, nhiễm trùng; cần kiểm tra chân hằng ngày.", ["ban chan", "loet chan"], ["foot problems feet sores"]),
]

E = [  # (nguồn, đích, quan hệ)
    ("type1", "diabetes", "is_a"), ("type2", "diabetes", "is_a"), ("gestational", "diabetes", "is_a"), ("prediabetes", "type2", "raises_risk"),
    ("insulin_res", "type2", "causes"), ("overweight", "insulin_res", "causes"),
    ("thirst", "diabetes", "symptom_of"), ("urination", "diabetes", "symptom_of"), ("fatigue", "diabetes", "symptom_of"), ("blurred", "diabetes", "symptom_of"),
    ("slow_heal", "diabetes", "symptom_of"), ("weight_loss", "type1", "symptom_of"), ("dka", "type1", "complication_of"),
    ("overweight", "type2", "raises_risk"), ("inactivity", "type2", "raises_risk"), ("age", "type2", "raises_risk"), ("family", "type2", "raises_risk"),
    ("high_bp", "type2", "raises_risk"), ("high_chol", "type2", "raises_risk"), ("smoking", "type2", "raises_risk"), ("hx_gdm", "type2", "raises_risk"),
    ("diabetes", "a1c", "diagnosed_by"), ("diabetes", "fpg", "diagnosed_by"), ("diabetes", "ogtt", "diagnosed_by"), ("diabetes", "rpg", "diagnosed_by"),
    ("prediabetes", "a1c", "diagnosed_by"), ("prediabetes", "fpg", "diagnosed_by"), ("gestational", "ogtt", "diagnosed_by"), ("diabetes", "screening", "diagnosed_by"),
    ("type2", "weight_mgmt", "prevented_by"), ("type2", "activity", "prevented_by"), ("prediabetes", "healthy_eating", "prevented_by"), ("prediabetes", "activity", "prevented_by"),
    ("overweight", "weight_mgmt", "treated_by"), ("inactivity", "activity", "treated_by"), ("smoking", "quit_smoking", "treated_by"),
    ("diabetes", "healthy_eating", "treated_by"), ("diabetes", "activity", "treated_by"), ("diabetes", "medicines", "treated_by"), ("type1", "insulin_tx", "treated_by"),
    ("diabetes", "glucose_monitor", "monitors"), ("glucose_monitor", "time_in_range", "monitors"), ("activity", "high_bp", "helps"), ("activity", "high_chol", "helps"),
    ("insulin_tx", "hypo", "causes"), ("medicines", "hypo", "causes"), ("diabetes", "hyper", "causes"),
    ("diabetes", "heart", "causes"), ("diabetes", "kidney", "causes"), ("diabetes", "eye", "causes"), ("diabetes", "nerve", "causes"), ("nerve", "foot", "causes"),
    ("heart", "high_bp", "raises_risk"), ("kidney", "high_bp", "raises_risk"),
]


def main():
    """Dựng kg.json (bản dự phòng) và 3 file Cypher trong db_diabet_cypher/ từ N, E và kho tài liệu (không có đầu vào; in thống kê khi xong)."""
    passages = load_passages()
    bm25 = BM25(passages)
    nodes = []
    for nid, label, vi, typ, desc, aliases, en in N:
        # gắn bằng chứng: tìm 3 đoạn NIDDK khớp nhất với từ khóa tiếng Anh của nút; chỉ giữ đoạn có điểm BM25 > 2,0 (lọc đoạn khớp yếu)
        found = bm25.search(" ".join(en), 3)
        nodes.append(dict(id=nid, label=label, vi=vi, type=typ, desc=desc, aliases=aliases, aliases_en=en, passages=[f["id"] for f in found if f["score"] > 2.0]))
    ids = {n["id"] for n in nodes}
    edges = [dict(source=s, target=t, rel=r) for s, t, r in E]
    assert all(e["source"] in ids and e["target"] in ids for e in edges)  # kiểm tra: không có cạnh nào trỏ tới nút không tồn tại
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "kg.json"), "w", encoding="utf-8") as f:
        json.dump(dict(nodes=nodes, edges=edges), f, ensure_ascii=False, indent=1)
    write_cypher(nodes, edges, passages)
    linked = sum(1 for n in nodes if n["passages"])
    print(f"{len(nodes)} nút, {len(edges)} cạnh, {linked} nút có đoạn trích; {len(passages)} đoạn tài liệu -> {os.path.relpath(CYPHER_DIR, ROOT)}/")


def q(s: str) -> str:
    """Thoát dấu \\ và ' rồi đổi xuống dòng thành khoảng trắng để chuỗi tiếng Việt/tiếng Anh nằm gọn trên MỘT dòng Cypher.
    Nói đơn giản: trong Cypher chuỗi nằm giữa hai dấu ' nên nếu văn bản có dấu ' (vd "don't") phải viết thành \\' kẻo câu lệnh bị cắt ngang."""
    return " ".join(str(s).split()).replace("\\", "\\\\").replace("'", "\\'")


def lst(items) -> str:
    """Danh sách chuỗi dạng Cypher: ['a', 'b']."""
    return "[" + ", ".join(f"'{q(x)}'" for x in items) + "]"


def write_cypher(nodes: list[dict], edges: list[dict], passages: list[dict]) -> None:
    """Sinh 3 file Cypher vào db_diabet_cypher/. Quy ước: MỖI CÂU LỆNH MỘT DÒNG, kết thúc bằng dấu ';' (push_to_neo4j.py đọc theo dòng);
    dòng bắt đầu bằng // là chú thích. Đoạn tài liệu nạp theo lô UNWIND (25 đoạn/lệnh) để đẩy lên Aura nhanh.

    NÓI ĐƠN GIẢN: hàm này "dịch" dữ liệu Python (danh sách nút N, cạnh E, đoạn tài liệu) thành câu lệnh Cypher để Neo4j hiểu.
      - File 01: CREATE (:KBNode {...}) = "tạo một nút"; MATCH (a),(b) CREATE (a)-[:QUAN_HE]->(b) = "vẽ mũi tên từ a sang b".
      - File 02: tạo nút KBSource (trang web/PDF) và KBPassage (từng đoạn văn), nối bằng HAS_PASSAGE = "trang này chứa đoạn này".
      - File 03: nối KBNode -> KBPassage bằng MENTIONS = "đoạn này là bằng chứng cho khái niệm này" (rank 1 = khớp nhất).
    Mỗi file mở đầu bằng DETACH DELETE (xóa dữ liệu cũ của chính nó) nên chạy lại bao nhiêu lần cũng không bị trùng."""
    os.makedirs(CYPHER_DIR, exist_ok=True)

    def save(name: str, lines: list[str]) -> None:
        with open(os.path.join(CYPHER_DIR, name), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")

    # --- 01: khái niệm y khoa. Mỗi nút -> CREATE (:KBNode {...}); mỗi cạnh -> MATCH hai nút rồi CREATE quan hệ (tên quan hệ viết HOA)
    l1 = ["// KB_diabet - FILE 1/3: khái niệm y khoa (KBNode) và quan hệ giữa chúng. Do project01/train/build_kg.py sinh ra, đừng sửa tay.",
          f"// {len(nodes)} nút, {len(edges)} cạnh. Chạy theo thứ tự 01 -> 02 -> 03:  cypher-shell -f 01_kb_concepts.cypher  (hoặc python scripts/push_to_neo4j.py --only kb)",
          "// Nói đơn giản: CREATE (:KBNode {...}) = tạo một nút khái niệm; MATCH (a),(b) CREATE (a)-[:TÊN_QUAN_HỆ]->(b) = vẽ mũi tên giữa hai nút.",
          "MATCH (n:KBNode) DETACH DELETE n;",
          "CREATE CONSTRAINT kb_id IF NOT EXISTS FOR (n:KBNode) REQUIRE n.id IS UNIQUE;"]
    for n in nodes:
        l1.append(f"CREATE (:KBNode {{id:'{n['id']}', label:'{q(n['label'])}', vi:'{q(n['vi'])}', type:'{n['type']}', desc:'{q(n['desc'])}', "
                  f"aliases:{lst(n['aliases'])}, aliases_en:{lst(n['aliases_en'])}}});")
    for e in edges:
        l1.append(f"MATCH (a:KBNode {{id:'{e['source']}'}}), (b:KBNode {{id:'{e['target']}'}}) CREATE (a)-[:{e['rel'].upper()}]->(b);")
    save("01_kb_concepts.cypher", l1)

    # --- 02: nguồn (trang web / PDF) và từng đoạn tài liệu
    sources: dict[str, dict] = {}
    for p in passages:
        s = sources.setdefault(p["url"], dict(url=p["url"], title=p["title"], org=p["org"], type=p["type"], n=0))
        s["n"] += 1
    l2 = ["// KB_diabet - FILE 2/3: nguồn tài liệu (KBSource: trang web hoặc PDF) và từng đoạn trích (KBPassage). Do build_kg.py sinh ra, đừng sửa tay.",
          f"// {len(sources)} nguồn ({sum(s['type'] == 'pdf' for s in sources.values())} PDF), {len(passages)} đoạn. Văn bản đoạn là nguyên văn tiếng Anh từ tài liệu gốc.",
          "// Nói đơn giản: KBSource = một trang web hoặc file PDF; KBPassage = một đoạn văn trích từ đó; HAS_PASSAGE = 'trang này chứa đoạn này'.",
          "MATCH (n:KBPassage) DETACH DELETE n;", "MATCH (n:KBSource) DETACH DELETE n;",
          "CREATE CONSTRAINT kb_passage_id IF NOT EXISTS FOR (n:KBPassage) REQUIRE n.id IS UNIQUE;",
          "CREATE CONSTRAINT kb_source_url IF NOT EXISTS FOR (n:KBSource) REQUIRE n.url IS UNIQUE;"]
    for s in sources.values():
        l2.append(f"CREATE (:KBSource {{url:'{q(s['url'])}', title:'{q(s['title'])}', org:'{q(s['org'])}', source_type:'{s['type']}', passages:{s['n']}}});")
    for i in range(0, len(passages), 25):
        rows = ", ".join(f"{{id:'{q(p['id'])}', text:'{q(p['text'])}', title:'{q(p['title'])}', section:'{q(p['section'])}', url:'{q(p['url'])}', "
                         f"org:'{q(p['org'])}', source_type:'{p['type']}'}}" for p in passages[i:i + 25])
        l2.append(f"UNWIND [{rows}] AS r CREATE (p:KBPassage) SET p = r;")
    l2.append("MATCH (p:KBPassage), (s:KBSource {url:p.url}) CREATE (s)-[:HAS_PASSAGE]->(p);")
    save("02_sources_passages.cypher", l2)

    # --- 03: bằng chứng của từng khái niệm (rank 1 = đoạn khớp nhất theo BM25)
    triples = [(n["id"], pid, r) for n in nodes for r, pid in enumerate(n["passages"], 1)]
    l3 = ["// KB_diabet - FILE 3/3: bằng chứng. (KBNode)-[:MENTIONS {rank}]->(KBPassage); rank 1 = đoạn khớp nhất. Do build_kg.py sinh ra, đừng sửa tay.",
          f"// {len(triples)} liên kết. Đây là chỗ RAG đi từ khái niệm trong đồ thị tới đoạn tài liệu (mas/neo4j_kb.py: evidence()).",
          "// Nói đơn giản: MENTIONS = 'đoạn văn này là bằng chứng cho khái niệm này'; rank 1 = đoạn khớp nhất. Mỗi dòng UNWIND nạp một lô liên kết cùng lúc cho nhanh."]
    for i in range(0, len(triples), 50):
        rows = ", ".join(f"['{a}', '{q(b)}', {r}]" for a, b, r in triples[i:i + 50])
        l3.append(f"UNWIND [{rows}] AS t MATCH (n:KBNode {{id:t[0]}}), (p:KBPassage {{id:t[1]}}) CREATE (n)-[:MENTIONS {{rank:t[2]}}]->(p);")
    save("03_evidence_links.cypher", l3)


if __name__ == "__main__":
    main()
