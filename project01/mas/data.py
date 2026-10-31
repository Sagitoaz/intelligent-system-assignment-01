"""Hồ sơ bệnh nhân: lược đồ nhập liệu thân thiện, ánh xạ sang 21 chỉ số BRFSS và các "vai" bệnh nhân mẫu."""
from __future__ import annotations

FEATURES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
            "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
            "Sex", "Age", "Education", "Income"]

AGE_EDGES = [24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]  # nhóm tuổi BRFSS 1..13
SYMPTOMS = [("thirst", "Khát nước nhiều"), ("frequent_urination", "Đi tiểu nhiều"), ("fatigue", "Mệt mỏi"),
            ("blurred_vision", "Nhìn mờ"), ("slow_healing", "Vết thương lâu lành"), ("weight_loss", "Sụt cân không rõ lý do")]

DEFAULT = dict(name="Bạn", age_years=45, sex="female", height_cm=165, weight_kg=62, high_bp=False, high_chol=False, chol_check=True,
               smoker=False, stroke=False, heart_disease=False, phys_activity=True, fruits_daily=True, veggies_daily=True,
               heavy_alcohol=False, any_healthcare=True, no_doc_cost=False, diff_walking=False, gen_health=2, mental_bad_days=2,
               physical_bad_days=2, education=5, income=6, fasting_glucose=None, symptoms=[])

PERSONAS = [
    dict(id="lan", name="Chị Lan, 34 tuổi", story="Nhân viên văn phòng, năng động, ăn uống cân đối. Đi khám để yên tâm.",
         profile=dict(DEFAULT, name="Chị Lan", age_years=34, sex="female", height_cm=162, weight_kg=54, gen_health=2, fasting_glucose=88)),
    dict(id="minh", name="Anh Minh, 52 tuổi", story="Quản lý hay ngồi họp, thừa cân, huyết áp cao, ít vận động và thích ăn đồ ngọt.",
         profile=dict(DEFAULT, name="Anh Minh", age_years=52, sex="male", height_cm=170, weight_kg=88, high_bp=True, high_chol=True,
                      smoker=True, phys_activity=False, fruits_daily=False, veggies_daily=False, gen_health=4, physical_bad_days=8,
                      fasting_glucose=118, symptoms=["fatigue", "thirst"])),
    dict(id="hung", name="Bác Hùng, 67 tuổi", story="Đã nghỉ hưu, gần đây khát nước, đi tiểu nhiều, sụt cân; đường huyết đo ở nhà khá cao.",
         profile=dict(DEFAULT, name="Bác Hùng", age_years=67, sex="male", height_cm=165, weight_kg=71, high_bp=True, high_chol=True,
                      heart_disease=True, diff_walking=True, phys_activity=False, gen_health=4, physical_bad_days=12, mental_bad_days=5,
                      fasting_glucose=262, symptoms=["thirst", "frequent_urination", "weight_loss", "blurred_vision"])),
    dict(id="an", name="Em An, 24 tuổi", story="Sinh viên hay thức khuya, ăn vặt nhiều nhưng còn trẻ, cân nặng bình thường.",
         profile=dict(DEFAULT, name="Em An", age_years=24, sex="female", height_cm=158, weight_kg=51, phys_activity=False,
                      fruits_daily=False, veggies_daily=False, gen_health=3, mental_bad_days=10, fasting_glucose=92)),
]


def fields():
    """Lược đồ biểu mẫu cho giao diện (nhãn tiếng Việt, gợi ý, giới hạn)."""
    num = lambda k, label, lo, hi, d, unit, help_, group: dict(key=k, label=label, type="number", min=lo, max=hi, default=d, unit=unit, help=help_, group=group)
    yes = lambda k, label, d, help_, group: dict(key=k, label=label, type="bool", default=d, help=help_, group=group)
    return [
        dict(key="name", label="Tên", type="text", default="Bạn", help="Tên hiển thị trong phòng khám", group="Thông tin chung"),
        num("age_years", "Tuổi", 18, 95, 45, "tuổi", "Tuổi càng cao nguy cơ càng tăng", "Thông tin chung"),
        dict(key="sex", label="Giới tính", type="select", options=[["female", "Nữ"], ["male", "Nam"]], default="female", help="", group="Thông tin chung"),
        num("height_cm", "Chiều cao", 130, 210, 165, "cm", "Dùng để tính BMI", "Thể trạng"),
        num("weight_kg", "Cân nặng", 35, 180, 62, "kg", "Dùng để tính BMI", "Thể trạng"),
        yes("high_bp", "Huyết áp cao", False, "Bác sĩ từng nói bạn bị cao huyết áp", "Bệnh nền"),
        yes("high_chol", "Mỡ máu cao", False, "Cholesterol máu cao", "Bệnh nền"),
        yes("heart_disease", "Bệnh tim / nhồi máu", False, "Từng bị bệnh mạch vành hoặc nhồi máu cơ tim", "Bệnh nền"),
        yes("stroke", "Từng đột quỵ", False, "", "Bệnh nền"),
        yes("diff_walking", "Đi lại khó khăn", False, "Khó đi bộ hoặc leo cầu thang", "Bệnh nền"),
        yes("smoker", "Hút thuốc", False, "Đã hút từ 100 điếu trở lên trong đời", "Lối sống"),
        yes("heavy_alcohol", "Uống nhiều rượu bia", False, "Nam trên 14 ly, nữ trên 7 ly mỗi tuần", "Lối sống"),
        yes("phys_activity", "Có vận động", True, "Có tập thể dục trong 30 ngày qua", "Lối sống"),
        yes("fruits_daily", "Ăn trái cây hằng ngày", True, "", "Lối sống"),
        yes("veggies_daily", "Ăn rau hằng ngày", True, "", "Lối sống"),
        dict(key="gen_health", label="Tự đánh giá sức khỏe", type="select", default=2, group="Cảm nhận sức khỏe", help="",
             options=[[1, "Rất tốt"], [2, "Tốt"], [3, "Khá"], [4, "Kém"], [5, "Rất kém"]]),
        num("mental_bad_days", "Ngày tinh thần kém", 0, 30, 2, "ngày/tháng", "Số ngày căng thẳng hoặc buồn trong 30 ngày", "Cảm nhận sức khỏe"),
        num("physical_bad_days", "Ngày thể chất kém", 0, 30, 2, "ngày/tháng", "Số ngày đau ốm hoặc chấn thương trong 30 ngày", "Cảm nhận sức khỏe"),
        yes("chol_check", "Đã đo mỡ máu 5 năm qua", True, "", "Khám chữa bệnh"),
        yes("any_healthcare", "Có bảo hiểm / chăm sóc y tế", True, "", "Khám chữa bệnh"),
        yes("no_doc_cost", "Từng ngại đi khám vì tốn kém", False, "", "Khám chữa bệnh"),
        dict(key="education", label="Học vấn", type="select", default=5, group="Khác", help="",
             options=[[1, "Chưa đi học"], [2, "Tiểu học"], [3, "THCS"], [4, "THPT"], [5, "Cao đẳng/Đại học"], [6, "Sau đại học"]]),
        dict(key="income", label="Mức thu nhập", type="select", default=6, group="Khác", help="",
             options=[[1, "Rất thấp"], [2, "Thấp"], [3, "Dưới trung bình"], [4, "Trung bình thấp"], [5, "Trung bình"], [6, "Khá"], [7, "Cao"], [8, "Rất cao"]]),
        num("fasting_glucose", "Đường huyết đói (nếu biết)", 50, 500, None, "mg/dL", "Để trống nếu chưa đo; dùng cho luật an toàn", "Xét nghiệm"),
        dict(key="symptoms", label="Triệu chứng gần đây", type="multi", options=SYMPTOMS, default=[], help="Chọn các triệu chứng bạn đang có", group="Xét nghiệm"),
    ]


def normalize(profile: dict) -> dict:
    p = dict(DEFAULT)
    for k, v in (profile or {}).items():
        if k in DEFAULT:
            p[k] = v
    p["age_years"] = int(min(95, max(18, float(p["age_years"]))))
    p["height_cm"] = float(min(210, max(130, float(p["height_cm"]))))
    p["weight_kg"] = float(min(180, max(35, float(p["weight_kg"]))))
    if p["fasting_glucose"] in ("", None):
        p["fasting_glucose"] = None
    else:
        p["fasting_glucose"] = float(min(500, max(40, float(p["fasting_glucose"]))))
    p["symptoms"] = [s for s in (p.get("symptoms") or []) if s in dict(SYMPTOMS)]
    p["sex"] = "male" if p["sex"] == "male" else "female"
    for k in ("gen_health", "education", "income"):
        p[k] = int(p[k])
    for k in ("mental_bad_days", "physical_bad_days"):
        p[k] = int(min(30, max(0, float(p[k]))))
    return p


def bmi(p: dict) -> float:
    return p["weight_kg"] / (p["height_cm"] / 100.0) ** 2


def age_group(age: float) -> int:
    return 1 + sum(age > e for e in AGE_EDGES)


def to_vector(p: dict) -> list[float]:
    """Hồ sơ thân thiện -> vector 21 chỉ số BRFSS đúng thứ tự FEATURES."""
    b = lambda k: 1.0 if p[k] else 0.0
    return [b("high_bp"), b("high_chol"), b("chol_check"), round(bmi(p), 1), b("smoker"), b("stroke"), b("heart_disease"),
            b("phys_activity"), b("fruits_daily"), b("veggies_daily"), b("heavy_alcohol"), b("any_healthcare"), b("no_doc_cost"),
            float(p["gen_health"]), float(p["mental_bad_days"]), float(p["physical_bad_days"]), b("diff_walking"),
            1.0 if p["sex"] == "male" else 0.0, float(age_group(p["age_years"])), float(p["education"]), float(p["income"])]
