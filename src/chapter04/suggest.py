"""Gợi ý "nên hỏi gì" và giải thích vì sao một câu hỏi bị từ chối (dùng lại được cho Chương 5 và Chương 7).

explain_refusal(question, domain, evidence) trả về dict gồm:
  why_refused     : lời giải thích bằng tiếng Việt (hoặc tiếng Anh nếu câu hỏi tiếng Anh)
  suggestions     : danh sách câu hỏi mẫu gần với câu vừa hỏi (đã được chạy thử và trả lời được)
  switch_domain   : "diabetes" / "finance" nếu từ khóa hợp lĩnh vực kia hơn, ngược lại None
  refusal_kind    : "wrong_domain" | "out_of_scope" | "not_in_documents"
"""
import re

from .stats import strip_accents

# Câu mẫu đã chạy thật trên hệ thống (scripts/chapter05/probe_questions.py) và được trả lời từ kho.
SAMPLE_GROUPS = {
    'diabetes': [
        ('Triệu chứng và nguyên nhân', ['Triệu chứng của bệnh tiểu đường là gì?', 'Tại sao bị tiểu đường?', 'Ai có nguy cơ mắc tiểu đường?']),
        ('Các loại tiểu đường', ['Tiểu đường típ 1 khác típ 2 chỗ nào?', 'Tiểu đường thai kỳ là gì?', 'Tiền tiểu đường là gì?']),
        ('Biến chứng', ['Biến chứng của tiểu đường là gì?', 'Biến chứng ở mắt do tiểu đường là gì?', 'Bệnh tiểu đường ảnh hưởng đến thận như thế nào?']),
        ('Ăn uống và vận động', ['Người bị tiểu đường nên ăn gì?', 'Nên tập thể dục bao nhiêu phút mỗi tuần?', 'Người bị tiểu đường cần chăm sóc bàn chân như thế nào?']),
        ('Thuốc và theo dõi', ['Metformin dùng để làm gì?', 'Xét nghiệm A1C cho biết điều gì?', 'Hạ đường huyết là gì và xử lý thế nào?']),
    ],
    'finance': [
        ('Công ty trong kho báo cáo', ['Công ty Visa hoạt động trong lĩnh vực nào?', 'Những mạng thanh toán điện tử lớn nhất là gì?']),
    ],
}

DOMAIN_LABEL = {'diabetes': 'bệnh tiểu đường', 'finance': 'tài chính (báo cáo thường niên công ty)'}
DOMAIN_SCOPE = {
    'diabetes': 'Kho tài liệu của lĩnh vực này gồm 28 trang y tế của NIDDK, CDC và MedlinePlus về bệnh tiểu đường: triệu chứng, nguyên nhân, '
                'yếu tố nguy cơ, biến chứng, ăn uống, vận động, thuốc, xét nghiệm A1C và đường huyết.',
    'finance': 'Kho tài liệu của lĩnh vực này là các báo cáo thường niên của công ty trong bộ dữ liệu FinQA nên chỉ hỏi được về công ty, '
               'hoạt động kinh doanh và số liệu có trong các báo cáo đó (ví dụ Visa), không có giá thị trường hằng ngày.',
}

# Từ khóa (đã bỏ dấu, viết thường) cho từng lĩnh vực.
KEYWORDS = {
    'diabetes': ['tieu duong', 'duong huyet', 'insulin', 'metformin', 'a1c', 'huyet ap', 'bien chung', 'trieu chung', 'benh', 'thuoc', 'suc khoe',
                 'diabet', 'glucose', 'blood sugar', 'symptom', 'complication', 'medicine', 'pancrea', 'kidney', 'neuropathy', 'retinopathy',
                 'than', 'mat', 'ban chan', 'an uong', 'tap the duc', 'exercise', 'diet', 'health'],
    'finance': ['doanh thu', 'loi nhuan', 'co phieu', 'cong ty', 'bao cao', 'tai chinh', 'thu nhap', 'chi phi', 'revenue', 'income', 'shares',
                'company', 'stock', 'earnings', 'profit', 'visa', 'mastercard', 'annual report', 'dividend', 'thanh toan', 'payment', 'bank', 'ngan hang'],
}
OUT_OF_SCOPE_HINTS = [
    ('thoi tiet', 'thời tiết'), ('weather', 'thời tiết'), ('gia vang', 'giá vàng'), ('gold price', 'giá vàng'), ('bong da', 'bóng đá'), ('football', 'bóng đá'),
    ('world cup', 'bóng đá'), ('nau ', 'nấu ăn'), ('cong thuc', 'công thức nấu ăn'), ('recipe', 'công thức nấu ăn'), ('gia co phieu', 'giá cổ phiếu hôm nay'),
    ('hom nay', 'thông tin theo ngày'), ('today', 'thông tin theo ngày'), ('bitcoin', 'giá tiền mã hóa'), ('phim', 'phim ảnh'), ('nhac', 'âm nhạc'),
]


def _norm(text):
    return ' ' + re.sub(r'\s+', ' ', strip_accents(str(text).lower())) + ' '


def _score(text, domain):
    return sum(1 for k in KEYWORDS[domain] if k in text)


def _tokens(text):
    return set(re.findall(r'[a-z0-9]{3,}', _norm(text)))


def related_samples(question, domain, limit=4):
    """Câu mẫu gần câu hỏi nhất (trùng từ), nếu không có thì lấy câu đầu mỗi nhóm."""
    pool = [q for _, qs in SAMPLE_GROUPS.get(domain, []) for q in qs]
    qt = _tokens(question)
    ranked = sorted(pool, key=lambda s: -len(qt & _tokens(s)))
    if not qt or not (qt & _tokens(ranked[0])):
        ranked = [qs[0] for _, qs in SAMPLE_GROUPS.get(domain, [])] + [q for q in pool]
    seen, out = set(), []
    for q in ranked:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:limit]


def explain_refusal(question, domain, evidence=None):
    """Giải thích vì sao câu hỏi bị từ chối và gợi ý hướng hỏi lại."""
    text = _norm(question)
    other = 'finance' if domain == 'diabetes' else 'diabetes'
    here, there = _score(text, domain), _score(text, other)
    outside = next((label for key, label in OUT_OF_SCOPE_HINTS if key in text), None)
    switch = other if there > here else None
    if switch:
        kind = 'wrong_domain'
        why = (f'Câu hỏi này có vẻ thuộc lĩnh vực {DOMAIN_LABEL[other]} nhưng bạn đang chọn lĩnh vực {DOMAIN_LABEL[domain]}, '
               f'nên các đoạn tìm được không chứa câu trả lời. Hãy bấm "Đổi sang {DOMAIN_LABEL[other]}" rồi hỏi lại. ' + DOMAIN_SCOPE[domain])
    elif outside or (here == 0 and there == 0):
        kind = 'out_of_scope'
        topic = f'"{outside}"' if outside else 'chủ đề này'
        why = (f'Hệ thống chỉ trả lời từ tài liệu đã có trong kho và {topic} không có trong đó, nên nó từ chối thay vì đoán hoặc bịa. ' + DOMAIN_SCOPE[domain])
    else:
        kind = 'not_in_documents'
        why = ('Chủ đề đúng lĩnh vực, nhưng các đoạn tài liệu tìm được cho câu này không nói đến ý bạn hỏi nên hệ thống không trả lời. '
               'Thử hỏi cụ thể hơn (nêu tên bệnh, biến chứng, thuốc hoặc công ty) hoặc chọn một câu mẫu bên dưới. ' + DOMAIN_SCOPE[domain])
    target = switch or domain
    out = {'why_refused': why, 'refusal_kind': kind, 'switch_domain': switch,
           'suggestions': related_samples(question, target)}
    if switch:
        out['switch_label'] = DOMAIN_LABEL[switch]
    return out
