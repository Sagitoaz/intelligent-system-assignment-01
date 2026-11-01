"""Kiểm tra và sửa đầu ra của mô hình sinh nhỏ: chữ ngoài La-tinh, đơn vị đo, số liệu, câu hỏi dạng liệt kê.

Không phụ thuộc torch nên kiểm thử được bằng bộ sinh giả.
"""
import re, unicodedata

NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
LIST_QUESTION = re.compile(
    r'\b(symptoms?|signs?|complications?|risk factors?|types? of|kinds? of|causes?|what foods?|what to eat|what should (?:i|people|a person)|'
    r'what can (?:i|you|people)|how can (?:i|you|people)|ways? to|steps?|examples?|tips?|which medicines|what medicines|who is more likely|'
    r'who should|who is at risk|who gets)\b', re.I)
UNIT_VARIANTS = re.compile(r'(?i)\b(?:mili\s?gam|milli\s?gram|mg)\s*(?:/|trên|per)\s*(?:lượng|đề\s?xi\s?lít|deci\s?lit(?:er|re)?|decilit|dl|dL|lít)\b')
FOREIGN_NOTE = 'Bản dịch tự động chưa đáng tin nên hiển thị nguyên văn tiếng Anh từ mô hình: '


def foreign_script(text):
    """Trả về các ký tự chữ cái không thuộc La-tinh (Hán, Hàn, Nhật, Kirin, Thái, Ả Rập...)."""
    bad = []
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, '')
            if name and not name.startswith('LATIN') and not name.startswith('GREEK SMALL LETTER MU'):
                bad.append(ch)
        elif unicodedata.category(ch) in ('So', 'Cs', 'Co') and ord(ch) > 0x2BFF:
            bad.append(ch)
        elif 0x3000 <= ord(ch) <= 0x303F or 0xFF00 <= ord(ch) <= 0xFFEF:  # dấu câu CJK, ký tự toàn chiều rộng
            bad.append(ch)
    return bad


def strip_foreign(text):
    """Bỏ ký tự ngoài La-tinh khỏi câu rồi gọn lại khoảng trắng (lưới an toàn cuối cùng)."""
    bad = set(foreign_script(text))
    if not bad:
        return text
    cleaned = ''.join(ch for ch in text if ch not in bad)
    return re.sub(r'[ \t]{2,}', ' ', cleaned).strip()


def fix_units(text):
    """mg/dL phải giữ nguyên, không được dịch thành "miligam/lượng"."""
    text = UNIT_VARIANTS.sub('mg/dL', text)
    return re.sub(r'(?i)\bmg/dl\b', 'mg/dL', text)


def numbers(text):
    """Tập số trong câu, chuẩn hóa dấu thập phân (5,7 và 5.7 là một)."""
    return {n.replace(',', '.') for n in NUMBER.findall(text)}


def translation_problems(english, vietnamese):
    """Liệt kê lỗi của bản dịch: chữ lạ, mất số, mất đơn vị mg/dL."""
    problems = []
    if foreign_script(vietnamese):
        problems.append('foreign_script')
    if not numbers(english) <= numbers(vietnamese):
        problems.append('missing_number')
    if re.search(r'(?i)mg/dl', english) and not re.search(r'(?i)mg/dl', vietnamese):
        problems.append('missing_unit')
    return problems


def is_list_question(question):
    return bool(LIST_QUESTION.search(question or ''))
