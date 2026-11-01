"""Hiểu câu tự nhiên cho máy tính tài chính và tra công ty trong đồ thị tri thức."""
from __future__ import annotations

import re

from .stats import strip_accents

NUM = r'-?\d[\d.,]*\d|\d'


def parse_number(text: str) -> float:
    """Đọc số kiểu 1.250,5 / 1,250.5 / 1250 / 12,5."""
    t = text.strip()
    if ',' in t and '.' in t:
        if t.rfind(',') > t.rfind('.'):
            t = t.replace('.', '').replace(',', '.')
        else:
            t = t.replace(',', '')
    elif ',' in t:
        t = t.replace(',', '') if re.fullmatch(r'-?\d{1,3}(,\d{3})+', t) else t.replace(',', '.')
    elif '.' in t and re.fullmatch(r'-?\d{1,3}(\.\d{3})+', t):
        t = t.replace('.', '')
    return float(t)


def extract_numbers(question: str) -> list[float]:
    """Tách các số trong câu, bỏ các số là năm đứng sau 'năm'/'year'/'FY'."""
    text = question
    text = re.sub(r'(?i)(năm|year|fy|nam)\s*\d{4}', ' ', text)
    return [parse_number(m) for m in re.findall(NUM, text)]


OPS = [
    ('growth', ['tang truong', 'growth', 'percentage change', 'percent change', 'thay doi bao nhieu phan tram', 'tang bao nhieu', 'giam bao nhieu',
                'tang len bao nhieu', 'ti le tang', 'change in', 'increase from', 'decrease from', 'tu .* len', 'from .* to']),
    ('add', ['cong ', ' plus ', 'tong cua', 'sum of', ' add ', 'tong ', 'cong lai', ' + ']),
    ('subtract', ['tru ', ' minus ', 'subtract', 'hieu cua', 'difference between', ' - ', 'chenh lech']),
    ('multiply', ['nhan ', ' times ', 'multiply', 'tich cua', 'product of', ' x ', '*', '×']),
    ('divide', ['chia ', 'divide', ' ti so ', 'ratio of', ' / ', '÷', 'divided']),
]
OP_VI = {'growth': 'tăng trưởng', 'add': 'phép cộng', 'subtract': 'phép trừ', 'multiply': 'phép nhân', 'divide': 'phép chia'}


def parse_calculation(question: str):
    """Trả về (operation, [a, b]) hoặc None nếu câu không phải một phép tính hai số."""
    q = ' ' + strip_accents(question.lower()).rstrip('?. ') + ' '
    numbers = extract_numbers(question)
    if len(numbers) < 2:
        return None
    for op, keys in OPS:
        for k in keys:
            hit = re.search(k, q) if ('.*' in k) else (k in q)
            if hit:
                a, b = numbers[0], numbers[1]
                if op == 'subtract' and re.search(r'subtract\s+[\d.,]+\s+from', q):
                    a, b = b, a
                return op, [a, b]
    return None


def find_company(question: str, companies: set[str]):
    """Tìm mã công ty (ví dụ V, AAPL) trong câu; chỉ nhận khi là từ viết hoa đứng riêng hoặc đi sau 'công ty/company'."""
    m = re.search(r'(?i)(?:công ty|cong ty|company|ticker|mã|ma cổ phiếu|của|of)\s+["\']?([A-Za-z]{1,6})\b', question)
    if m and m.group(1).upper() in companies and (m.group(1).isupper() or len(m.group(1)) > 1 or m.group(1) == 'V'):
        # từ thường như "of the" không phải mã; chỉ nhận khi viết hoa hoặc đứng ngay sau "công ty/company"
        keyword = m.group(0).split()[0].lower()
        if m.group(1).isupper() or keyword in ('công', 'cong', 'company', 'ticker', 'mã'):
            return m.group(1).upper()
    stop = {'A', 'I', 'C', 'L'}
    for token in re.findall(r'\b[A-Z]{1,6}\b', question):
        if token in companies and (token not in stop or re.search(r'(?i)(công ty|company|ticker)\s+' + token + r'\b', question)):
            return token
    return None


def wants_graph(question: str) -> bool:
    q = strip_accents(question.lower())
    return any(k in q for k in ['bao cao', 'report', 'cong ty', 'company', 'ticker', 'ma co phieu', 'published', 'cong bo', 'nhung nam nao'])
