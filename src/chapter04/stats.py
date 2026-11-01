"""Công cụ thống kê tự nhiên trên dữ liệu CDC BRFSS 2023 (mẫu 150.000 người).

Nhận câu hỏi tiếng Việt hoặc tiếng Anh, nhận ra ý định bằng luật và từ khóa, tính bằng pandas
và trả lời bằng một câu văn kèm bảng nhỏ. Không dùng mô hình sinh nên con số luôn chính xác
theo đúng dữ liệu.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

VI_CHARS = re.compile('[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]')
VI_WORDS = re.compile(r'\b(bao nhieu|ti le|ty le|nguoi|benh|tieu duong|cho toi|la gi|co bao|nhom tuoi)\b')


def detect_language(text: str) -> str:
    low = text.lower()
    return 'vi' if VI_CHARS.search(low) or VI_WORDS.search(strip_accents(low)) else 'en'


def strip_accents(text: str) -> str:
    import unicodedata
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def fmt_int(n, lang='vi') -> str:
    s = f'{int(n):,}'
    return s.replace(',', '.') if lang == 'vi' else s


def fmt_pct(x: float, lang='en') -> str:
    s = f'{x:.1f}%'
    return s.replace('.', ',') if lang == 'vi' else s


def fmt_num(x: float, lang='vi', digits=1) -> str:
    s = f'{x:.{digits}f}'
    return s.replace('.', ',') if lang == 'vi' else s


# ----------------------------------------------------------------------------------------------
# Từ điển khái niệm: mỗi mục = (khóa, nhãn Việt, nhãn Anh, các từ khóa đã bỏ dấu, cột, điều kiện)
# ----------------------------------------------------------------------------------------------
DIMENSIONS = {
    'sex': {'col': 'sex', 'vi': 'giới tính', 'en': 'sex', 'kw': ['gioi tinh', 'gioi', 'by sex', 'by gender', 'gender', 'sex'],
            'values': {'male': (['nam gioi', 'dan ong', 'dan ong', 'nam', 'men', 'male', 'man '], 'nam giới', 'men'),
                       'female': (['nu gioi', 'phu nu', 'nu ', 'women', 'female', 'woman'], 'nữ giới', 'women')}},
    'age_group': {'col': 'age_group', 'vi': 'nhóm tuổi', 'en': 'age group', 'kw': ['nhom tuoi', 'do tuoi', 'age group', 'by age', 'age'],
                  'values': {}},
    'bmi_category': {'col': 'bmi_category', 'vi': 'nhóm BMI', 'en': 'BMI category', 'kw': ['nhom bmi', 'muc bmi', 'bmi category', 'by bmi', 'can nang'],
                     'values': {'obese': (['beo phi', 'obese', 'obesity'], 'béo phì (BMI từ 30)', 'people with obesity (BMI 30+)'),
                                'overweight': (['thua can', 'overweight'], 'thừa cân (BMI 25–29,9)', 'overweight people (BMI 25-29.9)'),
                                'normal': (['binh thuong', 'can nang binh thuong', 'normal weight', 'normal'], 'có cân nặng bình thường (BMI 18,5–24,9)', 'normal-weight people (BMI 18.5-24.9)'),
                                'underweight': (['thieu can', 'underweight'], 'thiếu cân (BMI dưới 18,5)', 'underweight people (BMI under 18.5)')}},
    'smoker_status': {'col': 'smoker_status', 'vi': 'tình trạng hút thuốc', 'en': 'smoking status', 'kw': ['hut thuoc', 'smoking', 'smoker', 'by smoking'],
                      'values': {'daily': (['hut thuoc hang ngay', 'daily smoker', 'smoke daily'], 'hút thuốc hằng ngày', 'daily smokers'),
                                 'some_days': (['thinh thoang hut'], 'thỉnh thoảng hút thuốc', 'occasional smokers'),
                                 'former': (['da bo thuoc', 'tung hut', 'former smoker', 'ex-smoker', 'quit smoking', 'bo thuoc'], 'từng hút thuốc', 'former smokers'),
                                 'never': (['khong hut thuoc', 'chua bao gio hut', 'never smoked', 'non-smoker', 'nonsmoker', 'never smoker'], 'chưa bao giờ hút thuốc', 'never smokers')}},
    'income_group': {'col': 'income_group', 'vi': 'nhóm thu nhập', 'en': 'income group', 'kw': ['thu nhap', 'income'], 'values': {}},
    'education': {'col': 'education', 'vi': 'trình độ học vấn', 'en': 'education level', 'kw': ['hoc van', 'education'],
                  'values': {}},
    'state': {'col': 'state', 'vi': 'bang', 'en': 'state', 'kw': ['theo bang', 'by state', 'each state', 'tung bang'], 'values': {}},
    'general_health': {'col': 'general_health', 'vi': 'mức tự đánh giá sức khỏe', 'en': 'self-rated health', 'kw': ['suc khoe tong quat', 'tu danh gia suc khoe', 'general health', 'self-rated'], 'values': {}},
}
HEALTH_LABEL = {1: 'rất tốt', 2: 'tốt', 3: 'khá', 4: 'kém', 5: 'rất kém'}
HEALTH_LABEL_EN = {1: 'excellent', 2: 'very good', 3: 'good', 4: 'fair', 5: 'poor'}
EDU_VI = {'less_than_high_school': 'chưa tốt nghiệp trung học', 'high_school': 'tốt nghiệp trung học', 'some_college': 'từng học cao đẳng/đại học',
          'college_graduate': 'tốt nghiệp đại học'}
SMOKER_VI = {'daily': 'hút hằng ngày', 'some_days': 'thỉnh thoảng hút', 'former': 'đã bỏ thuốc', 'never': 'chưa bao giờ hút'}
BMI_VI = {'obese': 'béo phì', 'overweight': 'thừa cân', 'normal': 'bình thường', 'underweight': 'thiếu cân'}

# Điều kiện nhị phân: khóa -> (từ khóa đã bỏ dấu, mô tả Việt, mô tả Anh, hàm tạo mặt nạ)
CONDITIONS = {
    'diabetes': (['tieu duong', 'diabetes', 'diabetic'], 'được chẩn đoán tiểu đường', 'diagnosed with diabetes',
                 lambda d: d.diabetes_status == 'yes'),
    'prediabetes': (['tien tieu duong', 'prediabetes', 'pre-diabetes'], 'ở giai đoạn tiền tiểu đường', 'with prediabetes',
                    lambda d: d.diabetes_status == 'prediabetes'),
    'hbp': (['huyet ap cao', 'cao huyet ap', 'tang huyet ap', 'high blood pressure', 'hypertension'], 'bị huyết áp cao', 'with high blood pressure',
            lambda d: d.high_blood_pressure == 1),
    'chol': (['cholesterol cao', 'mo mau cao', 'high cholesterol'], 'bị cholesterol cao', 'with high cholesterol',
             lambda d: d.high_cholesterol == 1),
    'smoker': (['hut thuoc', 'smoker', 'smoking', 'smoke'], 'đang hút thuốc', 'who currently smoke',
               lambda d: d.smoker_status.isin(['daily', 'some_days'])),
    'inactive': (['khong van dong', 'it van dong', 'khong tap the duc', 'khong tap', 'physically inactive', 'inactive', 'sedentary', 'no exercise'],
                 'không vận động thể chất', 'who are physically inactive', lambda d: d.physical_activity == 0),
    'active': (['van dong', 'tap the duc', 'physically active', 'exercise', 'active'], 'có vận động thể chất', 'who are physically active',
               lambda d: d.physical_activity == 1),
    'heart': (['benh tim', 'tim mach', 'nhoi mau', 'heart disease', 'heart attack', 'coronary'], 'có bệnh tim (nhồi máu hoặc bệnh mạch vành)',
              'with heart disease', lambda d: d.heart_disease == 1),
    'stroke': (['dot quy', 'stroke'], 'từng bị đột quỵ', 'who had a stroke', lambda d: d.stroke == 1),
    'poorhealth': (['suc khoe kem', 'suc khoe yeu', 'poor health', 'bad health', 'fair or poor'], 'tự đánh giá sức khỏe kém hoặc rất kém',
                   'rating their health fair or poor', lambda d: d.general_health >= 4),
}
CONDITION_ORDER = ['prediabetes', 'diabetes', 'inactive', 'hbp', 'chol', 'heart', 'stroke', 'poorhealth', 'smoker', 'active']

AGE_BANDS = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
BAND_LOW = {b: int(b.split('-')[0].rstrip('+')) for b in AGE_BANDS}

RATE_KW = ['ti le', 'ty le', 'phan tram', '%', 'percent', 'rate', 'proportion', 'share', 'prevalence', 'ti trong', 'chiem bao nhieu', 'pho bien']
COUNT_KW = ['bao nhieu', 'how many', 'so luong', 'tong so', 'number of', 'count', 'dem ', 'co may']
MEAN_KW = ['trung binh', 'average', 'mean']
COMPARE_KW = ['so sanh', 'compare', 'comparison', 'khac nhau', 'versus', ' vs ', ' hon ', ' than ', 'differ', 'so voi']
GROUP_KW = ['theo ', 'by ', 'each', 'tung ', 'giua cac', 'per ', 'across']
DATA_WORDS = ['nguoi', 'people', 'person', 'respondent', 'adult', 'record', 'row', 'dataset', 'sample', 'survey', 'ban ghi', 'dong',
              'mau', 'khao sat', 'du lieu', 'brfss', 'participant', 'patient', 'benh nhan', 'population', 'cases', 'case']
ADVICE_KW = ['should', 'nen ', 'how do', 'how can', 'how to', 'lam sao', 'lam the nao', 'cach ', 'can i ', 'toi co nen', 'what should', 'tips', 'recommend']

US_STATES = ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'georgia',
             'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts',
             'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey',
             'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island',
             'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia',
             'wisconsin', 'wyoming', 'district of columbia', 'puerto rico', 'guam']

SAMPLE_QUESTIONS_VI = [
    'Có bao nhiêu người trong dữ liệu khảo sát?',
    'Tỉ lệ người mắc tiểu đường là bao nhiêu?',
    'Tỉ lệ tiểu đường theo nhóm tuổi',
    'So sánh tỉ lệ tiểu đường giữa nam và nữ',
    'BMI trung bình của người bị tiểu đường là bao nhiêu?',
    'Tỉ lệ tiểu đường ở người huyết áp cao',
]


def contains(text: str, words) -> bool:
    return any(w in text for w in words)


class DiabetesStats:
    """Bộ nhận ý định bằng luật + từ khóa cho dữ liệu BRFSS 2023."""

    source = 'CDC BRFSS 2023'
    url = 'https://www.cdc.gov/brfss/annual_data/annual_2023.html'

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.n = len(self.df)

    # ---- nhận diện -------------------------------------------------------------------------
    @staticmethod
    def metric_of(q: str):
        if contains(q, MEAN_KW) and 'bmi' in q:
            return 'mean_bmi'
        if contains(q, RATE_KW):
            return 'rate'
        if contains(q, COUNT_KW):
            return 'count'
        return None

    @staticmethod
    def looks_statistical(question: str) -> bool:
        q = ' ' + strip_accents(question.lower()).rstrip('?. ') + ' '
        if contains(q, ADVICE_KW):
            return False
        metric = DiabetesStats.metric_of(q)
        if metric is None and not (contains(q, COMPARE_KW) and contains(q, ['tieu duong', 'diabetes'])):
            return False
        has_entity = contains(q, DATA_WORDS) or any(contains(q, v[0]) for v in CONDITIONS.values())
        return bool(has_entity)

    def parse_age(self, q: str):
        """Trả về (danh sách nhóm tuổi, mô tả) hoặc None."""
        m = re.search(r'(?:tu |from |aged? |between |giua )?(\d{2})\s*(?:-|–|den|to|and|va)\s*(\d{2})', q)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            bands = [b for b in AGE_BANDS if BAND_LOW[b] >= lo and BAND_LOW[b] <= hi - 4]
            if bands:
                return bands, f'tuổi {bands[0].split("-")[0]}–{bands[-1].split("-")[-1] if bands[-1] != "80+" else "80+"}', f'age {bands[0].split("-")[0]}-{bands[-1].split("-")[-1]}'
        m = re.search(r'(?:tren|over|above|older than)\s*(\d{2})', q)
        m2 = re.search(r'(\d{2})\s*(?:\+|tuoi tro len|tro len|and over|or older|and older)', q)
        mm = m2 or m
        if mm and int(mm.group(1)) >= 18:
            lo = int(mm.group(1))
            bands = [b for b in AGE_BANDS if BAND_LOW[b] >= lo]
            if bands:
                return bands, f'từ {bands[0].split("-")[0]} tuổi trở lên', f'age {bands[0].split("-")[0]} and over'
        m = re.search(r'(?:duoi|under|below|younger than)\s*(\d{2})', q)
        if m and ('tuoi' in q or 'age' in q or 'young' in q or 'under' in q or 'below' in q or 'duoi' in q):
            hi = int(m.group(1))
            bands = [b for b in AGE_BANDS if BAND_LOW[b] + 5 <= hi]
            if bands:
                return bands, f'dưới {bands[-1].split("-")[-1] if bands[-1] != "80+" else "80"} tuổi'.replace(bands[-1].split('-')[-1], str(BAND_LOW[bands[-1]] + 5)), f'under age {BAND_LOW[bands[-1]] + 5}'
        if contains(q, ['nguoi cao tuoi', 'nguoi gia', 'elderly', 'seniors', 'older adults']):
            bands = [b for b in AGE_BANDS if BAND_LOW[b] >= 65]
            return bands, 'từ 65 tuổi trở lên', 'age 65 and over'
        if contains(q, ['nguoi tre', 'young adult', 'younger adults']):
            bands = [b for b in AGE_BANDS if BAND_LOW[b] < 40]
            return bands, 'dưới 40 tuổi', 'under age 40'
        return None

    def parse(self, question: str):
        """Tách câu hỏi thành ý định + bộ lọc. Trả về dict hoặc None nếu không hiểu."""
        q = ' ' + strip_accents(question.lower()).rstrip('?. ') + ' '
        raw = ' ' + question.lower().rstrip('?. ') + ' '
        lang = detect_language(question)
        metric = self.metric_of(q)
        info = {'lang': lang, 'metric': metric, 'filters': [], 'group_by': None, 'compare': None, 'target': None}
        # đã tách riêng điều kiện nhị phân: "tien tieu duong" chứa "tieu duong" nên duyệt theo thứ tự ưu tiên
        mentioned, rest = [], q
        for key in CONDITION_ORDER:
            kws = sorted(CONDITIONS[key][0], key=len, reverse=True)
            hit = next((k for k in kws if k in rest), None)
            if hit:
                mentioned.append(key)
                rest = rest.replace(hit, ' ')
        if 'active' in mentioned and 'inactive' in mentioned:
            mentioned.remove('active')
        # một cụm "hút thuốc" cụ thể (daily/former...) được xử lý như chiều smoker_status
        # chiều phân nhóm được nêu cụ thể (giới, BMI, hút thuốc...)
        dim_hits = {}
        for dim, spec in DIMENSIONS.items():
            vals = []
            for val, (kws, _, _) in spec['values'].items():
                if any(re.search(r'(?<![a-z])' + re.escape(k.strip()) + r'(?![a-z])', q) for k in kws):
                    vals.append(val)
            if vals:
                dim_hits[dim] = vals
        # 'nam' = giới tính chỉ khi không phải 'năm' (đã bỏ dấu nên kiểm tra trên câu gốc)
        if 'sex' in dim_hits and 'male' in dim_hits['sex'] and not re.search(r'(?<![\wă])nam(?![\wă])|\bmen\b|\bmale\b|\bman\b|đàn ông|nam giới', raw):
            dim_hits['sex'].remove('male')
            if not dim_hits['sex']:
                del dim_hits['sex']
        if 'smoker_status' in dim_hits and 'smoker' in mentioned:
            mentioned.remove('smoker')
            if contains(q, COMPARE_KW) and 'never' in dim_hits['smoker_status'] and len(dim_hits['smoker_status']) == 1:
                dim_hits['smoker_status'] = ['daily', 'some_days', 'never']
        if 'bmi_category' in dim_hits and 'normal' in dim_hits['bmi_category'] and not contains(q, ['bmi', 'can nang', 'weight', 'beo', 'thua can', 'thieu can', 'obes', 'overweight', 'underweight']):
            dim_hits['bmi_category'].remove('normal')
            if not dim_hits['bmi_category']:
                del dim_hits['bmi_category']
        for st in US_STATES:
            if re.search(r'(?<![a-z])' + st + r'(?![a-z])', q):
                dim_hits.setdefault('state', []).append(st.title() if st != 'district of columbia' else 'District of Columbia')
        age = self.parse_age(q)
        # mục tiêu của câu hỏi
        if metric == 'rate' or (metric is None and contains(q, COMPARE_KW)):
            if 'diabetes' in mentioned:
                info['target'] = 'diabetes'
                mentioned.remove('diabetes')
            elif 'prediabetes' in mentioned:
                info['target'] = 'prediabetes'
                mentioned.remove('prediabetes')
            elif mentioned:
                info['target'] = mentioned.pop(0)
            elif contains(q, ['mac benh', 'benh nhan', 'patient']):
                info['target'] = 'diabetes'
            if info['target'] is None and metric is None:
                return None
            metric = info['metric'] = metric or 'rate'
        elif metric in ('count', 'mean_bmi'):
            # đếm: các điều kiện là bộ lọc; mặc định đếm toàn bộ
            pass
        else:
            return None
        for key in mentioned:
            info['filters'].append(('cond', key))
        if age:
            info['filters'].append(('age', age))
        # chiều phân nhóm so sánh/theo nhóm
        for dim, vals in dim_hits.items():
            if dim == 'state' and len(vals) == 1 and not contains(q, ['so sanh', 'compare']):
                info['filters'].append(('dim', dim, vals))
            elif len(vals) >= 2 or (contains(q, COMPARE_KW) and dim != 'state'):
                if len(vals) >= 2:
                    info['compare'] = (dim, vals)
                else:
                    info['filters'].append(('dim', dim, vals))
            else:
                info['filters'].append(('dim', dim, vals))
        if info['compare'] is None and contains(q, COMPARE_KW) and 'sex' not in dim_hits:
            if contains(q, ['nam va nu', 'men and women', 'nam nu', 'men vs women', 'male and female']):
                info['compare'] = ('sex', ['male', 'female'])
        if info['compare'] is None and contains(q, COMPARE_KW) and contains(q, ['nam va nu', 'nam nu ', 'men and women', 'male and female', 'men vs women', 'nam so voi nu']):
            info['compare'] = ('sex', ['male', 'female'])
            info['filters'] = [f for f in info['filters'] if not (f[0] == 'dim' and f[1] == 'sex')]
        # phân nhóm "theo ..." / "by ..."
        if info['compare'] is None and contains(q, GROUP_KW) or contains(q, ['moi nhom', 'tung nhom']):
            for dim, spec in DIMENSIONS.items():
                if contains(q, [k for k in spec['kw'] if len(k) > 3]) or (dim == 'sex' and contains(q, [' gioi', ' sex', 'gender'])):
                    pat = [k for k in spec['kw'] if len(k) > 3] + ([' gioi', 'gender', ' sex'] if dim == 'sex' else [])
                    idx = [q.find(k) for k in pat if k in q]
                    if idx:
                        info['group_by'] = dim
                        info['filters'] = [f for f in info['filters'] if not (f[0] == 'dim' and f[1] == dim)
                                           and not (dim == 'age_group' and f[0] == 'age')]
                        break
            if info['group_by'] is None and contains(q, ['theo bang', 'by state', 'each state', 'tung bang']):
                info['group_by'] = 'state'
            if info['group_by'] is None and contains(q, ['hut thuoc', 'smoking']) and metric != 'mean_bmi' and 'smoker_status' not in dim_hits:
                info['group_by'] = 'smoker_status'
        if info['group_by'] == 'smoker_status':
            info['filters'] = [f for f in info['filters'] if f != ('cond', 'smoker')]
        if info['group_by'] is None and info['compare'] is None and metric == 'rate' and info['target'] is None:
            return None
        return info

    # ---- tính toán -------------------------------------------------------------------------
    def apply_filters(self, info):
        df, desc_vi, desc_en = self.df, [], []
        for f in sorted(info['filters'], key=lambda f: {'dim': 0, 'age': 1, 'cond': 2}[f[0]]):
            if f[0] == 'cond':
                _, vi, en, fn = CONDITIONS[f[1]]
                df = df[fn(df)]
                desc_vi.append(vi)
                desc_en.append(en)
            elif f[0] == 'age':
                bands, vi, en = f[1]
                df = df[df.age_group.isin(bands)]
                desc_vi.append(vi)
                desc_en.append(en)
            else:
                _, dim, vals = f
                col = DIMENSIONS[dim]['col']
                df = df[df[col].isin(vals)]
                spec = DIMENSIONS[dim]['values']
                if dim == 'state':
                    desc_vi.append('ở bang ' + ', '.join(vals))
                    desc_en.append('in ' + ', '.join(vals))
                else:
                    desc_vi.append(' hoặc '.join(spec[v][1] for v in vals))
                    desc_en.append(' or '.join(spec[v][2] for v in vals))
        return df, desc_vi, desc_en

    def target_mask(self, df, target):
        return CONDITIONS[target][3](df)

    def target_phrase(self, target, lang):
        return CONDITIONS[target][1 if lang == 'vi' else 2]

    def group_label(self, dim, val, lang):
        if dim == 'age_group':
            return f'{val} tuổi' if lang == 'vi' else f'age {val}'
        if dim == 'general_health':
            return HEALTH_LABEL[int(val)] if lang == 'vi' else HEALTH_LABEL_EN[int(val)]
        if dim == 'education':
            return EDU_VI.get(val, val) if lang == 'vi' else val.replace('_', ' ')
        if dim == 'smoker_status':
            return SMOKER_VI.get(val, val) if lang == 'vi' else val.replace('_', ' ')
        if dim == 'bmi_category':
            return BMI_VI.get(val, val) if lang == 'vi' else val
        if dim == 'sex':
            return {'male': 'nam', 'female': 'nữ'}.get(val, val) if lang == 'vi' else val
        return str(val)

    def answer(self, question: str):
        """Trả về dict kết quả hoặc None nếu không hiểu câu hỏi."""
        info = self.parse(question)
        if info is None:
            return None
        lang = info['lang']
        vi = lang == 'vi'
        base, dvi, den = self.apply_filters(info)
        scope = (' ' + ', '.join(dvi)) if vi and dvi else (' (' + ', '.join(den) + ')' if den else '')
        total = len(base)
        year = 'năm 2023' if vi else 'in 2023'
        metric, target = info['metric'], info['target']
        if total == 0:
            return {'answer': 'Không có người nào trong mẫu thỏa điều kiện này.' if vi else 'No one in the sample matches these conditions.',
                    'intent': metric, 'rows': 0, 'table': None, 'interpretation': 'Bộ lọc loại hết mọi dòng.' if vi else 'The filters removed every row.'}
        # ---------- so sánh hai nhóm / theo nhóm ----------
        dim = info['group_by'] or (info['compare'][0] if info['compare'] else None)
        if dim:
            col = DIMENSIONS[dim]['col']
            sub = base
            if info['compare']:
                sub = base[base[col].isin(info['compare'][1])]
            if metric == 'mean_bmi':
                grp = sub.groupby(col).bmi.agg(['mean', 'size'])
                head = ['BMI trung bình', 'Số người'] if vi else ['Average BMI', 'People']
                rows_raw = [(k, r['mean'], int(r['size'])) for k, r in grp.iterrows()]
                fmt = lambda v: fmt_num(v, lang)
            else:
                mask = self.target_mask(sub, target or 'diabetes').groupby(sub[col])
                grp = pd.DataFrame({'p': mask.mean() * 100, 'n': mask.size(), 'k': mask.sum()})
                head = ['Tỉ lệ', 'Số ca', 'Số người'] if vi else ['Rate', 'Cases', 'People']
                rows_raw = [(k, r['p'], int(r['n']), int(r['k'])) for k, r in grp.iterrows()]
                fmt = lambda v: fmt_pct(v, lang)
            order = AGE_BANDS if dim == 'age_group' else None
            if dim == 'bmi_category':
                order = ['underweight', 'normal', 'overweight', 'obese']
            elif dim == 'income_group':
                order = ['<15k', '15-25k', '25-35k', '35-50k', '50-100k', '100-200k', '200k+']
            elif dim == 'education':
                order = ['less_than_high_school', 'high_school', 'some_college', 'college_graduate']
            elif dim == 'smoker_status':
                order = ['never', 'former', 'some_days', 'daily']
            elif dim == 'general_health':
                order = [1, 2, 3, 4, 5]
            if order:
                rows_raw.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
            else:
                rows_raw.sort(key=lambda r: -r[1])
            if dim == 'state':
                rows_raw = [r for r in rows_raw if r[2] >= 50] if metric != 'mean_bmi' else rows_raw
                rows_raw = rows_raw[:10] if info['group_by'] else rows_raw
            table_rows = []
            for r in rows_raw:
                lab = self.group_label(dim, r[0], lang)
                if metric == 'mean_bmi':
                    table_rows.append([lab, fmt(r[1]), fmt_int(r[2], lang)])
                else:
                    table_rows.append([lab, fmt(r[1]), fmt_int(r[3], lang), fmt_int(r[2], lang)])
            what = ('BMI trung bình' if vi else 'average BMI') if metric == 'mean_bmi' else \
                (f'tỉ lệ người {self.target_phrase(target or "diabetes", lang)}' if vi else f'the share of people {self.target_phrase(target or "diabetes", lang)}')
            first, last = rows_raw[0], rows_raw[-1]
            top = max(rows_raw, key=lambda r: r[1])
            low = min(rows_raw, key=lambda r: r[1])
            if info['compare'] and len(rows_raw) == 2:
                a, b = rows_raw[0], rows_raw[1]
                la, lb = self.group_label(dim, a[0], lang), self.group_label(dim, b[0], lang)
                if vi:
                    ans = (f'So sánh {what}{scope} {year}: nhóm {la} là {fmt(a[1])}, nhóm {lb} là {fmt(b[1])}. '
                           f'Chênh lệch {fmt_num(abs(a[1] - b[1]), lang)}{"" if metric == "mean_bmi" else " điểm phần trăm"}.')
                else:
                    ans = (f'Comparing {what}{scope}: {la} = {fmt(a[1])}, {lb} = {fmt(b[1])}. '
                           f'The gap is {fmt_num(abs(a[1] - b[1]), lang)}{"" if metric == "mean_bmi" else " percentage points"}.')
            else:
                ln, lt = self.group_label(dim, top[0], lang), self.group_label(dim, low[0], lang)
                dn = DIMENSIONS[dim][lang]
                if vi:
                    ans = (f'Phân theo {dn}, {what}{scope} trong {fmt_int(total, lang)} người khảo sát {year} cao nhất ở nhóm {ln} ({fmt(top[1])}) '
                           f'và thấp nhất ở nhóm {lt} ({fmt(low[1])}). Bảng bên dưới có đủ các nhóm.')
                else:
                    ans = (f'By {dn}, {what}{scope} among {fmt_int(total, lang)} respondents is highest for {ln} ({fmt(top[1])}) '
                           f'and lowest for {lt} ({fmt(low[1])}). See the table for every group.')
            return {'answer': ans, 'intent': 'compare' if info['compare'] else 'group', 'rows': total,
                    'table': {'columns': [DIMENSIONS[dim][lang][0].upper() + DIMENSIONS[dim][lang][1:]] + head, 'rows': table_rows},
                    'interpretation': self.interpretation(info, lang, scope)}
        # ---------- một con số ----------
        if metric == 'count':
            if not info['filters']:
                ans = (f'Dữ liệu có {fmt_int(self.n, lang)} người trả lời khảo sát BRFSS {year[-4:]} (mỗi dòng là một người).' if vi
                       else f'The dataset contains {fmt_int(self.n, lang)} BRFSS 2023 respondents (one row per person).')
            else:
                ans = (f'Có {fmt_int(total, lang)} người{scope} trong {fmt_int(self.n, lang)} người khảo sát năm 2023 ({fmt_pct(total / self.n * 100, lang)}).' if vi
                       else f'{fmt_int(total, lang)} of {fmt_int(self.n, lang)} respondents{scope} ({fmt_pct(total / self.n * 100, lang)}).')
            return {'answer': ans, 'intent': 'count', 'rows': total, 'value': total, 'table': None,
                    'interpretation': self.interpretation(info, lang, scope)}
        if metric == 'mean_bmi':
            mean = base.bmi.mean()
            who = f'người{scope}' if vi else f'respondents{scope}'
            ans = (f'BMI trung bình của {who if info["filters"] else "người được khảo sát"} là {fmt_num(mean, lang)} (tính trên {fmt_int(total, lang)} người, năm 2023).' if vi
                   else f'The average BMI of {who if info["filters"] else "respondents"} is {fmt_num(mean, lang)} (over {fmt_int(total, lang)} people, 2023).')
            return {'answer': ans, 'intent': 'mean_bmi', 'rows': total, 'value': float(mean), 'table': None,
                    'interpretation': self.interpretation(info, lang, scope)}
        # rate
        mask = self.target_mask(base, target)
        k, pct = int(mask.sum()), mask.mean() * 100
        tp = self.target_phrase(target, lang)
        if vi:
            pop = f'{fmt_int(total, lang)} người{scope}' if info['filters'] else f'{fmt_int(total, lang)} người được khảo sát'
            ans = f'Trong {pop} năm 2023, {fmt_pct(pct, lang)} ({fmt_int(k, lang)} người) {tp}.'
        else:
            pop = f'{fmt_int(total, lang)} respondents{scope}' if info['filters'] else f'{fmt_int(total, lang)} surveyed people'
            ans = f'Among {pop} in 2023, {fmt_pct(pct)} ({fmt_int(k, lang)} people) are {tp}.'
        if info['filters'] and target == 'diabetes':
            overall = (self.df.diabetes_status == 'yes').mean() * 100
            ans += (f' Để so sánh, tỉ lệ chung của cả mẫu là {fmt_pct(overall, lang)}.' if vi else f' For reference, the whole sample has {fmt_pct(overall)}.')
        return {'answer': ans, 'intent': 'rate', 'rows': total, 'value': float(pct), 'cases': k, 'table': None,
                'interpretation': self.interpretation(info, lang, scope)}

    def interpretation(self, info, lang, scope):
        vi = lang == 'vi'
        names = {'count': 'đếm số người', 'rate': 'tính tỉ lệ phần trăm', 'mean_bmi': 'tính BMI trung bình'} if vi else \
            {'count': 'counting people', 'rate': 'computing a percentage', 'mean_bmi': 'computing average BMI'}
        parts = [names[info['metric']]]
        if info['target']:
            parts.append(('của người ' if vi else 'of people ') + self.target_phrase(info['target'], lang))
        if info['group_by']:
            parts.append((f'phân theo {DIMENSIONS[info["group_by"]]["vi"]}') if vi else f'by {DIMENSIONS[info["group_by"]]["en"]}')
        if info['compare']:
            parts.append(('so sánh theo ' + DIMENSIONS[info['compare'][0]]['vi']) if vi else f'comparing by {DIMENSIONS[info["compare"][0]]["en"]}')
        if scope:
            parts.append(('chỉ xét' if vi else 'restricted to') + scope)
        return ('Công cụ hiểu câu hỏi là: ' if vi else 'The tool understood this as: ') + ', '.join(parts) + '.'

    def help_text(self, lang='vi'):
        if lang == 'vi':
            return ('Mình chưa hiểu đây là câu hỏi thống kê nào. Bạn thử các dạng câu sau: ' + ' · '.join(SAMPLE_QUESTIONS_VI)
                    + '. Có thể nói về nhóm tuổi, giới tính, BMI, huyết áp cao, cholesterol cao, hút thuốc, vận động, bệnh tim, đột quỵ hoặc tên bang.')
        return ('I could not tell which statistic you want. Try: How many people are in the survey? · What percentage of people have diabetes? · '
                'Diabetes rate by age group · Compare diabetes rate between men and women · Average BMI of people with diabetes.')
