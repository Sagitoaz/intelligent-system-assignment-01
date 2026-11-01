"""Hệ đa agent Chương 7: điều phối, thống kê, truy hồi, tính toán, đồ thị, viết và phản biện.

Nguyên tắc thiết kế: đa agent chỉ đáng giá khi câu hỏi cần nhiều loại năng lực (số liệu từ dữ liệu, giải thích từ tài liệu,
tính toán chính xác) và có kiểm tra độc lập bằng bằng chứng chạy được. Câu hỏi một bước thì một RAG đơn là đủ.

Các agent giao tiếp bằng Message ghi trong State. Agent phản biện không phải mô hình ngôn ngữ: nó trích mọi con số trong bản nháp
và đối chiếu với sự kiện của công cụ và với các đoạn tài liệu. Có lỗi thì gửi yêu cầu sửa cho agent viết; tối đa 2 vòng sửa.
"""
import re
import time
from dataclasses import dataclass, field

MAX_REVISIONS = 2
SAMPLE_SIZE_NOTE_VI = ('Đây là số liệu của mẫu khảo sát CDC BRFSS 2023 gồm {n} người, chưa hiệu chỉnh trọng số, '
                       'nên chỉ mô tả mẫu này chứ không đại diện cho toàn dân Mỹ.')
SAMPLE_SIZE_NOTE_EN = ('These figures describe the CDC BRFSS 2023 survey sample of {n} people, without survey weights, '
                       'so they describe this sample and do not represent the whole US population.')


@dataclass
class Message:
    sender: str
    receiver: str
    kind: str
    payload: dict


@dataclass
class State:
    question: str
    domain: str
    report: str | None = None
    plan: dict = field(default_factory=dict)
    facts: list = field(default_factory=list)        # sự kiện từ agent thống kê
    derived: list = field(default_factory=list)      # kết quả của agent tính toán
    evidence: list = field(default_factory=list)     # các đoạn tài liệu truy hồi
    explanation: list = field(default_factory=list)  # câu giải thích rút ra, có mã nguồn
    graph_context: list = field(default_factory=list)
    drafts: list = field(default_factory=list)       # [{'round', 'by', 'text'}]
    reviews: list = field(default_factory=list)      # kết quả mỗi lần phản biện
    answer: str = ''
    english_answer: str = ''
    status: str = 'pending'
    messages: list = field(default_factory=list)
    query: str = ''
    language: str = ''
    steps: list = field(default_factory=list)
    dialogue: list = field(default_factory=list)

    def send(self, sender, receiver, kind, **payload):
        self.messages.append(Message(sender, receiver, kind, payload))


# ---------------------------------------------------------------------------------------------
# Tiện ích số học và văn bản
# ---------------------------------------------------------------------------------------------
def norm(text):
    from src.chapter04.stats import strip_accents
    return ' ' + strip_accents(text.lower()).rstrip('?. ') + ' '


NUM = re.compile(r'(?<![A-Za-z0-9])(\d{1,3}(?:[.,]\d{3})+(?![\d])|\d+(?:[.,]\d+)?)(\s*%|\s*(?:million|billion|thousand|triệu|tỷ|tỉ|nghìn|ngàn|lần|times|điểm phần trăm|percentage points?)\b)?',
                 re.I)
MULT = {'million': 1e6, 'triệu': 1e6, 'billion': 1e9, 'tỷ': 1e9, 'tỉ': 1e9, 'thousand': 1e3, 'nghìn': 1e3, 'ngàn': 1e3}
DOC_MARK = re.compile(r'according|niddk|cdc\b|medlineplus|the (?:passages?|documents?|websites?)|theo |tài liệu|nguồn', re.I)
SAMPLE_MARK = re.compile(r'\bsample\b|survey|brfss|respondents?|150[.,]?000|mẫu|khảo sát|dữ liệu|people in the data', re.I)
LIMIT_MARK = re.compile(r'sample|mẫu|brfss|survey|khảo sát|unweighted|không có trọng số|chưa hiệu chỉnh', re.I)


def parse_number(text, unit):
    unit = (unit or '').strip().lower()
    if re.fullmatch(r'\d{1,3}(?:[.,]\d{3})+', text):
        return float(re.sub(r'[.,]', '', text)), 0
    clean = text.replace(',', '.')
    decimals = len(clean.split('.')[1]) if '.' in clean else 0
    return float(clean) * MULT.get(unit, 1), decimals


def sentences_of(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p for p in parts if p.strip()]


def extract_numbers(text, ignore=()):
    """Trả về danh sách {'text','value','decimals','percent','scale','sentence'} cho các số cần có bằng chứng."""
    out = []
    skip_text = ' '.join(str(x) for x in ignore)
    skip_numbers = {m.group(1) for m in NUM.finditer(skip_text)}
    for sentence in sentences_of(text):
        for m in NUM.finditer(sentence):
            raw, unit = m.group(1), (m.group(2) or '').strip()
            before = sentence[max(0, m.start() - 6):m.start()].lower()
            if re.search(r'(type|típ|tip|loại)\s*$', before) or re.match(r'^\s*[-*]?\s*$', sentence[:m.start()]) and re.match(r'^\d\.?$', raw) and not unit:
                continue
            value, decimals = parse_number(raw, unit.lower())
            percent = unit.startswith('%')
            if not unit and decimals == 0 and (1900 <= value <= 2100 or 0 <= value <= 3):
                continue  # năm, số đếm nhỏ
            if raw in skip_numbers:
                continue
            scale = MULT.get(unit.lower(), 1)
            out.append({'text': (raw + (' ' + unit if unit else '')).strip(), 'value': value, 'decimals': decimals,
                        'percent': percent, 'scale': scale, 'sentence': sentence})
    return out


def close(x, decimals, scale, v):
    """x (viết với `decimals` chữ số thập phân, nhân `scale`) khớp v khi làm tròn hợp lý."""
    tol = 0.5 * (10 ** -decimals) * scale + 1e-9
    return abs(x - v) <= tol or (v != 0 and abs(x - v) / abs(v) <= 0.002)


def fact_numbers(state):
    """Mọi số mà công cụ đã cung cấp, kèm nhãn nguồn."""
    out = []
    for f in state.facts + state.derived:
        out.append((f['value'], f['id'], f))
        for key in ('numerator', 'denominator'):
            if f.get(key) is not None:
                out.append((float(f[key]), f['id'], f))
    return out


# ---------------------------------------------------------------------------------------------
# Agent điều phối (Planner)
# ---------------------------------------------------------------------------------------------
HEALTH_WORDS = ['tieu duong', 'diabet', 'insulin', 'duong huyet', 'duong trong mau', 'glucose', 'blood sugar', 'a1c', 'huyet ap', 'cholesterol', 'beo phi', 'bmi',
                'tuy', 'pancrea', 'suc khoe', 'benh', 'thuoc', 'tim', 'than ', 'mat ', 'than kinh', 'can nang', 'an uong', 'van dong', 'hut thuoc', 'dot quy',
                'metformin', 'bien chung', 'trieu chung', 'xet nghiem', 'thai ky', 'bnr', 'brfss', 'khao sat', 'gia dinh', 'tuoi', 'nam gioi', 'nu gioi',
                'hypoglyc', 'hyperglyc', 'cgm', 'chan', 'nguoi mac', 'chan doan', 'nguy co', 'mac benh', 'health', 'disease', 'symptom', 'diagnos']
FINANCE_WORDS = ['doanh thu', 'loi nhuan', 'cong ty', 'bao cao', 'revenue', 'income', 'visa', 'mastercard', 'apple', 'microsoft', 'tai chinh', 'co phieu', 'amazon', 'google',
                 'nvidia', 'tesla', 'meta', 'jpmorgan', '10-k', 'lĩnh vực', 'linh vuc', 'hoat dong']
EXPLAIN_WORDS = ['vi sao', 'tai sao', 'giai thich', 'nguyen nhan', 'la gi', 'the nao', 'co nghia', 'nen lam', 'trieu chung', 'bien chung', 'co the gay', 'tac dong',
                 'why', 'explain', 'what is', 'how does', 'cach nao', 'lam sao', 'co anh huong', 'dieu gi', 'yeu to nguy co', 'co lien quan']
RATIO_WORDS = ['gap may lan', 'gap bao nhieu lan', 'bao nhieu lan', 'gap doi', 'gap nhieu lan', 'lan so voi', 'gap ', 'times as']
GAP_WORDS = ['chenh lech', 'khac nhau bao nhieu', 'hon bao nhieu', 'cao hon bao nhieu', 'thap hon bao nhieu', 'khoang cach', 'gap between']


class PlannerAgent:
    """Đọc câu hỏi, phân loại và lập kế hoạch bằng luật và từ khóa (xác định, không dùng mô hình sinh)."""
    role = 'Agent điều phối'
    description = 'Đọc câu hỏi, quyết định cần những agent nào và gọi theo thứ tự.'

    def __init__(self, stats):
        self.stats = stats

    def act(self, state):
        t0 = time.perf_counter()
        q = norm(state.question)
        plan = {'scope': 'diabetes', 'mode': 'simple', 'facts': [], 'calc': None, 'explain': True, 'explain_question': state.question, 'agents': []}
        if state.domain == 'finance':
            plan.update(scope='finance', summary='Câu hỏi về công ty/báo cáo tài chính: cần truy hồi đoạn báo cáo và tra đồ thị để biết đoạn thuộc công ty nào.',
                        agents=['RetrievalAgent', 'GraphAgent', 'WriterAgent', 'CriticAgent'])
        elif not any(w in q for w in HEALTH_WORDS) and not any(w in q for w in FINANCE_WORDS):
            plan.update(scope='off_topic', mode='off_topic', explain=False, agents=[],
                        summary='Câu hỏi không liên quan đến tiểu đường hay sức khỏe nên kho tài liệu và dữ liệu khảo sát không thể trả lời. Nhóm từ chối ngay, không tốn công truy hồi.')
        else:
            self.plan_diabetes(state, q, plan)
        plan['mode_label'] = {'simple': 'Đơn giản: một RAG là đủ', 'data': 'Phức hợp: cần số liệu thật', 'off_topic': 'Ngoài phạm vi'}.get(plan['mode'], plan['mode'])
        state.plan = plan
        state.send('planner', 'coordinator', 'plan', mode=plan['mode'], agents=plan['agents'], facts=len(plan['facts']), calc=bool(plan['calc']))
        state.steps.append({'agent': 'PlannerAgent', 'role': self.role, 'role_description': self.description, 'summary': plan['summary'], 'outcome': plan['mode_label'],
                            'seconds': time.perf_counter() - t0, 'evidence': {'plan': plan}})

    def plan_diabetes(self, state, q, plan):
        info = self.stats.parse(state.question) if hasattr(self.stats, 'parse') else None
        explain_hit = next((w for w in EXPLAIN_WORDS if w in q), None)
        statistical = bool(info) and (info['metric'] in ('rate', 'count', 'mean_bmi') or info.get('compare') or info.get('group_by'))
        if statistical and not self.stats.looks_statistical(state.question):
            statistical = False
        extreme_dim = next((d for kw, d in [(' nhom tuoi', 'age_group'), (' nhom thu nhap', 'income_group'), (' hoc van', 'education'), (' nhom bmi', 'bmi_category')] if kw in q), None)
        extreme = bool(extreme_dim) and any(w in q for w in ['cao nhat', 'thap nhat', 'nhieu nhat', 'it nhat'])
        mean_bmi = 'bmi' in q and any(w in q for w in ['trung binh', 'average', 'mean'])
        two_diabetes = mean_bmi and any(w in q for w in ['khong mac', 'khong bi', 'so voi nguoi khong', 'khong tieu duong'])
        if not (statistical or extreme or two_diabetes):
            plan.update(summary='Câu hỏi hỏi về kiến thức chung, không cần số liệu từ dữ liệu khảo sát và không cần tính toán. Một RAG đơn (truy hồi rồi viết) là đủ, '
                                'nên nhóm chỉ gọi agent truy hồi và agent viết, rồi agent phản biện soát số trong câu trả lời. Đây là trường hợp nhóm agent không có lợi thế.',
                        agents=['RetrievalAgent', 'WriterAgent', 'CriticAgent'])
            return
        plan['mode'] = 'data'
        target = (info or {}).get('target') or 'diabetes'
        filters = list((info or {}).get('filters') or [])
        facts, calc = [], None
        metric = 'mean_bmi' if mean_bmi else ((info or {}).get('metric') if (info or {}).get('metric') in ('rate', 'count') else 'rate')
        if two_diabetes:
            facts = [{'kind': 'mean_bmi', 'target': 'diabetes', 'filters': [], 'extra': 'diabetes_yes', 'short_vi': 'nhóm mắc tiểu đường', 'short_en': 'the diabetes group'},
                     {'kind': 'mean_bmi', 'target': 'diabetes', 'filters': [], 'extra': 'diabetes_no', 'short_vi': 'nhóm không mắc tiểu đường', 'short_en': 'the group without diabetes'}]
            calc = {'op': 'subtract', 'a': 'F1', 'b': 'F2', 'unit': 'BMI'}
            if any(w in q for w in RATIO_WORDS[:6]):
                calc = {'op': 'divide', 'a': 'F1', 'b': 'F2', 'unit': 'BMI'}
        elif 'khong van dong' in q and any(w in q for w in ['co van dong', 'nguoi van dong', 'nguoi co tap', 'co tap the duc']):
            facts = [{'kind': 'rate', 'target': target, 'filters': [('cond', 'inactive')], 'short_vi': 'nhóm không vận động', 'short_en': 'the inactive group'},
                     {'kind': 'rate', 'target': target, 'filters': [('cond', 'active')], 'short_vi': 'nhóm có vận động', 'short_en': 'the active group'}]
            calc = {'op': 'divide' if any(w in q for w in RATIO_WORDS) else 'subtract', 'a': 'F1', 'b': 'F2', 'unit': '%'}
        elif extreme:
            dim = extreme_dim
            facts = [{'kind': 'extreme', 'which': 'top', 'dim': dim, 'target': target, 'filters': [f for f in filters if f[0] != 'age'], 'short_vi': 'nhóm cao nhất', 'short_en': 'the highest age group'},
                     {'kind': 'extreme', 'which': 'low', 'dim': dim, 'target': target, 'filters': [f for f in filters if f[0] != 'age'], 'short_vi': 'nhóm thấp nhất', 'short_en': 'the lowest age group'}]
            calc = {'op': 'divide', 'a': 'F1', 'b': 'F2', 'unit': '%'}
        elif info and info.get('compare'):
            dim, vals = info['compare']
            for v in vals[:2]:
                facts.append({'kind': metric if metric != 'mean_bmi' else 'mean_bmi', 'target': target, 'filters': filters + [('dim', dim, [v])], 'short_vi': '', 'short_en': ''})
            if len(facts) == 2:
                calc = {'op': 'divide' if any(w in q for w in RATIO_WORDS) else 'subtract', 'a': 'F1', 'b': 'F2', 'unit': 'BMI' if metric == 'mean_bmi' else '%'}
        elif info and info.get('group_by') and info['group_by'] in ('age_group', 'sex', 'bmi_category', 'smoker_status', 'income_group', 'education', 'general_health'):
            facts = [{'kind': 'extreme', 'which': 'top', 'dim': info['group_by'], 'target': target, 'filters': filters, 'short_vi': 'nhóm cao nhất', 'short_en': 'the highest group'},
                     {'kind': 'extreme', 'which': 'low', 'dim': info['group_by'], 'target': target, 'filters': filters, 'short_vi': 'nhóm thấp nhất', 'short_en': 'the lowest group'}]
            calc = {'op': 'divide', 'a': 'F1', 'b': 'F2', 'unit': '%'}
        else:
            facts = [{'kind': metric, 'target': target, 'filters': filters, 'short_vi': '', 'short_en': ''}]
            gap = any(w in q for w in GAP_WORDS + ['diem phan tram'])
            if filters and metric == 'rate' and any(w in q for w in RATIO_WORDS + GAP_WORDS + ['ca mau', 'so voi ca', 'chung']):
                facts.append({'kind': 'rate', 'target': target, 'filters': [], 'short_vi': 'cả mẫu', 'short_en': 'the whole sample'})
                ratio = any(w in q for w in RATIO_WORDS)
                calc = {'op': 'subtract' if (gap and not ratio) else 'divide', 'a': 'F1', 'b': 'F2', 'unit': '%'}
        for i, spec in enumerate(facts, 1):
            spec['id'] = f'F{i}'
        plan['facts'] = facts
        plan['calc'] = ({**calc, 'id': 'D1'} if calc else None)
        plan['explain'] = bool(explain_hit)
        if explain_hit:
            idx = q.find(explain_hit)
            rest = norm(state.question)[idx:]
            raw = state.question
            marker = re.search(r'(vì sao|tại sao|giải thích|nguyên nhân|là gì|thế nào|why|explain)', raw, re.I)
            plan['explain_question'] = raw[marker.start():].strip() if marker and len(raw) - marker.start() >= 12 else raw
        parts = ['Câu hỏi cần số liệu thật từ dữ liệu khảo sát (agent thống kê)']
        if calc:
            parts.append('một phép tính chính xác (agent tính toán, không để mô hình tự tính)')
        if plan['explain']:
            parts.append('một phần giải thích từ tài liệu (agent truy hồi)')
        plan['agents'] = ['StatsAgent'] + (['CalculatorAgent'] if calc else []) + (['RetrievalAgent'] if plan['explain'] else []) + ['WriterAgent', 'CriticAgent']
        plan['summary'] = (', '.join(parts) + '. Một RAG đơn chỉ đọc tài liệu nên không có các con số này; agent phản biện sẽ đối chiếu từng số trong bản nháp với công cụ.')


# ---------------------------------------------------------------------------------------------
# Agent thống kê
# ---------------------------------------------------------------------------------------------
class StatsAgent:
    role = 'Agent thống kê'
    description = 'Tính số liệu thật từ dữ liệu khảo sát CDC BRFSS 2023 (150.000 người) bằng pandas.'

    def __init__(self, stats):
        self.stats = stats

    def subset(self, spec):
        from src.chapter04.stats import CONDITIONS
        df, dvi, den = self.stats.apply_filters({'filters': spec['filters']})
        extra = spec.get('extra')
        if extra == 'diabetes_yes':
            df, dvi, den = df[df.diabetes_status == 'yes'], dvi + ['mắc tiểu đường'], den + ['with diabetes']
        elif extra == 'diabetes_no':
            df, dvi, den = df[df.diabetes_status == 'no'], dvi + ['không mắc tiểu đường'], den + ['without diabetes']
        return df, dvi, den

    def evaluate(self, spec, state):
        from src.chapter04.stats import fmt_pct, fmt_int, fmt_num, CONDITIONS, DIMENSIONS
        stats = self.stats
        spec = dict(spec)
        dim_label = None
        if spec['kind'] == 'extreme':
            df, dvi, den = self.subset(spec)
            col = DIMENSIONS[spec['dim']]['col']
            grouped = stats.target_mask(df, spec['target']).groupby(df[col])
            table = [(k, float(v) * 100, int(n), int(c)) for k, v, n, c in zip(grouped.mean().index, grouped.mean(), grouped.size(), grouped.sum())]
            table = [r for r in table if r[2] >= 200]
            pick = max(table, key=lambda r: r[1]) if spec['which'] == 'top' else min(table, key=lambda r: r[1])
            group, pct, n, k = pick
            dim_label = (spec['dim'], group)
            spec = {**spec, 'kind': 'rate', 'filters': list(spec['filters']) + [('dim', spec['dim'], [group])] if spec['dim'] != 'age_group' else list(spec['filters']), 'age_band': group if spec['dim'] == 'age_group' else None}
            gl_vi, gl_en = stats.group_label(dim_label[0], group, 'vi'), stats.group_label(dim_label[0], group, 'en')
            who_vi = f'người thuộc {DIMENSIONS[spec["dim"]]["vi"]} {gl_vi}' if spec['dim'] != 'age_group' else f'người thuộc nhóm tuổi {group}'
            who_en = f'people in the {DIMENSIONS[spec["dim"]]["en"]} {gl_en}' if spec['dim'] != 'age_group' else f'people aged {group}'
            tgt_vi, tgt_en = stats.target_phrase(spec['target'], 'vi'), stats.target_phrase(spec['target'], 'en')
            return {'metric': 'rate', 'value': pct, 'unit': '%', 'numerator': k, 'denominator': n, 'group': str(group),
                    'label_vi': f'Tỉ lệ người {tgt_vi} trong {who_vi} (nhóm có tỉ lệ {"cao" if spec["which"] == "top" else "thấp"} nhất)',
                    'label_en': f'Share of people {tgt_en} among {who_en} (the {"highest" if spec["which"] == "top" else "lowest"} group)',
                    'sentence_vi': f'Nhóm có tỉ lệ {"cao" if spec["which"] == "top" else "thấp"} nhất là {who_vi}: {fmt_pct(pct, "vi")} ({fmt_int(k, "vi")} trong {fmt_int(n, "vi")} người) {tgt_vi}.',
                    'sentence_en': f'The {"highest" if spec["which"] == "top" else "lowest"} rate is among {who_en}: {fmt_pct(pct)} ({fmt_int(k, "en")} of {fmt_int(n, "en")}) {tgt_en}.',
                    'short_vi': f'nhóm tuổi {group}' if spec['dim'] == 'age_group' else f'{DIMENSIONS[spec["dim"]]["vi"]} {gl_vi}', 'short_en': f'the {group} age group' if spec['dim'] == 'age_group' else f'the {DIMENSIONS[spec["dim"]]["en"]} {gl_en}'}
        df, dvi, den = self.subset(spec)
        n = len(df)
        scope_vi = ' '.join(dvi)
        scope_en = ' '.join(den)
        if spec['kind'] == 'mean_bmi':
            mean = float(df.bmi.mean())
            return {'metric': 'mean_bmi', 'value': mean, 'unit': 'BMI', 'denominator': n,
                    'label_vi': f'BMI trung bình của người {scope_vi}' if scope_vi else 'BMI trung bình của cả mẫu',
                    'label_en': f'Average BMI of people {scope_en}' if scope_en else 'Average BMI of the whole sample',
                    'sentence_vi': f'BMI trung bình của người {scope_vi or "trong cả mẫu"} là {fmt_num(mean, "vi")} (tính trên {fmt_int(n, "vi")} người).',
                    'sentence_en': f'The average BMI of people {scope_en or "in the whole sample"} is {fmt_num(mean, "en")} (over {fmt_int(n, "en")} people).',
                    'short_vi': spec.get('short_vi') or ('nhóm ' + scope_vi if scope_vi else 'cả mẫu'), 'short_en': spec.get('short_en') or ('the group ' + scope_en if scope_en else 'the whole sample')}
        if spec['kind'] == 'count':
            return {'metric': 'count', 'value': float(n), 'unit': 'người', 'denominator': self.stats.n, 'numerator': n,
                    'label_vi': f'Số người {scope_vi}' if scope_vi else 'Số người trong mẫu', 'label_en': f'Number of people {scope_en}' if scope_en else 'Sample size',
                    'sentence_vi': f'Có {fmt_int(n, "vi")} người {scope_vi} trong {fmt_int(self.stats.n, "vi")} người được khảo sát.'.replace('  ', ' '),
                    'sentence_en': f'{fmt_int(n, "en")} of {fmt_int(self.stats.n, "en")} respondents {scope_en}.'.replace('  ', ' '),
                    'short_vi': spec.get('short_vi') or ('nhóm ' + scope_vi if scope_vi else 'cả mẫu'), 'short_en': spec.get('short_en') or ('the group ' + scope_en if scope_en else 'the whole sample')}
        mask = stats.target_mask(df, spec['target'])
        k, pct = int(mask.sum()), float(mask.mean() * 100)
        tgt_vi, tgt_en = stats.target_phrase(spec['target'], 'vi'), stats.target_phrase(spec['target'], 'en')
        pop_vi = f'{fmt_int(n, "vi")} người {scope_vi}' if scope_vi else f'{fmt_int(n, "vi")} người được khảo sát'
        pop_en = f'{fmt_int(n, "en")} respondents {scope_en}' if scope_en else f'{fmt_int(n, "en")} surveyed people'
        return {'metric': 'rate', 'value': pct, 'unit': '%', 'numerator': k, 'denominator': n,
                'label_vi': f'Tỉ lệ người {tgt_vi}' + (f' trong nhóm {scope_vi}' if scope_vi else ' trong cả mẫu'),
                'label_en': f'Share of people {tgt_en}' + (f' among those {scope_en}' if scope_en else ' in the whole sample'),
                'sentence_vi': f'Trong {pop_vi}, {fmt_pct(pct, "vi")} ({fmt_int(k, "vi")} người) {tgt_vi}.',
                'sentence_en': f'Among {pop_en}, {fmt_pct(pct)} ({fmt_int(k, "en")} people) are {tgt_en}.',
                'short_vi': spec.get('short_vi') or ('nhóm ' + scope_vi if scope_vi else 'cả mẫu'), 'short_en': spec.get('short_en') or ('the group ' + scope_en if scope_en else 'the whole sample')}

    def act(self, state):
        t0 = time.perf_counter()
        for spec in state.plan['facts']:
            fact = self.evaluate(spec, state)
            fact.update(id=spec['id'], source=self.stats.source, source_url=self.stats.url, tool='DiabetesStats + pandas', sample=self.stats.n)
            state.facts.append(fact)
        from src.chapter04.stats import fmt_pct, fmt_int, fmt_num
        lines = [f'{f["id"]}: {f["label_vi"]} = {fmt_pct(f["value"], "vi") if f["unit"] == "%" else fmt_num(f["value"], "vi") if f["unit"] == "BMI" else fmt_int(f["value"], "vi")}'
                 + (f' ({fmt_int(f["numerator"], "vi")}/{fmt_int(f["denominator"], "vi")})' if f.get('numerator') is not None and f['unit'] == '%' else '') for f in state.facts]
        state.send('stats', 'calculator' if state.plan['calc'] else 'writer', 'facts', ids=[f['id'] for f in state.facts], values=[f['value'] for f in state.facts])
        state.steps.append({'agent': 'StatsAgent', 'role': self.role, 'role_description': self.description,
                            'summary': f'Agent thống kê đọc dữ liệu CDC BRFSS 2023 ({fmt_int(self.stats.n, "vi")} dòng) và tính {len(state.facts)} số liệu bằng pandas: ' + '; '.join(lines) + '.',
                            'outcome': f'{len(state.facts)} sự kiện', 'seconds': time.perf_counter() - t0, 'evidence': {'facts': state.facts}})


# ---------------------------------------------------------------------------------------------
# Agent tính toán
# ---------------------------------------------------------------------------------------------
class CalculatorAgent:
    role = 'Agent tính toán'
    description = 'Tính tỉ số, chênh lệch từ các sự kiện bằng safe_calculate, không để mô hình ngôn ngữ tự tính.'

    def act(self, state):
        from src.chapter04.knowledge import safe_calculate
        from src.chapter04.stats import fmt_num
        t0 = time.perf_counter()
        calc = state.plan['calc']
        by_id = {f['id']: f for f in state.facts}
        a, b = by_id[calc['a']], by_id[calc['b']]
        op = calc['op']
        va, vb = a['value'], b['value']
        if op == 'subtract' and va < vb:
            a, b, va, vb = b, a, vb, va
        res = safe_calculate(op, [va, vb], [a['unit'], b['unit']])
        value = res['value']
        unit_vi = 'điểm phần trăm' if (op == 'subtract' and a['unit'] == '%') else ('' if op == 'divide' else 'điểm BMI')
        if op == 'divide':
            sentence_vi = f'Tỉ lệ ở {a["short_vi"]} gấp {fmt_num(value, "vi", 2)} lần ở {b["short_vi"]}.' if a['unit'] == '%' else f'BMI trung bình của {a["short_vi"]} gấp {fmt_num(value, "vi", 2)} lần của {b["short_vi"]}.'
            sentence_en = f'The value for {a["short_en"]} is {fmt_num(value, "en", 2)} times the value for {b["short_en"]}.'
        else:
            sentence_vi = (f'{a["short_vi"][:1].upper() + a["short_vi"][1:]} cao hơn {b["short_vi"]} {fmt_num(value, "vi")} {unit_vi}.').replace('  ', ' ')
            sentence_en = f'{a["short_en"][:1].upper() + a["short_en"][1:]} is higher than {b["short_en"]} by {fmt_num(value, "en")} ' + ('percentage points.' if a['unit'] == '%' else 'BMI points.')
        digits = 2 if op == 'divide' else 1
        derived = {'id': calc['id'], 'metric': op, 'value': round(value, digits + 2), 'unit': 'lần' if op == 'divide' else unit_vi, 'formula': f'{a["id"]} {"/" if op == "divide" else "-"} {b["id"]} = {res["formula"]}',
                   'label_vi': 'Tỉ số' if op == 'divide' else 'Chênh lệch', 'label_en': 'Ratio' if op == 'divide' else 'Difference', 'sentence_vi': sentence_vi, 'sentence_en': sentence_en,
                   'inputs': [a['id'], b['id']], 'digits': digits, 'source': 'safe_calculate (src/chapter04/knowledge.py)', 'tool': 'safe_calculate'}
        state.derived.append(derived)
        state.send('calculator', 'writer', 'derived', id=derived['id'], value=derived['value'], formula=derived['formula'])
        state.steps.append({'agent': 'CalculatorAgent', 'role': self.role, 'role_description': self.description,
                            'summary': f'Agent tính toán lấy {a["id"]} = {fmt_num(va, "vi", 2)} và {b["id"]} = {fmt_num(vb, "vi", 2)} từ agent thống kê rồi gọi safe_calculate: {derived["formula"]}. '
                                       f'Kết quả {fmt_num(value, "vi", derived["digits"])}{(" " + derived["unit"]) if op != "divide" else " lần"}. Mô hình ngôn ngữ không được tự làm phép tính này.',
                            'outcome': f'{derived["label_vi"]} = {fmt_num(value, "vi", derived["digits"])}', 'seconds': time.perf_counter() - t0, 'evidence': {'derived': derived}})


# ---------------------------------------------------------------------------------------------
# Agent truy hồi và agent đồ thị
# ---------------------------------------------------------------------------------------------
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', re.sub(r'\s+', ' ', str(text))) if len(s.strip()) > 25]


class RetrievalAgent:
    role = 'Agent truy hồi'
    description = 'Tìm đoạn tài liệu liên quan (NIDDK, CDC, MedlinePlus) và rút ra câu giải thích ngắn kèm mã nguồn.'

    def __init__(self, engine):
        self.engine = engine

    def act(self, state):
        from src.chapter04.rag import content_tokens, lexical_support
        t0 = time.perf_counter()
        question = state.plan.get('explain_question') or state.question
        if hasattr(self.engine, 'to_english_query'):
            state.query, language = self.engine.to_english_query(question)
        else:
            state.query, language = question, 'en'
        state.language = state.language or language
        found = self.engine.retrieve(state.query, state.domain, k=getattr(self.engine, 'K_EVIDENCE', 3), report=state.report)
        supported = bool(found) and lexical_support(state.query, found)
        state.evidence = found if supported else []
        terms = set(content_tokens(state.query))
        scored = []
        for e in state.evidence:
            for s in split_sentences(e['text']):
                words = set(content_tokens(s))
                scored.append((len(terms & words) / (len(words) ** 0.5 + 1), s, e))
        scored.sort(key=lambda x: -x[0])
        state.explanation = [{'id': e['id'], 'sentence': s, 'score': round(sc, 3), 'url': e.get('source_url'), 'org': e.get('source_org', '')} for sc, s, e in scored[:2] if sc > 0]
        state.send('retrieval', 'writer', 'passages', ids=[e['id'] for e in state.evidence], explanation=len(state.explanation))
        if not state.evidence:
            summary = (f'Agent truy hồi dịch phần giải thích của câu hỏi thành truy vấn "{state.query}" và lấy các đoạn gần nhất, nhưng không đoạn nào chứa từ khóa của truy vấn. '
                       'Nó báo cho nhóm biết kho tài liệu có lẽ không nói về chủ đề này.')
            outcome = 'không tìm thấy tài liệu phù hợp'
        else:
            top = state.evidence[0]
            summary = (f'Agent truy hồi dịch phần giải thích của câu hỏi thành truy vấn "{state.query}" và tìm được {len(state.evidence)} đoạn từ kho tài liệu. '
                       f'Câu giải thích tốt nhất (nguồn {top.get("source_org", "")} {top["id"]}): «{state.explanation[0]["sentence"] if state.explanation else top["text"][:160]}»')
            outcome = f'{len(state.evidence)} đoạn'
        state.steps.append({'agent': 'RetrievalAgent', 'role': self.role, 'role_description': self.description, 'summary': summary, 'outcome': outcome,
                            'seconds': time.perf_counter() - t0, 'evidence': {'query': state.query, 'passages': [{'id': e['id'], 'title': e.get('title', ''), 'url': e.get('source_url'), 'text': str(e['text'])[:300]} for e in state.evidence],
                                                                          'explanation': state.explanation}})


class GraphAgent:
    role = 'Agent đồ thị'
    description = 'Tra đồ thị tri thức để biết mỗi đoạn thuộc báo cáo hay công ty nào.'

    def __init__(self, graphs):
        self.graphs = graphs

    def act(self, state):
        t0 = time.perf_counter()
        graph = self.graphs.get(state.domain)
        labels = []
        if graph is not None:
            for item in state.evidence:
                node = 'passage:' + item['id']
                if node in graph:
                    parents = list(graph.predecessors(node))
                    state.graph_context.append({'id': item['id'], 'parents': parents})
                    for p in parents:
                        data = graph.nodes[p]
                        label = (f"báo cáo {p.split(':')[1].split('/')[0]} năm {data.get('year')}" if data.get('type') == 'Report' else str(p))
                        if label not in labels:
                            labels.append(label)
        state.send('graph', 'writer', 'sources', matched=len(state.graph_context))
        summary = (f'Agent đồ thị tra {len(state.graph_context)} đoạn trong đồ thị tri thức và biết chúng thuộc: {"; ".join(labels)}.' if state.graph_context
                   else 'Agent đồ thị không tìm thấy các đoạn này trong đồ thị nên chuyển tiếp nguyên các đoạn.')
        state.steps.append({'agent': 'GraphAgent', 'role': self.role, 'role_description': self.description, 'summary': summary, 'outcome': f'{len(state.graph_context)} đoạn khớp',
                            'seconds': time.perf_counter() - t0, 'evidence': {'graph': state.graph_context, 'labels': labels}})


# ---------------------------------------------------------------------------------------------
# Agent viết
# ---------------------------------------------------------------------------------------------
class WriterAgent:
    role = 'Agent viết'
    description = 'Viết câu trả lời từ sự kiện của công cụ và các đoạn tài liệu (bản nháp đầu do mô hình Qwen viết tự do).'

    SYSTEM = ('You are the writer in a team that answers health-data questions for a lay reader. You receive NUMBERS from a statistics tool and PASSAGES from health websites. '
              'Write 3 plain sentences in English that answer the question. Use the tool numbers exactly as written and say they come from the survey sample. '
              'Use the passages only for explanations; if you mention a number from a passage, say it comes from the documents. Do not invent numbers.')

    SYSTEM_PLAIN = ('You answer a health question using ONLY the passages provided. Write 2 to 3 plain sentences in English. '
                    'Do not use outside knowledge and do not invent numbers; if you mention a number, say it comes from the documents. '
                    'If the passages do not contain the answer, reply with exactly the single word INSUFFICIENT.')

    def __init__(self, generator):
        self.generator = generator

    # --- dựng đầu vào cho bộ sinh
    def facts_block(self, state):
        lines = [f'{f["id"]}. {f["sentence_en"]}' for f in state.facts] + [f'{d["id"]}. {d["sentence_en"]}' for d in state.derived]
        return '\n'.join(lines)

    def english_question(self, state):
        return state.query if state.plan.get('mode') == 'simple' and state.query else state.question

    def prompt(self, state, feedback=None):
        parts = []
        if state.facts:
            n = state.facts[0].get('sample', 150000)
            parts.append(f'NUMBERS FROM THE STATISTICS TOOL (CDC BRFSS 2023 survey sample of {n:,} adults, unweighted):\n' + self.facts_block(state))
        if state.evidence:
            parts.append('PASSAGES:\n' + '\n---\n'.join(str(e['text'])[:900] for e in state.evidence[:3]))
        q = state.question if state.facts else (state.query or state.question)
        parts.append(f'Question: {q}')
        if feedback:
            parts.append('Your previous answer:\n' + state.drafts[-1]['text'])
            parts.append('Your previous answer was reviewed and had these problems:\n' + '\n'.join('- ' + f for f in feedback)
                         + '\nRewrite the answer and fix every problem. Use the tool numbers exactly as written.')
        parts.append('Answer:')
        return '\n\n'.join(parts)

    def llm_draft(self, state, feedback=None):
        if not hasattr(self.generator, 'chat'):
            return None
        text = self.generator.chat(self.SYSTEM if state.facts else self.SYSTEM_PLAIN, self.prompt(state, feedback), max_new_tokens=170)
        return re.sub(r'\[\d+\]', '', text).strip()

    def rebuild(self, state):
        """Dựng lại bản nháp từ số của công cụ và câu giải thích (xác định, không dùng mô hình)."""
        parts = [f['sentence_en'] for f in state.facts] + [d['sentence_en'] for d in state.derived]
        if state.explanation:
            e = state.explanation[0]
            parts.append(f'According to the documents ({e["org"] or "source"}): {e["sentence"]}')
        if state.facts:
            parts.append(SAMPLE_SIZE_NOTE_EN.format(n=f'{state.facts[0].get("sample", 150000):,}'))
        return ' '.join(parts)

    def rebuild_vi(self, state, translate=True):
        from src.chapter04.stats import fmt_int
        parts = [f['sentence_vi'] for f in state.facts] + [d['sentence_vi'] for d in state.derived]
        if state.explanation:
            e = state.explanation[0]
            sentence = e['sentence']
            if translate and hasattr(self.generator, 'to_vietnamese'):
                sentence = self.generator.to_vietnamese(sentence)[0]
            parts.append(f'Theo tài liệu ({e["org"] or "nguồn"}): {sentence}')
        if state.facts:
            parts.append(SAMPLE_SIZE_NOTE_VI.format(n=fmt_int(state.facts[0].get('sample', 150000), 'vi')))
        return ' '.join(parts)

    def extractive(self, state):
        if not state.evidence:
            return ''
        if state.explanation:
            return ' '.join(f'According to the documents ({e["org"] or "source"}): {e["sentence"]}' for e in state.explanation[:2])
        return 'According to the documents: ' + split_sentences(state.evidence[0]['text'])[0]

    def act(self, state, round_no=0, feedback=None):
        t0 = time.perf_counter()
        from src.chapter04.rag import INSUFFICIENT_EN, INSUFFICIENT_VI
        vi = state.language == 'vi'
        if round_no == 0:
            if state.plan['mode'] == 'simple' and not state.evidence:
                state.status = 'insufficient_evidence'
                state.answer = INSUFFICIENT_VI if vi else INSUFFICIENT_EN
                state.steps.append({'agent': 'WriterAgent', 'role': self.role, 'role_description': self.description, 'seconds': time.perf_counter() - t0,
                                    'summary': 'Agent viết thấy agent truy hồi không tìm được đoạn nào chứa câu trả lời, nên không viết gì và báo lại cho điều phối.', 'outcome': 'từ chối viết', 'evidence': {}})
                state.send('writer', 'coordinator', 'refusal')
                return None
            text = self.llm_draft(state)
            by = 'Qwen viết tự do'
            if text and self.generator_refused(text) and not state.facts:
                state.status = 'insufficient_evidence'
                state.answer = INSUFFICIENT_VI if vi else INSUFFICIENT_EN
                state.steps.append({'agent': 'WriterAgent', 'role': self.role, 'role_description': self.description, 'seconds': time.perf_counter() - t0,
                                    'summary': 'Agent viết đọc các đoạn tìm được, thấy chúng không chứa câu trả lời nên từ chối viết thay vì bịa.', 'outcome': 'từ chối viết', 'evidence': {}})
                state.send('writer', 'coordinator', 'refusal')
                return None
            if not text or self.generator_refused(text):
                if state.facts or state.evidence:
                    text, by = self.rebuild(state), 'dựng từ sự kiện (bộ sinh không viết được)'
        elif round_no == 1:
            text = self.llm_draft(state, feedback)
            by = 'Qwen sửa theo phản biện'
            if not text:
                text, by = self.rebuild(state), 'dựng lại từ số của công cụ'
        else:
            text = self.rebuild(state) if (state.facts or state.explanation) else self.extractive(state)
            by = 'dựng lại từ số của công cụ' if state.facts else 'trích nguyên câu từ tài liệu'
        state.drafts.append({'round': round_no, 'by': by, 'text': text})
        state.send('writer', 'critic', 'draft', round=round_no, by=by, text=text)
        label = 'bản nháp' if round_no == 0 else f'bản sửa lần {round_no}'
        state.steps.append({'agent': 'WriterAgent', 'role': self.role, 'role_description': self.description, 'seconds': time.perf_counter() - t0,
                            'summary': f'Agent viết soạn {label} ({by}): «{text}»', 'outcome': label, 'evidence': {'draft': text, 'round': round_no}})
        return text

    @staticmethod
    def generator_refused(text):
        low = text.strip().lower()
        return low.startswith('insufficient') or 'not enough information' in low

    def finalize(self, state, text):
        """Dịch bản nháp đã được duyệt sang tiếng Việt khi câu hỏi bằng tiếng Việt. Trả về (văn bản, lỗi dịch)."""
        if state.language != 'vi':
            return text, []
        if not hasattr(self.generator, 'to_vietnamese'):
            return text, []
        return self.generator.to_vietnamese(text)


# ---------------------------------------------------------------------------------------------
# Agent phản biện: bộ kiểm tra chạy được, không phải mô hình ngôn ngữ
# ---------------------------------------------------------------------------------------------
class CriticAgent:
    role = 'Agent phản biện'
    description = 'Đối chiếu từng con số trong bản nháp với sự kiện của công cụ và với đoạn tài liệu; không dùng mô hình ngôn ngữ.'

    def __init__(self, corpora=None):
        self.allowed = {d: {x['id'] for x in c} for d, c in (corpora or {}).items()}

    def check(self, text, state, language=None):
        """Trả về dict: ok, issues, numbers (từng số kèm nguồn)."""
        from src.chapter04.stats import fmt_pct, fmt_num, fmt_int
        issues, numbers = [], []
        known = fact_numbers(state)
        passage_text = ' '.join(str(e['text']) for e in state.evidence)
        passage_nums = extract_numbers(passage_text)
        has_facts = bool(state.facts)
        for item in extract_numbers(text, ignore=[state.question]):
            hit = next(((v, fid, f) for v, fid, f in known if close(item['value'], item['decimals'], item['scale'], v)), None)
            if hit:
                numbers.append({**{k: item[k] for k in ('text', 'value')}, 'verdict': 'ok', 'source': f'công cụ {hit[1]}', 'sentence': item['sentence']})
                continue
            if item['value'] in (150000.0, 2023.0):
                numbers.append({'text': item['text'], 'value': item['value'], 'verdict': 'ok', 'source': 'cỡ mẫu / năm của nguồn dữ liệu', 'sentence': item['sentence']})
                continue
            doc = next((p for p in passage_nums if close(item['value'], item['decimals'], item['scale'], p['value']) or close(p['value'], p['decimals'], p['scale'], item['value'])), None)
            if doc:
                misused = bool(SAMPLE_MARK.search(item['sentence'])) and has_facts
                unattributed = has_facts and not DOC_MARK.search(item['sentence'])
                if misused or unattributed:
                    why = 'số này nằm trong tài liệu (một quần thể khác) nhưng được viết như số liệu của mẫu khảo sát' if misused else 'số này chỉ có trong tài liệu nhưng câu không nói là lấy từ tài liệu'
                    right = '; '.join(f'{f["id"]}: {f["sentence_vi"]}' for f in state.facts[:2])
                    issues.append({'type': 'number_from_document', 'number': item['text'], 'sentence': item['sentence'],
                                   'message': f'Số {item["text"]}: {why}. Bằng chứng đúng của công cụ: {right}', 'message_en': f'The number {item["text"]} comes from a document, not from the tool. Use the tool numbers.'})
                    numbers.append({'text': item['text'], 'value': item['value'], 'verdict': 'sai nguồn', 'source': 'tài liệu', 'sentence': item['sentence']})
                else:
                    numbers.append({'text': item['text'], 'value': item['value'], 'verdict': 'ok', 'source': 'đoạn tài liệu ' + (state.evidence[0]['id'] if state.evidence else ''), 'sentence': item['sentence']})
                continue
            right = '; '.join(f'{f["id"]}: {f["sentence_vi"]}' for f in state.facts[:3]) if has_facts else 'không có công cụ nào cung cấp số này'
            issues.append({'type': 'unsupported_number', 'number': item['text'], 'sentence': item['sentence'],
                           'message': f'Số {item["text"]} không có trong bằng chứng (không khớp công cụ hay đoạn tài liệu). Bằng chứng đúng: {right}',
                           'message_en': f'The number {item["text"]} is not in the evidence. Correct tool numbers: ' + ' '.join(f['sentence_en'] for f in state.facts[:3])})
            numbers.append({'text': item['text'], 'value': item['value'], 'verdict': 'không có bằng chứng', 'source': '-', 'sentence': item['sentence']})
        # số kết quả chính phải xuất hiện
        written = extract_numbers(text, ignore=[state.question])
        for f in state.facts + state.derived:
            digits = f.get('digits', 1)
            ok = any(close(w['value'], w['decimals'], w['scale'], f['value']) for w in written)
            if not ok:
                shown = fmt_pct(f['value'], 'vi') if f.get('unit') == '%' else fmt_num(f['value'], 'vi', digits)
                issues.append({'type': 'missing_headline', 'number': shown, 'sentence': '',
                               'message': f'Thiếu số kết quả chính {f["id"]} = {shown}. {f["sentence_vi"]}', 'message_en': f'Missing key number: {f["sentence_en"]}'})
        if has_facts and not LIMIT_MARK.search(text):
            issues.append({'type': 'missing_limit', 'number': '', 'sentence': '', 'message': 'Thiếu câu nói rõ đây là số liệu của mẫu khảo sát (không đại diện cho toàn dân).',
                           'message_en': 'Say that the numbers describe the survey sample and not the whole population.'})
        cited = [e['id'] for e in state.evidence]
        valid = all(c in self.allowed.get(state.domain, set()) for c in cited) if self.allowed else True
        if not valid:
            issues.append({'type': 'bad_citation', 'number': '', 'sentence': '', 'message': 'Có mã nguồn trích dẫn không tồn tại trong kho tài liệu.', 'message_en': 'A citation id is not in the corpus.'})
        return {'ok': not issues, 'issues': issues, 'numbers': numbers, 'citations_valid': valid, 'checked_numbers': len(numbers)}

    def act(self, state, text, round_no, language=None):
        t0 = time.perf_counter()
        result = self.check(text, state)
        result['round'] = round_no
        state.reviews.append(result)
        if result['ok']:
            state.send('critic', 'coordinator', 'approval', round=round_no, numbers=result['checked_numbers'])
            summary = (f'Agent phản biện trích {result["checked_numbers"]} con số trong bản nháp và đối chiếu từng số với công cụ và tài liệu: tất cả đều khớp; '
                       f'mã nguồn trích dẫn hợp lệ{"; câu trả lời có nhắc giới hạn của mẫu" if state.facts else ""}. Chấp nhận.')
            outcome = 'chấp nhận'
        else:
            state.send('critic', 'writer', 'revision_request', round=round_no, issues=[i['message'] for i in result['issues']])
            summary = (f'Agent phản biện đối chiếu bản nháp và tìm thấy {len(result["issues"])} lỗi: ' + ' | '.join(i['message'] for i in result['issues'][:4]) + '. Gửi yêu cầu sửa cho agent viết.')
            outcome = f'{len(result["issues"])} lỗi, yêu cầu sửa'
        state.steps.append({'agent': 'CriticAgent', 'role': self.role, 'role_description': self.description, 'summary': summary, 'outcome': outcome,
                            'seconds': time.perf_counter() - t0, 'evidence': {'review': result}})
        return result


# ---------------------------------------------------------------------------------------------
# Bộ điều phối: chạy kế hoạch, vòng phản biện, kết luận
# ---------------------------------------------------------------------------------------------
CONCLUSIONS = {
    'verified_numbers': ('Đã kiểm chứng số liệu bằng công cụ', 'Mọi con số trong câu trả lời đã được đối chiếu với kết quả của công cụ thống kê/tính toán (và với tài liệu cho phần giải thích). Phần lời giải thích chưa được kiểm tra về ngữ nghĩa.'),
    'generated_unverified': ('Đã viết, chưa kiểm chứng nội dung', 'Câu hỏi chỉ cần tri thức từ tài liệu: các con số (nếu có) khớp đoạn tài liệu, nhưng hệ thống chưa kiểm tra được lời giải thích có đúng ý nguồn hay không.'),
    'warnings': ('Trả lời kèm cảnh báo', 'Sau hai vòng sửa, agent phản biện vẫn còn nghi ngờ một số điểm; câu trả lời được giữ lại kèm cảnh báo, hãy đối chiếu thủ công.'),
    'insufficient_evidence': ('Không đủ bằng chứng', 'Kho tài liệu và dữ liệu khảo sát không chứa câu trả lời, nên hệ thống từ chối thay vì bịa.'),
    'pending': ('Đang chờ', 'Chưa chạy xong.'),
    'error': ('Lỗi', 'Hệ thống gặp lỗi khi chạy.'),
}


class MultiAgentTeam:
    def __init__(self, engine, graphs, stats):
        self.engine = engine
        self.stats = stats
        self.planner = PlannerAgent(stats)
        self.stats_agent = StatsAgent(stats)
        self.calculator = CalculatorAgent()
        self.retrieval = RetrievalAgent(engine)
        self.graph = GraphAgent(graphs)
        self.writer = WriterAgent(engine.generator)
        self.critic = CriticAgent(engine.corpora)
        self.agents = [self.planner, self.stats_agent, self.calculator, self.retrieval, self.graph, self.writer, self.critic]

    def ask(self, question, domain, report=None, emit=None):
        if domain not in ('diabetes', 'finance'):
            raise ValueError('Unsupported domain')
        if not question.strip():
            raise ValueError('Empty question')
        state = State(question, domain, report)
        state.language = 'vi' if re.search('[à-ỹ]', question.lower()) or re.search(r'\b(bao nhieu|ti le|tieu duong)\b', norm(question)) else 'en'
        seen = [0]

        def flush():
            while emit and seen[0] < len(state.steps):
                emit(state.steps[seen[0]], state)
                seen[0] += 1
        self.planner.act(state)
        flush()
        agents = state.plan['agents']
        if state.plan['scope'] == 'off_topic':
            self.conclude(state, 'insufficient_evidence', refuse=True)
            flush()
            return state
        if 'StatsAgent' in agents:
            self.stats_agent.act(state); flush()
        if 'CalculatorAgent' in agents:
            self.calculator.act(state); flush()
        if 'RetrievalAgent' in agents:
            self.retrieval.act(state); flush()
        if 'GraphAgent' in agents:
            self.graph.act(state); flush()
        if state.plan['mode'] == 'simple' and not state.evidence:
            self.writer.act(state); flush()
            self.conclude(state, 'insufficient_evidence', refuse=True)
            flush()
            return state
        if state.plan['mode'] == 'data' and not state.facts:
            self.conclude(state, 'insufficient_evidence', refuse=True)
            flush()
            return state
        text = self.writer.act(state, 0); flush()
        if text is None:
            self.conclude(state, 'insufficient_evidence', refuse=True)
            flush()
            return state
        review = self.critic.act(state, text, 0); flush()
        revisions = 0
        while not review['ok'] and revisions < MAX_REVISIONS:
            revisions += 1
            feedback = [i['message_en'] for i in review['issues']]
            state.dialogue.append({'round': revisions, 'issues': review['issues']})
            text = self.writer.act(state, revisions, feedback); flush()
            review = self.critic.act(state, text, revisions); flush()
        state.revisions = revisions
        # dịch sang tiếng Việt rồi soát lại bản dịch bằng chính bộ phản biện
        t0 = time.perf_counter()
        final, problems = self.writer.finalize(state, text)
        state.english_answer = text if state.language == 'vi' else ''
        note = ''
        if state.language == 'vi':
            check = self.critic.check(final, state)
            if not check['ok'] and state.facts:
                final = self.writer.rebuild_vi(state)
                note = 'Bản dịch làm lệch số liệu nên agent viết dựng lại bản tiếng Việt trực tiếp từ số của công cụ.'
                check = self.critic.check(final, state)
            elif not check['ok'] and not state.facts and state.evidence:
                note = 'Bản dịch có số không khớp tài liệu; giữ nguyên văn tiếng Anh đã được duyệt.'
                final = text
                check = {'ok': True}
            review = check if not check['ok'] else review
            state.steps.append({'agent': 'WriterAgent', 'role': self.writer.role, 'role_description': self.writer.description, 'seconds': time.perf_counter() - t0,
                                'summary': 'Agent viết dịch bản đã được duyệt sang tiếng Việt, rồi agent phản biện kiểm lại các con số của bản dịch. ' + (note or 'Các con số vẫn khớp.'),
                                'outcome': 'bản tiếng Việt', 'evidence': {'answer': final}})
            flush()
        state.answer = final
        passed = review['ok'] if isinstance(review, dict) else True
        status = ('verified_numbers' if state.facts else 'generated_unverified') if passed else 'warnings'
        self.conclude(state, status)
        flush()
        return state

    def conclude(self, state, status, refuse=False):
        from src.chapter04.rag import INSUFFICIENT_EN, INSUFFICIENT_VI
        state.status = status
        if refuse:
            if state.plan.get('scope') == 'off_topic':
                state.answer = ('Câu hỏi này nằm ngoài phạm vi tiểu đường và tài chính của hệ thống, nên mình không trả lời.' if state.language == 'vi'
                                else 'This question is outside the diabetes and finance scope of the system, so I will not answer it.')
            elif not state.answer:
                state.answer = INSUFFICIENT_VI if state.language == 'vi' else INSUFFICIENT_EN
        revisions = getattr(state, 'revisions', 0)
        label, explain = CONCLUSIONS[status]
        if revisions and status in ('verified_numbers', 'generated_unverified'):
            explain += f' Bản nháp đầu bị phản biện yêu cầu sửa; câu trả lời cuối là bản sau {revisions} vòng sửa.'
        state.send('coordinator', 'user', 'verdict', status=status, label=label, revisions=revisions)
        state.steps.append({'agent': 'Coordinator', 'role': 'Bộ điều phối', 'role_description': 'Tổng hợp trạng thái cuối bằng lời.', 'summary': f'{label}. {explain}',
                            'outcome': label, 'seconds': 0.0, 'evidence': {'status': status, 'revisions': revisions}})
        state.revisions = revisions


def build_dialogue(state):
    """Cuộc trao đổi viết - phản biện, theo thứ tự thời gian, để giao diện hiển thị như hội thoại."""
    out = []
    for d in state.drafts:
        out.append({'speaker': 'Agent viết', 'kind': 'draft' if d['round'] == 0 else 'revision', 'round': d['round'], 'by': d['by'],
                    'text': d['text']})
        review = next((r for r in state.reviews if r['round'] == d['round']), None)
        if review:
            if review['ok']:
                out.append({'speaker': 'Agent phản biện', 'kind': 'approval', 'round': d['round'], 'ok': True,
                            'text': f'Đã đối chiếu {review["checked_numbers"]} con số với công cụ và tài liệu: tất cả khớp. Chấp nhận.',
                            'issues': []})
            else:
                out.append({'speaker': 'Agent phản biện', 'kind': 'objection', 'round': d['round'], 'ok': False,
                            'text': f'Phát hiện {len(review["issues"])} lỗi trong bản này.', 'issues': [i['message'] for i in review['issues']]})
    return out


def compare_baseline(text, state):
    """So câu trả lời của RAG đơn với các số do công cụ tính: có đủ số đúng không, có số lạ không."""
    heads = state.facts + state.derived
    if not heads:
        return {'comparable': False}
    written = extract_numbers(text, ignore=[state.question])
    matched, missing = 0, []
    for f in heads:
        if any(close(w['value'], w['decimals'], w['scale'], f['value']) for w in written):
            matched += 1
        else:
            missing.append(f['sentence_vi'])
    foreign = [w['text'] for w in written if not any(close(w['value'], w['decimals'], w['scale'], v) for v, _, _ in fact_numbers(state)) and w['value'] not in (150000.0, 2023.0)]
    return {'comparable': True, 'matched': matched, 'total': len(heads), 'has_all_numbers': matched == len(heads), 'missing': missing, 'foreign_numbers': foreign}
