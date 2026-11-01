"""Đánh giá định lượng: một RAG đơn so với nhóm agent (Chương 7) trên bộ câu hỏi có đáp án tính bằng pandas.

Chạy server trước (cổng 8044, xem RUN.md), rồi:
    python scripts/chapter05/eval_multiagent.py            # chạy đủ, ghi dần vào results/chapter05/multiagent_eval.json
    python scripts/chapter05/eval_multiagent.py --resume   # bỏ qua các câu đã có kết quả
    python scripts/chapter05/eval_multiagent.py --only C01 C02 S01
    python scripts/chapter05/eval_multiagent.py --report   # chỉ in bảng từ tệp kết quả

Tiêu chí đúng cho câu số liệu: số kết quả chính (tỉ số, chênh lệch hoặc giá trị đơn) xuất hiện trong câu trả lời,
sai số không quá 0,5 điểm phần trăm (với tỉ lệ), 1% tương đối (tỉ số, đếm) hoặc 0,1 (BMI). Đáp án được tính độc lập bằng pandas
từ data/chapter05/brfss2023_diabetes_150k.csv, không dùng mã của DiabetesStats.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tieuluan.complete_agents import extract_numbers  # noqa: E402

CSV = ROOT / 'data/chapter05/brfss2023_diabetes_150k.csv'
OUT = ROOT / 'results/chapter05/multiagent_eval.json'
BASE = 'http://127.0.0.1:8044'
TOL = {'pct': 0.5, 'bmi': 0.1}
BANDS_65 = ['65-69', '70-74', '75-79', '80+']
BANDS_U40 = ['18-24', '25-29', '30-34', '35-39']


def expected_table():
    """Trả về danh sách câu hỏi: id, loại, câu hỏi, đáp án số (danh sách (giá trị, kiểu, mô tả)) hoặc từ khóa."""
    df = pd.read_csv(CSV)
    d = (df.diabetes_status == 'yes')
    pct = lambda mask: float(d[mask].mean() * 100)
    overall = float(d.mean() * 100)
    cond = {'hbp': df.high_blood_pressure == 1, 'overweight': df.bmi_category == 'overweight', 'heart': df.heart_disease == 1, 'obese': df.bmi_category == 'obese',
            'inactive': df.physical_activity == 0, 'active': df.physical_activity == 1, 'daily': df.smoker_status == 'daily', 'never': df.smoker_status == 'never',
            'poor': df.general_health >= 4, 'male': df.sex == 'male', 'female': df.sex == 'female', 'old': df.age_group.isin(BANDS_65), 'young': df.age_group.isin(BANDS_U40)}

    def ratio_vs_all(key):
        a = pct(cond[key])
        return [(a / overall, 'ratio', 'tỉ số'), (a, 'pct', 'tỉ lệ nhóm'), (overall, 'pct', 'tỉ lệ chung')]

    def ratio_two(ka, kb):
        a, b = pct(cond[ka]), pct(cond[kb])
        return [(a / b, 'ratio', 'tỉ số'), (a, 'pct', 'tỉ lệ nhóm 1'), (b, 'pct', 'tỉ lệ nhóm 2')]

    def diff_two(ka, kb):
        a, b = pct(cond[ka]), pct(cond[kb])
        return [(abs(a - b), 'pct', 'chênh lệch'), (max(a, b), 'pct', 'nhóm cao'), (min(a, b), 'pct', 'nhóm thấp')]

    def extreme(col, min_n=200):
        g = d.groupby(df[col]).agg(['mean', 'size'])
        g = g[g['size'] >= min_n]
        top, low = g['mean'].max() * 100, g['mean'].min() * 100
        return [(top / low, 'ratio', 'tỉ số'), (top, 'pct', 'nhóm cao nhất'), (low, 'pct', 'nhóm thấp nhất')]

    bmi_y, bmi_n = float(df.bmi[d].mean()), float(df.bmi[df.diabetes_status == 'no'].mean())
    obese_m, obese_f = pct(cond['obese'] & cond['male']), pct(cond['obese'] & cond['female'])
    Q = [
        ('C01', 'phức hợp', 'Tỉ lệ tiểu đường ở người huyết áp cao gấp mấy lần cả mẫu, và vì sao huyết áp cao làm tăng nguy cơ?', ratio_vs_all('hbp')),
        ('C02', 'phức hợp', 'Chênh lệch tỉ lệ tiểu đường giữa nam và nữ là bao nhiêu điểm phần trăm?', diff_two('male', 'female')),
        ('C03', 'phức hợp', 'BMI trung bình của nhóm mắc tiểu đường cao hơn nhóm không mắc bao nhiêu?',
         [(bmi_y - bmi_n, 'bmi', 'chênh lệch BMI'), (bmi_y, 'bmi', 'BMI nhóm mắc'), (bmi_n, 'bmi', 'BMI nhóm không mắc')]),
        ('C04', 'phức hợp', 'Nhóm tuổi nào có tỉ lệ tiểu đường cao nhất và gấp mấy lần nhóm thấp nhất?', extreme('age_group')),
        ('C05', 'phức hợp', 'Tỉ lệ tiểu đường ở người béo phì gấp mấy lần cả mẫu, và vì sao béo phì làm tăng nguy cơ?', ratio_vs_all('obese')),
        ('C06', 'phức hợp', 'Người không vận động hơn người có vận động bao nhiêu điểm phần trăm về tỉ lệ tiểu đường, và vì sao vận động giúp ích?', diff_two('inactive', 'active')),
        ('C07', 'phức hợp', 'Người hút thuốc hằng ngày có tỉ lệ tiểu đường gấp mấy lần người chưa bao giờ hút thuốc?', ratio_two('daily', 'never')),
        ('C08', 'phức hợp', 'Tỉ lệ tiểu đường ở người trên 65 tuổi là bao nhiêu?', [(pct(cond['old']), 'pct', 'tỉ lệ')]),
        ('C09', 'phức hợp', 'Có bao nhiêu người từng bị đột quỵ trong dữ liệu khảo sát?', [(float((df.stroke == 1).sum()), 'count', 'số người')]),
        ('C10', 'phức hợp', 'Tỉ lệ tiểu đường ở người thừa cân gấp mấy lần cả mẫu?', ratio_vs_all('overweight')),
        ('C11', 'phức hợp', 'Tỉ lệ tiểu đường ở người có bệnh tim gấp mấy lần cả mẫu, và vì sao tiểu đường liên quan đến bệnh tim?', ratio_vs_all('heart')),
        ('C12', 'phức hợp', 'Tỉ lệ tiểu đường ở nam gấp mấy lần ở nữ?', ratio_two('male', 'female')),
        ('C13', 'phức hợp', 'BMI trung bình của người bị huyết áp cao là bao nhiêu?', [(float(df.bmi[cond['hbp']].mean()), 'bmi', 'BMI')]),
        ('C14', 'phức hợp', 'Người tự đánh giá sức khỏe kém có tỉ lệ tiểu đường gấp mấy lần cả mẫu?', ratio_vs_all('poor')),
        ('C15', 'phức hợp', 'Tỉ lệ tiểu đường ở người dưới 40 tuổi là bao nhiêu?', [(pct(cond['young']), 'pct', 'tỉ lệ')]),
        ('C16', 'phức hợp', 'Có bao nhiêu người được chẩn đoán tiểu đường trong dữ liệu khảo sát?', [(float(d.sum()), 'count', 'số người')]),
        ('C17', 'phức hợp', 'Nhóm thu nhập nào có tỉ lệ tiểu đường cao nhất và gấp mấy lần nhóm thấp nhất?', extreme('income_group')),
        ('C18', 'phức hợp', 'Nhóm học vấn nào có tỉ lệ tiểu đường cao nhất và gấp mấy lần nhóm thấp nhất?', extreme('education')),
        ('C19', 'phức hợp', 'BMI trung bình của cả mẫu khảo sát là bao nhiêu?', [(float(df.bmi.mean()), 'bmi', 'BMI')]),
        ('C20', 'phức hợp', 'Trong nhóm béo phì, chênh lệch tỉ lệ tiểu đường giữa nam và nữ là bao nhiêu điểm phần trăm?',
         [(abs(obese_m - obese_f), 'pct', 'chênh lệch'), (max(obese_m, obese_f), 'pct', 'nhóm cao'), (min(obese_m, obese_f), 'pct', 'nhóm thấp')]),
        ('C21', 'phức hợp', 'Tỉ lệ tiểu đường ở người béo phì cao hơn cả mẫu bao nhiêu điểm phần trăm?',
         [(pct(cond['obese']) - overall, 'pct', 'chênh lệch'), (pct(cond['obese']), 'pct', 'nhóm béo phì'), (overall, 'pct', 'cả mẫu')]),

        ('S01', 'đơn giản', 'Insulin do cơ quan nào trong cơ thể tạo ra?', ['pancrea', 'tuyến tụy', 'tụy']),
        ('S02', 'đơn giản', 'Xét nghiệm A1C cho biết điều gì?', ['average', 'trung bình', 'blood sugar', 'đường huyết']),
        ('S03', 'đơn giản', 'Các triệu chứng thường gặp của bệnh tiểu đường là gì?', ['thirst', 'khát', 'urinat', 'đi tiểu', 'tired', 'mệt']),
        ('S04', 'đơn giản', 'Tiểu đường thai kỳ là gì?', ['pregnan', 'thai', 'gestational']),
        ('S05', 'đơn giản', 'Hạ đường huyết là gì?', ['low blood', 'hạ đường', 'thấp', 'hypoglyc']),
        ('O01', 'ngoài phạm vi', 'Giá vàng hôm nay là bao nhiêu?', 'refuse'),
        ('O02', 'ngoài phạm vi', 'Thủ đô của nước Pháp là thành phố nào?', 'refuse'),
        ('O03', 'ngoài phạm vi', 'Đội nào vô địch World Cup 2022?', 'refuse'),
        ('O04', 'ngoài phạm vi', 'Hướng dẫn nấu món phở bò.', 'refuse'),
    ]
    return [q for q in Q if q[3] is not None]


def post(path, body, timeout=1500):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode('utf8'), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode('utf8'))


def numeric_ok(written, value, kind):
    for w in written:
        x = w['value']
        if kind == 'pct' and abs(x - value) <= TOL['pct']:
            return True
        if kind == 'bmi' and abs(x - value) <= TOL['bmi']:
            return True
        if kind in ('ratio', 'count') and abs(x - value) <= 0.01 * abs(value) + 1e-9:
            return True
        if kind == 'ratio' and round(value, 1) == round(x, 1):
            return True
    return False


def grade(text, spec, kind, status):
    """Trả về dict chấm một câu trả lời của một hệ."""
    refused = status == 'insufficient_evidence'
    if kind == 'ngoài phạm vi':
        return {'refused': refused, 'correct': refused}
    if kind == 'đơn giản':
        low = text.lower()
        hit = any(k in low for k in spec) and not refused
        return {'refused': refused, 'correct': hit}
    written = extract_numbers(text)
    flags = [numeric_ok(written, v, k) for v, k, _ in spec]
    return {'refused': refused, 'headline_correct': flags[0], 'all_correct': all(flags), 'numbers_found': [w['text'] for w in written], 'correct': flags[0]}


def run_one(item):
    qid, kind, question, spec = item
    r = post('/api/v1/agents/run', {'domain': 'diabetes', 'question': question, 'baseline': True})
    b = r.get('baseline') or {}
    team_text, base_text = r['answer'], b.get('answer', '')
    rounds = r.get('rounds') or []
    draft = rounds[0] if rounds else None
    team = grade(team_text, spec, kind, r['status'])
    base = grade(base_text, spec, kind, b.get('status', ''))
    team.update(seconds=r.get('team_seconds', r.get('seconds')), steps=r.get('step_count'), agents=r.get('agent_count'), revisions=r.get('revisions', 0), status=r['status'],
                has_source=bool(r.get('facts') or r.get('evidence')), answer=team_text)
    base.update(seconds=b.get('seconds'), steps=2, status=b.get('status'), has_source=bool(b.get('citations')), answer=base_text,
                foreign_numbers=b.get('foreign_numbers', []))
    if draft and kind == 'phức hợp':
        written = extract_numbers(draft['draft'])
        flags = [numeric_ok(written, v, k) for v, k, _ in spec]
        team['draft_headline_correct'] = flags[0]
        team['draft_all_correct'] = all(flags)
        team['draft_review_ok'] = bool(draft['review'] and draft['review']['ok'])
        team['draft_issues'] = [i['message'] for i in (draft['review'] or {}).get('issues', [])]
        team['draft_text'] = draft['draft']
    if draft and kind != 'phức hợp':
        team['draft_review_ok'] = bool(draft['review'] and draft['review']['ok'])
        team['draft_issues'] = [i['message'] for i in (draft['review'] or {}).get('issues', [])]
    return {'id': qid, 'kind': kind, 'question': question, 'expected': [(round(v, 3), k, n) for v, k, n in spec] if kind == 'phức hợp' else spec,
            'team': team, 'rag': base, 'plan_mode': r.get('mode')}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(results):
    out = {}
    cx = [r for r in results if r['kind'] == 'phức hợp']
    simple = [r for r in results if r['kind'] == 'đơn giản']
    off = [r for r in results if r['kind'] == 'ngoài phạm vi']
    frac = lambda xs: (sum(1 for x in xs if x), len(xs))
    for name in ('rag', 'team'):
        s = {}
        s['complex_headline_correct'] = frac([r[name].get('headline_correct') for r in cx])
        s['complex_all_numbers_correct'] = frac([r[name].get('all_correct') for r in cx])
        s['complex_wrongly_refused'] = frac([r[name]['refused'] for r in cx])
        s['complex_has_source'] = frac([r[name]['has_source'] for r in cx])
        s['simple_correct'] = frac([r[name]['correct'] for r in simple])
        s['simple_has_source'] = frac([r[name]['has_source'] for r in simple])
        s['off_scope_refused'] = frac([r[name]['refused'] for r in off])
        s['seconds_complex'] = mean([r[name]['seconds'] for r in cx])
        s['seconds_simple'] = mean([r[name]['seconds'] for r in simple])
        s['seconds_off_scope'] = mean([r[name]['seconds'] for r in off])
        s['steps_mean'] = mean([r[name]['steps'] for r in results])
        out[name] = s
    t = [r['team'] for r in cx if 'draft_headline_correct' in r['team']]
    out['critic'] = {
        'complex_questions': len(cx),
        'first_draft_rejected': sum(1 for x in t if not x['draft_review_ok']),
        'first_draft_headline_correct': frac([x['draft_headline_correct'] for x in t]),
        'first_draft_all_correct': frac([x['draft_all_correct'] for x in t]),
        'final_headline_correct': frac([x['headline_correct'] for x in t]),
        'final_all_correct': frac([x['all_correct'] for x in t]),
        'revisions_total': sum(x['revisions'] for x in t),
        'questions_with_revision': sum(1 for x in t if x['revisions']),
        'fixed_after_revision': sum(1 for x in t if x['revisions'] and x['status'] == 'verified_numbers'),
        'ended_with_warning': sum(1 for x in t if x['status'] == 'warnings'),
        'simple_first_draft_rejected': sum(1 for r in simple if r['team'].get('draft_review_ok') is False),
    }
    return out


def print_table(summary, n):
    r, t, c = summary['rag'], summary['team'], summary['critic']
    f = lambda p: f'{p[0]}/{p[1]} ({100 * p[0] / max(1, p[1]):.0f}%)'
    sec = lambda x: '-' if x is None else f'{x:.0f} s'
    rows = [('Số kết quả chính đúng (câu phức hợp)', f(r['complex_headline_correct']), f(t['complex_headline_correct'])),
            ('Mọi số đúng (câu phức hợp)', f(r['complex_all_numbers_correct']), f(t['complex_all_numbers_correct'])),
            ('Từ chối nhầm câu phức hợp', f(r['complex_wrongly_refused']), f(t['complex_wrongly_refused'])),
            ('Có nguồn (câu phức hợp)', f(r['complex_has_source']), f(t['complex_has_source'])),
            ('Câu đơn giản trả lời đúng', f(r['simple_correct']), f(t['simple_correct'])),
            ('Từ chối đúng câu ngoài phạm vi', f(r['off_scope_refused']), f(t['off_scope_refused'])),
            ('Thời gian TB, câu phức hợp', sec(r['seconds_complex']), sec(t['seconds_complex'])),
            ('Thời gian TB, câu đơn giản', sec(r['seconds_simple']), sec(t['seconds_simple'])),
            ('Số bước TB mỗi câu', f"{r['steps_mean']:.1f}", f"{t['steps_mean']:.1f}")]
    w = max(len(x[0]) for x in rows)
    print(f'\n{"Tiêu chí".ljust(w)} | {"Một RAG đơn".ljust(16)} | Nhóm agent')
    for a, b, d in rows:
        print(f'{a.ljust(w)} | {b.ljust(16)} | {d}')
    print('\nVai trò của agent phản biện (câu phức hợp):')
    print(f"  bản nháp đầu bị bác: {c['first_draft_rejected']}/{c['complex_questions']}; số kết quả chính đúng ở bản nháp đầu: {c['first_draft_headline_correct'][0]}/{c['first_draft_headline_correct'][1]}; "
          f"sau phản biện: {c['final_headline_correct'][0]}/{c['final_headline_correct'][1]}")
    print(f"  số vòng sửa: {c['revisions_total']} (trên {c['questions_with_revision']} câu), sửa xong và đã kiểm chứng: {c['fixed_after_revision']}, kết thúc kèm cảnh báo: {c['ended_with_warning']}")


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    table = expected_table()
    saved = json.loads(OUT.read_text(encoding='utf8')) if OUT.exists() and (args.resume or args.report or args.only) else {'results': []}
    results = saved.get('results', [])
    done = {r['id'] for r in results}
    if not args.report:
        for item in table:
            if args.only and item[0] not in args.only:
                continue
            if args.resume and item[0] in done:
                continue
            print('Chạy', item[0], item[2], flush=True)
            res = run_one(item)
            results = [r for r in results if r['id'] != item[0]] + [res]
            results.sort(key=lambda r: r['id'])
            payload = {'note': 'Một RAG đơn = RAGEngine.answer (Chương 5). Nhóm agent = MultiAgentTeam (Chương 7). Đáp án tính bằng pandas từ BRFSS 150.000 dòng.',
                       'csv': str(CSV.relative_to(ROOT)), 'questions': len(results), 'summary': summarize(results), 'results': results}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf8')
            print('   RAG đơn:', 'đúng' if res['rag']['correct'] else 'sai', f"{res['rag']['seconds']:.0f}s | nhóm agent:", 'đúng' if res['team']['correct'] else 'sai',
                  f"{res['team']['seconds']:.0f}s, {res['team']['revisions']} vòng sửa", flush=True)
    print_table(summarize(results), len(results))


if __name__ == '__main__':
    main()
