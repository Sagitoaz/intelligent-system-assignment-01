"""Thử hàng loạt câu hỏi tự nhiên trên server Chương 5/7 đang chạy và đo tỉ lệ được trả lời.

Khởi động server trước:
    .venv\\Scripts\\python.exe -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8045
Chạy:
    .venv\\Scripts\\python.exe scripts/chapter05/probe_questions.py [--endpoint agents|ask] [--only TEXT] [--limit N]
Ghi results/chapter05/probe_questions.json và in bảng Markdown. Cột `expect` là kỳ vọng của người viết: "answer" nếu chủ đề có
trong kho, "refuse" nếu câu nằm ngoài kho (phải bị từ chối). Việc câu trả lời có đúng nội dung hay không phải đọc tay, đối chiếu đoạn nguồn.
"""
from pathlib import Path
import argparse, json, re, sys, time
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.chapter04.textguard import foreign_script  # noqa: E402
sys.stdout.reconfigure(encoding='utf-8')

# (lĩnh vực, câu hỏi, kỳ vọng)
QUESTIONS = [
    ('diabetes', 'Triệu chứng của bệnh tiểu đường là gì?', 'answer'),
    ('diabetes', 'Biến chứng của tiểu đường là gì?', 'answer'),
    ('diabetes', 'Người bị tiểu đường nên ăn gì?', 'answer'),
    ('diabetes', 'Tiểu đường có chữa khỏi được không?', 'answer'),
    ('diabetes', 'Metformin dùng để làm gì?', 'answer'),
    ('diabetes', 'Tại sao bị tiểu đường?', 'answer'),
    ('diabetes', 'Tập thể dục có giúp gì không?', 'answer'),
    ('diabetes', 'Đường huyết bình thường là bao nhiêu?', 'answer'),
    ('diabetes', 'Ai có nguy cơ mắc tiểu đường?', 'answer'),
    ('diabetes', 'Nguyên nhân gây tiểu đường típ 2 là gì?', 'answer'),
    ('diabetes', 'Hạ đường huyết là gì và xử lý thế nào?', 'answer'),
    ('diabetes', 'Xét nghiệm A1C cho biết điều gì?', 'answer'),
    ('diabetes', 'Tiểu đường thai kỳ là gì?', 'answer'),
    ('diabetes', 'Biến chứng ở mắt do tiểu đường là gì?', 'answer'),
    ('diabetes', 'Làm sao để phòng ngừa tiểu đường típ 2?', 'answer'),
    ('diabetes', 'Tiền tiểu đường là gì?', 'answer'),
    ('diabetes', 'Người bị tiểu đường cần chăm sóc bàn chân như thế nào?', 'answer'),
    ('diabetes', 'Insulin được dùng như thế nào?', 'answer'),
    ('diabetes', 'Tiểu đường típ 1 khác típ 2 chỗ nào?', 'answer'),
    ('diabetes', 'Bệnh tiểu đường ảnh hưởng đến thận như thế nào?', 'answer'),
    ('diabetes', 'Nên tập thể dục bao nhiêu phút mỗi tuần?', 'answer'),
    ('diabetes', 'Tổn thương thần kinh do tiểu đường là gì?', 'answer'),
    ('diabetes', 'What are the symptoms of diabetes?', 'answer'),
    ('diabetes', 'What is the A1C test?', 'answer'),
    ('diabetes', 'What causes type 1 diabetes?', 'answer'),
    ('diabetes', 'How does diabetes affect the heart?', 'answer'),
    ('diabetes', 'What should I do if my blood sugar is too low?', 'answer'),
    ('diabetes', 'What is continuous glucose monitoring?', 'answer'),
    ('diabetes', 'How much exercise do people with diabetes need?', 'answer'),
    ('diabetes', 'Which medicines treat type 2 diabetes?', 'answer'),
    ('finance', 'Những mạng thanh toán điện tử lớn nhất là gì?', 'answer'),
    ('finance', 'Công ty Visa hoạt động trong lĩnh vực nào?', 'answer'),
    ('diabetes', 'Giá vàng hôm nay là bao nhiêu?', 'refuse'),
    ('diabetes', 'Thời tiết Hà Nội hôm nay thế nào?', 'refuse'),
    ('diabetes', 'Hướng dẫn nấu phở bò ngon', 'refuse'),
    ('diabetes', 'Who won the football World Cup in 2022?', 'refuse'),
    ('finance', 'Giá cổ phiếu Tesla hôm nay là bao nhiêu?', 'refuse'),
]
BAD_UNITS = re.compile(r'(?i)miligam\s*/|mili\s?gam\s*trên|mg\s*/\s*lượng')


def run(base, endpoint, domain, question):
    t0 = time.perf_counter()
    if endpoint == 'agents':
        r = requests.post(f'{base}/api/v1/agents/run', json={'domain': domain, 'question': question}, timeout=900).json()
        ids = [e['id'] for e in r.get('evidence', [])]
    else:
        r = requests.post(f'{base}/api/v1/knowledge/ask', json={'domain': domain, 'question': question}, timeout=900).json()
        ids = [c['id'] for c in r.get('citations', [])] or [e['id'] for e in r.get('evidence', [])]
    return {'domain': domain, 'question': question, 'status': r.get('status'), 'answer': r.get('answer', ''), 'english_answer': r.get('english_answer'),
            'query_used': r.get('query_used'), 'evidence_ids': ids, 'seconds': round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8045')
    ap.add_argument('--endpoint', choices=['agents', 'ask'], default='ask')
    ap.add_argument('--only', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'results/chapter05/probe_questions.json'))
    args = ap.parse_args()
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            st = requests.get(f'{args.base}/api/v1/status', timeout=10).json()
            if st.get('stage') == 'ready':
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    rows = []
    items = [q for q in QUESTIONS if args.only.lower() in q[1].lower()]
    for domain, question, expect in items[:args.limit or None]:
        row = run(args.base, args.endpoint, domain, question)
        row['expect'] = expect
        row['answered'] = row['status'] == 'generated_unverified'
        row['foreign_script'] = ''.join(foreign_script(row['answer']))
        row['bad_unit'] = bool(BAD_UNITS.search(row['answer']))
        row['ok'] = (row['answered'] == (expect == 'answer')) and not row['foreign_script'] and not row['bad_unit']
        rows.append(row)
        print(f"[{row['seconds']:5.1f}s] {row['status']:20s} {question} -> {row['answer'][:110]!r}", flush=True)
    inside = [r for r in rows if r['expect'] == 'answer']
    outside = [r for r in rows if r['expect'] == 'refuse']
    summary = {'endpoint': args.endpoint, 'questions': len(rows), 'in_scope': len(inside), 'answered_in_scope': sum(r['answered'] for r in inside),
               'answer_rate': sum(r['answered'] for r in inside) / max(1, len(inside)), 'out_of_scope': len(outside),
               'refused_out_of_scope': sum(not r['answered'] for r in outside), 'foreign_script_answers': sum(bool(r['foreign_script']) for r in rows),
               'bad_unit_answers': sum(r['bad_unit'] for r in rows), 'mean_seconds': sum(r['seconds'] for r in rows) / max(1, len(rows)),
               'max_seconds': max((r['seconds'] for r in rows), default=0)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
