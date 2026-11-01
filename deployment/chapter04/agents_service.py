"""Dịch vụ Chương 7: trang /agents và các endpoint chạy nhóm agent (chặn một lần, hoặc theo tiến độ từng agent)."""
import threading
import time
import uuid
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.chapter04.knowledge import load_graph

HERE = Path(__file__).parent
ROOT = HERE.parents[1]


class TeamRequest(BaseModel):
    domain: Literal['diabetes', 'finance'] = 'diabetes'
    question: str = Field(default='', max_length=2000)
    report: str | None = None
    baseline: bool = True


AGENT_NAMES = {'PlannerAgent': 'Agent điều phối', 'StatsAgent': 'Agent thống kê', 'CalculatorAgent': 'Agent tính toán', 'RetrievalAgent': 'Agent truy hồi',
               'GraphAgent': 'Agent đồ thị', 'WriterAgent': 'Agent viết', 'CriticAgent': 'Agent phản biện', 'Coordinator': 'Bộ điều phối'}


def make_router(rag, agent):
    router = APIRouter()
    jobs = {}
    lock = threading.Lock()

    @lru_cache(maxsize=1)
    def team():
        from scripts.tieuluan.complete_agents import MultiAgentTeam
        graphs = {'finance': load_graph(ROOT / 'knowledge_graph/chapter04/finance_graph.json.gz'),
                  'diabetes': load_graph(ROOT / 'knowledge_graph/chapter04/diabetes_documents_graph.json.gz')}
        return MultiAgentTeam(rag(), graphs, agent().stats)

    @router.get('/agents', include_in_schema=False)
    def agents_page():
        return FileResponse(HERE / 'agents_ui.html')

    def step_json(step):
        out = {k: v for k, v in step.items()}
        out['role'] = step.get('role') or AGENT_NAMES.get(step['agent'], step['agent'])
        return out

    def run_core(request, emit=None):
        """Chạy nhóm agent và (tuỳ chọn) một RAG đơn; trả về dict JSON cho giao diện."""
        from scripts.tieuluan.complete_agents import CONCLUSIONS, build_dialogue, compare_baseline
        started = time.perf_counter()
        question = request.question.strip()
        if not question:
            label, explain = CONCLUSIONS['pending']
            return {'question': '', 'domain': request.domain, 'steps': [], 'answer': 'Bạn chưa nhập câu hỏi. Hãy gõ một câu hỏi hoặc bấm một câu mẫu.', 'status': 'pending',
                    'evidence': [], 'seconds': 0, 'status_label': label, 'status_explanation': explain, 'plan': {}, 'facts': [], 'rounds': [], 'baseline': None}
        crew = team()
        stream = (lambda step, state: emit({'type': 'step', 'step': step_json(step)})) if emit else None
        t0 = time.perf_counter()
        state = crew.ask(question, request.domain, request.report, emit=stream)
        team_seconds = time.perf_counter() - t0
        baseline = None
        if request.baseline:
            if emit:
                emit({'type': 'baseline_start'})
            b0 = time.perf_counter()
            try:
                result = rag().answer(question, request.domain, request.report)
                baseline = {'answer': result['answer'], 'status': result['status'], 'seconds': time.perf_counter() - b0, 'steps': 2,
                            'step_names': ['Truy hồi các đoạn', 'Mô hình viết câu trả lời từ các đoạn'], 'citations': [c['id'] for c in result.get('citations', [])],
                            'english_answer': result.get('english_answer'), **compare_baseline(result['answer'], state)}
            except Exception as exc:  # RAG đơn lỗi không làm hỏng kết quả nhóm agent
                baseline = {'answer': f'RAG đơn không chạy được: {exc}', 'status': 'error', 'seconds': time.perf_counter() - b0, 'steps': 2, 'comparable': False}
        label, explain = CONCLUSIONS.get(state.status, (state.status, ''))
        evidence = [{'id': e['id'], 'title': e.get('title', ''), 'url': e.get('source_url'), 'text': str(e.get('text', ''))[:320]} for e in state.evidence]
        rounds = []
        for d in state.drafts:
            review = next((r for r in state.reviews if r['round'] == d['round']), None)
            rounds.append({'round': d['round'], 'by': d['by'], 'draft': d['text'], 'review': review})
        simple = state.plan.get('mode') == 'simple'
        return {'question': question, 'domain': request.domain, 'report': request.report, 'steps': [step_json(s) for s in state.steps], 'answer': state.answer,
                'status': state.status, 'status_label': label, 'status_explanation': explain, 'query_used': state.query, 'english_answer': state.english_answer,
                'language': state.language, 'evidence': evidence, 'seconds': time.perf_counter() - started, 'team_seconds': team_seconds,
                'mode': state.plan.get('mode'), 'plan': state.plan, 'facts': state.facts, 'derived': state.derived, 'rounds': rounds, 'dialogue': build_dialogue(state),
                'revisions': getattr(state, 'revisions', 0), 'critic': state.reviews[-1] if state.reviews else None, 'baseline': baseline,
                'agent_count': len({s['agent'] for s in state.steps}), 'step_count': len(state.steps),
                'note': 'Câu này một RAG đơn là đủ: nhóm agent không có thêm số liệu hay phép tính nào để kiểm chứng, nên không hơn mà còn chậm hơn.' if simple else '',
                'messages': [asdict(m) for m in state.messages]}

    def safe_run(request, emit=None):
        try:
            return run_core(request, emit)
        except Exception as exc:
            return {'question': request.question, 'domain': request.domain, 'steps': [], 'answer': f'Không chạy được nhóm agent: {exc}', 'status': 'error',
                    'evidence': [], 'seconds': 0, 'status_label': 'Lỗi', 'status_explanation': 'Hệ thống gặp lỗi khi chạy; hãy thử lại sau khi mô hình nạp xong (xem /api/v1/status).',
                    'plan': {}, 'facts': [], 'rounds': [], 'baseline': None}

    @router.post('/api/v1/agents/run')
    def run_team(request: TeamRequest):
        return safe_run(request)

    @router.post('/api/v1/agents/start')
    def start_job(request: TeamRequest):
        job_id = uuid.uuid4().hex[:12]
        job = {'id': job_id, 'status': 'running', 'events': [], 'started': time.time(), 'result': None}
        with lock:
            for key in list(jobs)[:-15]:
                jobs.pop(key, None)
            jobs[job_id] = job

        def emit(event):
            job['events'].append({**event, 't': time.time() - job['started']})

        def work():
            job['result'] = safe_run(request, emit)
            job['status'] = 'done'
        threading.Thread(target=work, daemon=True).start()
        return {'job_id': job_id}

    @router.get('/api/v1/agents/job/{job_id}')
    def job_state(job_id: str, since: int = 0):
        job = jobs.get(job_id)
        if job is None:
            return {'status': 'unknown', 'events': [], 'next': 0, 'message': 'Không tìm thấy lượt chạy này (có thể server đã khởi động lại).'}
        events = job['events'][since:]
        return {'status': job['status'], 'events': events, 'next': since + len(events), 'elapsed': time.time() - job['started'],
                'result': job['result'] if job['status'] == 'done' else None}

    return router
