from functools import lru_cache
from pathlib import Path
from typing import Literal
import json,os,threading,time
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from src.chapter04.agent import ToolAgent,doc_title,highlight_terms,snippet,CONFIDENCE
from src.chapter04.knowledge import load_graph
from src.chapter04.nlu import find_company
from src.chapter04.rag import RAGEngine
from src.chapter04.stats import DiabetesStats,SAMPLE_QUESTIONS_VI,detect_language

ROOT=Path(__file__).resolve().parents[2]
app=FastAPI(title='Chương 5 và 7: Knowledge, RAG và hệ đa agent',version='2.0.0')

# Các câu tiếng Việt mẫu của giao diện: được dịch sẵn lúc khởi động để lần bấm đầu không phải chờ.
WARM_QUESTIONS=['Tiểu đường thai kỳ là gì?','Insulin do cơ quan nào trong cơ thể tạo ra?','Xét nghiệm A1C cho biết điều gì?',
                'Giá vàng hôm nay là bao nhiêu?','Công ty Visa hoạt động trong lĩnh vực nào?']

class AskRequest(BaseModel):
    domain:Literal['diabetes','finance']='diabetes'
    question:str=Field(default='',max_length=2000)
    task:Literal['auto','rag','statistics','calculate','graph']='auto'
    operation:Literal['add','subtract','multiply','divide','growth']|None=None
    values:list[float]|None=None
    units:list[str]|None=None
    company:str|None=None
    report:str|None=None

STATUS={'stage':'idle','message':'Chưa khởi động','started':None,'seconds':None,'error':None}
_build_lock=threading.Lock()

@lru_cache(maxsize=1)
def agent():
    stats=DiabetesStats(ROOT/'data/chapter05/brfss2023_diabetes_150k.csv')
    graphs={'finance':load_graph(ROOT/'knowledge_graph/chapter04/finance_graph.json.gz')}
    return ToolAgent(None,stats,graphs)

_rag_engine=None
def rag():
    global _rag_engine
    with _build_lock:
        if _rag_engine is None:
            corpora={d:[json.loads(line) for line in (ROOT/f'data/chapter04/{d}_corpus.jsonl').read_text(encoding='utf-8').splitlines()] for d in ['diabetes','finance']}
            _rag_engine=RAGEngine(corpora,ROOT/'models/chapter04',ROOT/'results/chapter04')
        return _rag_engine

def warm_up():
    """Chạy ở luồng nền khi server khởi động: nạp dữ liệu, bộ truy hồi và mô hình ngôn ngữ."""
    STATUS.update(stage='loading',message='Đang nạp dữ liệu và mô hình ngôn ngữ (lần đầu khoảng 1–2 phút)…',started=time.time(),error=None)
    try:
        agent()
        engine=rag()
        engine.warm_up(WARM_QUESTIONS)
        STATUS.update(stage='ready',message='Sẵn sàng',seconds=time.time()-STATUS['started'])
    except Exception as exc:  # hiển thị lỗi thân thiện trên giao diện thay vì trang trắng
        STATUS.update(stage='error',message='Không nạp được mô hình: '+str(exc),error=repr(exc))

@app.on_event('startup')
def start_background_warmup():
    if os.environ.get('CH5_NO_WARMUP')!='1':
        threading.Thread(target=warm_up,daemon=True).start()

@app.get('/',include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent/'ui.html')

@app.get('/health')
def health():
    engine=_rag_engine
    return {'status':'ok','mode':'local CPU','generator':getattr(engine,'model_name','chưa nạp'),'warmup':STATUS['stage'],
            'agent':'deterministic tool router','statistics':'CDC BRFSS 2023 (150.000 dòng)'}

@app.get('/api/v1/status')
def status():
    elapsed=(time.time()-STATUS['started']) if STATUS['started'] and STATUS['stage']=='loading' else None
    return {**STATUS,'elapsed':elapsed}

@app.get('/api/v1/examples')
def examples():
    return {'statistics':SAMPLE_QUESTIONS_VI}

def friendly_error(message,lang='vi'):
    return {'answer':message,'status':'unsupported','route':'help','route_label':'Chưa xử lý được','steps':[],'sources':[],'citations':[],'trace':[],
            'confidence':{'kind':'unsupported','level':'info','label':CONFIDENCE['unsupported'][0]},'language':lang}

@app.post('/api/v1/knowledge/ask')
def ask(request:AskRequest):
    started=time.perf_counter();worker=agent()
    if not request.question.strip():
        return friendly_error('Bạn chưa nhập câu hỏi. Hãy gõ một câu hỏi hoặc bấm một câu mẫu.')
    lang=detect_language(request.question)
    try:
        resolved=worker.route(request.question,request.domain,request.task,request.operation,request.values,request.company)
        if resolved=='rag':worker.rag=rag()
        body=request.model_dump();body['task']=resolved
        result=worker.ask(**body)
    except (ValueError,TypeError,KeyError) as exc:
        result=friendly_error(f'Mình chưa xử lý được yêu cầu này ({exc}). Hãy thử một câu mẫu bên dưới.',lang)
    result['api_seconds']=time.perf_counter()-started
    result.setdefault('task',result.get('route'))
    return result


# ---- Chương 7: nhóm agent (điều phối, thống kê, truy hồi, tính toán, đồ thị, viết, phản biện); mã ở agents_service.py
from deployment.chapter04.agents_service import make_router
app.include_router(make_router(rag,agent))
