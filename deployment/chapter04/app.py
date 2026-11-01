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


# ---- Chương 7: hệ bốn agent chạy từng bước, trả về State/Message sau mỗi lượt
class TeamRequest(BaseModel):
    domain:Literal['diabetes','finance']='diabetes'
    question:str=Field(default='',max_length=2000)
    report:str|None=None

@lru_cache(maxsize=1)
def team():
    from scripts.tieuluan.complete_agents import MultiAgentTeam
    graphs={'finance':load_graph(ROOT/'knowledge_graph/chapter04/finance_graph.json.gz'),
            'diabetes':load_graph(ROOT/'knowledge_graph/chapter04/diabetes_documents_graph.json.gz')}
    return MultiAgentTeam(rag(),graphs)

@app.get('/agents',include_in_schema=False)
def agents_page():
    return FileResponse(Path(__file__).parent/'agents_ui.html')

ROLES={'RetrievalAgent':('Agent truy hồi','Người đi tìm tài liệu'),'GraphAgent':('Agent đồ thị','Người tra xem đoạn đó thuộc báo cáo hay công ty nào'),
       'WriterAgent':('Agent viết','Người soạn câu trả lời'),'VerifierAgent':('Agent kiểm tra','Người soát nguồn trích dẫn')}
STATUS_LABELS={'generated_unverified':('Đã viết, chưa kiểm chứng','Mọi mã nguồn trích dẫn đều có thật trong kho, nhưng hệ thống chưa kiểm tra được câu trả lời có đúng ý nguồn hay không.'),
               'insufficient_evidence':('Không đủ bằng chứng','Tài liệu tìm được không chứa câu trả lời, nên hệ thống từ chối trả lời thay vì bịa.'),
               'pending':('Đang chờ','Chưa chạy xong.')}

def source_label(node,graph):
    data=graph.nodes[node];kind=data.get('type')
    if kind=='Report':return f"báo cáo {node.split(':')[1].split('/')[0]} năm {data.get('year')}"
    if kind=='Company':return f"công ty {data.get('ticker')}"
    if kind=='Document':return doc_title({'source_url':data['source_url']}).split(' (')[0]
    return node

def summarize(name,state,worker,seconds):
    """Một đoạn văn tiếng Việt kể lại việc agent vừa làm."""
    ev=state.evidence
    if name=='RetrievalAgent':
        if not ev:return 'Agent truy hồi không tìm thấy đoạn nào.','0 đoạn'
        from src.chapter04.rag import content_tokens,lexical_support
        if not lexical_support(state.query or state.question,ev):
            missing=', '.join(sorted(set(content_tokens(state.query)))[:4])
            return (f'Agent truy hồi đã lấy {len(ev)} đoạn gần nhất, nhưng không đoạn nào chứa từ khóa của câu hỏi (truy vấn "{state.query}", thiếu các từ như {missing}). '
                    'Nên nó báo cho nhóm biết kho tài liệu có lẽ không nói về chủ đề này.'),f'{len(ev)} đoạn, không liên quan'
        top=ev[0];terms=highlight_terms(state.query,str(top['text']))
        head=f'Agent truy hồi đã tìm được {len(ev)} đoạn tài liệu liên quan.'
        if state.query and state.query!=state.question:head+=f' Vì kho tài liệu bằng tiếng Anh, nó dịch câu hỏi thành truy vấn: "{state.query}".'
        head+=f' Đoạn tốt nhất đến từ {doc_title(top).split(" (")[0]} và viết: «{snippet(top["text"],terms,200)}»'
        return head,f'{len(ev)} đoạn'
    if name=='GraphAgent':
        if not state.graph_context:
            return ('Agent đồ thị không có đồ thị cho lĩnh vực này nên chuyển tiếp nguyên các đoạn.' if worker.graphs.get(state.domain) is None
                    else 'Agent đồ thị không tìm thấy các đoạn này trong đồ thị.'),'0 đoạn khớp'
        graph=worker.graphs[state.domain]
        labels=[]
        for g in state.graph_context:
            for parent in g['parents']:
                label=source_label(parent,graph)
                if label not in labels:labels.append(label)
        return (f'Agent đồ thị đã tra {len(state.graph_context)} đoạn trong đồ thị tri thức và biết chúng thuộc: {"; ".join(labels)}. '
                'Nhờ vậy câu trả lời có thể nói rõ nguồn nào.'),f'{len(state.graph_context)} đoạn khớp'
    if name=='WriterAgent':
        if state.status=='insufficient_evidence':
            return 'Agent viết đọc các đoạn tìm được và thấy chúng không chứa câu trả lời, nên không viết gì và báo lại cho agent kiểm tra.','từ chối viết'
        return f'Agent viết đã soạn câu trả lời chỉ dựa trên các đoạn đó: «{state.answer}»','đã viết xong'
    ids=[e['id'] for e in ev]
    if state.status=='insufficient_evidence':
        return ('Agent kiểm tra xác nhận hệ thống không đủ bằng chứng, nên câu trả lời cuối là lời từ chối. Đây là kết quả mong muốn: từ chối tốt hơn bịa.','không đủ bằng chứng')
    return (f'Agent kiểm tra soát {len(ids)} mã nguồn trích dẫn: tất cả đều có thật trong kho tài liệu. '
            'Nhưng nó chưa so được nghĩa câu trả lời với nguồn, nên gắn nhãn "chưa kiểm chứng".','nguồn hợp lệ, chưa kiểm chứng ý nghĩa')

HANDOFF={'RetrievalAgent':'Gửi mã các đoạn vừa tìm cho Agent đồ thị','GraphAgent':'Gửi số đoạn đã tra được nguồn cho Agent viết',
         'WriterAgent':'Gửi bản nháp câu trả lời cho Agent kiểm tra','VerifierAgent':'Trả trạng thái cuối cho điều phối viên'}

@app.post('/api/v1/agents/run')
def run_team(request:TeamRequest):
    from scripts.tieuluan.complete_agents import State
    started=time.perf_counter()
    if not request.question.strip():
        return {'question':'','domain':request.domain,'steps':[],'answer':'Bạn chưa nhập câu hỏi. Hãy gõ một câu hỏi hoặc bấm một câu mẫu.','status':'pending','evidence':[],'seconds':0,
                'status_label':STATUS_LABELS['pending'][0],'status_explanation':STATUS_LABELS['pending'][1]}
    try:
        crew=team();state=State(request.question,request.domain,request.report);steps=[]
        for worker in crew.agents:
            t0=time.perf_counter();worker.act(state);seconds=time.perf_counter()-t0
            message=state.messages[-1];name=type(worker).__name__
            summary,outcome=summarize(name,state,worker,seconds)
            steps.append({'agent':name,'role':ROLES[name][0],'role_description':ROLES[name][1],'summary':summary,'outcome':outcome,'handoff':HANDOFF[name],
                          'seconds':seconds,'evidence':len(state.evidence),'graph':len(state.graph_context),'answer':state.answer,'status':state.status,
                          'message':{'sender':message.sender,'receiver':message.receiver,'payload':message.payload}})
    except Exception as exc:
        return {'question':request.question,'domain':request.domain,'steps':[],'answer':f'Không chạy được hệ agent: {exc}','status':'error','evidence':[],
                'seconds':time.perf_counter()-started,'status_label':'Lỗi','status_explanation':'Hệ thống gặp lỗi khi chạy; hãy thử lại sau khi mô hình nạp xong.'}
    label,explain=STATUS_LABELS.get(state.status,(state.status,''))
    evidence=[{'id':e['id'],'title':doc_title(e),'url':e.get('source_url'),'text':str(e.get('text',''))[:260],
               'snippet':snippet(e.get('text',''),highlight_terms(state.query,str(e.get('text','')))),'highlights':highlight_terms(state.query,str(e.get('text','')))} for e in state.evidence]
    out={'question':request.question,'domain':request.domain,'report':request.report,'steps':steps,'answer':state.answer,'status':state.status,
         'status_label':label,'status_explanation':explain,'query_used':state.query,'english_answer':state.english_answer,'language':state.language,
         'evidence':evidence,'seconds':time.perf_counter()-started}
    stats=agent().stats
    if request.domain=='diabetes' and DiabetesStats.looks_statistical(request.question):
        check=stats.answer(request.question)
        if check:out['stats_check']={'answer':check['answer'],'value':check.get('value'),'table':check.get('table'),'source':stats.source}
    return out
