from functools import lru_cache
from pathlib import Path
from typing import Literal
import json,time
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from src.chapter04.agent import ToolAgent
from src.chapter04.knowledge import load_graph
from src.chapter04.rag import RAGEngine

ROOT=Path(__file__).resolve().parents[2]
app=FastAPI(title='Chapter 4 Knowledge & RAG Agent',version='1.0.0')

class AskRequest(BaseModel):
    domain:Literal['diabetes','finance']
    question:str=Field(min_length=1,max_length=2000)
    task:Literal['auto','rag','statistics','calculate','graph']='auto'
    operation:Literal['add','subtract','multiply','divide','growth']|None=None
    values:list[float]|None=None
    units:list[str]|None=None
    company:str|None=None
    report:str|None=None

@lru_cache(maxsize=1)
def agent():
    stats=json.loads((ROOT/'results/chapter04/summary.json').read_text(encoding='utf-8'))['uci']
    graphs={'finance':load_graph(ROOT/'knowledge_graph/chapter04/finance_graph.json.gz')}
    return ToolAgent(None,stats,graphs)

@lru_cache(maxsize=1)
def rag():
    corpora={d:[json.loads(line) for line in (ROOT/f'data/chapter04/{d}_corpus.jsonl').read_text(encoding='utf-8').splitlines()] for d in ['diabetes','finance']}
    return RAGEngine(corpora,ROOT/'models/chapter04',ROOT/'results/chapter04')

@app.get('/health')
def health():
    return {'status':'ok','mode':'local CPU','generator':'google/flan-t5-small','agent':'deterministic tool router'}

@app.post('/api/v1/knowledge/ask')
def ask(request:AskRequest):
    started=time.perf_counter();worker=agent()
    try:
        resolved=worker.resolve_task(request.question,request.domain,request.task,request.operation,request.values,request.company)
        if resolved=='rag':worker.rag=rag()
        body=request.model_dump();body['task']=resolved
        result=worker.ask(**body)
        result['api_seconds']=time.perf_counter()-started
        return result
    except (ValueError,TypeError,KeyError) as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
