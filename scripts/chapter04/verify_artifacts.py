"""Check packaged data, scoring denominators and DOCX structure."""
from pathlib import Path
import hashlib,json,zipfile
from docx import Document
ROOT=Path(__file__).resolve().parents[2]
summary=json.loads((ROOT/'results/chapter04/summary.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'results/chapter04/dataset_manifest.json').read_text(encoding='utf-8'))
for entry in manifest:
    raw=(ROOT/'data/chapter04'/entry['file']).read_bytes()
    assert len(raw)==entry['bytes']
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
assert summary['uci']['rows']==sum(summary['uci']['readmitted'].values())==101766
assert summary['finqa_splits']=={'train':6251,'dev':883,'test':1147}
for metrics in summary['retrieval'].values():
    assert metrics['questions']==883
    assert all(0<=metrics[k]<=1 for k in ['hit_at_1','hit_at_5','recall_at_5','mrr_at_5'])
    assert metrics['hit_at_1']<=metrics['hit_at_5']
predictions=[json.loads(x) for x in (ROOT/'results/chapter04/finqa_retrieval_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(predictions)==883
assert abs(sum(p['hit_at_5'] for p in predictions)/883-summary['retrieval']['report_filtered']['hit_at_5'])<1e-12
path=ROOT/'docs/report_chapter_04/CHUONG_04_KNOWLEDGE_RAG.docx'
with zipfile.ZipFile(path) as z:
    assert z.testzip() is None
    assert len([n for n in z.namelist() if n.startswith('word/media/')])==8
doc=Document(path)
text='\n'.join(p.text for p in doc.paragraphs)+'\n'+'\n'.join(c.text for t in doc.tables for row in t.rows for c in row.cells)
assert '{{' not in text
for key in ['4.1. Mở đầu','4.2. Knowledge (Graph) – KB','4.3. RAG','4.4. KB và RAG cho bệnh tiểu đường','4.5. KB và RAG cho chứng khoán','4.6. Deploy với Agent','51.98%','80.97%','Tài liệu tham khảo']:
    assert key in text,key
assert len(doc.tables)>=20
assert len(doc.inline_shapes)==8
assert sum(any(r.font.name=='Consolas' for r in p.runs) for p in doc.paragraphs)>=12
import nbformat
execution=json.loads((ROOT/'results/chapter04/notebook_execution.json').read_text(encoding='utf-8'))
assert len(execution)==4
for record in execution:
    nbpath=ROOT/'notebooks/chapter04'/record['notebook']
    assert hashlib.sha256(nbpath.read_bytes()).hexdigest()==record['sha256']
    nb=nbformat.read(nbpath,as_version=4)
    cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
models=json.loads((ROOT/'models/chapter04/model_manifest.json').read_text(encoding='utf-8'))
for model in models:
    role='embedding' if 'MiniLM' in model['repo'] else 'generator'
    for entry in model['files']:
        assert hashlib.sha256((ROOT/'models/chapter04'/role/entry['file']).read_bytes()).hexdigest()==entry['sha256']
api=json.loads((ROOT/'results/chapter04/agent_deployment_results.json').read_text(encoding='utf-8'))
assert api['swagger_status']==200
assert [x['http_status'] for x in api['requests']]==[200,200,200,200,422,422]
finance=json.loads((ROOT/'results/chapter04/finance_rag_results.json').read_text(encoding='utf-8'))
assert len(finance['retrieval'])==8 and all(x['questions']==883 for x in finance['retrieval'])
assert len(finance['generation_outputs'])==72 and len(finance['gold_audit'])==24
assert not list((ROOT/'docs/report_chapter_04').glob('*.pdf'))
print('PASS: dataset/model hashes, counts, 4 executed notebooks with outputs, 8 retrieval configs, 72 finance generations, gold audit, real HTTP API, DOCX code/tables/8 figures, no PDF.')
print('Report bytes:',path.stat().st_size)
