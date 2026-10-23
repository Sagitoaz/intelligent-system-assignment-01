"""Download public data and run a transparent, CPU-only retrieval baseline."""
from pathlib import Path
import collections, csv, hashlib, io, json, math, re, time, zipfile
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/chapter04'
OUT = ROOT / 'results/chapter04'
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
manifest = []

def download(url, name):
    path = DATA / name
    if not path.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    raw = path.read_bytes()
    manifest.append(dict(file=name, url=url, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), accessed='2026-10-13'))
    return raw

raw = download('https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip', 'uci_diabetes.zip')
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        if name.endswith('.csv'):
            (DATA / Path(name).name).write_bytes(z.read(name))
with (DATA / 'diabetic_data.csv').open(encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)
stats = {'rows':len(rows),'columns':len(columns), 'unique_patients':len({r['patient_nbr'] for r in rows}),
         'readmitted':dict(collections.Counter(r['readmitted'] for r in rows)),
         'missing_question_mark':{c:sum(r[c]=='?' for r in rows) for c in columns}}
finqa = {}
for split in ['train','dev','test']:
    finqa[split] = json.loads(download(f'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json',f'finqa_{split}.json'))
download('https://raw.githubusercontent.com/czyssrs/FinQA/main/LICENSE', 'FINQA_LICENSE.txt')
pages = [
 ('what-is-diabetes','https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes'),
 ('symptoms-causes','https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes'),
 ('tests-diagnosis','https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis'),
 ('managing-diabetes','https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes')]
corpus=[]
for slug,url in pages:
    html=download(url,slug+'.html')
    soup=BeautifulSoup(html,'html.parser')
    main=soup.find('main') or soup
    for tag in main(['script','style','nav','header','footer']): tag.decompose()
    blocks=[p.get_text(' ',strip=True) for p in main.find_all(['h1','h2','h3','p','li'])]
    (DATA/(slug+'.txt')).write_text('\n\n'.join(blocks),encoding='utf-8')
    for i,block in enumerate(blocks):
        if len(block.split())>=12:
            corpus.append({'id':f'{slug}:{i}','text':block,'source_url':url,'domain':'diabetes','language':'en','accessed':'2026-10-13'})
(DATA/'diabetes_corpus.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in corpus)+'\n',encoding='utf-8')

def tokens(text): return re.findall(r'[a-z0-9]+',text.lower())
class BM25:
    def __init__(self, texts):
        self.lengths=[]; self.postings=collections.defaultdict(list)
        for idx,text in enumerate(texts):
            ts=tokens(text); self.lengths.append(len(ts))
            for term,tf in collections.Counter(ts).items(): self.postings[term].append((idx,tf))
        self.n=len(texts); self.avg=sum(self.lengths)/max(1,self.n)
    def rank(self, query, k=5, allowed=None):
        scores=collections.defaultdict(float)
        for term in set(tokens(query)):
            postings=self.postings.get(term,[])
            idf=math.log(1+(self.n-len(postings)+0.5)/(len(postings)+0.5))
            for idx,tf in postings:
                if allowed is not None and idx not in allowed: continue
                denom=tf+1.5*(0.25+0.75*self.lengths[idx]/max(self.avg,1))
                scores[idx]+=idf*tf*2.5/denom
        return sorted(scores,key=lambda i:(-scores[i],i))[:k]

# Dev-only candidate collection. No question, answer, program or gold evidence
# is inserted into the retrieval index. Report identity is an oracle filter.
chunks=[]; byreport=collections.defaultdict(set); seen={}
for item in finqa['dev']:
    rid=item['id'].rsplit('-',1)[0]
    context={'pre_text':item['pre_text'],'table':item['table'],'post_text':item['post_text']}
    # Include a context hash to avoid merging different source versions.
    reportkey=rid+':'+hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()[:12]
    item['_reportkey']=reportkey
    for kind, texts in [('text',item['pre_text']+item['post_text']),('table',[' | '.join(map(str,row)) for row in item['table']])]:
        for i,text in enumerate(texts):
            key=(reportkey,f'{kind}_{i}')
            if key not in seen:
                seen[key]=len(chunks)
                chunks.append({'report':reportkey,'evidence_key':key[1],'text':text})
            byreport[reportkey].add(seen[key])
index=BM25([c['text'] for c in chunks])
metrics={}; detailed=[]
for mode in ['global','report_filtered']:
    start=time.perf_counter(); hits1=hits5=0; recall=rr=0.; valid=0
    for item in finqa['dev']:
        gold=set(item['qa'].get('gold_inds',{}))
        mapped={seen[(item['_reportkey'],g)] for g in gold if (item['_reportkey'],g) in seen}
        if not gold or len(mapped)!=len(gold): continue
        valid+=1
        ranked=index.rank(item['qa']['question'],5,byreport[item['_reportkey']] if mode=='report_filtered' else None)
        overlap=set(ranked)&mapped
        hits1+=bool(ranked and ranked[0] in mapped); hits5+=bool(overlap)
        recall+=len(overlap)/len(mapped)
        rr+=next((1/(i+1) for i,c in enumerate(ranked) if c in mapped),0)
        if mode=='report_filtered':
            detailed.append({'id':item['id'],'question':item['qa']['question'],'gold_keys':sorted(gold),'retrieved_keys':[chunks[i]['evidence_key'] for i in ranked], 'hit_at_5':bool(overlap)})
    metrics[mode]={'questions':valid,'hit_at_1':hits1/valid,'hit_at_5':hits5/valid,'recall_at_5':recall/valid,'mrr_at_5':rr/valid,'seconds':time.perf_counter()-start}
summary={'uci':stats,'finqa_splits':{s:len(v) for s,v in finqa.items()},'diabetes_chunks':len(corpus),'finqa_dev_chunks':len(chunks),'finqa_dev_reports':len(byreport),'retrieval':metrics,'scope':'Retrieval only; no LLM generation, no clinical validation, no stock prediction.'}
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'finqa_retrieval_predictions.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in detailed)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='uci'},indent=2))
print('UCI:',stats['rows'],stats['columns'],stats['unique_patients'],stats['readmitted'])
