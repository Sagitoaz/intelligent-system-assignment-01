from collections import Counter,defaultdict
import hashlib,json,math,re,os
from pathlib import Path
import numpy as np
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')

def tokenize(text):
    return re.findall(r'[a-z0-9]+',text.lower())

def report_key(item):
    context={k:item[k] for k in ['pre_text','post_text','table']}
    digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()[:12]
    return item['id'].rsplit('-',1)[0]+':'+digest

def build_finance_chunks(items,table_headers=True):
    chunks=[];seen=set()
    for item in items:
        report=report_key(item)
        filename=item['id'].rsplit('-',1)[0]
        for kind,rows in [('text',item['pre_text']+item['post_text']),('table',item['table'])]:
            for i,row in enumerate(rows):
                key=f'{kind}_{i}';cid=report+'::'+key
                if cid in seen:continue
                seen.add(cid)
                if kind=='table':
                    text=' | '.join(map(str,row))
                    if table_headers and i>0:
                        headers=item['table'][0]
                        text=' ; '.join(f'{headers[j] if j<len(headers) else j}: {cell}' for j,cell in enumerate(row))
                else:text=row
                chunks.append({'id':cid,'text':text,'key':key,'report':report,'filename':filename,
                               'company':filename.split('/')[0],'year':filename.split('/')[1],
                               'domain':'finance','kind':kind,
                               'source_url':'https://github.com/czyssrs/FinQA','locator':filename+' '+key})
    return chunks

class BM25:
    def __init__(self,chunks,k1=1.5,b=0.75):
        self.chunks=chunks;self.k1=k1;self.b=b
        self.lengths=[];self.postings=defaultdict(list)
        for i,chunk in enumerate(chunks):
            ts=tokenize(chunk['text']);self.lengths.append(len(ts))
            for term,tf in Counter(ts).items():self.postings[term].append((i,tf))
        self.avg=sum(self.lengths)/max(1,len(chunks))
    def search(self,query,k=5,allowed=None):
        scores=defaultdict(float);n=len(self.chunks)
        for term in set(tokenize(query)):
            postings=self.postings.get(term,[])
            idf=math.log(1+(n-len(postings)+.5)/(len(postings)+.5))
            for i,tf in postings:
                if allowed is not None and self.chunks[i]['id'] not in allowed:continue
                denom=tf+self.k1*(1-self.b+self.b*self.lengths[i]/max(1,self.avg))
                scores[i]+=idf*tf*(self.k1+1)/denom
        ranked=sorted(scores,key=lambda i:(-scores[i],i))[:k]
        return [{**self.chunks[i],'score':float(scores[i])} for i in ranked]

def reciprocal_rank_fusion(rankings,k=5,c=60):
    scores=defaultdict(float)
    for ranking in rankings:
        for rank,cid in enumerate(ranking,1):scores[cid]+=1/(c+rank)
    return sorted(scores,key=lambda cid:(-scores[cid],cid))[:k]

class Encoder:
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer,AutoModel
        torch.set_num_threads(4)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True).eval()
    def encode(self,texts,batch_size=32):
        vectors=[]
        for start in range(0,len(texts),batch_size):
            batch=self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=256,return_tensors='pt')
            with self.torch.inference_mode():
                hidden=self.model(**batch).last_hidden_state
                mask=batch['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                pooled=self.torch.nn.functional.normalize(pooled,p=2,dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors) if vectors else np.empty((0,384),dtype='float32')

class DenseRetriever:
    def __init__(self,chunks,encoder,cache=None):
        self.chunks=chunks;self.encoder=encoder
        fingerprint=hashlib.sha256(json.dumps([(x['id'],x['text']) for x in chunks],ensure_ascii=False).encode()).hexdigest()
        cache=Path(cache) if cache else None
        meta=cache.with_suffix('.meta.json') if cache else None
        if cache and cache.exists() and meta.exists() and json.loads(meta.read_text())['corpus_sha256']==fingerprint:
            self.vectors=np.load(cache)
        else:
            self.vectors=encoder.encode([x['text'] for x in chunks])
            if cache:
                cache.parent.mkdir(parents=True,exist_ok=True);np.save(cache,self.vectors)
                meta.write_text(json.dumps({'corpus_sha256':fingerprint,'shape':list(self.vectors.shape)}))
    def search(self,query,k=5,allowed=None,query_vector=None):
        q=query_vector if query_vector is not None else self.encoder.encode([query])[0]
        scores=self.vectors@q
        indices=[i for i,x in enumerate(self.chunks) if allowed is None or x['id'] in allowed]
        ranked=sorted(indices,key=lambda i:(-float(scores[i]),i))[:k]
        return [{**self.chunks[i],'score':float(scores[i])} for i in ranked]

def hybrid_search(bm25,dense,query,k=5,allowed=None,query_vector=None):
    lexical=bm25.search(query,30,allowed);semantic=dense.search(query,30,allowed,query_vector)
    ids=reciprocal_rank_fusion([[x['id'] for x in lexical],[x['id'] for x in semantic]],k)
    lookup={x['id']:x for x in lexical+semantic}
    return [lookup[cid] for cid in ids]

def retrieval_metrics(rankings,golds):
    n=len(golds);hits1=hits5=recall=mrr=complete=0.
    for ranking,gold in zip(rankings,golds):
        overlap=set(ranking[:5])&set(gold)
        hits1+=bool(ranking and ranking[0] in gold);hits5+=bool(overlap)
        recall+=len(overlap)/len(gold)
        complete+=set(gold).issubset(ranking[:5])
        mrr+=next((1/(i+1) for i,x in enumerate(ranking[:5]) if x in gold),0)
    return dict(questions=n,hit_at_1=hits1/n,hit_at_5=hits5/n,recall_at_5=recall/n,mrr_at_5=mrr/n,complete_at_5=complete/n)
