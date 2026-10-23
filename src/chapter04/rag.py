from pathlib import Path
import re,time,os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
from .retrieval import BM25,DenseRetriever,Encoder,hybrid_search

def build_prompt(question,evidence=None):
    if evidence is None:return f'Answer the question briefly.\nQuestion: {question}\nAnswer:'
    # Put the question before context, so long evidence cannot truncate it away.
    # Source identifiers are attached to outputs separately, not put in T5 context.
    context='\n\n'.join(x['text'] for x in evidence)
    return f'Answer using the context. If the answer is absent, say unknown.\nQuestion: {question}\nContext: {context}\nAnswer:'

class LocalGenerator:
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
        self.torch=torch;torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(str(path),local_files_only=True).eval()
    def generate(self,question,evidence=None):
        prompt=build_prompt(question,evidence)
        inputs=self.tokenizer(prompt,return_tensors='pt',truncation=True,max_length=512)
        with self.torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=64,do_sample=False,num_beams=1)
        return self.tokenizer.decode(output[0],skip_special_tokens=True)

class RAGEngine:
    def __init__(self,corpora,model_root,cache_root):
        model_root=Path(model_root);cache_root=Path(cache_root)
        self.corpora=corpora;self.encoder=Encoder(model_root/'embedding')
        self.generator=LocalGenerator(model_root/'generator')
        self.bm25={d:BM25(c) for d,c in corpora.items()}
        self.dense={d:DenseRetriever(c,self.encoder,cache_root/f'{d}_embeddings.npy') for d,c in corpora.items()}
    def retrieve(self,question,domain,k=3,report=None,method='hybrid'):
        if domain not in self.corpora:raise ValueError('Unsupported domain')
        allowed={x['id'] for x in self.corpora[domain] if x.get('report')==report} if report else None
        if method=='bm25':return self.bm25[domain].search(question,k,allowed)
        if method=='dense':return self.dense[domain].search(question,k,allowed)
        return hybrid_search(self.bm25[domain],self.dense[domain],question,k,allowed)
    def answer(self,question,domain,report=None):
        start=time.perf_counter();evidence=self.retrieve(question,domain,3,report)
        if not evidence:return {'answer':'Insufficient evidence','status':'insufficient_evidence','citations':[],'seconds':time.perf_counter()-start}
        answer=self.generator.generate(question,evidence)
        return {'answer':answer,'status':'generated_unverified','citations':[{'id':x['id'],'url':x['source_url'],'locator':x.get('locator',x['id'])} for x in evidence],
                'evidence':evidence,'seconds':time.perf_counter()-start,'model':'google/flan-t5-small'}

def normalized_answer(text):
    return ' '.join(re.findall(r'[a-z0-9]+',str(text).lower()))

def token_f1(prediction,reference):
    from collections import Counter
    pred=normalized_answer(prediction).split();ref=normalized_answer(reference).split()
    common=sum((Counter(pred)&Counter(ref)).values())
    if not common:return 0.
    precision=common/max(1,len(pred));recall=common/max(1,len(ref))
    return 2*precision*recall/(precision+recall)
