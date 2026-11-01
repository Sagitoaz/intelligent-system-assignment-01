from pathlib import Path
import re,time,os,threading
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
    """Bộ sinh cũ (flan-t5-small), giữ lại để notebook và kiểm thử cũ chạy được."""
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

NL=chr(10)
STOPWORDS=set('a an the is are was were be been of in on at to for from by with and or what which who whom how why when where does do did can could should would about this that these those it its as into than then there their they you your i we our not no any some more most much many'.split())
INSUFFICIENT_VI='Không đủ thông tin trong tài liệu để trả lời câu hỏi này.'
INSUFFICIENT_EN='There is not enough information in the documents to answer this question.'

def detect_language(text):
    """'vi' nếu câu có dấu tiếng Việt hoặc từ khóa không dấu quen thuộc, ngược lại 'en'."""
    from .stats import detect_language as _detect
    return _detect(text)

def content_tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+',text.lower()) if t not in STOPWORDS and len(t)>2]

def lexical_support(query,evidence):
    """Có ít nhất một từ nội dung của truy vấn xuất hiện trong các đoạn truy hồi hay không."""
    terms=set(content_tokens(query))
    if not terms:return True
    pool=set()
    for x in evidence:pool.update(re.findall(r'[a-z0-9]+',str(x['text']).lower()))
    return bool(terms&pool)

class ChatGenerator:
    """Bộ sinh dựa trên mô hình instruct nhỏ (Qwen2.5-1.5B-Instruct) chạy trên CPU.

    Nạp một lần (lazy hoặc qua warm_up), an toàn khi gọi từ nhiều luồng. Chỉ trả lời dựa trên các đoạn truy hồi,
    bằng ngôn ngữ của câu hỏi; trả về cờ insufficient khi mô hình báo tài liệu không có câu trả lời.
    """
    def __init__(self,path,max_new_tokens=110,threads=8):
        self.path=Path(path);self.max_new_tokens=max_new_tokens;self.threads=threads
        self.tokenizer=None;self.model=None;self.state='not_loaded';self.error=None;self.load_seconds=None
        self._lock=threading.Lock();self._run_lock=threading.Lock();self._cache={}
        source=self.path/'SOURCE.txt'
        self.name=source.read_text(encoding='utf-8').strip() if source.exists() else self.path.name
    @property
    def ready(self):return self.state=='ready'
    def load(self):
        with self._lock:
            if self.state=='ready':return
            self.state='loading';start=time.perf_counter()
            try:
                import torch
                from transformers import AutoTokenizer,AutoModelForCausalLM
                self.torch=torch;torch.set_num_threads(self.threads)
                self.tokenizer=AutoTokenizer.from_pretrained(str(self.path),local_files_only=True)
                self.model=AutoModelForCausalLM.from_pretrained(str(self.path),local_files_only=True,dtype=torch.float32).eval()
                self.load_seconds=time.perf_counter()-start;self.state='ready'
            except Exception as exc:
                self.state='error';self.error=repr(exc);raise
    def chat(self,system,user,max_new_tokens=None):
        self.load()
        messages=[{'role':'system','content':system},{'role':'user','content':user}]
        ids=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt',return_dict=True)
        with self._run_lock,self.torch.inference_mode():
            out=self.model.generate(**ids,max_new_tokens=max_new_tokens or self.max_new_tokens,do_sample=False,repetition_penalty=1.05)
        return self.tokenizer.decode(out[0][ids['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    def translate_to_english(self,question):
        key=('tr',question)
        if key in self._cache:return self._cache[key]
        text=self.chat('You are a precise translator. Translate the user question from Vietnamese into one short English question suitable for searching medical or financial documents. Output only the English question.',question,max_new_tokens=40)
        text=text.strip().strip('"').splitlines()[0] if text.strip() else question
        self._cache[key]=text;return text
    GLOSSARY=('pancreas = tuyến tụy; liver = gan; kidney = thận; insulin = insulin; blood glucose / blood sugar = đường huyết; diabetes = bệnh tiểu đường; '
              'type 1 diabetes = tiểu đường típ 1; type 2 diabetes = tiểu đường típ 2; gestational diabetes = tiểu đường thai kỳ; symptoms = triệu chứng; '
              'thirst = khát nước; urination = đi tiểu; hunger = cảm giác đói; fatigue = mệt mỏi; blurry vision = nhìn mờ; weight loss = sụt cân; '
              'A1C test = xét nghiệm A1C; fasting = nhịn ăn; hormone = hormone; cells = tế bào; eyes = mắt; nerves = thần kinh; heart = tim; '
              'revenue = doanh thu; net income = lợi nhuận ròng; shares = cổ phiếu; million = triệu; billion = tỷ')
    def to_vietnamese(self,text):
        system=('You translate English into natural, simple Vietnamese. Use this glossary exactly: '+self.GLOSSARY+'. '
                'Keep every fact, number and unit (such as mg/dL) unchanged and add nothing. Output only the Vietnamese translation.')
        return self.chat(system,text,max_new_tokens=140).strip()
    @staticmethod
    def _is_refusal(text):
        low=text.strip().lower()
        return (not low or low.startswith('insufficient') or 'not enough information' in low or 'does not contain' in low
                or 'do not contain' in low or 'no information' in low or 'cannot be determined' in low or "passages don't" in low)
    def generate_full(self,question,evidence=None,language=None,english_question=None):
        """Trả về dict: answer, english_answer, insufficient. Mô hình nhỏ đọc tiếng Anh tốt hơn tiếng Việt nên
        viết câu trả lời bằng tiếng Anh từ các đoạn tiếng Anh, rồi dịch sang tiếng Việt khi câu hỏi bằng tiếng Việt."""
        language=language or detect_language(question)
        refuse=INSUFFICIENT_VI if language=='vi' else INSUFFICIENT_EN
        if not evidence:return {'answer':refuse,'english_answer':None,'insufficient':True}
        key=('gen',question,tuple(x['id'] for x in evidence))
        if key in self._cache:return self._cache[key]
        q_en=english_question or (self.translate_to_english(question) if language=='vi' else question)
        passages=(NL+'---'+NL).join(str(x['text'])[:700] for x in evidence)
        system=('You answer questions using ONLY the passages provided (separated by ---). Write 2 to 3 short, plain sentences. '
                'Do not use outside knowledge, do not invent numbers and do not write passage numbers or citations. If the passages do not contain the answer, reply with exactly the single word INSUFFICIENT.')
        english=self.chat(system,f'Passages:{NL}{passages}{NL}{NL}Question: {q_en}{NL}Answer:')
        english=re.sub(r'\[\d+\]','',english).strip()
        if self._is_refusal(english) or len(re.findall(r'\w+',english))<3:
            result={'answer':refuse,'english_answer':None,'insufficient':True}
        else:
            result={'answer':self.to_vietnamese(english) if language=='vi' else english.strip(),'english_answer':english.strip(),'insufficient':False}
        self._cache[key]=result;return result
    def generate_ex(self,question,evidence=None,language=None):
        """Trả về (câu trả lời, insufficient: bool)."""
        r=self.generate_full(question,evidence,language)
        return r['answer'],r['insufficient']
    def generate(self,question,evidence=None):
        return self.generate_ex(question,evidence)[0]

class RAGEngine:
    def __init__(self,corpora,model_root,cache_root,generator='auto'):
        """generator: 'auto' (Qwen nếu có ở models/chapter05/llm, ngược lại flan-t5), 'chat' hoặc 'flan'."""
        model_root=Path(model_root);cache_root=Path(cache_root)
        self.corpora=corpora;self.encoder=Encoder(model_root/'embedding')
        llm_path=model_root.parent/'chapter05/llm'
        if generator=='auto':generator='chat' if (llm_path/'config.json').exists() else 'flan'
        self.generator_kind=generator
        self.generator=ChatGenerator(llm_path) if generator=='chat' else LocalGenerator(model_root/'generator')
        self.model_name=self.generator.name if generator=='chat' else 'google/flan-t5-small'
        self.bm25={d:BM25(c) for d,c in corpora.items()}
        self.dense={d:DenseRetriever(c,self.encoder,cache_root/f'{d}_embeddings.npy') for d,c in corpora.items()}
    def warm_up(self,examples=()):
        """Nạp mô hình sinh và dịch sẵn vài câu mẫu để lần hỏi đầu không bị treo."""
        if self.generator_kind=='chat':
            self.generator.load()
            for q in examples:
                try:self.generator.translate_to_english(q)
                except Exception:pass
    def to_english_query(self,question):
        """Trả về (truy vấn tiếng Anh, ngôn ngữ câu hỏi). Câu tiếng Anh được giữ nguyên."""
        language=detect_language(question)
        if language=='vi' and self.generator_kind=='chat':return self.generator.translate_to_english(question),language
        return question,language
    def retrieve(self,question,domain,k=3,report=None,method='hybrid'):
        if domain not in self.corpora:raise ValueError('Unsupported domain')
        allowed={x['id'] for x in self.corpora[domain] if x.get('report')==report} if report else None
        if method=='bm25':return self.bm25[domain].search(question,k,allowed)
        if method=='dense':return self.dense[domain].search(question,k,allowed)
        return hybrid_search(self.bm25[domain],self.dense[domain],question,k,allowed)
    def answer(self,question,domain,report=None):
        start=time.perf_counter()
        query,language=self.to_english_query(question)
        translate_seconds=time.perf_counter()-start
        evidence=self.retrieve(query,domain,3,report)
        base={'query_used':query,'language':language,'translated':query!=question,'model':self.model_name,'translate_seconds':translate_seconds}
        if not evidence or not lexical_support(query,evidence):
            msg=INSUFFICIENT_VI if language=='vi' else INSUFFICIENT_EN
            return {**base,'answer':msg,'status':'insufficient_evidence','citations':[],'evidence':evidence,'seconds':time.perf_counter()-start}
        english_answer=None
        if hasattr(self.generator,'generate_full'):
            full=self.generator.generate_full(question,evidence,language,query);answer,insufficient,english_answer=full['answer'],full['insufficient'],full['english_answer']
        else:answer,insufficient=self.generator.generate(question,evidence),False
        base['english_answer']=english_answer
        if insufficient:
            return {**base,'answer':answer,'status':'insufficient_evidence','citations':[],'evidence':evidence,'seconds':time.perf_counter()-start}
        return {**base,'answer':answer,'status':'generated_unverified','citations':[{'id':x['id'],'url':x['source_url'],'locator':x.get('locator',x['id'])} for x in evidence],
                'evidence':evidence,'seconds':time.perf_counter()-start}

def normalized_answer(text):
    return ' '.join(re.findall(r'[a-z0-9]+',str(text).lower()))

def token_f1(prediction,reference):
    from collections import Counter
    pred=normalized_answer(prediction).split();ref=normalized_answer(reference).split()
    common=sum((Counter(pred)&Counter(ref)).values())
    if not common:return 0.
    precision=common/max(1,len(pred));recall=common/max(1,len(ref))
    return 2*precision*recall/(precision+recall)
