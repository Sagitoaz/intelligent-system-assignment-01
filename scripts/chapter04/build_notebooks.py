"""Create four Vietnamese notebooks containing executable, explained experiments."""
from pathlib import Path
import textwrap,sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'notebooks/chapter04';DEST.mkdir(parents=True,exist_ok=True)
def md(s):return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s):return nbf.v4.new_code_cell(textwrap.dedent(s).strip())
SETUP=code('''
from pathlib import Path
import sys, json, time, random, platform, importlib.metadata as metadata
ROOT = Path.cwd().resolve()
while not (ROOT / 'data/chapter04').exists():
    if ROOT.parent == ROOT: raise RuntimeError('Không tìm thấy thư mục dự án')
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/chapter04'
RESULT = ROOT / 'results/chapter04'
FIG = ROOT / 'docs/report_chapter_04'
RESULT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
random.seed(42); np.random.seed(42)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 11})
environment = {'python': platform.python_version(), 'platform': platform.platform(),
    'packages': {p: metadata.version(p) for p in ['numpy','pandas','networkx','torch','transformers','nbclient','fastapi']}}
display(pd.DataFrame(environment['packages'].items(), columns=['Thư viện','Phiên bản']))
''')
def save(name,cells):
    if len(sys.argv)>1 and name not in sys.argv[1:]:return
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec={'display_name':'Chapter 04 (.venv)','language':'python','name':'chapter04'}
    nb.metadata.language_info={'name':'python','version':'3.12'}
    nbf.write(nb,DEST/name)

save('01_kb_kg.ipynb',[
md('''# 01. Knowledge Base và Knowledge Graph
Chương 4.2 và 4.4–4.5. Đọc dataset gốc; thống kê chất lượng; xây và lưu KG thực tế; truy vấn và vẽ đồ thị.

UCI: 101.766 lượt nhập viện. Đồ thị bệnh án dùng mẫu ngẫu nhiên cố định 5.000 lượt để minh họa, không coi mã chẩn đoán hay quan hệ dữ liệu là khuyến nghị điều trị. KG tài liệu dùng toàn bộ corpus NIDDK và FinQA dev. Không đưa trường đáp án FinQA vào KG/corpus.
'''),SETUP,
md('## 1. Đọc và kiểm tra toàn bộ dataset'),
code('''
diabetes = pd.read_csv(DATA / 'diabetic_data.csv', dtype={'diag_1':str,'diag_2':str,'diag_3':str})
finqa = {s: json.loads((DATA / f'finqa_{s}.json').read_text(encoding='utf-8')) for s in ['train','dev','test']}
diabetes_chunks = [json.loads(x) for x in (DATA / 'diabetes_corpus.jsonl').read_text(encoding='utf-8').splitlines()]
profile = pd.DataFrame([
    ['UCI',len(diabetes),len(diabetes.columns),diabetes.patient_nbr.nunique()],
    ['FinQA train',len(finqa['train']),None,None], ['FinQA dev',len(finqa['dev']),None,None],
    ['FinQA test',len(finqa['test']),None,None], ['NIDDK chunks',len(diabetes_chunks),None,None]],
    columns=['Dataset','Số dòng/mẫu','Số cột','Bệnh nhân khác nhau'])
display(profile); display(diabetes.head(5))
assert len(diabetes) == 101766 and len(finqa['dev']) == 883
'''),
md('## 2. Phân tích nhãn và dữ liệu thiếu'),
code('''
label_counts = diabetes.readmitted.value_counts().reindex(['NO','>30','<30'])
missing = (diabetes.eq('?') | diabetes.isna()).sum().sort_values(ascending=False)
missing_table = pd.DataFrame({'Thiếu':missing,'Tỷ lệ %':missing/len(diabetes)*100})
display(label_counts.rename('Số lượt').to_frame()); display(missing_table.head(8))
fig, axes = plt.subplots(1,2,figsize=(12,4))
label_counts.plot.bar(ax=axes[0],color=['#15395B','#167C86','#CE9150'])
axes[0].set(title='UCI: readmission labels',ylabel='Encounters'); axes[0].tick_params(axis='x',rotation=0)
missing_table['Tỷ lệ %'].head(5).plot.barh(ax=axes[1],color='#167C86')
axes[1].set(title='Missing values',xlabel='%'); fig.tight_layout()
fig.savefig(FIG/'uci_profile.png',bbox_inches='tight'); plt.show()
'''),
md('## 3. Xây KG bệnh án mẫu và hai KG tài liệu'),
code('''
from src.chapter04.retrieval import build_finance_chunks
from src.chapter04.knowledge import build_document_graph, build_encounter_graph, save_graph, load_graph
sample = diabetes.sample(n=5000,random_state=42).copy()
encounter_graph = build_encounter_graph(sample)
diabetes_graph = build_document_graph(diabetes_chunks,'diabetes')
finance_chunks = build_finance_chunks(finqa['dev'],table_headers=True)
finance_graph = build_document_graph(finance_chunks,'finance')
(DATA/'finance_corpus.jsonl').write_text(chr(10).join(json.dumps(x,ensure_ascii=False) for x in finance_chunks)+chr(10),encoding='utf-8')
graphs = {'encounters_sample5000':encounter_graph,'diabetes_documents':diabetes_graph,'finance':finance_graph}
graph_stats = []
for name,graph in graphs.items():
    path = ROOT/'knowledge_graph/chapter04'/f'{name}_graph.json.gz'
    save_graph(graph,path)
    restored = load_graph(path)
    assert restored.number_of_nodes() == graph.number_of_nodes()
    assert restored.number_of_edges() == graph.number_of_edges()
    graph_stats.append({'graph':name,'nodes':graph.number_of_nodes(),'edges':graph.number_of_edges(),
                        'node_types':dict(pd.Series([a['type'] for _,a in graph.nodes(data=True)]).value_counts())})
display(pd.DataFrame(graph_stats).drop(columns='node_types'))
'''),
md('## 4. Truy vấn quan hệ và kiểm tra số liệu với CSV'),
code('''
patients = [n for n,a in encounter_graph.nodes(data=True) if a['type']=='Patient']
patient = max(patients,key=encounter_graph.out_degree)
encounters = list(encounter_graph.successors(patient))
display(pd.DataFrame([encounter_graph.nodes[n] for n in encounters]))
graph_count = sum(a.get('readmitted')=='<30' for _,a in encounter_graph.nodes(data=True))
csv_count = int((sample.readmitted=='<30').sum())
assert graph_count == csv_count
companies = [n for n,a in finance_graph.nodes(data=True) if a['type']=='Company']
company = max(companies,key=finance_graph.out_degree)
reports = list(finance_graph.successors(company))
display(pd.DataFrame([finance_graph.nodes[n] for n in reports]).head(6))
print('Graph/CSV <30:',graph_count,'; Company:',company,'; Indexed report contexts:',len(reports))
'''),
md('## 5. Vẽ đồ thị và lưu kết quả'),
code('''
import networkx as nx
focus = [company]+reports[:3]
for r in reports[:3]: focus += list(finance_graph.successors(r))[:4]
sub = finance_graph.subgraph(focus)
pos = nx.spring_layout(sub,seed=42)
fig,ax=plt.subplots(figsize=(12,6))
colors = {'Company':'#CE9150','Report':'#15395B','Passage':'#167C86','TableRow':'#75B8C0'}
nx.draw_networkx_nodes(sub,pos,node_color=[colors[sub.nodes[n]['type']] for n in sub],node_size=550,ax=ax)
nx.draw_networkx_edges(sub,pos,arrows=True,alpha=.5,ax=ax)
labels = {n:(n if n.startswith('company:') else 'Report' if n.startswith('report:') else sub.nodes[n]['type']) for n in sub}
nx.draw_networkx_labels(sub,pos,labels,font_size=8,ax=ax)
ax.set_title('Executed financial document KG (sample view)'); ax.axis('off'); fig.tight_layout()
fig.savefig(FIG/'executed_finance_graph.png',bbox_inches='tight');plt.show()
kb_result = {'profile':profile.fillna('').to_dict('records'),'graphs':graph_stats,
    'encounter_sample_size':5000,'seed':42,'sample_readmitted_under30':graph_count,
    'full_dataset_rows':len(diabetes),'finance_chunks':len(finance_chunks),'environment':environment}
(RESULT/'kb_results.json').write_text(json.dumps(kb_result,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
print('Saved graph artifacts and kb_results.json')
'''),md('''## Nhận xét
KG tài chính ở đây là đồ thị công ty → context báo cáo → đoạn/dòng bảng, giữ locator và nguồn. Chưa trích ontology kế toán đầy đủ hoặc thêm quan hệ nhân quả. Mẫu UCI chọn từ toàn dataset bằng seed 42; không báo cáo chỉ số của mẫu như thể đó là toàn bộ 101.766 lượt. Các query đã đối chiếu với CSV và graph round-trip đã được kiểm tra.
''')])

save('02_diabetes_rag.ipynb',[
md('''# 02. RAG thực thi cho kiến thức tiểu đường
Chương 4.3 và 4.4. Corpus NIDDK tiếng Anh đã tải; truy hồi BM25/dense/hybrid; mô hình FLAN-T5-small thực sự sinh câu trả lời trên CPU.

Tập đánh giá nhỏ do người viết xây từ corpus để minh họa, không phải benchmark y khoa độc lập. Notebook và thảo luận viết bằng tiếng Việt; câu hỏi và đầu ra đánh giá giữ tiếng Anh vì nguồn/model baseline. Không tự nhận đã đánh giá khả năng song ngữ hoặc an toàn lâm sàng.
'''),SETUP,
md('## 1. Tạo dataset hỏi đáp minh họa có bằng chứng'),
code('''
corpus = [json.loads(x) for x in (DATA/'diabetes_corpus.jsonl').read_text(encoding='utf-8').splitlines()]
# question, source substring, accepted references; no QA enters the retrieval corpus.
specs = [
 ('What organ makes insulin?','Insulin is a hormone made by the pancreas',['pancreas','the pancreas']),
 ('Which hormone helps glucose enter cells?','Insulin is a hormone made by the pancreas',['insulin']),
 ('What is the most common type of diabetes?','Type 2 diabetes is the most common',['type 2 diabetes','type 2']),
 ('When does gestational diabetes develop?','develops during pregnancy',['during pregnancy','pregnancy']),
 ('Are glucose levels in prediabetes above normal?','People with prediabetes have blood glucose',['yes','higher than normal']),
 ('What attacks insulin-producing cells in type 1 diabetes?','Your immune system attacks',['the immune system','immune system']),
 ('What causes monogenic diabetes?','a change in a single gene',['a change in a single gene','single gene mutation']),
 ('How many months are reflected in an A1C blood test?','The A1C test is a blood test',['3 months','the last 3 months','three months']),
 ('How many hours should a person fast before the fasting plasma glucose test?','fasted for at least 8 hours',['at least 8 hours','8 hours']),
 ('Can an over-the-counter blood glucose meter diagnose diabetes?','cannot diagnose diabetes',['no']),
 ('During which pregnancy weeks is gestational diabetes screening recommended?','between 24 and 28 weeks',['24 to 28 weeks','between 24 and 28 weeks']),
 ('Which health measure corresponds to B in the diabetes ABCs?','High blood pressure can damage',['blood pressure']),
 ('Which health measure corresponds to C in the diabetes ABCs?','Cholesterol is a fat',['cholesterol']),
 ('How many minutes of moderate activity per week does this source suggest?','at least 150 minutes',['150 minutes','at least 150 minutes']),
 ('How many hours of sleep per night should most adults aim for according to this source?','about 7 to 8 hours',['7 to 8 hours','about 7 to 8 hours']),
 ('Can type 2 diabetes occur during childhood?','even during childhood',['yes'])]
qa=[]
for i,(question,needle,refs) in enumerate(specs):
    matches=[x for x in corpus if needle.lower() in x['text'].lower()]
    assert matches,needle
    qa.append({'id':f'DM{i+1:02d}','question':question,'references':refs,'gold_ids':[matches[0]['id']],
               'annotation':'authored demonstration; source-anchored; not independently reviewed'})
(DATA/'diabetes_qa_demo.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
display(pd.DataFrame(qa)[['id','question','references']])
'''),
md('## 2. Lập chỉ mục BM25 và vector từ mô hình pretrained thật'),
code('''
from src.chapter04.retrieval import BM25, Encoder, DenseRetriever, hybrid_search, retrieval_metrics
from src.chapter04.rag import LocalGenerator, normalized_answer, token_f1
encoder=Encoder(ROOT/'models/chapter04/embedding')
bm25=BM25(corpus)
dense=DenseRetriever(corpus,encoder,RESULT/'diabetes_embeddings.npy')
query_vectors=encoder.encode([x['question'] for x in qa])
print('Corpus vectors:',dense.vectors.shape,'; Query vectors:',query_vectors.shape)
display(pd.DataFrame(hybrid_search(bm25,dense,qa[0]['question'],3))[['id','text','source_url']])
'''),
md('## 3. Đánh giá truy hồi trên 16 câu hỏi'),
code('''
retrieval=[]; rankings={}
for method in ['bm25','dense','hybrid']:
    started=time.perf_counter(); current=[]
    for i,item in enumerate(qa):
        if method=='bm25': hits=bm25.search(item['question'],5)
        elif method=='dense': hits=dense.search(item['question'],5,query_vector=query_vectors[i])
        else: hits=hybrid_search(bm25,dense,item['question'],5,query_vector=query_vectors[i])
        current.append([x['id'] for x in hits])
    metric=retrieval_metrics(current,[set(x['gold_ids']) for x in qa]);metric.update(method=method,seconds=time.perf_counter()-started)
    retrieval.append(metric); rankings[method]=current
display(pd.DataFrame(retrieval).set_index('method').round(4))
'''),
md('## 4. Sinh thật: không context, hybrid RAG và gold-context oracle'),
code('''
generator=LocalGenerator(ROOT/'models/chapter04/generator')
lookup={x['id']:x for x in corpus}; generations=[]
for i,item in enumerate(qa):
    for mode in ['no_context','hybrid_rag','gold_context_oracle']:
        evidence=None if mode=='no_context' else [lookup[x] for x in (rankings['hybrid'][i][:3] if mode=='hybrid_rag' else item['gold_ids'])]
        started=time.perf_counter();answer=generator.generate(item['question'],evidence)
        generations.append({'id':item['id'],'question':item['question'],'mode':mode,'answer':answer,
            'references':item['references'],'exact_match':max(normalized_answer(answer)==normalized_answer(r) for r in item['references']),
            'token_f1':max(token_f1(answer,r) for r in item['references']),
            'evidence_ids':[x['id'] for x in evidence or []],'seconds':time.perf_counter()-started})
generation_table=pd.DataFrame(generations)
display(generation_table[['id','mode','answer','exact_match','token_f1']].head(15))
generation_metrics=generation_table.groupby('mode').agg(questions=('id','size'),exact_match=('exact_match','mean'),token_f1=('token_f1','mean'),mean_seconds=('seconds','mean')).reset_index()
display(generation_metrics.round(4))
'''),
md('## 5. Biểu đồ, lỗi và lưu bằng chứng thực thi'),
code('''
fig,axes=plt.subplots(1,2,figsize=(12,4))
pd.DataFrame(retrieval).set_index('method')[['hit_at_5','recall_at_5','mrr_at_5']].plot.bar(ax=axes[0],ylim=(0,1.05))
axes[0].set_title('Diabetes retrieval (16 authored questions)');axes[0].tick_params(axis='x',rotation=0)
generation_metrics.set_index('mode')[['exact_match','token_f1']].plot.bar(ax=axes[1],ylim=(0,1.05))
axes[1].set_title('FLAN-T5-small generation');axes[1].tick_params(axis='x',rotation=20)
fig.tight_layout();fig.savefig(FIG/'diabetes_rag_metrics.png',bbox_inches='tight');plt.show()
display(generation_table[(generation_table['mode']=='hybrid_rag') & ~generation_table['exact_match']][['question','answer','references','token_f1']].head(6))
result={'questions':len(qa),'retrieval':retrieval,'generation':generation_metrics.to_dict('records'),
        'model':'google/flan-t5-small','embedding_model':'sentence-transformers/all-MiniLM-L6-v2',
        'generation_outputs':generations,'environment':environment,'benchmark':'authored demonstration, not clinical validation'}
(RESULT/'diabetes_rag_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
generation_table.to_csv(RESULT/'diabetes_generation_predictions.csv',index=False,encoding='utf-8-sig')
'''),md('''## Nhận xét
Gold-context là điều kiện oracle tách lỗi truy hồi khỏi lỗi sinh; không phải một hệ thống tự tìm hoàn hảo. EM/F1 đo trùng khớp từ với đáp án ngắn và có thể phạt câu đúng nhưng diễn đạt khác. Citation IDs cho biết nguồn context; chưa chứng minh từng câu sinh được nguồn hỗ trợ. Giữ nguyên câu trả lời sai trong CSV và báo cáo.
''')])

save('03_finance_rag.ipynb',[
md('''# 03. KB và RAG cho báo cáo tài chính/chứng khoán
Thực nghiệm toàn bộ 883 câu hỏi FinQA dev: BM25 dòng thô, BM25 có header, dense và hybrid; mỗi phương án global/oracle-report. Sinh thật trên mẫu 24 câu chọn bằng seed 42.

Train/test được tải đầy đủ nhưng không huấn luyện hoặc tối ưu trên test. Không đưa `answer`, `program`, `exe_ans`, `gold_inds`, `model_input` vào corpus. Báo cáo rõ corpus dev khép kín, lọc báo cáo oracle, giới hạn mẫu generation và không suy diễn thành hiệu quả giao dịch.
'''),SETUP,
md('## 1. Corpus, bằng chứng và phép biến đổi bảng'),
code('''
from src.chapter04.retrieval import BM25,Encoder,DenseRetriever,build_finance_chunks,report_key,hybrid_search,retrieval_metrics
items=json.loads((DATA/'finqa_dev.json').read_text(encoding='utf-8'))
raw_chunks=build_finance_chunks(items,table_headers=False)
chunks=build_finance_chunks(items,table_headers=True)
lookup={x['id']:x for x in chunks}
golds=[{report_key(item)+'::'+key for key in item['qa']['gold_inds']} for item in items]
assert all(g and g.issubset(lookup) for g in golds)
report_allowed={r:{x['id'] for x in chunks if x['report']==r} for r in {x['report'] for x in chunks}}
display(pd.DataFrame([x for x in chunks if x['kind']=='table'][:5])[['key','text','filename']])
print('Questions:',len(items),'Chunks:',len(chunks),'Report contexts:',len(report_allowed))
'''),
md('## 2. Truy hồi pretrained dense và baseline từ khóa'),
code('''
encoder=Encoder(ROOT/'models/chapter04/embedding')
raw_bm25=BM25(raw_chunks);bm25=BM25(chunks)
dense=DenseRetriever(chunks,encoder,RESULT/'finance_embeddings.npy')
queries=[x['qa']['question'] for x in items]
qvec=encoder.encode(queries)
print('Document vectors:',dense.vectors.shape,'; Query vectors:',qvec.shape)
'''),
md('## 3. Chạy 8 cấu hình trên đủ 883 câu hỏi'),
code('''
records=[]; ranked_by_mode={}
for method in ['bm25_raw','bm25_headers','dense','hybrid']:
    for scope in ['global','oracle_report']:
        started=time.perf_counter();ranks=[]
        for i,item in enumerate(items):
            allowed=report_allowed[report_key(item)] if scope=='oracle_report' else None
            if method=='bm25_raw': hits=raw_bm25.search(queries[i],5,allowed)
            elif method=='bm25_headers': hits=bm25.search(queries[i],5,allowed)
            elif method=='dense': hits=dense.search(queries[i],5,allowed,qvec[i])
            else: hits=hybrid_search(bm25,dense,queries[i],5,allowed,qvec[i])
            ranks.append([x['id'] for x in hits])
        metric=retrieval_metrics(ranks,golds);metric.update(method=method,scope=scope,seconds=time.perf_counter()-started)
        records.append(metric);ranked_by_mode[method+'_'+scope]=ranks
        print(method,scope,'Hit@5',round(metric['hit_at_5'],4),flush=True)
metrics=pd.DataFrame(records)
display(metrics.round(4))
metrics.to_csv(RESULT/'finance_retrieval_metrics.csv',index=False,encoding='utf-8-sig')
'''),
md('## 4. Sinh và chấm câu trả lời số trên mẫu cố định'),
code('''
from src.chapter04.rag import LocalGenerator
from src.chapter04.evaluation import parse_numeric_answer, numeric_answer_matches, gold_execution_consistency
import re,math
generator=LocalGenerator(ROOT/'models/chapter04/generator')
selected=sorted(random.Random(42).sample(range(len(items)),24))
outputs=[]
def numeric_value(text):
    parsed=parse_numeric_answer(text)
    return parsed['value'] if parsed else None
for i in selected:
    item=items[i];reference=str(item['qa']['answer']);exe_reference=float(item['qa']['exe_ans'])
    for mode in ['no_context','hybrid_rag_oracle_report','gold_context_oracle']:
        ids=[] if mode=='no_context' else ranked_by_mode['hybrid_oracle_report'][i][:3] if mode=='hybrid_rag_oracle_report' else sorted(golds[i])
        evidence=None if mode=='no_context' else [lookup[cid] for cid in ids]
        started=time.perf_counter();answer=generator.generate(queries[i],evidence);number=numeric_value(answer)
        outputs.append({'id':item['id'],'mode':mode,'question':queries[i],'answer':answer,'reference':reference,'exe_ans':exe_reference,
            'numeric_prediction':number,'numeric_accuracy':numeric_answer_matches(answer,reference),
            'gold_execution_consistent':gold_execution_consistency(reference,exe_reference),
            'evidence_ids':ids,'seconds':time.perf_counter()-started})
generation=pd.DataFrame(outputs)
gen_metrics=generation.groupby('mode').agg(questions=('id','size'),numeric_accuracy=('numeric_accuracy','mean'),mean_seconds=('seconds','mean')).reset_index()
display(gen_metrics.round(4));display(generation[['id','mode','answer','reference','numeric_accuracy']].head(12))
gold_audit=generation[generation['mode']=='no_context'][['id','reference','exe_ans','gold_execution_consistent']].copy()
display(gold_audit[~gold_audit['gold_execution_consistent']])
gold_audit.to_csv(RESULT/'finance_gold_audit.csv',index=False,encoding='utf-8-sig')
generation.to_csv(RESULT/'finance_generation_predictions.csv',index=False,encoding='utf-8-sig')
'''),
md('## 5. Công cụ tính toán trên số liệu thật, không lấy program chuẩn làm dự đoán'),
code('''
from src.chapter04.knowledge import safe_calculate
example=items[0]
# Inspect the actual source table, then select cells by company/column names.
table=pd.DataFrame(example['table'][1:],columns=example['table'][0])
display(table)
row=table[table.iloc[:,0].str.contains('american express')].iloc[0]
volume=float(re.sub(r'[^0-9.]','',row.iloc[1]));transactions=float(re.sub(r'[^0-9.]','',row.iloc[3]))
calculated=safe_calculate('divide',[volume,transactions])
print('Retrieved table calculation:',calculated)
print('Operands manually mapped from source table; not automatic end-to-end FinQA program inference.')
assert math.isclose(calculated['value'],127.4)
'''),
md('## 6. Biểu đồ và lưu kết quả đầy đủ'),
code('''
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
metrics.pivot(index='method',columns='scope',values='hit_at_5').plot.bar(ax=axes[0],ylim=(0,1.05),color=['#15395B','#167C86'])
axes[0].set_title('FinQA dev: Hit@5 on 883 questions');axes[0].tick_params(axis='x',rotation=20)
gen_metrics.set_index('mode')['numeric_accuracy'].plot.bar(ax=axes[1],ylim=(0,1.05),color='#167C86')
axes[1].set_title('FLAN-T5-small numeric accuracy (24 samples)');axes[1].tick_params(axis='x',rotation=20)
fig.tight_layout();fig.savefig(FIG/'finance_rag_extended.png',bbox_inches='tight');plt.show()
result={'retrieval':records,'generation':gen_metrics.to_dict('records'),'generation_sample_size':24,
        'generation_sample_indices':selected,'generation_outputs':outputs,'gold_audit':gold_audit.to_dict('records'),'questions':len(items),
        'chunks':len(chunks),'reports':len(report_allowed),'calculator_example':calculated,'environment':environment,
        'limitations':['oracle report filter','closed dev corpus','English','small generation sample','no trading validation']}
(RESULT/'finance_rag_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(RESULT/'finance_rankings.json').write_text(json.dumps(ranked_by_mode),encoding='utf-8')
'''),md('''## Nhận xét
Truy hồi đúng không đảm bảo mô hình nhỏ tính đúng. Numeric accuracy đo khớp qa.answer theo đơn vị hiển thị, dùng rel_tol=0,001 và abs_tol=0,01. Nếu chuẩn có '%' và model không ghi đơn vị thì diễn giải theo đơn vị chuẩn; '%' tường minh khi chuẩn là ratio được đổi về fraction. Audit qa.answer/exe_ans lưu riêng do một số nhãn chuẩn không nhất quán. Calculator chọn ô thủ công, không tính như bộ giải toàn FinQA.
''')])

save('04_agent_deployment.ipynb',[
md('''# 04. Agent công cụ và triển khai API cục bộ
Agent thực thi có định tuyến, KG query, thống kê CSV, calculator, dense/hybrid RAG và FLAN-T5. Đây là tool router xác định; không gắn nhãn nó thành một LLM tự lập kế hoạch.

Notebook khởi động process uvicorn thật trên loopback, gọi HTTP thật, lưu response và log, rồi dừng process. Không tự nhận đã deploy cloud. Swagger UI tại `/docs` được cung cấp khi người dùng khởi động lại dịch vụ.
'''),SETUP,
md('## 1. Kiểm tra công cụ trước triển khai'),
code('''
from src.chapter04.agent import ToolAgent
from src.chapter04.knowledge import load_graph
stats=json.loads((RESULT/'summary.json').read_text(encoding='utf-8'))['uci']
graph=load_graph(ROOT/'knowledge_graph/chapter04/finance_graph.json.gz')
tool_agent=ToolAgent(None,stats,{'finance':graph})
samples=[
    tool_agent.ask('How many encounters were readmitted within 30 days?','diabetes'),
    tool_agent.ask('Revenue growth','finance',task='calculate',operation='growth',values=[100,120],units=['USD','USD']),
    tool_agent.ask('Find Visa reports','finance',task='graph',company='V')]
display(pd.DataFrame([{'status':x['status'],'answer':x['answer'],'trace':x['trace']} for x in samples]))
'''),
md('## 2. Khởi động API thật, gọi HTTP và thu bằng chứng'),
code('''
import subprocess, socket, requests, os
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
base=f'http://127.0.0.1:{port}'
log_path=RESULT/'agent_api_server.log'
requests_to_run=[
    {'domain':'diabetes','question':'How many encounters were readmitted within 30 days?','task':'statistics'},
    {'domain':'finance','question':'Revenue growth','task':'calculate','operation':'growth','values':[100,120],'units':['USD','USD']},
    {'domain':'finance','question':'Find Visa reports','task':'graph','company':'V'},
    {'domain':'diabetes','question':'What organ makes insulin?','task':'rag'},
    {'domain':'finance','question':'Invalid divide','task':'calculate','operation':'divide','values':[1,0]},
    {'domain':'invalid','question':'wrong domain'}]
api_records=[]
with log_path.open('w',encoding='utf-8') as log:
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    process=subprocess.Popen([sys.executable,'-m','uvicorn','deployment.chapter04.app:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,stdout=log,stderr=log,creationflags=flags)
    try:
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError(log_path.read_text(encoding='utf-8'))
            try:
                health=requests.get(base+'/health',timeout=2)
                if health.status_code==200:break
            except requests.RequestException:pass
            time.sleep(.5)
        else:raise TimeoutError('API did not become healthy')
        print('HTTP GET /health:',health.json())
        swagger=requests.get(base+'/docs',timeout=10)
        assert swagger.status_code==200 and 'SwaggerUIBundle' in swagger.text
        for body in requests_to_run:
            started=time.perf_counter();response=requests.post(base+'/api/v1/knowledge/ask',json=body,timeout=180)
            api_records.append({'request':body,'http_status':response.status_code,'response':response.json(),'round_trip_seconds':time.perf_counter()-started})
            print('HTTP',response.status_code,body['question'],flush=True)
    finally:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()
assert [x['http_status'] for x in api_records]==[200,200,200,200,422,422]
assert api_records[0]['response']['value']==11357
assert api_records[1]['response']['value']==20
assert api_records[3]['response']['citation_ids_valid']
display(pd.DataFrame([{'question':x['request']['question'],'HTTP':x['http_status'],'answer':x['response'].get('answer',x['response'].get('detail')),'seconds':x['round_trip_seconds']} for x in api_records]))
'''),
md('## 3. Trace thật của Agent và kiểm tra nguồn'),
code('''
rag_response=api_records[3]['response']
display(pd.DataFrame(rag_response['trace']))
display(pd.DataFrame(rag_response['citations']))
print('Generated answer:',rag_response['answer'])
print('Status:',rag_response['status'],'; Citation ID check is not semantic correctness validation.')
deployment_result={'health':health.json(),'swagger_status':swagger.status_code,'requests':api_records,
    'deployment':'real local uvicorn HTTP process, stopped after verification','host':'127.0.0.1','port':port,
    'agent_type':'deterministic bounded tool router','environment':environment}
(RESULT/'agent_deployment_results.json').write_text(json.dumps(deployment_result,ensure_ascii=False,indent=2),encoding='utf-8')
'''),
md('''## 4. Chạy lại dịch vụ để trình diễn
Tại thư mục gốc dự án:
```powershell
.venv/Scripts/python.exe -m uvicorn deployment.chapter04.app:app --host 127.0.0.1 --port 8044
```
Mở http://127.0.0.1:8044/docs. Không cần khóa API, corpus và model đều tải trong dự án. Calculator nhận operands có cấu trúc; các con số đó do caller đưa vào, chưa tự trích từ câu hỏi. Mô hình nhỏ có thể trả lời sai; API đánh dấu `generated_unverified` cho đầu ra sinh.

Dockerfile đã chuẩn bị nhưng kiểm chứng triển khai trong notebook là HTTP cục bộ. Không công bố SLA, đánh giá lâm sàng hoặc lợi nhuận chứng khoán.
''')])
print('Created four notebooks:',DEST)
