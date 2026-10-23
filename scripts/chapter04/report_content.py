"""Populate the report with code and measured outputs from executed notebooks."""
from pathlib import Path
import ast,json

def markdown_table(headers,rows):
    def clean(x):return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(clean(x) for x in row)+' |' for row in rows])

def source_listing(root,file,target):
    source=(root/file).read_text(encoding='utf-8');tree=ast.parse(source)
    scope=tree
    for name in target.split('.'):
        scope=next(x for x in scope.body if isinstance(x,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    lines=source.splitlines()[scope.lineno-1:scope.end_lineno]
    return f'Code thực thi: {file}, {target}.\n\n```python\n'+'\n'.join(lines)+'\n```'

def enrich_report(text,root):
    result=root/'results/chapter04'
    read=lambda name:json.loads((result/name).read_text(encoding='utf-8'))
    kb=read('kb_results.json');diabetes=read('diabetes_rag_results.json');finance=read('finance_rag_results.json');api=read('agent_deployment_results.json')
    executions=sorted(read('notebook_execution.json'),key=lambda x:x['notebook'])
    code_map={
        '{{KG_CODE}}':('src/chapter04/knowledge.py','build_encounter_graph'),
        '{{CHUNK_CODE}}':('src/chapter04/retrieval.py','build_finance_chunks'),
        '{{RETRIEVER_CODE}}':('src/chapter04/retrieval.py','BM25.search'),
        '{{EMBEDDING_CODE}}':('src/chapter04/retrieval.py','Encoder.encode'),
        '{{GENERATOR_CODE}}':('src/chapter04/rag.py','LocalGenerator.generate'),
        '{{CALCULATOR_CODE}}':('src/chapter04/knowledge.py','safe_calculate'),
        '{{EVALUATION_CODE}}':('src/chapter04/retrieval.py','retrieval_metrics'),
        '{{NUMERIC_SCORE_CODE}}':('src/chapter04/evaluation.py','numeric_answer_matches'),
        '{{AGENT_CODE}}':('src/chapter04/agent.py','ToolAgent.ask'),
        '{{API_CODE}}':('deployment/chapter04/app.py','ask')}
    for placeholder,(file,target) in code_map.items():
        listing=source_listing(root,file,target)
        if placeholder=='{{GENERATOR_CODE}}':listing=source_listing(root,file,'build_prompt')+'\n\n'+listing
        text=text.replace(placeholder,listing)
    pct=lambda n:f'{n*100:.2f}%'
    replacements={}
    replacements['{{NOTEBOOK_TABLE}}']=markdown_table(['Notebook','Code cells','Cells có output','Thời gian'],[[x['notebook'],x['code_cells'],x['output_cells'],f"{x['seconds']:.2f} s"] for x in executions])
    replacements['{{KG_TABLE}}']=markdown_table(['Đồ thị','Số đỉnh','Số cạnh','Phạm vi'],[[x['graph'],x['nodes'],x['edges'],'5.000 lượt mẫu' if x['graph']=='encounters_sample5000' else 'Toàn corpus tương ứng'] for x in kb['graphs']])
    replacements['{{KG_QUERY_RESULT}}']=f"Truy vấn đếm Encounter có readmitted = '<30' trên graph mẫu trả {kb['sample_readmitted_under30']} lượt, trùng kết quả lọc CSV mẫu. Đây là số đếm trên 5.000 lượt, tách biệt với 11.357 lượt '<30' của toàn bộ UCI. Notebook cũng truy vấn công ty có nhiều context báo cáo nhất trong collection và liệt kê các báo cáo liên kết."
    replacements['{{DIABETES_RETRIEVAL_TABLE}}']=markdown_table(['Phương pháp','n','Hit@1','Hit@5','Recall@5','MRR@5'],[[x['method'],x['questions']]+[pct(x[k]) for k in ['hit_at_1','hit_at_5','recall_at_5','mrr_at_5']] for x in diabetes['retrieval']])
    replacements['{{DIABETES_GENERATION_TABLE}}']=markdown_table(['Cấu hình sinh','n','Exact match','Token F1','Giây/câu'],[[x['mode'],x['questions'],pct(x['exact_match']),pct(x['token_f1']),f"{x['mean_seconds']:.3f}"] for x in diabetes['generation']])
    actual=[x for x in diabetes['generation_outputs'] if x['mode']=='hybrid_rag']
    # Include successful and unsuccessful outputs. Never cherry-pick only successes.
    samples=actual[:2]+[x for x in actual[2:] if not x['exact_match']][:3]
    replacements['{{DIABETES_OUTPUT_TABLE}}']=markdown_table(['Câu hỏi','Đầu ra model nguyên văn','Đáp án chấp nhận','F1'],[[x['question'],x['answer'],'; '.join(x['references']),pct(x['token_f1'])] for x in samples])
    dm={x['mode']:x for x in diabetes['generation']}
    replacements['{{DIABETES_ANALYSIS}}']=f"Hybrid RAG đạt EM {pct(dm['hybrid_rag']['exact_match'])} và token F1 {pct(dm['hybrid_rag']['token_f1'])}; không context đạt EM {pct(dm['no_context']['exact_match'])}; gold context đạt EM {pct(dm['gold_context_oracle']['exact_match'])}. Đọc cùng các đầu ra nguyên văn: khác cách diễn đạt hoặc chứa thêm từ có thể làm giảm EM dù một phần nội dung đúng. Mẫu 16 câu neo nguồn không đủ để kết luận chất lượng lâm sàng. Độ dài context 512 token cũng có thể cắt mất thông tin ở các đoạn dài."
    replacements['{{RETRIEVAL_TABLE}}']=markdown_table(['Phương pháp / scope','Hit@1','Hit@5','Recall@5','MRR@5','Đủ gold@5','Thời gian'],[[x['method']+' / '+x['scope']]+[pct(x[k]) for k in ['hit_at_1','hit_at_5','recall_at_5','mrr_at_5','complete_at_5']]+[f"{x['seconds']:.2f} s"] for x in finance['retrieval']])
    fm={x['method']+'_'+x['scope']:x for x in finance['retrieval']}
    raw=fm['bm25_raw_global'];oracle=fm['bm25_raw_oracle_report'];hybrid=fm['hybrid_global']
    replacements['{{RESULT_DISCUSSION}}']=f"Corpus có {finance['chunks']} chunk và {finance['reports']} context báo cáo. Cả 883 câu hỏi có đủ khóa gold để chấm. BM25 dòng thô có Hit@5 {pct(raw['hit_at_5'])} ở global và {pct(oracle['hit_at_5'])} khi biết đúng report, chênh {(oracle['hit_at_5']-raw['hit_at_5'])*100:.2f} điểm phần trăm. Hybrid global đạt {pct(hybrid['hit_at_5'])}. Chỉ số Đủ gold@5 đo tỷ lệ tìm đủ mọi bằng chứng, nghiêm ngặt hơn Hit@5; bảng cho phép phân biệt tìm được một dữ kiện với tìm đủ dữ kiện cho phép tính. Thời gian truy hồi không bao gồm xây embedding hoặc sinh trả lời; các query vector đã được tính trước cho dense/hybrid, nên không dùng bảng thời gian này để so sánh latency end-to-end."
    replacements['{{FINANCE_GENERATION_TABLE}}']=markdown_table(['Cấu hình sinh','n','Numeric accuracy','Giây/câu'],[[x['mode'],x['questions'],pct(x['numeric_accuracy']),f"{x['mean_seconds']:.3f}"] for x in finance['generation']])
    outputs=[x for x in finance['generation_outputs'] if x['mode']=='hybrid_rag_oracle_report']
    replacements['{{FINANCE_OUTPUT_TABLE}}']=markdown_table(['Mã câu hỏi','Đầu ra model','Giá trị chuẩn','Đúng số'],[[x['id'],x['answer'],x['reference'],'Có' if x['numeric_accuracy'] else 'Không'] for x in outputs[:6]])
    gen={x['mode']:x for x in finance['generation']}
    replacements['{{FINANCE_ANALYSIS}}']=f"Trên 24 câu cố định, hybrid RAG với report oracle đạt numeric accuracy {pct(gen['hybrid_rag_oracle_report']['numeric_accuracy'])}; gold context đạt {pct(gen['gold_context_oracle']['numeric_accuracy'])}; không context đạt {pct(gen['no_context']['numeric_accuracy'])}. Cần xem đầu ra sai để hiểu giới hạn tính toán của FLAN-T5-small. Mẫu nhỏ và điều kiện oracle không đại diện một hệ thống hỏi đáp tài chính production. Không lấy program/gold đầu vào làm kết quả dự đoán."
    flagged=[x for x in finance['gold_audit'] if x['gold_execution_consistent'] is False]
    replacements['{{GOLD_AUDIT_TABLE}}']=f"Audit đối chiếu qa.answer và exe_ans phát hiện {len(flagged)}/{finance['generation_sample_size']} mẫu vượt ngưỡng nhất quán sau khi xét cả cách biểu diễn phần trăm và fraction. Đây là cảnh báo cần rà soát nhãn, không phải bằng chứng xác định mọi trường nào đúng. Chỉ số sinh phía trên được tính theo qa.answer, không âm thầm đổi qua exe_ans cho từng mẫu.\n\n"+markdown_table(['Mã mẫu cần rà soát','qa.answer','exe_ans'],[[x['id'],x['reference'],x['exe_ans']] for x in flagged])
    calc=finance['calculator_example']
    replacements['{{CALCULATOR_RESULT}}']=f"Kết quả công cụ trên dòng American Express của bảng FinQA đầu tiên: {calc['formula']} = {calc['value']}. Hai giá trị được chọn từ cột payments volume và total transactions, đều ở thang billions nên triệt tiêu thang đo khi chia. Đây là thao tác đọc bảng thủ công có kiểm chứng; không được cộng vào numeric accuracy của LLM."
    replacements['{{API_RESULT_TABLE}}']=markdown_table(['HTTP request','Status','Đáp án / lỗi','Giây HTTP'],[[x['request']['question'],x['http_status'],x['response'].get('answer',str(x['response'].get('detail')))[:200],f"{x['round_trip_seconds']:.3f}"] for x in api['requests']])
    response=next(x['response'] for x in api['requests'] if x['request'].get('task')=='rag')
    replacements['{{API_TRACE_TABLE}}']=markdown_table(['Bước','Tool','Kết quả'],[[i+1,x['tool'],json.dumps({k:v for k,v in x.items() if k!='tool'},ensure_ascii=False)] for i,x in enumerate(response['trace'])])
    model_manifest=json.loads((root/'models/chapter04/model_manifest.json').read_text(encoding='utf-8'))
    replacements['{{MODEL_TABLE}}']=markdown_table(['Model đã tải','Revision cố định','Trọng số'],[[m['repo'],m['revision'],f"{next(x['bytes'] for x in m['files'] if x['file']=='model.safetensors')/1e6:.2f} MB"] for m in model_manifest])
    for key,value in replacements.items():text=text.replace(key,value)
    return text
