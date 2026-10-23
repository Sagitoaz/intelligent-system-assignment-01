from .knowledge import safe_calculate
import re

class ToolAgent:
    """Deterministic bounded tool router; not a free-form LLM planning agent."""
    def __init__(self,rag,stats,graphs=None):
        self.rag=rag;self.stats=stats;self.graphs=graphs or {}
    @staticmethod
    def resolve_task(question,domain,task='auto',operation=None,values=None,company=None):
        if task!='auto':return task
        if operation is not None:
            if values is None:raise ValueError('Calculator operation requires values')
            return 'calculate'
        if company:return 'graph'
        lower=question.lower()
        count_question='how many' in lower or 'bao nhiêu' in lower
        dataset_words=any(x in lower for x in ['patient','encounter','dataset','readmit','bệnh nhân','lượt nhập viện','dữ liệu','tái nhập'])
        return 'statistics' if domain=='diabetes' and count_question and dataset_words else 'rag'
    def ask(self,question,domain,task='auto',operation=None,values=None,units=None,report=None,company=None):
        if domain not in ['diabetes','finance']:raise ValueError('Unsupported domain')
        if not question.strip():raise ValueError('Empty question')
        trace=[]
        task=self.resolve_task(question,domain,task,operation,values,company)
        if task=='statistics':
            if domain!='diabetes':raise ValueError('Statistics tool is for diabetes dataset')
            lower=re.sub(r'\s+',' ',question.lower().strip().rstrip('?.'))
            if lower in ['how many patients are in the dataset','how many unique patients are in the dataset','how many patients']:value=self.stats['unique_patients']
            elif lower in ['how many encounters are in the dataset','how many encounters']:value=self.stats['rows']
            elif re.fullmatch(r'how many encounters (?:were )?readmitted (?:within|under|after|more than) 30 days(?: in the dataset)?',lower):
                label='>30' if 'after' in lower or 'more than' in lower else '<30'
                value=self.stats['readmitted'][label]
            else:raise ValueError('Unsupported dataset statistic; use a supported exact count intent')
            trace.append({'tool':'dataset_statistics','result':value})
            return {'answer':str(value),'value':value,'status':'computed','citations':[{'id':'uci296','url':'https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008'}],'trace':trace}
        if task=='calculate':
            result=safe_calculate(operation,values,units)
            trace.append({'tool':'financial_calculator','inputs':{'operation':operation,'values':values,'units':units},'result':result})
            return {**result,'answer':str(result['value']),'status':'computed','trace':trace,'citations':[],'note':'User-supplied operands; not extracted automatically from reports.'}
        if task=='graph':
            graph=self.graphs.get(domain)
            node='company:'+str(company)
            if graph is None or node not in graph:return {'status':'insufficient_evidence','answer':'Company not found','citations':[],'trace':[{'tool':'query_graph','result':[]}]}
            reports=[{'node':n,**graph.nodes[n]} for n in graph.successors(node)]
            trace.append({'tool':'query_graph','company':company,'reports':len(reports)})
            return {'status':'queried','answer':f'{len(reports)} reports in indexed collection','reports':reports,'citations':[],'trace':trace}
        if task!='rag':raise ValueError('Unsupported task')
        if self.rag is None:raise ValueError('RAG engine is not configured')
        result=self.rag.answer(question,domain,report)
        trace.append({'tool':'search_documents','domain':domain,'count':len(result['citations'])})
        trace.append({'tool':'local_generator','model':'google/flan-t5-small'})
        known={x['id'] for x in self.rag.corpora[domain]}
        valid=all(c['id'] in known for c in result['citations'])
        trace.append({'tool':'verify_citation_ids','valid':valid,'semantic_support_checked':False})
        return {**result,'trace':trace,'citation_ids_valid':valid}
