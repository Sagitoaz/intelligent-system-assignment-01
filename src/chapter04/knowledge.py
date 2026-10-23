import gzip,json,math
from pathlib import Path
import networkx as nx

def build_document_graph(chunks,domain):
    graph=nx.DiGraph(domain=domain)
    for chunk in chunks:
        if domain=='finance':
            company='company:'+chunk['company'];report='report:'+chunk['report']
            graph.add_node(company,type='Company',ticker=chunk['company'])
            graph.add_node(report,type='Report',year=chunk['year'],filename=chunk['filename'],source_url=chunk['source_url'])
            graph.add_edge(company,report,relation='PUBLISHED')
        else:
            report='document:'+chunk['source_url']
            graph.add_node(report,type='Document',source_url=chunk['source_url'])
        node='passage:'+chunk['id']
        graph.add_node(node,type='TableRow' if chunk.get('kind')=='table' else 'Passage',chunk_id=chunk['id'],text=chunk['text'],source_url=chunk['source_url'])
        graph.add_edge(report,node,relation='HAS_PASSAGE')
    return graph

def build_encounter_graph(frame):
    graph=nx.DiGraph(domain='diabetes_research',source='UCI dataset 296')
    graph.add_node('dataset:uci296',type='Dataset',source_url='https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008')
    for row in frame.itertuples(index=False):
        patient='patient:'+str(row.patient_nbr);encounter='encounter:'+str(row.encounter_id)
        graph.add_node(patient,type='Patient',patient_nbr=str(row.patient_nbr))
        graph.add_node(encounter,type='Encounter',encounter_id=str(row.encounter_id),age=str(row.age),readmitted=row.readmitted,time_in_hospital=int(row.time_in_hospital))
        graph.add_edge(patient,encounter,relation='HAS_ENCOUNTER')
        graph.add_edge(encounter,'dataset:uci296',relation='FROM_DATASET')
        for field in ['diag_1','diag_2','diag_3']:
            code=str(getattr(row,field))
            if code not in ['?','nan','None']:
                node='diagnosis:'+code;graph.add_node(node,type='DiagnosisCode',code=code)
                graph.add_edge(encounter,node,relation='HAS_RECORDED_CODE')
    return graph

def save_graph(graph,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(nx.node_link_data(graph,edges='edges'),f,ensure_ascii=False)

def load_graph(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return nx.node_link_graph(json.load(f),edges='edges')

def safe_calculate(operation,values,units=None):
    if operation not in ['add','subtract','multiply','divide','growth']:raise ValueError('Unsupported operation')
    if len(values)!=2:raise ValueError('Exactly two values are required')
    a,b=map(float,values)
    if not all(math.isfinite(x) for x in [a,b]):raise ValueError('Non-finite input')
    if units is not None and len(units)!=2:raise ValueError('Two units are required')
    if units is not None and operation in ['add','subtract','growth'] and units[0]!=units[1]:raise ValueError('Units must match')
    if operation=='add':value=a+b;formula=f'{a} + {b}'
    elif operation=='subtract':value=a-b;formula=f'{a} - {b}'
    elif operation=='multiply':value=a*b;formula=f'{a} × {b}'
    elif operation=='divide':
        if b==0:raise ValueError('Division by zero')
        value=a/b;formula=f'{a} / {b}'
    else:
        if a==0:raise ValueError('Growth denominator is zero')
        value=(b-a)/a*100;formula=f'({b} - {a}) / {a} × 100%'
    if not math.isfinite(value):raise ValueError('Non-finite result')
    unit='unspecified'
    if operation=='growth':unit='%'
    elif units:
        if operation=='divide':unit='dimensionless' if units[0]==units[1] else units[0]+'/'+units[1]
        elif operation=='multiply':unit=units[0]+'×'+units[1]
        else:unit=units[0]
    return {'value':value,'formula':formula,'unit':unit}
