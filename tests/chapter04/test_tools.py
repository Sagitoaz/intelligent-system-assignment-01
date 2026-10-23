import math
import pytest
from src.chapter04.retrieval import BM25, build_finance_chunks, reciprocal_rank_fusion
from src.chapter04.knowledge import build_document_graph, safe_calculate
from src.chapter04.agent import ToolAgent
from src.chapter04.rag import build_prompt


def test_retriever_obeys_filter_and_unknown_query_returns_empty():
    chunks=[{'id':'a','text':'insulin blood glucose','report':'A'}, {'id':'b','text':'revenue insulin growth','report':'B'}]
    retriever=BM25(chunks)
    assert [x['id'] for x in retriever.search('insulin',allowed={'b'})]==['b']
    assert retriever.search('totallyunknown')==[]
    assert reciprocal_rank_fusion([['a','b'],['b','c']],2)==['b','a']


def test_finance_index_never_indexes_answer_or_gold_and_keeps_headers():
    item={'id':'ACME/2020/p.pdf-1','pre_text':['Annual report'], 'post_text':[],
          'table':[['Metric','2020'],['Revenue','$ 123']],
          'qa':{'question':'secret question','answer':'SECRETANSWER','gold_inds':{'table_1':'SECRETGOLD'}}}
    chunks=build_finance_chunks([item],table_headers=True)
    assert '2020' in next(x['text'] for x in chunks if x['key']=='table_1')
    assert 'SECRET' not in ' '.join(x['text'] for x in chunks)
    graph=build_document_graph(chunks,'finance')
    node=next(x for x in graph if graph.nodes[x].get('chunk_id')==chunks[0]['id'])
    assert list(graph.predecessors(node))


def test_calculator_checks_zero_division_units_and_finite_values():
    assert safe_calculate('growth',[100,120],units=['USD','USD'])['value']==20
    with pytest.raises(ValueError):safe_calculate('growth',[0,120])
    with pytest.raises(ValueError):safe_calculate('add',[100,120],units=['USD','VND'])
    with pytest.raises(ValueError):safe_calculate('divide',[math.inf,2])
    with pytest.raises(ValueError):safe_calculate('__import__',[1,2])


def test_agent_routes_statistics_and_safe_calculator_without_generator():
    agent=ToolAgent(rag=None,stats={'rows':101766,'readmitted':{'<30':11357}})
    result=agent.ask('How many encounters were readmitted within 30 days?',domain='diabetes',task='statistics')
    assert result['value']==11357
    assert result['trace'][0]['tool']=='dataset_statistics'
    result=agent.ask('growth',domain='finance',task='calculate',operation='growth',values=[100,120],units=['USD','USD'])
    assert result['value']==20
    assert result['trace'][0]['tool']=='financial_calculator'
    with pytest.raises(ValueError):agent.ask('x',domain='unknown')


def test_prompt_puts_question_before_long_context_without_citation_markers():
    prompt=build_prompt('What organ makes insulin?',[{'text':'word '*2000}])
    assert prompt.index('What organ makes insulin?')<prompt.index('word')
    assert '[1]' not in prompt


def test_calculator_returns_appropriate_ratio_units():
    assert safe_calculate('divide',[10,2],units=['USD','USD'])['unit']=='dimensionless'
    assert safe_calculate('divide',[10,2],units=['USD','share'])['unit']=='USD/share'
    assert safe_calculate('multiply',[10,2],units=['USD','share'])['unit']=='USD×share'


def test_statistics_does_not_invent_answers_for_unknown_counts():
    agent=ToolAgent(None,{'rows':101766,'unique_patients':71518,'readmitted':{'<30':11357}})
    assert agent.ask('How many patients are in the dataset?',domain='diabetes')['value']==71518
    with pytest.raises(ValueError):agent.ask('How many adults are exactly 30 years old?',domain='diabetes')


def test_statistics_handles_after30_and_rejects_filtered_patients():
    agent=ToolAgent(None,{'rows':101766,'unique_patients':71518,'readmitted':{'<30':11357,'>30':35545}})
    assert agent.ask('How many encounters were readmitted after 30 days?',domain='diabetes')['value']==35545
    with pytest.raises(ValueError):agent.ask('How many female patients are in the dataset?',domain='diabetes')


def test_automatic_routing_distinguishes_medical_questions_and_malformed_calculators():
    assert ToolAgent.resolve_task('How many hours should a person fast?', 'diabetes')=='rag'
    assert ToolAgent.resolve_task('How many months are reflected in an A1C blood test?', 'diabetes')=='rag'
    with pytest.raises(ValueError):ToolAgent.resolve_task('growth','finance',operation='growth')
