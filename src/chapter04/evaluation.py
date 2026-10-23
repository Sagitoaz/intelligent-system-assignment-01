"""Numeric scoring with explicit display units and no gold-program inference."""
import math,re

def parse_numeric_answer(text):
    cleaned=str(text).strip().replace(',','').replace('$','')
    match=re.fullmatch(r'([-+]?\d+(?:\.\d+)?)\s*(%)?',cleaned)
    if not match:return None
    return {'value':float(match.group(1)),'unit':'%' if match.group(2) else 'number'}

def numeric_answer_matches(prediction,reference):
    pred=parse_numeric_answer(prediction);ref=parse_numeric_answer(reference)
    if pred is None or ref is None:return False
    value=pred['value']
    # Missing unit is interpreted in the expected answer's display unit.
    # Explicit % on a prediction against a unitless ratio is converted.
    if ref['unit']=='number' and pred['unit']=='%':value/=100
    return math.isclose(value,ref['value'],rel_tol=.001,abs_tol=.01)

def gold_execution_consistency(answer,exe_ans):
    parsed=parse_numeric_answer(answer)
    if parsed is None:return None
    raw=parsed['value'];execution=float(exe_ans)
    candidates=[raw,raw/100] if parsed['unit']=='%' else [raw]
    return any(math.isclose(x,execution,rel_tol=.01,abs_tol=.001) for x in candidates)
