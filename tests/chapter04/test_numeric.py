import pytest
from src.chapter04.evaluation import numeric_answer_matches, parse_numeric_answer

def test_percent_reference_scored_in_its_display_unit():
    assert numeric_answer_matches('-3.0%','-3.0%')
    assert numeric_answer_matches('-3.0','-3.0%')
    assert not numeric_answer_matches('-0.03%','-3.0%')
    assert numeric_answer_matches('2%','0.02')
    assert not numeric_answer_matches('2%','2')

def test_numeric_formatting_and_ambiguous_prose():
    assert numeric_answer_matches('$1,234.50','1234.5')
    assert not numeric_answer_matches('1234 and 50','1234.5')
    assert parse_numeric_answer('unknown') is None
    assert not numeric_answer_matches('nan','nan')
