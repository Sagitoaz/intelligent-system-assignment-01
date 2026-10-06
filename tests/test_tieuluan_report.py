"""Nguồn chỉ được trích dẫn trong Phụ lục vẫn phải xuất hiện trong danh mục."""
from scripts.tieuluan.build_report import Writer


def test_appendix_only_reference_is_in_bibliography():
    writer = Writer({'main': 'Nguồn chương chính', 'appendix': 'Nguồn dữ liệu trực tiếp'})
    writer.markdown('# CHƯƠNG 1\n\nNội dung [@main].')
    appendix = '# PHỤ LỤC\n\nDữ liệu mới [@appendix].'
    writer.references_section(appendix)
    writer.markdown(appendix)
    paragraphs = [p.text for p in writer.doc.paragraphs]
    bibliography = paragraphs[paragraphs.index('TÀI LIỆU THAM KHẢO')+1:paragraphs.index('PHỤ LỤC')]
    assert any('[2]' in p and 'Nguồn dữ liệu trực tiếp' in p for p in bibliography)
    assert 'Dữ liệu mới [2].' in paragraphs
