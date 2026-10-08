"""Không cho xuất bảng trộn kết quả từ nhiều cấu hình."""
import hashlib
import json
import pytest
from src.tieuluan.course.large_reporting import export_chapter


def test_partial_selection_cannot_be_reported_as_complete(tmp_path):
    with pytest.raises(ValueError,match='đủ hai bộ'):
        export_chapter(tmp_path,'test',['covertype'],['mlp'],11,tmp_path/'missing.ipynb')


def test_report_rejects_mixed_configuration(tmp_path):
    folder=tmp_path/'results/tieuluan/course_large/test'
    folder.mkdir(parents=True)
    notebook=tmp_path/'test.ipynb'
    notebook.write_text(json.dumps({'cells':[]}),encoding='utf8')
    for ds in ['covertype','miniboone']:
        for kind in ['logistic','svm','mlp']:
            for fw in ['scratch','pytorch','keras']:
                record={'dataset':ds,'code_hash':hashlib.sha256(b'').hexdigest(),
                    'metadata':{'splits':{'train':{'n':100001}}},
                    'config':{'epochs':20 if fw=='scratch' else 30}}
                (folder/f'{ds}__{kind}__{fw}__11.json').write_text(json.dumps(record),encoding='utf8')
    with pytest.raises(ValueError,match='khác cấu hình'):
        export_chapter(tmp_path,'test',['covertype','miniboone'],['logistic','svm','mlp'],11,notebook)
