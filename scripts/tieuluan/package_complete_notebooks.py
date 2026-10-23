"""Validate executed notebooks and package the supplement with its DOCX."""
import json
from pathlib import Path
import shutil
import zipfile
import re
import nbformat
from docx import Document
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'deliverables/tieuluan_hoan_chinh_6_chuong'
OUTPUT=DEST/'TIEU_LUAN_6_CHUONG_CO_CELL_NOTEBOOK.docx'


def main():
    directory=ROOT/'notebooks/tieuluan/complete'
    manifest=json.loads((ROOT/'results/tieuluan/complete/notebooks/execution.json')
                        .read_text(encoding='utf8'))
    assert len(manifest)==5 and all(r['executed'] for r in manifest)
    labels=[]
    cells_count=0
    for path in sorted(directory.glob('*.ipynb')):
        notebook=nbformat.read(path,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type!='code':
                continue
            assert cell.execution_count is not None,(path,cell.id)
            assert not any(o.output_type=='error' for o in cell.outputs)
            label=cell.metadata['report_label']
            labels.append(label)
            cells_count+=1
            # A visible comment labels the executed cell without changing its logic.
            if not cell.source.startswith('# Cell '+label+'\n'):
                cell.source='# Cell '+label+'\n'+cell.source
        nbformat.validate(notebook)
        nbformat.write(notebook,path)
    assert len(labels)==len(set(labels))
    document=Document(OUTPUT)
    embedded=sum(bool(re.match(
        r'^Cell (?:CNN|RNN|LSTM|KG|AGENT)-\d{2}\b',p.text))
        for p in document.paragraphs)
    explanations=sum(p.text.startswith('Giải thích hàm và liên hệ công thức.')
                     for p in document.paragraphs)
    assert embedded==11 and explanations==11
    document_info=DEST/'verification_docx.json'
    if document_info.exists():
        info=json.loads(document_info.read_text(encoding='utf8'))
        info['embedded_notebook_cells']=embedded
        info['matching_function_explanations']=explanations
        document_info.write_text(json.dumps(info,ensure_ascii=False,indent=2),
                                 encoding='utf8')
    for kind in ['cnn','rnn','lstm']:
        payload=json.loads((ROOT/f'results/tieuluan/complete/notebooks/{kind}_training.json')
                           .read_text(encoding='utf8'))
        assert {r['framework'] for r in payload['results']}=={'scratch','keras','pytorch'}
        assert all(r['trainable_layers']==6 for r in payload['results'])
    target=DEST/'notebooks'
    target.mkdir(exist_ok=True)
    for source in directory.iterdir():
        if source.suffix in ['.ipynb','.md']:
            shutil.copy2(source,target/source.name)
    package=DEST/'BO_SUNG_NOTEBOOK_VA_BAO_CAO.zip'
    paths=[OUTPUT]
    paths+=list(directory.glob('*.ipynb'))+[directory/'README.md']
    paths+=list((ROOT/'results/tieuluan/complete/notebooks').glob('*'))
    paths+=list((ROOT/'src/tieuluan/scratch').glob('*.py'))
    paths+=list((ROOT/'src/chapter04').glob('*.py'))
    paths+=[ROOT/'deployment/chapter04/app.py']
    paths+=[ROOT/'scripts/tieuluan'/name for name in [
        'complete_models.py','complete_agents.py','verify_complete.py',
        'build_complete_notebooks.py','notebook_report_cells.py']]
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
    report={'notebooks':5,'executed_code_cells':cells_count,
            'error_outputs':0,'training_configurations':9,
            'embedded_report_cells':embedded,
            'matching_function_explanations':explanations,
            'zip':str(package),'zip_bytes':package.stat().st_size}
    (DEST/'notebook_verification.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
