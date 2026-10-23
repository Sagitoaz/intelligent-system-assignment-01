"""Package notebook/code/report and datasets as separate portable ZIP files."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'deliverables/chapter04';DEST.mkdir(parents=True,exist_ok=True)
def package(filename,folders):
    path=DEST/filename
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in folders:
            for file in sorted((ROOT/folder).rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts and '.ipynb_checkpoints' not in file.parts:
                    z.write(file,file.relative_to(ROOT).as_posix())
        for file in (ROOT/'models/chapter04').rglob('*manifest.json'):
            z.write(file,file.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    print(filename,f'{path.stat().st_size/1e6:.2f} MB')
    return {'file':filename,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
records=[package('CHUONG04_NOTEBOOK_CODE_BAOCAO.zip',['notebooks/chapter04','src/chapter04','scripts/chapter04','tests/chapter04','deployment/chapter04','knowledge_graph/chapter04','results/chapter04','docs/report_chapter_04']),
         package('CHUONG04_DATASET.zip',['data/chapter04'])]
(DEST/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
