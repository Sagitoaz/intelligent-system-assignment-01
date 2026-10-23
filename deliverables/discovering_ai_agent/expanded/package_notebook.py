from pathlib import Path
import json,zipfile
ROOT=Path(__file__).resolve().parent
verification=json.loads((ROOT/'verification_expanded.json').read_text(encoding='utf-8'))
assert verification['pages']==80 and verification['notebook_error_outputs']==0
target=ROOT/'agent01_notebook_va_ket_qua.zip'
files=[ROOT/'agent01_notebook_mo_rong.ipynb',ROOT/'agent01_notebook_mo_rong.html',
    ROOT/'requirements-notebook.txt',ROOT/'README_MO_RONG.md',ROOT/'verification_expanded.json']
files+=sorted((ROOT/'artifacts').glob('*'))
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in files:
        if path.is_file():archive.write(path,path.relative_to(ROOT))
with zipfile.ZipFile(target) as archive:assert archive.testzip() is None
print(target)
print('Files:',len(files),'| bytes:',target.stat().st_size)
