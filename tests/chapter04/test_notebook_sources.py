import ast
from pathlib import Path
import nbformat

def test_every_notebook_code_cell_is_valid_python():
    root=Path(__file__).resolve().parents[2]
    paths=sorted((root/'notebooks/chapter04').glob('*.ipynb'))
    assert len(paths)==4
    for path in paths:
        nb=nbformat.read(path,as_version=4)
        for i,cell in enumerate(nb.cells):
            if cell.cell_type=='code':ast.parse(cell.source,filename=f'{path.name}:cell{i}')
