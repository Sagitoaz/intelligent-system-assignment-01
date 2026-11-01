"""Dựng và chạy lại hai notebook của Chương 5 (05_knowledge_rag) và Chương 7 (07_multi_agent).

Dừng server cổng 8044 trước khi chạy (để không nạp mô hình hai lần trong RAM 16 GB).
Chạy: python scripts/chapter05/build_notebooks.py [--no-run]
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tieuluan import build_complete_notebooks as b  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-run', action='store_true')
    args = parser.parse_args()
    b.NBROOT.mkdir(parents=True, exist_ok=True)
    b.OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, cells in [('05_knowledge_rag', b.knowledge_notebook()), ('07_multi_agent', b.agents_notebook())]:
        for cell in cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': sys.version.split()[0]}})
        path = b.NBROOT / (name + '.ipynb')
        nbf.write(notebook, path)
        if args.no_run:
            continue
        print('Executing', name, flush=True)
        start = time.perf_counter()
        NotebookClient(notebook, timeout=1500, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
        nbf.validate(notebook)
        nbf.write(notebook, path)
        manifest.append({'notebook': str(path.relative_to(ROOT)), 'seconds': time.perf_counter() - start})
        print('  xong sau %.0f giây' % manifest[-1]['seconds'], flush=True)
    if manifest:
        (b.OUT / 'execution_ch5_ch7.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
