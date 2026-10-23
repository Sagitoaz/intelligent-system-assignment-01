"""Execute every notebook using this interpreter, retaining outputs on disk."""
from pathlib import Path
import sys,json,time,hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[2]
kernel_dir=ROOT/'.jupyter/kernels/chapter04';kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Chapter 04 (.venv)','language':'python'}),encoding='utf-8')
names=sys.argv[1:] or ['01_kb_kg.ipynb','02_diabetes_rag.ipynb','03_finance_rag.ipynb','04_agent_deployment.ipynb']
records=[]
checkpoints=ROOT/'results/chapter04/execution'
checkpoints.mkdir(exist_ok=True)
for name in names:
    path=ROOT/'notebooks/chapter04'/name
    nb=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name='chapter04',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    client=NotebookClient(nb,timeout=1800,startup_timeout=180,km=manager,resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code':print(f'  {name}: code cell {cell_index}',flush=True)
    client.on_cell_start=progress
    started=time.perf_counter();print('Executing',name,flush=True)
    try:
        client.execute()
    finally:
        nbformat.write(nb,path)
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type=='error' for c in codes for o in c.outputs)
    record={'notebook':name,'seconds':time.perf_counter()-started,'code_cells':len(codes),'output_cells':sum(bool(c.outputs) for c in codes),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'interpreter':sys.executable}
    records.append(record);print('Completed',record,flush=True)
    (checkpoints/(name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
result=ROOT/'results/chapter04/notebook_execution.json'
previous=json.loads(result.read_text()) if result.exists() else []
combined={x['notebook']:x for x in previous+records}
for path in checkpoints.glob('*.json'):
    record=json.loads(path.read_text(encoding='utf-8'))
    notebook=ROOT/'notebooks/chapter04'/record['notebook']
    if hashlib.sha256(notebook.read_bytes()).hexdigest()==record['sha256']:combined[record['notebook']]=record
result.write_text(json.dumps(list(combined.values()),indent=2),encoding='utf-8')
