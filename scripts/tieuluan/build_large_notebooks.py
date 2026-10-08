"""Sinh notebook có mã thật trong từng ô, không tự huấn luyện.

python -m scripts.tieuluan.build_large_notebooks
Không chạy lại lệnh này sau khi sửa notebook thủ công nếu chưa lưu bản sao.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import textwrap
import sys
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'notebooks/tieuluan'


def source_text(path,selected=None,rename=None):
    """Bỏ import mã nội bộ vì toàn bộ định nghĩa đã nằm trong các ô trước."""
    text=(ROOT/path).read_text(encoding='utf8')
    tree=ast.parse(text)
    blocks=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and (node.level or (node.module or '').startswith('src.tieuluan') or node.module=='__future__'):
            continue
        if selected is not None and not(isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in selected):
            continue
        block=ast.get_source_segment(text,node)
        if rename and isinstance(node,ast.FunctionDef) and node.name in rename:
            block=block.replace('def '+node.name+'(', 'def '+rename[node.name]+'(',1)
        blocks.append(block)
    return '\n\n'.join(blocks)


def digest(cells):
    return hashlib.sha256('\n'.join(c.source for c in cells).encode()).hexdigest()


def generate(group,filename,datasets,kinds,force=False):
    path=OUTPUT/filename
    if path.exists():
        old=nbf.read(path,as_version=4)
        saved=old.metadata.get('generated_digest')
        if saved and (saved!=digest(old.cells) or any(c.get('outputs') for c in old.cells)) and not force:
            raise RuntimeError('Notebook đã sửa hoặc chạy; giữ lại, không ghi đè: '+str(path))
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text,tag='definition'):
        cell=nbf.v4.new_code_cell(textwrap.dedent(text).strip())
        cell.metadata['tags']=[tag]; cells.append(cell)
    md('# '+{'tabular':'Chương 2 – Học máy trên dữ liệu bảng','image':'Chương 3 – CNN nhận dạng ảnh','sequence':'Chương 4 – RNN đọc chuỗi thời gian'}[group])
    # Ghi rõ quy mô và trạng thái để không hiểu notebook chưa chạy là kết quả thực nghiệm.
    cells[0].source='# '+{'tabular':'Chương 2 – Học máy trên dữ liệu bảng','image':'Chương 3 – CNN nhận dạng ảnh','sequence':'Chương 4 – RNN đọc chuỗi thời gian'}[group]
    md('''## Cách chạy và phạm vi

    Chọn kernel `.venv` (Python 3.12). Lưu notebook bằng Ctrl+S rồi **Run All**.
    Notebook đọc CSV, dùng toàn bộ train trên 100.000 mẫu của từng bộ, ba mô hình và ba cách cài đặt.
    Các ô bên dưới chứa mã tải/đọc dữ liệu, lớp NumPy tự viết, Keras, PyTorch và vòng học thực sự.
    File `.py` chỉ là nguồn ban đầu để sinh notebook; bạn có thể đọc và chỉnh mã trực tiếp tại đây.

    Chưa có kết quả huấn luyện quy mô lớn. Những ô không có output là chưa chạy, không phải lỗi.
    Không dùng số liệu từ thử nghiệm nhỏ `results/tieuluan/course/` để thay thế.
    Thời gian phụ thuộc CPU, dữ liệu và early stopping; không hứa hẹn cố định 4–5 giờ.
    Sau mỗi epoch lưu cả trọng số, Adam và trạng thái ngẫu nhiên; chạy lại để tiếp tục.
    Checkpoint chỉ đọc tệp cục bộ do bạn tạo, không nhận tệp pickle từ bên ngoài.
    ''')
    code('''import os, sys, json, hashlib, time
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'2')
os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/tieuluan/course').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
''','setup')
    code(f'''DATASETS={datasets!r}
MODEL_KINDS={kinds!r}
FRAMEWORKS=['scratch','pytorch','keras']
RUN_ID='large_csv_v1'
CONFIG={{'seed':11,'epochs':{20 if group=='image' else 30},'batch_size':128,'patience':5,'learning_rate':0.001}}
NOTEBOOK_PATH=ROOT/'notebooks/tieuluan/{filename}'
# Lưu Ctrl+S trước khi chạy để mã được ghi nhận đúng vào dấu vết thực nghiệm.
saved_notebook=json.loads(NOTEBOOK_PATH.read_text(encoding='utf8'))
CODE_HASH=hashlib.sha256('\u005cn'.join(''.join(c['source']) for c in saved_notebook['cells'] if c['cell_type']=='code' and any(t in c.get('metadata',{{}}).get('tags',[]) for t in ('definition','train'))).encode()).hexdigest()
print('Mã thực nghiệm:',CODE_HASH[:16], '| Thư mục:',RUN_ID)
''','setup')
    md('''## 1. Tải và đọc dữ liệu CSV

    Nguồn gốc và checksum được lưu trong manifest. CSV là nguồn dữ liệu notebook đọc;
    cache `.npy` chỉ giúp đọc theo mini-batch mà không tốn hàng GB RAM.
    Ảnh: 784 cột `pixel_*` nguyên 0–255. Bảng/chuỗi: đặc trưng đã chuẩn hóa bằng train.
    `sample_id`, `label`, `previous_direction` không được đưa vào đặc trưng.
    Với chuỗi, các cửa sổ trong một tập chồng lấn và không độc lập; ba tập không dùng chung quan sát.
    ''')
    code(source_text('src/tieuluan/course/large_data.py'))
    code(source_text('src/tieuluan/course/csv_data.py'))
    code('''for key in DATASETS:
    csv_manifest=ROOT/'data/tieuluan/course_large/csv'/key/'manifest.json'
    if not csv_manifest.exists():
        prepare_dataset(ROOT,key)
        export_dataset_csv(ROOT,key)
    data,meta=load_csv_dataset(ROOT,key)
    print(meta['name'],meta['source_page'])
    display(pd.DataFrame([{'Tập':s,'Số mẫu':m['n'],'Phân bố lớp':str(m['class_counts'])}
        for s,m in meta['splits'].items()]))
    print(meta.get('task_note',meta.get('split_policy','')))
    assert len(data['train_y'])>100000
    display(pd.read_csv(ROOT/'data/tieuluan/course_large/csv'/key/'train.csv',nrows=5))
    fig,ax=plt.subplots(figsize=(9,3))
    ax.bar(np.arange(len(meta['classes'])),np.bincount(data['train_y']))
    ax.set(title=meta['name']+' – phân bố tập học',xlabel='Mã lớp',ylabel='Số mẫu')
    plt.show()
    if meta['group']=='image':
        fig,axes=plt.subplots(2,5,figsize=(10,4))
        for i,ax in enumerate(axes.flat):
            ax.imshow(data['train_x'][i,0],cmap='gray'); ax.set_title('Nhãn '+str(data['train_y'][i])); ax.axis('off')
        plt.show()
    elif meta['group']=='sequence':
        plt.plot(data['train_x'][0,:,0]); plt.title('Cửa sổ đầu vào đã chuẩn hóa'); plt.show()
    del data
''','data')
    md('''## 2. NumPy tự viết: tham số, lan truyền xuôi và lan truyền ngược

    Các lớp dưới đây có mã trực tiếp, không gọi scikit-learn để huấn luyện bản scratch.
    Conv2d dùng im2col; RNN/LSTM/GRU dùng lan truyền ngược theo thời gian.
    Các lớp không dùng trong chương vẫn được giữ để có cùng nền tảng kiểm chứng.
    ''')
    for file in ('core','layers','recurrent','optim'):
        md('### Mã nền tảng: '+file)
        code(source_text(f'src/tieuluan/scratch/{file}.py'))
    md('## 3. Cùng kiến trúc trên NumPy, Keras và PyTorch')
    code('HIDDEN=8\n'+source_text('src/tieuluan/experiment_models.py',
        selected={'scratch_model','torch_model','keras_model','_scratch_layers_with_weights','copy_weights_to_torch','copy_weights_to_keras'},
        rename={'scratch_model':'recurrent_scratch','torch_model':'recurrent_torch','keras_model':'recurrent_keras'}))
    code(source_text('src/tieuluan/course/models.py'))
    md('''## 4. Kiểm tra trước khi huấn luyện

    So sánh logits trên cùng đầu vào và cùng trọng số. Sai số nhỏ do phép toán số thực;
    nếu kiểm tra thất bại phải sửa trước khi chạy dài. Bước này không huấn luyện toàn bộ dữ liệu.
    ''')
    code('''for key in DATASETS:
    data,meta=load_csv_dataset(ROOT,key)
    sample=batch_input(data['train_x'][:4])
    for kind in MODEL_KINDS:
        models=build_models(kind,tuple(meta['input_shape']),CONFIG['seed'],n_classes=len(meta['classes']))
        reference=logits(models['scratch'],'scratch',sample)
        for fw in ('pytorch','keras'):
            actual=logits(models[fw],fw,sample)
            error=float(np.max(np.abs(reference-actual)))
            np.testing.assert_allclose(reference,actual,atol=2e-5,rtol=1e-4)
            print(key,kind,fw,'– sai số logits:',error)
        del models
    del data
''','check')
    md('''## 5. Vòng học, checkpoint và đánh giá

    Cùng thứ tự mini-batch, Adam và clipping. Chọn epoch bằng validation loss, không bằng test.
    SVM dùng hinge loss và phạt L2, điểm SVM không được gọi là xác suất.
    Một seed là đối chiếu triển khai, chưa đủ kết luận thống kê về thứ hạng các mô hình.
    Khi đổi mã hoặc cấu hình hãy đổi RUN_ID để không lẫn kết quả.
    ''')
    code(source_text('src/tieuluan/course/large_training.py'))
    md('''## 6. Huấn luyện đầy đủ – ô chạy lâu

    Chạy ô này khi máy đã sẵn sàng. Có thể thu hẹp MODEL_KINDS/FRAMEWORKS để chạy từng lượt,
    nhưng phải hoàn thành tất cả tổ hợp trước khi tổng hợp báo cáo. Lượt hoàn thành được bỏ qua;
    lượt bị ngắt tiếp tục từ epoch đã lưu. Số mẫu không bị cắt giảm theo chế độ chạy.
    ''')
    code('''records=[]
for key in DATASETS:
    data,meta=load_csv_dataset(ROOT,key)
    for kind in MODEL_KINDS:
        for framework in FRAMEWORKS:
            result=run_experiment(ROOT,key,kind,framework,data,meta,CONFIG,RUN_ID,CODE_HASH)
            records.append(result)
    del data
display(pd.DataFrame([{'Dữ liệu':r['dataset'],'Mô hình':r['kind'],'Cài đặt':r['framework'],
    'Đúng cân bằng':r['metrics']['balanced_accuracy'],'F1 trung bình':r['metrics']['macro_f1'],
    'Giây huấn luyện':r['training_seconds']} for r in records]))
''','train')
    md('''## 7. Hình, bảng và mã đưa vào báo cáo

    Chỉ chạy sau khi đủ kết quả. Thiếu lượt nào thì công cụ báo thiếu, không điền số giả.
    Đọc các mẫu đoán sai, đường học và so sánh với cách đoán đơn giản trước khi viết nhận xét.
    ''')
    code('''from src.tieuluan.course.large_reporting import export_chapter
report_path=export_chapter(ROOT,RUN_ID,DATASETS,MODEL_KINDS,CONFIG['seed'],NOTEBOOK_PATH)
print('Đã xuất nội dung và tài nguyên báo cáo:',report_path)
for key in DATASETS:
    for kind in MODEL_KINDS:
        fig,ax=plt.subplots(figsize=(7,3))
        for fw in ('scratch','pytorch','keras'):
            path=ROOT/'results/tieuluan/course_large'/RUN_ID/f"{key}__{kind}__{fw}__{CONFIG['seed']}.json"
            item=json.loads(path.read_text(encoding='utf8'))
            ax.plot([p['epoch'] for p in item['history']],[p['val_loss'] for p in item['history']],label=fw)
        ax.set(title=key+' / '+kind,xlabel='Vòng học',ylabel='Mất mát validation'); ax.legend(); plt.show()
''','report')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (.venv)','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},'generated_digest':digest(cells),'training_executed':False})
    nbf.validate(nb); OUTPUT.mkdir(parents=True,exist_ok=True); nbf.write(nb,path)
    print(path.name,len(cells),'ô')


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser(); parser.add_argument('--force',action='store_true'); args=parser.parse_args()
    for group,file,data,kinds in [
        ('tabular','02_ml.ipynb',['covertype','miniboone'],['logistic','svm','mlp']),
        ('image','03_cnn.ipynb',['emnist_digits','k49'],['cnn4','cnn8','cnndeep']),
        ('sequence','04_rnn.ipynb',['jena_10min','household_power'],['rnn','lstm','gru'])]:
        generate(group,file,data,kinds,args.force)

if __name__=='__main__':
    main()
