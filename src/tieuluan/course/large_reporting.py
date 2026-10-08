"""Xuất hình/bảng và mã thật từ notebook sau khi người dùng huấn luyện đủ."""
import ast
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.tieuluan.course.large_data import file_hash


def export_chapter(root,run_id,datasets,kinds,seed,notebook_path):
    root=Path(root)
    result_folder=root/'results/tieuluan/course_large'/run_id
    groups={2:(['covertype','miniboone'],['logistic','svm','mlp']),
            3:(['emnist_digits','k49'],['cnn4','cnn8','cnndeep']),
            4:(['jena_10min','household_power'],['rnn','lstm','gru'])}
    chapter=next((n for n,(ds,_) in groups.items() if datasets and datasets[0] in ds),None)
    if chapter is None or set(datasets)!=set(groups[chapter][0]) or set(kinds)!=set(groups[chapter][1]):
        raise ValueError('Xuất báo cáo cần chọn đủ hai bộ dữ liệu và ba mô hình của chương; có thể huấn luyện từng phần trước.')
    target=root/'docs/tieuluan/course_large/generated'/run_id
    figures=root/'figures/tieuluan/course_large'/run_id
    expected=[f'{ds}__{kind}__{fw}__{seed}' for ds in datasets for kind in kinds for fw in ('scratch','pytorch','keras')]
    missing=[name for name in expected if not(result_folder/f'{name}.json').exists()]
    if missing:
        raise ValueError('Chưa đủ lượt huấn luyện, không sinh kết luận: '+', '.join(missing))
    records=[json.loads((result_folder/f'{name}.json').read_text(encoding='utf8')) for name in expected]
    notebook=json.loads(Path(notebook_path).read_text(encoding='utf8'))
    current_hash=hashlib.sha256('\n'.join(''.join(c['source']) for c in notebook['cells']
        if c['cell_type']=='code' and any(t in c.get('metadata',{}).get('tags',[]) for t in ('definition','train'))).encode()).hexdigest()
    if any(r['code_hash']!=current_hash for r in records):
        raise ValueError('Notebook hiện tại khác mã đã huấn luyện; không trích mã sai vào báo cáo.')
    if any(r['metadata']['splits']['train']['n']<=100000 for r in records):
        raise ValueError('Có lượt huấn luyện không đạt trên 100.000 mẫu.')
    if len({r['code_hash'] for r in records})!=1:
        raise ValueError('Các lượt trong chương không cùng mã notebook; cần thống nhất trước khi so sánh.')
    if len({json.dumps(r['config'],sort_keys=True) for r in records})!=1:
        raise ValueError('Các lượt trong chương khác cấu hình huấn luyện; không trộn thành một bảng.')
    for ds in datasets:
        subset=[r for r in records if r['dataset']==ds]
        if len({json.dumps(r['metadata']['array_hashes'],sort_keys=True) for r in subset})!=1:
            raise ValueError('Các lượt của '+ds+' dùng CSV khác nhau.')
        for name,digest in subset[0]['metadata']['array_hashes'].items():
            if file_hash(root/'data/tieuluan/course_large/csv'/ds/name)!=digest:
                raise ValueError('CSV hiện tại khác dữ liệu đã huấn luyện: '+ds+'/'+name)
    target.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    rows=[{'Dữ liệu':r['metadata']['name'],'Mô hình':r['kind'],'Cài đặt':r['framework'],
           'Accuracy':r['metrics']['accuracy'],'Balanced accuracy':r['metrics']['balanced_accuracy'],
           'Macro F1':r['metrics']['macro_f1'],'ROC-AUC':r['metrics'].get('roc_auc'),
           'Epoch tốt nhất':r['best_epoch'],'Thời gian (giây)':r['training_seconds']} for r in records]
    table=pd.DataFrame(rows)
    table.to_csv(target/f'chapter_{chapter}_results.csv',index=False,encoding='utf-8-sig')
    text=[f'# TÀI NGUYÊN THỰC NGHIỆM CHƯƠNG {chapter}',
        'Số liệu bên dưới đọc từ kết quả chạy thật. Đây là phần thực nghiệm để tích hợp vào báo cáo, không phải báo cáo hoàn chỉnh.',
        'Một seed dùng để đối chiếu cách cài đặt; chưa đủ kết luận thứ hạng mô hình có ý nghĩa thống kê.']
    for ds in datasets:
        subset=[r for r in records if r['dataset']==ds]
        meta=subset[0]['metadata']
        text+=['## '+meta['name'],'Nguồn: '+meta['source_page'],meta.get('task_note',meta.get('split_policy','')),
               '| Tập | Số mẫu | Phân bố lớp |','|---|---:|---|']
        text += [f"| {s} | {v['n']:,} | {v['class_counts']} |" for s,v in meta['splits'].items()]
        fig,axes=plt.subplots(1,len(kinds),figsize=(12,3.5))
        for ax,kind in zip(np.atleast_1d(axes),kinds):
            for r in subset:
                if r['kind']==kind:
                    ax.plot([v['epoch'] for v in r['history']],[v['val_loss'] for v in r['history']],label=r['framework'])
            ax.set(title=kind,xlabel='Vòng học',ylabel='Mất mát validation'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figures/f'{ds}_learning.png',dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,len(kinds),figsize=(12,3.6))
        for ax,kind in zip(np.atleast_1d(axes),kinds):
            r=next(r for r in subset if r['kind']==kind and r['framework']=='scratch')
            matrix=np.asarray(r['metrics']['confusion_matrix'])
            ax.imshow(matrix,cmap='Blues'); ax.set(title=kind,xlabel='Nhãn dự đoán',ylabel='Nhãn thật')
            if len(matrix)<=10:
                for index in np.ndindex(matrix.shape):
                    ax.text(index[1],index[0],str(matrix[index]),ha='center',fontsize=7)
        fig.tight_layout(); fig.savefig(figures/f'{ds}_confusion.png',dpi=160); plt.close(fig)
        text+=['Đường học: '+str((figures/f'{ds}_learning.png').relative_to(root)),
               'Ma trận nhầm lẫn: '+str((figures/f'{ds}_confusion.png').relative_to(root)),
               'Cần nhận xét sau khi xem hình: mô hình có quá khớp không, lớp nào dễ nhầm, có vượt đường cơ sở không?']
    text+=['## Bảng kết quả','| '+' | '.join(table.columns)+' |','|'+'---|'*len(table.columns)]
    for row in table.itertuples(index=False,name=None):
        text.append('| '+' | '.join(f'{v:.4f}'.replace('.',',') if isinstance(v,float) else str(v) for v in row)+' |')
    notebook=json.loads(Path(notebook_path).read_text(encoding='utf8'))
    text+=['## Mã cốt lõi trích từ notebook đã chạy']
    wanted={'scratch_model','torch_model','keras_model','loss_and_gradient','fit_large','temporal_arrays','tabular_arrays','image_arrays'}
    for cell in notebook['cells']:
        if cell['cell_type']!='code': continue
        source=''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef) and node.name in wanted:
                text += ['### '+node.name,'```python',ast.get_source_segment(source,node),'```']
    output=target/f'chapter_{chapter}_experiment.md'
    output.write_text('\n\n'.join(text),encoding='utf8')
    return output
