"""Xuất CSV và đọc CSV theo khối; bộ nhớ đệm không thay thế nguồn CSV.

Ảnh dùng 784 cột điểm ảnh uint8. Bảng và cửa sổ chuỗi giữ số float32 đã
chuẩn hóa bằng tập train. Thứ tự nhãn/định danh được giữ nguyên.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.tieuluan.course.large_data import file_hash,enforce_size

SPLIT_FILES={'train':'train.csv','val':'validation.csv','test':'test.csv'}


def export_dataset_csv(root,key,overwrite=False):
    base=Path(root)/'data/tieuluan/course_large'
    source=base/'prepared'/key
    target=base/'csv'/key
    meta=json.loads((source/'manifest.json').read_text(encoding='utf8'))
    if (target/'manifest.json').exists() and not overwrite:
        return json.loads((target/'manifest.json').read_text(encoding='utf8'))
    target.mkdir(parents=True,exist_ok=True)
    width=int(np.prod(meta['input_shape']))
    prefix={'image':'pixel','tabular':'feature','sequence':'step'}[meta['group']]
    names=[f'{prefix}_{i:03d}' for i in range(width)]
    hashes={}
    for split,name in SPLIT_FILES.items():
        x=np.load(source/f'{split}_x.npy',mmap_mode='r')
        y=np.load(source/f'{split}_y.npy',mmap_mode='r')
        ids=np.load(source/f'{split}_ids.npy',mmap_mode='r')
        persistence_path=source/f'{split}_persistence.npy'
        persistence=np.load(persistence_path,mmap_mode='r') if persistence_path.exists() else None
        temporary=target/(name+'.part')
        with temporary.open('w',encoding='utf8',newline='') as handle:
            for start in range(0,len(y),5000):
                end=min(len(y),start+5000)
                frame=pd.DataFrame(x[start:end].reshape(end-start,-1),columns=names)
                frame.insert(0,'sample_id',ids[start:end])
                frame['label']=y[start:end]
                if persistence is not None:
                    frame['previous_direction']=persistence[start:end]
                frame.to_csv(handle,index=False,header=start==0,float_format='%.9g')
        temporary.replace(target/name)
        hashes[name]=file_hash(target/name)
        print(key,name,len(y),flush=True)
    if (source/'raw_dates.npy').exists():
        pd.DataFrame({'timestamp':np.load(source/'raw_dates.npy'),
                      'value':np.load(source/'raw_values.npy')}).to_csv(target/'observations.csv',index=False)
    description=('Mỗi ảnh 28×28 được trải theo hàng thành 784 số nguyên 0–255; label là lớp gốc.'
        if meta['group']=='image' else 'Các cột đặc trưng đã chuẩn hóa bằng thống kê chỉ từ train; không chuẩn hóa lại trên toàn bộ CSV.')
    meta.update({'csv_features':names,'csv_hashes':hashes,'csv_description':description,
        'csv_extra_columns':['sample_id','label','previous_direction'],
        'csv_id_note':'ID của test ảnh thuộc tập test chính thức riêng; số ID có thể trùng số ID train nhưng không cùng ảnh.'})
    (target/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    (target/'README.md').write_text('# '+meta['name']+'\n\n'+description+
        '\n\n`sample_id` và `label` không được đưa vào đặc trưng. `previous_direction`, nếu có, chỉ dùng cho đường cơ sở. '+
        'Các cột dùng để học được liệt kê trong manifest.json. CSV dùng dấu chấm thập phân và dấu phẩy ngăn cột theo quy ước Python.\n'+
        '\nNguồn gốc: '+meta.get('source_page','Dữ liệu kiểm thử')+'\n',encoding='utf8')
    return meta


def load_csv_dataset(root,key,require_large=True):
    """Đọc CSV thật; đổi nội dung CSV sẽ đổi hash và làm mới cache NumPy.

    Memory mapping chỉ là bộ đệm để không nạp hàng trăm nghìn ảnh vào RAM.
    Không đọc các đặc trưng từ thư mục prepared/.
    """
    base=Path(root)/'data/tieuluan/course_large'
    folder=base/'csv'/key
    meta=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    hashes={name:file_hash(folder/name) for name in SPLIT_FILES.values()}
    identity=hashlib.sha256(json.dumps({'files':hashes,'shape':meta['input_shape'],
        'features':meta['csv_features']},sort_keys=True).encode()).hexdigest()[:20]
    cache=base/'csv_cache'/key/identity
    cache.mkdir(parents=True,exist_ok=True)
    marker=cache/'hashes.json'
    current=json.loads(marker.read_text()) if marker.exists() else {}
    names=meta['csv_features']; shape=tuple(meta['input_shape'])
    dtype='uint8' if meta['group']=='image' else 'float32'
    if hashes!=current:
        for split,name in SPLIT_FILES.items():
            n=meta['splits'][split]['n']
            x=np.lib.format.open_memmap(cache/f'{split}_x.npy',mode='w+',dtype=dtype,shape=(n,*shape))
            y=np.lib.format.open_memmap(cache/f'{split}_y.npy',mode='w+',dtype='int64',shape=(n,))
            ids=np.lib.format.open_memmap(cache/f'{split}_ids.npy',mode='w+',dtype='int64',shape=(n,))
            offset=0; persistence=None
            # Float64 trước khi kiểm tra ngăn số ngoài 0–255 bị cuộn vòng uint8.
            for chunk in pd.read_csv(folder/name,chunksize=5000):
                end=offset+len(chunk)
                if end>n:
                    raise ValueError('Số dòng CSV khác manifest: '+name)
                values=chunk[names].to_numpy(dtype='float64')
                labels=chunk['label'].to_numpy(dtype='float64')
                if not np.isfinite(values).all() or not np.isfinite(labels).all():
                    raise ValueError('CSV chứa đặc trưng/nhãn thiếu hoặc vô hạn.')
                if np.any(labels!=np.floor(labels)) or np.any(labels<0) or np.any(labels>=len(meta['classes'])):
                    raise ValueError('Nhãn CSV ngoài danh mục lớp.')
                if dtype=='uint8' and (np.any(values<0) or np.any(values>255) or np.any(values!=np.floor(values))):
                    raise ValueError('Điểm ảnh CSV phải là số nguyên từ 0 đến 255.')
                x[offset:end]=values.reshape(-1,*shape)
                y[offset:end]=labels.astype('int64')
                ids[offset:end]=chunk['sample_id'].to_numpy(dtype='int64')
                if 'previous_direction' in chunk:
                    if persistence is None:
                        persistence=np.lib.format.open_memmap(cache/f'{split}_persistence.npy',mode='w+',dtype='int64',shape=(n,))
                    persistence[offset:end]=chunk['previous_direction'].to_numpy(dtype='int64')
                offset=end
            if offset!=n:
                raise ValueError('Số dòng CSV khác manifest: '+name)
            x.flush(); y.flush(); ids.flush()
            if persistence is not None:
                persistence.flush()
            del x,y,ids,persistence
        marker.write_text(json.dumps(hashes,sort_keys=True),encoding='utf8')
    arrays={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in cache.glob('*.npy')}
    if require_large:
        enforce_size(len(arrays['train_y']))
    meta['csv_hashes']=hashes
    # Signature thực nghiệm phải theo CSV đã đọc, không theo cache cũ.
    meta['array_hashes']=hashes
    return arrays,meta
