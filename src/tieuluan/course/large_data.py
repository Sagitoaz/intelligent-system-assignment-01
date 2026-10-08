"""Dữ liệu quy mô lớn. Các hàm này được chép nguyên mã vào notebook.

Không nhân bản hoặc tăng cường ảnh để đạt ngưỡng 100.000. Không tạo kết quả giả.
Tệp .npy đọc bằng memory mapping, ảnh chỉ đổi sang float32 theo mini-batch.
"""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

SOURCES={
 'covertype':('Thảm phủ rừng Covertype','tabular','https://archive.ics.uci.edu/dataset/31/covertype'),
 'miniboone':('Nhận dạng hạt MiniBooNE','tabular','https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification'),
 'emnist_digits':('Chữ số EMNIST Digits','image','https://www.nist.gov/itl/products-and-services/emnist-dataset'),
 'k49':('Chữ viết Nhật Kuzushiji-49','image','https://github.com/rois-codh/kmnist'),
 'jena_10min':('Nhiệt độ Jena, chu kỳ 10 phút','sequence','https://weather.bgc-jena.mpg.de/weather_data.html'),
 'household_power':('Điện năng hộ gia đình, trung bình 10 phút','sequence','https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption')}


def enforce_size(n):
    if n<=100_000:
        raise ValueError(f'Tập train phải trên 100.000 mẫu thực, hiện có {n:,}. Không tự nhân bản mẫu.')


def batch_input(x):
    return x.astype('float32')/255 if x.dtype==np.uint8 else np.asarray(x,dtype='float32')


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(root,url,name):
    folder=Path(root)/'data/tieuluan/course_large/raw'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/name
    if not path.exists():
        partial=path.with_suffix(path.suffix+'.part')
        print('Đang tải:',name,flush=True)
        with requests.get(url,stream=True,timeout=(15,90),headers={'User-Agent':'Mozilla/5.0'}) as response:
            response.raise_for_status()
            size=0
            with partial.open('wb') as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk); size+=len(chunk)
            expected=response.headers.get('Content-Length')
            if expected and not response.headers.get('Content-Encoding') and size!=int(expected):
                raise ValueError('Tệp tải chưa đủ; chạy lại ô tải dữ liệu.')
        partial.replace(path)
    return path,{'url':url,'sha256':file_hash(path),'bytes':path.stat().st_size}


def tabular_arrays(x,y,seed=11):
    ids=np.arange(len(y))
    train,other=train_test_split(ids,test_size=.2,stratify=y,random_state=seed)
    val,test=train_test_split(other,test_size=.5,stratify=y[other],random_state=seed)
    mean=x[train].mean(0,dtype=np.float64)
    scale=x[train].std(0,dtype=np.float64); scale[scale==0]=1
    arrays={'mean':mean,'scale':scale}
    for s,idx in [('train',train),('val',val),('test',test)]:
        arrays.update({f'{s}_x':((x[idx]-mean)/scale).astype('float32'),f'{s}_y':y[idx],f'{s}_ids':idx})
    arrays['sample_raw']=x[test[:10]]
    return arrays


def temporal_arrays(values,window=24):
    """Chia chuỗi gốc trước khi tạo cửa sổ; không dùng chung quan sát giữa ba tập.

    Cửa sổ có NaN bị bỏ, không nối hai phía khoảng trống. Nhãn: bước sau tăng.
    Các cửa sổ trong một tập có thể chồng lấn, không phải mẫu độc lập thống kê.
    """
    values=np.asarray(values,dtype=np.float64)
    a,b=int(len(values)*.7),int(len(values)*.85)
    mean=float(np.nanmean(values[:a])); scale=float(np.nanstd(values[:a]))
    if not np.isfinite(scale) or scale==0:
        raise ValueError('Chuỗi train không có biến thiên hợp lệ.')
    arrays={'mean':np.array([mean]),'scale':np.array([scale])}
    for split,start,end in [('train',0,a),('val',a,b),('test',b,len(values))]:
        windows=np.lib.stride_tricks.sliding_window_view(values[start:end],window+1)
        valid=np.isfinite(windows).all(axis=1)
        sample=windows[valid]
        arrays[f'{split}_x']=((sample[:,:-1]-mean)/scale).astype('float32')[:,:,None]
        arrays[f'{split}_y']=(sample[:,-1]>sample[:,-2]).astype('int64')
        arrays[f'{split}_ids']=np.arange(start+window,end)[valid]
        arrays[f'{split}_persistence']=(sample[:,-2]>sample[:,-3]).astype('int64')
    return arrays,{'window':window,'split_boundaries':[a,b],
        'missing_rows':int(np.isnan(values).sum()),'label':'Bước tiếp theo tăng so với bước cuối cửa sổ',
        'split_policy':'70/15/15 theo thời gian; tạo cửa sổ riêng trong mỗi tập, bỏ mọi cửa sổ chứa thiếu dữ liệu'}


def image_arrays(x,y,xt,yt):
    train,val=train_test_split(np.arange(len(y)),test_size=.1,stratify=y,random_state=11)
    arrays={}
    for s,images,labels,ids in [('train',x[train],y[train],train),('val',x[val],y[val],val),
                               ('test',xt,yt,np.arange(len(yt)))]:
        arrays.update({f'{s}_x':images[:,None].astype('uint8'),f'{s}_y':labels.astype('int64'),f'{s}_ids':ids})
    return arrays


def prepare_dataset(root,key):
    root=Path(root)
    target=root/'data/tieuluan/course_large/prepared'/key
    if (target/'manifest.json').exists():
        meta=json.loads((target/'manifest.json').read_text(encoding='utf8'))
        for name,digest in meta['array_hashes'].items():
            if file_hash(target/name)!=digest:
                raise ValueError('Tài nguyên đã thay đổi: '+name)
        enforce_size(meta['splits']['train']['n'])
        return meta
    sources=[]; extra={}; classes=['Không','Có']
    if key=='covertype':
        path,source=fetch_file(root,'https://archive.ics.uci.edu/static/public/31/covertype.zip','covertype.zip')
        sources.append(source)
        with zipfile.ZipFile(path) as archive:
            with gzip.GzipFile(fileobj=archive.open('covtype.data.gz')) as handle:
                frame=pd.read_csv(handle,header=None)
        x=frame.iloc[:,:-1].to_numpy(dtype='float32')
        y=(frame.iloc[:,-1].to_numpy()==2).astype('int64')
        arrays=tabular_arrays(x,y)
        classes=['Loại rừng khác','Thông lodgepole (lớp gốc 2)']
        extra={'original_rows':len(frame),'task_note':'Bài toán nhị phân lớp 2 so với sáu lớp còn lại; không phải benchmark bảy lớp gốc. Chia ngẫu nhiên chưa chứng minh tổng quát sang vùng địa lý mới.'}
    elif key=='miniboone':
        path,source=fetch_file(root,'https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip','miniboone.zip')
        sources.append(source)
        with zipfile.ZipFile(path) as archive:
            with archive.open('MiniBooNE_PID.txt') as handle:
                signal,background=map(int,handle.readline().split())
                x=np.loadtxt(handle,dtype='float32')
        assert len(x)==signal+background
        y=np.r_[np.ones(signal,dtype='int64'),np.zeros(background,dtype='int64')]
        # Loại bản ghi toàn -999 (không có phép đo hữu ích), công bố số bị loại.
        valid=np.isfinite(x).all(axis=1)&~np.all(x==-999,axis=1)
        arrays=tabular_arrays(x[valid],y[valid])
        classes=['Nền muon','Tín hiệu electron']
        extra={'original_rows':len(x),'removed_rows':int((~valid).sum()),'task_note':'50 phép đo của sự kiện hạt; nhãn suy ra đúng thứ tự và số lượng ở dòng đầu tệp gốc.'}
    elif key=='emnist_digits':
        path,source=fetch_file(root,'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip','emnist-gzip.zip')
        sources.append(source); values=[]
        with zipfile.ZipFile(path) as archive:
            for split in ('train','test'):
                for field in ('images','labels'):
                    suffix='idx3' if field=='images' else 'idx1'
                    name=f'emnist-digits-{split}-{field}-{suffix}-ubyte.gz'
                    member=next(n for n in archive.namelist() if n.endswith(name))
                    raw=gzip.decompress(archive.read(member))
                    arr=np.frombuffer(raw,dtype='uint8',offset=16 if field=='images' else 8)
                    if field=='images':
                        arr=arr.reshape(-1,28,28).transpose(0,2,1).copy()
                    values.append(arr)
        arrays=image_arrays(*values); classes=[str(i) for i in range(10)]
        extra={'original_rows':len(values[1])+len(values[3]),'official_train':len(values[1]),'official_test':len(values[3]),'task_note':'Giữ tập test chính thức; lấy 10% train chính thức làm validation; chuyển vị ảnh để hiển thị đúng chiều.'}
    elif key=='k49':
        values=[]
        for name in ('k49-train-imgs.npz','k49-train-labels.npz','k49-test-imgs.npz','k49-test-labels.npz'):
            path,source=fetch_file(root,'https://codh.rois.ac.jp/kmnist/dataset/k49/'+name,name)
            sources.append(source)
            with np.load(path) as data:
                values.append(data['arr_0'])
        arrays=image_arrays(*values); classes=[f'Ký tự {i}' for i in range(49)]
        extra={'original_rows':len(values[1])+len(values[3]),'official_train':len(values[1]),'official_test':len(values[3]),'task_note':'49 lớp mất cân bằng; giữ mã nhãn gốc 0–48. Ghi công CODH/NIJL, giấy phép CC BY-SA 4.0. Không phải KMNIST 10 lớp.'}
    elif key in ('jena_10min','household_power'):
        if key=='jena_10min':
            path,source=fetch_file(root,'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip','jena.zip')
            with zipfile.ZipFile(path) as archive:
                frame=pd.read_csv(archive.open('jena_climate_2009_2016.csv'))
            dates=pd.to_datetime(frame['Date Time'],format='%d.%m.%Y %H:%M:%S')
            series=pd.Series(frame['T (degC)'].to_numpy(),index=dates).sort_index()
            series=series.groupby(level=0).mean().resample('10min').mean()
            series[series<=-999]=np.nan
        else:
            path,source=fetch_file(root,'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip','household.zip')
            with zipfile.ZipFile(path) as archive:
                frame=pd.read_csv(archive.open('household_power_consumption.txt'),sep=';',na_values=['?'],
                    usecols=['Date','Time','Global_active_power'])
            dates=pd.to_datetime(frame['Date']+' '+frame['Time'],format='%d/%m/%Y %H:%M:%S')
            raw=pd.Series(frame['Global_active_power'].to_numpy(dtype=float),index=dates)
            # Chỉ dùng khoảng có đủ 10 quan trắc; không điền bằng dữ liệu tương lai.
            series=raw.resample('10min').mean()
            series[raw.resample('10min').count()<10]=np.nan
        sources.append(source)
        arrays,extra=temporal_arrays(series.to_numpy())
        arrays['raw_values']=series.to_numpy(); arrays['raw_dates']=series.index.to_numpy(dtype='datetime64[s]')
        extra.update({'original_rows':len(frame),'regular_rows':len(series),'interval_minutes':10})
        classes=['Không tăng','Tăng']
    else:
        raise ValueError('Bộ dữ liệu không được hỗ trợ: '+key)
    enforce_size(len(arrays['train_y']))
    target.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,value in arrays.items():
        np.save(target/f'{name}.npy',value,allow_pickle=False)
        hashes[name+'.npy']=file_hash(target/f'{name}.npy')
    name,group,page=SOURCES[key]
    meta={'key':key,'name':name,'group':group,'source_page':page,'sources':sources,
        'classes':classes,'input_shape':list(arrays['train_x'].shape[1:]),'array_hashes':hashes,**extra,
        'splits':{s:{'n':len(arrays[f'{s}_y']),'class_counts':np.bincount(arrays[f'{s}_y'],minlength=len(classes)).tolist()} for s in ('train','val','test')}}
    (target/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    return meta


def load_dataset(root,key):
    folder=Path(root)/'data/tieuluan/course_large/prepared'/key
    meta=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    enforce_size(meta['splits']['train']['n'])
    arrays={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in folder.glob('*.npy')}
    return arrays,meta
