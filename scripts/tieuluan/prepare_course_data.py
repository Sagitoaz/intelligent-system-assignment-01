"""Tải và chia dữ liệu: python -m scripts.tieuluan.prepare_course_data.

Không sử dụng dữ liệu Assignment khác. Tệp nguồn được tải từ nhà cung cấp,
ghi SHA-256 và URL. Chỉ dữ liệu đã chuẩn bị nhỏ được đưa vào Git.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile
import sys
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/tieuluan/course'
RAW=OUT/'raw'
SEED=11


def download(url, name):
    RAW.mkdir(parents=True,exist_ok=True)
    path=RAW/name
    if not path.exists():
        response=requests.get(url,timeout=90,headers={'User-Agent':'Mozilla/5.0'})
        response.raise_for_status()
        path.write_bytes(response.content)
    return path, {'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'bytes':path.stat().st_size}


def save(key, arrays, meta):
    np.savez_compressed(OUT/f'{key}.npz',**arrays)
    meta['splits']={s:{'n':len(arrays[f'{s}_y']),
        'class_counts':np.bincount(arrays[f'{s}_y'].astype(int)).tolist()}
        for s in ('train','val','test')}
    meta['input_shape']=list(arrays['train_x'].shape[1:])
    return meta


def tabular():
    path,source=download('https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip','breast_cancer.zip')
    with zipfile.ZipFile(path) as archive:
        frame=pd.read_csv(archive.open('wdbc.data'),header=None)
    x=frame.iloc[:,2:].to_numpy(float)
    y=(frame.iloc[:,1]=='M').to_numpy(int)
    train,rest=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=SEED)
    val,test=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=SEED)
    mean,scale=x[train].mean(0),x[train].std(0)
    scale[scale==0]=1
    arrays={'mean':mean,'scale':scale}
    for split,ids in [('train',train),('val',val),('test',test)]:
        arrays.update({f'{split}_x':((x[ids]-mean)/scale).astype('float32'),
                       f'{split}_y':y[ids],f'{split}_ids':ids,f'{split}_raw':x[ids]})
    meta={'breast_cancer':save('breast_cancer',arrays,{'group':'tabular','name':'Tế bào vú Wisconsin',
        'sources':[source],'original_n':len(y),'classes':['Lành tính','Ác tính'],
        'description':'30 phép đo hình thái tế bào; nhãn ác tính bằng 1. Không dùng làm công cụ chẩn đoán.'})}
    with np.load(ROOT/'data/tieuluan/processed/credit_default.npz',allow_pickle=False) as data:
        arrays={f'{s}_{new}':data[f'{s}_{old}'] for s in ('train','val','test')
                for new,old in [('x','tabular'),('y','y'),('ids','ids')]}
        arrays.update({k:data[k] for k in ('feature_names','scaler_mean','scaler_scale','imputer_median')})
    meta['credit_default']=save('credit_default',arrays,{'group':'tabular','name':'Vỡ nợ thẻ tín dụng',
        'original_n':30000,'sources':[{'url':'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients',
        'snapshot':'data/tieuluan/processed/credit_default.npz',
        'sha256':hashlib.sha256((ROOT/'data/tieuluan/processed/credit_default.npz').read_bytes()).hexdigest()}],
        'classes':['Không vỡ nợ','Vỡ nợ'],'description':'Giữ nguyên phép chia và chuẩn hóa đã kiểm toán của bản trước.'})
    return meta


def images():
    meta={}
    for key in ('mnist','fashion_mnist'):
        sources=[]
        if key=='mnist':
            path,source=download('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz','mnist.npz')
            sources.append(source)
            with np.load(path) as data:
                x,y,xt,yt=[data[k] for k in ('x_train','y_train','x_test','y_test')]
        else:
            values=[]
            for name in ('train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'):
                path,source=download('https://storage.googleapis.com/tensorflow/tf-keras-datasets/'+name,name)
                sources.append(source)
                payload=gzip.decompress(path.read_bytes())
                value=np.frombuffer(payload,dtype=np.uint8,offset=16 if 'images' in name else 8)
                values.append(value.reshape(-1,28,28) if 'images' in name else value)
            x,y,xt,yt=values
        chosen,_=train_test_split(np.arange(len(y)),train_size=8000,stratify=y,random_state=SEED)
        train,val=train_test_split(chosen,test_size=2000,stratify=y[chosen],random_state=SEED)
        arrays={}
        for s,xx,yy,ids in [('train',x[train],y[train],train),('val',x[val],y[val],val),
                             ('test',xt,yt,np.arange(len(yt)))]:
            arrays.update({f'{s}_x':xx[:,None].astype('float32')/255,
                           f'{s}_y':yy.astype('int64'),f'{s}_ids':ids})
        meta[key]=save(key,arrays,{'group':'image','name':'MNIST' if key=='mnist' else 'Fashion-MNIST',
            'sources':sources,'original_n':len(y)+len(yt),'official_train_n':len(y),'official_test_n':len(yt),
            'classes':[str(i) for i in range(10)] if key=='mnist' else ['Áo phông','Quần dài','Áo chui đầu','Váy','Áo khoác','Dép','Áo sơ mi','Giày thể thao','Túi','Bốt'],
            'description':'Lấy mẫu phân tầng 6.000 ảnh học và 2.000 ảnh validation từ tập train chính thức; dùng toàn bộ 10.000 ảnh test, không trộn hai tập gốc.'})
    return meta


def sequences():
    path,source=download('https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip','jena.zip')
    with zipfile.ZipFile(path) as archive:
        frame=pd.read_csv(archive.open('jena_climate_2009_2016.csv'))
    series=pd.Series(frame['T (degC)'].to_numpy(),index=pd.to_datetime(frame['Date Time'],format='%d.%m.%Y %H:%M:%S'))
    daily=series.resample('D').mean().dropna()
    # Loại ngày biên không đủ 144 quan trắc 10 phút.
    daily=daily[series.resample('D').count().reindex(daily.index)>=144]
    raw=daily.to_numpy()
    dates=daily.index.strftime('%Y-%m-%d').to_numpy()
    x=np.array([raw[i-20:i] for i in range(20,len(raw))])[:,:,None]
    y=(raw[20:]>raw[19:-1]).astype('int64')
    targets=dates[20:]
    masks={'train':targets<='2013-12-31','val':(targets>'2013-12-31')&(targets<='2014-12-31'),
           'test':targets>'2014-12-31'}
    mean=x[masks['train']].mean(); scale=x[masks['train']].std()
    arrays={'mean':np.array([mean]),'scale':np.array([scale]),'raw_values':raw,'raw_dates':dates.astype('U10')}
    for s,mask in masks.items():
        arrays.update({f'{s}_x':((x[mask]-mean)/scale).astype('float32'),f'{s}_y':y[mask],
                       f'{s}_ids':targets[mask].astype('U10'),f'{s}_raw':x[mask],
                       f'{s}_persistence':(x[mask,-1,0]>x[mask,-2,0]).astype(int)})
    meta={'jena':save('jena',arrays,{'group':'sequence','name':'Nhiệt độ Jena',
        'sources':[source,{'url':'https://weather.bgc-jena.mpg.de/weather_data.html'}],
        'original_n':len(frame),'daily_n':len(daily),'classes':['Không tăng','Tăng'],
        'description':'Trung bình ngày từ quan trắc 10 phút, bỏ ngày biên thiếu quan trắc; dùng 20 ngày để đoán ngày kế tiếp ấm hơn hay không. Train đến 2013, validation 2014, test 2015–2016.'})}
    with np.load(ROOT/'data/tieuluan/processed/vnindex.npz') as data:
        arrays={f'{s}_{new}':data[f'{s}_{old}'] for s in ('train','val','test')
                for new,old in [('x','sequence'),('y','y'),('ids','target_date')]}
        arrays.update({k:data[k] for k in ('scaler_mean','scaler_scale')})
    for s in ('train','val','test'):
        arrays[f'{s}_persistence']=(arrays[f'{s}_x'][:,-1,0]*arrays['scaler_scale'][0]+arrays['scaler_mean'][0]>0).astype(int)
    meta['vnindex']=save('vnindex',arrays,{'group':'sequence','name':'VN-Index',
        'original_n':4175,'classes':['Không tăng','Tăng'],
        'sources':[{'url':'https://iboard.ssi.com.vn/','snapshot':'data/tieuluan/processed/vnindex.npz',
        'sha256':hashlib.sha256((ROOT/'data/tieuluan/processed/vnindex.npz').read_bytes()).hexdigest()}],
        'description':'20 lợi suất logarit dự đoán hướng phiên tiếp theo; giữ nguyên chia thời gian, train đến 2019, validation 2020–2022, test 2023–30/09/2026.'})
    return meta


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={}
    for prepare in (tabular,images,sequences):
        manifest.update(prepare())
        print('Đã chuẩn bị:', ', '.join(manifest),flush=True)
    for key,meta in manifest.items():
        meta['prepared_sha256']=hashlib.sha256((OUT/f'{key}.npz').read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
