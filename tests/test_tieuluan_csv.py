"""CSV là nguồn notebook đọc; kiểm thử không gọi mạng."""
import json
import numpy as np
from src.tieuluan.course.csv_data import export_dataset_csv, load_csv_dataset


def test_csv_round_trip_and_edit_is_used(tmp_path):
    folder=tmp_path/'data/tieuluan/course_large/prepared/example'
    folder.mkdir(parents=True)
    meta={'key':'example','name':'Ảnh kiểm thử','group':'image','input_shape':[1,2,2],
          'classes':['Không','Có'],'splits':{s:{'n':2} for s in ('train','val','test')}}
    x=np.array([[[[0,255],[128,1]]],[[[2,3],[4,5]]]],dtype='uint8')
    for split in ('train','val','test'):
        np.save(folder/f'{split}_x.npy',x)
        np.save(folder/f'{split}_y.npy',[0,1])
        np.save(folder/f'{split}_ids.npy',[10,20])
    (folder/'manifest.json').write_text(json.dumps(meta),encoding='utf8')
    export_dataset_csv(tmp_path,'example')
    csv_folder=tmp_path/'data/tieuluan/course_large/csv/example'
    assert (csv_folder/'train.csv').read_text().splitlines()[0]=='sample_id,pixel_000,pixel_001,pixel_002,pixel_003,label'
    data,loaded=load_csv_dataset(tmp_path,'example',require_large=False)
    np.testing.assert_array_equal(data['train_x'],x)
    np.testing.assert_array_equal(data['test_y'],[0,1])
    assert loaded['csv_hashes']['train.csv']
    text=(csv_folder/'train.csv').read_text().replace('10,0,255,128,1,0','10,7,255,128,1,0')
    (csv_folder/'train.csv').write_text(text)
    data,changed=load_csv_dataset(tmp_path,'example',require_large=False)
    assert data['train_x'][0,0,0,0]==7
    assert changed['csv_hashes']!=loaded['csv_hashes']


def test_float_csv_round_trip(tmp_path):
    folder=tmp_path/'data/tieuluan/course_large/prepared/example'
    folder.mkdir(parents=True)
    meta={'key':'example','name':'Bảng kiểm thử','group':'tabular','input_shape':[2],
          'classes':['Không','Có'],'splits':{s:{'n':2} for s in ('train','val','test')}}
    x=np.array([[.12345678,-13.456],[-.0000456,3.4]],dtype='float32')
    for split in ('train','val','test'):
        np.save(folder/f'{split}_x.npy',x)
        np.save(folder/f'{split}_y.npy',[0,1])
        np.save(folder/f'{split}_ids.npy',[1,2])
    (folder/'manifest.json').write_text(json.dumps(meta),encoding='utf8')
    export_dataset_csv(tmp_path,'example')
    arrays,_=load_csv_dataset(tmp_path,'example',require_large=False)
    np.testing.assert_array_equal(arrays['train_x'],x)
