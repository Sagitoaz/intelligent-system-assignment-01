"""Đọc lại toàn bộ CSV, kiểm tra round-trip, ghi kiểm kê; không huấn luyện."""
import json
from pathlib import Path
import sys
import numpy as np
from src.tieuluan.course.csv_data import load_csv_dataset,SPLIT_FILES
from src.tieuluan.course.large_data import SOURCES

def main():
    sys.stdout.reconfigure(encoding='utf8')
    root=Path(__file__).resolve().parents[2]
    base=root/'data/tieuluan/course_large'
    inventory={}
    for key in SOURCES:
        arrays,meta=load_csv_dataset(root,key)
        for split in SPLIT_FILES:
            expected=np.load(base/'prepared'/key/f'{split}_x.npy',mmap_mode='r')
            for start in range(0,len(expected),5000):
                np.testing.assert_array_equal(arrays[f'{split}_x'][start:start+5000],expected[start:start+5000])
            for field in ('y','ids'):
                np.testing.assert_array_equal(arrays[f'{split}_{field}'],np.load(base/'prepared'/key/f'{split}_{field}.npy'))
        inventory[key]={'name':meta['name'],'source_page':meta['source_page'],'sources':meta['sources'],
            'splits':meta['splits'],'input_shape':meta['input_shape'],'classes':meta['classes'],
            'csv_hashes':meta['csv_hashes'],'csv_bytes':{n:(base/'csv'/key/n).stat().st_size for n in SPLIT_FILES.values()},
            'exact_round_trip':True,'training_executed':False}
        print(key,'– CSV khớp toàn bộ đặc trưng, nhãn và ID; train:',len(arrays['train_y']),flush=True)
        del arrays
    (base/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':
    main()
