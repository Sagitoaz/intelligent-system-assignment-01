"""Chỉ tải/kiểm tra dữ liệu, không huấn luyện: python -m scripts.tieuluan.prepare_large_data."""
import argparse
import sys
from pathlib import Path
from src.tieuluan.course.large_data import SOURCES,prepare_dataset

def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=list(SOURCES))
    args=parser.parse_args()
    for key in args.datasets:
        meta=prepare_dataset(Path(__file__).resolve().parents[2],key)
        print(key,{s:m['n'] for s,m in meta['splits'].items()},flush=True)

if __name__=='__main__':
    main()
