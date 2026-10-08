"""python -m scripts.tieuluan.export_large_csv [--datasets ...]. Không huấn luyện."""
import argparse
from pathlib import Path
import sys
from src.tieuluan.course.large_data import SOURCES,prepare_dataset
from src.tieuluan.course.csv_data import export_dataset_csv

def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=list(SOURCES))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    for key in args.datasets:
        prepare_dataset(root,key)
        export_dataset_csv(root,key)

if __name__=='__main__':
    main()
