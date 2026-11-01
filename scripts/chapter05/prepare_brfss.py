"""Chuẩn bị bộ dữ liệu thống kê tiểu đường cho Chương 5/7 từ CDC BRFSS 2023.

Nguồn : CDC, Behavioral Risk Factor Surveillance System (BRFSS) 2023, tệp LLCP2023XPT.zip
        https://www.cdc.gov/brfss/annual_data/annual_2023.html
Giấy phép: dữ liệu của chính phủ Hoa Kỳ, công khai (public domain), được dùng tự do cho nghiên cứu.
Vì sao 2023: BRFSS 2024 đã phát hành nhưng bỏ các câu hỏi huyết áp cao và cholesterol cao
        (chỉ hỏi ở các năm lẻ), nên 2023 là năm mới nhất có đủ các yếu tố nguy cơ.
Kết quả: data/chapter05/brfss2023_diabetes_150k.csv (150.000 dòng, mỗi dòng một người trả lời hợp lệ)
         data/chapter05/brfss2023_manifest.json

Chạy: .venv\\Scripts\\python.exe scripts/chapter05/prepare_brfss.py
"""
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/chapter05'
URL = 'https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip'
N = 150_000
SEED = 2023

STATES = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California', 8: 'Colorado', 9: 'Connecticut',
          10: 'Delaware', 11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho',
          17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine',
          24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri',
          30: 'Montana', 31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico',
          36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon',
          42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee',
          48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia',
          55: 'Wisconsin', 56: 'Wyoming', 66: 'Guam', 72: 'Puerto Rico', 78: 'Virgin Islands'}
AGE = {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64',
       10: '65-69', 11: '70-74', 12: '75-79', 13: '80+'}
DIAB = {1: 'yes', 2: 'gestational_only', 3: 'no', 4: 'prediabetes'}
SMOKER = {1: 'daily', 2: 'some_days', 3: 'former', 4: 'never'}
INCOME = {1: '<15k', 2: '15-25k', 3: '25-35k', 4: '35-50k', 5: '50-100k', 6: '100-200k', 7: '200k+'}
EDU = {1: 'less_than_high_school', 2: 'high_school', 3: 'some_college', 4: 'college_graduate'}


def load_raw():
    cache = Path(tempfile.gettempdir()) / 'brfss2023_raw'
    cache.mkdir(exist_ok=True)
    xpt = cache / 'LLCP2023.XPT'
    if not xpt.exists():
        zpath = cache / 'LLCP2023XPT.zip'
        if not zpath.exists():
            print('Tải', URL)
            urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.strip().upper().endswith('.XPT'))
            with z.open(name) as src, open(xpt, 'wb') as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
    cols = ['DIABETE4', '_AGEG5YR', '_SEX', '_BMI5', '_TOTINDA', '_RFHYPE6', '_RFCHOL3', '_SMOKER3', 'GENHLTH',
            '_STATE', '_INCOMG1', '_EDUCAG', '_MICHD', 'CVDSTRK3']
    return pd.read_sas(xpt, format='xport', encoding='latin1')[cols]


def main():
    raw = load_raw()
    total = len(raw)
    ok = (raw.DIABETE4.isin(DIAB) & raw._AGEG5YR.between(1, 13) & raw._SEX.isin([1, 2]) & raw._BMI5.notna()
          & raw._TOTINDA.isin([1, 2]) & raw._RFHYPE6.isin([1, 2]) & raw._RFCHOL3.isin([1, 2])
          & raw._SMOKER3.isin(SMOKER) & raw.GENHLTH.between(1, 5) & raw._STATE.isin(STATES)
          & raw._INCOMG1.isin(INCOME) & raw._EDUCAG.isin(EDU) & raw._MICHD.isin([1, 2]) & raw.CVDSTRK3.isin([1, 2]))
    valid = raw[ok]
    sample = valid.sample(n=N, random_state=SEED).reset_index(drop=True)
    bmi = sample._BMI5 / 100
    df = pd.DataFrame({
        'diabetes_status': sample.DIABETE4.astype(int).map(DIAB),
        'age_group': sample._AGEG5YR.astype(int).map(AGE),
        'sex': sample._SEX.astype(int).map({1: 'male', 2: 'female'}),
        'bmi': bmi.round(2),
        'bmi_category': pd.cut(bmi, [0, 18.5, 25, 30, 1000], right=False,
                               labels=['underweight', 'normal', 'overweight', 'obese']).astype(str),
        'physical_activity': (sample._TOTINDA == 1).astype(int),
        'high_blood_pressure': (sample._RFHYPE6 == 2).astype(int),
        'high_cholesterol': (sample._RFCHOL3 == 2).astype(int),
        'smoker_status': sample._SMOKER3.astype(int).map(SMOKER),
        'general_health': sample.GENHLTH.astype(int),
        'heart_disease': (sample._MICHD == 1).astype(int),
        'stroke': (sample.CVDSTRK3 == 1).astype(int),
        'income_group': sample._INCOMG1.astype(int).map(INCOME),
        'education': sample._EDUCAG.astype(int).map(EDU),
        'state': sample._STATE.astype(int).map(STATES),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'brfss2023_diabetes_150k.csv', index=False)
    manifest = {
        'source': 'CDC BRFSS 2023 (LLCP2023XPT.zip)', 'url': URL,
        'license': 'US government public data (public domain)', 'year': 2023,
        'raw_rows': int(total), 'valid_rows': int(ok.sum()), 'sampled_rows': len(df), 'seed': SEED,
        'columns': list(df.columns), 'diabetes_status_counts': df.diabetes_status.value_counts().to_dict(),
        'note': 'Mẫu ngẫu nhiên không trọng số; tỉ lệ chỉ mô tả mẫu, không đại diện toàn dân Hoa Kỳ.'}
    (OUT / 'brfss2023_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
