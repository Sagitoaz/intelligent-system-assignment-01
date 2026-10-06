"""Sinh tài nguyên nhỏ, không huấn luyện và không sửa ảnh chụp thực nghiệm.

Chạy lại từ gốc repo bằng venv đầy đủ:
    .venv\\Scripts\\python.exe -m scripts.tieuluan.prepare_app_assets
App không cần xlrd/sklearn; Excel và chọn ngưỡng chỉ thực hiện tại đây.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scripts.tieuluan.run_experiments import select_threshold
from src.tieuluan.config import ANALYSIS_START, SNAPSHOT_END

ROOT = Path(__file__).resolve().parents[2]
BANK_COLUMNS = {
    ' ROA(A) before interest and % after tax': 'ROA: khả năng tạo lợi nhuận từ tài sản',
    ' Debt ratio %': 'Tỷ lệ nợ: phần tài sản được tài trợ bằng nợ',
    ' Net Income to Total Assets': 'Lãi ròng trên tổng tài sản',
    ' Current Ratio': 'Khả năng dùng tài sản ngắn hạn trả nợ ngắn hạn',
    ' Cash/Total Assets': 'Tiền mặt trên tổng tài sản',
    ' Net worth/Assets': 'Vốn chủ sở hữu trên tổng tài sản'}


def main():
    out = ROOT / 'data/tieuluan/app'; out.mkdir(parents=True, exist_ok=True)
    results = ROOT / 'results/tieuluan/full'
    inventory = json.loads((ROOT / 'data/tieuluan/processed/inventory.json').read_text(encoding='utf-8'))
    analysis = json.loads((results / 'analysis.json').read_text(encoding='utf-8'))
    metadata = {'snapshot_end': SNAPSHOT_END, 'seeds': [11, 22, 33], 'datasets': {}, 'source_sha256': {}}
    for dataset in ['taiwan_bankruptcy', 'credit_default', 'sp500', 'vnindex', 'btc']:
        path = ROOT / f'data/tieuluan/processed/{dataset}.npz'
        metadata['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as d:
            params = {k: d[k] for k in d.files if k.startswith(('scaler_', 'imputer_')) or k == 'feature_names'}
            item = {'history': {}}
            if dataset in ('taiwan_bankruptcy', 'credit_default'):
                if dataset == 'credit_default':
                    raw = pd.read_excel(next((ROOT / f'data/tieuluan/raw/uci/{dataset}').glob('*.xls')), header=1).iloc[:, 1:-1]
                    columns = list(d['feature_names'])
                else:
                    raw = pd.read_csv(ROOT / f'data/tieuluan/raw/uci/{dataset}/data.csv').iloc[:, 1:]
                    columns = list(BANK_COLUMNS)
                    item['display_labels'] = BANK_COLUMNS
                rows = raw.iloc[d['test_ids']].loc[:, columns]
                params.update(raw=rows.to_numpy(float), raw_names=np.array(columns), test_ids=d['test_ids'])
                item['base_rate'] = inventory[dataset]['positive'] / inventory[dataset]['raw_rows']
                kinds = ['mlp']
            else:
                frame = pd.read_csv(ROOT / f'data/tieuluan/raw/indices/{dataset}.csv')
                frame = frame[(frame.date >= ANALYSIS_START[dataset]) & (frame.date <= SNAPSHOT_END)]
                frame[['date', 'close']].to_csv(out / f'{dataset}_prices.csv', index=False)
                kinds = ['cnn4', 'lstm']
            for kind in kinds:
                runs = [json.loads((results / f'{dataset}__{kind}__scratch__{s}.json').read_text(encoding='utf-8')) for s in (11, 22, 33)]
                item['history'][kind] = {k: float(np.mean([r['metrics'][k] for r in runs]))
                                         for k in ('accuracy', 'balanced_accuracy', 'recall', 'roc_auc')}
                if dataset in ('sp500', 'vnindex', 'btc'):
                    item['history'][kind]['threshold'] = analysis['backtest'][dataset][kind]['threshold']
                elif dataset == 'credit_default':
                    predictions = []
                    for seed in (11, 22, 33):
                        with np.load(results / f'{dataset}__mlp__scratch__{seed}_predictions.npz') as p:
                            predictions.append({k: p[k] for k in ('probability', 'val_probability', 'val_y')})
                    pv = np.mean([p['val_probability'] for p in predictions], axis=0)
                    item['threshold'] = select_threshold(predictions[0]['val_y'], pv)
                    pt = np.mean([p['probability'] for p in predictions], axis=0)
                    item['presets'] = [int(np.argmin(abs(pt - np.quantile(pt, q)))) for q in (.1, .5, .9)]
                    item['preset_probabilities'] = pt[item['presets']].tolist()
            np.savez_compressed(out / f'{dataset}.npz', **params)
            metadata['datasets'][dataset] = item
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(list(out.iterdir())), 'bytes': sum(p.stat().st_size for p in out.iterdir())}))


if __name__ == '__main__':
    main()
