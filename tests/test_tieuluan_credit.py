"""Đối chiếu dữ liệu gốc tín dụng với phép biến đổi và dự báo đã lưu."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.tieuluan.credit import preprocess_credit
import pytest
from src.tieuluan.credit import parse_amount
from src.tieuluan.credit import risk_level
from src.tieuluan.live import load_model, predict_probability

ROOT = Path(__file__).resolve().parents[1]


def test_vietnamese_amounts():
    assert parse_amount('130.000,50') == 130000.5
    assert parse_amount('-1.234') == -1234
    assert parse_amount('1000') == 1000
    for text in ['1,234.00', 'NaN', '1.23', '1e1000', '']:
        with pytest.raises(ValueError): parse_amount(text)


def test_risk_level_boundaries():
    assert risk_level(.19,.4) == 'Thấp'
    assert risk_level(.2,.4) == 'Trung bình'
    assert risk_level(.399,.4) == 'Trung bình'
    assert risk_level(.4,.4) == 'Cao'

def test_twenty_raw_customers_match():
    raw = pd.read_excel(next((ROOT / 'data/tieuluan/raw/uci/credit_default').glob('*.xls')), header=1).iloc[:, 1:-1]
    with np.load(ROOT / 'data/tieuluan/processed/credit_default.npz') as d:
        ids = np.random.default_rng(2026).choice(len(d['test_ids']), 20, replace=False)
        rows = raw.iloc[d['test_ids'][ids]].loc[:, d['feature_names']].to_numpy(float)
        x = preprocess_credit(rows, d)
        np.testing.assert_allclose(x, d['test_tabular'][ids], atol=1e-5, rtol=0)
        model = load_model(ROOT, 'credit_default', 'mlp', (23,), 11)
        p = predict_probability([model], x)
    with np.load(ROOT / 'results/tieuluan/full/credit_default__mlp__scratch__11_predictions.npz') as saved:
        np.testing.assert_allclose(p, saved['probability'][ids], atol=1e-6, rtol=0)

def test_missing_values_use_training_medians():
    params = {'imputer_median': np.array([2., 3.]), 'scaler_mean': np.array([1., 1.]), 'scaler_scale': np.array([2., 4.])}
    np.testing.assert_array_equal(preprocess_credit([[np.nan, 5.]], params), [[.5, 1.]])


def test_assets_threshold_and_presets_are_reproducible():
    import json
    from scripts.tieuluan.run_experiments import select_threshold
    item = json.loads((ROOT / 'data/tieuluan/app/metadata.json').read_text(encoding='utf-8'))['datasets']['credit_default']
    runs = []
    for seed in (11,22,33):
        with np.load(ROOT / f'results/tieuluan/full/credit_default__mlp__scratch__{seed}_predictions.npz') as p:
            runs.append({k:p[k] for k in ['val_y','val_probability','probability']})
    assert item['threshold'] == select_threshold(runs[0]['val_y'],np.mean([r['val_probability'] for r in runs],axis=0))
    probability = np.mean([r['probability'] for r in runs],axis=0)
    assert item['presets'] == [int(np.argmin(abs(probability-np.quantile(probability,q)))) for q in (.1,.5,.9)]
