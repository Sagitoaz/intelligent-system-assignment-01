"""Đối chiếu dữ liệu gốc tín dụng với phép biến đổi và dự báo đã lưu."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.tieuluan.credit import preprocess_credit
from src.tieuluan.live import load_model, predict_probability

ROOT = Path(__file__).resolve().parents[1]

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
