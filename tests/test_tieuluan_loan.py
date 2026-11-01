"""Chương 2: pipeline đặc trưng dùng chung giữa huấn luyện và demo, dữ liệu Lending Club đã xử lý, mô hình đã lưu."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tieuluan import loan
from src.tieuluan.live import load_model, predict_probability

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'data/tieuluan/raw/lending_club/loans_36m_2012_2015_150k.csv'


def test_monthly_payment_formula():
    assert abs(float(loan.monthly_payment(12000, 13.0)) - 404.71) < .5
    assert abs(float(loan.monthly_payment(3600, 0.0)) - 100.0) < 1e-9


def test_risk_band_relative_to_base_rate():
    assert loan.risk_band(.05, .14) == 'thấp'
    assert loan.risk_band(.14, .14) == 'trung bình'
    assert loan.risk_band(.30, .14) == 'cao'


def test_preprocess_uses_training_median_for_missing():
    params = {'imputer_median': np.array([2., 3.]), 'scaler_mean': np.array([1., 1.]), 'scaler_scale': np.array([2., 4.])}
    np.testing.assert_array_equal(loan.preprocess([[np.nan, 5.]], params), [[.5, 1.]])
    with pytest.raises(ValueError):
        loan.preprocess([[1., 2., 3.]], params)


def test_dataset_size_time_split_and_no_leakage():
    inv = json.loads((ROOT / 'data/tieuluan/processed/inventory.json').read_text(encoding='utf-8'))['loan_default']
    assert 100_000 <= inv['raw_rows'] <= 200_000 and inv['features'] == len(loan.FEATURES)
    assert sum(s['n'] for s in inv['splits'].values()) == inv['raw_rows']
    with np.load(ROOT / 'data/tieuluan/processed/loan_default.npz') as d:
        assert d['train_date'].max() < d['val_date'].min() and d['val_date'].max() < d['test_date'].min()
        assert np.abs(d['train_tabular'].mean(0)).max() < 1e-3          # chuẩn hóa fit trên train
        assert d['train_tabular'].shape[1] == len(loan.FEATURES)
    cols = pd.read_csv(CSV, nrows=1).columns
    assert not any(c in cols for c in ('total_pymnt', 'last_pymnt_d', 'recoveries', 'out_prncp'))


def test_demo_pipeline_reproduces_training_matrix_and_saved_predictions():
    frame = pd.read_csv(ROOT / 'data/tieuluan/app/loan_test.csv')
    with np.load(ROOT / 'data/tieuluan/app/loan_default.npz') as params, np.load(ROOT / 'data/tieuluan/processed/loan_default.npz') as d:
        ids = np.random.default_rng(2026).choice(len(frame), 25, replace=False)
        x = loan.preprocess(loan.build_features(frame.iloc[ids]).to_numpy(), params)
        np.testing.assert_allclose(x, d['test_tabular'][ids], atol=1e-4, rtol=0)
        assert np.array_equal(frame['default'].to_numpy()[ids], d['test_y'][ids])
    model = load_model(ROOT, 'loan_default', 'mlp', (len(loan.FEATURES),), 11)
    p = predict_probability([model], x)
    with np.load(ROOT / 'results/tieuluan/full/loan_default__mlp__scratch__11_predictions.npz') as saved:
        np.testing.assert_allclose(p, saved['probability'][ids], atol=1e-5, rtol=0)


def test_saved_results_cover_three_models_three_seeds():
    folder = ROOT / 'results/tieuluan/full'
    for kind, fw in (('mlp', 'scratch'), ('mlp', 'pytorch'), ('mlp', 'keras'), ('logistic', 'sklearn'), ('random_forest', 'sklearn')):
        for seed in (11, 22, 33):
            r = json.loads((folder / f'loan_default__{kind}__{fw}__{seed}.json').read_text(encoding='utf-8'))
            assert r['metrics']['roc_auc'] > .6 and r['n_train'] > 80_000
    analysis = json.loads((folder / 'analysis.json').read_text(encoding='utf-8'))
    assert {'logistic', 'random_forest', 'mlp'} <= set(analysis['uci']['loan_default'])
