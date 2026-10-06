"""UI startup and inference checks independent of long-running training jobs."""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.tieuluan.experiment_models import scratch_model

APP = Path(__file__).resolve().parents[1] / 'deployment/tieuluan/app.py'


def test_missing_artifacts_is_informative(monkeypatch, tmp_path):
    monkeypatch.setenv('TIEULUAN_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 7
    assert any('Chưa có' in message.value for message in app.info)


def test_saved_model_inference_and_label_separation(monkeypatch, tmp_path):
    monkeypatch.setenv('TIEULUAN_ROOT', str(tmp_path))
    data = tmp_path / 'data/tieuluan/processed'
    models = tmp_path / 'models/tieuluan/full'
    results = tmp_path / 'results/tieuluan/full'
    for path in (data, models, results):
        path.mkdir(parents=True)
    stem = 'taiwan_bankruptcy__mlp__scratch__11'
    model = scratch_model('mlp', (2,), 11)
    model.save(models / f'{stem}.npz')
    x = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    p = .5 * (1 + np.tanh(model.forward(x).reshape(-1) / 2))
    np.savez(data / 'taiwan_bankruptcy.npz', test_tabular=x, test_y=[0, 1])
    np.savez(results / f'{stem}_predictions.npz', probability=p, y=[0, 1])
    record = dict(dataset='taiwan_bankruptcy', kind='mlp', framework='scratch', seed=11,
                  representation='tabular', metrics={'threshold': .45, 'roc_auc': .7,
                  'balanced_accuracy': .6}, training_seconds=1., history=[])
    (results / f'{stem}.json').write_text(json.dumps(record))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 7
    assert app.metric[0].value == f'{p[0]:.2%}'.replace('.', ',')
    first_probability = app.metric[0].value
    np.savez(data / 'taiwan_bankruptcy.npz', test_tabular=x, test_y=[1, 0])
    # Labels are read only for display; changing them cannot change inference.
    st.cache_data.clear()
    app.run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == first_probability


def test_backtest_tab_reads_saved_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv('TIEULUAN_ROOT', str(tmp_path))
    results = tmp_path / 'results/tieuluan/full'
    results.mkdir(parents=True)
    item = dict(annual_return=.1, annual_return_gross=.12, sharpe=.9, max_drawdown=.2, exposure=.6, entries=5)
    analysis = {'equity': {'vnindex': {'dates': ['2023-01-03', '2023-01-04'], 'buy_hold': [1.0, 1.01], 'lstm': [1.0, 1.0]}},
                'backtest': {'vnindex': {'buy_hold': item, 'lstm': item}},
                'assumptions': {'cost_per_side': {'vnindex': .002}}}
    (results / 'analysis.json').write_text(json.dumps(analysis))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 7
    assert any('Mua và giữ' in str(frame.value.values) for frame in app.dataframe)


def test_real_app_choices_and_credit(monkeypatch):
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    def choose(label, value):
        next(s for s in app.selectbox if s.label == label).set_value(value)
        app.run(timeout=30)
        assert not app.exception
    for dataset in ['sp500', 'vnindex', 'btc', 'credit_default', 'taiwan_bankruptcy']:
        choose('Bộ dữ liệu kiểm tra', dataset)
        for kind in (['lstm','cnn4'] if dataset in ['sp500','vnindex','btc'] else ['mlp']):
            choose('Mô hình kiểm tra', kind)
            assert any('Đã đối chiếu với dự báo đã lưu' in c.value for c in app.caption)
    for preset in range(3):
        choose('Hồ sơ mẫu theo xác suất trong tập kiểm tra', preset)
        next(b for b in app.button if b.label == 'Chấm điểm hồ sơ').click().run(timeout=30)
        assert not app.exception
        assert any(m.label == 'Xác suất vỡ nợ tháng tới' for m in app.metric)


def test_live_network_error_is_friendly(monkeypatch):
    from src.tieuluan import live
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    def fail(*args, **kwargs):
        raise live.LiveDataError('Chưa tải được dữ liệu. Vui lòng thử lại sau.')
    monkeypatch.setattr(live, 'fetch_market', fail)
    st.cache_data.clear()
    app = AppTest.from_file(str(APP)).run(timeout=30)
    next(b for b in app.button if b.label == 'Tải lại dữ liệu mới nhất').click().run(timeout=30)
    assert not app.exception
    assert any('Chưa tải được' in w.value for w in app.warning)


def test_successful_live_ensemble_and_threshold(monkeypatch):
    import pandas as pd
    from src.tieuluan import live
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    root = APP.parents[2]
    frame = pd.read_csv(root / 'data/tieuluan/app/sp500_prices.csv', parse_dates=['date']).tail(60)
    monkeypatch.setattr(live, 'fetch_market', lambda market: {
        'frame': frame, 'source': 'Nguồn kiểm thử', 'updated_at': '2026-10-06T08:00:00+00:00',
        'stale': False, 'fallback': False, 'age_days': 1})
    st.cache_data.clear()
    app = AppTest.from_file(str(APP)).run(timeout=30)
    next(b for b in app.button if b.label == 'Tải lại dữ liệu mới nhất').click().run(timeout=30)
    assert not app.exception
    with np.load(root / 'data/tieuluan/app/sp500.npz') as d:
        features = live.market_features(frame.close, d['scaler_mean'], d['scaler_scale'])
    expected = []
    analysis = json.loads((root / 'results/tieuluan/full/analysis.json').read_text())
    for kind, rep in [('lstm','sequence'), ('cnn4','image')]:
        models = [live.load_model(root, 'sp500', kind, features[rep].shape[1:], seed) for seed in (11,22,33)]
        p = live.predict_probability(models, features[rep])[0]
        expected.append(f'{p:.2%}'.replace('.', ','))
        label = 'Nghiêng về tăng' if p >= analysis['backtest']['sp500'][kind]['threshold'] else 'Nghiêng về không tăng'
        assert any(label in m.value for m in app.markdown)
    assert [m.value for m in app.metric if m.label == 'Xác suất tăng'] == expected
