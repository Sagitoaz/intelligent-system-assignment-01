"""UI startup and inference checks independent of long-running training jobs."""
import json
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.tieuluan.experiment_models import scratch_model

APP = Path(__file__).resolve().parents[1] / 'deployment/tieuluan/app.py'


def test_missing_artifacts_is_informative(monkeypatch, tmp_path):
    monkeypatch.setenv('TIEULUAN_ROOT', str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 4
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
    assert len(app.tabs) == 4
    assert app.metric[0].value == f'{p[0]:.2%}'
    first_probability = app.metric[0].value
    np.savez(data / 'taiwan_bankruptcy.npz', test_tabular=x, test_y=[1, 0])
    # Labels are read only for display; changing them cannot change inference.
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
    assert len(app.tabs) == 4
    assert any('Mua và giữ' in str(frame.value.values) for frame in app.dataframe)
