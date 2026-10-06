"""Đối chiếu với ảnh chụp thực nghiệm; toàn bộ tải mạng được giả lập."""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
import requests
from src.tieuluan.config import ANALYSIS_START
from src.tieuluan.live import market_features, predict_probability, load_model, closed_prices, fetch_market, LiveDataError

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('market', ['sp500', 'vnindex', 'btc'])
def test_raw_market_parity(market):
    frame = pd.read_csv(ROOT / f'data/tieuluan/raw/indices/{market}.csv')
    frame = frame[(frame.date >= ANALYSIS_START[market]) & (frame.date <= '2026-09-30')]
    with np.load(ROOT / f'data/tieuluan/processed/{market}.npz') as data:
        for index in [0, 17, len(data['test_y']) - 1]:
            close = frame.loc[frame.date <= str(data['test_date'][index]), 'close'].to_numpy()
            features = market_features(close, data['scaler_mean'], data['scaler_scale'])
            for kind, rep in [('lstm', 'sequence'), ('cnn4', 'image')]:
                np.testing.assert_allclose(features[rep][0], data[f'test_{rep}'][index], atol=1e-6, rtol=0)
                model = load_model(ROOT, market, kind, features[rep].shape[1:], 11)
                p = predict_probability([model], features[rep])[0]
                with np.load(ROOT / f'results/tieuluan/full/{market}__{kind}__scratch__11_predictions.npz') as saved:
                    assert abs(p - saved['probability'][index]) <= 1e-6

@pytest.mark.parametrize('market,now,last', [
    ('vnindex', '2026-10-06T07:59:00+00:00', '2026-10-05'),
    ('vnindex', '2026-10-06T08:00:00+00:00', '2026-10-06'),
    ('sp500', '2026-10-06T20:29:00+00:00', '2026-10-05'),
    ('sp500', '2026-10-06T20:30:00+00:00', '2026-10-06'),
    ('btc', '2026-10-06T23:59:00+00:00', '2026-10-05')])
def test_closed_sessions(market, now, last):
    f = pd.DataFrame({'date': pd.date_range('2026-08-01', '2026-10-07'), 'close': 100.})
    assert str(closed_prices(f, market, datetime.fromisoformat(now)).date.iloc[-1].date()) == last

def test_insufficient_and_invalid():
    with pytest.raises(ValueError):
        market_features([1.] * 20, [0.], [1.])
    with pytest.raises(ValueError):
        market_features([1.] * 20 + [np.nan], [0.], [1.])

def test_yahoo_and_fallback(monkeypatch):
    dates = pd.date_range('2026-08-01', periods=50, tz='UTC')
    payload = {'chart': {'result': [{'timestamp': [int(d.timestamp()) for d in dates],
               'indicators': {'quote': [{'close': list(range(100, 150))}]}}]}}
    response = Mock(); response.json.return_value = payload
    monkeypatch.setattr(requests, 'get', Mock(return_value=response))
    result = fetch_market('btc', now=datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert result['source'] == 'Yahoo Finance' and len(result['frame']) == 50
    assert result['stale']
    coinbase = Mock(); coinbase.json.return_value = [[int(d.timestamp()), 1, 2, 1, 100, 1] for d in dates]
    monkeypatch.setattr(requests, 'get', Mock(side_effect=[requests.Timeout(), requests.Timeout(), coinbase]))
    assert fetch_market('btc', now=datetime(2026, 10, 6, tzinfo=timezone.utc))['source'] == 'Coinbase Exchange'

def test_failure_bounded(monkeypatch):
    get = Mock(side_effect=requests.Timeout())
    monkeypatch.setattr(requests, 'get', get)
    with pytest.raises(LiveDataError, match='Chưa tải được'):
        fetch_market('vnindex')
    assert get.call_count == 4
    assert all(c.kwargs['timeout'] <= 10 for c in get.call_args_list)
