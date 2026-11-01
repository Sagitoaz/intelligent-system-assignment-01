"""Demo Streamlit Chương 2–4: khởi động, thao tác thật, số liệu khớp kết quả đã lưu, lỗi thân thiện."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'deployment/tieuluan/app.py'
sys.path.insert(0, str(APP.parent))
sys.path.insert(0, str(ROOT))


def run(chapter=None, **params):
    app = AppTest.from_file(str(APP), default_timeout=90)
    if chapter:
        app.query_params['chapter'] = chapter
    for key, value in params.items():
        app.query_params[key] = value
    return app.run()


def texts(app):
    parts = [m.value for m in app.markdown] + [c.value for c in app.caption] + [w.value for w in app.warning] + [i.value for i in app.info]
    return '\n'.join(parts)


def test_missing_artifacts_is_friendly_not_blank(monkeypatch, tmp_path):
    monkeypatch.setenv('TIEULUAN_ROOT', str(tmp_path))
    st.cache_data.clear()
    for chapter in '234':
        app = run(chapter)
        assert not app.exception
        assert any('chưa hiển thị được' in w.value or 'Chưa' in w.value for w in list(app.warning) + list(app.info))


def test_chapter2_scores_and_verifies_real_loan(monkeypatch):
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    st.cache_data.clear()
    app = run('2', embed='true', dataset='credit_default', kind='mlp')   # tham số cũ của hub vẫn hợp lệ
    assert not app.exception
    body = texts(app)
    assert 'Demo này làm gì?' in ''.join(m.value for m in app.markdown)
    assert 'Mô hình ước lượng khoảng' in ''.join(m.value for m in app.markdown)
    assert 'Kiểm chứng với kết cục thật' not in body            # hồ sơ tự nhập: chưa có nhãn thật
    next(b for b in app.button if b.label == 'Lấy một hồ sơ thật ngẫu nhiên').click().run()
    assert not app.exception
    html = ''.join(m.value for m in app.markdown)
    assert 'Kết cục thật của khoản vay này' in html
    assert 'ĐOÁN ĐÚNG' in html or 'ĐOÁN SAI' in html
    slider = next(s for s in app.slider if s.label.startswith('Lãi suất'))
    slider.set_value(30.0).run()
    assert not app.exception
    assert 'Kết cục thật của khoản vay này' not in ''.join(m.value for m in app.markdown)   # sửa hồ sơ -> thành tự nhập


def test_chapter2_high_risk_profile_scores_higher_than_low_risk(monkeypatch):
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    import ch2
    from src.tieuluan import loan
    params = ch2.read_npz(str(ROOT / 'data/tieuluan/app/loan_default.npz'))
    base = {f[0]: f[5] for f in loan.NUMERIC_FIELDS} | {'home_ownership': 'RENT', 'verification_status': 'Verified', 'purpose': 'small_business'}
    good = base | {'int_rate': 6.0, 'fico': 800, 'dti': 5.0, 'revol_util': 10.0}
    bad = base | {'int_rate': 28.0, 'fico': 640, 'dti': 40.0, 'revol_util': 95.0}
    p_good, _ = ch2.score(good, params, ROOT)
    p_bad, _ = ch2.score(bad, params, ROOT)
    assert p_bad['mlp'] > p_good['mlp'] + .1
    assert all(0 <= v <= 1 for v in p_bad.values())


def test_chapter3_and_4_day_demo_with_truth_and_scorecard(monkeypatch):
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    st.cache_data.clear()
    for chapter, kind in (('3', 'cnn4'), ('4', 'lstm')):
        app = run(chapter, embed='true', dataset='vnindex', kind=kind)
        assert not app.exception
        html = ''.join(m.value for m in app.markdown)
        assert 'Demo này làm gì?' in html and 'Cách đọc kết quả' in html
        assert 'Mô hình ước lượng khoảng' in html
        assert 'thực tế' in html and ('ĐOÁN ĐÚNG' in html or 'ĐOÁN SAI' in html)
        assert 'Nhận xét trung thực' in html
        app.selectbox(key=f'mk{chapter}').set_value('btc').run()
        assert not app.exception
        next(b for b in app.button if b.label == 'Chọn ngẫu nhiên').click().run()
        assert not app.exception


def test_chapter4_backtest_sentence_is_honest(monkeypatch):
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    app = run('4', embed='true', dataset='vnindex', kind='lstm')
    assert not app.exception
    html = ''.join(m.value for m in app.markdown)
    assert '100 triệu đồng' in html and 'Kết luận trung thực' in html
    analysis = json.loads((ROOT / 'results/tieuluan/full/analysis.json').read_text(encoding='utf-8'))
    market = 'vnindex'   # tham số dataset của hub chọn sẵn thị trường ở tab backtest
    rows = analysis['backtest'][market]
    bh = rows['buy_hold']['final_wealth'] * 100
    assert f'{bh:,.1f}'.translate(str.maketrans({',': '.', '.': ','})) in html


def test_saved_accuracy_matches_scorecard_inputs():
    """Tỉ lệ đúng hiển thị = tính lại từ dự báo đã lưu với đúng ngưỡng của bản ghi."""
    for market in ('sp500', 'vnindex', 'btc'):
        record = json.loads((ROOT / f'results/tieuluan/full/{market}__lstm__scratch__11.json').read_text(encoding='utf-8'))
        with np.load(ROOT / f'results/tieuluan/full/{market}__lstm__scratch__11_predictions.npz') as a:
            acc = float(np.mean((a['probability'] >= record['metrics']['threshold']) == (a['y'] == 1)))
        assert abs(acc - record['metrics']['accuracy']) < 1e-9


def test_live_network_error_is_friendly(monkeypatch):
    import market
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)

    def fail(*args, **kwargs):
        raise market.LiveDataError('Chưa tải được dữ liệu. Vui lòng thử lại sau.')
    monkeypatch.setattr(market, 'fetch_market', fail)
    st.cache_data.clear()
    app = run('4')
    next(b for b in app.button if b.label == 'Tải giá mới nhất và dự báo').click().run()
    assert not app.exception
    assert any('Chưa tải được' in w.value for w in app.warning)


def test_successful_live_prediction(monkeypatch):
    import market
    from src.tieuluan import live
    monkeypatch.delenv('TIEULUAN_ROOT', raising=False)
    frame = pd.read_csv(ROOT / 'data/tieuluan/app/sp500_prices.csv', parse_dates=['date']).tail(60)
    monkeypatch.setattr(market, 'fetch_market', lambda m: {
        'frame': frame, 'source': 'Nguồn kiểm thử', 'updated_at': '2026-10-06T08:00:00+00:00',
        'stale': False, 'fallback': False, 'age_days': 1})
    st.cache_data.clear()
    app = run('3')
    next(b for b in app.button if b.label == 'Tải giá mới nhất và dự báo').click().run()
    assert not app.exception
    with np.load(ROOT / 'data/tieuluan/app/sp500.npz') as d:
        features = live.market_features(frame.close, d['scaler_mean'], d['scaler_scale'])
    models = [live.load_model(ROOT, 'sp500', 'cnn4', features['image'].shape[1:], s) for s in (11, 22, 33)]
    p = live.predict_probability(models, features['image'])[0]
    assert f'{p * 100:.0f}%' in ''.join(m.value for m in app.markdown)


def test_standalone_and_unknown_chapter_show_picker():
    app = run()
    assert not app.exception
    assert len(app.radio) == 1
