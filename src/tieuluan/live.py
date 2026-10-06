"""Suy luận nhẹ và dữ liệu phiên đã đóng; không phụ thuộc thư viện huấn luyện.

Công thức sao chép từ experiment_data, được kiểm thử đối chiếu với ảnh chụp đã lưu.
Không ghi vào dữ liệu hay kết quả thực nghiệm. Cache do lớp giao diện quản lý.
"""
from datetime import datetime, timezone, time
from io import StringIO
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

SEEDS = (11, 22, 33)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36'}
ZONES = {'vnindex': 'Asia/Ho_Chi_Minh', 'sp500': 'America/New_York', 'btc': 'UTC'}


class LiveDataError(ValueError):
    """Không đủ dữ liệu thật để dự báo."""


def market_features(close, scaler_mean, scaler_scale):
    prices = np.asarray(close, dtype=np.float64)
    if prices.ndim != 1 or len(prices) < 21 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError('Cần ít nhất 21 giá đóng cửa dương, không thiếu.')
    mean, scale = float(np.asarray(scaler_mean)[0]), float(np.asarray(scaler_scale)[0])
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Tham số chuẩn hóa không hợp lệ.')
    # StandardScaler lúc huấn luyện nhận float32 và biến đổi tại chỗ qua hai bước.
    sequence = np.diff(np.log(prices[-21:])).astype(np.float32)
    sequence -= mean
    sequence /= scale
    p = prices[-20:]
    span = np.ptp(p)
    z = np.zeros_like(p) if span < 1e-12 else (p - p.min()) / span
    z = np.clip(z, 0, 1)
    s = np.sqrt(np.maximum(0, 1 - z * z))
    image = (np.outer(z, z) - np.outer(s, s)).astype(np.float32)
    return {'sequence': sequence.reshape(1, 20, 1), 'image': image.reshape(1, 1, 20, 20)}


def load_model(root, dataset, kind, input_shape, seed):
    from .experiment_models import scratch_model
    model = scratch_model(kind, input_shape, seed).load(
        Path(root) / f'models/tieuluan/full/{dataset}__{kind}__scratch__{seed}.npz').eval()
    return model, RLock()


def predict_probability(models, features):
    predictions = []
    for model, lock in models:
        # Các lớp tự viết lưu bộ đệm forward; khóa tránh hai phiên web ghi đè nhau.
        with lock:
            logits = np.asarray(model.forward(features).reshape(-1), dtype=float)
            predictions.append(.5 * (1 + np.tanh(logits / 2)))
    result = np.mean(predictions, axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError('Đầu vào khiến mô hình không tính được xác suất hữu hạn.')
    return result


def closed_prices(frame, market, now):
    local = now.astimezone(ZoneInfo(ZONES[market]))
    f = frame[['date', 'close']].copy()
    f['date'] = pd.to_datetime(f.date, errors='coerce').dt.tz_localize(None).dt.normalize()
    f['close'] = pd.to_numeric(f.close, errors='coerce')
    f = f.dropna().loc[lambda x: (x.close > 0) & np.isfinite(x.close)]
    today = pd.Timestamp(local.date())
    allow_today = (market == 'vnindex' and local.time() >= time(15)) or (
        market == 'sp500' and local.time() >= time(16, 30))
    f = f[f.date <= today] if allow_today else f[f.date < today]
    f = f.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)
    if len(f) < 21:
        raise LiveDataError('Nguồn dữ liệu chưa có đủ 21 phiên đã đóng cửa hợp lệ.')
    return f


def source_urls(market, now):
    end = int(now.timestamp()); start = end - 180 * 86400
    if market == 'vnindex':
        query = f'resolution=1D&symbol=VNINDEX&from={start}&to={end}'
        return [('SSI iBoard', 'https://iboard-api.ssi.com.vn/statistics/charts/history?' + query),
                ('DNSE Entrade', 'https://services.entrade.com.vn/chart-api/v2/ohlcs/index?' + query)]
    ticker = '%5EGSPC' if market == 'sp500' else 'BTC-USD'
    backup = ('FRED', 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500') if market == 'sp500' else (
        'Coinbase Exchange', 'https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400')
    return [('Yahoo Finance', f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=6mo&interval=1d'), backup]


def parse_prices(response, source, market):
    if source == 'FRED':
        frame = pd.read_csv(StringIO(response.text))
        return frame.rename(columns={frame.columns[0]: 'date', 'SP500': 'close'})[['date', 'close']]
    payload = response.json()
    if source == 'Yahoo Finance':
        data = payload['chart']['result'][0]
        stamps, close = data['timestamp'], data['indicators']['quote'][0]['close']
    elif source == 'Coinbase Exchange':
        stamps, close = [r[0] for r in payload], [r[4] for r in payload]
    else:
        data = payload['data'] if source == 'SSI iBoard' else payload
        stamps, close = data['t'], data['c']
    dates = pd.to_datetime(stamps, unit='s', utc=True).tz_convert(ZONES[market]).date
    return pd.DataFrame({'date': dates, 'close': close})


def fetch_market(market, now=None):
    """Hai nguồn, tối đa hai lần gọi mỗi nguồn, timeout chín giây mỗi lần.

    Không lấp giá bị thiếu, không dùng dữ liệu mẫu khi mạng hỏng.
    """
    if market not in ZONES:
        raise ValueError('Thị trường không được hỗ trợ.')
    now = now or datetime.now(timezone.utc)
    failures = []
    for source, url in source_urls(market, now):
        for attempt in range(2):
            try:
                response = requests.get(url, headers=HEADERS, timeout=9)
                response.raise_for_status()
                frame = closed_prices(parse_prices(response, source, market), market, now)
                age = (now.astimezone(ZoneInfo(ZONES[market])).date() - frame.date.iloc[-1].date()).days
                return {'frame': frame.tail(180).reset_index(drop=True), 'source': source,
                        'updated_at': now.isoformat(), 'stale': age > (2 if market == 'btc' else 5),
                        'age_days': age, 'fallback': bool(failures)}
            except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
                if attempt == 1:
                    failures.append(f'{source}: {type(error).__name__}')
    raise LiveDataError('Chưa tải được đủ dữ liệu từ các nguồn. Vui lòng thử lại sau. '
                        'Không có dự báo mới khi thiếu dữ liệu. (' + '; '.join(failures) + ')')
