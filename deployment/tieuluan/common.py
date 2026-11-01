"""Hàm dùng chung cho demo Chương 2–4: định dạng số kiểu Việt Nam, đọc dữ liệu, thanh đo, khung giải thích."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('TIEULUAN_ROOT', str(CODE_ROOT)))
RESULTS = ROOT / 'results/tieuluan/full'
ASSETS = ROOT / 'data/tieuluan/app'
PROCESSED = ROOT / 'data/tieuluan/processed'



def configure():
    """Đọc lại biến môi trường TIEULUAN_ROOT (kiểm thử đổi thư mục gốc giữa các lần chạy)."""
    global ROOT, RESULTS, ASSETS, PROCESSED
    ROOT = Path(os.environ.get('TIEULUAN_ROOT', str(CODE_ROOT)))
    RESULTS, ASSETS, PROCESSED = (ROOT / 'results/tieuluan/full', ROOT / 'data/tieuluan/app', ROOT / 'data/tieuluan/processed')


GREEN, AMBER, RED, BLUE, GRAY = '#12a37f', '#f0a017', '#e5484d', '#2589e0', '#667085'
MARKET_NAMES = {'sp500': 'S&P 500 (chứng khoán Mỹ)', 'vnindex': 'VN-Index (chứng khoán Việt Nam)', 'btc': 'Bitcoin'}
MARKETS = list(MARKET_NAMES)
KIND_NAMES = {'logistic': 'Hồi quy logistic', 'random_forest': 'Rừng ngẫu nhiên', 'mlp': 'Mạng MLP',
              'cnn4': 'CNN 4 bộ lọc', 'cnn8': 'CNN 8 bộ lọc', 'cnndeep': 'CNN hai tầng tích chập',
              'rnn': 'SimpleRNN', 'lstm': 'LSTM', 'gru': 'GRU',
              'majority': 'Luôn đoán nhóm đông hơn', 'persistence': 'Lặp lại hướng phiên trước'}
NOTICE = 'Chỉ là minh họa học thuật, không phải khuyến nghị đầu tư hay tín dụng.'


def vn(value, digits=1):
    return f'{float(value):,.{digits}f}'.translate(str.maketrans({',': '.', '.': ','}))


def pct(value, digits=1):
    return vn(float(value) * 100, digits) + '%'


def day(value):
    return pd.Timestamp(value).strftime('%d/%m/%Y')


@st.cache_data(show_spinner=False)
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_json(path, default=None):
    try:
        return read_json(str(path))
    except (OSError, ValueError):
        return {} if default is None else default


@st.cache_data(show_spinner=False)
def read_records(folder):
    records = []
    for path in sorted(Path(folder).glob('*__*.json')):
        try:
            r = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(r, dict) and 'metrics' in r:
                records.append(r)
        except (OSError, ValueError):
            pass
    return records


@st.cache_data(show_spinner=False)
def read_npz(path, prefixes=()):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files if not prefixes or k.startswith(tuple(prefixes))}


@st.cache_data(show_spinner=False)
def read_csv(path, **kw):
    return pd.read_csv(path, **kw)


def mean_metric(records, dataset, kind, framework, key):
    vals = [r['metrics'][key] for r in records if r['dataset'] == dataset and r['kind'] == kind
            and r['framework'] == framework and key in r['metrics']]
    return float(np.mean(vals)) if vals else None


CSS = """
<style>
.block-container{padding-top:1.1rem;max-width:1180px}
.hero{background:linear-gradient(135deg,#e8f3ff,#f4fbff);border:1px solid #bcd9f5;border-left:6px solid #2589e0;
 border-radius:14px;padding:14px 18px;margin:4px 0 14px 0;color:#101828}
.hero h4{margin:0 0 6px 0;font-size:1.05rem;color:#1b5fa6}
.hero p{margin:0 0 6px 0;line-height:1.55;font-size:.97rem}
.howto{background:#fff9e9;border:1px solid #f3dca0;border-radius:12px;padding:10px 16px;margin:8px 0 14px 0;
 color:#4a3b05;font-size:.93rem;line-height:1.5}
.verdict{border-radius:14px;padding:16px 20px;margin:10px 0;font-size:1.12rem;line-height:1.5;color:#101828;
 border:1px solid #d0d5dd;background:#fff}
.verdict b{font-weight:700}
.v-low{background:#e9f9f2;border-color:#9edcc4}.v-mid{background:#fff6e0;border-color:#f3d28a}
.v-high{background:#fdecec;border-color:#f3a5a8}.v-info{background:#eaf4ff;border-color:#a9cdf2}
.gauge{position:relative;height:26px;border-radius:13px;margin:34px 6px 6px 6px;
 background:linear-gradient(90deg,#12a37f 0%,#9bd64f 28%,#f0a017 55%,#e5484d 100%);box-shadow:inset 0 0 0 1px #0001}
.gauge.cool{background:linear-gradient(90deg,#e5484d 0%,#f0a017 35%,#9bd64f 70%,#12a37f 100%)}
.gauge .pin{position:absolute;top:-30px;transform:translateX(-50%);background:#101828;color:#fff;border-radius:8px;
 padding:2px 9px;font-size:.88rem;font-weight:700;white-space:nowrap}
.gauge .pin:after{content:"";position:absolute;left:50%;bottom:-6px;transform:translateX(-50%);
 border:6px solid transparent;border-top-color:#101828;border-bottom:0}
.gauge .tick{position:absolute;top:-4px;bottom:-4px;width:3px;background:#101828;opacity:.55;border-radius:2px}
.gauge-scale{display:flex;justify-content:space-between;color:#475467;font-size:.82rem;margin:0 6px 10px 6px}
.gauge-note{color:#475467;font-size:.84rem;margin:0 6px 14px 6px}
.chip{display:inline-block;border-radius:999px;padding:3px 12px;font-weight:600;font-size:.9rem;margin-right:6px}
.chip-up{background:#dff6ec;color:#0b7a5c}.chip-down{background:#fde4e4;color:#b42318}
.step{display:inline-block;background:#2589e0;color:#fff;border-radius:50%;width:26px;height:26px;line-height:26px;
 text-align:center;font-weight:700;margin-right:8px}
div[data-testid="stMetricValue"]{font-size:1.5rem}
</style>
"""
EMBED_CSS = """
<style>
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,#MainMenu{display:none!important}
.block-container{padding-top:.6rem!important;padding-bottom:1rem!important}
</style>
"""


def inject_css(embed):
    st.markdown(CSS + (EMBED_CSS if embed else ''), unsafe_allow_html=True)


def hero(title, *paragraphs):
    body = ''.join(f'<p>{p}</p>' for p in paragraphs)
    st.markdown(f'<div class="hero"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def howto(text):
    st.markdown(f'<div class="howto"><b>Cách đọc kết quả.</b> {text}</div>', unsafe_allow_html=True)


def verdict(text, tone='info'):
    cls = {'low': 'v-low', 'mid': 'v-mid', 'high': 'v-high', 'info': 'v-info'}[tone]
    st.markdown(f'<div class="verdict {cls}">{text}</div>', unsafe_allow_html=True)


def gauge(value, left, right, mark=None, mark_label='ngưỡng', cool=False, text=None):
    """Thanh đo 0–100%: ghim đen là giá trị; vạch mờ là mốc tham chiếu (nếu có)."""
    v = min(max(float(value), 0.0), 1.0)
    tick = f'<div class="tick" style="left:{min(max(mark, 0), 1) * 100:.1f}%"></div>' if mark is not None else ''
    label = text if text is not None else pct(v)
    html = (f'<div class="gauge{" cool" if cool else ""}">{tick}'
            f'<div class="pin" style="left:{min(max(v, .04), .96) * 100:.1f}%">{label}</div></div>'
            f'<div class="gauge-scale"><span>{left}</span><span>{right}</span></div>')
    if mark is not None:
        html += f'<div class="gauge-note">Vạch mờ trên thanh: {mark_label} ({pct(mark)}).</div>'
    st.markdown(html, unsafe_allow_html=True)


def step_title(n, text):
    st.markdown(f'<h4><span class="step">{n}</span>{text}</h4>', unsafe_allow_html=True)


def gasf_rgb(matrix, scale=14):
    """Ảnh GASF 20×20 (giá trị −1…1) -> ảnh màu phóng to, xanh = thấp, đỏ = cao."""
    from matplotlib import colormaps
    rgb = colormaps['RdBu_r']((np.clip(np.asarray(matrix), -1, 1) + 1) / 2)[..., :3]
    return np.kron(rgb, np.ones((scale, scale, 1)))
