"""Ứng dụng minh họa của tiểu luận (Streamlit), chỉ đọc kết quả đã lưu.

Suy luận dùng mô hình NumPy tự viết (hạt giống 11) trên mẫu thuộc tập test giữ riêng; không tải
TensorFlow/PyTorch, không huấn luyện, không chọn lại ngưỡng. Nhãn thật chỉ dùng để hiển thị đối chiếu.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
ROOT = Path(os.environ.get('TIEULUAN_ROOT', str(CODE_ROOT)))
RESULTS = ROOT / 'results/tieuluan/full'
NAMES = {'taiwan_bankruptcy': 'Rủi ro phá sản doanh nghiệp (UCI 572)',
         'credit_default': 'Rủi ro vỡ nợ thẻ tín dụng (UCI 350)',
         'sp500': 'S&P 500', 'vnindex': 'VN-Index', 'btc': 'Bitcoin'}
KINDS = {'logistic': 'Hồi quy logistic', 'random_forest': 'Rừng ngẫu nhiên', 'mlp': 'MLP', 'cnn4': 'CNN4',
         'cnn8': 'CNN8', 'cnndeep': 'CNN sâu', 'rnn': 'SimpleRNN', 'lstm': 'LSTM', 'gru': 'GRU',
         'majority': 'Luôn đoán lớp đa số', 'persistence': 'Lặp lại hướng phiên trước'}
FRAMEWORKS = {'scratch': 'NumPy tự viết', 'pytorch': 'PyTorch', 'keras': 'Keras', 'sklearn': 'scikit-learn',
              'baseline': 'Đường cơ sở'}
STRATEGIES = {'buy_hold': 'Mua và giữ', 'persistence': 'Lặp lại hướng phiên trước', 'cnn4': 'CNN4', 'cnn8': 'CNN8',
              'cnndeep': 'CNN sâu', 'rnn': 'SimpleRNN', 'lstm': 'LSTM', 'gru': 'GRU'}


# Kết quả đã lưu không đổi khi ứng dụng chạy, nên được đọc một lần rồi giữ trong bộ nhớ đệm:
# máy chủ miễn phí chỉ có khoảng 0,1 CPU, đọc lại 128 file ở mỗi lần bấm sẽ rất chậm.
@st.cache_data(show_spinner=False)
def read_records(folder):
    records = []
    for path in sorted(Path(folder).glob('*__*.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(record, dict) and 'metrics' in record:
                records.append(record)
        except (OSError, ValueError):
            continue
    return records


@st.cache_data(show_spinner=False)
def read_analysis(folder):
    try:
        return json.loads((Path(folder) / 'analysis.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def read_test_split(path, representation):
    with np.load(path, allow_pickle=False) as archive:
        return (archive[f'test_{representation}'], archive['test_y'],
                archive['test_date'] if 'test_date' in archive else None,
                archive['test_target_date'] if 'test_target_date' in archive else None)


def stem_for(record):
    return '__'.join(str(record[key]) for key in ('dataset', 'kind', 'framework', 'seed'))


def demo(records):
    choices = [r for r in records if r.get('framework') == 'scratch'
               and r.get('seed') == 11 and r.get('kind') in ('mlp', 'cnn4', 'lstm')]
    if not choices:
        st.info('Chưa có mô hình hoàn tất. Khi kết quả huấn luyện được lưu, tải lại trang để xem demo.')
        return
    chapter = {'mlp': 'Chương 2 · MLP', 'cnn4': 'Chương 3 · CNN4', 'lstm': 'Chương 4 · LSTM'}
    chosen = st.selectbox('Bài toán và mô hình', range(len(choices)),
                          format_func=lambda i: f"{chapter[choices[i]['kind']]} · {NAMES.get(choices[i]['dataset'], choices[i]['dataset'])}")
    record = choices[chosen]
    stem = stem_for(record)
    path = ROOT / 'data/tieuluan/processed' / f"{record['dataset']}.npz"
    weights = ROOT / 'models/tieuluan/full' / f'{stem}.npz'
    if not path.exists() or not weights.exists():
        st.info('Chưa có đủ dữ liệu kiểm tra và trọng số cho lựa chọn này.')
        return
    try:
        features, labels, dates, target_dates = read_test_split(str(path), record['representation'])
        if len(features) == 0 or len(features) != len(labels):
            raise ValueError('Kích thước tập kiểm tra không hợp lệ.')
        index = int(st.number_input('Số thứ tự mẫu trong tập kiểm tra (bắt đầu từ 0)',
                                    min_value=0, max_value=len(features) - 1, value=0, step=1))
        # Nhãn và dự báo đã lưu không phải là đầu vào của mô hình.
        from src.tieuluan.experiment_models import scratch_model
        model = scratch_model(record['kind'], features.shape[1:], 11).load(weights).eval()
        logit = float(model.forward(features[index:index + 1]).reshape(-1)[0])
        probability = float(.5 * (1 + np.tanh(logit / 2)))
        threshold = float(record['metrics']['threshold'])
        if not np.isfinite(probability) or not 0 <= threshold <= 1:
            raise ValueError('Xác suất hoặc ngưỡng không hợp lệ.')
        market = record['kind'] != 'mlp'
        positive = 'Tăng' if market else 'Có rủi ro (lớp 1)'
        negative = 'Không tăng' if market else 'Không thuộc lớp rủi ro (lớp 0)'
        left, right = st.columns(2)
        left.metric('Xác suất phiên kế tiếp tăng' if market else 'Xác suất thuộc lớp rủi ro', f'{probability:.2%}')
        right.metric('Nhãn dự báo', positive if probability >= threshold else negative)
        st.write(f"Nhãn thực tế trong dữ liệu: **{positive if labels[index] == 1 else negative}**")
        st.caption(f'Ngưỡng {threshold:.2f} được chọn trên tập validation. Xác suất của mô hình chưa được hiệu chuẩn.')
        if dates is not None:
            st.write(f'Ngày cuối của cửa sổ quan sát: {dates[index]} · Ngày của nhãn: {target_dates[index]}')
        st.caption(f'Mẫu {index + 1}/{len(features)} của tập test giữ riêng; mô hình NumPy tự viết, hạt giống 11.')
        if market and record['representation'] == 'sequence':
            st.line_chart(pd.DataFrame(features[index], columns=['Lợi suất']), x_label='Phiên trong cửa sổ 20 phiên',
                          y_label='Lợi suất log (đã chuẩn hóa)')
        elif market:
            image = np.clip((features[index, 0] + 1) / 2, 0, 1)
            st.image(np.kron(image, np.ones((12, 12))), width=240, clamp=True,
                     caption='Ảnh GASF 20 × 20 tạo từ 20 giá đóng cửa gần nhất (phóng to, không nội suy).')
        else:
            with st.expander('Đặc trưng đã tiền xử lý của mẫu'):
                st.dataframe(pd.DataFrame({'Giá trị chuẩn hóa': features[index]}))
        saved = RESULTS / f'{stem}_predictions.npz'
        if saved.exists():
            with np.load(saved, allow_pickle=False) as archive:
                expected = archive['probability'][index]
            if np.isclose(probability, expected, atol=1e-6):
                st.caption('Đã đối chiếu: xác suất vừa tính trùng với dự báo lưu từ lần đánh giá.')
            else:
                st.warning('Xác suất vừa tính khác dự báo đã lưu; cần kiểm tra sự đồng bộ dữ liệu và trọng số.')
    except (OSError, ValueError, KeyError, IndexError, EOFError) as error:
        st.warning(f'Chưa thể đọc đầy đủ dữ liệu minh họa: {error}')


def comparison(records):
    if not records:
        st.info('Chưa có kết quả đánh giá để so sánh.')
        return
    frame = pd.DataFrame([{**{k: r.get(k) for k in ('dataset', 'kind', 'framework', 'seed', 'training_seconds')},
                           **{k: v for k, v in r['metrics'].items() if isinstance(v, (int, float))}} for r in records])
    dataset = st.selectbox('Bộ dữ liệu', sorted(frame.dataset.unique()), format_func=lambda name: NAMES.get(name, name))
    subset = frame[frame.dataset == dataset]
    rows = []
    for (kind, framework), group in subset.groupby(['kind', 'framework']):
        auc = group['roc_auc']
        rows.append({'Mô hình': KINDS.get(kind, kind), 'Cài đặt': FRAMEWORKS.get(framework, framework),
                     'ROC-AUC (trung bình)': round(auc.mean(), 4),
                     'ROC-AUC (độ lệch chuẩn)': round(auc.std(ddof=1), 4) if len(auc) > 1 else None,
                     'Balanced accuracy': round(group['balanced_accuracy'].mean(), 4),
                     'AP': round(group['average_precision'].mean(), 4) if 'average_precision' in group else None,
                     'Giây/lượt': round(group['training_seconds'].mean(), 1) if group['training_seconds'].notna().any() else None,
                     'Số lượt chạy': len(group)})
    st.dataframe(pd.DataFrame(rows).sort_values(['Mô hình', 'Cài đặt']), hide_index=True)
    st.caption('Đánh giá trên tập test giữ riêng; mỗi cấu hình chạy với 3 hạt giống (11, 22, 33). '
               'ROC-AUC = 0,5 tương đương đoán ngẫu nhiên. Thời gian đo trên CPU, không phải xếp hạng tốc độ chung của các thư viện.')


def backtest(analysis):
    if not analysis or 'equity' not in analysis:
        st.info('Chưa có kết quả backtest.')
        return
    market = st.selectbox('Thị trường', [k for k in ('sp500', 'vnindex', 'btc') if k in analysis['equity']],
                          format_func=lambda name: NAMES.get(name, name))
    curves = analysis['equity'][market]
    available = [k for k in STRATEGIES if k in curves]
    picked = st.multiselect('Chiến lược', available, default=[k for k in ('buy_hold', 'lstm', 'gru') if k in available],
                            format_func=lambda k: STRATEGIES[k])
    if picked:
        import altair as alt
        labels = [STRATEGIES[k] for k in picked]
        equity = pd.DataFrame({STRATEGIES[k]: curves[k] for k in picked})
        equity.insert(0, 'Ngày', pd.to_datetime(curves['dates']))
        long = equity.melt('Ngày', var_name='Chiến lược', value_name='Giá trị')
        chart = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X('Ngày:T', axis=alt.Axis(format='%m/%Y', title=None)),
            y=alt.Y('Giá trị:Q', scale=alt.Scale(zero=False), title='Giá trị danh mục (vốn ban đầu = 1)'),
            color=alt.Color('Chiến lược:N', sort=labels, legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Chiến lược', alt.Tooltip('Ngày:T', format='%d/%m/%Y'), alt.Tooltip('Giá trị:Q', format='.3f')])
        st.altair_chart(chart.properties(height=380), width='stretch')
    rows = []
    for key, item in analysis['backtest'][market].items():
        rows.append({'Chiến lược': STRATEGIES.get(key, key),
                     'Lợi suất năm trước phí': f"{item.get('annual_return_gross', float('nan')):.1%}",
                     'Lợi suất năm sau phí': f"{item['annual_return']:.1%}", 'Sharpe': round(item['sharpe'], 2),
                     'Sụt giảm tối đa': f"{item['max_drawdown']:.1%}", 'Thời gian nắm giữ': f"{item['exposure']:.0%}",
                     'Số lần mua': item['entries']})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    cost = analysis.get('assumptions', {}).get('cost_per_side', {}).get(market)
    st.caption(f'Tập test 01/2023–09/2026. Nắm giữ phiên kế tiếp khi xác suất tăng (trung bình 3 hạt giống) không thấp hơn '
               f'ngưỡng chọn trên validation, ngược lại giữ tiền mặt. Phí mỗi chiều giao dịch: {cost:.2%}. '
               'Giả định khớp lệnh đúng giá đóng cửa, không tính lãi tiền gửi: đây là mô phỏng lạc quan, không phải khuyến nghị đầu tư.')


def main():
    st.set_page_config(page_title='AI trong đầu tư tài chính · Minh họa', layout='wide')
    st.title('AI trong đầu tư tài chính')
    st.write('Minh họa các mô hình của tiểu luận trên dữ liệu lịch sử: rủi ro doanh nghiệp, rủi ro tín dụng '
             'và hướng đi ngày kế tiếp của S&P 500, VN-Index, Bitcoin.')
    st.info('Dữ liệu thị trường chốt ngày 30/09/2026. Đây là kiểm tra hồi cứu, không phải dự báo trực tiếp hay khuyến nghị đầu tư.')
    records = read_records(str(RESULTS))
    tabs = st.tabs(['Thử mô hình', 'So sánh thực nghiệm', 'Backtest', 'Dữ liệu và giới hạn'])
    with tabs[0]:
        demo(records)
    with tabs[1]:
        comparison(records)
    with tabs[2]:
        backtest(read_analysis(str(RESULTS)))
    with tabs[3]:
        st.markdown('- **Dữ liệu bảng:** phá sản doanh nghiệp Đài Loan (UCI 572) và vỡ nợ thẻ tín dụng Đài Loan (UCI 350), '
                    'chia ngẫu nhiên phân tầng 60/20/20.\n'
                    '- **Dữ liệu thị trường:** S&P 500 và Bitcoin (Yahoo Finance), VN-Index (SSI iBoard, bù và đối chiếu bằng '
                    'DNSE, VNDirect); chia theo thời gian: train đến 2019, validation 2020–2022, test 2023–09/2026.\n'
                    '- Mọi phép tiền xử lý chỉ học trên tập train; ngưỡng chọn trên validation; ứng dụng không huấn luyện lại '
                    'và không chọn lại ngưỡng trên test.\n'
                    '- Dự báo đúng hướng không đồng nghĩa có lãi: xem tab Backtest để thấy ảnh hưởng của phí giao dịch.')
        st.caption(f'Số bản ghi thực nghiệm đã đọc: {len(records)}.')


if __name__ == '__main__':
    main()
