"""Chương 3 (CNN đọc ảnh GASF) và Chương 4 (RNN/LSTM đọc chuỗi lợi suất): dự báo hướng phiên sau và backtest."""
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (AMBER, BLUE, GRAY, GREEN, MARKETS, MARKET_NAMES, KIND_NAMES, NOTICE, RED, day, gasf_rgb, gauge, hero, howto, mean_metric, pct, read_csv, read_npz, safe_json, step_title, verdict, vn)
import common
from src.tieuluan.live import LiveDataError, fetch_market, load_model, market_features, predict_probability

CHAPTER = {
    '3': {'kind': 'cnn4', 'rep': 'image', 'seeds_note': 'CNN đọc ảnh',
          'title': 'Máy “nhìn” biểu đồ giá như nhìn một bức ảnh'},
    '4': {'kind': 'lstm', 'rep': 'sequence', 'seeds_note': 'LSTM đọc chuỗi',
          'title': 'Máy “đọc” 20 mức tăng giảm gần nhất theo thứ tự thời gian'},
}
STRATEGIES = {'buy_hold': 'Mua và giữ', 'persistence': 'Lặp lại hướng phiên trước',
              **{k: f'Theo mô hình {KIND_NAMES[k]}' for k in ('cnn4', 'cnn8', 'cnndeep', 'rnn', 'lstm', 'gru')}}
SEEDS = (11, 22, 33)


@st.cache_resource(show_spinner=False)
def cached_model(root, dataset, kind, shape, seed):
    return load_model(root, dataset, kind, shape, seed)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=3)
def latest_prices(market):
    return fetch_market(market)


def infer(dataset, kind, features, seeds=(11,)):
    models = [cached_model(str(common.ROOT), dataset, kind, tuple(features.shape[1:]), s) for s in seeds]
    return float(predict_probability(models, features)[0])


def intro(chapter):
    if chapter == '3':
        hero('Demo này làm gì?',
             'Mỗi phiên giao dịch, giá đóng cửa tạo thành một đường lên xuống. Demo biến <b>20 phiên gần nhất</b> thành một <b>bức ảnh màu</b> '
             '(ảnh GASF) rồi để mạng CNN, loại mạng chuyên “nhìn ảnh”, đoán <b>phiên kế tiếp tăng hay không tăng</b>.',
             'Bạn chọn một thị trường và một ngày <b>đã xảy ra</b> trong giai đoạn kiểm tra (01/2023–09/2026, mô hình chưa từng thấy). '
             'Máy đưa ra dự đoán, sau đó bạn so với kết quả thật. Phần cuối cho biết mô hình đúng bao nhiêu lần trên toàn bộ giai đoạn.')
        howto('Vùng tô màu trên đồ thị là 20 phiên mô hình được “nhìn”. Thanh đo cho biết xác suất phiên sau tăng. '
              'Điều cần nhớ: đoán đúng hơn 50% một cách đều đặn là rất khó vì giá thị trường gần như ngẫu nhiên.')
    else:
        hero('Demo này làm gì?',
             'Mô hình LSTM (một loại mạng nơ-ron có “trí nhớ”) đọc <b>20 mức tăng giảm giá liên tiếp</b> theo thứ tự thời gian, từ cũ đến mới, '
             'rồi đoán <b>phiên kế tiếp tăng hay không tăng</b>. Bạn thử trên một ngày đã biết kết quả, thử với giá mới nhất, '
             'và xem nếu <b>đem 100 triệu đồng ra mua bán theo mô hình</b> thì kết quả ra sao so với cứ mua rồi giữ.',
             'Phần backtest là thí nghiệm trên quá khứ (01/2023–09/2026): thử một cách làm cũ xem nó <i>sẽ</i> lãi hay lỗ, '
             'có trừ phí mua bán.')
        howto('Mỗi cột trong biểu đồ “20 mức thay đổi giá” là một phiên: cột xanh là tăng, cột đỏ là giảm. '
              'Ở backtest, đường nào nằm cao hơn ở cuối là cách làm cho nhiều tiền hơn; hãy luôn nhìn số <b>sau phí</b>.')


def pick_random(key, n):
    st.session_state[key] = int(np.random.default_rng().integers(n))


def price_chart(prices, shade_from, shade_to):
    base = alt.Chart(prices).encode(x=alt.X('date:T', axis=alt.Axis(format='%d/%m/%Y', title='Ngày')))
    area = alt.Chart(pd.DataFrame({'a': [shade_from], 'b': [shade_to], 't': ['20 phiên mô hình “nhìn”']})).mark_rect(
        opacity=.22, color=BLUE).encode(x='a:T', x2='b:T')
    line = base.mark_line(strokeWidth=2.4, color='#101828').encode(
        y=alt.Y('close:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='Giá đóng cửa', labelExpr="replace(format(datum.value, ',.0f'), /,/g, '.')")),
        tooltip=[alt.Tooltip('date:T', title='Ngày', format='%d/%m/%Y'), alt.Tooltip('close:Q', title='Giá đóng cửa', format=',.2f')])
    last = alt.Chart(prices.tail(1)).mark_point(size=130, color=RED, filled=True, opacity=1).encode(x='date:T', y='close:Q')
    st.altair_chart((area + line + last).properties(height=280), width='stretch')


def return_bars(returns):
    frame = pd.DataFrame({'Phiên': np.arange(1, len(returns) + 1), 'Thay đổi giá (%)': returns * 100})
    frame['Hướng'] = np.where(returns >= 0, 'Tăng', 'Giảm')
    chart = alt.Chart(frame).mark_bar().encode(
        x=alt.X('Phiên:O', axis=alt.Axis(title='20 phiên gần nhất (trái = cũ nhất, phải = mới nhất)', labelAngle=0)),
        y=alt.Y('Thay đổi giá (%):Q', axis=alt.Axis(title='Thay đổi giá so với phiên trước (%)')),
        color=alt.Color('Hướng:N', scale=alt.Scale(domain=['Tăng', 'Giảm'], range=[GREEN, RED]), legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Phiên', alt.Tooltip('Thay đổi giá (%):Q', format='.2f')])
    st.altair_chart(chart.properties(height=230), width='stretch')


def scorecard(records, analysis, dataset, kind, prediction_threshold):
    """Tỉ lệ đoán đúng trên toàn tập kiểm tra, đối chiếu đoán mò; nói thẳng khi mô hình yếu."""
    runs = [r for r in records if r['dataset'] == dataset and r['kind'] == kind and r['framework'] == 'scratch']
    if not runs:
        st.info('Chưa có số liệu đánh giá trên tập kiểm tra.')
        return
    with np.load(common.RESULTS / f'{dataset}__{kind}__scratch__11_predictions.npz') as a:
        y, p = a['y'], a['probability']
    acc = float(np.mean((p >= prediction_threshold) == (y == 1)))
    up = float(np.mean(y == 1))
    all_acc = float(np.mean([r['metrics']['accuracy'] for r in runs]))
    auc = mean_metric(records, dataset, kind, 'scratch', 'roc_auc')
    ci = analysis.get('auc_ci', {}).get(dataset, {}).get(kind)
    c = st.columns(3)
    c[0].metric('Mô hình đoán đúng', pct(acc, 1), help=f'Trên {vn(len(y), 0)} phiên kiểm tra, mô hình hạt giống 11.')
    c[1].metric('Đoán mò (tung đồng xu)', '50,0%')
    c[2].metric('Luôn đoán “tăng”', pct(up, 1), help='Cách đơn giản nhất: phiên nào cũng đoán tăng.')
    best_simple = max(.5, up, 1 - up)
    margin = acc - best_simple
    ci_text = ''
    if ci:
        ci_text = (f' ROC-AUC {vn(ci["auc"], 3)} (khoảng tin cậy 95%: {vn(ci["low"], 3)}–{vn(ci["high"], 3)}; 0,5 là đoán mò)'
                   + (' nằm gọn trên 0,5: khả năng xếp hạng có tín hiệu rất yếu, nhưng chưa đủ để thành lợi thế khi phải chọn tăng hay giảm.' if ci['low'] > .5 else ' chứa 0,5 nên chưa chứng minh hơn đoán mò.'))
    if margin > .02 and ci and ci['low'] > .5:
        text = (f'<b>Nhận xét trung thực:</b> trên {vn(len(y), 0)} phiên kiểm tra mô hình đoán đúng {pct(acc, 1)}, hơn cách đơn giản nhất '
                f'{vn(margin * 100, 1)} điểm phần trăm: có tín hiệu nhưng <b>rất nhỏ</b>.{ci_text}')
        tone = 'mid'
    else:
        text = (f'<b>Nhận xét trung thực:</b> trên {vn(len(y), 0)} phiên kiểm tra mô hình đoán đúng {pct(acc, 1)}; cách đơn giản nhất '
                f'(luôn đoán nhóm đông hơn) đã đạt {pct(best_simple, 1)}. Mô hình <b>chưa tốt hơn cách đoán đơn giản</b>.{ci_text}')
        tone = 'high'
    verdict(text, tone)
    st.caption(f'Trung bình ba lần huấn luyện (hạt giống 11, 22, 33): đúng {pct(all_acc, 1)}. Ngưỡng quyết định {pct(prediction_threshold, 1)} chọn trên giai đoạn kiểm định 2020–2022.')


def day_tab(chapter, records, analysis):
    cfg = CHAPTER[chapter]
    kind, rep = cfg['kind'], cfg['rep']
    intro(chapter)
    q = st.query_params.get('dataset')
    default = MARKETS.index(q) if q in MARKETS else 1
    step_title(1, 'Chọn thị trường và một ngày đã xảy ra')
    cols = st.columns([1.4, 1.6, 1])
    dataset = cols[0].selectbox('Thị trường', MARKETS, index=default, format_func=MARKET_NAMES.get, key=f'mk{chapter}')
    data = read_npz(str(common.PROCESSED / f'{dataset}.npz'), ('test_',))
    dates = [str(d) for d in data['test_date']]
    key = f'idx{chapter}_{dataset}'
    if key not in st.session_state:
        st.session_state[key] = len(dates) // 2
    cols[1].selectbox('Ngày cuối của 20 phiên mô hình nhìn', range(len(dates)), key=key, format_func=lambda i: day(dates[i]))
    cols[2].markdown('<div style="height:1.7rem"></div>', unsafe_allow_html=True)
    cols[2].button('Chọn ngẫu nhiên', key=f'rnd{chapter}{dataset}', on_click=pick_random, args=(key, len(dates)))
    index = int(st.session_state[key])
    anchor = pd.Timestamp(dates[index])
    prices = read_csv(str(common.ASSETS / f'{dataset}_prices.csv'), parse_dates=['date'])
    history = prices[prices.date <= anchor].tail(60)
    window = prices[prices.date <= anchor].tail(20)

    step_title(2, 'Máy được xem gì?')
    left, right = st.columns([1.5, 1])
    with left:
        price_chart(history, window.date.iloc[0], window.date.iloc[-1])
        st.caption('Đường đen: giá đóng cửa 60 phiên gần nhất. **Vùng xanh dương: 20 phiên máy được xem.** Chấm đỏ: ngày cuối cùng máy biết, máy không được thấy gì sau đó.')
    with right:
        if rep == 'image':
            st.image(gasf_rgb(data['test_image'][index, 0]), width=300, caption='Ảnh GASF 20 × 20 tạo từ 20 giá đóng cửa ở vùng xanh dương.')
            st.caption('**Vì sao đổi giá thành ảnh?** CNN rất giỏi tìm hoa văn trong ảnh. Mỗi ô (hàng i, cột j) ghi mối quan hệ giữa phiên i và phiên j '
                       'trong 20 phiên: đỏ là hai phiên cùng ở mức giá cao, xanh là lệch nhau. Hoa văn này là thứ mạng đọc để đoán.')
        else:
            seq = data['test_sequence'][index, :, 0]
            returns = seq * read_npz(str(common.ASSETS / f'{dataset}.npz'))['scaler_scale'][0] + read_npz(str(common.ASSETS / f'{dataset}.npz'))['scaler_mean'][0]
            return_bars(np.expm1(returns))
            st.caption('**Vì sao dùng mức thay đổi thay vì giá?** Giá của S&P 500 hay Bitcoin có thể gấp nhiều lần sau vài năm; mức tăng giảm mỗi phiên thì '
                       'luôn cùng thang đo nên mô hình học được ở giai đoạn này, dùng được ở giai đoạn khác.')

    step_title(3, 'Máy dự đoán')
    try:
        p = infer(dataset, kind, data[f'test_{rep}'][index:index + 1])
        record = next(r for r in records if r['dataset'] == dataset and r['kind'] == kind and r['framework'] == 'scratch' and r['seed'] == 11)
        threshold = record['metrics']['threshold']
    except (OSError, KeyError, StopIteration, ValueError) as error:
        st.warning('Chưa tải được mô hình hoặc dữ liệu cho lựa chọn này.')
        st.caption(f'Chi tiết cho người quản trị: {type(error).__name__}.')
        return
    nxt = day(data['test_target_date'][index])
    lean_up = p >= threshold
    verdict(f'Mô hình ước lượng khoảng <b>{pct(p, 0)}</b> khả năng phiên kế tiếp (<b>{nxt}</b>) sẽ <b>tăng</b> giá. '
            f'Mô hình nghiêng về: <b>{"TĂNG" if lean_up else "KHÔNG TĂNG"}</b>.', 'low' if lean_up else 'mid')
    gauge(p, 'Nhiều khả năng giảm', 'Nhiều khả năng tăng', mark=threshold, mark_label='ngưỡng mô hình dùng để chuyển từ “không tăng” sang “tăng”', cool=True)
    real = float(data['test_return'][index])
    right_ = bool(data['test_y'][index]) == lean_up
    step_title(4, 'Kết quả thật')
    verdict(f'Phiên {nxt} thực tế <b>{"tăng" if real > 0 else "không tăng"} {vn(abs(real) * 100, 2)}%</b>. '
            f'Mô hình đã <b>{"ĐOÁN ĐÚNG" if right_ else "ĐOÁN SAI"}</b>.', 'low' if right_ else 'high')
    st.caption('Một ngày chưa nói lên điều gì: đoán đúng một lần có thể chỉ là may mắn. Phần dưới tổng hợp toàn bộ giai đoạn kiểm tra.')
    step_title(5, 'Trên toàn bộ giai đoạn kiểm tra, mô hình đúng bao nhiêu lần?')
    scorecard(records, analysis, dataset, kind, threshold)
    st.caption(NOTICE)


def live_tab(chapter, records, analysis):
    cfg = CHAPTER[chapter]
    kind, rep = cfg['kind'], cfg['rep']
    hero('Dự báo cho phiên sắp tới',
         'Bấm nút để tải <b>giá đóng cửa thật mới nhất</b> từ Internet. Mô hình dùng 20 phiên gần nhất để đoán phiên kế tiếp tăng hay không. '
         'Đây là cùng mô hình ở tab trước, <b>không học thêm</b> từ dữ liệu mới.')
    market = st.selectbox('Thị trường', MARKETS, format_func=MARKET_NAMES.get, key=f'live{chapter}')
    if st.button('Tải giá mới nhất và dự báo', type='primary', key=f'liveb{chapter}'):
        latest_prices.clear()
        st.session_state['live_requested'] = market
    if st.session_state.get('live_requested') != market:
        st.info('Bấm nút “Tải giá mới nhất và dự báo” để bắt đầu. Dữ liệu được lưu tạm tối đa 30 phút.')
        return
    try:
        with st.spinner('Đang lấy giá các phiên đã đóng cửa…'):
            latest = latest_prices(market)
        f = latest['frame'].tail(60).copy()
        st.write(f"Nguồn: **{latest['source']}** · Phiên đã đóng cửa gần nhất: **{day(f.date.iloc[-1])}**")
        if latest['fallback']:
            st.caption('Đang dùng nguồn dự phòng; giá có thể lệch nhẹ so với nguồn đã dùng để huấn luyện.')
        if latest['stale']:
            st.warning(f"Phiên cuối đã cách {vn(latest['age_days'], 0)} ngày. Dự báo dựa trên dữ liệu cũ, không phải cho phiên sắp tới.")
        price_chart(f.rename(columns={'close': 'close'}), f.date.iloc[-20], f.date.iloc[-1])
        params = read_npz(str(common.ASSETS / f'{market}.npz'))
        features = market_features(latest['frame'].close.to_numpy(), params['scaler_mean'], params['scaler_scale'])
        p = infer(market, kind, features[rep], SEEDS)
        threshold = analysis['backtest'][market][kind]['threshold']
        verdict(f'Mô hình {KIND_NAMES[kind]} ước lượng khoảng <b>{pct(p, 0)}</b> khả năng phiên kế tiếp của {MARKET_NAMES[market]} <b>tăng</b>. '
                f'Nghiêng về: <b>{"TĂNG" if p >= threshold else "KHÔNG TĂNG"}</b>.', 'low' if p >= threshold else 'mid')
        gauge(p, 'Nhiều khả năng giảm', 'Nhiều khả năng tăng', mark=threshold, mark_label='ngưỡng quyết định', cool=True)
        auc = analysis.get('auc_ci', {}).get(market, {}).get(kind, {})
        st.warning('Đừng coi đây là lời khuyên: ' + ('trên giai đoạn kiểm tra, mô hình chưa hơn đoán mò một cách chắc chắn. ' if auc.get('low', 0) <= .5 else
                   'tín hiệu của mô hình trên giai đoạn kiểm tra rất yếu. ') + 'Xác suất là trung bình của ba lần huấn luyện và chưa được hiệu chuẩn.')
        updated = datetime.fromisoformat(latest['updated_at']).astimezone(ZoneInfo('Asia/Ho_Chi_Minh'))
        st.caption('Cập nhật lúc ' + updated.strftime('%H:%M:%S %d/%m/%Y') + ' (giờ Việt Nam). ' + NOTICE)
    except LiveDataError as error:
        st.warning(str(error))
    except (OSError, KeyError, ValueError) as error:
        st.warning('Chưa dự báo được vì thiếu dữ liệu hoặc mô hình.')
        st.caption(f'Chi tiết cho người quản trị: {type(error).__name__}.')


def backtest_tab(analysis):
    hero('Thử đem 100 triệu đồng ra mua bán theo mô hình',
         '<b>Backtest</b> là chạy thử một cách mua bán trên quá khứ để xem nó sẽ lãi hay lỗ. Giả sử đầu năm 2023 bạn có <b>100 triệu đồng</b> '
         'và chỉ đang dùng một trong ba cách: <b>mua và giữ</b> (mua đầu kỳ rồi không làm gì), <b>theo mô hình</b> '
         '(mô hình báo “tăng” thì giữ tài sản, báo “không tăng” thì ôm tiền mặt) hoặc <b>lặp lại hướng phiên trước</b>.')
    howto('Mỗi lần mua hoặc bán phải trả phí. Mô hình đổi trạng thái rất nhiều lần nên phí ăn mòn lợi nhuận: hãy so cột <b>trước phí</b> với <b>sau phí</b>.')
    if not analysis or not analysis.get('equity'):
        st.info('Chưa có kết quả backtest.')
        return
    markets = [m for m in MARKETS if m in analysis['equity']]
    q = st.query_params.get('dataset')
    market = st.selectbox('Thị trường mô phỏng', markets, index=markets.index(q) if q in markets else 0, format_func=MARKET_NAMES.get, key='bt_market')
    curves = analysis['equity'][market]
    available = [k for k in STRATEGIES if k in curves]
    picked = st.multiselect('Các cách mua bán muốn so', available, default=[k for k in ['buy_hold', 'lstm', 'gru'] if k in available],
                            format_func=STRATEGIES.get, key='bt_picked')
    periods = analysis.get('assumptions', {}).get('periods_per_year', {}).get(market, 365 if market == 'btc' else 252)
    steps = len(curves['dates'])
    cost = analysis.get('assumptions', {}).get('cost_per_side', {}).get(market)
    rows = {}
    for key, item in analysis['backtest'][market].items():
        final = item.get('final_wealth', curves.get(key, [float('nan')])[-1])
        gross = (1 + item['annual_return_gross']) ** (steps / periods)
        rows[key] = {'net': final * 100, 'gross': gross * 100, 'item': item}
    step_title(1, f'100 triệu đồng thành bao nhiêu đến {day(curves["dates"][-1])}?')
    if picked:
        frame = pd.DataFrame({STRATEGIES[k]: np.array(curves[k]) * 100 for k in picked}, index=pd.to_datetime(curves['dates']))
        long = frame.rename_axis('Ngày').reset_index().melt('Ngày', var_name='Cách mua bán', value_name='Triệu đồng')
        chart = alt.Chart(long).mark_line(strokeWidth=2.4).encode(
            x=alt.X('Ngày:T', axis=alt.Axis(format='%m/%Y', title='Thời gian')),
            y=alt.Y('Triệu đồng:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='Triệu đồng (sau phí)')),
            color=alt.Color('Cách mua bán:N', legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Cách mua bán', alt.Tooltip('Ngày:T', format='%d/%m/%Y'), alt.Tooltip('Triệu đồng:Q', format=',.1f')])
        rule = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(strokeDash=[5, 4], color=GRAY).encode(y='y:Q')
        st.altair_chart((chart + rule).properties(height=320), width='stretch')
        st.caption('Đường đứt xám: số vốn ban đầu 100 triệu đồng. Phía trên là lãi, phía dưới là lỗ.')
    else:
        st.info('Chọn ít nhất một cách mua bán ở trên để vẽ biểu đồ.')
    step_title(2, 'Trước phí và sau phí: phí lấy đi bao nhiêu?')
    keys = [k for k in picked if k in rows]
    if keys:
        long = pd.DataFrame([{'Cách mua bán': STRATEGIES[k], 'Lúc tính': lab, 'Triệu đồng': rows[k][col]}
                             for k in keys for lab, col in (('Trước phí', 'gross'), ('Sau phí', 'net'))])
        long['Nhãn'] = long['Triệu đồng'].map(lambda v: vn(v, 0))
        bars = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X('Cách mua bán:N', sort=None, axis=alt.Axis(labelAngle=0, title=None, labelLimit=170)),
            y=alt.Y('Triệu đồng:Q', axis=alt.Axis(title='Giá trị cuối kỳ (triệu đồng)')), xOffset=alt.XOffset('Lúc tính:N', sort=['Trước phí', 'Sau phí']),
            color=alt.Color('Lúc tính:N', scale=alt.Scale(domain=['Trước phí', 'Sau phí'], range=['#9bc4ef', BLUE]), legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Cách mua bán', 'Lúc tính', 'Nhãn'])
        text = alt.Chart(long).mark_text(dy=-8).encode(x=alt.X('Cách mua bán:N', sort=None), xOffset=alt.XOffset('Lúc tính:N', sort=['Trước phí', 'Sau phí']), y='Triệu đồng:Q', text='Nhãn:N')
        st.altair_chart((bars + text).properties(height=290), width='stretch')
    table = pd.DataFrame([{'Cách mua bán': STRATEGIES.get(k, k), 'Cuối kỳ trước phí (triệu đồng)': vn(v['gross'], 1),
                           'Cuối kỳ sau phí (triệu đồng)': vn(v['net'], 1), 'Lãi mỗi năm sau phí': pct(v['item']['annual_return']),
                           'Lúc tệ nhất mất tối đa': pct(v['item']['max_drawdown'], 0), 'Số lần mua': v['item']['entries']}
                          for k, v in rows.items()])
    st.dataframe(table, hide_index=True)
    if 'buy_hold' in rows:
        bh = rows['buy_hold']['net']
        models = {k: rows[k] for k in ('lstm', 'gru', 'rnn') if k in rows}
        best_key = max(models, key=lambda k: models[k]['net']) if models else None
        if best_key:
            b = models[best_key]
            beats_net = b['net'] > bh * 1.02
            beats_gross = b['gross'] > bh * 1.02
            sentence = (f'<b>Kết luận trung thực:</b> mua và giữ ra <b>{vn(bh, 1)} triệu</b>. Cách theo mô hình tốt nhất ({KIND_NAMES[best_key]}) ra '
                        f'<b>{vn(b["gross"], 1)} triệu trước phí</b> và <b>{vn(b["net"], 1)} triệu sau phí</b>. ')
            if beats_net:
                sentence += 'Sau phí cách theo mô hình nhỉnh hơn, nhưng chênh lệch nhỏ và chỉ là một giai đoạn, chưa đủ để kết luận mô hình có lợi thế thật.'
                tone = 'mid'
            else:
                sentence += ('Sau phí cách theo mô hình <b>không hơn mua và giữ</b>' + (' (trước phí có nhỉnh hơn, nhưng phí lấy đi lợi thế đó).' if beats_gross else '.'))
                tone = 'high'
            verdict(sentence, tone)
    st.caption(f'Phí mỗi lần mua hoặc bán: {pct(cost, 2) if cost is not None else "chưa có số liệu"} giá trị giao dịch. Vốn quy đổi 100 triệu đồng chỉ để dễ hình dung; '
               'mô phỏng chạy theo biến động của chỉ số, không tính tỷ giá. Giả định khớp lệnh đúng giá đóng cửa và tiền mặt không sinh lãi: mô phỏng này lạc quan hơn thực tế. ' + NOTICE)


def compare_tab(chapter, records, analysis):
    kinds = ['cnn4', 'cnn8', 'cnndeep'] if chapter == '3' else ['rnn', 'lstm', 'gru']
    what = 'ba cỡ mạng CNN' if chapter == '3' else 'ba loại mạng hồi quy (SimpleRNN, LSTM, GRU)'
    hero('So sánh các mô hình',
         f'Cùng bài toán “phiên sau tăng hay không”, so {what} với hai cách đoán đơn giản: <b>luôn đoán nhóm đông hơn</b> '
         'và <b>lặp lại hướng của phiên trước</b>. Một mô hình phức tạp chỉ đáng dùng nếu nó thắng được các cách đơn giản này.')
    howto('Thước đo là <b>ROC-AUC</b>: 50% là đoán mò, càng gần 100% càng tốt. Đoạn thẳng mỏng quanh mỗi cột là khoảng tin cậy 95%: '
          'nếu đoạn đó còn chạm vạch 50% thì chưa chứng minh được mô hình hơn đoán mò.')
    market = st.selectbox('Thị trường', MARKETS, format_func=MARKET_NAMES.get, key=f'cmp{chapter}',
                          index=MARKETS.index(st.query_params.get('dataset')) if st.query_params.get('dataset') in MARKETS else 1)
    ci = analysis.get('auc_ci', {}).get(market, {})
    rows = [{'Mô hình': KIND_NAMES[k], 'auc': ci[k]['auc'], 'low': ci[k]['low'], 'high': ci[k]['high']} for k in kinds + ['persistence'] if k in ci]
    if not rows:
        st.info('Chưa có kết quả so sánh.')
        return
    frame = pd.DataFrame(rows)
    frame['Nhãn'] = frame.auc.map(lambda v: pct(v, 1))
    bars = alt.Chart(frame).mark_bar(size=54, cornerRadiusEnd=5).encode(
        x=alt.X('Mô hình:N', sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y('auc:Q', scale=alt.Scale(domain=[.4, .65]), axis=alt.Axis(format='.0%', title='ROC-AUC')),
        color=alt.Color('Mô hình:N', legend=None), tooltip=['Mô hình', 'Nhãn'])
    err = alt.Chart(frame).mark_rule(color='#101828', strokeWidth=2).encode(x=alt.X('Mô hình:N', sort=None), y='low:Q', y2='high:Q')
    text = alt.Chart(frame).mark_text(dy=-24, fontWeight='bold').encode(x=alt.X('Mô hình:N', sort=None), y='high:Q', text='Nhãn:N')
    ref = alt.Chart(pd.DataFrame({'y': [.5]})).mark_rule(strokeDash=[5, 4], color=RED).encode(y='y:Q')
    st.altair_chart((bars + err + text + ref).properties(height=320), width='stretch')
    st.caption('Đường đứt đỏ: 50% (đoán mò). Trục dọc bắt đầu từ 40% để nhìn rõ chênh lệch nhỏ; vì vậy chiều cao cột không phản ánh tỉ lệ thật.')
    above = [r['Mô hình'] for r in rows if r['low'] > .5 and r['Mô hình'] not in ('Lặp lại hướng phiên trước',)]
    best = max((r for r in rows if r['Mô hình'] != KIND_NAMES['persistence']), key=lambda r: r['auc'])
    persistence = ci.get('persistence', {}).get('auc')
    if above:
        verdict(f'<b>Kết luận:</b> với {MARKET_NAMES[market]}, mô hình tốt nhất là <b>{best["Mô hình"]}</b> (ROC-AUC {vn(best["auc"] * 100, 1)}%), khoảng tin cậy nằm trên 50% nên có '
                f'tín hiệu, nhưng <b>rất yếu</b>' + (f'; cách “lặp lại hướng phiên trước” đạt {vn(persistence * 100, 1)}%.' if persistence else '.'), 'mid')
    else:
        verdict(f'<b>Kết luận:</b> với {MARKET_NAMES[market]}, không mô hình nào có khoảng tin cậy nằm hẳn trên 50%: <b>chưa chứng minh được hơn đoán mò</b>. '
                f'Mô hình có ROC-AUC cao nhất là {best["Mô hình"]} ({vn(best["auc"] * 100, 1)}%) nhưng chênh lệch này nằm trong dao động ngẫu nhiên.', 'high')
    shown = frame[['Mô hình']].copy()
    shown['ROC-AUC'] = frame.auc.map(lambda v: vn(v, 3))
    shown['Khoảng tin cậy 95%'] = [f'{vn(a, 3)} – {vn(b, 3)}' for a, b in zip(frame.low, frame.high)]
    st.dataframe(shown, hide_index=True)


def limits_tab():
    hero('Dữ liệu và giới hạn',
         'Phần này giải thích dữ liệu đến từ đâu, chia thế nào và vì sao không nên coi demo như công cụ đầu tư.')
    st.markdown('- **Dữ liệu:** giá đóng cửa hằng ngày của S&P 500 (2000), VN-Index (2010) và Bitcoin (2014), cắt cố định tại 30/09/2026. Mô hình học đến 2019, '
                'chọn ngưỡng trong 2020–2022 và được kiểm tra từ 01/2023 đến 30/09/2026.\n'
                '- **Mẫu:** mỗi mẫu là 20 phiên liên tiếp, nhãn là phiên kế tiếp tăng (1) hay không (0). Mẫu có nhãn vượt sang tập khác đã bị loại để không “nhìn trước”.\n'
                '- **Nguồn giá mới (tab dự báo):** Yahoo Finance, dự phòng FRED (S&P 500) hoặc Coinbase (Bitcoin); VN-Index lấy từ SSI iBoard, dự phòng DNSE. Chỉ dùng phiên đã đóng cửa.\n'
                '- **Giới hạn:** thị trường thay đổi theo thời gian nên mô hình có thể mất tác dụng; xác suất chưa hiệu chuẩn; kết quả quá khứ không bảo đảm tương lai.')
    st.caption(NOTICE)


def render(chapter, records, analysis):
    if chapter == '3':
        names = ['1 · Thử trên một ngày đã biết', '2 · Dự báo phiên sắp tới', '3 · So sánh mô hình', '4 · Dữ liệu và giới hạn']
        fns = [lambda: day_tab(chapter, records, analysis), lambda: live_tab(chapter, records, analysis),
               lambda: compare_tab(chapter, records, analysis), limits_tab]
    else:
        names = ['1 · Thử trên một ngày đã biết', '2 · Dự báo phiên sắp tới', '3 · Backtest 100 triệu đồng', '4 · So sánh mô hình']
        fns = [lambda: day_tab(chapter, records, analysis), lambda: live_tab(chapter, records, analysis),
               lambda: backtest_tab(analysis), lambda: compare_tab(chapter, records, analysis)]
    for tab, fn in zip(st.tabs(names), fns):
        with tab:
            fn()
