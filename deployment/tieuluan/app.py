"""Minh họa tiếng Việt, chỉ suy luận bằng trọng số đã lưu, không huấn luyện lại."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
from src.tieuluan.live import fetch_market, market_features, load_model, predict_probability, LiveDataError
from src.tieuluan.credit import preprocess_credit, parse_amount, risk_level, LABELS, CATEGORIES, PAY_CODES

ROOT = Path(os.environ.get('TIEULUAN_ROOT', str(CODE_ROOT)))
RESULTS = ROOT / 'results/tieuluan/full'
ASSETS = ROOT / 'data/tieuluan/app'
NAMES = {'taiwan_bankruptcy': 'Phá sản doanh nghiệp Đài Loan', 'credit_default': 'Vỡ nợ thẻ tín dụng Đài Loan',
         'sp500': 'S&P 500', 'vnindex': 'VN-Index', 'btc': 'Bitcoin'}
KINDS = {'logistic': 'Hồi quy logistic', 'random_forest': 'Rừng ngẫu nhiên', 'mlp': 'MLP', 'cnn4': 'CNN4',
         'cnn8': 'CNN8', 'cnndeep': 'CNN sâu', 'rnn': 'SimpleRNN', 'lstm': 'LSTM', 'gru': 'GRU',
         'majority': 'Luôn đoán nhóm đông hơn', 'persistence': 'Lặp lại hướng phiên trước'}
FRAMEWORKS = {'scratch': 'NumPy tự viết', 'pytorch': 'PyTorch', 'keras': 'Keras', 'sklearn': 'scikit-learn', 'baseline': 'Cách đoán đơn giản'}
STRATEGIES = {'buy_hold': 'Mua và giữ', 'persistence': 'Lặp lại hướng phiên trước',
              **{k: f'Theo mô hình {KINDS[k]}' for k in ('cnn4', 'cnn8', 'cnndeep', 'rnn', 'lstm', 'gru')}}
MARKETS = ['sp500', 'vnindex', 'btc']
NOTICE = 'Không phải khuyến nghị đầu tư'


def vn(value, digits=1):
    return f'{float(value):,.{digits}f}'.translate(str.maketrans({',': '.', '.': ','}))


def pct(value, digits=1):
    return vn(float(value)*100, digits) + '%'


def day(value):
    return pd.Timestamp(value).strftime('%d/%m/%Y')


@st.cache_data(show_spinner=False)
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def read_records(folder):
    records = []
    for path in sorted(Path(folder).glob('*__*.json')):
        try:
            r = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(r, dict) and 'metrics' in r: records.append(r)
        except (OSError, ValueError): pass
    return records


@st.cache_data(show_spinner=False)
def read_npz(path, prefixes=('test_',)):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files if not prefixes or k.startswith(prefixes)}


@st.cache_data(show_spinner=False)
def read_prices(path):
    return pd.read_csv(path, parse_dates=['date'])


@st.cache_resource(show_spinner=False)
def cached_model(root, dataset, kind, shape, seed):
    return load_model(root, dataset, kind, shape, seed)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=3)
def latest_prices(market):
    return fetch_market(market)


def infer(dataset, kind, features, seeds=(11,)):
    models = [cached_model(str(ROOT), dataset, kind, tuple(features.shape[1:]), s) for s in seeds]
    return float(predict_probability(models, features)[0])


def guide(text):
    st.info('**Cách đọc:** ' + text)


def price_chart(frame, ylabel='Giá đóng cửa'):
    """Ngày và số trên trục/ô gợi ý dùng định dạng Việt Nam."""
    import altair as alt
    long = frame.rename_axis('Ngày').reset_index().melt('Ngày', var_name='Đường', value_name='Giá trị').dropna()
    long['Ngày hiển thị'] = long['Ngày'].map(day)
    long['Giá trị hiển thị'] = long['Giá trị'].map(lambda v: vn(v,2))
    chart = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X('Ngày:T',axis=alt.Axis(format='%d/%m/%Y',title='Ngày')),
        y=alt.Y('Giá trị:Q',scale=alt.Scale(zero=False),axis=alt.Axis(title=ylabel,labelExpr="replace(format(datum.value, ',.0f'), /,/g, '.')")),
        color=alt.Color('Đường:N',legend=alt.Legend(title=None,orient='bottom')),
        tooltip=[alt.Tooltip('Đường:N'),alt.Tooltip('Ngày hiển thị:N',title='Ngày'),alt.Tooltip('Giá trị hiển thị:N',title=ylabel)])
    st.altair_chart(chart.properties(height=270),width='stretch')


def history(records, dataset, kind, analysis=None, seed=None):
    runs = [r for r in records if r['dataset'] == dataset and r['kind'] == kind and r['framework'] == 'scratch'
            and (seed is None or r['seed'] == seed)]
    if runs:
        bacc = np.mean([r['metrics']['balanced_accuracy'] for r in runs])
        st.write(f'Độ chính xác cân bằng trên dữ liệu kiểm tra: **{pct(bacc)}** '
                 '(tỷ lệ đoán đúng riêng cho mỗi nhóm rồi lấy trung bình; đoán mò là 50,0%).')
        st.caption('Số liệu của hạt giống 11.' if seed else 'Trung bình ba lần huấn luyện với hạt giống 11, 22, 33; không phải độ chính xác riêng của dự báo hôm nay.')
    else: st.warning('Chưa có số liệu độ chính xác lịch sử.')
    if analysis and dataset in MARKETS:
        ci = analysis.get('auc_ci', {}).get(dataset, {}).get(kind)
        if ci:
            st.caption(f"ROC-AUC: {vn(ci['auc'],3)}; khoảng tin cậy 95%: {vn(ci['low'],3)}–{vn(ci['high'],3)}. "
                       'Khoảng này thể hiện độ bất định khi lấy lại mẫu; lấy từ trung bình ba bản PyTorch đã lưu, gần tương đương NumPy.')
    if dataset in ('sp500', 'btc'):
        st.warning('Trên giai đoạn kiểm tra 01/2023–30/09/2026, mô hình không tốt hơn đoán mò. Xác suất này không chứng minh khả năng dự báo tương lai.')
    elif dataset == 'vnindex': st.caption('Giai đoạn kiểm tra: 01/2023–30/09/2026. Kết quả quá khứ không bảo đảm kết quả phiên tới.')
    st.caption(NOTICE + '. Xác suất chưa được hiệu chuẩn: con số mô hình đưa ra chưa chắc tương ứng tần suất xảy ra thực tế.')


def intro(records):
    guide('Bắt đầu ở đây, sau đó thử một ngày đã biết kết quả hoặc nhập thông tin để xem mô hình phản hồi.')
    def metric(dataset, kind, key):
        vals = [r['metrics'][key] for r in records if r['dataset'] == dataset and r['kind'] == kind
                and r['framework'] == 'scratch' and key in r['metrics']]
        return pct(np.mean(vals)) if vals else 'chưa có số liệu'
    st.markdown('Ứng dụng giúp bạn khám phá hai câu hỏi: ai có nguy cơ không trả được nợ, và giá phiên tới có tăng không?\n\n'
                '“Vỡ nợ” ở đây là không thanh toán khoản thẻ tín dụng theo yêu cầu; “phá sản” là doanh nghiệp không còn đáp ứng được nghĩa vụ tài chính.\n\n'
                f"Với khách hàng Đài Loan, MLP (mạng học từ một hàng thông tin) bắt được **{metric('credit_default','mlp','recall')}** số ca vỡ nợ trong tập kiểm tra, trung bình ba lần huấn luyện.\n\n"
                f"ROC-AUC có nghĩa là chọn ngẫu nhiên một ca rủi ro và một ca an toàn, mô hình chấm ca rủi ro cao hơn trong **{metric('credit_default','mlp','roc_auc')}** trường hợp; 50,0% là đoán mò.\n\n"
                f"Với VN-Index, LSTM (mạng đọc chuỗi thay đổi giá) đạt độ chính xác cân bằng **{metric('vnindex','lstm','balanced_accuracy')}**, so với mức đoán mò 50,0%; thước đo này tính tỷ lệ đúng riêng cho từng nhóm rồi lấy trung bình.\n\n"
                'Với S&P 500 và Bitcoin, mô hình không tốt hơn đoán mò trên giai đoạn kiểm tra; khi tính phí, các chiến lược theo mô hình không thắng mua và giữ.\n\n'
                'Bạn có thể đối chiếu dữ liệu kiểm tra, tải giá mới để dự báo phiên tới, nhập hồ sơ tín dụng, so sánh mô hình và xem mô phỏng bằng tiền.\n\n'
                '**' + NOTICE + '**; ứng dụng minh họa cả khả năng lẫn giới hạn của AI.')


def credit_notes():
    st.caption('Tiền là Đài tệ (TWD), không phải VND; hồ sơ Đài Loan năm 2005. '
               'Trạng thái trả nợ: −1 = trả đủ; 1–8 = trễ tương ứng 1–8 tháng; 9 = trễ từ 9 tháng. '
               'Hai mã −2 (không phát sinh chi tiêu) và 0 (chỉ trả tối thiểu) là diễn giải phổ biến, không được tài liệu UCI gốc mô tả.')


def show_credit_row(row, names):
    rows = []
    for name, value in zip(names, row):
        mapping = PAY_CODES if name.startswith('PAY_') and not name.startswith('PAY_AMT') else CATEGORIES.get(name)
        text = mapping.get(int(value), 'Không rõ') if mapping else vn(value,0)
        rows.append({'Thông tin': LABELS.get(name,name), 'Giá trị': text})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    credit_notes()


def demo(records, analysis, metadata):
    guide('Chọn một trường hợp đã xảy ra. Nhãn thật chỉ dùng để đối chiếu sau khi mô hình tính xong.')
    choices = [r for r in records if r.get('framework') == 'scratch' and r.get('seed') == 11 and r.get('kind') in ('mlp','cnn4','lstm')]
    if not choices:
        st.info('Chưa có đủ mô hình và dữ liệu kiểm tra.'); return
    dataset = st.selectbox('Bộ dữ liệu kiểm tra', list(dict.fromkeys(r['dataset'] for r in choices)), format_func=NAMES.get)
    available = [r for r in choices if r['dataset'] == dataset]
    kind = st.selectbox('Mô hình kiểm tra', [r['kind'] for r in available], format_func=KINDS.get)
    record = next(r for r in available if r['kind'] == kind)
    data = read_npz(str(ROOT / f'data/tieuluan/processed/{dataset}.npz'))
    features, labels = data['test_'+record['representation']], data['test_y']
    if not len(labels): st.info('Chưa có mẫu kiểm tra.'); return
    if dataset in MARKETS:
        dates = data['test_date'].astype(str).tolist()
        selected = st.selectbox('Ngày cuối cửa sổ quan sát', dates, format_func=day, key='test_date_'+dataset)
        index = dates.index(selected)
        frame = read_prices(str(ASSETS / f'{dataset}_prices.csv'))
        window = frame[frame.date <= pd.Timestamp(selected)].tail(20)
        price_chart(window.set_index('date').rename(columns={'close':'Giá đóng cửa'}))
        st.caption('CNN4 đọc ảnh giá; LSTM đọc 20 mức thay đổi giá liên tiếp, cần thêm một giá ngay trước cửa sổ này.')
    else:
        key = 'sample_'+dataset
        if st.button('Chọn ngẫu nhiên', key='random_'+dataset): st.session_state[key] = int(np.random.default_rng().integers(len(labels)))
        index = int(st.number_input('Dòng trong tập kiểm tra (bắt đầu từ 0)', 0, len(labels)-1, key=key))
        try:
            raw = read_npz(str(ASSETS / f'{dataset}.npz'), ())
            if dataset == 'credit_default': show_credit_row(raw['raw'][index], raw['raw_names'])
            else:
                names = metadata['datasets'][dataset]['display_labels']
                st.dataframe(pd.DataFrame({'Chỉ số':[names[n] for n in raw['raw_names']],
                    'Giá trị gốc trong UCI':[vn(v,4) for v in raw['raw'][index]]}), hide_index=True)
                st.caption('UCI đã biến đổi thang đo trước khi công bố; đây là giá trị gốc của tệp UCI, không phải phần trăm kế toán có thể đọc trực tiếp.')
        except (OSError, KeyError): st.info('Chưa có tài nguyên thông tin gốc; hãy chạy prepare_app_assets.')
    probability = infer(dataset, kind, features[index:index+1])
    threshold = record['metrics']['threshold']; positive = probability >= threshold
    market = dataset in MARKETS
    st.metric('Xác suất phiên kế tiếp tăng' if market else 'Xác suất thuộc nhóm rủi ro', pct(probability,2))
    st.write('Dự báo: **' + (('Tăng' if positive else 'Không tăng') if market else ('Có rủi ro' if positive else 'Không thuộc nhóm rủi ro')) + '**')
    if market:
        change = float(data['test_return'][index])
        st.write(f"Phiên tiếp theo {day(data['test_target_date'][index])}: **{'+' if change >= 0 else '−'}{pct(abs(change),2)}**.")
    else: st.write('Kết quả thật: **' + ('Có rủi ro' if labels[index] else 'Không thuộc nhóm rủi ro') + '**.')
    st.success('Đoán đúng') if positive == bool(labels[index]) else st.warning('Đoán sai')
    st.caption(f'Ngưỡng {pct(threshold)} chọn trên validation, tức dữ liệu dành riêng để chọn ngưỡng, không dùng tập kiểm tra.')
    with st.expander('Mô hình thực sự nhìn thấy gì'):
        if record['representation'] == 'image':
            st.image(np.kron(np.clip((features[index,0]+1)/2,0,1),np.ones((12,12))), width=240,
                     caption='Ảnh GASF: mã hóa quan hệ giữa các giá, kích thước 20 × 20.')
        else: st.dataframe(pd.DataFrame({'Giá trị đã chuẩn hóa':[vn(x,5) for x in features[index].reshape(-1)]}), hide_index=True)
    saved = read_npz(str(RESULTS / f'{dataset}__{kind}__scratch__11_predictions.npz'), ('probability',))
    if np.isclose(probability, saved['probability'][index], atol=1e-6, rtol=0):
        st.caption('Đã đối chiếu với dự báo đã lưu: xác suất vừa tính trùng trong sai số cho phép.')
    else: st.warning('Dự báo khác kết quả đã lưu; cần kiểm tra dữ liệu và trọng số.')
    history(records,dataset,kind,analysis,seed=11)


def live_tab(records, analysis):
    guide('Bấm tải để lấy giá thật. Mô hình dự báo hướng phiên sau phiên đã đóng cửa gần nhất, không dự báo mức giá hoặc lợi nhuận.')
    market = st.selectbox('Thị trường dự báo', MARKETS, format_func=NAMES.get)
    if st.button('Tải lại dữ liệu mới nhất', type='primary'):
        latest_prices.clear(); st.session_state['live_requested'] = market
    if st.session_state.get('live_requested') != market:
        st.info('Bấm “Tải lại dữ liệu mới nhất” để bắt đầu. Dữ liệu được lưu tạm tối đa 30 phút.'); return
    try:
        with st.spinner('Đang lấy các phiên đã đóng cửa…'): latest = latest_prices(market)
        f = latest['frame'].tail(60).copy()
        st.write(f"Nguồn thực tế: **{latest['source']}** · Phiên cuối: **{day(f.date.iloc[-1])}**")
        if latest['fallback']: st.caption('Đang dùng nguồn dự phòng; giá có thể khác nhẹ nguồn huấn luyện.')
        if latest['stale']: st.warning(f"Phiên cuối đã cách {vn(latest['age_days'],0)} ngày. Dự báo dựa trên dữ liệu cũ, không được coi là dự báo cập nhật cho phiên sắp tới.")
        f['20 phiên mô hình dùng'] = f.close.where(f.index >= f.index[-20])
        price_chart(f.set_index('date').rename(columns={'close':'Giá đóng cửa'}))
        params = read_npz(str(ASSETS / f'{market}.npz'), ())
        features = market_features(latest['frame'].close.to_numpy(), params['scaler_mean'], params['scaler_scale'])
        for col, kind, rep in zip(st.columns(2), ('lstm','cnn4'), ('sequence','image')):
            with col:
                p = infer(market,kind,features[rep],(11,22,33)); threshold = analysis['backtest'][market][kind]['threshold']
                st.subheader(KINDS[kind] + (' · đọc chuỗi thay đổi giá' if kind == 'lstm' else ' · đọc ảnh giá'))
                st.metric('Xác suất tăng',pct(p,2))
                st.write('**' + ('Nghiêng về tăng' if p >= threshold else 'Nghiêng về không tăng') + '**')
                st.caption(f'Theo ngưỡng validation {pct(threshold)}, không nhất thiết là 50,0%. Xác suất là trung bình ba mô hình NumPy.')
                history(records,market,kind,analysis)
        updated = datetime.fromisoformat(latest['updated_at']).astimezone(ZoneInfo('Asia/Ho_Chi_Minh'))
        st.caption('Cập nhật lúc '+updated.strftime('%H:%M:%S %d/%m/%Y')+' (giờ Việt Nam).')
    except LiveDataError as error: st.warning(str(error))


def credit_tab(records, metadata):
    guide('Chọn hồ sơ thật làm mẫu, sửa thông tin rồi bấm chấm điểm. Đây là minh họa dữ liệu Đài Loan năm 2005, không đánh giá tín dụng thực tế tại Việt Nam.')
    params = read_npz(str(ASSETS / 'credit_default.npz'), ()); info = metadata['datasets']['credit_default']
    preset = st.selectbox('Hồ sơ mẫu theo xác suất trong tập kiểm tra',[0,1,2],
        format_func=lambda i:['Ít rủi ro hơn · phân vị 10%','Ở giữa · phân vị 50%','Nhiều rủi ro hơn · phân vị 90%'][i])
    names = params['feature_names'].astype(str).tolist()
    initial = dict(zip(names,params['raw'][info['presets'][preset]]))
    pay_names = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    if st.session_state.get('credit_preset') != preset:
        for name in names:
            st.session_state['credit_'+name] = int(initial[name]) if name in CATEGORIES or name == 'AGE' or name in pay_names else vn(initial[name],0)
        st.session_state['credit_preset'] = preset; st.session_state.pop('credit_result',None)
    credit_notes(); values = {}
    def field(name):
        mapping = PAY_CODES if name in pay_names else CATEGORIES.get(name)
        if mapping: values[name] = st.selectbox(LABELS[name],list(mapping),format_func=mapping.get,key='credit_'+name)
        elif name == 'AGE': values[name] = st.number_input(LABELS[name],18,120,step=1,key='credit_'+name)
        else: values[name] = st.text_input(LABELS[name],key='credit_'+name,help='Ví dụ: 130.000 hoặc 130.000,50. Dư nợ có thể âm khi trả thừa.')
    primary = ['LIMIT_BAL','AGE']+pay_names+['BILL_AMT1','PAY_AMT1']
    with st.form('credit_form'):
        cols = st.columns(2)
        for i,name in enumerate(primary):
            with cols[i%2]: field(name)
        with st.expander('Thông tin thêm'):
            cols = st.columns(2)
            for i,name in enumerate(n for n in names if n not in primary):
                with cols[i%2]: field(name)
        submitted = st.form_submit_button('Chấm điểm hồ sơ',type='primary')
    if submitted:
        try:
            for name in names:
                if name not in CATEGORIES and name != 'AGE' and name not in pay_names:
                    values[name] = parse_amount(values[name])
                    if not name.startswith('BILL') and values[name] < 0:
                        raise ValueError('Hạn mức và số tiền đã trả không được âm.')
            x = preprocess_credit([[values[n] for n in names]],params)
            st.session_state['credit_result'] = infer('credit_default','mlp',x,(11,22,33))
        except ValueError as error:
            st.session_state.pop('credit_result',None)
            st.warning(str(error))
    if 'credit_result' in st.session_state:
        p = st.session_state['credit_result']; threshold = info['threshold']
        st.metric('Xác suất vỡ nợ tháng tới',pct(p,2))
        st.write('Mức rủi ro minh họa: **'+risk_level(p,threshold)+'**')
        st.caption(f'Thấp: dưới {pct(threshold/2)}; trung bình: từ {pct(threshold/2)} đến dưới {pct(threshold)}; cao: từ {pct(threshold)}. '
                   'Ba mức là quy ước minh họa quanh ngưỡng, chưa được kiểm định như một thang tín dụng.')
        st.write('Phân loại theo ngưỡng: **'+('Có rủi ro' if p >= threshold else 'Chưa thuộc nhóm rủi ro')+'**')
        st.caption(f'Ngưỡng {pct(threshold)} tối ưu trên trung bình dự báo validation của ba hạt giống. Kết quả thuộc lần bấm chấm điểm gần nhất.')
        st.write(f"Tỷ lệ vỡ nợ chung trong bộ dữ liệu: **{pct(info['base_rate'])}**.")
        history(records,'credit_default','mlp')
    st.warning('Lưu ý đạo đức: dữ liệu có giới tính, tuổi và tình trạng hôn nhân; dùng các yếu tố này để chấm điểm tín dụng thực tế có thể bị coi là phân biệt đối xử.')


def comparison(records):
    guide('ROC-AUC càng cao thì mô hình càng phân biệt được hai nhóm; vạch 0,5 là đoán mò. So sánh trong cùng một bộ dữ liệu.')
    if not records: st.info('Chưa có kết quả đánh giá để so sánh.'); return
    dataset = st.selectbox('Bộ dữ liệu so sánh',list(dict.fromkeys(r['dataset'] for r in records)),format_func=NAMES.get)
    rows = []
    for kind,fw in sorted(set((r['kind'],r['framework']) for r in records if r['dataset'] == dataset)):
        group = [r for r in records if r['dataset'] == dataset and r['kind'] == kind and r['framework'] == fw]
        rows.append({'Mô hình':KINDS[kind],'Cài đặt':FRAMEWORKS[fw],
            'ROC-AUC':float(np.mean([r['metrics']['roc_auc'] for r in group])),
            'Độ chính xác cân bằng':float(np.mean([r['metrics']['balanced_accuracy'] for r in group]))})
    import altair as alt
    frame = pd.DataFrame(rows)
    bars = alt.Chart(frame).mark_bar().encode(x=alt.X('Mô hình:N',sort=None),y=alt.Y('ROC-AUC:Q',scale=alt.Scale(domain=[0,1]),axis=alt.Axis(labelExpr="replace(format(datum.value, '.1f'), '.', ',')")),color='Cài đặt:N',xOffset='Cài đặt:N')
    ref = pd.DataFrame({'mức':[.5],'nhãn':['0,5 · đoán mò']})
    rule = alt.Chart(ref).mark_rule(strokeDash=[5,4],color='#b42318').encode(y='mức:Q')
    label = alt.Chart(ref).mark_text(align='left',dy=-9,color='#b42318').encode(y='mức:Q',text='nhãn:N',x=alt.value(5))
    st.altair_chart((bars+rule+label).properties(height=340),width='stretch')
    frame['ROC-AUC'] = frame['ROC-AUC'].map(lambda v:vn(v,3))
    frame['Độ chính xác cân bằng'] = frame['Độ chính xác cân bằng'].map(pct)
    st.dataframe(frame,hide_index=True)
    st.caption('ROC-AUC: tỷ lệ xếp một ca dương tính cao hơn một ca âm tính chọn ngẫu nhiên. Độ chính xác cân bằng: trung bình tỷ lệ đoán đúng của mỗi nhóm. Số liệu là trung bình các lượt chạy đã lưu.')


def backtest(analysis):
    guide('Backtest là thử cách mua bán trên quá khứ. Đọc số tiền sau phí rồi so với cách mua một lần và giữ đến cuối.')
    if not analysis or not analysis.get('equity'): st.info('Chưa có kết quả backtest.'); return
    market = st.selectbox('Thị trường mô phỏng',[m for m in MARKETS if m in analysis['equity']],format_func=NAMES.get)
    curves = analysis['equity'][market]; available = [k for k in STRATEGIES if k in curves]
    picked = st.multiselect('Chiến lược',available,default=[k for k in ['buy_hold','lstm','gru'] if k in available],format_func=STRATEGIES.get)
    st.write('**Mua và giữ:** mua đầu kỳ, giữ đến cuối. **Theo mô hình:** chỉ giữ tài sản khi xác suất tăng đạt ngưỡng, còn lại giữ tiền mặt. '
             '**Lặp lại hướng phiên trước:** phiên trước tăng thì giữ tài sản ở phiên sau, ngược lại giữ tiền mặt.')
    st.write(f"Nếu bỏ **100 triệu đồng** vào đầu giai đoạn kiểm tra, dưới đây là giá trị tương đương đến **{day(curves['dates'][-1])}**.")
    st.caption('Vốn quy đổi để dễ hình dung; mô phỏng theo biến động chỉ số, không tính tỷ giá, không khẳng định có thể mua trực tiếp chỉ số.')
    if picked:
        frame = pd.DataFrame({STRATEGIES[k]:np.array(curves[k])*100 for k in picked},index=pd.to_datetime(curves['dates']))
        price_chart(frame,'Triệu đồng sau phí')
    rows = []
    for key,item in analysis['backtest'][market].items():
        final = item.get('final_wealth',curves.get(key,[float('nan')])[-1])
        periods = analysis.get('assumptions',{}).get('periods_per_year',{}).get(market,365 if market == 'btc' else 252)
        gross = (1+item['annual_return_gross']) ** (len(curves['dates'])/periods)
        rows.append({'Chiến lược':STRATEGIES.get(key,key),'Cuối kỳ trước phí (triệu đồng)':vn(gross*100,2),
            'Cuối kỳ sau phí (triệu đồng)':vn(final*100,2),'Lợi suất năm trước phí':pct(item['annual_return_gross']),
            'Lợi suất năm sau phí':pct(item['annual_return'])})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    cost = analysis.get('assumptions',{}).get('cost_per_side',{}).get(market)
    st.caption(f'Phí mỗi lần mua hoặc bán: {pct(cost,2) if cost is not None else "chưa có số liệu"}. '
               'Trước phí chưa trừ chi phí giao dịch; sau phí đã trừ. Giả định khớp lệnh đúng giá đóng cửa và tiền mặt không sinh lãi, là mô phỏng lạc quan. '+NOTICE+'.')


def limits():
    guide('Phân biệt ngày chốt thực nghiệm với ngày tải giá mới. Tải giá mới không làm mô hình học thêm.')
    st.markdown('- **Dữ liệu kiểm tra:** thị trường chốt 30/09/2026; học đến 2019, chọn ngưỡng trong 2020–2022, kiểm tra từ 2023. UCI chia 60% học, 20% chọn ngưỡng, 20% kiểm tra.\n'
        '- **Nguồn giá mới:** S&P 500 từ Yahoo Finance (^GSPC), dự phòng FRED; Bitcoin từ Yahoo Finance (BTC-USD), dự phòng Coinbase Exchange; VN-Index từ SSI iBoard, dự phòng DNSE Entrade. Giá giữa các nguồn có thể khác nhẹ.\n'
        '- **Chỉ phiên đã đóng:** VN-Index sau 15:00 giờ Việt Nam; S&P 500 sau 16:30 giờ New York; Bitcoin bỏ nến ngày UTC hiện tại. UTC là giờ quốc tế, chậm hơn Việt Nam 7 giờ.\n'
        '- **Dữ liệu cũ hoặc mạng lỗi:** cảnh báo nếu chứng khoán cũ hơn 5 ngày hoặc Bitcoin cũ hơn 2 ngày; không thay bằng giá giả. Mốc đóng cửa cố định chưa bao gồm lịch nghỉ và phiên đặc biệt.\n'
        '- **Mô hình giữ nguyên:** CNN4 đọc ảnh giá, LSTM đọc lợi suất (mức thay đổi tương đối của giá), MLP đọc thông tin khách hàng. Chuẩn hóa là đổi thang đo bằng tham số học từ dữ liệu huấn luyện.\n'
        '- **Giới hạn:** kinh tế thay đổi có thể khiến mô hình mất hiệu quả; xác suất chưa hiệu chuẩn, không phải lời hứa có lãi. Hồ sơ nhập được tính trong phiên ứng dụng, không được ứng dụng ghi ra tệp.')
    st.caption(NOTICE)


def main():
    st.set_page_config(page_title='Hiểu AI qua dữ liệu tài chính',layout='wide')
    st.title('Hiểu AI qua dữ liệu tài chính')
    st.write('Thử một trường hợp, xem kết quả và hiểu vì sao một dự báo có thể sai.')
    records = read_records(str(RESULTS))
    try: analysis = read_json(str(RESULTS / 'analysis.json'))
    except (OSError,ValueError): analysis = {}
    try: metadata = read_json(str(ASSETS / 'metadata.json'))
    except (OSError,ValueError): metadata = {}
    tabs = st.tabs(['Giới thiệu','Thử trên dữ liệu kiểm tra','Dự báo phiên tới','Tự chấm điểm tín dụng','So sánh mô hình','Backtest','Dữ liệu và giới hạn'])
    renderers = [lambda:intro(records),lambda:demo(records,analysis,metadata),lambda:live_tab(records,analysis),
                 lambda:credit_tab(records,metadata),lambda:comparison(records),lambda:backtest(analysis),limits]
    for tab,render in zip(tabs,renderers):
        with tab:
            try: render()
            except (OSError,ValueError,KeyError,IndexError,EOFError) as error:
                st.warning('Chưa có đủ dữ liệu hoặc trọng số cho mục này. Hãy kiểm tra gói tài nguyên và chạy prepare_app_assets nếu cần.')
                st.caption(f'Chi tiết cho người quản trị: {type(error).__name__}.')


if __name__ == '__main__': main()
