"""Chương 2 · Học máy cơ bản: chấm điểm một hồ sơ vay bằng MLP, đối chiếu với hồ sơ thật."""
import numpy as np
import pandas as pd
import streamlit as st

from common import (AMBER, BLUE, GRAY, GREEN, RED, KIND_NAMES, gauge, hero, howto, mean_metric, pct, read_csv, read_npz, safe_json, step_title, verdict, vn)
import common
from src.tieuluan import loan
from src.tieuluan.live import load_model, predict_probability

SEEDS = (11, 22, 33)
DATASET = 'loan_default'
TYPICAL_CATEGORY = {'home_ownership': 'MORTGAGE', 'verification_status': 'Source Verified', 'purpose': 'debt_consolidation'}


@st.cache_resource(show_spinner=False)
def mlp_models(root):
    d = len(loan.FEATURES)
    return [load_model(root, DATASET, 'mlp', (d,), s) for s in SEEDS]


@st.cache_resource(show_spinner=False)
def sk_model(root, kind):
    import joblib
    return joblib.load(f'{root}/models/tieuluan/full/{DATASET}__{kind}__sklearn__11.joblib')


def score(values, params, root):
    """Trả về xác suất của ba mô hình cho một hồ sơ (dict giá trị thô)."""
    frame = pd.DataFrame([values])
    x = loan.preprocess(loan.build_features(frame).to_numpy(), params)
    out = {'mlp': float(predict_probability(mlp_models(str(root)), x)[0])}
    for kind in ('logistic', 'random_forest'):
        try:
            out[kind] = float(sk_model(str(root), kind).predict_proba(x)[0, 1])
        except (OSError, ValueError, ImportError):
            pass
    return out, x


def explain(values, params, base_p, root):
    """Với từng yếu tố: thay bằng mức thông thường rồi xem xác suất đổi bao nhiêu (đo ảnh hưởng thật của MLP)."""
    medians = dict(zip(params['feature_names'].astype(str).tolist(), params['imputer_median']))
    rows = []
    for key in loan.INPUT_COLUMNS:
        alt = dict(values)
        alt[key] = TYPICAL_CATEGORY[key] if key in TYPICAL_CATEGORY else medians[key]
        p_alt = score(alt, params, root)[0]['mlp']
        rows.append((key, base_p - p_alt, alt[key]))
    return sorted(rows, key=lambda r: -abs(r[1]))


def describe(key, value):
    if key in dict((c[0], c[2]) for c in loan.CATEGORICAL_FIELDS):
        mapping = {c[0]: c[2] for c in loan.CATEGORICAL_FIELDS}[key]
        return mapping.get(value, str(value))
    f = loan.FIELD[key]
    digits = 0 if isinstance(f[3], int) and f[6] >= 1 else 1
    return f'{vn(value, digits)} {f[2]}'


def cast(key, value):
    f = loan.FIELD[key]
    lo, hi = f[3], f[4]
    value = min(max(float(value), lo), hi)
    return int(round(value)) if isinstance(lo, int) and isinstance(f[6], int) else float(value)


def set_values(values):
    for key in loan.INPUT_COLUMNS:
        st.session_state['ln_' + key] = cast(key, values[key]) if key in loan.FIELD else values[key]


def pick_random(test, n):
    idx = int(np.random.default_rng().integers(n))
    row = test.iloc[idx]
    vals = {}
    for key in loan.INPUT_COLUMNS:
        v = row[key]
        vals[key] = loan.FIELD[key][5] if key in loan.FIELD and pd.isna(v) else v
    set_values(vals)
    st.session_state['ln_row'] = idx


def clear_row():
    st.session_state['ln_row'] = None


def inputs(test):
    """Sáu thông tin quan trọng nhất nằm ngay trên màn hình; phần còn lại thu vào ô mở rộng."""
    values = {}
    primary = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico', 'revol_util']
    for key in loan.INPUT_COLUMNS:                       # giá trị mặc định lần đầu
        if 'ln_' + key not in st.session_state and key in loan.FIELD:
            st.session_state['ln_' + key] = cast(key, loan.FIELD[key][5])
    st.session_state.setdefault('ln_home_ownership', 'MORTGAGE')
    st.session_state.setdefault('ln_verification_status', 'Source Verified')
    st.session_state.setdefault('ln_purpose', 'debt_consolidation')

    def slider(key):
        f = loan.FIELD[key]
        values[key] = st.slider(f'{f[1]} ({f[2]})', f[3], f[4], step=f[6], key='ln_' + key, help=f[7] or None, on_change=clear_row)
    with st.container(border=True):
        st.markdown('**Thông tin chính của hồ sơ**')
        for key in primary:
            slider(key)
    with st.expander('Thông tin chi tiết hơn (tùy chọn, đã có sẵn mức thông thường)'):
        for key in loan.INPUT_COLUMNS:
            if key in loan.FIELD and key not in primary:
                slider(key)
        for key, label, mapping in loan.CATEGORICAL_FIELDS:
            values[key] = st.selectbox(label, list(mapping), format_func=mapping.get, key='ln_' + key, on_change=clear_row)
    return values


def result_panel(values, params, base):
    """Cột kết quả: câu kết luận, thanh đo, ba yếu tố, so sánh ba mô hình. Trả về xác suất MLP hoặc None."""
    try:
        probs, _x = score(values, params, common.ROOT)
    except (OSError, ValueError, KeyError) as error:
        st.warning('Chưa chấm được hồ sơ này vì thiếu tệp mô hình hoặc dữ liệu. Hãy chạy lại bước huấn luyện Chương 2.')
        st.caption(f'Chi tiết cho người quản trị: {type(error).__name__}.')
        return None
    p = probs['mlp']
    band = loan.risk_band(p, base)
    tone = {'thấp': 'low', 'trung bình': 'mid', 'cao': 'high'}[band]
    step_title(2, 'Mô hình trả lời')
    verdict(f'Mô hình ước lượng khoảng <b>{pct(p, 0)}</b> khả năng khoản vay này <b>không được trả đủ</b>. '
            f'Mức rủi ro: <b>{band}</b> (trung bình toàn bộ khoản vay trong dữ liệu là {pct(base, 0)}).', tone)
    gauge(p, 'An toàn hơn', 'Rủi ro hơn', mark=base, mark_label='mức rủi ro trung bình của dữ liệu')
    st.caption(f'Hiểu nôm na: nếu có 100 hồ sơ giống hồ sơ này, mô hình kỳ vọng khoảng {vn(p * 100, 0)} khoản không được trả đủ. '
               'Con số chưa được hiệu chuẩn nghiêm ngặt nên chỉ dùng để so sánh tương đối giữa các hồ sơ.')

    step_title(3, 'Vì sao có kết luận này?')
    factors = explain(values, params, p, common.ROOT)[:3]
    lines = []
    for key, delta, typical in factors:
        direction = 'làm tăng' if delta > 0 else 'làm giảm'
        lines.append(f'- **{loan.FACTOR_NAME[key]}: {describe(key, values[key])}** (mức thông thường: {describe(key, typical)}) '
                     f'{direction} rủi ro khoảng **{vn(abs(delta) * 100, 1)} điểm phần trăm** so với hồ sơ ở mức thông thường.')
    st.markdown('Ba yếu tố ảnh hưởng nhiều nhất tới con số trên (thử thay từng yếu tố bằng mức thông thường rồi xem mô hình đổi bao nhiêu):\n\n'
                + '\n'.join(lines))

    step_title(4, 'Ba mô hình có đồng ý với nhau không?')
    frame = pd.DataFrame({'Mô hình': [KIND_NAMES[k] for k in probs], 'Khả năng không trả được nợ': list(probs.values())})
    frame['Nhãn'] = frame['Khả năng không trả được nợ'].map(pct)
    import altair as alt
    bars = alt.Chart(frame).mark_bar(size=46, cornerRadiusEnd=5).encode(
        x=alt.X('Mô hình:N', sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y('Khả năng không trả được nợ:Q', scale=alt.Scale(domain=[0, max(.5, float(frame.iloc[:, 1].max()) * 1.25)]),
                axis=alt.Axis(format='.0%', title='Khả năng không trả được nợ')),
        color=alt.Color('Mô hình:N', legend=None, scale=alt.Scale(range=[BLUE, AMBER, GREEN])), tooltip=['Mô hình', 'Nhãn'])
    text = alt.Chart(frame).mark_text(dy=-9, fontWeight='bold').encode(x=alt.X('Mô hình:N', sort=None), y='Khả năng không trả được nợ:Q', text='Nhãn:N')
    rule = alt.Chart(pd.DataFrame({'y': [base]})).mark_rule(strokeDash=[5, 4], color=RED).encode(y='y:Q')
    st.altair_chart((bars + text + rule).properties(height=230), width='stretch')
    spread = max(probs.values()) - min(probs.values())
    st.caption(f'Đường đứt đỏ là mức rủi ro trung bình ({pct(base, 0)}). ' + (
        'Ba mô hình cho kết quả khá gần nhau: hồ sơ này không gây tranh cãi.' if spread < .06 else
        'Ba mô hình lệch nhau đáng kể: hồ sơ này nằm ở vùng khó, nên đừng tin tuyệt đối vào một con số.'))
    return p


def scoring_tab(records, params, meta, test):
    base = meta.get('base_rate', 0.13)
    threshold = meta.get('threshold', base)
    hero('Demo này làm gì?',
         'Hình dung bạn là nhân viên một nền tảng cho vay. Một người xin vay, bạn muốn biết <b>khả năng họ không trả được '
         'nợ</b>. Mô hình AI (mạng MLP) đã học từ <b>gần 90 nghìn khoản vay thật</b> của Lending Club (Mỹ, 2012–2015), '
         'trong đó biết khoản nào cuối cùng bị “xóa nợ” (người vay bỏ cuộc giữa chừng).',
         'Bạn nhập hồ sơ bằng các thanh trượt, hoặc bấm nút bên dưới để lấy một <b>hồ sơ thật</b> chưa từng được dùng để huấn luyện; '
         'mô hình sẽ trả lời bằng một câu kết luận, và với hồ sơ thật bạn còn biết kết cục thật để kiểm chứng.')
    howto('Con số phần trăm là <i>khả năng ước lượng</i> khoản vay không được trả đủ, không phải sự chắc chắn. '
          'Thanh màu càng lệch về phải là càng rủi ro; vạch mờ là mức rủi ro trung bình của toàn bộ khoản vay trong dữ liệu.')
    step_title(1, 'Chọn hồ sơ')
    b1, b2, _ = st.columns([1.3, 1.6, 3])
    b1.button('Lấy một hồ sơ thật ngẫu nhiên', type='primary', on_click=pick_random, args=(test, len(test)),
              help='Lấy ngẫu nhiên một khoản vay trong tập kiểm tra (01/2015 đến 06/2015), các thanh trượt tự điền theo hồ sơ.')
    b2.button('Về hồ sơ tự nhập', on_click=clear_row)
    row = st.session_state.get('ln_row')
    if row is not None:
        st.success(f'Đang dùng hồ sơ thật số {row + 1:,} của tập kiểm tra '.replace(',', '.') +
                   f'(phát hành {pd.Timestamp(test.iloc[row].issue_d).strftime("%m/%Y")}). Bạn vẫn có thể kéo thanh trượt để thử “nếu thay đổi thì sao”; '
                   'khi kéo, hồ sơ trở thành hồ sơ tự nhập.')
    left, right = st.columns([1, 1.2])
    with left:
        values = inputs(test)
    with right:
        p = result_panel(values, params, base)
    if p is None:
        return
    row = st.session_state.get('ln_row')
    if row is not None:
        step_title(5, 'Kiểm chứng với kết cục thật')
        truth = int(test.iloc[row]['default'])
        pred = p >= threshold
        word = 'đã KHÔNG được trả đủ (bị xóa nợ)' if truth else 'đã được trả đủ đúng hạn'
        verdict(f'Kết cục thật của khoản vay này: <b>{word}</b>. Mô hình (ngưỡng cảnh báo {pct(threshold, 0)}) '
                f'<b>{"cảnh báo rủi ro" if pred else "không cảnh báo"}</b> ⇒ '
                f'<b>{"ĐOÁN ĐÚNG" if pred == bool(truth) else "ĐOÁN SAI"}</b>.', 'low' if pred == bool(truth) else 'high')
        recall = mean_metric(records, DATASET, 'mlp', 'scratch', 'recall')
        precision = mean_metric(records, DATASET, 'mlp', 'scratch', 'precision')
        auc = mean_metric(records, DATASET, 'mlp', 'scratch', 'roc_auc')
        if recall is not None:
            st.markdown(f'Một hồ sơ chưa đủ để đánh giá mô hình. Trên **toàn bộ {vn(len(test), 0)} khoản vay kiểm tra**: '
                        f'mô hình bắt được khoảng **{pct(recall, 0)}** khoản xấu, nhưng chỉ khoảng **{pct(precision, 0)}** cảnh báo là đúng '
                        f'(còn lại là báo nhầm); ROC-AUC là **{vn(auc, 3)}** (0,5 là đoán mò, 1 là hoàn hảo). '
                        '**Nhận xét:** mô hình có ích nhưng chỉ ở mức vừa phải, vì kết cục vay phụ thuộc cả yếu tố không có trong hồ sơ (mất việc, bệnh tật…).')
    st.warning('**Giới hạn cần biết.** Dữ liệu là khoản vay 36 tháng của một nền tảng ở Mỹ, 2012–2015, nên không áp dụng trực tiếp cho '
               'ngân hàng Việt Nam hay bối cảnh hiện nay. Mô hình chỉ thấy các cột trong hồ sơ; dùng thông tin như nơi ở, học vấn hay giới tính '
               'để chấm tín dụng thật có thể gây phân biệt đối xử. Đây là minh họa học thuật, không phải quyết định cho vay.')


def comparison_tab(records, analysis):
    hero('So sánh ba mô hình',
         'Cùng một bài toán, ba cách học khác nhau: <b>hồi quy logistic</b> (một công thức cộng điểm đơn giản), '
         '<b>rừng ngẫu nhiên</b> (nhiều cây quyết định bỏ phiếu) và <b>mạng MLP</b> (mạng nơ-ron nhỏ). '
         'Biểu đồ cho biết cách nào phân biệt khoản vay tốt và xấu giỏi hơn trên tập kiểm tra chưa từng thấy.')
    howto('<b>ROC-AUC</b>: chọn ngẫu nhiên một khoản xấu và một khoản tốt, đây là tỉ lệ mô hình chấm khoản xấu rủi ro hơn. '
          '50% là đoán mò, 100% là hoàn hảo. <b>Bắt được khoản xấu</b>: trong 100 khoản xấu thật, mô hình cảnh báo được bao nhiêu '
          '(đi kèm cái giá là báo nhầm khoản tốt).')
    rows = []
    for kind, fw in (('logistic', 'sklearn'), ('random_forest', 'sklearn'), ('mlp', 'scratch')):
        auc, rec = (mean_metric(records, DATASET, kind, fw, k) for k in ('roc_auc', 'recall'))
        if auc is not None:
            rows.append({'Mô hình': KIND_NAMES[kind], 'ROC-AUC': auc, 'Bắt được khoản xấu': rec})
    if not rows:
        st.info('Chưa có kết quả huấn luyện Chương 2 để so sánh.')
        return
    frame = pd.DataFrame(rows)
    import altair as alt
    long = frame.melt('Mô hình', var_name='Thước đo', value_name='Giá trị')
    long['Nhãn'] = long['Giá trị'].map(lambda v: pct(v, 1))
    chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('Mô hình:N', sort=None, axis=alt.Axis(labelAngle=0, title=None)),
        y=alt.Y('Giá trị:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%', title='Tỉ lệ')),
        xOffset='Thước đo:N', color=alt.Color('Thước đo:N', scale=alt.Scale(range=[BLUE, AMBER]), legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Mô hình', 'Thước đo', 'Nhãn'])
    labels = alt.Chart(long).mark_text(dy=-8, fontSize=12).encode(x=alt.X('Mô hình:N', sort=None), xOffset='Thước đo:N', y='Giá trị:Q', text='Nhãn:N')
    rule = alt.Chart(pd.DataFrame({'y': [.5]})).mark_rule(strokeDash=[5, 4], color=RED).encode(y='y:Q')
    st.altair_chart((chart + labels + rule).properties(height=330), width='stretch')
    st.caption('Đường đứt đỏ: mức 50% của việc đoán mò (áp dụng cho ROC-AUC). Số liệu là trung bình của 3 lần huấn luyện với 3 hạt giống khác nhau.')
    best = frame.sort_values('ROC-AUC', ascending=False).iloc[0]
    ci = analysis.get('uci', {}).get(DATASET, {})
    gap = frame['ROC-AUC'].max() - frame['ROC-AUC'].min()
    verdict(f'<b>Kết luận:</b> mô hình phân biệt tốt nhất là <b>{best["Mô hình"]}</b> với ROC-AUC {vn(best["ROC-AUC"], 3)}; '
            f'chênh lệch giữa ba mô hình chỉ {vn(gap * 100, 1)} điểm phần trăm. '
            'Cả ba hơn hẳn đoán mò nhưng còn xa mức hoàn hảo: bài toán này khó vì kết cục vay phụ thuộc nhiều yếu tố ngoài hồ sơ. '
            'Mạng nơ-ron không tự động hơn các mô hình đơn giản trên dữ liệu dạng bảng.', 'info')
    shown = frame.copy()
    shown['ROC-AUC'] = shown['ROC-AUC'].map(lambda v: vn(v, 3))
    shown['Bắt được khoản xấu'] = shown['Bắt được khoản xấu'].map(pct)
    st.dataframe(shown, hide_index=True)
    mlp_rows = []
    for fw, name in (('scratch', 'NumPy tự viết'), ('pytorch', 'PyTorch'), ('keras', 'Keras')):
        v = mean_metric(records, DATASET, 'mlp', fw, 'roc_auc')
        if v is not None:
            mlp_rows.append({'Cách cài đặt mạng MLP': name, 'ROC-AUC': vn(v, 4)})
    if mlp_rows:
        st.markdown('**Cùng một mạng MLP cài đặt bằng ba công cụ** cho kết quả gần như trùng nhau, nghĩa là phần NumPy tự viết của đề tài tính đúng:')
        st.dataframe(pd.DataFrame(mlp_rows), hide_index=True)
    if ci:
        d = ci.get('rf_minus_mlp')
        if d:
            same = d['low'] <= 0 <= d['high']
            st.caption(f'Hiệu ROC-AUC giữa rừng ngẫu nhiên và MLP là {vn(d["difference"], 3)} (khoảng tin cậy 95%: {vn(d["low"], 3)} đến {vn(d["high"], 3)}); '
                       + ('khoảng này chứa 0 nên chưa đủ bằng chứng nói mô hình nào hơn.' if same else 'khoảng này không chứa 0 nên khác biệt có ý nghĩa thống kê.'))


def data_tab(inventory, meta):
    info = inventory.get('loan_default', {})
    hero('Dữ liệu đằng sau demo',
         'Mỗi dòng dữ liệu là <b>một khoản vay thật</b> của Lending Club: thông tin người vay lúc xin vay và kết cục cuối cùng '
         '(trả đủ hay bị xóa nợ). Không có thông tin nào sinh ra sau khi khoản vay bắt đầu, để mô hình không “gian lận” nhìn trước tương lai.')
    if not info:
        st.info('Chưa có thông tin kiểm kê dữ liệu.')
        return
    s = info['splits']
    c = st.columns(4)
    c[0].metric('Số khoản vay', vn(info['raw_rows'], 0))
    c[1].metric('Số đặc trưng đưa vào mô hình', vn(info['features'], 0))
    c[2].metric('Tỉ lệ không trả đủ', pct(info['base_rate'], 1))
    c[3].metric('Giai đoạn phát hành', f'{info["issue_range"][0][:7]} → {info["issue_range"][1][:7]}')
    frame = pd.DataFrame([{'Tập': {'train': 'Huấn luyện (học)', 'val': 'Kiểm định (chọn ngưỡng)', 'test': 'Kiểm tra (chưa từng thấy)'}[k],
                           'Số khoản vay': vn(v['n'], 0), 'Tỉ lệ không trả đủ': pct(v['positive'] / v['n'], 1),
                           'Giai đoạn phát hành': f'{v["first"]} → {v["last"]}'} for k, v in s.items()])
    st.dataframe(frame, hide_index=True)
    st.markdown('**Vì sao chia theo thời gian?** Ngoài đời, mô hình học từ quá khứ rồi chấm hồ sơ mới. Nếu trộn ngẫu nhiên, mô hình có thể '
                'vô tình “thấy” khoản vay cùng thời điểm với khoản đang kiểm tra. Chia theo ngày phát hành mô phỏng đúng tình huống thực.')
    st.markdown(f'- **Nguồn:** Lending Club, bản Hugging Face `codesignal/lending-club-loan-accepted`, giấy phép CC0 (tự do sử dụng); lấy mẫu ngẫu nhiên {vn(info["raw_rows"], 0)} dòng từ {vn(info["eligible_rows"], 0)} khoản vay đủ điều kiện.\n'
                '- **Chỉ lấy** khoản vay 36 tháng phát hành 01/2012–06/2015, vì đã kết thúc trước 2018 nên kết cục đã biết; khoản còn đang trả bị loại để nhãn không bị lệch.\n'
                '- **Hạn chế:** dữ liệu từ Mỹ, đã hơn mười năm tuổi, chỉ gồm người đã được duyệt vay; ngưỡng cảnh báo chọn trên tập kiểm định; mô hình không giải thích được nguyên nhân.')
    st.caption(f'Tập tin dữ liệu: data/tieuluan/raw/lending_club/loans_36m_2012_2015_150k.csv. Cấu hình huấn luyện MLP: tối đa {meta.get("config", {}).get("epochs_max", 25)} epoch, '
               f'mini-batch {meta.get("config", {}).get("batch_size", 256)}, dừng sớm khi tập kiểm định hết cải thiện.')


def render(records, analysis):
    params = read_npz(str(common.ASSETS / 'loan_default.npz'))
    meta = safe_json(common.ASSETS / 'loan_meta.json')
    inventory = safe_json(common.PROCESSED / 'inventory.json')
    test = read_csv(str(common.ASSETS / 'loan_test.csv'), parse_dates=['issue_d'])
    tabs = st.tabs(['1 · Chấm điểm một hồ sơ', '2 · So sánh ba mô hình', '3 · Dữ liệu và giới hạn'])
    for tab, fn in zip(tabs, (lambda: scoring_tab(records, params, meta, test), lambda: comparison_tab(records, analysis),
                              lambda: data_tab(inventory, meta))):
        with tab:
            fn()
