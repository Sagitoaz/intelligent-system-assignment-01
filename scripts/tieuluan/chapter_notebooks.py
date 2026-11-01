"""Ba notebook của Chương 2, 3, 4 (mỗi chương đúng một notebook), đặt trong notebooks/tieuluan/complete/.

  02_ml.ipynb        Chương 2: dự đoán khoản vay không trả được nợ (Lending Club 150.000 dòng), hồi quy logistic,
                     rừng ngẫu nhiên, MLP bằng NumPy/PyTorch/Keras.
  03_cnn.ipynb       Chương 3: phần A thực nghiệm CNN trên S&P 500, VN-Index, Bitcoin (ảnh GASF);
                     phần B cài CNN 4 Conv + 2 Dense bằng Scratch/Keras/PyTorch.
  04_rnn_lstm.ipynb  Chương 4: phần A thực nghiệm RNN/LSTM/GRU và backtest; phần B cài RNN và LSTM 5 lớp hồi quy + Dense.

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.build_complete_notebooks   (dựng và chạy; mặc định chỉ ba notebook này)
"""
import json
import re
import sys
import tempfile
import textwrap
from pathlib import Path

import nbformat as nbf

from scripts.tieuluan import build_complete_notebooks as bc
from scripts.tieuluan import build_notebooks as bn

ROOT = bc.ROOT
EPOCHS_A = 12      # số epoch tối đa khi huấn luyện lại ngay trong notebook (phần A của Chương 3, 4)


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(label, text):
    return nbf.v4.new_code_cell('# Cell ' + label + '\n' + textwrap.dedent(text).strip(), metadata={'report_label': label})


SETUP2 = '''
import os, sys, json, time, inspect
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/tieuluan/loan.py').exists())
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from src.tieuluan import loan
from src.tieuluan.experiment_models import build_models, fit_model, predict_logits

RESULTS = ROOT / 'results/tieuluan/full'
records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(RESULTS.glob('loan_default__*.json'))]
sigmoid = lambda z: .5 * (1 + np.tanh(np.asarray(z, dtype=float) / 2))
print('Dự án:', ROOT, '| số bản ghi huấn luyện đã lưu của Chương 2:', len(records))
'''


def chapter2():
    cells = [md('''
    # Chương 2. Học máy cơ bản: dự đoán khoản vay có trả được nợ hay không

    **Vấn đề.** Một nền tảng cho vay nhận hồ sơ xin vay và phải trả lời: người này nhiều khả năng sẽ trả đủ hay sẽ bỏ cuộc
    giữa chừng (khoản vay bị *xóa nợ*, tiếng Anh *charged off*)? Đây là bài toán **phân loại nhị phân trên dữ liệu bảng**:
    mỗi dòng là một khoản vay, mỗi cột là một thông tin trong hồ sơ, nhãn `default` = 1 nếu khoản vay không được trả đủ.

    **Dữ liệu.** Lending Club (cho vay ngang hàng ở Mỹ), bản sao Hugging Face `codesignal/lending-club-loan-accepted`, giấy phép CC0.
    Từ 2.260.701 khoản vay gốc, giữ khoản vay **36 tháng phát hành 01/2012 đến 06/2015** có kết cục cuối cùng (427.252 khoản) rồi
    **lấy mẫu ngẫu nhiên 150.000 dòng** lưu CSV. Khoản còn "đang trả" bị loại để nhãn không bị lệch. Mọi cột đưa vào mô hình đều biết
    *tại thời điểm xin vay* (không dùng số tiền đã trả hay ngày thanh toán, vì đó là rò rỉ tương lai).

    **Ba mô hình so sánh.** Hồi quy logistic (một công thức cộng điểm), rừng ngẫu nhiên (nhiều cây quyết định bỏ phiếu) và mạng MLP
    d → 16 → 1 viết bằng ba cách (NumPy tự viết, PyTorch, Keras) với cùng trọng số khởi tạo.

    **Cách chạy.** *Restart Kernel → Run All*, khoảng 3–5 phút trên CPU. Huấn luyện dùng đúng cấu hình của kết quả đã lưu
    (mini-batch 256, tối đa 25 epoch, dừng sớm sau 5 epoch không cải thiện, hạt giống 11) nên số đo gần như trùng bản đã lưu.
    Notebook không ghi đè dữ liệu hay kết quả.
    '''), code('ML-01', SETUP2),
    md('''
    ## 1. Nhìn dữ liệu thô

    Ô dưới đọc tệp CSV gốc (không phải ma trận đã xử lý), in vài dòng, kích thước, tỉ lệ nhãn và cột thiếu giá trị.
    **Cách đọc:** `default` = 1 là khoản vay không được trả đủ. Tỉ lệ này chỉ khoảng 14%, tức dữ liệu **mất cân bằng**:
    một mô hình ngây ngô luôn đoán "trả đủ" đã đúng khoảng 86%. Vì vậy báo cáo dùng ROC-AUC, recall, balanced accuracy thay vì
    chỉ accuracy.
    '''), code('ML-02', '''
    raw = pd.read_csv(ROOT / 'data/tieuluan/raw/lending_club/loans_36m_2012_2015_150k.csv', parse_dates=['issue_d'])
    display(raw.head())
    print('Kích thước:', raw.shape, '| tỉ lệ không trả đủ:', round(raw.default.mean(), 4),
          '| luôn đoán "trả đủ" đúng:', round(1 - raw.default.mean(), 4))
    miss = raw.isna().mean()
    display(miss[miss > 0].rename('tỉ lệ thiếu').to_frame())
    '''),
    md('''
    ## 2. Chia tập theo thời gian, không chia ngẫu nhiên

    Ngoài đời mô hình học từ quá khứ rồi chấm hồ sơ mới. Nếu trộn ngẫu nhiên, mô hình có thể gặp khoản vay cùng tháng với khoản đang
    kiểm tra (rò rỉ thời gian). Vì vậy: **train** đến 31/08/2014, **validation** 09–12/2014 (dùng để chọn ngưỡng quyết định,
    dừng sớm), **test** 01–06/2015 (chỉ đọc ở cuối). Biểu đồ cho thấy tỉ lệ không trả đủ **tăng dần theo thời gian**
    (khoảng 12% năm 2013, 15% năm 2015): đây là *dịch chuyển phân phối*, một lý do mô hình ngoài đời luôn yếu hơn trên dữ liệu thử.
    '''), code('ML-03', '''
    TRAIN_END, VAL_END = pd.Timestamp('2014-08-31'), pd.Timestamp('2014-12-31')
    part = np.where(raw.issue_d <= TRAIN_END, 'train', np.where(raw.issue_d <= VAL_END, 'val', 'test'))
    summary = (raw.assign(tap=part).groupby('tap').agg(so_mau=('default', 'size'), khong_tra_du=('default', 'sum'),
               ti_le=('default', 'mean'), tu=('issue_d', 'min'), den=('issue_d', 'max')).loc[['train', 'val', 'test']])
    display(summary.assign(ti_le=summary.ti_le.round(4)))
    monthly = raw.groupby('issue_d').default.agg(['mean', 'size'])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(monthly.index, monthly['size'], width=20, color='#cfe6fb', label='số khoản vay (trục trái)')
    ax2 = ax.twinx()
    ax2.plot(monthly.index, 100 * monthly['mean'], color='#e5484d', lw=2, label='% không trả đủ (trục phải)')
    for edge in (TRAIN_END, VAL_END):
        ax.axvline(edge, color='k', ls='--', lw=.8)
    ax.set_title('Số khoản vay và tỉ lệ không trả đủ theo tháng phát hành'); ax.set_ylabel('Số khoản vay'); ax2.set_ylabel('% không trả đủ')
    plt.tight_layout(); plt.show()
    '''),
    md('''
    ## 3. Từ hồ sơ thô đến ma trận đặc trưng

    `build_features` nhận bảng thô (các cột thông tin hồ sơ) và trả ma trận số. Các bước: (1) ép kiểu số; (2) tính **tiền góp hằng tháng**
    bằng công thức niên kim `A·r / (1 − (1+r)^−n)` với A là số vay, r = lãi suất tháng, n = 36; (3) tính **gánh nặng trả nợ**
    `payment_to_income` = tiền góp cả năm / thu nhập năm; (4) lấy log thu nhập và dư nợ để giảm lệch phải; (5) mã hóa one-hot cho
    nhà ở, xác minh thu nhập, mục đích vay. Đầu ra 35 cột. **Cùng hàm này được dùng ở demo web**, nên người nhập tay và dữ liệu
    huấn luyện đi qua đúng một pipeline.
    '''), code('ML-04', '''
    print(inspect.getsource(loan.monthly_payment))
    features = loan.build_features(raw)
    print('Ma trận đặc trưng:', features.shape)
    display(features.head(3).T.round(3))
    print('Ví dụ: vay 12.000 USD, lãi 13%/năm, 36 tháng => góp', round(float(loan.monthly_payment(12000, 13.0)), 2), 'USD/tháng')
    '''),
    md('''
    ## 4. Điền khuyết và chuẩn hóa chỉ học trên train

    Giá trị thiếu (số năm đi làm, một ít mức dùng thẻ) được điền bằng **trung vị của train**; rồi mỗi cột được chuẩn hóa
    `z = (x − mean_train) / std_train`. Nếu tính trung bình trên cả test, thông tin của test đã lọt vào đầu vào (*data leakage*).
    Ô dưới tự tính lại các tham số từ train và **kiểm tra trùng khít** với tệp `processed/loan_default.npz` đã dùng để huấn luyện.
    '''), code('ML-05', '''
    is_train = (raw.issue_d <= TRAIN_END).to_numpy()
    median = features[is_train].median()
    filled = features.fillna(median)
    mean, std = filled[is_train].mean(), filled[is_train].std(ddof=0).replace(0, 1.0)
    with np.load(ROOT / 'data/tieuluan/processed/loan_default.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    assert np.allclose(median.to_numpy(), data['imputer_median']) and np.allclose(mean.to_numpy(), data['scaler_mean'])
    assert np.allclose(std.to_numpy(), data['scaler_scale'])
    x, y, xv, yv, xt, yt = (data[k] for k in ['train_tabular', 'train_y', 'val_tabular', 'val_y', 'test_tabular', 'test_y'])
    assert abs(x.mean()) < 1e-3 and len(y) == int(is_train.sum())
    print('Khớp dữ liệu đã xử lý. train', x.shape, '| val', xv.shape, '| test', xt.shape)
    print('Trung bình cột train sau chuẩn hóa ~ 0:', float(np.abs(x.mean(0)).max()), '; trên test lệch nhẹ (bình thường):',
          round(float(np.abs(xt.mean(0)).max()), 3))
    '''),
    md('''
    ## 5. Hai mô hình đối chứng: hồi quy logistic và rừng ngẫu nhiên

    **Hồi quy logistic:** xác suất `p = σ(w·x + b)`, với σ(z) = 1/(1+e⁻ᶻ); học w, b bằng cực tiểu hóa cross-entropy. Mỗi trọng số cho biết
    đặc trưng đó đẩy rủi ro lên hay xuống nên rất dễ giải thích. **Rừng ngẫu nhiên:** 150 cây quyết định (độ sâu tối đa 8, mỗi lá ≥ 20 mẫu)
    mỗi cây học trên một mẫu bootstrap rồi lấy trung bình xác suất. Cả hai là mô hình đơn giản nhưng thường rất mạnh trên dữ liệu bảng.
    '''), code('ML-06', '''
    def fit_sklearn(model, name):
        """Huấn luyện trên train, trả về xác suất val/test và thời gian."""
        start = time.perf_counter()
        model.fit(x, y)
        return {'name': name, 'val': model.predict_proba(xv)[:, 1], 'test': model.predict_proba(xt)[:, 1],
                'seconds': time.perf_counter() - start, 'model': model}

    runs = {'Hồi quy logistic': fit_sklearn(LogisticRegression(max_iter=2000, C=1., random_state=11), 'logistic'),
            'Rừng ngẫu nhiên': fit_sklearn(RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=20, n_jobs=4,
                                                                  random_state=11), 'random_forest')}
    for name, r in runs.items():
        print(f"{name}: ROC-AUC test = {roc_auc_score(yt, r['test']):.4f} ({r['seconds']:.1f} giây)")
    weights = pd.Series(runs['Hồi quy logistic']['model'].coef_[0], index=data['feature_names']).sort_values()
    display(pd.concat([weights.head(5), weights.tail(5)]).rename('trọng số (đã chuẩn hóa)').round(3).to_frame())
    '''),
    md('''
    **Đọc bảng trọng số.** Trọng số dương nghĩa là đặc trưng đó làm tăng khả năng không trả đủ (lãi suất, gánh nặng trả nợ, mức nợ trên thu nhập),
    âm nghĩa là giảm (điểm FICO cao, thu nhập cao). Vì đã chuẩn hóa nên có thể so độ mạnh giữa các trọng số. **Lưu ý:** `loan_amnt` (số vay) và
    `installment` (tiền góp hằng tháng) gần như cùng một đại lượng nên hai trọng số lớn, trái dấu, triệt tiêu nhau; đừng đọc riêng từng cái
    (hiện tượng *đa cộng tuyến*): điều đáng tin hơn là tác động chung của chúng.
    '''),
    md('''
    ## 6. Mạng MLP: ba cách cài đặt, cùng một kiến trúc

    MLP: `x (35 giá trị) → Dense(16) → ReLU → Dense(1) → logit`, rồi `p = σ(logit)`. Hàm mất mát là **BCE trên logit**
    `max(z,0) − z·y + log(1+e^−|z|)` (ổn định số). Cả ba bản (NumPy tự viết có `forward/backward` riêng, PyTorch autograd, Keras)
    dùng **cùng trọng số khởi tạo**, cùng thứ tự mini-batch 256, Adam (lr 0,001), cắt chuẩn gradient 1,0, dừng sớm theo BCE validation
    và khôi phục checkpoint tốt nhất. `fit_model` làm tất cả; ta chỉ gọi và đo.
    '''), code('ML-07', '''
    print(inspect.getsource(loan.preprocess))
    mlp = {}
    rows = []
    for framework, label in (('scratch', 'NumPy tự viết'), ('pytorch', 'PyTorch'), ('keras', 'Keras')):
        model = build_models('mlp', x.shape[1:], seed=11, frameworks=(framework,))[framework]
        fit = fit_model(model, framework, 'mlp', x, y, xv, yv, seed=11, epochs=25, batch_size=256, patience=5)
        mlp[framework] = {'val': sigmoid(predict_logits(model, framework, xv)), 'test': sigmoid(predict_logits(model, framework, xt))}
        saved = next(r for r in records if r['kind'] == 'mlp' and r['framework'] == framework and r['seed'] == 11)
        rows.append({'Cài đặt': label, 'Số epoch đã chạy': fit['epochs_run'], 'Epoch tốt nhất': fit['best_epoch'],
                     'Giây': round(fit['training_seconds'], 1), 'ROC-AUC test': round(roc_auc_score(yt, mlp[framework]['test']), 4),
                     'ROC-AUC đã lưu': round(saved['metrics']['roc_auc'], 4)})
    display(pd.DataFrame(rows))
    '''),
    md('''
    **Đọc kết quả.** Với hạt giống 11, mạng dùng hết 25 epoch (validation loss vẫn giảm rất chậm); hạt giống 22 và 33 dừng sớm ở epoch 15 và 21.
    Ba bản MLP cho ROC-AUC gần như trùng nhau và trùng bản đã lưu (Keras có thể lệch ở chữ số thập phân thứ ba–tư do
    cách đặt ε của Adam): chứng tỏ bản NumPy tự viết tính đúng. Cột "Giây" cho thấy cùng một mạng nhỏ nhưng thư viện lớn có chi phí khởi tạo cao hơn.
    '''),
    md('''
    ## 7. Ngưỡng quyết định và các thước đo

    Mô hình trả xác suất; muốn "cảnh báo / không cảnh báo" phải chọn **ngưỡng**. Vì lớp dương ít (14%), ngưỡng 0,5 gần như không bao giờ
    cảnh báo. Ta chọn ngưỡng trên **validation** sao cho **balanced accuracy** (trung bình tỉ lệ đúng của mỗi lớp) lớn nhất, rồi áp
    nguyên ngưỡng đó cho test. Các thước đo: **ROC-AUC** (xếp hạng khoản xấu cao hơn khoản tốt, 0,5 là đoán mò), **recall**
    (bắt được bao nhiêu % khoản xấu), **precision** (trong số cảnh báo, bao nhiêu % thật sự xấu).
    '''), code('ML-08', '''
    def pick_threshold(y_val, p_val):
        grid = np.linspace(.05, .95, 91)
        scores = [balanced_accuracy_score(y_val, p_val >= t) for t in grid]
        return float(grid[int(np.argmax(scores))]), grid, scores

    def evaluate(name, p_val, p_test):
        t, _, _ = pick_threshold(yv, p_val)
        pred = p_test >= t
        return {'Mô hình': name, 'Ngưỡng': round(t, 2), 'ROC-AUC': round(roc_auc_score(yt, p_test), 4),
                'Balanced acc.': round(balanced_accuracy_score(yt, pred), 4),
                'Bắt được khoản xấu (recall)': round(recall_score(yt, pred), 4),
                'Cảnh báo đúng (precision)': round(precision_score(yt, pred), 4)}

    table = [evaluate(n, r['val'], r['test']) for n, r in runs.items()]
    table.append(evaluate('MLP (NumPy)', mlp['scratch']['val'], mlp['scratch']['test']))
    table.append({'Mô hình': 'Luôn đoán "trả đủ" (đối chứng)', 'Ngưỡng': 0.5, 'ROC-AUC': 0.5,
                  'Balanced acc.': 0.5, 'Bắt được khoản xấu (recall)': 0.0, 'Cảnh báo đúng (precision)': float('nan')})
    display(pd.DataFrame(table))
    t, grid, scores = pick_threshold(yv, mlp['scratch']['val'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(grid, scores, color='#2589e0'); axes[0].axvline(t, color='#e5484d', ls='--')
    axes[0].set_title(f'Chọn ngưỡng trên validation (tốt nhất {t:.2f})'); axes[0].set_xlabel('Ngưỡng'); axes[0].set_ylabel('Balanced accuracy')
    for name, p in [('Logistic', runs['Hồi quy logistic']['test']), ('Rừng ngẫu nhiên', runs['Rừng ngẫu nhiên']['test']),
                    ('MLP', mlp['scratch']['test'])]:
        fpr, tpr, _ = roc_curve(yt, p)
        axes[1].plot(fpr, tpr, label=f'{name} (AUC {roc_auc_score(yt, p):.3f})')
    axes[1].plot([0, 1], [0, 1], 'k:', label='Đoán mò'); axes[1].legend(); axes[1].set_xlabel('Báo nhầm (tỉ lệ khoản tốt bị cảnh báo)')
    axes[1].set_ylabel('Bắt được khoản xấu'); axes[1].set_title('Đường ROC trên tập test')
    plt.tight_layout(); plt.show()
    print('Ma trận nhầm lẫn của MLP (hàng = thật, cột = dự đoán):')
    display(pd.DataFrame(confusion_matrix(yt, mlp['scratch']['test'] >= t), index=['thật: trả đủ', 'thật: không trả đủ'],
                         columns=['đoán: trả đủ', 'đoán: không trả đủ']))
    '''),
    md('''
    **Đọc kết quả.** ROC-AUC của cả ba mô hình đều xấp xỉ 0,68: **hơn đoán mò rõ rệt nhưng không cao**. Đường ROC xa đường chéo nhưng
    còn xa góc trên trái. Với ngưỡng chọn trên validation, mô hình bắt được khoảng 57–69 trong 100 khoản xấu (tùy mô hình và hạt giống) nhưng chỉ khoảng 23–24%
    cảnh báo là đúng (hơn ba phần tư số hồ sơ bị cảnh báo thật ra là hồ sơ tốt bị báo nhầm). Ba mô hình chênh nhau rất ít, nghĩa là **mạng nơ-ron không tự động tốt hơn
    mô hình đơn giản** trên dữ liệu bảng cỡ này.
    '''),
    md('''
    ## 8. Giải thích một hồ sơ: vì sao mô hình đưa ra con số này?

    Hàm `explain` lấy một hồ sơ thật ở tập test, rồi lần lượt **thay từng yếu tố bằng mức thông thường** (trung vị train; với nhóm: giá trị phổ biến)
    và đo xác suất của MLP thay đổi bao nhiêu. Yếu tố làm xác suất đổi nhiều nhất là yếu tố ảnh hưởng nhất với hồ sơ này.
    Đây chính là cách demo web trình bày "ba yếu tố ảnh hưởng nhất".
    '''), code('ML-09', '''
    test_csv = pd.read_csv(ROOT / 'data/tieuluan/app/loan_test.csv')
    params = {k: data[k] for k in ('imputer_median', 'scaler_mean', 'scaler_scale', 'feature_names')}
    mlp_model = build_models('mlp', x.shape[1:], seed=11, frameworks=('scratch',))['scratch'].load(
        ROOT / 'models/tieuluan/full/loan_default__mlp__scratch__11.npz')

    def risk(values):
        """Xác suất không trả đủ của MLP (hạt giống 11) cho một hồ sơ dạng dict."""
        z = loan.preprocess(loan.build_features(pd.DataFrame([values])).to_numpy(), params)
        return float(sigmoid(predict_logits(mlp_model, 'scratch', z))[0])

    TYPICAL = {'home_ownership': 'MORTGAGE', 'verification_status': 'Source Verified', 'purpose': 'debt_consolidation'}
    medians = dict(zip(params['feature_names'], params['imputer_median']))

    def explain(values):
        base = risk(values)
        rows = []
        for key in loan.INPUT_COLUMNS:
            alt = dict(values); alt[key] = TYPICAL.get(key, medians.get(key))
            rows.append({'Yếu tố': loan.FACTOR_NAME[key], 'Giá trị hồ sơ': values[key], 'Mức thông thường': alt[key],
                         'Xác suất đổi (điểm %)': round(100 * (base - risk(alt)), 2)})
        return base, pd.DataFrame(rows).sort_values('Xác suất đổi (điểm %)', key=abs, ascending=False)

    row = test_csv.iloc[7]
    values = {k: (loan.FIELD[k][5] if k in loan.FIELD and pd.isna(row[k]) else row[k]) for k in loan.INPUT_COLUMNS}
    p, table = explain(values)
    print(f'Hồ sơ thật số 8 của tập test: MLP ước lượng {p:.1%} khả năng không trả đủ; kết cục thật = {int(row["default"])} (1 = không trả đủ).')
    display(table.head(5).reset_index(drop=True))
    '''),
    md('''
    ## 9. Ba hạt giống và khoảng tin cậy (đọc từ kết quả đã lưu)

    Một lần huấn luyện có thể may rủi, nên mỗi mô hình được huấn luyện với 3 hạt giống (11, 22, 33). Bảng cho trung bình ± độ lệch chuẩn
    của ROC-AUC. Khoảng tin cậy 95% bootstrap (lấy mẫu lại tập test 2.000 lần) cho biết độ bất định do chỉ có 42.081 khoản test.
    '''), code('ML-10', '''
    frame = pd.DataFrame([{'Mô hình': r['kind'], 'Cài đặt': r['framework'], 'roc_auc': r['metrics']['roc_auc'],
                           'balanced_accuracy': r['metrics']['balanced_accuracy'], 'recall': r['metrics']['recall']}
                          for r in records if r['framework'] != 'baseline'])
    display(frame.groupby(['Mô hình', 'Cài đặt']).agg(auc_tb=('roc_auc', 'mean'), auc_dlc=('roc_auc', 'std'),
            bacc=('balanced_accuracy', 'mean'), recall=('recall', 'mean'), so_lan=('roc_auc', 'size')).round(4))
    analysis = json.loads((RESULTS / 'analysis.json').read_text(encoding='utf-8'))['uci']['loan_default']
    display(pd.DataFrame([{'So sánh': k, 'Giá trị': round(v.get('auc', v.get('difference')), 4), 'CI95 thấp': round(v['low'], 4),
                           'CI95 cao': round(v['high'], 4)} for k, v in analysis.items()]))
    '''),
    md('''
    ## 10. Câu hỏi để trình bày

    1. Vì sao chia train/val/test **theo thời gian** thay vì ngẫu nhiên? Nếu chia ngẫu nhiên kết quả sẽ đẹp hơn hay xấu đi, và vì sao đó là ảo tưởng?
    2. Tại sao accuracy 86% không chứng tỏ mô hình tốt khi 86% khoản vay đều trả đủ? Thước đo nào thay thế?
    3. Chuẩn hóa và điền khuyết chỉ học trên train để tránh rò rỉ nào?
    4. Cột nào trong dữ liệu gốc **cố ý không dùng** (ví dụ `total_pymnt`, `last_pymnt_d`) và vì sao?
    5. Ba mô hình chênh nhau rất ít: kết luận gì về lựa chọn MLP so với hồi quy logistic cho dữ liệu bảng?
    6. Lãi suất do chính nền tảng ấn định đã chứa đánh giá rủi ro của họ: điều đó ảnh hưởng thế nào tới việc giải thích "yếu tố quan trọng"?
    7. Nếu dùng mô hình này để từ chối cho vay ngoài đời, có rủi ro công bằng nào (tuổi, nơi ở, nhóm thu nhập)? Dữ liệu 2012–2015 ở Mỹ có áp dụng được cho Việt Nam không?
    ''')]
    return cells


def _renumber(cells, letter):
    for c in cells:
        if c.cell_type == 'markdown':
            c.source = re.sub(r'(?m)^(#{2,3}) (\d+)', lambda m: f'{m.group(1)} {letter}{m.group(2)}', c.source)
    return cells


def _market_part_a(chapter):
    spec = next(s for s in bn.SPECS if s[1] == chapter)
    name, ch, title, intro, datasets, kind, kinds, representation, demo = spec
    nb = bn.notebook(ch, title, intro, datasets, kind, kinds, representation, demo)
    cells = list(nb.cells[1:])                      # bỏ ô tiêu đề cũ
    for c in cells:
        if c.cell_type == 'markdown' and c.source.startswith('## 3. Huấn luyện lại'):
            c.source = textwrap.dedent(f'''
            ## 3. Huấn luyện lại bằng ba cách ({demo}, hạt giống 11, bản rút gọn)

            Cả ba bản dùng cùng trọng số khởi tạo, cùng thứ tự mini-batch (batch 64), Adam (lr 0,001), cắt chuẩn gradient 1,0.
            **Để notebook chạy trong vài phút, ở đây chỉ huấn luyện tối đa {EPOCHS_A} epoch** (dừng sớm sau 4 epoch không cải thiện),
            trong khi kết quả đã lưu của báo cáo dùng tối đa 60 epoch. Vì vậy ROC-AUC vừa huấn luyện có thể khác bản đầy đủ chút ít; cột
            "lệch xác suất" so với bản đầy đủ lớn hơn là bình thường. Điều cần kiểm tra là ba cách cài đặt cho kết quả gần nhau.
            ''').strip()
        if c.cell_type == 'code' and 'epochs=60, batch_size=64, patience=8' in c.source:
            c.source = (c.source.replace('epochs=60, batch_size=64, patience=8', f'epochs={EPOCHS_A}, batch_size=64, patience=4')
                        .replace("'ROC-AUC đã lưu'", "'ROC-AUC bản đầy đủ đã lưu'")
                        .replace("'Lệch xác suất tối đa so với file đã lưu'", "'Lệch xác suất tối đa so với bản đầy đủ đã lưu'"))
    note = {3: ('**Đọc kết quả.** Ba cách cài đặt cho ROC-AUC **giống hệt nhau** (đến 4–5 chữ số) vì cùng trọng số khởi tạo, cùng thứ tự mini-batch '
                'và cùng thuật toán: bằng chứng bản NumPy tự viết tính đúng. ROC-AUC của CNN4 trên VN-Index chỉ quanh 0,5 (bản huấn luyện đầy đủ đã lưu: 0,496; '
                'bản rút gọn dừng sớm còn thấp hơn): **chưa hơn đoán mò**. Đây là kết quả trung thực, không phải lỗi mã.'),
            4: ('**Đọc kết quả.** Ba cách cài đặt cho ROC-AUC trùng nhau đến 4–5 chữ số (vì cùng trọng số khởi tạo và thứ tự mini-batch), '
                'và bản rút gọn 12 epoch trùng bản đầy đủ đã lưu vì cả hai dừng sớm ở cùng epoch. LSTM trên VN-Index đạt ROC-AUC khoảng 0,546: '
                'cao hơn 0,5 chút ít, nhưng đây là một lần huấn luyện; mục A4 cho khoảng tin cậy và ba hạt giống.')}[chapter]
    for i, c in enumerate(cells):
        if c.cell_type == 'code' and f'epochs={EPOCHS_A}, batch_size=64' in c.source:
            cells.insert(i + 1, md(note))
            break
    return _renumber(cells, 'A')


def _model_part_b(kind, header, letter='B'):
    cells = bc.model_notebook(kind)
    cells[0].source = header
    return _renumber(cells, letter)


def chapter3():
    head = md('''
    # Chương 3. Mạng nơ-ron tích chập (CNN) đọc ảnh GASF của chuỗi giá

    Notebook này gộp hai phần:

    - **Phần A: thực nghiệm.** Dự báo hướng phiên kế tiếp của **S&P 500, VN-Index, Bitcoin** từ ảnh GASF 20 × 20 của 20 giá đóng cửa gần nhất;
      so sánh CNN4, CNN8, CNN sâu với hai đường cơ sở; kiểm tra ranh giới thời gian; huấn luyện lại bản rút gọn bằng ba công cụ.
    - **Phần B: cài đặt.** CNN **4 lớp tích chập + 2 lớp Dense** bằng NumPy tự viết, Keras và PyTorch, từ công thức hợp hàm,
      từng hàm, chép trọng số, vòng học, đến biểu đồ loss.

    **Cách đọc.** Đọc ô Markdown trước mỗi ô mã: vấn đề, công thức, ý nghĩa từng hàm, cách đọc kết quả, rồi câu hỏi trình bày ở cuối.
    **Cách chạy.** *Restart Kernel → Run All*, vài phút trên CPU; phần huấn luyện được thu nhỏ (ghi rõ ở từng mục) và chỉ ghi vào
    `results/tieuluan/complete/notebooks/`. Dữ liệu thị trường chốt tại 30/09/2026.
    ''')
    sep = md('# Phần A · Thực nghiệm CNN trên ba thị trường')
    part_b = md('# Phần B · Cài đặt CNN 4 Conv + 2 Dense bằng Scratch, Keras, PyTorch')
    return [head, sep] + _market_part_a(3) + [part_b] + _model_part_b(
        'cnn', '**CNN: sáu lớp học được, Scratch – Keras – PyTorch.** Mỗi nhóm ô trình bày **công thức → hàm → code → kết quả → giải thích**. '
               'Dữ liệu là VN-Index thật đã chia theo thời gian; để chạy nhanh chỉ dùng 256 mẫu train, 128 validation, 256 test và 3 epoch. '
               'Đây là thực nghiệm trên tập con để minh họa cơ chế, không thay thế kết quả đầy đủ của phần A.')


def chapter4():
    head = md('''
    # Chương 4. Mạng nơ-ron hồi quy (RNN, LSTM) đọc chuỗi lợi suất, kèm backtest

    Notebook này gộp hai phần:

    - **Phần A: thực nghiệm.** Mạng đọc trực tiếp **20 lợi suất logarit** gần nhất của S&P 500, VN-Index, Bitcoin để đoán phiên kế tiếp
      tăng hay không; so sánh SimpleRNN, LSTM, GRU với hai đường cơ sở; khoảng tin cậy ROC-AUC; **backtest có phí giao dịch**
      so với mua và giữ.
    - **Phần B và C: cài đặt.** RNN (phần B) và LSTM (phần C), mỗi mạng **5 lớp hồi quy + 1 Dense**, bằng NumPy tự viết, Keras và PyTorch
      (công thức, từng hàm, huấn luyện).

    **Cách đọc.** Đọc ô Markdown trước mỗi ô mã. **Cách chạy.** *Restart Kernel → Run All*, vài phút trên CPU; phần huấn luyện được thu nhỏ
    (ghi rõ ở từng mục). Kết quả backtest là mô phỏng trên quá khứ, không phải khuyến nghị đầu tư.
    ''')
    sep = md('# Phần A · Thực nghiệm RNN/LSTM/GRU và backtest')
    b = md('# Phần B · Cài đặt RNN 5 lớp hồi quy + Dense bằng Scratch, Keras, PyTorch')
    c = md('# Phần C · Cài đặt LSTM 5 lớp hồi quy + Dense bằng Scratch, Keras, PyTorch')
    return ([head, sep] + _market_part_a(4) + [b] + _model_part_b(
        'rnn', '**RNN: năm lớp hồi quy + Dense, Scratch – Keras – PyTorch.** Cùng cấu trúc **công thức → hàm → code → kết quả → giải thích**. '
               'Dữ liệu VN-Index thật; để chạy nhanh chỉ dùng 256 mẫu train, 128 validation, 256 test và 3 epoch (thực nghiệm minh họa trên tập con).')
        + [c] + _model_part_b('lstm', '**LSTM: năm lớp hồi quy + Dense, Scratch – Keras – PyTorch.** Giống phần B nhưng mỗi lớp là LSTM '
                                      '(cổng quên, cổng vào, cổng ra và ô nhớ). Cùng dữ liệu rút gọn và 3 epoch.', 'C'))


SPECS = [('02_ml', chapter2), ('03_cnn', chapter3), ('04_rnn_lstm', chapter4)]


def execute(nb):
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    with tempfile.TemporaryDirectory(prefix='tieuluan_kernel_') as temp:
        kernel_dir = Path(temp) / 'tieuluan'
        kernel_dir.mkdir()
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Tiểu luận venv', 'language': 'python'}), encoding='utf-8')
        manager = KernelManager(kernel_name='tieuluan', kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]))
        client = NotebookClient(nb, km=manager, timeout=1800, resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)


def build(names=None, run=True):
    out = ROOT / 'notebooks/tieuluan/complete'
    out.mkdir(parents=True, exist_ok=True)
    for name, maker in SPECS:
        if names and name not in names:
            continue
        nb = nbf.v4.new_notebook(cells=maker(), metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': sys.version.split()[0]}})
        if run:
            execute(nb)
            for c in nb.cells:
                if c.cell_type == 'code':
                    assert c.execution_count is not None
                    assert not any(o.output_type == 'error' for o in c.outputs)
        nbf.validate(nb)
        nbf.write(nb, out / f'{name}.ipynb')
        print(f'{name}.ipynb: {len(nb.cells)} ô, đã chạy = {run}', flush=True)
