"""Sinh (và tùy chọn thực thi) ba notebook minh họa cho Chương 2, 3, 4 của tiểu luận.

Mỗi notebook: (1) xem dữ liệu, kích thước, phân bố lớp và dòng dữ liệu gốc; (2) xem cùng một mô hình
viết bằng NumPy, Keras, PyTorch và kiểm tra song trùng; (3) HUẤN LUYỆN LẠI một cấu hình (hạt giống 11)
bằng cả ba cách ngay trong notebook rồi đối chiếu với kết quả đã lưu; (4) tổng hợp kết quả 3 hạt giống,
khoảng tin cậy (và backtest ở Chương 4). Notebook không ghi đè dữ liệu, mô hình hay kết quả.

Chạy:  .venv\\Scripts\\python.exe -m scripts.tieuluan.build_notebooks --execute
"""
from pathlib import Path
import argparse
import json
import sys
import tempfile
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = """
import os, sys, json, inspect
from pathlib import Path
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '2')          # cùng điều kiện với scripts/tieuluan/run_experiments.py
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/tieuluan/experiment_models.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import roc_auc_score
from src.tieuluan.config import ANALYSIS_START
from src.tieuluan.experiment_models import build_models, fit_model, predict_logits

PROCESSED = ROOT / 'data/tieuluan/processed'
RESULTS = ROOT / 'results/tieuluan/full'
inventory = json.loads((PROCESSED / 'inventory.json').read_text(encoding='utf-8'))
analysis = json.loads((RESULTS / 'analysis.json').read_text(encoding='utf-8'))
records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(RESULTS.glob('*__*.json'))]
NAMES = {'taiwan_bankruptcy': 'Phá sản DN (UCI 572)', 'credit_default': 'Vỡ nợ thẻ (UCI 350)',
         'sp500': 'S&P 500', 'vnindex': 'VN-Index', 'btc': 'Bitcoin'}
FW = {'scratch': 'NumPy tự viết', 'pytorch': 'PyTorch', 'keras': 'Keras', 'sklearn': 'scikit-learn', 'baseline': 'Đường cơ sở'}
sigmoid = lambda z: .5 * (1 + np.tanh(np.asarray(z, dtype=float) / 2))
print('Số bản ghi thực nghiệm đã lưu:', len(records))
"""

DATA_TABLE = """
data = {}
rows = []
for name in DATASETS:
    with np.load(PROCESSED / f'{name}.npz', allow_pickle=False) as archive:
        data[name] = {key: archive[key] for key in archive.files}
    for split in ('train', 'val', 'test'):
        y = data[name][f'{split}_y']
        assert len(y) == inventory[name]['splits'][split]['n']          # khớp số liệu đã kiểm kê
        rows.append({'Bộ dữ liệu': NAMES[name], 'Tập': split, 'Số mẫu': len(y), 'Lớp 0': int((y == 0).sum()),
                     'Lớp 1': int((y == 1).sum()), 'Tỉ lệ lớp 1': round(float(y.mean()), 4),
                     'Kích thước đầu vào': str(data[name][f'{split}_{REPRESENTATION}'].shape[1:])})
display(pd.DataFrame(rows))
"""

TRAIN_THREE = """
x, y = data[DEMO]['train_' + REPRESENTATION], data[DEMO]['train_y']
xv, yv = data[DEMO]['val_' + REPRESENTATION], data[DEMO]['val_y']
xt, yt = data[DEMO]['test_' + REPRESENTATION], data[DEMO]['test_y']
saved = {r['framework']: r for r in records if r['dataset'] == DEMO and r['kind'] == KIND and r['seed'] == 11}
rows, curves = [], {}
for framework in ('scratch', 'pytorch', 'keras'):
    model = build_models(KIND, x.shape[1:], seed=11, frameworks=(framework,))[framework]
    fit = fit_model(model, framework, KIND, x, y, xv, yv, seed=11, epochs=60, batch_size=64, patience=8)
    p = sigmoid(predict_logits(model, framework, xt))
    with np.load(RESULTS / f'{DEMO}__{KIND}__{framework}__11_predictions.npz') as archive:
        p_saved = archive['probability']
    curves[framework] = [h['val_loss'] for h in fit['history']]
    rows.append({'Cài đặt': FW[framework], 'Epoch tốt nhất': fit['best_epoch'], 'Số epoch đã chạy': fit['epochs_run'],
                 'Giây': round(fit['training_seconds'], 1), 'ROC-AUC vừa huấn luyện': round(roc_auc_score(yt, p), 5),
                 'ROC-AUC đã lưu': round(saved[framework]['metrics']['roc_auc'], 5),
                 'Lệch xác suất tối đa so với file đã lưu': float(np.max(np.abs(p - p_saved)))})
display(pd.DataFrame(rows))
fig, ax = plt.subplots(figsize=(7, 3.2))
for (framework, values), style in zip(curves.items(), ('-o', '--s', ':^')):
    ax.plot(range(1, len(values) + 1), values, style, ms=3, label=FW[framework])
ax.set_xlabel('Epoch'); ax.set_ylabel('BCE trên validation'); ax.set_title(f'{NAMES[DEMO]} · {KIND} · hạt giống 11')
ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.show()
"""

SUMMARY = """
rows = []
for r in records:
    if r['dataset'] in DATASETS and r['kind'] in KINDS:
        rows.append({'Bộ dữ liệu': NAMES[r['dataset']], 'Mô hình': r['kind'], 'Cài đặt': FW[r['framework']],
                     'roc_auc': r['metrics']['roc_auc'], 'balanced_accuracy': r['metrics']['balanced_accuracy']})
table = (pd.DataFrame(rows).groupby(['Bộ dữ liệu', 'Mô hình', 'Cài đặt'])
         .agg(ROC_AUC_TB=('roc_auc', 'mean'), ROC_AUC_ĐLC=('roc_auc', 'std'),
              Balanced_acc=('balanced_accuracy', 'mean'), Số_lượt=('roc_auc', 'size')).round(4))
display(table)
"""


def notebook(chapter, title, intro, datasets, kind, kinds, representation, demo):
    cells = [md(f"""
    # Chương {chapter}. {title}

    {intro}

    Notebook đọc dữ liệu đã xử lý tại `data/tieuluan/processed` và kết quả đã lưu tại `results/tieuluan/full`
    (sinh bởi `scripts/tieuluan/run_experiments.py`). Mục 3 **huấn luyện lại ngay trong notebook** cùng một cấu
    hình bằng ba cách cài đặt để đối chiếu với kết quả đã lưu; mô hình huấn luyện lại chỉ nằm trong bộ nhớ,
    không ghi đè file nào. Chạy bằng **Restart Kernel → Run All**.
    """), code(SETUP), code(f"DATASETS = {datasets!r}\nKIND = {kind!r}\nKINDS = {kinds!r}\n"
                            f"REPRESENTATION = {representation!r}\nDEMO = {demo!r}"),
             md("""
    ## 1. Dữ liệu

    Kích thước và phân bố lớp đọc trực tiếp từ mảng đã lưu (không chép tay). Bộ tiền xử lý (chuẩn hóa,
    điền khuyết) chỉ học trên tập train rồi áp dụng nguyên cho validation và test.
    """), code(DATA_TABLE)]
    if representation == 'tabular':
        cells += [md("""
        **Dòng dữ liệu gốc và phân bố lớp.** Hai bộ dữ liệu đều mất cân bằng: lớp 1 (phá sản, vỡ nợ) là thiểu số,
        nên tiểu luận dùng ROC-AUC, AP và balanced accuracy thay vì accuracy.
        """), code("""
        for name in DATASETS:
            info = inventory[name]
            print(f"{NAMES[name]}: {info['raw_rows']} dòng × {info['features']} đặc trưng, {info['positive']} mẫu lớp 1, "
                  f"SHA-256 file gốc {info['sha256'][:12]}…")
            display(pd.DataFrame(info['raw_sample']))
        counts = pd.DataFrame(rows).query("Tập == 'train'").set_index('Bộ dữ liệu')[['Lớp 0', 'Lớp 1']]
        counts.plot.barh(stacked=True, figsize=(7, 2.4), color=['#2a78d6', '#eb6834'], title='Phân bố lớp trên tập train')
        plt.xlabel('Số mẫu'); plt.tight_layout(); plt.show()
        """)]
    else:
        cells += [md("""
        **Dòng dữ liệu gốc và kiểm tra ranh giới thời gian.** Train gồm các mẫu có ngày nhãn đến hết 2019, validation
        2020–2022, test từ 01/2023 đến 30/09/2026; mẫu có nhãn vượt sang tập sau bị loại. Ô dưới kiểm tra lại điều đó.
        """), code("""
        for name in DATASETS:
            raw = pd.read_csv(ROOT / f'data/tieuluan/raw/indices/{name}.csv')
            raw = raw[raw.date >= ANALYSIS_START[name]]
            print(f"{NAMES[name]}: {len(raw)} phiên từ {raw.date.iloc[0]} đến {raw.date.iloc[-1]}")
            display(raw.head(3))
            arrays = data[name]
            assert arrays['train_target_date'].max() <= np.datetime64('2019-12-31')
            assert arrays['val_date'].min() >= np.datetime64('2020-01-01')
            assert arrays['val_target_date'].max() <= np.datetime64('2022-12-31')
            assert arrays['test_date'].min() >= np.datetime64('2023-01-01')
            assert arrays['test_target_date'].max() <= np.datetime64('2026-09-30')
        print('Đã kiểm tra: không mẫu nào có nhãn vượt ranh giới giữa các tập.')
        """)]
        if representation == 'image':
            cells += [md("""
            **Ảnh GASF.** Mỗi cửa sổ 20 giá đóng cửa được chuẩn hóa min–max về [0, 1] trong chính cửa sổ, đổi thành góc
            φ = arccos(x̃) rồi tạo ma trận G[i, j] = cos(φᵢ + φⱼ). Khoảng [0, 1] được chọn vì với [−1, 1], một cửa sổ tăng
            và cửa sổ giảm đối xứng với nó cho ra cùng một ảnh (mục 3.7.2).
            """), code("""
            fig, axes = plt.subplots(1, len(DATASETS), figsize=(10, 3.2))
            for ax, name in zip(axes, DATASETS):
                arrays = data[name]
                im = ax.imshow(arrays['test_image'][0, 0], cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_title(f"{NAMES[name]}\\ncửa sổ kết thúc {arrays['test_date'][0]}", fontsize=9)
                ax.set_xlabel('Phiên j'); ax.set_ylabel('Phiên i')
            fig.colorbar(im, ax=list(axes), shrink=.8, label='GASF')
            plt.show()
            """)]
        else:
            cells += [md("""
            **Chuỗi lợi suất.** Đầu vào của mạng hồi quy là 20 lợi suất logarit gần nhất, chuẩn hóa bằng trung bình và độ
            lệch chuẩn của tập train. Biểu đồ khôi phục về đơn vị phần trăm để dễ đọc.
            """), code("""
            fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 3))
            for ax, name in zip(axes, DATASETS):
                arrays = data[name]
                r = arrays['test_sequence'][0, :, 0] * arrays['scaler_scale'][0] + arrays['scaler_mean'][0]
                ax.bar(np.arange(1, 21), 100 * r, color=np.where(r > 0, '#1baf7a', '#eb6834'))
                ax.axhline(0, color='gray', lw=.6)
                ax.set_title(f"{NAMES[name]} · nhãn = {int(arrays['test_y'][0])}", fontsize=9)
                ax.set_xlabel('Phiên trong cửa sổ'); ax.set_ylabel('Lợi suất log (%)')
            plt.tight_layout(); plt.show()
            """)]
    cells += [md("""
    ## 2. Một mô hình, ba cách cài đặt

    Mã nguồn đầy đủ nằm trong `src/tieuluan/experiment_models.py` (hàm `scratch_model`, `torch_model`, `keras_model`)
    và thư viện tự viết `src/tieuluan/scratch/`. Ô dưới in phần định nghĩa mô hình của ba hàm, rồi khởi tạo cả ba
    bản bằng **cùng một bộ trọng số NumPy** và so sánh logit đầu ra trên vài mẫu (chưa huấn luyện).
    """), code("""
    import src.tieuluan.experiment_models as em
    for fn in (em.scratch_model, em.torch_model, em.keras_model):
        print(inspect.getsource(fn))
    """), code("""
    sample = data[DEMO]['train_' + REPRESENTATION][:8].astype(np.float32)
    models = build_models(KIND, sample.shape[1:], seed=11)
    logits = {FW[f]: predict_logits(m, f, sample) for f, m in models.items()}
    display(pd.DataFrame(logits).round(6))
    base = logits['NumPy tự viết']
    print({k: float(np.max(np.abs(v - base))) for k, v in logits.items()}, '← lệch tuyệt đối lớn nhất so với NumPy')
    """), md(f"""
    ## 3. Huấn luyện lại bằng ba cách ({demo}, hạt giống 11)

    Cả ba bản dùng cùng trọng số khởi tạo, cùng thứ tự mini-batch (batch 64), Adam với tốc độ học 0,001, cắt chuẩn
    gradient ở 1,0, tối đa 60 epoch và dừng sớm sau 8 epoch không cải thiện BCE trên validation. Bảng so sánh ROC-AUC
    vừa huấn luyện với giá trị đã lưu: bản NumPy và PyTorch thường trùng tới nhiều chữ số; Keras có thể lệch rất nhỏ do
    cách đặt hằng số ε của Adam và thứ tự phép tính.
    """), code(TRAIN_THREE), md("""
    ## 4. Kết quả đầy đủ (3 hạt giống) và khoảng tin cậy

    Bảng dưới tổng hợp các bản ghi đã lưu của chương (trung bình và độ lệch chuẩn qua 3 hạt giống), tiếp theo là
    khoảng tin cậy 95% của ROC-AUC cho dự báo trung bình 3 hạt giống.
    """), code(SUMMARY)]
    if representation == 'tabular':
        cells += [code("""
        rows = [{'Bộ dữ liệu': NAMES[name], 'So sánh': key, 'Giá trị': round(item.get('auc', item.get('difference')), 4),
                 'CI 95% thấp': round(item['low'], 4), 'CI 95% cao': round(item['high'], 4)}
                for name in DATASETS for key, item in analysis['uci'][name].items()]
        display(pd.DataFrame(rows))
        """)]
    else:
        cells += [code("""
        rows = [{'Chỉ số': NAMES[name], 'Mô hình': key, 'ROC-AUC': round(item['auc'], 4), 'CI 95% thấp': round(item['low'], 4),
                 'CI 95% cao': round(item['high'], 4), 'p (AUC ≤ 0,5)': round(item['p_value_auc_le_half'], 3)}
                for name in DATASETS for key, item in analysis['auc_ci'][name].items() if key in KINDS]
        display(pd.DataFrame(rows))
        """)]
    if chapter == 4:
        cells += [md("""
        ## 5. Backtest có phí giao dịch

        Nắm giữ chỉ số trong phiên kế tiếp khi xác suất tăng (trung bình 3 hạt giống) không thấp hơn ngưỡng chọn trên
        validation, ngược lại giữ tiền mặt. Cột "trước phí" cho thấy phần lợi thế bị chi phí giao dịch lấy đi.
        """), code("""
        rows = []
        for name in DATASETS:
            for key, b in analysis['backtest'][name].items():
                rows.append({'Chỉ số': NAMES[name], 'Chiến lược': key, 'Lợi suất năm trước phí': round(b['annual_return_gross'], 4),
                             'Lợi suất năm sau phí': round(b['annual_return'], 4), 'Sharpe': round(b['sharpe'], 2),
                             'Sụt giảm tối đa': round(b['max_drawdown'], 4), 'Thời gian nắm giữ': round(b['exposure'], 3),
                             'Số lần mua': b['entries']})
        display(pd.DataFrame(rows))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, name in zip(axes, DATASETS):
            eq = analysis['equity'][name]
            dates = pd.to_datetime(eq['dates'])
            for key, color in (('buy_hold', '#0b0b0b'), ('lstm', '#2a78d6'), ('gru', '#eb6834')):
                ax.plot(dates, eq[key], color=color, lw=1.2, label=key)
            ax.set_title(NAMES[name]); ax.grid(alpha=.3)
        axes[0].set_ylabel('Giá trị danh mục'); axes[0].legend()
        plt.tight_layout(); plt.show()
        """)]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python', 'version': sys.version.split()[0]}
    return nb


SPECS = [
    ('02_ml', 2, 'Các kỹ thuật học máy cơ bản',
     'Hai bài toán rủi ro trên dữ liệu bảng: dự báo phá sản doanh nghiệp và vỡ nợ thẻ tín dụng. Mô hình đại diện là '
     'MLP một lớp ẩn 16 nơ-ron, so sánh với hồi quy logistic và rừng ngẫu nhiên.',
     ['taiwan_bankruptcy', 'credit_default'], 'mlp', ['mlp', 'logistic', 'random_forest', 'majority'], 'tabular',
     'taiwan_bankruptcy'),
    ('03_cnn', 3, 'Mạng nơ-ron tích chập (CNN)',
     'Dự báo hướng đi phiên kế tiếp của S&P 500, VN-Index và Bitcoin từ ảnh GASF 20 × 20 của 20 giá đóng cửa gần nhất. '
     'Mô hình đại diện là CNN4, so sánh thêm CNN8, CNN sâu và hai đường cơ sở.',
     ['sp500', 'vnindex', 'btc'], 'cnn4', ['cnn4', 'cnn8', 'cnndeep', 'persistence', 'majority'], 'image', 'vnindex'),
    ('04_rnn', 4, 'Mạng nơ-ron hồi quy (RNN)',
     'Cùng bài toán với Chương 3 nhưng mạng đọc trực tiếp chuỗi 20 lợi suất logarit. Mô hình đại diện là LSTM, so sánh '
     'thêm SimpleRNN, GRU, hai đường cơ sở và backtest có phí giao dịch.',
     ['sp500', 'vnindex', 'btc'], 'lstm', ['rnn', 'lstm', 'gru', 'persistence', 'majority'], 'sequence', 'vnindex'),
]


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'notebooks/tieuluan'
    output.mkdir(parents=True, exist_ok=True)
    for name, chapter, title, intro, datasets, kind, kinds, representation, demo in SPECS:
        nb = notebook(chapter, title, intro, datasets, kind, kinds, representation, demo)
        if args.execute:
            from nbclient import NotebookClient
            from jupyter_client import KernelManager
            from jupyter_client.kernelspec import KernelSpecManager
            with tempfile.TemporaryDirectory(prefix='tieuluan_kernel_') as temp:
                kernel_dir = Path(temp) / 'tieuluan'
                kernel_dir.mkdir()
                (kernel_dir / 'kernel.json').write_text(json.dumps({
                    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                    'display_name': 'Tiểu luận venv', 'language': 'python'}), encoding='utf-8')
                manager = KernelManager(kernel_name='tieuluan', kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]))
                client = NotebookClient(nb, km=manager, timeout=900, resources={'metadata': {'path': str(ROOT)}})
                try:
                    client.execute()
                finally:
                    if manager.has_kernel:
                        manager.shutdown_kernel(now=True)
        nbf.validate(nb)
        path = output / f'{name}.ipynb'
        nbf.write(nb, path)
        print(f'{path.relative_to(ROOT)}: {len(nb.cells)} ô; đã thực thi = {args.execute}', flush=True)


if __name__ == '__main__':
    main()
