"""Create and execute presentation notebooks; export selected cells for Word."""
import ast
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NBROOT = ROOT/'notebooks/tieuluan/complete'
OUT = ROOT/'results/tieuluan/complete/notebooks'


def md(source):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(label, source):
    return nbf.v4.new_code_cell('# Cell '+label+'\n'+textwrap.dedent(source).strip(),
                              metadata={'report_label': label})


SETUP = '''
import os, sys, json, time, inspect, copy
from pathlib import Path
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p/'src/tieuluan').exists())
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
OUT = ROOT/'results/tieuluan/complete/notebooks'
OUT.mkdir(parents=True, exist_ok=True)
print('Project:', ROOT)
'''


def function_sources():
    source = (ROOT/'scripts/tieuluan/complete_models.py').read_text(encoding='utf8')
    lines = source.splitlines()
    return {node.name: '\n'.join(lines[node.lineno-1:node.end_lineno])
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


TRAIN = '''
from sklearn.metrics import roc_auc_score, accuracy_score
from src.tieuluan.scratch.optim import Adam
import torch, keras
torch.set_num_threads(2)

def logits_of(model, framework, x):
    """Logit B x 1; suy luận không xây graph gradient."""
    if framework == 'scratch':
        model.eval()
        return model.forward(x)
    if framework == 'pytorch':
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()
    return model(x, training=False).numpy()

def binary_loss(z, y):
    """BCE ổn định số từ logit, không sigmoid hai lần."""
    return float(np.mean(np.logaddexp(0, z) - y*z))

def snapshot(model, framework):
    if framework == 'scratch':
        return model.state_dict()
    if framework == 'pytorch':
        return copy.deepcopy(model.state_dict())
    return [w.copy() for w in model.get_weights()]

def restore(model, framework, state):
    if framework in ['scratch', 'pytorch']:
        model.load_state_dict(state)
    else:
        model.set_weights(state)

def run_comparison(kind, data, epochs=3, batch_size=32):
    """Huấn luyện trên train, chọn checkpoint bằng val, đọc test ở cuối."""
    models = {'scratch': scratch_deep(kind),
              'keras': keras_deep(kind),
              'pytorch': torch_deep(kind)}
    for fw in ['keras', 'pytorch']:
        align_weights(models['scratch'], models[fw], fw, kind)
    optimizers = {
        'scratch': Adam(models['scratch'].parameters(), lr=0.001),
        'keras': keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8),
        'pytorch': torch.optim.Adam(
            [p for p in models['pytorch'].parameters() if p.requires_grad],
            lr=0.001, eps=1e-8)}
    reference = logits_of(models['scratch'], 'scratch', data['train_x'][:4])
    rows, curves, predictions = [], {}, {}
    for fw, model in models.items():
        diff = float(np.max(np.abs(
            logits_of(model, fw, data['train_x'][:4])-reference)))
        assert diff < 1e-4
        best_loss, best_state, best_epoch = float('inf'), None, 0
        history = []
        # Cùng thứ tự mini-batch giữa các framework.
        rng = np.random.default_rng(2026)
        start = time.perf_counter()
        for epoch in range(epochs):
            order = rng.permutation(len(data['train_y']))
            losses = []
            for i in range(0, len(order), batch_size):
                ids = order[i:i+batch_size]
                losses.append(train_batch(model, fw,
                    data['train_x'][ids], data['train_y'][ids],
                    optimizers[fw]))
            val_loss = binary_loss(
                logits_of(model, fw, data['val_x']), data['val_y'])
            history.append({'epoch':epoch+1,
                'train_loss':float(np.mean(losses)), 'val_loss':val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = snapshot(model, fw)
                best_epoch = epoch+1
        restore(model, fw, best_state)
        # Test chỉ được đánh giá sau khi chọn checkpoint bằng validation.
        z = logits_of(model, fw, data['test_x']).reshape(-1)
        probability = 0.5*(1+np.tanh(z/2))
        target = data['test_y'].reshape(-1)
        assert np.isfinite(z).all()
        rows.append({'kind':kind, 'framework':fw,
            'trainable_layers':6, 'train_n':len(data['train_y']),
            'val_n':len(data['val_y']), 'test_n':len(target),
            'epochs':epochs, 'best_epoch':best_epoch,
            'val_loss':best_loss, 'test_bce':binary_loss(z,target),
            'test_auc':float(roc_auc_score(target,probability)),
            'test_accuracy':float(accuracy_score(target,probability>=0.5)),
            'initial_max_error':diff,
            'seconds':time.perf_counter()-start})
        curves[fw] = history
        predictions[fw] = probability.tolist()
    return models, rows, curves, predictions
'''


def model_notebook(kind):
    prefix = kind.upper()
    sources = function_sources()
    cells = [md(f'''# {prefix}: sáu lớp học được, Scratch – Keras – PyTorch

    Notebook bổ sung cho báo cáo Chương 3–4. Mỗi nhóm cell trình bày
    **công thức → hàm → code → kết quả → giải thích**.
    Chọn Run All trong môi trường `.venv` của dự án. Notebook dùng dữ liệu
    VN-Index thật đã xử lý và chia theo thời gian, không tải mạng.
    Chỉ dùng 256 mẫu train, 128 validation và 256 test để demo trong thời gian ngắn;
    đây là thực nghiệm trên tập con, không thay thế benchmark toàn bộ dữ liệu.'''),
    code(prefix+'-01',SETUP),
    md('''## Dữ liệu và kiểu đầu vào–đầu ra

    Giữ đúng ranh giới thời gian của bộ dữ liệu đã chuẩn hóa từ train.
    Lấy 256 mẫu cuối của train, 128 mẫu đầu val và 256 mẫu đầu test.
    In ngày quan sát/ngày nhãn và phân bố lớp để người xem kiểm tra.
    `train_x`, `val_x`, `test_x` là tensor; nhãn có shape B × 1.'''),
    code(prefix+'-02',f'''
    KIND = {kind!r}
    representation = 'image' if KIND == 'cnn' else 'sequence'
    with np.load(ROOT/'data/tieuluan/processed/vnindex.npz') as archive:
        data = {{}}
        ranges = {{'train':slice(-256,None), 'val':slice(0,128),
                  'test':slice(0,256)}}
        inventory = []
        for split, selection in ranges.items():
            data[split+'_x'] = archive[split+'_'+representation][selection].copy()
            data[split+'_y'] = archive[split+'_y'][selection].reshape(-1,1).copy()
            dates = archive[split+'_date'][selection]
            targets = archive[split+'_target_date'][selection]
            inventory.append({{'split':split, 'n':len(dates),
                'first':str(dates[0]), 'last_target':str(targets[-1]),
                'positive':int(data[split+'_y'].sum())}})
        assert archive['train_target_date'].max() < archive['val_date'].min()
        assert archive['val_target_date'].max() < archive['test_date'].min()
    display(pd.DataFrame(inventory))
    print('X:', data['train_x'].shape, 'y:', data['train_y'].shape)
    '''),
    md('''## 1. Biểu diễn mạng là hợp các hàm

    CNN: z = D6 ∘ ReLU ∘ D5 ∘ Flatten ∘ ReLU ∘ C4 ∘ ReLU ∘ C3
    ∘ ReLU ∘ C2 ∘ ReLU ∘ C1 (X).

    RNN/LSTM: z = Dense ∘ LastStep ∘ Cell5 ∘ Cell4 ∘ Cell3 ∘ Cell2 ∘ Cell1 (X).

    Đọc từ phải sang trái. CNN có 4 Conv + 2 Dense; RNN/LSTM có 5 lớp hồi quy
    + 1 Dense. ReLU, Flatten, LastStep và số bước thời gian không được đếm thành
    lớp học được. Xác suất chỉ tính bằng sigmoid sau logit.'''),
    code(prefix+'-03',
         'from src.tieuluan.scratch.core import Sequential\n'
         'from src.tieuluan.scratch.layers import Conv2d, Dense, Flatten, ReLU\n'
         'from src.tieuluan.scratch.recurrent import SimpleRNN, LSTM, LastStep\n\n'
         +sources['scratch_deep']),
    md('''### Giải thích `scratch_deep`

    Đầu vào là `kind` và seed; đầu ra là Sequential tự viết bằng NumPy.
    CNN: mỗi Conv giảm cạnh ảnh hai đơn vị, từ 20 xuống 12, nên Dense nhận
    8 × 12 × 12 = 1.152 giá trị. RNN/LSTM: lớp đầu nhận một đặc trưng,
    các lớp sau nhận tám chiều; chỉ lấy bước cuối sau lớp thứ năm.
    Mỗi lớp Scratch có forward/backward riêng, không gọi autograd.'''),
    code(prefix+'-04',sources['keras_deep']),
    md('''### Giải thích `keras_deep`

    Keras dùng Functional API. Permute đổi NCHW sang NHWC cho Conv và đưa về
    NCHW trước Flatten để khớp thứ tự với Scratch. Với hồi quy, bốn lớp đầu
    `return_sequences=True`; lớp cuối trả vector. Dense cuối chưa sigmoid,
    để dùng BCE trực tiếp từ logit.'''),
    code(prefix+'-05',sources['torch_deep']),
    md('''### Giải thích `torch_deep` và `Stacked.forward`

    CNN dùng nn.Sequential; hồi quy dùng ModuleList. Mỗi cell trả về toàn bộ
    chuỗi và trạng thái cuối; forward truyền chuỗi sang cell tiếp theo,
    rồi lấy x[:, -1] đưa qua Linear. `batch_first=True` giữ shape B × T × D.'''),
    code(prefix+'-06',sources['align_weights']),
    md('''### Giải thích `align_weights`

    Cùng seed không bảo đảm cùng trọng số giữa ba thư viện. Hàm sao chép cùng
    trọng số hiệu dụng; đổi trục kernel Conv và chuyển vị Dense cho Keras.
    Với hồi quy, giữ bias thứ hai bằng không để số tham số học khớp một bias
    của Keras. Kiểm tra logit ban đầu trong cell huấn luyện sẽ phát hiện lệch.'''),
    code(prefix+'-07','''
    # Quan sát hợp hàm bằng cách in shape sau từng hàm, trên một mẫu thật.
    trace_model = scratch_deep(KIND)
    value = data['train_x'][:1]
    trace = []
    for index, layer in enumerate(trace_model.layers, 1):
        before = value.shape
        value = layer.forward(value)
        trace.append({'step':index, 'function':type(layer).__name__,
                      'input':str(before), 'output':str(value.shape)})
    display(pd.DataFrame(trace))
    print('Logit:', value.ravel())
    print('Sigmoid:', (0.5*(1+np.tanh(value/2))).ravel())
    '''),
    md('''## 2. Giải thích bên trong hàm forward và backward

    Cell tiếp theo in mã thực của các lớp được dùng, giúp giải thích phép tính
    thay vì chỉ nói “gọi thư viện”. Với Dense: dW=GᵀX, db=sum(G), dX=GW.
    Với RNN/LSTM: backward chạy ngược thời gian và cộng gradient vào cùng trọng số.
    LSTM duy trì gradient ô nhớ c riêng với gradient trạng thái h.
    Đối chiếu từng dòng mã với công thức trong Chương 3/4.'''),
    code(prefix+'-08','''
    primitives = [Conv2d, ReLU, Flatten, Dense] if KIND == 'cnn' else [
        SimpleRNN if KIND == 'rnn' else LSTM, LastStep, Dense]
    for primitive in primitives:
        print(inspect.getsource(primitive))
    '''),
    code(prefix+'-09',sources['train_batch']),
    md('''### Giải thích `train_batch`

    Thứ tự bắt buộc: zero_grad → forward → BCE từ logit → backward → cắt chuẩn
    gradient → Adam.step. Scratch tự tính đạo hàm; PyTorch dùng autograd;
    Keras dùng GradientTape. Xóa gradient bias phụ ở Scratch trước Adam.
    Nếu quên zero_grad, gradient các batch cộng dồn sai thiết kế.
    Nếu sigmoid trước BCEWithLogits, mất mát nhận sai kiểu đầu vào.'''),
    code(prefix+'-10',TRAIN),
    md('''### Giải thích các hàm huấn luyện/đánh giá

    `logits_of` suy luận theo framework; `binary_loss` tính BCE ổn định số;
    `snapshot/restore` lưu và nạp checkpoint; `run_comparison` tạo ba mô hình
    cùng trọng số, học cùng thứ tự mini-batch và chọn checkpoint bằng val.
    Test được đánh giá sau khi chọn checkpoint, không dùng để chọn epoch.
    AUC của tập con ngắn chỉ phục vụ demo, không chứng minh lợi nhuận giao dịch.'''),
    code(prefix+'-11','''
    models, rows, curves, predictions = run_comparison(KIND, data)
    display(pd.DataFrame(rows).round(6))
    payload = {'kind':KIND, 'dataset':'vnindex',
               'scope':'real-data subset demonstration; 3 epochs',
               'inventory':inventory, 'results':rows, 'history':curves,
               'predictions':predictions,
               'target':data['test_y'].reshape(-1).tolist()}
    (OUT/(KIND+'_training.json')).write_text(
        json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    '''),
    code(prefix+'-12','''
    fig, axes = plt.subplots(1,2,figsize=(11,3.6))
    for fw, history in curves.items():
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs,[r['train_loss'] for r in history],label=fw)
        axes[1].plot(epochs,[r['val_loss'] for r in history],label=fw)
    axes[0].set_title(KIND.upper()+' – train BCE')
    axes[1].set_title(KIND.upper()+' – validation BCE')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/(KIND+'_learning.png'),dpi=180)
    plt.show()
    '''),
    md('''## 3. Câu hỏi bảo vệ

    1. Hãy đọc biểu thức hợp hàm từ phải sang trái và chỉ shape tương ứng.
    2. Những lớp nào có tham số? Vì sao 20 thời điểm không thành 20 lớp?
    3. Vì sao phải align_weights trước khi so sánh?
    4. Gradient truyền qua từng hàm thế nào? Tại sao cần clipping?
    5. Loss giảm có bảo đảm AUC test tăng hay có lợi nhuận sau phí không?

    Kết quả trên tập con có thể dao động; phải báo cáo cả ba framework và
    giới hạn mẫu, không chọn riêng kết quả đẹp. Notebook không gửi lệnh giao dịch.''')]
    return cells


def knowledge_notebook():
    return [md('''# Chương 5: Đồ thị tri thức (KG) và RAG, giải thích từng hàm và chạy demo

    Notebook này chạy đúng các hàm mà demo localhost (cổng 8044, trang chủ `/`) dùng. Chạy hoàn toàn trên máy,
    không cần API key. Có bốn đường xử lý: **RAG** (tìm đoạn tài liệu rồi để mô hình viết), **công cụ thống kê**
    (pandas trên dữ liệu CDC BRFSS 2023), **máy tính** và **tra đồ thị tri thức**. Bài học chính: câu hỏi kiến thức
    đi qua RAG, còn câu hỏi cần con số chính xác phải đi qua công cụ, vì mô hình ngôn ngữ nhỏ có thể sai.'''),
    code('KG-01',SETUP),
    md('''## 1. Dữ liệu và mô hình đang dùng

    - **Kho tài liệu tiếng Anh**: 183 đoạn từ bốn trang NIDDK về bệnh tiểu đường (`diabetes_corpus.jsonl`) và 9.177 đoạn
      từ các báo cáo tài chính FinQA (`finance_corpus.jsonl`).
    - **Dữ liệu thống kê mới**: CDC BRFSS 2023 (khảo sát sức khỏe người trưởng thành Hoa Kỳ), mẫu ngẫu nhiên 150.000
      người trả lời hợp lệ, lưu CSV ở `data/chapter05/brfss2023_diabetes_150k.csv`. Dữ liệu cũ (UCI 1999–2008) không còn được dùng.
    - **Bộ sinh câu trả lời**: `Qwen/Qwen2.5-1.5B-Instruct` chạy trên CPU (tải bằng `scripts/chapter05/download_llm.py`
      vào `models/chapter05/llm`). Bộ sinh cũ flan-t5-small chỉ cho ra vài từ nên được thay; lớp `LocalGenerator` cũ vẫn còn trong mã.'''),
    code('KG-02','''
    manifest = json.loads((ROOT/'data/chapter05/brfss2023_manifest.json').read_text(encoding='utf8'))
    print({k: manifest[k] for k in ['source','year','raw_rows','valid_rows','sampled_rows','license']})
    survey = pd.read_csv(ROOT/'data/chapter05/brfss2023_diabetes_150k.csv')
    print('Số dòng:', len(survey), '| Số cột:', survey.shape[1])
    display(survey.head(5))
    display(survey.diabetes_status.value_counts().rename('số người').to_frame())
    llm_name = (ROOT/'models/chapter05/llm/SOURCE.txt').read_text(encoding='utf8').strip()
    print('Mô hình sinh:', llm_name)
    '''),
    md('''## 2. Đồ thị tri thức: chuẩn hóa → tạo nút → tạo cạnh → truy vấn

    `build_document_graph(chunks, domain)` nhận các đoạn đã có metadata và trả về NetworkX DiGraph. Ví dụ dưới dùng một
    dòng giả định để dễ kiểm tra (không phải báo cáo thật). Sau đó nạp đồ thị tài chính thật (công ty → báo cáo → đoạn văn)
    và tra công ty V (Visa) để thấy câu hỏi "công ty này có những báo cáo nào" trả lời được bằng đồ thị mà không cần mô hình.'''),
    code('KG-03','''
    from src.chapter04.knowledge import build_document_graph, safe_calculate, load_graph
    chunks = [{'id':'demo:row1','company':'DEMO','report':'DEMO:2023',
        'year':'2023','filename':'demo-report','kind':'table',
        'text':'Revenue 2022: 100 USD; 2023: 120 USD',
        'source_url':'https://example.org/demo'}]
    graph = build_document_graph(chunks,'finance')
    print('Nút:',list(graph.nodes))
    print('Cạnh:',list(graph.edges(data=True)))
    print('Nguồn cha:',list(graph.predecessors('passage:demo:row1')))
    print(inspect.getsource(build_document_graph))
    fin = load_graph(ROOT/'knowledge_graph/chapter04/finance_graph.json.gz')
    kinds = pd.Series([d['type'] for _, d in fin.nodes(data=True)]).value_counts()
    display(kinds.rename('số nút').to_frame())
    reports = [fin.nodes[n] for n in fin.successors('company:V')]
    print('Công ty V có', len(reports), 'báo cáo:', sorted(r['year'] for r in reports))
    '''),
    md('''## 3. RRF: hợp thứ hạng thay vì cộng điểm khác thang đo

    Truy hồi lai kết hợp BM25 (khớp từ khóa) với vector (khớp nghĩa). `reciprocal_rank_fusion` nhận nhiều danh sách id và
    cộng 1/(60+hạng). Ví dụ hai danh sách dưới giúp tự tính điểm của B. Hàm chỉ trả về id theo điểm giảm dần.'''),
    code('KG-04','''
    from src.chapter04.retrieval import reciprocal_rank_fusion
    rankings = [['A','B','C'],['B','D','A']]
    print('Hạng hợp nhất:',reciprocal_rank_fusion(rankings,k=4))
    print('Điểm B:',1/62+1/61)
    print(inspect.getsource(reciprocal_rank_fusion))
    '''),
    md('''## 4. Một lượt RAG hoàn chỉnh bằng câu hỏi tiếng Việt

    `RAGEngine.answer` làm bốn việc: (1) **dịch** câu hỏi tiếng Việt thành truy vấn tiếng Anh vì kho tài liệu bằng tiếng Anh;
    (2) **truy hồi lai** ba đoạn gần nhất; (3) **mô hình viết** câu trả lời chỉ dựa trên ba đoạn đó (viết tiếng Anh trước vì mô hình
    nhỏ đọc tiếng Anh tốt hơn, rồi dịch sang tiếng Việt bằng bảng thuật ngữ); (4) đóng gói nguồn và trạng thái.
    Kết quả luôn có trạng thái `generated_unverified`: các mã nguồn tồn tại nhưng chưa ai kiểm chứng câu viết có đúng ý nguồn hay không.
    Lần chạy này mất khoảng nửa phút đến vài phút vì mô hình chạy trên CPU; ô sau in thời gian thực đo.'''),
    code('KG-05','''
    from deployment.chapter04.app import rag, agent
    engine = rag()
    t0 = time.perf_counter(); engine.generator.load(); print('Nạp mô hình sinh: %.1f giây' % (time.perf_counter()-t0))
    question = 'Xét nghiệm A1C cho biết điều gì?'
    answer = engine.answer(question,'diabetes')
    print('Câu hỏi        :', question)
    print('Truy vấn tiếng Anh:', answer['query_used'])
    print('Bản tiếng Anh  :', answer['english_answer'])
    print('Câu trả lời    :', answer['answer'])
    print('Trạng thái     :', answer['status'], '| thời gian %.1f giây' % answer['seconds'])
    display(pd.DataFrame([{'id':e['id'],'văn bản':e['text'][:160]} for e in answer['evidence']]))
    '''),
    md('''## 5. Khi tài liệu không có câu trả lời: từ chối thay vì bịa

    Có hai lớp bảo vệ. Lớp một là `lexical_support`: nếu không từ khóa nội dung nào của truy vấn xuất hiện trong ba đoạn tìm được
    thì dừng ngay, không gọi mô hình. Lớp hai là chính mô hình, được dặn trả lời đúng một từ `INSUFFICIENT` khi các đoạn không chứa đáp án.
    Hai lớp này không hoàn hảo, nhưng làm giảm nguy cơ mô hình bịa.'''),
    code('KG-06','''
    from src.chapter04.rag import lexical_support, content_tokens
    refused = engine.answer('Giá vàng hôm nay là bao nhiêu?','diabetes')
    print('Truy vấn:', refused['query_used'], '| từ khóa nội dung:', content_tokens(refused['query_used']))
    print('Trạng thái:', refused['status'], '|', refused['answer'])
    assert refused['status'] == 'insufficient_evidence'
    '''),
    md('''## 6. Công cụ thống kê: hiểu nhiều dạng câu hỏi tự nhiên

    `DiabetesStats` nhận ý định bằng luật và từ khóa (đã bỏ dấu để khớp cả câu không dấu): đếm, tỉ lệ, so sánh hai nhóm,
    phân theo nhóm, BMI trung bình, kèm bộ lọc như nhóm tuổi, giới, BMI, huyết áp cao, hút thuốc, bang. Mọi con số tính bằng pandas
    nên chính xác theo dữ liệu. Câu không hiểu được trả lời lịch sự kèm danh sách câu mẫu, không báo lỗi.'''),
    code('KG-07','''
    from src.chapter04.stats import DiabetesStats
    stats = DiabetesStats(ROOT/'data/chapter05/brfss2023_diabetes_150k.csv')
    questions = ['Có bao nhiêu người trong dữ liệu khảo sát?',
                 'Tỉ lệ người mắc tiểu đường là bao nhiêu?',
                 'So sánh tỉ lệ tiểu đường giữa nam và nữ',
                 'BMI trung bình của người bị tiểu đường là bao nhiêu?',
                 'Tỉ lệ tiểu đường ở người huyết áp cao',
                 'What percentage of people over 65 have diabetes?',
                 'Diabetes rate in Texas']
    rows = []
    for q in questions:
        r = stats.answer(q); rows.append({'câu hỏi': q, 'câu trả lời': r['answer']})
    display(pd.DataFrame(rows))
    table = stats.answer('Tỉ lệ tiểu đường theo nhóm tuổi')['table']
    display(pd.DataFrame(table['rows'], columns=table['columns']))
    print(stats.answer('Hôm nay trời đẹp'))
    # Kiểm chứng độc lập bằng pandas thuần:
    assert stats.answer('Tỉ lệ người mắc tiểu đường là bao nhiêu?')['value'] == (survey.diabetes_status == 'yes').mean()*100
    '''),
    md('''## 7. Bộ định tuyến công cụ (ToolAgent): chọn đúng đường cho từng câu

    `ToolAgent.route` đọc câu hỏi và chọn: có phép tính hai số thì dùng máy tính; có tên công ty kèm từ "báo cáo" thì tra đồ thị;
    có từ khóa thống kê thì dùng công cụ thống kê; còn lại là RAG. Đây là bộ định tuyến luật, không phải agent LLM tự lập kế hoạch.
    Ô dưới chạy các đường không cần mô hình sinh để thấy câu trả lời bằng văn xuôi cùng "cách agent xử lý".'''),
    code('KG-08','''
    worker = agent()
    rows = []
    for q, dom in [('Tính tăng trưởng doanh thu từ 100 lên 130','finance'),
                   ('250 chia 5','finance'),
                   ('Công ty V có những báo cáo nào?','finance'),
                   ('So sánh tỉ lệ tiểu đường giữa nam và nữ','diabetes')]:
        r = worker.ask(q, dom)
        rows.append({'câu hỏi': q, 'đường xử lý': r['route'], 'câu trả lời': r['answer'], 'nhãn tin cậy': r['confidence']['kind']})
    display(pd.DataFrame(rows))
    print(worker.ask('So sánh tỉ lệ tiểu đường giữa nam và nữ','diabetes')['steps'])
    print(safe_calculate('growth',[100,120],['USD','USD']))
    for operation, values, units in [('divide',[1,0],['USD','USD']), ('growth',[100,120],['USD','VND'])]:
        try:
            safe_calculate(operation,values,units)
        except ValueError as exc:
            print('Từ chối đúng:',exc)
        else:
            raise AssertionError('Đầu vào sai không bị chặn')
    '''),
    md('''## 8. Đọc kết quả thực nghiệm đã lưu

    Các bảng này đọc từ JSON của thí nghiệm truy hồi trên FinQA ở bài Chương 4, không chạy lại benchmark. Chúng được đo với bộ sinh cũ
    (flan-t5-small), nên cột sinh văn bản không áp dụng cho Qwen. So sánh global với oracle_report phải ghi đúng điều kiện.
    Câu hỏi để giải thích: Hit@5, Recall@5, Complete@5 khác nhau thế nào?'''),
    code('KG-09','''
    result = json.loads((ROOT/'results/chapter04/finance_rag_results.json')
                        .read_text(encoding='utf8'))
    display(pd.DataFrame(result['retrieval'])[
        ['method','scope','hit_at_5','recall_at_5','complete_at_5']])
    (OUT/'kg_demo.json').write_text(json.dumps({
        'question':question,'answer':answer['answer'],'english_answer':answer['english_answer'],
        'query_used':answer['query_used'],'status':answer['status'],'model':answer['model'],
        'evidence_ids':[e['id'] for e in answer['evidence']],
        'survey_rows':len(survey)},
        ensure_ascii=False,indent=2),encoding='utf8')
    '''),
    md('''## Giải thích từng hàm khi bảo vệ

    `build_document_graph`: tạo schema và cạnh có nguồn. `RAGEngine.retrieve`: chọn đoạn (BM25 + vector + RRF).
    `RAGEngine.to_english_query`: dịch truy vấn. `ChatGenerator.generate_full`: viết tiếng Anh từ ba đoạn rồi dịch.
    `lexical_support`: chặn trường hợp không có từ khóa nào khớp. `DiabetesStats.answer`: nhận ý định rồi tính bằng pandas.
    `safe_calculate`: tính số theo toán hạng cấu trúc, không dùng eval. `ToolAgent.route`: chọn công cụ.
    RAG không bảo đảm hết ảo giác; chỉ số liệu do công cụ tính mới đáng tin tuyệt đối.''')]


def agents_notebook():
    return [md('''# Chương 7: Hệ đa agent: State, Message, hợp hàm và kiểm tra lỗi

    Hệ điều phối xác định với bốn vai trò (truy hồi, đồ thị, viết, kiểm tra); chưa phải nhóm LLM tự lập kế hoạch.
    Chạy cục bộ trên cùng kho tài liệu và mô hình Qwen2.5-1.5B-Instruct của Chương 5. Demo localhost của chương này ở `/agents`
    (cổng 8044). Mỗi ô đi kèm điều cần giải thích.'''),
    code('AGENT-01',SETUP),
    md('''## 1. Mã thực của State, Message và các agent

    State chứa câu hỏi, truy vấn tiếng Anh, evidence, graph_context, answer, status và messages. Message ghi sender, receiver và payload.
    Bốn phép chuyển trạng thái hợp thành S4 = Verifier ∘ Writer ∘ Graph ∘ Retrieval (S0).
    Ô tiếp theo định nghĩa đầy đủ các lớp, không chỉ import tên. Vai trò đời thường: Agent truy hồi là người đi tìm tài liệu,
    Agent đồ thị là người tra nguồn của từng đoạn, Agent viết là người soạn câu trả lời, Agent kiểm tra là người soát nguồn trích dẫn.'''),
    code('AGENT-02',(ROOT/'scripts/tieuluan/complete_agents.py').read_text(encoding='utf8')),
    md('''## 2. Khởi tạo phụ thuộc

    `rag()` nạp kho tài liệu, hai bộ truy hồi và mô hình sinh; `load_graph` nạp hai đồ thị (tài chính và tài liệu NIDDK);
    MultiAgentTeam gắn từng phụ thuộc cho vai trò cần dùng. Chỉ dùng một mô hình sinh chung, không gọi bốn mô hình độc lập.'''),
    code('AGENT-03','''
    from deployment.chapter04.app import rag, agent, summarize
    from src.chapter04.knowledge import load_graph
    graphs = {'finance':load_graph(ROOT/'knowledge_graph/chapter04/finance_graph.json.gz'),
              'diabetes':load_graph(ROOT/'knowledge_graph/chapter04/diabetes_documents_graph.json.gz')}
    engine = rag(); engine.generator.load()
    team = MultiAgentTeam(engine,graphs)
    print([type(a).__name__ for a in team.agents])
    '''),
    md('''## 3. Quan sát hợp hàm bằng cách chạy riêng từng agent

    Sau mỗi `act`, in câu kể lại việc agent vừa làm (hàm `summarize` của demo), số bằng chứng, ngữ cảnh đồ thị, trạng thái và thông điệp vừa tạo.
    Agent truy hồi không viết đáp án; Agent đồ thị bổ sung nguồn cha; Agent viết sinh câu trả lời; Agent kiểm tra chỉ kiểm tra mã nguồn,
    chưa xác minh ngữ nghĩa nên trạng thái cuối là `generated_unverified`.'''),
    code('AGENT-04','''
    state = State('Xét nghiệm A1C cho biết điều gì?','diabetes')
    trace = []
    for worker in team.agents:
        t0 = time.perf_counter(); worker.act(state); seconds = time.perf_counter()-t0
        story, outcome = summarize(type(worker).__name__, state, worker, seconds)
        trace.append({'agent':type(worker).__name__,'kết quả':outcome,'evidence':len(state.evidence),
                      'graph':len(state.graph_context),'status':state.status,'giây':round(seconds,1)})
        message = state.messages[-1]
        print(type(worker).__name__, '=>', story)
        print('   Thông điệp:', message.sender,'->',message.receiver,message.payload)
    display(pd.DataFrame(trace))
    print('Câu trả lời cuối:', state.answer)
    assert len(state.messages)==4
    assert not state.messages[-1].payload['semantic_support_checked']
    '''),
    md('''## 4. Kiểm tra các đường lỗi

    `ask` chặn domain sai và câu hỏi rỗng. Khi report không tồn tại, truy hồi trả rỗng, Agent viết không sinh và Agent kiểm tra
    trả `insufficient_evidence`. Điểm dừng: mỗi agent chạy một lần, không có vòng lặp đến khi đồng ý.'''),
    code('AGENT-05','''
    missing = team.ask('Find revenue','finance',report='__missing_report__')
    print('Không có nguồn:',missing.status,missing.answer)
    assert missing.status=='insufficient_evidence'
    for question, domain in [('', 'finance'), ('test','invalid')]:
        try:
            team.ask(question,domain)
        except ValueError as exc:
            print('Đầu vào bị chặn:',exc)
        else:
            raise AssertionError('Đầu vào sai không bị chặn')
    '''),
    md('''## 5. Câu hỏi không có trong tài liệu: từ chối "không đủ bằng chứng"

    Với câu hỏi ngoài kho tài liệu, nhóm agent phải từ chối thay vì bịa. Agent truy hồi vẫn trả về ba đoạn gần nhất, nhưng không đoạn nào chứa
    từ khóa của câu hỏi; Agent viết nhận ra điều đó và Agent kiểm tra giữ nguyên lời từ chối.'''),
    code('AGENT-06','''
    gold = team.ask('Giá vàng hôm nay là bao nhiêu?','diabetes')
    print('Trạng thái:', gold.status, '| truy vấn:', gold.query)
    print('Câu trả lời:', gold.answer)
    assert gold.status=='insufficient_evidence'
    '''),
    md('''## 6. Vì sao nguồn tồn tại chưa chứng minh câu trả lời đúng?

    Chạy một câu hỏi đếm qua nhóm agent rồi đối chiếu công cụ thống kê của ToolAgent. Chỉ công cụ thống kê đọc số từ dữ liệu thật
    (CDC BRFSS 2023, 150.000 dòng). Câu mô hình viết có thể lấy một con số trong tài liệu (ví dụ số người mắc bệnh ở Hoa Kỳ) hoàn toàn
    không liên quan đến dữ liệu khảo sát. Mã nguồn trích dẫn vẫn hợp lệ, vì vậy Agent kiểm tra không bắt được lỗi này: đó là giới hạn của kiểm tra cấu trúc.'''),
    code('AGENT-07','''
    question = 'Có bao nhiêu người trong dữ liệu khảo sát?'
    generated = team.ask(question,'diabetes')
    computed = agent().ask(question,'diabetes')
    print('Nhóm agent viết:', generated.answer, '|', generated.status)
    print('Công cụ thống kê:', computed['answer'], '|', computed['status'])
    print('Nguồn id hợp lệ:', generated.messages[-1].payload['citation_ids_valid'])
    payload = {'trace':trace,
        'messages':[vars(m) for m in state.messages],
        'generated_count':generated.answer,
        'computed_count':computed['answer'],
        'missing_status':missing.status,'gold_status':gold.status}
    (OUT/'agents_demo.json').write_text(
        json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    '''),
    md('''## Câu hỏi trình bày

    Hãy giải thích riêng State.send, RetrievalAgent.act, GraphAgent.act, WriterAgent.act, VerifierAgent.act và MultiAgentTeam.ask.
    Chỉ ra trường State nào bị đổi ở từng bước, Message gửi cho ai và điều kiện dừng/lỗi. Tại sao không được đổi generated_unverified
    thành verified chỉ vì id hợp lệ?''')]


def main():
    """Mặc định chỉ dựng ba notebook Chương 2–4 (xem chapter_notebooks.py).

    Các notebook 05 (KG/RAG) và 07 (multi-agent) có hàm dựng ở trên nhưng do phần khác phụ trách;
    chỉ dựng lại khi truyền tường minh --legacy-extra, vì sẽ ghi đè tệp hiện có.
    """
    import argparse
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*', help='tên notebook cần dựng, ví dụ 03_cnn 04_rnn_lstm')
    parser.add_argument('--no-run', action='store_true', help='chỉ dựng, không chạy')
    parser.add_argument('--legacy-extra', action='store_true', help='dựng lại 05_knowledge_rag và 07_multi_agent')
    args = parser.parse_args()
    from scripts.tieuluan import chapter_notebooks
    chapter_notebooks.build(args.only, run=not args.no_run)
    if not args.legacy_extra:
        return
    NBROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [('05_knowledge_rag', knowledge_notebook()), ('07_multi_agent', agents_notebook())]
    manifest = []
    for name, cells in specs:
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': sys.version.split()[0]}})
        path = NBROOT / (name + '.ipynb')
        for cell in cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
        nbf.write(notebook, path)
        print('Executing', name, flush=True)
        start = time.perf_counter()
        client = NotebookClient(notebook, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
        client.execute()
        nbf.validate(notebook)
        nbf.write(notebook, path)
        manifest.append({'notebook': str(path.relative_to(ROOT)), 'seconds': time.perf_counter() - start})
    (OUT / 'execution.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
