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
    return [md('''# Chương 5: KG – RAG, giải thích từng hàm và chạy demo

    Corpus, embedding và generator đã có trong dự án. Run All chạy cục bộ,
    không cần API key. Phân biệt truy hồi bằng chứng với tính đúng của đáp án.'''),
    code('KG-01',SETUP),
    md('''## 1. Đồ thị là hợp các bước chuẩn hóa → tạo nút → tạo cạnh → truy vấn

    `build_document_graph(chunks, domain)` nhận các đoạn đã có metadata và trả
    NetworkX DiGraph. Ví dụ dưới dùng hai dòng giả định để dễ kiểm tra; không
    coi các con số minh họa là báo cáo kinh doanh thật.'''),
    code('KG-02','''
    from src.chapter04.knowledge import build_document_graph, safe_calculate
    chunks = [{'id':'demo:row1','company':'DEMO','report':'DEMO:2023',
        'year':'2023','filename':'demo-report','kind':'table',
        'text':'Revenue 2022: 100 USD; 2023: 120 USD',
        'source_url':'https://example.org/demo'}]
    graph = build_document_graph(chunks,'finance')
    print('Nút:',list(graph.nodes))
    print('Cạnh:',list(graph.edges(data=True)))
    print('Nguồn cha:',list(graph.predecessors('passage:demo:row1')))
    print(inspect.getsource(build_document_graph))
    '''),
    md('''## 2. RRF: hợp thứ hạng thay vì cộng điểm khác thang đo

    `reciprocal_rank_fusion` nhận nhiều danh sách id và cộng 1/(60+rank).
    Ví dụ hai danh sách dưới giúp tự tính điểm của B. Hàm trả về id theo điểm
    giảm dần; bản thân RRF chưa tạo văn bản trả lời.'''),
    code('KG-03','''
    from src.chapter04.retrieval import reciprocal_rank_fusion
    rankings = [['A','B','C'],['B','D','A']]
    print('Hạng hợp nhất:',reciprocal_rank_fusion(rankings,k=4))
    print('Điểm B:',1/62+1/61)
    print(inspect.getsource(reciprocal_rank_fusion))
    '''),
    md('''## 3. Hợp hàm RAG: retrieve → build_prompt → generate → đóng gói nguồn

    Chạy truy hồi thật trước khi sinh. In id, locator và một phần văn bản để
    kiểm tra. `build_prompt` đặt câu hỏi trước ngữ cảnh. Kết quả được sinh giữ
    trạng thái generated_unverified: nguồn tồn tại chưa chứng minh đáp án đúng.'''),
    code('KG-04','''
    from deployment.chapter04.app import rag
    from src.chapter04.rag import build_prompt
    engine = rag()
    question = 'What are the symptoms of diabetes?'
    evidence = engine.retrieve(question,'diabetes',k=3)
    display(pd.DataFrame([{'id':e['id'],'text':e['text'][:180]}
                          for e in evidence]))
    print(build_prompt(question,evidence)[:1400])
    answer = engine.answer(question,'diabetes')
    print('Answer:',answer['answer'])
    print('Status:',answer['status'])
    print('Sources:',answer['citations'])
    '''),
    md('''## 4. Tách phép tính ra công cụ

    `safe_calculate` nhận operation, hai values và units. Ví dụ growth từ
    100 lên 120 bằng 20%. Không dùng eval; không trích toán hạng tự động.
    Đơn vị không khớp hoặc chia cho không phải bị từ chối.'''),
    code('KG-05','''
    print(safe_calculate('growth',[100,120],['USD','USD']))
    for operation, values, units in [
        ('divide',[1,0],['USD','USD']),
        ('growth',[100,120],['USD','VND'])]:
        try:
            safe_calculate(operation,values,units)
        except ValueError as exc:
            print('Từ chối đúng:',exc)
        else:
            raise AssertionError('Đầu vào sai không bị chặn')
    '''),
    md('''## 5. Đọc kết quả thực nghiệm đã lưu

    Các bảng này đọc từ JSON thí nghiệm gốc, không chạy lại benchmark trong
    notebook demo. So sánh global với oracle_report phải ghi đúng điều kiện.
    Explain: Hit@5, Recall@5, Complete@5 khác nhau thế nào?'''),
    code('KG-06','''
    result = json.loads((ROOT/'results/chapter04/finance_rag_results.json')
                        .read_text(encoding='utf8'))
    display(pd.DataFrame(result['retrieval'])[
        ['method','scope','hit_at_5','recall_at_5','complete_at_5']])
    display(pd.DataFrame(result['generation']))
    (OUT/'kg_demo.json').write_text(json.dumps({
        'question':question,'answer':answer['answer'],
        'status':answer['status'],
        'evidence_ids':[e['id'] for e in evidence]},
        ensure_ascii=False,indent=2),encoding='utf8')
    '''),
    md('''## Giải thích từng hàm khi bảo vệ

    build_document_graph: tạo schema và cạnh có nguồn; retrieve: chọn đoạn;
    reciprocal_rank_fusion: hợp hạng; build_prompt: ghép câu hỏi và ngữ cảnh;
    generate: sinh văn bản; answer: đóng gói nguồn và trạng thái;
    safe_calculate: tính số theo toán hạng cấu trúc. RAG không bảo đảm hết ảo giác.''')]


def agents_notebook():
    return [md('''# Chương 6: Hệ đa Agent – State, Message, hợp hàm và kiểm tra lỗi

    Hệ điều phối xác định với bốn vai trò; chưa phải nhóm LLM tự lập kế hoạch.
    Chạy trên corpus và FLAN-T5-small cục bộ. Mỗi cell đi kèm điều cần giải thích.'''),
    code('AGENT-01',SETUP),
    md('''## 1. Mã thực của State, Message và các tác tử

    State chứa câu hỏi, evidence, graph_context, answer, status và messages.
    Message ghi sender, receiver và payload. Bốn phép chuyển trạng thái hợp thành
    S4 = Verifier ∘ Writer ∘ Graph ∘ Retrieval (S0).
    Cell tiếp theo định nghĩa đầy đủ các lớp, không chỉ import tên.'''),
    code('AGENT-02',(ROOT/'scripts/tieuluan/complete_agents.py').read_text(encoding='utf8')),
    md('''## 2. Khởi tạo phụ thuộc

    `rag()` nạp corpus, hai bộ truy hồi và generator; `load_graph` nạp đồ thị
    JSON nén; MultiAgentTeam gắn từng phụ thuộc cho vai trò cần dùng.
    Chỉ dùng một generator chung, không gọi bốn mô hình độc lập.'''),
    code('AGENT-03','''
    from deployment.chapter04.app import rag, agent
    from src.chapter04.knowledge import load_graph
    graphs = {'finance':load_graph(
        ROOT/'knowledge_graph/chapter04/finance_graph.json.gz')}
    team = MultiAgentTeam(rag(),graphs)
    print([type(a).__name__ for a in team.agents])
    '''),
    md('''## 3. Quan sát hợp hàm bằng cách chạy riêng từng tác tử

    Sau mỗi act, in số bằng chứng, ngữ cảnh đồ thị, câu trả lời, status và
    thông điệp vừa tạo. Retrieval không viết đáp án; Graph bổ sung nguồn cha;
    Writer sinh; Verifier chỉ kiểm tra cấu trúc nguồn, chưa xác minh ngữ nghĩa.'''),
    code('AGENT-04','''
    state = State('What is the percentage change in revenue?','finance')
    trace = []
    for worker in team.agents:
        worker.act(state)
        trace.append({'agent':type(worker).__name__,
            'evidence':len(state.evidence),'graph':len(state.graph_context),
            'answer':state.answer,'status':state.status})
        message = state.messages[-1]
        print(message.sender,'->',message.receiver,message.payload)
    display(pd.DataFrame(trace))
    assert len(state.messages)==4
    assert not state.messages[-1].payload['semantic_support_checked']
    '''),
    md('''## 4. Kiểm tra các đường lỗi

    `ask` chặn domain sai và câu hỏi rỗng. Khi report không tồn tại, truy hồi
    trả rỗng, Writer không sinh và Verifier trả insufficient_evidence.
    Điểm dừng: mỗi tác tử chạy một lần, không vòng lặp đến khi đồng ý.'''),
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
    md('''## 5. Vì sao nguồn tồn tại chưa chứng minh câu trả lời đúng?

    Chạy một câu hỏi đếm qua bộ sinh rồi đối chiếu công cụ statistics của
    ToolAgent. Chỉ statistics đọc số từ bảng thật; không dùng câu trả lời
    sai của generator như dữ kiện. Đây là phép thử giới hạn Verifier.'''),
    code('AGENT-06','''
    question = 'How many encounters are in the dataset?'
    generated = team.ask(question,'diabetes')
    computed = agent().ask(question,'diabetes',task='statistics')
    print('Sinh:',generated.answer,generated.status)
    print('Thống kê:',computed['answer'],computed['status'])
    print('Nguồn id hợp lệ:',
          generated.messages[-1].payload['citation_ids_valid'])
    payload = {'finance_trace':trace,
        'messages':[vars(m) for m in state.messages],
        'generated_count':generated.answer,
        'computed_count':computed['answer'],
        'missing_status':missing.status}
    (OUT/'agents_demo.json').write_text(
        json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    '''),
    md('''## Câu hỏi trình bày

    Hãy giải thích riêng State.send, RetrievalAgent.act, GraphAgent.act,
    WriterAgent.act, VerifierAgent.act và MultiAgentTeam.ask. Chỉ ra trường
    State nào bị đổi ở từng bước, Message gửi cho ai và điều kiện dừng/lỗi.
    Tại sao không được đổi generated_unverified thành verified chỉ vì id hợp lệ?''')]


def main():
    sys.stdout.reconfigure(encoding='utf8')
    NBROOT.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    specs = [('03_cnn_6_layers',model_notebook('cnn')),
             ('04a_rnn_6_layers',model_notebook('rnn')),
             ('04b_lstm_6_layers',model_notebook('lstm')),
             ('05_knowledge_rag',knowledge_notebook()),
             ('06_multi_agent',agents_notebook())]
    manifest = []
    for name, cells in specs:
        notebook = nbf.v4.new_notebook(cells=cells,metadata={
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python','version':sys.version.split()[0]}})
        path = NBROOT/(name+'.ipynb')
        # Catch malformed generated Python before starting a kernel.
        for cell in cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)
        nbf.write(notebook,path)
        print('Executing',name,flush=True)
        start = time.perf_counter()
        client = NotebookClient(notebook,timeout=900,kernel_name='python3',
                                resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        nbf.validate(notebook)
        for c in notebook.cells:
            if c.cell_type=='code':
                assert c.execution_count is not None
                assert not any(o.output_type=='error' for o in c.outputs)
        nbf.write(notebook,path)
        row = {'notebook':str(path.relative_to(ROOT)),
            'code_cells':sum(c.cell_type=='code' for c in cells),
            'executed':True,'seconds':time.perf_counter()-start}
        manifest.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'execution.json').write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('All five notebooks executed with no error outputs',flush=True)


if __name__=='__main__':
    main()
