"""Embed actual executed notebook cells and their explanations in the report."""
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT/'notebooks/tieuluan/complete'
RESULTS = ROOT/'results/tieuluan/complete/notebooks'


def extract(filename, label, explanation):
    notebook = nbformat.read(NB/(filename+'.ipynb'),as_version=4)
    cell = next(c for c in notebook.cells
                if c.get('metadata',{}).get('report_label')==label)
    assert cell.execution_count is not None, (filename,label,'not executed')
    assert not any(o.output_type=='error' for o in cell.outputs)
    output = []
    for item in cell.outputs:
        if item.output_type=='stream':
            output.append(item.text)
        elif item.output_type in ('display_data','execute_result'):
            value = item.get('data',{}).get('text/plain','')
            if value:
                output.append(value)
    content = '\n'.join(output).strip()
    # Definition-only cells legitimately produce no output.
    result = f'**Cell {label} – notebook `{filename}.ipynb`.**\n\n'
    source=cell.source
    if not source.startswith('# Cell '+label+'\n'):
        source='# Cell '+label+'\n'+source
    result += '```python\n'+source+'\n```\n\n'
    if content:
        lines = content.splitlines()
        compact = '\n'.join(lines[:24])
        if len(lines)>24:
            compact += '\n[Phần còn lại xem trực tiếp trong notebook.]'
        result += '**Đầu ra đã chạy (trích phần văn bản):**\n\n```text\n'+compact+'\n```\n\n'
    result += '**Giải thích hàm và liên hệ công thức.** '+explanation+'\n\n'
    return result


def model_section(kind):
    if kind=='cnn':
        filename='03_cnn_6_layers'
        title='### 3.10.4. Cell notebook: dựng mạng và quan sát hợp hàm'
        first='CNN-03'
        description=('scratch_deep nhận tên kiến trúc và seed, trả về Sequential. '
            'Bốn Conv và hai Dense là sáu lớp học được. ReLU tạo phi tuyến, Flatten '
            'đổi shape và không thêm tham số. Với ảnh 20 × 20, bốn Conv valid tạo '
            '8 × 12 × 12 giá trị trước Dense. Mọi forward/backward của các lớp '
            'được tự cài bằng NumPy, không sử dụng autograd.')
        trace='CNN-07'
    elif kind=='rnn':
        filename='04a_rnn_6_layers'
        title='### 4.11.4. Cell notebook: RNN xếp chồng và forward từng lớp'
        first='RNN-05'
        description=('torch_deep tạo ModuleList gồm năm RNN và một Linear. '
            'Stacked.forward đưa toàn bộ chuỗi của lớp dưới sang lớp trên, '
            'sau đó lấy x[:, -1] cho đầu ra một logit. Trong mỗi lớp, h_t phụ '
            'thuộc đầu vào tại t và trạng thái t−1; trọng số dùng chung theo '
            'thời gian nhưng khác giữa năm lớp. Cell tương ứng bản Scratch và '
            'Keras nằm ngay trong cùng notebook, không chỉ có bản PyTorch.')
        trace='RNN-07'
    else:
        filename='04b_lstm_6_layers'
        title='### 4.11.5. Cell notebook: LSTM, return_sequences và trạng thái cuối'
        first='LSTM-04'
        description=('keras_deep dựng năm LSTM và Dense. Bốn lớp đầu trả cả '
            'chuỗi, lớp thứ năm trả trạng thái cuối. Mỗi LSTM cập nhật các cổng '
            'i, f, g, o và ô nhớ c theo công thức mục 4.4; output truyền lên lớp '
            'trên là h, không phải ô nhớ c. Nhánh CNN trong hàm dùng Permute '
            'để khớp quy ước kênh; trong demo KIND=lstm nên chỉ nhánh hồi quy chạy. '
            'Các cell Scratch và PyTorch có trong cùng notebook để đối chiếu.')
        trace='LSTM-07'
    section = title+'\n\n'
    section += f'Notebook `notebooks/tieuluan/complete/{filename}.ipynb` '
    section += ('đã chạy Run All. Các cell dưới được chèn trực tiếp từ notebook, '
        'giữ nhãn cell để tra lại. Mỗi cell code đi cùng giải thích đầu vào, '
        'đầu ra và ý nghĩa của hàm; không chỉ đưa ảnh chụp code.\n\n')
    section += extract(filename,first,description)
    section += extract(filename,trace,
        'Vòng lặp áp dụng từng hàm forward vào kết quả của hàm trước, đúng '
        'với biểu thức hợp hàm đã trình bày. Cột input/output cho phép kiểm tra '
        'miền xác định và miền giá trị ở từng bước. Sigmoid chỉ áp dụng sau logit '
        'để hiển thị xác suất; nó không được chèn trước BCEWithLogits khi học.')
    if kind=='cnn':
        section += extract(filename,'CNN-09',
            'train_batch là hợp các bước zero_grad → forward → loss → backward '
            '→ clip → cập nhật. Nhánh Scratch tự lan truyền ngược, PyTorch '
            'dùng loss.backward và Keras dùng GradientTape. Hàm nhận model, '
            'framework, tensor x, nhãn y và optimizer, trả về một số loss. '
            'BCE nhận logit trực tiếp; cắt chuẩn gradient trước Adam. Hàm '
            'dùng chung cho CNN, RNN và LSTM; bias hồi quy phụ được giữ '
            'không học để khớp một bias hiệu dụng giữa ba công cụ. Đây '
            'là cell định nghĩa nên chưa có output riêng; các cell -11 '
            'gọi nó nhiều lần trong vòng huấn luyện và lưu loss thực tế.')
    section += '**Cell huấn luyện và kết quả trên dữ liệu thật.** '
    section += ('Các cell '+kind.upper()+'-09 đến '+kind.upper()+'-12 chứa '
        'train_batch, các hàm lưu checkpoint, vòng huấn luyện ba công cụ và '
        'đồ thị loss. Thực nghiệm demo dùng 256 mẫu train, 128 validation và '
        '256 test của VN-Index, giữ cách chia thời gian của dữ liệu gốc; '
        'ba epoch, batch 32, cùng trọng số hiệu dụng và cùng thứ tự batch. '
        'Checkpoint được chọn bằng BCE validation; test chỉ được đánh giá '
        'sau khi chọn checkpoint. Đây là tập con phục vụ trình bày, không '
        'thay thế kết quả toàn bộ dữ liệu trong chương.\n\n')
    payload=json.loads((RESULTS/(kind+'_training.json')).read_text(encoding='utf8'))
    for r in payload['results']:
        section += ('- '+r['framework']+': checkpoint epoch '+str(r['best_epoch'])+
            '; validation BCE = '+f"{r['val_loss']:.6f}"+
            '; test BCE = '+f"{r['test_bce']:.6f}"+
            '; test AUC = '+f"{r['test_auc']:.6f}"+'.\n')
    section += ('\nCác chỉ số trên đọc trực tiếp từ kết quả notebook. Không dùng '
        'AUC của tập con để kết luận có lợi nhuận, và không gán các số này '
        'cho mô hình nông đã đánh giá trên toàn bộ tập test.\n\n')
    return section


def knowledge_section():
    result='### 5.6.1. Cell notebook: truy vết KG và công cụ tính toán\n\n'
    result+=('Notebook `notebooks/tieuluan/complete/05_knowledge_rag.ipynb` '
        'chạy các bước KG, RRF, truy hồi và sinh thật. Hai cell sau minh họa '
        'hàm có đầu vào–đầu ra dễ kiểm chứng; dữ liệu công ty DEMO là số giả '
        'định để tính tay, không phải số liệu kinh doanh.\n\n')
    result+=extract('05_knowledge_rag','KG-02',
        'build_document_graph nhận chunks và domain, tạo nút Company, Report, '
        'TableRow cùng các cạnh PUBLISHED và HAS_PASSAGE. predecessors trả '
        'nguồn cha của đoạn; đây là truy vết tài liệu, không phải suy luận '
        'doanh thu hay dự báo giá. inspect.getsource trong cell giúp xem '
        'mã thực của hàm đã gọi; phần output in mã dài được xem đầy đủ trong notebook.')
    result+=extract('05_knowledge_rag','KG-05',
        'safe_calculate tính growth theo (b−a)/a × 100, nên 100 lên 120 cho 20%. '
        'Hai nhánh try/except minh họa kiểm tra mẫu số bằng không và đơn vị '
        'không khớp. Hàm không tự trích toán hạng từ báo cáo; trả kết quả số đúng '
        'khi được cung cấp đúng hai giá trị có cấu trúc.')
    return result


def agent_section():
    result='### 6.4.1. Cell notebook: chạy riêng từng Agent và quan sát thông điệp\n\n'
    result+=('Notebook `notebooks/tieuluan/complete/06_multi_agent.ipynb` '
        'định nghĩa đầy đủ State, Message và bốn tác tử ở cell AGENT-02. '
        'Các cell dưới không chỉ in kết quả cuối mà đi qua từng phép chuyển '
        'trạng thái, phù hợp yêu cầu giải thích hợp các hàm.\n\n')
    result+=extract('06_multi_agent','AGENT-04',
        'State tạo trạng thái ban đầu; worker.act là một hàm chuyển trạng '
        'thái; mỗi lần chạy thêm một Message. Bảng trace ghi evidence, '
        'graph_context, answer và status sau từng hàm. assert cuối xác nhận '
        'bốn thông điệp và ghi rõ chưa kiểm tra ngữ nghĩa. GraphAgent chỉ '
        'bổ sung nguồn cha trong bản minh họa, chưa thay đổi xếp hạng hay prompt.')
    result+=extract('06_multi_agent','AGENT-05',
        'MultiAgentTeam.ask kiểm tra domain và câu hỏi trước khi xử lý. '
        'Report không tồn tại tạo bằng chứng rỗng và kết thúc bằng '
        'insufficient_evidence; domain sai/câu hỏi rỗng gây ValueError. '
        'Đây là ba đường lỗi riêng, không phải một câu trả lời thất bại '
        'được che bằng thử lại nhiều vòng.')
    return result


def insert(texts):
    texts=list(texts)
    texts[3]=texts[3].replace('## 3.11. Tiểu kết chương',
        model_section('cnn')+'\n## 3.11. Tiểu kết chương')
    texts[4]=texts[4].replace('## 4.12. Tiểu kết chương',
        model_section('rnn')+'\n'+model_section('lstm')+'\n## 4.12. Tiểu kết chương')
    texts[5]=texts[5].replace('## 5.7. Tiểu kết chương',
        knowledge_section()+'\n## 5.7. Tiểu kết chương')
    texts[6]=texts[6].replace('## 6.5. Đánh giá hệ đa Agent và phân tích lỗi',
        agent_section()+'\n## 6.5. Đánh giá hệ đa Agent và phân tích lỗi')
    return texts
