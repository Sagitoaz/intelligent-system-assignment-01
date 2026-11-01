"""Dựng và chạy notebook Chương 7 (07_multi_agent.ipynb): nhóm agent phối hợp và phản biện.

Dừng mọi máy chủ giữ mô hình Qwen (cổng 8044, 8045) trước khi chạy để không nạp mô hình hai lần.
Chạy: python scripts/chapter05/build_agent_notebook.py [--no-run]
Đặt CH5_LOWMEM=1 nếu máy thiếu bộ nhớ ảo (nạp Qwen lượng tử hóa int8, xem lowmem_llm.py).
"""
import argparse
import ast
import json
import sys
import textwrap
import time
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tieuluan import build_complete_notebooks as b  # noqa: E402

md, code = b.md, b.code
SOURCE = (ROOT / 'scripts/tieuluan/complete_agents.py').read_text(encoding='utf8')
TITLES = ['# Agent điều phối', '# Agent thống kê', '# Agent tính toán', '# Agent truy hồi và agent đồ thị', '# Agent viết', '# Agent phản biện', '# Bộ điều phối']


def segments():
    """Cắt mã nguồn thành các đoạn theo các biển đề mục, mỗi đoạn một ô mã."""
    cuts = [0]
    for title in TITLES:
        cuts.append(SOURCE.rfind('\n# ----', 0, SOURCE.index(title)) + 1)
    cuts.append(len(SOURCE))
    return [SOURCE[a:b_].strip('\n') for a, b_ in zip(cuts, cuts[1:])]


SEGMENTS = segments()


def cells():
    s = SEGMENTS
    return [
        md('''# Chương 7: Hệ đa agent: phối hợp, phản biện và so với một RAG đơn

        **Câu hỏi của chương.** Khi nào nhiều agent đáng giá hơn một RAG? Câu trả lời của chương này: chỉ khi (1) câu hỏi cần nhiều loại năng lực khác nhau
        (số liệu đo từ dữ liệu, tính toán chính xác, giải thích từ tài liệu) và (2) có một bước kiểm tra độc lập bằng bằng chứng chạy được, chứ không phải
        một mô hình tự đồng ý với chính nó. Với câu hỏi một bước như "Insulin do cơ quan nào tạo ra?", một RAG đơn là đủ; notebook sẽ chạy cả hai loại câu hỏi để thấy rõ điều này.

        **Các agent.** Điều phối (lập kế hoạch bằng luật), thống kê (pandas trên CDC BRFSS 2023, 150.000 người), tính toán (`safe_calculate`), truy hồi (kho NIDDK/CDC/MedlinePlus),
        đồ thị (tra nguồn trong đồ thị tri thức, dùng cho tài chính), viết (mô hình Qwen2.5-1.5B-Instruct viết nháp tự do) và **phản biện**: một bộ kiểm tra chạy được, không phải mô hình ngôn ngữ,
        trích mọi con số trong bản nháp rồi đối chiếu từng số với kết quả công cụ và đoạn tài liệu. Có lỗi thì gửi Message yêu cầu agent viết sửa; tối đa 2 vòng.

        Demo localhost ở `/agents` (cổng 8044) chạy cùng mã này. Các ô bên dưới có nhãn AGENT-01 đến AGENT-15.'''),
        code('AGENT-01', textwrap.dedent(b.SETUP) + textwrap.dedent('''
        if os.environ.get('CH5_LOWMEM') == '1':
            import scripts.chapter05.lowmem_llm as lowmem; lowmem.install()
            print('Nạp Qwen ở chế độ tiết kiệm bộ nhớ (int8), xem lowmem_llm.py')
        ''')),
        md('''## 1. State, Message và các hàm kiểm số

        `State` là bảng trạng thái chung: câu hỏi, kế hoạch, sự kiện của công cụ (`facts`, `derived`), các đoạn tài liệu, các bản nháp (`drafts`), các lần phản biện (`reviews`) và danh sách `messages`.
        Mỗi agent không gọi trực tiếp agent khác mà ghi một `Message(sender, receiver, kind, payload)` vào State (kế hoạch, sự kiện, bản nháp, yêu cầu sửa, kết luận).
        `extract_numbers` tìm mọi con số (nhận cả định dạng Việt "14,7%" và "150.000", lẫn Anh "37.3 million"); `close` cho phép làm tròn hợp lý (số viết với d chữ số thập phân thì sai lệch tối đa nửa đơn vị cuối).'''),
        code('AGENT-02', s[0]),
        md('''## 2. Agent điều phối (Planner)

        Dùng luật và từ khóa (đã bỏ dấu) để quyết định: câu hỏi ngoài phạm vi thì từ chối ngay; câu chỉ cần kiến thức thì đánh dấu "đơn giản: một RAG là đủ" và chỉ gọi truy hồi, viết, phản biện;
        câu cần số liệu thì lập danh sách sự kiện cần tính (một tỉ lệ, hai nhóm, nhóm cao nhất và thấp nhất...) cùng một phép tính (chia hoặc trừ). Kế hoạch này xác định, nhanh, không dùng mô hình sinh.'''),
        code('AGENT-03', s[1]),
        md('''## 3. Agent thống kê

        Mỗi sự kiện được tính bằng pandas trên mẫu BRFSS (qua `DiabetesStats.apply_filters`) và trả về cấu trúc: giá trị, tử số, mẫu số, nhãn tiếng Việt và tiếng Anh, câu mô tả, nguồn.
        Số này là số đo, không phải số mô hình nhớ.'''),
        code('AGENT-04', s[2]),
        md('''## 4. Agent tính toán

        Lấy hai sự kiện của agent thống kê rồi gọi `safe_calculate` (không dùng `eval`, kiểm tra đơn vị và chia cho 0). Mô hình ngôn ngữ không được tự chia hay trừ.'''),
        code('AGENT-05', s[3]),
        md('''## 5. Agent truy hồi và agent đồ thị

        Agent truy hồi dịch phần giải thích của câu hỏi sang tiếng Anh, truy hồi hybrid (BM25 + vector) và rút hai câu giải thích tốt nhất có mã nguồn. Nếu không đoạn nào chứa từ khóa của truy vấn,
        nó báo kho tài liệu không có thông tin (nhóm sẽ trả lời chỉ bằng số liệu hoặc từ chối). Agent đồ thị chỉ dùng cho tài chính: tra đoạn thuộc báo cáo và công ty nào.'''),
        code('AGENT-06', s[4]),
        md('''## 6. Agent viết

        Bản nháp đầu do Qwen viết tự do từ số của công cụ và các đoạn tài liệu (nó có thể lẫn số trong tài liệu hoặc bịa). Khi bị phản biện, vòng 1 Qwen được nhắc lại cùng danh sách lỗi;
        vòng 2 agent viết dựng lại câu trả lời xác định từ số của công cụ và một câu giải thích trích nguyên văn, nên không thể sai số. Bản đã duyệt được dịch sang tiếng Việt rồi phản biện kiểm lại.'''),
        code('AGENT-07', s[5]),
        md('''## 7. Agent phản biện

        Không dùng mô hình ngôn ngữ. Các kiểm tra: (a) mỗi con số trong bản nháp phải khớp một sự kiện của công cụ (cho phép làm tròn) hoặc một số trong đoạn tài liệu;
        số chỉ có trong tài liệu mà câu lại nói về mẫu khảo sát, hay không nói là lấy từ tài liệu, bị gắn cờ; (b) bản nháp phải chứa mọi số kết quả chính; (c) phải nhắc rằng đây là số liệu của mẫu;
        (d) mã nguồn trích dẫn phải có thật. Giới hạn: không kiểm ngữ nghĩa (câu "huyết áp cao làm tăng nguy cơ" có đúng ý tài liệu hay không).'''),
        code('AGENT-08', s[6]),
        md('''## 8. Bộ điều phối và vòng phản biện

        `MultiAgentTeam.ask` chạy kế hoạch: thống kê, tính toán, truy hồi, đồ thị, viết nháp, rồi vòng "phản biện, viết lại" với tối đa 2 vòng sửa. Điều kiện dừng: phản biện chấp nhận, hoặc hết vòng
        (trả lời kèm cảnh báo). Cuối cùng bản tiếng Việt được kiểm lại một lần nữa và bộ điều phối kết luận trạng thái bằng lời.'''),
        code('AGENT-09', s[7]),
        md('''## 9. Khởi tạo hệ thống

        Nạp kho tài liệu, hai bộ truy hồi, mô hình Qwen, đồ thị tri thức và dữ liệu BRFSS. Mô hình chiếm 6 GB RAM (hoặc khoảng 2,5 GB ở chế độ int8), nên hãy dừng máy chủ 8044 trước khi chạy notebook này.'''),
        code('AGENT-10', '''
        from deployment.chapter04.app import rag, agent
        from src.chapter04.knowledge import load_graph
        from scripts.tieuluan.complete_agents import MultiAgentTeam, State, build_dialogue, compare_baseline, CriticAgent, CONCLUSIONS
        graphs = {'finance':load_graph(ROOT/'knowledge_graph/chapter04/finance_graph.json.gz'),
                  'diabetes':load_graph(ROOT/'knowledge_graph/chapter04/diabetes_documents_graph.json.gz')}
        engine = rag(); engine.generator.load()
        team = MultiAgentTeam(engine, graphs, agent().stats)
        print('Mô hình sinh:', engine.model_name, '| kho tài liệu tiểu đường:', len(engine.corpora['diabetes']), 'đoạn')
        print('Các agent:', [type(a).__name__ for a in team.agents])
        '''),
        md('''## 10. Một câu phức hợp chạy đủ các agent

        Câu hỏi cần ba loại việc: đếm trên dữ liệu (tỉ lệ ở nhóm huyết áp cao và ở cả mẫu), phép chia chính xác, và lời giải thích từ tài liệu. Mỗi agent kể lại việc của mình; bên dưới là cuộc trao đổi viết và phản biện.'''),
        code('AGENT-11', '''
        question = 'Tỉ lệ tiểu đường ở người huyết áp cao gấp mấy lần cả mẫu, và vì sao huyết áp cao làm tăng nguy cơ?'
        t0 = time.perf_counter(); state = team.ask(question, 'diabetes'); team_seconds = time.perf_counter()-t0
        print('KẾ HOẠCH:', state.plan['mode_label'], '|', state.plan['summary'], '\\n')
        for step in state.steps:
            print(f"[{step['role']}] ({step['seconds']:.1f} giây) {step['summary'][:700]}")
            print('    =>', step['outcome'], '\\n')
        print('CUỘC TRAO ĐỔI VIẾT - PHẢN BIỆN:')
        for m in build_dialogue(state):
            print(' ', m['speaker'], f"[vòng {m['round']}]", m['text'])
            for issue in m.get('issues', []): print('      -', issue)
        print('\\nCÂU TRẢ LỜI CUỐI:', state.answer)
        print('TRẠNG THÁI:', CONCLUSIONS[state.status][0], '| số vòng sửa:', state.revisions, '| tổng', round(team_seconds), 'giây')
        '''),
        md('''## 11. So với một RAG đơn trên cùng câu hỏi

        RAG đơn (như Chương 5) chỉ truy hồi rồi viết. Nó không có số liệu từ dữ liệu khảo sát, nên hoặc từ chối hoặc trích một con số của tài liệu (về cả nước Mỹ) mà không liên quan.
        Hàm `compare_baseline` kiểm xem câu trả lời có chứa các số đúng do công cụ tính hay không.'''),
        code('AGENT-12', '''
        t0 = time.perf_counter(); single = engine.answer(question, 'diabetes'); single_seconds = time.perf_counter()-t0
        verdict = compare_baseline(single['answer'], state)
        print('MỘT RAG ĐƠN (%.0f giây, 2 bước):' % single_seconds, single['answer'])
        print('Số đúng có trong câu trả lời:', verdict['matched'], '/', verdict['total'], '| số không đến từ dữ liệu:', verdict['foreign_numbers'])
        print('NHÓM AGENT (%.0f giây, %d bước):' % (team_seconds, len(state.steps)), state.answer)
        '''),
        md('''## 12. Kiểm thử agent phản biện bằng bản nháp sai cho trước

        Để thấy cách phản biện hoạt động, ô này đưa cho nó một bản nháp sai do **người viết notebook tạo ra** (không phải đầu ra của mô hình): lấy "37,3 triệu" trong tài liệu NIDDK làm số của mẫu
        và nêu tỉ số sai. Phản biện phải gắn cờ từng số, kèm bằng chứng đúng của công cụ. Đây là kiểm thử đơn vị của bộ kiểm tra; các lỗi do Qwen viết thật nằm ở phần đánh giá bên dưới.'''),
        code('AGENT-13', '''
        probe = State(question, 'diabetes'); probe.language = 'vi'
        from scripts.tieuluan.complete_agents import PlannerAgent, StatsAgent, CalculatorAgent
        PlannerAgent(agent().stats).act(probe); StatsAgent(agent().stats).act(probe); CalculatorAgent().act(probe)
        probe.evidence = state.evidence
        wrong = ('In the survey sample, 37.3 million people with high blood pressure have diabetes, which is 2.9 times the overall rate.')
        result = CriticAgent(engine.corpora).check(wrong, probe)
        for item in result['numbers']: print('SỐ', item['text'], '->', item['verdict'])
        for issue in result['issues']: print(' -', issue['type'], ':', issue['message'])
        assert not result['ok']
        '''),
        md('''## 13. Câu đơn giản và câu ngoài phạm vi

        Câu "Insulin do cơ quan nào tạo ra?" chỉ cần đọc tài liệu: kế hoạch chỉ gọi truy hồi và viết, nhóm agent không có thêm bằng chứng nào để kiểm chứng, và chậm hơn hoặc bằng một RAG đơn.
        Câu "Giá vàng hôm nay?" bị agent điều phối từ chối ngay.'''),
        code('AGENT-14', '''
        rows = []
        for q in ['Insulin do cơ quan nào trong cơ thể tạo ra?', 'Giá vàng hôm nay là bao nhiêu?']:
            t0 = time.perf_counter(); st = team.ask(q, 'diabetes'); ts = time.perf_counter()-t0
            t0 = time.perf_counter(); rg = engine.answer(q, 'diabetes'); rs = time.perf_counter()-t0
            rows.append({'câu hỏi':q, 'kế hoạch':st.plan['mode_label'], 'nhóm agent':st.answer[:140], 'trạng thái nhóm':st.status, 'giây nhóm':round(ts),
                         'một RAG đơn':rg['answer'][:140], 'trạng thái RAG':rg['status'], 'giây RAG':round(rs)})
        display(pd.DataFrame(rows).T)
        '''),
        md('''## 14. Đánh giá định lượng

        `scripts/chapter05/eval_multiagent.py` chạy 21 câu phức hợp, 5 câu đơn giản và 4 câu ngoài phạm vi qua cả hai hệ (qua API của máy chủ 8044). Đáp án của câu phức hợp được tính độc lập bằng pandas
        từ CSV (không dùng mã của agent thống kê). Một số được coi là đúng khi sai không quá 0,5 điểm phần trăm (tỉ lệ), 1% tương đối (tỉ số, số đếm) hoặc 0,1 (BMI). Bản đầy đủ chạy rất lâu (hàng chục phút đến hàng giờ trên CPU),
        nên notebook đọc kết quả từ `results/chapter05/multiagent_eval.json` thay vì chạy lại; chạy lại bằng `python scripts/chapter05/eval_multiagent.py --resume`.'''),
        code('AGENT-15', '''
        from scripts.chapter05.eval_multiagent import print_table, summarize
        data = json.loads((ROOT/'results/chapter05/multiagent_eval.json').read_text(encoding='utf8'))
        print('Số câu đã chấm:', data['questions'])
        print_table(data['summary'], data['questions'])
        table = pd.DataFrame([{'id':r['id'], 'loại':r['kind'], 'RAG đơn đúng':r['rag']['correct'], 'nhóm agent đúng':r['team']['correct'], 'nháp đầu đúng':r['team'].get('draft_headline_correct'),
                               'vòng sửa':r['team']['revisions'], 'giây RAG':round(r['rag']['seconds'] or 0), 'giây nhóm':round(r['team']['seconds'] or 0)} for r in data['results']])
        display(table)
        '''),
        md('''## 15. Phân tích: phản biện đã bắt lỗi thật nào?

        Dưới đây là các câu mà bản nháp đầu của Qwen bị phản biện bác (nếu có), kèm lỗi cụ thể và bản cuối. Nếu không có câu nào bị bác thì ô in ra như vậy: khi bản nháp đúng, phản biện chấp nhận ngay.'''),
        code('AGENT-16', '''
        caught = [r for r in data['results'] if r['team'].get('draft_review_ok') is False]
        print('Số câu có bản nháp đầu bị bác:', len(caught), '/', len(data['results']))
        for r in caught[:4]:
            print('\\n', r['id'], r['question'])
            print('  Bản nháp đầu:', r['team'].get('draft_text'))
            for issue in r['team'].get('draft_issues', []): print('  Phản biện:', issue)
            print('  Số vòng sửa:', r['team']['revisions'], '| trạng thái cuối:', r['team']['status'])
            print('  Câu trả lời cuối:', r['team']['answer'])
        '''),
        md('''## Câu hỏi trình bày

        1. Vì sao câu "Insulin do cơ quan nào tạo ra?" không cần nhiều agent, còn câu về huyết áp cao thì cần? Chỉ ra ba loại năng lực mà RAG đơn không có.
        2. Agent phản biện khác một mô hình tự kiểm tra ở điểm nào? Nó bắt được loại lỗi nào, và loại lỗi nào nó bỏ sót (gợi ý: một câu giải thích sai ý tài liệu nhưng không chứa số)?
        3. Điều kiện dừng của vòng phản biện là gì và vì sao cần giới hạn 2 vòng?
        4. Trong bảng đánh giá, nhóm agent hơn RAG đơn ở loại câu nào, hòa hoặc thua ở loại câu nào, và vì sao?''')]


def main():
    sys.stdout.reconfigure(encoding='utf8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-run', action='store_true')
    args = parser.parse_args()
    b.NBROOT.mkdir(parents=True, exist_ok=True)
    b.OUT.mkdir(parents=True, exist_ok=True)
    cs = cells()
    for cell in cs:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    notebook = nbf.v4.new_notebook(cells=cs, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                                       'language_info': {'name': 'python', 'version': sys.version.split()[0]}})
    path = b.NBROOT / '07_multi_agent.ipynb'
    nbf.write(notebook, path)
    if args.no_run:
        return
    print('Executing 07_multi_agent', flush=True)
    start = time.perf_counter()
    NotebookClient(notebook, timeout=2400, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbf.validate(notebook)
    nbf.write(notebook, path)
    (b.OUT / 'execution_ch7.json').write_text(json.dumps([{'notebook': str(path.relative_to(ROOT)), 'seconds': time.perf_counter() - start}], ensure_ascii=False, indent=2), encoding='utf8')
    print('  xong sau %.0f giây' % (time.perf_counter() - start), flush=True)


if __name__ == '__main__':
    main()
