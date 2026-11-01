from .knowledge import safe_calculate
from .nlu import OP_VI, find_company, parse_calculation, wants_graph
from .stats import DiabetesStats, detect_language, fmt_int, fmt_num, strip_accents
import re

DOC_TITLES = {
    'what-is-diabetes': 'NIDDK · Bệnh tiểu đường là gì (What is diabetes?)',
    'symptoms-causes': 'NIDDK · Triệu chứng và nguyên nhân (Symptoms & causes)',
    'tests-diagnosis': 'NIDDK · Xét nghiệm và chẩn đoán (Tests & diagnosis)',
    'managing-diabetes': 'NIDDK · Kiểm soát bệnh tiểu đường (Managing diabetes)',
}
CONFIDENCE = {
    'computed': ('Số liệu do công cụ tính trực tiếp từ dữ liệu hoặc từ các số trong câu hỏi nên chính xác.', 'ok'),
    'generated': ('Câu trả lời do mô hình ngôn ngữ viết dựa trên các đoạn tài liệu trên; chỉ kiểm tra được mã nguồn trích dẫn, nội dung chưa được kiểm chứng. '
                  'Không dùng làm lời khuyên y tế hoặc đầu tư.', 'warn'),
    'insufficient': ('Hệ thống từ chối trả lời vì tài liệu truy hồi không chứa câu trả lời; đây là hành vi mong muốn thay vì bịa.', 'info'),
    'lookup': ('Kết quả tra trực tiếp từ đồ thị tri thức đã dựng sẵn nên chính xác theo kho tài liệu.', 'ok'),
    'unsupported': ('Câu hỏi chưa thuộc các dạng mà công cụ hỗ trợ.', 'info'),
}


def doc_title(chunk):
    if chunk.get('domain') == 'finance' or 'company' in chunk:
        return f"{chunk.get('company')} · báo cáo năm {chunk.get('year')} ({chunk.get('filename')})"
    slug = chunk['source_url'].rstrip('/').rsplit('/', 1)[-1]
    return DOC_TITLES.get(slug, slug.replace('-', ' ').title())


def highlight_terms(query, text):
    from .rag import content_tokens
    words = set(re.findall(r'[a-z0-9]+', text.lower()))
    return sorted(t for t in set(content_tokens(query)) if t in words)


def snippet(text, terms, width=320):
    text = ' '.join(str(text).split())
    if len(text) <= width:
        return text
    low = text.lower()
    pos = min((low.find(t) for t in terms if t in low), default=0)
    start = max(0, pos - width // 4)
    cut = text[start:start + width]
    return ('… ' if start else '') + cut + (' …' if start + width < len(text) else '')


def num(x, lang='vi'):
    if abs(x - round(x)) < 1e-9:
        return fmt_int(round(x), lang)
    s = f'{x:,.2f}'.rstrip('0').rstrip('.')
    return s.replace(',', 'X').replace('.', ',').replace('X', '.') if lang == 'vi' else s


class ToolAgent:
    """Bộ định tuyến công cụ có kiểm soát (luật + từ khóa); không phải agent LLM tự lập kế hoạch."""
    def __init__(self, rag, stats, graphs=None):
        self.rag = rag
        self.stats = stats
        self.graphs = graphs or {}

    # ---- định tuyến ------------------------------------------------------------------------
    @staticmethod
    def resolve_task(question, domain, task='auto', operation=None, values=None, company=None):
        if task != 'auto':
            return task
        if operation is not None:
            if values is None:
                raise ValueError('Calculator operation requires values')
            return 'calculate'
        if company:
            return 'graph'
        lower = question.lower()
        count_question = 'how many' in lower or 'bao nhiêu' in lower
        dataset_words = any(x in lower for x in ['patient', 'encounter', 'dataset', 'readmit', 'bệnh nhân', 'lượt nhập viện', 'dữ liệu', 'tái nhập'])
        return 'statistics' if domain == 'diabetes' and count_question and dataset_words else 'rag'

    def route(self, question, domain, task='auto', operation=None, values=None, company=None):
        """Định tuyến mở rộng: nhận câu tự nhiên cho máy tính, đồ thị và thống kê."""
        if task != 'auto' or operation is not None or company:
            return self.resolve_task(question, domain, task, operation, values, company)
        if parse_calculation(question) is not None:
            return 'calculate'
        if wants_graph(question) and any(find_company(question, self.companies(d)) for d in self.graphs):
            return 'graph'
        if hasattr(self.stats, 'answer') and domain == 'diabetes' and DiabetesStats.looks_statistical(question):
            return 'statistics'
        if not hasattr(self.stats, 'answer'):
            return self.resolve_task(question, domain)
        return 'rag'

    def companies(self, domain):
        graph = self.graphs.get(domain)
        return {n.split(':', 1)[1] for n in graph if n.startswith('company:')} if graph is not None else set()

    # ---- trả lời ---------------------------------------------------------------------------
    def ask(self, question, domain, task='auto', operation=None, values=None, units=None, report=None, company=None):
        if domain not in ['diabetes', 'finance']:
            raise ValueError('Unsupported domain')
        if not question.strip():
            raise ValueError('Empty question')
        trace = []
        lang = detect_language(question)
        if task == 'auto':
            task = self.route(question, domain, task, operation, values, company)
        else:
            task = self.resolve_task(question, domain, task, operation, values, company)
        if task == 'statistics':
            return self._statistics(question, domain, lang, trace)
        if task == 'calculate':
            return self._calculate(question, lang, operation, values, units, trace)
        if task == 'graph':
            return self._graph(question, domain, lang, company, trace)
        if task != 'rag':
            raise ValueError('Unsupported task')
        if self.rag is None:
            raise ValueError('RAG engine is not configured')
        return self._rag(question, domain, report, lang, trace)

    def _finish(self, route, label, result, steps, confidence, sources=None, lang='vi'):
        text, level = CONFIDENCE[confidence]
        return {**result, 'route': route, 'route_label': label, 'steps': steps, 'sources': sources or [],
                'confidence': {'kind': confidence, 'level': level, 'label': text}, 'language': lang}

    def _statistics(self, question, domain, lang, trace):
        if domain != 'diabetes':
            raise ValueError('Statistics tool is for diabetes dataset')
        if hasattr(self.stats, 'answer'):
            result = self.stats.answer(question)
            if result is None:
                trace.append({'tool': 'dataset_statistics', 'result': None})
                return self._finish('statistics', 'Công cụ thống kê', {
                    'answer': self.stats.help_text(lang), 'status': 'unsupported', 'citations': [], 'trace': trace},
                    [{'title': 'Nhận ra đây có thể là câu hỏi số liệu', 'detail': 'Nhưng chưa khớp dạng nào mà công cụ thống kê hỗ trợ.'},
                     {'title': 'Trả lời lịch sự kèm danh sách câu mẫu', 'detail': 'Không đoán bừa số liệu.'}], 'unsupported', lang=lang)
            trace.append({'tool': 'dataset_statistics', 'result': {k: result.get(k) for k in ('intent', 'rows', 'value')}})
            steps = [
                {'title': 'Nhận ra đây là câu hỏi về số liệu', 'detail': 'Câu có từ như "bao nhiêu", "tỉ lệ", "so sánh", "trung bình" nên đi qua công cụ thống kê, không qua mô hình sinh.'},
                {'title': 'Hiểu câu hỏi', 'detail': result['interpretation']},
                {'title': f'Dùng pandas tính trên {fmt_int(self.stats.n)} dòng CSV', 'detail': 'Lọc các dòng phù hợp rồi đếm hoặc lấy trung bình; mọi số lấy trực tiếp từ dữ liệu.'},
                {'title': 'Viết kết quả thành câu văn', 'detail': 'Kèm bảng nhỏ nếu so sánh hoặc phân nhóm.'}]
            sources = [{'title': 'CDC BRFSS 2023: khảo sát sức khỏe người trưởng thành Hoa Kỳ', 'url': self.stats.url,
                        'snippet': f'Tệp {self.stats.csv_path.name}: mẫu ngẫu nhiên {fmt_int(self.stats.n)} người trả lời hợp lệ. Mẫu không có trọng số nên tỉ lệ chỉ mô tả mẫu này.',
                        'highlights': []}]
            out = {'answer': result['answer'], 'status': 'computed', 'value': result.get('value'), 'table': result.get('table'),
                   'interpretation': result['interpretation'], 'citations': [{'id': 'brfss2023', 'url': self.stats.url}], 'trace': trace}
            return self._finish('statistics', 'Công cụ thống kê', out, steps, 'computed', sources, lang)
        stats = self.stats
        lower = re.sub(r'\s+', ' ', question.lower().strip().rstrip('?.'))
        if lower in ['how many patients are in the dataset', 'how many unique patients are in the dataset', 'how many patients']:
            value = stats['unique_patients']
        elif lower in ['how many encounters are in the dataset', 'how many encounters']:
            value = stats['rows']
        elif re.fullmatch(r'how many encounters (?:were )?readmitted (?:within|under|after|more than) 30 days(?: in the dataset)?', lower):
            label = '>30' if 'after' in lower or 'more than' in lower else '<30'
            value = stats['readmitted'][label]
        else:
            raise ValueError('Unsupported dataset statistic; use a supported exact count intent')
        trace.append({'tool': 'dataset_statistics', 'result': value})
        return {'answer': str(value), 'value': value, 'status': 'computed', 'route': 'statistics',
                'citations': [{'id': 'uci296', 'url': 'https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008'}], 'trace': trace}

    def _calculate(self, question, lang, operation, values, units, trace):
        parsed = None
        if operation is None:
            parsed = parse_calculation(question)
            if parsed is None:
                return self._finish('calculate', 'Máy tính', {
                    'answer': ('Mình chưa tách được phép tính. Hãy nêu rõ hai số và phép tính, ví dụ: "Tính tăng trưởng doanh thu từ 100 lên 130" hoặc "250 chia 5".'
                               if lang == 'vi' else 'I could not extract a calculation. Give two numbers and an operation, e.g. "growth from 100 to 130".'),
                    'status': 'unsupported', 'citations': [], 'trace': trace}, [], 'unsupported', lang=lang)
            operation, values = parsed
        try:
            result = safe_calculate(operation, values, units)
        except ValueError as exc:
            if parsed is None:
                raise
            msg = {'Division by zero': 'Không thể chia cho 0.', 'Growth denominator is zero': 'Không tính được tăng trưởng vì số ban đầu bằng 0.'}.get(str(exc), str(exc))
            return self._finish('calculate', 'Máy tính', {'answer': msg, 'status': 'unsupported', 'citations': [], 'trace': trace}, [], 'unsupported', lang=lang)
        a, b = float(values[0]), float(values[1])
        v = result['value']
        vi = lang == 'vi'
        if operation == 'growth':
            direction = ('tăng' if v >= 0 else 'giảm') if vi else ('an increase' if v >= 0 else 'a decrease')
            answer = (f'Từ {num(a)} lên {num(b)} là {direction} {num(abs(v))}%, tức {"+" if v >= 0 else "−"}{num(abs(b - a))} so với số ban đầu.' if vi
                      else f'From {num(a, "en")} to {num(b, "en")} is {direction} of {num(abs(v), "en")}% ({num(b - a, "en")} on a base of {num(a, "en")}).')
            formula = f'({num(b)} − {num(a)}) ÷ {num(a)} × 100%'
        else:
            sym = {'add': '+', 'subtract': '−', 'multiply': '×', 'divide': '÷'}[operation]
            answer = f'{num(a, lang)} {sym} {num(b, lang)} = {num(v, lang)}.'
            formula = f'{num(a)} {sym} {num(b)}'
        trace.append({'tool': 'financial_calculator', 'inputs': {'operation': operation, 'values': values, 'units': units}, 'result': result})
        steps = [{'title': 'Nhận ra đây là một phép tính', 'detail': f'Câu có từ khóa của {OP_VI[operation]}.' if parsed else f'Yêu cầu nêu rõ {OP_VI[operation]}.'},
                 {'title': 'Tách hai số trong câu' if parsed else 'Nhận hai số từ yêu cầu', 'detail': f'Hai số là {num(a)} và {num(b)}.'},
                 {'title': 'Máy tính tính kết quả', 'detail': f'Công thức: {formula}. Dùng phép tính số học, không dùng mô hình sinh nên không thể sai số.'}]
        out = {**result, 'answer': answer, 'status': 'computed', 'trace': trace, 'citations': [], 'formula_text': formula,
               'note': 'Các số lấy từ câu hỏi hoặc từ người gọi; không tự trích từ báo cáo.'}
        return self._finish('calculate', 'Máy tính', out, steps, 'computed', [], lang)

    def _graph(self, question, domain, lang, company, trace):
        vi = lang == 'vi'
        if company is None:
            for d in [domain] + [x for x in self.graphs if x != domain]:
                company = find_company(question, self.companies(d))
                if company:
                    domain = d
                    break
        graph = self.graphs.get(domain)
        node = 'company:' + str(company)
        if graph is None or node not in graph:
            trace.append({'tool': 'query_graph', 'result': []})
            ans = (f'Không tìm thấy công ty "{company}" trong đồ thị tri thức. Hãy thử mã cổ phiếu như V, AAPL, JPM.' if vi
                   else f'Company "{company}" was not found in the knowledge graph. Try a ticker such as V, AAPL or JPM.')
            return self._finish('graph', 'Đồ thị tri thức', {'status': 'insufficient_evidence', 'answer': ans, 'citations': [], 'trace': trace}, [], 'unsupported', lang=lang)
        reports = []
        for n in graph.successors(node):
            reports.append({'node': n, **graph.nodes[n], 'passages': len(list(graph.successors(n)))})
        reports.sort(key=lambda r: (r.get('year', ''), r.get('filename', '')))
        years = sorted({r.get('year') for r in reports if r.get('year')})
        trace.append({'tool': 'query_graph', 'company': company, 'reports': len(reports)})
        n_pass = sum(r['passages'] for r in reports)
        ans = (f'Công ty {company} có {len(reports)} báo cáo trong kho: {", ".join(years)}. Tổng cộng {fmt_int(n_pass)} đoạn văn và dòng bảng đã được đánh chỉ mục.' if vi
               else f'Company {company} has {len(reports)} reports in the collection: {", ".join(years)}. {fmt_int(n_pass, "en")} passages and table rows are indexed.')
        steps = [{'title': 'Nhận ra đây là câu hỏi tra cứu đồ thị', 'detail': f'Câu nhắc tới công ty {company} và "báo cáo", nên tra đồ thị tri thức thay vì nhờ mô hình đoán.'},
                 {'title': f'Mở nút "Công ty {company}" trong đồ thị', 'detail': 'Đồ thị nối công ty → báo cáo → các đoạn văn/dòng bảng bằng các cạnh PUBLISHED và HAS_PASSAGE.'},
                 {'title': 'Đếm các báo cáo nối với công ty', 'detail': f'Tìm thấy {len(reports)} báo cáo.'}]
        table = {'columns': ['Năm', 'Tệp báo cáo', 'Số đoạn đã lập chỉ mục'] if vi else ['Year', 'Report file', 'Indexed passages'],
                 'rows': [[r.get('year'), r.get('filename'), fmt_int(r['passages'], lang)] for r in reports]}
        sources = [{'title': 'FinQA: báo cáo thường niên các công ty S&P 500', 'url': 'https://github.com/czyssrs/FinQA',
                    'snippet': f'{len(reports)} báo cáo của {company}: ' + ', '.join(r.get('filename', '') for r in reports[:4]) + (' …' if len(reports) > 4 else ''), 'highlights': []}]
        out = {'status': 'queried', 'answer': ans, 'reports': [{k: r[k] for k in ('node', 'year', 'filename', 'passages')} for r in reports], 'table': table,
               'citations': [], 'trace': trace}
        return self._finish('graph', 'Đồ thị tri thức', out, steps, 'lookup', sources, lang)

    def _rag(self, question, domain, report, lang, trace):
        result = self.rag.answer(question, domain, report)
        evidence = result.get('evidence', [])
        trace.append({'tool': 'search_documents', 'domain': domain, 'count': len(result['citations'])})
        trace.append({'tool': 'local_generator', 'model': result.get('model')})
        known = {x['id'] for x in self.rag.corpora[domain]}
        valid = all(c['id'] in known for c in result['citations'])
        trace.append({'tool': 'verify_citation_ids', 'valid': valid, 'semantic_support_checked': False})
        query = result.get('query_used', question)
        sources = []
        for x in evidence:
            terms = highlight_terms(query, str(x['text']))
            sources.append({'title': doc_title(x), 'url': x['source_url'], 'snippet': snippet(x['text'], terms), 'highlights': terms,
                            'locator': x.get('locator', x['id']), 'score': round(float(x.get('score', 0)), 4)})
        steps = [{'title': 'Nhận ra đây là câu hỏi kiến thức', 'detail': 'Không phải câu đếm hay phép tính nên đi theo đường RAG: tìm tài liệu rồi nhờ mô hình viết.'}]
        if result.get('translated'):
            steps.append({'title': 'Dịch câu hỏi sang truy vấn tiếng Anh', 'detail': f'Kho tài liệu bằng tiếng Anh, nên truy vấn dùng để tìm là: "{query}".'})
        steps.append({'title': f'Tìm {len(evidence)} đoạn tài liệu liên quan nhất', 'detail': 'Kết hợp tìm theo từ khóa (BM25) và tìm theo nghĩa (vector), rồi trộn thứ hạng.'})
        if result['status'] == 'insufficient_evidence':
            steps.append({'title': 'Mô hình không thấy câu trả lời trong các đoạn này', 'detail': 'Hệ thống từ chối trả lời thay vì dùng kiến thức ngoài hoặc bịa.'})
            conf = 'insufficient'
        else:
            if result.get('english_answer') and lang == 'vi':
                steps.append({'title': f'Mô hình {result.get("model")} viết câu trả lời tiếng Anh từ các đoạn', 'detail': 'Mô hình nhỏ đọc tiếng Anh chính xác hơn, nên nó viết bản tiếng Anh trước, chỉ dựa trên các đoạn vừa tìm.'})
                steps.append({'title': 'Dịch câu trả lời sang tiếng Việt', 'detail': 'Dùng bảng thuật ngữ y khoa/tài chính để dịch đúng từ (ví dụ pancreas là tuyến tụy).'})
            else:
                steps.append({'title': f'Mô hình {result.get("model")} viết câu trả lời', 'detail': 'Chỉ dựa trên các đoạn vừa tìm, trả lời cùng ngôn ngữ với câu hỏi.'})
            steps.append({'title': 'Kiểm tra mã nguồn trích dẫn', 'detail': ('Mọi mã đoạn trích dẫn đều có trong kho' if valid else 'Có mã trích dẫn không nằm trong kho')
                          + '; chưa kiểm tra được câu viết có đúng ý nguồn hay không.'})
            conf = 'generated'
        extra = {k: result[k] for k in ('query_used', 'translated', 'model', 'seconds', 'translate_seconds', 'english_answer') if k in result}
        out = {'answer': result['answer'], 'status': result['status'], 'citations': result['citations'],
               'evidence': [{'id': x['id'], 'text': x['text']} for x in evidence], 'trace': trace, 'citation_ids_valid': valid, **extra}
        return self._finish('rag', 'Tìm tài liệu rồi viết (RAG)', out, steps, conf, sources, lang)
