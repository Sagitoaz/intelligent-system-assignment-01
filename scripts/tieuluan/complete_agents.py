"""Bounded, role-based multi-agent demonstration with explicit messages."""
from dataclasses import dataclass, field


@dataclass
class Message:
    sender: str
    receiver: str
    payload: dict


@dataclass
class State:
    question: str
    domain: str
    report: str | None = None
    evidence: list = field(default_factory=list)
    graph_context: list = field(default_factory=list)
    answer: str = ''
    status: str = 'pending'
    messages: list = field(default_factory=list)
    query: str = ''
    language: str = ''
    english_answer: str = ''

    def send(self, sender, receiver, **payload):
        self.messages.append(Message(sender, receiver, payload))


class RetrievalAgent:
    def __init__(self, engine):
        self.engine = engine

    def act(self, state):
        # Câu tiếng Việt được dịch sang truy vấn tiếng Anh vì kho tài liệu bằng tiếng Anh.
        if hasattr(self.engine, 'to_english_query'):
            state.query, state.language = self.engine.to_english_query(state.question)
        else:
            state.query, state.language = state.question, 'en'
        state.evidence = self.engine.retrieve(
            state.query, state.domain, k=3, report=state.report)
        state.send('retrieval', 'graph',
                   ids=[e['id'] for e in state.evidence])


class GraphAgent:
    def __init__(self, graphs):
        self.graphs = graphs

    def act(self, state):
        graph = self.graphs.get(state.domain)
        if graph is not None:
            for item in state.evidence:
                node = 'passage:' + item['id']
                if node in graph:
                    state.graph_context.append({
                        'id': item['id'],
                        'parents': list(graph.predecessors(node))})
        state.send('graph', 'writer',
                   matched=len(state.graph_context))


class WriterAgent:
    def __init__(self, generator):
        self.generator = generator

    def act(self, state):
        from src.chapter04.rag import INSUFFICIENT_EN, INSUFFICIENT_VI, lexical_support
        refuse = INSUFFICIENT_VI if state.language == 'vi' else INSUFFICIENT_EN
        if not state.evidence or not lexical_support(state.query or state.question, state.evidence):
            state.status = 'insufficient_evidence'
            state.answer = refuse
        elif hasattr(self.generator, 'generate_full'):
            full = self.generator.generate_full(
                state.question, state.evidence, state.language or None,
                state.query or None)
            state.answer = full['answer']
            state.english_answer = full['english_answer'] or ''
            if full['insufficient']:
                state.status = 'insufficient_evidence'
        else:
            state.answer = self.generator.generate(
                state.question, state.evidence)
        state.send('writer', 'verifier', answer=state.answer)


class VerifierAgent:
    def __init__(self, corpora):
        self.allowed = {d: {x['id'] for x in c}
                        for d, c in corpora.items()}

    def act(self, state):
        ids = [e['id'] for e in state.evidence]
        valid = bool(ids) and all(
            cid in self.allowed[state.domain] for cid in ids)
        if state.status == 'insufficient_evidence' and state.answer.strip():
            pass  # Agent viết đã từ chối: giữ nguyên câu từ chối.
        elif not valid or not state.answer.strip():
            state.answer = ('Không đủ bằng chứng.' if state.language == 'vi'
                            else 'Insufficient evidence')
            state.status = 'insufficient_evidence'
        else:
            state.status = 'generated_unverified'
        state.send('verifier', 'coordinator', citation_ids_valid=valid,
                   semantic_support_checked=False, status=state.status)


class MultiAgentTeam:
    def __init__(self, engine, graphs):
        self.agents = [RetrievalAgent(engine), GraphAgent(graphs),
                       WriterAgent(engine.generator),
                       VerifierAgent(engine.corpora)]

    def ask(self, question, domain, report=None):
        if domain not in ('diabetes', 'finance'):
            raise ValueError('Unsupported domain')
        if not question.strip():
            raise ValueError('Empty question')
        state = State(question, domain, report)
        for agent in self.agents:  # Four bounded turns; no retry loop.
            agent.act(state)
        return state
