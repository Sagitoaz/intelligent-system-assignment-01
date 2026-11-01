"""Đánh giá truy hồi trên kho tiểu đường: BM25, vector, hybrid (RRF) và Graph-RAG (project01), rồi làm mới đồ thị tài liệu.

Chạy mặc định (kho hiện tại):
    .venv\\Scripts\\python.exe scripts/chapter05/evaluate_diabetes_retrieval.py
Chạy lại trên kho cũ để so sánh (đặt kho/QA/KG cũ ở thư mục khác):
    ... --corpus OLD/corpus.jsonl --qa OLD/qa.json --kg OLD/kg.json --cache OLD/emb.npy --out OLD/eval.json --no-graph-build
Ghi results/chapter04/diabetes_retrieval_eval.json và knowledge_graph/chapter04/diabetes_documents_graph.json.gz.
"""
from pathlib import Path
import argparse, json, os, sys, time
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.chapter04.retrieval import BM25, DenseRetriever, Encoder, hybrid_search  # noqa: E402
from src.chapter04.knowledge import build_document_graph, save_graph  # noqa: E402

KS = (1, 3, 5)
sys.stdout.reconfigure(encoding='utf-8')


def score(rankings, golds):
    n = len(golds)
    out = {'questions': n}
    for k in KS:
        out[f'hit_at_{k}'] = sum(bool(set(r[:k]) & set(g)) for r, g in zip(rankings, golds)) / n
    out['mrr_at_5'] = sum(next((1 / (i + 1) for i, x in enumerate(r[:5]) if x in g), 0) for r, g in zip(rankings, golds)) / n
    return out


def graph_rankings(qa, corpus_path, kg_path):
    """Graph-RAG của Project 01: nút khớp truy vấn, mở rộng một bước, cộng điểm cho đoạn gắn với nút rồi BM25."""
    import project01.mas.kb as kb
    kb.CORPUS = str(corpus_path)
    kb.KG_PATH = str(kg_path)
    base = kb.KnowledgeBase()
    result = {}
    for name, graph in (('bm25_project01', False), ('graph_rag', True)):
        result[name] = [[p['id'] for p in base.search(q['question'], 5, graph=graph)['passages']] for q in qa]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=str(ROOT / 'data/chapter04/diabetes_corpus.jsonl'))
    ap.add_argument('--qa', default=str(ROOT / 'data/chapter04/diabetes_qa_demo.json'))
    ap.add_argument('--kg', default=str(ROOT / 'project01/artifacts/kg.json'))
    ap.add_argument('--cache', default=str(ROOT / 'results/chapter04/diabetes_embeddings.npy'))
    ap.add_argument('--out', default=str(ROOT / 'results/chapter04/diabetes_retrieval_eval.json'))
    ap.add_argument('--no-graph-build', action='store_true')
    args = ap.parse_args()
    corpus = [json.loads(x) for x in Path(args.corpus).read_text(encoding='utf-8').splitlines()]
    qa = json.loads(Path(args.qa).read_text(encoding='utf-8'))
    encoder = Encoder(ROOT / 'models/chapter04/embedding')
    bm25 = BM25(corpus)
    start = time.perf_counter()
    dense = DenseRetriever(corpus, encoder, args.cache)
    embed_seconds = time.perf_counter() - start
    qvec = encoder.encode([q['question'] for q in qa])
    rankings = {'bm25': [], 'dense': [], 'hybrid': []}
    for i, q in enumerate(qa):
        rankings['bm25'].append([x['id'] for x in bm25.search(q['question'], 5)])
        rankings['dense'].append([x['id'] for x in dense.search(q['question'], 5, query_vector=qvec[i])])
        rankings['hybrid'].append([x['id'] for x in hybrid_search(bm25, dense, q['question'], 5, query_vector=qvec[i])])
    rankings.update(graph_rankings(qa, args.corpus, args.kg))
    golds = [q['gold_ids'] for q in qa]
    metrics = {name: score(r, golds) for name, r in rankings.items()}
    words = [len(x['text'].split()) for x in corpus]
    pages = {}
    for x in corpus:
        pages.setdefault(x['source_url'], {'org': x.get('source_org', ''), 'chunks': 0})['chunks'] += 1
    result = {'corpus_chunks': len(corpus), 'corpus_pages': len(pages), 'words_mean': sum(words) / len(words), 'words_min': min(words), 'words_max': max(words),
              'questions': len(qa), 'metrics': metrics, 'embedding_shape': list(dense.vectors.shape), 'embedding_seconds': embed_seconds,
              'per_question': [{'id': q['id'], 'question': q['question'], 'gold_ids': q['gold_ids'],
                                **{name: next((i + 1 for i, c in enumerate(r[j]) if c in q['gold_ids']), None) for name, r in rankings.items()}}
                               for j, q in enumerate(qa)],
              'note': 'rank = vị trí đầu tiên của một đoạn đúng trong top 5 (null = không có). Graph-RAG và BM25 project01 dùng bộ BM25 riêng của project01/mas/kb.py.'}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.no_graph_build:
        graph = build_document_graph(corpus, 'diabetes')
        save_graph(graph, ROOT / 'knowledge_graph/chapter04/diabetes_documents_graph.json.gz')
        print('Đồ thị tài liệu:', graph.number_of_nodes(), 'nút,', graph.number_of_edges(), 'cạnh')
    print(f'{len(corpus)} đoạn, {len(pages)} trang, {len(qa)} câu hỏi')
    for name, m in metrics.items():
        print(f"{name:15s} hit@1={m['hit_at_1']:.3f} hit@3={m['hit_at_3']:.3f} hit@5={m['hit_at_5']:.3f} MRR={m['mrr_at_5']:.3f}")


if __name__ == '__main__':
    main()
