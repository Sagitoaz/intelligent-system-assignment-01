FROM python:3.12-slim
WORKDIR /workspace
COPY scripts/chapter04/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY src/chapter04 /workspace/src/chapter04
COPY deployment/chapter04 /workspace/deployment/chapter04
COPY data/chapter04/diabetes_corpus.jsonl /workspace/data/chapter04/diabetes_corpus.jsonl
COPY data/chapter04/finance_corpus.jsonl /workspace/data/chapter04/finance_corpus.jsonl
COPY results/chapter04 /workspace/results/chapter04
COPY knowledge_graph/chapter04 /workspace/knowledge_graph/chapter04
COPY models/chapter04 /workspace/models/chapter04
EXPOSE 8044
CMD ["uvicorn", "deployment.chapter04.app:app", "--host", "0.0.0.0", "--port", "8044"]
