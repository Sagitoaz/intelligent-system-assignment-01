FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/backend:/workspace

WORKDIR /workspace

COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

COPY backend /workspace/backend
COPY models /workspace/models
COPY knowledge_graph /workspace/knowledge_graph
COPY shared_ml /workspace/shared_ml

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "/workspace/backend", "--host", "0.0.0.0", "--port", "8000"]
