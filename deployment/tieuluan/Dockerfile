FROM python:3.12-slim
WORKDIR /app
COPY deployment/tieuluan/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 1000 user
COPY --chown=user:user src/tieuluan /app/src/tieuluan
COPY --chown=user:user deployment/tieuluan/app.py /app/deployment/tieuluan/app.py
COPY --chown=user:user data/tieuluan/processed /app/data/tieuluan/processed
COPY --chown=user:user models/tieuluan/full/*__scratch__11.npz /app/models/tieuluan/full/
COPY --chown=user:user results/tieuluan/full /app/results/tieuluan/full
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["streamlit", "run", "deployment/tieuluan/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
