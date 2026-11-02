FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt requirements-nexus.txt ./
RUN pip install --no-cache-dir -r requirements-nexus.txt
COPY voicerag ./voicerag
COPY nexus ./nexus
COPY data/docs ./data/docs
COPY data/eval.json ./data/eval.json
RUN useradd -m nexus && mkdir -p /data && chown nexus /data
USER nexus
ENV NEXUS_DATA_DIR=/data
EXPOSE 8100
# one worker: the scheduler and rate limits live in-process
CMD ["uvicorn", "nexus.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
