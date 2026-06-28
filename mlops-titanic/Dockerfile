# =========================================================
# Dockerfile — Serviço de Inferência (API)
# =========================================================
FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema mínimas (compilação de libs científicas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY data/raw/ ./data/raw/
COPY models/ ./models/

EXPOSE 8000

# Healthcheck simples para orquestradores (docker-compose, k8s, etc)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
