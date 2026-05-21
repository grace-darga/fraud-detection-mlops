FROM python:3.11-slim

LABEL maintainer="fraud-detection-mlops"
LABEL version="1.0.0"
LABEL description="API de détection de fraude bancaire"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "xgboost==2.1.4" --no-deps && \
    pip install --no-cache-dir scipy

COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/scaler.pkl ./data/processed/scaler.pkl

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]