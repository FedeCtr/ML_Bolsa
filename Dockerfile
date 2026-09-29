FROM python:3.12-slim

WORKDIR /app

# dependencias primero (cache de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# codigo + modelos + vendor de charts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY config.yaml .

# usuario no-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
ENV PORT=8000

# API unificada FastAPI (Sprint 1): workers uvicorn, timeout alto (los escaneos tardan).
# 2 workers: cada uno carga el ensemble en memoria (~200-400MB).
CMD ["gunicorn", "src.api.fastapi_app:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", "--threads", "4", "--timeout", "120", \
     "--bind", "0.0.0.0:8000"]
