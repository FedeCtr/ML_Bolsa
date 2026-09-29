FROM python:3.12-slim

WORKDIR /app

# libgomp1: sklearn/xgboost lo necesitan en slim y no viene de serie
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# dependencias primero (cache de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# codigo + modelos + vendor de charts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY config.yaml .

# migraciones + entrypoint
COPY alembic.ini ./
COPY alembic_migrations/ ./alembic_migrations/
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# usuario no-root + directorios de escritura para los volumenes nombrados
# (app-data -> /app/data, logs del scheduler); si no existen con dueño correcto
# el montaje inicial queda root:root y el contenedor no puede escribir.
RUN useradd -m appuser \
    && mkdir -p /app/data /app/logs /app/models \
    && chown -R appuser /app
USER appuser

EXPOSE 8000
ENV PORT=8000

# API unificada FastAPI (Sprint 1): workers uvicorn, timeout alto (los escaneos tardan).
# 2 workers: cada uno carga el ensemble en memoria (~200-400MB).
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["gunicorn", "src.api.fastapi_app:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", "--threads", "4", "--timeout", "120", \
     "--bind", "0.0.0.0:8000"]
