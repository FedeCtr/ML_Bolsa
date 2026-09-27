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

# gunicorn: 4 workers, timeout alto (los escaneos tardan)
CMD ["gunicorn", "src.api.app:create_app()", \
     "--workers", "4", "--threads", "2", "--timeout", "120", \
     "--bind", "0.0.0.0:8000"]
