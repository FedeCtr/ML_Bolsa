# Despliegue en la nube (AWS / GCP)

Stack del SaaS (Sprint 1):

| Servicio | Puerto | Stack | Uso |
|---|---|---|---|
| API unificada | 8000 | FastAPI + gunicorn/uvicorn | senales, screener, paper trading, watchlists, transparencia (`/docs` con OpenAPI) |
| Frontend | 3000 | Next.js 15 + Tailwind 4 | dashboard institucional (`frontend/`, consume la API con `NEXT_PUBLIC_API_URL`) |
| Scheduler | — | APScheduler (proceso o embebido) | escaneo cada 5 min -> `signal_cache` + Redis (`scripts/run_scheduler.py` o `SCHEDULER_EMBEDDED=1`) |
| PostgreSQL | 5432 | postgres:16-alpine | estado del SaaS (users, paper, watchlists, signal_cache) |
| Redis | 6379 | redis:7-alpine | cache de lecturas calientes (<1s) |

En dev sin Docker, `src/db/session.py` cae a SQLite (`data/saas.db`) y
`src/cache.py` a un cache en memoria con TTL: la API funciona igual.

## Local

```bash
docker compose up -d --build    # api + postgres + redis (healthchecks)
# migraciones contra PG:
DATABASE_URL=postgresql+psycopg://mlbolsa:mlbolsa@localhost:5432/mlbolsa \
  alembic upgrade head
uvicorn src.api.fastapi_app:app --port 8010    # /docs con OpenAPI

# scheduler standalone (o SCHEDULER_EMBEDDED=1 al arrancar la API):
python scripts/run_scheduler.py

# frontend Next.js (Sprint 4):
cd frontend && npm install && npm run dev      # http://localhost:3000
```

- Migraciones: `alembic upgrade head` (env de `alembic_migrations/`, respeta
  `ALEMBIC_DATABASE_URL`/`DATABASE_URL`). En PG prod, Alembic manda; el
  `create_all` de `init_db()` es solo el atajo de dev.
- La terminal Flask heredada (`python -m src.api.app`) sigue disponible en dev,
  pero la API canónica del producto es FastAPI.

## AWS (ECS Fargate + ECR)

1. **Build & push**:
   ```bash
   aws ecr create-repository --repository-name mlbolsa
   docker build -t mlbolsa .
   docker tag mlbolsa:latest <account>.dkr.ecr.<region>.amazonaws.com/mlbolsa:latest
   aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker push <account>.dkr.ecr.<region>.amazonaws.com/mlbolsa:latest
   ```
2. **Task definition**: 1 vCPU / 2 GB, puerto 8000; `DATABASE_URL` apuntando a
   RDS PostgreSQL 16 y `REDIS_URL` a ElastiCache (o MemoryDB).
3. **Service**: ECS Fargate detrás de un ALB (health check `/health`),
   auto-scaling por CPU/requests.
4. **Latencia**: las lecturas calientes se sirven de Redis (<1s); las
   predicciones individuales son < 50 ms con el modelo en memoria; el cuello
   real es yfinance. Con feeds de pago (Polygon/Alpaca, Sprint 2) se
   sustituye `DataCollector` y el resto del pipeline no cambia.
5. **Programado** (EventBridge → ECS RunTask): tarea cada 5 min que refresca
   `signal_cache` y tras el cierre ejecuta `POST /api/paper/resolve`.

## GCP (Cloud Run)

```bash
gcloud run deploy mlbolsa --source . --region europe-west1 \
  --allow-unauthenticated --memory 2Gi --cpu 1 --timeout 300 \
  --set-env-vars PYTHONUNBUFFERED=1,DATABASE_URL=...,REDIS_URL=...
```

Cloud Run escala a cero; usar Cloud SQL (Postgres 16) + Memorystore (Redis).

## Variables y estado

| Variable / ruta | Contenido |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://user:pass@host:5432/db` (fallback SQLite dev) |
| `REDIS_URL` | `redis://host:6379/0` (fallback memoria con TTL) |
| `models/*.pkl` | ensemble, calibración, OOF, metadata |
| `reports/sp500_expectancy.json` | estudio de generalización (servido por `/api/expectancy`) |
| `data/raw/*.csv` | cache OHLCV (efímero, se re-descarga) |

Paper trading y watchlists ya no viven en ficheros: tablas `paper_signals`,
`paper_outcomes` y `watchlists` en Postgres (con importación automática de los
ficheros legado de una sola vez si la tabla está vacía).

## Reentrenado y monitorización

- Reentrenar cuando caiga la DA paper o haya drift de features
  (`scripts/train_advanced.py --optimize`); el modelo nuevo se toma con
  reinicio del contenedor (image tag por fecha).
- Logs estructurados en stdout (CloudWatch / Cloud Logging).
- `/health` expone DB, cache (backend + ping), modelo y estado del escaneo.
