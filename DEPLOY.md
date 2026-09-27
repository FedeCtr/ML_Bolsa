# Despliegue en la nube (AWS / GCP)

La app se empaqueta en Docker (`Dockerfile`) y expone:

| Servicio | Puerto | Stack | Uso |
|---|---|---|---|
| Terminal + API (Flask) | 8000 | gunicorn | dashboard, screener, watchlists, paper trading, backtests |
| Inferencia rápida | 8010 | FastAPI + uvicorn | `/infer/{ticker}` con docs OpenAPI en `/docs` |

## Local

```bash
docker compose up --build        # terminal + API en http://localhost:8000
pip install fastapi uvicorn      # (una vez) para el servidor de inferencia
uvicorn scripts.serve_fastapi:app --port 8010   # /docs con OpenAPI
```

> Nota: instalar `fastapi` y `uvicorn` no es necesario para la terminal Flask;
> solo para el servidor de inferencia de baja latencia.

## AWS (ECS Fargate + ECR)

1. **Build & push**:
   ```bash
   aws ecr create-repository --repository-name mlbolsa
   docker build -t mlbolsa .
   docker tag mlbolsa:latest <account>.dkr.ecr.<region>.amazonaws.com/mlbolsa:latest
   aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker push <account>.dkr.ecr.<region>.amazonaws.com/mlbolsa:latest
   ```
2. **Task definition**: 1 vCPU / 2 GB, puerto 8000; monta un volumen EFS en
   `/app/data` (persistencia de paper trading y watchlists) o usa S3 para
   snapshots de `models/`.
3. **Service**: ECS Fargate detrás de un ALB (health check `/health`),
   auto-scaling por CPU/requests. 4 workers de gunicorn por tarea.
4. **Latencia**: las predicciones individuales son < 50 ms una vez cargado el
   modelo; el cuello real es yfinance. Para producción con feeds de pago
   (Polygon/Alpaca) sustituir `DataCollector` por el proveedor — el resto del
   pipeline no cambia.
5. **Programado** (EventBridge → ECS RunTask): tarea diaria tras el cierre que
   ejecuta el escaneo del universo + `POST /api/paper/resolve` para resolver
   las señales paper contra los cierres reales.

## GCP (Cloud Run)

```bash
gcloud run deploy mlbolsa --source . --region europe-west1 \
  --allow-unauthenticated --memory 2Gi --cpu 1 --timeout 300 \
  --set-env-vars PYTHONUNBUFFERED=1
```

Cloud Run escala a cero; monta `data/` en un bucket via volume mount (Cloud
Run v2 + GCS FUSE) o mueve paper trading a Cloud SQL.

## Variables y archivos de estado

| Ruta | Contenido | Persistencia recomendada |
|---|---|---|
| `models/*.pkl` | ensemble, calibración, OOF, metadata | volumen/EFS o reentrenar en el deploy |
| `data/paper_trading.db` | señales y outcomes paper | volumen persistente |
| `data/watchlists.json` | listas de usuario | volumen persistente |
| `data/raw/*.csv` | cache OHLCV | efímero (se re-descarga) |

## Reentrenado y monitorización

- Reentrenar cuando caiga la DA en `data/paper_trading.db` o drift de features
  (`scripts/train_advanced.py --optimize`); el modelo nuevo se toma con
  reinicio del contenedor (image tag por fecha).
- Logs estructurados en stdout (recogidos por CloudWatch / Cloud Logging).
- `/health` expone estado del modelo y del job de escaneo para el ALB.
