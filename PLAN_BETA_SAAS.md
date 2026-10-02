# PLAN BETA SAAS — ML_Bolsa Premium (respuesta al PRD v1.0)

> **Requerimientos: CONFIRMADOS.** Este documento fija el stack tecnológico definitivo,
> el cronograma de sprints hasta Beta cerrada y la trazabilidad de cada requisito del
> PRD a un sprint concreto. Supone 1 desarrollador + agente de código; con 2 personas
> en paralelo (frontend/backend) el calendario se comprime ~40%.

---

## 0. Alcance de la Beta (qué entra y qué queda fuera)

**Entra en Beta cerrada:** señales ML con niveles operativos y gating de confianza,
S&P 500 + NASDAQ 100 + Forex major + top-20 crypto, jobs de inferencia cada 5 min,
SPA Next.js con auth, gráficos con TP/SL dibujados, AI Insights (explicabilidad),
paywall Free/Pro con Stripe (test mode), alertas Telegram/Email, modo demo comercial,
informe PDF de backtesting automático, guía MLOps.

**Fuera de Beta (post-beta):** ejecución real contra broker (solo paper trading),
app móvil nativa, multi-idioma, backtesting custom por usuario, tier enterprise.

---

## 1. Estado actual: lo que ya existe y se reutiliza

El refactor anterior dejó una base aprovechable — el plan NO arranca de cero:

| Componente PRD | Estado actual |
|---|---|
| Señales sin neutrales (4 categorías) | ✅ `src/ml/signal_engine.py` (COMPRA_FUERTE/COMPRA/VENTA/VENTA_FUERTE) |
| Niveles entrada/TP1/TP2/SL por ATR+pivotes | ✅ `build_levels()` con R/R y sizing por riesgo fijo |
| Filtro de régimen macro (VIX/ATR/gaps) | ✅ `src/ml/regime.py` (normal/caution/suspended) |
| Calibración de probabilidades (isotónica) | ✅ `src/ml/calibration.py` + umbral OOF 0.525 |
| Screener S&P 500 + watchlists | ✅ `src/data/universe.py` + `src/ml/watchlist.py` |
| Paper trading con métricas | ✅ `src/trading/paper.py` (SQLite; se migra a PostgreSQL en S1) |
| Motor de backtesting con costos | ✅ `src/backtesting/engine.py` walk-forward (se reutiliza en S3) |
| Docker + guía de despliegue | ✅ `Dockerfile`, `docker-compose.yml`, `DEPLOY.md` |
| Tests + CI | ✅ 49 tests pytest, GitHub Actions en verde |
| Gráficos TradingView | ✅ Lightweight Charts v4 vendorizado local |

**Deuda detectada en la exploración (se corrige en S3):** el campo `tickers` del OOF
guardado (`models/advanced_oof.pkl`) tiene 49,490 entradas para 7,070 filas — el trainer
extiende la lista por fold con un índice desalineado (`df['ticker'].loc[X_te.index]`).
Las probabilidades/targets/fechas sí son consistentes. El backtest de expectativa
requiere reconstruir el par (fecha, ticker) por estructura de bloques o regenerar el
OOF tras corregir el trainer.

---

## 2. Stack tecnológico definitivo

| Capa | Elección | Por qué | Alternativa |
|---|---|---|---|
| Frontend | **Next.js 15 (App Router) + TypeScript + Tailwind + shadcn/ui** | SPA fluida, SSR donde suma, ecosistema Tailwind cumple el diseño `#0B0E14` | Vite+React puro (menos convención) |
| Estado/datos FE | **TanStack Query + Zustand** | cache + revalidación para datos en vivo | SWR |
| Gráficos | **TradingView Lightweight Charts v4** (ya vendorizado) | cero CDN, API conocida en el repo | Highcharts (licencia) |
| Backend | **FastAPI único** (Flask se jubila al final de S4) | async, OpenAPI automático, ya existe `scripts/serve_fastapi.py` | Flask+gunicorn (legacy) |
| ORM/DB | **SQLAlchemy 2 + Alembic sobre PostgreSQL 16** | señales/paper/watchlists/usuarios/subscripciones con integridad y migraciones | — |
| Cache | **Redis 7** | lecturas <1s (top-señales, heatmap) + broker de Celery | — |
| Jobs | **Celery + Celery Beat** (prod Docker); **APScheduler** en dev Windows | Beat programa el escaneo cada 5 min; Celery no corre nativo en Windows → dev local simplificado | arq/RQ |
| Auth | **Clerk** | integración Next.js en horas, Google+email, JWT para el API | Firebase Auth; Auth0 (enterprise) |
| Pagos | **Stripe Checkout + Billing** | webhooks estándar, test mode para Beta | Paddle |
| Alertas | **Telegram Bot API + Resend** (email) | Telegram es gratis e inmediato; Resend ~3k mails/mes gratis | SendGrid/AWS SES |
| Datos de mercado | **Alpaca Markets** (equities, IEX free, websocket) + **FMP** (forex, crypto, fundamentales) + Polygon opcional; **yfinance solo fallback dev** | cumple el requisito de sustitución; tiers gratis alcanzan para Beta | Polygon ($29/mes) si se exige feed consolidado |
| Explicabilidad | **SHAP TreeExplainer** sobre los miembros del ensamble + plantillas narrativas basadas en stats OOF reales | XGB/LGBM/RF son SHAP-friendly; el VotingClassifier soft promedia, así que el SHAP del ensamble se aproxima por miembros | LIME |
| MLOps | **Reentrenamiento mensual** + monitores de drift (PSI sobre features, DA rodante) + retrenado disparado si drift | protocolo documentado en S7 | DVC (opcional) |
| Infra | **Docker Compose** (dev) · **Vercel** (front) · **Render/Fly/ECS** (API+worker+beat) · GitHub Actions (CI) | despliegue simple y barato en Beta | AWS ECS Fargate (ya documentado en DEPLOY.md) |

---

## 3. Arquitectura objetivo

```
                        ┌──────────────────────────────┐
  Next.js (Vercel) ────▶│  FastAPI (Render/ECS)        │──▶ PostgreSQL (Neon/RDS)
   Clerk JWT ───────────▶│   /api/signals /markets      │──▶ Redis (cache <1s)
   Stripe webhook ──────▶│   /api/account /alerts       │
                        └──────────┬───────────────────┘
                                   │ encola
                        ┌──────────▼───────────────────┐
                        │  Celery Beat (cada 5 min)     │
                        │  workers: scan → inferencia   │──▶ Alpaca / FMP
                        │  → gating confianza+régimen   │    (yfinance fallback)
                        │  → upsert señales validadas   │
                        └──────────┬───────────────────┘
                                   │ señal FUERTE / top-N
                        ┌──────────▼───────────────────┐
                        │  Alertas: Telegram + Email    │
                        └──────────────────────────────┘
```

---

## 4. Cronograma: 7 sprints de 1 semana → Beta en ~7 semanas

| Sprint | Entregable | Criterios de aceptación |
|---|---|---|
| **S1 — Backend SaaS** (sem 1) ✅ **COMPLETADO (2026-09-29)** | FastAPI como única API (`src/api/fastapi_app.py`); PostgreSQL 16 + Alembic (`alembic_migrations`, migración inicial probada en SQLite y PG real); migración de paper trading/watchlists/signals a SQLAlchemy con importación automática del legado; Redis para lecturas calientes (`src/cache.py`, fallback memoria con TTL) | `docker compose up` levanta api+postgres+redis con healthchecks y DATABASE_URL cableada; migraciones reproducibles (`alembic upgrade head`); `/health` expone DB+cache+modelo; suite 70 tests en verde (11 nuevos httpx sobre la API); CI con servicios postgres/redis — **success en 04c3411** |
| **S2 — Datos + Jobs** (sem 2) ✅ **COMPLETADO (2026-09-29; claves Alpaca/FMP por configurar — mientras, fallback yfinance)** | Conector Alpaca (equities, REST v2) + FMP (forex/crypto) vía httpx con degradación automática a yfinance (`src/data/providers.py`); universos sp500/ndx100/fx_major/crypto20 (`src/data/universes.py`); scheduler APScheduler cada 5 min con cola priorizada (watchlists primero), upsert en `signal_cache` + espejo Redis (`src/jobs/scheduler.py`, `src/ml/signal_store.py`); API `/api/signals-cache` sirve el espejo sin inferencia al vuelo | ciclo real validado: 8 señales en 4.0s (yfinance); cobertura actual 590 símbolos (sp500 500 + ndx100 60 + fx 10 + crypto 20); con claves de pago sube la calidad de datos, no cambia el pipeline; tests con proveedores falsos en verde |
| **S3 — Modelo v2 + Informe PDF** (sem 3) ✅ **COMPLETADO (2026-09-27)** | Bug `tickers` del trainer corregido (mapeo posicional); reentrenamiento con `class_weight`/`scale_pos_weight` (sin SMOTE); R/R estructural mínimo 1:2 garantizado en `build_levels`; simulador de expectativa `src/ml/expectancy.py` con ejecución walk (SL-first, costos 10pb, portafolio realista); **informe PDF automático** `scripts/generate_report.py` | **PF neto 1.505, Sharpe 1.48, MaxDD -15.9%, 246 ops** (largo-solo, p≥0.50, VIX≤30, 1 pos/ticker, máx 3 concurrentes, R/R 2:1). Edge dependiente de régimen (2023-2025 positivo; 2022 negativo) documentado en el PDF. Advertencia: gating a 0.65 NO es óptimo — la confianza del modelo no es precisión; el edge real vive en el filtrado por VIX y en excluir cortos |
| **S4 — Frontend core** (sem 4) ✅ **COMPLETADO (2026-09-29; Clerk scaffold listo, claves por configurar; jubilación de Flask pendiente de paridad final)** | Next.js 15 dark premium (#0B0E14 + dorado institucional, glassmorphism): dashboard con grid de señales y KPIs, `/screener` con filtros (señal, confianza, lado), `/asset/[ticker]` con TradingView Lightweight Charts y Entry/TP1/TP2/SL dibujados sobre las velas; auth Clerk condicional (`clerkEnabled`, middleware protege /screener solo si hay claves); **Docker integral**: compose de 5 servicios (api+frontend+scheduler+postgres+redis) con entrypoint que aplica migraciones Alembic | `/api/signal/{ticker}` 200 con sector/nombre/RSI/regime plano; build de producción verde; suite 92 tests; stack Docker verificado E2E: health ok, migraciones aplicadas, señal persistida en PG desde contenedor |
| **S5 — AI Insights + Backtest UI** (sem 5) 🔶 **EN CURSO (2026-09-29: AI Insights + alertas LISTOS; falta vista /backtest)** | AI Insights TreeSHAP real (`src/ml/insights.py`, `pred_contrib` de xgboost+lightgbm del VotingClassifier → `/api/insights/{ticker}` con top-factores y presiones buy/sell, sin cifras inventadas); tarjeta InsightsCard con barras bidireccionales en la ficha; **alertas** Telegram + Resend con dedup Redis (TTL 2d) para señales FUERTE ejecutables, hook best-effort en el scheduler (9 tests nuevos) | vista /backtest con equity curve vs benchmark pendiente; alerta E2E con usuario real pendiente de claves Telegram/Resend |
| **S6 — Monetización** (sem 6) ✅ **COMPLETADO (2026-10-02; Stripe por configurar — mientras, checkout responde 503 y el modo demo asigna free)** | Stripe Checkout + Billing condicional (`src/billing/stripe_gateway.py`): session de Checkout Pro (price fijo o inline), webhook con firma verificada que mantiene `subscriptions` + `users.tier` (checkout.session.completed, subscription.updated/deleted); endpoints `/api/billing/{config,checkout,webhook,status}`; paywalls: `/api/signal` 402 fuera de demo (AAPL/TSLA/BTC-USD) para free, `/api/signals-cache` filtrado por tier, blur + muro en `/asset`, universos bloqueados en `/screener`; alertas Telegram/Email exclusivas Pro; **Flask jubilado** (eliminadas app/routes/templates/static/app_ml); `/pricing` con comparativa y sesión local por email (Clerk pendiente de claves) | paywall verificado E2E en contenedores: NVDA free → 402, demo → 200, checkout sin claves → 503, webhook sin firma → 401; suite 105 tests en el contenedor Linux; CI success en f258cb2 |
| **S7 — Beta cerrada** (sem 7) | Modo demo comercial (usuario demo con datos precargados de alto impacto visual); guía MLOps de retrenamiento; despliegue producción + dominio; hardening (rate-limit, CORS, secretos); invitaciones a grupo cerrado | 20 usuarios invitados; señales generándose en horario de mercado; runbook de reentrenamiento mensual; checklist §8 completo |

**Nota sobre alertas "+85% de confianza" del PRD:** con probabilidades calibradas,
la masa sobre 0.85 es minúscula (alertas rarísimas). Se implementa **top-N diario** de
señales FUERTE (configurable), que es lo comercialmente útil. Ver §6.

---

## 5. Trazabilidad PRD → sprints

| Requisito PRD | Sprint |
|---|---|
| 1. Balanceo de clases, anti-sesgo alcista, R/R 1:2, gating >65% | S3 |
| 2. Sustituir yfinance (Alpaca/FMP), S&P500+NDX100+FX+crypto, Celery+Redis+PostgreSQL, web <1s | S1, S2 |
| 3. Next.js, dark mode premium, AI Insights, gráficos con TP/SL | S4, S5 |
| 4. Auth, Stripe Free/Pro, alertas Telegram/Email | S4 (auth), S6 |
| 5. Docker AWS/Render, PDF de backtesting, guía MLOps | S1/S7 (infra), S3 (PDF), S7 (MLOps) |

---

## 6. Riesgos y notas honestas (matemática de la expectativa)

1. **"Confianza >65%" ≠ precisión 65%.** En el OOF histórico, la banda p≥0.65 acertó
   ~52.4% (607 casos). Ningún umbral convierte la confianza en acierto. El gating a 0.65
   es correcto como **filtro de exposición**; el edge comercial viene del R/R:
   con TP2 a 2R, un acierto de ~52% rinde PF bruto ≈ 2.2 y **PF>1.3 solo exige acierto
   ≥ ~39.4%** — ese es el número que S3 debe validar con ejecución realista de niveles.
   El marketing del SaaS debe vender expectativa (R/R + filtrado), no "65% de acierto".
2. **Alertas a +85% de confianza:** casi nunca se dispararían con probs calibradas.
   Se sustituye por top-N diario (S6), parametrizable por tier.
3. **SMOTE/oversampling en series temporales:** los sintéticos pueden cruzar fronteras
   temporales y filtrar información. Se usa `class_weight` en los modelos + purga
   temporal (ya implementada en el CV). Si el desbalance persiste, se ajusta el umbral,
   no los datos.
4. **Celery no corre nativo en Windows:** dev local con APScheduler sobre el mismo
   código de tareas; Celery completo en Docker (prod). Sin doble implementación: las
   tareas son funciones puras que ambos runners invocan.
5. **Rate limits:** escanear 500+ tickers cada 5 min exige cola priorizada y caching de
   barras (yfinance no sobrevive a ese ritmo → migración Alpaca es prerequisito, no opción).
6. **Bug del OOF (`tickers` 49,490 vs 7,070):** corregido y OOF regenerado en S3 antes
   de cualquier número que entre al PDF. El informe PDF será auditado contra la matriz
   OOF; nada de cifras de demostración.
7. **Cumplimiento:** toda la UI mantiene el disclaimer ("no es asesoramiento
   financiero"); el modo demo marca los datos como simulados.

---

## 7. Costos de infraestructura en Beta

| Servicio | Plan | Costo/mes |
|---|---|---|
| Vercel (frontend) | Hobby | $0 |
| Neon/Render (PostgreSQL) | Free | $0 |
| Upstash/Render (Redis) | Free | $0 |
| Render/Fly (API + worker + beat) | Free–Starter | $0–7 |
| Alpaca (IEX real-time) | Free | $0 |
| FMP (forex/crypto) | Starter | $0–14 |
| Resend (email) | Free | $0 |
| Dominio | — | ~$1 |
| **Total** | | **~$0–22/mes** |

---

## 8. Checklist "Beta lista"

- [ ] Front y backend desplegados con dominio propio y HTTPS
- [ ] Registro Google/email funcionando (Clerk) con tiers Free/Pro (Stripe test mode)
- [ ] Señales generándose cada 5 min en horario de mercado y persistidas en PostgreSQL
- [ ] Lecturas de la web p95 < 1s (verificado con cache Redis)
- [ ] AI Insights visible en cada señal, trazable a SHAP + stats OOF
- [ ] Informe PDF regenerado con el modelo v2 (PF neto > 1.3 documentado)
- [ ] Alerta Telegram/Email E2E entregada
- [ ] Modo demo precargado y revisado visualmente
- [ ] Guía MLOps (retrenamiento mensual + drift) en el repo
- [ ] 49+ tests en verde en CI y humo de producción automatizado
