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
| **S1 — Backend SaaS** (sem 1) | FastAPI como única API; PostgreSQL + Alembic; migración de paper trading/watchlists/signals desde SQLite/JSON; Redis para lecturas calientes | `docker compose up` levanta api+db+redis; migraciones reproducibles; endpoints de lectura p95 < 1s; suite de tests actualizada en verde |
| **S2 — Datos + Jobs** (sem 2) | Conector Alpaca (equities) + FMP (forex/crypto) con fallback yfinance; universos S&P500 + NASDAQ100 + 10 pares FX + top-20 crypto; Celery Beat escaneo cada 5 min en horario de mercado con cola priorizada (watchlists y top-liquidez primero); señales validadas persistidas con upsert | ≥600 símbolos cobertura; ciclo de 5 min sin exceder rate limits (Alpaca 200 req/min); la web nunca calcula inferencia al vuelo |
| **S3 — Modelo v2 + Informe PDF** (sem 3) ✅ **COMPLETADO (2026-09-27)** | Bug `tickers` del trainer corregido (mapeo posicional); reentrenamiento con `class_weight`/`scale_pos_weight` (sin SMOTE); R/R estructural mínimo 1:2 garantizado en `build_levels`; simulador de expectativa `src/ml/expectancy.py` con ejecución walk (SL-first, costos 10pb, portafolio realista); **informe PDF automático** `scripts/generate_report.py` | **PF neto 1.505, Sharpe 1.48, MaxDD -15.9%, 246 ops** (largo-solo, p≥0.50, VIX≤30, 1 pos/ticker, máx 3 concurrentes, R/R 2:1). Edge dependiente de régimen (2023-2025 positivo; 2022 negativo) documentado en el PDF. Advertencia: gating a 0.65 NO es óptimo — la confianza del modelo no es precisión; el edge real vive en el filtrado por VIX y en excluir cortos |
| **S4 — Frontend core** (sem 4) | Next.js + Clerk (Google/email); dashboard (marquesina de índices, KPIs del modelo, grid de señales de alta confianza), screener con filtros (señal, confianza, ATR%, volumen), detalle de activo con gráfico y TP/SL/SL dibujados | Paridad funcional con el dashboard Flask actual en SPA; Lighthouse ≥ 85; responsive |
| **S5 — AI Insights + Backtest UI** (sem 5) | Panel de explicabilidad SHAP (top-features + narrativa plantillada con stats OOF reales del patrón, p.ej. "compresión de volatilidad + MACD alcista: X% histórico en este activo"); vista /backtest con equity curve vs benchmark y métricas | Cada señal muestra 3-5 razones audibles trazables a features; sin cifras inventadas (los % salen de la matriz OOF) |
| **S6 — Monetización** (sem 6) | Stripe Checkout+Billing: Free (3 activos, señales retrasadas 24h) / Pro (tiempo real, todo el mercado, alertas, webhooks); alertas Telegram/Email para top-N señales FUERTE diarias | Paywall enforced en API y UI; webhook de suscripción activa/revoca en PostgreSQL; alerta E2E entregada en <60s |
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
