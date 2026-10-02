"""API unificada FastAPI del SaaS ML_Bolsa.

Puerta de entrada JSON unica del producto: senales, screener, paper trading,
watchlists, transparencia (expectativa), monetizacion y salud del sistema.
Las vistas viven en frontend/ (Next.js); el Flask heredado fue jubilado en
el Sprint 6.

Uso:
    uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8010
    # o: python scripts/serve_fastapi.py
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.cache import cache_delete, cache_get, cache_health, cache_set
from src.data.collector import DataCollector
from src.data.processor import TechnicalProcessor
from src.data.providers import get_market_data
from src.data.universe import get_name_map, get_sector_map, get_universe
from src.data.universes import universe_info as saas_universe_info
from src.db.session import healthcheck as db_healthcheck
from src.db.session import init_db
from src.jobs.scheduler import scheduler_status
from src.ml.signal_store import SignalStore
from src.ml.advanced_predictor import AdvancedPredictor
from src.ml.expectancy import attach_market_data, load_oof_dataset, simulate_expectancy
from src.ml.regime import detect_regime
from src.ml.signal_engine import SIDE_BUY, SIDE_SELL, build_signal, find_pivot_levels
from src.ml.screener import apply_filters, last_results, scan_status, start_scan
from src.ml.watchlist import WatchlistStore
from src.trading.paper import PaperTrader
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Config validada del sistema (Sprint 3): PF 1.505, WR 46.7%, E[R] +0.268R.
EXPECTANCY_CONFIG = dict(
    prob_min=0.50, rr_multiple=2.0, vix_max=30.0, allow_short=False,
    one_position_per_ticker=True, max_concurrent=3, max_hold=20,
    risk_per_trade=0.01,
)
EXPECTANCY_TTL_SECONDS = 6 * 3600
EXPECTANCY_LOCK = threading.Lock()
EXPECTANCY_KEY = "expectancy:tech7_oof"
TOP_SIGNALS_KEY = "top_signals"
TOP_SIGNALS_TTL_SECONDS = 60

_state: Dict[str, Any] = {"predictor": None, "scheduler": None}
_watchlists: Optional[WatchlistStore] = None
_paper: Optional[PaperTrader] = None
_signal_store: Optional[SignalStore] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _watchlists, _paper
    init_db()
    _watchlists = WatchlistStore()
    _paper = PaperTrader()
    try:
        _state["predictor"] = AdvancedPredictor()
        logger.info("FastAPI: modelo ensemble cargado")
    except Exception as exc:  # CI/entorno sin modelos: la API sigue degradada
        _state["predictor"] = None
        logger.warning(f"FastAPI: sin modelo ({exc}); endpoints de inferencia 503")
    # scheduler embebido (Sprint 2): SCHEDULER_EMBEDDED=1 para el compose simple
    if os.environ.get("SCHEDULER_EMBEDDED") == "1":
        try:
            from src.jobs.scheduler import start_scheduler

            _state["scheduler"] = start_scheduler(run_immediately=True)
            logger.info("FastAPI: scheduler embebido arrancado (ciclo 5 min)")
        except Exception as exc:
            logger.warning(f"FastAPI: scheduler embebido no arranco: {exc}")
    yield


app = FastAPI(
    title="ML_Bolsa SaaS API",
    description=(
        "Senales operables del ensemble calibrado: screener, paper trading, "
        "watchlists y transparencia (expectativa real del sistema)."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# CORS: el frontend (Next.js) consume la API desde el navegador.
# CORS_ORIGINS lista separada por comas; por defecto localhost:3000.
_cors_origins = [
    o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _clean_nan(obj):
    """NaN/inf -> None recursivamente (JSON valido)."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def _get_watchlists() -> WatchlistStore:
    global _watchlists
    if _watchlists is None:  # fallback si no corrio el lifespan (tests)
        init_db()
        _watchlists = WatchlistStore()
    return _watchlists


def _get_paper() -> PaperTrader:
    global _paper
    if _paper is None:
        init_db()
        _paper = PaperTrader()
    return _paper


def _get_signal_store() -> SignalStore:
    global _signal_store
    if _signal_store is None:
        _signal_store = SignalStore()
    return _signal_store


def _predictor() -> AdvancedPredictor:
    p = _state["predictor"]
    if p is None:
        raise HTTPException(503, "modelo no disponible (entrena: scripts/train_advanced.py)")
    return p


def _download_with_context(ticker: str, period: str):
    collector, processor = DataCollector(), TechnicalProcessor()
    df = collector.download_ticker(ticker, period)
    if df is None or df.empty:
        return None
    spy = collector.download_ticker("SPY", period)
    vix = collector.download_ticker("^VIX", period)
    df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix)
    return df, vix


def _load_oof():
    path = os.path.join(Config().models_dir, "advanced_oof.pkl")
    if not os.path.exists(path):
        return None
    data = joblib.load(path)
    return data if data.get("dates") else None


def _load_sp500_study() -> Optional[Dict]:
    """estudio de generalizacion S&P 500 si ya fue ejecutado."""
    p = Path("reports/sp500_expectancy.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compute_expectancy() -> Dict:
    """simula el sistema sobre el OOF del modelo en produccion (7 tech).

    Costoso (~5-10s): cache en Redis/memoria TTL 6h + lock por proceso
    (lock + double-check).
    """
    cached = cache_get(EXPECTANCY_KEY)
    if cached is not None:
        return cached
    with EXPECTANCY_LOCK:
        cached = cache_get(EXPECTANCY_KEY)  # doble chequeo tras el lock
        if cached is not None:
            return cached
        now = datetime.now()
        df = load_oof_dataset()
        df = attach_market_data(df)
        res = simulate_expectancy(df, **EXPECTANCY_CONFIG)
        per_year = []
        for y in sorted(df["date"].str[:4].unique()):
            sub = df[df["date"].str[:4] == y]
            r = simulate_expectancy(sub, **EXPECTANCY_CONFIG)
            per_year.append({
                "year": y, "n_trades": r.n_trades,
                "win_rate": round(r.win_rate, 4),
                "profit_factor": round(r.profit_factor, 3),
                "expectancy_r": round(r.expectancy_r, 4),
            })
        # benchmark SPY en el mismo periodo
        bench: Dict[str, Any] = {}
        spy_path = Path(Config().raw_data_dir) / "SPY_raw.csv"
        if spy_path.exists():
            spy = pd.read_csv(spy_path)
            spy[spy.columns[0]] = pd.to_datetime(spy[spy.columns[0]], utc=True).dt.strftime("%Y-%m-%d")
            spy = spy.drop_duplicates(subset=spy.columns[0], keep="last") \
                .set_index(spy.columns[0]).sort_index() \
                .loc[df["date"].min():df["date"].max()]
            if len(spy) > 50:
                norm = spy["Close"] / spy["Close"].iloc[0]
                rets = spy["Close"].pct_change().dropna()
                years = max(
                    (pd.Timestamp(df["date"].max()) - pd.Timestamp(df["date"].min())).days / 365.25,
                    1e-9,
                )
                peak = np.maximum.accumulate(norm)
                bench = {
                    "total_return_pct": round(float((norm.iloc[-1] - 1) * 100), 1),
                    "cagr_pct": round(float((norm.iloc[-1] ** (1 / years) - 1) * 100), 1),
                    "max_drawdown_pct": round(float(((norm - peak) / peak).min() * 100), 1),
                    "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(252)), 2),
                }
        data = {
            "generated_at": now.isoformat(),
            "config": EXPECTANCY_CONFIG,
            "tech7_oof": {
                "n_trades": res.n_trades,
                "win_rate": round(res.win_rate, 4),
                "profit_factor": round(res.profit_factor, 3),
                "expectancy_r": round(res.expectancy_r, 4),
                "sharpe": round(res.sharpe, 2),
                "max_drawdown_pct": round(res.max_drawdown_pct, 2),
                "avg_hold_days": round(res.avg_hold_days, 2),
                "per_year": per_year,
            },
            "benchmark_spy": bench,
            "sp500_study": _load_sp500_study(),
            "methodology": (
                "Probabilidades out-of-sample del walk-forward purgado; entrada en apertura "
                "de t+1, SL 1.5 ATR (1-5%), TP 2x riesgo, holding max 20 sesiones, costos 10pb, "
                "criterio SL-first. Filtros: largo-solo, VIX<=30, 1 posicion por ticker, "
                "max 3 concurrentes. La confianza del modelo NO es la precision."
            ),
        }
        cache_set(EXPECTANCY_KEY, data, ttl=EXPECTANCY_TTL_SECONDS)
        return data


# ----------------------------------------------------------------------
# esquemas de request
# ----------------------------------------------------------------------

class WatchlistCreate(BaseModel):
    name: str
    tickers: Optional[list[str]] = None


class WatchlistTickers(BaseModel):
    tickers: list[str]


class ScanRequest(BaseModel):
    universe: str = "mega"
    period: str = "3mo"
    force: bool = False


# ----------------------------------------------------------------------
# health + meta
# ----------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        db_ok = bool(db_healthcheck())
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "cache": cache_health(),
        "model_loaded": _state["predictor"] is not None,
        "providers": get_market_data().providers_status(),
        "scheduler": scheduler_status(),
        "scan": scan_status(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/meta")
def api_meta():
    p = _predictor()
    meta_path = os.path.join(Config().models_dir, "advanced_metadata.pkl")
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    calib_threshold = None
    calib_path = os.path.join(Config().models_dir, "calibration.pkl")
    if os.path.exists(calib_path):
        calib_threshold = joblib.load(calib_path).get("selected_threshold")
    return _clean_nan({
        "model_type": "advanced",
        "calibrated": p.calibrator is not None,
        "signal_threshold": calib_threshold if calib_threshold is not None
        else p.confidence_threshold,
        "trained_at": meta.get("trained_at"),
        "walk_forward": meta.get("walk_forward_metrics", {}),
        "universes": saas_universe_info(),
    })


@app.get("/api/model-performance")
def api_model_performance():
    oof = _load_oof()
    if oof is None:
        return {"available": False, "reason": "reentrena para generar OOF con fechas"}
    probs = np.asarray(oof["probabilities"])
    targets = np.asarray(oof["targets"])
    edges = [0.0, 0.35, 0.45, 0.55, 0.65, 1.0]
    labels = ["<=35%", "35-45%", "45-55%", "55-65%", ">=65%"]
    bins = pd.cut(probs, bins=edges, labels=labels, include_lowest=True)
    bands = []
    for label in labels:
        mask = bins == label
        n = int(mask.sum())
        if n:
            band = targets[mask]
            acc = float((band == 0).mean()) if label in ("<=35%", "35-45%") else float(band.mean())
            bands.append({"band": label, "n": n, "accuracy": round(acc, 4),
                          "share": round(n / len(probs), 4)})
    overall = float(((probs >= 0.5).astype(int) == targets).mean())
    meta_path = os.path.join(Config().models_dir, "advanced_metadata.pkl")
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    return _clean_nan({
        "available": True,
        "n_samples": int(len(probs)),
        "overall_accuracy": round(overall, 4),
        "bands": bands,
        "range": [oof["dates"][0], oof["dates"][-1]],
        "trained_at": meta.get("trained_at"),
        "walk_forward": meta.get("walk_forward_metrics"),
    })


@app.get("/api/universe/info")
def api_universe_info():
    try:
        tickers = get_universe(limit=500)
        return {
            "total": len(tickers),
            "sample": tickers[:25],
            "sectors": sorted(set(get_sector_map().values())),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ----------------------------------------------------------------------
# senal individual + chart
# ----------------------------------------------------------------------

@app.get("/api/signal/{ticker}")
def api_signal(ticker: str, period: str = "1y", email: Optional[str] = None):
    ticker = ticker.upper()
    # paywall Sprint 6 ANTES de cargar modelo/datos (determinista y barato):
    # sin email = beta local abierta; con email, tier free solo accede a los
    # activos demo
    if email:
        from src.billing import stripe_gateway as sg

        u = sg.get_or_create_user(email)
        if not sg.can_access_ticker(ticker, u.tier or sg.FREE_TIER):
            raise HTTPException(402, detail={
                "code": "upgrade_required",
                "message": f"Desbloquea el analisis completo de {ticker} con Pro",
                "ticker": ticker,
            })
    p = _predictor()
    try:
        res = _download_with_context(ticker, period)
        if res is None:
            raise HTTPException(404, f"sin datos para {ticker}")
        df, vix = res
        X = df[p.feature_cols].tail(1)
        if X.isna().any(axis=1).iloc[0]:
            raise HTTPException(422, "datos insuficientes para inferencia")

        prob_up = p._raw_probability_up(X)
        regime = detect_regime(df, vix_series=vix["Close"] if vix is not None else None)
        sig = build_signal(ticker, df, prob_up, as_of=str(df.index[-1].date()), regime=regime)
        payload = sig.to_dict()
        if isinstance(payload.get("regime"), dict):  # JSON canónico: status plano
            payload["regime"] = payload["regime"].get("status")
        payload["regime_detail"] = regime if isinstance(regime, dict) else None
        # contexto tecnico para la ficha (misma fuente que el screener)
        payload["rsi"] = round(float(df["rsi"].iloc[-1]), 1) if "rsi" in df.columns else None
        payload["volume_ratio"] = (round(float(df["volumen_ratio"].iloc[-1]), 2)
                                   if "volumen_ratio" in df.columns else None)
        payload["sector"] = get_sector_map().get(ticker, "Otros")
        payload["name"] = get_name_map().get(ticker, ticker)
        payload["levels_nature"] = "riesgo_determinista_no_modelo"

        # registro automatico en paper trading
        payload["paper_signal_id"] = _get_paper().record_signal(payload)
        return _clean_nan(payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"error en signal {ticker}: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/chart/{ticker}")
def api_chart(ticker: str, period: str = "2y", days: int = 260):
    ticker = ticker.upper()
    try:
        collector, processor = DataCollector(), TechnicalProcessor()
        df = collector.download_ticker(ticker, period)
        if df is None or df.empty:
            raise HTTPException(404, f"sin datos para {ticker}")
        spy = collector.download_ticker("SPY", period)
        vix = collector.download_ticker("^VIX", period)
        df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix).tail(days)

        out = pd.DataFrame(index=df.index.astype(str))
        out["time"] = [str(d.date()) for d in df.index]
        for c in ("open", "high", "low", "close"):
            out[c] = df[c.capitalize()].round(2).values
        out["volume"] = df["Volume"].values
        if len(df) > 210:
            out["ema20"] = df["Close"].ewm(span=20, adjust=False).mean().round(2).values
            out["ema50"] = df["Close"].ewm(span=50, adjust=False).mean().round(2).values
            out["ema200"] = df["Close"].ewm(span=200, adjust=False).mean().round(2).values
            bb_mid = df["Close"].rolling(20).mean()
            bb_std = df["Close"].rolling(20).std()
            out["bb_upper"] = (bb_mid + 2 * bb_std).round(2).values
            out["bb_lower"] = (bb_mid - 2 * bb_std).round(2).values
        out["rsi"] = df["rsi"].round(1).values if "rsi" in df.columns else None
        out["macd"] = df["macd"].round(3).values if "macd" in df.columns else None
        out["macd_signal"] = df["macd_signal"].round(3).values if "macd_signal" in df.columns else None

        all_sup, all_res = find_pivot_levels(df["High"], df["Low"])
        last_px = float(df["Close"].iloc[-1])
        supports = sorted(all_sup, key=lambda v: abs(v - last_px))[:4]
        resistances = sorted(all_res, key=lambda v: abs(v - last_px))[:4]

        oof = _load_oof()
        oof_signals = []
        if oof is not None:
            for d, t, pr, y in zip(oof["dates"], oof["tickers"],
                                   oof["probabilities"], oof["targets"]):
                if t == ticker:
                    oof_signals.append({
                        "date": d, "prob_up": round(float(pr), 4),
                        "signal": "COMPRA" if pr >= 0.5 else "VENTA",
                        "actual_up": int(y),
                        "correct": int((pr >= 0.5) == bool(y)),
                    })
        return _clean_nan({
            "ticker": ticker,
            "candles": out.to_dict("records"),
            "supports": sorted(round(s, 2) for s in supports),
            "resistances": sorted(round(r, 2) for r in resistances),
            "oof_signals": oof_signals,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"error en chart {ticker}: {e}")
        raise HTTPException(500, str(e))


# ----------------------------------------------------------------------
# screener (jobs en segundo plano)
# ----------------------------------------------------------------------

@app.post("/api/scan")
def api_scan(body: Optional[ScanRequest] = None):
    p = _predictor()
    b = body or ScanRequest()
    res = start_scan(p, universe=b.universe, period=b.period, force=b.force)
    cache_delete(TOP_SIGNALS_KEY)  # nuevo escaneo -> invalida el panel
    return {
        "job_id": res["job_id"],
        "started": res["started"],
        "status": res["status"],
        "n_results": len(res["results"]),
        "results": res["results"][:20] if res["results"] else [],
    }


@app.get("/api/scan/status")
def api_scan_status():
    st = scan_status()
    out: Dict[str, Any] = {"status": st, "n_results": len(last_results())}
    if not st.get("running") and last_results():
        out["preview"] = apply_filters(last_results(), limit=10)
    return _clean_nan(out)


@app.get("/api/screener")
def api_screener(
    side: Optional[str] = None,
    signal: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_conviction: Optional[float] = None,
    max_atr_pct: Optional[float] = None,
    min_volume_ratio: Optional[float] = None,
    sector: Optional[str] = None,
    watchlist: Optional[str] = None,
    only_tradable: bool = False,
    limit: Optional[int] = None,
):
    rows = last_results()
    if not rows:
        return _clean_nan(
            {"error": "no hay escaneo; lanza POST /api/scan", "status": scan_status()}
        )
    filtered = apply_filters(
        rows,
        side=side, signal=signal, min_confidence=min_confidence,
        min_conviction=min_conviction, max_atr_pct=max_atr_pct,
        min_volume_ratio=min_volume_ratio, sector=sector,
        watchlist=_get_watchlists().get(watchlist)["tickers"] if watchlist else None,
        only_tradable=only_tradable, limit=limit,
    )
    return _clean_nan({
        "status": scan_status(),
        "total_scanned": len(rows),
        "total_returned": len(filtered),
        "buy_count": sum(1 for r in rows if r.get("side") == SIDE_BUY),
        "sell_count": sum(1 for r in rows if r.get("side") == SIDE_SELL),
        "suspended_count": sum(1 for r in rows if not r.get("trading_allowed", True)),
        "results": filtered,
    })


@app.get("/api/top-signals")
def api_top_signals():
    """panel: senales activas de alta conviccion (cache 60s)."""
    cached = cache_get(TOP_SIGNALS_KEY)
    if cached is not None:
        return cached
    rows = last_results()
    if not rows:
        return {"available": False, "reason": "lanza un escaneo", "status": scan_status()}
    tradable = [r for r in rows if r.get("trading_allowed", True)]
    top = sorted(tradable, key=lambda r: r.get("conviction_pct", 0), reverse=True)[:10]
    payload = _clean_nan({
        "available": True,
        "signals": top,
        "buy_count": sum(1 for r in tradable if r.get("side") == SIDE_BUY),
        "sell_count": sum(1 for r in tradable if r.get("side") == SIDE_SELL),
    })
    cache_set(TOP_SIGNALS_KEY, payload, ttl=TOP_SIGNALS_TTL_SECONDS)
    return payload


@app.get("/api/signals-cache")
def api_signals_cache(limit: int = 50, only_tradable: bool = False,
                      side: Optional[str] = None, signal: Optional[str] = None,
                      min_confidence: Optional[float] = None,
                      email: Optional[str] = None):
    """senales del ultimo ciclo del scheduler (Redis/DB, respuesta <1s).

    signal: lista separada por comas (p.ej. 'COMPRA_FUERTE,COMPRA').
    Fuente: espejo en cache con TTL corto; la DB (signal_cache) es la fuente
    de verdad. No calcula inferencia al vuelo nunca.
    """
    rows = _get_signal_store().latest_signals(
        limit=limit, only_tradable=only_tradable, side=side,
        signals=[s for s in (signal or "").split(",") if s.strip()] or None,
        min_confidence=min_confidence,
    )
    # paywall Sprint 6: tier free solo ve los activos demo (identidad por email
    # mientras Clerk consume la API; sin email = beta local sin limites)
    tier = "pro"
    if email:
        from src.billing import stripe_gateway as sg

        tier = sg.get_or_create_user(email).tier or sg.FREE_TIER
        if tier != sg.PRO_TIER:
            rows = [r for r in rows
                    if str(r.get("ticker", "")).upper() in sg.FREE_TICKERS]
    return _clean_nan({
        "available": bool(rows),
        "n": len(rows),
        "signals": rows,
        "tier": tier,
        "scheduler": scheduler_status(),
    })


# ----------------------------------------------------------------------
# monetizacion (Sprint 6): Stripe checkout + webhooks + entitlements
# ----------------------------------------------------------------------

class CheckoutRequest(BaseModel):
    email: str
    display_name: str = ""


@app.get("/api/billing/config")
def api_billing_config():
    """config para /pricing: features por tier y si Stripe esta activo."""
    from src.billing import stripe_gateway as sg

    return {
        "stripe_configured": sg.stripe_configured(),
        "price_amount_cents": int(os.environ.get("STRIPE_PRICE_PRO_AMOUNT", "2900")),
        "free_tickers": sorted(sg.FREE_TICKERS),
        "features_free": sg.tier_limit(sg.FREE_TIER),
        "features_pro": sg.tier_limit(sg.PRO_TIER),
    }


@app.post("/api/billing/checkout")
def api_billing_checkout(body: CheckoutRequest):
    """crea la session de Checkout Pro (503 si Stripe no esta configurado)."""
    from src.billing import stripe_gateway as sg

    user = sg.get_or_create_user(body.email, body.display_name)
    try:
        session = sg.create_checkout_session(user)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"checkout_url": session["url"], "session_id": session["id"]}


@app.post("/api/billing/webhook")
async def api_billing_webhook(request: Request):
    """webhook de Stripe: firma obligatoria; mantiene users.tier al dia."""
    from src.billing import stripe_gateway as sg

    payload = await request.body()
    try:
        event = sg.verify_webhook(payload, request.headers.get("stripe-signature"))
    except PermissionError as e:
        raise HTTPException(401, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"evento invalido: {e}")
    return sg.apply_stripe_event(event)


@app.get("/api/billing/status")
def api_billing_status(email: str):
    """tier + limites del usuario (lo consume el frontend para los paywalls)."""
    from src.billing import stripe_gateway as sg

    user = sg.get_or_create_user(email)
    st = sg.user_status(user.id)
    st["limits"] = sg.tier_limit(st["tier"])
    st["stripe_configured"] = sg.stripe_configured()
    return st


# ----------------------------------------------------------------------
# paper trading (forward testing)
# ----------------------------------------------------------------------

@app.post("/api/paper/resolve")
def api_paper_resolve():
    def provider(t):
        return DataCollector().download_ticker(t, "1y")

    n = _get_paper().resolve_pending(provider)
    return _clean_nan({"resolved": n, "performance": _get_paper().performance()})


@app.get("/api/paper/performance")
def api_paper_performance():
    return _clean_nan(_get_paper().performance())


@app.get("/api/paper/signals")
def api_paper_signals(limit: int = 100):
    return _clean_nan({"signals": _get_paper().list_signals(limit=limit)})


# ----------------------------------------------------------------------
# watchlists
# ----------------------------------------------------------------------

@app.get("/api/watchlists")
def api_watchlists_list():
    return {"watchlists": _get_watchlists().list()}


@app.post("/api/watchlists", status_code=201)
def api_watchlists_create(body: WatchlistCreate):
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "name requerido")
    try:
        return _get_watchlists().create(name, body.tickers)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get("/api/watchlists/{name}")
def api_watchlist_get(name: str):
    wl = _get_watchlists().get(name)
    if not wl:
        raise HTTPException(404, "no existe")
    return wl


@app.delete("/api/watchlists/{name}")
def api_watchlist_delete(name: str):
    return {"deleted": _get_watchlists().delete(name)}


@app.post("/api/watchlists/{name}/tickers")
def api_watchlist_add(name: str, body: WatchlistTickers):
    tickers = [t for t in body.tickers if t.strip()]
    if not tickers:
        raise HTTPException(422, "tickers requerido")
    try:
        return _get_watchlists().add_tickers(name, tickers)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/watchlists/{name}/tickers")
def api_watchlist_remove(name: str, body: WatchlistTickers):
    tickers = [t for t in body.tickers if t.strip()]
    if not tickers:
        raise HTTPException(422, "tickers requerido")
    try:
        return _get_watchlists().remove_tickers(name, tickers)
    except ValueError as e:
        raise HTTPException(404, str(e))


# ----------------------------------------------------------------------
# transparencia: expectativa del sistema
# ----------------------------------------------------------------------

@app.get("/api/expectancy")
def api_expectancy():
    try:
        data = dict(_compute_expectancy())
        # el estudio S&P 500 se re-lee en cada request: se actualiza cuando
        # el script de estudio termina, sin esperar al TTL de la cache
        data["sp500_study"] = _load_sp500_study()
        return _clean_nan(data)
    except FileNotFoundError as e:
        raise HTTPException(503, f"datos OOF no disponibles: {e}")
    except Exception as e:
        logger.error(f"error en /api/expectancy: {e}")
        raise HTTPException(500, str(e))


# ----------------------------------------------------------------------
# compatibilidad
# ----------------------------------------------------------------------

@app.get("/api/infer/{ticker}")
def api_infer(ticker: str, period: str = "1y", with_levels: bool = True):
    """senal operable completa (compat con el esqueleto serve_fastapi)."""
    payload = api_signal(ticker, period)
    if not with_levels:
        for k in ("take_profits", "supports", "resistances", "notes"):
            payload.pop(k, None)
    return payload


@app.get("/api/insights/{ticker}")
def api_insights(ticker: str, period: str = "1y"):
    """AI Insights: por que de la senal via TreeSHAP (boosters del ensemble).

    Devuelve top factores con direccion (buy/sell), presion alcista/bajista y
    resumen en lenguaje natural. 503 si el ensemble no soporta contribuciones.
    """
    p = _predictor()
    ticker = ticker.upper()
    try:
        res = _download_with_context(ticker, period)
        if res is None:
            raise HTTPException(404, f"sin datos para {ticker}")
        df, _ = res
        from src.ml.insights import explain_signal

        insights = explain_signal(p, df)
        if insights is None:
            raise HTTPException(503, "explicabilidad no disponible para este modelo")
        return _clean_nan(insights)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"error en insights {ticker}: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/predict/{ticker}")
def api_predict(ticker: str):
    p = _predictor()
    try:
        resultado = p.predict_ticker(ticker.upper())
        if resultado is None or "error" in resultado:
            msg = resultado.get("error", "sin datos") if resultado else "sin datos"
            raise HTTPException(404, msg)
        resultado["model_type"] = "advanced"
        resultado["timestamp"] = datetime.now().isoformat()
        if "recommendation" not in resultado and "prediccion" in resultado:
            resultado["recommendation"] = resultado["prediccion"].upper()
        return _clean_nan(resultado)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
