"""Scheduler de escaneos en background (Sprint 2).

Ciclo cada 5 min (APScheduler 3.x; en prod Docker puede sustituirse por
Celery Beat sin cambiar la logica del job):

1. Resuelve los universos y forma la cola priorizada: watchlists de usuario
   y top-liquidez primero, resto del universo detras.
2. Escanea con el ensemble calibrado (mismo pipeline que el screener).
3. Persiste con upsert en ``signal_cache`` (fuente de verdad) y refresca el
   espejo de cache que consumen las lecturas de la API (<1s).

El scheduler corre como proceso independiente (``scripts/run_scheduler.py``)
o embebido en el API (SCHEDULER_EMBEDDED=1) para el docker-compose simple.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from ..cache import cache_set
from ..data.collector import DataCollector
from ..data.processor import TechnicalProcessor
from ..data.providers import get_market_data
from ..data.universes import resolve_any
from ..db.models import Watchlist
from ..db.session import get_session_factory
from ..ml.regime import detect_regime
from ..ml.signal_engine import build_signal
from ..ml.signal_store import REDIS_KEY_LATEST, LATEST_TTL, SignalStore
from ..utils.logger import get_logger

logger = get_logger(__name__)

SCAN_INTERVAL_MIN = 5
MARKET_CONTEXT = ("SPY", "^VIX")

_state: Dict[str, Any] = {
    "running": False, "last_cycle_at": None, "last_cycle_seconds": None,
    "last_written": 0, "cycles": 0, "errors": 0, "last_error": None,
}
_state_lock = threading.Lock()

# senales espejo listas para la API: /api/signals-cache lee esto sin tocar DB
_latest_mirror: List[Dict[str, Any]] = []


def scheduler_status() -> Dict[str, Any]:
    """estado del scheduler para /health."""
    with _state_lock:
        st = dict(_state)
    st["mirror_signals"] = len(_latest_mirror)
    return st


def market_open_now() -> bool:
    """ True L-V 13:30-20:00 UTC (9:30-16:00 ET). Dev y FX/crypto 24/5: se
    escanea igual fuera de horario, solo cambia la etiqueta."""
    now = time.gmtime()
    wd = now.tm_wday  # 0=Monday
    minutes = now.tm_hour * 60 + now.tm_min
    return wd < 5 and 13*60+30 <= minutes < 20*60


def _watchlist_tickers() -> List[str]:
    """tickers de todas las watchlists de usuario (cola prioritaria)."""
    try:
        with get_session_factory()() as s:
            rows = s.query(Watchlist).all()
            tickers: List[str] = []
            for w in rows:
                for t in (w.tickers or []):
                    if t and t.upper() not in tickers:
                        tickers.append(t.upper())
            return tickers
    except Exception as exc:
        logger.warning(f"watchlists no disponibles para la cola priorizada: {exc}")
        return []


def _priority_queue(universe: str) -> List[str]:
    """watchlists primero, resto del universo detras (cola priorizada)."""
    universe_tickers = resolve_any(universe)
    priority = _watchlist_tickers()
    ordered = [t for t in priority if t in set(universe_tickers)] \
        + [t for t in universe_tickers if t not in set(priority)]
    return ordered


def _scan_one(
    ticker: str, predictor, processor: TechnicalProcessor,
    spy_df, vix_df, vix_series, period: str,
) -> Optional[Dict[str, Any]]:
    """senal validada de un ticker (mismo pipeline que el screener)."""
    try:
        # proveedor primario (Alpaca/FMP) con fallback yfinance + cache CSV
        df = get_market_data().fetch_daily(ticker, period)
        if df is None or df.empty:
            df = DataCollector().download_ticker(ticker, period)
        if df is None or df.empty:
            return None
        df = processor.process_all_indicators(df, spy_df=spy_df, vix_df=vix_df)
        X = df[predictor.feature_cols].tail(1)
        if X.isna().any(axis=1).iloc[0]:
            return None
        prob_up = predictor._raw_probability_up(X)
        regime = detect_regime(df, vix_series=vix_series)
        sig = build_signal(ticker, df, prob_up, as_of=str(df.index[-1].date()), regime=regime)
        d = sig.to_dict()
        d["volume_ratio"] = round(float(df["volumen_ratio"].iloc[-1]), 2) \
            if "volumen_ratio" in df.columns else None
        d["rsi"] = round(float(df["rsi"].iloc[-1]), 1) if "rsi" in df.columns else None
        return d
    except Exception as exc:
        logger.debug(f"scan {ticker}: {exc}")
        return None


def run_cycle(universe: str = "sp500", period: str = "3mo", max_workers: int = 8,
              limit: Optional[int] = None) -> Dict[str, Any]:
    """un ciclo completo: escaneo + persistencia. Devuelve resumen."""
    t0 = time.time()
    tickers = _priority_queue(universe)
    if limit:
        tickers = tickers[:limit]
    if not tickers:
        with _state_lock:
            _state.update(last_cycle_at=time.time(), last_cycle_seconds=time.time() - t0,
                          last_written=0, cycles=_state["cycles"] + 1)
        return {"written": 0, "n_tickers": 0, "seconds": round(time.time() - t0, 1)}

    collector = DataCollector()
    processor = TechnicalProcessor()
    spy = get_market_data().fetch_daily("SPY", period)
    if spy is None:
        spy = collector.download_ticker("SPY", period)
    vix = get_market_data().fetch_daily("^VIX", period)
    if vix is None:
        vix = collector.download_ticker("^VIX", period)
    vix_series = vix["Close"] if vix is not None else None

    # predictor: mismo ensemble calibrado que la API
    from ..ml.advanced_predictor import AdvancedPredictor
    predictor = AdvancedPredictor()

    store = SignalStore()
    signals: List[Dict[str, Any]] = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(
            lambda t: _scan_one(t, predictor, processor, spy, vix, vix_series, period),
            tickers,
        ):
            if r is not None:
                signals.append(r)

    written = store.upsert_batch(signals, universe=universe)

    # espejo listo para la API (ordenado por conviccion) + cache Redis
    signals.sort(key=lambda r: r.get("conviction_pct", 0), reverse=True)
    global _latest_mirror
    _latest_mirror = signals
    top = [_clean_signal_public(s) for s in signals[:50]]
    cache_set(REDIS_KEY_LATEST, top, ttl=LATEST_TTL)

    seconds = time.time() - t0
    with _state_lock:
        _state.update(running=False, last_cycle_at=time.time(),
                      last_cycle_seconds=round(seconds, 1),
                      last_written=written, cycles=_state["cycles"] + 1)
    logger.info(f"ciclo {universe}: {written} senales persistidas "
                f"({len(signals)} validas de {len(tickers)} tickers) en {seconds:.1f}s")
    return {"written": written, "n_valid": len(signals),
            "n_tickers": len(tickers), "seconds": round(seconds, 1)}


def _clean_signal_public(d: Dict[str, Any]) -> Dict[str, Any]:
    """dict publico para el espejo/cache (sin arrays grandes)."""
    keep = ("ticker", "as_of", "trade_date", "signal", "side", "prob_up",
            "confidence_pct", "conviction_pct", "entry", "stop_loss",
            "risk_reward", "position_size_pct", "regime", "trading_allowed",
            "price", "atr_pct", "volume_ratio", "rsi", "name", "sector")
    return {k: d.get(k) for k in keep if k in d}


def start_scheduler(universe: str = None, period: str = "3mo", max_workers: int = 8,
                    run_immediately: bool = True) -> "BackgroundScheduler":
    """arranca el scheduler APScheduler (intervalo fijo de 5 min)."""
    universe = universe or os.environ.get("SCHEDULER_UNIVERSE", "sp500")
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        run_cycle, "interval", minutes=SCAN_INTERVAL_MIN,
        kwargs={"universe": universe, "period": period, "max_workers": max_workers},
        id="scan-cycle", max_instances=1, coalesce=True,
        misfire_grace_time=120,
    )
    if run_immediately:
        th = threading.Thread(target=run_cycle, kwargs={"universe": universe, "period": period,
                                                        "max_workers": max_workers},
                              daemon=True, name="scan-first-cycle")
        th.start()
    sched.start()
    logger.info(f"Scheduler arrancado: ciclo cada {SCAN_INTERVAL_MIN} min "
                f"(universo={universe}, inmediato={run_immediately})")
    return sched
