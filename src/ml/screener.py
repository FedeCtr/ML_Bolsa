"""
Screener de mercado v2: escanea el universo con el ensemble calibrado.

Novedades (spec 3):
  - Universo extensible: mega-caps (rapido) o S&P 500 completo via
    data/universe.py (Wikipedia + fallback).
  - Escaneo en SEGUNDO PLANO con progreso consultable: 500 tickers tardan
    minutos; la API lanza el job y devuelve inmediatamente.
  - Contexto de mercado (SPY/VIX) descargado una vez por escaneo.
  - Regimen integrado por ticker (caution/suspended escalan sizing).
  - Cache TTL del ultimo escaneo completado.
"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ..data.collector import DataCollector
from ..data.processor import TechnicalProcessor
from ..data.universe import FALLBACK_TICKERS, get_name_map, get_sector_map
from ..utils.logger import get_logger
from .regime import detect_regime
from .signal_engine import SIDE_BUY, SIDE_SELL, build_signal

logger = get_logger(__name__)

# alias de universos predefinidos
UNIVERSES = {
    'mega': FALLBACK_TICKERS[:20],
    'sp500': None,        # se resuelve en runtime via universe.get_universe()
}

_scan_state: Dict = {
    'running': False,
    'job_id': None,
    'progress': 0,
    'total': 0,
    'started_at': None,
    'finished_at': None,
    'error': None,
}
_scan_lock = threading.Lock()
_results: List[Dict] = []
_results_ts: float = 0.0
_results_key: str = ''


def scan_status() -> Dict:
    """estado del job de escaneo actual/ultimo"""
    return {k: v for k, v in _scan_state.items()}


def last_results() -> List[Dict]:
    return _results


def scan_is_stale(max_age_seconds: int) -> bool:
    return (time.time() - _results_ts) > max_age_seconds


def _scan_worker(job_id: str, tickers: List[str], period: str, predictor, max_workers: int):
    global _results, _results_ts, _results_key
    collector = DataCollector()
    processor = TechnicalProcessor()

    try:
        spy = collector.download_ticker('SPY', period)
        vix = collector.download_ticker('^VIX', period)
        vix_series = vix['Close'] if vix is not None else None

        done = 0

        def scan_one(ticker: str) -> Optional[Dict]:
            nonlocal done
            try:
                df = collector.download_ticker(ticker, period)
                if df is None or df.empty:
                    return None
                df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix)
                X = df[predictor.feature_cols].tail(1)
                if X.isna().any(axis=1).iloc[0]:
                    return None
                prob_up = predictor._raw_probability_up(X)
                regime = detect_regime(df, vix_series=vix_series)
                sig = build_signal(ticker, df, prob_up,
                                   as_of=str(df.index[-1].date()), regime=regime)
                d = sig.to_dict()
                d['volume_ratio'] = round(float(df['volumen_ratio'].iloc[-1]), 2) \
                    if 'volumen_ratio' in df.columns else None
                d['rsi'] = round(float(df['rsi'].iloc[-1]), 1) if 'rsi' in df.columns else None
                return d
            except Exception as e:
                logger.debug(f"scan {ticker}: {e}")
                return None
            finally:
                with _scan_lock:
                    done += 1
                    _scan_state['progress'] = done

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            rows = [r for r in pool.map(scan_one, tickers) if r is not None]

        rows.sort(key=lambda r: r.get('conviction_pct', 0), reverse=True)
        _results = rows
        _results_ts = time.time()
        _results_key = f"{len(tickers)}:{period}"
        with _scan_lock:
            _scan_state.update(running=False, finished_at=time.time(), error=None)
        logger.info(f"scan job {job_id}: {len(rows)}/{len(tickers)} tickers con senal")

    except Exception as e:
        logger.error(f"scan job {job_id} fallo: {e}")
        with _scan_lock:
            _scan_state.update(running=False, finished_at=time.time(), error=str(e))


def start_scan(
    predictor,
    universe: str = 'mega',
    period: str = '3mo',
    max_workers: int = 8,
    force: bool = False,
) -> Dict:
    """
    lanza (o reutiliza) un escaneo en segundo plano.

    Returns:
        {'job_id', 'started': bool, 'status': {...}, 'results': [...] }
        - si hay resultados frescos del mismo universo y no es force,
          se devuelven directamente sin relanzar.
    """
    global _scan_state

    tickers = UNIVERSES.get(universe) or FALLBACK_TICKERS[:20]
    if universe == 'sp500':
        from ..data.universe import get_universe
        tickers = get_universe(limit=500)

    key = f"{len(tickers)}:{period}"
    fresh = (_results and not force and _results_key == key
             and not scan_is_stale(max_age_seconds=300))
    if fresh:
        return {'job_id': _scan_state.get('job_id'), 'started': False,
                'status': scan_status(), 'results': _results}

    with _scan_lock:
        if _scan_state['running']:
            return {'job_id': _scan_state['job_id'], 'started': False,
                    'status': scan_status(), 'results': []}
        job_id = uuid.uuid4().hex[:10]
        _scan_state.update(running=True, job_id=job_id, progress=0,
                           total=len(tickers), started_at=time.time(),
                           finished_at=None, error=None)

    th = threading.Thread(
        target=_scan_worker, args=(job_id, tickers, period, predictor, max_workers),
        daemon=True, name=f"scan-{job_id}",
    )
    th.start()
    logger.info(f"scan job {job_id} lanzado: {len(tickers)} tickers, universo={universe}")
    return {'job_id': job_id, 'started': True, 'status': scan_status(), 'results': []}


def apply_filters(
    rows: List[Dict],
    side: Optional[str] = None,
    signal: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_conviction: Optional[float] = None,
    max_atr_pct: Optional[float] = None,
    min_volume_ratio: Optional[float] = None,
    sector: Optional[str] = None,
    watchlist: Optional[List[str]] = None,
    only_tradable: bool = False,
    limit: Optional[int] = None,
) -> List[Dict]:
    """filtros de trader sobre las senales escaneadas (spec 3)"""
    out = rows
    if side in (SIDE_BUY, SIDE_SELL):
        out = [r for r in out if r.get('side') == side]
    if signal:
        out = [r for r in out if r.get('signal') == signal.upper()]
    if min_confidence is not None:
        out = [r for r in out if r.get('confidence_pct', 0) >= min_confidence]
    if min_conviction is not None:
        out = [r for r in out if r.get('conviction_pct', 0) >= min_conviction]
    if max_atr_pct is not None:
        out = [r for r in out if r.get('atr_pct') is not None and r['atr_pct'] <= max_atr_pct]
    if min_volume_ratio is not None:
        out = [r for r in out if r.get('volume_ratio') is not None
               and r['volume_ratio'] >= min_volume_ratio]
    if sector is not None:
        sectors = get_sector_map()
        out = [r for r in out if sectors.get(r.get('ticker')) == sector]
    if watchlist is not None:
        wl = {t.upper() for t in watchlist}
        out = [r for r in out if r.get('ticker') in wl]
    if only_tradable:
        out = [r for r in out if r.get('trading_allowed', True)]
    if limit:
        out = out[:limit]
    return out
