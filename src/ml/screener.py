"""
Screener de mercado: escanea un universo de tickers con el ensemble calibrado
y expone filtros para traders.

Diseño:
  - El contexto de mercado (SPY, VIX) se descarga UNA vez por escaneo y se
    comparte entre tickers (antes se descargaba 3N veces).
  - Cache TTL en memoria: repetir el escaneo dentro del TTL no toca la red.
  - Cada ticker fallido se reporta; un fallo no aborta el escaneo.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd

from ..data.collector import DataCollector
from ..data.processor import TechnicalProcessor
from ..utils.logger import get_logger
from .signal_engine import SIDE_BUY, SIDE_SELL, build_signal

logger = get_logger(__name__)

DEFAULT_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AVGO', 'NFLX', 'AMD',
    'JPM', 'V', 'UNH', 'XOM', 'WMT', 'PG', 'MA', 'COST', 'HD', 'KO',
]

SECTORS = {
    'AAPL': 'Tecnologia', 'MSFT': 'Tecnologia', 'GOOGL': 'Tecnologia',
    'AMZN': 'Consumo discrecional', 'TSLA': 'Consumo discrecional',
    'NVDA': 'Semiconductores', 'META': 'Tecnologia', 'AVGO': 'Semiconductores',
    'NFLX': 'Comunicacion', 'AMD': 'Semiconductores',
    'JPM': 'Financiero', 'V': 'Financiero', 'MA': 'Financiero',
    'UNH': 'Salud', 'XOM': 'Energia', 'WMT': 'Consumo defensivo',
    'PG': 'Consumo defensivo', 'COST': 'Consumo defensivo',
    'HD': 'Consumo discrecional', 'KO': 'Consumo defensivo',
}


class MarketScreener:
    """escaneo batch del universo con senales graduadas + cache TTL"""

    def __init__(self, cache_ttl_seconds: int = 300, period: str = '1y'):
        self.cache_ttl = cache_ttl_seconds
        self.period = period
        self._cache: Dict = {}
        self._cache_ts: float = 0.0

    # ------------------------------------------------------------------

    def scan(
        self,
        universe: Optional[List[str]] = None,
        period: Optional[str] = None,
        max_workers: int = 4,
    ) -> List[Dict]:
        """escanea el universo y devuelve senales por ticker (con cache)"""
        universe = universe or DEFAULT_UNIVERSE
        period = period or self.period

        cache_key = (tuple(sorted(universe)), period)
        now = time.time()
        if self._cache and self._cache.get('_key') == cache_key and now - self._cache_ts < self.cache_ttl:
            logger.info("screener: sirviendo desde cache")
            return self._cache['rows']

        collector = DataCollector()
        processor = TechnicalProcessor()

        # contexto de mercado una sola vez para todo el escaneo
        spy = collector.download_ticker('SPY', period)
        vix = collector.download_ticker('^VIX', period)

        def scan_one(ticker: str) -> Optional[Dict]:
            try:
                df = collector.download_ticker(ticker, period)
                if df is None or df.empty:
                    return None
                df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix)
                return {
                    'df': df,
                    'prob_up': None,   # lo rellena el caller con el modelo
                    'ticker': ticker,
                }
            except Exception as e:
                logger.warning(f"screener: fallo escaneando {ticker}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(scan_one, universe))

        rows = [r for r in results if r is not None]
        self._cache = {'_key': cache_key, 'rows': rows}
        self._cache_ts = now
        logger.info(f"screener: {len(rows)}/{len(universe)} tickers escaneados")
        return rows

    # ------------------------------------------------------------------

    @staticmethod
    def apply_filters(
        rows: List[Dict],
        side: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_atr_pct: Optional[float] = None,
        min_atr_pct: Optional[float] = None,
        min_volume_ratio: Optional[float] = None,
        sector: Optional[str] = None,
    ) -> List[Dict]:
        """filtros de trader sobre las senales escaneadas"""
        out = rows
        if side in (SIDE_BUY, SIDE_SELL):
            out = [r for r in out if r.get('side') == side]
        if min_confidence is not None:
            out = [r for r in out if r.get('confidence_pct', 0) >= min_confidence * 100]
        if max_atr_pct is not None:
            out = [r for r in out if r.get('atr_pct') is not None and r['atr_pct'] <= max_atr_pct]
        if min_atr_pct is not None:
            out = [r for r in out if r.get('atr_pct') is not None and r['atr_pct'] >= min_atr_pct]
        if min_volume_ratio is not None:
            out = [r for r in out if r.get('volume_ratio') is not None
                   and r['volume_ratio'] >= min_volume_ratio]
        if sector is not None:
            out = [r for r in out if SECTORS.get(r.get('ticker')) == sector]
        return out


def rows_to_signal_rows(raw_rows: List[Dict], predictor) -> List[Dict]:
    """convierte filas crudas del screener en senales con el predictor"""
    out = []
    for row in raw_rows:
        df, ticker = row['df'], row['ticker']
        try:
            X = df[predictor.feature_cols].tail(1)
            if X.isna().any(axis=1).iloc[0]:
                continue
            prob_up = predictor._raw_probability_up(X)
            sig = build_signal(ticker, df, prob_up, as_of=str(df.index[-1].date()))
            d = sig.to_dict()
            d['sector'] = SECTORS.get(ticker, 'Otros')
            vol_ratio = df['volumen_ratio'].iloc[-1] if 'volumen_ratio' in df.columns else None
            d['volume_ratio'] = round(float(vol_ratio), 2) if vol_ratio is not None else None
            d['rsi'] = round(float(df['rsi'].iloc[-1]), 1) if 'rsi' in df.columns else None
            out.append(d)
        except Exception as e:
            logger.warning(f"screener: senal fallida para {ticker}: {e}")
    out.sort(key=lambda r: r.get('confidence_pct', 0), reverse=True)
    return out
