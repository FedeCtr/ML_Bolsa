"""
Universo de activos: S&P 500 (+ NASDAQ-100) desde Wikipedia con cache
y fallback embebido si la red falla.

La spec pide cubrir S&P 500 y NASDAQ. La lista del S&P 500 se scrapea de
Wikipedia (tabla estable, usada por toda la industria); el NASDAQ-100 se
filtra por exchange NASDAQ de la misma tabla. El fallback embebido cubre
los ~60 tickers mas liquidos para que el sistema nunca quede sin universo.
"""
import time
from typing import Dict, List, Optional

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

_cache: Dict = {'df': None, 'ts': 0.0}
CACHE_TTL_SECONDS = 24 * 3600

# fallback: mega-caps liquidas (si Wikipedia no responde)
FALLBACK_TICKERS: List[str] = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'BRK.B',
    'LLY', 'JPM', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'PG', 'WMT', 'JNJ',
    'ORCL', 'MRK', 'ADBE', 'CRM', 'NFLX', 'AMD', 'KO', 'PEP', 'BAC', 'TMO',
    'LIN', 'CSCO', 'ACN', 'MCD', 'ABT', 'WFC', 'DIS', 'INTC', 'QCOM', 'TXN',
    'CAT', 'GE', 'VZ', 'IBM', 'NOW', 'AMGN', 'DHR', 'NEE', 'PM', 'UNP',
    'SPGI', 'LOW', 'RTX', 'HON', 'UPS', 'BA', 'SBUX', 'GS', 'MS', 'BLK',
]

# ETFs de referencia para el contexto de mercado
MARKET_CONTEXT = {'market': 'SPY', 'volatility': '^VIX'}


def fetch_sp500_universe(force_refresh: bool = False) -> pd.DataFrame:
    """
    devuelve DataFrame [ticker, name, sector, exchange] del S&P 500.

    - Scrapea la tabla 'Components' de List of S&P 500 companies (Wikipedia).
    - Cache 24h en memoria.
    - Fallback embebido (exchange='?') si la descarga falla.
    """
    now = time.time()
    if not force_refresh and _cache['df'] is not None and now - _cache['ts'] < CACHE_TTL_SECONDS:
        return _cache['df']

    df = None
    try:
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            storage_options={'User-Agent': 'Mozilla/5.0 (MLBolsa research)'},
        )
        raw = tables[0][['Symbol', 'Security', 'GICS Sector', 'Headquarters Location']]
        raw = raw.rename(columns={
            'Symbol': 'ticker', 'Security': 'name',
            'GICS Sector': 'sector', 'Headquarters Location': 'hq',
        })
        # yfinance usa puntos para clase B (BRK.B -> BRK-B)
        raw['ticker'] = raw['ticker'].str.replace('.', '-', regex=False).str.upper()
        raw['exchange'] = raw['hq'].fillna('').astype(str)
        df = raw[['ticker', 'name', 'sector', 'exchange']]
        logger.info(f"universo S&P 500 descargado: {len(df)} tickers")
    except Exception as e:
        logger.warning(f"no se pudo descargar el universo de Wikipedia ({e}); usando fallback embebido")
        df = pd.DataFrame({
            'ticker': FALLBACK_TICKERS,
            'name': FALLBACK_TICKERS,
            'sector': 'Desconocido',
            'exchange': '?',
        })

    _cache['df'] = df
    _cache['ts'] = now
    return df


def get_universe(
    include_nasdaq_only: bool = True,
    limit: Optional[int] = None,
    force_refresh: bool = False,
) -> List[str]:
    """lista de tickers del universo (S&P 500; opcionalmente solo NASDAQ)."""
    df = fetch_sp500_universe(force_refresh)
    if include_nasdaq_only:
        df = df[df['exchange'].str.contains('NY', na=False) == False]
    tickers = df['ticker'].tolist()
    return tickers[:limit] if limit else tickers


def get_sector_map() -> Dict[str, str]:
    """mapa ticker -> sector GICS (para filtros del screener)."""
    df = fetch_sp500_universe()
    return dict(zip(df['ticker'], df['sector']))


def get_name_map() -> Dict[str, str]:
    """mapa ticker -> nombre de la compania."""
    df = fetch_sp500_universe()
    return dict(zip(df['ticker'], df['name']))
