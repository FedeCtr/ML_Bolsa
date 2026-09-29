"""Universo SaaS: SP500 + NDX100 + FX + crypto (Sprint 2).

Universos embebidos (FX/crypto) y SP500/NDX100 resueltos en runtime:
- ``sp500``    : S&P 500 completo (Wikipedia + fallback embebido).
- ``mega``     : top-20 liquidas (compatibilidad screener heredado).
- ``ndx100``   : NASDAQ-100 embebido (mega-caps del indice).
- ``fx_major`` : 10 pares principales (convencion interna XXXYYY=X).
- ``crypto20`` : top-20 por liquidez (convencion interna XXX-USD).

``resolve_universe(name)`` -> lista de simbolos internos; los proveedores
convierten a su convencion nativa. ``universe_info()`` alimenta /api/meta.
"""
from __future__ import annotations

import time
from typing import Dict, List

from ..utils.logger import get_logger

logger = get_logger(__name__)

# NASDAQ-100: mega-caps del indice (lista estable embebida, dedup)
NDX100: List[str] = sorted(set([
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO", "TXN", "QCOM", "INTC",
    "AMGN", "HON", "AMAT", "BKNG", "ISRG", "SBUX", "INTU", "ADP", "GILD",
    "ADI", "MU", "REGN", "LULU", "CSX", "VRTX", "PANW", "FTNT",
    "CDNS", "SNPS", "MELI", "ASML", "CMCSA", "CHTR", "CTAS", "ODFL", "ROST",
    "MRNA", "MDB", "DDOG", "ZS", "TEAM", "PDD", "KDP", "MNST",
    "CEG", "XEL", "EXC", "CTSH", "PYPL", "ARM", "SMCI", "DASH",
]))

# 10 pares FX majors (convencion interna estilo yfinance)
FX_MAJOR: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
]

# top-20 crypto por liquidez (convencion interna estilo yfinance)
CRYPTO20: List[str] = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD",
    "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
    "LTC-USD", "BCH-USD", "UNI-USD", "ATOM-USD", "XLM-USD",
    "NEAR-USD", "APT-USD", "ARB-USD", "OP-USD", "FIL-USD",
]

_cache: Dict[str, tuple[float, List[str]]] = {}
_CACHE_TTL = 24 * 3600  # SP500 se re-resuelve como maximo cada 24h


def resolve_universe(name: str) -> List[str]:
    """devuelve la lista de simbolos internos de un universo registrado."""
    name = (name or "").lower()
    now = time.time()
    cached = _cache.get(name)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    if name == "sp500":
        from .universe import get_universe
        tickers = get_universe(limit=500)
    elif name == "ndx100":
        tickers = list(NDX100)
    elif name == "fx_major":
        tickers = list(FX_MAJOR)
    elif name == "crypto20":
        tickers = list(CRYPTO20)
    elif name == "mega":
        from .universe import FALLBACK_TICKERS
        tickers = FALLBACK_TICKERS[:20]
    else:
        raise ValueError(f"universo desconocido: {name} "
                         f"(validos: {', '.join(UNIVERSE_REGISTRY)})")

    _cache[name] = (now, tickers)
    return tickers


def resolve_any(name: str) -> List[str]:
    """resolve_universe tolerante: universo desconocido -> mega."""
    try:
        return resolve_universe(name)
    except ValueError:
        logger.warning(f"universo '{name}' desconocido; usando 'mega'")
        return resolve_universe("mega")


def universe_info() -> Dict[str, int]:
    """tamanos de los universos para /api/meta."""
    out: Dict[str, int] = {}
    for name in UNIVERSE_REGISTRY:
        try:
            out[name] = len(resolve_universe(name))
        except Exception as exc:
            logger.warning(f"universo {name} no resolvible: {exc}")
            out[name] = 0
    return out


UNIVERSE_REGISTRY = ("sp500", "mega", "ndx100", "fx_major", "crypto20")
