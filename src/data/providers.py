"""Proveedores de datos de mercado (Sprint 2).

Arquitectura:
- ``AlpacaProvider``  : equities (REST v2, bars 1Day, httpx; sin SDK pesado).
- ``FMPProvider``     : forex y crypto (historical-price-full de FMP).
- ``YfinanceProvider``: fallback de desarrollo, siempre disponible.

``get_market_data()`` devuelve una fachada que enruta por tipo de simbolo:
- equities  -> Alpaca si hay claves, si no yfinance
- FX        -> FMP si hay clave, si no yfinance
- crypto    -> FMP si hay clave, si no yfinance
Cualquier fallo del proveedor primario degrada a yfinance (nunca lanza).

Los simbolos internos son estilo yfinance (equity "AAPL", FX "EURUSD=X",
crypto "BTC-USD"); cada proveedor convierte a su convencion nativa.

Estos proveedores NO escriben cache en disco (eso lo hace DataCollector para
el entrenamiento); el scheduler consume la fachada y persiste en SignalCache.
"""
from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# columnas canonicas que consume TechnicalProcessor (estilo yfinance)
_OHLCV = ["Open", "High", "Low", "Close", "Volume"]

ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
FMP_HIST_URL = "https://financialmodelingprep.com/api/v3/historical-price-full"


def _coerce_ohlcv(rows: list[dict]) -> pd.DataFrame:
    """filas [{'date','open','high','low','close','volume'}] -> DataFrame canonico."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_OHLCV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    for c in _OHLCV:
        if c not in df.columns:
            df[c] = 0.0
    return df[_OHLCV].astype(float)


class MarketDataProvider(ABC):
    """interfaz comun: OHLCV diario de un simbolo (estilo yfinance)."""

    name = "abstract"

    @abstractmethod
    def fetch_daily(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        ...


class YfinanceProvider(MarketDataProvider):
    """fallback de desarrollo: yfinance tal cual (simbolo ya en su convencion)."""

    name = "yfinance"

    def fetch_daily(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf

            df = yf.Ticker(symbol).history(period=period, interval="1d")
            if df is None or df.empty:
                return None
            df = df.rename(columns={c: c.capitalize() for c in df.columns})
            for c in _OHLCV:
                if c not in df.columns:
                    df[c] = 0.0
            return df[_OHLCV].astype(float)
        except Exception as exc:
            logger.warning(f"yfinance fallo {symbol}: {exc}")
            return None


class AlpacaProvider(MarketDataProvider):
    """equities via Alpaca Market Data v2 (requiere ALPACA_API_KEY/SECRET)."""

    name = "alpaca"

    def __init__(self, api_key: str, api_secret: str, timeout: float = 15.0):
        self._key = api_key
        self._secret = api_secret
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._key and self._secret)

    def fetch_daily(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        if not self.configured:
            return None
        end = datetime.now(timezone.utc)
        start = end - _period_to_timedelta(period)
        params = {
            "symbols": symbol.upper(),
            "timeframe": "1Day",
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "limit": 10000,
            "feed": "iex",  # plan gratuito (IEX); sip con claves de pago
        }
        headers = {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret}
        try:
            import httpx

            resp = httpx.get(ALPACA_DATA_URL, params=params, headers=headers,
                             timeout=self._timeout)
            resp.raise_for_status()
            bars = resp.json().get("bars", {}).get(symbol.upper(), [])
            if not bars:
                return None
            rows = [{
                "date": b["t"][:10],
                "open": b["o"], "high": b["h"],
                "low": b["l"], "close": b["c"], "volume": b["v"],
            } for b in bars]
            return _coerce_ohlcv(rows)
        except Exception as exc:
            logger.warning(f"alpaca fallo {symbol}: {exc}")
            return None


class FMPProvider(MarketDataProvider):
    """forex y crypto via Financial Modeling Prep (requiere FMP_API_KEY)."""

    name = "fmp"

    def __init__(self, api_key: str, timeout: float = 15.0):
        self._key = api_key
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._key)

    def fetch_daily(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        if not self.configured:
            return None
        native = _fmp_symbol(symbol)
        if native is None:
            return None
        params = {"apikey": self._key, "serietype": "line"}  # line: el plan free no da OHLC
        try:
            import httpx

            resp = httpx.get(f"{FMP_HIST_URL}/{native}", params=params,
                             timeout=self._timeout)
            resp.raise_for_status()
            hist = resp.json().get("historical", [])
            if not hist:
                return None
            start = (datetime.now(timezone.utc) - _period_to_timedelta(period)).date()
            rows = [{
                "date": h["date"],
                "open": h.get("open") or h.get("close"),
                "high": h.get("high") or h.get("close"),
                "low": h.get("low") or h.get("close"),
                "close": h.get("close"),
                "volume": h.get("volume") or 0.0,
            } for h in hist if h.get("date", "") >= str(start)]
            return _coerce_ohlcv(rows)
        except Exception as exc:
            logger.warning(f"fmp fallo {symbol}: {exc}")
            return None


# ----------------------------------------------------------------------
# routing por tipo de simbolo
# ----------------------------------------------------------------------

def symbol_kind(symbol: str) -> str:
    """clasifica un simbolo interno: 'fx' | 'crypto' | 'equity'."""
    s = symbol.upper()
    if s.endswith("=X"):
        return "fx"
    if s.endswith("-USD") or s.endswith("-BTC") or s.endswith("-EUR"):
        return "crypto"
    return "equity"


def _fmp_symbol(symbol: str) -> Optional[str]:
    """EURUSD=X -> EURUSD ; BTC-USD -> BTCUSD ; equity -> None."""
    s = symbol.upper()
    if s.endswith("=X"):
        return s.replace("=X", "")
    if s.endswith("-USD"):
        return s.replace("-USD", "USD")
    return None


def _period_to_timedelta(period: str) -> timedelta:
    table = {"1d": 2, "5d": 7, "1mo": 31, "3mo": 92, "6mo": 183,
             "1y": 366, "2y": 731, "3y": 1096, "5y": 1827, "10y": 3653}
    days = table.get(str(period).lower())
    if days is None:
        days = 366 if period.endswith("y") and str(period)[:-1].isdigit() \
            else (int(period[:-1]) if str(period)[:-1].isdigit() else 366)
    return timedelta(days=days)


class MarketData:
    """fachada: enruta el simbolo al proveedor primario y degrada a yfinance."""

    def __init__(self, alpaca: Optional[AlpacaProvider] = None,
                 fmp: Optional[FMPProvider] = None,
                 yf_provider: Optional[YfinanceProvider] = None):
        self.alpaca = alpaca or _alpaca_from_env()
        self.fmp = fmp or _fmp_from_env()
        self.yf = yf_provider or YfinanceProvider()

    def fetch_daily(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        kind = symbol_kind(symbol)
        primary = self.alpaca if kind == "equity" else self.fmp
        for provider in (primary, self.yf):
            if provider is None:
                continue
            try:
                df = provider.fetch_daily(symbol, period)
            except Exception as exc:
                logger.warning(f"{provider.name} error {symbol}: {exc}")
                df = None
            if df is not None and not df.empty:
                return df
        logger.warning(f"sin datos de {symbol} en ningun proveedor")
        return None

    def providers_status(self) -> dict:
        return {
            "equity_primary": self.alpaca.name if self.alpaca and self.alpaca.configured
            else "yfinance",
            "fx_primary": self.fmp.name if self.fmp and self.fmp.configured else "yfinance",
            "crypto_primary": self.fmp.name if self.fmp and self.fmp.configured
            else "yfinance",
        }


_instance: Optional[MarketData] = None
_lock = threading.Lock()


def get_market_data() -> MarketData:
    """singleton de la fachada (lee env una vez)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MarketData()
    return _instance


def reset_market_data() -> None:
    """tests: fuerza relectura de env."""
    global _instance
    with _lock:
        _instance = None


def _alpaca_from_env() -> Optional[AlpacaProvider]:
    key, secret = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_API_SECRET", "")
    return AlpacaProvider(key, secret) if key and secret else None


def _fmp_from_env() -> Optional[FMPProvider]:
    key = os.environ.get("FMP_API_KEY", "")
    return FMPProvider(key) if key else None
