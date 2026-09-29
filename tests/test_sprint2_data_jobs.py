"""Sprint 2: proveedores de mercado (Alpaca/FMP/yfinance), universos
ampliados, signal_store (upsert + lecturas) y ciclo del scheduler.

Todo aislado: proveedores falsos (sin red), SQLite temporal y cache en
memoria. El ciclo del scheduler corre con un predictor falso para validar
el pipeline completo datos -> features -> senal -> persistencia.
"""
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("apscheduler")

os.environ.pop("REDIS_URL", None)

from src.cache import cache_get, reset_cache  # noqa: E402
from src.db.models import Base  # noqa: E402
from src.db.session import create_db_engine, set_engine  # noqa: E402


# ----------------------------------------------------------------------
# helpers: datos sinteticos y proveedores falsos
# ----------------------------------------------------------------------

def _synth_ohlcv(days: int = 260, base: float = 100.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-09-25", periods=days)
    close = base + np.cumsum(rng.normal(0.05, 1.0, days))
    high = close * (1 + np.abs(rng.normal(0, 0.008, days)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, days)))
    open_ = low + (high - low) * rng.random(days)
    vol = rng.integers(1_000_000, 5_000_000, days).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=dates)


class FakeProvider:
    """devuelve OHLCV sintetico para cualquier simbolo."""
    name = "fake"
    configured = True

    def fetch_daily(self, symbol: str, period: str = "1y"):
        if symbol == "^VIX":
            df = _synth_ohlcv(seed=3, base=15.0)
            df["Volume"] = 0.0
            return df
        return _synth_ohlcv(seed=abs(hash(symbol)) % 1000)


class BrokenProvider:
    name = "broken"
    configured = True

    def fetch_daily(self, symbol: str, period: str = "1y"):
        raise RuntimeError("proveedor caido")


class FakePredictor:
    feature_cols = ["rsi", "macd"]

    def _raw_probability_up(self, X):
        return 0.62


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 's2.db').as_posix()}")
    Base.metadata.create_all(engine)
    set_engine(engine)
    monkeypatch.setattr("src.trading.paper.LEGACY_DB_NAME", "_n.db")
    monkeypatch.setattr("src.ml.watchlist.LEGACY_JSON_NAME", "_n.json")
    monkeypatch.setattr("src.jobs.scheduler.get_market_data", lambda: FakeProvider())
    monkeypatch.setattr("src.ml.advanced_predictor.AdvancedPredictor", FakePredictor)
    # singletons de la API ligados al engine anterior -> recrear por test
    import src.api.fastapi_app as _fa

    _fa._signal_store = None
    _fa._watchlists = None
    _fa._paper = None
    reset_cache()
    yield engine
    set_engine(None)
    _fa._signal_store = None
    _fa._watchlists = None
    _fa._paper = None
    engine.dispose()
    reset_cache()


# ----------------------------------------------------------------------
# routing de proveedores
# ----------------------------------------------------------------------

def test_symbol_kind_and_fmp_mapping():
    from src.data.providers import _fmp_symbol, symbol_kind

    assert symbol_kind("MSFT") == "equity"
    assert symbol_kind("EURUSD=X") == "fx"
    assert symbol_kind("BTC-USD") == "crypto"
    assert _fmp_symbol("EURUSD=X") == "EURUSD"
    assert _fmp_symbol("BTC-USD") == "BTCUSD"
    assert _fmp_symbol("MSFT") is None


def test_market_data_routes_equity_to_alpaca():
    from src.data.providers import MarketData

    md = MarketData(alpaca=FakeProvider(), fmp=None, yf_provider=BrokenProvider())
    df = md.fetch_daily("MSFT", "1y")
    assert df is not None and len(df) > 100
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_market_data_degrades_to_yfinance_on_primary_failure():
    from src.data.providers import MarketData

    md = MarketData(alpaca=BrokenProvider(), fmp=None, yf_provider=FakeProvider())
    df = md.fetch_daily("AAPL", "1y")
    assert df is not None and len(df) > 100


def test_market_data_fx_uses_fmp_first():
    from src.data.providers import MarketData

    class FmpFake(FakeProvider):
        name = "fmp"

    md = MarketData(alpaca=None, fmp=FmpFake(), yf_provider=BrokenProvider())
    df = md.fetch_daily("EURUSD=X", "1y")
    assert df is not None  # fmp falso respondio; yf roto no hace falta

    # sin FMP configurado -> yfinance (roto en este test) -> None sin lanzar
    md2 = MarketData(alpaca=None, fmp=None, yf_provider=BrokenProvider())
    assert md2.fetch_daily("EURUSD=X", "1y") is None


def test_providers_status_reflects_env(monkeypatch):
    from src.data.providers import MarketData, reset_market_data

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    reset_market_data()
    from src.data.providers import get_market_data

    st = get_market_data().providers_status()
    assert st == {"equity_primary": "yfinance", "fx_primary": "yfinance",
                  "crypto_primary": "yfinance"}
    reset_market_data()


# ----------------------------------------------------------------------
# universos ampliados
# ----------------------------------------------------------------------

def test_universe_registry_sizes():
    from src.data.universes import resolve_universe

    assert len(resolve_universe("fx_major")) == 10
    assert len(resolve_universe("crypto20")) == 20
    assert len(resolve_universe("ndx100")) >= 50
    assert len(resolve_universe("mega")) == 20


def test_universe_unknown_raises_and_resolve_any_falls_back():
    from src.data.universes import resolve_any, resolve_universe

    with pytest.raises(ValueError):
        resolve_universe("no-existe")
    assert len(resolve_any("no-existe")) == 20  # cae a mega


# ----------------------------------------------------------------------
# signal_store: upsert idempotente + lecturas
# ----------------------------------------------------------------------

def _sig(ticker: str, conviction: float = 80.0, as_of: str = "2026-09-25") -> dict:
    return {
        "ticker": ticker, "as_of": as_of, "signal": "COMPRA", "side": "buy",
        "prob_up": 0.62, "confidence_pct": 62.0, "conviction_pct": conviction,
        "entry": 100.0, "stop_loss": 95.0,
        "take_profits": [{"price": 105.0}, {"price": 110.0}],
        "risk_reward": 2.0, "position_size_pct": 5.0, "regime": "normal",
        "trading_allowed": True, "price": 100.5, "atr_pct": 2.1,
    }


def test_signal_store_upsert_idempotent_and_latest(isolated_db):
    from src.ml.signal_store import SignalStore

    store = SignalStore()
    assert store.upsert_batch([_sig("MSFT", 80.0)], "ndx100") == 1
    # mismo ticker+fecha -> actualiza, no duplica
    assert store.upsert_batch([_sig("MSFT", 90.0)], "ndx100") == 1
    assert store.count() == 1

    rows = store.latest_signals(limit=10)
    assert len(rows) == 1
    assert rows[0]["conviction_pct"] == 90.0
    assert rows[0]["take_profits"][0]["price"] == 105.0


def test_signal_store_filters(isolated_db):
    from src.ml.signal_store import SignalStore

    store = SignalStore()
    store.upsert_batch([_sig("AAPL"), _sig("NVDA")], "ndx100")
    sell = dict(_sig("TSLA"), side="sell", signal="VENTA", trading_allowed=False)
    store.upsert_batch([sell], "ndx100")

    assert len(store.latest_signals(limit=50)) == 3
    assert len(store.latest_signals(limit=50, side="buy")) == 2
    assert len(store.latest_signals(limit=50, only_tradable=True)) == 2

    hist = store.ticker_history("aapl")  # case-insensitive
    assert len(hist) == 1 and hist[0]["ticker"] == "AAPL"


# ----------------------------------------------------------------------
# scheduler: ciclo completo con pipeline falso
# ----------------------------------------------------------------------

def test_scheduler_run_cycle_persists(isolated_db):
    from src.jobs.scheduler import run_cycle, scheduler_status
    from src.ml.signal_store import SignalStore

    res = run_cycle(universe="ndx100", limit=3)
    assert res["n_tickers"] == 3
    assert res["written"] >= 1
    assert res["seconds"] > 0

    st = scheduler_status()
    assert st["cycles"] == 1
    assert st["last_written"] == res["written"]

    store = SignalStore()
    assert store.count() == res["written"]

    # espejo de cache listo para la API
    mirror = cache_get("signals:latest")
    assert isinstance(mirror, list) and len(mirror) >= 1
    assert all("ticker" in s and "conviction_pct" in s for s in mirror)


def test_scheduler_second_cycle_updates_same_rows(isolated_db):
    from src.jobs.scheduler import run_cycle
    from src.ml.signal_store import SignalStore

    run_cycle(universe="ndx100", limit=2)
    store = SignalStore()
    n_first = store.count()
    assert n_first >= 1

    res = run_cycle(universe="ndx100", limit=2)
    assert store.count() == n_first  # misma fecha -> upsert, no crece


# ----------------------------------------------------------------------
# API: health expone proveedores y scheduler; signals-cache sirve el espejo
# ----------------------------------------------------------------------

def test_health_and_signals_cache_endpoints(isolated_db):
    from fastapi.testclient import TestClient

    from src.api.fastapi_app import app

    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert set(body["providers"]) == {"equity_primary", "fx_primary", "crypto_primary"}
        assert "cycles" in body["scheduler"]

        r = c.get("/api/signals-cache")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False  # sin ciclo previo en esta DB
        assert body["signals"] == []
        assert "scheduler" in body


def test_signals_cache_serves_after_cycle(isolated_db):
    from fastapi.testclient import TestClient

    from src.api.fastapi_app import app
    from src.jobs.scheduler import run_cycle

    run_cycle(universe="ndx100", limit=2)
    with TestClient(app) as c:
        r = c.get("/api/signals-cache?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["n"] == len(body["signals"])
    assert body["signals"][0]["ticker"]
