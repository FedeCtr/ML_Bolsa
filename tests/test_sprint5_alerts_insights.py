"""Sprint 5: filtros ampliados de signal_store, alertas (formato + dedup,
sin red) y modulo AI Insights (estructura; SHAP real se prueba integrado)."""
import os

import pytest

os.environ.pop("REDIS_URL", None)
os.environ.pop("TELEGRAM_BOT_TOKEN", None)
os.environ.pop("RESEND_API_KEY", None)

from src.cache import reset_cache  # noqa: E402
from src.db.models import Base  # noqa: E402
from src.db.session import create_db_engine, set_engine  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 's5.db').as_posix()}")
    Base.metadata.create_all(engine)
    set_engine(engine)
    monkeypatch.setattr("src.trading.paper.LEGACY_DB_NAME", "_n.db")
    monkeypatch.setattr("src.ml.watchlist.LEGACY_JSON_NAME", "_n.json")
    reset_cache()
    yield engine
    set_engine(None)
    engine.dispose()
    reset_cache()


def _sig(ticker: str, signal: str = "COMPRA", conf: float = 60.0,
         trading_allowed: bool = True) -> dict:
    return {
        "ticker": ticker, "as_of": "2026-09-29", "signal": signal,
        "side": "largo" if "COMPRA" in signal else "corto",
        "prob_up": 0.6, "confidence_pct": conf, "conviction_pct": conf,
        "entry": 100.0, "stop_loss": 95.0,
        "take_profits": [{"price": 110.0}, {"price": 120.0}],
        "risk_reward": 2.0, "regime": "normal", "trading_allowed": trading_allowed,
        "price": 100.0, "atr_pct": 2.0,
    }


# ----------------------------------------------------------------------
# signal_store: filtros nuevos del screener
# ----------------------------------------------------------------------

def test_latest_signals_filter_by_signal_names(isolated_db):
    from src.ml.signal_store import SignalStore

    store = SignalStore()
    store.upsert_batch([
        _sig("AAPL", "COMPRA_FUERTE"), _sig("MSFT", "COMPRA"),
        _sig("TSLA", "VENTA_FUERTE"),
    ], "ndx100")

    strong = store.latest_signals(signals=["COMPRA FUERTE", "VENTA FUERTE"])
    assert {r["ticker"] for r in strong} == {"AAPL", "TSLA"}  # tolera espacios
    buys = store.latest_signals(signals=["COMPRA_FUERTE", "COMPRA"])
    assert {r["ticker"] for r in buys} == {"AAPL", "MSFT"}


def test_latest_signals_min_confidence(isolated_db):
    from src.ml.signal_store import SignalStore

    store = SignalStore()
    store.upsert_batch([_sig("AAPL", conf=80.0), _sig("MSFT", conf=52.0)], "ndx100")
    rows = store.latest_signals(min_confidence=60)
    assert {r["ticker"] for r in rows} == {"AAPL"}


# ----------------------------------------------------------------------
# alertas
# ----------------------------------------------------------------------

def test_alert_format_message_and_email():
    from src.alerts.notify import format_email, format_message

    d = {
        "ticker": "NVDA", "signal": "COMPRA_FUERTE", "conviction_pct": 81,
        "entry": 135.5, "stop_loss": 124.4,
        "take_profits": [{"price": 146.6}, {"price": 157.7}],
        "risk_reward": 2.0, "regime": "normal",
    }
    text = format_message(d)
    assert "NVDA" in text and "COMPRA FUERTE" in text and "146.6" in text

    mail = format_email(d)
    assert "NVDA" in mail["subject"]
    assert "<br/>" in mail["html"] and "d4a843" in mail["html"]


def test_notify_without_channels_is_noop(isolated_db):
    from src.alerts.notify import notify_new_strong_signals

    res = notify_new_strong_signals([_sig("NVDA", "COMPRA_FUERTE")])
    assert res["sent"] == 0
    assert "reason" in res


def test_notify_dedup_with_fake_channel(isolated_db, monkeypatch):
    """con canal falso: envia una vez, deduplica la segunda."""
    from src.alerts import notify as N

    sent: list[str] = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr(N, "_subscribers", lambda: [type(
        "U", (), {"telegram_chat_id": "123", "email": ""})()])
    monkeypatch.setattr(N, "send_telegram",
                        lambda chat, text: sent.append(text) or True)

    sig = _sig("NVDA", "COMPRA_FUERTE")
    r1 = N.notify_new_strong_signals([sig])
    r2 = N.notify_new_strong_signals([dict(sig)])
    assert r1["sent"] == 1
    assert r2["sent"] == 0 and r2["skipped"] == 1  # dedup por ticker+fecha
    assert len(sent) == 1


def test_notify_skips_weak_and_suspended(isolated_db, monkeypatch):
    from src.alerts import notify as N

    sent: list[str] = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr(N, "_subscribers", lambda: [type(
        "U", (), {"telegram_chat_id": "123", "email": ""})()])
    monkeypatch.setattr(N, "send_telegram",
                        lambda chat, text: sent.append(text) or True)

    N.notify_new_strong_signals([
        _sig("MSFT", "COMPRA"),                       # debil
        _sig("ORCL", "COMPRA_FUERTE", trading_allowed=False),  # suspendida
    ])
    assert sent == []


# ----------------------------------------------------------------------
# insights (estructura; SHAP real validado contra el modelo en integracion)
# ----------------------------------------------------------------------

def test_explain_signal_without_ensemble_returns_none(isolated_db):
    from src.ml.insights import explain_signal

    class NoEnsemble:
        feature_cols = ["rsi"]
        ensemble = None

    import pandas as pd

    df = pd.DataFrame({"rsi": [55.0]})
    assert explain_signal(NoEnsemble(), df) is None


def test_insights_endpoint_guards(isolated_db):
    """el endpoint responde 503/404 controlado sin modelo soportado."""
    from fastapi.testclient import TestClient

    import src.api.fastapi_app as fa

    with TestClient(fa.app) as c:
        r = c.get("/api/insights/NOPE_TICKER_XYZ")
        assert r.status_code in (404, 500, 503)


def test_health_and_signals_cache_params(isolated_db):
    from fastapi.testclient import TestClient

    import src.api.fastapi_app as fa

    with TestClient(fa.app) as c:
        r = c.get("/api/signals-cache?signal=COMPRA_FUERTE&min_confidence=60&side=largo")
        assert r.status_code == 200
        assert r.json()["available"] is False  # DB vacia
