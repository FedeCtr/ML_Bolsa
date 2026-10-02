"""Sprint 6: monetizacion — Stripe gateway (entitlements, eventos webhook,
checkout sin claves), paywalls en la API y alertas exclusivas Pro."""
import os

import pytest

os.environ.pop("REDIS_URL", None)
os.environ.pop("STRIPE_SECRET_KEY", None)
os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

from src.cache import reset_cache  # noqa: E402
from src.db.models import Base, User  # noqa: E402
from src.db.session import create_db_engine, set_engine  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 's6.db').as_posix()}")
    Base.metadata.create_all(engine)
    set_engine(engine)
    monkeypatch.setattr("src.trading.paper.LEGACY_DB_NAME", "_n.db")
    monkeypatch.setattr("src.ml.watchlist.LEGACY_JSON_NAME", "_n.json")
    # singletons de la API ligados al engine de tests anteriores: reset
    import src.api.fastapi_app as _fa

    _fa._signal_store = None
    _fa._watchlists = None
    _fa._paper = None
    reset_cache()
    yield engine
    set_engine(None)
    engine.dispose()
    _fa._signal_store = None
    _fa._watchlists = None
    _fa._paper = None
    reset_cache()


def _event(etype: str, obj: dict) -> dict:
    return {"type": etype, "data": {"object": obj}}


# ----------------------------------------------------------------------
# gateway: entitlements
# ----------------------------------------------------------------------

def test_tier_limits_free_vs_pro():
    from src.billing.stripe_gateway import FREE_TIER, PRO_TIER, tier_limit

    free = tier_limit(FREE_TIER)
    pro = tier_limit(PRO_TIER)
    assert free["alerts"] is False and pro["alerts"] is True
    assert free["ai_insights"] is False and pro["ai_insights"] is True
    assert free["delayed_hours"] == 24 and pro["delayed_hours"] == 0
    assert set(free["screener_universes"]) == {"sp500"}
    assert len(pro["screener_universes"]) == 4


def test_can_access_ticker():
    from src.billing.stripe_gateway import can_access_ticker

    assert can_access_ticker("AAPL", "free")
    assert can_access_ticker("TSLA", "free")
    assert can_access_ticker("BTC-USD", "free")
    assert not can_access_ticker("NVDA", "free")
    assert can_access_ticker("NVDA", "pro")
    assert can_access_ticker("ZN=F", "pro")


def test_get_or_create_user_and_status(isolated_db):
    from src.billing.stripe_gateway import get_or_create_user, user_status

    u = get_or_create_user("Demo@Example.com ", "Demo")
    u2 = get_or_create_user("demo@example.com")  # misma identidad (normaliza)
    assert u.id == u2.id and u.tier == "free"

    st = user_status(u.id)
    assert st["tier"] == "free" and st["is_pro"] is False
    assert "AAPL" in st["free_tickers"] and st["subscription"] is None


# ----------------------------------------------------------------------
# gateway: eventos webhook (sin red)
# ----------------------------------------------------------------------

def test_apply_checkout_completed_activates_pro(isolated_db):
    from src.billing.stripe_gateway import (
        apply_stripe_event,
        get_or_create_user,
        user_status,
    )

    user = get_or_create_user("buyer@example.com")

    res = apply_stripe_event(_event("checkout.session.completed", {
        "client_reference_id": str(user.id),
        "customer": "cus_123",
        "subscription": "sub_456",
    }))
    assert res["applied"] and res["tier"] == "pro"
    st = user_status(user.id)
    assert st["is_pro"] and st["subscription"]["status"] == "active"
    assert st["subscription"]["stripe_customer_id"] == "cus_123"


def test_apply_subscription_deleted_downgrades(isolated_db):
    from src.billing.stripe_gateway import apply_stripe_event, get_or_create_user, user_status

    user = get_or_create_user("churn@example.com")
    apply_stripe_event(_event("checkout.session.completed", {
        "client_reference_id": str(user.id), "subscription": "sub_x"}))
    assert user_status(user.id)["is_pro"]

    res = apply_stripe_event(_event("customer.subscription.deleted", {
        "metadata": {"user_id": str(user.id)}, "id": "sub_x", "customer": "cus_1"}))
    assert res["applied"] and res["tier"] == "free"
    st = user_status(user.id)
    assert not st["is_pro"] and st["subscription"]["status"] == "canceled"


def test_apply_subscription_updated_status(isolated_db):
    from src.billing.stripe_gateway import apply_stripe_event, get_or_create_user, user_status

    user = get_or_create_user("pastdue@example.com")
    res = apply_stripe_event(_event("customer.subscription.updated", {
        "metadata": {"user_id": str(user.id)}, "id": "sub_y",
        "status": "past_due", "current_period_end": 1767225600}))
    assert res["applied"]
    st = user_status(user.id)
    assert not st["is_pro"] and st["subscription"]["status"] == "past_due"


def test_apply_event_without_user_id_is_noop():
    from src.billing.stripe_gateway import apply_stripe_event

    assert apply_stripe_event(_event("checkout.session.completed", {}))["applied"] is False
    assert "no gestionado" in apply_stripe_event(_event("invoice.paid", {}))["reason"]


# ----------------------------------------------------------------------
# webhooks: verificacion de firma
# ----------------------------------------------------------------------

def test_verify_webhook_requires_secret_and_signature():
    from src.billing.stripe_gateway import verify_webhook

    with pytest.raises(PermissionError):
        verify_webhook(b"{}", "sig")  # sin STRIPE_WEBHOOK_SECRET
    os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_test"
    try:
        with pytest.raises(PermissionError):
            verify_webhook(b"{}", None)  # sin cabecera
    finally:
        os.environ.pop("STRIPE_WEBHOOK_SECRET", None)


# ----------------------------------------------------------------------
# checkout sin claves -> 503
# ----------------------------------------------------------------------

def test_checkout_without_keys_raises():
    from src.billing.stripe_gateway import create_checkout_session

    class FakeUser:
        id = 1
        email = "x@y.z"

    with pytest.raises(RuntimeError):
        create_checkout_session(FakeUser())


# ----------------------------------------------------------------------
# endpoints API
# ----------------------------------------------------------------------

def test_billing_endpoints(isolated_db):
    from fastapi.testclient import TestClient

    import src.api.fastapi_app as fa

    with TestClient(fa.app, raise_server_exceptions=False) as c:
        cfg = c.get("/api/billing/config").json()
        assert cfg["stripe_configured"] is False
        assert set(cfg["free_tickers"]) == {"AAPL", "TSLA", "BTC-USD"}

        st = c.get("/api/billing/status", params={"email": "e2e@test.com"})
        assert st.status_code == 200
        body = st.json()
        assert body["tier"] == "free" and body["limits"]["alerts"] is False

        r = c.post("/api/billing/checkout", json={"email": "e2e@test.com"})
        assert r.status_code == 503  # sin STRIPE_SECRET_KEY

        r = c.post("/api/billing/webhook", content=b"{}",)
        assert r.status_code == 401  # firma ausente/secret no configurado


def test_signal_paywall_402_for_free(isolated_db):
    """con email de usuario free, un ticker no demo responde 402."""
    from fastapi.testclient import TestClient

    import src.api.fastapi_app as fa
    from src.billing.stripe_gateway import get_or_create_user

    user = get_or_create_user("free@test.com")

    with TestClient(fa.app, raise_server_exceptions=False) as c:
        r = c.get("/api/signal/NVDA", params={"email": "free@test.com"})
        assert r.status_code == 402
        assert r.json()["detail"]["code"] == "upgrade_required"

        # el user free SI puede ver el demo (aqui puede fallar 404/503 por
        # modelo/datos en CI, pero nunca 402)
        r2 = c.get("/api/signal/AAPL", params={"email": "free@test.com"})
        assert r2.status_code != 402

        # sin email -> beta local abierta (nunca 402)
        r3 = c.get("/api/signal/NVDA")
        assert r3.status_code != 402


def test_signals_cache_filters_free_tier(isolated_db):
    """con email free, /api/signals-cache solo devuelve los demo."""
    from fastapi.testclient import TestClient

    import src.api.fastapi_app as fa
    from src.billing.stripe_gateway import get_or_create_user
    from src.ml.signal_store import SignalStore

    store = SignalStore()
    store.upsert_batch([
        {"ticker": "AAPL", "as_of": "2026-09-29", "signal": "COMPRA_FUERTE",
         "side": "largo", "prob_up": 0.6, "confidence_pct": 70,
         "conviction_pct": 70, "entry": 100.0, "stop_loss": 95.0,
         "take_profits": [{"price": 110.0}], "risk_reward": 2.0,
         "regime": "normal", "trading_allowed": True},
        {"ticker": "NVDA", "as_of": "2026-09-29", "signal": "COMPRA_FUERTE",
         "side": "largo", "prob_up": 0.6, "confidence_pct": 80,
         "conviction_pct": 80, "entry": 100.0, "stop_loss": 95.0,
         "take_profits": [{"price": 110.0}], "risk_reward": 2.0,
         "regime": "normal", "trading_allowed": True},
    ], "sp500")
    get_or_create_user("free@test.com")

    with TestClient(fa.app, raise_server_exceptions=False) as c:
        r = c.get("/api/signals-cache", params={"email": "free@test.com"})
        body = r.json()
        assert body["tier"] == "free"
        assert {s["ticker"] for s in body["signals"]} == {"AAPL"}

        r2 = c.get("/api/signals-cache")  # beta local abierta
        assert r2.json()["n"] == 2


# ----------------------------------------------------------------------
# alertas exclusivas Pro
# ----------------------------------------------------------------------

def test_subscribers_only_pro(isolated_db):
    from src.alerts.notify import _subscribers
    from src.db.session import get_session_factory

    with get_session_factory()() as s:
        s.add(User(clerk_id="c1", email="free@x.com", tier="free",
                   telegram_chat_id="111"))
        s.add(User(clerk_id="c2", email="pro@x.com", tier="pro",
                   telegram_chat_id="222"))
        s.add(User(clerk_id="c3", email="legacy@x.com", tier="",
                   telegram_chat_id="333"))  # beta local: se trata como pro
        s.commit()

    subs = _subscribers()
    chats = {u.telegram_chat_id for u in subs}
    assert chats == {"222", "333"}  # free excluido
