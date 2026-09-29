"""Sprint 1: API unificada FastAPI — health, DB aislada, cache y contratos
heredados del Flask (watchlists CRUD + paper performance).

Estrategia de aislamiento: REDIS_URL vacia (fallback memoria) y set_engine
con SQLite temporal, para no tocar data/saas.db ni el PG de docker.
"""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

os.environ.pop("REDIS_URL", None)

from fastapi.testclient import TestClient  # noqa: E402

from src.cache import get_cache, reset_cache  # noqa: E402
from src.db.models import Base  # noqa: E402
from src.db.session import create_db_engine, set_engine  # noqa: E402


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """SQLite temporal como engine global de la app (aislado del repo).

    Ademas anula los nombres de los ficheros legados para que los stores
    globales no importen data/paper_trading.db ni data/watchlists.json.
    """
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'api_test.db').as_posix()}")
    Base.metadata.create_all(engine)
    set_engine(engine)
    monkeypatch.setattr("src.trading.paper.LEGACY_DB_NAME", "_sin_legado.db")
    monkeypatch.setattr("src.ml.watchlist.LEGACY_JSON_NAME", "_sin_legado.json")
    reset_cache()
    yield engine
    set_engine(None)
    engine.dispose()
    reset_cache()


@pytest.fixture()
def client(isolated_db):
    from src.api.fastapi_app import app

    with TestClient(app) as c:
        yield c


# ----------------------------------------------------------------------
# health
# ----------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["cache"]["backend"] == "memory"


# ----------------------------------------------------------------------
# watchlists CRUD (mismo contrato que el Flask heredado)
# ----------------------------------------------------------------------

def test_watchlists_crud_flow(client):
    # crear
    r = client.post("/api/watchlists", json={"name": "Tech Beta", "tickers": ["msft"]})
    assert r.status_code == 201
    wl = r.json()
    assert wl["name"] == "Tech Beta"
    assert wl["tickers"] == ["MSFT"]

    # listar y get
    r = client.get("/api/watchlists")
    assert r.status_code == 200
    assert any(w["name"] == "Tech Beta" for w in r.json()["watchlists"])
    r = client.get("/api/watchlists/Tech Beta")
    assert r.json()["tickers"] == ["MSFT"]

    # anadir y quitar tickers
    r = client.post("/api/watchlists/Tech Beta/tickers", json={"tickers": ["AAPL", ""]})
    assert r.status_code == 200
    assert r.json()["tickers"] == ["AAPL", "MSFT"]
    r = client.request("DELETE", "/api/watchlists/Tech Beta/tickers", json={"tickers": ["AAPL"]})
    assert r.status_code == 200
    assert r.json()["tickers"] == ["MSFT"]

    # duplicado -> 409
    r = client.post("/api/watchlists", json={"name": "Tech Beta"})
    assert r.status_code == 409

    # get inexistente -> 404
    assert client.get("/api/watchlists/Nope").status_code == 404

    # borrar
    r = client.delete("/api/watchlists/Tech Beta")
    assert r.status_code == 200
    assert r.json()["deleted"] is True


# ----------------------------------------------------------------------
# paper trading (contrato del Flask heredado)
# ----------------------------------------------------------------------

def test_paper_performance_empty(client):
    r = client.get("/api/paper/performance")
    assert r.status_code == 200
    perf = r.json()
    assert "available" in perf
    if perf["available"]:
        for key in ("n_resolved", "directional_accuracy", "win_rate",
                    "profit_factor", "by_signal"):
            assert key in perf
    else:
        assert perf["n_resolved"] == 0


def test_paper_signals_empty(client):
    r = client.get("/api/paper/signals")
    assert r.status_code == 200
    assert r.json()["signals"] == []


# ----------------------------------------------------------------------
# endpoints que dependen de artefactos (degradados pero sin 500)
# ----------------------------------------------------------------------

def test_screener_without_scan(client):
    r = client.get("/api/screener")
    assert r.status_code == 200
    body = r.json()
    assert body["error"]  # pide lanzar POST /api/scan
    assert "status" in body


def test_top_signals_without_scan(client):
    r = client.get("/api/top-signals")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False


def test_scan_status_shape(client):
    r = client.get("/api/scan/status")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body and "n_results" in body


def test_expectancy(tmp_path, isolated_db, monkeypatch):
    """con OOF presente responde 200 con metricas validadas; sin el, 503."""
    from src.utils.config import Config

    has_oof = os.path.exists(os.path.join(Config().models_dir, "advanced_oof.pkl"))
    if not has_oof:
        monkeypatch.chdir(tmp_path)  # reports/ y data/ ausentes
    from src.api.fastapi_app import app

    with TestClient(app) as c:
        r = c.get("/api/expectancy")
    if has_oof:
        assert r.status_code == 200
        body = r.json()
        assert body["tech7_oof"]["profit_factor"] > 1.0
        assert body["config"]["rr_multiple"] == 2.0
    else:
        assert r.status_code == 503


# ----------------------------------------------------------------------
# cache: fallback memoria + TTL + invalidacion del panel
# ----------------------------------------------------------------------

def test_cache_memory_fallback(isolated_db):
    reset_cache()
    assert get_cache().backend == "memory"
    from src.cache import cache_get, cache_set

    cache_set("k", {"a": 1}, ttl=60)
    assert cache_get("k") == {"a": 1}


def test_cache_expiry(isolated_db):
    reset_cache()
    from src.cache import cache_get, cache_set

    cache_set("ttl", "v", ttl=0.05)
    assert cache_get("ttl") == "v"
    import time

    time.sleep(0.08)
    assert cache_get("ttl") is None


def test_top_signals_cache_roundtrip(isolated_db, monkeypatch):
    """el panel se sirve desde cache sin necesidad de escaneo previo."""
    from src.cache import cache_set

    payload = {"available": True, "signals": [], "buy_count": 1, "sell_count": 0}
    cache_set("top_signals", payload, ttl=60)
    from src.api.fastapi_app import app

    with TestClient(app) as c:
        r = c.get("/api/top-signals")
    assert r.status_code == 200
    assert r.json() == payload
