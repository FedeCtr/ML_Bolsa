"""
Persistencia del SaaS: SQLAlchemy 2.0 con SQLite (dev) y PostgreSQL (prod).

Uso basico:
    from src.db.session import init_db, get_session_factory
    init_db()
    Session = get_session_factory()
    with Session() as s:
        s.query(PaperSignal).filter_by(ticker='MSFT').first()
"""
from .models import (Base, PaperOutcome, PaperSignal, SignalCache, Subscription,
                     User, Watchlist)
from .session import (get_database_url, get_engine, get_session_factory,
                      healthcheck, init_db, set_engine)

__all__ = [
    'Base', 'PaperSignal', 'PaperOutcome', 'Watchlist', 'User', 'Subscription',
    'SignalCache',
    'get_database_url', 'get_engine', 'get_session_factory', 'init_db',
    'set_engine', 'healthcheck',
]
