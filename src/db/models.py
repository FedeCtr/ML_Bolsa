"""
Modelos SQLAlchemy 2.0 del SaaS (Sprint 1).

Soporta SQLite (dev/CI, sin servidor) y PostgreSQL 16 (produccion via
DATABASE_URL). Tipos elegidos compatibles con ambos dialectos.
"""
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, Integer, String,
                        UniqueConstraint, func)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    """usuario del SaaS (Sprint 6 integrara Clerk: clerk_id como identidad)"""
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    clerk_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), default='')
    tier: Mapped[str] = mapped_column(String(16), default='free')   # free | pro
    telegram_chat_id: Mapped[str] = mapped_column(String(64), default='')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Subscription(Base):
    """estado de suscripcion Stripe (Sprint 6): el webhook la mantiene al dia"""
    __tablename__ = 'subscriptions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stripe_customer_id: Mapped[str] = mapped_column(String(64), default='')
    stripe_subscription_id: Mapped[str] = mapped_column(String(64), default='')
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default='inactive')  # active|canceled|past_due
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 onupdate=utcnow)


class PaperSignal(Base):
    """senal paper (forward testing): una por (trade_date, ticker)"""
    __tablename__ = 'paper_signals'
    __table_args__ = (UniqueConstraint('trade_date', 'ticker', name='uq_paper_date_ticker'),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[str] = mapped_column(String(40), nullable=False)          # ISO UTC del registro
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(12), index=True, nullable=False)
    signal: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    tp1: Mapped[float | None] = mapped_column(Float)
    tp2: Mapped[float | None] = mapped_column(Float)
    confidence_pct: Mapped[float | None] = mapped_column(Float)
    conviction_pct: Mapped[float | None] = mapped_column(Float)
    prob_up: Mapped[float | None] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float)
    position_size_pct: Mapped[float | None] = mapped_column(Float)
    regime: Mapped[str | None] = mapped_column(String(16))


class PaperOutcome(Base):
    """resultado resuelto de una senal paper (1:1 con PaperSignal)"""
    __tablename__ = 'paper_outcomes'

    signal_id: Mapped[int] = mapped_column(Integer, primary_key=True,
                                           index=True)  # FK logica a paper_signals.id
    next_close: Mapped[float | None] = mapped_column(Float)
    next_ret_pct: Mapped[float | None] = mapped_column(Float)
    direction_correct: Mapped[int] = mapped_column(Integer, default=0)
    hit_tp1: Mapped[int] = mapped_column(Integer, default=0)
    hit_sl: Mapped[int] = mapped_column(Integer, default=0)
    resolved_at: Mapped[str] = mapped_column(String(40), nullable=False)


class Watchlist(Base):
    """watchlist por usuario: tickers en columna JSON (SQLite y PG lo soportan)"""
    __tablename__ = 'watchlists'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, default=0, index=True)  # single-user hasta Clerk
    tickers: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


class SignalCache(Base):
    """senal validada persistida por el scheduler (Sprint 2, cada 5 min).

    Una fila por (ticker, trade_date): el escaneo hace upsert. Las lecturas
    calientes pasan por Redis; la DB es la fuente de verdad.
    """
    __tablename__ = 'signal_cache'
    __table_args__ = (UniqueConstraint('ticker', 'trade_date', name='uq_signal_ticker_date'),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(12), index=True, nullable=False)
    trade_date: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    signal: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    prob_up: Mapped[float | None] = mapped_column(Float)
    confidence_pct: Mapped[float | None] = mapped_column(Float)
    conviction_pct: Mapped[float | None] = mapped_column(Float)
    entry: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    tp1: Mapped[float | None] = mapped_column(Float)
    tp2: Mapped[float | None] = mapped_column(Float)
    risk_reward: Mapped[float | None] = mapped_column(Float)
    position_size_pct: Mapped[float | None] = mapped_column(Float)
    regime: Mapped[str | None] = mapped_column(String(16))
    trading_allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    price: Mapped[float | None] = mapped_column(Float)
    atr_pct: Mapped[float | None] = mapped_column(Float)
    universe: Mapped[str] = mapped_column(String(24), default='tech')
    payload: Mapped[dict | None] = mapped_column(JSON)   # dict completo para el front
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 onupdate=utcnow)
