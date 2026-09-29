"""Persistencia de senales validadas del scheduler (Sprint 2).

El ciclo de 5 min escanea un universo, valida con el ensemble y persiste:
- Fuente de verdad: tabla ``signal_cache`` (una fila por ticker+trade_date,
  upsert idempotente).
- Lecturas calientes: espejo en Redis/cache memoria con TTL corto; la API
  responde <1s sin tocar la DB.

La DB conserva el historial por fecha: permite reconstruir como evoluciona
una senal y es insumo del reentrenamiento online (futuro).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import select

from ..cache import cache_delete, cache_get, cache_set
from ..db.models import SignalCache
from ..db.session import get_session_factory, init_db

logger = logging.getLogger(__name__)

REDIS_KEY_LATEST = "signals:latest"
REDIS_KEY_TICKER = "signals:ticker:{ticker}"
LATEST_TTL = 600        # 10 min: el scheduler refresca cada 5
TICKER_TTL = 3600       # historico reciente por ticker


def _clean_for_json(obj: Any) -> Any:
    """NaN/inf/numpy -> tipos JSON puros (para la columna payload)."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()
                if not isinstance(v, (list, dict)) or v}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def _payload_from_row(row: SignalCache) -> Dict[str, Any]:
    """fila -> dict para el front (payload JSON completo si existe)."""
    if row.payload:
        d = dict(row.payload)
        d.setdefault("ticker", row.ticker)
        d.setdefault("trade_date", row.trade_date)
        d.setdefault("universe", row.universe)
        return d
    return {
        "ticker": row.ticker, "trade_date": row.trade_date,
        "signal": row.signal, "side": row.side,
        "prob_up": row.prob_up, "confidence_pct": row.confidence_pct,
        "conviction_pct": row.conviction_pct, "entry": row.entry,
        "stop_loss": row.stop_loss, "take_profits": [
            tp for tp in (row.tp1, row.tp2) if tp is not None
        ],
        "risk_reward": row.risk_reward,
        "position_size_pct": row.position_size_pct, "regime": row.regime,
        "trading_allowed": row.trading_allowed, "price": row.price,
        "atr_pct": row.atr_pct, "universe": row.universe,
    }


class SignalStore:
    """upsert y consulta de senales validadas (DB + espejo cache)."""

    def __init__(self):
        init_db()
        self._Session = get_session_factory()

    # ------------------------------------------------------------------
    # escritura (scheduler)
    # ------------------------------------------------------------------

    def upsert_batch(self, signals: List[Dict[str, Any]], universe: str) -> int:
        """upsert idempotente de un lote; devuelve filas escritas.

        El payload JSON guarda el dict completo de build_signal() para que
        el front reciba exactamente lo mismo que /api/signal.
        """
        if not signals:
            return 0
        written = 0
        with self._Session() as s:
            for d in signals:
                ticker = str(d.get("ticker", "")).upper()
                trade_date = str(d.get("as_of") or d.get("trade_date") or "")[:10]
                if not ticker or not trade_date:
                    continue
                row = s.execute(
                    select(SignalCache).where(
                        SignalCache.ticker == ticker,
                        SignalCache.trade_date == trade_date,
                    )
                ).scalar_one_or_none()
                if row is None:
                    row = SignalCache(ticker=ticker, trade_date=trade_date)
                    s.add(row)
                row.signal = d.get("signal", "")
                row.side = d.get("side", "")
                row.prob_up = d.get("prob_up")
                row.confidence_pct = d.get("confidence_pct")
                row.conviction_pct = d.get("conviction_pct")
                row.entry = d.get("entry")
                row.stop_loss = d.get("stop_loss")
                tps = d.get("take_profits") or []
                row.tp1 = tps[0]["price"] if tps and isinstance(tps[0], dict) else (tps[0] if tps else None)
                row.tp2 = tps[1]["price"] if len(tps) > 1 and isinstance(tps[1], dict) else (tps[1] if len(tps) > 1 else None)
                row.risk_reward = d.get("risk_reward")
                row.position_size_pct = d.get("position_size_pct")
                regime = d.get("regime")
                if isinstance(regime, dict):  # build_signal devuelve {status, reasons,...}
                    regime = regime.get("status")
                row.regime = regime
                row.trading_allowed = bool(d.get("trading_allowed", True))
                row.price = d.get("price")
                row.atr_pct = d.get("atr_pct")
                row.universe = universe
                row.payload = _clean_for_json(d)
                written += 1
            s.commit()
        self._invalidate_cache()
        return written

    def _invalidate_cache(self) -> None:
        cache_delete(REDIS_KEY_LATEST)
        # los espejos por-ticker expiran solos (TTL corto)

    # ------------------------------------------------------------------
    # lectura (API, <1s)
    # ------------------------------------------------------------------

    def latest_signals(self, limit: int = 50, only_tradable: bool = False,
                       side: Optional[str] = None,
                       signals: Optional[List[str]] = None,
                       min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """ultimas senales (cache 60s -> DB).

        signals: nombres de senal a incluir (p.ej. ['COMPRA_FUERTE']);
        min_confidence: confianza minima en puntos porcentuales.
        """
        sig_key = ",".join(sorted(signals)) if signals else "all"
        key = (f"{REDIS_KEY_LATEST}:{limit}:{only_tradable}:{side or 'all'}:"
               f"{sig_key}:{min_confidence or 0}")
        cached = cache_get(key)
        if cached is not None:
            return cached

        with self._Session() as s:
            # ultima fecha con senales, luego todas las filas de esa fecha
            last_date = s.execute(
                select(SignalCache.trade_date).order_by(SignalCache.trade_date.desc()).limit(1)
            ).scalar()
            if not last_date:
                return []
            q = s.execute(
                select(SignalCache)
                .where(SignalCache.trade_date == last_date)
                .order_by(SignalCache.conviction_pct.desc())
            ).scalars().all()
        rows = [_payload_from_row(r) for r in q]
        if only_tradable:
            rows = [r for r in rows if r.get("trading_allowed", True)]
        if side:
            # build_signal usa 'largo'/'corto'; aceptamos tambien buy/sell
            wanted = {"buy": {"buy", "largo"}, "sell": {"sell", "corto"}}.get(
                side.lower(), {side.lower()})
            rows = [r for r in rows if str(r.get("side", "")).lower() in wanted]
        if signals:
            wanted_signals = {s.upper().replace(" ", "_") for s in signals}
            rows = [r for r in rows
                    if str(r.get("signal", "")).upper().replace(" ", "_") in wanted_signals]
        if min_confidence is not None:
            rows = [r for r in rows
                    if (r.get("confidence_pct") or 0) >= min_confidence]
        rows = rows[:limit]
        cache_set(key, rows, ttl=60)
        return rows

    def ticker_history(self, ticker: str, limit: int = 30) -> List[Dict[str, Any]]:
        """historial reciente de senales de un ticker (cache 1h)."""
        key = f"{REDIS_KEY_TICKER.format(ticker=ticker.upper())}:{limit}"
        cached = cache_get(key)
        if cached is None:
            with self._Session() as s:
                q = s.execute(
                    select(SignalCache)
                    .where(SignalCache.ticker == ticker.upper())
                    .order_by(SignalCache.trade_date.desc())
                    .limit(limit)
                ).scalars().all()
            cached = [_payload_from_row(r) for r in q]
            cache_set(key, cached, ttl=TICKER_TTL)
        return cached

    def count(self) -> int:
        with self._Session() as s:
            return s.query(SignalCache).count()
