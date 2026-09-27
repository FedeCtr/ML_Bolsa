"""
Paper trading persistente (forward testing sin capital).

Ciclo diario (spec 5):
  1. record_signal(): registra la senal emitida hoy para cada ticker
     (entrada, SL, TP1/TP2, confianza, regimen) en SQLite — una por
     ticker/dia (UNIQUE trade_date+ticker).
  2. resolve_pending(): cuando existe el cierre del dia siguiente, resuelve
     cada senal: acierto direccional, toque de TP1 o SL (conservador: si
     ambos se tocan en la misma sesion cuenta SL) hasta 5 sesiones.
  3. performance(): metricas financieras continuas sobre los retornos por
     senal resuelta: Sharpe, Max Drawdown, Win/Loss, Profit Factor,
     directional accuracy.

La BBDD vive en data/paper_trading.db (ignorada por git).
"""
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from ..utils.logger import get_logger
from ..utils.config import Config
from ..backtesting.metrics import compute_metrics

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    signal TEXT NOT NULL,
    side TEXT NOT NULL,
    entry REAL,
    stop_loss REAL,
    tp1 REAL,
    tp2 REAL,
    confidence_pct REAL,
    conviction_pct REAL,
    prob_up REAL,
    price REAL,
    position_size_pct REAL,
    regime TEXT,
    UNIQUE(trade_date, ticker)
);
CREATE TABLE IF NOT EXISTS outcomes (
    signal_id INTEGER PRIMARY KEY REFERENCES signals(id),
    next_close REAL,
    next_ret_pct REAL,
    direction_correct INTEGER,
    hit_tp1 INTEGER,
    hit_sl INTEGER,
    resolved_at TEXT
);
"""


class PaperTrader:
    """registro y resolucion de senales paper en SQLite"""

    MAX_SESSIONS_TO_RESOLVE = 5   # ventanas para tocar TP1/SL

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(str(Config().DATA_DIR), 'paper_trading.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # 1) registro
    # ------------------------------------------------------------------

    def record_signal(self, s: Dict) -> Optional[int]:
        """registra (o actualiza) la senal del dia para un ticker"""
        now = datetime.now(timezone.utc).isoformat()
        tps = s.get('take_profits') or []
        tp1 = tps[0]['price'] if len(tps) > 0 else None
        tp2 = tps[1]['price'] if len(tps) > 1 else None
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO signals (ts, trade_date, ticker, signal, side,
                     entry, stop_loss, tp1, tp2, confidence_pct, conviction_pct,
                     prob_up, price, position_size_pct, regime)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(trade_date, ticker) DO UPDATE SET
                     signal=excluded.signal, side=excluded.side,
                     entry=excluded.entry, stop_loss=excluded.stop_loss,
                     tp1=excluded.tp1, tp2=excluded.tp2,
                     confidence_pct=excluded.confidence_pct,
                     conviction_pct=excluded.conviction_pct,
                     prob_up=excluded.prob_up, price=excluded.price,
                     position_size_pct=excluded.position_size_pct,
                     regime=excluded.regime, ts=excluded.ts
                """,
                (now, s.get('as_of'), s.get('ticker'), s.get('signal'), s.get('side'),
                 s.get('entry'), s.get('stop_loss'), tp1, tp2,
                 s.get('confidence_pct'), s.get('conviction_pct'),
                 s.get('prob_up'), s.get('price'), s.get('position_size_pct'),
                 (s.get('regime') or {}).get('status')),
            )
            row = c.execute(
                "SELECT id FROM signals WHERE trade_date=? AND ticker=?",
                (s.get('as_of'), s.get('ticker')),
            ).fetchone()
        logger.info(f"paper: senal registrada {s.get('ticker')} {s.get('signal')} ({s.get('as_of')})")
        return row['id'] if row else None

    # ------------------------------------------------------------------
    # 2) resolucion vs mercado real
    # ------------------------------------------------------------------

    def resolve_pending(self, ohlc_provider) -> int:
        """
        resuelve senales pendientes con datos reales.

        Args:
            ohlc_provider: fn(ticker) -> DataFrame OHLCV indexado por fecha.
        Returns:
            numero de senales resueltas ahora.
        """
        resolved = 0
        with self._conn() as c:
            pending = c.execute(
                """SELECT s.* FROM signals s
                   LEFT JOIN outcomes o ON o.signal_id = s.id
                   WHERE o.signal_id IS NULL""",
            ).fetchall()

        for sig in pending:
            try:
                df = ohlc_provider(sig['ticker'])
                if df is None or df.empty:
                    continue
                idx = df.index.strftime('%Y-%m-%d')
                pos = pd.Series(range(len(df)), index=idx)
                if sig['trade_date'] not in pos.index:
                    continue
                i = int(pos.loc[sig['trade_date']])
                future = df.iloc[i + 1: i + 1 + self.MAX_SESSIONS_TO_RESOLVE]
                if future.empty:
                    continue  # aun no hay cierre siguiente

                side_sign = 1.0 if sig['side'] == 'largo' else -1.0
                next_close = float(future['Close'].iloc[0])
                next_ret_pct = (next_close / float(sig['entry']) - 1) * 100 * side_sign \
                    if sig['entry'] else None
                direction_correct = int(next_ret_pct > 0) if next_ret_pct is not None else 0

                # TP1/SL: primer toque; empate en la misma sesion -> SL (conservador)
                hit_tp1, hit_sl = 0, 0
                if sig['entry'] and sig['stop_loss']:
                    long_side = sig['side'] == 'largo'
                    for _, r in future.iterrows():
                        sl_hit = r['Low'] <= sig['stop_loss'] if long_side \
                            else r['High'] >= sig['stop_loss']
                        tp_hit = (sig['tp1'] is not None and
                                  (r['High'] >= sig['tp1'] if long_side else r['Low'] <= sig['tp1']))
                        if sl_hit:
                            hit_sl = 1
                            break
                        if tp_hit:
                            hit_tp1 = 1
                            break

                with self._conn() as c:
                    c.execute(
                        """INSERT OR REPLACE INTO outcomes
                           (signal_id, next_close, next_ret_pct, direction_correct,
                            hit_tp1, hit_sl, resolved_at)
                           VALUES (?,?,?,?,?,?,?)""",
                        (sig['id'], next_close, next_ret_pct, direction_correct,
                         hit_tp1, hit_sl, datetime.now(timezone.utc).isoformat()),
                    )
                resolved += 1
            except Exception as e:
                logger.warning(f"paper: no se pudo resolver {sig['ticker']} {sig['trade_date']}: {e}")

        if resolved:
            logger.info(f"paper: {resolved} senales resueltas contra cierre real")
        return resolved

    # ------------------------------------------------------------------
    # 3) metricas continuas
    # ------------------------------------------------------------------

    def performance(self) -> Dict:
        """metricas financieras de todas las senales resueltas"""
        with self._conn() as c:
            rows = c.execute(
                """SELECT s.trade_date, s.ticker, s.signal, s.side, s.conviction_pct,
                          s.position_size_pct, o.next_ret_pct, o.direction_correct,
                          o.hit_tp1, o.hit_sl
                   FROM signals s JOIN outcomes o ON o.signal_id = s.id
                   ORDER BY s.trade_date""",
            ).fetchall()

        if not rows:
            return {'available': False, 'n_resolved': 0}

        df = pd.DataFrame([dict(r) for r in rows])
        # retorno por senal ponderado por sizing (paper: fraccion del sizing/100)
        weights = (df['position_size_pct'].fillna(0) / 100).clip(0, 1)
        raw = df['next_ret_pct'] / 100
        strat = (raw * weights).astype(float)

        # serie diaria: si hay varias senales el mismo dia, promedio
        daily = strat.groupby(df['trade_date']).mean()

        metrics = compute_metrics(daily, periods_per_year=252)
        wins = df[df['next_ret_pct'] > 0]
        losses = df[df['next_ret_pct'] <= 0]
        gross_win = float(wins['next_ret_pct'].clip(lower=0).sum())
        gross_loss = float(-losses['next_ret_pct'].clip(upper=0).sum())

        return {
            'available': True,
            'n_signals': int(len(df)),
            'n_resolved': int(len(df)),
            'directional_accuracy': float(df['direction_correct'].mean()),
            'win_rate': float((df['next_ret_pct'] > 0).mean()),
            'hit_tp1': int(df['hit_tp1'].sum()),
            'hit_sl': int(df['hit_sl'].sum()),
            'profit_factor': round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            'avg_return_pct': round(float(df['next_ret_pct'].mean()), 3),
            'by_signal': df.groupby('signal')['next_ret_pct'].agg(['count', 'mean']).round(3)
                          .reset_index().to_dict('records'),
            'period': [df['trade_date'].min(), df['trade_date'].max()],
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        }

    def list_signals(self, limit: int = 100) -> List[Dict]:
        """ultimas senales con su estado de resolucion"""
        with self._conn() as c:
            rows = c.execute(
                """SELECT s.*, o.next_ret_pct, o.direction_correct, o.hit_tp1, o.hit_sl
                   FROM signals s LEFT JOIN outcomes o ON o.signal_id = s.id
                   ORDER BY s.trade_date DESC, s.ticker LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
