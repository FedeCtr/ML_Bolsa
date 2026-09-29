"""
Paper trading persistente (forward testing sin capital) sobre SQLAlchemy.

Ciclo diario (spec 5):
  1. record_signal(): registra la senal emitida hoy para cada ticker
     (entrada, SL, TP1/TP2, confianza, regimen) — una por ticker/dia
     (UNIQUE trade_date+ticker, upsert).
  2. resolve_pending(): cuando existe el cierre del dia siguiente, resuelve
     cada senal: acierto direccional, toque de TP1 o SL (conservador: si
     ambos se tocan en la misma sesion cuenta SL) hasta 5 sesiones.
  3. performance(): metricas financieras continuas sobre los retornos por
     senal resuelta: Sharpe, Max Drawdown, Win/Loss, Profit Factor,
     directional accuracy.

Backend: DATABASE_URL (PostgreSQL en prod) o SQLite data/saas.db (dev).
Migracion: si existe el SQLite legado data/paper_trading.db con filas y la
tabla nueva esta vacia, se importan automaticamente al primer uso.
"""
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import PaperOutcome, PaperSignal
from ..db.session import get_session_factory, init_db
from ..utils.config import Config
from ..utils.logger import get_logger
from ..backtesting.metrics import compute_metrics

logger = get_logger(__name__)

LEGACY_DB_NAME = 'paper_trading.db'


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperTrader:
    """registro y resolucion de senales paper (SQLAlchemy)"""

    MAX_SESSIONS_TO_RESOLVE = 5   # ventanas para tocar TP1/SL

    def __init__(self, db_path: Optional[str] = None):
        # db_path se acepta por compatibilidad (tests): si apunta a un SQLite
        # temporal se usa como DATABASE_URL aislada de este trader.
        self._owns_engine = db_path is not None
        if db_path is not None:
            os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from ..db.models import Base
            self._engine = create_engine(f"sqlite:///{db_path.as_posix() if hasattr(db_path, 'as_posix') else db_path}",
                                         connect_args={'check_same_thread': False})
            Base.metadata.create_all(self._engine)
            self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
            self._legacy_path = None
        else:
            init_db()
            self._engine = None
            self._Session = get_session_factory()
            self._legacy_path = os.path.join(str(Config().DATA_DIR), LEGACY_DB_NAME)

        with self._session() as s:
            s.query(PaperSignal).limit(1).all()   # valida esquema/conexion
        if self._legacy_path and os.path.exists(self._legacy_path):
            self._import_legacy_if_empty()

    # ------------------------------------------------------------------
    # infraestructura
    # ------------------------------------------------------------------

    def _session(self) -> Session:
        return self._Session()

    def _import_legacy_if_empty(self) -> None:
        """importa el SQLite legado una sola vez si la tabla nueva esta vacia."""
        try:
            with self._session() as s:
                if s.query(PaperSignal).first() is not None:
                    return
            import sqlite3
            conn = sqlite3.connect(self._legacy_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT s.*, o.next_close, o.next_ret_pct, o.direction_correct,
                          o.hit_tp1, o.hit_sl, o.resolved_at
                   FROM signals s LEFT JOIN outcomes o ON o.signal_id = s.id"""
            ).fetchall()
            conn.close()
            if not rows:
                return
            with self._session() as s:
                for r in rows:
                    sig = PaperSignal(
                        ts=r['ts'], trade_date=r['trade_date'], ticker=r['ticker'],
                        signal=r['signal'], side=r['side'], entry=r['entry'],
                        stop_loss=r['stop_loss'], tp1=r['tp1'], tp2=r['tp2'],
                        confidence_pct=r['confidence_pct'],
                        conviction_pct=r['conviction_pct'], prob_up=r['prob_up'],
                        price=r['price'], position_size_pct=r['position_size_pct'],
                        regime=r['regime'],
                    )
                    s.add(sig)
                    s.flush()
                    if r['resolved_at'] is not None:
                        s.add(PaperOutcome(
                            signal_id=sig.id, next_close=r['next_close'],
                            next_ret_pct=r['next_ret_pct'],
                            direction_correct=int(r['direction_correct'] or 0),
                            hit_tp1=int(r['hit_tp1'] or 0), hit_sl=int(r['hit_sl'] or 0),
                            resolved_at=r['resolved_at'],
                        ))
                s.commit()
            logger.info(f"paper: importadas {len(rows)} senales del legado {self._legacy_path}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"paper: importacion legado omitida ({e})")

    # ------------------------------------------------------------------
    # 1) registro
    # ------------------------------------------------------------------

    def record_signal(self, s: Dict) -> Optional[int]:
        """registra (o actualiza) la senal del dia para un ticker"""
        tps = s.get('take_profits') or []
        tp1 = tps[0]['price'] if len(tps) > 0 else None
        tp2 = tps[1]['price'] if len(tps) > 1 else None
        with self._session() as db:
            row = db.query(PaperSignal).filter_by(
                trade_date=s.get('as_of'), ticker=s.get('ticker')).one_or_none()
            if row is None:
                row = PaperSignal(trade_date=s.get('as_of'), ticker=s.get('ticker'))
                db.add(row)
            row.ts = _utcnow_iso()
            row.signal = s.get('signal')
            row.side = s.get('side')
            row.entry = s.get('entry')
            row.stop_loss = s.get('stop_loss')
            row.tp1 = tp1
            row.tp2 = tp2
            row.confidence_pct = s.get('confidence_pct')
            row.conviction_pct = s.get('conviction_pct')
            row.prob_up = s.get('prob_up')
            row.price = s.get('price')
            row.position_size_pct = s.get('position_size_pct')
            row.regime = (s.get('regime') or {}).get('status')
            db.commit()   # SQLAlchemy 2.0: sin commit automatico al salir del with
            db.flush()
            sig_id = row.id
        logger.info(f"paper: senal registrada {s.get('ticker')} {s.get('signal')} ({s.get('as_of')})")
        return sig_id

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
        with self._session() as db:
            pending = (
                db.query(PaperSignal)
                .outerjoin(PaperOutcome, PaperOutcome.signal_id == PaperSignal.id)
                .filter(PaperOutcome.signal_id.is_(None))
                .all()
            )
            sigs = [
                {c: getattr(p, c) for c in
                 ('id', 'trade_date', 'ticker', 'side', 'entry', 'stop_loss', 'tp1')}
                for p in pending
            ]

        for sig in sigs:
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

                with self._session() as db:
                    exists = db.get(PaperOutcome, sig['id'])
                    if exists is None:
                        db.add(PaperOutcome(
                            signal_id=sig['id'], next_close=next_close,
                            next_ret_pct=next_ret_pct,
                            direction_correct=direction_correct,
                            hit_tp1=hit_tp1, hit_sl=hit_sl,
                            resolved_at=_utcnow_iso(),
                        ))
                        db.commit()
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
        with self._session() as db:
            rows = (
                db.query(PaperSignal, PaperOutcome)
                .join(PaperOutcome, PaperOutcome.signal_id == PaperSignal.id)
                .order_by(PaperSignal.trade_date)
                .all()
            )
            data = [
                {
                    'trade_date': ps.trade_date, 'ticker': ps.ticker,
                    'signal': ps.signal, 'side': ps.side,
                    'conviction_pct': ps.conviction_pct,
                    'position_size_pct': ps.position_size_pct,
                    'next_ret_pct': po.next_ret_pct,
                    'direction_correct': po.direction_correct,
                    'hit_tp1': po.hit_tp1, 'hit_sl': po.hit_sl,
                }
                for ps, po in rows
            ]

        if not data:
            return {'available': False, 'n_resolved': 0}

        df = pd.DataFrame(data)
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
        with self._session() as db:
            rows = (
                db.query(PaperSignal, PaperOutcome)
                .outerjoin(PaperOutcome, PaperOutcome.signal_id == PaperSignal.id)
                .order_by(PaperSignal.trade_date.desc(), PaperSignal.ticker)
                .limit(limit)
                .all()
            )
            out = []
            for ps, po in rows:
                d = {c: getattr(ps, c) for c in
                     ('id', 'ts', 'trade_date', 'ticker', 'signal', 'side', 'entry',
                      'stop_loss', 'tp1', 'tp2', 'confidence_pct', 'conviction_pct',
                      'prob_up', 'price', 'position_size_pct', 'regime')}
                d.update({
                    'next_ret_pct': po.next_ret_pct if po else None,
                    'direction_correct': po.direction_correct if po else None,
                    'hit_tp1': po.hit_tp1 if po else None,
                    'hit_sl': po.hit_sl if po else None,
                })
                out.append(d)
            return out
