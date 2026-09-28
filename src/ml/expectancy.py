"""
Expectativa matematica sobre probabilidades OOF (Sprint 3 - modelo v2).

Problema: la Directional Accuracy (49.5%) no mide rentabilidad. Con un R/R
estructural 1:2, un sistema acierta solo ~39.4% de las veces y aun asi tiene
Profit Factor > 1.3. Lo que hay que validar NO es acertar mas, sino cuantos
R gana el sistema por operacion tras costos, con niveles ejecutables.

Metodo (sin leakage):
  - Se usan SOLO probabilidades OOF del walk-forward purgado (cada fila fue
    predicha por un modelo que NO vio esa muestra).
  - Cada decision (fecha t, ticker) se ejecuta en la APERTURA de t+1 con
    datos REALES del CSV local: SL y TP se resuelven con el High/Low de t+1.
  - Empate (High y Low tocan ambos niveles) se resuelve como SL: criterio
    conservador, consistente con el paper trading.
  - Costos: comision + slippage + spread por lado (default 10 pb round trip).

Reconstruccion del par (fecha, ticker): el OOF se guarda por folds; cada fold
es un bloque de fechas ascendentes con los 7 tickers ciclando en el orden de
entrenamiento. La fecha SI es fiable por fila; el ticker se reconstruye por
estructura de bloques y se verifica contra el tamano esperado.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np
import os
import pandas as pd

from ..utils.config import Config
from ..utils.logger import get_logger
from .signal_engine import SIDE_BUY, SIDE_SELL

logger = get_logger(__name__)

COSTS_ROUND_TRIP = 0.001   # 10 pb por operacion (ida+vuelta)
RISK_ATR_MULT = 1.5        # SL a 1.5 ATR (fallback del motor de senales)
RISK_PCT_MIN = 0.01        # distancia minima del SL: 1%
RISK_PCT_MAX = 0.05        # distancia maxima del SL: 5%


# ----------------------------------------------------------------------
# Carga y reconstruccion del dataset OOF
# ----------------------------------------------------------------------

def load_oof_dataset(metadata_path: Optional[str] = None,
                     oof_path: Optional[str] = None,
                     raw_dir: Optional[str] = None) -> pd.DataFrame:
    """
    devuelve un DataFrame [date, ticker, prob_up, target] desde el OOF guardado.

    El campo 'tickers' del OOF historico esta corrupto (49,490 entradas para
    7,070 filas): el trainer extendia la lista con .loc sobre un indice de
    fechas duplicadas. La fecha por fila SI es fiable; el ticker se reconstruye
    por estructura de bloques (cada fold = un bloque ascendente de fechas con
    los tickers ciclando en el orden de entrenamiento).
    """
    config = Config()
    oof_path = oof_path or os.path.join(config.models_dir, 'advanced_oof.pkl')
    metadata_path = metadata_path or os.path.join(config.models_dir, 'advanced_metadata.pkl')

    oof = joblib.load(oof_path)
    meta = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}

    probs = np.asarray(oof['probabilities'], dtype=float).ravel()
    targets = np.asarray(oof['targets'], dtype=int).ravel()
    dates = [str(d) for d in oof['dates']]
    saved_tickers = oof.get('tickers')
    n = len(probs)
    if not (len(targets) == len(dates) == n):
        raise ValueError(f"OOF inconsistente: {len(probs)} probs / {len(targets)} targets / {len(dates)} fechas")

    train_tickers = meta.get('tickers') or []
    if not train_tickers:
        raise ValueError("metadata sin lista de tickers de entrenamiento")
    k = len(train_tickers)

    # CAMINO RAPIDO: si el OOF guardado trae su vector de tickers consistente
    # (trainer corregido: mapeo posicional exacto fila a fila), se usa directo
    # y la reconstruccion por bloques queda como fallback para OOF legacy.
    if (saved_tickers is not None and len(saved_tickers) == n
            and set(map(str, saved_tickers)) <= set(train_tickers)):
        df = pd.DataFrame({
            'date': dates,
            'ticker': [str(t) for t in saved_tickers],
            'prob_up': probs,
            'target': targets,
        })
        logger.info(
            f"OOF cargado (tickers del trainer, mapeo exacto): {n} filas, "
            f"{df['ticker'].nunique()} tickers, {df['date'].nunique()} fechas "
            f"({df['date'].min()} -> {df['date'].max()})"
        )
        return df

    # ESTRUCTURA del OOF: el panel se concatena ticker-major y los folds de CV
    # son por fecha. Dentro de un fold, las filas de cada ticker son un bloque
    # posicional contiguo con fechas ascendentes y SIN repetir fecha. Los
    # bloques se delimitan cuando la fecha "retrocede". No todos los folds
    # contienen los k tickers (histories cortas), asi que el numero de bloques
    # NO tiene por que ser multiplo de k: cada bloque se IDENTIFICA comparando
    # sus targets contra el OHLC real de cada candidato (el ticker verdadero
    # reproduce el target ~100%; los demas ~50%).
    config = Config()
    raw_dir = raw_dir or config.raw_data_dir
    close_map: Dict[str, pd.Series] = {}
    for t in train_tickers:
        p = os.path.join(raw_dir, f"{t}_raw.csv")
        if not os.path.exists(p):
            continue
        raw = pd.read_csv(p)
        raw[raw.columns[0]] = pd.to_datetime(raw[raw.columns[0]], utc=True).dt.strftime('%Y-%m-%d')
        s = raw.set_index(raw.columns[0])['Close'].sort_index()
        close_map[t] = s
    if not close_map:
        raise ValueError("sin CSV locales para verificar la estructura del OOF")

    block_id = np.zeros(n, dtype=int)
    for i in range(1, n):
        block_id[i] = block_id[i - 1] + (1 if dates[i] < dates[i - 1] else 0)

    ticker_by_pos = np.empty(n, dtype=object)
    excluded: List[int] = []
    block_report = []
    for b in range(int(block_id.max()) + 1):
        idx = np.flatnonzero(block_id == b)
        bdates = [dates[i] for i in idx]
        if any(bdates[i] >= bdates[i + 1] for i in range(len(bdates) - 1)):
            raise ValueError(f"bloque {b} sin fechas estrictamente ascendentes")
        if len(set(bdates)) != len(bdates):
            raise ValueError(f"bloque {b} con fechas repetidas (no es un solo ticker)")

        best_t, best_mismatch, best_cov = None, 1.0, 0.0
        for t, s in close_map.items():
            implied, covered = [], 0
            for d in bdates:
                pos = s.index.get_loc(d) if d in s.index else None
                if pos is None or pos + 1 >= len(s):
                    continue
                covered += 1
                implied.append(1 if s.iloc[pos + 1] > s.iloc[pos] else 0)
            cov = covered / len(bdates)
            if cov < 0.8:
                continue
            mism = float(np.mean(np.array(implied) != targets[idx[:covered]])) \
                if covered else 1.0
            if mism < best_mismatch:
                best_t, best_mismatch, best_cov = t, mism, cov

        if best_t is None or best_mismatch > 0.10:
            # Bloque NO verificable: los targets no se reproducen con ningun
            # ticker local (p.ej. yfinance reviso los factores de ajuste entre
            # la descarga de entrenamiento y la actual). Se EXCLUYEN esas filas
            # del analisis: mejor perder datos que simular sobre un mapeo
            # inventado. La exclusion es ciega a prob/target, no sesga.
            block_report.append({'bloque': b, 'ticker': None,
                                 'filas': len(idx), 'mismatch': round(best_mismatch, 4)})
            excluded.extend(idx)
            continue
        ticker_by_pos[idx] = best_t
        block_report.append({'bloque': b, 'ticker': best_t, 'filas': len(idx),
                             'mismatch': round(best_mismatch, 4)})

    verified = np.ones(n, dtype=bool)
    verified[list(excluded)] = False
    n_verified = int(verified.sum())
    if n_verified == 0:
        raise ValueError("ningun bloque del OOF pudo identificarse contra el OHLC local")
    if n_verified < 0.3 * n:
        logger.warning(
            f"solo {n_verified}/{n} filas verificables: la mayoria de los bloques "
            "no reproduce el OHLC local (posible cambio de ajustes de yfinance); "
            "el analisis cubrira solo los tickers identificados"
        )
    logger.info(
        f"filas excluidas por bloque no verificable: {len(excluded)}/{n}"
    )

    df = pd.DataFrame({
        'date': dates,
        'ticker': ticker_by_pos,
        'prob_up': probs,
        'target': targets,
        'verified': verified,
    })
    df = df[df['verified']].drop(columns=['verified']).reset_index(drop=True)

    n_unverified_tickers = len({r['ticker'] for r in block_report if r['ticker'] is None})
    logger.info(
        f"OOF reconstruido: {len(df)} filas verificadas, "
        f"{df['ticker'].nunique()} tickers (bloques no identificados: "
        f"{sum(1 for r in block_report if r['ticker'] is None)}, "
        f"~{n_unverified_tickers} tickers afectados)"
    )
    logger.info(f"bloques: {block_report}")
    logger.info(
        f"OOF cargado: {n} filas, {df['ticker'].nunique()} tickers, "
        f"{df['date'].nunique()} fechas ({df['date'].min()} -> {df['date'].max()})"
    )
    return df


def attach_market_data(oof_df: pd.DataFrame, raw_dir: Optional[str] = None) -> pd.DataFrame:
    """
    adjunta datos de ejecucion por fila: posicion de la fecha de decision en
    el calendario del ticker (para recorrer el OHLC real dia a dia durante el
    holding), ATR14 de t y VIX de t.
    """
    config = Config()
    raw_dir = raw_dir or config.raw_data_dir

    index_by_ticker: Dict[str, Dict[str, int]] = {}
    atr_by_ticker: Dict[str, pd.Series] = {}
    close_by_ticker: Dict[str, np.ndarray] = {}
    for ticker in oof_df['ticker'].unique():
        path = os.path.join(raw_dir, f"{ticker}_raw.csv")
        if not os.path.exists(path):
            logger.warning(f"sin CSV local para {ticker}; sus filas se descartaran")
            continue
        raw = pd.read_csv(path)
        date_col = raw.columns[0]
        raw[date_col] = pd.to_datetime(raw[date_col], utc=True).dt.strftime('%Y-%m-%d')
        # fechas duplicadas por el repaso DST de yfinance: quedarse con la ultima
        raw = raw.drop_duplicates(subset=date_col, keep='last')
        raw = raw.set_index(date_col).sort_index()
        index_by_ticker[ticker] = {d: i for i, d in enumerate(raw.index)}
        close_by_ticker[ticker] = raw['Close'].to_numpy(float)
        high, low, close = raw['High'], raw['Low'], raw['Close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_by_ticker[ticker] = tr.rolling(14).mean()

    dates = oof_df['date'].to_numpy()
    tickers = oof_df['ticker'].to_numpy()
    n = len(oof_df)
    pos_col = np.full(n, -1, dtype=int)
    atr_col = np.full(n, np.nan)
    close_t_col = np.full(n, np.nan)
    for i, (d, t) in enumerate(zip(dates, tickers)):
        pos = index_by_ticker.get(t, {}).get(d, -1)
        pos_col[i] = pos
        if pos >= 0:
            atr_series = atr_by_ticker[t]
            if d in atr_series.index:
                atr_col[i] = float(atr_series.loc[d])
            close_t_col[i] = close_by_ticker[t][pos]

    oof_df['pos_t'] = pos_col
    oof_df['atr_t'] = atr_col
    oof_df['close_t'] = close_t_col

    # VIX de mercado en la fecha de decision (gate de regimen macro)
    vix_path = os.path.join(raw_dir, '^VIX_raw.csv')
    if os.path.exists(vix_path):
        vix = pd.read_csv(vix_path)
        vix_date_col = vix.columns[0]
        vix[vix_date_col] = pd.to_datetime(vix[vix_date_col], utc=True).dt.strftime('%Y-%m-%d')
        vix = vix.drop_duplicates(subset=vix_date_col, keep='last')
        vix_map = vix.set_index(vix_date_col)['Close'].to_dict()
        oof_df['vix_t'] = oof_df['date'].map(vix_map)

    # tendencia macro: SPY sobre su SMA50 (gate de regimen para largos)
    spy_path = os.path.join(raw_dir, 'SPY_raw.csv')
    if os.path.exists(spy_path):
        spy = pd.read_csv(spy_path)
        spy_date_col = spy.columns[0]
        spy[spy_date_col] = pd.to_datetime(spy[spy_date_col], utc=True).dt.strftime('%Y-%m-%d')
        spy = spy.drop_duplicates(subset=spy_date_col, keep='last').set_index(spy_date_col).sort_index()
        spy['sma50'] = spy['Close'].rolling(50).mean()
        oof_df['spy_t'] = oof_df['date'].map(spy['Close'].to_dict())
        oof_df['spy_sma50_t'] = oof_df['date'].map(spy['sma50'].to_dict())

    valid = (oof_df['pos_t'] >= 0) & oof_df['atr_t'].notna() & (oof_df['atr_t'] > 0)
    logger.info(f"filas con datos de ejecucion: {int(valid.sum())}/{n}")
    return oof_df


# ----------------------------------------------------------------------
# Simulacion de expectativa
# ----------------------------------------------------------------------

@dataclass
class ExpectancyResult:
    prob_min: float
    rr_multiple: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float          # R medio por operacion (neto de costos)
    total_return_pct: float      # retorno compuesto del period (1% riesgo por trade)
    max_drawdown_pct: float
    sharpe: float                # anualizado por años del sample
    avg_hold_days: float
    per_side: Dict = field(default_factory=dict)
    equity: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'prob_min': self.prob_min, 'rr_multiple': self.rr_multiple,
            'n_trades': self.n_trades, 'win_rate': round(self.win_rate, 4),
            'profit_factor': round(self.profit_factor, 3),
            'expectancy_r': round(self.expectancy_r, 4),
            'total_return_pct': round(self.total_return_pct, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'sharpe': round(self.sharpe, 2),
            'avg_hold_days': round(self.avg_hold_days, 2),
            'per_side': self.per_side,
        }


def _resolve_trade_walk(side: str, entry: float, sl: float, tp: float,
                        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                        rr: float, max_hold: int) -> tuple:
    """
    recorre el OHLC real dia a dia hasta que se toca SL o TP (SL-first:
    si el dia toca ambos, cuenta SL — criterio conservador). Si vence
    max_hold sin tocar niveles, sale al cierre de ese dia.
    Devuelve (resultado en R neto de costos, dias retenido, desenlace:
    'tp' | 'sl' | 'timeout').
    """
    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return 0.0, 0, 'timeout'
    cost_r = COSTS_ROUND_TRIP / (risk_per_unit / entry)
    h = min(max_hold, len(highs))
    for j in range(h):
        hi, lo = highs[j], lows[j]
        if side == SIDE_BUY:
            hit_sl, hit_tp = lo <= sl, hi >= tp
            exit_px = closes[j]
        else:
            hit_sl, hit_tp = hi >= sl, lo <= tp
            exit_px = closes[j]
        if hit_sl:                       # empate -> SL (conservador)
            return -1.0 - cost_r, j + 1, 'sl'
        if hit_tp:
            return rr - cost_r, j + 1, 'tp'
        if j == h - 1:                   # timeout: salida a cierre
            gross = (exit_px - entry) / risk_per_unit if side == SIDE_BUY \
                else (entry - exit_px) / risk_per_unit
            return gross - cost_r, j + 1, 'timeout'
    return 0.0, 0, 'timeout'


def simulate_expectancy(
    df: pd.DataFrame,
    prob_min: float = 0.60,
    rr_multiple: float = 2.0,
    vix_max: Optional[float] = None,
    risk_per_trade: float = 0.01,
    max_hold: int = 20,
    allow_short: bool = True,
    spy_above_sma: Optional[bool] = None,
    one_position_per_ticker: bool = False,
    max_concurrent: Optional[int] = None,
    raw_dir: Optional[str] = None,
) -> ExpectancyResult:
    """
    simula el sistema: senal si prob del lado >= prob_min (simetrico para
    corto), SL a 1.5 ATR capado [1%, 5%], TP a rr_multiple x riesgo,
    entrada en apertura de t+1 y holding dia a dia sobre OHLC real hasta
    tocar SL/TP o agotar max_hold sesiones.

    Gates de regimen (todos opcionales, honestos: usan datos de t):
      vix_max: excluye decisiones con VIX por encima del nivel.
      spy_above_sma: si True, solo opera largos con SPY > SMA50.

    Realismo de portafolio (opcional):
      one_position_per_ticker: no acumula posiciones solapadas del mismo ticker.
      max_concurrent: limita posiciones abiertas simultaneas (aprox. por fechas).
    """
    config = Config()
    raw_dir = raw_dir or config.raw_data_dir

    work = df[(df['pos_t'] >= 0) & df['atr_t'].notna() & (df['atr_t'] > 0)].copy()
    long_mask = work['prob_up'] >= prob_min
    short_mask = work['prob_up'] <= (1.0 - prob_min)
    work['side'] = np.where(long_mask, SIDE_BUY, np.where(short_mask, SIDE_SELL, None))
    work = work[work['side'].notna()]
    if not allow_short:
        work = work[work['side'] == SIDE_BUY]
    if vix_max is not None and 'vix_t' in work.columns:
        work = work[work['vix_t'].fillna(0) <= vix_max]
    if spy_above_sma is True and 'spy_sma50_t' in work.columns:
        ok = work['spy_t'].notna() & work['spy_sma50_t'].notna() \
             & (work['spy_t'] > work['spy_sma50_t'])
        work = work[ok | (work['side'] != SIDE_BUY)]  # el gate solo afecta largos

    # cache de arrays OHLC por ticker
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    for ticker in work['ticker'].unique():
        path = os.path.join(raw_dir, f"{ticker}_raw.csv")
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path)
        raw[raw.columns[0]] = pd.to_datetime(raw[raw.columns[0]], utc=True).dt.strftime('%Y-%m-%d')
        raw = raw.drop_duplicates(subset=raw.columns[0], keep='last')
        raw = raw.set_index(raw.columns[0]).sort_index()
        cache[ticker] = {
            'Open': raw['Open'].to_numpy(float),
            'High': raw['High'].to_numpy(float),
            'Low': raw['Low'].to_numpy(float),
            'Close': raw['Close'].to_numpy(float),
        }

    results: List[float] = []
    hold_days: List[int] = []
    dates: List[str] = []
    sides: List[str] = []
    tickers_seq: List[str] = []
    entry_bars: List[int] = []
    equity = [1.0]
    n_tp = n_sl = n_timeout = 0

    # reserva de posiciones para el realismo de portafolio
    busy_until: Dict[str, int] = {}
    accepted: List[tuple] = []   # (entry_ts_approx, exit_ts_approx)

    work = work.sort_values(['date', 'ticker']).reset_index(drop=True)
    for row in work.itertuples(index=False):
        arrs = cache.get(row.ticker)
        if arrs is None:
            continue
        pos = int(row.pos_t)
        if pos + 1 >= len(arrs['Open']):
            continue

        # gates de portafolio (aproximacion con dias de calendario: 1 sesion ~ 1.45 dias)
        entry_ts = pd.Timestamp(row.date) + pd.Timedelta(days=1)
        if one_position_per_ticker and busy_until.get(row.ticker, -1) >= pos:
            continue
        if max_concurrent is not None:
            n_open = sum(1 for a_ts, b_ts in accepted if a_ts <= entry_ts <= b_ts)
            if n_open >= max_concurrent:
                continue

        entry = float(arrs['Open'][pos + 1])          # entrada: apertura de t+1
        risk_pct = float(np.clip(RISK_ATR_MULT * row.atr_t / row.close_t,
                                 RISK_PCT_MIN, RISK_PCT_MAX))
        sl_dist = entry * risk_pct
        if row.side == SIDE_BUY:
            sl = entry - sl_dist
            tp = entry + rr_multiple * sl_dist
        else:
            sl = entry + sl_dist
            tp = entry - rr_multiple * sl_dist

        r, held, outcome = _resolve_trade_walk(
            row.side, entry, sl, tp,
            arrs['High'][pos + 1:], arrs['Low'][pos + 1:], arrs['Close'][pos + 1:],
            rr_multiple, max_hold,
        )

        if one_position_per_ticker:
            busy_until[row.ticker] = pos + held
        if max_concurrent is not None:
            exit_ts = entry_ts + pd.Timedelta(days=max(1, int(held * 1.45)))
            accepted.append((entry_ts, exit_ts))

        results.append(r)
        equity.append(equity[-1] * (1 + risk_per_trade * r))
        hold_days.append(held)
        dates.append(row.date)
        sides.append(row.side)
        tickers_seq.append(row.ticker)
        entry_bars.append(pos + 1)
        if outcome == 'tp':
            n_tp += 1
        elif outcome == 'sl':
            n_sl += 1
        else:
            n_timeout += 1

    n_trades = len(results)
    if n_trades == 0:
        return ExpectancyResult(prob_min, rr_multiple, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    eq = np.array(equity)
    years = max((pd.Timestamp(max(dates)) - pd.Timestamp(min(dates))).days / 365.25, 1e-9)

    arr = np.array(results)
    wins, losses = arr[arr > 0], arr[arr <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    rets = np.diff(eq) / eq[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(n_trades / years)) if rets.std() > 0 else 0.0

    def side_stats(side: str) -> Dict:
        m = np.array(sides) == side
        if not m.any():
            return {}
        a = arr[m]
        return {
            'n': int(m.sum()),
            'win_rate': round(float((a > 0).mean()), 4),
            'expectancy_r': round(float(a.mean()), 4),
        }

    return ExpectancyResult(
        prob_min=prob_min,
        rr_multiple=rr_multiple,
        n_trades=n_trades,
        win_rate=float((arr > 0).mean()),
        profit_factor=gross_win / gross_loss if gross_loss > 0 else float('inf'),
        expectancy_r=float(arr.mean()),
        total_return_pct=float((eq[-1] - 1) * 100),
        max_drawdown_pct=dd,
        sharpe=sharpe,
        avg_hold_days=float(np.mean(hold_days)) if hold_days else 0.0,
        per_side={'largo': side_stats(SIDE_BUY), 'corto': side_stats(SIDE_SELL),
                  'n_tp': n_tp, 'n_sl': n_sl, 'n_timeout': n_timeout},
        equity=list(eq),
    )


def sweep_gates(
    df: pd.DataFrame,
    prob_mins: Optional[List[float]] = None,
    rr_multiples: Optional[List[float]] = None,
    vix_max: Optional[float] = None,
) -> pd.DataFrame:
    """barrido de prob_min x rr_multiple -> tabla comparable de expectativa."""
    prob_mins = prob_mins or [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.70]
    rr_multiples = rr_multiples or [1.5, 2.0]
    rows = []
    for pm in prob_mins:
        for rr in rr_multiples:
            res = simulate_expectancy(df, prob_min=pm, rr_multiple=rr, vix_max=vix_max)
            d = res.to_dict()
            d.pop('equity', None)
            rows.append(d)
    return pd.DataFrame(rows)
