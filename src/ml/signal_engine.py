"""
Motor de senales operables por niveles.

Version comercial (spec 2): las senales SIEMPRE expresan direccion operable
en 4 categorias — COMPRA_FUERTE, COMPRA, VENTA, VENTA_FUERTE — sin bandas
neutras. La direccion sale del signo de la probabilidad calibrada (>=0.5
largo, <0.5 corto) y la CONVICCION de su distancia a 0.5:

    conviction_pct = |p - 0.5| * 200        (0% .. 100%)

La categoria FUERTE exige margen >= SIGNAL_STRONG_MARGIN (0.55/0.45).

Niveles operativos (ingenieria de riesgo determinista, no salidas del
modelo): soportes/resistencias por pivotes, SL bajo/ sobre el nivel con
buffer ATR, TP1/TP2 por multiples de ATR capados por estructura, R/R y
sizing por riesgo fijo.

Integracion con el detector de regimen (regime.py): si el regimen esta en
caution el sizing se escala 0.5x; si esta suspended, la senal se EMITE pero
marcada no-ejecutable (trading_allowed=False, sizing 0).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .regime import RegimeStatus, detect_regime

logger = get_logger(__name__)

SIGNAL_STRONG_MARGIN = 0.55   # p >= 0.55 (o <= 0.45) para categoria FUERTE

SIDE_BUY = 'largo'
SIDE_SELL = 'corto'


@dataclass
class SignalResult:
    """salida completa del motor de senales para un ticker"""
    ticker: str
    signal: str                  # COMPRA_FUERTE | COMPRA | VENTA | VENTA_FUERTE
    side: str                    # largo | corto (siempre hay lado)
    confidence_pct: float        # probabilidad calibrada del lado anunciado (%)
    conviction_pct: float        # |p-0.5|*200: fuerza de la senal (0-100)
    prob_up: float
    prob_down: float
    price: float
    as_of: str
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: List[Dict] = field(default_factory=list)
    risk_reward: Optional[float] = None
    position_size_pct: Optional[float] = None
    supports: List[float] = field(default_factory=list)
    resistances: List[float] = field(default_factory=list)
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    levels_source: str = 'pivotes + ATR'
    notes: List[str] = field(default_factory=list)
    regime: Dict = field(default_factory=dict)
    trading_allowed: bool = True

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'signal': self.signal,
            'side': self.side,
            'confidence_pct': round(self.confidence_pct, 2),
            'conviction_pct': round(self.conviction_pct, 2),
            'prob_up': round(self.prob_up, 4),
            'prob_down': round(self.prob_down, 4),
            'price': self.price,
            'as_of': self.as_of,
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'take_profits': self.take_profits,
            'risk_reward': self.risk_reward,
            'position_size_pct': self.position_size_pct,
            'supports': self.supports,
            'resistances': self.resistances,
            'atr': self.atr,
            'atr_pct': self.atr_pct,
            'levels_source': self.levels_source,
            'notes': self.notes,
            'regime': self.regime,
            'trading_allowed': self.trading_allowed,
        }


def classify_signal(prob_up: float) -> Tuple[str, str]:
    """clasifica la probabilidad calibrada en tier SIEMPRE operable"""
    if prob_up >= 0.5:
        return ('COMPRA_FUERTE', SIDE_BUY) if prob_up >= SIGNAL_STRONG_MARGIN \
            else ('COMPRA', SIDE_BUY)
    return ('VENTA_FUERTE', SIDE_SELL) if prob_up <= (1 - SIGNAL_STRONG_MARGIN) \
        else ('VENTA', SIDE_SELL)


def find_pivot_levels(
    high: pd.Series,
    low: pd.Series,
    left: int = 5,
    right: int = 5,
    tolerance: float = 0.01,
) -> Tuple[List[float], List[float]]:
    """
    detecta soportes y resistencias con pivotes fractales.

    Un pivote alto (resistencia) es un maximo local rodeado de `left`/`right`
    barras menores a cada lado; uno bajo (soporte), el caso inverso. Los
    pivotes cercanos entre si (< tolerance) se agrupan en zonas.
    """
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    n = len(highs)
    piv_hi: List[float] = []
    piv_lo: List[float] = []

    for i in range(left, n - right):
        window_h = highs[i - left: i + right + 1]
        window_l = lows[i - left: i + right + 1]
        if highs[i] == window_h.max() and (window_h < highs[i]).sum() >= left:
            piv_hi.append(highs[i])
        if lows[i] == window_l.min() and (window_l > lows[i]).sum() >= left:
            piv_lo.append(lows[i])

    def cluster(levels: List[float]) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        zones: List[List[float]] = [[levels[0]]]
        for lv in levels[1:]:
            if abs(lv - zones[-1][-1]) / zones[-1][-1] <= tolerance:
                zones[-1].append(lv)
            else:
                zones.append([lv])
        return [float(np.mean(z)) for z in zones]

    return cluster(piv_lo), cluster(piv_hi)


def build_levels(
    df: pd.DataFrame,
    side: str,
    atr_value: float,
    price: float,
    risk_reward_min: float = 1.5,
) -> Dict:
    """
    construye entrada, SL y TPs para el lado indicado.

    Reglas (largo; el corto es el espejo):
      - SL: bajo el soporte mas cercano, buffer 0.5*ATR. Fallback 1.5*ATR.
      - TP1/TP2: 1x / 2x ATR, capados al techo/suelo tecnico si corta antes.
      - R/R hasta TP2; si < risk_reward_min se notifica (no se falsea).
    """
    if atr_value <= 0 or not np.isfinite(atr_value):
        return {'entry': None, 'stop_loss': None, 'take_profits': [],
                'risk_reward': None, 'position_size_pct': None,
                'notes': [], 'supports': [], 'resistances': []}

    supports, resistances = find_pivot_levels(df['High'], df['Low'])

    if side == SIDE_BUY:
        below = [s for s in supports if s < price]
        sl = max(below) - 0.5 * atr_value if below else price - 1.5 * atr_value
        entry = price
        targets_raw = [entry + k * atr_value for k in (1.0, 2.0)]
        ceilings = [r for r in resistances if r > entry]
    else:
        above = [r for r in resistances if r > price]
        sl = min(above) + 0.5 * atr_value if above else price + 1.5 * atr_value
        entry = price
        targets_raw = [entry - k * atr_value for k in (1.0, 2.0)]
        ceilings = [s for s in supports if s < entry]

    sign = 1.0 if side == SIDE_BUY else -1.0

    notes: List[str] = []
    tps: List[Dict] = []
    for i, raw in enumerate(targets_raw, start=1):
        ceilings_before = [c for c in ceilings if sign * (c - entry) > 0 and sign * (c - raw) < 0]
        capped = min(ceilings_before, key=lambda c: abs(c - raw)) if ceilings_before else None
        level = raw
        if capped is not None and i == 1 and sign * (capped - raw) < 0:
            level = capped
            notes.append(f"TP{i} capado por nivel tecnico ({capped:.2f})")
        risk = abs(entry - sl)
        reward = abs(level - entry)
        tps.append({
            'name': f'TP{i}',
            'price': round(level, 2),
            'move_pct': round(sign * (level - entry) / entry * 100, 2),
            'rr': round(reward / risk, 2) if risk > 0 else None,
        })

    seen = set()
    tps = [t for t in tps if not (t['price'] in seen or seen.add(t['price']))]

    risk = abs(entry - sl)
    tp2 = tps[-1]['price'] if tps else entry
    rr = abs(tp2 - entry) / risk if risk > 0 else None
    if rr is not None and rr < risk_reward_min:
        notes.append(
            f"R/R estructural {rr:.2f} < {risk_reward_min}: operar solo con "
            "conviccion alta y sizing reducido"
        )

    risk_pct = risk / entry
    position_size_pct = min(100.0, 0.01 / risk_pct * 100) if risk_pct > 0 else None

    return {
        'entry': round(entry, 2),
        'stop_loss': round(sl, 2),
        'take_profits': tps,
        'risk_reward': round(rr, 2) if rr is not None else None,
        'position_size_pct': round(position_size_pct, 1) if position_size_pct else None,
        'notes': notes,
        'supports': [round(s, 2) for s in supports[-4:]],
        'resistances': [round(r, 2) for r in resistances[-4:]],
    }


def build_signal(
    ticker: str,
    df: pd.DataFrame,
    prob_up: float,
    as_of: str,
    regime: Optional[RegimeStatus] = None,
) -> SignalResult:
    """
    probabilidad calibrada + OHLCV (+ regimen) -> senal operable completa.

    Sin bandas neutras: p>=0.5 es COMPRA(+FUERTE), p<0.5 es VENTA(+FUERTE).
    Si el regimen esta suspended, la senal se emite con trading_allowed=False.
    """
    signal, side = classify_signal(prob_up)

    if regime is None:
        regime = detect_regime(df)

    price = float(df['Close'].iloc[-1])
    atr_value = float(df['atr'].iloc[-1]) if 'atr' in df.columns else np.nan
    if not np.isfinite(atr_value) or atr_value <= 0:
        tr = np.maximum(df['High'] - df['Low'],
                        np.maximum(abs(df['High'] - df['Close'].shift()),
                                   abs(df['Low'] - df['Close'].shift())))
        atr_value = float(tr.rolling(14).mean().iloc[-1]) if len(df) > 20 else np.nan

    atr_pct = atr_value / price * 100 if np.isfinite(atr_value) and price else None

    levels = build_levels(df, side, atr_value, price)

    result = SignalResult(
        ticker=ticker,
        signal=signal,
        side=side,
        confidence_pct=round(max(prob_up, 1 - prob_up) * 100, 2),
        conviction_pct=round(abs(prob_up - 0.5) * 200, 2),
        prob_up=prob_up,
        prob_down=1 - prob_up,
        price=round(price, 2),
        as_of=as_of,
        atr=round(atr_value, 2) if np.isfinite(atr_value) else None,
        atr_pct=round(atr_pct, 2) if atr_pct else None,
        levels_source='pivotes + ATR',
        regime=regime.to_dict(),
        trading_allowed=regime.trading_allowed,
    )

    result.entry = levels.get('entry')
    result.stop_loss = levels.get('stop_loss')
    result.take_profits = levels.get('take_profits', [])
    result.risk_reward = levels.get('risk_reward')
    result.supports = levels.get('supports', [])
    result.resistances = levels.get('resistances', [])
    result.notes = list(levels.get('notes', []))

    # sizing escalado por regimen
    base_size = levels.get('position_size_pct')
    if base_size is not None:
        result.position_size_pct = round(base_size * regime.position_scale, 1)
    if not regime.trading_allowed:
        result.position_size_pct = 0.0
        result.notes.append(
            "regimen SUSPENDIDO: senal emitida pero NO ejecutable hasta que "
            "el regimen se normalice (" + "; ".join(regime.reasons) + ")"
        )
    elif regime.status == 'caution':
        result.notes.append(
            "regimen CAUTION: sizing reducido 0.5x (" + "; ".join(regime.reasons) + ")"
        )

    return result
