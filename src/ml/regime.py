"""
Detector de regimen de mercado: volatilidad extrema, gaps de apertura y
riesgo macroeconomico.

La especificacion pide suspender o ajustar senales durante aperturas
violentas o eventos macro. Este modulo cuantifica el regimen con tres
senales observables y devuelve una ACCION concreta:

  - normal:    senales operan con sizing completo.
  - caution:   senales operan con sizing reducido (0.5x) y aviso visible.
  - suspended: no se emiten nuevas senales (sizing 0) hasta que el regimen
               se normalice.

Senales observadas (todas calculables con datos EOD de yfinance):
  - VIX: z-score 60d y nivel absoluto.
  - Gap de apertura del propio ticker vs ATR (apertura violenta).
  - Rango del dia vs ATR (extension intradia extrema).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# umbrales (configurables por constructor para backtesting de regimenes)
VIX_SUSPEND_LEVEL = 35.0        # pánico extendido
VIX_SUSPEND_Z = 3.0             # extremo estadístico (3 sigma)
VIX_CAUTION_LEVEL = 25.0
VIX_CAUTION_Z = 1.8
GAP_ATR_SUSPEND = 2.5           # apertura > 2.5 ATR
GAP_ATR_CAUTION = 1.5
RANGE_ATR_CAUTION = 3.0         # dia con rango > 3 ATR


@dataclass
class RegimeStatus:
    """resultado del detector de regimen para un dia/ticker"""
    status: str                  # normal | caution | suspended
    reasons: List[str] = field(default_factory=list)
    vix_level: Optional[float] = None
    vix_zscore: Optional[float] = None
    gap_atr: Optional[float] = None
    range_atr: Optional[float] = None
    position_scale: float = 1.0  # multiplicador de sizing sugerido

    @property
    def trading_allowed(self) -> bool:
        return self.status != 'suspended'

    def to_dict(self) -> Dict:
        return {
            'status': self.status,
            'reasons': self.reasons,
            'vix_level': self.vix_level,
            'vix_zscore': self.vix_zscore,
            'gap_atr': round(self.gap_atr, 2) if self.gap_atr else None,
            'range_atr': round(self.range_atr, 2) if self.range_atr else None,
            'position_scale': self.position_scale,
            'trading_allowed': self.trading_allowed,
        }


def _atr_series(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift()),
                               abs(df['Low'] - df['Close'].shift())))
    return tr.rolling(window).mean()


def detect_regime(
    df: pd.DataFrame,
    vix_series: Optional[pd.Series] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> RegimeStatus:
    """
    evalua el regimen para la ULTIMA fila de df (o `as_of`).

    Args:
        df: OHLCV del ticker (index datetime).
        vix_series: serie de cierre del VIX alineada por fecha (opcional).
        as_of: fecha a evaluar (default: ultima disponible).
    """
    reasons: List[str] = []
    status = 'normal'

    if len(df) < 20:
        return RegimeStatus(status='caution',
                            reasons=['historial insuficiente (<20 dias)'], position_scale=0.5)

    row = df.iloc[-1] if as_of is None else df.loc[:as_of].iloc[-1]

    # ---- ATR y extension del dia ----
    atr_series = _atr_series(df)
    atr_value = float(atr_series.loc[row.name]) if row.name in atr_series.index else np.nan
    day_range = float(row['High'] - row['Low'])
    range_atr = day_range / atr_value if np.isfinite(atr_value) and atr_value > 0 else None
    if range_atr is not None and range_atr >= RANGE_ATR_CAUTION:
        status = 'caution'
        reasons.append(f"rango del dia = {range_atr:.1f}x ATR (extension extrema)")

    # ---- gap de apertura vs ATR ----
    if len(df.loc[:row.name]) >= 2:
        prev_close = float(df['Close'].shift(1).loc[row.name])
        gap = abs(float(row['Open']) - prev_close)
        gap_atr = gap / atr_value if np.isfinite(atr_value) and atr_value > 0 else None
        if gap_atr is not None:
            if gap_atr >= GAP_ATR_SUSPEND:
                return RegimeStatus(
                    status='suspended',
                    reasons=reasons + [f"gap de apertura = {gap_atr:.1f}x ATR (>= {GAP_ATR_SUSPEND})"],
                    gap_atr=gap_atr, range_atr=range_atr, position_scale=0.0,
                )
            if gap_atr >= GAP_ATR_CAUTION:
                status = 'caution'
                reasons.append(f"gap de apertura = {gap_atr:.1f}x ATR")
    else:
        gap_atr = None

    # ---- VIX (regimen de mercado global) ----
    vix_level = vix_z = None
    if vix_series is not None and not vix_series.empty:
        try:
            v = vix_series.dropna()
            v_today = float(v.loc[:row.name].iloc[-1])
            v_win = v.loc[:row.name].iloc[-60:]
            if len(v_win) >= 30:
                mu, sd = float(v_win.mean()), float(v_win.std())
                vix_z = (v_today - mu) / sd if sd > 0 else None
            vix_level = v_today
            if v_today >= VIX_SUSPEND_LEVEL or (vix_z is not None and vix_z >= VIX_SUSPEND_Z):
                return RegimeStatus(
                    status='suspended',
                    reasons=reasons + [f"VIX en panico ({v_today:.1f}"
                                       + (f", z={vix_z:.1f}" if vix_z is not None else "") + ")"],
                    vix_level=vix_level, vix_zscore=vix_z,
                    gap_atr=gap_atr, range_atr=range_atr, position_scale=0.0,
                )
            if v_today >= VIX_CAUTION_LEVEL or (vix_z is not None and vix_z >= VIX_CAUTION_Z):
                status = 'caution'
                reasons.append(f"VIX elevado ({v_today:.1f}"
                               + (f", z={vix_z:.1f}" if vix_z is not None else "") + ")")
        except Exception as e:
            logger.debug(f"regimen: VIX no evaluable: {e}")

    if status == 'caution':
        return RegimeStatus(status='caution', reasons=reasons or ['condiciones elevadas'],
                            vix_level=vix_level, vix_zscore=vix_z,
                            gap_atr=gap_atr, range_atr=range_atr, position_scale=0.5)
    return RegimeStatus(status='normal', reasons=['condiciones normales'],
                        vix_level=vix_level, vix_zscore=vix_z,
                        gap_atr=gap_atr, range_atr=range_atr, position_scale=1.0)
