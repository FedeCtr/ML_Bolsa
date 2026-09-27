"""tests del motor de senales operables (v2: sin bandas neutras)"""
import numpy as np
import pandas as pd
import pytest

from src.ml.regime import RegimeStatus
from src.ml.signal_engine import (
    SIDE_BUY, SIDE_SELL,
    build_levels, build_signal, classify_signal, find_pivot_levels,
)


def _ohlcv(n: int = 300, seed: int = 5, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.014, size=n)
    close = base * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.007, n)))
    open_ = np.roll(close, 1)
    open_[0] = base
    idx = pd.bdate_range('2023-01-02', periods=n)
    return pd.DataFrame({'Open': open_, 'High': np.maximum(high, open_),
                         'Low': np.minimum(low, open_), 'Close': close,
                         'Volume': rng.integers(1e6, 5e6, n).astype(float)}, index=idx)


def _with_atr(df: pd.DataFrame) -> pd.DataFrame:
    tr = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift()),
                               abs(df['Low'] - df['Close'].shift())))
    df = df.copy()
    df['atr'] = tr.rolling(14).mean()
    return df


def test_classify_signal_always_operable():
    """spec: solo 4 categorias operables, nunca NEUTRAL/sin accion"""
    for prob in (0.30, 0.42, 0.4999, 0.50, 0.501, 0.61, 0.83):
        signal, side = classify_signal(prob)
        assert signal in ('COMPRA_FUERTE', 'COMPRA', 'VENTA', 'VENTA_FUERTE')
        assert side in (SIDE_BUY, SIDE_SELL)
    assert classify_signal(0.90) == ('COMPRA_FUERTE', SIDE_BUY)
    assert classify_signal(0.56) == ('COMPRA_FUERTE', SIDE_BUY)   # >= 0.55
    assert classify_signal(0.52) == ('COMPRA', SIDE_BUY)          # 0.50-0.55
    assert classify_signal(0.47) == ('VENTA', SIDE_SELL)          # 0.45-0.50
    assert classify_signal(0.44) == ('VENTA_FUERTE', SIDE_SELL)   # <= 0.45
    assert classify_signal(0.10) == ('VENTA_FUERTE', SIDE_SELL)


def test_conviction_graduates_strength():
    df = _with_atr(_ohlcv())
    strong = build_signal('TEST', df, 0.85, as_of='2024-01-05')
    weak = build_signal('TEST', df, 0.53, as_of='2024-01-05')
    assert strong.signal == 'COMPRA_FUERTE' and weak.signal == 'COMPRA'
    assert strong.conviction_pct == 70.0 and weak.conviction_pct == 6.0
    assert strong.confidence_pct > weak.confidence_pct


def test_every_signal_has_levels():
    df = _with_atr(_ohlcv(seed=9))
    for prob in (0.35, 0.50, 0.65, 0.80):
        sig = build_signal('TEST', df, prob, as_of='2024-01-05')
        assert sig.entry is not None and sig.stop_loss is not None
        assert len(sig.take_profits) >= 1


def test_buy_levels_structure():
    df = _with_atr(_ohlcv(seed=9))
    sig = build_signal('TEST', df, 0.72, as_of='2024-01-05')
    assert sig.side == SIDE_BUY
    assert sig.stop_loss < sig.entry
    for tp in sig.take_profits:
        assert tp['price'] > sig.entry
    if sig.risk_reward is not None:
        assert sig.risk_reward > 0


def test_sell_levels_mirror():
    df = _with_atr(_ohlcv(seed=11))
    sig = build_signal('TEST', df, 0.28, as_of='2024-01-05')
    assert sig.side == SIDE_SELL
    assert sig.stop_loss > sig.entry
    for tp in sig.take_profits:
        assert tp['price'] < sig.entry


def test_suspended_regime_marks_not_tradable():
    df = _with_atr(_ohlcv())
    reg = RegimeStatus(status='suspended', reasons=['VIX en panico (40.0)'],
                       position_scale=0.0)
    sig = build_signal('TEST', df, 0.70, as_of='2024-01-05', regime=reg)
    assert sig.trading_allowed is False
    assert sig.position_size_pct == 0.0
    assert any('SUSPENDIDO' in n for n in sig.notes)
    # la senal se EMITE igualmente (con marca), no se elimina
    assert sig.signal == 'COMPRA_FUERTE'


def test_caution_regime_halves_sizing():
    df = _with_atr(_ohlcv())
    normal = RegimeStatus(status='normal', position_scale=1.0)
    caution = RegimeStatus(status='caution', position_scale=0.5)
    s1 = build_signal('TEST', df, 0.70, as_of='2024-01-05', regime=normal)
    s2 = build_signal('TEST', df, 0.70, as_of='2024-01-05', regime=caution)
    assert s2.position_size_pct == round(s1.position_size_pct * 0.5, 1)


def test_position_size_positive_and_bounded():
    df = _with_atr(_ohlcv(seed=3))
    sig = build_signal('TEST', df, 0.70, as_of='2024-01-05')
    assert 0 < sig.position_size_pct <= 100


def test_pivot_levels_find_structure():
    n = 200
    x = np.linspace(0, 4 * np.pi, n)
    close = 100 + 10 * np.sin(x)
    df = pd.DataFrame({
        'Open': close, 'High': close + 0.5, 'Low': close - 0.5, 'Close': close,
        'Volume': np.full(n, 1e6),
    }, index=pd.bdate_range('2023-01-02', periods=n))
    supports, resistances = find_pivot_levels(df['High'], df['Low'], left=3, right=3)
    assert len(supports) >= 1 and len(resistances) >= 1
    assert min(supports) < 100 < max(resistances)
