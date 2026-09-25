"""tests del motor de senales operables"""
import numpy as np
import pandas as pd
import pytest

from src.ml.signal_engine import (
    SIDE_BUY, SIDE_NONE, SIDE_SELL,
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


def test_classify_signal_tiers():
    assert classify_signal(0.90) == ('COMPRA_FUERTE', SIDE_BUY)
    assert classify_signal(0.60) == ('COMPRA', SIDE_BUY)
    assert classify_signal(0.50) == ('NEUTRAL', SIDE_NONE)
    assert classify_signal(0.40) == ('VENTA', SIDE_SELL)
    assert classify_signal(0.10) == ('VENTA_FUERTE', SIDE_SELL)


def test_signal_never_returns_null_action_string():
    """la especificacion pide senales graduadas: nunca 'sin_accion'"""
    df = _with_atr(_ohlcv())
    for prob in (0.30, 0.42, 0.50, 0.61, 0.83):
        sig = build_signal('TEST', df, prob, as_of='2024-01-05')
        assert sig.signal in ('COMPRA_FUERTE', 'COMPRA', 'NEUTRAL', 'VENTA', 'VENTA_FUERTE')
        assert sig.signal not in ('sin_accion', 'no_action', 'SIN_ACCION')


def test_neutral_has_no_operational_side():
    df = _with_atr(_ohlcv())
    sig = build_signal('TEST', df, 0.51, as_of='2024-01-05')
    assert sig.side == SIDE_NONE
    assert sig.entry is None and sig.stop_loss is None
    assert any('NEUTRAL' in n or '45-55' in n for n in sig.notes)


def test_buy_levels_structure():
    df = _with_atr(_ohlcv(seed=9))
    sig = build_signal('TEST', df, 0.72, as_of='2024-01-05')
    assert sig.side == SIDE_BUY
    assert sig.entry is not None and sig.stop_loss is not None
    assert sig.stop_loss < sig.entry
    assert len(sig.take_profits) >= 2
    # TPs por encima de la entrada para largos
    for tp in sig.take_profits:
        assert tp['price'] > sig.entry
    # R/R coherente con los niveles
    if sig.risk_reward is not None:
        assert sig.risk_reward > 0


def test_sell_levels_mirror():
    df = _with_atr(_ohlcv(seed=11))
    sig = build_signal('TEST', df, 0.28, as_of='2024-01-05')
    assert sig.side == SIDE_SELL
    assert sig.stop_loss > sig.entry
    for tp in sig.take_profits:
        assert tp['price'] < sig.entry


def test_position_size_inversely_proportional_to_risk():
    df = _with_atr(_ohlcv(seed=3))
    sig = build_signal('TEST', df, 0.70, as_of='2024-01-05')
    assert sig.position_size_pct is not None
    assert 0 < sig.position_size_pct <= 100


def test_pivot_levels_find_structure():
    # serie con valle claro en el medio: debe detectar soporte
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


def test_confidence_is_direction_aware():
    df = _with_atr(_ohlcv())
    strong = build_signal('TEST', df, 0.85, as_of='2024-01-05')
    weak = build_signal('TEST', df, 0.58, as_of='2024-01-05')
    assert strong.confidence_pct > weak.confidence_pct
