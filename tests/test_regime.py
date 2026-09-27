"""tests del detector de regimen de mercado"""
import numpy as np
import pandas as pd

from src.ml.regime import detect_regime


def _ohlcv(n=200, gap=0.0, range_mult=1.0, seed=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.010, size=n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.roll(close, 1)
    open_[0] = 100
    if gap:
        open_[-1] = close[-2] * (1 + gap)
    atr_proxy = np.std(rets) * close.mean()
    high = np.maximum(open_, close) + atr_proxy * 0.5 * range_mult
    low = np.minimum(open_, close) - atr_proxy * 0.5 * range_mult
    idx = pd.bdate_range('2023-01-02', periods=n)
    return pd.DataFrame({'Open': open_, 'High': high, 'Low': low,
                         'Close': close, 'Volume': np.full(n, 1e6)}, index=idx)


def _vix_aligned(df: pd.DataFrame, level: float) -> pd.Series:
    """serie VIX alineada al calendario del ticker (como entrega yfinance)"""
    return pd.Series(level, index=df.index)


def test_normal_regime_in_calm_market():
    df = _ohlcv()
    reg = detect_regime(df, vix_series=_vix_aligned(df, 15.0))
    assert reg.status == 'normal'
    assert reg.position_scale == 1.0
    assert reg.trading_allowed


def test_vix_panic_suspends():
    df = _ohlcv()
    reg = detect_regime(df, vix_series=_vix_aligned(df, 40.0))
    assert reg.status == 'suspended'
    assert reg.position_scale == 0.0
    assert not reg.trading_allowed
    assert any('VIX' in r for r in reg.reasons)


def test_vix_elevated_causes_caution():
    df = _ohlcv()
    reg = detect_regime(df, vix_series=_vix_aligned(df, 27.0))
    assert reg.status == 'caution'
    assert reg.position_scale == 0.5


def test_extreme_gap_suspends():
    # apertura con gap de ~4 ATR
    df = _ohlcv(n=200)
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, 199)
    base = 100 * np.cumprod(1 + rets)
    close = np.append(base, base[-1])
    open_ = np.roll(close, 1)
    open_[0] = base[0]
    open_[-1] = base[-1] * 1.15   # gap enorme
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    df2 = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close,
                        'Volume': np.full(200, 1e6)},
                       index=pd.bdate_range('2023-01-02', periods=200))
    reg = detect_regime(df2)
    assert reg.status == 'suspended'
    assert reg.gap_atr is not None and reg.gap_atr > 2.5


def test_regime_in_signal_result():
    from src.ml.signal_engine import build_signal
    df = _ohlcv()
    df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
    sig = build_signal('TEST', df, 0.62, as_of='2023-10-13')
    assert 'regime' in sig.to_dict()
    assert sig.to_dict()['regime']['status'] in ('normal', 'caution', 'suspended')
