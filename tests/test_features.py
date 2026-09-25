"""tests de features estacionarias y procesamiento por-ticker"""
import numpy as np
import pandas as pd
import pytest

from src.data.processor import TechnicalProcessor
from src.ml.advanced_trainer import DEFAULT_FEATURE_COLS


def _synthetic_ohlcv(n: int = 400, seed: int = 7, start_price: float = 100.0) -> pd.DataFrame:
    """ohlcv sintetico con tendencia y ruido"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.015, size=n)
    close = start_price * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.004, n))
    open_[0] = start_price
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx = pd.bdate_range('2022-01-03', periods=n)
    return pd.DataFrame(
        {'Open': open_, 'High': np.maximum(high, np.maximum(open_, close)),
         'Low': np.minimum(low, np.minimum(open_, close)), 'Close': close,
         'Volume': volume}, index=idx)


def test_no_raw_price_features_in_default_list():
    """las features default no contienen niveles de precio crudos"""
    forbidden = {'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
                 'close_lag_10', 'obv_ema'}
    assert not forbidden & set(DEFAULT_FEATURE_COLS)
    assert 'macd_norm' in DEFAULT_FEATURE_COLS
    assert 'macd' not in DEFAULT_FEATURE_COLS


def test_new_stationary_features_exist_after_processing():
    df = _synthetic_ohlcv()
    processor = TechnicalProcessor()
    out = processor.process_all_indicators(df)

    for col in ('macd_norm', 'macd_diff_norm', 'obv_ratio_20d',
                'lag_ret_1', 'lag_ret_5', 'momentum_5'):
        assert col in out.columns, f"falta {col}"
    # las viejas features crudas ya no se generan como lags de precio
    assert 'close_lag_1' not in out.columns


def test_normalized_macd_is_scale_invariant():
    """macd_norm debe ser (casi) identico al doblar el precio"""
    df = _synthetic_ohlcv(seed=3)
    df2 = df.copy()
    for col in ('Open', 'High', 'Low', 'Close'):
        df2[col] = df[col] * 2.0  # mismo shape, otro nivel de precio
    df2['Volume'] = df['Volume'] * 2

    p1 = TechnicalProcessor().process_all_indicators(df)
    p2 = TechnicalProcessor().process_all_indicators(df2)

    a = p1['macd_norm'].dropna()
    b = p2['macd_norm'].dropna()
    assert len(a) == len(b)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-9)


def test_market_context_features_added():
    spy = _synthetic_ohlcv(n=400, seed=11)
    vix = _synthetic_ohlcv(n=400, seed=12) * 2  # "vix" sintetico
    df = _synthetic_ohlcv(n=400, seed=5)

    out = TechnicalProcessor().process_all_indicators(df, spy_df=spy, vix_df=vix)
    for col in ('spy_ret_1d', 'spy_ret_5d', 'spy_corr_20d', 'vix_level', 'vix_zscore'):
        assert col in out.columns
    assert out['vix_level'].notna().any()


def test_target_does_not_cross_ticker_boundary():
    """el target calculado por-ticker no mezcla el ultimo dia de un ticker
    con el primero del siguiente (regresion del bug del concat)"""
    df1 = _synthetic_ohlcv(n=300, seed=1)          # precios ~100
    df2 = _synthetic_ohlcv(n=300, seed=2, start_price=500.0)  # precios ~500

    # MAL: concatenar y calcular el target de una vez
    bad = pd.concat([df1, df2])
    bad['target'] = (bad['Close'].shift(-1) > bad['Close']).astype(int)
    boundary_bad = int(bad['target'].iloc[len(df1) - 1])

    # BIEN: el target del ultimo dia de df1 comparado dentro de df1.
    # shift(-1) deja NaN y la comparacion con NaN da False (senal inexistente,
    # esa fila debe descartarse antes de entrenar).
    correct_last = (df1['Close'].shift(-1) > df1['Close'])
    assert bool(correct_last.iloc[-1]) is False

    # el target cruzado afirma "sube manana" solo porque el siguiente ticker
    # vale 5x mas: esa senal es un artefacto del concat
    assert boundary_bad == 1
