"""
Tests del simulador de expectativa (Sprint 3) y del contrato R/R 1:2.

Usa fixtures SINTETICOS (dataframes y OHLC artificiales): no dependen de
models/advanced_oof.pkl ni de los CSV reales, asi que corren rapido y en CI.
"""
import numpy as np
import pandas as pd
import pytest

from src.ml.expectancy import (
    _resolve_trade_walk,
    attach_market_data,
    load_oof_dataset,
    simulate_expectancy,
)
from src.ml.signal_engine import build_levels


# ----------------------------------------------------------------------
# Contrato de niveles: R/R 1:2 garantizado
# ----------------------------------------------------------------------

def _mk_df(n=120, base=100.0, seed=7):
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, base * 0.01, n))
    high = close * (1 + rng.uniform(0.002, 0.012, n))
    low = close * (1 - rng.uniform(0.002, 0.012, n))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        'Open': open_, 'High': np.maximum(high, open_),
        'Low': np.minimum(low, open_), 'Close': close,
        'Volume': np.full(n, 1e6),
    })


def test_levels_rr_min_guaranteed_long():
    """TP2 largo debe cumplir R/R >= 2 aunque haya resistencias antes."""
    df = _mk_df()
    atr = 2.0
    price = float(df['Close'].iloc[-1])
    out = build_levels(df, 'largo', atr, price, risk_reward_min=2.0)
    assert out['entry'] is not None and out['stop_loss'] is not None
    assert len(out['take_profits']) == 2
    risk = out['entry'] - out['stop_loss']
    tp2 = out['take_profits'][-1]
    assert tp2['price'] > out['entry']
    assert tp2['rr'] >= 2.0 - 1e-6


def test_levels_rr_min_guaranteed_short():
    df = _mk_df()
    atr = 2.0
    price = float(df['Close'].iloc[-1])
    out = build_levels(df, 'corto', atr, price, risk_reward_min=2.0)
    risk = out['stop_loss'] - out['entry']
    tp2 = out['take_profits'][-1]
    assert tp2['price'] < out['entry']
    assert tp2['rr'] >= 2.0 - 1e-6


def test_levels_tp1_equals_1r_when_atr_small():
    """Con ATR chico, TP1 = 1R (no puede quedar por debajo del riesgo)."""
    df = _mk_df()
    atr = 0.1   # ATR minimo frente a un SL de ~1-5% del precio
    price = float(df['Close'].iloc[-1])
    out = build_levels(df, 'largo', atr, price, risk_reward_min=2.0)
    tp1 = out['take_profits'][0]
    assert tp1['rr'] >= 1.0 - 1e-6


# ----------------------------------------------------------------------
# Resolvedor walk (SL-first conservador)
# ----------------------------------------------------------------------

def test_resolve_walk_sl_first_on_tie():
    """Si el mismo dia toca SL y TP, cuenta SL (conservador)."""
    highs = np.array([110.0])
    lows = np.array([90.0])
    closes = np.array([100.0])
    r, held, outcome = _resolve_trade_walk('largo', 100.0, 90.0, 110.0,
                                           highs, lows, closes, 2.0, 20)
    assert outcome == 'sl'
    assert held == 1
    assert r <= -1.0


def test_resolve_walk_tp_hit():
    highs = np.array([100.0, 111.0])
    lows = np.array([96.0, 99.0])
    closes = np.array([100.0, 107.0])
    # entrada en 100, SL 95 (5%), TP 110 (10% => 2R); el dia 2 el high 111 toca TP
    r, held, outcome = _resolve_trade_walk('largo', 100.0, 95.0, 110.0,
                                           highs, lows, closes, 2.0, 20)
    assert outcome == 'tp'
    assert held == 2
    assert r == pytest.approx(2.0, abs=0.05)


def test_resolve_walk_timeout_exits_at_close():
    highs = np.array([101.0, 102.0, 103.0])
    lows = np.array([99.0, 100.0, 101.0])
    closes = np.array([100.5, 101.5, 102.5])
    r, held, outcome = _resolve_trade_walk('largo', 100.0, 95.0, 110.0,
                                           highs, lows, closes, 2.0, 3)
    assert outcome == 'timeout'
    assert held == 3
    # cierre 102.5 con riesgo 5 => +0.5R aprox menos costos
    assert 0.3 < r < 0.6


def test_resolve_walk_short_side():
    highs = np.array([100.0, 98.0])
    lows = np.array([95.0, 89.0])
    closes = np.array([96.0, 90.0])
    # corto: entrada 100, SL 105, TP 90 (2R con riesgo 5)
    r, held, outcome = _resolve_trade_walk('corto', 100.0, 105.0, 90.0,
                                           highs, lows, closes, 2.0, 20)
    assert outcome == 'tp'
    assert held == 2


# ----------------------------------------------------------------------
# Pipeline OOF con fixtures sinteticos (monkeypatch de rutas)
# ----------------------------------------------------------------------

class _FakeConfig:
    models_dir = '/nonexistent'
    raw_data_dir = '/nonexistent'


def test_load_oof_dataset_legacy_block_reconstruction(monkeypatch, tmp_path):
    """OOF sin vector de tickers: reconstruccion por bloques identificados
    contra el OHLC local (bloques con mismatch>10% se excluyen)."""
    # dos tickers con patrones de subida/bajada DISTINTOS (oscilaciones de
    # fase/frecuencia diferentes) para que la identificacion por targets sea
    # unica; el segundo bloque EMPIEZA con fechas anteriores al fin del
    # primero (retroceso => nuevo bloque, como en el OOF real por folds)
    dates_tsla = pd.bdate_range('2024-01-01', periods=30).strftime('%Y-%m-%d')
    dates_aapl = pd.bdate_range('2024-01-02', periods=25).strftime('%Y-%m-%d')
    close_tsla = 200 + 30 * np.sin(np.linspace(0, 6.5, 30))
    close_aapl = 180 + 12 * np.sin(np.linspace(2.1, 9.3, 25))

    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    for name, dates, close in [('TSLA', dates_tsla, close_tsla),
                               ('AAPL', dates_aapl, close_aapl)]:
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates).tz_localize('UTC'),
            'Open': close, 'High': close * 1.01,
            'Low': close * 0.99, 'Close': close, 'Volume': 1e6,
        })
        df.to_csv(raw_dir / f'{name}_raw.csv', index=False)

    # OOF: bloque 0 = TSLA (fechas ascendentes), bloque 1 = AAPL
    probs = np.r_[np.full(29, 0.6), np.full(24, 0.4)]
    targets = np.r_[(close_tsla[1:] > close_tsla[:-1]).astype(int),
                    (close_aapl[1:] > close_aapl[:-1]).astype(int)]
    dates = list(dates_tsla[:-1]) + list(dates_aapl[:-1])
    oof_path = tmp_path / 'oof.pkl'
    meta_path = tmp_path / 'meta.pkl'
    import joblib
    joblib.dump({'probabilities': probs, 'targets': targets,
                 'dates': dates, 'tickers': None}, oof_path)
    joblib.dump({'tickers': ['TSLA', 'AAPL']}, meta_path)

    monkeypatch.setattr('src.ml.expectancy.Config', _FakeConfig)
    df = load_oof_dataset(oof_path=str(oof_path), metadata_path=str(meta_path),
                          raw_dir=str(raw_dir))
    assert set(df['ticker']) == {'TSLA', 'AAPL'}
    assert len(df) == 53   # todas las filas se identificaron sin perdida


def test_load_oof_fast_path_uses_saved_tickers(monkeypatch, tmp_path):
    """OOF con vector de tickers consistente: camino rapido sin reconstruccion."""
    import joblib
    probs = np.array([0.6, 0.4, 0.55, 0.45])
    targets = np.array([1, 0, 1, 0])
    dates = ['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03']
    tickers = ['AAPL', 'MSFT', 'AAPL', 'MSFT']
    oof_path = tmp_path / 'oof.pkl'
    meta_path = tmp_path / 'meta.pkl'
    joblib.dump({'probabilities': probs, 'targets': targets,
                 'dates': dates, 'tickers': tickers}, oof_path)
    joblib.dump({'tickers': ['AAPL', 'MSFT']}, meta_path)
    monkeypatch.setattr('src.ml.expectancy.Config', _FakeConfig)

    df = load_oof_dataset(oof_path=str(oof_path), metadata_path=str(meta_path),
                          raw_dir=str(tmp_path))
    assert df['ticker'].tolist() == tickers
    assert len(df) == 4


def test_attach_and_simulate_synthetic(monkeypatch, tmp_path):
    """Simulacion end-to-end sobre un mercado sintetico en tendencia alcista:
    el sistema largo con senal a favor debe tener expectativa positiva."""
    rng = np.random.default_rng(11)
    n = 200
    drift = 0.002
    close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))
    high = close * (1 + rng.uniform(0.003, 0.015, n))
    low = close * (1 - rng.uniform(0.003, 0.015, n))
    open_ = np.r_[close[0], close[:-1]]

    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    dates = pd.bdate_range('2024-01-01', periods=n).strftime('%Y-%m-%d')
    pd.DataFrame({
        'Date': pd.to_datetime(dates).tz_localize('UTC'),
        'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': 1e6,
    }).to_csv(raw_dir / 'AAPL_raw.csv', index=False)
    # VIX bajo constante (no bloquea) y SPY alcista
    pd.DataFrame({
        'Date': pd.to_datetime(dates).tz_localize('UTC'),
        'Close': np.full(n, 14.0),
    }).to_csv(raw_dir / '^VIX_raw.csv', index=False)
    pd.DataFrame({
        'Date': pd.to_datetime(dates).tz_localize('UTC'),
        'Close': np.linspace(400, 500, n),
    }).to_csv(raw_dir / 'SPY_raw.csv', index=False)

    probs = np.r_[np.full(n - 1, 0.62), np.nan]
    targets = (np.r_[close[1:], np.nan] > close).astype(float)
    df = pd.DataFrame({
        'date': dates, 'ticker': 'AAPL',
        'prob_up': np.nan_to_num(probs), 'target': np.nan_to_num(targets),
    })
    monkeypatch.setattr('src.ml.expectancy.Config', _FakeConfig)
    df = attach_market_data(df, raw_dir=str(raw_dir))
    assert (df['pos_t'] >= 0).sum() >= n - 20   # ATR necesita 14 barras

    res = simulate_expectancy(df, prob_min=0.60, rr_multiple=2.0,
                              allow_short=False, vix_max=30,
                              one_position_per_ticker=True,
                              raw_dir=str(raw_dir))
    # con 1 posicion por ticker y holding multi-dia, el numero de trades es
    # modesto pero debe haberlos, y con drift alcista + senal larga, ganancia
    assert res.n_trades >= 5
    assert res.expectancy_r > 0.2
    assert res.profit_factor > 1.3
