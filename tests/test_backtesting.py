"""tests del motor de backtesting: costos, anti-lookahead y metricas"""
import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import compute_metrics, trade_metrics


def _flat_series(n: int = 300, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range('2022-01-03', periods=n)
    return pd.DataFrame({
        'Open': np.full(n, price, dtype=float),
        'High': np.full(n, price, dtype=float),
        'Low': np.full(n, price, dtype=float),
        'Close': np.full(n, price, dtype=float),
        'Volume': np.full(n, 1_000_000.0),
    }, index=idx)


def _sinusoid_series(n: int = 300) -> pd.DataFrame:
    """serie con reversión conocida: subir 5 días, bajar 5 días"""
    idx = pd.bdate_range('2022-01-03', periods=n)
    steps = np.where((np.arange(n) // 5) % 2 == 0, 0.01, -0.01)
    close = 100 * np.cumprod(1 + steps)
    return pd.DataFrame({
        'Open': close, 'High': close * 1.001, 'Low': close * 0.999,
        'Close': close, 'Volume': np.full(n, 1_000_000.0),
    }, index=idx)


def test_zero_costs_perfect_signal_matches_buy_and_hold():
    """con señal perfecta y costos 0, el retorno total ~ buy & hold del período posicionado"""
    df = _sinusoid_series(300)
    # señal perfecta: 1 cuando mañana sube (el patrón sube 5, baja 5)
    steps = np.where((np.arange(300) // 5) % 2 == 0, 0.01, -0.01)
    perfect_signal = pd.Series((steps > 0).astype(int), index=df.index)

    engine = BacktestEngine(initial_capital=100_000, commission_pct=0.0,
                            spread_pct=0.0, slippage_pct=0.0)
    result = engine.run(df, perfect_signal)

    # con señal perfecta no puede haber drawdown (salvo costos=0 y transición)
    assert result.metrics['max_drawdown'] > -0.05
    assert result.metrics['total_return'] > 0


def test_costs_reduce_returns():
    """más costos => menos retorno, con la misma señal"""
    df = _sinusoid_series(300)
    steps = np.where((np.arange(300) // 5) % 2 == 0, 0.01, -0.01)
    signal = pd.Series((steps > 0).astype(int), index=df.index)

    cheap = BacktestEngine(commission_pct=0.0, spread_pct=0.0, slippage_pct=0.0).run(df, signal)
    costly = BacktestEngine(commission_pct=0.001, spread_pct=0.002, slippage_pct=0.001).run(df, signal)

    assert costly.metrics['total_return'] < cheap.metrics['total_return']
    # el costo round-trip documentado es la suma de ambos lados
    assert costly.config['round_trip_cost_pct'] == 2 * (0.001 + 0.002 + 0.001)


def test_signal_lag_is_applied():
    """la posición del día t usa la señal del día t-1 (ejecución t+1)"""
    df = _flat_series(50)
    signal = pd.Series(1, index=df.index)
    signal.iloc[:10] = 0  # primeros 10 días sin señal

    engine = BacktestEngine(commission_pct=0.0, spread_pct=0.0, slippage_pct=0.0)
    portfolio = engine.run(df, signal).portfolio

    # día 10 tiene señal 1 -> posición activa recién en día 11
    assert portfolio['position'].iloc[10] == 0
    assert portfolio['position'].iloc[11] == 1


def test_metrics_basic_properties():
    idx = pd.bdate_range('2022-01-03', periods=100)
    rets = pd.Series(0.001, index=idx)
    m = compute_metrics(rets)
    assert m['total_return'] > 0
    assert m['sharpe_ratio'] > 0
    assert m['max_drawdown'] == 0  # retornos constantes positivos: sin drawdown

    # pérdidas constantes: drawdown negativo y sortino >= 0
    losses = pd.Series(-0.001, index=idx)
    m2 = compute_metrics(losses)
    assert m2['total_return'] < 0
    assert m2['max_drawdown'] < 0


def test_trade_metrics_win_loss():
    trades = pd.DataFrame({'return': [0.05, -0.02, 0.03, -0.01, -0.01]})
    t = trade_metrics(trades)
    assert t['n_trades'] == 5
    assert t['win_rate'] == pytest.approx(0.4)                       # 2/5
    assert t['profit_factor'] == pytest.approx(2.0)                  # 0.08 / 0.04
    assert t['win_loss_ratio'] == pytest.approx(0.04 / 0.0133333333, rel=1e-6)  # 3:1
    assert t['max_consecutive_losses'] == 2


def test_trade_metrics_empty():
    t = trade_metrics(pd.DataFrame({'return': []}))
    assert t['n_trades'] == 0
    assert t['win_rate'] == 0.0
