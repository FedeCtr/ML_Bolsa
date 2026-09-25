"""
Metricas financieras del backtest.

Todas las metricas obligatorias del encargo: Sharpe, Sortino, Max Drawdown,
Profit Factor, Win/Loss. Calculadas sobre retornos NETOS de costos.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(
    strategy_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    metricas de rendimiento de la estrategia.

    Args:
        strategy_returns: retornos por periodo de la estrategia (netos de costos).
        periods_per_year: 252 para diario, 252*78 para intraday, etc.
        risk_free_rate: tasa libre de riesgo ANUAL (0 para paper trading corto).
    """
    returns = pd.Series(strategy_returns).dropna()
    if len(returns) == 0:
        return {k: 0.0 for k in (
            'total_return', 'annual_return', 'annual_vol', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'n_days',
        )}

    rf_daily = risk_free_rate / periods_per_year
    excess = returns - rf_daily

    total_return = float((1 + returns).prod() - 1)
    n_periods = len(returns)
    annual_return = float((1 + total_return) ** (periods_per_year / n_periods) - 1)
    annual_vol = float(returns.std() * np.sqrt(periods_per_year))

    downside = excess[excess < 0]
    downside_dev = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0

    sharpe = float(excess.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0.0
    sortino = float(excess.mean() / downside_dev * np.sqrt(periods_per_year)) if downside_dev > 0 else 0.0

    # drawdown sobre la curva de equity acumulada
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = float(drawdown.min())

    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': float(calmar),
        'n_days': n_periods,
    }


def trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    """
    metricas por operacion.

    Args:
        trades: dataframe con columna 'return' (retorno neto de cada trade).
    """
    if trades is None or len(trades) == 0:
        return {k: 0.0 for k in (
            'n_trades', 'win_rate', 'profit_factor', 'avg_win', 'avg_loss',
            'win_loss_ratio', 'expectancy', 'max_consecutive_losses',
        )}

    rets = pd.Series(trades['return']).dropna()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    win_rate = float(len(wins) / len(rets)) if len(rets) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else float('inf') if avg_win > 0 else 0.0

    # racha maxima de perdidas consecutivas
    max_streak, streak = 0, 0
    for r in rets:
        if r <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    expectancy = float(rets.mean()) if len(rets) > 0 else 0.0

    return {
        'n_trades': int(len(rets)),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'expectancy': expectancy,
        'max_consecutive_losses': max_streak,
    }


def per_regime_metrics(
    df: pd.DataFrame,
    regime_col: str = 'vol_regime',
) -> Dict[str, Dict[str, float]]:
    """
    desglose de metricas por regimen de mercado.

    Args:
        df: dataframe con 'strategy_return' y columna de regimen.
        regime_col: columna que define el regimen (ej. terciles de VIX o
                    volatilidad realizada: 'bajo'/'medio'/'alto').
    """
    out = {}
    for regime, group in df.groupby(regime_col):
        out[str(regime)] = compute_metrics(group['strategy_return'])
        out[str(regime)]['n_days'] = int(len(group))
    return out
