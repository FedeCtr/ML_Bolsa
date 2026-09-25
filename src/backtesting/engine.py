"""
Motor de backtesting walk-forward con costos de transaccion.

Convenciones anti-lookahead:
  - La senal se calcula con datos del cierre del dia t.
  - La orden se EJECUTA en la apertura del dia t+1 (signal lag = 1).
  - Los costos (comision + spread + slippage) se aplican a cada cambio de
    posicion sobre el valor operado.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .metrics import compute_metrics, per_regime_metrics, trade_metrics

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """resultado completo de un backtest"""
    portfolio: pd.DataFrame          # curva diaria: equity, returns, posicion
    trades: pd.DataFrame             # operaciones cerradas
    metrics: Dict                    # metricas de cartera (sharpe, sortino, ...)
    trade_stats: Dict                # metricas por operacion
    regime_metrics: Dict             # desglose por regimen de volatilidad
    config: Dict                     # parametros usados (costos, capital, ...)
    equity_curve: pd.Series = field(default=None)


class BacktestEngine:
    """
    motor de backtest simple y estricto para la senal del modelo.

    Estrategia evaluada: larga el dia siguiente si la senal es 1 (subida),
    plano (o corto si `allow_short`) si la senal es 0. La senal del dia t se
    ejecuta en la apertura del dia t+1; el retorno del trade va de la
    apertura de t+1 a la apertura del dia siguiente en que se cambie de
    posicion (o al cierre del ultimo dia).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.0005,      # 5 bps por lado (ej. Interactive Brokers)
        spread_pct: float = 0.0005,          # 5 bps de spread efectivo por lado
        slippage_pct: float = 0.0005,        # 5 bps de slippage por lado
        allow_short: bool = False,
        risk_free_rate: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.allow_short = allow_short
        self.risk_free_rate = risk_free_rate

    @property
    def one_side_cost(self) -> float:
        """costo fraccional por lado (entrada o salida)"""
        return self.commission_pct + self.spread_pct + self.slippage_pct

    def run(
        self,
        df: pd.DataFrame,
        signal: pd.Series,
        ticker: str = 'unknown',
    ) -> BacktestResult:
        """
        ejecuta el backtest.

        Args:
            df: OHLCV indexado por fecha (debe incluir 'Open' y 'Close').
            signal: senal binaria {0,1} indexada por la fecha de GENERACION
                    (se ejecuta en la apertura del dia siguiente).
            ticker: nombre informativo.
        """
        df = df.copy()
        signal = signal.reindex(df.index).fillna(0)

        # costo total ida y vuelta (entrada + salida)
        round_trip_cost = 2 * self.one_side_cost

        # posicion decidida el dia t -> vigente durante t+1
        position = signal.shift(1)
        if not self.allow_short:
            position = position.clip(lower=0)  # solo 0/1
        else:
            position = position * 2 - 1        # -1/0/+1

        # retorno del dia t+1 medido de apertura a apertura:
        # r_t+1 = Open_t+2 / Open_t+1 - 1... pero el dato disponible tras
        # ejecutar en la apertura de t+1 es: entrada Open_t+1, salida en la
        # proxima apertura tras cambiar de senal. Aproximacion estandar:
        # retorno diario de Open a Open.
        open_to_open = df['Open'].pct_change()

        # el trade de la senal generada en t se materializa como:
        # comprar en apertura t+1, vender en apertura t+2 -> retorno
        # open_to_open de t+2 con posicion de t+1.
        position = position.fillna(0)
        gross_return = open_to_open * position.shift(1).fillna(0)

        # costos: cada cambio de posicion (0->1 o 1->0) paga el round trip
        # completo (entrada + salida) sobre el valor operado
        pos_change = position.diff().abs().fillna(0)
        cost_return = pos_change * round_trip_cost

        strategy_return = gross_return - cost_return

        portfolio = pd.DataFrame({
            'Open': df['Open'],
            'Close': df['Close'],
            'signal': signal,
            'position': position,
            'gross_return': gross_return.fillna(0),
            'cost': cost_return.fillna(0),
            'strategy_return': strategy_return.fillna(0),
        }, index=df.index)

        # regimen de volatilidad: terciles de vol realizada 20d
        realized_vol = df['Close'].pct_change().rolling(20).std()
        try:
            portfolio['vol_regime'] = pd.qcut(realized_vol, q=3, labels=['bajo', 'medio', 'alto'], duplicates='drop')
        except ValueError:
            portfolio['vol_regime'] = 'unico'

        # equity
        portfolio['equity'] = self.initial_capital * (1 + portfolio['strategy_return']).cumprod()

        # construir trades cerrados (cada bloque de posicion constante = 1 trade)
        trades = self._extract_trades(portfolio)

        metrics = compute_metrics(portfolio['strategy_return'], risk_free_rate=self.risk_free_rate)
        trade_stats = trade_metrics(trades)
        regime_metrics = per_regime_metrics(portfolio.dropna(subset=['strategy_return']))

        config = {
            'ticker': ticker,
            'initial_capital': self.initial_capital,
            'commission_pct': self.commission_pct,
            'spread_pct': self.spread_pct,
            'slippage_pct': self.slippage_pct,
            'round_trip_cost_pct': round_trip_cost,
            'allow_short': self.allow_short,
            'start': str(df.index[0].date()),
            'end': str(df.index[-1].date()),
            'n_days': int(len(df)),
        }

        logger.info(
            f"backtest {ticker}: total {metrics['total_return']:+.2%}, "
            f"sharpe {metrics['sharpe_ratio']:.2f}, "
            f"maxDD {metrics['max_drawdown']:.2%}, "
            f"trades {trade_stats['n_trades']}, "
            f"PF {trade_stats['profit_factor']:.2f}"
        )

        return BacktestResult(
            portfolio=portfolio,
            trades=trades,
            metrics=metrics,
            trade_stats=trade_stats,
            regime_metrics=regime_metrics,
            config=config,
            equity_curve=portfolio['equity'],
        )

    def _extract_trades(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """agrupa la curva en trades cerrados con su retorno neto"""
        pos = portfolio['position']
        blocks = (pos != pos.shift()).cumsum()

        trades = []
        for _, group in portfolio.groupby(blocks):
            p = group['position'].iloc[0]
            if p == 0:
                continue  # fuera del mercado no es un trade
            entry_idx = group.index[0]
            exit_idx = group.index[-1]
            entry_px = group['Open'].iloc[0] * (1 + self.one_side_cost)
            exit_px = group['Close'].iloc[-1] * (1 - self.one_side_cost)
            gross = (exit_px / entry_px - 1) * p
            trades.append({
                'entry_date': entry_idx,
                'exit_date': exit_idx,
                'direction': 'largo' if p > 0 else 'corto',
                'entry_price': entry_px,
                'exit_price': exit_px,
                'return': gross,
                'days_held': len(group),
            })

        return pd.DataFrame(trades)


def run_walk_forward_backtest(
    df: pd.DataFrame,
    predictor_fn,
    n_splits: int = 5,
    purge_days: int = 60,
    embargo_days: int = 5,
    engine_kwargs: Optional[Dict] = None,
    feature_cols: Optional[List[str]] = None,
    ticker: str = 'walk_forward',
    threshold: float = 0.5,
) -> Dict[str, BacktestResult]:
    """
    backtest walk-forward: en cada fold se reentrena el modelo SOLO con el
    pasado y se generan senales sobre el futuro, que luego se evaluan con
    costos. Sin leakage entre folds.

    Args:
        df: dataframe con features + OHLCV + target_direccion (una serie temporal).
        predictor_fn: funcion (X_train, y_train) -> modelo con .predict_proba(X).
        n_splits: folds.
        purge_days / embargo_days: purga y embargo del split.
        engine_kwargs: argumentos para BacktestEngine (costos, capital...).
        feature_cols: columnas a usar como features. Por defecto, todo lo que
                      no es OHLCV/target/ticker (evita colar features crudas).
        ticker: nombre informativo para el informe.
    """
    from ..ml.cv import build_walk_forward_splits

    engine_kwargs = engine_kwargs or {}
    engine = BacktestEngine(**engine_kwargs)

    all_signals = pd.Series(0, index=df.index)

    exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'target_direccion', 'ticker'}
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df['target_direccion']

    # mascara posicional (robusta tambien con indices duplicados)
    valid = ~(X.isna().any(axis=1) | y.isna()).to_numpy()
    X, y = X.loc[valid], y.loc[valid]

    splits = build_walk_forward_splits(len(X), n_splits=n_splits, gap=purge_days, embargo=embargo_days)

    for i, (train_idx, test_idx) in enumerate(splits):
        model = predictor_fn(X.iloc[train_idx], y.iloc[train_idx])

        proba = model.predict_proba(X.iloc[test_idx])
        idx_one = int(np.where(model.classes_ == 1)[0][0])
        sig = (proba[:, idx_one] >= threshold).astype(int)

        dates = X.iloc[test_idx].index
        all_signals.loc[dates] = sig
        logger.info(f"fold {i}: {len(test_idx)} dias, senales largas: {int(sig.sum())}")

    result = engine.run(df, all_signals, ticker=ticker)
    return {'walk_forward': result}
