"""
Backtesting walk-forward con costos de transaccion e informe markdown.

En cada fold se reentrena un modelo SOLO con el pasado y se generan senales
sobre el futuro. Las senales se evaluan con un motor que ejecuta en la
apertura del dia siguiente y descuenta comision + spread + slippage.

Uso:
    python scripts/backtest.py AAPL
    python scripts/backtest.py AAPL --period 5y --splits 6
    python scripts/backtest.py AAPL --commission 0.001 --slippage 0.001
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.backtesting.engine import BacktestEngine, run_walk_forward_backtest
from src.backtesting.report import generate_report
from src.data.collector import DataCollector
from src.data.processor import TechnicalProcessor
from src.ml.advanced_trainer import DEFAULT_FEATURE_COLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_lightgbm_factory():
    """fabrica de modelos para el walk-forward.

    Contrato: recibe (X_train, y_train) y devuelve un modelo YA ENTRENADO
    con .predict_proba() (se reentrena desde cero en cada fold).
    """
    from lightgbm import LGBMClassifier

    def factory(X_tr, y_tr):
        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        return model

    return factory


def main():
    parser = argparse.ArgumentParser(description='Backtest walk-forward con costos')
    parser.add_argument('ticker', type=str, help='ticker a backtestear')
    parser.add_argument('--period', type=str, default='5y', help='periodo de datos (default: 5y)')
    parser.add_argument('--splits', type=int, default=5, help='folds walk-forward')
    parser.add_argument('--capital', type=float, default=100_000, help='capital inicial')
    parser.add_argument('--commission', type=float, default=0.0005, help='comision por lado (fraccion)')
    parser.add_argument('--spread', type=float, default=0.0005, help='spread por lado (fraccion)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='slippage por lado (fraccion)')
    parser.add_argument('--threshold', type=float, default=0.5, help='umbral de probabilidad para senal')
    parser.add_argument('--short', action='store_true', help='permitir cortos')
    parser.add_argument('--output', type=str, default=None, help='ruta del informe markdown')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"BACKTEST WALK-FORWARD: {args.ticker} ({args.period})")
    logger.info("=" * 80)

    # 1) datos + indicadores (por-ticker, con contexto de mercado)
    collector = DataCollector()
    processor = TechnicalProcessor()

    df = collector.download_ticker(args.ticker, args.period)
    if df is None or df.empty:
        logger.error(f"sin datos para {args.ticker}")
        sys.exit(1)

    spy = collector.download_ticker('SPY', args.period)
    vix = collector.download_ticker('^VIX', args.period)
    df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix)

    # target por-ticker (requerido por el walk-forward)
    df['target_direccion'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # 2) walk-forward: reentrenar en cada fold con solo el pasado
    engine_kwargs = {
        'initial_capital': args.capital,
        'commission_pct': args.commission,
        'spread_pct': args.spread,
        'slippage_pct': args.slippage,
        'allow_short': args.short,
    }

    from lightgbm import LGBMClassifier

    def model_factory(X_tr, y_tr):
        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        return model

    results = run_walk_forward_backtest(
        df,
        model_factory,
        n_splits=args.splits,
        purge_days=60,
        embargo_days=5,
        engine_kwargs=engine_kwargs,
        feature_cols=[c for c in DEFAULT_FEATURE_COLS if c in df.columns],
        ticker=args.ticker,
        threshold=args.threshold,
    )

    result = results['walk_forward']

    # 3) informe
    output_path = args.output or f"reports/backtest_{args.ticker}.md"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    generate_report(result, output_path=output_path,
                    title=f"Backtesting walk-forward: {args.ticker}")

    # resumen en consola
    m, t = result.metrics, result.trade_stats
    logger.info("\n" + "=" * 80)
    logger.info("RESULTADO (neto de costos)")
    logger.info("=" * 80)
    logger.info(f"Retorno total:      {m['total_return']:+.2%}")
    logger.info(f"Sharpe Ratio:       {m['sharpe_ratio']:.2f}")
    logger.info(f"Sortino Ratio:      {m['sortino_ratio']:.2f}")
    logger.info(f"Maximum Drawdown:   {m['max_drawdown']:.2%}")
    logger.info(f"Profit Factor:      {t['profit_factor']:.2f}")
    logger.info(f"Win Rate:           {t['win_rate']:.2%}")
    logger.info(f"Trades:             {t['n_trades']}")
    logger.info("=" * 80)
    logger.info(f"Informe completo: {output_path}")


if __name__ == '__main__':
    main()
