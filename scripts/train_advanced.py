"""
Entrena el ensemble avanzado con validacion temporal honesta (SIN leakage).

Correcciones respecto a la version anterior:
  - Cada ticker se procesa POR SEPARADO (indicadores + target) ANTES de
    concatenar. Antes se concatenaban raw y luego se calculaban rolling
    windows y shift(-1): las features de un ticker se contaminaban con el
    ultimo dia del ticker anterior y el target cruzaba la frontera.
  - Split walk-forward con purge (60d) + embargo (5d). Antes: shuffle
    aleatorio = metricas infladas.
  - Optuna evalua con la misma CV purgada.
  - Las probabilidades OOF se guardan para calibrar el umbral de confianza
    con datos reales (scripts/calibrate_thresholds.py).

Uso:
    python scripts/train_advanced.py                        # rapido
    python scripts/train_advanced.py --optimize --trials 50 # con optuna
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import pandas as pd

from src.data.collector import DataCollector
from src.data.processor import TechnicalProcessor
from src.ml.advanced_trainer import AdvancedEnsemble
from src.utils.logger import get_logger

logger = get_logger(__name__)

# tickers de contexto de mercado: el propio market (SPY) y el fear index (VIX)
MARKET_TICKER = 'SPY'
VIX_TICKER = '^VIX'


def add_target_direction(df: pd.DataFrame) -> pd.DataFrame:
    """target binario: 1 si manana cierra por encima de hoy (por ticker).

    IMPORTANTE: se calcula por-ticker, antes de concatenar. Si se calcula
    sobre el dataframe combinado, el target del ultimo dia de un ticker usa
    el primer dia del siguiente (frontera cruzada).
    """
    df = df.copy()
    df['target_direccion'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def prepare_ticker_frame(
    collector: DataCollector,
    processor: TechnicalProcessor,
    ticker: str,
    period: str,
    context: dict,
) -> pd.DataFrame | None:
    """descarga, calcula indicadores y target para UN ticker (sin cruce)"""
    logger.info(f"  procesando {ticker}...")
    df = collector.download_ticker(ticker, period)
    if df is None or df.empty:
        logger.warning(f"  {ticker}: sin datos")
        return None

    df = processor.process_all_indicators(df, spy_df=context.get('spy'), vix_df=context.get('vix'))
    df = add_target_direction(df)
    df['ticker'] = ticker
    return df


def main():
    parser = argparse.ArgumentParser(description='Entrena ensemble avanzado (validacion purgada)')
    parser.add_argument('--optimize', action='store_true', help='optimizar con optuna sobre CV purgada')
    parser.add_argument('--trials', type=int, default=30, help='trials de optuna por modelo')
    parser.add_argument('--n-splits', type=int, default=5, help='folds walk-forward')
    parser.add_argument('--tickers', nargs='+',
                        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                        help='tickers para entrenar')
    parser.add_argument('--period', type=str, default='3y', help='periodo de datos (default: 3y)')
    parser.add_argument('--no-market-context', action='store_true',
                        help='no anadir features de SPY/VIX')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ENTRENAMIENTO AVANZADO - validacion walk-forward purgada (sin leakage)")
    logger.info("=" * 80)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Periodo: {args.period} | Folds: {args.n_splits} | Optuna: {args.optimize}")
    logger.info("=" * 80)

    collector = DataCollector()
    processor = TechnicalProcessor()

    # 1) contexto de mercado (SPY + VIX), sin target
    context = {}
    if not args.no_market_context:
        logger.info("\n[1/3] DESCARGANDO CONTEXTO DE MERCADO (SPY, VIX)...")
        spy = collector.download_ticker(MARKET_TICKER, args.period)
        vix = collector.download_ticker(VIX_TICKER, args.period)
        context = {'spy': spy, 'vix': vix}
        if spy is None or vix is None:
            logger.warning("no se pudo descargar SPY/VIX; se continuan sin contexto de mercado")
            context = {}

    # 2) procesar cada ticker POR SEPARADO y luego concatenar
    logger.info("\n[2/3] PROCESANDO TICKERS (indicadores + target por ticker)...")
    frames = []
    for ticker in args.tickers:
        df = prepare_ticker_frame(collector, processor, ticker, args.period, context)
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("no se pudo procesar ningun ticker")
        sys.exit(1)

    df_combined = pd.concat(frames, axis=0)
    logger.info(f"dataset combinado: {len(df_combined)} filas, {df_combined['ticker'].nunique()} tickers")

    # 3) entrenar (split interno walk-forward; no usa el orden de concat)
    logger.info("\n[3/3] ENTRENANDO ENSEMBLE...")
    trainer = AdvancedEnsemble()

    data_start = str(min(df.index.min() for df in frames).date())
    data_end = str(max(df.index.max() for df in frames).date())

    results = trainer.train(
        df_combined,
        optimize=args.optimize,
        n_trials=args.trials,
        n_splits=args.n_splits,
        tickers=args.tickers,
        data_start=data_start,
        data_end=data_end,
    )

    # guardar OOF para calibracion de umbrales
    trainer.save_oof()

    logger.info("\n" + "=" * 80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"Directional Accuracy (walk-forward): {results['accuracy']*100:.2f}%")
    logger.info(f"Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f} | F1: {results['f1']:.4f}")
    logger.info(f"Features:  {results['n_features']} | Muestras: {results['n_samples']}")
    for name, acc in results.get('individual_accuracy', {}).items():
        logger.info(f"  {name:15s}: {acc*100:.2f}%")
    logger.info("=" * 80)
    logger.info("Nota: el accuracy honesto suele ser menor al reportado antes de la")
    logger.info("correccion de leakage; es el numero real explotable.")
    logger.info("")
    logger.info("Siguientes pasos:")
    logger.info("  1. python scripts/calibrate_thresholds.py   # umbral con datos, no al azar")
    logger.info("  2. python scripts/backtest.py AAPL          # backtest con costos")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
