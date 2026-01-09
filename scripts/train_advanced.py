"""
Script para entrenar el sistema avanzado de ensemble
Objetivo: 70%+ accuracy

Uso:
    python scripts/train_advanced.py --optimize --trials 100
"""
import sys
import os
from pathlib import Path

# agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.data.processor import TechnicalProcessor
from src.ml.advanced_trainer import AdvancedEnsemble
from src.utils.logger import get_logger
import argparse

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Entrena ensemble avanzado')
    parser.add_argument('--optimize', action='store_true', help='optimizar hyperparametros con optuna')
    parser.add_argument('--trials', type=int, default=50, help='numero de trials para optuna (default: 50)')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], 
                        help='tickers para entrenar')
    parser.add_argument('--period', type=str, default='2y', help='periodo de datos (default: 2y)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ENTRENAMIENTO AVANZADO - Sistema Ensemble para 70%+ accuracy")
    logger.info("="*80)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Periodo: {args.period}")
    logger.info(f"Optimizacion: {'SI' if args.optimize else 'NO'}")
    if args.optimize:
        logger.info(f"Trials Optuna: {args.trials}")
    logger.info("="*80)
    
    # 1. recolectar datos
    logger.info("\n[1/3] RECOLECTANDO DATOS...")
    collector = DataCollector()
    
    dfs = []
    for ticker in args.tickers:
        logger.info(f"  descargando {ticker}...")
        df = collector.download_ticker(ticker, args.period)
        if df is not None and not df.empty:
            dfs.append(df)
        else:
            logger.warning(f"  {ticker} fallo")
    
    if not dfs:
        logger.error("no se pudo descargar ningun ticker")
        return
    
    # combinar
    import pandas as pd
    df_combined = pd.concat(dfs, axis=0)
    logger.info(f"  total: {len(df_combined)} registros de {len(dfs)} tickers")
    
    # 2. calcular indicadores
    logger.info("\n[2/3] CALCULANDO 50+ INDICADORES TECNICOS...")
    processor = TechnicalProcessor()
    df_processed = processor.process_all_indicators(df_combined)
    logger.info(f"  features: {df_processed.shape[1]} columnas")
    
    # 3. entrenar ensemble
    logger.info("\n[3/3] ENTRENANDO ENSEMBLE DE 4 MODELOS...")
    trainer = AdvancedEnsemble()
    
    results = trainer.train(
        df_processed,
        optimize=args.optimize,
        n_trials=args.trials
    )
    
    # resumen final
    logger.info("\n" + "="*80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)
    logger.info(f"Accuracy Test:  {results['test_accuracy']*100:.2f}%")
    logger.info(f"Precision:      {results['precision']*100:.2f}%")
    logger.info(f"Recall:         {results['recall']*100:.2f}%")
    logger.info(f"F1 Score:       {results['f1_score']*100:.2f}%")
    logger.info(f"Features:       {results['n_features']}")
    logger.info(f"Muestras:       {results['n_samples']}")
    logger.info("="*80)
    
    if results['test_accuracy'] >= 0.70:
        logger.info("✅ OBJETIVO ALCANZADO: 70%+ accuracy")
    else:
        logger.warning(f"⚠️ OBJETIVO NO ALCANZADO: {results['test_accuracy']*100:.2f}% < 70%")
        logger.info("   Sugerencias:")
        logger.info("   - Entrenar con --optimize --trials 100")
        logger.info("   - Agregar mas tickers")
        logger.info("   - Usar periodo mas largo (3y o 5y)")
    
    logger.info("\nModelo guardado en: models/advanced_ensemble.pkl")
    logger.info("Para predecir: python scripts/predict_advanced.py AAPL")


if __name__ == '__main__':
    main()
