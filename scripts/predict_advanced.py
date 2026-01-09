"""
Script para predecir con el sistema avanzado
Usa filtro de confianza >75% para mayor precision

Uso:
    python scripts/predict_advanced.py AAPL
    python scripts/predict_advanced.py AAPL MSFT GOOGL --confidence 0.80
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.advanced_predictor import AdvancedPredictor
from src.utils.logger import get_logger
import argparse
import pandas as pd

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Prediccion con ensemble avanzado')
    parser.add_argument('tickers', nargs='+', help='ticker(s) para predecir')
    parser.add_argument('--confidence', type=float, default=0.75, 
                        help='umbral de confianza (default: 0.75)')
    parser.add_argument('--period', type=str, default='1y', help='periodo de datos (default: 1y)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("PREDICCION AVANZADA - Ensemble con filtro de confianza")
    logger.info("="*80)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Umbral confianza: {args.confidence*100:.0f}%")
    logger.info("="*80)
    
    try:
        predictor = AdvancedPredictor(confidence_threshold=args.confidence)
        
        if len(args.tickers) == 1:
            # prediccion individual
            resultado = predictor.predict_ticker(args.tickers[0], args.period)
            
            logger.info("\nRESULTADO:")
            logger.info("-"*80)
            logger.info(f"Ticker:             {resultado.get('ticker', 'N/A')}")
            logger.info(f"Precio actual:      ${resultado.get('precio_actual', 0):.2f}")
            logger.info(f"Prediccion:         {resultado.get('prediccion', 'N/A').upper()}")
            logger.info(f"Confianza:          {resultado.get('confianza', 0)*100:.2f}%")
            logger.info(f"Prob. Subida:       {resultado.get('probabilidad_subida', 0)*100:.2f}%")
            logger.info(f"Prob. Bajada:       {resultado.get('probabilidad_bajada', 0)*100:.2f}%")
            logger.info(f"Mensaje:            {resultado.get('mensaje', '')}")
            
            # Mostrar horizontes temporales
            if 'horizontes' in resultado:
                logger.info("\n📊 HORIZONTES DE INVERSIÓN:")
                h = resultado['horizontes']
                for key, data in h.items():
                    signo = '↗️' if data['magnitud_esperada'] > 0 else '↘️'
                    logger.info(f"  {data['dias']:2d} días: {signo} {data['magnitud_esperada']:+.2f}% → ${data['precio_objetivo']:.2f}")
            
            # Mostrar recomendación de holding
            if 'holding_recomendacion' in resultado:
                logger.info(f"\n⏱️  MANTENER: {resultado['holding_recomendacion']}")
            
            # Mostrar métricas de riesgo
            if 'stop_loss' in resultado and 'take_profit' in resultado:
                logger.info(f"\n📈 GESTIÓN DE RIESGO:")
                logger.info(f"  🎯 Take Profit:      ${resultado['take_profit']:.2f}")
                logger.info(f"  🛑 Stop Loss:        ${resultado['stop_loss']:.2f}")
                if 'riesgo_rendimiento' in resultado:
                    logger.info(f"  📊 Riesgo/Beneficio: 1:{resultado['riesgo_rendimiento']:.2f}")
            
            logger.info("-"*80)
            
        else:
            # prediccion multiple
            df_results = predictor.predict_multiple(args.tickers, args.period)
            
            logger.info("\nRESULTADOS:")
            logger.info("-"*80)
            
            # mostrar tabla
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', 30)
            
            cols_show = ['ticker', 'prediccion', 'confianza', 'precio_actual', 'mensaje']
            cols_available = [c for c in cols_show if c in df_results.columns]
            
            print(df_results[cols_available].to_string(index=False))
            logger.info("-"*80)
            
            # guardar csv
            output_path = 'predictions_advanced.csv'
            df_results.to_csv(output_path, index=False)
            logger.info(f"\nResultados guardados en: {output_path}")
    
    except FileNotFoundError as e:
        logger.error(f"\nERROR: {e}")
        logger.error("\nPrimero debes entrenar el modelo:")
        logger.error("  python scripts/train_advanced.py --optimize --trials 100")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
