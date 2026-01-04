"""
script simple para probar la nueva estructura
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """prueba rapida de la estructura"""
    logger.info("probando nueva estructura modular")
    
    # crear collector
    collector = DataCollector()
    
    # descargar datos de AAPL
    df = collector.download_ticker('AAPL', years=1)
    
    if df is not None:
        logger.info(f"descarga exitosa: {len(df)} filas")
        logger.info(f"precio actual: ${df['Close'].iloc[-1]:.2f}")
        logger.info(f"maximo: ${df['Close'].max():.2f}")
        logger.info(f"minimo: ${df['Close'].min():.2f}")
        
        print("\n" + "="*50)
        print("✓ Estructura modular funcionando correctamente")
        print("="*50)
        return True
    else:
        logger.error("fallo en descarga")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
