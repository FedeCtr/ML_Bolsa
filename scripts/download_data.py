"""
script para descargar y preparar datos
"""
import sys
from pathlib import Path

# añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preparer import MLDataPreparer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """descarga y prepara datos"""
    logger.info("iniciando descarga de datos")
    
    # crear preparador
    preparer = MLDataPreparer()
    
    # preparar datos
    datos = preparer.prepare_all(years=3)
    
    if datos:
        logger.info(f"datos preparados: {len(datos)} tickers")
        
        # crear dataset unificado
        dataset = preparer.create_unified_dataset()
        
        if dataset is not None:
            logger.info("dataset unificado creado")
        else:
            logger.error("no se pudo crear dataset")
    else:
        logger.error("no se pudieron preparar datos")


if __name__ == '__main__':
    main()
