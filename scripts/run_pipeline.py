"""
ejecuta el pipeline completo: datos, entrenamiento y api
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preparer import MLDataPreparer
from src.ml.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """ejecuta pipeline completo"""
    logger.info("=" * 60)
    logger.info("pipeline completo ML_Bolsa")
    logger.info("=" * 60)
    
    # paso 1: preparar datos
    logger.info("\npaso 1: preparando datos")
    preparer = MLDataPreparer()
    datos = preparer.prepare_all(years=2)
    
    if not datos:
        logger.error("fallo en preparacion de datos")
        return False
    
    dataset = preparer.create_unified_dataset()
    logger.info("datos preparados ok")
    
    # paso 2: entrenar modelo
    logger.info("\npaso 2: entrenando modelo")
    trainer = ModelTrainer()
    df = trainer.load_data(ticker='AAPL')
    
    if df is not None:
        X, y = trainer.prepare_features(df)
        if X is not None:
            accuracy = trainer.train(X, y)
            trainer.save_model()
            logger.info(f"modelo entrenado ok. accuracy: {accuracy:.2%}")
    
    # paso 3: informacion final
    logger.info("\npaso 3: listo para usar")
    logger.info("ejecuta: python scripts/run_api.py")
    logger.info("accede a: http://localhost:5000")
    
    logger.info("\n" + "=" * 60)
    logger.info("pipeline completado")
    logger.info("=" * 60)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
