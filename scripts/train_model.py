"""
script para entrenar modelo ML
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """entrena modelo ML"""
    logger.info("iniciando entrenamiento")
    
    # crear trainer
    trainer = ModelTrainer()
    
    # cargar datos
    df = trainer.load_data(ticker='AAPL', use_all=False)
    
    if df is None:
        logger.error("no se pudieron cargar datos")
        return
    
    # preparar features
    X, y = trainer.prepare_features(df)
    
    if X is None or y is None:
        logger.error("no se pudieron preparar features")
        return
    
    # entrenar
    accuracy = trainer.train(X, y)
    
    # guardar
    trainer.save_model(name="modelo_aapl_basico")
    
    logger.info(f"entrenamiento completado. accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    main()
