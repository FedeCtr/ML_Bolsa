"""
configuracion global del proyecto
"""
import os
from pathlib import Path


class Config:
    """configuracion centralizada"""
    
    # directorios base
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    SRC_DIR = BASE_DIR / "src"
    MODELS_DIR = BASE_DIR / "models"
    
    # subdirectorios de datos
    raw_data_dir = DATA_DIR / "raw"
    processed_data_dir = DATA_DIR / "processed"
    ml_ready_dir = DATA_DIR / "ml_ready"
    
    # modelo
    models_dir = MODELS_DIR
    default_model_name = "modelo_basico"
    
    # acciones por defecto
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    # parametros ML
    test_size = 0.2
    random_state = 42
    
    # API
    api_host = '0.0.0.0'
    api_port = 5000
    debug_mode = True
    
    @classmethod
    def create_dirs(cls):
        """crea todos los directorios necesarios"""
        for dir_path in [cls.raw_data_dir, cls.processed_data_dir, 
                         cls.ml_ready_dir, cls.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
