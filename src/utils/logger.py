"""
sistema de logging para el proyecto
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """crea y configura un logger"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # archivo (opcional)
        # log_dir = Path(__file__).parent.parent.parent / "logs"
        # log_dir.mkdir(exist_ok=True)
        # file_handler = logging.FileHandler(log_dir / "app.log")
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
    
    return logger
