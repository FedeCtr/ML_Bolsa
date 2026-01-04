"""
modulo principal del proyecto ML_Bolsa
"""
from .utils.config import Config
from .utils.logger import get_logger

__version__ = "1.0.0"

# crear directorios al importar
Config.create_dirs()
