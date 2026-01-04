"""
aplicacion Flask principal
"""
from flask import Flask
import os

from .routes import register_routes
from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


def create_app():
    """crea y configura la aplicacion Flask"""
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
    
    # configuracion
    config = Config()
    app.config['DEBUG'] = config.debug_mode
    
    # registrar rutas
    register_routes(app)
    
    logger.info("aplicacion Flask creada")
    
    return app


def run_app():
    """ejecuta la aplicacion"""
    config = Config()
    app = create_app()
    
    logger.info(f"iniciando servidor en {config.api_host}:{config.api_port}")
    app.run(host=config.api_host, port=config.api_port, debug=config.debug_mode)


if __name__ == '__main__':
    run_app()
