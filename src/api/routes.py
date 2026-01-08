"""
endpoints de la API
"""
from flask import jsonify, request, render_template
from datetime import datetime

from ..ml.predictor import StockPredictor
from ..utils.logger import get_logger
from ..utils.helpers import format_currency, format_percentage

logger = get_logger(__name__)


def register_routes(app):
    """registra todos los endpoints"""
    
    # inicializar predictor
    try:
        predictor = StockPredictor()
        logger.info("predictor inicializado")
    except:
        predictor = None
        logger.warning("no se pudo cargar predictor")
    
    @app.route('/')
    def home():
        """pagina principal"""
        return render_template('dashboard.html')
    
    @app.route('/health')
    def health():
        """endpoint de salud"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': predictor is not None and predictor.trainer.model is not None
        })
    
    @app.route('/api/predict/<ticker>', methods=['GET'])
    def predict_ticker(ticker):
        """predice direccion de una accion"""
        if predictor is None or predictor.trainer.model is None:
            return jsonify({
                'error': 'modelo no disponible',
                'message': 'entrena el modelo primero'
            }), 400
        
        try:
            resultado = predictor.predict_ticker(ticker.upper())
            
            if resultado is None:
                return jsonify({
                    'error': 'no se pudo predecir',
                    'ticker': ticker
                }), 404
            
            # añadir recomendacion
            recomendacion = predictor.get_recommendation(ticker.upper())
            resultado['recommendation'] = recomendacion
            resultado['timestamp'] = datetime.now().isoformat()
            
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"error en predict: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predict/batch', methods=['POST'])
    def predict_batch():
        """predice multiples acciones"""
        if predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({'error': 'no se proporcionaron tickers'}), 400
        
        resultados = predictor.predict_multiple([t.upper() for t in tickers])
        
        return jsonify({
            'resultados': resultados,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/api/recommendation/<ticker>', methods=['GET'])
    def get_recommendation(ticker):
        """obtiene recomendacion simple"""
        if predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        
        recomendacion = predictor.get_recommendation(ticker.upper())
        
        if recomendacion is None:
            return jsonify({'error': 'no se pudo obtener recomendacion'}), 404
        
        return jsonify({
            'ticker': ticker.upper(),
            'recomendacion': recomendacion,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.errorhandler(404)
    def not_found(error):
        """manejador 404"""
        return jsonify({'error': 'endpoint no encontrado'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """manejador 500"""
        logger.error(f"error interno: {error}")
        return jsonify({'error': 'error interno del servidor'}), 500
