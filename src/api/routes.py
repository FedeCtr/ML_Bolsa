"""
endpoints de la API
"""
from flask import jsonify, request, render_template
from datetime import datetime
import os

from ..ml.predictor import StockPredictor
from ..ml.advanced_predictor import AdvancedPredictor
from ..utils.logger import get_logger
from ..utils.helpers import format_currency, format_percentage
from ..utils.config import Config

logger = get_logger(__name__)


def register_routes(app):
    """registra todos los endpoints"""
    
    # inicializar predictor avanzado si existe, sino el basico
    predictor = None
    advanced_predictor = None
    
    config = Config()
    advanced_model_path = os.path.join(config.models_dir, 'advanced_ensemble.pkl')
    
    if os.path.exists(advanced_model_path):
        try:
            advanced_predictor = AdvancedPredictor()
            logger.info("✅ predictor AVANZADO inicializado (ensemble 4 modelos + 50+ features)")
        except Exception as e:
            logger.warning(f"no se pudo cargar predictor avanzado: {e}")
    
    if advanced_predictor is None:
        try:
            predictor = StockPredictor()
            logger.info("predictor BASICO inicializado")
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
        model_type = 'advanced' if advanced_predictor else ('basic' if predictor else 'none')
        model_loaded = advanced_predictor is not None or (predictor is not None and predictor.trainer.model is not None)
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'model_loaded': model_loaded,
            'features': '50+ indicators' if advanced_predictor else '10 indicators',
            'models': 'Ensemble (XGBoost + LightGBM + RF + ET)' if advanced_predictor else 'RandomForest'
        })
    
    @app.route('/api/predict/<ticker>', methods=['GET'])
    def predict_ticker(ticker):
        """predice direccion de una accion"""
        active_predictor = advanced_predictor if advanced_predictor else predictor
        
        if active_predictor is None:
            return jsonify({
                'error': 'modelo no disponible',
                'message': 'entrena el modelo primero con: python scripts/train_advanced.py'
            }), 400
        
        try:
            resultado = active_predictor.predict_ticker(ticker.upper())
            
            if resultado is None or 'error' in resultado:
                return jsonify({
                    'error': resultado.get('error', 'no se pudo predecir'),
                    'ticker': ticker
                }), 404
            
            # agregar info del modelo usado
            resultado['model_type'] = 'advanced' if advanced_predictor else 'basic'
            resultado['timestamp'] = datetime.now().isoformat()
            
            # compatibilidad con sistema basico
            if 'recommendation' not in resultado and 'prediccion' in resultado:
                resultado['recommendation'] = resultado['prediccion'].upper()
            
            return jsonify(resultado)
            
        except Exception as e:
            logger.error(f"error en predict: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/predict/batch', methods=['POST'])
    def predict_batch():
        """predice multiples acciones"""
        active_predictor = advanced_predictor if advanced_predictor else predictor
        
        if active_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({'error': 'no se proporcionaron tickers'}), 400
        
        resultados = active_predictor.predict_multiple([t.upper() for t in tickers])
        
        return jsonify({
            'resultados': resultados.to_dict('records') if hasattr(resultados, 'to_dict') else resultados,
            'model_type': 'advanced' if advanced_predictor else 'basic',
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
