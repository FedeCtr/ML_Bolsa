"""
modulo para hacer predicciones con modelos entrenados
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import yfinance as yf

from .trainer import ModelTrainer
from ..data.processor import TechnicalProcessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StockPredictor:
    """hace predicciones sobre acciones usando modelos ML"""
    
    def __init__(self, model_name: str = "modelo_basico"):
        self.trainer = ModelTrainer()
        self.processor = TechnicalProcessor()
        self.model_name = model_name
        
        # cargar modelo
        if not self.trainer.load_model(model_name):
            logger.warning(f"no se pudo cargar {model_name}")
    
    def predict_ticker(self, ticker: str) -> Optional[Dict]:
        """predice la direccion de una accion"""
        if self.trainer.model is None:
            logger.error("no hay modelo cargado")
            return None
        
        try:
            # descargar datos recientes
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo", interval="1d")
            
            if df.empty:
                logger.error(f"sin datos para {ticker}")
                return None
            
            # procesar datos
            df = self.processor.process_all_indicators(df)
            df = df.dropna()
            
            if len(df) == 0:
                logger.error(f"sin datos procesados para {ticker}")
                return None
            
            # preparar features
            X_actual = df[self.trainer.features].iloc[-1:].copy()
            
            # predecir
            prediccion = self.trainer.model.predict(X_actual)[0]
            probabilidad = self.trainer.model.predict_proba(X_actual)[0]
            
            resultado = {
                'ticker': ticker,
                'prediccion': 'SUBE' if prediccion == 1 else 'BAJA',
                'probabilidad_subida': float(probabilidad[1]),
                'probabilidad_bajada': float(probabilidad[0]),
                'confianza': float(max(probabilidad)),
                'fecha': str(df.index[-1].date()),
                'precio_actual': float(df['Close'].iloc[-1])
            }
            
            logger.info(f"prediccion {ticker}: {resultado['prediccion']} ({resultado['confianza']:.1%})")
            
            return resultado
            
        except Exception as e:
            logger.error(f"error prediciendo {ticker}: {e}")
            return None
    
    def predict_multiple(self, tickers: list) -> Dict[str, Dict]:
        """predice multiples acciones"""
        resultados = {}
        
        for ticker in tickers:
            resultado = self.predict_ticker(ticker)
            if resultado:
                resultados[ticker] = resultado
        
        return resultados
    
    def get_recommendation(self, ticker: str) -> Optional[str]:
        """obtiene recomendacion simple"""
        resultado = self.predict_ticker(ticker)
        
        if not resultado:
            return None
        
        if resultado['probabilidad_subida'] > 0.6:
            return 'COMPRA'
        elif resultado['probabilidad_bajada'] > 0.6:
            return 'VENTA'
        else:
            return 'MANTENER'
