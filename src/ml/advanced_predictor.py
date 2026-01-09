"""
predictor avanzado con ensemble de 4 modelos y 50+ features
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
import joblib

from ..data.collector import DataCollector
from ..data.processor import TechnicalProcessor
from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class AdvancedPredictor:
    """predictor con ensemble de 4 modelos y 50+ features"""
    
    def __init__(self, confidence_threshold: float = 0.0):
        self.config = Config()
        self.ensemble = None
        self.feature_cols = None
        self.confidence_threshold = confidence_threshold  # no se usa, solo para compatibilidad
        self.load_model()
        
    def load_model(self):
        """carga el ensemble"""
        model_path = os.path.join(self.config.models_dir, 'advanced_ensemble.pkl')
        features_path = os.path.join(self.config.models_dir, 'advanced_features.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"modelo no encontrado: {model_path}")
        
        self.ensemble = joblib.load(model_path)
        self.feature_cols = joblib.load(features_path)
        
        logger.info(f"modelo cargado: {model_path}")
        
    def predict_ticker(self, ticker: str, period: str = '1y') -> Dict:
        """predice con ensemble de 4 modelos + horizontes temporales"""
        logger.info(f"prediciendo {ticker} con ensemble avanzado")
        
        # descargar y procesar
        collector = DataCollector()
        processor = TechnicalProcessor()
        
        df = collector.download_ticker(ticker, period)
        if df is None or df.empty:
            return {'error': 'no se pudo descargar datos'}
        
        df = processor.process_all_indicators(df)
        
        # preparar features
        X = df[self.feature_cols].tail(1)
        
        if X.isna().any(axis=1).iloc[0]:
            return {'error': 'datos insuficientes para prediccion'}
        
        # prediccion con probabilidades
        pred_proba = self.ensemble.predict_proba(X)[0]
        pred_class = self.ensemble.predict(X)[0]
        
        confidence = pred_proba.max()
        direccion = 'comprar' if pred_class == 1 else 'vender'
        
        # NUEVO: calcular horizontes temporales y magnitudes
        precio_actual = float(df['Close'].iloc[-1])
        horizontes = self._calcular_horizontes(df, direccion, confidence)
        
        # calcular recomendacion de holding
        holding_recomendacion = self._calcular_holding_recomendacion(horizontes, confidence)
        
        # calcular stop loss y take profit
        stop_loss = precio_actual * (0.98 if direccion == 'comprar' else 1.02)
        take_profit = horizontes['horizonte_5d']['precio_objetivo']
        
        # mensaje segun nivel de confianza
        if confidence >= 0.75:
            mensaje = f'alta confianza ({confidence:.2%})'
        elif confidence >= 0.60:
            mensaje = f'confianza moderada ({confidence:.2%})'
        else:
            mensaje = f'confianza baja ({confidence:.2%}) - usar con precaución'
        
        resultado = {
            'ticker': ticker,
            'prediccion': direccion,
            'confianza': float(confidence),
            'probabilidad_subida': float(pred_proba[1]),
            'probabilidad_bajada': float(pred_proba[0]),
            'precio_actual': precio_actual,
            'fecha': str(df.index[-1].date()),
            'mensaje': mensaje,
            'recommendation': direccion.upper(),
            # NUEVO: horizontes temporales
            'horizontes': horizontes,
            'holding_recomendacion': holding_recomendacion,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'riesgo_rendimiento': round((take_profit - precio_actual) / (precio_actual - stop_loss), 2) if direccion == 'comprar' else round((precio_actual - take_profit) / (stop_loss - precio_actual), 2)
        }
        
        logger.info(f"prediccion: {direccion.upper()} (confianza: {confidence:.2%}) - Mantener: {holding_recomendacion}")
        
        return resultado
    
    def _calcular_horizontes(self, df: pd.DataFrame, direccion: str, confidence: float) -> Dict:
        """calcula predicciones para multiples horizontes temporales"""
        precio_actual = float(df['Close'].iloc[-1])
        
        # calcular volatilidad historica
        volatilidad_diaria = df['Close'].pct_change().std()
        
        # calcular momentum reciente
        momentum_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) if len(df) >= 6 else 0
        momentum_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) if len(df) >= 21 else 0
        
        # ajustar magnitudes segun confianza y direccion
        factor_direccion = 1 if direccion == 'comprar' else -1
        
        # 1 dia: basado en volatilidad diaria y confianza
        magnitud_1d = factor_direccion * volatilidad_diaria * confidence * 1.5
        
        # 5 dias: basado en momentum reciente
        magnitud_5d = factor_direccion * abs(momentum_5d) * confidence * 1.2 if momentum_5d * factor_direccion > 0 else magnitud_1d * 3
        
        # 20 dias: basado en momentum de largo plazo
        magnitud_20d = factor_direccion * abs(momentum_20d) * confidence * 1.1 if momentum_20d * factor_direccion > 0 else magnitud_5d * 2
        
        # limitar magnitudes a valores realistas
        magnitud_1d = max(min(magnitud_1d, 0.05), -0.05)  # max +-5% en 1 dia
        magnitud_5d = max(min(magnitud_5d, 0.12), -0.12)  # max +-12% en 5 dias
        magnitud_20d = max(min(magnitud_20d, 0.25), -0.25)  # max +-25% en 20 dias
        
        return {
            'horizonte_1d': {
                'dias': 1,
                'magnitud_esperada': round(magnitud_1d * 100, 2),
                'precio_objetivo': round(precio_actual * (1 + magnitud_1d), 2)
            },
            'horizonte_5d': {
                'dias': 5,
                'magnitud_esperada': round(magnitud_5d * 100, 2),
                'precio_objetivo': round(precio_actual * (1 + magnitud_5d), 2)
            },
            'horizonte_20d': {
                'dias': 20,
                'magnitud_esperada': round(magnitud_20d * 100, 2),
                'precio_objetivo': round(precio_actual * (1 + magnitud_20d), 2)
            }
        }
    
    def _calcular_holding_recomendacion(self, horizontes: Dict, confidence: float) -> str:
        """determina por cuanto tiempo mantener la posicion"""
        # obtener magnitudes
        mag_1d = abs(horizontes['horizonte_1d']['magnitud_esperada'])
        mag_5d = abs(horizontes['horizonte_5d']['magnitud_esperada'])
        mag_20d = abs(horizontes['horizonte_20d']['magnitud_esperada'])
        
        # decision basada en magnitud esperada y confianza
        if confidence >= 0.75:
            if mag_20d >= 8:
                return "15-20 días (posición de largo plazo)"
            elif mag_5d >= 4:
                return "5-10 días (swing trading)"
            else:
                return "1-3 días (day trading)"
        elif confidence >= 0.60:
            if mag_5d >= 3:
                return "3-7 días (swing trading moderado)"
            else:
                return "1-2 días (day trading)"
        else:
            return "Intraday o 1 día máximo (alta volatilidad)"
    
    def predict_multiple(self, tickers: list, period: str = '1y') -> pd.DataFrame:
        """predice multiples tickers con filtro"""
        logger.info(f"prediciendo {len(tickers)} tickers")
        
        results = []
        for ticker in tickers:
            try:
                resultado = self.predict_ticker(ticker, period)
                results.append(resultado)
            except Exception as e:
                logger.error(f"error en {ticker}: {e}")
                results.append({
                    'ticker': ticker,
                    'prediccion': 'error',
                    'error': str(e)
                })
        
        df_results = pd.DataFrame(results)
        
        # ordenar por confianza
        if 'confianza' in df_results.columns:
            df_results = df_results.sort_values('confianza', ascending=False)
        
        # estadisticas
        if 'prediccion' in df_results.columns:
            logger.info(f"\nresumen:")
            logger.info(f"  comprar:    {(df_results['prediccion'] == 'comprar').sum()}")
            logger.info(f"  vender:     {(df_results['prediccion'] == 'vender').sum()}")
            logger.info(f"  errores:    {(df_results['prediccion'] == 'error').sum()}")
            
            # confianza promedio
            if 'confianza' in df_results.columns:
                conf_promedio = df_results['confianza'].mean()
                logger.info(f"  confianza promedio: {conf_promedio:.2%}")
        
        return df_results
