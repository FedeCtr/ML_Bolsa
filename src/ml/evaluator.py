"""
modulo para evaluar modelos
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """evalua rendimiento de modelos ML"""
    
    def __init__(self):
        pass
    
    def evaluate_classification(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """evalua modelo de clasificacion"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        logger.info("metricas de evaluacion:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def backtest_predictions(self, df: pd.DataFrame, predictions: pd.Series) -> Dict:
        """realiza backtest de predicciones"""
        df = df.copy()
        df['prediction'] = predictions
        
        # calcular retornos reales
        df['actual_return'] = df['Close'].pct_change().shift(-1)
        
        # estrategia: comprar cuando predice subida
        df['strategy_return'] = df['actual_return'] * df['prediction']
        
        # metricas
        total_return = df['strategy_return'].sum()
        sharpe = df['strategy_return'].mean() / df['strategy_return'].std() if df['strategy_return'].std() > 0 else 0
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': (df['strategy_return'] > 0).mean()
        }
        
        logger.info("resultados backtest:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
