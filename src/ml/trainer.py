"""
modulo para entrenar modelos de ml
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import Tuple, Optional, List

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class ModelTrainer:
    """entrena modelos ml para prediccion de acciones"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.config = Config()
        self.model = None
        self.features = None
        self.model_type = model_type
        
        os.makedirs(self.config.models_dir, exist_ok=True)
    
    def load_data(self, ticker: str = None, use_all: bool = False) -> Optional[pd.DataFrame]:
        """carga datos para entrenamiento"""
        if use_all:
            filepath = os.path.join(self.config.ml_ready_dir, "dataset_completo_latest.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                logger.info(f"dataset completo cargado: {len(df)} filas")
                return df
            else:
                logger.error("no existe dataset unificado")
                return None
        else:
            ticker = ticker or 'AAPL'
            filepath = os.path.join(self.config.processed_data_dir, f"{ticker}_processed.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                logger.info(f"{ticker} cargado: {len(df)} dias")
                return df
            else:
                logger.error(f"no existe {filepath}")
                return None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """selecciona features para el modelo"""
        feature_cols = [
            'retorno_1d', 'retorno_5d', 'retorno_20d',
            'volatilidad_5d', 'volatilidad_20d',
            'dist_sma_10', 'dist_sma_20',
            'rsi',
            'volumen_ratio',
            'dia_semana', 'mes'
        ]
        
        available_features = [f for f in feature_cols if f in df.columns]
        target_col = 'target_direccion'
        
        if target_col not in df.columns:
            logger.error(f"no existe {target_col} en los datos")
            return None, None
        
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        self.features = available_features
        
        logger.info(f"features: {len(available_features)}, muestras: {len(X)}")
        logger.info(f"distribucion target: {dict(y.value_counts())}")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> float:
        """entrena el modelo"""
        logger.info("iniciando entrenamiento")
        
        # split temporal (no aleatorio)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"train: {len(X_train)}, test: {len(X_test)}")
        
        # crear modelo
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        
        # entrenar
        self.model.fit(X_train, y_train)
        
        # evaluar
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"precision: {accuracy:.2%}")
        logger.info(f"aciertos: {sum(y_pred == y_test)}/{len(y_test)}")
        
        # reporte
        report = classification_report(y_test, y_pred, target_names=['BAJA', 'SUBE'])
        logger.info(f"\n{report}")
        
        # feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nfeatures mas importantes:")
            for i, row in importance_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return accuracy
    
    def save_model(self, name: str = "modelo_basico"):
        """guarda el modelo entrenado"""
        if self.model is None:
            logger.error("no hay modelo para guardar")
            return
        
        model_path = os.path.join(self.config.models_dir, f"{name}.pkl")
        features_path = os.path.join(self.config.models_dir, f"{name}_features.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.features, features_path)
        
        logger.info(f"modelo guardado: {model_path}")
    
    def load_model(self, name: str = "modelo_basico"):
        """carga un modelo guardado"""
        model_path = os.path.join(self.config.models_dir, f"{name}.pkl")
        features_path = os.path.join(self.config.models_dir, f"{name}_features.pkl")
        
        if not os.path.exists(model_path):
            logger.error(f"no existe {model_path}")
            return False
        
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)
        
        logger.info(f"modelo cargado: {model_path}")
        return True
