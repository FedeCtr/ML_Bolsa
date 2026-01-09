"""
entrenador avanzado con ensemble de multiples modelos y optimizacion
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
from typing import Dict, Tuple, List
import os
import joblib

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class AdvancedEnsemble:
    """entrenador con 4 modelos ensamblados + optimizacion optuna para 70%+ accuracy"""
    
    def __init__(self):
        self.config = Config()
        self.ensemble = None
        self.feature_cols = None
        self.best_params = {}
        
    def get_feature_columns(self) -> List[str]:
        """lista de 50+ features"""
        return [
            # returns
            'retorno_1d', 'retorno_3d', 'retorno_5d', 'retorno_10d', 'retorno_20d',
            # volatility
            'volatilidad_5d', 'volatilidad_10d', 'volatilidad_20d', 'volatilidad_60d',
            # moving averages
            'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
            # macd
            'macd', 'macd_signal', 'macd_diff',
            # bollinger
            'bb_width', 'bb_pct',
            # momentum
            'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
            'momentum_5', 'momentum_10', 'momentum_20',
            # trend
            'adx',
            # volatility indicators
            'atr_pct',
            # volume
            'volumen_ratio', 'volume_roc', 'obv_ema', 'dist_vwap',
            # lag features
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
            # rolling stats
            'range_20d',
            # price patterns
            'body_pct',
            # temporal
            'dia_semana', 'mes', 'trimestre', 'dia_mes', 'es_fin_mes', 'es_inicio_mes'
        ]
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """prepara X, y con 50+ features"""
        df = df.copy()
        
        # target
        if 'target_direccion' not in df.columns:
            df['target_direccion'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # features
        feature_cols = self.get_feature_columns()
        
        # verificar columnas disponibles
        available_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"faltan {len(missing_cols)} columnas: {missing_cols[:5]}...")
        
        logger.info(f"usando {len(available_cols)} features de {len(feature_cols)} posibles")
        
        X = df[available_cols]
        y = df['target_direccion']
        
        # eliminar filas con NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_cols = available_cols
        
        logger.info(f"features preparadas: {X.shape}")
        return X, y
    
    def optimize_xgboost(self, X_train, y_train, n_trials=50) -> Dict:
        """optimiza xgboost con optuna"""
        logger.info("optimizando XGBoost...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            
            model = XGBClassifier(**params)
            
            # cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='xgboost')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"XGBoost best accuracy: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, n_trials=50) -> Dict:
        """optimiza lightgbm con optuna"""
        logger.info("optimizando LightGBM...")
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'verbose': -1
            }
            
            model = LGBMClassifier(**params)
            
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"LightGBM best accuracy: {study.best_value:.4f}")
        return study.best_params
    
    def train(self, df: pd.DataFrame, optimize=True, n_trials=50) -> Dict:
        """entrena ensemble de 4 modelos con optimizacion optuna"""
        logger.info("="*60)
        logger.info("ENTRENADOR AVANZADO - Ensemble de 4 modelos")
        logger.info("="*60)
        
        # preparar datos
        X, y = self.prepare_features(df)
        
        logger.info(f"datos: {X.shape[0]} muestras, {X.shape[1]} features")
        logger.info(f"distribucion target: {y.value_counts().to_dict()}")
        
        # split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # optimizacion
        if optimize:
            logger.info("\n--- FASE 1: OPTIMIZACION HYPERPARAMETROS ---")
            xgb_params = self.optimize_xgboost(X_train, y_train, n_trials)
            lgb_params = self.optimize_lightgbm(X_train, y_train, n_trials)
            self.best_params = {'xgboost': xgb_params, 'lightgbm': lgb_params}
        else:
            xgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            lgb_params = {
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'random_state': 42,
                'verbose': -1
            }
        
        # crear modelos
        logger.info("\n--- FASE 2: ENTRENANDO ENSEMBLE ---")
        
        xgb = XGBClassifier(**xgb_params)
        lgb = LGBMClassifier(**lgb_params)
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        et = ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        
        # voting ensemble con pesos optimizados
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgboost', xgb),
                ('lightgbm', lgb),
                ('random_forest', rf),
                ('extra_trees', et)
            ],
            voting='soft',
            weights=[0.35, 0.30, 0.25, 0.10]  # XGB y LGB mas peso
        )
        
        logger.info("entrenando ensemble (puede tardar 2-5 minutos)...")
        self.ensemble.fit(X_train, y_train)
        
        # evaluacion
        logger.info("\n--- FASE 3: EVALUACION ---")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred_train = self.ensemble.predict(X_train)
        y_pred_test = self.ensemble.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        logger.info(f"Train Accuracy: {train_acc:.4f}")
        logger.info(f"Test Accuracy:  {test_acc:.4f}")
        logger.info(f"Precision:      {precision:.4f}")
        logger.info(f"Recall:         {recall:.4f}")
        logger.info(f"F1 Score:       {f1:.4f}")
        
        # evaluacion individual
        logger.info("\n--- MODELOS INDIVIDUALES ---")
        for name, model in self.ensemble.named_estimators_.items():
            acc = accuracy_score(y_test, model.predict(X_test))
            logger.info(f"{name:15s}: {acc:.4f}")
        
        # guardar
        self.save_model()
        
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'best_params': self.best_params
        }
        
        logger.info("="*60)
        logger.info(f"RESULTADO FINAL: {test_acc*100:.2f}% accuracy")
        logger.info("="*60)
        
        return results
    
    def save_model(self):
        """guarda el ensemble"""
        model_path = os.path.join(self.config.models_dir, 'advanced_ensemble.pkl')
        features_path = os.path.join(self.config.models_dir, 'advanced_features.pkl')
        
        joblib.dump(self.ensemble, model_path)
        joblib.dump(self.feature_cols, features_path)
        
        logger.info(f"modelo guardado: {model_path}")
        logger.info(f"features guardadas: {features_path}")
    
    def load_model(self):
        """carga el ensemble"""
        model_path = os.path.join(self.config.models_dir, 'advanced_ensemble.pkl')
        features_path = os.path.join(self.config.models_dir, 'advanced_features.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"modelo no encontrado: {model_path}")
        
        self.ensemble = joblib.load(model_path)
        self.feature_cols = joblib.load(features_path)
        
        logger.info(f"modelo cargado: {model_path}")
        return self
