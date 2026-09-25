"""
entrenador avanzado con ensemble de multiples modelos y optimizacion

Version corregida (auditoria):
- Split temporal walk-forward con purge + embargo (NUNCA aleatorio).
- Optuna evalua con la misma CV purgada (antes usaba KFold aleatorio,
  lo que hacia que el tuning filtrara informacion del futuro).
- Probabilidades OOF para calibrar el umbral de confianza con datos reales.
- Drop de constantes y columnas 100% NaN (antes provocaba errores de
  entrenamiento cuando un ticker no tenia historia suficiente).
- Guarda metadatos de entrenamiento (config CV, periodo, tickers) para
  reproducibilidad.
"""
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from ..utils.logger import get_logger
from ..utils.config import Config
from .cv import (
    DEFAULT_EMBARGO_DAYS,
    MAX_FEATURE_LOOKBACK_DAYS,
    build_panel_walk_forward_splits,
    build_walk_forward_splits,
)

logger = get_logger(__name__)

# silenciar el log verbose de optuna en cada trial
optuna.logging.set_verbosity(optuna.logging.WARNING)

# features del modelo. Todas estacionarias: retornos, ratios, distancias
# relativas, z-scores y osciladores. SIN niveles de precio crudos.
DEFAULT_FEATURE_COLS: List[str] = [
    # retornos
    'retorno_1d', 'retorno_3d', 'retorno_5d', 'retorno_10d', 'retorno_20d',
    # volatilidad
    'volatilidad_5d', 'volatilidad_10d', 'volatilidad_20d', 'volatilidad_60d',
    # medias moviles (distancias relativas)
    'dist_sma_10', 'dist_sma_20', 'dist_sma_50',
    # MACD normalizado por precio
    'macd_norm', 'macd_diff_norm',
    # bollinger
    'bb_width', 'bb_pct',
    # osciladores
    'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
    # momentum como retorno relativo
    'momentum_5', 'momentum_10', 'momentum_20',
    # tendencia y volatilidad
    'adx', 'atr_pct',
    # volumen
    'volumen_ratio', 'volume_roc', 'obv_ratio_20d', 'dist_vwap',
    # lags de retornos (no de precios)
    'lag_ret_1', 'lag_ret_2', 'lag_ret_3', 'lag_ret_5', 'lag_ret_10',
    # rango reciente
    'range_20d',
    # patrones de vela
    'body_pct',
    # temporal
    'dia_semana', 'mes', 'trimestre', 'dia_mes', 'es_fin_mes', 'es_inicio_mes',
    # contexto de mercado (SPY / VIX) - solo presentes si se descargaron
    'spy_ret_1d', 'spy_ret_5d', 'spy_dist_sma_20', 'spy_vol_20d', 'spy_corr_20d',
    'vix_level', 'vix_zscore', 'vix_change_5d',
]


class AdvancedEnsemble:
    """ensemble de 4 modelos + optuna con validacion temporal purgada"""

    def __init__(self):
        self.config = Config()
        self.ensemble = None
        self.feature_cols: List[str] = []
        self.best_params: Dict = {}
        self.train_metadata: Dict = {}
        self.oof_probabilities: Optional[np.ndarray] = None
        self.oof_targets: Optional[np.ndarray] = None
        self.oof_dates: List = []
        self.oof_tickers: List = []

    @staticmethod
    def get_feature_columns() -> List[str]:
        """lista de features del modelo (todas estacionarias)"""
        return list(DEFAULT_FEATURE_COLS)

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ):
        """
        prepara X, y a partir de un dataframe YA procesado.

        Args:
            df: dataframe con features calculadas (por TechnicalProcessor) y
                target_direccion ya anadida. Cada fila debe corresponder a un
                unico ticker (procesar por-ticker ANTES de concatenar).
            feature_cols: columnas a usar. Default: DEFAULT_FEATURE_COLS.
        """
        df = df.copy()

        if 'target_direccion' not in df.columns:
            raise ValueError(
                "target_direccion no existe; llama a MLDataPreparer.add_targets "
                "o anadelo antes de entrenar"
            )

        # target: subida manana = 1. La fila de hoy usa SOLO informacion de hoy
        # y antes; el shift(-1) es el estandar y no introduce lookahead.
        y = df['target_direccion']

        requested = feature_cols or self.get_feature_columns()
        available = [c for c in requested if c in df.columns]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            logger.warning(f"faltan {len(missing)} columnas: {missing[:5]}...")

        X = df[available].copy()

        # eliminar filas con NaN o infinitos.
        # NOTA: se usa mascara posicional (numpy) porque el df combinado tiene
        # indices con etiquetas duplicadas (fechas repetidas entre tickers) y
        # el masking booleano por etiquetas de pandas produce resultados
        # incorrectos con indices no unicos.
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = ~(X.isna().any(axis=1) | y.isna())
        mask_arr = mask.to_numpy()
        X = X.loc[mask_arr]
        y = y.loc[mask_arr]

        # drop de columnas constantes (rompen LightGBM/XGBoost si el subset
        # de train tiene un solo valor, tipico con poco historial) y columnas
        # 100% vacias
        nunique = X.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        all_nan_cols = X.columns[X.isna().all()].tolist()
        drop_cols = set(constant_cols) | set(all_nan_cols)
        if drop_cols:
            logger.warning(
                f"descartando {len(drop_cols)} columnas sin senal "
                f"(constantes o vacias): {sorted(drop_cols)}"
            )
            X = X.drop(columns=list(drop_cols))

        self.feature_cols = list(X.columns)

        if 'ticker' in df.columns:
            per_ticker = df['ticker'].loc[mask_arr].value_counts()
            logger.info(f"muestras por ticker: {per_ticker.to_dict()}")

        logger.info(f"features preparadas: {X.shape}")
        return X, y

    # ------------------------------------------------------------------
    # Optuna: cada trial se evalua con la MISMA CV purgada del split final
    # ------------------------------------------------------------------

    def _make_objective(self, X: pd.DataFrame, y: pd.Series, model_name: str):
        splits = self.splits

        def objective(trial: optuna.Trial) -> float:
            if model_name == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'random_state': self.config.random_state,
                    'eval_metric': 'logloss',
                    'n_jobs': -1,
                }
                model = XGBClassifier(**params)
            elif model_name == 'lightgbm':
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'random_state': self.config.random_state,
                    'verbose': -1,
                    'n_jobs': -1,
                }
                model = LGBMClassifier(**params)
            else:
                raise ValueError(f"modelo desconocido: {model_name}")

            scores = []
            for train_idx, test_idx in splits:
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                # si un fold de train tiene una sola clase, el trial no es evaluable
                if y_tr.nunique() < 2:
                    return 0.5
                model.fit(X_tr, y_tr)
                scores.append(accuracy_score(y_te, model.predict(X_te)))
            return float(np.mean(scores))

        return objective

    def optimize(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> Dict:
        """optimiza xgboost y lightgbm con optuna sobre CV purgada"""
        logger.info(f"optimizando con optuna ({n_trials} trials, CV purgada)...")

        results = {}
        for name in ('xgboost', 'lightgbm'):
            study = optuna.create_study(direction='maximize', study_name=name)
            study.optimize(
                self._make_objective(X, y, name),
                n_trials=n_trials,
                show_progress_bar=False,
            )
            logger.info(f"{name}: best CV accuracy = {study.best_value:.4f}")
            results[name] = study.best_params

        return results

    # ------------------------------------------------------------------
    # Entrenamiento principal
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        optimize: bool = False,
        n_trials: int = 30,
        n_splits: int = 5,
        tickers: Optional[List[str]] = None,
        data_start: Optional[str] = None,
        data_end: Optional[str] = None,
    ) -> Dict:
        """
        entrena el ensemble con validacion walk-forward purgada.

        Args:
            df: dataframe con features + target_direccion (por ticker).
            optimize: si True, corre optuna con la CV purgada.
            n_trials: trials de optuna por modelo.
            n_splits: folds de walk-forward.
            tickers: lista de tickers incluidos (para metadata).
            data_start / data_end: rango de fechas (para metadata).
        """
        logger.info("=" * 60)
        logger.info("ENTRENADOR AVANZADO - Ensemble con validacion purgada")
        logger.info("=" * 60)

        X, y = self.prepare_features(df)

        # CV purgada compartida por optuna y por la evaluacion final.
        # PANEL (varios tickers): folds por FECHA para evitar fuga
        # cross-sectional (train con otros tickers en las mismas fechas).
        # SERIE UNICA: walk-forward posicional clasico.
        is_panel = 'ticker' in df.columns
        if is_panel:
            splits = build_panel_walk_forward_splits(
                X.index,
                n_splits=n_splits,
                gap_days=MAX_FEATURE_LOOKBACK_DAYS,
                embargo_days=DEFAULT_EMBARGO_DAYS,
            )
        else:
            splits = build_walk_forward_splits(
                n_samples=len(X),
                n_splits=n_splits,
                gap=MAX_FEATURE_LOOKBACK_DAYS,
                embargo=DEFAULT_EMBARGO_DAYS,
            )
        self.splits = splits

        # 1) tuning (opcional)
        if optimize:
            logger.info("\n--- FASE 1: OPTUNA sobre CV purgada ---")
            self.best_params = self.optimize(X, y, n_trials)
        else:
            self.best_params = {}

        default_xgb = {
            'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200,
            'random_state': self.config.random_state, 'eval_metric': 'logloss', 'n_jobs': -1,
        }
        default_lgb = {
            'num_leaves': 31, 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200,
            'random_state': self.config.random_state, 'verbose': -1, 'n_jobs': -1,
        }
        xgb_params = {**default_xgb, **self.best_params.get('xgboost', {})}
        lgb_params = {**default_lgb, **self.best_params.get('lightgbm', {})}

        # 2) entrenamiento final: en cada fold se entrena solo con el pasado
        #    y se evalua en el futuro (cada dia del test se predice mirando
        #    unicamente hacia atras).
        logger.info("\n--- FASE 2: EVALUACION WALK-FORWARD PURGADA ---")

        def make_models():
            return [
                ('xgboost', XGBClassifier(**xgb_params)),
                ('lightgbm', LGBMClassifier(**lgb_params)),
                ('random_forest', RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    random_state=self.config.random_state, n_jobs=-1,
                )),
                ('extra_trees', ExtraTreesClassifier(
                    n_estimators=200, max_depth=10,
                    random_state=self.config.random_state, n_jobs=-1,
                )),
            ]

        y_true_all, y_pred_all, y_prob_all = [], [], []
        oof_dates_all: List = []
        oof_tickers_all: List = []
        individual_scores = {name: [] for name in
                             ('xgboost', 'lightgbm', 'random_forest', 'extra_trees')}

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            if y_tr.nunique() < 2:
                logger.warning(f"fold {fold_i}: train con una sola clase; saltando")
                continue

            ensemble = VotingClassifier(estimators=make_models(), voting='soft')
            ensemble.fit(X_tr, y_tr)

            proba = ensemble.predict_proba(X_te)
            classes = ensemble.classes_
            # alinear probabilidad de clase 1 (subida)
            idx_one = int(np.where(classes == 1)[0][0])
            y_prob_fold = proba[:, idx_one]
            y_pred_fold = (y_prob_fold >= 0.5).astype(int)

            y_true_all.append(y_te.to_numpy())
            y_pred_all.append(y_pred_fold)
            y_prob_all.append(y_prob_fold)
            oof_dates_all.extend([str(d.date()) for d in X_te.index])
            if 'ticker' in df.columns:
                oof_tickers_all.extend(df['ticker'].loc[X_te.index].tolist())
            else:
                oof_tickers_all.extend(['?'] * len(test_idx))

            # score individual por modelo en este fold
            for name, model in ensemble.named_estimators_.items():
                p = model.predict_proba(X_te)
                i1 = int(np.where(model.classes_ == 1)[0][0])
                individual_scores[name].append(accuracy_score(
                    y_te, (p[:, i1] >= 0.5).astype(int)
                ))

            acc_fold = accuracy_score(y_te, y_pred_fold)
            logger.info(f"fold {fold_i}: train {len(train_idx):5d} / test {len(test_idx):4d} -> acc {acc_fold:.4f}")

        if not y_true_all:
            raise RuntimeError("no se pudo evaluar ningun fold (train sin ambas clases)")

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_prob_all = np.concatenate(y_prob_all)

        # guardar probabilidades OOF para calibracion de umbrales y para el
        # heatmap historico del dashboard (incluye fecha y ticker por fila)
        self.oof_probabilities = y_prob_all
        self.oof_targets = y_true_all
        self.oof_dates = oof_dates_all
        self.oof_tickers = oof_tickers_all

        accuracy = accuracy_score(y_true_all, y_pred_all)
        precision = precision_score(y_true_all, y_pred_all, zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

        logger.info("\n--- RESULTADOS WALK-FORWARD (sin leakage) ---")
        logger.info(f"Directional Accuracy: {accuracy:.4f}")
        logger.info(f"Precision:            {precision:.4f}")
        logger.info(f"Recall:               {recall:.4f}")
        logger.info(f"F1 Score:             {f1:.4f}")
        for name, scores in individual_scores.items():
            logger.info(f"  {name:15s}: {np.mean(scores):.4f}")

        # 3) modelo final: entrenado con TODOS los datos disponibles para
        #    produccion (la metrica honesta ya fue calculada arriba)
        logger.info("\n--- FASE 3: ENTRENANDO MODELO FINAL DE PRODUCCION ---")
        self.ensemble = VotingClassifier(estimators=make_models(), voting='soft')
        self.ensemble.fit(X, y)

        # metadata para reproducibilidad
        self.train_metadata = {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'feature_cols': list(X.columns),
            'cv': {
                'scheme': 'panel_walk_forward_purged_by_date' if is_panel
                          else 'walk_forward_purged',
                'n_splits': n_splits,
                'purge_days': MAX_FEATURE_LOOKBACK_DAYS,
                'embargo_days': DEFAULT_EMBARGO_DAYS,
            },
            'best_params': self.best_params,
            'walk_forward_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            },
            'individual_accuracy': {
                name: float(np.mean(s)) for name, s in individual_scores.items()
            },
            'tickers': tickers or [],
            'data_start': data_start,
            'data_end': data_end,
            'trained_at': pd.Timestamp.now().isoformat(),
        }

        self.save_model()

        logger.info("=" * 60)
        logger.info(f"RESULTADO HONESTO: {accuracy*100:.2f}% directional accuracy (walk-forward)")
        logger.info("=" * 60)

        return self.train_metadata['walk_forward_metrics'] | {
            'n_features': int(X.shape[1]),
            'n_samples': int(X.shape[0]),
            'best_params': self.best_params,
            'individual_accuracy': self.train_metadata['individual_accuracy'],
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save_model(self):
        """guarda el ensemble, features y metadatos"""
        model_path = os.path.join(self.config.models_dir, 'advanced_ensemble.pkl')
        features_path = os.path.join(self.config.models_dir, 'advanced_features.pkl')
        meta_path = os.path.join(self.config.models_dir, 'advanced_metadata.pkl')

        joblib.dump(self.ensemble, model_path)
        joblib.dump(self.feature_cols, features_path)
        joblib.dump(self.train_metadata, meta_path)

        logger.info(f"modelo guardado: {model_path}")

    def save_oof(self):
        """guarda probabilidades OOF (+fecha/ticker) para calibracion y dashboard"""
        if self.oof_probabilities is None:
            raise RuntimeError("no hay OOF; ejecuta train() primero")
        oof_path = os.path.join(self.config.models_dir, 'advanced_oof.pkl')
        joblib.dump({
            'probabilities': self.oof_probabilities,
            'targets': self.oof_targets,
            'dates': getattr(self, 'oof_dates', None),
            'tickers': getattr(self, 'oof_tickers', None),
        }, oof_path)
        logger.info(f"OOF guardado: {oof_path}")

    def load_model(self):
        """carga el ensemble, features y metadatos"""
        model_path = os.path.join(self.config.models_dir, 'advanced_ensemble.pkl')
        features_path = os.path.join(self.config.models_dir, 'advanced_features.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"modelo no encontrado: {model_path}")

        self.ensemble = joblib.load(model_path)
        self.feature_cols = joblib.load(features_path)

        meta_path = os.path.join(self.config.models_dir, 'advanced_metadata.pkl')
        if os.path.exists(meta_path):
            self.train_metadata = joblib.load(meta_path)

        logger.info(f"modelo cargado: {model_path}")
        return self
