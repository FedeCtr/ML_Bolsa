"""
predictor avanzado con ensemble de 4 modelos y 50+ features

Version corregida (auditoria):
- Usa la CALIBRACION guardada (models/calibration.pkl) y el umbral
  seleccionado con datos OOF (scripts/calibrate_thresholds.py). Antes el
  umbral de 75% era arbitrario sobre probabilidades sin calibrar.
- El contexto de mercado (SPY/VIX) se descarga tambien en inferencia para
  que las features coincidan con el entrenamiento.
- Las estimaciones de horizontes/magnitudes son HEURISTICAS y se marcan
  como tales: no son salidas del modelo y no deben presentarse como
  predicciones cuantificadas.
- 'sin_accion' reemplaza a 'no_action' para el dashboard.
"""
import os
from typing import Dict, Optional

import joblib
import pandas as pd

from ..data.collector import DataCollector
from ..data.processor import TechnicalProcessor
from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class AdvancedPredictor:
    """predictor con ensemble + calibracion de confianza"""

    def __init__(self, confidence_threshold: Optional[float] = None):
        """
        Args:
            confidence_threshold: umbral explicito. Si es None, se usa el
                umbral seleccionado con datos OOF en models/calibration.pkl
                (o 0.60 si no existe calibracion).
        """
        self.config = Config()
        self.ensemble = None
        self.feature_cols = None
        self.calibrator = None
        self.confidence_threshold = None
        self.load_model(confidence_threshold)

    def load_model(self, confidence_threshold: Optional[float] = None):
        """carga ensemble, features y calibracion"""
        model_path = os.path.join(self.config.models_dir, 'advanced_ensemble.pkl')
        features_path = os.path.join(self.config.models_dir, 'advanced_features.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"modelo no encontrado: {model_path}")

        self.ensemble = joblib.load(model_path)
        self.feature_cols = joblib.load(features_path)

        # calibracion (opcional)
        calib_path = os.path.join(self.config.models_dir, 'calibration.pkl')
        if os.path.exists(calib_path):
            try:
                from .calibration import ConfidenceCalibrator
                self.calibrator = ConfidenceCalibrator.load(calib_path)
            except Exception as e:
                logger.warning(f"no se pudo cargar calibracion: {e}")

        # umbral: explicito > calibrado > default
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        elif self.calibrator is not None and self.calibrator.selected_threshold is not None:
            self.confidence_threshold = self.calibrator.selected_threshold
        else:
            self.confidence_threshold = 0.60

        logger.info(
            f"modelo cargado: {model_path} | umbral de senal: "
            f"{self.confidence_threshold:.3f} "
            f"({'calibrado con OOF' if self.calibrator else 'default sin calibrar'})"
        )

    def _raw_probability_up(self, X: pd.DataFrame) -> float:
        """probabilidad de subida del ensemble, calibrada si hay calibrador"""
        proba = self.ensemble.predict_proba(X)[0]
        idx_one = int(list(self.ensemble.classes_).index(1)) if 1 in self.ensemble.classes_ else 1
        p_up_raw = float(proba[idx_one])

        if self.calibrator is not None:
            try:
                import numpy as np
                p_up = float(self.calibrator.transform(np.array([p_up_raw]))[0])
            except Exception as e:
                logger.warning(f"fallo la calibracion, usando probabilidad cruda: {e}")
                p_up = p_up_raw
        else:
            p_up = p_up_raw
        return p_up

    def predict_ticker(self, ticker: str, period: str = '1y') -> Dict:
        """predice direccion con ensemble calibrado + umbral con datos"""
        logger.info(f"prediciendo {ticker} con ensemble avanzado")

        # descargar ticker + contexto de mercado (mismas features que training)
        collector = DataCollector()
        processor = TechnicalProcessor()

        df = collector.download_ticker(ticker, period)
        if df is None or df.empty:
            return {'error': 'no se pudo descargar datos'}

        spy = collector.download_ticker('SPY', period)
        vix = collector.download_ticker('^VIX', period)
        df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix)

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"features faltantes en inferencia: {missing[:5]}...")

        X = df[self.feature_cols].tail(1)

        if X.isna().any(axis=1).iloc[0]:
            return {'error': 'datos insuficientes para prediccion'}

        p_up = self._raw_probability_up(X)
        p_down = 1.0 - p_up

        # decision con umbral: dos lados, no solo "max"
        if p_up >= self.confidence_threshold:
            direccion = 'comprar'
            confidence = p_up
        elif p_down >= self.confidence_threshold:
            direccion = 'vender'
            confidence = p_down
        else:
            direccion = 'sin_accion'
            confidence = max(p_up, p_down)

        precio_actual = float(df['Close'].iloc[-1])

        # mensaje segun nivel de confianza
        if direccion == 'sin_accion':
            mensaje = (
                f"senal descartada: confianza {confidence:.2%} < umbral "
                f"{self.confidence_threshold:.2%}"
            )
        elif confidence >= 0.75:
            mensaje = f'alta confianza ({confidence:.2%})'
        elif confidence >= 0.60:
            mensaje = f'confianza moderada ({confidence:.2%})'
        else:
            mensaje = f'confianza baja ({confidence:.2%}) - usar con precaucion'

        resultado = {
            'ticker': ticker,
            'prediccion': direccion,
            'confidence_passed': direccion != 'sin_accion',
            'confianza': float(confidence),
            'probabilidad_subida': float(p_up),
            'probabilidad_bajada': float(p_down),
            'umbral_confianza': float(self.confidence_threshold),
            'probabilidad_calibrada': self.calibrator is not None,
            'precio_actual': precio_actual,
            'fecha': str(df.index[-1].date()),
            'mensaje': mensaje,
            'recommendation': direccion.upper(),
            'model_type': 'advanced_ensemble',
        }

        # --------------------------------------------------------------
        # ESTIMACIONES HEURISTICAS (NO son salidas del modelo)
        # --------------------------------------------------------------
        # Los horizontes, holding y niveles de stop/take se derivan de
        # volatilidad y momentum recientes con reglas simples. Sirven como
        # referencia de riesgo, NO como predicciones del ensemble.
        horizontes = self._estimar_horizontes_heuristico(df, direccion, confidence)
        resultado['horizontes'] = horizontes
        resultado['horizontes_naturaleza'] = 'heuristico_no_modelo'
        resultado['holding_recomendacion'] = self._holding_heuristico(horizontes, confidence)
        resultado['holding_naturaleza'] = 'heuristico_no_modelo'

        # stop basado en ATR (volatilidad real), no en ±2% fijo
        atr_val = df['atr'].iloc[-1] if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]) else precio_actual * 0.02
        if direccion == 'comprar':
            stop_loss = precio_actual - 2.0 * float(atr_val)
            take_profit = horizontes['horizonte_5d']['precio_objetivo']
        elif direccion == 'vender':
            stop_loss = precio_actual + 2.0 * float(atr_val)
            take_profit = horizontes['horizonte_5d']['precio_objetivo']
        else:
            stop_loss, take_profit = None, None

        resultado['stop_loss'] = round(stop_loss, 2) if stop_loss else None
        resultado['take_profit'] = round(take_profit, 2) if take_profit else None
        resultado['stop_naturaleza'] = 'heuristico_no_modelo' if stop_loss else None

        if stop_loss and take_profit:
            riesgo = abs(precio_actual - stop_loss)
            beneficio = abs(take_profit - precio_actual)
            resultado['riesgo_rendimiento'] = round(beneficio / riesgo, 2) if riesgo > 0 else None

        logger.info(
            f"prediccion: {direccion.upper()} (confianza: {confidence:.2%}, "
            f"umbral: {self.confidence_threshold:.2%})"
        )
        return resultado

    def _estimar_horizontes_heuristico(self, df: pd.DataFrame, direccion: str, confidence: float) -> Dict:
        """estimacion heuristica de horizontes basada en vol/momentum.

        NO es una salida del modelo: solo dimensiona escenarios de riesgo.
        """
        precio_actual = float(df['Close'].iloc[-1])
        volatilidad_diaria = float(df['Close'].pct_change().std())

        momentum_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) if len(df) >= 6 else 0
        momentum_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) if len(df) >= 21 else 0

        factor = 1 if direccion == 'comprar' else -1

        magnitud_1d = factor * volatilidad_diaria * confidence * 1.5
        magnitud_5d = factor * abs(momentum_5d) * confidence * 1.2 if momentum_5d * factor > 0 else magnitud_1d * 3
        magnitud_20d = factor * abs(momentum_20d) * confidence * 1.1 if momentum_20d * factor > 0 else magnitud_5d * 2

        magnitud_1d = max(min(magnitud_1d, 0.05), -0.05)
        magnitud_5d = max(min(magnitud_5d, 0.12), -0.12)
        magnitud_20d = max(min(magnitud_20d, 0.25), -0.25)

        return {
            'horizonte_1d': {'dias': 1, 'magnitud_esperada': round(magnitud_1d * 100, 2),
                             'precio_objetivo': round(precio_actual * (1 + magnitud_1d), 2)},
            'horizonte_5d': {'dias': 5, 'magnitud_esperada': round(magnitud_5d * 100, 2),
                             'precio_objetivo': round(precio_actual * (1 + magnitud_5d), 2)},
            'horizonte_20d': {'dias': 20, 'magnitud_esperada': round(magnitud_20d * 100, 2),
                              'precio_objetivo': round(precio_actual * (1 + magnitud_20d), 2)},
        }

    def _holding_heuristico(self, horizontes: Dict, confidence: float) -> str:
        """sugerencia heuristica de holding (no es salida del modelo)"""
        mag_5d = abs(horizontes['horizonte_5d']['magnitud_esperada'])
        mag_20d = abs(horizontes['horizonte_20d']['magnitud_esperada'])

        if confidence >= 0.75:
            if mag_20d >= 8:
                return "15-20 dias (posicion de largo plazo)"
            elif mag_5d >= 4:
                return "5-10 dias (swing trading)"
            else:
                return "1-3 dias (day trading)"
        elif confidence >= 0.60:
            if mag_5d >= 3:
                return "3-7 dias (swing trading moderado)"
            else:
                return "1-2 dias (day trading)"
        else:
            return "Intraday o 1 dia maximo (alta volatilidad)"

    def predict_multiple(self, tickers: list, period: str = '1y') -> pd.DataFrame:
        """predice multiples tickers con el umbral calibrado"""
        logger.info(f"prediciendo {len(tickers)} tickers")

        results = []
        for ticker in tickers:
            try:
                resultado = self.predict_ticker(ticker, period)
                results.append(resultado)
            except Exception as e:
                logger.error(f"error en {ticker}: {e}")
                results.append({'ticker': ticker, 'prediccion': 'error', 'error': str(e)})

        df_results = pd.DataFrame(results)

        if 'confianza' in df_results.columns:
            df_results = df_results.sort_values('confianza', ascending=False)

        if 'prediccion' in df_results.columns:
            logger.info("\nresumen:")
            logger.info(f"  comprar:    {(df_results['prediccion'] == 'comprar').sum()}")
            logger.info(f"  vender:     {(df_results['prediccion'] == 'vender').sum()}")
            logger.info(f"  sin accion: {(df_results['prediccion'] == 'sin_accion').sum()}")
            logger.info(f"  errores:    {(df_results['prediccion'] == 'error').sum()}")

            if 'confianza' in df_results.columns:
                conf_promedio = df_results['confianza'].mean()
                logger.info(f"  confianza promedio: {conf_promedio:.2%}")

        return df_results
