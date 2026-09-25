"""
Calibración de probabilidades y análisis de umbrales de confianza.

Problema que resuelve: el `VotingClassifier` produce probabilidades sin
calibrar. Fijar el umbral en 75% sin medir la precisión real en cada nivel
es arbitrario: con el modelo descalibrado, "75% de confianza" puede
corresponder a una precisión real muy distinta.

Solución: usar las probabilidades out-of-fold (OOF) generadas durante la
evaluación walk-forward — cada muestra fue predicha por un modelo que NO
vio esa muestra en entrenamiento — para:
  1. Medir qué precisión REAL tiene cada nivel de confianza.
  2. Ajustar una calibración (Platt/isotónica) sobre esas probabilidades.
  3. Elegir el umbral de producción a partir de datos, no de intuición.
"""
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class ConfidenceCalibrator:
    """calibra probabilidades y selecciona el umbral de confianza con datos OOF"""

    def __init__(self, method: str = 'isotonic', min_bin_samples: int = 50):
        """
        Args:
            method: 'isotonic' (no parametrica, requiere ~1000+ muestras OOF)
                    o 'platt' (regresion logistica sobre el logit, robusta
                    con pocas muestras).
            min_bin_samples: muestras minimas por banda de confianza para
                    reportar su precision.
        """
        if method not in ('isotonic', 'platt'):
            raise ValueError("method debe ser 'isotonic' o 'platt'")
        self.method = method
        self.min_bin_samples = min_bin_samples
        self.calibrator = None
        self.threshold_analysis: List[Dict] = {}
        self.selected_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Calibracion
    # ------------------------------------------------------------------

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> 'ConfidenceCalibrator':
        """ajusta el calibrador sobre probabilidades OOF.

        Args:
            probs: probabilidad OOF de subida (clase 1), shape (n,).
            targets: target real (0/1), shape (n,).
        """
        probs = np.asarray(probs, dtype=float).ravel()
        targets = np.asarray(targets, dtype=int).ravel()
        if len(probs) != len(targets):
            raise ValueError("probs y targets deben tener la misma longitud")
        if len(np.unique(targets)) < 2:
            raise ValueError("se necesitan ambas clases para calibrar")

        if self.method == 'platt':
            # regresion logistica sobre el logit de la probabilidad
            eps = 1e-6
            p = np.clip(probs, eps, 1 - eps)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1e6, max_iter=1000)
            self.calibrator.fit(logit, targets)
        else:
            self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.01, y_max=0.99)
            self.calibrator.fit(probs, targets)

        brier_before = brier_score_loss(targets, probs)
        brier_after = brier_score_loss(targets, self.transform(probs))
        logger.info(f"calibracion {self.method}: Brier {brier_before:.4f} -> {brier_after:.4f}")
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """aplica la calibracion a probabilidades crudas"""
        if self.calibrator is None:
            raise RuntimeError("calibrador no ajustado; llama fit() primero")
        probs = np.asarray(probs, dtype=float).ravel()
        if self.method == 'platt':
            eps = 1e-6
            p = np.clip(probs, eps, 1 - eps)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            return self.calibrator.predict_proba(logit)[:, 1]
        return self.calibrator.predict(probs)

    # ------------------------------------------------------------------
    # Analisis de umbrales
    # ------------------------------------------------------------------

    def analyze_thresholds(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        thresholds: Optional[List[float]] = None,
        select_by: str = 'precision_min_signals',
        min_signals: int = 30,
    ) -> Dict:
        """
        calcula la precision real por umbral de confianza sobre datos OOF.

        Args:
            probs: probabilidad OOF de subida (cruda o ya calibrada).
            targets: target real (0/1).
            thresholds: umbrales a evaluar (default: 0.50..0.80 en pasos de 0.025).
            select_by: criterio de seleccion del umbral de produccion:
                - 'precision_min_signals': max precision con al menos
                  `min_signals` senales OOF.
            min_signals: senales minimas para aceptar un umbral.

        Returns:
            dict con la tabla de analisis y el umbral seleccionado.
        """
        probs = np.asarray(probs, dtype=float).ravel()
        targets = np.asarray(targets, dtype=int).ravel()

        if thresholds is None:
            thresholds = [round(t, 3) for t in np.arange(0.50, 0.801, 0.025)]

        rows = []
        for thr in thresholds:
            mask = probs >= thr
            n = int(mask.sum())
            if n > 0:
                precision = float(targets[mask].mean())
            else:
                precision = float('nan')
            rows.append({
                'threshold': float(thr),
                'signals': n,
                'signal_rate': float(n / len(probs)),
                'precision': precision,
                'n': len(probs),
            })

        self.threshold_analysis = rows

        # seleccion del umbral de produccion
        candidates = [r for r in rows if r['signals'] >= min_signals and not np.isnan(r['precision'])]
        if not candidates:
            # relajar el requisito si no hay suficientes senales
            candidates = [r for r in rows if not np.isnan(r['precision'])]
            if candidates:
                logger.warning(
                    f"ningun umbral alcanza {min_signals} senales OOF; "
                    "se selecciona el de mayor precision con las senales disponibles"
                )
        if not candidates:
            logger.warning("sin senales por encima de 0.50; usando umbral 0.50")
            self.selected_threshold = 0.50
        else:
            best = max(candidates, key=lambda r: (r['precision'], r['signals']))
            self.selected_threshold = best['threshold']
            logger.info(
                f"umbral seleccionado: {self.selected_threshold:.3f} "
                f"(precision OOF {best['precision']:.4f} con {best['signals']} senales)"
            )

        return {
            'analysis': rows,
            'selected_threshold': self.selected_threshold,
            'baseline_accuracy': float(max(targets.mean(), 1 - targets.mean())),
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None):
        """guarda calibrador + umbral seleccionado"""
        if self.calibrator is None:
            raise RuntimeError("nada que guardar; ajusta el calibrador primero")
        if path is None:
            path = os.path.join(Config().models_dir, 'calibration.pkl')
        joblib.dump({
            'method': self.method,
            'calibrator': self.calibrator,
            'selected_threshold': self.selected_threshold,
            'threshold_analysis': self.threshold_analysis,
        }, path)
        logger.info(f"calibracion guardada: {path}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'ConfidenceCalibrator':
        """carga un calibrador guardado"""
        config = Config()
        if path is None:
            path = os.path.join(config.models_dir, 'calibration.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"calibracion no encontrada: {path}")
        data = joblib.load(path)
        cal = cls(method=data['method'])
        cal.calibrator = data['calibrator']
        cal.selected_threshold = data.get('selected_threshold')
        cal.threshold_analysis = data.get('threshold_analysis', {})
        logger.info(f"calibracion cargada: {path}")
        return cal

    def get_production_threshold(self, default: float = 0.60) -> float:
        """umbral de produccion; usa el default si no hay analisis"""
        return self.selected_threshold if self.selected_threshold is not None else default
