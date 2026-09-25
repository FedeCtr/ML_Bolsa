"""
Validación cruzada temporal con purging y embargo.

Problema que resuelve: el split aleatorio (train_test_split) baraja días
consecutivos entre train y test. Como los retornos diarios están
autocorrelacionados, el test contiene casi-duplicados del train y las
métricas resultantes están infladas (data leakage).

Solución: splits solo hacia adelante en el tiempo, con:
  - purge: se descartan muestras de train cuya ventana de features/target
    se solapa temporalmente con el test (gap = horizonte de predicción +
    lookback máximo de las features).
  - embargo: buffer adicional tras el purge antes de que el train toque
    el período de test (López de Prado, "Advances in Financial Machine
    Learning", cap. 7).

Como las features usan hasta 60 días de historia (volatilidad_60d) y el
target mira 1 día adelante, el purge default de 60 días cubre el solape
completo de cualquier ventana rolling con el período de test.
"""
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Historia máxima usada por las features (rolling más largo = volatilidad_60d).
MAX_FEATURE_LOOKBACK_DAYS = 60
# Buffer adicional anti-filtración entre folds (en días de mercado).
DEFAULT_EMBARGO_DAYS = 5
# Tamaño mínimo de fold de test: folds más pequeños no tienen potencia
# estadistica y solo anaden ruido a la metrica agregada.
MIN_TEST_SIZE = 20


class PurgedTimeSeriesSplit:
    """
    Walk-forward CV con purge gap y embargo, compatible con la convención
    de splitters de scikit-learn (método split que yielding (train, test)).

    Geometría: los últimos n_splits * test_size bloques se usan como test
    (ventana deslizante); el train de cada fold es toda la historia previa
    al test, menos gap + embargo muestras de purga (train anclado).

    Args:
        n_splits: número de folds.
        gap: días de purge entre train y test (>= MAX_FEATURE_LOOKBACK_DAYS
             para eliminar el solape de ventanas rolling con el test).
        embargo: días extra de aislamiento tras el purge.
        test_size: tamaño de cada fold de test. Por defecto
                   n_samples // (n_splits + 1).
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = MAX_FEATURE_LOOKBACK_DAYS,
        embargo: int = DEFAULT_EMBARGO_DAYS,
        test_size: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits debe ser >= 2")
        if gap < 0 or embargo < 0:
            raise ValueError("gap y embargo deben ser >= 0")
        self.n_splits = n_splits
        self.gap = gap
        self.embargo = embargo
        self.test_size = test_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yields (train_idx, test_idx) purgados."""
        n_samples = len(X) if X is not None and hasattr(X, "__len__") else 0
        yield from build_walk_forward_splits(
            n_samples=n_samples,
            n_splits=self.n_splits,
            gap=self.gap,
            embargo=self.embargo,
            test_size=self.test_size,
        )


def build_walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    gap: int = MAX_FEATURE_LOOKBACK_DAYS,
    embargo: int = DEFAULT_EMBARGO_DAYS,
    test_size: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera splits walk-forward con purge + embargo.

    Returns:
        Lista de (train_idx, test_idx) como arrays numpy.

    Raises:
        ValueError: si no hay datos suficientes para los folds solicitados.
    """
    if n_samples <= 0:
        raise ValueError("n_samples debe ser > 0")
    if n_splits < 2:
        raise ValueError("n_splits debe ser >= 2")

    min_train = n_samples // 10  # el primer fold necesita algo de train

    if test_size is None:
        # dimensionar el test para que el PRIMER fold conserve train
        # suficiente despues de descontar purge + embargo
        test_size = (n_samples - min_train - gap - embargo) // n_splits
        if test_size < MIN_TEST_SIZE:
            raise ValueError(
                f"muy pocas muestras ({n_samples}) para {n_splits} folds con "
                f"gap={gap} y embargo={embargo}: el test por fold seria de "
                f"{test_size} dias (minimo {MIN_TEST_SIZE})"
            )
    if test_size <= 0:
        raise ValueError("test_size calculado <= 0; faltan datos")

    needed = min_train + gap + embargo + n_splits * test_size
    if n_samples < needed:
        raise ValueError(
            f"muy pocas muestras ({n_samples}) para {n_splits} folds "
            f"con gap={gap}, embargo={embargo} y test_size={test_size} "
            f"(se necesitan ~{needed})"
        )

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        # ventana de test deslizante anclada al final de la serie
        test_start = n_samples - (n_splits - i) * test_size
        test_end = test_start + test_size
        train_end = test_start - gap - embargo
        if train_end < min_train:
            raise ValueError(
                f"fold {i}: train insuficiente tras purge+embargo "
                f"(train_end={train_end})"
            )
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    logger.info(
        f"walk-forward: {n_splits} folds, test_size={test_size}, "
        f"purge={gap}, embargo={embargo}"
    )
    return splits


def build_panel_walk_forward_splits(
    index: pd.Index,
    n_splits: int = 5,
    gap_days: int = MAX_FEATURE_LOOKBACK_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    walk-forward para datos de PANEL (varios tickers apilados).

    Problema que resuelve: concatenar tickers apila bloques; un split por
    posiciones haria que el test sea "la cola del ultimo ticker" mientras el
    train contiene los demas tickers EN LAS MISMAS FECHAS (fuga
    cross-sectional de regimen de mercado, infla la metrica).

    Solucion: los folds se definen sobre el CALENDARIO de fechas unicas
    (comunes a todos los tickers). Train = todas las filas con fecha
    anterior a la ventana de test; test = todas las filas (de todos los
    tickers) dentro de la ventana. Asi el train de cada fold solo contiene
    el PASADO de todos los tickers, y el test evalua cross-sectional.

    Args:
        index: DatetimeIndex del panel (con duplicados: una fecha por ticker).
        gap_days / embargo_days: purge y embargo en SESIONES a nivel fecha.

    Returns:
        Lista de (train_idx, test_idx) POSICIONALES sobre las filas del panel.
    """
    dates = pd.DatetimeIndex(pd.unique(pd.Series(index))).sort_values()
    date_splits = build_walk_forward_splits(
        len(dates), n_splits=n_splits, gap=gap_days, embargo=embargo_days,
    )

    idx_values = index.values
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_date_idx, test_date_idx in date_splits:
        train_dates = dates[train_date_idx].values
        test_dates = dates[test_date_idx].values
        train_rows = np.flatnonzero(np.isin(idx_values, train_dates))
        test_rows = np.flatnonzero(np.isin(idx_values, test_dates))
        if len(train_rows) and len(test_rows):
            splits.append((train_rows, test_rows))

    logger.info(
        f"panel walk-forward: {len(splits)} folds por fecha "
        f"(purge={gap_days} sesiones, embargo={embargo_days})"
    )
    return splits
