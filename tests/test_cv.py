"""tests del splitter walk-forward purgado"""
import numpy as np
import pytest

from src.ml.cv import MAX_FEATURE_LOOKBACK_DAYS, build_walk_forward_splits, PurgedTimeSeriesSplit


def test_folds_are_strictly_forward_in_time():
    """train siempre antes que test, y tests no solapados"""
    n = 1500
    splits = build_walk_forward_splits(n, n_splits=5)

    prev_test_end = -1
    for train_idx, test_idx in splits:
        # todo el train es anterior al test
        assert train_idx.max() < test_idx.min()
        # los tests avanzan en el tiempo y no se solapan
        assert test_idx.min() >= prev_test_end
        prev_test_end = test_idx.max()


def test_purge_gap_between_train_and_test():
    """el train termina al menos gap+embargo muestras antes del test"""
    n, n_splits, gap, embargo = 1500, 5, 60, 5
    splits = build_walk_forward_splits(n, n_splits=n_splits, gap=gap, embargo=embargo)

    for train_idx, test_idx in splits:
        separation = test_idx.min() - train_idx.max() - 1
        assert separation >= gap + embargo - 1


def test_test_windows_contiguous_and_equal_size():
    n, n_splits = 1500, 5
    splits = build_walk_forward_splits(n, n_splits=n_splits)

    sizes = [len(test_idx) for _, test_idx in splits]
    assert all(s == sizes[0] for s in sizes)

    # los tests cubren el final de la serie sin huecos
    assert splits[-1][1].max() == n - 1
    for (_, t1), (tr2, t2) in zip(splits, splits[1:]):
        assert t2.min() == t1.max() + 1


def test_all_indices_covered_exactly_once_in_test():
    n, n_splits = 1200, 4
    splits = build_walk_forward_splits(n, n_splits=n_splits)
    test_indices = np.concatenate([t for _, t in splits])
    assert len(test_indices) == len(np.unique(test_indices))


def test_default_gap_covers_max_feature_lookback():
    """el gap default debe ser >= al lookback maximo de features"""
    assert MAX_FEATURE_LOOKBACK_DAYS >= 60


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        build_walk_forward_splits(100, n_splits=5, gap=60, embargo=5)


def test_sklearn_style_splitter():
    X = np.zeros((1500, 1))
    splitter = PurgedTimeSeriesSplit(n_splits=3)
    folds = list(splitter.split(X))
    assert len(folds) == 3
    assert splitter.get_n_splits() == 3
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
