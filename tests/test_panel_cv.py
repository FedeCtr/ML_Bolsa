"""tests del walk-forward de panel (multiples tickers por fecha)"""
import numpy as np
import pandas as pd
import pytest

from src.ml.cv import build_panel_walk_forward_splits


def _panel_index(tickers, n_days):
    dates = pd.bdate_range('2022-01-03', periods=n_days)
    return pd.DatetimeIndex(
        [d for d in dates for _ in tickers], name='Date'
    )


def test_no_date_overlap_between_train_and_test():
    """ninguna fecha del train puede aparecer en el test (anti fuga cross-sectional)"""
    tickers = ['A', 'B', 'C']
    idx = _panel_index(tickers, 1500)
    splits = build_panel_walk_forward_splits(idx, n_splits=5, gap_days=60, embargo_days=5)

    for train_rows, test_rows in splits:
        train_dates = set(idx[train_rows])
        test_dates = set(idx[test_rows])
        assert not (train_dates & test_dates), "fuga: fechas compartidas entre train y test"
        # separacion temporal estricta
        assert max(train_dates) < min(test_dates)


def test_test_covers_all_tickers_same_dates():
    """el test de cada fold contiene todas las fechas de la ventana, x todos los tickers"""
    tickers = ['A', 'B', 'C']
    idx = _panel_index(tickers, 1200)
    splits = build_panel_walk_forward_splits(idx, n_splits=4)

    for train_rows, test_rows in splits:
        test_dates = idx[test_rows]
        counts = pd.Series(test_dates).value_counts()
        # cada fecha del test debe tener los 3 tickers
        assert (counts == 3).all()


def test_train_includes_all_ticker_past():
    """el train de un fold tiene el pasado de TODOS los tickers (no solo el ultimo bloque)"""
    tickers = ['A', 'B', 'C']
    idx = _panel_index(tickers, 1500)
    splits = build_panel_walk_forward_splits(idx, n_splits=5, gap_days=60, embargo_days=5)

    train_rows, _ = splits[2]  # un fold intermedio
    per_date = pd.Series(idx[train_rows]).value_counts()
    # la mayoria de las fechas del train deben tener los 3 tickers
    assert (per_date == 3).mean() > 0.95


def test_positional_indices_valid():
    idx = _panel_index(['A', 'B'], 900)
    splits = build_panel_walk_forward_splits(idx, n_splits=3)
    n = len(idx)
    for train_rows, test_rows in splits:
        assert train_rows.min() >= 0 and train_rows.max() < n
        assert test_rows.min() >= 0 and test_rows.max() < n
        assert not (set(train_rows) & set(test_rows))
