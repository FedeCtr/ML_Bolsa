"""tests de calibracion de probabilidades y analisis de umbrales"""
import numpy as np
import pytest

from src.ml.calibration import ConfidenceCalibrator


def _synthetic_oof(n: int = 2000, seed: int = 42):
    """probabilidades OOF sinteticas con senal real pero imperfecta"""
    rng = np.random.default_rng(seed)
    true_score = rng.uniform(0.35, 0.65, size=n)      # senal debil
    probs = np.clip(true_score + rng.normal(0, 0.03, size=n), 0.01, 0.99)
    # la clase real depende de la senal + ruido: accuracy de base ~56%
    targets = (rng.uniform(size=n) < (probs - 0.5) * 2 + 0.5).astype(int)
    return probs, targets


def test_platt_calibration_improves_monotonicity():
    probs, targets = _synthetic_oof()
    cal = ConfidenceCalibrator(method='platt').fit(probs, targets)
    calibrated = cal.transform(probs)

    # todos los valores en (0,1)
    assert (calibrated > 0).all() and (calibrated < 1).all()
    # platt es monotona creciente: mayor prob cruda => mayor calibrada
    order = np.argsort(probs)
    assert (np.diff(calibrated[order]) >= -1e-9).all()


def test_isotonic_calibration_bounds():
    probs, targets = _synthetic_oof()
    cal = ConfidenceCalibrator(method='isotonic').fit(probs, targets)
    calibrated = cal.transform(probs)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_higher_threshold_selects_higher_precision():
    """a mayor umbral, la precision no deberia ser peor (senal monotona)"""
    probs, targets = _synthetic_oof()
    cal = ConfidenceCalibrator(method='platt').fit(probs, targets)
    analysis = cal.analyze_thresholds(probs, targets, min_signals=20)

    rows = [r for r in analysis['analysis'] if r['signals'] >= 20]
    low = next(r for r in rows if r['threshold'] == 0.5)
    high = next(r for r in rows if r['threshold'] == 0.65)
    assert high['precision'] >= low['precision'] - 0.03  # tolerancia estadistica


def test_selected_threshold_within_valid_range():
    probs, targets = _synthetic_oof()
    cal = ConfidenceCalibrator(method='isotonic').fit(probs, targets)
    analysis = cal.analyze_thresholds(probs, targets, min_signals=20)
    assert 0.5 <= analysis['selected_threshold'] <= 0.8


def test_save_and_load_roundtrip(tmp_path):
    probs, targets = _synthetic_oof(n=800)
    cal = ConfidenceCalibrator(method='platt').fit(probs, targets)
    cal.analyze_thresholds(probs, targets, min_signals=20)
    path = tmp_path / "calibration.pkl"
    cal.save(path=str(path))

    loaded = ConfidenceCalibrator.load(path=str(path))
    original = cal.transform(probs)
    restored = loaded.transform(probs)
    np.testing.assert_allclose(original, restored)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ConfidenceCalibrator().transform(np.array([0.5, 0.6]))


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        ConfidenceCalibrator(method='chichi')
