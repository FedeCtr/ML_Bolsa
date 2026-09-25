"""
Calibra las probabilidades del ensemble y elige el umbral de confianza con
datos OOF (out-of-fold), no con un numero arbitrario.

Requiere haber entrenado antes: python scripts/train_advanced.py
(que guarda models/advanced_oof.pkl con probabilidades out-of-fold).

Uso:
    python scripts/calibrate_thresholds.py
    python scripts/calibrate_thresholds.py --method platt
    python scripts/calibrate_thresholds.py --min-signals 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.ml.calibration import ConfidenceCalibrator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Calibracion y analisis de umbrales')
    parser.add_argument('--method', choices=['isotonic', 'platt'], default='isotonic',
                        help='metodo de calibracion (default: isotonic)')
    parser.add_argument('--min-signals', type=int, default=30,
                        help='senales OOF minimas para aceptar un umbral')
    args = parser.parse_args()

    oof_path = Path('models/advanced_oof.pkl')
    if not oof_path.exists():
        logger.error(f"no existe {oof_path}. Entrena primero:")
        logger.error("  python scripts/train_advanced.py")
        sys.exit(1)

    import joblib
    oof = joblib.load(oof_path)
    probs = oof['probabilities']
    targets = oof['targets']

    logger.info(f"OOF cargado: {len(probs)} muestras")
    logger.info(f"Precision de clase mayoritaria (baseline): {max(targets.mean(), 1 - targets.mean()):.4f}")

    # 1) calibrar
    calibrator = ConfidenceCalibrator(method=args.method).fit(probs, targets)

    # 2) analizar umbrales sobre probabilidades CALIBRADAS
    calibrated = calibrator.transform(probs)
    analysis = calibrator.analyze_thresholds(calibrated, targets, min_signals=args.min_signals)

    # tabla de precision por umbral
    logger.info("\n" + "=" * 72)
    logger.info(f"{'umbral':>8} | {'senales':>8} | {'tasa senal':>10} | {'precision real':>14}")
    logger.info("-" * 72)
    for row in analysis['analysis']:
        prec = f"{row['precision']:.4f}" if row['precision'] == row['precision'] else 'n/a'
        logger.info(
            f"{row['threshold']:>8.3f} | {row['signals']:>8d} | "
            f"{row['signal_rate']:>10.2%} | {prec:>14}"
        )
    logger.info("=" * 72)
    logger.info(f"Baseline (clase mayoritaria): {analysis['baseline_accuracy']:.4f}")
    logger.info(f"UMBRAL SELECCIONADO: {analysis['selected_threshold']:.3f}")

    # 3) persistir calibrador + umbral para produccion
    calibrator.save()
    logger.info("\nGuardado en models/calibration.pkl")
    logger.info("El predictor usara este umbral en produccion (ver src/ml/advanced_predictor.py)")


if __name__ == '__main__':
    main()
