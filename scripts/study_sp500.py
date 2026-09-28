"""
Estudio de generalizacion: expectativa del modelo v2 sobre el S&P 500 completo.

Pregunta de negocio: el edge validado en 7 mega-cap tech (PF 1.505) es
sistemico (aplica a todo el mercado) o construimos un "Mega-Cap Tech Algo"?

Metodo:
  1. Universo S&P 500 (Wikipedia, con sectores GICS).
  2. Descarga 5y de OHLCV a data/raw_sp500/ (directorio separado del cache
     de entrenamiento) + SPY/^VIX de contexto.
  3. Features tecnicas identicas al entrenamiento y probabilidad del modelo
     de PRODUCCION (ensemble + calibracion isotonica) para cada ticker.
     Los ~493 tickers ajenos al entrenamiento son out-of-sample por
     universo; las 7 mega-cap originales se reportan aparte como control
     (ventaja in-sample declarada).
  4. Simulacion con la configuracion validada del informe: largo-solo,
     p>=0.50, VIX<=30, R/R 1:2, 1 posicion/ticker, max 3 concurrentes,
     holding max 20 sesiones, costos 10 pb, SL-first.

Salidas:
  - reports/sp500_expectancy.json (resumen para /api/expectancy)
  - reports/sp500_study_summary.csv (estadistica por ticker/sector)

Uso:  venv/Scripts/python scripts/study_sp500.py [--max-tickers N] [--refresh]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd

from src.data.collector import DataCollector
from src.data.processor import TechnicalProcessor
from src.data.universe import fetch_sp500_universe
from src.ml.calibration import ConfidenceCalibrator
from src.ml.expectancy import (
    attach_market_data,
    simulate_expectancy,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

STUDY_DIR = Path('data/raw_sp500')
TRAIN_TICKERS = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'}
CONFIG = dict(prob_min=0.50, rr_multiple=2.0, vix_max=30.0, allow_short=False,
              one_position_per_ticker=True, max_concurrent=3, max_hold=20,
              risk_per_trade=0.01)


# ----------------------------------------------------------------------
# Descarga
# ----------------------------------------------------------------------

def download_universe(tickers: list, refresh: bool = False) -> list:
    """descarga 5y de cada ticker al directorio del estudio; devuelve OKs."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    collector = DataCollector()
    collector.raw_dir = str(STUDY_DIR)   # separado del cache de entrenamiento

    pendientes = []
    for t in tickers:
        p = STUDY_DIR / f"{t}_raw.csv"
        if refresh or not p.exists() or p.stat().st_size < 5000:
            pendientes.append(t)
    if pendientes:
        logger.info(f"descargando {len(pendientes)}/{len(tickers)} tickers...")
        ok = collector.download_batch(pendientes, period='5y', max_workers=8)
        logger.info(f"descarga OK: {len(ok)}")
    # contexto de mercado para las features y los gates
    for t in ['SPY', '^VIX']:
        p = STUDY_DIR / f"{t}_raw.csv"
        if not p.exists():
            collector.download_ticker(t, '5y')
    validos = [t for t in tickers if (STUDY_DIR / f"{t}_raw.csv").exists()]
    logger.info(f"tickers con datos: {len(validos)}/{len(tickers)}")
    return validos


# ----------------------------------------------------------------------
# Prediccion del modelo de produccion por ticker
# ----------------------------------------------------------------------

def predict_ticker(processor: TechnicalProcessor, spy: pd.DataFrame,
                   vix: pd.DataFrame, ensemble, feature_cols: list,
                   calibrator: ConfidenceCalibrator, ticker: str):
    """devuelve DataFrame [date, ticker, prob_up] con probs calibradas."""
    path = STUDY_DIR / f"{ticker}_raw.csv"
    df = pd.read_csv(path)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], utc=True)
    df = df.drop_duplicates(subset=df.columns[0], keep='last').set_index(df.columns[0]).sort_index()
    # el procesador necesita DatetimeIndex para las features temporales
    df.index = pd.DatetimeIndex(df.index)
    df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix)

    X = df[[c for c in feature_cols if c in df.columns]].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < 60:
        return None
    proba = ensemble.predict_proba(X)
    idx_one = int(np.where(ensemble.classes_ == 1)[0][0])
    p_raw = proba[:, idx_one]
    p_cal = calibrator.transform(p_raw)

    return pd.DataFrame({
        # fechas como '%Y-%m-%d' strings: son las claves del calendario que
        # usa attach_market_data contra los CSV
        'date': [str(d.date()) for d in X.index],
        'ticker': ticker,
        'prob_up': p_cal,
    }).reset_index(drop=True)


# ----------------------------------------------------------------------
# Analisis
# ----------------------------------------------------------------------

def _stats_from_result(res) -> dict:
    return {
        'n_trades': res.n_trades, 'win_rate': round(res.win_rate, 4),
        'profit_factor': round(res.profit_factor, 3),
        'expectancy_r': round(res.expectancy_r, 4),
        'sharpe': round(res.sharpe, 2),
        'max_drawdown_pct': round(res.max_drawdown_pct, 2),
        'total_return_pct': round(res.total_return_pct, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-tickers', type=int, default=0,
                        help='limitar universo (0 = todo el S&P 500)')
    parser.add_argument('--refresh', action='store_true',
                        help='re-descargar datos ignorando el cache local')
    args = parser.parse_args()

    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    # 1) universo
    uni = fetch_sp500_universe()
    tickers = uni['ticker'].tolist()
    sector_map = dict(zip(uni['ticker'], uni['sector']))
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    tickers = [t for t in tickers if t not in TRAIN_TICKERS]   # control aparte
    logger.info(f"universo a evaluar: {len(tickers)} tickers (excluye {len(TRAIN_TICKERS)} de entrenamiento)")

    # 2) descarga
    validos = download_universe(tickers, refresh=args.refresh)
    if len(validos) < 50:
        raise RuntimeError(f"solo {len(validos)} tickers con datos; estudio no representativo")

    # 3) modelo de produccion + contexto
    ensemble = joblib.load('models/advanced_ensemble.pkl')
    feature_cols = joblib.load('models/advanced_features.pkl')
    calibrator = ConfidenceCalibrator.load()

    processor = TechnicalProcessor()
    # contexto con DatetimeIndex (el procesador accede a atributos de fecha)
    spy = pd.read_csv(STUDY_DIR / 'SPY_raw.csv')
    spy[spy.columns[0]] = pd.to_datetime(spy[spy.columns[0]], utc=True)
    spy = spy.drop_duplicates(subset=spy.columns[0], keep='last').set_index(spy.columns[0]).sort_index()
    spy.index = pd.DatetimeIndex(spy.index)
    vix = pd.read_csv(STUDY_DIR / '^VIX_raw.csv')
    vix[vix.columns[0]] = pd.to_datetime(vix[vix.columns[0]], utc=True)
    vix = vix.drop_duplicates(subset=vix.columns[0], keep='last').set_index(vix.columns[0]).sort_index()
    vix.index = pd.DatetimeIndex(vix.index)

    # 4) predicciones por ticker
    frames = []
    n_err = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(predict_ticker, processor, spy, vix, ensemble,
                               feature_cols, calibrator, t): t for t in validos}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                out = fut.result()
                if out is not None:
                    frames.append(out)
            except Exception as e:   # noqa: BLE001
                n_err += 1
                logger.debug(f"{t}: error en prediccion: {e}")
    preds = pd.concat(frames, ignore_index=True)
    logger.info(f"predicciones: {len(preds)} filas, {preds['ticker'].nunique()} tickers "
                f"(errores: {n_err})")
    if preds.empty:
        raise RuntimeError("ningun ticker produjo predicciones; revisar logs")

    # 5) datos de ejecucion + simulacion global (portafolio compartido)
    preds = attach_market_data(preds, raw_dir=str(STUDY_DIR))
    res_all = simulate_expectancy(preds, **CONFIG, raw_dir=str(STUDY_DIR))

    # 6) control: las 7 mega-cap con el mismo pipeline
    preds_tech = preds[preds['ticker'].isin(TRAIN_TICKERS)]
    res_tech = simulate_expectancy(preds_tech, **CONFIG, raw_dir=str(STUDY_DIR)) \
        if len(preds_tech) else None

    # 7) por ano y por sector
    per_year = []
    for y in sorted(preds['date'].str[:4].unique()):
        sub = preds[preds['date'].str[:4] == y]
        r = simulate_expectancy(sub, **CONFIG, raw_dir=str(STUDY_DIR))
        d = _stats_from_result(r)
        d['year'] = y
        per_year.append(d)

    per_sector = []
    preds['sector'] = preds['ticker'].map(sector_map).fillna('Desconocido')
    for sector, sub in preds.groupby('sector'):
        r = simulate_expectancy(sub, **CONFIG, raw_dir=str(STUDY_DIR))
        if r.n_trades < 30:
            continue
        d = _stats_from_result(r)
        d['sector'] = sector
        d['n_tickers'] = int(sub['ticker'].nunique())
        per_sector.append(d)
    per_sector.sort(key=lambda d: -d['n_trades'])

    summary = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'config': {k: v for k, v in CONFIG.items()},
        'universe_tickers': int(preds['ticker'].nunique()),
        'sp500_ex_tech': _stats_from_result(res_all),
        'control_tech7': _stats_from_result(res_tech) if res_tech else None,
        'per_year': per_year,
        'per_sector': per_sector,
        'equity': [round(x, 5) for x in res_all.equity[::max(1, len(res_all.equity)//400)]],
    }
    Path('reports').mkdir(exist_ok=True)
    Path('reports/sp500_expectancy.json').write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')

    logger.info("=" * 74)
    logger.info(f"S&P 500 ex-tech : PF {res_all.profit_factor:.3f} | WR {res_all.win_rate:.1%} "
                f"| E[R] {res_all.expectancy_r:+.3f}R | Sharpe {res_all.sharpe:.2f} "
                f"| {res_all.n_trades} ops")
    if res_tech:
        logger.info(f"Control tech-7  : PF {res_tech.profit_factor:.3f} | WR {res_tech.win_rate:.1%} "
                    f"| E[R] {res_tech.expectancy_r:+.3f}R | {res_tech.n_trades} ops")
    for d in per_year:
        logger.info(f"  {d['year']}: PF {d['profit_factor']:.3f} ({d['n_trades']} ops)")
    logger.info("=" * 74)
    logger.info("resumen: reports/sp500_expectancy.json")


if __name__ == '__main__':
    main()
