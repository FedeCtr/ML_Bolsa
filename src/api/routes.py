"""
endpoints de la API (v2 comercial)

Nuevos (spec 2-5): /api/scan (jobs), /api/top-signals, /api/watchlists,
/api/paper/*, /api/model-performance, /api/universe/info.
Originales mantenidos: /api/signal, /api/chart, /api/backtest, /api/predict.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from flask import jsonify, render_template, request

from ..ml.predictor import StockPredictor
from ..ml.advanced_predictor import AdvancedPredictor
from ..ml.signal_engine import SIDE_BUY, SIDE_SELL, build_signal
from ..ml.screener import (UNIVERSES, apply_filters, last_results,
                           scan_status, start_scan)
from ..ml.watchlist import WatchlistStore
from ..trading.paper import PaperTrader
from ..data.collector import DataCollector
from ..data.processor import TechnicalProcessor
from ..data.universe import (FALLBACK_TICKERS, get_name_map, get_sector_map,
                             get_universe)
from ..backtesting.engine import run_walk_forward_backtest
from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


def _clean_nan(obj):
    """NaN/inf -> None recursivamente (JSON valido)"""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def register_routes(app):
    """registra todos los endpoints"""

    predictor = None
    advanced_predictor = None
    watchlists = WatchlistStore()
    paper = PaperTrader()

    config = Config()
    advanced_model_path = os.path.join(config.models_dir, 'advanced_ensemble.pkl')

    if os.path.exists(advanced_model_path):
        try:
            advanced_predictor = AdvancedPredictor()
            logger.info("predictor AVANZADO inicializado (ensemble + calibracion)")
        except Exception as e:
            logger.warning(f"no se pudo cargar predictor avanzado: {e}")
    if advanced_predictor is None:
        try:
            predictor = StockPredictor()
        except Exception:
            predictor = None

    def _download_with_context(collector, processor, ticker, period):
        df = collector.download_ticker(ticker, period)
        if df is None or df.empty:
            return None
        spy = collector.download_ticker('SPY', period)
        vix = collector.download_ticker('^VIX', period)
        return processor.process_all_indicators(df, spy_df=spy, vix_df=vix), vix

    def _load_oof():
        path = os.path.join(config.models_dir, 'advanced_oof.pkl')
        if not os.path.exists(path):
            return None
        data = joblib.load(path)
        return data if data.get('dates') else None

    # ------------------------------------------------------------------
    # paginas
    # ------------------------------------------------------------------

    @app.route('/')
    def home():
        return render_template('dashboard.html')

    @app.route('/health')
    def health():
        return jsonify(_clean_nan({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_type': 'advanced' if advanced_predictor else ('basic' if predictor else 'none'),
            'model_loaded': advanced_predictor is not None or predictor is not None,
            'scan': scan_status(),
        }))

    # ------------------------------------------------------------------
    # escaneo del universo (jobs en segundo plano)
    # ------------------------------------------------------------------

    @app.route('/api/scan', methods=['POST', 'GET'])
    def api_scan():
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        body = request.get_json(silent=True) or {}
        universe = body.get('universe', request.args.get('universe', 'mega'))
        force = bool(body.get('force', request.args.get('force') == '1'))
        period = body.get('period', request.args.get('period', '3mo'))
        res = start_scan(advanced_predictor, universe=universe, period=period, force=force)
        return jsonify(_clean_nan({
            'job_id': res['job_id'],
            'started': res['started'],
            'status': res['status'],
            'n_results': len(res['results']),
            'results': res['results'][:20] if res['results'] else [],
        }))

    @app.route('/api/scan/status')
    def api_scan_status():
        st = scan_status()
        out = {'status': st, 'n_results': len(last_results())}
        if not st['running'] and last_results():
            preview = apply_filters(last_results(), limit=10)
            out['preview'] = preview
        return jsonify(_clean_nan(out))

    @app.route('/api/screener')
    def api_screener():
        """resultados del ultimo escaneo con filtros (requiere scan previo)"""
        rows = last_results()
        if not rows:
            return jsonify({'error': 'no hay escaneo; lanza POST /api/scan', 'status': scan_status()}), 409
        f = request.args
        filtered = apply_filters(
            rows,
            side=f.get('side'),
            signal=f.get('signal'),
            min_confidence=f.get('min_confidence', type=float),
            min_conviction=f.get('min_conviction', type=float),
            max_atr_pct=f.get('max_atr_pct', type=float),
            min_volume_ratio=f.get('min_volume_ratio', type=float),
            sector=f.get('sector'),
            watchlist=watchlists.get(f['watchlist'])['tickers'] if f.get('watchlist') else None,
            only_tradable=f.get('only_tradable') == '1',
            limit=f.get('limit', type=int),
        )
        return jsonify(_clean_nan({
            'status': scan_status(),
            'total_scanned': len(rows),
            'total_returned': len(filtered),
            'buy_count': sum(1 for r in rows if r.get('side') == SIDE_BUY),
            'sell_count': sum(1 for r in rows if r.get('side') == SIDE_SELL),
            'suspended_count': sum(1 for r in rows if not r.get('trading_allowed', True)),
            'results': filtered,
        }))

    @app.route('/api/top-signals')
    def api_top_signals():
        """panel: senales activas de alta probabilidad (spec 4)"""
        rows = last_results()
        if not rows:
            return jsonify({'available': False, 'reason': 'lanza un escaneo', 'status': scan_status()})
        tradable = [r for r in rows if r.get('trading_allowed', True)]
        top = sorted(tradable, key=lambda r: r.get('conviction_pct', 0), reverse=True)[:10]
        return jsonify(_clean_nan({
            'available': True,
            'signals': top,
            'buy_count': sum(1 for r in tradable if r.get('side') == SIDE_BUY),
            'sell_count': sum(1 for r in tradable if r.get('side') == SIDE_SELL),
        }))

    # ------------------------------------------------------------------
    # watchlists
    # ------------------------------------------------------------------

    @app.route('/api/watchlists', methods=['GET', 'POST'])
    def api_watchlists():
        if request.method == 'GET':
            return jsonify({'watchlists': watchlists.list()})
        body = request.get_json(silent=True) or {}
        name = (body.get('name') or '').strip()
        if not name:
            return jsonify({'error': 'name requerido'}), 400
        try:
            wl = watchlists.create(name, body.get('tickers'))
            return jsonify(wl), 201
        except ValueError as e:
            return jsonify({'error': str(e)}), 409

    @app.route('/api/watchlists/<name>', methods=['GET', 'DELETE'])
    def api_watchlist(name):
        if request.method == 'GET':
            wl = watchlists.get(name)
            return jsonify(wl) if wl else (jsonify({'error': 'no existe'}), 404)
        return jsonify({'deleted': watchlists.delete(name)})

    @app.route('/api/watchlists/<name>/tickers', methods=['POST', 'DELETE'])
    def api_watchlist_tickers(name):
        body = request.get_json(silent=True) or {}
        tickers = [t for t in body.get('tickers', []) if t.strip()]
        if not tickers:
            return jsonify({'error': 'tickers requerido'}), 400
        try:
            wl = (watchlists.add_tickers(name, tickers) if request.method == 'POST'
                  else watchlists.remove_tickers(name, tickers))
            return jsonify(wl)
        except ValueError as e:
            return jsonify({'error': str(e)}), 404

    # ------------------------------------------------------------------
    # senal individual + chart + backtest (con regimen)
    # ------------------------------------------------------------------

    @app.route('/api/signal/<ticker>')
    def api_signal(ticker):
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        try:
            collector, processor = DataCollector(), TechnicalProcessor()
            period = request.args.get('period', '1y')
            res = _download_with_context(collector, processor, ticker.upper(), period)
            if res is None:
                return jsonify({'error': f'sin datos para {ticker}'}), 404
            df, vix = res
            X = df[advanced_predictor.feature_cols].tail(1)
            if X.isna().any(axis=1).iloc[0]:
                return jsonify({'error': 'datos insuficientes'}), 400

            from ..ml.regime import detect_regime
            prob_up = advanced_predictor._raw_probability_up(X)
            regime = detect_regime(df, vix_series=vix['Close'] if vix is not None else None)
            sig = build_signal(ticker.upper(), df, prob_up,
                               as_of=str(df.index[-1].date()), regime=regime)
            payload = sig.to_dict()
            payload['sector'] = get_sector_map().get(ticker.upper(), 'Otros')
            payload['name'] = get_name_map().get(ticker.upper(), ticker.upper())
            payload['levels_nature'] = 'riesgo_determinista_no_modelo'

            # registro automatico en paper trading
            sig_id = paper.record_signal(payload)
            payload['paper_signal_id'] = sig_id
            return jsonify(_clean_nan(payload))
        except Exception as e:
            logger.error(f"error en signal {ticker}: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/chart/<ticker>')
    def api_chart(ticker):
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        try:
            import pandas as pd
            collector, processor = DataCollector(), TechnicalProcessor()
            period = request.args.get('period', '2y')
            days = request.args.get('days', 260, type=int)
            df = collector.download_ticker(ticker.upper(), period)
            if df is None or df.empty:
                return jsonify({'error': f'sin datos para {ticker}'}), 404
            spy = collector.download_ticker('SPY', period)
            vix = collector.download_ticker('^VIX', period)
            df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix).tail(days)

            out = pd.DataFrame(index=df.index.astype(str))
            out['time'] = [str(d.date()) for d in df.index]
            for c in ('open', 'high', 'low', 'close'):
                out[c] = df[c.capitalize()].round(2).values
            out['volume'] = df['Volume'].values
            if len(df) > 210:
                out['ema20'] = df['Close'].ewm(span=20, adjust=False).mean().round(2).values
                out['ema50'] = df['Close'].ewm(span=50, adjust=False).mean().round(2).values
                out['ema200'] = df['Close'].ewm(span=200, adjust=False).mean().round(2).values
                bb_mid = df['Close'].rolling(20).mean()
                bb_std = df['Close'].rolling(20).std()
                out['bb_upper'] = (bb_mid + 2 * bb_std).round(2).values
                out['bb_lower'] = (bb_mid - 2 * bb_std).round(2).values
            out['rsi'] = df['rsi'].round(1).values if 'rsi' in df.columns else None
            out['macd'] = df['macd'].round(3).values if 'macd' in df.columns else None
            out['macd_signal'] = df['macd_signal'].round(3).values if 'macd_signal' in df.columns else None

            from ..ml.signal_engine import find_pivot_levels
            all_sup, all_res = find_pivot_levels(df['High'], df['Low'])
            last_px = float(df['Close'].iloc[-1])
            supports = sorted(all_sup, key=lambda v: abs(v - last_px))[:4]
            resistances = sorted(all_res, key=lambda v: abs(v - last_px))[:4]

            oof = _load_oof()
            oof_signals = []
            if oof is not None:
                for d, t, p, y in zip(oof['dates'], oof['tickers'],
                                      oof['probabilities'], oof['targets']):
                    if t == ticker.upper():
                        oof_signals.append({
                            'date': d, 'prob_up': round(float(p), 4),
                            'signal': 'COMPRA' if p >= 0.5 else 'VENTA',
                            'actual_up': int(y),
                            'correct': int((p >= 0.5) == bool(y)),
                        })

            return jsonify(_clean_nan({
                'ticker': ticker.upper(),
                'candles': out.to_dict('records'),
                'supports': sorted(round(s, 2) for s in supports),
                'resistances': sorted(round(r, 2) for r in resistances),
                'oof_signals': oof_signals,
            }))
        except Exception as e:
            logger.error(f"error en chart {ticker}: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/backtest/<ticker>')
    def api_backtest(ticker):
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        try:
            period = request.args.get('period', '3y')
            threshold = request.args.get('threshold', type=float) \
                or advanced_predictor.confidence_threshold or 0.5
            collector, processor = DataCollector(), TechnicalProcessor()
            res = _download_with_context(collector, processor, ticker.upper(), period)
            if res is None:
                return jsonify({'error': f'sin datos para {ticker}'}), 404
            df, _ = res
            df['target_direccion'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            from lightgbm import LGBMClassifier
            from ..ml.advanced_trainer import DEFAULT_FEATURE_COLS

            def factory(X_tr, y_tr):
                m = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=42, verbose=-1, n_jobs=-1)
                m.fit(X_tr, y_tr)
                return m

            results = run_walk_forward_backtest(
                df, factory, n_splits=5, purge_days=60, embargo_days=5,
                feature_cols=[c for c in DEFAULT_FEATURE_COLS if c in df.columns],
                ticker=ticker.upper(), threshold=threshold,
            )
            result = results['walk_forward']
            equity = result.portfolio[['equity']].copy()
            equity['time'] = [str(d.date()) for d in equity.index]
            equity['value'] = equity['equity'].round(2).values
            return jsonify(_clean_nan({
                'ticker': ticker.upper(),
                'threshold_used': round(threshold, 3),
                'metrics': result.metrics,
                'trade_stats': result.trade_stats,
                'regime_metrics': result.regime_metrics,
                'config': result.config,
                'equity_curve': equity[['time', 'value']].to_dict('records'),
            }))
        except Exception as e:
            logger.error(f"error en backtest {ticker}: {e}")
            return jsonify({'error': str(e)}), 500

    # ------------------------------------------------------------------
    # paper trading (forward testing, spec 5)
    # ------------------------------------------------------------------

    @app.route('/api/paper/resolve', methods=['POST'])
    def api_paper_resolve():
        def provider(t):
            df = DataCollector().download_ticker(t, '1y')
            return df
        n = paper.resolve_pending(provider)
        return jsonify({'resolved': n, 'performance': paper.performance()})

    @app.route('/api/paper/performance')
    def api_paper_performance():
        return jsonify(_clean_nan(paper.performance()))

    @app.route('/api/paper/signals')
    def api_paper_signals():
        return jsonify(_clean_nan({'signals': paper.list_signals(
            limit=request.args.get('limit', 100, type=int))}))

    # ------------------------------------------------------------------
    # rendimiento historico del modelo (OOF) y meta
    # ------------------------------------------------------------------

    @app.route('/api/model-performance')
    def api_model_performance():
        oof = _load_oof()
        if oof is None:
            return jsonify({'available': False,
                            'reason': 'reentrena para generar OOF con fechas'})
        probs = np.asarray(oof['probabilities'])
        targets = np.asarray(oof['targets'])
        edges = [0.0, 0.35, 0.45, 0.55, 0.65, 1.0]
        labels = ['<=35%', '35-45%', '45-55%', '55-65%', '>=65%']
        import pandas as pd
        bins = pd.cut(probs, bins=edges, labels=labels, include_lowest=True)
        bands = []
        for label in labels:
            mask = bins == label
            n = int(mask.sum())
            if n:
                band = targets[mask]
                acc = float((band == 0).mean()) if label in ('<=35%', '35-45%') else float(band.mean())
                bands.append({'band': label, 'n': n, 'accuracy': round(acc, 4),
                              'share': round(n / len(probs), 4)})
        overall = float(((probs >= 0.5).astype(int) == targets).mean())
        meta_path = os.path.join(config.models_dir, 'advanced_metadata.pkl')
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        return jsonify(_clean_nan({
            'available': True,
            'n_samples': int(len(probs)),
            'overall_accuracy': round(overall, 4),
            'bands': bands,
            'range': [oof['dates'][0], oof['dates'][-1]],
            'trained_at': meta.get('trained_at'),
            'walk_forward': meta.get('walk_forward_metrics'),
        }))

    @app.route('/api/meta')
    def api_meta():
        meta_path = os.path.join(config.models_dir, 'advanced_metadata.pkl')
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        calib_threshold = None
        calib_path = os.path.join(config.models_dir, 'calibration.pkl')
        if os.path.exists(calib_path):
            calib_threshold = joblib.load(calib_path).get('selected_threshold')
        return jsonify(_clean_nan({
            'model_type': 'advanced' if advanced_predictor else 'basic',
            'calibrated': advanced_predictor is not None and advanced_predictor.calibrator is not None,
            'signal_threshold': calib_threshold if calib_threshold is not None
            else (advanced_predictor.confidence_threshold if advanced_predictor else None),
            'trained_at': meta.get('trained_at'),
            'walk_forward': meta.get('walk_forward_metrics', {}),
            'universes': {k: len(v) if v else 500 for k, v in UNIVERSES.items()},
        }))

    @app.route('/api/universe/info')
    def api_universe_info():
        try:
            tickers = get_universe(limit=500)
            return jsonify({'total': len(tickers), 'sample': tickers[:25],
                            'sectors': sorted(set(get_sector_map().values()))})
        except Exception as e:
            return jsonify({'error': str(e), 'fallback_size': len(FALLBACK_TICKERS)}), 500

    # ------------------------------------------------------------------
    # compatibilidad
    # ------------------------------------------------------------------

    @app.route('/api/predict/<ticker>', methods=['GET'])
    def predict_ticker(ticker):
        active = advanced_predictor or predictor
        if active is None:
            return jsonify({'error': 'modelo no disponible',
                            'message': 'entrena: python scripts/train_advanced.py'}), 400
        try:
            resultado = active.predict_ticker(ticker.upper())
            if resultado is None or 'error' in resultado:
                return jsonify({'error': resultado.get('error', 'sin datos'),
                                'ticker': ticker}), 404
            resultado['model_type'] = 'advanced' if advanced_predictor else 'basic'
            resultado['timestamp'] = datetime.now().isoformat()
            if 'recommendation' not in resultado and 'prediccion' in resultado:
                resultado['recommendation'] = resultado['prediccion'].upper()
            return jsonify(_clean_nan(resultado))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ------------------------------------------------------------------
    # Transparencia: expectativa matematica del sistema (Sprint 3)
    # ------------------------------------------------------------------
    _expectancy_cache: Dict = {'data': None, 'computed_at': None}
    _expectancy_lock = threading.Lock()
    EXPECTANCY_CONFIG = dict(prob_min=0.50, rr_multiple=2.0, vix_max=30.0,
                             allow_short=False, one_position_per_ticker=True,
                             max_concurrent=3, max_hold=20, risk_per_trade=0.01)

    def _compute_expectancy() -> Dict:
        """simula el sistema sobre el OOF del modelo en produccion (7 tech).
        Costoso (~5-10s): se cachea por proceso; TTL 6 horas."""
        now = datetime.now()
        if (_expectancy_cache['data'] is not None
                and _expectancy_cache['computed_at'] is not None
                and (now - _expectancy_cache['computed_at']).total_seconds() < 6 * 3600):
            return _expectancy_cache['data']
        with _expectancy_lock:
            # doble chequeo tras adquirir el lock
            if (_expectancy_cache['data'] is not None
                    and (datetime.now() - _expectancy_cache['computed_at']).total_seconds() < 6 * 3600):
                return _expectancy_cache['data']
            from ..ml.expectancy import load_oof_dataset, attach_market_data, simulate_expectancy
            df = load_oof_dataset()
            df = attach_market_data(df)
            res = simulate_expectancy(df, **EXPECTANCY_CONFIG)
            per_year = []
            for y in sorted(df['date'].str[:4].unique()):
                sub = df[df['date'].str[:4] == y]
                r = simulate_expectancy(sub, **EXPECTANCY_CONFIG)
                per_year.append({
                    'year': y, 'n_trades': r.n_trades,
                    'win_rate': round(r.win_rate, 4),
                    'profit_factor': round(r.profit_factor, 3),
                    'expectancy_r': round(r.expectancy_r, 4),
                })
            # benchmark SPY en el mismo periodo
            bench = {}
            spy_path = Path(Config().raw_data_dir) / 'SPY_raw.csv'
            if spy_path.exists():
                spy = pd.read_csv(spy_path)
                spy[spy.columns[0]] = pd.to_datetime(spy[spy.columns[0]], utc=True).dt.strftime('%Y-%m-%d')
                spy = spy.drop_duplicates(subset=spy.columns[0], keep='last') \
                    .set_index(spy.columns[0]).sort_index().loc[df['date'].min():df['date'].max()]
                if len(spy) > 50:
                    norm = spy['Close'] / spy['Close'].iloc[0]
                    rets = spy['Close'].pct_change().dropna()
                    years = max((pd.Timestamp(df['date'].max()) - pd.Timestamp(df['date'].min())).days / 365.25, 1e-9)
                    peak = np.maximum.accumulate(norm)
                    bench = {
                        'total_return_pct': round(float((norm.iloc[-1] - 1) * 100), 1),
                        'cagr_pct': round(float((norm.iloc[-1] ** (1 / years) - 1) * 100), 1),
                        'max_drawdown_pct': round(float(((norm - peak) / peak).min() * 100), 1),
                        'sharpe': round(float(rets.mean() / rets.std() * np.sqrt(252)), 2),
                    }
            data = {
                'generated_at': now.isoformat(),
                'config': EXPECTANCY_CONFIG,
                'tech7_oof': {
                    'n_trades': res.n_trades,
                    'win_rate': round(res.win_rate, 4),
                    'profit_factor': round(res.profit_factor, 3),
                    'expectancy_r': round(res.expectancy_r, 4),
                    'sharpe': round(res.sharpe, 2),
                    'max_drawdown_pct': round(res.max_drawdown_pct, 2),
                    'avg_hold_days': round(res.avg_hold_days, 2),
                    'per_year': per_year,
                },
                'benchmark_spy': bench,
                'sp500_study': _load_sp500_study(),
                'methodology': (
                    'Probabilidades out-of-sample del walk-forward purgado; entrada en apertura '
                    'de t+1, SL 1.5 ATR (1-5%), TP 2x riesgo, holding max 20 sesiones, costos 10pb, '
                    'criterio SL-first. Filtros: largo-solo, VIX<=30, 1 posicion por ticker, '
                    'max 3 concurrentes. La confianza del modelo NO es la precision.'
                ),
            }
            _expectancy_cache['data'] = data
            _expectancy_cache['computed_at'] = now
            return data

    def _load_sp500_study() -> Optional[Dict]:
        """estudio de generalizacion S&P 500 si ya fue ejecutado."""
        p = Path('reports/sp500_expectancy.json')
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return None

    @app.route('/api/expectancy')
    def api_expectancy():
        try:
            data = _compute_expectancy()
            # el estudio S&P 500 se re-lee en cada request: se actualiza cuando
            # el script de estudio termina, sin esperar al TTL de la cache
            data = dict(data)
            data['sp500_study'] = _load_sp500_study()
            return jsonify(_clean_nan(data))
        except FileNotFoundError as e:
            return jsonify({'error': f'datos OOF no disponibles: {e}'}), 503
        except Exception as e:
            logger.error(f"error en /api/expectancy: {e}")
            return jsonify({'error': str(e)}), 500

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'endpoint no encontrado'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"error interno: {error}")
        return jsonify({'error': 'error interno del servidor'}), 500
