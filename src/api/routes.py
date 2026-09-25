"""
endpoints de la API

Endpoints originales (predict, batch, recommendation) + nuevos para la
plataforma comercial: screener, senal por niveles, chart, backtest y
resumen OOF (heatmap de precision historica).
"""
import os
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from flask import jsonify, render_template, request

from ..ml.predictor import StockPredictor
from ..ml.advanced_predictor import AdvancedPredictor
from ..ml.signal_engine import SIDE_BUY, SIDE_SELL, build_signal
from ..ml.screener import DEFAULT_UNIVERSE, SECTORS, MarketScreener, rows_to_signal_rows
from ..backtesting.engine import run_walk_forward_backtest
from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


def _clean_nan(obj):
    """convierte NaN/inf a None recursivamente para JSON valido"""
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

    # ------------------------------------------------------------------
    # modelos (inicializacion unica al arrancar)
    # ------------------------------------------------------------------
    predictor = None
    advanced_predictor = None
    screener = MarketScreener(cache_ttl_seconds=300)

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
            logger.warning("no se pudo cargar predictor")

    def _download_with_context(collector, processor, ticker, period):
        """descarga un ticker + contexto de mercado (SPY/VIX)"""
        df = collector.download_ticker(ticker, period)
        if df is None or df.empty:
            return None
        spy = collector.download_ticker('SPY', period)
        vix = collector.download_ticker('^VIX', period)
        return processor.process_all_indicators(df, spy_df=spy, vix_df=vix)

    def _load_oof():
        """carga OOF si existe (con o sin fechas/ticker)"""
        path = os.path.join(config.models_dir, 'advanced_oof.pkl')
        if not os.path.exists(path):
            return None
        data = joblib.load(path)
        if data.get('dates') is None:
            return None
        return data

    # ------------------------------------------------------------------
    # paginas
    # ------------------------------------------------------------------

    @app.route('/')
    def home():
        return render_template('dashboard.html')

    @app.route('/health')
    def health():
        model_type = 'advanced' if advanced_predictor else ('basic' if predictor else 'none')
        model_loaded = advanced_predictor is not None or (predictor is not None and predictor.trainer.model is not None)
        return jsonify(_clean_nan({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'model_loaded': model_loaded,
        }))

    # ------------------------------------------------------------------
    # meta: info del modelo para el header del dashboard
    # ------------------------------------------------------------------

    @app.route('/api/meta')
    def api_meta():
        meta = {}
        meta_path = os.path.join(config.models_dir, 'advanced_metadata.pkl')
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
        calib_threshold = None
        calib_path = os.path.join(config.models_dir, 'calibration.pkl')
        if os.path.exists(calib_path):
            calib_threshold = joblib.load(calib_path).get('selected_threshold')
        wf = meta.get('walk_forward_metrics', {}) if meta else {}
        return jsonify(_clean_nan({
            'model_type': 'advanced' if advanced_predictor else 'basic',
            'calibrated': advanced_predictor is not None and advanced_predictor.calibrator is not None,
            'signal_threshold': calib_threshold if calib_threshold is not None
            else (advanced_predictor.confidence_threshold if advanced_predictor else None),
            'trained_at': meta.get('trained_at') if meta else None,
            'tickers': meta.get('tickers') if meta else [],
            'data_range': [meta.get('data_start'), meta.get('data_end')] if meta else None,
            'walk_forward': wf,
            'universe': DEFAULT_UNIVERSE,
        }))

    # ------------------------------------------------------------------
    # screener: escaneo del universo con filtros
    # ------------------------------------------------------------------

    @app.route('/api/screener')
    def api_screener():
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400

        side = request.args.get('side')            # largo | corto
        min_conf = request.args.get('min_confidence', type=float)
        sector = request.args.get('sector')
        max_atr = request.args.get('max_atr_pct', type=float)
        min_atr = request.args.get('min_atr_pct', type=float)
        limit = request.args.get('limit', type=int)
        force = request.args.get('force', '0') == '1'

        if force:
            screener._cache, screener._cache_ts = {}, 0.0

        raw = screener.scan(universe=DEFAULT_UNIVERSE)
        rows = rows_to_signal_rows(raw, advanced_predictor)
        filtered = MarketScreener.apply_filters(
            rows, side=side, min_confidence=min_conf, max_atr_pct=max_atr,
            min_atr_pct=min_atr, sector=sector,
        )
        if limit:
            filtered = filtered[:limit]

        return jsonify(_clean_nan({
            'total_scanned': len(rows),
            'total_returned': len(filtered),
            'buy_count': sum(1 for r in rows if r.get('side') == SIDE_BUY),
            'sell_count': sum(1 for r in rows if r.get('side') == SIDE_SELL),
            'sectors': sorted(set(SECTORS.values())),
            'results': filtered,
            'cached': bool(screener._cache),
            'timestamp': datetime.now().isoformat(),
        }))

    # ------------------------------------------------------------------
    # senal operable por niveles para un ticker
    # ------------------------------------------------------------------

    @app.route('/api/signal/<ticker>')
    def api_signal(ticker):
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        try:
            from ..data.collector import DataCollector
            from ..data.processor import TechnicalProcessor

            collector = DataCollector()
            processor = TechnicalProcessor()
            period = request.args.get('period', '1y')
            df = _download_with_context(collector, processor, ticker.upper(), period)
            if df is None:
                return jsonify({'error': f'sin datos para {ticker}'}), 404

            X = df[advanced_predictor.feature_cols].tail(1)
            if X.isna().any(axis=1).iloc[0]:
                return jsonify({'error': 'datos insuficientes'}), 400

            prob_up = advanced_predictor._raw_probability_up(X)
            sig = build_signal(ticker.upper(), df, prob_up, as_of=str(df.index[-1].date()))
            payload = sig.to_dict()
            payload['sector'] = SECTORS.get(ticker.upper(), 'Otros')
            payload['levels_nature'] = 'riesgo_determinista_no_modelo'
            return jsonify(_clean_nan(payload))
        except Exception as e:
            logger.error(f"error en signal {ticker}: {e}")
            return jsonify({'error': str(e)}), 500

    # ------------------------------------------------------------------
    # chart: OHLCV + indicadores + niveles + senales OOF historicas
    # ------------------------------------------------------------------

    @app.route('/api/chart/<ticker>')
    def api_chart(ticker):
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        try:
            from ..data.collector import DataCollector
            from ..data.processor import TechnicalProcessor

            collector = DataCollector()
            processor = TechnicalProcessor()
            period = request.args.get('period', '2y')
            days = request.args.get('days', 260, type=int)
            df = _download_with_context(collector, processor, ticker.upper(), period)
            if df is None:
                return jsonify({'error': f'sin datos para {ticker}'}), 404

            df = df.tail(days)
            # EMAs y Bollinger para overlay
            out = pd.DataFrame(index=df.index.astype(str))
            out['time'] = [str(d.date()) for d in df.index]
            out['open'] = df['Open'].round(2).values
            out['high'] = df['High'].round(2).values
            out['low'] = df['Low'].round(2).values
            out['close'] = df['Close'].round(2).values
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

            # niveles tecnicos (pivotes) para superponer:
            # los 4 de cada lado MAS CERCANOS al precio actual (los pivotes
            # lejanos tras una tendencia larga son ruido visual)
            from ..ml.signal_engine import find_pivot_levels
            all_sup, all_res = find_pivot_levels(df['High'], df['Low'])
            last_px = float(df['Close'].iloc[-1])
            supports = sorted(all_sup, key=lambda v: abs(v - last_px))[:4]
            resistances = sorted(all_res, key=lambda v: abs(v - last_px))[:4]

            # senales OOF historicas de este ticker (para heatmap/marcadores)
            oof = _load_oof()
            oof_signals = []
            if oof is not None:
                for d, t, p, y in zip(oof['dates'], oof['tickers'],
                                      oof['probabilities'], oof['targets']):
                    if t == ticker.upper():
                        oof_signals.append({
                            'date': d,
                            'prob_up': round(float(p), 4),
                            'signal': 'COMPRA' if p >= 0.55 else ('VENTA' if p <= 0.45 else 'NEUTRAL'),
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

    # ------------------------------------------------------------------
    # backtest on-demand por ticker
    # ------------------------------------------------------------------

    @app.route('/api/backtest/<ticker>')
    def api_backtest(ticker):
        if advanced_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        try:
            from ..data.collector import DataCollector
            from ..data.processor import TechnicalProcessor
            from ..ml.advanced_trainer import DEFAULT_FEATURE_COLS

            period = request.args.get('period', '3y')
            threshold = request.args.get('threshold', type=float)
            if threshold is None:
                threshold = advanced_predictor.confidence_threshold or 0.5

            collector = DataCollector()
            processor = TechnicalProcessor()
            df = _download_with_context(collector, processor, ticker.upper(), period)
            if df is None:
                return jsonify({'error': f'sin datos para {ticker}'}), 404
            df['target_direccion'] = (df['Close'].shift(-1) > df['Close']).astype(int)

            from lightgbm import LGBMClassifier

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
    # resumen OOF global: heatmap de precision por banda de probabilidad
    # ------------------------------------------------------------------

    @app.route('/api/oof-summary')
    def api_oof_summary():
        oof = _load_oof()
        if oof is None:
            return jsonify({'available': False,
                            'reason': 'reentrena el modelo para generar OOF con fechas'})
        probs = np.asarray(oof['probabilities'])
        targets = np.asarray(oof['targets'])
        edges = [0.0, 0.35, 0.45, 0.55, 0.65, 1.0]
        labels = ['<=35%', '35-45%', '45-55%', '55-65%', '>=65%']
        bins = pd.cut(probs, bins=edges, labels=labels, include_lowest=True)
        summary = []
        for label in labels:
            mask = bins == label
            n = int(mask.sum())
            if n:
                # precision del lado anunciado en esa banda:
                # bandas de venta (prob baja) aciertan cuando target==0
                band = targets[mask]
                if label in ('<=35%', '35-45%'):
                    acc = float((band == 0).mean())
                else:
                    acc = float(band.mean())
                summary.append({
                    'band': label, 'n': n, 'accuracy': round(acc, 4),
                    'share': round(n / len(probs), 4),
                })
        overall_acc = float(((probs >= 0.5).astype(int) == targets).mean())
        return jsonify(_clean_nan({
            'available': True,
            'n_samples': int(len(probs)),
            'overall_accuracy': round(overall_acc, 4),
            'bands': summary,
            'range': [oof['dates'][0], oof['dates'][-1]],
        }))

    # ------------------------------------------------------------------
    # endpoints originales (compatibilidad)
    # ------------------------------------------------------------------

    @app.route('/api/predict/<ticker>', methods=['GET'])
    def predict_ticker(ticker):
        active_predictor = advanced_predictor if advanced_predictor else predictor
        if active_predictor is None:
            return jsonify({
                'error': 'modelo no disponible',
                'message': 'entrena el modelo primero con: python scripts/train_advanced.py'
            }), 400
        try:
            resultado = active_predictor.predict_ticker(ticker.upper())
            if resultado is None or 'error' in resultado:
                return jsonify({
                    'error': resultado.get('error', 'no se pudo predecir'),
                    'ticker': ticker
                }), 404
            resultado['model_type'] = 'advanced' if advanced_predictor else 'basic'
            resultado['timestamp'] = datetime.now().isoformat()
            if 'recommendation' not in resultado and 'prediccion' in resultado:
                resultado['recommendation'] = resultado['prediccion'].upper()
            return jsonify(_clean_nan(resultado))
        except Exception as e:
            logger.error(f"error en predict: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/predict/batch', methods=['POST'])
    def predict_batch():
        active_predictor = advanced_predictor if advanced_predictor else predictor
        if active_predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        data = request.get_json()
        tickers = data.get('tickers', [])
        if not tickers:
            return jsonify({'error': 'no se proporcionaron tickers'}), 400
        resultados = active_predictor.predict_multiple([t.upper() for t in tickers])
        return jsonify(_clean_nan({
            'resultados': resultados.to_dict('records') if hasattr(resultados, 'to_dict') else resultados,
            'model_type': 'advanced' if advanced_predictor else 'basic',
            'timestamp': datetime.now().isoformat()
        }))

    @app.route('/api/recommendation/<ticker>', methods=['GET'])
    def get_recommendation(ticker):
        if predictor is None:
            return jsonify({'error': 'modelo no disponible'}), 400
        recomendacion = predictor.get_recommendation(ticker.upper())
        if recomendacion is None:
            return jsonify({'error': 'no se pudo obtener recomendacion'}), 404
        return jsonify({
            'ticker': ticker.upper(),
            'recomendacion': recomendacion,
            'timestamp': datetime.now().isoformat()
        })

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'endpoint no encontrado'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"error interno: {error}")
        return jsonify({'error': 'error interno del servidor'}), 500
