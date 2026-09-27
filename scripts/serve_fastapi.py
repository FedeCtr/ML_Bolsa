"""
Servidor FastAPI de inferencia de baja latencia (spec 5: backend de produccion).

El modelo ensemble se carga UNA vez al arrancar y las features se calculan
en memoria; la ruta /infer sirve predicciones con validacion pydantic y
documentacion OpenAPI automatica (/docs).

Uso:
    uvicorn scripts.serve_fastapi:app --host 0.0.0.0 --port 8010
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.collector import DataCollector
from src.data.processor import TechnicalProcessor
from src.data.universe import get_name_map, get_sector_map
from src.ml.advanced_predictor import AdvancedPredictor
from src.ml.regime import detect_regime
from src.ml.signal_engine import build_signal
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="ML_Bolsa Inference API",
    description="Senales operables del ensemble calibrado (inferencia de baja latencia).",
    version="2.0.0",
)

_state = {"predictor": None}


@app.on_event("startup")
def _load_model():
    _state["predictor"] = AdvancedPredictor()
    logger.info("FastAPI: modelo cargado y en memoria")


class SignalOut(BaseModel):
    ticker: str
    signal: str
    side: str
    confidence_pct: float
    conviction_pct: float
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None
    trading_allowed: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["predictor"] is not None}


@app.get("/infer/{ticker}", response_model=SignalOut,
         responses={404: {"description": "sin datos para el ticker"}})
def infer(ticker: str, period: str = "1y", with_levels: bool = True):
    """
    senal operable completa para un ticker.

    - probability calibrada -> COMPRA(+FUERTE) / VENTA(+FUERTE), sin neutrales
    - niveles entry/SL/TP1/TP2 por pivotes + ATR
    - regimen de mercado: caution escala sizing 0.5x; suspended bloquea
    """
    p = _state["predictor"]
    if p is None:
        raise HTTPException(503, "modelo no cargado")
    ticker = ticker.upper()

    collector, processor = DataCollector(), TechnicalProcessor()
    df = collector.download_ticker(ticker, period)
    if df is None or df.empty:
        raise HTTPException(404, f"sin datos para {ticker}")
    spy = collector.download_ticker("SPY", period)
    vix = collector.download_ticker("^VIX", period)
    df = processor.process_all_indicators(df, spy_df=spy, vix_df=vix)

    X = df[p.feature_cols].tail(1)
    if X.isna().any(axis=1).iloc[0]:
        raise HTTPException(422, "datos insuficientes para inferencia")

    prob_up = p._raw_probability_up(X)
    regime = detect_regime(df, vix_series=vix["Close"] if vix is not None else None)
    sig = build_signal(ticker, df, prob_up, as_of=str(df.index[-1].date()), regime=regime)
    d = sig.to_dict()
    d["sector"] = get_sector_map().get(ticker, "Otros")
    d["name"] = get_name_map().get(ticker, ticker)
    if not with_levels:
        for k in ("take_profits", "supports", "resistances", "notes"):
            d.pop(k, None)
    return d


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
