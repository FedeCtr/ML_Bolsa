"""AI Insights: explicabilidad de senales via TreeSHAP (Sprint 5).

Extrae contribuciones SHAP por feature usando el soporte nativo de los
boosters del ensemble (``pred_contrib`` de XGBoost y LightGBM, ambos
implementan TreeSHAP en C++ sin dependencias extra) y los convierte en
lenguaje natural: que apoya la senal y que la presiona.

Las contribuciones estan en espacio de log-odds de P(alza): positivas empujan
hacia COMPRA, negativas hacia VENTA. Se promedian los dos boosters y se
ignoran las contribuciones de features ausentes (valor base del booster).

Uso:
    from src.ml.insights import explain_signal
    insights = explain_signal(predictor, df)   # df ya con features
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# etiquetas legibles de features clave (el resto se muestra tal cual)
FEATURE_LABELS = {
    "rsi": "RSI (14)",
    "macd_norm": "MACD normalizado",
    "macd_diff_norm": "Histograma MACD",
    "bb_pct": "%B de Bollinger",
    "bb_width": "Ancho de Bollinger",
    "stoch_k": "Estocástico %K",
    "williams_r": "Williams %R",
    "cci": "CCI",
    "dist_sma_10": "Distancia a SMA10",
    "dist_sma_20": "Distancia a SMA20",
    "dist_sma_50": "Distancia a SMA50",
    "dist_sma_200": "Distancia a SMA200",
    "volatilidad_20": "Volatilidad 20d",
    "volatilidad_60d": "Volatilidad 60d",
    "volumen_ratio": "Volumen relativo",
    "retorno_5d": "Retorno 5d",
    "retorno_20d": "Retorno 20d",
    "momentum_20": "Momentum 20d",
    "atr_pct": "ATR %",
    "adx": "ADX",
}


def _shap_contributions(booster: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """contribuciones TreeSHAP de un booster para la ultima fila de X."""
    try:
        contribs = booster.predict(X, pred_contrib=True)
        arr = np.asarray(contribs, dtype=float)
        if arr.ndim == 2:
            return arr[-1]          # fila unica: (n_features+1,)
        return arr.reshape(-1)
    except Exception:
        return None


def compute_shap(predictor, X: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """contribuciones SHAP promediadas de los boosters del ensemble.

    Returns:
        {'features': [nombres], 'shap': [valores], 'logodds': float} o None.
    """
    ens = getattr(predictor, "ensemble", None)
    if ens is None:
        return None
    member_map = {n: est for n, est in zip(ens.named_estimators_.keys(), ens.estimators_)}
    vectors, names = [], None
    for member in ("xgboost", "lightgbm"):
        est = member_map.get(member)
        if est is None:
            continue
        booster = getattr(est, "booster_", None) or est
        vec = _shap_contributions(booster, X)
        if vec is None:
            continue
        vectors.append(vec)

    if not vectors:
        return None
    # los dos boosters comparten orden de columnas (mismo X) pero pueden
    # diferir en longitud (bias column); alineamos por el minimo comun
    n = min(len(v) for v in vectors)
    stacked = np.vstack([v[:n] for v in vectors])
    mean = stacked.mean(axis=0)
    names = list(predictor.feature_cols)[: n - 1]
    if len(names) != n - 1:  # desalineacion defensiva
        return None
    return {
        "features": names,
        "shap": mean[:-1].tolist(),          # ultima posicion = valor base
        "logodds": float(mean[-1] + mean[:-1].sum()),
        "base": float(mean[-1]),
    }


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _fmt(v: float) -> str:
    return f"{v:+.3f}"


def explain_signal(predictor, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """explica la senal del dia en lenguaje natural (top drivers).

    Returns:
        {'top_factors': [...], 'buy_pressure': float, 'sell_pressure': float,
         'summary': str} o None si no hay soporte SHAP.
    """
    X = df[predictor.feature_cols].tail(1)
    if X.isna().any(axis=1).iloc[0]:
        return None
    shap = compute_shap(predictor, X)
    if shap is None:
        return None

    pairs = sorted(zip(shap["features"], shap["shap"]),
                   key=lambda t: abs(t[1]), reverse=True)
    top = pairs[:5]
    factors: List[Dict[str, Any]] = []
    for feat, val in top:
        if abs(val) < 1e-4:
            continue
        factors.append({
            "feature": feat,
            "label": _label(feat),
            "value": _fmt(float(val)),
            "direction": "buy" if val > 0 else "sell",
            "raw": X.iloc[0][feat] if feat in X.columns else None,
        })

    pos = sum(v for _, v in pairs if v > 0)
    neg = sum(v for _, v in pairs if v < 0)
    prob = 1 / (1 + np.exp(-shap["logodds"]))
    summary = (
        f"Factores alcistas suman {_fmt(pos)} en log-odds y los bajistas {_fmt(neg)}; "
        f"el balance neto ({_fmt(pos + neg)}) equivale a una P(alza) ~{prob*100:.0f}% "
        f"del ensemble (boosters). "
        + ("El impulso domina: momentum y medias empujan a favor."
           if pos > abs(neg) else
           "La senal combina impulso favorable con presion de sobrecompra/volatilidad.")
    )
    return {
        "top_factors": factors,
        "buy_pressure": round(float(pos), 4),
        "sell_pressure": round(float(neg), 4),
        "logodds": round(shap["logodds"], 4),
        "summary": summary,
    }
