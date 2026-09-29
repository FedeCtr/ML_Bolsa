"""Seed de senales demo en signal_cache (para demos y revision visual).

Escribe 7 senales realistas del dia en la DB global (data/saas.db). El
scheduler las sobrescribira con datos reales en su primer ciclo.

Uso:
    python scripts/seed_demo_signals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.signal_store import SignalStore  # noqa: E402

DEMOS = [
    ("AAPL", "COMPRA_FUERTE", "largo", 232.4, 221.8, [243.0, 253.6], 62, 78),
    ("MSFT", "COMPRA", "largo", 428.7, 411.5, [446.0, 463.3], 55, 64),
    ("NVDA", "COMPRA_FUERTE", "largo", 135.5, 124.4, [146.6, 157.7], 66, 81),
    ("AMZN", "COMPRA", "largo", 221.1, 209.9, [232.3, 243.5], 54, 61),
    ("META", "COMPRA", "largo", 563.2, 535.0, [591.4, 619.6], 57, 68),
    ("GOOGL", "COMPRA", "largo", 171.8, 163.0, [180.6, 189.4], 53, 58),
    ("TSLA", "VENTA", "corto", 252.1, 269.7, [234.5, 217.0], -52, 55),
]


def _demo(t) -> dict:
    ticker, signal, side, entry, sl, tps, conf, conv = t
    rr = round(abs(tps[0] - entry) / abs(entry - sl), 2)
    return {
        "ticker": ticker, "as_of": None,  # se rellena con la fecha de hoy
        "signal": signal, "side": side,
        "prob_up": round(0.5 + conf / 200, 3),
        "confidence_pct": float(conf), "conviction_pct": float(conv),
        "entry": entry, "stop_loss": sl,
        "take_profits": [{"price": tps[0], "r_multiple": 1.0},
                         {"price": tps[1], "r_multiple": 2.0}],
        "risk_reward": rr, "position_size_pct": 6.1,
        "regime": "normal", "trading_allowed": True,
        "price": entry, "atr_pct": 2.3, "volume_ratio": 1.4,
        "rsi": 58.0, "name": ticker,
    }


if __name__ == "__main__":
    import datetime

    today = datetime.date.today().isoformat()
    rows = []
    for t in DEMOS:
        d = _demo(t)
        d["as_of"] = today
        rows.append(d)
    n = SignalStore().upsert_batch(rows, universe="ndx100")
    print(f"seed: {n} senales demo escritas para {today}")
