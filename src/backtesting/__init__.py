"""
Paquete de backtesting: motor walk-forward con costos de transaccion,
metricas financieras y generacion de informes.

Filosofia: la senal se decide con la informacion del cierre del dia t y se
EJECUTA en la apertura del dia t+1. Nunca se opera con el precio del mismo
dia en que la senal se genero (lookahead bias).
"""
from .engine import BacktestEngine, BacktestResult
from .metrics import compute_metrics, trade_metrics
from .report import generate_report

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'compute_metrics',
    'trade_metrics',
    'generate_report',
]
