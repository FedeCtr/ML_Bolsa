"""
Generacion del informe tecnico de backtesting (markdown).

Entregable del encargo: informe de backtesting con metricas obligatorias,
desglose por regimen de volatilidad y parametros de costos usados.
"""
from typing import Optional

import pandas as pd

from ..utils.logger import get_logger
from .engine import BacktestResult

logger = get_logger(__name__)


def _pct(x: float) -> str:
    return f"{x:+.2%}"


def _ratio(x: float) -> str:
    if x == float('inf'):
        return "inf"
    return f"{x:.2f}"


def generate_report(
    result: BacktestResult,
    output_path: Optional[str] = None,
    title: str = "Informe técnico de Backtesting",
) -> str:
    """
    genera el informe markdown del backtest.

    Args:
        result: resultado del BacktestEngine.run().
        output_path: si se da, escribe el informe a ese archivo.
        title: titulo del informe.

    Returns:
        el markdown como string.
    """
    m = result.metrics
    t = result.trade_stats
    cfg = result.config

    lines = [
        f"# {title}",
        "",
        f"**Ticker:** {cfg.get('ticker', 'N/A')}  ",
        f"**Período:** {cfg.get('start', '?')} → {cfg.get('end', '?')} ({cfg.get('n_days', '?')} días)  ",
        f"**Capital inicial:** ${cfg.get('initial_capital', 0):,.0f}",
        "",
        "## 1. Parámetros de costos",
        "",
        "| Concepto | Valor |",
        "|---|---|",
        f"| Comisión | {cfg.get('commission_pct', 0)*100:.3f}% por lado |",
        f"| Spread | {cfg.get('spread_pct', 0)*100:.3f}% por lado |",
        f"| Slippage | {cfg.get('slippage_pct', 0)*100:.3f}% por lado |",
        f"| **Costo round-trip total** | **{cfg.get('round_trip_cost_pct', 0)*100:.3f}%** |",
        f"| Permite cortos | {'sí' if cfg.get('allow_short') else 'no'} |",
        "",
        "> Convención anti-lookahead: la señal se genera con el cierre del día t",
        "> y se ejecuta en la apertura del día t+1.",
        "",
        "## 2. Métricas de cartera (netas de costos)",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Retorno total | {_pct(m['total_return'])} |",
        f"| Retorno anualizado | {_pct(m['annual_return'])} |",
        f"| Volatilidad anualizada | {m['annual_vol']:.2%} |",
        f"| **Sharpe Ratio** | **{_ratio(m['sharpe_ratio'])}** |",
        f"| **Sortino Ratio** | **{_ratio(m['sortino_ratio'])}** |",
        f"| **Maximum Drawdown** | **{m['max_drawdown']:.2%}** |",
        f"| Calmar Ratio | {_ratio(m['calmar_ratio'])} |",
        f"| Días operados | {m['n_days']} |",
        "",
        "## 3. Métricas por operación",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Nº de trades | {t['n_trades']} |",
        f"| Win Rate | {t['win_rate']:.2%} |",
        f"| **Profit Factor** | **{_ratio(t['profit_factor'])}** |",
        f"| Win/Loss Ratio | {_ratio(t['win_loss_ratio'])} |",
        f"| Ganancia media | {_pct(t['avg_win'])} |",
        f"| Pérdida media | {_pct(t['avg_loss'])} |",
        f"| Expectancy por trade | {_pct(t['expectancy'])} |",
        f"| Máx. pérdidas consecutivas | {t['max_consecutive_losses']} |",
        "",
        "## 4. Desglose por régimen de volatilidad",
        "",
        "| Régimen | Días | Retorno | Sharpe | Max DD |",
        "|---|---|---|---|---|",
    ]

    for regime, rm in result.regime_metrics.items():
        lines.append(
            f"| {regime} | {rm['n_days']} | {_pct(rm['total_return'])} | "
            f"{_ratio(rm['sharpe_ratio'])} | {rm['max_drawdown']:.2%} |"
        )

    lines += [
        "",
        "## 5. Notas y limitaciones",
        "",
        "- Resultados **netos de comisiones, spread y slippage**; no incluyen",
        "  impacto de mercado para órdenes grandes ni costes de financiación de",
        "  cortos.",
        "- Ejecución simulada a apertura; en producción real la latencia del",
        "  feed y del broker puede degradar los resultados.",
        "- Un mismo modelo puede render distinto por régimen: revisar la",
        "  sección 4 antes de decidir el despliegue.",
        "- Este informe es un ejercicio metodológico y **no constituye",
        "  asesoramiento financiero**.",
        "",
    ]

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"informe escrito: {output_path}")

    return report
