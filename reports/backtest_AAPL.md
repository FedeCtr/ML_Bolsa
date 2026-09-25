# Backtesting walk-forward: AAPL

**Ticker:** AAPL  
**Período:** 2023-09-25 → 2026-09-24 (753 días)  
**Capital inicial:** $100,000

## 1. Parámetros de costos

| Concepto | Valor |
|---|---|
| Comisión | 0.050% por lado |
| Spread | 0.050% por lado |
| Slippage | 0.050% por lado |
| **Costo round-trip total** | **0.300%** |
| Permite cortos | no |

> Convención anti-lookahead: la señal se genera con el cierre del día t
> y se ejecuta en la apertura del día t+1.

## 2. Métricas de cartera (netas de costos)

| Métrica | Valor |
|---|---|
| Retorno total | -17.06% |
| Retorno anualizado | -6.07% |
| Volatilidad anualizada | 20.84% |
| **Sharpe Ratio** | **-0.20** |
| **Sortino Ratio** | **-0.01** |
| **Maximum Drawdown** | **-44.72%** |
| Calmar Ratio | -0.14 |
| Días operados | 753 |

## 3. Métricas por operación

| Métrica | Valor |
|---|---|
| Nº de trades | 80 |
| Win Rate | 51.25% |
| **Profit Factor** | **1.08** |
| Win/Loss Ratio | 1.03 |
| Ganancia media | +2.35% |
| Pérdida media | -2.29% |
| Expectancy por trade | +0.09% |
| Máx. pérdidas consecutivas | 5 |

## 4. Desglose por régimen de volatilidad

| Régimen | Días | Retorno | Sharpe | Max DD |
|---|---|---|---|---|
| bajo | 245 | -14.89% | -2.07 | -19.75% |
| medio | 244 | -7.31% | -0.44 | -18.06% |
| alto | 244 | +5.14% | 0.32 | -28.24% |

## 5. Notas y limitaciones

- Resultados **netos de comisiones, spread y slippage**; no incluyen
  impacto de mercado para órdenes grandes ni costes de financiación de
  cortos.
- Ejecución simulada a apertura; en producción real la latencia del
  feed y del broker puede degradar los resultados.
- Un mismo modelo puede render distinto por régimen: revisar la
  sección 4 antes de decidir el despliegue.
- Este informe es un ejercicio metodológico y **no constituye
  asesoramiento financiero**.
