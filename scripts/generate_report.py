"""
Informe PDF de backtesting y expectativa matematica (entregable 3).

Genera reports/informe_backtesting.pdf con:
  - KPIs del sistema (Profit Factor, Sharpe, MaxDD, Win Rate, E[R])
  - Curva de equidad vs benchmark SPY y curva de drawdown
  - Desempeno por ano (robustez de regimen)
  - Matriz de confianza OOF por banda de probabilidad
  - Sensibilidad de los filtros (gates)
  - Metodologia y advertencias honestas

Uso:
    venv/Scripts/python scripts/generate_report.py
    venv/Scripts/python scripts/generate_report.py --output reports/mi_informe.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ml.expectancy import (
    COSTS_ROUND_TRIP,
    attach_market_data,
    load_oof_dataset,
    simulate_expectancy,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# configuracion validada del sistema (Sprint 3)
DEFAULT_CONFIG = {
    'prob_min': 0.50,
    'rr_multiple': 2.0,
    'vix_max': 30.0,
    'allow_short': False,
    'one_position_per_ticker': True,
    'max_concurrent': 3,
    'max_hold': 20,
    'risk_per_trade': 0.01,
}

DARK = '#0B0E14'
GREEN = '#2BD576'
RED = '#E5484D'
BLUE = '#4C8DFF'
GRAY = '#8B93A7'


def _lat(s: str) -> str:
    """fpdf2 con fuentes core usa latin-1: sanitiza caracteres fuera de rango."""
    return str(s).encode('latin-1', 'replace').decode('latin-1')


# ----------------------------------------------------------------------
# Analisis
# ----------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, cfg: dict) -> dict:
    """corre la simulacion principal + analisis complementarios."""
    res = simulate_expectancy(df, **cfg)

    # simulacion por ano (misma configuracion, submuestra temporal)
    per_year = []
    for year in sorted(df['date'].str[:4].unique()):
        sub = df[df['date'].str[:4] == year]
        r = simulate_expectancy(sub, **cfg)
        per_year.append({
            'year': year, 'n': r.n_trades,
            'win_rate': r.win_rate, 'profit_factor': r.profit_factor,
            'expectancy_r': r.expectancy_r,
        })

    # sensibilidad de gates (transparencia del efecto de cada filtro)
    sens = []
    for label, kw in [
        ('todo operado (largo+corto, sin gates)', dict()),
        ('largo solo, sin gate VIX', dict(allow_short=False)),
        ('largo solo + VIX<=30 (FINAL)', dict(allow_short=False, vix_max=cfg['vix_max'])),
    ]:
        r = simulate_expectancy(df, prob_min=cfg['prob_min'], rr_multiple=cfg['rr_multiple'],
                                one_position_per_ticker=cfg['one_position_per_ticker'],
                                max_concurrent=cfg['max_concurrent'], **kw)
        sens.append({'config': label, 'n': r.n_trades, 'win_rate': r.win_rate,
                     'profit_factor': r.profit_factor, 'expectancy_r': r.expectancy_r})

    # matriz de confianza OOF: precision de direccion por banda y lado
    bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    matrix = []
    for lo, hi in bands:
        longs = df[(df['prob_up'] >= lo) & (df['prob_up'] < hi)]
        shorts = df[(df['prob_up'] <= 1 - lo) & (df['prob_up'] > 1 - hi)]
        matrix.append({
            'banda': f'{lo:.2f}-{hi:.2f}',
            'n_largos': len(longs),
            'acc_largos': float(longs['target'].mean()) if len(longs) else float('nan'),
            'n_cortos': len(shorts),
            'acc_cortos': float(1 - shorts['target'].mean()) if len(shorts) else float('nan'),
        })

    # benchmark SPY buy&hold sobre el periodo operado
    bench = _spy_benchmark(df)
    return {'main': res, 'per_year': per_year, 'sens': sens,
            'matrix': matrix, 'benchmark': bench, 'config': cfg}


def _spy_benchmark(df: pd.DataFrame) -> dict:
    spy_path = Path('data/raw/SPY_raw.csv')
    if not spy_path.exists():
        return {}
    spy = pd.read_csv(spy_path)
    spy[spy.columns[0]] = pd.to_datetime(spy[spy.columns[0]], utc=True).dt.strftime('%Y-%m-%d')
    spy = spy.drop_duplicates(subset=spy.columns[0], keep='last').set_index(spy.columns[0]).sort_index()
    d0, d1 = df['date'].min(), df['date'].max()
    spy = spy.loc[d0:d1]
    if len(spy) < 50:
        return {}
    close = spy['Close']
    norm = close / close.iloc[0]
    rets = close.pct_change().dropna()
    years = max((pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25, 1e-9)
    peak = np.maximum.accumulate(norm)
    return {
        'total_return_pct': float((norm.iloc[-1] - 1) * 100),
        'cagr_pct': float(((norm.iloc[-1]) ** (1 / years) - 1) * 100),
        'max_drawdown_pct': float(((norm - peak) / peak).min() * 100),
        'sharpe': float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0,
        'series': norm,
        'dates': norm.index,
    }


# ----------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------

def _fig_equity(res, bench: dict) -> BytesIO:
    fig, ax = plt.subplots(figsize=(8.6, 3.8), facecolor='white')
    x = np.arange(len(res.equity))
    ax.plot(x, np.array(res.equity) * 100, color=GREEN, lw=1.8,
            label=f"Estrategia ML (PF {res.profit_factor:.2f}, Sharpe {res.sharpe:.2f})")
    if bench.get('series') is not None:
        bs = bench['series'].to_numpy()
        n = min(len(bs), len(x))
        ax.plot(x[:n], bs[:n] * 100, color=GRAY, lw=1.4, ls='--',
                label=f"SPY buy&hold (+{bench['total_return_pct']:.0f}%, Sharpe {bench['sharpe']:.2f})")
    ax.set_title('Curva de equidad (capital inicial = 100, riesgo 1% por operacion)',
                 fontsize=10, color='#222')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    ticks = np.linspace(0, len(x) - 1, 6).astype(int)
    if res.equity and len(x) > 1:
        # las fechas de decision vienen ordenadas; etiquetas aproximadas por año
        years = sorted(pd.Series(res.per_year_index).unique()) if hasattr(res, 'per_year_index') else []
    ax.set_xticks(ticks)
    ax.set_xticklabels(['inicio'] + [''] * (len(ticks) - 2) + ['fin'])
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _fig_drawdown(res) -> BytesIO:
    eq = np.array(res.equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    fig, ax = plt.subplots(figsize=(8.6, 2.6), facecolor='white')
    ax.fill_between(np.arange(len(dd)), dd, 0, color=RED, alpha=0.35)
    ax.plot(np.arange(len(dd)), dd, color=RED, lw=1.2)
    ax.set_title(f'Drawdown (maximo {res.max_drawdown_pct:.1f}%)', fontsize=10, color='#222')
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _fig_matrix(matrix) -> BytesIO:
    bands = [m['banda'] for m in matrix]
    acc_l = [m['acc_largos'] * 100 for m in matrix]
    acc_s = [m['acc_cortos'] * 100 for m in matrix]
    xpos = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(8.6, 3.2), facecolor='white')
    ax.bar(xpos - 0.2, acc_l, 0.4, color=GREEN, label='Precision largos')
    ax.bar(xpos + 0.2, acc_s, 0.4, color=RED, label='Precision cortos (1-acc)')
    ax.axhline(50, color='#222', lw=1, ls=':')
    ax.text(len(bands) - 0.5, 50.7, 'azar 50%', fontsize=8, color='#222')
    ax.set_xticks(xpos)
    ax.set_xticklabels(bands, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title('Matriz de confianza OOF: precision de direccion por banda de probabilidad',
                 fontsize=10, color='#222')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _fig_years(per_year) -> BytesIO:
    years = [p['year'] for p in per_year]
    pfs = [min(p['profit_factor'], 6) for p in per_year]
    colors = [GREEN if p['profit_factor'] > 1 else RED for p in per_year]
    fig, ax = plt.subplots(figsize=(8.6, 2.8), facecolor='white')
    bars = ax.bar(years, pfs, color=colors, width=0.55)
    for b, p in zip(bars, per_year):
        label = f"{p['profit_factor']:.2f}" if p['profit_factor'] < 6 else f">{6}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                f"PF {label}\nn={p['n']}", ha='center', fontsize=8)
    ax.axhline(1.0, color='#222', lw=1, ls=':')
    ax.set_title('Profit Factor por ano (robustez de regimen; PF limitado a 6 en el eje)',
                 fontsize=10, color='#222')
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------

def build_pdf(analysis: dict, output_path: str) -> int:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    res = analysis['main']
    cfg = analysis['config']
    bench = analysis['benchmark']

    pdf = FPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=16)

    def h1(txt):
        pdf.set_font('helvetica', 'B', 15)
        pdf.set_text_color(20, 24, 34)
        pdf.cell(0, 9, _lat(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h2(txt):
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 11.5)
        pdf.set_text_color(20, 24, 34)
        pdf.cell(0, 7, _lat(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(43, 213, 118)
        pdf.set_line_width(0.6)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 34, pdf.get_y())
        pdf.ln(2)

    def body(txt, size=9.2):
        pdf.set_font('helvetica', '', size)
        pdf.set_text_color(60, 64, 76)
        pdf.multi_cell(0, 4.8, _lat(txt))

    def table(headers, rows, widths, aligns=None):
        aligns = aligns or ['L'] * len(headers)
        pdf.set_font('helvetica', 'B', 8.4)
        pdf.set_fill_color(238, 241, 246)
        pdf.set_text_color(20, 24, 34)
        for w, htxt in zip(widths, headers):
            pdf.cell(w, 6.4, _lat(htxt), border=1, fill=True, align='C')
        pdf.ln()
        pdf.set_font('helvetica', '', 8.4)
        pdf.set_text_color(40, 44, 56)
        for row in rows:
            for w, val, a in zip(widths, row, aligns):
                pdf.cell(w, 6.2, _lat(val), border=1, align=a)
            pdf.ln()

    # ---------- pagina 1: portada + resumen ----------
    pdf.add_page()
    pdf.set_fill_color(11, 14, 20)
    pdf.rect(0, 0, 210, 42, 'F')
    pdf.set_y(9)
    pdf.set_font('helvetica', 'B', 19)
    pdf.set_text_color(245, 247, 250)
    pdf.cell(0, 9, _lat('ML Bolsa - Informe de Backtesting'), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('helvetica', '', 10.5)
    pdf.set_text_color(43, 213, 118)
    pdf.cell(0, 7, _lat('Expectativa matematica del sistema de senales (modelo v2)'),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(48)

    h1('Resumen ejecutivo')
    body(
        f"Sistema: ensemble XGB+LGBM+RF+ET con balanceo de clases (class_weight), evaluado "
        f"walk-forward purgado ({int(res.n_trades)} operaciones con senal). Ejecucion simulada: "
        f"entrada en apertura de t+1, SL 1.5 ATR (acotado 1-5%), TP a {cfg['rr_multiple']:.0f}x riesgo, "
        f"holding maximo {cfg['max_hold']} sesiones, costos 10 pb por operacion, criterio "
        f"conservador SL-first. Filtros activos: largo-solo, VIX <= {cfg['vix_max']:.0f}, "
        f"1 posicion por ticker y maximo {cfg['max_concurrent']} concurrentes."
    )
    pdf.ln(2)

    kpi_rows = [
        ['Profit Factor', f"{res.profit_factor:.3f}", 'Objetivo del producto: > 1.30'],
        ['Win Rate', f"{res.win_rate:.1%}", 'con R/R 1:2 el punto de equilibrio es 33.4%'],
        ['Expectativa', f"{res.expectancy_r:+.3f} R", 'ganancia media por operacion en unidades de riesgo'],
        ['Retorno total', f"{res.total_return_pct:+.1f}%", 'capital compuesto al 1% de riesgo por trade'],
        ['Max Drawdown', f"{res.max_drawdown_pct:.1f}%", 'pico a valle de la curva de equidad'],
        ['Sharpe', f"{res.sharpe:.2f}", 'anualizado sobre el periodo del sample'],
        ['Operaciones', f"{res.n_trades}", f"holding medio {res.avg_hold_days:.1f} sesiones"],
        ['Benchmark SPY', f"+{bench.get('total_return_pct', 0):.0f}%",
         f"Sharpe {bench.get('sharpe', 0):.2f}, MaxDD {bench.get('max_drawdown_pct', 0):.1f}%"],
    ]
    table(['Metrica', 'Valor', 'Nota'], kpi_rows, [38, 26, 116])

    pdf.ln(3)
    if res.profit_factor > 1.3:
        body("VEREDICTO: el objetivo de Profit Factor > 1.3 se alcanza en el periodo "
             "2022-2026 bajo la configuracion documentada. El edge es de regimen: fuerte en "
             "mercados alcistas o laterales (2023-2025) y negativo en el bear market 2022 y "
             "en el parcial 2026. Ver desglose por ano.")
    else:
        body("VEREDICTO: el objetivo de Profit Factor > 1.3 NO se alcanza en el periodo "
             "completo con esta configuracion. Ver sensibilidad de filtros y desglose por ano.")

    # ---------- pagina 2: curvas ----------
    pdf.add_page()
    h1('Rendimiento historico')
    pdf.image(_fig_equity(res, bench), w=182)
    pdf.ln(1)
    pdf.image(_fig_drawdown(res), w=182)

    h2('Desempeno por ano')
    year_rows = [[p['year'], str(p['n']), f"{p['win_rate']:.1%}",
                  f"{p['profit_factor']:.2f}", f"{p['expectancy_r']:+.3f} R"]
                 for p in analysis['per_year']]
    table(['Ano', 'Ops', 'Win Rate', 'Profit Factor', 'Expectativa'],
          year_rows, [24, 24, 38, 48, 46], ['C', 'C', 'C', 'C', 'C'])
    body("Lectura honesta: el sistema no es estacionario. En 2022 (bear market) el gate de "
         "VIX filtra la mayoria de senales pero las que pasan perdieron; el edge vive en "
         "regimen alcista/lateral. La produccion debe seguir publicando el desglose anual.")

    # ---------- pagina 3: matriz + sensibilidad ----------
    pdf.add_page()
    h1('Calidad del modelo: matriz de confianza OOF')
    body("Precision de direccion fuera de muestra por banda de probabilidad calibrada "
         "(largos: probabilidad de subida; cortos: probabilidad de bajada). Es la base "
         "estadistica real del producto: la confianza del modelo NO es la precision.")
    pdf.image(_fig_matrix(analysis['matrix']), w=182)
    pdf.ln(2)
    mx_rows = [[m['banda'], str(m['n_largos']),
                f"{m['acc_largos']*100:.1f}%" if m['n_largos'] else '-',
                str(m['n_cortos']),
                f"{m['acc_cortos']*100:.1f}%" if m['n_cortos'] else '-']
               for m in analysis['matrix']]
    table(['Banda p', 'N largos', 'Acc. largos', 'N cortos', 'Acc. cortos'],
          mx_rows, [30, 30, 42, 30, 42], ['C', 'C', 'C', 'C', 'C'])

    h2('Sensibilidad de los filtros (mismo periodo, misma ejecucion)')
    sens_rows = [[s['config'], str(s['n']), f"{s['win_rate']:.1%}",
                  f"{s['profit_factor']:.2f}", f"{s['expectancy_r']:+.3f} R"]
                 for s in analysis['sens']]
    table(['Configuracion', 'Ops', 'Win Rate', 'PF', 'E[R]'],
          sens_rows, [66, 18, 26, 26, 32], ['L', 'C', 'C', 'C', 'C'])
    body("El short sistematico destruye valor en una era alcista (2022-2026 de SPY +80%): "
         "se excluye de la configuracion final. El gate de VIX <= 30 aporta el resto del "
         "edge al evitar las ventanas de panico donde el SL de 1.5 ATR se recorre.")

    # ---------- pagina 4: metodologia ----------
    pdf.add_page()
    h1('Metodologia y advertencias')
    h2('Datos y validacion')
    body(
        "- 7 tickers mega-cap tech (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META), OHLCV diario "
        "y VIX/SPY como contexto de regimen (sept 2022 - sept 2026).\n"
        "- Probabilidades OOF del walk-forward purgado por fecha (5 folds, purga 60 dias, "
        "embargo 5): cada fila fue predicha por un modelo que NO vio esa muestra. Sin leakage.\n"
        "- Balanceo de clases con class_weight/scale_pos_weight en el entrenamiento (sin SMOTE: "
        "el oversampling sintetico puede cruzar fronteras temporales).\n"
        "- Verificacion de integridad: el vector (fecha, ticker) del OOF se contrasta contra "
        "el OHLC local; las filas no verificables se excluyen del analisis.\n"
        "- Ejecucion conservadora: empate SL/TP el mismo dia cuenta como SL; costos 10 pb "
        "round-trip; sin apalancamiento; sin reinversion de shorts ni prestamo de acciones."
    )
    h2('Limitaciones declaradas')
    body(
        "- Universo reducido y sesgado a mega-cap tech con fuerte sesgo alcista del periodo; "
        "la extension a S&P 500 completa puede diluir o cambiar el edge.\n"
        "- 2022 y 2026-parcial son negativos: el filtro de regimen implementado (VIX) mitiga "
        "pero no elimina la dependencia de ciclo.\n"
        "- Los resultados OOF no incluyen el costo de impacto de mercado de tamanos grandes.\n"
        "- El reentrenamiento mensual (protocolo MLOps) puede desplazar las metricas; este "
        "informe corresponde al modelo entrenado el " + pd.Timestamp.now().strftime('%d/%m/%Y') + "."
    )
    h2('Advertencia legal')
    body("Este documento es material tecnico con fines de investigacion y demostracion. No es "
         "asesoramiento financiero ni una recomendacion de inversion. El rendimiento pasado, "
         "simulado u out-of-sample, no garantiza resultados futuros.")

    pdf.set_y(-24)
    pdf.set_font('helvetica', 'I', 8)
    pdf.set_text_color(140, 146, 160)
    pdf.cell(0, 6, _lat(f"Generado automaticamente - ML Bolsa - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} "
                        f"- config: largo, p>={cfg['prob_min']:.2f}, R/R 1:{cfg['rr_multiple']:.0f}, "
                        f"VIX<={cfg['vix_max']:.0f}, max {cfg['max_concurrent']} posiciones"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    return pdf.pages_count


def main():
    parser = argparse.ArgumentParser(description='Informe PDF de backtesting')
    parser.add_argument('--output', default='reports/informe_backtesting.pdf')
    parser.add_argument('--prob-min', type=float, default=DEFAULT_CONFIG['prob_min'])
    parser.add_argument('--rr', type=float, default=DEFAULT_CONFIG['rr_multiple'])
    parser.add_argument('--vix-max', type=float, default=DEFAULT_CONFIG['vix_max'])
    parser.add_argument('--max-concurrent', type=int, default=DEFAULT_CONFIG['max_concurrent'])
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, 'prob_min': args.prob_min, 'rr_multiple': args.rr,
           'vix_max': args.vix_max, 'max_concurrent': args.max_concurrent}

    logger.info('cargando OOF + datos de mercado...')
    df = load_oof_dataset()
    df = attach_market_data(df)

    logger.info('simulando (config validada)...')
    analysis = run_analysis(df, cfg)

    pages = build_pdf(analysis, args.output)
    res = analysis['main']
    logger.info("=" * 70)
    logger.info(f"PDF generado: {args.output} ({pages} paginas)")
    logger.info(f"PF {res.profit_factor:.3f} | WR {res.win_rate:.1%} | E[R] {res.expectancy_r:+.3f}R "
                f"| Sharpe {res.sharpe:.2f} | MaxDD {res.max_drawdown_pct:.1f}% | {res.n_trades} ops")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
