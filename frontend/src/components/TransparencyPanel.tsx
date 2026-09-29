const METRICS = [
  { label: "Profit Factor", value: "1.505", sub: "neto, 10 pb costos" },
  { label: "Win Rate", value: "46.7%", sub: "246 operaciones" },
  { label: "E[R]", value: "+0.268R", sub: "por operación" },
  { label: "Sharpe", value: "1.48", sub: "walk-forward purgado" },
  { label: "Max Drawdown", value: "-15.9%", sub: "largo-solo" },
  { label: "R/R estructural", value: "1:2", sub: "TP2 = máx(2·ATR, 2R)" },
];

export default function TransparencyPanel() {
  return (
    <section className="glass rounded-xl p-5">
      <div className="flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-gold-400">
          Transparencia Extrema
        </h2>
        <span className="text-[10px] text-mist-400">
          OOF walk-forward 2021–2026 · 7 mega-caps tech
        </span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {METRICS.map((m) => (
          <div
            key={m.label}
            className="rounded-lg border border-mist-400/10 bg-ink-850/70 p-3"
          >
            <div className="text-[10px] uppercase tracking-wider text-mist-400">
              {m.label}
            </div>
            <div className="mt-1 text-xl font-bold tabular-nums text-mist-100">
              {m.value}
            </div>
            <div className="mt-0.5 text-[10px] text-mist-400">{m.sub}</div>
          </div>
        ))}
      </div>

      <p className="mt-4 text-xs leading-relaxed text-mist-400">
        Probabilidades out-of-sample del walk-forward purgado; entrada en apertura
        de t+1, SL 1.5·ATR, TP 2× riesgo, holding máx. 20 sesiones, costos 10 pb,
        criterio SL-first. Filtros: largo-solo, VIX ≤ 30, 1 posición por ticker,
        máx. 3 concurrentes.{" "}
        <span className="text-mist-200">
          La confianza del modelo NO es la precisión.
        </span>
      </p>
    </section>
  );
}
