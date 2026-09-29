import type { Signal } from "@/lib/api";

const SIGNAL_STYLES: Record<string, string> = {
  COMPRA_FUERTE: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  COMPRA: "bg-emerald-500/10 text-emerald-400/90 border-emerald-500/20",
  VENTA_FUERTE: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  VENTA: "bg-rose-500/10 text-rose-400/90 border-rose-500/20",
};

function fmt(n?: number | null, digits = 2): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString("es-ES", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export default function SignalCard({ signal }: { signal: Signal }) {
  const tp = signal.take_profits ?? [];
  const style = SIGNAL_STYLES[signal.signal] ??
    "bg-ink-700/60 text-mist-200 border-mist-400/20";

  return (
    <div className="glass rounded-xl p-4 transition-transform duration-200 hover:-translate-y-0.5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold tracking-tight text-mist-100">
              {signal.ticker}
            </span>
            {signal.trading_allowed === false && (
              <span className="pill border border-rose-500/30 bg-rose-500/10 text-rose-300">
                suspendido
              </span>
            )}
          </div>
          <div className="mt-0.5 text-xs text-mist-400">
            {signal.name ?? ""} {signal.sector ? `· ${signal.sector}` : ""}
          </div>
        </div>
        <span className={`pill border ${style}`}>{signal.signal.replace("_", " ")}</span>
      </div>

      <div className="mt-3 grid grid-cols-4 gap-2 text-center">
        <Metric label="Entrada" value={fmt(signal.entry)} />
        <Metric label="Stop" value={fmt(signal.stop_loss)} accent="text-rose-300" />
        <Metric
          label="TP1"
          value={fmt(tp[0]?.price)}
          accent="text-emerald-300"
        />
        <Metric
          label="TP2"
          value={fmt(tp[1]?.price)}
          accent="text-emerald-300"
        />
      </div>

      <div className="mt-3 flex items-center justify-between text-xs text-mist-400">
        <span>
          R/R <span className="font-semibold text-gold-400">{fmt(signal.risk_reward, 1)}</span>
        </span>
        <span>
          convicción{" "}
          <span className="font-semibold text-mist-100">
            {fmt(signal.conviction_pct, 0)}%
          </span>
        </span>
        <span>
          RSI <span className="font-semibold text-mist-100">{fmt(signal.rsi, 0)}</span>
        </span>
        <span>
          régimen{" "}
          <span className="font-semibold text-mist-100">{signal.regime ?? "—"}</span>
        </span>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  accent = "text-mist-100",
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="rounded-lg border border-mist-400/10 bg-ink-850/60 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-mist-400">{label}</div>
      <div className={`text-sm font-semibold tabular-nums ${accent}`}>{value}</div>
    </div>
  );
}
