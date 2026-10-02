import Link from "next/link";
import SignalCard from "@/components/SignalCard";
import TransparencyPanel from "@/components/TransparencyPanel";
import { SessionButton } from "@/lib/clerk";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

async function getData() {
  try {
    const [health, signals] = await Promise.all([
      api.health(),
      api.signalsCache(),
    ]);
    return { health, signals, error: null as string | null };
  } catch {
    return { health: null, signals: null, error: "API no disponible" };
  }
}

export default async function Dashboard() {
  const { health, signals, error } = await getData();

  const rows = signals?.signals ?? [];
  const buys = rows.filter((s) => s.side === "largo" || s.side === "buy").length;
  const sells = rows.length - buys;
  const sched = (health?.scheduler ?? {}) as Record<string, unknown>;

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      {/* header */}
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="glow-gold flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-gold-400 to-gold-500 text-lg font-black text-ink-950">
            ML
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-mist-100">
              ML_Bolsa <span className="text-gold-400">Terminal</span>
            </h1>
            <p className="text-xs text-mist-400">
              Screener cuantitativo · S&P 500 · señales sin neutrales
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <Link
            href="/screener"
            className="glass rounded-lg px-3 py-1.5 font-medium text-mist-200 hover:text-gold-400"
          >
            Screener
          </Link>
          <Link
            href="/pricing"
            className="rounded-lg bg-gold-500 px-3 py-1.5 font-bold text-ink-950 hover:opacity-90"
          >
            Pro
          </Link>
          <StatusDot
            ok={!!health && health.status === "ok"}
            label={
              health
                ? `API ${health.status} · DB ${health.database}`
                : "API offline"
            }
          />
          <StatusDot
            ok={!!health?.cache?.ping_ok}
            label={health ? `cache ${health.cache.backend}` : "cache —"}
          />
          <StatusDot
            ok={!!health?.model_loaded}
            label={health?.model_loaded ? "modelo cargado" : "modelo —"}
          />
          <SessionButton />
        </div>
      </header>

      {/* banda de estado del scheduler */}
      <div className="glass mt-6 flex flex-wrap items-center justify-between gap-3 rounded-xl px-5 py-3 text-xs text-mist-400">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-gold-400 opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-gold-400" />
          </span>
          <span className="text-mist-200">
            Scheduler cada 5 min
          </span>
          <span>
            · ciclos: {String(sched.cycles ?? 0)} · última escritura:{" "}
            {String(sched.last_written ?? 0)} señales
            {sched.last_cycle_seconds
              ? ` · ${String(sched.last_cycle_seconds)}s`
              : ""}
          </span>
        </div>
        <div>
          proveedores:{" "}
          <span className="text-mist-200">
            {health
              ? Object.entries(health.providers)
                  .map(([k, v]) => `${k}: ${v}`)
                  .join(" · ")
              : "—"}
          </span>
        </div>
      </div>

      <div className="mt-6">
        <TransparencyPanel />
      </div>

      {/* señales */}
      <section className="mt-8">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-mist-200">
            Señales del ciclo
          </h2>
          <span className="text-xs text-mist-400">
            {rows.length} señales · {buys} largo · {sells} corto
          </span>
        </div>

        {error || rows.length === 0 ? (
          <div className="glass mt-4 rounded-xl p-10 text-center">
            <p className="text-sm text-mist-200">
              {error
                ? "No se pudo contactar la API. Arranca el backend con: uvicorn src.api.fastapi_app:app --port 8010"
                : "Sin señales aún: el scheduler aún no ha completado su primer ciclo."}
            </p>
            <p className="mt-2 text-xs text-mist-400">
              Ejecuta <code className="text-gold-400">python scripts/run_scheduler.py</code>{" "}
              o arranca la API con <code className="text-gold-400">SCHEDULER_EMBEDDED=1</code>. O
              abre el <Link href="/screener" className="text-gold-400 hover:underline">screener</Link>.
            </p>
          </div>
        ) : (
          <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {rows.slice(0, 12).map((s) => (
              <SignalCard key={`${s.ticker}-${s.trade_date ?? ""}`} signal={s} />
            ))}
          </div>
        )}
      </section>

      <footer className="mt-12 border-t border-mist-400/10 pt-4 text-[11px] text-mist-400">
        Datos con retardo (IEX/yfinance) · No es asesoramiento financiero ·
        Expectativa medida sobre datos out-of-sample, no garantizada.
      </footer>
    </main>
  );
}

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="glass flex items-center gap-1.5 rounded-full px-3 py-1">
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          ok ? "bg-emerald-400" : "bg-rose-400"
        }`}
      />
      <span className="text-mist-200">{label}</span>
    </span>
  );
}
