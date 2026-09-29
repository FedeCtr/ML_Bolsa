"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import SignalCard from "@/components/SignalCard";
import { API_URL, type Signal } from "@/lib/api";

const MARKETS: { id: string; label: string }[] = [
  { id: "sp500", label: "S&P 500" },
  { id: "ndx100", label: "NASDAQ 100" },
  { id: "fx_major", label: "Forex" },
  { id: "crypto20", label: "Cripto" },
];

const SIGNALS = ["COMPRA_FUERTE", "COMPRA", "VENTA", "VENTA_FUERTE"] as const;

export default function ScreenerPage() {
  const [market, setMarket] = useState("sp500");
  const [signals, setSignals] = useState<string[]>([]);
  const [minConfidence, setMinConfidence] = useState(50);
  const [onlyTradable, setOnlyTradable] = useState(true);
  const [rows, setRows] = useState<Signal[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);

  const nFilters = useMemo(
    () => signals.length + (minConfidence > 0 ? 1 : 0) + (onlyTradable ? 1 : 0),
    [signals, minConfidence, onlyTradable],
  );

  async function runScan() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/api/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          universe: market,
          period: "3mo",
          force: lastQuery !== null,
        }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      await res.json();
      setLastQuery(`${market}:${signals.join(",")}:${minConfidence}`);
      // el escaneo corre en background; traemos resultados parciales ya
      setTimeout(() => void load(), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "error lanzando escaneo");
    } finally {
      setLoading(false);
    }
  }

  async function load() {
    setError(null);
    try {
      const qs = new URLSearchParams({ limit: "60" });
      if (signals.length) qs.set("signal", signals.join(","));
      if (minConfidence > 0) qs.set("min_confidence", String(minConfidence));
      if (onlyTradable) qs.set("only_tradable", "1");
      const res = await fetch(`${API_URL}/api/signals-cache?${qs}`);
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      setRows(data.signals ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "error cargando señales");
    }
  }

  function toggleSignal(s: string) {
    setSignals((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s],
    );
  }

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <header className="flex flex-wrap items-baseline justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-mist-100">Screener</h1>
          <p className="mt-0.5 text-xs text-mist-400">
            Señales del ciclo del scheduler · filtros del lado del cliente sobre
            el espejo en cache (respuesta en menos de 1s)
          </p>
        </div>
        <Link href="/" className="text-xs text-gold-400 hover:underline">
          ← dashboard
        </Link>
      </header>

      {/* filtros */}
      <section className="glass mt-6 rounded-xl p-5">
        <div className="grid gap-5 lg:grid-cols-[auto_1fr_auto]">
          <div>
            <Label>Mercado</Label>
            <div className="mt-2 flex flex-wrap gap-2">
              {MARKETS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMarket(m.id)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                    market === m.id
                      ? "border-gold-400/50 bg-gold-500/15 text-gold-400"
                      : "border-mist-400/15 text-mist-400 hover:text-mist-200"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <Label>Tipo de señal</Label>
            <div className="mt-2 flex flex-wrap gap-2">
              {SIGNALS.map((s) => (
                <button
                  key={s}
                  onClick={() => toggleSignal(s)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                    signals.includes(s)
                      ? s.startsWith("COMPRA")
                        ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-300"
                        : "border-rose-500/50 bg-rose-500/15 text-rose-300"
                      : "border-mist-400/15 text-mist-400 hover:text-mist-200"
                  }`}
                >
                  {s.replace("_", " ")}
                </button>
              ))}
            </div>
          </div>

          <div className="min-w-[220px]">
            <div className="flex items-baseline justify-between">
              <Label>Confianza mínima</Label>
              <span className="text-xs font-semibold text-gold-400">
                {minConfidence}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={90}
              step={5}
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              className="mt-3 w-full accent-gold-400"
            />
            <label className="mt-3 flex items-center gap-2 text-xs text-mist-400">
              <input
                type="checkbox"
                checked={onlyTradable}
                onChange={(e) => setOnlyTradable(e.target.checked)}
                className="accent-gold-400"
              />
              solo operables (régimen normal)
            </label>
          </div>
        </div>

        <div className="mt-5 flex items-center gap-3 border-t border-mist-400/10 pt-4">
          <button
            onClick={load}
            disabled={loading}
            className="rounded-lg bg-gold-500 px-4 py-2 text-xs font-bold text-ink-950 transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            Aplicar filtros
          </button>
          <button
            onClick={runScan}
            disabled={loading}
            className="rounded-lg border border-gold-400/40 px-4 py-2 text-xs font-semibold text-gold-400 transition-colors hover:bg-gold-500/10 disabled:opacity-50"
            title="Lanza un escaneo en background del universo seleccionado"
          >
            Escanear {MARKETS.find((m) => m.id === market)?.label} ahora
          </button>
          {lastQuery && (
            <span className="text-[10px] text-mist-400">
              escaneo en background; los resultados llegan al espejo al terminar el ciclo
            </span>
          )}
          <span className="ml-auto text-xs text-mist-400">
            {nFilters} filtro{nFilters === 1 ? "" : "s"} activo{nFilters === 1 ? "" : "s"}
          </span>
        </div>
      </section>

      {/* resultados */}
      <section className="mt-6">
        {error && (
          <div className="glass rounded-xl p-6 text-center text-sm text-rose-300">
            {error}
          </div>
        )}
        {rows === null && !error && (
          <div className="glass rounded-xl p-6 text-center text-sm text-mist-400">
            Configura los filtros y pulsa «Aplicar filtros».
          </div>
        )}
        {rows !== null && rows.length === 0 && !error && (
          <div className="glass rounded-xl p-6 text-center text-sm text-mist-400">
            Sin señales para estos filtros. Lanza un escaneo o relaja los criterios.
          </div>
        )}
        {rows !== null && rows.length > 0 && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {rows.map((s) => (
              <Link
                key={`${s.ticker}-${s.trade_date ?? ""}`}
                href={`/asset/${s.ticker}`}
                className="block"
              >
                <SignalCard signal={s} />
              </Link>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wider text-mist-400">
      {children}
    </span>
  );
}
