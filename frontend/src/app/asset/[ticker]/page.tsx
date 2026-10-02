"use client";

import Link from "next/link";
import { use, useEffect, useState } from "react";
import InsightsCard from "@/components/InsightsCard";
import PriceChart from "@/components/PriceChart";
import { API_URL, type Signal } from "@/lib/api";
import { billingStatus, localEmail, type BillingStatus } from "@/lib/billing";

function fmt(n?: number | null, digits = 2): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString("es-ES", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

const SIGNAL_STYLES: Record<string, string> = {
  COMPRA_FUERTE: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  COMPRA: "bg-emerald-500/10 text-emerald-400/90 border-emerald-500/20",
  VENTA_FUERTE: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  VENTA: "bg-rose-500/10 text-rose-400/90 border-rose-500/20",
};

export default function AssetPage({
  params,
}: {
  params: Promise<{ ticker: string }>;
}) {
  const { ticker: raw } = use(params);
  const ticker = raw.toUpperCase();
  const [signal, setSignal] = useState<Signal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [needUpgrade, setNeedUpgrade] = useState(false);
  const [billing, setBilling] = useState<BillingStatus | null>(null);

  useEffect(() => {
    const em = localEmail();
    if (!em) return; // beta local abierta
    billingStatus(em).then(setBilling).catch(() => setBilling(null));
  }, []);

  useEffect(() => {
    let alive = true;
    setSignal(null);
    setError(null);
    setNeedUpgrade(false);
    const em = localEmail();
    fetch(`${API_URL}/api/signal/${ticker}${em ? `?email=${encodeURIComponent(em)}` : ""}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`API ${r.status}`))))
      .then((d: Signal) => {
        if (alive) setSignal(d);
      })
      .catch((e: Error) => {
        if (!alive) return;
        if (e.message.endsWith("402")) setNeedUpgrade(true);
        else setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, [ticker]);

  if (needUpgrade) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-10">
        <div className="glass mx-auto max-w-md rounded-xl p-10 text-center">
          <div className="text-3xl">🔒</div>
          <h1 className="mt-3 text-lg font-bold text-mist-100">{ticker} es Pro</h1>
          <p className="mt-2 text-sm text-mist-400">
            Desbloquea el análisis completo y las predicciones de esta acción con
            Pro: gráfico interactivo, indicadores avanzados y AI Insights.
          </p>
          <Link
            href="/pricing"
            className="mt-5 inline-block rounded-lg bg-gold-500 px-5 py-2.5 text-xs font-bold text-ink-950 hover:opacity-90"
          >
            Ver Pro
          </Link>
          <Link href="/" className="mt-4 block text-xs text-mist-400 hover:underline">
            ← volver al dashboard
          </Link>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-10">
        <div className="glass rounded-xl p-10 text-center text-sm text-mist-200">
          No se pudo obtener la señal de {ticker} ({error}).
          <div className="mt-2 text-xs text-mist-400">
            Verifica que la API corre en {API_URL}
          </div>
          <Link href="/" className="mt-4 inline-block text-gold-400 hover:underline">
            ← volver al dashboard
          </Link>
        </div>
      </main>
    );
  }

  if (!signal) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-10">
        <div className="glass rounded-xl p-10 text-center text-sm text-mist-400">
          cargando {ticker}…
        </div>
      </main>
    );
  }

  const tp = signal.take_profits ?? [];
  const style = SIGNAL_STYLES[signal.signal] ??
    "bg-ink-700/60 text-mist-200 border-mist-400/20";
  const rr = signal.risk_reward ?? null;
  // paywall visual Sprint 6: con sesion free, el panel avanzado se difumina
  const locked = !!billing && !billing.is_pro && !billing.free_tickers.includes(ticker);

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-mist-100">
              {ticker}
            </h1>
            <span className={`pill border ${style}`}>
              {signal.signal.replace("_", " ")}
            </span>
            {signal.trading_allowed === false && (
              <span className="pill border border-rose-500/30 bg-rose-500/10 text-rose-300">
                trading suspendido
              </span>
            )}
          </div>
          <p className="mt-1 text-xs text-mist-400">
            {signal.name ?? ""} {signal.sector ? `· ${signal.sector}` : ""} ·
            régimen {typeof signal.regime === "string" ? signal.regime : "—"} ·
            precio {fmt(signal.price)}
          </p>
        </div>
        <Link href="/" className="text-xs text-gold-400 hover:underline">
          ← dashboard
        </Link>
      </header>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Level label="Entrada" value={fmt(signal.entry)} />
        <Level label="Stop Loss" value={fmt(signal.stop_loss)} accent="text-rose-300" />
        <Level label="TP1" value={fmt(tp[0]?.price)} accent="text-emerald-300" />
        <Level label="TP2" value={fmt(tp[1]?.price)} accent="text-emerald-300" />
        <Level label="R/R" value={rr ? `1:${fmt(rr, 1)}` : "—"} accent="text-gold-400" />
        <Level label="Tamaño" value={signal.position_size_pct ? `${fmt(signal.position_size_pct, 1)}%` : "—"} />
      </div>

      <div className="relative mt-6 grid grid-cols-1 gap-6 xl:grid-cols-3">
        <div className={`xl:col-span-2 ${locked ? "pointer-events-none blur-sm" : ""}`}>
          <PriceChart signal={signal} />
        </div>
        <div className="space-y-6">
          <div className={locked ? "pointer-events-none blur-sm" : ""}>
            <InsightsCard ticker={ticker} />
          </div>
          <div className={`glass rounded-xl p-5 ${locked ? "pointer-events-none blur-sm" : ""}`}>
            <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-mist-200">
              Confianza
            </h3>
            <div className="mt-4 space-y-4">
              <Bar
                label="P(alza) calibrada"
                value={signal.prob_up ?? null}
                display={signal.prob_up ? `${fmt(signal.prob_up * 100, 1)}%` : "—"}
                color="bg-gold-400"
              />
              <Bar
                label="Confianza"
                value={signal.confidence_pct ? signal.confidence_pct / 100 : null}
                display={signal.confidence_pct ? `${fmt(signal.confidence_pct, 0)}%` : "—"}
                color="bg-emerald-400"
              />
              <Bar
                label="Convicción"
                value={signal.conviction_pct ? signal.conviction_pct / 100 : null}
                display={signal.conviction_pct ? `${fmt(signal.conviction_pct, 0)}%` : "—"}
                color="bg-emerald-300"
              />
            </div>
            <p className="mt-4 text-[11px] leading-relaxed text-mist-400">
              La confianza del modelo NO es la precisión. Los niveles de riesgo son
              deterministas (ATR + pivotes), no salen del modelo.
            </p>
          </div>
          <div className={`glass rounded-xl p-5 text-xs text-mist-400 ${locked ? "pointer-events-none blur-sm" : ""}`}>
            <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-mist-200">
              Contexto técnico
            </h3>
            <dl className="mt-3 space-y-2">
              <Row k="RSI (14)" v={fmt(signal.rsi, 1)} />
              <Row k="ATR %" v={fmt(signal.atr_pct, 2)} />
              <Row k="Volumen rel." v={fmt(signal.volume_ratio, 2)} />
              <Row k="Fecha señal" v={signal.trade_date ?? signal.as_of ?? "—"} />
            </dl>
          </div>
          {locked && (
            <div className="glass absolute inset-x-0 top-1/3 mx-auto max-w-sm rounded-xl border border-gold-400/40 p-6 text-center shadow-[0_0_60px_-10px_rgba(212,168,67,0.5)]">
              <div className="text-2xl">🔒</div>
              <p className="mt-2 text-sm font-semibold text-mist-100">
                Desbloquea el análisis completo y las predicciones de esta acción con Pro
              </p>
              <Link
                href="/pricing"
                className="mt-4 inline-block rounded-lg bg-gold-500 px-5 py-2 text-xs font-bold text-ink-950 hover:opacity-90"
              >
                Ver Pro
              </Link>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

function Level({
  label,
  value,
  accent = "text-mist-100",
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="glass rounded-xl px-4 py-3">
      <div className="text-[10px] uppercase tracking-wider text-mist-400">{label}</div>
      <div className={`mt-1 text-lg font-bold tabular-nums ${accent}`}>{value}</div>
    </div>
  );
}

function Bar({
  label,
  value,
  display,
  color,
}: {
  label: string;
  value: number | null;
  display: string;
  color: string;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs">
        <span className="text-mist-400">{label}</span>
        <span className="font-semibold text-mist-100">{display}</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink-700">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${Math.min(100, Math.max(0, (value ?? 0) * 100))}%` }}
        />
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between">
      <dt>{k}</dt>
      <dd className="font-semibold text-mist-100">{v}</dd>
    </div>
  );
}
