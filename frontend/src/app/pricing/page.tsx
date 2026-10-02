"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  billingConfig,
  billingStatus,
  localEmail,
  setLocalEmail,
  startCheckout,
  type BillingConfig,
  type BillingStatus,
} from "@/lib/billing";

const FEATURE_ROWS: { label: string; free: string; pro: string }[] = [
  { label: "Señales diarias del scheduler", free: "Solo activos demo", pro: "Todo el mercado" },
  { label: "Screener", free: "3 activos demo", pro: "590 activos (S&P, NDX, FX, Crypto)" },
  { label: "Indicadores avanzados y gráfico", free: "—", pro: "✓" },
  { label: "AI Insights (TreeSHAP)", free: "—", pro: "✓" },
  { label: "Alertas Telegram / Email", free: "—", pro: "✓" },
  { label: "Retardo de señales", free: "24 h", pro: "Tiempo real" },
];

export default function PricingPage() {
  const [cfg, setCfg] = useState<BillingConfig | null>(null);
  const [status, setStatus] = useState<BillingStatus | null>(null);
  const [email, setEmail] = useState("");
  const [emailInput, setEmailInput] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const e = localEmail();
    setEmail(e);
    setEmailInput(e);
    billingConfig().then(setCfg).catch(() => setCfg(null));
  }, []);

  useEffect(() => {
    if (!email) { setStatus(null); return; }
    billingStatus(email).then(setStatus).catch(() => setStatus(null));
  }, [email]);

  async function saveEmail() {
    setLocalEmail(emailInput.trim());
    setEmail(emailInput.trim());
    setMsg(emailInput.trim() ? "sesión local guardada" : "sesión local borrada");
  }

  async function goPro() {
    if (!email) { setMsg("guarda tu email primero"); return; }
    setLoading(true);
    setMsg(null);
    try {
      const { checkout_url } = await startCheckout(email);
      window.location.href = checkout_url;
    } catch (e) {
      setMsg(e instanceof Error && e.message.includes(": 503")
        ? "Stripe no está configurado (STRIPE_SECRET_KEY).Producer beta: el plan Pro se activa manualmente."
        : e instanceof Error ? e.message : "error iniciando checkout");
    } finally {
      setLoading(false);
    }
  }

  const isPro = status?.is_pro ?? false;

  return (
    <main className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-mist-100">
            Planes de ML_Bolsa
          </h1>
          <p className="mt-1 text-xs text-mist-400">
            Señales cuantitativas con riesgo determinista · cancela cuando quieras
          </p>
        </div>
        <Link href="/" className="text-xs text-gold-400 hover:underline">
          ← dashboard
        </Link>
      </header>

      {/* sesion local */}
      <section className="glass mt-6 rounded-xl p-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-mist-400">
            Tu email
          </span>
          <input
            type="email"
            value={emailInput}
            onChange={(e) => setEmailInput(e.target.value)}
            placeholder="tu@email.com"
            className="min-w-[240px] flex-1 rounded-lg border border-mist-400/15 bg-ink-800/60 px-3 py-2 text-sm text-mist-100 outline-none placeholder:text-mist-400/50 focus:border-gold-400/50"
          />
          <button
            onClick={saveEmail}
            className="rounded-lg bg-gold-500 px-4 py-2 text-xs font-bold text-ink-950 hover:opacity-90"
          >
            Guardar
          </button>
          {status && (
            <span className="pill border border-mist-400/20 bg-ink-700/60 text-xs text-mist-300">
              plan actual:{" "}
              <b className={isPro ? "text-emerald-300" : "text-mist-100"}>
                {isPro ? "PRO" : "FREE"}
              </b>
            </span>
          )}
        </div>
        {msg && <p className="mt-2 text-xs text-gold-400">{msg}</p>}
        {status?.subscription?.status === "active" && (
          <p className="mt-2 text-xs text-emerald-300">
            Suscripción activa · próxima renovación{" "}
            {status.subscription.period_end
              ? new Date(status.subscription.period_end).toLocaleDateString("es-ES")
              : "—"}
          </p>
        )}
      </section>

      {/* planes */}
      <section className="mt-6 grid gap-5 md:grid-cols-2">
        <PlanCard
          name="Free"
          price="$0"
          desc="Prueba el motor con los activos demo."
          features={[
            "3 activos demo (AAPL, TSLA, BTC)",
            "Resumen de señal diario",
            "Señales con 24 h de retardo",
          ]}
          cta={isPro ? "Ya eres Pro" : email ? "Plan actual" : "Guarda tu email"}
          disabled={true}
        />
        <PlanCard
          name="Pro"
          price={
            cfg ? `$${(cfg.price_amount_cents / 100).toFixed(0)}` : "$29"
          }
          desc="Todo el mercado, insights de IA y alertas instantáneas."
          features={[
            "590 activos: S&P 500 + NDX + FX + Crypto",
            "Indicadores avanzados y gráfico interactivo",
            "AI Insights TreeSHAP por señal",
            "Alertas Telegram / Email instantáneas",
            "Señales en tiempo real",
          ]}
          highlight
          cta={isPro ? "Suscripción activa ✓" : loading ? "Redirigiendo…" : "Suscribirme con Stripe"}
          disabled={isPro || loading}
          onClick={goPro}
        />
      </section>

      {/* comparativa */}
      <section className="glass mt-6 overflow-x-auto rounded-xl p-5">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="text-mist-400">
              <th className="pb-2 font-semibold uppercase tracking-wider">Característica</th>
              <th className="pb-2 font-semibold uppercase tracking-wider">Free</th>
              <th className="pb-2 font-semibold uppercase tracking-wider text-gold-400">Pro</th>
            </tr>
          </thead>
          <tbody>
            {FEATURE_ROWS.map((r) => (
              <tr key={r.label} className="border-t border-mist-400/10">
                <td className="py-2.5 text-mist-200">{r.label}</td>
                <td className="py-2.5 text-mist-400">{r.free}</td>
                <td className="py-2.5 font-semibold text-gold-300">{r.pro}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-4 text-[10px] leading-relaxed text-mist-400">
          La confianza del modelo no es la precisión. Los niveles de riesgo son
          deterministas (ATR + pivotes), no salen del modelo. Nada de esto es
          asesoramiento financiero.
        </p>
      </section>
    </main>
  );
}

function PlanCard({
  name, price, desc, features, cta, highlight, disabled, onClick,
}: {
  name: string;
  price: string;
  desc: string;
  features: string[];
  cta: string;
  highlight?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}) {
  return (
    <div
      className={`glass relative rounded-xl p-6 ${
        highlight ? "border border-gold-400/40 shadow-[0_0_40px_-15px_rgba(212,168,67,0.4)]" : ""
      }`}
    >
      {highlight && (
        <span className="absolute -top-2.5 right-4 rounded-full bg-gold-500 px-2.5 py-0.5 text-[10px] font-bold text-ink-950">
          RECOMENDADO
        </span>
      )}
      <h2 className="text-sm font-bold uppercase tracking-wider text-mist-100">{name}</h2>
      <div className="mt-2 flex items-baseline gap-1">
        <span className={`text-3xl font-bold ${highlight ? "text-gold-400" : "text-mist-100"}`}>
          {price}
        </span>
        {highlight && <span className="text-xs text-mist-400">/ mes</span>}
      </div>
      <p className="mt-1 text-xs text-mist-400">{desc}</p>
      <ul className="mt-4 space-y-2 text-xs text-mist-200">
        {features.map((f) => (
          <li key={f} className="flex gap-2">
            <span className="text-gold-400">·</span>
            {f}
          </li>
        ))}
      </ul>
      <button
        onClick={onClick}
        disabled={disabled}
        className={`mt-5 w-full rounded-lg px-4 py-2.5 text-xs font-bold transition-opacity ${
          highlight
            ? "bg-gold-500 text-ink-950 hover:opacity-90"
            : "border border-mist-400/20 text-mist-300"
        } disabled:cursor-not-allowed disabled:opacity-50`}
      >
        {cta}
      </button>
    </div>
  );
}
