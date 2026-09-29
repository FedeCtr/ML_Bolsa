"use client";

import { useEffect, useState } from "react";
import { API_URL } from "@/lib/api";

interface InsightFactor {
  feature: string;
  label: string;
  value: string;
  direction: "buy" | "sell";
  raw?: number;
}

interface Insights {
  top_factors: InsightFactor[];
  buy_pressure: number;
  sell_pressure: number;
  logodds?: number;
  summary: string;
}

const FETCH_REVALIDATE_MS = 5 * 60 * 1000;

export default function InsightsCard({ ticker }: { ticker: string }) {
  const [data, setData] = useState<Insights | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(`${API_URL}/api/insights/${ticker}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`API ${r.status}`))))
      .then((d: Insights) => {
        if (alive) setData(d);
      })
      .catch((e: Error) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, [ticker]);

  if (error) return null; // explicabilidad opcional: silenciosa si falla

  if (!data) {
    return (
      <div className="glass rounded-xl p-5 text-xs text-mist-400">
        AI Insights: analizando factores…
      </div>
    );
  }

  const maxAbs = Math.max(
    ...data.top_factors.map((f) => Math.abs(parseFloat(f.value))),
    0.001,
  );

  return (
    <div className="glass rounded-xl p-5">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-gold-400">
          AI Insights
        </h3>
        <span className="text-[10px] text-mist-400">TreeSHAP · boosters</span>
      </div>

      <p className="mt-3 text-xs leading-relaxed text-mist-200">{data.summary}</p>

      <ul className="mt-4 space-y-2.5">
        {data.top_factors.map((f) => (
          <li key={f.feature}>
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-mist-200">{f.label}</span>
              <span
                className={`font-semibold tabular-nums ${
                  f.direction === "buy" ? "text-emerald-300" : "text-rose-300"
                }`}
              >
                {f.value}
              </span>
            </div>
            <div className="mt-1 flex h-1.5 overflow-hidden rounded-full bg-ink-700">
              <div className="flex w-full">
                <div className="flex w-1/2 justify-end">
                  {f.direction === "sell" && (
                    <div
                      className="h-full rounded-l-full bg-rose-400/80"
                      style={{ width: `${(Math.abs(parseFloat(f.value)) / maxAbs) * 100}%` }}
                    />
                  )}
                </div>
                <div className="flex w-1/2">
                  {f.direction === "buy" && (
                    <div
                      className="h-full rounded-r-full bg-emerald-400/80"
                      style={{ width: `${(Math.abs(parseFloat(f.value)) / maxAbs) * 100}%` }}
                    />
                  )}
                </div>
              </div>
            </div>
          </li>
        ))}
      </ul>

      <div className="mt-4 flex justify-between border-t border-mist-400/10 pt-3 text-[10px] text-mist-400">
        <span>
          presión alcista{" "}
          <span className="font-semibold text-emerald-300">
            {data.buy_pressure >= 0 ? "+" : ""}
            {data.buy_pressure.toFixed(2)}
          </span>
        </span>
        <span>
          presión bajista{" "}
          <span className="font-semibold text-rose-300">
            {data.sell_pressure.toFixed(2)}
          </span>
        </span>
      </div>
    </div>
  );
}
