"use client";

import { useEffect, useRef, useState } from "react";
import { API_URL, type Signal } from "@/lib/api";

interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface ChartPayload {
  ticker: string;
  candles: Candle[];
  supports?: number[];
  resistances?: number[];
  oof_signals?: unknown[];
}

/** carga el vendor una sola vez (global window.LightweightCharts, v4.2) */
let scriptPromise: Promise<void> | null = null;
function loadVendor(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  const w = window as unknown as { LightweightCharts?: unknown };
  if (w.LightweightCharts) return Promise.resolve();
  if (!scriptPromise) {
    scriptPromise = new Promise<void>((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "/vendor/lightweight-charts.standalone.production.js";
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("no se pudo cargar lightweight-charts"));
      document.head.appendChild(s);
    });
  }
  return scriptPromise;
}

/** tipado minimo del API v4 que usamos (el vendor no expone tipos) */
interface LWChart {
  remove(): void;
  applyOptions(o: Record<string, unknown>): void;
  addCandlestickSeries(o: Record<string, unknown>): CandleSeries;
  addLineSeries(o: Record<string, unknown>): PriceLineHost;
  timeScale(): { fitContent(): void };
}
interface CandleSeries {
  setData(data: unknown): void;
  createPriceLine(o: Record<string, unknown>): unknown;
}
interface PriceLineHost {
  createPriceLine(o: Record<string, unknown>): unknown;
}

const GOLD = "#d4a843";
const GREEN = "#34d399";
const RED = "#fb7185";
const GRID = "rgba(139,148,173,0.08)";

export default function PriceChart({ signal }: { signal: Signal }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<LWChart | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const ticker = signal.ticker;
    let disposed = false;

    async function build() {
      const container = containerRef.current;
      if (!container) return;
      setLoading(true);
      setError(null);
      try {
        await loadVendor();
        if (disposed || !containerRef.current) return;
        const w = window as unknown as { LightweightCharts: {
          createChart: (el: HTMLElement, o: Record<string, unknown>) => LWChart;
        } };

        // velas desde la API unificada
        const res = await fetch(`${API_URL}/api/chart/${ticker}?period=2y&days=260`);
        if (!res.ok) throw new Error(`API chart ${res.status}`);
        const data = (await res.json()) as ChartPayload;

        const chart = w.LightweightCharts.createChart(container, {
          layout: {
            background: { color: "transparent" },
            textColor: "#8b94ad",
            fontSize: 11,
          },
          grid: {
            vertLines: { color: GRID },
            horzLines: { color: GRID },
          },
          rightPriceScale: { borderColor: "rgba(139,148,173,0.15)" },
          timeScale: { borderColor: "rgba(139,148,173,0.15)" },
          crosshair: {
            vertLine: { color: "rgba(212,168,67,0.35)", labelBackgroundColor: GOLD },
            horzLine: { color: "rgba(212,168,67,0.35)", labelBackgroundColor: GOLD },
          },
          width: container.clientWidth,
          height: 420,
        });

        const series = chart.addCandlestickSeries({
          upColor: GREEN,
          downColor: RED,
          wickUpColor: GREEN,
          wickDownColor: RED,
          borderVisible: false,
        });
        series.setData(
          data.candles.map((c) => ({
            time: c.time,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          })),
        );

        // niveles operativos del modelo: Entry / SL / TP1 / TP2
        const levels: [string, number | null | undefined, string, boolean][] = [
          ["Entrada", signal.entry, GOLD, false],
          ["Stop", signal.stop_loss, RED, true],
          ["TP1", signal.take_profits?.[0]?.price, GREEN, false],
          ["TP2", signal.take_profits?.[1]?.price, GREEN, true],
        ];
        for (const [title, price, color, dashed] of levels) {
          if (price === null || price === undefined) continue;
          series.createPriceLine({
            price,
            color,
            lineWidth: 1,
            lineStyle: dashed ? 2 : 0,
            axisLabelVisible: true,
            title,
          });
        }

        chart.timeScale().fitContent();
        chartRef.current = chart;
      } catch (e) {
        if (!disposed) setError(e instanceof Error ? e.message : "error del grafico");
      } finally {
        if (!disposed) setLoading(false);
      }
    }

    build();

    return () => {
      disposed = true;
      chartRef.current?.remove();
      chartRef.current = null;
    };
  }, [signal]);

  return (
    <div className="glass rounded-xl p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-mist-200">
          {signal.ticker} · velas 1D
        </h3>
        <div className="flex gap-3 text-[10px] text-mist-400">
          <LegendDot color={GOLD} label="Entrada" />
          <LegendDot color={RED} label="Stop" />
          <LegendDot color={GREEN} label="TP1/TP2" />
        </div>
      </div>
      <div className="relative mt-3">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center text-xs text-mist-400">
            cargando velas…
          </div>
        )}
        {error && (
          <div className="absolute inset-0 z-10 flex items-center justify-center text-xs text-rose-300">
            {error}
          </div>
        )}
        <div ref={containerRef} className="h-[420px] w-full" />
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="h-1.5 w-4 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}
