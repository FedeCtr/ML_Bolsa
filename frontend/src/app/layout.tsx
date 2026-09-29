import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ML_Bolsa — Señales institucionales",
  description:
    "Screener cuantitativo del S&P 500 con expectativa transparente: PF 1.505, E[R] +0.268R. Señales con TP/SL R/R 1:2 y régimen de mercado.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es" className="dark">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
