// Proxy runtime /api/proxy/* -> API (Sprint 6).
// Los rewrites de Next se hornean en build; este route handler lee
// API_INTERNAL_URL en runtime, asi el compose decide el destino sin
// reconstruir la imagen (y el navegador nunca ve la URL del backend).
import { NextRequest, NextResponse } from "next/server";

const API_INTERNAL = process.env.API_INTERNAL_URL || "http://localhost:8010";

const HOP_BY_HOP = new Set([
  "connection", "keep-alive", "transfer-encoding", "upgrade",
  "proxy-authenticate", "proxy-authorization", "te", "trailers", "host",
]);

async function forward(req: NextRequest, method: string, path: string[]) {
  const qs = req.nextUrl.search ?? "";
  const target = `${API_INTERNAL}/${path.join("/")}${qs}`;
  const headers = new Headers();
  req.headers.forEach((v, k) => {
    if (!HOP_BY_HOP.has(k.toLowerCase())) headers.set(k, v);
  });
  const body = method === "GET" || method === "HEAD"
    ? undefined
    : await req.arrayBuffer();

  const res = await fetch(target, { method, headers, body, cache: "no-store" });
  const out = new NextResponse(res.body, { status: res.status });
  const ct = res.headers.get("content-type");
  if (ct) out.headers.set("content-type", ct);
  return out;
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, ctx: Ctx) {
  return forward(req, "GET", (await ctx.params).path);
}
export async function POST(req: NextRequest, ctx: Ctx) {
  return forward(req, "POST", (await ctx.params).path);
}
export async function PUT(req: NextRequest, ctx: Ctx) {
  return forward(req, "PUT", (await ctx.params).path);
}
export async function PATCH(req: NextRequest, ctx: Ctx) {
  return forward(req, "PATCH", (await ctx.params).path);
}
export async function DELETE(req: NextRequest, ctx: Ctx) {
  return forward(req, "DELETE", (await ctx.params).path);
}
