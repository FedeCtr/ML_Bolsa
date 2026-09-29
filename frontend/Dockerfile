# Frontend Next.js 15 (output standalone)

# ---- build ----
FROM node:22-alpine AS builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
RUN npm ci
COPY frontend/ .
# NEXT_PUBLIC_* se inlinea en build: pasar por --build-arg
ARG NEXT_PUBLIC_API_URL=http://localhost:8010
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

# ---- runtime ----
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
