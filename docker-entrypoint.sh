#!/bin/sh
set -e

# migraciones al arrancar (solo el servicio que las ejecute; idempotente)
if [ "${RUN_MIGRATIONS:-1}" = "1" ]; then
  echo "[entrypoint] aplicando migraciones alembic..."
  alembic upgrade head || echo "[entrypoint] aviso: migraciones omitidas (DB no lista)"
fi

exec "$@"
