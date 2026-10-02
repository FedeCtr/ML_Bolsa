- El servidor Flask heredado fue jubilado (Sprint 6): la unica API
  es FastAPI (`src.api.fastapi_app`); las vistas viven en `frontend/`.

## Sprint 6: monetización (Stripe)

Variables de entorno (todas opcionales; sin ellas la plataforma corre en
modo demo con tier free):

| Variable | Uso |
|---|---|
| `STRIPE_SECRET_KEY` | activa el checkout Pro (sin ella, /api/billing/checkout responde 503) |
| `STRIPE_WEBHOOK_SECRET` | firma del webhook `/api/billing/webhook` (sin ella, 401) |
| `STRIPE_PRICE_PRO` | price_xxx del plan Pro en Stripe (recomendado) |
| `STRIPE_PRICE_PRO_AMOUNT` | si no hay price: precio inline en centavos (default 2900) |
| `STRIPE_SUCCESS_URL` / `STRIPE_CANCEL_URL` | retornos del checkout (default localhost:3000/pricing) |

El webhook mantiene `subscriptions` y `users.tier` al día. El frontend proxea
`/api/proxy/*` a la API con `API_INTERNAL_URL` (compose: `http://api:8000`),
así el navegador nunca ve la URL del backend.

Configuración del webhook en Stripe Dashboard → Developers → Webhooks:
endpoint `https://TU-DOMINIO/api/billing/webhook`, eventos
`checkout.session.completed`, `customer.subscription.updated`,
`customer.subscription.deleted`.
