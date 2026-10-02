"""Stripe (Sprint 6): checkout, webhooks y entitlements Free/Pro.

Stripe es OPCIONAL: sin STRIPE_SECRET_KEY la plataforma sigue siendo usable
(beta local) y el modo demo asigna tier free por defecto. Con claves
configuradas:

- POST /api/billing/checkout  -> session de Checkout (plan mensual Pro).
- POST /api/billing/webhook   -> mantiene subscriptions + users.tier al dia.
- GET  /api/billing/status    -> tier + estado de suscripcion del usuario.

Webhooks: firma verificada con STRIPE_WEBHOOK_SECRET via stripe SDK si esta
disponible; en su defecto, rechaza (401). Eventos gestionados:
checkout.session.completed, customer.subscription.updated/deleted.

Los precios vienen de STRIPE_PRICE_PRO (price_xxx recomendado); si no hay
price, se crea un Price inline con STRIPE_PRICE_PRO_AMOUNT (centavos).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ..db.models import Subscription, User
from ..db.session import get_session_factory, init_db
from ..utils.logger import get_logger

logger = get_logger(__name__)

FREE_TIER = "free"
PRO_TIER = "pro"

# activos demo accesibles en tier free (mandato Sprint 6)
FREE_TICKERS = {"AAPL", "TSLA", "BTC-USD"}

# universos que sirve el screener (para /api/universes)
UNIVERSES = [
    {"id": "sp500", "label": "S&P 500"},
    {"id": "ndx100", "label": "NASDAQ 100"},
    {"id": "fx_major", "label": "Forex"},
    {"id": "crypto20", "label": "Crypto"},
]


def stripe_configured() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY", "").strip())


def price_pro_id() -> str:
    return os.environ.get("STRIPE_PRICE_PRO", "").strip()


def success_url() -> str:
    return os.environ.get(
        "STRIPE_SUCCESS_URL", "http://localhost:3000/pricing?status=success"
    )


def cancel_url() -> str:
    return os.environ.get(
        "STRIPE_CANCEL_URL", "http://localhost:3000/pricing?status=cancel"
    )


# ----------------------------------------------------------------------
# usuarios (Sprint 6 usa email como identidad; Clerk slurpara users.clerk_id)
# ----------------------------------------------------------------------

def get_or_create_user(email: str, display_name: str = "") -> User:
    """identidad minima mientras Clerk no consume la API (beta local).

    clerk_id se rellena con un marcador derivado del email (la columna es
    NOT NULL + unique); el login real de Clerk lo sustituira por su id."""
    init_db()
    with get_session_factory()() as s:
        email = (email or "").strip().lower()
        u = s.query(User).filter_by(email=email).one_or_none()
        if u is None:
            u = User(email=email,
                     clerk_id=f"local:{email}",
                     display_name=display_name or email.split("@")[0],
                     tier=FREE_TIER)
            s.add(u)
        elif display_name and not u.display_name:
            u.display_name = display_name
        s.commit()
        s.refresh(u)
        return u


def user_status(user_id: int) -> Dict[str, Any]:
    """tier + resumen de suscripcion para /api/billing/status."""
    init_db()
    with get_session_factory()() as s:
        u = s.get(User, user_id)
        if u is None:
            return {"user_id": user_id, "tier": FREE_TIER, "exists": False,
                    "is_pro": False, "subscription": None}
        sub = (s.query(Subscription).filter_by(user_id=user_id)
               .order_by(Subscription.updated_at.desc()).first())
        return {
            "user_id": u.id,
            "email": u.email,
            "tier": u.tier or FREE_TIER,
            "exists": True,
            "is_pro": (u.tier or FREE_TIER) == PRO_TIER,
            "free_tickers": sorted(FREE_TICKERS),
            "subscription": None if sub is None else {
                "status": sub.status,
                "period_end": sub.period_end.isoformat() if sub.period_end else None,
                "stripe_customer_id": sub.stripe_customer_id,
            },
        }


# ----------------------------------------------------------------------
# entitlements
# ----------------------------------------------------------------------

def can_access_ticker(ticker: str, tier: str) -> bool:
    """el detalle de activo es libre en demo; pro desbloquea todo el universo."""
    if (tier or FREE_TIER) == PRO_TIER:
        return True
    return ticker.upper() in FREE_TICKERS


def tier_limit(tier: str) -> Dict[str, Any]:
    """limites declarativos del tier (los consume API y UI)."""
    pro = (tier or FREE_TIER) == PRO_TIER
    return {
        "tier": PRO_TIER if pro else FREE_TIER,
        "max_watchlist_tickers": 50 if pro else 5,
        "screener_universes": ["sp500", "ndx100", "fx_major", "crypto20"] if pro
        else ["sp500"],
        "advanced_indicators": pro,
        "ai_insights": pro,
        "interactive_chart": pro,
        "alerts": pro,               # alertas Telegram/Email exclusivas Pro
        "delayed_hours": 0 if pro else 24,
    }


# ----------------------------------------------------------------------
# checkout + webhooks
# ----------------------------------------------------------------------

def create_checkout_session(user: User) -> Dict[str, Any]:
    """crea una session de Checkout para el plan Pro (requiere claves)."""
    if not stripe_configured():
        raise RuntimeError("Stripe no configurado (STRIPE_SECRET_KEY)")
    try:
        import stripe
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("falta el SDK de stripe (pip install stripe)") from exc

    amount_cents = int(os.environ.get("STRIPE_PRICE_PRO_AMOUNT", "2900"))
    params: Dict[str, Any] = {
        "mode": "subscription",
        "success_url": success_url(),
        "cancel_url": cancel_url(),
        "client_reference_id": str(user.id),
        "customer_email": user.email,
        "line_items": [{
            "quantity": 1,
            **({"price": price_pro_id()} if price_pro_id() else {
                "price_data": {
                    "currency": "usd",
                    "unit_amount": amount_cents,
                    "recurring": {"interval": "month"},
                    "product_data": {"name": "ML_Bolsa Pro"},
                },
            }),
        }],
        "metadata": {"user_id": str(user.id), "email": user.email},
        "subscription_data": {"metadata": {"user_id": str(user.id)}},
    }
    session = stripe.checkout.Session.create(**params)
    return {"id": session.id, "url": session.url}


def apply_stripe_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """aplica un evento de webhook y devuelve que hizo (idempotente-ish).

    checkout.session.completed        -> Subscription activa + tier pro
    customer.subscription.updated     -> sincroniza status/period_end
    customer.subscription.deleted     -> tier free + subscription canceled
    """
    etype = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}

    if etype == "checkout.session.completed":
        uid = int(obj.get("client_reference_id")
                  or (obj.get("metadata") or {}).get("user_id") or 0)
        if not uid:
            return {"applied": False, "reason": "sin user_id"}
        return _upsert_subscription(
            uid, customer=obj.get("customer") or "",
            subscription=obj.get("subscription") or "",
            status="active", period_end=None,
        )

    if etype == "customer.subscription.updated":
        uid = int((obj.get("metadata") or {}).get("user_id") or 0)
        if not uid:
            return {"applied": False, "reason": "sin user_id en metadata"}
        period_end = None
        pe = obj.get("current_period_end")
        if pe:
            from datetime import datetime, timezone
            period_end = datetime.fromtimestamp(int(pe), tz=timezone.utc)
        status = obj.get("status") or "inactive"
        return _upsert_subscription(
            uid, customer=obj.get("customer") or "",
            subscription=obj.get("id") or "",
            status=status, period_end=period_end,
            pro=(status == "active"),
        )

    if etype == "customer.subscription.deleted":
        uid = int((obj.get("metadata") or {}).get("user_id") or 0)
        if not uid:
            return {"applied": False, "reason": "sin user_id en metadata"}
        return _upsert_subscription(uid, customer=obj.get("customer") or "",
                                    subscription=obj.get("id") or "",
                                    status="canceled", period_end=None, pro=False)

    return {"applied": False, "reason": f"evento no gestionado: {etype}"}


def _upsert_subscription(user_id: int, customer: str, subscription: str,
                         status: str, period_end, pro: bool = True) -> Dict[str, Any]:
    """una fila Subscription por usuario + tier del User sincronizado."""
    init_db()
    tier = PRO_TIER if (pro and status == "active") else FREE_TIER
    with get_session_factory()() as s:
        sub = s.query(Subscription).filter_by(user_id=user_id).one_or_none()
        if sub is None:
            sub = Subscription(user_id=user_id)
            s.add(sub)
        if customer:
            sub.stripe_customer_id = customer
        if subscription:
            sub.stripe_subscription_id = subscription
        sub.status = status
        if period_end is not None:
            sub.period_end = period_end
        u = s.get(User, user_id)
        if u is not None:
            u.tier = tier
        s.commit()
    logger.info(f"stripe: user {user_id} -> tier={tier} status={status}")
    return {"applied": True, "user_id": user_id, "tier": tier, "status": status}


def verify_webhook(payload: bytes, signature: Optional[str]) -> Dict[str, Any]:
    """verifica la firma con el SDK (STRIPE_WEBHOOK_SECRET) o rechaza."""
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise PermissionError("STRIPE_WEBHOOK_SECRET no configurado")
    if not signature:
        raise PermissionError("falta la cabecera stripe-signature")
    import stripe

    event = stripe.Webhook.construct_event(payload, signature, secret)
    return event if isinstance(event, dict) else event.to_dict()
