"""Notificaciones instantaneas de senales fuertes (Sprint 5).

Canales:
- Telegram: bot API (TELEGRAM_BOT_TOKEN) a los chat_id guardados en users.
- Email: Resend (RESEND_API_KEY) a los emails de users.

Dedup: una senal (ticker, trade_date) se notifica una sola vez, guardando la
marca en Redis/cache (TTL 2 dias) para sobrevivir reinicios del scheduler.

Todo es best-effort: nunca lanza y siempre devuelve resumen; si no hay
credenciales o no hay usuarios suscriptos, no hace nada.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from ..cache import cache_get, cache_set
from ..db.models import User
from ..db.session import get_session_factory, init_db
from ..utils.logger import get_logger

logger = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
RESEND_API = "https://api.resend.com/emails"
ALERT_FROM = os.environ.get("ALERT_EMAIL_FROM", "ML_Bolsa <onboarding@resend.dev>")
DEDUP_TTL = 2 * 24 * 3600

STRONG_SIGNALS = {"COMPRA_FUERTE", "VENTA_FUERTE"}


def _telegram_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")


def _resend_key() -> str:
    return os.environ.get("RESEND_API_KEY", "")


def alerts_configured() -> bool:
    return bool(_telegram_token() or _resend_key())


# ----------------------------------------------------------------------
# formateo
# ----------------------------------------------------------------------

def format_message(d: Dict[str, Any]) -> str:
    """mensaje de texto (Telegram) de una senal fuerte."""
    icon = "🟢" if str(d.get("signal", "")).startswith("COMPRA") else "🔴"
    tps = d.get("take_profits") or []
    tp1 = tps[0]["price"] if tps and isinstance(tps[0], dict) else tps[0] if tps else None
    tp2 = tps[1]["price"] if len(tps) > 1 and isinstance(tps[1], dict) else (tps[1] if len(tps) > 1 else None)
    rr = d.get("risk_reward")
    lines = [
        f"{icon} ML_Bolsa · {d.get('ticker', '?')}",
        f"{str(d.get('signal', '')).replace('_', ' ')} · convicción {d.get('conviction_pct', '—')}%",
        f"Entrada {d.get('entry', '—')} · SL {d.get('stop_loss', '—')}"
        f" · TP1 {tp1 if tp1 is not None else '—'} · TP2 {tp2 if tp2 is not None else '—'}",
        f"R/R {f'1:{rr}' if rr else '—'} · régimen {d.get('regime', '—')}",
    ]
    return "\n".join(lines)


def format_email(d: Dict[str, Any]) -> Dict[str, str]:
    """subject + html de una senal fuerte."""
    text = format_message(d)
    subject = f"ML_Bolsa · {d.get('ticker')} · {str(d.get('signal', '')).replace('_', ' ')}"
    body = text.replace("\n", "<br/>")
    html = (
        '<div style="background:#0b0e14;color:#e8ebf4;font-family:Arial,sans-serif;'
        'padding:24px;border-radius:12px;max-width:520px">'
        '<div style="color:#d4a843;font-weight:bold;letter-spacing:2px;font-size:12px">'
        "ML_BOLSA · SEÑAL DE ALTA PROBABILIDAD</div>"
        f'<div style="margin-top:12px;font-size:15px;line-height:1.6">{body}</div>'
        '<div style="margin-top:16px;font-size:11px;color:#8b94ad">La confianza del modelo '
        "no es la precisión. No es asesoramiento financiero.</div></div>"
    )
    return {"subject": subject, "html": html}


# ----------------------------------------------------------------------
# canales
# ----------------------------------------------------------------------

def send_telegram(chat_id: str, text: str) -> bool:
    token = _telegram_token()
    if not token or not chat_id:
        return False
    try:
        import httpx

        resp = httpx.post(
            TELEGRAM_API.format(token=token),
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as exc:
        logger.warning(f"telegram fallo ({chat_id}): {exc}")
        return False


def send_email(to: str, subject: str, html: str) -> bool:
    key = _resend_key()
    if not key or not to:
        return False
    try:
        import httpx

        resp = httpx.post(
            RESEND_API,
            headers={"Authorization": f"Bearer {key}"},
            json={"from": ALERT_FROM, "to": [to], "subject": subject, "html": html},
            timeout=15,
        )
        return resp.status_code < 300
    except Exception as exc:
        logger.warning(f"email fallo ({to}): {exc}")
        return False


# ----------------------------------------------------------------------
# orquestacion
# ----------------------------------------------------------------------

def _subscribers() -> List[User]:
    """usuarios con telegram o email configurado."""
    try:
        init_db()
        with get_session_factory()() as s:
            users = s.query(User).all()
            return [u for u in users if (u.telegram_chat_id or "").strip() or (u.email or "").strip()]
    except Exception as exc:
        logger.warning(f"no se pudieron leer suscriptores: {exc}")
        return []


def _already_sent(ticker: str, trade_date: str) -> bool:
    key = f"alert:sent:{ticker}:{trade_date}"
    if cache_get(key) is not None:
        return True
    cache_set(key, True, ttl=DEDUP_TTL)
    return False


def notify_new_strong_signals(signals: List[Dict[str, Any]]) -> Dict[str, int]:
    """notifica senales fuertes nuevas a los suscriptos. Idempotente."""
    if not alerts_configured():
        return {"sent": 0, "skipped": 0, "reason": "sin canales configurados"}
    subscribers = _subscribers()
    if not subscribers:
        return {"sent": 0, "skipped": 0, "reason": "sin suscriptores"}

    sent = skipped = 0
    for d in signals:
        sig = str(d.get("signal", "")).upper()
        if sig not in STRONG_SIGNALS:
            continue
        if not d.get("trading_allowed", True):
            continue
        ticker = str(d.get("ticker", ""))
        trade_date = str(d.get("as_of") or d.get("trade_date") or "")[:10]
        if not ticker or not trade_date or _already_sent(ticker, trade_date):
            skipped += 1
            continue

        text = format_message(d)
        mail = format_email(d)
        for u in subscribers:
            ok = False
            if (u.telegram_chat_id or "").strip():
                ok = send_telegram(u.telegram_chat_id.strip(), text) or ok
            if (u.email or "").strip():
                ok = send_email(u.email.strip(), mail["subject"], mail["html"]) or ok
            if ok:
                sent += 1
        logger.info(f"alerta {ticker} {sig}: enviada a {len(subscribers)} suscriptores")
    return {"sent": sent, "skipped": skipped}
