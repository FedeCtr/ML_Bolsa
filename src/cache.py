"""Capa de cache para lecturas calientes del API.

Estrategia:
- Si ``REDIS_URL`` apunta a un Redis alcanzable, se usa Redis (compartido
  entre procesos/instancias, ideal para prod con docker-compose).
- Si no hay Redis o falla la conexion, se degrada a un cache en memoria con
  TTL (suficiente para dev y para un solo proceso Uvicorn/Flask).

La serializacion es JSON para que cualquier valor numpy/pandas de las rutas
pase limpio por Redis y por el fallback (``default=str`` como salvavidas).

API publica:
    cache_get(key) -> Any | None
    cache_set(key, value, ttl=None)
    cache_delete(*keys)
    cached(key, ttl)  -> decorador
    cache_health()    -> dict para /health
    reset_cache()     -> tests
"""
from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

KEY_PREFIX = "mlb:"
DEFAULT_TTL_SECONDS = 3600

try:  # redis es dependencia instalada; se importa perezoso por si falta en dev
    import redis as _redis_lib
except ImportError:  # pragma: no cover
    _redis_lib = None


class MemoryCache:
    """Cache en memoria con TTL monotonic; fallback sin Redis."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, Optional[float]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expires_at = item
            if expires_at is not None and time.monotonic() > expires_at:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires_at = time.monotonic() + ttl if ttl else None
        with self._lock:
            self._data[key] = (value, expires_at)

    def delete(self, *keys: str) -> None:
        with self._lock:
            for key in keys:
                self._data.pop(key, None)

    @property
    def backend(self) -> str:
        return "memory"


class RedisCache:
    """Wrapper JSON sobre redis-py; nunca lanza (degrada a miss)."""

    def __init__(self, url: str) -> None:
        if _redis_lib is None:
            raise RuntimeError("paquete redis no instalado")
        self._client = _redis_lib.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
        )

    def get(self, key: str) -> Any:
        try:
            raw = self._client.get(key)
        except Exception as exc:  # redis caido -> miss silencioso
            logger.warning("cache redis get fallo (%s): %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        try:
            raw = json.dumps(value, default=str)
            self._client.set(key, raw, ex=int(ttl) if ttl else None)
        except Exception as exc:
            logger.warning("cache redis set fallo (%s): %s", key, exc)

    def delete(self, *keys: str) -> None:
        try:
            if keys:
                self._client.delete(*keys)
        except Exception as exc:
            logger.warning("cache redis delete fallo: %s", exc)

    @property
    def backend(self) -> str:
        return "redis"


_cache: Any = None
_cache_lock = threading.Lock()


def _build_cache() -> Any:
    url = os.environ.get("REDIS_URL", "").strip()
    if url and _redis_lib is not None:
        try:
            cache = RedisCache(url)
            cache._client.ping()  # fail fast si el Redis no responde
            return cache
        except Exception as exc:
            logger.warning("Redis no disponible (%s); fallback en memoria", exc)
    return MemoryCache()


def get_cache() -> Any:
    """Singleton del backend activo (Redis o memoria)."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = _build_cache()
    return _cache


def reset_cache() -> None:
    """Reinicia el singleton (para tests que cambian REDIS_URL)."""
    global _cache
    with _cache_lock:
        _cache = None


def _full_key(key: str) -> str:
    return key if key.startswith(KEY_PREFIX) else f"{KEY_PREFIX}{key}"


def cache_get(key: str) -> Any:
    return get_cache().get(_full_key(key))


def cache_set(key: str, value: Any, ttl: Optional[float] = DEFAULT_TTL_SECONDS) -> None:
    get_cache().set(_full_key(key), value, ttl)


def cache_delete(*keys: str) -> None:
    get_cache().delete(*[_full_key(k) for k in keys])


def cached(key: str, ttl: Optional[float] = DEFAULT_TTL_SECONDS) -> Callable:
    """Decorador: responde desde cache si existe; si no, computa y guarda.

    Con lock por-clave para evitar stampede de la misma lectura caliente.
    """

    def decorator(func: Callable) -> Callable:
        locks: dict[str, threading.Lock] = {}
        locks_guard = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            suffix = f":{args}:{sorted(kwargs.items())}" if (args or kwargs) else ""
            full = _full_key(key) + suffix
            hit = cache_get(full)
            if hit is not None:
                return hit
            with locks_guard:
                lock = locks.setdefault(full, threading.Lock())
            with lock:
                hit = cache_get(full)  # double-check tras adquirir el lock
                if hit is not None:
                    return hit
                value = func(*args, **kwargs)
                cache_set(full, value, ttl)
                return value

        return wrapper

    return decorator


def cache_health() -> dict[str, Any]:
    """Estado del cache para el endpoint de health."""
    cache = get_cache()
    info: dict[str, Any] = {"backend": cache.backend}
    if cache.backend == "redis":
        try:
            ping_start = time.perf_counter()
            cache._client.ping()
            info["ping_ok"] = True
            info["ping_ms"] = round((time.perf_counter() - ping_start) * 1000, 1)
        except Exception as exc:
            info["ping_ok"] = False
            info["error"] = str(exc)
    return info
