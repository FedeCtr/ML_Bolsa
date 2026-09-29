"""
Sesion y engine de base de datos (Sprint 1).

DATABASE_URL decide el backend:
  - sin variable: SQLite en data/saas.db (dev/CI: cero dependencias externas)
  - postgresql+psycopg://user:pass@host:5432/db  (produccion, docker-compose)
"""
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ..utils.config import Config
from ..utils.logger import get_logger
from .models import Base

logger = get_logger(__name__)


def get_database_url() -> str:
    url = os.environ.get('DATABASE_URL', '').strip()
    if url:
        # heroku/render usan postgres:// ; SQLAlchemy necesita el driver
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql+psycopg://', 1)
        elif url.startswith('postgresql://'):
            url = url.replace('postgresql://', 'postgresql+psycopg://', 1)
        return url
    # dev/CI: SQLite junto al resto de datos
    data_dir = Path(str(Config().DATA_DIR))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'saas.db').as_posix()}"


def create_db_engine(database_url: str | None = None):
    url = database_url or get_database_url()
    if url.startswith('sqlite'):
        engine = create_engine(url, connect_args={'check_same_thread': False})
    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    logger.info(f"DB engine: {url.split('@')[-1] if '@' in url else url}")
    return engine


_engine = None
_SessionLocal = None


def get_engine():
    """engine singleton del proceso (tests usan set_engine)."""
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_db_engine()
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session_factory():
    get_engine()
    return _SessionLocal


def set_engine(engine) -> None:
    """permite a los tests inyectar un engine propio (SQLite temporal)."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """crea las tablas si no existen (Alembic gestiona el esquema en prod)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("DB inicializada (create_all)")


def healthcheck() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text('SELECT 1'))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error(f"DB healthcheck fallo: {e}")
        return False
