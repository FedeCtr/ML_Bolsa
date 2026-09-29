"""
Watchlists personalizadas persistidas en la base del SaaS (SQLAlchemy).

Migracion desde el JSON legado (data/watchlists.json): si existe y la tabla
esta vacia, se importa automaticamente al primer uso.

Firmas identicas a la version JSON: list/get/create/delete/add_tickers/
remove_tickers. El constructor acepta `path` por compatibilidad (tests): si
se pasa, usa un SQLite aislado en esa ruta.
"""
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import sessionmaker

from ..db.models import Base, Watchlist
from ..db.session import get_session_factory, init_db
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)

LEGACY_JSON_NAME = 'watchlists.json'


class WatchlistStore:
    """almacen de watchlists por usuario (single-user hasta Clerk, Sprint 6)"""

    def __init__(self, path: Optional[str] = None):
        if path is not None:
            # modo aislado (tests): SQLite en la ruta dada
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{path}",
                                   connect_args={'check_same_thread': False})
            Base.metadata.create_all(engine)
            self._Session = sessionmaker(bind=engine, expire_on_commit=False)
            self._legacy_path = None
        else:
            init_db()
            self._Session = get_session_factory()
            self._legacy_path = os.path.join(str(Config().DATA_DIR), LEGACY_JSON_NAME)
            if os.path.exists(self._legacy_path):
                self._import_legacy_if_empty()

    # ------------------------------------------------------------------
    # infraestructura
    # ------------------------------------------------------------------

    def _import_legacy_if_empty(self) -> None:
        try:
            with self._Session() as s:
                if s.query(Watchlist).first() is not None:
                    return
                with open(self._legacy_path, encoding='utf-8') as f:
                    data = json.load(f)
                for name, w in (data or {}).items():
                    s.add(Watchlist(name=name, user_id=0,
                                    tickers=w.get('tickers', []),
                                    created_at=w.get('created_at') or
                                    datetime.now(timezone.utc).isoformat()))
                s.commit()
            logger.info(f"watchlists: importadas {len(data)} del legado {self._legacy_path}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"watchlists: importacion legado omitida ({e})")

    def _row_to_dict(self, w: Watchlist) -> Dict:
        return {'name': w.name, 'tickers': list(w.tickers or []),
                'created_at': w.created_at}

    # ------------------------------------------------------------------
    # CRUD (firmas identicas a la version JSON)
    # ------------------------------------------------------------------

    def list(self) -> List[Dict]:
        with self._Session() as s:
            rows = s.query(Watchlist).order_by(Watchlist.name).all()
            return [self._row_to_dict(w) for w in rows]

    def get(self, name: str) -> Optional[Dict]:
        with self._Session() as s:
            w = s.query(Watchlist).filter_by(name=name).one_or_none()
            return self._row_to_dict(w) if w else None

    def create(self, name: str, tickers: Optional[List[str]] = None) -> Dict:
        with self._Session() as s:
            if s.query(Watchlist).filter_by(name=name).one_or_none() is not None:
                raise ValueError(f"la watchlist '{name}' ya existe")
            w = Watchlist(
                name=name, user_id=0,
                tickers=sorted({t.upper().strip() for t in (tickers or [])}),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            s.add(w)
            s.commit()
            return self._row_to_dict(w)

    def delete(self, name: str) -> bool:
        with self._Session() as s:
            w = s.query(Watchlist).filter_by(name=name).one_or_none()
            if w is None:
                return False
            s.delete(w)
            s.commit()
            return True

    def add_tickers(self, name: str, tickers: List[str]) -> Dict:
        with self._Session() as s:
            w = s.query(Watchlist).filter_by(name=name).one_or_none()
            if w is None:
                raise ValueError(f"la watchlist '{name}' no existe")
            current = set(w.tickers or [])
            for t in tickers:
                current.add(t.upper().strip())
            w.tickers = sorted(current)
            s.commit()
            return self._row_to_dict(w)

    def remove_tickers(self, name: str, tickers: List[str]) -> Dict:
        with self._Session() as s:
            w = s.query(Watchlist).filter_by(name=name).one_or_none()
            if w is None:
                raise ValueError(f"la watchlist '{name}' no existe")
            remove = {t.upper().strip() for t in tickers}
            w.tickers = [t for t in (w.tickers or []) if t not in remove]
            s.commit()
            return self._row_to_dict(w)
