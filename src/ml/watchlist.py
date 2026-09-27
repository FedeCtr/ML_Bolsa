"""
Watchlists personalizadas persistidas en JSON (data/watchlists.json).

Cada lista: {name, tickers[], created_at}. Operaciones CRUD basicas + util
para filtrar el screener por listas del usuario (spec 3).
"""
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class WatchlistStore:
    """almacen simple de watchlists por usuario (single-user SaaS inicial)"""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(str(Config().DATA_DIR), 'watchlists.json')
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._data: Dict[str, Dict] = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"watchlists: archivo corrupto ({e}); iniciando vacio")
        return {}

    def _save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def list(self) -> List[Dict]:
        return [{'name': n, 'tickers': w['tickers'], 'created_at': w['created_at']}
                for n, w in sorted(self._data.items())]

    def get(self, name: str) -> Optional[Dict]:
        w = self._data.get(name)
        return {'name': name, 'tickers': w['tickers'], 'created_at': w['created_at']} if w else None

    def create(self, name: str, tickers: Optional[List[str]] = None) -> Dict:
        if name in self._data:
            raise ValueError(f"la watchlist '{name}' ya existe")
        self._data[name] = {
            'tickers': sorted({t.upper().strip() for t in (tickers or [])}),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return self.get(name)

    def delete(self, name: str) -> bool:
        if name not in self._data:
            return False
        del self._data[name]
        self._save()
        return True

    def add_tickers(self, name: str, tickers: List[str]) -> Dict:
        if name not in self._data:
            raise ValueError(f"la watchlist '{name}' no existe")
        current = set(self._data[name]['tickers'])
        for t in tickers:
            current.add(t.upper().strip())
        self._data[name]['tickers'] = sorted(current)
        self._save()
        return self.get(name)

    def remove_tickers(self, name: str, tickers: List[str]) -> Dict:
        if name not in self._data:
            raise ValueError(f"la watchlist '{name}' no existe")
        remove = {t.upper().strip() for t in tickers}
        self._data[name]['tickers'] = [t for t in self._data[name]['tickers'] if t not in remove]
        self._save()
        return self.get(name)
