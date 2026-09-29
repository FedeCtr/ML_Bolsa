"""Runner del API unificada FastAPI (Sprint 1).

El codigo vive en src.api.fastapi_app para que Docker pueda arrancarlo con
gunicorn/uvicorn sin depender de la carpeta scripts/.

Uso:
    python scripts/serve_fastapi.py            # uvicorn directo
    uvicorn src.api.fastapi_app:app --port 8010
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

from src.api.fastapi_app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
