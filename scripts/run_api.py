"""
script para iniciar la API
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.app import run_app


if __name__ == '__main__':
    run_app()
