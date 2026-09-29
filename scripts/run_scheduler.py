"""Runner del scheduler de escaneos (Sprint 2).

Proceso independiente que corre el ciclo cada 5 min. Para el compose simple
tambien puede embeberse en el API con SCHEDULER_EMBEDDED=1.

Uso:
    python scripts/run_scheduler.py                # universo sp500
    SCHEDULER_UNIVERSE=ndx100 python scripts/run_scheduler.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jobs.scheduler import start_scheduler
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    universe = os.environ.get("SCHEDULER_UNIVERSE", "sp500")
    period = os.environ.get("SCHEDULER_PERIOD", "3mo")
    logger.info(f"arrancando scheduler (universo={universe}, periodo={period})")
    sched = start_scheduler(universe=universe, period=period)
    try:
        import time

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("scheduler detenido")
        sched.shutdown(wait=False)
