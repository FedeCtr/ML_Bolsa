"""
modulo para descargar datos de yahoo finance
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Optional
import os

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class DataCollector:
    """descarga datos historicos de yahoo finance"""
    
    def __init__(self, tickers: List[str] = None):
        self.config = Config()
        self.tickers = tickers or self.config.default_tickers
        self.raw_dir = self.config.raw_data_dir
        os.makedirs(self.raw_dir, exist_ok=True)
    
    def download_ticker(self, ticker: str, period: str = '5y') -> Optional[pd.DataFrame]:
        """descarga datos de un ticker especifico"""
        try:
            logger.info(f"descargando {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d")
            
            if df.empty:
                logger.warning(f"sin datos para {ticker}")
                return None
            
            # guardar raw data SOLO si el historial nuevo es mas largo que el
            # cacheado: los escaneos del dashboard usan periodos cortos (3mo)
            # y no deben degradar el cache de entrenamiento (5y).
            filepath = os.path.join(self.raw_dir, f"{ticker}_raw.csv")
            try:
                existing_rows = sum(1 for _ in open(filepath, encoding='utf-8', errors='ignore')) - 1
            except OSError:
                existing_rows = 0
            if len(df) >= existing_rows:
                df.to_csv(filepath)
                logger.info(f"guardado {ticker}: {len(df)} dias")
            else:
                logger.info(f"cache {ticker} conservado ({existing_rows} dias > {len(df)})")
            
            return df
            
        except Exception as e:
            logger.error(f"error descargando {ticker}: {e}")
            return None

    def download_batch(self, tickers: list, period: str = '1y', max_workers: int = 8) -> dict:
        """descarga N tickers en paralelo (para escaneos del universo)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out: dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.download_ticker, t, period): t for t in tickers}
            for fut in as_completed(futures):
                t = futures[fut]
                df = fut.result()
                if df is not None:
                    out[t] = df
        logger.info(f"batch: {len(out)}/{len(tickers)} tickers descargados")
        return out
    
    def download_all(self, years: int = 5) -> dict:
        """descarga datos de todos los tickers"""
        logger.info("iniciando descarga de datos")
        datos = {}
        
        for ticker in self.tickers:
            df = self.download_ticker(ticker, years)
            if df is not None:
                datos[ticker] = df
        
        logger.info(f"descarga completada: {len(datos)}/{len(self.tickers)} tickers")
        return datos
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """obtiene el ultimo precio de un ticker"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d")
            if not df.empty:
                return float(df['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"error obteniendo precio de {ticker}: {e}")
            return None
