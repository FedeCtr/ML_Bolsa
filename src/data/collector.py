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
    
    def download_ticker(self, ticker: str, years: int = 5) -> Optional[pd.DataFrame]:
        """descarga datos de un ticker especifico"""
        try:
            logger.info(f"descargando {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{years}y", interval="1d")
            
            if df.empty:
                logger.warning(f"sin datos para {ticker}")
                return None
            
            # guardar raw data
            filepath = os.path.join(self.raw_dir, f"{ticker}_raw.csv")
            df.to_csv(filepath)
            logger.info(f"guardado {ticker}: {len(df)} dias")
            
            return df
            
        except Exception as e:
            logger.error(f"error descargando {ticker}: {e}")
            return None
    
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
