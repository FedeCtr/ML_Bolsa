"""
modulo para preparar datos para machine learning
"""
import pandas as pd
import os
from datetime import datetime
from typing import List, Optional

from .collector import DataCollector
from .processor import TechnicalProcessor
from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class MLDataPreparer:
    """prepara datasets listos para entrenar modelos ML"""
    
    def __init__(self, tickers: List[str] = None):
        self.config = Config()
        self.tickers = tickers or self.config.default_tickers
        self.collector = DataCollector(self.tickers)
        self.processor = TechnicalProcessor()
        
        # crear directorios
        os.makedirs(self.config.processed_data_dir, exist_ok=True)
        os.makedirs(self.config.ml_ready_dir, exist_ok=True)
    
    def add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """añade variables target para ML"""
        # target direccion (clasificacion)
        df['target_direccion'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # target retorno (regresion)
        df['target_retorno'] = df['Close'].shift(-1).pct_change()
        
        # target retorno 5 dias
        df['target_retorno_5d'] = df['Close'].shift(-5).pct_change()
        
        return df
    
    def prepare_ticker(self, ticker: str, years: int = 5) -> Optional[pd.DataFrame]:
        """prepara datos de un ticker para ML"""
        logger.info(f"preparando {ticker}")
        
        # descargar datos
        df = self.collector.download_ticker(ticker, years)
        if df is None:
            return None
        
        # calcular indicadores
        df = self.processor.process_all_indicators(df)
        
        # añadir targets
        df = self.add_targets(df)
        
        # eliminar nans
        df = df.dropna()
        
        # guardar procesado
        filepath = os.path.join(self.config.processed_data_dir, f"{ticker}_processed.csv")
        df.to_csv(filepath)
        logger.info(f"guardado {ticker}: {len(df)} registros")
        
        return df
    
    def prepare_all(self, years: int = 3) -> dict:
        """prepara datos de todos los tickers"""
        logger.info("preparando datos para ML")
        datos = {}
        
        for ticker in self.tickers:
            df = self.prepare_ticker(ticker, years)
            if df is not None:
                datos[ticker] = df
        
        logger.info(f"preparacion completada: {len(datos)} tickers")
        return datos
    
    def create_unified_dataset(self) -> Optional[pd.DataFrame]:
        """crea dataset unificado con todos los tickers"""
        logger.info("creando dataset unificado")
        
        todos_datos = []
        
        for ticker in self.tickers:
            filepath = os.path.join(self.config.processed_data_dir, f"{ticker}_processed.csv")
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                df['ticker'] = ticker
                todos_datos.append(df)
                logger.info(f"añadido {ticker}: {len(df)} filas")
        
        if not todos_datos:
            logger.error("no hay datos procesados")
            return None
        
        # concatenar todos
        dataset = pd.concat(todos_datos, axis=0)
        
        # guardar
        fecha = datetime.now().strftime("%Y%m%d")
        path_dated = os.path.join(self.config.ml_ready_dir, f"dataset_completo_{fecha}.csv")
        path_latest = os.path.join(self.config.ml_ready_dir, "dataset_completo_latest.csv")
        
        dataset.to_csv(path_dated)
        dataset.to_csv(path_latest)
        
        logger.info(f"dataset creado: {len(dataset)} filas, {len(dataset.columns)} columnas")
        return dataset
