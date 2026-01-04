"""
modulo para calcular indicadores tecnicos
"""
import pandas as pd
import numpy as np
from typing import Optional
import os

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class TechnicalProcessor:
    """calcula indicadores tecnicos sobre datos historicos"""
    
    def __init__(self):
        self.config = Config()
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """calcula retornos en diferentes periodos"""
        df['retorno_1d'] = df['Close'].pct_change()
        df['retorno_5d'] = df['Close'].pct_change(5)
        df['retorno_20d'] = df['Close'].pct_change(20)
        return df
    
    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """calcula volatilidad"""
        df['volatilidad_5d'] = df['retorno_1d'].rolling(5).std()
        df['volatilidad_20d'] = df['retorno_1d'].rolling(20).std()
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """calcula medias moviles"""
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        # distancia a medias
        df['dist_sma_10'] = (df['Close'] - df['sma_10']) / df['sma_10']
        df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """calcula relative strength index"""
        cambios = df['Close'].diff()
        ganancias = cambios.where(cambios > 0, 0)
        perdidas = -cambios.where(cambios < 0, 0)
        
        avg_ganancia = ganancias.rolling(period).mean()
        avg_perdida = perdidas.rolling(period).mean()
        
        rs = avg_ganancia / avg_perdida
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """calcula indicadores de volumen"""
        df['volumen_sma_20'] = df['Volume'].rolling(20).mean()
        df['volumen_ratio'] = df['Volume'] / df['volumen_sma_20']
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """añade features temporales"""
        df['dia_semana'] = df.index.dayofweek
        df['mes'] = df.index.month
        df['trimestre'] = df.index.quarter
        return df
    
    def process_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """aplica todos los indicadores tecnicos"""
        logger.info("calculando indicadores tecnicos")
        
        df = df.copy()
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_rsi(df)
        df = self.calculate_volume_indicators(df)
        df = self.add_temporal_features(df)
        
        logger.info("indicadores calculados")
        return df
