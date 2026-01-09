"""
modulo para calcular indicadores tecnicos avanzados
"""
import pandas as pd
import numpy as np
from typing import Optional
import os
import ta

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class TechnicalProcessor:
    """calcula 50+ indicadores tecnicos avanzados para prediccion de bolsa"""
    
    def __init__(self):
        self.config = Config()
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """retornos multiples periodos"""
        df['retorno_1d'] = df['Close'].pct_change()
        df['retorno_3d'] = df['Close'].pct_change(3)
        df['retorno_5d'] = df['Close'].pct_change(5)
        df['retorno_10d'] = df['Close'].pct_change(10)
        df['retorno_20d'] = df['Close'].pct_change(20)
        return df
    
    def calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """volatilidad multiples periodos"""
        df['volatilidad_5d'] = df['retorno_1d'].rolling(5).std()
        df['volatilidad_10d'] = df['retorno_1d'].rolling(10).std()
        df['volatilidad_20d'] = df['retorno_1d'].rolling(20).std()
        df['volatilidad_60d'] = df['retorno_1d'].rolling(60).std()
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """medias moviles simples y exponenciales"""
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # distancias
        df['dist_sma_10'] = (df['Close'] - df['sma_10']) / df['sma_10']
        df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        df['dist_sma_50'] = (df['Close'] - df['sma_50']) / df['sma_50']
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD - Moving Average Convergence Divergence"""
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bandas de Bollinger"""
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_pct'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        return df
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR - Average True Range"""
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['Close']
        return df
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX - Average Directional Index"""
        df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        return df
    
    def calculate_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Williams %R"""
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r()
        return df
    
    def calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI - Commodity Channel Index"""
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()
        return df
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI - Relative Strength Index"""
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        return df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV - On Balance Volume"""
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        return df
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """VWAP - Volume Weighted Average Price"""
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        df['dist_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """indicadores de volumen"""
        df['volumen_sma_20'] = df['Volume'].rolling(20).mean()
        df['volumen_ratio'] = df['Volume'] / df['volumen_sma_20']
        df['volume_roc'] = df['Volume'].pct_change(5)
        return df
    
    def calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """momentum multiples periodos"""
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_20'] = df['Close'] - df['Close'].shift(20)
        return df
    
    def calculate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """precios pasados - lag features"""
        df['close_lag_1'] = df['Close'].shift(1)
        df['close_lag_2'] = df['Close'].shift(2)
        df['close_lag_3'] = df['Close'].shift(3)
        df['close_lag_5'] = df['Close'].shift(5)
        df['close_lag_10'] = df['Close'].shift(10)
        return df
    
    def calculate_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """estadisticas rolling"""
        # max y min
        df['high_20d'] = df['High'].rolling(20).max()
        df['low_20d'] = df['Low'].rolling(20).min()
        df['range_20d'] = (df['high_20d'] - df['low_20d']) / df['Close']
        
        # mediana
        df['median_20d'] = df['Close'].rolling(20).median()
        
        return df
    
    def calculate_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """patrones de precio"""
        df['body'] = abs(df['Close'] - df['Open'])
        df['upper_shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
        df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['total_range'] = df['High'] - df['Low']
        df['body_pct'] = df['body'] / df['total_range']
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """features temporales"""
        df['dia_semana'] = df.index.dayofweek
        df['mes'] = df.index.month
        df['trimestre'] = df.index.quarter
        df['dia_mes'] = df.index.day
        df['es_fin_mes'] = (df.index.day > 25).astype(int)
        df['es_inicio_mes'] = (df.index.day <= 5).astype(int)
        return df
    
    def process_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """calcula TODOS los indicadores - 50+ features"""
        logger.info("calculando 50+ indicadores tecnicos avanzados")
        
        df = df.copy()
        
        # basicos
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_moving_averages(df)
        
        # indicadores tecnicos avanzados
        df = self.calculate_macd(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_atr(df)
        df = self.calculate_adx(df)
        df = self.calculate_stochastic(df)
        df = self.calculate_williams_r(df)
        df = self.calculate_cci(df)
        df = self.calculate_rsi(df)
        df = self.calculate_obv(df)
        df = self.calculate_vwap(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_momentum(df)
        
        # lag features y rolling stats
        df = self.calculate_lag_features(df)
        df = self.calculate_rolling_stats(df)
        df = self.calculate_price_patterns(df)
        
        # temporales
        df = self.add_temporal_features(df)
        
        logger.info(f"indicadores calculados - total: {df.shape[1]} columnas")
        return df
