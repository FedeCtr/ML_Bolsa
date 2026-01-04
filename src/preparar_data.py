import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

class DataPreparerML:
    def __init__(self, tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']):
        self.tickers = tickers
        self.data_dir = "data/ml_ready/"
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs("data/raw/", exist_ok=True)
        os.makedirs("data/processed/", exist_ok=True)
    
    def descargar_datos_completos(self, years=5):
        """descargar datos historicos y procesarlos para ml"""
        print("Descargando datos históricos...")
        
        datos_completos = {}
        
        for ticker in self.tickers:
            try:
                print(f"  Descargando {ticker}...")
                stock = yf.Ticker(ticker)
                
                # bajar datos (5 años, diarios)
                df = stock.history(period=f"{years}y", interval="1d")
                
                if not df.empty:
                    # guardar raw data
                    df.to_csv(f"data/raw/{ticker}_raw.csv")
                    
                    # procesar para ml
                    df_procesado = self.procesar_para_ml(df, ticker)
                    datos_completos[ticker] = df_procesado
                    
                    print(f"  OK {ticker}: {len(df)} días descargados")
                else:
                    print(f"  ERROR {ticker}: Sin datos")
                    
            except Exception as e:
                print(f"  Error con {ticker}: {str(e)}")
        
        return datos_completos
    
    def procesar_para_ml(self, df, ticker):
        """pasar datos en features"""
        
        df_ml = df.copy()
        
        # features básicas
        # precios y retornos
        df_ml['retorno_1d'] = df_ml['Close'].pct_change()
        df_ml['retorno_5d'] = df_ml['Close'].pct_change(5)
        df_ml['retorno_20d'] = df_ml['Close'].pct_change(20)
        
        # volatilidad
        df_ml['volatilidad_5d'] = df_ml['retorno_1d'].rolling(5).std()
        df_ml['volatilidad_20d'] = df_ml['retorno_1d'].rolling(20).std()
        
        # indicadores tecnicos
        # medias moviles
        df_ml['sma_10'] = df_ml['Close'].rolling(10).mean()
        df_ml['sma_20'] = df_ml['Close'].rolling(20).mean()
        df_ml['sma_50'] = df_ml['Close'].rolling(50).mean()
        
        # distancia a medias
        df_ml['dist_sma_10'] = (df_ml['Close'] - df_ml['sma_10']) / df_ml['sma_10']
        df_ml['dist_sma_20'] = (df_ml['Close'] - df_ml['sma_20']) / df_ml['sma_20']
        
        # RSI simple
        cambios = df_ml['Close'].diff()
        ganancias = cambios.where(cambios > 0, 0)
        perdidas = -cambios.where(cambios < 0, 0)
        
        avg_ganancia = ganancias.rolling(14).mean()
        avg_perdida = perdidas.rolling(14).mean()
        rs = avg_ganancia / avg_perdida
        df_ml['rsi'] = 100 - (100 / (1 + rs))
        
        # volumen
        df_ml['volumen_sma_20'] = df_ml['Volume'].rolling(20).mean()
        df_ml['volumen_ratio'] = df_ml['Volume'] / df_ml['volumen_sma_20']
        
        # features temporales
        df_ml['dia_semana'] = df_ml.index.dayofweek
        df_ml['mes'] = df_ml.index.month
        df_ml['trimestre'] = df_ml.index.quarter
        
        # targets
        # target 1 dirección mañana (1=sube, 0=baja)
        df_ml['target_direccion'] = (df_ml['Close'].shift(-1) > df_ml['Close']).astype(int)
        
        # target 2 retorno mañana (regresion)
        df_ml['target_retorno'] = df_ml['Close'].shift(-1).pct_change()
        
        # target 3 retorno en 5 dias
        df_ml['target_retorno_5d'] = df_ml['Close'].shift(-5).pct_change()
        
        # eliminar NaN
        df_ml = df_ml.dropna()
        
        # guardar procesado
        df_ml.to_csv(f"data/processed/{ticker}_processed.csv")
        
        return df_ml
    
    def crear_dataset_unificado(self):
        """dataset para todas las acciones"""
        print("\nCreando dataset...")
        
        todos_datos = []
        
        for ticker in self.tickers:
            archivo = f"data/processed/{ticker}_processed.csv"
            
            if os.path.exists(archivo):
                df = pd.read_csv(archivo, index_col=0, parse_dates=True)
                df['ticker'] = ticker  # Identificador
                todos_datos.append(df)
                print(f"  OK Añadido: {ticker} ({len(df)} filas)")
        
        if todos_datos:
            dataset_unificado = pd.concat(todos_datos, axis=0)
            
            # guardar dataset completo
            fecha = datetime.now().strftime("%Y%m%d")
            dataset_unificado.to_csv(f"{self.data_dir}dataset_completo_{fecha}.csv")
            dataset_unificado.to_csv(f"{self.data_dir}dataset_completo_latest.csv")
            
            print(f"\nDataset creado:")
            print(f"   Filas: {len(dataset_unificado):,}")
            print(f"   Columnas: {len(dataset_unificado.columns)}")
            print(f"   Guardado en: {self.data_dir}")
            
            return dataset_unificado
        else:
            print("Error No hay datos procesados")
            return None

# ejecutar para preparar datos
if __name__ == "__main__":
    preparador = DataPreparerML()
    
    # descargar y procesar datos
    datos = preparador.descargar_datos_completos(years=3)  # empieza con 3 años
    
    # crear dataset
    dataset = preparador.crear_dataset_unificado()
    
    if dataset is not None:
        print("\nfeature disponibles para ML:")
        for i, col in enumerate(dataset.columns[:20], 1):  # primeras 20
            print(f"   {i:2d}. {col}")
        
        if len(dataset.columns) > 20:
            print(f"   ... y {len(dataset.columns)-20} más")
        print(dataset[['Close', 'retorno_1d', 'rsi', 'target_direccion']].head())