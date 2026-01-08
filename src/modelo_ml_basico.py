import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import yfinance as yf
from preparar_data import DataPreparerML

class ModeloMLBasico:
    def __init__(self):
        self.model = None
        self.features = None
        self.model_dir = "models/"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def cargar_datos(self, ticker='AAPL', use_all=False):
        """datos para entrenamiento"""
        if use_all:
            # usar dataset unificado
            data_path = "data/ml_ready/dataset_completo_latest.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, parse_dates=True)
                print(f"dataset completo cargado: {len(df):,} filas")
                return df
            else:
                print("no existe dataset unificado. ejecutar preparar_data.py primero")
                return None
        else:
            # usar solo 1 ticker
            data_path = f"data/processed/{ticker}_processed.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                print(f"{ticker} cargado: {len(df)} dias")
                return df
            else:
                print(f"no existe {data_path}")
                return None
    
    def preparar_features(self, df, ticker_specific=None):
        """selecciona features para el modelo"""
        
        # features para clasificacion (sube/baja)
        feature_cols = [
            # retornos pasados
            'retorno_1d', 'retorno_5d', 'retorno_20d',
            
            # volatilidad
            'volatilidad_5d', 'volatilidad_20d',
            
            # indicadores tecnicos
            'dist_sma_10', 'dist_sma_20',
            'rsi',
            
            # volumen
            'volumen_ratio',
            
            # temporales
            'dia_semana', 'mes'
        ]
        
        # filtrar solo las que existen
        available_features = [f for f in feature_cols if f in df.columns]
        
        # target
        target_col = 'target_direccion'
        
        if target_col not in df.columns:
            print(f"no existe {target_col} en los datos")
            return None, None
        
        # separar features y target
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        print(f"features seleccionadas: {len(available_features)}")
        print(f"   muestras: {len(X)}")
        print(f"   target: {y.value_counts().to_dict()}")
        
        self.features = available_features
        return X, y
    
    def entrenar_modelo(self, X, y, test_size=0.2):
        """entrena un modelo simple"""
        print("\nentrenando modelo...")
        
        # importante: split temporal, no aleatorio
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"   entrenamiento: {len(X_train)} muestras")
        print(f"   prueba: {len(X_test)} muestras")
        
        # crear y entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # evaluar
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nresultados:")
        print(f"   precision: {accuracy:.2%}")
        print(f"   aciertos: {sum(y_pred == y_test)}/{len(y_test)}")
        
        # reporte detallado
        print("\nreporte de clasificacion:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['BAJA', 'SUBE']))
        
        # feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nfeatures mas importantes:")
            print(importance_df.head(10).to_string(index=False))
        
        return accuracy
    
    def guardar_modelo(self, nombre="modelo_basico"):
        """guarda el modelo entrenado"""
        if self.model is not None:
            joblib.dump(self.model, f"{self.model_dir}{nombre}.pkl")
            joblib.dump(self.features, f"{self.model_dir}{nombre}_features.pkl")
            print(f"modelo guardado en: {self.model_dir}{nombre}.pkl")
    
    def predecir_actual(self, ticker='AAPL'):
        """hace prediccion con datos actuales"""
        preparador = DataPreparerML(tickers=[ticker])
        stock = yf.Ticker(ticker)
        df_raw = stock.history(period="3mo", interval="1d")
        
        if df_raw.empty:
            print(f"no hay datos para {ticker}")
            return None
        
        df_procesado = preparador.procesar_para_ml(df_raw, ticker)
        
        if self.model is None or self.features is None:
            print("modelo no entrenado. entrena primero.")
            return None
        
        X_actual = df_procesado[self.features].iloc[-1:].copy()
        
        prediccion = self.model.predict(X_actual)[0]
        probabilidad = self.model.predict_proba(X_actual)[0]
        
        resultado = {
            'ticker': ticker,
            'prediccion': 'SUBE' if prediccion == 1 else 'BAJA',
            'probabilidad_subida': float(probabilidad[1]),
            'probabilidad_bajada': float(probabilidad[0]),
            'confianza': float(max(probabilidad)),
            'fecha': str(df_procesado.index[-1]),
            'precio_actual': float(df_procesado['Close'].iloc[-1])
        }
        
        print(f"\nprediccion para {ticker}:")
        print(f"   senal: {resultado['prediccion']}")
        print(f"   probabilidad: {resultado['probabilidad_subida']:.1%}")
        print(f"   confianza: {resultado['confianza']:.1%}")
        print(f"   precio actual: ${resultado['precio_actual']:.2f}")
        
        return resultado

# ejecucion paso a paso
if __name__ == "__main__":
    print("=" * 60)
    print("entrenamiento de modelo ml basico")
    print("=" * 60)
    
    # inicializar
    ml = ModeloMLBasico()
    
    # cargar datos (empezar con Apple)
    print("\ncargando datos...")
    df = ml.cargar_datos(ticker='AAPL', use_all=False)
    
    if df is not None:
        # preparar features
        print("\npreparando features...")
        X, y = ml.preparar_features(df)
        
        if X is not None and y is not None:
            # entrenar modelo
            print("\nentrenando...")
            accuracy = ml.entrenar_modelo(X, y, test_size=0.2)
            
            # guardar modelo
            print("\nguardando modelo...")
            ml.guardar_modelo(nombre="modelo_aapl_basico")
            
            # probar prediccion actual
            print("\nprobando prediccion...")
            ml.predecir_actual('AAPL')
            
            print("\n" + "=" * 60)
            print("proceso completado")
            print("   modelo listo para usar en la api")
            print("=" * 60)