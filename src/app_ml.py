# src/app_ml.py
from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import json

app = Flask(__name__)

# cargar modelo entrenado si existe
MODEL_PATH = "models/modelo_aapl_basico.pkl"
FEATURES_PATH = "models/modelo_aapl_basico_features.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    modelo = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    print(f"modelo cargado: {len(features)} features")
else:
    modelo = None
    features = None
    print("modelo no encontrado. entrena primero con modelo_ml_basico.py")

def preparar_datos_para_modelo(ticker, dias_historia=60):
    """prepara datos actuales para el modelo"""
    try:
        # descarga datos recientes
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{dias_historia}d", interval="1d")
        
        if df.empty:
            return None
        
        # calcula features igual que en entrenamiento
        # features basicas
        df['retorno_1d'] = df['Close'].pct_change()
        df['retorno_5d'] = df['Close'].pct_change(5)
        df['retorno_20d'] = df['Close'].pct_change(20)
        
        # volatilidad
        df['volatilidad_5d'] = df['retorno_1d'].rolling(5).std()
        df['volatilidad_20d'] = df['retorno_1d'].rolling(20).std()
        
        # medias moviles
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['dist_sma_10'] = (df['Close'] - df['sma_10']) / df['sma_10']
        df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        
        # rsi simple
        cambios = df['Close'].diff()
        ganancias = cambios.where(cambios > 0, 0)
        perdidas = -cambios.where(cambios < 0, 0)
        avg_ganancia = ganancias.rolling(14).mean()
        avg_perdida = perdidas.rolling(14).mean()
        rs = avg_ganancia / avg_perdida
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # volumen
        df['volumen_sma_20'] = df['Volume'].rolling(20).mean()
        df['volumen_ratio'] = df['Volume'] / df['volumen_sma_20']
        
        # temporales
        df['dia_semana'] = df.index.dayofweek
        df['mes'] = df.index.month
        
        # tomar ultima fila
        df = df.dropna()
        
        if len(df) == 0:
            return None
        
        # features que usa el modelo
        if features:
            # verificar features necesarias
            features_disponibles = [f for f in features if f in df.columns]
            if len(features_disponibles) < len(features):
                print(f"faltan features: {set(features) - set(features_disponibles)}")
            
            X_actual = df[features_disponibles].iloc[-1:]
            return X_actual
        else:
            return df.iloc[-1:]
            
    except Exception as e:
        print(f"error preparando datos: {e}")
        return None

@app.route('/')
def home():
    return render_template('dashboard_ml.html')

@app.route('/api/predict/<ticker>')
def predict_ticker(ticker):
    """endpoint de prediccion con ml"""
    if modelo is None:
        return jsonify({
            'error': 'Modelo no entrenado',
            'message': 'Ejecuta primero modelo_ml_basico.py'
        }), 400
    
    # preparar datos
    X_actual = preparar_datos_para_modelo(ticker)
    
    if X_actual is None or len(X_actual) == 0:
        return jsonify({'error': 'No se pudieron obtener datos'}), 400
    
    # hacer prediccion
    try:
        prediccion = modelo.predict(X_actual)[0]
        probabilidades = modelo.predict_proba(X_actual)[0]
        
        # obtener precio actual
        stock = yf.Ticker(ticker)
        precio_actual = stock.history(period="1d")['Close'].iloc[-1]
        
        resultado = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'precio_actual': float(precio_actual),
            'prediccion': 'SUBE' if prediccion == 1 else 'BAJA',
            'probabilidad_subida': float(probabilidades[1]),
            'probabilidad_bajada': float(probabilidades[0]),
            'confianza': float(max(probabilidades)),
            'features_utilizadas': len(X_actual.columns),
            'recomendacion': 'COMPRA' if prediccion == 1 and probabilidades[1] > 0.6 else 
                            'VENTA' if prediccion == 0 and probabilidades[0] > 0.6 else 
                            'MANTENER'
        }
        
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analisis/<ticker>')
def analisis_completo(ticker):
    """analisis completo con ml y tecnico"""
    # prediccion ml
    prediccion_ml = predict_ticker(ticker)
    
    if isinstance(prediccion_ml, tuple):
        return prediccion_ml
    
    # datos adicionales
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo")
    
    # analisis tecnico simple
    precio = df['Close'].iloc[-1]
    sma_20 = df['Close'].rolling(20).mean().iloc[-1]
    rsi_value = calcular_rsi_simple(df['Close'])
    
    # combinar resultados
    resultado = prediccion_ml.get_json()
    resultado = json.loads(resultado)
    
    resultado.update({
        'analisis_tecnico': {
            'precio_vs_sma20': 'ARRIBA' if precio > sma_20 else 'ABAJO',
            'distancia_sma20_pct': float((precio - sma_20) / sma_20 * 100),
            'rsi': float(rsi_value),
            'estado_rsi': 'SOBRECOMPRA' if rsi_value > 70 else 
                         'SOBREVENTA' if rsi_value < 30 else 'NEUTRAL',
            'volumen_promedio': float(df['Volume'].mean())
        },
        'resumen': generar_resumen(resultado)
    })
    
    return jsonify(resultado)

def calcular_rsi_simple(precios, periodo=14):
    """calcula rsi simplificado"""
    cambios = precios.diff()
    ganancias = cambios.where(cambios > 0, 0)
    perdidas = -cambios.where(cambios < 0, 0)
    
    avg_ganancia = ganancias.rolling(periodo).mean().iloc[-1]
    avg_perdida = perdidas.rolling(periodo).mean().iloc[-1]
    
    if avg_perdida == 0:
        return 100
    
    rs = avg_ganancia / avg_perdida
    return 100 - (100 / (1 + rs))

def generar_resumen(resultado):
    """genera resumen en texto"""
    if resultado['probabilidad_subida'] > 0.7:
        return "Fuerte señal de compra según ML y análisis técnico"
    elif resultado['probabilidad_subida'] > 0.55:
        return "Leve señal de compra"
    elif resultado['probabilidad_bajada'] > 0.7:
        return "Fuerte señal de venta"
    elif resultado['probabilidad_bajada'] > 0.55:
        return "Leve señal de venta"
    else:
        return "Señal neutral, esperar mejor oportunidad"

# template html mejorado
@app.route('/dashboard/<ticker>')
def dashboard_ticker(ticker):
    """dashboard web para un ticker"""
    resultado = predict_ticker(ticker)
    
    if isinstance(resultado, tuple):
        return f"<h1>Error con {ticker}</h1>"
    
    resultado = resultado.get_json()
    resultado = json.loads(resultado)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard ML - {ticker}</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .card {{ 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin: 15px; 
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .compra {{ color: green; font-weight: bold; }}
            .venta {{ color: red; font-weight: bold; }}
            .neutral {{ color: orange; }}
            .prob-bar {{ 
                height: 20px; 
                background: #eee; 
                margin: 5px 0;
                border-radius: 10px;
                overflow: hidden;
            }}
            .prob-fill {{ 
                height: 100%; 
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                width: {resultado['probabilidad_subida']*100}%;
            }}
        </style>
    </head>
    <body>
        <h1>Dashboard ML - {ticker}</h1>
        
        <div class="card">
            <h2>Prediccion Machine Learning</h2>
            <h3 class="{resultado['prediccion'].lower()}">
                Señal: {resultado['prediccion']} 
                ({resultado['confianza']:.1%} confianza)
            </h3>
            
            <p>Precio actual: ${resultado['precio_actual']:.2f}</p>
            
            <div class="prob-bar">
                <div class="prob-fill"></div>
            </div>
            <p>Probabilidad de subida: {resultado['probabilidad_subida']:.1%}</p>
            <p>Probabilidad de bajada: {resultado['probabilidad_bajada']:.1%}</p>
            
            <h3>Recomendacion: {resultado['recomendacion']}</h3>
        </div>
        
        <div class="card">
            <h2>Detalles Tecnicos</h2>
            <p>Modelo: Random Forest</p>
            <p>Features utilizadas: {resultado['features_utilizadas']}</p>
            <p>Última actualización: {resultado['timestamp']}</p>
        </div>
        
        <div class="card">
            <h2>Analizar otra accion</h2>
            <form action="/dashboard/" method="GET">
                <input type="text" name="ticker" placeholder="Símbolo (AAPL, MSFT...)" 
                       value="{ticker}">
                <button type="submit">Analizar</button>
            </form>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    print("=" * 60)
    print("api ml iniciada")
    print("http://localhost:5000/dashboard/AAPL")
    print("http://localhost:5000/api/predict/AAPL")
    print("=" * 60)
    app.run(debug=True, port=5000)