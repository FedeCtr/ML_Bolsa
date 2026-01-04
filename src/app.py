from flask import Flask, jsonify, render_template_string, request
import yfinance as yf
import pandas as pd
import os

app = Flask(__name__)

# html de inicio
HTML_INICIO = """
<!DOCTYPE html>
<html>
<head>
    <title>API de Bolsa</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            padding: 20px; 
            background: #f8f9fa;
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .card {
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #34495e;
        }
        input, button { 
            padding: 12px; 
            margin: 5px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        button {
            background: #3498db;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background: #2980b9;
        }
        .option {
            padding: 10px;
            margin: 5px 0;
            background: #ecf0f1;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .option strong {
            color: #2c3e50;
        }
        .endpoint {
            font-family: monospace;
            color: #27ae60;
        }
    </style>
</head>
<body>
    <h1>Sistema de Predicción</h1>
    
    <div class="card">
        <h3>Analizar una acción:</h3>
        <form action="/analizar" method="GET">
            <input type="text" name="ticker" placeholder="Ej: AAPL, MSFT, SPY" required>
            <button type="submit">Analizar</button>
        </form>
    </div>
    
    
    <div class="card">
        <h3> Ejemplos:</h3>
        <p>
            <strong>Tecnología:</strong> AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA<br>
            <strong>Índices:</strong> SPY (S&P 500), ^GSPC (S&P 500 Index), ^DJI (Dow Jones)<br>
            <strong>Finanzas:</strong> JPM, BAC, V, MA<br>
            <strong>Otros:</strong> WMT, KO, NKE, PFE
        </p>
    </div>
</body>
</html>
"""

@app.route('/')
def inicio():
    return render_template_string(HTML_INICIO)

@app.route('/precio/<ticker>')
def precio(ticker):
    try:
        stock = yf.Ticker(ticker)
        datos = stock.history(period="1d")
        
        if datos.empty:
            return jsonify({'error': 'no hay datos para ' + ticker}), 404
            
        return jsonify({
            'ticker': ticker,
            'precio': float(datos['Close'].iloc[-1]),
            'fecha': str(datos.index[-1]),
            'moneda': 'USD'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analizar')
def analizar():
    ticker = request.args.get('ticker', 'AAPL').upper()
    
    try:
        # logica de analisis
        stock = yf.Ticker(ticker)
        datos = stock.history(period="1mo")
        
        precio_actual = datos['Close'].iloc[-1]
        media_10d = datos['Close'].tail(10).mean()
        
        # recomendacion
        if precio_actual > media_10d:
            recomendacion = "COMPRA"
            clase_css = "compra"
        else:
            recomendacion = "VENTA"
            clase_css = "venta"
        
        # HTML de respuesta
        html_respuesta = f"""
        <!DOCTYPE html>
        <html>
        <head><title>analisis de {ticker}</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>analisis de {ticker}</h1>
            <div class="card" style="background: #f0f0f0; padding: 20px; border-radius: 10px;">
                <h2>precio actual: ${precio_actual:.2f}</h2>
                <p>media 10 días: ${media_10d:.2f}</p>
                <h3 class="{clase_css}">recomendacion: {recomendacion}</h3>
                <p><a href="/">← Volver al inicio</a></p>
            </div>
        </body>
        </html>
        """
        
        return render_template_string(html_respuesta)
        
    except Exception as e:
        return f"<h1>Error</h1><p>no se pudo analizar {ticker}: {str(e)}</p>"

@app.route('/api/<ticker>')
def api_json(ticker):
    """endpoint json puro"""
    try:
        stock = yf.Ticker(ticker)
        datos = stock.history(period="1mo")
        
        precio = float(datos['Close'].iloc[-1])
        media_10 = float(datos['Close'].tail(10).mean())
        
        return jsonify({
            'ticker': ticker,
            'precio': precio,
            'media_10d': media_10,
            'recomendacion': 'COMPRA' if precio > media_10 else 'VENTA',
            'diferencia_porcentaje': float(((precio - media_10) / media_10) * 100)
        })
    except:
        return jsonify({'error': 'Ticker no válido'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)