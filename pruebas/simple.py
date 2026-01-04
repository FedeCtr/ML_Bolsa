import yfinance as yf
import pandas as pd

print("version simple de analisis de acciones")
print("=" * 50)

# preguntar al usuario qué acción quiere
ticker = input("Accion a analizar: ").upper()

#  descargar datos
print(f"\n descargando datos de {ticker}...")
stock = yf.Ticker(ticker)

try:
    datos = stock.history(period="1mo")  # ultimo mes
    print(f" datos descargados correctamente")
    print(f"   periodo: {datos.index[0].date()} a {datos.index[-1].date()}")
    print(f"   dias de datos: {len(datos)}")
except:
    print(f" Error: {ticker}")
    print("   verificar")
    exit()

#  mostrar datos básicos
print("\n datos básicos:")
print(f"precio actual: ${datos['Close'].iloc[-1]:.2f}")
print(f"precio más alto del mes: ${datos['High'].max():.2f}")
print(f"precio más bajo del mes: ${datos['Low'].min():.2f}")
print(f"volumen promedio: {datos['Volume'].mean():,.0f} acciones")

#  calcular media móvil simple
print("\n análisis técnico simple:")
precio_actual = datos['Close'].iloc[-1]

# Media de 5 días (corta)
media_5d = datos['Close'].tail(5).mean()
print(f"media 5 días: ${media_5d:.2f}")

# Media de 10 días (media)
media_10d = datos['Close'].tail(10).mean()
print(f"media 10 días: ${media_10d:.2f}")
# recomendación basada en reglas simples
print("\n recomendación:")

if precio_actual > media_10d and precio_actual > media_5d:
    print("señal de compra")
    print("razón: precio por encima de ambas medias")
elif precio_actual < media_10d and precio_actual < media_5d:
    print("señal de venta")
    print("razón: precio por debajo de ambas medias")
elif precio_actual > media_5d:
    print("señal neutra (leve compra)")
    print("razón: precio por encima de media corta")
else:
    print("señal neutra (leve venta)")
    print("razón: precio por debajo de media corta")

# guardar resultados
print("\n guardar en csv")
resultados = pd.DataFrame({
    'fecha': [pd.Timestamp.now()],
    'ticker': [ticker],
    'precio_actual': [precio_actual],
    'media_5d': [media_5d],
    'media_10d': [media_10d],
    'recomendacion': ['COMPRA' if precio_actual > media_10d else 'VENTA' if precio_actual < media_10d else 'NEUTRAL']
})

# guardar en data
resultados.to_csv(f"data/analisis_{ticker}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", index=False)
print(f"guardado en: data/analisis_{ticker}_...csv")
