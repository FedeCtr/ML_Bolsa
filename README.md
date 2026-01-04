# ML_Bolsa

Prediccion de acciones con Machine Learning.

## Que hace

Descarga datos de Yahoo Finance, calcula indicadores tecnicos y predice si una accion va a subir o bajar usando RandomForest.

## Features

- Descarga automatica desde Yahoo Finance
- Indicadores: SMA, RSI, volatilidad, volumen
- API REST para hacer predicciones
- Dashboard web simple
- Recomendaciones: COMPRAR/MANTENER/VENDER

## Estructura

```
prediction_api/
├── src/
│   ├── data/         # descarga y procesamiento
│   ├── ml/           # modelos y predicciones
│   ├── api/          # Flask + endpoints
│   └── utils/        # config y helpers
├── scripts/          # ejecutables
├── notebooks/        # jupyter
├── data/             # raw/processed/ml_ready
└── models/           # .pkl entrenados
```

## Setup

```bash
# clonar
git clone https://github.com/FedeCtr/ML_Bolsa.git
cd ML_Bolsa

# crear venv
python -m venv venv
venv\Scripts\activate  # windows
# source venv/bin/activate  # linux/mac

# instalar
pip install -r requirements.txt
```

## Uso

Todo en uno:
```bash
python scripts/run_pipeline.py
```

O paso a paso:
```bash
python scripts/download_data.py   # descargar datos
python scripts/train_model.py     # entrenar modelo
python scripts/run_api.py         # levantar API
```

Abre http://localhost:5000

## API

**Prediccion**
```bash
curl http://localhost:5000/api/predict/AAPL
```

**Respuesta**
```json
{
  "ticker": "AAPL",
  "prediction": 1,
  "confidence": 0.75,
  "current_price": 175.50,
  "recommendation": "COMPRAR"
}
```

**Batch**
```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "GOOGL", "MSFT"]}'
```

## Codigo

```python
from src.ml.predictor import StockPredictor

predictor = StockPredictor()
resultado = predictor.predict('AAPL')

print(resultado['prediction'])       # 1 = sube, 0 = baja
print(resultado['confidence'])       # confianza del modelo
print(resultado['recommendation'])   # COMPRAR/MANTENER/VENDER
```

## Tech Stack

- Python 3.14
- Flask - API REST
- yfinance - datos de Yahoo Finance
- scikit-learn - RandomForest
- pandas/numpy - procesamiento
- joblib - persistencia de modelos

## Datos

Los datos se guardan automaticamente en:
- `data/raw/` - datos descargados
- `data/processed/` - con indicadores tecnicos
- `data/ml_ready/` - listos para entrenar
- `models/` - modelos .pkl entrenados

## Notebooks

Abre `notebooks/01_exploracion.ipynb` para ver ejemplos de:
- Descarga de datos
- Calculo de indicadores
- Entrenamiento de modelos
- Visualizaciones
- Predicciones

## Config

Edita `src/utils/config.py` o `config.yaml` para cambiar:
- Tickers por defecto
- Parametros del modelo
- Periodos de descarga
- Rutas de archivos

## Notas

- Las predicciones son solo para fines educativos
- No usar como consejo financiero
- El modelo necesita ser reentrenado periodicamente
- Minimo 2 anos de datos historicos recomendado

## License

MIT
