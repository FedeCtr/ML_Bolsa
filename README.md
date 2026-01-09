# ML_Bolsa - Predictor Avanzado 70%+ 📈

Sistema de predicción de mercado de valores con **ensemble de 4 modelos** y **50+ features** para alcanzar **70%+ accuracy**.

## 🎯 Características

### Sistema Básico
- ✅ Modelo RandomForest con 10 features → ~53% accuracy
- ✅ API Flask funcional
- ✅ Dashboard web
- ✅ Jupyter notebook exploratorio

### Sistema Avanzado (NUEVO) 🚀
- ✅ **4 modelos ensamblados**: XGBoost, LightGBM, RandomForest, ExtraTrees
- ✅ **50+ features técnicas**: MACD, Bollinger, ADX, RSI, ATR, Stochastic, Williams, CCI, OBV, VWAP, lag features
- ✅ **Optimización Optuna**: hyperparameter tuning automático
- ✅ **Filtro de confianza >75%**: solo actúa en señales de alta confianza
- ✅ **Objetivo: 70%+ accuracy**

## 📂 Estructura

```
prediction_api/
├── src/
│   ├── data/
│   │   ├── collector.py              - descarga datos (yfinance)
│   │   └── processor.py              - 50+ indicadores tecnicos
│   ├── ml/
│   │   ├── trainer.py                - modelo basico (RandomForest)
│   │   ├── advanced_trainer.py       - ensemble 4 modelos + Optuna
│   │   ├── predictor.py              - predictor basico
│   │   ├── advanced_predictor.py     - predictor con filtro confianza
│   │   └── evaluator.py              - metricas
│   ├── api/
│   │   ├── routes.py                 - Flask API
│   │   └── templates/dashboard.html  - interfaz web
│   └── utils/
│       ├── config.py                 - configuracion
│       └── logger.py                 - logging
├── scripts/
│   ├── train_advanced.py             - entrenar ensemble
│   └── predict_advanced.py           - predecir con filtro
├── notebooks/
│   └── 01_exploracion.ipynb          - analisis + sistema avanzado
├── data/                              - raw/processed/ml_ready
└── models/                            - .pkl entrenados
```

## 🚀 Instalación

```bash
# 1. clonar
git clone https://github.com/FedeCtr/ML_Bolsa.git
cd ML_Bolsa

# 2. crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows

# 3. instalar dependencias base
pip install -r requirements.txt

# 4. instalar dependencias avanzadas
pip install xgboost lightgbm optuna ta scikit-optimize
```

## 📊 Uso - Sistema Básico

```bash
# entrenar modelo basico
python run_pipeline.py

# iniciar API
python run_api.py

# dashboard: http://localhost:5000
```

## 🎯 Uso - Sistema Avanzado (70%+ accuracy)

### Entrenar Ensemble

```bash
# entrenamiento basico (sin optimizacion)
python scripts/train_advanced.py

# entrenamiento optimizado con Optuna (RECOMENDADO)
python scripts/train_advanced.py --optimize --trials 100

# entrenar con mas datos
python scripts/train_advanced.py --optimize --trials 100 --tickers AAPL MSFT GOOGL AMZN TSLA NVDA META --period 3y
```

### Predecir con Filtro de Confianza

```bash
# prediccion individual
python scripts/predict_advanced.py AAPL

# prediccion multiple
python scripts/predict_advanced.py AAPL MSFT GOOGL

# ajustar umbral de confianza
python scripts/predict_advanced.py AAPL --confidence 0.80
```

**Ejemplo de salida:**
```
Ticker:             AAPL
Precio actual:      $175.43
Prediccion:         COMPRAR
Confianza:          82.45%
Prob. Subida:       82.45%
Prob. Bajada:       17.55%
Mensaje:            alta confianza (82.45%)
```

Si la confianza es <75%, retorna `no_action` para evitar señales débiles.
## 🧠 Detalles Técnicos

### 50+ Features Técnicas

**Momentum (8)**:
- Retornos: 1d, 3d, 5d, 10d, 20d
- Momentum: 5, 10, 20

**Trend (13)**:
- SMA: 10, 20, 50, 200
- EMA: 12, 26
- Distancias a SMA: 10, 20, 50
- MACD: macd, signal, diff
- ADX

**Volatility (7)**:
- Volatilidad: 5d, 10d, 20d, 60d
- ATR, ATR %
- Bollinger Width

**Oscillators (8)**:
- RSI
- Stochastic: K, D
- Williams %R
- CCI
- Bollinger %

**Volume (4)**:
- Volume ratio
- Volume ROC
- OBV EMA
- VWAP distance

**Lag Features (5)**:
- Close lag: 1, 2, 3, 5, 10

**Price Patterns (2)**:
- Range 20d
- Body %

**Temporal (6)**:
- Día semana, mes, trimestre, día mes
- Fin de mes, inicio de mes

### Ensemble - Pesos Optimizados

- **XGBoost**: 35% - mejor para patrones complejos
- **LightGBM**: 30% - rápido y preciso
- **RandomForest**: 25% - estable y robusto
- **ExtraTrees**: 10% - diversidad adicional

Votación suave (probabilidades) con `VotingClassifier`.

### Optimización Optuna

Hiperparámetros optimizados:
- `max_depth`: 3-10
- `learning_rate`: 0.01-0.3
- `n_estimators`: 100-500
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- Y más...

### Filtro de Confianza

Solo retorna señal si `confidence > 75%`, caso contrario `no_action`.

Mejora la **precisión** sacrificando **recall** → menos señales pero más confiables.

## 📈 Rendimiento Esperado

| Sistema | Features | Modelos | Accuracy Esperado |
|---------|----------|---------|-------------------|
| Básico | 10 | RandomForest | ~52-55% |
| Avanzado (sin optimizar) | 50+ | Ensemble 4 | ~60-65% |
| **Avanzado (optimizado)** | **50+** | **Ensemble 4 + Optuna** | **70%+** |
| Avanzado + Filtro 75% | 50+ | Ensemble 4 + Optuna | **75-80%** (señales filtradas) |

*Nota: rendimiento real depende de datos de entrenamiento y condiciones de mercado*

## 🔧 Tecnologías

- **Python 3.14**
- **ML**: scikit-learn, xgboost, lightgbm
- **Optimización**: optuna, scikit-optimize
- **Indicadores**: ta (technical analysis)
- **Data**: yfinance, pandas, numpy
- **API**: Flask
- **Visualización**: matplotlib, seaborn

## 🎓 Roadmap Futuro

- [ ] Agregar LSTM para series temporales
- [ ] Integrar CatBoost (requiere Python <3.14 o compilación manual)
- [ ] Sentiment analysis con noticias
- [ ] Walk-forward validation
- [ ] Backtesting completo
- [ ] API de predicción en tiempo real

## 📝 Licencia

Proyecto educacional para aprendizaje de ML aplicado a finanzas 🚀

---

**Creado con 🧠 para alcanzar 70%+ accuracy en predicción de acciones**
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
