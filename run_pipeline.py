import os
import sys

print("iniciando...")

# preparar datos
print("\n1. descargando datos de yahoo finance")
try:
    from src.preparar_data import DataPreparerML
    prep = DataPreparerML()
    prep.descargar_datos_completos(years=2)
    prep.crear_dataset_unificado()
    print("   ok")
except Exception as e:
    print(f"   fallo: {e}")
    sys.exit(1)

# entrenar modelo
print("\n2. entrenando modelo con los datos")
try:
    from src.modelo_ml_basico import ModeloMLBasico
    modelo = ModeloMLBasico()
    datos = modelo.cargar_datos(ticker='AAPL')
    
    if datos is not None:
        X, y = modelo.preparar_features(datos)
        if X is not None:
            modelo.entrenar_modelo(X, y)
            modelo.guardar_modelo()
            print("   ok")
    else:
        print("   no hay datos")
        sys.exit(1)
except Exception as e:
    print(f"   fallo: {e}")
    sys.exit(1)

# iniciar api
print("\n3. iniciando api web")
print("   abriendo en http://localhost:5000")

if os.name == 'nt':
    os.system('start cmd /k python src/app_ml.py')
else:
    os.system('python3 src/app_ml.py &')

print("\nterminado")