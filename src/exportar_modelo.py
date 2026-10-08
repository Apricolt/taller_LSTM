"""Empaqueta el modelo de producción en un único archivo .joblib para el despliegue.

El paquete contiene todo lo necesario para predecir sin volver a entrenar ni recalcular nada:
el modelo Keras, los parámetros de normalización (calculados solo con train), la lista
ordenada de variables, la ventana y las métricas.

Uso:  python src/exportar_modelo.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joblib
import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

import preparacion as P

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "app" / "modelo_demanda.joblib"


def exportar():
    cfg = json.loads((RAIZ / "resultados/modelo_produccion.json").read_text(encoding="utf-8"))
    modelo = keras.models.load_model(RAIZ / "resultados/modelos" / f"{cfg['id']}.keras")
    df = pd.read_parquet(RAIZ / "data/processed/dataset_limpio.parquet")
    esc = P.preparar(df, cfg["ventana"], cfg["tratamiento"])["escaladores"]

    paquete = {
        "modelo": modelo,
        "config": cfg,
        "features": esc.features,
        # Normalización como arreglos simples: no depende de clases propias ni de la versión de sklearn.
        "escalador_x": {"indices": list(esc.idx), "media": esc.x.mean_.astype("float32"),
                        "escala": esc.x.scale_.astype("float32")},
        "escalador_y": {"media": float(esc.y.mean_[0]), "escala": float(esc.y.scale_[0])},
        "versiones": {"tensorflow": tf.__version__, "keras": keras.__version__,
                      "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    joblib.dump(paquete, SALIDA, compress=3)

    # Verificación: el paquete recargado debe predecir exactamente igual que el modelo original.
    cargado = joblib.load(SALIDA)
    X = P.preparar(df, cfg["ventana"], cfg["tratamiento"])["test"]["X"][:256]
    iguales = np.allclose(modelo.predict(X, verbose=0), cargado["modelo"].predict(X, verbose=0), atol=1e-5)
    print(f"Guardado {SALIDA.relative_to(RAIZ)} ({SALIDA.stat().st_size / 1e6:.1f} MB) · "
          f"predicciones idénticas tras recargar: {iguales}")


if __name__ == "__main__":
    exportar()
