"""Entrenamiento y registro de experimentos. Cada corrida se guarda al terminar, así
que si el proceso se interrumpe se puede relanzar y retoma donde quedó."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import keras

import preparacion as P
import modelos as M
from metricas import metricas, metricas_picos

RAIZ = Path(__file__).resolve().parent.parent
DIR_RES = RAIZ / "resultados" / "corridas"
DIR_MOD = RAIZ / "resultados" / "modelos"

# Early Stopping: se detiene si en 5 épocas la pérdida de validación no mejora al menos `min_delta`.
# min_delta = 5e-4 en MSE escalado equivale a ~0.1 MW de RMSE (con RMSE ≈ 25 MW y desv. de y ≈ 98 MW).
# Con paciencia 10 y sin min_delta, el 73 % del tiempo se gastaba en ganar < 0.5 MW
# (ver resultados/analisis_paciencia10.csv).
CONFIG_ENTRENAMIENTO = {"optimizador": "adam", "lr": 1e-3, "batch_size": 128, "max_epocas": 100,
                        "paciencia_early_stopping": 5, "min_delta": 5e-4, "paciencia_reduce_lr": 3}


def conjunto_comun_test(df):
    """Horas objetivo de test válidas en TODAS las configuraciones (la más restrictiva es
    tratamiento A con ventana 48). Todas las comparaciones se hacen sobre este mismo conjunto."""
    return set(P.preparar(df, max(P.VENTANAS), "A")["test"]["ts_objetivo"])


def epoca_restaurada(val_loss, min_delta=CONFIG_ENTRENAMIENTO["min_delta"]):
    """Época cuyos pesos restaura EarlyStopping: la última que mejoró la validación en más de
    `min_delta` (misma regla que Keras). Es la columna 'Épocas' de la tabla de resultados."""
    mejor, epoca = None, 0
    for i, v in enumerate(val_loss):
        if mejor is None or v + min_delta < mejor:
            mejor, epoca = v, i
    return epoca + 1


def _predecir(modelo, datos, esc):
    return esc.invertir_y(modelo.predict(datos["X"], batch_size=1024, verbose=0))


def entrenar(df, id_corrida, arq, n, tratamiento="B", semilla=0, early_stopping=True,
             kwargs_modelo=None, exogenas=None, comun=None, guardar_modelo=True, max_epocas=None):
    archivo = DIR_RES / f"{id_corrida}.json"
    if archivo.exists():
        return json.loads(archivo.read_text(encoding="utf-8"))

    keras.utils.set_random_seed(semilla)
    datos = P.preparar(df, n, tratamiento, exogenas)
    esc = datos["escaladores"]
    modelo = M.ARQUITECTURAS[arq](n, len(datos["features"]), **(kwargs_modelo or {}))
    modelo.compile(optimizer=keras.optimizers.Adam(CONFIG_ENTRENAMIENTO["lr"]), loss="mse")

    callbacks = [keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=CONFIG_ENTRENAMIENTO["paciencia_reduce_lr"])]
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            patience=CONFIG_ENTRENAMIENTO["paciencia_early_stopping"], min_delta=CONFIG_ENTRENAMIENTO["min_delta"],
            restore_best_weights=True))

    t0 = time.time()
    hist = modelo.fit(datos["train"]["X"], datos["train"]["y"],
                      validation_data=(datos["val"]["X"], datos["val"]["y"]),
                      epochs=max_epocas or CONFIG_ENTRENAMIENTO["max_epocas"], batch_size=CONFIG_ENTRENAMIENTO["batch_size"],
                      shuffle=True, callbacks=callbacks, verbose=0)
    segundos = time.time() - t0
    val_loss = hist.history["val_loss"]
    mejor_epoca = epoca_restaurada(val_loss) if early_stopping else len(val_loss)

    res = {"id": id_corrida, "arquitectura": arq, "ventana": n, "tratamiento": tratamiento,
           "semilla": semilla, "early_stopping": early_stopping, "kwargs_modelo": kwargs_modelo or {},
           "features": datos["features"], "parametros": int(modelo.count_params()),
           "epocas_entrenadas": len(val_loss), "mejor_epoca": mejor_epoca,
           "segundos": round(segundos, 1), "historial": {k: [float(v) for v in vs] for k, vs in hist.history.items()
                                                          if k in ("loss", "val_loss")},
           "muestras": {s: int(len(datos[s]["y"])) for s in ("train", "val", "test")}}

    for split in ("train", "val", "test"):
        pred = _predecir(modelo, datos[split], esc)
        res[f"metricas_{split}"] = metricas(datos[split]["y_mw"], pred)
        if split == "test":
            m = np.isin(datos["test"]["ts_objetivo"], list(comun)) if comun else np.ones(len(pred), bool)
            res["metricas_test_comun"] = {**metricas(datos["test"]["y_mw"][m], pred[m]),
                                          **metricas_picos(datos["test"]["y_mw"][m], pred[m])}
            pd.DataFrame({"ts_objetivo": datos["test"]["ts_objetivo"], "real": datos["test"]["y_mw"],
                          "pred": pred, "en_comun": m}).to_parquet(DIR_RES / f"{id_corrida}_pred_test.parquet")

    if guardar_modelo:
        modelo.save(DIR_MOD / f"{id_corrida}.keras")
    archivo.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def cargar_resultados():
    filas = []
    for f in sorted(DIR_RES.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        fila = {k: r[k] for k in ("id", "arquitectura", "ventana", "tratamiento", "semilla", "early_stopping",
                                  "parametros", "epocas_entrenadas", "mejor_epoca", "segundos")}
        if r["early_stopping"]:  # se recalcula con la regla de Keras para que coincida con los pesos restaurados
            fila["mejor_epoca"] = epoca_restaurada(r["historial"]["val_loss"])
        fila["kwargs_modelo"] = json.dumps(r["kwargs_modelo"])
        fila["n_features"] = len(r["features"])
        for split in ("train", "val", "test_comun"):
            for k, v in r[f"metricas_{split}"].items():
                fila[f"{k}_{split}"] = v
        filas.append(fila)
    return pd.DataFrame(filas)
