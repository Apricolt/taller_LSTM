"""Fase 3: variables, split cronológico, normalización y ventanas.

Convención temporal: la fila t contiene lo observado en la hora t y su
`demanda_objetivo` es la demanda de t+1. Una muestra usa las filas
[t-n+1 ... t] como entrada y `demanda_objetivo(t)` como salida, así que la
entrada nunca contiene información posterior a t.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from limpieza import rellenar_exogenas_causal

INICIO_VAL = pd.Timestamp("2026-07-01")
INICIO_TEST = pd.Timestamp("2026-10-01")
VENTANAS = (12, 24, 48)

# Columna de demanda según el tratamiento de anomalías (ver Fase 1).
COL_DEMANDA = {"A": "demanda_mw", "B": "demanda_mw_recuperada"}

EXOGENAS = ["temperatura_c", "humedad_pct", "radiacion_wm2", "precio_kwh"]
CALENDARIO = ["hora_sin", "hora_cos", "dia_sin", "dia_cos", "mes_sin", "mes_cos", "fin_semana", "festivo"]
BANDERAS = ["flag_exogena_rellenada"]
FEATURES = ["demanda"] + EXOGENAS + CALENDARIO + BANDERAS


def construir_variables(df, tratamiento="B", exogenas=None):
    """Agrega codificación cíclica, rellena exógenas (ffill causal) y fija la columna `demanda`."""
    exogenas = EXOGENAS if exogenas is None else exogenas
    df = rellenar_exogenas_causal(df, columnas=exogenas)
    df["demanda"] = df[COL_DEMANDA[tratamiento]]
    # Seno/coseno: la hora 23 y la 0 quedan juntas, como en la realidad.
    for col, periodo, nombre in [("hora", 24, "hora"), ("dia_semana", 7, "dia"), ("mes", 12, "mes")]:
        valor = df[col] - (1 if col == "mes" else 0)
        df[f"{nombre}_sin"] = np.sin(2 * np.pi * valor / periodo)
        df[f"{nombre}_cos"] = np.cos(2 * np.pi * valor / periodo)
    return df


def split_de_objetivo(timestamps):
    """Asigna cada muestra al split según la hora del OBJETIVO (t+1), no la de la entrada."""
    t_obj = timestamps + pd.Timedelta(hours=1)
    return np.select([t_obj < INICIO_VAL, t_obj < INICIO_TEST], ["train", "val"], "test")


class Escaladores:
    """StandardScaler para X (solo columnas continuas) y para y, ajustados SOLO con train."""

    def __init__(self, features):
        self.features = list(features)
        # Se estandariza toda variable continua (demanda y exógenas); sin/cos y binarias no.
        self.idx = [i for i, c in enumerate(self.features) if c not in CALENDARIO + BANDERAS]
        self.x = StandardScaler()
        self.y = StandardScaler()

    def ajustar(self, df, filas_train, y_train):
        self.x.fit(df.loc[filas_train, [self.features[i] for i in self.idx]].dropna().values)
        self.y.fit(np.asarray(y_train).reshape(-1, 1))
        return self

    def transformar_x(self, matriz):
        matriz = matriz.astype("float32").copy()
        matriz[..., self.idx] = (matriz[..., self.idx] - self.x.mean_) / self.x.scale_
        return matriz

    def transformar_y(self, y):
        return ((np.asarray(y) - self.y.mean_[0]) / self.y.scale_[0]).astype("float32")

    def invertir_y(self, y_esc):
        return np.asarray(y_esc).ravel() * self.y.scale_[0] + self.y.mean_[0]


def crear_ventanas(df, n, features):
    """Devuelve X (muestras, n, features), y, timestamp de la última hora de entrada (t) y validez.

    Una ventana es válida si sus n filas de entrada y su objetivo no tienen NaN.
    Como la rejilla horaria está completa (Fase 1), n filas consecutivas = n horas consecutivas.
    """
    datos = df[features].to_numpy(dtype="float32")
    y = df["demanda_objetivo"].to_numpy(dtype="float32")
    fila_ok = ~np.isnan(datos).any(axis=1)
    # Ventanas deslizantes sin copiar memoria: forma (muestras, features, n) -> (muestras, n, features)
    X = np.lib.stride_tricks.sliding_window_view(datos, n, axis=0).transpose(0, 2, 1)
    ok_ventana = np.lib.stride_tricks.sliding_window_view(fila_ok, n).all(axis=1)
    fin = np.arange(n - 1, len(df))           # índice de la última fila (t) de cada ventana
    y_v = y[fin]
    valida = ok_ventana & ~np.isnan(y_v)
    return X, y_v, df["timestamp"].to_numpy()[fin], valida


def preparar(df_limpio, n, tratamiento="B", exogenas=None):
    """Pipeline completo para una ventana y un tratamiento. Devuelve un dict con los splits."""
    exogenas = EXOGENAS if exogenas is None else exogenas
    features = ["demanda"] + exogenas + CALENDARIO + BANDERAS
    df = construir_variables(df_limpio, tratamiento, exogenas)
    X, y, ts, valida = crear_ventanas(df, n, features)
    split = split_de_objetivo(pd.to_datetime(ts))

    m_train = valida & (split == "train")
    esc = Escaladores(features).ajustar(df, df["timestamp"] < INICIO_VAL, y[m_train])

    salida = {"features": features, "escaladores": esc, "n": n, "tratamiento": tratamiento}
    for nombre in ("train", "val", "test"):
        m = valida & (split == nombre)
        salida[nombre] = {"X": esc.transformar_x(X[m]), "y": esc.transformar_y(y[m]),
                          "y_mw": y[m], "ts_objetivo": pd.to_datetime(ts[m]) + pd.Timedelta(hours=1),
                          "demanda_t": df["demanda_mw_recuperada"].to_numpy()[n - 1:][m]}
    return salida
