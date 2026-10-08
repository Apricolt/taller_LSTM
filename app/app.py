"""App de predicción de demanda eléctrica (hora siguiente) con el modelo LSTM elegido.

Ejecutar desde la raíz del proyecto:   streamlit run app/app.py

El modelo se carga desde app/modelo_demanda.joblib (generado con src/exportar_modelo.py),
que incluye la red entrenada y los parámetros de normalización calculados con train.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import preparacion as P
import limpieza as L
from graficos import aplicar_estilo, AZUL, NARANJA, TINTA_2

aplicar_estilo()
st.set_page_config(page_title="Predicción de demanda LSTM", page_icon="⚡", layout="wide")


@st.cache_resource
def cargar():
    paquete = joblib.load(RAIZ / "app" / "modelo_demanda.joblib")
    limpio = pd.read_parquet(RAIZ / "data/processed/dataset_limpio.parquet")
    feats = P.construir_variables(limpio, paquete["config"]["tratamiento"])
    return paquete["config"], paquete["modelo"], paquete, feats


def predecir(modelo, paquete, ventana_df, features):
    """Normaliza la ventana con los parámetros de train, predice y devuelve MW."""
    X = ventana_df[features].to_numpy(dtype="float32")[None, ...].copy()
    ex, ey = paquete["escalador_x"], paquete["escalador_y"]
    X[..., ex["indices"]] = (X[..., ex["indices"]] - ex["media"]) / ex["escala"]
    y_esc = float(modelo.predict(X, verbose=0).ravel()[0])
    return y_esc * ey["escala"] + ey["media"]


def grafico_ventana(ventana_df, t_obj, pred, real=None):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(ventana_df["timestamp"], ventana_df["demanda"], color=AZUL, marker="o", markersize=3,
            label="demanda observada (entrada)")
    ax.scatter([t_obj], [pred], color=NARANJA, s=90, zorder=3, marker="D", label=f"predicción: {pred:.1f} MW")
    if real is not None and not np.isnan(real):
        ax.scatter([t_obj], [real], color=TINTA_2, s=70, zorder=3, marker="o", facecolors="none",
                   linewidths=2, label=f"real: {real:.1f} MW")
    ax.set_ylabel("MW")
    # Leyenda debajo del gráfico para que no tape la serie
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=9)
    ax.set_title(f"Ventana de {len(ventana_df)} h usada como entrada → predicción para {t_obj:%Y-%m-%d %H:%M}")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    return fig


cfg, modelo, paquete, feats = cargar()
n, features = cfg["ventana"], cfg["features"]

st.title("⚡ Predicción de demanda eléctrica — hora siguiente")
st.caption(f"Modelo: **{cfg['nombre']}** · ventana de **{n} h** · tratamiento {cfg['tratamiento']} · "
           f"RMSE test {cfg['RMSE_test']:.1f} MW · R² test {cfg['R2_test']:.3f}")
st.info("Una LSTM no predice con una sola fila: necesita las **últimas n horas** (variables X) "
        "para estimar la demanda de la hora siguiente (y). Por eso cada modo arma una ventana completa.")

modo = st.radio("Modo", ["1 · Histórico del dataset", "2 · Escenario editable (what-if)", "3 · Cargar CSV propio"],
                horizontal=True)

if modo.startswith("1") or modo.startswith("2"):
    validos = feats["timestamp"].iloc[n - 1:-1]
    col1, col2 = st.columns(2)
    fecha = col1.date_input("Fecha de la última hora observada (t)", value=pd.Timestamp("2026-12-01").date(),
                            min_value=validos.min().date(), max_value=validos.max().date())
    hora = col2.slider("Hora de t", 0, 23, 17)
    t = pd.Timestamp(fecha) + pd.Timedelta(hours=hora)
    i = feats.index[feats["timestamp"] == t]
    if len(i) == 0 or i[0] < n - 1:
        st.error("No hay suficientes horas previas para esa fecha."); st.stop()
    i = i[0]
    ventana = feats.iloc[i - n + 1: i + 1].copy()
    if ventana[features].isna().any().any():
        st.warning("La ventana contiene datos faltantes (horas anómalas del tratamiento A). Elija otra hora."); st.stop()
    t_obj = t + pd.Timedelta(hours=1)
    real = feats.at[i, "demanda_objetivo"]

    if modo.startswith("1"):
        pred = predecir(modelo, paquete, ventana, features)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicción", f"{pred:.1f} MW")
        c2.metric("Real", "—" if np.isnan(real) else f"{real:.1f} MW")
        c3.metric("Error", "—" if np.isnan(real) else f"{pred - real:+.1f} MW")
        st.pyplot(grafico_ventana(ventana, t_obj, pred, real))
        with st.expander("Ver las variables de entrada (X) de la ventana"):
            st.dataframe(ventana[["timestamp"] + [c for c in features if not c.endswith(("_sin", "_cos"))]],
                         hide_index=True)
    else:
        st.subheader("Modifique las variables independientes de la última hora observada (t)")
        base = ventana.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        temp = c1.slider("Temperatura (°C)", 0.0, 40.0, float(round(base.temperatura_c, 1)), 0.5)
        hum = c2.slider("Humedad (%)", 0.0, 100.0, float(round(base.humedad_pct, 1)), 1.0)
        rad = c3.slider("Radiación (W/m²)", 0.0, 1100.0, float(round(base.radiacion_wm2, 0)), 10.0)
        precio = c4.slider("Precio (por kWh)", 5.0, 50.0, float(round(base.precio_kwh, 2)), 0.5)
        c5, c6 = st.columns(2)
        dia_t = ventana["timestamp"].dt.date == t.date()
        festivo = c5.checkbox(f"Festivo (todas las horas del {t:%d-%b} en la ventana)", value=bool(base.festivo),
                              help="Un festivo dura el día completo. Marcar solo una hora casi no cambia la predicción, "
                                   "porque el modelo aprendió el efecto a partir de días festivos completos.")
        delta = c6.slider("Desplazar la temperatura de TODA la ventana (°C)", -10.0, 10.0, 0.0, 0.5,
                          help="Simula un día más caluroso o más frío que el real.")
        escenario = ventana.copy()
        escenario["temperatura_c"] += delta
        ult = escenario.index[-1]
        escenario.loc[ult, ["temperatura_c", "humedad_pct", "radiacion_wm2", "precio_kwh"]] = \
            [temp + delta, hum, rad, precio]
        escenario.loc[dia_t, "festivo"] = int(festivo)
        p_orig, p_esc = predecir(modelo, paquete, ventana, features), predecir(modelo, paquete, escenario, features)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicción con datos reales", f"{p_orig:.1f} MW")
        c2.metric("Predicción del escenario", f"{p_esc:.1f} MW", f"{p_esc - p_orig:+.1f} MW")
        c3.metric("Real", "—" if np.isnan(real) else f"{real:.1f} MW")
        st.pyplot(grafico_ventana(escenario, t_obj, p_esc, real))
        st.caption("Nota: el modelo aprendió de combinaciones observadas. Escenarios muy alejados de los datos "
                   "de entrenamiento (p. ej. 40 °C en diciembre) son extrapolaciones poco confiables.")

else:
    st.subheader("Cargar un CSV con al menos n horas consecutivas")
    st.markdown(f"Debe tener las mismas columnas que el dataset original y **al menos {n} horas consecutivas**. "
                "La app ordena, quita duplicados, normaliza el texto y rellena huecos cortos de las exógenas "
                "igual que en el entrenamiento, y predice la hora siguiente a la última fila.")
    plantilla = pd.read_csv(RAIZ / "data/raw/dataset_demanda_energia_LSTM_2025_2026.csv", parse_dates=["timestamp"])
    plantilla = plantilla.drop_duplicates().sort_values("timestamp").tail(n + 24)
    st.download_button("Descargar plantilla de ejemplo", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_ventana.csv", "text/csv")
    archivo = st.file_uploader("CSV", type="csv")
    if archivo:
        crudo = pd.read_csv(archivo, parse_dates=["timestamp"])
        d, _ = L.eliminar_duplicados(crudo)
        d, faltan = L.ordenar_y_verificar(d)
        if len(faltan):
            st.error(f"Faltan {len(faltan)} horas en la secuencia: deben ser horas consecutivas."); st.stop()
        if len(d) < n:
            st.error(f"Se necesitan al menos {n} horas; el archivo tiene {len(d)}."); st.stop()
        d, _, _ = L.corregir_periodo_dia(d)
        # Sin columna objetivo en el CSV la recuperación B no aplica: la demanda observada se usa tal cual.
        d["demanda_mw_recuperada"] = d["demanda_mw"]
        if "demanda_objetivo" not in d:
            d["demanda_objetivo"] = np.nan
        f = P.construir_variables(d, cfg["tratamiento"])
        ventana = f.tail(n)
        if ventana[features].isna().any().any():
            st.error("La ventana tiene valores faltantes que no se pueden rellenar (huecos > 3 h)."); st.stop()
        t_obj = ventana["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        pred = predecir(modelo, paquete, ventana, features)
        st.metric(f"Demanda predicha para {t_obj:%Y-%m-%d %H:%M}", f"{pred:.1f} MW")
        st.pyplot(grafico_ventana(ventana, t_obj, pred))
