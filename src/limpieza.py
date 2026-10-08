"""Limpieza del dataset de demanda energética.

Principio rector: un valor *atípico* (raro respecto al resto) no es lo mismo
que un valor *anómalo* (que no pudo haber ocurrido). Solo se anulan los
anómalos, y nunca se reemplazan por valores estimados.

Pruebas usadas para decidir si un valor es un error:
  1. Imposibilidad física      (ej. humedad > 100 %).
  2. Contexto temporal         (pico aislado de 1 hora que regresa de inmediato,
                                en variables con inercia física).
  3. Consistencia interna      (demanda_mw(t) vs demanda_objetivo(t-1)).
  4. Coherencia con las causas (ej. picos de precio sin demanda alta).
"""
import numpy as np
import pandas as pd

FIN_TRAIN = pd.Timestamp("2026-07-01")  # los umbrales se calibran solo con datos previos

COLUMNAS_EXOGENAS = [
    "temperatura_c", "humedad_pct", "viento_kmh",
    "radiacion_wm2", "precipitacion_mm", "precio_kwh",
]

# Horas de cada periodo del día (regla determinista derivada de `hora`).
PERIODOS = {"madrugada": range(0, 6), "mañana": range(6, 12),
            "tarde": range(12, 18), "noche": range(18, 24)}

# Prueba 2: cuántas "desviaciones típicas robustas" debe alejarse un valor de
# sus vecinos para considerarse un pico aislado. 10 es deliberadamente
# conservador: preferimos dejar pasar un error leve que borrar un dato real.
K_CONTEXTO = 10

# Prueba 1: límites físicos.
VIENTO_MAX_AISLADO = 80  # km/h: nivel de temporal; no aparece 1 hora aislada con calma alrededor.


def cargar_crudo(ruta):
    return pd.read_csv(ruta, parse_dates=["timestamp"])


def eliminar_duplicados(df):
    """Elimina filas idénticas. Una hora repetida pesaría doble en el entrenamiento
    y rompería la regla '1 paso de la secuencia = 1 hora'."""
    n_antes = len(df)
    df = df.drop_duplicates()
    if df["timestamp"].duplicated().any():
        raise ValueError("Hay timestamps repetidos con valores distintos: revisar manualmente.")
    return df, n_antes - len(df)


def ordenar_y_verificar(df):
    """Ordena cronológicamente y comprueba que la rejilla horaria esté completa."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    esperado = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    faltantes = esperado.difference(df["timestamp"])
    return df, faltantes


def corregir_periodo_dia(df):
    """Normaliza el texto y recalcula `periodo_dia` desde `hora`.

    No es inventar un valor: `periodo_dia` es una etiqueta derivada de la hora,
    así que la hora (verificada contra el timestamp) es la fuente de verdad.
    """
    texto = df["periodo_dia"].str.strip().str.lower()
    derivado = df["hora"].map({h: p for p, horas in PERIODOS.items() for h in horas})
    n_texto = int((df["periodo_dia"] != texto).sum())
    n_incoherentes = int((texto != derivado).sum())
    df = df.assign(periodo_dia=derivado)
    return df, n_texto, n_incoherentes


def _vecinos(x):
    prev, sig = x.shift(1), x.shift(-1)
    desvio = (x - (prev + sig) / 2).abs()      # distancia al promedio de sus vecinos
    entre_vecinos = (prev - sig).abs()          # cuánto difieren los vecinos entre sí
    return prev, sig, desvio, entre_vecinos


def pico_aislado(df, col, k=K_CONTEXTO):
    """Prueba 2 (contexto temporal).

    Un valor es pico aislado si (a) se aleja de sus vecinos mucho más de lo
    normal y (b) los vecinos coinciden entre sí. La condición (b) evita marcar
    a los vecinos de un pico, que también parecen 'alejados' de su promedio.
    El umbral = mediana + k * MAD del desvío, calculado SOLO en entrenamiento.
    """
    _, _, desvio, entre_vecinos = _vecinos(df[col])
    en_train = df["timestamp"] < FIN_TRAIN
    mediana = desvio[en_train].median()
    mad = 1.4826 * (desvio[en_train] - mediana).abs().median()
    umbral = mediana + k * mad
    return (desvio > umbral) & (entre_vecinos < desvio), umbral


def clasificar_extremos(df, precio_extremo_es_error=True):
    """Devuelve una tabla de auditoría: una fila por valor marcado, con la prueba
    que falló y su contexto. Veredictos: 'error' (se anula) o 'dudoso' (se conserva
    o se trata según el experimento)."""
    marcas = []

    def registrar(mascara, col, prueba, veredicto):
        prev, sig, _, _ = _vecinos(df[col])
        for i in df.index[mascara.fillna(False)]:
            marcas.append({"timestamp": df.at[i, "timestamp"], "variable": col,
                           "anterior": prev[i], "valor": df.at[i, col], "siguiente": sig[i],
                           "prueba": prueba, "veredicto": veredicto})

    # Temperatura y humedad: fuerte inercia física -> un pico aislado extremo es error.
    # (Se registra primero la prueba física para que quede como motivo principal.)
    fisico_hum = (df["humedad_pct"] < 0) | (df["humedad_pct"] > 100)
    registrar(fisico_hum, "humedad_pct", "1-fisica", "error")
    for col in ["temperatura_c", "humedad_pct"]:
        aislado, _ = pico_aislado(df, col)
        registrar(aislado, col, "2-contexto", "error")

    # Viento: negativo es imposible; > 80 km/h aislado no es una tormenta (duran horas).
    # Picos aislados por debajo de 80 podrían ser ráfagas reales -> dudosos, se conservan.
    aislado, _ = pico_aislado(df, "viento_kmh")
    v = df["viento_kmh"]
    registrar(v < 0, "viento_kmh", "1-fisica", "error")
    registrar(aislado & (v > VIENTO_MAX_AISLADO), "viento_kmh", "1-fisica+2-contexto", "error")
    registrar(aislado & v.between(0, VIENTO_MAX_AISLADO), "viento_kmh", "2-contexto", "dudoso")

    # Precio: en mercados reales hay picos y hasta precios negativos, así que no basta
    # con que sea raro. Aquí son de 1 hora y no coinciden con demanda alta (prueba 4).
    aislado, _ = pico_aislado(df, "precio_kwh")
    registrar(aislado, "precio_kwh", "2-contexto+4-causas",
              "error" if precio_extremo_es_error else "dudoso")

    # Precipitación y radiación son intermitentes por naturaleza (la lluvia empieza de
    # golpe, una nube tapa el sol): la prueba de inercia no aplica. Solo prueba física.
    for col in ["precipitacion_mm", "radiacion_wm2"]:
        registrar(df[col] < 0, col, "1-fisica", "error")
    registrar((df["radiacion_wm2"] > 0) & df["hora"].isin([0, 1, 2, 3, 4, 21, 22, 23]),
              "radiacion_wm2", "1-fisica", "error")

    # Demanda: dos mediciones de la misma hora deben coincidir (prueba 3).
    otra_medicion = df["demanda_objetivo"].shift(1)
    conflicto = (df["demanda_mw"] - otra_medicion).abs() > 0.01
    relativo = (df["demanda_mw"] / otra_medicion - 1).abs()
    registrar(conflicto & (relativo >= 0.25), "demanda_mw", "3-consistencia+2-contexto", "error")
    registrar(conflicto & (relativo < 0.25), "demanda_mw", "3-consistencia", "dudoso")

    auditoria = pd.DataFrame(marcas)
    if not auditoria.empty:
        auditoria = auditoria.drop_duplicates(["timestamp", "variable"]).sort_values(
            ["variable", "timestamp"]).reset_index(drop=True)
    return auditoria


def aplicar_limpieza(df, auditoria):
    """Anula (NaN) los valores con veredicto 'error'. No reemplaza nada.

    Para la demanda se anulan errores y dudosos (tratamiento A), y además se crea
    `demanda_mw_recuperada` (tratamiento B) usando la otra medición real de esa
    misma hora: demanda_objetivo(t-1).
    """
    df = df.copy()
    errores = auditoria[auditoria["veredicto"] == "error"]
    for col, grupo in errores.groupby("variable"):
        if col != "demanda_mw":
            df.loc[df["timestamp"].isin(grupo["timestamp"]), col] = np.nan

    marcados_demanda = df["timestamp"].isin(
        auditoria.loc[auditoria["variable"] == "demanda_mw", "timestamp"])
    df["demanda_mw_recuperada"] = df["demanda_mw"].where(
        ~marcados_demanda, df["demanda_objetivo"].shift(1))
    df.loc[marcados_demanda, "demanda_mw"] = np.nan
    df["flag_demanda_corregida"] = marcados_demanda.astype(int)
    return df


def rellenar_exogenas_causal(df, columnas=None, limite=3):
    """Forward-fill causal: repite el ÚLTIMO valor conocido, como haría un sistema
    en tiempo real que no ha recibido la lectura nueva. No usa el futuro (a
    diferencia de interpolar) y no fabrica valores nuevos. Huecos > `limite`
    horas quedan en NaN y sus ventanas se descartan después.
    """
    columnas = columnas or COLUMNAS_EXOGENAS
    df = df.copy()
    faltaba = df[columnas].isna().any(axis=1)
    df[columnas] = df[columnas].ffill(limit=limite)
    df["flag_exogena_rellenada"] = (faltaba & df[columnas].notna().all(axis=1)).astype(int)
    return df


def limpiar(ruta_crudo, precio_extremo_es_error=True):
    """Pipeline completo. Devuelve (df_limpio, auditoria, resumen)."""
    df = cargar_crudo(ruta_crudo)
    resumen = {"filas_crudas": len(df)}

    df, resumen["duplicados_eliminados"] = eliminar_duplicados(df)
    df, faltantes = ordenar_y_verificar(df)
    resumen["horas_faltantes_en_rejilla"] = len(faltantes)
    df, resumen["periodo_dia_texto_corregido"], resumen["periodo_dia_incoherente_con_hora"] = \
        corregir_periodo_dia(df)

    resumen["nulos_originales"] = df.isna().sum()[lambda s: s > 0].to_dict()
    auditoria = clasificar_extremos(df, precio_extremo_es_error)
    df = aplicar_limpieza(df, auditoria)

    # La última hora no tiene objetivo (no existe t+1 en el dataset). No se elimina
    # la fila, porque sus X sí son válidas como contexto; solo no se usará como objetivo.
    resumen["objetivo_nulo"] = df.loc[df["demanda_objetivo"].isna(), "timestamp"].astype(str).tolist()
    resumen["filas_finales"] = len(df)
    return df, auditoria, resumen
