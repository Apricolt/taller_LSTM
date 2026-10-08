"""Métricas de evaluación, siempre en MW (escala original)."""
import numpy as np


def metricas(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real, float).ravel(), np.asarray(y_pred, float).ravel()
    e = y_real - y_pred
    mse = np.mean(e ** 2)
    return {
        "MAE": np.mean(np.abs(e)),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAPE": 100 * np.mean(np.abs(e / y_real)),
        "R2": 1 - np.sum(e ** 2) / np.sum((y_real - y_real.mean()) ** 2),
    }


def metricas_picos(y_real, y_pred, percentil=95):
    """Métricas solo en las horas de mayor demanda (por defecto, el 5 % más alto)."""
    y_real, y_pred = np.asarray(y_real, float).ravel(), np.asarray(y_pred, float).ravel()
    m = y_real >= np.percentile(y_real, percentil)
    e = y_real[m] - y_pred[m]
    return {"MAE_picos": np.mean(np.abs(e)), "sesgo_picos": np.mean(-e)}  # sesgo < 0: subestima picos
