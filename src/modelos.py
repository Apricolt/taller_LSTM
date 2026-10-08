"""Las tres arquitecturas del taller (Keras)."""
import keras
from keras import layers


def lstm_base(n, f, unidades=64):
    """A. Una sola capa LSTM: el mínimo viable y punto de comparación."""
    return keras.Sequential([
        keras.Input((n, f)),
        layers.LSTM(unidades),
        layers.Dense(1),
    ], name=f"A_LSTM_base_{unidades}")


def lstm_profunda(n, f, dropout=0.2):
    """B. Tres LSTM apiladas. `return_sequences=True` entrega la secuencia completa a la
    capa siguiente (no solo el último estado), para que cada capa procese la serie entera."""
    capas = [keras.Input((n, f)), layers.LSTM(128, return_sequences=True)]
    if dropout: capas.append(layers.Dropout(dropout))
    capas.append(layers.LSTM(64, return_sequences=True))
    if dropout: capas.append(layers.Dropout(dropout))
    capas += [layers.LSTM(32), layers.Dense(16, activation="relu"), layers.Dense(1)]
    return keras.Sequential(capas, name="B_LSTM_profunda" + ("" if dropout else "_sin_dropout"))


def conv_lstm(n, f):
    """C. Conv1D + LSTM. Las convoluciones detectan patrones locales de pocas horas (rampas
    de subida o bajada) y la LSTM modela cómo evolucionan a lo largo de la ventana.
    padding='causal': cada salida solo mira la hora actual y las anteriores."""
    return keras.Sequential([
        keras.Input((n, f)),
        layers.Conv1D(64, 3, padding="causal", activation="relu"),
        layers.Conv1D(64, 3, padding="causal", activation="relu"),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(1),
    ], name="C_Conv1D_LSTM")


ARQUITECTURAS = {"A": lstm_base, "B": lstm_profunda, "C": conv_lstm}
NOMBRES = {"A": "A. LSTM base", "B": "B. LSTM profunda", "C": "C. Conv1D + LSTM"}
