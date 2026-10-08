"""Estilo común para todas las gráficas del taller (matplotlib)."""
import matplotlib.pyplot as plt

# Paleta categórica en orden fijo (validada para daltonismo en pares adyacentes).
AZUL, NARANJA, AQUA, AMARILLO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [AZUL, NARANJA, AQUA, AMARILLO, MAGENTA]
TINTA, TINTA_2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def aplicar_estilo():
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": REJILLA, "grid.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "text.color": TINTA,
        "lines.linewidth": 2, "lines.markersize": 8,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })

ROJO, GRIS_MEDIO = "#e34948", "#f0efec"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def cmap_secuencial():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)


def cmap_divergente():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [ROJO, GRIS_MEDIO, AZUL])
