"""Paleta y estilo común de las figuras del TP2.

Es la paleta del TP1 (tp1/src/graficos.py), la misma del deck: azul para lo que se compara contra,
naranja para lo que importa. `tests/test_paleta.py` verifica contraste y separación bajo las tres
formas de daltonismo. Importar este módulo aplica los rcParams.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

AZUL = "#2a78d6"
NARANJA = "#eb6834"
MAGENTA = "#a8447f"
AZUL_OSCURO = "#104281"

GRIS_REFERENCIA = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
SUPERFICIE = "#fcfcfb"

# Cada familia de colores se usa sólo en su tipo de figura: la clase en las del EDA, el conjunto en
# las curvas de validación y el modelo en las que comparan los cuatro clasificadores.
COLOR_CLASE = {"no": AZUL, "yes": NARANJA}
COLOR_CONJUNTO = {"train": AZUL, "validacion": NARANJA}
COLOR_MODELO = {"nb": MAGENTA, "svm": AZUL_OSCURO, "knn": AZUL, "rf": NARANJA}

# La línea de base es una sola en todo el TP, con un solo rótulo (guía de presentaciones, C1).
COLOR_LINEA_BASE = GRIS_REFERENCIA
ROTULO_LINEA_BASE = "sin modelo"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": EJE,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlecolor": TINTA,
        "text.color": TINTA,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "axes.grid": True,
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": EJE,
        "legend.facecolor": SUPERFICIE,
        "figure.dpi": 150,
    }
)
