"""Correr con: python -m src.graficos

Figuras de análisis del TP2 (plan/PLAN.md §5: pasos 1.6, 3.4 y 4.5). Leen los CSV de resultados/,
en el formato largo del contrato (src/resultados.py; N0-4 y N0-8 de plan/EXEC_STATE.md), y
escriben PNG en figuras/:

  01-ablaciones.png              ΔAUC de validación de cada ablación contra la referencia A0, por
                                 modelo: media ± 1 desvío de las diferencias pareadas por fold. A1
                                 (duration) va en un panel propio porque su escala es otra
  01b-ablaciones-efecto.png      la misma medición como grilla variante × modelo, coloreada según
                                 el efecto del criterio del paso 1.5 (mejora, empeora o neutra)
  02-modelos-vs-linea-base.png   AUC y recall en el presupuesto de validación de cada clasificador,
                                 de mayor a menor AUC, contra «sin modelo»
  03-curva-<modelo>-<parametro>[-<sufijo>].png
                                 una por CSV de resultados/curvas/: train y validación con banda de
                                 ±1 desvío, el máximo de validación y la brecha en ese punto. Las
                                 curvas del mismo eje que difieren en un valor fijo (KNN con
                                 weights uniform y distance) van además juntas en una figura

Opciones: `--solo ablaciones|modelos|curvas` dibuja sólo ese grupo; `--resultados DIR` y
`--figuras DIR` cambian los directorios de entrada y de salida. Un archivo de entrada que todavía
no existe no es un error: se informa y se sigue con lo demás. Un archivo que existe pero no se
puede leer (vacío, cortado a mitad de una fila porque otro proceso todavía lo escribe) o dibujar
también se informa y se sigue con las demás figuras, pero el comando termina con código 1.

Las funciones `figura_*` son puras: reciben DataFrames y devuelven una Figure sin guardarla, así
tests/test_graficos.py las prueba con datos sintéticos. Las figuras de proyección del deck son
otras (src/graficos_presentacion.py, ola 7).
"""

import argparse
import io
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# Importar src.estilo fija el backend Agg y aplica los rcParams, así que va antes que pyplot.
from src.estilo import (
    AZUL,
    COLOR_CONJUNTO,
    COLOR_LINEA_BASE,
    COLOR_MODELO,
    GRIS_REFERENCIA,
    NARANJA,
    REJILLA,
    ROTULO_LINEA_BASE,
    SUPERFICIE,
    TINTA,
    TINTA_SECUNDARIA,
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import Bbox, offset_copy  # noqa: E402

from src.datos import RAIZ  # noqa: E402
from src.metricas import PRESUPUESTO  # noqa: E402
from src.resultados import CAMPOS, leer_largo, resumen  # noqa: E402

DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = RAIZ / "figuras"
DPI = 150
ALFA_BANDA = 0.16

FIGURA_ABLACIONES = "01-ablaciones.png"
FIGURA_EFECTO = "01b-ablaciones-efecto.png"
FIGURA_MODELOS = "02-modelos-vs-linea-base.png"
PREFIJO_CURVA = "03-curva"

RESUMEN_ABLACIONES = "ablaciones_resumen.csv"
RESUMEN_CV = "cv_referencia_resumen.csv"
LINEA_BASE = "linea_base.csv"
DIR_CURVAS = "curvas"
OMITIDOS = "omitidos.txt"

SIN_MODELO = "sin_modelo"
BASE = "A0"
MENOS = "\N{MINUS SIGN}"  # el signo de las marcas de matplotlib, no el guion

# --- Modelos ----------------------------------------------------------------------------------

ORDEN_MODELOS = ("nb_gaussiano", "nb_categorico", "svm", "knn", "rf")
ROTULO_MODELO = {
    "nb_gaussiano": "NB gaussiano",
    "nb_categorico": "NB categórico",
    "svm": "SVM",
    "knn": "KNN",
    "rf": "RF",
    SIN_MODELO: ROTULO_LINEA_BASE,
}
# Las dos variantes de Naive Bayes comparten el color de la familia y se distinguen por el
# marcador; los demás modelos también llevan el suyo, así la figura se lee sin color.
MARCADOR_MODELO = {"nb_gaussiano": "o", "nb_categorico": "D", "svm": "s", "knn": "^", "rf": "p"}

# --- Ablaciones -------------------------------------------------------------------------------

# Qué cambia cada variante respecto de A0, en pocas palabras. La descripción completa está en
# VARIANTES de src/ablaciones.py, y tests/test_graficos.py comprueba que las dos listas coincidan.
ROTULO_VARIANTE = {
    "A1": "con duration (techo)",
    "A2": "sin pdays",
    "A3": "default como indicadora",
    "A4": "funde categorías raras",
    "A5": "log1p(campaign)",
    "A6": "bloque macro reducido",
    "A7": "sin bloque macro",
    "A8": "sin bloque macro ni month",
    "A9": "sin day_of_week",
    "A10": "edad en tramos",
    "A11": "unknown → moda",
    "A12": "sin escalar",
}
APARTE = ("A1",)          # en otra escala: +0,06 a +0,21 contra |Δ| < 0,1 del resto
ANOTADAS = ("A1", "A7", "A8")
EFECTOS = ("mejora", "empeora", "neutra")
COLOR_EFECTO = {"mejora": AZUL, "empeora": NARANJA, "neutra": REJILLA}
TINTA_EFECTO = {"mejora": SUPERFICIE, "empeora": TINTA, "neutra": TINTA_SECUNDARIA}
ALTO_FILA = 0.6           # pulgadas por variante en la figura 01

# --- Curvas -----------------------------------------------------------------------------------

ESCALA_LOG = frozenset({"C", "n_neighbors", "n_estimators", "gamma"})
ROTULO_PARAMETRO = {
    "max_depth": "max_depth, profundidad máxima de cada árbol",
    "n_estimators": "n_estimators, número de árboles",
    "n_neighbors": "n_neighbors, número de vecinos k",
    "C": "C, un C mayor regulariza menos",
    "pesos_clase": "pesos de clase (class_weight)",
    "kernel": "kernel de la SVM",
    "gamma": "gamma del kernel",
    "weights": "weights, peso de cada vecino",
}
ROTULO_PUNTO = {
    ("max_depth", "None"): "sin límite",
    ("pesos_clase", "None"): "sin pesos",
    ("kernel", "linear"): "lineal",
    ("kernel", "poly"): "polinómico",
    ("kernel", "rbf"): "RBF",
}
# El valor con que src/modelos.crear_modelo arma cada modelo cuando la configuración no trae la
# clave: sin pesos de clase y, en RF, árboles sin límite de profundidad (el de scikit-learn). Sirve
# para rotular una curva cuyo JSON no trae ese valor fijo (rf_n_estimators no nombra max_depth) y
# una combinada cuyo valor fijo sólo figura en la otra.
DEFECTO_AUSENTE = {"rf": {"max_depth": None, "pesos_clase": None}, "svm": {"pesos_clase": None}}
# Hiperparámetros que el estimador no usa en cierta configuración: (clave que lo decide, valor,
# aclaración). Con kernel lineal, ni LinearSVC (implementacion="liblinear") ni SVC miran gamma
# (src/modelos._svm), así que nombrarlo entre los fijos sugeriría un efecto que no tiene.
NO_APLICA = {("svm", "gamma"): ("kernel", "linear", "no aplica con kernel lineal")}
# Claves que no distinguen curvas por sí mismas: implementacion="liblinear" acompaña siempre a
# kernel="linear" (src/curvas.py), así que nombrarla repetiría el kernel.
CLAVES_ACCESORIAS = frozenset({"implementacion"})
# El orden de las curvas de una figura combinada: primero la de la referencia (src/modelos.py).
ORDEN_SUFIJOS = ("uniform", "distance")
ESTILOS_LINEA = ("-", "--", ":", "-.")
MARCADORES_SERIE = ("o", "s", "D", "^")
# El mismo marcador como carácter, delante del valor del máximo en una figura combinada: así cada
# rótulo dice de qué curva es, con el marcador que la leyenda le asigna.
GLIFO_MARCADOR = {"o": "●", "s": "■", "D": "◆", "^": "▲"}
DESPLAZAMIENTO_CONJUNTO = {"train": -0.07, "validacion": 0.07}  # sólo en ejes categóricos
TAMANO_MAXIMO = 10.5    # puntos: el marcador relleno del máximo de validación
ESQUIVE_MAXIMO = 12.0   # puntos entre los máximos de dos curvas que caen en el mismo x
DISTANCIA_GUIA = 20.0   # puntos: un rótulo más lejos de su punto se une a él con una línea


# --- Formato ----------------------------------------------------------------------------------

def numero(valor, decimales=3, signo=False):
    """Coma decimal y el signo menos tipográfico, como las marcas de los ejes:
    numero(0.7681) -> '0,768'; numero(-0.0931, 3, signo=True) -> '−0,093'."""
    texto = format(float(valor), f"{'+' if signo else ''}.{decimales}f")
    return texto.replace("-", MENOS).replace(".", ",")


def delta(valor):
    """Un ΔAUC con signo: tres decimales desde 0,01 y cuatro por debajo, para que un efecto
    pequeño no se lea como cero."""
    return numero(valor, 3 if abs(valor) >= 0.01 else 4, signo=True)


def texto_valor(valor):
    """Un hiperparámetro como se lee en un rótulo: 300, 0,3, scale, None."""
    if valor is None or isinstance(valor, bool):
        return str(valor)
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    if isinstance(valor, (float, np.floating)):
        if float(valor).is_integer():
            return str(int(valor))
        return format(float(valor), "g").replace("-", MENOS).replace(".", ",")
    return str(valor)


def _decimales(valores, maximo=4):
    for d in range(maximo + 1):
        if all(abs(round(v, d) - v) < 1e-9 for v in valores):
            return d
    return maximo


def _marcas_con_coma(ax, eje="y"):
    """Fija las marcas que matplotlib elige para los límites actuales y las rotula con coma
    decimal y los decimales justos (0,50 y no 0.5)."""
    axis = ax.yaxis if eje == "y" else ax.xaxis
    lo, hi = sorted(ax.get_ylim() if eje == "y" else ax.get_xlim())
    tolerancia = 1e-9 * max(1.0, hi - lo)
    marcas = [v for v in axis.get_major_locator().tick_values(lo, hi)
              if lo - tolerancia <= v <= hi + tolerancia]
    d = _decimales(marcas)
    axis.set_major_locator(FixedLocator(marcas))
    axis.set_major_formatter(FuncFormatter(
        lambda v, _pos: numero(0.0 if abs(v) < tolerancia else v, d)))


def _texto_folds(n):
    return f"{int(n)} folds" if n and np.isfinite(n) else "los folds"


def color_modelo(modelo):
    return COLOR_MODELO.get("nb" if str(modelo).startswith("nb") else modelo, TINTA_SECUNDARIA)


def rotulo_modelo(modelo, en_eje=False):
    """El nombre de un modelo; en un eje, las dos variantes de NB en dos renglones."""
    rotulo = ROTULO_MODELO.get(modelo, str(modelo))
    return rotulo.replace(" ", "\n", 1) if en_eje and str(modelo).startswith("nb") else rotulo


def ordenar_modelos(modelos):
    def clave(m):
        return (ORDEN_MODELOS.index(m) if m in ORDEN_MODELOS else len(ORDEN_MODELOS), str(m))
    return sorted(set(modelos), key=clave)


def ordenar_variantes(variantes):
    """A1, A2, …, A10, A11, A12: por el número, no alfabéticamente."""
    def clave(v):
        m = re.fullmatch(r"A(\d+)", str(v))
        return (0, int(m.group(1)), "") if m else (1, 0, str(v))
    return sorted(set(variantes), key=clave)


def rotulo_variante(variante):
    corto = ROTULO_VARIANTE.get(variante)
    return f"{variante}  {corto}" if corto else str(variante)


# --- Figura 01: ablaciones (paso 1.6) ---------------------------------------------------------

def efecto(delta_media, delta_desvio):
    """El criterio mecánico del paso 1.5, el mismo que src/ablaciones.efecto: mejora o empeora
    sólo si la diferencia media supera el desvío de las diferencias pareadas."""
    if delta_media > delta_desvio:
        return "mejora"
    if delta_media < -delta_desvio:
        return "empeora"
    return "neutra"


def _tabla_ablaciones(tabla):
    """Las filas de ablaciones_resumen.csv que se dibujan: sin A0, con desvío y efecto."""
    faltan = [c for c in ("modelo", "variante", "delta_media", "delta_desvio")
              if c not in tabla.columns]
    if faltan:
        raise ValueError(f"al resumen de ablaciones le faltan columnas: {', '.join(faltan)}")
    t = tabla[tabla["variante"] != BASE].copy()
    if t.empty:
        raise ValueError("el resumen de ablaciones no tiene variantes distintas de A0")
    if t.duplicated(["modelo", "variante"]).any():
        raise ValueError("el resumen de ablaciones tiene más de una fila por modelo y variante")
    t["delta_desvio"] = t["delta_desvio"].fillna(0.0)
    if "efecto" not in t.columns:
        t["efecto"] = [efecto(m, d) for m, d in zip(t["delta_media"], t["delta_desvio"])]
    raros = sorted(set(t["efecto"]) - set(EFECTOS))
    if raros:
        raise ValueError(f"efectos desconocidos en el resumen de ablaciones: {', '.join(raros)}")
    return t


def _anotada(fila, anotadas):
    return fila.variante in anotadas or fila.efecto in ("mejora", "empeora")


def _rotulo_referencia(ax, afuera):
    """El rótulo de la línea vertical en 0: sobre el borde superior del panel o, si arriba va la
    leyenda, adentro, junto a la línea."""
    if afuera:
        ax.annotate("referencia", xy=(0, 1), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    color=GRIS_REFERENCIA, annotation_clip=False, gid="rotulo-referencia")
    else:
        ax.annotate("referencia", xy=(0, 1), xycoords=("data", "axes fraction"), xytext=(4, -3),
                    textcoords="offset points", ha="left", va="top", fontsize=9,
                    color=GRIS_REFERENCIA, gid="rotulo-referencia")


def _panel_ablaciones(ax, t, variantes, modelos, anotadas, referencia_afuera):
    fila_de = {v: i for i, v in enumerate(variantes)}
    paso = 0.8 / max(len(modelos), 1)
    desplazamiento = {m: (j - (len(modelos) - 1) / 2) * paso for j, m in enumerate(modelos)}

    for i in range(0, len(variantes), 2):
        ax.axhspan(i - 0.5, i + 0.5, color=REJILLA, alpha=0.4, lw=0, zorder=0)
    ax.axvline(0, color=GRIS_REFERENCIA, ls="--", lw=1.2, zorder=1, gid="referencia")
    _rotulo_referencia(ax, referencia_afuera)

    for f in t.itertuples(index=False):
        y = fila_de[f.variante] + desplazamiento[f.modelo]
        color = color_modelo(f.modelo)
        relleno = f.efecto in ("mejora", "empeora")
        barras = ax.errorbar(f.delta_media, y, xerr=f.delta_desvio,
                             fmt=MARCADOR_MODELO.get(f.modelo, "o"), ms=7, mew=1.4, color=color,
                             mfc=color if relleno else SUPERFICIE, ecolor=color, elinewidth=1.6,
                             capsize=3, capthick=1.4, zorder=3)
        barras.lines[0].set_gid(f"ablacion:{f.modelo}:{f.variante}")
        if _anotada(f, anotadas):
            derecha = f.delta_media >= 0
            x = f.delta_media + (f.delta_desvio if derecha else -f.delta_desvio)
            ax.annotate(delta(f.delta_media), xy=(x, y), xytext=(5 if derecha else -5, 0),
                        textcoords="offset points", ha="left" if derecha else "right",
                        va="center", fontsize=8.5, color=color, zorder=4,
                        gid=f"valor:{f.modelo}:{f.variante}")

    # Margen para los valores anotados, sólo del lado en que los hay.
    lo = min(0.0, float((t["delta_media"] - t["delta_desvio"]).min()))
    hi = max(0.0, float((t["delta_media"] + t["delta_desvio"]).max()))
    ancho = (hi - lo) or 0.01
    anotada = np.array([_anotada(f, anotadas) for f in t.itertuples(index=False)])
    negativa = (t["delta_media"] < 0).to_numpy()
    ax.set_xlim(lo - (0.11 if (anotada & negativa).any() else 0.04) * ancho,
                hi + (0.11 if (anotada & ~negativa).any() else 0.04) * ancho)
    _marcas_con_coma(ax, "x")

    ax.set_yticks(range(len(variantes)))
    ax.set_yticklabels([rotulo_variante(v) for v in variantes])
    ax.set_ylim(len(variantes) - 0.5, -0.5)  # A1 arriba
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)


def por_filas(manijas, ncol):
    """Reordena una leyenda para que se lea por filas: matplotlib la llena por columnas."""
    filas = -(-len(manijas) // ncol)
    return [manijas[f * ncol + c] for c in range(ncol) for f in range(filas)
            if f * ncol + c < len(manijas)]


def _leyenda_ablaciones(modelos):
    marcas = [Line2D([], [], ls="none", marker=MARCADOR_MODELO.get(m, "o"), ms=7,
                     color=color_modelo(m), label=rotulo_modelo(m)) for m in modelos]
    return marcas + [
        Line2D([], [], ls="none", marker="o", ms=7, color=TINTA_SECUNDARIA,
               label="relleno: mejora o empeora"),
        Line2D([], [], ls="none", marker="o", ms=7, mew=1.4, mfc=SUPERFICIE,
               color=TINTA_SECUNDARIA, label="hueco: neutra"),
        Line2D([], [], color=GRIS_REFERENCIA, ls="--", lw=1.2, label="referencia A0 (Δ = 0)"),
    ]


def figura_ablaciones(tabla, aparte=APARTE, anotadas=ANOTADAS):
    """Paso 1.6: el ΔAUC de validación de cada variante contra A0, por modelo.

    `tabla` es ablaciones_resumen.csv: una fila por modelo y variante distinta de A0, con
    delta_media, delta_desvio y efecto. Las variantes van de A1 arriba a A12 abajo, y los
    modelos, desplazados en vertical dentro de cada fila. Punto = ΔAUC medio; barra = ±1 desvío
    de las diferencias pareadas por fold; relleno si la variante mejora o empeora según el
    criterio del paso 1.5, hueco si es neutra. Las variantes de `aparte` van en un panel propio,
    arriba y con su escala, así nada queda fuera de los ejes; se anota el ΔAUC medio junto a cada
    punto de `anotadas` y de toda variante que mejora o empeora.
    """
    t = _tabla_ablaciones(tabla)
    variantes = ordenar_variantes(t["variante"])
    modelos = ordenar_modelos(t["modelo"])
    grupos = [g for g in ([v for v in variantes if v in aparte],
                          [v for v in variantes if v not in aparte]) if g]

    alto = ALTO_FILA * len(variantes) + 2.0 + 0.8 * (len(grupos) - 1)
    fig = plt.figure(figsize=(10.5, alto), layout="constrained")
    ejes = fig.subplots(len(grupos), 1, squeeze=False,
                        gridspec_kw={"height_ratios": [len(g) for g in grupos]})[:, 0]
    for i, (ax, grupo) in enumerate(zip(ejes, grupos)):
        _panel_ablaciones(ax, t[t["variante"].isin(grupo)], grupo, modelos, anotadas,
                          referencia_afuera=i > 0)
    if len(grupos) > 1:
        ejes[0].set_xlabel(f"ΔAUC de {', '.join(grupos[0])}, en su propia escala")
    ejes[-1].set_xlabel("ΔAUC de validación: variante − referencia")

    n = t["n_folds"].max() if "n_folds" in t.columns else None
    fig.suptitle("ΔAUC de validación de cada ablación contra la referencia A0")
    ncol = max(len(modelos), 3)
    ejes[0].legend(handles=por_filas(_leyenda_ablaciones(modelos), ncol), loc="lower center",
                   bbox_to_anchor=(0.5, 1.03), ncol=ncol, fontsize=9.5,
                   title=f"punto = media de las diferencias pareadas en {_texto_folds(n)}; "
                         "barra = ±1 desvío; relleno si |Δ| > desvío (paso 1.5)",
                   title_fontsize=9.5)
    return fig


def figura_ablaciones_efecto(tabla):
    """Paso 1.6, versión compacta: una celda por variante y modelo, azul si la variante mejora,
    naranja si empeora y gris si es neutra, con el ΔAUC medio y su desvío anotados. Las
    combinaciones que no se miden quedan en blanco con «no se mide»."""
    t = _tabla_ablaciones(tabla)
    variantes = ordenar_variantes(t["variante"])
    modelos = ordenar_modelos(t["modelo"])
    celdas = t.set_index(["variante", "modelo"])

    fig, ax = plt.subplots(figsize=(3.2 + 1.45 * len(modelos), 1.9 + 0.5 * len(variantes)),
                           layout="constrained")
    for i, v in enumerate(variantes):
        for j, m in enumerate(modelos):
            if (v, m) not in celdas.index:
                ax.text(j, i, "no se mide", ha="center", va="center", fontsize=8,
                        style="italic", color=GRIS_REFERENCIA, gid=f"sin-medir:{m}:{v}")
                continue
            f = celdas.loc[(v, m)]
            ax.add_patch(Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, lw=0,
                                   facecolor=COLOR_EFECTO[f["efecto"]],
                                   gid=f"celda:{m}:{v}:{f['efecto']}"))
            tinta = TINTA_EFECTO[f["efecto"]]
            ax.text(j, i - 0.1, delta(f["delta_media"]), ha="center", va="center", fontsize=10,
                    fontweight="bold", color=tinta, gid=f"valor:{m}:{v}")
            ax.text(j, i + 0.22, "± " + numero(f["delta_desvio"], 4), ha="center", va="center",
                    fontsize=7.5, color=tinta, gid=f"desvio:{m}:{v}")

    ax.set_xlim(-0.5, len(modelos) - 0.5)
    ax.set_ylim(len(variantes) - 0.5, -0.5)
    ax.set_xticks(range(len(modelos)))
    ax.set_xticklabels([rotulo_modelo(m) for m in modelos], fontsize=10.5, color=TINTA)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(variantes)))
    ax.set_yticklabels([rotulo_variante(v) for v in variantes])
    ax.tick_params(length=0)
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)

    fig.suptitle("Efecto de cada ablación sobre el AUC de validación, por modelo")
    fig.legend(handles=[Patch(facecolor=COLOR_EFECTO[e], label=e) for e in EFECTOS],
               loc="outside lower center", ncol=len(EFECTOS), fontsize=9.5,
               title="mejora o empeora: |ΔAUC medio| > desvío de las diferencias pareadas; "
                     "si no, neutra", title_fontsize=9)
    return fig


# --- Figura 02: los clasificadores contra «sin modelo» (paso 3.4) -----------------------------

METRICAS_MODELOS = ("auc", "recall_q")
LINEA_BASE_POR_DEFECTO = {"auc": 0.5, "recall_q": PRESUPUESTO}


def _titulo_metrica(metrica):
    if metrica == "recall_q":
        return f"Recall llamando al {numero(100 * PRESUPUESTO, 0)} % de la lista (recall_q)"
    return "AUC"


def _panel_modelos(ax, metrica, orden, medias, desvios, base):
    extremos = [base]
    for x, modelo in enumerate(orden):
        media = medias.at[modelo, metrica] if metrica in medias.columns else np.nan
        if pd.isna(media):
            continue
        desvio = desvios.at[modelo, metrica] if metrica in desvios.columns else np.nan
        desvio = 0.0 if pd.isna(desvio) else float(desvio)
        barras = ax.errorbar(x, media, yerr=desvio, fmt=MARCADOR_MODELO.get(modelo, "o"), ms=9,
                             color=color_modelo(modelo), elinewidth=1.8, capsize=5,
                             capthick=1.8, zorder=3)
        barras.lines[0].set_gid(f"modelo:{modelo}:{metrica}")
        ax.annotate(numero(media, 3), xy=(x, media), xytext=(11, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10.5, color=TINTA,
                    gid=f"valor:{modelo}:{metrica}")
        extremos += [media - desvio, media + desvio]

    ax.axhline(base, color=COLOR_LINEA_BASE, ls=":", lw=1.8, zorder=2, gid="linea-base")
    ax.annotate(ROTULO_LINEA_BASE, xy=(0, base), xycoords=("axes fraction", "data"),
                xytext=(6, 4), textcoords="offset points", ha="left", va="bottom", fontsize=10,
                color=COLOR_LINEA_BASE, gid="rotulo-linea-base")

    lo, hi = min(extremos), max(extremos)
    margen = 0.12 * ((hi - lo) or 0.1)
    ax.set_ylim(lo - margen, hi + margen)
    _marcas_con_coma(ax, "y")
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([rotulo_modelo(m, en_eje=True) for m in orden])
    ax.set_xlim(-0.5, len(orden) - 0.25)
    ax.grid(axis="x", visible=False)
    ax.set_title(_titulo_metrica(metrica), loc="left")
    ax.set_ylabel("media ± 1 desvío entre folds")


def figura_modelos(tabla, linea_base=None, conjunto="validacion"):
    """Paso 3.4: AUC y recall_q de validación de cada clasificador, punto = media y barra = ±1
    desvío entre folds, de mayor a menor AUC (el mismo orden en los dos paneles), con el valor
    anotado y «sin modelo» como línea horizontal punteada.

    `tabla` es cv_<etiqueta>_resumen.csv de src/experimentos.py: modelo, conjunto, metrica,
    media, desvio (y n). La línea de base sale de sus filas `sin_modelo`; si no las tiene, de
    `linea_base` ({metrica: valor}) y, si tampoco, de AUC 0,5 y recall_q = q.
    """
    faltan = [c for c in ("modelo", "conjunto", "metrica", "media", "desvio")
              if c not in tabla.columns]
    if faltan:
        raise ValueError("al resumen de validación cruzada le faltan columnas: "
                         + ", ".join(faltan))
    r = tabla[tabla["conjunto"] == conjunto]
    if "etiqueta" in r.columns and r["etiqueta"].nunique() > 1:
        raise ValueError("el resumen mezcla etiquetas: "
                         + ", ".join(map(str, r["etiqueta"].unique())))

    base = {**LINEA_BASE_POR_DEFECTO, **(linea_base or {})}
    sin = r[r["modelo"] == SIN_MODELO].groupby("metrica")["media"].mean()
    base.update({m: float(v) for m, v in sin.items() if m in base and pd.notna(v)})

    r = r[(r["modelo"] != SIN_MODELO) & r["metrica"].isin(METRICAS_MODELOS)]
    repetidos = r[r.duplicated(["modelo", "metrica"], keep=False)]["modelo"].unique()
    if len(repetidos):
        raise ValueError(f"más de una configuración de {', '.join(map(str, repetidos))} en el "
                         "resumen")
    medias = r.pivot(index="modelo", columns="metrica", values="media")
    desvios = r.pivot(index="modelo", columns="metrica", values="desvio")
    if "auc" not in medias.columns or medias["auc"].isna().all():
        raise ValueError(f"el resumen no tiene el AUC de {conjunto} de ningún modelo")

    def clave(m):
        auc = medias.at[m, "auc"]
        posicion = ORDEN_MODELOS.index(m) if m in ORDEN_MODELOS else len(ORDEN_MODELOS)
        return (-auc if pd.notna(auc) else np.inf, posicion, str(m))

    orden = sorted(medias.index, key=clave)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metrica in zip(ejes, METRICAS_MODELOS):
        _panel_modelos(ax, metrica, orden, medias, desvios, base[metrica])
    n = r["n"].max() if "n" in r.columns else None
    fig.suptitle(f"Los clasificadores contra «{ROTULO_LINEA_BASE}», de mayor a menor AUC "
                 f"(validación cruzada, {_texto_folds(n)})")
    return fig


# --- Figuras 03: curvas de validación (paso 4.5) ----------------------------------------------

def _texto_punto(punto):
    if punto is None or (isinstance(punto, float) and np.isnan(punto)):
        return "None"
    if isinstance(punto, (float, np.floating)) and float(punto).is_integer():
        return str(int(punto))
    return str(punto)


def puntos_de(tabla):
    """El valor del parámetro en cada fila, como texto ("2", "None", "0.001", "balanced").

    Es la misma reconstrucción que src/curvas.leer_curvas: pandas lee el texto «None» como NaN
    (y convierte "2" en 2.0), así que el valor se toma del JSON de `configuracion`, que conserva
    su tipo; si no está ahí, de la columna `punto`."""
    parametros = tabla["parametro"].dropna().unique()
    if len(parametros) != 1:
        raise ValueError("una curva tiene un solo parámetro")
    parametro = str(parametros[0])
    leidas = {}

    def valor(configuracion, punto):
        if configuracion not in leidas:
            try:
                leidas[configuracion] = json.loads(configuracion)
            except (TypeError, ValueError):
                leidas[configuracion] = None
        d = leidas[configuracion]
        if isinstance(d, dict) and parametro in d:
            return str(d[parametro])
        return _texto_punto(punto)

    return [valor(c, p) for c, p in zip(tabla["configuracion"], tabla["punto"])]


def identidad_curva(tabla, nombre=None):
    """(modelo, parametro, sufijo) de una curva. El sufijo sale del nombre del archivo,
    <modelo>_<parametro>[_<sufijo>], como lo arma src/curvas.nombre_curva."""
    modelos = tabla["modelo"].dropna().unique()
    parametros = tabla["parametro"].dropna().unique()
    if len(modelos) != 1 or len(parametros) != 1:
        raise ValueError("una curva tiene un solo modelo y un solo parámetro")
    modelo, parametro = str(modelos[0]), str(parametros[0])
    prefijo = f"{modelo}_{parametro}"
    sufijo = ""
    if nombre and str(nombre).startswith(prefijo):
        sufijo = str(nombre)[len(prefijo):].lstrip("_")
    return modelo, parametro, sufijo


def resumen_curva(tabla, metrica="auc"):
    """Media, desvío y n de `metrica` por punto y conjunto (resumen() del contrato), una fila por
    punto en el orden del CSV: media_train, media_validacion, desvio_train, …"""
    t = tabla.assign(punto=puntos_de(tabla))
    t = t[t["metrica"] == metrica]
    if t.empty:
        raise ValueError(f"la curva no tiene la métrica {metrica}")
    orden = list(dict.fromkeys(t["punto"]))
    r = resumen(t, por=["punto", "conjunto"])
    ancho = r.pivot(index="punto", columns="conjunto", values=["media", "desvio", "n"])
    ancho.columns = [f"{estadistico}_{conjunto}" for estadistico, conjunto in ancho.columns]
    return ancho.reindex(orden)


def configuraciones_de(tabla):
    """Las configuraciones distintas de una curva, como dicts, desde el JSON de `configuracion`."""
    configuraciones = []
    for texto in tabla["configuracion"].dropna().unique():
        try:
            d = json.loads(texto)
        except (TypeError, ValueError):
            continue
        if isinstance(d, dict):
            configuraciones.append(d)
    return configuraciones


def fijos_de(tabla):
    """Los hiperparámetros que no cambian a lo largo de la curva, fuera del que la recorre."""
    parametro = identidad_curva(tabla)[1]
    configuraciones = configuraciones_de(tabla)
    if not configuraciones:
        return {}
    comunes = set.intersection(*(set(c) for c in configuraciones)) - {parametro}
    return {k: configuraciones[0][k] for k in sorted(comunes)
            if len({json.dumps(c[k], sort_keys=True) for c in configuraciones}) == 1}


def _como_numero(texto):
    try:
        valor = float(texto)
    except (TypeError, ValueError):
        return None
    return valor if np.isfinite(valor) else None


def rotulo_punto(parametro, punto):
    if (parametro, punto) in ROTULO_PUNTO:
        return ROTULO_PUNTO[(parametro, punto)]
    if punto == "None":
        return "ninguno"
    valor = _como_numero(punto)
    return punto if valor is None else texto_valor(valor)


def eje_x(parametro, puntos):
    """Dónde va cada punto de una curva en el eje x.

    Devuelve un dict con `tipo` («log», «lineal» o «categorico»), `puntos` en el orden del eje,
    `posiciones` y `rotulos`. Un eje es numérico si todos los puntos salvo "None" son números (y
    hay al menos dos); es logarítmico si además el parámetro está en ESCALA_LOG (C, n_neighbors,
    n_estimators, gamma). "None" (max_depth sin límite) va en el extremo derecho, un paso más allá
    del último número. Si no, el eje es categórico (pesos_clase, kernel, weights) y cada punto va
    en un entero, en el orden en que llega."""
    puntos = list(dict.fromkeys(puntos))
    otros = [p for p in puntos if p != "None"]
    valores = [_como_numero(p) for p in otros]
    if len(otros) < 2 or any(v is None for v in valores):
        return {"tipo": "categorico", "puntos": puntos,
                "posiciones": [float(i) for i in range(len(puntos))],
                "rotulos": [rotulo_punto(parametro, p) for p in puntos]}
    pares = sorted(zip(valores, otros))
    log = parametro in ESCALA_LOG and pares[0][0] > 0
    posiciones = [v for v, _ in pares]
    ordenados = [p for _, p in pares]
    if "None" in puntos:
        u = np.log10(posiciones) if log else np.asarray(posiciones, float)
        extra = u[-1] + max(u[-1] - u[-2], 0.1 * (u[-1] - u[0]))
        posiciones.append(float(10 ** extra) if log else float(extra))
        ordenados.append("None")
    return {"tipo": "log" if log else "lineal", "puntos": ordenados, "posiciones": posiciones,
            "rotulos": [rotulo_punto(parametro, p) for p in ordenados]}


def _serie(nombre, tabla, metrica):
    modelo, parametro, sufijo = identidad_curva(tabla, nombre)
    return {"nombre": str(nombre or f"{modelo}_{parametro}"), "modelo": modelo,
            "parametro": parametro, "sufijo": sufijo, "resumen": resumen_curva(tabla, metrica),
            "fijos": fijos_de(tabla), "configuraciones": configuraciones_de(tabla)}


def _x(eje, punto, desplazamiento=0.0):
    return eje["posiciones"][eje["puntos"].index(punto)] + desplazamiento


def _dibujar_serie(ax, serie, eje, estilo, marcador, desplazamiento):
    """Train y validación de una curva: línea con banda de ±1 desvío en un eje numérico; puntos
    con barras de error, sin unir, en uno categórico."""
    r = serie["resumen"]
    presentes = [p for p in eje["puntos"] if p in r.index]
    numericos = [p for p in presentes if p != "None"]
    for conjunto in ("train", "validacion"):
        columna = f"media_{conjunto}"
        if columna not in r.columns or r[columna].isna().all():
            continue
        color = COLOR_CONJUNTO[conjunto]
        desvio = r[f"desvio_{conjunto}"].fillna(0.0)
        gid = f"serie:{serie['nombre']}:{conjunto}"
        comunes = dict(color=color, marker=marcador, ms=5.5, mew=1.3, mfc=SUPERFICIE)
        if eje["tipo"] == "categorico":
            d = desplazamiento + DESPLAZAMIENTO_CONJUNTO[conjunto]
            barras = ax.errorbar([_x(eje, p, d) for p in presentes], r.loc[presentes, columna],
                                 yerr=desvio.loc[presentes], ls="none", ms=7, mew=1.4,
                                 mfc=SUPERFICIE, marker=marcador, color=color, elinewidth=1.4,
                                 capsize=4, zorder=3)
            barras.lines[0].set_gid(gid)
            continue
        x = [_x(eje, p) for p in numericos]
        m = r.loc[numericos, columna].to_numpy(float)
        s = desvio.loc[numericos].to_numpy(float)
        ax.plot(x, m, ls=estilo, lw=1.8, zorder=3, gid=gid, **comunes)
        ax.fill_between(x, m - s, m + s, color=color, alpha=ALFA_BANDA, lw=0, zorder=1,
                        gid=f"banda:{serie['nombre']}:{conjunto}")
        if "None" in presentes:
            # El salto hasta "sin límite" no es una distancia: se une con un trazo punteado y el
            # desvío va como barra, no como banda.
            xn, mn, sn = _x(eje, "None"), r.at["None", columna], desvio.at["None"]
            if numericos:
                ax.plot([x[-1], xn], [m[-1], mn], ls=":", lw=1.4, color=color, zorder=2)
            barras = ax.errorbar(xn, mn, yerr=sn, ls="none", elinewidth=1.2, capsize=3,
                                 zorder=3, **comunes)
            barras.lines[0].set_gid(f"punto-none:{serie['nombre']}:{conjunto}")


def _densificar(xy, paso=2.0):
    """Los vértices de una polilínea en píxeles, con puntos intermedios cada `paso` píxeles."""
    xy = np.asarray(xy, float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return xy
    partes = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / paso)))
        partes.append(a + np.linspace(0, 1, n + 1)[1:, None] * (b - a))
    return np.vstack(partes)


def _obstaculos(ax, renderer):
    """Lo que un rótulo no debe tapar: los trazos de las líneas, de las barras de error y de las
    líneas guía (como puntos en píxeles), las cajas de los marcadores y las de los textos ya
    puestos. Las bandas no cuentan: son translúcidas."""
    trazos, marcas, textos = [], [], []
    for linea in ax.lines:
        if not linea.get_visible():
            continue
        xy = linea.get_transform().transform(np.asarray(linea.get_xydata(), float))
        if linea.get_linestyle() not in ("None", "none", "", " "):
            trazos.append(_densificar(xy))
        if linea.get_marker() not in (None, "None", "none", "", " "):
            radio = renderer.points_to_pixels(linea.get_markersize()) / 2 + 1
            marcas += [Bbox.from_extents(x - radio, y - radio, x + radio, y + radio)
                       for x, y in xy if np.isfinite(x) and np.isfinite(y)]
    for coleccion in ax.collections:
        if isinstance(coleccion, LineCollection):
            transformacion = coleccion.get_transform()
            trazos += [_densificar(transformacion.transform(np.asarray(s, float)))
                       for s in coleccion.get_segments()]
    for t in ax.texts:
        if not (t.get_visible() and t.get_text()):
            continue
        # La extensión de una anotación incluye su línea guía: la caja es la del texto, y la guía
        # cuenta como trazo. Pedir primero la extensión completa fija la posición de las dos.
        t.get_window_extent(renderer)
        textos.append(Text.get_window_extent(t, renderer))
        guia = getattr(t, "arrow_patch", None)
        if guia is not None:
            trazos.append(_densificar(
                guia.get_transform().transform_path(guia.get_path()).vertices))
    puntos = np.vstack(trazos) if trazos else np.empty((0, 2))
    return puntos, marcas, textos


def _costo(caja, marco, obstaculos):
    """Cuánto estorba un rótulo en `caja`. Salir de los ejes pesa más que todo; después, tapar
    otro texto (dos rótulos encimados no se leen); después, cada marcador tapado y cada punto de
    trazo, que el fondo del rótulo atenúa. Cero es un lugar libre."""
    puntos, marcas, textos = obstaculos
    afuera = (max(0.0, marco.x0 - caja.x0) + max(0.0, caja.x1 - marco.x1)
              + max(0.0, marco.y0 - caja.y0) + max(0.0, caja.y1 - marco.y1))
    tapados = 0
    if len(puntos):
        tapados = int(((puntos[:, 0] >= caja.x0) & (puntos[:, 0] <= caja.x1)
                       & (puntos[:, 1] >= caja.y0) & (puntos[:, 1] <= caja.y1)).sum())
    return (10_000 * (afuera > 0) + 5 * afuera + 1_000 * sum(caja.overlaps(t) for t in textos)
            + 15 * sum(caja.overlaps(m) for m in marcas) + tapados)


def _colocar(ax, crear, candidatos):
    """Pone uno o más rótulos con `crear(xy, dx, dy, ha, va)`, que los crea y los devuelve, en el
    primer candidato (xy en datos, desplazamiento en puntos) que queda dentro de los ejes sin
    tapar líneas, marcadores ni textos; si ninguno queda libre, en el que menos estorba. Devuelve
    los rótulos y el candidato elegido. Exige el diseño de la figura ya fijo: los lugares se miden
    en píxeles."""
    renderer = ax.figure.canvas.get_renderer()
    marco = ax.get_window_extent(renderer)
    obstaculos = _obstaculos(ax, renderer)
    elegidos, elegido, menor = None, None, None
    for candidato in candidatos:
        textos = crear(*candidato)
        caja = Bbox.union([t.get_window_extent(renderer) for t in textos]).padded(2)
        costo = _costo(caja, marco, obstaculos)
        if menor is None or costo < menor:
            for t in elegidos or ():
                t.remove()
            elegidos, elegido, menor = textos, candidato, costo
            if costo == 0:
                break
        else:
            for t in textos:
                t.remove()
    return elegidos, elegido


def _caja_del_texto(anotacion):
    """Coordenadas para anclar un texto a la caja de otro: la del texto solo, porque la extensión
    de una anotación incluye su línea guía, y anclarse a ella correría el segundo renglón."""
    def caja(renderer):
        anotacion.get_window_extent(renderer)  # fija la posición de la anotación
        return Text.get_window_extent(anotacion, renderer)
    return caja


def _rotulos_maximo(ax, valor, brecha, gid, coordenadas):
    """Crea el rótulo del máximo: el valor en negrita y, debajo, la brecha, como un bloque que
    se ubica entero respecto del ancla según `va`. El ancla está en `coordenadas`: los datos,
    corridos como el punto si el máximo se esquivó. Con `guia`, el texto anclado lleva una línea
    fina hasta el borde del punto (para un bloque que quedó lejos de él)."""
    caja = dict(boxstyle="round,pad=0.12", fc=SUPERFICIE, ec="none", alpha=0.85)
    primero = dict(fontsize=10, fontweight="bold", color=COLOR_CONJUNTO["validacion"], bbox=caja,
                   gid=f"valor-maximo:{gid}")
    segundo = dict(fontsize=9, color=TINTA_SECUNDARIA, bbox=caja, gid=f"valor-brecha:{gid}")

    def crear(xy, dx, dy, ha, va, guia=False):
        comunes = dict(textcoords="offset points", ha=ha, zorder=7)
        ancla = dict(xy=xy, xycoords=coordenadas)
        if guia:
            ancla["arrowprops"] = dict(arrowstyle="-", color=TINTA_SECUNDARIA, lw=0.8,
                                       shrinkA=1, shrinkB=TAMANO_MAXIMO / 2 + 1.5)
        if brecha is None:
            return [ax.annotate(valor, xytext=(dx, dy), va=va, **ancla, **comunes, **primero)]
        fx = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]
        if va == "bottom":  # el bloque crece hacia arriba desde el ancla
            abajo = ax.annotate(brecha, xytext=(dx, dy), va="bottom", **ancla, **comunes,
                                **segundo)
            arriba = ax.annotate(valor, xy=(fx, 1), xycoords=_caja_del_texto(abajo),
                                 xytext=(0, 1), va="bottom", **comunes, **primero)
        elif va == "top":  # hacia abajo
            arriba = ax.annotate(valor, xytext=(dx, dy), va="top", **ancla, **comunes, **primero)
            abajo = ax.annotate(brecha, xy=(fx, 0), xycoords=_caja_del_texto(arriba),
                                xytext=(0, -1), va="top", **comunes, **segundo)
        else:  # centrado en el ancla
            arriba = ax.annotate(valor, xytext=(dx, dy + 1), va="bottom", **ancla, **comunes,
                                 **primero)
            abajo = ax.annotate(brecha, xy=xy, xycoords=coordenadas, xytext=(dx, dy - 1),
                                va="top", **comunes, **segundo)
        return [arriba, abajo]

    return crear


def _candidatos_maximo(x, va):
    """Alrededor del punto, de cerca a lejos: abajo primero, que es donde la curva de validación
    deja lugar. Los lejanos (desde DISTANCIA_GUIA puntos) llevan una línea guía al punto, así que
    sirven cuando el entorno está lleno (varias curvas con el máximo en el mismo x)."""
    p = (x, va)
    return [(p, 0, -11, "center", "top"), (p, 11, -8, "left", "top"),
            (p, -11, -8, "right", "top"), (p, 0, 11, "center", "bottom"),
            (p, 11, 8, "left", "bottom"), (p, -11, 8, "right", "bottom"),
            (p, 14, 0, "left", "center"), (p, -14, 0, "right", "center"),
            (p, 0, -26, "center", "top"), (p, 0, 26, "center", "bottom"),
            (p, 26, -8, "left", "top"), (p, -26, -8, "right", "top"),
            (p, 26, 8, "left", "bottom"), (p, -26, 8, "right", "bottom"),
            (p, 26, -26, "left", "top"), (p, -26, -26, "right", "top"),
            (p, 26, 26, "left", "bottom"), (p, -26, 26, "right", "bottom"),
            (p, 0, -42, "center", "top"), (p, 0, 42, "center", "bottom"),
            (p, 42, -8, "left", "top"), (p, -42, -8, "right", "top"),
            (p, 42, 8, "left", "bottom"), (p, -42, 8, "right", "bottom")]


def maximo_de(serie):
    """El punto de mayor AUC medio de validación de una curva, o None si no tiene validación."""
    r = serie["resumen"]
    if "media_validacion" not in r.columns or r["media_validacion"].isna().all():
        return None
    return r["media_validacion"].idxmax()


def esquives(mejores, categorico):
    """Cuántos puntos se corre en horizontal el máximo de cada curva de una figura: nada, salvo que
    varias curvas lo tengan en el mismo x de un eje numérico (en la de SVM, C = 0,01 con RBF y con
    kernel lineal); entonces se reparten a ESQUIVE_MAXIMO puntos entre sí, centradas en ese x. En
    un eje categórico cada curva ya tiene su propio x."""
    corrimientos = [0.0] * len(mejores)
    if categorico:
        return corrimientos
    grupos = {}
    for i, mejor in enumerate(mejores):
        if mejor is not None:
            grupos.setdefault(mejor, []).append(i)
    for indices in grupos.values():
        for j, i in enumerate(indices):
            corrimientos[i] = (j - (len(indices) - 1) / 2) * ESQUIVE_MAXIMO
    return corrimientos


def _marcar_maximo(ax, serie, eje, mejor, marcador, desplazamiento, corrimiento=0.0, glifo=None):
    """El máximo de validación (`mejor`, un punto del eje) con un punto relleno, y la brecha
    train − validación en ese punto como un segmento vertical; junto al punto, su valor y, debajo,
    la brecha. `corrimiento` (en puntos) corre juntos el punto, el segmento y el rótulo cuando otra
    curva tiene el máximo en el mismo x, para que ninguno tape al otro. `glifo`, el marcador de la
    curva como carácter, va delante del valor en una figura combinada. Si el rótulo queda lejos
    del punto, una línea fina los une."""
    r = serie["resumen"]
    x = _x(eje, mejor, desplazamiento + (DESPLAZAMIENTO_CONJUNTO["validacion"]
                                          if eje["tipo"] == "categorico" else 0.0))
    media_va = float(r.at[mejor, "media_validacion"])
    color = COLOR_CONJUNTO["validacion"]
    corrida = offset_copy(ax.transData, fig=ax.figure, x=corrimiento, y=0, units="points")
    ax.plot([x], [media_va], ls="none", marker=marcador, ms=TAMANO_MAXIMO, mfc=color,
            mec=SUPERFICIE, mew=1.2, color=color, zorder=6, transform=corrida,
            gid=f"maximo:{serie['nombre']}")
    media_tr = r.at[mejor, "media_train"] if "media_train" in r.columns else np.nan
    brecha = None
    if pd.notna(media_tr):
        ax.plot([x, x], [media_va, media_tr], color=TINTA_SECUNDARIA, lw=1.1, marker="_", ms=9,
                mew=1.1, zorder=4, transform=corrida, gid=f"brecha:{serie['nombre']}")
        brecha = f"brecha {numero(media_tr - media_va, 3, signo=True)}"
    valor = numero(media_va, 3) if glifo is None else f"{glifo} {numero(media_va, 3)}"
    crear = _rotulos_maximo(ax, valor, brecha, serie["nombre"], corrida)
    textos, (xy, dx, dy, ha, va) = _colocar(ax, crear, _candidatos_maximo(x, media_va))
    if np.hypot(dx, dy) >= DISTANCIA_GUIA:
        for t in textos:
            t.remove()
        crear(xy, dx, dy, ha, va, guia=True)


def _limites_x(ax, eje):
    posiciones = np.asarray(eje["posiciones"], float)
    if eje["tipo"] == "categorico":
        ax.set_xlim(-0.6, len(posiciones) - 0.4)
    elif eje["tipo"] == "log":
        u = np.log10(posiciones)
        margen = 0.04 * ((u[-1] - u[0]) or 1.0)
        ax.set_xlim(10 ** (u[0] - margen), 10 ** (u[-1] + margen))
    else:
        margen = 0.04 * ((posiciones[-1] - posiciones[0]) or 1.0)
        ax.set_xlim(posiciones[0] - margen, posiciones[-1] + margen)


def _limites_y(ax, series):
    valores = []
    for s in series:
        r = s["resumen"]
        for conjunto in ("train", "validacion"):
            if f"media_{conjunto}" in r.columns:
                m = r[f"media_{conjunto}"]
                d = r[f"desvio_{conjunto}"].fillna(0.0)
                valores += list((m - d).dropna()) + list((m + d).dropna())
    lo, hi = min(valores), max(valores)
    alto = (hi - lo) or 0.02
    ax.set_ylim(lo - 0.14 * alto, hi + 0.1 * alto)
    _marcas_con_coma(ax, "y")


def _marcas_x(ax, eje):
    if eje["tipo"] == "log":
        ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(eje["posiciones"]))
    rotulos = eje["rotulos"]
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: rotulos[pos] if pos is not None and pos < len(rotulos) else ""))
    ax.xaxis.set_minor_locator(NullLocator())
    if eje["tipo"] == "categorico":
        ax.grid(axis="x", visible=False)


def _marcas_superpuestas(ax, holgura=6):
    """Si dos rótulos vecinos del eje x quedan a menos de `holgura` píxeles (KNN, con k hasta
    801 en escala logarítmica)."""
    renderer = ax.figure.canvas.get_renderer()
    cajas = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()]
    return any(a.x1 + holgura > b.x0 for a, b in zip(cajas, cajas[1:]))


def _resaltar_marca(ax, eje, punto):
    """El rótulo del eje x del punto elegido, en negrita y del color de validación."""
    rotulos = ax.get_xticklabels()
    i = eje["puntos"].index(punto)
    if i < len(rotulos):
        rotulos[i].set_fontweight("bold")
        rotulos[i].set_color(COLOR_CONJUNTO["validacion"])


def _leyenda_curvas(ax, series, rotulos, categorico):
    val = COLOR_CONJUNTO["validacion"]
    trazo = dict(ls="none") if categorico else dict(lw=1.8)
    manijas = [Line2D([], [], color=COLOR_CONJUNTO["train"], marker="o", ms=5.5, mfc=SUPERFICIE,
                      label="train", **trazo),
               Line2D([], [], color=val, marker="o", ms=5.5, mfc=SUPERFICIE, label="validación",
                      **trazo)]
    if len(series) > 1:
        manijas += [Line2D([], [], color=TINTA_SECUNDARIA, ms=5.5,
                           marker=MARCADORES_SERIE[i % len(MARCADORES_SERIE)],
                           mfc=SUPERFICIE, label=rotulos[i],
                           **(dict(ls="none") if categorico
                              else dict(ls=ESTILOS_LINEA[i % len(ESTILOS_LINEA)], lw=1.5)))
                    for i in range(len(series))]
    if not categorico:  # en un eje categórico el desvío va como barra de error
        manijas.append(Patch(facecolor=TINTA_SECUNDARIA, alpha=ALFA_BANDA * 1.6,
                             label="±1 desvío entre folds"))
    manijas += [Line2D([], [], ls="none", marker="o", ms=9, mfc=val, mec=SUPERFICIE, color=val,
                       label="máximo de validación"),
                Line2D([], [], ls="none", marker="|", ms=13, mew=1.3, color=TINTA_SECUNDARIA,
                       label="brecha train − validación\nen el máximo")]
    ax.legend(handles=manijas, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              fontsize=9.5)


_SIN_VALOR = object()


def valor_fijo(serie, clave):
    """El valor fijo de `clave` en una curva: el de su JSON o, si ninguna configuración trae la
    clave, el que usa el modelo por omisión (DEFECTO_AUSENTE); _SIN_VALOR si no hay ninguno de los
    dos (la clave cambia a lo largo de la curva, o no tiene valor conocido)."""
    if clave in serie["fijos"]:
        return serie["fijos"][clave]
    defectos = DEFECTO_AUSENTE.get(serie["modelo"], {})
    if clave in defectos and all(clave not in c for c in serie["configuraciones"]):
        return defectos[clave]
    return _SIN_VALOR


def rotulo_fijo(clave, serie):
    """Un hiperparámetro fijo de una curva como se lee en un rótulo: «max_depth = 8»,
    «max_depth = sin límite», «pesos_clase = sin pesos»."""
    valor = valor_fijo(serie, clave)
    if valor is _SIN_VALOR:
        return f"{clave} por omisión"
    return f"{clave} = {rotulo_punto(clave, 'None' if valor is None else str(valor))}"


def _clave_valor(serie, clave):
    valor = valor_fijo(serie, clave)
    return "sin valor" if valor is _SIN_VALOR else json.dumps(valor, sort_keys=True, default=str)


def _rotulos_series(series):
    """Cómo se llama cada curva en una figura combinada: por los valores fijos en que difieren
    (weights = uniform) o, si no se encuentran, por el sufijo del archivo."""
    claves = set().union(*(s["fijos"] for s in series)) - CLAVES_ACCESORIAS
    distintas = sorted(k for k in claves if len({_clave_valor(s, k) for s in series}) > 1)
    rotulos = [", ".join(rotulo_fijo(k, s) for k in distintas) or s["sufijo"] or s["nombre"]
               for s in series]
    return rotulos, distintas


def texto_fijos(series, distintas=()):
    """Los hiperparámetros fijos comunes a todas las curvas de una figura, como se leen en el
    subtítulo: los de su JSON y, si ninguna configuración trae la clave, el valor por omisión del
    modelo («max_depth = sin límite» en rf_n_estimators). Uno que el estimador no usa en ningún
    punto no se nombra (gamma con kernel lineal); si lo ignora en algunos, se aclara."""
    modelo, parametro = series[0]["modelo"], series[0]["parametro"]
    claves = ((set().union(*(s["fijos"] for s in series)) | set(DEFECTO_AUSENTE.get(modelo, {})))
              - set(distintas) - {parametro})
    configuraciones = [c for s in series for c in s["configuraciones"]]
    textos = []
    for clave in sorted(claves):
        valores = {_clave_valor(s, clave) for s in series}
        if len(valores) != 1 or "sin valor" in valores:
            continue
        regla = NO_APLICA.get((modelo, clave))
        ignorada = [regla is not None and c.get(regla[0]) == regla[1] for c in configuraciones]
        if ignorada and all(ignorada):
            continue
        textos.append(rotulo_fijo(clave, series[0]) + (f" ({regla[2]})" if any(ignorada) else ""))
    return textos


def _figura_curvas(series, omitidos=()):
    modelo, parametro = series[0]["modelo"], series[0]["parametro"]
    if any((s["modelo"], s["parametro"]) != (modelo, parametro) for s in series):
        raise ValueError("las curvas de una figura comparten modelo y parámetro")
    eje = eje_x(parametro, [p for s in series for p in s["resumen"].index])
    rotulos, distintas = _rotulos_series(series)
    varias = len(series) > 1

    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    marcadores = []
    for i, s in enumerate(series):
        marcador = MARCADORES_SERIE[i % len(MARCADORES_SERIE)] if varias else "o"
        desplazamiento = ((i - (len(series) - 1) / 2) * 0.22
                          if eje["tipo"] == "categorico" and varias else 0.0)
        marcadores.append((marcador, desplazamiento))
        _dibujar_serie(ax, s, eje, ESTILOS_LINEA[i % len(ESTILOS_LINEA)], marcador,
                       desplazamiento)

    _marcas_x(ax, eje)
    _limites_x(ax, eje)
    _limites_y(ax, series)
    ax.set_xlabel(ROTULO_PARAMETRO.get(parametro, parametro)
                  + (" (escala logarítmica)" if eje["tipo"] == "log" else ""))
    ax.set_ylabel("AUC")

    titulo = f"{rotulo_modelo(modelo)}: curva de validación de {parametro}"
    if varias:
        titulo += f" según {', '.join(distintas)}" if distintas else f" ({' y '.join(rotulos)})"
    elif series[0]["sufijo"]:
        titulo += f" ({series[0]['sufijo']})"
    ax.set_title(titulo, loc="left", pad=24)
    fijos = texto_fijos(series, distintas)
    n = max((s["resumen"][c].max() for s in series for c in s["resumen"].columns
             if c.startswith("n_")), default=None)
    subtitulo = f"AUC, media ± 1 desvío en {_texto_folds(n)}"
    if fijos:
        subtitulo += "; fijos: " + ", ".join(fijos)
    ax.annotate(subtitulo, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", ha="left", va="bottom", fontsize=9.5,
                color=TINTA_SECUNDARIA, gid="subtitulo")
    _leyenda_curvas(ax, series, rotulos, eje["tipo"] == "categorico")
    if omitidos:
        nota = ("Sin medir (resultados/curvas/omitidos.txt): " + ", ".join(omitidos))
        ax.annotate(textwrap.fill(nota, 34), xy=(1.02, 0), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=8.5, color=TINTA_SECUNDARIA,
                    annotation_clip=False, gid="omitidos")

    # El diseño se fija antes de rotular el máximo: los rótulos se ubican en píxeles.
    fig.canvas.draw()
    if _marcas_superpuestas(ax):
        ax.tick_params(axis="x", labelrotation=45)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        fig.canvas.draw()
    fig.set_layout_engine("none")
    mejores = [maximo_de(s) for s in series]
    corrimientos = esquives(mejores, eje["tipo"] == "categorico")
    for s, (marcador, desplazamiento), mejor, corrimiento in zip(series, marcadores, mejores,
                                                                 corrimientos):
        if mejor is None:
            continue
        _marcar_maximo(ax, s, eje, mejor, marcador, desplazamiento, corrimiento,
                       GLIFO_MARCADOR.get(marcador) if varias else None)
        _resaltar_marca(ax, eje, mejor)
    return fig


def figura_curva(tabla, nombre=None, omitidos=(), metrica="auc"):
    """Paso 4.5: la curva de validación de un CSV de resultados/curvas/.

    `tabla` es el CSV en formato largo (con `parametro` y `punto`), tal como lo lee pandas;
    `nombre`, el del archivo sin extensión (de él sale el sufijo). Train y validación en
    COLOR_CONJUNTO: media con banda de ±1 desvío entre folds en un eje numérico, o puntos con
    barras en uno categórico. El máximo de validación va relleno, con su valor, y la brecha
    train − validación se marca y se rotula en ese punto. `omitidos`: las etiquetas de los
    puntos que src/curvas.py no pudo medir, que se listan al pie de la leyenda."""
    return _figura_curvas([_serie(nombre, tabla, metrica)], omitidos)


def figura_curvas_comparadas(tablas, metrica="auc"):
    """Varias curvas del mismo modelo y el mismo parámetro en una figura (KNN con weights
    uniform y distance): el color sigue diciendo train o validación, y cada curva lleva su
    estilo de línea y su marcador. El valor del máximo de cada curva empieza con su marcador como
    carácter (● ■ ◆ ▲), y dos máximos en el mismo x se corren a los lados para no taparse.
    `tablas`: {nombre del archivo: tabla}."""
    if not tablas:
        raise ValueError("no hay curvas que comparar")
    series = [_serie(nombre, tabla, metrica) for nombre, tabla in tablas.items()]
    series.sort(key=lambda s: (ORDEN_SUFIJOS.index(s["sufijo"]) if s["sufijo"] in ORDEN_SUFIJOS
                               else len(ORDEN_SUFIJOS), s["sufijo"]))
    return _figura_curvas(series)


def kebab(texto):
    return re.sub(r"[^a-z0-9]+", "-", str(texto).lower()).strip("-")


def nombre_figura_curva(modelo, parametro, *sufijos):
    """03-curva-knn-n-neighbors-uniform.png: en kebab-case, como las demás figuras."""
    partes = [PREFIJO_CURVA, modelo, parametro, *sufijos]
    return "-".join(kebab(p) for p in partes if p) + ".png"


def nombre_figura_comparadas(tablas):
    """El nombre de la figura combinada de varias curvas del mismo eje, por los valores fijos en
    que difieren: 03-curva-knn-n-neighbors-por-weights.png. No coincide con el de ninguna curva
    sola, que termina en su sufijo."""
    series = []
    for nombre, tabla in tablas.items():
        modelo, _, sufijo = identidad_curva(tabla, nombre)
        series.append({"nombre": nombre, "modelo": modelo, "sufijo": sufijo,
                       "fijos": fijos_de(tabla), "configuraciones": configuraciones_de(tabla)})
    _, distintas = _rotulos_series(series)
    modelo, parametro, _ = identidad_curva(next(iter(tablas.values())))
    if distintas:
        return nombre_figura_curva(modelo, parametro, "por", *distintas)
    return nombre_figura_curva(modelo, parametro, "comparadas",
                               *sorted(s["sufijo"] or "vigente" for s in series))


# --- Lectura de resultados/ -------------------------------------------------------------------

class ArchivoIlegible(ValueError):
    """Un CSV de entrada que existe pero no se puede leer: vacío, con la última fila cortada
    (otro proceso todavía lo escribe) o mal formado. El CLI lo informa como error."""


def leer_csv(ruta):
    """Un CSV de resultados/ (en formato largo o un resumen), con leer_largo del contrato.

    Levanta ArchivoIlegible si el archivo está vacío, si no termina en un salto de línea o si
    pandas no lo puede leer. pandas termina cada fila, también la última, con un salto de línea:
    un archivo sin él quedó cortado a mitad de una fila, y leerlo daría una tabla incompleta sin
    ningún error."""
    ruta = Path(ruta)
    try:
        datos = ruta.read_bytes()
    except OSError as error:
        raise ArchivoIlegible(f"no se puede abrir ({type(error).__name__}: {error})") from error
    if not datos.strip():
        raise ArchivoIlegible("está vacío")
    if not datos.endswith(b"\n"):
        raise ArchivoIlegible("la última fila está cortada (el archivo no termina en un salto de "
                              "línea); puede que otro proceso todavía lo esté escribiendo")
    try:
        return leer_largo(io.BytesIO(datos))
    except ValueError as error:  # EmptyDataError, ParserError y UnicodeDecodeError lo son
        raise ArchivoIlegible(f"{type(error).__name__}: {error}") from error


def leer_curva(ruta):
    """Un CSV de resultados/curvas/. Uno que no se puede leer levanta ArchivoIlegible, y el CLI lo
    informa como error; uno sin las columnas de una curva, ValueError, y el CLI lo informa y lo
    omite, como src/curvas.leer_curvas."""
    tabla = leer_csv(ruta)
    faltan = [c for c in ["parametro", "punto", *CAMPOS] if c not in tabla.columns]
    if faltan:
        raise ValueError(f"no es una curva: faltan {', '.join(faltan)}")
    if tabla.empty:
        raise ValueError("no es una curva: no tiene filas")
    identidad_curva(tabla)
    return tabla


def leer_omitidos(directorio):
    """{curva: [etiqueta, …]} según resultados/curvas/omitidos.txt (curva, etiqueta y motivo,
    separados por tabuladores; lo escribe src/curvas.registrar_omitidos)."""
    ruta = Path(directorio) / OMITIDOS
    omitidos = {}
    if ruta.exists():
        try:
            texto = ruta.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"Aviso: no se pudo leer {_mostrar(ruta)} ({type(error).__name__}: {error}); "
                  "las figuras 03 salen sin la lista de puntos sin medir.")
            return omitidos
        for linea in texto.splitlines():
            curva, _, resto = linea.partition("\t")
            etiqueta = resto.partition("\t")[0]
            if curva and etiqueta:
                omitidos.setdefault(curva, []).append(etiqueta)
    return omitidos


def linea_base_de(ruta):
    """{metrica: media de validación} de linea_base.csv, o None si no existe. Levanta
    ArchivoIlegible si existe pero no se puede leer o no tiene las columnas del contrato."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    tabla = leer_csv(ruta)
    faltan = [c for c in ("conjunto", "metrica", "valor") if c not in tabla.columns]
    if faltan:
        raise ArchivoIlegible(f"le faltan columnas del contrato: {', '.join(faltan)}")
    validacion = tabla[tabla["conjunto"] == "validacion"]
    return {m: float(v) for m, v in validacion.groupby("metrica")["valor"].mean().items()}


# --- CLI --------------------------------------------------------------------------------------

def _mostrar(ruta):
    ruta = Path(ruta).resolve()
    try:
        return ruta.relative_to(RAIZ).as_posix()
    except ValueError:
        return str(ruta)


def _guardar(fig, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Escrita {_mostrar(ruta)}")
    return ruta


def _dibujar(escritas, errores, que, funcion, ruta):
    """Dibuja y guarda una figura; si falla, lo informa y sigue con las demás."""
    try:
        escritas.append(_guardar(funcion(), ruta))
    except Exception as error:  # noqa: BLE001 — se informa y el comando termina con código 1
        plt.close("all")
        errores.append(f"{que}: {type(error).__name__}: {error}")
        print(f"No se pudo dibujar {que}: {type(error).__name__}: {error}")


def _ilegible(errores, ruta, error, consecuencia):
    """Informa un archivo que existe pero no se puede leer, con lo que eso omite, y lo anota
    como error (el comando termina con código 1)."""
    errores.append(f"{Path(ruta).name}: {error}")
    print(f"No se pudo leer {_mostrar(ruta)}: {error}. {consecuencia}")


def modelos_faltantes(tabla):
    """Los clasificadores de ORDEN_MODELOS sin AUC de validación en un resumen de la CV."""
    if not {"modelo", "conjunto", "metrica", "media"} <= set(tabla.columns):
        return []
    r = tabla[(tabla["conjunto"] == "validacion") & (tabla["metrica"] == "auc")]
    presentes = set(r.dropna(subset=["media"])["modelo"])
    return [m for m in ORDEN_MODELOS if m not in presentes]


def generar_ablaciones(resultados, figuras):
    ruta = Path(resultados) / RESUMEN_ABLACIONES
    escritas, errores = [], []
    if not ruta.exists():
        print(f"Falta {_mostrar(ruta)} (python -m src.ablaciones --resumen): se omiten "
              f"{FIGURA_ABLACIONES} y {FIGURA_EFECTO}.")
        return escritas, errores
    try:
        tabla = leer_csv(ruta)
    except ArchivoIlegible as error:
        _ilegible(errores, ruta, error, f"Se omiten {FIGURA_ABLACIONES} y {FIGURA_EFECTO}.")
        return escritas, errores
    _dibujar(escritas, errores, FIGURA_ABLACIONES, lambda: figura_ablaciones(tabla),
             Path(figuras) / FIGURA_ABLACIONES)
    _dibujar(escritas, errores, FIGURA_EFECTO, lambda: figura_ablaciones_efecto(tabla),
             Path(figuras) / FIGURA_EFECTO)
    return escritas, errores


def generar_modelos(resultados, figuras):
    ruta = Path(resultados) / RESUMEN_CV
    escritas, errores = [], []
    if not ruta.exists():
        print(f"Falta {_mostrar(ruta)} (python -m src.experimentos --etiqueta referencia "
              f"--resumen): se omite {FIGURA_MODELOS}.")
        return escritas, errores
    try:
        tabla = leer_csv(ruta)
    except ArchivoIlegible as error:
        _ilegible(errores, ruta, error, f"Se omite {FIGURA_MODELOS}.")
        return escritas, errores
    ruta_base = Path(resultados) / LINEA_BASE
    try:
        base = linea_base_de(ruta_base)
    except ArchivoIlegible as error:
        # La 02 se dibuja igual: la línea de base sale de las filas sin_modelo del resumen.
        base = None
        _ilegible(errores, ruta_base, error,
                  f"En {FIGURA_MODELOS}, «{ROTULO_LINEA_BASE}» sale de las filas sin_modelo del "
                  "resumen o, si no las tiene, de AUC 0,5 y recall_q = q.")
    faltan = modelos_faltantes(tabla)
    if faltan:
        print(f"Aviso: {_mostrar(ruta)} no trae el AUC de validación de {', '.join(faltan)}: "
              f"{FIGURA_MODELOS} sale sin {'ellos' if len(faltan) > 1 else 'él'}.")
    _dibujar(escritas, errores, FIGURA_MODELOS, lambda: figura_modelos(tabla, base),
             Path(figuras) / FIGURA_MODELOS)
    return escritas, errores


def generar_curvas(resultados, figuras):
    directorio = Path(resultados) / DIR_CURVAS
    escritas, errores = [], []
    rutas = sorted(directorio.glob("*.csv")) if directorio.is_dir() else []
    if not rutas:
        print(f"No hay curvas en {_mostrar(directorio)} todavía (python -m src.curvas): se "
              "omiten las figuras 03.")
        return escritas, errores
    omitidos = leer_omitidos(directorio)
    grupos, nombres, ilegibles = {}, set(), 0
    for ruta in rutas:
        try:
            tabla = leer_curva(ruta)
        except ArchivoIlegible as error:
            ilegibles += 1
            _ilegible(errores, ruta, error, "Se omiten las figuras 03 que salen de él.")
            continue
        except ValueError as error:
            print(f"Se ignora {ruta.name}: {error}.")
            continue
        modelo, parametro, sufijo = identidad_curva(tabla, ruta.stem)
        puntos = set(puntos_de(tabla))
        # Sólo los omitidos que el CSV no tiene: los otros son de una corrida anterior.
        faltan = [e for e in omitidos.get(ruta.stem, []) if e.partition("=")[2] not in puntos]
        nombre = nombre_figura_curva(modelo, parametro, sufijo)
        nombres.add(nombre)
        _dibujar(escritas, errores, ruta.name,
                 lambda: figura_curva(tabla, ruta.stem, omitidos=faltan), Path(figuras) / nombre)
        grupos.setdefault((modelo, parametro), {})[ruta.stem] = tabla
    for (modelo, parametro), tablas in grupos.items():
        if len(tablas) < 2:
            continue
        que = f"{modelo}/{parametro} combinadas"
        try:
            nombre = nombre_figura_comparadas(tablas)
        except Exception as error:  # noqa: BLE001 — se informa y el comando termina con código 1
            errores.append(f"{que}: {type(error).__name__}: {error}")
            print(f"No se pudo dibujar {que}: {type(error).__name__}: {error}")
            continue
        if nombre in nombres:  # nunca se sobrescribe la figura de una curva sola
            nombre = nombre.removesuffix(".png") + "-comparadas.png"
        nombres.add(nombre)
        _dibujar(escritas, errores, que, lambda: figura_curvas_comparadas(tablas),
                 Path(figuras) / nombre)
    # No se borra nada: sólo se avisa qué figuras 03 ya no salen de ningún CSV.
    viejas = sorted(p.name for p in Path(figuras).glob(f"{PREFIJO_CURVA}-*.png")
                    if p.name not in nombres)
    if viejas:
        origen = ("que se haya podido leer; algunas pueden ser de los CSV ilegibles"
                  if ilegibles else "(de una corrida anterior; se pueden borrar)")
        print(f"Aviso: en {_mostrar(figuras)} quedan figuras 03 que no salen de ningún CSV de "
              f"{_mostrar(directorio)} {origen}: " + ", ".join(viejas))
    return escritas, errores


GRUPOS = {"ablaciones": generar_ablaciones, "modelos": generar_modelos, "curvas": generar_curvas}


def _argumentos():
    p = argparse.ArgumentParser(
        prog="python -m src.graficos",
        description="Figuras de análisis de los pasos 1.6, 3.4 y 4.5 -> figuras/. Un archivo de "
                    "entrada que todavía no existe se informa y no es un error.")
    p.add_argument("--solo", choices=list(GRUPOS),
                   help="dibuja sólo ese grupo: ablaciones (01, 01b), modelos (02) o curvas (03)")
    p.add_argument("--resultados", type=Path, default=DIR_RESULTADOS,
                   help=f"directorio de entrada (por defecto {_mostrar(DIR_RESULTADOS)})")
    p.add_argument("--figuras", type=Path, default=DIR_FIGURAS,
                   help=f"directorio de salida (por defecto {_mostrar(DIR_FIGURAS)})")
    return p


def main(argv=None):
    a = _argumentos().parse_args(argv)
    escritas, errores = [], []
    for nombre in ([a.solo] if a.solo else list(GRUPOS)):
        e, f = GRUPOS[nombre](a.resultados, a.figuras)
        escritas += e
        errores += f
    resumen_final = (f"{len(escritas)} figura{'' if len(escritas) == 1 else 's'} en "
                     f"{_mostrar(a.figuras)}")
    if errores:
        resumen_final += f"; {len(errores)} con error: " + "; ".join(errores)
    print(resumen_final + ".")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
