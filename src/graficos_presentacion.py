"""Correr con: python -m src.graficos_presentacion   (opcional: --solo NOMBRE …, --figuras DIR,
--resultados DIR)

Las figuras de proyección del deck (plan/PLAN.md, paso 7.3; guía de presentaciones, B5, B6 y B8).
Son otras que las de análisis de src/graficos.py: aquéllas llevan título y leyendas para leerse
solas en una página, y éstas se proyectan detrás de alguien que habla. Por eso:

- todas usan el mismo lienzo, ASPECTO_SLIDE (10,6 × 5,0 in) a DPI 200, y se guardan sin recorte:
  2 120 × 1 000 píxeles exactos. Así entran en el frame con la misma escala, y un rótulo de 15 pt
  mide lo mismo en todas las slides;
- no llevan título: lo que diría el título lo dicen el título del frame y el orador;
- ningún texto baja de TAM_MINIMO (14 pt), y la leyenda va fuera de los ejes, a la derecha, en una
  columna (bajo el título se leía como un subtítulo); orden-temporal la conserva arriba;
- un solo panel por figura, salvo orden-temporal (la tasa de «yes» y el euribor tienen escalas
  que no se mezclan en un eje doble) y modelos-final (AUC y recall, cada uno con su línea de base);
- los valores anotados llevan coma decimal y van en la tinta del texto, no en el color de la serie;
- los revelados por pasos (curva-rf-1 a 3, curva-svm y curva-svm-2, robustez-folds y
  robustez-folds-2) dibujan todo en todos los pasos, fijan el diseño, ubican los rótulos y sólo
  después ocultan lo que todavía no entró: el encuadre, los ejes, la leyenda y cada rótulo quedan
  en el mismo píxel, y la figura no salta al pasar de un paso al siguiente (B8).

Las funciones figura_* son puras: reciben DataFrames y devuelven una Figure sin guardarla, y
tests/test_graficos_presentacion.py las prueba con datos sintéticos. Lo que ya calcula
src/graficos.py (el resumen de cada curva, su eje x, el AUC por fold, el formato de los números, la
ubicación de los rótulos) se importa de ahí; los bloques de 2 000 filas y el año inferido, de
src/eda_html.py. Los datos de train entran por cargar_train(): nada aquí abre el test.

Lista canónica (FIGURAS), en figuras/presentacion/:

  duration-techo.png           slide 4: AUC de validación sin y con duration, por modelo (A0 y A1)
  orden-temporal.png           slide 5: % de «yes» por bloque de 2 000 filas y euribor3m, con los
                               cambios de año inferidos
  pdays-999.png                slide 6: % de «yes» con pdays = 999 sin contacto previo, con 999 y
                               previous ≥ 1, y con pdays < 999
  modelos-referencia.png, -zoom.png
                               slide 9: AUC de validación de los cinco modelos de referencia, con
                               «sin modelo» y marcas cada 0,1; + el zoom a las barras de error
  curva-rf-1/2/3.png           slide 10: max_depth de RF: train; + validación; + bandas, zonas y
                               la profundidad elegida con su brecha
  curva-knn.png                slide 11: n_neighbors de KNN (uniform), con distance atenuada
  curva-svm.png, -2.png        slide 12: C de la SVM lineal; + la RBF atenuada, en el mismo encuadre
  modelos-final.png, -zoom.png slide 13: AUC y recall al 20 % de los cuatro modelos finales, en
                               escala entera y después con zoom
  robustez-modelos.png         slide 18: AUC barajado contra hacia adelante, por modelo
  robustez-folds.png, -2.png   slide 19: RF hacia adelante bloque a bloque; + el barajado en las
                               mismas filas, unido por la pérdida (la lectura b y c de H1)

Opciones: `--solo` dibuja sólo esos grupos (los nombres de FIGURAS); `--figuras DIR` cambia el
directorio de salida y `--resultados DIR` el de entrada. Un archivo de entrada que falta o no se
puede leer es un error: se informa, se sigue con las demás figuras y el comando termina con
código 1, porque sin esa figura el deck no compila.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Importar src.estilo fija el backend Agg y aplica los rcParams, así que va antes que pyplot.
from src.estilo import (
    AZUL,
    COLOR_CLASE,
    COLOR_CONJUNTO,
    COLOR_LINEA_BASE,
    GRIS_REFERENCIA,
    NARANJA,
    REJILLA,
    ROTULO_LINEA_BASE,
    SUPERFICIE,
    TINTA,
    TINTA_SECUNDARIA,
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.configuracion import HIPERPARAMETROS_FINALES  # noqa: E402
from src.datos import FILA, OBJETIVO, cargar_train  # noqa: E402
from src.eda_html import (  # noqa: E402
    BLOQUE,
    MESES_CALENDARIO,
    MESES_EN_ES,
    PDAYS_CENTINELA,
    anio_inferido,
    leer_names,
)
from src.graficos import (  # noqa: E402
    ALFA_BANDA,
    AUC_SIN_MODELO,
    DIR_FIGURAS,
    DIR_RESULTADOS,
    ESQUEMA_ADELANTE,
    ESQUEMA_BARAJADO,
    LINEA_BASE_POR_DEFECTO,
    MARCADOR_MODELO,
    SIN_MODELO,
    VARIANTE_COMPLETA,
    _auc_por_fold,
    _bloques,
    _colocar,
    _decimales,
    _fijar_diseno,
    _mostrar,
    _tabla_robustez,
    _texto_folds,
    _x,
    color_modelo,
    eje_x,
    identidad_curva,
    leer_csv,
    leer_curva,
    miles,
    numero,
    ordenar_modelos,
    porcentaje,
    resumen_curva,
    rotulo_modelo,
)
from src.metricas import PRESUPUESTO  # noqa: E402

DIR_PRESENTACION = DIR_FIGURAS / "presentacion"

# --- El lienzo y la tipografía de proyección ----------------------------------------------------

# Apaisado, como el del TP1 (tp1/src/graficos_presentacion.py): el alto de un frame 16:9 menos el
# título y el pie deja una caja de proporción cercana a 2:1. En el beamer 16:9 del TP1 (márgenes de
# 1,6em, ancho de texto de 420 pt) \figgrande[0,97] la deja al 53 % de su tamaño: un rótulo de
# 15 pt se ve como uno de 8 pt en la slide, y TAM_MINIMO, 14 pt, como uno de 7,4 pt.
ASPECTO_SLIDE = (10.6, 5.0)
DPI = 200

TAM_MINIMO = 14      # ningún texto por debajo (regla B6; lo comprueban los tests)
TAM_MARCAS = 15      # números y rótulos de las marcas de los ejes
TAM_EJE = 16         # el nombre de cada eje
TAM_LEYENDA = 15
TAM_VALOR = 17       # los valores anotados: son los números que se leen desde el fondo del aula
TAM_ROTULO = 15      # rótulos de líneas de referencia y de zonas

LINEA = 2.8          # grosor de las series, en puntos
MARCADOR = 10        # diámetro de los marcadores, en puntos
ANILLO = 1.6         # el anillo del color del fondo alrededor de cada marcador
ALFA_ATENUADA = 0.5  # las curvas de comparación: KNN con distance, la SVM con RBF
ESTRELLA = 24        # el marcador del valor elegido de un hiperparámetro

# El valor elegido de cada hiperparámetro es el de la configuración vigente (src/configuracion.py,
# N0-3 y N0-13), no el que propone hiperparametros.json.
PROFUNDIDAD_ELEGIDA = HIPERPARAMETROS_FINALES["rf"]["max_depth"]
VECINOS_ELEGIDOS = HIPERPARAMETROS_FINALES["knn"]["n_neighbors"]
C_ELEGIDO = HIPERPARAMETROS_FINALES["svm"]["C"]
MODELO_ELEGIDO = "rf"  # D-23

MESES_ES = dict(zip(MESES_CALENDARIO, MESES_EN_ES.values()))  # 'aug' -> 'ago'

# Qué figuras lleva cada grupo del CLI: la lista canónica del deck, la de la ola 7 sin las cuatro
# figuras de reserva (N0-15).
FIGURAS = {
    "duration-techo": ("duration-techo.png",),
    "orden-temporal": ("orden-temporal.png",),
    "pdays-999": ("pdays-999.png",),
    "modelos-referencia": ("modelos-referencia.png", "modelos-referencia-zoom.png"),
    "curva-rf": ("curva-rf-1.png", "curva-rf-2.png", "curva-rf-3.png"),
    "curva-knn": ("curva-knn.png",),
    "curva-svm": ("curva-svm.png", "curva-svm-2.png"),
    "modelos-final": ("modelos-final.png", "modelos-final-zoom.png"),
    "robustez-modelos": ("robustez-modelos.png",),
    "robustez-folds": ("robustez-folds.png", "robustez-folds-2.png"),
}
# Las que llevan la línea de base «sin modelo», con ese rótulo (guía C1; gate G11). Las versiones
# -zoom de modelos-referencia y modelos-final acotan el eje a las barras de error y dejan la línea
# fuera: ahí «sin modelo» va en el rótulo del eje (_rotulo_con_base). Las curvas de hiperparámetros no la llevan: comparan train contra validación dentro de un modelo, y un eje que llegara a
# 0,5 aplastaría la forma de la curva de validación, que es lo que se usa para elegir.
CON_LINEA_BASE = frozenset({
    "duration-techo.png", "modelos-referencia.png", "modelos-final.png", "robustez-modelos.png",
    "robustez-folds.png", "robustez-folds-2.png",
})


# --- Piezas comunes -----------------------------------------------------------------------------

def _lienzo(nrows=1, ncols=1, **kw):
    """Una figura del tamaño de proyección con diseño restringido; devuelve (fig, ejes 2D)."""
    fig, ejes = plt.subplots(nrows, ncols, figsize=ASPECTO_SLIDE, dpi=DPI, layout="constrained",
                             squeeze=False, **kw)
    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.1, wspace=0.05, hspace=0.06)
    for ax in ejes.flat:
        _estilo_ejes(ax)
    return fig, ejes


def _estilo_ejes(ax):
    """Marcas grandes y trazos de ejes y rejilla un poco más gruesos que en las de análisis: el
    proyector se come las líneas de un punto. Los tamaños van en tick_params y no en un rc_context
    porque matplotlib crea las marcas al dibujar, cuando el contexto ya se cerró."""
    ax.tick_params(axis="both", which="major", labelsize=TAM_MARCAS, length=6, width=1.2, pad=7)
    ax.tick_params(axis="both", which="minor", length=3.5, width=1.0)
    for lado in ax.spines.values():
        lado.set_linewidth(1.2)
    ax.grid(True, color=REJILLA, lw=1.1)


def _rotulo_eje(ax, x=None, y=None):
    if x is not None:
        ax.set_xlabel(x, fontsize=TAM_EJE, labelpad=8)
    if y is not None:
        ax.set_ylabel(y, fontsize=TAM_EJE, labelpad=10)


def _leyenda_arriba(ax, manijas, ncol=None):
    """Leyenda en una fila (o en `ncol` columnas), ARRIBA y fuera del área de datos, sin marco."""
    return ax.legend(handles=manijas, ncol=ncol or len(manijas), frameon=False,
                     fontsize=TAM_LEYENDA, handlelength=1.8, handletextpad=0.5, columnspacing=1.4,
                     borderaxespad=0.2, loc="lower center", bbox_to_anchor=(0.5, 1.0))


def _leyenda_derecha(ax, manijas):
    """Leyenda FUERA de los ejes, a la derecha, en una columna alineada con el borde superior del
    área de datos. Arriba, centrada bajo el título, se leía como un subtítulo; adentro competía con
    los datos. A la derecha queda pegada a la figura, del lado del ancho que sobra en 16:9, y el
    diseño restringido le hace lugar."""
    return ax.legend(handles=manijas, ncol=1, loc="upper left", bbox_to_anchor=(1.015, 1.0),
                     frameon=False, fontsize=TAM_LEYENDA, handlelength=1.8, handletextpad=0.6,
                     labelspacing=0.9, borderaxespad=0.0, borderpad=0.0)


def _entradas_leyenda(leyenda, indices):
    """El símbolo y el texto de las entradas `indices` de una leyenda, para ocultarlas sin que la
    leyenda se reacomode (el texto oculto conserva su lugar)."""
    return [a for i in indices for a in (leyenda.legend_handles[i], leyenda.get_texts()[i])]


def _marcas(ax, eje, paso, decimales=None, formato=None):
    """Marcas fijas cada `paso` dentro de los límites actuales, con coma decimal."""
    axis = ax.yaxis if eje == "y" else ax.xaxis
    lo, hi = sorted(ax.get_ylim() if eje == "y" else ax.get_xlim())
    tolerancia = 1e-9 * max(1.0, abs(hi))
    marcas = np.arange(np.ceil(lo / paso - tolerancia) * paso, hi + tolerancia, paso)
    # + 0.0 convierte el −0,0 de np.ceil en 0,0: si no, la primera marca dice «−0 %».
    marcas = [float(np.round(m, 10)) + 0.0 for m in marcas]
    d = _decimales(marcas) if decimales is None else decimales
    axis.set_major_locator(FixedLocator(marcas))
    axis.set_major_formatter(FuncFormatter(formato or (lambda v, _p: numero(v, d))))


def _paso_marcas(lo, hi, maximo=7):
    """El paso más fino entre 0,01 y 0,2 que deja como mucho `maximo` marcas en [lo, hi]."""
    for paso in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if np.floor(hi / paso + 1e-9) - np.ceil(lo / paso - 1e-9) + 1 <= maximo:
            return paso
    return 1.0


def _formato_pct(v, _pos=None):
    return porcentaje(v, 0)


def _creador(ax, texto, gid, fontsize=TAM_VALOR, color=TINTA, fondo=True, peso="normal",
             coordenadas="data", estilo="normal"):
    """crear(xy, dx, dy, ha, va) para _colocar (src/graficos.py): `texto` anclado en xy, a dx, dy
    puntos. El fondo del color de la superficie lo despega de las líneas y las bandas."""
    caja = dict(boxstyle="round,pad=0.12", fc=SUPERFICIE, ec="none", alpha=0.9) if fondo else None

    def crear(xy, dx, dy, ha, va):
        return [ax.annotate(texto, xy=xy, xycoords=coordenadas, xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va=va, fontsize=fontsize,
                            color=color, fontweight=peso, fontstyle=estilo, bbox=caja, zorder=7,
                            gid=gid, annotation_clip=False)]
    return crear


def _alrededor(x, y, dx=12, dy=12, preferido=("derecha", "arriba", "abajo", "izquierda")):
    """Lugares para el valor de un punto, en el orden de `preferido` y a dos distancias."""
    lugares = {"derecha": (dx, 0, "left", "center"), "izquierda": (-dx, 0, "right", "center"),
               "arriba": (0, dy, "center", "bottom"), "abajo": (0, -dy, "center", "top")}
    return [((x, y), k * lugares[l][0], k * lugares[l][1], lugares[l][2], lugares[l][3])
            for k in (1, 2) for l in preferido]


def _rotular_vertical(ax, x, texto, gid, color=TINTA_SECUNDARIA):
    """El rótulo de una línea vertical de referencia en `x`: arriba o abajo, a un lado u otro de
    la línea, donde no tape nada."""
    crear = _creador(ax, texto, gid, fontsize=TAM_ROTULO, color=color,
                     coordenadas=("data", "axes fraction"))
    candidatos = [((x, 1.0), 7, -6, "left", "top"), ((x, 0.0), 7, 6, "left", "bottom"),
                  ((x, 1.0), -7, -6, "right", "top"), ((x, 0.0), -7, 6, "right", "bottom")]
    return _colocar(ax, crear, candidatos)[0][0]


def _rotular_horizontal(ax, y, texto, gid, color=TINTA_SECUNDARIA, preferido="izquierda"):
    """El rótulo de una línea horizontal de referencia en `y`: sobre la línea, en el extremo
    `preferido`; si ahí tapa algo, en el otro, y si no, debajo."""
    crear = _creador(ax, texto, gid, fontsize=TAM_ROTULO, color=color,
                     coordenadas=("axes fraction", "data"))
    extremos = {"izquierda": (0.0, 8, "left"), "derecha": (1.0, -8, "right")}
    orden = [preferido] + [lado for lado in extremos if lado != preferido]
    candidatos = [((extremos[l][0], y), extremos[l][1], dy, extremos[l][2], va)
                  for dy, va in ((5, "bottom"), (-5, "top")) for l in orden]
    return _colocar(ax, crear, candidatos)[0][0]


def _linea_base_vertical(ax, x=AUC_SIN_MODELO):
    return ax.axvline(x, color=COLOR_LINEA_BASE, ls=":", lw=2.4, zorder=2, gid="linea-base")


def _linea_base_horizontal(ax, y=AUC_SIN_MODELO):
    return ax.axhline(y, color=COLOR_LINEA_BASE, ls=":", lw=2.4, zorder=2, gid="linea-base")


def _ocultar(artistas):
    """Oculta lo que todavía no entró en un paso del revelado. Se llama con el diseño ya fijo y los
    rótulos ya ubicados, así nada se corre (B8)."""
    for artista in artistas:
        if artista is not None:
            artista.set_visible(False)


def _filas_modelo(ax, orden, rotulos):
    """Un eje y categórico con una fila por modelo, la primera arriba y sin rejilla horizontal."""
    ys = np.arange(len(orden), dtype=float)
    ax.set_yticks(ys, rotulos)
    ax.set_ylim(len(orden) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelsize=TAM_EJE, labelcolor=TINTA)
    ax.grid(axis="y", visible=False)
    return ys


def _punto_con_barra(ax, x, y, d, color, marcador="o", gid=None, lleno=True, tam=MARCADOR + 2):
    """Un punto con una barra horizontal de ±d y el anillo del color del fondo."""
    barras = ax.errorbar(x, y, xerr=d, fmt=marcador, ms=tam, color=color,
                         mfc=color if lleno else SUPERFICIE, mec=color if not lleno else SUPERFICIE,
                         mew=2.2 if not lleno else ANILLO, elinewidth=2.6, capsize=6,
                         capthick=2.4, zorder=4)
    if gid:
        barras.lines[0].set_gid(gid)
    return barras


def _valor_a_la_derecha(ax, x, y, d, texto, gid):
    """El valor de un punto con barra horizontal: a la derecha del extremo de la barra; si ahí
    tapa algo, a la izquierda del otro extremo o sobre el punto."""
    candidatos = [((x + d, y), 10, 0, "left", "center"), ((x - d, y), -10, 0, "right", "center"),
                  ((x, y), 0, 13, "center", "bottom"), ((x, y), 0, -13, "center", "top"),
                  ((x + d, y), 24, 0, "left", "center")]
    return _colocar(ax, _creador(ax, texto, gid), candidatos)[0][0]


def _linea_base_de(resumen, metrica, conjunto="validacion"):
    """El valor de «sin modelo» en un resumen de la CV: sus filas sin_modelo o, si no tiene, el que
    vale por construcción (AUC 0,5; recall = q)."""
    r = resumen[(resumen["conjunto"] == conjunto) & (resumen["metrica"] == metrica)
                & (resumen["modelo"] == SIN_MODELO)]["media"].dropna()
    return float(r.mean()) if len(r) else float(LINEA_BASE_POR_DEFECTO[metrica])


def _medidas(resumen, metrica, conjunto="validacion"):
    """Media y desvío entre folds de `metrica` por modelo (sin «sin modelo»), de un resumen de la
    validación cruzada (cv_*_resumen.csv)."""
    faltan = [c for c in ("modelo", "conjunto", "metrica", "media", "desvio")
              if c not in resumen.columns]
    if faltan:
        raise ValueError(f"al resumen de la validación cruzada le faltan columnas: "
                         f"{', '.join(faltan)}")
    r = resumen[(resumen["conjunto"] == conjunto) & (resumen["metrica"] == metrica)
                & (resumen["modelo"] != SIN_MODELO)].dropna(subset=["media"])
    if r.empty:
        raise ValueError(f"el resumen no tiene {metrica} de {conjunto} de ningún modelo")
    if r["modelo"].duplicated().any():
        raise ValueError(f"el resumen tiene más de una fila de {metrica} por modelo")
    return r.set_index("modelo")[["media", "desvio"]].fillna({"desvio": 0.0})


def _orden_por(medidas):
    """Los modelos de mayor a menor media; a igual media, en el orden de src/graficos.py."""
    base = ordenar_modelos(medidas.index)
    return sorted(medidas.index, key=lambda m: (-medidas.at[m, "media"], base.index(m)))


# --- Slide 4: duration, el techo de la fuga -----------------------------------------------------

def figura_duration_techo(ablaciones, variante="A1"):
    """El AUC de validación de cada modelo sin duration (A0, azul) y con ella (A1, naranja), unidos
    por una flecha, con los valores anotados y «sin modelo» en 0,5. Las filas van de mayor a
    menor techo. `ablaciones` es ablaciones_resumen.csv: auc_referencia es A0 y auc_variante, A1,
    los dos con los hiperparámetros de referencia de src/ablaciones.py."""
    faltan = [c for c in ("modelo", "variante", "auc_referencia", "auc_variante")
              if c not in ablaciones.columns]
    if faltan:
        raise ValueError(f"al resumen de ablaciones le faltan columnas: {', '.join(faltan)}")
    t = ablaciones[ablaciones["variante"] == variante].dropna(
        subset=["auc_referencia", "auc_variante"])
    if t.empty:
        raise ValueError(f"el resumen de ablaciones no tiene la variante {variante}")
    if t["modelo"].duplicated().any():
        raise ValueError(f"el resumen de ablaciones tiene más de una fila de {variante} por modelo")
    t = t.set_index("modelo")
    orden = sorted(t.index, key=lambda m: (-t.at[m, "auc_variante"], str(m)))
    sin = t.loc[orden, "auc_referencia"].to_numpy(float)
    con = t.loc[orden, "auc_variante"].to_numpy(float)

    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    ys = _filas_modelo(ax, orden, [rotulo_modelo(m) for m in orden])
    _linea_base_vertical(ax)
    for y, a, b, m in zip(ys, sin, con, orden):
        ax.annotate("", xy=(b, y), xytext=(a, y), gid=f"salto:{m}", zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=GRIS_REFERENCIA, lw=2.4,
                                    mutation_scale=24, shrinkA=10, shrinkB=11))
    ax.plot(sin, ys, ls="none", marker="o", ms=MARCADOR + 4, color=AZUL, mec=SUPERFICIE,
            mew=ANILLO, zorder=5, gid="sin-duration")
    ax.plot(con, ys, ls="none", marker="o", ms=MARCADOR + 4, color=NARANJA, mec=SUPERFICIE,
            mew=ANILLO, zorder=5, gid="con-duration")

    lo = min(AUC_SIN_MODELO, float(sin.min())) - 0.05
    hi = max(1.0, float(con.max()) + 0.07)
    ax.set_xlim(lo, hi)
    _marcas(ax, "x", 0.1, 1)
    n = t["n_folds"].max() if "n_folds" in t.columns else None
    _rotulo_eje(ax, x=f"AUC de validación (media de {_texto_folds(n)})")
    _leyenda_derecha(ax, [
        Line2D([], [], ls="none", marker="o", ms=MARCADOR + 2, color=AZUL, mec=SUPERFICIE,
               label="sin duration"),
        Line2D([], [], ls="none", marker="o", ms=MARCADOR + 2, color=NARANJA, mec=SUPERFICIE,
               label="con duration")])

    _fijar_diseno(fig)
    for y, a, b, m in zip(ys, sin, con, orden):
        ax.annotate(numero(a, 3), xy=(a, y), xytext=(-15, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=TAM_VALOR, color=TINTA, zorder=7,
                    gid=f"valor:sin:{m}")
        ax.annotate(numero(b, 3), xy=(b, y), xytext=(15, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=TAM_VALOR, color=TINTA, zorder=7,
                    gid=f"valor:con:{m}")
    _rotular_vertical(ax, AUC_SIN_MODELO, ROTULO_LINEA_BASE, "rotulo-linea-base")
    return fig


# --- Slide 5: el archivo está ordenado por fecha ------------------------------------------------

def serie_temporal(train, names):
    """Lo que dibuja orden-temporal, calculado como en src/eda_html.py (analizar, «tiempo»): la
    tasa de «yes» por bloque de BLOQUE filas consecutivas del CSV original, el euribor fila por
    fila, el año inferido de cada fila y los cambios de año, en el punto medio entre la última fila
    de un año y la primera del siguiente. `names` es leer_names() (sólo se usa anio_ini)."""
    faltan = [c for c in (FILA, OBJETIVO, "month", "euribor3m") if c not in train.columns]
    if faltan:
        raise ValueError(f"a train le faltan columnas: {', '.join(faltan)}")
    df = train.sort_values(FILA).reset_index(drop=True)
    y = (df[OBJETIVO] == "yes").astype(float)
    bloque = df[FILA] // BLOQUE
    bloques = pd.DataFrame({"x": df[FILA].groupby(bloque).mean(),
                            "tasa": 100 * y.groupby(bloque).mean(),
                            "n": y.groupby(bloque).size()})
    anio = anio_inferido(df, names)
    anios = sorted({int(a) for a in anio})
    fronteras = [(df.loc[anio == a - 1, FILA].max() + df.loc[anio == a, FILA].min()) / 2
                 for a in anios[1:]]
    return SimpleNamespace(bloques=bloques, anios=anios, fronteras=[float(f) for f in fronteras],
                           fila=df[FILA].to_numpy(float), euribor=df["euribor3m"].to_numpy(float),
                           fila_min=float(df[FILA].min()), fila_max=float(df[FILA].max()))


def figura_orden_temporal(train, names):
    """Dos paneles con el mismo eje x, la fila del CSV original: arriba, el % de «yes» de cada
    bloque de 2 000 filas, con el primero y el último anotados; abajo, euribor3m fila por fila.
    Líneas verticales en los cambios de año inferidos, y cada año rotulado en el eje x, centrado
    en su tramo. Van en dos paneles y no en un eje doble: dos escalas en un mismo gráfico inventan
    una correlación."""
    s = serie_temporal(train, names)
    b = s.bloques
    fig, ejes = _lienzo(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.8, 1]})
    arriba, abajo = ejes[:, 0]
    color = COLOR_CLASE["yes"]
    arriba.plot(b["x"], b["tasa"], color=color, lw=LINEA, marker="o", ms=MARCADOR - 1,
                mec=SUPERFICIE, mew=ANILLO, zorder=4, gid="tasa-yes")
    abajo.plot(s.fila, s.euribor, color=TINTA_SECUNDARIA, lw=2.4, zorder=3, gid="euribor")
    for ax in (arriba, abajo):
        for i, f in enumerate(s.fronteras):
            ax.axvline(f, color=GRIS_REFERENCIA, lw=1.5, zorder=1, gid=f"cambio-anio:{i}")

    margen = 0.012 * (s.fila_max - s.fila_min or 1.0)
    abajo.set_xlim(s.fila_min - margen, s.fila_max + margen)
    # El eje x se lee en años, no en números de fila: cada año, centrado en su tramo, y los
    # cambios de año como líneas verticales en los dos paneles.
    x_lo, x_hi = abajo.get_xlim()
    bordes = [x_lo, *s.fronteras, x_hi]
    abajo.set_xticks([(x0 + x1) / 2 for x0, x1 in zip(bordes[:-1], bordes[1:])],
                     [str(a) for a in s.anios])
    abajo.tick_params(axis="x", length=0, labelsize=TAM_EJE, labelcolor=TINTA)
    arriba.tick_params(axis="x", length=0)
    tope = max(float(b["tasa"].max()), 1.0) * 1.3
    arriba.set_ylim(0, tope)
    _marcas(arriba, "y", _paso_pct(tope), formato=_formato_pct)
    e_lo, e_hi = float(np.nanmin(s.euribor)), float(np.nanmax(s.euribor))
    alto = (e_hi - e_lo) or 1.0
    abajo.set_ylim(e_lo - 0.12 * alto, e_hi + 0.18 * alto)
    _marcas(abajo, "y", _paso_entero(e_lo, e_hi), 0)
    arriba.grid(axis="x", visible=False)
    abajo.grid(axis="x", visible=False)
    _rotulo_eje(arriba, y="% de «yes»")
    _rotulo_eje(abajo, x="filas del CSV original, en el orden del archivo (por fecha)",
                y="euribor3m")
    fig.align_ylabels([arriba, abajo])
    _leyenda_arriba(arriba, [
        Line2D([], [], color=color, lw=LINEA, marker="o", ms=MARCADOR - 1, mec=SUPERFICIE,
               label=f"% de «yes» en cada bloque de {miles(BLOQUE)} filas"),
        Line2D([], [], color=GRIS_REFERENCIA, lw=1.5, label="cambio de año (inferido)")])

    _fijar_diseno(fig)
    x0, v0 = float(b["x"].iloc[0]), float(b["tasa"].iloc[0])
    x1, v1 = float(b["x"].iloc[-1]), float(b["tasa"].iloc[-1])
    _colocar(arriba, _creador(arriba, porcentaje(v0, 1), "valor:primer-bloque"),
             [((x0, v0), 0, 14, "center", "bottom"), ((x0, v0), 12, 12, "left", "bottom"),
              ((x0, v0), 14, 0, "left", "center")])
    _colocar(arriba, _creador(arriba, porcentaje(v1, 1), "valor:ultimo-bloque"),
             [((x1, v1), 0, 14, "center", "bottom"), ((x1, v1), -12, 12, "right", "bottom"),
              ((x1, v1), -14, 0, "right", "center"), ((x1, v1), 0, -14, "center", "top")])
    return fig


def _paso_pct(tope):
    """El paso de las marcas de un eje de porcentajes de 0 a `tope`: 5, 10, 20 o 25 puntos."""
    for paso in (5, 10, 20, 25, 50):
        if tope / paso <= 6:
            return paso
    return 100


def _paso_entero(lo, hi):
    for paso in (1, 2, 5, 10):
        if (hi - lo) / paso <= 4:
            return paso
    return 20


# --- Slide 6: 999 no es «nunca contactado» ------------------------------------------------------

GRUPOS_PDAYS = ("sin_previos", "centinela_con_previos", "contactados")
GRUPO_RESALTADO_PDAYS = "centinela_con_previos"
GRIS_CONTEXTO = "#c9c7bf"  # las barras de contexto: un gris que no compite con el resalte


def grupos_pdays(train):
    """Las filas de train en tres grupos, con su tamaño, sus «yes» y su tasa (en %):
    sin_previos, pdays = 999 y previous = 0; centinela_con_previos, pdays = 999 y previous ≥ 1
    (contactados antes, con el centinela igual); contactados, pdays < 999."""
    faltan = [c for c in ("pdays", "previous", OBJETIVO) if c not in train.columns]
    if faltan:
        raise ValueError(f"a train le faltan columnas: {', '.join(faltan)}")
    yes = train[OBJETIVO] == "yes"
    centinela = train["pdays"] == PDAYS_CENTINELA
    mascaras = {"sin_previos": centinela & (train["previous"] == 0),
                "centinela_con_previos": centinela & (train["previous"] >= 1),
                "contactados": ~centinela}
    filas = [{"grupo": g, "n": int(m.sum()), "yes": int(yes[m].sum()),
              "tasa": 100 * float(yes[m].mean()) if m.any() else np.nan}
             for g, m in mascaras.items()]
    return pd.DataFrame(filas).set_index("grupo")


def figura_pdays(train):
    """Barras horizontales con el % de «yes» de cada grupo de grupos_pdays, el valor en la punta
    y, en el rótulo de cada fila, la condición y cuántas filas son; la del medio, la que desmiente
    que 999 sea «nunca contactado», en negrita."""
    g = grupos_pdays(train)
    rotulos = {
        "sin_previos": (f"pdays = {PDAYS_CENTINELA} y previous = 0",
                        f"{miles(g.at['sin_previos', 'n'])} filas · sin contacto previo"),
        "centinela_con_previos": (f"pdays = {PDAYS_CENTINELA} y previous ≥ 1",
                                  f"{miles(g.at['centinela_con_previos', 'n'])} filas · "
                                  "contactados antes"),
        "contactados": (f"pdays < {PDAYS_CENTINELA}",
                        f"{miles(g.at['contactados', 'n'])} filas · contactados antes"),
    }
    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    ys = _filas_modelo(ax, list(GRUPOS_PDAYS), ["\n".join(rotulos[k]) for k in GRUPOS_PDAYS])
    tasas = [float(g.at[k, "tasa"]) for k in GRUPOS_PDAYS]
    # Un solo color de resalte: la barra que desmiente que 999 sea «nunca contactado». Las otras
    # dos son el contexto que la enmarca, en gris.
    colores = [COLOR_CLASE["yes"] if k == GRUPO_RESALTADO_PDAYS else GRIS_CONTEXTO
               for k in GRUPOS_PDAYS]
    barras = ax.barh(ys, tasas, height=0.5, color=colores, zorder=3)
    for barra, k in zip(barras, GRUPOS_PDAYS):
        barra.set_gid(f"tasa:{k}")
    tope = max(max(tasas), 1.0) * 1.22
    ax.set_xlim(0, tope)
    _marcas(ax, "x", _paso_pct(tope), formato=_formato_pct)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", labelsize=TAM_MARCAS)
    ax.get_yticklabels()[1].set_fontweight("bold")
    _rotulo_eje(ax, x="% de «yes» del grupo")

    _fijar_diseno(fig)
    for y, v, k in zip(ys, tasas, GRUPOS_PDAYS):
        ax.annotate(porcentaje(v, 1), xy=(v, y), xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=TAM_VALOR, color=TINTA, zorder=7,
                    gid=f"valor:{k}")
    return fig


# --- Slides 9 y 13: los modelos contra «sin modelo» ---------------------------------------------

def _panel_filas(ax, orden, medidas, base, gid_metrica):
    """Una métrica por modelo, una fila cada uno: punto con barra de ±1 desvío entre folds, del
    color y con el marcador del modelo. Devuelve los puntos (x, y, desvío, modelo) para
    rotularlos con el diseño fijo. «Sin modelo» lo dibuja _limites_metrica en el paso sin zoom, y
    en los dos pasos va en el rótulo del eje (_rotulo_con_base)."""
    ys = np.arange(len(orden), dtype=float)
    puntos = []
    for y, m in zip(ys, orden):
        media, desvio = float(medidas.at[m, "media"]), float(medidas.at[m, "desvio"])
        _punto_con_barra(ax, media, y, desvio, color_modelo(m), MARCADOR_MODELO.get(m, "o"),
                         gid=f"{gid_metrica}:{m}", tam=MARCADOR + 5)
        puntos.append((media, y, desvio, m))
    return puntos


def _ventana_zoom(medidas, antes=0.25, despues=0.45):
    """Los límites del eje x acotado a las barras de ±1 desvío, con `antes` y `despues` del rango
    de margen (el de la derecha, para los valores). El rango nunca baja de 0,03: si no, una
    diferencia de milésimas parecería enorme."""
    lo = float((medidas["media"] - medidas["desvio"]).min())
    hi = float((medidas["media"] + medidas["desvio"]).max())
    rango = max(hi - lo, 0.03)
    centro = (lo + hi) / 2
    return centro - rango / 2 - antes * rango, centro + rango / 2 + despues * rango


def _limites_metrica(ax, medidas, base, zoom, antes=0.25, despues=0.45, margen=0.11):
    """El eje x de un panel de modelos, en uno de los dos pasos de su revelado. Sin zoom: la escala
    entera, de «sin modelo» (dibujada, con su rótulo) a los valores, con marcas cada 0,1, y la
    ventana del paso siguiente sombreada. Con zoom: sólo esa ventana, donde la barra de error se
    lee. Devuelve la línea de base (o None)."""
    a, b = _ventana_zoom(medidas, antes, despues)
    if zoom:
        ax.set_xlim(a, b)
        lo, hi = ax.get_xlim()
        _marcas(ax, "x", _paso_marcas(lo, hi, maximo=6))
        return None
    lo = min(base, float((medidas["media"] - medidas["desvio"]).min())) - 0.05
    hi = float((medidas["media"] + medidas["desvio"]).max()) + margen
    ax.set_xlim(lo, hi)
    _marcas(ax, "x", 0.1, 1)
    ax.axvspan(a, b, color=REJILLA, alpha=0.55, lw=0, zorder=0.5, gid="ventana-zoom")
    return _linea_base_vertical(ax, base)


def _rotulo_con_base(texto, base, sep="  ·  "):
    """El rótulo del eje x con «sin modelo» al final: la línea de base queda fuera del eje
    acotado, y así se sigue leyendo en la slide. En un panel angosto, `sep="\n"`."""
    return f"{texto}{sep}{ROTULO_LINEA_BASE}: {numero(base, 1)}"


def figura_modelos_referencia(resumen, zoom=False):
    """Slide 9: el AUC de validación de cada modelo con los hiperparámetros de referencia (media ±
    1 desvío entre folds, valor anotado), de mayor a menor, contra «sin modelo». `resumen` es
    cv_referencia_resumen.csv de src/experimentos.py (los cinco modelos, NB gaussiano incluido)."""
    medidas = _medidas(resumen, "auc")
    base = _linea_base_de(resumen, "auc")
    orden = _orden_por(medidas)
    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    _filas_modelo(ax, orden, [rotulo_modelo(m) for m in orden])
    puntos = _panel_filas(ax, orden, medidas, base, "auc")
    _limites_metrica(ax, medidas, base, zoom)
    n = resumen["n"].max() if "n" in resumen.columns else None
    _rotulo_eje(ax, x=_rotulo_con_base(
        f"AUC de validación, media ± 1 desvío entre {_texto_folds(n)}", base))

    _fijar_diseno(fig)
    for x, y, d, m in puntos:
        _valor_a_la_derecha(ax, x, y, d, numero(x, 3), f"valor:auc:{m}")
    if not zoom:
        _rotular_vertical(ax, base, ROTULO_LINEA_BASE, "rotulo-linea-base")
    return fig


def figura_modelos_final(resumen, zoom=False):
    """Slide 13: AUC (izquierda) y recall llamando al 20 % de la lista (derecha) de los modelos
    con los hiperparámetros elegidos, media ± 1 desvío entre folds y valor anotado, ordenados por
    AUC en los dos paneles, cada uno contra su «sin modelo» (0,5 y el 20 %). `resumen` es
    cv_final_resumen.csv."""
    auc = _medidas(resumen, "auc")
    recall = _medidas(resumen, "recall_q")
    orden = _orden_por(auc)
    faltan = [m for m in orden if m not in recall.index]
    if faltan:
        raise ValueError(f"el resumen no tiene el recall_q de {', '.join(faltan)}")
    fig, ejes = _lienzo(1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    izq, der = ejes[0]
    _filas_modelo(izq, orden, [rotulo_modelo(m) for m in orden])
    der.tick_params(axis="y", length=0)
    der.grid(axis="y", visible=False)
    paneles = []
    for ax, metrica, medidas in ((izq, "auc", auc), (der, "recall_q", recall)):
        base = _linea_base_de(resumen, metrica)
        paneles.append((ax, metrica, base, _panel_filas(ax, orden, medidas, base, metrica)))
        # Cada panel tiene la mitad del ancho: el margen derecho es el lugar de los valores, y
        # el izquierdo, el mismo, para que los puntos queden al centro de su panel.
        _limites_metrica(ax, medidas.loc[orden], base, zoom, antes=0.6, despues=0.9,
                         margen=0.18)
    _rotulo_eje(izq, x=_rotulo_con_base("AUC de validación", _linea_base_de(resumen, "auc"),
                                          sep="\n"))
    _rotulo_eje(der, x=_rotulo_con_base(
        f"recall al {porcentaje(100 * PRESUPUESTO, 0)} de la lista",
        _linea_base_de(resumen, "recall_q"), sep="\n"))

    _fijar_diseno(fig)
    for ax, metrica, base, puntos in paneles:
        for x, y, d, m in puntos:
            _valor_a_la_derecha(ax, x, y, d, numero(x, 3), f"valor:{metrica}:{m}")
        if not zoom:
            _rotular_vertical(ax, base, ROTULO_LINEA_BASE, "rotulo-linea-base")
    return fig


# --- Slides 10 a 12: las curvas de validación ---------------------------------------------------

def _curva(tabla):
    """El modelo, el parámetro y el resumen de una curva: media y desvío por punto y conjunto
    (resumen_curva de src/graficos.py), con los desvíos faltantes en 0."""
    modelo, parametro, _ = identidad_curva(tabla)
    r = resumen_curva(tabla)
    faltan = [c for c in ("media_train", "media_validacion") if c not in r.columns]
    if faltan:
        raise ValueError(f"la curva de {parametro} no tiene {', '.join(faltan)}")
    return SimpleNamespace(modelo=modelo, parametro=parametro, resumen=r.fillna(
        {"desvio_train": 0.0, "desvio_validacion": 0.0}))


def _punto_elegido(eje, valor):
    """El punto del eje que corresponde a `valor` (8, 801, 0.001), igual como texto o como
    número."""
    for punto in eje["puntos"]:
        if punto == str(valor):
            return punto
        try:
            if np.isclose(float(punto), float(valor), rtol=1e-9, atol=0.0):
                return punto
        except (TypeError, ValueError):
            continue
    raise ValueError(f"la curva no tiene el valor elegido {valor}")


def _serie_curva(ax, curva, eje, conjunto, estilo="-", alfa=1.0, bandas=True, gid="curva"):
    """Una serie (train o validación) de una curva: la línea por los puntos numéricos con su
    banda de ±1 desvío y, si la curva tiene «sin límite», un trazo punteado hasta ese punto, que
    lleva el desvío como barra. Una curva atenuada (alfa < 1, la de comparación) va sin
    marcadores: es contexto, y los puntos de la grilla ya los marca la curva principal. Devuelve
    los artistas que dibujó, por tipo."""
    r = curva.resumen
    color = COLOR_CONJUNTO[conjunto]
    numericos = [p for p in eje["puntos"] if p in r.index and p != "None"]
    x = [_x(eje, p) for p in numericos]
    m = r.loc[numericos, f"media_{conjunto}"].to_numpy(float)
    s = r.loc[numericos, f"desvio_{conjunto}"].to_numpy(float)
    marcador = (dict(marker="o", ms=MARCADOR - 1, mfc=color, mec=SUPERFICIE, mew=ANILLO)
                if alfa >= 1 else dict(marker="none"))
    linea, = ax.plot(x, m, ls=estilo, color=color, lw=LINEA, alpha=alfa, zorder=3,
                     gid=f"{gid}:{conjunto}", **marcador)
    salida = {"linea": [linea], "banda": [], "extremo": []}
    if bandas:
        salida["banda"].append(ax.fill_between(x, m - s, m + s, color=color, alpha=ALFA_BANDA,
                                               lw=0, zorder=1, gid=f"banda:{gid}:{conjunto}"))
    if "None" in r.index:
        xn = _x(eje, "None")
        mn = float(r.at["None", f"media_{conjunto}"])
        sn = float(r.at["None", f"desvio_{conjunto}"])
        tramo, = ax.plot([x[-1], xn], [m[-1], mn], ls=":", lw=LINEA - 0.6, color=color,
                         alpha=alfa, zorder=2, gid=f"tramo-sin-limite:{gid}:{conjunto}")
        punto, = ax.plot([xn], [mn], ls="none", color=color, alpha=alfa, zorder=3,
                         gid=f"sin-limite:{gid}:{conjunto}",
                         **{**marcador, "marker": "o", "ms": MARCADOR - 1, "mfc": color,
                            "mec": SUPERFICIE, "mew": ANILLO})
        salida["extremo"] += [tramo, punto]
        if bandas:
            barra = ax.errorbar(xn, mn, yerr=sn, fmt="none", ecolor=color, elinewidth=2.2,
                                capsize=5, capthick=2.0, zorder=2)
            barra.lines[2][0].set_gid(f"barra-sin-limite:{gid}:{conjunto}")
            salida["banda"] += [*barra.lines[1], *barra.lines[2]]
    return salida


def _limites_curvas(ax, curvas, abajo=0.08, arriba=0.05):
    """Los límites de y de una o más curvas, con train, validación y sus bandas; fijos, así todos
    los pasos de un revelado tienen el mismo encuadre."""
    valores = []
    for c in curvas:
        r = c.resumen
        for conjunto in ("train", "validacion"):
            m, s = r[f"media_{conjunto}"], r[f"desvio_{conjunto}"]
            valores += list((m - s).dropna()) + list((m + s).dropna())
    lo, hi = min(valores), max(valores)
    alto = (hi - lo) or 0.02
    ax.set_ylim(lo - abajo * alto, hi + arriba * alto)
    y_lo, y_hi = ax.get_ylim()
    _marcas(ax, "y", _paso_marcas(y_lo, y_hi))


def _eje_x_curva(ax, eje, rotulados=None):
    """El eje x de una curva: logarítmico si corresponde, una marca menor en cada punto de la
    grilla y rótulo en `rotulados` (todos, si no se pasa)."""
    posiciones = list(eje["posiciones"])
    if eje["tipo"] == "log":
        ax.set_xscale("log")
        u = np.log10(posiciones)
        margen = 0.035 * ((u[-1] - u[0]) or 1.0)
        ax.set_xlim(10 ** (u[0] - margen), 10 ** (u[-1] + margen))
    else:
        margen = 0.035 * ((posiciones[-1] - posiciones[0]) or 1.0)
        ax.set_xlim(posiciones[0] - margen, posiciones[-1] + margen)
    rotulados = eje["puntos"] if rotulados is None else rotulados
    mayores = [(_x(eje, p), r) for p, r in zip(eje["puntos"], eje["rotulos"]) if p in rotulados]
    ax.xaxis.set_major_locator(FixedLocator([x for x, _ in mayores]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: mayores[pos][1] if pos is not None and pos < len(mayores) else ""))
    ax.xaxis.set_minor_locator(FixedLocator(posiciones))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", visible=False)


def _rotulados_log(eje, obligatorios, separacion=0.22):
    """Los puntos de un eje logarítmico que llevan rótulo: los `obligatorios` (el primero, el
    último, el elegido) y, entre ellos, los que quedan a `separacion` décadas de los ya elegidos,
    así los números no se pisan con letra de proyección."""
    u = {p: np.log10(_x(eje, p)) for p in eje["puntos"]}
    elegidos = [p for p in eje["puntos"] if p in obligatorios]
    for p in eje["puntos"]:
        if p not in elegidos and all(abs(u[p] - u[q]) >= separacion for q in elegidos):
            elegidos.append(p)
    return set(elegidos)


def _marcar_elegido(ax, curva, eje, punto, gid="elegido"):
    """El valor elegido de un hiperparámetro: una estrella sobre la validación, la brecha con train
    como un segmento vertical, y un rótulo con el valor, su AUC y la brecha. Devuelve los
    artistas, para ocultarlos en los pasos anteriores de un revelado."""
    r = curva.resumen
    x = _x(eje, punto)
    va = float(r.at[punto, "media_validacion"])
    tr = float(r.at[punto, "media_train"])
    # Por debajo de los marcadores (zorder 3 y 6), así el segmento va de un punto al otro.
    brecha, = ax.plot([x, x], [va, tr], color=TINTA_SECUNDARIA, lw=2.4, zorder=2.5,
                      solid_capstyle="butt", gid=f"brecha:{gid}")
    estrella, = ax.plot([x], [va], ls="none", marker="*", ms=ESTRELLA,
                        color=COLOR_CONJUNTO["validacion"], mec=SUPERFICIE, mew=1.8, zorder=6,
                        gid=gid)
    return x, va, tr, [brecha, estrella]


def _rotular_elegido(ax, x, va, tr, texto_punto, gid="rotulo-elegido", preferido="abajo"):
    """El rótulo del valor elegido, en dos renglones: el valor con su AUC de validación, y la
    brecha con train. Primero donde se pide y después alrededor, siempre sin tapar las curvas."""
    texto = (f"{texto_punto}: AUC {numero(va, 3)}\n"
             f"brecha con train: {numero(tr - va, 3)}")
    base = {"abajo": [((x, va), 0, -22, "center", "top"), ((x, va), 16, -18, "left", "top"),
                      ((x, va), -16, -18, "right", "top")],
            "derecha": [((x, va), 18, -14, "left", "top"), ((x, va), 18, 0, "left", "center")],
            "izquierda": [((x, va), -18, -14, "right", "top"),
                          ((x, va), -18, 0, "right", "center")]}
    orden = [preferido] + [lado for lado in base if lado != preferido]
    candidatos = [c for lado in orden for c in base[lado]]
    candidatos += [((x, va), dx, dy, ha, "top") for dy in (-40, -60)
                   for dx, ha in ((0, "center"), (24, "left"), (-24, "right"))]
    return _colocar(ax, _creador(ax, texto, gid, fontsize=TAM_ROTULO + 1), candidatos)[0]


def _resaltar_marca(ax, x):
    """El rótulo del eje x del valor elegido, en negrita y en la tinta del texto."""
    for posicion, rotulo in zip(ax.xaxis.get_majorticklocs(), ax.get_xticklabels()):
        if np.isclose(posicion, x):
            rotulo.set_fontweight("bold")
            rotulo.set_color(TINTA)
            return rotulo
    return None


def _rotulo_zona(ax, texto, lado, y=0.5, gid=None):
    """El nombre de una zona de la curva (subajuste, sobreajuste) contra un borde de los ejes."""
    x, ha = (0.015, "left") if lado == "izquierda" else (0.985, "right")
    return ax.text(x, y, texto, transform=ax.transAxes, ha=ha, va="center",
                   fontsize=TAM_ROTULO + 1, fontstyle="italic", color=TINTA_SECUNDARIA,
                   zorder=6, gid=gid or f"zona:{lado}")


def _manijas_curva(extra=()):
    color_tr, color_va = COLOR_CONJUNTO["train"], COLOR_CONJUNTO["validacion"]
    comunes = dict(lw=LINEA, marker="o", ms=MARCADOR - 1, mec=SUPERFICIE)
    return [Line2D([], [], color=color_tr, label="train", **comunes),
            Line2D([], [], color=color_va, label="validación", **comunes),
            Patch(facecolor=TINTA_SECUNDARIA, alpha=ALFA_BANDA * 1.8, label="±1 desvío"),
            *extra]


def _estrella_leyenda(texto):
    return Line2D([], [], ls="none", marker="*", ms=ESTRELLA - 6,
                  color=COLOR_CONJUNTO["validacion"], mec=SUPERFICIE, label=texto)


def figura_curva_rf(tabla, paso=3, elegido=PROFUNDIDAD_ELEGIDA):
    """Slide 10, en tres pasos con el mismo encuadre (como la curva en U del TP1): 1, el AUC de
    train de RF según max_depth; 2, más el de validación; 3, más las bandas de ±1 desvío entre
    folds, las zonas de subajuste (izquierda) y sobreajuste (derecha), y la profundidad elegida
    (D-22) con su AUC y su brecha con train. `tabla` es resultados/curvas/rf_max_depth.csv."""
    if paso not in (1, 2, 3):
        raise ValueError("la curva de RF se revela en tres pasos: paso = 1, 2 o 3")
    curva = _curva(tabla)
    eje = eje_x(curva.parametro, list(curva.resumen.index))
    punto = _punto_elegido(eje, elegido)

    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    train = _serie_curva(ax, curva, eje, "train")
    validacion = _serie_curva(ax, curva, eje, "validacion")
    x, va, tr, marca = _marcar_elegido(ax, curva, eje, punto)
    _eje_x_curva(ax, eje)
    _limites_curvas(ax, [curva], abajo=0.2)
    _rotulo_eje(ax, x="max_depth: profundidad máxima de cada árbol",
                y=f"AUC (media de {_texto_folds(_n_folds(curva))})")
    leyenda = _leyenda_derecha(ax, _manijas_curva([_estrella_leyenda("profundidad\nelegida")]))

    _fijar_diseno(fig)
    zonas = [_rotulo_zona(ax, "← subajuste", "izquierda", 0.55, "zona:subajuste"),
             _rotulo_zona(ax, "sobreajuste →", "derecha", 0.55, "zona:sobreajuste")]
    rotulo = _rotular_elegido(ax, x, va, tr, f"profundidad {_rotulo_en_eje(eje, punto)}")
    por_paso = {2: validacion["linea"] + validacion["extremo"] + _entradas_leyenda(leyenda, [1]),
                3: (train["banda"] + validacion["banda"] + marca + zonas + rotulo
                    + _entradas_leyenda(leyenda, [2, 3]))}
    for p, artistas in por_paso.items():
        if p > paso:
            _ocultar(artistas)
    if paso >= 3:
        _resaltar_marca(ax, x)
    return fig


def _rotulo_en_eje(eje, punto):
    """Cómo se lee `punto` en el eje x: «8», «sin límite», «0,001»."""
    return eje["rotulos"][eje["puntos"].index(punto)]


def _n_folds(curva):
    columnas = [c for c in curva.resumen.columns if c.startswith("n_")]
    return max((curva.resumen[c].max() for c in columnas), default=None)


def figura_curva_knn(uniforme, distancia=None, elegido=VECINOS_ELEGIDOS):
    """Slide 11: el AUC de train y de validación de KNN con weights = uniform según n_neighbors,
    en escala logarítmica y con la grilla extendida, las bandas de ±1 desvío, el k elegido (D-22)
    con su brecha, y la zona de sobreajuste a la izquierda (k pequeño). Si se pasa `distancia`
    (weights = distance), sus dos curvas van atenuadas y a trazos, sin bandas, como comparación."""
    curva = _curva(uniforme)
    otra = _curva(distancia) if distancia is not None else None
    puntos = list(curva.resumen.index) + (list(otra.resumen.index) if otra else [])
    eje = eje_x(curva.parametro, puntos)
    punto = _punto_elegido(eje, elegido)

    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    if otra is not None:
        for conjunto in ("train", "validacion"):
            _serie_curva(ax, otra, eje, conjunto, estilo="--", alfa=ALFA_ATENUADA, bandas=False,
                         gid="distance")
    _serie_curva(ax, curva, eje, "train", gid="uniform")
    _serie_curva(ax, curva, eje, "validacion", gid="uniform")
    x, va, tr, _ = _marcar_elegido(ax, curva, eje, punto)
    obligatorios = {eje["puntos"][0], eje["puntos"][-1], punto}
    _eje_x_curva(ax, eje, _rotulados_log(eje, obligatorios) if eje["tipo"] == "log" else None)
    _limites_curvas(ax, [curva] + ([otra] if otra else []), abajo=0.1)
    _rotulo_eje(ax, x="n_neighbors: número de vecinos k (escala logarítmica)",
                y=f"AUC (media de {_texto_folds(_n_folds(curva))})")
    extra = [_estrella_leyenda("k elegido")]
    if otra is not None:
        extra.insert(0, Line2D([], [], color=TINTA_SECUNDARIA, ls="--", lw=LINEA,
                               alpha=ALFA_ATENUADA + 0.2, label="weights =\ndistance"))
    _leyenda_derecha(ax, _manijas_curva(extra))

    _fijar_diseno(fig)
    _rotulo_zona(ax, "← sobreajuste", "izquierda", 0.5, "zona:sobreajuste")
    _rotular_elegido(ax, x, va, tr, f"k = {_rotulo_en_eje(eje, punto)}")
    _resaltar_marca(ax, x)
    return fig


def figura_curva_svm(lineal, rbf=None, paso=1, elegido=C_ELEGIDO):
    """Slide 12, en dos pasos con el mismo encuadre. 1: el AUC de train y de validación de la SVM
    lineal con pesos balanceados según C (escala logarítmica), con bandas de ±1 desvío y el C
    elegido (D-22) con su brecha. 2: además, la SVM con kernel RBF y los mismos pesos, atenuada y a
    trazos, para comparar: en los datos del TP, con C grande sobreajusta y en ningún C supera a la
    lineal (D-22). El encuadre del paso 1 ya
    reserva el lugar de la RBF, así que hace falta `rbf` en los dos pasos para que coincidan; sin
    ella sólo se puede dibujar el paso 1, en su propio encuadre."""
    if paso not in (1, 2):
        raise ValueError("la curva de la SVM se revela en dos pasos: paso = 1 o 2")
    if paso == 2 and rbf is None:
        raise ValueError("el paso 2 de la curva de la SVM necesita la curva RBF")
    curva = _curva(lineal)
    otra = _curva(rbf) if rbf is not None else None
    puntos = list(curva.resumen.index) + (list(otra.resumen.index) if otra else [])
    eje = eje_x(curva.parametro, puntos)
    punto = _punto_elegido(eje, elegido)

    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    atenuadas = []
    if otra is not None:
        for conjunto in ("train", "validacion"):
            s = _serie_curva(ax, otra, eje, conjunto, estilo="--", alfa=ALFA_ATENUADA,
                             bandas=False, gid="rbf")
            atenuadas += s["linea"] + s["extremo"]
    _serie_curva(ax, curva, eje, "train", gid="lineal")
    _serie_curva(ax, curva, eje, "validacion", gid="lineal")
    x, va, tr, _ = _marcar_elegido(ax, curva, eje, punto)
    _eje_x_curva(ax, eje)
    _limites_curvas(ax, [curva] + ([otra] if otra else []), abajo=0.12 if otra else 0.3)
    _rotulo_eje(ax, x="C, escala logarítmica: un C mayor regulariza menos",
                y=f"AUC (media de {_texto_folds(_n_folds(curva))})")
    extra = [_estrella_leyenda("C elegido")]
    if otra is not None:
        extra.append(Line2D([], [], color=TINTA_SECUNDARIA, ls="--", lw=LINEA,
                            alpha=ALFA_ATENUADA + 0.2, label="kernel RBF"))
    leyenda = _leyenda_derecha(ax, _manijas_curva(extra))

    _fijar_diseno(fig)
    tardios = []
    if otra is not None:
        tardios.append(_rotulo_zona(ax, "sobreajuste con RBF →", "derecha", 0.62,
                                    "zona:sobreajuste"))
    _rotular_elegido(ax, x, va, tr, f"C = {_rotulo_en_eje(eje, punto)}", preferido="derecha")
    _resaltar_marca(ax, x)
    if otra is not None and paso < 2:
        _ocultar(atenuadas + tardios + _entradas_leyenda(leyenda, [4]))
    return fig


# --- Slides 18 y 19: el hallazgo H1 -------------------------------------------------------------

COLOR_ESQUEMA = {ESQUEMA_BARAJADO: AZUL, ESQUEMA_ADELANTE: NARANJA}
DESPLAZAMIENTO_MISMAS = 0.0  # el punto barajado en las mismas filas, en la x de su bloque
LEYENDA_ESQUEMA = {ESQUEMA_BARAJADO: "barajado", ESQUEMA_ADELANTE: "hacia\nadelante"}
DESPLAZAMIENTO_ESQUEMA = {ESQUEMA_BARAJADO: -0.17, ESQUEMA_ADELANTE: 0.17}


def _panel_pares(ax, t, modelos, rotulos):
    """Una fila por modelo: el AUC barajado arriba (azul) y el de hacia adelante abajo (naranja),
    media ± 1 desvío entre folds, y «sin modelo» en 0,5. Devuelve los puntos (x, y, desvío,
    esquema, modelo) para rotularlos con el diseño fijo."""
    _filas_modelo(ax, modelos, rotulos)
    _linea_base_vertical(ax)
    puntos = []
    for esquema in (ESQUEMA_BARAJADO, ESQUEMA_ADELANTE):
        filas = t[t["esquema"] == esquema].set_index("modelo")
        for i, m in enumerate(modelos):
            if m not in filas.index:
                continue
            media, desvio = float(filas.at[m, "media"]), float(filas.at[m, "desvio"])
            y = i + DESPLAZAMIENTO_ESQUEMA[esquema]
            _punto_con_barra(ax, media, y, desvio, COLOR_ESQUEMA[esquema],
                             gid=f"{esquema}:{m}")
            puntos.append((media, y, desvio, esquema, m))
    dibujadas = t[t["modelo"].isin(modelos)]
    lo = min(AUC_SIN_MODELO, float((dibujadas["media"] - dibujadas["desvio"]).min())) - 0.04
    hi = max(AUC_SIN_MODELO, float((dibujadas["media"] + dibujadas["desvio"]).max())) + 0.075
    ax.set_xlim(lo, hi)
    _marcas(ax, "x", 0.1, 1)
    _rotulo_eje(ax, x="AUC de validación, media ± 1 desvío entre folds")
    _leyenda_derecha(ax, [Line2D([], [], ls="none", marker="o", ms=MARCADOR + 1,
                                color=COLOR_ESQUEMA[e], mec=SUPERFICIE, label=LEYENDA_ESQUEMA[e])
                          for e in (ESQUEMA_BARAJADO, ESQUEMA_ADELANTE)])
    return puntos


def _rotular_pares(ax, puntos):
    for x, y, d, esquema, m in puntos:
        _valor_a_la_derecha(ax, x, y, d, numero(x, 3), f"valor:{esquema}:{m}")
    _rotular_vertical(ax, AUC_SIN_MODELO, ROTULO_LINEA_BASE, "rotulo-linea-base")


def figura_robustez_modelos(resumen, variante=VARIANTE_COMPLETA):
    """Slide 18: por modelo, el AUC de validación con los folds barajados de D-06 (azul) contra
    hacia adelante en el tiempo (naranja, D-25), con todas las variables, ordenados por el AUC
    barajado. `resumen` es robustez_temporal_resumen.csv de src/robustez.py."""
    t = _tabla_robustez(resumen)
    t = t[t["variante"] == variante]
    if t.empty:
        raise ValueError(f"el resumen de robustez no tiene la variante {variante}")
    barajado = t[t["esquema"] == ESQUEMA_BARAJADO].set_index("modelo")["media"]
    modelos = ordenar_modelos(t["modelo"])
    orden = sorted(modelos, key=lambda m: (-barajado.get(m, -np.inf), modelos.index(m)))
    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    puntos = _panel_pares(ax, t, orden, [rotulo_modelo(m) for m in orden])
    _fijar_diseno(fig)
    _rotular_pares(ax, puntos)
    return fig


def periodos_bloques(train, folds, names):
    """{fold: (meses, años)} de cada bloque de validación hacia adelante, según el mes y el año
    inferido de su primera y su última fila de train: ('may–jul', '2008') o
    ('nov–may', '2008–2009'). `folds` es robustez_folds.csv; `names`, leer_names()."""
    df = train.sort_values(FILA).reset_index(drop=True)
    anio = pd.Series(anio_inferido(df, names), index=df.index)
    salida = {}
    for fold, fila in _bloques(folds).items():
        dentro = df[FILA].between(fila["fila_min_validacion"], fila["fila_max_validacion"])
        if not dentro.any():
            continue
        primera, ultima = df.index[dentro][0], df.index[dentro][-1]
        mes0, mes1 = (MESES_ES[df.at[i, "month"]] for i in (primera, ultima))
        a0, a1 = int(anio[primera]), int(anio[ultima])
        meses = mes0 if (mes0, a0) == (mes1, a1) else f"{mes0}–{mes1}"
        salida[fold] = (meses, str(a0) if a0 == a1 else f"{a0}–{a1}")
    return salida


def auc_barajado_por_bloque(oof, train, folds):
    """El AUC de los puntajes fuera de fold del esquema barajado (oof_final_rf.csv: fila, fold,
    puntaje) en las filas de cada bloque de validación hacia adelante: qué tan bien ordena esas
    mismas filas un modelo que vio todas las épocas (resultados/conclusiones.md, «Cómo se
    calculó»). Devuelve una Series por fold; un bloque sin las dos clases queda en NaN."""
    faltan = [c for c in ("fila", "puntaje") if c not in oof.columns]
    if faltan:
        raise ValueError(f"a los puntajes fuera de fold les faltan columnas: {', '.join(faltan)}")
    y = train[[FILA, OBJETIVO]].rename(columns={FILA: "fila"})
    d = oof.merge(y, on="fila", how="inner", validate="one_to_one")
    salida = {}
    for fold, fila in _bloques(folds).items():
        b = d[d["fila"].between(fila["fila_min_validacion"], fila["fila_max_validacion"])]
        clases = (b[OBJETIVO] == "yes").astype(int)
        salida[fold] = (float(roc_auc_score(clases, b["puntaje"])) if clases.nunique() == 2
                        else np.nan)
    return pd.Series(salida, dtype=float).sort_index()


def figura_robustez_folds(largo, folds, mismas_filas=None, periodos=None, paso=1,
                          modelo=MODELO_ELEGIDO, variante=VARIANTE_COMPLETA):
    """Slide 19, en dos pasos con el mismo encuadre. 1: el AUC de validación de RF hacia adelante
    en cada bloque (naranja), unido por un tallo a «sin modelo» (0,5), con su valor, y el AUC
    barajado medio del mismo modelo (en el TP, 0,795) como referencia. 2: además, el AUC barajado
    en las mismas filas de cada bloque (azul, hueco; auc_barajado_por_bloque), la lectura (b) y
    (c) de H1: en los datos del TP, en 2008 ni barajado ordena bien, y en 2009–2010 hacia
    adelante pierde contra él. El eje x rotula cada bloque con su período (`periodos`, de
    periodos_bloques) y su % de «yes».

    `largo` es robustez_temporal.csv; `folds`, robustez_folds.csv. Los dos pasos necesitan
    `mismas_filas`: el paso 1 reserva su lugar para que el encuadre no cambie."""
    if paso not in (1, 2):
        raise ValueError("la figura por bloque se revela en dos pasos: paso = 1 o 2")
    if paso == 2 and mismas_filas is None:
        raise ValueError("el paso 2 necesita el AUC barajado en las mismas filas")
    faltan = [c for c in ("esquema", "variante", "modelo", "fold", "conjunto", "metrica",
                          "valor") if c not in largo.columns]
    if faltan:
        raise ValueError(f"a la robustez temporal le faltan columnas: {', '.join(faltan)}")
    adelante = _auc_por_fold(largo, modelo, ESQUEMA_ADELANTE, variante)
    if adelante.empty:
        raise ValueError(f"la robustez temporal no tiene el AUC hacia adelante de {modelo}")
    barajado = _auc_por_fold(largo, modelo, ESQUEMA_BARAJADO, variante)
    media_barajada = float(barajado.mean()) if len(barajado) else None
    bloques = _bloques(folds)
    periodos = periodos or {}
    xs = adelante.index.to_numpy(int).astype(float)
    ys = adelante.to_numpy(float)
    mismas = (pd.Series(mismas_filas, dtype=float).reindex(adelante.index)
              if mismas_filas is not None else None)

    fig, ejes = _lienzo()
    ax = ejes[0, 0]
    _linea_base_horizontal(ax)
    color = COLOR_ESQUEMA[ESQUEMA_ADELANTE]
    tardios = []
    # Una pesa por bloque, en la misma x: el barajado en las mismas filas (azul, hueco) arriba,
    # hacia adelante (naranja) abajo, y entre los dos un segmento gris que es lo que se pierde al
    # entrenar sólo con el pasado. La distancia del naranja a «sin modelo» dice si queda algo.
    if mismas is not None:
        ym = mismas.to_numpy(float)
        validos = np.isfinite(ym)
        tardios.append(ax.vlines(xs[validos], ys[validos], ym[validos], color=GRIS_REFERENCIA,
                                 lw=4.0, alpha=0.55, zorder=2, gid="perdida"))
        punto_azul, = ax.plot(xs + DESPLAZAMIENTO_MISMAS, ym, ls="none", marker="o",
                              ms=MARCADOR + 5, mfc=SUPERFICIE, mec=AZUL, mew=3.2, zorder=5,
                              gid="barajado-mismas-filas")
        tardios.append(punto_azul)
    ax.plot(xs, ys, ls="none", marker="o", ms=MARCADOR + 6, color=color, mec=SUPERFICIE,
            mew=ANILLO, zorder=6, gid="hacia-adelante")

    ax.set_xticks(xs, [_rotulo_bloque(int(f), bloques.get(int(f)), periodos.get(int(f)))
                       for f in xs])
    ax.set_xlim(xs.min() - 0.5, xs.max() + 0.55)
    ax.tick_params(axis="x", length=0, labelsize=TAM_MARCAS)
    ax.grid(axis="x", visible=False)
    extremos = [AUC_SIN_MODELO, *ys]
    extremos += list(mismas.dropna()) if mismas is not None else []
    lo, hi = min(extremos), max(extremos)
    alto = (hi - lo) or 0.1
    ax.set_ylim(lo - 0.12 * alto, hi + 0.2 * alto)
    _marcas(ax, "y", 0.1, 1)
    _rotulo_eje(ax, y="AUC de validación de RF")
    manijas = [Line2D([], [], ls="none", marker="o", ms=MARCADOR + 3, color=color,
                      mec=SUPERFICIE, label="hacia\nadelante"),
               Line2D([], [], ls="none", marker="o", ms=MARCADOR + 2, mfc=SUPERFICIE, mec=AZUL,
                      mew=3.0, label="barajado,\nmismas filas")]
    leyenda = _leyenda_derecha(ax, manijas if mismas is not None else manijas[:1])

    _fijar_diseno(fig)
    _rotular_horizontal(ax, AUC_SIN_MODELO, ROTULO_LINEA_BASE, "rotulo-linea-base",
                        preferido="izquierda")
    # Los valores: el naranja a la izquierda de su punto, el azul a la derecha del suyo. Se ubican
    # con los puntos azules ya dibujados, aunque el paso 1 los oculte: así nada se corre (B8).
    for x, y in zip(xs, ys):
        _colocar(ax, _creador(ax, numero(y, 3), f"valor:adelante:{int(x)}"),
                 _alrededor(x, y, dx=16, dy=14, preferido=("izquierda", "abajo", "derecha")))
    if mismas is not None:
        for x, y in zip(xs, mismas.to_numpy(float)):
            if np.isfinite(y):
                tardios += _colocar(ax, _creador(ax, numero(y, 3), f"valor:mismas:{int(x)}"),
                                    _alrededor(x + DESPLAZAMIENTO_MISMAS, y, dx=16, dy=14,
                                               preferido=("derecha", "arriba", "abajo")))[0]
    if paso < 2 and mismas is not None:
        _ocultar(tardios + _entradas_leyenda(leyenda, [1]))
    return fig


def _rotulo_bloque(fold, fila=None, periodo=None):
    """El rótulo del eje x de un bloque: su período en dos renglones (meses y años) y su % de
    «yes»; sin período, su número."""
    partes = list(periodo) if periodo else [f"bloque {int(fold)}"]
    if fila is not None and pd.notna(fila.get("pct_yes_validacion")):
        partes.append(f"{porcentaje(fila['pct_yes_validacion'], 1)} «yes»")
    return "\n".join(partes)


# --- Lectura de resultados/ y CLI ---------------------------------------------------------------

class EntradaFaltante(FileNotFoundError):
    """Un archivo de entrada que no existe: sin él, la figura no se puede dibujar."""


def _leer(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        raise EntradaFaltante(f"falta {ruta}")
    return leer_csv(ruta)


def _leer_curva(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        raise EntradaFaltante(f"falta {ruta}")
    return leer_curva(ruta)


def _leer_oof(ruta):
    """Los puntajes fuera de fold, con los flotantes tal como se escribieron (round_trip), como en
    resultados/conclusiones.md."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise EntradaFaltante(f"falta {ruta}")
    return pd.read_csv(ruta, float_precision="round_trip")


class _Contexto:
    """Lo que varias figuras comparten: el directorio de resultados, train y names.txt, que se
    leen una sola vez y sólo si alguna figura los pide."""

    def __init__(self, resultados):
        self.resultados = Path(resultados)
        self._train = None
        self._names = None

    @property
    def train(self):
        if self._train is None:
            self._train = cargar_train()
        return self._train

    @property
    def names(self):
        if self._names is None:
            self._names = leer_names()
        return self._names

    def csv(self, nombre):
        return _leer(self.resultados / nombre)

    def curva(self, nombre):
        return _leer_curva(self.resultados / "curvas" / f"{nombre}.csv")


def _dibujos(ctx):
    """{grupo: función que devuelve las Figure del grupo, en el orden de FIGURAS[grupo]}."""
    def curva_rf():
        tabla = ctx.curva("rf_max_depth")
        return [figura_curva_rf(tabla, paso) for paso in (1, 2, 3)]

    def curva_svm():
        lineal, rbf = ctx.curva("svm_C_linear_balanced"), ctx.curva("svm_C_balanced")
        return [figura_curva_svm(lineal, rbf, paso) for paso in (1, 2)]

    def robustez_folds():
        largo, folds = ctx.csv("robustez_temporal.csv"), ctx.csv("robustez_folds.csv")
        mismas = auc_barajado_por_bloque(_leer_oof(ctx.resultados / "oof_final_rf.csv"),
                                         ctx.train, folds)
        periodos = periodos_bloques(ctx.train, folds, ctx.names)
        return [figura_robustez_folds(largo, folds, mismas, periodos, paso) for paso in (1, 2)]

    return {
        "duration-techo": lambda: [figura_duration_techo(ctx.csv("ablaciones_resumen.csv"))],
        "orden-temporal": lambda: [figura_orden_temporal(ctx.train, ctx.names)],
        "pdays-999": lambda: [figura_pdays(ctx.train)],
        "modelos-referencia": lambda: [figura_modelos_referencia(
            ctx.csv("cv_referencia_resumen.csv"), zoom) for zoom in (False, True)],
        "curva-rf": curva_rf,
        "curva-knn": lambda: [figura_curva_knn(ctx.curva("knn_n_neighbors_uniform"),
                                               ctx.curva("knn_n_neighbors_distance"))],
        "curva-svm": curva_svm,
        "modelos-final": lambda: [figura_modelos_final(ctx.csv("cv_final_resumen.csv"), zoom)
                                  for zoom in (False, True)],
        "robustez-modelos": lambda: [figura_robustez_modelos(
            ctx.csv("robustez_temporal_resumen.csv"))],
        "robustez-folds": robustez_folds,
    }


def guardar(fig, ruta):
    """Guarda sin recorte (bbox_inches no), así todas miden ASPECTO_SLIDE × DPI píxeles."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(ruta, dpi=DPI, facecolor=SUPERFICIE)
    finally:
        plt.close(fig)
    return ruta


def generar(grupos, resultados, figuras):
    """Dibuja y guarda los grupos pedidos; devuelve (rutas escritas, errores). Un grupo que falla
    se informa y no frena a los demás."""
    ctx = _Contexto(resultados)
    dibujos = _dibujos(ctx)
    escritas, errores = [], []
    for grupo in grupos:
        try:
            figs = dibujos[grupo]()
        except Exception as error:  # noqa: BLE001 — se informa y el comando termina con código 1
            plt.close("all")
            errores.append(f"{grupo}: {type(error).__name__}: {error}")
            print(f"No se pudo dibujar {grupo}: {type(error).__name__}: {error}")
            continue
        for fig, nombre in zip(figs, FIGURAS[grupo]):
            ruta = guardar(fig, Path(figuras) / nombre)
            escritas.append(ruta)
            print(f"Escrita {_mostrar(ruta)} ({ruta.stat().st_size / 1024:.0f} KB)")
    return escritas, errores


def _argumentos():
    p = argparse.ArgumentParser(
        prog="python -m src.graficos_presentacion",
        description="Figuras de proyección del deck (paso 7.3) -> figuras/presentacion/.")
    p.add_argument("--solo", nargs="+", choices=list(FIGURAS), metavar="NOMBRE",
                   help="dibuja sólo esos grupos: " + ", ".join(FIGURAS))
    p.add_argument("--figuras", type=Path, default=DIR_PRESENTACION,
                   help="directorio de salida (por defecto figuras/presentacion)")
    p.add_argument("--resultados", type=Path, default=DIR_RESULTADOS,
                   help="directorio de entrada (por defecto resultados)")
    return p


def main(argv=None):
    a = _argumentos().parse_args(argv)
    grupos = list(dict.fromkeys(a.solo)) if a.solo else list(FIGURAS)
    escritas, errores = generar(grupos, a.resultados, a.figuras)
    resumen_final = (f"{len(escritas)} figura{'' if len(escritas) == 1 else 's'} en "
                     f"{_mostrar(a.figuras)}")
    if errores:
        resumen_final += f"; {len(errores)} con error: " + "; ".join(errores)
    print(resumen_final + ".")
    return 1 if errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
