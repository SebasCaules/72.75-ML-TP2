"""Gráficos del documento. Uso: python3 graficos.py (escribe en fig/).

Todos los datos se leen del CSV del TP2 (../bank-marketing/bank-additional-full.csv, sep=";"),
el mismo archivo contra el que se calcularon las cifras de la wiki
(wiki/fuentes/dataset-bank-marketing.md). Nada se copia a mano.
"""
import sys

sys.dont_write_bytecode = True  # sin __pycache__ en la carpeta del documento

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from estilo_graficos import P, SERIES, barras_h, figura, guardar, num, pct  # noqa: E402

CSV = Path(__file__).resolve().parent.parent / "bank-marketing" / "bank-additional-full.csv"
DF = pd.read_csv(CSV, sep=";")
Y = DF["y"].eq("yes")
GLOBAL = Y.mean() * 100
NO, SI = SERIES[0], SERIES[1]  # azul = no, naranja = yes, en todo el documento
GRIS = P.get("linea", "#b9c0cc")
TINTA = P.get("tinta", "#16233a")
APAGADO = P.get("apagado", "#5f6779")
MESES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def anio_inferido():
    """El archivo está ordenado por fecha (names.txt §4) y no trae el año: cada vez que el
    mes retrocede empieza un año nuevo. Da 2008 / 2009 / 2010 con cortes en las filas
    27 690 y 39 130, como en la wiki (dataset-bank-marketing §9)."""
    m = DF["month"].map(MESES.index).to_numpy()
    return 2008 + np.concatenate([[0], np.cumsum(np.diff(m) < 0)])


def _linea_global(ax, horizontal=False):
    kw = dict(color=APAGADO, lw=0.7, ls=(0, (3, 2)), zorder=0)
    (ax.axvline if horizontal else ax.axhline)(GLOBAL, **kw)


# 1. Balance de clases ------------------------------------------------------------------
# Fuente: CSV, value_counts de y (36 548 no, 4640 yes)
def balance():
    fig, ax = figura(alto_mm=19, ancho=0.46)
    n = DF["y"].value_counts()
    ax.barh([1, 0], [n["no"], n["yes"]], color=[NO, SI], height=0.62)
    ax.set_yticks([1, 0], ["no", "yes"])
    for yy, v in zip([1, 0], [n["no"], n["yes"]]):
        ax.text(v + 600, yy, f"{num(v)}  ({pct(v / len(DF) * 100, 2)})", va="center", color=TINTA)
    ax.set_xlim(0, n["no"] * 1.5)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("¿Contrató el depósito? (y)")
    guardar(fig, "f01_balance.pdf")


# 2. unknown por columna ------------------------------------------------------------------
# Fuente: CSV, (df[c] == "unknown").mean() por columna categórica
def unknown():
    fig, ax = figura(alto_mm=30, ancho=0.46)
    cols = [c for c in DF.select_dtypes("object") if c != "y" and DF[c].eq("unknown").any()]
    barras_h(ax, cols, [DF[c].eq("unknown").mean() * 100 for c in cols],
             destacar="default", fmt=lambda v: pct(v, 2))
    ax.set_title("Filas con «unknown», por columna")
    guardar(fig, "f02_unknown.pdf")


# 3. duration: tasa de yes por tramo -----------------------------------------------------------
# Fuente: CSV, pd.cut(duration, [0, 60, 120, 180, 300, 600, 1000, inf], right=False)
def duration():
    fig, ax = figura(alto_mm=40, ancho=0.5)
    cortes = [0, 60, 120, 180, 300, 600, 1000, np.inf]
    rot = ["<60", "60–\n119", "120–\n179", "180–\n299", "300–\n599", "600–\n999", "≥1000"]
    tr = pd.cut(DF["duration"], cortes, right=False, labels=rot)
    tasa = Y.groupby(tr, observed=False).mean() * 100
    barras = ax.bar(rot, tasa.values, color=SI, width=0.66)
    for b, v in zip(barras, tasa.values):
        ax.annotate(pct(v, 2 if v < 1 else 1), (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", color=TINTA, fontsize=7.5,
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9), zorder=3)
    _linea_global(ax)
    ax.text(-0.45, GLOBAL + 1.2, f"global {pct(GLOBAL, 2)}", ha="left", va="bottom",
            color=APAGADO, fontsize=7)
    ax.set_ylim(0, 70)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7, length=0)
    ax.set_xlabel("duración de la última llamada (segundos)")
    ax.set_title("% de «yes» según cuánto duró la llamada")
    guardar(fig, "f03_duration.pdf")


# 4. pdays = 999: qué hay adentro ---------------------------------------------------------
# Fuente: CSV, máscaras sobre pdays y previous (35 563 / 4110 / 1515)
def pdays():
    fig, ax = figura(alto_mm=24, ancho=0.5)
    c999 = DF["pdays"].eq(999)
    prev = DF["previous"].gt(0)
    etiquetas = ["999 y previous = 0\n(nunca contactado)", "999 y previous ≥ 1\n(sí contactado: failure)",
                 "pdays real (0–27)"]
    valores = [(c999 & ~prev).sum(), (c999 & prev).sum(), (~c999).sum()]
    barras_h(ax, etiquetas, valores, destacar=etiquetas[1], color_destacado=SI)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Qué esconde pdays = 999 (filas)")
    guardar(fig, "f04_pdays.pdf")


# 5. Tasa de yes por año inferido ---------------------------------------------------------
# Fuente: CSV + año inferido por reinicio del ciclo de meses (4,84 / 19,48 / 52,14 %)
def por_anio():
    fig, ax = figura(alto_mm=40, ancho=0.3)
    a = pd.Series(anio_inferido())
    tasa = Y.groupby(a).mean() * 100
    filas = a.value_counts().sort_index()
    xs = [f"{k}\n{num(filas[k])} filas" for k in tasa.index]
    barras = ax.bar(xs, tasa.values, color=SI, width=0.6)
    for b, v in zip(barras, tasa.values):
        ax.annotate(pct(v, 2), (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", color=TINTA)
    ax.set_ylim(0, 62)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=7, length=0)
    ax.set_title("% de «yes» por año")
    guardar(fig, "f05_por_anio.pdf")


# 6. La tasa a lo largo del archivo -----------------------------------------------------------
# Fuente: CSV, media móvil de 1000 filas y 5 pliegues contiguos (KFold sin barajar)
def orden_temporal():
    fig, ax = figura(alto_mm=50, ancho=1.0)
    idx = np.arange(len(DF))
    movil = Y.astype(float).rolling(1000, center=True, min_periods=500).mean() * 100
    ax.plot(idx, movil, color=SI, lw=1.4, label="media móvil de 1000 filas")
    for k, parte in enumerate(np.array_split(idx, 5)):
        v = Y.iloc[parte].mean() * 100
        ax.plot([parte[0], parte[-1]], [v, v], color=TINTA, lw=2.2,
                label="pliegue contiguo (k = 5, sin barajar)" if k == 0 else None)
        ax.text((parte[0] + parte[-1]) / 2, v + 2.2, pct(v, 1), ha="center", va="bottom",
                fontsize=7, color=TINTA, fontweight="semibold",
                bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
    a = anio_inferido()
    for corte in np.flatnonzero(np.diff(a)) + 1:
        ax.axvline(corte, color=APAGADO, lw=0.7, ls=(0, (3, 2)))
    for x, t in [(13800, "2008"), (33400, "2009"), (40150, "2010")]:
        ax.text(x, 66, t, ha="center", va="top", color=APAGADO, fontsize=7)
    ax.set_ylim(0, 68)
    ax.set_xlim(0, len(DF))
    ax.set_ylabel("% de «yes»")
    ax.set_xlabel("fila del archivo, en el orden en que viene")
    ax.xaxis.set_major_formatter(lambda v, _: num(v))
    ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, 0.93))
    ax.set_title("El archivo va de 2008 a 2010, y la tasa sube con el tiempo")
    guardar(fig, "f06_orden_temporal.pdf")


# 7. Correlación entre numéricas -------------------------------------------------------------
# Fuente: CSV, Pearson sobre las 10 numéricas
def correlacion():
    fig, ax = figura(alto_mm=82, ancho=0.56)
    num_cols = ["age", "duration", "campaign", "pdays", "previous", "cons.price.idx",
                "cons.conf.idx", "emp.var.rate", "euribor3m", "nr.employed"]
    c = DF[num_cols].corr()
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    ax.imshow(c.values, cmap=cmap, vmin=-1, vmax=1)
    n = len(num_cols)
    ax.set_xticks(range(n), num_cols, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(n), num_cols, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = c.values[i, j]
            v = 0.0 if abs(v) < 0.005 else v
            ax.text(j, i, f"{v:.2f}".replace(".", ",").replace("-", "−"), ha="center", va="center",
                    fontsize=6.2, color="white" if abs(v) > 0.75 else TINTA)
    bloque = [list(num_cols).index(x) for x in ("emp.var.rate", "euribor3m", "nr.employed")]
    lo, hi = min(bloque), max(bloque)
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1, fill=False, ec=TINTA, lw=1.1))
    ax.set_title("Pearson entre las 10 numéricas")
    guardar(fig, "f07_correlacion.pdf")


# 8. Quién domina una distancia sin escalar ---------------------------------------------------
# Fuente: CSV, var() de cada numérica sobre la suma de las diez (62,52 / 32,49 / 4,85 %)
def varianza():
    fig, ax = figura(alto_mm=30, ancho=0.46)
    v = DF.select_dtypes("number").var()
    cuota = (v / v.sum() * 100).sort_values(ascending=False)
    etiquetas = list(cuota.index[:3]) + ["las otras 7"]
    valores = list(cuota.values[:3]) + [cuota.values[3:].sum()]
    barras_h(ax, etiquetas, valores, destacar=["duration", "pdays"], fmt=lambda x: pct(x, 2))
    ax.set_title("Parte de la varianza total, sin escalar")
    guardar(fig, "f08_varianza.pdf")


# 9. Tasa de yes por categoría: tres columnas ----------------------------------------------------
# Fuente: CSV, groupby de y por nivel
def categorias():
    fig, axs = figura(alto_mm=46, ancho=1.0, ncols=3,
                      gridspec_kw={"width_ratios": [1.35, 2.6, 0.95], "wspace": 0.3})
    orden_mes = [m for m in MESES if m in set(DF["month"])]
    specs = [("poutcome", ["nonexistent", "failure", "success"]), ("month", orden_mes),
             ("contact", ["cellular", "telephone"])]
    for ax, (col, niveles) in zip(axs, specs):
        tasa = Y.groupby(DF[col]).mean().reindex(niveles) * 100
        cnt = DF[col].value_counts().reindex(niveles)
        colores = [GRIS if cnt[l] / len(DF) < 0.01 else NO for l in niveles]
        barras = ax.bar(range(len(niveles)), tasa.values, color=colores, width=0.66)
        for b, v in zip(barras, tasa.values):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 1.5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=TINTA)
        _linea_global(ax)
        ax.set_xticks(range(len(niveles)), niveles, fontsize=6.5)
        ax.set_ylim(0, 75)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_title(col, fontsize=8.5)
    axs[0].text(-0.5, 72, "% de «yes»", fontsize=7, color=APAGADO, va="top")
    guardar(fig, "f09_categorias.pdf")


# 10. Edad: una relación no lineal con y ---------------------------------------------------------
# Fuente: CSV, pd.cut(age, [17, 25, 30, 40, 50, 60, 99], right=False) y media de y por tramo
def edad():
    fig, ax = figura(alto_mm=40, ancho=0.39)
    cortes = [17, 25, 30, 40, 50, 60, 99]
    rot = ["<25", "25–29", "30–39", "40–49", "50–59", "≥60"]
    tr = pd.cut(DF["age"], cortes, right=False, labels=rot)
    tasa = Y.groupby(tr, observed=False).mean() * 100
    n = tr.value_counts().reindex(rot)
    barras = ax.bar(rot, tasa.values, color=NO, width=0.66)
    for b, v in zip(barras, tasa.values):
        ax.annotate(pct(v, 1), (b.get_x() + b.get_width() / 2, v), xytext=(0, 1.5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.8, color=TINTA,
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9), zorder=3)
    _linea_global(ax)
    ax.set_ylim(0, max(tasa.values) * 1.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.8, length=0)
    ax.set_xticks(range(len(rot)), [f"{r}\n{num(n[r])}" for r in rot])
    ax.set_title("% de «yes» por edad (y filas)")
    guardar(fig, "f10_edad.pdf")
    print("edad", tasa.round(2).to_dict())


if __name__ == "__main__":
    for f in (balance, unknown, duration, pdays, por_anio, orden_temporal, correlacion,
              varianza, categorias, edad):
        f()
    # control: las cifras que el texto cita
    a = pd.Series(anio_inferido())
    print("filas", len(DF), "yes %", round(GLOBAL, 2), "cortes", list(np.flatnonzero(np.diff(a.values)) + 1))
    print("por año", (Y.groupby(a).mean() * 100).round(2).to_dict())
    v = DF.select_dtypes("number").var()
    print("varianza", (v / v.sum() * 100).round(2).sort_values(ascending=False).head(3).to_dict())
    print("pdays", DF["pdays"].eq(999).sum(), (DF["pdays"].eq(999) & DF["previous"].gt(0)).sum())
