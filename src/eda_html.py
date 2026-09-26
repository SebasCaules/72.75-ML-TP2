"""Correr con: python -m src.eda_html            (opcional: --fragmento RUTA)

El EDA del punto 1 en una página HTML: todos los gráficos, qué se ve en cada uno y qué hacer
con eso. Complementa a src/eda.py (el reporte de texto) y, como él, lee sólo train con
cargar_train(): el conjunto de prueba no se abre aquí (DECISIONES.md, D-04).

Toda cifra del texto la calcula este script y entra por f-string; ninguna está escrita a mano.
Las frases que afirman un orden («el mes con más llamadas», «lo que más separa») se verifican
con assert: si train.csv cambiara, el script falla en vez de publicar una frase falsa.

Salida: resultados/eda/eda.html, un documento completo y autocontenido (los gráficos son PNG
de matplotlib embebidos en base64). Con --fragmento RUTA escribe además la misma página sin
<!doctype>, <html>, <head> ni <body>, que empieza por <title> y <style>, para una plataforma
que agrega esa envoltura.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from dataclasses import dataclass
from datetime import date
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.datos import EXCLUIDAS, FILA, OBJETIVO, PROP_TEST, RAIZ, RUTA_TRAIN, cargar_train  # noqa: E402
from src.metricas import PRESUPUESTO  # noqa: E402
from src.preproceso import FUSION_RARAS  # noqa: E402

SALIDA = RAIZ / "resultados" / "eda" / "eda.html"
RUTA_NAMES = RAIZ / "data" / "raw" / "bank-additional-names.txt"

# Parámetros del análisis: son elecciones, no resultados. Entran al texto por f-string.
PDAYS_CENTINELA = 999
UMBRAL_RARA = 0.01          # nivel «poco frecuente»: menos del 1 % de las filas
BLOQUE = 2000               # filas consecutivas del CSV original por punto
DUR_CORTA = 30              # segundos: llamadas «muy cortas»
N_DECILES = 10
CORTES_EDAD = [0, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
TOPE_CAMPAIGN = 11          # 11 o más contactos van juntos
TOPE_PREVIOUS = 3           # 3 o más contactos previos van juntos
POCOS_CONTACTOS = 3         # «3 contactos o menos»
MES_CHICO = 2.0             # % de filas: mes con pocas llamadas
TASA_ALTA = 40.0            # % de «yes»: tasa alta
DESVIO_CHICO = 1.0          # puntos porcentuales: variable «plana»
UMBRAL_COLINEAL = 0.9       # |r| desde el que dos variables son casi la misma
PLANAS = ["housing", "loan", "day_of_week"]
MACRO = ["emp.var.rate", "euribor3m", "nr.employed", "cons.price.idx", "cons.conf.idx"]
ORDEN_MESES = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
ORDEN_DIAS = ["mon", "tue", "wed", "thu", "fri"]
MESES_CALENDARIO = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                    "dec"]
MESES_EN_ES = {"January": "ene", "February": "feb", "March": "mar", "April": "abr", "May": "may",
               "June": "jun", "July": "jul", "August": "ago", "September": "sep",
               "October": "oct", "November": "nov", "December": "dic"}

# ------------------------------------------------------------------------------------------
# Formato de números: miles con punto, decimales con coma, espacio fino antes de %
# ------------------------------------------------------------------------------------------
NBSP = " "


def num(x, dec=0):
    """12345.6 -> '12.345,6'; los negativos llevan el signo menos tipográfico."""
    x = round(float(x), dec) + 0.0
    s = f"{abs(x):,.{dec}f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return ("−" if x < 0 else "") + s


def pct(x, dec=1):
    """x ya expresado en porcentaje: 11.27 -> '11,27 %'."""
    return f"{num(x, dec)}{NBSP}%"


def desvio(x):
    """Tres cifras significativas, aprox.: 0,50 / 1,57 / 10,5 / 186,6."""
    return num(x, 2 if x < 10 else 1)


def c(nombre):
    return f"<code>{escape(str(nombre))}</code>"


NUMERALES = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete",
             8: "ocho", 9: "nueve", 10: "diez"}


def palabra(n):
    return NUMERALES.get(int(n), num(n))


def lista(items, y="y"):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {y} " + items[-1]


# ------------------------------------------------------------------------------------------
# Diccionario de variables (bank-additional-names.txt): se lee, no se copia
# ------------------------------------------------------------------------------------------
def leer_names():
    texto = RUTA_NAMES.read_text(encoding="utf-8", errors="replace")
    periodo = re.search(r"ordered by date \(from (\w+) (\d{4}) to (\w+) (\d{4})\)", texto)
    mes_ini, anio_ini, mes_fin, anio_fin = periodo.groups()
    seccion = re.search(r"^(\d+)\.\s*Relevant Information", texto, flags=re.M).group(1)
    atributos = {nombre: int(n) for n, nombre in
                 re.findall(r"^\s*(\d+)\s*-\s*([\w.]+)", texto, flags=re.M)}
    cita_pdays = re.search(r"(\d+ means client was not previously contacted)", texto).group(1)
    return SimpleNamespace(
        mes_ini=MESES_EN_ES[mes_ini], anio_ini=int(anio_ini),
        mes_fin=MESES_EN_ES[mes_fin], anio_fin=int(anio_fin),
        mes_ini_dato=MESES_CALENDARIO[list(MESES_EN_ES).index(mes_ini)],
        mes_fin_dato=MESES_CALENDARIO[list(MESES_EN_ES).index(mes_fin)],
        seccion=seccion, atributo=atributos, cita_pdays=cita_pdays,
        periodo=f"{MESES_EN_ES[mes_ini]}-{anio_ini} a {MESES_EN_ES[mes_fin]}-{anio_fin}",
    )


# ------------------------------------------------------------------------------------------
# Estadísticos
# ------------------------------------------------------------------------------------------
def cramers_v(x, y):
    """V de Cramér (con corrección de sesgo) entre dos variables categóricas. Igual que en
    src/eda.py; se copia porque importar ese módulo corre el EDA de texto entero."""
    tabla = pd.crosstab(x, y).to_numpy()
    n = tabla.sum()
    esperado = tabla.sum(1, keepdims=True) @ tabla.sum(0, keepdims=True) / n
    chi2 = ((tabla - esperado) ** 2 / esperado).sum()
    r, k = tabla.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k_c - 1, r_c - 1))))


def auc_separacion(y, x):
    """AUC de la variable sola como puntaje, sin importar el sentido: max(AUC, 1 - AUC)."""
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def auc_tasa(y, x):
    """AUC de reemplazar cada nivel por su tasa de «yes» en los mismos datos. Descriptivo:
    se ajusta y se evalúa sobre train, no es un modelo."""
    return float(roc_auc_score(y, x.map(y.groupby(x, observed=True).mean()).astype(float)))


def anio_inferido(df, names):
    """El CSV está ordenado por fecha y no trae el año (names.txt, §4): cada vez que el mes
    retrocede empieza un año nuevo. train conserva ese orden (datos.partir lo reordena por
    `fila`), y quitar filas de una secuencia ordenada no crea retrocesos nuevos."""
    assert df[FILA].is_monotonic_increasing
    meses = df["month"].map(MESES_CALENDARIO.index).to_numpy()
    return names.anio_ini + np.concatenate([[0], np.cumsum(np.diff(meses) < 0)])


def analizar(df, names):
    r = SimpleNamespace()
    y = (df[OBJETIVO] == "yes").astype(int)
    r.y = y
    r.n = len(df)
    r.num = [col for col in df.select_dtypes("number").columns if col != FILA]
    r.cat = [col for col in df.select_dtypes(exclude="number").columns if col != OBJETIVO]
    r.num_modelo = [col for col in r.num if col not in EXCLUIDAS]
    r.n_pred = len(r.num) + len(r.cat)
    r.nan = int(df.isna().sum().sum())
    r.prop_train = 100 * (1 - PROP_TEST)

    # -- objetivo --------------------------------------------------------------------------
    r.n_yes = int(y.sum())
    r.n_no = r.n - r.n_yes
    r.tasa = 100 * r.n_yes / r.n
    r.acc_base = 100 * r.n_no / r.n
    r.ratio = r.n_no / r.n_yes
    r.rotulo_global = f"tasa global {pct(r.tasa, 1)}"

    # -- duration --------------------------------------------------------------------------
    d = df["duration"]
    r.auc_dur = roc_auc_score(y, d)
    corta = d < DUR_CORTA
    r.n_corta, r.yes_corta = int(corta.sum()), int(y[corta].sum())
    r.min_dur_yes = int(d[y == 1].min())
    dec = pd.qcut(d, N_DECILES)
    g = y.groupby(dec, observed=True)
    lims = d.groupby(dec, observed=True).agg(["min", "max"])
    r.dur_dec = pd.DataFrame({"lo": lims["min"].to_numpy(), "hi": lims["max"].to_numpy(),
                              "tasa": 100 * g.mean().to_numpy(), "n": g.size().to_numpy()})
    assert (np.diff(r.dur_dec["tasa"]) >= 0).all(), "«crece decil a decil» dejó de ser cierto"

    # -- tiempo ----------------------------------------------------------------------------
    bloque = df[FILA] // BLOQUE
    r.bloques = pd.DataFrame({
        "tasa": 100 * y.groupby(bloque).mean(),
        "x": df[FILA].groupby(bloque).mean(),
        "eur": df["euribor3m"].groupby(bloque).mean(),
    })
    r.anio = anio_inferido(df, names)
    anios = sorted(set(r.anio))
    assert anios[-1] == names.anio_fin
    assert df["month"].iloc[0] == names.mes_ini_dato and df["month"].iloc[-1] == names.mes_fin_dato
    r.anios = anios
    r.tasa_anio = {a: 100 * y[r.anio == a].mean() for a in anios}
    r.fronteras = [(df[FILA][r.anio == a - 1].max() + df[FILA][r.anio == a].min()) / 2
                   for a in anios[1:]]
    r.fila_max = int(df[FILA].max())

    mes = pd.DataFrame({"n": df["month"].value_counts(),
                        "tasa": 100 * y.groupby(df["month"]).mean(),
                        "eur": df.groupby("month")["euribor3m"].mean()}).reindex(ORDEN_MESES)
    mes["pct"] = 100 * mes["n"] / r.n
    r.mes = mes
    r.mes_max = mes["n"].idxmax()
    assert r.mes_max == mes["tasa"].idxmin(), "el mes con más llamadas ya no es el de menor tasa"
    r.meses_top = list(mes["n"].nlargest(4).index)
    r.meses_altos = [m for m in ORDEN_MESES
                     if mes.loc[m, "pct"] < MES_CHICO and mes.loc[m, "tasa"] > TASA_ALTA]
    en_altos = df["month"].isin(r.meses_altos)
    r.eur_altos = df.loc[en_altos, "euribor3m"].mean()
    r.eur_resto = df.loc[~en_altos, "euribor3m"].mean()
    r.altos_ult = 100 * (r.anio[en_altos.to_numpy()] > anios[0]).mean()
    assert r.altos_ult > 90, "«caen casi siempre después del primer año» dejó de ser cierto"

    # -- macro -----------------------------------------------------------------------------
    r.orden_corr = MACRO + [col for col in r.num if col not in MACRO]
    r.corr = df[r.orden_corr].corr()
    pares = [(a, b, r.corr.loc[a, b]) for i, a in enumerate(MACRO) for b in MACRO[i + 1:]]
    r.pares_colineales = sorted([p for p in pares if abs(p[2]) >= UMBRAL_COLINEAL],
                                key=lambda p: -abs(p[2]))
    r.trio = [v for v in MACRO if any(v in p[:2] for p in r.pares_colineales)]
    assert len(r.trio) == 3, "el texto habla de un trío («tres veces», «triplican»)"
    r.r_trio = [abs(p[2]) for p in r.pares_colineales]
    otras = [v for v in MACRO if v not in r.trio]
    r.r_medio = {v: [abs(r.corr.loc[v, t]) for t in r.trio] for v in otras}
    fuera = [(a, b, r.corr.loc[a, b]) for i, a in enumerate(r.num) for b in r.num[i + 1:]
             if not (a in MACRO and b in MACRO)]
    r.par_fuera = max(fuera, key=lambda p: abs(p[2]))

    dec_e = pd.qcut(df["euribor3m"], N_DECILES, duplicates="drop")
    g = y.groupby(dec_e, observed=True)
    lims = df["euribor3m"].groupby(dec_e, observed=True).agg(["min", "max"])
    r.eur_dec = pd.DataFrame({"lo": lims["min"].to_numpy(), "hi": lims["max"].to_numpy(),
                              "tasa": 100 * g.mean().to_numpy(), "n": g.size().to_numpy()})
    mitad = len(r.eur_dec) // 2
    r.eur_alta = r.eur_dec.iloc[mitad:]

    # -- campañas previas ------------------------------------------------------------------
    s999 = df["pdays"] == PDAYS_CENTINELA
    r.n999 = int(s999.sum())
    r.pct999 = 100 * r.n999 / r.n
    contra = s999 & (df["previous"] >= 1)
    r.n_contra = int(contra.sum())
    r.pout_contra = df.loc[contra, "poutcome"].value_counts()
    sin_previo = s999 & (df["previous"] == 0)
    assert int(((df["previous"] == 0) & ~s999).sum()) == 0
    r.n_sin_previo = int(sin_previo.sum())
    r.t_contra = 100 * y[contra].mean()
    r.t_sin_previo = 100 * y[sin_previo].mean()
    r.n_lt999 = int((~s999).sum())
    r.lt999 = pd.DataFrame({"n": df.loc[~s999, "poutcome"].value_counts(),
                            "tasa": 100 * y[~s999].groupby(df.loc[~s999, "poutcome"]).mean()})
    r.pdays_mediana = df.loc[~s999, "pdays"].median()
    r.pout = pd.DataFrame({"n": df["poutcome"].value_counts(),
                           "tasa": 100 * y.groupby(df["poutcome"]).mean()}
                          ).reindex(["nonexistent", "failure", "success"])
    prev = df["previous"].clip(upper=TOPE_PREVIOUS)
    r.prev = pd.DataFrame({"n": prev.value_counts(), "tasa": 100 * y.groupby(prev).mean()}
                          ).sort_index()
    r.pct_prev0 = 100 * (df["previous"] == 0).mean()

    # -- faltantes -------------------------------------------------------------------------
    filas = []
    for col in r.cat:
        u = df[col] == "unknown"
        if u.any():
            filas.append((col, int(u.sum()), 100 * u.mean(), 100 * y[u].mean(),
                          100 * y[~u].mean()))
    r.unk = pd.DataFrame(filas, columns=["col", "n", "pct", "t_unk", "t_con"]
                         ).sort_values("pct", ascending=False).reset_index(drop=True)
    hu, lu = df["housing"] == "unknown", df["loan"] == "unknown"
    r.hl_iguales = bool((hu == lu).all())
    r.n_hl = int(hu.sum())
    r.n_default_yes = int((df["default"] == "yes").sum())
    r.yes_default_yes = int(y[df["default"] == "yes"].sum())
    fila_def = r.unk.set_index("col").loc["default"]
    r.def_pct, r.def_tu, r.def_tc = fila_def["pct"], fila_def["t_unk"], fila_def["t_con"]
    resto = r.unk[r.unk["col"] != "default"]
    r.dif_resto = (resto["t_unk"] - resto["t_con"])
    assert (r.dif_resto > 0).all(), "«algo mayores» dejó de ser cierto"

    # -- categóricas -----------------------------------------------------------------------
    r.niveles = {}
    raras = []
    for col in r.cat:
        t = pd.DataFrame({"n": df[col].value_counts(), "tasa": 100 * y.groupby(df[col]).mean()})
        t["pct"] = 100 * t["n"] / r.n
        orden = {"month": ORDEN_MESES, "day_of_week": ORDEN_DIAS}.get(col)
        t = t.reindex([o for o in orden if o in t.index]) if orden else t.sort_values("tasa")
        r.niveles[col] = t
        for nivel, fila in t[t["pct"] < 100 * UMBRAL_RARA].iterrows():
            raras.append((col, nivel, int(fila["n"]), fila["pct"], fila["tasa"]))
    r.raras = raras
    r.max_raras_col = max(pd.Series([x[0] for x in raras]).value_counts())
    r.v = pd.Series({col: cramers_v(df[col], df[OBJETIVO]) for col in r.cat}
                    ).sort_values(ascending=False)
    r.desvio_planas = {col: (r.niveles[col]["tasa"] - r.tasa).abs().max() for col in PLANAS}
    assert max(r.desvio_planas.values()) < DESVIO_CHICO
    r.niveles_altos = [(col, nivel, fila["tasa"]) for col in r.cat
                       for nivel, fila in r.niveles[col].iterrows() if fila["tasa"] > TASA_ALTA]
    r.n_onehot = int(sum(df[col].nunique() for col in r.cat))
    r.dims = r.n_onehot + len(r.num_modelo)
    r.dec = r.niveles["month"].loc["dec"]

    # -- numéricas -------------------------------------------------------------------------
    etiquetas_edad = []
    for lo, hi in zip(CORTES_EDAD[:-1], CORTES_EDAD[1:]):
        etiquetas_edad.append(f"<{hi}" if lo == 0 else f"{lo}+" if hi == np.inf
                              else f"{lo}–{hi - 1}")
    tramo = pd.cut(df["age"], CORTES_EDAD, right=False, labels=etiquetas_edad)
    r.edad = pd.DataFrame({"n": tramo.value_counts(sort=False),
                           "tasa": 100 * y.groupby(tramo, observed=False).mean()})
    r.edad_min = r.edad["tasa"].idxmin()
    assert r.edad["tasa"].iloc[0] > r.edad.loc[r.edad_min, "tasa"] < r.edad["tasa"].iloc[-1]
    r.auc_age = auc_separacion(y, df["age"])
    r.auc_age_tramos = auc_tasa(y, tramo)

    camp = df["campaign"].clip(upper=TOPE_CAMPAIGN)
    r.camp = pd.DataFrame({"n": camp.value_counts(), "tasa": 100 * y.groupby(camp).mean()}
                          ).sort_index()
    r.camp["pct"] = 100 * r.camp["n"] / r.n
    r.asim = df[r.num].skew()
    r.camp_pocos = 100 * (df["campaign"] <= POCOS_CONTACTOS).mean()
    r.camp_max = int(df["campaign"].max())
    r.mediana = df.groupby(OBJETIVO)[r.num].median()
    r.unicos = df[r.num].nunique()

    # -- escalas ---------------------------------------------------------------------------
    r.sd = df[r.num].std()
    sd_m = r.sd[r.num_modelo]
    r.sd_max, r.sd_min = sd_m.idxmax(), sd_m.idxmin()
    r.sd_ratio = sd_m.max() / sd_m.min()
    var = df[r.num_modelo].var()
    r.var_share = (100 * var / var.sum()).sort_values(ascending=False)
    r.var_top2 = r.var_share.iloc[:2].sum()

    # -- ranking ---------------------------------------------------------------------------
    r.auc_num = {col: auc_separacion(y, df[col]) for col in r.num}
    r.auc_cat = {col: auc_tasa(y, df[col]) for col in r.cat}
    r.ranking = pd.Series({**r.auc_num, **r.auc_cat}).sort_values(ascending=False)
    sin_fuga = r.ranking.drop(EXCLUIDAS, errors="ignore")
    r.top2 = list(sin_fuga.index[:2])
    assert all(v in MACRO for v in r.top2), "lo que más separa ya no es el contexto económico"
    assert r.ranking.index[0] == "duration"
    return r


# ------------------------------------------------------------------------------------------
# Gráficos: paleta del proyecto; «no» azul y «yes» naranja en todos
# ------------------------------------------------------------------------------------------
P = {"azul": "#2A78D6", "naranja": "#EB6834", "azuloscuro": "#104281", "tinta": "#0B0B0B",
     "tintasec": "#52514E", "apagado": "#898781", "rejilla": "#E1E0D9", "superficie": "#FCFCFB"}
NO, SI = P["azul"], P["naranja"]
ANCHO_FIG = 9.0   # pulgadas; la página lo muestra a ~880 px
DPI = 160
FUENTES = ["Fira Sans", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
LINEA_GLOBAL = dict(color=P["tintasec"], lw=1.3, ls=(0, (4, 3)), zorder=4)
# Fondo de los rótulos de valor: tapan la línea de referencia donde la cruzan.
FONDO_ROTULO = dict(boxstyle="square,pad=0.12", fc=P["superficie"], ec="none")
RAYADO = dict(facecolor=P["superficie"], edgecolor=P["apagado"], hatch="////", lw=1.0)


def estilo():
    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    familias = [f for f in FUENTES if f in instaladas] or ["DejaVu Sans"]
    if "DejaVu Sans" not in familias:
        familias.append("DejaVu Sans")   # respaldo para glifos que falten (≥, −)
    plt.rcParams.update({
        "font.family": familias,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "figure.facecolor": P["superficie"],
        "axes.facecolor": P["superficie"],
        "savefig.facecolor": P["superficie"],
        "axes.edgecolor": P["apagado"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": P["tintasec"],
        "axes.titlecolor": P["tinta"],
        "axes.axisbelow": True,
        "text.color": P["tinta"],
        "xtick.color": P["apagado"],
        "ytick.color": P["apagado"],
        "xtick.labelcolor": P["tintasec"],
        "ytick.labelcolor": P["tintasec"],
        "grid.color": P["rejilla"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "hatch.color": P["apagado"],
        "hatch.linewidth": 0.8,
        "lines.solid_capstyle": "round",
        "figure.constrained_layout.h_pad": 0.06,
        "figure.constrained_layout.w_pad": 0.06,
    })


@dataclass
class Grafico:
    b64: str
    ancho: int
    alto: int
    bytes: int
    alt: str


def png(fig, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, pil_kwargs={"optimize": True})
    plt.close(fig)
    datos = buf.getvalue()
    ancho, alto = Image.open(io.BytesIO(datos)).size
    return Grafico(base64.b64encode(datos).decode("ascii"), ancho, alto, len(datos), alt)


def figura(alto, **kw):
    return plt.subplots(figsize=(ANCHO_FIG, alto), layout="constrained", **kw)


def eje_pct(eje, dec=0):
    # pasos de 1, 2, 5 o 10: con 2,5 el rótulo redondeado mentiría («2 %» en 2,5)
    eje.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    eje.set_major_formatter(FuncFormatter(lambda v, _: pct(v, dec)))


def eje_miles(eje):
    eje.set_major_formatter(FuncFormatter(lambda v, _: num(v)))


def h_tasa(etiqueta="% de «yes»"):
    return Patch(facecolor=SI, label=etiqueta)


def h_global(r):
    return Line2D([], [], label=r.rotulo_global, **LINEA_GLOBAL)


def leyenda(fig, handles, ncols=None):
    fig.legend(handles=handles, loc="outside upper left", ncols=ncols or len(handles),
               handlelength=1.8, columnspacing=1.8, borderaxespad=0.2)


def rotular_columnas(ax, xs, valores, fmt, dy=3, **kw):
    for x, v in zip(xs, valores):
        ax.annotate(fmt(v), (x, v), xytext=(0, dy), textcoords="offset points", ha="center",
                    va="bottom", fontsize=kw.get("fontsize", 11), color=P["tinta"],
                    bbox=FONDO_ROTULO, zorder=6)


def rotular_barras(ax, ys, valores, textos, dx=4, fontsize=11):
    for yy, v, t in zip(ys, valores, textos):
        ax.annotate(t, (v, yy), xytext=(dx, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=fontsize, color=P["tinta"], bbox=FONDO_ROTULO,
                    zorder=6)


def columnas_tasa(ax, r, etiquetas, tasas, ancho=0.62, fmt=None):
    xs = np.arange(len(etiquetas))
    ax.bar(xs, tasas, width=ancho, color=SI, zorder=2)
    rotular_columnas(ax, xs, tasas, fmt or (lambda v: pct(v, 1)))
    ax.axhline(r.tasa, **LINEA_GLOBAL)
    ax.set_xticks(xs, etiquetas)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(tasas) * 1.17)
    eje_pct(ax.yaxis)
    ax.grid(axis="y")
    return xs


# -- a. objetivo ---------------------------------------------------------------------------
def g_balance(r):
    fig, ax = figura(1.9)
    ys, vals = [1, 0], [r.n_no, r.n_yes]
    ax.barh(ys, vals, color=[NO, SI], height=0.58)
    rotular_barras(ax, ys, vals, [f"{num(v)} filas · {pct(100 * v / r.n, 1)}" for v in vals],
                   dx=6, fontsize=12.5)
    ax.set_yticks(ys, ["no", "yes"])
    ax.tick_params(axis="y", length=0, labelsize=13)
    ax.set_xlim(0, r.n_no * 1.36)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    return png(fig, f"Barras horizontales de y: {num(r.n_no)} filas «no» "
                    f"({pct(r.acc_base, 1)}) y {num(r.n_yes)} filas «yes» ({pct(r.tasa, 1)}).")


# -- b. fuga -------------------------------------------------------------------------------
def rotulo_rango(lo, hi, i, n, dec=0, unidad=""):
    if i == 0:
        return f"≤{NBSP}{num(hi, dec)}{unidad}"
    if i == n - 1:
        return f"≥{NBSP}{num(lo, dec)}{unidad}"
    return f"{num(lo, dec)}–{num(hi, dec)}{unidad}"


def g_duration(r):
    fig, ax = figura(3.9)
    t = r.dur_dec
    rot = [rotulo_rango(lo, hi, i, len(t)) for i, (lo, hi) in enumerate(zip(t["lo"], t["hi"]))]
    columnas_tasa(ax, r, rot, t["tasa"].to_numpy())
    ax.set_xlabel("Duración de la última llamada, por decil (segundos)")
    ax.set_ylabel("% de «yes»")
    leyenda(fig, [h_tasa(), h_global(r)])
    return png(fig, f"Tasa de «yes» por decil de duration: {pct(t['tasa'].iloc[0], 1)} en el "
                    f"decil más corto y {pct(t['tasa'].iloc[-1], 1)} en el más largo.")


# -- c. tiempo -----------------------------------------------------------------------------
def g_bloques(r, df):
    fig, (a1, a2) = figura(5.6, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.35, 1]})
    b = r.bloques
    a1.plot(b["x"], b["tasa"], color=SI, lw=2, marker="o", ms=7, mec=P["superficie"], mew=1.6,
            zorder=5)
    a1.axhline(r.tasa, **LINEA_GLOBAL)
    for (x, v), dx, ha in [((b["x"].iloc[0], b["tasa"].iloc[0]), 4, "left"),
                           ((b["x"].iloc[-1], b["tasa"].iloc[-1]), 4, "right")]:
        a1.annotate(pct(v, 1), (x, v), xytext=(dx, 10), textcoords="offset points", ha=ha,
                    va="bottom", fontsize=11.5, color=P["tinta"], bbox=FONDO_ROTULO, zorder=6)
    a1.set_ylim(0, b["tasa"].max() * 1.32)
    a1.set_ylabel("% de «yes»")
    eje_pct(a1.yaxis)
    a1.grid(axis="y")
    bordes = [0, *r.fronteras, r.fila_max]
    arriba = blended_transform_factory(a1.transData, a1.transAxes)
    for anio, x0, x1 in zip(r.anios, bordes[:-1], bordes[1:]):
        a1.text((x0 + x1) / 2, 0.97, str(anio), transform=arriba, ha="center", va="top",
                fontsize=11.5, color=P["tintasec"])
    for ax in (a1, a2):
        for x in r.fronteras:
            ax.axvline(x, color=P["apagado"], lw=1.0, zorder=1)
    a2.plot(df[FILA], df["euribor3m"], color=P["tintasec"], lw=1.4)
    a2.set_ylabel("euribor3m")
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 0)))
    a2.grid(axis="y")
    a2.set_xlabel("Fila del CSV original (ordenado por fecha)")
    a2.set_xlim(0, r.fila_max)
    eje_miles(a2.xaxis)
    leyenda(fig, [Line2D([], [], color=SI, lw=2, marker="o", ms=7, mec=P["superficie"],
                         label=f"% de «yes» por bloque de {num(BLOQUE)} filas"),
                  h_global(r),
                  Line2D([], [], color=P["apagado"], lw=0, marker="|", ms=13, mew=1.2,
                         label="cambio de año (inferido)")])
    return png(fig, f"Arriba, tasa de «yes» por bloque de {num(BLOQUE)} filas del CSV original, "
                    f"de {pct(b['tasa'].iloc[0], 1)} a {pct(b['tasa'].iloc[-1], 1)}. Abajo, "
                    f"euribor3m fila por fila, que baja de {num(b['eur'].iloc[0], 2)} a "
                    f"{num(b['eur'].min(), 2)}.")


def g_meses(r):
    fig, (a1, a2) = figura(5.2, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    m = r.mes
    xs = np.arange(len(m))
    a1.bar(xs, m["n"], width=0.62, color=P["apagado"], zorder=2)
    rotular_columnas(a1, xs, m["n"], lambda v: pct(100 * v / r.n, 1))
    a1.set_ylim(0, m["n"].max() * 1.22)
    a1.set_ylabel("Filas")
    eje_miles(a1.yaxis)
    a1.grid(axis="y")
    columnas_tasa(a2, r, list(m.index), m["tasa"].to_numpy())
    a2.set_ylabel("% de «yes»")
    a2.set_xlabel("month, en orden de calendario")
    leyenda(fig, [Patch(facecolor=P["apagado"], label="filas del mes (el rótulo: % del total)"),
                  h_tasa(), h_global(r)])
    return png(fig, f"Filas y tasa de «yes» por mes. {r.mes_max} tiene el "
                    f"{pct(m.loc[r.mes_max, 'pct'], 1)} de las filas y "
                    f"{pct(m.loc[r.mes_max, 'tasa'], 1)} de «yes»; "
                    f"{lista(r.meses_altos)} superan el {pct(TASA_ALTA, 0)}.")


# -- d. contexto económico -----------------------------------------------------------------
def g_correlacion(r):
    fig, ax = figura(7.8)
    corr = r.corr
    cmap = LinearSegmentedColormap.from_list("r_abs", [P["superficie"], P["azuloscuro"]])
    im = ax.imshow(corr.abs().to_numpy(), cmap=cmap, vmin=0, vmax=1)
    k = len(corr)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            v = corr.iat[i, j]
            ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=11,
                    color="white" if abs(v) > 0.55 else P["tinta"])
    ax.set_xticks(range(k), corr.columns, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(k), corr.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((-0.5, -0.5), len(MACRO), len(MACRO), fill=False,
                           ec=P["tinta"], lw=2.2, zorder=5))
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02, aspect=28)
    cb.set_label("|r|  (el número de cada celda conserva el signo)")
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 1)))
    leyenda(fig, [Patch(fill=False, ec=P["tinta"], lw=2.2, label="bloque macro")])
    return png(fig, f"Matriz de correlación de Pearson de las {k} numéricas, con el bloque macro "
                    f"arriba a la izquierda: r entre {num(min(r.r_trio), 2)} y "
                    f"{num(max(r.r_trio), 2)} entre {lista(r.trio)}.")


def g_euribor(r):
    fig, ax = figura(4.7)
    t = r.eur_dec
    ys = np.arange(len(t))
    ax.barh(ys, t["tasa"], height=0.62, color=SI, zorder=2)
    rotular_barras(ax, ys, t["tasa"], [pct(v, 1) for v in t["tasa"]])
    ax.axvline(r.tasa, **LINEA_GLOBAL)
    ax.set_yticks(ys, [f"{num(lo, 3)}–{num(hi, 3)}" for lo, hi in zip(t["lo"], t["hi"])])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, t["tasa"].max() * 1.15)
    eje_pct(ax.xaxis)
    ax.grid(axis="x")
    ax.set_xlabel("% de «yes»")
    ax.set_ylabel("Decil de euribor3m (de menor a mayor)")
    leyenda(fig, [h_tasa(), h_global(r)])
    return png(fig, f"Tasa de «yes» por decil de euribor3m: {pct(t['tasa'].iloc[0], 1)} en el "
                    f"decil más bajo y entre {pct(r.eur_alta['tasa'].min(), 1)} y "
                    f"{pct(r.eur_alta['tasa'].max(), 1)} en la mitad alta.")


# -- e. campañas previas -------------------------------------------------------------------
def g_previas(r):
    fig, (a1, a2) = figura(4.3, ncols=2, sharey=True, gridspec_kw={"width_ratios": [3, 4]})
    po, pv = r.pout, r.prev
    columnas_tasa(a1, r, [f"{i}\n{num(n)} filas" for i, n in zip(po.index, po["n"])],
                  po["tasa"].to_numpy())
    etiquetas_prev = [f"{i}{'+' if i == TOPE_PREVIOUS else ''}\n{num(n)} filas"
                      for i, n in zip(pv.index, pv["n"])]
    columnas_tasa(a2, r, etiquetas_prev, pv["tasa"].to_numpy())
    tope = max(po["tasa"].max(), pv["tasa"].max()) * 1.17
    a1.set_ylim(0, tope)
    a1.set_title("poutcome")
    a2.set_title("previous (contactos antes de esta campaña)")
    a1.set_ylabel("% de «yes»")
    leyenda(fig, [h_tasa(), h_global(r)])
    return png(fig, f"Tasa de «yes» por poutcome ({pct(po.loc['success', 'tasa'], 1)} con "
                    f"success) y por previous ({pct(pv['tasa'].iloc[0], 1)} con 0 contactos, "
                    f"{pct(pv['tasa'].iloc[-1], 1)} con {TOPE_PREVIOUS} o más).")


def g_pdays(r):
    fig, ax = figura(3.2)
    gris, oscuro, borde = P["apagado"], P["azuloscuro"], P["superficie"]
    alto = 0.56
    # barra de arriba: pdays = 999
    f0 = r.n_sin_previo / r.n999
    ax.barh(1, f0, height=alto, color=gris, edgecolor=borde, lw=2)
    ax.barh(1, 1 - f0, left=f0, height=alto, color=oscuro, edgecolor=borde, lw=2)
    ax.text(f0 / 2, 1, f"previous = 0 · poutcome = nonexistent\n{num(r.n_sin_previo)} filas · "
                       f"{pct(r.t_sin_previo, 1)} de «yes»",
            ha="center", va="center", fontsize=11, color=P["tinta"])
    pout = lista(r.pout_contra.index)
    ax.annotate(f"previous ≥ 1 · poutcome = {pout}\n{num(r.n_contra)} filas · "
                f"{pct(r.t_contra, 1)} de «yes»", (1, 1), xytext=(8, 0),
                textcoords="offset points", ha="left", va="center", fontsize=11,
                color=P["tinta"])
    # barra de abajo: pdays < 999 (todas con previous ≥ 1)
    lt = r.lt999.reindex(["success", "failure"])
    f_s = lt.loc["success", "n"] / r.n_lt999
    ax.barh(0, f_s, height=alto, color=oscuro, edgecolor=borde, lw=2)
    ax.barh(0, 1 - f_s, left=f_s, height=alto, color=oscuro, edgecolor=borde, lw=2)
    ax.text(f_s / 2, 0, f"previous ≥ 1 · poutcome = success\n{num(lt.loc['success', 'n'])} "
                        f"filas · {pct(lt.loc['success', 'tasa'], 1)} de «yes»",
            ha="center", va="center", fontsize=11, color="white")
    ax.annotate(f"previous ≥ 1 · poutcome = failure\n{num(lt.loc['failure', 'n'])} filas · "
                f"{pct(lt.loc['failure', 'tasa'], 1)} de «yes»", (1, 0), xytext=(8, 0),
                textcoords="offset points", ha="left", va="center", fontsize=11,
                color=P["tinta"])
    ax.set_yticks([1, 0], [f"pdays = {PDAYS_CENTINELA}\n{num(r.n999)} filas",
                           f"pdays < {PDAYS_CENTINELA}\n{num(r.n_lt999)} filas"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], [pct(v, 0) for v in (0, 25, 50, 75, 100)])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xlabel("Proporción de las filas del grupo", loc="left")
    leyenda(fig, [Patch(facecolor=gris, label="sin contactos previos (previous = 0)"),
                  Patch(facecolor=oscuro, label="con contactos previos (previous ≥ 1)")])
    return png(fig, f"De las {num(r.n999)} filas con pdays = {PDAYS_CENTINELA}, "
                    f"{num(r.n_contra)} tienen previous ≥ 1 y poutcome = {pout}.")


# -- f. faltantes --------------------------------------------------------------------------
def g_unknown(r):
    u = r.unk
    fig, (a1, a2) = figura(3.9, ncols=2, sharey=True, gridspec_kw={"width_ratios": [1, 1.2]})
    ys = np.arange(len(u))
    a1.barh(ys, u["pct"], height=0.6, color=P["azuloscuro"], zorder=2)
    textos = [f"{pct(p, 1)} · {num(n)}" for p, n in zip(u["pct"], u["n"])]
    if r.hl_iguales:
        textos = [t + ("  (mismas filas)" if col in ("housing", "loan") else "")
                  for t, col in zip(textos, u["col"])]
    rotular_barras(a1, ys, u["pct"], textos, fontsize=11)
    a1.set_yticks(ys, u["col"])
    a1.invert_yaxis()
    a1.tick_params(axis="y", length=0)
    a1.set_xlim(0, u["pct"].max() * 1.75)
    eje_pct(a1.xaxis)
    a1.grid(axis="x")
    a1.set_xlabel("% de filas con «unknown» · filas")
    for yy, tu, tc in zip(ys, u["t_unk"], u["t_con"]):
        a2.plot([tc, tu], [yy, yy], color=P["rejilla"], lw=2.2, zorder=2)
    a2.scatter(u["t_con"], ys, s=70, color=P["apagado"], edgecolor=P["superficie"], lw=1.6,
               zorder=3)
    a2.scatter(u["t_unk"], ys, s=70, color=P["azuloscuro"], edgecolor=P["superficie"], lw=1.6,
               zorder=4)
    for yy, tu, tc in zip(ys, u["t_unk"], u["t_con"]):
        a2.annotate(pct(tu, 1), (tu, yy), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, color=P["tinta"],
                    bbox=FONDO_ROTULO, zorder=6)
    a2.axvline(r.tasa, **LINEA_GLOBAL)
    a2.set_xlim(0, max(u["t_unk"].max(), u["t_con"].max()) * 1.2)
    a2.set_ylim(len(u) - 0.4, -0.75)
    eje_pct(a2.xaxis)
    a2.grid(axis="x")
    a2.set_xlabel("% de «yes»")
    leyenda(fig, [Line2D([], [], lw=0, marker="o", ms=9, color=P["azuloscuro"],
                         label="«yes» si es «unknown»"),
                  Line2D([], [], lw=0, marker="o", ms=9, color=P["apagado"],
                         label="«yes» si se conoce"),
                  h_global(r)])
    return png(fig, f"Columnas con «unknown»: default lo tiene en el {pct(r.def_pct, 1)} de las "
                    f"filas, con {pct(r.def_tu, 1)} de «yes» frente a {pct(r.def_tc, 1)} "
                    f"cuando se conoce.")


# -- g. categorías -------------------------------------------------------------------------
def g_categorias(r):
    mitad = []
    total = sum(len(r.niveles[col]) for col in r.cat)
    acumulado = 0
    for col in r.cat:   # columna izquierda hasta cubrir ~la mitad de las barras
        if acumulado < total / 2:
            mitad.append(col)
            acumulado += len(r.niveles[col])
    columnas = [mitad, [col for col in r.cat if col not in mitad]]
    alto = 1.0 + max(sum(0.3 * len(r.niveles[col]) + 0.5 for col in cols) for cols in columnas)
    fig = plt.figure(figsize=(ANCHO_FIG, alto), layout="constrained")
    subfigs = fig.subfigures(1, 2, wspace=0.03)
    xmax = max(t["tasa"].max() for t in r.niveles.values()) * 1.42
    for sub, cols in zip(subfigs, columnas):
        axs = sub.subplots(len(cols), 1, gridspec_kw={
            "height_ratios": [len(r.niveles[col]) + 0.5 for col in cols]})
        for ax, col in zip(np.atleast_1d(axs), cols):
            t = r.niveles[col]
            ys = np.arange(len(t))
            rara = (t["pct"] < 100 * UMBRAL_RARA).to_numpy()
            for yy, v, es_rara in zip(ys, t["tasa"], rara):
                if es_rara:
                    ax.barh(yy, v, height=0.64, zorder=2, **RAYADO)
                else:
                    ax.barh(yy, v, height=0.64, color=SI, zorder=2)
            textos = [pct(v, 1) + (f" · {num(n)} filas" if es_rara else "")
                      for v, n, es_rara in zip(t["tasa"], t["n"], rara)]
            rotular_barras(ax, ys, t["tasa"], textos, fontsize=11)
            ax.axvline(r.tasa, **LINEA_GLOBAL)
            ax.set_yticks(ys, t.index)
            ax.set_ylim(-0.75, len(t) - 0.25)   # cada nivel ocupa lo mismo en todos los paneles
            if col in ("month", "day_of_week"):
                ax.invert_yaxis()   # calendario de arriba hacia abajo
            ax.tick_params(axis="y", length=0, labelsize=11)
            ax.set_xlim(0, xmax)
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.set_title(col, fontsize=12.5)
    leyenda(fig, [h_tasa("% de «yes» del nivel"),
                  Patch(label=f"nivel con menos del {pct(100 * UMBRAL_RARA, 0)} de las filas",
                        **RAYADO),
                  h_global(r)])
    return png(fig, "Tasa de «yes» por nivel de cada una de las "
                    f"{len(r.cat)} categóricas; los niveles con menos del "
                    f"{pct(100 * UMBRAL_RARA, 0)} de las filas van rayados.")


def g_cramer(r):
    v = r.v
    fig, ax = figura(3.9)
    ys = np.arange(len(v))
    ax.barh(ys, v.to_numpy(), height=0.6, color=P["azuloscuro"], zorder=2)
    rotular_barras(ax, ys, v.to_numpy(), [num(x, 3) for x in v])
    ax.set_yticks(ys, v.index)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, v.max() * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: num(x, 2)))
    ax.grid(axis="x")
    ax.set_xlabel("V de Cramér con y (con corrección de sesgo; 0 = sin asociación)")
    return png(fig, f"Ranking de V de Cramér: {v.index[0]} {num(v.iloc[0], 3)}, "
                    f"{v.index[1]} {num(v.iloc[1], 3)}; al final, {v.index[-1]} "
                    f"{num(v.iloc[-1], 3)}.")


# -- h. numéricas --------------------------------------------------------------------------
def g_edad(r):
    fig, ax = figura(3.9)
    e = r.edad
    columnas_tasa(ax, r, [f"{i}\n{num(n)}" for i, n in zip(e.index, e["n"])],
                  e["tasa"].to_numpy())
    ax.set_xlabel("Tramo de edad (años) y filas del tramo")
    ax.set_ylabel("% de «yes»")
    leyenda(fig, [h_tasa(), h_global(r)])
    return png(fig, f"Tasa de «yes» por tramo de edad, en forma de U: "
                    f"{pct(e['tasa'].iloc[0], 1)} en {e.index[0]}, "
                    f"{pct(e.loc[r.edad_min, 'tasa'], 1)} en {r.edad_min} y "
                    f"{pct(e['tasa'].iloc[-1], 1)} en {e.index[-1]}.")


def g_campaign(r):
    fig, (a1, a2) = figura(5.0, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    t = r.camp
    etiquetas = [f"{i}{'+' if i == TOPE_CAMPAIGN else ''}" for i in t.index]
    xs = np.arange(len(t))
    a1.bar(xs, t["pct"], width=0.62, color=P["apagado"], zorder=2)
    rotular_columnas(a1, xs, t["pct"], lambda v: pct(v, 1), fontsize=11)
    a1.set_ylim(0, t["pct"].max() * 1.22)
    a1.set_ylabel("% de filas")
    eje_pct(a1.yaxis)
    a1.grid(axis="y")
    columnas_tasa(a2, r, etiquetas, t["tasa"].to_numpy())
    a2.set_ylabel("% de «yes»")
    a2.set_xlabel("campaign: contactos en esta campaña (incluido el último)")
    leyenda(fig, [Patch(facecolor=P["apagado"], label="% de las filas"), h_tasa(), h_global(r)])
    return png(fig, f"Distribución de campaign y tasa de «yes» por número de contactos: "
                    f"{pct(t['tasa'].iloc[0], 1)} con 1 y {pct(t['tasa'].iloc[-1], 1)} con "
                    f"{TOPE_CAMPAIGN} o más.")


def _bordes_discretos(valores):
    u = np.sort(np.unique(valores))
    medios = (u[1:] + u[:-1]) / 2
    paso_ini = (u[1] - u[0]) / 2 if len(u) > 1 else 0.5
    paso_fin = (u[-1] - u[-2]) / 2 if len(u) > 1 else 0.5
    return np.concatenate([[u[0] - paso_ini], medios, [u[-1] + paso_fin]])


def g_densidades(r, df):
    vistas = [v for v in r.num if v not in ("age", "campaign")]
    filas = int(np.ceil(len(vistas) / 2))
    fig, axs = figura(2.15 * filas + 0.5, nrows=filas, ncols=2)
    y = r.y
    for ax, col in zip(axs.flat, vistas):
        serie = df[col]
        titulo = col
        log = False
        if col == "pdays":
            serie = serie[serie != PDAYS_CENTINELA]
            titulo = f"pdays (sin {PDAYS_CENTINELA}: {num(len(serie))} filas)"
        if col == "duration":
            serie = serie.clip(lower=1)
            titulo = "duration (fuga; eje logarítmico)"
            log = True
        if log:
            bordes = np.geomspace(1, serie.max() + 1, 45)
        elif serie.nunique() <= 30:
            bordes = _bordes_discretos(serie.to_numpy())
        else:
            bordes = np.linspace(serie.min(), serie.max(), 41)
        for clase, color in ((0, NO), (1, SI)):
            v = serie[y.loc[serie.index] == clase]
            w = np.full(len(v), 1 / len(v))
            ax.hist(v, bins=bordes, weights=w, histtype="stepfilled", color=color, alpha=0.22,
                    lw=0)
            ax.hist(v, bins=bordes, weights=w, histtype="step", color=color, lw=1.8)
        if log:
            ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: num(v, 0) if abs(v) >= 100 or float(v).is_integer() else num(v, 1)))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(titulo, fontsize=12)
    for ax in axs.flat[len(vistas):]:
        ax.axis("off")
    leyenda(fig, [Patch(facecolor=NO, alpha=0.5, label="no"),
                  Patch(facecolor=SI, alpha=0.5, label="yes"),
                  Line2D([], [], lw=0, label="(altura: proporción de las filas de cada clase)")])
    return png(fig, f"Distribución por clase de {len(vistas)} numéricas; en «yes», la mediana "
                    f"de euribor3m es {num(r.mediana.loc['yes', 'euribor3m'], 2)} y en «no», "
                    f"{num(r.mediana.loc['no', 'euribor3m'], 2)}.")


# -- i. escalas ----------------------------------------------------------------------------
def g_escalas(r):
    sd = r.sd.sort_values()
    fig, ax = figura(4.2)
    ys = np.arange(len(sd))
    for yy, (col, v) in zip(ys, sd.items()):
        if col in EXCLUIDAS:
            ax.barh(yy, v, height=0.6, zorder=2, **RAYADO)
        else:
            ax.barh(yy, v, height=0.6, color=P["azuloscuro"], zorder=2)
    rotular_barras(ax, ys, sd.to_numpy(), [desvio(v) for v in sd])
    ax.set_xscale("log")
    ax.set_xlim(sd.min() / 1.6, sd.max() * 3.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 1 if v < 1 else 0)))
    ax.set_yticks(ys, sd.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_xlabel("Desvío estándar en train (eje logarítmico)")
    leyenda(fig, [Patch(facecolor=P["azuloscuro"], label="numérica del modelo"),
                  Patch(label="fuera del modelo (duration, D-05)", **RAYADO)])
    return png(fig, f"Desvío estándar de cada numérica en escala logarítmica: de "
                    f"{desvio(r.sd[r.sd_min])} ({r.sd_min}) a {desvio(r.sd[r.sd_max])} "
                    f"({r.sd_max}) entre las del modelo.")


# -- j. ranking ----------------------------------------------------------------------------
def g_ranking(r):
    rk = r.ranking.sort_values()
    fig, ax = figura(6.6)
    ys = np.arange(len(rk))
    for yy, (col, v) in zip(ys, rk.items()):
        if col in EXCLUIDAS:
            ax.barh(yy, v - 0.5, left=0.5, height=0.64, zorder=2, **RAYADO)
        else:
            color = P["azuloscuro"] if col in r.num else P["apagado"]
            ax.barh(yy, v - 0.5, left=0.5, height=0.64, color=color, zorder=2)
    rotular_barras(ax, ys, rk.to_numpy(), [num(v, 3) + ("  fuga" if col in EXCLUIDAS else "")
                                           for col, v in rk.items()], fontsize=11)
    ax.set_yticks(ys, rk.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, rk.max() + 0.07)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 2)))
    ax.grid(axis="x")
    ax.set_xlabel("AUC univariado en train (0,5 = azar)")
    leyenda(fig, [Patch(facecolor=P["azuloscuro"], label="numérica: la variable sola"),
                  Patch(facecolor=P["apagado"], label="categórica: tasa de «yes» del nivel"),
                  Patch(label="fuga (fuera del modelo)", **RAYADO)])
    return png(fig, f"Ranking de AUC univariado: {rk.index[-1]} {num(rk.iloc[-1], 3)} (fuga); "
                    f"luego {r.top2[0]} {num(r.ranking[r.top2[0]], 3)} y {r.top2[1]} "
                    f"{num(r.ranking[r.top2[1]], 3)}.")


# ------------------------------------------------------------------------------------------
# Página
# ------------------------------------------------------------------------------------------
CSS = """
@import url("https://fonts.googleapis.com/css2?family=Fira+Mono:wght@400;500&family=Fira+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap");

:root {
  --fondo: #FCFCFB;
  --panel: #F4F3EF;
  --tinta: #0B0B0B;
  --tinta-sec: #52514E;
  --rejilla: #E1E0D9;
  --acento: #104281;
  --figura: #FCFCFB;
  --borde-figura: #E1E0D9;
  --sans: "Fira Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "Fira Mono", ui-monospace, Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --fondo: #151514;
    --panel: #1F1F1D;
    --tinta: #EDECE7;
    --tinta-sec: #B8B6AE;
    --rejilla: #3A3935;
    --acento: #8DB5EE;
    --borde-figura: #3A3935;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --fondo: #151514;
  --panel: #1F1F1D;
  --tinta: #EDECE7;
  --tinta-sec: #B8B6AE;
  --rejilla: #3A3935;
  --acento: #8DB5EE;
  --borde-figura: #3A3935;
  color-scheme: dark;
}

*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--fondo);
  color: var(--tinta);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.55;
  -webkit-text-size-adjust: 100%;
}
.pagina {
  max-width: 960px;
  margin-inline: auto;
  padding-inline: clamp(16px, 4vw, 32px);
  padding-block: 40px 64px;
}
h1, h2, h3, h4 { text-wrap: balance; line-height: 1.2; margin: 0; }
h1 { font-size: 2.1rem; font-weight: 600; letter-spacing: -0.01em; }
h2 {
  font-size: 1.5rem; font-weight: 600;
  margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--rejilla);
}
h3 { font-size: 1.2rem; font-weight: 600; margin-top: 36px; }
h4 {
  font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--tinta-sec); margin-top: 18px;
}
p, li { max-width: 70ch; }
p { margin: 8px 0 0; }
ul { margin: 8px 0 0; padding-left: 1.2em; }
li + li { margin-top: 4px; }
a { color: var(--acento); text-underline-offset: 2px; }
code {
  font-family: var(--mono); font-size: 0.88em;
  background: var(--panel); border: 1px solid var(--rejilla); border-radius: 3px;
  padding: 0 0.28em; white-space: nowrap;
}
.tabla code, .cifra code { white-space: normal; overflow-wrap: anywhere; }
.antetitulo { font-size: 0.9rem; color: var(--tinta-sec); margin: 0 0 6px; }
.bajada { font-size: 1.05rem; color: var(--tinta-sec); margin-top: 10px; }
.cifras {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px; margin-top: 22px;
}
.cifra { background: var(--panel); border: 1px solid var(--rejilla); border-radius: 6px; padding: 12px 14px; }
.cifra .etiqueta { font-size: 0.85rem; color: var(--tinta-sec); }
.cifra .valor { font-family: var(--mono); font-size: 1.45rem; font-weight: 500; line-height: 1.25; margin-top: 2px; }
.cifra .nota { font-size: 0.8rem; color: var(--tinta-sec); margin-top: 2px; }
nav.indice { margin-top: 24px; font-size: 0.93rem; }
nav.indice ul { list-style: none; padding: 0; margin: 6px 0 0; display: flex; flex-wrap: wrap; gap: 4px 16px; }
nav.indice li { margin: 0; max-width: none; }
nav.indice a { text-decoration: none; }
nav.indice a:hover { text-decoration: underline; }
.claves li { margin-top: 8px; }
.cifra-texto { font-family: var(--mono); font-weight: 500; }
figure.grafico {
  margin: 14px 0 0; padding: 10px; background: var(--figura);
  border: 1px solid var(--borde-figura); border-radius: 6px;
}
figure.grafico img { display: block; width: 100%; max-width: 100%; height: auto; }
.tabla { overflow-x: auto; margin-top: 16px; border: 1px solid var(--rejilla); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; min-width: 680px; font-size: 0.93rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; vertical-align: top; padding: 10px 12px; border-bottom: 1px solid var(--rejilla); }
thead th { background: var(--panel); font-weight: 600; }
tbody tr:last-child td { border-bottom: 0; }
td:first-child { width: 36%; }
.estado {
  display: inline-block; white-space: nowrap; font-size: 0.82rem; line-height: 1.4;
  padding: 1px 8px; border-radius: 999px; border: 1px solid var(--rejilla);
}
.estado.hecha { color: var(--acento); border-color: var(--acento); }
.estado.propuesta { color: var(--tinta-sec); background: var(--panel); }
.estado.descartada { color: var(--tinta-sec); border-color: var(--tinta-sec); border-style: dotted; }
.estado.consultar { color: var(--tinta); border-color: var(--tinta-sec); border-style: dashed; }
.supuestos td:first-child { width: 12%; font-weight: 600; }
footer { margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--rejilla); font-size: 0.9rem; color: var(--tinta-sec); }
"""


def bloque(g, titulo, se_ve, hacer):
    items = "\n".join(f"      <li>{h}</li>" for h in hacer)
    return f"""
  <article class="bloque">
    <h3>{titulo}</h3>
    <figure class="grafico"><img src="data:image/png;base64,{g.b64}" width="{g.ancho}" height="{g.alto}" alt="{escape(g.alt)}"></figure>
    <h4>Qué se ve</h4>
    <p>{se_ve}</p>
    <h4>Qué hacer</h4>
    <ul>
{items}
    </ul>
  </article>"""


def seccion(id_, titulo, contenido):
    return f'\n<section id="{id_}">\n  <h2>{titulo}</h2>{contenido}\n</section>'


def estado(texto):
    clase = ("hecha" if texto.startswith(("Hecha", "Decidida")) else
             "descartada" if texto.startswith(("Medida", "Descartada")) else
             "consultar" if texto.startswith("Consultar") else "propuesta")
    return f'<span class="estado {clase}">{texto}</span>'


def armar(r, names, G):
    t = r.tasa
    b = r.bloques
    rg = r.rotulo_global
    po, pv = r.pout, r.prev
    trio_txt = lista([c(v) for v in r.trio])
    rmin, rmax = num(min(r.r_trio), 2), num(max(r.r_trio), 2)
    a1, a2 = r.top2
    n_raras = len(r.raras)
    raras_txt = lista([f"{c(col)} {escape(str(nivel))}" for col, nivel, *_ in r.raras])
    altos_txt = lista([c(m) for m in r.meses_altos])
    anio_ini, anio_fin = r.anios[0], r.anios[-1]
    edad0, edadN = r.edad.index[0], r.edad.index[-1]
    fila_mes = r.mes.loc[r.mes_max]
    success = po.loc["success", "tasa"]
    euro_med = r.mediana["euribor3m"]
    fuera_a, fuera_b, fuera_r = r.par_fuera
    otra_medio = [v for v in MACRO if v not in r.trio]
    medio_1 = otra_medio[0] if otra_medio else None
    medio_2 = otra_medio[1] if len(otra_medio) > 1 else None
    un_punto = "un punto" if DESVIO_CHICO == 1 else f"{num(DESVIO_CHICO)} puntos"
    if r.max_raras_col == 1:
        frase_minfreq = (f"Ninguna columna tiene más de un nivel raro, así que "
                         f"{c('min_frequency')}, por sí solo, no los funde con nada.")
        atencion_minfreq = (
            f"Atención con {c('OneHotEncoder(min_frequency=…)')}: junta los niveles raros de "
            f"una misma columna, y aquí ninguna tiene más de uno. Por sí solo, sólo los "
            f"renombra; para que el agrupamiento tenga efecto hay que decidir con qué nivel se "
            f"funde cada uno.")
        como_agrupar = "fundiendo cada uno con otro nivel de su columna"
    else:
        frase_minfreq = (f"Hay columnas con hasta {palabra(r.max_raras_col)} niveles raros: "
                         f"{c('min_frequency')} los junta en uno solo.")
        atencion_minfreq = (f"{c('OneHotEncoder(min_frequency=…)')} junta los niveles raros de "
                            f"cada columna en una sola categoría.")
        como_agrupar = f"(p. ej. {c('OneHotEncoder')} con {c('min_frequency')})"

    # ---- claves ------------------------------------------------------------------------
    def cifra(x):
        return f'<span class="cifra-texto">{x}</span>'

    claves = [
        f"<strong>Desbalance.</strong> {cifra(pct(t, 2))} de «yes»: decir siempre «no» acierta "
        f"el {cifra(pct(r.acc_base, 2))}.",
        f"<strong>{c('duration')} es fuga.</strong> Sola logra un AUC de "
        f"{cifra(num(r.auc_dur, 3))}, y ninguna de las {cifra(num(r.n_corta))} llamadas de menos "
        f"de {DUR_CORTA} s terminó en «yes».",
        f"<strong>El archivo es una serie de tiempo.</strong> Ordenado por fecha, su tasa de "
        f"«yes» va del {cifra(pct(b['tasa'].iloc[0], 1))} en el primer bloque de "
        f"{num(BLOQUE)} filas al {cifra(pct(b['tasa'].iloc[-1], 1))} en el último.",
        f"<strong>El contexto económico domina.</strong> {c(a1)} y {c(a2)} son lo que más "
        f"separa (AUC {cifra(num(r.ranking[a1], 3))} y {cifra(num(r.ranking[a2], 3))}), y "
        f"{trio_txt} son casi colineales (r entre {cifra(rmin)} y {cifra(rmax)}).",
        f"<strong>{c('pdays')} = {PDAYS_CENTINELA} no es «nunca contactado».</strong> Aparece en "
        f"el {cifra(pct(r.pct999, 1))} de las filas, y {cifra(num(r.n_contra))} de ellas "
        f"tienen {c('previous')} ≥ 1.",
        f"<strong>El historial pesa.</strong> Con {c('poutcome')} = success la tasa es "
        f"{cifra(pct(success, 1))}, {cifra(num(success / t, 1))} veces la global.",
        f"<strong>«unknown» informa.</strong> {c('default')} es «unknown» en el "
        f"{cifra(pct(r.def_pct, 1))} de las filas, con {cifra(pct(r.def_tu, 1))} de «yes» "
        f"frente a {cifra(pct(r.def_tc, 1))} cuando se conoce.",
        f"<strong>Escalas muy distintas.</strong> Los desvíos estándar de las numéricas del "
        f"modelo difieren {cifra(num(r.sd_ratio, 0))} veces: KNN y SVM necesitan escalado.",
    ]

    # ---- qué hacer (tabla) -------------------------------------------------------------
    # Las propuestas son las del EDA, anteriores a las ablaciones de la ola 1; el estado dice qué
    # se decidió después y en qué fila de DECISIONES.md está el porqué.
    decisiones = [
        (f"{c('duration')} fuera del modelo; a lo sumo, un modelo techo con ella.",
         f'<a href="#fuga">Fuga de datos</a>: AUC {num(r.auc_dur, 3)} por sí sola; 0 «yes» en '
         f"las {num(r.n_corta)} llamadas de menos de {DUR_CORTA} s y "
         f"{pct(r.dur_dec['tasa'].iloc[-1], 2)} en el decil más largo.",
         "Hecha (D-05)"),
        ("Partición aleatoria estratificada y validación cruzada barajada.",
         f'<a href="#tiempo">El tiempo</a>: la tasa por bloque del archivo va del '
         f"{pct(b['tasa'].iloc[0], 1)} al {pct(b['tasa'].iloc[-1], 1)}; sin barajar, cada "
         f"<i>fold</i> caería en una época.",
         "Hecha (D-02, D-03, D-06)"),
        ("«unknown» se mantiene como categoría propia, sin imputar.",
         f'<a href="#faltantes">Faltantes</a>: en {c("default")}, {pct(r.def_tu, 1)} de «yes» '
         f"con «unknown» frente a {pct(r.def_tc, 1)} con el valor conocido; en las otras "
         f"{palabra(len(r.dif_resto))} columnas el «unknown» también tiene más «yes».",
         "Hecha (D-08)"),
        (f"{c('default')}: casi no tiene «yes»; tratarla como indicadora «unknown» / «no».",
         f'<a href="#faltantes">Faltantes</a>: el nivel «yes» tiene {num(r.n_default_yes)} filas '
         f"de {num(r.n)}; «unknown», el {pct(r.def_pct, 1)}.",
         "Hecha (D-09)"),
        (f"{c('pdays')} = {PDAYS_CENTINELA} no significa «nunca contactado»: usar "
         f"{c('previous')} &gt; 0 como indicadora de contacto previo y no usar el "
         f"{PDAYS_CENTINELA} como número.",
         f'<a href="#previas">Campañas previas</a>: {num(r.n_contra)} filas con '
         f"{PDAYS_CENTINELA} y {c('previous')} ≥ 1, con {pct(r.t_contra, 1)} de «yes» frente a "
         f"{pct(r.t_sin_previo, 1)} sin contactos previos.",
         "Descartada (D-10)"),
        (f"Niveles con menos del {pct(100 * UMBRAL_RARA, 0)} de las filas: agruparlos dentro del "
         f"pipeline {como_agrupar}, salvo los de tasa distintiva como {c('dec')}.",
         f'<a href="#categorias">Categorías</a>: {palabra(n_raras)} niveles bajo el '
         f"{pct(100 * UMBRAL_RARA, 0)}; {c('dec')} tiene {pct(r.dec['tasa'], 1)} de «yes» con "
         f"{num(r.dec['n'])} filas. {frase_minfreq}",
         "Hecha en parte (D-11)"),
        (f"Escalado ({c('StandardScaler')}) dentro del pipeline para KNN y SVM; RF no lo "
         f"necesita.",
         f'<a href="#escalas">Escalas</a>: los desvíos difieren {num(r.sd_ratio, 0)} veces; '
         f"sin escalar, {c(r.var_share.index[0])} aporta el {pct(r.var_share.iloc[0], 1)} de la "
         f"varianza.",
         "Hecha (D-12)"),
        (f"{c('campaign')} tiene cola larga: probar {c('log1p')} para KNN y SVM.",
         f'<a href="#numericas">Numéricas</a>: asimetría {num(r.asim["campaign"], 2)}; el '
         f"{pct(r.camp_pocos, 1)} tiene {POCOS_CONTACTOS} contactos o menos y el máximo es "
         f"{num(r.camp_max)}.",
         "Medida (D-13)"),
        ("Bloque macro casi colineal: medir con y sin reducirlo a una o dos variables, sobre "
         "todo para Naive Bayes; y medir el modelo sin el bloque para cuantificar cuánto "
         "«aprende la época».",
         f'<a href="#macro">Contexto económico</a>: r entre {rmin} y {rmax} en {trio_txt}; '
         f"{c(a1)} y {c(a2)} son lo que más separa (AUC {num(r.ranking[a1], 3)} y "
         f"{num(r.ranking[a2], 3)}).",
         "Medida (D-14, D-25)"),
        (f"{c('month')} codifica en parte la época; {c('day_of_week')} separa poco: candidata "
         f"a descartar tras medir.",
         f'<a href="#tiempo">El tiempo</a>: los meses con más del {pct(TASA_ALTA, 0)} de «yes» '
         f"tienen euribor3m medio {num(r.eur_altos, 2)} (resto: {num(r.eur_resto, 2)}). "
         f'<a href="#categorias">Categorías</a>: V de {c("day_of_week")} = '
         f"{num(r.v['day_of_week'], 3)}.",
         "Medida (D-15)"),
        ("Edad con forma de U: discretizarla en tramos para Naive Bayes; RF y SVM con RBF la "
         "capturan.",
         f'<a href="#numericas">Numéricas</a>: {pct(r.edad["tasa"].iloc[0], 1)} en {edad0}, '
         f"{pct(r.edad.loc[r.edad_min, 'tasa'], 1)} en {r.edad_min} y "
         f"{pct(r.edad['tasa'].iloc[-1], 1)} en {edadN}; AUC de {c('age')} sola: "
         f"{num(r.auc_age, 3)}.",
         "Medida (D-16)"),
        ("Variante de Naive Bayes para datos mixtos: GaussianNB o CategoricalNB con las "
         "numéricas discretizadas.",
         f'<a href="#numericas">Numéricas</a>: ninguna numérica tiene forma de campana por '
         f"clase; {c('campaign')} tiene asimetría {num(r.asim['campaign'], 2)} y {c('pdays')} "
         f"vale {PDAYS_CENTINELA} en el {pct(r.pct999, 1)} de las filas.",
         "Decidida (D-18)"),
        ("Tratar o no el desbalance (pesos de clase).",
         f'<a href="#objetivo">El objetivo</a>: {num(r.ratio, 1)} «no» por cada «yes»; decir '
         f"siempre «no» acierta el {pct(r.acc_base, 1)}.",
         "Decidida (D-21)"),
    ]
    filas_dec = "\n".join(
        f"      <tr><td>{d}</td><td>{e}</td><td>{estado(s)}</td></tr>" for d, e, s in decisiones)

    # ---- bloques de gráficos -----------------------------------------------------------
    s_obj = bloque(
        G["balance"],
        f"Decir siempre «no» ya acierta el {pct(r.acc_base, 1)}",
        f"En train hay {num(r.n_no)} filas «no» y {num(r.n_yes)} «yes» ({pct(t, 2)}): "
        f"{num(r.ratio, 1)} «no» por cada «yes». Un clasificador que responde siempre «no» "
        f"acierta el {pct(r.acc_base, 2)} de las veces sin encontrar a un solo cliente que "
        f"contrate.",
        ["Elegir métricas que no premien decir siempre «no»: precisión, <i>recall</i> o F1 de la "
         "clase «yes», o el área bajo la curva ROC. La <i>accuracy</i> sola no sirve. Se "
         "eligieron el AUC, para comparar modelos, y el <i>recall</i> llamando al "
         f"{pct(100 * PRESUPUESTO, 0)} de la lista, para medir el uso real (D-19, D-20).",
         "Mantener la estratificación en la partición y en cada <i>fold</i> (D-02, D-06): así "
         f"todos conservan el {pct(t, 1)} de «yes».",
         "Compensar o no el desbalance con pesos de clase era una consulta a la cátedra. Se "
         "resolvió con los datos (D-21): no se remuestrea, y los pesos de clase se miden como un "
         "hiperparámetro más; entraron en la SVM y no en RF (D-22)."])

    dd = r.dur_dec
    s_fuga = bloque(
        G["duration"],
        f"{c('duration')} separa casi sola (AUC {num(r.auc_dur, 3)}), pero sólo se conoce al "
        f"terminar la llamada",
        f"La tasa de «yes» crece decil a decil: {pct(dd['tasa'].iloc[0], 1)} en las llamadas más "
        f"cortas (hasta {num(dd['hi'].iloc[0])} s) y {pct(dd['tasa'].iloc[-1], 2)} en las más "
        f"largas (desde {num(dd['lo'].iloc[-1])} s). Ninguna de las {num(r.n_corta)} llamadas de "
        f"menos de {DUR_CORTA} s terminó en «yes», y el «yes» más corto duró "
        f"{num(r.min_dur_yes)} s. Sola, {c('duration')} tiene un AUC de {num(r.auc_dur, 3)} en "
        f"train. {c('bank-additional-names.txt')} (atributo {names.atributo['duration']}) "
        f"advierte que la duración no se conoce antes de llamar y que, al terminar la llamada, el "
        f"resultado ya se conoce.",
        [f"Dejar {c('duration')} fuera de los cuatro modelos (D-05).",
         f"A lo sumo, entrenar un modelo con {c('duration')} y reportarlo como techo de "
         f"referencia, nunca como resultado. Así se midió, como la ablación A1 (D-05)."])

    ta = r.tasa_anio
    s_tiempo = bloque(
        G["bloques"],
        f"La tasa de «yes» pasa del {pct(b['tasa'].iloc[0], 1)} al {pct(b['tasa'].iloc[-1], 1)} "
        f"a lo largo del archivo",
        f"El CSV original está ordenado por fecha, de {names.periodo} "
        f"({c('bank-additional-names.txt')}, §{names.seccion}); cada punto es un bloque de "
        f"{num(BLOQUE)} filas consecutivas de ese archivo, contando sólo las que quedaron en "
        f"train. La tasa sube del {pct(b['tasa'].iloc[0], 1)} al {pct(b['tasa'].iloc[-1], 1)} "
        f"mientras {c('euribor3m')} baja de {num(b['eur'].iloc[0], 2)} a "
        f"{num(b['eur'].min(), 2)}. El año no viene en los datos; inferido del orden de los "
        f"meses (cada vez que el mes retrocede empieza uno nuevo), la tasa es "
        + lista([f"{pct(ta[a], 1)} en {a}" for a in r.anios]) + ".",
        ["La partición aleatoria (D-03) deja la misma mezcla de épocas en train y en test; "
         "como train sigue ordenado por fecha, la validación cruzada tiene que barajar (D-06). "
         "Las dos cosas ya están hechas.",
         "El modelo puede aprender «cuándo» se llamó además de «a quién»: comparar el modelo "
         "completo con uno sin el bloque macro. Se comparó en las ablaciones (A7, D-14) y, "
         "además, validando hacia adelante en el tiempo: el modelo aprende la época, y sacar el "
         "bloque no lo arregla (D-25).",
         f"Declararlo como limitación en las conclusiones: la estimación vale para clientes "
         f"parecidos a la mezcla de {anio_ini}–{anio_fin}. Es la primera limitación del punto 5 "
         f"(D-03, D-25)."])

    mt = r.mes.loc[r.meses_top]
    s_meses = bloque(
        G["meses"],
        f"{c(r.mes_max)} concentra el {pct(fila_mes['pct'], 1)} de las llamadas y tiene la tasa "
        f"más baja: {pct(fila_mes['tasa'], 1)}",
        f"Arriba, las filas de cada mes; abajo, su tasa de «yes». Los {palabra(len(r.meses_top))} "
        f"meses con más "
        f"llamadas ({lista([c(m) for m in r.meses_top])}) tienen tasas de entre "
        f"{pct(mt['tasa'].min(), 1)} y {pct(mt['tasa'].max(), 1)}. Los "
        f"{palabra(len(r.meses_altos))} meses con menos del {pct(MES_CHICO, 0)} de las filas cada uno "
        f"({altos_txt}) superan el {pct(TASA_ALTA, 0)}. Esos meses caen casi siempre después de "
        f"{anio_ini} (año inferido: {pct(r.altos_ult, 1)} de sus filas) y tienen un "
        f"{c('euribor3m')} medio de {num(r.eur_altos, 2)}, frente a {num(r.eur_resto, 2)} en el "
        f"resto.",
        [f"{c('month')} mezcla estacionalidad con época: incluirla en la comparación con y sin "
         f"contexto económico. Se midió (A8): sin ella ni el bloque macro, el AUC cae en los "
         f"cuatro modelos, y {c('month')} se queda (D-15).",
         f"Codificarla con <i>one-hot</i>, no como número de mes: la relación no es monótona. "
         f"Así quedó (D-15)."])

    lineas_medio = []
    if medio_1:
        rr = r.r_medio[medio_1]
        lineas_medio.append(f"{c(medio_1)} las acompaña a medias (|r| entre {num(min(rr), 2)} y "
                            f"{num(max(rr), 2)})")
    if medio_2:
        lineas_medio.append(f"{c(medio_2)}, poco (|r| ≤ {num(max(r.r_medio[medio_2]), 2)})")
    s_corr = bloque(
        G["correlacion"],
        f"{trio_txt} se mueven juntas: r entre {rmin} y {rmax}",
        f"Dentro del recuadro de las {palabra(len(MACRO))} variables macro, "
        f"{palabra(len(r.trio))} están casi duplicadas: "
        + lista([f"{c(a)}–{c(bb)} {num(v, 2)}" for a, bb, v in r.pares_colineales]) + ". "
        + (lista(lineas_medio) + ". " if lineas_medio else "")
        + f"Fuera del bloque, el par más correlacionado es {c(fuera_a)}–{c(fuera_b)} "
        f"({num(fuera_r, 2)}): las dos describen la campaña anterior.",
        ["Para Naive Bayes, el trío viola la independencia condicional: la misma señal entra "
         "tres veces. Medir con el bloque completo y reducido a una o dos variables. Se midió "
         "(A6): reducirlo empeora Naive Bayes y KNN, y el bloque entra completo (D-14).",
         "Para KNN y SVM, tres columnas casi iguales triplican el peso de esa señal en la "
         "distancia: la misma comparación sirve, y es la misma medición (D-14).",
         "RF tolera la redundancia, pero reparte la importancia entre las tres: no leerlas por "
         "separado."])

    ed = r.eur_dec
    s_eur = bloque(
        G["euribor"],
        f"Con {c('euribor3m')} bajo la tasa llega al {pct(ed['tasa'].iloc[0], 1)}; en la mitad "
        f"alta no pasa del {pct(r.eur_alta['tasa'].max(), 1)}",
        f"Cada barra es un decil de {c('euribor3m')} en train, de menor a mayor. En el decil más "
        f"bajo ({num(ed['lo'].iloc[0], 3)} a {num(ed['hi'].iloc[0], 3)}) la tasa de «yes» es "
        f"{pct(ed['tasa'].iloc[0], 1)}. En los {palabra(len(r.eur_alta))} deciles más altos (desde "
        f"{num(r.eur_alta['lo'].iloc[0], 3)}) está entre {pct(r.eur_alta['tasa'].min(), 1)} y "
        f"{pct(r.eur_alta['tasa'].max(), 1)}. El salto coincide con el cambio de época del "
        f"gráfico de bloques: el euribor también funciona como reloj.",
        ["Medir el modelo sin el bloque macro para separar cuánto de la señal es «la época» y "
         "cuánto es el cliente. Se midió: con los <i>folds</i> barajados, sacarlo cuesta AUC "
         "(A7, D-14), y validando hacia adelante en el tiempo el modelo cae también sin él "
         "(D-25).",
         "RF y SVM con kernel RBF representan el escalón sin ayuda; para Naive Bayes, "
         "discretizar lo representa mejor que una normal por clase. El Naive Bayes del TP "
         "discretiza (D-18)."])

    s_prev = bloque(
        G["previas"],
        f"Un éxito en la campaña anterior lleva la tasa al {pct(success, 1)}",
        f"Con {c('poutcome')} = success la tasa de «yes» es {pct(success, 1)}, "
        f"{num(success / t, 1)} veces la global; con failure, {pct(po.loc['failure', 'tasa'], 1)}; "
        f"sin campaña previa (nonexistent), {pct(po.loc['nonexistent', 'tasa'], 1)}. Con "
        f"{c('previous')} pasa lo mismo: "
        + lista([f"{pct(v, 1)} con {i}{' o más' if i == TOPE_PREVIOUS else ''}"
                 for i, v in pv["tasa"].items()])
        + f" contactos previos. Pero el {pct(r.pct_prev0, 1)} de las filas no tiene contactos "
        f"previos: la señal es fuerte y alcanza a pocos clientes.",
        [f"Mantener {c('poutcome')} y {c('previous')}: separan mucho en la minoría con "
         f"historial. Se mantienen.",
         f"Representar el contacto previo sin el {PDAYS_CENTINELA} de {c('pdays')}: ver el "
         f"gráfico siguiente. Se descartó después (D-10)."])

    s_pdays = bloque(
        G["pdays"],
        f"{c('pdays')} = {PDAYS_CENTINELA} no significa «nunca contactado»: "
        f"{num(r.n_contra)} de esas filas tienen contactos previos",
        f"{c('bank-additional-names.txt')} (atributo {names.atributo['pdays']}) dice "
        f"«{escape(names.cita_pdays)}»: que el cliente no fue contactado antes. En train, "
        f"{num(r.n999)} filas ({pct(r.pct999, 1)}) tienen {c('pdays')} = {PDAYS_CENTINELA} y "
        f"{num(r.n_contra)} de ellas tienen {c('previous')} ≥ 1, todas con {c('poutcome')} = "
        f"{lista(r.pout_contra.index)}. Su tasa de «yes» es {pct(r.t_contra, 1)}, frente a "
        f"{pct(r.t_sin_previo, 1)} en las que realmente no tienen contactos previos. El "
        f"{PDAYS_CENTINELA} indica que no hay días registrados, no una distancia de "
        f"{PDAYS_CENTINELA} días.",
        [f"Marcar el contacto previo con {c('previous')} &gt; 0, no con {c('pdays')} ≠ "
         f"{PDAYS_CENTINELA}.",
         f"No usar el {PDAYS_CENTINELA} como número: en KNN y SVM dejaría a un cliente sin fecha "
         f"a {PDAYS_CENTINELA} «días» de uno contactado hace {num(r.pdays_mediana)} días (la "
         f"mediana). Si {c('pdays')} se conserva, que sea sólo donde hay días registrados, junto "
         f"con la indicadora.",
         f"Estas dos propuestas se descartaron después (D-10): {c('pdays')} se queda con el "
         f"{PDAYS_CENTINELA}. Sacarla empeora Naive Bayes y KNN; escalado, el centinela separa a "
         f"los contactados del resto, y para los árboles es un corte. La indicadora no se agrega: "
         f"{c('previous')} &gt; 0 ya está en {c('poutcome')} (nonexistent equivale a "
         f"{c('previous')} = 0)."])

    s_unk = bloque(
        G["unknown"],
        f"{c('default')} es «unknown» en el {pct(r.def_pct, 1)} de las filas, y ahí la tasa de "
        f"«yes» cae al {pct(r.def_tu, 1)}",
        f"No hay NaN explícitos ({r.nan} en todo train): los faltantes vienen como la categoría "
        f"«unknown», en {palabra(len(r.unk))} columnas. {c('default')} es la más afectada: con "
        f"«unknown» la tasa es {pct(r.def_tu, 1)}, con el valor conocido {pct(r.def_tc, 1)}. En "
        f"las otras {palabra(len(r.dif_resto))}, el «unknown» tiene entre "
        f"{num(r.dif_resto.min(), 1)} y {num(r.dif_resto.max(), 1)} puntos más de «yes» que el "
        f"valor conocido. Además, "
        + (f"{c('housing')} y {c('loan')} son «unknown» exactamente en las mismas "
           f"{num(r.n_hl)} filas, y " if r.hl_iguales else "")
        + f"{c('default')} casi no tiene el nivel «yes»: {num(r.n_default_yes)} filas de "
        f"{num(r.n)}, {'ninguna' if r.yes_default_yes == 0 else num(r.yes_default_yes)} con "
        f"y = «yes».",
        ["Mantener «unknown» como categoría propia, sin imputar: el faltante informa. Así quedó "
         "(D-08): imputar por la moda, medido como la ablación A11, es neutro.",
         f"Tratar {c('default')} como indicadora «unknown» / «no»: el nivel «yes» no basta "
         f"para estimar nada. Adoptada (D-09).",
         f"Para Naive Bayes, el «unknown» de {c('housing')} y {c('loan')} cuenta dos veces el "
         f"mismo hecho."])

    ilit = [x for x in r.raras if x[0] == "education"]
    dyes = [x for x in r.raras if x[0] == "default"]
    # D-11, tal como quedó en el pipeline: las fusiones de src/preproceso.py y los niveles raros
    # que no se funden (default «yes» lo resuelve D-09 y dec, su tasa)
    fusiones_txt = lista([f"{c(nivel)} con {c(destino)} en {c(col)}"
                          for col, fusion in FUSION_RARAS.items()
                          for nivel, destino in fusion.items()])
    sin_fundir = [f"{c(col)} {escape(str(nivel))}" for col, nivel, *_ in r.raras
                  if nivel not in FUSION_RARAS.get(col, {}) and col not in ("default", "month")]
    ejemplos_pocas = lista([f"{num(n)} en {escape(col)} = {escape(str(nivel))}"
                            for col, nivel, n, *_ in ilit + dyes])
    altos_niv = lista([f"{c(col)} = {escape(str(nivel))} ({pct(tasa, 1)})"
                       for col, nivel, tasa in r.niveles_altos if col != "month"])
    s_cat = bloque(
        G["categorias"],
        f"Algunos niveles superan el {pct(TASA_ALTA, 0)} de «yes»; {c('housing')}, {c('loan')} y "
        f"{c('day_of_week')} no se alejan ni {un_punto} de la tasa global",
        f"Cada panel muestra la tasa de «yes» de cada nivel; la línea punteada es la {rg}. "
        f"Superan el {pct(TASA_ALTA, 0)} {palabra(len(r.meses_altos))} meses ({altos_txt}) y "
        f"{altos_niv}. "
        f"En el otro extremo, {c('housing')}, {c('loan')} y "
        f"{c('day_of_week')} quedan a menos de {un_punto} de la tasa global en "
        f"todos sus niveles. Los {palabra(n_raras)} niveles con menos del "
        f"{pct(100 * UMBRAL_RARA, 0)} de "
        f"las filas van rayados ({raras_txt}): sus tasas se apoyan en pocas filas "
        f"({ejemplos_pocas}).",
        [f"Agrupar los niveles con menos del {pct(100 * UMBRAL_RARA, 0)} dentro del pipeline, "
         f"para que cada <i>fold</i> aprenda la agrupación con sus propios datos; {c('dec')} "
         f"({pct(r.dec['tasa'], 1)} de «yes») queda aparte por su tasa distintiva. Se hizo en "
         f"parte (D-11), con una regla fija que no aprende nada de cada <i>fold</i>: "
         f"{fusiones_txt}. El nivel «yes» de {c('default')} lo resuelve la indicadora de D-09, "
         f"{c('dec')} no se toca"
         + (f" y {lista(sin_fundir)} {'queda' if len(sin_fundir) == 1 else 'quedan'} como "
            f"{'está' if len(sin_fundir) == 1 else 'están'}" if sin_fundir else "") + ".",
         atencion_minfreq])

    v = r.v
    s_v = bloque(
        G["cramer"],
        f"{c(v.index[0])} (V = {num(v.iloc[0], 3)}) y {c(v.index[1])} ({num(v.iloc[1], 3)}) son "
        f"las categóricas más asociadas con y; {c(v.index[-1])} ({num(v.iloc[-1], 3)}), la que "
        f"menos",
        f"La V de Cramér, con corrección de sesgo, mide la asociación de cada categórica con y, "
        f"de 0 (ninguna) a 1 (total). Después de las dos primeras vienen {c(v.index[2])} "
        f"({num(v.iloc[2], 3)}) y {c(v.index[3])} ({num(v.iloc[3], 3)}). "
        + lista([f"{c(p)} ({num(v[p], 3)})" for p in v[PLANAS].sort_values(ascending=False).index])
        + " casi no se asocian.",
        [f"{c('day_of_week')} es candidata a descartar: medir el modelo con y sin ella antes de "
         f"decidir. Se midió (A9): sin ella la SVM empeora, y {c('day_of_week')} se queda "
         f"(D-15).",
         f"Lo mismo vale para {c('housing')} y {c('loan')}. Para ellas no hubo ablación, y "
         f"quedan en el modelo."])

    e = r.edad
    s_edad = bloque(
        G["edad"],
        f"La edad tiene forma de U: {pct(e['tasa'].iloc[0], 1)} en {edad0}, "
        f"{pct(e.loc[r.edad_min, 'tasa'], 1)} en {r.edad_min} y {pct(e['tasa'].iloc[-1], 1)} en "
        f"{edadN}",
        f"Cada barra es un tramo de edad; debajo, cuántas filas tiene. La tasa baja desde los "
        f"más jóvenes hasta un mínimo de {pct(e.loc[r.edad_min, 'tasa'], 1)} en {r.edad_min} y "
        f"vuelve a subir hasta {pct(e['tasa'].iloc[-1], 1)} en {edadN}. Los extremos tienen "
        f"pocas filas: {num(e['n'].iloc[0])} en {edad0} y {num(e['n'].iloc[-1])} en {edadN}. "
        f"Como la relación no es monótona, el AUC de {c('age')} sola es "
        f"{num(r.auc_age, 3)}, casi azar.",
        ["Para Naive Bayes, discretizar la edad en tramos como estos: una normal por clase no "
         "representa una U. Se midió en el Naive Bayes gaussiano (A10) y no se adoptó: es neutro, "
         "y la edad entra en años (D-16). El Naive Bayes del TP, el categórico, ya discretiza "
         "todas las numéricas (D-18).",
         "RF y SVM con kernel RBF capturan la U sin transformar la variable; KNN también, con la "
         "edad escalada. La SVM elegida terminó lineal (D-22)."])

    cp = r.camp
    s_camp = bloque(
        G["campaign"],
        f"Más contactos en la campaña, menos «yes»: del {pct(cp['tasa'].iloc[0], 1)} con uno al "
        f"{pct(cp['tasa'].iloc[-1], 1)} con {TOPE_CAMPAIGN} o más",
        f"Arriba, qué parte de las filas tiene cada número de contactos; abajo, la tasa de «yes». "
        f"El {pct(r.camp_pocos, 1)} de las filas tiene {POCOS_CONTACTOS} contactos o menos, pero "
        f"la cola llega a {num(r.camp_max)}: la asimetría es {num(r.asim['campaign'], 2)}. La "
        f"tasa tiende a bajar a medida que aumentan los contactos.",
        [f"Para KNN y SVM, probar {c('log1p(campaign)')}: acorta la cola sin cambiar el orden. "
         f"Se probó (A5) y no se adoptó: el efecto es nulo, y {c('campaign')} queda sin "
         f"transformar (D-13).",
         "RF no lo necesita: sus cortes no dependen de la escala."])

    s_dens = bloque(
        G["densidades"],
        f"En «yes», la mediana de {c('euribor3m')} es {num(euro_med['yes'], 2)}; en «no», "
        f"{num(euro_med['no'], 2)}",
        f"Cada panel compara la distribución de una numérica en «no» (azul) y en «yes» "
        f"(naranja), como proporción de las filas de cada clase. Las macro separan: «yes» se "
        f"concentra en los valores bajos de {c('euribor3m')}, {c('nr.employed')} y "
        f"{c('emp.var.rate')}. Ninguna tiene forma de campana: las macro toman pocos valores "
        f"({c('nr.employed')} tiene {num(r.unicos['nr.employed'])} distintos), {c('previous')} y "
        f"{c('pdays')} son discretas y con cola, y {c('duration')} sólo se lee bien en escala "
        f"logarítmica.",
        ["GaussianNB supone una normal por clase para cada numérica, y aquí no se cumple. Por "
         "eso el Naive Bayes del TP es el categórico, con las numéricas discretizadas dentro del "
         "<i>pipeline</i>; el gaussiano queda como comparación (D-18).",
         f"{c('duration')} aparece sólo como referencia: queda fuera del modelo (D-05)."])

    otras_var = len(r.num_modelo) - 2
    s_esc = bloque(
        G["escalas"],
        f"Los desvíos estándar difieren {num(r.sd_ratio, 0)} veces: sin escalar, "
        f"{c(r.var_share.index[0])} sola aporta el {pct(r.var_share.iloc[0], 1)} de la varianza",
        f"El eje es logarítmico. Entre las {palabra(len(r.num_modelo))} numéricas que entran al "
        f"modelo, "
        f"el desvío estándar va de {desvio(r.sd[r.sd_min])} ({c(r.sd_min)}) a "
        f"{desvio(r.sd[r.sd_max])} ({c(r.sd_max)}). En una distancia euclídea sin escalar, cada "
        f"variable pesa según su varianza: {c(r.var_share.index[0])} y "
        f"{c(r.var_share.index[1])} suman el {pct(r.var_top2, 1)} y las otras {palabra(otras_var)}, el "
        f"resto. {c('duration')}, fuera del modelo, tendría el desvío más grande "
        f"({desvio(r.sd['duration'])}).",
        [f"{c('StandardScaler')} dentro del pipeline para KNN y SVM, ajustado en cada "
         f"<i>fold</i>. Así quedó (D-12). Sin escalar, KNN sube, pero por la razón equivocada: "
         f"la distancia pasa a medir casi sólo {c(r.var_share.index[0])} y "
         f"{c(r.var_share.index[1])} (A12).",
         "RF no lo necesita. GaussianNB tampoco: estima media y desvío de cada variable por "
         "separado."])

    s_rank = bloque(
        G["ranking"],
        f"Fuera de {c('duration')}, lo que más separa es el contexto económico: {c(a1)} "
        f"({num(r.ranking[a1], 3)}) y {c(a2)} ({num(r.ranking[a2], 3)})",
        f"Para cada numérica, el AUC de usar la variable sola como puntaje; se toma "
        f"max(AUC, 1 − AUC), así que 0,5 es azar y el sentido no importa. Para cada categórica, "
        f"el AUC de reemplazar cada nivel por su tasa de «yes» en train: es una medida "
        f"descriptiva, no un modelo, porque se calcula y se evalúa sobre los mismos datos. "
        f"{c('duration')} encabeza ({num(r.auc_dur, 3)}), pero es fuga; {c('age')} queda en "
        f"{num(r.auc_age, 3)} porque su relación es una U: con la edad en los tramos de la "
        f"sección Numéricas y la tasa de cada tramo, sube a {num(r.auc_age_tramos, 3)}.",
        ["Medir el modelo sin el bloque macro: si el desempeño cae mucho, lo que se aprende es "
         "sobre todo la época. Se midió: con los <i>folds</i> barajados, sacarlo cuesta AUC (A7, "
         "D-14); la época se ve validando hacia adelante en el tiempo, que es el hallazgo del TP "
         "(D-25).",
         "No descartar variables por este ranking: mide cada una sola, sin interacciones."])

    # ---- supuestos ---------------------------------------------------------------------
    supuestos = [
        ("NB", "Independencia condicional de las predictoras dada la clase. GaussianNB, además, "
               "una normal por clase para cada numérica.",
         f"{trio_txt} son casi el mismo dato (r entre {rmin} y {rmax}): NB los cuenta tres "
         f"veces. Las numéricas no son normales: {c('campaign')} tiene asimetría "
         f"{num(r.asim['campaign'], 2)}, {c('previous')} {num(r.asim['previous'], 2)}, "
         f"{c('pdays')} vale {PDAYS_CENTINELA} en el {pct(r.pct999, 1)} de las filas y la edad "
         f"tiene forma de U. {c('housing')} y {c('loan')} repiten el mismo «unknown»."),
        ("SVM", "Productos internos y distancias: necesita variables en escalas comparables. Con "
                "kernel RBF capta relaciones no lineales. El costo de entrenar crece más rápido "
                "que el número de filas.",
         f"Sin escalar, los desvíos difieren {num(r.sd_ratio, 0)} veces. La U de la edad y el "
         f"escalón del euribor son no lineales: el kernel RBF puede representarlos. Son "
         f"{num(r.n)} filas y {r.dims} columnas después del <i>one-hot</i>."),
        ("KNN", "Clasifica por distancia a los vecinos: todas las variables deben pesar "
                "parecido, y rinde peor con muchas dimensiones. El voto por mayoría favorece a "
                "la clase frecuente.",
         f"Sin escalar, {c(r.var_share.index[0])} y {c(r.var_share.index[1])} son el "
         f"{pct(r.var_top2, 1)} de la varianza: la distancia sería casi sólo esas dos. El "
         f"<i>one-hot</i> de las {palabra(len(r.cat))} categóricas da {r.n_onehot} columnas "
         f"binarias, que con las {palabra(len(r.num_modelo))} numéricas suman {r.dims} "
         f"dimensiones. El "
         f"{pct(r.acc_base, 1)} de las filas es «no»."),
        ("RF", "No necesita escalado ni supuestos de distribución. Captura interacciones y "
               "relaciones no monótonas con cortes sucesivos.",
         f"La diferencia de escalas ({num(r.sd_ratio, 0)} veces) no lo afecta. Puede combinar "
         f"{c('month')} con el bloque macro para reconocer la época (los meses de tasa alta "
         f"tienen {c('euribor3m')} medio {num(r.eur_altos, 2)} frente a {num(r.eur_resto, 2)}): "
         f"por eso conviene la comparación sin macro. La U de la edad y el escalón del euribor "
         f"se representan con cortes."),
    ]
    filas_sup = "\n".join(f"      <tr><td>{m}</td><td>{s}</td><td>{e}</td></tr>"
                          for m, s, e in supuestos)

    indice = [("claves", "Lo más importante"), ("acciones", "Qué hacer con esto"),
              ("objetivo", "El objetivo"), ("fuga", "Fuga de datos"), ("tiempo", "El tiempo"),
              ("macro", "Contexto económico"), ("previas", "Campañas previas"),
              ("faltantes", "Faltantes"), ("categorias", "Categorías"),
              ("numericas", "Numéricas"), ("escalas", "Escalas"),
              ("ranking", "Qué separa más"), ("supuestos", "Supuestos de cada modelo")]
    items_indice = "\n".join(f'      <li><a href="#{a}">{t_}</a></li>' for a, t_ in indice)
    items_claves = "\n".join(f"    <li>{k}</li>" for k in claves)

    cuerpo = f"""<div class="pagina">
<header>
  <p class="antetitulo">TP2 · Clasificación supervisada · análisis exploratorio</p>
  <h1>EDA de Bank Marketing</h1>
  <p class="bajada">Calculado sólo sobre el conjunto de entrenamiento: {num(r.n)} filas, el {pct(r.prop_train, 0)} del dataset sin duplicados. El conjunto de test no se abrió.</p>
  <div class="cifras">
    <div class="cifra"><div class="etiqueta">Filas de train</div><div class="valor">{num(r.n)}</div><div class="nota">{num(r.n_yes)} «yes» y {num(r.n_no)} «no»</div></div>
    <div class="cifra"><div class="etiqueta">Predictoras</div><div class="valor">{r.n_pred}</div><div class="nota">{len(r.num)} numéricas y {len(r.cat)} categóricas</div></div>
    <div class="cifra"><div class="etiqueta">Tasa de «yes»</div><div class="valor">{pct(t, 2)}</div><div class="nota">{num(r.ratio, 1)} «no» por cada «yes»</div></div>
    <div class="cifra"><div class="etiqueta">Período</div><div class="valor">{names.periodo}</div><div class="nota">según bank-additional-names.txt, §{names.seccion}</div></div>
  </div>
  <nav class="indice" aria-label="Contenido">
    <ul>
{items_indice}
    </ul>
  </nav>
</header>
<main>
<section id="claves">
  <h2>Lo más importante</h2>
  <ul class="claves">
{items_claves}
  </ul>
</section>
<section id="acciones">
  <h2>Qué hacer con esto</h2>
  <p>Cada fila es una propuesta de este análisis, anterior a las ablaciones de la ola 1, y remite al gráfico que la respalda; los «Qué hacer» de cada gráfico son esas mismas propuestas, cada una seguida de lo que se decidió. El estado dice qué se decidió después, y el porqué está en esa fila de <code>DECISIONES.md</code>. «Hecha»: se aplicó («en parte» si la decisión la acota). «Medida»: se probó en una ablación, y la decisión dice qué quedó. «Descartada»: se decidió lo contrario. «Decidida»: se resolvió con los datos, y la decisión cita la consulta a la cátedra sobre el tema.</p>
  <div class="tabla">
    <table>
      <thead><tr><th>Propuesta del EDA</th><th>Evidencia</th><th>Estado</th></tr></thead>
      <tbody>
{filas_dec}
      </tbody>
    </table>
  </div>
</section>
{seccion("objetivo", "El objetivo", s_obj)}
{seccion("fuga", "Fuga de datos", s_fuga)}
{seccion("tiempo", "El tiempo", s_tiempo + s_meses)}
{seccion("macro", "Contexto económico", s_corr + s_eur)}
{seccion("previas", "Campañas previas", s_prev + s_pdays)}
{seccion("faltantes", "Faltantes", s_unk)}
{seccion("categorias", "Categorías", s_cat + s_v)}
{seccion("numericas", "Numéricas", s_edad + s_camp + s_dens)}
{seccion("escalas", "Escalas", s_esc)}
{seccion("ranking", "Qué variables separan más", s_rank)}
<section id="supuestos">
  <h2>Supuestos de cada modelo</h2>
  <p>Qué necesita cada uno de los cuatro modelos y qué dice este EDA al respecto.</p>
  <div class="tabla">
    <table class="supuestos">
      <thead><tr><th>Modelo</th><th>Qué supone o necesita</th><th>Qué muestra este EDA</th></tr></thead>
      <tbody>
{filas_sup}
      </tbody>
    </table>
  </div>
</section>
</main>
<footer>
  <p>Para regenerar la página: <code>python -m src.eda_html</code> desde la carpeta del TP (con <code>--fragmento RUTA</code> escribe además la versión sin envoltura). Datos: <code>{RUTA_TRAIN.relative_to(RAIZ)}</code>, {num(r.n)} filas. Todas las cifras de esta página las calcula el script; ninguna está escrita a mano. Generada el {date.today().strftime("%d/%m/%Y")}.</p>
</footer>
</div>"""
    return cuerpo, claves


def documento(cuerpo, completo):
    cabeza = f"<title>EDA de Bank Marketing</title>\n<style>{CSS}</style>\n"
    if not completo:
        return cabeza + cuerpo + "\n"
    return ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"{cabeza}</head>\n<body>\n{cuerpo}\n</body>\n</html>\n")


class _Texto(HTMLParser):
    def __init__(self):
        super().__init__()
        self.partes = []

    def handle_data(self, data):
        self.partes.append(data)


def texto_plano(fragmento_html):
    p = _Texto()
    p.feed(fragmento_html)
    return re.sub(r"\s+", " ", "".join(p.partes)).strip()


def main():
    parser = argparse.ArgumentParser(prog="python -m src.eda_html",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--fragmento", type=Path,
                        help="escribe también la página sin <!doctype>/<html>/<head>/<body>")
    args = parser.parse_args()

    estilo()
    names = leer_names()
    df = cargar_train()
    r = analizar(df, names)

    G = {
        "balance": g_balance(r),
        "duration": g_duration(r),
        "bloques": g_bloques(r, df),
        "meses": g_meses(r),
        "correlacion": g_correlacion(r),
        "euribor": g_euribor(r),
        "previas": g_previas(r),
        "pdays": g_pdays(r),
        "unknown": g_unknown(r),
        "categorias": g_categorias(r),
        "cramer": g_cramer(r),
        "edad": g_edad(r),
        "campaign": g_campaign(r),
        "densidades": g_densidades(r, df),
        "escalas": g_escalas(r),
        "ranking": g_ranking(r),
    }
    cuerpo, claves = armar(r, names, G)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(documento(cuerpo, completo=True), encoding="utf-8")
    salidas = [SALIDA]
    if args.fragmento:
        args.fragmento.parent.mkdir(parents=True, exist_ok=True)
        args.fragmento.write_text(documento(cuerpo, completo=False), encoding="utf-8")
        salidas.append(args.fragmento)

    print(f"Train: {num(r.n)} filas, {pct(r.tasa, 2)} de «yes» (sólo train).")
    print(f"Gráficos: {len(G)}, PNG en total {num(sum(g.bytes for g in G.values()) / 1024)} KB")
    for nombre, g in G.items():
        print(f"  {nombre:<12} {g.ancho}x{g.alto} px  {num(g.bytes / 1024)} KB")
    print("Lo más importante:")
    for k in claves:
        print(f"  - {texto_plano(k)}")
    for ruta in salidas:
        print(f"Escrito: {ruta}  ({num(ruta.stat().st_size / 1024)} KB)")


if __name__ == "__main__":
    main()
