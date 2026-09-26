"""Correr con: python -m tests.test_graficos_presentacion

Las figuras de proyección del deck (src/graficos_presentacion.py) se prueban con DataFrames
sintéticos en el formato de sus entradas: ningún CSV de resultados/ y ningún modelo ajustado. Lo
dibujado se compara con valores que salen de los datos sintéticos, no del módulo: la altura de cada
punto y de cada barra, el valor de cada rótulo con coma decimal, y los colores de src/estilo.py. Las
series se buscan por su `gid`.

Además, a cada figura se le exige lo que la hace de proyección (guía de presentaciones, B5, B6 y
B8): el lienzo de 10,6 × 5,0 in, ningún título, ningún texto por debajo de 14 pt, ningún texto
fuera del lienzo, la leyenda por encima de los ejes, y en los revelados, el mismo encuadre en
todos los pasos. El test del CLI escribe la lista canónica completa en un directorio temporal, con
resultados sintéticos y el train real (cargar_train()): train es parte del repositorio.
"""

import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from PIL import Image

from src import estilo
from src import graficos_presentacion as gp
from src.datos import FILA, OBJETIVO, cargar_train
from src.estilo import COLOR_LINEA_BASE, ROTULO_LINEA_BASE
from src.graficos import numero
from src.metricas import METRICAS
from src.resultados import CAMPOS, configuracion_json

DESPLAZAMIENTOS = (-2, -1, 0, 1, 2)  # cinco folds: media exacta, desvío d·√(10/4)
RAIZ = np.sqrt(10 / 4)
ALFA_BANDA = 0.16

# La lista canónica del deck (la de la ola 7 sin las cuatro figuras de reserva, N0-15), escrita
# aquí y no tomada del módulo: si el módulo cambia un nombre, el deck deja de encontrar la figura,
# y este test lo tiene que ver.
CANONICAS = [
    "duration-techo.png", "orden-temporal.png", "pdays-999.png", "modelos-referencia.png",
    "curva-rf-1.png", "curva-rf-2.png", "curva-rf-3.png", "curva-knn.png", "curva-svm.png",
    "curva-svm-2.png", "modelos-final.png", "robustez-modelos.png", "robustez-folds.png",
    "robustez-folds-2.png",
]


def _color_modelo(modelo):
    return estilo.COLOR_MODELO["nb" if modelo.startswith("nb") else modelo]


# --- Utilidades ---------------------------------------------------------------------------------

@contextlib.contextmanager
def _abierta(fig):
    try:
        yield fig
    finally:
        plt.close(fig)


def _mismo_color(a, b):
    return np.allclose(to_rgba(a)[:3], to_rgba(b)[:3])


def _linea(fig, gid):
    for ax in fig.axes:
        for linea in ax.lines:
            if linea.get_gid() == gid:
                return linea
    raise AssertionError(f"no hay línea con gid {gid}")


def _texto(fig, gid):
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_gid() == gid:
                return t
    raise AssertionError(f"no hay texto con gid {gid}")


def _textos(fig, prefijo):
    return {t.get_gid(): t for ax in fig.axes for t in ax.texts
            if (t.get_gid() or "").startswith(prefijo)}


def _barra(ax, gid):
    """(línea de los puntos, segmentos de la barra, color de la barra) del errorbar `gid`."""
    for contenedor in ax.containers:
        if isinstance(contenedor, ErrorbarContainer) and contenedor.lines[0].get_gid() == gid:
            segmentos = [np.asarray(s, float) for s in contenedor.lines[2][0].get_segments()]
            return contenedor.lines[0], segmentos, contenedor.lines[2][0].get_color()[0]
    raise AssertionError(f"no hay barra de error con gid {gid}")


def _comprobar_barra_horizontal(ax, gid, x, y, d, color):
    puntos, segmentos, color_barra = _barra(ax, gid)
    assert np.allclose(puntos.get_xdata(), [x]) and np.allclose(puntos.get_ydata(), [y]), \
        (gid, puntos.get_xdata(), puntos.get_ydata(), x, y)
    assert np.allclose(segmentos[0], [[x - d, y], [x + d, y]]), (gid, segmentos[0], x, d)
    assert _mismo_color(puntos.get_color(), color) and _mismo_color(color_barra, color), gid


def _textos_visibles(fig):
    """Todos los textos que se ven: los anotados en los ejes, los de las marcas, los nombres de
    los ejes y los de las leyendas."""
    textos = []
    for ax in fig.axes:
        textos += list(ax.texts)
        textos += [t for t in ax.get_xticklabels() + ax.get_yticklabels()]
        textos += [ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend() is not None:
            textos += list(ax.get_legend().get_texts())
    for leyenda in fig.legends:
        textos += list(leyenda.get_texts())
    return [t for t in textos if t.get_visible() and t.get_text().strip()]


def _leyendas(fig):
    return [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)


def _comprobar_proyeccion(fig, con_leyenda=True):
    """Lo que toda figura de proyección cumple: el lienzo de ASPECTO_SLIDE a DPI, sin título ni
    supertítulo, ningún texto por debajo de TAM_MINIMO ni fuera del lienzo, ningún par de rótulos
    encimados, ningún valor anotado sobre un marcador, y la leyenda (si la hay) sobre el borde
    superior de los ejes."""
    assert np.allclose(fig.get_size_inches(), gp.ASPECTO_SLIDE), fig.get_size_inches()
    assert fig.dpi == gp.DPI, fig.dpi
    assert fig.get_suptitle() == "", fig.get_suptitle()
    for ax in fig.axes:
        for lado in ("center", "left", "right"):
            assert ax.get_title(lado) == "", (lado, ax.get_title(lado))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lienzo = fig.bbox
    for t in _textos_visibles(fig):
        assert t.get_fontsize() >= gp.TAM_MINIMO - 1e-9, (t.get_text(), t.get_fontsize())
        caja = t.get_window_extent(renderer)
        assert (caja.x0 >= lienzo.x0 - 1 and caja.x1 <= lienzo.x1 + 1
                and caja.y0 >= lienzo.y0 - 1 and caja.y1 <= lienzo.y1 + 1), \
            f"«{t.get_text()}» sale del lienzo: {caja} fuera de {lienzo}"
    rotulos = [t for ax in fig.axes for t in ax.texts if t.get_visible() and t.get_text()]
    cajas = [(t.get_text(), Text.get_window_extent(t, renderer)) for t in rotulos]
    for i, (a, ca) in enumerate(cajas):
        for b, cb in cajas[i + 1:]:
            assert not ca.overlaps(cb), f"los rótulos «{a}» y «{b}» se tapan"
    # Ningún valor anotado tapa un marcador visible (el de su punto incluido).
    marcadores = []
    for ax in fig.axes:
        for linea in ax.lines:
            if not linea.get_visible() or linea.get_marker() in (None, "None", "none", ""):
                continue
            radio = renderer.points_to_pixels(linea.get_markersize()) / 2
            xy = linea.get_transform().transform(np.asarray(linea.get_xydata(), float))
            marcadores += [(linea.get_gid(), x, y, radio) for x, y in xy
                           if np.isfinite(x) and np.isfinite(y)]
    for t in rotulos:
        if not (t.get_gid() or "").startswith(("valor", "rotulo-elegido")):
            continue
        caja = Text.get_window_extent(t, renderer).padded(-1)
        for gid, x, y, r in marcadores:
            assert not caja.overlaps(Bbox.from_extents(x - r, y - r, x + r, y + r)), \
                f"el valor «{t.get_text()}» tapa un marcador de {gid}"
    leyendas = _leyendas(fig)
    if con_leyenda:
        assert leyendas, "la figura no tiene leyenda"
        techo = max(ax.get_window_extent(renderer).y1 for ax in fig.axes)
        for leyenda in leyendas:
            assert leyenda.get_window_extent(renderer).y0 >= techo - 1, \
                "la leyenda no está arriba, fuera del área de datos"
    else:
        assert not leyendas, "esta figura no lleva leyenda"


def _comprobar_linea_base(fig, rotulo=ROTULO_LINEA_BASE):
    """«sin modelo»: la línea punteada gris de src/estilo.py y un rótulo visible que empieza con
    ese nombre, el mismo en todo el TP (guía C1)."""
    lineas = [l for ax in fig.axes for l in ax.lines if l.get_gid() == "linea-base"]
    assert lineas, "falta la línea de «sin modelo»"
    for l in lineas:
        assert _mismo_color(l.get_color(), COLOR_LINEA_BASE) and l.get_linestyle() == ":"
    rotulos = [t.get_text() for t in _textos(fig, "rotulo-linea-base").values()
               if t.get_visible()]
    assert rotulos and all(r.startswith(rotulo) for r in rotulos), rotulos


def _sin_linea_base(fig):
    assert not [l for ax in fig.axes for l in ax.lines if l.get_gid() == "linea-base"]
    assert not _textos(fig, "rotulo-linea-base")


# --- Datos sintéticos -----------------------------------------------------------------------------

# AUC de validación sin duration (A0) y con ella (A1), por modelo, como ablaciones_resumen.csv.
A1 = {"nb_gaussiano": (0.768, 0.831), "svm": (0.701, 0.907), "knn": (0.755, 0.911),
      "rf": (0.770, 0.939)}
# ΔAUC (media, desvío) de las demás variantes, que duration-techo tiene que dejar fuera: sólo
# dibuja A1. Los efectos se fijan a mano con el criterio del paso 1.5 (|media| contra desvío),
# para que las filas tengan el formato de ablaciones_resumen.csv.
DELTAS = {
    "A2": {"nb_gaussiano": (-0.0012, 0.0003), "svm": (0.0023, 0.0037)},
    "A3": {"rf": (0.0029, 0.0021), "knn": (0.0, 0.0028)},
    "A8": {"rf": (-0.093, 0.005), "svm": (-0.048, 0.006)},
    "A12": {"knn": (0.0084, 0.0067)},
}
EFECTOS = {("nb_gaussiano", "A2"): "empeora", ("svm", "A2"): "neutra", ("rf", "A3"): "mejora",
           ("knn", "A3"): "neutra", ("rf", "A8"): "empeora", ("svm", "A8"): "empeora",
           ("knn", "A12"): "mejora"}


def _ablaciones():
    filas = []
    for modelo, (sin, con) in A1.items():
        filas.append({"modelo": modelo, "variante": "A1", "auc_referencia": sin,
                      "auc_variante": con, "delta_media": con - sin, "delta_desvio": 0.01,
                      "n_folds": 5, "efecto": "mejora"})
    for variante, por_modelo in DELTAS.items():
        for modelo, (media, desvio) in por_modelo.items():
            filas.append({"modelo": modelo, "variante": variante,
                          "auc_referencia": A1[modelo][0],
                          "auc_variante": A1[modelo][0] + media, "delta_media": media,
                          "delta_desvio": desvio, "n_folds": 5,
                          "efecto": EFECTOS[(modelo, variante)]})
    return pd.DataFrame(filas)


# AUC y recall_q de validación (media, desvío) por modelo, como cv_*_resumen.csv.
CV_REFERENCIA = {
    "nb_gaussiano": {"auc": (0.769, 0.009), "recall_q": (0.566, 0.017)},
    "nb_categorico": {"auc": (0.782, 0.007), "recall_q": (0.614, 0.017)},
    "svm": {"auc": (0.702, 0.009), "recall_q": (0.549, 0.015)},
    "knn": {"auc": (0.755, 0.012), "recall_q": (0.593, 0.021)},
    "rf": {"auc": (0.772, 0.006), "recall_q": (0.605, 0.012)},
}
CV_FINAL = {
    "nb_categorico": {"auc": (0.782, 0.007), "recall_q": (0.614, 0.017)},
    "svm": {"auc": (0.774, 0.009), "recall_q": (0.6136, 0.018)},
    "knn": {"auc": (0.784, 0.008), "recall_q": (0.617, 0.016)},
    "rf": {"auc": (0.795, 0.006), "recall_q": (0.629, 0.018)},
}


def _resumen_cv(valores, base=(0.5, 0.2001)):
    filas = []
    for modelo, metricas in valores.items():
        for metrica, (media, desvio) in metricas.items():
            filas.append({"etiqueta": "x", "modelo": modelo, "configuracion": "{}",
                          "conjunto": "validacion", "metrica": metrica, "media": media,
                          "desvio": desvio, "n": 5, "error_estandar": desvio / np.sqrt(5)})
            # Un train distinto, para que nada se lea de ahí por error.
            filas.append({**filas[-1], "conjunto": "train", "media": 0.99, "desvio": 0.001})
    if base is not None:
        for metrica, valor in zip(("auc", "recall_q"), base):
            filas.append({"etiqueta": "x", "modelo": "sin_modelo", "configuracion": "{}",
                          "conjunto": "validacion", "metrica": metrica, "media": valor,
                          "desvio": 0.0, "n": 5, "error_estandar": 0.0})
    return pd.DataFrame(filas)


def _paso(i, conjunto):
    return 0.002 + 0.001 * i if conjunto == "validacion" else 0.0005 + 0.0005 * i


def _curva(modelo, parametro, valores, validacion, train, fijos=None):
    """Una curva en el formato largo de src/curvas.py: en el i-ésimo punto los cinco folds valen
    media + d·(−2, …, 2), con d = _paso(i, conjunto)."""
    filas = []
    for i, (valor, media_va, media_tr) in enumerate(zip(valores, validacion, train)):
        configuracion = configuracion_json({**(fijos or {}), parametro: valor})
        for fold, k in enumerate(DESPLAZAMIENTOS, start=1):
            for conjunto, media in (("train", media_tr), ("validacion", media_va)):
                for metrica in METRICAS:
                    filas.append({"parametro": parametro, "punto": str(valor), "modelo": modelo,
                                  "configuracion": configuracion, "fold": fold,
                                  "conjunto": conjunto, "metrica": metrica,
                                  "valor": media + _paso(i, conjunto) * k if metrica == "auc"
                                  else 0.4})
    tabla = pd.DataFrame(filas)[["parametro", "punto"] + CAMPOS]
    return pd.read_csv(io.StringIO(tabla.to_csv(index=False)))  # como la lee pandas


# RF: max_depth en desorden y con «sin límite» primero; el eje tiene que ordenarlos.
RF = ([None, 8, 2, 4, 12], [0.772, 0.795, 0.782, 0.789, 0.79], [1.0, 0.826, 0.783, 0.793, 0.902])
KNN = ([1, 5, 25, 101, 801], {"uniform": ([0.62, 0.72, 0.76, 0.78, 0.784],
                                          [0.99, 0.93, 0.85, 0.81, 0.79]),
                              "distance": ([0.61, 0.71, 0.75, 0.765, 0.77],
                                           [0.99, 1.0, 1.0, 1.0, 1.0])})
SVM = ([0.001, 0.01, 0.1, 1], {"lineal": ([0.774, 0.7743, 0.768, 0.767],
                                          [0.776, 0.777, 0.771, 0.770]),
                               "rbf": ([0.768, 0.772, 0.771, 0.73],
                                       [0.770, 0.784, 0.837, 0.93])})


def _rf():
    return _curva("rf", "max_depth", *RF, fijos={"n_estimators": 300})


def _knn(weights):
    return _curva("knn", "n_neighbors", KNN[0], *KNN[1][weights], fijos={"weights": weights})


def _svm(kernel):
    return _curva("svm", "C", SVM[0], *SVM[1][kernel],
                  fijos={"kernel": "linear" if kernel == "lineal" else "rbf",
                         "pesos_clase": "balanced"})


def _esperado(valores, medias, conjunto):
    return {str(v): (m, _paso(i, conjunto) * RAIZ) for i, (v, m) in enumerate(zip(valores, medias))}


# Robustez temporal: AUC (media, desvío) por modelo, esquema y variante. Las variantes de RF sin
# macro están para comprobar que robustez-modelos dibuja sólo la de todas las variables.
ROBUSTEZ = {
    ("rf", "barajado", "todas"): (0.795, 0.006), ("rf", "hacia_adelante", "todas"): (0.558, 0.123),
    ("knn", "barajado", "todas"): (0.784, 0.008),
    ("knn", "hacia_adelante", "todas"): (0.539, 0.065),
    ("svm", "barajado", "todas"): (0.774, 0.009),
    ("svm", "hacia_adelante", "todas"): (0.541, 0.078),
    ("rf", "barajado", "sin_macro"): (0.772, 0.010),
    ("rf", "hacia_adelante", "sin_macro"): (0.556, 0.110),
    ("rf", "barajado", "sin_macro_ni_month"): (0.736, 0.009),
    ("rf", "hacia_adelante", "sin_macro_ni_month"): (0.547, 0.098),
}


def _robustez_resumen():
    filas = [{"modelo": m, "esquema": e, "variante": v, "metrica": "auc", "media": media,
              "desvio": desvio, "n": 5, "error_estandar": desvio / np.sqrt(5)}
             for (m, e, v), (media, desvio) in ROBUSTEZ.items()]
    # Una fila de diferencia, que la figura no dibuja.
    filas.append({"modelo": "rf", "esquema": "barajado_menos_hacia_adelante",
                  "variante": "todas", "metrica": "auc", "media": 0.237, "desvio": np.nan,
                  "n": np.nan, "error_estandar": np.nan})
    return pd.DataFrame(filas)


ADELANTE = [0.489, 0.500, 0.426, 0.664, 0.711]
BARAJADO = [0.80, 0.79, 0.79, 0.80, 0.795]
MISMAS = [0.536, 0.519, 0.584, 0.748, 0.762]
PCT_YES = [4.7, 6.5, 5.6, 12.7, 35.2]


def _robustez_largo():
    filas = []
    for esquema, valores in (("hacia_adelante", ADELANTE), ("barajado", BARAJADO)):
        for fold, v in enumerate(valores, start=1):
            for conjunto in ("train", "validacion"):
                filas.append({"esquema": esquema, "variante": "todas", "modelo": "rf",
                              "configuracion": "{}", "fold": fold, "conjunto": conjunto,
                              "metrica": "auc", "valor": v if conjunto == "validacion" else 0.9})
    return pd.DataFrame(filas)


def _robustez_folds(filas_bloque=((100, 199), (200, 299), (300, 399), (400, 499), (500, 599))):
    filas = []
    for fold, ((desde, hasta), pct) in enumerate(zip(filas_bloque, PCT_YES), start=1):
        filas.append({"esquema": "hacia_adelante", "fold": fold, "n_train": 100 * fold,
                      "n_validacion": 100, "pct_yes_train": 3.0, "pct_yes_validacion": pct,
                      "fila_min_validacion": desde, "fila_max_validacion": hasta,
                      "fila_max_train": desde - 1})
        filas.append({"esquema": "barajado", "fold": fold, "n_train": 480, "n_validacion": 120,
                      "pct_yes_train": 11.3, "pct_yes_validacion": 11.3,
                      "fila_min_validacion": 0, "fila_max_validacion": 599,
                      "fila_max_train": 599})
    return pd.DataFrame(filas)


def _train_temporal():
    """8 000 filas en cuatro bloques de 2 000 con 10, 20, 30 y 50 % de «yes»; mayo y noviembre
    de 2008 hasta la fila 4 999, marzo de 2009 hasta la 6 999 y enero de 2010 al final (cada vez
    que el mes retrocede empieza un año). El euribor baja de 5 a 2 y a 1 con cada año."""
    fila = np.arange(8000)
    k = np.array([1, 2, 3, 5])[fila // 2000]
    mes = np.select([fila < 3000, fila < 5000, fila < 7000], ["may", "nov", "mar"], "jan")
    euribor = np.select([fila < 5000, fila < 7000], [5.0, 2.0], 1.0)
    return pd.DataFrame({FILA: fila, OBJETIVO: np.where(fila % 10 < k, "yes", "no"),
                         "month": mes, "euribor3m": euribor})


NAMES = SimpleNamespace(anio_ini=2008)


def _train_pdays():
    """100 filas con pdays = 999 y previous = 0 (10 «yes»), 40 con 999 y previous ≥ 1 (8 «yes»)
    y 20 con pdays < 999 (13 «yes»)."""
    def grupo(n, yes, pdays, previous):
        return pd.DataFrame({"pdays": pdays, "previous": previous,
                             OBJETIVO: ["yes"] * yes + ["no"] * (n - yes)})
    return pd.concat([grupo(100, 10, 999, 0), grupo(40, 8, 999, [1, 2] * 20),
                      grupo(20, 13, [3, 6, 9, 12] * 5, 1)], ignore_index=True)


# --- Tests ----------------------------------------------------------------------------------------

def test_lista_canonica_y_linea_base():
    archivos = [a for grupo in gp.FIGURAS.values() for a in grupo]
    assert archivos == CANONICAS, archivos
    assert gp.CON_LINEA_BASE <= set(CANONICAS)
    assert not {a for a in gp.CON_LINEA_BASE if a.startswith("curva-")}
    assert gp.ASPECTO_SLIDE == (10.6, 5.0) and gp.DPI == 200 and gp.TAM_MINIMO >= 14
    print("ok  la lista canónica tiene las 14 figuras del deck, en 16:9 apaisado a 200 dpi")


def test_duration_techo():
    with _abierta(gp.figura_duration_techo(_ablaciones())) as fig:
        _comprobar_proyeccion(fig)
        _comprobar_linea_base(fig)
        orden = sorted(A1, key=lambda m: -A1[m][1])  # de mayor a menor techo
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_yticklabels()] == \
            [gp.rotulo_modelo(m) for m in orden]
        sin, con = _linea(fig, "sin-duration"), _linea(fig, "con-duration")
        assert np.allclose(sin.get_xdata(), [A1[m][0] for m in orden])
        assert np.allclose(con.get_xdata(), [A1[m][1] for m in orden])
        assert np.allclose(sin.get_ydata(), range(4)) and np.allclose(con.get_ydata(), range(4))
        assert _mismo_color(sin.get_color(), estilo.AZUL)
        assert _mismo_color(con.get_color(), estilo.NARANJA)
        for m in orden:
            assert _texto(fig, f"valor:sin:{m}").get_text() == numero(A1[m][0], 3)
            assert _texto(fig, f"valor:con:{m}").get_text() == numero(A1[m][1], 3)
            flecha = _texto(fig, f"salto:{m}")
            assert np.allclose(flecha.xy, (A1[m][1], orden.index(m)))
        assert _texto(fig, "valor:con:rf").get_text() == "0,939"
    print("ok  duration-techo: A0 azul y A1 naranja por modelo, de mayor a menor techo, contra "
          "«sin modelo»")


def test_orden_temporal():
    s = gp.serie_temporal(_train_temporal(), NAMES)
    assert np.allclose(s.bloques["tasa"], [10, 20, 30, 50])
    assert np.allclose(s.bloques["x"], [999.5, 2999.5, 4999.5, 6999.5])
    assert s.anios == [2008, 2009, 2010] and np.allclose(s.fronteras, [4999.5, 6999.5])
    with _abierta(gp.figura_orden_temporal(_train_temporal(), NAMES)) as fig:
        _comprobar_proyeccion(fig)
        arriba, abajo = fig.axes
        tasa = _linea(fig, "tasa-yes")
        assert np.allclose(tasa.get_ydata(), [10, 20, 30, 50])
        assert _mismo_color(tasa.get_color(), estilo.COLOR_CLASE["yes"])
        assert np.allclose(_linea(fig, "euribor").get_ydata()[[0, 5000, 7999]], [5, 2, 1])
        for ax in (arriba, abajo):
            cambios = [l for l in ax.lines if (l.get_gid() or "").startswith("cambio-anio")]
            assert sorted(l.get_xdata()[0] for l in cambios) == [4999.5, 6999.5]
        assert [t.get_text() for t in abajo.get_xticklabels()] == ["2008", "2009", "2010"]
        x_lo, x_hi = abajo.get_xlim()
        assert np.allclose(abajo.get_xticks(),
                           [(x_lo + 4999.5) / 2, (4999.5 + 6999.5) / 2, (6999.5 + x_hi) / 2])
        assert _texto(fig, "valor:primer-bloque").get_text() == "10,0 %"
        assert _texto(fig, "valor:ultimo-bloque").get_text() == "50,0 %"
        assert arriba.get_ylim()[0] == 0 and "−" not in arriba.get_yticklabels()[0].get_text()
    print("ok  orden-temporal: % de «yes» por bloque de 2 000 filas, euribor y cambios de año, "
          "en dos paneles")


def test_pdays():
    g = gp.grupos_pdays(_train_pdays())
    assert g["n"].tolist() == [100, 40, 20] and g["yes"].tolist() == [10, 8, 13]
    assert np.allclose(g["tasa"], [10.0, 20.0, 65.0])
    with _abierta(gp.figura_pdays(_train_pdays())) as fig:
        _comprobar_proyeccion(fig, con_leyenda=False)
        ax = fig.axes[0]
        barras = {p.get_gid(): p for p in ax.patches}
        for grupo, tasa in zip(gp.GRUPOS_PDAYS, (10.0, 20.0, 65.0)):
            assert np.isclose(barras[f"tasa:{grupo}"].get_width(), tasa)
            assert _mismo_color(barras[f"tasa:{grupo}"].get_facecolor(),
                                estilo.COLOR_CLASE["yes"])
            assert _texto(fig, f"valor:{grupo}").get_text() == numero(tasa, 1) + " %"
        rotulos = [t.get_text() for t in ax.get_yticklabels()]
        assert "pdays = 999 y previous ≥ 1" in rotulos[1] and "40 filas" in rotulos[1]
        assert "100 filas" in rotulos[0] and "pdays < 999" in rotulos[2]
        assert ax.get_yticklabels()[1].get_fontweight() == "bold"
        assert ax.get_xticklabels()[0].get_text() == "0 %"
    print("ok  pdays-999: % de «yes» de los tres grupos, con 999 y previous ≥ 1 en negrita")


def test_modelos_referencia():
    with _abierta(gp.figura_modelos_referencia(_resumen_cv(CV_REFERENCIA))) as fig:
        _comprobar_proyeccion(fig, con_leyenda=False)
        _comprobar_linea_base(fig)
        ax = fig.axes[0]
        orden = sorted(CV_REFERENCIA, key=lambda m: -CV_REFERENCIA[m]["auc"][0])
        assert [t.get_text() for t in ax.get_yticklabels()] == \
            [gp.rotulo_modelo(m) for m in orden]
        for y, m in enumerate(orden):
            media, desvio = CV_REFERENCIA[m]["auc"]
            _comprobar_barra_horizontal(ax, f"auc:{m}", media, y, desvio, _color_modelo(m))
            assert _texto(fig, f"valor:auc:{m}").get_text() == numero(media, 3)
        assert np.allclose(_linea(fig, "linea-base").get_xdata(), [0.5, 0.5])
    # Sin filas sin_modelo, «sin modelo» es el AUC de un puntaje constante: 0,5.
    with _abierta(gp.figura_modelos_referencia(_resumen_cv(CV_REFERENCIA, base=None))) as fig:
        assert np.allclose(_linea(fig, "linea-base").get_xdata(), [0.5, 0.5])
    print("ok  modelos-referencia: los cinco de mayor a menor AUC, con su color y su valor, "
          "contra «sin modelo»")


def test_modelos_final():
    with _abierta(gp.figura_modelos_final(_resumen_cv(CV_FINAL))) as fig:
        _comprobar_proyeccion(fig, con_leyenda=False)
        _comprobar_linea_base(fig)
        izq, der = fig.axes
        orden = sorted(CV_FINAL, key=lambda m: -CV_FINAL[m]["auc"][0])
        assert orden[0] == "rf"
        for ax, metrica, base in ((izq, "auc", 0.5), (der, "recall_q", 0.2001)):
            for y, m in enumerate(orden):
                media, desvio = CV_FINAL[m][metrica]
                _comprobar_barra_horizontal(ax, f"{metrica}:{m}", media, y, desvio,
                                            _color_modelo(m))
                assert _texto(fig, f"valor:{metrica}:{m}").get_text() == numero(media, 3)
            linea = [l for l in ax.lines if l.get_gid() == "linea-base"][0]
            assert np.allclose(linea.get_xdata(), [base, base])
        assert "20 %" in der.get_xlabel()
    print("ok  modelos-final: AUC y recall al 20 %, en el mismo orden, cada uno contra su "
          "«sin modelo»")


def _pasos_rf():
    return [gp.figura_curva_rf(_rf(), paso) for paso in (1, 2, 3)]


def test_curva_rf_tres_pasos_con_el_mismo_encuadre():
    figs = _pasos_rf()
    try:
        for fig in figs:
            _comprobar_proyeccion(fig)
            _sin_linea_base(fig)
        ejes = [f.axes[0] for f in figs]
        for ax in ejes[1:]:
            assert ax.get_xlim() == ejes[0].get_xlim(), (ax.get_xlim(), ejes[0].get_xlim())
            assert ax.get_ylim() == ejes[0].get_ylim(), (ax.get_ylim(), ejes[0].get_ylim())
            assert np.allclose(ax.get_position().bounds, ejes[0].get_position().bounds)
        cajas = [f.axes[0].get_legend().get_window_extent(f.canvas.get_renderer()).bounds
                 for f in figs]
        assert all(np.allclose(c, cajas[0]) for c in cajas), cajas

        valores = RF[0]
        orden = ["2", "4", "8", "12", "None"]
        tr = _esperado(valores, RF[2], "train")
        va = _esperado(valores, RF[1], "validacion")
        eje = gp.eje_x("max_depth", orden)
        x_de = {p: gp._x(eje, p) for p in orden}
        numericos = orden[:-1]
        for fig in figs:
            linea = _linea(fig, "curva:train")
            assert np.allclose(linea.get_xdata(), [x_de[p] for p in numericos])
            assert np.allclose(linea.get_ydata(), [tr[p][0] for p in numericos])
            assert _mismo_color(linea.get_color(), estilo.COLOR_CONJUNTO["train"])
            validacion = _linea(fig, "curva:validacion")
            assert np.allclose(validacion.get_ydata(), [va[p][0] for p in numericos])
            assert _mismo_color(validacion.get_color(), estilo.COLOR_CONJUNTO["validacion"])
            assert np.allclose(_linea(fig, "sin-limite:curva:train").get_ydata(), [1.0])
            assert x_de["None"] > x_de["12"]
        # La banda de validación: ±1 desvío en cada x numérico, con alpha ≈ 0,16.
        banda = [c for c in figs[2].axes[0].collections
                 if c.get_gid() == "banda:curva:validacion"][0]
        vertices = np.vstack([p.vertices for p in banda.get_paths()])
        for p in numericos:
            ys = vertices[np.isclose(vertices[:, 0], x_de[p]), 1]
            assert np.isclose(ys.min(), va[p][0] - va[p][1]) and \
                np.isclose(ys.max(), va[p][0] + va[p][1]), (p, ys)
        assert abs(banda.get_facecolor()[0][3] - ALFA_BANDA) < 0.02

        def visible(fig, gid):
            return _linea(fig, gid).get_visible()
        # Paso 1: sólo train; paso 2: más validación; paso 3: bandas, zonas y la elegida.
        assert visible(figs[0], "curva:train") and not visible(figs[0], "curva:validacion")
        assert visible(figs[1], "curva:validacion") and not visible(figs[1], "elegido")
        assert not banda_visible(figs[1]) and banda_visible(figs[2]) and not banda_visible(figs[0])
        assert visible(figs[2], "elegido") and visible(figs[2], "brecha:elegido")
        estrella = _linea(figs[2], "elegido")
        assert np.allclose(estrella.get_xdata(), [x_de["8"]])
        assert np.allclose(estrella.get_ydata(), [va["8"][0]])
        assert np.allclose(_linea(figs[2], "brecha:elegido").get_ydata(),
                           [va["8"][0], tr["8"][0]])
        rotulo = _texto(figs[2], "rotulo-elegido")
        assert rotulo.get_text() == (f"profundidad 8: AUC {numero(0.795, 3)}\n"
                                     f"brecha con train: {numero(0.826 - 0.795, 3)}")
        assert rotulo.get_text().endswith("0,031")
        zonas = _textos(figs[2], "zona:")
        assert {t.get_text() for t in zonas.values()} == {"← subajuste", "sobreajuste →"}
        for fig in figs[:2]:
            assert not any(t.get_visible() for t in _textos(fig, "zona:").values())
            assert not _texto(fig, "rotulo-elegido").get_visible()
        leyendas = [f.axes[0].get_legend() for f in figs]
        visibles = [[t.get_visible() for t in l.get_texts()] for l in leyendas]
        assert visibles == [[True, False, False, False], [True, True, False, False],
                            [True, True, True, True]], visibles
        negrita = [t.get_text() for t in figs[2].axes[0].get_xticklabels()
                   if t.get_fontweight() == "bold"]
        assert negrita == ["8"], negrita
        assert not [t for t in figs[0].axes[0].get_xticklabels() if t.get_fontweight() == "bold"]
    finally:
        for fig in figs:
            plt.close(fig)
    print("ok  curva-rf-1/2/3: mismo encuadre, ejes y leyenda; train, + validación, + bandas, "
          "zonas y profundidad 8")


def banda_visible(fig):
    return any(c.get_visible() for c in fig.axes[0].collections
               if (c.get_gid() or "").startswith("banda:"))


def test_curva_knn():
    with _abierta(gp.figura_curva_knn(_knn("uniform"), _knn("distance"), elegido=801)) as fig:
        _comprobar_proyeccion(fig)
        _sin_linea_base(fig)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        va = _esperado(KNN[0], KNN[1]["uniform"][0], "validacion")
        linea = _linea(fig, "uniform:validacion")
        assert np.allclose(linea.get_xdata(), KNN[0])
        assert np.allclose(linea.get_ydata(), KNN[1]["uniform"][0])
        atenuada = _linea(fig, "distance:validacion")
        assert np.allclose(atenuada.get_ydata(), KNN[1]["distance"][0])
        assert atenuada.get_linestyle() == "--" and atenuada.get_alpha() < 1
        assert atenuada.get_marker() in ("None", "none", None, "")
        assert not [c for c in ax.collections if "distance" in (c.get_gid() or "")]
        estrella = _linea(fig, "elegido")
        assert np.allclose(estrella.get_xdata(), [801]) and \
            np.allclose(estrella.get_ydata(), [va["801"][0]])
        assert _texto(fig, "rotulo-elegido").get_text().startswith("k = 801: AUC 0,784")
        assert _texto(fig, "zona:sobreajuste").get_text() == "← sobreajuste"
        rotulados = [t.get_text() for t in ax.get_xticklabels()]
        assert rotulados[0] == "1" and rotulados[-1] == "801", rotulados
        assert [t.get_text() for t in ax.get_xticklabels() if t.get_fontweight() == "bold"] == \
            ["801"]
    print("ok  curva-knn: uniform en escala log con k = 801 marcado y distance atenuada")


def test_curva_svm_dos_pasos_con_el_mismo_encuadre():
    figs = [gp.figura_curva_svm(_svm("lineal"), _svm("rbf"), paso) for paso in (1, 2)]
    try:
        for fig in figs:
            _comprobar_proyeccion(fig)
            _sin_linea_base(fig)
        a, b = (f.axes[0] for f in figs)
        assert a.get_xlim() == b.get_xlim() and a.get_ylim() == b.get_ylim()
        assert np.allclose(a.get_position().bounds, b.get_position().bounds)
        assert a.get_xscale() == "log"
        # El encuadre reserva el lugar de la RBF desde el paso 1.
        assert a.get_ylim()[1] >= max(SVM[1]["rbf"][1])
        rbf_1, rbf_2 = _linea(figs[0], "rbf:train"), _linea(figs[1], "rbf:train")
        assert not rbf_1.get_visible() and rbf_2.get_visible()
        assert np.allclose(rbf_2.get_ydata(), SVM[1]["rbf"][1]) and rbf_2.get_linestyle() == "--"
        assert not _texto(figs[0], "zona:sobreajuste").get_visible()
        assert _texto(figs[1], "zona:sobreajuste").get_visible()
        for fig in figs:
            estrella = _linea(fig, "elegido")
            assert np.allclose(estrella.get_xdata(), [0.001]) and estrella.get_visible()
            assert np.allclose(_linea(fig, "lineal:validacion").get_ydata(), SVM[1]["lineal"][0])
            assert _texto(fig, "rotulo-elegido").get_text().startswith("C = 0,001: AUC 0,774")
        visibles = [[t.get_visible() for t in f.axes[0].get_legend().get_texts()] for f in figs]
        assert visibles == [[True, True, True, True, False], [True] * 5], visibles
    finally:
        for fig in figs:
            plt.close(fig)
    try:
        gp.figura_curva_svm(_svm("lineal"), None, paso=2)
        raise AssertionError("el paso 2 sin la curva RBF tenía que fallar")
    except ValueError:
        pass
    print("ok  curva-svm y -2: la lineal con C = 0,001 marcado y, en el mismo encuadre, la RBF")


def test_robustez_modelos():
    with _abierta(gp.figura_robustez_modelos(_robustez_resumen())) as fig:
        _comprobar_proyeccion(fig)
        _comprobar_linea_base(fig)
        ax = fig.axes[0]
        orden = ["rf", "knn", "svm"]  # por el AUC barajado
        assert [t.get_text() for t in ax.get_yticklabels()] == \
            [gp.rotulo_modelo(m) for m in orden]
        for i, m in enumerate(orden):
            for esquema, color, dy in (("barajado", estilo.AZUL, -0.17),
                                       ("hacia_adelante", estilo.NARANJA, 0.17)):
                media, desvio = ROBUSTEZ[(m, esquema, "todas")]
                _comprobar_barra_horizontal(ax, f"{esquema}:{m}", media, i + dy, desvio, color)
                assert _texto(fig, f"valor:{esquema}:{m}").get_text() == numero(media, 3)
    print("ok  robustez-modelos: barajado azul contra hacia adelante naranja, con «sin modelo»")


def test_auc_barajado_por_bloque_y_periodos():
    # Bloque 1 ordenado a la perfección (AUC 1), bloque 2 al revés (AUC 0) y bloque 3 con una
    # sola clase (sin AUC). Las filas fuera de los bloques no cuentan.
    train = pd.DataFrame({FILA: np.arange(30), OBJETIVO: ["no", "yes"] * 15})
    y = (train[OBJETIVO] == "yes").to_numpy()
    puntaje = np.where(y, 1.0, 0.0)
    puntaje[10:20] = 1 - puntaje[10:20]
    train.loc[20:29, OBJETIVO] = "no"
    oof = pd.DataFrame({"fila": np.arange(30), "fold": 1, "puntaje": puntaje})
    folds = _robustez_folds(((0, 9), (10, 19), (20, 29), (30, 39), (40, 49)))
    auc = gp.auc_barajado_por_bloque(oof, train, folds)
    assert auc.index.tolist() == [1, 2, 3, 4, 5]
    assert auc[1] == 1.0 and auc[2] == 0.0 and all(np.isnan(auc[k]) for k in (3, 4, 5))

    meses = ["may"] * 10 + ["jul"] * 10 + ["nov"] * 5 + ["mar"] * 5
    train_meses = pd.DataFrame({FILA: np.arange(30), "month": meses})
    periodos = gp.periodos_bloques(train_meses, folds, NAMES)
    assert periodos == {1: ("may", "2008"), 2: ("jul", "2008"), 3: ("nov–mar", "2008–2009")}, \
        periodos
    print("ok  el AUC barajado en las filas de cada bloque y el período de cada bloque")


def _pasos_folds():
    largo, folds = _robustez_largo(), _robustez_folds()
    mismas = pd.Series(MISMAS, index=range(1, 6))
    periodos = {1: ("may–jul", "2008"), 4: ("nov–may", "2008–2009")}
    return [gp.figura_robustez_folds(largo, folds, mismas, periodos, paso) for paso in (1, 2)]


def test_robustez_folds_dos_pasos():
    figs = _pasos_folds()
    try:
        for fig in figs:
            _comprobar_proyeccion(fig)
            _comprobar_linea_base(fig)
        a, b = (f.axes[0] for f in figs)
        assert a.get_xlim() == b.get_xlim() and a.get_ylim() == b.get_ylim()
        assert np.allclose(a.get_position().bounds, b.get_position().bounds)
        for fig in figs:
            adelante = _linea(fig, "hacia-adelante")
            assert np.allclose(adelante.get_xdata(), range(1, 6))
            assert np.allclose(adelante.get_ydata(), ADELANTE)
            assert _mismo_color(adelante.get_color(), estilo.NARANJA)
            referencia = _linea(fig, "referencia-barajado")
            assert np.allclose(referencia.get_ydata(), [np.mean(BARAJADO)] * 2)
            for fold, v in enumerate(ADELANTE, start=1):
                assert _texto(fig, f"valor:adelante:{fold}").get_text() == numero(v, 3)
            assert _texto(fig, "rotulo-barajado").get_text().endswith(numero(np.mean(BARAJADO), 3))
            rotulos = [t.get_text() for t in fig.axes[0].get_xticklabels()]
            assert rotulos[0] == "may–jul\n2008\n4,7 % «yes»", rotulos[0]
            assert rotulos[3] == "nov–may\n2008–2009\n12,7 % «yes»", rotulos[3]
            assert rotulos[1] == "bloque 2\n6,5 % «yes»", rotulos[1]
            tallos = [c for c in fig.axes[0].collections if c.get_gid() == "tallos"][0]
            for segmento, v in zip(tallos.get_segments(), ADELANTE):
                assert np.allclose(np.asarray(segmento)[:, 1], [0.5, v])
        azul_1, azul_2 = (_linea(f, "barajado-mismas-filas") for f in figs)
        assert not azul_1.get_visible() and azul_2.get_visible()
        assert np.allclose(azul_2.get_ydata(), MISMAS)
        assert np.allclose(azul_2.get_xdata(), np.arange(1, 6) + gp.DESPLAZAMIENTO_MISMAS)
        assert _mismo_color(azul_2.get_markeredgecolor(), estilo.AZUL)
        for fold, v in enumerate(MISMAS, start=1):
            assert not _texto(figs[0], f"valor:mismas:{fold}").get_visible()
            assert _texto(figs[1], f"valor:mismas:{fold}").get_text() == numero(v, 3)
            # El rótulo hacia adelante queda en el mismo lugar en los dos pasos.
            assert _texto(figs[0], f"valor:adelante:{fold}").xyann == \
                _texto(figs[1], f"valor:adelante:{fold}").xyann
    finally:
        for fig in figs:
            plt.close(fig)
    print("ok  robustez-folds y -2: RF hacia adelante por bloque y, en el mismo encuadre, el "
          "barajado en las mismas filas")


def test_marcas_sin_menos_cero():
    fig, ejes = gp._lienzo()
    try:
        ax = ejes[0, 0]
        ax.set_xlim(-0.0, 60)
        gp._marcas(ax, "x", 20, formato=gp._formato_pct)
        fig.canvas.draw()
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0 %", "20 %", "40 %", "60 %"]
        assert gp._paso_marcas(0.72, 1.01) == 0.05 and gp._paso_marcas(0.58, 1.02) == 0.1
    finally:
        plt.close(fig)
    print("ok  las marcas no dicen «−0» y el paso deja como mucho siete")


# --- CLI ------------------------------------------------------------------------------------------

def _resultados_sinteticos(directorio):
    """Un resultados/ con las entradas de todas las figuras, sintéticas, y las filas de
    validación hacia adelante tomadas del train real (para los períodos y el AUC por bloque)."""
    d = Path(directorio)
    (d / "curvas").mkdir(parents=True)
    _ablaciones().to_csv(d / "ablaciones_resumen.csv", index=False)
    _resumen_cv(CV_REFERENCIA).to_csv(d / "cv_referencia_resumen.csv", index=False)
    _resumen_cv(CV_FINAL).to_csv(d / "cv_final_resumen.csv", index=False)
    _rf().to_csv(d / "curvas" / "rf_max_depth.csv", index=False)
    _knn("uniform").to_csv(d / "curvas" / "knn_n_neighbors_uniform.csv", index=False)
    _knn("distance").to_csv(d / "curvas" / "knn_n_neighbors_distance.csv", index=False)
    _svm("lineal").to_csv(d / "curvas" / "svm_C_linear_balanced.csv", index=False)
    _svm("rbf").to_csv(d / "curvas" / "svm_C_balanced.csv", index=False)
    _robustez_resumen().to_csv(d / "robustez_temporal_resumen.csv", index=False)
    _robustez_largo().to_csv(d / "robustez_temporal.csv", index=False)
    filas = np.sort(cargar_train()[FILA].to_numpy())
    seis = np.array_split(filas, 6)
    _robustez_folds([(int(b[0]), int(b[-1])) for b in seis[1:]]).to_csv(
        d / "robustez_folds.csv", index=False)
    azar = np.random.default_rng(42)
    pd.DataFrame({"fila": filas, "fold": azar.integers(1, 6, len(filas)),
                  "puntaje": azar.random(len(filas))}).to_csv(d / "oof_final_rf.csv",
                                                              index=False)


def test_cli_escribe_la_lista_canonica():
    with tempfile.TemporaryDirectory() as tmp:
        resultados, figuras = Path(tmp) / "resultados", Path(tmp) / "figuras"
        _resultados_sinteticos(resultados)
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            codigo = gp.main(["--resultados", str(resultados), "--figuras", str(figuras)])
        assert codigo == 0, salida.getvalue()
        escritas = sorted(p.name for p in figuras.glob("*.png"))
        assert escritas == sorted(CANONICAS), escritas
        for nombre in CANONICAS:
            with Image.open(figuras / nombre) as im:
                assert im.size == (2120, 1000), (nombre, im.size)
        assert "14 figuras" in salida.getvalue()

        # Una entrada que falta es un error: se informa, se siguen las demás, y sale con 1.
        (resultados / "cv_final_resumen.csv").unlink()
        otras = Path(tmp) / "otras"
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            codigo = gp.main(["--resultados", str(resultados), "--figuras", str(otras),
                              "--solo", "modelos-final", "modelos-referencia"])
        assert codigo == 1
        assert sorted(p.name for p in otras.glob("*.png")) == ["modelos-referencia.png"]
        assert "modelos-final" in salida.getvalue() and "con error" in salida.getvalue()
    assert not plt.get_fignums(), f"quedaron figuras abiertas: {plt.get_fignums()}"
    print("ok  el CLI escribe las 14 figuras de la lista canónica a 2 120 × 1 000 píxeles; una "
          "entrada que falta sale con código 1")


def main():
    test_lista_canonica_y_linea_base()
    test_duration_techo()
    test_orden_temporal()
    test_pdays()
    test_modelos_referencia()
    test_modelos_final()
    test_curva_rf_tres_pasos_con_el_mismo_encuadre()
    test_curva_knn()
    test_curva_svm_dos_pasos_con_el_mismo_encuadre()
    test_robustez_modelos()
    test_auc_barajado_por_bloque_y_periodos()
    test_robustez_folds_dos_pasos()
    test_marcas_sin_menos_cero()
    test_cli_escribe_la_lista_canonica()
    assert not plt.get_fignums(), f"quedaron figuras abiertas: {plt.get_fignums()}"
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
