"""Correr con: python -m tests.test_graficos

Las figuras de análisis (src/graficos.py) se prueban con DataFrames sintéticos en el formato del
contrato (src/resultados.py): ningún CSV de resultados/ y ningún modelo ajustado. Cada test mira la
Figure que devuelve la función —ejes, series, rótulos y valores anotados— y la cierra. Las series
se buscan por su `gid`, que src/graficos.py les pone para esto.

Lo dibujado se compara con valores esperados que salen de los datos sintéticos, no de
src/graficos.py: la altura de cada punto de cada serie, el largo de cada barra de error y de cada
banda (±1 desvío entre folds) y los colores de src/estilo.py. Así un error en el módulo no pasa por
bueno al compararse consigo mismo. El test del CLI escribe los PNG en un directorio temporal.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.text import Text
from matplotlib.transforms import Bbox, Transform
from PIL import Image

from src import estilo
from src.ablaciones import VARIANTES
from src.ablaciones import efecto as efecto_ablaciones
from src.estilo import COLOR_LINEA_BASE, ROTULO_LINEA_BASE, SUPERFICIE
from src.graficos import (
    FIGURA_ABLACIONES,
    FIGURA_EFECTO,
    FIGURA_MODELOS,
    ROTULO_MODELO,
    ROTULO_VARIANTE,
    delta,
    eje_x,
    efecto,
    figura_ablaciones,
    figura_ablaciones_efecto,
    figura_curva,
    figura_curvas_comparadas,
    figura_modelos,
    nombre_figura_curva,
    numero,
    por_filas,
    puntos_de,
)
from src.graficos import main as cli
from src.metricas import METRICAS, PRESUPUESTO
from src.resultados import CAMPOS, configuracion_json

DESPLAZAMIENTOS = (-2, -1, 0, 1, 2)  # cinco folds: media exacta, desvío d·√(10/4)
RAIZ = np.sqrt(10 / 4)               # el desvío (ddof=1) de d·(−2, −1, 0, 1, 2), dividido por d
MODELOS_ABLACION = ("nb_gaussiano", "svm", "knn", "rf")
CONJUNTOS = ("train", "validacion")

# La especificación de colores, tomada de src/estilo.py y no de src/graficos.py.
COLOR_EFECTO = {"mejora": estilo.AZUL, "empeora": estilo.NARANJA, "neutra": estilo.REJILLA}
# El glifo que tiene que acompañar al valor del máximo de cada curva en una figura combinada: el
# carácter con la forma del marcador de la curva.
GLIFO = {"o": "●", "s": "■", "D": "◆", "^": "▲"}
ALFA_BANDA = 0.16  # ±1 desvío con alpha ≈ 0,16, como en el TP1


def _color_modelo(modelo):
    """El color de un modelo según src/estilo.py: las dos variantes de NB comparten el de la
    familia."""
    return estilo.COLOR_MODELO["nb" if modelo.startswith("nb") else modelo]


# --- Datos sintéticos -------------------------------------------------------------------------

# (media, desvío) de ΔAUC por variante y modelo, en el orden nb_gaussiano, svm, knn, rf. Las
# variantes de un solo modelo o de dos se miden sólo en esos, como en src/ablaciones.py.
DELTAS = {
    "A1": {"nb_gaussiano": (0.063, 0.006), "svm": (0.206, 0.016), "knn": (0.156, 0.013),
           "rf": (0.170, 0.007)},
    "A2": {"nb_gaussiano": (-0.0012, 0.0003), "svm": (0.0023, 0.0037), "knn": (-0.0019, 0.0007),
           "rf": (0.0003, 0.0022)},
    "A3": {"nb_gaussiano": (-0.0003, 0.0017), "svm": (0.0004, 0.0012), "knn": (0.0, 0.0028),
           "rf": (0.0029, 0.0021)},
    "A4": {m: (0.0002, 0.0006) for m in MODELOS_ABLACION},
    "A5": {"svm": (-0.0006, 0.0020), "knn": (0.0008, 0.0026)},
    "A6": {m: (0.0005, 0.0028) for m in MODELOS_ABLACION},
    "A7": {"nb_gaussiano": (-0.0225, 0.0037), "svm": (-0.0037, 0.0115),
           "knn": (-0.0475, 0.0080), "rf": (-0.0276, 0.0041)},
    "A8": {"nb_gaussiano": (-0.050, 0.005), "svm": (-0.048, 0.006), "knn": (-0.076, 0.007),
           "rf": (-0.093, 0.005)},
    "A9": {m: (-0.0010, 0.0030) for m in MODELOS_ABLACION},
    "A10": {"nb_gaussiano": (0.0001, 0.0022)},
    "A11": {m: (0.0001, 0.0039) for m in MODELOS_ABLACION},
    "A12": {"knn": (0.0084, 0.0067)},
}
FILAS_ABLACION = sum(len(v) for v in DELTAS.values())  # 40
# Las que se anotan: todo A1, A7 y A8, y las que mejoran o empeoran fuera de esas.
ANOTADAS_ESPERADAS = ({(m, v) for v in ("A1", "A7", "A8") for m in DELTAS[v]}
                      | {("nb_gaussiano", "A2"), ("knn", "A2"), ("rf", "A3"), ("knn", "A12")})


def _resumen_ablaciones():
    """Como ablaciones_resumen.csv: una fila por modelo y variante, sin A0."""
    filas = []
    for variante, por_modelo in DELTAS.items():
        for modelo, (media, desvio) in por_modelo.items():
            filas.append({"modelo": modelo, "variante": variante, "auc_referencia": 0.75,
                          "auc_variante": 0.75 + media, "delta_media": media,
                          "delta_desvio": desvio, "delta_error_estandar": desvio / np.sqrt(5),
                          "delta_min": media - desvio, "delta_max": media + desvio,
                          "n_folds": 5, "delta_recall_media": 0.0, "delta_recall_desvio": 0.0,
                          "columnas": 60, "efecto": efecto_ablaciones(media, desvio)})
    return pd.DataFrame(filas)


# AUC y recall_q medios de validación (y desvío) por modelo, y un AUC de train distinto, para que
# el orden de la figura 02 no pueda salir de train.
CV = {
    "nb_gaussiano": {"auc": (0.769, 0.009), "recall_q": (0.566, 0.017), "train": 0.775},
    "nb_categorico": {"auc": (0.782, 0.007), "recall_q": (0.614, 0.017), "train": 0.790},
    "svm": {"auc": (0.702, 0.009), "recall_q": (0.549, 0.015), "train": 0.950},
    "knn": {"auc": (0.755, 0.012), "recall_q": (0.593, 0.021), "train": 0.870},
    "rf": {"auc": (0.772, 0.006), "recall_q": (0.605, 0.012), "train": 1.000},
}
ORDEN_POR_AUC = ["nb_categorico", "rf", "nb_gaussiano", "knn", "svm"]


def _resumen_cv(con_linea_base=True):
    """Como cv_referencia_resumen.csv de src/experimentos.py: media, desvío, n y error estándar
    por etiqueta, modelo, configuración, conjunto (train, validacion, brecha) y métrica."""
    filas = []

    def fila(modelo, conjunto, metrica, media, desvio):
        filas.append({"etiqueta": "referencia", "modelo": modelo, "configuracion": "{}",
                      "conjunto": conjunto, "metrica": metrica, "media": media,
                      "desvio": desvio, "n": 5, "error_estandar": desvio / np.sqrt(5)})

    for modelo, valores in CV.items():
        for metrica in METRICAS:
            media, desvio = valores.get(metrica, (0.3, 0.01))
            fila(modelo, "validacion", metrica, media, desvio)
            fila(modelo, "train", metrica, valores["train"] if metrica == "auc" else 0.9, 0.001)
        fila(modelo, "brecha", "auc", valores["train"] - valores["auc"][0], 0.004)
    if con_linea_base:
        for conjunto in CONJUNTOS:
            for metrica, valor in (("auc", 0.5), ("ap", 0.1127), ("recall_q", 0.2001),
                                   ("precision_q", 0.1127), ("f1_q", 0.1441)):
                fila("sin_modelo", conjunto, metrica, valor, 0.0)
    return pd.DataFrame(filas)


def _paso(i, conjunto):
    """El paso d de los folds en el i-ésimo punto de una curva sintética: distinto en cada punto
    y en cada conjunto, para que una barra o una banda de otro punto o del otro conjunto no pase
    por buena."""
    return 0.002 + 0.001 * i if conjunto == "validacion" else 0.0005 + 0.0005 * i


def _curva(modelo, parametro, valores, validacion, train, fijos=None):
    """Una curva como la escribe src/curvas.py: `parametro` y `punto` (el valor como texto) y las
    seis columnas del contrato. En el i-ésimo punto los cinco folds valen media + d·(−2, …, 2),
    con d = _paso(i, conjunto): la media es exacta y el desvío, d·√(10/4)."""
    filas = []
    for i, (valor, media_va, media_tr) in enumerate(zip(valores, validacion, train)):
        configuracion = configuracion_json({**(fijos or {}), parametro: valor})
        for fold, k in enumerate(DESPLAZAMIENTOS, start=1):
            for conjunto, media in (("train", media_tr), ("validacion", media_va)):
                for metrica in METRICAS:
                    auc = media + _paso(i, conjunto) * k
                    filas.append({"parametro": parametro, "punto": str(valor), "modelo": modelo,
                                  "configuracion": configuracion, "fold": fold,
                                  "conjunto": conjunto, "metrica": metrica,
                                  "valor": auc if metrica == "auc" else 0.4})
    return pd.DataFrame(filas)[["parametro", "punto"] + CAMPOS]


def _esperado(valores, validacion, train):
    """{conjunto: {punto como texto: (media, desvío)}}: lo que la figura de una curva sintética
    tiene que dibujar en cada punto, calculado desde los datos y no desde src/graficos.py."""
    return {conjunto: {str(v): (m, _paso(i, conjunto) * RAIZ)
                       for i, (v, m) in enumerate(zip(valores, medias))}
            for conjunto, medias in (("train", train), ("validacion", validacion))}


def _como_la_lee_pandas(tabla):
    """El ida y vuelta por CSV: pandas lee el texto «None» como NaN y «2» como 2.0."""
    return pd.read_csv(io.StringIO(tabla.to_csv(index=False)))


MAX_DEPTH = ([None, 8, 2, 4, 12], [0.772, 0.797, 0.782, 0.789, 0.795],
             [1.0, 0.858, 0.783, 0.793, 0.902])


def _max_depth():
    # En desorden, con None primero: el eje tiene que ordenarlos igual.
    return _curva("rf", "max_depth", *MAX_DEPTH, fijos={"n_estimators": 300})


VECINOS = [1, 5, 25, 101]
KNN = {"uniform": ([0.62, 0.72, 0.76, 0.78], [0.99, 0.93, 0.85, 0.81]),
       "distance": ([0.61, 0.73, 0.75, 0.74], [0.99, 1.0, 1.0, 1.0])}


def _knn(weights):
    return _curva("knn", "n_neighbors", VECINOS, *KNN[weights], fijos={"weights": weights})


# Tres curvas de C de la SVM; las dos balanceadas tienen el máximo en el mismo C = 0,01.
VALORES_C = [0.001, 0.01, 0.1, 1]
SVM_C = {
    "svm_C": ([0.700, 0.703, 0.706, 0.702], [0.75, 0.80, 0.87, 0.90],
              {"kernel": "rbf", "gamma": "scale"}),
    "svm_C_balanced": ([0.768, 0.772, 0.770, 0.765], [0.770, 0.784, 0.837, 0.876],
                       {"kernel": "rbf", "gamma": "scale", "pesos_clase": "balanced"}),
    "svm_C_linear_balanced": ([0.773, 0.774, 0.771, 0.770], [0.775, 0.777, 0.772, 0.771],
                              {"kernel": "linear", "gamma": "scale",
                               "implementacion": "liblinear", "pesos_clase": "balanced"}),
}


def _svm_c(nombre):
    validacion, train, fijos = SVM_C[nombre]
    return _curva("svm", "C", VALORES_C, validacion, train, fijos=fijos)


# --- Utilidades -------------------------------------------------------------------------------

@contextlib.contextmanager
def _abierta(fig):
    try:
        yield fig
    finally:
        plt.close(fig)


def _lineas(ax, prefijo):
    return {l.get_gid(): l for l in ax.lines if (l.get_gid() or "").startswith(prefijo)}


def _textos(ax, prefijo=""):
    return {t.get_gid(): t for t in ax.texts if (t.get_gid() or "").startswith(prefijo)}


def _rotulos_x(ax):
    return [t.get_text().replace("\n", " ") for t in ax.get_xticklabels()]


def _cerca(a, b, tol=1e-9):
    return abs(a - b) < tol


def _mismo_color(a, b):
    return np.allclose(to_rgba(a), to_rgba(b))


def _en_silencio(funcion, *argumentos):
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        resultado = funcion(*argumentos)
    return resultado, salida.getvalue()


def _renglon(salida, comienzo, *fragmentos):
    """Si la salida del CLI tiene un renglón que empieza con `comienzo` y tiene los fragmentos."""
    return any(r.startswith(comienzo) and all(f in r for f in fragmentos)
               for r in salida.splitlines())


def _barra(ax, gid):
    """La barra de error cuyos puntos llevan `gid`: (línea de los puntos, segmentos de la barra
    en el orden de los puntos, color de la barra)."""
    for contenedor in ax.containers:
        if isinstance(contenedor, ErrorbarContainer) and contenedor.lines[0].get_gid() == gid:
            colecciones = contenedor.lines[2]
            assert len(colecciones) == 1, f"{gid}: la barra va en una sola dirección"
            segmentos = [np.asarray(s, float) for s in colecciones[0].get_segments()]
            return contenedor.lines[0], segmentos, colecciones[0].get_color()[0]
    raise AssertionError(f"no hay barra de error con gid {gid}")


def _comprobar_barras_verticales(ax, gid, medias, desvios):
    """Cada punto de la barra `gid` a la altura de su media, con una barra de media − desvío a
    media + desvío; devuelve la línea de los puntos."""
    puntos, segmentos, _ = _barra(ax, gid)
    xs = np.asarray(puntos.get_xdata(), float)
    assert np.allclose(puntos.get_ydata(), medias), (gid, list(puntos.get_ydata()), medias)
    assert len(segmentos) == len(medias), gid
    for x, m, s, segmento in zip(xs, medias, desvios, segmentos):
        assert np.allclose(segmento, [[x, m - s], [x, m + s]]), (gid, segmento, m, s)
    return puntos


def _comprobar_banda(ax, gid, xs, medias, desvios, color):
    """La banda `gid` va de media − desvío a media + desvío en cada x de la curva, y en ningún
    otro x, del color de su serie con alpha ≈ 0,16."""
    bandas = [c for c in ax.collections if c.get_gid() == gid]
    assert len(bandas) == 1, gid
    vertices = np.vstack([p.vertices for p in bandas[0].get_paths()])
    for x, m, s in zip(xs, medias, desvios):
        ys = vertices[np.isclose(vertices[:, 0], x), 1]
        assert len(ys) and np.isclose(ys.min(), m - s) and np.isclose(ys.max(), m + s), \
            (gid, x, ys, m, s)
    assert all(np.isclose(x, xs).any() for x in vertices[:, 0]), f"{gid}: un x fuera de la curva"
    rgba = bandas[0].get_facecolor()[0]
    assert np.allclose(rgba[:3], to_rgba(color)[:3]), gid
    assert abs(rgba[3] - ALFA_BANDA) < 0.02, (gid, rgba[3])


def _comprobar_curva_numerica(ax, nombre, esperado, numericos, x_de):
    """Train y validación de una curva en un eje numérico: la línea une los puntos numéricos en
    orden con su media, la banda es ±1 desvío y el color es el de su conjunto en src/estilo.py."""
    xs = [x_de(p) for p in numericos]
    for conjunto in CONJUNTOS:
        color = estilo.COLOR_CONJUNTO[conjunto]
        linea = _lineas(ax, f"serie:{nombre}:{conjunto}")[f"serie:{nombre}:{conjunto}"]
        medias = [esperado[conjunto][p][0] for p in numericos]
        desvios = [esperado[conjunto][p][1] for p in numericos]
        assert np.allclose(linea.get_xdata(), xs), (nombre, conjunto, list(linea.get_xdata()))
        assert np.allclose(linea.get_ydata(), medias), (nombre, conjunto,
                                                        list(linea.get_ydata()), medias)
        assert _mismo_color(linea.get_color(), color), (nombre, conjunto, linea.get_color())
        _comprobar_banda(ax, f"banda:{nombre}:{conjunto}", xs, medias, desvios, color)


def _pixeles(linea):
    return linea.get_transform().transform(np.asarray(linea.get_xydata(), float))


# --- Formato y rótulos ------------------------------------------------------------------------

def test_formato_de_numeros_y_nombres():
    assert numero(0.7681) == "0,768"
    assert numero(-0.0931, 3, signo=True) == "−0,093"
    assert numero(0.5, 2) == "0,50"
    assert delta(0.2056) == "+0,206" and delta(-0.0765) == "−0,076"
    assert delta(0.0029) == "+0,0029" and delta(-0.0012) == "−0,0012"
    assert nombre_figura_curva("svm", "C") == "03-curva-svm-c.png"
    assert (nombre_figura_curva("knn", "n_neighbors", "uniform")
            == "03-curva-knn-n-neighbors-uniform.png")
    assert nombre_figura_curva("rf", "max_depth", "") == "03-curva-rf-max-depth.png"
    print("ok  coma decimal, signo menos tipográfico y nombres de archivo en kebab-case")


def test_rotulos_de_variantes_y_criterio_de_efecto_coinciden_con_src_ablaciones():
    assert set(ROTULO_VARIANTE) == set(VARIANTES) - {"A0"}, "falta el rótulo corto de una variante"
    for media, desvio in [(0.003, 0.002), (-0.003, 0.002), (0.001, 0.002), (-0.001, 0.002),
                          (0.002, 0.002)]:
        assert efecto(media, desvio) == efecto_ablaciones(media, desvio)
    print("ok  un rótulo corto por variante de src/ablaciones.py, y el mismo criterio de efecto")


def test_leyenda_por_filas():
    # Siete entradas en cuatro columnas: matplotlib llena por columnas, así que se reordenan.
    assert por_filas(list("abcdefg"), 4) == list("aebfcgd")
    assert por_filas(list("abcdef"), 4) == list("aebfcd")
    print("ok  la leyenda se reordena para leerse por filas")


# --- Figura 01 --------------------------------------------------------------------------------

def test_ablaciones_dos_paneles_y_orden_de_variantes():
    tabla = _resumen_ablaciones()
    with _abierta(figura_ablaciones(tabla)) as fig:
        assert len(fig.axes) == 2, "A1 va en un panel propio"
        arriba, abajo = fig.axes
        assert [t.get_text() for t in arriba.get_yticklabels()] == ["A1  con duration (techo)"]
        assert [t.get_text().split()[0] for t in abajo.get_yticklabels()] == \
            [f"A{i}" for i in range(2, 13)]
        for ax in fig.axes:
            assert ax.get_ylim()[0] > ax.get_ylim()[1], "A1 arriba: el eje y va invertido"
        assert fig._suptitle is not None and "ΔAUC" in fig._suptitle.get_text()
        assert "ΔAUC" in abajo.get_xlabel()
    with _abierta(figura_ablaciones(tabla, aparte=())) as fig:
        assert len(fig.axes) == 1 and len(fig.axes[0].get_yticklabels()) == 12
    print("ok  01: A1 en su panel, A2 a A12 debajo en orden numérico, A1 arriba")


def test_ablaciones_punto_barra_color_y_relleno_de_cada_fila():
    tabla = _resumen_ablaciones()
    with _abierta(figura_ablaciones(tabla)) as fig:
        arriba, abajo = fig.axes
        puntos = {**_lineas(arriba, "ablacion:"), **_lineas(abajo, "ablacion:")}
        assert len(puntos) == FILAS_ABLACION
        assert all(g.endswith(":A1") for g in _lineas(arriba, "ablacion:"))
        for f in tabla.itertuples():
            ax = arriba if f.variante == "A1" else abajo
            punto, segmentos, color_barra = _barra(ax, f"ablacion:{f.modelo}:{f.variante}")
            x, y = punto.get_xdata()[0], punto.get_ydata()[0]
            assert _cerca(x, f.delta_media), (f.modelo, f.variante, x)
            # Barra = ±1 desvío de las diferencias pareadas: ni el error estándar ni dos desvíos.
            assert len(segmentos) == 1 and np.allclose(
                segmentos[0], [[f.delta_media - f.delta_desvio, y],
                               [f.delta_media + f.delta_desvio, y]]), \
                (f.modelo, f.variante, segmentos)
            color = _color_modelo(f.modelo)
            assert _mismo_color(punto.get_color(), color), (f.modelo, punto.get_color())
            assert _mismo_color(color_barra, color), (f.modelo, color_barra)
            relleno = color if f.efecto != "neutra" else SUPERFICIE
            assert _mismo_color(punto.get_markerfacecolor(), relleno), \
                (f.modelo, f.variante, f.efecto)
            lo, hi = ax.get_xlim()
            assert lo <= f.delta_media - f.delta_desvio and f.delta_media + f.delta_desvio <= hi, \
                f"{f.modelo} {f.variante} queda fuera del eje"
        # Dentro de una fila, cada modelo a su altura, en el orden nb, svm, knn, rf.
        alturas = [puntos[f"ablacion:{m}:A8"].get_ydata()[0] for m in MODELOS_ABLACION]
        assert alturas == sorted(alturas) and len(set(alturas)) == 4
        for ax in fig.axes:
            referencia = _lineas(ax, "referencia")["referencia"]
            assert list(referencia.get_xdata()) == [0, 0]
            assert any(t.get_text() == "referencia" for t in ax.texts)
    print(f"ok  01: {FILAS_ABLACION} puntos en su ΔAUC medio, barra de ±1 desvío, color del modelo "
          "en src/estilo.py, relleno sólo si mejora o empeora, ninguno recortado")


def test_ablaciones_anota_a1_a7_a8_y_las_que_cambian():
    tabla = _resumen_ablaciones().set_index(["modelo", "variante"])
    with _abierta(figura_ablaciones(tabla.reset_index())) as fig:
        valores = {}
        for ax in fig.axes:
            valores.update(_textos(ax, "valor:"))
        anotadas = {tuple(g.split(":")[1:]) for g in valores}
        assert anotadas == ANOTADAS_ESPERADAS, anotadas ^ ANOTADAS_ESPERADAS
        for (modelo, variante) in anotadas:
            esperado = delta(tabla.loc[(modelo, variante), "delta_media"])
            assert valores[f"valor:{modelo}:{variante}"].get_text() == esperado
    print(f"ok  01: {len(ANOTADAS_ESPERADAS)} valores anotados (A1, A7, A8 y las que mejoran o "
          "empeoran), con el ΔAUC del resumen")


def test_ablaciones_grilla_de_efecto():
    tabla = _resumen_ablaciones()
    with _abierta(figura_ablaciones_efecto(tabla)) as fig:
        ax = fig.axes[0]
        celdas = {p.get_gid(): p for p in ax.patches if (p.get_gid() or "").startswith("celda:")}
        assert len(celdas) == FILAS_ABLACION
        textos = _textos(ax)
        for f in tabla.itertuples():
            celda = celdas[f"celda:{f.modelo}:{f.variante}:{f.efecto}"]
            # mejora azul, empeora naranja, neutra gris: los de src/estilo.py
            assert _mismo_color(celda.get_facecolor(), COLOR_EFECTO[f.efecto]), \
                (f.modelo, f.variante, f.efecto)
            assert textos[f"valor:{f.modelo}:{f.variante}"].get_text() == delta(f.delta_media)
            assert textos[f"desvio:{f.modelo}:{f.variante}"].get_text() == \
                "± " + numero(f.delta_desvio, 4)
        sin_medir = _textos(ax, "sin-medir:")
        assert len(sin_medir) == 12 * 4 - FILAS_ABLACION
        assert all(t.get_text() == "no se mide" for t in sin_medir.values())
        assert _rotulos_x(ax) == [ROTULO_MODELO[m] for m in MODELOS_ABLACION]
        assert [t.get_text().split()[0] for t in ax.get_yticklabels()] == \
            [f"A{i}" for i in range(1, 13)]
    print("ok  01b: una celda por modelo y variante medida, azul, naranja o gris según el efecto, "
          "con su ΔAUC y su desvío")


# --- Figura 02 --------------------------------------------------------------------------------

def test_modelos_ordenados_por_auc_de_validacion_contra_sin_modelo():
    with _abierta(figura_modelos(_resumen_cv())) as fig:
        assert len(fig.axes) == 2
        esperado = [ROTULO_MODELO[m] for m in ORDEN_POR_AUC]
        for ax, metrica, base in zip(fig.axes, ("auc", "recall_q"), (0.5, 0.2001)):
            assert _rotulos_x(ax) == esperado, "mismo orden, de mayor a menor AUC, en los dos"
            assert len(_lineas(ax, "modelo:")) == 5
            for x, modelo in enumerate(ORDEN_POR_AUC):
                media, desvio = CV[modelo][metrica]
                punto = _comprobar_barras_verticales(ax, f"modelo:{modelo}:{metrica}", [media],
                                                     [desvio])
                assert _cerca(punto.get_xdata()[0], x)
                _, _, color_barra = _barra(ax, f"modelo:{modelo}:{metrica}")
                assert _mismo_color(punto.get_color(), _color_modelo(modelo)), modelo
                assert _mismo_color(color_barra, _color_modelo(modelo)), modelo
                texto = _textos(ax, f"valor:{modelo}:{metrica}")[f"valor:{modelo}:{metrica}"]
                assert texto.get_text() == numero(media, 3)
            # Las dos variantes de NB: el color de la familia, con marcadores distintos.
            nb = [_lineas(ax, f"modelo:{m}:")[f"modelo:{m}:{metrica}"]
                  for m in ("nb_gaussiano", "nb_categorico")]
            assert nb[0].get_marker() != nb[1].get_marker()
            linea_base = _lineas(ax, "linea-base")["linea-base"]
            assert all(_cerca(y, base) for y in linea_base.get_ydata())
            assert linea_base.get_linestyle() == ":"
            assert _mismo_color(linea_base.get_color(), COLOR_LINEA_BASE)
            assert [t.get_text() for t in ax.texts if t.get_gid() == "rotulo-linea-base"] == \
                [ROTULO_LINEA_BASE]
            lo, hi = ax.get_ylim()
            assert lo < base < hi
    print("ok  02: de mayor a menor AUC de validación (no de train), barra de ±1 desvío, color del "
          "modelo, y «sin modelo» en 0,5 y 0,2001 como línea punteada gris")


def test_modelos_linea_base_sin_filas_sin_modelo():
    sin = _resumen_cv(con_linea_base=False)
    with _abierta(figura_modelos(sin)) as fig:
        auc, recall = (_lineas(ax, "linea-base")["linea-base"] for ax in fig.axes)
        assert _cerca(auc.get_ydata()[0], 0.5) and _cerca(recall.get_ydata()[0], PRESUPUESTO)
    with _abierta(figura_modelos(sin, linea_base={"auc": 0.5, "recall_q": 0.25})) as fig:
        assert _cerca(_lineas(fig.axes[1], "linea-base")["linea-base"].get_ydata()[0], 0.25)
    repetido = pd.concat([_resumen_cv(), _resumen_cv().assign(configuracion='{"otra": 1}')])
    try:
        figura_modelos(repetido)
    except ValueError:
        pass
    else:
        raise AssertionError("dos configuraciones del mismo modelo deberían ser un error")
    print("ok  02: sin filas sin_modelo usa linea_base.csv o AUC 0,5 y recall q; dos "
          "configuraciones de un modelo son un error")


# --- Figuras 03 -------------------------------------------------------------------------------

def test_eje_x_numerico_logaritmico_y_categorico():
    eje = eje_x("max_depth", ["None", "8", "2", "4", "12"])
    assert eje["tipo"] == "lineal"
    assert eje["puntos"] == ["2", "4", "8", "12", "None"]
    assert eje["rotulos"][-1] == "sin límite" and eje["posiciones"][-1] > 12
    assert eje_x("C", ["0.001", "0.1", "1", "100"])["tipo"] == "log"
    assert eje_x("C", ["0.001", "0.1", "1", "100"])["rotulos"] == ["0,001", "0,1", "1", "100"]
    assert eje_x("n_neighbors", ["1", "5", "25", "101"])["tipo"] == "log"
    categorico = eje_x("pesos_clase", ["None", "balanced"])
    assert categorico["tipo"] == "categorico"
    assert categorico["rotulos"] == ["sin pesos", "balanced"]
    assert eje_x("kernel", ["linear", "poly", "rbf"])["rotulos"] == ["lineal", "polinómico", "RBF"]
    print("ok  eje x: max_depth lineal con None al final, C y n_neighbors logarítmicos, pesos de "
          "clase y kernel categóricos")


def test_puntos_se_reconstruyen_como_en_src_curvas():
    leida = _como_la_lee_pandas(_max_depth())
    assert leida["punto"].isna().any(), "pandas convierte «None» en NaN"
    assert set(puntos_de(leida)) == {"None", "8", "2", "4", "12"}
    print("ok  el «None» que pandas lee como NaN, y el 2.0 de «2», vuelven a su texto desde el "
          "JSON")


def test_curva_max_depth():
    tabla = _como_la_lee_pandas(_max_depth())
    esperado = _esperado(*MAX_DEPTH)
    with _abierta(figura_curva(tabla, "rf_max_depth")) as fig:
        ax = fig.axes[0]
        assert ax.get_xscale() == "linear"
        fig.canvas.draw()
        rotulos = [t.get_text() for t in ax.get_xticklabels()]
        assert rotulos == ["2", "4", "8", "12", "sin límite"], rotulos
        marcas = list(ax.get_xticks())
        assert marcas[-1] == max(marcas), "«sin límite» va en el extremo derecho"
        series = _lineas(ax, "serie:")
        assert set(series) == {"serie:rf_max_depth:train", "serie:rf_max_depth:validacion"}
        assert all(l.get_linestyle() == "-" for l in series.values())
        # La línea continua sólo une los números, en orden y con la media de cada punto; la banda
        # es ±1 desvío. «Sin límite» va aparte, en el extremo, con su media y su barra.
        _comprobar_curva_numerica(ax, "rf_max_depth", esperado, ["2", "4", "8", "12"], int)
        assert set(_lineas(ax, "punto-none:")) == {"punto-none:rf_max_depth:train",
                                                    "punto-none:rf_max_depth:validacion"}
        for conjunto in CONJUNTOS:
            media, desvio = esperado[conjunto]["None"]
            punto = _comprobar_barras_verticales(ax, f"punto-none:rf_max_depth:{conjunto}",
                                                 [media], [desvio])
            assert _cerca(punto.get_xdata()[0], marcas[-1])
            assert _mismo_color(punto.get_color(), estilo.COLOR_CONJUNTO[conjunto])
        maximo = _lineas(ax, "maximo:")["maximo:rf_max_depth"]
        assert maximo.get_xdata()[0] == 8 and _cerca(maximo.get_ydata()[0], 0.797)
        assert _mismo_color(maximo.get_markerfacecolor(), estilo.COLOR_CONJUNTO["validacion"]), \
            "el máximo va relleno, del color de validación"
        brecha = _lineas(ax, "brecha:")["brecha:rf_max_depth"]
        assert list(brecha.get_xdata()) == [8, 8]
        assert np.allclose(sorted(brecha.get_ydata()), [0.797, 0.858])
        assert np.allclose(_pixeles(brecha)[:, 0], _pixeles(maximo)[0, 0])
        textos = _textos(ax, "valor-")
        assert textos["valor-maximo:rf_max_depth"].get_text() == "0,797"
        assert textos["valor-brecha:rf_max_depth"].get_text() == "brecha +0,061"
        assert [t.get_fontweight() for t in ax.get_xticklabels()][2] == "bold", \
            "el rótulo del punto elegido se resalta"
        assert "RF" in ax.get_title(loc="left") and "max_depth" in ax.get_title(loc="left")
        subtitulo = _textos(ax, "subtitulo")["subtitulo"].get_text()
        assert "n_estimators = 300" in subtitulo and "pesos_clase = sin pesos" in subtitulo
        assert ax.get_legend() is not None
        leyenda = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "train" in leyenda and "validación" in leyenda
    print("ok  03 max_depth: eje lineal, «sin límite» al final, train y validación con su media y "
          "banda de ±1 desvío en su color, máximo relleno con su valor y la brecha")


def test_curvas_logaritmicas_para_c_y_n_neighbors():
    c = _curva("svm", "C", [0.001, 0.01, 0.1, 1, 10], [0.70, 0.703, 0.706, 0.702, 0.70],
               [0.75, 0.80, 0.87, 0.90, 0.93], fijos={"kernel": "rbf", "gamma": "scale"})
    with _abierta(figura_curva(_como_la_lee_pandas(c), "svm_C")) as fig:
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert "escala logarítmica" in ax.get_xlabel()
        assert _textos(ax, "valor-maximo:")["valor-maximo:svm_C"].get_text() == "0,706"
    with _abierta(figura_curva(_como_la_lee_pandas(_knn("uniform")),
                               "knn_n_neighbors_uniform")) as fig:
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert "(uniform)" in ax.get_title(loc="left")
        _comprobar_curva_numerica(ax, "knn_n_neighbors_uniform",
                                  _esperado(VECINOS, *KNN["uniform"]), [str(k) for k in VECINOS],
                                  int)
    with _abierta(figura_curva(_como_la_lee_pandas(_max_depth()), "rf_max_depth")) as fig:
        assert fig.axes[0].get_xscale() == "linear"
    print("ok  03: escala logarítmica en C y n_neighbors, lineal en max_depth")


def test_curva_categorica_sin_lineas_ni_bandas():
    valores, validacion, train = [None, "balanced"], [0.702, 0.770], [0.901, 0.876]
    pesos = _curva("svm", "pesos_clase", valores, validacion, train,
                   fijos={"C": 1.0, "kernel": "rbf"})
    esperado = _esperado(valores, validacion, train)
    with _abierta(figura_curva(_como_la_lee_pandas(pesos), "svm_pesos_clase")) as fig:
        ax = fig.axes[0]
        fig.canvas.draw()
        assert [t.get_text() for t in ax.get_xticklabels()] == ["sin pesos", "balanced"]
        series = _lineas(ax, "serie:")
        assert len(series) == 2 and all(l.get_linestyle() == "None" for l in series.values())
        assert not [c for c in ax.collections if (c.get_gid() or "").startswith("banda:")]
        xs = {}
        for conjunto in CONJUNTOS:
            puntos = _comprobar_barras_verticales(
                ax, f"serie:svm_pesos_clase:{conjunto}",
                [esperado[conjunto][p][0] for p in ("None", "balanced")],
                [esperado[conjunto][p][1] for p in ("None", "balanced")])
            xs[conjunto] = np.asarray(puntos.get_xdata(), float)
            assert np.all(np.abs(xs[conjunto] - [0, 1]) < 0.25), "cada punto junto a su categoría"
            assert _mismo_color(puntos.get_color(), estilo.COLOR_CONJUNTO[conjunto])
        assert np.all(xs["train"] < xs["validacion"]), "train a la izquierda de validación"
        maximo = _lineas(ax, "maximo:")["maximo:svm_pesos_clase"]
        assert _cerca(maximo.get_xdata()[0], xs["validacion"][1])
        assert _cerca(maximo.get_ydata()[0], 0.770)
        assert _textos(ax, "valor-")["valor-brecha:svm_pesos_clase"].get_text() == "brecha +0,106"
    print("ok  03 pesos_clase: categórico, puntos con su media y barras de ±1 desvío, sin unir, "
          "con el máximo y su brecha")


def test_knn_uniform_y_distance_en_una_figura():
    tablas = {"knn_n_neighbors_distance": _como_la_lee_pandas(_knn("distance")),
              "knn_n_neighbors_uniform": _como_la_lee_pandas(_knn("uniform"))}
    with _abierta(figura_curvas_comparadas(tablas)) as fig:
        ax = fig.axes[0]
        series = _lineas(ax, "serie:")
        assert len(series) == 4
        estilos = {g: l.get_linestyle() for g, l in series.items()}
        assert estilos["serie:knn_n_neighbors_uniform:train"] == "-"
        assert estilos["serie:knn_n_neighbors_uniform:validacion"] == "-"
        assert estilos["serie:knn_n_neighbors_distance:train"] == "--"
        assert estilos["serie:knn_n_neighbors_distance:validacion"] == "--"
        for weights in ("uniform", "distance"):
            _comprobar_curva_numerica(ax, f"knn_n_neighbors_{weights}",
                                      _esperado(VECINOS, *KNN[weights]),
                                      [str(k) for k in VECINOS], int)
        maximos = _lineas(ax, "maximo:")
        assert len(maximos) == 2
        # uniform: máximo en k = 101 (0,78); distance: en k = 25 (0,75).
        for weights, k, valor in (("uniform", 101, "0,780"), ("distance", 25, "0,750")):
            maximo = maximos[f"maximo:knn_n_neighbors_{weights}"]
            assert maximo.get_xdata()[0] == k
            texto = _textos(ax, "valor-maximo:")[f"valor-maximo:knn_n_neighbors_{weights}"]
            assert texto.get_text() == f"{GLIFO[maximo.get_marker()]} {valor}", texto.get_text()
        leyenda = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "weights = uniform" in leyenda and "weights = distance" in leyenda
        assert "según weights" in ax.get_title(loc="left")
    print("ok  03 KNN: uniform y distance juntas, cada una con su estilo de línea, su media, su "
          "banda y su máximo, rotulado con el glifo de su marcador")


def test_maximos_en_el_mismo_x_no_se_tapan_y_cada_rotulo_dice_su_curva():
    tablas = {nombre: _como_la_lee_pandas(_svm_c(nombre)) for nombre in SVM_C}
    with _abierta(figura_curvas_comparadas(tablas)) as fig:
        ax = fig.axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        maximos = _lineas(ax, "maximo:")
        brechas = _lineas(ax, "brecha:")
        centro = {g.split(":", 1)[1]: _pixeles(l)[0] for g, l in maximos.items()}
        assert set(centro) == set(SVM_C)
        # Los dos máximos de C = 0,01 se corren a los lados de su x, sin taparse; el de C = 0,1,
        # solo en su x, no se corre.
        a, b = centro["svm_C_balanced"], centro["svm_C_linear_balanced"]
        diametro = maximos["maximo:svm_C_balanced"].get_markersize() * fig.dpi / 72
        assert abs(a[0] - b[0]) >= diametro, "un máximo tapa al otro"
        assert np.isclose((a[0] + b[0]) / 2, ax.transData.transform([(0.01, 0.772)])[0, 0])
        assert np.isclose(centro["svm_C"][0], ax.transData.transform([(0.1, 0.706)])[0, 0])
        for nombre, (x, _) in centro.items():
            assert np.allclose(_pixeles(brechas[f"brecha:{nombre}"])[:, 0], x), \
                f"la brecha de {nombre} sale de su máximo"
        # Cada rótulo dice de qué curva es: el glifo del marcador de su máximo, el mismo que la
        # leyenda le da a la curva.
        marcadores = {nombre: maximos[f"maximo:{nombre}"].get_marker() for nombre in SVM_C}
        assert len(set(marcadores.values())) == 3
        leyenda = dict(zip((t.get_text() for t in ax.get_legend().get_texts()),
                           ax.get_legend().legend_handles))
        guias = 0
        for nombre, rotulo, valor in (
                ("svm_C", "kernel = RBF, pesos_clase = sin pesos", "0,706"),
                ("svm_C_balanced", "kernel = RBF, pesos_clase = balanced", "0,772"),
                ("svm_C_linear_balanced", "kernel = lineal, pesos_clase = balanced", "0,774")):
            assert leyenda[rotulo].get_marker() == marcadores[nombre], rotulo
            texto = _textos(ax, "valor-maximo:")[f"valor-maximo:{nombre}"]
            assert texto.get_text() == f"{GLIFO[marcadores[nombre]]} {valor}", texto.get_text()
            # El bloque se ancla en el máximo, ya corrido. Si el ancla queda a 20 puntos o más
            # del punto, una línea guía los une; con o sin ella, los dos renglones del bloque
            # siguen alineados entre sí.
            rotulos = [texto, _textos(ax, "valor-brecha:")[f"valor-brecha:{nombre}"]]
            cajas = []
            for t in rotulos:
                t.get_window_extent(renderer)
                cajas.append(Text.get_window_extent(t, renderer))
            anclado = [t for t in rotulos if isinstance(t.xycoords, Transform)]
            assert anclado, f"{nombre}: ningún renglón se ancla en el punto"
            for t in anclado:
                assert np.allclose(t.xycoords.transform([t.xy])[0], centro[nombre]), nombre
            lejos = np.hypot(*anclado[0].xyann) >= 20
            assert any(t.arrow_patch is not None for t in anclado) == lejos, nombre
            assert all(t.arrow_patch is None for t in rotulos if t not in anclado)
            guias += lejos
            ha = texto.get_horizontalalignment()
            borde = {"left": lambda c: c.x0, "right": lambda c: c.x1,
                     "center": lambda c: (c.x0 + c.x1) / 2}[ha]
            assert abs(borde(cajas[0]) - borde(cajas[1])) < 1.5, \
                f"{nombre}: los dos renglones del rótulo no quedan alineados"
            x, y = centro[nombre]
            caja = Bbox.union(cajas)
            separacion = np.hypot(max(caja.x0 - x, 0, x - caja.x1),
                                  max(caja.y0 - y, 0, y - caja.y1))
            assert lejos or separacion < 20 * fig.dpi / 72, f"{nombre}: rótulo lejano sin guía"
        assert guias, "ningún rótulo necesitó guía: la figura ya no prueba ese caso"
        subtitulo = _textos(ax, "subtitulo")["subtitulo"].get_text()
        assert "gamma = scale (no aplica con kernel lineal)" in subtitulo, subtitulo
    print("ok  03 SVM: dos máximos en el mismo C se corren sin taparse, con su brecha; cada "
          "rótulo lleva el glifo de su curva, y uno lejano, una línea guía")


def test_subtitulo_nombra_los_fijos_que_valen():
    def subtitulo(tabla, nombre):
        with _abierta(figura_curva(_como_la_lee_pandas(tabla), nombre)) as fig:
            return _textos(fig.axes[0], "subtitulo")["subtitulo"].get_text()

    # rf_n_estimators no trae max_depth en el JSON: es el de scikit-learn, sin límite.
    arboles = _curva("rf", "n_estimators", [10, 100, 400], [0.74, 0.77, 0.772], [0.999, 1.0, 1.0])
    texto = subtitulo(arboles, "rf_n_estimators")
    assert "max_depth = sin límite" in texto and "pesos_clase = sin pesos" in texto, texto
    # Con kernel lineal, gamma no hace nada: no se nombra entre los fijos.
    validacion, train, fijos = SVM_C["svm_C_linear_balanced"]
    texto = subtitulo(_curva("svm", "C", VALORES_C, validacion, train, fijos=fijos),
                      "svm_C_linear_balanced")
    assert "gamma" not in texto and "kernel = lineal" in texto, texto
    # En la curva de kernel, gamma vale para poly y RBF pero no para el lineal: se aclara.
    kernel = _curva("svm", "kernel", ["linear", "poly", "rbf"], [0.774, 0.773, 0.768],
                    [0.776, 0.782, 0.770], fijos={"C": 0.001, "gamma": "scale",
                                                  "pesos_clase": "balanced"})
    texto = subtitulo(kernel, "svm_kernel")
    assert "gamma = scale (no aplica con kernel lineal)" in texto and "C = 0,001" in texto, texto
    print("ok  03: el subtítulo nombra max_depth sin límite cuando el JSON no lo trae, y gamma "
          "sólo donde el kernel lo usa")


def test_curva_con_puntos_omitidos():
    c = _curva("svm", "C", [0.001, 0.1, 1], [0.70, 0.706, 0.702], [0.75, 0.87, 0.90])
    with _abierta(figura_curva(c, "svm_C", omitidos=["C=100"])) as fig:
        nota = _textos(fig.axes[0], "omitidos")["omitidos"].get_text().replace("\n", " ")
        assert "C=100" in nota and "omitidos.txt" in nota
    print("ok  03: los puntos que src/curvas.py omitió se listan en la figura")


# --- CLI --------------------------------------------------------------------------------------

def test_cli_escribe_los_png_y_no_falla_si_falta_un_csv():
    with tempfile.TemporaryDirectory() as tmp:
        resultados, figuras = Path(tmp) / "resultados", Path(tmp) / "figuras"
        curvas = resultados / "curvas"
        curvas.mkdir(parents=True)
        _resumen_ablaciones().to_csv(resultados / "ablaciones_resumen.csv", index=False)
        _max_depth().to_csv(curvas / "rf_max_depth.csv", index=False)
        _knn("uniform").to_csv(curvas / "knn_n_neighbors_uniform.csv", index=False)
        _knn("distance").to_csv(curvas / "knn_n_neighbors_distance.csv", index=False)
        # Una curva sin sufijo y otra del mismo eje con sufijo: la combinada no sobrescribe a
        # ninguna de las dos.
        _curva("rf", "n_estimators", [10, 100, 400], [0.74, 0.77, 0.772], [0.999, 1.0, 1.0]).to_csv(
            curvas / "rf_n_estimators.csv", index=False)
        _curva("rf", "n_estimators", [10, 100, 400], [0.79, 0.795, 0.796], [0.817, 0.824, 0.826],
               fijos={"max_depth": 8}).to_csv(curvas / "rf_n_estimators_depth8.csv", index=False)
        (curvas / "omitidos.txt").write_text("rf_max_depth\tmax_depth=25\tTimeoutError: límite\n",
                                             encoding="utf-8")
        pd.DataFrame({"a": [1]}).to_csv(curvas / "notas.csv", index=False)  # no es una curva

        codigo, salida = _en_silencio(cli, ["--resultados", str(resultados),
                                            "--figuras", str(figuras)])
        assert codigo == 0, salida
        esperadas = {FIGURA_ABLACIONES, FIGURA_EFECTO, "03-curva-rf-max-depth.png",
                     "03-curva-knn-n-neighbors-uniform.png",
                     "03-curva-knn-n-neighbors-distance.png",
                     "03-curva-knn-n-neighbors-por-weights.png",
                     "03-curva-rf-n-estimators.png", "03-curva-rf-n-estimators-depth8.png",
                     "03-curva-rf-n-estimators-por-max-depth.png"}
        escritas = {p.name for p in figuras.glob("*.png")}
        assert escritas == esperadas, escritas ^ esperadas
        assert FIGURA_MODELOS not in escritas
        assert "cv_referencia_resumen.csv" in salida, "informa que falta el resumen de la CV"
        assert "Se ignora notas.csv" in salida
        assert all(f"Escrita {p.name}" in salida or p.name in salida for p in figuras.glob("*.png"))
        for ruta in figuras.glob("*.png"):
            with Image.open(ruta) as imagen:
                ancho, alto = imagen.size
                assert ancho > 600 and alto > 300
                assert round(imagen.info["dpi"][0]) == 150

        # Cuando llega el resumen de la CV, --solo modelos dibuja la 02 y nada más.
        _resumen_cv().to_csv(resultados / "cv_referencia_resumen.csv", index=False)
        antes = {p: p.stat().st_mtime_ns for p in figuras.glob("*.png")}
        codigo, salida = _en_silencio(cli, ["--solo", "modelos", "--resultados", str(resultados),
                                            "--figuras", str(figuras)])
        assert codigo == 0 and (figuras / FIGURA_MODELOS).exists()
        assert all(p.stat().st_mtime_ns == t for p, t in antes.items())

        # Sin nada en resultados/: informa y termina bien, sin escribir.
        vacio = Path(tmp) / "vacio"
        codigo, salida = _en_silencio(cli, ["--resultados", str(vacio),
                                            "--figuras", str(Path(tmp) / "otras")])
        assert codigo == 0 and not (Path(tmp) / "otras").exists()
        assert "Falta" in salida and "No hay curvas" in salida
    print(f"ok  CLI: {len(esperadas)} PNG a 150 dpi, la 02 omitida con aviso hasta que llega su "
          "CSV, un CSV ajeno ignorado y sin error si falta todo")


def test_cli_informa_una_curva_que_no_se_puede_dibujar_y_sigue():
    with tempfile.TemporaryDirectory() as tmp:
        resultados, figuras = Path(tmp) / "resultados", Path(tmp) / "figuras"
        curvas = resultados / "curvas"
        curvas.mkdir(parents=True)
        _max_depth().to_csv(curvas / "rf_max_depth.csv", index=False)
        sin_auc = _curva("svm", "C", [0.1, 1], [0.7, 0.7], [0.8, 0.9])
        sin_auc[sin_auc["metrica"] != "auc"].to_csv(curvas / "svm_C.csv", index=False)
        codigo, salida = _en_silencio(cli, ["--solo", "curvas", "--resultados", str(resultados),
                                            "--figuras", str(figuras)])
        assert codigo == 1, "un CSV que no se puede dibujar termina con código 1"
        assert "No se pudo dibujar svm_C.csv" in salida
        assert (figuras / "03-curva-rf-max-depth.png").exists(), "las demás se dibujan igual"
    print("ok  CLI: una curva que no se puede dibujar se informa, las demás se escriben y el "
          "código es 1")


def test_cli_un_csv_vacio_o_cortado_se_informa_y_no_frena_lo_demas():
    with tempfile.TemporaryDirectory() as tmp:
        resultados, figuras = Path(tmp) / "resultados", Path(tmp) / "figuras"
        curvas = resultados / "curvas"
        curvas.mkdir(parents=True)
        argumentos = ["--resultados", str(resultados), "--figuras", str(figuras)]
        texto = _resumen_cv().to_csv(index=False)
        corte = texto.index("\n", len(texto) // 2) - 7  # a mitad de una fila, como si otro
        assert texto[corte - 1] != "\n"                  # proceso lo estuviera escribiendo
        (resultados / "ablaciones_resumen.csv").write_bytes(b"")
        (resultados / "cv_referencia_resumen.csv").write_text(texto[:corte], encoding="utf-8")
        (resultados / "linea_base.csv").write_bytes(b"")
        (curvas / "svm_C.csv").write_bytes(b"")
        _max_depth().to_csv(curvas / "rf_max_depth.csv", index=False)

        codigo, salida = _en_silencio(cli, argumentos)
        assert codigo == 1, salida
        assert _renglon(salida, "No se pudo leer", "ablaciones_resumen.csv: está vacío"), salida
        assert _renglon(salida, "No se pudo leer",
                        "cv_referencia_resumen.csv: la última fila está cortada"), salida
        assert _renglon(salida, "No se pudo leer", "svm_C.csv: está vacío"), salida
        escritas = {p.name for p in figuras.glob("*.png")}
        assert escritas == {"03-curva-rf-max-depth.png"}, escritas
        # Una curva vacía es un error, no un CSV ajeno que se ignora: sola, también da código 1.
        codigo, salida = _en_silencio(cli, ["--solo", "curvas", *argumentos])
        assert codigo == 1 and not _renglon(salida, "Se ignora", "svm_C.csv"), salida

        # Con el resumen completo, la 02 sale aunque linea_base.csv esté vacío: la línea de base
        # viene de las filas sin_modelo. El archivo ilegible se informa igual, con código 1.
        (resultados / "cv_referencia_resumen.csv").write_text(texto, encoding="utf-8")
        codigo, salida = _en_silencio(cli, ["--solo", "modelos", *argumentos])
        assert codigo == 1 and (figuras / FIGURA_MODELOS).exists(), salida
        assert "linea_base.csv: está vacío" in salida

        # Un resumen al que le faltan modelos se dibuja con un aviso.
        (resultados / "linea_base.csv").unlink()
        parcial = _resumen_cv()
        parcial[~parcial["modelo"].isin(["svm", "knn"])].to_csv(
            resultados / "cv_referencia_resumen.csv", index=False)
        codigo, salida = _en_silencio(cli, ["--solo", "modelos", *argumentos])
        assert codigo == 0 and "Aviso" in salida and "svm, knn" in salida, salida
    print("ok  CLI: un CSV vacío o cortado a mitad de una fila se informa, el código es 1 y las "
          "demás figuras se escriben; un resumen sin algún modelo, con aviso")


def main():
    test_formato_de_numeros_y_nombres()
    test_rotulos_de_variantes_y_criterio_de_efecto_coinciden_con_src_ablaciones()
    test_leyenda_por_filas()
    test_ablaciones_dos_paneles_y_orden_de_variantes()
    test_ablaciones_punto_barra_color_y_relleno_de_cada_fila()
    test_ablaciones_anota_a1_a7_a8_y_las_que_cambian()
    test_ablaciones_grilla_de_efecto()
    test_modelos_ordenados_por_auc_de_validacion_contra_sin_modelo()
    test_modelos_linea_base_sin_filas_sin_modelo()
    test_eje_x_numerico_logaritmico_y_categorico()
    test_puntos_se_reconstruyen_como_en_src_curvas()
    test_curva_max_depth()
    test_curvas_logaritmicas_para_c_y_n_neighbors()
    test_curva_categorica_sin_lineas_ni_bandas()
    test_knn_uniform_y_distance_en_una_figura()
    test_maximos_en_el_mismo_x_no_se_tapan_y_cada_rotulo_dice_su_curva()
    test_subtitulo_nombra_los_fijos_que_valen()
    test_curva_con_puntos_omitidos()
    test_cli_escribe_los_png_y_no_falla_si_falta_un_csv()
    test_cli_informa_una_curva_que_no_se_puede_dibujar_y_sigue()
    test_cli_un_csv_vacio_o_cortado_se_informa_y_no_frena_lo_demas()
    assert not plt.get_fignums(), f"quedaron figuras abiertas: {plt.get_fignums()}"
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
