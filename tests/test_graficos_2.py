"""Correr con: python -m tests.test_graficos_2

Las figuras 04 a 07 de src/graficos.py (robustez temporal, curva de ganancia y sensibilidad a q),
la 02 sobre la etiqueta final y el CLI de los grupos nuevos. Como en tests/test_graficos.py, todo
con DataFrames sintéticos en el formato de las entradas (las columnas salen de src/robustez.py y
de src/experimentos.py, que escriben esos CSV): ningún CSV de resultados/ y ningún modelo
ajustado. Lo dibujado se compara con valores que salen de los datos sintéticos, y los colores, con
los de src/estilo.py; las series se buscan por su `gid`.

Cada figura se prueba además contra solapes: ningún texto (valores, rótulos, marcas de los ejes,
títulos, leyendas) tapa a otro, y ningún valor anotado tapa un marcador.
"""

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.text import Text
from PIL import Image

from src import estilo
from src import robustez as modulo_robustez
from src.configuracion import MODELOS_FINALES
from src.estilo import COLOR_LINEA_BASE, ROTULO_LINEA_BASE
from src.experimentos import CAMPOS_GANANCIA, CAMPOS_SENSIBILIDAD, Q_SENSIBILIDAD
from src.graficos import (
    ESQUEMAS_ROBUSTEZ,
    FIGURA_GANANCIA,
    FIGURA_MODELOS,
    FIGURA_MODELOS_REFERENCIA,
    FIGURA_ROBUSTEZ,
    FIGURA_ROBUSTEZ_FOLDS,
    FIGURA_SENSIBILIDAD,
    MODELOS_DE_ETIQUETA,
    VARIANTES_ROBUSTEZ,
    figura_ganancia,
    figura_modelos,
    figura_robustez,
    figura_robustez_folds,
    figura_sensibilidad,
    miles,
    porcentaje,
    rotulo_bloque,
)
from src.graficos import main as cli
from src.metricas import METRICAS
from src.resultados import CAMPOS, configuracion_json
from tests.test_graficos import (
    DESPLAZAMIENTOS,
    RAIZ,
    _abierta,
    _barra,
    _cerca,
    _comprobar_barras_verticales,
    _como_la_lee_pandas,
    _en_silencio,
    _lineas,
    _mismo_color,
    _renglon,
    _resumen_cv,
    _rotulos_x,
    _textos,
)

MODELOS = tuple(MODELOS_FINALES)  # nb_categorico, svm, knn, rf: el orden de las figuras
ROTULOS = ["NB categórico", "SVM", "KNN", "RF"]
FINO = "\N{NARROW NO-BREAK SPACE}"  # el espacio de miles


def _coma(valor, decimales=3):
    """El número como tiene que leerse en la figura, armado aquí y no con src/graficos.py."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def _con_miles(entero):
    """Un entero con el espacio fino de miles, armado aquí y no con src/graficos.py."""
    return f"{entero:,}".replace(",", FINO)


# --- Datos sintéticos -------------------------------------------------------------------------

# Figura 04: (media, desvío) del AUC de validación por (modelo, variante) y esquema. KNN también
# tiene una variante: la figura sólo dibuja las de RF.
ROBUSTEZ = {
    ("nb_categorico", "todas"): {"barajado": (0.781, 0.007), "hacia_adelante": (0.601, 0.086)},
    ("svm", "todas"): {"barajado": (0.772, 0.009), "hacia_adelante": (0.542, 0.078)},
    ("knn", "todas"): {"barajado": (0.786, 0.008), "hacia_adelante": (0.537, 0.065)},
    ("rf", "todas"): {"barajado": (0.793, 0.006), "hacia_adelante": (0.561, 0.121)},
    ("rf", "sin_macro"): {"barajado": (0.771, 0.010), "hacia_adelante": (0.553, 0.110)},
    ("rf", "sin_macro_ni_month"): {"barajado": (0.737, 0.009), "hacia_adelante": (0.549, 0.098)},
    ("knn", "sin_macro"): {"barajado": (0.752, 0.009), "hacia_adelante": (0.557, 0.099)},
}
VARIANTES_RF = ["todas", "sin_macro", "sin_macro_ni_month"]


def _resumen_robustez():
    """Como robustez_temporal_resumen.csv: AUC y recall_q por modelo, esquema y variante, más las
    filas de la diferencia entre esquemas (sin desvío), en otro orden que el de la figura."""
    filas = []
    for (modelo, variante), por_esquema in ROBUSTEZ.items():
        for esquema, (media, desvio) in por_esquema.items():
            filas.append({"modelo": modelo, "esquema": esquema, "variante": variante,
                          "metrica": "auc", "media": media, "desvio": desvio, "n": 5,
                          "error_estandar": desvio / np.sqrt(5)})
            filas.append({"modelo": modelo, "esquema": esquema, "variante": variante,
                          "metrica": "recall_q", "media": 0.3, "desvio": 0.2, "n": 5,
                          "error_estandar": 0.2 / np.sqrt(5)})
        filas.append({"modelo": modelo, "esquema": "barajado_menos_hacia_adelante",
                      "variante": variante, "metrica": "auc",
                      "media": por_esquema["barajado"][0] - por_esquema["hacia_adelante"][0],
                      "desvio": np.nan, "n": np.nan, "error_estandar": np.nan})
    tabla = pd.DataFrame(filas)[modulo_robustez.CAMPOS_RESUMEN]
    return _como_la_lee_pandas(tabla.iloc[::-1].reset_index(drop=True))


# Figura 05: el AUC de validación de RF por fold, y los bloques de robustez_folds.csv.
ADELANTE_RF = [0.472, 0.515, 0.431, 0.652, 0.738]
BARAJADO_RF = [0.781, 0.797, 0.792, 0.788, 0.802]  # media 0,792
# fold: (fila_min_validacion, fila_max_validacion, pct_yes_validacion, n_train, n_validacion)
BLOQUES = {1: (5000, 9999, 3.21, 4000, 5000), 2: (10000, 14999, 7.46, 9000, 5000),
           3: (15000, 19999, 5.58, 14000, 5000), 4: (20000, 24999, 14.04, 19000, 5000),
           5: (25000, 29999, 31.96, 24000, 5000)}


def _largo_robustez():
    """Como robustez_temporal.csv: esquema y variante delante de las seis columnas del contrato.
    Train vale otra cosa, KNN también está y la variante sin_macro está corrida, para que la
    figura no tome filas de otro conjunto, de otro modelo o de otra variante."""
    filas = []
    for modelo, adelante, barajado in (("rf", ADELANTE_RF, BARAJADO_RF),
                                       ("knn", [0.50] * 5, [0.78] * 5)):
        configuracion = configuracion_json({"modelo": modelo})
        for esquema, valores in (("hacia_adelante", adelante), ("barajado", barajado)):
            for variante, corrimiento in (("todas", 0.0), ("sin_macro", -0.02)):
                for fold, auc in enumerate(valores, start=1):
                    for conjunto in ("train", "validacion"):
                        for metrica in METRICAS:
                            valor = 0.4
                            if metrica == "auc":
                                valor = auc + corrimiento if conjunto == "validacion" else 0.93
                            filas.append({"esquema": esquema, "variante": variante,
                                          "modelo": modelo, "configuracion": configuracion,
                                          "fold": fold, "conjunto": conjunto,
                                          "metrica": metrica, "valor": valor})
    return _como_la_lee_pandas(pd.DataFrame(filas)[["esquema", "variante"] + CAMPOS])


def _folds_robustez():
    """Como robustez_folds.csv: una fila por esquema y fold."""
    filas = []
    for fold, (desde, hasta, pct, n_train, n_validacion) in BLOQUES.items():
        filas.append({"esquema": "hacia_adelante", "fold": fold, "n_train": n_train,
                      "n_validacion": n_validacion, "pct_yes_train": 3.0,
                      "pct_yes_validacion": pct, "fila_min_validacion": desde,
                      "fila_max_validacion": hasta, "fila_max_train": desde - 1})
    for fold in range(1, 6):
        filas.append({"esquema": "barajado", "fold": fold, "n_train": 24000,
                      "n_validacion": 6000, "pct_yes_train": 11.27, "pct_yes_validacion": 11.27,
                      "fila_min_validacion": 0, "fila_max_validacion": 29999,
                      "fila_max_train": 29999})
    return _como_la_lee_pandas(pd.DataFrame(filas)[modulo_robustez.CAMPOS_FOLDS])


# Figura 06: curvas de ganancia 100·(1 − (1 − p)^k), un k por modelo, y la diagonal.
EXPONENTE = {"nb_categorico": 3.6, "svm": 3.4, "knn": 3.8, "rf": 4.3}
P = np.arange(1, 101)


def _alcanzado(modelo, p):
    return 100 * (1 - (1 - np.asarray(p, float) / 100) ** EXPONENTE[modelo])


def _ganancia(con_diagonal=True):
    """Como ganancia_final.csv: modelo, p de 1 a 100 y pct_yes_alcanzado, con RF primero."""
    filas = [{"modelo": m, "p": int(p), "pct_yes_alcanzado": float(_alcanzado(m, p))}
             for m in ("rf", "knn", "nb_categorico", "svm") for p in P]
    if con_diagonal:
        filas += [{"modelo": "sin_modelo", "p": int(p), "pct_yes_alcanzado": float(p)}
                  for p in P]
    return _como_la_lee_pandas(pd.DataFrame(filas)[CAMPOS_GANANCIA])


# Figura 07: la media de cada modelo por métrica y q (10, 20 y 30 %). Los cinco folds valen
# media + d·(−2, …, 2), con un d distinto en cada punto: el desvío es d·√(10/4).
SENSIBILIDAD = {
    "recall_q": {"nb_categorico": (0.408, 0.611, 0.699), "svm": (0.418, 0.612, 0.693),
                 "knn": (0.412, 0.619, 0.688), "rf": (0.449, 0.633, 0.707)},
    "precision_q": {"nb_categorico": (0.461, 0.344, 0.263), "svm": (0.471, 0.345, 0.260),
                    "knn": (0.465, 0.349, 0.258), "rf": (0.505, 0.357, 0.266)},
}
TASA_YES = 0.1127


def _paso_q(j, i):
    """El paso de los folds del modelo j en el i-ésimo q."""
    return 0.002 + 0.001 * i + 0.0005 * j


def _sensibilidad():
    """Como sensibilidad_q_final.csv: modelo, q, fold, metrica y valor, con sin_modelo."""
    filas = []
    for metrica, por_modelo in SENSIBILIDAD.items():
        for j, (modelo, medias) in enumerate(por_modelo.items()):
            for i, (q, media) in enumerate(zip(Q_SENSIBILIDAD, medias)):
                for fold, k in enumerate(DESPLAZAMIENTOS, start=1):
                    filas.append({"modelo": modelo, "q": q, "fold": fold, "metrica": metrica,
                                  "valor": media + _paso_q(j, i) * k})
    for q in Q_SENSIBILIDAD:
        for fold in range(1, 6):
            for metrica, valor in (("recall_q", q), ("precision_q", TASA_YES), ("f1_q", 0.15)):
                filas.append({"modelo": "sin_modelo", "q": q, "fold": fold, "metrica": metrica,
                              "valor": valor})
    for modelo in MODELOS:  # f1_q también está en el CSV, y la figura no la dibuja
        for q in Q_SENSIBILIDAD:
            filas += [{"modelo": modelo, "q": q, "fold": f, "metrica": "f1_q", "valor": 0.4}
                      for f in range(1, 6)]
    return _como_la_lee_pandas(pd.DataFrame(filas)[CAMPOS_SENSIBILIDAD])


# Figura 02 final: los cuatro modelos, en otro orden de AUC que la referencia sintética.
CV_FINAL = {"nb_categorico": (0.781, 0.612), "svm": (0.771, 0.611), "knn": (0.786, 0.618),
            "rf": (0.793, 0.628)}
ORDEN_FINAL = ["rf", "knn", "nb_categorico", "svm"]


def _resumen_cv_final():
    """Como cv_final_resumen.csv: la etiqueta final, sin NB gaussiano."""
    filas = []
    for modelo, (auc, recall) in CV_FINAL.items():
        for metrica in METRICAS:
            valor = {"auc": auc, "recall_q": recall}.get(metrica, 0.35)
            for conjunto, media in (("validacion", valor), ("train", valor + 0.02)):
                filas.append({"etiqueta": "final", "modelo": modelo, "configuracion": "{}",
                              "conjunto": conjunto, "metrica": metrica, "media": media,
                              "desvio": 0.008, "n": 5, "error_estandar": 0.008 / np.sqrt(5)})
    for metrica, valor in (("auc", 0.5), ("ap", 0.1127), ("recall_q", 0.2001),
                           ("precision_q", 0.1127), ("f1_q", 0.1441)):
        filas.append({"etiqueta": "final", "modelo": "sin_modelo", "configuracion": "{}",
                      "conjunto": "validacion", "metrica": metrica, "media": valor,
                      "desvio": 0.0, "n": 5, "error_estandar": 0.0})
    return pd.DataFrame(filas)


# --- Utilidades -------------------------------------------------------------------------------

def _contenedores(ax, prefijo):
    """{gid: ErrorbarContainer} de las series con barra de error cuyo gid empieza con `prefijo`."""
    return {c.lines[0].get_gid(): c for c in ax.containers
            if isinstance(c, ErrorbarContainer)
            and (c.lines[0].get_gid() or "").startswith(prefijo)}


def _textos_visibles(fig):
    """(nombre, caja en píxeles) de cada texto visible: los de los ejes, las marcas, los rótulos
    de los ejes, los títulos, el título general y las leyendas."""
    renderer = fig.canvas.get_renderer()
    cajas = []
    for i, ax in enumerate(fig.axes):
        for t in ax.texts:
            if t.get_visible() and t.get_text():
                t.get_window_extent(renderer)  # fija la posición de una anotación
                cajas.append((f"eje {i}, {t.get_gid()}: {t.get_text()!r}",
                              Text.get_window_extent(t, renderer)))
        otros = [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label,
                 ax.title, ax._left_title, ax._right_title]
        cajas += [(f"eje {i}: {t.get_text()!r}", t.get_window_extent(renderer)) for t in otros
                  if t.get_visible() and t.get_text()]
        if ax.get_legend() is not None:
            cajas.append((f"eje {i}: leyenda", ax.get_legend().get_window_extent(renderer)))
    if fig._suptitle is not None and fig._suptitle.get_text():
        cajas.append(("título general", fig._suptitle.get_window_extent(renderer)))
    cajas += [(f"leyenda {j}", l.get_window_extent(renderer)) for j, l in enumerate(fig.legends)]
    return cajas


def _se_tocan(a, b, holgura=0.5):
    return (a.x0 + holgura < b.x1 and b.x0 + holgura < a.x1
            and a.y0 + holgura < b.y1 and b.y0 + holgura < a.y1)


def _comprobar_sin_solapes(fig, ax_leyenda=None):
    """Ningún texto visible tapa a otro, y ningún valor anotado (gid «valor…») tapa un marcador de
    los datos. Una leyenda dentro de los ejes puede tapar los datos, no un texto."""
    fig.canvas.draw()
    cajas = _textos_visibles(fig)
    solapes = [(a, b) for i, (a, ca) in enumerate(cajas) for b, cb in cajas[i + 1:]
               if _se_tocan(ca, cb)]
    assert not solapes, "textos que se tapan:\n" + "\n".join(f"{a}  <->  {b}" for a, b in solapes)
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        valores = [t for t in ax.texts if (t.get_gid() or "").startswith("valor")]
        marcadores = []
        for linea in ax.lines:
            if linea.get_marker() in (None, "None", "none", "", " ", "_"):
                continue
            radio = renderer.points_to_pixels(linea.get_markersize()) / 2
            xy = linea.get_transform().transform(np.asarray(linea.get_xydata(), float))
            marcadores += [(linea.get_gid(), x, y, radio) for x, y in xy
                           if np.isfinite(x) and np.isfinite(y)]
        for t in valores:
            t.get_window_extent(renderer)
            caja = Text.get_window_extent(t, renderer)
            tapados = [g for g, x, y, r in marcadores
                       if caja.x0 < x + r and x - r < caja.x1
                       and caja.y0 < y + r and y - r < caja.y1]
            assert not tapados, f"{t.get_gid()} ({t.get_text()!r}) tapa marcadores de {tapados}"


def _dentro_de_los_ejes(ax, prefijo):
    """Los textos con gid que empieza con `prefijo` quedan dentro del marco de los ejes."""
    renderer = ax.figure.canvas.get_renderer()
    marco = ax.get_window_extent(renderer)
    for t in ax.texts:
        if (t.get_gid() or "").startswith(prefijo):
            t.get_window_extent(renderer)
            caja = Text.get_window_extent(t, renderer)
            assert (marco.x0 - 1 <= caja.x0 and caja.x1 <= marco.x1 + 1
                    and marco.y0 - 1 <= caja.y0 and caja.y1 <= marco.y1 + 1), \
                f"{t.get_gid()} ({t.get_text()!r}) sale de los ejes"


def _linea_base(ax, y):
    """La línea «sin modelo» de un panel: horizontal en y, punteada y del gris de src/estilo.py,
    con su rótulo."""
    linea = _lineas(ax, "linea-base")["linea-base"]
    assert np.allclose(linea.get_ydata(), y), (list(linea.get_ydata()), y)
    assert linea.get_linestyle() == ":"
    assert _mismo_color(linea.get_color(), COLOR_LINEA_BASE)
    rotulos = [t.get_text() for t in ax.texts if t.get_gid() == "rotulo-linea-base"]
    assert len(rotulos) == 1 and rotulos[0].startswith(ROTULO_LINEA_BASE), rotulos
    return rotulos[0]


# --- Nombres y formato ------------------------------------------------------------------------

def test_constantes_coinciden_con_src_robustez_y_src_configuracion():
    assert set(ESQUEMAS_ROBUSTEZ) == set(modulo_robustez.ESQUEMAS)
    assert tuple(VARIANTES_ROBUSTEZ) == tuple(modulo_robustez.VARIANTES)
    assert tuple(MODELOS_DE_ETIQUETA["final"]) == tuple(MODELOS_FINALES)
    assert len(MODELOS_DE_ETIQUETA["referencia"]) == 5
    nombres = [FIGURA_MODELOS, FIGURA_MODELOS_REFERENCIA, FIGURA_ROBUSTEZ, FIGURA_ROBUSTEZ_FOLDS,
               FIGURA_GANANCIA, FIGURA_SENSIBILIDAD]
    assert nombres == ["02-modelos-vs-linea-base.png", "02b-modelos-referencia.png",
                       "04-robustez-temporal.png", "05-robustez-folds-rf.png", "06-ganancia.png",
                       "07-sensibilidad-q.png"]
    print("ok  esquemas y variantes de src/robustez.py, modelos finales de src/configuracion.py y "
          "los seis nombres de archivo")


def test_formato_de_miles_porcentajes_y_bloques():
    assert miles(13731) == f"13{FINO}731" and miles(928) == "928" and miles(5490.0) == f"5{FINO}490"
    assert porcentaje(62.759) == "62,8 %" and porcentaje(20, 0) == "20 %"
    fila = pd.Series({"fila_min_validacion": 6882, "fila_max_validacion": 13731,
                      "pct_yes_validacion": 4.7358})
    assert rotulo_bloque(1, fila) == f"bloque 1\nfilas 6{FINO}882–13{FINO}731\n4,7 % de «yes»"
    assert rotulo_bloque(3) == "bloque 3"
    print("ok  espacio fino de miles, porcentajes con coma y el rótulo de cada bloque")


# --- Figura 02 sobre la etiqueta final --------------------------------------------------------

def test_modelos_final_cuatro_modelos_y_referencia_con_su_etiqueta():
    with _abierta(figura_modelos(_resumen_cv_final())) as fig:
        for ax, metrica in zip(fig.axes, ("auc", "recall_q")):
            assert _rotulos_x(ax) == [ROTULOS[MODELOS.index(m)] for m in ORDEN_FINAL]
            assert len(_contenedores(ax, "modelo:")) == 4
            for modelo in ORDEN_FINAL:
                media = CV_FINAL[modelo][0 if metrica == "auc" else 1]
                _comprobar_barras_verticales(ax, f"modelo:{modelo}:{metrica}", [media], [0.008])
        assert "hiperparámetros elegidos" in fig._suptitle.get_text()
        _comprobar_sin_solapes(fig)
    with _abierta(figura_modelos(_resumen_cv())) as fig:
        assert "configuración de referencia" in fig._suptitle.get_text()
        assert len(_contenedores(fig.axes[0], "modelo:")) == 5
    print("ok  02: los cuatro modelos finales de mayor a menor AUC, con «hiperparámetros elegidos»;"
          " la 02b, cinco con «configuración de referencia»")


# --- Figura 04 --------------------------------------------------------------------------------

def test_robustez_pares_por_modelo_y_variantes_de_rf():
    with _abierta(figura_robustez(_resumen_robustez())) as fig:
        assert len(fig.axes) == 2
        izquierda, derecha = fig.axes
        assert _rotulos_x(izquierda) == ROTULOS, "el orden: NB categórico, SVM, KNN, RF"
        assert _rotulos_x(derecha) == ["todas", "sin macro", "sin macro ni month"]
        for ax, prefijo, claves in ((izquierda, "robustez", [(m, "todas") for m in MODELOS]),
                                    (derecha, "variantes", [("rf", v) for v in VARIANTES_RF])):
            series = _contenedores(ax, f"{prefijo}:")
            assert set(series) == {f"{prefijo}:barajado", f"{prefijo}:hacia_adelante"}, series
            xs = {}
            for esquema, color in (("barajado", estilo.AZUL), ("hacia_adelante", estilo.NARANJA)):
                medias = [ROBUSTEZ[c][esquema][0] for c in claves]
                desvios = [ROBUSTEZ[c][esquema][1] for c in claves]
                puntos = _comprobar_barras_verticales(ax, f"{prefijo}:{esquema}", medias, desvios)
                _, segmentos, color_barra = _barra(ax, f"{prefijo}:{esquema}")
                assert len(segmentos) == len(claves)
                assert _mismo_color(puntos.get_color(), color), (esquema, puntos.get_color())
                assert _mismo_color(color_barra, color), (esquema, color_barra)
                xs[esquema] = np.asarray(puntos.get_xdata(), float)
                assert np.all(np.abs(xs[esquema] - np.arange(len(claves))) < 0.3)
                # El valor de cada punto, anotado junto a él.
                textos = _textos(ax, "valor:")
                for (modelo, variante), x, media in zip(claves, xs[esquema], medias):
                    categoria = modelo if prefijo == "robustez" else variante
                    texto = textos[f"valor:{categoria}:{esquema}"]
                    assert texto.get_text() == _coma(media), (categoria, esquema, texto.get_text())
                    assert _cerca(texto.xy[0], x), "el valor se ancla en su propio punto"
            assert np.all(xs["barajado"] < xs["hacia_adelante"]), "barajado a la izquierda"
            assert _linea_base(ax, 0.5) == ROTULO_LINEA_BASE
            lo, hi = ax.get_ylim()
            assert lo < 0.5 and hi > max(m + d for c in claves for m, d in ROBUSTEZ[c].values())
        assert izquierda.get_shared_y_axes().joined(izquierda, derecha), "el mismo eje y"
        leyenda = [t.get_text() for t in fig.legends[0].get_texts()]
        assert leyenda[0].startswith("barajado") and leyenda[1].startswith("hacia adelante")
        assert leyenda[2] == ROTULO_LINEA_BASE
        _comprobar_sin_solapes(fig)
        for ax in fig.axes:
            _dentro_de_los_ejes(ax, "valor:")
    print("ok  04: pares barajado (azul) y hacia adelante (naranja) con ±1 desvío en el orden NB, "
          "SVM, KNN, RF; RF por variante en su panel; valores anotados, «sin modelo» en 0,5, sin "
          "solapes")


def test_robustez_un_panel_sin_variantes_y_errores():
    tabla = _resumen_robustez()
    solo_todas = tabla[tabla["variante"] == "todas"]
    with _abierta(figura_robustez(solo_todas)) as fig:
        assert len(fig.axes) == 1 and _rotulos_x(fig.axes[0]) == ROTULOS
    una_fila = tabla[(tabla["esquema"] == "barajado") & (tabla["metrica"] == "auc")].head(1)
    for mala in (pd.concat([tabla, una_fila]), tabla.drop(columns="desvio"),
                 tabla[tabla["variante"] != "todas"]):
        try:
            figura_robustez(mala)
        except ValueError:
            pass
        else:
            raise AssertionError("una fila repetida, una columna faltante o la falta de «todas» "
                                 "deberían ser un error")
        finally:
            plt.close("all")
    print("ok  04: sin variantes de RF sale un panel; filas repetidas, columnas faltantes o sin "
          "«todas», ValueError")


# --- Figura 05 --------------------------------------------------------------------------------

def test_robustez_folds_rf_bloque_por_bloque():
    with _abierta(figura_robustez_folds(_largo_robustez(), _folds_robustez())) as fig:
        ax = fig.axes[0]
        puntos = _lineas(ax, "folds:")
        assert set(puntos) == {"folds:rf:hacia_adelante"}, "una sola serie: RF hacia adelante"
        serie = puntos["folds:rf:hacia_adelante"]
        assert list(serie.get_xdata()) == [1, 2, 3, 4, 5]
        assert np.allclose(serie.get_ydata(), ADELANTE_RF), list(serie.get_ydata())
        assert _mismo_color(serie.get_color(), estilo.NARANJA)
        tallos = [c for c in ax.collections if isinstance(c, LineCollection)
                  and c.get_gid() == "tallos:rf:hacia_adelante"]
        assert len(tallos) == 1
        segmentos = [np.asarray(s, float) for s in tallos[0].get_segments()]
        assert len(segmentos) == 5
        for fold, (segmento, auc) in enumerate(zip(segmentos, ADELANTE_RF), start=1):
            assert np.allclose(segmento, [[fold, 0.5], [fold, auc]]), (fold, segmento)
        assert _mismo_color(tallos[0].get_color()[0], estilo.NARANJA)
        assert _linea_base(ax, 0.5) == ROTULO_LINEA_BASE
        # La media barajada de RF, gris y a trazos, rotulada con su valor.
        barajado = _lineas(ax, "referencia-barajado")["referencia-barajado"]
        media = float(np.mean(BARAJADO_RF))
        assert np.allclose(barajado.get_ydata(), media)
        assert barajado.get_linestyle() == "--"
        assert _mismo_color(barajado.get_color(), estilo.TINTA_SECUNDARIA)
        assert _textos(ax, "rotulo-barajado")["rotulo-barajado"].get_text() == \
            f"barajado, {_coma(media)}"
        # Cada bloque: su número, sus filas y su % de «yes».
        esperados = [f"bloque {f}\nfilas {_con_miles(desde)}–{_con_miles(hasta)}\n"
                     f"{_coma(pct, 1)} % de «yes»"
                     for f, (desde, hasta, pct, _, _) in BLOQUES.items()]
        assert [t.get_text() for t in ax.get_xticklabels()] == esperados, \
            [t.get_text() for t in ax.get_xticklabels()]
        assert "de 4" + FINO + "000 a 24" + FINO + "000" in ax.get_xlabel()
        # El valor de cada fold, del lado opuesto a «sin modelo»: arriba si supera 0,5.
        renderer = fig.canvas.get_renderer()
        centros = serie.get_transform().transform(np.asarray(serie.get_xydata(), float))
        textos = _textos(ax, "valor:fold:")
        for fold, (auc, (_, y)) in enumerate(zip(ADELANTE_RF, centros), start=1):
            texto = textos[f"valor:fold:{fold}"]
            assert texto.get_text() == _coma(auc)
            texto.get_window_extent(renderer)
            caja = Text.get_window_extent(texto, renderer)
            assert (caja.y0 > y) if auc >= 0.5 else (caja.y1 < y), (fold, auc, caja, y)
        subtitulo = _textos(ax, "subtitulo")["subtitulo"].get_text()
        assert f"{_coma(np.mean(ADELANTE_RF))} ± {_coma(np.std(ADELANTE_RF, ddof=1))}" in subtitulo
        _comprobar_sin_solapes(fig)
        _dentro_de_los_ejes(ax, "valor:")
    print("ok  05: RF hacia adelante fold a fold en naranja con tallo desde 0,5, el valor del lado "
          "opuesto a «sin modelo», la media barajada gris a trazos, bloques con filas y % de «yes»")


def test_robustez_folds_sin_tabla_de_folds_y_errores():
    with _abierta(figura_robustez_folds(_largo_robustez())) as fig:
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == \
            [f"bloque {f}" for f in range(1, 6)]
    largo = _largo_robustez()
    for malo in (largo[largo["esquema"] != "hacia_adelante"], pd.concat([largo, largo.head(40)]),
                 largo.drop(columns="variante")):
        try:
            figura_robustez_folds(malo, _folds_robustez())
        except ValueError:
            pass
        else:
            raise AssertionError("sin filas hacia adelante, con un fold repetido o sin variante "
                                 "debería ser un error")
        finally:
            plt.close("all")
    print("ok  05: sin robustez_folds.csv, los bloques sólo con su número; sin datos hacia "
          "adelante o con folds repetidos, ValueError")


# --- Figura 06 --------------------------------------------------------------------------------

def test_ganancia_curvas_diagonal_y_presupuesto():
    with _abierta(figura_ganancia(_ganancia())) as fig:
        ax = fig.axes[0]
        curvas = _lineas(ax, "ganancia:")
        assert set(curvas) == {f"ganancia:{m}" for m in MODELOS}, "una curva por modelo"
        for modelo in MODELOS:
            linea = curvas[f"ganancia:{modelo}"]
            # Desde el origen, y después los 100 puntos del CSV en orden de p.
            assert np.allclose(linea.get_xdata(), np.r_[0, P])
            assert np.allclose(linea.get_ydata(), np.r_[0, _alcanzado(modelo, P)])
            assert _mismo_color(linea.get_color(), estilo.COLOR_MODELO[
                "nb" if modelo.startswith("nb") else modelo]), modelo
        grosor = {m: curvas[f"ganancia:{m}"].get_linewidth() for m in MODELOS}
        assert all(grosor["rf"] > grosor[m] for m in MODELOS if m != "rf"), "RF resaltado"
        diagonal = _lineas(ax, "linea-base")["linea-base"]
        assert np.allclose(diagonal.get_xdata(), np.r_[0, P])
        assert np.allclose(diagonal.get_ydata(), np.r_[0, P])
        assert diagonal.get_linestyle() == ":"
        assert _mismo_color(diagonal.get_color(), COLOR_LINEA_BASE)
        assert _textos(ax, "rotulo-linea-base")["rotulo-linea-base"].get_text() == \
            ROTULO_LINEA_BASE
        # La línea vertical del presupuesto, en el 20 %, con el valor de RF en el cruce.
        presupuesto = _lineas(ax, "presupuesto")["presupuesto"]
        assert list(presupuesto.get_xdata()) == [20, 20]
        assert "20 %" in _textos(ax, "rotulo-presupuesto")["rotulo-presupuesto"].get_text()
        en_20 = float(_alcanzado("rf", 20))
        cruce = _lineas(ax, "cruce:rf")["cruce:rf"]
        assert _cerca(cruce.get_xdata()[0], 20) and _cerca(cruce.get_ydata()[0], en_20)
        assert _mismo_color(cruce.get_color(), estilo.NARANJA)
        valor = _textos(ax, "valor-presupuesto:rf")["valor-presupuesto:rf"]
        assert valor.get_text() == f"RF: {_coma(en_20, 1)} %", valor.get_text()
        assert _textos(ax, "valor-presupuesto:sin_modelo")[
            "valor-presupuesto:sin_modelo"].get_text() == "20,0 %"
        # La leyenda, en el orden de los modelos, con lo que alcanza cada uno al 20 %.
        leyenda = [t.get_text() for t in ax.get_legend().get_texts()]
        assert leyenda == [f"{r}: {_coma(float(_alcanzado(m, 20)), 1)} %"
                           for m, r in zip(MODELOS, ROTULOS)] + ["sin modelo: 20,0 %"], leyenda
        assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 100 and ax.get_ylim()[1] >= 100
        _comprobar_sin_solapes(fig)
        _dentro_de_los_ejes(ax, "valor-presupuesto:")
    with _abierta(figura_ganancia(_ganancia(con_diagonal=False))) as fig:
        diagonal = _lineas(fig.axes[0], "linea-base")["linea-base"]
        assert np.allclose(diagonal.get_xdata(), [0, 100]) and \
            np.allclose(diagonal.get_ydata(), [0, 100])
    print("ok  06: una curva por modelo desde el origen con su color, RF resaltado, la diagonal "
          "«sin modelo» punteada, el presupuesto en el 20 % con el valor de RF en el cruce")


# --- Figura 07 --------------------------------------------------------------------------------

def test_sensibilidad_recall_y_precision_por_q():
    with _abierta(figura_sensibilidad(_sensibilidad())) as fig:
        assert len(fig.axes) == 2
        for ax, metrica in zip(fig.axes, ("recall_q", "precision_q")):
            assert _rotulos_x(ax) == ["10 %", "20 %", "30 %"]
            pesos = [t.get_fontweight() for t in ax.get_xticklabels()]
            assert pesos[1] == "bold" and pesos[0] != "bold" and pesos[2] != "bold", pesos
            series = _contenedores(ax, "sensibilidad:")
            assert set(series) == {f"sensibilidad:{m}:{metrica}" for m in MODELOS}
            xs = []
            for j, modelo in enumerate(MODELOS):
                medias = SENSIBILIDAD[metrica][modelo]
                desvios = [_paso_q(j, i) * RAIZ for i in range(3)]
                puntos = _comprobar_barras_verticales(ax, f"sensibilidad:{modelo}:{metrica}",
                                                      medias, desvios)
                _, _, color_barra = _barra(ax, f"sensibilidad:{modelo}:{metrica}")
                color = estilo.COLOR_MODELO["nb" if modelo.startswith("nb") else modelo]
                assert _mismo_color(puntos.get_color(), color) and _mismo_color(color_barra, color)
                xs.append(np.asarray(puntos.get_xdata(), float))
                textos = _textos(ax, f"valor:{modelo}:")
                for q, media in zip(("10", "20", "30"), medias):
                    assert textos[f"valor:{modelo}:{q}:{metrica}"].get_text() == _coma(media)
            xs = np.array(xs)  # modelos × q
            assert np.all(np.diff(xs, axis=0) > 0), "dentro de cada q, en el orden NB, SVM, KNN, RF"
            assert np.all(np.abs(xs - np.arange(3)) < 0.45), "cada modelo junto a su q"
            # «Sin modelo»: un segmento por q, en el valor de sus filas.
            base = _lineas(ax, "linea-base")["linea-base"]
            ys = np.asarray(base.get_ydata(), float)
            esperado = list(Q_SENSIBILIDAD) if metrica == "recall_q" else [TASA_YES] * 3
            assert np.allclose(ys[~np.isnan(ys)], np.repeat(esperado, 2)), list(ys)
            assert base.get_linestyle() == ":"
            assert _mismo_color(base.get_color(), COLOR_LINEA_BASE)
            rotulo = _textos(ax, "rotulo-linea-base")["rotulo-linea-base"].get_text()
            assert rotulo == ("sin modelo: recall = q" if metrica == "recall_q"
                              else "sin modelo: 0,113, la tasa de «yes»"), rotulo
            _dentro_de_los_ejes(ax, "valor:")
        leyenda = [t.get_text() for t in fig.legends[0].get_texts()]
        assert leyenda == ROTULOS + [ROTULO_LINEA_BASE], leyenda
        _comprobar_sin_solapes(fig)
    print("ok  07: recall y precisión por q con ±1 desvío, los modelos lado a lado en su orden y "
          "color, sus valores, «sin modelo» por q y el 20 % en negrita, sin solapes")


# --- CLI --------------------------------------------------------------------------------------

def _escribir_entradas(resultados):
    resultados.mkdir(parents=True, exist_ok=True)
    _resumen_cv_final().to_csv(resultados / "cv_final_resumen.csv", index=False)
    _resumen_cv().to_csv(resultados / "cv_referencia_resumen.csv", index=False)
    _resumen_robustez().to_csv(resultados / "robustez_temporal_resumen.csv", index=False)
    _largo_robustez().to_csv(resultados / "robustez_temporal.csv", index=False)
    _folds_robustez().to_csv(resultados / "robustez_folds.csv", index=False)
    _ganancia().to_csv(resultados / "ganancia_final.csv", index=False)
    _sensibilidad().to_csv(resultados / "sensibilidad_q_final.csv", index=False)


NUEVAS = {FIGURA_MODELOS, FIGURA_MODELOS_REFERENCIA, FIGURA_ROBUSTEZ, FIGURA_ROBUSTEZ_FOLDS,
          FIGURA_GANANCIA, FIGURA_SENSIBILIDAD}


def test_cli_escribe_las_figuras_nuevas_y_respeta_solo_y_etiqueta():
    with tempfile.TemporaryDirectory() as tmp:
        resultados, figuras = Path(tmp) / "resultados", Path(tmp) / "figuras"
        _escribir_entradas(resultados)
        argumentos = ["--resultados", str(resultados), "--figuras", str(figuras)]

        codigo, salida = _en_silencio(cli, argumentos)
        assert codigo == 0, salida
        escritas = {p.name for p in figuras.glob("*.png")}
        assert escritas == NUEVAS, escritas ^ NUEVAS
        assert "ablaciones_resumen.csv" in salida and "No hay curvas" in salida
        for ruta in figuras.glob("*.png"):
            with Image.open(ruta) as imagen:
                assert imagen.size[0] > 600 and imagen.size[1] > 300
                assert round(imagen.info["dpi"][0]) == 150

        # --solo y --etiqueta: sólo se reescribe lo pedido.
        casos = [(["--solo", "modelos", "--etiqueta", "final"], {FIGURA_MODELOS}),
                 (["--solo", "modelos", "--etiqueta", "referencia"], {FIGURA_MODELOS_REFERENCIA}),
                 (["--solo", "modelos"], {FIGURA_MODELOS, FIGURA_MODELOS_REFERENCIA}),
                 (["--solo", "robustez"], {FIGURA_ROBUSTEZ, FIGURA_ROBUSTEZ_FOLDS}),
                 (["--solo", "ganancia"], {FIGURA_GANANCIA}),
                 (["--solo", "sensibilidad"], {FIGURA_SENSIBILIDAD})]
        viejo = 10**18  # ns: septiembre de 2001; lo que se reescriba tendrá otra fecha
        for opciones, esperadas in casos:
            for ruta in figuras.glob("*.png"):
                os.utime(ruta, ns=(viejo, viejo))
            codigo, salida = _en_silencio(cli, [*opciones, *argumentos])
            assert codigo == 0, (opciones, salida)
            cambiadas = {p.name for p in figuras.glob("*.png") if p.stat().st_mtime_ns != viejo}
            assert cambiadas == esperadas, (opciones, cambiadas)
    print("ok  CLI: las seis figuras nuevas a 150 dpi; --solo modelos|robustez|ganancia|"
          "sensibilidad y --etiqueta final|referencia reescriben sólo lo suyo")


def test_cli_sin_un_csv_informa_y_sigue():
    with tempfile.TemporaryDirectory() as tmp:
        resultados, figuras = Path(tmp) / "resultados", Path(tmp) / "figuras"
        _escribir_entradas(resultados)
        argumentos = ["--resultados", str(resultados), "--figuras", str(figuras)]

        # Sin robustez_folds.csv: la 04 sale y la 05 se omite con aviso, sin error.
        (resultados / "robustez_folds.csv").unlink()
        codigo, salida = _en_silencio(cli, ["--solo", "robustez", *argumentos])
        assert codigo == 0, salida
        assert {p.name for p in figuras.glob("*.png")} == {FIGURA_ROBUSTEZ}
        assert _renglon(salida, "Falta", "robustez_folds.csv", FIGURA_ROBUSTEZ_FOLDS), salida

        # Sin cv_final_resumen.csv: la 02b sale y la 02 se omite con aviso.
        (resultados / "cv_final_resumen.csv").unlink()
        codigo, salida = _en_silencio(cli, ["--solo", "modelos", *argumentos])
        assert codigo == 0 and (figuras / FIGURA_MODELOS_REFERENCIA).exists(), salida
        assert not (figuras / FIGURA_MODELOS).exists()
        assert _renglon(salida, "Falta", "cv_final_resumen.csv", FIGURA_MODELOS), salida
        # Y con --etiqueta final, no hay nada que dibujar: se informa y no es un error.
        codigo, salida = _en_silencio(cli, ["--solo", "modelos", "--etiqueta", "final",
                                            *argumentos])
        assert codigo == 0 and "0 figuras" in salida, salida

        # Un CSV cortado a mitad de una fila se informa con código 1, y lo demás se dibuja.
        texto = _ganancia().to_csv(index=False)
        corte = texto.index("\n", len(texto) // 2) - 3  # a mitad de una fila
        assert texto[corte - 1] != "\n"
        (resultados / "ganancia_final.csv").write_text(texto[:corte], encoding="utf-8")
        codigo, salida = _en_silencio(cli, argumentos)
        assert codigo == 1, salida
        assert _renglon(salida, "No se pudo leer", "ganancia_final.csv",
                        "la última fila está cortada", FIGURA_GANANCIA), salida
        assert (figuras / FIGURA_SENSIBILIDAD).exists() and not (figuras / FIGURA_GANANCIA).exists()

        # Sin nada en resultados/: cada entrada que falta se informa, y termina bien sin escribir.
        vacio, otras = Path(tmp) / "vacio", Path(tmp) / "otras"
        codigo, salida = _en_silencio(cli, ["--resultados", str(vacio), "--figuras", str(otras)])
        assert codigo == 0 and not otras.exists(), salida
        for archivo in ("cv_final_resumen.csv", "cv_referencia_resumen.csv",
                        "robustez_temporal_resumen.csv", "robustez_temporal.csv",
                        "robustez_folds.csv", "ganancia_final.csv", "sensibilidad_q_final.csv"):
            assert _renglon(salida, "Falta", archivo), (archivo, salida)
    print("ok  CLI: sin robustez_folds.csv o sin cv_final_resumen.csv, lo que se puede se dibuja "
          "y lo demás se informa; un CSV cortado da código 1; sin nada, sin error")


def main():
    test_constantes_coinciden_con_src_robustez_y_src_configuracion()
    test_formato_de_miles_porcentajes_y_bloques()
    test_modelos_final_cuatro_modelos_y_referencia_con_su_etiqueta()
    test_robustez_pares_por_modelo_y_variantes_de_rf()
    test_robustez_un_panel_sin_variantes_y_errores()
    test_robustez_folds_rf_bloque_por_bloque()
    test_robustez_folds_sin_tabla_de_folds_y_errores()
    test_ganancia_curvas_diagonal_y_presupuesto()
    test_sensibilidad_recall_y_precision_por_q()
    test_cli_escribe_las_figuras_nuevas_y_respeta_solo_y_etiqueta()
    test_cli_sin_un_csv_informa_y_sigue()
    assert not plt.get_fignums(), f"quedaron figuras abiertas: {plt.get_fignums()}"
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
