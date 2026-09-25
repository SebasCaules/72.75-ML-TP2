"""Correr con: python -m tests.test_curvas

Los casos de --elegir usan curvas sintéticas calculadas a mano en los comentarios: en cada punto,
los cinco folds valen media + d·(−2, −1, 0, 1, 2), así que la media es la dada, el desvío muestral
es d·√(10/4) y el error estándar, d·√(10/4)/√5 = d/√2. Los casos que ajustan modelos usan la
submuestra de --rapido (3 000 filas, un proceso) con KNN, y terminan en pocos segundos.
"""

import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.curvas import main as cli
from src.curvas import (
    COLUMNAS,
    OMITIDOS,
    correr_curva,
    elegir,
    hiperparametros_del_punto,
    interpretar,
    leer_curvas,
    parsear_fijos,
    parsear_valores,
    punto_de,
    recortar_para_rapido,
    registrar_omitidos,
)
from src.metricas import METRICAS
from src.modelos import crear_modelo
from src.resultados import CAMPOS, con_identidad, escribir_largo
from src.validacion import K

DESPLAZAMIENTOS = (-2, -1, 0, 1, 2)


def _cerca(a, b, tol=1e-12):
    return abs(a - b) < tol


def _escribir_curva(directorio, nombre, modelo, parametro, puntos, fijos=None):
    """`puntos`: (valor, AUC medio de validación, d, AUC medio de train). La configuración de cada
    punto sale de `hiperparametros_del_punto`, como en una curva real."""
    partes = []
    for valor, media_va, d, media_tr in puntos:
        filas = [{"fold": fold, "conjunto": conjunto, "metrica": metrica,
                  "valor": media + d * k if metrica == "auc" else 0.5}
                 for fold, k in enumerate(DESPLAZAMIENTOS, start=1)
                 for conjunto, media in (("train", media_tr), ("validacion", media_va))
                 for metrica in METRICAS]
        hiper = hiperparametros_del_punto(modelo, parametro, valor, fijos)
        partes.append(con_identidad(pd.DataFrame(filas), modelo, hiper, parametro=parametro,
                                    punto=punto_de(valor)))
    escribir_largo(Path(directorio) / f"{nombre}.csv", pd.concat(partes, ignore_index=True))


def _elegir_en_silencio(directorio):
    with contextlib.redirect_stdout(io.StringIO()):
        return elegir(directorio)


def _falla(funcion, *argumentos):
    try:
        funcion(*argumentos)
    except ValueError:
        return True
    return False


def test_interpreta_valores_y_fijos():
    assert interpretar("None") is None
    assert interpretar("0.1") == 0.1 and type(interpretar("0.1")) is float
    assert interpretar("5") == 5 and type(interpretar("5")) is int
    assert interpretar("distance") == "distance"
    assert interpretar("True") is True and interpretar("1e-3") == 0.001
    assert parsear_valores("2,4,None") == [2, 4, None]
    assert parsear_valores("linear, poly ,rbf") == ["linear", "poly", "rbf"]
    assert parsear_valores("None,balanced") == [None, "balanced"]
    assert parsear_fijos(["weights=distance", "C=0.1"]) == {"weights": "distance", "C": 0.1}
    assert parsear_fijos([]) == {}
    assert all(_falla(parsear_fijos, [malo]) for malo in ("weights", "=3", "C="))
    assert _falla(parsear_valores, "1,,2") and _falla(parsear_valores, "5,5")
    assert punto_de(None) == "None" and punto_de(0.1) == "0.1" and punto_de(5) == "5"
    print("ok  --valores y --fijo: None, 0,1 (float), 5 (int) y 'distance' con su tipo; "
          "rechaza vacíos, repetidos y pares sin '='")


def test_kernel_lineal_usa_liblinear():
    lineal = hiperparametros_del_punto("svm", "kernel", "linear", {"C": 0.1})
    assert lineal["implementacion"] == "liblinear" and lineal["C"] == 0.1
    assert type(crear_modelo("svm", **lineal)[-1]).__name__ == "LinearSVC"
    for kernel in ("poly", "rbf"):
        otro = hiperparametros_del_punto("svm", "kernel", kernel, {"C": 0.1})
        assert "implementacion" not in otro
        assert type(crear_modelo("svm", **otro)[-1]).__name__ == "SVC"
    explicito = hiperparametros_del_punto("svm", "kernel", "linear", {"implementacion": "libsvm"})
    assert explicito["implementacion"] == "libsvm"
    print("ok  kernel lineal -> implementacion=liblinear (LinearSVC); poly y RBF siguen en SVC; "
          "un --fijo explícito manda")


def test_rapido_recorta_grillas_y_arboles():
    assert recortar_para_rapido("knn", "n_neighbors", [1, 3, 5, 9, 15], False) == [1, 5, 15]
    assert recortar_para_rapido("knn", "n_neighbors", [1, 3, 5, 9, 15], True) == [1, 3, 5, 9, 15]
    # [10, 50, 800] -> hasta 20 árboles: [10, 20, 20] -> sin repetidos: [10, 20]
    assert recortar_para_rapido("rf", "n_estimators", [10, 25, 50, 100, 800], False) == [10, 20]
    assert hiperparametros_del_punto("rf", "max_depth", 4, rapido=True)["n_estimators"] == 20
    print("ok  --rapido: extremos y centro de la grilla, valores explícitos intactos, RF con 20 árboles")


def test_rapido_de_knn_escribe_el_contrato():
    with tempfile.TemporaryDirectory() as tmp:
        directorio = Path(tmp) / "curvas"
        with contextlib.redirect_stdout(io.StringIO()):
            codigo = cli(["--modelo", "knn", "--parametro", "n_neighbors", "--valores", "5,15",
                           "--rapido", "--salida", str(directorio)])
        assert codigo == 0
        crudo = pd.read_csv(directorio / "knn_n_neighbors.csv")
        assert list(crudo.columns) == COLUMNAS == ["parametro", "punto"] + CAMPOS
        assert len(crudo) == 2 * K * 2 * len(METRICAS)
        assert set(crudo["conjunto"]) == {"train", "validacion"} and set(crudo["metrica"]) == set(METRICAS)
        assert crudo["valor"].between(0, 1).all()
        tabla = leer_curvas(directorio)["knn_n_neighbors"]
        assert list(dict.fromkeys(tabla["punto"])) == ["5", "15"]
        configuracion = json.loads(tabla["configuracion"].iloc[0])
        assert configuracion["n_neighbors"] == 5 and "weights" in configuracion
        assert not (directorio / OMITIDOS).exists()
    print(f"ok  --rapido de knn/n_neighbors con 5 y 15: {2 * K * 2 * len(METRICAS)} filas con las "
          "columnas del contrato y los hiperparámetros completos")


def test_punto_que_falla_se_omite_y_se_registra():
    with tempfile.TemporaryDirectory() as tmp:
        directorio = Path(tmp) / "curvas"
        with contextlib.redirect_stdout(io.StringIO()):
            ruta = correr_curva("knn", "n_neighbors", [5, 0], rapido=True, directorio=directorio)
        assert ruta == directorio / "knn_n_neighbors.csv"
        assert set(leer_curvas(directorio)["knn_n_neighbors"]["punto"]) == {"5"}
        lineas = (directorio / OMITIDOS).read_text(encoding="utf-8").splitlines()
        assert len(lineas) == 1 and lineas[0].startswith("knn_n_neighbors\tn_neighbors=0\t")
        # Otra corrida de la misma curva reemplaza sus líneas: la omisión vieja no queda.
        with contextlib.redirect_stdout(io.StringIO()):
            correr_curva("knn", "n_neighbors", [5], rapido=True, directorio=directorio)
        assert (directorio / OMITIDOS).read_text(encoding="utf-8") == ""
    print("ok  n_neighbors=0 falla: se omite, el CSV queda con el resto y omitidos.txt dice por qué; "
          "la corrida siguiente limpia su línea")


def test_limite_de_tiempo_omite_los_puntos_y_elegir_lo_avisa():
    with tempfile.TemporaryDirectory() as tmp:
        directorio = Path(tmp) / "curvas"
        with contextlib.redirect_stdout(io.StringIO()):
            primera = correr_curva("knn", "n_neighbors", [5, 15], rapido=True, directorio=directorio)
            ruta = correr_curva("knn", "n_neighbors", [5, 15], rapido=True, directorio=directorio,
                                limite=0.001)
            correr_curva("knn", "n_neighbors", [5], {"weights": "distance"}, "distance",
                         rapido=True, directorio=directorio, limite=0.001)
        assert primera is not None and ruta is None
        assert not (directorio / "knn_n_neighbors_distance.csv").exists()
        lineas = (directorio / OMITIDOS).read_text(encoding="utf-8").splitlines()
        assert len(lineas) == 3 and all("TimeoutError: excedió el límite" in l for l in lineas)
        avisos = _elegir_en_silencio(directorio)["avisos"]
    assert any(a.startswith("knn_n_neighbors: la última corrida omitió") and "corrida anterior" in a
               for a in avisos)
    assert "knn_n_neighbors_distance: la última corrida omitió todos sus puntos y no hay CSV" in avisos
    print("ok  con --limite de 1 ms los puntos exceden y quedan en omitidos.txt; --elegir avisa del "
          "CSV de una corrida anterior y de la curva sin CSV")


def test_elegir_curva_plana_gana_el_mas_simple():
    # max_depth 2, 6 y None con AUC de validación 0,760, 0,790 y 0,795; d = 0,01, ES = 0,00707.
    # El mejor es None; el umbral, 0,795 − 0,00707 = 0,78793. Quedan dentro 6 y None, y el más
    # simple es 6 (None, sin límite, es el más complejo). Brecha en 6: 0,830 − 0,790 = 0,040.
    with tempfile.TemporaryDirectory() as tmp:
        directorio = Path(tmp) / "curvas"
        _escribir_curva(directorio, "rf_max_depth", "rf", "max_depth",
                        [(2, 0.760, 0.01, 0.765), (6, 0.790, 0.01, 0.830), (None, 0.795, 0.01, 0.990)])
        c = _elegir_en_silencio(directorio)["curvas"]["rf_max_depth"]
    assert c["valor"] == 6 and type(c["valor"]) is int and c["punto"] == "6"
    assert c["mejor"] == "None" and c["curva_plana"] and c["puntos_dentro_1es"] == ["6", "None"]
    assert _cerca(c["umbral_1es"], 0.795 - 0.01 / np.sqrt(2))
    assert _cerca(c["auc_validacion_media"], 0.790) and _cerca(c["auc_train_media"], 0.830)
    assert _cerca(c["brecha"], 0.040) and _cerca(c["error_estandar"], 0.01 / np.sqrt(2))
    print("ok  curva plana de max_depth: dentro de 1 ES quedan 6 y None, y gana 6, el más simple")


def test_elegir_gana_el_maximo():
    # C = 0,1, 1 y 10 con AUC 0,770, 0,800 y 0,780; d = 0,004, ES = 0,00283. El umbral es
    # 0,800 − 0,00283 = 0,79717: sólo C = 1 queda dentro, y gana aunque 0,1 sea más simple.
    # Brecha en C = 1: 0,815 − 0,800 = 0,015.
    with tempfile.TemporaryDirectory() as tmp:
        directorio = Path(tmp) / "curvas"
        _escribir_curva(directorio, "svm_C", "svm", "C",
                        [(0.1, 0.770, 0.004, 0.772), (1, 0.800, 0.004, 0.815), (10, 0.780, 0.004, 0.860)])
        c = _elegir_en_silencio(directorio)["curvas"]["svm_C"]
    assert c["valor"] == 1 and type(c["valor"]) is int and c["mejor"] == "1"
    assert not c["curva_plana"] and c["puntos_dentro_1es"] == ["1"]
    assert _cerca(c["brecha"], 0.015) and _cerca(c["error_estandar"], 0.004 / np.sqrt(2))
    print("ok  C con un solo punto dentro de 1 ES: gana el máximo, C = 1, y no el más simple")


def test_elegir_n_neighbors_mas_simple_es_el_mayor():
    # k = 5, 25 y 101 con AUC 0,740, 0,790 y 0,785; d = 0,01, ES = 0,00707. El mejor es 25; el
    # umbral, 0,790 − 0,00707 = 0,78293. También queda dentro 101, y en KNN lo más simple es el k
    # mayor: gana 101, no 25. Brecha en 101: 0,790 − 0,785 = 0,005.
    with tempfile.TemporaryDirectory() as tmp:
        directorio = Path(tmp) / "curvas"
        _escribir_curva(directorio, "knn_n_neighbors", "knn", "n_neighbors",
                        [(5, 0.740, 0.01, 0.900), (25, 0.790, 0.01, 0.810), (101, 0.785, 0.01, 0.790)])
        c = _elegir_en_silencio(directorio)["curvas"]["knn_n_neighbors"]
    assert c["valor"] == 101 and c["mejor"] == "25" and c["puntos_dentro_1es"] == ["25", "101"]
    assert _cerca(c["brecha"], 0.005) and _cerca(c["auc_validacion_media"], 0.785)
    print("ok  n_neighbors: dentro de 1 ES quedan 25 y 101, y gana 101, el mayor")


def test_elegir_por_modelo_con_tipos_listos_para_crear_modelo():
    # KNN uniforme: el caso anterior, elige k = 101 con 0,785. KNN por distancia: 0,750, 0,780 y
    # 0,770 con d = 0,001 (ES = 0,000707); sólo 25 queda dentro y elige 25 con 0,780. Gana la
    # uniforme (0,785 > 0,780): k = 101 con weights = uniform.
    # RF pesos_clase: None 0,793 y balanced 0,790, d = 0,01. Los dos quedan dentro de 1 ES
    # (umbral 0,78593), pero no hay orden de simplicidad: gana None, la mayor media.
    # RF max_depth: 6 con 0,780 y None con 0,795, d = 0,001 (ES = 0,000707); sólo None queda
    # dentro. Cada curva de RF se midió sin el parámetro de la otra, que vale None por omisión, así
    # que ninguna genera aviso.
    # SVM: C elige 1 (el caso de «gana el máximo»); kernel lineal 0,795, poly 0,770 y RBF 0,790,
    # d = 0,01: gana el lineal, la mayor media, y el modelo lleva implementacion = liblinear. La
    # curva de C se midió con RBF: es el único aviso.
    with tempfile.TemporaryDirectory() as tmp:
        directorio = Path(tmp) / "curvas"
        _escribir_curva(directorio, "knn_n_neighbors_uniform", "knn", "n_neighbors",
                        [(5, 0.740, 0.01, 0.900), (25, 0.790, 0.01, 0.810), (101, 0.785, 0.01, 0.790)],
                        {"weights": "uniform"})
        _escribir_curva(directorio, "knn_n_neighbors_distance", "knn", "n_neighbors",
                        [(5, 0.750, 0.001, 1.0), (25, 0.780, 0.001, 1.0), (101, 0.770, 0.001, 1.0)],
                        {"weights": "distance"})
        _escribir_curva(directorio, "rf_pesos_clase", "rf", "pesos_clase",
                        [(None, 0.793, 0.01, 0.95), ("balanced", 0.790, 0.01, 0.94)])
        _escribir_curva(directorio, "rf_max_depth", "rf", "max_depth",
                        [(6, 0.780, 0.001, 0.85), (None, 0.795, 0.001, 0.99)])
        _escribir_curva(directorio, "svm_C", "svm", "C",
                        [(0.1, 0.770, 0.004, 0.772), (1, 0.800, 0.004, 0.815), (10, 0.780, 0.004, 0.860)],
                        {"kernel": "rbf"})
        _escribir_curva(directorio, "svm_kernel", "svm", "kernel",
                        [("linear", 0.795, 0.01, 0.80), ("poly", 0.770, 0.01, 0.85),
                         ("rbf", 0.790, 0.01, 0.82)], {"C": 1})
        registrar_omitidos(directorio, "svm_kernel", [("kernel=sigmoid", "ValueError: prueba")])
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli(["--elegir", "--salida", str(directorio)]) == 0
        escrito = json.loads((Path(tmp) / "hiperparametros.json").read_text(encoding="utf-8"))

    modelos, curvas = escrito["modelos"], escrito["curvas"]
    assert modelos["knn"] == {"n_neighbors": 101, "weights": "uniform"}
    assert type(modelos["knn"]["n_neighbors"]) is int
    assert modelos["rf"] == {"max_depth": None, "pesos_clase": None}
    assert modelos["svm"] == {"C": 1, "kernel": "linear", "implementacion": "liblinear"}
    assert curvas["knn_n_neighbors_uniform"]["adoptada"]
    assert not curvas["knn_n_neighbors_distance"]["adoptada"]
    assert curvas["knn_n_neighbors_distance"]["valor"] == 25
    assert curvas["rf_pesos_clase"]["curva_plana"] and curvas["rf_pesos_clase"]["valor"] is None
    assert curvas["svm_kernel"]["curva_plana"] and curvas["svm_kernel"]["valor"] == "linear"
    assert curvas["svm_kernel"]["omitidos"] == ["kernel=sigmoid: ValueError: prueba"]
    assert len(escrito["avisos"]) == 1 and escrito["avisos"][0].startswith("svm_C se midió con")
    assert "kernel=rbf" in escrito["avisos"][0]
    assert type(crear_modelo("svm", **modelos["svm"])[-1]).__name__ == "LinearSVC"
    for nombre in ("knn", "rf"):
        crear_modelo(nombre, **modelos[nombre])
    print("ok  por modelo: KNN k = 101 uniforme, RF sin límite de profundidad y pesos_clase None, "
          "SVM C = 1 lineal con liblinear; el JSON conserva los tipos, crear_modelo los acepta y el "
          "único aviso es el C medido con RBF")


def test_ayuda():
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        try:
            cli(["--help"])
        except SystemExit as fin:
            assert fin.code == 0
    assert "--elegir" in salida.getvalue() and "--rapido" in salida.getvalue()
    print("ok  --help funciona")


def main():
    test_interpreta_valores_y_fijos()
    test_kernel_lineal_usa_liblinear()
    test_rapido_recorta_grillas_y_arboles()
    test_rapido_de_knn_escribe_el_contrato()
    test_punto_que_falla_se_omite_y_se_registra()
    test_limite_de_tiempo_omite_los_puntos_y_elegir_lo_avisa()
    test_elegir_curva_plana_gana_el_mas_simple()
    test_elegir_gana_el_maximo()
    test_elegir_n_neighbors_mas_simple_es_el_mayor()
    test_elegir_por_modelo_con_tipos_listos_para_crear_modelo()
    test_ayuda()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
