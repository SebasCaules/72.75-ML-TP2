"""Correr con: python -m tests.test_resultados"""

import numpy as np
import pandas as pd

from src.datos import FILA, SEMILLA, cargar_train
from src.metricas import METRICAS
from src.modelos import crear_modelo, separar_X_y
from src.resultados import (
    CAMPOS,
    con_identidad,
    configuracion_json,
    diferencias_pareadas,
    resumen,
    validacion_cruzada,
)
from src.validacion import K


def _muestra(n=3000):
    train = cargar_train().sample(n, random_state=SEMILLA).reset_index(drop=True)
    X, y = separar_X_y(train)
    return train, X, y


def test_validacion_cruzada_tiene_la_forma_del_contrato(train, X, y):
    tabla, oof = validacion_cruzada(crear_modelo("nb_gaussiano"), X, y, filas=train[FILA])
    assert len(tabla) == K * 2 * len(METRICAS)
    assert set(tabla["conjunto"]) == {"train", "validacion"}
    assert sorted(tabla["fold"].unique()) == list(range(1, K + 1))
    assert set(tabla["metrica"]) == set(METRICAS)
    assert tabla["valor"].between(0, 1).all()
    print(f"ok  la tabla tiene {K} folds × 2 conjuntos × {len(METRICAS)} métricas, todas entre 0 y 1")


def test_oof_cubre_cada_fila_una_vez(train, X, y):
    _, oof = validacion_cruzada(crear_modelo("nb_gaussiano"), X, y, filas=train[FILA])
    assert len(oof) == len(X) and oof["fila"].is_unique
    assert set(oof["fila"]) == set(train[FILA])
    assert np.isfinite(oof["puntaje"]).all()
    print("ok  los puntajes OOF cubren cada fila de train exactamente una vez")


def test_secuencial_y_paralelo_dan_lo_mismo(train, X, y):
    a, oof_a = validacion_cruzada(crear_modelo("nb_gaussiano"), X, y, n_jobs=1)
    b, oof_b = validacion_cruzada(crear_modelo("nb_gaussiano"), X, y, n_jobs=5)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(oof_a, oof_b)
    print("ok  con 1 o 5 procesos la validación cruzada da exactamente lo mismo")


def test_con_identidad_ordena_las_columnas(train, X, y):
    tabla, _ = validacion_cruzada(crear_modelo("nb_gaussiano"), X, y, n_jobs=1)
    t = con_identidad(tabla, "nb_gaussiano", {"b": 1, "a": 2}, variante="A0")
    assert list(t.columns) == ["variante"] + CAMPOS
    assert t["configuracion"].iloc[0] == '{"a": 2, "b": 1}' == configuracion_json({"b": 1, "a": 2})
    print("ok  con_identidad agrega modelo y configuración (JSON ordenado) y las columnas extra")


def test_resumen_y_diferencias_pareadas_calculadas_a_mano(train, X, y):
    filas = []
    for variante, valores in (("A0", [0.70, 0.72, 0.74]), ("A1", [0.71, 0.75, 0.76])):
        filas += [{"variante": variante, "modelo": "m", "configuracion": "{}", "fold": f,
                   "conjunto": "validacion", "metrica": "auc", "valor": v}
                  for f, v in enumerate(valores, start=1)]
    tabla = pd.DataFrame(filas)
    r = resumen(tabla, por=["variante"]).set_index("variante")
    assert abs(r.loc["A0", "media"] - 0.72) < 1e-12
    assert abs(r.loc["A0", "desvio"] - 0.02) < 1e-12
    assert abs(r.loc["A0", "error_estandar"] - 0.02 / np.sqrt(3)) < 1e-12
    d = diferencias_pareadas(tabla, "variante", "A0").iloc[0]
    # deltas: 0,01, 0,03, 0,02 → media 0,02, desvío muestral 0,01
    assert abs(d["delta_media"] - 0.02) < 1e-12 and abs(d["delta_desvio"] - 0.01) < 1e-12
    assert abs(d["delta_min"] - 0.01) < 1e-12 and abs(d["delta_max"] - 0.03) < 1e-12
    print("ok  resumen y diferencias pareadas coinciden con el cálculo a mano")


def main():
    datos = _muestra()
    test_validacion_cruzada_tiene_la_forma_del_contrato(*datos)
    test_oof_cubre_cada_fila_una_vez(*datos)
    test_secuencial_y_paralelo_dan_lo_mismo(*datos)
    test_con_identidad_ordena_las_columnas(*datos)
    test_resumen_y_diferencias_pareadas_calculadas_a_mano(*datos)
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
