"""Correr con: python -m tests.test_datos

Verifica la partición guardada en data/particion/ contra el CSV original. Requiere haber corrido
antes `python -m src.datos`.
"""

import pandas as pd

from src.datos import (
    FILA,
    OBJETIVO,
    PROP_TEST,
    cargar,
    cargar_test,
    cargar_train,
    limpiar,
    partir,
)


FILAS_CSV = 41_188
DUPLICADOS = 12


def _datos():
    return limpiar(cargar()), cargar_train(), cargar_test()


def test_limpiar_quita_solo_las_repeticiones(df, train, test):
    crudo = cargar()
    assert len(crudo) == FILAS_CSV
    assert len(df) == FILAS_CSV - DUPLICADOS
    # Cada fila distinta del CSV aparece exactamente una vez, en su primera aparición.
    sin_fila = crudo.drop(columns=FILA)
    primeras = crudo.loc[~sin_fila.duplicated(keep="first"), FILA]
    assert list(df[FILA]) == list(primeras)
    assert len(sin_fila.drop_duplicates()) == len(df)
    print(f"ok  limpiar() quita las {DUPLICADOS} repeticiones y conserva la primera aparición de cada fila")


def test_train_y_test_son_disjuntos(df, train, test):
    comunes = set(train[FILA]) & set(test[FILA])
    assert not comunes, f"filas en train y en test a la vez: {sorted(comunes)[:10]}"
    print("ok  ninguna fila está en train y en test a la vez")


def test_la_union_es_el_dataset_sin_duplicados(df, train, test):
    union = pd.concat([train, test]).sort_values(FILA).reset_index(drop=True)
    pd.testing.assert_frame_equal(union, df, check_dtype=False)
    print(f"ok  train + test reconstruyen las {len(df):,} filas sin duplicados, celda por celda")


def test_no_quedan_duplicados(df, train, test):
    for nombre, parte in [("train", train), ("test", test)]:
        assert not parte.drop(columns=FILA).duplicated().any(), f"hay duplicados en {nombre}"
    assert not pd.concat([train, test]).drop(columns=FILA).duplicated().any()
    print("ok  no quedan duplicados exactos, ni dentro de cada parte ni entre las dos")


def test_proporcion_del_test(df, train, test):
    assert abs(len(test) / len(df) - PROP_TEST) < 1e-3
    print(f"ok  el test es el {len(test) / len(df):.2%} del dataset")


def test_estratificado_por_y(df, train, test):
    # Estratificar deja en el test PROP_TEST de los «yes», redondeado: a lo sumo una fila de
    # diferencia. Un sorteo sin estratificar se aleja varias decenas (desvío ≈ 26 filas).
    n_yes = int((df[OBJETIVO] == "yes").sum())
    n_yes_test = int((test[OBJETIVO] == "yes").sum())
    esperado = PROP_TEST * n_yes
    assert abs(n_yes_test - esperado) <= 1, f"{n_yes_test} 'yes' en test contra {esperado:.1f}"
    print(f"ok  el test tiene {n_yes_test} 'yes', a menos de una fila de los {esperado:.1f} esperados")


def test_orden_temporal_preservado(df, train, test):
    for nombre, parte in [("train", train), ("test", test)]:
        assert parte[FILA].is_monotonic_increasing, f"{nombre} no está ordenado por fila"
    print("ok  train y test están ordenados por la fila original")


def test_la_particion_es_reproducible(df, train, test):
    train_2, test_2 = partir(df)
    assert list(train_2[FILA]) == list(train[FILA])
    assert list(test_2[FILA]) == list(test[FILA])
    print("ok  volver a partir con la misma semilla da exactamente las mismas filas")


if __name__ == "__main__":
    datos = _datos()
    test_limpiar_quita_solo_las_repeticiones(*datos)
    test_train_y_test_son_disjuntos(*datos)
    test_la_union_es_el_dataset_sin_duplicados(*datos)
    test_no_quedan_duplicados(*datos)
    test_proporcion_del_test(*datos)
    test_estratificado_por_y(*datos)
    test_orden_temporal_preservado(*datos)
    test_la_particion_es_reproducible(*datos)
    print("TODOS LOS TESTS OK")
