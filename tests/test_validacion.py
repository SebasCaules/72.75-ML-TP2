"""Correr con: python -m tests.test_validacion"""

from src.datos import FILA, cargar_train
from src.validacion import K, composicion, folds


def test_folds_mezclan_todas_las_epocas(train):
    tabla = composicion(folds(), train)
    mediana = train[FILA].median()
    assert (tabla["fila_mediana"] - mediana).abs().max() < 1000, tabla
    assert (tabla["euribor3m_medio"] - train["euribor3m"].mean()).abs().max() < 0.1, tabla
    print(f"ok  los {K} folds cubren todo el período: fila mediana a menos de 1000 de la global")


def test_folds_estratificados(train):
    tabla = composicion(folds(), train)
    assert tabla["pct_yes"].max() - tabla["pct_yes"].min() <= 0.1, tabla
    print(f"ok  los {K} folds tienen la misma proporción de 'yes'")


def test_folds_reproducibles(train):
    y = (train["y"] == "yes").astype(int)
    a = [list(v) for _, v in folds().split(train, y)]
    b = [list(v) for _, v in folds().split(train, y)]
    assert a == b
    print("ok  folds() da siempre los mismos folds")


if __name__ == "__main__":
    train = cargar_train()
    test_folds_mezclan_todas_las_epocas(train)
    test_folds_estratificados(train)
    test_folds_reproducibles(train)
