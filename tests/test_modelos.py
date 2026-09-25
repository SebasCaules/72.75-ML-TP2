"""Correr con: python -m tests.test_modelos

Prueba rápida: cada clasificador entrena y predice sobre una muestra pequeña de train. No mide
desempeño: eso es de las olas 1 a 4.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.datos import SEMILLA, cargar_train
from src.metricas import evaluar, puntajes
from src.modelos import GRILLAS, MODELOS, crear_modelo, separar_X_y
from src.preproceso import Opciones


def _muestra():
    X, y = separar_X_y(cargar_train())
    X_a, X_b, y_a, y_b = train_test_split(X, y, train_size=500, test_size=300, stratify=y,
                                          random_state=SEMILLA)
    return X_a, X_b, y_a, y_b


def test_cada_modelo_entrena_y_predice(X_a, X_b, y_a, y_b):
    for nombre in MODELOS:
        s = puntajes(crear_modelo(nombre).fit(X_a, y_a), X_b)
        assert s.shape == (len(X_b),) and np.isfinite(s).all(), nombre
        assert 0 <= evaluar(y_b, s)["auc"] <= 1
    print(f"ok  los {len(MODELOS)} clasificadores entrenan con 500 filas y puntúan 300")


def test_la_svm_lineal_de_liblinear(X_a, X_b, y_a, y_b):
    s = puntajes(crear_modelo("svm", kernel="linear", implementacion="liblinear").fit(X_a, y_a), X_b)
    assert np.isfinite(s).all()
    print("ok  la SVM lineal con liblinear entrena y puntúa")


def test_escalar_se_puede_apagar(X_a, X_b, y_a, y_b):
    con = crear_modelo("knn").named_steps["columnas"].transformers[0][1]
    sin = crear_modelo("knn", Opciones(escalar=False)).named_steps["columnas"].transformers[0][1]
    assert type(con).__name__ == "StandardScaler" and sin == "passthrough"
    print("ok  KNN escala por defecto y deja de escalar con Opciones(escalar=False)")


def test_rf_es_reproducible(X_a, X_b, y_a, y_b):
    a = puntajes(crear_modelo("rf", n_estimators=20).fit(X_a, y_a), X_b)
    b = puntajes(crear_modelo("rf", n_estimators=20).fit(X_a, y_a), X_b)
    assert np.array_equal(a, b)
    print("ok  RF da los mismos puntajes en dos corridas: la semilla está fija")


def test_las_grillas_arman_modelos_validos(X_a, X_b, y_a, y_b):
    for (nombre, parametro), valores in GRILLAS.items():
        for valor in (valores[0], valores[-1]):
            crear_modelo(nombre, **{parametro: valor}).fit(X_a.iloc[:200], y_a[:200])
    print(f"ok  los extremos de las {len(GRILLAS)} grillas arman modelos que entrenan")


def main():
    datos = _muestra()
    test_cada_modelo_entrena_y_predice(*datos)
    test_la_svm_lineal_de_liblinear(*datos)
    test_escalar_se_puede_apagar(*datos)
    test_rf_es_reproducible(*datos)
    test_las_grillas_arman_modelos_validos(*datos)
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
