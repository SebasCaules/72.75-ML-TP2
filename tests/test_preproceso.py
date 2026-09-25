"""Correr con: python -m tests.test_preproceso

Lo central es la ausencia de fuga: todo lo que el preprocesamiento aprende de los datos tiene que
salir sólo del fold de entrenamiento (TP2, p. 1). Se comprueba sobre el primer fold de `folds()`.
"""

import numpy as np
import pandas as pd

from src.datos import cargar_train
from src.modelos import crear_modelo, separar_X_y
from src.preproceso import (
    CATEGORICAS,
    MACRO,
    VOCABULARIO,
    Derivadas,
    Opciones,
    categorias_minimas,
    columnas,
)
from src.validacion import folds


def _datos():
    X, y = separar_X_y(cargar_train())
    tr, va = next(folds().split(X, y))
    return X, y, tr, va


def test_el_vocabulario_cubre_train(X, y, tr, va):
    for c in CATEGORICAS:
        fuera = set(X[c].unique()) - set(VOCABULARIO[c])
        assert not fuera, f"{c}: niveles fuera del vocabulario {fuera}"
    print("ok  todos los niveles de train están en el vocabulario de names.txt")


def test_columnas_de_referencia(X, y, tr, va):
    numericas, categoricas, binarias = columnas(Opciones())
    assert len(numericas) == 9 and len(categoricas) == 10 and not binarias
    assert "duration" not in numericas
    salida = crear_modelo("svm")[:-1].fit(X.iloc[tr], y[tr]).transform(X.iloc[va])
    assert salida.shape == (len(va), 62)
    print("ok  referencia: 9 numéricas sin duration + 53 columnas one-hot = 62")


def test_cada_opcion_cambia_lo_que_dice(X, y, tr, va):
    casos = {
        Opciones(duration=True): lambda n, c, b: "duration" in n,
        Opciones(pdays="indicadora"): lambda n, c, b: "pdays" not in n and b == ["contactado_antes"],
        Opciones(default="indicadora"): lambda n, c, b: "default" not in c and b == ["default_unknown"],
        Opciones(macro="reducido"): lambda n, c, b: [m for m in MACRO if m in n] == ["euribor3m", "nr.employed"],
        Opciones(macro="sin"): lambda n, c, b: not set(MACRO) & set(n),
        Opciones(month=False): lambda n, c, b: "month" not in c,
        Opciones(day_of_week=False): lambda n, c, b: "day_of_week" not in c,
        Opciones(edad_tramos=True): lambda n, c, b: "age" not in n and "edad_tramo" in c,
    }
    for opciones, condicion in casos.items():
        assert condicion(*columnas(opciones)), opciones
        Derivadas(opciones).transform(X.iloc[:50])
    print(f"ok  las {len(casos)} opciones de columnas quitan y agregan lo que dicen")


def test_reglas_fijas_de_derivadas(X, y, tr, va):
    muestra = pd.DataFrame({
        "age": [24, 25, 29, 30, 60, 95], "campaign": [1, 2, 3, 4, 5, 6],
        "pdays": [999, 3, 999, 999, 999, 999], "previous": [0, 1, 2, 0, 0, 0],
        **{m: [1.0] * 6 for m in MACRO},
        "job": ["admin."] * 6, "marital": ["unknown", "single", "married", "divorced", "married", "single"],
        "education": ["illiterate", "basic.4y", "unknown", "high.school", "basic.6y", "basic.9y"],
        "default": ["no", "unknown", "yes", "no", "no", "no"], "housing": ["no"] * 6, "loan": ["no"] * 6,
        "contact": ["cellular"] * 6, "month": ["may"] * 6, "day_of_week": ["mon"] * 6,
        "poutcome": ["nonexistent", "failure", "failure", "nonexistent", "nonexistent", "nonexistent"],
    })
    d = Derivadas(Opciones(pdays="indicadora", default="indicadora", raras=True,
                           campaign_log=True, edad_tramos=True)).transform(muestra)
    assert list(d["edad_tramo"]) == ["<25", "25-29", "25-29", "30-39", "60+", "60+"]
    assert list(d["contactado_antes"]) == [0, 1, 1, 0, 0, 0]
    assert list(d["default_unknown"]) == [0, 1, 0, 0, 0, 0]
    assert d["marital"].iloc[0] == "married" and d["education"].iloc[0] == "basic.4y"
    assert np.allclose(d["campaign"], np.log1p([1, 2, 3, 4, 5, 6]))
    print("ok  tramos de edad, contacto previo, default, fusión de raras y log1p, contra valores a mano")


def test_derivadas_no_aprende(X, y, tr, va):
    op = Opciones(pdays="indicadora", raras=True, edad_tramos=True, unknown="moda")
    a = Derivadas(op).fit(X.iloc[tr]).transform(X.iloc[va])
    b = Derivadas(op).fit(X.iloc[va]).transform(X.iloc[va])
    pd.testing.assert_frame_equal(a, b)
    print("ok  Derivadas da lo mismo sin importar con qué se ajustó: no aprende de los datos")


def test_el_escalado_sale_solo_del_fold_de_entrenamiento(X, y, tr, va):
    modelo = crear_modelo("knn").fit(X.iloc[tr], y[tr])
    escalador = modelo.named_steps["columnas"].named_transformers_["num"]
    numericas = columnas(Opciones())[0]
    assert np.allclose(escalador.mean_, X.iloc[tr][numericas].mean().to_numpy())
    assert not np.allclose(escalador.mean_, X[numericas].mean().to_numpy())
    print("ok  el StandardScaler guarda las medias del fold de entrenamiento, no las de todo train")


def test_la_discretizacion_sale_solo_del_fold_de_entrenamiento(X, y, tr, va):
    modelo = crear_modelo("nb_categorico").fit(X.iloc[tr], y[tr])
    disc = modelo.named_steps["columnas"].named_transformers_["num"]
    edad_tr = np.unique(np.quantile(X.iloc[tr]["age"], np.linspace(0, 1, 11)[1:-1]))
    edad_va = np.unique(np.quantile(X.iloc[va]["age"], np.linspace(0, 1, 11)[1:-1]))
    assert np.allclose(disc.cortes_[0], edad_tr)
    assert not np.array_equal(edad_tr, edad_va)
    print("ok  los cortes de la discretización son los cuantiles del fold de entrenamiento")


def test_la_imputacion_sale_solo_del_fold_de_entrenamiento(X, y, tr, va):
    op = Opciones(unknown="moda")
    modelo = crear_modelo("rf", op, n_estimators=5).fit(X.iloc[tr], y[tr])
    imputador = modelo.named_steps["columnas"].named_transformers_["cat"].named_steps["imputar"]
    categoricas = columnas(op)[1]
    modas = [X.iloc[tr][c].replace("unknown", np.nan).mode()[0] for c in categoricas]
    assert list(imputador.statistics_) == modas
    print("ok  la imputación por la moda usa la moda del fold de entrenamiento")


def test_codigos_dentro_de_las_categorias_minimas(X, y, tr, va):
    for op in (Opciones(), Opciones(pdays="indicadora", edad_tramos=True, raras=True)):
        modelo = crear_modelo("nb_categorico", op).fit(X.iloc[tr], y[tr])
        codigos = modelo[:-1].transform(X.iloc[va])
        assert (codigos.max(axis=0) < categorias_minimas(op, 10)).all()
        assert codigos.min() >= 0
    print("ok  los códigos de CategoricalNB caben en min_categories, también en validación")


def main():
    datos = _datos()
    test_el_vocabulario_cubre_train(*datos)
    test_columnas_de_referencia(*datos)
    test_cada_opcion_cambia_lo_que_dice(*datos)
    test_reglas_fijas_de_derivadas(*datos)
    test_derivadas_no_aprende(*datos)
    test_el_escalado_sale_solo_del_fold_de_entrenamiento(*datos)
    test_la_discretizacion_sale_solo_del_fold_de_entrenamiento(*datos)
    test_la_imputacion_sale_solo_del_fold_de_entrenamiento(*datos)
    test_codigos_dentro_de_las_categorias_minimas(*datos)
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
