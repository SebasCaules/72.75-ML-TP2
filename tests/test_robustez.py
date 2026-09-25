"""Correr con: python -m tests.test_robustez

Ninguna prueba ajusta modelos sobre todo train: el esquema hacia adelante se comprueba con los
índices de los folds, y las corridas de punta a punta usan --rapido (3 000 filas) con Naive Bayes en
directorios temporales. Los valores de robustez_<modelo>.csv y de robustez_folds.csv se comparan con
lo que da rehacer aquí cada esquema y cada variante con el particionador y las opciones que les
asigna D-25 (PARTICIONES y OPCIONES), sin pasar por el módulo. El resumen se comprueba contra una
tabla sintética calculada a mano.
"""

import contextlib
import io
import tempfile
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from src import robustez
from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES, OPCIONES_FINALES
from src.datos import FILA, SEMILLA, cargar_train
from src.metricas import METRICAS
from src.modelos import crear_modelo, separar_X_y
from src.preproceso import MACRO, columnas
from src.resultados import (
    CAMPOS,
    DIR,
    N_JOBS,
    configuracion_json,
    escribir_largo,
    leer_largo,
    validacion_cruzada,
)
from src.robustez import (
    CAMPOS_FOLDS,
    CAMPOS_RESUMEN,
    DIFERENCIA,
    ESQUEMAS,
    N_RAPIDO,
    VARIANTES,
    combinar,
    composicion,
    imprimir,
    opciones_de,
    ordenar,
    particionador,
    resumir,
)
from src.validacion import K, folds

# Lo que D-25 le asigna a cada esquema y a cada variante, escrito aquí sin pasar por el módulo.
PARTICIONES = {"hacia_adelante": lambda: TimeSeriesSplit(n_splits=K), "barajado": folds}
OPCIONES = {"todas": OPCIONES_FINALES,
            "sin_macro": replace(OPCIONES_FINALES, macro="sin"),
            "sin_macro_ni_month": replace(OPCIONES_FINALES, macro="sin", month=False)}

# Las corridas de punta a punta usan el Naive Bayes vigente, que ajusta en décimas de segundo.
NB = ("nb_gaussiano" if "nb_gaussiano" in MODELOS_FINALES
      else next(m for m in MODELOS_FINALES if m.startswith("nb_")))


def _cerca(a, b, tol=1e-12):
    return abs(a - b) < tol


def _callado(argv):
    with contextlib.redirect_stdout(io.StringIO()) as impreso:
        robustez.main(argv)
    return impreso.getvalue()


@contextlib.contextmanager
def _salida_rapida(directorio):
    """SALIDA_RAPIDO apunta a `directorio` mientras dura el bloque, para que las pruebas no
    escriban en el directorio temporal del sistema."""
    original = robustez.SALIDA_RAPIDO
    robustez.SALIDA_RAPIDO = directorio
    try:
        yield directorio
    finally:
        robustez.SALIDA_RAPIDO = original


def _composicion_esperada(ordenado):
    """robustez_folds.csv rehecho con PARTICIONES y las tasas contadas sobre la columna `y`."""
    X, y = separar_X_y(ordenado)
    registros = []
    for esquema in ESQUEMAS:
        for numero, (tr, va) in enumerate(PARTICIONES[esquema]().split(X, y), start=1):
            entrena, valida = ordenado.iloc[tr], ordenado.iloc[va]
            registros.append({
                "esquema": esquema, "fold": numero, "n_train": len(entrena),
                "n_validacion": len(valida),
                "pct_yes_train": 100 * (entrena["y"] == "yes").mean(),
                "pct_yes_validacion": 100 * (valida["y"] == "yes").mean(),
                "fila_min_validacion": valida[FILA].min(),
                "fila_max_validacion": valida[FILA].max(), "fila_max_train": entrena[FILA].max()})
    return pd.DataFrame(registros, columns=CAMPOS_FOLDS)


def _igual_composicion(obtenida, esperada):
    pd.testing.assert_frame_equal(obtenida.reset_index(drop=True), esperada, check_dtype=False,
                                  check_exact=False, rtol=0, atol=1e-9)


def test_hacia_adelante_valida_despues_de_entrenar(train):
    # Se desordena a propósito: el orden temporal lo tiene que poner ordenar().
    desordenado = train.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    ordenado = ordenar(desordenado)
    X, y = separar_X_y(ordenado)
    for tr, va in particionador("hacia_adelante").split(X, y):
        assert ordenado[FILA].iloc[tr].max() < ordenado[FILA].iloc[va].min()

    # Todas las columnas de robustez_folds.csv, también la tasa de «yes» de train y de validación.
    c = composicion(ordenado)
    _igual_composicion(c, _composicion_esperada(ordenado))
    c = c.set_index(["esquema", "fold"])
    adelante, barajado = c.loc["hacia_adelante"], c.loc["barajado"]
    assert (adelante["fila_max_train"] < adelante["fila_min_validacion"]).all()
    assert (barajado["fila_max_train"] > barajado["fila_min_validacion"]).all()
    # La prueba distingue: sin ordenar, la misma partición mezcla las épocas.
    sin_orden = composicion(desordenado).set_index(["esquema", "fold"]).loc["hacia_adelante"]
    assert not (sin_orden["fila_max_train"] < sin_orden["fila_min_validacion"]).all()

    # TimeSeriesSplit(5): seis bloques de n // 6 filas; el fold k entrena con los k primeros.
    n, bloque = len(train), len(train) // (K + 1)
    assert (adelante["n_validacion"] == bloque).all()
    assert list(adelante["n_train"]) == [n - (K - i) * bloque for i in range(K)]
    assert (barajado["n_train"] + barajado["n_validacion"] == n).all()

    # train.csv ya viene en orden temporal, así que el barajado usa los folds del resto de los
    # módulos.
    pd.testing.assert_frame_equal(ordenar(train), train)
    print(f"ok  hacia adelante, cada validación de {bloque} filas es posterior a su entrenamiento "
          f"(de {adelante['n_train'].iloc[0]} a {adelante['n_train'].iloc[-1]} filas); en el "
          f"barajado, no. «yes» en validación hacia adelante: de "
          f"{adelante['pct_yes_validacion'].min():.1f} % a "
          f"{adelante['pct_yes_validacion'].max():.1f} %, igual que el recuento directo")


def test_variantes_sin_las_columnas_de_la_epoca(train):
    for variante in VARIANTES:
        assert opciones_de(variante) == OPCIONES[variante], variante
    numericas, categoricas, binarias = columnas(opciones_de("todas"))
    for variante in ("sin_macro", "sin_macro_ni_month"):
        n, c, b = columnas(opciones_de(variante))
        assert not set(MACRO) & set(n + c + b), variante
        assert n == [x for x in numericas if x not in MACRO] and b == binarias, variante
    assert columnas(opciones_de("sin_macro"))[1] == categoricas
    assert columnas(opciones_de("sin_macro_ni_month"))[1] == [x for x in categoricas
                                                               if x != "month"]

    # De punta a punta: las columnas que salen del preprocesamiento ya ajustado.
    X, y = separar_X_y(train.sample(500, random_state=SEMILLA))
    for variante in VARIANTES:
        pipeline = crear_modelo("nb_gaussiano", opciones_de(variante)).fit(X, y)
        nombres = set(pipeline.named_steps["columnas"].get_feature_names_out())
        con_macro = bool(set(MACRO) & nombres)
        con_month = any(nombre.startswith("month_") for nombre in nombres)
        assert con_macro == (variante == "todas" and OPCIONES_FINALES.macro != "sin"), variante
        assert con_month == (variante != "sin_macro_ni_month" and OPCIONES_FINALES.month), variante
    print("ok  sin_macro no tiene columnas macro, sin_macro_ni_month tampoco tiene month, y el "
          "resto de las columnas es el de OPCIONES_FINALES")


def test_los_folds_del_csv_largo_son_los_de_robustez_folds(train):
    # Una figura cruza el AUC de cada fold (robustez_<modelo>.csv) con su rango de filas
    # (robustez_folds.csv): el número de fold tiene que nombrar la misma partición en los dos.
    muestra = ordenar(robustez.submuestra(train))
    X, y = separar_X_y(muestra)
    c = composicion(muestra).set_index(["esquema", "fold"])
    for esquema in ESQUEMAS:
        _, oof = validacion_cruzada(crear_modelo("nb_gaussiano"), X, y, filas=muestra[FILA],
                                    n_jobs=1, particionador=particionador(esquema))
        rangos = oof.groupby("fold")["fila"].agg(["min", "max", "count"])
        esperado = c.loc[esquema]
        assert (rangos["min"] == esperado["fila_min_validacion"]).all(), esquema
        assert (rangos["max"] == esperado["fila_max_validacion"]).all(), esquema
        assert (rangos["count"] == esperado["n_validacion"]).all(), esquema
    print(f"ok  en los {len(ESQUEMAS)} esquemas, el fold k de la validación cruzada es la fila k "
          "de robustez_folds.csv")


def test_recortes_de_rapido(train):
    muestra = robustez.submuestra(train)
    esperada, _ = train_test_split(train, train_size=N_RAPIDO, stratify=train["y"],
                                   random_state=SEMILLA)
    pd.testing.assert_frame_equal(muestra, esperada)
    tasa, tasa_train = (muestra["y"] == "yes").mean(), (train["y"] == "yes").mean()
    assert len(muestra) == N_RAPIDO and abs(tasa - tasa_train) < 1e-3
    assert robustez.hiperparametros_de("rf", rapido=True)["n_estimators"] <= robustez.ARBOLES_RAPIDO
    for modelo in HIPERPARAMETROS_FINALES:
        assert robustez.hiperparametros_de(modelo) == HIPERPARAMETROS_FINALES[modelo], modelo
    print(f"ok  --rapido: {len(muestra)} filas con {tasa:.2%} de «yes» (train: {tasa_train:.2%}) y "
          f"RF con {robustez.hiperparametros_de('rf', rapido=True)['n_estimators']} árboles; sin "
          "--rapido, los hiperparámetros de HIPERPARAMETROS_FINALES")


def _sintetica():
    # rf, todas: AUC hacia adelante 0,60 / 0,70 / 0,80 → media 0,70, desvío 0,10, ES 0,10/√3;
    # barajado 0,78 / 0,80 / 0,82 → media 0,80, desvío 0,02. Diferencia 0,80 − 0,70 = +0,10.
    # rf, sin_macro: 0,75 / 0,76 / 0,77 → 0,76 y 0,77 / 0,78 / 0,79 → 0,78. Diferencia +0,02: la
    # brecha se reduce sin las variables de la época.
    # rf, todas, recall_q: 0,30 / 0,40 / 0,50 → media 0,40, desvío 0,10; barajado 0,50 los tres →
    # desvío 0. knn sólo tiene hacia adelante: no lleva diferencia.
    # Las filas de train (0,99) y la métrica ap (0,123) no entran en el resumen.
    casos = {
        ("rf", "hacia_adelante", "todas", "auc"): [0.60, 0.70, 0.80],
        ("rf", "barajado", "todas", "auc"): [0.78, 0.80, 0.82],
        ("rf", "hacia_adelante", "sin_macro", "auc"): [0.75, 0.76, 0.77],
        ("rf", "barajado", "sin_macro", "auc"): [0.77, 0.78, 0.79],
        ("rf", "hacia_adelante", "todas", "recall_q"): [0.30, 0.40, 0.50],
        ("rf", "barajado", "todas", "recall_q"): [0.50, 0.50, 0.50],
        ("knn", "hacia_adelante", "todas", "auc"): [0.65, 0.70, 0.75],
    }
    filas = []
    for (modelo, esquema, variante, metrica), valores in casos.items():
        for fold, valor in enumerate(valores, start=1):
            base = {"esquema": esquema, "variante": variante, "modelo": modelo,
                    "configuracion": "{}", "fold": fold, "metrica": metrica}
            filas += [{**base, "conjunto": "validacion", "valor": valor},
                      {**base, "conjunto": "train", "valor": 0.99}]
            if metrica == "auc":
                filas.append({**base, "conjunto": "validacion", "metrica": "ap", "valor": 0.123})
    return pd.DataFrame(filas)[["esquema", "variante"] + CAMPOS]


def test_resumen_calculado_a_mano():
    tabla = _sintetica()
    r = resumir(tabla)
    assert list(r.columns) == CAMPOS_RESUMEN
    assert list(r["modelo"].unique()) == ["knn", "rf"]  # el orden de MODELOS
    r = r.set_index(["modelo", "esquema", "variante", "metrica"])
    assert len(r) == 9 and "ap" not in r.index.get_level_values("metrica")

    fila = r.loc[("rf", "hacia_adelante", "todas", "auc")]
    assert _cerca(fila["media"], 0.70) and _cerca(fila["desvio"], 0.10) and fila["n"] == 3
    assert _cerca(fila["error_estandar"], 0.10 / np.sqrt(3))
    fila = r.loc[("rf", "barajado", "todas", "auc")]
    assert _cerca(fila["media"], 0.80) and _cerca(fila["desvio"], 0.02)
    assert _cerca(r.loc[("rf", DIFERENCIA, "todas", "auc"), "media"], 0.10)
    assert _cerca(r.loc[("rf", DIFERENCIA, "sin_macro", "auc"), "media"], 0.02)
    fila = r.loc[("rf", "hacia_adelante", "todas", "recall_q")]
    assert _cerca(fila["media"], 0.40) and _cerca(fila["desvio"], 0.10)
    fila = r.loc[("rf", "barajado", "todas", "recall_q")]
    assert _cerca(fila["media"], 0.50) and _cerca(fila["desvio"], 0.0)
    assert ("knn", DIFERENCIA, "todas", "auc") not in r.index
    assert _cerca(r.loc[("knn", "hacia_adelante", "todas", "auc"), "media"], 0.70)

    with contextlib.redirect_stdout(io.StringIO()) as impreso:
        imprimir(resumir(tabla), tabla)
    texto = impreso.getvalue()
    assert "0.700 ± 0.100" in texto and "0.800 ± 0.020" in texto
    assert "+0.100" in texto and "+0.020" in texto
    assert texto.count(DIFERENCIA) == 1  # sólo rf tiene los dos esquemas
    todas, sin_macro = (r.loc[("rf", DIFERENCIA, v, "auc"), "media"] for v in ("todas", "sin_macro"))
    print("ok  resumen: media, desvío y error estándar por esquema y variante, y la diferencia "
          f"barajado − hacia adelante ({todas:+.2f} y {sin_macro:+.2f}), como en el cálculo a mano")


def _archivo(pares, valor, configuracion="{}"):
    """Un robustez_<modelo>.csv sintético: el AUC de validación de dos folds por par."""
    filas = [{"esquema": esquema, "variante": variante, "modelo": "rf",
              "configuracion": configuracion, "fold": fold, "conjunto": "validacion",
              "metrica": "auc", "valor": valor}
             for esquema, variante in pares for fold in (1, 2)]
    return pd.DataFrame(filas)[["esquema", "variante"] + CAMPOS]


def test_combinar_reemplaza_solo_los_pares_medidos():
    # El archivo anterior y la corrida nueva vienen fuera de orden a propósito.
    anterior = _archivo([("barajado", "todas"), ("hacia_adelante", "todas"),
                         ("hacia_adelante", "sin_macro")], 0.5)
    nueva = _archivo([("barajado", "todas"), ("hacia_adelante", "sin_macro_ni_month")], 0.9)
    t = combinar(anterior, nueva)
    esperado = [("hacia_adelante", "todas", 0.5), ("hacia_adelante", "sin_macro", 0.5),
                ("hacia_adelante", "sin_macro_ni_month", 0.9), ("barajado", "todas", 0.9)]
    assert list(zip(t["esquema"], t["variante"], t["valor"])) == [x for x in esperado
                                                                  for _ in (1, 2)]
    assert list(t["fold"]) == [1, 2] * len(esperado)
    assert list(t.columns) == list(anterior.columns)
    try:
        combinar(anterior, _archivo([("barajado", "todas")], 0.9, '{"n_estimators": 20}'))
        raise AssertionError("debió negarse a combinar hiperparámetros distintos")
    except ValueError:
        pass
    print("ok  combinar: los pares medidos reemplazan a los anteriores, los demás se conservan y "
          "todo queda en el orden de una corrida completa")


def test_linea_de_comandos():
    with contextlib.redirect_stdout(io.StringIO()) as ayuda:
        try:
            robustez.main(["--help"])
            raise AssertionError("--help debió terminar el proceso")
        except SystemExit as e:
            assert e.code == 0
    assert "--rapido" in ayuda.getvalue() and "--resumen" in ayuda.getvalue()
    # Con argumentos(), que sólo interpreta la línea: si un control fallara, la prueba no llegaría
    # a medir ni a escribir en resultados/.
    erroneos = [
        [], ["--modelo", "rf", "--resumen"], ["--modelo", "rf", "--esquemas", "x"],
        ["--modelo", "rf", "--variantes", "todas,sin_edad"], ["--modelo", "rf", "--n-jobs", "0"],
        # --resumen junta lo que ya está medido: no lleva las opciones de una medición.
        ["--resumen", "--esquemas", "barajado"], ["--resumen", "--variantes", "todas"],
        ["--resumen", "--n-jobs", "3"],
        # --rapido nunca escribe ni resume dentro de resultados/.
        ["--modelo", "rf", "--rapido", "--salida", str(DIR)],
        ["--modelo", "rf", "--rapido", "--salida", str(DIR / "robustez")],
        ["--resumen", "--rapido", "--salida", str(DIR)],
    ]
    for argv in erroneos:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                robustez.argumentos(argv)
                raise AssertionError(f"debió fallar: {argv}")
            except SystemExit as e:
                assert e.code == 2, argv
    a = robustez.argumentos(["--modelo", "rf", "--variantes", "sin_macro,todas"])
    assert a.variantes == ["todas", "sin_macro"] and a.esquemas == list(ESQUEMAS)
    assert a.salida == DIR and a.n_jobs == N_JOBS
    assert robustez.argumentos(["--resumen"]).salida == DIR
    with tempfile.TemporaryDirectory() as tmp, _salida_rapida(Path(tmp)):
        a = robustez.argumentos(["--modelo", "rf", "--rapido"])
        assert a.salida == Path(tmp) and a.n_jobs == 1
        assert robustez.argumentos(["--resumen", "--rapido"]).salida == Path(tmp)
    print(f"ok  --help funciona; la línea de comandos rechaza {len(erroneos)} combinaciones "
          "(valores desconocidos, opciones de medición en --resumen y --rapido dentro de "
          "resultados/), y --rapido y --resumen --rapido comparten el directorio temporal")


def test_rapido_de_punta_a_punta(train):
    with tempfile.TemporaryDirectory() as tmp, _salida_rapida(Path(tmp) / "rapido") as salida:
        # Antes de correr: sin --salida, --rapido tiene que ir al directorio temporal de la prueba.
        assert robustez.argumentos(["--modelo", NB, "--rapido"]).salida == salida
        inicio = time.perf_counter()
        _callado(["--modelo", NB, "--rapido"])
        segundos = time.perf_counter() - inicio

        tabla = leer_largo(salida / f"robustez_{NB}.csv")
        assert list(tabla.columns) == ["esquema", "variante"] + CAMPOS
        assert len(tabla) == len(ESQUEMAS) * len(VARIANTES) * K * 2 * len(METRICAS)
        assert set(tabla["configuracion"]) == {configuracion_json(HIPERPARAMETROS_FINALES[NB])}
        assert tabla["valor"].between(0, 1).all()

        # Cada valor, contra una validación cruzada rehecha con el particionador y las opciones
        # de su esquema y su variante, sobre la misma submuestra en orden temporal.
        muestra = robustez.submuestra(train).sort_values(FILA).reset_index(drop=True)
        X, y = separar_X_y(muestra)
        aucs = {}
        for esquema in ESQUEMAS:
            for variante in VARIANTES:
                modelo = crear_modelo(NB, OPCIONES[variante], **HIPERPARAMETROS_FINALES[NB])
                esperado, _ = validacion_cruzada(modelo, X, y, n_jobs=1,
                                                 particionador=PARTICIONES[esquema]())
                propio = tabla[(tabla["esquema"] == esquema) & (tabla["variante"] == variante)]
                m = propio.merge(esperado, on=["fold", "conjunto", "metrica"],
                                 validate="one_to_one")
                assert len(m) == len(esperado) == len(propio), (esquema, variante)
                assert np.allclose(m["valor_x"], m["valor_y"], rtol=0, atol=1e-12), (esquema,
                                                                                       variante)
                auc = esperado[(esperado["conjunto"] == "validacion")
                               & (esperado["metrica"] == "auc")].sort_values("fold")
                aucs[esquema, variante] = auc["valor"].to_numpy()
        # La comparación distingue: cada esquema, y cada variante con otras opciones, da otro AUC.
        for variante in VARIANTES:
            assert not np.allclose(aucs["hacia_adelante", variante], aucs["barajado", variante])
        for esquema in ESQUEMAS:
            for una, otra in combinations(VARIANTES, 2):
                if OPCIONES[una] != OPCIONES[otra]:
                    assert not np.allclose(aucs[esquema, una], aucs[esquema, otra]), (una, otra)

        _igual_composicion(pd.read_csv(salida / "robustez_folds.csv"),
                           _composicion_esperada(muestra))

        # --resumen junta varios modelos en el orden de MODELOS (el Naive Bayes antes que knn, al
        # revés que el alfabético). La copia de knn sólo tiene el barajado: no lleva diferencia.
        copia = tabla[tabla["esquema"] == "barajado"].assign(modelo="knn")
        escribir_largo(salida / "robustez_knn.csv", copia)
        impreso = _callado(["--resumen", "--rapido"])
        pd.testing.assert_frame_equal(leer_largo(salida / "robustez_temporal.csv"),
                                      pd.concat([tabla, copia], ignore_index=True))
        r = pd.read_csv(salida / "robustez_temporal_resumen.csv")
        assert list(r.columns) == CAMPOS_RESUMEN
        assert list(r["modelo"].unique()) == [NB, "knn"]
        de_nb, de_knn = r[r["modelo"] == NB], r[r["modelo"] == "knn"]
        assert len(de_nb) == len(ESQUEMAS) * len(VARIANTES) * 2 + len(VARIANTES)
        assert len(de_knn) == len(VARIANTES) * 2 and set(de_knn["esquema"]) == {"barajado"}
        barajado_nb = de_nb[de_nb["esquema"] == "barajado"]
        pd.testing.assert_frame_equal(de_knn.drop(columns="modelo").reset_index(drop=True),
                                      barajado_nb.drop(columns="modelo").reset_index(drop=True))
        assert impreso.count(DIFERENCIA) == 1 and all(v in impreso for v in VARIANTES)
        faltan = [m for m in MODELOS_FINALES if m not in (NB, "knn")]
        assert f"Faltan los modelos: {', '.join(faltan)}." in impreso
        assert "knn, sin medir: hacia_adelante × todas" in impreso
    assert segundos < 60, segundos
    print(f"ok  --rapido con {NB}: los {len(tabla)} valores coinciden con la validación cruzada "
          "rehecha por esquema y variante, igual que los folds; --resumen junta dos modelos en "
          f"{len(r)} filas; {segundos:.1f} s")
    return tabla


def test_corridas_parciales_se_combinan(tabla):
    with tempfile.TemporaryDirectory() as tmp:
        ruta, ruta_folds = Path(tmp) / f"robustez_{NB}.csv", Path(tmp) / "robustez_folds.csv"
        base = ["--modelo", NB, "--rapido", "--salida", tmp]
        _callado(base + ["--esquemas", "hacia_adelante"])
        assert set(leer_largo(ruta)["esquema"]) == {"hacia_adelante"}
        impreso = _callado(base + ["--esquemas", "barajado"])
        pd.testing.assert_frame_equal(leer_largo(ruta), tabla)
        assert "se conservan hacia_adelante × todas" in impreso
        assert set(pd.read_csv(ruta_folds)["esquema"]) == set(ESQUEMAS)
        # Medir de nuevo un par lo reemplaza en su lugar, sin duplicarlo.
        _callado(base + ["--esquemas", "barajado", "--variantes", "sin_macro"])
        pd.testing.assert_frame_equal(leer_largo(ruta), tabla)

        # Con otros hiperparámetros, una corrida parcial se niega antes de medir o escribir nada.
        ajena = tabla.assign(configuracion='{"otro": 1}')
        escribir_largo(ruta, ajena)
        ruta_folds.unlink()
        try:
            _callado(base + ["--variantes", "todas"])
            raise AssertionError("debió negarse a combinar hiperparámetros distintos")
        except SystemExit as e:
            assert "otros hiperparámetros" in str(e.code), e.code
        pd.testing.assert_frame_equal(leer_largo(ruta), ajena)
        assert not ruta_folds.exists()
        # Una corrida completa, en cambio, reemplaza el archivo entero.
        _callado(base)
        pd.testing.assert_frame_equal(leer_largo(ruta), tabla)
    print("ok  dos corridas parciales, un esquema cada una, dan el mismo archivo que una completa; "
          "medir de nuevo un par lo reemplaza, y con otros hiperparámetros sólo una corrida "
          "completa reescribe el archivo")


def main():
    train = cargar_train()
    test_hacia_adelante_valida_despues_de_entrenar(train)
    test_variantes_sin_las_columnas_de_la_epoca(train)
    test_los_folds_del_csv_largo_son_los_de_robustez_folds(train)
    test_recortes_de_rapido(train)
    test_resumen_calculado_a_mano()
    test_combinar_reemplaza_solo_los_pares_medidos()
    test_linea_de_comandos()
    tabla = test_rapido_de_punta_a_punta(train)
    test_corridas_parciales_se_combinan(tabla)
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
