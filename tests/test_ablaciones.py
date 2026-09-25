"""Correr con: python -m tests.test_ablaciones

Las ablaciones y la línea de base de la ola 1 (pasos 1.2 y 1.4). Las columnas por variante y el
criterio de efecto están calculados a mano en los comentarios. La corrida de punta a punta usa
--rapido y escribe en una carpeta temporal, nunca en resultados/.
"""

import contextlib
import io
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import src.ablaciones as ablaciones
from src.ablaciones import (
    BASE,
    COLUMNAS_RESUMEN,
    MODELOS_ABLACION,
    N_RAPIDO,
    VARIANTES,
    columnas_salida,
    comprobar,
    correr_resumen,
    efecto,
    hiperparametros,
    linea_base,
    main as main_ablaciones,
    medir_variante,
    ordenar,
    reemplazar_variantes,
    resumir,
    ruta_oof,
    salida_por_defecto,
    submuestra,
    variantes_de,
)
from src.datos import FILA, OBJETIVO, RAIZ, cargar_train
from src.metricas import METRICAS, llamadas
from src.modelos import MODELOS, REFERENCIA, crear_modelo, separar_X_y
from src.preproceso import Opciones
from src.resultados import (
    CAMPOS,
    CAMPOS_OOF,
    DIR,
    configuracion_json,
    escribir_largo,
    leer_largo,
)
from src.validacion import K, folds

# La tabla «Las ablaciones del paso 1.4» de plan/PLAN.md, transcrita a mano: qué campos de
# Opciones cambia cada variante, y en qué modelos se mide la que no va en todos.
CAMBIOS_DEL_PLAN = {
    "A0": {},
    "A1": {"duration": True},
    "A2": {"pdays": "sin"},
    "A3": {"default": "indicadora"},
    "A4": {"raras": True},
    "A5": {"campaign_log": True},
    "A6": {"macro": "reducido"},
    "A7": {"macro": "sin"},
    "A8": {"macro": "sin", "month": False},
    "A9": {"day_of_week": False},
    "A10": {"edad_tramos": True},
    "A11": {"unknown": "moda"},
    "A12": {"escalar": False},
}
SOLO_EN = {"A5": {"svm", "knn"}, "A10": {"nb_gaussiano"}, "A12": {"knn"}}

# Los tres efectos del criterio mecánico del paso 1.5.
EFECTOS = {"mejora", "empeora", "neutra"}

# Referencia: 9 numéricas (age, campaign, pdays, previous y las 5 macro) más 53 niveles one-hot
# (job 12, marital 4, education 8, default 3, housing 3, loan 3, contact 2, month 10,
# day_of_week 5 y poutcome 3) = 62. Cada variante suma o resta lo que dice su comentario.
COLUMNAS_A_MANO = {
    "A0": 62,
    "A1": 63,   # + duration
    "A2": 61,   # - pdays
    "A3": 60,   # - los 3 niveles de default + 1 indicadora
    "A4": 60,   # - illiterate - marital unknown
    "A5": 62,   # log1p no cambia cuántas columnas hay
    "A6": 59,   # 5 macro -> 2
    "A7": 57,   # - 5 macro
    "A8": 47,   # - 5 macro - 10 meses
    "A9": 57,   # - 5 días
    "A10": 70,  # - age + 9 tramos
    "A11": 56,  # - unknown en job, marital, education, default, housing y loan
    "A12": 62,  # sin escalar, las mismas columnas
}


def _cerca(a, b, tol=1e-12):
    return abs(a - b) < tol


def _cambios(opciones, base=Opciones()):
    a, b = asdict(opciones), asdict(base)
    return {campo: valor for campo, valor in a.items() if valor != b[campo]}


def test_las_variantes_son_las_del_plan():
    assert list(VARIANTES) == list(CAMBIOS_DEL_PLAN)
    assert MODELOS_ABLACION == ("nb_gaussiano", "svm", "knn", "rf")
    assert set(MODELOS_ABLACION) <= set(MODELOS)
    for nombre, variante in VARIANTES.items():
        assert _cambios(variante.opciones) == CAMBIOS_DEL_PLAN[nombre], nombre
        assert set(variante.modelos) == SOLO_EN.get(nombre, set(MODELOS_ABLACION)), nombre
        assert variante.descripcion, nombre
        if nombre not in (BASE, "A8"):
            assert len(_cambios(variante.opciones)) == 1, nombre
    # A8 cambia dos campos respecto de A0 porque el plan la define como A7 sin month: respecto de
    # A7 cambia uno solo.
    assert _cambios(VARIANTES["A8"].opciones, VARIANTES["A7"].opciones) == {"month": False}

    # 10 variantes en todos, más A5 en svm y knn, A10 en nb_gaussiano y A12 en knn: 44 corridas
    # de 5 folds, los «unos 200 ajustes» del plan.
    cuantas = {m: len(variantes_de(m)) for m in MODELOS_ABLACION}
    assert cuantas == {"nb_gaussiano": 11, "svm": 11, "knn": 12, "rf": 10}
    for modelo in MODELOS_ABLACION:
        assert variantes_de(modelo)[0] == BASE
        for nombre in variantes_de(modelo):
            crear_modelo(modelo, VARIANTES[nombre].opciones, **hiperparametros(modelo))
    print("ok  A0 a A12 cambian lo que dice la tabla del plan (A8 = A7 sin month), en sus modelos")


def test_rapido_recorta_y_no_escribe_en_resultados():
    assert hiperparametros("rf", rapido=True) == {"n_estimators": 20}
    for modelo in ("nb_gaussiano", "svm", "knn"):
        assert hiperparametros(modelo, rapido=True) == REFERENCIA[modelo]
    for modelo in MODELOS_ABLACION:
        assert hiperparametros(modelo) == REFERENCIA[modelo]
    assert salida_por_defecto(rapido=False) == DIR
    rapida = salida_por_defecto(rapido=True).resolve()
    assert rapida != DIR.resolve() and DIR.resolve() not in rapida.parents
    print("ok  --rapido usa RF con 20 árboles y, sin --salida, escribe fuera de resultados/")


def test_el_oof_de_a0_no_pisa_el_de_experimentos():
    # src/experimentos.py escribe oof_<etiqueta>_<modelo>.csv (N0-5) con las etiquetas referencia y
    # final, y con la configuración vigente, que no es A0: si el de A0 se llamara
    # oof_referencia_<modelo>.csv, el último en correr pisaría al otro sin aviso.
    for modelo in MODELOS_ABLACION:
        assert ruta_oof(DIR, modelo) == DIR / f"oof_ablacion_A0_{modelo}.csv", modelo
    print("ok  el OOF de A0 va a oof_ablacion_A0_<modelo>.csv, no al oof_referencia_<modelo>.csv "
          "de src/experimentos.py")


def test_submuestra_estratificada(train):
    a, b = submuestra(train), submuestra(train)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == N_RAPIDO and a[FILA].is_unique and a[FILA].is_monotonic_increasing
    tasa = (train[OBJETIVO] == "yes").mean()
    assert abs((a[OBJETIVO] == "yes").mean() - tasa) < 1 / N_RAPIDO
    print(f"ok  la submuestra de --rapido tiene {N_RAPIDO} filas, la tasa de «yes» de train "
          "y sale igual dos veces")


def test_columnas_por_variante_calculadas_a_mano(X):
    for nombre, variante in VARIANTES.items():
        for modelo in variante.modelos:
            cuantas = columnas_salida(modelo, variante.opciones, X)
            assert cuantas == COLUMNAS_A_MANO[nombre], (nombre, modelo, cuantas)
    print("ok  las columnas que produce cada variante coinciden con la cuenta a mano")


def test_cada_variante_se_mide_con_sus_opciones(X, y, filas):
    """medir_variante con un espía en lugar de validacion_cruzada: sin ajustar nada, mira qué
    Pipeline recibe cada variante de cada modelo, con y sin --rapido."""
    recibidos = []

    def espia(modelo, X_, y_, filas=None, n_jobs=None, particionador=None):
        recibidos.append((modelo, X_, y_, filas))
        tabla = pd.DataFrame({"fold": [1], "conjunto": ["validacion"], "metrica": ["auc"],
                              "valor": [0.5]})
        return tabla, pd.DataFrame({"fila": [0], "fold": [1], "puntaje": [0.0]})

    original = ablaciones.validacion_cruzada
    ablaciones.validacion_cruzada = espia
    try:
        for modelo in MODELOS_ABLACION:
            for rapido in (False, True):
                # Escritos aparte de hiperparametros(): los de REFERENCIA, y en --rapido RF con 20
                # árboles.
                esperados = dict(REFERENCIA[modelo])
                if rapido and modelo == "rf":
                    esperados["n_estimators"] = 20
                for nombre in variantes_de(modelo):
                    caso = (modelo, nombre, rapido)
                    tabla, _, _ = medir_variante(modelo, nombre, X, y, filas, rapido, n_jobs=1)
                    pipeline, X_, y_, filas_ = recibidos[-1]
                    derivadas = pipeline.named_steps["derivadas"]
                    assert derivadas.opciones == VARIANTES[nombre].opciones, caso
                    parametros = pipeline.named_steps["modelo"].get_params()
                    assert all(parametros[k] == v for k, v in esperados.items()), caso
                    assert X_ is X and filas_ is filas and np.array_equal(y_, y), caso
                    assert set(tabla["variante"]) == {nombre}, caso
                    assert set(tabla["modelo"]) == {modelo}, caso
                    assert set(tabla["configuracion"]) == {configuracion_json(esperados)}, caso
    finally:
        ablaciones.validacion_cruzada = original
    assert len(recibidos) == 2 * sum(len(variantes_de(m)) for m in MODELOS_ABLACION)
    print("ok  cada variante de cada modelo se mide con sus Opciones y los hiperparámetros de "
          "referencia (RF con 20 árboles en --rapido), sobre las filas que recibe")


def test_linea_base_empata_a_todos(X, y, filas):
    tabla = linea_base(X, y, filas, n_jobs=1)
    assert list(tabla.columns) == ["variante"] + CAMPOS
    assert set(tabla["variante"]) == {BASE} and set(tabla["modelo"]) == {"sin_modelo"}
    assert set(tabla["configuracion"]) == {'{"estrategia": "prior"}'}
    assert len(tabla) == K * 2 * len(METRICAS)
    for numero, (tr, va) in enumerate(folds().split(X, y), start=1):
        for conjunto, indices in (("train", tr), ("validacion", va)):
            m = tabla[(tabla["fold"] == numero) & (tabla["conjunto"] == conjunto)]
            m = m.set_index("metrica")["valor"]
            n, positivos = len(indices), y[indices].sum()
            assert m["auc"] == 0.5, (numero, conjunto)
            # Todos empatados: con k = llamadas(n) llamadas se espera la fracción k/n de los «yes».
            assert _cerca(m["recall_q"], llamadas(n) / n), (numero, conjunto)
            # Con un solo puntaje, la precisión en cualquier corte es la proporción de «yes».
            assert _cerca(m["precision_q"], positivos / n) and _cerca(m["ap"], positivos / n)
    print("ok  línea de base: AUC 0,5 exacto y recall_q = llamadas(n)/n en cada fold")


def _correr(*argumentos):
    r = subprocess.run([sys.executable, "-m", "src.ablaciones", *argumentos], cwd=RAIZ,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"falló {' '.join(argumentos)}:\n{r.stderr}"
    return r.stdout


def test_rapido_de_punta_a_punta(train):
    variantes = ["A0", "A2", "A7"]
    ayuda = _correr("--help")
    assert "--rapido" in ayuda and "--salida" in ayuda and all(v in ayuda for v in VARIANTES)
    with tempfile.TemporaryDirectory() as carpeta:
        salida = Path(carpeta)
        corrida = _correr("--modelo", "nb_gaussiano", "--variantes", ",".join(variantes),
                          "--rapido", "--salida", carpeta)
        _correr("--linea-base", "--rapido", "--salida", carpeta)
        resumida = _correr("--resumen", "--salida", carpeta)

        tabla = leer_largo(salida / "ablaciones_nb_gaussiano.csv")
        assert list(tabla.columns) == ["variante"] + CAMPOS
        assert list(tabla["variante"].unique()) == variantes
        for nombre, grupo in tabla.groupby("variante"):
            assert len(grupo) == K * 2 * len(METRICAS), nombre
            assert not grupo.duplicated(["fold", "conjunto", "metrica"]).any(), nombre
            assert sorted(grupo["fold"].unique()) == list(range(1, K + 1))
            assert set(grupo["conjunto"]) == {"train", "validacion"}
            assert set(grupo["metrica"]) == set(METRICAS)
            assert re.search(rf"^\s+{nombre}\s.*\s\d+\.\d s\s", corrida, re.M), nombre
        # El separador de miles es un espacio, no la coma del inglés.
        assert f"{N_RAPIDO:,} filas".replace(",", " ") in corrida
        pd.testing.assert_frame_equal(leer_largo(salida / "ablaciones.csv"), tabla)
        base = leer_largo(salida / "linea_base.csv")
        assert list(base.columns) == ["variante"] + CAMPOS and len(base) == K * 2 * len(METRICAS)

        validacion = tabla[tabla["conjunto"] == "validacion"]
        auc = validacion[validacion["metrica"] == "auc"].pivot(index="fold", columns="variante",
                                                                values="valor")
        recall = validacion[validacion["metrica"] == "recall_q"].pivot(
            index="fold", columns="variante", values="valor")
        # Si una variante se midiera con las opciones de A0, daría su mismo AUC en cada fold.
        for nombre in ("A2", "A7"):
            assert (auc[nombre] != auc[BASE]).any(), nombre

        # El OOF es el de A0, con la fila del CSV original: cubre la submuestra una vez, cada fold
        # tiene las filas de validación de folds() y su AUC es el de A0 en la tabla.
        muestra = submuestra(train)
        oof = pd.read_csv(ruta_oof(salida, "nb_gaussiano"))
        assert list(oof.columns) == CAMPOS_OOF and len(oof) == N_RAPIDO and oof[FILA].is_unique
        assert not (salida / "oof_referencia_nb_gaussiano.csv").exists()
        X_m, y_m = separar_X_y(muestra)
        for numero, (_, va) in enumerate(folds().split(X_m, y_m), start=1):
            assert set(oof.loc[oof["fold"] == numero, FILA]) == set(muestra[FILA].iloc[va]), numero
        con_y = oof.merge(muestra[[FILA, OBJETIVO]], on=FILA, validate="one_to_one")
        for numero, grupo in con_y.groupby("fold"):
            desde_oof = roc_auc_score(grupo[OBJETIVO] == "yes", grupo["puntaje"])
            assert _cerca(desde_oof, auc.loc[numero, BASE]), numero

        # El resumen tiene una fila por variante distinta de A0.
        r = pd.read_csv(salida / "ablaciones_resumen.csv")
        assert list(r.columns) == COLUMNAS_RESUMEN and list(r["variante"]) == ["A2", "A7"]
        assert set(r["efecto"]) <= EFECTOS
        r = r.set_index("variante")
        for nombre in ("A2", "A7"):
            deltas = auc[nombre] - auc[BASE]
            assert _cerca(r.loc[nombre, "delta_media"], deltas.mean()), nombre
            assert _cerca(r.loc[nombre, "delta_desvio"], deltas.std(ddof=1)), nombre
            assert _cerca(r.loc[nombre, "auc_variante"], auc[nombre].mean())
            assert _cerca(r.loc[nombre, "auc_referencia"], auc[BASE].mean())
            assert _cerca(r.loc[nombre, "delta_recall_media"],
                          (recall[nombre] - recall[BASE]).mean())
            assert r.loc[nombre, "n_folds"] == K
            assert r.loc[nombre, "efecto"] == efecto(deltas.mean(), deltas.std(ddof=1))
        assert r["columnas"].to_dict() == {v: COLUMNAS_A_MANO[v] for v in ("A2", "A7")}
        referencia = f"Columnas de {BASE}, la referencia: nb_gaussiano {COLUMNAS_A_MANO[BASE]}"
        assert referencia in resumida
    print("ok  --rapido de punta a punta: 5 folds × 2 conjuntos × 5 métricas por variante, tiempos "
          "impresos, OOF de A0 con sus filas y folds, y ΔAUC del resumen = media de las "
          "diferencias fold a fold")


def test_una_corrida_cortada_se_completa_con_variantes():
    original = ablaciones.medir_variante

    def se_corta_en_a7(modelo, nombre, *args, **kwargs):
        if nombre == "A7":
            raise RuntimeError("corte simulado")
        return original(modelo, nombre, *args, **kwargs)

    with tempfile.TemporaryDirectory() as carpeta, contextlib.redirect_stdout(io.StringIO()):
        salida = Path(carpeta)
        ruta = salida / "ablaciones_nb_gaussiano.csv"
        ablaciones.medir_variante = se_corta_en_a7
        try:
            ablaciones.correr_modelo("nb_gaussiano", ["A0", "A2", "A7"], parcial=False,
                                     rapido=True, n_jobs=1, salida=salida)
        except RuntimeError:
            pass
        else:
            raise AssertionError("el corte simulado no ocurrió")
        finally:
            ablaciones.medir_variante = original
        assert list(leer_largo(ruta)["variante"].unique()) == ["A0", "A2"]
        assert ruta_oof(salida, "nb_gaussiano").exists()

        main_ablaciones(["--modelo", "nb_gaussiano", "--variantes", "A7", "--rapido",
                         "--salida", carpeta])
        tabla = leer_largo(ruta)
        assert list(tabla["variante"].unique()) == ["A0", "A2", "A7"]
        assert len(tabla) == 3 * K * 2 * len(METRICAS)

        # Un archivo medido con otros hiperparámetros no se completa: mezclaría configuraciones.
        tabla.assign(configuracion='{"var_smoothing": 0.001}').to_csv(ruta, index=False)
        antes = ruta.read_bytes()
        try:
            main_ablaciones(["--modelo", "nb_gaussiano", "--variantes", "A2", "--rapido",
                             "--salida", carpeta])
        except SystemExit as e:
            assert "hiperparámetros" in str(e.code)
        else:
            raise AssertionError("completó un archivo medido con otros hiperparámetros")
        assert ruta.read_bytes() == antes
    print("ok  una corrida cortada conserva lo medido y se completa con --variantes; "
          "con otros hiperparámetros se niega")


def test_no_completa_un_archivo_medido_sobre_otras_filas():
    """--variantes sólo completa un archivo medido sobre las mismas filas, las que registra el OOF
    de A0. Cada negativa deja la carpeta como estaba."""
    with tempfile.TemporaryDirectory() as carpeta, contextlib.redirect_stdout(io.StringIO()):
        salida = Path(carpeta)

        def se_niega(argumentos, texto):
            antes = {p.name: p.read_bytes() for p in salida.iterdir()}
            try:
                main_ablaciones([*argumentos, "--salida", carpeta])
            except SystemExit as e:
                assert texto in str(e.code), (argumentos, e.code)
            else:
                raise AssertionError(f"aceptó {argumentos}")
            assert {p.name: p.read_bytes() for p in salida.iterdir()} == antes, argumentos

        # Sin archivo previo, una corrida sin A0 no tiene contra qué compararse.
        se_niega(["--modelo", "nb_gaussiano", "--variantes", "A2", "--rapido"],
                 f"tiene que incluir {BASE}")
        main_ablaciones(["--modelo", "nb_gaussiano", "--variantes", "A0", "--rapido",
                         "--salida", carpeta])
        # El archivo es de la submuestra de --rapido: una corrida sobre train completo no lo
        # completa, porque aparearía folds de datos distintos.
        se_niega(["--modelo", "nb_gaussiano", "--variantes", "A2", "--n-jobs", "1"],
                 "otras filas")
        # Sobre las mismas filas, sí.
        main_ablaciones(["--modelo", "nb_gaussiano", "--variantes", "A2", "--rapido",
                         "--salida", carpeta])
        ruta = salida / "ablaciones_nb_gaussiano.csv"
        assert list(leer_largo(ruta)["variante"].unique()) == ["A0", "A2"]

        # Una corrida cortada entre la escritura del archivo y la del OOF de A0 no deja el OOF
        # anterior, que podría ser de otras filas.
        original = ablaciones.escribir_oof

        def se_corta(*args, **kwargs):
            raise RuntimeError("corte simulado")

        ablaciones.escribir_oof = se_corta
        try:
            ablaciones.correr_modelo("nb_gaussiano", [BASE], parcial=False, rapido=True,
                                     n_jobs=1, salida=salida)
        except RuntimeError:
            pass
        else:
            raise AssertionError("el corte simulado no ocurrió")
        finally:
            ablaciones.escribir_oof = original
        assert ruta.exists() and not ruta_oof(salida, "nb_gaussiano").exists()
        # Y sin el OOF de A0 no se sabe sobre qué filas se midió el archivo.
        se_niega(["--modelo", "nb_gaussiano", "--variantes", "A7", "--rapido"], "Falta")
    print("ok  --variantes no mezcla filas: exige A0 en la primera corrida y no completa con train "
          "completo un archivo de --rapido, ni uno sin el OOF de A0, que un corte no deja viejo")


def _tabla_sintetica():
    """Dos modelos con AUC de validación elegidos para que cada efecto se calcule a mano. El recall
    es 0,3 + 2·(AUC − 0,7), así que sus diferencias son el doble de las del AUC. Las filas de train
    llevan valores absurdos: si el resumen las usara, las cuentas no darían."""
    auc = {
        ("knn", "A0"): [0.70, 0.72, 0.74, 0.71, 0.73],
        # Δ = +0,02 en cada fold: media 0,02 y desvío 0 -> mejora
        ("knn", "A1"): [0.72, 0.74, 0.76, 0.73, 0.75],
        # Δ = -0,03, -0,01, -0,02, -0,02, -0,02: media -0,02, desvío √(0,0002/4) ≈ 0,0071 -> empeora
        ("knn", "A2"): [0.67, 0.71, 0.72, 0.69, 0.71],
        # Δ = +0,03, -0,01, +0,02, -0,01, +0,02: media 0,01, desvío √(0,0014/4) ≈ 0,0187 -> neutra
        ("knn", "A3"): [0.73, 0.71, 0.76, 0.70, 0.75],
        ("rf", "A0"): [0.80, 0.80, 0.80, 0.80, 0.80],
        # Δ = +0,01, -0,01, +0,02, -0,02, 0: media 0, desvío √(0,001/4) ≈ 0,0158 -> neutra
        ("rf", "A2"): [0.81, 0.79, 0.82, 0.78, 0.80],
    }
    configuracion = {"knn": '{"n_neighbors": 15}', "rf": '{"n_estimators": 300}'}
    filas = []
    for (modelo, variante), valores in auc.items():
        for fold, v in enumerate(valores, start=1):
            comun = {"variante": variante, "modelo": modelo,
                     "configuracion": configuracion[modelo], "fold": fold}
            filas += [
                {**comun, "conjunto": "validacion", "metrica": "auc", "valor": v},
                {**comun, "conjunto": "validacion", "metrica": "recall_q",
                 "valor": 0.3 + 2 * (v - 0.7)},
                {**comun, "conjunto": "train", "metrica": "auc",
                 "valor": 0.99 if variante == BASE else 0.01},
                {**comun, "conjunto": "train", "metrica": "recall_q", "valor": 0.5},
            ]
    return pd.DataFrame(filas)[["variante"] + CAMPOS]


def test_efecto_calculado_a_mano():
    assert efecto(0.02, 0.01) == "mejora" and efecto(-0.02, 0.01) == "empeora"
    # El borde es neutro: hay que superar el desvío, no igualarlo.
    assert efecto(0.01, 0.01) == "neutra" and efecto(-0.01, 0.01) == "neutra"
    assert efecto(0.0, 0.0) == "neutra"

    r = resumir(_tabla_sintetica())
    assert list(r.columns) == [c for c in COLUMNAS_RESUMEN if c != "columnas"]
    # A0 no lleva fila propia: es la referencia de las demás.
    assert list(zip(r["modelo"], r["variante"])) == [
        ("knn", "A1"), ("knn", "A2"), ("knn", "A3"), ("rf", "A2")]
    assert set(r["efecto"]) <= EFECTOS
    r = r.set_index(["modelo", "variante"])
    esperados = {
        ("knn", "A1"): (0.02, 0.0, "mejora"),
        ("knn", "A2"): (-0.02, np.sqrt(0.0002 / 4), "empeora"),
        ("knn", "A3"): (0.01, np.sqrt(0.0014 / 4), "neutra"),
        ("rf", "A2"): (0.0, np.sqrt(0.001 / 4), "neutra"),
    }
    for clave, (media, desvio, esperado) in esperados.items():
        fila = r.loc[clave]
        assert _cerca(fila["delta_media"], media, 1e-9), clave
        assert _cerca(fila["delta_desvio"], desvio, 1e-9), clave
        assert _cerca(fila["delta_error_estandar"], desvio / np.sqrt(5), 1e-9), clave
        assert _cerca(fila["delta_recall_media"], 2 * media, 1e-9), clave
        assert _cerca(fila["delta_recall_desvio"], 2 * desvio, 1e-9), clave
        assert fila["n_folds"] == 5 and fila["efecto"] == esperado, clave
    # knn: AUC medio de A0 = 3,60/5 = 0,72 y de A2 = 3,50/5 = 0,70; Δ entre -0,03 y -0,01.
    fila = r.loc[("knn", "A2")]
    assert _cerca(fila["auc_referencia"], 0.72, 1e-9) and _cerca(fila["auc_variante"], 0.70, 1e-9)
    assert _cerca(fila["delta_min"], -0.03, 1e-9) and _cerca(fila["delta_max"], -0.01, 1e-9)
    print("ok  mejora, neutra y empeora coinciden con la cuenta a mano; train no entra al resumen")


def test_el_resumen_no_junta_modelos_medidos_sobre_otras_filas():
    t = _tabla_sintetica()
    with tempfile.TemporaryDirectory() as carpeta:
        salida = Path(carpeta)
        ruta = salida / "ablaciones_resumen.csv"

        def escribir(modelo, filas, tabla=t):
            escribir_largo(salida / f"ablaciones_{modelo}.csv", tabla[tabla["modelo"] == modelo])
            pd.DataFrame({FILA: filas, "fold": 1, "puntaje": 0.0}).to_csv(
                ruta_oof(salida, modelo), index=False)

        def resumir_carpeta():
            impreso = io.StringIO()
            with contextlib.redirect_stdout(impreso):
                correr_resumen(salida)
            return impreso.getvalue()

        def se_niega(texto):
            antes = ruta.read_bytes()
            try:
                resumir_carpeta()
            except SystemExit as e:
                assert texto in str(e.code), e.code
            else:
                raise AssertionError(f"resumió en lugar de decir «{texto}»")
            assert ruta.read_bytes() == antes

        escribir("knn", [3, 5, 8])
        escribir("rf", [3, 5, 8])
        impreso = resumir_carpeta()
        r = pd.read_csv(ruta)
        # Los efectos de test_efecto_calculado_a_mano, y las columnas de cada variante.
        assert list(zip(r["modelo"], r["variante"], r["efecto"])) == [
            ("knn", "A1", "mejora"), ("knn", "A2", "empeora"), ("knn", "A3", "neutra"),
            ("rf", "A2", "neutra")]
        assert list(r["columnas"]) == [COLUMNAS_A_MANO[v] for v in r["variante"]]
        assert f"Columnas de {BASE}, la referencia: knn 62, rf 62" in impreso

        # rf medido sobre otras filas: no se junta con knn.
        escribir("rf", [3, 5, 9])
        se_niega("filas distintas")
        # Sin la A0 de rf: un error que lo dice, en lugar de un traceback.
        escribir("rf", [3, 5, 8], t[t["variante"] != BASE])
        se_niega(f"falta la referencia {BASE} de: rf")
        # Sin el OOF de rf no se puede comprobar: resume y lo avisa.
        escribir("rf", [3, 5, 8])
        ruta_oof(salida, "rf").unlink()
        assert f"Aviso: sin OOF de {BASE} en rf" in resumir_carpeta()
    print("ok  el resumen no junta modelos medidos sobre filas distintas y, sin A0, lo dice sin "
          "traceback")


def test_reemplazar_y_ordenar_variantes():
    t = _tabla_sintetica()
    knn = t[t["modelo"] == "knn"].reset_index(drop=True)
    nueva = knn[knn["variante"] == "A2"].assign(valor=lambda d: d["valor"] + 0.001)
    combinada = reemplazar_variantes(knn, nueva)
    assert list(combinada["variante"].unique()) == ["A0", "A1", "A2", "A3"]
    assert len(combinada) == len(knn)
    pd.testing.assert_frame_equal(combinada[combinada["variante"] == "A2"].reset_index(drop=True),
                                  nueva.reset_index(drop=True))
    pd.testing.assert_frame_equal(combinada[combinada["variante"] != "A2"].reset_index(drop=True),
                                  knn[knn["variante"] != "A2"].reset_index(drop=True))
    try:
        reemplazar_variantes(knn, nueva.assign(configuracion='{"n_neighbors": 5}'))
    except ValueError:
        pass
    else:
        raise AssertionError("mezcló variantes con hiperparámetros distintos")

    desordenada = pd.DataFrame({"modelo": ["rf", "knn", "rf", "knn", "knn"],
                                "variante": ["A2", "A12", "A0", "A10", "A2"], "fila": range(5)})
    o = ordenar(desordenada)
    assert list(zip(o["modelo"], o["variante"])) == [
        ("knn", "A2"), ("knn", "A10"), ("knn", "A12"), ("rf", "A0"), ("rf", "A2")]
    print("ok  volver a correr una variante reemplaza sólo la suya, y A2 va antes que A10")


def test_comprobar_rechaza_lo_que_no_se_puede_resumir():
    t = _tabla_sintetica()
    comprobar(t)
    casos = {
        "sin A0 de rf": t[~((t["modelo"] == "rf") & (t["variante"] == BASE))],
        "filas repetidas": pd.concat([t, t.head(1)], ignore_index=True),
        "dos configuraciones": t.assign(configuracion=np.where(t["variante"] == "A1", '{"a": 1}',
                                                                t["configuracion"])),
        "variante desconocida": t.assign(variante=t["variante"].replace("A3", "A99")),
        "folds distintos": t[~((t["variante"] == "A2") & (t["fold"] == 5))],
        "sin columna variante": t.drop(columns="variante"),
    }
    for nombre, tabla in casos.items():
        try:
            comprobar(tabla)
        except ValueError:
            continue
        raise AssertionError(f"aceptó una tabla con {nombre}")
    print(f"ok  el resumen rechaza {len(casos)} tablas que darían diferencias sin sentido")


def test_la_linea_de_comandos_rechaza_combinaciones_invalidas():
    casos = [
        [],
        ["--modelo", "rf", "--linea-base"],
        ["--modelo", "rf", "--variantes", "A12"],
        ["--modelo", "knn", "--variantes", "A99"],
        ["--modelo", "knn", "--variantes", ","],
        ["--resumen", "--variantes", "A0"],
        ["--modelo", "rf", "--n-jobs", "0"],
        ["--modelo", "rf", "--rapido", "--salida", str(DIR)],
        ["--linea-base", "--rapido", "--salida", str(DIR / "rapido")],
    ]
    for argumentos in casos:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                main_ablaciones(argumentos)
            except SystemExit as e:
                assert e.code == 2, argumentos
                continue
        raise AssertionError(f"aceptó {argumentos}")
    print(f"ok  la línea de comandos rechaza {len(casos)} combinaciones inválidas antes de medir")


def main():
    train = cargar_train()
    muestra = submuestra(train)
    X, y = separar_X_y(muestra)
    filas = muestra[FILA].to_numpy()
    test_las_variantes_son_las_del_plan()
    test_rapido_recorta_y_no_escribe_en_resultados()
    test_el_oof_de_a0_no_pisa_el_de_experimentos()
    test_submuestra_estratificada(train)
    test_columnas_por_variante_calculadas_a_mano(X)
    test_cada_variante_se_mide_con_sus_opciones(X, y, filas)
    test_linea_base_empata_a_todos(X, y, filas)
    test_rapido_de_punta_a_punta(train)
    test_una_corrida_cortada_se_completa_con_variantes()
    test_no_completa_un_archivo_medido_sobre_otras_filas()
    test_efecto_calculado_a_mano()
    test_el_resumen_no_junta_modelos_medidos_sobre_otras_filas()
    test_reemplazar_y_ordenar_variantes()
    test_comprobar_rechaza_lo_que_no_se_puede_resumir()
    test_la_linea_de_comandos_rechaza_combinaciones_invalidas()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
