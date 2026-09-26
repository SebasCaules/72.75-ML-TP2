"""Correr con: python -m tests.test_seleccion

Cada caso se arma con CSV y JSON sintéticos en un directorio temporal, escritos con las funciones
del contrato (`con_identidad`, `escribir_largo`), y sus números están calculados a mano en el
comentario que lo acompaña. Ningún caso ajusta modelos.
"""

import io
import json
import math
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES, OPCIONES_FINALES
from src.datos import SEMILLA, cargar_train
from src.estilo import ROTULO_LINEA_BASE
from src.metricas import METRICAS
from src.modelos import REFERENCIA
from src.resultados import DIR, con_identidad, escribir_largo
from src.seleccion import ETIQUETA_RAPIDA, MODELOS_RAPIDOS, N_RAPIDO, SIN_MODELO, seleccionar
from src.seleccion import main as linea_de_comandos
from src.validacion import K

# AUC de validación del mejor modelo de los casos: media 0,80; desvío muestral √(0,0002 / 4) =
# 0,01/√2; error estándar 0,01/√2/√5 = 0,01/√10 ≈ 0,00316. El umbral de empate queda en 0,79684.
AUC_MEJOR = [0.80, 0.81, 0.79, 0.80, 0.80]
DESVIO_MEJOR = 0.01 / math.sqrt(2)
ES_MEJOR = 0.01 / math.sqrt(10)
RELLENO = {"ap": 0.40, "precision_q": 0.30, "f1_q": 0.35}
CLAVES = {"modelo", "hiperparametros", "opciones", "metricas", "brecha_auc", "empate_dentro_1es",
          "desempate_por_recall", "linea_base", "ranking", "n_train", "semilla", "k_folds"}


def _cerca(a, b, tol=1e-12):
    return abs(a - b) < tol


def _tabla(modelo, auc_va, recall_va, hiper=None, etiqueta="final", folds=None):
    """Las filas largas de un modelo: en validación, el AUC de cada fold y un recall_q fijo; en
    train, los dos 0,05 más altos; las otras tres métricas, constantes."""
    folds = list(range(1, K + 1)) if folds is None else folds
    hiper = HIPERPARAMETROS_FINALES.get(modelo, {}) if hiper is None else hiper
    filas = []
    for fold, auc in zip(folds, auc_va):
        for conjunto, delta in (("train", 0.05), ("validacion", 0.0)):
            valores = {"auc": auc + delta, "recall_q": recall_va + delta, **RELLENO}
            filas += [{"fold": fold, "conjunto": conjunto, "metrica": m, "valor": valores[m]}
                      for m in METRICAS]
    extra = {} if etiqueta is None else {"etiqueta": etiqueta}
    return con_identidad(pd.DataFrame(filas), modelo, hiper, **extra)


def _escribir_cv(directorio, *tablas, etiqueta="final"):
    escribir_largo(Path(directorio) / f"cv_{etiqueta}.csv", pd.concat(tablas, ignore_index=True))


def _escribir_linea_base(directorio, auc=0.5, recall=0.2):
    escribir_largo(Path(directorio) / "linea_base.csv",
                   _tabla(SIN_MODELO, [auc] * K, recall, hiper={}, etiqueta=None))


def _falla(funcion, excepcion, *fragmentos):
    try:
        funcion()
    except excepcion as error:
        mensaje = str(error)
        assert all(f in mensaje for f in fragmentos), mensaje
        return
    raise AssertionError(f"se esperaba {excepcion.__name__}")


def test_sin_empate_gana_el_mayor_auc():
    # rf: AUC 0,80 y umbral 0,79684. knn (0,78), nb_categorico (0,77) y nb_gaussiano (0,75) quedan
    # por debajo, así que decide el AUC aunque knn tenga más recall_q (0,70 contra 0,60). Las dos
    # variantes de Naive Bayes compiten por separado.
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60), _tabla("knn", [0.78] * K, 0.70),
                     _tabla("nb_categorico", [0.77] * K, 0.50),
                     _tabla("nb_gaussiano", [0.75] * K, 0.55),
                     _tabla(SIN_MODELO, [0.5] * K, 0.2, hiper={}))
        _escribir_linea_base(d)
        e = seleccionar(d, modelos=["rf", "knn", "nb_categorico", "nb_gaussiano"], n_train=100)
    assert e["modelo"] == "rf"
    assert e["empate_dentro_1es"] == [] and e["desempate_por_recall"] is False
    assert [f["modelo"] for f in e["ranking"]] == ["rf", "knn", "nb_categorico", "nb_gaussiano"]
    assert _cerca(e["ranking"][0]["auc"], 0.80)
    assert _cerca(e["ranking"][0]["auc_error_estandar"], ES_MEJOR)
    assert _cerca(e["ranking"][1]["recall_q"], 0.70)
    print("ok  sin empate gana el mayor AUC (0,80), aunque otro tenga más recall_q; "
          "NB compite por variante")


def test_empate_dentro_de_1_es_se_resuelve_por_recall():
    # El ES que cuenta es el del mejor (rf, 0,00316), no el de cada candidato:
    # - svm: AUC 0,798 en los cinco folds (ES propio 0), a 0,002 de rf. Empata.
    # - knn: AUC [0,75, 0,83, 0,79, 0,79, 0,79], media 0,79; desvío √(0,0032 / 4) = 0,0283 y ES
    #   propio 0,0126. Queda a 0,01 de rf, más que 0,00316: no empata, aunque 0,01 sea menos que
    #   su propio ES y su recall_q (0,70) sea el mayor de todos.
    # Entre rf (0,60) y svm (0,62) decide el recall_q: gana svm.
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60), _tabla("svm", [0.798] * K, 0.62),
                     _tabla("knn", [0.75, 0.83, 0.79, 0.79, 0.79], 0.70),
                     _tabla(SIN_MODELO, [0.5] * K, 0.2, hiper={}))
        _escribir_linea_base(d)
        e = seleccionar(d, modelos=["rf", "svm", "knn"], n_train=100)
    assert e["modelo"] == "svm"
    assert e["empate_dentro_1es"] == ["rf", "svm"] and e["desempate_por_recall"] is True
    assert [f["modelo"] for f in e["ranking"]] == ["rf", "svm", "knn"]
    assert _cerca(e["ranking"][1]["auc"], 0.798) and _cerca(e["ranking"][2]["auc"], 0.79)
    assert _cerca(e["ranking"][2]["auc_error_estandar"], math.sqrt(0.0008 / 5))
    assert _cerca(e["metricas"]["validacion"]["recall_q"]["media"], 0.62)
    print("ok  svm, a 0,002 de rf con ES 0,00316, empata y gana por recall_q (0,62 contra 0,60); "
          "knn, a 0,01 y con más recall_q, no: el ES es el del mejor")


def test_sin_modelo_no_puede_ganar():
    # sin_modelo figura en el CSV con AUC 0,95 y recall_q 0,90, más que cualquier modelo, pero es
    # la línea de base: con la lista por defecto no entra al ranking ni al empate, y nombrarlo en
    # la lista es un error. La línea de base del JSON sale de linea_base.csv (0,5 y 0,2) y, como
    # no coincide con la del CSV, queda un aviso.
    with tempfile.TemporaryDirectory() as d:
        primero = MODELOS_FINALES[0]
        tablas = [_tabla(primero, AUC_MEJOR, 0.60)]
        tablas += [_tabla(m, [0.78] * K, 0.70) for m in MODELOS_FINALES[1:]]
        _escribir_cv(d, *tablas, _tabla(SIN_MODELO, [0.95] * K, 0.90, hiper={}))
        _escribir_linea_base(d)
        e = seleccionar(d, n_train=100)
        _falla(lambda: seleccionar(d, modelos=["rf", SIN_MODELO], n_train=100), ValueError,
               SIN_MODELO, "no compite")
    assert e["modelo"] == primero and SIN_MODELO not in [f["modelo"] for f in e["ranking"]]
    assert len(e["ranking"]) == len(MODELOS_FINALES)
    assert SIN_MODELO not in e["empate_dentro_1es"]
    assert _cerca(e["linea_base"]["auc"], 0.5) and _cerca(e["linea_base"]["recall_q"], 0.2)
    assert e["fuentes"]["linea_base"].endswith("linea_base.csv")
    assert any("linea_base.csv" in a and "no coincide" in a for a in e["avisos"])
    print("ok  sin_modelo, aun con AUC 0,95 en el CSV, no compite; la línea de base es la de "
          "linea_base.csv y la diferencia queda como aviso")


def test_linea_base_desde_el_cv_si_falta_el_archivo():
    # Sin linea_base.csv, vale sin_modelo del CSV: AUC 0,5 y recall_q 0,2. Sin ninguna de las dos,
    # no hay contra qué comparar y la selección se detiene.
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60), _tabla(SIN_MODELO, [0.5] * K, 0.2, hiper={}))
        e = seleccionar(d, modelos=["rf"], n_train=100)
    assert _cerca(e["linea_base"]["auc"], 0.5) and _cerca(e["linea_base"]["recall_q"], 0.2)
    assert e["fuentes"]["linea_base"].endswith(f"({SIN_MODELO})") and e["avisos"] == []
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60))
        _falla(lambda: seleccionar(d, modelos=["rf"], n_train=100), FileNotFoundError,
               "línea de base")
    print("ok  sin linea_base.csv la línea de base sale de sin_modelo del CSV; sin ninguna, error")


def test_hiperparametros_del_json_o_los_vigentes():
    # Sin hiperparametros.json valen los de src/configuracion.py sobre la referencia. Con el
    # archivo, con la forma que escribe src/curvas.py --elegir, los de "modelos" completan lo que
    # la configuración vigente no fija, pero la vigente gana donde los dos hablan (N0-11, N0-12);
    # un modelo que el archivo no nombra (knn) conserva los vigentes y sus "avisos" pasan a la
    # salida.
    vigentes_rf = {**REFERENCIA["rf"], **HIPERPARAMETROS_FINALES["rf"]}
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60), _tabla("knn", [0.78] * K, 0.70))
        _escribir_linea_base(d)
        sin_json = seleccionar(d, modelos=["rf", "knn"], n_train=100)

    de_la_ola_4 = {"n_estimators": 400, "max_depth": None}
    de_curvas = {"criterio": "D-22", "modelos": {"rf": de_la_ola_4},
                 "curvas": {"rf_max_depth": {"modelo": "rf", "parametro": "max_depth"}},
                 "avisos": ["rf_n_estimators se midió con otra max_depth"]}
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60, hiper={**de_la_ola_4, **vigentes_rf}),
                     _tabla("knn", [0.78] * K, 0.70))
        _escribir_linea_base(d)
        (Path(d) / "hiperparametros.json").write_text(json.dumps(de_curvas))
        con_json = seleccionar(d, modelos=["rf", "knn"], n_train=100)
        assert con_json["fuentes"]["hiperparametros"] == (
            f"{(Path(d) / 'hiperparametros.json').resolve()} sobre src/configuracion.py")

    assert sin_json["hiperparametros"] == vigentes_rf
    assert sin_json["fuentes"]["hiperparametros"] == "src/configuracion.py"
    assert con_json["hiperparametros"] == {**de_la_ola_4, **vigentes_rf}
    assert con_json["hiperparametros"]["n_estimators"] == vigentes_rf["n_estimators"]
    assert con_json["hiperparametros"]["max_depth"] is None
    assert '"max_depth": null' in json.dumps(con_json["hiperparametros"])
    assert con_json["avisos"] == ["hiperparametros.json: rf_n_estimators se midió con otra "
                                  "max_depth"]
    print("ok  hiperparámetros: los vigentes sin hiperparametros.json; con el de src/curvas.py, "
          "los de \"modelos\" completan y los vigentes ganan; sus avisos en la salida")


def test_configuracion_medida_distinta_de_la_declarada_es_error():
    # La configuración vigente (main() la fija a la referencia: rf con 300 árboles y nada más)
    # prevalece sobre hiperparametros.json, que sólo completa lo que ella no fija (N0-13). El CSV
    # midió rf con la vigente. Si el JSON agrega max_depth = 10, que la vigente no fija, declararía
    # un modelo que no se validó, y la selección se detiene. En cambio, los 400 árboles de otro JSON
    # quedan pisados por los 300 vigentes, que son los medidos: no es error, y se declaran 300.
    # Y 300 guardado como texto (un np.int64 pasa a "300" en configuracion_json) es el mismo valor.
    vigentes_rf = {**REFERENCIA["rf"], **HIPERPARAMETROS_FINALES["rf"]}
    assert vigentes_rf == {"n_estimators": 300}, "main() fija la vigente a la referencia"
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60, hiper=vigentes_rf))
        _escribir_linea_base(d)
        (Path(d) / "hiperparametros.json").write_text(json.dumps({"rf": {"max_depth": 10}}))
        _falla(lambda: seleccionar(d, modelos=["rf"], n_train=100), ValueError,
               'rf: medida {"n_estimators": 300}, declarada {"max_depth": 10, "n_estimators": 300}')

        (Path(d) / "hiperparametros.json").write_text(json.dumps({"rf": {"n_estimators": 400}}))
        pisado = seleccionar(d, modelos=["rf"], n_train=100)
    assert pisado["hiperparametros"] == {"n_estimators": 300}

    with tempfile.TemporaryDirectory() as d:
        tabla = _tabla("rf", AUC_MEJOR, 0.60, hiper={"n_estimators": np.int64(300)})
        assert tabla["configuracion"].iloc[0] == '{"n_estimators": "300"}'
        _escribir_cv(d, tabla)
        _escribir_linea_base(d)
        (Path(d) / "hiperparametros.json").write_text(json.dumps({"rf": {"n_estimators": 300}}))
        e = seleccionar(d, modelos=["rf"], n_train=100)
    assert e["hiperparametros"]["n_estimators"] == 300
    print("ok  si hiperparametros.json agrega un valor que la vigente no fija y el CSV no midió, "
          "error; los 400 árboles del JSON quedan pisados por los 300 vigentes (N0-13); 300 y "
          "\"300\" son el mismo valor")


def test_validacion_cruzada_incompleta_es_error():
    # Un fold de menos, un modelo que falta o dos configuraciones del mismo modelo harían que la
    # media no fuera la de los K folds de una sola configuración. Un modelo del CSV que no está en
    # la lista no compite, y queda como aviso. Del CSV sólo se leen las filas de su etiqueta.
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR[:4], 0.60, folds=[1, 2, 3, 4]))
        _escribir_linea_base(d)
        _falla(lambda: seleccionar(d, modelos=["rf"], n_train=100), ValueError, "incompleta")
        _falla(lambda: seleccionar(d, modelos=["rf", "svm"], n_train=100), ValueError,
               "faltan", "svm")

    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60, hiper={"n_estimators": 300}),
                     _tabla("rf", AUC_MEJOR, 0.60, hiper={"n_estimators": 400}))
        _escribir_linea_base(d)
        _falla(lambda: seleccionar(d, modelos=["rf"], n_train=100), ValueError, "configuraciones")

    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60), _tabla("knn", [0.78] * K, 0.70),
                     _tabla("svm", [0.99] * K, 0.99), _tabla("rf", [0.99] * K, 0.99,
                                                              etiqueta="otra"))
        _escribir_linea_base(d)
        e = seleccionar(d, modelos=["rf", "knn"], n_train=100)
    assert e["modelo"] == "rf" and [f["modelo"] for f in e["ranking"]] == ["rf", "knn"]
    assert _cerca(e["ranking"][0]["auc"], 0.80)
    assert any("no compiten" in a and "svm" in a for a in e["avisos"])
    print("ok  folds faltantes, modelos ausentes o dos configuraciones detienen la selección; "
          "los modelos fuera de la lista y las otras etiquetas no compiten")


def test_aviso_si_la_validacion_corrio_sobre_otra_muestra():
    # El OOF del elegido tiene una fila por fila de la validación cruzada (N0-5). Con 7 filas y
    # n_train = 100 la validación no corrió sobre todo train, y queda un aviso; con 100, ninguno.
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60), _tabla("knn", [0.78] * K, 0.70))
        _escribir_linea_base(d)
        oof = Path(d) / "oof_final_rf.csv"
        pd.DataFrame({"fila": range(7), "fold": 1, "puntaje": 0.5}).to_csv(oof, index=False)
        con_aviso = seleccionar(d, modelos=["rf", "knn"], n_train=100)
        pd.DataFrame({"fila": range(100), "fold": 1, "puntaje": 0.5}).to_csv(oof, index=False)
        sin_aviso = seleccionar(d, modelos=["rf", "knn"], n_train=100)
    assert con_aviso["avisos"] == ["la validación cruzada de rf corrió sobre 7 filas "
                                   "(oof_final_rf.csv), no sobre las 100 de train"]
    assert sin_aviso["avisos"] == [] and sin_aviso["n_train"] == 100
    print("ok  si el OOF del elegido no tiene las filas de train, queda un aviso")


def _solo_tipos_json(valor):
    if isinstance(valor, dict):
        return all(isinstance(k, str) and _solo_tipos_json(v) for k, v in valor.items())
    if isinstance(valor, list):
        return all(_solo_tipos_json(v) for v in valor)
    if isinstance(valor, float):
        return math.isfinite(valor)
    return valor is None or type(valor) in (str, int, bool)


def test_json_tiene_todas_las_claves_y_tipos_validos():
    # Con la lista por defecto (MODELOS_FINALES) y n_train por defecto (las filas de train). El
    # primero de la lista tiene el AUC de validación 0,80 y el de train 0,85: brecha 0,05. Los
    # demás, 0,70 o menos.
    with tempfile.TemporaryDirectory() as d:
        primero = MODELOS_FINALES[0]
        tablas = [_tabla(primero, AUC_MEJOR, 0.60)]
        tablas += [_tabla(m, [0.70 - 0.01 * i] * K, 0.50) for i, m in enumerate(MODELOS_FINALES[1:])]
        _escribir_cv(d, *tablas, _tabla(SIN_MODELO, [0.5] * K, 0.2, hiper={}))
        _escribir_linea_base(d)
        e = seleccionar(d)

    assert CLAVES <= set(e), CLAVES - set(e)
    texto = json.dumps(e, allow_nan=False)
    assert json.loads(texto) == e and _solo_tipos_json(e)
    assert e["modelo"] == primero and isinstance(e["hiperparametros"], dict)
    assert e["opciones"] == asdict(OPCIONES_FINALES)
    for conjunto in ("validacion", "train"):
        assert set(e["metricas"][conjunto]) == set(METRICAS)
        for metrica in METRICAS:
            assert set(e["metricas"][conjunto][metrica]) == {"media", "desvio", "error_estandar"}
    va, tr = e["metricas"]["validacion"], e["metricas"]["train"]
    assert _cerca(va["auc"]["media"], 0.80) and _cerca(va["auc"]["desvio"], DESVIO_MEJOR)
    assert _cerca(va["auc"]["error_estandar"], ES_MEJOR) and _cerca(tr["auc"]["media"], 0.85)
    assert _cerca(va["ap"]["media"], 0.40) and _cerca(va["ap"]["desvio"], 0.0)
    assert _cerca(e["brecha_auc"], 0.05)
    assert isinstance(e["empate_dentro_1es"], list) and isinstance(e["desempate_por_recall"], bool)
    assert set(e["linea_base"]) == {"auc", "recall_q"}
    assert [f["modelo"] for f in e["ranking"]] == list(MODELOS_FINALES)
    assert all(a["auc"] >= b["auc"] for a, b in zip(e["ranking"], e["ranking"][1:]))
    assert e["n_train"] == len(cargar_train()) and type(e["n_train"]) is int
    assert e["semilla"] == SEMILLA and e["k_folds"] == K
    print(f"ok  el JSON tiene las {len(CLAVES)} claves, sólo tipos de JSON (json.dumps sin default "
          f"ni NaN) y n_train = {e['n_train']}, las filas de train")


def test_linea_de_comandos():
    with tempfile.TemporaryDirectory() as d:
        _escribir_cv(d, _tabla("rf", AUC_MEJOR, 0.60), _tabla("knn", [0.78] * K, 0.70),
                     _tabla(SIN_MODELO, [0.5] * K, 0.2, hiper={}))
        _escribir_linea_base(d)
        with redirect_stdout(io.StringIO()) as salida:
            linea_de_comandos(["--resultados", d, "--modelos", "rf", "knn"])
        escrito = json.loads((Path(d) / "modelo_elegido.json").read_text(encoding="utf-8"))
    assert escrito["modelo"] == "rf"
    assert "Modelo elegido: rf" in salida.getvalue() and ROTULO_LINEA_BASE in salida.getvalue()

    with redirect_stdout(io.StringIO()) as ayuda:
        _falla(lambda: linea_de_comandos(["--help"]), SystemExit)
    assert "--rapido" in ayuda.getvalue() and "--etiqueta" in ayuda.getvalue()

    with tempfile.TemporaryDirectory() as d:
        try:
            linea_de_comandos(["--resultados", d])
        except SystemExit as salida_con_error:
            assert str(salida_con_error.code).startswith("error: no existe")
        else:
            raise AssertionError("sin cv_final.csv la línea de comandos debería terminar con error")

    # --rapido no puede pisar la selección real: se niega antes de ajustar nada.
    destino = DIR / "prueba_rapido_no_debe_existir.json"
    with redirect_stderr(io.StringIO()) as errores:
        _falla(lambda: linea_de_comandos(["--rapido", "--salida", str(destino)]), SystemExit)
    assert "no escribe dentro de resultados/" in errores.getvalue() and not destino.exists()
    print("ok  la línea de comandos escribe modelo_elegido.json, imprime el ranking contra "
          "«sin modelo», tiene --help, termina con error si falta el CSV y --rapido no escribe "
          "en resultados/")


def test_rapido_termina_con_la_configuracion_vigente():
    # --rapido mide cada modelo con su configuración vigente, la de src/configuracion.py (este caso
    # corre antes de que main() la fije a la referencia), sobre 3 000 filas de train, y elige con
    # la misma verificación que la corrida real: lo medido tiene que ser lo declarado. Cuando medía
    # RF con 20 árboles y la vigente tenía 200, esa verificación lo detenía (N0-13).
    with tempfile.TemporaryDirectory() as d:
        destino = Path(d) / "elegido.json"
        with redirect_stdout(io.StringIO()) as salida:
            linea_de_comandos(["--rapido", "--salida", str(destino)])
        e = json.loads(destino.read_text(encoding="utf-8"))
    assert e["etiqueta"] == ETIQUETA_RAPIDA and e["n_train"] == N_RAPIDO
    assert sorted(f["modelo"] for f in e["ranking"]) == sorted(MODELOS_RAPIDOS)
    assert e["hiperparametros"] == {**REFERENCIA[e["modelo"]],
                                    **HIPERPARAMETROS_FINALES[e["modelo"]]}
    assert e["avisos"] == [] and f"Modelo elegido: {e['modelo']}" in salida.getvalue()
    print(f"ok  --rapido termina: mide {', '.join(MODELOS_RAPIDOS)} con la configuración vigente "
          f"sobre {N_RAPIDO} filas y elige {e['modelo']}, declarado con lo que se midió")


def main():
    # --rapido tiene que correr con la configuración vigente de verdad: va antes de fijarla.
    test_rapido_termina_con_la_configuracion_vigente()
    # La regla se prueba con tablas sintéticas medidas con la referencia de cada modelo, así que
    # la configuración vigente se fija a esa referencia: si no, un hiperparámetro elegido en la
    # ola 4 (por ejemplo max_depth) aparecería como «declarado» y no medido.
    import src.seleccion as modulo
    modulo.HIPERPARAMETROS_FINALES = {m: dict(v) for m, v in REFERENCIA.items()}
    HIPERPARAMETROS_FINALES.clear()
    HIPERPARAMETROS_FINALES.update(modulo.HIPERPARAMETROS_FINALES)
    test_sin_empate_gana_el_mayor_auc()
    test_empate_dentro_de_1_es_se_resuelve_por_recall()
    test_sin_modelo_no_puede_ganar()
    test_linea_base_desde_el_cv_si_falta_el_archivo()
    test_hiperparametros_del_json_o_los_vigentes()
    test_configuracion_medida_distinta_de_la_declarada_es_error()
    test_validacion_cruzada_incompleta_es_error()
    test_aviso_si_la_validacion_corrio_sobre_otra_muestra()
    test_json_tiene_todas_las_claves_y_tipos_validos()
    test_linea_de_comandos()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
