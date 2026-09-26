"""Correr con: python -m tests.test_evaluar_test

Prueba las funciones puras de src/evaluar_test.py con datos sintéticos. Ninguna abre el test ni
corre la evaluación: el caso de punta a punta toma 400 filas de train y las parte en 300 y 100,
como si fueran train y datos nuevos. Los casos pequeños están calculados a mano en su comentario.
"""

import copy
import io
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import evaluar_test, metricas
from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES, OPCIONES_FINALES
from src.datos import FILA, OBJETIVO, SEMILLA, cargar_train
from src.evaluar_test import (
    MACROS,
    RUTA_EVALUACION,
    armar_registro,
    decimal_tex,
    diferencias_con_configuracion,
    empates_en_el_corte,
    entero_tex,
    escribir_marcadores,
    evaluar,
    guardia_de_reevaluacion,
    imprimir_resumen,
    intervalo_bootstrap,
    intervalo_tex,
    leer_eleccion,
    llamados_al_corte,
    matriz_al_corte,
    metrica_sola,
    rehacer_tex,
    renderizar_tex,
    texto_del_corte,
    verificar_tamanos,
)
from src.metricas import METRICAS, puntajes
from src.modelos import MODELOS, crear_modelo, separar_X_y
from src.numeros import a_texto, formatear_decimal, formatear_miles
from src.preproceso import Opciones
from src.seleccion import hiperparametros_de

# Una definición por línea; el valor puede tener llaves adentro (la coma decimal es {,}).
NUEVA = re.compile(r"^\\newcommand\{\\([a-zA-Z]+)\}\{(.*)\}$", re.M)
# Los macros que llevan la raya «--» del intervalo y por eso van sólo en modo texto.
INTERVALOS = ("aucic", "recallic")


def _registro_sintetico(n_evaluaciones=1):
    # Valores inventados para probar el formato; la matriz es coherente: 504 + 1 143 = 1 647
    # llamadas, 504 + 424 = 928 «yes» y 8 236 filas en total. En el corte empatan 1 098 filas y se
    # llama a 30, así que la matriz (recall 504/928 = 0,543) y recall_q (0,546) difieren.
    return {
        "modelo": "rf", "hiperparametros": {"n_estimators": 300}, "fecha": "2026-10-01T10:00:00",
        "n_train": 32940, "n_test": 8236, "n_yes_test": 928, "presupuesto": 0.2,
        "llamadas": 1647, "semilla": 42, "n_evaluaciones": n_evaluaciones,
        "metricas": {
            "auc": {"valor": 0.8234, "ic95": [0.8121, 0.8339]},
            "ap": {"valor": 0.4512},
            "recall_q": {"valor": 0.5462, "ic95": [0.5102, 0.5768]},
            "precision_q": {"valor": 0.3077},
            "f1_q": {"valor": 0.3936},
        },
        "matriz_confusion": {"vp": 504, "fp": 1143, "fn": 424, "vn": 6165},
        "desempate": {"empatados": 1098, "llamados_entre_empatados": 30,
                      "recall_matriz": 504 / 928, "precision_matriz": 504 / 1647},
    }


@contextmanager
def _configuracion(hiperparametros):
    """Fija los hiperparámetros vigentes durante el bloque, con todos los modelos como finales. El
    dict es el mismo objeto en src/configuracion.py, src/seleccion.py y src/evaluar_test.py."""
    with patch.dict(HIPERPARAMETROS_FINALES, hiperparametros), \
            patch.object(evaluar_test, "MODELOS_FINALES", list(MODELOS)):
        yield


def _termina_con_codigo_1(funcion, *args):
    """Corre `funcion` y devuelve lo que escribió en stderr; falla si no termina con código 1."""
    errores = io.StringIO()
    try:
        with redirect_stderr(errores), redirect_stdout(io.StringIO()):
            funcion(*args)
    except SystemExit as salida:
        assert salida.code == 1, salida.code
        return errores.getvalue()
    raise AssertionError(f"{funcion.__name__} debería terminar con código 1")


def test_bootstrap_con_orden_perfecto():
    # 10 clientes, los 2 «yes» con los dos puntajes más altos. Cada remuestreo estratificado tiene
    # 2 «yes» arriba de 8 «no»: AUC 1 y, con 2 llamadas (20 % de 10), recall 1 en todos.
    y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert intervalo_bootstrap(y, s, "auc") == (1.0, 1.0)
    assert intervalo_bootstrap(y, s, "recall_q") == (1.0, 1.0)
    print("ok  bootstrap con orden perfecto: IC 95 % del AUC y del recall = [1, 1]")


def test_bootstrap_con_los_parametros_de_la_especificacion():
    # D-24: 2 000 remuestreos con semilla 42 y percentiles 2,5 y 97,5. Se replican aquí con esos
    # valores escritos en el test, no con las constantes del módulo: cada remuestreo sortea con
    # reposición los «yes» entre los «yes» y después los «no» entre los «no».
    datos = np.random.default_rng(SEMILLA)
    y = np.r_[np.ones(8, int), np.zeros(52, int)]
    s = y + datos.normal(0, 1, len(y))
    positivos, negativos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    sorteo = np.random.default_rng(42)
    auc, recall = [], []
    for _ in range(2000):
        i = np.r_[positivos[sorteo.integers(0, len(positivos), len(positivos))],
                  negativos[sorteo.integers(0, len(negativos), len(negativos))]]
        auc.append(roc_auc_score(y[i], s[i]))
        recall.append(metricas.en_presupuesto(y[i], s[i])["recall_q"])
    assert np.allclose(intervalo_bootstrap(y, s, "auc"), np.percentile(auc, [2.5, 97.5]))
    assert np.allclose(intervalo_bootstrap(y, s, "recall_q"), np.percentile(recall, [2.5, 97.5]))
    print("ok  bootstrap: 2 000 remuestreos estratificados, semilla 42 y percentiles 2,5 y 97,5")


def test_bootstrap_reproducible_y_alrededor_del_valor():
    rng = np.random.default_rng(SEMILLA)
    y = np.r_[np.ones(40, int), np.zeros(260, int)]
    s = y + rng.normal(0, 1, len(y))
    ic = intervalo_bootstrap(y, s, "auc", n=500)
    assert ic == intervalo_bootstrap(y, s, "auc", n=500)
    assert ic != intervalo_bootstrap(y, s, "auc", n=500, semilla=SEMILLA + 1)
    assert 0 < ic[0] < metricas.evaluar(y, s)["auc"] < ic[1] < 1
    try:
        intervalo_bootstrap(np.zeros(10, int), np.arange(10), "auc", n=10)
    except ValueError:
        pass
    else:
        raise AssertionError("sin «yes» el bootstrap debería fallar")
    print("ok  el bootstrap es reproducible con la semilla, cambia con otra y rodea al AUC medido")


def test_metrica_sola_coincide_con_src_metricas():
    rng = np.random.default_rng(SEMILLA)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y
    todas = metricas.evaluar(y, s)
    assert all(metrica_sola(nombre)(y, s) == todas[nombre] for nombre in METRICAS)
    print("ok  cada métrica del bootstrap es la misma de src/metricas.py")


def test_matriz_al_corte_calculada_a_mano():
    # 10 filas, 2 «yes» (filas 0 y 1), k = 2. Se llama a las de puntaje 0,9 (fila 0, «yes») y
    # 0,8 (fila 2, «no»): VP 1 y FP 1. Sin llamar quedan la fila 1 («yes», 0,1), FN 1, y 7 «no».
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05]
    assert matriz_al_corte(y, s, 2) == {"vp": 1, "fp": 1, "fn": 1, "vn": 7}
    assert empates_en_el_corte(s, 2) == {"puntaje_corte": 0.8, "empatados": 1,
                                         "llamados_entre_empatados": 1}
    print("ok  matriz al corte con 10 filas y k = 2: VP 1, FP 1, FN 1, VN 7")


def test_empate_en_el_corte_se_desempata_por_fila():
    # Puntajes 0,9 / 0,5 / 0,5 / 0,5 y 0,1 en el resto; «yes» en las filas 2 y 9; k = 2. La fila 0
    # («no») entra sola y la segunda llamada sale de las tres empatadas en 0,5. Por posición va a
    # la fila 1 («no»): VP 0, FP 2, FN 2, VN 6. Si la fila 2 va primero en el desempate: VP 1.
    y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    s = [0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert matriz_al_corte(y, s, 2) == {"vp": 0, "fp": 2, "fn": 2, "vn": 6}
    assert matriz_al_corte(y, s, 2, desempate=[5, 7, 6, 8, 9, 10, 11, 12, 13, 14])["vp"] == 1
    assert empates_en_el_corte(s, 2) == {"puntaje_corte": 0.5, "empatados": 3,
                                         "llamados_entre_empatados": 1}
    # recall_q reparte el empate en proporción: es el promedio de los tres desempates posibles,
    # (0 + 1/2 + 0) / 3 = 1/6.
    recalls = []
    for primera in (1, 2, 3):
        clave = np.arange(10)
        clave[primera] = -1
        m = matriz_al_corte(y, s, 2, desempate=clave)
        recalls.append(m["vp"] / (m["vp"] + m["fn"]))
    assert abs(np.mean(recalls) - metricas.en_presupuesto(y, s, 0.2)["recall_q"]) < 1e-12
    print("ok  el empate en el corte se desempata por fila; recall_q es el promedio de los desempates")


def test_texto_del_corte_distingue_si_el_empate_queda_partido():
    # 15 filas, k = 3. Con s = 0,9 / 0,5 / 0,5 y 0,1 en el resto, las dos filas en 0,5 empatan con
    # el corte y se llama a las dos: hay empate, pero no queda partido. Con k = 2 se llama a una
    # sola de las dos y sí queda partido. Con puntajes distintos no hay empate.
    y = [1, 0, 1] + [0] * 12
    casos = (([0.9, 0.5, 0.5] + [0.1] * 12, 3, "se llama a todas"),
             ([0.9, 0.5, 0.5] + [0.1] * 12, 2, "Empate partido en el corte: 2 filas"),
             (list(np.linspace(1, 0, 15)), 3, "Sin empates en el corte"))
    for s, k, esperado in casos:
        m = matriz_al_corte(y, s, k)
        desempate = {**empates_en_el_corte(s, k), "recall_matriz": m["vp"] / 2,
                     "precision_matriz": m["vp"] / k}
        registro = {**_registro_sintetico(), "llamadas": k, "desempate": desempate}
        texto = texto_del_corte(registro)
        assert esperado in texto, (k, texto)
        assert ("Sin empates" in texto) == (esperado == "Sin empates en el corte"), texto
        with redirect_stdout(io.StringIO()) as salida:
            imprimir_resumen(registro)
        assert texto in salida.getvalue()
    print("ok  el resumen distingue sin empate, empate con todas las filas llamadas y empate partido")


def test_renderizar_tex_sin_resultados():
    tex = renderizar_tex(None)
    macros = NUEVA.findall(tex)
    assert [nombre for nombre, _ in macros] == list(MACROS)
    assert all(valor == "?" for _, valor in macros)
    assert r"\newif\iftestpendiente\testpendientetrue" in tex and "testpendientefalse" not in tex
    print(f"ok  marcadores: las {len(MACROS)} macros con «?» y la bandera testpendiente en verdadero")


def test_formato_tex_es_el_de_src_numeros():
    # El mismo texto que src/numeros.py para los mismos valores: la coma decimal como {,}, los
    # miles con espacio fino. Los valores cubren el cero, el redondeo hacia arriba en el tercer
    # decimal (0,9995 -> 1{,}000) y los miles en la parte entera.
    for x, decimales in ((0.0, 3), (0.5, 3), (0.8234, 3), (0.0005, 3), (0.99951, 3),
                         (0.54625, 3), (1234.5678, 1), (1234.5678, 3)):
        assert decimal_tex(x, decimales) == formatear_decimal(x, decimales), (x, decimales)
    for n in (0, 7, 30, 928, 1000, 1647, 8236, 32940, 1234567):
        assert entero_tex(n) == formatear_miles(n), n
    assert decimal_tex(0.8234) == "0{,}823" and entero_tex(8236) == r"8\,236"
    assert intervalo_tex((0.8121, 0.8339)) == "0{,}812--0{,}834"
    # En el guion (src/numeros.py, a_texto), como en numeros.md: coma, espacio común y la raya.
    assert a_texto(intervalo_tex((0.8121, 0.8339))) == "0,812–0,834"
    assert a_texto(entero_tex(8236)) == "8 236"
    print("ok  el formato de resultados-test.tex es el de src/numeros.py: 0{,}823, 8\\,236 y "
          "0{,}812--0{,}834; en el guion, 0,823, 8 236 y 0,812–0,834")


def test_renderizar_tex_con_resultados():
    registro = _registro_sintetico()
    antes = copy.deepcopy(registro)
    tex = renderizar_tex(registro)
    assert registro == antes, "renderizar_tex no debe modificar lo que recibe"
    valores = dict(NUEVA.findall(tex))
    assert list(valores) == list(MACROS)
    assert valores["auctest"] == "0{,}823" and valores["aucic"] == "0{,}812--0{,}834"
    assert valores["recalltest"] == "0{,}546" and valores["recallic"] == "0{,}510--0{,}577"
    assert valores["precisiontest"] == "0{,}308" and valores["fitest"] == "0{,}394"
    assert valores["aptest"] == "0{,}451"
    assert (valores["vptest"], valores["fptest"], valores["fntest"], valores["vntest"]) == (
        "504", r"1\,143", "424", r"6\,165")
    assert (valores["recallmatriztest"], valores["precisionmatriztest"]) == ("0{,}543", "0{,}306")
    assert (valores["empatadostest"], valores["llamadosempatadostest"]) == (r"1\,098", "30")
    assert valores["ntest"] == r"8\,236" and valores["llamadastest"] == r"1\,647"
    assert valores["nevaluaciones"] == "1"
    assert r"\newif\iftestpendiente\testpendientefalse" in tex and "testpendientetrue" not in tex
    todos = "".join(valores.values())
    assert "?" not in todos and "0.823" not in tex
    # Ni la coma decimal suelta, ni el punto de miles, ni la raya Unicode.
    assert not re.search(r"\d,\d", todos.replace("{,}", "")), valores
    assert not re.search(r"\d\.\d", todos) and "–" not in todos, valores
    assert renderizar_tex(_registro_sintetico(n_evaluaciones=2)) != tex
    print("ok  con resultados: coma decimal entre llaves, tres decimales, espacio fino de miles, "
          "«--» en los intervalos y la bandera en falso")


def test_los_macros_de_test_se_leen_igual_en_texto_y_en_modo_matematico():
    """S-12: con «0,823» sin llaves, $\\auctest$ se imprimía «0, 823». Con el registro sintético,
    se compila cada macro en texto y, salvo los intervalos, dentro de $…$ (si hay pdflatex): sin
    avisos, con el mismo ancho en los dos modos, y el intervalo se lee con su raya."""
    tex = renderizar_tex(_registro_sintetico())
    valores = dict(NUEVA.findall(tex))
    if not shutil.which("pdflatex"):
        print("ok  los macros de test usan {,} y \\, (sin pdflatex: no se compiló)")
        return
    en_matematica = [n for n in MACROS if n not in INTERVALOS]
    anchos = "\n".join(f"\\setbox0\\hbox{{\\{n}}}\\setbox2\\hbox{{$\\{n}$}}"
                       f"\\typeout{{ANCHO {n} \\the\\wd0 \\space y \\the\\wd2}}" for n in en_matematica)
    cuerpo = "\n".join(f"\\noindent\\{n}{{}}" + ("" if n in INTERVALOS else f" y ${{\\{n}}}$")
                       + "\\par" for n in MACROS)
    documento = ("\\documentclass{article}\n\\usepackage[T1]{fontenc}\n"
                 "\\usepackage[utf8]{inputenc}\n\\usepackage[spanish,es-noquoting]{babel}\n"
                 "\\input{resultados-test.tex}\n\\begin{document}\n" + anchos + "\n" + cuerpo + "\n"
                 "\\noindent PRUEBA \\aucic{} y $\\auctest$ FIN\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "resultados-test.tex").write_text(tex, encoding="utf-8")
        (tmp / "prueba.tex").write_text(documento, encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "prueba.tex"],
                           cwd=tmp, capture_output=True, text=True, timeout=300)
        log = (tmp / "prueba.log").read_text(encoding="utf-8", errors="replace")
        if r.returncode != 0 and re.search(r"not found|! Font .* not loadable", log):
            print("ok  los macros de test usan {,} y \\, (a esta instalación de LaTeX le falta un "
                  "paquete o una fuente: no se compiló)")
            return
        assert r.returncode == 0, log[-3000:]
        problemas = [l for l in log.splitlines()
                     if "invalid in math mode" in l or "Missing character" in l or l.startswith("!")]
        assert not problemas, problemas
        medidas = re.findall(r"^ANCHO (\w+) ([\d.]+)pt y ([\d.]+)pt$", log, re.M)
        assert len(medidas) == len(en_matematica), (len(medidas), len(en_matematica))
        distintos = [(n, a, b) for n, a, b in medidas if abs(float(a) - float(b)) > 0.1]
        assert not distintos, f"se ven distinto en $…$ que en texto: {distintos}"
        if shutil.which("pdftotext"):
            texto = subprocess.run(["pdftotext", str(tmp / "prueba.pdf"), "-"], capture_output=True,
                                   text=True).stdout
            prueba = " ".join(texto[texto.index("PRUEBA"):texto.index("FIN")].split())
            esperado = f"{a_texto(valores['aucic'])} y {a_texto(valores['auctest'])}"
            assert esperado == prueba.removeprefix("PRUEBA ").strip(), (esperado, prueba)
    print(f"ok  los {len(MACROS)} macros de test compilan sin avisos; los {len(en_matematica)} que no "
          "son intervalos miden lo mismo en texto y en $…$, y el intervalo se lee «0,812–0,834»")


def test_guardia_sin_evaluacion_previa_continua():
    def no_preguntar():
        raise AssertionError("sin evaluación previa no debería preguntar")

    with tempfile.TemporaryDirectory() as d:
        assert guardia_de_reevaluacion(Path(d) / RUTA_EVALUACION.name, no_preguntar) == (True, [])
    print("ok  guardia: sin evaluación previa continúa sin preguntar")


def test_guardia_con_evaluacion_previa_y_no_se_detiene():
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / RUTA_EVALUACION.name
        ruta.write_text(json.dumps(armar_registro(_registro_sintetico(), [], "2026-10-01T10:00:00")),
                        encoding="utf-8")
        antes = ruta.read_bytes()
        for respuesta in ("no", "", "s", "si, pero no"):
            salida = io.StringIO()
            with redirect_stdout(salida):
                continuar, _ = guardia_de_reevaluacion(ruta, confirmar=lambda: respuesta)
            assert not continuar, respuesta
            assert "ya se evaluó 1 vez" in salida.getvalue()
        assert ruta.read_bytes() == antes
    print("ok  guardia: con una evaluación previa y sin «si» no continúa ni toca el JSON")


def test_guardia_con_si_acumula_las_previas():
    anterior = armar_registro(_registro_sintetico(n_evaluaciones=2),
                              [{"fecha": "2026-10-01T09:00:00", "auc": 0.8}],
                              "2026-10-01T10:00:00")
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / RUTA_EVALUACION.name
        ruta.write_text(json.dumps(anterior), encoding="utf-8")
        with redirect_stdout(io.StringIO()) as salida:
            continuar, previas = guardia_de_reevaluacion(ruta, confirmar=lambda: " Sí ")
    assert continuar and "ya se evaluó 2 veces" in salida.getvalue()
    assert previas == [{"fecha": "2026-10-01T09:00:00", "auc": 0.8},
                       {"fecha": "2026-10-01T10:00:00", "auc": 0.8234}]
    assert armar_registro({}, previas, "2026-10-02T10:00:00")["n_evaluaciones"] == 3
    print("ok  guardia: con «si» continúa y el registro nuevo suma las previas (tercera evaluación)")


def test_leer_eleccion():
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "modelo_elegido.json"
        assert "modelo_elegido.json" in _termina_con_codigo_1(leer_eleccion, ruta)

        ruta.write_text(json.dumps({"modelo": "rf", "hiperparametros": {"n_estimators": 300},
                                    "opciones": asdict(Opciones(month=False))}), encoding="utf-8")
        assert leer_eleccion(ruta) == ("rf", {"n_estimators": 300}, Opciones(month=False))

        for eleccion in ({"modelo": "rf", "hiperparametros": {}},
                         {"modelo": "rf", "hiperparametros": {}, "opciones": {"no_existe": 1}},
                         {"modelo": "xgboost", "hiperparametros": {}, "opciones": {}}):
            ruta.write_text(json.dumps(eleccion), encoding="utf-8")
            assert "ERROR" in _termina_con_codigo_1(leer_eleccion, ruta)
    print("ok  modelo elegido: si falta o está incompleto termina con código 1 y dice por qué")


def test_diferencias_con_configuracion():
    modelo = "rf" if "rf" in MODELOS_FINALES else MODELOS_FINALES[0]
    vigentes = dict(HIPERPARAMETROS_FINALES[modelo])
    assert diferencias_con_configuracion(modelo, vigentes, OPCIONES_FINALES) == []

    otros = {**vigentes, "clave_nueva": 1}
    diferencias = diferencias_con_configuracion(modelo, otros, OPCIONES_FINALES)
    assert len(diferencias) == 1 and "clave_nueva" in diferencias[0]

    otras = replace(OPCIONES_FINALES, month=not OPCIONES_FINALES.month)
    diferencias = diferencias_con_configuracion(modelo, vigentes, otras)
    assert len(diferencias) == 1 and "opciones.month" in diferencias[0]
    print("ok  el modelo elegido se compara con src/configuracion.py y se nombra cada diferencia")


def test_diferencias_con_configuracion_compara_el_modelo_que_se_arma():
    # La configuración escribe sólo lo que cambia, y src/seleccion.py declara además lo que traiga
    # hiperparametros.json, que puede ser un valor por defecto: max_depth=None o pesos_clase=None
    # arman el mismo modelo que omitirlos. max_features=None no: si falta, el RF usa "sqrt". Se
    # fija una configuración propia para que el caso no dependa de la vigente.
    with _configuracion({"rf": {"n_estimators": 300}, "svm": {"C": 1.0}}):
        de_la_ola_4 = {"rf": {"max_depth": None, "n_estimators": 300, "pesos_clase": None},
                       "svm": {"pesos_clase": None, "implementacion": "libsvm"}}
        for modelo in de_la_ola_4:
            declarados = hiperparametros_de(modelo, de_la_ola_4)
            assert diferencias_con_configuracion(modelo, declarados, OPCIONES_FINALES) == [], modelo

        otra = hiperparametros_de("rf", {"rf": {"max_depth": 10}})
        diferencias = diferencias_con_configuracion("rf", otra, OPCIONES_FINALES)
        assert diferencias == ["hiperparametros.max_depth: 10 en el JSON y None (no está: el valor "
                               "por defecto) en la configuración"], diferencias
        diferencias = diferencias_con_configuracion("rf", {"max_features": None}, OPCIONES_FINALES)
        assert len(diferencias) == 1 and "max_features: None en el JSON y 'sqrt'" in diferencias[0]

    # Al revés: la configuración escribe los valores por defecto y el JSON no los trae.
    with _configuracion({"rf": {"n_estimators": 300, "max_depth": None, "pesos_clase": None}}):
        assert diferencias_con_configuracion("rf", {"n_estimators": 300}, OPCIONES_FINALES) == []
        balanceado = {"n_estimators": 300, "pesos_clase": "balanced"}
        diferencias = diferencias_con_configuracion("rf", balanceado, OPCIONES_FINALES)
        assert len(diferencias) == 1 and "pesos_clase: 'balanced'" in diferencias[0]
    print("ok  un valor por defecto declarado de un lado y omitido del otro no es una diferencia")


def test_verificar_tamanos():
    verificar_tamanos(32_940, 8_236)
    for n_train, n_test in ((2_400, 600), (32_941, 8_236), (32_941, 8_235)):
        try:
            verificar_tamanos(n_train, n_test)
        except AssertionError as error:
            assert "filas" in str(error)
        else:
            raise AssertionError(f"{n_train} + {n_test} debería fallar")
    print("ok  tamaños: 32 940 + 8 236 pasa; cualquier otra partición falla con mensaje")


def test_marcadores_no_pisan_una_evaluacion():
    with tempfile.TemporaryDirectory() as d:
        tex, registro = Path(d) / "resultados.tex", Path(d) / RUTA_EVALUACION.name
        with redirect_stdout(io.StringIO()):
            escribir_marcadores(tex, registro)
        assert r"\testpendientetrue" in tex.read_text(encoding="utf-8")

        registro.write_text(json.dumps(_registro_sintetico()), encoding="utf-8")
        antes = tex.read_bytes()
        _termina_con_codigo_1(escribir_marcadores, tex, registro)
        assert tex.read_bytes() == antes

        with redirect_stdout(io.StringIO()):
            rehacer_tex(tex, registro)
        assert r"\testpendientefalse" in tex.read_text(encoding="utf-8")
    print("ok  --marcadores se niega si ya hay evaluación; --rehacer-tex la pasa al .tex")


def test_evaluar_sobre_400_filas_de_train():
    train = cargar_train()
    muestra, _ = train_test_split(train, train_size=400, stratify=train[OBJETIVO],
                                  random_state=SEMILLA)
    a, b = train_test_split(muestra, test_size=100, stratify=muestra[OBJETIVO],
                            random_state=SEMILLA)
    # b queda en el orden del sorteo, no por fila: así se nota si el corte se desempata por
    # posición en lugar de por fila.
    a, b = a.sort_values(FILA).reset_index(drop=True), b.reset_index(drop=True)
    X_a, y_a = separar_X_y(a)
    X_b, y_b = separar_X_y(b)
    # Una configuración distinta de la de por defecto, para que se note si evaluar() la ignora.
    opciones, hiperparametros = Opciones(month=False), {"n_neighbors": 7}
    resultados, tabla = evaluar(X_a, y_a, X_b, y_b, "knn", hiperparametros, opciones,
                                filas_test=b[FILA])
    registro = armar_registro(resultados, [], "2026-10-01T10:00:00")

    esperadas = {"ADVERTENCIA", "modelo", "hiperparametros", "opciones", "n_train", "n_test",
                 "n_yes_test", "llamadas", "metricas", "matriz_confusion", "fecha",
                 "n_evaluaciones", "evaluaciones_previas", "semilla"}
    assert esperadas <= set(registro) and next(iter(registro)) == "ADVERTENCIA"
    assert registro["hiperparametros"] == hiperparametros
    assert registro["opciones"] == asdict(opciones) and registro["semilla"] == 42
    assert registro["bootstrap"]["remuestreos"] == 2000
    assert registro["bootstrap"]["percentiles"] == [2.5, 97.5]
    assert set(registro["metricas"]) == set(METRICAS)
    for nombre in METRICAS:
        assert ("ic95" in registro["metricas"][nombre]) == (nombre in ("auc", "recall_q"))
    for nombre in ("auc", "recall_q"):
        inf, sup = registro["metricas"][nombre]["ic95"]
        assert 0 <= inf <= sup <= 1

    # Los puntajes son los de un ajuste independiente con esa configuración y sólo con las 300
    # filas de train: no los de otra configuración ni los de un ajuste con los datos evaluados.
    ajustado = crear_modelo("knn", opciones, **hiperparametros).fit(X_a, y_a)
    esperado = (pd.DataFrame({"fila": b[FILA], "y": y_b, "puntaje": puntajes(ajustado, X_b)})
                .sort_values("fila").reset_index(drop=True))
    assert list(tabla.columns) == ["fila", "y", "puntaje", "llamado"]
    assert list(tabla["fila"]) == list(esperado["fila"]) == sorted(b[FILA])
    assert np.allclose(tabla["puntaje"], esperado["puntaje"])
    assert (tabla["y"] == esperado["y"]).all()

    c = registro["matriz_confusion"]
    assert (registro["n_train"], registro["n_test"], registro["n_evaluaciones"]) == (300, 100, 1)
    assert c["vp"] + c["fp"] == registro["llamadas"] == 20
    assert c["vp"] + c["fn"] == registro["n_yes_test"] == int(y_b.sum())
    assert sum(c.values()) == 100

    # KNN con k = 7 da puntajes en séptimos, y el corte parte un empate: la tabla y la matriz lo
    # desempatan por fila, que en la tabla ordenada es la posición.
    d = registro["desempate"]
    assert d["empatados"] > d["llamados_entre_empatados"]
    assert (tabla["llamado"].to_numpy() == llamados_al_corte(tabla["puntaje"], 20)).all()
    assert c["vp"] == int(((tabla["llamado"] == 1) & (tabla["y"] == 1)).sum())
    assert d["recall_matriz"] == c["vp"] / int(y_b.sum()) and d["precision_matriz"] == c["vp"] / 20
    recalculadas = metricas.evaluar(tabla["y"], tabla["puntaje"])
    assert all(abs(recalculadas[n] - registro["metricas"][n]["valor"]) < 1e-12 for n in METRICAS)
    json.dumps(registro, allow_nan=False)
    tex = renderizar_tex(registro)
    assert "?" not in tex and f"% {registro['ADVERTENCIA']}" in tex
    print("ok  evaluar() con 300 filas de train y 100 nuevas: ajusta sólo con train y con la "
          "configuración dada, y el corte se desempata por fila")


def main():
    test_bootstrap_con_orden_perfecto()
    test_bootstrap_con_los_parametros_de_la_especificacion()
    test_bootstrap_reproducible_y_alrededor_del_valor()
    test_metrica_sola_coincide_con_src_metricas()
    test_matriz_al_corte_calculada_a_mano()
    test_empate_en_el_corte_se_desempata_por_fila()
    test_texto_del_corte_distingue_si_el_empate_queda_partido()
    test_renderizar_tex_sin_resultados()
    test_formato_tex_es_el_de_src_numeros()
    test_renderizar_tex_con_resultados()
    test_los_macros_de_test_se_leen_igual_en_texto_y_en_modo_matematico()
    test_guardia_sin_evaluacion_previa_continua()
    test_guardia_con_evaluacion_previa_y_no_se_detiene()
    test_guardia_con_si_acumula_las_previas()
    test_leer_eleccion()
    test_diferencias_con_configuracion()
    test_diferencias_con_configuracion_compara_el_modelo_que_se_arma()
    test_verificar_tamanos()
    test_marcadores_no_pisan_una_evaluacion()
    test_evaluar_sobre_400_filas_de_train()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
