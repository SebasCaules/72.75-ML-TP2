"""Correr con: python -m src.evaluar_test

Punto 4 del enunciado; paso 5.3 del plan (D-24). La evaluación única del test: es el único módulo de
cómputo que abre el test, y tests/test_aislamiento.py lo hace cumplir.

Sin opciones:
1. Lee resultados/modelo_elegido.json, la salida del paso 5.1 (D-23), con las claves `modelo`,
   `hiperparametros` (dict) y `opciones` (los campos de `Opciones`), y comprueba que coincida con
   src/configuracion.py. Si falta o no coincide, termina con código 1 sin abrir el test.
2. Si el test ya se evaluó (existe resultados/evaluacion_test.json), dice cuántas veces y pide
   escribir «si»; con cualquier otra respuesta termina con código 1 sin tocar nada. Si se confirma,
   la corrida nueva queda registrada: `n_evaluaciones` sube y `evaluaciones_previas` guarda la fecha
   y el AUC de cada corrida anterior. Además hay que anotarla en DECISIONES.md.
3. Reentrena el modelo con todo train (Clase 2, slides 88-89), puntúa el test una sola vez y mide
   las cinco métricas de src/metricas.py con el presupuesto de D-20, la matriz de confusión de ese
   corte y el intervalo del 95 % por bootstrap del AUC y del recall.
4. Escribe resultados/evaluacion_test.json, resultados/puntajes_test.csv (fila, y, puntaje,
   llamado) e informe/resultados-test.tex, las macros de LaTeX que lee el deck.

Los otros modos no abren el test:
  --marcadores   escribe informe/resultados-test.tex con «?» en cada número y la bandera
                 testpendiente en verdadero, para que el deck compile antes de la evaluación
                 (N0-1). Se niega si el test ya se evaluó, para no pisar los números.
  --rehacer-tex  rehace informe/resultados-test.tex desde resultados/evaluacion_test.json, por
                 ejemplo si el deck necesita una macro nueva después de la evaluación.
  --rapido       ensayo de punta a punta: 3 000 filas de train partidas 80/20 hacen de train y de
                 test. Escribe en un directorio temporal, nunca en resultados/ ni en informe/.
  --forzar       sólo para pruebas manuales: responde «si» a la confirmación. La corrida igual
                 queda registrada como una evaluación más.
"""

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import metricas
from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES, OPCIONES_FINALES
from src.datos import FILA, OBJETIVO, PROP_TEST, RAIZ, SEMILLA, cargar_test, cargar_train
from src.metricas import METRICAS, PRESUPUESTO, en_presupuesto, llamadas, puntajes
from src.modelos import MODELOS, REFERENCIA, crear_modelo, separar_X_y
from src.preproceso import Opciones
from src.resultados import DIR, configuracion_json

RUTA_ELECCION = DIR / "modelo_elegido.json"
RUTA_EVALUACION = DIR / "evaluacion_test.json"
RUTA_PUNTAJES = DIR / "puntajes_test.csv"
RUTA_TEX = RAIZ / "informe" / "resultados-test.tex"

# D-01 y D-02: 41 188 filas menos 12 duplicados exactos, y el 20 % estratificado para test.
N_TOTAL = 41_176
N_TEST = 8_236

N_BOOTSTRAP = 2000
PERCENTILES = (2.5, 97.5)
CON_INTERVALO = ("auc", "recall_q")

N_RAPIDO = 3000
ARBOLES_RAPIDO = 20

ADVERTENCIA = ("Evaluación única del test (D-24): este número se mide una vez; no participa de "
               "ninguna decisión. Si se cambia el modelo mirando este resultado y se vuelve a "
               "evaluar, el número nuevo deja de estimar el desempeño en datos nuevos.")
ADVERTENCIA_ENSAYO = ("Ensayo con --rapido: 3 000 filas de train partidas 80/20 hacen de train y "
                      "de test. No es el test y no se usa para nada.")
DESEMPATE = ("Con empate en el corte, la matriz llama primero a la fila de menor `fila` (el orden "
             "del CSV original, que es temporal) para que sus celdas sean enteras. recall_q y "
             "precision_q reparten el empate en proporción (src/metricas.py), así que difieren de "
             "recall_matriz y precision_matriz sólo si empatados > llamados_entre_empatados.")

MACROS = ("auctest", "aucic", "recalltest", "recallic", "precisiontest", "fitest", "aptest",
          "vptest", "fptest", "fntest", "vntest", "recallmatriztest", "precisionmatriztest",
          "empatadostest", "llamadosempatadostest", "ntest", "llamadastest", "nevaluaciones")

# Los hiperparámetros que `crear_modelo` saca antes de armar el clasificador, con el valor que usa
# cuando faltan (src/modelos.py). Cualquier otro que falte toma el valor por defecto del
# clasificador.
PROPIOS_DE_CREAR_MODELO = {"pesos_clase": None, "implementacion": "libsvm"}
_AUSENTE = "(no está)"


# --- Bootstrap y matriz de confusión ---------------------------------------------------------

def metrica_sola(nombre, q=PRESUPUESTO):
    """Una de las cinco métricas, con la misma definición que `src.metricas.evaluar`."""
    if nombre == "auc":
        return roc_auc_score
    if nombre == "ap":
        return average_precision_score
    if nombre in ("recall_q", "precision_q", "f1_q"):
        return lambda y, s: en_presupuesto(y, s, q)[nombre]
    raise ValueError(f"métrica desconocida: {nombre}; las válidas son {METRICAS}")


def intervalo_bootstrap(y, s, metrica, n=N_BOOTSTRAP, semilla=SEMILLA, q=PRESUPUESTO):
    """Intervalo del 95 % de `metrica` por bootstrap de percentiles (2,5 y 97,5) sobre las
    predicciones ya hechas: no se vuelve a ajustar ni a predecir nada.

    Cada remuestreo sortea con reposición los «yes» entre los «yes» y los «no» entre los «no»: el
    test se separó estratificado por `y` (D-02), y así cada remuestreo conserva su composición y
    su número de llamadas. Con la misma semilla, todas las métricas usan los mismos remuestreos.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    positivos, negativos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positivos) or not len(negativos):
        raise ValueError("el bootstrap necesita al menos un «yes» y un «no»")
    medir = metrica_sola(metrica, q)
    rng = np.random.default_rng(semilla)
    valores = np.empty(n)
    for b in range(n):
        idx = np.concatenate([positivos[rng.integers(0, len(positivos), len(positivos))],
                              negativos[rng.integers(0, len(negativos), len(negativos))]])
        valores[b] = medir(y[idx], s[idx])
    inf, sup = np.percentile(valores, PERCENTILES)
    return float(inf), float(sup)


def llamados_al_corte(s, k, desempate=None):
    """1 para los k de mayor puntaje y 0 para el resto. A igual puntaje se llama primero al de
    menor `desempate` (por defecto, la posición en `s`)."""
    s = np.asarray(s, dtype=float)
    if not 1 <= k <= len(s):
        raise ValueError(f"k = {k} tiene que estar entre 1 y {len(s)}")
    desempate = np.arange(len(s)) if desempate is None else np.asarray(desempate)
    orden = np.lexsort((desempate, -s))
    llamado = np.zeros(len(s), dtype=int)
    llamado[orden[:k]] = 1
    return llamado


def matriz_al_corte(y, s, k, desempate=None):
    """La matriz de confusión si se llama a los k de mayor puntaje: vp, fp, fn y vn."""
    y = np.asarray(y).astype(int)
    llamado = llamados_al_corte(s, k, desempate)
    return {"vp": int(((llamado == 1) & (y == 1)).sum()),
            "fp": int(((llamado == 1) & (y == 0)).sum()),
            "fn": int(((llamado == 0) & (y == 1)).sum()),
            "vn": int(((llamado == 0) & (y == 0)).sum())}


def empates_en_el_corte(s, k):
    """Cuántos puntajes empatan con el del k-ésimo llamado y a cuántos de ellos se llama."""
    s = np.asarray(s, dtype=float)
    corte = np.sort(s)[::-1][k - 1]
    return {"puntaje_corte": float(corte), "empatados": int((s == corte).sum()),
            "llamados_entre_empatados": int(k - (s > corte).sum())}


# --- La evaluación ------------------------------------------------------------------------------

def evaluar(X_train, y_train, X_test, y_test, modelo, hiperparametros=None, opciones=Opciones(),
            filas_test=None, q=PRESUPUESTO, n_bootstrap=N_BOOTSTRAP, semilla=SEMILLA):
    """Reentrena `modelo` con todo `X_train`, puntúa `X_test` una sola vez y lo mide.

    Recibe los datos ya leídos, así se prueba con datos sintéticos. `filas_test` es la columna
    `fila` del test (por defecto, el índice de `X_test`): identifica cada puntaje y desempata el
    corte. Devuelve el registro de la evaluación, sin la fecha ni el historial que agrega
    `armar_registro`, y la tabla de puntajes con fila, y, puntaje y llamado (0/1).
    """
    hiperparametros = dict(hiperparametros or {})
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    filas = np.asarray(X_test.index if filas_test is None else filas_test)

    ajustado = crear_modelo(modelo, opciones, **hiperparametros).fit(X_train, y_train)
    s = puntajes(ajustado, X_test)

    k = llamadas(len(y_test), q)
    valores = metricas.evaluar(y_test, s, q)
    medidas = {nombre: {"valor": valores[nombre]} for nombre in METRICAS}
    for nombre in CON_INTERVALO:
        medidas[nombre]["ic95"] = list(intervalo_bootstrap(y_test, s, nombre, n_bootstrap,
                                                           semilla, q))
    llamado = llamados_al_corte(s, k, desempate=filas)
    matriz = matriz_al_corte(y_test, s, k, desempate=filas)
    positivos = matriz["vp"] + matriz["fn"]

    registro = {
        "modelo": modelo,
        "hiperparametros": hiperparametros,
        "opciones": asdict(opciones),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_yes_test": int(y_test.sum()),
        "presupuesto": q,
        "llamadas": k,
        "metricas": medidas,
        "matriz_confusion": matriz,
        "desempate": {
            "criterio": DESEMPATE,
            **empates_en_el_corte(s, k),
            "recall_matriz": matriz["vp"] / positivos if positivos else float("nan"),
            "precision_matriz": matriz["vp"] / k,
        },
        "bootstrap": {"remuestreos": n_bootstrap, "estratificado": True,
                      "percentiles": list(PERCENTILES), "metricas": list(CON_INTERVALO)},
        "semilla": semilla,
    }
    tabla = (pd.DataFrame({"fila": filas, "y": y_test, "puntaje": s, "llamado": llamado})
             .sort_values("fila", kind="stable").reset_index(drop=True))
    return registro, tabla


def armar_registro(resultados, previas, fecha, advertencia=ADVERTENCIA):
    """El JSON de la evaluación: la advertencia primero y, al final, el historial de corridas."""
    return {"ADVERTENCIA": advertencia, **resultados, "fecha": fecha,
            "n_evaluaciones": len(previas) + 1, "evaluaciones_previas": list(previas)}


def verificar_tamanos(n_train, n_test):
    """D-01 y D-02: el test tiene 8 236 filas y, con train, suma las 41 176 sin duplicados."""
    assert n_test == N_TEST, (
        f"el test tiene {n_test} filas y debería tener {N_TEST} (D-02): la partición cambió")
    assert n_train + n_test == N_TOTAL, (
        f"train y test suman {n_train + n_test} filas y deberían sumar {N_TOTAL} (D-01)")


# --- Guardas ------------------------------------------------------------------------------------

def _salir(mensaje):
    print(f"ERROR: {mensaje}", file=sys.stderr)
    raise SystemExit(1)


def _mostrar(ruta):
    ruta = Path(ruta)
    try:
        return ruta.relative_to(RAIZ).as_posix()
    except ValueError:
        return str(ruta)


def _preguntar():
    try:
        return input("Escriba si para evaluar el test de nuevo (cualquier otra respuesta cancela): ")
    except EOFError:
        return ""


def guardia_de_reevaluacion(ruta_json, confirmar=_preguntar):
    """Decide si se puede evaluar. Devuelve (continuar, previas).

    Sin evaluación anterior: (True, []). Con una, muestra cuántas hubo y continúa sólo si
    `confirmar()` devuelve «si»; `previas` trae la fecha y el AUC de cada corrida anterior, para
    que el JSON nuevo las registre. No escribe nada.
    """
    ruta_json = Path(ruta_json)
    if not ruta_json.exists():
        return True, []
    try:
        anterior = json.loads(ruta_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{_mostrar(ruta_json)} existe pero no es un JSON válido ({error}); "
                         "revíselo antes de volver a evaluar") from error
    previas = list(anterior.get("evaluaciones_previas", [])) + [{
        "fecha": anterior.get("fecha"),
        "auc": anterior.get("metricas", {}).get("auc", {}).get("valor"),
    }]

    veces = len(previas)
    print("=" * 78)
    print(f"AVISO: el test ya se evaluó {veces} {'vez' if veces == 1 else 'veces'}.")
    print("=" * 78)
    for numero, previa in enumerate(previas, start=1):
        auc = "?" if previa.get("auc") is None else formato_decimal(previa["auc"])
        print(f"  {numero}. {previa.get('fecha') or '?'}   AUC {auc}")
    print(f"Volver a evaluar da el mismo número si nada cambió, porque todo usa la semilla "
          f"{SEMILLA}.\nPero si entretanto se cambió algo mirando el resultado anterior, el número "
          "nuevo ya no\nestima el desempeño en datos nuevos. Si continúa, esta corrida queda "
          "registrada en el\nJSON (n_evaluaciones) y hay que anotarla en DECISIONES.md.")
    respuesta = str(confirmar()).strip().lower()
    return respuesta in ("si", "sí"), previas


def leer_eleccion(ruta=RUTA_ELECCION):
    """El modelo que eligió el paso 5.1: (modelo, hiperparametros, opciones). Termina con código 1
    si el archivo falta o no tiene las claves `modelo`, `hiperparametros` y `opciones`."""
    ruta = Path(ruta)
    if not ruta.exists():
        _salir(f"falta {_mostrar(ruta)}, la salida del paso 5.1 (D-23). Corra primero la "
               "selección del modelo final: python3 -m src.seleccion")
    try:
        eleccion = json.loads(ruta.read_text(encoding="utf-8"))
        modelo = eleccion["modelo"]
        hiperparametros = dict(eleccion["hiperparametros"])
        opciones = Opciones(**eleccion["opciones"])
    except (KeyError, TypeError, ValueError) as error:
        _salir(f"{_mostrar(ruta)} no tiene la forma esperada ({type(error).__name__}: {error}). "
               "Debe tener «modelo», «hiperparametros» (un dict) y «opciones» (los campos de "
               "Opciones).")
    if modelo not in MODELOS:
        _salir(f"modelo desconocido en {_mostrar(ruta)}: {modelo!r}; los válidos son {MODELOS}")
    return modelo, hiperparametros, opciones


def _normalizar(valor):
    return json.loads(json.dumps(valor, default=str))


def _parametros_del_clasificador(modelo, hiperparametros, opciones):
    """Los parámetros del clasificador que arma `crear_modelo`; vacío si no se puede armar, y
    entonces una clave que falta no tiene valor por defecto y cuenta como diferencia."""
    try:
        return crear_modelo(modelo, opciones, **hiperparametros)[-1].get_params()
    except (TypeError, ValueError, KeyError):
        return {}


def _efectivo(hiperparametros, del_clasificador, clave):
    """El valor con que se arma `clave`: el dado o, si falta, el que usa `crear_modelo` o el
    clasificador por defecto."""
    if clave in hiperparametros:
        return hiperparametros[clave]
    if clave in PROPIOS_DE_CREAR_MODELO:
        return PROPIOS_DE_CREAR_MODELO[clave]
    return del_clasificador.get(clave, _AUSENTE)


def _describir(hiperparametros, clave, valor):
    if valor is _AUSENTE:
        return _AUSENTE
    return repr(valor) if clave in hiperparametros else f"{valor!r} (no está: el valor por defecto)"


def diferencias_con_configuracion(modelo, hiperparametros, opciones):
    """Lo que el modelo elegido tiene distinto de src/configuracion.py; vacío si coincide.

    Un modelo_elegido.json anterior al último cambio de la configuración evaluaría el test con un
    modelo que ya no es el vigente, y la evaluación es una sola. Los hiperparámetros se comparan
    por el valor con que `crear_modelo` arma el modelo: completados con REFERENCIA y, si una clave
    falta de un lado, con su valor por defecto. src/seleccion.py declara lo que traiga
    hiperparametros.json, y un ganador como max_depth=None o pesos_clase=None que la
    configuración no escribe da el mismo modelo; max_features=None, en cambio, no es el valor por
    defecto del RF y sí cuenta como diferencia.
    """
    diferencias = []
    if modelo not in MODELOS_FINALES:
        diferencias.append(f"modelo: {modelo!r} no está en MODELOS_FINALES {MODELOS_FINALES}")
    for campo in fields(Opciones):
        elegido, vigente = getattr(opciones, campo.name), getattr(OPCIONES_FINALES, campo.name)
        if elegido != vigente:
            diferencias.append(f"opciones.{campo.name}: {elegido!r} en el JSON y {vigente!r} en "
                               "la configuración")
    if modelo in MODELOS:
        elegidos = {**REFERENCIA[modelo], **hiperparametros}
        vigentes = {**REFERENCIA[modelo], **HIPERPARAMETROS_FINALES.get(modelo, {})}
        del_elegido = _parametros_del_clasificador(modelo, elegidos, opciones)
        del_vigente = _parametros_del_clasificador(modelo, vigentes, OPCIONES_FINALES)
        for clave in sorted(set(elegidos) | set(vigentes)):
            elegido = _efectivo(elegidos, del_elegido, clave)
            vigente = _efectivo(vigentes, del_vigente, clave)
            if _normalizar(elegido) != _normalizar(vigente):
                diferencias.append(
                    f"hiperparametros.{clave}: {_describir(elegidos, clave, elegido)} en el JSON "
                    f"y {_describir(vigentes, clave, vigente)} en la configuración")
    return diferencias


# --- Macros de LaTeX ----------------------------------------------------------------------------

_A_ESPANOL = str.maketrans({",": ".", ".": ","})


def formato_decimal(x, decimales=3):
    """Coma decimal y punto de miles: 0.8234 -> «0,823»; 1234.5 -> «1.234,500»."""
    return f"{x:,.{decimales}f}".translate(_A_ESPANOL)


def formato_entero(n):
    """Punto de miles: 8236 -> «8.236»."""
    return f"{int(n):,d}".translate(_A_ESPANOL)


def formato_intervalo(ic, decimales=3):
    inf, sup = ic
    return f"{formato_decimal(inf, decimales)}–{formato_decimal(sup, decimales)}"


def valores_tex(registro):
    """El texto de cada macro a partir del registro de la evaluación, en el orden de MACROS."""
    m, c, d = registro["metricas"], registro["matriz_confusion"], registro["desempate"]
    valores = {
        "auctest": formato_decimal(m["auc"]["valor"]),
        "aucic": formato_intervalo(m["auc"]["ic95"]),
        "recalltest": formato_decimal(m["recall_q"]["valor"]),
        "recallic": formato_intervalo(m["recall_q"]["ic95"]),
        "precisiontest": formato_decimal(m["precision_q"]["valor"]),
        "fitest": formato_decimal(m["f1_q"]["valor"]),
        "aptest": formato_decimal(m["ap"]["valor"]),
        "vptest": formato_entero(c["vp"]),
        "fptest": formato_entero(c["fp"]),
        "fntest": formato_entero(c["fn"]),
        "vntest": formato_entero(c["vn"]),
        "recallmatriztest": formato_decimal(d["recall_matriz"]),
        "precisionmatriztest": formato_decimal(d["precision_matriz"]),
        "empatadostest": formato_entero(d["empatados"]),
        "llamadosempatadostest": formato_entero(d["llamados_entre_empatados"]),
        "ntest": formato_entero(registro["n_test"]),
        "llamadastest": formato_entero(registro["llamadas"]),
        "nevaluaciones": formato_entero(registro["n_evaluaciones"]),
    }
    return {nombre: valores[nombre] for nombre in MACROS}


# Las dos lecturas del corte, para que el deck elija cuál muestra y pueda explicar la diferencia.
LEYENDA_TEX = (
    r"% \recalltest y \precisiontest reparten en proporción el empate en el corte, como",
    r"% src/metricas.py. \recallmatriztest y \precisionmatriztest salen de la matriz (\vptest,",
    r"% \fptest, \fntest), que lo desempata por fila. Difieren sólo si \llamadosempatadostest",
    r"% es menor que \empatadostest.",
)


def renderizar_tex(registro=None):
    """El contenido de informe/resultados-test.tex. Sin registro, «?» en cada macro y la bandera
    testpendiente en verdadero; con el registro de la evaluación, sus números y la bandera en
    falso. No lee ni escribe nada."""
    pendiente = registro is None
    valores = dict.fromkeys(MACROS, "?") if pendiente else valores_tex(registro)
    if pendiente:
        encabezado = ["% Marcadores: el test todavía no se evaluó (N0-1). La evaluación única",
                      "% los reemplaza: python3 -m src.evaluar_test"]
        bandera = r"\testpendientetrue"
    else:
        encabezado = [f"% Evaluación del test del {registro['fecha']} (evaluación número "
                      f"{registro['n_evaluaciones']}), modelo {registro['modelo']}."]
        if "ADVERTENCIA" in registro:
            encabezado.append(f"% {registro['ADVERTENCIA']}")
        bandera = r"\testpendientefalse"
    lineas = ["% GENERADO POR src/evaluar_test.py; no editar a mano.", *encabezado, *LEYENDA_TEX,
              r"\newif\iftestpendiente" + bandera]
    lineas += [f"\\newcommand{{\\{nombre}}}{{{valores[nombre]}}}" for nombre in MACROS]
    return "\n".join(lineas) + "\n"


# --- Escritura ----------------------------------------------------------------------------------

def _a_json(valor):
    if isinstance(valor, np.generic):
        return valor.item()
    raise TypeError(f"no se puede escribir en JSON: {valor!r}")


def escribir_json(ruta, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=_a_json) + "\n",
                    encoding="utf-8")


def escribir_texto(ruta, texto):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")


def escribir_salidas(registro, tabla, ruta_json, ruta_puntajes, ruta_tex):
    """El JSON va primero: si algo falla después, la evaluación ya quedó registrada."""
    escribir_json(ruta_json, registro)
    Path(ruta_puntajes).parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta_puntajes, index=False)
    escribir_texto(ruta_tex, renderizar_tex(registro))


def escribir_marcadores(ruta_tex=RUTA_TEX, ruta_evaluacion=RUTA_EVALUACION):
    if Path(ruta_evaluacion).exists():
        _salir(f"el test ya se evaluó ({_mostrar(ruta_evaluacion)}) y los marcadores pisarían "
               "sus números. Para rehacer el .tex desde la evaluación: --rehacer-tex")
    escribir_texto(ruta_tex, renderizar_tex(None))
    print(f"Marcadores en {_mostrar(ruta_tex)}: {len(MACROS)} macros con «?» y la bandera "
          "testpendiente en verdadero. No se abrió el test.")


def rehacer_tex(ruta_tex=RUTA_TEX, ruta_evaluacion=RUTA_EVALUACION):
    if not Path(ruta_evaluacion).exists():
        _salir(f"todavía no hay evaluación en {_mostrar(ruta_evaluacion)}; mientras tanto, el "
               "deck compila con --marcadores")
    registro = json.loads(Path(ruta_evaluacion).read_text(encoding="utf-8"))
    escribir_texto(ruta_tex, renderizar_tex(registro))
    print(f"{_mostrar(ruta_tex)} rehecho desde {_mostrar(ruta_evaluacion)} (evaluación número "
          f"{registro['n_evaluaciones']}). No se abrió el test.")


# --- Corridas -----------------------------------------------------------------------------------

def texto_del_corte(registro):
    """Qué pasa en el corte: si un empate queda partido, la matriz y recall_q difieren."""
    m, d = registro["metricas"], registro["desempate"]
    empatados, llamados = d["empatados"], d["llamados_entre_empatados"]
    if empatados > llamados:
        return (f"Empate partido en el corte: {formato_entero(empatados)} filas tienen el puntaje "
                f"del corte y se llama a {formato_entero(llamados)}, por orden de fila. La matriz "
                f"da recall {formato_decimal(d['recall_matriz'])} y precisión "
                f"{formato_decimal(d['precision_matriz'])}; recall_q y precision_q, que reparten "
                f"el empate en proporción, {formato_decimal(m['recall_q']['valor'])} y "
                f"{formato_decimal(m['precision_q']['valor'])}.")
    if empatados > 1:
        return (f"{formato_entero(empatados)} filas tienen el puntaje del corte y se llama a "
                "todas: ningún empate queda partido, y la matriz y recall_q cuentan los mismos "
                "aciertos.")
    return "Sin empates en el corte: la matriz y recall_q cuentan los mismos aciertos."


def imprimir_resumen(registro):
    m, c = registro["metricas"], registro["matriz_confusion"]
    print(f"Modelo: {registro['modelo']}  {configuracion_json(registro['hiperparametros'])}")
    print(f"Train: {formato_entero(registro['n_train'])} filas.  Test: "
          f"{formato_entero(registro['n_test'])} filas, {formato_entero(registro['n_yes_test'])} "
          f"«yes».  Semilla {registro['semilla']}.")
    print(f"Presupuesto: el {registro['presupuesto'] * 100:.0f} % de la lista, "
          f"{formato_entero(registro['llamadas'])} llamadas.")
    print()
    for nombre in METRICAS:
        linea = f"  {nombre:12s} {formato_decimal(m[nombre]['valor'])}"
        if "ic95" in m[nombre]:
            linea += f"   IC 95 % {formato_intervalo(m[nombre]['ic95'])}"
        print(linea)
    print()
    print("Matriz de confusión al corte (filas: decisión; columnas: respuesta real):")
    print(f"  {'':12s} {'yes':>8s} {'no':>8s}")
    print(f"  {'llamado':12s} {formato_entero(c['vp']):>8s} {formato_entero(c['fp']):>8s}")
    print(f"  {'no llamado':12s} {formato_entero(c['fn']):>8s} {formato_entero(c['vn']):>8s}")
    print(texto_del_corte(registro))


def ensayo_rapido():
    """Todo el camino de la evaluación sin abrir el test: una submuestra estratificada de train,
    partida 80/20, hace de train y de test. Escribe en un directorio temporal."""
    inicio = time.perf_counter()
    train = cargar_train()
    muestra, _ = train_test_split(train, train_size=N_RAPIDO, stratify=train[OBJETIVO],
                                  random_state=SEMILLA)
    parte_train, parte_test = (
        p.sort_values(FILA).reset_index(drop=True)
        for p in train_test_split(muestra, test_size=PROP_TEST, stratify=muestra[OBJETIVO],
                                  random_state=SEMILLA))

    if RUTA_ELECCION.exists():
        modelo, hiperparametros, opciones = leer_eleccion(RUTA_ELECCION)
        origen = _mostrar(RUTA_ELECCION)
        for diferencia in diferencias_con_configuracion(modelo, hiperparametros, opciones):
            print(f"AVISO: no coincide con src/configuracion.py: {diferencia}")
    else:
        modelo = "rf"
        hiperparametros = dict(HIPERPARAMETROS_FINALES.get(modelo, {}))
        opciones = OPCIONES_FINALES
        origen = f"src/configuracion.py, porque todavía no existe {_mostrar(RUTA_ELECCION)}"
    if modelo == "rf":
        hiperparametros["n_estimators"] = min(
            hiperparametros.get("n_estimators", REFERENCIA["rf"]["n_estimators"]), ARBOLES_RAPIDO)

    print(f"ENSAYO RÁPIDO: no abre el test. {formato_entero(len(muestra))} filas de train "
          f"partidas en {formato_entero(len(parte_train))} y {formato_entero(len(parte_test))}.")
    print(f"Modelo tomado de {origen}.")
    destino = Path(tempfile.mkdtemp(prefix="tp2-evaluar-test-rapido-"))
    ruta_json, ruta_puntajes, ruta_tex = (destino / r.name
                                          for r in (RUTA_EVALUACION, RUTA_PUNTAJES, RUTA_TEX))
    _, previas = guardia_de_reevaluacion(ruta_json)

    X_train, y_train = separar_X_y(parte_train)
    X_test, y_test = separar_X_y(parte_test)
    resultados, tabla = evaluar(X_train, y_train, X_test, y_test, modelo, hiperparametros,
                                opciones, filas_test=parte_test[FILA])
    registro = armar_registro(resultados, previas, datetime.now().isoformat(timespec="seconds"),
                              ADVERTENCIA_ENSAYO)
    escribir_salidas(registro, tabla, ruta_json, ruta_puntajes, ruta_tex)
    print()
    imprimir_resumen(registro)
    print()
    print(f"Escrito en {destino}: {ruta_json.name}, {ruta_puntajes.name} y {ruta_tex.name}.")
    print(f"Ensayo terminado en {time.perf_counter() - inicio:.1f} s.")


def evaluacion_unica(forzar=False):
    modelo, hiperparametros, opciones = leer_eleccion(RUTA_ELECCION)
    diferencias = diferencias_con_configuracion(modelo, hiperparametros, opciones)
    if diferencias:
        _salir(f"{_mostrar(RUTA_ELECCION)} no coincide con src/configuracion.py:\n  - "
               + "\n  - ".join(diferencias)
               + "\nVuelva a correr la selección del paso 5.1 antes de abrir el test.")
    if forzar:
        print("--forzar: se omite la confirmación; la corrida igual queda registrada.")
    try:
        continuar, previas = guardia_de_reevaluacion(
            RUTA_EVALUACION, confirmar=(lambda: "si") if forzar else _preguntar)
    except ValueError as error:
        _salir(str(error))
    if not continuar:
        print("Cancelado: no se abrió el test.")
        raise SystemExit(1)

    inicio = time.perf_counter()
    train, test = cargar_train(), cargar_test()
    verificar_tamanos(len(train), len(test))
    X_train, y_train = separar_X_y(train)
    X_test, y_test = separar_X_y(test)
    print("=" * 78)
    print("EVALUACIÓN ÚNICA DEL TEST")
    print("=" * 78)
    resultados, tabla = evaluar(X_train, y_train, X_test, y_test, modelo, hiperparametros,
                                opciones, filas_test=test[FILA])
    registro = armar_registro(resultados, previas, datetime.now().isoformat(timespec="seconds"))
    escribir_salidas(registro, tabla, RUTA_EVALUACION, RUTA_PUNTAJES, RUTA_TEX)
    imprimir_resumen(registro)
    print()
    print(f"Escrito: {_mostrar(RUTA_EVALUACION)}, {_mostrar(RUTA_PUNTAJES)} y "
          f"{_mostrar(RUTA_TEX)} ({time.perf_counter() - inicio:.0f} s).")
    print("Paso 5.3: haga el commit ahora, con esos tres archivos, y recompile el deck.")
    if registro["n_evaluaciones"] > 1:
        print(f"Es la evaluación número {registro['n_evaluaciones']} del test: regístrela en "
              "DECISIONES.md.")


def _argumentos(argv=None):
    parser = argparse.ArgumentParser(
        prog="python3 -m src.evaluar_test",
        description="Evaluación única del test (paso 5.3, D-24). Sin opciones, reentrena el modelo "
                    "de resultados/modelo_elegido.json con todo train y lo evalúa sobre el test "
                    "una sola vez.")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--marcadores", action="store_true",
                      help="escribe informe/resultados-test.tex con «?» en cada número, para que "
                           "el deck compile antes de la evaluación; no abre el test")
    modo.add_argument("--rehacer-tex", action="store_true",
                      help="rehace informe/resultados-test.tex desde "
                           "resultados/evaluacion_test.json; no abre el test")
    modo.add_argument("--rapido", action="store_true",
                      help="ensayo de punta a punta sobre 3 000 filas de train partidas 80/20; "
                           "escribe en un directorio temporal y no abre el test")
    parser.add_argument("--forzar", action="store_true",
                        help="sólo para pruebas manuales: responde «si» a la confirmación de una "
                             "reevaluación; la corrida igual queda registrada")
    return parser.parse_args(argv)


def main(argv=None):
    args = _argumentos(argv)
    if args.marcadores:
        escribir_marcadores(RUTA_TEX, RUTA_EVALUACION)
    elif args.rehacer_tex:
        rehacer_tex(RUTA_TEX, RUTA_EVALUACION)
    elif args.rapido:
        ensayo_rapido()
    else:
        evaluacion_unica(forzar=args.forzar)


if __name__ == "__main__":
    main()
