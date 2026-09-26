r"""Correr con: python3 -m src.numeros
            python3 -m src.numeros --verificar informe/presentacion.tex

Paso 7.2 del plan (gate G3). Todo número que aparece en la presentación y en el guion sale de un
macro generado aquí desde los resultados, nunca tipeado (guía de presentaciones, regla C7). Este
módulo es la única fuente de esos números.

Sin --verificar lee resultados/ (los archivos del contrato de N0-4 y N0-5), train con
cargar_train() para las cifras del EDA, el diccionario data/raw/bank-additional-names.txt e
informe/resultados-test.tex, y escribe:

- informe/numeros.tex: un \newcommand por cifra, cada uno en su línea con un comentario % que dice
  de qué archivo y columna sale. Los nombres son camelCase de letras solamente, porque LaTeX no
  admite dígitos en el nombre de un macro: \aucRfVal, \pctYesBloqueUno.
- informe/numeros.md: la misma lista legible (| macro | valor | fuente |), agrupada por slide del
  esqueleto del plan (§5, ola 7), más los macros de informe/resultados-test.tex, para que el guion y
  el revisor tengan todos los números en un solo lugar.

Formato español: coma decimal; tres decimales en las métricas (AUC, recall, precisión); cuatro en
los errores estándar y en los umbrales de 1 error estándar; los Δ con signo, con tres decimales
desde 0,01 y cuatro por debajo, como src/graficos.py; los porcentajes con un decimal y el signo
separado por un espacio fino (11{,}3\,\%); los miles con espacio fino (32\,940); el signo menos
tipográfico. Cada macro se lee igual en texto y dentro de $…$: la coma decimal va entre llaves
({,}), porque en modo matemático una coma suelta es puntuación y deja un espacio detrás («0, 795»),
y el menos es \signoMenos, que numeros.tex define en su cabecera como \textminus en texto y «-» en
matemática (\textminus es un comando de texto: dentro de $…$ LaTeX avisa y lo compone mal). En
numeros.md van los mismos valores con espacios comunes, coma y «−», como en el guion del TP1.

Si falta un archivo de entrada, sus macros valen «?» y numeros.md lo anota; nada falla. Es el caso
de resultados/evaluacion_test.json hasta la evaluación única (N0-1), y el de
resultados/evidencia_particion.json (D-02, D-03) hasta que src/evidencia_particion.py lo escriba:
esas cifras se miden sobre el CSV completo, que este módulo no puede leer (D-04). Los números de
test los escribe src/evaluar_test.py en informe/resultados-test.tex; aquí sólo se derivan los que
ese archivo no tiene (los «yes» del test y la diferencia con validación, que se dice en voz alta).

--verificar RUTA.tex [...] lista los números escritos a mano en el cuerpo de cada .tex, y en los
\input y \include de ese cuerpo, a cualquier profundidad, con archivo:línea, y termina con código 1
si encuentra alguno o si un \input no se encuentra. Cada \input se busca como lo hace LaTeX: en la
carpeta del .tex raíz, que es donde se compila, y si no está ahí, en la del archivo que lo incluye.
No cuentan como escritos a mano: los comentarios; el preámbulo; las definiciones (\newcommand,
\def, \setlength y parecidas), salvo las que definen un macro cuyo cuerpo es sólo una cifra
(\newcommand{\miauc}{0,79}), que se marcan fuera de numeros.tex y resultados-test.tex; la sintaxis de
TikZ (opciones, coordenadas, estilos), aunque el texto de cada nodo se lee; los argumentos que son
sintaxis (\includegraphics, \figgrande, \definecolor, \setbeamerfont, \fontsize, \multicolumn,
\phantom, colores, espacios); las opciones entre corchetes, salvo las que son texto (el título
corto de \section y parecidos, y el rótulo de \item, salvo «1.» o «a)»); las especificaciones de
overlay <…> (\begin{frame}<…>, \only<…>, \item<…>); los números que son parte de un nombre
(euribor3m, A7, F_1, x^2, azul!20) o de una medida (0.5em); y la lista PERMITIDOS (años 2008–2010,
fechas dd/mm, referencias a clases, slides y páginas, «10 minutos», «5 folds», «80/20», «k = 5», el
«20 %» del presupuesto cuando se habla de llamar o del presupuesto), que se amplía con --permitir
REGEX. Además avisa, sin fallar, si el .tex carga babel en español sin \spanishplainpercent: sin él,
babel agrega su propio espacio antes de % y el \,\% de los macros queda con espacio doble.
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.ablaciones import VARIANTES
from src.configuracion import OPCIONES_FINALES
from src.curvas import elegir_punto, leer_curvas, resumen_por_punto
from src.datos import EXCLUIDAS, FILA, OBJETIVO, PROP_TEST, RAIZ, cargar_train
from src.eda_html import (
    BLOQUE,
    N_DECILES,
    PDAYS_CENTINELA,
    RUTA_NAMES,
    UMBRAL_COLINEAL,
    anio_inferido,
    leer_names,
)
from src.estilo import ROTULO_LINEA_BASE
from src.evaluar_test import MACROS as MACROS_TEST
from src.evaluar_test import N_BOOTSTRAP, N_TOTAL, PERCENTILES
from src.experimentos import con_objetivo
from src.metricas import PRESUPUESTO, en_presupuesto, llamadas
from src.modelos import GRILLAS
from src.preproceso import MACRO, NUMERICAS_CLIENTE, Opciones, columnas, vocabulario
from src.resultados import DIR
from src.validacion import K, composicion, folds

RUTA_TEX = RAIZ / "informe" / "numeros.tex"
RUTA_MD = RAIZ / "informe" / "numeros.md"
RUTA_RESULTADOS_TEST = RAIZ / "informe" / "resultados-test.tex"
# Los archivos de macros generados: --verificar no los sigue cuando el cuerpo los incluye, porque
# sólo tienen definiciones.
GENERADOS = frozenset({RUTA_TEX.name, RUTA_RESULTADOS_TEST.name})

# Parámetros de los cálculos sobre train. Son elecciones, no resultados.
FACTOR_IQR = 1.5         # regla de Tukey, la de resultados/eda/reporte.txt, §5 (D-17)
ULTIMO_TRAMO = 0.2       # «el último 20 % por fecha» de la ablación A8 (plan, ola 1)
# Una lectura de la curva, no una regla: DECISIONES.md, D-22, «el AUC sube con k hasta una meseta
# desde 301 (0,7831) hasta 801 (0,7842)». El macro vale «?» si el punto deja de estar dentro de
# 1 error estándar del mejor (resultados/hiperparametros.json): entonces D-22 está desactualizada.
VECINOS_MESETA = 301
ESQUEMA_ADELANTE = "hacia_adelante"
ESQUEMA_BARAJADO = "barajado"

# El menos y la coma decimal, escritos para que el macro se lea bien en texto y dentro de $…$.
SIGNO_MENOS = "signoMenos"
MENOS = f"\\{SIGNO_MENOS} "
DEFINICION_MENOS = f"\\DeclareRobustCommand{{\\{SIGNO_MENOS}}}{{\\ifmmode-\\else\\textminus\\fi}}"
COMA = "{,}"
ESPACIO_FINO = "\\,"
POR_CIENTO = "\\,\\%"
DESCONOCIDO = "?"


# --- Formato ------------------------------------------------------------------------------------

def _miles(digitos):
    """'32940' -> '32\\,940'."""
    grupos = []
    while len(digitos) > 3:
        digitos, grupo = digitos[:-3], digitos[-3:]
        grupos.insert(0, grupo)
    return ESPACIO_FINO.join([digitos, *grupos])


def formatear_decimal(x, decimales=3, signo=False):
    r"""Coma decimal entre llaves y espacio fino de miles: 0.7952 -> '0{,}795'; 1234.5 con un
    decimal -> '1\,234{,}5'. Los negativos llevan \signoMenos; con signo=True, los demás llevan
    '+'. El signo sigue al valor sin redondear, como el formato de Python y las figuras: -0.00001
    con cuatro decimales es '\signoMenos 0{,}0000'."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"no se puede escribir {x} como cifra")
    entero, _, fraccion = f"{abs(x):.{decimales}f}".partition(".")
    cuerpo = _miles(entero) + (COMA + fraccion if fraccion else "")
    if x < 0:
        return MENOS + cuerpo
    return ("+" if signo else "") + cuerpo


def formatear_delta(x):
    """Un Δ con signo, con la regla de src/graficos.py: tres decimales desde 0,01 y cuatro por
    debajo, para que un efecto pequeño no se lea como cero."""
    return formatear_decimal(x, 3 if abs(float(x)) >= 0.01 else 4, signo=True)


def formatear_porcentaje(x, decimales=1):
    r"""x ya en porcentaje: 11.2663 -> '11{,}3\,\%'."""
    return formatear_decimal(x, decimales) + POR_CIENTO


def formatear_miles(n):
    r"""Un entero con espacio fino de miles: 32940 -> '32\,940'; 928 -> '928'."""
    valor = float(n)
    if not valor.is_integer():
        raise ValueError(f"{n} no es un entero")
    entero = int(valor)
    return (MENOS if entero < 0 else "") + _miles(str(abs(entero)))


def formatear_parametro(valor):
    """Un hiperparámetro como se lee: 801, 8, 0{,}001, 0{,}01. Los textos quedan como están."""
    if isinstance(valor, str):
        return valor
    v = float(valor)
    if v.is_integer():
        return formatear_miles(v)
    return np.format_float_positional(v, trim="-").replace(".", COMA)


def a_texto(valor):
    r"""El valor de un macro como se escribe en el guion: '32\,940' -> '32 940',
    '11{,}3\,\%' -> '11,3 %', '\signoMenos 0{,}028' -> '−0,028'. Los intervalos de
    informe/resultados-test.tex llevan «--», la raya de LaTeX: '0{,}812--0{,}834' -> '0,812–0,834'."""
    return (valor.replace(MENOS, "\N{MINUS SIGN}").replace(COMA, ",").replace(POR_CIENTO, " %")
            .replace(ESPACIO_FINO, " ").replace("\\%", "%").replace("--", "\N{EN DASH}"))


_UNIDADES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
             "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
             "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
             "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
             "veintinueve"]
_DECENAS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta",
            8: "ochenta", 9: "noventa"}
_CENTENAS = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos",
             5: "quinientos", 6: "seiscientos", 7: "setecientos", 8: "ochocientos",
             9: "novecientos"}


def numero_en_palabras(n):
    """0 <= n < 1 000 000 en palabras: 1 -> 'uno', 801 -> 'ochocientos uno', 2008 -> 'dos mil
    ocho'. Sirve para nombrar macros, que no admiten dígitos."""
    n = int(n)
    if n < 0 or n >= 1_000_000:
        raise ValueError(f"fuera de rango para un nombre de macro: {n}")
    if n < 30:
        return _UNIDADES[n]
    if n < 100:
        decena, unidad = divmod(n, 10)
        return _DECENAS[decena] + (f" y {_UNIDADES[unidad]}" if unidad else "")
    if n == 100:
        return "cien"
    if n < 1000:
        centena, resto = divmod(n, 100)
        return _CENTENAS[centena] + (f" {numero_en_palabras(resto)}" if resto else "")
    miles, resto = divmod(n, 1000)
    prefijo = "mil" if miles == 1 else f"{numero_en_palabras(miles)} mil"
    return prefijo + (f" {numero_en_palabras(resto)}" if resto else "")


def _sin_acentos(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def _palabras(parte):
    if isinstance(parte, (int, np.integer)) and not isinstance(parte, bool):
        return _sin_acentos(numero_en_palabras(int(parte))).split()
    palabras = []
    for trozo in re.findall(r"[A-Za-z]+|\d+", _sin_acentos(str(parte))):
        if trozo.isdigit():
            palabras += _sin_acentos(numero_en_palabras(int(trozo))).split()
        else:
            palabras += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", trozo)
    return palabras


def nombre_de_macro(*partes):
    """El nombre de un macro en camelCase de letras solamente, sin acentos. Los números van en
    palabras: ('pct yes bloque', 1) -> 'pctYesBloqueUno'; ('ganancia', 'RF', 20) ->
    'gananciaRfVeinte'; ('atípicos', 'cons.conf.idx') -> 'atipicosConsConfIdx'."""
    palabras = [p for parte in partes for p in _palabras(parte)]
    if not palabras:
        raise ValueError("un macro necesita al menos una palabra")
    nombre = palabras[0].lower() + "".join(p[:1].upper() + p[1:].lower() for p in palabras[1:])
    if not re.fullmatch(r"[A-Za-z]+", nombre):
        raise ValueError(f"nombre de macro inválido: {nombre!r}")
    return nombre


# --- Entradas -----------------------------------------------------------------------------------

class Faltante(Exception):
    """Un archivo o un dato de entrada que no está. Los macros que dependen de él valen «?»."""


class Fuentes:
    """Las entradas de los macros. Cada una se lee una sola vez; si no existe, levanta Faltante.

    `train` permite pasar un DataFrame con el formato de data/particion/train.csv en lugar de
    leerlo con cargar_train() (para las pruebas)."""

    def __init__(self, resultados=DIR, train=None, resultados_test=RUTA_RESULTADOS_TEST):
        self.resultados = Path(resultados)
        self.resultados_test = Path(resultados_test)
        self._train = train
        self._memo = {}

    def memoria(self, clave, calcular):
        if clave not in self._memo:
            try:
                self._memo[clave] = calcular()
            except Faltante as falta:
                self._memo[clave] = falta
        valor = self._memo[clave]
        if isinstance(valor, Faltante):
            raise Faltante(str(valor))
        return valor

    def ruta(self, nombre):
        ruta = self.resultados / nombre
        if not ruta.exists():
            raise Faltante(f"resultados/{nombre}")
        return ruta

    def csv(self, nombre):
        # round_trip: los mismos flotantes que se escribieron (src/experimentos.py, leer_oof).
        return self.memoria(("csv", nombre), lambda: pd.read_csv(self.ruta(nombre),
                                                                 float_precision="round_trip"))

    def json(self, nombre):
        return self.memoria(("json", nombre),
                            lambda: json.loads(self.ruta(nombre).read_text(encoding="utf-8")))

    def train(self):
        def leer():
            if self._train is not None:
                return self._train
            try:
                return cargar_train()
            except FileNotFoundError:
                raise Faltante("data/particion/train.csv") from None
        return self.memoria("train", leer)

    def curva(self, nombre):
        """La curva resultados/curvas/<nombre>.csv como la lee src/curvas.py (el punto sale del
        JSON de la configuración: pandas leería «None» como NaN) y su resumen por punto."""
        curvas = self.memoria("curvas", lambda: leer_curvas(self.resultados / "curvas"))
        if nombre not in curvas:
            raise Faltante(f"resultados/curvas/{nombre}.csv")
        return self.memoria(("curva", nombre),
                            lambda: (curvas[nombre], resumen_por_punto(curvas[nombre])))

    def names(self):
        def leer():
            if not RUTA_NAMES.exists():
                raise Faltante("data/raw/bank-additional-names.txt")
            texto = RUTA_NAMES.read_text(encoding="utf-8", errors="replace")
            instancias = re.search(r"Number of Instances:\s*(\d+)", texto)
            atributos = re.search(r"Number of Attributes:\s*(\d+)", texto)
            if not (instancias and atributos):
                raise Faltante("data/raw/bank-additional-names.txt: §5 y §6")
            return {"instancias": int(instancias.group(1)), "atributos": int(atributos.group(1)),
                    "names": leer_names()}
        return self.memoria("names", leer)

    def macros_test(self):
        """Los macros de informe/resultados-test.tex, {nombre: valor}, en su orden."""
        def leer():
            if not self.resultados_test.exists():
                raise Faltante(_mostrar(self.resultados_test))
            texto = self.resultados_test.read_text(encoding="utf-8")
            return dict(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", texto, re.M))
        return self.memoria("macros_test", leer)


def _mostrar(ruta):
    ruta = Path(ruta)
    try:
        return ruta.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return ruta.as_posix()


def _una_fila(tabla, que, **filtros):
    seleccion = tabla
    for columna, valor in filtros.items():
        if isinstance(valor, float):
            seleccion = seleccion[np.isclose(seleccion[columna].astype(float), valor)]
        else:
            seleccion = seleccion[seleccion[columna] == valor]
    if seleccion.empty:
        raise Faltante(f"{que}: " + ", ".join(f"{c} = {v}" for c, v in filtros.items()))
    if len(seleccion) > 1:
        raise ValueError(f"{que}: más de una fila con {filtros}")
    return seleccion.iloc[0]


def _anidado(datos, claves, que):
    for clave in claves:
        if not isinstance(datos, (dict, list)) or (isinstance(datos, dict) and clave not in datos):
            raise Faltante(f"{que}: " + ".".join(map(str, claves)))
        datos = datos[clave]
    return datos


# --- Datos: cada función devuelve otra que, dadas las Fuentes, da el número crudo -----------------

def _cv(etiqueta, modelo, conjunto, metrica, columna="media"):
    archivo = f"cv_{etiqueta}_resumen.csv"

    def dato(f):
        return _una_fila(f.csv(archivo), f"resultados/{archivo}", modelo=modelo, conjunto=conjunto,
                         metrica=metrica)[columna]
    return dato


def _cv_configuracion(etiqueta, modelo, clave):
    """Un hiperparámetro de la configuración con que se midió `modelo` en cv_<etiqueta>."""
    archivo = f"cv_{etiqueta}_resumen.csv"

    def dato(f):
        tabla = f.csv(archivo)
        configuraciones = tabla.loc[tabla["modelo"] == modelo, "configuracion"].unique()
        if len(configuraciones) != 1:
            raise Faltante(f"resultados/{archivo}: la configuración de {modelo}")
        return _anidado(json.loads(configuraciones[0]), [clave], f"resultados/{archivo}")
    return dato


def _cv_pareado(etiqueta, a, b, metrica, estadistico):
    """a − b fold a fold en validación, desde cv_<etiqueta>.csv: media, desvío, mínimo o máximo."""
    archivo = f"cv_{etiqueta}.csv"

    def dato(f):
        tabla = f.csv(archivo)
        val = tabla[(tabla["conjunto"] == "validacion") & (tabla["metrica"] == metrica)]
        por_fold = val.pivot_table(index="fold", columns="modelo", values="valor")
        if a not in por_fold or b not in por_fold:
            raise Faltante(f"resultados/{archivo}: {a} y {b}")
        diferencia = por_fold[a] - por_fold[b]
        return {"media": diferencia.mean(), "desvio": diferencia.std(ddof=1),
                "min": diferencia.min(), "max": diferencia.max()}[estadistico]
    return dato


def _ablacion(variante, modelo, columna):
    def dato(f):
        return _una_fila(f.csv("ablaciones_resumen.csv"), "resultados/ablaciones_resumen.csv",
                         modelo=modelo, variante=variante)[columna]
    return dato


def _ablacion_columnas(variante):
    def dato(f):
        tabla = f.csv("ablaciones_resumen.csv")
        valores = tabla.loc[tabla["variante"] == variante, "columnas"].unique()
        if len(valores) != 1:
            raise Faltante(f"resultados/ablaciones_resumen.csv: columnas de {variante}")
        return valores[0]
    return dato


def _columnas_de(opciones):
    """Las columnas que le llegan a los modelos con one-hot (RF, KNN, SVM, NB gaussiano) con estas
    opciones: numéricas, un nivel por categoría de VOCABULARIO y las binarias. Es la cuenta de
    src/preproceso.py sin ajustar nada; tests/test_numeros.py la compara con la columna
    `columnas` de resultados/ablaciones_resumen.csv."""
    numericas, categoricas, binarias = columnas(opciones)
    return len(numericas) + sum(len(vocabulario(c, opciones)) for c in categoricas) + len(binarias)


def _robustez(modelo, esquema, variante, metrica="auc", columna="media"):
    def dato(f):
        return _una_fila(f.csv("robustez_temporal_resumen.csv"),
                         "resultados/robustez_temporal_resumen.csv", modelo=modelo,
                         esquema=esquema, variante=variante, metrica=metrica)[columna]
    return dato


def _robustez_fold(modelo, esquema, variante, fold, metrica="auc"):
    def dato(f):
        return _una_fila(f.csv("robustez_temporal.csv"), "resultados/robustez_temporal.csv",
                         modelo=modelo, esquema=esquema, variante=variante, fold=fold,
                         conjunto="validacion", metrica=metrica)["valor"]
    return dato


def _bloque(esquema, fold, columna):
    def dato(f):
        return _una_fila(f.csv("robustez_folds.csv"), "resultados/robustez_folds.csv",
                         esquema=esquema, fold=fold)[columna]
    return dato


def _sensibilidad(modelo, q, metrica):
    """La media de los folds de sensibilidad_q_final.csv para un modelo, un q y una métrica."""
    def dato(f):
        tabla = f.csv("sensibilidad_q_final.csv")
        sel = tabla[(tabla["modelo"] == modelo) & np.isclose(tabla["q"], q)
                    & (tabla["metrica"] == metrica)]
        if sel.empty:
            raise Faltante(f"resultados/sensibilidad_q_final.csv: {modelo}, q = {q}, {metrica}")
        return sel["valor"].mean()
    return dato


def _q_extremo(cual):
    def dato(f):
        qs = f.csv("sensibilidad_q_final.csv")["q"]
        return 100 * (qs.min() if cual == "min" else qs.max())
    return dato


def _ganancia(modelo, p):
    def dato(f):
        return _una_fila(f.csv("ganancia_final.csv"), "resultados/ganancia_final.csv",
                         modelo=modelo, p=p)["pct_yes_alcanzado"]
    return dato


def _ganancia_alcanza(modelo, pct):
    """El menor p (% de la lista llamado) con el que se alcanza al menos el pct % de los «yes»."""
    def dato(f):
        tabla = f.csv("ganancia_final.csv")
        sel = tabla[(tabla["modelo"] == modelo) & (tabla["pct_yes_alcanzado"] >= pct)]
        if sel.empty:
            raise Faltante(f"resultados/ganancia_final.csv: {modelo} no alcanza el {pct} %")
        return sel["p"].min()
    return dato


def _curva(nombre, punto, columna):
    def dato(f):
        _, resumen = f.curva(nombre)
        if punto not in resumen.index:
            raise Faltante(f"resultados/curvas/{nombre}.csv: punto {punto}")
        return resumen.loc[punto, columna]
    return dato


def _curva_configuracion(nombre, punto, clave):
    def dato(f):
        tabla, _ = f.curva(nombre)
        configuraciones = tabla.loc[tabla["punto"] == punto, "configuracion"].unique()
        if len(configuraciones) != 1:
            raise Faltante(f"resultados/curvas/{nombre}.csv: punto {punto}")
        return _anidado(json.loads(configuraciones[0]), [clave], f"resultados/curvas/{nombre}.csv")
    return dato


def _hiper(curva, clave):
    def dato(f):
        return _anidado(f.json("hiperparametros.json"), ["curvas", curva, clave],
                        "resultados/hiperparametros.json")
    return dato


def _eleccion(curva, clave):
    """Lo que elige D-22 (src/curvas.py, elegir_punto) sobre resultados/curvas/<curva>.csv tal
    como está, sin pasar por resultados/hiperparametros.json: una curva que se extiende después de
    escribir ese JSON deja ahí su entrada vieja. Es el caso de KNN con weights = distance, cuya
    grilla se extendió hasta 801 después de escribirlo."""
    def dato(f):
        tabla, _ = f.curva(curva)
        eleccion = f.memoria(("eleccion", curva), lambda: elegir_punto(tabla))
        if eleccion.get(clave) is None:
            raise Faltante(f"resultados/curvas/{curva}.csv: {clave}")
        return eleccion[clave]
    return dato


def _hiper_minimo_dentro(curva):
    """El menor de los puntos que quedan dentro de 1 error estándar del mejor."""
    def dato(f):
        puntos = _hiper(curva, "puntos_dentro_1es")(f)
        return min(float(p) for p in puntos)
    return dato


def _meseta_knn(f):
    """VECINOS_MESETA, mientras siga dentro de 1 error estándar del mejor de la curva de KNN."""
    dentro = _hiper("knn_n_neighbors_uniform", "puntos_dentro_1es")(f)
    if str(VECINOS_MESETA) not in dentro:
        raise Faltante(f"DECISIONES.md, D-22: k = {VECINOS_MESETA} ya no está dentro de 1 ES en "
                       "resultados/hiperparametros.json (curvas.knn_n_neighbors_uniform)")
    return VECINOS_MESETA


def _grilla_minima(modelo, parametro):
    """El menor valor numérico de la grilla de src/modelos.py (max_depth también tiene None)."""
    return min(v for v in GRILLAS[(modelo, parametro)] if v is not None)


def _elegido(*claves):
    def dato(f):
        return _anidado(f.json("modelo_elegido.json"), list(claves), "resultados/modelo_elegido.json")
    return dato


def _ranking(posicion, clave):
    def dato(f):
        ranking = _elegido("ranking")(f)
        if not -len(ranking) <= posicion < len(ranking):
            raise Faltante(f"resultados/modelo_elegido.json: ranking[{posicion}]")
        return ranking[posicion][clave]
    return dato


def _evaluacion(*claves):
    def dato(f):
        return _anidado(f.json("evaluacion_test.json"), list(claves),
                        "resultados/evaluacion_test.json")
    return dato


# La evidencia de D-02 y D-03 se mide sobre el CSV completo, test incluido, así que este módulo no
# puede calcularla (D-04, tests/test_aislamiento.py): la lee de lo que escriba
# src/evidencia_particion.py. El contrato de ese archivo:
#   {"n_semillas": 400,
#    "sin_estratificar": {"desvio_pp": ..., "minimo_pct": ..., "maximo_pct": ...},
#    "temporal": {"pct_yes_train": ..., "pct_yes_test": ...}}
# con los porcentajes en puntos (11.27, no 0.1127): el % de «yes» del test en n_semillas
# particiones aleatorias sin estratificar (desvío, mínimo y máximo), y el de train y test si el
# test fuera el último 20 % del archivo por fecha.
EVIDENCIA_PARTICION = "evidencia_particion.json"


def _evidencia(*claves):
    def dato(f):
        return _anidado(f.json(EVIDENCIA_PARTICION), list(claves), f"resultados/{EVIDENCIA_PARTICION}")
    return dato


def _costo_excedido(clave):
    """De resultados/costos.csv, la corrida que pasó el límite: el límite en segundos o su C."""
    def dato(f):
        tabla = f.csv("costos.csv")
        excedidas = tabla[tabla["estado"].astype(str).str.startswith("excedido")]
        if excedidas.empty:
            raise Faltante("resultados/costos.csv: ninguna corrida excedida")
        fila = excedidas.iloc[0]
        if clave == "segundos":
            return int(re.search(r"(\d+)\s*s\b", fila["estado"]).group(1))
        return _anidado(json.loads(fila["parametros"]), [clave], "resultados/costos.csv")
    return dato


def _matriz_oof(modelo, celda):
    """La matriz de confusión de validación con todos los puntajes fuera de fold juntos, llamando
    al PRESUPUESTO de la lista con mayor puntaje: el mismo corte que la curva de ganancia
    (src/experimentos.py, ganancia) y que la matriz de test. Los empates en el corte se reparten
    en proporción, como src/metricas.py."""
    def dato(f):
        def calcular():
            oof = con_objetivo(f.csv(f"oof_final_{modelo}.csv"), f.train())
            y, s = oof["y"].to_numpy(), oof["puntaje"].to_numpy()
            r = en_presupuesto(y, s, PRESUPUESTO)
            k, positivos = r["llamadas"], int(y.sum())
            vp = r["recall_q"] * positivos
            return {"vp": vp, "fp": k - vp, "fn": positivos - vp, "vn": len(y) - k - (positivos - vp)}
        return f.memoria(("matriz", modelo), calcular)[celda]
    return dato


def _oof_rf(f):
    """Los puntajes fuera de fold de RF (configuración final) con la `y` (0/1) de train."""
    return f.memoria("oof_rf", lambda: con_objetivo(f.csv("oof_final_rf.csv"), f.train()))


def _pares_por_anio(clave):
    """El AUC fuera de fold de RF separado en los pares «yes»–«no» del mismo año y los de años
    distintos, con el año inferido como eda.html: la lectura (a) del hallazgo
    (resultados/conclusiones.md, §2 y «Cómo se calculó»). El AUC es la proporción de pares bien
    ordenados (un empate, medio), así que el de los pares del mismo año es el AUC de cada año
    ponderado por sus pares, y el de los pares de años distintos sale por diferencia con el AUC de
    todo junto. Claves: pct_entre, auc_entre, auc_dentro y auc_<año>."""
    def dato(f):
        def calcular():
            train = f.train()
            anio = pd.Series(anio_inferido(train, f.names()["names"]), index=train[FILA].to_numpy())
            d = _oof_rf(f)
            d = d.assign(anio=d[FILA].map(anio).to_numpy())
            y = d["y"].to_numpy()
            positivos = int(y.sum())
            pares = positivos * (len(y) - positivos)
            if not pares:
                raise Faltante("resultados/oof_final_rf.csv: sin «yes» o sin «no»")
            r, dentro, suma = {}, 0, 0.0
            for a, g in d.groupby("anio"):
                p = int(g["y"].sum())
                pares_anio = p * (len(g) - p)
                if pares_anio:
                    r[f"auc_{int(a)}"] = roc_auc_score(g["y"], g["puntaje"])
                    dentro += pares_anio
                    suma += r[f"auc_{int(a)}"] * pares_anio
            if not dentro or dentro == pares:
                raise Faltante("resultados/oof_final_rf.csv: no hay pares de un solo año y de dos")
            r["pct_entre"] = 100 * (pares - dentro) / pares
            r["auc_dentro"] = suma / dentro
            r["auc_entre"] = (roc_auc_score(y, d["puntaje"]) * pares - suma) / (pares - dentro)
            return r
        pares_por_anio = f.memoria("pares_por_anio", calcular)
        if clave not in pares_por_anio:
            raise Faltante(f"resultados/oof_final_rf.csv: {clave}")
        return pares_por_anio[clave]
    return dato


def _barajado_en_bloque(fold):
    """El AUC fuera de fold de RF (validación barajada) sólo sobre las filas del bloque `fold` de
    la validación hacia adelante (resultados/robustez_folds.csv): la referencia «barajado, en las
    mismas filas» de la slide 19, lecturas (b) y (c) del hallazgo."""
    def dato(f):
        bloque = _una_fila(f.csv("robustez_folds.csv"), "resultados/robustez_folds.csv",
                           esquema=ESQUEMA_ADELANTE, fold=fold)
        d = _oof_rf(f)
        en_bloque = d[d[FILA].between(bloque["fila_min_validacion"], bloque["fila_max_validacion"])]
        if en_bloque["y"].nunique() < 2:
            raise Faltante(f"resultados/oof_final_rf.csv: el bloque {fold} no tiene las dos clases")
        return roc_auc_score(en_bloque["y"], en_bloque["puntaje"])
    return dato


def _yes_entrenamiento_adelante(fold):
    """Los «yes» de train con que entrena el fold `fold` hacia adelante: los de las filas hasta su
    fila_max_train (resultados/robustez_folds.csv)."""
    def dato(f):
        bloque = _una_fila(f.csv("robustez_folds.csv"), "resultados/robustez_folds.csv",
                           esquema=ESQUEMA_ADELANTE, fold=fold)
        train = f.train()
        return int(((train[FILA] <= bloque["fila_max_train"]) & (train[OBJETIVO] == "yes")).sum())
    return dato


def _eda(clave):
    """Una cifra calculada sobre train (ver _calcular_eda)."""
    def dato(f):
        eda = f.memoria("eda", lambda: _calcular_eda(f.train()))
        if clave not in eda:
            raise KeyError(f"_calcular_eda no calcula {clave!r}")
        return eda[clave]
    return dato


def _calcular_eda(df):
    """Las cifras del EDA que usa el deck, sobre train y con las mismas definiciones que
    src/eda_html.py (analizar) y resultados/eda/reporte.txt. tests/test_numeros.py compara las
    compartidas con analizar()."""
    y = (df[OBJETIVO] == "yes").astype(int)
    n, n_yes = len(df), int(y.sum())
    r = {"filas": n, "yes": n_yes, "no": n - n_yes, "pct_yes": 100 * n_yes / n,
         "exactitud_no": 100 * (n - n_yes) / n, "razon_no_yes": (n - n_yes) / n_yes}

    # duration (D-05): los deciles de reporte.txt, §8, y su AUC como puntaje solo.
    d = df["duration"]
    deciles, cortes = pd.qcut(d, N_DECILES, retbins=True)
    tasa_decil = 100 * y.groupby(deciles, observed=True).mean()
    r.update(duration_corto=cortes[1], duration_largo=cortes[-2],
             pct_yes_duration_corto=tasa_decil.iloc[0], pct_yes_duration_largo=tasa_decil.iloc[-1],
             duration_cero=int((d == 0).sum()), auc_duration=roc_auc_score(y, d))

    # Orden temporal (eda.html, «El tiempo»): bloques de BLOQUE filas consecutivas del CSV original.
    bloque = df[FILA] // BLOQUE
    tasa_bloque = 100 * y.groupby(bloque).mean()
    euribor_bloque = df["euribor3m"].groupby(bloque).mean()
    r.update(pct_yes_primer_bloque=tasa_bloque.iloc[0], pct_yes_ultimo_bloque=tasa_bloque.iloc[-1],
             euribor_primer_bloque=euribor_bloque.iloc[0], euribor_minimo_bloque=euribor_bloque.min())

    # A8: qué parte de cada mes cae en el último tramo de train ordenado por fecha (por `fila`).
    meses = df.sort_values(FILA)["month"].to_numpy()
    ultimo = np.arange(n) >= int(np.floor((1 - ULTIMO_TRAMO) * n))
    for mes in ("sep", "oct", "dec"):
        r[f"pct_ultimo_{mes}"] = 100 * ultimo[meses == mes].mean()

    # D-06: el euribor medio de cada fold, sin barajar (lo que hace cv=5) y con folds().
    sin_barajar = composicion(StratifiedKFold(K), df)["euribor3m_medio"]
    barajado = composicion(folds(), df)["euribor3m_medio"]
    r.update(euribor_fold_sin_barajar_primero=sin_barajar.iloc[0],
             euribor_fold_sin_barajar_ultimo=sin_barajar.iloc[-1],
             euribor_fold_barajado_min=barajado.min(), euribor_fold_barajado_max=barajado.max())

    # pdays (D-10) y poutcome.
    centinela = df["pdays"] == PDAYS_CENTINELA
    exito = df["poutcome"] == "success"
    r.update(pct_pdays_centinela=100 * centinela.mean(),
             centinela_con_previo=int((centinela & (df["previous"] >= 1)).sum()),
             pdays_contactados=int((~centinela).sum()),
             pct_yes_pdays_contactados=100 * y[~centinela].mean(),
             pct_yes_pdays_centinela=100 * y[centinela].mean(),
             pct_poutcome_inexistente=100 * (df["poutcome"] == "nonexistent").mean(),
             pct_yes_poutcome_exito=100 * y[exito].mean())

    # default (D-08, D-09).
    unknown = df["default"] == "unknown"
    r.update(pct_default_unknown=100 * unknown.mean(), pct_yes_default_unknown=100 * y[unknown].mean(),
             pct_yes_default_conocido=100 * y[~unknown].mean(),
             default_yes=int((df["default"] == "yes").sum()))

    # campaign (D-07, D-13).
    campaign = df["campaign"]
    r.update(media_campaign_yes=campaign[y == 1].mean(), media_campaign_no=campaign[y == 0].mean(),
             mediana_campaign_yes=campaign[y == 1].median(),
             mediana_campaign_no=campaign[y == 0].median(),
             maximo_campaign=int(campaign.max()), asimetria_campaign=campaign.skew())

    # Atípicos por IQR (D-17), sobre las 9 numéricas del modelo: sin duration ni fila.
    numericas = NUMERICAS_CLIENTE + MACRO
    q1, q3 = df[numericas].quantile(0.25), df[numericas].quantile(0.75)
    iqr = q3 - q1
    fuera = (df[numericas] < q1 - FACTOR_IQR * iqr) | (df[numericas] > q3 + FACTOR_IQR * iqr)
    alguna = fuera.any(axis=1)
    r.update(atipicas=int(alguna.sum()), pct_atipicas=100 * alguna.mean())
    for columna in numericas:
        r[f"atipicos_{columna}"] = int(fuera[columna].sum())
        r[f"pct_yes_atipicos_{columna}"] = (100 * y[fuera[columna]].mean() if fuera[columna].any()
                                            else float("nan"))

    # Escalas (D-12): desvíos y parte de la varianza total de las numéricas del modelo.
    desvios, varianzas = df[numericas].std(), df[numericas].var()
    r.update(desvio_pdays=desvios["pdays"], desvio_nr_employed=desvios["nr.employed"],
             pct_varianza_pdays=100 * varianzas["pdays"] / varianzas.sum(),
             pct_varianza_nr_employed=100 * varianzas["nr.employed"] / varianzas.sum(),
             razon_desvios=desvios.max() / desvios.min())

    # Bloque macro (D-14): los pares casi colineales, |r| >= UMBRAL_COLINEAL.
    correlacion = df[MACRO].corr().abs()
    pares = [correlacion.loc[a, b] for i, a in enumerate(MACRO) for b in MACRO[i + 1:]
             if correlacion.loc[a, b] >= UMBRAL_COLINEAL]
    r.update(correlacion_macro_min=min(pares) if pares else float("nan"),
             correlacion_macro_max=max(pares) if pares else float("nan"))
    return r


def _tasa_anio(anio):
    """% de «yes» de train en un año, con el año inferido del orden de los meses como eda.html."""
    def dato(f):
        def calcular():
            df = f.train()
            anios = anio_inferido(df, f.names()["names"])
            y = (df[OBJETIVO] == "yes").astype(int).to_numpy()
            return {int(a): 100 * y[anios == a].mean() for a in sorted(set(anios))}
        tasas = f.memoria("tasa_anio", calcular)
        if anio not in tasas:
            raise Faltante(f"train: no hay filas de {anio}")
        return tasas[anio]
    return dato


def _names(clave):
    def dato(f):
        names = f.names()
        if clave in ("anio_ini", "anio_fin"):
            return getattr(names["names"], clave)
        return names[clave]
    return dato


def _constante(valor):
    return lambda f: valor


def _resta(a, b):
    return lambda f: a(f) - b(f)


def _cociente(a, b):
    return lambda f: a(f) / b(f)


# --- Catálogo -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Numero:
    nombre: str
    fuente: str
    calcular: Callable


def _num(nombre, fuente, dato, formato):
    return Numero(nombre, fuente, lambda f: formato(dato(f)))


def _metrica(x):
    return formatear_decimal(x, 3)


def _error(x):
    return formatear_decimal(x, 4)


def _dos(x):
    return formatear_decimal(x, 2)


def _uno(x):
    return formatear_decimal(x, 1)


def _pct(x):
    return formatear_porcentaje(x, 1)


def _pct_entero(x):
    return formatear_porcentaje(x, 0)


def _desvio_delta(x):
    """El desvío de un Δ, con los decimales del Δ: tres desde 0,01 y cuatro por debajo."""
    return formatear_decimal(x, 3 if abs(float(x)) >= 0.01 else 4)


def _anio(x):
    return str(int(x))


def _conteo(x):
    """Un conteo que puede ser fraccionario si un empate en el corte se reparte en proporción."""
    x = float(x)
    return formatear_miles(round(x)) if abs(x - round(x)) < 1e-6 else formatear_decimal(x, 1)


def _texto(x):
    return str(x)


# Nombres de los modelos en los macros y en el texto.
MACRO_MODELO = {"rf": "Rf", "knn": "Knn", "svm": "Svm", "nb_categorico": "Nb",
                "nb_gaussiano": "NbGaussiano", "sin_modelo": "SinModelo"}
ROTULO_MODELO = {"rf": "Random Forest", "knn": "KNN", "svm": "SVM", "nb_categorico": "Naive Bayes",
                 "nb_gaussiano": "Naive Bayes gaussiano"}
ROTULO_KERNEL = {"linear": "lineal", "poly": "polinómico", "rbf": "RBF"}
FINALES = ("rf", "knn", "nb_categorico", "svm")
# Los modelos de cv_referencia y la base de sus macros. El NB gaussiano sólo se midió aquí, con el
# nombre que pide el esqueleto (\aucNbGaussianoVal), y la SVM de referencia es la de RBF y C = 1.
BASE_REFERENCIA = {"nb_categorico": "NbReferencia", "nb_gaussiano": "NbGaussianoVal",
                   "svm": "SvmRbfReferencia", "knn": "KnnReferencia", "rf": "RfReferencia"}
# Qué cambia cada ablación, en el nombre de sus macros (src/ablaciones.py, VARIANTES).
MACRO_VARIANTE = {"A1": "Duration", "A2": "SinPdays", "A3": "DefaultIndicadora", "A4": "Raras",
                  "A5": "LogCampaign", "A6": "MacroReducido", "A7": "SinMacro",
                  "A8": "SinMacroNiMonth", "A9": "SinDiaSemana", "A10": "EdadTramos",
                  "A11": "ImputarModa", "A12": "SinEscalar"}
# Las variantes de columnas de D-25 (src/robustez.py) en el nombre de sus macros.
MACRO_ROBUSTEZ = {"todas": "", "sin_macro": "SinMacro", "sin_macro_ni_month": "SinMacroNiMonth"}

CV_FINAL = "resultados/cv_final_resumen.csv"
CV_REFERENCIA = "resultados/cv_referencia_resumen.csv"
ABLACIONES = "resultados/ablaciones_resumen.csv"
ROBUSTEZ = "resultados/robustez_temporal_resumen.csv"
TRAIN = "train (cargar_train())"
NAMES = "data/raw/bank-additional-names.txt"


def _ablaciones_de(variantes):
    """Δ AUC de validación pareado contra A0, con su desvío, para cada modelo en que se midió."""
    numeros = []
    for variante in variantes:
        for modelo in VARIANTES[variante].modelos:
            base = nombre_de_macro("delta auc", MACRO_VARIANTE[variante], MACRO_MODELO[modelo])
            donde = f"{ABLACIONES}: {modelo}, {variante}"
            numeros += [
                _num(base, f"{donde}, delta_media (ΔAUC pareado contra A0)",
                     _ablacion(variante, modelo, "delta_media"), formatear_delta),
                _num(base + "Desvio", f"{donde}, delta_desvio",
                     _ablacion(variante, modelo, "delta_desvio"), _desvio_delta),
            ]
    return numeros


def _seccion_problema():
    return [
        _num("filasCsv", f"{NAMES}, §5: Number of Instances", _names("instancias"), formatear_miles),
        _num("predictoras", f"{NAMES}, §6: Number of Attributes (sin y)", _names("atributos"),
             formatear_miles),
        _num("filasTrain", f"{TRAIN}: filas", _eda("filas"), formatear_miles),
        _num("yesTrain", f"{TRAIN}: filas con y = yes", _eda("yes"), formatear_miles),
        _num("noTrain", f"{TRAIN}: filas con y = no", _eda("no"), formatear_miles),
        _num("pctYesTrain", f"{TRAIN}: % de y = yes", _eda("pct_yes"), _pct),
        _num("exactitudSiempreNo", f"{TRAIN}: % de y = no, la exactitud de decir siempre «no»",
             _eda("exactitud_no"), _pct),
        _num("razonNoYes", f"{TRAIN}: filas «no» por cada «yes»", _eda("razon_no_yes"), _uno),
    ]


def _seccion_datos():
    numeros = [
        _num("filasSinDuplicados", "src/evaluar_test.py: N_TOTAL, el CSV sin duplicados (D-01)",
             _constante(N_TOTAL), formatear_miles),
        _num("duplicados", f"{NAMES}, §5, menos N_TOTAL de src/evaluar_test.py (D-01)",
             _resta(_names("instancias"), _constante(N_TOTAL)), formatear_miles),
        _num("filasTest", "src/evaluar_test.py: N_TOTAL menos filasTrain (D-02)",
             _resta(_constante(N_TOTAL), _eda("filas")), formatear_miles),
        _num("llamadasTest", "src/metricas.py: llamadas(filasTest), el presupuesto de D-20",
             lambda f: llamadas(N_TOTAL - _eda("filas")(f), PRESUPUESTO), formatear_miles),
        _num("proporcionTrain", "src/datos.py: 1 − PROP_TEST", _constante(100 * (1 - PROP_TEST)),
             _pct_entero),
        _num("proporcionTest", "src/datos.py: PROP_TEST", _constante(100 * PROP_TEST), _pct_entero),
        _num("kFolds", "resultados/modelo_elegido.json: k_folds", _elegido("k_folds"),
             formatear_miles),
        _num("semilla", "resultados/modelo_elegido.json: semilla", _elegido("semilla"),
             formatear_miles),
        _num("filasFoldEntrenamiento", "resultados/robustez_folds.csv: barajado, fold 1, n_train",
             _bloque(ESQUEMA_BARAJADO, 1, "n_train"), formatear_miles),
        _num("filasFoldValidacion", "resultados/robustez_folds.csv: barajado, fold 1, n_validacion",
             _bloque(ESQUEMA_BARAJADO, 1, "n_validacion"), formatear_miles),
        _num("llamadasFoldValidacion", "src/metricas.py: llamadas(filasFoldValidacion)",
             lambda f: llamadas(int(_bloque(ESQUEMA_BARAJADO, 1, "n_validacion")(f)), PRESUPUESTO),
             formatear_miles),
        _num("llamadasTrain", "src/metricas.py: llamadas(filasTrain), el corte de la lista fuera "
             "de fold", lambda f: llamadas(_eda("filas")(f), PRESUPUESTO), formatear_miles),
    ]
    evidencia = f"resultados/{EVIDENCIA_PARTICION}"
    return numeros + [
        _num("semillasSinEstratificar", f"{evidencia}: n_semillas, las particiones sin estratificar "
             "(D-02)", _evidencia("n_semillas"), formatear_miles),
        _num("desvioPctYesTestSinEstratificar", f"{evidencia}: sin_estratificar.desvio_pp, el desvío "
             "del % de «yes» del test entre semillas, en puntos porcentuales (D-02)",
             _evidencia("sin_estratificar", "desvio_pp"), _dos),
        _num("pctYesTestSinEstratificarMinimo", f"{evidencia}: sin_estratificar.minimo_pct, el menor "
             "% de «yes» del test entre semillas (D-02)", _evidencia("sin_estratificar", "minimo_pct"),
             _pct),
        _num("pctYesTestSinEstratificarMaximo", f"{evidencia}: sin_estratificar.maximo_pct, el mayor "
             "(D-02)", _evidencia("sin_estratificar", "maximo_pct"), _pct),
        _num("pctYesTrainTemporal", f"{evidencia}: temporal.pct_yes_train, % de «yes» de train si el "
             "test fuera el último 20 % del archivo (D-03)", _evidencia("temporal", "pct_yes_train"),
             _pct),
        _num("pctYesTestTemporal", f"{evidencia}: temporal.pct_yes_test, el del test (D-03)",
             _evidencia("temporal", "pct_yes_test"), _pct),
    ]


def _seccion_duration():
    numeros = [
        _num("durationDecilCorto", f"{TRAIN}: duration, borde superior del primer decil (s)",
             _eda("duration_corto"), formatear_parametro),
        _num("durationDecilLargo", f"{TRAIN}: duration, borde inferior del último decil (s)",
             _eda("duration_largo"), formatear_parametro),
        _num("pctYesDurationDecilCorto", f"{TRAIN}: % de «yes» en el primer decil de duration",
             _eda("pct_yes_duration_corto"), _pct),
        _num("pctYesDurationDecilLargo", f"{TRAIN}: % de «yes» en el último decil de duration",
             _eda("pct_yes_duration_largo"), _pct),
        _num("filasDurationCero", f"{TRAIN}: filas con duration = 0 (todas «no»)",
             _eda("duration_cero"), formatear_miles),
        _num("aucDurationSola", f"{TRAIN}: AUC de duration sola como puntaje", _eda("auc_duration"),
             _metrica),
        _num("predictorasDisponibles", f"{NAMES}, §6, menos las predictoras que src/datos.py deja "
             "fuera del modelo (EXCLUIDAS sin fila: duration); D-05 y D-07",
             lambda f: _names("atributos")(f) - len([c for c in EXCLUIDAS if c != FILA]),
             formatear_miles),
    ]
    for modelo in ("rf", "nb_gaussiano", "svm", "knn"):
        numeros.append(_num(nombre_de_macro("auc", MACRO_MODELO[modelo], "con duration"),
                            f"{ABLACIONES}: {modelo}, A1, auc_variante (techo con duration)",
                            _ablacion("A1", modelo, "auc_variante"), _metrica))
    return numeros + _ablaciones_de(["A1"])


def _seccion_tiempo():
    numeros = [
        _num("anioInicio", f"{NAMES}, §4: primer año del archivo", _names("anio_ini"), _anio),
        _num("anioFin", f"{NAMES}, §4: último año del archivo", _names("anio_fin"), _anio),
        _num("filasBloqueEda", "src/eda_html.py: BLOQUE, filas del CSV original por bloque",
             _constante(BLOQUE), formatear_miles),
        _num("pctYesPrimerBloqueEda", f"{TRAIN}: % de «yes» en el primer bloque de BLOQUE filas",
             _eda("pct_yes_primer_bloque"), _pct),
        _num("pctYesUltimoBloqueEda", f"{TRAIN}: % de «yes» en el último bloque de BLOQUE filas",
             _eda("pct_yes_ultimo_bloque"), _pct),
        _num("factorYesBloquesEda", f"{TRAIN}: % de «yes» del último bloque / del primero",
             _cociente(_eda("pct_yes_ultimo_bloque"), _eda("pct_yes_primer_bloque")), _uno),
    ]
    anios = (2008, 2009, 2010)
    for anio in anios:
        numeros.append(_num(nombre_de_macro("pct yes", anio),
                            f"{TRAIN}: % de «yes» en {anio} (año inferido del orden de los meses, "
                            "como eda.html)", _tasa_anio(anio), _pct))
    numeros += [
        _num("factorYesAnios", f"{TRAIN}: % de «yes» de {anios[-1]} / de {anios[0]}",
             _cociente(_tasa_anio(anios[-1]), _tasa_anio(anios[0])), _uno),
        _num("euriborPrimerBloqueEda", f"{TRAIN}: euribor3m medio del primer bloque de BLOQUE filas",
             _eda("euribor_primer_bloque"), _dos),
        _num("euriborMinimoBloqueEda", f"{TRAIN}: euribor3m medio del bloque más bajo",
             _eda("euribor_minimo_bloque"), _dos),
        _num("euriborFoldSinBarajarPrimero",
             f"{TRAIN}: euribor3m medio del fold 1 con StratifiedKFold(5) sin barajar, cv=5 (D-06)",
             _eda("euribor_fold_sin_barajar_primero"), _dos),
        _num("euriborFoldSinBarajarUltimo", f"{TRAIN}: ídem, fold 5 (D-06)",
             _eda("euribor_fold_sin_barajar_ultimo"), _dos),
        _num("euriborFoldBarajadoMinimo", f"{TRAIN}: euribor3m medio, el menor de los folds() (D-06)",
             _eda("euribor_fold_barajado_min"), _dos),
        _num("euriborFoldBarajadoMaximo", f"{TRAIN}: ídem, el mayor (D-06)",
             _eda("euribor_fold_barajado_max"), _dos),
    ]
    for mes, palabra in (("sep", "Septiembre"), ("oct", "Octubre"), ("dec", "Diciembre")):
        numeros.append(_num(f"pct{palabra}UltimoTramo",
                            f"{TRAIN}: % de las filas de month = {mes} que caen en el último "
                            "20 % de train por fecha (A8)", _eda(f"pct_ultimo_{mes}"), _pct))
    return numeros


def _seccion_pdays():
    return [
        _num("pdaysCentinela", "src/eda_html.py: PDAYS_CENTINELA, el pdays de quien no fue contactado "
             "en una campaña anterior (D-10)", _constante(PDAYS_CENTINELA), formatear_miles),
        _num("pctPdaysCentinela", f"{TRAIN}: % de filas con pdays = 999", _eda("pct_pdays_centinela"),
             _pct),
        _num("filasCentinelaConPrevio", f"{TRAIN}: filas con pdays = 999 y previous >= 1",
             _eda("centinela_con_previo"), formatear_miles),
        _num("filasPdaysContactados", f"{TRAIN}: filas con pdays < 999", _eda("pdays_contactados"),
             formatear_miles),
        _num("pctYesPdaysContactados", f"{TRAIN}: % de «yes» con pdays < 999",
             _eda("pct_yes_pdays_contactados"), _pct),
        _num("pctYesPdaysCentinela", f"{TRAIN}: % de «yes» con pdays = 999",
             _eda("pct_yes_pdays_centinela"), _pct),
        _num("pctPoutcomeInexistente", f"{TRAIN}: % de filas con poutcome = nonexistent",
             _eda("pct_poutcome_inexistente"), _pct),
        _num("pctYesPoutcomeExito", f"{TRAIN}: % de «yes» con poutcome = success",
             _eda("pct_yes_poutcome_exito"), _pct),
    ] + _ablaciones_de(["A2"])


def _seccion_pipeline():
    numeros = [
        _num("columnasReferencia", "src/preproceso.py: columnas del one-hot con Opciones(), la "
             "referencia de las ablaciones", _constante(_columnas_de(Opciones())), formatear_miles),
        _num("columnasFinales", "src/preproceso.py: columnas del one-hot con OPCIONES_FINALES "
             "(src/configuracion.py)", _constante(_columnas_de(OPCIONES_FINALES)), formatear_miles),
    ]
    for variante, nombre in MACRO_VARIANTE.items():
        numeros.append(_num(nombre_de_macro("columnas", nombre), f"{ABLACIONES}: {variante}, columnas",
                            _ablacion_columnas(variante), formatear_miles))
    for modelo in ("rf", "nb_gaussiano", "svm", "knn"):
        numeros.append(_num(nombre_de_macro("auc referencia ablacion", MACRO_MODELO[modelo]),
                            f"{ABLACIONES}: {modelo}, auc_referencia (A0)",
                            _ablacion("A1", modelo, "auc_referencia"), _metrica))
    for variante in ("A7", "A8"):
        for modelo in ("rf", "nb_gaussiano", "svm", "knn"):
            numeros.append(_num(nombre_de_macro("auc", MACRO_MODELO[modelo], MACRO_VARIANTE[variante]),
                                f"{ABLACIONES}: {modelo}, {variante}, auc_variante",
                                _ablacion(variante, modelo, "auc_variante"), _metrica))
    numeros += _ablaciones_de([v for v in MACRO_VARIANTE if v not in ("A1", "A2")])
    numeros += [
        _num("pctDefaultUnknown", f"{TRAIN}: % de filas con default = unknown",
             _eda("pct_default_unknown"), _pct),
        _num("pctYesDefaultUnknown", f"{TRAIN}: % de «yes» con default = unknown",
             _eda("pct_yes_default_unknown"), _pct),
        _num("pctYesDefaultConocido", f"{TRAIN}: % de «yes» con default conocido",
             _eda("pct_yes_default_conocido"), _pct),
        _num("filasDefaultYes", f"{TRAIN}: filas con default = yes (D-09)", _eda("default_yes"),
             formatear_miles),
        _num("mediaCampaignYes", f"{TRAIN}: media de campaign en los «yes» (D-07)",
             _eda("media_campaign_yes"), _metrica),
        _num("mediaCampaignNo", f"{TRAIN}: media de campaign en los «no» (D-07)",
             _eda("media_campaign_no"), _metrica),
        _num("medianaCampaignYes", f"{TRAIN}: mediana de campaign en los «yes» (D-07)",
             _eda("mediana_campaign_yes"), formatear_parametro),
        _num("medianaCampaignNo", f"{TRAIN}: mediana de campaign en los «no» (D-07)",
             _eda("mediana_campaign_no"), formatear_parametro),
        _num("maximoCampaign", f"{TRAIN}: máximo de campaign (D-13)", _eda("maximo_campaign"),
             formatear_miles),
        _num("asimetriaCampaign", f"{TRAIN}: asimetría de campaign, pandas skew (D-13)",
             _eda("asimetria_campaign"), _dos),
        _num("factorIqr", "src/numeros.py: FACTOR_IQR, la regla de Tukey de resultados/eda/"
             "reporte.txt, §5 (D-17)", _constante(FACTOR_IQR), _uno),
        _num("filasAtipicasIqr", f"{TRAIN}: filas con algún valor fuera de 1,5·IQR en las 9 "
             "numéricas del modelo (D-17)", _eda("atipicas"), formatear_miles),
        _num("pctAtipicasIqr", f"{TRAIN}: ídem, en % de las filas (D-17)", _eda("pct_atipicas"), _pct),
    ]
    for columna in ("previous", "campaign", "pdays", "age", "cons.conf.idx"):
        numeros += [
            _num(nombre_de_macro("atipicos", columna), f"{TRAIN}: filas con {columna} fuera de "
                 "1,5·IQR (D-17)", _eda(f"atipicos_{columna}"), formatear_miles),
            _num(nombre_de_macro("pct yes atipicos", columna), f"{TRAIN}: % de «yes» entre los "
                 f"atípicos de {columna} (D-17)", _eda(f"pct_yes_atipicos_{columna}"), _pct),
        ]
    return numeros


def _seccion_metricas():
    return [
        _num("aucSinModelo", f"{CV_FINAL}: sin_modelo, validacion, auc, media",
             _cv("final", "sin_modelo", "validacion", "auc"), _metrica),
        _num("aucAzar", f"{CV_FINAL}: sin_modelo, validacion, auc, media, con un decimal",
             _cv("final", "sin_modelo", "validacion", "auc"), _uno),
        _num("recallSinModelo", f"{CV_FINAL}: sin_modelo, validacion, recall_q, media",
             _cv("final", "sin_modelo", "validacion", "recall_q"), _metrica),
        _num("precisionSinModelo", f"{CV_FINAL}: sin_modelo, validacion, precision_q, media",
             _cv("final", "sin_modelo", "validacion", "precision_q"), _metrica),
        _num("apSinModelo", f"{CV_FINAL}: sin_modelo, validacion, ap, media",
             _cv("final", "sin_modelo", "validacion", "ap"), _metrica),
        _num("fUnoSinModelo", f"{CV_FINAL}: sin_modelo, validacion, f1_q, media",
             _cv("final", "sin_modelo", "validacion", "f1_q"), _metrica),
        _num("presupuesto", "src/metricas.py: PRESUPUESTO (D-20)", _constante(100 * PRESUPUESTO),
             _pct_entero),
        _num("presupuestoBajo", "resultados/sensibilidad_q_final.csv: el menor q",
             _q_extremo("min"), _pct_entero),
        _num("presupuestoAlto", "resultados/sensibilidad_q_final.csv: el mayor q",
             _q_extremo("max"), _pct_entero),
        _num("rotuloLineaBase", "src/estilo.py: ROTULO_LINEA_BASE (texto)",
             _constante(ROTULO_LINEA_BASE), _texto),
    ]


def _seccion_referencia():
    numeros = []
    for modelo, base in BASE_REFERENCIA.items():
        donde = f"{CV_REFERENCIA}: {modelo}"
        numeros += [
            _num(f"auc{base}", f"{donde}, validacion, auc, media",
                 _cv("referencia", modelo, "validacion", "auc"), _metrica),
            _num(f"auc{base}Desvio", f"{donde}, validacion, auc, desvio",
                 _cv("referencia", modelo, "validacion", "auc", "desvio"), _metrica),
            _num(f"recall{base}", f"{donde}, validacion, recall_q, media",
                 _cv("referencia", modelo, "validacion", "recall_q"), _metrica),
            _num(f"aucTrain{base}", f"{donde}, train, auc, media",
                 _cv("referencia", modelo, "train", "auc"), _metrica),
            _num(f"brecha{base}", f"{donde}, brecha, auc, media (train − validación)",
                 _cv("referencia", modelo, "brecha", "auc"), _metrica),
        ]
    def mejor_referencia(f):
        return max(FINALES, key=lambda m: _cv("referencia", m, "validacion", "auc")(f))

    numeros += [
        _num("modeloMejorReferencia", f"{CV_REFERENCIA}: el de mayor AUC de validación entre los "
             "cuatro del enunciado (texto)", lambda f: ROTULO_MODELO[mejor_referencia(f)], _texto),
        _num("margenAucReferencia", f"{CV_REFERENCIA}: el mayor AUC de validación de los cuatro "
             "menos el de sin_modelo",
             lambda f: (_cv("referencia", mejor_referencia(f), "validacion", "auc")(f)
                        - _cv("referencia", "sin_modelo", "validacion", "auc")(f)), _metrica),
        _num("deltaAucNbCategorico", "resultados/cv_referencia.csv: nb_categorico − nb_gaussiano, "
             "AUC de validación fold a fold, media (D-18)",
             _cv_pareado("referencia", "nb_categorico", "nb_gaussiano", "auc", "media"),
             formatear_delta),
        _num("deltaAucNbCategoricoDesvio", "resultados/cv_referencia.csv: ídem, desvío",
             _cv_pareado("referencia", "nb_categorico", "nb_gaussiano", "auc", "desvio"),
             _desvio_delta),
        _num("deltaAucNbCategoricoMinimo", "resultados/cv_referencia.csv: ídem, el menor fold",
             _cv_pareado("referencia", "nb_categorico", "nb_gaussiano", "auc", "min"),
             formatear_delta),
        _num("deltaAucNbCategoricoMaximo", "resultados/cv_referencia.csv: ídem, el mayor fold",
             _cv_pareado("referencia", "nb_categorico", "nb_gaussiano", "auc", "max"),
             formatear_delta),
        _num("deltaRecallNbCategorico", "resultados/cv_referencia.csv: nb_categorico − "
             "nb_gaussiano, recall_q de validación fold a fold, media (D-18)",
             _cv_pareado("referencia", "nb_categorico", "nb_gaussiano", "recall_q", "media"),
             formatear_delta),
        _num("deltaRecallNbCategoricoDesvio", "resultados/cv_referencia.csv: ídem, desvío",
             _cv_pareado("referencia", "nb_categorico", "nb_gaussiano", "recall_q", "desvio"),
             _desvio_delta),
        _num("vecinosKnnReferencia", f"{CV_REFERENCIA}: knn, configuracion, n_neighbors",
             _cv_configuracion("referencia", "knn", "n_neighbors"), formatear_parametro),
        _num("arbolesRfReferencia", f"{CV_REFERENCIA}: rf, configuracion, n_estimators",
             _cv_configuracion("referencia", "rf", "n_estimators"), formatear_parametro),
        _num("cSvmReferencia", f"{CV_REFERENCIA}: svm, configuracion, C",
             _cv_configuracion("referencia", "svm", "C"), formatear_parametro),
        _num("cortesNb", f"{CV_FINAL}: nb_categorico, configuracion, n_cortes (D-18, D-22)",
             _cv_configuracion("final", "nb_categorico", "n_cortes"), formatear_parametro),
        _num("alfaNb", f"{CV_FINAL}: nb_categorico, configuracion, alpha, la corrección de Laplace "
             "(D-18)", _cv_configuracion("final", "nb_categorico", "alpha"), formatear_parametro),
    ]
    return numeros


def _seccion_curva_rf():
    curva = "resultados/curvas/rf_max_depth.csv"
    numeros = [
        _num("profundidadRf", f"{CV_FINAL}: rf, configuracion, max_depth",
             _cv_configuracion("final", "rf", "max_depth"), formatear_parametro),
        _num("arbolesRf", f"{CV_FINAL}: rf, configuracion, n_estimators",
             _cv_configuracion("final", "rf", "n_estimators"), formatear_parametro),
        _num("aucRfProfundidadElegida", "resultados/hiperparametros.json: "
             "curvas.rf_max_depth.auc_validacion_media", _hiper("rf_max_depth", "auc_validacion_media"),
             _metrica),
        _num("errorEstandarRfProfundidadElegida", "resultados/hiperparametros.json: "
             "curvas.rf_max_depth.error_estandar", _hiper("rf_max_depth", "error_estandar"), _error),
        _num("brechaRfProfundidadElegida", "resultados/hiperparametros.json: "
             "curvas.rf_max_depth.brecha", _hiper("rf_max_depth", "brecha"), _metrica),
        _num("profundidadRfMejor", "resultados/hiperparametros.json: curvas.rf_max_depth.mejor",
             lambda f: float(_hiper("rf_max_depth", "mejor")(f)), formatear_parametro),
        _num("aucRfProfundidadMejor", "resultados/hiperparametros.json: "
             "curvas.rf_max_depth.auc_validacion_mejor", _hiper("rf_max_depth", "auc_validacion_mejor"),
             _metrica),
        _num("umbralUnoEsRfProfundidad", "resultados/hiperparametros.json: "
             "curvas.rf_max_depth.umbral_1es", _hiper("rf_max_depth", "umbral_1es"), _error),
        _num("errorEstandarRfProfundidadMejor", "resultados/hiperparametros.json: "
             "curvas.rf_max_depth.error_estandar_mejor",
             _hiper("rf_max_depth", "error_estandar_mejor"), _error),
        _num("profundidadRfMinima", "src/modelos.py: GRILLAS, la menor max_depth de la grilla de RF "
             "(el extremo del subajuste)", _constante(_grilla_minima("rf", "max_depth")),
             formatear_parametro),
        _num("arbolesRfRegla", "resultados/hiperparametros.json: curvas.rf_n_estimators_depth8.valor, "
             "lo que elegía la regla de 1 ES (N0-11)", _hiper("rf_n_estimators_depth8", "valor"),
             formatear_parametro),
        _num("arbolesRfMejor", "resultados/hiperparametros.json: curvas.rf_n_estimators_depth8.mejor "
             "(N0-11)", lambda f: float(_hiper("rf_n_estimators_depth8", "mejor")(f)),
             formatear_parametro),
        _num("umbralUnoEsRfArboles", "resultados/hiperparametros.json: "
             "curvas.rf_n_estimators_depth8.umbral_1es (N0-11)",
             _hiper("rf_n_estimators_depth8", "umbral_1es"), _error),
    ]
    for profundidad in GRILLAS[("rf", "max_depth")]:
        punto = str(profundidad)
        nombre = nombre_de_macro("auc rf profundidad",
                                 "sin limite" if profundidad is None else profundidad)
        numeros.append(_num(nombre, f"{curva}: punto {punto}, validacion, auc, media",
                            _curva("rf_max_depth", punto, "auc_validacion_media"), _metrica))
    for punto, palabra in (("None", "SinLimite"), ("2", "Dos")):
        numeros += [
            _num(f"aucTrainRfProfundidad{palabra}", f"{curva}: punto {punto}, train, auc, media",
                 _curva("rf_max_depth", punto, "auc_train_media"), _metrica),
            _num(f"brechaRfProfundidad{palabra}", f"{curva}: punto {punto}, train − validación",
                 _curva("rf_max_depth", punto, "brecha"), _metrica),
        ]
    for arboles in GRILLAS[("rf", "n_estimators")]:
        numeros.append(_num(nombre_de_macro("auc rf arboles", arboles),
                            f"resultados/curvas/rf_n_estimators_depth8.csv: punto {arboles}, "
                            "validacion, auc, media (N0-11)",
                            _curva("rf_n_estimators_depth8", str(arboles), "auc_validacion_media"),
                            _metrica))
    pesos = "resultados/curvas/rf_pesos_clase_depth8.csv"
    numeros += [
        _num("aucRfSinPesos", f"{pesos}: punto None, validacion, auc, media (N0-12)",
             _curva("rf_pesos_clase_depth8", "None", "auc_validacion_media"), _metrica),
        _num("aucRfPesosBalanceados", f"{pesos}: punto balanced, validacion, auc, media (N0-12)",
             _curva("rf_pesos_clase_depth8", "balanced", "auc_validacion_media"), _metrica),
        _num("deltaAucPesosRf", f"{pesos}: balanced − None (N0-12)",
             _resta(_curva("rf_pesos_clase_depth8", "balanced", "auc_validacion_media"),
                    _curva("rf_pesos_clase_depth8", "None", "auc_validacion_media")),
             formatear_delta),
        _num("errorEstandarRfPesosBalanceados", f"{pesos}: punto balanced, error estándar (N0-12)",
             _curva("rf_pesos_clase_depth8", "balanced", "error_estandar"), _error),
    ]
    return numeros


def _seccion_curva_knn():
    curva = "resultados/curvas/knn_n_neighbors_uniform.csv"
    numeros = [
        _num("vecinosKnn", f"{CV_FINAL}: knn, configuracion, n_neighbors",
             _cv_configuracion("final", "knn", "n_neighbors"), formatear_parametro),
        _num("vecinosKnnMejor", "resultados/hiperparametros.json: "
             "curvas.knn_n_neighbors_uniform.mejor",
             lambda f: float(_hiper("knn_n_neighbors_uniform", "mejor")(f)), formatear_parametro),
        _num("aucKnnMejor", "resultados/hiperparametros.json: "
             "curvas.knn_n_neighbors_uniform.auc_validacion_mejor",
             _hiper("knn_n_neighbors_uniform", "auc_validacion_mejor"), _metrica),
        _num("vecinosKnnMinimoUnoEs", "resultados/hiperparametros.json: el menor de "
             "curvas.knn_n_neighbors_uniform.puntos_dentro_1es",
             _hiper_minimo_dentro("knn_n_neighbors_uniform"), formatear_parametro),
        _num("umbralUnoEsKnn", "resultados/hiperparametros.json: "
             "curvas.knn_n_neighbors_uniform.umbral_1es",
             _hiper("knn_n_neighbors_uniform", "umbral_1es"), _error),
        _num("errorEstandarKnnMejor", "resultados/hiperparametros.json: "
             "curvas.knn_n_neighbors_uniform.error_estandar_mejor",
             _hiper("knn_n_neighbors_uniform", "error_estandar_mejor"), _error),
        _num("vecinosKnnMinimo", "src/modelos.py: GRILLAS, el menor n_neighbors de la grilla de KNN "
             "(el extremo del sobreajuste)", _constante(_grilla_minima("knn", "n_neighbors")),
             formatear_parametro),
        _num("vecinosKnnMeseta", "DECISIONES.md, D-22: el k desde el que la curva de KNN es una "
             "meseta (una lectura de la curva, no una regla); «?» si ya no está dentro de 1 ES",
             _meseta_knn, formatear_parametro),
    ]
    for vecinos in GRILLAS[("knn", "n_neighbors")]:
        numeros.append(_num(nombre_de_macro("auc knn", vecinos),
                            f"{curva}: punto {vecinos}, validacion, auc, media",
                            _curva("knn_n_neighbors_uniform", str(vecinos), "auc_validacion_media"),
                            _metrica))
    # weights = distance: D-22 aplicada a la curva misma, que ya tiene la grilla extendida (ver
    # _eleccion). Los cuatro macros son del mismo punto.
    distancia = ("resultados/curvas/knn_n_neighbors_distance.csv: el punto que elige D-22 sobre "
                 "la curva (src/curvas.py, elegir_punto; weights = distance)")
    numeros += [
        _num("aucTrainKnnUno", f"{curva}: punto 1, train, auc, media",
             _curva("knn_n_neighbors_uniform", "1", "auc_train_media"), _metrica),
        _num("brechaKnnUno", f"{curva}: punto 1, train − validación",
             _curva("knn_n_neighbors_uniform", "1", "brecha"), _metrica),
        _num("vecinosKnnDistancia", f"{distancia}, n_neighbors",
             _eleccion("knn_n_neighbors_distance", "valor"), formatear_parametro),
        _num("aucKnnDistancia", f"{distancia}, validacion, auc, media",
             _eleccion("knn_n_neighbors_distance", "auc_validacion_media"), _metrica),
        _num("aucTrainKnnDistancia", f"{distancia}, train, auc, media",
             _eleccion("knn_n_neighbors_distance", "auc_train_media"), _metrica),
        _num("brechaKnnDistancia", f"{distancia}, train − validación",
             _eleccion("knn_n_neighbors_distance", "brecha"), _metrica),
    ]
    return numeros


def _seccion_curva_svm():
    kernel = "resultados/curvas/svm_kernel.csv"
    pesos = "resultados/curvas/svm_pesos_clase.csv"
    hiper = "resultados/hiperparametros.json: curvas"
    numeros = [
        _num("cSvm", f"{CV_FINAL}: svm, configuracion, C", _cv_configuracion("final", "svm", "C"),
             formatear_parametro),
        _num("kernelSvm", f"{CV_FINAL}: svm, configuracion, kernel (texto)",
             lambda f: ROTULO_KERNEL.get(_cv_configuracion("final", "svm", "kernel")(f),
                                         _cv_configuracion("final", "svm", "kernel")(f)), _texto),
        _num("aucSvmBalanceada", f"{pesos}: punto balanced, validacion, auc, media (RBF, C = 1)",
             _curva("svm_pesos_clase", "balanced", "auc_validacion_media"), _metrica),
        _num("deltaAucPesosSvm", f"{pesos}: balanced − None (D-21, N0-12)",
             _resta(_curva("svm_pesos_clase", "balanced", "auc_validacion_media"),
                    _curva("svm_pesos_clase", "None", "auc_validacion_media")), formatear_delta),
        _num("cSvmSinPesosMejor", f"{hiper}.svm_C.mejor (RBF, sin pesos)",
             lambda f: float(_hiper("svm_C", "mejor")(f)), formatear_parametro),
        _num("aucSvmSinPesosMejor", f"{hiper}.svm_C.auc_validacion_mejor",
             _hiper("svm_C", "auc_validacion_mejor"), _metrica),
        _num("cSvmBalanceadaMejor", f"{hiper}.svm_C_balanced.mejor (RBF, pesos balanceados)",
             lambda f: float(_hiper("svm_C_balanced", "mejor")(f)), formatear_parametro),
        _num("aucSvmBalanceadaMejor", f"{hiper}.svm_C_balanced.auc_validacion_mejor",
             _hiper("svm_C_balanced", "auc_validacion_mejor"), _metrica),
        _num("cSvmKernel", f"{kernel}: C de la configuración en que se comparan los kernels",
             _curva_configuracion("svm_kernel", "linear", "C"), formatear_parametro),
    ]
    for punto, palabra in (("linear", "Lineal"), ("poly", "Polinomico"), ("rbf", "Rbf")):
        numeros.append(_num(f"aucSvmKernel{palabra}", f"{kernel}: punto {punto}, validacion, auc, media",
                            _curva("svm_kernel", punto, "auc_validacion_media"), _metrica))
    numeros += [
        _num("cSvmLinealMejor", f"{hiper}.svm_C_linear_balanced.mejor",
             lambda f: float(_hiper("svm_C_linear_balanced", "mejor")(f)), formatear_parametro),
        _num("aucSvmLinealMejor", f"{hiper}.svm_C_linear_balanced.auc_validacion_mejor",
             _hiper("svm_C_linear_balanced", "auc_validacion_mejor"), _metrica),
        _num("segundosLimiteCosto", "resultados/costos.csv: el límite de la corrida excedida (s)",
             _costo_excedido("segundos"), formatear_miles),
        _num("cSvmCostoExcedido", "resultados/costos.csv: C de la corrida excedida (SVC lineal)",
             _costo_excedido("C"), formatear_parametro),
    ]
    return numeros


def _seccion_final():
    numeros = []
    for modelo in FINALES:
        m = MACRO_MODELO[modelo]
        donde = f"{CV_FINAL}: {modelo}"
        numeros += [
            _num(f"auc{m}Val", f"{donde}, validacion, auc, media",
                 _cv("final", modelo, "validacion", "auc"), _metrica),
            _num(f"auc{m}ValDesvio", f"{donde}, validacion, auc, desvio",
                 _cv("final", modelo, "validacion", "auc", "desvio"), _metrica),
            _num(f"recall{m}Val", f"{donde}, validacion, recall_q, media",
                 _cv("final", modelo, "validacion", "recall_q"), _metrica),
            _num(f"recall{m}ValDesvio", f"{donde}, validacion, recall_q, desvio",
                 _cv("final", modelo, "validacion", "recall_q", "desvio"), _metrica),
            _num(f"precision{m}Val", f"{donde}, validacion, precision_q, media",
                 _cv("final", modelo, "validacion", "precision_q"), _metrica),
            _num(f"ap{m}Val", f"{donde}, validacion, ap, media",
                 _cv("final", modelo, "validacion", "ap"), _metrica),
            _num(f"fUno{m}Val", f"{donde}, validacion, f1_q, media",
                 _cv("final", modelo, "validacion", "f1_q"), _metrica),
            _num(f"aucTrain{m}", f"{donde}, train, auc, media",
                 _cv("final", modelo, "train", "auc"), _metrica),
            _num(f"brecha{m}", f"{donde}, brecha, auc, media (train − validación)",
                 _cv("final", modelo, "brecha", "auc"), _metrica),
        ]
    for modelo in ("rf", "knn", "svm"):
        numeros.append(_num(nombre_de_macro("delta auc ajuste", MACRO_MODELO[modelo]),
                            f"{CV_FINAL} menos {CV_REFERENCIA}: {modelo}, validacion, auc, media",
                            _resta(_cv("final", modelo, "validacion", "auc"),
                                   _cv("referencia", modelo, "validacion", "auc")), formatear_delta))
    elegido = "resultados/modelo_elegido.json"
    numeros += [
        _num("modeloFinal", f"{elegido}: ranking, 1.º (texto)",
             lambda f: ROTULO_MODELO.get(_ranking(0, "modelo")(f), _ranking(0, "modelo")(f)), _texto),
        _num("modeloSegundoFinal", f"{elegido}: ranking, 2.º (texto)",
             lambda f: ROTULO_MODELO.get(_ranking(1, "modelo")(f), _ranking(1, "modelo")(f)), _texto),
        _num("modeloUltimoFinal", f"{elegido}: ranking, último (texto)",
             lambda f: ROTULO_MODELO.get(_ranking(-1, "modelo")(f), _ranking(-1, "modelo")(f)),
             _texto),
        _num("errorEstandarAucRf", f"{CV_FINAL}: rf, validacion, auc, error_estandar",
             _cv("final", "rf", "validacion", "auc", "error_estandar"), _error),
        _num("umbralEmpateFinal", f"{elegido}: ranking, AUC del 1.º menos su error estándar (D-23)",
             _resta(_ranking(0, "auc"), _ranking(0, "auc_error_estandar")), _error),
        _num("modelosDentroUnoEs", f"{elegido}: cuántos quedan dentro de 1 ES del mejor "
             "(empate_dentro_1es)", lambda f: len(_elegido("empate_dentro_1es")(f)), formatear_miles),
        _num("margenAucFinal", f"{elegido}: ranking, AUC del 1.º menos el del 2.º",
             _resta(_ranking(0, "auc"), _ranking(1, "auc")), _metrica),
        _num("margenAucSinModeloFinal", f"{elegido}: ranking, AUC del 1.º menos linea_base.auc",
             _resta(_ranking(0, "auc"), _elegido("linea_base", "auc")), _metrica),
        _num("rangoAucFinal", f"{elegido}: ranking, AUC del 1.º menos el del último",
             _resta(_ranking(0, "auc"), _ranking(-1, "auc")), _metrica),
    ]
    return numeros


def _seccion_test():
    elegido = "resultados/modelo_elegido.json"
    evaluacion = "resultados/evaluacion_test.json"
    numeros = [
        _num("yesTest", f"{evaluacion}: n_yes_test", _evaluacion("n_yes_test"), formatear_miles),
        _num("pctYesTest", f"{evaluacion}: n_yes_test / n_test",
             _cociente(lambda f: 100 * _evaluacion("n_yes_test")(f), _evaluacion("n_test")), _pct),
        _num("diferenciaAucTestValidacion", f"{evaluacion}: metricas.auc.valor menos {elegido}: "
             "metricas.validacion.auc.media (sólo en voz alta, guía C2)",
             _resta(_evaluacion("metricas", "auc", "valor"),
                    _elegido("metricas", "validacion", "auc", "media")), formatear_delta),
        _num("diferenciaRecallTestValidacion", f"{evaluacion}: metricas.recall_q.valor menos "
             f"{elegido}: metricas.validacion.recall_q.media (sólo en voz alta, guía C2)",
             _resta(_evaluacion("metricas", "recall_q", "valor"),
                    _elegido("metricas", "validacion", "recall_q", "media")), formatear_delta),
    ]
    for celda in ("vp", "fp", "fn", "vn"):
        numeros.append(_num(f"{celda}RfVal", f"resultados/oof_final_rf.csv con la y de {TRAIN}: "
                            f"{celda.upper()} llamando al 20 % de la lista fuera de fold",
                            _matriz_oof("rf", celda), _conteo))
    numeros += [
        _num("llamadasPorYesRf", f"resultados/oof_final_rf.csv con la y de {TRAIN}: llamadas por "
             "cada VP, llamando al 20 % de la lista fuera de fold, (VP + FP) / VP",
             _cociente(lambda f: _matriz_oof("rf", "vp")(f) + _matriz_oof("rf", "fp")(f),
                       _matriz_oof("rf", "vp")), _uno),
        _num("llamadasPorYesSinModelo", f"{CV_FINAL}: 1 / (sin_modelo, validacion, precision_q, "
             "media), las llamadas por cada «yes» sin modelo",
             lambda f: 1 / _cv("final", "sin_modelo", "validacion", "precision_q")(f), _uno),
        _num("nivelIntervalo", "src/evaluar_test.py: PERCENTILES, el nivel del intervalo por "
             "bootstrap del AUC y del recall de test (D-24)",
             _constante(PERCENTILES[1] - PERCENTILES[0]), _pct_entero),
        _num("remuestreosBootstrap", "src/evaluar_test.py: N_BOOTSTRAP, los remuestreos de ese "
             "intervalo (D-24)", _constante(N_BOOTSTRAP), formatear_miles),
    ]
    return numeros


def _seccion_por_que():
    return [
        _num("correlacionMacroMinima", f"{TRAIN}: el menor |r| entre los pares del bloque macro "
             "con |r| >= 0,9 (D-14)", _eda("correlacion_macro_min"), _dos),
        _num("correlacionMacroMaxima", f"{TRAIN}: el mayor |r| entre esos pares (D-14)",
             _eda("correlacion_macro_max"), _dos),
        _num("desvioPdays", f"{TRAIN}: desvío estándar de pdays (D-12)", _eda("desvio_pdays"), _uno),
        _num("desvioNrEmployed", f"{TRAIN}: desvío estándar de nr.employed (D-12)",
             _eda("desvio_nr_employed"), _uno),
        _num("pctVarianzaPdays", f"{TRAIN}: % de la varianza de las 9 numéricas del modelo que es "
             "de pdays (D-12)", _eda("pct_varianza_pdays"), _pct),
        _num("pctVarianzaNrEmployed", f"{TRAIN}: ídem, de nr.employed (D-12)",
             _eda("pct_varianza_nr_employed"), _pct),
        _num("razonDesviosNumericas", f"{TRAIN}: el mayor desvío de las 9 numéricas del modelo / "
             "el menor", _eda("razon_desvios"), lambda x: formatear_decimal(x, 0)),
    ]


def _seccion_limitaciones():
    numeros = []
    sensibilidad = "resultados/sensibilidad_q_final.csv"
    for modelo in ("rf", "knn", "nb_categorico", "svm", "sin_modelo"):
        for q in (0.1, 0.3):
            numeros.append(_num(nombre_de_macro("recall", MACRO_MODELO[modelo], round(100 * q)),
                                f"{sensibilidad}: {modelo}, q = {q}, recall_q, media de los folds",
                                _sensibilidad(modelo, q, "recall_q"), _metrica))
    for q in (0.1, 0.2, 0.3):
        numeros.append(_num(nombre_de_macro("precision rf", round(100 * q)),
                            f"{sensibilidad}: rf, q = {q}, precision_q, media de los folds",
                            _sensibilidad("rf", q, "precision_q"), _metrica))
    return numeros


def _seccion_hallazgo():
    numeros = []
    for modelo in FINALES:
        m = MACRO_MODELO[modelo]
        for variante, v in MACRO_ROBUSTEZ.items():
            for esquema, e in ((ESQUEMA_ADELANTE, "Adelante"), (ESQUEMA_BARAJADO, "Barajado")):
                if esquema == ESQUEMA_BARAJADO and variante == "todas":
                    continue  # es la validación cruzada final: \auc<M>Val
                donde = f"{ROBUSTEZ}: {modelo}, {esquema}, {variante}, auc"
                numeros += [
                    _num(f"auc{m}{v}{e}", f"{donde}, media", _robustez(modelo, esquema, variante),
                         _metrica),
                    _num(f"auc{m}{v}{e}Desvio", f"{donde}, desvio",
                         _robustez(modelo, esquema, variante, columna="desvio"), _metrica),
                ]
        numeros += [
            _num(f"recall{m}Adelante", f"{ROBUSTEZ}: {modelo}, hacia_adelante, todas, recall_q, media",
                 _robustez(modelo, ESQUEMA_ADELANTE, "todas", "recall_q"), _metrica),
            _num(f"caidaAuc{m}Adelante", f"{ROBUSTEZ}: {modelo}, barajado_menos_hacia_adelante, "
                 "todas, auc, media", _robustez(modelo, "barajado_menos_hacia_adelante", "todas"),
                 _metrica),
        ]
        for variante in ("sin_macro", "sin_macro_ni_month"):
            numeros.append(_num(f"deltaAuc{MACRO_ROBUSTEZ[variante]}{m}Barajado",
                                f"{ROBUSTEZ}: {modelo}, barajado, {variante} menos todas, auc, media",
                                _resta(_robustez(modelo, ESQUEMA_BARAJADO, variante),
                                       _robustez(modelo, ESQUEMA_BARAJADO, "todas")),
                                formatear_delta))
    for fold in range(1, K + 1):
        numeros += [
            _num(nombre_de_macro("auc rf adelante fold", fold), "resultados/robustez_temporal.csv: "
                 f"rf, hacia_adelante, todas, fold {fold}, validacion, auc",
                 _robustez_fold("rf", ESQUEMA_ADELANTE, "todas", fold), _metrica),
            _num(nombre_de_macro("pct yes bloque", fold), "resultados/robustez_folds.csv: "
                 f"hacia_adelante, fold {fold}, pct_yes_validacion",
                 _bloque(ESQUEMA_ADELANTE, fold, "pct_yes_validacion"), _pct),
            _num(nombre_de_macro("pct yes entrenamiento adelante", fold),
                 f"resultados/robustez_folds.csv: hacia_adelante, fold {fold}, pct_yes_train",
                 _bloque(ESQUEMA_ADELANTE, fold, "pct_yes_train"), _pct),
            _num(nombre_de_macro("filas entrenamiento adelante", fold),
                 f"resultados/robustez_folds.csv: hacia_adelante, fold {fold}, n_train",
                 _bloque(ESQUEMA_ADELANTE, fold, "n_train"), formatear_miles),
        ]
    numeros.append(_num("filasBloqueAdelante", "resultados/robustez_folds.csv: hacia_adelante, "
                        "fold 1, n_validacion", _bloque(ESQUEMA_ADELANTE, 1, "n_validacion"),
                        formatear_miles))
    # Las tres lecturas del hallazgo (resultados/conclusiones.md, §2): (a) cuánto del AUC barajado
    # es ordenar años; (b) y (c), el AUC barajado en las filas de cada bloque hacia adelante, cuánto
    # pierde hacia adelante en esas mismas filas y con cuántos «yes» entrena cada bloque.
    oof_anio = f"resultados/oof_final_rf.csv con la y y el año inferido de {TRAIN}"
    numeros += [
        _num("pctParesEntreAnios", f"{oof_anio}: % de los pares «yes»–«no» que son de años "
             "distintos", _pares_por_anio("pct_entre"), _pct),
        _num("aucOofEntreAnios", f"{oof_anio}: AUC fuera de fold de RF sobre los pares de años "
             "distintos", _pares_por_anio("auc_entre"), _metrica),
        _num("aucOofDentroAnio", f"{oof_anio}: AUC fuera de fold de RF sobre los pares del mismo año",
             _pares_por_anio("auc_dentro"), _metrica),
    ]
    for anio in (2008, 2009, 2010):
        numeros.append(_num(nombre_de_macro("auc oof", anio), f"{oof_anio}: AUC fuera de fold de RF "
                            f"en las filas de {anio}", _pares_por_anio(f"auc_{anio}"), _metrica))
    for fold in range(1, K + 1):
        donde = (f"resultados/oof_final_rf.csv con la y de {TRAIN}, en las filas del bloque {fold} "
                 "hacia adelante (resultados/robustez_folds.csv)")
        numeros += [
            _num(nombre_de_macro("auc rf barajado bloque", fold), f"{donde}: AUC fuera de fold de RF "
                 "(barajado)", _barajado_en_bloque(fold), _metrica),
            _num(nombre_de_macro("caida auc rf bloque", fold), f"{donde}: AUC barajado menos el "
                 f"hacia adelante (resultados/robustez_temporal.csv: rf, todas, fold {fold})",
                 _resta(_barajado_en_bloque(fold), _robustez_fold("rf", ESQUEMA_ADELANTE, "todas",
                                                                   fold)), _metrica),
            _num(nombre_de_macro("yes entrenamiento adelante", fold), f"{TRAIN}: «yes» con fila <= "
                 f"fila_max_train del fold {fold} hacia adelante (resultados/robustez_folds.csv)",
                 _yes_entrenamiento_adelante(fold), formatear_miles),
        ]
    return numeros


def _seccion_ganancia():
    numeros = []
    for p in (10, 20, 30, 50):
        numeros.append(_num(nombre_de_macro("ganancia rf", p), f"resultados/ganancia_final.csv: rf, "
                            f"p = {p}, pct_yes_alcanzado", _ganancia("rf", p), _pct))
    for modelo in ("knn", "nb_categorico", "svm", "sin_modelo"):
        numeros.append(_num(nombre_de_macro("ganancia", MACRO_MODELO[modelo], 20),
                            f"resultados/ganancia_final.csv: {modelo}, p = 20, pct_yes_alcanzado",
                            _ganancia(modelo, 20), _pct))
    for pct, palabra in ((50, "Mitad"), (80, "Ochenta")):
        numeros.append(_num(f"pctListaYes{palabra}Rf", "resultados/ganancia_final.csv: rf, el menor "
                            f"p con pct_yes_alcanzado >= {pct}", _ganancia_alcanza("rf", pct),
                            _pct_entero))
    return numeros


def catalogo():
    """Las secciones del deck, en el orden del esqueleto del plan (§5, ola 7), con sus macros.
    Cada macro aparece una sola vez, en la primera slide que lo usa."""
    secciones = [
        ("Slide 2 · El problema", _seccion_problema()),
        ("Slide 3 · Qué datos deciden", _seccion_datos()),
        ("Slide 4 · EDA: la duración es fuga", _seccion_duration()),
        ("Slide 5 · EDA: el archivo está ordenado por fecha", _seccion_tiempo()),
        ("Slide 6 · EDA: 999 no es «nunca contactado»", _seccion_pdays()),
        ("Slide 7 · EDA: el pipeline y la consecuencia de cada decisión", _seccion_pipeline()),
        ("Slide 8 · Métricas y presupuesto de llamadas", _seccion_metricas()),
        ("Slide 9 · Los modelos sin ajustar", _seccion_referencia()),
        ("Slide 10 · Curva de RF", _seccion_curva_rf()),
        ("Slide 11 · Curva de KNN", _seccion_curva_knn()),
        ("Slide 12 · Curva de la SVM", _seccion_curva_svm()),
        ("Slide 13 · El modelo final en validación", _seccion_final()),
        ("Slides 14 y 15 · Test y matriz de confusión", _seccion_test()),
        ("Slide 16 · Por qué cada modelo rindió lo que rindió", _seccion_por_que()),
        ("Slide 17 · Limitaciones: sensibilidad al presupuesto", _seccion_limitaciones()),
        ("Slides 18 y 19 · Hallazgo: el modelo aprende la época", _seccion_hallazgo()),
        ("Reserva · Curva de ganancia (H3)", _seccion_ganancia()),
    ]
    vistos = set()
    for _, numeros in secciones:
        for numero in numeros:
            if not re.fullmatch(r"[A-Za-z]+", numero.nombre):
                raise ValueError(f"nombre de macro inválido: {numero.nombre!r}")
            if numero.nombre in vistos:
                raise ValueError(f"macro repetido: {numero.nombre}")
            vistos.add(numero.nombre)
    return secciones


# --- Generación ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Generado:
    tex: str
    md: str
    valores: dict      # nombre -> valor en LaTeX, en el orden del catálogo
    faltan: dict       # qué falta -> macros que valen «?»


def resolver(fuentes, secciones=None):
    """[(título, [(Numero, valor en LaTeX)])] y {qué falta: [macros]}. Un Faltante deja «?»."""
    resueltas, faltan = [], {}
    for titulo, numeros in secciones or catalogo():
        filas = []
        for numero in numeros:
            try:
                valor = numero.calcular(fuentes)
            except Faltante as falta:
                valor = DESCONOCIDO
                faltan.setdefault(str(falta), []).append(numero.nombre)
            filas.append((numero, valor))
        resueltas.append((titulo, filas))
    return resueltas, faltan


ANCHO_COMENTARIO = 60


def renderizar_tex(resueltas, faltan):
    lineas = [
        "% GENERADO POR src/numeros.py; no editar a mano. Se regenera con: python3 -m src.numeros",
        "% Cada macro es una cifra de resultados/ (o de train, para el EDA) en el formato del deck:",
        "% coma decimal, miles y % con espacio fino, menos tipográfico. Su fuente está en el",
        "% comentario de su línea y, legible, en informe/numeros.md. Los números de test están en",
        "% informe/resultados-test.tex (src/evaluar_test.py).",
        "% Se leen igual en texto y dentro de $…$: la coma decimal va como {,} (en modo matemático",
        "% una coma suelta deja un espacio detrás) y el menos es \\signoMenos, definido aquí: en",
        "% texto es \\textminus, que dentro de $…$ LaTeX compone mal, y en matemática es «-».",
        "% Con babel-spanish, \\% ya agrega su propio espacio fino y \\,\\% queda doble: el preámbulo",
        "% del deck lleva \\spanishplainpercent después de \\usepackage[spanish]{babel}.",
    ]
    if faltan:
        lineas.append("% Valen «?» porque falta su entrada: "
                      + "; ".join(f"{que} ({len(nombres)})" for que, nombres in faltan.items()) + ".")
    lineas += ["", DEFINICION_MENOS]
    for titulo, filas in resueltas:
        lineas += ["", f"% --- {titulo} " + "-" * max(3, 94 - len(titulo))]
        for numero, valor in filas:
            definicion = f"\\newcommand{{\\{numero.nombre}}}{{{valor}}}"
            relleno = " " * max(1, ANCHO_COMENTARIO - len(definicion))
            lineas.append(f"{definicion}{relleno}% {numero.fuente}")
    return "\n".join(lineas) + "\n"


def _celda(texto):
    return str(texto).replace("|", "\\|")


def _fuente_md(fuente):
    """Las rutas y las funciones de la fuente, en código."""
    fuente = re.sub(r"((?:resultados|data|src|informe)/[\w./-]*\w)", r"`\1`", fuente)
    return re.sub(r"(cargar_train\(\))", r"`\1`", fuente)


def renderizar_md(resueltas, faltan, macros_test, falta_test, resultados):
    lineas = [
        "# Números del TP2",
        "",
        "Generado por `src/numeros.py` (`python3 -m src.numeros`); no editar a mano. Cada fila es un "
        "macro de `informe/numeros.tex`: en el deck se escribe el macro, nunca la cifra, y en el "
        "guion se escribe el valor de esta tabla. Todo número del guion tiene que figurar en la "
        "columna **Valor** (gate G3 del plan); los números de las otras columnas y de los títulos "
        "no cuentan. La columna **Fuente** dice de qué archivo y columna sale cada cifra.",
        "",
        "Formato: coma decimal; tres decimales en las métricas (AUC, recall, precisión); cuatro en "
        "los errores estándar y en los umbrales de un error estándar; los Δ con signo, con tres "
        "decimales desde una centésima y cuatro por debajo; los porcentajes con un decimal. En "
        "LaTeX, la coma decimal va como `{,}`, los miles y el signo % van con espacio fino y el "
        "menos es `\\signoMenos`, para que el macro se lea igual en texto y dentro de `$…$`; aquí, "
        "con coma, espacio común y «−», como en el guion.",
        "",
        f"Resultados leídos de `{_mostrar(resultados)}/`.",
    ]
    if faltan:
        lineas += ["", "## Pendientes", "",
                   "Estos macros valen «?» porque falta su entrada. El deck compila igual, y se "
                   "completan solos al volver a correr `python3 -m src.numeros`.", ""]
        for que, nombres in faltan.items():
            lineas.append(f"- {_fuente_md(que)}: " + ", ".join(f"`\\{n}`" for n in nombres))
    for titulo, filas in resueltas:
        lineas += ["", f"## {titulo}", "", "| Macro | Valor | Fuente |", "|---|---|---|"]
        for numero, valor in filas:
            fuente = _fuente_md(numero.fuente)
            if valor == DESCONOCIDO:
                fuente += " (pendiente)"
            lineas.append(f"| `\\{numero.nombre}` | {_celda(a_texto(valor))} | {_celda(fuente)} |")
    lineas += ["", "## Números de test (`informe/resultados-test.tex`)", "",
               "Los escribe `src/evaluar_test.py`, no este módulo: el deck los toma de ese archivo. "
               "Se repiten aquí para que el guion tenga todos los números en un solo lugar. "
               "Mientras el test no se evalúe, valen «?».", ""]
    if falta_test:
        lineas += [f"Falta `{falta_test}`: los macros van con «?».", ""]
    lineas += ["| Macro | Valor | Fuente |", "|---|---|---|"]
    for nombre, valor in macros_test.items():
        lineas.append(f"| `\\{nombre}` | {_celda(a_texto(valor))} | `informe/resultados-test.tex` |")
    lineas += [
        "", "## Cifras que este módulo no mide", "",
        "- **D-02** (la dispersión de la proporción de «yes» del test entre semillas) y **D-03** "
        "(la tasa de «yes» de train y de test con una partición por fecha) se miden sobre el CSV "
        "completo, test incluido, con `src/evidencia_particion.py`. Este módulo no puede leerlo "
        "(D-04, `tests/test_aislamiento.py`): sus macros (slide 3) leen "
        f"`resultados/{EVIDENCIA_PARTICION}`, con las claves `n_semillas`, "
        "`sin_estratificar.desvio_pp`, `sin_estratificar.minimo_pct`, `sin_estratificar.maximo_pct`, "
        "`temporal.pct_yes_train` y `temporal.pct_yes_test` (porcentajes en puntos), y valen «?» "
        "hasta que ese módulo escriba el archivo.",
        f"- `\\vecinosKnnMeseta` ({VECINOS_MESETA}) es una lectura de la curva que hace D-22, no el "
        "resultado de una regla: se copia de DECISIONES.md y vale «?» si ese punto deja de estar "
        "dentro de 1 error estándar del mejor.",
    ]
    return "\n".join(lineas) + "\n"


def generar(fuentes):
    """Los textos de numeros.tex y numeros.md. No escribe nada."""
    resueltas, faltan = resolver(fuentes)
    try:
        macros_test, falta_test = fuentes.macros_test(), None
    except Faltante as falta:
        macros_test, falta_test = dict.fromkeys(MACROS_TEST, DESCONOCIDO), str(falta)
    valores = {numero.nombre: valor for _, filas in resueltas for numero, valor in filas}
    return Generado(tex=renderizar_tex(resueltas, faltan),
                    md=renderizar_md(resueltas, faltan, macros_test, falta_test,
                                     fuentes.resultados),
                    valores=valores, faltan=faltan)


def escribir(ruta, texto):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")


# --- Verificación (gate G3) ---------------------------------------------------------------------

# Números que no son datos. Cada uno es (qué es, expresión regular); --permitir agrega más. Si la
# expresión tiene un grupo «n», sólo ese grupo queda permitido: el resto es el contexto que lo
# habilita, y los números del contexto se siguen revisando.
_SEP = r"(?:\s|~|\\,|\\ )*"
# Un número de referencia (7, 2.3, 8.2.1): nunca uno con coma decimal, ni el comienzo de otro.
_NUM_REF = r"\d+(?:\.\d+)*(?![.,]?\d)"
_REF_SINGULAR = (r"clase|slide|diapositiva|punto|paso|ola|consulta|atributo|secci[oó]n|cap[ií]tulo"
                 r"|cap\.|p\.|p[aá]g\.")
_REF_PLURAL = (r"clases|slides|diapositivas|puntos|pasos|olas|consultas|atributos|secciones"
               r"|cap[ií]tulos|pp\.|p[aá]gs\.")
_RANGO = r"(?:–|—|--|-)"
_ENLACE = r"(?:,|\by\b|\be\b|\ba\b|\bal\b|\bo\b)"
_POR_CIENTO_TEX = _SEP + r"\\%"
PERMITIDOS = (
    ("años 2008–2010", r"\b20(?:0[89]|10)\b"),
    ("fechas dd/mm y dd/mm/aaaa",
     r"\b(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])(?:/(?:\d{4}|\d{2}))?\b"),
    # Después de una forma singular, sólo un número o un rango pegado (slide 50, slides 45–51):
    # «la Clase 7, 0,79» o «el punto 4, 801» no esconden el segundo número.
    ("referencia a una clase, slide, página, punto o paso",
     r"(?i)\b(?:" + _REF_SINGULAR + r")" + _SEP + _NUM_REF + r"(?:" + _RANGO + _NUM_REF + r")?"),
    ("referencia a varias: rangos y enumeraciones (slides 26 y 27, atributos 1 a 7)",
     r"(?i)\b(?:" + _REF_PLURAL + r")" + _SEP + _NUM_REF
     + r"(?:(?:" + _RANGO + r"|" + _SEP + _ENLACE + _SEP + r")" + _NUM_REF + r")*"),
    ("secciones", r"(?:§|\\S\b)" + _SEP + r"\d+(?:\.\d+)*(?!,\d)"),
    ("años de las citas bibliográficas",
     r"(?:\b(?:Mitchell|Alpaydin|Moro)|\bet(?:\s|~)+al\.)(?:\s|~|,|\()*\d{4}\b"),
    ("identificadores de decisiones y avisos", r"\b(?:[DSGC]|N0)-\d+\b"),
    ("código de la materia y número de TP", r"\b72\.75\b|(?i:\btrabajo\s+pr[aá]ctico\s+\d\b)"),
    ("enumeración al principio de un título", r"(?<=[{\[])\d{1,2}\.(?=\s)"),
    ("identificador de un fold o de un bloque",
     r"(?i)\b(?:fold|bloque|pliegue)(?:\s|~)*\d\b(?![.,]\d)"),
    ("10 minutos", r"(?i)\b10" + _SEP + r"min(?:utos)?\b\.?"),
    ("5 folds", r"(?i)\b5" + _SEP + r"-?" + _SEP + r"(?:folds?|particiones|pliegues)\b"),
    ("5 folds en código", r"(?:KFold|TimeSeriesSplit)\s*\(\s*(?:n\\?_splits\s*=\s*)?5\b"),
    ("k = 5", r"\b[kK]" + _SEP + r"=" + _SEP + r"5\b"),
    ("80/20", r"\b80" + _SEP + r"/" + _SEP + r"20\b"),
    # El 20 % es la definición del presupuesto (D-20) sólo cuando se habla de llamar o del
    # presupuesto; «sin modelo se alcanza el 20 % de los "yes"» es un resultado y tiene macro.
    ("el 20 % del presupuesto, en una frase sobre llamar o sobre el presupuesto",
     r"(?i)(?:\b(?:llam\w*|presupuesto|recall|precisi[oó]n|contact\w*)\b|\bq(?:\s|~)*=)"
     r"[^.;:\d]{0,40}?(?P<n>\b20" + _POR_CIENTO_TEX + r")"),
    ("el 20 % de la lista o de las llamadas",
     r"(?i)(?P<n>\b20" + _POR_CIENTO_TEX + r")(?:\s|~)+(?:de\s+(?:la\s+lista|las\s+llamadas"
     r"|cada\s+lista)|del\s+presupuesto)\b"),
    ("1 error estándar", r"\b1" + _SEP + r"(?:ES\b|error(?:\s|~)+est[aá]ndar|desv[ií]o)"),
    ("±1", r"(?:±|\\pm)(?:\s|~|\\,|\$)*1\b"),
    ("la esquina (0,1) de la ROC", r"\(\s*0\s*,\s*1\s*\)"),
    ("medidas relativas", r"\d+(?:[.,]\d+)?(?:\s|~)*\\(?:textwidth|linewidth|columnwidth|paperwidth"
                          r"|textheight|paperheight|baselineskip|parskip|parindent|hsize|vsize)\b"),
    ("ordinales", r"\b\d+\.?(?:º|ª|°|\\textordfeminine|\\textordmasculine)|\b\d+\.er\b"),
    ("nombres de archivo", r"[\w./-]*\d[\w./-]*\.(?:png|pdf|jpe?g|svg|eps|tex|csv|json|md|py|txt)\b"),
    # Una ruta con al menos una carpeta con letras (figuras/presentacion/03-curva-rf, sin
    # extensión): «1/3» no es una ruta.
    ("rutas", r"(?:[\w.-]*[A-Za-z][\w.-]*/)+[\w.-]+"),
)

# Entornos que se saltean enteros: el texto literal es código.
ENTORNOS_OCULTOS = ("verbatim", "Verbatim", "lstlisting", "minted", "comment")
# Dibujos: su sintaxis (opciones, coordenadas, estilos, bucles) se saltea, pero el texto de cada
# nodo se lee, porque se proyecta: «1070 filas» en un nodo es un número escrito a mano.
ENTORNOS_TIKZ = ("tikzpicture", "pgfpicture")
# Definiciones: el comando y todos sus argumentos (cuántos obligatorios lleva).
DEFINICIONES = {
    "newcommand": 2, "renewcommand": 2, "providecommand": 2, "DeclareRobustCommand": 2,
    "newenvironment": 3, "renewenvironment": 3, "NewDocumentCommand": 3,
    "RenewDocumentCommand": 3, "ProvideDocumentCommand": 3, "DeclareDocumentCommand": 3,
    "setlength": 2, "addtolength": 2, "newlength": 1, "setcounter": 2, "addtocounter": 2,
    "newcounter": 1, "newif": 1, "let": 2, "pgfmathsetmacro": 2, "pgfmathparse": 1,
    "setbeamerfont": 2, "setbeamercolor": 2, "setbeamertemplate": 2, "setbeamersize": 1,
    "setbeamercovered": 1, "setbeameroption": 1, "definecolor": 3, "colorlet": 2,
    "usepackage": 1, "RequirePackage": 1, "documentclass": 1, "usetheme": 1, "usecolortheme": 1,
    "usefonttheme": 1, "useinnertheme": 1, "useoutertheme": 1, "graphicspath": 1,
    "input": 1, "include": 1, "includeonly": 1, "tikzset": 1, "pgfkeys": 1, "pgfplotsset": 1,
    "usetikzlibrary": 1, "usepgfplotslibrary": 1, "hypersetup": 1, "captionsetup": 1,
    "sisetup": 1,
}
# Las que definen un macro: si su cuerpo es sólo una cifra, esa cifra está escrita a mano.
DEFINICIONES_DE_MACRO = frozenset({
    "newcommand", "renewcommand", "providecommand", "DeclareRobustCommand", "NewDocumentCommand",
    "RenewDocumentCommand", "ProvideDocumentCommand", "DeclareDocumentCommand"})
DEF_TEX = frozenset({"def", "gdef", "edef", "xdef"})
# Comandos cuyos primeros argumentos son sintaxis; el resto, si lo hay, es contenido y se lee.
# \figgrande es el envoltorio de \includegraphics del preámbulo del TP1 (plan, paso 7.1).
ARGUMENTOS_DE_SINTAXIS = {
    "includegraphics": 1, "figgrande": 1, "includepdf": 1, "includesvg": 1, "pgfuseimage": 1,
    "pgfdeclareimage": 2, "label": 1, "ref": 1, "eqref": 1, "pageref": 1, "autoref": 1,
    "cite": 1, "citep": 1, "citet": 1, "hyperlink": 1, "hypertarget": 1, "url": 1, "href": 1,
    "againframe": 1, "color": 1, "textcolor": 1, "colorbox": 1, "fcolorbox": 2, "pagecolor": 1,
    "rowcolor": 1, "rowcolors": 3, "cellcolor": 1, "columncolor": 1, "arrayrulecolor": 1,
    "usebeamercolor": 1, "usebeamerfont": 1, "fontsize": 2, "linespread": 1, "setstretch": 1,
    "multicolumn": 2, "multirow": 2, "cline": 1, "cmidrule": 1, "scalebox": 1, "resizebox": 2,
    "rotatebox": 1, "raisebox": 1, "parbox": 1, "rule": 2, "hspace": 1, "vspace": 1,
    "addvspace": 1, "enlargethispage": 1, "phantom": 1, "hphantom": 1, "vphantom": 1,
    "settowidth": 2, "settoheight": 2, "settodepth": 2, "transduration": 1, "ding": 1,
}
CON_PARENTESIS = frozenset({"cmidrule", "framezoom"})
# Entornos con argumentos de sintaxis después del nombre (anchos, columnas de tabla).
ENTORNOS_CON_ARGUMENTOS = {"tabular": 1, "tabular*": 2, "tabularx": 2, "array": 1, "column": 1,
                           "minipage": 1, "overlayarea": 2, "thebibliography": 1, "multicols": 1,
                           "wrapfigure": 2, "adjustbox": 1}
# Comandos que admiten espacio antes de sus corchetes o de su overlay.
CON_ESPACIO = frozenset({"item", "\\", "note", "pause", "begin", "includegraphics"})
# Comandos cuyo opcional es texto que se proyecta: el título corto de una sección va al índice de
# la cabecera, y el rótulo de \item se ve en la lista. Se leen, salvo un rótulo como «1.» o «a)».
OPCIONAL_DE_TEXTO = frozenset({
    "item", "section", "subsection", "subsubsection", "part", "chapter", "paragraph",
    "subparagraph", "frametitle", "framesubtitle", "title", "subtitle", "author", "date",
    "institute", "caption"})
_ROTULO = re.compile(r"\s*(?:\(?(?:\d{1,2}|[A-Za-z]|[ivxlcdm]{1,4}|[IVXLCDM]{1,4})[.)]"
                     r"|\((?:\d{1,2}|[A-Za-z]|[ivxlcdm]{1,4}|[IVXLCDM]{1,4})\))\s*")

_CONTROL = re.compile(r"\\([A-Za-z@]+|.)", re.S)
_NOMBRE_ENTORNO = re.compile(r"\s*\{([^}]*)\}")
_CIFRA = r"\d+(?:(?:\\,|\\thinspace\s?|\{,\}|[.,])\d+)*"
_NUMERO = re.compile(_CIFRA)
# El cuerpo de una definición que es sólo una cifra (0,79; \signoMenos 0{,}028; 11,3\,\%)...
_CUERPO_CIFRA = re.compile(r"\s*(?:[+-]|\\textminus\s*|\\signoMenos\s*)?\s*" + _CIFRA
                           + r"(?:\s|~|\\,)*(?:\\%)?\s*")
# ...salvo un factor de medida (\renewcommand{\figgrandealto}{0.86}) o un dígito suelto.
_CUERPO_SINTAXIS = re.compile(r"\s*(?:\d|\d*\.\d{1,2})\s*")
_NODO = re.compile(r"\\node\b|(?<![\\A-Za-z@])node\b|\\matrix\b")
_UNIDAD_TEX = re.compile(r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|em|ex|mu|px|nd|nc|fil{1,3})(?![A-Za-z])")


def _sin_comentarios(texto):
    """El texto con cada comentario (de un % sin escapar al fin de la línea) cambiado por espacios:
    mismo largo y mismas líneas."""
    lineas = texto.split("\n")
    for i, linea in enumerate(lineas):
        desde = 0
        while (j := linea.find("%", desde)) >= 0:
            barras = len(linea[:j]) - len(linea[:j].rstrip("\\"))
            if barras % 2 == 0:
                lineas[i] = linea[:j] + " " * (len(linea) - j)
                break
            desde = j + 1
    return "\n".join(lineas)


def _cerrar(texto, i, abre, cierra):
    """Desde texto[i] == abre, el índice siguiente a su cierre, con anidamiento y sin contar las
    llaves escapadas. Dentro de corchetes, las llaves protegen: [a={x]y}] cierra en el último ]."""
    profundidad, llaves, j = 0, 0, i
    while j < len(texto):
        c = texto[j]
        if c == "\\":
            j += 2
            continue
        if abre != "{" and c in "{}":
            llaves += 1 if c == "{" else -1
        elif c == abre and not llaves:
            profundidad += 1
        elif c == cierra and not llaves:
            profundidad -= 1
            if profundidad == 0:
                return j + 1
        j += 1
    return len(texto)


def _saltar_espacios(texto, i, saltos=1):
    while i < len(texto) and texto[i] in " \t\n":
        if texto[i] == "\n":
            if saltos == 0:
                break
            saltos -= 1
        i += 1
    return i


def _saltar_opcionales(texto, i, espacios=False, parentesis=False):
    """Después de un comando: la estrella, los overlays <…>, los opcionales […] y, si
    corresponde, los paréntesis (…). Devuelve dónde termina lo saltado."""
    if i < len(texto) and texto[i] == "*":
        i += 1
    while True:
        j = _saltar_espacios(texto, i) if espacios else i
        if j < len(texto) and texto[j] == "<":
            cierre = texto.find(">", j)
            if cierre < 0:
                return i
            i = cierre + 1
        elif j < len(texto) and texto[j] == "[":
            i = _cerrar(texto, j, "[", "]")
        elif parentesis and j < len(texto) and texto[j] == "(":
            i = _cerrar(texto, j, "(", ")")
        else:
            return i


def _saltar_argumentos(texto, i, obligatorios, parentesis=False):
    """Salta los opcionales y `obligatorios` argumentos: {…}, o un comando suelto (\\x) como
    nombre de una definición."""
    return _grupos_de_argumentos(texto, i, obligatorios, parentesis)[0]


def _grupos_de_argumentos(texto, i, obligatorios, parentesis=False):
    """(fin, [(inicio, fin) de cada argumento entre llaves]) de los `obligatorios` argumentos."""
    grupos = []
    i = _saltar_opcionales(texto, i, espacios=True, parentesis=parentesis)
    for _ in range(obligatorios):
        j = _saltar_espacios(texto, i)
        if j >= len(texto):
            return j, grupos
        if texto[j] == "{":
            i = _cerrar(texto, j, "{", "}")
            grupos.append((j, i))
        elif texto[j] == "\\":
            control = _CONTROL.match(texto, j)
            i = control.end() if control else j + 1
        else:
            return i, grupos
        i = _saltar_opcionales(texto, i, espacios=True, parentesis=parentesis)
    return i, grupos


def _cuerpo_def(texto, i):
    """\\def\\x#1#2{cuerpo}: (inicio, fin) del cuerpo, llaves incluidas, o None."""
    control = _CONTROL.match(texto, _saltar_espacios(texto, i))
    if not control:
        return None
    llave = texto.find("{", control.end())
    return None if llave < 0 else (llave, _cerrar(texto, llave, "{", "}"))


def _saltar_def(texto, i):
    """\\def\\x#1#2{cuerpo}: el nombre, el texto de parámetros y el cuerpo."""
    if not _CONTROL.match(texto, _saltar_espacios(texto, i)):
        return i
    cuerpo = _cuerpo_def(texto, i)
    return len(texto) if cuerpo is None else cuerpo[1]


def _ocultar(oculto, i, j):
    oculto[i:j] = b"\x01" * (j - i)


def _ocultar_entornos(texto, oculto, nombres):
    patron = re.compile(r"\\(begin|end)\s*\{(" + "|".join(map(re.escape, nombres)) + r")\*?\}")
    pila = []
    for m in patron.finditer(texto):
        if m.group(1) == "begin":
            pila.append(m.start())
        elif pila:
            inicio = pila.pop()
            if not pila:
                _ocultar(oculto, inicio, m.end())
    if pila:
        _ocultar(oculto, pila[0], len(texto))


def _ocultar_literales(texto, oculto):
    """\\verb y los entornos de texto literal."""
    for m in re.finditer(r"\\verb\*?(\S)(.*?)\1", texto):
        _ocultar(oculto, m.start(), m.end())
    _ocultar_entornos(texto, oculto, ENTORNOS_OCULTOS)


def _regiones_tikz(texto, oculto):
    """(inicio, fin) de cada tikzpicture y de cada \\tikz suelto que no estén ya ocultos."""
    regiones, pila = [], []
    patron = re.compile(r"\\(begin|end)\s*\{(" + "|".join(ENTORNOS_TIKZ) + r")\}")
    for m in patron.finditer(texto):
        if oculto[m.start()]:
            continue
        if m.group(1) == "begin":
            pila.append(m.start())
        elif pila:
            inicio = pila.pop()
            if not pila:
                regiones.append((inicio, m.end()))
    if pila:
        regiones.append((pila[0], len(texto)))
    for m in re.finditer(r"\\tikz\b", texto):
        if oculto[m.start()] or any(a <= m.start() < b for a, b in regiones):
            continue
        i = _saltar_opcionales(texto, m.end(), espacios=True)
        j = _saltar_espacios(texto, i)
        if j < len(texto) and texto[j] == "{":
            fin = _cerrar(texto, j, "{", "}")
        else:
            fin = texto.find(";", j) + 1 or len(texto)
        regiones.append((m.start(), fin))
    return regiones


def _texto_de_nodo(texto, i, fin):
    """Después de «node» o de «\\matrix»: salta los overlays <…>, las opciones […], el nombre (…)
    y «at (…)». Si sigue una llave, devuelve ((inicio, fin) de su interior, [opciones]); si no,
    (None, [opciones])."""
    opciones = []
    while i < fin:
        i = _saltar_espacios(texto, i, saltos=fin)
        if i >= fin:
            break
        c = texto[i]
        if c == "<":
            cierre = texto.find(">", i, fin)
            if cierre < 0:
                break
            i = cierre + 1
        elif c in "[(":
            j = _cerrar(texto, i, c, "]" if c == "[" else ")")
            if c == "[":
                opciones.append(texto[i:j])
            i = j
        elif texto.startswith("at", i) and not texto[i + 2:i + 3].isalpha():
            i += 2
        elif c == "{":
            return (i + 1, min(_cerrar(texto, i, "{", "}") - 1, fin)), opciones
        else:
            break
    return None, opciones


def _ocultar_tikz(texto, oculto):
    """Oculta cada dibujo de TikZ salvo el texto de sus nodos: el {…} que sigue a \\node o a node
    (con sus opciones, su nombre y su «at (…)» en el medio), y las celdas de un «matrix of nodes»
    sin sus opciones |[…]|. Las opciones de cualquier comando (label=, pin=) quedan ocultas."""
    for a, b in _regiones_tikz(texto, oculto):
        previo = bytes(oculto[a:b])
        _ocultar(oculto, a, b)
        for m in _NODO.finditer(texto, a, b):
            grupo, opciones = _texto_de_nodo(texto, m.end(), b)
            matriz = m.group() == "\\matrix"
            if grupo is None or (matriz and not any("matrix of" in o for o in opciones)):
                continue
            x, y = grupo
            oculto[x:y] = previo[x - a:y - a]
            if matriz:
                for celda in re.finditer(r"\|[^|\n]*\|", texto[x:y]):
                    _ocultar(oculto, x + celda.start(), x + celda.end())


def _ocultar_opcionales_de_texto(texto, oculto, i, rotulo):
    """Como _saltar_opcionales, pero el contenido de […] es texto: se ocultan sólo la estrella, los
    overlays y los corchetes. Con rotulo=True (\\item), un rótulo como «1.», «a)» o «(3)» también
    es sintaxis."""
    if i < len(texto) and texto[i] == "*":
        i += 1
    while True:
        j = _saltar_espacios(texto, i)
        if j < len(texto) and texto[j] == "<":
            cierre = texto.find(">", j)
            if cierre < 0:
                return
            _ocultar(oculto, j, cierre + 1)
            i = cierre + 1
        elif j < len(texto) and texto[j] == "[":
            k = _cerrar(texto, j, "[", "]")
            _ocultar(oculto, j, j + 1)
            _ocultar(oculto, k - 1, k)
            if rotulo and _ROTULO.fullmatch(texto[j + 1:k - 1]):
                _ocultar(oculto, j, k)
            i = k
        else:
            return


def _ocultar_estructura(texto, oculto):
    """Preámbulo, texto literal, sintaxis de TikZ, definiciones, argumentos de sintaxis, overlays y
    opcionales."""
    inicio = re.search(r"\\begin\s*\{document\}", texto)
    if inicio:
        _ocultar(oculto, 0, inicio.end())
    fin = re.search(r"\\end\s*\{document\}", texto)
    if fin:
        _ocultar(oculto, fin.start(), len(texto))
    _ocultar_literales(texto, oculto)
    _ocultar_tikz(texto, oculto)

    for m in _CONTROL.finditer(texto):
        if oculto[m.start()]:
            continue
        nombre, fin = m.group(1), m.end()
        if nombre in DEFINICIONES:
            _ocultar(oculto, m.start(), _saltar_argumentos(texto, fin, DEFINICIONES[nombre]))
        elif nombre in DEF_TEX:
            _ocultar(oculto, m.start(), _saltar_def(texto, fin))
        elif nombre == "begin":
            entorno = _NOMBRE_ENTORNO.match(texto, fin)
            if not entorno:
                continue
            despues = entorno.end()
            obligatorios = ENTORNOS_CON_ARGUMENTOS.get(entorno.group(1), 0)
            final = (_saltar_argumentos(texto, despues, obligatorios) if obligatorios
                     else _saltar_opcionales(texto, despues, espacios=True))
            _ocultar(oculto, despues, final)
        elif nombre in ARGUMENTOS_DE_SINTAXIS:
            _ocultar(oculto, m.start(), _saltar_argumentos(texto, fin, ARGUMENTOS_DE_SINTAXIS[nombre],
                                                          parentesis=nombre in CON_PARENTESIS))
        elif nombre in OPCIONAL_DE_TEXTO:
            _ocultar_opcionales_de_texto(texto, oculto, fin, rotulo=nombre == "item")
        elif nombre.isalpha() or nombre == "\\":
            final = _saltar_opcionales(texto, fin, espacios=nombre in CON_ESPACIO,
                                       parentesis=nombre in CON_PARENTESIS)
            _ocultar(oculto, fin, final)


def _definiciones_con_cifra(texto, literal):
    """[(inicio, fin, cuerpo)] de las definiciones de macro cuyo cuerpo es sólo una cifra
    (\\newcommand{\\miauc}{0,79}): esa cifra está escrita a mano. Un factor de medida o un dígito
    suelto (\\renewcommand{\\figgrandealto}{0.86}) es sintaxis."""
    hallados = []
    for m in _CONTROL.finditer(texto):
        if literal[m.start()]:
            continue
        nombre = m.group(1)
        if nombre in DEF_TEX:
            cuerpo = _cuerpo_def(texto, m.end())
        elif nombre in DEFINICIONES_DE_MACRO:
            grupos = _grupos_de_argumentos(texto, m.end(), DEFINICIONES[nombre])[1]
            cuerpo = grupos[-1] if grupos else None
        else:
            continue
        if cuerpo is None:
            continue
        a, b = cuerpo[0] + 1, cuerpo[1] - 1
        interior = texto[a:b]
        if _CUERPO_CIFRA.fullmatch(interior) and not _CUERPO_SINTAXIS.fullmatch(interior):
            hallados.append((a, b, interior.strip()))
    return hallados


def _patrones(permitidos):
    patrones = []
    for permitido in permitidos:
        if isinstance(permitido, tuple):
            permitido = permitido[1]
        patrones.append(permitido if isinstance(permitido, re.Pattern) else re.compile(permitido))
    return patrones


def _parte_de_un_nombre(texto, i, j):
    """El número texto[i:j] es parte de un nombre (euribor3m, A7, F_1, x^2, F_{1}, basic.4y), de un
    color (azul!20), de un parámetro (#1) o de una medida (0.5em, 2pt). Una letra que no es una
    unidad de TeX después del número no lo esconde («600s», «33k», «0,31pp»), y una x suelta
    delante tampoco: «x27» es un factor."""
    if j < len(texto) and _UNIDAD_TEX.match(texto, j):
        return True
    if i == 0:
        return False
    antes = texto[i - 1]
    if antes in "_^@#!":
        return True
    if antes == "{" and i >= 2 and texto[i - 2] in "_^":
        return True     # F_{1}: un subíndice
    if antes == "." and i >= 2 and texto[i - 2].isalpha():
        return True     # basic.4y: un nivel de education
    if antes.isalpha():
        suelta = i < 2 or not (texto[i - 2].isalnum() or texto[i - 2] in "_\\")
        return not (antes in "xX" and suelta)
    return False


def numeros_a_mano(texto_tex, permitidos=PERMITIDOS, generado=False):
    """Los números escritos a mano en un .tex: una lista de (línea, fragmento), en orden.

    `permitidos` son expresiones regulares (texto, compiladas o pares (qué es, expresión)) de
    números que no son datos; si una tiene un grupo «n», sólo ese grupo queda permitido. Con
    generado=True (numeros.tex, resultados-test.tex), las definiciones cuyo cuerpo es una cifra no
    cuentan: son los macros. Ver la documentación del módulo para lo que no se cuenta."""
    texto = _sin_comentarios(texto_tex)
    oculto = bytearray(len(texto))
    _ocultar_estructura(texto, oculto)
    permitido = bytearray(len(texto))
    for patron in _patrones(permitidos):
        for m in patron.finditer(texto):
            grupo = "n" in patron.groupindex and m.start("n") >= 0
            _ocultar(permitido, *(m.span("n") if grupo else m.span()))

    hallados = []
    for m in _NUMERO.finditer(texto):
        i, j = m.span()
        if all(oculto[k] or permitido[k] for k in range(i, j) if texto[k].isdigit()):
            continue
        if _parte_de_un_nombre(texto, i, j):
            continue
        hallados.append((i, m.group()))
    if not generado:
        literal = bytearray(len(texto))
        _ocultar_literales(texto, literal)
        for i, j, cuerpo in _definiciones_con_cifra(texto, literal):
            if not all(permitido[k] for k in range(i, j) if texto[k].isdigit()):
                hallados.append((i, cuerpo))
    return [(texto.count("\n", 0, i) + 1, fragmento) for i, fragmento in sorted(hallados)]


def _es_generado(nombre):
    """Un \\input de los macros generados: sólo tiene definiciones y no se sigue."""
    return Path(nombre).name in GENERADOS or Path(nombre + ".tex").name in GENERADOS


def entradas_del_cuerpo(texto_tex):
    """Los \\input y \\include del cuerpo, [(línea, nombre)], sin los de los macros generados."""
    texto = _sin_comentarios(texto_tex)
    literal = bytearray(len(texto))
    _ocultar_literales(texto, literal)
    inicio = re.search(r"\\begin\s*\{document\}", texto)
    desde = inicio.end() if inicio else 0
    entradas = []
    for m in re.finditer(r"\\(?:input|include)\s*\{([^}]+)\}", texto):
        nombre = m.group(1).strip()
        if m.start() < desde or literal[m.start()] or _es_generado(nombre):
            continue
        entradas.append((texto.count("\n", 0, m.start()) + 1, nombre))
    return entradas


def resolver_entrada(nombre, carpetas):
    """Dónde encuentra LaTeX un \\input: en cada carpeta, en orden, primero con .tex agregado y
    después tal cual. None si no está en ninguna."""
    for carpeta in carpetas:
        for candidato in (nombre + ".tex", nombre):
            ruta = Path(carpeta) / candidato
            if ruta.is_file():
                return ruta
    return None


def avisos_de_formato(texto_tex):
    """Avisos sobre el preámbulo que no son números escritos a mano."""
    texto = _sin_comentarios(texto_tex)
    inicio = re.search(r"\\begin\s*\{document\}", texto)
    preambulo = texto[:inicio.start()] if inicio else texto
    babel = re.search(r"\\usepackage\s*(\[[^\]]*\])?\s*\{babel\}", preambulo)
    clase = re.search(r"\\documentclass\s*\[([^\]]*)\]", preambulo)
    espanol = babel and any(re.search(r"\bspanish\b", opciones or "")
                            for opciones in (babel.group(1), clase and clase.group(1)))
    if espanol and not re.search(r"\\spanishplainpercent\b", texto):
        return ["carga babel en español sin \\spanishplainpercent: babel agrega su propio espacio "
                "fino antes de %, y el \\,\\% de los macros queda con espacio doble"]
    return []


@dataclass(frozen=True)
class Verificacion:
    hallados: list        # (ruta, línea, fragmento, texto de la línea)
    archivos: int         # cuántos .tex se leyeron
    sin_resolver: list    # (ruta, línea, nombre) de cada \input que no se encontró
    avisos: list          # (ruta, aviso)


def verificar(rutas, permitidos=PERMITIDOS):
    """Cada .tex y los que incluye su cuerpo, a cualquier profundidad. Cada \\input se busca
    primero en la carpeta del .tex raíz, que es donde LaTeX compila, y después en la del archivo
    que lo incluye."""
    hallados, sin_resolver, avisos, vistas = [], [], [], set()
    pendientes = [(Path(r), Path(r).parent, True) for r in rutas]
    while pendientes:
        ruta, raiz, es_raiz = pendientes.pop(0)
        clave = ruta.resolve()
        if clave in vistas:
            continue
        vistas.add(clave)
        texto = ruta.read_text(encoding="utf-8")
        lineas = texto.split("\n")
        hallados += [(ruta, n, fragmento, lineas[n - 1].strip()) for n, fragmento in
                     numeros_a_mano(texto, permitidos, generado=ruta.name in GENERADOS)]
        if es_raiz:
            avisos += [(ruta, aviso) for aviso in avisos_de_formato(texto)]
        for linea, nombre in entradas_del_cuerpo(texto):
            destino = resolver_entrada(nombre, dict.fromkeys([raiz, ruta.parent]))
            if destino is None:
                sin_resolver.append((ruta, linea, nombre))
            else:
                pendientes.append((destino, raiz, False))
    return Verificacion(hallados, len(vistas), sin_resolver, avisos)


# --- Línea de comandos --------------------------------------------------------------------------

def _argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m src.numeros",
        description="Genera informe/numeros.tex e informe/numeros.md desde resultados/, o verifica "
                    "que un .tex no tenga números escritos a mano (gate G3).")
    p.add_argument("--resultados", type=Path, default=DIR,
                   help="directorio de resultados (por defecto, resultados/)")
    p.add_argument("--salida-tex", type=Path, default=RUTA_TEX,
                   help="dónde escribir los macros (por defecto, informe/numeros.tex)")
    p.add_argument("--salida-md", type=Path, default=RUTA_MD,
                   help="dónde escribir la lista legible (por defecto, informe/numeros.md)")
    p.add_argument("--resultados-test", type=Path, default=RUTA_RESULTADOS_TEST,
                   help="los macros de test que se repiten en el .md (por defecto, "
                        "informe/resultados-test.tex)")
    p.add_argument("--verificar", type=Path, nargs="+", metavar="RUTA.tex",
                   help="lista los números escritos a mano en el cuerpo de estos .tex; código 1 si "
                        "hay alguno")
    p.add_argument("--permitir", action="append", default=[], metavar="REGEX",
                   help="con --verificar, una expresión regular más de números permitidos")
    return p.parse_args(argv)


def _misma(a, b):
    return Path(a).resolve() == Path(b).resolve()


def main(argv=None):
    args = _argumentos(argv)
    if args.verificar:
        permitidos = list(PERMITIDOS) + args.permitir
        faltan = [r for r in args.verificar if not r.exists()]
        if faltan:
            print("ERROR: no existe " + ", ".join(map(str, faltan)), file=sys.stderr)
            return 1
        resultado = verificar(args.verificar, permitidos)
        for ruta, linea, fragmento, contexto in resultado.hallados:
            print(f"{_mostrar(ruta)}:{linea}: {fragmento}    | {contexto[:100]}")
        for ruta, linea, nombre in resultado.sin_resolver:
            print(f"{_mostrar(ruta)}:{linea}: \\input{{{nombre}}} no se encuentra (se buscó en la "
                  "carpeta del .tex raíz y en la del archivo que lo incluye)")
        for ruta, aviso in resultado.avisos:
            print(f"AVISO: {_mostrar(ruta)} {aviso}.")
        if resultado.hallados:
            print(f"\n{len(resultado.hallados)} números escritos a mano en {resultado.archivos} "
                  "archivo(s): cada uno tiene que ser un macro de informe/numeros.tex o de "
                  "informe/resultados-test.tex, o entrar en la lista permitida (--permitir REGEX).")
        if resultado.sin_resolver:
            print(f"\n{len(resultado.sin_resolver)} \\input sin resolver: su contenido no se "
                  "verificó.")
        if resultado.hallados or resultado.sin_resolver:
            return 1
        print(f"OK: ningún número escrito a mano en {resultado.archivos} archivo(s).")
        return 0

    # Un directorio de resultados que no es el de siempre no pisa los macros del deck.
    if not _misma(args.resultados, DIR) and (_misma(args.salida_tex, RUTA_TEX)
                                              or _misma(args.salida_md, RUTA_MD)):
        print("ERROR: con --resultados distinto de resultados/ hay que pasar --salida-tex y "
              "--salida-md, para no pisar los macros del deck.", file=sys.stderr)
        return 1
    fuentes = Fuentes(args.resultados, resultados_test=args.resultados_test)
    generado = generar(fuentes)
    escribir(args.salida_tex, generado.tex)
    escribir(args.salida_md, generado.md)
    pendientes = sum(len(n) for n in generado.faltan.values())
    print(f"Escritos {_mostrar(args.salida_tex)} y {_mostrar(args.salida_md)}: "
          f"{len(generado.valores)} macros, {pendientes} con «?».")
    for que, nombres in generado.faltan.items():
        print(f"  falta {que}: " + ", ".join(nombres))
    return 0


if __name__ == "__main__":
    sys.exit(main())
