"""Correr con: python -m src.curvas --modelo rf --parametro max_depth

Ola 4 del plan (pasos 4.1 a 4.4; D-21 y D-22): las curvas de validación. Es el protocolo de la
Clase 7, slide 50: para cada valor de un hiperparámetro se mide el AUC en train y en validación con
los cinco folds de `folds()`, y se lee la brecha entre los dos. Este módulo sólo calcula; las
figuras son del paso 4.5.

Cada curva recorre un parámetro sobre la configuración vigente (`src/configuracion.py`) y escribe
resultados/curvas/<modelo>_<parametro>[_<sufijo>].csv en formato largo: `parametro`, `punto` (el
valor como texto) y las seis columnas del contrato de `src/resultados.py`; `configuracion` lleva
los hiperparámetros completos del punto. Los valores salen de `GRILLAS` salvo que se pasen con
--valores. Las curvas del plan, cada una en su propio proceso si se quiere:

    python -m src.curvas --modelo rf --parametro max_depth
    python -m src.curvas --modelo rf --parametro n_estimators
    python -m src.curvas --modelo knn --parametro n_neighbors --fijo weights=uniform --sufijo uniform
    python -m src.curvas --modelo knn --parametro n_neighbors --fijo weights=distance --sufijo distance
    python -m src.curvas --modelo svm --parametro C
    python -m src.curvas --modelo svm --parametro kernel --valores linear,poly,rbf --fijo C=<mejor C>
    python -m src.curvas --modelo rf --parametro pesos_clase --valores None,balanced
    python -m src.curvas --modelo svm --parametro pesos_clase --valores None,balanced
    python -m src.curvas --elegir

Kernel lineal: todo punto con kernel="linear" usa implementacion="liblinear" (LinearSVC con pérdida
hinge, la misma SVM lineal), salvo que se fije otra con --fijo. Con libsvm, un solo ajuste lineal
con C = 10 pasa de 600 s (resultados/costos.csv), y el presupuesto de la comparación de kernels
está calculado con liblinear (resultados/costos_grillas.csv).

Sin recortes en silencio: un punto que falla, o que pasa el límite de --limite, se omite, se informa
en pantalla y queda en resultados/curvas/omitidos.txt con su motivo. Cada corrida reemplaza en ese
archivo las líneas de su curva.

--elegir aplica D-22 a cada CSV de resultados/curvas/: el punto de mayor AUC medio de validación y,
si otros quedan dentro de 1 error estándar del mejor (el del mejor, de `resumen()`), el más simple
entre ellos. Más simple es menor max_depth (None, sin límite, es el más complejo), menos árboles,
más vecinos (una frontera más suave) y menor C (más regularización). Los parámetros sin orden de
simplicidad (kernel, weights, pesos_clase) se quedan con el de mayor media; entre dos curvas del
mismo eje que difieren en un valor fijo (weights en KNN), gana la de mayor media en su punto
elegido. Escribe resultados/hiperparametros.json con tres claves:
- "modelos": por modelo, los valores elegidos, listos para `crear_modelo(nombre, **valores)`. Si
  gana el kernel lineal, lleva también implementacion="liblinear";
- "curvas": por curva, el punto elegido con su AUC de validación y de train, la brecha, el error
  estándar, si la curva es plana y qué puntos quedan dentro de 1 error estándar;
- "avisos": las curvas adoptadas que se midieron con una configuración distinta de la final (por
  ejemplo, C elegido con RBF cuando gana el kernel lineal).

--resumen concatena las curvas y escribe resultados/curvas_resumen.csv: media, desvío y error
estándar por curva, punto, conjunto y métrica. --rapido corre con una submuestra estratificada de
3 000 filas, un proceso, como máximo tres valores de la grilla y RF con hasta 20 árboles, y escribe
en un directorio temporal, nunca en resultados/.
"""

import argparse
import json
import re
import signal
import tempfile
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configuracion import HIPERPARAMETROS_FINALES, OPCIONES_FINALES
from src.datos import FILA, OBJETIVO, RAIZ, SEMILLA, cargar_train
from src.modelos import GRILLAS, MODELOS, REFERENCIA, crear_modelo, separar_X_y
from src.resultados import (
    CAMPOS,
    DIR,
    N_JOBS,
    con_identidad,
    configuracion_json,
    escribir_largo,
    leer_largo,
    resumen,
    validacion_cruzada,
)
from src.validacion import K

try:
    import fcntl
except ImportError:  # Windows: sin cerrojo, así que las curvas no deben correr en paralelo
    fcntl = None

DIR_CURVAS = DIR / "curvas"
# Los números de --rapido son de una submuestra: si cayeran en resultados/curvas/, --elegir los
# mezclaría con las curvas de verdad.
DIR_RAPIDO = Path(tempfile.gettempdir()) / "tp2-rapido" / "curvas"
OMITIDOS = "omitidos.txt"
JSON_ELEGIDOS = "hiperparametros.json"
COLUMNAS = ["parametro", "punto"] + CAMPOS

FILAS_RAPIDO = 3000
PUNTOS_RAPIDO = 3
ARBOLES_RAPIDO = 20

# D-22: hacia dónde es más simple cada eje. Un parámetro que no figura aquí no tiene orden de
# simplicidad y se elige por la mayor media.
SIMPLICIDAD = {
    "max_depth": ("menor", "árboles menos profundos"),
    "n_estimators": ("menor", "menos árboles"),
    "n_neighbors": ("mayor", "más vecinos, una frontera más suave"),
    "C": ("menor", "más regularización"),
}
CRITERIO = ("D-22: el punto de mayor AUC medio de validación; si otros quedan dentro de 1 error "
            "estándar del mejor, el más simple entre ellos. Sin orden de simplicidad (kernel, "
            "weights, pesos_clase), el de mayor media. Entre curvas del mismo eje que difieren en un "
            "valor fijo, la de mayor media en su punto elegido")

# Los hiperparámetros que `crear_modelo` saca antes de armar el estimador, con su valor cuando
# faltan.
DEFECTOS_PROPIOS = {"pesos_clase": None, "implementacion": "libsvm"}

_ENTERO = re.compile(r"[+-]?\d+")
_REAL = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?")
_AUSENTE = "(ausente)"


class CurvaInvalida(ValueError):
    """Un pedido de curva mal armado. Se detecta antes de ajustar nada."""


def interpretar(texto):
    """Un valor de la línea de comandos con su tipo: None, bool, int, float o texto."""
    texto = texto.strip()
    if texto in ("None", "none"):
        return None
    if texto in ("True", "true", "False", "false"):
        return texto.lower() == "true"
    if _ENTERO.fullmatch(texto):
        return int(texto)
    if _REAL.fullmatch(texto):
        return float(texto)
    return texto


def punto_de(valor):
    """El valor de un punto como texto: el mismo en el CSV, en omitidos.txt y en el JSON."""
    return str(valor)


def parsear_valores(texto):
    """'2,4,None' -> [2, 4, None]."""
    partes = [p.strip() for p in texto.split(",")]
    if not all(partes):
        raise CurvaInvalida(f"--valores tiene un valor vacío: {texto!r}")
    valores = [interpretar(p) for p in partes]
    if len({punto_de(v) for v in valores}) < len(valores):
        raise CurvaInvalida(f"--valores repite un valor: {texto!r}")
    return valores


def parsear_fijos(pares):
    """['weights=distance', 'C=0.1'] -> {'weights': 'distance', 'C': 0.1}."""
    fijos = {}
    for par in pares:
        clave, igual, texto = par.partition("=")
        if not igual or not clave.strip() or not texto.strip():
            raise CurvaInvalida(f"--fijo espera clave=valor, no {par!r}")
        fijos[clave.strip()] = interpretar(texto)
    return fijos


def nombre_curva(modelo, parametro, sufijo=None):
    return "_".join([modelo, parametro] + ([sufijo] if sufijo else []))


def hiperparametros_del_punto(modelo, parametro, valor, fijos=None, rapido=False):
    """Los hiperparámetros completos de un punto: la referencia, la configuración vigente, los fijos
    y el valor de la curva, cada uno sobre el anterior."""
    fijos = fijos or {}
    h = {**REFERENCIA.get(modelo, {}), **HIPERPARAMETROS_FINALES.get(modelo, {}), **fijos,
         parametro: valor}
    if modelo == "svm" and "implementacion" not in fijos and parametro != "implementacion":
        if h.get("kernel") == "linear":
            h["implementacion"] = "liblinear"
        elif h.get("implementacion") == "liblinear":
            # Heredada de una configuración con kernel lineal: liblinear no tiene otros kernels.
            del h["implementacion"]
    if rapido and modelo == "rf" and h.get("n_estimators", 0) > ARBOLES_RAPIDO:
        h["n_estimators"] = ARBOLES_RAPIDO
    return h


def recortar_para_rapido(modelo, parametro, valores, explicitos):
    """Los valores de --rapido: como máximo tres de la grilla (los extremos y el del medio), salvo
    que se hayan pasado a mano, y RF con hasta 20 árboles."""
    if not explicitos and len(valores) > PUNTOS_RAPIDO:
        valores = [valores[0], valores[len(valores) // 2], valores[-1]]
    if modelo == "rf" and parametro == "n_estimators":
        valores = list(dict.fromkeys(min(v, ARBOLES_RAPIDO) for v in valores))
    return valores


def muestra_rapida(train):
    muestra, _ = train_test_split(train, train_size=FILAS_RAPIDO, stratify=train[OBJETIVO],
                                  random_state=SEMILLA)
    return muestra.sort_values(FILA).reset_index(drop=True)


@contextmanager
def limite_de_tiempo(segundos):
    """Levanta TimeoutError si el bloque tarda más de `segundos` (None: sin límite).

    Usa SIGALRM. Con varios procesos, joblib termina los suyos al recibir la excepción. Con
    n_jobs=1 la señal se atiende entre instrucciones de Python: un ajuste de libsvm en curso
    termina antes de que el punto se omita.
    """
    if not segundos:
        yield
        return

    def vencido(*_):
        raise TimeoutError(f"excedió el límite de {segundos:g} s")

    anterior = signal.signal(signal.SIGALRM, vencido)
    signal.setitimer(signal.ITIMER_REAL, segundos)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, anterior)


def registrar_omitidos(directorio, curva, omitidos):
    """Reemplaza en omitidos.txt las líneas de `curva` por las de esta corrida: `curva`, `etiqueta`
    y `motivo`, separados por tabuladores. Otras curvas pueden estar corriendo en otros procesos,
    así que el archivo se reescribe bajo un cerrojo."""
    ruta = Path(directorio) / OMITIDOS
    if not omitidos and not ruta.exists():
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a+", encoding="utf-8") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        lineas = [l for l in f.read().splitlines() if l and l.split("\t", 1)[0] != curva]
        lineas += [f"{curva}\t{etiqueta}\t{motivo}" for etiqueta, motivo in omitidos]
        f.seek(0)
        f.truncate()
        f.write("".join(l + "\n" for l in lineas))


def leer_omitidos(directorio):
    """{curva: [(etiqueta, motivo), ...]} según omitidos.txt."""
    ruta = Path(directorio) / OMITIDOS
    omitidos = {}
    if ruta.exists():
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            curva, _, resto = linea.partition("\t")
            etiqueta, _, motivo = resto.partition("\t")
            if curva:
                omitidos.setdefault(curva, []).append((etiqueta, motivo))
    return omitidos


def correr_curva(modelo, parametro, valores=None, fijos=None, sufijo=None, n_jobs=None,
                 rapido=False, directorio=None, limite=None):
    """Mide la curva y escribe su CSV. Devuelve la ruta, o None si ningún punto terminó."""
    if modelo not in MODELOS:
        raise CurvaInvalida(f"modelo desconocido: {modelo}; los válidos son {', '.join(MODELOS)}")
    fijos = dict(fijos or {})
    if parametro in fijos:
        raise CurvaInvalida(f"{parametro} no puede ser a la vez el eje de la curva y un valor fijo")
    if limite and not hasattr(signal, "SIGALRM"):
        raise CurvaInvalida("--limite usa la señal SIGALRM, que este sistema no tiene")
    explicitos = valores is not None
    if not explicitos:
        if (modelo, parametro) not in GRILLAS:
            hay = ", ".join(f"{m}/{p}" for m, p in GRILLAS)
            raise CurvaInvalida(f"no hay grilla de {modelo}/{parametro} en src/modelos.py (las hay "
                                f"de {hay}); pase los valores con --valores")
        valores = list(GRILLAS[(modelo, parametro)])
    if not valores:
        raise CurvaInvalida("la curva no tiene valores")
    if rapido:
        valores = recortar_para_rapido(modelo, parametro, valores, explicitos)
    n_jobs = n_jobs or (1 if rapido else N_JOBS)
    directorio = Path(directorio) if directorio else (DIR_RAPIDO if rapido else DIR_CURVAS)
    curva = nombre_curva(modelo, parametro, sufijo)
    ruta = directorio / f"{curva}.csv"

    train = cargar_train()
    if rapido:
        train = muestra_rapida(train)
    X, y = separar_X_y(train)

    base = {**REFERENCIA[modelo], **HIPERPARAMETROS_FINALES.get(modelo, {})}
    print(f"Curva {curva}: {parametro} en {', '.join(punto_de(v) for v in valores)}")
    print(f"  configuración vigente de {modelo}: {configuracion_json(base)}"
          + (f"; fijos: {configuracion_json(fijos)}" if fijos else ""))
    rapida = ""
    if rapido:
        rapida = (" (submuestra de --rapido"
                  + (f"; RF con hasta {ARBOLES_RAPIDO} árboles)" if modelo == "rf" else ")"))
    print(f"  {len(X):,} filas{rapida}, {K} folds, n_jobs={n_jobs}"
          + (f", límite de {limite:g} s por punto" if limite else "")
          + f"; destino {_mostrar(ruta)}", flush=True)

    partes, omitidos = [], []
    inicio = time.perf_counter()
    for valor in valores:
        etiqueta = f"{parametro}={punto_de(valor)}"
        t0 = time.perf_counter()
        try:
            hiper = hiperparametros_del_punto(modelo, parametro, valor, fijos, rapido)
            with limite_de_tiempo(limite):
                tabla, _ = validacion_cruzada(crear_modelo(modelo, OPCIONES_FINALES, **hiper), X, y,
                                              filas=train[FILA], n_jobs=n_jobs)
        except Exception as error:
            motivo = _motivo(error)
            omitidos.append((etiqueta, motivo))
            print(f"  {etiqueta:<26s} OMITIDO a los {time.perf_counter() - t0:.1f} s. {motivo}",
                  flush=True)
            continue
        segundos = time.perf_counter() - t0
        partes.append(con_identidad(tabla, modelo, hiper, parametro=parametro,
                                    punto=punto_de(valor)))
        auc = resumen(tabla[tabla["metrica"] == "auc"], por=["conjunto"]).set_index("conjunto")
        va, tr = auc.loc["validacion"], auc.loc["train"]
        mostrada = etiqueta + (" (liblinear)" if hiper.get("implementacion") == "liblinear" else "")
        print(f"  {mostrada:<26s} AUC validación {va['media']:.4f} ± {va['error_estandar']:.4f}   "
              f"train {tr['media']:.4f}   brecha {tr['media'] - va['media']:+.4f}   "
              f"{segundos:6.1f} s", flush=True)

    registrar_omitidos(directorio, curva, omitidos)
    total = time.perf_counter() - inicio
    if not partes:
        anterior = (f" Queda el CSV de una corrida anterior: {_mostrar(ruta)}."
                    if ruta.exists() else "")
        print(f"Ningún punto de {curva} terminó y no se escribe la curva.{anterior} Los motivos "
              f"están en {_mostrar(directorio / OMITIDOS)}.")
        return None
    # Se escribe aparte y se renombra: un proceso interrumpido no deja una curva a medias que
    # --elegir tomaría por completa.
    temporal = ruta.with_name(ruta.name + ".tmp")
    escribir_largo(temporal, pd.concat(partes, ignore_index=True))
    temporal.replace(ruta)
    detalle = f" (ver {_mostrar(directorio / OMITIDOS)})" if omitidos else ""
    print(f"Escrito {_mostrar(ruta)}: {_cantidad(len(partes), 'punto')}, "
          f"{_cantidad(len(omitidos), 'omitido')}{detalle}, {total:.1f} s en total.")
    return ruta


def leer_curvas(directorio):
    """Las curvas de un directorio, {nombre: tabla}, con `punto` reconstruido desde `configuracion`.

    pandas lee el texto «None» como NaN, así que el valor de cada punto se toma del JSON de
    `configuracion`, que conserva su tipo."""
    curvas = {}
    for ruta in sorted(Path(directorio).glob("*.csv")):
        tabla = leer_largo(ruta)
        faltan = [c for c in COLUMNAS if c not in tabla.columns]
        if faltan or tabla.empty:
            print(f"Se ignora {ruta.name}: no es una curva"
                  + (f" (faltan {', '.join(faltan)})." if faltan else " (está vacía)."))
            continue
        if tabla["modelo"].nunique() != 1 or tabla["parametro"].nunique() != 1:
            raise ValueError(f"{ruta.name}: una curva tiene un solo modelo y un solo parámetro")
        parametro = str(tabla["parametro"].iloc[0])
        leidas = {c: json.loads(c) for c in tabla["configuracion"].unique()}
        tabla["parametro"] = parametro
        tabla["punto"] = [punto_de(leidas[c][parametro]) for c in tabla["configuracion"]]
        if (tabla.groupby("punto")["configuracion"].nunique() > 1).any():
            raise ValueError(f"{ruta.name}: hay un punto con dos configuraciones distintas")
        curvas[ruta.stem] = tabla
    return curvas


def resumen_por_punto(tabla):
    """AUC medio de validación (con su error estándar) y de train por punto, y la brecha, en el
    orden de la grilla."""
    orden = list(dict.fromkeys(tabla["punto"]))
    r = resumen(tabla[tabla["metrica"] == "auc"], por=["punto", "conjunto"])
    ancho = r.pivot(index="punto", columns="conjunto",
                    values=["media", "error_estandar"]).reindex(orden)

    def columna(estadistico, conjunto):
        clave = (estadistico, conjunto)
        return ancho[clave] if clave in ancho.columns else pd.Series(np.nan, index=ancho.index)

    por_punto = pd.DataFrame({
        "auc_validacion_media": columna("media", "validacion"),
        "error_estandar": columna("error_estandar", "validacion"),
        "auc_train_media": columna("media", "train"),
    })
    por_punto["brecha"] = por_punto["auc_train_media"] - por_punto["auc_validacion_media"]
    por_punto.index.name = "punto"
    return por_punto


def _complejidad(valor, direccion):
    if valor is None:  # max_depth=None: sin límite, el extremo más complejo
        return np.inf
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"el valor {valor!r} no tiene orden de simplicidad")
    return valor if direccion == "menor" else -valor


def elegir_punto(tabla):
    """D-22 sobre una curva: el punto elegido y las cifras que lo justifican."""
    parametro = tabla["parametro"].iloc[0]
    por_punto = resumen_por_punto(tabla)
    configuraciones = (tabla.drop_duplicates("punto").set_index("punto")["configuracion"]
                       .map(json.loads))
    medias = por_punto["auc_validacion_media"]
    errores = por_punto["error_estandar"].fillna(0.0)  # un solo fold: sin error estándar
    mejor = medias.idxmax()
    umbral = medias[mejor] - errores[mejor]
    dentro = [p for p in por_punto.index if medias[p] >= umbral]
    direccion, razon = SIMPLICIDAD.get(parametro, (None, None))
    if direccion:
        elegido = min(dentro, key=lambda p: _complejidad(configuraciones[p][parametro], direccion))
        criterio = f"dentro de 1 ES del mejor, el más simple: {razon}"
    else:
        elegido = mejor
        criterio = "sin orden de simplicidad: el de mayor AUC medio de validación"
    fila = por_punto.loc[elegido]
    return {
        "modelo": str(tabla["modelo"].iloc[0]),
        "parametro": parametro,
        "valor": configuraciones[elegido][parametro],
        "punto": elegido,
        "criterio": criterio,
        "auc_validacion_media": _numero(fila["auc_validacion_media"]),
        "auc_train_media": _numero(fila["auc_train_media"]),
        "brecha": _numero(fila["brecha"]),
        "error_estandar": _numero(fila["error_estandar"]),
        "curva_plana": len(dentro) > 1,
        "puntos_dentro_1es": dentro,
        "mejor": mejor,
        "auc_validacion_mejor": _numero(medias[mejor]),
        "error_estandar_mejor": _numero(por_punto.loc[mejor, "error_estandar"]),
        "umbral_1es": _numero(umbral),
        "puntos": list(por_punto.index),
        "configuracion": configuraciones[elegido],
    }


def por_modelo(curvas):
    """Junta las elecciones por modelo. Marca en cada curva si quedó adoptada y devuelve
    (modelos, avisos)."""
    grupos = {}
    for nombre, info in curvas.items():
        grupos.setdefault((info["modelo"], info["parametro"]), []).append(nombre)

    modelos, avisos = {}, []
    for (modelo, parametro), nombres in grupos.items():
        ganadora = max(nombres, key=lambda n: -np.inf if curvas[n]["auc_validacion_media"] is None
                       else curvas[n]["auc_validacion_media"])
        # Las claves en las que difieren las curvas del mismo eje (weights en KNN) viajan con el
        # valor elegido: sin ellas, el k de la curva ganadora se usaría con otro weights.
        configuraciones = [curvas[n]["configuracion"] for n in nombres]
        claves = set().union(*configuraciones) - {parametro}
        variantes = sorted(k for k in claves
                           if len({json.dumps(c.get(k), sort_keys=True) for c in configuraciones}) > 1)
        elegidos = {parametro: curvas[ganadora]["valor"],
                    **{k: curvas[ganadora]["configuracion"].get(k) for k in variantes}}
        destino = modelos.setdefault(modelo, {})
        for clave, valor in elegidos.items():
            if clave in destino and destino[clave] != valor:
                avisos.append(f"{modelo}: {clave} sale de dos curvas con valores distintos "
                              f"({punto_de(destino[clave])} y {punto_de(valor)}); queda el de "
                              f"{ganadora}")
            destino[clave] = valor
        for n in nombres:
            curvas[n]["adoptada"] = n == ganadora

    if modelos.get("svm", {}).get("kernel") == "linear":
        modelos["svm"]["implementacion"] = "liblinear"

    for nombre, info in curvas.items():
        if not info["adoptada"]:
            continue
        modelo = info["modelo"]
        final = {**REFERENCIA.get(modelo, {}), **HIPERPARAMETROS_FINALES.get(modelo, {}),
                 **modelos[modelo]}
        medida = info["configuracion"]
        distintas = sorted(k for k in set(final) | set(medida)
                           if _efectivo(modelo, medida, k) != _efectivo(modelo, final, k))
        if distintas:
            avisos.append(f"{nombre} se midió con {_pares(modelo, medida, distintas)}, pero la "
                          f"configuración final de {modelo} tiene {_pares(modelo, final, distintas)}")
    return modelos, avisos


def _efectivo(modelo, configuracion, clave):
    """El valor con que se arma el modelo: el de la configuración o, si falta, el que usan
    `crear_modelo` o el estimador. Así, pesos_clase ausente y pesos_clase=None no difieren."""
    if clave in configuracion:
        return configuracion[clave]
    if clave in DEFECTOS_PROPIOS:
        return DEFECTOS_PROPIOS[clave]
    return _parametros_por_defecto(modelo).get(clave, _AUSENTE)


@lru_cache(maxsize=None)
def _parametros_por_defecto(modelo):
    return crear_modelo(modelo)[-1].get_params()


def elegir(directorio=DIR_CURVAS, ruta=None):
    """--elegir: D-22 sobre todas las curvas de `directorio`. Escribe el JSON junto al directorio
    (resultados/hiperparametros.json por defecto) y lo devuelve, o None si no hay curvas."""
    directorio = Path(directorio)
    ruta = Path(ruta) if ruta else directorio.parent / JSON_ELEGIDOS
    tablas = leer_curvas(directorio)
    if not tablas:
        print(f"No hay curvas en {_mostrar(directorio)}.")
        return None
    omitidos = leer_omitidos(directorio)
    curvas = {nombre: elegir_punto(tabla) for nombre, tabla in tablas.items()}

    avisos = []
    for nombre, info in curvas.items():
        propios = omitidos.get(nombre, [])
        info["omitidos"] = [f"{etiqueta}: {motivo}" for etiqueta, motivo in propios]
        viejos = [e for e, _ in propios if e.partition("=")[2] in info["puntos"]]
        if viejos:
            avisos.append(f"{nombre}: la última corrida omitió {', '.join(viejos)}, pero el CSV los "
                          f"tiene, así que es de una corrida anterior")
    for nombre in sorted(set(omitidos) - set(curvas)):
        avisos.append(f"{nombre}: la última corrida omitió todos sus puntos y no hay CSV")
    modelos, avisos_modelo = por_modelo(curvas)

    resultado = {"criterio": CRITERIO, "modelos": modelos, "curvas": curvas,
                 "avisos": avisos + avisos_modelo}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    _imprimir_eleccion(resultado, directorio, ruta)
    return resultado


def _imprimir_eleccion(resultado, directorio, ruta):
    print(CRITERIO + ".")
    print(f"{_cantidad(len(resultado['curvas']), 'curva')} en {_mostrar(directorio)}:")
    for nombre, c in resultado["curvas"].items():
        if c["curva_plana"]:
            detalle = (f"plana: {len(c['puntos_dentro_1es'])} puntos dentro de 1 ES "
                       f"({', '.join(c['puntos_dentro_1es'])}), mejor {c['mejor']}")
        else:
            detalle = "un solo punto dentro de 1 ES"
        if not c["adoptada"]:
            detalle += "; no adoptada"
        eleccion = f"{c['parametro']} = {c['punto']}"
        print(f"  {nombre:<28s} {eleccion:<22s} AUC val {_cifra(c['auc_validacion_media'])} ± "
              f"{_cifra(c['error_estandar'])}  train {_cifra(c['auc_train_media'])}  "
              f"brecha {_cifra(c['brecha'], '+.4f')}  ({detalle})")
    print("Por modelo:")
    for modelo, valores in resultado["modelos"].items():
        print(f"  {modelo:<14s} {json.dumps(valores, ensure_ascii=False)}")
    if resultado["avisos"]:
        print("Avisos:")
        for aviso in resultado["avisos"]:
            print(f"  - {aviso}")
    print(f"Escrito {_mostrar(ruta)}.")


def resumir(directorio=DIR_CURVAS, ruta=None):
    """--resumen: concatena las curvas y escribe media, desvío y error estándar por curva, punto,
    conjunto y métrica en <directorio>_resumen.csv, junto al directorio."""
    directorio = Path(directorio)
    ruta = Path(ruta) if ruta else directorio.parent / f"{directorio.name}_resumen.csv"
    tablas = leer_curvas(directorio)
    if not tablas:
        print(f"No hay curvas en {_mostrar(directorio)}.")
        return None
    partes = []
    for nombre, tabla in tablas.items():
        orden = {p: i for i, p in enumerate(dict.fromkeys(tabla["punto"]))}
        r = resumen(tabla, por=["modelo", "parametro", "punto", "conjunto", "metrica"])
        r = r.sort_values(["punto", "conjunto", "metrica"], kind="stable",
                          key=lambda s: s.map(orden) if s.name == "punto" else s)
        partes.append(r.assign(curva=nombre)[["curva", "modelo", "parametro", "punto", "conjunto",
                                               "metrica", "media", "desvio", "n",
                                               "error_estandar"]])
        print(f"{nombre} (AUC):")
        print(resumen_por_punto(tabla).to_string(float_format=lambda x: f"{x:.4f}"))
        print()
    total = pd.concat(partes, ignore_index=True)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(ruta, index=False)
    print(f"Escrito {_mostrar(ruta)}: {_cantidad(len(tablas), 'curva')}, "
          f"{_cantidad(len(total), 'fila')}.")
    return total


def _cantidad(n, palabra):
    return f"{n} {palabra}" + ("" if n == 1 else "s")


def _numero(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _cifra(x, formato=".4f"):
    return "nan" if x is None else format(x, formato)


def _pares(modelo, configuracion, claves):
    return ", ".join(f"{k}={punto_de(_efectivo(modelo, configuracion, k))}" for k in claves)


def _motivo(error):
    """Una línea: el tipo de la excepción y su mensaje, sin tabuladores ni saltos de línea."""
    return " ".join(f"{type(error).__name__}: {error}".split())[:300]


def _mostrar(ruta):
    ruta = Path(ruta).resolve()
    try:
        return ruta.relative_to(RAIZ).as_posix()
    except ValueError:
        return str(ruta)


def _argumentos():
    p = argparse.ArgumentParser(
        prog="python -m src.curvas",
        description="Curvas de validación de la ola 4 (pasos 4.1 a 4.4) y elección de "
                    "hiperparámetros (D-22). Ver el docstring del módulo.",
    )
    p.add_argument("--modelo", choices=MODELOS, help="el clasificador")
    p.add_argument("--parametro", help="el hiperparámetro que recorre la curva")
    p.add_argument("--valores", help="valores separados por comas: int, float, None, True/False o "
                                     "texto. Por defecto, los de GRILLAS en src/modelos.py")
    p.add_argument("--fijo", nargs="+", action="extend", default=[], metavar="CLAVE=VALOR",
                   help="hiperparámetros fijos sobre la configuración vigente")
    p.add_argument("--sufijo", help="se agrega al nombre del CSV: <modelo>_<parametro>_<sufijo>.csv")
    p.add_argument("--n-jobs", type=int,
                   help=f"procesos para los folds (por defecto {N_JOBS}; 1 con --rapido)")
    p.add_argument("--limite", type=float, metavar="SEGUNDOS",
                   help="omite el punto (sus cinco folds) que tarde más; sin límite por defecto")
    p.add_argument("--rapido", action="store_true",
                   help=f"prueba de punta a punta: {FILAS_RAPIDO} filas, un proceso, como máximo "
                        f"{PUNTOS_RAPIDO} valores y RF con hasta {ARBOLES_RAPIDO} árboles. Escribe "
                        f"en {DIR_RAPIDO}")
    p.add_argument("--salida", type=Path,
                   help=f"directorio de las curvas (por defecto {_mostrar(DIR_CURVAS)}). "
                        f"{JSON_ELEGIDOS} y el resumen van en el directorio que lo contiene")
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--elegir", action="store_true",
                      help=f"aplica D-22 a todas las curvas y escribe {JSON_ELEGIDOS}")
    modo.add_argument("--resumen", action="store_true",
                      help="concatena las curvas y escribe <salida>_resumen.csv")
    return p


def main(argv=None):
    parser = _argumentos()
    a = parser.parse_args(argv)
    directorio = a.salida or (DIR_RAPIDO if a.rapido else DIR_CURVAS)
    if a.elegir or a.resumen:
        if a.modelo or a.parametro:
            parser.error("--elegir y --resumen leen todas las curvas: no llevan --modelo ni "
                         "--parametro")
        hecho = elegir(directorio) if a.elegir else resumir(directorio)
        return 0 if hecho is not None else 1
    if not (a.modelo and a.parametro):
        parser.error("faltan --modelo y --parametro (o --elegir, o --resumen)")
    if a.sufijo is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", a.sufijo):
        parser.error(f"--sufijo sólo admite letras, dígitos, '_' y '-': {a.sufijo!r}")
    try:
        valores = parsear_valores(a.valores) if a.valores is not None else None
        ruta = correr_curva(a.modelo, a.parametro, valores, parsear_fijos(a.fijo), a.sufijo,
                            a.n_jobs, a.rapido, directorio, a.limite)
    except CurvaInvalida as error:
        parser.error(str(error))
    return 0 if ruta else 1


if __name__ == "__main__":
    raise SystemExit(main())
