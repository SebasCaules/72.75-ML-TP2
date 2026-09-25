"""Correr con: python -m src.ablaciones --modelo rf     (un proceso por modelo; ver --help)
            python -m src.ablaciones --linea-base
            python -m src.ablaciones --resumen

Pasos 1.2 y 1.4 del plan (ola 1): la línea de base «sin modelo» y las ablaciones A1 a A12. El
enunciado sugiere «comparar el desempeño del modelo con y sin ciertas variables problemáticas»
(TP2, p. 1). Cada observación del EDA es aquí una variante que cambia una sola cosa respecto de la
referencia A0 y se mide en los mismos folds, así que la diferencia de AUC se aparea fold a fold.

- `--modelo M` corre las variantes que aplican a M y escribe resultados/ablaciones_<M>.csv, en
  formato largo con la columna `variante` adelante; si corre A0, también sus puntajes fuera de fold
  en resultados/oof_ablacion_A0_<M>.csv (N0-5). No se llaman oof_referencia_<M>.csv porque ese es
  el nombre que usa src/experimentos.py para la configuración vigente, que no es A0. Un archivo por
  modelo permite correr los cuatro en procesos paralelos (N0-6). El archivo se escribe después de
  cada variante, y con `--variantes` se reemplazan sólo esas: una corrida cortada se completa con
  las que faltan. Sólo se completa un archivo medido sobre las mismas filas, que el OOF de A0
  registra; por eso la primera corrida de un modelo tiene que incluir A0.
- `--linea-base` escribe resultados/linea_base.csv: un DummyClassifier que les da a todos el mismo
  puntaje, en los mismos folds. Con todos empatados, AUC 0,5 y recall en el presupuesto k/n.
- `--resumen` junta los archivos por modelo en resultados/ablaciones.csv y escribe
  resultados/ablaciones_resumen.csv: una fila por variante distinta de A0, con ΔAUC y Δrecall de
  validación pareados contra A0, cuántas columnas produce y su efecto según el criterio mecánico
  del paso 1.5 (mejora, empeora o neutra). La adopción (D-08 a D-17) no se decide aquí, porque
  exige juicio («neutra y simplifica»). No junta modelos medidos sobre filas distintas.

A0 es la configuración de referencia (`Opciones()` y `REFERENCIA`), no la vigente de
src/configuracion.py: las ablaciones son las que deciden la vigente, y medirlas contra ella movería
la base con cada decisión adoptada.

`--rapido` prueba todo de punta a punta en menos de un minuto: N_RAPIDO filas estratificadas, un
proceso y RF con menos árboles. Sin `--salida` escribe en una carpeta temporal fija (así
`--resumen --rapido` encuentra lo que escribieron las corridas rápidas) y no acepta una carpeta
dentro de resultados/: sus números son de una submuestra y no deben mezclarse con los reales.
"""

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from src.datos import FILA, OBJETIVO, RAIZ, SEMILLA, cargar_train
from src.modelos import REFERENCIA, crear_modelo, separar_X_y
from src.preproceso import Opciones
from src.resultados import (
    CAMPOS,
    DIR,
    N_JOBS,
    con_identidad,
    configuracion_json,
    diferencias_pareadas,
    escribir_largo,
    escribir_oof,
    leer_largo,
    resumen,
    validacion_cruzada,
)
from src.validacion import K, folds

# Los cuatro clasificadores del enunciado. La variante categórica de Naive Bayes se decide en D-18.
MODELOS_ABLACION = ("nb_gaussiano", "svm", "knn", "rf")
BASE = "A0"


@dataclass(frozen=True)
class Variante:
    descripcion: str
    opciones: Opciones
    modelos: tuple = MODELOS_ABLACION


# La tabla «Las ablaciones del paso 1.4» de plan/PLAN.md. Cada variante cambia un solo campo de
# Opciones respecto de A0, salvo A8, que es A7 sin `month`: mide el bloque entero que codifica la
# época.
VARIANTES = {
    "A0": Variante("Referencia: Opciones() y los hiperparámetros de REFERENCIA", Opciones()),
    "A1": Variante("Agrega duration, como techo: no se conoce antes de llamar (D-05)",
                   Opciones(duration=True)),
    "A2": Variante("Saca pdays: el contacto previo (previous > 0) ya está en poutcome (D-10)",
                   Opciones(pdays="sin")),
    "A3": Variante("default como indicadora de unknown contra el resto (D-09)",
                   Opciones(default="indicadora")),
    "A4": Variante("Funde las categorías raras: illiterate y marital unknown (D-11)",
                   Opciones(raras=True)),
    "A5": Variante("log1p sobre campaign (D-13)", Opciones(campaign_log=True), ("svm", "knn")),
    "A6": Variante("Reduce el bloque macro a euribor3m y nr.employed (D-14)",
                   Opciones(macro="reducido")),
    "A7": Variante("Sin el bloque macro; no se adopta, alimenta el hallazgo (D-14)",
                   Opciones(macro="sin")),
    "A8": Variante("Sin el bloque macro ni month, que también codifica la época (D-14, D-15)",
                   Opciones(macro="sin", month=False)),
    "A9": Variante("Sin day_of_week (D-15)", Opciones(day_of_week=False)),
    "A10": Variante("Edad en tramos (D-16)", Opciones(edad_tramos=True), ("nb_gaussiano",)),
    "A11": Variante("unknown imputado por la moda, como contraste de D-08",
                    Opciones(unknown="moda")),
    "A12": Variante("Sin escalar (D-12)", Opciones(escalar=False), ("knn",)),
}

MODELO_BASE = "sin_modelo"
CONFIGURACION_BASE = {"estrategia": "prior"}

N_RAPIDO = 3000
HIPERPARAMETROS_RAPIDO = {"rf": {"n_estimators": 20}}
SALIDA_RAPIDO = Path(tempfile.gettempdir()) / "tp2-ablaciones-rapido"
FILAS_COLUMNAS = 200

COLUMNAS_DELTA = ["delta_media", "delta_desvio", "delta_error_estandar", "delta_min", "delta_max"]
COLUMNAS_RESUMEN = ["modelo", "variante", "auc_referencia", "auc_variante", *COLUMNAS_DELTA,
                    "n_folds", "delta_recall_media", "delta_recall_desvio", "columnas", "efecto"]

_POSICION_MODELO = {m: i for i, m in enumerate(MODELOS_ABLACION)}
_POSICION_VARIANTE = {v: i for i, v in enumerate(VARIANTES)}


def variantes_de(modelo):
    """Las variantes que se miden en `modelo`, en el orden de VARIANTES."""
    return [nombre for nombre, v in VARIANTES.items() if modelo in v.modelos]


def hiperparametros(modelo, rapido=False):
    return {**REFERENCIA[modelo], **(HIPERPARAMETROS_RAPIDO.get(modelo, {}) if rapido else {})}


def salida_por_defecto(rapido):
    return SALIDA_RAPIDO if rapido else DIR


def ruta_oof(salida, modelo):
    """Los puntajes fuera de fold de A0 (N0-5), con la etiqueta `ablacion_A0`."""
    return Path(salida) / f"oof_ablacion_{BASE}_{modelo}.csv"


def filas_del_oof(ruta):
    """Las filas del CSV original sobre las que se midió A0, o None si su OOF no está."""
    return frozenset(pd.read_csv(ruta, usecols=[FILA])[FILA]) if ruta.exists() else None


def submuestra(train, n=N_RAPIDO):
    """Muestra estratificada por `y` con la semilla 42, en el orden de la fila original."""
    muestra, _ = train_test_split(train, train_size=n, stratify=train[OBJETIVO],
                                  random_state=SEMILLA)
    return muestra.sort_values(FILA).reset_index(drop=True)


def leer_datos(rapido=False):
    train = cargar_train()
    if rapido:
        train = submuestra(train)
    X, y = separar_X_y(train)
    return X, y, train[FILA].to_numpy()


def linea_base(X, y, filas, n_jobs=N_JOBS):
    """La línea de base «sin modelo»: puntúa a todos con la proporción de «yes» del fold de
    entrenamiento. Con todos empatados, el AUC es 0,5 y el recall en el presupuesto, k/n
    (src/metricas.py)."""
    clasificador = DummyClassifier(strategy=CONFIGURACION_BASE["estrategia"])
    tabla, _ = validacion_cruzada(clasificador, X, y, filas=filas, n_jobs=n_jobs)
    return con_identidad(tabla, MODELO_BASE, CONFIGURACION_BASE, variante=BASE)


def medir_variante(modelo, nombre, X, y, filas, rapido=False, n_jobs=N_JOBS):
    """Una variante en validación cruzada: la tabla larga con su identidad, los puntajes fuera de
    fold y los segundos que tardó."""
    h = hiperparametros(modelo, rapido)
    inicio = time.perf_counter()
    tabla, oof = validacion_cruzada(crear_modelo(modelo, VARIANTES[nombre].opciones, **h),
                                    X, y, filas=filas, n_jobs=n_jobs)
    return con_identidad(tabla, modelo, h, variante=nombre), oof, time.perf_counter() - inicio


def ordenar(tabla):
    """Por modelo y por variante en el orden de VARIANTES (A2 antes que A10), sin cambiar el orden
    de las filas dentro de cada variante."""
    n = len(VARIANTES) + 1
    clave = (tabla["modelo"].map(_POSICION_MODELO).fillna(len(MODELOS_ABLACION)).to_numpy() * n
             + tabla["variante"].map(_POSICION_VARIANTE).fillna(n - 1).to_numpy())
    return tabla.iloc[np.argsort(clave, kind="stable")].reset_index(drop=True)


def reemplazar_variantes(anterior, nueva):
    """El archivo de un modelo después de volver a correr algunas variantes: las filas nuevas
    reemplazan a las de esas variantes y el resto se conserva."""
    conservadas = anterior[~anterior["variante"].isin(set(nueva["variante"]))]
    tabla = ordenar(pd.concat([conservadas, nueva], ignore_index=True))
    if (tabla.groupby("modelo")["configuracion"].nunique() > 1).any():
        raise ValueError("las variantes nuevas tienen otros hiperparámetros que las guardadas: "
                         "hay que volver a correr todas las variantes del modelo")
    return tabla


def comprobar(tabla):
    """Lo que el resumen necesita de la tabla larga. Si algo falta, el error dice qué."""
    faltan = [c for c in ["variante", *CAMPOS] if c not in tabla.columns]
    if faltan:
        raise ValueError(f"faltan columnas: {', '.join(faltan)}")
    desconocidas = sorted(set(tabla["variante"]) - set(VARIANTES))
    if desconocidas:
        raise ValueError(f"variantes desconocidas: {', '.join(desconocidas)}")
    clave = ["modelo", "variante", "fold", "conjunto", "metrica"]
    if tabla.duplicated(clave).any():
        raise ValueError(f"hay filas repetidas para una misma combinación de {', '.join(clave)}")
    sin_base = sorted(set(tabla["modelo"]) - set(tabla.loc[tabla["variante"] == BASE, "modelo"]))
    if sin_base:
        raise ValueError(f"falta la referencia {BASE} de: {', '.join(sin_base)}")
    configuraciones = tabla.groupby("modelo")["configuracion"].nunique()
    if (configuraciones > 1).any():
        raise ValueError("hay más de una configuración de hiperparámetros en: "
                         + ", ".join(configuraciones[configuraciones > 1].index))
    folds_de = {grupo: set(f) for grupo, f in tabla.groupby(["modelo", "variante"])["fold"]}
    distintos = [f"{m} {v}" for (m, v), f in folds_de.items() if f != folds_de[(m, BASE)]]
    if distintos:
        raise ValueError(f"folds distintos de los de {BASE} en: {', '.join(distintos)}")


def efecto(delta_media, delta_desvio):
    """El criterio mecánico del paso 1.5: la variante mejora o empeora sólo si la diferencia media
    supera el desvío de las diferencias pareadas; si no, es neutra."""
    if delta_media > delta_desvio:
        return "mejora"
    if delta_media < -delta_desvio:
        return "empeora"
    return "neutra"


def _deltas(tabla, metrica):
    """Las diferencias pareadas de validación contra A0, de src/resultados.py."""
    d = diferencias_pareadas(tabla, "variante", BASE, metrica, "validacion")
    return d.reindex(columns=["modelo", "variante", "n_folds", *COLUMNAS_DELTA])


def resumir(tabla):
    """Una fila por modelo y variante distinta de A0: el AUC medio de validación de A0 y de la
    variante, ΔAUC y Δrecall pareados contra A0 y el efecto. A0 no lleva fila propia: su AUC es
    `auc_referencia` y su diferencia consigo misma es cero. Todo sale de la tabla larga; la columna
    `columnas` la agrega `con_columnas`."""
    comprobar(tabla)
    auc = tabla[(tabla["conjunto"] == "validacion") & (tabla["metrica"] == "auc")]
    medias = auc.groupby(["modelo", "variante"])["valor"].mean()
    recall = (_deltas(tabla, "recall_q")[["modelo", "variante", "delta_media", "delta_desvio"]]
              .rename(columns={"delta_media": "delta_recall_media",
                               "delta_desvio": "delta_recall_desvio"}))
    r = _deltas(tabla, "auc").merge(recall, on=["modelo", "variante"], how="left")
    r["auc_referencia"] = [medias[(m, BASE)] for m in r["modelo"]]
    r["auc_variante"] = [medias[(m, v)] for m, v in zip(r["modelo"], r["variante"])]
    r["efecto"] = [efecto(m, d) for m, d in zip(r["delta_media"], r["delta_desvio"])]
    return ordenar(r)[[c for c in COLUMNAS_RESUMEN if c != "columnas"]]


def columnas_salida(modelo, opciones, X):
    """Cuántas columnas le llegan al clasificador: el Pipeline sin ajustar, sin su último paso,
    ajustado sobre FILAS_COLUMNAS filas. No depende de cuáles: los niveles de cada categórica
    están fijos en VOCABULARIO."""
    muestra = X.sample(min(FILAS_COLUMNAS, len(X)), random_state=SEMILLA)
    return crear_modelo(modelo, opciones)[:-1].fit_transform(muestra).shape[1]


def con_columnas(r, X):
    cuantas = [columnas_salida(m, VARIANTES[v].opciones, X)
               for m, v in zip(r["modelo"], r["variante"])]
    return r.assign(columnas=cuantas)[COLUMNAS_RESUMEN]


def _mostrar(ruta):
    try:
        return ruta.resolve().relative_to(RAIZ)
    except ValueError:
        return ruta


def _dentro(ruta, carpeta):
    ruta, carpeta = Path(ruta).resolve(), Path(carpeta).resolve()
    return ruta == carpeta or carpeta in ruta.parents


def _procesos(n_jobs):
    return f"{n_jobs} proceso" + ("" if n_jobs == 1 else "s")


def _miles(n):
    """El separador de miles es un espacio: la coma es la separación decimal en español, y el
    punto se confundiría con el decimal de las métricas que se imprimen al lado."""
    return f"{n:,}".replace(",", " ")


def correr_modelo(modelo, variantes, parcial, rapido, n_jobs, salida):
    """Escribe el archivo del modelo después de cada variante: si la corrida se corta, lo medido
    queda, y se completa con `--variantes` y las que faltan. Sólo se completa un archivo medido con
    los mismos hiperparámetros y sobre las mismas filas, las que registra el OOF de A0."""
    if not variantes:
        raise ValueError("no hay variantes para medir")
    ruta, ruta_a0 = salida / f"ablaciones_{modelo}.csv", ruta_oof(salida, modelo)
    anterior = leer_largo(ruta) if parcial and ruta.exists() else None
    filas_antes = None
    if parcial and anterior is None and BASE not in variantes:
        raise SystemExit(f"No hay {_mostrar(ruta)}: la primera corrida de {modelo} tiene que "
                         f"incluir {BASE}, contra la que se comparan las demás variantes")
    if anterior is not None:
        vigente = configuracion_json(hiperparametros(modelo, rapido))
        otras = set(anterior["configuracion"]) - {vigente}
        if otras:
            raise SystemExit(f"{_mostrar(ruta)} se midió con otros hiperparámetros "
                             f"({', '.join(sorted(otras))}): hay que correr todas las variantes "
                             f"de {modelo}, sin --variantes")
        filas_antes = filas_del_oof(ruta_a0)
        if filas_antes is None:
            raise SystemExit(f"Falta {_mostrar(ruta_a0)}, que registra sobre qué filas se midió "
                             f"{_mostrar(ruta)}: hay que correr todas las variantes de {modelo}, "
                             "sin --variantes")
    X, y, filas = leer_datos(rapido)
    if filas_antes is not None and filas_antes != set(filas):
        raise SystemExit(f"{_mostrar(ruta)} se midió sobre otras filas ({_miles(len(filas_antes))} "
                         f"según {_mostrar(ruta_a0)}) que las de esta corrida "
                         f"({_miles(len(filas))}{', con --rapido' if rapido else ''}): no se "
                         f"mezclan en un archivo. Hay que correr todas las variantes de {modelo}, "
                         "sin --variantes, o escribir en otra --salida")
    print(f"Ablaciones de {modelo}{' (rápido)' if rapido else ''}: {len(variantes)} variantes, "
          f"{_miles(len(X))} filas, {K} folds, {_procesos(n_jobs)}", flush=True)
    inicio = time.perf_counter()
    medidas = []
    for nombre in variantes:
        tabla, oof, segundos = medir_variante(modelo, nombre, X, y, filas, rapido, n_jobs)
        medidas.append(tabla)
        auc = tabla.loc[(tabla["conjunto"] == "validacion") & (tabla["metrica"] == "auc"), "valor"]
        print(f"  {nombre:<4} AUC de validación {auc.mean():.4f} ± {auc.std():.4f}"
              f"  {segundos:7.1f} s   {VARIANTES[nombre].descripcion}", flush=True)
        archivo = pd.concat(medidas, ignore_index=True)
        if anterior is not None:
            archivo = reemplazar_variantes(anterior, archivo)
        if nombre == BASE:
            # Si la corrida se corta entre las dos escrituras, el archivo queda sin OOF, y
            # --variantes se niega a completarlo, en lugar de quedar con el OOF de otras filas.
            ruta_a0.unlink(missing_ok=True)
        escribir_largo(ruta, archivo)
        if nombre == BASE:
            escribir_oof(ruta_a0, oof)
    print(f"Total: {time.perf_counter() - inicio:.1f} s. Variantes en el archivo: "
          f"{', '.join(archivo['variante'].unique())} -> {_mostrar(ruta)}")
    if BASE in variantes:
        print(f"Puntajes fuera de fold de {BASE} -> {_mostrar(ruta_a0)}")


def correr_linea_base(rapido, n_jobs, salida):
    X, y, filas = leer_datos(rapido)
    print(f"Línea de base «{MODELO_BASE}» (DummyClassifier, {CONFIGURACION_BASE}"
          f"{', rápido' if rapido else ''}): {_miles(len(X))} filas, {K} folds, "
          f"{_procesos(n_jobs)}")
    inicio = time.perf_counter()
    tabla = linea_base(X, y, filas, n_jobs)
    segundos = time.perf_counter() - inicio
    ruta = salida / "linea_base.csv"
    escribir_largo(ruta, tabla)
    r = resumen(tabla, por=["conjunto", "metrica"])
    for fila in r[r["conjunto"] == "validacion"].itertuples():
        print(f"  {fila.metrica:<12} {fila.media:.4f} ± {fila.desvio:.4f}")
    exactitud = np.mean([1 - y[va].mean() for _, va in folds().split(X, y)])
    print(f"  Exactitud de decir siempre «no» en validación: {exactitud:.4f}")
    print(f"{segundos:.1f} s -> {_mostrar(ruta)}")


def correr_resumen(salida):
    rutas = {m: salida / f"ablaciones_{m}.csv" for m in MODELOS_ABLACION}
    presentes = [m for m, ruta in rutas.items() if ruta.exists()]
    if not presentes:
        raise SystemExit(f"No hay ningún ablaciones_<modelo>.csv en {salida}")
    filas_de = {m: filas_del_oof(ruta_oof(salida, m)) for m in presentes}
    registradas = {m: f for m, f in filas_de.items() if f is not None}
    if len(set(registradas.values())) > 1:
        detalle = ", ".join(f"{m} {_miles(len(f))}" for m, f in registradas.items())
        raise SystemExit(f"Los modelos de {salida} se midieron sobre filas distintas (según el OOF "
                         f"de {BASE}: {detalle}), así que el resumen no los junta. ¿Hay corridas "
                         "con y sin --rapido en la misma carpeta?")
    tabla = ordenar(pd.concat([leer_largo(rutas[m]) for m in presentes], ignore_index=True))
    try:
        r = resumir(tabla)
    except ValueError as e:
        raise SystemExit(f"No se puede resumir {salida}: {e}") from None
    X, _, _ = leer_datos()
    r = con_columnas(r, X)
    escribir_largo(salida / "ablaciones.csv", tabla)
    ruta = salida / "ablaciones_resumen.csv"
    r.to_csv(ruta, index=False)

    formatos = {c: "{:.4f}".format for c in ("auc_referencia", "auc_variante", "delta_desvio",
                                             "delta_recall_desvio")}
    formatos |= {c: "{:+.4f}".format for c in ("delta_media", "delta_min", "delta_max",
                                               "delta_recall_media")}
    visibles = ["modelo", "variante", "auc_referencia", "auc_variante", "delta_media",
                "delta_desvio", "delta_min", "delta_max", "delta_recall_media",
                "delta_recall_desvio", "columnas", "efecto"]
    if len(r):
        print(r[visibles].to_string(index=False, formatters=formatos))
    else:
        print(f"Ninguna variante medida además de {BASE}")
    print()
    print(f"Columnas de {BASE}, la referencia: " + ", ".join(
        f"{m} {columnas_salida(m, VARIANTES[BASE].opciones, X)}" for m in presentes))
    if registradas:
        print(f"Filas medidas según el OOF de {BASE}: "
              f"{_miles(len(next(iter(registradas.values()))))} en {', '.join(registradas)}")
    sin_oof = [m for m in presentes if m not in registradas]
    if sin_oof:
        print(f"Aviso: sin OOF de {BASE} en {', '.join(sin_oof)}; no se puede comprobar sobre qué "
              "filas se midieron")
    ausentes = [m for m in MODELOS_ABLACION if m not in presentes]
    if ausentes:
        print(f"Aviso: sin archivo de {', '.join(ausentes)}")
    for m in presentes:
        medidas = set(tabla.loc[tabla["modelo"] == m, "variante"])
        faltan = [v for v in variantes_de(m) if v not in medidas]
        if faltan:
            print(f"Aviso: en {m} faltan {', '.join(faltan)}")
    print(f"{_miles(len(tabla))} filas de {len(presentes)} modelos -> "
          f"{_mostrar(salida / 'ablaciones.csv')}")
    print(f"Resumen -> {_mostrar(ruta)}")


def _parser():
    todos = set(MODELOS_ABLACION)
    lista = "\n".join(
        f"  {nombre:<4} {'todos' if set(v.modelos) == todos else ', '.join(v.modelos):<14}"
        f" {v.descripcion}" for nombre, v in VARIANTES.items())
    p = argparse.ArgumentParser(
        prog="python -m src.ablaciones",
        description="Línea de base «sin modelo» y ablaciones A1 a A12 de la ola 1, en validación "
                    "cruzada sobre train.",
        epilog="Variantes y modelos donde se miden:\n" + lista,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    accion = p.add_mutually_exclusive_group(required=True)
    accion.add_argument("--modelo", choices=MODELOS_ABLACION,
                        help="corre las variantes que aplican a ese modelo y escribe "
                             f"ablaciones_<modelo>.csv (y {ruta_oof('', '<modelo>').name} con "
                             f"{BASE})")
    accion.add_argument("--linea-base", action="store_true",
                        help="escribe linea_base.csv con el DummyClassifier")
    accion.add_argument("--resumen", action="store_true",
                        help="junta los ablaciones_<modelo>.csv en ablaciones.csv y escribe "
                             "ablaciones_resumen.csv")
    p.add_argument("--variantes",
                   help="con --modelo: lista separada por comas, por ejemplo A0,A2,A7. Por "
                        "defecto, todas las que aplican; si se da, reemplaza sólo esas en el "
                        "archivo del modelo y conserva las demás, si se midieron sobre las mismas "
                        f"filas. La primera corrida de un modelo tiene que incluir {BASE}")
    p.add_argument("--n-jobs", type=int,
                   help=f"procesos en paralelo, uno por fold (por defecto {N_JOBS}; "
                        "1 con --rapido)")
    p.add_argument("--rapido", action="store_true",
                   help=f"prueba de punta a punta: {_miles(N_RAPIDO)} filas estratificadas, un "
                        f"proceso, RF con {HIPERPARAMETROS_RAPIDO['rf']['n_estimators']} árboles")
    p.add_argument("--salida", type=Path,
                   help=f"carpeta de salida (por defecto {_mostrar(DIR)}/; con --rapido, "
                        f"{SALIDA_RAPIDO}, y nunca dentro de {_mostrar(DIR)}/)")
    return p


def _variantes_pedidas(parser, args):
    aplican = variantes_de(args.modelo)
    if args.variantes is None:
        return aplican, False
    pedidas = [v.strip() for v in args.variantes.split(",") if v.strip()]
    if not pedidas:
        parser.error("--variantes está vacía")
    desconocidas = [v for v in pedidas if v not in VARIANTES]
    if desconocidas:
        parser.error(f"variantes desconocidas: {', '.join(desconocidas)}; "
                     f"las válidas son {', '.join(VARIANTES)}")
    ajenas = [v for v in pedidas if v not in aplican]
    if ajenas:
        parser.error(f"{args.modelo} no lleva {', '.join(ajenas)}; "
                     f"sus variantes son {', '.join(aplican)}")
    return [v for v in aplican if v in pedidas], True


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.variantes is not None and args.modelo is None:
        parser.error("--variantes sólo va con --modelo")
    if args.n_jobs is not None and args.n_jobs < 1:
        parser.error("--n-jobs tiene que ser al menos 1")
    salida = args.salida if args.salida is not None else salida_por_defecto(args.rapido)
    if args.rapido and not args.resumen and _dentro(salida, DIR):
        parser.error("--rapido no escribe dentro de resultados/: sus números son de una submuestra")
    n_jobs = args.n_jobs if args.n_jobs is not None else (1 if args.rapido else N_JOBS)
    if args.resumen:
        correr_resumen(salida)
    elif args.linea_base:
        correr_linea_base(args.rapido, n_jobs, salida)
    else:
        variantes, parcial = _variantes_pedidas(parser, args)
        correr_modelo(args.modelo, variantes, parcial, args.rapido, n_jobs, salida)


if __name__ == "__main__":
    main()
