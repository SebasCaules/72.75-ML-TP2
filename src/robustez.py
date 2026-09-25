"""Correr con: python -m src.robustez --modelo rf   (un proceso por modelo; al final, --resumen)

Paso 5.2 del plan, D-25: la robustez temporal, medida dentro de train y sin abrir el test.

El CSV original está ordenado por fecha, de mayo de 2008 a noviembre de 2010, y train conserva la
posición de cada fila en la columna `fila`. Los folds barajados de `folds()` mezclan todas las
épocas: miden qué tan bien se predice a un cliente de una época que el modelo ya vio. Este módulo
mide además hacia adelante: entrena con el pasado y valida con lo que viene después.

Esquemas:
- "hacia_adelante": train ordenado por `fila` y `TimeSeriesSplit(n_splits=5)`, que lo parte en seis
  bloques consecutivos. El fold k valida con el bloque k + 1 y entrena con los k anteriores. Todas
  las validaciones tienen n // 6 filas (división entera), pero los entrenamientos crecen: el primero
  usa cerca de una sexta parte de train (el primer bloque se queda con el resto de la división) y
  el último, cerca de cinco sextas partes; el primer bloque nunca se valida.
  La media sobre los folds mezcla, entonces, modelos entrenados con cantidades de datos muy
  distintas (el barajado entrena siempre con cuatro quintos), y parte de la brecha entre los dos
  esquemas puede deberse a eso y no a la época. Por eso el tamaño, la tasa de «yes» y el rango de
  filas de cada fold se guardan en robustez_folds.csv, y el AUC se imprime también fold por fold.
- "barajado": los folds de `folds()`, los mismos del resto de los módulos.

Variantes de columnas, sobre OPCIONES_FINALES y con los hiperparámetros de HIPERPARAMETROS_FINALES:
- "todas": OPCIONES_FINALES tal cual.
- "sin_macro": sin el bloque macro, que describe la economía de cada mes (D-14).
- "sin_macro_ni_month": además sin `month`, que también codifica la época (A8).

Si el AUC barajado supera al de hacia adelante y la brecha se reduce al sacar las variables de la
época, el modelo aprendió la época (hallazgo H1). Hay un matiz en el fold 1 hacia adelante: con
train completo, ese fold entrena sólo con mayo de 2008, así que `month` tiene un único nivel,
cuatro de las cinco columnas macro son constantes y euribor3m casi. El escalado de la SVM y de KNN
deja una columna constante con escala 1 y una casi constante con un desvío mínimo, y la validación
de ese fold, que llega hasta julio, cae muy lejos de lo que se vio al entrenar: con «todas», el
modelo extrapola. Esa extrapolación viene del bloque macro, que sólo está en «todas», así que no
se cancela al comparar variantes: baja el AUC de «todas» por una razón que no es aprender la
época. Por eso H1 se lee también en los folds 2 a 5, que la CLI imprime uno por uno.

El recall en el presupuesto se compara con cuidado: en un bloque cuya tasa de «yes» p supera al
presupuesto q, llamar al q % de la lista alcanza como mucho q / p de los «yes». El AUC no depende
de la tasa.

Archivos, en --salida (por defecto resultados/):
- robustez_<modelo>.csv: el formato largo del contrato (src/resultados.py), con `esquema` y
  `variante` primero. Una corrida con los dos esquemas y las tres variantes lo reemplaza entero.
  Con --esquemas o --variantes, reemplaza sólo los pares (esquema, variante) que mide y conserva
  los demás, siempre que el archivo se haya medido con los mismos hiperparámetros (si no, se niega
  antes de medir). El archivo se relee justo antes de escribirlo, así que dos procesos del mismo
  modelo con pares distintos (la SVM partida por esquema, por ejemplo) no se pisan, salvo que
  terminen en el mismo instante.
- robustez_folds.csv: una fila por esquema y fold con n_train, n_validacion, pct_yes_train,
  pct_yes_validacion (en %), fila_min_validacion, fila_max_validacion y fila_max_train. Depende de
  los datos y no del modelo: cada corrida lo reemplaza entero, con los dos esquemas.
- Con --resumen: robustez_temporal.csv, que concatena los archivos por modelo, y
  robustez_temporal_resumen.csv, con la media, el desvío, n y el error estándar del AUC y del
  recall_q de validación por modelo, esquema, variante y métrica. En las filas con esquema
  "barajado_menos_hacia_adelante", `media` es la diferencia entre las medias del AUC de los dos
  esquemas, por modelo y variante. --resumen no lleva --esquemas, --variantes ni --n-jobs.

--rapido lo prueba de punta a punta en menos de un minuto: 3 000 filas de train estratificadas, un
solo proceso y RF con 20 árboles. Sin --salida escribe en SALIDA_RAPIDO (tp2-robustez-rapido, en el
directorio temporal del sistema), el mismo que lee `--resumen --rapido`. Nunca escribe ni resume
dentro de resultados/, porque sus cifras son de una submuestra.
"""

import argparse
import os
import tempfile
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES, OPCIONES_FINALES
from src.datos import FILA, OBJETIVO, RAIZ, SEMILLA, cargar_train
from src.metricas import METRICAS, PRESUPUESTO
from src.modelos import MODELOS, crear_modelo, separar_X_y
from src.resultados import (
    DIR,
    N_JOBS,
    con_identidad,
    configuracion_json,
    escribir_largo,
    leer_largo,
    resumen,
    validacion_cruzada,
)
from src.validacion import K, folds

ESQUEMAS = ("hacia_adelante", "barajado")
VARIANTES = ("todas", "sin_macro", "sin_macro_ni_month")
CAMBIOS = {
    "todas": {},
    "sin_macro": {"macro": "sin"},
    "sin_macro_ni_month": {"macro": "sin", "month": False},
}
DIFERENCIA = "barajado_menos_hacia_adelante"
METRICAS_RESUMEN = ("auc", "recall_q")
# Los pares (esquema, variante) en el orden en que una corrida completa los escribe.
PARES = tuple(product(ESQUEMAS, VARIANTES))

N_RAPIDO = 3000
ARBOLES_RAPIDO = 20
SALIDA_RAPIDO = Path(tempfile.gettempdir()) / "tp2-robustez-rapido"

ARCHIVO_FOLDS = "robustez_folds.csv"
ARCHIVO_TEMPORAL = "robustez_temporal.csv"
ARCHIVO_RESUMEN = "robustez_temporal_resumen.csv"
CAMPOS_FOLDS = ["esquema", "fold", "n_train", "n_validacion", "pct_yes_train",
                "pct_yes_validacion", "fila_min_validacion", "fila_max_validacion",
                "fila_max_train"]
CAMPOS_RESUMEN = ["modelo", "esquema", "variante", "metrica", "media", "desvio", "n",
                  "error_estandar"]

ORDEN = {"modelo": MODELOS, "esquema": ESQUEMAS + (DIFERENCIA,), "variante": VARIANTES,
         "metrica": METRICAS}
ROTULOS = {"auc": "AUC", "recall_q": f"Recall llamando al {PRESUPUESTO:.0%} de la lista"}


def archivo_modelo(modelo):
    return f"robustez_{modelo}.csv"


def opciones_de(variante):
    return replace(OPCIONES_FINALES, **CAMBIOS[variante])


def particionador(esquema):
    if esquema == "hacia_adelante":
        return TimeSeriesSplit(n_splits=K)
    if esquema == "barajado":
        return folds()
    raise ValueError(f"esquema desconocido: {esquema}; los válidos son {ESQUEMAS}")


def ordenar(train):
    """Las filas en orden temporal, porque `TimeSeriesSplit` parte por posición. train.csv ya está
    escrito en ese orden (src/datos.py, `partir`), así que el esquema barajado usa los mismos folds
    que el resto de los módulos."""
    return train.sort_values(FILA, kind="stable").reset_index(drop=True)


def submuestra(train, n=N_RAPIDO):
    parte, _ = train_test_split(train, train_size=n, stratify=train[OBJETIVO],
                                random_state=SEMILLA)
    return parte


def hiperparametros_de(modelo, rapido=False):
    h = dict(HIPERPARAMETROS_FINALES[modelo])
    if rapido and modelo == "rf":
        h["n_estimators"] = min(h.get("n_estimators", ARBOLES_RAPIDO), ARBOLES_RAPIDO)
    return h


def composicion(train, esquemas=ESQUEMAS):
    """Por esquema y fold: los tamaños, la tasa de «yes» en % y el rango de filas del CSV original.
    `train` tiene que venir de `ordenar()`."""
    X, y = separar_X_y(train)
    filas = train[FILA].to_numpy()
    registros = []
    for esquema in esquemas:
        for numero, (tr, va) in enumerate(particionador(esquema).split(X, y), start=1):
            registros.append({
                "esquema": esquema, "fold": numero, "n_train": len(tr), "n_validacion": len(va),
                "pct_yes_train": 100 * y[tr].sum() / len(tr),
                "pct_yes_validacion": 100 * y[va].sum() / len(va),
                "fila_min_validacion": filas[va].min(), "fila_max_validacion": filas[va].max(),
                "fila_max_train": filas[tr].max(),
            })
    return pd.DataFrame(registros, columns=CAMPOS_FOLDS)


def _reemplazar(ruta, escribir):
    """Escribe en un temporal y lo renombra, para que ningún proceso lea un archivo a medias: los
    procesos de todos los modelos escriben robustez_folds.csv, y una corrida parcial relee el
    archivo del modelo que otra puede estar escribiendo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(f".{ruta.name}.{os.getpid()}")
    escribir(temporal)
    os.replace(temporal, ruta)


def escribir_folds(ruta, por_fold):
    _reemplazar(ruta, lambda destino: por_fold.to_csv(destino, index=False))


def escribir_modelo(ruta, tabla):
    _reemplazar(ruta, lambda destino: escribir_largo(destino, tabla))


def medir(modelo, train, esquemas=ESQUEMAS, variantes=VARIANTES, n_jobs=N_JOBS, rapido=False):
    """La validación cruzada de `modelo` en cada esquema y variante, en el formato largo del
    contrato con `esquema` y `variante` primero. `train` tiene que venir de `ordenar()`."""
    hiperparametros = hiperparametros_de(modelo, rapido)
    X, y = separar_X_y(train)
    tablas = []
    for esquema in esquemas:
        for variante in variantes:
            inicio = time.perf_counter()
            tabla, _ = validacion_cruzada(
                crear_modelo(modelo, opciones_de(variante), **hiperparametros), X, y,
                filas=train[FILA], n_jobs=n_jobs, particionador=particionador(esquema))
            auc = tabla.loc[(tabla["conjunto"] == "validacion") & (tabla["metrica"] == "auc"),
                            "valor"]
            print(f"{modelo:13s} {esquema:15s} {variante:19s} AUC de validación "
                  f"{auc.mean():.3f} ± {auc.std():.3f}   ({time.perf_counter() - inicio:.1f} s)",
                  flush=True)
            tablas.append(con_identidad(tabla, modelo, hiperparametros, esquema=esquema,
                                        variante=variante))
    return pd.concat(tablas, ignore_index=True)


def es_parcial(esquemas, variantes):
    """True si la corrida no mide los seis pares: en ese caso se combina con el archivo anterior
    en lugar de reemplazarlo."""
    return list(esquemas) != list(ESQUEMAS) or list(variantes) != list(VARIANTES)


def _pares_de(tabla):
    return pd.MultiIndex.from_frame(tabla[["esquema", "variante"]])


def combinar(anterior, nueva):
    """El archivo de un modelo después de una corrida parcial: las filas de `nueva` reemplazan a
    las de sus pares (esquema, variante) en `anterior`, las demás se conservan y todo queda en el
    orden de PARES, el mismo de una corrida completa."""
    ajenas = set(anterior["configuracion"]) - set(nueva["configuracion"])
    if ajenas:
        raise ValueError("el archivo anterior se midió con otros hiperparámetros: "
                         f"{'; '.join(sorted(ajenas))}")
    conservadas = anterior[~_pares_de(anterior).isin(_pares_de(nueva))]
    tabla = pd.concat([conservadas, nueva], ignore_index=True)
    posicion = {par: i for i, par in enumerate(PARES)}
    orden = np.argsort([posicion[par] for par in _pares_de(tabla)], kind="stable")
    return tabla.iloc[orden].reset_index(drop=True)


def pares_sin_medir(tabla):
    """Por modelo, los pares (esquema, variante) que su tabla no tiene, en el orden de PARES."""
    faltan = {}
    for modelo, grupo in tabla.groupby("modelo", sort=False):
        medidos = set(_pares_de(grupo))
        if sin := [par for par in PARES if par not in medidos]:
            faltan[modelo] = sin
    return faltan


def _nombrar(pares):
    return ", ".join(f"{esquema} × {variante}" for esquema, variante in pares)


def _ordenar_filas(df):
    """En el orden de MODELOS, ESQUEMAS, VARIANTES y METRICAS, no en el alfabético."""
    claves = [c for c in ORDEN if c in df.columns]
    rango = {c: {v: i for i, v in enumerate(ORDEN[c])} for c in claves}
    return df.sort_values(claves, key=lambda s: s.map(rango[s.name]),
                          kind="stable").reset_index(drop=True)


def resumir(tabla):
    """Media, desvío, n y error estándar del AUC y del recall_q de validación por modelo, esquema,
    variante y métrica. Agrega, con esquema DIFERENCIA, la diferencia barajado − hacia_adelante del
    AUC medio por modelo y variante, cuando están los dos esquemas."""
    validacion = tabla[(tabla["conjunto"] == "validacion")
                       & tabla["metrica"].isin(METRICAS_RESUMEN)]
    r = resumen(validacion, por=["modelo", "esquema", "variante", "metrica"])
    auc = r[r["metrica"] == "auc"].pivot(index=["modelo", "variante"], columns="esquema",
                                         values="media")
    if set(ESQUEMAS) <= set(auc.columns):
        diferencia = (auc["barajado"] - auc["hacia_adelante"]).dropna().rename("media")
        r = pd.concat([r, diferencia.reset_index().assign(esquema=DIFERENCIA, metrica="auc")],
                      ignore_index=True)
    r["n"] = r["n"].astype("Int64")
    return _ordenar_filas(r)[CAMPOS_RESUMEN]


def concatenar(salida):
    """Los robustez_<modelo>.csv de `salida`, en el orden de MODELOS."""
    rutas = [Path(salida) / archivo_modelo(m) for m in MODELOS
             if (Path(salida) / archivo_modelo(m)).exists()]
    if not rutas:
        raise SystemExit(f"No hay archivos robustez_<modelo>.csv en {salida}.")
    return pd.concat([leer_largo(ruta) for ruta in rutas], ignore_index=True)


def _celda(sub, esquema, variante):
    if (esquema, variante) not in sub.index:
        return "—"
    fila = sub.loc[(esquema, variante)]
    if esquema == DIFERENCIA:
        return f"{fila['media']:+.3f}"
    return f"{fila['media']:.3f} ± {fila['desvio']:.3f}"


def _imprimir_por_fold(tabla, por_fold=None):
    adelante = tabla[(tabla["esquema"] == "hacia_adelante") & (tabla["conjunto"] == "validacion")
                     & (tabla["metrica"] == "auc")]
    if adelante.empty:
        return
    numeros = sorted(adelante["fold"].unique())
    print("  hacia_adelante, fold por fold: valida con un bloque y entrena con los anteriores")
    print(f"  {'fold':30s}" + "".join(f"{f:>8d}" for f in numeros))
    if por_fold is not None:
        pf = por_fold[por_fold["esquema"] == "hacia_adelante"].set_index("fold")
        print(f"  {'filas de entrenamiento':30s}"
              + "".join(f"{pf.loc[f, 'n_train']:>8d}" for f in numeros))
        print(f"  {'% de «yes» en validación':30s}"
              + "".join(f"{pf.loc[f, 'pct_yes_validacion']:>8.1f}" for f in numeros))
    for variante in VARIANTES:
        valores = adelante[adelante["variante"] == variante].set_index("fold")["valor"]
        if not valores.empty:
            print(f"  {'AUC ' + variante:30s}" + "".join(f"{valores[f]:>8.3f}" for f in numeros))


def imprimir(r, tabla=None, por_fold=None):
    """Por modelo, la tabla esquema × variante del AUC y la del recall_q de validación (media ±
    desvío, sobre los folds), la diferencia barajado − hacia_adelante del AUC y, si se pasa la
    tabla larga, el AUC de cada fold hacia adelante."""
    for modelo in r["modelo"].unique():
        print(f"\n{modelo}")
        for metrica in METRICAS_RESUMEN:
            sub = r[(r["modelo"] == modelo) & (r["metrica"] == metrica)]
            if sub.empty:
                continue
            sub = sub.set_index(["esquema", "variante"])
            variantes = [v for v in VARIANTES if v in sub.index.get_level_values("variante")]
            esquemas = [e for e in ESQUEMAS + (DIFERENCIA,)
                        if e in sub.index.get_level_values("esquema")]
            print(f"  {ROTULOS[metrica]} (validación; media ± desvío sobre los folds)")
            print(f"  {'':30s}" + "".join(f"{v:>21s}" for v in variantes))
            for esquema in esquemas:
                print(f"  {esquema:30s}"
                      + "".join(f"{_celda(sub, esquema, v):>21s}" for v in variantes))
        if tabla is not None:
            _imprimir_por_fold(tabla[tabla["modelo"] == modelo], por_fold)


def _mostrar(ruta):
    ruta = Path(ruta).resolve()
    return str(ruta.relative_to(RAIZ)) if ruta.is_relative_to(RAIZ) else str(ruta)


def _dentro(ruta, carpeta):
    ruta, carpeta = Path(ruta).resolve(), Path(carpeta).resolve()
    return ruta == carpeta or carpeta in ruta.parents


def _lista(validos):
    def convertir(texto):
        elegidos = [x.strip() for x in texto.split(",") if x.strip()]
        if not elegidos or any(x not in validos for x in elegidos):
            raise argparse.ArgumentTypeError(
                f"«{texto}»; los valores válidos, separados por comas, son {', '.join(validos)}")
        return [x for x in validos if x in elegidos]

    return convertir


def argumentos(argv=None):
    """Los argumentos ya resueltos: `salida`, `esquemas`, `variantes` y `n_jobs` nunca quedan en
    None."""
    p = argparse.ArgumentParser(
        prog="python -m src.robustez",
        description="Robustez temporal dentro de train (D-25): la validación hacia adelante "
                    "contra la barajada, con y sin las variables que codifican la época.")
    p.add_argument("--modelo", choices=MODELOS_FINALES, help="el clasificador que se mide")
    p.add_argument("--esquemas", type=_lista(ESQUEMAS),
                   help=f"con --modelo, separados por comas; por defecto, {','.join(ESQUEMAS)}. "
                        "Si no son todos, reemplaza sólo esos pares en el archivo del modelo y "
                        "conserva los demás")
    p.add_argument("--variantes", type=_lista(VARIANTES),
                   help=f"con --modelo, separadas por comas; por defecto, {','.join(VARIANTES)}. "
                        "Si no son todas, igual que con --esquemas: reemplaza sólo esos pares")
    p.add_argument("--n-jobs", type=int,
                   help=f"con --modelo, procesos, uno por fold; por defecto {N_JOBS}, y 1 con "
                        "--rapido")
    p.add_argument("--rapido", action="store_true",
                   help=f"{N_RAPIDO} filas de train estratificadas, un proceso y RF con "
                        f"{ARBOLES_RAPIDO} árboles. Sin --salida escribe (y --resumen lee) en "
                        f"{SALIDA_RAPIDO.name}, dentro del directorio temporal del sistema; "
                        "nunca dentro de resultados/")
    p.add_argument("--salida", type=Path,
                   help="directorio de los CSV; por defecto, resultados/, y con --rapido, el "
                        "que indica --rapido")
    p.add_argument("--resumen", action="store_true",
                   help="concatena los robustez_<modelo>.csv de --salida y escribe el resumen")
    a = p.parse_args(argv)
    if a.resumen == (a.modelo is not None):
        p.error("indique --modelo o --resumen, uno de los dos")
    if a.resumen:
        sobrantes = [opcion for opcion, valor in (("--esquemas", a.esquemas),
                                                  ("--variantes", a.variantes),
                                                  ("--n-jobs", a.n_jobs)) if valor is not None]
        if sobrantes:
            p.error(f"{', '.join(sobrantes)} sólo van con --modelo: --resumen junta lo que ya "
                    "está medido")
    if a.n_jobs is not None and a.n_jobs < 1:
        p.error("--n-jobs tiene que ser al menos 1")
    if a.salida is None:
        a.salida = SALIDA_RAPIDO if a.rapido else DIR
    if a.rapido and _dentro(a.salida, DIR):
        p.error("--rapido no escribe ni resume dentro de resultados/: sus cifras son de una "
                f"submuestra de {N_RAPIDO} filas")
    a.esquemas = a.esquemas or list(ESQUEMAS)
    a.variantes = a.variantes or list(VARIANTES)
    if a.n_jobs is None:
        a.n_jobs = 1 if a.rapido else N_JOBS
    return a


def _comprobar_configuracion(ruta, modelo, hiperparametros):
    """Antes de medir: una corrida parcial no combina filas de hiperparámetros distintos."""
    otras = set(leer_largo(ruta)["configuracion"]) - {configuracion_json(hiperparametros)}
    if otras:
        raise SystemExit(f"{_mostrar(ruta)} se midió con otros hiperparámetros "
                         f"({'; '.join(sorted(otras))}): hay que medir de nuevo todos los "
                         f"esquemas y variantes de {modelo}, sin --esquemas ni --variantes.")


def main(argv=None):
    a = argumentos(argv)
    salida = a.salida
    if a.resumen:
        tabla = concatenar(salida)
        r = resumir(tabla)
        escribir_largo(salida / ARCHIVO_TEMPORAL, tabla)
        r.to_csv(salida / ARCHIVO_RESUMEN, index=False)
        ruta_folds = salida / ARCHIVO_FOLDS
        imprimir(r, tabla, pd.read_csv(ruta_folds) if ruta_folds.exists() else None)
        faltan = [m for m in MODELOS_FINALES if m not in set(tabla["modelo"])]
        if faltan:
            print(f"\nFaltan los modelos: {', '.join(faltan)}.")
        for modelo, pares in pares_sin_medir(tabla).items():
            print(f"{modelo}, sin medir: {_nombrar(pares)}.")
        print(f"\nConcatenado en {_mostrar(salida / ARCHIVO_TEMPORAL)}; resumen en "
              f"{_mostrar(salida / ARCHIVO_RESUMEN)}.")
        return

    ruta = salida / archivo_modelo(a.modelo)
    parcial = es_parcial(a.esquemas, a.variantes)
    if parcial and ruta.exists():
        _comprobar_configuracion(ruta, a.modelo, hiperparametros_de(a.modelo, a.rapido))
    train = cargar_train()
    if a.rapido:
        train = submuestra(train)
    train = ordenar(train)
    por_fold = composicion(train)
    escribir_folds(salida / ARCHIVO_FOLDS, por_fold)
    tabla = medir(a.modelo, train, a.esquemas, a.variantes, a.n_jobs, a.rapido)
    conservados = []
    if parcial and ruta.exists():
        anterior = leer_largo(ruta)
        previos, medidos = set(_pares_de(anterior)), set(product(a.esquemas, a.variantes))
        conservados = [par for par in PARES if par in previos and par not in medidos]
        tabla = combinar(anterior, tabla)
    escribir_modelo(ruta, tabla)
    imprimir(resumir(tabla), tabla, por_fold)
    if conservados:
        print(f"\nDel archivo anterior se conservan {_nombrar(conservados)}.")
    procesos = "1 proceso" if a.n_jobs == 1 else f"{a.n_jobs} procesos"
    print(f"\n{len(train):,} filas de train, {procesos}. Resultados en {_mostrar(ruta)}; folds "
          f"en {_mostrar(salida / ARCHIVO_FOLDS)}.")


if __name__ == "__main__":
    main()
