"""El contrato de resultados del TP2 y el corredor común de validación cruzada.

Lo mantiene el orquestador (plan/EXEC_STATE.md, N0-4 y N0-5). Todo módulo que mida algo en
validación cruzada pasa por `validacion_cruzada()` y escribe con `escribir_largo()`, así los CSV
tienen siempre las mismas columnas y una sola función dibuja cualquier curva.

Formato largo: una fila por `modelo, configuracion, fold, conjunto, metrica, valor`.
- `configuracion`: los hiperparámetros como JSON con claves ordenadas (`configuracion_json`).
- `fold`: 1 a 5, los de `folds()`.
- `conjunto`: "train" o "validacion".
- `metrica`: las cinco de `src.metricas.METRICAS`.
Los módulos pueden agregar columnas propias (por ejemplo `variante` en las ablaciones) antes de
las seis del contrato.

Puntajes fuera de fold (OOF): una fila por `fila, fold, puntaje`, con la fila del CSV original.
"""

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from src.datos import FILA, RAIZ
from src.metricas import METRICAS, evaluar, puntajes
from src.validacion import K, folds

DIR = RAIZ / "resultados"
CAMPOS = ["modelo", "configuracion", "fold", "conjunto", "metrica", "valor"]
CAMPOS_OOF = ["fila", "fold", "puntaje"]
N_JOBS = 5  # un proceso por fold


def configuracion_json(hiperparametros):
    return json.dumps(hiperparametros, sort_keys=True, default=str)


def _un_fold(modelo, X, y, tr, va, numero):
    ajustado = clone(modelo).fit(X.iloc[tr], y[tr])
    s_va = puntajes(ajustado, X.iloc[va])
    s_tr = puntajes(ajustado, X.iloc[tr])
    return numero, evaluar(y[tr], s_tr), evaluar(y[va], s_va), s_va


def validacion_cruzada(modelo, X, y, filas=None, n_jobs=N_JOBS):
    """Ajusta `modelo` (un Pipeline sin ajustar) en los K folds de `folds()`.

    Devuelve dos DataFrames: `tabla`, con las columnas fold, conjunto, metrica y valor (sin modelo
    ni configuracion, que agrega quien llama), y `oof`, con fila, fold y puntaje de validación.
    `filas` son las filas del CSV original de cada X (la columna `fila` de train); si no se pasan,
    se usa el índice de X.
    """
    y = np.asarray(y).astype(int)
    filas = np.asarray(X.index if filas is None else filas)
    particiones = list(folds().split(X, y))
    salidas = Parallel(n_jobs=n_jobs)(
        delayed(_un_fold)(modelo, X, y, tr, va, numero)
        for numero, (tr, va) in enumerate(particiones, start=1))

    registros, oof = [], []
    for numero, m_tr, m_va, s_va in sorted(salidas, key=lambda s: s[0]):
        for conjunto, metricas in (("train", m_tr), ("validacion", m_va)):
            registros += [{"fold": numero, "conjunto": conjunto, "metrica": nombre,
                           "valor": metricas[nombre]} for nombre in METRICAS]
        _, va = particiones[numero - 1]
        oof.append(pd.DataFrame({"fila": filas[va], "fold": numero, "puntaje": s_va}))
    return pd.DataFrame(registros), pd.concat(oof, ignore_index=True)


def con_identidad(tabla, modelo, hiperparametros, **extra):
    """Agrega las columnas modelo y configuracion (y las extra) y ordena según el contrato."""
    tabla = tabla.assign(modelo=modelo, configuracion=configuracion_json(hiperparametros), **extra)
    return tabla[list(extra) + CAMPOS]


def escribir_largo(ruta, tabla):
    faltan = [c for c in CAMPOS if c not in tabla.columns]
    if faltan:
        raise ValueError(f"faltan columnas del contrato: {faltan}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)


def leer_largo(ruta):
    return pd.read_csv(ruta)


def escribir_oof(ruta, oof):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    oof[CAMPOS_OOF].to_csv(ruta, index=False)


def resumen(tabla, por=("modelo", "configuracion", "conjunto", "metrica")):
    """Media, desvío (ddof=1) y error estándar sobre los folds, por cada grupo."""
    por = list(por)
    g = tabla.groupby(por)["valor"]
    r = g.agg(media="mean", desvio="std", n="count").reset_index()
    r["error_estandar"] = r["desvio"] / np.sqrt(r["n"])
    return r


def diferencias_pareadas(tabla, columna, referencia, metrica="auc", conjunto="validacion"):
    """Para cada valor de `columna` (por ejemplo la variante de una ablación), la diferencia con
    `referencia` fold a fold, con su media, desvío y error estándar. Mismo modelo, misma métrica."""
    t = tabla[(tabla["metrica"] == metrica) & (tabla["conjunto"] == conjunto)]
    base = t[t[columna] == referencia].set_index(["modelo", "fold"])["valor"]
    filas = []
    for (modelo, valor), grupo in t.groupby(["modelo", columna]):
        if valor == referencia:
            continue
        propio = grupo.set_index("fold")["valor"]
        delta = np.array([propio[f] - base[(modelo, f)] for f in propio.index])
        filas.append({"modelo": modelo, columna: valor, "n_folds": len(delta),
                      "delta_media": delta.mean(), "delta_desvio": delta.std(ddof=1),
                      "delta_error_estandar": delta.std(ddof=1) / np.sqrt(len(delta)),
                      "delta_min": delta.min(), "delta_max": delta.max()})
    return pd.DataFrame(filas)
