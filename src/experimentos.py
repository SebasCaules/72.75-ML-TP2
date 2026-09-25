"""Correr con: python -m src.experimentos --etiqueta referencia [--modelo knn] [--n-jobs 5]

Pasos 3.3 y 5.1 del plan: la validación cruzada de los clasificadores con la configuración vigente
de src/configuracion.py, con las cinco métricas de src/metricas.py en train y en validación, por
fold, y los puntajes fuera de fold (OOF, N0-5). Sobre esos OOF, sin volver a ajustar nada, calcula
la sensibilidad del presupuesto de llamadas a q (D-20, paso 2.2) y la curva de ganancia acumulada
(H3, ola 6).

La etiqueta distingue la corrida con la configuración de referencia (paso 3.3) de la corrida con
los hiperparámetros elegidos (paso 5.1). El módulo no decide cuál es cuál: usa lo que
src/configuracion.py tenga en el momento de correr.

Cada modelo escribe sus propios archivos (N0-6), así que los modelos pueden correr en procesos
paralelos; los otros modos leen lo que haya en el directorio de salida (resultados/ por defecto):

  --etiqueta E [--modelo M ...]  cv_E_M.csv (formato largo) y oof_E_M.csv. Sin --modelo, los de
                                 MODELOS_FINALES, uno tras otro
  --etiqueta E --resumen         cv_E.csv, con las filas de linea_base.csv si existe, y
                                 cv_E_resumen.csv: media, desvío y error estándar por modelo,
                                 conjunto y métrica, más la brecha train − validación del AUC
  --etiqueta E --sensibilidad-q  sensibilidad_q_E.csv: recall, precisión y F1 llamando al 10, 20
                                 y 30 % de cada fold de validación
  --etiqueta E --ganancia        ganancia_E.csv: qué % de los «yes» se alcanza llamando al p % de
                                 la lista, con p de 1 a 100
  --rapido                       todo lo anterior sobre 3 000 filas de train, con un proceso y RF
                                 de 20 árboles, en un directorio temporal. Prueba el código de
                                 punta a punta; sus cifras no son resultados

La sensibilidad y la ganancia leen el OOF de cada modelo que tiene cv_E_M.csv, y antes de usarlo
comprueban que reproduzca, fold a fold, las métricas de validación de ese cv. Un oof_E_M.csv sin su
cv se ignora con un aviso, y uno que no reproduce su cv es un error: en los dos casos es un archivo
de otra corrida con el mismo nombre, como el OOF de A0 que una versión anterior de
src/ablaciones.py guardaba como oof_referencia_M.csv, medido con Opciones() y no con la
configuración vigente.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES, OPCIONES_FINALES
from src.datos import FILA, OBJETIVO, RAIZ, SEMILLA, cargar_train
from src.metricas import METRICAS, en_presupuesto, evaluar
from src.modelos import MODELOS, REFERENCIA, crear_modelo, separar_X_y
from src.resultados import (
    CAMPOS,
    DIR,
    N_JOBS,
    con_identidad,
    escribir_largo,
    escribir_oof,
    leer_largo,
    resumen,
    validacion_cruzada,
)

ETIQUETAS = ("referencia", "final")
SIN_MODELO = "sin_modelo"
LINEA_BASE = "linea_base.csv"
CLAVES = ["etiqueta", "modelo", "configuracion", "fold", "conjunto", "metrica"]

# La brecha va en el resumen como un conjunto más, calculada fold a fold, para que el archivo siga
# en formato largo y la brecha tenga su propio desvío.
BRECHA = "brecha"
METRICA_BRECHA = "auc"

# D-20: el presupuesto es el 20 %; la sensibilidad lo mueve al 10 y al 30 %.
Q_SENSIBILIDAD = (0.10, 0.20, 0.30)
METRICAS_PRESUPUESTO = ("recall_q", "precision_q", "f1_q")
CAMPOS_SENSIBILIDAD = ["modelo", "q", "fold", "metrica", "valor"]

PORCENTAJES = range(1, 101)
CAMPOS_GANANCIA = ["modelo", "p", "pct_yes_alcanzado"]
P_IMPRESOS = (10, 20, 30, 50)

# Recalculadas desde su OOF, las métricas de un fold sólo difieren de las del cv en el redondeo del
# CSV. Un OOF de otra corrida se aleja mucho más: un solo par de filas que cambie de orden mueve el
# AUC en 1/(positivos × negativos) del fold.
TOLERANCIA_OOF = 1e-9

FILAS_RAPIDO = 3000
ARBOLES_RAPIDO = 20


# --- Validación cruzada -------------------------------------------------------------------------

def hiperparametros_vigentes(nombre, rapido=False):
    """Los de src/configuracion.py sobre la referencia, combinados como los combina crear_modelo:
    así la columna configuracion dice todo lo que corrió. En modo rápido, RF con 20 árboles o
    menos."""
    if nombre not in HIPERPARAMETROS_FINALES:
        raise ValueError(f"{nombre} no tiene hiperparámetros en src/configuracion.py")
    h = {**REFERENCIA[nombre], **HIPERPARAMETROS_FINALES[nombre]}
    if rapido and "n_estimators" in h:
        h["n_estimators"] = min(h["n_estimators"], ARBOLES_RAPIDO)
    return h


def submuestra(train, n=FILAS_RAPIDO):
    """n filas de train, estratificadas por `y` con la semilla 42, en el orden de la fila original."""
    parte, _ = train_test_split(train, train_size=n, stratify=train[OBJETIVO],
                                random_state=SEMILLA)
    return parte.sort_values(FILA).reset_index(drop=True)


def validar(nombre, train, etiqueta, n_jobs=N_JOBS, rapido=False):
    """Validación cruzada de un modelo con la configuración vigente. Devuelve la tabla larga, con la
    columna etiqueta primero, y los puntajes OOF."""
    h = hiperparametros_vigentes(nombre, rapido)
    X, y = separar_X_y(train)
    tabla, oof = validacion_cruzada(crear_modelo(nombre, OPCIONES_FINALES, **h), X, y,
                                    filas=train[FILA], n_jobs=n_jobs)
    return con_identidad(tabla, nombre, h, etiqueta=etiqueta), oof


def ruta_cv(salida, etiqueta, modelo=None):
    return Path(salida) / (f"cv_{etiqueta}_{modelo}.csv" if modelo else f"cv_{etiqueta}.csv")


def ruta_oof(salida, etiqueta, modelo):
    return Path(salida) / f"oof_{etiqueta}_{modelo}.csv"


def correr(modelos, train, etiqueta, salida, n_jobs=N_JOBS, rapido=False):
    for nombre in modelos:
        inicio = time.perf_counter()
        tabla, oof = validar(nombre, train, etiqueta, n_jobs, rapido)
        escribir_largo(ruta_cv(salida, etiqueta, nombre), tabla)
        escribir_oof(ruta_oof(salida, etiqueta, nombre), oof)
        auc = tabla[tabla["metrica"] == "auc"].groupby("conjunto")["valor"].agg(["mean", "std"])
        print(f"{nombre:14s} {time.perf_counter() - inicio:7.1f} s   AUC validación "
              f"{auc.loc['validacion', 'mean']:.3f} ± {auc.loc['validacion', 'std']:.3f}, "
              f"train {auc.loc['train', 'mean']:.3f}   -> {ruta_cv(salida, etiqueta, nombre).name}, "
              f"{ruta_oof(salida, etiqueta, nombre).name}", flush=True)


# --- Resumen ------------------------------------------------------------------------------------

def modelos_medidos(salida, etiqueta):
    """Los modelos con cv_<etiqueta>_<modelo>.csv en `salida`, buscados por nombre exacto en el
    orden de MODELOS: un comodín como cv_<etiqueta>_*.csv también tomaría
    cv_<etiqueta>_resumen.csv."""
    return [m for m in MODELOS if ruta_cv(salida, etiqueta, m).exists()]


def _sin_cv(salida, etiqueta):
    return FileNotFoundError(f"no hay ningún cv_{etiqueta}_<modelo>.csv en {salida}: falta correr "
                             f"la validación cruzada de la etiqueta {etiqueta}")


def leer_cv(salida, etiqueta):
    """Las rutas de los cv_<etiqueta>_<modelo>.csv y sus filas, concatenadas."""
    rutas = [ruta_cv(salida, etiqueta, m) for m in modelos_medidos(salida, etiqueta)]
    if not rutas:
        raise _sin_cv(salida, etiqueta)
    return rutas, pd.concat([leer_largo(r) for r in rutas], ignore_index=True)


def leer_linea_base(salida, etiqueta):
    """Las filas de linea_base.csv (paso 1.2) con las columnas del contrato y la etiqueta, o None."""
    ruta = Path(salida) / LINEA_BASE
    if not ruta.exists():
        return None
    base = leer_largo(ruta)
    faltan = [c for c in CAMPOS if c not in base.columns]
    if faltan:
        raise ValueError(f"{ruta} no tiene las columnas del contrato: faltan {faltan}")
    return base[CAMPOS].assign(etiqueta=etiqueta)[["etiqueta"] + CAMPOS]


def _verificar(tabla):
    if tabla[CLAVES].isna().any().any():
        raise ValueError("hay claves vacías: groupby descartaría esas filas sin avisar")
    repetidas = int(tabla.duplicated(CLAVES).sum())
    if repetidas:
        raise ValueError(f"{repetidas} filas repiten etiqueta, modelo, configuración, fold, "
                         "conjunto y métrica")


def brechas(tabla, metrica=METRICA_BRECHA):
    """Train − validación de `metrica`, fold a fold, como filas con conjunto = "brecha". Los modelos
    sin filas de train (una línea de base medida sólo en validación) quedan sin brecha."""
    t = tabla[tabla["metrica"] == metrica]
    ancha = t.set_index(CLAVES[:4] + ["conjunto"])["valor"].unstack("conjunto")
    if not {"train", "validacion"} <= set(ancha.columns):
        return None
    diferencia = (ancha["train"] - ancha["validacion"]).dropna().rename("valor")
    return diferencia.reset_index().assign(conjunto=BRECHA, metrica=metrica)[CLAVES + ["valor"]]


def resumir_tabla(tabla):
    """resumen() de src/resultados.py por etiqueta, modelo, configuración, conjunto y métrica, con
    la brecha del AUC como un conjunto más."""
    _verificar(tabla)
    partes = [tabla[CLAVES + ["valor"]], brechas(tabla)]
    return resumen(pd.concat([p for p in partes if p is not None and len(p)], ignore_index=True),
                   por=["etiqueta", "modelo", "configuracion", "conjunto", "metrica"])


def tabla_validacion(r):
    """Una fila por modelo con la media ± desvío de cada métrica en validación y la brecha del AUC,
    ordenada por AUC de validación de mayor a menor. Un modelo con más de una configuración (dos
    líneas de base, por ejemplo) tiene una fila por configuración."""
    varias = r.groupby("modelo")["configuracion"].transform("nunique") > 1
    r = r.assign(fila=r["modelo"].where(~varias, r["modelo"] + " " + r["configuracion"]))
    val = r[r["conjunto"] == "validacion"]
    celdas = val.assign(celda=[f"{m:.3f} ± {d:.3f}" for m, d in zip(val["media"], val["desvio"])])
    t = celdas.pivot(index="fila", columns="metrica", values="celda")
    t = t[[m for m in METRICAS if m in t.columns]].copy()
    brecha = r[(r["conjunto"] == BRECHA) & (r["metrica"] == METRICA_BRECHA)]
    brecha = brecha.set_index("fila")["media"]
    t[f"brecha_{METRICA_BRECHA}"] = [f"{brecha[m]:+.3f}" if m in brecha.index else "" for m in t.index]
    auc = val[val["metrica"] == "auc"].set_index("fila")["media"].reindex(t.index)
    return t.loc[auc.sort_values(ascending=False, na_position="last").index].rename_axis("modelo")


def resumir(salida, etiqueta):
    salida = Path(salida)
    rutas, tabla = leer_cv(salida, etiqueta)
    base = leer_linea_base(salida, etiqueta)
    if base is not None:
        tabla = pd.concat([tabla, base], ignore_index=True)
    tabla = tabla[["etiqueta"] + CAMPOS]
    r = resumir_tabla(tabla)
    escribir_largo(ruta_cv(salida, etiqueta), tabla)
    ruta_resumen = salida / f"cv_{etiqueta}_resumen.csv"
    r.to_csv(ruta_resumen, index=False)

    print(f"Etiqueta {etiqueta}, en {_mostrar(salida)}: {', '.join(ruta.name for ruta in rutas)}"
          + (f" y {LINEA_BASE}" if base is not None else f"; sin {LINEA_BASE}, que no existe"))
    print(f"  -> {ruta_cv(salida, etiqueta).name} ({len(tabla)} filas) y {ruta_resumen.name}")
    print(f"Validación: media ± desvío de {tabla['fold'].nunique()} folds, de mayor a menor AUC; "
          f"brecha_{METRICA_BRECHA} = train − validación")
    print(tabla_validacion(r).reset_index().to_string(index=False))
    return tabla, r


# --- Sensibilidad a q y curva de ganancia, desde los OOF ------------------------------------------

def con_objetivo(oof, train):
    """El OOF con la `y` (0/1) de cada fila, cruzada por `fila` con train."""
    _, y = separar_X_y(train)
    objetivo = pd.DataFrame({FILA: train[FILA].to_numpy(), "y": y})
    cruzado = oof.merge(objetivo, on=FILA, how="left", validate="one_to_one")
    if cruzado["y"].isna().any():
        raise ValueError(f"{int(cruzado['y'].isna().sum())} filas del OOF no están en train")
    return cruzado.astype({"y": int})


def _mismos_folds(oofs):
    (primero, oof), *resto = oofs.items()
    clave = oof.sort_values(FILA)[[FILA, "fold"]].to_numpy()
    for modelo, otro in resto:
        if not np.array_equal(otro.sort_values(FILA)[[FILA, "fold"]].to_numpy(), clave):
            raise ValueError(f"los OOF de {primero} y {modelo} no cubren las mismas filas con los "
                             "mismos folds")


def verificar_oof(modelo, oof, cv):
    """Comprueba que el OOF de `modelo` (fold, puntaje e y) reproduzca, fold a fold, las métricas de
    validación que registró su cv. Si no las reproduce, es el de otra corrida guardado con el mismo
    nombre: el de una corrida interrumpida entre la escritura del cv y la del OOF, o el de A0 de una
    versión anterior de src/ablaciones.py."""
    val = cv[cv["conjunto"] == "validacion"]
    claves = [(int(f), m) for f, m in zip(val["fold"], val["metrica"])]
    folds_oof = sorted(int(f) for f in oof["fold"].unique())
    if len(set(claves)) != len(claves) or set(claves) != {(f, m) for f in folds_oof
                                                          for m in METRICAS}:
        raise ValueError(f"el cv de {modelo} no tiene exactamente una fila de validación por "
                         f"métrica en cada uno de los folds de su OOF ({folds_oof})")
    medido = dict(zip(claves, val["valor"]))
    for fold, parte in oof.groupby("fold"):
        propio = evaluar(parte["y"].to_numpy(), parte["puntaje"].to_numpy())
        distintas = [m for m in METRICAS
                     if not np.isclose(propio[m], medido[(int(fold), m)], rtol=0,
                                       atol=TOLERANCIA_OOF, equal_nan=True)]
        if distintas:
            raise ValueError(
                f"el OOF de {modelo} no reproduce {', '.join(distintas)} de validación del fold "
                f"{int(fold)} de su cv: es de otra corrida guardada con el mismo nombre. Hay que "
                f"volver a correr la validación cruzada de {modelo} con esta etiqueta")


def leer_oof(salida, etiqueta, train):
    """Los OOF de los modelos con cv_<etiqueta>_<modelo>.csv, con la `y` de cada fila, en el orden
    de MODELOS. Cada uno tiene que reproducir las métricas de validación de su cv, y todos tienen
    que cubrir las mismas filas con los mismos folds. Un OOF sin su cv no se lee (oof_sin_cv)."""
    modelos = modelos_medidos(salida, etiqueta)
    if not modelos:
        raise _sin_cv(salida, etiqueta)
    faltan = [r.name for m in modelos if not (r := ruta_oof(salida, etiqueta, m)).exists()]
    if faltan:
        raise FileNotFoundError(f"faltan {', '.join(faltan)} en {salida}: la validación cruzada "
                                "escribe cada cv junto con su OOF, así que hay que volver a "
                                "correrla para esos modelos")
    oofs = {}
    for m in modelos:
        # Con la lectura por defecto, pandas devuelve muchos puntajes con un ulp de diferencia;
        # round_trip devuelve exactamente los que se escribieron.
        oof = pd.read_csv(ruta_oof(salida, etiqueta, m), float_precision="round_trip")
        oofs[m] = con_objetivo(oof, train)
        verificar_oof(m, oofs[m], leer_largo(ruta_cv(salida, etiqueta, m)))
    _mismos_folds(oofs)
    return oofs


def oof_sin_cv(salida, etiqueta):
    """Los oof_<etiqueta>_<modelo>.csv de `salida` cuyo modelo no tiene cv_<etiqueta>_<modelo>.csv:
    los que leer_oof deja fuera."""
    medidos = set(modelos_medidos(salida, etiqueta))
    return [r for m in MODELOS
            if m not in medidos and (r := ruta_oof(salida, etiqueta, m)).exists()]


def con_sin_modelo(oofs):
    """Agrega la referencia sin modelo: el mismo puntaje para todas las filas. en_presupuesto
    reparte el corte en proporción, así que llamar al q % alcanza el q % de los «yes»."""
    primero = next(iter(oofs.values()))
    return {**oofs, SIN_MODELO: primero.assign(puntaje=0.0)}


def sensibilidad_q(oofs, qs=Q_SENSIBILIDAD):
    """recall_q, precision_q y f1_q por modelo, q y fold, cada fold con su propia lista. `oofs`:
    {modelo: DataFrame con fold, puntaje e y}."""
    filas = []
    for modelo, oof in oofs.items():
        for fold, parte in oof.groupby("fold"):
            for q in qs:
                r = en_presupuesto(parte["y"], parte["puntaje"], q)
                filas += [{"modelo": modelo, "q": q, "fold": int(fold), "metrica": m,
                           "valor": r[m]} for m in METRICAS_PRESUPUESTO]
    return pd.DataFrame(filas, columns=CAMPOS_SENSIBILIDAD)


def ganancia(oofs, porcentajes=PORCENTAJES):
    """Curva de ganancia acumulada: con la lista de todo el OOF junto, ordenada por puntaje de mayor
    a menor, qué % de los «yes» se alcanza llamando al p % de ella."""
    return pd.DataFrame(
        [{"modelo": modelo, "p": p,
          "pct_yes_alcanzado": 100 * en_presupuesto(oof["y"], oof["puntaje"], p / 100)["recall_q"]}
         for modelo, oof in oofs.items() for p in porcentajes],
        columns=CAMPOS_GANANCIA)


def escribir_sensibilidad(salida, etiqueta, oofs):
    """sensibilidad_q_<etiqueta>.csv desde `oofs` (los de leer_oof), más la línea sin modelo."""
    t = sensibilidad_q(con_sin_modelo(oofs))
    ruta = Path(salida) / f"sensibilidad_q_{etiqueta}.csv"
    t.to_csv(ruta, index=False)

    filas = []
    for (modelo, q), g in t.groupby(["modelo", "q"], sort=False):
        fila = {"modelo": modelo, "q": f"{q:.0%}"}
        for m in METRICAS_PRESUPUESTO:
            v = g.loc[g["metrica"] == m, "valor"]
            fila[m] = f"{v.mean():.3f} ± {v.std():.3f}"
        filas.append(fila)
    print(f"\nSensibilidad al presupuesto (D-20), etiqueta {etiqueta}: media ± desvío de "
          f"{t['fold'].nunique()} folds  -> {ruta.name}")
    print(pd.DataFrame(filas).to_string(index=False))
    return t


def escribir_ganancia(salida, etiqueta, oofs):
    """ganancia_<etiqueta>.csv desde `oofs` (los de leer_oof), más la diagonal sin modelo."""
    oofs = con_sin_modelo(oofs)
    t = ganancia(oofs)
    ruta = Path(salida) / f"ganancia_{etiqueta}.csv"
    t.to_csv(ruta, index=False)

    puntos = t[t["p"].isin(P_IMPRESOS)].pivot(index="modelo", columns="p",
                                               values="pct_yes_alcanzado").reindex(list(oofs))
    puntos.columns = [f"p = {p} %" for p in puntos.columns]
    print(f"\nCurva de ganancia (H3), etiqueta {etiqueta}: % de los «yes» alcanzado llamando al "
          f"p % de la lista  -> {ruta.name}")
    print(puntos.map("{:.1f}".format).rename_axis("modelo").reset_index().to_string(index=False))
    return t


# --- Línea de comandos ----------------------------------------------------------------------------

def _mostrar(ruta):
    ruta = Path(ruta).resolve()
    try:
        return ruta.relative_to(RAIZ)
    except ValueError:
        return ruta


def _dentro_de_resultados(ruta):
    ruta, raiz = Path(ruta).resolve(), DIR.resolve()
    return ruta == raiz or raiz in ruta.parents


def _procesos(n_jobs):
    return f"{n_jobs} proceso" + ("" if n_jobs == 1 else "s")


def _oofs(salida, etiqueta, train):
    """leer_oof, con un aviso por cada OOF del directorio que queda fuera por no tener su cv."""
    oofs = leer_oof(salida, etiqueta, train)
    ajenos = oof_sin_cv(salida, etiqueta)
    if ajenos:
        n = "" if len(ajenos) == 1 else "n"
        print(f"\nSe ignora{n} {', '.join(r.name for r in ajenos)}: no tiene{n} su "
              f"cv_{etiqueta}_<modelo>.csv, así que no {'es' if len(ajenos) == 1 else 'son'} de "
              "esta validación cruzada.")
    return oofs


def _parser():
    p = argparse.ArgumentParser(
        prog="python -m src.experimentos",
        description="Validación cruzada de los clasificadores con la configuración vigente "
                    "(src/configuracion.py), su resumen, la sensibilidad a q y la curva de ganancia.")
    p.add_argument("--etiqueta", choices=ETIQUETAS,
                   help="referencia (paso 3.3) o final (paso 5.1); obligatoria salvo con --rapido")
    p.add_argument("--modelo", nargs="+", choices=MODELOS, metavar="MODELO",
                   help=f"uno o más de: {', '.join(MODELOS)}. Por defecto, los de MODELOS_FINALES")
    p.add_argument("--n-jobs", type=int, default=N_JOBS,
                   help="procesos de la validación cruzada, uno por fold: al menos 1 (por defecto "
                        "%(default)s; con --rapido, siempre 1)")
    p.add_argument("--salida", type=Path,
                   help="directorio de los CSV (por defecto resultados/; con --rapido, uno temporal)")
    p.add_argument("--resumen", action="store_true",
                   help="concatena los cv_<etiqueta>_<modelo>.csv, agrega linea_base.csv y resume")
    p.add_argument("--sensibilidad-q", action="store_true",
                   help="recall, precisión y F1 con q = 10, 20 y 30 %% por fold, desde los OOF")
    p.add_argument("--ganancia", action="store_true",
                   help="curva de ganancia acumulada, desde los OOF")
    p.add_argument("--rapido", action="store_true",
                   help="validación, resumen, sensibilidad y ganancia sobre 3000 filas, con un "
                        "proceso y RF de 20 árboles")
    return p


def _rapido(args):
    etiqueta = args.etiqueta or ETIQUETAS[0]
    salida = args.salida or Path(tempfile.mkdtemp(prefix="experimentos_rapido_"))
    salida.mkdir(parents=True, exist_ok=True)
    modelos = args.modelo or MODELOS_FINALES
    train = cargar_train()
    muestra = submuestra(train)
    print(f"Modo rápido: {len(muestra)} filas de train ({int((muestra[OBJETIVO] == 'yes').sum())} "
          f"«yes»), {_procesos(1)}, RF con {ARBOLES_RAPIDO} árboles o menos. Etiqueta {etiqueta}, "
          f"en {_mostrar(salida)}.")
    print(f"Opciones vigentes: {OPCIONES_FINALES}")
    inicio = time.perf_counter()
    correr(modelos, muestra, etiqueta, salida, n_jobs=1, rapido=True)
    print()
    resumir(salida, etiqueta)
    oofs = _oofs(salida, etiqueta, train)
    escribir_sensibilidad(salida, etiqueta, oofs)
    escribir_ganancia(salida, etiqueta, oofs)
    print(f"\nModo rápido completo en {time.perf_counter() - inicio:.1f} s, en {_mostrar(salida)}")


def _validacion(args):
    salida = args.salida or DIR
    modelos = args.modelo or MODELOS_FINALES
    print(f"Etiqueta {args.etiqueta}: {', '.join(modelos)}, {_procesos(args.n_jobs)}, en "
          f"{_mostrar(salida)}.")
    print(f"Opciones vigentes: {OPCIONES_FINALES}")
    correr(modelos, cargar_train(), args.etiqueta, salida, n_jobs=args.n_jobs)


def _derivados(args):
    salida = args.salida or DIR
    if args.resumen:
        resumir(salida, args.etiqueta)
    if args.sensibilidad_q or args.ganancia:
        oofs = _oofs(salida, args.etiqueta, cargar_train())
        if args.sensibilidad_q:
            escribir_sensibilidad(salida, args.etiqueta, oofs)
        if args.ganancia:
            escribir_ganancia(salida, args.etiqueta, oofs)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    derivados = args.resumen or args.sensibilidad_q or args.ganancia
    if args.n_jobs < 1:
        parser.error("--n-jobs tiene que ser al menos 1")
    if args.rapido:
        if args.salida is not None and _dentro_de_resultados(args.salida):
            parser.error("--rapido no escribe dentro de resultados/: sus cifras son de una submuestra")
    elif args.etiqueta is None:
        parser.error("falta --etiqueta (referencia o final)")
    elif args.modelo and derivados:
        parser.error("--modelo no se combina con --resumen, --sensibilidad-q ni --ganancia, que "
                     "leen todos los modelos de la etiqueta")
    try:
        if args.rapido:
            _rapido(args)
        elif derivados:
            _derivados(args)
        else:
            _validacion(args)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
