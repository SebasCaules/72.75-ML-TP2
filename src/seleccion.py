"""Correr con: python -m src.seleccion

Paso 5.1 del plan, D-23: elige el modelo final entre los clasificadores ya medidos en validación
cruzada. No abre el test ni ajusta ningún modelo: sólo lee resultados.

Entradas, en --resultados (por defecto, resultados/):
- cv_<etiqueta>.csv: la validación cruzada de los modelos finales en formato largo, escrita por
  src/experimentos.py, con la línea de base `sin_modelo` entre sus filas.
- hiperparametros.json: los valores de la ola 4 por modelo (src/curvas.py --elegir, D-22), bajo
  la clave "modelos"; también se acepta {modelo: {parametro: valor}} en la raíz. Si no existe, o
  no nombra a un modelo, valen los de src/configuracion.py. Sus "avisos" pasan al JSON de salida.
- linea_base.csv: la línea de base «sin modelo» del paso 1.2. Si no existe, se toma `sin_modelo`
  de cv_<etiqueta>.csv.

La regla de D-23: gana el mayor AUC medio de validación. Si otros modelos quedan dentro de un error
estándar del mejor (el del mejor, sobre sus 5 folds), decide el mayor recall en el presupuesto, que
es comparable porque todos llaman al mismo 20 %. Las dos variantes de Naive Bayes compiten como
modelos distintos, y `sin_modelo` no compite.

Antes de elegir se verifica que cada modelo se haya medido con los hiperparámetros que se van a
declarar. Si difieren, el JSON describiría un modelo distinto del validado, y la selección se
detiene. Si está el OOF del elegido (oof_<etiqueta>_<modelo>.csv, N0-5) y tiene otro número de
filas que train, queda un aviso: la validación corrió sobre una submuestra.

Salida: resultados/modelo_elegido.json.

`--rapido` prueba el recorrido completo en segundos: corre con el corredor común una validación
cruzada pequeña (3 000 filas de train, sin la SVM) en un directorio temporal, y elige sobre ella.
Mide la configuración vigente tal cual, sin achicar RF: la vigente es la que se declara (N0-13),
así que medir otra cosa para ir más rápido haría fallar la verificación. Es el único camino que
ajusta modelos, y no escribe en resultados/.
"""

import argparse
import json
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES, OPCIONES_FINALES
from src.datos import FILA, OBJETIVO, RAIZ, SEMILLA, cargar_train
from src.estilo import ROTULO_LINEA_BASE
from src.metricas import METRICAS, PRESUPUESTO
from src.modelos import REFERENCIA
from src.resultados import (
    CAMPOS,
    DIR,
    con_identidad,
    configuracion_json,
    escribir_largo,
    leer_largo,
    resumen,
    validacion_cruzada,
)
from src.validacion import K

SIN_MODELO = "sin_modelo"
CONJUNTOS = ("validacion", "train")
ESTADISTICOS = ("media", "desvio", "error_estandar")
ETIQUETA = "final"
SALIDA = "modelo_elegido.json"
HIPERPARAMETROS = "hiperparametros.json"
LINEA_BASE = "linea_base.csv"

ETIQUETA_RAPIDA = "rapido"
N_RAPIDO = 3000
MODELOS_RAPIDOS = ("nb_gaussiano", "nb_categorico", "knn", "rf")

# Un modelo justo en el borde de 1 ES cuenta como empatado aunque la resta pierda el último bit.
HOLGURA = 1e-12


def _legible(ruta):
    ruta = Path(ruta).resolve()
    try:
        return ruta.relative_to(RAIZ).as_posix()
    except ValueError:
        return str(ruta)


def leer_cv(ruta, etiqueta=ETIQUETA):
    """Las filas de `etiqueta` de un CSV largo de validación cruzada."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"no existe {_legible(ruta)}: se escribe con el modo --resumen de "
                                "src.experimentos")
    tabla = leer_largo(ruta)
    faltan = [c for c in CAMPOS if c not in tabla.columns]
    if faltan:
        raise ValueError(f"{ruta.name}: faltan columnas del contrato: {faltan}")
    if "etiqueta" in tabla.columns:
        tabla = tabla[tabla["etiqueta"].astype(str) == str(etiqueta)]
        if tabla.empty:
            raise ValueError(f"{ruta.name} no tiene filas con etiqueta {etiqueta!r}")
    return tabla.reset_index(drop=True)


def _modelos_que_compiten(modelos):
    modelos = list(dict.fromkeys(MODELOS_FINALES if modelos is None else modelos))
    if not modelos:
        raise ValueError("no hay modelos que compitan")
    if SIN_MODELO in modelos:
        raise ValueError(f"{SIN_MODELO} es la línea de base y no compite")
    return modelos


def _fuera_de_la_comparacion(tabla, modelos):
    """Verifica que estén todos los que compiten; devuelve los demás modelos del CSV, que no
    compiten."""
    presentes = set(tabla["modelo"]) - {SIN_MODELO}
    faltan = [m for m in modelos if m not in presentes]
    if faltan:
        raise ValueError(f"faltan en la validación cruzada: {', '.join(faltan)}. Compiten "
                         f"{', '.join(modelos)}; otra lista se pasa con --modelos")
    return sorted(presentes - set(modelos))


def _verificar_folds(tabla, modelos):
    """Cada modelo: una sola configuración, y los K folds en train y validación con las cinco
    métricas. Si no, la media mezclaría configuraciones o saldría de una corrida parcial."""
    esperados = list(range(1, K + 1))
    problemas = []
    for modelo in modelos:
        propio = tabla[tabla["modelo"] == modelo]
        configuraciones = propio["configuracion"].nunique()
        if configuraciones != 1:
            problemas.append(f"{modelo}: {configuraciones} configuraciones, se espera una")
        for conjunto in CONJUNTOS:
            for metrica in METRICAS:
                elegidas = (propio["conjunto"] == conjunto) & (propio["metrica"] == metrica)
                folds = sorted(int(f) for f in propio.loc[elegidas, "fold"])
                if folds != esperados:
                    problemas.append(f"{modelo}, {conjunto}, {metrica}: folds {folds}")
        decisivas = propio.loc[(propio["conjunto"] == "validacion")
                               & propio["metrica"].isin(["auc", "recall_q"]), "valor"]
        if not np.isfinite(decisivas.to_numpy(dtype=float)).all():
            problemas.append(f"{modelo}: AUC o recall_q de validación no finitos")
    if problemas:
        raise ValueError(f"la validación cruzada está incompleta (se esperan los folds 1 a {K} "
                         f"en train y validación, con las {len(METRICAS)} métricas):\n  "
                         + "\n  ".join(problemas[:12]))


def leer_hiperparametros(ruta):
    """Los hiperparámetros de la ola 4, {modelo: {parametro: valor}}, y los avisos del archivo.
    Sin archivo, (None, [])."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None, []
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    avisos = []
    if isinstance(datos, dict) and isinstance(datos.get("modelos"), dict):
        avisos = [f"{ruta.name}: {aviso}" for aviso in datos.get("avisos") or []]
        datos = datos["modelos"]
    if not isinstance(datos, dict):
        raise ValueError(f"{ruta.name}: se espera {{modelo: {{parametro: valor}}}}")
    malos = [m for m, h in datos.items()
             if not isinstance(h, dict) or any(isinstance(v, dict) for v in h.values())]
    if malos:
        raise ValueError(f"{ruta.name}: se espera {{modelo: {{parametro: valor}}}}, en la raíz o "
                         f"bajo la clave \"modelos\"; no cumplen: {', '.join(map(str, malos))}")
    return datos, avisos


def hiperparametros_de(modelo, de_la_ola_4=None):
    """Los hiperparámetros efectivos: la referencia, los de hiperparametros.json (la propuesta
    mecánica de D-22) y, encima, la configuración vigente de src/configuracion.py, que es la que
    entrena los modelos y puede apartarse de la propuesta con un desvío registrado (N0-11, N0-12
    en DECISIONES.md). `crear_modelo` también parte de la referencia, así que son exactamente los
    que usaría."""
    return {**REFERENCIA.get(modelo, {}), **(de_la_ola_4 or {}).get(modelo, {}),
            **HIPERPARAMETROS_FINALES.get(modelo, {})}


def _normalizar(valor):
    """`configuracion_json` guarda como texto lo que no es nativo de JSON (un np.int64 queda
    "300"), así que 300 y "300", o True y "True", son el mismo hiperparámetro."""
    if isinstance(valor, str):
        texto = valor.strip()
        if texto in ("None", "null"):
            return None
        if texto in ("True", "False"):
            return texto == "True"
        try:
            return float(texto)
        except ValueError:
            return texto
    if valor is None or isinstance(valor, (bool, np.bool_)):
        return None if valor is None else bool(valor)
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor)
    if isinstance(valor, (list, tuple)):
        return [_normalizar(v) for v in valor]
    return valor


def _mismo_valor(a, b):
    a, b = _normalizar(a), _normalizar(b)
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_mismo_valor(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _verificar_configuracion(tabla, modelos, de_la_ola_4):
    """La configuración con que se midió cada modelo tiene que ser la que se va a declarar."""
    distintas = []
    for modelo in modelos:
        texto = tabla.loc[tabla["modelo"] == modelo, "configuracion"].iloc[0]
        try:
            medida = json.loads(texto)
        except (TypeError, ValueError):
            medida = None
        if not isinstance(medida, dict):
            raise ValueError(f"{modelo}: la columna configuracion no es un JSON de "
                             f"hiperparámetros: {texto!r}")
        medida = {**REFERENCIA.get(modelo, {}), **medida}
        declarada = hiperparametros_de(modelo, de_la_ola_4)
        if not all(_mismo_valor(medida.get(c), declarada.get(c))
                   for c in set(medida) | set(declarada)):
            distintas.append(f"{modelo}: medida {configuracion_json(medida)}, declarada "
                             f"{configuracion_json(declarada)}")
    if distintas:
        raise ValueError("la validación cruzada no se midió con los hiperparámetros que se "
                         "declararían:\n  " + "\n  ".join(distintas) + "\nVuelva a correr "
                         "src.experimentos con la configuración vigente, o corrija "
                         "hiperparametros.json.")


def ranking(tabla, modelos):
    """Los modelos por AUC medio de validación, de mayor a menor, con el error estándar del AUC y
    el recall_q medio."""
    validacion = tabla[(tabla["conjunto"] == "validacion") & tabla["modelo"].isin(modelos)]
    r = resumen(validacion, por=("modelo", "metrica")).set_index(["modelo", "metrica"])
    filas = [{"modelo": m,
              "auc": float(r.loc[(m, "auc"), "media"]),
              "auc_error_estandar": float(r.loc[(m, "auc"), "error_estandar"]),
              "recall_q": float(r.loc[(m, "recall_q"), "media"])} for m in modelos]
    return sorted(filas, key=lambda f: (-f["auc"], f["modelo"]))


def regla_d23(filas):
    """D-23 sobre un ranking ya ordenado. Devuelve el modelo elegido y el grupo empatado dentro de
    1 ES del mejor, el mejor incluido, o una lista vacía si ningún otro modelo queda dentro. En el
    grupo decide el mayor recall_q; si también empata, el mayor AUC."""
    mejor = filas[0]
    umbral = mejor["auc"] - mejor["auc_error_estandar"]
    grupo = [f for f in filas if f["auc"] >= umbral - HOLGURA]
    elegido = max(grupo, key=lambda f: (f["recall_q"], f["auc"]))
    return elegido["modelo"], [f["modelo"] for f in grupo] if len(grupo) > 1 else []


def resumen_del_modelo(tabla, modelo):
    """Media, desvío y error estándar de las cinco métricas, en validación y en train."""
    r = resumen(tabla[tabla["modelo"] == modelo], por=("conjunto", "metrica"))
    r = r.set_index(["conjunto", "metrica"])
    return {conjunto: {metrica: {e: float(r.loc[(conjunto, metrica), e]) for e in ESTADISTICOS}
                       for metrica in METRICAS}
            for conjunto in CONJUNTOS}


def _medias_de_validacion(filas):
    medias = filas[filas["conjunto"] == "validacion"].groupby("metrica")["valor"].mean()
    if not {"auc", "recall_q"} <= set(medias.index):
        return None
    return {"auc": float(medias["auc"]), "recall_q": float(medias["recall_q"])}


def linea_base(directorio, tabla, ruta_cv):
    """AUC y recall_q medios de validación de la línea de base, con su fuente y los avisos."""
    ruta = Path(directorio) / LINEA_BASE
    desde_cv = _medias_de_validacion(tabla[tabla["modelo"] == SIN_MODELO])
    avisos = []
    if ruta.exists():
        propia = leer_largo(ruta)
        faltan = [c for c in ("conjunto", "metrica", "valor") if c not in propia.columns]
        if faltan:
            raise ValueError(f"{ruta.name}: faltan columnas del contrato: {faltan}")
        if "modelo" in propia.columns and propia["modelo"].nunique() > 1:
            propia = propia[propia["modelo"] == SIN_MODELO]
        valores = _medias_de_validacion(propia)
        if valores is None:
            raise ValueError(f"{ruta.name} no tiene AUC y recall_q de validación")
        fuente = _legible(ruta)
        if desde_cv is not None and not all(math.isclose(valores[m], desde_cv[m], abs_tol=1e-9)
                                            for m in valores):
            avisos.append(f"la línea de base de {ruta.name} (AUC {valores['auc']:.4f}, recall_q "
                          f"{valores['recall_q']:.4f}) no coincide con {SIN_MODELO} de "
                          f"{Path(ruta_cv).name} (AUC {desde_cv['auc']:.4f}, recall_q "
                          f"{desde_cv['recall_q']:.4f}); vale la de {ruta.name}")
    elif desde_cv is not None:
        valores, fuente = desde_cv, f"{_legible(ruta_cv)} ({SIN_MODELO})"
    else:
        raise FileNotFoundError(f"no hay línea de base: falta {_legible(ruta)} y "
                                f"{Path(ruta_cv).name} no tiene filas de {SIN_MODELO}")
    return valores, fuente, avisos


def _a_json(valor):
    """Sólo tipos nativos de JSON: los escalares de numpy pasan a los de Python y NaN, a null."""
    if isinstance(valor, dict):
        return {str(k): _a_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_a_json(v) for v in valor]
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        return float(valor) if math.isfinite(valor) else None
    if valor is None or isinstance(valor, str):
        return valor
    raise TypeError(f"valor sin representación en JSON: {valor!r}")


def seleccionar(directorio=DIR, etiqueta=ETIQUETA, modelos=None, n_train=None):
    """Aplica D-23 sobre los resultados de `directorio` y devuelve el contenido de
    modelo_elegido.json, ya en tipos de JSON. `n_train` es el número de filas de train que usó la
    validación cruzada; por defecto, todas las de train."""
    directorio = Path(directorio)
    modelos = _modelos_que_compiten(modelos)
    ruta_cv = directorio / f"cv_{etiqueta}.csv"
    tabla = leer_cv(ruta_cv, etiqueta)
    fuera = _fuera_de_la_comparacion(tabla, modelos)
    _verificar_folds(tabla, modelos)
    ruta_hiper = directorio / HIPERPARAMETROS
    de_la_ola_4, avisos_hiper = leer_hiperparametros(ruta_hiper)
    _verificar_configuracion(tabla, modelos, de_la_ola_4)

    filas = ranking(tabla, modelos)
    elegido, empate = regla_d23(filas)
    base, fuente_base, avisos_base = linea_base(directorio, tabla, ruta_cv)
    avisos = ([f"no compiten, por no estar en la lista de modelos: {', '.join(fuera)}"]
              if fuera else []) + avisos_hiper + avisos_base
    filas_train = len(cargar_train()) if n_train is None else n_train
    ruta_oof = directorio / f"oof_{etiqueta}_{elegido}.csv"
    if ruta_oof.exists():
        filas_cv = len(pd.read_csv(ruta_oof, usecols=[FILA]))
        if filas_cv != filas_train:
            avisos.append(f"la validación cruzada de {elegido} corrió sobre {filas_cv} filas "
                          f"({ruta_oof.name}), no sobre las {filas_train} de train")
    medidas = resumen_del_modelo(tabla, elegido)
    eleccion = {
        "modelo": elegido,
        "hiperparametros": hiperparametros_de(elegido, de_la_ola_4),
        "opciones": asdict(OPCIONES_FINALES),
        "metricas": medidas,
        "brecha_auc": medidas["train"]["auc"]["media"] - medidas["validacion"]["auc"]["media"],
        "empate_dentro_1es": empate,
        "desempate_por_recall": bool(empate),
        "linea_base": base,
        "ranking": filas,
        "n_train": filas_train,
        "semilla": SEMILLA,
        "k_folds": K,
        "presupuesto": PRESUPUESTO,
        "etiqueta": etiqueta,
        "fuentes": {
            "cv": _legible(ruta_cv),
            "hiperparametros": ("src/configuracion.py" if de_la_ola_4 is None
                                else f"{_legible(ruta_hiper)} sobre src/configuracion.py"),
            "linea_base": fuente_base,
        },
        "avisos": avisos,
    }
    return _a_json(eleccion)


def escribir(ruta, eleccion):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(eleccion, ensure_ascii=False, indent=2, allow_nan=False)
    ruta.write_text(texto + "\n", encoding="utf-8")


def imprimir(eleccion):
    base, filas = eleccion["linea_base"], eleccion["ranking"]
    print(f"D-23 sobre {eleccion['fuentes']['cv']}: AUC medio de {eleccion['k_folds']} folds de "
          f"validación; recall_q llamando al {eleccion['presupuesto']:.0%} de cada lista.")
    print(f"\n  {'#':>2}  {'modelo':14s} {'AUC':>7s} {'ES':>7s} {'recall_q':>9s}   "
          f"{'ΔAUC':>7s} {'Δrecall_q':>9s}   (Δ contra «{ROTULO_LINEA_BASE}»)")
    for i, f in enumerate(filas, start=1):
        print(f"  {i:2d}  {f['modelo']:14s} {f['auc']:7.4f} {f['auc_error_estandar']:7.4f} "
              f"{f['recall_q']:9.4f}   {f['auc'] - base['auc']:+7.4f} "
              f"{f['recall_q'] - base['recall_q']:+9.4f}")
    print(f"  {'':2s}  {ROTULO_LINEA_BASE:14s} {base['auc']:7.4f} {'':7s} {base['recall_q']:9.4f}")

    mejor = filas[0]
    print(f"\nMayor AUC: {mejor['modelo']}. Umbral de empate, 1 ES por debajo: "
          f"{mejor['auc'] - mejor['auc_error_estandar']:.4f}.")
    if eleccion["desempate_por_recall"]:
        print(f"Empatan dentro de 1 ES: {', '.join(eleccion['empate_dentro_1es'])}. Decide el "
              "mayor recall_q.")
    else:
        print("Ningún otro modelo queda dentro de 1 ES: decide el AUC.")
    va, tr = eleccion["metricas"]["validacion"], eleccion["metricas"]["train"]
    print(f"Modelo elegido: {eleccion['modelo']} {configuracion_json(eleccion['hiperparametros'])}")
    print(f"  AUC: validación {va['auc']['media']:.4f} ± {va['auc']['desvio']:.4f}, train "
          f"{tr['auc']['media']:.4f}, brecha {eleccion['brecha_auc']:+.4f}")
    print(f"  recall_q: validación {va['recall_q']['media']:.4f} ± {va['recall_q']['desvio']:.4f}")
    for aviso in eleccion["avisos"]:
        print(f"Aviso: {aviso}")


def validacion_rapida(directorio):
    """Escribe en `directorio` las tres entradas con el formato del contrato, medidas sobre una
    submuestra estratificada de train: cv_rapido.csv con `sin_modelo`, linea_base.csv y un
    hiperparametros.json con la forma del de src/curvas.py. Cada modelo se mide con su
    configuración vigente, y el JSON propone esos mismos valores: como la vigente prevalece
    (N0-13), cualquier otro valor del JSON se declararía sin haberse medido. Devuelve el número de
    filas."""
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split

    from src.modelos import crear_modelo, separar_X_y

    directorio = Path(directorio)
    train = cargar_train()
    muestra, _ = train_test_split(train, train_size=N_RAPIDO, stratify=train[OBJETIVO],
                                  random_state=SEMILLA)
    X, y = separar_X_y(muestra)
    tablas = []
    vigentes = {modelo: dict(HIPERPARAMETROS_FINALES[modelo]) for modelo in MODELOS_RAPIDOS}
    for modelo, hiper in vigentes.items():
        tabla, _ = validacion_cruzada(crear_modelo(modelo, OPCIONES_FINALES, **hiper), X, y,
                                      filas=muestra[FILA], n_jobs=1)
        tablas.append(con_identidad(tabla, modelo, hiper, etiqueta=ETIQUETA_RAPIDA))
    base, _ = validacion_cruzada(DummyClassifier(strategy="prior"), X, y, filas=muestra[FILA],
                                 n_jobs=1)
    tablas.append(con_identidad(base, SIN_MODELO, {}, etiqueta=ETIQUETA_RAPIDA))
    escribir_largo(directorio / f"cv_{ETIQUETA_RAPIDA}.csv", pd.concat(tablas, ignore_index=True))
    escribir_largo(directorio / LINEA_BASE, con_identidad(base, SIN_MODELO, {}))
    (directorio / HIPERPARAMETROS).write_text(json.dumps({"modelos": vigentes, "avisos": []}),
                                              encoding="utf-8")
    return len(muestra)


def _dentro_de_resultados(ruta):
    ruta, raiz = Path(ruta).resolve(), DIR.resolve()
    return ruta == raiz or raiz in ruta.parents


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.seleccion",
        description="D-23: elige el modelo final con la validación cruzada ya medida, sin abrir "
                    "el test, y escribe modelo_elegido.json.")
    parser.add_argument("--etiqueta", default=ETIQUETA,
                        help="lee cv_<etiqueta>.csv (por defecto: %(default)s)")
    parser.add_argument("--resultados", type=Path, default=DIR,
                        help="directorio de las entradas (por defecto: resultados/)")
    parser.add_argument("--salida", type=Path,
                        help=f"dónde escribir el JSON (por defecto: <resultados>/{SALIDA})")
    parser.add_argument("--modelos", nargs="+",
                        help="los modelos que compiten (por defecto: MODELOS_FINALES de "
                             "src/configuracion.py)")
    parser.add_argument("--rapido", action="store_true",
                        help=f"prueba de punta a punta sobre una validación cruzada de {N_RAPIDO} "
                             "filas en un directorio temporal; ignora --etiqueta, --resultados y "
                             "--modelos. Con --salida escribe el JSON ahí, nunca dentro de "
                             "resultados/")
    args = parser.parse_args(argv)
    if args.rapido and args.salida is not None and _dentro_de_resultados(args.salida):
        parser.error("--rapido no escribe dentro de resultados/: sus cifras son de una submuestra")

    try:
        if args.rapido:
            with tempfile.TemporaryDirectory() as temporal:
                directorio = Path(temporal)
                n = validacion_rapida(directorio)
                eleccion = seleccionar(directorio, ETIQUETA_RAPIDA, MODELOS_RAPIDOS, n_train=n)
                salida = args.salida or directorio / SALIDA
                escribir(salida, eleccion)
                imprimir(eleccion)
                print(f"\nEscrito: {_legible(salida)}")
                print(json.dumps(eleccion, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            eleccion = seleccionar(args.resultados, args.etiqueta, args.modelos)
            salida = args.salida or args.resultados / SALIDA
            escribir(salida, eleccion)
            imprimir(eleccion)
            print(f"\nEscrito: {_legible(salida)}")
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
