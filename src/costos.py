"""Correr con: python -m src.costos   (tarda hasta una hora; conviene en segundo plano)

Paso 0.6 del plan: el presupuesto de cómputo. Mide cuánto tarda un ajuste de cada clasificador en
los extremos de sus grillas, sobre el primer fold de `folds()` (26 352 filas de entrenamiento y
6 588 de validación). Mide también la predicción sobre el fold de entrenamiento, porque las curvas
de validación necesitan la métrica en train (Clase 7, slide 50).

Cada medición corre en un proceso aparte con un límite de tiempo: si no termina, queda registrada
como «excedido» en lugar de colgar todo, y si falla, como «error». Escribe resultados/costos.csv y,
a partir de él, resultados/costos_grillas.csv: el costo estimado de cada grilla de la ola 4, con
el peor caso medido de su modelo por cada ajuste. `--estimar` sólo rehace esa segunda tabla.
"""

import csv
import json
import subprocess
import sys
import time

from src.datos import RAIZ
from src.modelos import GRILLAS

RUTA = RAIZ / "resultados" / "costos.csv"
RUTA_GRILLAS = RAIZ / "resultados" / "costos_grillas.csv"
LIMITE_SEGUNDOS = 600
FOLDS = 5
TOPE_MINUTOS = 45  # paso 0.6 del plan: ninguna grilla pasa de 45 minutos

MEDICIONES = [
    ("nb_gaussiano", {}),
    ("nb_categorico", {}),
    ("knn", {"n_neighbors": 1}),
    ("knn", {"n_neighbors": 201}),
    ("rf", {"n_estimators": 10}),
    ("rf", {"n_estimators": 800}),
    ("rf", {"n_estimators": 300, "max_depth": 2}),
    ("svm", {"kernel": "rbf", "C": 0.001}),
    ("svm", {"kernel": "rbf", "C": 1}),
    ("svm", {"kernel": "rbf", "C": 100}),
    ("svm", {"kernel": "poly", "degree": 3, "C": 1}),
    ("svm", {"kernel": "poly", "degree": 3, "C": 100}),
    ("svm", {"kernel": "linear", "C": 0.01}),
    ("svm", {"kernel": "linear", "C": 1}),
    ("svm", {"kernel": "linear", "C": 10}),
    ("svm", {"kernel": "linear", "implementacion": "liblinear", "C": 0.01}),
    ("svm", {"kernel": "linear", "implementacion": "liblinear", "C": 1}),
    ("svm", {"kernel": "linear", "implementacion": "liblinear", "C": 100}),
]


def medir_una(indice):
    """Corre en el proceso hijo: un ajuste y dos predicciones, y devuelve los tiempos en JSON."""
    from src.datos import cargar_train
    from src.metricas import puntajes
    from src.modelos import crear_modelo, separar_X_y
    from src.validacion import folds

    nombre, hiper = MEDICIONES[indice]
    X, y = separar_X_y(cargar_train())
    tr, va = next(folds().split(X, y))
    modelo = crear_modelo(nombre, **hiper)
    t0 = time.perf_counter()
    modelo.fit(X.iloc[tr], y[tr])
    t1 = time.perf_counter()
    puntajes(modelo, X.iloc[va])
    t2 = time.perf_counter()
    puntajes(modelo, X.iloc[tr])
    t3 = time.perf_counter()
    print(json.dumps({"ajuste": t1 - t0, "prediccion_validacion": t2 - t1,
                      "prediccion_train": t3 - t2}))


def medir_todas():
    filas = []
    for indice, (nombre, hiper) in enumerate(MEDICIONES):
        fila = {"modelo": nombre, "parametros": json.dumps(hiper, sort_keys=True)}
        try:
            salida = subprocess.run([sys.executable, "-m", "src.costos", "--una", str(indice)],
                                    capture_output=True, text=True, timeout=LIMITE_SEGUNDOS,
                                    cwd=RAIZ, check=True)
            tiempos = json.loads(salida.stdout.strip().splitlines()[-1])
            fila.update({k: round(v, 2) for k, v in tiempos.items()}, estado="ok")
        except subprocess.TimeoutExpired:
            fila.update(ajuste="", prediccion_validacion="", prediccion_train="",
                        estado=f"excedido (más de {LIMITE_SEGUNDOS} s)")
        except subprocess.CalledProcessError as e:
            fila.update(ajuste="", prediccion_validacion="", prediccion_train="",
                        estado="error: " + e.stderr.strip().splitlines()[-1][:120])
        filas.append(fila)
        print(f"{nombre:14s} {fila['parametros']:55s} {fila['estado']:>10s} "
              f"{fila.get('ajuste', '')}", flush=True)
        _escribir(filas)
    return filas


def _escribir(filas):
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    campos = ["modelo", "parametros", "ajuste", "prediccion_validacion", "prediccion_train", "estado"]
    with open(RUTA, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)


def estimar_grillas():
    """El costo de cada grilla de la ola 4: valores × folds × el peor ajuste medido de su modelo.

    Las curvas necesitan un ajuste y dos predicciones (validación y train) por valor y por fold.
    Para la SVM, la grilla de C usa las filas del kernel RBF; la comparación de kernels se estima
    aparte, con el peor de cada kernel, y el kernel lineal de libsvm queda fuera si excedió.
    """
    with open(RUTA, newline="") as f:
        medidas = [m for m in csv.DictReader(f) if m["estado"] == "ok"]

    def peor(filtro):
        costos = [float(m["ajuste"]) + float(m["prediccion_validacion"]) + float(m["prediccion_train"])
                  for m in medidas if filtro(m)]
        return max(costos) if costos else None

    filas = []
    for (modelo, parametro), valores in GRILLAS.items():
        if modelo == "svm":
            segundos = peor(lambda m: m["modelo"] == "svm" and '"rbf"' in m["parametros"])
        else:
            segundos = peor(lambda m: m["modelo"] == modelo)
        filas.append({"grilla": f"{modelo}: {parametro}", "ajustes": len(valores) * FOLDS,
                      "peor_ajuste_s": round(segundos, 1), "minutos": round(segundos * len(valores) * FOLDS / 60, 1)})
    for kernel, filtro in [("linear (liblinear)", lambda m: "liblinear" in m["parametros"]),
                           ("poly", lambda m: '"poly"' in m["parametros"]),
                           ("rbf", lambda m: '"rbf"' in m["parametros"])]:
        segundos = peor(lambda m, f=filtro: m["modelo"] == "svm" and f(m))
        filas.append({"grilla": f"svm: kernel {kernel} en el mejor C", "ajustes": FOLDS,
                      "peor_ajuste_s": round(segundos, 1), "minutos": round(segundos * FOLDS / 60, 1)})
    excedidos = [m["parametros"] for m in csv.DictReader(open(RUTA, newline=""))
                 if m["estado"] != "ok"]
    for fila in filas:
        fila["entra"] = "sí" if fila["minutos"] <= TOPE_MINUTOS else f"no (más de {TOPE_MINUTOS} min)"
    with open(RUTA_GRILLAS, "w", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["grilla", "ajustes", "peor_ajuste_s", "minutos", "entra"])
        escritor.writeheader()
        escritor.writerows(filas)
    for fila in filas:
        print(f"{fila['grilla']:42s} {fila['ajustes']:4d} ajustes  peor {fila['peor_ajuste_s']:6.1f} s  "
              f"{fila['minutos']:6.1f} min  {fila['entra']}")
    if excedidos:
        print("Fuera de las grillas por exceder el límite: " + "; ".join(excedidos))
    return filas


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--una":
        medir_una(int(sys.argv[2]))
        return
    if "--estimar" not in sys.argv:
        medir_todas()
        print()
    estimar_grillas()
    print(f"\nMediciones en {RUTA.relative_to(RAIZ)}; estimación por grilla en "
          f"{RUTA_GRILLAS.relative_to(RAIZ)}.")


if __name__ == "__main__":
    main()
