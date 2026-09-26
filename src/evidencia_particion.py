"""Correr con: python -m src.evidencia_particion   (opcional: --salida RUTA)

La evidencia numérica de D-02 (estratificar) y D-03 (partición aleatoria, no temporal). Trabaja
sobre el dataset sin duplicados, test incluido: mide propiedades de la partición misma, no elige
nada del modelo, así que no contradice D-04.

Imprime las cifras y las escribe en resultados/evidencia_particion.json, con el contrato que lee
src/numeros.py para los macros de D-02 y D-03 (numeros.py no puede calcularlas: no lee el CSV
completo):

    {"n_semillas": 400,
     "sin_estratificar": {"desvio_pp": ..., "minimo_pct": ..., "maximo_pct": ...},
     "temporal": {"pct_yes_train": ..., "pct_yes_test": ...}}

Los porcentajes van en puntos (11.27, no 0.1127). El desvío es el poblacional (ddof = 0), el mismo
que se imprime.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.datos import OBJETIVO, PROP_TEST, RAIZ, cargar, limpiar

N_SEMILLAS = 400
SALIDA = RAIZ / "resultados" / "evidencia_particion.json"


def sin_estratificar(df, n_semillas=N_SEMILLAS):
    """% de «yes» del test en `n_semillas` particiones aleatorias sin estratificar."""
    yes = (df[OBJETIVO] == "yes").to_numpy()
    indices = np.arange(len(df))
    tasas = []
    for semilla in range(n_semillas):
        _, idx_test = train_test_split(indices, test_size=PROP_TEST, random_state=semilla)
        tasas.append(100 * yes[idx_test].mean())
    return np.array(tasas)


def temporal(df):
    """% de «yes» de train y test si el test es el último PROP_TEST del archivo."""
    n_test = round(len(df) * PROP_TEST)
    yes = df[OBJETIVO] == "yes"
    return 100 * yes.iloc[:-n_test].mean(), 100 * yes.iloc[-n_test:].mean()


def evidencia(df, n_semillas=N_SEMILLAS):
    """Las cifras de D-02 y D-03 con el contrato de resultados/evidencia_particion.json."""
    tasas = sin_estratificar(df, n_semillas)
    tasa_train, tasa_test = temporal(df)
    return {
        "n_semillas": int(n_semillas),
        "sin_estratificar": {"desvio_pp": float(tasas.std()), "minimo_pct": float(tasas.min()),
                             "maximo_pct": float(tasas.max())},
        "temporal": {"pct_yes_train": float(tasa_train), "pct_yes_test": float(tasa_test)},
    }


def escribir(ruta, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mostrar(ruta):
    ruta = Path(ruta).resolve()
    try:
        return ruta.relative_to(RAIZ).as_posix()
    except ValueError:
        return str(ruta)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.evidencia_particion",
        description="Las cifras de D-02 y D-03 sobre el dataset sin duplicados; las imprime y las "
                    "escribe en el JSON que lee src/numeros.py.")
    parser.add_argument("--salida", type=Path, default=SALIDA,
                        help=f"dónde escribir el JSON (por defecto, {_mostrar(SALIDA)})")
    args = parser.parse_args(argv)

    df = limpiar(cargar())
    datos = evidencia(df)
    s, t = datos["sin_estratificar"], datos["temporal"]
    n = datos["n_semillas"]
    print(f"D-02. % de 'yes' del test sin estratificar, en {n} semillas (0 a {n - 1}):")
    print(f"  desvío {s['desvio_pp']:.2f} pp, mínimo {s['minimo_pct']:.2f} %, "
          f"máximo {s['maximo_pct']:.2f} %")
    print(f"  (global: {100 * (df[OBJETIVO] == 'yes').mean():.2f} %)")
    print()
    print("D-03. Partición temporal (el último 20 % del archivo como test):")
    print(f"  train {t['pct_yes_train']:.2f} % de 'yes', test {t['pct_yes_test']:.2f} %")
    escribir(args.salida, datos)
    print(f"\nEscrito {_mostrar(args.salida)}: lo lee src/numeros.py (D-02, D-03).")


if __name__ == "__main__":
    main()
