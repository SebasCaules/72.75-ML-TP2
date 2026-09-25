"""Correr con: python -m src.evidencia_particion

La evidencia numérica de D-02 (estratificar) y D-03 (partición aleatoria, no temporal). Trabaja
sobre el dataset sin duplicados, test incluido: mide propiedades de la partición misma, no elige
nada del modelo, así que no contradice D-04.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.datos import OBJETIVO, PROP_TEST, cargar, limpiar

N_SEMILLAS = 400


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


def main():
    df = limpiar(cargar())

    tasas = sin_estratificar(df)
    print(f"D-02. % de 'yes' del test sin estratificar, en {N_SEMILLAS} semillas (0 a {N_SEMILLAS - 1}):")
    print(f"  desvío {tasas.std():.2f} pp, mínimo {tasas.min():.2f} %, máximo {tasas.max():.2f} %")
    print(f"  (global: {100 * (df[OBJETIVO] == 'yes').mean():.2f} %)")
    print()

    tasa_train, tasa_test = temporal(df)
    print("D-03. Partición temporal (el último 20 % del archivo como test):")
    print(f"  train {tasa_train:.2f} % de 'yes', test {tasa_test:.2f} %")


if __name__ == "__main__":
    main()
