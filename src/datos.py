"""Correr con: python -m src.datos

Paso 1 del TP: separar el test antes de cualquier otra cosa. Carga el CSV original, elimina los
duplicados exactos y parte en train y test, estratificado por `y`. Escribe
data/particion/train.csv y data/particion/test.csv. Ver DECISIONES.md, D-01 a D-04.

El enunciado lo pide en el punto 1 («realizar la división entre entrenamiento y test antes de
aplicar cualquier transformación que pueda producir data leakage») y el punto 2.2 lo supone: la
validación cruzada usa únicamente el conjunto de entrenamiento.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "raw" / "bank-additional-full.csv"
DIR_PARTICION = RAIZ / "data" / "particion"
RUTA_TRAIN = DIR_PARTICION / "train.csv"
RUTA_TEST = DIR_PARTICION / "test.csv"

OBJETIVO = "y"
# Posición de la fila en el CSV original (0 = primera fila de datos). El archivo está ordenado
# por fecha (bank-additional-names.txt, §4), así que esta columna es el único rastro del orden
# temporal: sirve para analizarlo, nunca como predictora.
FILA = "fila"

PROP_TEST = 0.2
SEMILLA = 42

# Columnas que no entran al modelo. `duration` es la duración de la última llamada: no se conoce
# antes de llamar y determina casi por completo el resultado (bank-additional-names.txt,
# atributo 11). Se conserva en los archivos de la partición para poder reportar un modelo con
# `duration` como techo de referencia. Ver D-05.
EXCLUIDAS = [FILA, "duration"]


def cargar():
    df = pd.read_csv(RUTA_CSV, sep=";")
    return df.rename_axis(FILA).reset_index()


def duplicadas(df):
    return df.drop(columns=FILA).duplicated(keep="first")


def limpiar(df):
    return df[~duplicadas(df)].reset_index(drop=True)


def partir(df):
    train, test = train_test_split(
        df,
        test_size=PROP_TEST,
        stratify=df[OBJETIVO],
        shuffle=True,
        random_state=SEMILLA,
    )
    # Se reordena por la fila original para que el análisis temporal siga siendo posible.
    return (
        train.sort_values(FILA).reset_index(drop=True),
        test.sort_values(FILA).reset_index(drop=True),
    )


def cargar_train():
    """Puerta de entrada de todo lo que se hace antes del modelo final: EDA, validación cruzada y
    curvas de validación."""
    return pd.read_csv(RUTA_TRAIN)


def cargar_test():
    """Sólo para el punto 4, una vez elegido el modelo final. Ver D-04."""
    return pd.read_csv(RUTA_TEST)


def tasa_yes(df):
    return float((df[OBJETIVO] == "yes").mean())


def main():
    crudo = cargar()
    repetidas = crudo[duplicadas(crudo)]
    df = limpiar(crudo)
    train, test = partir(df)

    DIR_PARTICION.mkdir(parents=True, exist_ok=True)
    train.to_csv(RUTA_TRAIN, index=False)
    test.to_csv(RUTA_TEST, index=False)

    sin_objetivo = crudo.drop(columns=[FILA, OBJETIVO]).duplicated().sum()
    print(f"CSV original: {len(crudo):,} filas x {crudo.shape[1] - 1} columnas")
    print(f"Duplicados exactos eliminados: {len(repetidas)} "
          f"(filas {', '.join(map(str, repetidas[FILA]))})")
    conflicto = sin_objetivo - len(repetidas)
    print(f"  Duplicados si se ignora y: {sin_objetivo} -> "
          + ("ningún par tiene etiquetas distintas" if conflicto == 0
             else f"{conflicto} pares con la misma fila y distinta etiqueta"))
    print(f"Sin duplicados: {len(df):,} filas, {tasa_yes(df):.2%} de 'yes'")
    print()
    for nombre, parte, ruta in [("train", train, RUTA_TRAIN), ("test", test, RUTA_TEST)]:
        n_yes = int((parte[OBJETIVO] == "yes").sum())
        print(f"{nombre:>5}: {len(parte):6,} filas ({len(parte) / len(df):.1%}), "
              f"{n_yes:5,} 'yes' ({tasa_yes(parte):.2%})  -> {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
