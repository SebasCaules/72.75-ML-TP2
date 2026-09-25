"""Correr con: python -m src.validacion

El particionador de la validación cruzada (punto 2.2). Toda llamada de scikit-learn que reciba
`cv=` usa `folds()`, nunca `cv=5` a secas. Ver DECISIONES.md, D-06.

`cv=5` en `cross_val_score`, `GridSearchCV` o `validation_curve` equivale a
`StratifiedKFold(5)` **sin barajar**. Como train.csv está ordenado por la fila original, y el
CSV original por fecha, cada fold quedaría concentrado en una época: la proporción de «yes»
sería la misma en todos, pero no la campaña ni el contexto económico. `main()` lo muestra.
"""

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.datos import FILA, OBJETIVO, SEMILLA, cargar_train

K = 5


def folds(k=K):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=SEMILLA)


def composicion(cv, df):
    y = (df[OBJETIVO] == "yes").astype(int)
    filas = []
    for i, (_, idx_val) in enumerate(cv.split(df, y), start=1):
        val = df.iloc[idx_val]
        filas.append({
            "fold": i,
            "filas": len(val),
            "pct_yes": round(100 * y.iloc[idx_val].mean(), 1),
            "fila_mediana": int(val[FILA].median()),
            "euribor3m_medio": round(val["euribor3m"].mean(), 2),
        })
    return pd.DataFrame(filas).set_index("fold")


def main():
    train = cargar_train()
    casos = [
        ("KFold(5) sin barajar", KFold(K)),
        ("StratifiedKFold(5) sin barajar  <- lo que hace cv=5", StratifiedKFold(K)),
        ("folds(): StratifiedKFold(5) barajado", folds()),
    ]
    for nombre, cv in casos:
        print(nombre)
        print(composicion(cv, train).to_string())
        print()


if __name__ == "__main__":
    main()
