"""Correr con: python -m src.metricas

Las métricas del TP2 (DECISIONES.md, D-19 y D-20). Se calculan siempre las cinco candidatas, para
que una respuesta de la cátedra cambie qué columna se lee y no qué hay que volver a correr:

- auc: área bajo la curva ROC (Clase 4, slides 45-51). Compara y elige modelos.
- ap: precisión promedio, el área bajo la curva de precisión y recall.
- recall_q, precision_q, f1_q: llamando al q % de la lista con mayor puntaje, el presupuesto de
  D-20. Todos los modelos hacen el mismo número de llamadas, así que sus recalls se comparan.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PRESUPUESTO = 0.20
METRICAS = ("auc", "ap", "recall_q", "precision_q", "f1_q")


def puntajes(modelo, X):
    """Puntaje de «yes»: `decision_function` si el modelo la tiene (la SVM), si no la probabilidad.

    La SVM no da probabilidades sin el escalado de Platt (Alpaydin §13.9), y para ordenar clientes
    no hace falta: el AUC y el presupuesto sólo usan el orden.
    """
    if hasattr(modelo, "decision_function"):
        return np.asarray(modelo.decision_function(X), dtype=float)
    return np.asarray(modelo.predict_proba(X)[:, 1], dtype=float)


def llamadas(n, q=PRESUPUESTO):
    return max(1, int(round(q * n)))


def en_presupuesto(y, s, q=PRESUPUESTO):
    """Recall, precisión y F1 si se llama a los k = q·n clientes de mayor puntaje.

    Los empates en el corte se reparten en proporción, que es el valor esperado de desempatar al
    azar. Así, un modelo que les da el mismo puntaje a todos alcanza exactamente el q % de los «yes»,
    sin depender del orden de las filas, que en este dataset es el orden temporal.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    k = llamadas(len(y), q)
    corte = np.sort(s)[::-1][k - 1]
    arriba = s > corte
    en_corte = s == corte
    fraccion = (k - arriba.sum()) / en_corte.sum()
    aciertos = y[arriba].sum() + fraccion * y[en_corte].sum()

    positivos = y.sum()
    recall = aciertos / positivos if positivos else float("nan")
    precision = aciertos / k
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"llamadas": k, "recall_q": float(recall), "precision_q": float(precision),
            "f1_q": float(f1)}


def evaluar(y, s, q=PRESUPUESTO):
    presupuesto = en_presupuesto(y, s, q)
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "recall_q": presupuesto["recall_q"],
        "precision_q": presupuesto["precision_q"],
        "f1_q": presupuesto["f1_q"],
    }


def scoring(q=PRESUPUESTO):
    """Para `cross_validate(scoring=scoring())`: calcula los puntajes una sola vez por fold y por
    conjunto, y devuelve las cinco métricas juntas."""

    def _scoring(modelo, X, y):
        return evaluar(y, puntajes(modelo, X), q)

    return _scoring


def main():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05]
    print(f"Ejemplo: 10 clientes, 2 «yes», presupuesto {PRESUPUESTO:.0%} ({llamadas(len(y))} llamadas)")
    for nombre, valor in evaluar(y, s).items():
        print(f"  {nombre:12s} {valor:.3f}")


if __name__ == "__main__":
    main()
