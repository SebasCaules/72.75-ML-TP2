"""La configuración vigente del TP2: las opciones de preprocesamiento y los hiperparámetros que
todos los módulos de cómputo usan.

Lo mantiene el orquestador (plan/EXEC_STATE.md, N0-3). Arranca en la configuración de referencia
de las ablaciones y se actualiza dos veces: al cerrar la ola 1, con las decisiones D-08 a D-17, y
al cerrar la ola 4, con los hiperparámetros elegidos (D-22). Nada más se decide aquí.
"""

from src.modelos import REFERENCIA
from src.preproceso import Opciones

# Ola 1, D-08 a D-17 (DECISIONES.md): default como indicadora y categorías raras fundidas. El
# resto quedó como la referencia porque las ablaciones no lo justificaron.
OPCIONES_FINALES = Opciones(default="indicadora", raras=True)

# Ola 4, D-22 (DECISIONES.md §6), con los desvíos N0-11 (200 árboles, no 25) y N0-12 (RF sin
# pesos de clase). Naive Bayes no tiene curva: el enunciado no pide ajustarle nada.
HIPERPARAMETROS_FINALES = {
    "nb_gaussiano": {},
    "nb_categorico": {"n_cortes": 10, "alpha": 1.0},
    "svm": {"kernel": "linear", "implementacion": "liblinear", "C": 0.001, "pesos_clase": "balanced"},
    "knn": {"n_neighbors": 801, "weights": "uniform"},
    "rf": {"n_estimators": 200, "max_depth": 8},
}

# Los clasificadores del enunciado (punto 2.1). D-18 eligió el Naive Bayes categórico; el gaussiano
# queda medido en cv_referencia_* como comparación.
MODELOS_FINALES = ["nb_categorico", "svm", "knn", "rf"]
