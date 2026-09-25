"""La configuración vigente del TP2: las opciones de preprocesamiento y los hiperparámetros que
todos los módulos de cómputo usan.

Lo mantiene el orquestador (plan/EXEC_STATE.md, N0-3). Arranca en la configuración de referencia
de las ablaciones y se actualiza dos veces: al cerrar la ola 1, con las decisiones D-08 a D-17, y
al cerrar la ola 4, con los hiperparámetros elegidos (D-22). Nada más se decide aquí.
"""

from src.modelos import REFERENCIA
from src.preproceso import Opciones

# Ola 1, D-08 a D-17. Hasta que se midan las ablaciones, la referencia.
OPCIONES_FINALES = Opciones()

# Ola 4, D-22. Hasta que se midan las curvas, la referencia de cada modelo.
HIPERPARAMETROS_FINALES = {nombre: dict(valores) for nombre, valores in REFERENCIA.items()}

# Los clasificadores del enunciado (punto 2.1). Naive Bayes tiene dos variantes hasta D-18.
MODELOS_FINALES = ["nb_gaussiano", "nb_categorico", "svm", "knn", "rf"]
