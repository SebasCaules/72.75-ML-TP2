"""Los clasificadores del TP2 como Pipelines (punto 2.1), con sus configuraciones.

`crear_modelo(nombre, opciones, **hiperparametros)` arma Derivadas → codificación → clasificador.
Naive Bayes tiene dos variantes, porque la decisión entre ellas es D-18. REFERENCIA es la
configuración de las ablaciones de la ola 1 y GRILLAS, la de las curvas de la ola 4.

Ningún clasificador usa varios núcleos por su cuenta: el paralelismo va por folds, en quien corre
la validación cruzada, para no pedir más núcleos de los que hay.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from src.datos import FILA, OBJETIVO, SEMILLA
from src.preproceso import Derivadas, Opciones, categorias_minimas, codificador

MODELOS = ("nb_gaussiano", "nb_categorico", "svm", "knn", "rf")

# D-12: la SVM y KNN miden distancias, así que necesitan escalar (Clase 8, slide 54). A RF no le
# cambia nada (Clase 7, slide 26), y al NB gaussiano tampoco: estima media y varianza por columna.
ESCALAR_POR_DEFECTO = {"nb_gaussiano": False, "nb_categorico": False, "svm": True, "knn": True,
                       "rf": False}

REFERENCIA = {
    "nb_gaussiano": {},
    "nb_categorico": {"n_cortes": 10, "alpha": 1.0},  # Laplace, α = 1 (Clase 6, slide 46)
    "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    "knn": {"n_neighbors": 15, "weights": "uniform"},
    "rf": {"n_estimators": 300},
}

GRILLAS = {
    ("rf", "max_depth"): [2, 4, 6, 8, 10, 12, 15, 20, 25, None],
    ("rf", "n_estimators"): [10, 25, 50, 100, 200, 400, 800],
    # La grilla llegaba hasta 201 y el máximo de validación quedó en el borde (ola 4), así que se
    # extendió hasta 801 para ver la meseta.
    ("knn", "n_neighbors"): [1, 3, 5, 9, 15, 25, 41, 61, 101, 151, 201, 301, 401, 601, 801],
    ("svm", "C"): [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100],
}


def separar_X_y(df):
    """X con las 20 predictoras, incluida `duration` (que `Derivadas` saca salvo en A1), e y en 0/1."""
    return df.drop(columns=[FILA, OBJETIVO]), (df[OBJETIVO] == "yes").astype(int).to_numpy()


def crear_modelo(nombre, opciones=Opciones(), **hiperparametros):
    if nombre not in MODELOS:
        raise ValueError(f"modelo desconocido: {nombre}; los válidos son {MODELOS}")
    h = {**REFERENCIA[nombre], **hiperparametros}
    escalar = ESCALAR_POR_DEFECTO[nombre] if opciones.escalar is None else opciones.escalar

    if nombre == "nb_categorico":
        n_cortes = h.pop("n_cortes")
        codificacion = codificador(opciones, "ordinal", escalar=False, n_cortes=n_cortes)
        clasificador = CategoricalNB(min_categories=categorias_minimas(opciones, n_cortes), **h)
    else:
        codificacion = codificador(opciones, "onehot", escalar)
        if nombre == "nb_gaussiano":
            clasificador = GaussianNB(**h)
        elif nombre == "svm":
            clasificador = _svm(h)
        elif nombre == "knn":
            clasificador = KNeighborsClassifier(**h)
        else:
            clasificador = RandomForestClassifier(random_state=SEMILLA,
                                                  class_weight=h.pop("pesos_clase", None), **h)

    return Pipeline([("derivadas", Derivadas(opciones)), ("columnas", codificacion),
                     ("modelo", clasificador)])


def _svm(h):
    """SVC de libsvm por defecto. Con `implementacion="liblinear"` usa LinearSVC con pérdida hinge,
    que es la misma SVM lineal pero escala mucho mejor con el número de filas (paso 0.6)."""
    pesos = h.pop("pesos_clase", None)
    if h.pop("implementacion", "libsvm") == "liblinear":
        if h.get("kernel", "linear") != "linear":
            raise ValueError("liblinear sólo tiene kernel lineal")
        return LinearSVC(C=h["C"], loss="hinge", class_weight=pesos,
                         max_iter=h.get("max_iter", 50_000), random_state=SEMILLA)
    return SVC(class_weight=pesos, cache_size=1000, **h)
