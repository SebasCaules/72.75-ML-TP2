"""Preprocesamiento del TP2: variables derivadas y codificación de columnas por modelo.

Todo se usa dentro de un `Pipeline`, así que lo que se aprende de los datos (medias y desvíos del
escalado, cortes de la discretización, la moda de la imputación) se ajusta en cada fold sólo con su
parte de entrenamiento (TP2, p. 1; D-06). `Derivadas` no aprende nada: aplica reglas fijas,
decididas con el EDA de train.

Las opciones reproducen las decisiones D-08 a D-16 y las ablaciones A1 a A12 de `plan/PLAN.md`.
Los valores por defecto son la configuración de referencia de las ablaciones, no las decisiones
finales: esas quedan en DECISIONES.md cuando la ola 1 las mida.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
MACRO_REDUCIDO = ["euribor3m", "nr.employed"]
NUMERICAS_CLIENTE = ["age", "campaign", "pdays", "previous"]
CATEGORICAS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month",
               "day_of_week", "poutcome"]
CON_UNKNOWN = ["job", "marital", "education", "default", "housing", "loan"]

# Los niveles de cada categórica según bank-additional-names.txt (atributos 2 a 10 y 15). Se fijan
# de antemano en lugar de aprenderse en cada fold: un nivel que falte en un fold (en train hay sólo
# 2 filas con default = yes) no rompe la codificación. Enero y febrero no figuran porque ninguna
# campaña cayó en esos meses.
VOCABULARIO = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                  "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["failure", "nonexistent", "success"],
}

# D-11: los niveles con menos del 1 % de las filas de train que se funden con otro. `dec` también
# es raro, pero no se toca: su tasa de «yes» es de las más altas.
FUSION_RARAS = {"education": {"illiterate": "basic.4y"}, "marital": {"unknown": "married"}}

# D-16: los mismos tramos de edad del EDA (src/eda_html.py, CORTES_EDAD). Cada uno incluye su
# borde inferior.
TRAMOS_EDAD = [0, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
ROTULOS_EDAD = ["<25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60+"]


@dataclass(frozen=True)
class Opciones:
    duration: bool = False       # A1: sólo como techo de referencia (D-05)
    pdays: str = "crudo"         # A2: "crudo" | "sin" (saca pdays; ver D-10), D-10
    default: str = "categorica"  # A3: "categorica" | "indicadora" (unknown contra el resto), D-09
    raras: bool = False          # A4: funde los niveles de FUSION_RARAS, D-11
    campaign_log: bool = False   # A5: log1p(campaign), D-13
    macro: str = "completo"      # A6 y A7: "completo" | "reducido" | "sin", D-14
    month: bool = True           # A8: False saca `month` (junto con macro="sin"), D-15
    day_of_week: bool = True     # A9: False saca `day_of_week`, D-15
    edad_tramos: bool = False    # A10: la edad en tramos en lugar de años, D-16
    unknown: str = "categoria"   # A11: "categoria" | "moda", D-08
    escalar: bool | None = None  # A12: None deja la decisión al modelo, D-12


def columnas(op):
    """Las columnas que salen de `Derivadas`, agrupadas por cómo se codifican."""
    macro = {"completo": MACRO, "reducido": MACRO_REDUCIDO, "sin": []}[op.macro]
    numericas = [c for c in NUMERICAS_CLIENTE if not (
        (c == "pdays" and op.pdays == "sin") or (c == "age" and op.edad_tramos))]
    numericas += macro + (["duration"] if op.duration else [])

    categoricas = [c for c in CATEGORICAS if not (
        (c == "default" and op.default == "indicadora")
        or (c == "month" and not op.month)
        or (c == "day_of_week" and not op.day_of_week))]
    if op.edad_tramos:
        categoricas.append("edad_tramo")

    # D-10: «contactado antes» no se agrega como columna: es previous > 0, que ya está en el
    # one-hot de poutcome (nonexistent equivale a previous == 0). Agregarla la duplicaría y pesaría
    # doble en GaussianNB y en las distancias de KNN. Sacar pdays deja esa información en poutcome.
    binarias = ["default_unknown"] if op.default == "indicadora" else []
    return numericas, categoricas, binarias


def vocabulario(columna, op):
    if columna == "edad_tramo":
        return list(ROTULOS_EDAD)
    niveles = list(VOCABULARIO[columna])
    if op.raras:
        niveles = [n for n in niveles if n not in FUSION_RARAS.get(columna, {})]
    if op.unknown == "moda" and columna in CON_UNKNOWN:
        niveles = [n for n in niveles if n != "unknown"]
    return niveles


class Derivadas(BaseEstimator, TransformerMixin):
    """Reglas fijas sobre las columnas crudas. No aprende nada de los datos: `fit` no hace nada."""

    def __init__(self, opciones=Opciones()):
        self.opciones = opciones

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        op = self.opciones
        X = X.copy()
        if op.default == "indicadora":
            X["default_unknown"] = (X["default"] == "unknown").astype(int)
        if op.raras:
            for columna, fusion in FUSION_RARAS.items():
                X[columna] = X[columna].replace(fusion)
        if op.campaign_log:
            X["campaign"] = np.log1p(X["campaign"])
        if op.edad_tramos:
            X["edad_tramo"] = pd.cut(X["age"], TRAMOS_EDAD, right=False,
                                     labels=ROTULOS_EDAD).astype(str)
        if op.unknown == "moda":
            for columna in CON_UNKNOWN:
                X[columna] = X[columna].replace("unknown", np.nan)
        numericas, categoricas, binarias = columnas(op)
        return X[numericas + categoricas + binarias]


class Discretizador(BaseEstimator, TransformerMixin):
    """Pasa cada numérica a hasta `n_cortes` intervalos por cuantiles del fold de entrenamiento,
    para CategoricalNB.

    Los cortes repetidos se descartan (variables con pocos valores, como `previous`), así que una
    columna puede terminar con menos intervalos. Un valor fuera del rango de entrenamiento cae en el
    intervalo del extremo.
    """

    def __init__(self, n_cortes=10):
        self.n_cortes = n_cortes

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        cuantiles = np.linspace(0, 1, self.n_cortes + 1)[1:-1]
        self.cortes_ = [np.unique(np.quantile(X[:, j], cuantiles)) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.searchsorted(c, X[:, j], side="right")
                                for j, c in enumerate(self.cortes_)])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def codificador(op, codificacion, escalar, n_cortes=10):
    """El ColumnTransformer que sigue a `Derivadas`. Sale en el orden numéricas, categóricas,
    binarias.

    - "onehot": numéricas escaladas o no, categóricas en one-hot. Para NB gaussiano, SVM, KNN y RF.
    - "ordinal": numéricas discretizadas y categóricas como códigos enteros. Para CategoricalNB.
    """
    numericas, categoricas, binarias = columnas(op)
    categorias = [vocabulario(c, op) for c in categoricas]
    if codificacion == "onehot":
        num = StandardScaler() if escalar else "passthrough"
        cat = OneHotEncoder(categories=categorias, handle_unknown="error", sparse_output=False)
    elif codificacion == "ordinal":
        num = Discretizador(n_cortes)
        cat = OrdinalEncoder(categories=categorias, handle_unknown="error")
    else:
        raise ValueError(f"codificación desconocida: {codificacion}")
    if op.unknown == "moda":
        cat = Pipeline([("imputar", SimpleImputer(strategy="most_frequent")), ("codificar", cat)])
    return ColumnTransformer(
        [("num", num, numericas), ("cat", cat, categoricas), ("bin", "passthrough", binarias)],
        verbose_feature_names_out=False,
    )


def categorias_minimas(op, n_cortes):
    """Cuántos valores puede tomar cada columna de la codificación "ordinal", en su orden. Es el
    `min_categories` de CategoricalNB: reserva lugar para los niveles que falten en un fold."""
    numericas, categoricas, binarias = columnas(op)
    return np.array([n_cortes] * len(numericas)
                    + [len(vocabulario(c, op)) for c in categoricas]
                    + [2] * len(binarias))
