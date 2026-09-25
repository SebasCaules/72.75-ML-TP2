# TP2 — Clasificación supervisada

**72.75 Aprendizaje Automático (Machine Learning) — ITBA — 2026 Q2**
Defensa: 07/10/2026 según el enunciado; presentación y código se mandan 24 h antes.
Enunciado: [`TP2-Clasificación supervisada.pdf`](TP2-Clasificación%20supervisada.pdf)

Predecir si un cliente contrata un depósito a plazo fijo (`y` = yes/no) a partir de datos
demográficos, de campañas previas y del contexto económico, con Naive Bayes, SVM, KNN y Random
Forest (scikit-learn) evaluados con *k-fold cross-validation*.

**Estado:** paso 1 hecho. El test está separado y el EDA corre sobre train.

---

## Datos

`data/raw/`: `bank-additional-full.csv` (41 188 filas × 21 columnas, separador `;`) y
`bank-additional-names.txt`, el diccionario de variables. Es el dataset Bank Marketing de UCI
enriquecido con cinco indicadores macroeconómicos, el que indica el enunciado:
<https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing>.

Puntos del dataset que condicionan todo lo demás (detalle en `resultados/eda/reporte.txt`):

- **Desbalance:** 11,27 % de «yes». Predecir siempre «no» ya acierta el 88,73 %, así que
  *accuracy* sola no sirve.
- **`duration` es una fuga de información:** no se conoce antes de llamar. Queda fuera del
  modelo (D-05).
- **Orden temporal:** el CSV está ordenado por fecha y la tasa de «yes» sube de menos de 5 % a
  más de 50 %. De ahí D-03 y D-06.
- **`pdays = 999` no significa «nunca contactado»:** en 3 302 filas de train con 999,
  `previous ≥ 1`.

## Partición (paso 1)

`python -m src.datos` elimina los 12 duplicados exactos y parte 80/20, estratificado por `y`,
con semilla 42:

| | Filas | «yes» | % «yes» | Archivo |
|---|---:|---:|---:|---|
| train | 32 940 | 3 711 | 11,27 % | `data/particion/train.csv` |
| test | 8 236 | 928 | 11,27 % | `data/particion/test.csv` |

Los dos archivos llevan la columna `fila`, la posición en el CSV original, que es el único rastro
del orden temporal. **Nunca es una predictora**: está en `EXCLUIDAS`, junto con `duration`.

Reglas de uso:

- Todo lo que viene antes del modelo final lee **`cargar_train()`**.
- `cargar_test()` se usa **una sola vez**, en el punto 4 (D-04).
- Toda validación cruzada usa **`folds()`**, nunca `cv=5` a secas (D-06).

## Instalación y uso

Requiere Python 3.10+.

```bash
pip install -r requirements.txt

python -m src.datos         # paso 1: duplicados y partición train/test
python -m src.eda           # EDA sobre train -> resultados/eda/
python -m src.validacion    # por qué la CV tiene que barajar
python -m src.evidencia_particion  # las cifras de D-02 y D-03

python -m tests.test_datos        # la partición, contra el CSV original
python -m tests.test_validacion   # los folds de la CV
```

La partición está versionada, así que `src.datos` sólo hace falta si cambia el CSV o la
semilla.

## Estructura

```
data/raw/            el dataset tal como se bajó (no se edita)
data/particion/      train.csv y test.csv, generados por src/datos.py
src/datos.py         carga, duplicados, partición; cargar_train() y cargar_test()
src/eda.py           análisis exploratorio sobre train (punto 1)
src/validacion.py    folds() para la validación cruzada
src/evidencia_particion.py  por qué estratificar y por qué no partir por fecha
tests/               verificación de la partición y de los folds
resultados/eda/      reporte.txt y figuras del EDA
resumen-tp2/         resumen del enunciado y del dataset para el grupo (PDF)
DECISIONES.md        cada decisión con su justificación (D-01, D-02, …)
```

## Próximos pasos

1. EDA sobre train, con consecuencias concretas para el preprocesamiento: `unknown`, `pdays`,
   categorías raras, colinealidad de las variables macroeconómicas (punto 1).
2. Un `Pipeline` por modelo, con el preprocesamiento ajustado dentro de cada fold (punto 2).
3. Dos métricas vistas en clase, justificadas para el desbalance (punto 2.3).
4. Curvas de validación de SVM, KNN y RF (punto 3).
5. Modelo final y una única evaluación en test (punto 4); conclusiones (punto 5).
